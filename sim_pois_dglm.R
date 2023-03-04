e_sim = new.env(parent=.GlobalEnv)
assign("cdir",getwd(),envir=e_sim)
assign("repo","/Users/meinitang/Dropbox/Repository/poisson-dlm",envir=e_sim)

assign("UPBND",700,envir=e_sim)
assign("EPS",.Machine$double.eps,envir=e_sim)

Rcpp::sourceCpp(file.path(e_sim$repo,"model_utils.cpp"),env=e_sim)
Rcpp::sourceCpp(file.path(e_sim$repo,"lbe_poisson.cpp"),env=e_sim)
source(file.path(e_sim$repo,"model_info.R"),local=e_sim)



# TODO: compare it to the older version
sim_pois_dglm2 = function(
    n = 200, # number of observations for training
    m = 20, # number of observations for testing
    obs_dist = "nbinom", # {"nbinom","poisson"}
    link_func = "identity", # {"identity","exponential"}
    trans_func = "koyama", # {"koyama","solow","koyck"}
    gain_func = "logistic", # {"exponential","softplus","logistic"}
    err_dist = "gaussian", # {"gaussian","laplace","cauchy","left_skewed_normal"}
    mu0 = 0.,
    theta0 = NULL,
    psi0 = 0.,
    W = 0.01, # Evolution variance
    L = 2, # length of nonzero transmission delay (koyama) or number of failures (solow)
    rho = 0.7, # parameter for negative binomial transmission delay
    coef = c(0.2,0,5), # coefficients for the hyperbolic tangent gain function
    alpha = 1, # power on the transmission delay
    solow_alpha = 0, # 0 - raise AR coef to alpha; 1 - raise MA & AR coef to alpha
    delta_nb = 10., # rho_nb = 34.08792
    rng.seed = NULL,
    delta_grid = seq(from=0.8,to=0.99,by=0.01)) { # searching range for LBE discount factor
  
  
  # ------ Initialization ------ #
  ntotal = m + n
  model_code = e_sim$get_model_code(obs_dist=obs_dist,
                                    link_func=link_func,
                                    trans_func=trans_func,
                                    gain_func=gain_func,
                                    err_dist=err_dist)

  c1 = 2*rho
  c2 = rho^2
  c3 = (1-rho)^(L*alpha)
  alpha2 = ifelse(solow_alpha==1,alpha,1)
  
  if (is.null(mu0)) {mu0 = 0.}
  if (is.null(theta0)) {
    set.seed(rng.seed)
    theta0 = runif(1,0,2)
  }
  if (is.null(psi0)) {
    set.seed(rng.seed)
    psi0 = rnorm(1,0,sd=1)
  }
  # ------ Initialization ------ #
  
  
  # --------- Gain --------- #
  wt = rep(0,ntotal)
  if (!is.null(rng.seed)) { set.seed(rng.seed)}
  
  err_dist = tolower(err_dist)
  if (err_dist == "gaussian") {
    wt = rnorm(ntotal,0,sqrt(W)) # wt[1] = 0
  } else if (err_dist == "laplace") {
    wt = rmutil::rlaplace(ntotal,0,sqrt(W))
  } else if (err_dist == "cauchy") {
    wt = rcauchy(ntotal-1,0,sqrt(W))
  } else if (err_dist == "left_skewed_normal") {
    wt = sn::rsn(ntotal,xi=0,omega=sqrt(W),alpha=-0.1)
  }
  stopifnot(all(is.finite(wt)))
  # wt - Checked. Correct.
  
  
  psi = cumsum(wt) + psi0
  hpsi = psi
  gain_func = tolower(gain_func)
  if (gain_func=="ramp") {
    # Ramp
    hpsi[hpsi<e_sim$EPS] = e_sim$EPS # Reproduction number after maximum thresholding
  } else if (gain_func=="exponential") {
    # Exponential
    hpsi[hpsi>e_sim$UPBND] = e_sim$UPBND
    hpsi = exp(hpsi)
  } else if (gain_func=="identity") {
  } else if (gain_func=="softplus") {
    hpsi[hpsi>e_sim$UPBND] = e_sim$UPBND
    hpsi = log(1. + exp(hpsi))
  } else if (gain_func=="tanh") {
    hpsi = 0.5*coef[3] * (tanh(coef[1]*hpsi + coef[2]) + 1.)
  } else if (gain_func=="logistic") {
    hpsi = -coef[1]*hpsi + coef[1]*coef[2]
    hpsi[hpsi>e_sim$UPBND] = e_sim$UPBND
    hpsi = coef[3] / (exp(hpsi) + 1)
  } else {
    stop("Not supported gain function.")
  }
  stopifnot(all(is.finite(hpsi)))
  # hpsi - Checked. Correct.
  # --------- Gain --------- #
  
  
  # ------ Transfer ------ #
  theta = rep(0,ntotal+1)
  theta[1] = theta0

  lambda = rep(0,ntotal)
  y = rep(0,ntotal)
  
  Fphi = e_sim$get_Fphi(L)^alpha
  trans_func = tolower(trans_func)
  link_func = tolower(link_func)
  obs_dist = tolower(obs_dist)
  
  # ------ Transfer ------ #
  if (trans_func == "solow") {
    # theta[2] = (2*rho)^alpha2*theta0
    theta[2] = -binom(L,1)*(-rho)^1*theta[1]
  } else {
    theta[2] = e_sim$EPS
  }
  # ------ Transfer ------ #
  
  # ------ Link ------ #
  lambda[1] = mu0 + theta[2]
  if (link_func == "exponential") {
    lambda[1] = exp(min(c(lambda[1],e_sim$UPBND)))
  }
  # ------ Link ------ #
  
  # ------ Obs ------ #
  if (!is.null(rng.seed)) { set.seed(rng.seed)}
  if (obs_dist == "nbinom") {
    y[1] = rnbinom(1,delta_nb,delta_nb/(lambda[1]+delta_nb))
  } else {
    y[1] = rpois(1,lambda[1])
  }
  # ------ Obs ------ #
  
  for (t in 2:ntotal) {
    # ------ Transfer ------ #
    if (trans_func == "koyck") {
      theta[t+1] = rho*theta[t] + y[t-1]*hpsi[t-1]
    } else if (trans_func == "koyama") {
      ytilde = y[(t-1):max(c(1,t-L))]
      nlag = length(ytilde)
      theta[t+1] = sum(Fphi[1:nlag]*hpsi[t:(t-nlag+1)]*ytilde) # <state>
    } else if (trans_func == "solow") {
      theta[t+1] = c3*hpsi[t-1]*y[t-1]
      for (k in 1:min(c(t,L))) {
        theta[t+1] = theta[t+1] - binom(L,k)*(-rho)^k*theta[t+1-k]
      }
    }
    # ------ Transfer ------ #
    
    # ------ Link ------ #
    lambda[t] = mu0 + theta[t+1]
    if (link_func == "exponential") {
      lambda[t] = exp(min(c(lambda[t], e_sim$UPBND)))
    }
    # ------ Link ------ #
    
    # ------ Obs ------ #
    if (obs_dist == "nbinom") {
      y[t] = rnbinom(1, delta_nb, delta_nb/(lambda[t]+delta_nb))
    } else {
      y[t] = rpois(1,lambda[t])
    }
    # ------ Obs ------ #
  }
  
  stopifnot(all(is.finite(theta)))
  stopifnot(all(is.finite(lambda)))
  stopifnot(all(is.finite(y)))
  

  
  delta = NULL

  tryCatch({
    delta = e_sim$get_optimal_delta(y[1:n],model_code,delta_grid,
                                    rho=rho,L=L,mu0=mu0,
                                    delta_nb=delta_nb,
                                    ctanh=coef,alpha=alpha)$delta_optim[1]
  },error=function(e){delta=NA})
  
  
  
  params = list(model_code=model_code,
                mu0=mu0,theta0=theta0,psi0=psi0,
                W=W,rho=rho,L=L,
                delta_nb=delta_nb,
                delta_lbe=delta,
                ctanh=coef,alpha=alpha)
  pred = list(y=y[(n+1):ntotal],
              psi=psi[(n+1):ntotal],
              theta=theta[(n+2):(ntotal+1)],
              lambda=lambda[(n+1):ntotal])
  
  return(list(y=y[1:n], # Observation
              lambda=lambda[1:n], # Intensity
              theta=theta[2:(n+1)], # Transfer Function Block
              psi=psi[1:n], # Gain Factor
              hpsi=hpsi[1:n], # Reproduction Number - Function of the Gain Factor
              wt=wt[1:n], # Evolution Variance
              params=params,
              pred=pred)) # Model settings and initial values
}



sim_pois_dglm = function(
    n = 200, # number of observations for training
    m = 20, # number of observations for testing
    ModelCode = 0, # 
    obs_type = 1, # 0 - negative-binomial; 1 - poisson
    err_type = 0, # 0 - gaussian N(0,W); 1 - laplace; 2 - cauchy; 3 - left skewed normal
    mu0 = 0., # the baseline intensity
    alpha = 1, # power on the transmission delay
    solow_alpha = 0, # 0 - raise AR coef to alpha; 1 - raise MA & AR coef to alpha
    psi0 = 0., # initial value of the gain factor
    theta0 = 0., # initial value for the transfer function block; set it to NULL to sample from a uniform distribution(0,10). Only used in Solow
    W = 0.01, # Evolution variance
    L = 0, # length of nonzero transmission delay (Koyama - ModelCode = 0 or 1)
    rho = 0.7, # parameter for negative binomial transmission delay (Solow - ModelCode = 2 or 3)
    delta_nb = 1., # rho_nb = 34.08792
    coef = c(0.3,0,1), # coefficients for the hyperbolic tangent gain function
    rng.seed = NULL,
    delta_grid = seq(from=0.8,to=0.99,by=0.01)) { # searching range for LBE discount factor
  
  UPBND = 700
  
  c1 = 2*rho
  c2 = rho^2
  c3 = (1-rho)^2
  alpha2 = ifelse(solow_alpha==1,alpha,1)
  
  ntotal = m + n
  
  wt = rep(0,ntotal)
  if (!is.null(rng.seed)) { set.seed(rng.seed)}
  
  if (err_type == 0) { # Normal
    wt = rnorm(ntotal,0,sqrt(W)) # wt[1] = 0
  } else if (err_type == 1) { # Laplace
    wt = rmutil::rlaplace(ntotal,0,sqrt(W))
  } else if (err_type == 2) { # Cauchy
    wt = rcauchy(ntotal-1,0,sqrt(W))
  } else if (err_type == 3) { # left skewed normal
    wt = sn::rsn(ntotal,xi=0,omega=sqrt(W),alpha=-0.1)
    
  }
  
  
  if (is.null(psi0)) {
    set.seed(rng.seed)
    psi0 = rnorm(1,0,sd=1)
  }
  psi = cumsum(wt) + psi0
  hpsi = psi
  
  theta = rep(0,ntotal)
  lambda = rep(0,ntotal)
  y = rep(0,ntotal)
  
  if (is.null(theta0)) {
    set.seed(rng.seed)
    theta0 = runif(1,0,2)
  }
  
  if (ModelCode == 0) { # KoyamaMax
    # <obs> y[t] | lambda[t] ~ Pois(lambda[t])
    # <link> lambda[t] = lambda[0] + theta[t]
    # <state> theta[t] = phi[1] max(psi[t],0) y[t-1] + phi[2] max(psi[t-1],0) y[t-2] + ... + phi[L] max(psi[t-L+1],0) y[t-L]
    # <state> psi[t] = psi[t-1] + omega[t], omega[t] ~ N(0,W)
    hpsi[hpsi<.Machine$double.eps] = .Machine$double.eps # Reproduction number after maximum thresholding
    Fphi = get_Fphi(L)^alpha
    lambda[1] = mu0 + max(c(0,psi0))
    
    if (!is.null(rng.seed)) { set.seed(rng.seed)}
    if (obs_type==1) { # Poisson
      y[1] = rpois(1,lambda[1])
    } else if (obs_type==0) { # Negative-binomial
      y[1] = rnbinom(1, delta_nb, delta_nb/(lambda[1]+delta_nb))
    } else {
      stop("Not supported likelihood.")
    }
    
    for (t in 2:ntotal) {
      ytilde = y[(t-1):max(c(1,t-L))]
      nlag = length(ytilde)
      theta[t] = sum(Fphi[1:nlag]*hpsi[t:(t-nlag+1)]*ytilde) # <state>
      lambda[t] = mu0 + theta[t] # <link - intensity>
      
      if (obs_type==1) { # Poisson
        y[t] = rpois(1,lambda[t])
      } else if (obs_type==0) { # Negative-binomial
        y[t] = rnbinom(1, delta_nb, delta_nb/(lambda[t]+delta_nb))
      } else {
        stop("Not supported likelihood.")
      }
    }
  } else if (ModelCode == 1) { # KoyamaExp
    # <obs> y[t] | lambda[t] ~ Pois(lambda[t])
    # <link> lambda[t] = lambda[0] + theta[t]
    # <state> theta[t] = phi[1] exp(psi[t]) y[t-1] + phi[2] exp(psi[t-1]) y[t-2] + ... + phi[L] exp(psi[t-L+1]) y[t-L]
    # <state> psi[t] = psi[t-1] + omega[t], omega[t] ~ N(0,W)
    hpsi[hpsi>UPBND] = UPBND
    hpsi = exp(hpsi) # exponentiated reproduction number
    Fphi = get_Fphi(L)^alpha
    lambda[1] = mu0
    
    if (!is.null(rng.seed)) { set.seed(rng.seed)}
    if (obs_type==1) { # Poisson
      y[1] = rpois(1,lambda[1])
    } else if (obs_type==0) { # Negative-binomial
      y[1] = rnbinom(1, delta_nb, delta_nb/(lambda[1]+delta_nb))
    } else {
      stop("Not supported likelihood.")
    }
    for (t in 2:ntotal) {
      ytilde = y[(t-1):max(c(1,t-L))]
      nlag = length(ytilde)
      theta[t] = sum(Fphi[1:nlag]*hpsi[t:(t-nlag+1)]*ytilde) # <state>
      lambda[t] = mu0 + theta[t] # <link - intensity>
      
      if (obs_type==1) { # Poisson
        y[t] = rpois(1,lambda[t])
      } else if (obs_type==0) { # Negative-binomial
        y[t] = rnbinom(1, delta_nb, delta_nb/(lambda[t]+delta_nb))
      } else {
        stop("Not supported likelihood.")
      }
    }
  } else if (ModelCode == 2) { # SolowMax
    # <obs> y[t] ~ Pois(lambda[t])
    # <link> lambda[t] = lambda[0] + theta[t]
    # <state> theta[t] = 2*rho*theta[t-1] - rho^2*theta[t-2] + (1-rho)^2*y[t-1]*max(psi[t-1],0)
    # <state> psi[t] = psi[t-1] + omega[t]
    hpsi[hpsi<.Machine$double.eps] = .Machine$double.eps # Reproduction number after maximum thresholding
    lambda[1] = mu0
    
    if (!is.null(rng.seed)) { set.seed(rng.seed)}
    if (obs_type==1) { # Poisson
      y[1] = rpois(1,lambda[1])
    } else if (obs_type==0) { # Negative-binomial
      y[1] = rnbinom(1, delta_nb, delta_nb/(lambda[1]+delta_nb))
    } else {
      stop("Not supported likelihood.")
    }
    for (t in 2:ntotal) {
      if (t==2) {
        theta[t] = c1^alpha2*theta[1] + c3^alpha*y[t-1]*hpsi[t-1]
      } else {
        theta[t] = c1^alpha2*theta[t-1] - c2^alpha2*theta[t-2] + c3^alpha*y[t-1]*hpsi[t-1]
      }
      lambda[t] = mu0 + theta[t]
      if (obs_type==1) { # Poisson
        y[t] = rpois(1,lambda[t])
      } else if (obs_type==0) { # Negative-binomial
        y[t] = rnbinom(1, delta_nb, delta_nb/(lambda[t]+delta_nb))
      } else {
        stop("Not supported likelihood.")
      }
    }
    
  } else if (ModelCode == 3) { # SolowExp
    # <obs> y[t] ~ Pois(lambda[t])
    # <link> lambda[t] = mu0 + theta[t]
    # <state> theta[t] = 2*rho*theta[t-1] - rho^2*theta[t-2] + (1-rho)^2*y[t-1]*exp(psi[t-1])
    # <state> psi[t] = psi[t-1] + omega[t]
    hpsi[hpsi>UPBND] = UPBND
    hpsi = exp(hpsi)
    theta[1] = (2*rho)^alpha2*theta0
    lambda[1] = mu0 + theta[1]
    if (!is.null(rng.seed)) { set.seed(rng.seed)}
    if (obs_type==1) { # Poisson
      y[1] = rpois(1,lambda[1])
    } else if (obs_type==0) { # Negative-binomial
      y[1] = rnbinom(1, delta_nb, delta_nb/(lambda[1]+delta_nb))
    } else {
      stop("Not supported likelihood.")
    }
    
    for (t in 2:ntotal) {
      if (t==2) {
        theta[t] = c1^alpha2*theta[1] - c2^alpha2*theta0 + c3^alpha*y[t-1]*hpsi[t-1]
      } else {
        theta[t] = c1^alpha2*theta[t-1] - c2^alpha2*theta[t-2] + c3^alpha*y[t-1]*hpsi[t-1]
      }
      lambda[t] = mu0 + theta[t]
      
      if (obs_type==1) { # Poisson
        y[t] = rpois(1,lambda[t])
      } else if (obs_type==0) { # Negative-binomial
        y[t] = rnbinom(1, delta_nb, delta_nb/(lambda[t]+delta_nb))
      }
    }
    
  } else if (ModelCode == 4) { # KoyckMax
    # <obs> y[t] ~ Pois(lambda[t])
    # <link> lambda[t] = theta[t]
    # <state> theta[t] = rho*theta[t-1] + y[t-1]*max(psi[t-1],0)
    # <state> psi[t] = psi[t-1] + omega[t]
    hpsi[hpsi<.Machine$double.eps] = .Machine$double.eps # Reproduction number after maximum thresholding
    lambda[1] = mu0
    
    if (!is.null(rng.seed)) { set.seed(rng.seed)}
    if (obs_type==1) { # Poisson
      y[1] = rpois(1,lambda[1])
    } else if (obs_type==0) { # Negative-binomial
      y[1] = rnbinom(1, delta_nb, delta_nb/(lambda[1]+delta_nb))
    } else {
      stop("Not supported likelihood.")
    }
    for (t in 2:ntotal) {
      theta[t] = rho*theta[t-1] + y[t-1]*hpsi[t-1]
      lambda[t] = mu0 + theta[t]
      if (obs_type==1) { # Poisson
        y[t] = rpois(1,lambda[t])
      } else if (obs_type==0) { # Negative-binomial
        y[t] = rnbinom(1, delta_nb, delta_nb/(lambda[t]+delta_nb))
      } else {
        stop("Not supported likelihood.")
      }
    }
    
  } else if (ModelCode == 5) { # KoyckExp
    # <obs> y[t] ~ Pois(lambda[t])
    # <link> lambda[t] = theta[t]
    # <state> theta[t] = rho*theta[t-1] + y[t-1]*exp(psi[t-1])
    # <state> psi[t] = psi[t-1] + omega[t]
    hpsi[hpsi>UPBND] = UPBND
    hpsi = exp(hpsi)
    lambda[1] = mu0
    
    if (!is.null(rng.seed)) { set.seed(rng.seed)}
    if (obs_type==1) { # Poisson
      y[1] = rpois(1,lambda[1])
    } else if (obs_type==0) { # Negative-binomial
      y[1] = rnbinom(1, delta_nb, delta_nb/(lambda[1]+delta_nb))
    } else {
      stop("Not supported likelihood.")
    }
    for (t in 2:ntotal) {
      theta[t] = rho*theta[t-1] + y[t-1]*hpsi[t-1]
      lambda[t] = mu0 + theta[t]
      if (obs_type==1) { # Poisson
        y[t] = rpois(1,lambda[t])
      } else if (obs_type==0) { # Negative-binomial
        y[t] = rnbinom(1, delta_nb, delta_nb/(lambda[t]+delta_nb))
      } else {
        stop("Not supported likelihood.")
      }
    }
  } else if (ModelCode == 6) { # KoyamaEye
    # <obs> y[t] | lambda[t] ~ Pois(lambda[t])
    # <link> lambda[t] = exp(theta[t])
    # <state> theta[t] = phi[1] psi[t] y[t-1] + phi[2] psi[t-1] y[t-2] + ... + phi[L] psi[t-L+1] y[t-L]
    # <state> psi[t] = psi[t-1] + omega[t], omega[t] ~ N(0,W)
    Fphi = get_Fphi(L)^alpha
    theta[1] = .Machine$double.eps
    lambda[1] = exp(min(c(mu0+theta[1],UPBND)))
    
    if (!is.null(rng.seed)) { set.seed(rng.seed)}
    if (obs_type==1) { # Poisson
      y[1] = rpois(1,lambda[1])
    } else if (obs_type==0) { # Negative-binomial
      y[1] = rnbinom(1, delta_nb, delta_nb/(lambda[1]+delta_nb))
    } else {
      stop("Not supported likelihood.")
    }
    for (t in 2:ntotal) {
      ytilde = y[(t-1):max(c(1,t-L))]
      nlag = length(ytilde)
      theta[t] = sum(Fphi[1:nlag]*hpsi[t:(t-nlag+1)]*ytilde) # <state>
      lambda[t] = exp(min(c(mu0+theta[t],UPBND))) # <link - intensity>
      if (obs_type==1) { # Poisson
        y[t] = rpois(1,lambda[t])
      } else if (obs_type==0) { # Negative-binomial
        y[t] = rnbinom(1, delta_nb, delta_nb/(lambda[t]+delta_nb))
      } else {
        stop("Not supported likelihood.")
      }
    }
  } else if (ModelCode == 7) { # SolowEye
    # <obs> y[t] ~ Pois(lambda[t])
    # <link> lambda[t] = exp(theta[t])
    # <state> theta[t] = 2*rho*theta[t-1] - rho^2*theta[t-2] + (1-rho)^2*y[t-1]*psi[t-1]
    # <state> psi[t] = psi[t-1] + omega[t]
    theta[1] = 2.*rho*theta0
    lambda[1] = exp(min(c(mu0+theta[1],UPBND)))
    
    if (!is.null(rng.seed)) { set.seed(rng.seed)}
    if (obs_type==1) { # Poisson
      y[1] = rpois(1,lambda[1])
    } else if (obs_type==0) { # Negative-binomial
      y[1] = rnbinom(1, delta_nb, delta_nb/(lambda[1]+delta_nb))
    } else {
      stop("Not supported likelihood.")
    }
    for (t in 2:ntotal) {
      if (t==2) {
        theta[t] = c1^alpha2*theta[1] - c2^alpha2*theta0 + c3^alpha*y[t-1]*hpsi[t-1]
      } else {
        theta[t] = c1^alpha2*theta[t-1] - c2^alpha2*theta[t-2] + c3^alpha*y[t-1]*hpsi[t-1]
      }
      lambda[t] =  exp(min(c(mu0+theta[t],UPBND)))
      if (obs_type==1) { # Poisson
        y[t] = rpois(1,lambda[t])
      } else if (obs_type==0) { # Negative-binomial
        y[t] = rnbinom(1, delta_nb, delta_nb/(lambda[t]+delta_nb))
      } else {
        stop("Not supported likelihood.")
      }
    }
  } else if (ModelCode == 8) { # KoyckEye
    # <obs> y[t] ~ Pois(lambda[t])
    # <link> lambda[t] = exp(mu0 + theta[t])
    # <state> theta[t] = rho*theta[t-1] + y[t-1]*psi[t-1]
    # <state> psi[t] = psi[t-1] + omega[t]
    theta[1] = .Machine$double.eps
    lambda[1] = exp(min(c(mu0+theta[1],UPBND)))
    
    if (!is.null(rng.seed)) { set.seed(rng.seed)}
    if (obs_type==1) { # Poisson
      y[1] = rpois(1,lambda[1])
    } else if (obs_type==0) { # Negative-binomial
      y[1] = rnbinom(1, delta_nb, delta_nb/(lambda[1]+delta_nb))
    } else {
      stop("Not supported likelihood.")
    }
    for (t in 2:ntotal) {
      theta[t] = rho*theta[t-1] + y[t-1]*hpsi[t-1]
      lambda[t] = exp(min(c(mu0+theta[t],UPBND)))
      if (obs_type==1) { # Poisson
        y[t] = rpois(1,lambda[t])
      } else if (obs_type==0) { # Negative-binomial
        y[t] = rnbinom(1, delta_nb, delta_nb/(lambda[t]+delta_nb))
      } else {
        stop("Not supported likelihood.")
      }
    }
  } else if (ModelCode == 9) { # VanillaPois
    # <obs> y[t] ~ Pois(lambda[t])
    # <link> lambda[t] = exp(mu0 + theta[t])
    # <state> theta[t] = rho*theta[t-1] + omega[t]
    theta[1] = .Machine$double.eps + rho*theta0 + wt[1]
    lambda[1] = exp(min(c(mu0+theta[1],UPBND)))
    
    if (!is.null(rng.seed)) { set.seed(rng.seed)}
    if (obs_type==1) { # Poisson
      y[1] = rpois(1,lambda[1])
    } else if (obs_type==0) { # Negative-binomial
      y[1] = rnbinom(1, delta_nb, delta_nb/(lambda[1]+delta_nb))
    } else {
      stop("Not supported likelihood.")
    }
    for (t in 2:ntotal) {
      theta[t] = rho*theta[t-1] + wt[t]
      lambda[t] = exp(min(c(mu0+theta[t],UPBND)))
      if (obs_type==1) { # Poisson
        y[t] = rpois(1,lambda[t])
      } else if (obs_type==0) { # Negative-binomial
        y[t] = rnbinom(1, delta_nb, delta_nb/(lambda[t]+delta_nb))
      } else {
        stop("Not supported likelihood.")
      }
    }
    
  } else if (ModelCode == 10) { # KoyckSoftplus
    # <obs> y[t] ~ Pois(lambda[t])
    # <link> lambda[t] = theta[t]
    # <state> theta[t] = rho*theta[t-1] + y[t-1]*ln( 1+exp(psi[t-1]) )
    # <state> psi[t] = psi[t-1] + omega[t]
    hpsi[hpsi>UPBND] = UPBND
    hpsi = log(1. + exp(hpsi))
    lambda[1] = mu0
    
    if (!is.null(rng.seed)) { set.seed(rng.seed)}
    if (obs_type==1) { # Poisson
      y[1] = rpois(1,lambda[1])
    } else if (obs_type==0) { # Negative-binomial
      y[1] = rnbinom(1, delta_nb, delta_nb/(lambda[1]+delta_nb))
    } else {
      stop("Not supported likelihood.")
    }
    for (t in 2:ntotal) {
      theta[t] = rho*theta[t-1] + y[t-1]*hpsi[t-1]
      lambda[t] = mu0 + theta[t]
      if (obs_type==1) { # Poisson
        y[t] = rpois(1,lambda[t])
      } else if (obs_type==0) { # Negative-binomial
        y[t] = rnbinom(1, delta_nb, delta_nb/(lambda[t]+delta_nb))
      } else {
        stop("Not supported likelihood.")
      }
    }
  } else if (ModelCode == 11) { # KoyamaSoftplus
    # <obs> y[t] | lambda[t] ~ Pois(lambda[t])
    # <link> lambda[t] = lambda[0] + theta[t]
    # <state> theta[t] = phi[1] exp(psi[t]) y[t-1] + phi[2] exp(psi[t-1]) y[t-2] + ... + phi[L] exp(psi[t-L+1]) y[t-L]
    # <state> psi[t] = psi[t-1] + omega[t], omega[t] ~ N(0,W)
    hpsi[hpsi>UPBND] = UPBND
    hpsi = log(1. + exp(hpsi)) # softplused reproduction number
    Fphi = get_Fphi(L)^alpha
    lambda[1] = mu0
    
    if (!is.null(rng.seed)) { set.seed(rng.seed)}
    if (obs_type==1) { # Poisson
      y[1] = rpois(1,lambda[1])
    } else if (obs_type==0) { # Negative-binomial
      y[1] = rnbinom(1, delta_nb, delta_nb/(lambda[1]+delta_nb))
    } else {
      stop("Not supported likelihood.")
    }
    for (t in 2:ntotal) {
      ytilde = y[(t-1):max(c(1,t-L))]
      nlag = length(ytilde)
      theta[t] = sum(Fphi[1:nlag]*hpsi[t:(t-nlag+1)]*ytilde) # <state>
      lambda[t] = mu0 + theta[t] # <link - intensity>
      if (obs_type==1) { # Poisson
        y[t] = rpois(1,lambda[t])
      } else if (obs_type==0) { # Negative-binomial
        y[t] = rnbinom(1, delta_nb, delta_nb/(lambda[t]+delta_nb))
      } else {
        stop("Not supported likelihood.")
      }
    }
    
  } else if (ModelCode == 12) { # SolowSoftplus
    # <obs> y[t] ~ Pois(lambda[t])
    # <link> lambda[t] = mu0 + theta[t]
    # <state> theta[t] = 2*rho*theta[t-1] - rho^2*theta[t-2] + (1-rho)^2*y[t-1]*exp(psi[t-1])
    # <state> psi[t] = psi[t-1] + omega[t]
    hpsi[hpsi>UPBND] = UPBND
    hpsi = log(1. + exp(hpsi))
    theta[1] = (2*rho)^alpha2*theta0
    lambda[1] = mu0 + theta[1]
    if (!is.null(rng.seed)) { set.seed(rng.seed)}
    if (obs_type==1) { # Poisson
      y[1] = rpois(1,lambda[1])
    } else if (obs_type==0) { # Negative-binomial
      y[1] = rnbinom(1, delta_nb, delta_nb/(lambda[1]+delta_nb))
    } else {
      stop("Not supported likelihood.")
    }
    
    for (t in 2:ntotal) {
      if (t==2) {
        theta[t] = c1^alpha2*theta[1] - c2^alpha2*theta0 + c3^alpha*y[t-1]*hpsi[t-1]
      } else {
        theta[t] = c1^alpha2*theta[t-1] - c2^alpha2*theta[t-2] + c3^alpha*y[t-1]*hpsi[t-1]
      }
      lambda[t] = mu0 + theta[t]
      if (obs_type==1) { # Poisson
        y[t] = rpois(1,lambda[t])
      } else if (obs_type==0) { # Negative-binomial
        y[t] = rnbinom(1, delta_nb, delta_nb/(lambda[t]+delta_nb))
      } else {
        stop("Not supported likelihood.")
      }
    }
    
  } else if (ModelCode == 13) { # KoyckTanh
    # <obs> y[t] ~ Pois(lambda[t])
    # <link> lambda[t] = theta[t]
    # <state> theta[t] = rho*theta[t-1] + y[t-1]*c*{tanh(a*psi[t]+b)+1}
    # <state> psi[t] = psi[t-1] + omega[t]
    hpsi = 0.5*coef[3] * (tanh(coef[1]*hpsi + coef[2]) + 1)
    lambda[1] = mu0
    
    if (!is.null(rng.seed)) { set.seed(rng.seed)}
    if (obs_type==1) { # Poisson
      y[1] = rpois(1,lambda[1])
    } else if (obs_type==0) { # Negative-binomial
      y[1] = rnbinom(1, delta_nb, delta_nb/(lambda[1]+delta_nb))
    } else {
      stop("Not supported likelihood.")
    }
    for (t in 2:ntotal) {
      theta[t] = rho*theta[t-1] + y[t-1]*hpsi[t-1]
      lambda[t] = mu0 + theta[t]
      if (obs_type==1) { # Poisson
        y[t] = rpois(1,lambda[t])
      } else if (obs_type==0) { # Negative-binomial
        y[t] = rnbinom(1, delta_nb, delta_nb/(lambda[t]+delta_nb))
      } else {
        stop("Not supported likelihood.")
      }
    }
  } else if (ModelCode == 14) { # KoyamaTanh
    # <obs> y[t] | lambda[t] ~ Pois(lambda[t])
    # <link> lambda[t] = lambda[0] + theta[t]
    # <state> theta[t] = phi[1] h(psi[t]) y[t-1] + phi[2] h(psi[t-1]) y[t-2] + ... + phi[L] exp(psi[t-L+1]) y[t-L]
    # <state> psi[t] = psi[t-1] + omega[t], omega[t] ~ N(0,W)
    hpsi = 0.5*coef[3] * (tanh(coef[1]*hpsi + coef[2]) + 1)
    Fphi = get_Fphi(L)^alpha
    lambda[1] = mu0
    
    if (!is.null(rng.seed)) { set.seed(rng.seed)}
    if (obs_type==1) { # Poisson
      y[1] = rpois(1,lambda[1])
    } else if (obs_type==0) { # Negative-binomial
      y[1] = rnbinom(1, delta_nb, delta_nb/(lambda[1]+delta_nb))
    } else {
      stop("Not supported likelihood.")
    }
    for (t in 2:ntotal) {
      ytilde = y[(t-1):max(c(1,t-L))]
      nlag = length(ytilde)
      theta[t] = sum(Fphi[1:nlag]*hpsi[t:(t-nlag+1)]*ytilde) # <state>
      lambda[t] = mu0 + theta[t] # <link - intensity>
      if (obs_type==1) { # Poisson
        y[t] = rpois(1,lambda[t])
      } else if (obs_type==0) { # Negative-binomial
        y[t] = rnbinom(1, delta_nb, delta_nb/(lambda[t]+delta_nb))
      } else {
        stop("Not supported likelihood.")
      }
    }
  } else if (ModelCode == 15) { # SolowTanh
    # <obs> y[t] ~ Pois(lambda[t])
    # <link> lambda[t] = mu0 + theta[t]
    # <state> theta[t] = 2*rho*theta[t-1] - rho^2*theta[t-2] + (1-rho)^2*y[t-1]*exp(psi[t-1])
    # <state> psi[t] = psi[t-1] + omega[t]
    hpsi = 0.5*coef[3] * (tanh(coef[1]*hpsi + coef[2]) + 1)
    theta[1] = (2*rho)^alpha2*theta0
    lambda[1] = mu0 + theta[1]
    if (!is.null(rng.seed)) { set.seed(rng.seed)}
    if (obs_type==1) { # Poisson
      y[1] = rpois(1,lambda[1])
    } else if (obs_type==0) { # Negative-binomial
      y[1] = rnbinom(1, delta_nb, delta_nb/(lambda[1]+delta_nb))
    } else {
      stop("Not supported likelihood.")
    }
    
    for (t in 2:ntotal) {
      if (t==2) {
        theta[t] = c1^alpha2*theta[1] - c2^alpha2*theta0 + c3^alpha*y[t-1]*hpsi[t-1]
      } else {
        theta[t] = c1^alpha2*theta[t-1] - c2^alpha2*theta[t-2] + c3^alpha*y[t-1]*hpsi[t-1]
      }
      lambda[t] = mu0 + theta[t]
      if (obs_type==1) { # Poisson
        y[t] = rpois(1,lambda[t])
      } else if (obs_type==0) { # Negative-binomial
        y[t] = rnbinom(1, delta_nb, delta_nb/(lambda[t]+delta_nb))
      } else {
        stop("Not supported likelihood.")
      }
    }
  } else if (ModelCode == 16) { # KoyckLogistic
    # <obs> y[t] ~ Pois(lambda[t])
    # <link> lambda[t] = theta[t]
    # <state> theta[t] = rho*theta[t-1] + y[t-1]*c/{exp(-a*psi[t]+a*b)+1}
    # <state> psi[t] = psi[t-1] + omega[t]
    hpsi = -coef[1]*hpsi + coef[1]*coef[2]
    hpsi[hpsi>UPBND] = UPBND
    hpsi = coef[3] / (exp(hpsi) + 1)
    lambda[1] = mu0
    
    if (!is.null(rng.seed)) { set.seed(rng.seed)}
    if (obs_type==1) { # Poisson
      y[1] = rpois(1,lambda[1])
    } else if (obs_type==0) { # Negative-binomial
      y[1] = rnbinom(1, delta_nb, delta_nb/(lambda[1]+delta_nb))
    } else {
      stop("Not supported likelihood.")
    }
    for (t in 2:ntotal) {
      theta[t] = rho*theta[t-1] + y[t-1]*hpsi[t-1]
      lambda[t] = mu0 + theta[t]
      if (obs_type==1) { # Poisson
        y[t] = rpois(1,lambda[t])
      } else if (obs_type==0) { # Negative-binomial
        y[t] = rnbinom(1, delta_nb, delta_nb/(lambda[t]+delta_nb))
      } else {
        stop("Not supported likelihood.")
      }
    }
  } else if (ModelCode == 17) { # KoyamaLogistic
    # <obs> y[t] | lambda[t] ~ Pois(lambda[t])
    # <link> lambda[t] = lambda[0] + theta[t]
    # <state> theta[t] = phi[1] h(psi[t]) y[t-1] + phi[2] h(psi[t-1]) y[t-2] + ... + phi[L] exp(psi[t-L+1]) y[t-L]
    # <state> psi[t] = psi[t-1] + omega[t], omega[t] ~ N(0,W)
    hpsi = -coef[1]*hpsi + coef[1]*coef[2]
    hpsi[hpsi>UPBND] = UPBND
    hpsi = coef[3] / (exp(hpsi) + 1)
    Fphi = get_Fphi(L)^alpha
    lambda[1] = mu0
    
    if (!is.null(rng.seed)) { set.seed(rng.seed)}
    if (obs_type==1) { # Poisson
      y[1] = rpois(1,lambda[1])
    } else if (obs_type==0) { # Negative-binomial
      y[1] = rnbinom(1, delta_nb, delta_nb/(lambda[1]+delta_nb))
    } else {
      stop("Not supported likelihood.")
    }
    for (t in 2:ntotal) {
      ytilde = y[(t-1):max(c(1,t-L))]
      nlag = length(ytilde)
      theta[t] = sum(Fphi[1:nlag]*hpsi[t:(t-nlag+1)]*ytilde) # <state>
      lambda[t] = mu0 + theta[t] # <link - intensity>
      if (obs_type==1) { # Poisson
        y[t] = rpois(1,lambda[t])
      } else if (obs_type==0) { # Negative-binomial
        y[t] = rnbinom(1, delta_nb, delta_nb/(lambda[t]+delta_nb))
      } else {
        stop("Not supported likelihood.")
      }
    }
  } else if (ModelCode == 18) { # SolowLogistic
    # <obs> y[t] ~ Pois(lambda[t])
    # <link> lambda[t] = mu0 + theta[t]
    # <state> theta[t] = 2*rho*theta[t-1] - rho^2*theta[t-2] + (1-rho)^2*y[t-1]*exp(psi[t-1])
    # <state> psi[t] = psi[t-1] + omega[t]
    hpsi = -coef[1]*hpsi + coef[1]*coef[2]
    hpsi[hpsi>UPBND] = UPBND
    hpsi = coef[3] / (exp(hpsi) + 1)
    theta[1] = (2*rho)^alpha2*theta0
    lambda[1] = mu0 + theta[1]
    if (!is.null(rng.seed)) { set.seed(rng.seed)}
    if (obs_type==1) { # Poisson
      y[1] = rpois(1,lambda[1])
    } else if (obs_type==0) { # Negative-binomial
      y[1] = rnbinom(1, delta_nb, delta_nb/(lambda[1]+delta_nb))
    } else {
      stop("Not supported likelihood.")
    }
    
    for (t in 2:ntotal) {
      if (t==2) {
        theta[t] = c1^alpha2*theta[1] - c2^alpha2*theta0 + c3^alpha*y[t-1]*hpsi[t-1]
      } else {
        theta[t] = c1^alpha2*theta[t-1] - c2^alpha2*theta[t-2] + c3^alpha*y[t-1]*hpsi[t-1]
      }
      lambda[t] = mu0 + theta[t]
      if (obs_type==1) { # Poisson
        y[t] = rpois(1,lambda[t])
      } else if (obs_type==0) { # Negative-binomial
        y[t] = rnbinom(1, delta_nb, delta_nb/(lambda[t]+delta_nb))
      } else {
        stop("Not supported likelihood.")
      }
    }
  } else {
    stop("Not implemented yet.")
  }
  
  model_code = get_model_list(model_code=ModelCode)
  model_code = as.integer(c(obs_type,c(model_code),err_type))
  
  delta = NULL
  if (!(ModelCode %in% c(0,2,4))) {
    tryCatch({
      delta = get_optimal_delta(y[1:n],model_code,delta_grid,
                                rho=rho,L=L,mu0=mu0,
                                delta_nb=delta_nb,
                                ctanh=coef,
                                obs_type=obs_type)$delta_optim[1]
    },error=function(e){delta=NA})
  }
  
  
  
  
  
  params = list(model_code=model_code,
                mu0=mu0,theta0=theta0,psi0=psi0,
                W=W,rho=rho,L=L,
                delta_nb=delta_nb,
                delta_lbe=delta,
                ctanh=coef,alpha=alpha)
  pred = list(y=y[(n+1):ntotal],
              psi=psi[(n+1):ntotal],
              theta=theta[(n+1):ntotal],
              lambda=lambda[(n+1):ntotal])
  
  return(list(y=y[1:n], # Observation
              lambda=lambda[1:n], # Intensity
              theta=theta[1:n], # Transfer Function Block
              psi=psi[1:n], # Gain Factor
              hpsi=hpsi[1:n], # Reproduction Number - Function of the Gain Factor
              wt=wt[1:n], # Evolution Variance
              params=params,
              pred=pred)) # Model settings and initial values
}