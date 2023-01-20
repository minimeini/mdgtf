cdir = getwd()
repo = "/Users/meinitang/Dropbox/Repository/poisson-dlm"

Rcpp::sourceCpp(file.path(repo,"model_utils.cpp"))
Rcpp::sourceCpp(file.path(repo,"lbe_poisson.cpp"))

# TODO
# R can only parse default values in the Cpp file not the header file.
# We need a hacky way to adapt to it.

# ------------------------
# ------ Model Code ------
# ---------------------------------------------------------------------------------------
#       NAME           LINK        |  Transmission  |  Gain Function
# ---------------------------------------------------------------------------------------
# 0  - (KoyamaMax)     Identity    |  LogNorm       |  ramp,        max(psi[t],0)
# 1  - (KoyamaExp)     Identity    |  LogNorm       |  exponential, exp(psi[t])
# 2  - (SolowMax)      Identity    |  NegBinom      |  ramp,        max(psi[t],0)
# 3  - (SolowExp)      Identity    |  NegBinom      |  exponential, exp(psi[t])
# 4  - (KoyckMax)      Identity    |  Exponential   |  ramp,        max(psi[t],0)
# 5  - (KoyckExp)      Identity    |  Exponential   |  exponential, exp(psi[t])
# 6  - (KoyamaEye)     Exponential |  LogNorm       |  identity,    psi[t]
# 7  - (SolowEye)      Exponential |  NegBinom      |  identity,    psi[t]
# 8  - (KoyckEye)      Exponential |  Exponential   |  identity,    psi[t]
# 9  - (Vanilla)       Exponential |  Exponential   |  No gain
# 10 - (KoyckSoftplus) Identity    |  Exponential   |  Softplus,    ln(1+exp(psi[t]))
# 11 - (KoyamaSoftplus)Identity    |  Exponential   |  Softplus,    ln(1+exp(psi[t]))
# 12 - (SolowSoftplus) Identity    |  Exponential   |  Softplus,    ln(1+exp(psi[t]))
# ---------------------------------------------------------------------------------------



sim_pois_dglm = function(
    n = 200, # number of observations for training
    m = 20, # number of observations for testing
    ModelCode = 0, # 0 - KoyamaMax; 1 - KoyamaExp; 2 - SolowMax; 3 - SolowExp
    obs_type = 1, # 0 - negative-binomial; 1 - poisson
    mu0 = 0., # the baseline intensity
    psi0 = 0., # initial value of the gain factor
    theta0 = 0., # initial value for the transfer function block; set it to NULL to sample from a uniform distribution(0,10). Only used in Solow
    W = 0.01, # Evolution variance
    L = 0, # length of nonzero transmission delay (Koyama - ModelCode = 0 or 1)
    rho = 0.7, # parameter for negative binomial transmission delay (Solow - ModelCode = 2 or 3)
    delta_nb = 1., # rho_nb = 34.08792
    rng.seed = NULL,
    delta_grid = seq(from=0.7,to=0.99,by=0.01)) { # searching range for LBE discount factor
  
  UPBND = 700
  
  c1 = 2*rho
  c2 = rho^2
  c3 = (1-rho)^2
  
  ntotal = m + n
  
  wt = rep(0,ntotal)
  if (!is.null(rng.seed)) { set.seed(rng.seed)}
  wt[2:ntotal] = rnorm(ntotal-1,0,sqrt(W)) # wt[1] = 0
  
  if (is.null(psi0)) {
    set.seed(rng.seed)
    psi0 = rnorm(1,0,sd=1)
  }
  psi = cumsum(wt) + psi0
  psi[psi>UPBND] = UPBND
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
    Fphi = get_Fphi(L)
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
    hpsi = exp(hpsi) # exponentiated reproduction number
    Fphi = get_Fphi(L)
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
        theta[t] = c1*theta[1] + c3*y[t-1]*hpsi[t-1]
      } else {
        theta[t] = c1*theta[t-1] - c2*theta[t-2] + c3*y[t-1]*hpsi[t-1]
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
    hpsi = exp(hpsi)
    theta[1] = 2*rho*theta0
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
        theta[t] = c1*theta[1] + c3*y[t-1]*hpsi[t-1]
      } else {
        theta[t] = c1*theta[t-1] - c2*theta[t-2] + c3*y[t-1]*hpsi[t-1]
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
    Fphi = get_Fphi(L)
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
      if (t==2) {
        theta[t] = c1*theta[1] + c3*y[t-1]*hpsi[t-1]
      } else {
        theta[t] = c1*theta[t-1] - c2*theta[t-2] + c3*y[t-1]*hpsi[t-1]
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
    hpsi = log(1. + exp(hpsi)) # softplused reproduction number
    Fphi = get_Fphi(L)
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
    hpsi = log(1. + exp(hpsi))
    theta[1] = 2*rho*theta0
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
        theta[t] = c1*theta[1] + c3*y[t-1]*hpsi[t-1]
      } else {
        theta[t] = c1*theta[t-1] - c2*theta[t-2] + c3*y[t-1]*hpsi[t-1]
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
  
  delta = NULL
  if (!(ModelCode %in% c(0,2,4))) {
    tryCatch({
      delta = get_optimal_delta(y[1:n],ModelCode,delta_grid,
                                rho=rho,L=L,mu0=mu0,
                                delta_nb=delta_nb,
                                obs_type=obs_type)$delta_optim[1]
    })
  }
  
  
  
  params = list(ModelCode=ModelCode,obs_type=obs_type,
                mu0=mu0,theta0=theta0,psi0=psi0,
                W=W,rho=rho,L=L,
                delta_nb=delta_nb,
                delta_lbe=delta)
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




sim_transfer_function = function(
    x, # n x 1 vector
    ModelCode = 0,
    mu0 = 0., # the baseline intensity
    psi0 = 0.,
    theta0 = 0., # initial value for the transfer function block; set it to NULL to sample from a uniform distribution(0,10).
    W = 0.01, # Evolution variance
    L = 12, # length of nonzero transmission delay (Koyama - ModelCode = 0 or 1)
    rho = 0.7, # parameter for negative binomial transmission delay (Solow - ModelCode = 2 or 3)
    rng.seed = NULL) {
  
  n = length(x)
  
  UPBND = 100
  
  c1 = 2*rho
  c2 = rho^2
  c3 = (1-rho)^2
  
  wt = rep(0,n)
  if (!is.null(rng.seed)) { set.seed(rng.seed)}
  wt[2:n] = rnorm(n-1,0,sqrt(W)) # wt[1] = 0
  
  if (is.null(psi0)) {
    set.seed(rng.seed)
    psi0 = rnorm(1,0,sd=1)
  }
  psi = cumsum(wt) + psi0
  psi[psi>UPBND] = UPBND
  hpsi = psi
  
  theta = rep(0,n)
  if (is.null(theta0)) {
    set.seed(rng.seed)
    theta0 = runif(1,0,2)
  }
  
  if (ModelCode == 0) { # KoyamaMax
    # <state> theta[t] = phi[1] max(psi[t],0) y[t-1] + phi[2] max(psi[t-1],0) y[t-2] + ... + phi[L] max(psi[t-L+1],0) y[t-L]
    # <state> psi[t] = psi[t-1] + omega[t], omega[t] ~ N(0,W)
    hpsi[hpsi<.Machine$double.eps] = .Machine$double.eps # Reproduction number after maximum thresholding
    Fphi = get_Fphi(L)
    
    if (!is.null(rng.seed)) { set.seed(rng.seed)}
    for (t in 2:n) {
      xtilde = x[(t-1):max(c(1,t-L))]
      nlag = length(xtilde)
      theta[t] = sum(Fphi[1:nlag]*hpsi[t:(t-nlag+1)]*xtilde) # <state>
    }
  } else if (ModelCode == 1) { # KoyamaExp
    # <obs> y[t] | lambda[t] ~ Pois(lambda[t])
    # <link> lambda[t] = lambda[0] + theta[t]
    # <state> theta[t] = phi[1] exp(psi[t]) y[t-1] + phi[2] exp(psi[t-1]) y[t-2] + ... + phi[L] exp(psi[t-L+1]) y[t-L]
    # <state> psi[t] = psi[t-1] + omega[t], omega[t] ~ N(0,W)
    hpsi = exp(hpsi) # exponentiated reproduction number
    Fphi = get_Fphi(L)
    
    if (!is.null(rng.seed)) { set.seed(rng.seed)}
    for (t in 2:n) {
      xtilde = x[(t-1):max(c(1,t-L))]
      nlag = length(xtilde)
      theta[t] = sum(Fphi[1:nlag]*hpsi[t:(t-nlag+1)]*xtilde) # <state>
    }
  } else if (ModelCode == 2) { # SolowMax
    # <obs> y[t] ~ Pois(lambda[t])
    # <link> lambda[t] = lambda[0] + theta[t]
    # <state> theta[t] = 2*rho*theta[t-1] - rho^2*theta[t-2] + (1-rho)^2*y[t-1]*max(psi[t-1],0)
    # <state> psi[t] = psi[t-1] + omega[t]
    hpsi[hpsi<.Machine$double.eps] = .Machine$double.eps # Reproduction number after maximum thresholding
    
    if (!is.null(rng.seed)) { set.seed(rng.seed)}
    for (t in 2:n) {
      if (t==2) {
        theta[t] = c1*theta[1] + c3*x[t-1]*hpsi[t-1]
      } else {
        theta[t] = c1*theta[t-1] - c2*theta[t-2] + c3*x[t-1]*hpsi[t-1]
      }
    }
    
  } else if (ModelCode == 3) { # SolowExp
    # <obs> y[t] ~ Pois(lambda[t])
    # <link> lambda[t] = mu0 + theta[t]
    # <state> theta[t] = 2*rho*theta[t-1] - rho^2*theta[t-2] + (1-rho)^2*y[t-1]*exp(psi[t-1])
    # <state> psi[t] = psi[t-1] + omega[t]
    hpsi = exp(hpsi)
    theta[1] = 2*rho*theta0
    if (!is.null(rng.seed)) { set.seed(rng.seed)}
    
    for (t in 2:n) {
      if (t==2) {
        theta[t] = c1*theta[1] + c3*x[t-1]*hpsi[t-1]
      } else {
        theta[t] = c1*theta[t-1] - c2*theta[t-2] + c3*x[t-1]*hpsi[t-1]
      }
    }
    
  } else if (ModelCode == 4) { # KoyckMax
    # <obs> y[t] ~ Pois(lambda[t])
    # <link> lambda[t] = theta[t]
    # <state> theta[t] = rho*theta[t-1] + y[t-1]*max(psi[t-1],0)
    # <state> psi[t] = psi[t-1] + omega[t]
    hpsi[hpsi<.Machine$double.eps] = .Machine$double.eps # Reproduction number after maximum thresholding
    if (!is.null(rng.seed)) { set.seed(rng.seed)}
    for (t in 2:n) {
      theta[t] = rho*theta[t-1] + x[t-1]*hpsi[t-1]
    }
    
  } else if (ModelCode == 5) { # KoyckExp
    # <state> theta[t] = rho*theta[t-1] + y[t-1]*exp(psi[t-1])
    # <state> psi[t] = psi[t-1] + omega[t]
    hpsi = exp(hpsi)
    
    if (!is.null(rng.seed)) { set.seed(rng.seed)}
    x[1] = rpois(1,lambda[1])
    for (t in 2:n) {
      theta[t] = rho*theta[t-1] + x[t-1]*hpsi[t-1]
    }
  } else if (ModelCode == 6) { # KoyamaEye
    # <state> theta[t] = phi[1] psi[t] y[t-1] + phi[2] psi[t-1] y[t-2] + ... + phi[L] psi[t-L+1] y[t-L]
    # <state> psi[t] = psi[t-1] + omega[t], omega[t] ~ N(0,W)
    Fphi = get_Fphi(L)
    theta[1] = .Machine$double.eps
    
    if (!is.null(rng.seed)) { set.seed(rng.seed)}
    for (t in 2:n) {
      xtilde = x[(t-1):max(c(1,t-L))]
      nlag = length(xtilde)
      theta[t] = sum(Fphi[1:nlag]*hpsi[t:(t-nlag+1)]*xtilde) # <state>
    }
  } else if (ModelCode == 7) { # SolowEye
    # <state> theta[t] = 2*rho*theta[t-1] - rho^2*theta[t-2] + (1-rho)^2*y[t-1]*psi[t-1]
    # <state> psi[t] = psi[t-1] + omega[t]
    theta[1] = .Machine$double.eps
    if (!is.null(rng.seed)) { set.seed(rng.seed)}
    for (t in 2:n) {
      if (t==2) {
        theta[t] = c1*theta[1] + c3*x[t-1]*hpsi[t-1]
      } else {
        theta[t] = c1*theta[t-1] - c2*theta[t-2] + c3*x[t-1]*hpsi[t-1]
      }
    }
  } else if (ModelCode == 8) { # KoyckEye
    # <state> theta[t] = rho*theta[t-1] + y[t-1]*psi[t-1]
    # <state> psi[t] = psi[t-1] + omega[t]
    theta[1] = .Machine$double.eps
    if (!is.null(rng.seed)) { set.seed(rng.seed)}
    for (t in 2:n) {
      theta[t] = rho*theta[t-1] + x[t-1]*hpsi[t-1]
    }
  } else if (ModelCode == 9) {
    # <state> theta[t] = rho*theta[t-1] + omega[t]
    theta[1] = .Machine$double.eps + rho*theta0 + wt[1]
    
    if (!is.null(rng.seed)) { set.seed(rng.seed)}
    for (t in 2:n) {
      theta[t] = rho*theta[t-1] + wt[t]
    }
    
  } else {
    stop("Not implemented yet.")
  }
  
  return(list(theta=theta, # Transfer Function Block
              psi=psi, # Gain Factor
              hpsi=hpsi, # Reproduction Number - Function of the Gain Factor
              wt=wt, # Evolution Variance
              params=params)) # Model settings and initial values
}