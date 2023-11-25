cdir = getwd()
repo = "/Users/meinitang/Dropbox/Repository/poisson-dlm"
UPBND = 700
EPS = .Machine$double.eps

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
    delta_nb = 30., # rho_nb = 34.08792
    ci_coverage = 0.95,
    rng.seed = NULL,
    get_discount = TRUE,
    delta_grid = seq(from=0.8,to=0.99,by=0.01)) { # searching range for LBE discount factor
  
  
  # ------ Initialization ------ #
  ntotal = m + n
  model_code = get_model_code(
    obs_dist=obs_dist,
    link_func=link_func,
    trans_func=trans_func,
    gain_func=gain_func,
    err_dist=err_dist)
  
  obs_code = model_code[1]
  link_code = model_code[2]
  trans_code = model_code[3]
  gain_code = model_code[4]
  err_code = model_code[5]

  c1 = 2*rho
  c2 = rho^2
  c3 = (1-rho)^(L)

  if (is.null(mu0)) {mu0 = 0.}
  if (is.null(theta0)) {
    set.seed(rng.seed)
    theta0 = runif(1,0,2)
  }
  if (is.null(psi0)) {
    set.seed(rng.seed)
    psi0 = abs(rnorm(1,0,sd=1))
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
  
  
  psi = cumsum(wt)
  hpsi = psi
  gain_func = tolower(gain_func)
  if (gain_func=="ramp") {
    # Ramp
    hpsi[hpsi<EPS] = EPS # Reproduction number after maximum thresholding
  } else if (gain_func=="exponential") {
    # Exponential
    hpsi[hpsi>UPBND] = UPBND
    hpsi = exp(hpsi)
  } else if (gain_func=="identity") {
  } else if (gain_func=="softplus") {
    hpsi[hpsi>UPBND] = UPBND
    hpsi = log(1. + exp(hpsi))
  } else {
    stop("Not supported gain function.")
  }
  stopifnot(all(is.finite(hpsi)))
  # hpsi - Checked. Correct.
  # --------- Gain --------- #
  
  
  # ------ Transfer ------ #
  theta = rep(0,ntotal+1)

  lambda = rep(0,ntotal)
  y = rep(0,ntotal)
  
  # get_Fphi: exported Cpp function.
  # checked. OK
  if (trans_code == 1)
  {
    Fphi = get_Fphi(L,1,rho,trans_code)
  }
  else
  {
    Fphi = get_Fphi(n,L,rho,trans_code)
  }

  trans_func = tolower(trans_func)
  link_func = tolower(link_func)
  obs_dist = tolower(obs_dist)
  
  # ------ Transfer ------ #
  if (trans_func == "solow") {
    # theta[2] = (2*rho)^alpha2*theta0
    # binom: exported Cpp function
    theta[2] = -binom(L,1)*(-rho)^1*theta[1]
  } else {
    # theta[2] = EPS
    theta[2] = theta0
  }
  # ------ Transfer ------ #
  
  # ------ Link ------ #
  lambda[1] = mu0 + theta[2]
  if (link_func == "exponential") {
    lambda[1] = exp(min(c(lambda[1],UPBND)))
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
        # binom: exported Cpp function
        theta[t+1] = theta[t+1] - binom(L,k)*(-rho)^k*theta[t+1-k]
      }
    }
    # ------ Transfer ------ #
    
    # ------ Link ------ #
    lambda[t] = mu0 + theta[t+1]
    if (link_func == "exponential") {
      lambda[t] = exp(min(c(lambda[t], UPBND)))
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
  
  if (get_discount) {
    # get_optimal_delta: exported Cpp function
    delta = get_optimal_delta(y[1:n],model_code,delta_grid,
                              rho=rho,L=L,mu0=mu0,
                              delta_nb=delta_nb)$delta_optim[1]
  } else {
    delta = 0.95
  }

  
  params = list(model_code=model_code,
                mu0=mu0,theta0=theta0,psi0=psi0,
                W=W,rho=rho,L=L,
                delta_nb=delta_nb,
                delta_lbe=delta)
  # pred = list(y=y[(n+1):ntotal],
  #             psi=psi[(n+1):ntotal],
  #             theta=theta[(n+2):(ntotal+1)],
  #             lambda=lambda[(n+1):ntotal])
  
  return(list(y=y[1:n], # Observation
              lambda=lambda[1:n], # Intensity
              theta=theta[2:(n+1)], # Transfer Function Block
              psi=psi[1:n], # Gain Factor
              hpsi=hpsi[1:n], # Reproduction Number - Function of the Gain Factor
              wt=wt[1:n], # Evolution Variance
              params=params
              # pred=pred
              )) # Model settings and initial values
}
