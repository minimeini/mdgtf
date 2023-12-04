cdir = getwd()
repo = "/Users/meinitang/Dropbox/Repository/poisson-dlm"
UPBND = 700
EPS = .Machine$double.eps


# TODO: compare it to the older version
sim_pois_dglm2 = function(
    n = 200, # number of observations for training
    m = 0, # number of observations for testing
    obs_dist = "nbinom", # {"nbinom","poisson"}
    link_func = "identity", # {"identity","exponential"}
    trans_func = "koyama", # {"koyama","solow","koyck"}
    gain_func = "logistic", # {"exponential","softplus","logistic"}
    err_dist = "gaussian", # {"gaussian","laplace","cauchy","left_skewed_normal"},
    obs_params = c(0.,30.), # (mu0, delta_nb),
    lag_params = c(0.5,6), # (rho, L) or (mu, sg2)
    W_params = c(0.01,2,0.01,0.01),
    nlag = 20,
    ci_coverage = 0.95,
    rng.seed = NULL,
    get_discount = TRUE,
    truncated = TRUE,
    delta_grid = seq(from=0.8,to=0.99,by=0.01)) { # searching range for LBE discount factor
  
  mu0 = obs_params[1]
  delta_nb = obs_params[2]
  
  par1 = lag_params[1] # rho or mu
  par2 = lag_params[2] # L or sg2
  
  ntotal = m + n
  model_code = get_model_code(
    obs_dist,link_func,trans_func,gain_func,err_dist)
  
  obs_code = model_code[1]
  link_code = model_code[2]
  trans_code = model_code[3]
  gain_code = model_code[4]
  err_code = model_code[5]


  wt = rep(0,ntotal)
  if (!is.null(rng.seed)) { set.seed(rng.seed)}
  
  W = W_params[1]
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
  hpsi = psi2hpsi(matrix(psi,ncol=1),gain_code)


  link_func = tolower(link_func)
  obs_dist = tolower(obs_dist)
  
  # ------ Transfer ------ #
  hpsi_pad = rep(0,ntotal + 1)
  hpsi_pad[(2): (ntotal+1)] = hpsi
  theta_pad = rep(0,ntotal + 1)
  lambda_pad = rep(0,ntotal + 1)
  ypad = rep(0,ntotal + 1)

  
  for (t in 2:(ntotal+1)) {
    tidx = t - 1
    theta_pad[t] = theta_new_nobs(hpsi_pad,ypad,tidx,lag_params,trans_code,nlag,truncated)
    lambda_pad[t] = mu0 + theta_pad[t]
    
    if (link_func == "exponential") {
      lambda_pad[t] = exp(min(c(lambda_pad[t], UPBND)))
    }

    if (obs_dist == "nbinom") {
      ypad[t] = rnbinom(1, delta_nb, delta_nb/(lambda_pad[t]+delta_nb))
    } else {
      ypad[t] = rpois(1,lambda_pad[t])
    }
  }
  
  stopifnot(all(is.finite(theta_pad)))
  stopifnot(all(is.finite(lambda_pad)))
  stopifnot(all(is.finite(ypad)))
  
  y = ypad[2:(ntotal+1)]
  theta = theta_pad[2:(ntotal+1)]
  lambda = lambda_pad[2:(ntotal+1)]
  
  if (get_discount) {
    # get_optimal_delta: exported Cpp function
    delta = get_optimal_delta(y[1:n],model_code,delta_grid,
                              obs_params,lag_params,2.,1,
                              ci_coverage,nlag,truncated)$delta_optim[1]
  } else 
  {
    delta = 0.95
  }

  
  params = list(model_code = model_code,
                obs_params = obs_params,
                lag_params = lag_params,
                W_params = W_params,
                nlag=nlag, delta_lbe=delta)
  # pred = list(y=y[(n+1):ntotal],
  #             psi=psi[(n+1):ntotal],
  #             theta=theta[(n+2):(ntotal+1)],
  #             lambda=lambda[(n+1):ntotal])
  
  return(list(y=y[1:n], # Observation
              lambda=lambda[1:n], # Intensity
              theta=theta[1:n], # Transfer Function Block
              psi=psi[1:n], # Gain Factor
              hpsi=hpsi[1:n], # Reproduction Number - Function of the Gain Factor
              wt=wt[1:n], # Evolution Variance
              params=params
              # pred=pred
              )) # Model settings and initial values
}
