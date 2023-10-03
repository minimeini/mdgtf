.info = new.env(parent=.GlobalEnv)
assign("repo","/Users/meinitang/Dropbox/Repository/poisson-dlm",envir=.info)
source(file.path(.info$repo,"model_info.R"),local=.info)

default_opts = function(
    n=NA,
    W = 0.01,
    mu0 = 0.1,
    obs_dist = "nbinom", # {"nbinom","poisson"}
    link_func = "identity", # {"identity","exponential"}
    trans_func = "koyama", # {"koyama","solow","koyck"}
    gain_func = "softplus", # {"exponential","softplus","logistic"}
    err_dist = "gaussian"){ # {"gaussian","laplace","cauchy","left_skewed_normal"}
  
  model_code = .info$get_model_code(
    obs_dist=obs_dist,
    link_func=link_func,
    trans_func=trans_func,
    gain_func=gain_func,
    err_dist=err_dist)
  
  opts = list(n = n,
              model_code = model_code,
              L = ifelse(model_code[3]==2,6,30),
              rho = 0.56,
              mu0 = mu0,
              psi0 = 0,
              theta0 = 0,
              W = W,
              delta = 0.99,
              delta_nb = 30,
              m0 = NULL,
              C0 = NULL,
              W_prior = NULL,
              mu0_prior = NULL,
              alpha = 1,
              ctanh = c(0.2,0,5))
  return(opts)
}


test_model = function(n,
                      obs_dist = "nbinom", # {"nbinom","poisson"}
                      link_func = "identity", # {"identity","exponential"}
                      trans_func = "koyama", # {"koyama","solow","koyck"}
                      gain_func = "logistic", # {"exponential","softplus","logistic"}
                      err_dist = "gaussian", # {"gaussian","laplace","cauchy","left_skewed_normal"}
                      mu0=1,theta0=NULL,psi0=0,
                      W=0.01,L=12,rho=0.5,
                      M=5,alpha=1,
                      delta_nb=5,
                      eta_prior_type = c(0,0,0,0,0,0),
                      eta_select = c(1,0,0,0,0,0),
                      nburnin = 50000,
                      nthin = 5,
                      nsample = 5000) {
  

  sim_dat = sim_pois_dglm(
    n=n,
    obs_dist = obs_dist, # {"nbinom","poisson"}
    link_func = link_func, # {"identity","exponential"}
    trans_func = trans_func, # {"koyama","solow","koyck"}
    gain_func = gain_func, # {"exponential","softplus","logistic"}
    err_dist = err_dist, # {"gaussian","laplace","cauchy","left_skewed_normal"}
    mu0=mu0, theta0 = theta0, psi0 = psi0,
    W=W, L=L, rho=rho,
    alpha = alpha,
    coef = c(1/M,0,M),
    delta_nb=delta_nb)

  lbe_out = lbe_poisson(
    sim_dat$y,
    sim_dat$params$model_code,
    rho=sim_dat$params$rho, 
    L=sim_dat$params$L,
    delta=sim_dat$params$delta_lbe,
    ctanh = sim_dat$params$ctanh,
    alpha = sim_dat$params$alpha,
    mu0=sim_dat$params$mu0,
    delta_nb=sim_dat$params$delta_nb,
    summarize_return=TRUE)

  mcs_out = mcs_poisson(
    sim_dat$y, 
    sim_dat$params$model_code,
    sim_dat$params$W, 
    L=sim_dat$params$L,
    rho=sim_dat$params$rho,
    mu0=sim_dat$params$mu0,
    B=length(sim_dat$y)*0.1,
    ctanh = sim_dat$params$ctanh,
    alpha = sim_dat$params$alpha,
    delta_nb=sim_dat$params$delta_nb,
    N=5000)
  

  aw = 1e3*sim_dat$params$W
  bw = 1e3
  amu = 1
  bmu = 1
  eta_prior_val = cbind(c(aw,bw),
                        c(amu,bmu),
                        c(0,0),
                        c(0,0))
  
  eta_init = c(sim_dat$params$W,
               sim_dat$params$mu0,
               sim_dat$params$rho,
               sim_dat$params$ctanh[3])
  
  
  hvb_out = hva_poisson(
    sim_dat$y,
    sim_dat$params$model_code,
    eta_select[1:4], eta_init,
    eta_prior_type[1:4], eta_prior_val,
    ctanh = sim_dat$params$ctanh,
    alpha = sim_dat$params$alpha,
    L = sim_dat$params$L,
    delta = sim_dat$params$delta_lbe,
    psi_init=mcs_out$psi[,2],
    scale_sd = 1e-1, # doesn't work for Koyama
    rtheta_type = 0,sampler_type = 1,
    delta_nb=sim_dat$params$delta_nb,
    nsample=nsample,nburnin=nburnin,nthin=nthin,
    verbose=FALSE,summarize_return=TRUE)
  
  eta_prior_val = cbind(c(sim_dat$params$W*1e3,1e3),
                        c(0,1),
                        c(1,1),
                        c(1,1),
                        c(0,1),
                        c(0,1))
  eta_init = c(sim_dat$params$W,
               sim_dat$params$mu0,
               sim_dat$params$rho,
               ctanh[3],
               sim_dat$params$theta0,
               sim_dat$params$psi0)
  MH_var = c(0.05,0.001)
  mcmc_out = mcmc_disturbance_pois(
    sim_dat$y,
    sim_dat$params$model_code,
    eta_select,
    eta_init,
    eta_prior_type,
    eta_prior_val,
    L=sim_dat$params$L,
    ht_=mcs_out$psi[,2],
    w1_prior=c(0,10),
    wt_init=diff(mcs_out$psi[,2]),
    MH_sd=sqrt(MH_var),
    delta_nb=sim_dat$params$delta_nb,
    nburnin=nburnin,nthin=nthin,nsample=nsample,
    summarize_return=TRUE, verbose=FALSE)
  
  p1 = plot_psi(sim_dat=sim_dat,
                psi_list=list(LBE=lbe_out$psi,
                              MCS=mcs_out$psi,
                              HVB=hvb_out$psi,
                              MCMC=mcmc_out$psi),
                plot_hpsi=TRUE,plot_lambda=TRUE)
  return(p1)
}




test_model_real = function(
    y,opts,
    eta_select=c(1,0,0,0,0,0),
    eta_prior_type=c(0,0,0,0,0,0),
    model_compare=c(1,0,1,0,0,0,0,0,0),
    theta0_upbnd = 2,
    Blag = 30,
    Blag_pct = 0.15,
    mcsN=5000,
    nburnin = 50000,
    nthin = 5,
    nsample = 5000,
    koyama2021 = NULL,
    ci_coverage = 0.95,
    npara = 20,
    use_discount = FALSE,
    smc_resample = FALSE,
    verbose=TRUE) {
  
  psi_list = list()
  psi_name = NULL
  eta_list = list()
  eta_name = NULL
  
  perf = matrix(NA,nrow=length(model_compare),ncol=3)
  colnames(perf) = c("RMSE","MAE","lik")
  rownames(perf) = c(
    "LBE","MCS","FFBS",
    "VB","HVB","MCMC",
    "Koyama","EpiEstim","WT"
  )
  
  if (is.null(koyama2021)) {
    model_compare[6] = 0
  }
  
  # Run LBE and MCS with mu0 and W set to their initial values
  if (model_compare[1]==1) {
    if (use_discount) {
      lbe_out = lbe_poisson(
        y,opts$model_code,
        L=opts$L,rho=opts$rho,
        mu0=opts$mu0,
        delta=opts$delta_lbe,
        W=NA,
        alpha=opts$alpha,
        ctanh=opts$ctanh,
        m0_prior=opts$m0,
        C0_prior=opts$C0,
        delta_nb=opts$delta_nb,
        ci_coverage=0.95,
        summarize_return=TRUE)
    } else {
      lbe_out = lbe_poisson(
        y,opts$model_code,
        L=opts$L,rho=opts$rho,
        mu0=opts$mu0,
        W=opts$W,
        delta=NA,
        alpha=opts$alpha,
        ctanh=opts$ctanh,
        m0_prior=opts$m0,
        C0_prior=opts$C0,
        delta_nb=opts$delta_nb,
        ci_coverage=0.95,
        summarize_return=TRUE)
    }
    
    psi_list = c(psi_list,list(lbe_out$psi))
    psi_name = c(psi_name,"LBE")
    eta_list = c(eta_list,list(lbe_out$Wt))
    eta_name = c(eta_name,"LBE")
    
    hpsi = psi2hpsi(lbe_out$psi,
                    opts$model_code[4],
                    coef=opts$ctanh)
    lambda = hpsi2theta(hpsi,y,
                        opts$model_code[3],
                        opts$theta0,
                        opts$alpha,
                        opts$L,opts$rho)
    lambda = lambda + opts$mu0
    lambda = lambda[(length(lambda)-length(y)+1):length(lambda)]
    perf[1,] = c(lbe_out$rmse,lbe_out$mae,lbe_out$logpred)
  }
  
  if (model_compare[2]==1) {
    if (use_discount) {
      mcs_out = mcs_poisson(
        y, opts$model_code, NA, 
        L=opts$L, rho=opts$rho,
        mu0=opts$mu0, 
        B=Blag,
        theta0_upbnd = theta0_upbnd,
        alpha=opts$alpha,
        ctanh = opts$ctanh,
        delta_nb=opts$delta_nb,
        delta_discount=opts$delta_smc,
        m0_prior=opts$m0,
        C0_prior=opts$C0,N=mcsN
      )
    } else {
      mcs_out = mcs_poisson(
        y, opts$model_code, opts$W, 
        L=opts$L, rho=opts$rho,
        mu0=opts$mu0, 
        B=Blag,
        theta0_upbnd = theta0_upbnd,
        alpha=opts$alpha,
        ctanh = opts$ctanh,
        delta_nb=opts$delta_nb,
        m0_prior=opts$m0,
        C0_prior=opts$C0,N=mcsN
        )
    }
    psi_list = c(psi_list,list(mcs_out$psi))
    psi_name = c(psi_name,"MCS")
    
    hpsi = psi2hpsi(mcs_out$psi,
                    opts$model_code[4],
                    coef=opts$ctanh)
    lambda = hpsi2theta(hpsi,y,
                        opts$model_code[3],
                        opts$theta0,
                        opts$alpha,
                        opts$L,opts$rho)
    lambda = lambda + opts$mu0
    lambda = lambda[(length(lambda)-length(y)+1):length(lambda)]
    perf[2,] = c(
      sqrt(mean((lambda-y)^2)),mean(abs(lambda-y))
    )
  }
  
  
  if (model_compare[3]==1) {
    if (use_discount) {
      ffbs_out = ffbs_poisson(
        y,opts$model_code,NA,
        rho=opts$rho, L=opts$L, 
        mu0=opts$mu0, N=mcsN,
        theta0_upbnd = theta0_upbnd,
        verbose=FALSE,
        smoothing=TRUE,
        resample=smc_resample,
        delta_nb = opts$delta_nb,
        delta_discount = opts$delta_smc,
        qProb = c(
          (1.-ci_coverage)/2,
          0.5,
          ci_coverage+(1.-ci_coverage)/2)
      )
    } else {
      ffbs_out = ffbs_poisson(
        y,opts$model_code,opts$W,
        rho=opts$rho, L=opts$L,
        mu0=opts$mu0, N=mcsN,
        theta0_upbnd = theta0_upbnd,
        verbose=FALSE,
        smoothing=TRUE,
        resample=smc_resample,
        delta_nb=opts$delta_nb,
        qProb = c(
          (1.-ci_coverage)/2,
          0.5,
          ci_coverage+(1.-ci_coverage)/2)
      )
    }
    psi_list = c(psi_list,list(ffbs_out$psi))
    psi_name = c(psi_name,"FFBS")
    perf[3,] = c(ffbs_out$rmse,ffbs_out$mae,ffbs_out$log_marg_lik)
  }
  
  
  eta_prior_val = cbind(opts$W_prior,
                        opts$mu0_prior,
                        c(0,0),
                        c(0,0))
  
  eta_init = c(opts$W,
               opts$mu0,
               ifelse("rho" %in% names(opts),opts$rho,0.7),
               ifelse(is.null(opts$ctanh),1,opts$ctanh[3]))
  
  if (model_compare[4]==1) {
    vb_out = vb_poisson(
      y,opts$model_code,
      eta_select,
      eta_init,
      eta_prior_type,
      eta_prior_val,
      L=opts$L,
      alpha=opts$alpha,
      m0=opts$m0,C0=opts$C0,
      ctanh=opts$ctanh,
      delta_nb=opts$delta_nb,
      use_smoothing = TRUE,
      nburnin=nburnin,nthin=nthin,nsample=nsample)
    psi_list = c(psi_list,list(vb_out$psi))
    psi_name = c(psi_name,"VB")
    eta_list = c(eta_list,list(vb_out$W))
    eta_name = c(eta_name,"VB")
  }
  
  if (model_compare[5]==1) {
    hvb_out = hva_poisson(
      y,opts$model_code,
      eta_select[1:4], 
      eta_init[1:4],
      eta_prior_type[1:4], 
      eta_prior_val[,1:4],
      L = opts$L,
      delta = opts$delta,
      m0_prior=opts$m0,C0_prior=opts$C0,
      ctanh=opts$ctanh,
      psi_init=mcs_out1$psi[,2],
      scale_sd = 1e-1, # doesn't work for Koyama
      rtheta_type = 0,sampler_type = 1,
      Blag_pct = Blag_pct,
      delta_nb=opts$delta_nb,
      nsample=nsample,nburnin=nburnin,nthin=nthin,
      summarize_return=TRUE,verbose=verbose)
    psi_list = c(psi_list,list(hvb_out$psi))
    psi_name = c(psi_name,"HVB")
    eta_list = c(eta_list,list(hvb_out$W))
    eta_name = c(eta_name,"HVB")
  }
  
  
  eta_prior_val = cbind(opts$W_prior,
                        opts$mu0_prior,
                        c(1,1),
                        c(1,1),
                        c(0,1),
                        c(0,1))
  eta_init = c(opts$W,
               opts$mu0,
               ifelse("rho" %in% names(opts),opts$rho,0.7),
               ifelse(is.null(opts$ctanh),1,opts$ctanh[3]),
               0.01,
               0.01)
  MH_var = c(0.5,0.001)
  if (model_compare[6]==1) {
    mcmc_out = mcmc_disturbance_pois(
      y,opts$model_code,
      eta_select,
      eta_init,
      eta_prior_type,
      eta_prior_val,
      L=opts$L,
      ht_=mcs_out1$psi[,2],
      w1_prior=c(0,1),
      wt_init=diff(mcs_out1$psi[,2]),
      MH_sd=sqrt(MH_var),
      delta_nb=opts$delta_nb,
      nburnin=nburnin,nthin=nthin,nsample=nsample,
      summarize_return=TRUE, verbose=verbose)
    psi_list = c(psi_list,list(mcmc_out$psi))
    psi_name = c(psi_name,"MCMC")
    eta_list = c(eta_list,list(mcmc_out$W))
    eta_name = c(eta_name,"MCMC")
  }
  
  
  # if (model_compare[6]==1 & is.null(koyama2021)) {
  #   koyama_out = hawke_ss2(y)
  #   colnames(koyama_out) = c("lobnd","est","hibnd")
  #   psi_list = c(psi_list,list(koyama_out))
  #   psi_name = c(psi_name,"Koyama2021")
  # } else if (!is.null(koyama2021)) {
  #   koyama_out = koyama2021
  #   colnames(koyama_out) = c("lobnd","est","hibnd")
  #   psi_list = c(psi_list,list(koyama_out))
  #   psi_name = c(psi_name,"Koyama2021")
  # }
  if (model_compare[7]==1 && !is.null(koyama2021)) {
    koyama_out = koyama2021
    colnames(koyama_out) = c("lobnd","est","hibnd")
    psi_list = c(psi_list,list(koyama_out))
    psi_name = c(psi_name,"Koyama2021")
  }
  
  if (model_compare[8]==1) {
    config = list(mean_si = 4.7, 
                  std_si = 2.9,
                  t_start = 2:(length(y)-6),
                  t_end = 8:length(y))
    epi_out = EpiEstim::estimate_R(y, method="parametric_si",
                                   config=EpiEstim::make_config(config))$R
    epi_out = epi_out[,c("Quantile.0.025(R)","Median(R)","Quantile.0.975(R)")]
    colnames(epi_out) = c("lobnd", "est", "hibnd")
    psi_list = c(psi_list,list(epi_out))
    psi_name = c(psi_name,"EpiEstim")
  } 
  
  if (model_compare[9]==1) {
    config = list(mean_si = 4.7, 
                  std_si = 2.9,
                  n_sim = 10,
                  t_start = 2:(length(y)-6),
                  t_end = 8:length(y))
    wt_out = EpiEstim::wallinga_teunis(y,method="parametric_si",
                                       config=config)$R
    wt_out = wt_out[,c("Quantile.0.025(R)","Mean(R)","Quantile.0.975(R)")]
    colnames(wt_out) = c("lobnd", "est", "hibnd")
    # wt_out$time = (1:dim(wt_out)[1]) + 7
    psi_list = c(psi_list,list(wt_out))
    psi_name = c(psi_name,"WT")
  }
  
  names(psi_list) = psi_name
  names(eta_list) = eta_name
  
  p1 = plot_psi(psi_list=psi_list, 
                ytrue=y, opts=opts,
                npara=npara,
                plot_hpsi=TRUE,
                plot_lambda=TRUE)
  
  # p3 = plot_eta("W",eta_list)
  # if (eta_select[2]==1) {
  #   p4 = plot_eta("mu0",eta_list)
  #   p3 = c(p3,p4)
  # } else {
  #   p3 = list(p3)
  # }
  
  return(list(plot=p1,perf=perf))
}