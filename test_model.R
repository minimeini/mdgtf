default_opts = function(n=NA,ModelCode=NA){
  opts = list(n = n,
              ModelCode = ModelCode,
              L = 30,
              rho = 0.7,
              mu0 = 0.01,
              psi0 = 0,
              theta0 = 0,
              W = 0.01,
              delta = 0.99,
              delta_nb = 10,
              m0 = NULL,
              C0 = NULL,
              W_prior = NULL,
              mu0_prior = NULL,
              alpha = 1,
              ctanh = c(0.3,0,4),
              obs_type = 0)
  return(opts)
}


test_model = function(ModelCode,M=1,alpha=1,psi0=0,
                      n=200,W=0.01,L=12,rho=0.8,
                      mu0=1,theta0=NULL,delta_nb=5,
                      err_type = 0,
                      eta_prior_type = c(0,0,0,0,0,0),
                      eta_select = c(1,0,0,0,0,0),
                      nburnin = 50000,
                      nthin = 5,
                      nsample = 5000) {
  
  if (ModelCode %in% c(0,1,6,11,14,17)) {
    TransferCode = 1 # Koyama
    m0 = rep(0,L)
    C0 = diag(rep(10,L))
  } else if (ModelCode %in% c(2,3,7,12,15,18)) {
    TransferCode = 2 # Solow
    m0 = rep(0,3)
    C0 = diag(rep(10,3))
  } else {
    stop("Only support Koyama or Solow family")
  }
  
  sim_dat = sim_pois_dglm(n=n,
                          ModelCode=ModelCode,
                          mu0=mu0,
                          theta0 = theta0,
                          psi0 = psi0,
                          W=W,L=L,
                          obs_type=0, 
                          err_type = err_type,
                          alpha = alpha,
                          coef = c(0.1,0,M),
                          delta_nb=delta_nb)
  is_bounded = ifelse(ModelCode %in% c(13,14,15,16,17,18),TRUE,FALSE)
  if (is_bounded) {
    ctanh = sim_dat$params$ctanh
  } else {
    ctanh = c(0.3,0,1)
  }

  lbe_out = lbe_poisson(sim_dat$y,
                        sim_dat$params$ModelCode,
                        rho=sim_dat$params$rho, 
                        delta=sim_dat$params$delta_lbe,
                        ctanh = ctanh,
                        alpha = sim_dat$params$alpha,
                        L=sim_dat$params$L,
                        mu0=sim_dat$params$mu0,
                        obs_type=sim_dat$params$obs_type, 
                        delta_nb=sim_dat$params$delta_nb,
                        m0_prior=m0,C0_prior=C0)

  mcs_out = mcs_poisson(sim_dat$y, 
                        sim_dat$params$ModelCode,
                        sim_dat$params$W, 
                        L=sim_dat$params$L,
                        rho=sim_dat$params$rho,
                        mu0=sim_dat$params$mu0,
                        B=sim_dat$params$L,
                        ctanh = ctanh,
                        alpha = sim_dat$params$alpha,
                        obstype=sim_dat$params$obs_type,
                        delta_nb=sim_dat$params$delta_nb,
                        m0_prior=m0,C0_prior=C0,N=5000)
  

  aw = 1e4*sim_dat$params$W
  bw = 1e4
  amu = 1
  bmu = 1
  eta_prior_val = cbind(c(aw,bw),
                        c(amu,bmu),
                        c(0,0),
                        c(0,0))
  
  eta_init = c(sim_dat$params$W,
               sim_dat$params$mu0,
               sim_dat$params$rho,
               1)
  if (is_bounded) {eta_init[4]=sim_dat$params$ctanh[3]}
  
  hvb_out = hva_poisson(sim_dat$y,
                        sim_dat$params$ModelCode,
                        eta_select, eta_init,
                        eta_prior_type, eta_prior_val,
                        ctanh = ctanh,
                        alpha = sim_dat$params$alpha,
                        L = sim_dat$params$L,
                        delta = sim_dat$params$delta_lbe,
                        m0_prior=m0,C0_prior=C0,
                        psi_init=mcs_out$quantiles[,2],
                        scale_sd = 1e-1, # doesn't work for Koyama
                        rtheta_type = 0,sampler_type = 1,
                        obs_type = sim_dat$params$obs_type,
                        delta_nb=sim_dat$params$delta_nb,
                        niter=5000,verbose=FALSE)
  
  p1 = plot_results(sim_dat=sim_dat,
                    mcs_output=mcs_out,
                    lbe_output=lbe_out,
                    hvb_output=hvb_out,
                    plot_hpsi=TRUE)
  p2 = plot_eta("W",hvb_out,nburnin=2000)
  return(c(p1,p2))
}




test_model_real = function(y,opts,
                           eta_select=c(1,0,0,0,0,0),
                           eta_prior_type=c(0,0,0,0,0,0),
                           model_compare=c(1,1,1),
                           mcsN=5000,
                           nburnin = 50000,
                           nthin = 5,
                           nsample = 5000) {

  # Run LBE and MCS with mu0 and W set to their initial values
  lbe_out1 = lbe_poisson(y,opts$ModelCode,
                        L=opts$L,rho=opts$rho,
                        mu0=opts$mu0,
                        delta=opts$delta,
                        alpha=opts$alpha,
                        ctanh=opts$ctanh,
                        m0_prior=opts$m0,C0_prior=opts$C0,
                        obs_type=opts$obs_type, 
                        delta_nb=opts$delta_nb,
                        summarize_return=TRUE)
  
  mcs_out1 = mcs_poisson(y, opts$ModelCode, opts$W, 
                        L=opts$L, rho=opts$rho,
                        mu0=opts$mu0, B=opts$L,
                        alpha=opts$alpha,
                        ctanh = opts$ctanh,
                        obstype=opts$obs_type,
                        delta_nb=opts$delta_nb,
                        m0_prior=opts$m0,C0_prior=opts$C0,N=mcsN)
  
  eta_prior_val = cbind(opts$W_prior,
                        opts$mu0_prior,
                        c(0,0),
                        c(0,0))
  
  eta_init = c(opts$W,
               opts$mu0,
               ifelse("rho" %in% names(opts),opts$rho,0.7),
               ifelse(is.null(opts$ctanh),1,opts$ctanh[3]))
  
  hvb_out = hva_poisson(y,opts$ModelCode,
                        eta_select[1:4], 
                        eta_init[1:4],
                        eta_prior_type[1:4], 
                        eta_prior_val[,1:4],
                        L = opts$L,
                        delta = opts$delta,
                        m0_prior=opts$m0,C0_prior=opts$C0,
                        ctanh=opts$ctanh,
                        psi_init=lbe_out1$ht[,1],
                        scale_sd = 1e-1, # doesn't work for Koyama
                        rtheta_type = 0,sampler_type = 1,
                        obs_type = 0,
                        delta_nb=opts$delta_nb,
                        nsample=nsample,nburnin=nburnin,nthin=nthin,
                        summarize_return=TRUE,verbose=FALSE)
  
  if (eta_select[2]==1) {
    mu0_hat = mean(hvb_out$mu0_stored[-c(1:hvbBurnin)])
  } else {
    mu0_hat = opts$mu0
  }
  
  W_hat = mean(hvb_out$W_stored[-c(1:hvbBurnin)])
  
  mcs_out2 = mcs_poisson(y, opts$ModelCode, 
                         W_hat, mu0=mu0_hat, 
                         L=opts$L, rho=opts$rho,
                         B=opts$L,
                         ctanh = opts$ctanh,
                         obstype=opts$obs_type,
                         delta_nb=opts$delta_nb,
                         m0_prior=opts$m0,C0_prior=opts$C0,N=mcsN)
  
  lbe_out2 = lbe_poisson(y,opts$ModelCode,
                         L=opts$L,rho=opts$rho,
                         mu0=mu0_hat,
                         delta=opts$delta,
                         ctanh=opts$ctanh,
                         m0_prior=opts$m0,C0_prior=opts$C0,
                         obs_type=opts$obs_type, 
                         delta_nb=opts$delta_nb,
                         summarize_return=TRUE)
  
  if (model_compare[1]==1) {
    koyama_out = hawke_ss2(y)
    colnames(koyama_out) = c("lobnd","Koyama2021","hibnd")
    koyama_out = as.data.frame(koyama_out)
    koyama_out$time = (1:dim(koyama_out)[1])+1
  } else {
    koyama_out = NULL
  }
  if (model_compare[2]==1) {
    config = list(mean_si = 4.7, 
                  std_si = 2.9,
                  t_start = 2:(length(y)-6),
                  t_end = 8:length(y))
    epi_out = estimate_R(y, method="parametric_si", 
                         config=make_config(config))$R
    epi_out = epi_out[,c("Quantile.0.025(R)","Median(R)","Quantile.0.975(R)")]
    colnames(epi_out) = c("lobnd", "EpiEstim", "hibnd")
    epi_out$time = (1:dim(epi_out)[1]) + 7
  } else {
    epi_out = NULL
  }
  if (model_compare[3]==1) {
    config = list(mean_si = 4.7, 
                  std_si = 2.9,
                  n_sim = 10,
                  t_start = 2:(length(y)-6),
                  t_end = 8:length(y))
    wt_out = wallinga_teunis(y,method="parametric_si",
                             config=config)$R
    wt_out = wt_out[,c("Quantile.0.025(R)","Mean(R)","Quantile.0.975(R)")]
    colnames(wt_out) = c("lobnd", "WT", "hibnd")
    wt_out$time = (1:dim(wt_out)[1]) + 7
  } else {
    wt_out = NULL
  }
  
  
  
  p1 = plot_results(lbe_output=lbe_out1,
                    mcs_output=mcs_out1,
                    hvb_output=hvb_out,
                    epi_output=epi_out,
                    wt_output=wt_out,
                    koyama_output=koyama_out,
                    ytrue=y, opts=opts,
                    plot_hpsi=TRUE,
                    plot_lambda=TRUE,
                    plot_coverage=FALSE,
                    nburnin=hvbBurnin)
  
  opts$mu0 = mu0_hat
  p2 = plot_results(lbe_output=lbe_out2,
                    mcs_output=mcs_out2,
                    hvb_output=hvb_out,
                    epi_output=epi_out,
                    wt_output=wt_out,
                    koyama_output=koyama_out,
                    ytrue=y, opts=opts,
                    plot_hpsi=TRUE,
                    plot_lambda=TRUE,
                    plot_coverage=FALSE,
                    nburnin=hvbBurnin)
  
  p3 = plot_eta("W",hvb_out,nburnin=hvbBurnin)
  if (eta_select[2]==1) {
    p4 = plot_eta("mu0",hvb_out,nburnin=hvbBurnin)
    p3 = c(p3,p4)
  } else {
    p3 = list(p3)
  }
  
  
  return(c(p1,p2,p3))
}