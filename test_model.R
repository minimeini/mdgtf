test_model = function(ModelCode,M,alpha,psi0,
                      n=200,W=0.01,L=12,rho=0.8,
                      mu0=1,theta0=NULL,delta_nb=5,
                      err_type = 0,
                      eta_prior_type = c(0,0,0,0),
                      eta_select = c(1,0,0,0)) {
  
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
                          coef = c(0.3,0,M),
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
                        m0=m0,C0=C0,
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
  p2 = plot_W(sim_dat,hvb_out)
  return(c(p1,p2))
}