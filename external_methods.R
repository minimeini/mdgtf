epi_poisson = function(y, mean_si, sd2_si) {
  require(EpiEstim)
  ntime = length(c(y))
  ydat = data.frame(I = c(y))

  config = list(
    mean_si = mean_si, 
    std_si = sqrt(sd2_si),
    t_start = 2:(ntime - 6),
    t_end = 8:ntime
  )
  
  epi_out = EpiEstim::estimate_R(
    ydat, method="parametric_si",
    config=EpiEstim::make_config(config))$R
  epi_out = epi_out[,c("Quantile.0.025(R)","Median(R)","Quantile.0.975(R)")]
  colnames(epi_out) = c("lobnd", "est", "hibnd")
  epi_out = as.matrix(epi_out)
  return(epi_out)
}


epi_poisson_data <- function(y, mean_si, sd2_si) {
  require(EpiEstim)
  ntime <- length(c(y))
  ydat = data.frame(I = c(y))

  config <- list(
    mean_si = mean_si, std_mean_si = 2,
    min_mean_si = 1,
    max_mean_si = 10,
    std_si = sqrt(sd2_si), std_std_si = 1,
    min_std_si = 1,
    max_std_si =  5,
    t_start = 2:(ntime - 6),
    t_end = 8:ntime,
    n1 = 500,
    n2 = 500
  )

  # config <- list(
  #   si_distr = diff(plnorm(0:30, 1.386262, 0.3226017)),
  #   t_start = 2:(ntime - 6),
  #   t_end = 8:ntime
  # )
  
  epi_out <- EpiEstim::estimate_R(ydat,
    method = "uncertain_si",
    config = EpiEstim::make_config(config)
  )$R
  epi_out <- epi_out[, c("Quantile.0.025(R)", "Median(R)", "Quantile.0.975(R)")]
  colnames(epi_out) <- c("lobnd", "est", "hibnd")
  epi_out <- as.matrix(epi_out)
  return(epi_out)
}


wt_poisson = function(y, mean_si, sd2_si) {
  require(EpiEstim)
  ntime = length(c(y))
  config = list(mean_si = mean_si, 
                std_si = sqrt(sd2_si),
                n_sim = 100,
                t_start = 2:(ntime - 6),
                t_end = 8:ntime)
  wt_out = EpiEstim::wallinga_teunis(y, method="parametric_si", config=config)$R
  wt_out = wt_out[,c("Quantile.0.025(R)","Mean(R)","Quantile.0.975(R)")]
  colnames(wt_out) = c("lobnd", "est", "hibnd")
  wt_out = as.matrix(wt_out)
  return(wt_out)
}



external_forecast = function(
  model, 
  param_si,
  y, # c(y[1], ..., y[ntime])
  W_samples, # nsamples x 1
  mu0 = 0,
  algo = "EpiEstim", 
  loss_func = "quadratic", 
  k = 1, 
  time_start = 0,
  nsample = 100,
  stop_on_error = FALSE) {
  
  stopifnot(exists("dgtf_forecast"))

  y = c(y)
  ntime = length(y) - 1
  tstart = max(c(model$dim$nlag, k, time_start)) + 1

  if (length(W_samples) < nsample) {
    Wlast = rev(W_samples)[1]
    W_samples = c(W_samples, rep(Wlast, nsample - length(W_samples)))
  }
  W_samples = W_samples[1:nsample]

  y_loss = array(0, dim = c(ntime + 1, k))
  y_covered = y_loss
  y_width = y_loss

  pb = progress::progress_bar$new(total = (ntime + 1))
  pb$tick(0)

  for (t in tstart:(ntime - k)) {
    ypast = y[1:t]
    model$dim$ntime = length(ypast) - 1

    ycast_true = y[c((t + 1) : (t + k))]

    psi_samples = NULL
    try(
      {
        if (algo == "EpiEstim" | algo == "epi" | algo == "epi_parametric_si") {
          Rt_out = epi_poisson(ypast, param_si[1], param_si[2]) # t x 3
        } else if (algo == "EpiEstim2" | algo == "epi2" | algo == "epi_uncertain_si") {
          Rt_out = epi_poisson_data(ypast, param_si[1], param_si[2]) # t x 3
        } else {
          Rt_out = wt_poisson(ypast, param_si[1], param_si[2])
        }

        psi_out = log(exp(Rt_out) - 1) # t x 3
        psi_mean = c(psi_out[, 2])
        psi_sd = c(abs(psi_out[, 3] - psi_out[, 2]) / qnorm(0.975))

        psi_mean = c(rep(0, length(ypast) - length(c(psi_mean))), psi_mean)
        psi_sd = c(rep(0, length(ypast) - length(c(psi_sd))), psi_sd)
        psi_stat = cbind(psi_mean, psi_sd) # t x 2

        psi_samples = t(apply(psi_stat, 1, function(para, nsample) {
          return(para[1] + para[2] * rnorm(nsample))
        }, nsample)) # t x nsample
      },
      silent = !stop_on_error
    )

    
    try(
      {
        dtmp = dgtf_forecast(
          model, ypast, psi_samples, W_samples, 
          mu0, ycast_true,
          loss_func = loss_func, k = k, verbose = FALSE
        ) 

        y_loss[t, ] = dtmp$y_loss # k x 1
        y_covered[t, ] = dtmp$y_covered
        y_width[t, ] = dtmp$y_width
      },
      silent = !stop_on_error
    )

    pb$tick()

  } # loop over time

  y_covered_all = apply(y_covered[c(tstart:(ntime - k)), ], 2, mean) * 100
  y_width_all = apply(y_width[c(tstart:(ntime - k)), ], 2, mean)


  if (loss_func == "absolute" | loss_func == "L1") {
    y_loss_all = apply(y_loss[c(tstart : (ntime - k)), ], 2, mean)
  } else {
    y_loss = y_loss^2
    y_loss_all = apply(y_loss[c(tstart:(ntime - k)), ], 2, mean)
    y_loss_all = sqrt(y_loss_all)
  }
  
  return(list(
    y_loss_all = c(y_loss_all),
    y_covered_all = c(y_covered_all),
    y_width_all = c(y_width_all)
  ))
}
