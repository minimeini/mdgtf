epi_poisson = function(y, mean_si, sd2_si) {
  require(EpiEstim)
  ntime = length(c(y))
  config = list(mean_si = mean_si, 
                std_si = sqrt(sd2_si),
                t_start = 2:(ntime - 6),
                t_end = 8:ntime)
  epi_out = EpiEstim::estimate_R(y, method="parametric_si",
                                 config=EpiEstim::make_config(config))$R
  epi_out = epi_out[,c("Quantile.0.025(R)","Median(R)","Quantile.0.975(R)")]
  colnames(epi_out) = c("lobnd", "est", "hibnd")
  epi_out = as.matrix(epi_out)
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
  wt_out = EpiEstim::wallinga_teunis(y,method="parametric_si",
                                     config=config)$R
  wt_out = wt_out[,c("Quantile.0.025(R)","Mean(R)","Quantile.0.975(R)")]
  colnames(wt_out) = c("lobnd", "est", "hibnd")
  wt_out = as.matrix(wt_out)
  return(wt_out)
}



external_forecast = function(
  model, 
  param_si,
  y, # c(y[1], ..., y[ntime])
  algo = "EpiEstim", 
  loss_func = "quadratic", 
  k = 1, 
  time_start = 0,
  nsample = 5000,
  stop_on_error = FALSE) {
  
  ntime = length(c(y)) - 1
  tstart = max(c(model$dim$nlag, k, time_start)) + 1

  ycast = array(0, dim = c(ntime + 1, nsample, k))
  y_err_cast = array(0, dim = c(ntime + 1, nsample, k))

  pb = progress::progress_bar$new(total = (ntime + 1))
  pb$tick(0)

  for (t in 1:(ntime + 1)) {
    if ( (t < tstart) | t > (ntime - k) ) {
      idxs = t + 1
      idxe = min(c(t + k, ntime))
      nelem = idxe - idxs + 1;
      ytmp = y[idxs:idxe] # nelem x 1
      
      for (i in 1:nsample) {
        try({
          ycast[t, i, 1:nelem] = ytmp
        }, silent = !stop_on_error)
        
      }

    } else {
      try({
        if (algo == "EpiEstim" | algo == "epi") {
          Rt_out = epi_poisson(y[1:t], param_si[1], param_si[2]) # t x 3
        } else {
          Rt_out = wt_poisson(y[1:t], param_si[1], param_si[2])
        }
      }, silent = !stop_on_error)
      

      ytmp = y[1:(t + nelem)] # t x 1
     
      psi_out = log(exp(Rt_out) - 1) # t x 3
      psi_mean = c(psi_out[, 2], rep(0, nelem)) # (t + k) x 1
      psi_sd = c(abs(psi_out[, 3] - psi_out[, 2]) / qnorm(0.975), rep(0, nelem)) # (t + k) x 1

      psi_mean = c(rep(0, length(ytmp) - length(c(psi_mean))), psi_mean)
      psi_sd = c(rep(0, length(ytmp) - length(c(psi_sd))), psi_sd)

      psi = t(apply(cbind(psi_mean, psi_sd), 1, function(para, nsample){return(para[1] + para[2] * rnorm(nsample))}, nsample)) # t x nsample

      model$dim$ntime = length(ytmp) - 1
      try({
        ycast[t, , 1:nelem] = dgtf_forecast(model, ytmp, psi, loss_func = loss_func, k = nelem)$y_cast_all[t, , ]
      }, silent = !stop_on_error)
      
      for (j in 1:k) {
        if (t + j <= (ntime + 1)) {
          try({
            y_err_cast[t, , j] = abs(y[t + j] - ycast[t, , j])
          }, silent = !stop_on_error)
        }
      }

    }

    pb$tick()

  }

  yqt = apply(ycast, c(1, 3), quantile, c(0.025, 0.5, 0.975))
  if (loss_func == "absolute" | loss_func == "L1") {
    y_loss = apply(y_err_cast, c(1, 3), mean) # (ntime + 1) x k
    y_loss_all = apply(y_loss, 2, mean)
  } else {
    y_loss = apply(y_err_cast^2, c(1, 3), mean) # (ntime + 1) x k
    y_loss_all = apply(y_loss, 2, mean)

    y_loss = sqrt(y_loss)
    y_loss_all = sqrt(y_loss_all)
  }
  
  return(list(
    y_cast = yqt,
    y_cast_all = ycast,
    y = c(y),
    y_loss = y_loss,
    y_loss_all = c(y_loss_all)
  ))
}
