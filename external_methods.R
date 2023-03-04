epi_poisson = function(y) {
  require(EpiEstim)
  config = list(mean_si = 4.7, 
                std_si = 2.9,
                t_start = 2:(length(y)-6),
                t_end = 8:length(y))
  epi_out = EpiEstim::estimate_R(y, method="parametric_si",
                                 config=EpiEstim::make_config(config))$R
  epi_out = epi_out[,c("Quantile.0.025(R)","Median(R)","Quantile.0.975(R)")]
  colnames(epi_out) = c("lobnd", "est", "hibnd")
  epi_out = as.matrix(epi_out)
  return(epi_out)
}


wt_poisson = function(y) {
  require(EpiEstim)
  config = list(mean_si = 4.7, 
                std_si = 2.9,
                n_sim = 10,
                t_start = 2:(length(y)-6),
                t_end = 8:length(y))
  wt_out = EpiEstim::wallinga_teunis(y,method="parametric_si",
                                     config=config)$R
  wt_out = wt_out[,c("Quantile.0.025(R)","Mean(R)","Quantile.0.975(R)")]
  colnames(wt_out) = c("lobnd", "est", "hibnd")
  wt_out = as.matrix(wt_out)
  return(wt_out)
}