load_dat = function(opath, seed, type = "sim") {
  env = new.env(parent = baseenv())
  try({
    load(file.path(
      opath, "data", 
      paste0(type, "-", seed, ".RData")),
      envir = env
    )
  })
  
  try({
    load(file.path(
      opath, "data", 
      paste0(type, "-", seed, "-lba.RData")),
      envir = env
    )
  })
  
  try({
    load(file.path(
      opath, "data", 
      paste0(type, "-", seed, "-mcs.RData")),
      envir = env
    )
  })
  
  try({
    load(file.path(
      opath, "data", 
      paste0(type, "-", seed, "-ffbs.RData")),
      envir = env
    )
  })

  try({
    load(
      file.path(
        opath, "data",
        paste0(type, "-", seed, "-tfs.RData")
      ),
      envir = env
    )
  })

  try({
    load(
      file.path(
        opath, "data",
        paste0(type, "-", seed, "-tfs2.RData")
      ),
      envir = env
    )
  })
  
  try({
    load(file.path(
      opath, "data", 
      paste0(type, "-", seed, "-pl.RData")),
      envir = env
    )
  })
  
  try({
    load(file.path(
      opath, "data", 
      paste0(type, "-", seed, "-hva.RData")),
      envir = env
    )
  })
  
  try({
    load(file.path(
      opath, "data", 
      paste0(type, "-", seed, "-mcmc.RData")),
      envir = env
    )
  })
  
  try({
    load(file.path(
      opath, "data", 
      paste0(type, "-", seed, "-epi.RData")),
      envir = env
    )
  })
  
  try({
    load(file.path(
      opath, "data", 
      paste0(type, "-", seed, "-wt.RData")),
      envir = env
    )
  })
  
  return(env)
}


load_real = function(dpath, seed, envir) {
  data = read.csv(file.path(dpath,"county","covid19cases_test.csv"))
  data = data[data$area == seed,]
  data$date = as.Date(data$date)
  data = data[data$date >= "2020-03-01",]
  data = data[1:(nrow(data)-1),]

  y = data$cases
  n = length(y)

  data2 = data[data$date >= "2020-07-01" & data$date < "2021-12-01",]
  y2 = c(0, data2$cases)
  
  
  component <- list(
    obs_dist = "nbinom",
    link_func = "identity",
    trans_func = "sliding",
    gain_func = "softplus",
    lag_dist = "nbinom",
    err_dist = "gaussian"
  )
  param <- list(
    obs = c(0, 5),
    lag = c(0.4, 6), # mode = 2.8969
    err = c(0.01, 0)
  )
  dim <- list(
    nlag = 16,
    ntime = length(y2) - 1,
    truncated = TRUE
  )

  mod <- list(
    model = component,
    param = param,
    dim = dim
  )

  rm(dim)
  rm(param)
  rm(component)

  assign("mod", mod, envir = envir)
  assign("y2", y2, envir = envir)
  
  return(envir)
}


load_sim = function(seed, envir) {
  component <- list(
    obs_dist = "nbinom",
    link_func = "identity",
    trans_func = "sliding",
    gain_func = "softplus",
    lag_dist = "lognorm",
    err_dist = "gaussian"
  )
  param <- list(
    obs = c(1, 30),
    lag = c(1.386262, 0.3226017), # mode = 2.8969
    err = c(0.01, 0)
  )
  dim <- list(
    nlag = 16,
    ntime = 200,
    truncated = TRUE
  )

  mod1 <- list(
    model = component,
    param = param,
    dim = dim
  )

  rm(dim)
  rm(param)
  rm(component)


  set.seed(seed)
  sim1 <- dgtf_simulate(mod1)
  
  assign("sim1", sim1, envir = envir)
  assign("mod1", mod1, envir = envir)
  return(envir)
}