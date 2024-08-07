load_dat <- function(opath, seed, type = "sim", param_infer = "W") {
  env <- new.env(parent = baseenv())
  flab <- paste0(type, "-", seed)

  fname <- file.path(
    opath, "data",
    paste0(type, "-", seed, ".RData")
  )
  if (file.exists(fname)) {
    load(fname, envir = env)
  }


  fname <- file.path(
    opath, "data",
    paste0(type, "-", seed, "-lba.RData")
  )
  if (file.exists(fname)) {
    load(fname, envir = env)
  }

  fname <- file.path(
    opath, "data",
    paste0(type, "-", seed, "-mcs.RData")
  )
  if (file.exists(fname)) {
    load(fname, envir = env)
  }

  fname <- file.path(
    opath, "data",
    paste0(type, "-", seed, "-ffbs.RData")
  )
  if (file.exists(fname)) {
    load(fname, envir = env)
  }

  fname <- file.path(
    opath, "data",
    paste0(type, "-", seed, "-tfs.RData")
  )
  if (file.exists(fname)) {
    load(fname, envir = env)
  }


  fname <- file.path(
    opath, "data",
    paste0(type, "-", seed, "-tfs2.RData")
  )
  if (file.exists(fname)) {
    load(fname, envir = env)
  }


  # var_type = c("W", "rho", "mu0", "W-rho", "lag", "all-but-mu0", "all")

  if (all(param_infer == "W")) {
    fn <- paste0(flab, c("-pl-W.RData", "-pl.RData"))
    fname <- file.path(opath, "data", fn)
    fname <- fname[sapply(fname, file.exists)]
    if (length(fname) > 0) {
      load(fname[1], envir = env)
      print(fname[1])
    }
  } else if (all(param_infer == "rho")) {
    fname <- file.path(opath, "data", paste0(flab, "-pl-rho.RData"))
    if (file.exists(fname)) {
      load(fname, envir = env)
    }
  } else if (all(sort(param_infer) == c("rho", "W"))) {
    fname <- file.path(opath, "data", paste0(flab, "-pl-W-rho.RData"))
    if (file.exists(fname)) {
      load(fname, envir = env)
    }
  } else if (all(sort(param_infer) == c("par1", "par2"))) {
    fname <- file.path(opath, "data", paste0(flab, "-pl-lag.RData"))
    if (file.exists(fname)) {
      load(fname, envir = env)
    }
  } else if (all(sort(param_infer) == c("par1", "par2", "rho", "W"))) {
    fname <- file.path(opath, "data", paste0(flab, "-pl-all-but-mu0.RData"))
    if (file.exists(fname)) {
      load(fname, envir = env)
    }
  } else if (all(sort(param_infer) == c("mu0", "par1", "par2", "rho", "W"))) {
    fname <- file.path(opath, "data", paste0(flab, "-pl-all.RData"))
    if (file.exists(fname)) {
      load(fname, envir = env)
    }
  }
  # try({
  #   load(file.path(
  #     opath, "data",
  #     paste0(type, "-", seed, "-pl.RData")),
  #     envir = env
  #   )
  # })


  if (all(param_infer == "W")) {
    fn <- paste0(flab, c("-hva-W.RData", "-hva.RData"))
    fname <- file.path(opath, "data", fn)
    fname <- fname[sapply(fname, file.exists)]
    if (length(fname) > 0) {
      load(fname[1], envir = env)
      print(fname[1])
    }
  } else if (all(param_infer == "rho")) {
    fn <- paste0(flab, "-hva-rho.RData")
    fname <- file.path(opath, "data", fn)
    if (file.exists(fname)) {
      load(fname, envir = env)
    }
  } else if (all(sort(param_infer) == c("rho", "W"))) {
    fn <- paste0(flab, c("-hva-W-rho.RData", "-hva-rhow.RData", "-hva-wrho.RData"))
    fname <- file.path(opath, "data", fn)
    fname <- fname[sapply(fname, file.exists)]
    if (length(fname) > 0) {
      load(fname[1], envir = env)
    }
  } else if (all(sort(param_infer) == c("par1", "par2"))) {
    fname <- file.path(opath, "data", paste0(flab, "-hva-lag.RData"))
    if (file.exists(fname)) {
      load(fname, envir = env)
    }
  } else if (all(sort(param_infer) == c("par1", "par2", "rho", "W"))) {
    fname <- file.path(opath, "data", paste0(flab, "-hva-all-but-mu0.RData"))
    if (file.exists(fname)) {
      load(fname, envir = env)
    }
  } else if (all(sort(param_infer) == c("mu0", "par1", "par2", "rho", "W"))) {
    fname <- file.path(opath, "data", paste0(flab, "-hva-all.RData"))
    if (file.exists(fname)) {
      load(fname, envir = env)
    }
  }

  # try({
  #   load(file.path(
  #     opath, "data",
  #     paste0(type, "-", seed, "-hva.RData")),
  #     envir = env
  #   )
  # })


  if (all(param_infer == "W")) {
    fn <- paste0(flab, c("-mcmc-W.RData", "-mcmc.RData"))
    fname <- file.path(opath, "data", fn)
    fname <- fname[sapply(fname, file.exists)]
    if (length(fname) > 0) {
      load(fname[1], envir = env)
      print(fname[1])
    }
  } else if (all(param_infer == "rho")) {
    fname <- file.path(opath, "data", paste0(flab, "-mcmc-rho.RData"))
    if (file.exists(fname)) {
      load(fname, envir = env)
    }
  } else if (all(sort(param_infer) == c("rho", "W"))) {
    fname <- file.path(opath, "data", paste0(flab, "-mcmc-W-rho.RData"))
    if (file.exists(fname)) {
      load(fname, envir = env)
    }
  } else if (all(sort(param_infer) == c("par1", "par2"))) {
    fname <- file.path(opath, "data", paste0(flab, "-mcmc-lag.RData"))
    if (file.exists(fname)) {
      load(fname, envir = env)
    }
  } else if (all(sort(param_infer) == c("par1", "par2", "rho", "W"))) {
    fname <- file.path(opath, "data", paste0(flab, "-mcmc-all-but-mu0.RData"))
    if (file.exists(fname)) {
      load(fname, envir = env)
    }
  } else if (all(sort(param_infer) == c("mu0", "par1", "par2", "rho", "W"))) {
    fname <- file.path(opath, "data", paste0(flab, "-mcmc-all.RData"))
    if (file.exists(fname)) {
      load(fname, envir = env)
    }
  }

  # try({
  #   load(file.path(
  #     opath, "data",
  #     paste0(type, "-", seed, "-mcmc.RData")),
  #     envir = env
  #   )
  # })

  try({
    load(
      file.path(
        opath, "data",
        paste0(type, "-", seed, "-epi.RData")
      ),
      envir = env
    )
  })

  try({
    load(
      file.path(
        opath, "data",
        paste0(type, "-", seed, "-wt.RData")
      ),
      envir = env
    )
  })

  return(env)
}


load_real <- function(dpath, seed, envir) {
  data <- read.csv(file.path(dpath, "county", "covid19cases_test.csv"))
  data <- data[data$area == seed, ]
  data$date <- as.Date(data$date)
  data <- data[data$date >= "2020-03-01", ]
  data <- data[1:(nrow(data) - 1), ]

  y <- data$cases
  n <- length(y)

  data2 <- data[data$date >= "2020-07-01" & data$date < "2021-12-01", ]
  y2 <- c(0, data2$cases)


  component <- list(
    obs_dist = "nbinom",
    link_func = "identity",
    trans_func = "sliding",
    gain_func = "softplus",
    lag_dist = "lognorm",
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


load_sim <- function(seed, envir) {
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

  if (component$lag_dist == "lognorm") {
    nlag <- round(qlnorm(0.99, param$lag[1], sqrt(param$lag[2])) + 0.5)
  } else {
    stop("nlag of lag distribution.")
  }
  dim <- list(
    nlag = nlag,
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
