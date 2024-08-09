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

  load_dat1 = function(algo, flab, envir, opath) {
    fname <- file.path(
      opath, "data",
      paste0(flab, "-", algo, ".RData")
    )
    if (file.exists(fname)) {
      load(fname, envir = envir)
    } else {
      warning(paste0("File ", fname, " does not exist."))
    }

    return(envir)
  }

  env = load_dat1("lba", flab, env, opath)
  env = load_dat1("mcs", flab, env, opath)
  env = load_dat1("ffbs", flab, env, opath)
  env = load_dat1("tfs2", flab, env, opath)
  env = load_dat1("wt", flab, env, opath)
  env = load_dat1("epi", flab, env, opath)


  load_dat0 = function(algo, param_infer, flab, envir, opath) {
    fn = paste0(flab, "-", algo, "-", paste(sort(param_infer), collapse = ""), ".RData")
    fname <- file.path(opath, "data", fn)
    if (file.exists(fname)) {
      load(fname[1], envir = envir)
    } else {
      warning(paste0("File ", fname, " does not exist."))
    }

    return(envir)
  }

  env = load_dat0("pl", param_infer, flab, env, opath)
  env = load_dat0("hva", param_infer, flab, env, opath)
  env = load_dat0("mcmc", param_infer, flab, env, opath)

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
