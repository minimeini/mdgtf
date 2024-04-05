#!/usr/bin/env Rscript

# data_type <- "sim"
# tag <- 3269

args <- commandArgs(trailingOnly = TRUE)
data_type <- args[1]
tag <- args[2]

dat <- read.csv("/Users/meinitang/Downloads/prior-sensitivity-mu0-pl-sim-3269.csv")


dat$Data.Type <- data_type
dat$Tag <- tag

save_data <- FALSE
save_figures <- FALSE
plot_figures <- FALSE

kstep_forecast_err <- 10
eval_forecast_err <- TRUE
eval_fitted_err <- FALSE
loss <- "quadratic"

library(ggplot2)
library(gridExtra)

repo <- "/Users/meinitang/Dropbox/Repository/poisson-dlm"
Rcpp::sourceCpp(file.path(repo, "export.cpp"))
source(file.path(repo, "plot_timeseries_creditble_interval.r"))
source(file.path(repo, "external_methods.r"))

if (data_type == "sim") {
    opath <- "/Users/meinitang/Dropbox/Research/Project1/simulation"

    nlag <- 200
    if (as.numeric(tag) == 8434) {
        lag_dist <- "nbinom"
        lag_param <- c(0.4, 6)
        obs_param <- c(1, 5)
    } else {
        lag_dist <- "lognorm"
        lag_param <- c(1.386262, 0.3226017)
        obs_param <- c(1, 30)
    }
} else {
    opath <- "/Users/meinitang/Dropbox/Research/Project1/real"
    dpath <- "/Users/meinitang/Dropbox/Research/Project1"

    data <- read.csv(file.path(dpath, "county", "covid19cases_test.csv"))

    if (tag == "santacruz") {
        Tag2 <- "Santa Cruz"
    } else if (tag == "santaclara") {
        Tag2 <- "Santa Clara"
    } else if (tag == "monterey") {
        Tag2 <- "Monterey"
    }

    data <- data[data$area == Tag2, ]
    data$date <- as.Date(data$date)
    data <- data[data$date >= "2020-03-01", ]
    data <- data[1:(nrow(data) - 1), ]

    y <- data$cases
    n <- length(y)

    data2 <- data[data$date >= "2020-07-01" & data$date < "2021-12-01", ]
    y2 <- c(0, data2$cases)

    data <- list(y = c(y2))

    lag_dist <- "nbinom"
    lag_param <- c(0.4, 6)


    if (tag == "monterey") {
        obs_param <- c(0, 5)
    } else {
        obs_param <- c(0, 30)
    }

    nlag <- length(data$y) - 1
}

component <- list(
    obs_dist = "nbinom",
    link_func = "identity",
    trans_func = "sliding",
    gain_func = "softplus",
    lag_dist = lag_dist,
    err_dist = "gaussian"
)
param <- list(
    obs = obs_param,
    lag = lag_param, # mode = 2.8969
    err = c(0.01, 0)
)


dim <- list(
    nlag = 16,
    ntime = nlag,
    truncated = TRUE,
    regressor_baseline = FALSE
)

mod <- list(
    model = component,
    param = param,
    dim = dim
)

rm(dim)
rm(param)
rm(component)

if (data_type == "sim") {
    set.seed(as.numeric(tag))
    data <- dgtf_simulate(mod)
}


niter <- dim(dat)[1]

for (i in 1:niter) {
    print(paste0("i = ", i))

    opts.pl <- dgtf_default_algo_settings("pl")
    opts.pl$num_particle <- 10000
    opts.pl$num_smooth <- 500
    opts.pl$num_step_ahead_forecast <- 0
    opts.pl$do_smoothing <- TRUE
    opts.pl$max_iter <- 100

    opts.pl$mu0 <- list(
        init = mod$param$obs[1],
        infer = TRUE,
        prior_name = "normal",
        prior_param = c(dat$Prior.Par1[i], dat$Prior.Par2[i])
    )

    opts.pl$mu0_init <- list(
        prior_name = dat$mu0.Init.Type[i],
        prior_param = c(dat$Init.Par1[i], dat$Init.Par2[i])
    )

    opts.pl$W <- list(
        init = 0.01,
        infer = ifelse(dat$W.Prior.Type[i] == "invgamma", TRUE, FALSE),
        prior_name = "invgamma",
        prior_param = c(0.01, 0.01)
    )

    opts.pl$W_init <- list(
        prior_name = ifelse(dat$W.Prior.Type[i] == "invgamma", "gamma", "constant"),
        prior_param = c(
            ifelse(dat$W.Prior.Type[i] == "invgamma", 1, 0.01), 
            ifelse(dat$W.Prior.Type[i] == "invgamma", 1, 0.01))
    )

    try(
        {
            out.pl <- dgtf_infer(
                mod, data$y + 1,
                "pl", opts.pl,
                forecast_error = eval_forecast_err,
                fitted_error = eval_fitted_err,
                k = kstep_forecast_err
            )

            if (dat$W.Prior.Type[i] == "invgamma") {
                dat$Wmed[i] <- median(c(out.pl$fit$W))
            } else {
                dat$Wmed[i] = 0.01
            }

            dat$Posterior.median.of.baseline[i] = median(c(out.pl$fit$mu0))

            out.pl.loss_all <- print_loss_all(out.pl)[1:5]
            cname <- paste0("X", 1:5, c(".step", rep(".steps", 4)))
            dat[i, cname] <- out.pl.loss_all
        },
        silent = TRUE
    )
}

fname <- paste0("prior-sensitivity-mu0-pl-", data_type, "-", tag, ".csv")
write.csv(dat, file.path(opath, fname))