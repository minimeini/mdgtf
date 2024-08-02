#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#| label: global argument
save_data <- TRUE
save_figures <- FALSE
plot_figures <- TRUE

seed <- 3269

kstep_forecast_err <- 10
eval_forecast_err <- TRUE
eval_fitted_err <- TRUE

loss <- "quadratic"

fig_height <- 10.1
fig_width <- 13.8

fig_width_rect <- 11.5
fig_height_rect <- 4

param_list <- c("mu0", "rho", "W", "par1", "par2")
param_infer <- c("W")
#
#
#
#| label: source code

library(ggplot2)
library(gridExtra)
library(latex2exp)

repo <- "/Users/meinitang/Dropbox/Repository/poisson-dlm"
Rcpp::sourceCpp(file.path(repo, "export.cpp"))
source(file.path(repo, "plot_timeseries_creditble_interval.r"))
source(file.path(repo, "external_methods.r"))

opath <- "/Users/meinitang/Dropbox/Research/Project1/simulation"
#
#
#
#
#
#
#
#| label: model

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
    nlag = round(qlnorm(0.99, param$lag[1], sqrt(param$lag[2])) + 0.5),
    ntime = 200,
    truncated = TRUE,
    regressor_baseline = FALSE
)

mod1 <- list(
    model = component,
    param = param,
    dim = dim
)

rm(dim)
rm(param)
rm(component)
#
#
#
#
#
#| label: simulated data(sim1)

set.seed(seed)
sim1 <- dgtf_simulate(mod1)

p1 <- ggplot(
    data = data.frame(y = sim1$y, t = 0:(length(sim1$y) - 1)),
    aes(x = t, y = y)
) +
    geom_line() +
    theme_light() +
    xlab("Time t") +
    ylab("Observed Count Values y")

p2 <- ggplot(
    data = data.frame(psi = sim1$psi, t = 0:(length(sim1$psi) - 1)),
    aes(x = t, y = psi)
) +
    geom_line() +
    theme_light() +
    xlab("Time t") +
    ylab("Latent State psi")


p <- arrangeGrob(grobs = list(y = p1, psi = p2), ncol = 1)

if (plot_figures) {
    plot(p)
}
#
#
#
p <- ggplot(data.frame(y = sim1$y[-1], x = 1:mod1$dim$ntime), aes(x = x, y = y)) +
    geom_point() +
    geom_line() +
    theme_minimal() +
    labs(x = "Time", y = "Incidents")

if (save_figures) {
    ggsave(
        file.path(opath, paste0("sim-", seed, "-y.pdf")),
        plot = p,
        device = "pdf", dpi = 300,
        height = fig_height, width = fig_width
    )
}

p2 <- ggplot(data.frame(y = log(exp(sim1$psi[-1]) + 1), x = 1:mod1$dim$ntime), aes(x = x, y = y)) +
    geom_point() +
    geom_line() +
    theme_minimal() +
    labs(x = "Time", y = "Rt")

if (plot_figures) {
    plot(p2)
}


if (save_figures) {
    ggsave(
        file.path(opath, paste0("sim-", seed, "-Rt.pdf")),
        plot = p2,
        device = "pdf", dpi = 300,
        height = fig_height, width = fig_width
    )
}
#
#
#
#
#| label: save figure of y

if (save_figures) {
    ggsave(
        file.path(opath, paste0("sim-", seed, "-y.pdf")),
        plot = p,
        device = "pdf", dpi = 300,
        height = fig_height, width = fig_width
    )
}
#
#
#
#
#
#
#
#
#
#| label: lba tuning discount factor
#| include: false

opts1.lba <- dgtf_default_algo_settings("lba")
opts1.lba$discount_type <- "all_lag_elems"
opts1.lba$W <- 0.01
opts1.lba$use_discount <- TRUE
opts1.lba$use_custom <- TRUE
opts1.lba$num_step_ahead_forecast <- 10

opts1.lba$use_discount <- TRUE
opts1.lba$use_custom <- TRUE

delta.lba <- dgtf_tuning(
    mod1, sim1$y, "lba", opts1.lba,
    "discount_factor",
    from = 0.6, to = 0.95, delta = 0.01,
    loss = "absolute"
)

p1 <- ggplot(
    data.frame(delta = delta.lba[delta.lba[, 1] > 0.6, 1], error = delta.lba[delta.lba[, 1] > 0.6, 2]),
    aes(x = delta, y = error)
) +
    geom_point() +
    theme_light() +
    geom_line(alpha = 0.7, color = "grey") +
    xlab("Discount Factor") +
    ylab("MFE") +
    theme(text = element_text(size = 16))

if (plot_figures) {
    plot(p1)
}
#
#
#
#
#| label: lba tuning W
#| include: false

W_grid <- seq(from = 0.001, to = 0.04, by = 0.001)
opts1.lba$use_discount <- FALSE

W.lba <- dgtf_tuning(
    mod1, sim1$y, "lba", opts1.lba,
    "W",
    grid = W_grid,
    loss = "absolute"
)

p2 <- ggplot(
    data.frame(W = W.lba[, 1], error = W.lba[, 2]),
    aes(x = W, y = error)
) +
    geom_point() +
    theme_light() +
    geom_line(alpha = 0.7, color = "grey") +
    xlab("W") +
    ylab("MFE") +
    theme(text = element_text(size = 16))

if (plot_figures) {
    plot(p2)
}
#
#
#
#| label: figure of lba tuning

if (save_figures) {
    ggsave(
        file.path(
            opath, "discount-factor",
            paste0("sim-", seed, "-lba-optimal-discount-factor.pdf")
        ),
        plot = p1, device = "pdf", dpi = 300,
        height = fig_height, width = fig_width
    )

    ggsave(
        file.path(opath, "W", paste0("sim-", seed, "-lba-optimal-W.pdf")),
        plot = p2, device = "pdf", dpi = 300,
        height = fig_height, width = fig_width
    )
}

#
#
#
#
#
#| label: optimal settings of lba


opts1.lba <- dgtf_default_algo_settings("lba")

opts1.lba$W <- 0.012

opts1.lba$use_discount <- TRUE
opts1.lba$use_custom <- TRUE


opts1.lba$custom_discount_factor <- 0.87
opts1.lba$discount_type <- "first_elem"

# opts1.lba$discount_type = "all_elems"
# opts1.lba$custom_discount_factor <- 0.8
#
#
#
#
#
#| label: run dgtf_infer of lba with discount factor
#| include: false
opts1.lba$use_discount <- TRUE
out1.lba <- dgtf_infer(
    mod1, sim1$y,
    "lba", opts1.lba,
    forecast_error = eval_forecast_err,
    fitted_error = eval_fitted_err,
    k = kstep_forecast_err,
    loss_func = loss
)

#
#
#
#| label: run dgtf_infer of lba with W
#| include: false

opts1.lba$use_discount <- FALSE

out1.lba2 <- dgtf_infer(
    mod1, sim1$y,
    "lba", opts1.lba,
    forecast_error = eval_forecast_err,
    fitted_error = eval_fitted_err,
    k = kstep_forecast_err,
    loss_func = loss
)

#
#
#
#| label: evaluation of lba
#| error: true

out1.lba.loss_all <- print_loss_all(out1.lba)
out1.lba2.loss_all <- print_loss_all(out1.lba2)

print(out1.lba.loss_all)
print(out1.lba2.loss_all)
#
#
#
#| label: figure of lba inference
if (exists("out1.lba")) {
    tag <- paste0("sim-", seed, "-lba-")
    plots <- plot_output(
        out1.lba, sim1$y,
        psi_true = sim1$psi, W_true = sim1$param$err[1], mu0_true = sim1$param$obs[1],
        save_figures = save_figures, tag = tag, opath = opath,
        plot_figures = plot_figures
    )
}

if (exists("out1.lba2")) {
    tag <- paste0("sim-", seed, "-lba2-")
    plots <- plot_output(
        out1.lba2, sim1$y,
        psi_true = sim1$psi, W_true = sim1$param$err[1], mu0_true = sim1$param$obs[1],
        save_figures = save_figures, tag = tag, opath = opath,
        plot_figures = plot_figures
    )
}

#
#
#
#| label: save lba data

if (save_data) {
    save(
        out1.lba, out1.lba2, opts1.lba, delta.lba, W.lba,
        out1.lba.loss_all, out1.lba2.loss_all,
        file = file.path(
            opath, "data",
            paste0("sim-", seed, "-lba.RData")
        )
    )
}
#
#
#
#
#
#
#
#
#
#| label: mcs tuning discount factor
#| include: false
#| eval: false

opts1.mcs <- dgtf_default_algo_settings("mcs")
opts1.mcs$num_backward <- 4
opts1.mcs$num_particle <- 5000
opts1.mcs$num_step_ahead_forecast <- 10

opts1.mcs$W <- 0.01
opts1.mcs$use_discount <- TRUE
opts1.mcs$use_custom <- TRUE

delta.mcs <- dgtf_tuning(
    mod1, sim1$y, "mcs", opts1.mcs,
    "discount_factor",
    from = 0.65, to = 0.98, delta = 0.01
)

p1 <- ggplot(
    data.frame(
        delta = delta.mcs[delta.mcs[, 1] >= 0.7, 1],
        error = delta.mcs[delta.mcs[, 1] >= 0.7, 2]
    ),
    aes(x = delta, y = error)
) +
    geom_point() +
    theme_light() +
    geom_line(alpha = 0.7, color = "grey") +
    xlab("Discount Factor") +
    ylab("MFE") +
    theme(text = element_text(size = 16))

if (plot_figures) {
    plot(p1)
}
#
#
#
#
#| label: mcs tuning W
#| include: false
#| eval: false

W_grid <- seq(from = 0.001, to = 0.05, by = 0.001)
opts1.mcs$use_discount <- FALSE

W.mcs <- dgtf_tuning(
    mod1, sim1$y, "mcs", opts1.mcs,
    "W",
    grid = W_grid
)

p2 <- ggplot(
    data.frame(W = W.mcs[, 1], error = W.mcs[, 2], error2 = W.mcs[, 3]),
    aes(x = W, y = error)
) +
    geom_point() +
    theme_light() +
    geom_line(alpha = 0.7, color = "grey") +
    xlab("W") +
    ylab("MFE") +
    theme(text = element_text(size = 16))

if (plot_figures) {
    plot(p2)
}
#
#
#
#
#| label: mcs tuning B
#| include: false
#| eval: false

opts1.mcs$use_discount <- TRUE
opts1.mcs$use_custom <- TRUE
opts1.mcs$custom_discount_factor <- 0.88

B.mcs <- dgtf_tuning(
    mod1, sim1$y, "mcs", opts1.mcs,
    "num_backward",
    from = 1, to = 30, delta = 1
)

p3 <- ggplot(
    data.frame(B = B.mcs[, 1], error = B.mcs[, 2]),
    aes(x = B, y = error)
) +
    geom_point() +
    theme_light() +
    geom_line(alpha = 0.7, color = "grey") +
    xlab("B") +
    ylab("Mean One-step-ahead Forecasting Error")

if (plot_figures) {
    plot(p3)
}
#
#
#
#| label: figure of mcs tuning
#| eval: false

if (save_figures) {
    ggsave(
        file.path(
            opath, "discount-factor",
            paste0("sim-", seed, "-mcs-optimal-discount-factor.pdf")
        ),
        plot = p1, device = "pdf", dpi = 300,
        height = fig_height, width = fig_width
    )

    ggsave(
        file.path(
            opath, "W",
            paste0("sim-", seed, "-mcs-optimal-W.pdf")
        ),
        plot = p2, device = "pdf", dpi = 300,
        height = fig_height, width = fig_width
    )

    ggsave(
        file.path(
            opath, "num-backward",
            paste0("sim-", seed, "-mcs-optimal-B.pdf")
        ),
        plot = p3, device = "pdf", dpi = 300,
        height = fig_height, width = fig_width
    )
}

#
#
#
#
#
#| label: optimal settings of MCS

# mod1$model$obs_dist = "poisson"

opts1.mcs <- dgtf_default_algo_settings("mcs")
opts1.mcs$num_particle <- 10000
opts1.mcs$num_step_ahead_forecast <- 10

opts1.mcs$W <- 0.008

opts1.mcs$use_discount <- TRUE
opts1.mcs$use_custom <- TRUE
opts1.mcs$custom_discount_factor <- 0.88

opts1.mcs$num_backward <- 5

opts1.mcs$mu0 <- list(
    prior_name = "uniform",
    prior_param = c(0., min(100, median(sim1$y)))
)
#
#
#
#
#
#| label: run dgtf_infer of mcs with discount factor
#| include: false
opts1.mcs$use_discount <- TRUE
out1.mcs <- dgtf_infer(
    mod1, sim1$y,
    "mcs", opts1.mcs,
    forecast_error = eval_forecast_err,
    fitted_error = eval_fitted_err,
    k = kstep_forecast_err,
    loss_func = loss
)
#
#
#
#
#| label: run dgtf_infer of mcs with W
#| include: false
opts1.mcs$use_discount <- FALSE
out1.mcs2 <- dgtf_infer(
    mod1, sim1$y,
    "mcs", opts1.mcs,
    forecast_error = eval_forecast_err,
    fitted_error = eval_fitted_err,
    k = kstep_forecast_err,
    loss_func = loss
)
#
#
#
#| label: evaluation of mcs

if (exists("out1.mcs")) {
    print(print_loss_all(out1.mcs))
}

if (exists("out1.mcs2")) {
    print(print_loss_all(out1.mcs2))
}
#
#
#
#
#| label: figure of mcs inference

if (exists("out1.mcs")) {
    tag <- paste0("sim-", seed, "-mcs-")
    plots <- plot_output(
        out1.mcs, sim1$y,
        psi_true = sim1$psi, W_true = sim1$param$err[1], mu0_true = sim1$param$obs[1],
        save_figures = save_figures, tag = tag, opath = opath,
        plot_figures = plot_figures
    )
}

if (exists("out1.mcs2")) {
    tag <- paste0("sim-", seed, "-mcs2-")
    plots <- plot_output(
        out1.mcs2, sim1$y,
        psi_true = sim1$psi, W_true = sim1$param$err[1], mu0_true = sim1$param$obs[1],
        save_figures = save_figures, tag = tag, opath = opath,
        plot_figures = plot_figures
    )
}
#
#
#
#
#| label: save mcs data

if (save_data) {
    if (exists("out1.mcs") && exists("out1.mcs2")) {
        save(
            out1.mcs, out1.mcs2, opts1.mcs,
            # delta.mcs, W.mcs, B.mcs,
            file = file.path(
                opath, "data",
                paste0("sim-", seed, "-mcs.RData")
            )
        )
    } else if (exists("out1.mcs")) {
        save(
            out1.mcs, opts1.mcs,
            # delta.mcs, W.mcs, B.mcs,
            file = file.path(
                opath, "data",
                paste0("sim-", seed, "-mcs.RData")
            )
        )
    } else if (exists("out1.mcs2")) {
        save(
            out1.mcs2, opts1.mcs,
            # delta.mcs, W.mcs, B.mcs,
            file = file.path(
                opath, "data",
                paste0("sim-", seed, "-mcs.RData")
            )
        )
    }
}
#
#
#
#
#
#
#
#
#| label: ffbs tuning of discount factor
#| include: false
#| eval: false


opts1.ffbs <- dgtf_default_algo_settings("ffbs")
opts1.ffbs$num_particle <- 5000
opts1.ffbs$num_smooth <- 1000
opts1.ffbs$do_smoothing <- TRUE
opts1.ffbs$num_step_ahead_forecast <- 10

opts1.ffbs$W <- 0.01
opts1.ffbs$use_discount <- TRUE
opts1.ffbs$use_custom <- TRUE
opts1.ffbs$custom_discount_factor <- 0.88

delta.ffbs <- dgtf_tuning(
    mod1, sim1$y, "ffbs", opts1.ffbs,
    "discount_factor",
    from = 0.8, to = 0.95, delta = 0.005
)

p1 <- ggplot(
    data.frame(delta = delta.ffbs[delta.ffbs[, 1] > 0.79, 1], error = delta.ffbs[delta.ffbs[, 1] > 0.79, 2]),
    aes(x = delta, y = error)
) +
    geom_point() +
    theme_light() +
    geom_line(alpha = 0.7, color = "grey") +
    xlab("Discount Factor") +
    ylab("Mean One-step-ahead Forecasting Error") +
    theme(text = element_text(size = 16))

if (plot_figures) {
    plot(p1)
}

#
#
#
#| label: ffbs tuning of W
#| include: false
#| eval: false

W_grid <- seq(from = 0.001, to = 0.04, by = 0.001)
opts1.ffbs$use_discount <- FALSE

W.ffbs <- dgtf_tuning(
    mod1, sim1$y, "ffbs", opts1.ffbs,
    "W",
    grid = W_grid
)

p2 <- ggplot(
    data.frame(W = W.ffbs[, 1], error = W.ffbs[, 2]),
    aes(x = W, y = error)
) +
    geom_point() +
    theme_light() +
    geom_line(alpha = 0.7, color = "grey") +
    xlab("W") +
    ylab("Mean One-step-ahead Forecasting Error")

if (plot_figures) {
    plot(p2)
}

#
#
#
#| label: figure of ffbs tuning
#| eval: false

if (save_figures) {
    ggsave(
        file.path(
            opath, "discount-factor",
            paste0("sim-", seed, "-ffbs-optimal-discount-factor.pdf")
        ),
        plot = p1, device = "pdf", dpi = 300,
        height = fig_height, width = fig_width
    )

    ggsave(
        file.path(
            opath, "W",
            paste0("sim-", seed, "-ffbs-optimal-W.pdf")
        ),
        plot = p2, device = "pdf", dpi = 300,
        height = fig_height, width = fig_width
    )
}
#
#
#
#
#
#
#| label: optimal settings of ffbs
#| eval: false

opts1.ffbs <- dgtf_default_algo_settings("ffbs")
opts1.ffbs$num_particle <- 50000
opts1.ffbs$num_smooth <- 1000
opts1.ffbs$do_smoothing <- TRUE

opts1.ffbs$num_step_ahead_forecast <- 10

opts1.ffbs$W <- list(
    init = 0.006,
    infer = FALSE
)
opts1.ffbs$mu0$init <- mod1$param$obs[1]

opts1.ffbs$use_discount <- TRUE
opts1.ffbs$use_custom <- TRUE
opts1.ffbs$custom_discount_factor <- 0.91

#
#
#
#
#
#| label: run dgtf_infer with ffbs with discount factor
#| include: false
#| eval: false


opts1.ffbs$use_discount <- TRUE
out1.ffbs <- dgtf_infer(
    mod1, sim1$y,
    "ffbs", opts1.ffbs,
    forecast_error = eval_forecast_err,
    fitted_error = eval_fitted_err,
    k = kstep_forecast_err,
    loss_func = loss
)
#
#
#
#| label: run dgtf_infer with ffbs with W
#| include: false
#| eval: false

opts1.ffbs$do_smoothing <- TRUE
opts1.ffbs$do_backward <- TRUE
opts1.ffbs$use_discount <- FALSE
out1.ffbs2 <- dgtf_infer(
    mod1, sim1$y,
    "ffbs", opts1.ffbs,
    forecast_error = eval_forecast_err,
    fitted_error = eval_fitted_err,
    k = kstep_forecast_err,
    loss_func = loss
)

#
#
#
#| error: true
#| label: evaluation of ffbs
#| eval: false

if (exists("out1.ffbs")) {
    print(print_loss_all(out1.ffbs))
}

if (exists("out1.ffbs2")) {
    print(print_loss_all(out1.ffbs2))
}

#
#
#
#
#| label: figure of ffbs inference
#| eval: false

if (exists("out1.ffbs")) {
    tag <- paste0("sim-", seed, "-ffbs-")
    plots <- plot_output(
        out1.ffbs, sim1$y,
        psi_true = sim1$psi, W_true = sim1$param$err[1], mu0_true = sim1$param$obs[1],
        save_figures = save_figures, tag = tag, opath = opath,
        plot_figures = plot_figures
    )
}


if (exists("out1.ffbs2")) {
    tag <- paste0("sim-", seed, "-ffbs2-")
    plots <- plot_output(
        out1.ffbs2, sim1$y,
        psi_true = sim1$psi, W_true = sim1$param$err[1], mu0_true = sim1$param$obs[1],
        save_figures = save_figures, tag = tag, opath = opath,
        plot_figures = plot_figures
    )
}



#
#
#
#
#| label: save ffbs data
#| eval: false

if (save_data) {
    save_vars <- c("out1.ffbs", "out1.ffbs2", "opts1.ffbs", "delta.ffbs", "W.ffbs")
    save_vars <- save_vars[sapply(save_vars, exists)]
    save(list = save_vars, file = file.path(
        opath, "data",
        paste0("sim-", seed, "-ffbs.RData")
    ))
}
#
#
#
#
#
#
#
#
#
#| label: optimal settings of tfs

opts1.tfs <- dgtf_default_algo_settings("tfs")
opts1.tfs$num_particle <- 10000
opts1.tfs$num_smooth <- 1000
opts1.tfs$do_smoothing <- TRUE

opts1.tfs$num_step_ahead_forecast <- 10

opts1.tfs$W <- list(
    init = mod1$param$err[1],
    infer = FALSE
)
opts1.tfs$mu0 <- list(
    init = mod1$param$obs[1],
    infer = FALSE
)

opts1.tfs$use_discount <- TRUE
opts1.tfs$use_custom <- TRUE
opts1.tfs$custom_discount_factor <- 0.91

#
#
#
#
#
#| label: run dgtf_infer with tfs with discount factor
#| include: false


# opts1.tfs$use_discount <- TRUE
# out1.tfs <- dgtf_infer(
#     mod1, sim1$y,
#     "tfs", opts1.tfs,
#     forecast_error = eval_forecast_err,
#     fitted_error = eval_fitted_err,
#     k = kstep_forecast_err,
#     loss_func = loss
# )
#
#
#
#| label: run dgtf_infer with tfs with W
#| include: false

opts1.tfs$do_smoothing <- TRUE
opts1.tfs$use_discount <- FALSE
out1.tfs2 <- dgtf_infer(
    mod1, sim1$y,
    "tfs", opts1.tfs,
    forecast_error = eval_forecast_err,
    fitted_error = eval_fitted_err,
    k = kstep_forecast_err,
    loss_func = loss
)

#
#
#
#| error: true
#| label: evaluation of tfs

if (exists("out1.tfs")) {
    print(print_loss_all(out1.tfs))
}

if (exists("out1.tfs2")) {
    print(print_loss_all(out1.tfs2))
}

#
#
#
#| label: figure of tfs inference

if (exists("out1.tfs")) {
    tag <- paste0("sim-", seed, "-tfs-")
    plots <- plot_output(
        out1.tfs, sim1$y,
        psi_true = sim1$psi, W_true = sim1$param$err[1], mu0_true = sim1$param$obs[1],
        save_figures = save_figures, tag = tag, opath = opath,
        plot_figures = plot_figures
    )
}


if (exists("out1.tfs2")) {
    tag <- paste0("sim-", seed, "-tfs2-")
    plots <- plot_output(
        out1.tfs2, sim1$y,
        psi_true = sim1$psi, W_true = sim1$param$err[1], mu0_true = sim1$param$obs[1],
        save_figures = save_figures, tag = tag, opath = opath,
        plot_figures = plot_figures
    )
}



#
#
#
#| label: save tfs data

if (save_data) {
    if (exists("out1.tfs")) {
        save(
            out1.tfs, opts1.tfs,
            file = file.path(
                opath, "data",
                paste0("sim-", seed, "-tfs.RData")
            )
        )
    }

    if (exists("out1.tfs2")) {
        save(
            out1.tfs2, opts1.tfs,
            file = file.path(
                opath, "data",
                paste0("sim-", seed, "-tfs2.RData")
            )
        )
    }
}
#
#
#
#
#
#
#
opts1.pl <- dgtf_default_algo_settings("pl")

opts1.pl$num_particle <- 1000
opts1.pl$num_step_ahead_forecast <- 0
opts1.pl$do_smoothing <- TRUE

opts1.pl$max_iter <- 100

opts1.pl$mu0 <- list(
    init = mod1$param$obs[1],
    infer = FALSE,
    prior_name = "normal",
    prior_param = c(0, 10.)
)

opts1.pl$mu0_init <- list(
    prior_name = "constant",
    prior_param = c(mod1$param$obs[1], 10.)
)

opts1.pl$W <- list(
    init = mod1$param$err[1],
    infer = "W" %in% param_infer,
    prior_name = "invgamma",
    prior_param = c(1, 1)
)

opts1.pl$W_init <- list(
    prior_name = "invgamma",
    prior_param = c(1, 1)
)


opts1.pl$rho <- list(
    infer = FALSE,
    init = mod1$param$obs[2],
    mh_sd = 0.01,
    prior_name = "gaussian",
    prior_param = c(0., 100.)
)

opts1.pl$par1 <- list(
    infer = FALSE,
    init = mod1$param$lag[1],
    mh_sd = 0.002,
    prior_name = "gaussian",
    prior_param = c(1, 1.)
)


opts1.pl$par2 <- list(
    infer = FALSE,
    init = mod1$param$lag[2],
    mh_sd = 0.01,
    prior_name = "invgamma",
    prior_param = c(1, 1.)
)


out1.pl <- dgtf_infer(
    mod1, sim1$y,
    "pl", opts1.pl,
    forecast_error = FALSE,
    fitted_error = FALSE,
    k = 0
)
#
#
#
#
#
#| label: run dgtf_infer with pl
#| include: false
opts1.pl <- dgtf_default_algo_settings("pl")

opts1.pl$num_particle <- 10000
opts1.pl$num_smooth <- 5000
opts1.pl$num_step_ahead_forecast <- 0
opts1.pl$do_smoothing <- TRUE

opts1.pl$max_iter <- 100

opts1.pl$mu0 <- list(
    init = mod1$param$obs[1],
    infer = "mu0" %in% param_infer,
    prior_name = "normal",
    prior_param = c(0, 10.)
)

opts1.pl$mu0_init <- list(
    prior_name = "constant",
    prior_param = c(mod1$param$obs[1], 10.)
)

opts1.pl$W <- list(
    init = mod1$param$err[1],
    infer = "W" %in% param_infer,
    prior_name = "invgamma",
    prior_param = c(1, 1)
)

opts1.pl$W_init <- list(
    prior_name = "invgamma",
    prior_param = c(1, 1)
)


opts1.pl$rho <- list(
    infer = "rho" %in% param_infer,
    init = mod1$param$obs[2],
    mh_sd = 0.01,
    prior_name = "gaussian",
    prior_param = c(0., 100.)
)

opts1.pl$par1 <- list(
    infer = "par1" %in% param_infer,
    init = mod1$param$lag[1],
    mh_sd = 0.002,
    prior_name = "gaussian",
    prior_param = c(1, 1.)
)


opts1.pl$par2 <- list(
    infer = "par2" %in% param_infer,
    init = mod1$param$lag[2],
    mh_sd = 0.01,
    prior_name = "invgamma",
    prior_param = c(1, 1.)
)


out1.pl <- dgtf_infer(
    mod1, sim1$y,
    "pl", opts1.pl,
    forecast_error = eval_forecast_err,
    fitted_error = eval_fitted_err,
    k = kstep_forecast_err
)
#
#
#
#| label: evaluation of pl
if (exists("out1.pl")) {
    out1.pl.loss_all <- print_loss_all(out1.pl)
    print(out1.pl.loss_all)
}


# if ("W" %in% names(out1.pl$fit)) {
#     print(summary(c(out1.pl$fit$W)))
#     p <- plot_prior_sensitivity_pl(out1.pl, opts1.pl, "W", 1, FALSE)
#     if (plot_figures) {
#         plot(p) + labs(title = "prior of W, IG(0.1, 0.1), is in gray; posterior of W is in red; initial values in blue")
#     }
# }

# if ("mu0" %in% names(out1.pl$fit)) {
#     print(summary(c(out1.pl$fit$mu0)))
#     p <- plot_prior_sensitivity_pl(out1.pl, opts1.pl, "mu0", 2, FALSE)
#     if (plot_figures) {
#         plot(p) + labs(title = "prior of a, N(0, 10), is in gray; posterior of a is in red; initial values in blue")
#     }
# }
#
#
#
#| label: figure of pl inference

if (exists("out1.pl")) {
    tag <- paste0("sim-", seed, "-pl2-")
    plots <- plot_output(
        out1.pl, sim1$y,
        psi_true = sim1$psi, mu0_true = sim1$param$obs[1],
        save_figures = save_figures, tag = tag, opath = opath,
        plot_figures = plot_figures
    )
}

#
#
#
#| label: run dgtf_infer with tfs with W estimated by pl
#| include: false

mod2 <- mod1
if ("rho" %in% param_infer) {
    mod2$param$obs[2] <- median(out1.pl$fit$rho)
}

if ("par1" %in% param_infer || "par2" %in% param_infer) {
    mod2$param$lag[1] <- median(out1.pl$fit$par1)
    mod2$param$lag[2] <- median(out1.pl$fit$par2)
}

opts1.tfs <- dgtf_default_algo_settings("tfs")
opts1.tfs$num_particle <- 100000
opts1.tfs$do_smoothing <- TRUE
opts1.tfs$num_step_ahead_forecast <- 10
opts1.tfs$use_discount <- FALSE

opts1.tfs$W <- list(
    init = ifelse("W" %in% param_infer, median(c(out1.pl$fit$W)), mod1$param$err[1]),
    infer = FALSE
)


opts1.tfs$mu0 <- list(
    init = ifelse("mu0" %in% param_infer, median(c(out1.pl$fit$mu0)), mod1$param$obs[1]),
    infer = FALSE
)


opts1.tfs$do_smoothing <- TRUE
opts1.tfs$use_discount <- FALSE
out1.tfs3 <- dgtf_infer(
    mod2, sim1$y,
    "tfs", opts1.tfs,
    forecast_error = eval_forecast_err,
    fitted_error = eval_fitted_err,
    k = kstep_forecast_err,
    loss_func = loss
)


if (exists("out1.tfs3")) {
    tag <- paste0("sim-", seed, "-tfs3-")
    plots <- plot_output(
        out1.tfs3, sim1$y,
        psi_true = sim1$psi, W_true = sim1$param$err[1], mu0_true = sim1$param$obs[1],
        save_figures = save_figures, tag = tag, opath = opath,
        plot_figures = plot_figures
    )
}


#
#
#
#
#| label: save pl data
if (save_data && exists("out1.pl")) {
    var_list <- c("out1.pl", "out1.tfs3", "opts1.pl", "opts1.tfs")
    var_list <- var_list[sapply(var_list, exists)]
    save(list = var_list, file = file.path(
        opath, "data",
        paste0("sim-", seed, "-pl-W.RData")
    ))
}
#
#
#
#
#
#
#
#
#
#
#
#| label: hva tuning of learning rate
#| include: false
#| eval: false

opts1.hva <- dgtf_default_algo_settings("hva")
opts1.hva$nsample <- 1000
opts1.hva$nthin <- 2
opts1.hva$nburnin <- 1000
opts1.hva$mcs$num_particle <- 100
opts1.hva$mcs$num_backward <- 5
opts1.hva$num_step_ahead_forecast <- 10

opts1.hva$W$init <- 0.01
opts1.hva$W$infer <- TRUE

opts1.hva$eps_step_size <- 1.e-5
opts1.hva$k <- 1


lrate.hva <- dgtf_tuning(
    mod1, sim1$y, "hva", opts1.hva,
    "learning_rate",
    from = 0.01, to = 0.1, delta = 0.01
)

p1 <- ggplot(
    data.frame(delta = lrate.hva[, 1], error = lrate.hva[, 2]),
    aes(x = delta, y = error)
) +
    geom_point() +
    theme_light() +
    geom_line(alpha = 0.7, color = "grey") +
    xlab("Learning Rate") +
    ylab("Mean One-step-ahead Forecasting Error")


if (plot_figures) {
    plot(p1)
}
#
#
#
#| label: hva tuning of step size
#| include: false
#| eval: false

opts1.hva$learning_rate <- 0.02

eps_grid <- c(1.e-6, 1.e-5, 1.e-3, 1.e-2)
eps.hva <- dgtf_tuning(
    mod1, sim1$y, "hva", opts1.hva,
    "step_size",
    grid = eps_grid
)

p2 <- ggplot(
    data.frame(delta = eps.hva[, 1], error = eps.hva[, 2]),
    aes(x = delta, y = error)
) +
    geom_point() +
    theme_light() +
    geom_line(alpha = 0.7, color = "grey") +
    xlab("Step Size") +
    ylab("Mean One-step-ahead Forecasting Error")

if (plot_figures) {
    plot(p2)
}
#
#
#
#| label: figure of hva tuning

if (save_figures) {
    if (exists("lrate.hva") && exists("p1")) {
        ggsave(
            file.path(
                opath, "hva",
                paste0("sim-", seed, "-hva-optimal-learning-rate.pdf")
            ),
            plot = p1, device = "pdf", dpi = 300,
            height = fig_height, width = fig_width
        )
    }

    if (exists("eps.hva") && exists("p2")) {
        ggsave(
            file.path(
                opath, "hva",
                paste0("sim-", seed, "-hva-step-size.pdf")
            ),
            plot = p2, device = "pdf", dpi = 300,
            height = fig_height, width = fig_width
        )
    }
}

#
#
#
#
#
#
opts1.hva <- dgtf_default_algo_settings("hva")
opts1.hva$nsample <- 5000
opts1.hva$nthin <- 2
opts1.hva$nburnin <- 5000

opts1.hva$learning_rate <- 0.02
opts1.hva$eps_step_size <- 1.e-5
opts1.hva$k <- length(param_infer)


opts1.hva$mu0 <- list(
    infer = "mu0" %in% param_infer,
    init = mod1$param$obs[1]
)

opts1.hva$W <- list(
    infer = "W" %in% param_infer,
    init = mod1$param$err[1],
    prior_name = "invgamma",
    prior_param = c(1, 1)
)

opts1.hva$rho$init <- mod1$param$obs[2]
opts1.hva$rho$infer <- "rho" %in% param_infer

opts1.hva$par1 <- list(
    infer = "par1" %in% param_infer,
    init = 2,
    prior_name = "gaussian",
    prior_param = c(0., 1.)
)

opts1.hva$par2 <- list(
    infer = "par2" %in% param_infer,
    init = mod1$param$lag[2],
    prior_name = "invgamma",
    prior_param = c(1, 1)
)



opts1.hva$smc$num_backward <- 1
opts1.hva$smc$do_smoothing <- TRUE

num_particle <- c(10, 50, 100, 500)

for (npar in num_particle) {
    opts1.hva$smc$num_particle = npar
    out1.hva <- dgtf_infer(
        mod1, sim1$y,
        "hva", opts1.hva,
        forecast_error = FALSE,
        fitted_error = TRUE,
        k = 0
    )

    print(print_loss_all(out1.hva))
}
#
#
#
#
#
#| label: optimal settings of hva

opts1.hva <- dgtf_default_algo_settings("hva")
opts1.hva$nsample <- 5000
opts1.hva$nthin <- 2
opts1.hva$nburnin <- 5000
opts1.hva$num_step_ahead_forecast <- 10
opts1.hva$tstart_pct = 0.9

opts1.hva$smc$num_particle <- 50
opts1.hva$smc$num_backward <- 1
opts1.hva$smc$do_smoothing <- FALSE


opts1.hva$learning_rate <- 0.02
opts1.hva$eps_step_size <- 1.e-5
opts1.hva$k <- length(param_infer)


opts1.hva$mu0 <- list(
    infer = "mu0" %in% param_infer,
    init = mod1$param$obs[1]
)

opts1.hva$W <- list(
    infer = "W" %in% param_infer,
    init = mod1$param$err[1],
    prior_name = "invgamma",
    prior_param = c(1, 1)
)

opts1.hva$rho$init <- mod1$param$obs[2]
opts1.hva$rho$infer <- "rho" %in% param_infer

opts1.hva$par1 <- list(
    infer = "par1" %in% param_infer,
    init = 2,
    prior_name = "gaussian",
    prior_param = c(0., 1.)
)

opts1.hva$par2 <- list(
    infer = "par2" %in% param_infer,
    init = mod1$param$lag[2],
    prior_name = "invgamma",
    prior_param = c(1, 1)
)


#
#
#
#
#
#| label: run dgtf_infer with hva
#| include: false
#| eval: true

opts1.hva$learning_rate <- 0.02
opts1.hva$eps_step_size <- 1.e-5
out1.hva <- dgtf_infer(
    mod1, sim1$y,
    "hva", opts1.hva,
    forecast_error = TRUE,
    fitted_error = eval_fitted_err,
    k = kstep_forecast_err
)
#
#
#
#
#| error: true
#| label: evaluation of hva
if (exists("out1.hva")) {
    out1.hva.loss_all <- print_loss_all(out1.hva)
    print(out1.hva.loss_all)
}

#
#
#
#
#| label: figure of hva inference

if (exists("out1.hva")) {
    tag <- paste0("sim-", seed, "-hva-")
    plots <- plot_output(
        out1.hva, sim1$y,
        psi_true = sim1$psi, W_true = sim1$param$err[1], mu0_true = sim1$param$obs[1],
        save_figures = save_figures, tag = tag, opath = opath,
        plot_figures = plot_figures
    )
}


#
#
#
#| label: save hva data

if (save_data) {
    var_list <- c(
        "out1.hva", "out1.hva2",
        "opts1.hva", "opts1.hva2",
        "out1.hva.loss_all", "out1.hva2.loss_all"
    )

    var_list <- var_list[sapply(var_list, exists)]
    save(list = var_list, file = file.path(
        opath, "data", paste0("sim-", seed, "-hva-W-forecast_20pts.RData")
    ))
}

#
#
#
#
#
#
#
#
#
#
#
#
#| label: run dgtf_infer with mcmc
#| include: false
#| eval: true

opts1.mcmc <- dgtf_default_algo_settings("mcmc")
opts1.mcmc$nsample <- 5000
opts1.mcmc$nthin <- 2
opts1.mcmc$nburnin <- 5000
opts1.mcmc$num_step_ahead_forecast <- 10
opts1.mcmc$tstart_pct = 0.9

opts1.mcmc$mh_sd <- 1
opts1.mcmc$epsilon <- 0.005
opts1.mcmc$L <- 10
opts1.mcmc$m <- c(0.1, 0.1)

opts1.mcmc$mu0$init <- mod1$param$obs[1]
opts1.mcmc$mu0$infer <- "mu0" %in% param_infer
opts1.mcmc$mu0$mh_sd <- 0.1

opts1.mcmc$W$init <- mod1$param$err[1]
opts1.mcmc$W$infer <- "W" %in% param_infer

opts1.mcmc$rho$init <- mod1$param$obs[2]
opts1.mcmc$rho$infer <- "rho" %in% param_infer
opts1.mcmc$rho$mh_sd <- 0.5

opts1.mcmc$par1 <- list(
    infer = "par1" %in% param_infer,
    init = mod1$param$lag[1],
    prior_name = "gaussian",
    prior_param = c(0., 1.),
    mh_sd = 0.1
)

opts1.mcmc$par2 <- list(
    infer = "par2" %in% param_infer,
    init = mod1$param$lag[2],
    prior_name = "invgamma",
    prior_param = c(1, 1),
    mh_sd = 0.1
)

out1.mcmc <- dgtf_infer(
    mod1, sim1$y,
    "mcmc", opts1.mcmc,
    forecast_error = TRUE,
    fitted_error = eval_fitted_err,
    k = kstep_forecast_err
)
#
#
#
#| label: evaluation of mcmc

if (exists("out1.mcmc")) {
    out1.mcmc.loss_all <- print_loss_all(out1.mcmc)
    print(out1.mcmc.loss_all)
}


#
#
#
#
#| label: figure of mcmc

if (exists("out1.mcmc")) {
    tag <- paste0("sim-", seed, "-mcmc-")
    plots <- plot_output(
        out1.mcmc, sim1$y,
        psi_true = sim1$psi, W_true = sim1$param$err[1], mu0_true = sim1$param$obs[1],
        save_figures = save_figures, tag = tag, opath = opath,
        plot_figures = plot_figures
    )
}

#
#
#
#
#| label: save mcmc data

if (exists("out1.mcmc")) {
    if (save_data) {
        var_list <- c("out1.mcmc", "opts1.mcmc", "out1.mcmc.loss_all")
        var_list <- var_list[sapply(var_list, exists)]
        save(list = var_list, file = file.path(
            opath, "data", paste0("sim-", seed, "-mcmc-mu0.RData")
        ))
    }
}
#
#
#
#
#
#
#
#
#
#
#
mod_g1 <- mod1
mod_g1$model$gain_func <- "ramp"

tfs.opts <- dgtf_default_algo_settings("tfs")
tfs.opts$num_particle <- 100000
tfs.opts$do_smoothing <- TRUE
tfs.opts$num_step_ahead_forecast <- 10
tfs.opts$use_discount <- FALSE

tfs.opts$W <- list(
    init = mod_g1$param$err[1],
    infer = FALSE
)


tfs.opts$mu0 <- list(
    init = mod_g1$param$obs[1],
    infer = FALSE
)


tfs.opts$do_smoothing <- TRUE
tfs.opts$use_discount <- FALSE
#
#
#
ramp.tfs <- dgtf_infer(
    mod_g1, sim1$y,
    "tfs", tfs.opts,
    forecast_error = eval_forecast_err,
    fitted_error = eval_fitted_err,
    k = kstep_forecast_err,
    loss_func = loss
)

#
#
#
mod_g2 <- mod1
mod_g2$model$gain_func <- "exponential"
exp.tfs <- dgtf_infer(
    mod_g2, sim1$y,
    "tfs", tfs.opts,
    forecast_error = eval_forecast_err,
    fitted_error = eval_fitted_err,
    k = kstep_forecast_err,
    loss_func = loss
)

mod_g3 <- mod1
mod_g3$model$gain_func <- "softplus"
softplus.tfs <- dgtf_infer(
    mod_g3, sim1$y,
    "tfs", tfs.opts,
    forecast_error = eval_forecast_err,
    fitted_error = eval_fitted_err,
    k = kstep_forecast_err,
    loss_func = loss
)


#
#
#
if (!exists("exp.tfs") | !exists("softplus.tfs")) {
    load(file.path(opath, "data", "sim-3269-gain-tfs.RData"))
}

p <- plot_ts_ci_multi(
    psi_list = list(
        Exponential = exp(exp.tfs$fit$psi),
        Softplus = log(exp(softplus.tfs$fit$psi) + 1),
        True = cbind(
            log(exp(sim1$psi) + 1),
            log(exp(sim1$psi) + 1),
            log(exp(sim1$psi) + 1)
        )
    ),
    legend.position = "top", ylab = expression(R[t])
) + scale_fill_manual(
    name = "Gain function",
    values = c("maroon", "royalblue", "black"),
    labels = c("Exponential", "Softplus", "True"),
    guide = guide_legend(nrow = 1)
) +
    scale_color_manual(
        name = "Gain function",
        values = c("maroon", "royalblue", "black"),
        labels = c("Exponential", "Softplus", "True")
    ) +
    theme(text = element_text(size = 16)) +
    geom_hline(yintercept = 1, linetype = "dashed")

if (plot_figures) {
    plot(p)
}

if (save_figures) {
    ggsave(
        file.path(
            opath,
            paste0("sim-", seed, "-Rt-gain-function-tfs.pdf")
        ),
        plot = p, device = "pdf", dpi = 300,
        height = 0.2 * fig_width_rect, width = fig_width_rect
    )
}
#
#
#
if (save_data) {
    var_list <- c("exp.tfs", "softplus.tfs", "tfs.opts", "mod_g2", "mod_g3")
    var_list <- var_list[sapply(var_list, exists)]
    save(list = var_list, file = file.path(
        opath, "data", paste0("sim-", seed, "-gain-tfs.RData")
    ))
}
#
#
#
#
#
#| label: optimal obs with lba
#| eval: false

mod1$model$obs_dist <- "nbinom"
delta_grid <- 1:100
out1.stats.obs <- dgtf_optimal_obs(mod1, sim1$y, "lba", opts1.lba, delta_grid)

mod1$model$obs_dist <- "poisson"
out1.lba2 <- dgtf_infer(mod1, sim1$y, "lba", opts1.lba)

ymin <- min(c(out1.stats.obs[, 2], out1.lba2$error$forecast$y_loss_all[1]))
ymax <- max(c(out1.stats.obs[, 2], out1.lba2$error$forecast$y_loss_all[1]))

plot(out1.stats.obs[1:100, 1], out1.stats.obs[1:100, 2], ylim = c(ymin, ymax))
abline(h = out1.lba2$error$forecast$y_loss_all[1], col = "maroon")

out1.stats.obs <- as.data.frame(out1.stats.obs)
colnames(out1.stats.obs) <- c("delta", "forecast-err", "fit-err")
p <- ggplot(out1.stats.obs, aes(x = delta, y = `forecast-err`)) +
    geom_point() +
    theme_light() +
    geom_hline(aes(yintercept = out1.lba2$error$forecast$y_loss_all[1]), color = "maroon") +
    ylab("RMSE of one-step-ahead forecasting") +
    xlab(expression(delta))

if (plot_figures) {
    plot(p)
}

if (save_figures) {
    ggsave(
        file.path(
            opath, "model-selection",
            paste0("sim-", seed, "-lba-optimal-obs.pdf")
        ),
        plot = p, device = "pdf", dpi = 300,
        height = fig_height, width = fig_width
    )
}
#
#
#
#| label: optimal obs with mcs
#| eval: false
mod1$model$obs_dist <- "nbinom"
delta_grid <- 1:100
out1.stats.obs2 <- dgtf_optimal_obs(mod1, sim1$y, "mcs", opts1.mcs, delta_grid)

mod1$model$obs_dist <- "poisson"
out1.mcs2 <- dgtf_infer(mod1, sim1$y, "mcs", opts1.mcs)

ymin <- min(c(out1.stats.obs2[, 2], out1.mcs2$error$forecast$y_loss_all[1]))
ymax <- max(c(out1.stats.obs2[, 2], out1.mcs2$error$forecast$y_loss_all[1]))

out1.stats.obs2 <- as.data.frame(out1.stats.obs2)
colnames(out1.stats.obs2) <- c("delta", "forecast-err", "fit-err")
p <- ggplot(out1.stats.obs2, aes(x = delta, y = `forecast-err`)) +
    geom_point() +
    theme_light() +
    geom_hline(aes(yintercept = out1.mcs2$error$forecast$y_loss_all[1]), color = "maroon") +
    ylab("RMSE of one-step-ahead forecasting") +
    xlab(expression(delta))

plot(p)
#
#
#
#| label: optimal obs
#| eval: false

mod1$model$obs_dist <- "poisson"
#
#
#
#
#
#
#
mod_lognorm <- mod1
mod_lognorm$model$lag_dist <- "lognorm"
mod_lognorm$param$lag <- c(1.39, 0.32)

mod_nbinom <- mod1
mod_nbinom$model$lag_dist <- "nbinom"
mod_nbinom$param$lag <- c(0.395, 6)

p <- ggplot(data = data.frame(x = rnbinom(100000, 6, 1 - 0.395)), aes(x = x)) +
    geom_histogram(alpha = 0.5, bins = 100) +
    geom_histogram(data = data.frame(x = round(rlnorm(100000, 1.39, 0.32))), bins = 100, alpha = 0.5) +
    theme_minimal() +
    labs(xlab = "Lag", ylab = "Counts")

if (plot_figures) {
    plot(p)
}

if (save_figures) {
    ggsave(
        file.path(
            opath,
            paste0("sim-", seed, "-lag-distributions.pdf")
        ),
        plot = p, device = "pdf", dpi = 300,
        height = fig_height, width = fig_width
    )
}
#
#
#
#
opts.lag <- dgtf_default_algo_settings("tfs")
opts.lag$num_particle <- 100000
opts.lag$num_smooth <- 1000
opts.lag$do_smoothing <- TRUE

opts.lag$num_step_ahead_forecast <- 10

opts.lag$W <- list(
    init = mod1$param$err[1],
    infer = FALSE
)
opts.lag$mu0 <- list(
    init = mod1$param$obs[1],
    infer = FALSE
)

opts.lag$use_discount <- FALSE

lognorm.tfs <- dgtf_infer(
    mod_lognorm, sim1$y,
    "tfs", opts.lag,
    forecast_error = eval_forecast_err,
    fitted_error = eval_fitted_err,
    k = kstep_forecast_err,
    loss_func = loss
)

nbinom.tfs <- dgtf_infer(
    mod_nbinom, sim1$y,
    "tfs", opts.lag,
    forecast_error = eval_forecast_err,
    fitted_error = eval_fitted_err,
    k = kstep_forecast_err,
    loss_func = loss
)


if (save_data) {
    var_list <- c("mod_lognorm", "mod_nbinom", "opts.lag", "lognorm.tfs", "nbinom.tfs")
    var_list <- var_list[sapply(var_list, exists)]
    save(list = var_list, file = file.path(opath, "data", paste0(
        "sim-", seed, "-lag-tfs.RData"
    )))
}
#
#
#
if (!exists("nbinom.tfs") | !exists("lognorm.tfs")) {
    load(file.path(opath, "data", paste0(
        "sim-", seed, "-lag-tfs.RData"
    )))
}

p <- plot_ts_ci_multi(psi_list = list(
    nbinom = log(exp(nbinom.tfs$fit$psi) + 1),
    lognorm = log(exp(lognorm.tfs$fit$psi) + 1),
    True = cbind(
        log(exp(sim1$psi) + 1),
        log(exp(sim1$psi) + 1),
        log(exp(sim1$psi) + 1)
    )
), ylab = expression(R[t])) +
    theme(legend.position = "top", text = element_text(size = 16)) +
    scale_fill_manual(
        name = "Lag distribution",
        values = c("maroon", "royalblue", "black"),
        guide = guide_legend(nrow = 1),
        labels = c("Log-normal(1.4, 0.3)", "NB(6, 0.395)", "True")
    ) +
    scale_color_manual(
        name = "Lag distribution",
        values = c("maroon", "royalblue", "black"),
        labels = c("Log-normal(1.4, 0.3)", "NB(6, 0.395)", "True"),
        guide = NULL
    ) + geom_hline(yintercept = 1, linetype = "dashed")

if (plot_figures) {
    plot(p)
}

if (save_figures) {
    ggsave(
        file.path(
            opath,
            paste0("sim-", seed, "-Rt-lag-distribution-tfs.pdf")
        ),
        plot = p, device = "pdf", dpi = 300,
        height = 0.2 * fig_width_rect, width = fig_width_rect
    )
}

#
#
#
#
#
#
#
#| label: lognormal lag dist with lba
#| eval: false

mod1$model$obs_dist <- "poisson"
mod1$model$lag_dist <- "lognorm"

mu_grid <- seq(from = 1, to = 3, by = 0.1)
sd2_grid <- seq(from = 0.1, to = 3, by = 0.1)
grids <- expand.grid(mu_grid, sd2_grid)
grids$mode <- exp(grids[, 1] - grids[, 2])
grids <- grids[grids$mode < 30, ]
mu_grid <- sort(unique(grids[, 1]))
sd2_grid <- sort(unique(grids[, 2]))

out1.stats.lag.lognorm <- dgtf_optimal_lag(
    mod1, sim1$y, "lba", opts1.mcs,
    mu_grid, sd2_grid
)

out1.stats.lag.lognorm <- out1.stats.lag.lognorm[!apply(out1.stats.lag.lognorm, 1, anyNA), ]
out1.stats.lag.lognorm <- as.data.frame(out1.stats.lag.lognorm)
colnames(out1.stats.lag.lognorm) <- c("mu", "sd2", "mean", "var", "mode", "forecast", "fit")
out1.stats.lag.lognorm$mode_int <- as.factor(round(out1.stats.lag.lognorm$mode))
lattice::contourplot(forecast ~ mu * sd2, data = out1.stats.lag.lognorm)

ggplot(out1.stats.lag.lognorm, aes(mu, sd2, fill = forecast)) +
    geom_tile()

p <- ggplot(data = data.frame(
    mode = c(out1.stats.lag.lognorm$mode_int),
    mfe = c(out1.stats.lag.lognorm$forecast)
), aes(x = mode, y = mfe)) +
    theme_light() +
    geom_boxplot() +
    xlab("Mode") +
    ylab("RMSE of one-step-ahead forecasting")
plot(p)

if (save_figures) {
    ggsave(
        file.path(
            opath, "model-selection",
            paste0("sim-", seed, "-lag-lognorm.pdf")
        ),
        plot = p, device = "pdf", dpi = 300,
        height = fig_height, width = fig_width
    )
}
#
#
#
#| label: optimal param for lognormal lags
#| eval: false

lognorm_param <- c(3, 3.5)
#
#
#
#
#
#| label: nbinom lag dist with lba
#| eval: false

mod1$model$obs_dist <- "poisson"
mod1$model$lag_dist <- "nbinom"
mod1$param$lag <- c(0.395, 6)
kappa_grid <- seq(from = 0.1, to = 0.9, by = 0.05)
r_grid <- seq(from = 1, to = 6, by = 1)

out1.stats.lag.nbinom <- dgtf_optimal_lag(
    mod1, sim1$y, "lba", opts1.lba,
    kappa_grid, r_grid
)

out1.stats.lag.nbinom <- out1.stats.lag.nbinom[!apply(out1.stats.lag.nbinom, 1, anyNA), ]
out1.stats.lag.nbinom <- as.data.frame(out1.stats.lag.nbinom)
colnames(out1.stats.lag.nbinom) <- c("kappa", "r", "mean", "var", "mode", "forecast", "fit")
out1.stats.lag.nbinom$mode_int <- as.factor(round(out1.stats.lag.nbinom$mode))


lattice::contourplot(forecast ~ kappa * r, data = out1.stats.lag.nbinom)

ggplot(out1.stats.lag.nbinom, aes(kappa, r, fill = forecast)) +
    geom_tile()

p <- ggplot(data = data.frame(
    mode = c(out1.stats.lag.nbinom$mode_int),
    mfe = c(out1.stats.lag.nbinom$forecast)
), aes(x = mode, y = mfe)) +
    theme_light() +
    geom_boxplot() +
    xlab("Mode") +
    ylab("RMSE of one-step-ahead forecasting")
plot(p)


if (save_figures) {
    ggsave(
        file.path(
            opath, "model-selection",
            paste0("sim-", seed, "-lag-nbinom.pdf")
        ),
        plot = p, device = "pdf", dpi = 300,
        height = fig_height, width = fig_width
    )
}
#
#
#
#
#| label: save model selection
#| eval: false

if (save_data) {
    save(
        mod1, sim1,
        out1.stats.obs,
        # out1.stats.obs2,
        out1.stats.lag.nbinom,
        out1.stats.lag.lognorm,
        file = file.path(
            opath, "data",
            paste0("sim-", seed, ".RData")
        )
    )
}
#
#
#
#
#
#
#| eval: false

opts1.lba <- dgtf_default_algo_settings("lba")
opts1.lba$W <- 0.01
opts1.lba$use_discount <- FALSE



out1.lba2 <- dgtf_infer(
    mod1, sim1$y,
    "lba", opts1.lba,
    forecast_error = eval_forecast_err,
    fitted_error = eval_fitted_err,
    k = kstep_forecast_err,
    loss_func = loss
)

#
#
#
#
#
#
#| label: run external methods
#| include: false

si_para <- lognorm2serial(1.386262, 0.3226017)
out1.epi <- epi_poisson(c(sim1$y), si_para[1], si_para[2])
out1.wt <- wt_poisson(c(sim1$y), si_para[1], si_para[2])
#
#
#
#| label: figure of external methods
p1 <- plot_ts_ci_single(out1.epi)
p2 <- plot_ts_ci_single(out1.wt)

plot(p1)
plot(p2)

p3 <- plot_ts_ci_multi(list(
    LBA = log(exp(out1.lba$fit$psi) + 1),
    # HVB = log(exp(out1.hva$fit$psi) + 1),
    FFBS = log(exp(out1.ffbs$fit$psi) + 1),
    PL = log(exp(out1.pl$fit$psi) + 1),
    MCS = log(exp(out1.mcs$fit$psi) + 1),
    EpiEstim = out1.epi,
    WT = out1.wt,
    True = log(exp(cbind(sim1$psi, sim1$psi, sim1$psi) + 1))
), legend.position = "top")

plot(p3)
#
#
#
#| label: k step forecast for EpiEstim
#| include: false
forecast1.epi <- external_forecast(
    mod1, si_para, c(sim1$y),
    mod1$param$err[1],
    mod1$param$obs[1],
    "EpiEstim",
    loss, kstep_forecast_err
)


if (save_data) {
    save(
        out1.epi, forecast1.epi,
        file = file.path(
            opath, "data",
            paste0("sim-", seed, "-epi.RData")
        )
    )
}
#
#
#
#| label: k step forecast for WT
#| include: false
forecast1.wt <- external_forecast(
    mod1, si_para, c(sim1$y),
    mod1$param$err[1],
    mod1$param$obs[1],
    "wt",
    loss, kstep_forecast_err
)


if (save_data) {
    save(
        out1.wt, forecast1.wt,
        file = file.path(
            opath, "data",
            paste0("sim-", seed, "-wt.RData")
        )
    )
}
#
#
#
#
#| label: figure of forcast comparison
dat <- data.frame(
    LBA = out1.lba$error$forecast$y_loss_all,
    MCS = out1.mcs$error$forecast$y_loss_all,
    FFBS = out1.ffbs$error$forecast$y_loss_all,
    PL = out1.pl$error$forecast$y_loss_all,
    # EPI = forecast1.epi$y_loss_all,
    WT = forecast1.wt$y_loss_all,
    k = c(1:kstep_forecast_err)
)

dat2 <- reshape2::melt(dat, id.vars = "k", value.name = "MFE", variable.name = "algo")

p <- ggplot(dat2, aes(x = k, y = MFE, group = algo, color = algo)) +
    geom_line() +
    theme_light()

plot(p)
#
#
#
#
#
#| label: forecasting and fitting error of hva
#| include: false
#| eval: false


opts1.hva <- dgtf_default_algo_settings("hva")
opts1.hva$nsample <- 5000
opts1.hva$nthin <- 2
opts1.hva$nburnin <- 5000
opts1.hva$num_step_ahead_forecast <- 10
opts1.hva$tstart_pct <- 0.85
opts1.hva$num_eval_forecast_error <- mod1$dim$ntime - kstep_forecast_err - mod1$dim$ntime * opts1.hva$tstart_pct + 1

opts1.hva$smc$num_particle <- 100
opts1.hva$smc$num_backward <- 1
opts1.hva$smc$do_smoothing <- TRUE


opts1.hva$learning_rate <- 0.02
opts1.hva$eps_step_size <- 1.e-5
opts1.hva$k <- length(param_infer)


opts1.hva$mu0 <- list(
    infer = FALSE,
    init = mod1$param$obs[1]
)

opts1.hva$W <- list(
    infer = TRUE,
    init = mod1$param$err[1],
    prior_name = "invgamma",
    prior_param = c(1, 1)
)

opts1.hva$rho$init <- mod1$param$obs[2]
opts1.hva$rho$infer <- FALSE

opts1.hva$par1 <- list(
    infer = FALSE,
    init = 2,
    prior_name = "gaussian",
    prior_param = c(0., 1.)
)

opts1.hva$par2 <- list(
    infer = FALSE,
    init = mod1$param$lag[2],
    prior_name = "invgamma",
    prior_param = c(1, 1)
)

out1.hva <- dgtf_infer(
    mod1, sim1$y,
    "hva", opts1.hva,
    forecast_error = TRUE,
    fitted_error = TRUE,
    k = kstep_forecast_err
)

# tag <- paste0("sim-", seed, "-hva-")
# plots <- plot_output(
#     out1.hva, sim1$y,
#     psi_true = sim1$psi, W_true = sim1$param$err[1], mu0_true = sim1$param$obs[1],
#     save_figures = save_figures, tag = tag, opath = opath,
#     plot_figures = plot_figures
# )


if (save_data) {
    save(
        out1.hva, opts1.hva,
        file = file.path(
            opath, "data",
            paste0("sim-", seed, "-hva-W-forecast.RData")
        )
    )
}
#
#
#
#| label: forecasting and fitting error of hva (mu0 and W)
#| include: false
#| eval: false

opts1.hva2 <- opts1.hva
opts1.hva2$mu0 <- list(
    infer = TRUE,
    init = mod1$param$obs[1]
)

opts1.hva2$k <- 2

mod2 <- mod1
mod2$dim$regressor_baseline <- FALSE

out1.hva2 <- dgtf_infer(
    mod2, sim1$y,
    "hva", opts1.hva2,
    forecast_error = TRUE,
    fitted_error = TRUE,
    k = kstep_forecast_err
)


tag <- paste0("sim-", seed, "-hva2-")
plots <- plot_output(
    out1.hva2, sim1$y,
    psi_true = sim1$psi, W_true = sim1$param$err[1], mu0_true = sim1$param$obs[1],
    save_figures = save_figures, tag = tag, opath = opath,
    plot_figures = plot_figures
)

if (save_data) {
    save(
        out1.hva, out1.hva2,
        opts1.hva, opts1.hva2,
        file = file.path(
            opath, "data",
            paste0("sim-", seed, "-hva.RData")
        )
    )
}
#
#
#
#
#| label: forecasting and fitting error of mcmc
#| include: false
#| eval: false



opts1.mcmc <- dgtf_default_algo_settings("mcmc")
opts1.mcmc$nsample <- 5000
opts1.mcmc$nthin <- 5
opts1.mcmc$nburnin <- 50000
opts1.mcmc$num_step_ahead_forecast <- 10

opts1.mcmc$mh_sd <- 0.1
opts1.mcmc$epsilon <- 0.005
opts1.mcmc$L <- 10
opts1.mcmc$m <- c(0.1, 0.1)

opts1.mcmc$mu0$init <- mod1$param$obs[1]
opts1.mcmc$mu0$infer <- "mu0" %in% param_infer
opts1.mcmc$mu0$mh_sd <- 0.1

opts1.mcmc$W$init <- mod1$param$err[1]
opts1.mcmc$W$infer <- "W" %in% param_infer

opts1.mcmc$rho$init <- mod1$param$obs[2]
opts1.mcmc$rho$infer <- "rho" %in% param_infer
opts1.mcmc$rho$mh_sd <- 0.5

opts1.mcmc$par1 <- list(
    infer = "par1" %in% param_infer,
    init = mod1$param$lag[1],
    prior_name = "gaussian",
    prior_param = c(0., 1.),
    mh_sd = 0.1
)

opts1.mcmc$par2 <- list(
    infer = "par2" %in% param_infer,
    init = mod1$param$lag[2],
    prior_name = "invgamma",
    prior_param = c(1, 1),
    mh_sd = 0.1
)

out1.mcmc <- dgtf_infer(
    mod1, sim1$y,
    "mcmc", opts1.mcmc,
    forecast_error = TRUE,
    fitted_error = TRUE,
    k = kstep_forecast_err
)

tag <- paste0("sim-", seed, "-mcmc-")
plots <- plot_output(
    out1.mcmc, sim1$y,
    psi_true = sim1$psi, W_true = sim1$param$err[1], mu0_true = sim1$param$obs[1],
    save_figures = save_figures, tag = tag, opath = opath,
    plot_figures = plot_figures
)

if (save_data) {
    save(
        out1.mcmc, opts1.mcmc,
        file = file.path(
            opath, "data",
            paste0("sim-", seed, "-mcmc-W-forecast.RData")
        )
    )
}
#
#
#
