get_posterior_summary = function(output, mod, param_infer) {
    qt = c(0.025, 0.5, 0.975)
    par_qt = NULL
    par_name = NULL
    par_true = NULL

    for (par in param_infer) {
        if (par == "alpha") {
            par_qt = rbind(
                par_qt,
                quantile(c(output$log_alpha), qt)
            )
            par_name = c(par_name, "log_alpha")
            par_true = c(par_true, log(mod$param$alpha))
        } else if (par == "car") {
            par_qt = rbind(
                par_qt,
                quantile(c(output$car_beta_mu), qt),
                quantile(c(output$car_beta_tau2), qt),
                quantile(c(output$car_beta_rho), qt),
                quantile(c(output$car_wt_tau2), qt),
                quantile(c(output$car_wt_rho), qt)
            )
            par_name = c(
                par_name,
                c(
                    "car_beta_mu", "car_beta_tau2", "car_beta_rho",
                    "car_wt_tau2", "car_wt_rho"
                )
            )
            par_true = c(par_true, mod$spatial$car_beta, mod$spatial$car_wt[2:3])
        } else if (par == "par1") {
            par_qt = rbind(
                par_qt,
                quantile(c(output$lag_par1), qt)
            )
            par_name = c(par_name, "lag_par1")
            par_true = c(par_true, mod$param$lag[1])
        } else if (par == "par2") {
            par_qt = rbind(
                par_qt,
                quantile(c(output$lag_par2), qt)
            )
            par_name = c(par_name, "lag_par2")
            par_true = c(par_true, mod$param$lag[2])
        } else if (par == "beta") {
            par_qt = rbind(
                par_qt,
                t(apply(output$log_beta, 1, quantile, qt))
            )
            par_name = c(
                par_name,
                paste("log_beta", 1:nrow(output$log_beta))
            )
            par_true = c(par_true, c(mod$param$spatial_effect_beta))
        } else if (par == "rho") {
            par_qt = rbind(
                par_qt,
                t(apply(output$rho, 1, quantile, qt))
            )
            par_name = c(
                par_name,
                paste("rho", 1:nrow(output$rho))
            )
            par_true = c(par_true, c(mod$param$obs))
        }
    }

    par_qt = data.frame(par_qt)
    par_qt$name = par_name
    par_qt$true_val = par_true
    par_qt = par_qt[, c("name", "true_val", "X50.", "X2.5.", "X97.5.")]
    colnames(par_qt) = c("name", "true_val", "est", "lobnd", "upbnd")

    return(par_qt)
}


get_mh_diagnostics = function(output) {
    mh_diagnostics = data.frame(
        param = c("global", "local", "log_beta"),
        accept_rate = c(
            output$global_accept_rate,
            mean(output$local_accept_rate),
            output$logbeta_accept_rate
        ),
        step_size = c(
            output$global_hmc_settings$leapfrog_step_size,
            mean(output$local_hmc_settings$leapfrog_step_size),
            output$logbeta_hmc_settings$leapfrog_step_size
        ),
        num_steps = c(
            output$global_hmc_settings$nleapfrog,
            mean(output$local_hmc_settings$nleapfrog),
            output$logbeta_hmc_settings$nleapfrog
        )
    )

    return(mh_diagnostics)
}


get_in_sample_fit_stats = function(error_stats, loc_name = NULL) {
    if (is.null(loc_name)) {
        loc_name = paste("location", 1:nrow(error_stats$Y_pred))
    }
    acc_stats = data.frame(
        location = loc_name,
        y_chisq = rep(error_stats$chi_sqr_avg, nrow(error_stats$Y_pred)),
        y_crps = c(error_stats$crps),
        y_coverage = c(error_stats$Y_coverage),
        y_width = c(error_stats$Y_width)
    )

    if ("Rt_mae" %in% names(error_stats)) {
        acc_stats$Rt_mae = c(error_stats$Rt_mae)
    }
    if ("Rt_coverage" %in% names(error_stats)) {
        acc_stats$Rt_coverage = c(error_stats$Rt_coverage)
    }
    if ("Rt_width" %in% names(error_stats)) {
        acc_stats$Rt_width = c(error_stats$Rt_width)
    }

    return(acc_stats)
}


get_gelman_rubin = function(file_list, param_infer) {
    output = vector("list", length(file_list))
    for (i in 1:length(file_list)) {
        tmp = new.env()
        load(file_list[i], envir = tmp)
        output[[i]] = tmp$out.mcmc
    }

    par_rhat = NULL
    par_name = NULL
    for (par in param_infer) {
        if (par == "alpha") {
            par_rhat = c(
                par_rhat, 
                gelman_rubin_cpp(
                    t(do.call(rbind, lapply(output, function(x) c(x$log_alpha))))
                )
            )
            par_name = c(par_name, "log_alpha")
        } else if (par == "car") {
            par_rhat = c(
                par_rhat,
                gelman_rubin_cpp(
                    t(do.call(rbind, lapply(output, function(x) c(x$car_beta_mu))))
                ),
                gelman_rubin_cpp(
                    t(do.call(rbind, lapply(output, function(x) c(x$car_beta_tau2))))
                ),
                gelman_rubin_cpp(
                    t(do.call(rbind, lapply(output, function(x) c(x$car_beta_rho))))
                ),
                gelman_rubin_cpp(
                    t(do.call(rbind, lapply(output, function(x) c(x$car_wt_tau2))))
                ),
                gelman_rubin_cpp(
                    t(do.call(rbind, lapply(output, function(x) c(x$car_wt_rho))))
                )
            )
            par_name = c(
                par_name,
                c(
                    "car_beta_mu", "car_beta_tau2", "car_beta_rho",
                    "car_wt_tau2", "car_wt_rho"
                )
            )
        } else if (par == "par1") {
            par_rhat = c(
                par_rhat, 
                gelman_rubin_cpp(
                    t(do.call(rbind, lapply(output, function(x) c(x$lag_par1))))
                )
            )
            par_name = c(par_name, "lag_par1")
        } else if (par == "par2") {
            par_rhat = c(
                par_rhat, 
                gelman_rubin_cpp(
                    t(do.call(rbind, lapply(output, function(x) c(x$lag_par2))))
                )
            )
            par_name = c(par_name, "lag_par2")
        } else if (par == "beta") {
            for (s in 1:nrow(output[[1]]$log_beta)) {
                par_rhat = c(
                    par_rhat, 
                    gelman_rubin_cpp(
                        t(do.call(rbind, lapply(output, function(x) c(x$log_beta[s,]))))
                    )
                )
                par_name = c(par_name, paste("log_beta", s))
            }
        } else if (par == "rho") {
            for (s in 1:nrow(output[[1]]$rho)) {
                par_rhat = c(
                    par_rhat, 
                    gelman_rubin_cpp(
                        t(do.call(rbind, lapply(output, function(x) c(x$rho[s,]))))
                    )
                )
                par_name = c(par_name, paste("rho", s))
            }
        }
    }

    return(data.frame(name = par_name, rhat = par_rhat))
}