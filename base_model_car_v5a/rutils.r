get_posterior_summary = function(output, mod, param_infer) {
    qt = c(0.025, 0.5, 0.975)
    par_qt = NULL
    par_name = NULL
    par_true = NULL

    for (par in param_infer) {
        if (par == "intercept_sigma2") {
            par_qt = rbind(
                par_qt,
                quantile(c(output$intercept_sigma2), qt)
            )
            par_name = c(par_name, "intercept_sigma2")
            par_true = c(par_true, mod$intercept$sigma2)
        } else if (par == "coef_self_intercept") {
            par_qt = rbind(
                par_qt,
                quantile(c(output$coef_self_intercept), qt)
            )
            par_name = c(par_name, "coef_self_intercept")
            par_true = c(par_true, mod$coef_self$intercept)
        } else if (par == "coef_cross_intercept") {
            par_qt = rbind(
                par_qt,
                quantile(c(output$coef_cross_intercept), qt)
            )
            par_name = c(par_name, "coef_cross_intercept")
            par_true = c(par_true, mod$coef_cross$intercept)
        } else if (par == "rho") {
            par_qt = rbind(
                par_qt,
                t(apply(output$rho, 1, quantile, qt))
            )
            par_name = c(
                par_name,
                paste("rho", 1:nrow(output$rho))
            )
            par_true = c(par_true, mod$param$obs)
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
        param = c("global", "local"),
        accept_rate = c(
            output$global_accept_rate,
            mean(output$local_accept_rate)
        ),
        step_size = c(
            output$global_hmc_settings$leapfrog_step_size,
            mean(output$local_hmc_settings$leapfrog_step_size)
        ),
        num_steps = c(
            output$global_hmc_settings$nleapfrog,
            mean(output$local_hmc_settings$nleapfrog)
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

    if ("log_a_mae" %in% names(error_stats)) {
        acc_stats$log_a_mae = c(error_stats$log_a_mae)
    }
    if ("log_a_coverage" %in% names(error_stats)) {
        acc_stats$log_a_coverage = c(error_stats$log_a_coverage)
    }
    if ("log_a_width" %in% names(error_stats)) {
        acc_stats$log_a_width = c(error_stats$log_a_width)
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
        if (par == "intercept_sigma2") {
            par_rhat = c(
                par_rhat, 
                gelman_rubin_cpp(
                    t(do.call(rbind, lapply(output, function(x) c(x$intercept_sigma2))))
                )
            )
            par_name = c(par_name, "intercept_sigma2")
        } else if (par == "coef_self_intercept") {
            par_rhat = c(
                par_rhat, 
                gelman_rubin_cpp(
                    t(do.call(rbind, lapply(output, function(x) c(x$coef_self_intercept))))
                )
            )
            par_name = c(par_name, "coef_self_intercept")
        } else if (par == "coef_cross_intercept") {
            par_rhat = c(
                par_rhat, 
                gelman_rubin_cpp(
                    t(do.call(rbind, lapply(output, function(x) c(x$coef_cross_intercept))))
                )
            )
            par_name = c(par_name, "coef_cross_intercept")
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