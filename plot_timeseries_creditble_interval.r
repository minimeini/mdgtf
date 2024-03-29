plot_ts_ci_multi = function(
                    psi_list = NULL,
                    fontsize=20,
                    time_label=NULL,
                    legend.position="none",
                    xlab = "Time",
                    ylab = expression(psi[t]),
                    alpha = 0.2) {
  mlist_external = c("EpiEstim", "WT", "Koyama2021")
  clist_external = c("seagreen", "orange", "burlywood4")

  mlist_dgtf = c(
    "LBA", "LBE", 
    "HVB", "HVA", 
    "MCMC", 
    "True", 
    "VB", 
    "PL", 
    "SMCS-FL", "MCS",  
    "APF", 
    "SMCF-BF",
    "SMCS-BS", "BS", "FFBS"
  )
  clist_dgtf = c(
    rep("maroon", 2), # LBA, LBE
    rep("purple", 2), # HVB, HVA **
    "darkturquoise", # MCMC
    "black", # True
    "salmon", # VB
    "royalblue", # PL **
    rep("cornflowerblue", 2), # SMCS-FL, MCS **
    "gold", # APF
    "sandybrown", # SMCF-BF
    rep("mediumaquamarine", 3) # SMCS-BS
  )

  mlist = c(mlist_external, mlist_dgtf)
  clist = c(clist_external, clist_dgtf)
  

  nl = length(psi_list)
  n = max(c(unlist(lapply(psi_list,function(psi){dim(psi)[1]}))))
  nl2 = sum(names(psi_list)%in%c(mlist_dgtf, "Koyama2021"))
  
  if (is.null(time_label)) {
    time_label = c(1:n)
  } else {
    time_label = time_label[1:n]
  }


  for (i in 1:nl) {
    if (names(psi_list)[i] == "Koyama2021") {
      psi_list[[i]] = psi_list[[i]][-1,1:3]
    }

    psi_list[[i]] = data.frame(
      lobnd=psi_list[[i]][,1],
      est=psi_list[[i]][,2],
      hibnd=psi_list[[i]][,3],
      time=time_label[(n-dim(psi_list[[i]])[1]+1) : n]
    )
    psi_list[[i]]$method = rep(names(psi_list)[i], dim(psi_list[[i]])[1])
  }
  
  psi_list = do.call(rbind,psi_list)
  psi_list = as.data.frame(psi_list)

  
  ####
  methods = unique(psi_list$method)
  # cols = sapply(methods,function(m,mlist){which(m==mlist)},mlist)
  col_tmp = NULL
  for (m in methods) {
    ccctmp = which(m == mlist)
    col_tmp = c(col_tmp,ccctmp)
  }

  cols = clist[col_tmp]

  p = ggplot(psi_list,aes(x=time,y=est,group=method)) +
    geom_line(aes(color=method),na.rm=TRUE) +
    geom_ribbon(aes(ymin=lobnd,ymax=hibnd,fill=method),
                alpha=alpha,na.rm=TRUE) +
    scale_color_manual(name="Method",breaks=methods,values=cols) +
    scale_fill_manual(name="Method",breaks=methods,values=cols) +
    theme_minimal() + xlab(xlab) +
    theme(text=element_text(size=fontsize),
          legend.position=legend.position) +
    guides(colour = guide_legend(nrow = 2)) +
    ylab(ylab)
  

  return(p)

}


plot_ts_ci_single <- function(
    psi_quantile, psi_true = NULL,
    main = "Posterior",
    ylab = "psi",
    show_caption = FALSE) {
    ymin <- min(c(psi_quantile))
    ymax <- max(c(psi_quantile))

    dat <- data.frame(
        psi = psi_quantile[, 2],
        psi_min = psi_quantile[, 1],
        psi_max = psi_quantile[, 3],
        time = 0:(dim(psi_quantile)[1] - 1)
    )

    if (!is.null(psi_true)) {
        dat$true <- psi_true
    }

    p <- ggplot(data = dat, aes(x = time, y = psi)) +
        theme_light() +
        geom_line() +
        geom_ribbon(aes(ymin = psi_min, ymax = psi_max), alpha = 0.5, fill = "lightgrey") +
        ylab(ylab) +
        labs(title = main)

    if (!is.null(psi_true)) {
        p <- p + geom_line(
            aes(y = true, x = time),
            data = dat,
            color = "maroon", alpha = 0.8
        )
    }

    if (show_caption) {
        p <- p + labs(caption = "Red line represents true values, black line is the posterior median with a credible interval from the 2.5% quantile to 97.5% quantile.")
    }

    return(p)
}



plot_ypred <- function(
    ypred_quantile,
    yfit,
    start_time = 0.9) {
  
  if (!is.null(dim(yfit))) {
    nfit = dim(yfit)[1]
  } else {
    nfit = length(c(yfit))    
  }

  ntotal <- dim(ypred_quantile)[1] + nfit
  

  dat <- data.frame(
    y = ypred_quantile[, 2],
    ymin = ypred_quantile[, 1],
    ymax = ypred_quantile[, 3],
    time = nfit:(ntotal - 1)
  )

  dat0 <- data.frame(
    time = 0:(ntotal - 1),
    y = c(yfit, ypred_quantile[, 2])
  )

  tidx <- round(ntotal * 0.8):ntotal

  p <- ggplot(data = dat0[round(ntotal * 0.8):ntotal, ], aes(x = time, y = y)) +
    theme_light() +
    geom_line(col = "maroon") +
    geom_line(data = dat, aes(x = time, y = y)) +
    geom_ribbon(data = dat, aes(ymin = ymin, ymax = ymax), alpha = 0.5, fill = "lightgray")

  return(p)
}


plot_output <- function(
  out_list, ytrue, 
  psi_true = NULL, 
  W_true = NULL, 
  mu0_true = NULL, 
  save_figures = FALSE, 
  tag = NULL, 
  opath = NULL,
  plot_figures = TRUE,
  return_figures = FALSE,
  height = 10.1,
  width = 13.8) {

  plots <- list()

  stopifnot(!is.null(out_list))
  stopifnot(!is.null(ytrue))
  if (save_figures) {
    stopifnot(!is.null(opath))

    if (is.null(tag)) {
      tag <- "output-"
    }
  }

  posterior_psi <- plot_ts_ci_single(
    out_list$fit$psi, psi_true, 
    main = "Posterior distribution of psi") +
    ylab(expression(psi[t]))
  if (plot_figures) {
    plot(posterior_psi)
  }

  if (return_figures) {
    plots <- append(plots, list(posterior_psi = posterior_psi))
  }
  
  if (save_figures) {
    ggsave(
      file.path(
        opath, "posterior",
        paste0(tag, "posterior-psi.pdf")
      ),
      plot = posterior_psi, device = "pdf", dpi = 300,
      height = height, width = width
    )
  }



  if ("W" %in% names(out_list$fit)) {
    nd = length(dim(out_list$fit$W))
    if (nd == 1) {
      Wtmp = c(out_list$fit$W)
    } else if (nd == 2) {
      Wtmp = out_list$fit$W[, ncol(out_list$fit$w)]
    } else {
      Wtmp = NULL
    }
    posterior_W <- ggplot(
      data = data.frame(w = c(out_list$fit$W), idx = 1:length(out_list$fit$W)),
      aes(x = w)
    ) +
      theme_light() +
      geom_histogram(bins = 50) +
      labs(title = "Posterior distribution of W")

    if (!is.null(W_true)) {
      posterior_W <- posterior_W + geom_vline(aes(xintercept = W_true), col = "maroon")
    }

    if (plot_figures) {
      plot(posterior_W)
    }

    if (return_figures) {
      plots <- append(plots, list(posterior_W = posterior_W))
    }

    if (save_figures) {
      ggsave(
        file.path(
          opath, "posterior",
          paste0(tag, "posterior-W.pdf")
        ),
        plot = posterior_W, device = "pdf", dpi = 300,
        height = height, width = width
      )
    }
  }


  if ("mu0" %in% names(out_list$fit)) {
    posterior_mu0 <- ggplot(
      data = data.frame(mu0 = c(out_list$fit$mu0), idx = 1:length(out_list$fit$mu0)),
      aes(x = mu0)
    ) +
      theme_light() +
      geom_histogram(bins = 50) +
      labs(title = "Posterior distribution of mu0")

    if (!is.null(mu0_true)) {
      posterior_mu0 <- posterior_mu0 + geom_vline(aes(xintercept = mu0_true), col = "maroon")
    }

    if (plot_figures) {
      plot(posterior_mu0)
    }

    if (return_figures) {
      plots <- append(plots, list(posterior_mu0 = posterior_mu0))
    }

    if (save_figures) {
      ggsave(
        file.path(
          opath, "posterior",
          paste0(tag, "posterior-mu0.pdf")
        ),
        plot = posterior_mu0, device = "pdf", dpi = 300,
        height = height, width = width
      )
    }
  }


  if ("error" %in% names(out_list)) {
    if ("fitted" %in% names(out_list$error)) {
      if ("filter" %in% names(out_list$error$fitted)) {
        yfit_filter <- plot_ts_ci_single(
          out_list$error$fitted$filter$yhat, ytrue, 
          main = "Fitted y after filtering") + 
          ylab(expression(y[t]))

        if (plot_figures) {
          plot(yfit_filter)
        }

        if (return_figures) {
          plots <- append(plots, list(yfit_filter = yfit_filter))
        }

        if (save_figures) {
          ggsave(
            file.path(
              opath, "fit",
              paste0(tag, "fit-filter.pdf")
            ),
            plot = yfit_filter, device = "pdf", dpi = 300,
            height = height, width = width
          )
        }
      }

      if ("smooth" %in% names(out_list$error$fitted)) {
        yfit_smooth <- plot_ts_ci_single(
          out_list$error$fitted$smooth$yhat, ytrue,
          main = "Fitted y after smoothing"
        ) + ylab(expression(y[t]))

        if (plot_figures) {
          plot(yfit_smooth)
        }

        if (return_figures) {
          plots <- append(plots, list(yfit_smooth = yfit_smooth))
        }

        if (save_figures) {
          ggsave(
            file.path(
              opath, "fit",
              paste0(tag, "fit-smooth.pdf")
            ),
            plot = yfit_smooth, device = "pdf", dpi = 300,
            height = height, width = width
          )
        }
      }

      if ("yhat" %in% names(out_list$error$fitted)) {
        yfit <- plot_ts_ci_single(
          out_list$error$fitted$yhat, ytrue, main = "Fitted y") + 
          ylab(expression(y[t]))

        if (plot_figures) {
          plot(yfit)
        }

        if (return_figures) {
          plots <- append(plots, list(yfit = yfit))
        }

        if (save_figures) {
          ggsave(
            file.path(
              opath, "fit",
              paste0(tag, "fit.pdf")
            ),
            plot = yfit, device = "pdf", dpi = 300,
            height = height, width = width
          )
        }
      }
    }

    if ("forecast" %in% names(out_list$error)) {
      kstep_forecast_err <- length(c(out_list$error$forecast$y_loss_all))

      forecast <- vector("list", length = kstep_forecast_err)
      for (j in 1:kstep_forecast_err) {
        forecast[[j]] <- plot_ts_ci_single(
          out_list$error$forecast$y_cast[c(1:(length(ytrue) - j)), , j],
          ytrue[(j + 1):length(ytrue)],
          main = paste0(j, "step-ahead forecast of y")) +
          ylab(expression(y[t + k]))

        if (plot_figures) {
          plot(forecast[[j]])
        }

        if (save_figures) {
          ggsave(
            file.path(
              opath, "forecast",
              paste0(tag, paste0("forecast-", j, "step.pdf"))
            ),
            plot = forecast[[j]], device = "pdf", dpi = 300,
            height = height, width = width
          )
        }
      }

      if (return_figures) {
        plots <- append(plots, list(forecast = forecast))
      }
    }
  }


  if ("pred" %in% names(out_list)) {
    ypred_qt <- t(apply(out_list$pred$ypred, 1, quantile, c(0.025, 0.5, 0.975)))
    next10 <- plot_ypred(ypred_qt, ytrue) + labs(title = "Forecasting of the next 10 time points.")

    if (plot_figures) {
      plot(next10)
    }

    if (return_figures) {
      plots <- append(plots, list(next10 = next10))
    }

    if (save_figures) {
      ggsave(
        file.path(
          opath, "forecast",
          paste0(tag, paste0("forecast-next10.pdf"))
        ),
        plot = next10, device = "pdf", dpi = 300,
        height = height, width = width
      )
    }
  }


  return(plots)
}



print_loss_all <- function(out_list) {
  loss_all <- NULL
  old_names <- NULL

  if (!("error" %in% names(out_list))) {
    return(loss_all)
  }

  if ("forecast" %in% names(out_list$error)) {
    loss_all <- c(loss_all, out_list$error$forecast$y_loss_all)
    old_names <- c(old_names, paste0("forecast", 1:length(c(out_list$error$forecast$y_loss_all))))
  }

  if ("fitted" %in% names(out_list$error)) {
    if ("filter" %in% names(out_list$error$fitted)) {
      loss_all <- c(loss_all, c(out_list$error$fitted$filter$y_loss_all))
      old_names <- c(old_names, "fitted-filter")
    }

    if ("smooth" %in% names(out_list$error$fitted)) {
      loss_all <- c(loss_all, c(out_list$error$fitted$smooth$y_loss_all))
      old_names <- c(old_names, "fitted-smooth")
    }

    if ("y_loss_all" %in% names(out_list$error$fitted)) {
      loss_all <- c(loss_all, c(out_list$error$fitted$y_loss_all))
      old_names <- c(old_names, "fitted")
    }

    names(loss_all) <- old_names
  }

  return(loss_all)
}


get_dat_sim_loss_all = function(dat_env, type = "loss") {
  kstep = length(c(dat_env$out1.lba$error$forecast$y_loss_all))

  if ("mu0" %in% names(dat_env$out1.lba2$fit)) {
    mu0 = median(dat_env$out1.lba$fit$mu0)
  } else {
    mu0 = NA
  }
  discount_factor = dat_env$opts1.lba$custom_discount_factor
  W = dat_env$opts1.lba$W
  ytmp = rep(NA, kstep)
  if (type == "loss") {
    ytmp = dat_env$out1.lba$error$forecast$y_loss_all
  } else if (type == "coverage") {
    ytmp = dat_env$out1.lba$error$forecast$y_covered_all
  } else if (type == "width") {
    ytmp = dat_env$out1.lba$error$forecast$y_width_all
  }

  dat = data.frame(
    k = factor(c(1:kstep, "discount", "W", "mu0"), levels = c(1:kstep, "discount", "W", "mu0")),
    LBA.DF = c(ytmp, discount_factor, W, mu0)
  )

  if (exists("out1.lba2", dat_env)) {
    if ("mu0" %in% names(dat_env$out1.lba2$fit)) {
      mu0 = median(dat_env$out1.lba2$fit$mu0)
    } else {
      mu0 = NA
    }
    
    ytmp = rep(NA, kstep)
    if (type == "loss") {
      ytmp = dat_env$out1.lba2$error$forecast$y_loss_all
    } else if (type == "coverage") {
      ytmp = dat_env$out1.lba2$error$forecast$y_covered_all
    } else if (type == "width") {
      ytmp = dat_env$out1.lba2$error$forecast$y_width_all
    }
    
    dat$LBA.W = c(ytmp, discount_factor, W, mu0)
  }

  if (exists("out1.mcs", dat_env)) {
    if ("mu0" %in% names(dat_env$out1.mcs$fit)) {
      mu0 = median(dat_env$out1.mcs$fit$mu0)
    } else {
      mu0 = NA
    }
    discount_factor = dat_env$opts1.mcs$custom_discount_factor
    W = dat_env$opts1.mcs$W

    ytmp = rep(NA, kstep)
    if (type == "loss") {
      ytmp = dat_env$out1.mcs$error$forecast$y_loss_all
    } else if (type == "coverage") {
      ytmp = dat_env$out1.mcs$error$forecast$y_covered_all
    } else if (type == "width") {
      ytmp = dat_env$out1.mcs$error$forecast$y_width_all
    }

    dat$MCS.DF = c(ytmp, discount_factor, W, mu0)
  }

  if (exists("out1.mcs2", dat_env)) {
    if ("mu0" %in% names(dat_env$out1.mcs2$fit)) {
      mu0 = median(dat_env$out1.mcs2$fit$mu0)
    } else {
      mu0 = NA
    }

    ytmp = rep(NA, kstep)
    if (type == "loss") {
      ytmp = dat_env$out1.mcs2$error$forecast$y_loss_all
    } else if (type == "coverage") {
      ytmp = dat_env$out1.mcs2$error$forecast$y_covered_all
    } else if (type == "width") {
      ytmp = dat_env$out1.mcs2$error$forecast$y_width_all
    }

    dat$MCS.W = c(ytmp, discount_factor, W, mu0)
  }

  if (exists("out1.ffbs", dat_env)) {
    discount_factor = dat_env$opts1.ffbs$custom_discount_factor
    W = dat_env$opts1.ffbs$W
    if ("mu0" %in% names(dat_env$out1.ffbs$fit)) {
      mu0 = median(dat_env$out1.ffbs$fit$mu0[nrow(dat_env$out1.ffbs$fit$mu0), ])
    } else {
      mu0 = NA
    }
    
    ytmp = rep(NA, kstep)
    if (type == "loss") {
      ytmp = dat_env$out1.ffbs$error$forecast$y_loss_all
    } else if (type == "coverage") {
      ytmp = dat_env$out1.ffbs$error$forecast$y_covered_all
    } else if (type == "width") {
      ytmp = dat_env$out1.ffbs$error$forecast$y_width_all
    }

    dat$FFBS.DF = c(ytmp, discount_factor, W, mu0)
  }

  if (exists("out1.ffbs2", dat_env)) {
    if ("mu0" %in% names(dat_env$out1.ffbs2$fit)) {
      mu0 = median(dat_env$out1.ffbs2$fit$mu0[nrow(dat_env$out1.ffbs2$fit$mu0), ])
    } else {
      mu0 = NA
    }
    
    ytmp = rep(NA, kstep)
    if (type == "loss") {
      ytmp = dat_env$out1.ffbs2$error$forecast$y_loss_all
    } else if (type == "coverage") {
      ytmp = dat_env$out1.ffbs2$error$forecast$y_covered_all
    } else if (type == "width") {
      ytmp = dat_env$out1.ffbs2$error$forecast$y_width_all
    }

    dat$FFBS.W = c(ytmp, discount_factor, W, mu0)
  }

  if (exists("out1.pl", dat_env)) {
    if ("mu0" %in% names(dat_env$out1.pl$fit)) {
      mu0 = median(dat_env$out1.pl$fit$mu0[nrow(dat_env$out1.pl$fit$mu0), ])
    } else {
      mu0 = NA
    }
    discount_factor = NA
    W = median(dat_env$out1.pl$fit$W[, ncol(dat_env$out1.pl$fit$W)])
    
    ytmp = rep(NA, kstep)
    if (type == "loss") {
      ytmp = dat_env$out1.pl$error$forecast$y_loss_all
    } else if (type == "coverage") {
      ytmp = dat_env$out1.pl$error$forecast$y_covered_all
    } else if (type == "width") {
      ytmp = dat_env$out1.pl$error$forecast$y_width_all
    }

    dat$PL = c(ytmp, discount_factor, W, mu0)
  }

  if (exists("out1.hva", dat_env)) {
    if ("mu0" %in% names(dat_env$out1.hva$fit)) {
      mu0 = median(c(dat_env$out1.hva$fit$mu0))
    } else {
      mu0 = NA
    }
    discount_factor = NA
    W = median(dat_env$out1.hva$fit$W)

    yloss = rep(NA, kstep)
    if ("error" %in% names(dat_env$out1.hva)) {
      if ("forecast" %in% names(dat_env$out1.hva)) {
        yloss = rep(NA, kstep)
        if (type == "loss") {
          yloss = dat_env$out1.hva$error$forecast$y_loss_all
        } else if (type == "coverage") {
          yloss = dat_env$out1.hva$error$forecast$y_covered_all
        } else if (type == "width") {
          yloss = dat_env$out1.hva$error$forecast$y_width_all
        }
      }
    }
    
    dat$HVA = c(yloss, discount_factor, W, mu0)
  }

  if (exists("out1.mcmc", dat_env)) {
    if ("mu0" %in% names(dat_env$out1.mcmc$fit)) {
      mu0 = median(c(dat_env$out1.mcmc$fit$mu0))
    } else {
      mu0 = NA
    }
    discount_factor = NA
    W = median(dat_env$out1.mcmc$fit$W)

    yloss = rep(NA, kstep)
    if ("error" %in% names(dat_env$out1.mcmc)) {
      if ("forecast" %in% names(dat_env$out1.mcmc)) {
        yloss = rep(NA, kstep)
        if (type == "loss") {
          yloss = dat_env$out1.mcmc$error$forecast$y_loss_all
        } else if (type == "coverage") {
          yloss = dat_env$out1.mcmc$error$forecast$y_covered_all
        } else if (type == "width") {
          yloss = dat_env$out1.mcmc$error$forecast$y_width_all
        }
      }
    }
    
    dat$MCMC = c(yloss, discount_factor, W, mu0)
  }

  if (exists("forecast1.epi", dat_env)) {
    discount_factor = NA
    W = NA
    mu0 = NA

    ytmp = rep(NA, kstep)
    if (type == "loss") {
      ytmp = dat_env$forecast1.epi$y_loss_all
    } else if (type == "coverage") {
      ytmp = dat_env$forecast1.epi$y_covered_all
    } else if (type == "width") {
      ytmp = dat_env$forecast1.epi$y_width_all
    }
    dat$EPI = c(ytmp, discount_factor, W, mu0)
  }

  if (exists("forecast1.wt", dat_env)) {
    discount_factor = NA
    W = NA
    mu0 = NA

    ytmp = rep(NA, kstep)
    if (type == "loss") {
      ytmp = dat_env$forecast1.wt$y_loss_all
    } else if (type == "coverage") {
      ytmp = dat_env$forecast1.wt$y_covered_all
    } else if (type == "width") {
      ytmp = dat_env$forecast1.wt$y_width_all
    }
    dat$WT = c(dat_env$forecast1.wt$y_loss_all, discount_factor, W, mu0)
  }

  return(dat)
}


get_dat_real_loss_all <- function(dat_env, type = "loss") {
  kstep <- length(c(dat_env$out.lba$error$forecast$y_loss_all))

  if ("mu0" %in% names(dat_env$out.lba2$fit)) {
    mu0 <- median(dat_env$out.lba$fit$mu0)
  } else {
    mu0 <- NA
  }
  discount_factor <- dat_env$opts.lba$custom_discount_factor
  W <- dat_env$opts.lba$W

  ytmp = rep(NA, kstep)
  if (type == "loss") {
    ytmp = dat_env$out.lba$error$forecast$y_loss_all
  } else if (type == "coverage") {
    ytmp = dat_env$out.lba$error$forecast$y_covered_all
  } else if (type == "width") {
    ytmp = dat_env$out.lba$error$forecast$y_width_all
  }

  dat <- data.frame(
    k = factor(c(1:kstep, "discount", "W", "mu0"), levels = c(1:kstep, "discount", "W", "mu0")),
    LBA.DF = c(ytmp, discount_factor, W, mu0)
  )

  if (exists("out.lba2", dat_env)) {
    if ("mu0" %in% names(dat_env$out.lba2$fit)) {
      mu0 <- median(dat_env$out.lba2$fit$mu0)
    } else {
      mu0 <- NA
    }

    ytmp = rep(NA, kstep)
    if (type == "loss") {
      ytmp = dat_env$out.lba2$error$forecast$y_loss_all
    } else if (type == "coverage") {
      ytmp = dat_env$out.lba2$error$forecast$y_covered_all
    } else if (type == "width") {
      ytmp = dat_env$out.lba2$error$forecast$y_width_all
    }
    dat$LBA.W <- c(ytmp, discount_factor, W, mu0)
  }

  if (exists("out.mcs", dat_env)) {
    if ("mu0" %in% names(dat_env$out.mcs$fit)) {
      mu0 <- median(dat_env$out.mcs$fit$mu0)
    } else {
      mu0 <- NA
    }
    discount_factor <- dat_env$opts.mcs$custom_discount_factor
    W <- dat_env$opts.mcs$W

    ytmp = rep(NA, kstep)
    if (type == "loss") {
      ytmp = dat_env$out.mcs$error$forecast$y_loss_all
    } else if (type == "coverage") {
      ytmp = dat_env$out.mcs$error$forecast$y_covered_all
    } else if (type == "width") {
      ytmp = dat_env$out.mcs$error$forecast$y_width_all
    }
    dat$MCS.DF <- c(ytmp, discount_factor, W, mu0)
  }

  if (exists("out.mcs2", dat_env)) {
    if ("mu0" %in% names(dat_env$out.mcs2$fit)) {
      mu0 <- median(dat_env$out.mcs2$fit$mu0)
    } else {
      mu0 <- NA
    }

    ytmp = rep(NA, kstep)
    if (type == "loss") {
      ytmp = dat_env$out.mcs2$error$forecast$y_loss_all
    } else if (type == "coverage") {
      ytmp = dat_env$out.mcs2$error$forecast$y_covered_all
    } else if (type == "width") {
      ytmp = dat_env$out.mcs2$error$forecast$y_width_all
    }
    dat$MCS.W <- c(ytmp, discount_factor, W, mu0)
  }

  if (exists("out.ffbs", dat_env)) {
    discount_factor <- dat_env$opts.ffbs$custom_discount_factor
    W <- dat_env$opts.ffbs$W
    if ("mu0" %in% names(dat_env$out.ffbs$fit)) {
      mu0 <- median(dat_env$out.ffbs$fit$mu0[nrow(dat_env$out.ffbs$fit$mu0), ])
    } else {
      mu0 <- NA
    }

    ytmp = rep(NA, kstep)
    if (type == "loss") {
      ytmp = dat_env$out.ffbs$error$forecast$y_loss_all
    } else if (type == "coverage") {
      ytmp = dat_env$out.ffbs$error$forecast$y_covered_all
    } else if (type == "width") {
      ytmp = dat_env$out.ffbs$error$forecast$y_width_all
    }
    dat$FFBS.DF <- c(ytmp, discount_factor, W, mu0)
  }

  if (exists("out.ffbs2", dat_env)) {
    if ("mu0" %in% names(dat_env$out.ffbs2$fit)) {
      mu0 <- median(dat_env$out.ffbs2$fit$mu0[nrow(dat_env$out.ffbs2$fit$mu0), ])
    } else {
      mu0 <- NA
    }

    ytmp = rep(NA, kstep)
    if (type == "loss") {
      ytmp = dat_env$out.ffbs2$error$forecast$y_loss_all
    } else if (type == "coverage") {
      ytmp = dat_env$out.ffbs2$error$forecast$y_covered_all
    } else if (type == "width") {
      ytmp = dat_env$out.ffbs2$error$forecast$y_width_all
    }
    dat$FFBS.W <- c(ytmp, discount_factor, W, mu0)
  }

  if (exists("out.pl", dat_env)) {
    if ("mu0" %in% names(dat_env$out.pl$fit)) {
      mu0 <- median(dat_env$out.pl$fit$mu0[nrow(dat_env$out.pl$fit$mu0), ])
    } else {
      mu0 <- NA
    }
    discount_factor <- NA
    W <- median(dat_env$out.pl$fit$W[, ncol(dat_env$out.pl$fit$W)])

    ytmp = rep(NA, kstep)
    if (type == "loss") {
      ytmp = dat_env$out.pl$error$forecast$y_loss_all
    } else if (type == "coverage") {
      ytmp = dat_env$out.pl$error$forecast$y_covered_all
    } else if (type == "width") {
      ytmp = dat_env$out.pl$error$forecast$y_width_all
    }
    dat$PL <- c(ytmp, discount_factor, W, mu0)
  }

  if (exists("out.hva", dat_env)) {
    if ("mu0" %in% names(dat_env$out.hva$fit)) {
      mu0 <- median(c(dat_env$out.hva$fit$mu0))
    } else {
      mu0 <- NA
    }
    discount_factor <- NA
    W <- median(dat_env$out.hva$fit$W)
    yloss <- rep(NA, kstep)
    if ("error" %in% names(dat_env$out.hva)) {
      if ("forecast" %in% names(dat_env$out.hva)) {
        if (type == "loss") {
          yloss = dat_env$out.hva$error$forecast$y_loss_all
        } else if (type == "coverage") {
          yloss = dat_env$out.hva$error$forecast$y_covered_all
        } else if (type == "width") {
          yloss = dat_env$out.hva$error$forecast$y_width_all
        }
      }
    }

    dat$HVA <- c(yloss, discount_factor, W, mu0)
  }

  if (exists("out.mcmc", dat_env)) {
    if ("mu0" %in% names(dat_env$out.mcmc$fit)) {
      mu0 <- median(c(dat_env$out.mcmc$fit$mu0))
    } else {
      mu0 <- NA
    }
    discount_factor <- NA
    W <- median(dat_env$out.mcmc$fit$W)
    yloss <- rep(NA, kstep)
    if ("error" %in% names(dat_env$out.mcmc)) {
      if ("forecast" %in% names(dat_env$out.mcmc)) {
        if (type == "loss") {
          yloss = dat_env$out.mcmc$error$forecast$y_loss_all
        } else if (type == "coverage") {
          yloss = dat_env$out.mcmc$error$forecast$y_covered_all
        } else if (type == "width") {
          yloss = dat_env$out.mcmc$error$forecast$y_width_all
        }
      }
    }

    dat$MCMC <- c(yloss, discount_factor, W, mu0)
  }

  if (exists("forecast.epi", dat_env)) {
    discount_factor <- NA
    W <- NA
    mu0 <- NA

    ytmp = rep(NA, kstep)
    if (type == "loss") {
      ytmp = dat_env$forecast.epi$y_loss_all
    } else if (type == "coverage") {
      ytmp = dat_env$forecast.epi$y_covered_all
    } else if (type == "width") {
      ytmp = dat_env$forecast.epi$y_width_all
    }
    dat$EPI <- c(ytmp, discount_factor, W, mu0)
  }

  if (exists("forecast.wt", dat_env)) {
    discount_factor <- NA
    W <- NA
    mu0 <- NA

    ytmp = rep(NA, kstep)
    if (type == "loss") {
      ytmp = dat_env$forecast.wt$y_loss_all
    } else if (type == "coverage") {
      ytmp = dat_env$forecast.wt$y_covered_all
    } else if (type == "width") {
      ytmp = dat_env$forecast.wt$y_width_all
    }
    dat$WT <- c(ytmp, discount_factor, W, mu0)
  }

  return(dat)
}


subset_dat_loss_all = function(dat, kstep, type = "W") {
  dat_sub = data.frame(k = dat$k[1:kstep])

  if (type == "W") {
    cnames = c("LBA.W", "MCS.W", "FFBS.W", "PL", "HVA", "MCMC", "EPI", "WT")
  } else if (type == "Wt") {
    cnames = c("LBA.DF", "MCS.DF", "FFBS.DF", "EPI", "WT")
  } else if (type == "WvsWt") {
    cnames = c("LBA.DF", "MCS.DF", "FFBS.DF", "LBA.W", "MCS.W", "FFBS.W", "EPI", "WT")
  }

  for (cn in cnames) {
    if (cn %in% colnames(dat)) {
      ncol_old = ncol(dat_sub)
      dat_sub$new = c(dat[1:kstep, cn])
      colnames(dat_sub) = c(colnames(dat_sub)[1:ncol_old], cn)
    }
  }

  return(dat_sub)
}


plot_mfe <- function(dat, upbnd = NULL, fontsize = 16) {
  dat$k <- as.numeric(dat$k)
  if (!is.null(upbnd)) {
    dat[dat > upbnd] <- NA
  }

  dat$k <- factor(dat$k, levels = c(1:dim(dat)[1]))

  dat2 <- reshape2::melt(
    dat,
    id.vars = "k",
    variable.name = "Algorithm",
    value.name = "RMSE"
  )

  p <- ggplot(dat2, aes(x = as.factor(k), y = RMSE, group = Algorithm, color = Algorithm)) +
    theme_light() +
    geom_line(na.rm = TRUE) +
    geom_point(na.rm = TRUE) +
    xlab(paste0(expression(k), "-step-ahead forecasting")) +
    ylab("RMSE") +
    theme(legend.position = "top", text = element_text(size = fontsize)) +
    guides(color = guide_legend(nrow = 1))

  return(p)
}


get_samples = function(prior_name, prior_param, nsample, cnst_val = 0.) {
  if (prior_name == "gamma") {
    samples = rgamma(nsample, prior_param[1], rate = prior_param[2])
  } else if (prior_name == "invgamma") {
    samples = 1. / rgamma(nsample, prior_param[1], rate = prior_param[2])
  } else if (prior_name == "uniform") {
    samples = runif(nsample, prior_param[1], prior_param[2])
  } else if (prior_name == "normal") {
    samples = rnorm(nsample, prior_param[1], prior_param[2])
  } else if (prior_name == "constant") {
    samples = rep(cnst_val, nsample)
  }

  return(samples)
}


get_prior_sensitivity_dat_pl = function(out, opts, par_name, upbnd = 1.e3) {
  stopifnot(par_name %in% names(out$fit))

  samples = c(out$fit[[par_name]])
  samples[!is.finite(samples)] = NA
  nsample = length(samples)

  cnst_val = opts[[par_name]]$init

  prior_name = opts[[par_name]]$prior_name
  prior_param = opts[[par_name]]$prior_param
  priors = get_samples(prior_name, prior_param, nsample, cnst_val)
  priors[!is.finite(priors)] = NA
  priors[priors > upbnd] = NA
  priors[priors < 0] = NA

  opts_name = paste0(par_name, "_init")
  prior_name = opts[[opts_name]]$prior_name
  prior_param = opts[[opts_name]]$prior_param
  inits = get_samples(prior_name, prior_param, nsample, cnst_val)
  inits[!is.finite(inits)] = NA
  inits[inits > upbnd] = NA
  inits[inits < 0] = NA

  dat <- data.frame(
    val = c(samples, priors, inits),
    label = c(
      rep("sample", nsample),
      rep("prior", nsample),
      rep("init", nsample)
    )
  )

  return(dat)
}

plot_prior_sensitivity_pl = function(out, opts, par_name, upbnd = 1.e3, plot_init = TRUE) {
  dat = get_prior_sensitivity_dat_pl(out, opts, par_name, upbnd)
  dat = dat[!is.na(dat$val),]

  p <- ggplot(dat, aes(x = val)) +
    geom_bar(stat = "bin", data = subset(dat, label == "sample"), fill = "red", alpha = 0.2, bins = 100) +
    geom_bar(stat = "bin", data = subset(dat, label == "prior"), fill = "gray", alpha = 0.6, bins = 100) +
    theme_minimal()
  
  if (plot_init) {
    p = p + geom_bar(stat = "bin", data = subset(dat, label == "init"), fill = "royalblue", alpha = 0.2, bins = 100)
  }

  return(p)
}
