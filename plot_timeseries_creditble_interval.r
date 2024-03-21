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
  return_figures = FALSE) {

  plots <- list()

  stopifnot(!is.null(out_list))
  stopifnot(!is.null(ytrue))
  if (save_figures) {
    stopifnot(!is.null(opath))

    if (is.null(tag)) {
      tag <- "output-"
    }
  }

  posterior_psi <- plot_ts_ci_single(out_list$fit$psi, psi_true, main = "Posterior distribution of psi")
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
      plot = posterior_psi, device = "pdf", dpi = 300
    )
  }



  if ("W" %in% names(out_list$fit)) {
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
        plot = posterior_W, device = "pdf", dpi = 300
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
        plot = posterior_mu0, device = "pdf", dpi = 300
      )
    }
  }


  if ("error" %in% names(out_list)) {
    if ("fitted" %in% names(out_list$error)) {
      if ("filter" %in% names(out_list$error$fitted)) {
        yfit_filter <- plot_ts_ci_single(out_list$error$fitted$filter$yhat, ytrue, main = "Fitted y after filtering")

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
            plot = yfit_filter, device = "pdf", dpi = 300
          )
        }
      }

      if ("smooth" %in% names(out_list$error$fitted)) {
        yfit_smooth <- plot_ts_ci_single(
          out_list$error$fitted$smooth$yhat, ytrue,
          main = "Fitted y after smoothing"
        )

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
            plot = yfit_smooth, device = "pdf", dpi = 300
          )
        }
      }

      if ("yhat" %in% names(out_list$error$fitted)) {
        yfit <- plot_ts_ci_single(out_list$error$fitted$yhat, ytrue, main = "Fitted y")

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
            plot = yfit, device = "pdf", dpi = 300
          )
        }
      }
    }

    if ("forecast" %in% names(out_list$error)) {
      kstep_forecast_err <- length(c(out_list$error$forecast$y_loss_all))

      forecast <- vector("list", length = kstep_forecast_err)
      for (j in 1:kstep_forecast_err) {
        forecast[[j]] <- plot_ts_ci_single(
          out_list$error$forecast$y_cast[c(1:(length(ytrue) - j)), , 2],
          ytrue[(j + 1):length(ytrue)],
          main = paste0(j, "step-ahead forecast of y")
        )

        if (plot_figures) {
          plot(forecast[[j]])
        }

        if (save_figures) {
          ggsave(
            file.path(
              opath, "forecast",
              paste0(tag, paste0("forecast-", j, "step.pdf"))
            ),
            plot = forecast[[j]], device = "pdf", dpi = 300
            # height = 4.51, width = 7.29
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
        plot = next10, device = "pdf", dpi = 300
        # height = 4.51, width = 7.29
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