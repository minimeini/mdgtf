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