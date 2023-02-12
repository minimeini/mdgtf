plot_sim = function(sim_dat) {
  require(ggplot2)
  
  n = length(sim_dat$y)
  p1 = ggplot(data.frame(Time=1:n, lambda=sim_dat$lambda, y=sim_dat$y), 
              aes(x=Time)) +
    geom_line(aes(y=lambda)) + geom_point(aes(y=y), size=0.5) +
    theme_light() + ylab("Observations and Intensity")
  
  p2 = ggplot(data.frame(Time=1:n,hpsi=sim_dat$hpsi),aes(x=Time,y=hpsi)) +
    geom_line() + ylab("Reproduction Number") +
    theme_light()
  
  if (!is.null(sim_dat$params$ctanh)) {
    p2 = p2 + scale_y_continuous(limits=c(0,sim_dat$params$ctanh[3]))
  }
  
  p3 = ggplot(data.frame(Time=1:n,psi=sim_dat$psi),aes(x=Time,y=psi)) +
    geom_line() + ylab("Gain Factor") + theme_light()
  
  return(list(p1,p2,p3))
  
}


plot_psi = function(sim_dat=NULL,
                    psi_list = NULL,
                    tstart=0,
                    opts=NULL,# number of observations
                    ytrue=NULL,
                    plot_hpsi=FALSE, # if FALSE, plot psi by default
                    plot_lambda=FALSE,
                    methods = NULL,
                    cols = NULL) {
  mlist = c("True","EpiEstim","WT","Koyama2021","MCS","LBE","HVB")
  clist = c("black","lightseagreen","orange","burlywood4","royalblue","maroon","purple")
  
  if (!is.null(sim_dat)&&("y" %in% names(sim_dat))) {
    y = sim_dat$y
  } else if (!is.null(ytrue)) {
    y = ytrue
  } else {
    y = NULL
  }
  
  if (plot_lambda) {plot_hpsi=TRUE}
  
  if (is.null(methods)) {
    methods = names(psi_list)
  }
  
  if (is.null(cols) & !is.null(sim_dat)) {
    cols = clist[1:(length(psi_list)+1)]
  } else if (is.null(cols)) {
    cols = clist[1:length(psi_list)]
  }
  ####
  ####

  if (!is.null(sim_dat)) {
    opts = sim_dat$params
    if (is.null(opts$ctanh)) {
      opts$ctanh = c(0.3,0,1)
    }
    psi_list$True = as.matrix(c(opts$psi0,sim_dat$psi),ncol=1)
    methods = c(methods,"True")
  }
  
  nl = length(psi_list)
  lambda_list = vector(mode="list",length=nl)
  TransferCode = get_transcode(opts$ModelCode)
  if (plot_hpsi) {
    for (i in 1:nl) {
      psi_list[[i]] = psi2hpsi(psi_list[[i]],
                               opts$ModelCode,
                               coef=opts$ctanh)
      if (plot_lambda) {
        lambda_list[[i]] = hpsi2theta(psi_list[[i]],y,
                                      TransferCode,
                                      opts$theta0,
                                      opts$alpha,
                                      opts$L,opts$rho)
        lambda_list[[i]] = lambda_list[[i]] + opts$mu0
      }
      
    }
    if (plot_lambda & !is.null(sim_dat) & "lambda" %in% names(sim_dat)) {
      lambda_list[[nl]] = matrix(sim_dat$lambda,ncol=1)
      names(lambda_list) = names(psi_list)
    }
  }
  
  for (i in 1:nl) {
    tmp = psi_list[[i]]
    if (tstart>0) {tmp = tmp[-c(1:tstart),]}
    psi_list[[i]] = as.data.frame(tmp)
    if (dim(psi_list[[i]])[2]>=3) {
      colnames(psi_list[[i]])[1:3] = c("lobnd", "est", "hibnd")
    } else {
      colnames(psi_list[[i]]) = "est"
    }
    
    psi_list[[i]]$time = (1:dim(psi_list[[i]])[1]) + tstart
    
    if (plot_lambda) {
      tmp = lambda_list[[i]]
      if (tstart>0) {tmp = tmp[-c(1:tstart),]}
      lambda_list[[i]] = as.data.frame(tmp)
      if (dim(lambda_list[[i]])[2]>=3) {
        colnames(lambda_list[[i]])[1:3] = c("lobnd", "est", "hibnd")
      } else {
        colnames(lambda_list[[i]]) = "est"
      }
      lambda_list[[i]]$time = (1:dim(lambda_list[[i]])[1]) + tstart
    }
  }
  
  ####
  p = ggplot(psi_list[[1]],aes(x=time,y=est)) +
    geom_line(col=cols[1],na.rm=TRUE)
  if ("lobnd"%in%colnames(psi_list[[1]]) & "hibnd"%in%colnames(psi_list[[1]])) {
    p = p + geom_ribbon(aes(ymin=lobnd,ymax=hibnd),fill=cols[1],
                        alpha=0.2,na.rm=TRUE)
  }
  if (nl>1) {
    for (i in 2:nl) {
      p = p + geom_line(col=cols[i],data=psi_list[[i]],na.rm=TRUE)
      if ("lobnd"%in%colnames(psi_list[[i]]) & "hibnd"%in%colnames(psi_list[[i]])) {
        p = p + geom_ribbon(aes(ymin=lobnd,ymax=hibnd),data=psi_list[[i]],
                            fill=cols[i],alpha=0.2,na.rm=TRUE)
      }
    }
  }
  
  
  p = p + theme_light() + xlab("Time")
  
  # p = p + 
  #   scale_color_identity(name ="Method", breaks=cols, 
  #                        labels=methods, guide="legend") +
  #   scale_fill_identity(name = "Method", breaks=cols, labels=methods) +
  #   theme(legend.position = "right")
  if (plot_hpsi) {
    p = p + ylab(expression(R[t])) +
      geom_hline(yintercept=1,col="black",linetype=2)
    # if (!is.null(opts$ctanh)) {
    #   p = p + scale_y_continuous(limits=c(0,opts$ctanh[3]))
    # }
  } else {
    p = p + ylab("Gain Factor")
  }
  
  if (plot_lambda) {
    p2 = ggplot(lambda_list[[1]],aes(x=time,y=est)) +
      geom_line(col=cols[1],na.rm=TRUE)
    if ("lobnd"%in%colnames(lambda_list[[1]]) & "hibnd"%in%colnames(lambda_list[[1]])) {
      p2 = p2 + geom_ribbon(aes(ymin=lobnd,ymax=hibnd),
                            fill=cols[1],alpha=0.2,na.rm=TRUE)
    }
    
    if (nl>1) {
      for (i in 2:nl) {
        p2 = p2 + geom_line(col=cols[i],data=lambda_list[[i]],na.rm=TRUE)
        if ("lobnd"%in%colnames(lambda_list[[i]]) & "hibnd"%in%colnames(lambda_list[[i]])) {
          p2 = p2 + geom_ribbon(aes(ymin=lobnd,ymax=hibnd),fill=cols[i],
                                data=lambda_list[[i]],alpha=0.2,na.rm=TRUE)
        }
      }
    }
    
    p2 = p2 + theme_light() +
      ylab(expression(lambda[t])) + xlab("Time")
    # p2 = p2 + 
    #   scale_color_identity(name ="Method", breaks=cols, 
    #                        labels=methods,guide="legend") +
    #   scale_fill_identity(name = "Method", breaks=cols, labels=methods) +
    #   geom_point(aes(y=y),data=data.frame(time=1:length(y),y=y),size=1,alpha=0.8) +
    #   theme(legend.position = "right")
  }
  
  
  ####
  
  
  if (plot_lambda) {
    return(list(p,p2))
  } else {
    return(list(p))
  }
}


plot_eta = function(vname,hvb_output,opts=NULL) {
  if (vname == "W") {
    tmp = data.frame(Iteration=1:length(hvb_output$W_stored),
                     W=hvb_output$W_stored)
    p = ggplot(tmp,aes(x=Iteration,y=W)) +
      geom_point(size=1,col="mediumpurple1") +
      geom_hline(yintercept=median(hvb_output$W_stored),
                 col="purple",linewidth=1) +
      theme_light() + ylab(expression(W))
    if (!is.null(opts) & ("W" %in% names(opts))) {
      p = p + geom_hline(yintercept=opts$W,col="black")
    }
  } else if (vname == "mu0") {
    tmp = data.frame(Iteration=1:length(hvb_output$mu0_stored),
                     mu0=hvb_output$mu0_stored)
    p = ggplot(tmp,aes(x=Iteration,y=mu0)) +
      geom_point(size=1,col="mediumpurple1") +
      geom_hline(yintercept=median(hvb_output$mu0_stored),
                 col="purple",linewidth=1) +
      theme_light() + ylab(expression(mu[0]))
    if (!is.null(opts) & ("mu0" %in% names(opts))) {
      p = p + geom_hline(yintercept=opts$mu0,col="black")
    }
  } else {
    stop("Variable not supported.")
  }
  
    
  return(p)
}


plot_pred = function(sim_dat,
                     m,
                     lbe_output = NULL,
                     mcs_output = NULL,
                     hvb_output = NULL,
                     nsample = 5000,
                     nplot = 20,
                     qProb = c(0.05,0.5,0.95),
                     type="y") { # type could be: "y", "psi", "lambda"
  
  n = length(sim_dat$y)
  if (type=="y") {
    ypred_dat = data.frame(true=c(sim_dat$y[(n-nplot):n],sim_dat$pred$y[1:m]),
                           time=(n-nplot):(n+m))
  } else if (type=="psi") {
    ypred_dat = data.frame(true=c(sim_dat$psi[(n-nplot):n],sim_dat$pred$psi[1:m]),
                           time=(n-nplot):(n+m))
  } else if (type=="lambda") {
    ypred_dat = data.frame(true=c(sim_dat$lambda[(n-nplot):n],sim_dat$pred$lambda[1:m]),
                           time=(n-nplot):(n+m))
  }
  
  ncol = 2
  if (!is.null(lbe_output)) {
    What = lbe_What(lbe_output$ht,
                    aw = 1.1,bw = 2*sim_dat$params$W)
    lbe_pred = predict_poisson(sim_dat$y[(n-nplot):n],m,5000,
                               sim_dat$params$ModelCode,
                               What,as.matrix(lbe_output$mt[,n+1],ncol=1),
                               L=sim_dat$params$L,
                               mu0=sim_dat$params$mu0,
                               delta_nb=sim_dat$params$delta_nb,
                               obs_type=sim_dat$params$obs_type,
                               qProb_=qProb)
    if (type=="y") {
      ypred_dat = cbind(ypred_dat,rbind(array(NA,c(nplot+1,3)),lbe_pred$ypred))
    } else if (type=="psi") {
      ypred_dat = cbind(ypred_dat,rbind(array(NA,c(nplot+1,3)),lbe_pred$psi))
    } else if (type=="lambda") {
      ypred_dat = cbind(ypred_dat,rbind(array(NA,c(nplot+1,3)),lbe_pred$lambda))
    }
    ncol = ncol + 3
    colnames(ypred_dat)[(ncol-2):ncol] = c("lbe_lobnd","lbe","lbe_hibnd")
  }
  
  
  if (!is.null(mcs_output)) {
    mcs_pred = predict_poisson(sim_dat$y[(n-nplot):n],m,
                               dim(mcs_output$theta_last)[2],
                               sim_dat$params$ModelCode,
                               sim_dat$params$W,
                               mcs_output$theta_last,
                               L=sim_dat$params$L,
                               mu0=sim_dat$params$mu0,
                               delta_nb=sim_dat$params$delta_nb,
                               obs_type=sim_dat$params$obs_type,
                               qProb_=qProb)
    if (type=="y") {
      ypred_dat = cbind(ypred_dat,rbind(array(NA,c(nplot+1,3)),mcs_pred$ypred))
    } else if (type=="psi") {
      ypred_dat = cbind(ypred_dat,rbind(array(NA,c(nplot+1,3)),mcs_pred$psi))
    } else if (type=="lambda") {
      ypred_dat = cbind(ypred_dat,rbind(array(NA,c(nplot+1,3)),mcs_pred$lambda))
    }
    ncol = ncol + 3
    colnames(ypred_dat)[(ncol-2):ncol] = c("mcs_lobnd","mcs","mcs_hibnd")
  }
  
  
  if (!is.null(hvb_output)) {
    hvb_pred = predict_poisson(sim_dat$y[(n-nplot):n],m,
                               dim(hvb_output$psi_last)[2],
                               sim_dat$params$ModelCode,
                               hvb_output$W_stored,
                               hvb_output$psi_last,
                               L=sim_dat$params$L,
                               mu0=sim_dat$params$mu0,
                               delta_nb=sim_dat$params$delta_nb,
                               obs_type=sim_dat$params$obs_type,
                               qProb_=qProb)
    if (type=="y") {
      ypred_dat = cbind(ypred_dat,rbind(array(NA,c(nplot+1,3)),hvb_pred$ypred))
    } else if (type=="psi") {
      ypred_dat = cbind(ypred_dat,rbind(array(NA,c(nplot+1,3)),hvb_pred$psi))
    } else if (type=="lambda") {
      ypred_dat = cbind(ypred_dat,rbind(array(NA,c(nplot+1,3)),hvb_pred$lambda))
    }
    ncol = ncol + 3
    colnames(ypred_dat)[(ncol-2):ncol] = c("hvb_lobnd","hvb","hvb_hibnd")
  }
  
  p = ggplot(ypred_dat,aes(x=time)) +
    geom_line(aes(y=true,color="black"))
  cols = "black"
    labs = "True"
    
    if (!is.null(lbe_output)) {
      p = p + geom_line(aes(y=lbe,color="maroon"),na.rm=TRUE) +
        geom_ribbon(aes(y=lbe,ymin=lbe_lobnd,ymax=lbe_hibnd),
                    fill="maroon",alpha=0.2,na.rm=TRUE)
      cols = c(cols,"maroon")
      labs = c(labs,"LBE")
    }
    if (!is.null(mcs_output)) {
      p = p + geom_line(aes(y=mcs,color="royalblue"),na.rm=TRUE) +
        geom_ribbon(aes(y=mcs,ymin=mcs_lobnd,ymax=mcs_hibnd),
                    fill="royalblue",alpha=0.2,na.rm=TRUE)
      cols = c(cols,"royalblue")
      labs = c(labs,"MCS")
    }
    if (!is.null(hvb_output)) {
      p = p + geom_line(aes(y=hvb,color="purple"),na.rm=TRUE) +
        geom_ribbon(aes(y=hvb,ymin=hvb_lobnd,ymax=hvb_hibnd),
                    fill="purple",alpha=0.2,na.rm=TRUE)
      cols = c(cols,"purple")
      labs = c(labs,"HVB")
    }
    
    p = p + theme_light() + xlab("Time") +
      scale_color_identity(name="Method",breaks=cols,labels=labs,guide="legend") +
      theme(legend.position = "bottom")
    
    if (type=="y") {
      p = p + ylab(expression(y[t]))
    } else if (type=="psi") {
      p = p + ylab(expression(psi[t]))
    } else if (type=="lambda") {
      p = p + ylab(expression(lambda[t]))
    }
    
    return(p)
}