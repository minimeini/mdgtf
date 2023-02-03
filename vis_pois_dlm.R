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


plot_results = function(sim_dat=NULL,
                        mcs_output=NULL,
                        lbe_output=NULL,
                        hvb_output=NULL,
                        nburnin=2000,
                        tstart=0,
                        n=NA,# number of observations
                        plot_hpsi=FALSE, # if FALSE, plot psi by default
                        plot_coverage=TRUE,
                        plot_theta=TRUE) {
  if (is.null(mcs_output)&is.null(lbe_output)&is.null(hvb_output)) {
    return(NULL)
  }
  if (is.null(sim_dat)&&is.na(n)) {
    return(NULL)
  }
  ####
  ####
  
  if (!is.null(sim_dat)) {
    psi_vec = c(sim_dat$params$psi0,sim_dat$psi) # 1:(n+1) <=> 0:n
    if (plot_hpsi) {
      psi_vec = c(psi2hpsi(matrix(psi_vec,ncol=1),
                           sim_dat$params$ModelCode,
                           coef=sim_dat$params$ctanh))
    }
    if (tstart>0) {
      psi = cbind(psi_vec[-c(1:tstart)],tstart:length(sim_dat$y))
    } else { # tstart==0
      psi = cbind(psi_vec,0:length(sim_dat$y))
    }
    colnames(psi) = c("true","time")
    psi = as.data.frame(psi)
    
    if (plot_theta) {
      theta_vec = sim_dat$theta
      if (tstart>0) {
        theta = data.frame(true=theta_vec[-c(1:tstart)],
                           time=(tstart+1):length(sim_dat$y))
      } else {
        theta = data.frame(true=theta_vec,
                           time=1:length(sim_dat$y))
      }
    }
    ncol = 2
  } else {
    psi = data.frame(time=tstart:n)
    if (plot_theta) {theta=data.frame(time=tstart:n)}
    ncol = 1
  }
  
  ####
  if (!is.null(mcs_output)) {
    tmp = mcs_output$quantiles
    tmp2 = psi2hpsi(tmp,sim_dat$params$ModelCode,
                    coef=sim_dat$params$ctanh)
    if (plot_hpsi) { tmp = tmp2 }
    
    if (tstart>0) {tmp = tmp[-c(1:tstart),]}
    psi = cbind(psi,tmp)
    ncol = dim(psi)[2]
    colnames(psi)[(ncol-2):ncol] = c("mcs_lobnd", "mcs", "mcs_hibnd")
    
    
    if (plot_theta) {
      theta_tmp = hpsi2theta(tmp2,sim_dat$y,
                             sim_dat$params$ModelCode,
                             sim_dat$params$theta0,
                             sim_dat$params$alpha,
                             sim_dat$params$L,
                             sim_dat$params$rho)
      if (tstart>0) {theta_tmp = theta_tmp[-c(1:tstart),]}
      theta = cbind(theta,theta_tmp)
      ncol = dim(theta)[2]
      colnames(theta)[(ncol-2):ncol] = c("mcs_lobnd", "mcs", "mcs_hibnd")
    }
    
    
    if (plot_coverage) {
      psi$mcs_width = psi$mcs_hibnd - psi$mcs_lobnd
      if (plot_hpsi & !is.null(sim_dat$params$ctanh)) {
        psi$mcs_width = 100 * psi$mcs_width / sim_dat$params$ctanh[3]
      }
    }
  }
  ####
  if (!is.null(lbe_output)) {
    tmp = cbind(lbe_output$ht-2*sqrt(abs(lbe_output$Ht)),
                lbe_output$ht,
                lbe_output$ht+2*sqrt(abs(lbe_output$Ht)))
    tmp2 = psi2hpsi(tmp,sim_dat$params$ModelCode,
                    coef=sim_dat$params$ctanh)
    if (plot_hpsi) { tmp = tmp2 }
    if (tstart>0) {tmp = tmp[-c(1:tstart),]}
    psi = cbind(psi,tmp)
    ncol = dim(psi)[2]
    colnames(psi)[(ncol-2):ncol] = c("lbe_lobnd","lbe","lbe_hibnd")
    
    if (plot_theta) {
      theta_tmp = hpsi2theta(tmp2,sim_dat$y,
                             sim_dat$params$ModelCode,
                             sim_dat$params$theta0,
                             sim_dat$params$alpha,
                             sim_dat$params$L,
                             sim_dat$params$rho)
      if (tstart>0) {theta_tmp = theta_tmp[-c(1:tstart),]}
      theta = cbind(theta,theta_tmp)
      ncol = dim(theta)[2]
      colnames(theta)[(ncol-2):ncol] = c("lbe_lobnd", "lbe", "lbe_hibnd")
    }
    
    if (plot_coverage) {
      psi$lbe_width = psi$lbe_hibnd - psi$lbe_lobnd
      if (plot_hpsi & !is.null(sim_dat$params$ctanh)) {
        psi$lbe_width = 100 * psi$lbe_width / sim_dat$params$ctanh[3]
      }
    }
  }
  ####
  if (!is.null(hvb_output)) {
    tmp = hvb_output$psi_stored[,-c(1:nburnin)]
    tmp2 = psi2hpsi(tmp,sim_dat$params$ModelCode,
                    coef=sim_dat$params$ctanh)
    if (plot_hpsi) { tmp = tmp2 }
    tmp = t(apply(tmp,1,quantile,c(0.025,0.5,0.975)))
    if (tstart>0) {tmp = tmp[-c(1:tstart),]}
    psi = cbind(psi,tmp)
    ncol = dim(psi)[2]
    colnames(psi)[(ncol-2):ncol] = c("hvb_lobnd","hvb","hvb_hibnd")
    
    if (plot_theta) {
      theta_tmp = hpsi2theta(tmp2,sim_dat$y,
                             sim_dat$params$ModelCode,
                             sim_dat$params$theta0,
                             sim_dat$params$alpha,
                             sim_dat$params$L,
                             sim_dat$params$rho)
      theta_tmp = t(apply(theta_tmp,1,quantile,c(0.025,0.5,0.975)))
      if (tstart>0) {theta_tmp = theta_tmp[-c(1:tstart),]}
      theta = cbind(theta,theta_tmp)
      ncol = dim(theta)[2]
      colnames(theta)[(ncol-2):ncol] = c("hvb_lobnd", "hvb", "hvb_hibnd")
    }
    
    if (plot_coverage) {
      psi$hvb_width = psi$hvb_hibnd - psi$hvb_lobnd
      if (plot_hpsi & !is.null(sim_dat$params$ctanh)) {
        psi$hvb_width = 100 * psi$hvb_width / sim_dat$params$ctanh[3]
      }
    }
  }
  psi = as.data.frame(psi)
  ####
  ####
  labs = NULL
  cols = NULL
  p = ggplot(psi,aes(x=time))
  p2 = ggplot(psi,aes(x=time))
  p3 = ggplot(theta,aes(x=time))
  
  if (!is.null(sim_dat)) {
    labs = c(labs,"True")
    cols = c(cols,"black")
    p = p + geom_line(aes(y=true,color="black"), na.rm=TRUE)
    
    if (plot_theta) {
      p3 = p3 + geom_line(aes(y=true,color="black"), na.rm=TRUE)
    }
  }
  
  if (!is.null(mcs_output)) {
    labs = c(labs,"MCS")
    cols = c(cols,"royalblue")
    
    p = p + geom_line(aes(y=mcs,color="royalblue"), na.rm=TRUE) +
      geom_ribbon(aes(y=mcs,ymin=mcs_lobnd,
                      ymax=mcs_hibnd,fill="royalblue"), 
                  alpha=0.2,na.rm=TRUE)
    
    if (plot_coverage) {
      p2 = p2 + geom_area(aes(y=mcs_width,fill="royalblue"),
                          alpha=0.2,na.rm=TRUE)
    }
    if (plot_theta) {
      p3 = p3 + geom_line(aes(y=mcs,color="royalblue"), na.rm=TRUE) +
        geom_ribbon(aes(y=mcs,ymin=mcs_lobnd,
                        ymax=mcs_hibnd,fill="royalblue"), 
                    alpha=0.2,na.rm=TRUE)
    }
  }
  ####
  if (!is.null(lbe_output)) {
    labs = c(labs,"LBE")
    cols = c(cols,"maroon")
    p = p + geom_line(aes(y=lbe,color="maroon"), na.rm=TRUE) +
      geom_ribbon(aes(y=lbe,ymin=lbe_lobnd,
                      ymax=lbe_hibnd,fill="maroon"), 
                  alpha=0.2,na.rm=TRUE) 
    
    if (plot_coverage) {
      p2 = p2 + geom_area(aes(y=lbe_width,fill="maroon"),
                          alpha=0.2,na.rm=TRUE)
    }
    
    if (plot_theta) {
      p3 = p3 + geom_line(aes(y=lbe,color="maroon"), na.rm=TRUE) +
        geom_ribbon(aes(y=lbe,ymin=lbe_lobnd,
                        ymax=lbe_hibnd,fill="maroon"), 
                    alpha=0.2,na.rm=TRUE)
    }
  }
  ####
  if (!is.null(hvb_output)) {
    labs = c(labs,"HVB")
    cols = c(cols,"purple")
    p = p + geom_line(aes(y=hvb,color="purple"), na.rm=TRUE) +
      geom_ribbon(aes(y=hvb,ymin=hvb_lobnd,
                      ymax=hvb_hibnd,fill="purple"), 
                  alpha=0.2,na.rm=TRUE)
    
    if (plot_coverage) {
      p2 = p2 + geom_area(aes(y=hvb_width,fill="purple"),
                          alpha=0.2,na.rm=TRUE)
    }
    
    if (plot_theta) {
      p3 = p3 + geom_line(aes(y=hvb,color="purple"), na.rm=TRUE) +
        geom_ribbon(aes(y=hvb,ymin=hvb_lobnd,
                        ymax=hvb_hibnd,fill="purple"), 
                    alpha=0.2,na.rm=TRUE)
    }
  }
  ####
  p = p + theme_light() +
    scale_color_identity(name ="Method", breaks=cols, 
                         labels=labs, guide="legend") +
    scale_fill_identity(name = "Method", breaks=cols, labels=labs) +
    theme(legend.position = "bottom") + xlab("Time")
  if (plot_hpsi & !is.null(sim_dat$params$ctanh)) {
    p = p + scale_y_continuous(limits=c(0,sim_dat$params$ctanh[3])) +
      ylab("Reproduction Number")
  } else {
    p = p + ylab("Gain Factor")
  }
  
  if (plot_theta) {
    p3 = p3 + theme_light() +
      scale_color_identity(name ="Method", breaks=cols, 
                           labels=labs, guide="legend") +
      scale_fill_identity(name = "Method", breaks=cols, labels=labs) +
      theme(legend.position = "bottom") +
      ylab(expression(theta[t])) + xlab("Time")
  }
  
  
  if (plot_coverage) {
    p2 = p2 + theme_light() +
      scale_fill_identity(name = "Method", breaks=cols, labels=labs,
                          guide="legend") +
      theme(legend.position = "bottom") +
      ylab("Width of CI (%)") + xlab("Time")
  }
  
  
  if (plot_coverage & plot_theta) {
    return(list(p,p2,p3))
  } else if (plot_coverage) {
    return(list(p,p2))
  } else if (plot_theta) {
    return(list(p,p3))
  } else {
    return(p)
  }
}


plot_W = function(sim_dat,hvb_output,nburnin=2000) {
  plot(hvb_output$W_stored[-c(1:nburnin)],
       cex=0.5,col="mediumpurple1",
       xlab="Iteration",ylab="W")
  abline(h=median(hvb_output$W_stored[-c(1:nburnin)]),
         col="purple",lwd=2)
  abline(h=sim_dat$params$W,col="black",lwd=2)
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