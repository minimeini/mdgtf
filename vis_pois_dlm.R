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
  
  p3 = ggplot(data.frame(Time=1:n,psi=sim_dat$psi),aes(x=Time,y=psi)) +
    geom_line() + ylab("Gain Factor") + theme_light()
  
  return(list(p1,p2,p3))
  
}


plot_lbe = function(sim_dat,lbe_dat,ModelCode) {
  require(ggplot2)
  
  plots = NULL

  n = length(sim_dat$y)
  tmp = data.frame(time=1:n, y=sim_dat$y,
                   mt=c(lbe_dat$mt[1,-1]),
                   mt_lo=c(lbe_dat$mt[1,-1])-2*sqrt(c(lbe_dat$Ct[1,1,-1])),
                   mt_hi=c(lbe_dat$mt[1,-1])+2*sqrt(c(lbe_dat$Ct[1,1,-1])))
  
  if (ModelCode<9) {
    tmp$true = sim_dat$psi
  } else {
    tmp$true = sim_dat$theta
  }
  
  # Gain (Filtered)
  p1 = ggplot(tmp,aes(x=time)) +
    geom_ribbon(aes(ymin=mt_lo,ymax=mt_hi),
                fill="royalblue",alpha=0.2) +
    geom_line(aes(y=mt),color="royalblue") +
    geom_line(aes(y=true)) + theme_bw() +
    xlab("Time")
  
  if (ModelCode<9) {
    p1 = p1 + ylab("Gain (Filtered)")
  } else {
    p1 = p1 + ylab("State (Filtered)")
  }
  
  tmp = data.frame(time=1:n, y=sim_dat$y,
                   mt=c(lbe_dat$ht[-1]),
                   mt_lo=c(lbe_dat$ht[-1])-2*sqrt(c(lbe_dat$Ht[-1])),
                   mt_hi=c(lbe_dat$ht[-1])+2*sqrt(c(lbe_dat$Ht[-1])))
  if (ModelCode<9) {
    tmp$true = sim_dat$psi
  } else {
    tmp$true = sim_dat$theta
  }
  
  # Gain (Smoothed)
  p2 = ggplot(tmp,aes(x=time)) +
    geom_ribbon(aes(ymin=mt_lo,ymax=mt_hi),
                fill="royalblue",alpha=0.2) +
    geom_line(aes(y=mt),color="royalblue") +
    geom_line(aes(y=true)) + theme_bw() +
    xlab("Time")
  
  if (ModelCode<9) {
    p2 = p2 + ylab("Gain (Smoothed)")
  } else {
    p2 = p2 + ylab("State (Smoothed)")
  }

  if (ModelCode == 0 | ModelCode == 1 | ModelCode == 6) {
    eta = get_eta_koyama(sim_dat$y,lbe_dat$mt,lbe_dat$Ct,ModelCode)
    tmp = data.frame(time=1:n, true=sim_dat$theta,
                     y=sim_dat$y,
                     mt=c(eta$mean[-1]),
                     mt_lo=c(eta$mean[-1])-2*sqrt(c(eta$var[-1])),
                     mt_hi=c(eta$mean[-1])+2*sqrt(c(eta$var[-1])))
  } else if (ModelCode < 9) {
    tmp = data.frame(time=1:n, true=sim_dat$theta,
                     y=sim_dat$y,
                     mt=c(lbe_dat$mt[2,-1]),
                     mt_lo=c(lbe_dat$mt[2,-1])-2*sqrt(c(lbe_dat$Ct[2,2,-1])),
                     mt_hi=c(lbe_dat$mt[2,-1])+2*sqrt(c(lbe_dat$Ct[2,2,-1])))
  }
  
  if (ModelCode < 9) {
    # theta (Filtered)
    p3 = ggplot(tmp,aes(x=time)) +
      geom_ribbon(aes(ymin=mt_lo,ymax=mt_hi),
                  fill="royalblue",alpha=0.2) +
      geom_line(aes(y=mt),color="royalblue") +
      geom_line(aes(y=true)) + theme_bw() +
      xlab("Time") + ylab("theta (Filtered)")
  }
  
  tmp = data.frame(true = sim_dat$wt,
                   # filtered = diff(lbe_dat$mt[1,]),
                   smoothed = diff(lbe_dat$ht),
                   time = 1:n)
  tmp = reshape2::melt(tmp,id.vars="time")
  
  # Evolution Error
  p4 = ggplot(tmp,aes(x=time,y=value,group=variable,color=variable)) +
    geom_line() + theme_light() + 
    theme(legend.position="bottom") +
    ylab("omega: evolution error")

  if (ModelCode<9) {
    return(list(p1,p2,p3,p4))
  } else {
    return(list(p1,p2,p4))
  }
}



plot_mcmc = function(sim_dat,mcmc_dat,lbe_dat=NULL) {
  require(ggplot2)
  
  n = length(sim_dat$y)
  nsample = length(mcmc_dat$W)
  
  if (ModelCode==9) { # Models with no external gain
    if (!is.null(lbe_dat)) {
      Fx = update_Fx(sim_dat$params$ModelCode,n,sim_dat$y,
                     sim_dat$params$rho,sim_dat$params$L,
                     lbe_dat$ht)
      theta0w = update_theta0(sim_dat$params$ModelCode,n,sim_dat$y,
                              sim_dat$params$theta0,sim_dat$params$psi0,
                              sim_dat$params$rho,sim_dat$params$L,lbe_dat$ht)
    } else {
      Fx = update_Fx(sim_dat$params$ModelCode,n,sim_dat$y,
                     sim_dat$params$rho,sim_dat$params$L,
                     c(0,sim_dat$psi))
      theta0w = update_theta0(sim_dat$params$ModelCode,n,sim_dat$y,
                              sim_dat$params$theta0,sim_dat$params$psi0,
                              sim_dat$params$rho,sim_dat$params$L,
                              c(0,sim_dat$psi))
    }
    
    theta = array(0,c(n,nsample))
    for (i in 1:nsample) {
      theta[,i] = theta0w + Fx %*% mcmc_dat$wt[,i]
    }
    dat = apply(theta,1,quantile,c(0.025,0.5,0.975))
    dat = as.data.frame(t(dat))
    dat$true = sim_dat$theta
    
  } else { # Models with external gain and thus psi
    psi = apply(mcmc_dat$wt,2,cumsum)
    dat = as.data.frame(t(apply(psi,1,quantile,c(0.025,0.5,0.975))))
    dat$true = sim_dat$psi
  }
  
  colnames(dat) = c("lobnd","est","hibnd","true")
  if (!is.null(lbe_dat)) {
    dat$lbe = lbe_dat$ht[-1]
  }
  dat$time = 1:n
  
  p1 = ggplot(dat,aes(x=time)) +
    geom_line(aes(y=est),color="royalblue") +
    geom_ribbon(aes(ymin=lobnd,ymax=hibnd,y=est),
                alpha=0.2, show.legend = FALSE, fill="royalblue")
  
  if (!is.null(lbe_dat)) {
    p1 = p1 + geom_ribbon(aes(ymin=lbe_dat$ht[-1]-2*sqrt(lbe_dat$Ht[-1]),
                              ymax=lbe_dat$ht[-1]+2*sqrt(lbe_dat$Ht[-1]),
                              y=lbe),
                          alpha=0.2,show.legend = FALSE, fill="black") +
      geom_line(aes(y=lbe),color="black")
  }
  
  p1 = p1 + geom_line(aes(y=true),color="maroon") +
    theme_light() + theme(legend.position="bottom")
  
  if (ModelCode == 9) {
    p1 = p1 + labs(x="Time",y="State", color="Legend") 
  } else {
    p1 = p1 + labs(x="Time",y="Gain Factor (psi)", color="Legend") 
  }
  
  # p2 = ggplot(data.frame(time=1:n,rate=mcmc_dat$wt_accept),
  #             aes(x=time,y=rate)) +
  #   geom_line() + theme_light() + labs(y="Acceptance Rate") +
  #   theme(axis.text.x = element_blank(),
  #         axis.title.x = element_blank())
  
  p3 = ggplot(data.frame(W=mcmc_dat$W)) +
    geom_histogram(aes(x=W),position="identity",bins=50,alpha=0.8) +
    geom_vline(xintercept=sim_dat$params$W,color="maroon") +
    theme_light() + xlab("W: Evolution Variance")
  
  if (!is.null(lbe_dat)) {
    p3 = p3 + geom_vline(xintercept=var(diff(lbe_dat$ht)),color="royalblue")
  }
  
  p4 = ggplot(data.frame(index=1:length(mcmc_dat$W),W=mcmc_dat$W),
              aes(x=index,y=W)) +
    geom_line() + geom_hline(yintercept=sim_dat$params$W,color="maroon") +
    theme_light() + xlab("Iteration") + ylab("W: Evolution Variance")
  
  return(list(p1,p3,p4))
}


plot_vb = function(sim_dat,vb_dat,lbe_dat=NULL) {
  require(ggplot2)
  
  n = length(sim_dat$y)
  
  psi = data.frame(lobnd=apply(vb_dat$ht[-1,],1,median)-2*sqrt(abs(apply(vb_dat$Ht[-1,],1,median))),
                   est=apply(vb_dat$ht[-1,],1,median),
                   hibnd=apply(vb_dat$ht[-1,],1,median)+2*sqrt(abs(apply(vb_dat$Ht[-1,],1,median))))
  
  if (sim_dat$params$ModelCode == 9) {
    psi$true = sim_dat$theta
  } else {
    psi$true = sim_dat$psi
  }
  
  
  if (!is.null(lbe_dat)) {
    psi$lbe = lbe_dat$ht[-1]
  }
  psi$time = 1:n
  
  p1 = ggplot(psi,aes(x=time)) +
    geom_line(aes(y=est,color="VB"),alpha=0.6) +
    geom_ribbon(aes(ymin=lobnd,ymax=hibnd,y=est),
                alpha=0.2, show.legend = FALSE, fill="royalblue")
  
  if (!is.null(lbe_dat)) {
    p1 = p1 + geom_ribbon(aes(ymin=lbe_dat$ht[-1]-2*sqrt(lbe_dat$Ht[-1]),
                              ymax=lbe_dat$ht[-1]+2*sqrt(lbe_dat$Ht[-1]),
                              y=est),
                          alpha=0.2,show.legend = FALSE, fill="black") +
      geom_line(aes(y=lbe,color="LBE (Smoothed)"),alpha=0.6)
  }
  
  p1 = p1 + geom_line(aes(y=true,color="True"),alpha=0.6) +
    labs(x="Time",y="Gain Factor (psi)", color="Legend") +
    scale_color_manual(values=c("black","royalblue","maroon")) +
    theme_light() +
    theme(legend.position="bottom")
  
  West = 1./rgamma(1000,vb_dat$aw, rate=vb_dat$bw)
  
  p3 = ggplot(data.frame(W=West)) +
    geom_histogram(aes(x=W),position="identity",bins=50,alpha=0.8) +
    theme_light() + xlab(paste0("W (true=",sim_dat$params$W,")"))
  
  if (!is.null(lbe_dat) & sim_dat$params$ModelCode != 9) {
    p3 = p3 + geom_vline(xintercept=var(diff(lbe_dat$ht)),color="royalblue")
  } else if (!is.null(lbe_dat) & sim_dat$params$ModelCode == 9) {
    wt_hat = rep(0,n)
    for (t in n:1) {
      wt_hat[t] = lbe_dat$ht[t+1] - sim_dat$params$rho*lbe_dat$ht[t]
    }
    p3 = p3 + geom_vline(xintercept=var(wt_hat),color="royalblue")
  }
  
  return(list(p1,p3))
}