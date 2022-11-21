plot_sim = function(sim_dat) {
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
  n = length(sim_dat$y)
  tmp = data.frame(time=1:n, true=sim_dat$psi,y=sim_dat$y,
                   mt=c(lbe_dat$mt[1,-1]),
                   mt_lo=c(lbe_dat$mt[1,-1])-2*sqrt(c(lbe_dat$Ct[1,1,-1])),
                   mt_hi=c(lbe_dat$mt[1,-1])+2*sqrt(c(lbe_dat$Ct[1,1,-1])))
  
  # Gain (Filtered)
  p1 = ggplot(tmp,aes(x=time)) +
    geom_ribbon(aes(ymin=mt_lo,ymax=mt_hi),
                fill="royalblue",alpha=0.2) +
    geom_line(aes(y=mt),color="royalblue") +
    geom_line(aes(y=true)) + theme_bw() +
    xlab("Time") + ylab("Gain (Filtered)")
  
  
  tmp = data.frame(time=1:n, true=sim_dat$psi,
                   y=sim_dat$y,
                   mt=c(lbe_dat$ht[-1]),
                   mt_lo=c(lbe_dat$ht[-1])-2*sqrt(c(lbe_dat$Ht[-1])),
                   mt_hi=c(lbe_dat$ht[-1])+2*sqrt(c(lbe_dat$Ht[-1])))
  
  # Gain (Smoothed)
  p2 = ggplot(tmp,aes(x=time)) +
    geom_ribbon(aes(ymin=mt_lo,ymax=mt_hi),
                fill="royalblue",alpha=0.2) +
    geom_line(aes(y=mt),color="royalblue") +
    geom_line(aes(y=true)) + theme_bw() +
    xlab("Time") + ylab("Gain (Smoothed)")
  
  if (ModelCode == 0 | ModelCode == 1 | ModelCode == 6) {
    eta = get_eta_koyama(sim_dat$y,lbe_dat$mt,lbe_dat$Ct,ModelCode)
    tmp = data.frame(time=1:n, true=sim_dat$theta,
                     y=sim_dat$y,
                     mt=c(eta$mean[-1]),
                     mt_lo=c(eta$mean[-1])-2*sqrt(c(eta$var[-1])),
                     mt_hi=c(eta$mean[-1])+2*sqrt(c(eta$var[-1])))
  } else {
    tmp = data.frame(time=1:n, true=sim_dat$theta,
                     y=sim_dat$y,
                     mt=c(lbe_dat$mt[2,-1]),
                     mt_lo=c(lbe_dat$mt[2,-1])-2*sqrt(c(lbe_dat$Ct[2,2,-1])),
                     mt_hi=c(lbe_dat$mt[2,-1])+2*sqrt(c(lbe_dat$Ct[2,2,-1])))
  }
  
  # theta (Filtered)
  p3 = ggplot(tmp,aes(x=time)) +
    geom_ribbon(aes(ymin=mt_lo,ymax=mt_hi),
                fill="royalblue",alpha=0.2) +
    geom_line(aes(y=mt),color="royalblue") +
    geom_line(aes(y=true)) + theme_bw() +
    xlab("Time") + ylab("theta (Filtered)")
  
  
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
  
  return(list(p1,p2,p3,p4))
}



plot_mcmc = function(sim_dat,mcmc_dat,lbe_dat=NULL) {
  n = length(sim_dat$y)
  
  psi = apply(mcmc_dat$wt,2,cumsum)
  psi = as.data.frame(t(apply(psi,1,quantile,
                              c(0.025,0.5,0.975))))
  colnames(psi) = c("lobnd","est","hibnd")
  psi$true = sim_dat$psi
  
  if (!is.null(lbe_dat)) {
    psi$lbe = lbe_dat$ht[-1]
  }
  psi$time = 1:n
  
  p1 = ggplot(psi,aes(x=time)) +
    geom_line(aes(y=est,color="MCMC")) +
    geom_ribbon(aes(ymin=lobnd,ymax=hibnd,y=est),
                alpha=0.2, show.legend = FALSE, fill="royalblue")
  
  if (!is.null(lbe_dat)) {
    p1 = p1 + geom_ribbon(aes(ymin=lbe_dat$ht[-1]-2*sqrt(lbe_dat$Ht[-1]),
                              ymax=lbe_dat$ht[-1]+2*sqrt(lbe_dat$Ht[-1]),
                              y=est),
                          alpha=0.2,show.legend = FALSE, fill="black") +
      geom_line(aes(y=lbe,color="LBE (Smoothed)"))
  }
  
  p1 = p1 + geom_line(aes(y=true,color="True")) +
    labs(x="Time",y="Gain Factor (psi)", color="Legend") +
    scale_color_manual(values=c("black","royalblue","maroon")) +
    theme_light() +
    theme(legend.position="bottom")
  
  p2 = ggplot(data.frame(time=1:n,rate=mcmc_dat$wt_accept),
              aes(x=time,y=rate)) +
    geom_line() + theme_light() + labs(y="Acceptance Rate") +
    theme(axis.text.x = element_blank(),
          axis.title.x = element_blank())
  
  return(list(p1,p2))
}