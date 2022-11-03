plot_sim = function(Y,X,E,v) {
  # require(ggplot2)
  # require(gridExtra)
  n = length(Y)
  dat = data.frame(Y=Y,X=X,theta=E,w=v,time=1:n)
  dat = reshape2::melt(dat,id.vars="time")
  
  p = ggplot(dat,aes(x=time,y=value,group=variable,color=variable)) +
    facet_wrap(vars(variable),scales="free",ncol=1) +
    geom_line() + theme_bw() + theme(legend.position="bottom")
  return(p)
}


plot_vt_sample = function(output,v) {
  n = length(v)
  vt_est = t(apply(output$v,1,quantile,c(0.025,0.5,0.975)))
  vt_est = as.data.frame(vt_est)
  colnames(vt_est) = c("lobnd", "vt", "hibnd")
  vt_est$Time = 1:n
  p = ggplot(vt_est,aes(x=Time)) + theme_bw() +
    geom_ribbon(aes(ymin=lobnd,ymax=hibnd),
                fill="royalblue",alpha=0.2) +
    geom_line(aes(y=vt),color="royalblue") +
    geom_line(aes(y=True),data=data.frame(True=v,Time=1:n))
  return(p)
}


plot_Et_sample = function(output,E,Fx,nsample) {
  n = length(E)
  Et_est = matrix(0,nrow=n,ncol=nsample)
  for (i in 1:nsample) {
    Et_est[,i] = c(Fx %*% as.matrix(output$v[,i],ncol=1))
  }
  Et_est = apply(Et_est,1,quantile,c(0.025,0.5,0.975))
  Et_est = as.data.frame(t(Et_est))
  colnames(Et_est) = c("lobnd","Et","hibnd")
  Et_est$Time = 1:n
  p = ggplot(Et_est,aes(x=Time,y=Et)) + theme_bw() +
    geom_ribbon(aes(ymin=lobnd,ymax=hibnd),
                alpha=0.2,fill="royalblue") +
    geom_line(color="royalblue") +
    geom_line(aes(x=time,y=true),
              data=data.frame(true=E,time=1:n))
  return(p)
}

