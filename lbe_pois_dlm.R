mu=2.2204e-16
m = 4.7
gm=0.01
s= 2.9
pk.mu = log(m/sqrt(1+(s/m)^2))
pk.sg2 = log(1+(s/m)^2)

#function that computes \Phi_d in Koyama et al. (2021)
Pd_r=function(d,mu,sigmasq){
  phid=0.5*pracma::erfc(-(log(d)-mu)/sqrt(2*sigmasq))
  phid
}
#function that computes \phi_d in Koyama et al. (2021)
knl_r=function(t,mu,sd2){
  Pd_r(t,mu,sd2)-Pd_r(t-1,mu,sd2)
}


F_deriv=function(at,Fphi,Fy){
  L=length(Fphi)
  res=0
  Fy[Fy<=0] = 0.01/L
  res = Fphi * Fy * exp(at)
  sum(res)
}


Ft_vector=function(at,Fphi,Fy){
  L=length(Fphi)
  Fy[Fy<=0] = 0.01/L
  res = Fphi * Fy * exp(at)
  res
}


lbe_pois_dlm = function(y,delta=0.7744,L=30, W=NULL,Ftype=0){

  T = length(y)
  at=array(0,c(T,L))
  mt=array(0,c(T,L))
  ft=rep(0,T)
  qt=rep(1,T)
  Ft=array(0,c(T,L))
  Rt=array(1,c(T,L,L))
  Ct=array(1,c(T,L,L))
  gt=rep(0,T)
  pt=rep(1,T)
  beta_t=rep(1,T)
  alpha_t=rep(1,T)

  Wtill = array(0,c(L,L))
  if (!is.null(W)) Wtill[1,1] = W
  
  phis=knl_r(1:L,pk.mu,pk.sg2)

  m0=rep(0,L)
  C0=0.1*diag(L)
  G=matrix(0,L,L)
  G[1,1]=1
  G[2:L,1:(L-1)]=diag((L-1)) # a (L-1) x (L-1) identity matrix

  ytmp = rev(y[1:L])
  at[1,]=G%*%m0
  if (is.null(W)) {
    Rt[1,,]=G%*%C0%*%t(G)/delta
  } else {
    Rt[1,,] = G%*%C0%*%t(G) + Wtill
  }
  if (Ftype == 0) {
    Fa = exp(at[1,])
  } else {
    Fa = sapply(at[1,],max,0)
  }
  Ft_vec = phis * ytmp * Fa
  ft[1] = sum(Ft_vec)
  qt[1]=t(Ft_vec)%*%Rt[1,,]%*%Ft_vec
  if (qt[1]<.Machine$double.eps) {
    qt[1] = .Machine$double.eps
  }
  beta_t[1]=ft[1]/qt[1]
  alpha_t[1]=ft[1]*beta_t[1]
  gt[1]=(alpha_t[1]+y[1])/(beta_t[1]+1)
  pt[1]=(alpha_t[1]+y[1])/(beta_t[1]+1)^2
  mt[1,]=at[1,]+Rt[1,,]%*%Ft_vec*(gt[1]-ft[1])/qt[1]
  Ft_vec2 = rev(phis) * ytmp * Fa
  Ct[1,,]=Rt[1,,]-
    Rt[1,,]%*%Ft_vec%*%t(Ft_vec2)%*%Rt[1,,]*(1-pt[1]/qt[1])/qt[1]
  
  for (t in 1:L){
    at[t,] = at[1,]
    Rt[t,,] = Rt[1,,]
    ft[t] = ft[1]
    qt[t] = qt[1]
    gt[t] = gt[1]
    pt[t] = pt[1]
    mt[t,] = mt[1,]
    Ct[t,,]=Ct[1,,]
  }
  
  for (t in (L+1):T){
    ytmp = rev(y[(t-L):(t-1)])
    ytmp[ytmp<=0] = 0.01/L

    at[t,]=G%*%mt[t-1,]
    if (is.null(W)) {
      Rt[t,,]=G%*%Ct[t-1,,]%*%t(G)/delta
    } else {
      Rt[t,,]=G%*%Ct[t-1,,]%*%t(G) + Wtill
    }
    
    if (Ftype == 0) {
      Fa = exp(at[t,])
    } else {
      Fa = sapply(at[t,],max,0)
    }
    Ft_vec = phis * ytmp * Fa
    ft[t] = sum(Ft_vec)
    qt[t] = t(Ft_vec) %*% Rt[t,,] %*% Ft_vec
    if (qt[t]<.Machine$double.eps) {
      qt[t] = .Machine$double.eps
    }
    beta_t[t]=ft[t]/qt[t]
    alpha_t[t]=qt[t]*(beta_t[t])^2
    gt[t]=(alpha_t[t]+y[t])/(beta_t[t]+1)
    pt[t]=(alpha_t[t]+y[t])/(beta_t[t]+1)^2
    mt[t,] = at[t,] + Rt[t,,]%*%Ft_vec*(gt[t]-ft[t])/qt[t]
    Ft_vec2 = rev(phis) * ytmp * Fa
    Ct[t,,]=Rt[t,,]-
      Rt[t,,]%*%Ft_vec%*%t(Ft_vec2)%*%Rt[t,,]*(1-pt[t]/qt[t])/qt[t]
  }
  
  #smoothing? The moments are linear for the evolution equations 
  mt_smooth=array(0,c(T,L))
  Ct_smooth=array(0,c(T,L,L))
  mt_smooth[T,]=mt[T,]
  Ct_smooth[T,,]=Ct[T,,]
  for (t in (T-1):1){
    if (!is.null(W)) {
      # Bt = G %*% Ct[t,,] %*% t(G)
      # delta = Bt[1,1] / Rt[t+1,1,1]
      Bt = Ct[t,,] %*% t(G) %*% solve(Rt[t+1,,])
      mt_smooth[t,] = mt[t,] + Bt%*%(mt_smooth[t+1,] - at[t+1,])
    } else {
      mt_smooth[t,1] = mt[t,1] + delta*(mt_smooth[t+1,1] - mt[t,1])
    }    
  }
  
  rate = rep(0,T)
  h=0
  h_low=0
  h_up=0
  for (t in 1:T){
    if (t>1){
      if (Ftype == 0) {
        Fa = exp(mt_smooth[2:t,1])
      } else {
        Fa = sapply(mt_smooth[2:t,1],max,0)
      }
      h = sum(Fa*y[(1:t-1)]*knl_r((t-1):1,pk.mu,pk.sg2))
    }
    rate[t] = mu+h;
  }
  
  return(list(mt=mt,mt_smooth=mt_smooth,Ct=Ct,rate=rate,Rt=Rt))
}





plot_output = function(output,y) {
  nt = length(y)
  plot(exp(output$mt[2:nt,1]),xlab="time",
       ylab=expression(e^{psi(t)}),type='l')
  abline(h=1,col='red',lty=2)
  plot(output$rate[1:nt],type='l',ylim=c(0,max(c(y,output$rate))),
       col='darkgreen',ylab="Estimated cases/Cases",xlab="time",lwd=2)
  points(y[1:nt],cex=0.2)
}