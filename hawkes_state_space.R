###                          ###
# Monte Carlo Smoothing #
###                          ###
# Source: https://github.com/shigerushinomoto/COVID
# depends on `model_utils.cpp`


cauchyrnd=function(x0,gm,m){ # == `cauchyrnd`
  r = x0+gm*tan(pi*(runif(m)-0.5))
  r = matrix(r,nrow=1,ncol=m)
  r}



calc_rho = function(v,d=7){
  ns = length(v)
  np = ceiling((d-1)/2)
  v_pad = c(rep(0,np),v,rep(0,np))
  lm_bar = sapply(c(1:ns),function(i){sum(v_pad[c(i:(2*np+i))])/d})
  idx = lm_bar!=0
  rho = mean((v[idx]-lm_bar[idx])^2/lm_bar[idx]) - 1
  return(rho)
}


# knf=function(states,data,time,lags){
#   result=rep(0,N)
#   for (l in 1:N){ 
#     suma=0
#     for (t in (time-lags):(time-1)){
#       if (t>0){
#       suma=suma+data[t]*max(c(0,states[l,t]))*Pd(time-t,pk.mu,pk.sg2)}no
#     }
#     result[l]=suma
#   }
#   result
# }

dnb2 <- function(x,prob,size,log=FALSE) {
  lg <- lgamma(x+size)-lgamma(size)-lgamma(x+1)+size*log(prob)+x*log(1-prob)
  if (log) lg else exp(lg)
}

# Density of Negative Binomial
dnb3 <- function(x,lambda,rho,log=FALSE) {
  lg <- lgamma(x+(lambda/rho))-lgamma(x+1)-lgamma(lambda/rho)+(lambda/rho)*log(1/(1+rho))+x*log(rho/(1+rho))
  if (log) lg else exp(lg)
}

#set.seed(202201)

# cases=as.double(read.table('/Volumes/GoogleDrive/My Drive/Study_Resources/PhD/Research/Project1/sample.csv',sep=","))
# plot(cases,type='l')

# Bootstrap Particle Filtering
hawke_ss2 = function(
  cases,N=5000,
  L=30,
  rho=34.08792,
  W=0.01,
  obstype="nb", # either "pois" or "nb"
  errtype="cauchy") { # either "cauchy" or "normal"
  
  T=length(cases) # number of observations 
  F=array(0,c(L,L)) # L x L state transition, G, matrix
  F[1,]=c(1,rep(0,L-1))
  F[2:(L),1:(L-1)]=diag((L-1))
  # print(F)
  G=c(1,rep(0,(L-1))) # 1 x L state-to-obs vector, F

  # the lag
  mu=.Machine$double.eps
  m = 4.7
  s= 2.9
  pk.mu = log(m/sqrt(1+(s/m)^2))
  pk.sg2 = log(1+(s/m)^2)


  # storage
  theta = array(0,c(L,N)) #initial set of particles
  for (i in 1:L){
    theta[i,] = runif(N,0,10)
  }
  theta_stored = array(0,c(L,N,L)) # the last dimension is the length of backshifting

  # lambda=array(0,c(N,T))
  R=array(0,c(T,3))

  # debug
  # Rt_stored = array(0,c(T,N))
  # n0tmp = NULL

  Fphi = knl(1:L,pk.mu,pk.sg2)
  # print(Fphi)


  for (t in (1:T)){
    theta_stored[,,L]=theta 

    # n0 = (y[t-1],...,y[t-L]), aka reverse of y's recording order
    Fy = cases[seq(from=max(c(t-1,0)),to=max(t-L,0))] 
    if (t<=L) {Fy = c(c(Fy),rep(0,L-t+1))}

    # debug
    # n0tmp = rbind(n0tmp,n0)

    # Calculate 1 x N array of particles lambda[t](1),...,lambda[t](N)
    # in one-go via matrix operation
    # theta = R[t](1)        ...     R[t](N)
    #      .              .       .
    #      .              .       .
    #      .              .       .
    #      R[t-L+1](1)    ...     R[t-L+1](N)
    # theta is a L x N matrix
    # n0*knl = (phi[1]y[t-1], ..., phi[L]y[t-L]) is a 1 x L vector
    FF = matrix(Fy*Fphi, nrow=1,ncol=L)
    lambda = mu + FF%*%theta # 1 x N

    ###
    #    1. Resample lambda[t,i] for i=0,1,...,N, where i is the index of particles
    #    using likelihood P(y[t]|lambda[t,i])
    #    This is step 2-3 of Kitagawa and Sato
    ###
    if (obstype == "nb") {
      w = dnb3(cases[t],lambda,rho) # Pr(y[t] | lambda[t])
    } else if (obstype == "pois") {
      w = dpois(cases[t],lambda)
    }


    # This is step 2-4 or step 2-4S of Kitagawa and Sato
    # Backshifting is somehow the Monte Carlo Smoothing by Kitagawa and Sato
    if (sum(c(w))>0) {
      k = sample(1:N,N,replace=TRUE,prob=w)
      theta_stored = theta_stored[,k,]
    }
    
    
    # why is it sx[1,,1] not sx[1,,L]
    # ==> sx[1,,1] is the sx[1,,L] which is resampled/backshifted L times
    R[t,1] = median(theta_stored[1,,1])
    R[t,2] = quantile(theta_stored[1,,1],0.025)
    R[t,3] = quantile(theta_stored[1,,1],0.975)

    ###
    #    2. Propagate theta[t,i] to theta[t+1,i] for i=0,1,...,N 
    #    using state evolution distribution P(theta[t+1,i]|theta[t,i])
    ###

    # This is the step 2-1 by Kitagawa and Sato
    if (errtype == "cauchy") {
      err = cauchyrnd(0,W,N)
    } else if (errtype == "normal") {
      err = t(rnorm(N,0,sqrt(W)))
    }

    # This is the step 2-2 by Kitagawa and Sato
    theta = F%*%theta_stored[,,L] + G%*%err
    
    theta[theta<.Machine$double.eps] = .Machine$double.eps # Ramp function - Rt is nonnegative
#   aux=exp(aux)
    theta_stored[,,1:(L-1)]=theta_stored[,,2:L]
    # Rt_stored[t,] = theta[1,]
  }

  Rnew_median=R[,1]
  Rnew_lci=R[,2]
  Rnew_uci=R[,3]
  for(t in 1:(L-1)){
    Rnew_median = c(Rnew_median, median(theta_stored[1,,t]))
    Rnew_lci=c(Rnew_lci,quantile(theta_stored[1,,t],0.025))
    Rnew_uci=c(Rnew_uci,quantile(theta_stored[1,,t],0.975))
  }
  Rfinal_median= Rnew_median[(L+1):length(Rnew_median)]
  Rfinal_lci= Rnew_lci[(L+1):length(Rnew_lci)]
  Rfinal_uci= Rnew_uci[(L+1):length(Rnew_uci)]

  rate = rep(0,T)
  h=0 # the lambda
  h_low=0
  h_up=0
  for (t in 1:T){
    if (t>1){
      h = sum(Rfinal_median[1:(t-1)]*cases[(1:t-1)]*knl((t-1):1,pk.mu,pk.sg2))
    }
  rate[t] = mu+h;
  }

  return(cbind(Rfinal_lci,Rfinal_median,Rfinal_uci))
}



plot_hawkes_ss = function(cases,Rfinal_median,itvl,rate,x=NULL,plot_rate=TRUE,plot_r=TRUE) {
  Rfinal_lci = itvl[1,]
  Rfinal_uci = itvl[2,]

  nplot = sum(plot_rate+plot_r)
  par(mfrow=c(nplot,1))

  if (plot_r){ # reproduction number
    if (is.null(x)) {
      plot(Rfinal_median,type='l',ylim=c(0,5))
      lines(Rfinal_uci,lty=2)
      lines(Rfinal_lci,lty=2)
    } else {
      plot(x[-1],Rfinal_median,type='l',ylim=c(0,5))
      lines(x[-1],Rfinal_uci,lty=2)
      lines(x[-1],Rfinal_lci,lty=2)
    }
    
  }
  
  if (plot_rate) { # instataneous rate
    if (is.null(x)) {
      plot(rate,type='l',ylim=range(cases))
      points(cases)
    } else {
      plot(x[-1],rate,type='l',ylim=range(cases))
      points(x[-1],cases)
    }
  }
}


#   lambda[,t]=mu+knf(sx,cases,t,L)
#   w=dnbinom(cases[t],size=lambda[,t]/rho,prob=1/(1+rho))
#   newsample=sample(1:N,N,replace = TRUE,w)
#   sx[,t]=sx[newsample,t]
#   px=sx[,t]+cauchyrnd(0,gm,N)






