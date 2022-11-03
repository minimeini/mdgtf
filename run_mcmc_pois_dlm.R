rm(list=ls())
library(Rcpp)
library(ggplot2)
library(gridExtra)
sourceCpp("/Users/meinitang/Repository/dlm/testing.cpp")
sourceCpp("/Users/meinitang/Repository/pois_dglm_inference/lbe_poisson.cpp")
sourceCpp("/Users/meinitang/Repository/pois_dglm_inference/mcmc_disturbance_poisson.cpp")
source("/Users/meinitang/Repository/pois_dglm_inference/lbe_pois_utils.R")
source("/Users/meinitang/Repository/pois_dglm_inference/vis_pois_dlm.R")

opath = "/Volumes/GoogleDrive/My Drive/Output/pois_dglm"


timestamp = as.numeric(Sys.time())

n = 200
rho = 0.9
Q = 0.001
E0 = 0

X = rep(0.5,n)
v = rnorm(n,mean=0,sd=sqrt(Q))
Fx = update_Fx1(n,rho,X)
E = update_Et(n,Fx,v,rho,E0)
lambda = exp(E)
Y = rpois(n,lambda)
fname = paste0("sim_",timestamp,".RData")
save(X,Y,E,v,file=file.path(opath,fname))

p = plot_sim(Y,X,E,v)
fname = paste0("sim_",timestamp,".png")
ggsave(file.path(opath,fname),plot=p,width=5,height=7)

delta_grid = seq(0.2,0.9,by=0.1)
rho_grid = seq(0.4,0.9,by=0.05)
ndelta = length(delta_grid)
delta_prob = rep(0,ndelta)
rho_sel = rep(0,ndelta)
m0 = c(0,0)
C0 = diag(c(0.1,0.1))

for (i in 1:ndelta) {
  delta = delta_grid[i]
  logprob = get_rho_prob(Y,X,delta,rho_grid,m0,C0)
  rho_idx = which.max(logprob)
  rho_sel[i] = rho_grid[rho_idx]
  delta_prob[i] = logprob[rho_idx]
}
# plot(delta_grid,delta_prob,type="l",
     # xlab="delta",ylab="logprob")

# delta = readline("Choose your delta: ")
# delta = as.numeric(delta)
# dev.off()

# delta = 0.6

logprob = get_rho_prob(Y,X,delta,rho_grid,m0,C0)
# plot(rho_grid,logprob,type="l")

rho_idx = which.max(logprob)
rho_hat = rho_grid[rho_idx]
output = lbe_poissonX(Y,X,rho_hat,delta,m0,C0)
wt_hat = diff(output$mt[2,-1])


av = 0
Rv = 0.01
v1Prior = c(av,Rv)
QPrior=c(1e-2,1)
Vxi = 0.15
E0Prior = c(0,0.01)
What = estimate_state_var(wt_hat,QPrior)

nburnin = 800000
nthin = 20
nsample = 10000
output = mcmc_disturbance_pois(Y,X,
                               E0Prior = E0Prior,
                               Vxi = Vxi,
                               QPrior = QPrior,
                               v1Prior = v1Prior,
                               E0_init = E0,
                               rho_init = rho_hat,
                               vt_init = c(0,wt_hat),
                               Q_init = What,
                               nburnin = nburnin,
                               nthin = nthin,
                               nsample = nsample)
print(output$rho_accept)
print(output$E0_accept)
print(delta)


p = plot_vt_sample(output,v)
fname = paste0("mcmc_vt_",timestamp,".png")
ggsave(file.path(opath,fname),plot=p,width=7,height=7)


p = plot_Et_sample(output,E,Fx,nsample)
fname = paste0("mcmc_Et_",timestamp,".png")
ggsave(file.path(opath,fname),plot=p,width=7,height=7)


fname = paste0("mcmc_rho_",timestamp,".png")
png(filename=file.path(opath,fname))
hist(c(output$rho),breaks=50,
     xlim=c(min(c(rho,output$rho)),
            max(c(rho,output$rho))),
     main=paste0("rho ",round(output$rho_accept*100),"%"))
abline(v=rho,col="maroon")
dev.off()


fname = paste0("mcmc_E0_",timestamp,".png")
png(filename=file.path(opath,fname))
hist(c(output$E0),breaks=50,
     main=paste0("E0 ",round(output$E0_accept*100),"%"))
abline(v=E0,col="maroon")
dev.off()


fname = paste0("mcmc_Q_",timestamp,".png")
png(filename=file.path(opath,fname))
hist(c(output$Q),breaks=50)
abline(v=Q,col="maroon")
dev.off()





