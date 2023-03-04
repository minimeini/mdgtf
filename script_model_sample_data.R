args = commandArgs(trailingOnly=TRUE)

pkgs = c("ggplot2","gridExtra","reshape2","EpiEstim","dplyr","zoo")
installed_pkgs = pkgs %in% rownames(installed.packages())
if (any(installed_pkgs == FALSE)) {
  install.packages(pkgs[!installed_pkgs])
}

invisible(lapply(pkgs,library,character.only=TRUE))
rm(list=c("pkgs","installed_pkgs"))

# repo: location of source code
repo = "/Users/meinitang/Dropbox/Repository/poisson-dlm" 
# opath: location to stored output, if any
opath = "/Users/meinitang/Library/Mobile Documents/com~apple~CloudDocs/Research/Project1/sample" 


if (exists(".funcs")) {
  detach(.funcs)
  rm(.funcs)
}

.funcs = new.env()
source(file.path(repo,"model_info.R"),local=.funcs)
source(file.path(repo,"test_model.R"),local=.funcs)
source(file.path(repo,"sim_pois_dglm.R"),local=.funcs)
source(file.path(repo,"vis_pois_dlm.R"),local=.funcs)
source(file.path(repo,"hawkes_state_space.R"),local=.funcs)

Rcpp::sourceCpp(file.path(repo,"model_utils.cpp"),
                verbose=FALSE,env=.funcs)
Rcpp::sourceCpp(file.path(repo,"lbe_poisson.cpp"),
                verbose=FALSE,env=.funcs)
Rcpp::sourceCpp(file.path(repo,"pl_poisson.cpp"),
                verbose=FALSE,env=.funcs)
Rcpp::sourceCpp(file.path(repo,"vb_poisson.cpp"),
                verbose=FALSE,env=.funcs)
Rcpp::sourceCpp(file.path(repo,"hva_poisson.cpp"),
                verbose=FALSE,env=.funcs)
Rcpp::sourceCpp(file.path(repo,"mcmc_disturbance_poisson.cpp"),
                verbose=FALSE,env=.funcs)
Rcpp::sourceCpp(file.path(repo,"predict_poisson.cpp"),
                verbose=FALSE,env=.funcs)
attach(.funcs)

# input arguments:
# - ModelCode
# - obs_type: 0 for NB or 1 for Poisson
# - eta_select
if (length(args)>=1) {
  trans_func = as.character(args[1])
} else {
  trans_func = "koyama"
}

if (length(args)>=2) {
  gain_func = as.character(args[2])
} else {
  gain_func = "exponential"
}

if (length(args)>=3) {
  eta_select = as.integer(strsplit(args[3],"")[[1]])
} else {
  eta_select = c(1,0,0,0,0,0)
}
if (length(args)>=5) {
  aw = as.numeric(args[4])*as.numeric(args[5])
  bw = as.numeric(args[5])
} else {
  aw = 0.01*1e3
  bw = 1e3
}
if (length(args)>=7) {
  amu = as.numeric(args[6])*as.numeric(args[7])
  bmu = as.numeric(args[7])
} else {
  amu = 0.1*1e1
  bmu = 1e1
}
if (length(args)>=8) {
  ctanhM = as.numeric(args[8])
} else {
  ctanhM = 5
}

obs_dist = "nbinom"
link_func="identity"
err_dist="gaussian"

nburnin = 100000
nthin = 5
nsample = 5000

if (gain_func == "tanh") {
  Blag_pct = 0.1
} else if (gain_func == "logistic") {
  Blag_pct = 0.09
} else {
  Blag_pct = 0.15
}


delta_grid = seq(from=0.8,to=0.95,by=0.01)
y = unlist(as.vector(read.table(file.path(opath,"sample.csv"),sep=",",header=FALSE)))
n = length(y)
err_type = 0 # 0 - Gaussian N(0,W); 1 - Laplace; 2 - Cauchy; 3 - Left skewed normal
eta_prior_type = c(0,0,0,0,0,0)

opts = default_opts(length(y),
                    obs_dist=obs_dist,
                    link_func=link_func,
                    trans_func=trans_func,
                    gain_func=gain_func,
                    err_dist=err_dist)

opts$W_prior = setNames(c(aw,bw),c("aw","bw"))
opts$mu0_prior = setNames(c(amu,bmu),c("amu","bmu"))
opts$ctanh[3] = ctanhM


delta = get_optimal_delta(y,opts$model_code,delta_grid,
                          L=opts$L,rho=opts$rho,
                          mu0=opts$mu0,
                          ctanh=opts$ctanh,
                          alpha=opts$alpha,
                          delta_nb=opts$delta_nb)$delta_optim
opts$delta = max(delta)

if (eta_select[1]==1 & eta_select[2]==0) {
  fname = paste0("sample_",trans_func,"_",gain_func,
                 "_eta",paste(eta_select,collapse=""),
                 "_",aw,"W",bw,".pdf")
} else if (eta_select[2]==1) {
  fname = paste0("sample_",trans_func,"_",gain_func,
                 "_eta",paste(eta_select,collapse=""),
                 "_",aw,"W",bw,
                 "_",amu,"mu0",bmu,".pdf")
}

model_compare = c(1,1,0,1,0,1,1,1)
p = test_model_real(y,opts,eta_select=eta_select,
                    nburnin=nburnin,nthin=nthin,nsample=nsample,
                    eta_prior_type=eta_prior_type,
                    Blag_pct = Blag_pct,
                    model_compare=model_compare)
pdf(file.path(opath,fname))
for (pp in p) {plot(pp)}
dev.off()

rm(list=ls())
gc(verbose=FALSE,reset=TRUE,full=TRUE)