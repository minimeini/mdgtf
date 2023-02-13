args = commandArgs(trailingOnly=TRUE)

pkgs = c("ggplot2", "gridExtra", "kableExtra", "reshape2")
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

source(file.path(repo,"model_info.R"))
source(file.path(repo,"test_model.R"))
source(file.path(repo,"sim_pois_dglm.R"))
source(file.path(repo,"vis_pois_dlm.R"))

Rcpp::sourceCpp(file.path(repo,"model_utils.cpp"),verbose=FALSE)
Rcpp::sourceCpp(file.path(repo,"lbe_poisson.cpp"),verbose=FALSE)
Rcpp::sourceCpp(file.path(repo,"pl_poisson.cpp"),verbose=FALSE)
Rcpp::sourceCpp(file.path(repo,"hva_poisson.cpp"),verbose=FALSE)
Rcpp::sourceCpp(file.path(repo,"predict_poisson.cpp"),verbose=FALSE)


# input arguments:
# - ModelCode
# - obs_type: 0 for NB or 1 for Poisson
# - eta_select
if (length(args)>=1) {
  ModelCode = as.integer(args[1])
} else {
  ModelCode = 1
}
if (length(args)>=2) {
  obs_type = as.integer(args[2])
} else {
  obs_type = 0
}
if (length(args)>=3) {
  eta_select = as.integer(strsplit(args[3],"")[[1]])
} else {
  eta_select = c(1,0,0,0)
}
if (length(args)>=5) {
  aw = as.numeric(args[4])*as.numeric(args[5])
  bw = as.numeric(args[5])
} else {
  aw = 1*1
  bw = 1
}
if (length(args)>=7) {
  amu = as.numeric(args[6])*as.numeric(args[7])
  bmu = as.numeric(args[7])
} else {
  amu = 0.1*1e1
  bmu = 1e1
}


nburnin = 100000
nthin = 5
nsample = 5000

delta_grid = seq(from=0.8,to=0.95,by=0.01)
y = unlist(as.vector(read.table(file.path(opath,"sample.csv"),sep=",",header=FALSE)))
n = length(y)
err_type = 0 # 0 - Gaussian N(0,W); 1 - Laplace; 2 - Cauchy; 3 - Left skewed normal
eta_prior_type = c(0,0,0,0)

opts = default_opts(n,ModelCode)
opts$W_prior = setNames(c(aw,bw),c("aw","bw"))
opts$mu0_prior = setNames(c(amu,bmu),c("amu","bmu"))

if (ModelCode %in% c(0,1,6,11,14,17)) { # Koyama
  p = opts$L
} else if (ModelCode %in% c(2,3,7,12,15,18)) { # Solow
  p = 3
} else if (ModelCode %in% c(4,5,8,10,13,16)) { # Koyck
  p = 2
}
opts$m0 = rep(0,p)
opts$C0 = diag(rep(0.1,p))

delta = get_optimal_delta(y,opts$ModelCode,delta_grid,
                          L=opts$L,rho=opts$rho,
                          mu0=opts$mu0,
                          ctanh=opts$ctanh,
                          alpha=opts$alpha,
                          m0_prior=opts$m0, C0_prior=opts$C0,
                          delta_nb=opts$delta_nb, 
                          obs_type=opts$obs_type)$delta_optim
opts$delta = max(delta)

if (eta_select[1]==1 & eta_select[2]==0) {
  fname = paste0("sample_model",ModelCode,"_obs",obs_type,
                 "_eta",paste(eta_select,collapse=""),
                 "_",args[4],"W",args[5],".pdf")
} else if (eta_select[2]==1) {
  fname = paste0("sample_model",ModelCode,"_obs",obs_type,
                 "_eta",paste(eta_select,collapse=""),
                 "_",args[4],"W",args[5],
                 "_",args[6],"mu0",args[7],".pdf")
}

p = test_model_real(y,opts,eta_select=eta_select,
                    eta_prior_type=eta_prior_type)
pdf(file.path(opath,fname))
for (pp in p) {plot(pp)}
dev.off()

rm(list=ls())
gc(verbose=FALSE,reset=TRUE,full=TRUE)