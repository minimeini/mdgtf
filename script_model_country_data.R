args = commandArgs(trailingOnly=TRUE)

pkgs = c("ggplot2", "gridExtra", "kableExtra", "reshape2","EpiEstim","dplyr","zoo")
installed_pkgs = pkgs %in% rownames(installed.packages())
if (any(installed_pkgs == FALSE)) {
  install.packages(pkgs[!installed_pkgs])
}

invisible(lapply(pkgs,library,character.only=TRUE))
rm(list=c("pkgs","installed_pkgs"))

# repo: location of source code
repo = "/Users/meinitang/Dropbox/Repository/poisson-dlm" 
# opath: location to stored output, if any
opath = "/Users/meinitang/Library/Mobile Documents/com~apple~CloudDocs/Research/Project1" 

source(file.path(repo,"model_info.R"))
source(file.path(repo,"test_model.R"))
source(file.path(repo,"sim_pois_dglm.R"))
source(file.path(repo,"vis_pois_dlm.R"))
source(file.path(repo,"hawkes_state_space.R"))

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
if (length(args)>=8) {
  ctanhM = as.numeric(args[8])
} else {
  ctanhM = 5
}


delta_grid = seq(from=0.8,to=0.95,by=0.01)
err_type = 0 # 0 - Gaussian N(0,W); 1 - Laplace; 2 - Cauchy; 3 - Left skewed normal
eta_prior_type = c(0,0,0,0)

opts = default_opts(1,ModelCode)
opts$W_prior = setNames(c(aw,bw),c("aw","bw"))
opts$mu0_prior = setNames(c(amu,bmu),c("amu","bmu"))
opts$ctanh[3] = ctanhM
if (ModelCode %in% c(0,1,6,11,14,17)) { # Koyama
  p = opts$L
} else if (ModelCode %in% c(2,3,7,12,15,18)) { # Solow
  p = 3
} else if (ModelCode %in% c(4,5,8,10,13,16)) { # Koyck
  p = 2
}
opts$m0 = rep(0,p)
opts$C0 = diag(rep(0.1,p))

data = read.csv(file.path(opath,"country/owid-covid-data.csv"))
data$date = as.Date(data$date)
clist = unique(data$location)
dlist = sapply(clist,function(cnty){list(data[data$location==cnty,-1])})
rm(data)
beta = array(NA,dim=c(8,7))
i = 1
for (i in 1:length(dlist)) {
  dlist[[i]] = dlist[[i]] %>% mutate(new_cases2=zoo::na.approx(new_cases))
  wday = as.POSIXlt(dlist[[i]]$date)$wday + 1
  beta[i,] = sapply(1:7,function(k){sd(dlist[[i]]$new_cases[wday==k],na.rm=TRUE)})
  beta[i,] = 7 * beta[i,] / sum(beta[i,],na.rm=TRUE)
  dlist[[i]]$new_cases3 = round(dlist[[i]]$new_cases2 / beta[i,wday])
  i = i + 1
}
rm(beta)
ncty = length(clist)
grob_Rt1 = NULL
grob_lambda1 = NULL
grob_W = NULL

for (i in 1:ncty) {
  y = dlist[[i]]$new_cases3
  cty = clist[i]
  opts$n = length(y)
  delta = get_optimal_delta(y,opts$ModelCode,delta_grid,
                            L=opts$L,rho=opts$rho,
                            mu0=opts$mu0,
                            ctanh=opts$ctanh,
                            alpha=opts$alpha,
                            m0_prior=opts$m0, C0_prior=opts$C0,
                            delta_nb=opts$delta_nb, 
                            obs_type=opts$obs_type)$delta_optim
  opts$delta = max(delta)
  
  pout = test_model_real(y,opts,eta_select=eta_select,
                         eta_prior_type=eta_prior_type)
  for (i in c(1:length(pout))) {
    pout[[i]] = pout[[i]] + xlab(cty) 
    if (i < length(pout)) {
      pout[[i]] = pout[[i]] + 
        scale_x_continuous(breaks=c(1,100,200),
                           labels=dlist[[i]]$date[c(1,100,200)])
    }
  }
  grob_Rt1 = c(grob_Rt1,list(pout[[1]]))
  grob_lambda1 = c(grob_lambda1,list(pout[[2]]))
  grob_W = c(grob_W,list(pout[[5]]))
}


if (eta_select[1]==1 & eta_select[2]==0) {
  fname = paste0("sample_model",ModelCode,
                 "_obs",obs_type,
                 "_eta",paste(eta_select,collapse=""),
                 "_",args[4],"W",args[5],
                 "_M",ctanhM,".pdf")
} else if (eta_select[2]==1) {
  fname = paste0("sample_model",ModelCode,
                 "_obs",obs_type,
                 "_eta",paste(eta_select,collapse=""),
                 "_",args[4],"W",args[5],
                 "_",args[6],"mu0",args[7],
                 "_M",ctanhM,".pdf")
}
grobs = marrangeGrob(c(grob_Rt1,grob_lambda1,grob_W),
                     nrow=4,ncol=1,padding=unit(0,units="cm"))
ggsave(file.path(opath,"country",fname),grobs)


rm(list=ls())
gc(verbose=FALSE,reset=TRUE,full=TRUE)
