plot_sim = function(sim_dat) {
  require(ggplot2)
  GainCode = sim_dat$params$model_code[4]
  
  n = length(sim_dat$y)
  p1 = ggplot(data.frame(Time=1:n, lambda=sim_dat$lambda, y=sim_dat$y), 
              aes(x=Time)) +
    geom_line(aes(y=lambda)) + geom_point(aes(y=y), size=0.5) +
    theme_light() + ylab("Observations and Intensity")
  
  p2 = ggplot(data.frame(Time=1:n,hpsi=sim_dat$hpsi),aes(x=Time,y=hpsi)) +
    geom_line() + ylab("Reproduction Number") +
    theme_light()
  
  if (GainCode==4 | GainCode==5) {
    p2 = p2 + scale_y_continuous(limits=c(0,sim_dat$params$ctanh[3]))
  }
  
  p3 = ggplot(data.frame(Time=1:n,psi=sim_dat$psi),aes(x=Time,y=psi)) +
    geom_line() + ylab("Gain Factor") + theme_light()
  
  return(list(p1,p2,p3))
  
}


plot_psi = function(sim_dat=NULL,
                    psi_list = NULL,
                    opts=NULL,# number of observations
                    ytrue=NULL,
                    yadjust=NULL,
                    plot_hpsi=FALSE, # if FALSE, plot psi by default
                    plot_lambda=FALSE,
                    npara=0,
                    max_Rt=2.5,
                    max_y=NULL,
                    time_label=NULL) {
  mlist = c("EpiEstim","WT","Koyama2021",
            "LBA","HVB","MCMC",
            "True","VB","PL",
            "SMCS-FL","APF","SMCF-BF","SMCS-BS")
  clist = c("seagreen","orange","burlywood4",
            "maroon","purple","darkturquoise",
            "black","salmon","mediumspringgreen",
            "mediumaquamarine","cornflowerblue","sandybrown","royalblue")
  
  if (!is.null(sim_dat)&&("y" %in% names(sim_dat))) {
    y = sim_dat$y
  } else if (!is.null(ytrue)) {
    y = ytrue
  } else {
    y = NULL
  }
  if (is.null(yadjust)) {
    yadjust = y
  }

  if (plot_lambda) {plot_hpsi=TRUE}
  ####
  ####

  if (!is.null(sim_dat)) {
    opts = sim_dat$params
    if (is.null(opts$ctanh)) {
      opts$ctanh = c(0.2,0,5)
    }
    psi_list$True = cbind(c(opts$psi0,sim_dat$psi),
                          c(opts$psi0,sim_dat$psi),
                          c(opts$psi0,sim_dat$psi))
  }
  
  nl = length(psi_list)
  n = max(c(unlist(lapply(psi_list,function(psi){dim(psi)[1]})),length(y)))
  
  nl2 = sum(names(psi_list)%in%c("LBA","HVB","MCMC",
                                 "SMCS-FL","SMCS-BS","APF","SMCF-BF",
                                 "Koyama2021"))
  
  if (is.null(time_label)) {
    time_label = c(1:length(y))
    use_date = FALSE
  } else {
    time_label = time_label[1:length(y)]
    use_date = TRUE
  }

  if (!is.null(sim_dat) & "lambda" %in% names(sim_dat)) {nl2 = nl2 + 1}
  lambda_list = vector(mode="list",length=nl2)
  
  TransferCode = opts$model_code[3]
  # GainCode = get_gaincode(opts$ModelCode)
  GainCode = opts$model_code[4]
  
  PsiMethod = NULL
  lambdaMethod = NULL
  cnt = 1

  for (i in 1:nl) {
    if (names(psi_list)[i] == "Koyama2021") {
      lambda_tmp = psi_list[[i]][,4:6]
      psi_list[[i]] = psi_list[[i]][-1,1:3]
    }
    
    if (plot_hpsi & 
        names(psi_list)[i]%in%c(
          "LBA","HVB","MCMC",
          "VB","True","PL","SMCS-FL",
          "SMCS-BS","APF","SMCF-BF")) {
      psi_list[[i]] = psi2hpsi(
        psi_list[[i]],GainCode,coef=opts$ctanh)
    }

    
    if (plot_lambda & names(psi_list)[i]%in%c(
      "LBA","HVB","MCMC",
      "VB","PL","SMCS-FL",
      "SMCS-BS","APF","SMCF-BF","Koyama2021")) {
      if (names(psi_list)[i] == "Koyama2021") {
        lambda_list[[cnt]] = lambda_tmp
      } else {
        lambda_list[[cnt]] = hpsi2theta(psi_list[[i]][,1:3],
                                        yadjust,
                                        TransferCode,
                                        opts$theta0,
                                        opts$alpha,
                                        opts$L,opts$rho)
        lambda_list[[cnt]] = lambda_list[[cnt]] + opts$mu0
      }
      
      # lambda_list[[cnt]] = cbind(
      #   lambda_list[[cnt]], # lambda
      #   time_label[c((n-dim(lambda_list[[cnt]])[1]+1) : n)]
      #   ) # time
      # colnames(lambda_list[[cnt]]) = c("lobnd","est","hibnd","time")
      
      lambda_list[[cnt]] = data.frame(
        lobnd = lambda_list[[cnt]][,1],
        est = lambda_list[[cnt]][,2],
        hibnd = lambda_list[[cnt]][,3],
        time = time_label[c((n-dim(lambda_list[[cnt]])[1]+1) : n)]
      )
      lambdaMethod = c(lambdaMethod,rep(names(psi_list)[cnt],dim(lambda_list[[cnt]])[1]))
      lambda_list[[cnt]][c(1:(1+npara)),c(1,3)] = NA # remove CI for the first npara time points
      
      cnt = cnt + 1
    }
    
    # psi_list[[i]] = cbind(psi_list[[i]], # psi
    #                       time_label[(n-dim(psi_list[[i]])[1]+1) : n]) # time
    # colnames(psi_list[[i]]) = c("lobnd","est","hibnd","time")
    
    psi_list[[i]] = data.frame(
      lobnd=psi_list[[i]][,1],
      est=psi_list[[i]][,2],
      hibnd=psi_list[[i]][,3],
      time=time_label[(n-dim(psi_list[[i]])[1]+1) : n]
    )
    PsiMethod = c(PsiMethod,rep(names(psi_list)[i],dim(psi_list[[i]])[1]))
    psi_list[[i]][c(1:(1+npara)),c(1,3)] = NA # remove CI for the first npara time points
  }
  
  

  # true values
  if (plot_lambda & !is.null(sim_dat) & "lambda" %in% names(sim_dat)) {
    lambda_list[[cnt]] = cbind(
      sim_dat$lambda,
      sim_dat$lambda,
      sim_dat$lambda)
    
    # lambda_list[[cnt]] = cbind(lambda_list[[cnt]],
    #                            time_label[(n-dim(lambda_list[[cnt]])[1]+1) : n])
    # colnames(lambda_list[[cnt]]) = c("lobnd","est","hibnd","time")
    
    lambda_list[[cnt]] = data.frame(
      lobnd=lambda_list[[cnt]][,1],
      est=lambda_list[[cnt]][,2],
      hibnd=lambda_list[[cnt]][,3],
      time=time_label[(n-dim(lambda_list[[cnt]])[1]+1) : n]
    )
    
    lambdaMethod = c(lambdaMethod,
                     rep(names(psi_list)[cnt],
                         dim(lambda_list[[cnt]])[1]))
  }

  
  psi_list = do.call(rbind,psi_list)
  psi_list = as.data.frame(psi_list)
  psi_list$method = PsiMethod
  lambda_list = do.call(rbind,lambda_list)
  lambda_list = as.data.frame(lambda_list)
  lambda_list$method = lambdaMethod
  
  
  ####
  methods = unique(psi_list$method)
  cols = sapply(methods,function(m,mlist){which(m==mlist)},mlist)
  cols = clist[cols]
  p = ggplot(psi_list,aes(x=time,y=est,group=method)) +
    geom_line(aes(color=method),na.rm=TRUE) +
    geom_ribbon(aes(ymin=lobnd,ymax=hibnd,fill=method),
                alpha=0.2,na.rm=TRUE) +
    scale_color_manual(name="Method",breaks=methods,values=cols) +
    scale_fill_manual(name="Method",breaks=methods,values=cols) +
    theme_light() + xlab("Time") +
    theme(text=element_text(size=20),
          legend.position="bottom") +
    guides(colour = guide_legend(nrow = 2))
  
  if (plot_hpsi) {
    p = p + ylab(expression(R[t])) +
      geom_hline(yintercept=1,col="black",linetype=2) +
      scale_y_continuous(limits=c(0,max_Rt))
    # if (!is.null(opts$ctanh)) {
    #   p = p + scale_y_continuous(limits=c(0,opts$ctanh[3]))
    # }
  } else {
    p = p + ylab(expression(psi[t]))
  }
  
  
  if (plot_lambda) {
    methods = unique(lambda_list$method)
    cols = sapply(methods,function(m,mlist){which(m==mlist)},mlist)
    cols = clist[cols]
    p2 = ggplot(lambda_list,aes(x=time,y=est,group=method)) +
      geom_line(aes(color=method),na.rm=TRUE) +
      geom_ribbon(aes(ymin=lobnd,ymax=hibnd,fill=method),
                  alpha=0.2,na.rm=TRUE) +
      scale_color_manual(name="Method",breaks=methods,values=cols) +
      scale_fill_manual(name="Method",breaks=methods,values=cols) +
      theme_light() + xlab("Time") + ylab(expression(lambda[t])) +
      theme(text=element_text(size=20),
            legend.position = "bottom") +
      geom_point(aes(x=time,y=y),
                 data=data.frame(
                   time=time_label,y=y,
                   method=rep("True",length(y))),
                 size=1,alpha=0.8)
    
    if (!is.null(max_y)) {
      p2 = p2 + scale_y_continuous(limits=c(0,max_y))
    } else {
      p2 = p2 + scale_y_continuous(limits=c(0,max(y))) 
    }
    
    return(list(p,p2))
  } else {
    return(list(p))
  }

}


plot_eta = function(vname,eta_list,opts=NULL) {
  mlist = c("EpiEstim","WT","Koyama2021","SMC","LBE","HVB","MCMC","True","VB")
  clist = c("seagreen","orange","burlywood4","royalblue","maroon","purple","cyan","black","salmon")
  
  if (vname == "W") {
    methods = names(eta_list)[names(eta_list) %in% c("HVB","MCMC","VB")]
    cols = sapply(methods,function(m){which(mlist %in% m)})
    cols = clist[cols]
    eta_mat = NULL
    cname = NULL
    if ("HVB" %in% methods) {
      eta_mat = cbind(eta_mat,eta_list$HVB)
      cname = c(cname,"HVB")
    }
    if ("MCMC" %in% methods) {
      eta_mat = cbind(eta_mat,eta_list$MCMC)
      cname = c(cname,"MCMC")
    }
    if ("VB" %in% methods) {
      eta_mat = cbind(eta_mat,eta_list$VB)
      cname = c(cname,"VB")
    }
    colnames(eta_mat) = cname
    eta_mat = as.data.frame(eta_mat)
    eta_mat = reshape2::melt(eta_mat)
    
    p = ggplot(eta_mat) +
      geom_histogram(aes(x=value,group=variable,fill=variable),
                     position="identity",alpha=0.5,bins=100) +
      scale_fill_manual(name="Method",values=cols,labels=methods) +
      theme_light() + xlab(expression(W))
    if ("LBE" %in% names(eta_list)) {
      p = p + geom_vline(xintercept=rev(eta_list$LBE)[1],
                         col=clist[which(mlist %in% "LBE")])
    }
  } else if (vname == "mu0") {
    tmp = data.frame(Iteration=1:length(hvb_output$mu0),
                     mu0=hvb_output$mu0)
    p = ggplot(tmp,aes(x=Iteration,y=mu0)) +
      geom_point(size=1,col="mediumpurple1") +
      geom_hline(yintercept=median(hvb_output$mu0),
                 col="purple",linewidth=1) +
      theme_light() + ylab(expression(mu[0]))
    if (!is.null(opts) & ("mu0" %in% names(opts))) {
      p = p + geom_hline(yintercept=opts$mu0,col="black")
    }
  } else {
    stop("Variable not supported.")
  }
  
    
  return(p)
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