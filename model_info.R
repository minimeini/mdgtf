get_model_list = function(model_code = NULL, 
                          return_type = "code"){ # {"code", "name"}
    ModelName = c(
        "KoyamaMax", "KoyamaExp", "SolowMax",
        "SolowExp", "KoyckMax", "KoyckExp",
        "KoyamaEye", "SolowEye", "KoyckEye",
        "VanillaPois",
        "KoyckSoftplus","KoyamaSoftplus", "SolowSoftplus",
        "KoyckTanh","KoyamaTanh","SolowTanh",
        "KoyckLogistic","KoyamaLogistic","SolowLogistic")

    ModelCode = c(
        0, 1, 2,
        3, 4, 5,
        6, 7, 8,
        9,
        10,11,12,
        13,14,15,
        16,17,18)

    # ------ Transfer Function ------
    # State equation in transfer function form
    TransferKernel = c(
        "LogNorm", "LogNorm", "NegBinom",
        "NegBinom", "Exponential", "Exponential",
        "LogNorm", "NegBinom", "Exponential",
        "Exponential",
        "Exponential","LogNorm","NegBinom",
        "Exponential","LogNorm","NegBinom",
        "Exponential","LogNorm","NegBinom")
    
    TransferCode = c(
      1,1,2,
      2,0,0,
      1,2,0,
      3,
      0,1,2,
      0,1,2,
      0,1,2
    )
    # ------ Transfer Function ------

    # ------ Reproduction Kernel ------
    # h(.) or Gt(.) in the state eq of DLM form
    # If reproduction kernel is identity, Gt(.) = Gt is linear.
    # If reproduction kernel is ramp or exponential, Gt(.) is nonlinear.
    # => Linear approximation via Taylor expansion.
    Reproduction = c(
        "Ramp", "Exponential", "Ramp",
        "Exponential", "Ramp", "Exponential",
        "Identity", "Identity", "Identity",
        "None",
        "Softplus","Softplus","Softplus",
        "Tanh","Tanh","Tanh",
        "Logistic","Logistic","Logistic")
    
    GainCode = c(
      0,1,0,
      1,0,1,
      2,2,2,
      -1,
      3,3,3,
      4,4,4,
      5,5,5
    )
    # ------ Reproduction Kernel ------

    # ------ Link Function ------
    # g(.) or Ft(.) in the obs eq of DLM form
    # If link function is identity, g(.) = 1, then Ft(.) = Ft is linear.
    # If link function is exponential, then Ft(.) is nonlinear.
    # => Moment matching via conjugate priors.
    LinkFunction = c(
        "Identity", "Identity", "Identity",
        "Identity", "Identity", "Identity",
        "Exponential", "Exponential", "Exponential",
        "Exponential",
        "Identity", "Identity", "Identity",
        "Identity", "Identity", "Identity",
        "Identity", "Identity", "Identity")
    
    LinkCode = c(0,0,0,
                 0,0,0,
                 1,1,1,
                 1,
                 0,0,0,
                 0,0,0,
                 0,0,0)
    # ------ Link Function ------

    ModelList = data.frame(
        Code = ModelCode,
        Name = ModelName,
        LinkCode = LinkCode,
        LinkFunction = LinkFunction,
        TransferCode = TransferCode,
        TransferKernel = TransferKernel,
        GainCode = GainCode,
        Reproduction = Reproduction)

    if (is.null(model_code)) {
      return(ModelList)
    } else if (return_type == "name") {
      return(ModelList[ModelList$Code==model_code,c(2,4,6,8)])
    } else if (return_type == "code") {
      return(ModelList[ModelList$Code==model_code,c(3,5,7)])
    }
}



get_model_code = function(obs_dist = "nbinom", # {"nbinom","poisson"}
                          link_func = "identity", # {"identity","exponential"}
                          trans_func = "koyama", # {"koyama","solow","koyck","vanilla"}
                          gain_func = "logistic", # {"exponential","softplus","logistic"}
                          err_dist = "gaussian"){ # {"gaussian","laplace","cauchy","left_skewed_normal"}
  
  obs_code = switch(tolower(obs_dist),
                    "nbinom"=0,"poisson"=1,"nbinom_p"=2,"gaussian"=3)
  link_code = switch(tolower(link_func),
                     "identity"=0,"exponential"=1)
  trans_code = switch(tolower(trans_func),
                      "koyck"=0,"koyama"=1, 
                      "solow"=2,"vanilla"=3)
  gain_code = switch(tolower(gain_func),
                     "ramp"=0,"exponential"=1,"identity"=2,
                     "softplus"=3,"tanh"=4,"logistic"=5)
  err_code = switch(tolower(err_dist),
                    "gaussian"=0,"laplace"=1,
                    "cauchy"=2,"left_skewed_normal"=3)
  
  model_code = c(obs_code,link_code,trans_code,gain_code,err_code)
  return(model_code)
}