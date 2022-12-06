get_model_list = function(ModelCode = NULL){
    ModelName = c(
        "KoyamaMax", "KoyamaExp", "SolowMax",
        "SolowExp", "KoyckMax", "KoyckExp",
        "KoyamaEye", "SolowEye", "KoyckEye",
        "VanillaPois")

    ModelCode = c(
        0, 1, 2,
        3, 4, 5,
        6, 7, 8,
        9)

    # ------ Transfer Function ------
    # State equation in transfer function form
    TransferKernel = c(
        "LogNorm", "LogNorm", "NegBinom",
        "NegBinom", "Exponential", "Exponential",
        "LogNorm", "NegBinom", "Exponential",
        "Exponential")
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
        "None")
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
        "Exponential")
    # ------ Link Function ------

    ModelList = data.frame(
        Code = ModelCode,
        Name = ModelName,
        TransferKernel = TransferKernel,
        Reproduction = Reproduction,
        LinkFunction = LinkFunction)

    if (is.null(ModelCode)) {
        return(ModelList)
    } else {
        return(ModelList[ModelList$Code==ModelCode,])
    }
}