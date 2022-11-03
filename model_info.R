get_model_list = function(ModelCode=NULL){
    ModelName = c(
        "KoyamaMax", "KoyamaExp", "SolowMax",
        "SolowExp", "KoyckMax", "KoyckExp",
        "KoyamaEye", "SolowEye", "KoyckEye")
    
    ModelCode = c(
        0, 1, 2,
        3, 4, 5,
        6, 7, 8)

    TransferKernel = c(
        "LogNorm", "LogNorm", "NegBinom",
        "NegBinom", "Exponential", "Exponential",
        "LogNorm", "NegBinom", "Exponential")
    Reproduction = c(
        "Ramp", "Exponential", "Ramp",
        "Identity", "Identity", "Identity")

    LinkFunction = c(
        "Identity", "Identity", "Identity",
        "Identity", "Identity", "Identity",
        "Exponential", "Exponential", "Exponential")
    
    ModelList = data.frame(
        Code=ModelCode, 
        Name=ModelName,
        TransferKernel=TransferKernel,
        Reproduction=Reproduction,
        LinkFunction=LinkFunction)
    
    if (is.null(ModelCode)) {
        return(ModelList)
    } else {
        return(ModelList[ModelList$Code==ModelCode,])
    }

}