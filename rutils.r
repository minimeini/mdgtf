check_params = function(param_infer, strict = TRUE) {
    param_list = c("seas", "rho", "W", "par1", "par2", "zintercept", "zzcoef")
    out = sapply(param_infer, function(x) {
        x %in% param_list
    })
    flag = all(out)
    if (!flag) {
        if (strict) {
            stop("Unknown parameters")
        } else {
            warning("Unknown parameters.")
        }
    }
    
    return(flag)
}