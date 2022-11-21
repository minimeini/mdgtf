get_rho_prob = function(Y,X,delta,rho_grid,m0,C0,model="X") {
  n = length(Y)
  nrho = length(rho_grid)
  logprob = rep(0,nrho)
  
  for (k in 1:nrho) {
    if (model=="X") {
      output = lbe_poissonX(Y,X,rho_grid[k],delta,m0,C0)
    } else if (model=="Solow") {
      output = lbe_poissonSolow(Y,X,rho_grid[k],delta,m0,C0)
    }
    
    alpha = c(output$alphat)[-1]
    beta = c(output$betat)[-1]
    logprob[k] = sum(lgamma(alpha+Y) - lgamma(Y+1) - lgamma(alpha) -
      alpha*log(beta) - (alpha+Y)*log(beta+1))
  }
  return(logprob)
}



estimate_state_var = function(state_err_hat,QPrior) {
  nv = QPrior[1]
  nSv = nv*QPrior[2]
  n = length(state_err_hat)
  nv_new = nv + n
  nSv_new = nSv + sum(state_err_hat^2)
  Qhat = nSv_new/nv_new
  return(Qhat)
}