# Sampling of Evolution Coefficient

aka the $\rho$


## Reference Prior with a Normal conditional posterior

This method doesn't work because Fx and E0tilde are functions of rho, and Et is not independent of rho.

```
Et = E0tilde + Fx * vt;
Xtilde = arma::reverse(Et);
for (unsigned int t=0; t<(n-1); t++) {
    WtildeInv.at(t,t) = 1./Q;
    F.at(t) = Et.at(n-1-t);
}
Vrho = 1./arma::as_scalar(F*WtildeInv*F.t());
rho_hat = Vrho * arma::as_scalar(F*WtildeInv*Xtilde);
rho = R::rnorm(rho_hat,std::sqrt(Vrho));

Fx = update_Fx0(n,rho);
for (unsigned int t=0; t<n; t++) {
    E0tilde.at(t) = std::pow(rho,static_cast<double>(t))*E0;
}
```

## MH on the logit of rho with normal random walk proposal

> Ref: Alves et al. (2010)

```
xi_old = std::log(rho/(1.-rho));
Et = E0tilde + Fx * vt;
lambda = arma::exp(Et);
logp_old = xi_old - 2.*std::log(std::exp(xi_old)+1.);
for (unsigned int t=0; t<n; t++) {
    logp_old += R::dpois(Y.at(t),lambda.at(t),true);
}

xi_new = R::rnorm(xi_old,std::sqrt(Vxi));
rho = 1./(1.+std::exp(-xi_new));

Fx = update_Fx0(n,rho);
E0tilde.at(0) = E0;
for (unsigned int t=1; t<n; t++) { // TODO - CHECK HERE
    E0tilde.at(t) = rho*E0tilde.at(t-1);
}
Et = E0tilde + Fx * vt;
lambda = arma::exp(Et);
logp_new = xi_new - 2.*std::log(std::exp(xi_new)+1.);
for (unsigned int t=0; t<n; t++) {
    logp_new += R::dpois(Y.at(t),lambda.at(t),true);
}

logratio = std::min(0.,logp_new-logp_old);
if (std::log(R::runif(0.,1.)) >= logratio) { // reject
    rho = 1./(1.+std::exp(-xi_old));
    Fx = update_Fx0(n,rho);
} else {
    rho_accept += 1.;
}
```