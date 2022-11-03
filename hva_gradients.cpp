#include "hva_gradients.h"
using namespace Rcpp;
// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::depends(RcppArmadillo)]]


/*
Everything related to the Yeo-Johnson Transform
*/
//' @export
// [[Rcpp::export]]
double tYJ_gm(
    const double c,
    const double gamma) {
    double nu;
    if (c<arma::datum::eps) {
        nu = - (std::pow(-c+1.,2.-gamma)-1.)/(2.-gamma);
    } else {
        nu = (std::pow(c+1.,gamma) - 1.) / gamma;
    }
    return nu; // Check -- Correct
}


//' @export
// [[Rcpp::export]]
double tYJinv_gm(
    const double nu,
    const double gamma) {
    double c,gmt;
    if (nu<arma::datum::eps) {
        gmt = 2.-gamma;
        c = 1. - std::pow(1.-gmt*nu,1./gmt);
    } else {
        gmt = gamma;
        c = std::pow(1.+gmt*nu,1./gmt) - 1.;
    }
    return c; // Check -- Correct
}



// Transform gamma in (0,2) to the real line, denoted by tau
//' @export
// [[Rcpp::export]]
double gm2tau(const double gamma) {
    return -std::log(2./gamma - 1.);
}



// Transform tau in real line to (0,2), denoted by gamma
//' @export
// [[Rcpp::export]]
double tau2gm(const double tau) {
    return 2./(std::exp(-tau) + 1.);
}



// First-order derivative of gamma with respect to tau
// given the value of tau
//' @export
// [[Rcpp::export]]
double dgm_dtau(const double tau) {
    double etau = std::exp(tau);
    return 2.*etau / std::pow(1.+etau,2.);
}



// First-order derivative of gamma with respect to tau
// given the value of gamma
//' @export
// [[Rcpp::export]]
double dgm_dtau_gm(const double gamma) {
    return 0.5*gamma*(2.-gamma);
}



/* 
First-order derivative of tYJ(c;gamma) with respect to c
given the value of c and gamma

Corresponds to ./Derivatives/dtYJ_dtheta.m
*/
//' @export
// [[Rcpp::export]]
double dYJ_dc(const double c, const double gamma) {
    double res = 1.+std::abs(c);
    if (c<arma::datum::eps) {
        return std::pow(res,1.-gamma);
    } else {
        return std::pow(res,gamma-1.);
    }
}



/*Loaiza-Maya et al., PDF, bottom of p26*/
//' @export
// [[Rcpp::export]]
double dYJ_dgamma(const double c, const double gamma) {
    double res,gmt;
    if (c<arma::datum::eps) {
        gmt = 2.-gamma;
        res = 1. - std::pow(1.-c,gmt)*(1.-gmt*std::log(1.-c));
        res *= std::pow(gmt,-2.);
    } else {
        gmt = gamma;
        res = 1. + std::pow(1.+c,gmt)*(std::log(1.+c)*gmt-1.);
        res *= std::pow(gmt,-2.);
    }
    return res;
} // Check -- Correct



//' @export
// [[Rcpp::export]]
double dlogdYJ_dc(const double c, const double gamma) {
    return (gamma-1.)/(1.+std::abs(c));
}



/*
First order derivatives of YJinv with respect to nu
Correspond to ./Derivatives/dtheta_dphi
*/
//' @export
// [[Rcpp::export]]
double dYJinv_dnu(const double nu, const double gamma) {
    double gmt,res;
    if (nu<arma::datum::eps) {
        gmt = 2.-gamma;
        res = std::pow(1.-gmt*nu,(1.-gmt)/gmt);
    } else {
        gmt = gamma;
        res = std::pow(1.+gmt*nu,(1.-gmt)/gmt);
    }
    return res;
}



/*
--- dYJinv_dgamma ---
First order derivatives of YJinv with respect to gamma
Corresponds to ./Derivatives/dtheta_deta.m

--- Note the following equivalency ---
c = 10
gm = 0.5
nu = tYJ_gm(c,gm)

dYJinv_dgamma(nu,gm)
-dYJ_dgamma(c,gm)/dYJ_dc(c,gm) 
--- Note the following equivalency ---
*/
//' @export
// [[Rcpp::export]]
double dYJinv_dgamma(const double nu, const double gamma) {
    double gmt,res;
    if (nu<arma::datum::eps) {
        gmt = 2.-gamma;
        res = std::log(1.-gmt*nu) + gmt*nu/(1.-gmt*nu);
        res *= std::pow(gmt,-2.);
        res *= -std::pow(1.-gmt*nu,1./gmt);
    } else {
        gmt = gamma;
        res = -std::log(1.+gmt*nu) + gmt*nu/(1.+gmt*nu);
        res *= std::pow(gmt,-2.);
        res *= std::pow(1.+gmt*nu,1./gmt);
    }
    return res;
}



/*
First order derivatives of YJ inv with respect to mu, sig, tau
*/
//' @export
// [[Rcpp::export]]
arma::vec dYJinv(
    const double nu, // c = YJinv(nu), nu = mu+sig*eps
    const double eps,
    const double gamma) {

    arma::vec dd(3);
    dd.at(0) = dYJinv_dnu(nu,gamma); // == line 20 of ./Derivatives/dtheta_dmu.m
    dd.at(1) = dYJinv_dnu(nu,gamma) * eps; // TODO: Check line 26 of ./Derivatives/dtheta_dBDelta.m
    dd.at(2) = dYJinv_dgamma(nu,gamma) * dgm_dtau_gm(gamma); // == line 22 of ./Derivatives/dtheta_dtau.m
    return dd;
}



double dlogJoint_dc(
    const arma::vec& psi,
    const double c) {
    const double n_ = static_cast<double>(psi.n_elem) - 1.;
    return 0.5*n_ - 0.5*arma::accu(arma::pow(arma::diff(psi),2.))*std::exp(c);
}



//' @export
// [[Rcpp::export]]
double dlogJoint_dc(
    const arma::vec& psi,
    const double c,
    const double aw,
    const double bw) {
    const double n_ = static_cast<double>(psi.n_elem) - 1.;
    // return 0.5*n_+aw+1. - (bw+0.5*arma::accu(arma::pow(arma::diff(psi),2.)))*std::exp(c);
    return 0.5*n_+aw - (bw+0.5*arma::accu(arma::pow(arma::diff(psi),2.)))*std::exp(c);
}



//' @export
// [[Rcpp::export]]
double dlogVB_dc(
    const double c, 
    const double mu, 
    const double sig,
    const double gamma) {
    
    double sig2 = sig*sig;
    return -(tYJ_gm(c,gamma)-mu)*dYJ_dc(c,gamma)/sig2 + dlogdYJ_dc(c,gamma);
}



// arma::vec dYJinv_deprecated(
//     const arma::vec& psi,
//     const double mu, 
//     const double sig,
//     const double gm, 
//     const double zeps) {
    
//     const double n_ = static_cast<double>(psi.n_elem) - 1.;
//     double cstar = mu + sig*zeps;
//     double c,cp;

//     if (cstar < arma::datum::eps) { // w. negative values
//         c = 1. - std::pow(1.-(2.-gm)*cstar,1./(2.-gm));
//         cp = std::pow(1.-c,1.-gm);
//     } else { // w. positive values
//         c = std::pow(1.+gm*cstar,1./gm) - 1.;
//         cp = std::pow(1.+c,gm-1.);
//     }

//     // Rcout << "c: " << c << std::endl; // Correct
//     // Rcout << "cp: " << cp << std::endl; // Correct

//     double grad_q0 = -(cstar - mu)*cp/std::pow(sig,2.) + (gm-1.)/(1.+std::abs(c));
//     double grad_g = 0.5*n_ - 0.5*arma::accu(arma::pow(arma::diff(psi),2.))*std::exp(c);
//     double grad_elbo_mu = 1./cp;
//     double grad_elbo_sig = zeps/cp;

//     double tau = -std::log(2./gm-1.);
//     double tmpd,grad_elbo_tau;
//     if (c < arma::datum::eps) {
//         tmpd = 1 - 2*cstar + cstar*gm;
//         grad_elbo_tau = std::log(tmpd)/std::pow(2.-gm,2.);
//         grad_elbo_tau += cstar/tmpd/(2.-gm);
//         grad_elbo_tau *= -std::exp(std::log(tmpd)/(2.-gm));
//     } else {
//         tmpd = 1. + cstar*gm;
//         grad_elbo_tau = -std::log(tmpd)/std::pow(gm,2.);
//         grad_elbo_tau += cstar/tmpd/gm;
//         grad_elbo_tau *= std::exp(std::log(tmpd)/gm);
//     }
//     grad_elbo_tau *= 2.*std::exp(-tau) / std::pow(1.+std::exp(-tau),2.);

//     Rcout << "grad_g: " << grad_g << std::endl;
//     Rcout << "grad_q0: " << grad_q0 << std::endl;

//     tmpd = grad_g - grad_q0;
//     // grad_elbo_mu *= tmpd;
//     // grad_elbo_sig *= tmpd;
//     // grad_elbo_tau *= tmpd;

//     arma::vec dd = {grad_elbo_mu,grad_elbo_sig,grad_elbo_tau};
//     return dd;
// }