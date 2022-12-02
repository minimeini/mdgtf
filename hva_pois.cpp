#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <algorithm>
#include <RcppArmadillo.h>
#include "model_utils.h"
#include "yjtrans.h"
using namespace Rcpp;
// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::depends(RcppArmadillo)]]



// No corresponding function in Matlab => DOUBLE CHECK IT.
double dlogJoint_dWtilde(
    const arma::vec& theta, // (n+1) x 1, (theta[0],theta[1],...,theta[n])
    const double G, // evolution transition matrix
    const double W, // evolution variance conditional on V
    const double aw,
    const double bw) {
    
    const double n = static_cast<double>(theta.n_elem) - 1.;
    const unsigned int ni = theta.n_elem - 1;
    double res = 0.;
    for (unsigned int t=0; t<ni; t++) {
        res += (theta.at(t+1)-G*theta.at(t)) * (theta.at(t+1)-G*theta.at(t));
    }
    res *= 0.5;
    res += bw;
    res /= -W;
    res += aw + 0.5*n;
    return res;
}


// // No corresponding function in Matlab => DOUBLE CHECK IT.
// //' @export
// // [[Rcpp::export]]
// double dlogJoint_dtheta0(
//     const arma::vec& theta, // (n+1) x 1
//     const double G,
//     const double W,
//     const double V,
//     const double m0,
//     const double C0){

//     double res = 2.*theta.at(0)*std::pow(G,2.) - 2.*theta.at(1)*G;
//     res /= -(2.*V*W);
//     res -= 0.5*(2.*theta.at(0)/C0-2.*m0/C0);
//     return res;
// }

// eta = (Vtilde, Wtilde, theta0)
arma::vec dlogJoint_deta(
    const arma::vec& y, // n x 1
    const arma::vec& theta, // (n+1) x 1
    const double G,
    const double W,
    const double V,
    const double av = 0.1,
    const double bv = 0.1,
    const double aw = 0.1,
    const double bw = 0.1,
    const double m0 = 0.,
    const double C0 = 1.) {
    arma::vec deriv(1);
    deriv.at(0) = dlogJoint_dWtilde(theta,G,W,aw,bw);
    return deriv;
}


/*
Forward Filtering and Backward Sampling
*/
arma::vec rtheta_ffbs(
    const arma::vec& y, // n x 1
    const double G,
    const double W,
    const double m0,
    const double C0,
    const double scale_sd = 1.) {
    
    const unsigned int n = y.n_elem;
    const unsigned int npad = n + 1;

    arma::vec ypad(n+1,arma::fill::zeros); ypad.tail(n) = y;
    arma::vec mt(n+1,arma::fill::zeros); mt.at(0) = m0;
    arma::vec Ct(n+1,arma::fill::zeros); Ct.at(0) = C0;
    arma::vec at(n+1,arma::fill::zeros);
    arma::vec Rt(n+1,arma::fill::zeros);
    arma::vec ht(n+1,arma::fill::zeros);
    arma::vec Ht(n+1,arma::fill::zeros);
    arma::vec theta(n+1,arma::fill::zeros);

    double At, Bt, et;
    double ft, qt, ft_ast, qt_ast, alphat, betat;
    const double F = 1.;

    for (unsigned int t=1; t<=n; t++) {
        at.at(t) = G*mt.at(t-1);
        Rt.at(t) = G*Ct.at(t-1)*G + W;

        ft = F*at.at(t);
        qt = F*Rt.at(t)*F;
        alphat = optimize_trigamma(qt);
        betat = std::exp(R::digamma(alphat) - ft);
        ft_ast = R::digamma(alphat+ypad.at(t)) - std::log(betat+1.);
		qt_ast = R::trigamma(alphat+ypad.at(t));

        At = Rt.at(t)*F/qt;
        et = ft_ast - ft;

        mt.at(t) = at.at(t) + At*et;
        Ct.at(t) = Rt.at(t) + At*(qt_ast-qt)*At;
    }

    ht.at(n) = mt.at(n);
    Ht.at(n) = Ct.at(n);
    theta.at(n) = R::rnorm(ht.at(n),scale_sd*std::sqrt(Ht.at(n)));

    for (unsigned int t=(n-1); t>0; t--) {
        Bt = Ct.at(t)*G/Rt.at(t+1);
        ht.at(t) = mt.at(t) + Bt*(theta.at(t+1) - at.at(t+1));
        Ht.at(t) = Ct.at(t) - Bt * Rt.at(t+1) * Bt;
        theta.at(t) = R::rnorm(ht.at(t),scale_sd*std::sqrt(Ht.at(t)));
    }

    Bt = Ct.at(0)*G/Rt.at(1);
    ht.at(0) = mt.at(0) + Bt*(theta.at(1) - at.at(1));
    Ht.at(0) = Ct.at(0) - Bt * Rt.at(1) * Bt;
    theta.at(0) = R::rnorm(ht.at(0),scale_sd*std::sqrt(Ht.at(0)));

    return theta;

}



//' @export
// [[Rcpp::export]]
Rcpp::List hva_pois(
    const arma::vec& y, // n x 1
    const double G,
    const double aw = 0.1,
    const double bw = 0.1,
    const double m0 = 0.,
    const double C0 = 1.,
    const double rho = 0.95,
    const double eps_step = 1.e-6,
    const unsigned int niter = 100) {

    const unsigned int n = y.n_elem;

    arma::vec eta(1,arma::fill::zeros); // ( Wtilde)
    arma::vec nu(1,arma::fill::zeros);

    arma::vec gamma(1,arma::fill::ones);
    arma::vec tau = gamma2tau(gamma);
    arma::vec mu(1,arma::fill::zeros);
    arma::mat B(1,1,arma::fill::zeros); // Lower triangular
    arma::vec d(1,arma::fill::ones);

    double W;

    arma::vec theta(n+1,arma::fill::zeros);
    arma::vec delta_logJoint(1);
    arma::vec delta_logq(1);
    arma::vec delta_diff(1);

    arma::vec xi(1);
    arma::vec eps(1);

    arma::vec L_mu(1);
    arma::mat L_B(1,1);
    arma::vec L_d(1);
    arma::vec L_tau(1);

    arma::vec oldEg2_mu(1,arma::fill::zeros);
    arma::vec curEg2_mu(1,arma::fill::zeros);
    arma::vec oldEdelta2_mu(1,arma::fill::zeros);
    arma::vec curEdelta2_mu(1,arma::fill::zeros);
    arma::vec Change_delta_mu(1,arma::fill::zeros);

    arma::vec oldEg2_d(1,arma::fill::zeros);
    arma::vec curEg2_d(1,arma::fill::zeros);
    arma::vec oldEdelta2_d(1,arma::fill::zeros);
    arma::vec curEdelta2_d(1,arma::fill::zeros);
    arma::vec Change_delta_d(1,arma::fill::zeros);

    arma::vec oldEg2_tau(1,arma::fill::zeros);
    arma::vec curEg2_tau(1,arma::fill::zeros);
    arma::vec oldEdelta2_tau(1,arma::fill::zeros);
    arma::vec curEdelta2_tau(1,arma::fill::zeros);
    arma::vec Change_delta_tau(1,arma::fill::zeros);


    arma::mat mu_stored(1,niter);
    arma::mat d_stored(1,niter);
    arma::mat gamma_stored(1,niter);
    arma::mat theta_stored(n+1,niter);
    arma::vec W_stored(niter);

    
    for (unsigned int s=0; s<niter; s++) {
        /*
        Step 1. Sample static parameters from the variational distribution
        using reparameterisation.
        */
        eta = rtheta(xi,eps,gamma,mu,B,d);
        nu = tYJ(eta,gamma);
        W = std::exp(-eta.at(0));

        /*
        Step 2. Sample state parameters via posterior
        */
        theta = rtheta_ffbs(y,G,W,m0,C0);

        /*
        Step 3. Compute gradient of the log variational distribution
        */
        delta_logJoint = dlogJoint_deta(y,theta,G,W,aw,bw,m0,C0); // 2 x 1
        delta_logq = dlogq_dtheta(nu,eta,gamma,mu,B,d); // 2 x 1
        delta_diff = delta_logJoint - delta_logq; // 2 x 1

        // TODO: transpose or no transpose
        L_mu = dYJinv_dnu(nu,gamma) * delta_diff;
        // L_B = arma::reshape(arma::inplace_trans(dYJinv_dB(nu,gamma,xi))*delta_diff,2,2);
        // L_B.elem(arma::trimatu_ind(arma::size(L_B),1)).zeros();
        L_d = dYJinv_dD(nu,gamma,eps)*delta_diff;
        L_tau = dYJinv_dtau(nu,gamma)*delta_diff;


        /*
        Step 4. Update Variational Parameters
        */
        // mu
        oldEg2_mu = curEg2_mu;
        oldEdelta2_mu = curEdelta2_mu;

        curEg2_mu = rho*oldEg2_mu + (1.-rho)*arma::pow(L_mu,2.); // 2 x 1
        Change_delta_mu = arma::sqrt(oldEdelta2_mu + eps_step)/arma::sqrt(curEg2_mu + eps_step) % L_mu;
        mu = mu + Change_delta_mu;
        curEdelta2_mu = rho*oldEdelta2_mu + (1.-rho)*arma::pow(Change_delta_mu,2.);

        // d
        oldEg2_d = curEg2_d;
        oldEdelta2_d = curEdelta2_d;

        curEg2_d = rho*oldEg2_d + (1.-rho)*arma::pow(L_d,2.); // 2 x 1
        Change_delta_d = arma::sqrt(oldEdelta2_d + eps_step)/arma::sqrt(curEg2_d + eps_step) % L_d;
        d = d + Change_delta_d;
        curEdelta2_d = rho*oldEdelta2_d + (1.-rho)*arma::pow(Change_delta_d,2.);

        // tau
        oldEg2_tau = curEg2_tau;
        oldEdelta2_tau = curEdelta2_tau;

        curEg2_tau = rho*oldEg2_tau + (1.-rho)*arma::pow(L_tau,2.); // 2 x 1
        Change_delta_tau = arma::sqrt(oldEdelta2_tau + eps_step)/arma::sqrt(curEg2_tau + eps_step) % L_tau;
        tau = tau + Change_delta_tau;
        curEdelta2_tau = rho*oldEdelta2_tau + (1.-rho)*arma::pow(Change_delta_tau,2.);
        gamma = tau2gamma(tau);

        mu_stored.col(s) = mu;
        d_stored.col(s) = d;
        gamma_stored.col(s) = gamma;
        theta_stored.col(s) = theta;
        W_stored.at(s) = W;

    }

    Rcpp::List output;
    output["mu"] = Rcpp::wrap(mu_stored);
    output["d"] = Rcpp::wrap(d_stored);
    output["gamma"] = Rcpp::wrap(gamma_stored);
    output["theta_stored"] = Rcpp::wrap(theta_stored);
    output["W_stored"] = Rcpp::wrap(W_stored);

    return output;
}