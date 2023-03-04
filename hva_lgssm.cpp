#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <algorithm>
#include <RcppArmadillo.h>
#include "yjtrans.h"
using namespace Rcpp;
// [[Rcpp::plugins(cpp17)]]
// [[Rcpp::depends(RcppArmadillo)]]




// No corresponding function in Matlab => DOUBLE CHECK IT.
double dlogJoint_dVtilde(
    const arma::vec& y, // n x 1
    const arma::vec& theta, // (n+1) x 1
    const double G,
    const double W,
    const double V,
    const double av,
    const double bv) {

    const double n = static_cast<double>(theta.n_elem) - 1.;
    const unsigned int ni = theta.n_elem - 1;
    const double p = 1.;
    double res = 0.;
    for (unsigned int t=0; t<ni; t++) {
        res += 0.5*std::pow(y.at(t)-theta.at(t+1),2.) + 0.5/W*std::pow(theta.at(t+1)-G*theta.at(t),2.);
    }
    res += bv;
    res /= -V;
    res += av + 0.5*n*(p+1);
    return res;
}



// No corresponding function in Matlab => DOUBLE CHECK IT.
double dlogJoint_dWtilde(
    const arma::vec& theta, // (n+1) x 1, (theta[0],theta[1],...,theta[n])
    const double G, // evolution transition matrix
    const double W, // evolution variance conditional on V
    const double V, // observational variance.
    const double aw,
    const double bw) {
    
    const double n = static_cast<double>(theta.n_elem) - 1.;
    const unsigned int ni = theta.n_elem - 1;
    const double p = 1.;
    double res = 0.;
    for (unsigned int t=0; t<ni; t++) {
        res += (theta.at(t+1)-G*theta.at(t)) * (theta.at(t+1)-G*theta.at(t));
    }
    res *= 0.5/V;
    res += bw;
    res /= -W;
    res += aw + 0.5*n*p;
    return res;
}


// // No corresponding function in Matlab => DOUBLE CHECK IT.
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
    arma::vec deriv(2);
    deriv.at(0) = dlogJoint_dVtilde(y,theta,G,W,V,av,bv);
    deriv.at(1) = dlogJoint_dWtilde(theta,G,W,V,aw,bw);
    // deriv.at(2) = dlogJoint_dtheta0(theta,G,W,V,m0,C0);
    return deriv;
}


/*
Forward Filtering and Backward Sampling
*/
arma::vec rtheta_ffbs(
    const arma::vec& y, // n x 1
    const double G,
    const double W_ast,
    const double V,
    const double m0,
    const double C0) {
    
    const unsigned int n = y.n_elem;
    const unsigned int npad = n + 1;

    arma::vec ypad(n+1,arma::fill::zeros); ypad.tail(n) = y;
    arma::vec mt(n+1,arma::fill::zeros); mt.at(0) = m0;
    arma::vec Ct_ast(n+1,arma::fill::zeros); Ct_ast.at(0) = C0;
    arma::vec at(n+1,arma::fill::zeros);
    arma::vec Rt_ast(n+1,arma::fill::zeros);
    arma::vec ht(n+1,arma::fill::zeros);
    arma::vec Ht_ast(n+1,arma::fill::zeros);
    arma::vec theta(n+1,arma::fill::zeros);

    double At_ast, Bt_ast, et;
    double ft, qt_ast;
    const double F = 1.;

    for (unsigned int t=1; t<=n; t++) {
        at.at(t) = G*mt.at(t-1);
        Rt_ast.at(t) = G*Ct_ast.at(t-1)*G + W_ast;

        ft = F*at.at(t);
        qt_ast = F*Rt_ast.at(t)*F + 1.;

        At_ast = Rt_ast.at(t)*F/qt_ast;
        et = ypad.at(t) - ft;

        mt.at(t) = at.at(t) + At_ast*et;
        Ct_ast.at(t) = Rt_ast.at(t) - At_ast*qt_ast*At_ast;
    }

    ht.at(n) = mt.at(n);
    Ht_ast.at(n) = Ct_ast.at(n);
    theta.at(n) = R::rnorm(ht.at(n),std::sqrt(V*Ht_ast.at(n)));

    for (unsigned int t=(n-1); t>0; t--) {
        Bt_ast = Ct_ast.at(t)*G/Rt_ast.at(t+1);
        ht.at(t) = mt.at(t) + Bt_ast*(theta.at(t+1) - at.at(t+1));
        Ht_ast.at(t) = Ct_ast.at(t) - Bt_ast * Rt_ast.at(t+1) * Bt_ast;
        theta.at(t) = R::rnorm(ht.at(t),std::sqrt(V*Ht_ast.at(t)));
    }

    Bt_ast = Ct_ast.at(0)*G/Rt_ast.at(1);
    ht.at(0) = mt.at(0) + Bt_ast*(theta.at(1) - at.at(1));
    Ht_ast.at(0) = Ct_ast.at(0) - Bt_ast * Rt_ast.at(1) * Bt_ast;
    theta.at(0) = R::rnorm(ht.at(0),std::sqrt(V*Ht_ast.at(0)));

    return theta;

}



//' @export
// [[Rcpp::export]]
Rcpp::List hva_lgssm(
    const arma::vec& y, // n x 1
    const double G,
    const double av = 0.1,
    const double bv = 0.1,
    const double aw = 0.1,
    const double bw = 0.1,
    const double m0 = 0.,
    const double C0 = 1.,
    const double rho = 0.95,
    const double eps_step = 1.e-6,
    const unsigned int niter = 100) {

    const unsigned int n = y.n_elem;

    arma::vec eta(2,arma::fill::zeros); // (Vtilde, Wtilde)
    arma::vec nu(2,arma::fill::zeros);

    arma::vec gamma(2,arma::fill::ones);
    arma::vec tau = gamma2tau(gamma);
    arma::vec mu(2,arma::fill::zeros);
    arma::mat B(2,2,arma::fill::zeros); // Lower triangular
    arma::vec d(2,arma::fill::ones);

    double V,W_ast;

    arma::vec theta(n+1,arma::fill::zeros);
    arma::vec delta_logJoint(2);
    arma::vec delta_logq(2);
    arma::vec delta_diff(2);

    arma::vec xi(2);
    arma::vec eps(2);

    arma::vec L_mu(2);
    arma::mat L_B(2,2);
    arma::vec L_d(2);
    arma::vec L_tau(2);

    arma::vec oldEg2_mu(2,arma::fill::zeros);
    arma::vec curEg2_mu(2,arma::fill::zeros);
    arma::vec oldEdelta2_mu(2,arma::fill::zeros);
    arma::vec curEdelta2_mu(2,arma::fill::zeros);
    arma::vec Change_delta_mu(2,arma::fill::zeros);

    arma::vec oldEg2_d(2,arma::fill::zeros);
    arma::vec curEg2_d(2,arma::fill::zeros);
    arma::vec oldEdelta2_d(2,arma::fill::zeros);
    arma::vec curEdelta2_d(2,arma::fill::zeros);
    arma::vec Change_delta_d(2,arma::fill::zeros);

    arma::vec oldEg2_tau(2,arma::fill::zeros);
    arma::vec curEg2_tau(2,arma::fill::zeros);
    arma::vec oldEdelta2_tau(2,arma::fill::zeros);
    arma::vec curEdelta2_tau(2,arma::fill::zeros);
    arma::vec Change_delta_tau(2,arma::fill::zeros);


    arma::mat mu_stored(2,niter);
    arma::mat d_stored(2,niter);
    arma::mat gamma_stored(2,niter);
    arma::mat theta_stored(n+1,niter);
    arma::vec V_stored(niter);
    arma::vec Wast_stored(niter);

    
    for (unsigned int s=0; s<niter; s++) {
        /*
        Step 1. Sample static parameters from the variational distribution
        using reparameterisation.
        */
        eta = rtheta(xi,eps,gamma,mu,B,d);
        nu = tYJ(eta,gamma);
        V = std::exp(-eta.at(0));
        W_ast = std::exp(-eta.at(1));

        /*
        Step 2. Sample state parameters via posterior
        */
        theta = rtheta_ffbs(y,G,W_ast,V,m0,C0);

        /*
        Step 3. Compute gradient of the log variational distribution
        */
        delta_logJoint = dlogJoint_deta(y,theta,G,W_ast,V,av,bv,aw,bw,m0,C0); // 2 x 1
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
        V_stored.at(s) = V;
        Wast_stored.at(s) = W_ast;

    }

    Rcpp::List output;
    output["mu"] = Rcpp::wrap(mu_stored);
    output["d"] = Rcpp::wrap(d_stored);
    output["gamma"] = Rcpp::wrap(gamma_stored);
    output["theta_stored"] = Rcpp::wrap(theta_stored);
    output["V_stored"] = Rcpp::wrap(V_stored);
    output["Wast_stored"] = Rcpp::wrap(Wast_stored);

    return output;
}



//' @export
// [[Rcpp::export]]
Rcpp::List va_lgssm(
    const arma::vec& y, // n x 1
    const double G,
    const double av = 0.1,
    const double bv = 0.1,
    const double aw = 0.1,
    const double bw = 0.1,
    const double m0 = 0.,
    const double C0 = 1.,
    const unsigned int niter = 100,
    const unsigned int ngibbs = 100) {

    const unsigned int n = y.n_elem;
    const double ngibbs_ = static_cast<double>(ngibbs);

    arma::vec eta(2,arma::fill::zeros); // (Vtilde, Wtilde)
    arma::vec nu(2,arma::fill::zeros);

    arma::vec gamma(2,arma::fill::ones);
    arma::vec tau = gamma2tau(gamma);
    arma::vec mu(2,arma::fill::zeros);
    arma::mat B(2,2,arma::fill::zeros); // Lower triangular
    arma::vec d(2,arma::fill::ones);

    double V,W_ast;
    double av_new, bv_new, aw_new, bw_new;
    av_new = av + static_cast<double>(n);
    aw_new = aw + 0.5*static_cast<double>(n);

    arma::vec theta(n+1,arma::fill::zeros);
    arma::vec delta_logJoint(2);
    arma::vec delta_logq(2);
    arma::vec delta_diff(2);

    double theta_ds=0.;
    double theta_do=0.;
    arma::mat theta_gibbs(n+1,ngibbs);
    arma::vec theta_tmp(n+1);
    arma::vec theta_diff_state(ngibbs);
    arma::vec theta_diff_obs(ngibbs);

    arma::mat theta_stored(n+1,niter);
    arma::vec V_stored(niter);
    arma::vec Wast_stored(niter);

    
    for (unsigned int s=0; s<niter; s++) {
        /*
        Step 1. Sample static parameters from the variational distribution
        using reparameterisation.
        */
        bv_new = bv + 0.5/W_ast * theta_ds + 0.5*theta_do;
        V = bv_new / (av_new - 1.);

        bw_new = bw + 0.5/V * theta_ds;
        W_ast = bw_new / (aw_new - 1.);

        /*
        Step 2. Sample state parameters via posterior
        */
        for (unsigned int b=0; b<ngibbs; b++) {
            theta_tmp = rtheta_ffbs(y,G,W_ast,V,m0,C0);
            theta_gibbs.col(b) = theta_tmp;
            theta_diff_state.at(b) = arma::accu(arma::pow(theta_tmp.tail(n) - G*theta_tmp.head(n),2.));
            theta_diff_obs.at(b) = arma::accu(arma::pow(y - theta_tmp.tail(n),2.));
        }
        theta = arma::sum(theta_gibbs,1) / ngibbs_;
        theta_ds = arma::accu(theta_diff_state) / ngibbs_;
        theta_do = arma::accu(theta_diff_obs) / ngibbs_;

        theta_stored.col(s) = theta;
        V_stored.at(s) = V;
        Wast_stored.at(s) = W_ast;

    }

    Rcpp::List output;
    output["theta_stored"] = Rcpp::wrap(theta_stored);
    output["V_stored"] = Rcpp::wrap(V_stored);
    output["Wast_stored"] = Rcpp::wrap(Wast_stored);

    return output;
}