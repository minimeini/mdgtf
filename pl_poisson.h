#ifndef _PL_POISSON_H
#define _PL_POISSON_H

#include "lbe_poisson.h"




void init_Ft(
    arma::vec &Ft, 
    const arma::vec &ypad, 
    const arma::vec &Fphi,
    const unsigned int &t, 
    const unsigned int &p);

Rcpp::List mcs_poisson(
    const arma::vec &Y, // nt x 1, the observed response
    const arma::uvec &model_code,
    const double &W_true, // Use discount factor if W is not given
    const double &rho,
    const unsigned int &L_order, // number of lags
    const unsigned int &nlag_,
    const double &mu0,
    const unsigned int &B,                               // length of the B-lag fixed-lag smoother (Anderson and Moore 1979; Kitagawa and Sato)
    const unsigned int &N,                               // number of particles
    const Rcpp::Nullable<Rcpp::NumericVector> &m0_prior, // mean of normal prior for theta0
    const Rcpp::Nullable<Rcpp::NumericMatrix> &C0_prior, // variance of normal prior for theta0
    const double &theta0_upbnd,                          // Upper bound of uniform prior for theta0
    const Rcpp::NumericVector &qProb,
    const double &delta_nb,
    const double &delta_discount);

void smc_propagate_bootstrap(
    arma::mat &Theta_new, // p x N
    arma::vec &weights,   // N x 1
    double &wt,
    arma::mat &Gt, // no need to update Gt
    const double &y_new,
    const double &y_old,        // (n+1) x 1, the observed response
    const arma::mat &Theta_old, // p x N
    const arma::vec &Ft,        // must be already updated if used
    const arma::uvec &model_code,
    const double mu0,
    const double rho,
    const double delta_nb,
    const double delta_discount,
    const unsigned int p, // dimension of DLM state space
    const unsigned int N, // number of particles
    const bool use_discount,
    const bool use_default_val,
    const unsigned int t = 9999,
    const unsigned int nlag = 0,
    const unsigned int nobs = 0);

void smc_resample(
    arma::cube &theta_stored, // p x N x (nt + B)
    arma::vec &weights, // N x 1
    double &meff,
    bool &resample,
    const unsigned int &t, // number of particles
    const unsigned int &B);

Rcpp::List ffbs_poisson(
    const arma::vec &Y, // n x 1, the observed response
    const arma::uvec &model_code,
    const double W_true,
    const double rho,
    const unsigned int L_order, // number of lags
    const unsigned int nlag_,
    const double mu0,
    const unsigned int N, // number of particles
    const Rcpp::Nullable<Rcpp::NumericVector> &m0_prior,
    const Rcpp::Nullable<Rcpp::NumericMatrix> &C0_prior,
    const double theta0_upbnd,
    const Rcpp::NumericVector &qProb,
    const double delta_nb,
    const double delta_discount,
    const bool smoothing);

void mcs_poisson(
    arma::mat &R,       // (n+1) x 2, (psi,theta)
    arma::vec &pmarg_y, // n x 1, marginal likelihood of y
    double &W,
    const arma::vec &ypad,        // (n+1) x 1, the observed response
    const arma::uvec &model_code, // (obs_code,link_code,transfer_code,gain_code,err_code)
    const double &rho,
    const unsigned int &L_order, // number of lags
    const unsigned int &nlag_,
    const double &mu0,
    const unsigned int &B, // length of the B-lag fixed-lag smoother (Anderson and Moore 1979; Kitagawa and Sato)
    const unsigned int &N, // number of particles
    const arma::vec &m0_prior,
    const arma::mat &C0_prior,
    const double &theta0_upbnd = 2.,
    const double &delta_nb = 1.,
    const double &delta_discount = 0.95);

/*
Particle Learning
- Reference: Carvalho et al., 2010.

- eta = (W, mu[0], rho, M)
*/
// Rcpp::List pl_poisson(
//     const arma::vec &Y, // nt x 1, the observed response
//     const arma::uvec &model_code,
//     const Rcpp::NumericVector &W_prior, // IG[aw,bw]
//     const double W_true,
//     const unsigned int L,    // number of lags
//     const unsigned int N, // number of particles
//     const Rcpp::Nullable<Rcpp::NumericVector> &m0_prior,
//     const Rcpp::Nullable<Rcpp::NumericMatrix> &C0_prior,
//     const Rcpp::NumericVector &qProb,
//     const double mu0,
//     const double rho,
//     const double delta_nb,
//     const double theta0_upbnd);


#endif