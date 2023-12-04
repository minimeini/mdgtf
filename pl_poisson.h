#ifndef _PL_POISSON_H
#define _PL_POISSON_H

#include "lbe_poisson.h"

arma::vec update_Fy(
    const arma::vec &ypad,
    const unsigned int &t,
    const unsigned int &nlag);

void update_Ft(
    arma::vec &Ft,
    const arma::vec &ypad,
    const arma::vec &Fphi,
    const unsigned int &t,
    const unsigned int &nlag);

Rcpp::List mcs_poisson(
    const arma::vec &Y, // nt x 1, the observed response
    const arma::uvec &model_code,
    const double &W, // (init, prior type, prior par1, prior par2)
    const Rcpp::NumericVector &obs_par_in,
    const Rcpp::NumericVector &lag_par_in, // init/true values of (mu, sg2) or (rho, L)
    const unsigned int &nlag_in,
    const unsigned int &B,              // length of the B-lag fixed-lag smoother
    const unsigned int &N,                                     // number of particles
    const double &theta0_upbnd,    // Upper bound of uniform prior for theta0
    const Rcpp::NumericVector &qProb,
    const double &delta_discount,
    const bool &truncated,
    const bool &use_discount);

void smc_propagate_bootstrap(
    arma::mat &Theta_new, // p x N
    arma::vec &weights,   // N x 1
    double &wt,
    arma::mat &Gt,
    const double &y_new,
    const double &y_old,        // (n+1) x 1, the observed response
    const arma::mat &Theta_old, // p x N
    const arma::vec &Ft,        // must be already updated if used
    const arma::uvec &model_code,
    const arma::vec &obs_par,
    const arma::vec &lag_par,
    const unsigned int &p, // dimension of DLM state space
    const int &t = -1,
    const unsigned int &nlag = 0,
    const unsigned int &N = 5000, // number of particles
    const double &delta_discount = 0.88,
    const bool &truncated = true,
    const bool &use_discount = false,
    const bool &use_custom_val = false);

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
    const Rcpp::NumericVector &W_par_in, // (init, prior type, prior par1, prior par2)
    const Rcpp::NumericVector &obs_par_in,
    const Rcpp::NumericVector &lag_par_in, // init/true values of (mu, sg2) or (rho, L)
    const unsigned int &nlag_in,
    const unsigned int &N, // number of particles
    const double &theta0_upbnd,
    const Rcpp::NumericVector &qProb,
    const double &delta_discount,
    const bool &truncated,
    const bool &use_discount,
    const bool &smoothing);

void mcs_poisson(
    arma::vec &psi,       // (n+1) x 1
    arma::vec &pmarg_y, // n x 1, marginal likelihood of y
    double &W,
    const arma::vec &ypad,        // (n+1) x 1, the observed response
    const arma::uvec &model_code, // (obs_code,link_code,transfer_code,gain_code,err_code)
    const arma::vec &obs_par,
    const arma::vec &lag_par, // init/true values of (mu, sg2) or (rho, L)
    const unsigned int &nlag_in = 20,
    const unsigned int &B = 10,   // length of the B-lag fixed-lag smoother
    const unsigned int &N = 5000, // number of particles
    const double &theta0_upbnd = 2.,
    const double &delta_discount = 0.88,
    const bool &truncated = true,
    const bool &use_discount = false);

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