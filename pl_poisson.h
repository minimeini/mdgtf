#ifndef _PL_POISSON_H
#define _PL_POISSON_H

#include "lbe_poisson.h"

/*
------------------------
------ Model Code ------
------------------------

0 - (KoyamaMax) Identity link    + log-normal transmission delay kernel        + ramp function (aka max(x,0)) on gain factor (psi)
1 - (KoyamaExp) Identity link    + log-normal transmission delay kernel        + exponential function on gain factor
2 - (SolowMax)  Identity link    + negative-binomial transmission delay kernel + ramp function on gain factor
3 - (SolowExp)  Identity link    + negative-binomial transmission delay kernel + exponential function on gain factor
4 - (KoyckMax)  Identity link    + exponential transmission delay kernel       + ramp function on gain factor
5 - (KoyckExp)  Identity link    + exponential transmission delay kernel       + exponential function on gain factor
6 - (KoyamaEye) Exponential link + log-normal transmission delay kernel        + identity function on gain factor
7 - (SolowEye)  Exponential link + negative-binomial transmission delay kernel + identity function on gain factor
8 - (KoyckEye)  Exponential link + exponential transmission delay kernel       + identity function on gain factor
*/



/*
---- Method ----
- Monte Carlo Smoothing with static parameter W known
- B-lag fixed-lag smoother (Anderson and Moore 1979)
- Using the DLM formulation
- Reference: 
    Kitagawa and Sato at Ch 9.3.4 of Doucet et al.
    Check out Algorithm step 2-4L for the explanation of B-lag fixed-lag smoother
- Note: 
    1. Initial version is copied from the `bf_pois_koyama_exp`
    2. This is intended to be the Rcpp version of `hawkes_state_space.R`
    3. The difference is that the R version the states are backshifted L times, where L is the maximum transmission delay to be considered.
        To make this a Monte Carlo smoothing, we need to resample the states n times, where n is the total number of temporal observations.

---- Algorithm ----

1. Generate a random number psi[0](j) ~ an initial distribution.
2. Repeat the following steps for t = 1,...,n:
    2-1. Generate a random number omega[t] ~ Normal(0,W)


---- Model ----
<obs>   y[t] ~ Pois(lambda[t])
<link>  lambda[t] = phi[1]*y[t-1]*exp(psi[t]) + ... + phi[L]*y[t-L]*exp(psi[t-L+1])
<state> psi[t] = psi[t-1] + omega[t]
        omega[t] ~ Normal(0,W)

Unknown parameters: psi[1:n]
Known parameters: W, phi[1:L]
Kwg: Identity link, exp(psi) state space
*/


void init_Ft(
    arma::vec &Ft, 
    const arma::vec &ypad, 
    const arma::vec &Fphi,
    const unsigned int &t, 
    const unsigned int &p);


Rcpp::List mcs_poisson(
    const arma::vec &Y, // nt x 1, the observed response
    const arma::uvec &model_code,
    const double W_true, // Use discount factor if W is not given
    const double rho,
    const unsigned int L, // number of lags
    const double mu0,
    const unsigned int B,                                // length of the B-lag fixed-lag smoother (Anderson and Moore 1979; Kitagawa and Sato)
    const unsigned int N,                                // number of particles
    const Rcpp::Nullable<Rcpp::NumericVector> &m0_prior, // mean of normal prior for theta0
    const Rcpp::Nullable<Rcpp::NumericMatrix> &C0_prior, // variance of normal prior for theta0
    const double theta0_upbnd,                           // Upper bound of uniform prior for theta0
    const Rcpp::NumericVector &qProb,
    const double delta_nb,
    const double delta_discount);

void smc_propagate_bootstrap(
    arma::mat &Theta_new, // p x N
    arma::vec &weights,   // N x 1
    double &wt,
    const double &y_new,
    const double &y_old,        // (n+1) x 1, the observed response
    const arma::mat &Theta_old, // p x N
    const arma::mat &Gt,        // no need to update Gt
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
    const unsigned int t = 9999);

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
    const unsigned int L, // number of lags
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
    arma::mat& R, // (n+1) x 2, (psi,theta)
    arma::vec& pmarg_y, // n x 1, marginal likelihood of y
    const arma::vec& ypad, // (n+1) x 1, the observed response
    const arma::uvec& model_code, // (obs_code,link_code,transfer_code,gain_code,err_code)
	const double W,
    const double rho,
    const unsigned int L, // number of lags
    const double mu0,
    const unsigned int B, // length of the B-lag fixed-lag smoother (Anderson and Moore 1979; Kitagawa and Sato)
	const unsigned int N, // number of particles
    const arma::vec& m0_prior,
	const arma::mat& C0_prior,
    const double theta0_upbnd,
    const double delta_nb,
    const double delta_discount);




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