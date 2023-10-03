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
Rcpp::List mcs_poisson(
    const arma::vec& Y, // n x 1, the observed response
    const arma::uvec& model_code,
	const double W,
    const double rho,
    const double alpha,
    const unsigned int L, // number of lags
    const double mu0,
    const unsigned int B, // length of the B-lag fixed-lag smoother (Anderson and Moore 1979; Kitagawa and Sato)
	const unsigned int N, // number of particles
    const Rcpp::Nullable<Rcpp::NumericVector>& m0_prior,
	const Rcpp::Nullable<Rcpp::NumericMatrix>& C0_prior,
    const double theta0_upbnd,
    const Rcpp::NumericVector& qProb,
    const Rcpp::NumericVector& ctanh,
    const double delta_nb,
    const double delta_discount,
    const bool resample,
    const bool verbose,
    const bool debug);




void bf_poisson(
    arma::cube& theta_stored, // p x N x (n+1)
    arma::vec& w_stored, // (n+1) x 1
    arma::mat& Gt,
    const arma::vec& ypad, // (n+1) x 1, the observed response
    const arma::uvec& model_code,
	const double W,
    const double mu0,
    const double rho,
    const double alpha,
    const unsigned int p, // dimension of DLM state space
    const unsigned int L, // number of lags
	const unsigned int N, // number of particles
    const arma::vec& Ft0,
    const arma::vec& Fphi,
    const Rcpp::NumericVector& ctanh,
    const double delta_nb);



Rcpp::List ffbs_poisson(
    const arma::vec& Y, // n x 1, the observed response
    const arma::uvec& model_code,
	const double W,
    const double rho,
    const double alpha,
    const unsigned int L, // number of lags
    const double mu0,
	const unsigned int N, // number of particles
    const Rcpp::Nullable<Rcpp::NumericVector>& m0_prior,
	const Rcpp::Nullable<Rcpp::NumericMatrix>& C0_prior,
    const double theta0_upbnd,
    const Rcpp::NumericVector& qProb,
    const Rcpp::NumericVector& ctanh,
    const double delta_nb,
    const double delta_discount,
    const unsigned int npara,
    const bool resample, // true = auxiliary particle filtering; false = bootstrap filtering
    const bool smoothing, // true = particle smoothing; false = no smoothing
    const bool verbose,
    const bool debug);


Rcpp::List pmmh_poisson(
    const arma::vec& Y, // n x 1, the observed response
    const arma::uvec& model_code,
	const arma::uvec& eta_select, // 4 x 1, indicator for unknown (=1) or known (=0)
    const arma::vec& eta_init, // 4 x 1, if true/initial values should be provided here
    const arma::uvec& eta_prior_type, // 4 x 1
    const arma::mat& eta_prior_val, // 2 x 4, priors for each element of eta
    const double alpha,
    const unsigned int L, // number of lags
	const unsigned int N, // number of particles
    const unsigned int nsample,
    const unsigned int nburnin,
    const unsigned int nthin,
    const Rcpp::Nullable<Rcpp::NumericVector>& m0_prior,
	const Rcpp::Nullable<Rcpp::NumericMatrix>& C0_prior,
    const Rcpp::NumericVector& qProb,
    const double MH_sd,
    const double delta_nb,
    const bool verbose,
    const bool debug);



void mcs_poisson(
    arma::mat& R, // (n+1) x 2, (psi,theta)
    arma::vec& pmarg_y, // n x 1, marginal likelihood of y
    const arma::vec& ypad, // (n+1) x 1, the observed response
    const arma::uvec& model_code, // (obs_code,link_code,transfer_code,gain_code,err_code)
	const double W,
    const double rho,
    const double alpha,
    const unsigned int L, // number of lags
    const double mu0,
    const unsigned int B, // length of the B-lag fixed-lag smoother (Anderson and Moore 1979; Kitagawa and Sato)
	const unsigned int N, // number of particles
    const Rcpp::Nullable<Rcpp::NumericVector>& m0_prior,
	const Rcpp::Nullable<Rcpp::NumericMatrix>& C0_prior,
    const double theta0_upbnd,
    const Rcpp::NumericVector& ctanh,
    const double delta_nb,
    const double delta_discount);




/*
Particle Learning
- Reference: Carvalho et al., 2010.

- eta = (W, mu[0], rho, M)
*/
Rcpp::List pl_poisson(
    const arma::vec& Y, // n x 1, the observed response
    const arma::uvec& model_code,
    const arma::uvec& eta_select, // 4 x 1, indicator for unknown (=1) or known (=0)
    const arma::vec& eta_init, // 4 x 1, if true/initial values should be provided here
    const arma::uvec& eta_prior_type, // 4 x 1
    const arma::mat& eta_prior_val, // 2 x 4, priors for each element of eta
    const double alpha,
    const unsigned int L, // number of lags
	const unsigned int N, // number of particles
    const Rcpp::Nullable<Rcpp::NumericVector>& m0_prior,
	const Rcpp::Nullable<Rcpp::NumericMatrix>& C0_prior,
    const Rcpp::NumericVector& qProb,
    const double delta_nb,
    const bool verbose,
    const bool debug);


    #endif