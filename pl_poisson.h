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
---- Model ----
<obs>   y[t] ~ Pois(lambda[t])
<link>  lambda[t] = theta[t]
<state> theta[t] = 2*rho*theta[t-1] - rho^2*theta[t-2] + (1-rho)^2*x[t-1]*exp(beta[t-1])
        beta[t] = beta[t-1] + omega[t]
        omega[t] ~ Normal(0,W)

Unknown parameters: W, theta[0:n], beta[0:n]
*/
Rcpp::List pl_pois_solow_eye_exp(
    const arma::vec& Y, // n x 1, the observed response
	const arma::vec& X, // n x 1, the exogeneous variable in the transfer function
    const double rho,
	const Rcpp::Nullable<Rcpp::NumericVector>& QPrior, // (aw, Rw), W ~ IG(aw/2, Rw/2), prior for v[1], the first state/evolution/state error/disturbance.
	const double Q_init,
	const double Q_true, // true value of state/evolution error variance
	const unsigned int N);

/*
---- Method ----
- Particle learning with rho known
- Using the DLM formulation

---- Model ----
<obs>   y[t] ~ Pois(lambda[t])
<link>  lambda[t] = theta[t]
<state> theta[t] = 2*rho*theta[t-1] - rho^2*theta[t-2] + (1-rho)^2*x[t-1]*exp(beta[t-1])
        beta[t] = beta[t-1] + omega[t]
        omega[t] ~ Normal(0,W)

Unknown parameters: theta[0:n], beta[0:n]
Kwg: Identity link, exponential state space
*/
Rcpp::List pl_pois_solow_eye_exp2(
    const arma::vec& Y, // n x 1, the observed response
	const arma::vec& X, // n x 1, the exogeneous variable in the transfer function
    const double rho,
	const Rcpp::Nullable<Rcpp::NumericVector>& QPrior, // (aw, Rw), W ~ IG(aw/2, Rw/2), prior for v[1], the first state/evolution/state error/disturbance.
	const double Q_init,
    const double Q_true,
	const unsigned int N, // number of particles
    const unsigned int B,
    const bool resample);



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
    const Rcpp::NumericVector& qProb,
    const Rcpp::NumericVector& ctanh,
    const double delta_nb,
    const bool verbose,
    const bool debug);



void mcs_poisson(
    arma::vec& R, // (n+1) x 1
    const arma::vec& Y, // n x 1, the observed response
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
    const Rcpp::NumericVector& ctanh,
    const double delta_nb);



/*
---- Method ----
- Auxiliary Particle filtering with all static parameters (rho,W) known
- Using the DLM formulation

---- Model ----
<obs>   y[t] ~ Pois(lambda[t])
<link>  lambda[t] = theta[t]
<state> theta[t] = 2*rho*theta[t-1] - rho^2*theta[t-2] + (1-rho)^2*x[t-1]*exp(beta[t-1])
        beta[t] = beta[t-1] + omega[t]
        omega[t] ~ Normal(0,W)

Unknown parameters: theta[0:n], beta[0:n]
Kwg: Identity link, exponential state space
*/
Rcpp::List apf_pois_solow_eye_exp(
    const arma::vec& Y, // n x 1, the observed response
	const arma::vec& X, // n x 1, the exogeneous variable in the transfer function
    const double rho,
	const double Q,
	const unsigned int N, // number of particles
    const unsigned int B);


/*
---- Method ----
- Storvik's Filtering with rho known
- Using the DLM formulation
- Storvik's Filter can be viewed as an extension of 
  bootstrap filter with inference of static parameter W

---- Model ----
<obs>   y[t] ~ Pois(lambda[t])
<link>  lambda[t] = theta[t]
<state> theta[t] = 2*rho*theta[t-1] - rho^2*theta[t-2] + (1-rho)^2*x[t-1]*max(beta[t-1],0)
        beta[t] = beta[t-1] + omega[t]
        omega[t] ~ Normal(0,W)

Unknown parameters: theta[0:n], beta[0:n], W
Kwg: Identity link, max(beta,0) state space
*/
Rcpp::List sf_pois_solow_eye_max(
    const arma::vec& Y, // n x 1, the observed response
	const arma::vec& X, // n x 1, the exogeneous variable in the transfer function
    const double rho,
	const Rcpp::Nullable<Rcpp::NumericVector>& QPrior, // (aw, Rw), W ~ IG(aw/2, Rw/2), prior for v[1], the first state/evolution/state error/disturbance.
	const double Q_init,
    const double Q_true,
	const unsigned int N, // number of particles
    const unsigned int B);


#endif