#ifndef _MODEL_KOYAMA_MAX_H
#define _MODEL_KOYAMA_MAX_H

#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <algorithm>
#include <RcppArmadillo.h>
#include <nlopt.h>
#include "nloptrAPI.h"
#include "hva_gradients.h"
#include "model_utils.h"


/*
------------------------
------ Model Code ------
------------------------

0 - (KoyamaMax) Log-normal transmission delay with maximum thresholding on the reproduction number.
1 - (KoyamaExp) Log-normal transmission delay with exponential function on the reproduction number.
2 - (Solow)
*/

/*
------ ModelCode = 0 -------
----------------------------
------ Koyama's Model ------
----------------------------

------ Discretized Hawkes Form ------
<obs> y[t] | lambda[t] ~ Pois(lambda[t])
<link> lambda[t] = phi[1] max(psi[t],0) y[t-1] + phi[2] max(psi[t-1],0) y[t-2] + ... + phi[L] max(psi[t-L+1],0) y[t-L]
<state> psi[t] = psi[t-1] + omega[t], omega[t] ~ N(0,W)


------ Dynamic Linear Model Form ------
<obs> y[t] | lambda[t] = Ft(theta[t]), 
    where Ft(theta[t]) = phi[1] y[t-1] max(theta[t,1],0) + ... + phi[L] y[t-L] max(theta[t,L],0)
<Link> theta[t] = G theta[t-1] + Omega[t], 
    where G = 1 0 ... 0 0
              1 0 ... 0 0
              . .     . .
              .   .   . .
              .     . . .
              0 0 ... 1 0
        
    and Omega[t] = (omega[t],0,...,0) ~ N(0,W[t]), W[t][1,1] = W and 0 otherwise.

-----------------------
------ Inference ------
-----------------------

1. [x] Linear Bayes Approximation with first order Taylor expansion
2. [x] Linear Bayes Approximation with second order Taylor expansion >> doesn't exist because the Hessian will be exactly zero
3. [x] MCMC Disturbance sampler
4. [x] Sequential Monte Carlo filtering and smoothing
5. [x] Vanila variational Bayes
6. [x] Hybrid variational Bayes

*/



/*
Linear Bayes approximation based filtering for koyama's model with exponential on the transmission delay.
- Observational linear approximation - This function uses 1st order Taylor expansion for the nonlinear Ft at the observational equation.
- Known evolution variance - This function assumes the evolution variance, W, is fixed and known
*/
void forwardFilterW(
	const arma::vec& Y, // n x 1, the observation (scalar), n: num of obs
    const arma::mat& G, // L x L, state transition matrix
    const arma::vec& Fphi, // L x 1, state-to-obs nonlinear mapping vector
	const double W, // state variance
	arma::mat& mt, // L x (n+1), t=0 is the mean for initial value theta[0]
	arma::mat& at, // L x (n+1)
	arma::cube& Ct, // L x L x (n+1), t=0 is the var for initial value theta[0]
	arma::cube& Rt);


/*
Linear Bayes approximation based filtering for koyama's model with exponential on the transmission delay.
- Observational linear approximation - This function uses 1st order Taylor expansion for the nonlinear Ft at the observational equation.
- Evolutional discounting - This function assumes the evolution variance is time-varying, which we are not interested in. Instead, 
we preselect a discount factor, delta, to account for the time-varying evolutional variance.
*/
void forwardFilterWt(
	const arma::vec& Y, // n x 1, the observation (scalar), n: num of obs
    const arma::mat& G, // L x L, state transition matrix
    const arma::vec& Fphi, // L x 1, state-to-obs nonlinear mapping vector
	const double delta, // state variance
	arma::mat& mt, // L x (n+1), t=0 is the mean for initial value theta[0]
	arma::mat& at, // L x (n+1)
	arma::cube& Ct, // L x L x (n+1), t=0 is the var for initial value theta[0]
	arma::cube& Rt);


/*
---- Method ----
- Koyama's Discretized Hawkes Process
- Ordinary Kalman filtering with all static parameters known
- Using the DLM formulation
- 1st order linear approximation at the observational equation
- Fixed and known evolutional variance

---- Model ----
<obs>   y[t] ~ Pois(lambda[t]), t=1,...,n
<link>  lambda[t] = phi[1]*y[t-1]*exp(psi[t]) + ... + phi[L]*y[t-L]*exp(psi[t-L+1]),t=1,...,n
<state> psi[t] = psi[t-1] + omega[t], t=1,...,n
        omega[t] ~ Normal(0,W)

<prior> theta_till[0] ~ Norm(m0, C0)
        W ~ IG(aw,bw)
*/
void backwardSmootherW(
	const arma::mat& G, // L x L, state transition matrix
	const arma::mat& mt, // L x (n+1), t=0 is the mean for initial value theta[0]
	const arma::mat& at, // L x (n+1)
	const arma::cube& Ct, // L x L x (n+1), t=0 is the var for initial value theta[0]
	const arma::cube& Rt, // L x L x (n+1)
	arma::mat& ht, // L x (n+1)
    arma::cube& Ht);

/*
---- Method ----
- Koyama's Discretized Hawkes Process
- Ordinary Kalman filtering with all static parameters known
- Using the DLM formulation
- 1st order linear approximation at the observational equation
- Discounting for the time-varying and unknown evolutional variance

---- Model ----
<obs>   y[t] ~ Pois(lambda[t]), t=1,...,n
<link>  lambda[t] = phi[1]*y[t-1]*exp(psi[t]) + ... + phi[L]*y[t-L]*exp(psi[t-L+1]),t=1,...,n
<state> psi[t] = psi[t-1] + omega[t], t=1,...,n
        omega[t] ~ Normal(0,W)

<prior> theta_till[0] ~ Norm(m0, C0)
        W ~ IG(aw,bw)
*/
void backwardSmootherWt(
	const arma::mat& G, // L x L, state transition matrix
	const arma::mat& mt, // L x (n+1), t=0 is the mean for initial value theta[0]
	const arma::mat& at, // L x (n+1)
	const arma::cube& Ct, // L x L x (n+1), t=0 is the var for initial value theta[0]
	const arma::cube& Rt, // L x L x (n+1)
	const double delta, // discount factor
	arma::vec& ht, // (n+1) x 1
    arma::vec& Ht);


/*
---- Method ----
- Koyama's Discretized Hawkes Process
- Ordinary Kalman filtering and smoothing
- with all static parameters known
- Using the DLM formulation

---- Model ----
<obs>   y[t] ~ Pois(lambda[t]), t=1,...,n
<link>  lambda[t] = phi[1]*y[t-1]*max(psi[t],0) + ... + phi[L]*y[t-L]*max(psi[t-L+1],0),t=1,...,n
<state> psi[t] = psi[t-1] + omega[t], t=1,...,n
        omega[t] ~ Normal(0,W)

<prior> theta_till[0] ~ Norm(m0, C0)
        W ~ IG(aw,bw)
*/
//' @export
// [[Rcpp::export]]
Rcpp::List lbe_pois_koyama_max_W(
	const arma::vec& Y, // n x 1, the observed response
    const unsigned int L,
    const double W, 
    const Rcpp::Nullable<Rcpp::NumericVector>& theta0_init);


/*
---- Method ----
- Koyama's Discretized Hawkes Process
- Ordinary Kalman filtering and smoothing
- with all static parameters known
- Using the DLM formulation

---- Model ----
<obs>   y[t] ~ Pois(lambda[t]), t=1,...,n
<link>  lambda[t] = phi[1]*y[t-1]*exp(psi[t]) + ... + phi[L]*y[t-L]*exp(psi[t-L+1]),t=1,...,n
<state> psi[t] = psi[t-1] + omega[t], t=1,...,n
        omega[t] ~ Normal(0,Wt)

<prior> theta_till[0] ~ Norm(m0, C0)
        W[0] ~ IG(aw,bw)
*/
//' @export
// [[Rcpp::export]]
Rcpp::List lbe_pois_koyama_max_Wt(
	const arma::vec& Y, // n x 1, the observed response
    const unsigned int L,
    const double delta, 
    const Rcpp::Nullable<Rcpp::NumericVector>& theta0_init);

/*
---- Method ----
- Monte Carlo Filtering with static parameter W known
- Using the discretized Hawkes formulation
- Reference: Kitagawa and Sato at Ch 9.3.4 of Doucet et al.
- Note: 
    1. Initial version is copied from the `bf_pois_koyama_exp`
    2. This is intended to be the modified Rcpp version of `hawkes_state_space.R`
    3. The difference is that the R version the states are backshifted L times, 
        where L is the maximum transmission delay to be considered.
        To make this a Monte Carlo filtering, we don't backshifting/smoot.

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
//' @export
// [[Rcpp::export]]
arma::mat mcf_pois_koyama_max_W(
    const arma::vec& Y, // n x 1, the observed response
	const double W,
    const unsigned int L, // number of lags
	const unsigned int N, // number of particles
    const double rho, // parameter for negative binomial likelihood
    const unsigned int obstype);

/*
---- Method ----
- Monte Carlo Smoothing with static parameter W known
- B-lag fixed-lag smoother (Anderson and Moore 1979)
- Using the discretized Hawkes formulation
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
//' @export
// [[Rcpp::export]]
arma::mat mcs_pois_koyama_max_W(
    const arma::vec& Y, // n x 1, the observed response
	const double W,
    const unsigned int L, // number of lags
    const unsigned int B, // length of the B-lag fixed-lag smoother (Anderson and Moore 1979; Kitagawa and Sato)
	const unsigned int N, // number of particles
    const double rho, // parameter for negative binomial likelihood
    const unsigned int obstype);



//' @export
// [[Rcpp::export]]
Rcpp::List mfva_koyama_max_Wt(
    const arma::vec& Y, // n x 1, the observed response
    const unsigned int L,
    const unsigned int niter, // number of iterations for variational inference
    const double W0, 
    const double delta,
    const Rcpp::Nullable<Rcpp::NumericVector>& W_init, // (aw0:shape, bw0:rate)
    const Rcpp::Nullable<Rcpp::NumericVector>& theta0_init,
    const unsigned int Ftype);


/*
---- Method ----
- Hybrid variational approximation
- Using the DLM formulation

---- Model ----
<obs>   y[t] ~ Pois(lambda[t])
<link>  lambda[t] = phi[1]*y[t-1]*max(psi[t],0) + ... + phi[L]*y[t-L]*max(psi[t-L+1],0)
<state> psi[t] = psi[t-1] + omega[t]
        omega[t] ~ Normal(0,W)

Unknown parameters: psi[1:n], W
Known parameters: phi[1:L]
Kwg: Identity link, exp(psi) state space
*/
//' @export
// [[Rcpp::export]]
Rcpp::List hva_koyama_max_W(
    const arma::vec& Y, // n x 1, the observed response
    const unsigned int niter, // number of iterations for variational inference
    const unsigned int L,
    const double W0, 
    const double delta,
    const Rcpp::Nullable<Rcpp::NumericVector>& W_init,
    const unsigned int Ftype);






/*
Reference: Gamerman (1998); Alves et al. (2010).
ModelCode = 0
*/
//' @export
// [[Rcpp::export]]
Rcpp::List mcmc_disturbance_pois_koyama_max(
	const arma::vec& Y, // n x 1, the observed response
    const unsigned int L, // lag of transmission delay
    const Rcpp::Nullable<Rcpp::NumericVector>& lambda0_in, // baseline intensity
	const Rcpp::Nullable<Rcpp::NumericVector>& WPrior, // (nv,Sv), prior for W~IG(shape=nv/2,rate=nv*Sv/2), W is the evolution error variance
	const Rcpp::Nullable<Rcpp::NumericVector>& w1Prior, // (aw, Rw), w1 ~ N(aw, Rw), prior for w[1], the first state/evolution/state error/disturbance.
	const double W_init,
	const Rcpp::Nullable<Rcpp::NumericVector>& wt_init, // n x 1
	const double W_true, // true value of state/evolution error variance
	const Rcpp::Nullable<Rcpp::NumericVector>& wt_true, // n x 1, true value of system/evolution/state error/disturbance
	const unsigned int nburnin,
	const unsigned int nthin,
	const unsigned int nsample);


#endif