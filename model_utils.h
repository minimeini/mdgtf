#ifndef _MODEL_UTILS_H
#define _MODEL_UTILS_H

#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <algorithm>
#include <boost/math/special_functions/beta.hpp>
#include <RcppArmadillo.h>
#include <nlopt.h>
#include "nloptrAPI.h"


inline constexpr double EPS = 2.220446e-16;
inline constexpr double EPS8 = 1.e-10;
inline constexpr double UPBND = 700.;
inline constexpr double covid_m = 4.7;
inline constexpr double covid_s = 2.9;
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

void init_by_trans(
	unsigned int& p, // dimension of DLM state space
	unsigned int& L_,
	const unsigned int trans_code,
	const unsigned int L);



void init_by_trans(
	unsigned int& p, // dimension of DLM state space
	unsigned int& L_,
	arma::vec& Ft,
    arma::vec& Fphi,
    arma::mat& Gt,
	const unsigned int trans_code,
	const unsigned int L);



double binom(int n, int k);


/*
CDF of the log-normal distribution.
*/
double Pd(
    const double d,
    const double mu,
    const double sigmasq);


/*
Difference of the subsequent CDFs of the log-normal distribution, which is the PDF at the discrete scale.
*/
double knl(
    const double t,
    const double mu,
    const double sd2);


arma::vec knl(
	const arma::vec& tvec,
	const double mu,
	const double sd2);


/*
------ get_Fphi ------
Update the log-normal transmission delay distribution

------ Default settings ------
const double mu = 2.2204e-16
const double m = 4.7
const double s = 2.9
const unsigned int ModelCode = 0

*/
arma::vec get_Fphi(const unsigned int L);



/*
------ update_Fx ------

------ Default settings ------
const unsigned int ModelCode = 0
Checked. OK.
*/
arma::mat update_Fx(
	const unsigned int trans_code,
	const unsigned int n, // number of observations
	const arma::vec& Y, // n x 1, (y[1],...,y[n]), observtions
	const arma::vec& hph,// (n+1) x 1, derivative of the gain function at h[t]
	const double rho, // memory of previous state, for exponential and negative binomial transmission delay kernel
	const unsigned int L, // length of nonzero transmission delay for log-normal
	const Rcpp::Nullable<Rcpp::NumericVector>& Fphi_);



arma::vec update_theta0(
	const unsigned int trans_code,
	const unsigned int n, // number of observations
	const arma::vec& Y, // n x 1, observations, (Y[1],...,Y[n])
	const arma::vec& hhat, // (n+1) x 1, 1st Taylor expansion of h(psi[t]) at h[t]
	const double theta00, // Initial value of the transfer function block
	const double rho, // memory of previous state, for exponential and negative binomial transmission delay kernel
	const unsigned int L, // length of nonzero transmission delay for log-normal
	const Rcpp::Nullable<Rcpp::NumericVector>& Fphi_);


/*
------ update_theta ------
Update the hidden state vector

------ Formula ------
ModelCode = 0 and ApproxModel = true:
    theta[t] = Fphi[1]*y[t-1]*psi[t] + ... + Fphi[L]*y[t-L]*psi[t-L+1]
    >>>>>> This is using an alternate model which holds in most cases for the reproduction number,
since it barely drops below 0.
-------------------
*/
arma::vec update_theta(
	const unsigned int n,
	const arma::mat& Fx, // n x n
	const arma::vec& w);



double trigamma_obj(
	unsigned n,
	const double *x, 
	double *grad, 
	void *my_func_data);



double optimize_trigamma(double q);


typedef struct {
	double a1,a2,a3;
} coef_W;



double postW_gamma_obj(
	unsigned n,
	const double *x, // Wtilde = log(W)
	double *grad,
	void *my_func_data);


double optimize_postW_gamma(coef_W& coef);


double postW_deriv2(
	const double Wtilde,
	const double a2,
	const double a3);


/*
Testing example:
------ IMPLEMENTATION IN R
> optim(0.1,function(w){-(-5*w - 1.e2*exp(w)-0.5*exp(-w))},gr=function(w){-(-5-1.e2*exp(w)+0.5*exp(-w))},method="BFGS")
$par
[1] -2.995732
------

------ IMPLEMENTATION IN CPP
double test_postW_gamma(){
	coef_W coef[1] = {{-5.,1.e2,0.5}};
	double What = optimize_postW_gamma(coef[0]);
	return What;
}

> test_postW_gamma()
[1] -2.995733
------
*/
double test_postW_gamma(
	const double a1,
	const double a2,
	const double a3);


/**
 * Apply gain function h(.) to 
 * the latent random walk variable, psi[t]
*/
arma::mat psi2hpsi(
	const arma::mat& psi,
	const unsigned int gain_code,
	const Rcpp::NumericVector& coef);

/**
 * Apply gain function h(.) to
 * the latent random walk variable, psi[t]
 */
double psi2hpsi(
	const double psi,
	const unsigned int gain_code,
	const Rcpp::NumericVector& coef);



void hpsi_deriv(
	arma::mat& hpsi,
	const arma::mat& psi,
	const unsigned int gain_code,
	const Rcpp::NumericVector& coef);


double hpsi_deriv(
	const double psi,
	const unsigned int gain_code,
	const Rcpp::NumericVector& coef);


arma::mat hpsi_deriv(
	const arma::mat& psi,
	const unsigned int gain_code,
	const Rcpp::NumericVector& coef);


arma::mat hpsi2theta(
	const arma::mat& hpsi, // (n+1) x k
	const arma::vec& y, // n x 1
	const unsigned int trans_code,
	const double theta0,
	const double alpha,
	const unsigned int L,
	const double rho);



double loglike_obs(
	const double y, 
	const double lambda,
	const unsigned int obs_code,
	const double delta_nb,
	const bool return_log);



#endif