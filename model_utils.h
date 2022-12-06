#ifndef _MODEL_UTILS_H
#define _MODEL_UTILS_H

#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <algorithm>
#include <RcppArmadillo.h>
#include <nlopt.h>
#include "nloptrAPI.h"


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


/*
------ get_Fphi ------
Update the log-normal transmission delay distribution

------ Default settings ------
const double mu = 2.2204e-16
const double m = 4.7
const double s = 2.9
const unsigned int ModelCode = 0

*/
arma::vec get_Fphi(
    const unsigned int L, // number of Lags to be considered
    const double mu,
    const double m,
    const double s);


arma::vec get_Fphi(const unsigned int L);



/*
------ update_Fx ------

------ Default settings ------
const unsigned int ModelCode = 0
*/
arma::mat update_Fx(
	const unsigned int ModelCode,
	const unsigned int n, // number of observations
	const arma::vec& Y, // n x 1, (y[1],...,y[n]), observtions
	const double rho, // memory of previous state, for exponential and negative binomial transmission delay kernel
	const unsigned int L, // length of nonzero transmission delay for log-normal
	const Rcpp::Nullable<Rcpp::NumericVector>& ht_, // (n+1) x 1, (h[0],h[1],...,h[n]), smoothing means of (psi[0],...,psi[n]) | Y
	const Rcpp::Nullable<Rcpp::NumericVector>& Fphi_);



arma::vec update_theta0(
	const unsigned int ModelCode,
	const unsigned int n, // number of observations
	const arma::vec& Y, // n x 1, observations, (Y[1],...,Y[n])
	const double theta00, // Initial value of the transfer function block
	const double psi0, // Initial value of the evolution error
	const double rho, // memory of previous state, for exponential and negative binomial transmission delay kernel
	const unsigned int L, // length of nonzero transmission delay for log-normal
	const Rcpp::Nullable<Rcpp::NumericVector>& ht_, // (n+1) x 1, (h[0],h[1],...,h[n]), smoothing means of (psi[0],...,psi[n]) | Y
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
	const arma::vec& w,
    const unsigned int ModelCode);


/*
------ update_lambda ------
Update the intensity function

------ Formula ------
ModelCode = 0 and ApproxModel = true: 
    lambda[t] = max(lambda[0] + theta[t], 0)
    >>>>>> This is using an alternate model which holds in most cases for the reproduction number,
since it barely drops below 0.
*/
arma::vec update_lambda(
    const arma::vec& theta, // n x 1
    const arma::vec& lambda0, // n x 1
    const unsigned int ModelCode,
    const bool ApproxModel);



/*
------ KoyckEye with ModelCode == 8 and Y[t] == 1 ------
Koyck's exponential decaying kernel with no external input or y[t]==1.
*/
arma::mat update_Fx0(
	const unsigned int n,
	const double G);


/*
------ KoyckEye with ModelCode == 8 ------
Koyck's exponential transmission kernel using y[t-1] as input at time t
*/
arma::mat update_Fx1(
	const unsigned int n,
	const double G,
	const arma::vec& X);


/*
------ SolowEye with ModelCode == 7 ------
Solow's negative-binomial transmission kernel using y[t-1] as input at time t
*/
arma::mat update_Fx_Solow(
	const unsigned int n,
	const double rho,
	const arma::vec& X);



/*
------ KoyamaEye with ModelCode == 0 ------
Solow's negative-binomial transmission kernel using y[t-1] as input at time t
*/
arma::mat update_Fx_Koyama(
	const unsigned int n, // number of observations
    const unsigned int L, // number of transmission delays
	const arma::vec& Fphi, // L x 1, Fphi = (phi[1],...,phi[L])
	const arma::vec& Y);



double trigamma_obj(
	unsigned n,
	const double *x, 
	double *grad, 
	void *my_func_data);



double optimize_trigamma(double q);


#endif