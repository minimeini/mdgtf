#ifndef _LBE_POISSON_H
#define _LBE_POISSON_H

#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <algorithm>
#include <RcppArmadillo.h>
#include <nlopt.h>
#include "nloptrAPI.h"
#include "model_utils.h"



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



arma::vec update_at(
	const unsigned int p,
	const unsigned int GainCode,
	const unsigned int TransferCode, // 0 - Koyck, 1 - Koyama, 2 - Solow
	const arma::vec& mt, // p x 1, mt = (psi[t], theta[t], theta[t-1])
	const arma::mat& Gt, // p x p
	const Rcpp::NumericVector& ctanh, // 3 x 1, coefficients for the hyperbolic tangent gain function
	const double alpha,
	const double y,  // n x 1
	const double rho);



void update_Gt(
	arma::mat& Gt, // p x p
	const unsigned int GainCode, 
	const unsigned int TransferCode, // 0 - Koyck, 1 - Koyama, 2 - Solow
	const arma::vec& mt, // p x 1
	const Rcpp::NumericVector& ctanh, // 3 x 1
	const double alpha,
	const double y,  // obs
	const double rho);



void update_Rt(
	arma::mat& Rt, // p x p
	const arma::mat& Ct, // p x p
	const arma::mat& Gt, // p x p
	const bool use_discount, // true if !R_IsNA(delta)
	const double W, // known evolution error of psi
	const double delta);


void update_Ft(
	arma::vec& Ft, // L x 1
	arma::vec& Fy, // L x 1
	const unsigned int TransferCode, // 0 - Koyck, 1 - Koyama, 2 - Solow
	const unsigned int t, // current time point
	const unsigned int L, // lag
	const arma::vec& Y,  // (n+1) x 1, obs
	const arma::vec& Fphi,
	const double alpha);


/* Forward Filtering */
void forwardFilter(
	arma::mat& mt, // p x (n+1), t=0 is the mean for initial value theta[0]
	arma::mat& at, // p x (n+1)
	arma::cube& Ct, // p x p x (n+1), t=0 is the var for initial value theta[0]
	arma::cube& Rt, // p x p x (n+1)
	arma::cube& Gt, // p x p x (n+1)
	arma::vec& alphat, // (n+1) x 1
	arma::vec& betat, // (n+1) x 1
	const unsigned int ModelCode,
	const unsigned int TransferCode,
	const unsigned int n, // number of observations
	const unsigned int p, // dimension of the state space
	const arma::vec& Y, // (n+1) x 1, the observation (scalar), n: num of obs
	const Rcpp::NumericVector& ctanh, // 3 x 1, coefficients for the hyperbolic tangent gain function
	const double alpha,
	const unsigned int L,
	const double rho,
	const double mu0,
	const double W,
	const double delta,
	const double delta_nb,
	const unsigned int obs_type,
	const bool debug);


void backwardSmoother(
	arma::vec& ht, // (n+1) x 1
	arma::vec& Ht, // (n+1) x 1
	const unsigned int n,
	const unsigned int p,
	const arma::mat& mt, // p x (n+1), t=0 is the mean for initial value theta[0]
	const arma::mat& at, // p x (n+1)
	const arma::cube& Ct, // p x p x (n+1), t=0 is the var for initial value theta[0]
	const arma::cube& Rt,
	const arma::cube& Gt,
	const double W,
	const double delta);


void backwardSampler(
	arma::vec& theta, // (n+1) x 1
	const unsigned int n,
	const unsigned int p,
	const arma::mat& mt, // p x (n+1), t=0 is the mean for initial value theta[0]
	const arma::mat& at, // p x (n+1)
	const arma::cube& Ct, // p x p x (n+1), t=0 is the var for initial value theta[0]
	const arma::cube& Rt, // p x p x (n+1)
	const arma::cube& Gt,
	const double W,
	const double scale_sd);


Rcpp::List lbe_poisson(
	const arma::vec& Y, // n x 1, the observed response
	const unsigned int ModelCode,
	const double rho,
    const unsigned int L,
	const double mu0,
    const double delta, 
	const double W,
	const Rcpp::Nullable<Rcpp::NumericVector>& m0_prior,
	const Rcpp::Nullable<Rcpp::NumericMatrix>& C0_prior,
	const Rcpp::NumericVector& ctanh,
	const double alpha,
	const double delta_nb,
	const unsigned int obs_type,
	const bool summarize_return,
	const bool debug);



Rcpp::List get_eta_koyama(
	const arma::vec& Y, // n x 1, the observation (scalar), n: num of obs
	const arma::mat& mt, // p x (n+1), t=0 is the mean for initial value theta[0]
	const arma::cube& Ct, // p x p x (n+1)
	const unsigned int ModelCode,
	const double alpha,
	const double mu0);



Rcpp::List get_optimal_delta(
	const arma::vec& Y, // n x 1
	const unsigned int ModelCode,
	const arma::vec& delta_grid, // m x 1
	const double rho,
    const unsigned int L,
	const double mu0,
	const Rcpp::Nullable<Rcpp::NumericVector>& m0_prior,
	const Rcpp::Nullable<Rcpp::NumericMatrix>& C0_prior,
	const Rcpp::NumericVector& ctanh,
	const double alpha,
	const double delta_nb,
	const unsigned int obs_type);



double lbe_What(
    const arma::vec& ht, // (n+1) x 1
    const unsigned int nsample,
    const double aw,
    const double bw);


#endif