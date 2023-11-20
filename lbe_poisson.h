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





/**
 * Calculate expected state evolution of time (t+1), given D[t],
 * aka, prediction of [t+1]
 *     E(theta[t+1] | D[t]) = gt(theta[t]).
 * This is an exact calculation.
 * using prior/previous information.
 *
 * Gt is only needed for the koyama case, in which Gt is invariant over time
 *
 * If input is m[t], then the output is a[t+1].
 *
 */
arma::mat update_at(
	arma::mat &Gt, // p x p
	const unsigned int p,
	const unsigned int gain_code,
	const unsigned int trans_code, // 0 - Koyck, 1 - Koyama, 2 - Solow
	const arma::mat &mt,		   // p x N, mt = (psi[t], theta[t], theta[t-1])
	const double y = NA_REAL,
	const double rho = NA_REAL,
	const unsigned int t = 9999);

/**
 * Update the Gt,
 * which is gradient of dynamic evolution function, gt(.).
 * 
 * Gt is calculated at mt.
*/
void update_Gt(
	arma::mat& Gt, // p x p
	const unsigned int gain_code, 
	const unsigned int trans_code, // 0 - Koyck, 1 - Koyama, 2 - Solow
	const arma::vec& mt, // p x 1
	const double y,  // obs
	const double rho);



/**
 * Calculate approximate state evolution variance of time [t+1], given D[t],
 * aka, prediction of [t+1].
 *      Var(theta[t+1] | D[t]) = G[t+1] * C[t] * t(G[t+1]) + W[t+1].
 * W[t+1] can be substituted with discount factor, delta.
*/
void update_Rt(
	arma::mat& Rt, // p x p
	const arma::mat& Ct, // p x p
	const arma::mat& Gt, // p x p
	const bool use_discount, // true if !R_IsNA(delta)
	const double W, // known evolution error of psi
	const double delta);


/**
 * Update the derivative of the state-to-obs function, f[t].
 * Only used for Koyama's transmission delay, i.e., discretized Hawkes.
*/
void update_Ft_koyama(
	arma::vec& Ft, // L x 1
	arma::vec& Fy, // L x 1
	const unsigned int t, // current time point
	const unsigned int L, // lag
	const arma::vec& Y,  // (nt+1) x 1, obs
	const arma::vec& Fphi);

/**
* Forward filter for univariate and multivariate input, Y.
*/
void forwardFilter(
	arma::mat& mt, // p x (nt+1), t=0 is the mean for initial value theta[0]
	arma::mat& at, // p x (nt+1)
	arma::cube& Ct, // p x p x (nt+1), t=0 is the var for initial value theta[0]
	arma::cube& Rt, // p x p x (nt+1)
	arma::cube& Gt, // p x p x (nt+1)
	arma::vec& alphat, // (nt+1) x 1
	arma::vec& betat, // (nt+1) x 1
	const unsigned int obs_code,
	const unsigned int link_code,
	const unsigned int trans_code,
	const unsigned int gain_code,
	const unsigned int n, // number of observations
	const unsigned int p, // dimension of the state space
	const arma::vec& Y, // (nt+1) x 1, the observation (scalar), n: num of obs
	const unsigned int L,
	const double rho,
	const double mu0,
	const double W,
	const double delta,
	const double delta_nb);


void backwardSmoother(
	arma::vec& ht, // (nt+1) x 1
	arma::vec& Ht, // (nt+1) x 1
	const unsigned int n,
	const unsigned int p,
	const arma::mat& mt, // p x (nt+1), t=0 is the mean for initial value theta[0]
	const arma::mat& at, // p x (nt+1)
	const arma::cube& Ct, // p x p x (nt+1), t=0 is the var for initial value theta[0]
	const arma::cube& Rt,
	const arma::cube& Gt,
	const double W,
	const double delta);


void backwardSampler(
	arma::vec& theta, // (nt+1) x 1
	const unsigned int n,
	const unsigned int p,
	const arma::mat& mt, // p x (nt+1), t=0 is the mean for initial value theta[0]
	const arma::mat& at, // p x (nt+1)
	const arma::cube& Ct, // p x p x (nt+1), t=0 is the var for initial value theta[0]
	const arma::cube& Rt, // p x p x (nt+1)
	const arma::cube& Gt,
	const double W,
	const double scale_sd);


Rcpp::List lbe_poisson(
	const arma::vec& Y, // nt x 1, the observed response
	const arma::uvec& model_code,
	const double rho,
    const unsigned int L,
	const double mu0,
    const double delta, 
	const double W,
	const Rcpp::Nullable<Rcpp::NumericVector>& m0_prior,
	const Rcpp::Nullable<Rcpp::NumericMatrix>& C0_prior,
	const double delta_nb,
	const double ci_coverage,
	const unsigned int npara,
	const double theta0_upbnd,
	const bool summarize_return);



Rcpp::List get_eta_koyama(
	const arma::vec& Y, // nt x 1, the observation (scalar), n: num of obs
	const arma::mat& mt, // p x (nt+1), t=0 is the mean for initial value theta[0]
	const arma::cube& Ct, // p x p x (nt+1)
	const unsigned int gain_code,
	const double mu0);



Rcpp::List get_optimal_delta(
	const arma::vec& Y, // nt x 1
	const arma::uvec& model_code,
	const arma::vec& delta_grid, // m x 1
	const double rho,
    const unsigned int L,
	const double mu0,
	const Rcpp::Nullable<Rcpp::NumericVector>& m0_prior,
	const Rcpp::Nullable<Rcpp::NumericMatrix>& C0_prior,
	const double delta_nb,
	const unsigned int npara,
	const double ci_coverage);



double lbe_What(
    const arma::vec& ht, // (nt+1) x 1
    const unsigned int nsample,
    const double aw,
    const double bw);


#endif