#ifndef _MODEL_UTILS_H
#define _MODEL_UTILS_H

#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
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
------ Model Code ------

obs_code
- 0: nbinom
- 1: poisson
- 2: nbinom_p
- 3: gaussian

link_code
- 0: identity
- 1: exponential

trans_code
- 0: koyck
- 1: koyama
- 2: solow
- 3: vanilla

gain_code
- 0: ramp
- 1: exponential
- 2: identity
- 3: softplus

err_code
- 0: gaussian
- 1: laplace
- 2: cauchy
- 3: left_skewed_normal



------ ModelCode -------
*/

inline void tolower(std::string &S)
{
	for (char &x : S)
	{
		x = tolower(x);
	}
	return;
}

inline const char *const bool2string(bool b)
{
	return b ? "true" : "false";
}

arma::uvec get_model_code(
	const std::string &obs_dist,
	const std::string &link_func,
	const std::string &trans_func,
	const std::string &gain_func,
	const std::string &err_dist);

void get_model_code(
	unsigned int &obs_code,
	unsigned int &link_code,
	unsigned int &trans_code,
	unsigned int &gain_code,
	unsigned int &err_code,
	const arma::uvec &model_code);



void bound_check(
	const arma::mat &input,
	const std::string &name = "function",
	const bool &check_zero = false,
	const bool &check_negative = false);


void bound_check(
	const double &input,
	const std::string &name = "function",
	const bool &check_zero = false,
	const bool &check_negative = false);



bool bound_check(
	const double &input,
	const bool &check_zero,
	const bool &check_negative);



void bound_check(
	const double &input, 
	const std::string &name,
	const double &lobnd, 
	const double &upbnd);


arma::uvec sample(
	const unsigned int n,
	const unsigned int size,
	const arma::vec &weights,
	bool replace = true,
	bool zero_start = true);

unsigned int sample(
	const int n,
	const arma::vec &weights,
	bool zero_start = true);

unsigned int sample(
	const int n,
	bool zero_start);

arma::vec init_Ft(
	const unsigned int &p, // dimension of DLM state space
	const unsigned int &trans_code);

void init_Gt(
	arma::mat &Gt,
	const arma::vec &lag_par,
	const unsigned int &p = 20,
	const unsigned int &nlag = 20,
	const bool &truncated = true);

void init_Gt(
	arma::cube &Gt,
	const arma::vec &lag_par,
	const unsigned int &p = 20,
	const unsigned int &nlag = 20,
	const bool &truncated = true);

double binom(double n, double k);


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
double dlognorm0(
	const double &lag,
	const double &mu,
	const double &sd2);

arma::vec dlognorm(
	const double &nlag,
	const double &mu,
	const double &sd2);


double dnbinom0(
	const double &lag, // starting from 1
	const double &rho,
	const double &L_order);



arma::vec dnbinom(
	const double &nlag,
	const double &rho,
	const double &L_order);



unsigned int get_truncation_nlag(
	const unsigned int &trans_code,
	const double &err_margin,
	const unsigned int &L_order,
	const double &rho);

arma::vec dlags(
	const unsigned int &nlags,
	const arma::vec &params);

double cross_entropy(
	const unsigned int &nlags,
	const arma::vec &params_p, // 3 x 1, params1[0] = trans_code: type of distribution
	const arma::vec &params_q);

arma::vec match_params(
	const arma::vec &params_in,
	const unsigned int &trans_code_out,
	const arma::vec &par1_grid, // n1 x 1
	const arma::vec &par2_grid,	// n2 x 1
	const unsigned int &nlags);

unsigned int get_truncation_nlag(
	const unsigned int &trans_code,
	const double &err_margin,
	const Rcpp::NumericVector &lag_par);


arma::vec get_Fphi(
	const unsigned int &nlag,		 // number of Lags
	const arma::vec &lag_par,
	const unsigned int &trans_code);

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
// double test_postW_gamma(
// 	const double a1,
// 	const double a2,
// 	const double a3);


/**
 * Apply gain function h(.) to 
 * the latent random walk variable, psi[t]
*/
arma::mat psi2hpsi(
	const arma::mat& psi,
	const unsigned int &gain_code);

/**
 * Apply gain function h(.) to
 * the latent random walk variable, psi[t]
 */
double psi2hpsi(
	const double &psi,
	const unsigned int &gain_code);



void hpsi_deriv(
	arma::mat& hpsi,
	const arma::mat& psi,
	const unsigned int &gain_code);


double hpsi_deriv(
	const double &psi,
	const unsigned int &gain_code);


arma::mat hpsi_deriv(
	const arma::mat& psi,
	const unsigned int &gain_code);

arma::vec get_theta_coef_solow(const unsigned int &L, const double &rho);

// double theta_new_solow_ar(
// 	const arma::vec &theta_past, // L x 1
// 	const double &hpsi_cur,		 // (n+1) x 1
// 	const double &yt_prev,		 // y[t-1]
// 	const unsigned int &tidx,	 // time index t, t=1,...,n
// 	const double &rho,
// 	const unsigned int &L);

// double theta_new_solow_ar(
// 	const arma::vec &theta_pad, // (n+L) x 1
// 	const arma::vec &hpsi_pad,	// (n+1) x 1
// 	const arma::vec &ypad,		// (n+1) x 1
// 	const unsigned int &tidx,	// t = 1, ..., n
// 	const unsigned int &L,
// 	const arma::vec &coef,
// 	const double &cnst);

arma::vec Fphi_times_hpsi(
	const arma::vec &Fphi_sub, // nelem x 1
	const arma::vec &hpsi_sub);

unsigned int theta_nelem(
	const unsigned int &nobs,
	const unsigned int &nlag_in,
	const unsigned int &tidx,
	const bool &truncated);

void theta_subset(
	arma::vec &Fphi_sub,
	arma::vec &hpsi_sub,
	arma::vec &ysub,
	const arma::vec &hpsi_pad, // (n+1) x 1
	const arma::vec &ypad,	   // (n+1) x 1
	const arma::vec &lag_par,
	const unsigned int &tidx,
	const unsigned int &nelem,
	const unsigned int &trans_code);

void theta_subset(
	arma::vec &Fphi_sub,
	arma::vec &hpsi_sub,
	const arma::vec &hpsi_pad, // (n+1) x 1
	const arma::vec &lag_par,
	const unsigned int &tidx,
	const unsigned int &nelem,
	const unsigned int &trans_code);

double theta_new_nobs(
	const arma::vec &Fphi_sub, // nelem x 1
	const arma::vec &hpsi_sub, // nelem x 1
	const arma::vec &ysub);

double theta_new_nobs(
	const arma::vec &hpsi_pad, // (n+1) x 1
	const arma::vec &ypad,	   // (n+1) x 1
	const unsigned int &tidx,  // t = 1, ..., n
	const arma::vec &lag_par,
	const unsigned int &trans_code,
	const unsigned int &nlag_in,
	const bool &truncated);

arma::mat hpsi2theta(
	const arma::mat &hpsi_pad, // (n+1) x k, each row is a different time point
	const arma::vec &ypad,	   // (n+1) x 1
	const arma::vec &lag_par,
	const unsigned int &trans_code,
	const unsigned int &nlag_in,
	const bool &truncated);

void hpsi2theta(
	arma::vec &theta,
	const arma::vec &hpsi, // (n+1) x 1, each row is a different time point
	const arma::vec &ypad, // (n+1) x 1
	const arma::vec &lag_par,
	const unsigned int &trans_code = 2,
	const unsigned int &nlag_in = 20,
	const bool &truncated = true);

void wt2theta(
	arma::vec &theta,	 // n x 1
	const arma::vec &wt, // n x 1
	const arma::vec &y,	 // n x 1
	const arma::vec &lag_par,
	const unsigned int &gain_code = 3,
	const unsigned int &trans_code = 2,
	const unsigned int &nlag = 20,
	const bool &truncated = true);

double loglike_obs(
	const double &y, 
	const double &lambda,
	const unsigned int &obs_code = 1,
	const double &delta_nb = 30.,
	const bool &return_log = false);

double dloglike_dlambda(
	const double &y,
	const double &lambda,
	const double &delta_nb = 30.,
	const unsigned int &obs_code = 0.);

#endif