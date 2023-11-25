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

void init_by_trans(
	unsigned int &p, // dimension of DLM state space
	unsigned int &L_,
	const unsigned int trans_code,
	const unsigned int L = 2);

void init_by_trans(
	unsigned int& p, // dimension of DLM state space
	unsigned int& L_,
	arma::vec& Ft,
    arma::vec& Fphi,
	const unsigned int trans_code,
	const unsigned int L = 2);

void init_Gt(
	arma::cube &Gt,
	const double &rho,
	const unsigned int &p,
	const unsigned int &trans_code);


void init_Gt(
	arma::mat &Gt,
	const double &rho,
	const unsigned int &p,
	const unsigned int &trans_code);


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
arma::vec get_Fphi(
	const unsigned int &n,	 // number of Lags
	const unsigned int &L = 0,	 // dimension of state space (-1 for solow)
	const double &rho = -1., // prob of negative binomial
	const unsigned int &trans_code = 1);

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
	const unsigned int gain_code);

/**
 * Apply gain function h(.) to
 * the latent random walk variable, psi[t]
 */
double psi2hpsi(
	const double psi,
	const unsigned int gain_code);



void hpsi_deriv(
	arma::mat& hpsi,
	const arma::mat& psi,
	const unsigned int gain_code);


double hpsi_deriv(
	const double psi,
	const unsigned int gain_code);


arma::mat hpsi_deriv(
	const arma::mat& psi,
	const unsigned int gain_code);


arma::mat hpsi2theta(
	const arma::mat& hpsi, // (n+1) x k
	const arma::vec& ypad, // n x 1
	const unsigned int &trans_code,
	const double &theta0 = 0.,
	const unsigned int &L = 2,
	const double &rho = 0.9);



void hpsi2theta(
	arma::vec &theta,
	const arma::vec &hpsi, // (n+1) x 1, each row is a different time point
	const arma::vec &ypad, // (n+1) x 1
	const unsigned int &trans_code,
	const double &theta0 = 0.,
	const unsigned int &L = 2,
	const double &rho = 0.9);



double loglike_obs(
	const double &y, 
	const double &lambda,
	const unsigned int &obs_code = 1,
	const double &delta_nb = 1,
	const bool &return_log = false);



#endif