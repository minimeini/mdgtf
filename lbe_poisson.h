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



double trigamma_obj(
	unsigned n,
	const double *x, 
	double *grad, 
	void *my_func_data);


double optimize_trigamma(double q);


void forwardFilter(
	const unsigned int model,
	const arma::vec& Y, // n x 1, the observation (scalar), n: num of obs
	const double G, // state transition matrix
	const double W, // p x p, state error
	arma::vec& mt, // (n+1) x 1, t=0 is the mean for initial value theta[0]
	arma::vec& at, // (n+1) x 1
	arma::vec& Ct, // (n+1) x 1, t=0 is the var for initial value theta[0]
	arma::vec& Rt, // (n+1) x 1
	arma::vec& alphat, // (n+1) x 1
	arma::vec& betat);



void forwardFilterX(
	const unsigned int model,
	const arma::vec& Y, // n x 1, the observation (scalar), n: num of obs
	const arma::vec& X, // n x 1
	const double W, // evolution/state error
	arma::mat& Gt, // 2 x 2, state transition matrix
	arma::mat& Wt, // 2 x 2, state error
	arma::mat& mt, // 2 x (n+1), t=0 is the mean for initial value theta[0]
	arma::mat& at, // 2 x (n+1)
	arma::cube& Ct, // 2 x 2 x (n+1), t=0 is the var for initial value theta[0]
	arma::cube& Rt, // 2 x 2 x (n+1)
	arma::vec& alphat, // (n+1) x 1
	arma::vec& betat);



void forwardFilterX2(
	const unsigned int model,
	const arma::vec& Y, // n x 1, the observation (scalar), n: num of obs
	const arma::vec& X, // n x 1
	const double delta, // discount factor for evolution/state error
	arma::mat& Gt, // 2 x 2, state transition matrix
	arma::mat& mt, // 2 x (n+1), t=0 is the mean for initial value theta[0]
	arma::mat& at, // 2 x (n+1)
	arma::cube& Ct, // 2 x 2 x (n+1), t=0 is the var for initial value theta[0]
	arma::cube& Rt, // 2 x 2 x (n+1)
	arma::vec& alphat, // (n+1) x 1
	arma::vec& betat);


/*
---- Method ----
- Koyama's Discretized Hawkes Process
- Ordinary Kalman filtering with all static parameters known
- Using the DLM formulation

---- Model ----
<obs>   y[t] ~ Pois(lambda[t]), t=1,...,n
<link>  lambda[t] = phi[1]*y[t-1]*exp(psi[t]) + ... + phi[L]*y[t-L]*exp(psi[t-L+1]),t=1,...,n
<state> psi[t] = psi[t-1] + omega[t], t=1,...,n
        omega[t] ~ Normal(0,W)

<prior> theta_till[0] ~ Norm(m0, C0)
        W ~ IG(aw,bw)
*/

/*
Linear Bayes approximation based filtering for koyama's model with exponential on the transmission delay.
- Observational linear approximation - This function uses 1st order Taylor expansion for the nonlinear Ft at the observational equation.
- Known evolution variance - This function assumes the evolution variance, W, is fixed and known
*/
void forwardFilterKoyamaExp(
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
void forwardFilterKoyamaExp2(
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
void backwardSmootherKoyamaExp(
	const arma::mat& G, // L x L, state transition matrix
	const arma::mat& mt, // L x (n+1), t=0 is the mean for initial value theta[0]
	const arma::mat& at, // L x (n+1)
	const arma::cube& Ct, // L x L x (n+1), t=0 is the var for initial value theta[0]
	const arma::cube& Rt, // L x L x (n+1)
	arma::mat& ht, // L x n
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
void backwardSmootherKoyamaExp2(
	const arma::mat& G, // L x L, state transition matrix
	const arma::mat& mt, // L x (n+1), t=0 is the mean for initial value theta[0]
	const arma::mat& at, // L x (n+1)
	const arma::cube& Ct, // L x L x (n+1), t=0 is the var for initial value theta[0]
	const arma::cube& Rt, // L x L x (n+1)
	const double delta, // discount factor
	arma::vec& ht, // n
    arma::vec& Ht);


/*
Solow's Pascal Disctributed Lags with r=2
*/
void forwardFilterSolow2_0(
	const arma::vec& Y, // n x 1, the observation (scalar), n: num of obs
	const arma::vec& X, // (n+1) x 1, x[0],x[1],...,x[n]
	const double W, // evolution error of the beta's
	const double rho,
	arma::mat& Gt, // 3 x 3, state transition matrix
	arma::mat& Wt, // 3 x 3, evolution error matrix
	arma::mat& mt, // 3 x (n+1), t=0 is the mean for initial value theta[0]
	arma::mat& at, // 3 x (n+1)
	arma::cube& Ct, // 3 x 3 x (n+1), t=0 is the var for initial value theta[0]
	arma::cube& Rt, // 3 x 3 x (n+1)
	arma::vec& alphat, // (n+1) x 1
	arma::vec& betat);


/*
Solow's Pascal Disctributed Lags with r=2
*/
void forwardFilterSolow2(
	const arma::vec& Y, // n x 1, the observation (scalar), n: num of obs
	const arma::vec& X, // (n+1) x 1, x[0],x[1],...,x[n]
	const double delta, // discount factor for evolution/state error, delta = 1 is a constant value with zero variance
	const double rho,
	arma::mat& Gt, // 3 x 3, state transition matrix
	arma::mat& mt, // 3 x (n+1), t=0 is the mean for initial value theta[0]
	arma::mat& at, // 3 x (n+1)
	arma::cube& Ct, // 3 x 3 x (n+1), t=0 is the var for initial value theta[0]
	arma::cube& Rt, // 3 x 3 x (n+1)
	arma::vec& alphat, // (n+1) x 1
	arma::vec& betat);

/*
Solow's Pascal Disctributed Lags with r=2
and an identity link function
*/
void forwardFilterSolow2_Identity(
	const arma::vec& Y, // n x 1, the observation (scalar), n: num of obs
	const arma::vec& X, // n x 1, x[0],x[1],...,x[n]
	const double delta, // discount factor for evolution/state error, delta = 1 is a constant value with zero variance
	const double rho,
	arma::mat& Gt, // 3 x 3, state transition matrix
	arma::mat& mt, // 3 x (n+1), t=0 is the mean for initial value theta[0]
	arma::mat& at, // 3 x (n+1)
	arma::cube& Ct, // 3 x 3 x (n+1), t=0 is the var for initial value theta[0]
	arma::cube& Rt, // 3 x 3 x (n+1)
	arma::vec& alphat, // (n+1) x 1
	arma::vec& betat);


/*
Solow's Pascal Disctributed Lags with r=2
and an identity link function
and nonlinear state space

y[t] ~ Pois(lambda[t])
lambda[t] = theta[t]
theta[t] = 2*rho*theta[t-1] - rho^2*theta[t-2] + (1-rho)^2*x[t-1]*exp(beta[t-1])
beta[t] = beta[t-1] + omega[t]

omega[t] ~ Normal(0,W), where W is modeled by discount factor delta

*/
void forwardFilterSolow2_Identity2(
	const arma::vec& Y, // n x 1, the observation (scalar), n: num of obs
	const arma::vec& X, // n x 1, x[0],x[1],...,x[n]
	const double delta, // discount factor for evolution/state error, delta = 1 is a constant value with zero variance
	const double rho,
	arma::mat& Gt, // 3 x 3, state transition matrix
	arma::mat& mt, // 3 x (n+1), t=0 is the mean for initial value theta[0]
	arma::mat& at, // 3 x (n+1)
	arma::cube& Ct, // 3 x 3 x (n+1), t=0 is the var for initial value theta[0]
	arma::cube& Rt, // 3 x 3 x (n+1)
	arma::vec& alphat, // (n+1) x 1
	arma::vec& betat);


/*
Linear Bayes Estimator - Version 0
Assume  rho, V, W are known;
        E[0] is unknown with prior N(m[0],C[0])
STATUS - Checked and Correct
*/
Rcpp::List lbe_poisson0(
	const arma::vec& Y, // n x 1, obs data
	const double rho, // aka G, state transition matrix: assume time-invariant
    const double W,
    const double m0,
    const double C0);


/*
Linear Bayes Estimator - Version 1
Assume  rho, V, W, E[0] are known;
STATUS - Checked and Correct
*/
Rcpp::List lbe_poisson1(
	const arma::vec& Y, // n x 1, obs data
	const double rho, // aka G, state transition matrix: assume time-invariant
    const double W,
    const double E0,
    const double Rw);


/*
Linear Bayes Estimator with Transfer Function - Version 0
Assume  rho, W are known;
        E[0] and beta[0] is unknown with prior N(m[0],C[0])
STATUS - Checked and Correct
*/
Rcpp::List lbe_poissonX0(
	const arma::vec& Y, // n x 1, obs data
	const arma::vec& X,
	const double rho, // aka G, state transition matrix: assume time-invariant
    const double W,
    const arma::vec& m0, // 2 x 1
    const arma::mat& C0);


/*
Linear Bayes Estimator with Transfer Function - Version 0
Assume  rho is known;
		use discount factor `delta` for Wt
        E[0] and beta[0] is unknown with prior N(m[0],C[0])
STATUS - Checked and Correct
*/
Rcpp::List lbe_poissonX(
	const arma::vec& Y, // n x 1, obs data
	const arma::vec& X, // n x 1
	const double rho, // aka G, state transition matrix: assume time-invariant
    const double delta, // discount factor
    const arma::vec& m0, // 2 x 1
    const arma::mat& C0);


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
        omega[t] ~ Normal(0,W)

<prior> theta_till[0] ~ Norm(m0, C0)
        W ~ IG(aw,bw)
*/
Rcpp::List lbe_poissonKoyamaExp(
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
        omega[t] ~ Normal(0,W)

<prior> theta_till[0] ~ Norm(m0, C0)
        W ~ IG(aw,bw)
*/
Rcpp::List lbe_poissonKoyamaExp2(
	const arma::vec& Y, // n x 1, the observed response
    const unsigned int L,
    const double delta, 
    const Rcpp::Nullable<Rcpp::NumericVector>& theta0_init);


/*
Linear Bayes Estimator with Solow's Transfer Function - Version 0
Assume  rho is known;
		use discount factor `delta` for Wt
        E[0] and beta[0] is unknown with prior N(m[0],C[0])
STATUS - Checked and Correct
*/
Rcpp::List lbe_poissonSolow0(
	const arma::vec& Y, // n x 1, obs data
	const arma::vec& X, // n x 1
	const double rho, // aka G, state transition matrix: assume time-invariant
    const double W, // evolution error variance
    const arma::vec& m0, // 3 x 1
    const arma::mat& C0);



/*
Linear Bayes Estimator with Solow's Transfer Function - Version 0
Assume  rho is known;
		use discount factor `delta` for Wt
        E[0] and beta[0] is unknown with prior N(m[0],C[0])
STATUS - Checked and Correct
*/
Rcpp::List lbe_poissonSolow(
	const arma::vec& Y, // n x 1, obs data
	const arma::vec& X, // n x 1
	const double rho, // aka G, state transition matrix: assume time-invariant
    const double delta, // discount factor
    const arma::vec& m0, // 3 x 1
    const arma::mat& C0);

/*
Linear Bayes Estimator with Solow's Transfer Function
and an identity link

Assume  rho is known;
		use discount factor `delta` for Wt
        E[0] and beta[0] is unknown with prior N(m[0],C[0])
STATUS - Checked and Correct
*/
Rcpp::List lbe_poissonSolowIdentity(
	const arma::vec& Y, // n x 1, obs data
	const arma::vec& X, // n x 1
	const double rho, // aka G, state transition matrix: assume time-invariant
    const double delta, // discount factor
    const arma::vec& m0, // 3 x 1
    const arma::mat& C0);


/*
Linear Bayes Estimator with Solow's Transfer Function
and an identity link
and nonlinear state space

y[t] ~ Pois(lambda[t])
lambda[t] = theta[t]
theta[t] = 2*rho*theta[t-1] - rho^2*theta[t-2] + (1-rho)^2*x[t-1]*exp(beta[t-1])
beta[t] = beta[t-1] + omega[t]

omega[t] ~ Normal(0,W), where W is modeled by discount factor delta

Assume  rho is known;
		use discount factor `delta` for Wt
        theta[0] and beta[0] is unknown with prior N(m[0],C[0])
STATUS - Checked and Correct
*/
Rcpp::List lbe_poissonSolowIdentity2(
	const arma::vec& Y, // n x 1, obs data
	const arma::vec& X, // n x 1
	const double rho, // aka G, state transition matrix: assume time-invariant
    const double delta, // discount factor
    const arma::vec& m0, // 3 x 1
    const arma::mat& C0);



#endif