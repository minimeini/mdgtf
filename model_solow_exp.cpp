#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <algorithm>
#include <RcppArmadillo.h>
#include <nlopt.h>
#include "nloptrAPI.h"

using namespace Rcpp;
// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::depends(RcppArmadillo,nloptr)]]

/*
Solow's Pascal Disctributed Lags with r=2
and an identity link function
and nonlinear state space

<obs> y[t] ~ Pois(lambda[t])
<link> lambda[t] = theta[t]
<state> theta[t] = 2*rho*theta[t-1] - rho^2*theta[t-2] + (1-rho)^2*x[t-1]*exp(psi[t-1])
<state> psi[t] = psi[t-1] + omega[t]

omega[t] ~ Normal(0,W), where W is modeled by discount factor delta

*/
void forwardFilterSolow2_Identity2(
	const arma::vec& Y, // n x 1, the observation (scalar), n: num of obs
	const arma::vec& X, // n x 1, x[0],x[1],...,x[n-1]
	const double delta, // discount factor for evolution/state error, delta = 1 is a constant value with zero variance
	const double rho,
	arma::mat& Gt, // 3 x 3, state transition matrix
	arma::mat& mt, // 3 x (n+1), t=0 is the mean for initial value theta[0]
	arma::mat& at, // 3 x (n+1)
	arma::cube& Ct, // 3 x 3 x (n+1), t=0 is the var for initial value theta[0]
	arma::cube& Rt, // 3 x 3 x (n+1)
	arma::vec& alphat, // (n+1) x 1
	arma::vec& betat) { // (n+1) x 1

	const unsigned int n = Y.n_elem; // number of observation
    arma::vec F(3,arma::fill::zeros); F.at(0) = 1.;

	double et,ft,Qt,ft_ast,Qt_ast;
	double coef = (1.-rho) * (1.-rho);
	double rho2 = rho * rho;
	arma::vec At(3);
	arma::vec gt(3);

	for (unsigned int t=1; t<=n; t++) {
		R_CheckUserInterrupt();
		Gt.at(0,2) = coef*X.at(t-1)*std::exp(mt.at(2,t-1));

		gt.at(0) = 2*rho*mt.at(0,t-1) - rho2*mt.at(1,t-1) + Gt.at(0,2);
		gt.at(1) = mt.at(0,t-1);
		gt.at(2) = mt.at(2,t-1);
		
		// Prior at time t: theta[t] | D[t-1] ~ N(at, Rt)
        Rt.slice(t) = Gt * Ct.slice(t-1) * Gt.t() / delta;
		at.col(t) = gt;
		
		// One-step ahead forecast: Y[t]|D[t-1] ~ N(ft, Qt)
		ft = arma::as_scalar(F.t() * at.col(t));
		Qt = arma::as_scalar(F.t() * Rt.slice(t) * F);
        
		betat.at(t) = ft/Qt;
		alphat.at(t) = ft*betat.at(t);

        ft_ast = (alphat.at(t)+Y.at(t-1)) / (betat.at(t)+1.);
		Qt_ast = ft_ast / (betat.at(t)+1.);

		// Posterior at time t: theta[t] | D[t] ~ N(mt, Ct)
		et = ft_ast - ft;
		At = Rt.slice(t) * F / Qt;
		mt.col(t) = at.col(t) + At * et;
		Ct.slice(t) = Rt.slice(t) - At*(Qt-Qt_ast)*At.t();
	}

}




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
//' @export
// [[Rcpp::export]]
Rcpp::List lbe_poissonSolowIdentity2(
	const arma::vec& Y, // n x 1, obs data
	const arma::vec& X, // n x 1
	const double rho, // aka G, state transition matrix: assume time-invariant
    const double delta, // discount factor
    const arma::vec& m0, // 3 x 1
    const arma::mat& C0) {  // 3 x 3, systematic variance

	const unsigned int n = Y.n_elem;
	arma::mat Gt(3,3,arma::fill::zeros); 
	Gt.at(0,0) = 2*rho;
	Gt.at(0,1) = -rho*rho;
	Gt.at(1,0) = 1.;
	Gt.at(2,2) = 1.;

	// arma::vec X_(n+1,arma::fill::zeros); 
	// X_.tail(n) = X;

	arma::mat mt(3,n+1,arma::fill::zeros); mt.col(0) = m0;
    arma::cube Ct(3,3,n+1,arma::fill::zeros); Ct.slice(0) = C0;
	arma::mat at(3,n+1,arma::fill::zeros);
    arma::cube Rt(3,3,n+1,arma::fill::zeros);
	arma::vec alphat(n+1,arma::fill::zeros);
	arma::vec betat(n+1,arma::fill::zeros);
	// arma::vec ht(n,arma::fill::zeros);
	// arma::vec Ht(n,arma::fill::zeros);

	forwardFilterSolow2_Identity2(Y,X,delta,rho,Gt,mt,at,Ct,Rt,alphat,betat);
	// backwardSmoother(rho,mt,at,Ct,Rt,ht,Ht);

	Rcpp::List output;
	output["mt"] = Rcpp::wrap(mt);
	output["Ct"] = Rcpp::wrap(Ct);
	output["alphat"] = Rcpp::wrap(alphat);
	output["betat"] = Rcpp::wrap(betat);
	output["delta"] = delta;
	output["rho"] = rho;
	// output["ht"] = Rcpp::wrap(ht);
	// output["Ht"] = Rcpp::wrap(Ht);

	return output;
}

