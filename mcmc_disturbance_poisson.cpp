#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <algorithm>
#include <RcppArmadillo.h>
#include "lbe_poisson.h"
#include "model_utils.h"

using namespace Rcpp;
// [[Rcpp::plugins(cpp17)]]
// [[Rcpp::depends(RcppArmadillo)]]


/**
 * MCMC
 * Important functions:
 * 1. get_increment_matrix
 * 2. update_Fn
 * 3. update_theta0
*/


void get_increment_matrix(
	arma::vec &t_row, // n x 1
	const unsigned int &tidx,
	const arma::vec &ypad,	  // (n+1)
	const arma::vec &hpad,	  // (n+1)
	const arma::vec &Fphi_pad // (n+1)
)
{
	arma::vec phi_sub = Fphi_pad.subvec(1, tidx); // t x 1, phi[1], ..., phi[t]
	arma::vec ysub = ypad.subvec(0, tidx - 1);
	arma::vec hsub = hpad.subvec(1, tidx);
	arma::vec yh_tmp = ysub % hsub;
	arma::vec yh_sub = arma::reverse(yh_tmp); // y[t-1]hph[t], ..., y[0]hph[1]
	t_row = phi_sub % yh_sub;
}

void get_Fphi_pad(
	arma::vec &Fphi_pad,
	const unsigned int &trans_code,
	const unsigned int &nobs,
	const unsigned int &nlag,
	const unsigned int &L_order,
	const unsigned int &rho)
{
	unsigned int nlag_ = nlag;
	if (nlag == 0)
	{
		nlag_ = nobs;
	}
	arma::vec Fphi = get_Fphi(nlag_, L_order, rho, trans_code); // nlag x 1

	Fphi_pad.set_size(nobs + 1);
	Fphi_pad.zeros();

	for (unsigned int l = 1; l <= nlag_; l++)
	{
		Fphi_pad.at(l) = Fphi.at(l - 1);
	}

	return;
}

arma::mat update_Fn(
	const arma::vec &y,		   // nobs x 1 or (nobs+1) x 1, (y[1],...,y[n]), observtions
	const arma::vec &Fphi_pad, // (nobs+1) x 1
	const arma::vec &hderiv,   // nobs x 1
	const unsigned int &nobs)
{								  
	arma::mat Fn(nobs,nobs,arma::fill::zeros);

	arma::vec ypad(nobs + 1, arma::fill::zeros);
	ypad.tail(y.n_elem) = y; // Ypad = (Y[0]=0,Y[1],...,Y[nobs])

	arma::vec hph_pad(nobs + 1, arma::fill::zeros);
	hph_pad.tail(hderiv.n_elem) = hderiv;

	for (unsigned int t = 1; t <= nobs; t++)
	{
		arma::vec t_row(nobs, arma::fill::zeros);
		get_increment_matrix(
			t_row, t, ypad, hph_pad, Fphi_pad);

		arma::vec t_cumsum = arma::cumsum(t_row);
		if (!(std::abs(t_row.at(0) - t_cumsum.at(0)) < EPS8))
		{
			throw std::invalid_argument("update_Fn: wrong cumsum index alignment.");
		}

		arma::vec Fn_row_t = arma::reverse(t_cumsum); // t x 1
		for (unsigned int k = 1; k <= t; k++)
		{
			Fn.at(t - 1, k - 1) = Fn_row_t.at(k - 1);
		}
	}

	return Fn;
}

/*
TODO: check it.
*/
arma::vec update_theta0(
	const arma::vec &y,		  // n x 1, observations, (Y[1],...,Y[n])
	const arma::vec &Fphi_pad, // (n+1) x 1
	const arma::vec &psi_hat, // n x 1, where the taylor expansion is performed
	const arma::vec &hderiv,
	const unsigned int &nobs,
	const unsigned int &gain_code)
{
	arma::vec theta0(nobs,arma::fill::zeros);

	arma::vec ypad(nobs + 1, arma::fill::zeros);
	ypad.tail(y.n_elem) = y; // Ypad = (Y[0]=0,Y[1],...,Y[n])

	arma::vec hpsi = psi2hpsi(psi_hat, gain_code);	   // n x 1
	arma::vec htmp = hderiv % psi_hat;				   // n x 1
	arma::vec h0 = hpsi - htmp;
	arma::vec h0_pad(nobs + 1, arma::fill::zeros);
	h0_pad.tail(h0.n_elem) = h0;

	for (unsigned int t = 1; t <= nobs; t++)
	{
		arma::vec t_row(nobs, arma::fill::zeros);
		get_increment_matrix(
			t_row, t, ypad, h0_pad, Fphi_pad);
		theta0.at(t - 1) = arma::accu(t_row);
	}

	if (theta0.at(0) < EPS8)
	{
		theta0.at(0) = EPS8;
	}

	bound_check(theta0, "update_theta0", true, false);

	return theta0;
}



double proposal_mh_var(
	const arma::vec &y,			 // n x 1
	const arma::vec &wt,		 // n x 1
	const arma::vec &theta0_hat, // n x 1
	const arma::mat &Fn_hat,	 // n x n
	const unsigned int &sidx,
	const unsigned int &link_code,
	const unsigned int &obs_code,
	const double &delta_nb = 1.,
	const double &mu0 = 0.)
{
	if (link_code != 0)
	{
		throw std::invalid_argument("proposal_mh_var: only support identity link.");
	}
	unsigned int n = y.n_elem;

	arma::vec theta1 = Fn_hat * wt;
	arma::vec lambda = theta0_hat + theta1;
	lambda.for_each([&mu0](arma::vec::elem_type &val)
					{ val += mu0; });

	arma::vec Vt(n, arma::fill::zeros);
	if (obs_code == 0)
	{
		// nbinom obs + identity link
		// lambda[t] + (lambda[t])^2 / delta_nb
		arma::vec tmp1 = lambda;
		tmp1.for_each([&delta_nb](arma::vec::elem_type &val)
					  { val += delta_nb; });

		arma::vec tmp2 = lambda % tmp1;
		tmp2.for_each([&delta_nb](arma::vec::elem_type &val)
					  { val /= delta_nb; });

		Vt = tmp2;
	}
	else if (obs_code == 1)
	{
		// Poisson obs + identity link
		Vt = lambda;
	}
	else
	{
		throw std::invalid_argument("proposal_mh_var: only support Poisson or NB likelihood.");
	}
	Vt.elem(arma::find(Vt < EPS8)).fill(EPS8);

	arma::vec Fn_col_s = Fn_hat.col(sidx);
	arma::vec Fn_s2 = Fn_col_s % Fn_col_s;
	arma::vec tmp = Fn_s2 / Vt; // tmp too big: Vt too small or Fn_s2 too big
	double mh_prec = arma::accu(tmp); // precision of MH proposal
	bound_check(mh_prec,"proposal_mh_var: mh_prec",true,true);
	double mh_var = 1./mh_prec;

	return mh_var;
}

/**
 * logpost_wt: Checked. OK.
*/
double logpost_wt(
	const arma::vec &y,	 // n x 1
	const arma::vec &wt, // n x 1
	const unsigned int &sidx,
	const unsigned int &gain_code,
	const unsigned int &trans_code,
	const unsigned int &link_code,
	const unsigned int &obs_code,
	const double &mu0,
	const unsigned int &delta_nb,
	const double &aw, // prior mean of w[s]
	const double &Rw,
	const double &rho,
	const unsigned int &L,
	const unsigned int &nlag = 0) // prior SD of w[s]
{
	const unsigned int n = y.n_elem;
	double Rw_sqrt = std::sqrt(Rw);

	arma::vec theta(n, arma::fill::zeros);
	wt2theta(
		theta, wt, y,
		gain_code, trans_code,
		rho, L, nlag);

	arma::vec lambda_core = mu0 + theta;
	arma::vec lambda(n, arma::fill::zeros);
	if (link_code == 1)
	{
		lambda_core.elem(arma::find(lambda > UPBND)).fill(UPBND);
		lambda = arma::exp(lambda_core);
	}
	else
	{
		lambda = lambda_core;
	}

	double logpost = R::dnorm4(wt.at(sidx), aw, Rw_sqrt, true);
	for (unsigned int t = sidx; t < n; t++)
	{
		double logp_increment = loglike_obs(
			y.at(t), lambda.at(t),
			obs_code, delta_nb, true);

		logpost += logp_increment;
	}

	bound_check(logpost, "logpost_wt");

	return logpost;
}



void update_wt(
	arma::vec &wt,
	arma::vec &wt_accept,
	arma::vec &Bs,
	arma::vec &logr,
	const arma::vec &y, // nobs x 1
	const arma::vec &Fphi_pad, // (nobs + 1)
	const arma::vec &lags_params, // (L_order, rho) or (mu, sg2)
	const arma::vec &obs_params, // delta_nb, mu0
	const arma::vec &prior_params, // w[t] ~ N(0, W)
	const double &mh_sd,
	const unsigned int &nobs,
	const unsigned int &nlag,
	const unsigned int &gain_code,
	const unsigned int &trans_code,
	const unsigned int &link_code,
	const unsigned int &obs_code)
{
	double rho = lags_params.at(0);
	unsigned int L_order = lags_params.at(1);
	double delta_nb = obs_params.at(0);
	double mu0 = obs_params.at(1);
	double mu_wt = prior_params.at(0);
	double sg2_wt = prior_params.at(1);

	for (unsigned int s = 0; s < nobs; s++)
	{
		arma::vec psi = arma::cumsum(wt);
		arma::vec hderiv = hpsi_deriv(psi, gain_code); // n x 1
		arma::vec theta0_hat = update_theta0(y, Fphi_pad, psi, hderiv, nobs, gain_code);
		arma::mat Fn_hat = update_Fn(y, Fphi_pad, hderiv, nobs);

		// logp_old
		double wt_old = wt.at(s);
		double logp_old = logpost_wt(
			y, wt, s,
			gain_code, trans_code, link_code, obs_code,
			mu0, delta_nb, mu_wt, sg2_wt, rho, L_order, nlag);

		Bs.at(s) = proposal_mh_var(
			y, wt, theta0_hat, Fn_hat, s,
			link_code, obs_code,
			delta_nb, mu0);

		double Btmp = std::sqrt(Bs.at(s));
		Btmp *= mh_sd;

		double wt_new = R::rnorm(wt_old, Btmp);

		wt.at(s) = wt_new;
		double logp_new = logpost_wt(
			y, wt, s,
			gain_code, trans_code, link_code, obs_code,
			mu0, delta_nb, mu_wt, sg2_wt, rho, L_order, nlag);

		double logratio = logp_new - logp_old;
		// logratio += logq_old - logq_new;
		logratio = std::min(0., logratio);

		if (std::log(R::runif(0., 1.)) < logratio)
		{
			// accept
			wt_accept.at(s) += 1.;
		}
		else
		{
			// reject
			wt.at(s) = wt_old;
		}

		logr.at(s) = logratio;
	}
}


double update_W(
	double &W_accept,
	const double &W_old,
	const arma::vec &wt,
	const unsigned int &W_prior_type, // eta_prior_type.at(0)
	const arma::vec &prior_params,	  // (aw_new, bw_prior), aw_new = eta_prior_val.at(0, 0) - 0.5 * ((double)nobs - 1.)
	const double &mh_sd,
	const unsigned int &nobs)
{
	double W = W_old;

	double res = arma::accu(arma::pow(wt.tail(nobs - 1), 2.));
	double aw_new = prior_params.at(0);
	// double bw_prior = prior_params.at(1); // eta_prior_val.at(1, 0)

	switch (W_prior_type) // eta_prior_type.at(0)
	{
	case 0: // Gamma(aw=shape, bw=rate)
	{
		double logp_old = prior_params.at(0) * std::log(W_old) - prior_params.at(1) * W_old - 0.5 * res / W_old;
		double W_new = std::exp(std::min(R::rnorm(std::log(W_old), mh_sd), UPBND));
		double logp_new = prior_params.at(0) * std::log(W_new) - prior_params.at(1) * W_new - 0.5 * res / W_new;
		double logratio = std::min(0., logp_new - logp_old);
		if (std::log(R::runif(0., 1.)) < logratio)
		{ // accept
			W = W_new;
			W_accept += 1.;
		}
	}
	break;
	case 1: // Half-Cauchy(aw=location==0, bw=scale)
	{
		throw std::invalid_argument("Half-cauchy prior for W is not implemented yet.");
	}
	break;
	case 2: // Inverse-Gamma(nw=shape, nSw=rate)
	{
		double nSw_new = prior_params.at(1) + res; // prior_params.at(1) = nSw
		W = 1. / R::rgamma(0.5 * prior_params.at(0), 2. / nSw_new); // prior_params.at(0) = nw_new
		W_accept += 1.;
	}
	break;
	default:
	{
		throw std::invalid_argument("Unimplemented prior for W.");
	}
	}

	return W;
}




/*
MCMC disturbance sampler for different transfer kernels, link functions, and reproduction number functions.

Unknown Parameters
	- local parameters: psi[1:n]
	- global parameters `eta`: W, mu0, rho, M,th0, psi0
*/
//' @export
// [[Rcpp::export]]
Rcpp::List mcmc_disturbance_pois(
	const arma::vec &y, // n x 1, the observed response
	const arma::uvec &model_code,
	const Rcpp::IntegerVector &eta_select = Rcpp::IntegerVector::create(1, 0, 0),		 // W, mu0, rho
	const Rcpp::NumericVector &W_par = Rcpp::NumericVector::create(0.01, 2, 0.01, 0.01), // (Winit/Wtrue,WpriorType,par1,par2)
	const Rcpp::NumericVector &mu_par = Rcpp::NumericVector::create(0., 0, 0., 0.),		 // (muInit/muTrue,muPriorType,par1,par2)
	const Rcpp::NumericVector &rho_par = Rcpp::NumericVector::create(1., 0., 1., 1.),	 // similar
	const double &L_order = 6,															 // Lag Koyama's model; order for Solow's model
	const unsigned int &nlag_ = 0,														 // 0: no truncation; nlag > 0: truncated to nlag
	const Rcpp::NumericVector &mh_var = Rcpp::NumericVector::create(1., 0.8),			 // mh and discount factor for wt
	const bool &use_lambda_bound = false,
	const double &delta_nb = 30.,
	const unsigned int &nburnin = 0,
	const unsigned int &nthin = 1,
	const unsigned int &nsample = 1,
	const bool &summarize_return = true)
{ // n x 1

	unsigned int obs_code, link_code, trans_code, gain_code, err_code;
	get_model_code(obs_code, link_code, trans_code, gain_code, err_code, model_code);

	const unsigned int n = y.n_elem;
	const unsigned int ntotal = nburnin + nthin * nsample + 1;

	unsigned int nlag, p, L;
	set_dim(nlag, p, L, n, nlag_, L_order);

	arma::vec mh_sd(mh_var.begin(), 2);
	mh_sd.for_each([](arma::vec::elem_type &val)
				   { val = std::sqrt(val); });

	// Global Parameter
	// eta = (W, mu0, rho)
	double W = W_par[0];
	double mu0 = mu_par[0];
	double rho = rho_par[0];


	arma::vec W_stored(nsample);
	bool W_selected = eta_select[1] == 1;
	double W_accept = 0.;
	double Wrt = std::sqrt(W);
	arma::vec W_params = {0.,0.};
	switch ((unsigned int) W_par[1])
	{
	case 0: // Gamma(aw=shape, bw=rate)
	{
		double aw_new = W_par[2]- 0.5 * ((double)n - 1.);
		double bw_prior = W_par[3];
		W_params.at(0) = aw_new;
		W_params.at(1) = bw_prior;
	}
	break;
	case 1: // Half-Cauchy(aw=location==0, bw=scale)
	{
		throw std::invalid_argument("Half-cauchy prior for W is not implemented yet.");
	}
	break;
	case 2: // Inverse-Gamma(nw=shape, nSw=rate)
	{
		double nw = W_par[2];
		double nSw = W_par[2] * W_par[3];
		double nw_new = nw + (double)n - 1.;
		W_params.at(0) = nw_new;
		W_params.at(1) = nSw;
	}
	break;
	default:
	break;
	}


	arma::vec lags_params = {(double) L, rho};
	arma::vec obs_params = {delta_nb, mu0};
	arma::vec prior_params = {0., W};

	// double th0 = eta_init.at(4);
	// arma::vec th0_stored(nsample);
	// double m_th0 = eta_prior_val.at(0,4);
	// double C_th0 = eta_prior_val.at(1,4);
	// double C_th0rt = std::sqrt(C_th0);
	// double m_th, C_th, C_thrt;
	// double th0_new;
	// double th0_accept = 0.;

	// double psi0 = eta_init.at(5);

	// Local Parameter
	arma::vec wt = arma::randn(n) * 0.01;
	for (unsigned int t = 0; t < L; t++)
	{
		wt.at(t) = std::abs(wt.at(t));
	}
	arma::vec wt_accept(n, arma::fill::zeros);
	arma::mat wt_stored(n, nsample, arma::fill::zeros);




	// double rho, xi, xi_old, xi_new, rho_sq;
	// arma::vec rho_stored(nsample);
	// double rho_accept = 0.;
	// bool rhoflag;
	// if (R_IsNA(rho_true)) {
	// 	rhoflag = true;
	// 	if (R_IsNA(Vxi)) {
	// 		stop("Error: You must provide either true value or prior for rho.");
	// 	}
	// 	if (R_IsNA(rho_init)) {
	// 		rho = 0.5;
	// 	} else {
	// 		rho = rho_init;
	// 	}
	// } else {
	// 	rhoflag = false;
	// 	rho = rho_true;
	// }
	// rho_sq = rho * rho;

	bool saveiter;
	arma::mat logratio_stored(n, nsample, arma::fill::zeros);
	arma::mat Bs_stored(n, nsample, arma::fill::zeros);


	arma::vec Fphi_pad; // nobs x 1
	get_Fphi_pad(Fphi_pad, trans_code, n, nlag, L, rho);


	for (unsigned int b = 0; b < ntotal; b++)
	{
		R_CheckUserInterrupt();
		saveiter = b > nburnin && ((b - nburnin - 1) % nthin == 0);

		// [OK] Update evolution/state disturbances/errors, denoted by wt.
		arma::vec Bs(n, arma::fill::zeros);
		arma::vec logr(n, arma::fill::zeros);
		update_wt(
			wt, wt_accept, Bs, logr, 
			y, Fphi_pad, 
			lags_params, // (r, rho)
			obs_params,   // delta_nb, mu0
			prior_params, // w[t] ~ N(0, W)
			mh_sd.at(0), n, nlag, 
			gain_code, trans_code, link_code, obs_code);


		// [OK] Update state/evolution error variance W
		if (W_selected)
		{
			double W_old = W;
			W = update_W(W_accept, W_old, wt, 
			(unsigned int) W_par[1], W_params, mh_sd.at(1), n);

			Wrt = std::sqrt(W);
			prior_params.at(1) = W;
		}

		// // Update initial state
		// if (E0flag) {
		// 	// Target distribution P(E0_old|)
		// 	Et = E0tilde + Fx * vt;
		// 	lambda = arma::exp(Et);
		// 	logp_old = R::dnorm(E0,mE0,CE0_sqrt,true);
		// 	for (unsigned int t=0;t<n;t++) {
		// 		logp_old += R::dpois(Y.at(t),lambda.at(t),true);
		// 	}

		// 	// E0_new | E0_old
		// 	Ytilde = Et + (Y - lambda) / lambda; // Linearlisation
		//     Yhat = Ytilde - Et + E0tilde;
		// 	CE = 1./CE0;
		// 	mE = mE0/CE0;
		// 	rho_misc = 1.;
		// 	rho2_misc = 1.;
		// 	for (unsigned int t=0; t<n; t++) {
		// 		rho_misc *= rho;
		// 		rho2_misc *= rho_sq;
		// 		CE += lambda.at(t) * rho2_misc;
		// 		mE += lambda.at(t) * Yhat.at(t) * rho_misc;
		// 		/*
		// 		t = 0, rho_misc = rho^2, CE += lambda[0]*(rho^2)
		// 		t = 1, rho_misc = rho^4, CE += lambda[1]*(rho^4)
		// 		*/
		// 	}
		// 	CE = 1./CE;
		// 	mE *= CE;
		// 	CE_sqrt = std::sqrt(CE);
		// 	E0_new = R::rnorm(mE,CE_sqrt);
		// 	logq_new = R::dnorm(E0_new,mE,CE_sqrt,true);

		// 	// Target distribution P(E0_new|)
		// 	E0tilde.at(0) = rho*E0_new;
		// 	for (unsigned int t=1; t<n; t++) { // TODO - CHECK HERE
		// 		E0tilde.at(t) = rho*E0tilde.at(t-1);
		// 	}
		// 	lambda = arma::exp(Et);
		// 	logp_new = R::dnorm(E0_new,mE0,CE0_sqrt,true);
		// 	for (unsigned int t=0;t<n;t++) {
		// 		logp_new += R::dpois(Y.at(t),lambda.at(t),true);
		// 	}

		// 	// E0_old | E0_new
		// 	Ytilde = Et + (Y - lambda) / lambda; // Linearlisation
		//     Yhat = Ytilde - Et + E0tilde;
		// 	CE = 1./CE0;
		// 	mE = mE0/CE0;
		// 	rho_misc = 1.;
		// 	rho2_misc = 1.;
		// 	for (unsigned int t=0; t<n; t++) {
		// 		rho_misc *= rho;
		// 		rho2_misc *= rho_sq;
		// 		CE += lambda.at(t) * rho2_misc;
		// 		mE += lambda.at(t) * Yhat.at(t) * rho_misc;
		// 		/*
		// 		t = 0, rho_misc = rho^2, CE += lambda[0]*(rho^2)
		// 		t = 1, rho_misc = rho^4, CE += lambda[1]*(rho^4)
		// 		*/
		// 	}
		// 	CE = 1./CE;
		// 	mE *= CE;
		// 	CE_sqrt = std::sqrt(CE_sqrt);
		// 	logq_old = R::dnorm(E0,mE,CE_sqrt,true);

		// 	logratio = std::min(0.,logp_new-logp_old+logq_old-logq_new);
		// 	if (std::log(R::runif(0.,1.)) < logratio) { // accept
		// 		E0 = E0_new;
		// 		E0_accept += 1.;
		// 	} else { // reject
		// 		E0tilde.at(0) = rho*E0;
		// 		for (unsigned int t=1; t<n; t++) { // TODO - CHECK HERE
		// 			E0tilde.at(t) = rho*E0tilde.at(t-1);
		// 		}
		// 	}

		// }

		// // Update rho: the state/evolution coefficient
		// if (rhoflag) {
		// 	// if (std::abs(rho)<arma::datum::eps) {
		// 	// 	rho += arma::datum::eps;
		// 	// }
		// 	xi_old = std::log(rho/(1.-rho));
		// 	Et = E0tilde + Fx * vt;
		// 	// Et.elem(arma::find(Et>EBOUND)).fill(EBOUND);
		// 	lambda = arma::exp(Et);
		// 	logp_old = xi_old - 2.*std::log(std::exp(xi_old)+1.);
		// 	for (unsigned int t=0; t<n; t++) {
		// 		logp_old += R::dpois(Y.at(t),lambda.at(t),true);
		// 	}

		// 	xi_new = R::rnorm(xi_old,std::sqrt(Vxi));
		// 	rho = 1./(1.+std::exp(-xi_new));
		// 	Fx = update_Fx1(n,rho,X);
		// 	E0tilde.at(0) = rho*E0;
		// 	for (unsigned int t=1; t<n; t++) { // TODO - CHECK HERE
		// 		E0tilde.at(t) = rho*E0tilde.at(t-1);
		// 	}
		// 	Et = E0tilde + Fx * vt;
		// 	// Et.elem(arma::find(Et>EBOUND)).fill(EBOUND);
		// 	lambda = arma::exp(Et);
		// 	logp_new = xi_new - 2.*std::log(std::exp(xi_new)+1.);
		// 	for (unsigned int t=0; t<n; t++) {
		// 		logp_new += R::dpois(Y.at(t),lambda.at(t),true);
		// 	}

		// 	logratio = std::min(0.,logp_new-logp_old);
		// 	if (std::log(R::runif(0.,1.)) >= logratio) { // reject
		// 		rho = 1./(1.+std::exp(-xi_old));
		// 	} else {
		// 		rho_accept += 1.;
		// 	}
		// 	rho_sq = rho * rho;

		// 	if (!std::isfinite(rho)) {
		// 		Rcout << "rho_new=" << rho << std::endl;
		// 		stop("Non finite value for rho");
		// 	}
		// }

		// Fx = update_Fx1(n,rho,X);
		// E0tilde.at(0) = rho*E0;
		// for (unsigned int t=1; t<n; t++) { // TODO - CHECK HERE
		// 	E0tilde.at(t) = rho*E0tilde.at(t-1);
		// }
		// if (!Fx.is_finite() || !E0tilde.is_finite()) {
		// 	stop("Non finite value for Fx or E0tilde");
		// }

		// store samples after burnin and thinning
		if (saveiter || b == (ntotal - 1))
		{
			unsigned int idx_run;
			if (saveiter)
			{
				idx_run = (b - nburnin - 1) / nthin;
			}
			else
			{
				idx_run = nsample - 1;
			}

			wt_stored.col(idx_run) = wt;
			W_stored.at(idx_run) = W;

			logratio_stored.col(idx_run) = logr;
			Bs_stored.col(idx_run) = Bs;

			// rho_stored.at(idx_run) = rho;
			// E0_stored.at(idx_run) = E0;
		}

		Rcout << "\rProgress: " << b << "/" << ntotal - 1;
	}

	Rcout << std::endl;

	Rcpp::List output;
	output["wt"] = Rcpp::wrap(wt_stored); // n x nsample

	arma::mat psi_stored(n + 1, nsample);
	psi_stored.tail_rows(n) = arma::cumsum(wt_stored, 0);

	if (summarize_return)
	{
		arma::vec qProb = {0.025, 0.5, 0.975};
		output["psi"] = Rcpp::wrap(arma::quantile(psi_stored, qProb, 1)); // (n+1) x 3
	}
	else
	{
		output["psi"] = Rcpp::wrap(psi_stored);
	}

	wt_accept /= static_cast<double>(ntotal);
	output["wt_accept"] = Rcpp::wrap(wt_accept);
	output["logratio"] = Rcpp::wrap(logratio_stored);
	output["Bs"] = Rcpp::wrap(Bs_stored);

	output["W"] = Rcpp::wrap(W_stored);
	output["W_accept"] = W_accept / static_cast<double>(ntotal);

	// output["rho"] = Rcpp::wrap(rho_stored);
	// output["rho_accept"] = rho_accept / static_cast<double>(ntotal);
	// output["E0"] = Rcpp::wrap(E0_stored);
	// output["E0_accept"] = E0_accept / static_cast<double>(ntotal);
	return output;
}

// /*
// With Solows's pascal-distributed transfer function
// */
// //' @export
// // [[Rcpp::export]]
// Rcpp::List mcmc_disturbance_pois_solow(
// 	const arma::vec& Y, // n x 1, the observed response
// 	const arma::vec& X, // n x 1, the exogeneous variable in the transfer function
// 	const double Vxi = NA_REAL, // variance of mh normal random walk proposal for xi=logit(rho)
// 	const Rcpp::Nullable<Rcpp::NumericVector>& QPrior = R_NilValue, // (nv,Sv), prior for Q~IG(nv/2,nv*Sv/2)
// 	const Rcpp::Nullable<Rcpp::NumericVector>& v1Prior = R_NilValue, // (aw, Rw), v1 ~ N(aw, Rw), prior for v[1], the first state/evolution/state error/disturbance.
// 	const double rho_init = NA_REAL,
// 	const double Q_init = NA_REAL,
// 	const Rcpp::Nullable<Rcpp::NumericVector>& vt_init = R_NilValue,
// 	const double rho_true = NA_REAL, // true value of state/evolution coefficient
// 	const double Q_true = NA_REAL, // true value of state/evolution error variance
// 	const Rcpp::Nullable<Rcpp::NumericVector>& vt_true = R_NilValue, // true value of system/evolution/state error/disturbance
// 	const unsigned int nburnin = 0,
// 	const unsigned int nthin = 1,
// 	const unsigned int nsample = 1) { // n x 1

// 	const double EBOUND = 700.;

// 	const unsigned int n = Y.n_elem;
// 	const double n_ = static_cast<double>(n);
// 	const unsigned int ntotal = nburnin + nthin*nsample + 1;

// 	// Hyperparameter
// 	double Q,nv,nSv,nv_new,nSv_new;
// 	arma::vec Q_stored(nsample);
// 	bool Qflag;
// 	if (R_IsNA(Q_true)) {
// 		Qflag = true;
// 		if (QPrior.isNull()) {
// 			stop("Error: You must provide either true value or prior for Q.");
// 		}
// 		arma::vec QPrior_ = Rcpp::as<arma::vec>(QPrior);
// 		nv = QPrior_.at(0);
// 		nSv = nv*QPrior_.at(1);
// 		nv_new = nv + n_ - 1.;
// 		if (R_IsNA(Q_init)) {
// 			Q = 1./R::rgamma(0.5*nv,2./nSv);
// 		} else {
// 			Q = Q_init;
// 		}

// 	} else {
// 		Qflag = false;
// 		Q = Q_true;
// 	}
// 	double Qsqrt = std::sqrt(Q);

// 	double rho, xi, xi_old, xi_new;
// 	arma::vec rho_stored(nsample);
// 	double rho_accept = 0.;
// 	bool rhoflag;
// 	if (R_IsNA(rho_true)) {
// 		rhoflag = true;
// 		if (R_IsNA(Vxi)) {
// 			stop("Error: You must provide either true value or prior for rho.");
// 		}
// 		if (R_IsNA(rho_init)) {
// 			rho = 0.5;
// 		} else {
// 			rho = rho_init;
// 		}
// 	} else {
// 		rhoflag = false;
// 		rho = rho_true;
// 	}
// 	// double rho_sq = rho * rho;

// 	bool vtflag = true;
// 	double aw,Rw,Rwsqrt;
// 	arma::vec vt(n);
// 	arma::vec vt_accept(n);
// 	arma::mat v_stored(n,nsample,arma::fill::zeros);
// 	if (!vt_true.isNull()) {
// 		vtflag = false;
// 		vt = Rcpp::as<arma::vec>(vt_true);
// 	} else {
// 		vtflag = true;
// 		if (!v1Prior.isNull()) {
// 			arma::vec v1Prior_ = Rcpp::as<arma::vec>(v1Prior);
// 			aw = v1Prior_.at(0);
// 			Rw = v1Prior_.at(1);
// 			Rwsqrt = std::sqrt(Rw);
// 		}
// 		if (!vt_init.isNull()) {
// 			vt = Rcpp::as<arma::vec>(vt_init);
// 		} else {
// 			vt = aw + arma::randn(n)*std::sqrt(Rw);
// 		}
// 	}

// 	arma::mat Fx = update_Fx_Solow(n,rho,X);

//     double bt,Bt,Btsqrt;
//     arma::vec Et(n);
// 	arma::vec lambda(n);
// 	arma::vec Ytilde(n);
// 	arma::vec Yhat(n);

// 	double vt_old, vt_new;
// 	double logp_old,logp_new,logq_old,logq_new,logratio;

// 	bool saveiter;

// 	for (unsigned int b=0; b<ntotal; b++) {
// 		R_CheckUserInterrupt();
// 		saveiter = b > nburnin && ((b-nburnin-1)%nthin==0);

// 		// [OK] Update evolution/state disturbances/errors, denoted by vt.
// 		for (unsigned int t=0; t<n; t++) {
// 			if (!vtflag) { break; }

// 			/* Part 1. Target full conditional based on old value */
// 			vt_old = vt.at(t);
// 		    Et = Fx * vt;
// 			// Et.elem(arma::find(Et>EBOUND)).fill(EBOUND); //**
// 			lambda = arma::exp(Et);
// 			logp_old = 0.;
// 			for (unsigned int j=t;j<n;j++) {
// 				logp_old += R::dpois(Y.at(j),lambda.at(j),true);
// 			}
// 			if (t==0) {
// 				logp_old += R::dnorm(vt_old,aw,Rwsqrt,true);
// 			} else {
// 				logp_old += R::dnorm(vt_old,0.,Qsqrt,true);
// 			}
// 			/* Part 1. Checked - OK */

// 			/* Part 2. Proposal new | old */
// 			Ytilde = Et + (Y - lambda) / lambda; // Linearlisation
// 		    Yhat = Ytilde - Et + Fx.col(t)*vt_old; // TODO - CHECK HERE
// 		    if (t==0) {
//         	    Bt = 1. / (1./Rw + arma::accu(arma::pow(Fx.col(t),2.) % lambda));
//         	    bt = Bt * (aw/Rw + arma::accu(Fx.col(t)%Yhat%lambda));
//       	    } else {
//       		    Bt = 1./(1./Q + arma::accu(arma::pow(Fx.col(t),2.) % lambda));
//         	    bt = Bt * (arma::accu(Fx.col(t)%Yhat%lambda));
//       	    }
// 			Btsqrt = std::sqrt(Bt);
// 		    vt_new = R::rnorm(bt, Btsqrt);
// 			if (!std::isfinite(vt_new)) {
// 				Rcout << "(bt=" << bt << ", Bt=" << Bt << ")" << std::endl;
// 				Rcout << "vt: " << vt.t() << std::endl;
// 				Rcout << "Yhat: " << Yhat.t() << std::endl;
// 				Rcout << "Et: " << Et.t() << std::endl;
// 				Rcout << "lambda: " << lambda.t() << std::endl;
// 				stop("Non finite vt");
// 			}
// 			logq_new = R::dnorm(vt_new,bt,Btsqrt,true);
// 			/* Part 2. */

// 			vt.at(t) = vt_new;
// 			Et = Fx * vt;
// 			// Et.elem(arma::find(Et>EBOUND)).fill(EBOUND);
// 			lambda = arma::exp(Et);
// 			logp_new = 0.;
// 			for (unsigned int j=t;j<n;j++) {
// 				logp_new += R::dpois(Y.at(j),lambda.at(j),true);
// 			}
// 			if (t==0) {
// 				logp_new += R::dnorm(vt_new,aw,Rwsqrt,true);
// 			} else {
// 				logp_new += R::dnorm(vt_new,0.,Qsqrt,true);
// 			}

// 			Ytilde = Et + (Y - lambda) / lambda; // linearlisation
// 		    Yhat = Ytilde - Et + Fx.col(t)*vt_new;
// 		    if (t==0) {
//         	    Bt = 1. / (1./Rw + arma::accu(arma::pow(Fx.col(t),2.)%lambda));
//         	    bt = Bt * (aw/Rw + arma::accu(Fx.col(t)%Yhat%lambda));
//       	    } else {
//       		    Bt = 1./(1./Q + arma::accu(arma::pow(Fx.col(t),2.)%lambda));
//         	    bt = Bt * (arma::accu(Fx.col(t)%Yhat%lambda));
//       	    }
// 			Btsqrt = std::sqrt(Bt);
// 		    vt_new = R::rnorm(bt, Btsqrt);
// 			logq_old = R::dnorm(vt_old,bt,Btsqrt,true);

// 			logratio = std::min(0.,logp_new-logp_old+logq_old-logq_new);
// 			if (std::log(R::runif(0.,1.)) >= logratio) { // reject
// 				vt.at(t) = vt_old;
// 			} else {
// 				vt_accept.at(t) += 1.;
// 			}
// 	    }

// 		// [OK] Update state/evolution error variance
// 		if (Qflag) {
// 			nSv_new = nSv + arma::accu(arma::pow(vt.tail(n-1),2.));
// 			Q = 1./R::rgamma(0.5*nv_new,2./nSv_new);
// 		}

// 		// Update rho: the state/evolution coefficient
// 		if (rhoflag) {
// 			// if (std::abs(rho)<arma::datum::eps) {
// 			// 	rho += arma::datum::eps;
// 			// }
// 			xi_old = std::log(rho/(1.-rho));
// 			Et = Fx * vt;
// 			// Et.elem(arma::find(Et>EBOUND)).fill(EBOUND);
// 			lambda = arma::exp(Et);
// 			logp_old = xi_old - 2.*std::log(std::exp(xi_old)+1.);
// 			for (unsigned int t=0; t<n; t++) {
// 				logp_old += R::dpois(Y.at(t),lambda.at(t),true);
// 			}

// 			xi_new = R::rnorm(xi_old,std::sqrt(Vxi));
// 			rho = 1./(1.+std::exp(-xi_new));
// 			Fx = update_Fx_Solow(n,rho,X);
// 			Et = Fx * vt;
// 			// Et.elem(arma::find(Et>EBOUND)).fill(EBOUND);
// 			lambda = arma::exp(Et);
// 			logp_new = xi_new - 2.*std::log(std::exp(xi_new)+1.);
// 			for (unsigned int t=0; t<n; t++) {
// 				logp_new += R::dpois(Y.at(t),lambda.at(t),true);
// 			}

// 			logratio = std::min(0.,logp_new-logp_old);
// 			if (std::log(R::runif(0.,1.)) >= logratio) { // reject
// 				rho = 1./(1.+std::exp(-xi_old));
// 				Fx = update_Fx_Solow(n,rho,X);
// 			} else {
// 				rho_accept += 1.;
// 			}
// 			// rho_sq = rho * rho;

// 			if (!std::isfinite(rho)) {
// 				Rcout << "rho_new=" << rho << std::endl;
// 				stop("Non finite value for rho");
// 			}
// 		}

// 		// store samples after burnin and thinning
// 		if (saveiter || b==(ntotal-1)) {
// 			unsigned int idx_run;
// 			if (saveiter) {
// 				idx_run = (b-nburnin-1)/nthin;
// 			} else {
// 				idx_run = nsample - 1;
// 			}

// 			v_stored.col(idx_run) = vt;
// 			Q_stored.at(idx_run) = Q;
// 			rho_stored.at(idx_run) = rho;
// 		}

// 		// Rcout << "\rProgress: " << b << "/" << ntotal-1;
// 	}

// 	// Rcout << std::endl;

// 	Rcpp::List output;
// 	output["v"] = Rcpp::wrap(v_stored);
// 	vt_accept /= static_cast<double>(ntotal);
// 	output["v_accept"] = Rcpp::wrap(vt_accept);
// 	output["Q"] = Rcpp::wrap(Q_stored);
// 	output["rho"] = Rcpp::wrap(rho_stored);
// 	output["rho_accept"] = rho_accept / static_cast<double>(ntotal);
// 	return output;
// }

// /*
// With Solows's pascal-distributed transfer function
// and identity link function
// */
// //' @export
// // [[Rcpp::export]]
// Rcpp::List mcmc_disturbance_pois_solow_eye(
// 	const arma::vec& Y, // n x 1, the observed response
// 	const arma::vec& X, // n x 1, the exogeneous variable in the transfer function
// 	const double Vxi = NA_REAL, // variance of mh normal random walk proposal for xi=logit(rho)
// 	const Rcpp::Nullable<Rcpp::NumericVector>& QPrior = R_NilValue, // (nv,Sv), prior for Q~IG(nv/2,nv*Sv/2)
// 	const Rcpp::Nullable<Rcpp::NumericVector>& v1Prior = R_NilValue, // (aw, Rw), v1 ~ N(aw, Rw), prior for v[1], the first state/evolution/state error/disturbance.
// 	const double rho_init = NA_REAL,
// 	const double Q_init = NA_REAL,
// 	const Rcpp::Nullable<Rcpp::NumericVector>& vt_init = R_NilValue,
// 	const double rho_true = NA_REAL, // true value of state/evolution coefficient
// 	const double Q_true = NA_REAL, // true value of state/evolution error variance
// 	const Rcpp::Nullable<Rcpp::NumericVector>& vt_true = R_NilValue, // true value of system/evolution/state error/disturbance
// 	const unsigned int nburnin = 0,
// 	const unsigned int nthin = 1,
// 	const unsigned int nsample = 1) { // n x 1

// 	const double EBOUND = 700.;

// 	const unsigned int n = Y.n_elem;
// 	const double n_ = static_cast<double>(n);
// 	const unsigned int ntotal = nburnin + nthin*nsample + 1;

// 	// Hyperparameter
// 	double Q,nv,nSv,nv_new,nSv_new;
// 	arma::vec Q_stored(nsample);
// 	bool Qflag;
// 	if (R_IsNA(Q_true)) {
// 		Qflag = true;
// 		if (QPrior.isNull()) {
// 			stop("Error: You must provide either true value or prior for Q.");
// 		}
// 		arma::vec QPrior_ = Rcpp::as<arma::vec>(QPrior);
// 		nv = QPrior_.at(0);
// 		nSv = nv*QPrior_.at(1);
// 		nv_new = nv + n_ - 1.;
// 		if (R_IsNA(Q_init)) {
// 			Q = 1./R::rgamma(0.5*nv,2./nSv);
// 		} else {
// 			Q = Q_init;
// 		}

// 	} else {
// 		Qflag = false;
// 		Q = Q_true;
// 	}
// 	double Qsqrt = std::sqrt(Q);

// 	double rho, xi, xi_old, xi_new, Vxi_sqrt;
// 	arma::vec rho_stored(nsample);
// 	double rho_accept = 0.;
// 	bool rhoflag;
// 	if (R_IsNA(rho_true)) {
// 		rhoflag = true;
// 		if (R_IsNA(Vxi)) {
// 			stop("Error: You must provide either true value or prior for rho.");
// 		}
// 		if (R_IsNA(rho_init)) {
// 			rho = 0.5;
// 		} else {
// 			rho = rho_init;
// 		}
// 		Vxi_sqrt = std::sqrt(Vxi);
// 	} else {
// 		rhoflag = false;
// 		rho = rho_true;
// 	}
// 	// double rho_sq = rho * rho;

// 	bool vtflag = true;
// 	double aw,Rw,Rwsqrt;
// 	arma::vec vt(n);
// 	arma::vec vt_accept(n);
// 	arma::mat v_stored(n,nsample,arma::fill::zeros);
// 	if (!vt_true.isNull()) {
// 		vtflag = false;
// 		vt = Rcpp::as<arma::vec>(vt_true);
// 	} else {
// 		vtflag = true;
// 		if (!v1Prior.isNull()) {
// 			arma::vec v1Prior_ = Rcpp::as<arma::vec>(v1Prior);
// 			aw = v1Prior_.at(0);
// 			Rw = v1Prior_.at(1);
// 			Rwsqrt = std::sqrt(Rw);
// 		}
// 		if (!vt_init.isNull()) {
// 			vt = Rcpp::as<arma::vec>(vt_init);
// 		} else {
// 			vt = aw + arma::randn(n)*std::sqrt(Rw);
// 		}
// 	}

// 	arma::mat Fx = update_Fx_Solow(n,rho,X);

//     double bt,Bt,Btsqrt;
//     arma::vec Et(n);
// 	arma::vec lambda(n);
// 	arma::vec Ytilde(n);
// 	arma::vec Yhat(n);

// 	double vt_old, vt_new;
// 	double logp_old,logp_new,logq_old,logq_new,logratio;

// 	bool saveiter;

// 	for (unsigned int b=0; b<ntotal; b++) {
// 		R_CheckUserInterrupt();
// 		saveiter = b > nburnin && ((b-nburnin-1)%nthin==0);

// 		// [OK] Update evolution/state disturbances/errors, denoted by vt.
// 		for (unsigned int t=0; t<n; t++) {
// 			// Metropolis-Hastings for each of the vt
// 			if (!vtflag) { break; }

// 			/* Part 1. Target full conditional based on old value */
// 			vt_old = vt.at(t);
// 		    Et = Fx * vt;
// 			// Et.elem(arma::find(Et>EBOUND)).fill(EBOUND); //**
// 			// lambda = arma::exp(Et);
// 			lambda = Et;
// 			lambda.elem(arma::find(lambda<=0)).fill(arma::datum::eps);
// 			logp_old = 0.;
// 			for (unsigned int j=t;j<n;j++) {
// 				logp_old += R::dpois(Y.at(j),lambda.at(j),true);
// 			}
// 			if (t==0) {
// 				logp_old += R::dnorm(vt_old,aw,Rwsqrt,true);
// 			} else {
// 				logp_old += R::dnorm(vt_old,0.,Qsqrt,true);
// 			}
// 			/* Part 1. Checked - OK */

// 			/* Part 2. Proposal new | old */
// 			// Ytilde = Et + (Y - lambda) / lambda; // Linearlisation
// 		    Yhat = Y - Et + Fx.col(t)*vt_old; // TODO - CHECK HERE
// 		    if (t==0) {
//         	    Bt = 1. / (1./Rw + arma::accu(arma::pow(Fx.col(t),2.) % lambda));
//         	    bt = Bt * (aw/Rw + arma::accu(Fx.col(t)%Yhat%lambda));
//       	    } else {
//       		    Bt = 1./(1./Q + arma::accu(arma::pow(Fx.col(t),2.) % lambda));
//         	    bt = Bt * (arma::accu(Fx.col(t)%Yhat%lambda));
//       	    }
// 			Btsqrt = std::sqrt(Bt);
// 		    vt_new = R::rnorm(bt, Btsqrt);
// 			if (!std::isfinite(vt_new)) {
// 				Rcout << "(bt=" << bt << ", Bt=" << Bt << ")" << std::endl;
// 				Rcout << "vt: " << vt.t() << std::endl;
// 				Rcout << "Yhat: " << Yhat.t() << std::endl;
// 				Rcout << "Et: " << Et.t() << std::endl;
// 				Rcout << "lambda: " << lambda.t() << std::endl;
// 				stop("Non finite vt");
// 			}
// 			logq_new = R::dnorm(vt_new,bt,Btsqrt,true);
// 			/* Part 2. */

// 			vt.at(t) = vt_new;
// 			Et = Fx * vt;
// 			// Et.elem(arma::find(Et>EBOUND)).fill(EBOUND);
// 			// lambda = arma::exp(Et);
// 			lambda = Et;
// 			lambda.elem(arma::find(lambda<=0)).fill(arma::datum::eps);
// 			logp_new = 0.;
// 			for (unsigned int j=t;j<n;j++) {
// 				logp_new += R::dpois(Y.at(j),lambda.at(j),true);
// 			}
// 			if (t==0) {
// 				logp_new += R::dnorm(vt_new,aw,Rwsqrt,true);
// 			} else {
// 				logp_new += R::dnorm(vt_new,0.,Qsqrt,true);
// 			}

// 			// Ytilde = Et + (Y - lambda) / lambda; // linearlisation
// 		    Yhat = Y - Et + Fx.col(t)*vt_new;
// 		    if (t==0) {
//         	    Bt = 1. / (1./Rw + arma::accu(arma::pow(Fx.col(t),2.)%lambda));
//         	    bt = Bt * (aw/Rw + arma::accu(Fx.col(t)%Yhat%lambda));
//       	    } else {
//       		    Bt = 1./(1./Q + arma::accu(arma::pow(Fx.col(t),2.)%lambda));
//         	    bt = Bt * (arma::accu(Fx.col(t)%Yhat%lambda));
//       	    }
// 			Btsqrt = std::sqrt(Bt);
// 			logq_old = R::dnorm(vt_old,bt,Btsqrt,true);

// 			logratio = std::min(0.,logp_new-logp_old+logq_old-logq_new);
// 			if (std::log(R::runif(0.,1.)) >= logratio) { // reject
// 				vt.at(t) = vt_old;
// 			} else {
// 				vt_accept.at(t) += 1.;
// 			}
// 	    }

// 		// [OK] Update state/evolution error variance
// 		if (Qflag) {
// 			nSv_new = nSv + arma::accu(arma::pow(vt.tail(n-1),2.));
// 			Q = 1./R::rgamma(0.5*nv_new,2./nSv_new);
// 		}

// 		// Update rho: the state/evolution coefficient
// 		// ATTENTION: POOR PERFORMANCE
// 		if (rhoflag) {
// 			// if (std::abs(rho)<arma::datum::eps) {
// 			// 	rho += arma::datum::eps;
// 			// }
// 			xi_old = std::log(rho/(1.-rho));
// 			Et = Fx * vt;
// 			// Et.elem(arma::find(Et>EBOUND)).fill(EBOUND);
// 			// lambda = arma::exp(Et);
// 			lambda = Et;
// 			lambda.elem(arma::find(lambda<=0)).fill(arma::datum::eps);
// 			logp_old = xi_old - 2.*std::log(std::exp(xi_old)+1.);
// 			for (unsigned int t=0; t<n; t++) {
// 				logp_old += R::dpois(Y.at(t),lambda.at(t),true);
// 			}

// 			xi_new = R::rnorm(xi_old,Vxi_sqrt);
// 			rho = 1./(1.+std::exp(-xi_new));
// 			Fx = update_Fx_Solow(n,rho,X);
// 			Et = Fx * vt;
// 			// Et.elem(arma::find(Et>EBOUND)).fill(EBOUND);
// 			// lambda = arma::exp(Et);
// 			lambda = Et;
// 			lambda.elem(arma::find(lambda<=0)).fill(arma::datum::eps);
// 			logp_new = xi_new - 2.*std::log(std::exp(xi_new)+1.);
// 			for (unsigned int t=0; t<n; t++) {
// 				logp_new += R::dpois(Y.at(t),lambda.at(t),true);
// 			}

// 			logratio = std::min(0.,logp_new-logp_old);
// 			if (std::log(R::runif(0.,1.)) >= logratio) { // reject
// 				rho = 1./(1.+std::exp(-xi_old));
// 				Fx = update_Fx_Solow(n,rho,X);
// 			} else {
// 				rho_accept += 1.;
// 			}
// 			// rho_sq = rho * rho;

// 			if (!std::isfinite(rho)) {
// 				Rcout << "rho_new=" << rho << std::endl;
// 				stop("Non finite value for rho");
// 			}
// 		}

// 		// store samples after burnin and thinning
// 		if (saveiter || b==(ntotal-1)) {
// 			unsigned int idx_run;
// 			if (saveiter) {
// 				idx_run = (b-nburnin-1)/nthin;
// 			} else {
// 				idx_run = nsample - 1;
// 			}

// 			v_stored.col(idx_run) = vt;
// 			Q_stored.at(idx_run) = Q;
// 			rho_stored.at(idx_run) = rho;
// 		}

// 		// Rcout << "\rProgress: " << b << "/" << ntotal-1;
// 	}

// 	// Rcout << std::endl;

// 	Rcpp::List output;
// 	output["v"] = Rcpp::wrap(v_stored);
// 	vt_accept /= static_cast<double>(ntotal);
// 	output["v_accept"] = Rcpp::wrap(vt_accept);
// 	output["Q"] = Rcpp::wrap(Q_stored);
// 	output["rho"] = Rcpp::wrap(rho_stored);
// 	output["rho_accept"] = rho_accept / static_cast<double>(ntotal);
// 	return output;
// }