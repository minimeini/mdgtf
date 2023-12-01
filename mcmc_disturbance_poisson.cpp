#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <algorithm>
#include <RcppArmadillo.h>
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

arma::vec get_Vt_hat(
	const arma::vec &y,			 // n x 1
	const arma::vec &wt,		 // n x 1
	const arma::vec &Fphi_pad,
	const unsigned int &gain_code,
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
	arma::vec lambda_hat(n,arma::fill::zeros);
	arma::vec Vt_hat(n,arma::fill::zeros);
	arma::mat Fn_hat(n,n,arma::fill::zeros);



	arma::vec psi = arma::cumsum(wt);
	arma::vec hderiv = hpsi_deriv(psi, gain_code); // n x 1
	Fn_hat = update_Fn(y, Fphi_pad, hderiv, n);

	arma::vec theta0_hat = update_theta0(y, Fphi_pad, psi, hderiv, n, gain_code);

	arma::vec theta1 = Fn_hat * wt;
	lambda_hat = theta0_hat + theta1;
	lambda_hat.for_each([&mu0](arma::vec::elem_type &val)
						{ val += mu0; });

	if (obs_code == 0)
	{
		// nbinom obs + identity link
		// lambda[t] + (lambda[t])^2 / delta_nb
		arma::vec tmp1 = lambda_hat;
		tmp1.for_each([&delta_nb](arma::vec::elem_type &val)
					  { val += delta_nb; });

		arma::vec tmp2 = lambda_hat % tmp1;
		tmp2.for_each([&delta_nb](arma::vec::elem_type &val)
					  { val /= delta_nb; });

		Vt_hat = tmp2;
	}
	else if (obs_code == 1)
	{
		// Poisson obs + identity link
		Vt_hat = lambda_hat;
	}
	else
	{
		throw std::invalid_argument("proposal_mh_var: only support Poisson or NB likelihood.");
	}

	bound_check(Vt_hat, "get_yhat_params: Vt_hat", true, true);
	Vt_hat.elem(arma::find(Vt_hat < EPS8)).fill(EPS8);
	return Vt_hat;
}

arma::vec get_Vt_hat(
	arma::mat &Fn_hat,
	const arma::vec &y,	 // n x 1
	const arma::vec &wt, // n x 1
	const arma::vec &Fphi_pad,
	const unsigned int &gain_code,
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
	arma::vec lambda_hat(n, arma::fill::zeros);
	arma::vec Vt_hat(n, arma::fill::zeros);

	arma::vec psi = arma::cumsum(wt);
	arma::vec hderiv = hpsi_deriv(psi, gain_code); // n x 1

	Fn_hat.set_size(n,n); Fn_hat.zeros();
	Fn_hat = update_Fn(y, Fphi_pad, hderiv, n);

	arma::vec theta0_hat = update_theta0(y, Fphi_pad, psi, hderiv, n, gain_code);

	arma::vec theta1 = Fn_hat * wt;
	lambda_hat = theta0_hat + theta1;
	lambda_hat.for_each([&mu0](arma::vec::elem_type &val)
						{ val += mu0; });

	if (obs_code == 0)
	{
		// nbinom obs + identity link
		// lambda[t] + (lambda[t])^2 / delta_nb
		arma::vec tmp1 = lambda_hat;
		tmp1.for_each([&delta_nb](arma::vec::elem_type &val)
					  { val += delta_nb; });

		arma::vec tmp2 = lambda_hat % tmp1;
		tmp2.for_each([&delta_nb](arma::vec::elem_type &val)
					  { val /= delta_nb; });

		Vt_hat = tmp2;
	}
	else if (obs_code == 1)
	{
		// Poisson obs + identity link
		Vt_hat = lambda_hat;
	}
	else
	{
		throw std::invalid_argument("proposal_mh_var: only support Poisson or NB likelihood.");
	}

	bound_check(Vt_hat, "get_yhat_params: Vt_hat", true, true);
	Vt_hat.elem(arma::find(Vt_hat < EPS8)).fill(EPS8);
	return Vt_hat;
}

double proposal_mh_var(
	const arma::vec &y,			 // n x 1
	const arma::vec &Vt_hat,
	const arma::mat &Fn_hat,	 // n x n
	const unsigned int &sidx)
{
	

	arma::vec Fn_col_s = Fn_hat.col(sidx);
	arma::vec Fn_s2 = Fn_col_s % Fn_col_s;
	arma::vec tmp = Fn_s2 / Vt_hat;	  // tmp too big: Vt too small or Fn_s2 too big
	double mh_prec = arma::accu(tmp); // precision of MH proposal
	bound_check(mh_prec,"proposal_mh_var: mh_prec",true,true);
	double mh_var = 1./mh_prec;

	return mh_var;
}

/**
 * logpost_wt: Checked. OK.
*/
double loglike_obs(
	const arma::vec &y,	 // n x 1
	const arma::vec &wt, // n x 1
	const unsigned int &sidx,
	const unsigned int &gain_code,
	const unsigned int &trans_code,
	const unsigned int &link_code,
	const unsigned int &obs_code,
	const double &mu0,
	const unsigned int &delta_nb,
	const double &rho,
	const unsigned int &L,
	const unsigned int &nlag = 0) // prior SD of w[s]
{
	const unsigned int n = y.n_elem;
	// double Rw_sqrt = std::sqrt(Rw);

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

	// double logpost = R::dnorm4(wt.at(sidx), aw, Rw_sqrt, true);
	double logpost = 0.;
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
	arma::vec &logp,
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
	double sg_wt = std::sqrt(prior_params.at(1));

	
	for (unsigned int s = 0; s < nobs; s++)
	{
		arma::mat Fn_hat(nobs,nobs,arma::fill::zeros);
		arma::vec Vt_hat = get_Vt_hat(
			Fn_hat, y, wt, Fphi_pad,
			gain_code, link_code, obs_code,
			delta_nb, mu0);

		// logp_old
		double wt_old = wt.at(s);

		double logp_old = loglike_obs(
			y, wt, s,
			gain_code, trans_code, link_code, obs_code,
			mu0, delta_nb, rho, L_order, nlag);
		logp_old += R::dnorm4(wt_old, mu_wt, sg_wt, true);

		Bs.at(s) = proposal_mh_var(y, Vt_hat, Fn_hat, s);

		double Btmp = std::sqrt(Bs.at(s));
		Btmp *= mh_sd;

		double wt_new = R::rnorm(wt_old, Btmp);

		wt.at(s) = wt_new;
		double logp_new = loglike_obs(
			y, wt, s,
			gain_code, trans_code, link_code, obs_code,
			mu0, delta_nb, rho, L_order, nlag); // likelihood
		logp_new += R::dnorm4(wt_new, mu_wt, sg_wt, true); // prior

		double logratio = logp_new - logp_old;
		// logratio += logq_old - logq_new;
		logratio = std::min(0., logratio);

		double logps = 0.;
		if (std::log(R::runif(0., 1.)) < logratio)
		{
			// accept
			logps = logp_new;
			wt_accept.at(s) += 1.;
		}
		else
		{
			// reject
			logps = logp_old;
			wt.at(s) = wt_old;
		}

		logp.at(s) = logps;
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

	bound_check(W, "update: W", true, true);
	return W;
}

void update_mu0(
	double &mu0,
	double &mu0_accept,
	double &logp_mu0,
	const arma::vec &y,			   // nobs x 1
	const arma::vec &wt,
	const arma::vec &Fphi_pad,	   // (nobs + 1)
	const arma::vec &lags_params,  // (L_order, rho) or (mu, sg2)
	const double &delta_nb,
	const double &mh_sd,
	const unsigned int &nlag,
	const unsigned int &gain_code,
	const unsigned int &trans_code,
	const unsigned int &link_code,
	const unsigned int &obs_code)
{
	double rho = lags_params.at(0);
	unsigned int L_order = lags_params.at(1);
	
	arma::vec Vt_hat = get_Vt_hat(
		y, wt, Fphi_pad,
		gain_code, link_code, obs_code,
		delta_nb, mu0);

	double mu0_old = mu0;
	double logp_old = loglike_obs(
		y, wt, 0,
		gain_code, trans_code, link_code, obs_code,
		mu0_old, delta_nb, rho, L_order, nlag);

	arma::vec tmp = 1. / Vt_hat;
	double mu0_prec = arma::accu(tmp);
	double mu0_var = 1. / mu0_prec;
	double mu0_sd = std::sqrt(mu0_var);
	double mu0_new = R::rnorm(mu0_old, mu0_sd * mh_sd);

	logp_mu0 = logp_old;
	if (mu0_new > -EPS) // non-negative
	{
		double logp_new = loglike_obs(
			y, wt, 0,
			gain_code, trans_code, link_code, obs_code,
			mu0_new, delta_nb, rho, L_order, nlag);
		double logratio = std::min(0., logp_new - logp_old);
		if (std::log(R::runif(0., 1.)) < logratio)
		{ // accept
			mu0 = mu0_new;
			mu0_accept += 1.;
			logp_mu0 = logp_new;
		}
	}
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
	const Rcpp::NumericVector &rho_par = Rcpp::NumericVector::create(1., 0., 1., 1.),	 // similar
	const double &L_order = 6,															 // Lag Koyama's model; order for Solow's model
	const unsigned int &nlag_ = 0,														 // 0: no truncation; nlag > 0: truncated to nlag
	const Rcpp::NumericVector &mh_var = Rcpp::NumericVector::create(1., 1.),			 // mh and discount factor for wt
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

	arma::vec mh_sd(mh_var.begin(), mh_var.length());
	mh_sd.for_each([](arma::vec::elem_type &val)
				   { val = std::sqrt(val); });

	// Global Parameter
	// eta = (W, mu0, rho)
	double W = W_par[0];
	double mu0 = 0.;
	double rho = rho_par[0];

	arma::vec lags_params = {(double)L, rho};
	arma::vec obs_params = {delta_nb, mu0};
	arma::vec prior_params = {0., W};

	// idx = std::max(1,(int) npar);
	arma::mat param_stored(nsample, 3, arma::fill::zeros);

	bool W_selected = std::abs(eta_select[0] - 1.) < EPS8;
	bool mu0_selected = std::abs(eta_select[1] - 1.) < EPS8;

	
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
	

	arma::vec wt = arma::randn(n) * 0.01;
	for (unsigned int t = 0; t < L; t++)
	{
		wt.at(t) = std::abs(wt.at(t));
	}
	

	bool saveiter;
	arma::mat logp_stored(n, nsample, arma::fill::zeros);
	arma::mat logp2_stored(3, nsample, arma::fill::zeros);

	arma::vec wt_accept(n, arma::fill::zeros);
	arma::mat wt_stored(n, nsample, arma::fill::zeros);
	arma::mat Bs_stored(n, nsample, arma::fill::zeros);
	arma::vec Fphi_pad; // nobs x 1
	get_Fphi_pad(Fphi_pad, trans_code, n, nlag, L, rho);

	double W_accept = 0.;
	double mu0_accept = 0.;

	for (unsigned int b = 0; b < ntotal; b++)
	{
		R_CheckUserInterrupt();
		saveiter = b > nburnin && ((b - nburnin - 1) % nthin == 0);

		

		// [OK] Update evolution/state disturbances/errors, denoted by wt.
		arma::vec Bs(n, arma::fill::zeros);
		arma::vec logp(n, arma::fill::zeros);
		update_wt(
			wt, wt_accept, Bs, logp, 
			y, Fphi_pad,
			lags_params,  // (r, rho)
			obs_params,	  // delta_nb, mu0
			prior_params, // w[t] ~ N(0, W)
			mh_sd.at(0), n, nlag,
			gain_code, trans_code, link_code, obs_code);

		// [OK] Update state/evolution error variance W
		if (W_selected)
		{
			double W_old = W;
			W = update_W(W_accept, W_old, wt, 
			(unsigned int) W_par[1], W_params, mh_sd.at(1), n);

			prior_params.at(1) = W;
		}


		double logp_mu0 = 0.;
		if (mu0_selected)
		{
			update_mu0(
				mu0, mu0_accept, logp_mu0, 
				y, wt, Fphi_pad, 
				lags_params, delta_nb, mh_sd.at(1), nlag, 
				gain_code, trans_code, link_code, obs_code);
			
			obs_params.at(1) = mu0;
		}



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
			logp_stored.col(idx_run) = logp;
			Bs_stored.col(idx_run) = Bs;

			if (W_selected)
			{
				param_stored.at(idx_run, 0) = W;
			}

			if (mu0_selected)
			{
				param_stored.at(idx_run, 1) = mu0;
				logp2_stored.at(1, idx_run) = logp_mu0;
			}

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
	output["logp"] = Rcpp::wrap(logp_stored);
	output["Bs"] = Rcpp::wrap(Bs_stored);

	output["params"] = Rcpp::wrap(param_stored);
	output["logp2"] = Rcpp::wrap(logp2_stored);

	if (W_selected)
	{
		output["W_accept"] = W_accept / static_cast<double>(ntotal);
	}
	if (mu0_selected)
	{
		output["mu0_accept"] = mu0_accept / static_cast<double>(ntotal);
	}

	return output;
}
