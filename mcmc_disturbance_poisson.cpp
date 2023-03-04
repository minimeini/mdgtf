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




/*
MCMC disturbance sampler for different transfer kernels, link functions, and reproduction number functions.

Unknown Parameters
	- local parameters: psi[1:n]
	- global parameters `eta`: W, mu0, rho, M,th0, psi0
*/
//' @export
// [[Rcpp::export]]
Rcpp::List mcmc_disturbance_pois(
	const arma::vec& Y, // n x 1, the observed response
	const arma::uvec& model_code, 
	const arma::uvec& eta_select, // 6 x 1, indicator for unknown (=1) or known (=0), global parameters: W, mu0, rho, M,th0, psi0
    const arma::vec& eta_init, // 6 x 1, if true/initial values should be provided here
    const arma::uvec& eta_prior_type, // 6 x 1
    const arma::mat& eta_prior_val, // 2 x 6, priors for each element of eta
	const double L = 12, // For Koyama's model
	const Rcpp::NumericVector& ctanh = Rcpp::NumericVector::create(0.2,0,5.),
	const Rcpp::Nullable<Rcpp::NumericVector>& w1_prior = R_NilValue, // (aw, Rw), w1 ~ N(aw, Rw), prior for w[1], the first state/evolution/state error/disturbance.
	const Rcpp::Nullable<Rcpp::NumericVector>& wt_init = R_NilValue, // initial value of the evolution error at time t = 1
	const Rcpp::Nullable<Rcpp::NumericVector>& wt_true = R_NilValue, // true value of system/evolution/state error/disturbance
	const Rcpp::Nullable<Rcpp::NumericVector>& ht_ = R_NilValue, // (n+1) x 1, (h[0],h[1],...,h[n]), smoothing means or Taylor expansion locations of (psi[0],...,psi[n]) | Y
	const Rcpp::NumericVector& MH_sd = Rcpp::NumericVector::create(1.4,0.01), // wt, W
	const unsigned int wt_mh_type = 0, // 0 - N(0,W); 1 - 
	const bool use_lambda_bound = false,
	const double delta_nb = 1.,
	const unsigned int nburnin = 0,
	const unsigned int nthin = 1,
	const unsigned int nsample = 1,
	const bool summarize_return = false,
	const bool verbose = true) { // n x 1

	const unsigned int obs_code = model_code.at(0);
    const unsigned int link_code = model_code.at(1);
    const unsigned int trans_code = model_code.at(2);
    const unsigned int gain_code = model_code.at(3);
    const unsigned int err_code = model_code.at(4);

	const unsigned int n = Y.n_elem;
	const double n_ = static_cast<double>(n);
	const unsigned int ntotal = nburnin + nthin*nsample + 1;

	// Global Parameter
	// eta = (W, mu0, rho, M, th0, psi0, w1)
	double W = eta_init.at(0);
	arma::vec W_stored(nsample);
	double W_accept = 0.;
	double nw = eta_prior_val.at(0,0);
	double nSw = eta_prior_val.at(0,0)*eta_prior_val.at(1,0);
	double nw_new = nw + n_ - 1.;
	double nSw_new;
	double Wrt = std::sqrt(W);
	double W_new;

	double mu0 = eta_init.at(1);
	double rho = eta_init.at(2);

	
	double th0 = eta_init.at(4);
	arma::vec th0_stored(nsample);
	double m_th0 = eta_prior_val.at(0,4);
	double C_th0 = eta_prior_val.at(1,4);
	double C_th0rt = std::sqrt(C_th0);
	double m_th, C_th, C_thrt; 
	double th0_new;
	double th0_accept = 0.;

	double psi0 = eta_init.at(5);

	// Local Parameter
	bool wt_flag = true;
	double aw,Rw,Rwrt;
	arma::vec wt(n); 
	arma::vec wt_accept(n);
	arma::mat wt_stored(n,nsample,arma::fill::zeros);
	if (!wt_true.isNull()) {
		wt_flag = false;
		wt = Rcpp::as<arma::vec>(wt_true);
	} else {
		wt_flag = true;
		if (!w1_prior.isNull()) {
			arma::vec w1Prior_ = Rcpp::as<arma::vec>(w1_prior);
			aw = w1Prior_.at(0);
			Rw = w1Prior_.at(1);
		} else {
			// Setting from Alves et al. (2010)
			aw = 0.;
			Rw = 100.;
		}
		Rwrt = std::sqrt(Rw);

		if (!wt_init.isNull()) {
			wt = Rcpp::as<arma::vec>(wt_init);
		} else {
			wt = aw + arma::randn(n)*Rwrt;
		}
	}


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

	

	// Initialize the states via linear bayes smoothing
	// Implemented: KoyamaMax (ModelCode = 0)
	arma::vec ht(n+1,arma::fill::zeros); // (h[0],h[1],...,h[n])
	if (!ht_.isNull()) {
		ht = Rcpp::as<arma::vec>(ht_);
		ht.at(0) = psi0;
	}

	// TODO: check this
	// First-order Taylor expansion of h(psi[t]) at h[t]
	arma::vec hh = psi2hpsi(ht,gain_code,ctanh);
	arma::vec hph(n+1,arma::fill::zeros);
	hpsi_deriv(hph,ht,gain_code,ctanh);
	arma::vec hhat = hh + hph%(psi0-ht); 

	arma::vec Fphi = get_Fphi(L); // L x 1

	// TODO: check this
	arma::mat Fx = update_Fx(trans_code,n,Y,hph,rho,L,Rcpp::wrap(Fphi));
	arma::vec th0tilde = update_theta0(trans_code,n,Y,hhat,th0,rho,L,Rcpp::wrap(Fphi));

	arma::vec theta(n,arma::fill::zeros); // n x 1
	arma::vec lambda(n,arma::fill::zeros); // n x 1

    double bt,Bt,Btrt;
	arma::vec Ytilde = Y;
	arma::vec Yhat(n);
	arma::vec Vt(n);
	arma::mat bt_stored(n,nsample);
	arma::mat Bt_stored(n,nsample);
	arma::vec bt_old(n);
	arma::vec Bt_old(n);

	double wt_old, wt_new;
	double logp_old,logp_new,logq_old,logq_new,logratio;
	
	arma::mat theta_stored(n,nsample,arma::fill::zeros);
	bool saveiter;

	for (unsigned int b=0; b<ntotal; b++) {
		R_CheckUserInterrupt();
		saveiter = b > nburnin && ((b-nburnin-1)%nthin==0);

		// [OK] Update evolution/state disturbances/errors, denoted by wt.
		for (unsigned int t=0; t<n; t++) {
			if (!wt_flag) { break; }

			/* Part 1. Checked - OK */
			wt_old = wt.at(t);
		    theta = th0tilde + Fx * wt;
			// Et.elem(arma::find(Et>EBOUND)).fill(EBOUND); //**

			// lambda filled with old values of wt[t]
			if (link_code==1) {
				// Exponential Link
				// 6 - KoyamaEye, 7 - SolowEye, 8 - KoyckEye
				lambda = arma::exp(mu0+theta);
			} else {
				// Others using identity link.
				lambda = mu0+theta;
			}

			logp_old = 0.;
			for (unsigned int j=t;j<n;j++) {
				logp_old += loglike_obs(Y.at(j), lambda.at(j), obs_code, delta_nb,true);
			}
			if (t==0) {
				logp_old += R::dnorm(wt_old,aw,Rwrt,true);
			} else {
				logp_old += R::dnorm(wt_old,0.,Wrt,true);
			}
			/* Part 1. Checked - OK */

			/* Part 2. */
			// Sample wt[t](new) given wt[t](old)
			if (link_code==1) {
				// Linearlisation for exponential link
				Ytilde = theta + (Y - lambda) / lambda; 
			} // Otherwise, Ytilde == Y.
			
		    Yhat = Ytilde - theta + Fx.col(t)*wt_old; // TODO - CHECK HERE
			if (obs_code==1 && link_code==1) {
				Vt = 1./lambda;
			} else if (obs_code==1 && link_code==0) {
				Vt = lambda;
			} else if (obs_code==0) {
				Vt = lambda % (lambda+delta_nb) / delta_nb;
				// Vt = lambda;
			}
			// if (t==0) {
			// 	Bt = 1. / (1./Rw + arma::accu(arma::pow(Fx.col(t),2.)/Vt));
        	//     bt = Bt * (aw/Rw + arma::accu(Fx.col(t)%Yhat/Vt));
			// } else {
			// 	Bt = 1./(1./W + arma::accu(arma::pow(Fx.col(t),2.)/Vt));
        	//     bt = Bt * (arma::accu(Fx.col(t)%Yhat/Vt));
			// }
			bt = 0.;
			Bt = W;
			bt_old.at(t)= bt;
			Bt_old.at(t) = Bt;

			Btrt = std::sqrt(Bt)*MH_sd[0];
		    wt_new = R::rnorm(bt, Btrt);
			if (!std::isfinite(wt_new)) {
				// Just reject it and move onto next t.
				wt.at(t) = wt_old;
				continue;
			}

			logq_new = R::dnorm(wt_new,bt,Btrt,true);
			/* Part 2. */

			/* Part 3. */
			wt.at(t) = wt_new;
			theta = th0tilde + Fx * wt;
			// Et.elem(arma::find(Et>EBOUND)).fill(EBOUND);
			// lambda filled with new values of wt[t]
			if (link_code==1) {
				// Exponential Link
				// 6 - KoyamaEye, 7 - SolowEye, 8 - KoyckEye, 9 - Vanilla
				lambda = arma::exp(mu0+theta);
				if (use_lambda_bound && (lambda.has_inf() || lambda.has_nan())) {
					wt.at(t) = wt_old; // Reject if out of bound
					continue;
				}
			} else {
				// Others using identity link.
				lambda = mu0+theta;
				if (use_lambda_bound && arma::any(lambda<arma::datum::eps)) {
					wt.at(t) = wt_old; // Reject if out of bound
					continue;
				}
			}
			logp_new = 0.;
			for (unsigned int j=t;j<n;j++) {
				logp_new += loglike_obs(Y.at(j), lambda.at(j), obs_code, delta_nb,true);
			}
			if (t==0) {
				logp_new += R::dnorm(wt_new,aw,Rwrt,true);
			} else {
				logp_new += R::dnorm(wt_new,0.,Wrt,true);
			}
			/* Part 3. */

			/* Part 4. */
			// Transition probability to wt[t](old) from wt[t](new)
			if (link_code==1) {
				// Linearlisation for exponential link
				Ytilde = theta + (Y - lambda) / lambda; 
			} // Otherwise, Ytilde == Y.

		    Yhat = Ytilde - theta + Fx.col(t)*wt_new;
			if (obs_code==1 && link_code==1) {
				Vt = 1./lambda;
			} else if (obs_code==1 && link_code==0) {
				Vt = lambda;
			} else if (obs_code==0) {
				Vt = lambda % (lambda+delta_nb) / delta_nb;
				// Vt = lambda;
			}
			// if (t==0) {
			// 	Bt = 1. / (1./Rw + arma::accu(arma::pow(Fx.col(t),2.)/Vt));
        	//     bt = Bt * (aw/Rw + arma::accu(Fx.col(t)%Yhat/Vt));
			// } else {
			// 	Bt = 1./(1./W + arma::accu(arma::pow(Fx.col(t),2.)/Vt));
        	//     bt = Bt * (arma::accu(Fx.col(t)%Yhat/Vt));
			// }
			bt = 0.;
			Bt = W;

			Btrt = std::sqrt(Bt)*MH_sd[0];
			logq_old = R::dnorm(wt_old,bt,Btrt,true);
			/* Part 4. */

			logratio = std::min(0.,logp_new-logp_old+logq_old-logq_new);
			if (std::log(R::runif(0.,1.)) >= logratio) { // reject
				wt.at(t) = wt_old;
			} else {
				wt_accept.at(t) += 1.;
			}
	    } // Loops


		// [OK] Update state/evolution error variance
		if (eta_select.at(0)==1) {
			switch (eta_prior_type.at(0)) {
				case 0: // Gamma(aw=shape, bw=rate)
				{
					double res = arma::accu(arma::pow(wt.tail(n-1),2.));
					logp_old = (eta_prior_val.at(0,0)-0.5*(n_-1.))*std::log(W) - eta_prior_val.at(1,0)*W - 0.5*res/W;

					W_new = std::exp(std::min(R::rnorm(std::log(W),MH_sd[1]),UPBND));

					logp_new = (eta_prior_val.at(0,0)-0.5*(n_-1.))*std::log(W_new) - eta_prior_val.at(1,0)*W_new - 0.5*res/W_new;

					logratio = std::min(0.,logp_new-logp_old);
					if (std::log(R::runif(0.,1.)) < logratio) { // accept
						W = W_new;
						W_accept += 1.;
					}
				}
				break;
				case 1: // Half-Cauchy(aw=location==0, bw=scale)
				{
					::Rf_error("Half-cauchy prior for W is not implemented yet.");
				}
				break;
				case 2: // Inverse-Gamma(nw=shape, nSw=rate)
				{
					nSw_new = nSw + arma::accu(arma::pow(wt.tail(n-1),2.));
					W = 1./R::rgamma(0.5*nw_new,2./nSw_new);
				}
				break;
				default:
				{
					::Rf_error("Unimplemented prior for W.");
				}
			}
			
			Wrt = std::sqrt(W);
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
		if (saveiter || b==(ntotal-1)) {
			unsigned int idx_run;
			if (saveiter) {
				idx_run = (b-nburnin-1)/nthin;
			} else {
				idx_run = nsample - 1;
			}

			wt_stored.col(idx_run) = wt;
			bt_stored.col(idx_run) = bt_old;
			Bt_stored.col(idx_run) = Bt_old;
			W_stored.at(idx_run) = W;
			theta_stored.col(idx_run) = th0tilde + Fx * wt;
			// rho_stored.at(idx_run) = rho;
			// E0_stored.at(idx_run) = E0;
		}

		if (verbose) {
			Rcout << "\rProgress: " << b << "/" << ntotal-1;
		}
		
	}

	if (verbose) {
		Rcout << std::endl;
	}

	Rcpp::List output;
	output["theta"] = Rcpp::wrap(theta_stored); // n x nsample
	arma::mat psi_stored(n+1,nsample);
	psi_stored.tail_rows(n) = arma::cumsum(wt_stored,0);
	psi_stored += psi0;
	if (summarize_return) {
		arma::vec qProb = {0.025,0.5,0.975};
		output["psi"] = Rcpp::wrap(arma::quantile(psi_stored,qProb,1)); // (n+1) x 3
	} else {
		output["psi"] = Rcpp::wrap(psi_stored);
	}
	wt_accept /= static_cast<double>(ntotal);
	output["wt_accept"] = Rcpp::wrap(wt_accept);
	output["bt"] = Rcpp::wrap(arma::median(bt_stored,1));
	output["Bt"] = Rcpp::wrap(arma::median(Bt_stored,1));
	if (eta_select.at(0)==1) {
		output["W"] = Rcpp::wrap(W_stored);
		if (eta_prior_type.at(0) != 2) {
			output["W_accept"] = W_accept / static_cast<double>(ntotal);
		}
	}
	
	
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