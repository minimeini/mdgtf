#include "lbe_poisson.h"
using namespace Rcpp;
// [[Rcpp::plugins(cpp17)]]
// [[Rcpp::depends(RcppArmadillo,nloptr)]]


/*
---------------------------------------
------ Desiderata of lbe_poisson ------
---------------------------------------

- Accommodate (1) known and fixed evolution variance $W$; (2) use of discount factor $\delta$ by assuming a time-varying evolution variance $W_t$.
- Different dimension of the state space, aka, dimension of $\widetilde{\boldsymbol{\theta}}_t$.
- Function that maps state equations to DLM form for different models.
- Accommodate different link functions, different gain functions, different transfer functions.
*/


/*
a[t] = gt(m[t-1])
*/
arma::mat update_at(
	const unsigned int p,
	const unsigned int gain_code,
	const unsigned int trans_code, // 0 - Koyck, 1 - Koyama, 2 - Solow
	const arma::mat& mt, // p x N, mt = (psi[t], theta[t], theta[t-1])
	const arma::mat& Gt, // p x p
	const Rcpp::NumericVector& ctanh = Rcpp::NumericVector::create(0.2,0,5.), // 3 x 1, coefficients for the hyperbolic tangent gain function
	const double alpha = 1.,
	const double y = NA_REAL,  // n x 1
	const double rho = NA_REAL) {
	
	const unsigned int N = mt.n_cols;
	arma::mat at(p,N,arma::fill::zeros);
	arma::rowvec hpsi;
	if (trans_code != 1) {
		hpsi = psi2hpsi(mt.row(0),gain_code,ctanh);
	}
	unsigned int r = p - 1;

	switch (trans_code) {
		case 0: // Koyck
		{
            at.row(0) = mt.row(0); // psi[t]
			at.row(1) = y * hpsi;
			at.row(1) += rho*mt.row(1);
		}
		break;
		case 1: // Koyama
		{
			at = Gt * mt;
		}
		break;
		case 2: // Solow - Checked. Correct.
		{
			double coef1 = std::pow(1.-rho,r*alpha);
			double coef2 = -rho;
            at.row(0) = mt.row(0); // psi[t]
			at.row(1) = coef1*y*hpsi - binom(r,1)*coef2*mt.row(1);
			for (unsigned int k=2; k<p; k++) {
				coef2 *= -rho;
				at.row(1) -= binom(r,k)*coef2*mt.row(k);
				at.row(k) = mt.row(k-1);
			}

			// at.elem(arma::find(at<EPS)).fill(EPS);

			// double coef1 = std::pow((1.-rho)*(1.-rho),alpha);
            // at.at(0) = mt.at(0); // psi[t]
			// at.at(1) = coef1*y*hpsi;
			// at.at(1) += 2*rho*mt.at(1) - rho*rho*mt.at(2);
            // at.at(2) = mt.at(1); // theta[t-1]
		}
		break;
		case 3: // Vanilla
		{
			at = Gt * mt;
		}
		break;
		default:
		{
			::Rf_error("get_at function is only defined for Vanilla, Koyama, Solow, or Koyck's transmission kernels.");
		}
	}

	return at;
}



/*
matrix Gt is the derivative of gt(.)
*/
void update_Gt(
	arma::mat& Gt, // p x p
	const unsigned int gain_code, 
	const unsigned int trans_code, // 0 - Koyck, 1 - Koyama, 2 - Solow
	const arma::vec& mt, // p x 1
	const Rcpp::NumericVector& ctanh = Rcpp::NumericVector::create(0.2,0,5.), // 3 x 1, coefficients for the hyperbolic tangent gain function
	const double alpha = 1.,
	const double y = NA_REAL, // obs
	const double rho = NA_REAL) {

    if (trans_code == 2) { // Solow - Checked. Correct.
		Gt.at(1,0) = hpsi_deriv(mt.at(0),gain_code,ctanh);
		Gt.at(1,0) *= std::pow(1.-rho,(Gt.n_rows-1.)*alpha)*y;
		// Gt.at(1,0) *= std::pow((1.-rho)*(1.-rho),alpha)*y;

	} else if (trans_code == 0) { // Koyck
		Gt.at(1,0) = hpsi_deriv(mt.at(0),gain_code,ctanh);
		Gt.at(1,0) *= y;
	}
}



void update_Rt(
	arma::mat& Rt, // p x p
	const arma::mat& Ct, // p x p
	const arma::mat& Gt, // p x p
	const bool use_discount, // true if !R_IsNA(delta)
	const double W = NA_REAL, // known evolution error of psi
	const double delta = NA_REAL) { // discount factor

	const unsigned int p = Rt.n_cols;
	Rt = Gt * Ct * Gt.t();
	if (use_discount) {
		// We should not use component discounting because Gt is not block-diagonal.
		Rt /= delta; // single discount factor
	} else {
		Rt.at(0,0) += W;
	}
}


/*
------------ update_Ft ------------
- only for Koyama transmission delay.
- only contain the phi[l]*y[t-l] part.
- doesn't include the gain function.
*/
void update_Ft(
	arma::vec& Ft, // L x 1
	arma::vec& Fy, // L x 1
	const unsigned int trans_code, // 0 - Koyck, 1 - Koyama, 2 - Solow
	const unsigned int t, // current time point
	const unsigned int L, // lag
	const arma::vec& Y,  // (n+1) x 1, obs
	const arma::vec& Fphi, // L x 1
	const double alpha = 1.) { 
		
	if (trans_code == 1) { // Koyama
		double L_ = static_cast<double>(L);

		if (t <= L) {
			arma::vec Fy = arma::reverse(Y.subvec(0,L-1));
		} else {
			Fy = arma::reverse(Y.subvec(t-L,t-1));
		}
		Fy.elem(arma::find(Fy<=EPS)).fill(0.01/L_);
		Ft = Fphi % Fy;
	}
}


/* Forward Filtering */
void forwardFilter(
	arma::mat& mt, // p x (n+1), t=0 is the mean for initial value theta[0]
	arma::mat& at, // p x (n+1)
	arma::cube& Ct, // p x p x (n+1), t=0 is the var for initial value theta[0]
	arma::cube& Rt, // p x p x (n+1)
	arma::cube& Gt, // p x p x (n+1)
	arma::vec& alphat, // (n+1) x 1
	arma::vec& betat, // (n+1) x 1
	const unsigned int obs_code,
	const unsigned int link_code,
	const unsigned int trans_code,
	const unsigned int gain_code,
	const unsigned int n, // number of observations
	const unsigned int p, // dimension of the state space
	const arma::vec& Y, // (n+1) x 1, the observation (scalar), n: num of obs
	const Rcpp::NumericVector& ctanh = Rcpp::NumericVector::create(0.2,0,5.), // 3 x 1, coefficients for the hyperbolic tangent gain function
	const double alpha = 1.,
	const unsigned int L = 0,
	const double rho = 0.9,
	const double mu0 = 0.,
	const double W = NA_REAL,
	const double delta = NA_REAL,
	const double delta_nb = 1.,
	const bool debug = false) { 
	
	/*
	------ Initialization ------
	*/
	const bool use_discount = !R_IsNA(delta) && R_IsNA(W); // Not prioritize discount factor if it is given.

	arma::vec Ft(p,arma::fill::zeros);
    arma::vec Fphi;
	arma::vec Fy;
	if (trans_code==0 || trans_code==2) { // Koyck or Solow
		Ft.at(1) = 1.;
	} else if (trans_code==3) { // Vanilla
		Ft.at(0) = 1.;
	} else if (trans_code==1 && L>0) { // Koyama
		Fphi = get_Fphi(L);
		Fphi = arma::pow(Fphi,alpha);
		Fy.zeros(L);
    }

	/*
	------ Initialization ------
	*/

	
	double et,ft,Qt,ft_ast,Qt_ast,alpha_ast,beta_ast,phi;
	arma::vec At(p);
    arma::vec hpsi(p);
	arma::vec hph(p);

	/*
	------ Reference Analysis for Koyama's Transfer Kernel ------
	*/
	if (trans_code == 1 && L>0) { // Koyama
        at.col(1) = update_at(p,gain_code,trans_code,mt.col(0),Gt.slice(0),ctanh,alpha,Y.at(0),rho);
		update_Rt(Rt.slice(1), Ct.slice(0), Gt.slice(0), use_discount, W, delta);
		update_Ft(Ft, Fy, trans_code, 0, L, Y, Fphi);
        hpsi = psi2hpsi(at.col(1),gain_code,ctanh);
		hph = hpsi_deriv(at.col(1),gain_code,ctanh);
		ft = arma::accu(Ft % hpsi);
		Ft = Ft % hph;
		Qt = arma::as_scalar(Ft.t() * Rt.slice(1) * Ft);

		/*
		------ Moment Matching ------
		Depends on specific link function
		*/
		phi = mu0 + ft; // regression on link function
		if (link_code == 0 && obs_code == 1) {
			// Poisson DLM with Identity Link
			if (debug) {
				Rcout << "f[" << 1 << "]=" << ft << ", q["<< 1 << "]=" << Qt << std::endl;
			}

			betat.at(1) = phi / Qt;
			alphat.at(1) = phi * betat.at(1);
			Qt_ast = (alphat.at(1)+Y.at(0)) / (betat.at(1)+1.) / (betat.at(1)+1.);
			ft_ast = Qt_ast * (betat.at(1)+1.) - mu0;

			if (debug) {
				Rcout << "alpha[" << 1 << "]=" << alphat.at(1) << ", beta["<< 1 << "]=" << betat.at(1) << std::endl;
				Rcout << "f_ast[" << 1 << "]=" << ft_ast << ", q_ast["<< 1 << "]=" << Qt_ast << std::endl;
			}

		} else if (link_code == 0 && obs_code == 0) {
			// Negative-binomial DLM with Identity Link
			if (debug) {
				Rcout << "f[" << 1 << "]=" << ft << ", q["<< 1 << "]=" << Qt << std::endl;
			}

			// betat.at(1) = phi*(phi+delta_nb)/Qt + 2.;
			// alphat.at(1) = phi/delta_nb * (betat.at(1) - 1.);
			betat.at(1) = std::exp(std::log(phi)+std::log(phi+delta_nb)-std::log(Qt))+2.;
			alphat.at(1) = std::exp(std::log(phi)-std::log(delta_nb)+std::log(betat.at(1)-1.));
			ft_ast = delta_nb*(alphat.at(1)+Y.at(0))/(betat.at(1)+delta_nb-1.) - mu0;
			Qt_ast = (ft_ast+mu0)*delta_nb*(alphat.at(1)+Y.at(0)+betat.at(1)+delta_nb-1.)/(betat.at(1)+delta_nb-1.)/(betat.at(1)+delta_nb-2.);

			if (debug) {
				Rcout << "alpha[" << 1 << "]=" << alphat.at(1) << ", beta["<< 1 << "]=" << betat.at(1) << std::endl;
				Rcout << "f_ast[" << 1 << "]=" << ft_ast << ", q_ast["<< 1 << "]=" << Qt_ast << std::endl;
			}
		} else if (link_code == 1 && obs_code == 1) {
			// Poisson DLM with Exponential Link
			alphat.at(1) = optimize_trigamma(Qt);
			betat.at(1) = std::exp(R::digamma(alphat.at(1)) - mu0 - ft);
        	ft_ast = R::digamma(alphat.at(1)+Y.at(0)) - std::log(betat.at(1)+1.) - mu0;
			Qt_ast = R::trigamma(alphat.at(1)+Y.at(0));

		} else {
			::Rf_error("This combination of likelihood and link function is not supported yet.");
		}

        

		// Posterior at time t: theta[t] | D[t] ~ N(mt, Ct)
		et = ft_ast - ft;
		At = Rt.slice(1) * Ft / Qt; // p x 1
		mt.col(1) = at.col(1) + At * et;
		Ct.slice(1) = Rt.slice(1) - At*(Qt-Qt_ast)*At.t();

		if (L > 1) {
			for (unsigned int t=2; t<=L; t++) {
        		at.col(t) = at.col(1);
        		Rt.slice(t) = Rt.slice(1);
        		mt.col(t) = mt.col(1);
        		Ct.slice(t) = Ct.slice(1);
				alphat.at(t) = alphat.at(1);
				betat.at(t) = betat.at(1);
    		}
		}
		
	}

	
	/*
	------ Reference Analysis for Koyama's Transfer Kernel ------
	*/

	for (unsigned int t=(L+1); t<=n; t++) {
		R_CheckUserInterrupt();

		// Prior at time t: theta[t] | D[t-1] ~ (at, Rt)
		// Linear approximation is implemented if the state equation is nonlinear
		update_Gt(Gt.slice(t),gain_code, trans_code, mt.col(t-1), ctanh, alpha, Y.at(t-1), rho);
        at.col(t) = update_at(p,gain_code,trans_code,mt.col(t-1),Gt.slice(t),ctanh,alpha,Y.at(t-1),rho);
		update_Rt(Rt.slice(t), Ct.slice(t-1), Gt.slice(t), use_discount, W, delta);

		if (t<10 && debug) {
			Rcout << std::endl << "t=" << t << std::endl;
			Rcout << "Gt = " << Gt.slice(t) << std::endl;
			Rcout << "at = " << at.col(t).t() << std::endl;
			Rcout << "Rt = " << Rt.slice(t) << std::endl;
		}
		
		// One-step ahead forecast: Y[t]|D[t-1] ~ N(ft, Qt)
        if (trans_code == 1) { // Koyama
			update_Ft(Ft, Fy, trans_code, t, L, Y, Fphi);
			if (Ft.has_nonfinite()) {
				Rcout << "t = " << t; 
				Ft.brief_print();
				::Rf_error("Nonfinite value in Ft.");
			}
			hpsi = psi2hpsi(at.col(t),gain_code,ctanh);
			if (hpsi.has_nonfinite()) {
				Rcout << "t = " << t << std::endl; 
				Rcout << "at = " << at.col(t) << std::endl;
				hpsi.brief_print();
				::Rf_error("Nonfinite value in hpsi.");
			}
			hph = hpsi_deriv(at.col(t),gain_code,ctanh);
			if (hph.has_nonfinite()) {
				Rcout << "t = " << t; 
				hph.brief_print();
				::Rf_error("Nonfinite value in hph.");
			}
			ft = arma::accu(Ft % hpsi);
			Ft = Ft % hph;
			Qt = arma::as_scalar(Ft.t() * Rt.slice(t) * Ft);
			if (Qt<1.e-32) {
				Rcout << "t = " << t << std::endl; 
				Rcout << "at = " << at.col(t) << std::endl;
				Rcout << "Qt = " << Qt << std::endl;
				::Rf_error("Near zero value in Qt.");
			}
		} else { // Vanilla, Koyck, Solow
			ft = arma::as_scalar(Ft.t() * at.col(t));
			Qt = arma::as_scalar(Ft.t() * Rt.slice(t) * Ft);
		}

		/*
		------ Moment Matching ------
		Depends on specific link function
		*/
		phi = mu0 + ft; // regression on link function
		if (link_code == 0 && obs_code == 1) {
			// Poisson DLM with Identity Link
			if (debug) {
				Rcout << "f[" << t << "]=" << ft << ", q["<< t << "]=" << Qt << std::endl;
			}
			

			betat.at(t) = phi / Qt;
			alphat.at(t) = phi * betat.at(t);
			Qt_ast = (alphat.at(t)+Y.at(t)) / (betat.at(t)+1.) / (betat.at(t)+1.);
			ft_ast = Qt_ast * (betat.at(t)+1.) - mu0;

			if (debug) {
				Rcout << "alpha[" << t << "]=" << alphat.at(t) << ", beta["<< t << "]=" << betat.at(t) << std::endl;
				Rcout << "f_ast[" << t << "]=" << ft_ast << ", q_ast["<< t << "]=" << Qt_ast << std::endl;
			}
			
		} else if (link_code == 0 && obs_code == 0) {
			// Negative-binomial DLM with Identity Link
			if (debug) {
				Rcout << "f[" << t << "]=" << ft << ", q["<< t << "]=" << Qt << std::endl;
			}
			

			// betat.at(t) = (mu0+ft)*(mu0+ft+delta_nb)/Qt + 2.;
			// alphat.at(t) = (mu0+ft)/delta_nb * (betat.at(t) - 1.);
			betat.at(t) = std::exp(std::log(phi)+std::log(phi+delta_nb)-std::log(Qt))+2.;
			alphat.at(t) = std::exp(std::log(phi)-std::log(delta_nb)+std::log(betat.at(t)-1.));
			ft_ast = delta_nb*(alphat.at(t)+Y.at(t))/(betat.at(t)+delta_nb-1.) - mu0;
			Qt_ast = (ft_ast+mu0)*delta_nb*(alphat.at(t)+Y.at(t)+betat.at(t)+delta_nb-1.)/(betat.at(t)+delta_nb-1.)/(betat.at(t)+delta_nb-2.);

			if (debug) {
				Rcout << "alpha[" << t << "]=" << alphat.at(t) << ", beta["<< t << "]=" << betat.at(t) << std::endl;
				Rcout << "f_ast[" << t << "]=" << ft_ast << ", q_ast["<< t << "]=" << Qt_ast << std::endl;
			}

		} else if (link_code == 1 && obs_code == 1) {
			// Poisson DLM with Exponential Link
			alphat.at(t) = optimize_trigamma(Qt);
			betat.at(t) = std::exp(R::digamma(alphat.at(t)) - phi);
        	ft_ast = R::digamma(alphat.at(t)+Y.at(t)) - std::log(betat.at(t)+1.) - mu0;
			Qt_ast = R::trigamma(alphat.at(t)+Y.at(t));

		} else {
			::Rf_error("This combination of likelihood and link function is not supported yet.");
		}
        

		// Posterior at time t: theta[t] | D[t] ~ N(mt, Ct)
		et = ft_ast - ft;
		At = Rt.slice(t) * Ft / Qt; // p x 1
		mt.col(t) = at.col(t) + At * et;
		Ct.slice(t) = Rt.slice(t) - At*(Qt-Qt_ast)*At.t();

		if (debug) {
			Rcout << std::endl;
		}

		// Rcout << "\rFiltering: " << t << "/" << n;
	}
	// Rcout << std::endl;
}



void backwardSmoother(
	arma::vec& ht, // (n+1) x 1
	arma::vec& Ht, // (n+1) x 1
	const unsigned int n,
	const unsigned int p,
	const arma::mat& mt, // p x (n+1), t=0 is the mean for initial value theta[0]
	const arma::mat& at, // p x (n+1)
	const arma::cube& Ct, // p x p x (n+1), t=0 is the var for initial value theta[0]
	const arma::cube& Rt, // p x p x (n+1)
	const arma::cube& Gt,
	const double W = NA_REAL,
	const double delta = 0.9) { // 1: smoothing, 2: sampling (conditional)

	ht.at(n) = mt.at(0,n);
	Ht.at(n) = std::abs(Ct.at(0,0,n));

	double coef1, coef2;
	arma::mat Bt,Ht_prev,Ht_cur;
	arma::vec ht_prev, ht_cur;
	if (!R_IsNA(delta) && R_IsNA(W)) {
		coef1 = 1. - delta;
		coef2 = delta * delta;
	}
	
	if (!R_IsNA(W)) {
		Bt.set_size(p,p);
		Ht_cur.set_size(p,p);
		ht_cur.set_size(p);

		Ht_prev = Ct.slice(n);
		ht_prev = mt.col(n);
	}

	
	for (unsigned int t=(n-1); t>0; t--) {
		R_CheckUserInterrupt();
		if (!R_IsNA(delta) && R_IsNA(W)) {
			ht.at(t) = coef1 * mt.at(0,t) + delta * ht.at(t+1);
			Ht.at(t) = coef1 * Ct.at(0,0,t) + coef2 * Ht.at(t+1);
		} else if (!R_IsNA(W)) {
			try {
				Bt = Ct.slice(t) * Gt.slice(t+1).t() * Rt.slice(t+1).i();
				ht_cur = mt.col(t) + Bt * (ht_prev - at.col(t+1));
				Ht_cur = Ct.slice(t) + Bt*(Ht_prev - Rt.slice(t+1))*Bt.t();

				ht.at(t) = ht_cur.at(0);
				Ht.at(t) = Ht_cur.at(0,0);

				ht_prev = ht_cur;
				Ht_prev = Ht_cur;
			} catch (...) {
				// Rcout << "t=" << t << std::endl;
				// Rcout << "W=" << W << std::endl;
				// Rcout << "R[t+1]=" << Rt.slice(t+1) << std::endl;
				if (!R_IsNA(delta)) {
					ht.at(t) = coef1 * mt.at(0,t) + delta * ht.at(t+1);
					Ht.at(t) = coef1 * Ct.at(0,0,t) + coef2 * Ht.at(t+1);
				} else {
					Bt = Ct.slice(t) * Gt.slice(t+1).t() * arma::pinv(Rt.slice(t+1),1.e-6);
					ht_cur = mt.col(t) + Bt * (ht_prev - at.col(t+1));
					Ht_cur = Ct.slice(t) + Bt*(Ht_prev - Rt.slice(t+1))*Bt.t();

					ht.at(t) = ht_cur.at(0);
					Ht.at(t) = Ht_cur.at(0,0);

					ht_prev = ht_cur;
					Ht_prev = Ht_cur;
				}
			}
			
		}
		
		// Rcout << "\rMarginal Smoothing: " << n+1-t << "/" << n;
	}
	// Rcout << std::endl;

	// t = 0
	if (!R_IsNA(delta) && R_IsNA(W)) {
		ht.at(0) = coef1 * mt.at(0,0) + delta * ht.at(1);
		Ht.at(0) = coef1 * Ct.at(0,0,0) + coef2 * Ht.at(1);
	} else if (!R_IsNA(W)) {
		try {
			Bt = Ct.slice(0) * Gt.slice(1).t() * Rt.slice(1).i();
			ht_cur = mt.col(0) + Bt * (ht_prev - at.col(1));
			Ht_cur = Ct.slice(0) + Bt*(Ht_prev - Rt.slice(1))*Bt.t();

			ht.at(0) = ht_cur.at(0);
			Ht.at(0) = Ht_cur.at(0,0);
		} catch (...) {
			if (!R_IsNA(delta)) {
				ht.at(0) = coef1 * mt.at(0,0) + delta * ht.at(1);
				Ht.at(0) = coef1 * Ct.at(0,0,0) + coef2 * Ht.at(1);
			} else {
				Bt = Ct.slice(0) * Gt.slice(1).t() * arma::pinv(Rt.slice(1),1.e-6);
				ht_cur = mt.col(0) + Bt * (ht_prev - at.col(1));
				Ht_cur = Ct.slice(0) + Bt*(Ht_prev - Rt.slice(1))*Bt.t();

				ht.at(0) = ht_cur.at(0);
				Ht.at(0) = Ht_cur.at(0,0);
			}
			
			// Rcout << "t=" << 0 << std::endl;
			// Rcout << "W=" << W << std::endl;
			// Rcout << "R[t+1]=" << Rt.slice(1) << std::endl;
			// ::Rf_error("Inversion of R[t+1] failed.");
		}
	}
	
}



void backwardSampler(
	arma::vec& theta, // (n+1) x 1
	const unsigned int n,
	const unsigned int p,
	const arma::mat& mt, // p x (n+1), t=0 is the mean for initial value theta[0]
	const arma::mat& at, // p x (n+1)
	const arma::cube& Ct, // p x p x (n+1), t=0 is the var for initial value theta[0]
	const arma::cube& Rt, // p x p x (n+1)
	const arma::cube& Gt,
	const double W = NA_REAL,
	const double scale_sd = arma::datum::eps) {

	arma::mat Bt(p,p);
	arma::mat Ip(p,p,arma::fill::eye); Ip *= scale_sd;

	arma::vec ht = mt.col(n);
	arma::mat Ht = Ct.slice(n);
	try{
		Ht = arma::chol(arma::symmatu(Ht + Ip));
	} catch(...) {
		Rcout << "W=" << W << std::endl;
		::Rf_error("Cholesky decomposition failed");
	}
	
	arma::vec tmp = arma::randn(p,1);
	arma::vec theta_prev = ht + Ht.t()*tmp;
	theta.at(n) = theta_prev.at(0);

	arma::vec theta_cur(p);
	// Rcout << "start" << std::endl;
	
	for (unsigned int t=(n-1); t>0; t--) {
		R_CheckUserInterrupt();
		try {
			Bt = Ct.slice(t) * Gt.slice(t+1).t() * Rt.slice(t+1).i();
		} catch (...) {
			Rcout << "t=" << t << std::endl;
			Rcout << "W=" << W << std::endl;
			Rcout << "R[t+1]=" << Rt.slice(t+1) << std::endl;
			::Rf_error("Inversion of R[t+1] failed.");
		}
		

		// Conditional Sampling
		ht = mt.col(t) + Bt*(theta_prev - at.col(t+1));
		Ht = Ct.slice(t) - Bt * Rt.slice(t+1) * Bt.t();
		try {
			Ht = arma::chol(arma::symmatu(Ht + Ip));
		} catch(...) {
			Rcout << "W=" << W << std::endl;
			::Rf_error("Cholesky decomposition failed");
		}
		
		tmp = arma::randn(p,1);
		theta_cur = ht + Ht.t() * tmp;
		theta.at(t) = theta_cur.at(0);

		// Prep for the next iteration
		theta_prev = theta_cur;

		// Rcout << "\rConditional Sampling: " << n+1-t << "/" << n;
	}
	// Rcout << std::endl;

	// t = 0
	Bt = Ct.slice(0) * Gt.slice(1).t() * Rt.slice(1).i();

	// Conditional Sampling
	ht = mt.col(0) + Bt*(theta_prev - at.col(1));
	Ht = Ct.slice(0) - Bt * Rt.slice(1) * Bt.t();
	try {
		Ht = arma::chol(Ht);
		tmp = arma::randn(p,1);
		theta_cur = ht + scale_sd*Ht.t() * tmp;
		theta.at(0) = theta_cur.at(0);
	} catch (...) {

	}
	

	// Rcout << "Done" << std::endl;
}




//' @export
// [[Rcpp::export]]
Rcpp::List lbe_poisson(
	const arma::vec& Y, // n x 1, the observed response
	const arma::uvec& model_code,
	const double rho = 0.9,
    const unsigned int L = 2, // Number of lags for Koyama or number of trials for Solow
	const double mu0 = 0.,
    const double delta = 0.9,
	const double W = NA_REAL,
	const Rcpp::Nullable<Rcpp::NumericVector>& m0_prior = R_NilValue,
	const Rcpp::Nullable<Rcpp::NumericMatrix>& C0_prior = R_NilValue,
	const Rcpp::NumericVector& ctanh = Rcpp::NumericVector::create(0.2,0,5.),
	const double alpha = 1.,
	const double delta_nb = 10.,
	const double ci_coverage = 0.95,
	const unsigned int npara = 20,
    const bool summarize_return = false,
	const bool debug = false) { 


	if (R_IsNA(delta) && R_IsNA(W)) {
		::Rf_error("Either evolution error W or discount factor delta must be provided.");
	}

	const double critval = R::qnorm(1.-0.5*(1.-ci_coverage),0.,1.,true,false);

	const unsigned int n = Y.n_elem;
	const unsigned int npad = n+1;

	const unsigned int obs_code = model_code.at(0); // Poisson or negative binomial
    const unsigned int link_code = model_code.at(1); // identity or exponential
    const unsigned int trans_code = model_code.at(2); // Koyck, Koyama, or Solow
    const unsigned int gain_code = model_code.at(3); // Exponential, Softplus, logistic, ...
    const unsigned int err_code = model_code.at(4); // normal, laplace, ...

	unsigned int p, L_;
    init_by_trans(p,L_,trans_code,L);

	arma::vec Ypad(npad,arma::fill::zeros); // (n+1) x 1
	Ypad.tail(n) = Y;
	arma::mat mt(p,npad,arma::fill::zeros);
	arma::mat at(p,npad,arma::fill::zeros);
	arma::cube Ct(p,p,npad); 
	arma::mat Ip(p,p,arma::fill::eye);
	Ip *= 1.;
	Ct.each_slice() = Ip;
	arma::cube Rt(p,p,npad);
	arma::vec alphat(npad,arma::fill::zeros);
	arma::vec betat(npad,arma::fill::zeros);
	arma::vec ht(npad,arma::fill::zeros);
	arma::vec Ht(npad,arma::fill::zeros);

	arma::cube Gt(p,p,npad);
	arma::mat Gt0(p,p,arma::fill::zeros);
	Gt0.at(0,0) = 1.;
	switch (trans_code) {
		case 0: // Koyck
		{
			Gt0.at(1,1) = rho;
		}
		break;
		case 1: // Koyama
		{
			Gt0.diag(-1).ones();	
		}
		break;
		case 2: // Solow - Checked. Correct.
		{
			double coef2 = -rho;
			Gt0.at(1,1) = -binom(L,1)*coef2;
			for (unsigned int k=2; k<p; k++) {
				coef2 *= -rho;
				Gt0.at(1,k) = -binom(L,k)*coef2;
				Gt0.at(k,k-1) = 1.;
			}

			// Gt0.at(1,1) = 2.*rho;
			// Gt0.at(1,2) = -rho*rho;
			// Gt0.at(2,1) = 1.;
			// // Rcout << Gt0 << std::endl;
		}
		break;
		case 3: // Vanilla
		{
			Gt0.at(0,0) = rho;	
		}
		break;
		default:
		{
			::Rf_error("Not supported transfer function.\n");
		}
	}

	for (unsigned int t=0; t<npad; t++) {
		Gt.slice(t) = Gt0;
	}
	if (!m0_prior.isNull()) {
		mt.col(0) = Rcpp::as<arma::vec>(m0_prior);
	}
	if (!C0_prior.isNull()) {
		Ct.slice(0) = Rcpp::as<arma::mat>(C0_prior);
	}

    
	forwardFilter(mt,at,Ct,Rt,Gt,alphat,betat,obs_code,link_code,trans_code,gain_code,n,p,Ypad,ctanh,alpha,L_,rho,mu0,W,delta,delta_nb,debug);
	arma::vec Wt = (1.-delta) * arma::vectorise(Rt.tube(0,0));
	backwardSmoother(ht,Ht,n,p,mt,at,Ct,Rt,Gt,W,delta);
	

	Rcpp::List output;
	if (summarize_return) {
		arma::mat psi(npad,3);
		psi.col(0) = ht - critval*arma::sqrt(arma::abs(Ht));
		psi.col(1) = ht; // (n+1) x 1
		psi.col(2) = ht + critval*arma::sqrt(arma::abs(Ht));
		output["psi"] = Rcpp::wrap(psi); // smoothing outcome

		arma::mat psi0(npad,3);
		psi0.col(0) = mt.row(0).t() - critval*arma::vectorise(arma::sqrt(arma::abs(Ct.tube(0,0))));
		psi0.col(1) = mt.row(0).t();
		psi0.col(2) = mt.row(0).t() + critval*arma::vectorise(arma::sqrt(arma::abs(Ct.tube(0,0))));
		output["psi0"] = Rcpp::wrap(psi0);

	} else {
		output["mt"] = Rcpp::wrap(mt.row(0).t()); // mt: p x npad, filtering mean (1st order moment)
		output["Ct"] = Rcpp::wrap(Ct.tube(0,0)); // Ct: p x p x npad, filtering variance (2nd order moment)
		output["ht"] = Rcpp::wrap(ht); // npad x 1, smoothing mean (2nd order moment)
		output["Ht"] = Rcpp::wrap(Ht); // npad x 1, smoothing variance (2nd order moment)
		output["at"] = Rcpp::wrap(at.row(0).t());
		output["Rt"] = Rcpp::wrap(Rt.tube(0,0));
		output["alphat"] = Rcpp::wrap(alphat); // npad x 1
		output["betat"] = Rcpp::wrap(betat); // npad x 1
		output["critval"] = critval;
	}

	output["Wt"] = Rcpp::wrap(Wt); // npad x 1
	arma::vec hpsiR = psi2hpsi(ht,model_code.at(3),ctanh); // hpsi: p x N
	double theta0 = 0;
    arma::vec lambdaR = hpsi2theta(hpsiR, Y, trans_code, theta0, alpha, L, rho); // n x 1
    output["rmse"] = std::sqrt(arma::as_scalar(arma::mean(arma::pow(lambdaR.tail(n-npara) - Y.tail(n-npara),2.0))));
    output["mae"] = arma::as_scalar(arma::mean(arma::abs(lambdaR.tail(n-npara) - Y.tail(n-npara))));

	double logpred = 0.;
	double prob_success;
	for (unsigned int j=npara; j<=n; j++) {
		// Marginal log predictive likelihood
		if (obs_code == 1) {
			// Poisson DLM
			prob_success = std::max(betat.at(j)/(1.+betat.at(j)),EPS);
			logpred += R::dnbinom(Y.at(j-1),std::max(alphat.at(j),EPS),prob_success,true);

		} else if (obs_code == 0 && alphat.at(j)>1.e-32 && betat.at(j)>1.e-32) {
			// Negative binomial DLM
			logpred += R::lgammafn(Y.at(j-1)+delta_nb)-R::lgammafn(delta_nb)-R::lgammafn(Y.at(j-1)+1.);
			logpred += R::lgammafn(std::max(alphat.at(j)+betat.at(j),EPS))-R::lgammafn(std::max(alphat.at(j),EPS))-R::lgammafn(std::max(betat.at(j),EPS));
			logpred += R::lgammafn(alphat.at(j)+Y.at(j-1))+R::lgammafn(std::max(betat.at(j)+delta_nb,EPS))-R::lgammafn(alphat.at(j)+Y.at(j-1)+betat.at(j)+delta_nb);
		}
	}
	output["logpred"] = logpred;

	return output;
}


//' @export
// [[Rcpp::export]]
Rcpp::List get_eta_koyama(
	const arma::vec& Y, // n x 1, the observation (scalar), n: num of obs
	const arma::mat& mt, // p x (n+1), t=0 is the mean for initial value theta[0]
	const arma::cube& Ct, // p x p x (n+1)
	const unsigned int gain_code,
	const double alpha = 1.,
	const double mu0 = 0.) {
	
	const unsigned int n = Y.n_elem;
	const unsigned int p = mt.n_rows;
	const unsigned int trans_code = 1; // Koyama

	arma::vec ft(n+1);
	arma::vec Qt(n+1,arma::fill::zeros);

	arma::vec Ft(p,arma::fill::zeros);
	arma::vec Fy(p,arma::fill::zeros);
	arma::vec Fphi = get_Fphi(p);

	arma::mat mt_ramp;
	if (gain_code == 0) {
		mt_ramp = mt;
		mt_ramp.elem(arma::find(mt_ramp<EPS)).zeros(); // Ramp function
	}

	for (unsigned int t=0; t<=n; t++) {
		update_Ft(Ft, Fy, trans_code, t, p, Y, Fphi, alpha);
		switch (gain_code) {
			case 0: // KoyamaMax
			{
				ft.at(t) = mu0 + arma::accu(Ft % mt_ramp.col(t));
			}
			break;
			case 1: // KoyamaExp
			{
				ft.at(t) = mu0 + arma::accu(Ft % arma::exp(mt.col(t)));
				Ft = Ft % arma::exp(mt.col(t));
				Qt.at(t) = arma::as_scalar(Ft.t() * Ct.slice(t) * Ft);
			}
			break;
			case 2: // KoyamaEye
			{
				ft.at(t) = mu0 + arma::accu(Ft % mt.col(t));
				Qt.at(t) = arma::as_scalar(Ft.t() * Ct.slice(t) * Ft);
			}
			break;
			default:
			{
				::Rf_error("get_Ft function is only defined for Koyama transmission kernels.");
			}
		} // END switch block
	}

	Rcpp::List output;
	output["mean"] = Rcpp::wrap(ft);
	output["var"] = Rcpp::wrap(Qt);
	return output;
}



//' @export
// [[Rcpp::export]]
Rcpp::List get_optimal_delta(
	const arma::vec& Y, // n x 1
	const arma::uvec& model_code,
	const arma::vec& delta_grid, // m x 1
	const double rho = 0.9,
    const unsigned int L = 0,
	const double mu0 = 0.,
	const Rcpp::Nullable<Rcpp::NumericVector>& m0_prior = R_NilValue,
	const Rcpp::Nullable<Rcpp::NumericMatrix>& C0_prior = R_NilValue,
	const Rcpp::NumericVector& ctanh = Rcpp::NumericVector::create(0.2,0,5),
	const double alpha = 1.,
	const double delta_nb = 1.,
	const unsigned int npara = 1,
	const double ci_coverage = 0.95) {

	const unsigned int n = Y.n_elem;
	const unsigned int m = delta_grid.n_elem;
	const double W = NA_REAL;
	const double n_ = static_cast<double>(n-npara+1);

	const unsigned int obs_code = model_code.at(0);

	arma::vec logpred(m,arma::fill::zeros);
	arma::vec mse(m,arma::fill::zeros);
	arma::vec mae(m,arma::fill::zeros);
	double delta;
	double ymean,prob_success;
	for (unsigned int i=0; i<m; i++) {
		delta = delta_grid.at(i);
		Rcpp::List lbe = lbe_poisson(Y,model_code,rho,L,mu0,delta,W,m0_prior,C0_prior,ctanh,alpha,delta_nb,ci_coverage,false,false);
		arma::vec alphat = lbe["alphat"];
		arma::vec betat = lbe["betat"];
		for (unsigned int j=npara; j<=n; j++) {
			// Marginal log predictive likelihood
			if (obs_code == 1) {
				// Poisson DLM
				prob_success = betat.at(j)/(1.+betat.at(j));
				logpred.at(i) += R::dnbinom(Y.at(j-1),alphat.at(j),prob_success,true);

				// MSE
				ymean = (1.-prob_success)/prob_success*alphat.at(j);
				mse.at(i) += (Y.at(j-1)-ymean)*(Y.at(j-1)-ymean);

				// MAE
				mae.at(i) = std::abs(Y.at(j-1) - ymean);

			} else if (obs_code == 0 && alphat.at(j)>1.e-32 && betat.at(j)>1.e-32) {
				// Negative binomial DLM
				logpred.at(i) += R::lgammafn(Y.at(j-1)+delta_nb)-R::lgammafn(delta_nb)-R::lgammafn(Y.at(j-1)+1.);
				logpred.at(i) += R::lgammafn(alphat.at(j)+betat.at(j))-R::lgammafn(alphat.at(j))-R::lgammafn(betat.at(j));
				logpred.at(i) += R::lgammafn(alphat.at(j)+Y.at(j-1))+R::lgammafn(betat.at(j)+delta_nb)-R::lgammafn(alphat.at(j)+Y.at(j-1)+betat.at(j)+delta_nb);
			}
		}
	}

	mse /= n_;
	mae /= n_;

	Rcpp::List output;
	output["delta"] = Rcpp::wrap(delta_grid);
	output["logpred"] = Rcpp::wrap(logpred);

	output["mse"] = Rcpp::wrap(mse);
	output["mae"] = Rcpp::wrap(mae);

	arma::vec delta_optim = {delta_grid.at(logpred.index_max()),delta_grid.at(mse.index_min()),delta_grid.at(mae.index_min())};

	output["delta_optim"] = Rcpp::wrap(delta_optim);

	return output;
}



//' @export
// [[Rcpp::export]]
double lbe_What(
    const arma::vec& ht, // (n+1) x 1
    const unsigned int nsample = 2000,
    const double aw = 0.1,
    const double bw = 0.1) {
    
    double n_ = static_cast<double>(ht.n_elem) - 1.;
    double aw_new = aw + 0.5*n_;
    double bw_new = bw + 0.5* arma::accu(arma::pow(arma::diff(ht),2));
    double What = bw_new / (aw_new - 1.);
    return What;
}