#include "lbe_poisson.h"
using namespace Rcpp;
// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::depends(RcppArmadillo,nloptr)]]


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


/*
---------------------------------------
------ Desiderata of lbe_poisson ------
---------------------------------------

- Accommodate (1) known and fixed evolution variance $W$; (2) use of discount factor $\delta$ by assuming a time-varying evolution variance $W_t$.
- Different dimension of the state space, aka, dimension of $\widetilde{\boldsymbol{\theta}}_t$.
- Function that maps state equations to DLM form for different models.
- Accommodate different link functions, different gain functions, different transfer functions.
*/



arma::vec update_at(
	const unsigned int p,
	const unsigned int ModelCode,
	const unsigned int TransferCode, // 0 - Koyck, 1 - Koyama, 2 - Solow
	const arma::vec& mt, // p x 1, mt = (psi[t], theta[t], theta[t-1])
	const arma::mat& Gt, // p x p
	const double y = NA_REAL,  // n x 1
	const double rho = NA_REAL,
	const unsigned int L = 1) {
	
	arma::vec at(p,arma::fill::zeros);
	const double UPBND = 700.;
	
	if (TransferCode == 2) { // Solow
		/*------ START Solow ------*/
		at.at(0) = mt.at(0); // psi[t]
		at.at(2) = mt.at(1); // theta[t-1]
		const double coef1 = (1.-rho)*(1.-rho);
		switch (ModelCode) { // theta[t]
			case 2: // SolowMax
			{
				at.at(1) = coef1*y*std::max(mt.at(0),arma::datum::eps);
			}
			break;
			case 3: // SolowEXP
			{
				// at.at(1) = coef1*y*std::exp(mt.at(0));
				at.at(1) = coef1*y*std::exp(std::min(mt.at(0),UPBND));
			}
			break;
			case 7: // SolowEye
			{
				at.at(1) = coef1*y*mt.at(0);
			}
			break;
			case 12: // SolowSoftplus
			{
				at.at(1) = coef1*y*std::log(1.+std::exp(std::min(mt.at(0),UPBND)));
			}
			break;
			default:
			{
				::Rf_error("at - This block only support SolowMax(2), SolowExp(3), SolowEye(7)");
			}
		} // END switch block
		at.at(1) += 2*rho*mt.at(1) - rho*rho*mt.at(2);
		/*------ END Solow ------*/

	} else if (TransferCode == 0) { // Koyck
		/*------ START Koyck ------*/
		at.at(0) = mt.at(0); // psi[t]
		switch (ModelCode) { // theta[t]
			case 4: // KoyckMax
			{
				at.at(1) = y * std::max(mt.at(0),arma::datum::eps); 
			}
			break;
			case 5: // KoyckExp
			{
				// at.at(1) = y*std::exp(mt.at(0));
				at.at(1) = y * std::exp(std::min(mt.at(0),UPBND));
			}
			break;
			case 8: // KoyckEye
			{
				at.at(1) = y * mt.at(0);
			}
			break;
			case 10: // KoyckSoftplus
			{
				// at.at(1) = y * std::log(1. + std::exp(mt.at(0)));
				at.at(1) = y * std::log(1. + std::exp(std::min(mt.at(0),UPBND)));
			}
			break;
			default:
			{
				::Rf_error("at - This block only support KoyckMax(4), KoyckExp(5), KoyckEye(8)");
			}
		} // END switch block
		at.at(1) += rho*mt.at(1);
		/*------ END Koyck ------*/

	} else if (TransferCode==1 || TransferCode==3) { // Koyama or Vanilla
		/*------ START Koyama ------*/
		at = Gt * mt;
		/*------ END Koyama ------*/

	} else {
		::Rf_error("get_at function is only defined for Vanilla, Koyama, Solow, or Koyck's transmission kernels.");
	}

	return at;
}



void update_Gt(
	arma::mat& Gt, // p x p
	const unsigned int ModelCode, 
	const unsigned int TransferCode, // 0 - Koyck, 1 - Koyama, 2 - Solow
	const arma::vec& mt, // p x 1
	const double y = NA_REAL,  // obs
	const double rho = NA_REAL) {

	const double UPBND = 700.;
	
	if (TransferCode == 2) { // Solow
		switch (ModelCode) {
			case 2: // SolowMax
			{
				// Gt.at(1,0) = std::max(mt.at(0),arma::datum::eps);
				::Rf_error("How are we gonna do Taylor Expansion on the ramp function?");
			}
			break;
			case 3: // SolowExp
			{
				// Gt.at(1,0) = std::exp(mt.at(0));
				Gt.at(1,0) = std::exp(std::min(mt.at(0),UPBND));
			}
			break;
			case 7: // SolowEye
			{
				Gt.at(1,0) = 1.;
			}
			break;
			case 12: // SolowSoftplus
			{
				Gt.at(1,0) = 1. / (1. + std::exp(std::min(-mt.at(0), UPBND)));
			}
			break;
			default:
			{
				::Rf_error("This block only supports Solow");
			}
		} // End switch block
		Gt.at(1,0) *= (1.-rho)*(1.-rho)*y;

	} else if (TransferCode == 0) { // Koyck
		switch (ModelCode) {
			case 4: // KoyckMax
			{
				::Rf_error("How are we gonna do Taylor Expansion on the ramp function?");
			}
			break;
			case 5: // KoyckExp
			{
				// Gt.at(1,0) = std::exp(mt.at(0));
				Gt.at(1,0) = std::exp(std::min(mt.at(0),UPBND));
			}
			break;
			case 8: // KoyckEye
			{
				Gt.at(1,0) = 1.;
			}
			break;
			case 10: // KoyckSoftplus
			{
				// Gt.at(1,0) = 1. / (1. + std::exp(-mt.at(0)));
				Gt.at(1,0) = 1. / (1. + std::exp(std::min(-mt.at(0), UPBND)));
			}
			break;
			default:
			{
				::Rf_error("This block only supports Koyck");
			}
		} // End switch block

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

	Rt = Gt * Ct * Gt.t();
	if (use_discount) {
		Rt.for_each( [&delta](arma::mat::elem_type& val) { val /= delta; } );
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
	const unsigned int TransferCode, // 0 - Koyck, 1 - Koyama, 2 - Solow
	const unsigned int t, // current time point
	const unsigned int L, // lag
	const arma::vec& Y,  // (n+1) x 1, obs
	const arma::vec& Fphi) { // L x 1
		
	if (TransferCode == 1) { // Koyama
		double L_ = static_cast<double>(L);

		if (t <= L) {
			arma::vec Fy = arma::reverse(Y.subvec(0,L-1));
		} else {
			Fy = arma::reverse(Y.subvec(t-L,t-1));
		}
		Fy.elem(arma::find(Fy<=arma::datum::eps)).fill(0.01/L_);
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
	const unsigned int ModelCode,
	const unsigned int TransferCode,
	const unsigned int n, // number of observations
	const unsigned int p, // dimension of the state space
	const arma::vec& Y, // (n+1) x 1, the observation (scalar), n: num of obs
	const unsigned int L = 0,
	const double rho = 0.9,
	const double mu0 = 0.,
	const double W = NA_REAL,
	const double delta = NA_REAL,
	const double delta_nb = 1.,
	const unsigned int obs_type = 1, // 0: negative binomial DLM; 1: poisson DLM 
	const bool debug = false) { 

	const double UPBND = 700.;
	
	/*
	------ Initialization ------
	*/
	unsigned int LinkCode = 1; // 0 - exponential; 1 - identity
	if (ModelCode==6 || ModelCode==7 || ModelCode==8 || ModelCode==9) {LinkCode = 0;}
	const bool use_discount = !R_IsNA(delta) && R_IsNA(W); // Not prioritize discount factor if it is given.


	arma::vec Ft(p,arma::fill::zeros);
	if (TransferCode==0 || TransferCode==2) { // Koyck or Solow
		Ft.at(1) = 1.;
	} else if (TransferCode==3) { // Vanilla
		Ft.at(0) = 1.;
	}

	arma::vec Fphi;
	arma::vec Fy;
	if (TransferCode == 1 && L>0) { // koyama
		const double mu = 2.2204e-16;
		const double m = 4.7;
		const double s = 2.9;
		Fphi = get_Fphi(L,mu,m,s);
		Fy.zeros(L);
	}
	/*
	------ Initialization ------
	*/

	
	double et,ft,Qt,ft_ast,Qt_ast,alpha_ast,beta_ast,phi;
	arma::vec At(p);


	/*
	------ Reference Analysis for Koyama's Transfer Kernel ------
	*/
	if (TransferCode == 1 && L>0) { // Koyama
		at.col(1) = update_at(p,ModelCode,TransferCode,mt.col(0),Gt.slice(0),Y.at(0),rho,L);
		update_Rt(Rt.slice(1), Ct.slice(0), Gt.slice(0), use_discount, W, delta);
		update_Ft(Ft, Fy, TransferCode, 0, L, Y, Fphi);
		switch (ModelCode) {
			case 0: // KoyamaMax
			{
				::Rf_error("How are we gonna do Taylor Expansion on the ramp function?");
			}
			break;
			case 1: // KoyamaExp
			{
				at.elem(arma::find(at>UPBND)).fill(UPBND);
				Ft = Ft % arma::exp(at.col(1));
				ft = arma::accu(Ft);
				Qt = arma::as_scalar(Ft.t() * Rt.slice(1) * Ft);
			}
			break;
			case 6: // KoyamaEye
			{
				ft = arma::accu(Ft % at.col(1));
				Qt = arma::as_scalar(Ft.t() * Rt.slice(1) * Ft);
			}
			break;
			case 11: // KoyamaSoftplus
			{
				at.elem(arma::find(at>UPBND)).fill(UPBND);
				arma::vec at_exp = arma::exp(at.col(1));
				ft = arma::accu(Ft % arma::log(1. + at_exp));

				Ft = Ft % at_exp / (1. + at_exp);
				Qt = arma::as_scalar(Ft.t() * Rt.slice(1) * Ft);
			}
			break;
			default:
			{
				::Rf_error("get_Ft function is only defined for Koyama transmission kernels.");
			}
		} // END switch block

		/*
		------ Moment Matching ------
		Depends on specific link function
		*/
		phi = mu0 + ft; // regression on link function
		if (LinkCode == 1 && obs_type == 1) {
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

		} else if (LinkCode == 1 && obs_type == 0) {
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
		} else if (LinkCode == 0 && obs_type == 1) {
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
		update_Gt(Gt.slice(t), ModelCode, TransferCode, mt.col(t-1), Y.at(t-1), rho);
		at.col(t) = update_at(p,ModelCode,TransferCode,mt.col(t-1),Gt.slice(t),Y.at(t-1),rho,L);
		update_Rt(Rt.slice(t), Ct.slice(t-1), Gt.slice(t), use_discount, W, delta);

		if (t<10 && debug) {
			Rcout << std::endl << "t=" << t << std::endl;
			Rcout << "Gt = " << Gt.slice(t) << std::endl;
			Rcout << "at = " << at.col(t).t() << std::endl;
			Rcout << "Rt = " << Rt.slice(t) << std::endl;
		}
		
		// One-step ahead forecast: Y[t]|D[t-1] ~ N(ft, Qt)
		if (TransferCode == 1) { // Koyama
			update_Ft(Ft, Fy, TransferCode, t, L, Y, Fphi);
			switch (ModelCode) {
				case 0: // KoyamaMax
				{
					::Rf_error("How are we gonna do Taylor Expansion on the ramp function?");
				}
				break;
				case 1: // KoyamaExp
				{
					at.elem(arma::find(at>UPBND)).fill(UPBND);
					Ft = Ft % arma::exp(at.col(t));
					ft = arma::accu(Ft);
					Qt = arma::as_scalar(Ft.t() * Rt.slice(t) * Ft);
				}
				break;
				case 6: // KoyamaEye
				{
					ft = arma::accu(Ft % at.col(t));
					Qt = arma::as_scalar(Ft.t() * Rt.slice(t) * Ft);
				}
				break;
				case 11: // KoyamaSoftplus
				{
					at.elem(arma::find(at>UPBND)).fill(UPBND);
					arma::vec at_exp = arma::exp(at.col(t));
					ft = arma::accu(Ft % arma::log(1. + at_exp));

					Ft = Ft % at_exp / (1. + at_exp);
					Qt = arma::as_scalar(Ft.t() * Rt.slice(t) * Ft);
				}
				break;
				default:
				{
					::Rf_error("get_Ft function is only defined for Koyama transmission kernels.");
				}
			} // END switch block
		} else { // Vanilla, Koyck, Solow
			ft = arma::as_scalar(Ft.t() * at.col(t));
			Qt = arma::as_scalar(Ft.t() * Rt.slice(t) * Ft);
		}

		/*
		------ Moment Matching ------
		Depends on specific link function
		*/
		phi = mu0 + ft; // regression on link function
		if (LinkCode == 1 && obs_type == 1) {
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
			
		} else if (LinkCode == 1 && obs_type == 0) {
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

		} else if (LinkCode == 0 && obs_type == 1) {
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
	const unsigned int ModelCode,
	const double rho = 0.9,
    const unsigned int L = 0,
	const double mu0 = 0.,
    const double delta = 0.9, 
	const double W = NA_REAL,
	const Rcpp::Nullable<Rcpp::NumericVector>& m0_prior = R_NilValue,
	const Rcpp::Nullable<Rcpp::NumericMatrix>& C0_prior = R_NilValue,
	const double delta_nb = 1.,
	const unsigned int obs_type = 1, // 0: negative binomial DLM; 1: poisson DLM
	const bool debug = false) { 

	if (R_IsNA(delta) && R_IsNA(W)) {
		::Rf_error("Either evolution error W or discount factor delta must be provided.");
	}

	const unsigned int n = Y.n_elem;
	const unsigned int npad = n+1;

	const bool is_solow = ModelCode == 2 || ModelCode == 3 || ModelCode == 7 || ModelCode == 12;
	const bool is_koyck = ModelCode == 4 || ModelCode == 5 || ModelCode == 8 || ModelCode == 10;
	const bool is_koyama = ModelCode == 0 || ModelCode == 1 || ModelCode == 6 || ModelCode == 11;
	const bool is_vanilla = ModelCode == 9;
	unsigned int TransferCode; // integer indicator for the type of transfer function
	unsigned int p; // dimension of DLM state space
	unsigned int L_;
	if (is_koyck) { 
		TransferCode = 0; 
		p = 2;
		L_ = 0;
	} else if (is_koyama) { 
		TransferCode = 1; 
		p = L;
		L_ = L;
	} else if (is_solow) { 
		TransferCode = 2; 
		p = 3;
		L_ = 0;
	} else if (is_vanilla) {
		TransferCode = 3;
		p = 1;
		L_ = 0;
	} else {
		::Rf_error("Unknown type of model.");
	}

	arma::vec Ypad(npad,arma::fill::zeros); // (n+1) x 1
	Ypad.tail(n) = Y;
	arma::mat mt(p,npad,arma::fill::zeros);
	arma::mat at(p,npad,arma::fill::zeros);
	arma::cube Ct(p,p,npad); 
	const arma::mat Ip(p,p,arma::fill::eye);
	Ct.each_slice() = Ip;
	arma::cube Rt(p,p,npad);
	arma::vec alphat(npad,arma::fill::zeros);
	arma::vec betat(npad,arma::fill::zeros);
	arma::vec ht(npad,arma::fill::zeros);
	arma::vec Ht(npad,arma::fill::zeros);

	arma::cube Gt(p,p,npad);
	arma::mat Gt0(p,p,arma::fill::zeros);
	Gt0.at(0,0) = 1.;
	if (TransferCode == 0) { // Koyck
		Gt0.at(1,1) = rho;
	} else if (TransferCode == 1) { // Koyama
		Gt0.diag(-1).ones();
	} else if (TransferCode == 2) { // Solow
		Gt0.at(1,1) = 2.*rho;
		Gt0.at(1,2) = -rho*rho;
		Gt0.at(2,1) = 1.;
	} else if (TransferCode == 3) { // Vanilla
		Gt0.at(0,0) = rho;
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
    
	forwardFilter(mt,at,Ct,Rt,Gt,alphat,betat,ModelCode,TransferCode,n,p,Ypad,L_,rho,mu0,W,delta,delta_nb,obs_type,debug);
	backwardSmoother(ht,Ht,n,p,mt,at,Ct,Rt,Gt,W,delta);
	

	Rcpp::List output;
	output["mt"] = Rcpp::wrap(mt);
	output["Ct"] = Rcpp::wrap(Ct);
	output["ht"] = Rcpp::wrap(ht);
	output["Ht"] = Rcpp::wrap(Ht);
	output["alphat"] = Rcpp::wrap(alphat);
	output["betat"] = Rcpp::wrap(betat);

	return output;
}


//' @export
// [[Rcpp::export]]
Rcpp::List get_eta_koyama(
	const arma::vec& Y, // n x 1, the observation (scalar), n: num of obs
	const arma::mat& mt, // p x (n+1), t=0 is the mean for initial value theta[0]
	const arma::cube& Ct, // p x p x (n+1)
	const unsigned int ModelCode,
	const double mu0 = 0.) {
	
	const unsigned int n = Y.n_elem;
	const unsigned int p = mt.n_rows;
	const unsigned int TransferCode = 1; // Koyama

	arma::vec ft(n+1);
	arma::vec Qt(n+1,arma::fill::zeros);

	arma::vec Ft(p,arma::fill::zeros);
	arma::vec Fy(p,arma::fill::zeros);
	const double mu = 2.2204e-16;
	const double m = 4.7;
	const double s = 2.9;
	arma::vec Fphi = get_Fphi(p,mu,m,s);

	arma::mat mt_ramp;
	if (ModelCode == 0) {
		mt_ramp = mt;
		mt_ramp.elem(arma::find(mt_ramp<arma::datum::eps)).zeros(); // Ramp function
	}

	for (unsigned int t=0; t<=n; t++) {
		update_Ft(Ft, Fy, TransferCode, t, p, Y, Fphi);
		switch (ModelCode) {
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
			case 6: // KoyamaEye
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
	const unsigned int ModelCode,
	const arma::vec& delta_grid, // m x 1
	const double rho = 0.9,
    const unsigned int L = 0,
	const double mu0 = 0.,
	const Rcpp::Nullable<Rcpp::NumericVector>& m0_prior = R_NilValue,
	const Rcpp::Nullable<Rcpp::NumericMatrix>& C0_prior = R_NilValue,
	const double delta_nb = 1.,
	const unsigned int obs_type = 1) { // 0: negative binomial DLM; 1: poisson DLM

	const unsigned int n = Y.n_elem;
	const unsigned int m = delta_grid.n_elem;
	const double W = NA_REAL;
	const double n_ = static_cast<double>(n);

	arma::vec logpred(m,arma::fill::zeros);
	arma::vec mse(m,arma::fill::zeros);
	arma::vec mae(m,arma::fill::zeros);
	double delta;
	double ymean,prob_success;
	for (unsigned int i=0; i<m; i++) {
		delta = delta_grid.at(i);
		Rcpp::List lbe = lbe_poisson(Y,ModelCode,rho,L,mu0,delta,W,m0_prior,C0_prior,delta_nb,obs_type,false);
		arma::vec alphat = lbe["alphat"];
		arma::vec betat = lbe["betat"];
		for (unsigned int j=1; j<=n; j++) {
			// Marginal log predictive likelihood
			if (obs_type == 1) {
				// Poisson DLM
				prob_success = betat.at(j)/(1.+betat.at(j));
				logpred.at(i) += R::dnbinom(Y.at(j-1),alphat.at(j),prob_success,true);

				// MSE
				ymean = (1.-prob_success)/prob_success*alphat.at(j);
				mse.at(i) += (Y.at(j-1)-ymean)*(Y.at(j-1)-ymean);

				// MAE
				mae.at(i) = std::abs(Y.at(j-1) - ymean);

			} else if (obs_type == 0) {
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