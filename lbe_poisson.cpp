#include "lbe_poisson.h"
using namespace Rcpp;
// [[Rcpp::plugins(cpp11)]]
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

arma::mat update_at(
	arma::mat &Gt,			  // p x p
	const arma::mat &mt_prev, // p x N, m[t-1] = (psi[t-1], theta[t-1], theta[t-2])
	const double &yprev,	  // y[t-1]
	const arma::vec &lag_par,
	const unsigned int &gain_code,
	const unsigned int &nlag,
	const bool &truncated)
{
	const unsigned int N = mt_prev.n_cols;
	arma::mat at_cur(Gt.n_rows, N, arma::fill::zeros);
	

	if (!truncated)
	{
		// no truncation - only work for negative binomial for now.
		arma::vec hpsi_cur(N, arma::fill::zeros);
		// arma::mat Gtmp(Gt.begin(),Gt.n_elem);

		arma::vec psi_cur = arma::vectorise(mt_prev.row(0));
		hpsi_cur = psi2hpsi(psi_cur, gain_code); // 1 x N

		double tmp1 = std::pow(1. - lag_par.at(0), lag_par.at(1));
		tmp1 *= yprev;


		for (unsigned int i = 0; i < N; i++)
		{

			Gt.at(1, 0) = hpsi_cur.at(i) * tmp1;

			arma::vec old_vec = mt_prev.col(i);
			double old_psi = old_vec.at(0); // psi[t-1]
			old_vec.at(0) = 1.;

			arma::vec new_vec = Gt * old_vec;
			new_vec.at(0) = old_psi;

			double aval = new_vec.at(1);
			// double aval = theta_new_solow(old_vec.tail(r),hpsi.at(i),y,tidx,rho,L);
			new_vec.at(1) = std::max(EPS, aval);

			at_cur.col(i) = new_vec;
		}
	} 
	else
	{
		// truncated to nlag
		at_cur = Gt * mt_prev;
	}

	bound_check(at_cur,"update_at: at");
	return at_cur;
}



/**
 * Matrix Gt is the derivative of gt(.)
 * It is only updated when there is no truncation and using negative-binomial distributed lags.
 */
void update_Gt(
	arma::mat &Gt, // p x p
	const arma::vec &mt, // p x 1
	const double &y,	 // obs
	const arma::vec &lag_par,
	const unsigned int &gain_code,
	const unsigned int &trans_code, // 0 - Koyck, 1 - Koyama, 2 - Solow
	const unsigned int &nlag,
	const bool &truncated)
{
	if (!truncated && (trans_code == 2))
	{ // Solow - Checked. Correct.
		Gt.at(1,0) = hpsi_deriv(mt.at(0),gain_code);
		Gt.at(1,0) *= std::pow(1.-lag_par.at(0),lag_par.at(1)) * y;
		// Gt.at(1,0) *= std::pow((1.-rho)*(1.-rho),alpha)*y;
	}
	else if (!truncated && (trans_code == 0))
	{ // Koyck
		Gt.at(1,0) = hpsi_deriv(mt.at(0),gain_code);
		Gt.at(1,0) *= y;
	}
}

void update_Rt(
	arma::mat &Rt,			   // p x p
	const arma::mat &Ct,	   // p x p
	const arma::mat &Gt,	   // p x p
	const bool &use_discount,  // true if !R_IsNA(delta)
	const double &W, // known evolution error of psi
	const double &delta)
{ // discount factor

	const unsigned int p = Rt.n_cols;
	Rt = Gt * Ct * Gt.t();
	if (use_discount) {
		// We should not use component discounting because Gt is not block-diagonal.
		Rt /= delta; // single discount factor
	} else {
		Rt.at(0,0) += W;
	}

	bound_check(Rt,"update_Rt: Rt",true,false);
}

/*
Checked. OK.
------------ update_Ft ------------
- only for truncated distributed lags.
- only contain the phi[l]*y[t-l] part.
- doesn't include the gain function.
*/
void update_Ft_truncated(
	arma::vec &Ft,		   // L x 1
	arma::vec &Fy,		   // L x 1
	const int &t, // current time point, t = 1, ..., nobs
	const arma::vec &Y,	   // (n+1) x 1, obs
	const arma::vec &Fphi,
	const int &nlag,
	const bool &truncated)
{
	if (!truncated)
	{
		// no truncation
		throw std::invalid_argument("update_Ft_truncated: Ft is only updated using truncated distributed lags.");
	}

	double L_ = static_cast<double>(nlag);
	int nstart = std::max(0, t-nlag);
	unsigned int nend = std::max(nlag - 1, t - 1);
	unsigned int nelem = nend - nstart + 1;

	Fy = arma::reverse(Y.subvec((unsigned int) nstart, nend));
	Fy.elem(arma::find(Fy<=EPS)).fill(0.01/L_);
	Ft = Fphi.head(nelem) % Fy;
	return;
}

/* Forward Filtering */
void forwardFilter(
	arma::mat &mt,			// p x (n+1), t=0 is the mean for initial value theta[0]
	arma::mat &at,			// p x (n+1)
	arma::cube &Ct,			// p x p x (n+1), t=0 is the var for initial value theta[0]
	arma::cube &Rt,			// p x p x (n+1)
	arma::cube &Gt,			// p x p x (n+1)
	arma::vec &alphat,		// (n+1) x 1
	arma::vec &betat,		// (n+1) x 1
	const arma::vec &Y,		// (n+1) x 1, the observation (scalar), n: num of obs
	const unsigned int &nt, // number of observations
	const unsigned int &p,	// dimension of the state space
	const arma::vec &obs_par,
	const arma::vec &lag_par,
	const double &W,
	const unsigned int &obs_code,
	const unsigned int &link_code,
	const unsigned int &trans_code,
	const unsigned int &gain_code,
	const unsigned int &nlag,
	const double &delta_discount,
	const bool &truncated,
	const bool &use_discount)
{
	double mu0 = obs_par.at(0);
	double delta_nb = obs_par.at(1);
	/*
	------ Initialization ------
	*/
	arma::vec Ft(p,arma::fill::zeros);
	if (trans_code == 0 || trans_code == 2)
	{ // Koyck or Solow
		Ft.at(1) = 1.;
	}
	else if (trans_code == 3)
	{ // Vanilla
		Ft.at(0) = 1.;
	}

	arma::vec Fphi = get_Fphi(nlag, lag_par, trans_code);
	arma::vec Fy(nlag, arma::fill::zeros);

	
	/*
	------ Initialization ------
	*/

	
	double et,ft,Qt,ft_ast,Qt_ast,alpha_ast,beta_ast,phi;
	arma::vec At(p);
    arma::vec hpsi(p);
	arma::vec hph(p);

	/*
	------ Reference Analysis ------
	*/
	if ( truncated )
	{
		at.col(1) = update_at(
			Gt.slice(1), mt.col(0), Y.at(0),
			lag_par, gain_code, nlag, truncated); // p x 1

		update_Gt(
			Gt.slice(1), mt.col(0), Y.at(0), 
			lag_par, gain_code, trans_code, 
			nlag, truncated);
		update_Rt(Rt.slice(1), Ct.slice(0), Gt.slice(1), use_discount, W, delta_discount);

		update_Ft_truncated(Ft, Fy, 1, Y, Fphi, nlag, truncated);

		hpsi = psi2hpsi(at.col(1), gain_code);
		hph = hpsi_deriv(at.col(1), gain_code);

		ft = arma::accu(Ft % hpsi);
		Ft = Ft % hph;
		Qt = arma::as_scalar(Ft.t() * Rt.slice(1) * Ft);

		bound_check(Qt, "forwardFilter: Qt", true, false);

		/*
		------ Moment Matching ------
		Depends on specific link function
		*/
		phi = mu0 + ft; // regression on link function
		if (link_code == 0 && obs_code == 1) {
			// Poisson DLM with Identity Link
			betat.at(1) = phi / Qt;
			alphat.at(1) = phi * betat.at(1);
			Qt_ast = (alphat.at(1) + Y.at(0)) / (betat.at(1) + 1.) / (betat.at(1) + 1.);
			ft_ast = Qt_ast * (betat.at(1)+1.) - mu0;

		} else if (link_code == 0 && obs_code == 0) {
			// Negative-binomial DLM with Identity Link
			betat.at(1) = std::exp(std::log(phi)+std::log(phi+delta_nb)-std::log(Qt))+2.;
			alphat.at(1) = std::exp(std::log(phi)-std::log(delta_nb)+std::log(betat.at(1)-1.));
			ft_ast = delta_nb * (alphat.at(1) + Y.at(0)) / (betat.at(1) + delta_nb - 1.) - mu0;
			Qt_ast = (ft_ast + mu0) * delta_nb * (alphat.at(1) + Y.at(0) + betat.at(1) + delta_nb - 1.) / (betat.at(1) + delta_nb - 1.) / (betat.at(1) + delta_nb - 2.);
		} else if (link_code == 1 && obs_code == 1) {
			// Poisson DLM with Exponential Link
			alphat.at(1) = optimize_trigamma(Qt);
			betat.at(1) = std::exp(R::digamma(alphat.at(1)) - mu0 - ft);
			ft_ast = R::digamma(alphat.at(1) + Y.at(0)) - std::log(betat.at(1) + 1.) - mu0;
			Qt_ast = R::trigamma(alphat.at(1) + Y.at(0));
		} else {
			 throw std::invalid_argument("This combination of likelihood and link function is not supported yet.");
		}

		// Posterior at time t: theta[t] | D[t] ~ N(mt, Ct)
		et = std::abs(ft_ast - ft);
		At = Rt.slice(1) * Ft / Qt; // p x 1

		arma::vec mnew = at.col(1) + At * et;
		mt.col(1) = mnew;

		arma::mat cnew = Rt.slice(1) - At * (Qt - Qt_ast) * At.t();
		if (cnew.is_zero())
		{
			throw std::invalid_argument("C[1] is zero");
		}
		Ct.slice(1) = cnew;


		
		if (nlag > 1) {
			for (unsigned int t=2; t<=nlag; t++) {
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
	unsigned int tstart = 1;
	if (nlag < nt) {tstart = nlag + 1;}

	for (unsigned int t=tstart; t<=nt; t++) {
		R_CheckUserInterrupt();

		// Prior at time t: theta[t] | D[t-1] ~ (at, Rt)
		// Linear approximation is implemented if the state equation is nonlinear
		at.col(t) = update_at(
			Gt.slice(t), mt.col(t - 1), Y.at(t - 1),
			lag_par, gain_code, nlag, truncated); // p x 1
		update_Gt(
			Gt.slice(t), mt.col(t - 1), Y.at(t - 1),
			lag_par, gain_code, trans_code,
			nlag, truncated);
		update_Rt(Rt.slice(t), Ct.slice(t - 1), Gt.slice(t), use_discount, W, delta_discount);

		

		// One-step ahead forecast: Y[t]|D[t-1] ~ N(ft, Qt)
        if (truncated) { // truncated
			update_Ft_truncated(Ft, Fy, t, Y, Fphi, nlag, truncated);
			hpsi = psi2hpsi(at.col(t),gain_code);
			hph = hpsi_deriv(at.col(t),gain_code);

			ft = arma::accu(Ft % hpsi);
			Ft = Ft % hph;
			Qt = arma::as_scalar(Ft.t() * Rt.slice(t) * Ft);
		} else { // Vanilla, Koyck, Solow
			// no truncations
			ft = arma::as_scalar(Ft.t() * at.col(t));
			Qt = arma::as_scalar(Ft.t() * Rt.slice(t) * Ft);
		}

		ft = std::max(ft,EPS);
		bound_check(Qt,"ForwardFilter: Qt",true);

		/*
		------ Moment Matching ------
		Depends on specific link function
		*/
		phi = mu0 + ft; // regression on link function
		if (link_code == 0 && obs_code == 1) {
			// Poisson DLM with Identity Link
			betat.at(t) = phi / Qt;
			alphat.at(t) = phi * betat.at(t);
			Qt_ast = (alphat.at(t) + Y.at(t)) / (betat.at(t) + 1.) / (betat.at(t) + 1.);
			ft_ast = Qt_ast * (betat.at(t)+1.) - mu0;

		} else if (link_code == 0 && obs_code == 0) {
			// Negative-binomial DLM with Identity Link

			betat.at(t) = std::exp(std::log(phi)+std::log(phi+delta_nb)-std::log(Qt))+2.;
			alphat.at(t) = std::exp(std::log(phi)-std::log(delta_nb)+std::log(betat.at(t)-1.));
			ft_ast = delta_nb * (alphat.at(t) + Y.at(t)) / (betat.at(t) + delta_nb - 1.) - mu0;
			Qt_ast = (ft_ast + mu0) * delta_nb * (alphat.at(t) + Y.at(t) + betat.at(t) + delta_nb - 1.) / (betat.at(t) + delta_nb - 1.) / (betat.at(t) + delta_nb - 2.);
		} else if (link_code == 1 && obs_code == 1) {
			// Poisson DLM with Exponential Link
			alphat.at(t) = optimize_trigamma(Qt);
			betat.at(t) = std::exp(R::digamma(alphat.at(t)) - phi);
			ft_ast = R::digamma(alphat.at(t) + Y.at(t)) - std::log(betat.at(t) + 1.) - mu0;
			Qt_ast = R::trigamma(alphat.at(t) + Y.at(t));
		} else {
			 throw std::invalid_argument("This combination of likelihood and link function is not supported yet.");
		}

		ft_ast = std::max(ft_ast,EPS);
		bound_check(Qt_ast, "ForwardFilter: Qt_ast", true);
		
		

		// Posterior at time t: theta[t] | D[t] ~ N(mt, Ct)
		et = ft_ast - ft;
		At = Rt.slice(t) * Ft / Qt; // p x 1

		arma::vec mnew = at.col(t) + At * et;
		double mtmp = mnew.at(1);
		if (trans_code != 1)
		{
			mnew.at(1) = std::max(mtmp, EPS);
		}


		mt.col(t) = mnew;
		arma::mat cnew = Rt.slice(t) - At * (Qt - Qt_ast) * At.t();
		Ct.slice(t) = cnew;

		// Rcout << "\rFiltering: " << t << "/" << n;
	}
	// Rcout << std::endl;
}

void backwardSmoother(
	arma::vec &ht, // (n+1) x 1
	arma::vec &Ht, // (n+1) x 1
	const unsigned int &n,
	const unsigned int &p,
	const arma::mat &mt,  // p x (n+1), t=0 is the mean for initial value theta[0]
	const arma::mat &at,  // p x (n+1)
	const arma::cube &Ct, // p x p x (n+1), t=0 is the var for initial value theta[0]
	const arma::cube &Rt, // p x p x (n+1)
	const arma::cube &Gt,
	const double &W,
	const double &delta,
	const bool &use_discount)
{ // 1: smoothing, 2: sampling (conditional)

	ht.at(n) = mt.at(0,n);
	Ht.at(n) = std::abs(Ct.at(0,0,n));

	double coef1 = 1. - delta;
	double coef2 = delta * delta;

	arma::mat Bt, Ht_prev, Ht_cur;
	arma::vec ht_prev, ht_cur;
	if (!use_discount)
	{
		Bt.set_size(p, p); Bt.zeros();
		Ht_cur.set_size(p, p); Ht_cur.zeros();
		ht_cur.set_size(p); ht_cur.zeros();

		Ht_prev = Ct.slice(n);
		ht_prev = mt.col(n);
	}


	for (unsigned int t=(n-1); t>0; t--) 
	{
		R_CheckUserInterrupt();
		if (use_discount) 
		{
			ht.at(t) = coef1 * mt.at(0,t) + delta * ht.at(t+1);
			Ht.at(t) = coef1 * Ct.at(0,0,t) + coef2 * Ht.at(t+1);
		} 
		else 
		{
			// don't use discount
			try 
			{
				Bt = Ct.slice(t) * Gt.slice(t+1).t() * Rt.slice(t+1).i();
				ht_cur = mt.col(t) + Bt * (ht_prev - at.col(t+1));
				Ht_cur = Ct.slice(t) + Bt*(Ht_prev - Rt.slice(t+1))*Bt.t();

				ht.at(t) = ht_cur.at(0);
				Ht.at(t) = Ht_cur.at(0,0);

				ht_prev = ht_cur;
				Ht_prev = Ht_cur;
			} 
			catch (...) 
			{

				ht.at(t) = coef1 * mt.at(0, t) + delta * ht.at(t + 1);
				Ht.at(t) = coef1 * Ct.at(0, 0, t) + coef2 * Ht.at(t + 1);

				// Bt = Ct.slice(t) * Gt.slice(t + 1).t() * arma::pinv(Rt.slice(t + 1), 1.e-6); // Pseudo inverse
				// ht_cur = mt.col(t) + Bt * (ht_prev - at.col(t + 1));
				// Ht_cur = Ct.slice(t) + Bt * (Ht_prev - Rt.slice(t + 1)) * Bt.t();

				// ht.at(t) = ht_cur.at(0);
				// Ht.at(t) = Ht_cur.at(0, 0);

				// ht_prev = ht_cur;
				// Ht_prev = Ht_cur;
			}
			
		}
		
		// Rcout << "\rMarginal Smoothing: " << n+1-t << "/" << n;
	}
	// Rcout << std::endl;

	// t = 0
	if (use_discount) 
	{
		ht.at(0) = coef1 * mt.at(0,0) + delta * ht.at(1);
		Ht.at(0) = coef1 * Ct.at(0,0,0) + coef2 * Ht.at(1);
	} 
	else 
	{
		try 
		{
			Bt = Ct.slice(0) * Gt.slice(1).t() * Rt.slice(1).i();
			ht_cur = mt.col(0) + Bt * (ht_prev - at.col(1));
			Ht_cur = Ct.slice(0) + Bt*(Ht_prev - Rt.slice(1))*Bt.t();

			ht.at(0) = ht_cur.at(0);
			Ht.at(0) = Ht_cur.at(0,0);
		} 
		catch (...) 
		{
			ht.at(0) = coef1 * mt.at(0, 0) + delta * ht.at(1);
			Ht.at(0) = coef1 * Ct.at(0, 0, 0) + coef2 * Ht.at(1);

			// Bt = Ct.slice(t) * Gt.slice(t + 1).t() * arma::pinv(Rt.slice(t + 1), 1.e-6); // Pseudo inverse
			// ht_cur = mt.col(t) + Bt * (ht_prev - at.col(t + 1));
			// Ht_cur = Ct.slice(t) + Bt * (Ht_prev - Rt.slice(t + 1)) * Bt.t();

			// ht.at(t) = ht_cur.at(0);
			// Ht.at(t) = Ht_cur.at(0, 0);

			// ht_prev = ht_cur;
			// Ht_prev = Ht_cur;
		}
	}
}

void backwardSampler(
	arma::vec &theta, // (n+1) x 1
	const unsigned int &n,
	const unsigned int &p,
	const arma::mat &mt,  // p x (n+1), t=0 is the mean for initial value theta[0]
	const arma::mat &at,  // p x (n+1)
	const arma::cube &Ct, // p x p x (n+1), t=0 is the var for initial value theta[0]
	const arma::cube &Rt, // p x p x (n+1)
	const arma::cube &Gt,
	const double &W,
	const double &scale_sd)
{

	arma::mat Bt(p,p);
	arma::mat Ip(p,p,arma::fill::eye); Ip *= scale_sd;

	arma::vec ht = mt.col(n);
	arma::mat Ht = Ct.slice(n);
	try{
		Ht = arma::chol(arma::symmatu(Ht + Ip));
	} catch(...) {
		Rcout << "W=" << W << std::endl;
		 throw std::invalid_argument("Cholesky decomposition failed");
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
			 throw std::invalid_argument("Inversion of R[t+1] failed.");
		}
		

		// Conditional Sampling
		ht = mt.col(t) + Bt*(theta_prev - at.col(t+1));
		Ht = Ct.slice(t) - Bt * Rt.slice(t+1) * Bt.t();
		try {
			Ht = arma::chol(arma::symmatu(Ht + Ip));
		} catch(...) {
			Rcout << "W=" << W << std::endl;
			 throw std::invalid_argument("Cholesky decomposition failed");
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
	const arma::vec &Y, // n x 1, the observed response
	const arma::uvec &model_code,
	const double &W = 0.01, 
	const Rcpp::NumericVector &obs_par_in = Rcpp::NumericVector::create(0., 30.),
	const Rcpp::NumericVector &lag_par_in = Rcpp::NumericVector::create(0.5, 6), // init/true values of (mu, sg2) or (rho, L)
	const unsigned int &nlag_in = 20,
	const double &ci_coverage = 0.95,
	const unsigned int &npara = 20,
	const double &theta0_upbnd = 2.,
	const double &delta_discount = 0.88,
	const bool &truncated = true,
	const bool &use_discount = false,
	const bool &summarize_return = false)
{
	const unsigned int n = Y.n_elem;
	const unsigned int npad = n+1;

	arma::vec lag_par(lag_par_in.begin(), lag_par_in.length());
	arma::vec obs_par(obs_par_in.begin(), obs_par_in.length());

	unsigned int nlag = nlag_in;
	unsigned int p = nlag;
	if (!truncated)
	{
		nlag = n;
		p = (unsigned int) lag_par.at(1) + 1;
	}

	double mu0 = obs_par.at(0);
	double delta_nb = obs_par.at(1);

	const unsigned int obs_code = model_code.at(0); // Poisson or negative binomial
    const unsigned int link_code = model_code.at(1); // identity or exponential
    const unsigned int trans_code = model_code.at(2); // Koyck, Koyama, or Solow
    const unsigned int gain_code = model_code.at(3); // Exponential, Softplus, logistic, ...
    const unsigned int err_code = model_code.at(4); // normal, laplace, ...


	arma::vec Ypad(npad,arma::fill::zeros); // (n+1) x 1
	Ypad.tail(n) = Y; // **
	// Ypad.elem(arma::find(Ypad < EPS)).fill(EPS); // **
	Ypad.at(0) = static_cast<double>(sample(5,true));
	
	arma::mat at(p,npad,arma::fill::zeros);
	
	
	arma::cube Rt(p,p,npad);
	arma::vec alphat(npad,arma::fill::zeros);
	arma::vec betat(npad,arma::fill::zeros);
	arma::vec ht(npad,arma::fill::zeros);
	arma::vec Ht(npad,arma::fill::zeros);

	arma::cube Gt(p,p,npad);
	init_Gt(Gt,lag_par,p,nlag,truncated);

	arma::mat mt(p, npad, arma::fill::zeros);
	if (trans_code != 1)
	{
		mt.fill(0.1 / static_cast<double>(p));
	}
	// if (!m0_prior.isNull()) {
	// 	arma::vec m00 = Rcpp::as<arma::vec>(m0_prior);
	// 	mt.at(0,0) = std::abs(m00.at(0));
	// }

	arma::cube Ct(p, p, npad);
	Ct.slice(0).eye();
	Ct.for_each([&theta0_upbnd](arma::mat::elem_type &val){
		val *= std::pow(theta0_upbnd * 0.5,2.) + EPS;});


	forwardFilter(
		mt, at, Ct, Rt, Gt, alphat, betat,
		Ypad, n, p, obs_par, lag_par, W, 
		obs_code, link_code, trans_code, gain_code,
		nlag, delta_discount, truncated, use_discount);
	arma::vec Wt = (1. - delta_discount) * arma::vectorise(Rt.tube(0, 0));

	backwardSmoother(ht, Ht, n, p, mt, at, Ct, Rt, Gt, W, delta_discount, use_discount);

	Rcpp::List output;
	const double critval = R::qnorm(1. - 0.5 * (1. - ci_coverage), 0., 1., true, false);
	if (summarize_return) 
	{
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

	} 
	else 
	{
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
	arma::vec hpsiR = psi2hpsi(ht, gain_code); // hpsi: npad x 1
	arma::vec thetaR(npad,arma::fill::zeros);
	hpsi2theta(thetaR,hpsiR,Ypad,lag_par,trans_code,nlag,truncated);
	output["theta"] = Rcpp::wrap(thetaR);

    output["rmse"] = std::sqrt(arma::as_scalar(arma::mean(arma::pow(thetaR.tail(n-npara) - Y.tail(n-npara),2.0))));
    output["mae"] = arma::as_scalar(arma::mean(arma::abs(thetaR.tail(n-npara) - Y.tail(n-npara))));


	double logpred = 0.;
	double prob_success;
	for (unsigned int j=npara; j<=n; j++) 
	{
		// Marginal log predictive likelihood
		if (obs_code == 1) 
		{
			// Poisson DLM
			prob_success = std::max(betat.at(j)/(1.+betat.at(j)),EPS);
			logpred += R::dnbinom(Y.at(j-1),std::max(alphat.at(j),EPS),prob_success,true);

		} else if (obs_code == 0 && alphat.at(j)>1.e-32 && betat.at(j)>1.e-32) 
		{
			// Negative binomial DLM
			logpred += R::lgammafn(Y.at(j-1)+delta_nb)-R::lgammafn(delta_nb)-R::lgammafn(Y.at(j-1)+1.);
			logpred += R::lgammafn(std::max(alphat.at(j)+betat.at(j),EPS))-R::lgammafn(std::max(alphat.at(j),EPS))-R::lgammafn(std::max(betat.at(j),EPS));
			logpred += R::lgammafn(alphat.at(j)+Y.at(j-1))+R::lgammafn(std::max(betat.at(j)+delta_nb,EPS))-R::lgammafn(alphat.at(j)+Y.at(j-1)+betat.at(j)+delta_nb);
		}
	}
	output["logpred"] = logpred;


	return output;
}

// //' @export
// // [[Rcpp::export]]
// Rcpp::List get_eta_truncated(
// 	const arma::vec& Y, // n x 1, the observation (scalar), n: num of obs
// 	const arma::mat& mt, // p x (n+1), t=0 is the mean for initial value theta[0]
// 	const arma::cube& Ct, // p x p x (n+1)
// 	const unsigned int gain_code,
// 	const double mu0 = 0.) {
	
// 	const unsigned int n = Y.n_elem;
// 	const unsigned int p = mt.n_rows;

// 	arma::vec ft(n+1);
// 	arma::vec Qt(n+1,arma::fill::zeros);

// 	arma::vec Ft(p,arma::fill::zeros);
// 	arma::vec Fy(p,arma::fill::zeros);
// 	arma::vec Fphi = get_Fphi(p,p,0.9,1);

// 	arma::mat mt_ramp;
// 	if (gain_code == 0) {
// 		mt_ramp = mt;
// 		mt_ramp.elem(arma::find(mt_ramp<EPS)).zeros(); // Ramp function
// 	}

// 	for (unsigned int t=0; t<=n; t++) {
// 		update_Ft_truncated(Ft, Fy, t, Y, Fphi, p, n);
// 		switch (gain_code) {
// 			case 0: // KoyamaMax
// 			{
// 				ft.at(t) = mu0 + arma::accu(Ft % mt_ramp.col(t));
// 			}
// 			break;
// 			case 1: // KoyamaExp
// 			{
// 				ft.at(t) = mu0 + arma::accu(Ft % arma::exp(mt.col(t)));
// 				Ft = Ft % arma::exp(mt.col(t));
// 				Qt.at(t) = arma::as_scalar(Ft.t() * Ct.slice(t) * Ft);
// 			}
// 			break;
// 			case 2: // KoyamaEye
// 			{
// 				ft.at(t) = mu0 + arma::accu(Ft % mt.col(t));
// 				Qt.at(t) = arma::as_scalar(Ft.t() * Ct.slice(t) * Ft);
// 			}
// 			break;
// 			default:
// 			{
// 				 throw std::invalid_argument("get_Ft function is only defined for Koyama transmission kernels.");
// 			}
// 		} // END switch block
// 	}

// 	Rcpp::List output;
// 	output["mean"] = Rcpp::wrap(ft);
// 	output["var"] = Rcpp::wrap(Qt);
// 	return output;
// }


/**
 * Doesn't work for poisson + solow + softplus + gaussian
*/
//' @export
// [[Rcpp::export]]
Rcpp::List get_optimal_delta(
	const arma::vec &Y, // n x 1
	const arma::uvec &model_code,
	const arma::vec &delta_grid,											  // m x 1
	const Rcpp::NumericVector &obs_par_in = Rcpp::NumericVector::create(0., 30.), // (mu0, delta_nb)
	const Rcpp::NumericVector &lag_par_in = Rcpp::NumericVector::create(0.5, 6),
	const double &theta0_upbnd = 2.,
	const unsigned int &npara = 1,
	const double &ci_coverage = 0.95,
	const unsigned int &nlag_in = 20,
	const bool &truncated = true)
{
	const double delta_nb = obs_par_in[1];
	
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
	for (unsigned int i=0; i<m; i++) 
	{
		delta = delta_grid.at(i);
		Rcpp::List lbe = lbe_poisson(
			Y,model_code,W,obs_par_in,lag_par_in,
			nlag_in, ci_coverage,npara,
			theta0_upbnd,delta, truncated, true, false);
		
		arma::vec alphat = lbe["alphat"];
		arma::vec betat = lbe["betat"];
		for (unsigned int j=npara; j<=n; j++) 
		{
			// Marginal log predictive likelihood
			if (obs_code == 1) 
			{
				// Poisson DLM
				prob_success = betat.at(j)/(1.+betat.at(j));
				logpred.at(i) += R::dnbinom(Y.at(j-1),alphat.at(j),prob_success,true);

				// MSE
				ymean = (1.-prob_success)/prob_success*alphat.at(j);
				mse.at(i) += (Y.at(j-1)-ymean)*(Y.at(j-1)-ymean);

				// MAE
				mae.at(i) = std::abs(Y.at(j-1) - ymean);

			} 
			else if (obs_code == 0 && alphat.at(j)>1.e-32 && betat.at(j)>1.e-32) 
			{
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

// //' @export
// // [[Rcpp::export]]
// double lbe_What(
//     const arma::vec& ht, // (n+1) x 1
//     const unsigned int nsample = 2000,
//     const double aw = 0.1,
//     const double bw = 0.1) {
    
//     double n_ = static_cast<double>(ht.n_elem) - 1.;
//     double aw_new = aw + 0.5*n_;
//     double bw_new = bw + 0.5* arma::accu(arma::pow(arma::diff(ht),2));
//     double What = bw_new / (aw_new - 1.);
//     return What;
// }