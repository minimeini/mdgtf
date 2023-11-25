#include "model_utils.h"

using namespace Rcpp;
// [[Rcpp::plugins(cpp17)]]
// [[Rcpp::depends(RcppArmadillo,nloptr,BH)]]

void bound_check(
	const arma::mat &input,
	const std::string &name,
	const bool &check_zero,
	const bool &check_negative)
{
	bool is_infinite = !input.is_finite();
	if (is_infinite)
	{
		std::cout << name + " = " << std::endl;
		input.t().print();
		throw std::invalid_argument("bound_check: " + name + " is nonfinite");
	}
	if (check_zero)
	{
		bool is_zero = input.is_zero();
		if (is_zero)
		{
			throw std::invalid_argument("bound_check: " + name + " is zeros");
		}
	}
	if (check_negative)
	{
		bool is_negative = arma::any(arma::vectorise(input) < -EPS);
		if (is_negative)
		{
			throw std::invalid_argument("bound_check: " + name + " is negative");
		}
	}

	return;
}

void bound_check(
	const double &input,
	const std::string &name,
	const bool &check_zero,
	const bool &check_negative)
{
	bool is_infinite = !std::isfinite(input);
	if (is_infinite)
	{
		std::cout << name + " = " << input << std::endl;
		throw std::invalid_argument("bound_check: " + name + " is nonfinite");
	}
	if (check_zero)
	{
		bool is_zero = std::abs(input) < EPS;
		if (is_zero)
		{
			throw std::invalid_argument("bound_check: " + name + " is zeros");
		}
	}
	if (check_negative)
	{
		bool is_negative = input < -EPS;
		if (is_negative)
		{
			throw std::invalid_argument("bound_check: " + name + " is negative");
		}
	}

	return;
}

arma::uvec sample(
	const unsigned int n, 
	const unsigned int size, 
	const arma::vec &weights, 
	bool replace,
	bool zero_start){
	Rcpp::NumericVector w_ = Rcpp::NumericVector(weights.begin(), weights.end());
	Rcpp::IntegerVector idx_ = Rcpp::sample(n, size, true, w_);

	arma::uvec idx = Rcpp::as<arma::uvec>(Rcpp::wrap(idx_));
	if (zero_start)
	{
		idx.for_each([](arma::uvec::elem_type &val)
					 { val -= 1; });
	}
	
	return idx;
}

unsigned int sample(
	const int n,
	const arma::vec &weights,
	bool zero_start)
{
	Rcpp::NumericVector w_ = Rcpp::NumericVector(weights.begin(), weights.end());
	Rcpp::IntegerVector idx_ = Rcpp::sample(n, 1, true, w_);

	arma::uvec idx = Rcpp::as<arma::uvec>(Rcpp::wrap(idx_));
	unsigned int idx0 = idx[0];
	if (zero_start)
	{
		idx0 -= 1;
	}

	return idx0;
}

unsigned int sample(
	const int n,
	bool zero_start)
{
	arma::vec weights(n,arma::fill::ones);
	Rcpp::NumericVector w_ = Rcpp::NumericVector(weights.begin(), weights.end());
	Rcpp::IntegerVector idx_ = Rcpp::sample(n, 1, true, w_);

	arma::uvec idx = Rcpp::as<arma::uvec>(Rcpp::wrap(idx_));
	unsigned int idx0 = idx[0];
	if (zero_start)
	{
		idx0 -= 1;
	}

	return idx0;
}

void init_by_trans(
	unsigned int& p, // dimension of DLM state space
	unsigned int& L_,
	const unsigned int trans_code,
	const unsigned int L) {

	switch (trans_code) {
		case 0: // Koyck
		{
			p = 2;
			L_ = 0;
		}
		break;
		case 1: // Koyama
		{
			p = L;
			L_ = L;
		}
		break;
		case 2: // Solow
		{
			p = L + 1;
			L_ = 0;
		}
		break;
		case 3: // Vanilla
		{
			p = 1;
			L_ = 0;
		}
		break;
		default:
		{
			::Rf_error("Not supported transfer function.");
		}
	}
}



void init_by_trans(
	unsigned int& p, // dimension of DLM state space
	unsigned int& L_,
	arma::vec& Ft,
    arma::vec& Fphi,
	const unsigned int trans_code,
	const unsigned int L) {

	switch (trans_code) {
		case 0: // Koyck
		{
			p = 2;
			L_ = 0;

			Ft.set_size(p); Ft.zeros();
			Ft.at(1) = 1.;
			Fphi.set_size(p); Fphi.zeros();
		}
		break;
		case 1: // Koyama
		{
			p = L;
			L_ = L;

			Ft.set_size(p); Ft.zeros();
			Fphi = get_Fphi(p,p,0.9,1);
		}
		break;
		case 2: // Solow
		{
			p = L + 1;
			L_ = 0;

			Ft.set_size(p); Ft.zeros();
			Ft.at(1) = 1.;
			Fphi.set_size(p); Fphi.zeros();
		}
		break;
		case 3: // Vanilla
		{
			p = 1;
			L_ = 0;
			
			Ft.set_size(p); Ft.at(0) = 1.;
			Fphi.set_size(p); Fphi.zeros();
		}
		break;
		default:
		{
			::Rf_error("Not supported transfer function.");
		}
	}
}

void init_Gt(
	arma::mat &Gt,
	const double &rho,
	const unsigned int &p,
	const unsigned int &trans_code)
{
	Gt.set_size(p,p);
	Gt.zeros();
	const unsigned int r = p - 1;
	Gt.at(0, 0) = 1.;

	switch (trans_code)
	{
	case 0: // Koyck
	{
		Gt.at(1, 1) = rho;
	}
	break;
	case 1: // Koyama
	{
		Gt.diag(-1).ones();
	}
	break;
	case 2: // Solow - Checked. Correct.
	{
		// double coef2 = -rho;
		// Gt0.at(1, 1) = -binom(L, 1) * coef2;
		// for (unsigned int k = 2; k < p; k++)
		// {
		// 	coef2 *= -rho;
		// 	Gt0.at(1, k) = -binom(L, k) * coef2;
		// 	Gt0.at(k, k - 1) = 1.;
		// }
		
		for (unsigned int i = 1; i < r; i++)
		{
			Gt.at(i + 1, i) = 1.;
		}

		double tmp1 = std::pow(1. - rho, r);
		for (unsigned int i = 1; i < p; i++)
		{
			double c1 = std::pow(-1., static_cast<double>(i));
			double c2 = binom(static_cast<double>(r), static_cast<double>(i));
			double c3 = std::pow(rho, static_cast<double>(i));
			Gt.at(1, i) = -c1;
			Gt.at(1, i) *= c2;
			Gt.at(1, i) *= c3;
		}
	}
	break;
	case 3: // Vanilla
	{
		Gt.at(0, 0) = rho;
	}
	break;
	default:
	{
		throw std::invalid_argument("Not supported transfer function.\n");
	}
	}

	return;
}

void init_Gt(
	arma::cube &Gt, 
	const double &rho, 
	const unsigned int &p,
	const unsigned int &trans_code)
{
	const unsigned int r = p - 1;
	const unsigned int npad = Gt.n_slices;

	for (unsigned int t = 0; t < npad; t++)
	{
		Gt.slice(t).zeros();
		Gt.at(0, 0, t) = 1.;
		switch (trans_code)
		{
		case 0: // Koyck
		{
			Gt.at(1, 1, t) = rho;
		}
		break;
		case 1: // Koyama
		{
			Gt.slice(t).diag(-1).ones();
		}
		break;
		case 2: // Solow - Checked. Correct.
		{
			// double coef2 = -rho;
			// Gt0.at(1, 1) = -binom(L, 1) * coef2;
			// for (unsigned int k = 2; k < p; k++)
			// {
			// 	coef2 *= -rho;
			// 	Gt0.at(1, k) = -binom(L, k) * coef2;
			// 	Gt0.at(k, k - 1) = 1.;
			// }
			for (unsigned int i = 1; i < r; i++)
			{
				Gt.at(i + 1, i, t) = 1.;
			}

			double tmp1 = std::pow(1. - rho, r);

			for (unsigned int i = 1; i < p; i++)
			{
				double c1 = std::pow(-1., static_cast<double>(i));
				double c2 = binom(static_cast<double>(r), static_cast<double>(i));
				double c3 = std::pow(rho, static_cast<double>(i));
				Gt.at(1, i, t) = -c1;
				Gt.at(1, i, t) *= c2;
				Gt.at(1, i, t) *= c3;
			}
		}
		break;
		case 3: // Vanilla
		{
			Gt.at(0, 0, t) = rho;
		}
		break;
		default:
		{
			throw std::invalid_argument("Not supported transfer function.\n");
		}
		}
	}
}


/*
Binomial coefficients, i.e., n-choose-k
*/
//' @export
// [[Rcpp::export]]
double binom(double n, double k) { 
	return 1./((n+1.)*boost::math::beta(std::max(n-k+1,EPS),std::max(k+1,EPS))); }



/*
Function that computes \Phi_d in Koyama et al. (2021)
CDF of the log-normal distribution.

------ R Version ------
Pd=function(d,mu,sigmasq){
  phid=0.5*pracma::erfc(-(log(d)-mu)/sqrt(2*sigmasq))
  phid
}
*/
double Pd(
    const double d,
    const double mu,
    const double sigmasq) {
    arma::vec tmpv1(1);
    tmpv1.at(0) = -(std::log(d)-mu)/std::sqrt(2.*sigmasq);
    return arma::as_scalar(0.5*arma::erfc(tmpv1));
}


/*
Function that computes \phi_d in Koyama et al. (2021)
Difference of the subsequent CDFs of the log-normal distribution, which is the PDF at the discrete scale.

------ R Version ------
knl=function(t,mu,sd2){
 Pd(t,mu,sd2)-Pd(t-1,mu,sd2)
}
*/
double knl(
    const double t,
    const double mu,
    const double sd2) {
    return Pd(t,mu,sd2) - Pd(t-1.,mu,sd2);
}


//' @export
// [[Rcpp::export]]
arma::vec knl(
	const arma::vec& tvec,
	const double mu,
	const double sd2) {
	
	const unsigned int n = tvec.n_elem;
	arma::vec output(n);
	for (unsigned int i=0; i<n; i++) {
		output.at(i) = knl(tvec.at(i),mu,sd2);
	}

	return output;
}


//' @export
// [[Rcpp::export]]
arma::vec get_Fphi(
	const unsigned int &n, // number of Lags
	const unsigned int &L, // dimension of state space (-1 for solow)
	const double &rho,
	const unsigned int &trans_code)
{
	double Last = L;
	if (Last == 0)
	{
		Last = n;
	}
	arma::vec Fphi(n, arma::fill::zeros);

	switch (trans_code) 
	{
		case 0: 
		{
			// koyck
		}
		break;
		case 1:
		{
			// koyama
			Last = n;

			double tmpd;
			const double sm2 = std::pow(covid_s / covid_m, 2);
			const double pk_mu = std::log(covid_m / std::sqrt(1. + sm2));
			const double pk_sg2 = std::log(1. + sm2);
			
			Fphi.at(0) = knl(1., pk_mu, pk_sg2);
			for (unsigned int d = 1; d < n; d++)
			{
				tmpd = static_cast<double>(d) + 1.;
				Fphi.at(d) = knl(tmpd, pk_mu, pk_sg2);
			}
		}
		break;
		case 2: 
		{
			// solow
			double c3 = std::pow(1.-rho,static_cast<double>(Last));
			for (unsigned int d = 0; d < n; d++)
			{
				unsigned int k = d + 1;
				double a = k + Last - 2.;
				double b = k - 1.;
				double c1 = binom(a,b);
				double c2 = std::pow(-rho,k-1.);
				Fphi.at(d) = (c1 * c2) * c3;
			}
		}
	}
    return Fphi;
}

double trigamma_obj(
	unsigned n, // not sure what it is for
	const double *x, 
	double *grad, 
	void *my_func_data) { // extra parameters: q

	double *q = (double*)my_func_data;

	if (grad) {
		grad[0] = 2*(R::trigamma(x[0])-(*q))*R::psigamma(x[0],2);
	}

	return std::pow(R::trigamma(x[0])-(*q),2);
}



double optimize_trigamma(double q) {
	nlopt_opt opt;
	opt = nlopt_create(NLOPT_LD_MMA, 1);

	double lb[1] = {0}; // lower bound
	nlopt_set_lower_bounds(opt,lb);
	nlopt_set_xtol_rel(opt,1e-4);
	nlopt_set_maxeval(opt, 50);
	nlopt_set_maxtime(opt, 5.);
	nlopt_set_min_objective(opt,trigamma_obj,(void *) &q);

	double x[1] = {1e-6};
	double minf;
	if (nlopt_optimize(opt, x, &minf) < 0) {
    	Rprintf("nlopt failed!\\n");
	}
	
	double result = x[0];
	nlopt_destroy(opt);
	return result;
}


/*
postW_gamma_obj

Objective function should be in the convex form with minimum.

1. Evaluate the objective function and return the value in the output
2. Update first-order derivative via input arguments


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
double postW_gamma_obj(
	unsigned n,
	const double *x, // Wtilde = log(W)
	double *grad,
	void *my_func_data){
	
	coef_W *coef = (coef_W*)my_func_data;
	double a1 = coef -> a1;
	double a2 = coef -> a2;
	double a3 = coef -> a3;

	// logarithm of the conditional posterior of Wtilde = log(W)
	double logpost = a1*x[0]-a2*std::exp(x[0])-a3*std::exp(-x[0]);
	logpost *= -1.; // flip concave function to convex

	// First-order derivative
	if (grad) {
		grad[0] = a1-a2*std::exp(x[0])+a3*std::exp(-x[0]);
		grad[0] *= -1.;
	}

	return logpost;
}


double optimize_postW_gamma(coef_W& coef) {
	nlopt_opt opt;
	opt = nlopt_create(NLOPT_LD_MMA, 1); // algorithm and dimensionality

	double lb[1] = {-700.}; // lower bound
	double ub[1] = {700.}; // upper bound
	
	nlopt_set_lower_bounds(opt,lb);
	nlopt_set_upper_bounds(opt,ub);
	nlopt_set_xtol_rel(opt,1e-4);
	nlopt_set_maxeval(opt, 50);
	nlopt_set_maxtime(opt, 5.);
	nlopt_set_min_objective(opt,postW_gamma_obj,(void *) &coef);

	double x[1] = {1e-6};
	double minf;
	if (nlopt_optimize(opt, x, &minf) < 0) {
    	Rprintf("nlopt failed!\\n");
	}
	
	double result = x[0];
	nlopt_destroy(opt);
	return result;
}


double postW_deriv2(
	const double Wtilde,
	const double a2,
	const double a3) {
	
	double deriv2 = - a2*std::exp(Wtilde) - a3*std::exp(-Wtilde);
	return deriv2;
}

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
// 	const double a1 = -0.5,
// 	const double a2 = 1.e2,
// 	const double a3 = 0.5){
// 	coef_W coef[1] = {{a1,a2,a3}};
// 	double What = optimize_postW_gamma(coef[0]);
// 	return What;
// }



//' @export
// [[Rcpp::export]]
arma::mat psi2hpsi(
	const arma::mat& psi,
	const unsigned int gain_code) {

	if (!psi.is_finite())
	{
		throw std::invalid_argument("psi2hpsi<mat>: Input psi is not finite.");
	}
	
	arma::mat hpsi;
	hpsi.copy_size(psi);

	switch (gain_code) {
		case 0: // Ramp
		{
			hpsi = psi;
			hpsi.elem(arma::find(psi<EPS)).fill(EPS);
		}
		break;
		case 1: // Exponential
		{
			hpsi = psi;
			hpsi.elem(arma::find(hpsi>UPBND)).fill(UPBND);
			hpsi = arma::exp(hpsi);
		}
		break;
		case 2: // Identity
		{
			hpsi = psi;
		}
		break;
		case 3: // Softplus
		{
			arma::mat hpsi0 = psi;
			hpsi0.elem(arma::find(hpsi0>UPBND)).fill(UPBND);
			arma::mat hpsi1 = arma::exp(hpsi0);
			hpsi = arma::log(1. + hpsi1);
		}
		break;
		default:
		{
			::Rf_error("Not supported gain function.");
		}
	}

	bound_check(hpsi, "psi2hpsi<mat>: hpsi", false, true);
	hpsi.elem(arma::find(hpsi < EPS)).fill(EPS);

	return hpsi;
}



double psi2hpsi(
	const double psi,
	const unsigned int gain_code) {
	
	double hpsi;

	switch (gain_code) {
		case 0: // Ramp
		{
			hpsi = std::max(psi,EPS);
		}
		break;
		case 1: // Exponential
		{
			hpsi = std::exp(std::min(psi,UPBND));
		}
		break;
		case 2: // Identity
		{
			hpsi = psi;
		}
		break;
		case 3: // Softplus
		{
			hpsi = std::log(1. + std::exp(std::min(psi,UPBND)));
		}
		break;
		default:
		{
			::Rf_error("Not supported gain function.");
		}
	}

	bound_check(hpsi,"psi2hpsi<double>: hpsi",false,true);
	hpsi = std::max(hpsi,EPS);

	return hpsi;
}




void hpsi_deriv(
	arma::mat & hpsi,
	const arma::mat& psi,
	const unsigned int gain_code) {
	
	hpsi.copy_size(psi);

	switch (gain_code) {
		case 0: // Ramp
		{
			::Rf_error("Ramp function is non-differentiable.");
		}
		break;
		case 1: // Exponential
		{
			arma::mat tmp = psi;
			tmp.elem(arma::find(tmp>UPBND)).fill(UPBND);
			hpsi = arma::exp(tmp);
		}
		break;
		case 2: // Identity
		{
			hpsi.ones();
		}
		break;
		case 3: // Softplus
		{
			arma::mat tmp = -psi;
			tmp.elem(arma::find(tmp>UPBND)).fill(UPBND);
			hpsi = 1./(1.+arma::exp(tmp));
		}
		break;
		default:
		{
			::Rf_error("Not supported gain function.");
		}
	}

	bound_check(hpsi, "hpsi_deriv<void>: hpsi");
	return;
}


double hpsi_deriv(
	const double psi,
	const unsigned int gain_code) {
	
	double hpsi;

	switch (gain_code) {
		case 0: // Ramp
		{
			::Rf_error("Ramp function is non-differentiable.");
		}
		break;
		case 1: // Exponential
		{
			hpsi = std::exp(std::min(psi,UPBND));
		}
		break;
		case 2: // Identity
		{
			hpsi = 1.;
		}
		break;
		case 3: // Softplus
		{
			hpsi = 1. / (1. + std::exp(std::min(-psi, UPBND)));
		}
		break;
		default:
		{
			::Rf_error("Not supported gain function.");
		}
	}

	bound_check(hpsi,"hpsi_deriv<double>: hpsi");

	return hpsi;
}



//' @export
// [[Rcpp::export]]
arma::mat hpsi_deriv(
	const arma::mat& psi,
	const unsigned int gain_code) {
	
	arma::mat hpsi;
	hpsi.copy_size(psi);

	switch (gain_code) {
		case 0: // Ramp
		{
			::Rf_error("Ramp function is non-differentiable.");
		}
		break;
		case 1: // Exponential
		{
			arma::mat tmp = psi;
			tmp.elem(arma::find(tmp>UPBND)).fill(UPBND);
			hpsi = arma::exp(tmp);
		}
		break;
		case 2: // Identity
		{
			hpsi.ones();
		}
		break;
		case 3: // Softplus
		{
			arma::mat tmp0 = -psi;
			tmp0.elem(arma::find(tmp0 > UPBND)).fill(UPBND);
			arma::mat tmp = arma::exp(tmp0);
			hpsi = 1./(1.+ tmp);
		}
		break;
		default:
		{
			::Rf_error("Not supported gain function.");
		}
	}

	bound_check(hpsi, "hpsi_deriv<mat>: hpsi");

	return hpsi;
}





//' @export
// [[Rcpp::export]]
arma::mat hpsi2theta(
	const arma::mat& hpsi, // (n+1) x k, each row is a different time point
	const arma::vec& ypad, // (n+1) x 1
	const unsigned int &trans_code,
	const double &theta0,
	const unsigned int &L,
	const double &rho) {
	
	const unsigned int n = ypad.n_elem - 1;
	const unsigned int k = hpsi.n_cols;
	arma::mat theta(n,k,arma::fill::zeros);

	switch(trans_code) {
		case 0: // Koyckget_model_code
		{
			theta.row(0).fill(rho*theta0);
			for (unsigned int t=1; t<n; t++) {
				// theta: if t = 1 in cpp <=> t = 2 in math
				// y: if t = 1 in cpp <=> t = 2 in math
				// hpsi: if t = 1 in cpp <=> t = 1 in math
				theta.row(t) = rho*theta.row(t-1) + hpsi.row(t)*ypad.at(t);
			}
		}
		break;
		case 1: // Koyama
		{
			arma::vec Fphi = get_Fphi(L,L,rho,1); // L x 1
			arma::vec ypad2(n+L,1,arma::fill::zeros);
			arma::mat hpsi_pad2(n+L,k,arma::fill::zeros);
			for (unsigned int t=L; t<(n+L); t++)
			{
				ypad2.at(t) = ypad.at(t-L+1);
				hpsi_pad2.row(t) = hpsi.row(t-L+1);
			}

			for (unsigned int t=1; t<=n; t++)
			{
				arma::vec ysub = ypad2.subvec(t-1, t-1+L-1); // L x 1
				arma::mat hsub = hpsi_pad2.rows(t,t-1+L); // L x k
				for (unsigned int i=0; i<=k; i++)
				{
					arma::vec tmp0 = ysub % hsub.col(i);
					arma::vec tmp1 = arma::reverse(tmp0);
					arma::vec tmp2 = Fphi % tmp1;
					double tsum = arma::accu(tmp2);
					bound_check(tsum, "hpsi2theta-theta", false, true);
					tsum = std::max(tsum, EPS);

					theta.at(t - 1, i) = tsum;
				}
			}
		}
		break;
		case 2: // Solow
		{
			/*
			Solow: exact update of theta via the ARMA like equation with the past thetas.

			theta_pad: (n+L) x k
			Value: theta[-r+1], ..., theta[-1], theta[0], theta[1], ..., theta[n]
			Index:	   0		    	r-2		   r-1		 r			   r+n-1
			The indices refer to row index. The column index is the replications of theta.

			hpsi: (n+1) x k
			Value: 								hpsi[0],  hpsi[1], ..., hpsi[n]

			y: n x 1
			Value: 											y[0],  ...,  y[n-1]
			*/
			arma::mat theta_pad(n + L, k, arma::fill::zeros);
			arma::vec coef(L,arma::fill::zeros); // L x 1
			for (unsigned int t=L; t>=1; t--)
			{
				double c1 = binom(static_cast<double>(L),static_cast<double>(t));
				double c2 = std::pow(-rho, static_cast<double>(t));
				coef.at(L-t) = c1 * c2;
			}

			double cnst = std::pow(1.-rho,static_cast<double>(L));

			for (unsigned int t=0; t<n; t++)
			{
				arma::mat theta_sub = theta_pad.rows(t, t+L-1); // L x k
				double yt_old = ypad.at(t);
				for (unsigned int i=0; i<k; i++)
				{
					arma::vec tmp = coef % theta_sub.col(i);
					double hpsi_cur = hpsi.at(t+1,i);

					double tmp1 = arma::accu(tmp);
					double tmp2 = cnst * hpsi_cur;
					tmp2 *= yt_old;
					double tsum = -tmp1 + tmp2;
					bound_check(tsum,"hpsi2theta-theta",false,true);
					tsum = std::max(tsum, EPS);

					theta_pad.at(t+L,i) = tsum;
					theta.at(t,i) = tsum;
				}
			}
		}
		break;
		default: // Otherwise
		{
			::Rf_error("Unsupported type of transfer function.");
		}
	}

	return theta;
}

void hpsi2theta(
	arma::vec &theta, // n x 1
	const arma::vec &hpsi, // (n+1) x 1, each row is a different time point
	const arma::vec &ypad, // (n+1) x 1
	const unsigned int &trans_code,
	const double &theta0,
	const unsigned int &L,
	const double &rho)
{

	const unsigned int n = ypad.n_elem - 1;
	theta.set_size(n);
	theta.zeros();

	switch (trans_code)
	{
	case 0: // Koyckget_model_code
	{
		theta.row(0).fill(rho * theta0);
		for (unsigned int t = 1; t < n; t++)
		{
			// theta: if t = 1 in cpp <=> t = 2 in math
			// y: if t = 1 in cpp <=> t = 2 in math
			// hpsi: if t = 1 in cpp <=> t = 1 in math
			theta.row(t) = rho * theta.row(t - 1) + hpsi.row(t) * ypad.at(t);
		}
	}
	break;
	case 1: // Koyama
	{
		arma::vec Fphi = get_Fphi(L,L,rho,1); // L x 1
		arma::vec ypad2(n + L, 1, arma::fill::zeros);
		arma::vec hpsi_pad2(n + L, arma::fill::zeros);
		for (unsigned int t = L; t < (n + L); t++)
		{
			ypad2.at(t) = ypad.at(t - L + 1);
			hpsi_pad2.at(t) = hpsi.at(t - L + 1);
		}

		for (unsigned int t = 1; t <= n; t++)
		{
			arma::vec ysub = ypad2.subvec(t - 1, t - 1 + L - 1); // L x 1
			arma::vec hsub = hpsi_pad2.subvec(t, t - 1 + L);		 // L x 1
			arma::vec tmp0 = ysub % hsub;
			arma::vec tmp1 = arma::reverse(tmp0);
			arma::vec tmp2 = Fphi % tmp1;
			double tsum = arma::accu(tmp2);
			bound_check(tsum, "hpsi2theta-theta", false, true);
			tsum = std::max(tsum, EPS);

			theta.at(t - 1) = tsum;
		}
	}
	break;
	case 2: // Solow
	{
		/*
		Solow: exact update of theta via the ARMA like equation with the past thetas.

		theta_pad: (n+L) x 1
		Value: theta[-r+1], ..., theta[-1], theta[0], theta[1], ..., theta[n]
		Index:	   0		    	r-2		   r-1		 r			   r+n-1

		hpsi: (n+1) x 1
		Value: 								hpsi[0],  hpsi[1], ..., hpsi[n]

		y: n x 1
		Value: 											y[0],  ...,  y[n-1]
		*/
		arma::vec theta_pad(n + L, arma::fill::zeros);
		arma::vec coef(L, arma::fill::zeros); // L x 1, in reverse order from large to small
		for (unsigned int t = L; t >= 1; t--)
		{
			double c1 = binom(static_cast<double>(L), static_cast<double>(t));
			double c2 = std::pow(-rho, static_cast<double>(t));
			coef.at(L - t) = c1 * c2;
		}

		double cnst = std::pow(1. - rho, static_cast<double>(L));

		for (unsigned int t = 0; t < n; t++)
		{
			arma::vec theta_sub = theta_pad.subvec(t, t + L - 1); // L x 1, in natural order from small to large
			double yt_old = ypad.at(t);
			arma::vec tmp = coef % theta_sub;

			double hpsi_cur = hpsi.at(t + 1);

			double tmp1 = arma::accu(tmp);
			double tmp2 = cnst * hpsi_cur;
			tmp2 *= yt_old;
			double tsum = -tmp1 + tmp2;
			bound_check(tsum, "hpsi2theta-theta", false, true);
			tsum = std::max(tsum, EPS);

			theta_pad.at(t + L) = tsum;
			theta.at(t) = tsum;
		}
	}
	break;
	default: // Otherwise
	{
		::Rf_error("Unsupported type of transfer function.");
	}
	}

	return;
}

double loglike_obs(
	const double &y, 
	const double &lambda,
	const unsigned int &obs_code,
	const double &delta_nb,
	const bool &return_log) {
	
	if (!std::isfinite(y) || !std::isfinite(lambda))
	{
		throw std::invalid_argument("loglike_obs: y or lambda is not finite.");
	}
	
	double loglike = -9999.;
	double yabs = std::abs(y);
	double labs = std::max(lambda,EPS);

	switch (obs_code) {
		case 0: // negative-binomial
		{
			/*
        	Negative-binomial likelihood
        	- mean: lambda.at(i)
        	- delta_nb: degree of over-dispersion

        	sample variance exceeds the sample mean
        	*/
			loglike = R::lgammafn(yabs+delta_nb);
			loglike -=R::lgammafn(yabs+1.);
			loglike -= R::lgammafn(delta_nb);
			double c1 = std::log(delta_nb);
			double c2 = std::log(delta_nb+labs);
			loglike += delta_nb*(c1-c2);
			double c3 = std::log(labs);
			double c4 = std::log(delta_nb+labs);
			loglike += yabs*(c3 - c4);


			if (!return_log) {
				loglike = std::exp(loglike);
				// loglike = std::exp(std::min(loglike,UPBND));
			}
		}
		break;
		case 1: // poisson
		{
			/*
        	Poisson likelihood
        	- mean: lambda.at(i)
        	- var: lambda.at(i)

        	sample variance == sample mean
        	*/
        	loglike = R::dpois(yabs,labs,return_log);
		}
		break;
		default:
		{
			::Rf_error("Not supported observational distribution.\n");
		}
	}

	bound_check(loglike,"loglike_obs: loglike");
	return loglike;
}