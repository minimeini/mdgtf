#include "model_utils.h"

using namespace Rcpp;
// [[Rcpp::plugins(cpp17)]]
// [[Rcpp::depends(RcppArmadillo,nloptr,BH)]]

void bound_check(
	const arma::mat &input,
	const bool &check_zero = false,
	const bool &check_negative = false)
{
	bool is_infinite = !input.is_finite();
	if (is_infinite)
	{
		throw std::invalid_argument("bound_check: infinite.");
	}
	if (check_zero)
	{
		bool is_zero = input.is_zero();
		if (is_zero)
		{
			throw std::invalid_argument("bound_check: zeros");
		}
	}
	if (check_negative)
	{
		bool is_negative = arma::any(arma::vectorise(input) < EPS);
		if (is_negative)
		{
			throw std::invalid_argument("bound_check: negative");
		}
	}

	return;
}

arma::uvec sample(
	const unsigned int n, 
	const unsigned int size, 
	const arma::vec &weights, 
	bool replace=true,
	bool zero_start=true){
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



void init_by_trans(
	unsigned int& p, // dimension of DLM state space
	unsigned int& L_,
	const unsigned int trans_code,
	const unsigned int L = 2) {

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
	const unsigned int L = 2) {

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
			Fphi = get_Fphi(p);
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


arma::vec get_solow_coef(
	const unsigned int &rho, 
	const unsigned int &r // number of theta
)
{
	arma::vec coef(r,arma::fill::zeros);
	for (unsigned int i=0; i<r; i++)
	{
		double c1 = binom(static_cast<double>(r),static_cast<double>(i+1));
		double c2 = std::pow(-rho,static_cast<double>(i+1));
		coef.at(i) = c1 * c2;
	}

	return coef;
}


/*
Function that computes \Phi_d in Koyama et al. (2021)
CDF of the log-normal distribution.

------ R Version ------
Pd=function(d,mu,sigmasq){
  phid=0.5*pracma::erfc(-(log(d)-mu)/sqrt(2*sigmasq))
  phid
}
*/
//' @export
// [[Rcpp::export]]
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
arma::vec get_Fphi(const unsigned int L=30){ // number of Lags to be considered
    double tmpd;
    const double sm2 = std::pow(covid_s/covid_m,2);
    const double pk_mu = std::log(covid_m/std::sqrt(1.+sm2));
    const double pk_sg2 = std::log(1.+sm2);

    arma::vec Fphi(L,arma::fill::zeros);
    Fphi.at(0) = knl(1.,pk_mu,pk_sg2);
    for (unsigned int d=1; d<L; d++) {
        tmpd = static_cast<double>(d) + 1.;
        Fphi.at(d) = knl(tmpd,pk_mu,pk_sg2);
    }
    return Fphi;
}





/*
update_Fx

TODO: check it.
*/
//' @export
// [[Rcpp::export]]
arma::mat update_Fx(
	const unsigned int trans_code,
	const unsigned int n, // number of observations
	const arma::vec& Y, // n x 1, (y[1],...,y[n]), observtions
	const arma::vec& hph, // (n+1) x 1, derivative of the gain function at h[t]
	const double rho = 0.99, // memory of previous state, for exponential and negative binomial transmission delay kernel
	const unsigned int L = 2, // length of nonzero transmission delay for Koyama's log-normal or number of trials for Solow's negative-binomial
	const Rcpp::Nullable<Rcpp::NumericVector>& Fphi_ = R_NilValue) { // L x 1, (phi[1],...,phi[L]), for log-normal transmission delay kernel only

	arma::mat Fx(n,n,arma::fill::zeros);
	unsigned int tmpi;
	double tmpd;

	arma::vec Ypad(n+1,arma::fill::zeros);
	Ypad.tail(n) = Y; // Ypad = (Y[0]=0,Y[1],...,Y[n])

	switch (trans_code) {
		case 0: // Koyck
		{
			for (unsigned int t=1; t<=n; t++) { // t-1 is the row index in cpp, theta[t]
				tmpd = 1.;
				Fx.at(t-1,t-1) = tmpd*Ypad.at(t-1)*hph.at(t);
				for (unsigned int i=(t-1); i>=1; i--) { // i-1 is the column index in cpp, w[i]
					tmpd *= rho;
					Fx.at(t-1,i-1) = Fx.at(t-1,i) + tmpd*Ypad.at(i-1)*hph.at(i);
				}
			}
		}
		break;
		case 1: // Koyama
		{
			arma::vec Fphi(L,arma::fill::ones);
			if (!Fphi_.isNull()) {
				Fphi = Rcpp::as<arma::vec>(Fphi_);
			}

			// Fill out the first two columns, starting from the second row of Fx
			arma::vec Fy(L,arma::fill::zeros);
			arma::vec Fh(L,arma::fill::zeros);
        	for (unsigned int r=1; r<n; r++) { 
            	// loop through row, r from 1 to n-1, aka time from 2 to n.
            	tmpi = std::min(L,r); // number of nonzero terms
            	Fy.zeros();
				Fy.head(tmpi) = arma::reverse(Y.subvec(r-tmpi,r-1));
				Fh.head(tmpi) = arma::reverse(hph.subvec(r-tmpi+2,r+1));
            	tmpd = arma::accu(Fphi % Fy % Fh);
            	Fx.at(r,0) = tmpd;
            	Fx.at(r,1) = tmpd;
            	// Fill out the rest of the columns until the diagonal element
            	if (r>1) { // start from the third row
                	for (unsigned int c=2; c<=r; c++) { 
						// loop through columns until the diagonal
                    	tmpi = std::min(L,r-c+1);
                    	Fx.at(r,c) = arma::accu(Fphi.head(tmpi) % Fy.head(tmpi) % Fh.head(tmpi));
                	}
            	}
        	}
		}
		break;
		case 2: // Solow
		{
			// double coef = 0.;
			for (unsigned int t=1; t<=n; t++) { // t-1 is the row index in cpp, theta[t]
				tmpd = 1.;
				// coef = static_cast<double>(t-t+1);
				Fx.at(t-1,t-1) = tmpd*Ypad.at(t-1)*hph.at(t);
				for (unsigned int i=(t-1); i>=1; i--) { 
					// i-1 is the column index in cpp, w[i]
					tmpd *= rho;
					// coef = static_cast<double>(t-i+1);
					Fx.at(t-1,i-1) = Fx.at(t-1,i) + binom(t-1-i+L,t-i)*tmpd*Ypad.at(i-1)*hph.at(i);
				}
			}
			double coef2 = std::pow(1.-rho,static_cast<double>(L));
			Fx.for_each( [&coef2](arma::mat::elem_type& val) {val *= coef2;});
		}
		break;
		case 3: //Vanilla
		{
			for (unsigned int t=1; t<=n; t++) { // t-1 is the row index in cpp, theta[t]
				tmpd = 1.;
				Fx.at(t-1,t-1) = tmpd;
				for (unsigned int i=(t-1); i>=1; i--) { // i-1 is the column index in cpp, w[i]
					tmpd *= rho;
					Fx.at(t-1,i-1) = Fx.at(t-1,i) + tmpd;
				}
			}
		}
		break;
		default:
		{
			::Rf_error("Undefined model.");
		}
	}

	return Fx;
}


/*
TODO: check it.
*/
//' @export
// [[Rcpp::export]]
arma::vec update_theta0(
	const unsigned int trans_code,
	const unsigned int n, // number of observations
	const arma::vec& Y, // n x 1, observations, (Y[1],...,Y[n])
	const arma::vec& hhat, // (n+1) x 1, 1st Taylor expansion of h(psi[t]) at h[t]
	const double theta00 = 0., // Initial value of the transfer function block for Vanilla or Koyck
	const double rho = 0.99, // memory of previous state, for exponential and negative binomial transmission delay kernel
	const unsigned int L = 1., // length of nonzero transmission delay for log-normal
	const Rcpp::Nullable<Rcpp::NumericVector>& Fphi_ = R_NilValue) { // L x 1
	
	arma::vec theta0(n,arma::fill::zeros);
	unsigned int tmpi;
	double tmpd;

	arma::vec Ypad(n+1,arma::fill::zeros);
	Ypad.tail(n) = Y; // Ypad = (Y[0]=0,Y[1],...,Y[n])

	// if (ModelCode==0) {
	// 	// KoyamaMax
	// 	theta0.zeros();

	switch (trans_code) {
		case 0:
		{
			// Koyck
			::Rf_error("Koyck not supported yet.");
		}
		break;
		case 1: // Koyama
		{
			arma::vec Fphi(L,arma::fill::ones);
			if (!Fphi_.isNull()) {
				Fphi = Rcpp::as<arma::vec>(Fphi_);
			}

			// Fill out the first two columns, starting from the second row of Fx
			arma::vec Fy(L,arma::fill::zeros);
			arma::vec Fh(L,arma::fill::zeros);
        	for (unsigned int r=1; r<n; r++) { 
            	// loop through row, r from 1 to n-1, aka time from 2 to n.
            	tmpi = std::min(L,r); // number of nonzero terms
            	Fy.zeros();
				Fh.zeros();
				Fy.head(tmpi) = arma::reverse(Y.subvec(r-tmpi,r-1));

				Fh.head(tmpi) = arma::reverse(hhat.subvec(r-tmpi+2,r+1));
                // 	// if r = 2, the third row, it would be (y[1],y[0]) in C, aka (y[2],y[1]) in R; 2 total elements
                // 	// if r = (L-1), the L row, it would be (y[L-2],...,y[0]) in C, aka (y[L-1],...,y[1]) in R; (L-1) total elements
                // 	// if r = L, the (L+1) row, it would be (y[L-1],...,y[0]) in C, aka (y[L],...,y[1]) in R; L total elements
                // 	// if r = n-1, the n row, it would be (y[n-2],...,y[n-L-1]) in C, aka (y[n-1],...,y[n-L])
                // 	// if r = 1 (t=2), second row, it would be (y[0]) in C, aka (y[1]) in R

            	theta0.at(r) = arma::accu(Fphi % Fy % Fh);
        	}
		}
		break;
		case 2: // Solow
		{
			// double coef = 0.;
			double coef2 = std::pow(1.-rho,static_cast<double>(L)); 
			for (unsigned int t=1; t<=n; t++) { // t-1 is the row index in cpp, theta[t]
				// tmpd = 1.;
				// coef = static_cast<double>(t-t+1.);
				// theta0.at(t-1) = coef*tmpd*Ypad.at(t-1)*hhat.at(t);
				tmpd = 1.;
				theta0.at(t-1) = 0.;
				for (unsigned int l=0; l<=(t-1); l++) { 
					// i-1 is the column index in cpp, w[i]
					// tmpd *= rho;
					// coef = static_cast<double>(t-i+1);
					// theta0.at(t-1) += coef*tmpd*Ypad.at(i-1)*hhat.at(i);
					theta0.at(t-1) += binom(L+l-1,l)*tmpd*Ypad.at(t-l-1)*hhat.at(t-l-1);
					tmpd *= rho;
				}
				theta0.at(t-1) *= coef2;
			}
		}
		break;
		case 3:
		{
			// Vanilla
			double tmpd2 = 1.;
			for (unsigned int t=1; t<=n; t++) { // t-1 in cpp == theta[t]
				tmpd2 *= rho;
				theta0.at(t-1) = tmpd2*theta00;
			}
		}
		break;
		default:
		{
			::Rf_error("Undefined model.");
		}
	}


	return theta0;
}



/*
This is using an alternate model which holds in most cases for the reproduction number,
since it barely drops below 0.

------ Model ------
<obs>   y[t] | lambda[t] ~ Pois(lambda[t]),
<link>  lambda[t] = max(lambda[0] + theta[t], 0),
<state> theta[t] = phi[1] psi[t] y[t-1] + phi[2] psi[t-1] y[t-2] + ... + phi[L] psi[t-L+1] y[t-L],
        psi[t] = psi[t-1] + omega[t], omega[t] ~ N(0,W).
-------------------
*/
arma::vec update_theta(
	const unsigned int n,
	const arma::mat& Fx, // n x n
	const arma::vec& w) { // n x 1
	
	arma::vec theta = Fx * w; // n x 1
	return theta; 
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
double test_postW_gamma(
	const double a1 = -0.5,
	const double a2 = 1.e2,
	const double a3 = 0.5){
	coef_W coef[1] = {{a1,a2,a3}};
	double What = optimize_postW_gamma(coef[0]);
	return What;
}



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

	bool has_negative = arma::any(arma::vectorise(hpsi)<-EPS);
	bool has_nonfinite = !hpsi.is_finite();

	if (has_negative || has_nonfinite)
	{
		hpsi.print();
		throw std::invalid_argument("psi2hpsi<mat>: hpsi is not positive or not finite.");
	} else if (arma::any(arma::vectorise(hpsi) < EPS))
	{
		hpsi.elem(arma::find(hpsi<EPS)).fill(EPS);
	}

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

	if ( (hpsi < -EPS) || !std::isfinite(hpsi))
	{
		throw std::invalid_argument("psi2hpsi<double> is not positive or not finite.");
	} else if (hpsi < EPS)
	{
		hpsi = EPS;
	}


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

	if (!hpsi.is_finite())
	{
		throw std::invalid_argument("hpsi_deriv<void>: non-finite output");
	}

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

	if (!std::isfinite(hpsi))
	{
		std::cout << "psi = " << psi << std::endl;
		throw std::invalid_argument("hpsi_deriv<double>: non-finite output");
	}


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

	if (!hpsi.is_finite())
	{
		throw std::invalid_argument("hpsi_deriv<mat>: non-finite output");
	}

	return hpsi;
}





//' @export
// [[Rcpp::export]]
arma::mat hpsi2theta(
	const arma::mat& hpsi, // (n+1) x k, each row is a different time point
	const arma::vec& y, // n x 1
	const unsigned int trans_code,
	const double theta0 = 0.,
	const unsigned int L = 2,
	const double rho = 0.9) {
	
	const unsigned int n = y.n_elem;
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
				theta.row(t) = rho*theta.row(t-1) + hpsi.row(t)*y.at(t-1);
			}
		}
		break;
		case 1: // Koyama
		{
			arma::vec Fphi = get_Fphi(L);
			arma::vec Fy(L,arma::fill::zeros);
			arma::mat Fhpsi(L,k);
			theta.row(0).zeros();
			unsigned int tmpi;
			for (unsigned int t=1; t<n; t++) {
				Fy.zeros();
				tmpi = std::min(t,L);
				Fy.head(tmpi) = arma::reverse(y.subvec(t-tmpi,t-1)); // p x 1
				Fhpsi.zeros();
				Fhpsi.head_rows(tmpi) = arma::reverse(hpsi.rows(t-tmpi+1,t)); // p x k
				for (unsigned int i=0; i<k; i++) {
					theta.at(t,i) = arma::accu(Fphi%Fy%Fhpsi.col(i));
				}
			}
		}
		break;
		case 2: // Solow
		{
			// double c1 = 2.*rho;
			// double c2 = rho*rho;
			arma::mat theta_ext(n+1,k,arma::fill::zeros);
			theta_ext.row(0).fill(theta0);

			double c1 = std::pow(1.-rho,static_cast<double>(L));
			double c2 = -rho;
			theta_ext.row(1) = -binom(L,1)*c2*theta_ext.row(0);
			// theta.row(1) = c1*theta.row(0) - c2*theta0 + c3*hpsi.row(1)*y.at(0);
			for (unsigned int t=2; t<(n+1); t++) {
				theta_ext.row(t) = c1*hpsi.row(t-1)*y.at(t-2);
				c2 = -rho;
				for (unsigned int k=1; k<=std::min(t,L); k++) {
					theta_ext.row(t) -= binom(L,k)*c2*theta_ext.row(t-k);
					c2 *= -rho;
				}
				// theta.row(t) = c1*theta.row(t-1) - c2*theta.row(t-2) + c3*hpsi.row(t)*y.at(t-1);
			}
			theta = theta_ext.tail_rows(n);
		}
		break;
		default: // Otherwise
		{
			::Rf_error("Unsupported type of transfer function.");
		}
	}

	if (arma::any(arma::vectorise(theta)<-EPS) || !theta.is_finite())
	{
		throw std::invalid_argument("hpsi2theta: non-positive or non-finite theta.");
	}
	return theta;
}


double loglike_obs(
	const double &y, 
	const double &lambda,
	const unsigned int &obs_code = 1,
	const double &delta_nb = 1.,
	const bool &return_log = false) {
	
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
				// loglike = std::exp(loglike);
				loglike = std::exp(std::min(loglike,UPBND));
				if (!std::isfinite(loglike))
				{
					std::cout << "loglike = " << loglike << std::endl;
					throw std::invalid_argument("loglike_obs: non-finite probabilitties.");
				}
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

	return loglike;
}