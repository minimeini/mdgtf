#include "model_utils.h"

using namespace Rcpp;
// [[Rcpp::plugins(cpp17)]]
// [[Rcpp::depends(RcppArmadillo,nloptr,BH)]]


void tolower(std::string &S)
{
	for (char &x : S)
	{
		x = tolower(x);
	}
	return;
}

//' @export
// [[Rcpp::export]]
arma::uvec get_model_code(
	const std::string &obs_dist = "nbinom",
	const std::string &link_func = "identity",
	const std::string &trans_func = "koyama",
	const std::string &gain_func = "softplus",
	const std::string &err_dist = "gaussian")
{
	std::string obs = obs_dist; tolower(obs);
	std::string link = link_func; tolower(link);
	std::string trans = trans_func; tolower(trans);
	std::string gain = gain_func; tolower(gain);
	std::string err = err_dist; tolower(err);

	arma::uvec model_code(5,arma::fill::zeros);
	unsigned int obs_code;
	if (obs == "nbinom")
	{
		obs_code = 0;
	} else if (obs == "poisson")
	{
		obs_code = 1;
	} else if (obs == "nbinom_p")
	{
		obs_code = 2;
	} else if (obs == "gaussian")
	{
		obs_code = 3;
	} else
	{
		throw std::invalid_argument("obs_dist: 'nbinom', 'poisson', 'nbinom_p', 'gaussian' ");
	}
	model_code.at(0) = obs_code;

	unsigned int link_code;
	if (link == "identity")
	{
		link_code = 0;
	} else if (link == "exponential")
	{
		link_code = 1;
	} else
	{
		throw std::invalid_argument("link_func: 'identity', 'exponential'. ");
	}
	model_code.at(1) = link_code;

	unsigned int trans_code;
	if (trans == "koyck")
	{
		trans_code = 0;
	} else if (trans == "koyama") // lognormal
	{
		trans_code = 1;
	} else if (trans == "solow") // negative-binomial
	{
		trans_code = 2;
	} else if (trans == "vanilla")
	{
		trans_code = 3;
	} else
	{
		throw std::invalid_argument("trans_func: 'koyck', 'koyama'. 'solow', 'vanilla'. ");
	}
	model_code.at(2) = trans_code;

	unsigned int gain_code;
	if (gain == "ramp")
	{
		gain_code = 0;
	} else if (gain == "exponential")
	{
		gain_code = 1;
	} else if (gain == "identity")
	{
		gain_code = 2;
	} else if (gain == "softplus")
	{
		gain_code = 3;
	} else
	{
		throw std::invalid_argument("gain_func: 'ramp', 'exponential'. 'identity', 'softplus'. ");
	}
	model_code.at(3) = gain_code;

	unsigned int err_code;
	if (err == "gaussian")
	{
		err_code = 0;
	} else if (err == "laplace")
	{
		err_code = 1;
	} else if (err == "cauchy")
	{
		err_code = 2;
	} else
	{
		throw std::invalid_argument("err_dist: 'gaussian', 'laplace'. 'cauchy'. ");
	}
	model_code.at(4) = err_code;

	return model_code;
}



void get_model_code(
	unsigned int &obs_code,
	unsigned int &link_code,
	unsigned int &trans_code,
	unsigned int &gain_code,
	unsigned int &err_code,
	const arma::uvec &model_code
)
{
	obs_code = model_code.at(0);
	link_code = model_code.at(1);
	trans_code = model_code.at(2);
	gain_code = model_code.at(3);
	err_code = model_code.at(4);

	return;
}






void bound_check(
	const arma::mat &input,
	const std::string &name,
	const bool &check_zero,
	const bool &check_negative)
{
	bool is_infinite = !input.is_finite();
	if (is_infinite)
	{
		// std::cout << name + " = " << std::endl;
		// input.t().print();
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
		// std::cout << name + " = " << input << std::endl;
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

bool bound_check(
	const double &input,
	const bool &check_zero,
	const bool &check_negative)
{
	bool is_infinite = !std::isfinite(input);
	bool out_of_bound = is_infinite;
	if (check_zero)
	{
		bool is_zero = std::abs(input) < EPS;
		out_of_bound = out_of_bound || is_zero;
	}
	if (check_negative)
	{
		bool is_negative = input < -EPS;
		out_of_bound = out_of_bound || is_negative;
	}

	return out_of_bound;
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


arma::vec init_Ft(
	const unsigned int& p, // dimension of DLM state space
	const unsigned int &trans_code) {

	arma::vec Ft(p, arma::fill::zeros);
	if (trans_code == 2 || trans_code == 0)
	{
		Ft.at(1) = 1.;
	} 
	else if (trans_code == 3)
	{
		Ft.at(0) = 1.;
	}

	return Ft;
}

void init_Gt(
	arma::mat &Gt,
	const arma::vec &lag_par,
	const unsigned int &p,
	const unsigned int &nlag,
	const bool &truncated)
{
	Gt.set_size(p,p);
	Gt.zeros();
	Gt.at(0, 0) = 1.;

	if (truncated)
	{
		// truncated
		if (p == 1)
		{
			throw std::invalid_argument("init_Gt<mat>: p = 1 in truncation mode.");
		}
		Gt.diag(-1).ones();
	}
	else
	{
		// not truncated and use negative-binomial lags
		unsigned int L = lag_par.at(1);

		for (unsigned int i = 1; i < L; i++)
		{
			Gt.at(i + 1, i) = 1.;
		}

		for (unsigned int i = 1; i < p; i++)
		{
			double c1 = std::pow(-1., static_cast<double>(i));
			double c2 = binom(static_cast<double>(L), static_cast<double>(i));
			double c3 = std::pow(lag_par.at(0), static_cast<double>(i));
			Gt.at(1, i) = -c1;
			Gt.at(1, i) *= c2;
			Gt.at(1, i) *= c3;
		}
	}

	return;
}

void init_Gt(
	arma::cube &Gt,
	const arma::vec &lag_par,
	const unsigned int &p,
	const unsigned int &nlag,
	const bool &truncated)
{
	const unsigned int nslice = Gt.n_slices;

	for (unsigned int t = 0; t < nslice; t++)
	{
		arma::mat Gt_tmp(p,p,arma::fill::zeros);
		init_Gt(Gt_tmp, lag_par, p, nlag, truncated);
		Gt.slice(t) = Gt_tmp;
	}

	return;
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
double dlognorm0(
    const double &lag, // starting from 1
    const double &mu,
    const double &sd2) {
	double output = Pd(lag, mu, sd2) - Pd(lag - 1., mu, sd2);
	bound_check(output, "dlognorm0",false,true);
	return output;
}


//' @export
// [[Rcpp::export]]
arma::vec dlognorm(
	const double &nlag,
	const double &mu,
	const double &sd2)
{
	arma::vec output(nlag);
	for (unsigned int d=0; d<nlag; d++) {
		output.at(d) = dlognorm0(d+1.,mu,sd2);
	}

	return output;
}



double dnbinom0(
	const double &lag, // starting from 1
	const double &rho, 
	const double &L_order)
{
	double c3 = std::pow(1. - rho, L_order);
	double a = lag + L_order - 2.;
	double b = lag - 1.;
	double c1 = binom(a, b);
	double c2 = std::pow(rho, b);

	// double c1 = R::dnbinom(k-1,(double)Last,1.-rho);
	// double c2 = std::pow(-1.,k-1.);
	// Fphi.at(d) = c1 * c2;
	double output = (c1 * c2) * c3;
	bound_check(output,"dnbinom0",false,true);
	return output;
}



//' @export
// [[Rcpp::export]]
arma::vec dnbinom(
	const double &nlag,
	const double &rho,
	const double &L_order
)
{
	arma::vec output(nlag,arma::fill::zeros);
	double c3 = std::pow(1. - rho, L_order);
	for (unsigned int d=0; d<nlag; d++)
	{
		double lag = static_cast<double>(d) + 1.;
		double a = lag + L_order - 2.;
		double b = lag - 1.;
		double c1 = binom(a, b);
		double c2 = std::pow(rho, b);

		output.at(d) = c1 * c2;
		output.at(d) *= c3;
	}

	bound_check(output,"dnbinom",false,true);
	return output;
}

arma::vec dlags(
	const unsigned int &nlags,
	const arma::vec &params) // 3 x 1, params[0] = trans_code: type of distribution
{
	unsigned int trans_code = (unsigned int)params[0];
	arma::vec y;
	switch (trans_code)
	{
	case 1:
	{
		y = dlognorm(nlags, params[1], params[2]);
	}
	break;
	case 2:
	{
		y = dnbinom(nlags, params[1], params[2]);
	}
	break;
	default:
	{
		throw std::invalid_argument("dlags: undefined distribution.");
	}
	break;
	}

	bound_check(y,"dlags");
	return y;
}

//' @export
// [[Rcpp::export]]
double cross_entropy(
	const unsigned int &nlags,
	const arma::vec &params_p, // 3 x 1, params1[0] = trans_code: type of distribution
	const arma::vec &params_q)
{
	arma::vec y1 = dlags(nlags, params_p);
	arma::vec y2 = dlags(nlags, params_q);
	arma::vec logy2 = arma::log(y2);
	arma::vec tmp = y1 % logy2;
	double output = arma::accu(tmp);
	bound_check(output,"cross_entropy");
	return output;
}

//' @export
// [[Rcpp::export]]
arma::vec match_params(
	const arma::vec &params_in,
	const unsigned int &trans_code_out,
	const arma::vec &par1_grid, // n1 x 1, mu or rho
	const arma::vec &par2_grid,	// n2 x 1, sg2 or L_order
	const unsigned int &nlags = 30
)
{
	unsigned int n1 = par1_grid.n_elem;
	unsigned int n2 = par2_grid.n_elem;
	arma::mat output(n1*n2,3,arma::fill::zeros);

	unsigned int idx = 0;
	for (unsigned int i=0; i<n1; i++)
	{
		double par1 = par1_grid.at(i);
		for (unsigned int j=0; j<n2; j++)
		{
			double par2 = par2_grid.at(j);
			output.at(idx,0) = par1;
			output.at(idx,1) = par2;

			arma::vec params_out = {(double)trans_code_out,par1,par2};
			output.at(idx,2) = cross_entropy(nlags,params_in,params_out);
			idx ++;
		}
	}


	arma::vec par2_hat(n1,arma::fill::zeros);
	arma::vec out_hat(n1,arma::fill::zeros);
	for (unsigned int i=0; i<n1; i++)
	{
		double par1 = par1_grid.at(i); 
		arma::vec col1 = output.col(0);
		arma::vec diff1 = arma::abs(col1 - par1);
		arma::uvec idx_row = arma::find(diff1 < EPS);
		arma::uvec idx_col = {0,1,2};
		arma::mat out_sub = output.submat(idx_row,idx_col); // n2 x 3

		arma::vec out = out_sub.col(2);
		unsigned int i2 = out.index_max();
		double par2 = out_sub.at(i2, 1);
		par2_hat.at(i) = par2;
		out_hat.at(i) = out_sub.at(i2,2); // cross_entropy (p1[i],p2_hat[i])
	}

	unsigned int imax = out_hat.index_max();
	double par1 = par1_grid.at(imax);
	double par2 = par2_hat.at(imax);

	arma::vec params_out = {(double)trans_code_out,par1,par2};
	return params_out;
}



//' @export
// [[Rcpp::export]]
unsigned int get_truncation_nlag(
	const unsigned int &trans_code,
	const double &err_margin = 0.01,
	const Rcpp::NumericVector &lag_par = Rcpp::NumericVector::create(0.5,6))
{
	unsigned int nlag = 1;
	bool cont = true;
	arma::vec lag_par_arma(lag_par.begin(),lag_par.length());
	while (cont)
	{
		arma::vec Fphi = get_Fphi(nlag, lag_par_arma, trans_code);
		double prob = arma::accu(Fphi);

		if (1 - prob <= err_margin)
		{
			cont = false;
		}
		else
		{
			nlag += 1;
		}
	}

	return nlag;

}



/**
 * get_Fphi
 * 
 * Used in: LBA::forwardFilter, pl_poisson.cpp, MCMC::get_Fphi_pad
*/
//' @export
// [[Rcpp::export]]
arma::vec get_Fphi( // equivalent to `dlags` but fixed params for lognorm
	const unsigned int &nlag, // number of Lags
	const arma::vec &lag_par,
	const unsigned int &trans_code)
{
	arma::vec Fphi(nlag, arma::fill::zeros);

	switch (trans_code) 
	{
		case 0: 
		{
			// koyck
		}
		break;
		case 1: // Checked. OK
		{
			// koyama: discretized lognormal
			// const double sm2 = std::pow(covid_s / covid_m, 2);
			// const double pk_mu = std::log(covid_m / std::sqrt(1. + sm2));
			// const double pk_sg2 = std::log(1. + sm2);
			// const double pk_mu = 1.386262;
			// const double pk_sg2 = 0.3226017;
			Fphi = dlognorm(nlag, lag_par.at(0), lag_par.at(1));
		}
		break;
		case 2:
		{
			// solow: negative binomial and thus no truncation
			// const double rho_hat = 0.395;
			// const double r_hat = 6;
			Fphi = dnbinom(nlag, lag_par.at(0), lag_par.at(1));
		}
		break;
		default:
		{
			throw std::invalid_argument(
				"get_Fphi: undefined distributed lag distribution.");
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



// Checked. OK
//' @export
// [[Rcpp::export]]
arma::mat psi2hpsi(
	const arma::mat& psi,
	const unsigned int &gain_code) {

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
			throw std::invalid_argument("Not supported gain function.");
		}
	}

	bound_check(hpsi, "psi2hpsi<mat>: hpsi", false, true);
	hpsi.elem(arma::find(hpsi < EPS)).fill(EPS);

	return hpsi;
}

// Checked. OK
double psi2hpsi(
	const double &psi,
	const unsigned int &gain_code) {
	
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
			throw std::invalid_argument("Not supported gain function.");
		}
	}

	bound_check(hpsi,"psi2hpsi<double>: hpsi",false,true);
	hpsi = std::max(hpsi,EPS);

	return hpsi;
}



/**
 * hpsi_deriv: Checked. OK.
*/
void hpsi_deriv(
	arma::mat & hpsi,
	const arma::mat& psi,
	const unsigned int &gain_code) {
	
	hpsi.copy_size(psi);

	switch (gain_code) {
		case 0: // Ramp
		{
			throw std::invalid_argument("Ramp function is non-differentiable.");
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
			throw std::invalid_argument("Not supported gain function.");
		}
	}

	bound_check(hpsi, "hpsi_deriv<void>: hpsi");
	return;
}



/**
 * hpsi_deriv: Checked. OK.
*/
double hpsi_deriv(
	const double &psi,
	const unsigned int &gain_code) {
	
	double hpsi;

	switch (gain_code) {
		case 0: // Ramp
		{
			throw std::invalid_argument("Ramp function is non-differentiable.");
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
			throw std::invalid_argument("Not supported gain function.");
		}
	}

	bound_check(hpsi,"hpsi_deriv<double>: hpsi");

	return hpsi;
}



/**
 * hpsi_deriv: Checked. OK.
*/
//' @export
// [[Rcpp::export]]
arma::mat hpsi_deriv(
	const arma::mat& psi,
	const unsigned int &gain_code) {
	
	arma::mat hpsi;
	hpsi.copy_size(psi);

	switch (gain_code) {
		case 0: // Ramp
		{
			throw std::invalid_argument("Ramp function is non-differentiable.");
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
			throw std::invalid_argument("Not supported gain function.");
		}
	}

	bound_check(hpsi, "hpsi_deriv<mat>: hpsi");

	return hpsi;
}




arma::vec get_theta_coef_solow(const unsigned int &L, const double &rho)
{
	arma::vec coef(L, arma::fill::zeros); // L x 1, in reverse order from large to small
	for (unsigned int k = L; k >= 1; k--)
	{
		double c1 = binom(static_cast<double>(L), static_cast<double>(k));
		double c2 = std::pow(-rho, static_cast<double>(k));
		coef.at(L - k) = c1 * c2;
	}
	return coef;
}



double theta_new_nobs(
	const arma::vec &Fphi_sub, // nelem x 1
	const arma::vec &hpsi_sub, // nelem x 1
	const arma::vec &ysub)  // nelem x 1
{
	arma::vec yh = hpsi_sub % ysub;
	arma::vec yh2 = arma::reverse(yh);
	arma::vec coef = Fphi_sub % yh2;
	double theta_new = arma::accu(coef);
	bound_check(theta_new, "theta_new_nobs: theta_new", false, true);
	theta_new = std::max(theta_new, EPS);
	return theta_new;
}


//' @export
// [[Rcpp::export]]
double theta_new_nobs(
	const arma::vec &hpsi_pad, // (n+1) x 1
	const arma::vec &ypad,	   // (n+1) x 1
	const unsigned int &tidx,  // t = 1, ..., n
	const arma::vec &lag_par,
	const unsigned int &trans_code = 2,
	const unsigned int &nlag_in = 20,
	const bool &truncated = true)
{
	unsigned int nobs = ypad.n_elem - 1;
	unsigned int nlag = nlag_in;
	if (!truncated) { nlag = nobs; }
	unsigned int nelem = std::min(tidx, nlag);
	if (nelem == 0) {
		throw std::invalid_argument("theta_new_nobs: nelem must be positive integer.");
	}

	arma::vec Fphi = get_Fphi(nelem, lag_par, trans_code); // nelem x 1
	arma::vec hpsi_sub = hpsi_pad.subvec(tidx-nelem + 1, tidx);
	arma::vec ysub = ypad.subvec(tidx-nelem, tidx - 1);
	double theta_new = theta_new_nobs(Fphi,hpsi_sub,ysub);
	return theta_new;
}



//' @export
// [[Rcpp::export]]
arma::mat hpsi2theta(
	const arma::mat& hpsi_pad, // (n+1) x k, each row is a different time point
	const arma::vec& ypad, // (n+1) x 1
	const arma::vec &lag_par,
	const unsigned int &trans_code = 2,
	const unsigned int &nlag_in = 20,
	const bool &truncated = true) 
{
	const unsigned int nobs = ypad.n_elem - 1;
	unsigned int nlag = nlag_in;
	if (!truncated) {nlag = nobs;}

	const unsigned int k = hpsi_pad.n_cols;
	arma::mat theta(nobs,k,arma::fill::zeros);
	arma::vec Fphi = get_Fphi(nlag, lag_par, trans_code); // nlag0 x 1


	for (unsigned int t = 1; t <= nobs; t++)
	{
		unsigned int nelem = std::min(t, nlag);
		arma::vec ysub = ypad.subvec(t - nelem, t - 1); // nelem x 1
		arma::mat hsub = hpsi_pad.rows(t - nelem + 1, t);		 // nelem x k
		arma::vec Fphi_sub = Fphi.head(nelem);

		for (unsigned int i = 0; i < k; i++)
		{
			double tsum = theta_new_nobs(Fphi_sub,hsub.col(i),ysub);
			theta.at(t - 1, i) = tsum;
		}
	}

	return theta;
}

void hpsi2theta(
	arma::vec &theta,		   // n x 1
	const arma::vec &hpsi_pad, // (n+1) x 1, each row is a different time point
	const arma::vec &ypad,	   // (n+1) x 1
	const arma::vec &lag_par,
	const unsigned int &trans_code,
	const unsigned int &nlag_in,
	const bool &truncated)
{

	const unsigned int nobs = ypad.n_elem - 1;
	unsigned int nlag = nlag_in;
	if (!truncated)
	{
		nlag = nobs;
	}

	theta.set_size(nobs);
	theta.zeros();

	arma::vec Fphi = get_Fphi(nlag, lag_par, trans_code); // nlag0 x 1

	for (unsigned int t = 1; t <= nobs; t++)
	{
		unsigned int nelem = std::min(t, nlag);
		arma::vec ysub = ypad.subvec(t - nelem, t - 1);	  // nelem x 1
		arma::vec hsub = hpsi_pad.subvec(t - nelem + 1, t); // nelem x k
		arma::vec Fphi_sub = Fphi.head(nelem);

		double tsum = theta_new_nobs(Fphi_sub, hsub, ysub);
		theta.at(t - 1) = tsum;
	}

	return;
}

void wt2theta(
	arma::vec &theta,	 // n x 1
	const arma::vec &wt, // n x 1
	const arma::vec &y,	 // n x 1
	const arma::vec &lag_par,
	const unsigned int &gain_code,
	const unsigned int &trans_code,
	const unsigned int &nlag,
	const bool &truncated)
{
	unsigned int n = y.n_elem;
	arma::vec psi = arma::cumsum(wt);		   // n x 1
	arma::vec hpsi = psi2hpsi(psi, gain_code); // n x 1

	arma::vec ypad(n + 1, arma::fill::zeros);
	ypad.tail(y.n_elem) = y;
	arma::vec hpsi_pad(n + 1, arma::fill::zeros);
	hpsi_pad.tail(hpsi.n_elem) = hpsi;

	theta.zeros();
	hpsi2theta(theta, hpsi_pad, ypad, lag_par, trans_code, nlag, truncated);
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
			throw std::invalid_argument("Not supported observational distribution.");
		}
	}

	bound_check(loglike,"loglike_obs: loglike");
	return loglike;
}