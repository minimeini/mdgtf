#include "model_utils.h"

using namespace Rcpp;
// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::depends(RcppArmadillo,nloptr)]]


/*
--------------------------
------ Related Code ------
--------------------------

- `mcmc_disturbance_poisson.cpp`
- `sim_pois_dglm.R`
*/



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
------ ModelCode = 0 -------
----------------------------
------ Koyama's Model ------
----------------------------

------ Discretized Hawkes Form ------
<obs> y[t] | lambda[t] ~ Pois(lambda[t])
<link> lambda[t] = phi[1] max(psi[t],0) y[t-1] + phi[2] max(psi[t-1],0) y[t-2] + ... + phi[L] max(psi[t-L+1],0) y[t-L]
<state> psi[t] = psi[t-1] + omega[t], omega[t] ~ N(0,W)


------ Dynamic Linear Model Form ------
<obs> y[t] | lambda[t] = Ft(theta[t]), 
    where Ft(theta[t]) = phi[1] y[t-1] max(theta[t,1],0) + ... + phi[L] y[t-L] max(theta[t,L],0)
<Link> theta[t] = G theta[t-1] + Omega[t], 
    where G = 1 0 ... 0 0
              1 0 ... 0 0
              . .     . .
              .   .   . .
              .     . . .
              0 0 ... 1 0
        
    and Omega[t] = (omega[t],0,...,0) ~ N(0,W[t]), W[t][1,1] = W and 0 otherwise.

-----------------------
------ Inference ------
-----------------------

1. [x] Linear Bayes Approximation with first order Taylor expansion
2. [x] Linear Bayes Approximation with second order Taylor expansion >> doesn't exist because the Hessian will be exactly zero
3. [x] MCMC Disturbance sampler
4. [x] Sequential Monte Carlo filtering and smoothing
5. [x] Vanila variational Bayes
6. [x] Hybrid variational Bayes

*/


//' @export
// [[Rcpp::export]]
unsigned int get_gaincode(
	const unsigned int ModelCode
) {
	unsigned int GainCode = -1; 
	if (ModelCode==0 || ModelCode==2 || ModelCode==4) {
		GainCode = 0; // 0 - Ramp
	} else if (ModelCode==1 || ModelCode==3 || ModelCode==5) {
		GainCode = 1; // 1 - Exponential
	} else if (ModelCode==6 || ModelCode==7 || ModelCode==8) {
		GainCode = 2; // 2 - Identity
	} else if (ModelCode==10 || ModelCode==11 || ModelCode==12) {
		GainCode = 3; // 3 - Softplus
	} else if (ModelCode==13 || ModelCode==14 || ModelCode==15) {
		GainCode = 4; // 4 - Hyperbolic Tangent with coefficients (a,b,c)
	} else if (ModelCode==16 || ModelCode==17 || ModelCode==18) {
		GainCode = 5; // 5 - Logistic with coefficients (a,b,c)
	} else {
		::Rf_error("Unsupported model.");
	}
	return GainCode;
}


void get_transcode(
	unsigned int& TransferCode, // integer indicator for the type of transfer function
	unsigned int& p, // dimension of DLM state space
	unsigned int& L_,
	const unsigned int ModelCode,
	const unsigned int L = 30
) {
	const bool is_solow = ModelCode == 2 || ModelCode == 3 || ModelCode == 7 || ModelCode == 12 || ModelCode == 15;
	const bool is_koyck = ModelCode == 4 || ModelCode == 5 || ModelCode == 8 || ModelCode == 10 || ModelCode == 13;
	const bool is_koyama = ModelCode == 0 || ModelCode == 1 || ModelCode == 6 || ModelCode == 11 || ModelCode == 14;
	const bool is_vanilla = ModelCode == 9;

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
}


//' @export
// [[Rcpp::export]]
unsigned int get_transcode(
	const unsigned int ModelCode
) {
	const bool is_solow = ModelCode == 2 || ModelCode == 3 || ModelCode == 7 || ModelCode == 12 || ModelCode == 15;
	const bool is_koyck = ModelCode == 4 || ModelCode == 5 || ModelCode == 8 || ModelCode == 10 || ModelCode == 13;
	const bool is_koyama = ModelCode == 0 || ModelCode == 1 || ModelCode == 6 || ModelCode == 11 || ModelCode == 14;
	const bool is_vanilla = ModelCode == 9;

	unsigned int TransferCode;

	if (is_koyck) { 
		TransferCode = 0; 
	} else if (is_koyama) { 
		TransferCode = 1; 
	} else if (is_solow) { 
		TransferCode = 2; 
	} else if (is_vanilla) {
		TransferCode = 3;
	} else {
		::Rf_error("Unknown type of model.");
	}
	return TransferCode;
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
arma::vec get_Fphi(
    const unsigned int L, // number of Lags to be considered
    const double mu = 2.2204e-16,
    const double m = 4.7,
    const double s = 2.9){

    double tmpd;
    const double sm2 = std::pow(s/m,2);
    const double pk_mu = std::log(m/std::sqrt(1.+sm2));
    const double pk_sg2 = std::log(1.+sm2);

    arma::vec Fphi(L,arma::fill::zeros);
    Fphi.at(0) = knl(1.,pk_mu,pk_sg2);
    for (unsigned int d=1; d<L; d++) {
        tmpd = static_cast<double>(d) + 1.;
        Fphi.at(d) = knl(tmpd,pk_mu,pk_sg2);
    }
    return Fphi;
}



arma::vec get_Fphi(const unsigned int L){ // number of Lags to be considered

    const double m = 4.7;
    const double s = 2.9;

    double tmpd;
    const double sm2 = std::pow(s/m,2);
    const double pk_mu = std::log(m/std::sqrt(1.+sm2));
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
update_Fx - Checked. OK.
*/
//' @export
// [[Rcpp::export]]
arma::mat update_Fx(
	const unsigned int TransferCode,
	const unsigned int n, // number of observations
	const arma::vec& Y, // n x 1, (y[1],...,y[n]), observtions
	const arma::vec& hph, // (n+1) x 1, derivative of the gain function at h[t]
	const double rho = 0.99, // memory of previous state, for exponential and negative binomial transmission delay kernel
	const unsigned int L = 1., // length of nonzero transmission delay for log-normal
	const Rcpp::Nullable<Rcpp::NumericVector>& Fphi_ = R_NilValue) { // L x 1, (phi[1],...,phi[L]), for log-normal transmission delay kernel only

	arma::mat Fx(n,n,arma::fill::zeros);
	unsigned int tmpi;
	double tmpd;

	arma::vec Ypad(n+1,arma::fill::zeros);
	Ypad.tail(n) = Y; // Ypad = (Y[0]=0,Y[1],...,Y[n])

	switch (TransferCode) {
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
			double coef = 0.;
			for (unsigned int t=1; t<=n; t++) { // t-1 is the row index in cpp, theta[t]
				tmpd = 1.;
				coef = static_cast<double>(t-t+1);
				Fx.at(t-1,t-1) = coef*tmpd*Ypad.at(t-1)*hph.at(t);
				for (unsigned int i=(t-1); i>=1; i--) { 
					// i-1 is the column index in cpp, w[i]
					tmpd *= rho;
					coef = static_cast<double>(t-i+1);
					Fx.at(t-1,i-1) = Fx.at(t-1,i) + coef*tmpd*Ypad.at(i-1)*hph.at(i);
				}
			}
			double coef2 = (1.-rho) * (1.-rho);
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



//' @export
// [[Rcpp::export]]
arma::vec update_theta0(
	const unsigned int ModelCode,
	const unsigned int n, // number of observations
	const arma::vec& Y, // n x 1, observations, (Y[1],...,Y[n])
	const arma::vec& hhat, // (n+1) x 1, 1st Taylor expansion of h(psi[t]) at h[t]
	const double theta00 = 0., // Initial value of the transfer function block for Vanilla or Koyck
	const double rho = 0.99, // memory of previous state, for exponential and negative binomial transmission delay kernel
	const unsigned int L = 1., // length of nonzero transmission delay for log-normal
	const Rcpp::Nullable<Rcpp::NumericVector>& Fphi_ = R_NilValue) { // L x 1
	
	const unsigned int TransferCode = get_transcode(ModelCode);
	arma::vec theta0(n,arma::fill::zeros);
	unsigned int tmpi;
	double tmpd;

	arma::vec Ypad(n+1,arma::fill::zeros);
	Ypad.tail(n) = Y; // Ypad = (Y[0]=0,Y[1],...,Y[n])

	if (ModelCode==0) {
		// KoyamaMax
		theta0.zeros();
	} else {
		switch (TransferCode) {
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
					// Fh.head(tmpi) = arma::reverse(ht.subvec(r-tmpi+2,r+1));
					// First-order Taylor expansion of h(h[t])
					Fh.head(tmpi) = arma::reverse(hhat.subvec(r-tmpi+2,r+1));
            		// if (r>1) {
                	// 	Fy.head(tmpi) = arma::reverse(Y.subvec(r-tmpi,r-1));
					// 	// Fh.head(tmpi) = arma::reverse(ht.subvec(r-tmpi+2,r+1));
					// 	Fh.head(tmpi) = arma::reverse(hh.subvec(r-tmpi+2,r+1)) + arma::reverse(hph.subvec(r-tmpi+2,r+1))%(psi0-ht.subvec(r-tmpi+2,r+1)); // First-order Taylor expansion of h(h[t])
                	// 	// if r = 2, the third row, it would be (y[1],y[0]) in C, aka (y[2],y[1]) in R; 2 total elements
                	// 	// if r = (L-1), the L row, it would be (y[L-2],...,y[0]) in C, aka (y[L-1],...,y[1]) in R; (L-1) total elements
                	// 	// if r = L, the (L+1) row, it would be (y[L-1],...,y[0]) in C, aka (y[L],...,y[1]) in R; L total elements
                	// 	// if r = n-1, the n row, it would be (y[n-2],...,y[n-L-1]) in C, aka (y[n-1],...,y[n-L])
            		// } else {
                	// 	Fy.at(0) = Y.at(0);
					// 	Fh.at(0) = ht.at(2);
                	// 	// if r = 1 (t=2), second row, it would be (y[0]) in C, aka (y[1]) in R
            		// }

            		theta0.at(r) = arma::accu(Fphi % Fy % Fh);
        		}
			}
			break;
			case 2: // Solow
			{
				double coef = 0.;
				double coef2 = (1.-rho)*(1.-rho);
				double tmpd2 = 0.;
				for (unsigned int t=1; t<=n; t++) { // t-1 is the row index in cpp, theta[t]
					tmpd = 1.;
					coef = static_cast<double>(t-t+1.);
					tmpd2 = hhat.at(t);
					theta0.at(t-1) = coef*tmpd*Ypad.at(t-1)*tmpd2;
					for (unsigned int i=(t-1); i>=1; i--) { // i-1 is the column index in cpp, w[i]
						tmpd *= rho;
						coef = static_cast<double>(t-i+1);
						tmpd2 = hhat.at(i);
						theta0.at(t-1) += coef*tmpd*Ypad.at(i-1)*tmpd2;
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
	const arma::vec& w,
    const unsigned int ModelCode = 0) { // n x 1
	
	arma::vec theta = Fx * w; // n x 1
	return theta; 
}


arma::vec update_lambda(
    const arma::vec& theta, // n x 1
    const arma::vec& lambda0, // n x 1
    const unsigned int ModelCode = 0,
    const bool ApproxModel = true) {
    
    arma::vec lambda;

    switch(ModelCode) {
        case 0:
            if (ApproxModel) {
                lambda = lambda0 + theta;
                lambda.elem(arma::find(lambda<arma::datum::eps)).fill(arma::datum::eps);
            } else {
                ::Rf_error("Undefined for ModelCode=0 and ApproxModel=false.");
            }
            break;
        default:
            ::Rf_error("Undefined ModelCode!=0.");
            break;
    }

    return lambda;
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
double test_postW_gamma(){
	coef_W coef[1] = {{-5.,1.e2,0.5}};
	double What = optimize_postW_gamma(coef[0]);
	return What;
}


/*
Return the minimum value of alpha such that
	sum_{l=1}^{ntrunc} \phi_{l}^{alpha} = 1/M.

We search alpha in a linearly spaced grid {alpha_min:prec:alpha_max}
*/

//' @export
// [[Rcpp::export]]
double calc_power_sum(
	const double rho, // rho for the negative-binomial distribution
	const double M, // upper bound of the gain function h(.)
	const unsigned int TransferCode, // 0 - Koyck; 1 - Koyama; 2 - Solow; 3 - Vanilla
	const double alpha_min = 1., // lower bound of the power
	const double alpha_max = 2., // upper bound of the power
	const double prec = 1.e-4, // precision
	const unsigned int ntrunc=30) {

	const double mu = 2.2204e-16;
    const double m = 4.7;
    const double s = 2.9;

	arma::vec alpha_grid = arma::regspace(alpha_min,prec,alpha_max);
	const unsigned int nalpha = alpha_grid.n_elem;
	arma::vec pfunc(nalpha,arma::fill::zeros);

	arma::vec phi(ntrunc,arma::fill::zeros);
	switch (TransferCode) {
		case 0: // Koyck
		{
			for (unsigned int i=0; i<ntrunc; i++) {
				phi.at(i) = std::pow(rho,static_cast<double>(i)+1.);
			}
		}
		break;
		case 1: // Koyama
		{
			phi = get_Fphi(ntrunc,mu,m,s);
		}
		break;
		case 2: // Solow
		{
			for (unsigned int i=0; i<ntrunc; i++) {
				phi.at(i) = std::pow(rho,static_cast<double>(i)+1.)*(static_cast<double>(i)+1.);
			}
			phi *= (1.-rho) * (1.-rho);
		}
		break;
		default:
		{
			::Rf_error("Unsupported distributed lag kernel.");
		}
	}

	for (unsigned int i=0; i<nalpha; i++) {
		pfunc.at(i) = arma::accu(arma::pow(phi,alpha_grid.at(i)));
	}

	pfunc = arma::abs(pfunc - 1./M);

	return alpha_grid.at(pfunc.index_min());
}



//' @export
// [[Rcpp::export]]
Rcpp::List calc_power_sum2(
	const double rho, // rho for the negative-binomial distribution
	const double M, // upper bound of the gain function h(.)
	const unsigned int TransferCode, // 0 - Koyck; 1 - Koyama; 2 - Solow; 3 - Vanilla
	const double alpha_min = 1., // lower bound of the power
	const double alpha_max = 2., // upper bound of the power
	const double prec = 1.e-4, // precision
	unsigned int ntrunc=30) {

	const double mu = 2.2204e-16;
    const double m = 4.7;
    const double s = 2.9;

	arma::vec alpha_grid = arma::regspace(alpha_min,prec,alpha_max);
	const unsigned int nalpha = alpha_grid.n_elem;
	arma::vec pfunc(nalpha,arma::fill::zeros);

	arma::vec phi(ntrunc,arma::fill::zeros);
	switch (TransferCode) {
		case 0: // Koyck
		{
			for (unsigned int i=0; i<ntrunc; i++) {
				phi.at(i) = std::pow(rho,static_cast<double>(i)+1.);
			}
		}
		break;
		case 1: // Koyama
		{
			phi = get_Fphi(ntrunc,mu,m,s);
		}
		break;
		case 2: // Solow
		{
			for (unsigned int i=0; i<ntrunc; i++) {
				phi.at(i) = std::pow(rho,static_cast<double>(i)+1.)*(static_cast<double>(i)+1.);
			}
			phi *= (1.-rho) * (1.-rho);
		}
		break;
		default:
		{
			::Rf_error("Unsupported distributed lag kernel.");
		}
	}
	

	for (unsigned int i=0; i<nalpha; i++) {
		pfunc.at(i) = arma::accu(arma::pow(phi,alpha_grid.at(i)));
	}

	Rcpp::List output;
	output["alpha"] = Rcpp::wrap(alpha_grid);
	output["psum"] = Rcpp::wrap(pfunc);

	pfunc = arma::abs(pfunc - 1./M);
	double alpha_equal = alpha_grid.at(pfunc.index_min());
	output["alpha_equal"] = alpha_equal;
	output["phi"] = Rcpp::wrap(arma::pow(phi,alpha_equal));

	return output;
}



//' @export
// [[Rcpp::export]]
Rcpp::List calc_power_sum3(
	const double rho, // rho for the negative-binomial distribution
	const double M, // upper bound of the gain function h(.)
	unsigned int ntrunc=30) {

	double alpha_equal = 1. - 0.5*std::log(M)/std::log(1.-rho);

	arma::vec phi(ntrunc,arma::fill::zeros);
	for (unsigned int i=0; i<ntrunc; i++) {
		phi.at(i) = std::pow(rho,static_cast<double>(i)+1.)*(static_cast<double>(i)+1.);
	}
	phi *= std::pow(1.-rho, alpha_equal);
	
	Rcpp::List output;
	output["alpha_equal"] = alpha_equal;
	output["phi"] = Rcpp::wrap(phi);

	return output;
}



//' @export
// [[Rcpp::export]]
arma::mat psi2hpsi(
	const arma::mat& psi,
	const unsigned int GainCode,
	const Rcpp::NumericVector& coef = Rcpp::NumericVector::create(0.3,0,1)) {
	
	const double EPS = arma::datum::eps;
	const double UPBND = 700.;


	arma::mat hpsi;
	hpsi.copy_size(psi);

	switch (GainCode) {
		case 0: // Ramp
		{
			hpsi = psi;
			hpsi.elem(arma::find(psi<EPS)).fill(0.);
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
			hpsi = psi;
			hpsi.elem(arma::find(hpsi>UPBND)).fill(UPBND);
			hpsi = arma::log(1. + arma::exp(hpsi));
		}
		break;
		case 4: // Hyperbolic Tangent
		{
			hpsi = 0.5*coef[2] * (arma::tanh(coef[0]*psi + coef[1]) + 1.);
		}
		break;
		case 5: // Logistic
		{
			hpsi = -coef[0] * (psi-coef[1]);
			hpsi.elem(arma::find(hpsi>UPBND)).fill(UPBND);
			hpsi = coef[2] / (1. + arma::exp(hpsi));
		}
		break;
		default:
		{
			::Rf_error("Not supported gain function.");
		}
	}


	return hpsi;
}



double psi2hpsi(
	const double psi,
	const unsigned int GainCode,
	const Rcpp::NumericVector& coef = Rcpp::NumericVector::create(0.3,0,1)) {
	
	const double EPS = arma::datum::eps;
	const double UPBND = 700.;
	double hpsi;

	switch (GainCode) {
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
		case 4: // Hyperbolic Tangent
		{
			hpsi = 0.5*coef[2] * (std::tanh(coef[0]*psi + coef[1]) + 1.);
		}
		break;
		case 5: // Logistic
		{
			hpsi = coef[2]/(1. + std::exp(std::min(-coef[0]*(psi-coef[1]),UPBND)));
		}
		break;
		default:
		{
			::Rf_error("Not supported gain function.");
		}
	}


	return hpsi;
}




void hpsi_deriv(
	arma::mat & hpsi,
	const arma::mat& psi,
	const unsigned int GainCode,
	const Rcpp::NumericVector& coef = Rcpp::NumericVector::create(0.3,0,1)) {
	
	const double UPBND = 700.;

	hpsi.copy_size(psi);

	switch (GainCode) {
		case 0: // Ramp
		{
			::Rf_error("Ramp function is non-differentiable.");
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
			hpsi.ones();
		}
		break;
		case 3: // Softplus
		{
			hpsi = -psi;
			hpsi.elem(arma::find(hpsi>UPBND)).fill(UPBND);
			hpsi = 1./(1.+arma::exp(hpsi));
		}
		break;
		case 4: // Hyperbolic Tangent
		{
			hpsi = coef[0]*psi + coef[1];
			hpsi = 0.5*coef[0]*coef[2]/arma::pow(arma::cosh(hpsi),2.);
		}
		break;
		case 5: // Logistic
		{
			arma::vec c = Rcpp::as<arma::vec>(coef);
			hpsi = -coef[0] * (psi-coef[1]);
			hpsi.elem(arma::find(hpsi>UPBND)).fill(UPBND);
			hpsi = coef[2] / (1. + arma::exp(hpsi));
			hpsi = (coef[0]/coef[2])*(hpsi%(coef[2] - hpsi));
		}
		break;
		default:
		{
			::Rf_error("Not supported gain function.");
		}
	}


	return;
}


double hpsi_deriv(
	const double psi,
	const unsigned int GainCode,
	const Rcpp::NumericVector& coef = Rcpp::NumericVector::create(0.3,0,1)) {
	
	const double UPBND = 700.;
	double hpsi;

	switch (GainCode) {
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
		case 4: // Hyperbolic Tangent
		{
			hpsi = 0.5*coef[0]*coef[2]/std::pow(std::cosh(coef[0]*psi+coef[1]),2);
		}
		break;
		case 5: // Logistic
		{
			hpsi = coef[2]/(1.+std::exp(std::min(-coef[0]*(psi-coef[1]),UPBND)));
			hpsi = (coef[0]/coef[2])*(hpsi*(coef[2] - hpsi));
		}
		break;
		default:
		{
			::Rf_error("Not supported gain function.");
		}
	}


	return hpsi;
}



//' @export
// [[Rcpp::export]]
arma::mat hpsi_deriv(
	const arma::mat& psi,
	const unsigned int GainCode,
	const Rcpp::NumericVector& coef = Rcpp::NumericVector::create(0.3,0,1)) {
	
	const double UPBND = 700.;
	arma::mat hpsi;
	hpsi.copy_size(psi);

	switch (GainCode) {
		case 0: // Ramp
		{
			::Rf_error("Ramp function is non-differentiable.");
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
			hpsi.ones();
		}
		break;
		case 3: // Softplus
		{
			hpsi = -psi;
			hpsi.elem(arma::find(hpsi>UPBND)).fill(UPBND);
			hpsi = 1./(1.+arma::exp(hpsi));
		}
		break;
		case 4: // Hyperbolic Tangent
		{
			hpsi = coef[0]*psi + coef[1];
			hpsi = 0.5*coef[0]*coef[2]/arma::pow(arma::cosh(hpsi),2.);
		}
		break;
		case 5: // Logistic
		{
			arma::vec c = Rcpp::as<arma::vec>(coef);
			hpsi = -coef[0] * (psi-coef[1]);
			hpsi.elem(arma::find(hpsi>UPBND)).fill(UPBND);
			hpsi = coef[2] / (1. + arma::exp(hpsi));
			hpsi = (coef[0]/coef[2])*(hpsi%(coef[2] - hpsi));
		}
		break;
		default:
		{
			::Rf_error("Not supported gain function.");
		}
	}


	return hpsi;
}





//' @export
// [[Rcpp::export]]
arma::mat hpsi2theta(
	const arma::mat& hpsi, // (n+1) x k, each row is a different time point
	const arma::vec& y, // n x 1
	const unsigned int TransferCode,
	const double theta0 = 0.,
	const double alpha = 1.,
	const unsigned int L = 0,
	const double rho = 0.9) {
	
	const unsigned int n = y.n_elem;
	const unsigned int k = hpsi.n_cols;
	arma::mat theta(n,k,arma::fill::zeros);

	switch(TransferCode) {
		case 0: // Koyck
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
			double mu = 2.2204e-16;
			double m = 4.7;
			double s = 2.9;
			arma::vec Fphi = get_Fphi(L,mu,m,s);
        	Fphi = arma::pow(Fphi,alpha);
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
			double c1 = 2.*rho;
			double c2 = rho*rho;
			double c3 = std::pow((1.-rho)*(1.-rho),alpha);
			theta.row(0).fill(2.*rho*theta0);
			theta.row(1) = c1*theta.row(0) - c2*theta0 + c3*hpsi.row(1)*y.at(0);
			for (unsigned int t=2; t<n; t++) {
				theta.row(t) = c1*theta.row(t-1) - c2*theta.row(t-2) + c3*hpsi.row(t)*y.at(t-1);
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



double loglike_obs(
	const double y, 
	const double lambda,
	const unsigned int obs_type = 1,
	const double delta_nb = 1.,
	const bool return_log = false) {
	
	double loglike;
	if (obs_type == 0) {
        /*
        Negative-binomial likelihood
        - mean: lambda.at(i)
        - delta_nb: degree of over-dispersion

        sample variance exceeds the sample mean
        */
        loglike = R::lgammafn(y+delta_nb) - R::lgammafn(y+1.) - R::lgammafn(delta_nb) + delta_nb*(std::log(delta_nb)-std::log(delta_nb+lambda)) + y*(std::log(lambda)-std::log(delta_nb+lambda));
		if (!return_log) {
			loglike = std::exp(loglike);
		}
                
    } else if (obs_type == 1) {
        /*
        Poisson likelihood
        - mean: lambda.at(i)
        - var: lambda.at(i)

        sample variance == sample mean
        */
        loglike = R::dpois(y,lambda,return_log);
    }
	return loglike;
}