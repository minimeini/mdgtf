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
//' @export
// [[Rcpp::export]]
double knl(
    const double t,
    const double mu,
    const double sd2) {
    return Pd(t,mu,sd2) - Pd(t-1.,mu,sd2);
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

	const double mu = 2.2204e-16;
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




//' @export
// [[Rcpp::export]]
arma::mat update_Fx(
	const unsigned int ModelCode,
	const unsigned int n, // number of observations
	const arma::vec& Y, // n x 1, (y[1],...,y[n]), observtions
	const double rho = 0.99, // memory of previous state, for exponential and negative binomial transmission delay kernel
	const unsigned int L = 1., // length of nonzero transmission delay for log-normal
	const Rcpp::Nullable<Rcpp::NumericVector>& ht_ = R_NilValue, // (n+1) x 1, (h[0],h[1],...,h[n]), smoothing means of (psi[0],...,psi[n]) | Y
	const Rcpp::Nullable<Rcpp::NumericVector>& Fphi_ = R_NilValue) { // L x 1, (phi[1],...,phi[L]), for log-normal transmission delay kernel only

	arma::mat Fx(n,n,arma::fill::zeros);
	unsigned int tmpi;
	double tmpd;

	arma::vec ht(n+1,arma::fill::zeros); // (h[0],h[1],...,h[n])
	if (!ht_.isNull()) {
		ht = Rcpp::as<arma::vec>(ht_);
	}

	arma::vec Ypad(n+1,arma::fill::zeros);
	Ypad.tail(n) = Y; // Ypad = (Y[0]=0,Y[1],...,Y[n])

	switch (ModelCode) {
		case 0: // KoyamaMax [Checked - Equal to KoyamaEye if ht[t] > 0 for all t=0,1,...,n.]
		{
			arma::vec Fphi(L,arma::fill::ones);
			if (!Fphi_.isNull()) {
				Fphi = Rcpp::as<arma::vec>(Fphi_);
			} else {
				Fphi = get_Fphi(L,2.2204e-16,4.7,2.9);
			}

			// Fill out the first two columns, starting from the second row of Fx
			arma::vec Fy(L,arma::fill::zeros);
			arma::vec Fh(L,arma::fill::zeros);
        	for (unsigned int r=1; r<n; r++) { 
            	// loop through row, r from 1 to n-1, aka time from 2 to n.
            	tmpi = std::min(L,r); // number of nonzero terms
            	Fy.zeros();
				Fh.zeros();
            	if (r>1) {
                	Fy.head(tmpi) = arma::reverse(Y.subvec(r-tmpi,r-1));
					Fh.head(tmpi) = arma::reverse(ht.subvec(r-tmpi+2,r+1));
                	// if r = 2, the third row, it would be (y[1],y[0]) in C, aka (y[2],y[1]) in R; 2 total elements
                	// if r = (L-1), the L row, it would be (y[L-2],...,y[0]) in C, aka (y[L-1],...,y[1]) in R; (L-1) total elements
                	// if r = L, the (L+1) row, it would be (y[L-1],...,y[0]) in C, aka (y[L],...,y[1]) in R; L total elements
                	// if r = n-1, the n row, it would be (y[n-2],...,y[n-L-1]) in C, aka (y[n-1],...,y[n-L])
            	} else {
                	Fy.at(0) = Y.at(0);
					Fh.at(0) = ht.at(2);
                	// if r = 1 (t=2), second row, it would be (y[0]) in C, aka (y[1]) in R
            	}

				// Ramp function
				Fh = arma::conv_to<arma::vec>::from(Fh > arma::datum::eps);

            	tmpd = arma::accu(Fphi % Fy % Fh);
            	Fx.at(r,0) = tmpd;
            	Fx.at(r,1) = tmpd;
            	// Fill out the rest of the columns until the diagonal element
            	if (r>1) { // start from the third row
                	for (unsigned int c=2; c<=r; c++) { // loop through columns until the diagonal
                    	tmpi = std::min(L,r-c+1);
                    	Fx.at(r,c) = arma::accu(Fphi.head(tmpi) % Fy.head(tmpi) % Fh.head(tmpi));
                	}
            	}
        	}
		}
		break;
		case 1: // KoyamaExp [Checked - Equal to KoyamaEye if ht[t] == 0 for all t=0,1,...,n.]
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
            	if (r>1) {
                	Fy.head(tmpi) = arma::reverse(Y.subvec(r-tmpi,r-1));
					Fh.head(tmpi) = arma::reverse(ht.subvec(r-tmpi+2,r+1));
                	// if r = 2, the third row, it would be (y[1],y[0]) in C, aka (y[2],y[1]) in R; 2 total elements
                	// if r = (L-1), the L row, it would be (y[L-2],...,y[0]) in C, aka (y[L-1],...,y[1]) in R; (L-1) total elements
                	// if r = L, the (L+1) row, it would be (y[L-1],...,y[0]) in C, aka (y[L],...,y[1]) in R; L total elements
                	// if r = n-1, the n row, it would be (y[n-2],...,y[n-L-1]) in C, aka (y[n-1],...,y[n-L])
            	} else {
                	Fy.at(0) = Y.at(0);
					Fh.at(0) = ht.at(2);
                	// if r = 1 (t=2), second row, it would be (y[0]) in C, aka (y[1]) in R
            	}

				Fh = arma::exp(Fh);

            	tmpd = arma::accu(Fphi % Fy % Fh);
            	Fx.at(r,0) = tmpd;
            	Fx.at(r,1) = tmpd;
            	// Fill out the rest of the columns until the diagonal element
            	if (r>1) { // start from the third row
                	for (unsigned int c=2; c<=r; c++) { // loop through columns until the diagonal
                    	tmpi = std::min(L,r-c+1);
                    	Fx.at(r,c) = arma::accu(Fphi.head(tmpi) % Fy.head(tmpi) % Fh.head(tmpi));
                	}
            	}
        	}
		}
		break;
		case 2: // SolowMax [Checked - Equal to SolowEye if ht[t] > 0 for all t=0,1,...,n.]
		{
			double coef = 0.;
			double delta = 0.;
			for (unsigned int t=1; t<=n; t++) { // t-1 is the row index in cpp, theta[t]
				tmpd = 1.;
				coef = static_cast<double>(t-t+1);
				delta = static_cast<double>(ht.at(t)>arma::datum::eps);
				Fx.at(t-1,t-1) = coef*tmpd*Ypad.at(t-1)*delta;
				for (unsigned int i=(t-1); i>=1; i--) { // i-1 is the column index in cpp, w[i]
					tmpd *= rho;
					coef = static_cast<double>(t-i+1);
					delta = static_cast<double>(ht.at(i)>arma::datum::eps);
					Fx.at(t-1,i-1) = Fx.at(t-1,i) + coef*tmpd*Ypad.at(i-1)*delta;
				}
			}
		}
		break;
		case 3: // SolowExp [Checked - Equal to SolowEye if ht[t] == 0 for all t=0,1,...,n.]
		{
			double coef = 0.;
			for (unsigned int t=1; t<=n; t++) { // t-1 is the row index in cpp, theta[t]
				tmpd = 1.;
				coef = static_cast<double>(t-t+1);
				Fx.at(t-1,t-1) = coef*tmpd*Ypad.at(t-1)*std::exp(ht.at(t));
				for (unsigned int i=(t-1); i>=1; i--) { // i-1 is the column index in cpp, w[i]
					tmpd *= rho;
					coef = static_cast<double>(t-i+1);
					Fx.at(t-1,i-1) = Fx.at(t-1,i) + coef*tmpd*Ypad.at(i-1)*std::exp(ht.at(i));
				}
			}
		}
		break;
		case 4: // KoyckMax [Checked - Equal to KoyckEye if ht[t] > 0 for all t=0,1,...,n.]
		{
			double delta = 0.;
			for (unsigned int t=1; t<=n; t++) { // t-1 is the row index in cpp, theta[t]
				tmpd = 1.;
				delta = static_cast<double>(ht.at(t)>arma::datum::eps);
				Fx.at(t-1,t-1) = tmpd*Ypad.at(t-1)*delta;
				for (unsigned int i=(t-1); i>=1; i--) { // i-1 is the column index in cpp, w[i]
					tmpd *= rho;
					delta = static_cast<double>(ht.at(i)>arma::datum::eps);
					Fx.at(t-1,i-1) = Fx.at(t-1,i) + tmpd*Ypad.at(i-1)*delta;
				}
			}
		}
		break;
		case 5: // KoyckExp [Checked - Equal to KoyckEye if ht[t] == 0 for all t=0,1,...,n.]
		{
			for (unsigned int t=1; t<=n; t++) { // t-1 is the row index in cpp, theta[t]
				tmpd = 1.;
				Fx.at(t-1,t-1) = tmpd*Ypad.at(t-1)*std::exp(ht.at(t));
				for (unsigned int i=(t-1); i>=1; i--) { // i-1 is the column index in cpp, w[i]
					tmpd *= rho;
					Fx.at(t-1,i-1) = Fx.at(t-1,i) + tmpd*Ypad.at(i-1)*std::exp(ht.at(i));
				}
			}
		}
		break;
		case 6: // [Same as update_Fx_Koyama] KoyamaEye
		{
			arma::vec Fphi(L,arma::fill::ones);
			if (!Fphi_.isNull()) {
				Fphi = Rcpp::as<arma::vec>(Fphi_);
			}
			
			// Fill out the first two columns, starting from the second row of Fx
			arma::vec Fy(L,arma::fill::zeros);
        	for (unsigned int r=1; r<n; r++) { 
            	// loop through row, r from 1 to n-1, aka time from 2 to n.
            	tmpi = std::min(L,r); // number of nonzero terms
            	Fy.zeros();
            	if (r>1) {
                	Fy.head(tmpi) = arma::reverse(Y.subvec(r-tmpi,r-1));
                	// if r = 2, the third row, it would be (y[1],y[0]) in C, aka (y[2],y[1]) in R; 2 total elements
                	// if r = (L-1), the L row, it would be (y[L-2],...,y[0]) in C, aka (y[L-1],...,y[1]) in R; (L-1) total elements
                	// if r = L, the (L+1) row, it would be (y[L-1],...,y[0]) in C, aka (y[L],...,y[1]) in R; L total elements
                	// if r = n-1, the n row, it would be (y[n-2],...,y[n-L-1]) in C, aka (y[n-1],...,y[n-L])
            	} else {
                	Fy.at(0) = Y.at(0);
                	// if r = 1 (t=2), second row, it would be (y[0]) in C, aka (y[1]) in R
            	}

            	tmpd = arma::accu(Fy % Fphi);
            	Fx.at(r,0) = tmpd;
            	Fx.at(r,1) = tmpd;
            	// Fill out the rest of the columns until the diagonal element
            	if (r>1) { // start from the third row
                	for (unsigned int c=2; c<=r; c++) { // loop through columns until the diagonal
                    	tmpi = std::min(L,r-c+1);
                    	Fx.at(r,c) = arma::accu(Fphi.head(tmpi) % Fy.head(tmpi));
                	}
            	}
        	}
		}
		break;
		case 7: // [Correct] SolowEye
		{
			double coef = 0.;
			for (unsigned int t=1; t<=n; t++) { // t-1 is the row index in cpp, theta[t]
				tmpd = 1.;
				coef = static_cast<double>(t-t+1);
				Fx.at(t-1,t-1) = coef*tmpd*Ypad.at(t-1);
				for (unsigned int i=(t-1); i>=1; i--) { // i-1 is the column index in cpp, w[i]
					tmpd *= rho;
					coef = static_cast<double>(t-i+1);
					Fx.at(t-1,i-1) = Fx.at(t-1,i) + coef*tmpd*Ypad.at(i-1);
				}
			}
		}
		break;
		case 8: // [Correct] KoyckEye
		{
			for (unsigned int t=1; t<=n; t++) { // t-1 is the row index in cpp, theta[t]
				tmpd = 1.;
				Fx.at(t-1,t-1) = tmpd*Ypad.at(t-1);
				for (unsigned int i=(t-1); i>=1; i--) { // i-1 is the column index in cpp, w[i]
					tmpd *= rho;
					Fx.at(t-1,i-1) = Fx.at(t-1,i) + tmpd*Ypad.at(i-1);
				}
			}
		}
		break;
		case 9: // Vanilla
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

	if (ModelCode == 2 || ModelCode == 3 || ModelCode == 7) {
		// Solows
		double coef2 = (1.-rho) * (1.-rho);
		Fx.for_each( [&coef2](arma::mat::elem_type& val) {val *= coef2;});
	}

	return Fx;
}



//' @export
// [[Rcpp::export]]
arma::vec update_theta0(
	const unsigned int ModelCode,
	const unsigned int n, // number of observations
	const arma::vec& Y, // n x 1, observations, (Y[1],...,Y[n])
	const double theta00 = 0., // Initial value of the transfer function block
	const double psi0 = 0., // Initial value of the evolution error
	const double rho = 0.99, // memory of previous state, for exponential and negative binomial transmission delay kernel
	const unsigned int L = 1., // length of nonzero transmission delay for log-normal
	const Rcpp::Nullable<Rcpp::NumericVector>& ht_ = R_NilValue, // (n+1) x 1, (h[0],h[1],...,h[n]), smoothing means of (psi[0],...,psi[n]) | Y
	const Rcpp::Nullable<Rcpp::NumericVector>& Fphi_ = R_NilValue) { //  L x 1
	
	arma::vec theta0(n,arma::fill::zeros);
	unsigned int tmpi;
	double tmpd;

	arma::vec ht(n+1,arma::fill::zeros); // (h[0],h[1],...,h[n])
	if (!ht_.isNull()) {
		ht = Rcpp::as<arma::vec>(ht_);
	}

	arma::vec Ypad(n+1,arma::fill::zeros);
	Ypad.tail(n) = Y; // Ypad = (Y[0]=0,Y[1],...,Y[n])

	switch (ModelCode) {
		case 0: // KoyamaMax
		{
			theta0.zeros();
		}
		break;
		case 1: // KoyamaExp
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
            	if (r>1) {
                	Fy.head(tmpi) = arma::reverse(Y.subvec(r-tmpi,r-1));
					Fh.head(tmpi) = arma::reverse(ht.subvec(r-tmpi+2,r+1));
                	// if r = 2, the third row, it would be (y[1],y[0]) in C, aka (y[2],y[1]) in R; 2 total elements
                	// if r = (L-1), the L row, it would be (y[L-2],...,y[0]) in C, aka (y[L-1],...,y[1]) in R; (L-1) total elements
                	// if r = L, the (L+1) row, it would be (y[L-1],...,y[0]) in C, aka (y[L],...,y[1]) in R; L total elements
                	// if r = n-1, the n row, it would be (y[n-2],...,y[n-L-1]) in C, aka (y[n-1],...,y[n-L])
            	} else {
                	Fy.at(0) = Y.at(0);
					Fh.at(0) = ht.at(2);
                	// if r = 1 (t=2), second row, it would be (y[0]) in C, aka (y[1]) in R
            	}

            	theta0.at(r) = arma::accu(Fphi % Fy % arma::exp(Fh) % (1.-Fh + psi0));
        	}
		}
		break;
		case 2: // SolowMax
		{
			theta0.zeros();
		}
		break;
		case 3: // SolowExp
		{
			double coef = 0.;
			double coef2 = (1.-rho)*(1.-rho);
			for (unsigned int t=1; t<=n; t++) { // t-1 is the row index in cpp, theta[t]
				tmpd = 1.;
				coef = static_cast<double>(t-t+1);
				theta0.at(t-1) = coef*tmpd*Ypad.at(t-1)*std::exp(ht.at(t))*(1.-ht.at(t)+psi0);
				for (unsigned int i=(t-1); i>=1; i--) { // i-1 is the column index in cpp, w[i]
					tmpd *= rho;
					coef = static_cast<double>(t-i+1);
					theta0.at(t-1) += coef*tmpd*Ypad.at(i-1)*std::exp(ht.at(i))*(1.-ht.at(i)+psi0);
				}
				theta0.at(t-1) *= coef2;
			}
		}
		break;
		case 4: // KoyckMax [Checked - Equal to KoyckEye if ht[t] > 0 for all t=0,1,...,n.]
		{
			tmpd = 1.;
			for (unsigned int t=1; t<=n; t++) { // t-1 is the row index in cpp, theta[t]
				tmpd *= rho;
				theta0.at(t-1) = tmpd*theta00;
			}
		}
		break;
		case 5: // KoyckExp [Checked - Equal to KoyckEye if ht[t] == 0 for all t=0,1,...,n.]
		{
			double tmpd2 = 1.;
			for (unsigned int t=1; t<=n; t++) { // t-1 is the row index in cpp, theta[t]
				tmpd = 1.;
				tmpd2 *= rho;
				theta0.at(t-1) = tmpd2*theta00 + tmpd*Ypad.at(t-1)*std::exp(ht.at(t))*(1.-ht.at(t)+psi0);
				for (unsigned int i=(t-1); i>=1; i--) { // i-1 is the column index in cpp, w[i]
					tmpd *= rho;
					theta0.at(t-1) += tmpd*Ypad.at(i-1)*std::exp(ht.at(i))*(1.-ht.at(i)+psi0);
				}
			}
		}
		break;
		case 6: // KoyamaEye
		{
			arma::vec Fphi(L,arma::fill::ones);
			if (!Fphi_.isNull()) {
				Fphi = Rcpp::as<arma::vec>(Fphi_);
			}

			// Fill out the first two columns, starting from the second row of Fx
			arma::vec Fy(L,arma::fill::zeros);
        	for (unsigned int r=1; r<n; r++) { 
            	// loop through row, r from 1 to n-1, aka time from 2 to n.
            	tmpi = std::min(L,r); // number of nonzero terms
            	Fy.zeros();
            	if (r>1) {
                	Fy.head(tmpi) = arma::reverse(Y.subvec(r-tmpi,r-1));
                	// if r = 2, the third row, it would be (y[1],y[0]) in C, aka (y[2],y[1]) in R; 2 total elements
                	// if r = (L-1), the L row, it would be (y[L-2],...,y[0]) in C, aka (y[L-1],...,y[1]) in R; (L-1) total elements
                	// if r = L, the (L+1) row, it would be (y[L-1],...,y[0]) in C, aka (y[L],...,y[1]) in R; L total elements
                	// if r = n-1, the n row, it would be (y[n-2],...,y[n-L-1]) in C, aka (y[n-1],...,y[n-L])
            	} else {
                	Fy.at(0) = Y.at(0);
                	// if r = 1 (t=2), second row, it would be (y[0]) in C, aka (y[1]) in R
            	}

            	theta0.at(r) = psi0 * arma::accu(Fphi % Fy);
        	}
		}
		break;
		case 7: // SolowEye
		{
			double coef = 0.;
			double coef2 = (1.-rho)*(1.-rho);
			for (unsigned int t=1; t<=n; t++) { // t-1 is the row index in cpp, theta[t]
				tmpd = 1.;
				coef = static_cast<double>(t-t+1);
				theta0.at(t-1) = coef*tmpd*Ypad.at(t-1)*psi0;
				for (unsigned int i=(t-1); i>=1; i--) { // i-1 is the column index in cpp, w[i]
					tmpd *= rho;
					coef = static_cast<double>(t-i+1);
					theta0.at(t-1) += coef*tmpd*Ypad.at(i-1)*psi0;
				}
				theta0.at(t-1) *= coef2;
			}
		}
		break;
		case 8: // KoyckEye
		{
			double tmpd2 = 1.;
			for (unsigned int t=1; t<=n; t++) { // t-1 is the row index in cpp, theta[t]
				tmpd = 1.;
				tmpd2 *= rho;
				theta0.at(t-1) = tmpd2*theta00 + tmpd*Ypad.at(t-1)*psi0;
				for (unsigned int i=(t-1); i>=1; i--) { // i-1 is the column index in cpp, w[i]
					tmpd *= rho;
					theta0.at(t-1) += tmpd*Ypad.at(i-1)*psi0;
				}
			}
		}
		break;
		case 9: // Vanilla
		{
			double tmpd2 = 1.;
			for (unsigned int t=1; t<=n; t++) { // t-1 is the row index in cpp, theta[t]
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


/*
------ KoyckEye with ModelCode == 8 and Y[t] == 1 ------
Koyck's exponential decaying kernel with no external input or y[t]==1.
*/
arma::mat update_Fx0(
	const unsigned int n,
	const double G) { // n x 1
	/*
	1           0       0        ...    0
	G           1       0        ...    0
	G^2         G       1        ...    0
	.			.		.				.
	.			.		.				.
	.			.		.				.
	G^(n-1)  G^(n-2)  G^(n-3)    ...    1
	*/
	arma::mat Fx(n,n,arma::fill::eye);
	for (unsigned int s=1; s<n; s++) { // i = 1,...,(n-1) in Cpp or i = 2,...,n in R
		for (unsigned int t=(s-1); t>0; t--) {
            Fx.at(s,t) = Fx.at(s,t+1) * G;
        }
        Fx.at(s,0) = Fx.at(s,1) * G;
	}
	return Fx;
}


/*
------ KoyckEye with ModelCode == 8 ------
Koyck's exponential transmission kernel using y[t-1] as input at time t
*/
arma::mat update_Fx1(
	const unsigned int n,
	const double G,
	const arma::vec& X) { // n x 1
	arma::mat Fx(n,n,arma::fill::zeros);
	double rho;
	// Fx.at(0,0) = X.at(0);
	Fx.at(0,0) = 0.;
	for (unsigned int s=1; s<n; s++) { // i = 1,...,(n-1) in Cpp or i = 2,...,n in R
		rho = 1.;
		Fx.at(s,s) = rho*X.at(s-1);
		for (unsigned int t=(s-1); t>0; t--) {
			rho *= G; // use iterative form instead of std::pow to speed up
            Fx.at(s,t) = Fx.at(s,t+1) + rho*X.at(t-1);
        }
		rho *= G;
        Fx.at(s,0) = Fx.at(s,1);
	}
	return Fx;
}


/*
------ SolowEye with ModelCode == 7 ------
Solow's negative-binomial transmission kernel using y[t-1] as input at time t
*/
arma::mat update_Fx_Solow(
	const unsigned int n,
	const double rho,
	const arma::vec& X) { // n x 1
	arma::mat Fx(n,n,arma::fill::zeros);
	double coef,idx_diff;
	// Fx.at(0,0) = X.at(0);
	Fx.at(0,0) = 0.;
	for (unsigned int s=1; s<n; s++) { // row i = 1,...,(n-1) in Cpp or i = 2,...,n in R
		coef = 1.;
		Fx.at(s,s) = coef*X.at(s-1);
		for (unsigned int t=(s-1); t>0; t--) { // column from right to left
			idx_diff = static_cast<double>(s-t);
			coef *= rho * (idx_diff+1.)/idx_diff; // use iterative form instead of std::pow to speed up
            Fx.at(s,t) = Fx.at(s,t+1) + coef*X.at(t-1);
        }
		idx_diff = static_cast<double>(s);
		coef *= rho * (idx_diff+1.)/idx_diff;
        // Fx.at(s,0) = Fx.at(s,1) + coef*X.at(0);
		Fx.at(s,0) = Fx.at(s,1);
	}
	Fx *= (1.-rho) * (1.-rho);
	return Fx;
}



/*
------ KoyamaEye with ModelCode == 0 ------
Solow's negative-binomial transmission kernel using y[t-1] as input at time t
*/
arma::mat update_Fx_Koyama(
	const unsigned int n, // number of observations
    const unsigned int L, // number of transmission delays
	const arma::vec& Fphi, // L x 1, Fphi = (phi[1],...,phi[L])
	const arma::vec& Y) { // n x 1, the observation (scalar), n: num of obs

	arma::mat Fx(n,n,arma::fill::zeros);
    // Fx.at(0,0) = X.at(0);
	Fx.at(0,0) = 0.; 

	double coef,idx_diff;
    double tmp;
    unsigned int tmpi;

    arma::vec Fy(L,arma::fill::zeros);

    // Fill out the first two columns, starting from the second row of Fx
    for (unsigned int r=1; r<n; r++) { 
        // loop through row, r from 1 to n-1, aka time from 2 to n.
        tmpi = std::min(L,r); // number of nonzero terms
        Fy.zeros();
        if (r>1) {
            Fy.head(tmpi) = arma::reverse(Y.subvec(r-tmpi,r-1));
            // if r = 2, the third row, it would be (y[1],y[0]) in C, aka (y[2],y[1]) in R; 2 total elements
            // if r = (L-1), the L row, it would be (y[L-2],...,y[0]) in C, aka (y[L-1],...,y[1]) in R; (L-1) total elements
            // if r = L, the (L+1) row, it would be (y[L-1],...,y[0]) in C, aka (y[L],...,y[1]) in R; L total elements
            // if r = n-1, the n row, it would be (y[n-2],...,y[n-L-1]) in C, aka (y[n-1],...,y[n-L])
        } else {
            Fy.at(0) = Y.at(0);
            // if r = 1 (t=2), second row, it would be (y[0]) in C, aka (y[1]) in R
        }

        tmp = arma::accu(Fy % Fphi);
        Fx.at(r,0) = tmp;
        Fx.at(r,1) = tmp;
        // Fill out the rest of the columns until the diagonal element
        if (r>1) { // start from the third row
            for (unsigned int c=2; c<=r; c++) { // loop through columns until the diagonal
                tmpi = std::min(L,r-c+1);
                Fx.at(r,c) = arma::accu(Fphi.head(tmpi) % Fy.head(tmpi));
            }
        }
    }

	return Fx;
}



double trigamma_obj(
	unsigned n,
	const double *x, 
	double *grad, 
	void *my_func_data) {

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