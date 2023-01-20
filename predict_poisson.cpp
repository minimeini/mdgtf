#include "lbe_poisson.h"
using namespace Rcpp;
// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::depends(RcppArmadillo)]]

/*
we need theta_tilde[n-1]
for psi[n-1], either mean+var, or all the samples; quantiles don't work
How to get W[n-1] if we use discount factor for LBE -> deriv the posterior distribution of W
*/
//' @export
// [[Rcpp::export]]
Rcpp::List predict_poisson(
    const arma::vec& Y, // n x 1
    const unsigned int m, // maximum step of forecasting
    const unsigned int nsample,
    const unsigned int ModelCode,
    const arma::vec& W_sample, // nsample x 1 for LBE or HVB, or 1 x 1 for MCS
    const arma::mat& theta_last, // p x nsample for MCS or HVB, or p x 1 for LBE
    const Rcpp::Nullable<Rcpp::NumericMatrix>& Ct_last = R_NilValue, // p x p for LBE
    const Rcpp::Nullable<Rcpp::NumericVector>& qProb_ = R_NilValue,
    const double rho = 0.9,
    const unsigned int L = 0,
    const double mu0 = 0.,
    const double delta_nb = 1.,
    const unsigned int obs_type = 1.) {

    const double UPBND = 700.;
    const double EPS = arma::datum::eps;

    const unsigned int n = Y.n_elem; // number of observed counts
    const unsigned int npred = n+m;

    arma::vec qProb;
    if (qProb_.isNull()) {
        qProb = {0.025,0.5,0.975};
    } else {
        qProb = Rcpp::as<arma::vec>(qProb_);
    }

    const bool is_solow = ModelCode == 2 || ModelCode == 3 || ModelCode == 7 || ModelCode == 12;
	const bool is_koyck = ModelCode == 4 || ModelCode == 5 || ModelCode == 8 || ModelCode == 10;
	const bool is_koyama = ModelCode == 0 || ModelCode == 1 || ModelCode == 6 || ModelCode == 11;
	const bool is_vanilla = ModelCode == 9;
	unsigned int TransferCode; // integer indicator for the type of transfer function
	unsigned int p; // dimension of DLM state space
	unsigned int L_;
    arma::mat Gt;
	if (is_koyck) { 
		TransferCode = 0; 
		p = 2;
		L_ = 0;
	} else if (is_koyama) { 
		TransferCode = 1; 
		p = L;
		L_ = L;
        Gt.set_size(p,p);
        Gt.zeros();
        Gt.at(0,0) = 1.;
        Gt.diag(-1).ones();
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


    arma::vec ypred(npred,arma::fill::zeros);
    ypred.head(n) = Y;

    // theta_last: p x nsample
    arma::mat theta_last_(p,nsample);
    if (theta_last.n_cols == 1) {
        arma::vec mt = arma::vectorise(theta_last);
        // arma::mat Ct_chol = arma::chol(Rcpp::as<arma::mat>(Ct_last));
        for (unsigned int i=0; i<nsample; i++) {
            theta_last_.col(i) = mt;
            // theta_last_.col(i) = mt + Ct_chol.t()*arma::randn(p);
        }
    } else {
        theta_last_ = theta_last;
    }

    // W_sample: nsample x 1
    arma::vec W_sample_(nsample);
    if (W_sample.n_elem == 1) {
        W_sample_.fill(W_sample.at(0));
    } else {
        W_sample_ = W_sample;
    }

    arma::vec wt(npred);
    arma::mat theta_pred(p,npred,arma::fill::zeros);
    double phi,lambda;

    arma::mat ypred_stored(m,nsample);
    arma::mat lambda_stored(m,nsample);
    arma::mat psi_stored(m,nsample);

    for (unsigned int i=0; i<nsample; i++) {
        theta_pred.col(n-1) = theta_last_.col(i);
        wt = arma::randn(npred,1) * std::sqrt(W_sample_.at(i));

        for (unsigned int t=n; t<npred; t++) {
            // state - theta, especially psi
            theta_pred.col(t) = update_at(p,ModelCode,TransferCode,theta_pred.col(t-1),Gt,ypred.at(t-1),rho,L_);
            theta_pred.at(0,t) += wt.at(t);

            // Link - phi
            psi_stored.at(t-n,i) = theta_pred.at(0,t);

            if (TransferCode == 1 && L>0) { // Koyama
                update_Ft(Ft, Fy, TransferCode, t, L_, ypred, Fphi);
			    switch (ModelCode) {
				    case 0: // KoyamaMax
				    {
					    theta_pred.elem(arma::find(theta_pred<EPS)).fill(EPS);
                        phi = arma::accu(Ft % theta_pred.col(t));
				    }
				    break;
				    case 1: // KoyamaExp
				    {
					    theta_pred.elem(arma::find(theta_pred>UPBND)).fill(UPBND);
					    Ft = Ft % arma::exp(theta_pred.col(t));
					    phi = arma::accu(Ft);
				    }
				    break;
				    case 6: // KoyamaEye
				    {
					    phi = arma::accu(Ft % theta_pred.col(t));
				    }
				    break;
				    case 11: // KoyamaSoftplus
				    {
					    theta_pred.elem(arma::find(theta_pred>UPBND)).fill(UPBND);
					    arma::vec theta_exp = arma::exp(theta_pred.col(t));
					    phi = arma::accu(Ft % arma::log(1. + theta_exp));
				    }
				    break;
				    default:
				    {
					    ::Rf_error("get_Ft function is only defined for Koyama transmission kernels.");
				    }
			    } // END switch block
		    } else { // Vanilla, Koyck, Solow
			    phi = arma::as_scalar(Ft.t() * theta_pred.col(t));
		    }

            phi += mu0;

            // Mean - lambda
            if (ModelCode==6||ModelCode==7||ModelCode==8||ModelCode==9) {
                // Exponential link
                lambda = std::exp(phi);
            } else {
                // Identity link
                lambda = phi;
            }

            lambda_stored.at(t-n,i) = lambda;

            // Observation - y
            if (obs_type == 0) {
                // negative-binomial
                ypred.at(t) = R::rnbinom(delta_nb,std::exp(std::log(delta_nb)-std::log(lambda+delta_nb)));
            } else {
                // Poisson
                ypred.at(t) = R::rpois(lambda);
            }

            ypred_stored.at(t-n,i) = ypred.at(t);
        }
    }
    
    Rcpp::List output;
    output["ypred"] = Rcpp::wrap(arma::quantile(ypred_stored,qProb,1));
    output["lambda"] = Rcpp::wrap(arma::quantile(lambda_stored,qProb,1));
    output["psi"] = Rcpp::wrap(arma::quantile(psi_stored,qProb,1));
    return output;
}