#include "lbe_poisson.h"
using namespace Rcpp;
// [[Rcpp::plugins(cpp17)]]
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
    const arma::uvec model_code,
    const arma::vec& W_sample, // nsample x 1 for LBE or HVB, or 1 x 1 for MCS
    const arma::mat& theta_last, // p x nsample for MCS or HVB, or p x 1 for LBE
    const Rcpp::Nullable<Rcpp::NumericMatrix>& Ct_last = R_NilValue, // p x p for LBE
    const Rcpp::Nullable<Rcpp::NumericVector>& qProb_ = R_NilValue,
    const double rho = 0.9,
    const unsigned int L = 2,
    const double mu0 = 0.,
    const double delta_nb = 1.) {

    const unsigned int n = Y.n_elem; // number of observed counts
    const unsigned int npred = n+m;

    arma::vec qProb;
    if (qProb_.isNull()) {
        qProb = {0.025,0.5,0.975};
    } else {
        qProb = Rcpp::as<arma::vec>(qProb_);
    }


    const unsigned int obs_code = model_code.at(0);
    const unsigned int link_code = model_code.at(1);
    const unsigned int trans_code = model_code.at(2);
    const unsigned int gain_code = model_code.at(3);
    const unsigned int err_code = model_code.at(4);
	unsigned int p, L_;
    arma::vec Ft, Fphi;
    arma::mat Gt;
    init_by_trans(p,L_,Ft,Fphi,Gt,trans_code,L);
	arma::vec Fy(L,arma::fill::zeros);


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
            theta_pred.col(t) = update_at(p,gain_code,trans_code,theta_pred.col(t-1),Gt,ypred.at(t-1),rho);
            theta_pred.at(0,t) += wt.at(t);

            // Link - phi
            psi_stored.at(t-n,i) = theta_pred.at(0,t);

            if (trans_code == 1 && L>0) { // Koyama
                update_Ft_koyama(Ft, Fy, t, L_, ypred, Fphi);
                switch (gain_code) {
				    case 0: // Ramp
				    {
					    theta_pred.elem(arma::find(theta_pred<EPS)).fill(EPS);
                        phi = arma::accu(Ft % theta_pred.col(t));
				    }
				    break;
				    case 1: // Exponential
				    {
					    theta_pred.elem(arma::find(theta_pred>UPBND)).fill(UPBND);
					    Ft = Ft % arma::exp(theta_pred.col(t));
					    phi = arma::accu(Ft);
				    }
				    break;
				    case 2: // Identity
				    {
					    phi = arma::accu(Ft % theta_pred.col(t));
				    }
				    break;
				    case 3: // Softplus
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
            if (link_code==1) {
                // Exponential link
                lambda = std::exp(phi);
            } else {
                // Identity link
                lambda = phi;
            }

            lambda_stored.at(t-n,i) = lambda;

            // Observation - y
            if (obs_code==0) {
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