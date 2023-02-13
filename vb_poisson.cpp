#include "lbe_poisson.h"
using namespace Rcpp;
// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::depends(RcppArmadillo)]]






//' @export
// [[Rcpp::export]]
Rcpp::List vb_poisson(
    const arma::vec& Y, // n x 1, the observed response
    const unsigned int ModelCode,
    const arma::uvec& eta_select, // 4 x 1, indicator for unknown (=1) or known (=0), global parameters: W, mu0, rho, M
    const arma::vec& eta_init, // 4 x 1, if true/initial values should be provided here
    const arma::uvec& eta_prior_type, // 4 x 1
    const arma::mat& eta_prior_val, // 2 x 4, priors for each element of eta
    const unsigned int L = 0,
    const double delta = NA_REAL,
    const double alpha = 1.,
    const unsigned int niter = 5000, // number of iterations for variational inference
    const Rcpp::Nullable<Rcpp::NumericVector>& m0_prior = R_NilValue,
	const Rcpp::Nullable<Rcpp::NumericMatrix>& C0_prior = R_NilValue,
    const Rcpp::NumericVector& ctanh = Rcpp::NumericVector::create(0.3,0,1),
    const double aw_prior = 0.01, // aw: shape of inverse gamma
    const double bw_prior = 0.01, // bw: rate of inverse gamma
    const double delta_nb = 1.,
    const unsigned int obs_type = 1, // 0: NB; 1: Pois
    const bool use_smoothing = true,
    const bool verbose = true) { 

    const unsigned int n = Y.n_elem;
    const double n_ = static_cast<double>(n);
	const unsigned int npad = n+1;
    arma::vec Ypad(npad,arma::fill::zeros); // (n+1) x 1
	Ypad.tail(n) = Y;

	unsigned int TransferCode;
	unsigned int p; // dimension of DLM state space
	unsigned int L_;
    get_transcode(TransferCode,p,L_,ModelCode,L);



    /* ----- Hyperparameter and Initialization ----- */
    double W = eta_init.at(0);
    double aw,bw;
    double a1,a2,a3;
    switch (eta_prior_type.at(0)) {
        case 0: // Gamma
        {
            a1 = eta_prior_val.at(0,0) - 0.5*n_;
            a2 = eta_prior_val.at(1,0);
            a3 = 0.;
        }
        break;
        case 1: // Half-Cauchy
        {

        }
        break;
        case 2: // Inverse-Gamma
        {
            aw = eta_prior_val.at(0,0) + 0.5*n_;
            bw = eta_prior_val.at(1,0);
        }
        break;
        default:
        {

        }
    }

    double mu0 = eta_init.at(1);
    double rho = eta_init.at(2);
    

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
    
	forwardFilter(mt,at,Ct,Rt,Gt,alphat,betat,ModelCode,TransferCode,n,p,Ypad,ctanh,alpha,L_,rho,mu0,W,NA_REAL,delta_nb,obs_type,false);
    
	if (use_smoothing) {
		backwardSmoother(ht,Ht,n,p,mt,at,Ct,Rt,Gt,W,delta);
	} else {
        ht = mt.row(0).t();
        Ht = Ct.tube(0,0);
    }

    /* ----- Hyperparameter and Initialization ----- */


    /* ----- Storage ----- */
    arma::mat ht_stored(npad,niter,arma::fill::zeros);
    arma::mat Ht_stored(npad,niter,arma::fill::zeros);
    arma::vec bw_stored(niter,arma::fill::zeros);
    arma::vec W_stored(niter,arma::fill::zeros);
    double tmp,psi_n_sq, psi_0_sq, Exx, Eyy, Exy1, Exy2;
    /* ----- Storage ----- */

    for (unsigned int i=0; i<niter; i++) {
        R_CheckUserInterrupt();

        // TODO: check this part
        if (eta_select.at(0)==1) {
            // Exx = arma::accu(Ht.tail(n) + ht.tail(n)%ht.tail(n));
            // Eyy = arma::accu(Ht.head(n) + ht.head(n)%ht.head(n));
            // Exy1 = arma::accu(Ht.head(n) + ht.tail(n)%ht.head(n));
            // Exy2 = arma::accu(Ht.tail(n) + ht.tail(n)%ht.head(n));
            // Exx = arma::accu(ht.tail(n)%ht.tail(n));
            // Eyy = arma::accu(ht.head(n)%ht.head(n));
            // Exy1 = arma::accu(ht.tail(n)%ht.head(n));
            // Exy2 = arma::accu(ht.tail(n)%ht.head(n));
            // tmp = Exx - Exy1 - Exy2 + Eyy;
            switch (eta_prior_type.at(0)) {
                case 0: // Gamma
                {
                    a3 = 0.5*arma::accu(arma::pow(arma::diff(ht),2.));
                    coef_W wcoef[1] = {{a1,a2,a3}};
                    W = optimize_postW_gamma(wcoef[0]); // optimal Wtilde=log(W)
                    W = std::exp(W);
                }
                break;
                case 1: // Half-Cauchy
                {

                }
                break;
                case 2: // Inverse-Gamma
                {
                    psi_n_sq = Ht.at(n) + ht.at(n)*ht.at(n);
                    psi_0_sq = Ht.at(0) + ht.at(0)*ht.at(0);
                    tmp = psi_n_sq - psi_0_sq;
                    bw = bw_prior;
                    if (tmp>arma::datum::eps) {
                        bw += 0.5*tmp;
                    }
                    bw_stored.at(i) = bw;
                    W = bw/(aw-1.);
                }
                break;
                default:
                {

                }
            }
            
        }


        forwardFilter(mt,at,Ct,Rt,Gt,alphat,betat,ModelCode,TransferCode,n,p,Ypad,ctanh,alpha,L_,rho,mu0,W,delta,delta_nb,obs_type,false);
        
	    if (use_smoothing) {
            backwardSmoother(ht,Ht,n,p,mt,at,Ct,Rt,Gt,W,delta);
	    } else {
            ht = mt.row(0).t();
            Ht = Ct.tube(0,0);
        }

        ht_stored.col(i) = ht;
        Ht_stored.col(i) = Ht;
        W_stored.at(i) = W;

        if (verbose) {
            Rcout << "\rProgress: " << i+1 << "/" << niter;
        }
        
    }
    if (verbose) {
        Rcout << std::endl;
    }
    

    Rcpp::List output;
    // output["aw"] = aw;
    // output["bw"] = Rcpp::wrap(bw_stored);
    output["ht"] = Rcpp::wrap(ht_stored);
    output["Ht"] = Rcpp::wrap(Ht_stored);
    output["W"] = Rcpp::wrap(W_stored);
    return output;
}