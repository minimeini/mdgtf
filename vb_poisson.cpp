#include "lbe_poisson.h"
#include "pl_poisson.h"
using namespace Rcpp;
// [[Rcpp::plugins(cpp17)]]
// [[Rcpp::depends(RcppArmadillo,nloptr)]]

/**
 * Mean-field variation: W is working; the latent state part is problem matic
*/
//' @export
// [[Rcpp::export]]
Rcpp::List vb_poisson(
    const arma::vec &Y, // n x 1, the observed response
    const arma::uvec &model_code,
    const arma::uvec &eta_select,     // 4 x 1, indicator for unknown (=1) or known (=0), global parameters: W, mu0, rho, M
    const arma::vec &eta_init,        // 4 x 1, if true/initial values should be provided here
    const arma::uvec &eta_prior_type, // 4 x 1
    const arma::mat &eta_prior_val,   // 2 x 4, priors for each element of eta
    const unsigned int L = 0,
    const double delta = NA_REAL,
    const double alpha = 1.,
    const unsigned int nburnin = 1000,
    const unsigned int nthin = 2,
    const unsigned int nsample = 5000, // number of iterations for variational inference
    const Rcpp::Nullable<Rcpp::NumericVector> &m0_prior = R_NilValue,
    const Rcpp::Nullable<Rcpp::NumericMatrix> &C0_prior = R_NilValue,
    const Rcpp::NumericVector &ctanh = Rcpp::NumericVector::create(0.2, 0, 5.),
    const double aw_prior = 0.01, // aw: shape of inverse gamma
    const double bw_prior = 0.01, // bw: rate of inverse gamma
    const double delta_nb = 1.,
    const double theta0_upbnd = 2.,
    const double Blag_pct = 0.15,
    const double delta_discount = 0.95,
    const bool use_smoothing = true,
    const bool summarize_return = true,
    const bool verbose = true)
{


    const unsigned int n = Y.n_elem;
    const double n_ = static_cast<double>(n);
	const unsigned int npad = n+1;
    arma::vec Ypad(npad,arma::fill::zeros); // (n+1) x 1
	Ypad.tail(n) = Y;

    const unsigned int ntotal = nburnin + nthin * nsample + 1;
    const unsigned int Blag = static_cast<unsigned int>(Blag_pct * n); // B-fixed-lags Monte Carlo smoother
    const unsigned int N = 100;                                        // number of particles for SMC

    const unsigned int obs_code = model_code.at(0);
    const unsigned int link_code = model_code.at(1);
    const unsigned int trans_code = model_code.at(2);
    const unsigned int gain_code = model_code.at(3);
    const unsigned int err_code = model_code.at(4);
	unsigned int p, L_;
    init_by_trans(p,L_,trans_code,L);



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
    

	arma::mat at(p,npad,arma::fill::zeros);
	const arma::mat Ip(p,p,arma::fill::eye);
	
	arma::cube Rt(p,p,npad);
	arma::vec alphat(npad,arma::fill::zeros);
	arma::vec betat(npad,arma::fill::zeros);
	arma::vec ht(npad,arma::fill::zeros);
	arma::vec Ht(npad,arma::fill::zeros);

    arma::cube Gt(p,p,npad);
	arma::mat Gt0(p,p,arma::fill::zeros);
	Gt0.at(0,0) = 1.;
	if (trans_code == 0) { // Koyck
		Gt0.at(1,1) = rho;
	} else if (trans_code == 1) { // Koyama
		Gt0.diag(-1).ones();
	} else if (trans_code == 2) { // Solow
        double coef2 = -rho;
		Gt0.at(1,1) = -binom(L,1)*coef2;
		for (unsigned int k=2; k<p; k++) {
			coef2 *= -rho;
			Gt0.at(1,k) = -binom(L,k)*coef2;
			Gt0.at(k,k-1) = 1.;
		}
	} else if (trans_code == 3) { // Vanilla
        Gt0.at(0,0) = rho;
    }
	for (unsigned int t=0; t<npad; t++) {
		Gt.slice(t) = Gt0;
	}



    arma::mat mt(p, n + 1, arma::fill::zeros);
    arma::cube Ct(p, p, n + 1, arma::fill::zeros);
    Ct.each_slice() = Ip;
    arma::vec m0(p, arma::fill::zeros);
    arma::mat C0(p, p, arma::fill::eye);

    if (!m0_prior.isNull())
    {
        m0 = Rcpp::as<arma::vec>(m0_prior);
        mt.col(0) = m0;
    }
    if (!C0_prior.isNull())
    {
        C0 = Rcpp::as<arma::mat>(C0_prior);
        Ct.slice(0) = C0;
    }
    else
    {
        C0 *= std::pow(theta0_upbnd * 0.5, 2.);
    }

    forwardFilter(mt,at,Ct,Rt,Gt,alphat,betat,obs_code,link_code,trans_code,gain_code,n,p,Ypad,ctanh,alpha,L_,rho,mu0,W,NA_REAL,delta_nb,false);
    
	if (use_smoothing) {
		backwardSmoother(ht,Ht,n,p,mt,at,Ct,Rt,Gt,W,delta);
	} else {
        ht = mt.row(0).t();
        Ht = Ct.tube(0,0);
    }

    /* ----- Hyperparameter and Initialization ----- */


    /* ----- Storage ----- */
    arma::mat ht_stored(npad,nsample,arma::fill::zeros);
    arma::mat Ht_stored(npad,nsample,arma::fill::ones);
    arma::mat psi_stored(npad,nsample,arma::fill::zeros);
    arma::vec bw_stored(nsample,arma::fill::ones);
    arma::vec aw_stored(nsample,arma::fill::ones);
    arma::vec W_stored(nsample,arma::fill::zeros);
    double tmp,psi_n_sq, psi_0_sq, Exx, Eyy, Exy1, Exy2;
    /* ----- Storage ----- */

    bool saveiter = false;
    arma::mat R(n + 1, 2, arma::fill::zeros); // (psi,theta)
    arma::vec pmarg_y(n, arma::fill::zeros);

    for (unsigned int i=0; i<ntotal; i++) {
        R_CheckUserInterrupt();
        saveiter = i > nburnin && ((i-nburnin-1)%nthin==0);

        // TODO: check this part
        if (eta_select.at(0)==1) {
            switch (eta_prior_type.at(0)) {
                case 0: // Gamma
                {
                    a3 = 0.5*arma::accu(arma::pow(arma::diff(ht),2.));
                    if (!std::isfinite(a3)) {
                        Rcout << "mt=" << mt.row(i) << std::endl;
                        Rcout << "ht=" << ht.t() <<  std::endl;
                        ::Rf_error("Infinite a3.");
                    }
                    coef_W wcoef[1] = {{a1,a2,a3}};

                    // aw, bw are sufficient statistics
                    aw = optimize_postW_gamma(wcoef[0]); // optimal Wtilde=log(W)
                    bw = postW_deriv2(aw,a2,a3);
                    W = std::exp(std::min(aw,UPBND));
                }
                break;
                case 1: // Half-Cauchy
                {
                    ::Rf_error("Half-Cauchy prior for W not implemented yet.");
                }
                break;
                case 2: // Inverse-Gamma
                {
                    psi_n_sq = Ht.at(n) + ht.at(n)*ht.at(n);
                    psi_0_sq = Ht.at(0) + ht.at(0)*ht.at(0);
                    tmp = psi_n_sq - psi_0_sq;
                    // tmp = 0.5*arma::accu(arma::pow(arma::diff(ht),2.));
                    // bw = bw_prior + tmp;
                    
                    if (tmp>EPS) {
                        bw += 0.5*tmp;
                    }
                    bw_stored.at(i) = bw;
                    W = bw/(aw-1.);
                }
                break;
                default:
                {
                    ::Rf_error("This prior for W is not supported yet.");
                }
            }
            
        }

        // if (eta_select.at(0) == 0 || i == 0)
        // {
        //     mcs_poisson(
        //         R, pmarg_y, Ypad, model_code,
        //         NA_REAL, rho, L, mu0,
        //         Blag, 5000,
        //         m0, C0,
        //         theta0_upbnd,
        //         delta_nb,
        //         delta_discount);
        // }
        // else
        // {
        //     mcs_poisson(
        //         R, pmarg_y, Ypad, model_code,
        //         W, rho, L, mu0,
        //         Blag, N,
        //         m0, C0,
        //         theta0_upbnd,
        //         delta_nb,
        //         delta_discount);
        // }

        // ht = R.col(0);

        forwardFilter(mt,at,Ct,Rt,Gt,alphat,betat,obs_code,link_code,trans_code,gain_code,n,p,Ypad,ctanh,alpha,L_,rho,mu0,W,delta,delta_nb,false);
        
	    if (use_smoothing) {
            backwardSmoother(ht,Ht,n,p,mt,at,Ct,Rt,Gt,W,delta);
	    } else {
            ht = mt.row(0).t();
            Ht = Ct.tube(0,0);
        }

        // store samples after burnin and thinning
		if (saveiter || i==(ntotal-1)) {
			unsigned int idx_run;
			if (saveiter) {
				idx_run = (i-nburnin-1)/nthin;
			} else {
				idx_run = nsample - 1;
			}

			ht_stored.col(idx_run) = ht;
            // psi_stored.col(idx_run) = ht;
            Ht_stored.col(idx_run) = Ht;
            for (unsigned int j=0; j<npad; j++) {
                psi_stored.at(j,idx_run) = R::rnorm(ht.at(j),std::sqrt(std::abs(Ht.at(j))));
            }

            aw_stored.at(idx_run) = aw;
            bw_stored.at(idx_run) = bw;

            switch (eta_prior_type.at(0)) {
                case 0: // Gamma
                {
                    W_stored.at(idx_run) = std::exp(std::min(R::rnorm(aw,std::sqrt(-1./bw)),UPBND));
                }
                break;
                case 1: // Half-Cauchy
                {
                    ::Rf_error("Half-Cauchy prior for W not implemented yet.");
                }
                break;
                case 2: // Inverse-Gamma
                {
                    W_stored.at(idx_run) = 1./R::rgamma(aw,1./bw);
                }
                break;
                default:
                {
                    ::Rf_error("This prior for W is not supported yet.");
                }
            }
            
			// rho_stored.at(idx_run) = rho;
			// E0_stored.at(idx_run) = E0;
		}


        

        if (verbose) {
            Rcout << "\rProgress: " << i << "/" << ntotal-1;
        }
        
    }
    if (verbose) {
        Rcout << std::endl;
    }
    

    Rcpp::List output;
    // output["aw"] = aw;
    // output["bw"] = Rcpp::wrap(bw_stored);
    if (summarize_return) {
        arma::vec qProb = {0.025,0.5,0.975};
		output["psi"] = Rcpp::wrap(arma::quantile(psi_stored,qProb,1)); // (n+1) x 3
    } else {
        output["psi"] = Rcpp::wrap(psi_stored);
    }
    // output["ht"] = Rcpp::wrap(ht_stored);
    // output["Ht"] = Rcpp::wrap(Ht_stored);
    output["W"] = Rcpp::wrap(W_stored);
    return output;
}