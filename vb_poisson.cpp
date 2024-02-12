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
    const Rcpp::IntegerVector &eta_select = Rcpp::IntegerVector::create(1, 0, 0),           // W, mu0, rho
    const Rcpp::NumericVector &W_par_in = Rcpp::NumericVector::create(0.01, 2, 0.01, 0.01), // (Winit/Wtrue,WpriorType,par1,par2)
    const Rcpp::NumericVector &lag_par_in = Rcpp::NumericVector::create(0.5, 6),
    const Rcpp::NumericVector &obs_par_in = Rcpp::NumericVector::create(0., 30.),
    const unsigned int &nlag_in = 20,
    const double &delta = NA_REAL,
    const unsigned int &nburnin = 1000,
    const unsigned int &nthin = 2,
    const unsigned int &nsample = 5000, // number of iterations for variational inference
    const double &theta0_upbnd = 2.,
    const double &delta_discount = 0.95,
    const bool &truncated = true,
    const bool &use_smoothing = true,
    const bool &use_discount = false,
    const bool &summarize_return = true)
{


    const unsigned int n = Y.n_elem;
    const double n_ = static_cast<double>(n);
	const unsigned int npad = n+1;
    arma::vec Ypad(npad,arma::fill::zeros); // (n+1) x 1
	Ypad.tail(n) = Y;

    const unsigned int ntotal = nburnin + nthin * nsample + 1;
    const unsigned int N = 100;                                        // number of particles for SMC

    unsigned int obs_code, link_code, trans_code, gain_code, err_code;
    get_model_code(obs_code, link_code, trans_code, gain_code, err_code, model_code);

    arma::vec W_par(W_par_in.begin(), W_par_in.length());
    arma::vec lag_par(lag_par_in.begin(), lag_par_in.length());
    arma::vec obs_par(obs_par_in.begin(), obs_par_in.length());

    double W = W_par.at(0);
    double mu0 = obs_par.at(0);
    double delta_nb = obs_par.at(1);
    double rho = lag_par.at(0);

    const unsigned int Blag = nlag_in;
    unsigned int nlag = nlag_in;
    unsigned int p = nlag;
    if (!truncated)
    {
        nlag = n;
        p = (unsigned int)lag_par.at(1) + 1;
    }

    arma::vec Fphi = get_Fphi(nlag, lag_par, trans_code);
    arma::vec Ft = init_Ft(p, trans_code);


    /* ----- Hyperparameter and Initialization ----- */
    unsigned int W_prior_type = W_par.at(1);
    double aw, bw;
    double a1, a2, a3;
    switch (W_prior_type)
    {
    case 0: // Gamma
    {
        a1 = W_par.at(2) - 0.5 * n_;
        a2 = W_par.at(3);
        a3 = 0.;
    }
    break;
    case 1: // Half-Cauchy
    {
    }
    break;
    case 2: // Inverse-Gamma
    {
        aw = W_par.at(2);
        bw = W_par.at(3);
    }
    break;
    default:
    {
    }
    }

    arma::mat at(p,npad,arma::fill::zeros);
	const arma::mat Ip(p,p,arma::fill::eye);
	
	arma::cube Rt(p,p,npad);
	arma::vec alphat(npad,arma::fill::zeros);
	arma::vec betat(npad,arma::fill::zeros);
	arma::vec ht(npad,arma::fill::zeros);
	arma::vec Ht(npad,arma::fill::zeros);

    arma::cube Gt(p,p,npad);
	arma::mat Gt0(p,p,arma::fill::zeros);
    Gt0.at(0, 0) = 1.;
    if (trans_code == 0)
    { // Koyck
        Gt0.at(1, 1) = rho;
    }
    else if (trans_code == 1)
    { // Koyama
        Gt0.diag(-1).ones();
    }
    else if (trans_code == 2)
    { // Solow
        double coef2 = -rho;
        Gt0.at(1, 1) = -binom(lag_par.at(1), 1) * coef2;
        for (unsigned int k = 2; k < p; k++)
        {
            coef2 *= -rho;
            Gt0.at(1, k) = -binom(lag_par.at(1), k) * coef2;
            Gt0.at(k, k - 1) = 1.;
        }
    }
    else if (trans_code == 3)
    { // Vanilla
        Gt0.at(0, 0) = rho;
    }
    for (unsigned int t = 0; t < npad; t++)
    {
        Gt.slice(t) = Gt0;
    }

    arma::mat mt(p, n + 1, arma::fill::zeros);
    arma::cube Ct(p, p, n + 1, arma::fill::zeros);
    Ct.each_slice() = Ip;
    arma::vec m0(p, arma::fill::zeros);
    arma::mat C0(p, p, arma::fill::eye);
    C0 *= std::pow(theta0_upbnd * 0.5, 2.);


    // forwardFilter(mt,at,Ct,Rt,Gt,alphat,betat,obs_code,link_code,trans_code,gain_code,n,p,Ypad,L,nlag,rho,mu0,W,NA_REAL,delta_nb);
    forwardFilter(
        mt, at, Ct, Rt, Gt, alphat, betat,
        Ypad, n, p, obs_par, lag_par, W,
        obs_code, link_code, trans_code, gain_code,
        nlag, delta_discount, truncated, use_discount);

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
            switch (W_prior_type) {
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

        forwardFilter(
            mt, at, Ct, Rt, Gt, alphat, betat,
            Ypad, n, p, obs_par, lag_par, W, 
            obs_code, link_code, trans_code, gain_code,
            nlag, delta_discount, truncated, use_discount);
        
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

            switch (W_prior_type) {
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


        

        Rcout << "\rProgress: " << i << "/" << ntotal-1;
        
    }

    Rcout << std::endl;
    

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