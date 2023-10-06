#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <algorithm>
#include <RcppArmadillo.h>
#include "lbe_poisson.h"
#include "model_utils.h"
#include "pg.h"

using namespace Rcpp;
// [[Rcpp::plugins(cpp17)]]
// [[Rcpp::depends(RcppArmadillo)]]




/**
* Gibbs sampler with Metropolis-Hasting steps and Polya-Gamma augmentation.
* The model is:
* @f{align*}
* y_{s,t}       &\sim\  \text{NB}(n_s,p_{s,t}) \\
* p_{s,t}       &=      \frac{\exp(\lambda_{s,t})}{1 + \exp(\lambda_{s,t})} \\
* \lambda_{s,t} &=      a_s + \sum_{l=1}^L R_{s,t-l} g_{l} \frac{y_{s,t-l}}{n_s} \\
* R_{s,t}       &=      b_s + R_t \\
* R_{t}         &=      R_{t-1} + w_{t},\ w_{t} \sim\mathcal{N}(0,W_{t})
* @f}
*
*
*
* Random variables: 
* (1) @f$\{\bm{\theta}\}_{0}^{T}@f$: latent state.
* (2) @f$\{z_{s,t}\}@f$ and @f$\{\omega_{s,t}\}@f$: auxiliary variable for Polya-Gamma.
* (3) @f$\{W_t\}@f$: evolution variance(s).
*     - 0: known.
*     - 1: static evolution variance with IG(a_w,b_w) prior.
*     - 2: time-varying evolution variance using discount factor.
* (4) @f$\{a_s\}_{s=1}^{S}@f$: baseline with CAR prior.
*     - 0: known.
*     - 1: CAR prior.
* (5) @f$\{b_s\}_{s=1}^{S}@f$: spatial effect with CAR prior.
*     - 0: known.
*     - 1: CAR prior.
* (6) @f$\{\sigma^{2}_{a},\delta_{a}\}@f$: CAR prior parameters for a, with IG(0,0,a)-beta(a_{a\delta}, b_{a\delta}) prior.
* (7) @f$\{\sigma^{2}_{b},\delta_{b}\}@f$: CAR prior parameters for b, with IG(0,0,a)-beta(a_{b\delta}, b_{b\delta}) prior.
*
*
*
* @param Y S x T matrix, where S is the number of locations and T is the number of temporal observations.
* @param npop S x 1 vector, number of population normalizer for each location.
* @param L integer as double value, length of temporal lags for transfer effect.
* @param nburnin
* @param nthin
* @param nsample
* @param eta_type 7 x 1 integer vector, specific which entry of eta is fixed or random with specific prior.
* @f$\eta = \{
*   W_t
*   a_s，\sigma^{2}_{a}, \delta_{a}, 
*   b_s, \sigma^{2}_{b}, \delta_{b}
* \}@f$.
* @param eta_prior_val 4 x 5 numeric matrix
*               (W)         (sig2_a)     (delta_a)     (sig2_b)      (delta_b)
* [fixed/init]  0.01          0.1           1             0.1           1
* [param - 1]   0.01           0            1             0             1
* [param - 2]   0.01           0            1             0             1
* [param - 3]   0(x)           1            0(x)          1             0(x)
* @param summarize_return
* @param verbose
* 
Unknown Parameters
	- local parameters: psi[1:n]
	- global parameters `eta`: W, mu0, rho, M,th0, psi0
*/
//' @export
// [[Rcpp::export]]
Rcpp::List mcmc_polya_gamma(
	const arma::mat& Y, // S x T, the observed response
    const arma::vec& npop, // S x 1, number of population for each location
	const unsigned int L = 12, // temporal lags for transfer effect
	const unsigned int nburnin = 0,
	const unsigned int nthin = 1,
	const unsigned int nsample = 1,
    const Rcpp::IntegerVector& eta_type = Rcpp::IntegerVector::create(1,1,1,1,1,1,1), // 7 x 1 integer vector, specific which entry of eta is fixed or random with specific prior.
    const Rcpp::Nullable<Rcpp::NumericMatrix>& eta_prior_val = R_NilValue,  // 4 x 5 numeric matrix
    const Rcpp::Nullable<Rcpp::NumericVector>& a_fixed = R_NilValue, // specify values of a if assuming known.
    const Rcpp::Nullable<Rcpp::NumericVector>& b_fixed = R_NilValue, // specify values of b if assuming known.
    const Rcpp::Nullable<Rcpp::NumericVector>& m0_prior = R_NilValue,
	const Rcpp::Nullable<Rcpp::NumericMatrix>& C0_prior = R_NilValue,
	const bool summarize_return = false,
	const bool verbose = true) { // n x 1

    /*
    -----------------
    Input Processing
    -----------------
    */
    if (eta_type[2] == 1 || eta_type[3] == 1)
    {
        if (eta_type[1] != 1) {
            ::Rf_error("a[s] must be random if sigma2_a or delta_a is random");
        }
    }
    if (eta_type[5] == 1 || eta_type[6] == 1)
    {
        if (eta_type[4] != 1) {
            ::Rf_error("b[s] must be random if sigma2_b or delta_b is random");
        }
    }

    arma::mat eta_init;
    if (eta_prior_val.isNull()) {
        eta_init.set_size(4,5);
        eta_init.zeros();

        // W
        eta_init.at(0,0) = 0.01; // Initial/fixed value for W
        eta_init.at(1,0) = 0.01; // a_w in IG(a_w,b_w)
        eta_init.at(2,0) = 0.01; // b_w in IG(a_w,b_w)

        // sig2_a
        eta_init.at(0,1) = 0.1; // Initial/fixed value for sig2_a
        eta_init.at(3,1) = 1.;

        //delta_a
        eta_init.at(0,2) = 1.; // Initial/fixed value for delta_a
        eta_init.at(1,2) = 1.; // a_adelta for Beta(a_adelta, b_adelta)
        eta_init.at(2,2) = 1.; // b_adelta for Beta(a_adelta, b_adelta)

        // sig2_b
        eta_init.at(0,3) = 0.1; // Initial/fixed value for sig2_b
        eta_init.at(3,3) = 1.;

        //delta_b
        eta_init.at(0,4) = 1.; // Initial/fixed value for delta_b
        eta_init.at(1,4) = 1.; // a_bdelta for Beta(a_bdelta, b_bdelta)
        eta_init.at(2,4) = 1.; // b_bdelta for Beta(a_bdelta, b_bdelta)

    }
    /* ----------------- */


    
    /*
    -----------------
    Variable - Counting
    -----------------
    */
    const unsigned int ns = Y.n_rows; // S, number of location
	const unsigned int nt = Y.n_cols; // T, number of temporal observations
    const double nt_ = static_cast<double>(nt);

	const unsigned int ntotal = nburnin + nthin*nsample + 1; // number of iterations for MCMC

    const arma::vec Js(ns,arma::fill::zeros);
    const arma::vec Jl(L,arma::fill::ones);
    const arma::mat Ypad = arma::join_rows(Js,Y); // ns x (nt+1)
    /* ----------------- */


    /*
    -----------------
    Variables - Polya Gamma
    -----------------
    */
    arma::mat Omega(ns,nt,arma::fill::ones);
    arma::mat Lambda(ns,nt,arma::fill::zeros);
    arma::mat Z(ns,nt,arma::fill::zeros);
    /* ----------------- */


    /*
    -----------------
    Variables - Forward Filtering Backward Sampling
    -----------------
    */
    arma::vec theta(L,arma::fill::zeros);
    arma::mat Theta(L,nt+1,arma::fill::zeros);
    arma::mat G(L,L,arma::fill::zeros);
    G.at(0,0) = 1.;
    G.diag(-1).ones();
    arma::mat Ot_inv(ns,ns,arma::fill::zeros);

    arma::vec Fphi = get_Fphi(L);
    arma::vec Fy(L,arma::fill::zeros);
    arma::vec Fts(L,arma::fill::zeros);
    arma::mat Ft(L,ns,arma::fill::zeros);

    arma::mat at(L,nt+1,arma::fill::zeros);
    arma::cube Rt(L,L,nt+1);
    arma::vec ft(ns,arma::fill::zeros);
    arma::mat Qt(ns,ns,arma::fill::eye);

    arma::mat mt(L,nt+1,arma::fill::zeros);
    arma::cube Ct(L,L,nt+1);
    if (!m0_prior.isNull()) {
		mt.col(0) = Rcpp::as<arma::vec>(m0_prior);
	}
	if (!C0_prior.isNull()) {
		Ct.slice(0) = Rcpp::as<arma::mat>(C0_prior);
	}

    arma::vec ht(L,arma::fill::zeros);
    arma::mat Ht(L,L,arma::fill::eye);
    arma::mat Rinv(L,L,arma::fill::zeros);
    /* ----------------- */




    /*
    -----------------
    Temporal parameters for MCMC
    -----------------
    */
    arma::vec atmp(ns,arma::fill::zeros);
    if (!a_fixed.isNull()) {
        atmp = arma::vec(Rcpp::as<Rcpp::NumericVector>(a_fixed).begin(),ns);
    }
    arma::vec btmp(ns,arma::fill::zeros); 
    if (!b_fixed.isNull()) {
        btmp = arma::vec(Rcpp::as<Rcpp::NumericVector>(b_fixed).begin(),ns);
    }
    arma::mat Btmp = arma::diagmat(btmp);
    arma::vec Ztmp(ns,arma::fill::zeros);

    double Wtmp = eta_init.at(0,0);
    double aw = eta_init.at(1,0) + 0.5*nt_ ;
    double bw;
    arma::mat Wmat(L,L,arma::fill::zeros);
    Wmat.at(0,0) = Wtmp;

    double sig2_a = eta_init.at(0,1);
    double delta_a = eta_init.at(0,2);
    double sig2_b = eta_init.at(0,3);
    double delta_b = eta_init.at(0,4);

    

    bool saveiter;
    /*
    -----------------
    Temporal parameters for MCMC
    -----------------
    */
    

    /**/




	for (unsigned int b=0; b<ntotal; b++) {
		R_CheckUserInterrupt();
		saveiter = b > nburnin && ((b-nburnin-1)%nthin==0);

        /*
        (1) Sample auxliary Polya-Gamma variables: Omega, Lambda, Zmat.
        */
        for (unsigned int s=0; s<ns; s++) {
            for (unsigned int t=0; t<nt; t++) {
                Omega.at(s,t) = pg::rpg_normal(npop.at(s) + Y.at(s,t),Lambda.at(s,t));
                Z.at(s,t) = 0.5 * (Y.at(s,t) - npop.at(s)) / Omega.at(s,t); 
                Lambda.at(s,t) = R::rnorm(Z.at(s,t), 1./Omega.at(s,t));
            }
        }

        /*
        (2) Sample augmented latent states Theta using forward filtering and backward sampling.
        */

        // Forward filtering
        for (unsigned int t=1; t<=nt; t++) {
            Ot_inv = arma::diagmat(1./Omega.col(t));

            if (t>1 && L>1 && t<=L) {
                // Reference Analysis when t <= L
                at.col(t) = at.col(1);
                Rt.slice(t) = Rt.slice(1);
                mt.col(t) = mt.col(1);
                Ct.slice(t) = Ct.slice(1);
            } else {
                for (unsigned int s=0; s<ns; s++) {
                    update_Ft_koyama(Fts,Fy,t,L,Ypad.row(s),Fphi,1.);
                    Ft.col(s) = Fts; // Ft: L x ns
                }

                Ztmp = Z.col(t) - atmp - Btmp * Ft.t() * Jl;

                at.col(t) = G * mt.col(t-1); // L x 1
                Rt.slice(t) = G * Ct.slice(t-1) * G.t() + Wmat; // L x L
                ft = Ft.t() * at.col(t); // ns x 1 = (ns x L) * (L x 1)

                Qt = Ft.t() * Rt.slice(t) * Ft + Ot_inv; // ns x ns
                Qt = arma::chol(arma::symmatu(Qt)); // Q = R'R, R: upper triangular part of Cholesky decomposition
                Qt = arma::inv(arma::trimatu(Qt)); // R^{-1}: inverse of upper-triangular cholesky component
                Qt = Qt * Qt.t(); // inverse Qt^{-1} = R^{-1} (R^{-1})'

                mt.col(t) = at.col(t) + Rt.slice(t) * Ft * Qt * (Ztmp - ft);
                Ct.slice(t) = Rt.slice(t) - Rt.slice(t) * Ft * Qt * Ft.t() * Rt.slice(t);
            }            
        }


        // Backward sampling
        ht = mt.col(nt);
        Ht = Ct.slice(nt);
        Ht = arma::chol(arma::symmatu(Ht));
        Theta.col(nt) = ht + Ht.t() * arma::randn(L,1);
        for (unsigned int t=(nt-1); t>0; t--) {
            Rinv = arma::chol(arma::symmatu(Rt.slice(t+1)));
            Rinv = arma::inv(arma::trimatu(Rinv));
            Rinv = Rinv * Rinv.t();
            ht = mt.col(t) + Ct.slice(t) * G.t() * Rinv * (Theta.col(t+1) - at.col(t+1));
            Ht = Ct.slice(t) - Ct.slice(t) * G.t() * Rinv * G * Ct.slice(t);
            Ht = arma::chol(arma::symmatu(Ht));
            Theta.col(t) = ht + Ht.t() * arma::randn(L,1);
        }
        Rinv = arma::chol(arma::symmatu(Rt.slice(1)));
        Rinv = arma::inv(arma::trimatu(Rinv));
        Rinv = Rinv * Rinv.t();
        ht = mt.col(0) + Ct.slice(0) * G.t() * Rinv * (Theta.col(1) - at.col(1));
        Ht = Ct.slice(0) - Ct.slice(0) * G.t() * Rinv * G * Ct.slice(0);
        Ht = arma::chol(arma::symmatu(Ht));
        Theta.col(0) = ht + Ht.t() * arma::randn(L,1);


        // (3) Sample evolution variance W
        if (eta_type.at(0) == 1) {
            bw = eta_init.at(2,0); // rate
            for (unsigned int ti=1; ti<=nt; ti++) {
                bw += 0.5 * (Theta.at(0,ti)-Theta.at(0,ti-1)) * (Theta.at(0,ti)-Theta.at(0,ti-1));
            }

            Wtmp = 1./R::rgamma(aw,1./bw);
            Wmat.at(0,0) = Wtmp;
        }


        // (4) Sample spatial baseline a=(a[1],...,a[ns])
        if (eta_type.at(1) == 1) {

        }


        // (5) Sample sig2_a
        if (eta_type.at(2) == 1) {

        }


        // (6) Sample delta_a
        if (eta_type.at(3) == 1) {

        }


        // (7) Sample spatial covariates b=(b[1],...,b[ns])
        if (eta_type.at(4) == 1) {

        }

        // (8) Sample sig2_b
        if (eta_type.at(5) == 1) {

        }


        // (9) Sample delta_b
        if (eta_type.at(6) == 1) {

        }

		
		
		// store samples after burnin and thinning
		if (saveiter || b==(ntotal-1)) {
			unsigned int idx_run;
			if (saveiter) {
				idx_run = (b-nburnin-1)/nthin;
			} else {
				idx_run = nsample - 1;
			}

		}

		if (verbose) {
			Rcout << "\rProgress: " << b << "/" << ntotal-1;
		}
		
	}

	if (verbose) {
		Rcout << std::endl;
	}

	Rcpp::List output;
	
	return output;
}


