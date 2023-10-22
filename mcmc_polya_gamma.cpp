#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <algorithm>
#include <RcppArmadillo.h>
#include <pg.h>
#include "lbe_poisson.h"
#include "model_utils.h"


using namespace Rcpp;
// [[Rcpp::plugins(cpp17)]]
// [[Rcpp::depends(RcppArmadillo,pg)]]

/*
Debugging notebook - project1/multivariate_dlm.qmd
*/


//' @export
// [[Rcpp::export]]
arma::mat get_neighbor_mat(
    const unsigned int ns,
    const double prob){

    const unsigned int ntri = ns*(ns-1)/2;
    arma::vec uptri(ns*(ns-1)/2,arma::fill::zeros);
    for (unsigned int i=0; i<ntri; i++) {
        uptri.at(i) = R::rbinom(1,prob);
    }

    arma::mat V(ns,ns,arma::fill::zeros);

    arma::uvec up_idx = arma::trimatu_ind(arma::size(V),1);
    V.elem(up_idx) = uptri;
    arma::mat Vsym = arma::symmatu(V);

    return Vsym;
}


/**
 * Calculate the logarithmic integrated likelihood of <delta>, conditional on the spatial effect.
 * 
 * @param delta double, the delta we are calculating the likelihood for.
 * @param z ns x nt matrix of spatial effects.
 * @param V ns x ns unnormalize neighboring structure matrix, with 0 and 1.
 * @param Tv ns x ns diagonal matrix. Entries are inverse of row sums of V.
 * @param mu ns x 1 vector, mean of the spatial effect.
 * @param Is ns x ns identity matrix.
 * @param ns double, number of locations.
 * @param a_delta (default = 1) a of the beta prior for delta.
 * @param b_delta (default = 1) b of the beta prior for delta.
 * @param a_sig2 (default = 1) a of the uninformative prior for sigma2.
*/
double delta_integrated_loglik(
    double delta,
    const arma::mat z, // ns x nt, spatial effects.
    const arma::mat V, // ns x ns, neighboring structure matrix
    const arma::mat Tvi, // ns x ns diagonal, entries are row sums of V
    const arma::mat Is, // ns x ns identity matrix
    const double ns, 
    const unsigned int nt = 1,
    const double a_delta = 1., // a of the beta prior for delta
    const double b_delta = 1., // b of the beta prior for delta
    const double a_sig2 = 1.,
    const double b_sig2 = 1.){ // a of the uninformative prior for sigma2

    arma::mat Q = Is - delta*Tvi*V;
    double b_sig2_tilde = 0;
    for (unsigned int i=0; i<nt; i++) {
        b_sig2_tilde += arma::as_scalar(z.col(i).t() * Q * z.col(i));
    }
    b_sig2_tilde *= 0.5;
    b_sig2_tilde += b_sig2;
    double a_sig2_tilde = a_sig2 + 0.5*static_cast<double>(nt)*ns;

    double loglik = 0.5*static_cast<double>(nt)*std::log(arma::det(Q));
    loglik -= a_sig2_tilde*std::log(b_sig2_tilde);
    loglik += (a_delta-1.)*std::log(delta) + (b_delta-1.)*std::log(1.-delta);
    return loglik;
}



/**
 * Calculate the logarithmic integrated likelihood of <logit of delta>, conditional on the spatial effect.
 * 
 * @param delta double, the delta before logit transform.
 * @param z ns x nt matrix of spatial effects.
 * @param V ns x ns unnormalize neighboring structure matrix, with 0 and 1.
 * @param Tvi ns x ns diagonal matrix. Entries are inverse of row sums of V.
 * @param mu ns x 1 vector, mean of the spatial effect.
 * @param Is ns x ns identity matrix.
 * @param ns double, number of locations.
 * @param a_delta (default = 1) a of the beta prior for delta.
 * @param b_delta (default = 1) b of the beta prior for delta.
 * @param a_sig2 (default = 1) a of the uninformative prior for sigma2.
*/
//' @export
// [[Rcpp::export]]
arma::vec logit_delta_integrated_loglik(
    const double delta, // delta before logit transform.
    const arma::mat z, // ns x nt, spatial effects.
    const arma::mat V, // ns x ns, neighboring structure matrix
    const arma::mat Tvi, // ns x ns diagonal, entries are row sums of V
    const arma::mat Is, // ns x ns identity matrix
    const double ns, 
    const unsigned int nt = 1,
    const double a_delta = 1., // a of the beta prior for delta
    const double b_delta = 1., // b of the beta prior for delta
    const double a_sig2 = 0.01,
    const double b_sig2 = 0.01){ // a of the uninformative prior for sigma2

    arma::mat Tv = Tvi.i();
    arma::mat Q = Tv * (Is - delta*Tvi*V);
    double b_sig2_tilde = 0;
    for (unsigned int i=0; i<nt; i++) {
        b_sig2_tilde += arma::as_scalar(z.col(i).t() * Q * z.col(i));
    }
    b_sig2_tilde *= 0.5;
    b_sig2_tilde += b_sig2;
    double a_sig2_tilde = a_sig2 + 0.5*static_cast<double>(nt*ns);

    double loglik = 0.5*static_cast<double>(nt)*std::log(arma::det(Q));
    loglik -= a_sig2_tilde*std::log(b_sig2_tilde);
    loglik += a_delta*std::log(delta) + b_delta*std::log(1.-delta);

    arma::vec output(3);
    output.at(0) = loglik;
    output.at(1) = a_sig2_tilde;
    output.at(2) = b_sig2_tilde;

    return output;
}

// double logprob_joint_c(
//     const double sigma2,
//     const double delta,
//     const arma::vec& b, // ns x 1
//     // const arma::mat& Omega, //  ns x ns
//     const arma::vec& kvec, // ns x 1
//     const arma::mat& Sigma2_inv,
//     // const arma::mat& V, // ns x ns
//     // const arma::mat& Tv, // ns x ns
//     // const arma::mat& Tvi, // ns x ns
//     const double a_delta = 1.,
//     const double b_delta = 1.,
//     const double a_sig2 = 0.01,
//     const double b_sig2 = 0.01)


// //' @export
// // [[Rcpp::export]]
// double logprob_joint_c(
//     const double sigma2,
//     const double delta,
//     const arma::vec& b, // ns x 1
//     const arma::vec& kvec, // ns x 1
//     const arma::mat& Sigma_inv,
//     const double a_delta = 1.,
//     const double b_delta = 1.,
//     const double a_sig2 = 0.01,
//     const double b_sig2 = 0.01){

//     arma::mat Sigma = Sigma_inv.i();
//     arma::vec mu = Sigma * kvec;

//     double sse = arma::as_scalar((b.t()-mu.t()) * Sigma_inv * (b-mu));

//     // Sigma_inv.diag() += 1.e-5;
//     double ldet = arma::log_det_sympd(arma::symmatu(Sigma_inv));

//     double loglik = 0.5*ldet - 0.5*sse - a_sig2*std::log(sigma2) - b_sig2/sigma2;
//     loglik += a_delta*std::log(delta) + b_delta*std::log(1.-delta);
//     return loglik;
// }



//' @export
// [[Rcpp::export]]
arma::vec update_car_para(
    const arma::vec& car_old, // (delta_old,sigma2_old,logp_old,mh_accept)
    const arma::vec& b, // ns x 1
    const arma::mat& Omega, // ns x ns
    const arma::vec& kvec, // ns x 1
    const arma::mat& V, // ns x ns
    const arma::mat& Tv, // ns x ns
    const arma::mat& Tvi, // ns x ns 
    const Rcpp::NumericVector mh_sd = Rcpp::NumericVector::create(0.2,0.2), // (mh_sd_delta,mh_sd_sigma2)
    const double a_delta = 1.,
    const double b_delta = 1.,
    const double a_sig2 = 0.01,
    const double b_sig2 = 0.01){
    
    const unsigned int ns = b.n_elem;
    const arma::mat Is(ns,ns,arma::fill::eye);
    
    double delta_old = car_old.at(0);
    // double logp_old = car_old.at(2);

    double logit_delta_old = std::log(delta_old) - std::log(1.-delta_old);
    double logit_delta_new = R::rnorm(logit_delta_old,mh_sd[0]);
    double delta_new = std::exp(logit_delta_new) / (1. + std::exp(logit_delta_new));
    
    arma::vec car_new = car_old;
    car_new.at(3) = 0.; // not accept (yet)

    // try{
        // arma::mat Sigma_inv = arma::symmatu(Tv * (Is - delta_new*Tvi*V) / sigma2_new + Omega);
        // double tmp = arma::log_det_sympd(arma::symmatu(Sigma_inv));

        arma::vec output0 = logit_delta_integrated_loglik(
            delta_old,b,V,Tvi,Is,ns,1,
            a_delta,b_delta,a_sig2,b_sig2);
        
        double logp_old = output0.at(0);

        arma::vec output = logit_delta_integrated_loglik(
            delta_new,b,V,Tvi,Is,ns,1,
            a_delta,b_delta,a_sig2,b_sig2);
        
        double logp_new = output.at(0);
        
        // double logratio = std::min(0.,logp_new-logp_old);
        // if (std::log(R::runif(0.,1.)) < logratio) {
        double acc_ratio = std::log(R::runif(0.,1.));
        // Rcout << "\n" << exp(logp_new - logp_old) << " " << exp(acc_ratio) << "\n";
        if (acc_ratio < (logp_new-logp_old)) {
            car_new.at(0) = delta_new;
            double tmp = 1./R::rgamma(output.at(1),1./output.at(2));
            // Rcout << tmp << "\n";
            car_new.at(1) = tmp; 
            car_new.at(2) = logp_new;
            car_new.at(3) = 1.; // accept
        }
        // else{
        //     Rcout << "\n\n" << exp(logp_new - logp_old) << " " << acc_ratio << "\n\n";
        // }

    // } catch(...) {
    //     car_new.at(3) = 0.;
    // }

    return car_new;
}



/*
The augmented model: temporal + spatio-temporal effects
*/
//' @export
// [[Rcpp::export]]
Rcpp::List mcmc_polya_gamma0(
	const arma::mat& Y, // S x T, the observed response
    const arma::vec& npop, // S x 1, number of population for each location
    const Rcpp::Nullable<Rcpp::NumericVector>& V_neighbor = R_NilValue, // S x S, neighboring structure matrix with 0 or 1 entries
	const unsigned int L = 12, // temporal lags for transfer effect
	const unsigned int nburnin = 0,
	const unsigned int nthin = 1,
	const unsigned int nsample = 1,
    const Rcpp::IntegerVector& eta_type = Rcpp::IntegerVector::create(0,0,0,0), // 4 x 1 integer vector, specific which entry of eta is fixed or random with specific prior.
    const Rcpp::Nullable<Rcpp::NumericMatrix>& eta_prior_val = R_NilValue,  // 3 x 3 numeric matrix
    const Rcpp::Nullable<Rcpp::NumericVector>& mu_init = R_NilValue, // specify values of mu if assuming known.
    const Rcpp::Nullable<Rcpp::NumericVector>& m0_prior = R_NilValue,
	const Rcpp::Nullable<Rcpp::NumericMatrix>& C0_prior = R_NilValue,
    const Rcpp::Nullable<Rcpp::NumericVector>& Fphi_g = R_NilValue,
    const double sd_mh = 0.1,
    const double discount_factor = 0.9,
    const double npop_scale = 1.,
	const bool summarize_return = false,
	const bool verbose = true) { // n x 1

    /*
    -----------------
    Input Processing
    -----------------
    */
    arma::mat eta_init(3,3,arma::fill::zeros);
    if (eta_prior_val.isNull()) {
        // W
        eta_init.at(0,0) = 0.01; // Initial/fixed value for W
        eta_init.at(1,0) = 0.01; // a_w in IG(a_w,b_w)
        eta_init.at(2,0) = 0.01; // b_w in IG(a_w,b_w)

        // sig2_b
        eta_init.at(0,1) = 0.1; // Initial/fixed value for sig2_b
        eta_init.at(1,1) = 0.01; // a[sigma]
        eta_init.at(2,1) = 0.01; // b[sigma]

        //delta_b
        eta_init.at(0,2) = 1.; // Initial/fixed value for b
        eta_init.at(1,2) = 1.; // a[delta] for Beta(a[delta], b[delta])
        eta_init.at(2,2) = 1.; // b[delta] for Beta(a[delta], b[delta])

    } else {
        eta_init = Rcpp::as<arma::mat>(eta_prior_val);
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
    const unsigned int np_ast = L;

	const unsigned int ntotal = nburnin + nthin*nsample + 1; // number of iterations for MCMC

    const arma::vec Js(ns,arma::fill::zeros);
    const arma::vec Jl(L,arma::fill::ones);
    const arma::vec Jt(nt+1,arma::fill::ones);
    const arma::mat Is(ns,ns,arma::fill::eye);

    arma::mat Zpad(ns,L+nt,arma::fill::zeros); // ns x (nt+L)
    arma::mat Ypad(ns,L+nt,arma::fill::zeros);
    Zpad.cols(L,L+nt-1) = Y;
    Ypad.cols(L,L+nt-1) = Y; 

    arma::mat V(ns,ns,arma::fill::eye);
    if (!V_neighbor.isNull()) {
        V = Rcpp::as<arma::mat>(V_neighbor);
    }
    const arma::vec vrow = arma::sum(V,1);
    const arma::mat Tv = arma::diagmat(vrow);
    const arma::mat Tvi = arma::diagmat(1./vrow);
    /* ----------------- */


    /*
    -----------------
    Variables - Forward Filtering Backward Sampling
    np_ast = L * (ns+1)
    -----------------
    */
    arma::vec theta_tilde(np_ast,arma::fill::zeros);
    arma::mat Theta_tilde(np_ast,L+nt,arma::fill::zeros);
    arma::mat G_tilde(L,L,arma::fill::zeros);
    G_tilde.at(0,0) = 1.;
    G_tilde.diag(-1).ones();
    


    arma::mat Ot_inv(ns,ns,arma::fill::zeros);

    arma::mat at(np_ast,nt+L,arma::fill::zeros);
    arma::cube Rt(np_ast,np_ast,nt+L);
    arma::vec ft(ns,arma::fill::zeros);
    arma::mat Qt(ns,ns,arma::fill::eye);

    arma::mat At(np_ast,ns,arma::fill::zeros);
    arma::mat Bt(np_ast,np_ast,arma::fill::zeros);

    arma::mat mt(np_ast,L+nt,arma::fill::zeros);
    arma::cube Ct(np_ast,np_ast,L+nt);
    if (!m0_prior.isNull()) {
		mt.col(L-1) = Rcpp::as<arma::vec>(m0_prior);
	}

	if (!C0_prior.isNull()) {
		Ct.slice(L-1) = Rcpp::as<arma::mat>(C0_prior);
	} else {
        Ct.slice(L-1) = 10. * arma::mat(Ct.n_rows,Ct.n_cols,arma::fill::eye);
    }

    arma::vec ht(np_ast,arma::fill::zeros);
    arma::mat Ht(np_ast,np_ast,arma::fill::eye);


    double coef1 = 1. - discount_factor;
    double coef2 = discount_factor * discount_factor;
    /* ----------------- */



    /*
    -----------------
    Temporal parameters for MCMC
    -----------------
    */
    arma::vec mu(ns,arma::fill::zeros); 
    if (!mu_init.isNull()) {
        mu = arma::vec(Rcpp::as<Rcpp::NumericVector>(mu_init).begin(),ns);
    }
    arma::vec mu_b(ns,arma::fill::zeros);
    arma::mat Sig_b(ns,ns,arma::fill::eye);

    

    double W = eta_init.at(0,0);
    double aw = eta_init.at(1,0) + 0.5*nt_;
    double bw = eta_init.at(2,0);
    arma::mat W_tilde(L,L,arma::fill::zeros);
    W_tilde.at(0,0) = W;


    double sigma2 = eta_init.at(0,1);
    // double a_sig2 = eta_init.at(1,1);
    // double b_sig2 = eta_init.at(2,1);
    // double a_sig2_tilde = a_sig2 + 0.5*static_cast<double>(nt*ns);
    // double b_sig2_tilde;


    double delta = eta_init.at(0,2);
    // double a_delta = eta_init.at(1,2);
    // double b_delta = eta_init.at(2,2);
    // double logit_delta_new, delta_new, logp_new;
    // double logit_delta= std::log(delta) - std::log(1.-delta);
    // double logp = logit_delta_integrated_loglik(delta,Theta_tilde.submat(L,L,L+ns-1,L+nt-1),V,Tvi,Is,ns,nt,a_delta,b_delta,a_sig2,b_sig2);
    // double logratio;
    // arma::vec mh_accept(ntotal,arma::fill::zeros);


    arma::mat Omat(ns,ns,arma::fill::eye);
    arma::vec kvec(ns,arma::fill::zeros);
    arma::vec dvec(ns,arma::fill::zeros);

    arma::cube Theta_stored(np_ast,nt+L,nsample,arma::fill::zeros);
    arma::mat mt_stored(L+nt,nsample,arma::fill::zeros);
    arma::mat Ct_stored(L+nt,nsample,arma::fill::zeros);
    arma::vec W_stored(nsample,arma::fill::zeros);
    arma::mat mu_stored(ns,nsample,arma::fill::zeros);
    arma::vec sigma2_stored(nsample,arma::fill::zeros);
    arma::vec delta_stored(nsample,arma::fill::zeros);


    
    bool saveiter;
    unsigned int ir, ic;
    
    /*
    -----------------
    Temporal parameters for MCMC
    -----------------
    */
    arma::vec Fphi = get_Fphi(L);
    if (!Fphi_g.isNull()) {
        Fphi = Rcpp::as<arma::vec>(Fphi_g);
    }
    arma::cube Ft(L,ns,L+nt);
    arma::mat X(ns,L+nt,arma::fill::zeros);
    for (unsigned int t=L; t<(L+nt); t++) {
        arma::mat Ysub = arma::reverse(Ypad.cols(t-L,t-1),1); // ns x L
        for (unsigned int s=0; s<ns; s++) {
            Ysub.row(s) /= (npop_scale*npop.at(s));
        }

        Ft.slice(t) = Ysub.t();

        for (unsigned int s=0; s<ns; s++) {
            X.at(s,t) = arma::accu(Ft.slice(t).col(s));
        }
    }
    for (unsigned int t=0; t<L; t++) {
        Ft.slice(t).zeros();
    }

    arma::mat Omega(L+nt,ns,arma::fill::zeros);
    arma::mat K(L+nt,ns,arma::fill::zeros);
    arma::mat Lambda(ns,L+nt,arma::fill::zeros); // ns x (nt+1)
    arma::mat Psi(ns,L+nt,arma::fill::zeros);
    arma::mat Z_tilde(ns,L+nt,arma::fill::zeros);
    
    arma::cube Lambda_stored(ns,L+nt,nsample);


	for (unsigned int nn=0; nn<ntotal; nn++) {
		R_CheckUserInterrupt();
		saveiter = nn > nburnin && ((nn-nburnin-1)%nthin==0);

        /*
        (1) Sample auxliary Polya-Gamma variables: Omega, Lambda, Zmat.
        */
        for (unsigned int s=0; s<ns; s++) {
            for (unsigned int t=L; t<(L+nt); t++) {
                K.at(t,s) = 0.5 * (Ypad.at(s,t) - npop.at(s));

                Lambda.at(s,t) = X.at(s,t)*mu.at(s) + Psi.at(s,t);
                Omega.at(t,s) = pg::rpg_scalar_hybrid(npop.at(s) + Ypad.at(s,t),Lambda.at(s,t));

                Z_tilde.at(s,t) = K.at(t,s)/Omega.at(t,s) - X.at(s,t)*mu.at(s);
                Psi.at(s,t) = R::rnorm(Z_tilde.at(s,t), 1./std::sqrt(Omega.at(t,s)));
            }
        }



        /*
        (2) Sample augmented latent states Theta using forward filtering and backward sampling.
        */

        // Forward filtering
        // [Checked - OK]
        for (unsigned int t=L; t<(L+nt); t++) {
            Ot_inv = arma::diagmat(1./Omega.row(t)); // ns x ns

            at.col(t) = G_tilde * mt.col(t-1); 
            // (L+ns) x 1, the last ns elements are all zeros
            Rt.slice(t) = G_tilde * Ct.slice(t-1) * G_tilde.t();

            // if (nn<nburnin) {
            //     Rt.at(0,0,t) = Rt.at(0,0,t) / discount_factor;
            // } else {
                Rt.slice(t) = Rt.slice(t) + W_tilde;
                // only the first L x L elements are nonzero.
            // }
            
                
            ft = Ft.slice(t).t() * at.col(t); // ns x 1 = (ns x L) * (L x 1)
            Qt = Ft.slice(t).t() * Rt.slice(t) * Ft.slice(t) + Ot_inv; // ns x ns
            Qt = arma::symmatu(Qt);

            arma::mat Qt_chol;
            try {
                Qt_chol = arma::chol(Qt); // Q = R'R, R: upper triangular part of Cholesky decomposition
            } catch (...) {
                arma::vec eigval;
                arma::mat eigvec;
                arma::eig_sym(eigval,eigvec,Qt);
                    
                Rcout << "(nn = " << nn << ", t = " << t;
                Rcout << ", rcond = " << arma::rcond(Qt) << ")" << std::endl;
                Rcout << "eigval = " << eigval.t() << std::endl;
            }
                
            arma::mat Qt_ichol = arma::inv(arma::trimatu(Qt_chol)); // R^{-1}: inverse of upper-triangular cholesky component
            arma::mat Qt_inv = Qt_ichol * Qt_ichol.t(); // inverse Qt^{-1} = R^{-1} (R^{-1})'

            At = Rt.slice(t) * Ft.slice(t) * Qt_inv;
            mt.col(t) = at.col(t) + At * (Z_tilde.col(t) - ft); 
            Ct.slice(t) = Rt.slice(t) - At * Ft.slice(t).t() * Rt.slice(t);
        }



        // Backward sampling
        // [Checked - OK]
        ht = mt.col(nt+L-1);
        Ht = Ct.slice(nt+L-1);
        {
            arma::mat Ht_chol;
            try {
                Ht_chol = arma::chol(arma::symmatu(Ht));
            } catch(...) {
                Rcout << "\n nn=" << nn << ", t=" << nt+L-1 << std::endl;

                arma::vec eigval;
                arma::mat eigvec;
                arma::eig_sym(eigval,eigvec,Ht);

                Rcout << "Eigenvalues of Ht(nt+L-1): " << eigval.t() << std::endl;
                ::Rf_error("Cholesky of Ht(nt+L-1) in smoothing failed");

            }

            Theta_tilde.col(nt+L-1) = ht + Ht_chol.t() * arma::randn(ht.n_elem,1);
        }
        
        // Theta: L x (nt+1)
        // Theta.at(0,nt) = R::rnorm(ht.at(nt),std::sqrt(Ht.at(nt)));
        for (unsigned int t=(L+nt-2); t>(2*L); t--) {
            arma::mat Rt_chol;
            try {
                Rt_chol = arma::chol(arma::symmatu(Rt.slice(t+1)));
            } catch(...) {
                Rcout << "\n nn=" << nn << ", t=" << t << std::endl;

                arma::vec eigval;
                arma::mat eigvec;
                arma::eig_sym(eigval,eigvec,Rt.slice(t+1));

                Rcout << "Eigenvalues of Rt(t+1): " << eigval.t() << std::endl;
                ::Rf_error("Cholesky of Rt(t+1) in smoothing failed");

            }

            arma::mat Rt_ichol = arma::inv(arma::trimatu(Rt_chol));
            arma::mat Rt_inv = Rt_ichol * Rt_ichol.t();

            Bt = Ct.slice(t) * G_tilde.t() * Rt_inv;

            ht = mt.col(t) + Bt * (Theta_tilde.col(t+1) - at.col(t+1));
            Ht = Ct.slice(t) - Bt * G_tilde * Ct.slice(t).t();
            Ht = arma::symmatu(Ht);
            Ht.diag() += 1.e-5;

            arma::mat Ht_chol;
            try {
                Ht_chol = arma::chol(Ht);
                Theta_tilde.col(t) = ht + Ht_chol.t() * arma::randn(ht.n_elem,1);
            } catch(...) {
                arma::vec eigval;
                arma::mat eigvec;
                arma::eig_sym(eigval,eigvec,Ht);
                // Ht = U V U' = UV^{1/2} V^{1/2}U'
                // 
                arma::uvec idx = arma::find(eigval>0.);
                arma::vec eigval2 = eigval.elem(idx);
                arma::mat eigvec2 = eigvec.cols(idx);
                Ht_chol = arma::diagmat(arma::sqrt(eigval2)) * eigvec2.t();
                Theta_tilde.col(t) = ht + Ht_chol.t() * arma::randn(idx.n_elem,1);
                // Rcout << "\n nn=" << nn << ", t=" << t << std::endl;

                // arma::vec eigval;
                // arma::mat eigvec;
                // arma::eig_sym(eigval,eigvec,Ht);

                // Rcout << "Eigenvalues of Ht(t): " << eigval.t() << std::endl;
                // ::Rf_error("Cholesky of Ht(t) in smoothing failed");

            }
        }


        for (unsigned int t=(2*L); t>L; t--) {
            unsigned int idx = t - L - 1;
            arma::mat Ct_tmp = Ct.slice(t).submat(0,0,idx,idx);
            // Ct: (L+ns) x (L+ns)
            // For the L X L part
            
            arma::mat G_tmp = G_tilde.submat(0,0,idx,idx);

            arma::mat R_tmp = Rt.slice(t+1).submat(0,0,idx,idx);
            arma::mat Rt_chol;
            try {
                Rt_chol = arma::chol(arma::symmatu(R_tmp));
            } catch(...) {
                Rcout << "\n nn=" << nn << ", t=" << t << std::endl;

                arma::vec eigval;
                arma::mat eigvec;
                arma::eig_sym(eigval,eigvec,R_tmp);

                Rcout << "Eigenvalues of Rt(t+1): " << eigval.t() << std::endl;
                ::Rf_error("Cholesky of Rt(t+1) in smoothing failed");

            }

            arma::mat Rt_ichol = arma::inv(arma::trimatu(Rt_chol));
            arma::mat Rt_inv = Rt_ichol * Rt_ichol.t();

            Bt.zeros();
            Bt.submat(0,0,idx,idx) = Ct_tmp * G_tmp.t() * Rt_inv;

            ht = mt.col(t) + Bt * (Theta_tilde.col(t+1) - at.col(t+1));
            Ht = Ct.slice(t) - Bt * G_tilde * Ct.slice(t).t();
            Ht = arma::symmatu(Ht);
            Ht.diag() += 1.e-5;

            arma::mat Ht_chol;
            try {
                Ht_chol = arma::chol(Ht);
            } catch(...) {
                Rcout << "\n nn=" << nn << ", t=" << t << std::endl;

                arma::vec eigval;
                arma::mat eigvec;
                arma::eig_sym(eigval,eigvec,Ht);

                Rcout << "Eigenvalues of Ht(t): " << eigval.t() << std::endl;
                ::Rf_error("Cholesky of Ht(t) in smoothing failed");

            }

            Theta_tilde.col(t) = ht + Ht_chol.t() * arma::randn(ht.n_elem,1);
        }



        for (unsigned int t=L; t>0; t--) {
            Theta_tilde.col(t).zeros();
        }


        // (3) Sample evolution variance W
        // [Checked - OK]
        if (eta_type.at(0) == 1) {
            bw = eta_init.at(2,0); // rate
            for (unsigned int ti=L; ti<(L+nt); ti++) {
                bw += 0.5 * (Theta_tilde.at(0,ti)-Theta_tilde.at(0,ti-1)) * (Theta_tilde.at(0,ti)-Theta_tilde.at(0,ti-1));
            }

            W = 1./R::rgamma(aw,1./bw);
            W_tilde.at(0,0) = W;
        }


        if (eta_type.at(1) == 1) {
            // X: ns x (L+nt)
            // Omega: (L+nt) x ns
            arma::mat Omat(ns,ns,arma::fill::zeros);
            arma::vec Ovec = arma::sum(Omega.t() % X % X,1); // ns x (L+nt)
            Omat.diag() = Ovec;


            // K: (L+nt)xns
            // X: ns x (L+nt)
            // Psi: ns x (L+nt)
            // Omega: (L+nt) x ns
            arma::mat Kmat =  K.t() % X - Omega.t()%X%Psi;// ns x (L+nt)
            arma::vec kvec = arma::sum(Kmat,1);

            Sig_b = Tv*(Is - delta*Tvi*V)/sigma2 + Omat; // Prior 
            Sig_b = arma::inv_sympd(arma::symmatu(Sig_b)); 
            mu_b = Sig_b * kvec;

            // Sig_b = arma::diagmat(1./Ovec);
            // mu_b = Sig_b * kvec;
            arma::mat Chol_b = arma::chol(Sig_b);

            mu = mu_b + Chol_b.t() * arma::randn(ns,1);

        }

        // // (6) Sample delta_a
        // if (eta_type.at(3) == 1) {
        //     logit_delta_new = R::rnorm(logit_delta,sd_mh);
        //     delta_new = std::exp(logit_delta_new) / (1. + std::exp(logit_delta_new));
        //     logp_new = logit_delta_integrated_loglik(delta_new,Vtmp,V,Tvi,Is,ns,nt,a_delta,b_delta,a_sig2,b_sig2);

        //     logratio = std::min(0.,logp_new - logp);
        //     if (std::log(R::runif(0.,1.)) < logratio) {
        //         // accept
        //         mh_accept.at(nn) = 1.;
                    
        //         delta = delta_new;
        //         logit_delta = logit_delta_new;
        //         logp = logp_new;
        //     }
        // }

        // arma::mat Q = Tv*(Is-delta*Tvi*V); // ns x ns
        // // (5) Sample sig2_a
        // if (eta_type.at(2) == 1) {
            
        //     b_sig2_tilde = 0;
        //     for (unsigned int t=0; t<nt; t++) {
        //         arma::vec vtmp = Vtmp.col(t); // ns x 1
        //         b_sig2_tilde += arma::as_scalar(vtmp.t()*Q*vtmp);
        //     }
        //     b_sig2_tilde *= 0.5;
        //     b_sig2_tilde += b_sig2;
        //     sigma2 = 1./R::rgamma(a_sig2_tilde,1./b_sig2_tilde); // [TODO] check this part   
        // }

        // W_tilde.submat(L,L,L+ns-1,L+ns-1) = sigma2*Q.i();
        
		
		// store samples after burnin and thinning
		if (saveiter || nn==(ntotal-1)) {
			unsigned int idx_run;
			if (saveiter) {
				idx_run = (nn-nburnin-1)/nthin;
			} else {
				idx_run = nsample - 1;
			}

            Lambda_stored.slice(idx_run) = Lambda;
            Theta_stored.slice(idx_run) = Theta_tilde;

            mt_stored.col(idx_run) = mt.row(0).t();
            Ct_stored.col(idx_run) = arma::vectorise(Ct.tube(0,0));

            W_stored.at(idx_run) = W;
            mu_stored.col(idx_run) = mu;
            // sigma2_stored.at(idx_run) = sigma2;
            // delta_stored.at(idx_run) = delta;
		}

		if (verbose) {
			Rcout << "\rProgress: " << nn << "/" << ntotal-1;
		}
		
	}

	if (verbose) {
		Rcout << std::endl;
	}

	Rcpp::List output;
    output["Theta"] = Rcpp::wrap(Theta_stored);
    output["Lambda"] = Rcpp::wrap(Lambda_stored);
    output["mt"] = Rcpp::wrap(mt_stored);
    output["Ct"] = Rcpp::wrap(Ct_stored);
    
    if (eta_type.at(0) == 1) {
        // (3) Sample evolution variance W
        output["W"] = Rcpp::wrap(W_stored);
    }

    if (eta_type.at(1)==1) {
        // sample b and/or sig2_b,delta_b
        output["mu"] = Rcpp::wrap(mu_stored);
    }

    // if (eta_type.at(2)==1) {
    //     output["sigma2"] = Rcpp::wrap(sigma2_stored);
    // }

    // if (eta_type.at(3)==1) {
    //     output["delta"] = Rcpp::wrap(delta_stored);
    //     output["mh_accept"] = Rcpp::wrap(mh_accept);
    // }

	return output;
}






/**
 * Gibbs sampler with Metropolis-Hasting steps and Polya-Gamma augmentation.
 * 
 * 
 * Model
 * @f{align*}
 * y_{s,t}       &\sim\  \text{NB}(n_s,p_{s,t}) \\
 * p_{s,t}       &=      \frac{\exp(\lambda_{s,t})}{1 + \exp(\lambda_{s,t})} \\
 * \lambda_{s,t} &=      a_s + \sum_{l=1}^L R_{s,t-l} g_{l} \frac{y_{s,t-l}}{n_s} \\
 * R_{s,t}       &=      b_s + R_t \\
 * R_{t}         &=      R_{t-1} + w_{t},\ w_{t} \sim\mathcal{N}(0,W_{t})
 * @f}
 * 
 * 
 * Random variables
 * (1) @f$\{\bm{\theta}\}_{0}^{T}@f$: latent state.
 * (3) @f$\{W_t\}@f$: evolution variance(s).
 *      - 0: known.
 *      - 1: static evolution variance with IG(a_w,b_w) prior.
 *      - 2: time-varying evolution variance using discount factor.
 * (4) @f$\{a_s\}_{s=1}^{S}@f$: baseline with CAR prior.
 *      - 0: known.
 *      - 1: CAR prior.
 * (5) @f$\{b_s\}_{s=1}^{S}@f$: spatial effect with CAR prior.
 *      - 0: known.
 *      - 1: CAR prior.
 * (6) @f$\{\sigma^{2}_{a},\delta_{a}\}@f$: CAR prior parameters for a, with IG(0,0,a)-beta(a_{a\delta}, b_{a\delta}) prior.
 * (7) @f$\{\sigma^{2}_{b},\delta_{b}\}@f$: CAR prior parameters for b, with IG(0,0,a)-beta(a_{b\delta}, b_{b\delta}) prior.
 *      
 * 
 * 
 * @param Y S x T matrix, where S is the number of locations and T is the number of temporal observations.
 * @param npop S x 1 vector, number of population normalizer for each location.
 * @param V S x S unnormalize neighboring structure matrix with 0 and 1 as entries.
 * @param L integer as double value, length of temporal lags for transfer effect.
 * @param nburnin
 * @param nthin
 * @param nsample
 * @param eta_type 4 x 1 integer vector, specific which entry of eta is fixed or random with specific prior.
 * @f$\eta = \{
 *      W_t, b, \sigma^{2}_{b} and \delta_{b}
 * \}@f$.
 * @param eta_prior_val 3 x 3 numeric matrix
 *               (W)            (sig2_b)      (delta_b)
 * [fixed/init]  0.01             0.1            1
 * [param - 1]   a[w]=0.01      a[sigma]       a[delta]
 * [param - 2]   b[w]=0.01      b[sigma]       b[delta]
 * @param sig_mh standard deviaitons, step size for (delta_a, delta_b)
 * @param discount_factor
 * @param summarize_return
 * @param verbose
 * 
 * 
*/
//' @export
// [[Rcpp::export]]
Rcpp::List mcmc_polya_gamma(
	const arma::mat& Y, // S x T, the observed response
    const arma::vec& npop, // S x 1, number of population for each location
    const Rcpp::Nullable<Rcpp::NumericVector>& V_neighbor = R_NilValue, // S x S, neighboring structure matrix with 0 or 1 entries
	const unsigned int L = 12, // temporal lags for transfer effect
	const unsigned int nburnin = 0,
	const unsigned int nthin = 1,
	const unsigned int nsample = 1,
    const Rcpp::IntegerVector& eta_type = Rcpp::IntegerVector::create(0,0,0), // 3 x 1 integer vector, specific which entry of eta is fixed or random with specific prior.
    const Rcpp::Nullable<Rcpp::NumericMatrix>& eta_prior_val = R_NilValue,  // 4 x 3 numeric matrix
    const Rcpp::Nullable<Rcpp::NumericVector>& b_init = R_NilValue, // specify values of b if assuming known.
    const Rcpp::Nullable<Rcpp::NumericMatrix>& Theta_init = R_NilValue, // specify values of theta
    const Rcpp::Nullable<Rcpp::NumericVector>& m0_prior = R_NilValue,
	const Rcpp::Nullable<Rcpp::NumericMatrix>& C0_prior = R_NilValue,
    const Rcpp::Nullable<Rcpp::NumericVector>& Fphi_g = R_NilValue,
    const Rcpp::NumericVector mh_sd = Rcpp::NumericVector::create(1.,1.), // (delta,sigma2)
    const double discount_factor = 0.9,
    const double npop_scale = 1.,
	const bool summarize_return = false,
	const bool verbose = true) { // n x 1

    // Rcout << "\n enter";

    /*
    -----------------
    Input Processing
    -----------------
    */
    arma::mat eta_init(3,3,arma::fill::zeros);
    if (eta_prior_val.isNull()) {
        // W
        eta_init.at(0,0) = 0.01; // Initial/fixed value for W
        eta_init.at(1,0) = 0.01; // a_w in IG(a_w,b_w)
        eta_init.at(2,0) = 0.01; // b_w in IG(a_w,b_w)

        // sig2_b
        eta_init.at(0,1) = 0.1; // Initial/fixed value for sig2_b
        eta_init.at(1,1) = 0.01; // a_sig2
        eta_init.at(2,1) = 0.01; // b_sig2

        //delta_b
        eta_init.at(0,2) = 1.; // Initial/fixed value for delta_b
        eta_init.at(1,2) = 1.; // a_delta for Beta(a_delta, b_delta)
        eta_init.at(2,2) = 1.; // b_delta for Beta(a_delta, b_delta)

    } else {
        eta_init = Rcpp::as<arma::mat>(eta_prior_val);
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
    const arma::vec Jt(nt+1,arma::fill::ones);

    arma::mat Zpad(ns,L+nt,arma::fill::zeros); // ns x (nt+L)
    arma::mat Ypad(ns,L+nt,arma::fill::zeros);
    Zpad.cols(L,L+nt-1) = Y;
    Ypad.cols(L,L+nt-1) = Y; 

    const arma::mat Is(ns,ns,arma::fill::eye);
    const arma::mat Il(L,L,arma::fill::eye);

    arma::mat V(ns,ns,arma::fill::eye);
    if (!V_neighbor.isNull()) {
        V = Rcpp::as<arma::mat>(V_neighbor);
    }
    const arma::vec vrow = arma::sum(V,1);
    const arma::mat Tv = arma::diagmat(vrow);
    const arma::mat Tvi = arma::diagmat(1./vrow);
    /* ----------------- */
    // Rcout << "\n Done pad";


    /*
    -----------------
    Variables - Forward Filtering Backward Sampling
    -----------------
    */
    arma::vec theta(L,arma::fill::zeros);
    arma::mat Theta(L,L+nt,arma::fill::zeros);
    if (!Theta_init.isNull()) {
        Theta = Rcpp::as<arma::mat>(Theta_init);
    }
    arma::mat G(L,L,arma::fill::zeros);
    G.at(0,0) = 1.;
    G.diag(-1).ones();

    arma::mat at(L,nt+L,arma::fill::zeros);
    arma::cube Rt(L,L,nt+L);
    // arma::vec ft(ns,arma::fill::zeros);
    // arma::mat Qt(ns,ns,arma::fill::eye);

    // arma::mat At(L,ns,arma::fill::zeros);
    // arma::mat Bt(L,L,arma::fill::zeros);

    arma::mat mt(L,L+nt,arma::fill::zeros);
    arma::cube Ct(L,L,L+nt);
    if (!m0_prior.isNull()) {
		mt.col(L-1) = Rcpp::as<arma::vec>(m0_prior);
	}

	if (!C0_prior.isNull()) {
		Ct.slice(L-1) = Rcpp::as<arma::mat>(C0_prior);
	} else {
        Ct.slice(L-1) = 10. * arma::mat(L,L,arma::fill::eye);
    }

    arma::vec ht(L,arma::fill::zeros);
    arma::mat Ht(L,L,arma::fill::ones);


    double coef1 = 1. - discount_factor;
    double coef2 = discount_factor * discount_factor;
    /* ----------------- */
    // Rcout << "\n Done FFBS";




    /*
    -----------------
    Temporal parameters for MCMC
    -----------------
    */
    arma::vec b(ns,arma::fill::zeros); 
    if (!b_init.isNull()) {
        b = arma::vec(Rcpp::as<Rcpp::NumericVector>(b_init).begin(),ns);
    }
    // arma::vec mu_b(ns,arma::fill::zeros);
    // arma::mat Sig_b(ns,ns,arma::fill::eye);

/*               (W)        (sig2_b)      (delta_b)
 * [fixed/init]  0.01         0.1           1
 * [param - 1]   0.01         0             1
 * [param - 2]   0.01         0             1
 * [param - 3]   0(x)         1             0(x)
 */   

    double W = eta_init.at(0,0);
    double aw = eta_init.at(1,0) + 0.5*nt_;
    double bw;

    arma::vec car_para(4); // (delta,sigma2,logp,mh_accept)
    car_para.at(0) = eta_init.at(0,2); // delta
    car_para.at(1) = eta_init.at(0,1); // sigma2
    car_para.at(2) = 1.;
    car_para.at(3) = 0.;

    double a_sig2 = eta_init.at(1,1);
    double b_sig2 = eta_init.at(2,1);
    double a_delta = eta_init.at(1,2);
    double b_delta = eta_init.at(2,2);


    arma::mat Omat(ns,ns,arma::fill::eye);
    arma::vec kvec(ns,arma::fill::zeros);
    arma::vec dvec(ns,arma::fill::zeros);

    arma::cube Theta_stored(L,nt+L,nsample);
    arma::mat mt_stored(L+nt,nsample);
    arma::mat Ct_stored(L+nt,nsample);
    arma::vec W_stored(nsample);
    arma::mat a_stored(ns,nsample);
    arma::mat a_param(2,nsample); // (a_sig2a,b_sig2a)
    arma::mat b_stored(ns,nsample);
    arma::mat b_param(2,nsample); // (a_sig2b,b_sig2b)
    arma::vec sigma2_stored(nsample);
    arma::vec delta_stored(nsample);

    // Rcout << "\n Done W";

    arma::vec mh_accept(ntotal,arma::fill::zeros); // (col 1 = a, col 2 = b)
    bool saveiter;
    unsigned int ir, ic;

    // Rcout << "\n Done MH";
    
    /*
    -----------------
    Temporal parameters for MCMC
    -----------------
    */
    arma::vec Fphi;
    if (Fphi_g.isNull()) {
        Fphi = get_Fphi(L);
    } else {
        Fphi = Rcpp::as<arma::vec>(Fphi_g);
    }
    arma::vec Fy(L,arma::fill::zeros);
    arma::vec Fts(L,arma::fill::zeros);
    arma::cube Ft(L,ns,L+nt);
    arma::mat X(ns,L+nt,arma::fill::zeros);
    arma::mat K(ns,L+nt,arma::fill::zeros);

    for (unsigned int t=L; t<(L+nt); t++) {
        arma::mat Ytmp = arma::reverse(Ypad.cols(t-L,t-1),1); // ns x L
        for (unsigned int s=0; s<ns; s++) {
            K.at(s,t) = 0.5*(Ypad.at(s,t)-npop.at(s));
            Ytmp.row(s) = Ytmp.row(s) % Fphi.t() / (npop_scale*npop.at(s));
        }
        Ft.slice(t) = Ytmp.t(); // L x ns
        X.col(t) = arma::vectorise(arma::sum(Ft.slice(t),0));
    }

    X.elem(arma::find(X<1.e-5)).fill(1.e-5);


    // arma::cube Lambda_full(ns,L+nt,ntotal,arma::fill::zeros); // ns x (nt+1)
    // Lambda_full.slice(0).zeros();
    // arma::cube Omega_full(ns,L+nt,ntotal,arma::fill::zeros);
    // Omega_full.slice(0).zeros();
    // for (unsigned int nn=1; nn<ntotal; nn++) {
	// 	R_CheckUserInterrupt();
    //     /*
    //     (1) Sample auxliary Polya-Gamma variables: Omega, Lambda, Zmat.
    //     */
        
    // }


    arma::mat Psi(ns,L+nt,arma::fill::zeros);
    arma::mat Z(ns,L+nt,arma::fill::zeros);
    arma::mat Omega(ns,L+nt,arma::fill::zeros);
    arma::mat Lambda(ns,L+nt,arma::fill::zeros);

    arma::cube Lambda_stored(ns,L+nt,nsample);
    arma::cube Omega_stored(ns,L+nt,nsample);

	for (unsigned int nn=0; nn<ntotal; nn++) {
		R_CheckUserInterrupt();
		saveiter = nn > nburnin && ((nn-nburnin-1)%nthin==0);

        // arma::mat Omega = Omega_full.slice(nn); // ns x (L+nt)
        for (unsigned int s=0; s<ns; s++) {
            for (unsigned int t=L; t<(L+nt); t++) {
                // [TODO] order?
                Omega.at(s,t) = pg::rpg_scalar_hybrid(npop.at(s) + Ypad.at(s,t),Lambda.at(s,t));
                Lambda.at(s,t) = K.at(s,t)/Omega.at(s,t) + 1./std::sqrt(Omega.at(s,t)) * R::rnorm(0.,1.);
            }
        }

        if (eta_type.at(0) == 1) { // Sample theta and W

        for (unsigned int s=0; s<ns; s++) {
            for (unsigned int t=L; t<(L+nt); t++) {
                // [TODO] order?
                Z.at(s,t) = K.at(s,t) / Omega.at(s,t) - X.at(s,t)*b.at(s);
            }
        }

        // Rcout << "\n Done pg";


        /*
        (2) Sample augmented latent states Theta using forward filtering and backward sampling.
        */

        // Forward filtering
        // [Checked - OK]
        for (unsigned int t=L; t<(L+nt); t++) {
            arma::mat Ot_inv = arma::diagmat(1./Omega.col(t)); // ns x ns
            arma::mat Ft_tmp = Ft.slice(t); // L x ns

            at.col(t) = G * mt.col(t-1); // L x 1
            Rt.slice(t) = G * Ct.slice(t-1) * G.t();

            if (nn<nburnin) {
                Rt.at(0,0,t) = Rt.at(0,0,t) / discount_factor;
            } else {
                Rt.at(0,0,t) = Rt.at(0,0,t) + W;
            }
            
                
            arma::vec ft = Ft_tmp.t() * at.col(t); // ns x 1 = (ns x L) * (L x 1)
            arma::mat Qt = Ft_tmp.t() * Rt.slice(t) * Ft_tmp + Ot_inv; // ns x ns
            Qt = arma::symmatu(Qt);

            arma::mat Qt_chol;
            try {
                Qt_chol = arma::chol(Qt); // Q = R'R, R: upper triangular part of Cholesky decomposition
            } catch (...) {
                arma::vec eigval;
                arma::mat eigvec;
                arma::eig_sym(eigval,eigvec,Qt);
                    
                Rcout << "(nn = " << nn << ", t = " << t;
                Rcout << ", rcond = " << arma::rcond(Qt) << ")" << std::endl;
                Rcout << "eigval = " << eigval.t() << std::endl;
            }
                
            arma::mat Qt_ichol = arma::inv(arma::trimatu(Qt_chol)); // R^{-1}: inverse of upper-triangular cholesky component
            arma::mat Qt_inv = Qt_ichol * Qt_ichol.t(); // inverse Qt^{-1} = R^{-1} (R^{-1})'

            arma::mat At = Rt.slice(t) * Ft_tmp * Qt_inv; // L x ns
            mt.col(t) = at.col(t) + At * (Z.col(t) - ft); // Z[s,t] or Psi[s,t]
            Ct.slice(t) = Rt.slice(t) - At * Ft_tmp.t() * Rt.slice(t);
        }


        // Rcout << "\n Done FF";


        // Backward sampling
        // [Checked - OK]
        ht = mt.col(nt+L-1);
        Ht = Ct.slice(nt+L-1);
        {
            arma::mat Ht_chol;
            try {
                Ht_chol = arma::chol(arma::symmatu(Ht));
            } catch(...) {
                Ht_chol = arma::chol(arma::symmatu(Ct.slice(nt+L-1)*(1.-discount_factor)));

            }

            Theta.col(nt+L-1) = ht + Ht_chol.t() * arma::randn(L,1);
        }

        // Rcout << "\n Done bs1";
        
        // Theta: L x (nt+1)
        // Theta.at(0,nt) = R::rnorm(ht.at(nt),std::sqrt(Ht.at(nt)));
        for (unsigned int t=(L+nt-2); t>(2*L); t--) {
            arma::mat Rt_chol;
            try {
                Rt_chol = arma::chol(arma::symmatu(Rt.slice(t+1)));
            } catch(...) {
                Rcout << "\n nn=" << nn << ", t=" << t << std::endl;

                arma::vec eigval;
                arma::mat eigvec;
                arma::eig_sym(eigval,eigvec,Rt.slice(t+1));

                Rcout << "Eigenvalues of Rt(t+1): " << eigval.t() << std::endl;
                ::Rf_error("Cholesky of Rt(t+1) in smoothing failed");

            }

            arma::mat Rt_ichol = arma::inv(arma::trimatu(Rt_chol));
            arma::mat Rt_inv = Rt_ichol * Rt_ichol.t();

            arma::mat Bt = Ct.slice(t) * G.t() * Rt_inv;

            ht = mt.col(t) + Bt * (Theta.col(t+1) - at.col(t+1));
            Ht = Ct.slice(t) - Bt * G * Ct.slice(t).t();
            Ht = arma::symmatu(Ht + 1.e-5*Il);

            arma::mat Ht_chol;
            try {
                Ht_chol = arma::chol(Ht);
            } catch(...) {
                Ht_chol = arma::chol(arma::symmatu(Ct.slice(t)*(1.-discount_factor)));

            }

            Theta.col(t) = ht + Ht_chol.t() * arma::randn(L,1);

        }
        // Rcout << "\n Done bs2";

        for (unsigned int t=(2*L); t>L; t--) {
            unsigned int idx = t - L - 1;
            arma::mat Ct_tmp = Ct.slice(t).submat(0,0,idx,idx);
            
            arma::mat G_tmp = G.submat(0,0,idx,idx);

            arma::mat R_tmp = Rt.slice(t+1).submat(0,0,idx,idx);
            arma::mat Rt_chol;
            try {
                Rt_chol = arma::chol(arma::symmatu(R_tmp));
            } catch(...) {
                Rcout << "\n nn=" << nn << ", t=" << t << std::endl;

                arma::vec eigval;
                arma::mat eigvec;
                arma::eig_sym(eigval,eigvec,R_tmp);

                Rcout << "Eigenvalues of Rt(t+1): " << eigval.t() << std::endl;
                ::Rf_error("Cholesky of Rt(t+1) in smoothing failed");

            }

            arma::mat Rt_ichol = arma::inv(arma::trimatu(Rt_chol));
            arma::mat Rt_inv = Rt_ichol * Rt_ichol.t();

            arma::mat Bt(L,L,arma::fill::zeros);
            Bt.submat(0,0,idx,idx) = Ct_tmp * G_tmp.t() * Rt_inv;

            ht = mt.col(t) + Bt * (Theta.col(t+1) - at.col(t+1));
            Ht = Ct.slice(t) - Bt * G * Ct.slice(t).t();
            Ht = arma::symmatu(Ht + 1.e-5*Il);


            arma::mat Ht_chol;
            try {
                Ht_chol = arma::chol(Ht);
            } catch(...) {
                Ht_chol = arma::chol(arma::symmatu(Ct.slice(t)*(1.-discount_factor)));

            }

            Theta.col(t) = ht + Ht_chol.t() * arma::randn(L,1);
        }

        // Rcout << "\n Done bs3";

        for (unsigned int t=L; t>0; t--) {
            Theta.col(t).zeros();
        }



        // (3) Sample evolution variance W
        // [Checked - OK]
            bw = eta_init.at(2,0); // rate
            for (unsigned int ti=L; ti<(L+nt); ti++) {
                bw += 0.5 * (Theta.at(0,ti)-Theta.at(0,ti-1)) * (Theta.at(0,ti)-Theta.at(0,ti-1));
            }

            W = 1./R::rgamma(aw,1./bw);
        }

        // Rcout << "\n Done W";

        arma::vec Cs(ns,arma::fill::zeros);
        arma::vec Ds(ns,arma::fill::zeros);
 
        for (unsigned int s=0; s<ns; s++) {
            double ctmp = 0.;
            double dtmp = 0.;
            for (unsigned int t=L; t<(L+nt); t++) {
                // arma::cube Ft(L,ns,L+nt);
                // arma::mat Theta(L,L+nt,arma::fill::zeros);
                arma::mat Ft_tmp = Ft.slice(t); // L x ns
                // double tmp = arma::dot(Fs_tmp.col(t),Theta.col(t));
                Z.at(s,t) = K.at(s,t) / Omega.at(s,t);
                Psi.at(s,t) = arma::dot(Ft_tmp.col(s),Theta.col(t));
                ctmp += (Omega.at(s,t) * X.at(s,t) * X.at(s,t));
                dtmp += (Omega.at(s,t) * X.at(s,t) * (Z.at(s,t) - Psi.at(s,t)));
            }

            Cs.at(s) = ctmp;
            Ds.at(s) = dtmp;
        }



        // car_para(4) = (delta,sigma2,logp,mh_accept)
        if (eta_type.at(1)==1) {
            // (4) Sample spatial effect b=(b[1],...,b[ns])
            arma::mat Q = Tv * (Is - car_para.at(0)*Tvi*V) / car_para.at(1);
            arma::mat Cmat = arma::diagmat(Cs);
            arma::mat Prec = Q + Cmat; // precision
            arma::mat Sig = arma::inv_sympd(arma::symmatu(Prec)); // variance

            arma::vec mu_b = Sig * Ds;
            arma::mat Sig_chol = arma::chol(arma::symmatu(Sig));

            arma::vec btmp = mu_b + Sig_chol.t()*arma::randn(ns,1);
            b = btmp - arma::accu(btmp)/static_cast<double>(ns);
        }

        // Rcout << "\n Done mu";

        // (5) Sample delta and sigma2
        if (eta_type.at(2) == 1) {
            // b: ns x 1
            // m
            // car_para(4) = (delta,sigma2,logp,mh_accept)
            try{
                arma::vec car_tmp = update_car_para(
                    car_para,b,Omat,kvec,V,Tv,Tvi,mh_sd,
                    a_delta,b_delta,a_sig2,b_sig2);
                car_para = car_tmp;
                mh_accept.at(nn) = car_para.at(3);
            } catch(...) {
                mh_accept.at(nn) = 0.;
            }
        }
        
		
		// store samples after burnin and thinning
		if (saveiter || nn==(ntotal-1)) {
			unsigned int idx_run;
			if (saveiter) {
				idx_run = (nn-nburnin-1)/nthin;
			} else {
				idx_run = nsample - 1;
			}

            Lambda_stored.slice(idx_run) = Lambda;
            Omega_stored.slice(idx_run) = Omega;

            Theta_stored.slice(idx_run) = Theta;

            mt_stored.col(idx_run) = mt.row(0).t();
            Ct_stored.col(idx_run) = arma::vectorise(Ct.tube(0,0));

            W_stored.at(idx_run) = W;
            b_stored.col(idx_run) = b;

            delta_stored.at(idx_run) = car_para.at(0);
            sigma2_stored.at(idx_run) = car_para.at(1);

		}

		if (verbose) {
			Rcout << "\rProgress: " << nn << "/" << ntotal-1;
		}
		
	}

	if (verbose) {
		Rcout << std::endl;
	}

	Rcpp::List output;
    output["Theta"] = Rcpp::wrap(Theta_stored);
    output["Lambda"] = Rcpp::wrap(Lambda_stored);
    output["Omega"] = Rcpp::wrap(Omega_stored);
    output["K"] = Rcpp::wrap(K);

    output["mt"] = Rcpp::wrap(mt_stored);
    output["Ct"] = Rcpp::wrap(Ct_stored);
    output["W"] = Rcpp::wrap(W_stored);
    output["b"] = Rcpp::wrap(b_stored);
    output["sigma2"] = Rcpp::wrap(sigma2_stored);
    output["delta"] = Rcpp::wrap(delta_stored);
    output["mh_accept"] = Rcpp::wrap(mh_accept);

    output["Omat"] = Rcpp::wrap(Omat);
    output["kvec"] = Rcpp::wrap(kvec);
    output["X"] = Rcpp::wrap(X);


	return output;
}

/**
 * Transpose of Ft
*/
//' @export
// [[Rcpp::export]]
arma::cube get_Ftt(
    const arma::mat& Ypad, // ns x (nt+np)
    const arma::vec& npop, // ns x 1
    const arma::vec& Fphi_g,
    const double npop_scale=1.) { // np x 1

    const unsigned int np = Fphi_g.n_elem;
    const unsigned int ns = npop.n_elem;
    const unsigned int nt = Ypad.n_cols - np;

    const unsigned int np_ast = np*(ns+1);
    const arma::vec Js(ns,arma::fill::ones);

    arma::cube Ftt(ns,np_ast,nt+np);
    for (unsigned int t=0; t<np; t++) {
        Ftt.slice(t).zeros();
    }


    arma::mat Ysub(ns,np,arma::fill::zeros);
    arma::mat Ft(ns,np,arma::fill::zeros);
    arma::vec Iy(ns*np,arma::fill::zeros);
    arma::mat Iy2(ns,ns*np,arma::fill::zeros);

    for (unsigned int t=np; t<(np+nt); t++) {
        Ysub = arma::reverse(Ypad.cols(t-np,t-1),1); // ns x np
        for (unsigned int s=0; s<ns; s++) {
            Ysub.row(s) /= (npop_scale*npop.at(s));
        }

        Ft = arma::kron(Js,Fphi_g.t()) % Ysub; // ns x np
        Iy = arma::vectorise(Ft); // (ns*np) x 1
        Iy2 = arma::kron(Js,Iy.t()); // ns x (ns*np)
        Ftt.slice(t).cols(0,np-1) = Ft; // ns x np
        Ftt.slice(t).cols(np,np_ast-1) = Iy2; // ns x (ns*np)
    }

    return Ftt; // ns x (np*(ns+1)) x (nt+np)
}




//' @export
// [[Rcpp::export]]
arma::mat get_G(
    const unsigned int ns,
    const unsigned int np
) {
    unsigned int np_ast = np*(ns+1);
    arma::mat Is(ns,ns,arma::fill::eye);
    
    arma::mat G0(np,np,arma::fill::zeros);
    G0.at(0,0) = 1.;
    G0.diag(-1).ones();

    arma::mat G_tilde(np_ast,np_ast,arma::fill::zeros);
    G_tilde.submat(0,0,np-1,np-1) = G0;
    for (unsigned int i=1; i<np; i++) {
        unsigned int i0 = np + (i-1) * ns; // L to L*(ns+1)-2*ns
        unsigned int i1 = np + i * ns; // L+ns to L*ns
        unsigned int i2 = np + (i+1) * ns; // L+2*ns to L*(ns+1)

        G_tilde.submat(i1,i0,i2-1,i1-1) = Is;
    }

    return G_tilde;
}


void mcmc_ffbs(
    const arma::mat& Z, // ns x nt
    const arma::mat& G // 
){
    return;
}


// /*
// The augmented model: temporal + spatio-temporal effects
// */
// //' @export
// // [[Rcpp::export]]
// Rcpp::List mcmc_polya_gamma2(
// 	const arma::mat& Y, // S x T, the observed response
//     const arma::vec& npop, // S x 1, number of population for each location
//     const Rcpp::Nullable<Rcpp::NumericVector>& V_neighbor = R_NilValue, // S x S, neighboring structure matrix with 0 or 1 entries
// 	const unsigned int L = 12, // temporal lags for transfer effect
// 	const unsigned int nburnin = 0,
// 	const unsigned int nthin = 1,
// 	const unsigned int nsample = 1,
//     const Rcpp::IntegerVector& eta_type = Rcpp::IntegerVector::create(0,0,0,0), // 4 x 1 integer vector, specific which entry of eta is fixed or random with specific prior.
//     const Rcpp::Nullable<Rcpp::NumericMatrix>& eta_prior_val = R_NilValue,  // 3 x 3 numeric matrix
//     const Rcpp::Nullable<Rcpp::NumericVector>& mu_init = R_NilValue, // specify values of mu if assuming known.
//     const Rcpp::Nullable<Rcpp::NumericVector>& m0_prior = R_NilValue,
// 	const Rcpp::Nullable<Rcpp::NumericMatrix>& C0_prior = R_NilValue,
//     const Rcpp::Nullable<Rcpp::NumericVector>& Fphi_g = R_NilValue,
//     const double sd_mh = 0.1,
//     const double discount_factor = 0.9,
//     const double npop_scale = 1.,
// 	const bool summarize_return = false,
// 	const bool verbose = true) { // n x 1

//     /*
//     -----------------
//     Input Processing
//     -----------------
//     */
//     arma::mat eta_init(3,3,arma::fill::zeros);
//     if (eta_prior_val.isNull()) {
//         // W
//         eta_init.at(0,0) = 0.01; // Initial/fixed value for W
//         eta_init.at(1,0) = 0.01; // a_w in IG(a_w,b_w)
//         eta_init.at(2,0) = 0.01; // b_w in IG(a_w,b_w)

//         // sig2_b
//         eta_init.at(0,1) = 0.1; // Initial/fixed value for sig2_b
//         eta_init.at(1,1) = 0.01; // a[sigma]
//         eta_init.at(2,1) = 0.01; // b[sigma]

//         //delta_b
//         eta_init.at(0,2) = 1.; // Initial/fixed value for b
//         eta_init.at(1,2) = 1.; // a[delta] for Beta(a[delta], b[delta])
//         eta_init.at(2,2) = 1.; // b[delta] for Beta(a[delta], b[delta])

//     } else {
//         eta_init = Rcpp::as<arma::mat>(eta_prior_val);
//     }
//     /* ----------------- */





//     /*
//     -----------------
//     Variable - Counting
//     -----------------
//     */
//     const unsigned int ns = Y.n_rows; // S, number of location
// 	const unsigned int nt = Y.n_cols; // T, number of temporal observations
//     const double nt_ = static_cast<double>(nt);
//     const unsigned int np_ast = L*(ns+1);

// 	const unsigned int ntotal = nburnin + nthin*nsample + 1; // number of iterations for MCMC

//     const arma::vec Js(ns,arma::fill::zeros);
//     const arma::vec Jl(L,arma::fill::ones);
//     const arma::vec Jt(nt+1,arma::fill::ones);
//     const arma::mat Is(ns,ns,arma::fill::eye);

//     arma::mat Zpad(ns,L+nt,arma::fill::zeros); // ns x (nt+L)
//     arma::mat Ypad(ns,L+nt,arma::fill::zeros);
//     Zpad.cols(L,L+nt-1) = Y;
//     Ypad.cols(L,L+nt-1) = Y; 

//     arma::mat V(ns,ns,arma::fill::eye);
//     if (!V_neighbor.isNull()) {
//         V = Rcpp::as<arma::mat>(V_neighbor);
//     }
//     const arma::vec vrow = arma::sum(V,1);
//     const arma::mat Tv = arma::diagmat(vrow);
//     const arma::mat Tvi = arma::diagmat(1./vrow);
//     /* ----------------- */


//     /*
//     -----------------
//     Variables - Forward Filtering Backward Sampling
//     np_ast = L * (ns+1)
//     -----------------
//     */
//     arma::vec theta_tilde(np_ast,arma::fill::zeros);
//     arma::mat Theta_tilde(np_ast,L+nt,arma::fill::zeros);
//     arma::mat G_tilde = get_G(ns,L); // np_ast * np_ast
    


//     arma::mat Ot_inv(ns,ns,arma::fill::zeros);

//     arma::mat at(np_ast,nt+L,arma::fill::zeros);
//     arma::cube Rt(np_ast,np_ast,nt+L);
//     arma::vec ft(ns,arma::fill::zeros);
//     arma::mat Qt(ns,ns,arma::fill::eye);

//     arma::mat At(np_ast,ns,arma::fill::zeros);
//     arma::mat Bt(np_ast,np_ast,arma::fill::zeros);

//     arma::mat mt(np_ast,L+nt,arma::fill::zeros);
//     arma::cube Ct(np_ast,np_ast,L+nt);
//     if (!m0_prior.isNull()) {
// 		mt.col(L-1) = Rcpp::as<arma::vec>(m0_prior);
// 	}

// 	if (!C0_prior.isNull()) {
// 		Ct.slice(L-1) = Rcpp::as<arma::mat>(C0_prior);
// 	} else {
//         Ct.slice(L-1) = 10. * arma::mat(Ct.n_rows,Ct.n_cols,arma::fill::eye);
//     }

//     arma::vec ht(np_ast,arma::fill::zeros);
//     arma::mat Ht(np_ast,np_ast,arma::fill::eye);


//     double coef1 = 1. - discount_factor;
//     double coef2 = discount_factor * discount_factor;
//     /* ----------------- */



//     /*
//     -----------------
//     Temporal parameters for MCMC
//     -----------------
//     */
//     arma::vec mu(ns,arma::fill::zeros); 
//     if (!mu_init.isNull()) {
//         mu = arma::vec(Rcpp::as<Rcpp::NumericVector>(mu_init).begin(),ns);
//     }
//     arma::vec mu_b(ns,arma::fill::zeros);
//     arma::mat Sig_b(ns,ns,arma::fill::eye);

    

//     double W = eta_init.at(0,0);
//     double aw = eta_init.at(1,0) + 0.5*nt_;
//     double bw = eta_init.at(2,0);
//     arma::mat W_tilde(np_ast,np_ast,arma::fill::zeros);
//     W_tilde.at(0,0) = W;
//     W_tilde.submat(L,L,L+ns-1,L+ns-1) = Sig_b;


//     double sigma2 = eta_init.at(0,1);
//     double a_sig2 = eta_init.at(1,1);
//     double b_sig2 = eta_init.at(2,1);
//     double a_sig2_tilde = a_sig2 + 0.5*static_cast<double>(nt*ns);
//     double b_sig2_tilde;


//     double delta = eta_init.at(0,2);
//     double a_delta = eta_init.at(1,2);
//     double b_delta = eta_init.at(2,2);
//     double logit_delta_new, delta_new, logp_new;
//     double logit_delta= std::log(delta) - std::log(1.-delta);
//     arma::vec out_tmp = logit_delta_integrated_loglik(delta,Theta_tilde.submat(L,L,L+ns-1,L+nt-1),V,Tvi,Is,ns,nt,a_delta,b_delta,a_sig2,b_sig2);
//     double logp = out_tmp.at(0);
//     double logratio;
//     arma::vec mh_accept(ntotal,arma::fill::zeros);


//     arma::mat Omat(ns,ns,arma::fill::eye);
//     arma::vec kvec(ns,arma::fill::zeros);
//     arma::vec dvec(ns,arma::fill::zeros);

//     arma::cube Theta_stored(np_ast,nt+L,nsample,arma::fill::zeros);
//     arma::mat mt_stored(L+nt,nsample,arma::fill::zeros);
//     arma::mat Ct_stored(L+nt,nsample,arma::fill::zeros);
//     arma::vec W_stored(nsample,arma::fill::zeros);
//     arma::mat mu_stored(ns,nsample,arma::fill::zeros);
//     arma::vec sigma2_stored(nsample,arma::fill::zeros);
//     arma::vec delta_stored(nsample,arma::fill::zeros);


    
//     bool saveiter;
//     unsigned int ir, ic;
    
//     /*
//     -----------------
//     Temporal parameters for MCMC
//     -----------------
//     */
//     arma::vec Fphi = get_Fphi(L);
//     if (!Fphi_g.isNull()) {
//         Fphi = Rcpp::as<arma::vec>(Fphi_g);
//     }
//     arma::cube Ftt = get_Ftt(Ypad,npop,Fphi,npop_scale); // ns x (L*(ns+1)) x (nt*L)

//     arma::mat Omega(L+nt,ns,arma::fill::zeros);
//     arma::mat K(L+nt,ns,arma::fill::zeros);
//     arma::mat Lambda(ns,L+nt,arma::fill::zeros); // ns x (nt+1)
//     arma::mat Z_tilde(ns,L+nt,arma::fill::zeros);
    
//     arma::cube Lambda_stored(ns,L+nt,nsample);


// 	for (unsigned int nn=0; nn<ntotal; nn++) {
// 		R_CheckUserInterrupt();
// 		saveiter = nn > nburnin && ((nn-nburnin-1)%nthin==0);

//         /*
//         (1) Sample auxliary Polya-Gamma variables: Omega, Lambda, Zmat.
//         */
//         for (unsigned int s=0; s<ns; s++) {
//             for (unsigned int t=L; t<(L+nt); t++) {
//                 K.at(t,s) = 0.5 * (Ypad.at(s,t) - npop.at(s));
//                 Omega.at(t,s) = pg::rpg_scalar_hybrid(npop.at(s) + Ypad.at(s,t),Lambda.at(s,t));
//                 Z_tilde.at(s,t) = K.at(t,s)/Omega.at(t,s);
//                 Lambda.at(s,t) = R::rnorm(Z_tilde.at(s,t), 1./std::sqrt(Omega.at(t,s)));
//             }
//         }



//         /*
//         (2) Sample augmented latent states Theta using forward filtering and backward sampling.
//         */

//         // Forward filtering
//         // [Checked - OK]
//         for (unsigned int t=L; t<(L+nt); t++) {
//             Ot_inv = arma::diagmat(1./Omega.row(t)); // ns x ns

//             at.col(t) = G_tilde * mt.col(t-1); 
//             // (L+ns) x 1, the last ns elements are all zeros
//             Rt.slice(t) = G_tilde * Ct.slice(t-1) * G_tilde.t();

//             // if (nn<nburnin) {
//             //     Rt.at(0,0,t) = Rt.at(0,0,t) / discount_factor;
//             // } else {
//                 Rt.slice(t) = Rt.slice(t) + W_tilde;
//                 // only the first L x L elements are nonzero.
//             // }
            
                
//             ft = Ftt.slice(t) * at.col(t); // ns x 1 = (ns x L) * (L x 1)
//             Qt = Ftt.slice(t) * Rt.slice(t) * Ftt.slice(t).t() + Ot_inv; // ns x ns
//             Qt = arma::symmatu(Qt);

//             arma::mat Qt_chol;
//             try {
//                 Qt_chol = arma::chol(Qt); // Q = R'R, R: upper triangular part of Cholesky decomposition
//             } catch (...) {
//                 arma::vec eigval;
//                 arma::mat eigvec;
//                 arma::eig_sym(eigval,eigvec,Qt);
                    
//                 Rcout << "(nn = " << nn << ", t = " << t;
//                 Rcout << ", rcond = " << arma::rcond(Qt) << ")" << std::endl;
//                 Rcout << "eigval = " << eigval.t() << std::endl;
//             }
                
//             arma::mat Qt_ichol = arma::inv(arma::trimatu(Qt_chol)); // R^{-1}: inverse of upper-triangular cholesky component
//             arma::mat Qt_inv = Qt_ichol * Qt_ichol.t(); // inverse Qt^{-1} = R^{-1} (R^{-1})'

//             At = Rt.slice(t) * Ftt.slice(t).t() * Qt_inv;
//             mt.col(t) = at.col(t) + At * (Z_tilde.col(t) - ft); 
//             Ct.slice(t) = Rt.slice(t) - At * Ftt.slice(t) * Rt.slice(t);
//         }



//         // Backward sampling
//         // [Checked - OK]
//         ht = mt.col(nt+L-1);
//         Ht = Ct.slice(nt+L-1);
//         {
//             arma::mat Ht_chol;
//             try {
//                 Ht_chol = arma::chol(arma::symmatu(Ht));
//             } catch(...) {
//                 Rcout << "\n nn=" << nn << ", t=" << nt+L-1 << std::endl;

//                 arma::vec eigval;
//                 arma::mat eigvec;
//                 arma::eig_sym(eigval,eigvec,Ht);

//                 Rcout << "Eigenvalues of Ht(nt+L-1): " << eigval.t() << std::endl;
//                 ::Rf_error("Cholesky of Ht(nt+L-1) in smoothing failed");

//             }

//             Theta_tilde.col(nt+L-1) = ht + Ht_chol.t() * arma::randn(ht.n_elem,1);
//         }
        
//         // Theta: L x (nt+1)
//         // Theta.at(0,nt) = R::rnorm(ht.at(nt),std::sqrt(Ht.at(nt)));
//         for (unsigned int t=(L+nt-2); t>(2*L); t--) {
//             arma::mat Rt_chol;
//             try {
//                 Rt_chol = arma::chol(arma::symmatu(Rt.slice(t+1)));
//             } catch(...) {
//                 Rcout << "\n nn=" << nn << ", t=" << t << std::endl;

//                 arma::vec eigval;
//                 arma::mat eigvec;
//                 arma::eig_sym(eigval,eigvec,Rt.slice(t+1));

//                 Rcout << "Eigenvalues of Rt(t+1): " << eigval.t() << std::endl;
//                 ::Rf_error("Cholesky of Rt(t+1) in smoothing failed");

//             }

//             arma::mat Rt_ichol = arma::inv(arma::trimatu(Rt_chol));
//             arma::mat Rt_inv = Rt_ichol * Rt_ichol.t();

//             Bt = Ct.slice(t) * G_tilde.t() * Rt_inv;

//             ht = mt.col(t) + Bt * (Theta_tilde.col(t+1) - at.col(t+1));
//             Ht = Ct.slice(t) - Bt * G_tilde * Ct.slice(t).t();
//             Ht = arma::symmatu(Ht);
//             Ht.diag() += 1.e-5;

//             arma::mat Ht_chol;
//             try {
//                 Ht_chol = arma::chol(Ht);
//                 Theta_tilde.col(t) = ht + Ht_chol.t() * arma::randn(ht.n_elem,1);
//             } catch(...) {
//                 arma::vec eigval;
//                 arma::mat eigvec;
//                 arma::eig_sym(eigval,eigvec,Ht);
//                 // Ht = U V U' = UV^{1/2} V^{1/2}U'
//                 // 
//                 arma::uvec idx = arma::find(eigval>0.);
//                 arma::vec eigval2 = eigval.elem(idx);
//                 arma::mat eigvec2 = eigvec.cols(idx);
//                 Ht_chol = arma::diagmat(arma::sqrt(eigval2)) * eigvec2.t();
//                 Theta_tilde.col(t) = ht + Ht_chol.t() * arma::randn(idx.n_elem,1);
//                 // Rcout << "\n nn=" << nn << ", t=" << t << std::endl;

//                 // arma::vec eigval;
//                 // arma::mat eigvec;
//                 // arma::eig_sym(eigval,eigvec,Ht);

//                 // Rcout << "Eigenvalues of Ht(t): " << eigval.t() << std::endl;
//                 // ::Rf_error("Cholesky of Ht(t) in smoothing failed");

//             }
//         }


//         for (unsigned int t=(2*L); t>L; t--) {
//             unsigned int idx = t - L - 1;
//             arma::mat Ct_tmp = Ct.slice(t).submat(0,0,idx,idx);
//             // Ct: (L+ns) x (L+ns)
//             // For the L X L part
            
//             arma::mat G_tmp = G_tilde.submat(0,0,idx,idx);

//             arma::mat R_tmp = Rt.slice(t+1).submat(0,0,idx,idx);
//             arma::mat Rt_chol;
//             try {
//                 Rt_chol = arma::chol(arma::symmatu(R_tmp));
//             } catch(...) {
//                 Rcout << "\n nn=" << nn << ", t=" << t << std::endl;

//                 arma::vec eigval;
//                 arma::mat eigvec;
//                 arma::eig_sym(eigval,eigvec,R_tmp);

//                 Rcout << "Eigenvalues of Rt(t+1): " << eigval.t() << std::endl;
//                 ::Rf_error("Cholesky of Rt(t+1) in smoothing failed");

//             }

//             arma::mat Rt_ichol = arma::inv(arma::trimatu(Rt_chol));
//             arma::mat Rt_inv = Rt_ichol * Rt_ichol.t();

//             Bt.zeros();
//             Bt.submat(0,0,idx,idx) = Ct_tmp * G_tmp.t() * Rt_inv;

//             ht = mt.col(t) + Bt * (Theta_tilde.col(t+1) - at.col(t+1));
//             Ht = Ct.slice(t) - Bt * G_tilde * Ct.slice(t).t();
//             Ht = arma::symmatu(Ht);
//             Ht.diag() += 1.e-5;

//             arma::mat Ht_chol;
//             try {
//                 Ht_chol = arma::chol(Ht);
//             } catch(...) {
//                 Rcout << "\n nn=" << nn << ", t=" << t << std::endl;

//                 arma::vec eigval;
//                 arma::mat eigvec;
//                 arma::eig_sym(eigval,eigvec,Ht);

//                 Rcout << "Eigenvalues of Ht(t): " << eigval.t() << std::endl;
//                 ::Rf_error("Cholesky of Ht(t) in smoothing failed");

//             }

//             Theta_tilde.col(t) = ht + Ht_chol.t() * arma::randn(ht.n_elem,1);
//         }



//         for (unsigned int t=L; t>0; t--) {
//             Theta_tilde.col(t).zeros();
//         }


//         // (3) Sample evolution variance W
//         // [Checked - OK]
//         if (eta_type.at(0) == 1) {
//             bw = eta_init.at(2,0); // rate
//             for (unsigned int ti=L; ti<(L+nt); ti++) {
//                 bw += 0.5 * (Theta_tilde.at(0,ti)-Theta_tilde.at(0,ti-1)) * (Theta_tilde.at(0,ti)-Theta_tilde.at(0,ti-1));
//             }

//             W = 1./R::rgamma(aw,1./bw);
//             W_tilde.at(0,0) = W;
//         }


//         arma::mat Vtmp = Theta_tilde.submat(L,L,L+ns-1,L+nt-1); // ns x nt

//         // (6) Sample delta_a
//         if (eta_type.at(3) == 1) {
//             logit_delta_new = R::rnorm(logit_delta,sd_mh);
//             delta_new = std::exp(logit_delta_new) / (1. + std::exp(logit_delta_new));
//             logp_new = logit_delta_integrated_loglik(delta_new,Vtmp,V,Tvi,Is,ns,nt,a_delta,b_delta,a_sig2,b_sig2);

//             logratio = std::min(0.,logp_new - logp);
//             if (std::log(R::runif(0.,1.)) < logratio) {
//                 // accept
//                 mh_accept.at(nn) = 1.;
                    
//                 delta = delta_new;
//                 logit_delta = logit_delta_new;
//                 logp = logp_new;
//             }
//         }

//         arma::mat Q = Tv*(Is-delta*Tvi*V); // ns x ns
//         // (5) Sample sig2_a
//         if (eta_type.at(2) == 1) {
            
//             b_sig2_tilde = 0;
//             for (unsigned int t=0; t<nt; t++) {
//                 arma::vec vtmp = Vtmp.col(t); // ns x 1
//                 b_sig2_tilde += arma::as_scalar(vtmp.t()*Q*vtmp);
//             }
//             b_sig2_tilde *= 0.5;
//             b_sig2_tilde += b_sig2;
//             sigma2 = 1./R::rgamma(a_sig2_tilde,1./b_sig2_tilde); // [TODO] check this part   
//         }

//         W_tilde.submat(L,L,L+ns-1,L+ns-1) = sigma2*Q.i();
        
		
// 		// store samples after burnin and thinning
// 		if (saveiter || nn==(ntotal-1)) {
// 			unsigned int idx_run;
// 			if (saveiter) {
// 				idx_run = (nn-nburnin-1)/nthin;
// 			} else {
// 				idx_run = nsample - 1;
// 			}

//             Lambda_stored.slice(idx_run) = Lambda;
//             Theta_stored.slice(idx_run) = Theta_tilde;

//             mt_stored.col(idx_run) = mt.row(0).t();
//             Ct_stored.col(idx_run) = arma::vectorise(Ct.tube(0,0));

//             W_stored.at(idx_run) = W;
//             mu_stored.col(idx_run) = mu;
//             sigma2_stored.at(idx_run) = sigma2;
//             delta_stored.at(idx_run) = delta;
// 		}

// 		if (verbose) {
// 			Rcout << "\rProgress: " << nn << "/" << ntotal-1;
// 		}
		
// 	}

// 	if (verbose) {
// 		Rcout << std::endl;
// 	}

// 	Rcpp::List output;
//     output["Theta"] = Rcpp::wrap(Theta_stored);
//     output["Lambda"] = Rcpp::wrap(Lambda_stored);
//     output["mt"] = Rcpp::wrap(mt_stored);
//     output["Ct"] = Rcpp::wrap(Ct_stored);
    
//     if (eta_type.at(0) == 1) {
//         // (3) Sample evolution variance W
//         output["W"] = Rcpp::wrap(W_stored);
//     }

//     if (eta_type.at(1)==1) {
//         // sample b and/or sig2_b,delta_b
//         output["mu"] = Rcpp::wrap(mu_stored);
//     }

//     if (eta_type.at(2)==1) {
//         output["sigma2"] = Rcpp::wrap(sigma2_stored);
//     }

//     if (eta_type.at(3)==1) {
//         output["delta"] = Rcpp::wrap(delta_stored);
//         output["mh_accept"] = Rcpp::wrap(mh_accept);
//     }

// 	return output;
// }





//' @export
// [[Rcpp::export]]
Rcpp::List lbe_ffbs_normal(
	const arma::mat& Y, // S x T, the observed response
    const arma::mat& Vt, // S x S, observation variance
    const double W, // double, evolution variance
	const unsigned int L = 12, // temporal lags for transfer effect
    const Rcpp::Nullable<Rcpp::NumericVector>& m0_prior = R_NilValue,
	const Rcpp::Nullable<Rcpp::NumericMatrix>& C0_prior = R_NilValue,
    const Rcpp::Nullable<Rcpp::NumericVector>& Fphi_g = R_NilValue,
    const double discount_factor = 0.9) {

    /*
    -----------------
    Variable - Counting
    -----------------
    */
    const unsigned int ns = Y.n_rows; // S, number of location
	const unsigned int nt = Y.n_cols; // T, number of temporal observations
    const double nt_ = static_cast<double>(nt);

    const arma::vec Js(ns,arma::fill::zeros);
    const arma::vec Jl(L,arma::fill::ones);
    const arma::vec Jt(nt,arma::fill::ones);
    arma::mat Ypad(ns,L+nt,arma::fill::zeros);
    Ypad.cols(L,L+nt-1) = Y;

    const arma::mat Is(ns,ns,arma::fill::eye);
    /* ----------------- */


    /*
    -----------------
    Variables - Forward Filtering Backward Sampling
    -----------------
    */
    arma::vec theta(L,arma::fill::zeros);
    arma::mat Theta(L,L+nt,arma::fill::zeros);
    arma::mat G(L,L,arma::fill::zeros);
    G.at(0,0) = 1.;
    G.diag(-1).ones();


    arma::vec Fphi;
    if (Fphi_g.isNull()) {
        Fphi = get_Fphi(L);
    } else {
        Fphi = Rcpp::as<arma::vec>(Fphi_g);
    }
    arma::vec Fy(L,arma::fill::zeros);
    arma::vec Fts(L,arma::fill::zeros);
    arma::cube Ft(L,ns,L+nt);

    arma::mat Wmat(L,L,arma::fill::zeros);
    Wmat.at(0,0) = W;

    arma::mat at(L,L+nt,arma::fill::zeros);
    arma::cube Rt(L,L,L+nt+1);

    arma::vec ft(ns,arma::fill::zeros);
    arma::mat Qt(ns,ns,arma::fill::eye);

    arma::mat At(L,ns,arma::fill::zeros);
    arma::mat Bt(L,L,arma::fill::zeros);

    arma::mat mt(L,L+nt,arma::fill::zeros);
    arma::cube Ct(L,L,L+nt);
    if (!m0_prior.isNull()) {
        mt.col(L-1) = Rcpp::as<arma::vec>(m0_prior);
	}

	if (!C0_prior.isNull()) {
        Ct.slice(L-1) = Rcpp::as<arma::mat>(C0_prior);
	} else {
        Ct.slice(L-1) = 10. * arma::mat(L,L,arma::fill::eye);
    }

    arma::mat ht(L,L+nt,arma::fill::zeros);
    arma::cube Ht(L,L,L+nt);
    
    arma::mat Ytmp(ns,L,arma::fill::zeros);

    double coef1 = 1. - discount_factor;
    double coef2 = discount_factor * discount_factor;
    /* ----------------- */

    unsigned int ir, ic;
    
    /*
    -----------------
    Temporal parameters for MCMC
    -----------------
    */




        // Forward filtering
        for (unsigned int t=L; t<(L+nt); t++) {
            Ytmp = arma::reverse(Ypad.cols(t-L,t-1),1);
            Ytmp.each_row([&Fphi](arma::rowvec& y){y = y % Fphi.t();return;});
            Ft.slice(t) = Ytmp.t(); // ns x L


            at.col(t) = G * mt.col(t-1); // L x 1
            Rt.slice(t) = G * Ct.slice(t-1) * G.t() + Wmat; // L x L
                
            ft = Ft.slice(t).t() * at.col(t); // ns x 1 = (ns x L) * (L x 1)
            Qt = Ft.slice(t).t() * Rt.slice(t) * Ft.slice(t) + Vt; // ns x ns
            Qt = arma::symmatu(Qt);

            arma::mat Qt_chol = arma::chol(Qt); 
            arma::mat Qt_ichol = arma::inv(arma::trimatu(Qt_chol));
            arma::mat Qt_inv = Qt_ichol * Qt_ichol.t();
            At = Rt.slice(t) * Ft.slice(t) * Qt_inv;


            mt.col(t) = at.col(t) + At * (Ypad.col(t) - ft);
            Ct.slice(t) = Rt.slice(t) - At * Ft.slice(t).t() * Rt.slice(t);
        }

        // Rcout << "mt = " << mt << std::endl;

        // // Backward sampling
        ht.col(nt+L-1) = mt.col(nt+L-1);
        Ht.slice(nt+L-1) = Ct.slice(nt+L-1);
        arma::mat Ht_chol = arma::chol(arma::symmatu(Ht.slice(nt+L-1)));


        for (unsigned int t=(L+nt-2); t>(2*L); t--) {
            arma::mat Rt_chol = arma::chol(arma::symmatu(Rt.slice(t+1)));
            arma::mat Rt_ichol = arma::inv(arma::trimatu(Rt_chol));
            arma::mat Rt_inv = Rt_ichol * Rt_ichol.t();

            Bt = Ct.slice(t) * G.t() * Rt_inv;

            ht.col(t) = mt.col(t) + Bt * (ht.col(t+1) - at.col(t+1));
            Ht.slice(t) = Ct.slice(t) + Bt * (Ht.slice(t+1) - Rt.slice(t+1)) * Bt.t();
        }

        for (unsigned int t=(2*L); t>L; t--) {
            unsigned int idx = t - L - 1;
            arma::mat Ct_tmp = Ct.slice(t).submat(0,0,idx,idx);
            
            arma::mat G_tmp = G.submat(0,0,idx,idx);;

            arma::mat R_tmp = Rt.slice(t+1).submat(0,0,idx,idx);
            arma::mat Rt_chol = arma::chol(arma::symmatu(R_tmp));
            arma::mat Rt_ichol = arma::inv(arma::trimatu(Rt_chol));
            arma::mat Rt_inv = Rt_ichol * Rt_ichol.t();

            Bt.zeros();
            Bt.submat(0,0,idx,idx) = Ct_tmp * G_tmp.t() * Rt_inv;

            ht.col(t) = mt.col(t) + Bt * (ht.col(t+1) - at.col(t+1));
            Ht.slice(t) = Ct.slice(t) + Bt * (Ht.slice(t+1) - Rt.slice(t+1)) * Bt.t();
        }


        for (unsigned int t=L; t>0; t--) {
            ht.col(t).zeros();
            Ht.slice(t).zeros();
        }




	Rcpp::List output;
    output["mt"] = Rcpp::wrap(mt); // L x (nt+1)
    output["Ct"] = Rcpp::wrap(Ct);
    output["ht"] = Rcpp::wrap(ht);
    output["Ht"] = Rcpp::wrap(Ht);
    output["Ft"] = Rcpp::wrap(Ft);

	return output;
}





//' @export
// [[Rcpp::export]]
Rcpp::List mcmc_ffbs_normal(
	const arma::mat& Y, // S x T, the observed response
    const arma::mat& Vt, // S x S, observation variance
    const double W, // double, evolution variance
	const unsigned int L = 12, // temporal lags for transfer effect
    const unsigned int nburnin = 0,
	const unsigned int nthin = 1,
	const unsigned int nsample = 1,
    const Rcpp::Nullable<Rcpp::NumericVector>& m0_prior = R_NilValue,
	const Rcpp::Nullable<Rcpp::NumericMatrix>& C0_prior = R_NilValue,
    const Rcpp::Nullable<Rcpp::NumericVector>& Fphi_g = R_NilValue,
    const double discount_factor = 0.9,
    const bool verbose = true) {

    /*
    -----------------
    Variable - Counting
    -----------------
    */
    const unsigned int ntotal = nburnin + nthin*nsample + 1; // number of iterations for MCMC

    const unsigned int ns = Y.n_rows; // S, number of location
	const unsigned int nt = Y.n_cols; // T, number of temporal observations
    const double nt_ = static_cast<double>(nt);

    const arma::vec Js(ns,arma::fill::zeros);
    const arma::vec Jl(L,arma::fill::ones);
    const arma::vec Jt(nt,arma::fill::ones);
    arma::mat Ypad(ns,L+nt,arma::fill::zeros);
    Ypad.cols(L,L+nt-1) = Y;

    const arma::mat Is(ns,ns,arma::fill::eye);
    const arma::mat Il(L,L,arma::fill::eye);
    /* ----------------- */


    /*
    -----------------
    Variables - Forward Filtering Backward Sampling
    -----------------
    */
    arma::vec theta(L,arma::fill::zeros);
    arma::mat Theta(L,L+nt,arma::fill::zeros);
    arma::mat G(L,L,arma::fill::zeros);
    G.at(0,0) = 1.;
    G.diag(-1).ones();


    arma::vec Fphi;
    if (Fphi_g.isNull()) {
        Fphi = get_Fphi(L);
    } else {
        Fphi = Rcpp::as<arma::vec>(Fphi_g);
    }
    arma::vec Fy(L,arma::fill::zeros);
    arma::vec Fts(L,arma::fill::zeros);
    arma::cube Ft(L,ns,L+nt);

    arma::mat Wmat(L,L,arma::fill::zeros);
    Wmat.at(0,0) = W;

    arma::mat at(L,L+nt,arma::fill::zeros);
    arma::cube Rt(L,L,L+nt+1);

    arma::vec ft(ns,arma::fill::zeros);
    arma::mat Qt(ns,ns,arma::fill::eye);

    arma::mat At(L,ns,arma::fill::zeros);
    arma::mat Bt(L,L,arma::fill::zeros);

    arma::mat mt(L,L+nt,arma::fill::zeros);
    arma::cube Ct(L,L,L+nt);
    if (!m0_prior.isNull()) {
        mt.col(L-1) = Rcpp::as<arma::vec>(m0_prior);
	}

	if (!C0_prior.isNull()) {
        Ct.slice(L-1) = Rcpp::as<arma::mat>(C0_prior);
	} else {
        Ct.slice(L-1) = 10. * arma::mat(L,L,arma::fill::eye);
    }

    arma::vec ht(L,arma::fill::zeros);
    arma::mat Ht(L,L);
    
    arma::mat Ytmp(ns,L,arma::fill::zeros);

    double coef1 = 1. - discount_factor;
    double coef2 = discount_factor * discount_factor;
    /* ----------------- */

    unsigned int ir, ic;
    bool saveiter;
    arma::cube Theta_stored(L,L+nt,nsample);
    
    /*
    -----------------
    Temporal parameters for MCMC
    -----------------
    */

    for (unsigned int nn=0; nn<ntotal; nn++){
        R_CheckUserInterrupt();
		saveiter = nn > nburnin && ((nn-nburnin-1)%nthin==0);

        // Forward filtering
        for (unsigned int t=L; t<(L+nt); t++) {
            Ytmp = arma::reverse(Ypad.cols(t-L,t-1),1);
            Ytmp.each_row([&Fphi](arma::rowvec& y){y = y % Fphi.t();return;});
            Ft.slice(t) = Ytmp.t(); // ns x L

            at.col(t) = G * mt.col(t-1); // L x 1
            Rt.slice(t) = G * Ct.slice(t-1) * G.t() + Wmat; // L x L
                
            ft = Ft.slice(t).t() * at.col(t); // ns x 1 = (ns x L) * (L x 1)
            Qt = Ft.slice(t).t() * Rt.slice(t) * Ft.slice(t) + Vt; // ns x ns
            Qt = arma::symmatu(Qt);

            arma::mat Qt_chol;
            try {
                Qt_chol = arma::chol(Qt); 
            } catch(...) {
                Rcout << "\n nn=" << nn << ", t=" << t << std::endl;

                arma::vec eigval;
                arma::mat eigvec;
                arma::eig_sym(eigval,eigvec,Qt);

                Rcout << "Eigenvalues of Qt: " << eigval.t() << std::endl;
                ::Rf_error("Cholesky of Qt in filtering failed");

            }
            
            arma::mat Qt_ichol = arma::inv(arma::trimatu(Qt_chol));
            arma::mat Qt_inv = Qt_ichol * Qt_ichol.t();
            At = Rt.slice(t) * Ft.slice(t) * Qt_inv;

            mt.col(t) = at.col(t) + At * (Ypad.col(t) - ft);
            Ct.slice(t) = Rt.slice(t) - At * Ft.slice(t).t() * Rt.slice(t);
        }

        // Rcout << "mt = " << mt << std::endl;

        // // Backward sampling
        ht = mt.col(nt+L-1);
        Ht = Ct.slice(nt+L-1);
        {
            arma::mat Ht_chol;
            try {
                Ht_chol = arma::chol(arma::symmatu(Ht));
            } catch(...) {
                Rcout << "\n nn=" << nn << ", t=" << nt+L-1 << std::endl;

                arma::vec eigval;
                arma::mat eigvec;
                arma::eig_sym(eigval,eigvec,Ht);

                Rcout << "Eigenvalues of Ht(nt+L-1): " << eigval.t() << std::endl;
                ::Rf_error("Cholesky of Ht(nt+L-1) in smoothing failed");

            }

            Theta.col(nt+L-1) = ht + Ht_chol.t() * arma::randn(L,1);
        }
        
        // Theta: L x (nt+1)
        for (unsigned int t=(L+nt-2); t>(2*L); t--) {
            arma::mat Rt_chol;
            try {
                Rt_chol = arma::chol(arma::symmatu(Rt.slice(t+1)));
            } catch(...) {
                Rcout << "\n nn=" << nn << ", t=" << t << std::endl;

                arma::vec eigval;
                arma::mat eigvec;
                arma::eig_sym(eigval,eigvec,Rt.slice(t+1));

                Rcout << "Eigenvalues of Rt(t+1): " << eigval.t() << std::endl;
                ::Rf_error("Cholesky of Rt(t+1) in smoothing failed");

            }

            arma::mat Rt_ichol = arma::inv(arma::trimatu(Rt_chol));
            arma::mat Rt_inv = Rt_ichol * Rt_ichol.t();

            Bt = Ct.slice(t) * G.t() * Rt_inv;


            // if (nn<nburnin) {
            //     ht.col(t) = mt.col(t) + Bt * (ht.col(t+1) - at.col(t+1));
            //     Ht.slice(t) = Ct.slice(t) + Bt * (Ht.slice(t+1) - Rt.slice(t+1)) * Bt.t();
            // } else {
                ht = mt.col(t) + Bt * (Theta.col(t+1) - at.col(t+1));
                Ht = Ct.slice(t) - Bt * G * Ct.slice(t).t();
            // }

            Ht = arma::symmatu(Ht + 1.e-5*Il);

            arma::mat Ht_chol;
            try {
                Ht_chol = arma::chol(Ht);
            } catch(...) {
                Rcout << "\n nn=" << nn << ", t=" << t << std::endl;

                arma::vec eigval;
                arma::mat eigvec;
                arma::eig_sym(eigval,eigvec,Ht);

                Rcout << "Eigenvalues of Ht(t): " << eigval.t() << std::endl;
                Rcout << "Ht(t): " << Ht << std::endl;
                ::Rf_error("Cholesky of Ht(t) in smoothing failed");

            }

            Theta.col(t) = ht + Ht_chol.t() * arma::randn(L,1);
        }

        for (unsigned int t=(2*L); t>L; t--) {
            unsigned int idx = t - L - 1;
            arma::mat Ct_tmp = Ct.slice(t).submat(0,0,idx,idx);
            
            arma::mat G_tmp = G.submat(0,0,idx,idx);

            arma::mat R_tmp = Rt.slice(t+1).submat(0,0,idx,idx);
            arma::mat Rt_chol;
            try {
                Rt_chol = arma::chol(arma::symmatu(R_tmp));
            } catch(...) {
                Rcout << "\n nn=" << nn << ", t=" << t << std::endl;

                arma::vec eigval;
                arma::mat eigvec;
                arma::eig_sym(eigval,eigvec,R_tmp);

                Rcout << "Eigenvalues of Rt(t+1): " << eigval.t() << std::endl;
                ::Rf_error("Cholesky of Rt(t+1) in smoothing failed");

            }

            arma::mat Rt_ichol = arma::inv(arma::trimatu(Rt_chol));
            arma::mat Rt_inv = Rt_ichol * Rt_ichol.t();

            Bt.zeros();
            Bt.submat(0,0,idx,idx) = Ct_tmp * G_tmp.t() * Rt_inv;

            // if (nn<nburnin) {
            //     ht.col(t) = mt.col(t) + Bt * (ht.col(t+1) - at.col(t+1));
            //     Ht.slice(t) = Ct.slice(t) + Bt * (Ht.slice(t+1) - Rt.slice(t+1)) * Bt.t();                
            // } else {
                ht = mt.col(t) + Bt * (Theta.col(t+1) - at.col(t+1));
                Ht = Ct.slice(t) - Bt * G * Ct.slice(t).t();
            // }

            Ht = arma::symmatu(Ht + 1.e-5*Il);


            arma::mat Ht_chol;
            try {
                Ht_chol = arma::chol(Ht);
            } catch(...) {
                Rcout << "\n nn=" << nn << ", t=" << t << std::endl;

                arma::vec eigval;
                arma::mat eigvec;
                arma::eig_sym(eigval,eigvec,Ht);

                Rcout << "Eigenvalues of Ht(t): " << eigval.t() << std::endl;
                Rcout << "Ht(t): " << Ht << std::endl;
                ::Rf_error("Cholesky of Ht(t) in smoothing failed");

            }

            Theta.col(t) = ht + Ht_chol.t() * arma::randn(L,1);
        }


        for (unsigned int t=L; t>0; t--) {
            Theta.col(t).zeros();
        }


        // store samples after burnin and thinning
		if (saveiter || nn==(ntotal-1)) {
			unsigned int idx_run;
			if (saveiter) {
				idx_run = (nn-nburnin-1)/nthin;
			} else {
				idx_run = nsample - 1;
			}

            Theta_stored.slice(idx_run) = Theta;
		}

		if (verbose) {
			Rcout << "\rProgress: " << nn << "/" << ntotal-1;
		}
    }

    if (verbose) {
		Rcout << std::endl;
	}



	Rcpp::List output;
    output["mt"] = Rcpp::wrap(mt); // L x (nt+1)
    output["Ct"] = Rcpp::wrap(Ct);
    output["Theta"] = Rcpp::wrap(Theta_stored);

	return output;
}





//' @export
// [[Rcpp::export]]
Rcpp::List mcmc_polya_gamma_simple(
	const arma::mat& Y, // S x T, the observed response
    const arma::mat& F, // L x S
    const arma::mat& G, // L X L
    const arma::mat& W, // L x L
    const arma::vec& npop, // S x 1, number of population for each location
	const unsigned int nburnin = 0,
	const unsigned int nthin = 1,
	const unsigned int nsample = 1,
    const Rcpp::Nullable<Rcpp::NumericVector>& a_init = R_NilValue,
    const Rcpp::Nullable<Rcpp::NumericVector>& mu_a_in = R_NilValue, // specify values of a if assuming known.
    const Rcpp::Nullable<Rcpp::NumericMatrix>& Sig_a_in = R_NilValue, // specify values of b if assuming known.
    const Rcpp::Nullable<Rcpp::NumericVector>& m0_prior = R_NilValue,
	const Rcpp::Nullable<Rcpp::NumericMatrix>& C0_prior = R_NilValue,
    const Rcpp::NumericVector sig_mh = Rcpp::NumericVector::create(1.,1.),
    const double discount_factor = 0.9,
    const double npop_scale = 1.,
	const bool summarize_return = false,
	const bool verbose = true) { // n x 1

    
    /*
    -----------------
    Variable - Counting
    -----------------
    */
    const unsigned int ns = Y.n_rows; // S, number of location
	const unsigned int nt = Y.n_cols; // T, number of temporal observations
    const double nt_ = static_cast<double>(nt);
    const unsigned int np = G.n_rows;// number of latent variables

	const unsigned int ntotal = nburnin + nthin*nsample + 1; // number of iterations for MCMC

    const arma::vec Js(ns,arma::fill::zeros);
    const arma::vec Jp(np,arma::fill::ones);
    const arma::vec Jt(nt+1,arma::fill::ones);
    const arma::mat Ip(np,np,arma::fill::eye);

    arma::mat Zpad(ns,1+nt,arma::fill::zeros); // ns x (nt+L)
    arma::mat Ypad(ns,1+nt,arma::fill::zeros);
    Zpad.cols(1,nt) = Y;
    Ypad.cols(1,nt) = Y; 
    /* ----------------- */


    /*
    -----------------
    Variables - Forward Filtering Backward Sampling
    -----------------
    */
    arma::vec theta(np,arma::fill::zeros);
    arma::mat Theta(np,1+nt,arma::fill::zeros);
    arma::mat Ot_inv(ns,ns,arma::fill::zeros);

    arma::mat at(np,nt+1,arma::fill::zeros);
    arma::cube Rt(np,np,nt+1);
    arma::vec ft(ns,arma::fill::zeros);
    arma::mat Qt(ns,ns,arma::fill::eye);

    arma::mat At(np,ns,arma::fill::zeros);
    arma::mat Bt(np,np,arma::fill::zeros);

    arma::mat mt(np,1+nt,arma::fill::zeros);
    arma::cube Ct(np,np,1+nt);
    if (!m0_prior.isNull()) {
		mt.col(0) = Rcpp::as<arma::vec>(m0_prior);
	}

	if (!C0_prior.isNull()) {
		Ct.slice(0) = Rcpp::as<arma::mat>(C0_prior);
	} else {
        Ct.slice(0) = 10. * arma::mat(np,np,arma::fill::eye);
    }

    Theta.col(0) = mt.col(0) + arma::chol(Ct.slice(0)).t()*arma::randn(np,1);

    arma::vec ht(np,arma::fill::zeros);
    arma::mat Ht(np,np,arma::fill::ones);


    double coef1 = 1. - discount_factor;
    double coef2 = discount_factor * discount_factor;
    /* ----------------- */




    /*
    -----------------
    Temporal parameters for MCMC
    -----------------
    */
    arma::vec mu_a(ns,arma::fill::zeros);
    if (!mu_a_in.isNull()) {
        mu_a = Rcpp::as<arma::vec>(mu_a_in);
    }
    arma::mat Sig_a(ns,ns,arma::fill::eye);
    if (!Sig_a_in.isNull()) {
        Sig_a = Rcpp::as<arma::mat>(Sig_a_in);
    } else {
        Sig_a = 10.* arma::mat(ns,ns,arma::fill::eye);
    }
    
    arma::vec a(ns,arma::fill::zeros);
    if (!a_init.isNull()) {
        a = Rcpp::as<arma::vec>(a_init);
    } else {
        a = mu_a = arma::chol(Sig_a).t()*arma::randn(ns,1);
    }


    arma::vec mu_b(ns,arma::fill::zeros);
    arma::mat Sig_b(ns,ns,arma::fill::eye);
    arma::vec b = mu_b;
    

    arma::mat Omat(ns,ns,arma::fill::eye);
    arma::vec kvec(ns,arma::fill::zeros);
    arma::vec dvec(ns,arma::fill::zeros);

    arma::cube Theta_stored(np,nt+1,nsample);
    arma::mat mt_stored(1+nt,nsample);
    arma::mat Ct_stored(1+nt,nsample);
    arma::mat a_stored(ns,nsample);
    arma::mat a_param(2,nsample); // (a_sig2a,b_sig2a)
    arma::mat b_stored(ns,nsample);
    arma::mat b_param(2,nsample); // (a_sig2b,b_sig2b)

    double blogitdel_new, bdel_new, blogp_new;
    double bdel_old;
    double blogitdel_old = std::log(bdel_old) - std::log(1.-bdel_old);
    double blogp_old;

    double alogitdel_new, adel_new, alogp_new;
    double adel_old;
    double alogitdel_old = std::log(adel_old) - std::log(1.-adel_old);
    double alogp_old;

    double logratio;
    arma::mat mh_accept(ntotal,2,arma::fill::zeros); // (col 1 = a, col 2 = b)
    bool saveiter;
    unsigned int ir, ic;
    
    /*
    -----------------
    Temporal parameters for MCMC
    -----------------
    */

    arma::mat Omega(ns,1+nt,arma::fill::ones);
    arma::mat K(ns,1+nt,arma::fill::zeros);
    arma::mat Psi(ns,1+nt,arma::fill::zeros);
    arma::mat Lambda(ns,1+nt,arma::fill::zeros); // ns x (nt+1)
    arma::mat Z(ns,1+nt,arma::fill::zeros);

    arma::cube Lambda_stored(ns,1+nt,nsample);


	for (unsigned int nn=0; nn<ntotal; nn++) {
		R_CheckUserInterrupt();
		saveiter = nn > nburnin && ((nn-nburnin-1)%nthin==0);

        /*
        (1) Sample auxliary Polya-Gamma variables: Omega, Lambda, Zmat.
        */
        for (unsigned int s=0; s<ns; s++) {
            for (unsigned int t=1; t<(1+nt); t++) {
                // [TODO] order?
                K.at(s,t) = 0.5 * (Ypad.at(s,t) - npop.at(s));

                Lambda.at(s,t) = a.at(s) + Psi.at(s,t);
                Omega.at(s,t) = pg::rpg_scalar_hybrid(npop.at(s) + Ypad.at(s,t),Lambda.at(s,t));

                Z.at(s,t) = K.at(s,t)/Omega.at(s,t) - a.at(s);
                
                Psi.at(s,t) = R::rnorm(Z.at(s,t), 1./std::sqrt(Omega.at(s,t)));
            }
        }


        /*
        (2) Sample augmented latent states Theta using forward filtering and backward sampling.
        */

        // Forward filtering
        // [Checked - OK]
        for (unsigned int t=1; t<(1+nt); t++) {
            Ot_inv = arma::diagmat(1./Omega.col(t));

            at.col(t) = G * mt.col(t-1); // L x 1
            Rt.slice(t) = G * Ct.slice(t-1) * G.t() + W;
                
            ft = F.t() * at.col(t); // ns x 1 = (ns x L) * (L x 1)
            Qt = F.t() * Rt.slice(t) * F + Ot_inv; // ns x ns
            Qt = arma::symmatu(Qt);

            arma::mat Qt_chol = arma::chol(Qt);
            arma::mat Qt_ichol = arma::inv(arma::trimatu(Qt_chol)); // R^{-1}: inverse of upper-triangular cholesky component
            arma::mat Qt_inv = Qt_ichol * Qt_ichol.t(); // inverse Qt^{-1} = R^{-1} (R^{-1})'

            At = Rt.slice(t) * F * Qt_inv;
            mt.col(t) = at.col(t) + At * (Z.col(t) - ft); // Z[s,t] or Psi[s,t]
            Ct.slice(t) = Rt.slice(t) - At * F.t() * Rt.slice(t);
        }



        // Backward sampling
        // [Checked - OK]
        ht = mt.col(nt);
        Ht = Ct.slice(nt);
        {
            arma::mat Ht_chol = arma::chol(arma::symmatu(Ht));
            Theta.col(nt) = ht + Ht_chol.t() * arma::randn(np,1);
        }
        
        // Theta: L x (nt+1)
        // Theta.at(0,nt) = R::rnorm(ht.at(nt),std::sqrt(Ht.at(nt)));
        for (unsigned int t=(nt-1); t>0; t--) {
            arma::mat Rt_chol = arma::chol(arma::symmatu(Rt.slice(t+1)));
            arma::mat Rt_ichol = arma::inv(arma::trimatu(Rt_chol));
            arma::mat Rt_inv = Rt_ichol * Rt_ichol.t();

            Bt = Ct.slice(t) * G.t() * Rt_inv;

            ht = mt.col(t) + Bt * (Theta.col(t+1) - at.col(t+1));
            Ht = Ct.slice(t) - Bt * G * Ct.slice(t).t();
            Ht = arma::symmatu(Ht + 1.e-5*Ip);

            arma::mat Ht_chol = arma::chol(Ht);
            Theta.col(t) = ht + Ht_chol.t() * arma::randn(np,1);

        }




        // if (eta_type.at(4)==1 || eta_type.at(1)==1) {
            // for (unsigned int t=1; t<=nt; t++) {
            //     Kt.slice(t) = arma::diagmat(arma::vectorise(Jl.as_row()*Ft.slice(t))); // ns x ns
            //     rt.col(t) = Ft.slice(t).t()*Theta.col(t);
            // }
            // // (7) Sample spatial covariates b=(b[1],...,b[ns])
            // if (eta_type.at(4) == 1) {
            //     Ztmp2 = Z - arma::kron(a,Jt.t()) - rt; // ns x (nt+1)
            //     Sig_b = Tv * (Is - delta_b*Tvi*V) / sig2_b;
            //     mu_b.zeros();
            //     for (unsigned int t=1; t<=nt; t++) {
            //         Sig_b = Sig_b + Kt.slice(t).t()*arma::diagmat(Omega.col(t))*Kt.slice(t);
            //         mu_b = mu_b + Kt.slice(t).t() * arma::diagmat(Omega.col(t)) * Ztmp2.col(t);
            //     }

            //     Sig_b = arma::inv_sympd(arma::symmatu(Sig_b)); // [TODO] check this part
            //     mu_b = Sig_b * mu_b;
            //     Sig_b = arma::chol(Sig_b);
            //     b = mu_b + Sig_b.t()*arma::randn(ns,1);

            //     // (8) Sample sig2_b
            //     if (eta_type.at(5) == 1) {
            //         b_sig2b = 0.5 * arma::as_scalar((b.t()-mu_b.t())*Tv*(Is-delta_b*Tvi*V)*(b-mu_b));
            //         sig2_b = 1./R::rgamma(a_sig2b,1./b_sig2b); // [TODO] check this part
            //     }


            //     // (9) Sample delta_b
            //     if (eta_type.at(6) == 1) {
            //         blogitdel_new = std::exp(R::rnorm(blogitdel_old,sig_mh[1]));
            //         bdel_new = blogitdel_new / (1.+blogitdel_new);
            //         blogp_new = logit_delta_integrated_loglik(bdel_new,b,V,Tvi,mu_b,Is,ns);

            //         logratio = std::min(0.,blogp_new - blogp_old);
            //         if (std::log(R::runif(0.,1.)) < logratio) {
            //             // accept
            //             mh_accept.at(nn,1) = 1.;

            //             delta_b = bdel_new;
            //             blogitdel_old = std::log(bdel_new) - std::log(1.-bdel_new);
            //             blogp_old = blogp_new;
            //         }
            //     }
            // }



            // (4) Sample spatial baseline a=(a[1],...,a[ns])
            // if (eta_type.at(1) == 1) {
            //     Omat = arma::diagmat(arma::sum(Omega,1));
            //     kvec = arma::vectorise(arma::sum(K,1));
            //     dvec = arma::vectorise(arma::sum(Psi % Omega,1));

            //     Sig_a = Tv*(Is - delta_a*Tvi*V)/sig2_a + Omat; // Prior 
            //     Sig_a = arma::inv_sympd(arma::symmatu(Sig_a)); 
            //     mu_a = Sig_a * (kvec - dvec);

            //     Sig_a = arma::chol(Sig_a);
            //     a = mu_a + Sig_a.t()*arma::randn(ns,1);

            //     // (6) Sample delta_a
            //     if (eta_type.at(3) == 1) {
            //         alogitdel_new = std::exp(R::rnorm(alogitdel_old,sig_mh[0]));
            //         adel_new = alogitdel_new / (1.+alogitdel_new);
            //         alogp_new = logit_delta_integrated_loglik(adel_new,a,V,Tvi,mu_a,Is,ns);

            //         logratio = std::min(0.,alogp_new - alogp_old);
            //         if (std::log(R::runif(0.,1.)) < logratio) {
            //             // accept
            //             mh_accept.at(nn,0) = 1.;
                    
            //             delta_a = adel_new;
            //             alogitdel_old = std::log(adel_new) - std::log(1.-adel_new);
            //             alogp_old = alogp_new;
            //         }
            //     }


            //     // (5) Sample sig2_a
            //     if (eta_type.at(2) == 1) {
            //         b_sig2a = 0.5 * arma::as_scalar((a.t()-mu_a.t())*Tv*(Is-delta_a*Tvi*V)*(a-mu_a));
            //         sig2_a = 1./R::rgamma(a_sig2a,1./b_sig2a); // [TODO] check this part
            //     }
            // }
        // }        
        
		
		// store samples after burnin and thinning
		if (saveiter || nn==(ntotal-1)) {
			unsigned int idx_run;
			if (saveiter) {
				idx_run = (nn-nburnin-1)/nthin;
			} else {
				idx_run = nsample - 1;
			}

            Lambda_stored.slice(idx_run) = Lambda;
            Theta_stored.slice(idx_run) = Theta;

            mt_stored.col(idx_run) = mt.row(0).t();
            Ct_stored.col(idx_run) = arma::vectorise(Ct.tube(0,0));
		}

		if (verbose) {
			Rcout << "\rProgress: " << nn << "/" << ntotal-1;
		}
		
	}

	if (verbose) {
		Rcout << std::endl;
	}

	Rcpp::List output;
    output["Theta"] = Rcpp::wrap(Theta_stored);
    output["Lambda"] = Rcpp::wrap(Lambda_stored);
    output["mt"] = Rcpp::wrap(mt_stored);
    output["Ct"] = Rcpp::wrap(Ct_stored);

	return output;
}
