#include "pl_poisson.h"
using namespace Rcpp;
// [[Rcpp::plugins(cpp17)]]
// [[Rcpp::depends(RcppArmadillo,nloptr)]]



/*
---- Method ----
- Monte Carlo Smoothing with static parameter W known
- B-lag fixed-lag smoother (Anderson and Moore 1979)
- Using the DLM formulation
- Reference: 
    Kitagawa and Sato at Ch 9.3.4 of Doucet et al.
    Check out Algorithm step 2-4L for the explanation of B-lag fixed-lag smoother
- Note: 
    1. Initial version is copied from the `bf_pois_koyama_exp`
    2. This is intended to be the Rcpp version of `hawkes_state_space.R`
    3. The difference is that the R version the states are backshifted L times, where L is the maximum transmission delay to be considered.
        To make this a Monte Carlo smoothing, we need to resample the states n times, where n is the total number of temporal observations.

---- Algorithm ----

1. Generate a random number psi[0](j) ~ an initial distribution.
2. Repeat the following steps for t = 1,...,n:
    2-1. Generate a random number omega[t] ~ Normal(0,W)


---- Model ----
<obs>   y[t] ~ Pois(lambda[t])
<link>  lambda[t] = phi[1]*y[t-1]*exp(psi[t]) + ... + phi[L]*y[t-L]*exp(psi[t-L+1])
<state> psi[t] = psi[t-1] + omega[t]
        omega[t] ~ Normal(0,W)

Unknown parameters: psi[1:n]
Known parameters: W, phi[1:L]
Kwg: Identity link, exp(psi) state space
*/
//' @export
// [[Rcpp::export]]
Rcpp::List mcs_poisson(
    const arma::vec& Y, // n x 1, the observed response
    const arma::uvec& model_code,
	const double W = NA_REAL, // Use discount factor if W is not given
    const double rho = 0.9,
    const double alpha = 1.,
    const unsigned int L = 12, // number of lags
    const double mu0 = 2.220446e-16,
    const unsigned int B = 12, // length of the B-lag fixed-lag smoother (Anderson and Moore 1979; Kitagawa and Sato)
	const unsigned int N = 5000, // number of particles
    const Rcpp::Nullable<Rcpp::NumericVector>& m0_prior = R_NilValue,
	const Rcpp::Nullable<Rcpp::NumericMatrix>& C0_prior = R_NilValue,
    const Rcpp::NumericVector& qProb = Rcpp::NumericVector::create(0.025,0.5,0.975),
    const Rcpp::NumericVector& ctanh = Rcpp::NumericVector::create(0.2,0,5.),
    const double delta_nb = 1.,
    const double delta_discount = 0.95,
    const bool resample = false,
    const bool verbose = false,
    const bool debug = false){ 
    
    unsigned int tmpi; // store temporary integer value
    const unsigned int n = Y.n_elem; // number of observations
    double Wsqrt;
    if (!R_IsNA(W)) {
        Wsqrt = std::sqrt(W);
    }
    const double min_eff = 0.8*static_cast<double>(N);
    arma::vec ypad(n+1,arma::fill::zeros); ypad.tail(n) = Y;

    if (debug) {
        Rcout << "Evolution variance W=" << W << std::endl;
    }


    /* 
    Dimension of state space depends on type of transfer functions 
    - p: diemsnion of DLM state space
    - Ft: vector for the state-to-observation function
    - Gt: matrix for the state-to-state function
    */
    const unsigned int obs_code = model_code.at(0);
    const unsigned int link_code = model_code.at(1);
    const unsigned int trans_code = model_code.at(2);
    const unsigned int gain_code = model_code.at(3);
    const unsigned int err_code = model_code.at(4);
	unsigned int p, L_;
	arma::vec Ft, Fphi;
    arma::mat Gt;
    init_by_trans(p,L_,Ft,Fphi,Gt,trans_code,L);

    Fphi = arma::pow(Fphi,alpha);
    arma::vec Fy(p,arma::fill::zeros);
    /* Dimension of state space depends on type of transfer functions */




    // theta: DLM state vector
    // p - dimension of state space
    // N - number of particles
    arma::vec lambda(N); // instantaneous intensity
    arma::vec omega(N); // evolution variance
    arma::vec w(N); // importance weight of each particle
    Rcpp::IntegerVector idx_(N);
    arma::uvec idx(N);

    /*
    ------ Step 1. Initialization at time t = -1 ------
        - Sample theta[-1] from the prior of theta[-1]
    
    ------
    The first B-2 rows (indices from 0 to B-2) of R are all zeros, as illustrated as follows:

    At the beginning of t = 0
        - propagate to theta[0]
        - theta_stored.slice(B-2) = theta[-1], theta_stored.slice(B-1) = theta[0]
        - resample theta_stored
        - save theta_stored.slice(0)==ZERO to R.row(0)

    At the beginning of t = 1
        - propagate to theta[1]
        - theta_stored.slice(B-3) = theta[-1], ..., theta_stored.slice(B-1) = theta[1]
        - resample theta_stored
        - save theta_stored.slice(0)==ZERO to R.row(1)

    ...

    At the beginning of t = B-2
        - propagate to theta[B-2]
        - theta_stored.slice(0) = theta[-1], ..., theta_stored.slice(B-1) = theta[B-2]
        - resample theta_stored
        - save theta_stored.slice(0)==theta[-1] to R.row(B-2); *** theta[-1] has been resampled B-1 times

    At the beginning of t = B-1
        - propagate to theta[B-1]
        - theta_stored.slice(0) = theta[0], ..., theta_stored.slice(B-1) = theta[B-1]
        - resample theta_stored
        - save theta_stored.slice(0)==theta[0] to R.row(B-1); *** theta[0] has been resampled B times

    At the beginning of t = n-1, theta_stored.slice(B-1) = theta[n-2]; resample; save theta_stored.slice(0)=theta[n-B-1] to R.row(n-1); propagate to theta[n-1]
        - propagate to theta[n-1]
        - theta_stored.slice(0) = theta[n-B], ..., theta_stored.slice(B-1) = theta[n-1]
        - resample theta_stored
        - save theta_stored.slice(0)==theta[n-B] to R.row(n-1); *** theta[n-B] has been resampled B times
            - theta[n-B+1] has been resample B-1 times
            .....
            - theta[n-2] is resampled 2 times
            - theta[n-1] is resample once
    ------
    >>>>>> Outside of the for loop,
        - The first B-2 rows (indices from 0 to B-1) of R are all zeros
        - Shift the the last (B-2):(n-1) rows (n-B+2 in totals) of R to 0:n-B+1
        - For the last B-1 rows (indices from n-B+2 to n), takes values from theta_stored.slice(1,B-1)


    In order to save the theta resampled B-1 times to R, we just need to save the first slice of theta_stored.
    */
        
    arma::mat theta(p,N);
    arma::cube theta_stored(p,N,B);
    arma::mat hpsi;
    if (link_code==1) {
        // Exponential Link
        arma::vec m0(p,arma::fill::zeros);
        arma::mat C0(p,p,arma::fill::eye); C0 *= 3.;
        if (!m0_prior.isNull()) {
		    m0 = Rcpp::as<arma::vec>(m0_prior);
	    }
	    if (!C0_prior.isNull()) {
		    C0 = Rcpp::as<arma::mat>(C0_prior);
            C0 = arma::chol(C0);
	    }

        for (unsigned int i=0; i<N; i++) {
            theta.col(i) = m0 + C0.t() * arma::randn(p);
        }

    } else {
        // Identity Link
        theta = arma::randu(p,N,arma::distr_param(0.,10.)); // Consider it as a flat prior
    }
    theta_stored.slice(B-1) = theta;
    /*
    ------ Step 1. Initialization theta[0,] at time t = 0 ------
    */
    const double c1 = std::pow(1.-rho,static_cast<double>(L_)*alpha);
    double c2 = -rho;

    // const double c1_ = std::pow((1.-rho)*(1.-rho),alpha);
    // const double c2_ = 2.*rho;
    // const double c3_ = -rho*rho;

    
    arma::mat R(n+1,3); // quantiles
    arma::vec Meff(n,arma::fill::zeros); // Effective sample size (Ref: Lin, 1996; Prado, 2021, page 196)
    arma::uvec resample_status(n,arma::fill::zeros);
    arma::vec mt(p);
    arma::mat Ct(p,p);
    arma::vec Wt(n);

    for (unsigned int t=0; t<n; t++) {
        R_CheckUserInterrupt();
        // theta_stored: p x N x B, with B is the lag of the B-lag fixed-lag smoother
        // theta: p x N

        if (trans_code == 1) { // Koyama
            Fy.zeros();
            tmpi = std::min(t,p);
            if (t>0) {
                // Checked Fy - CORRECT
                Fy.head(tmpi) = arma::reverse(ypad.subvec(t+1-tmpi,t));
            }

            Ft = Fphi % Fy; // L(p) x 1
        }

        /*
        ------ Step 1.0 Resample ------
        Auxiliary Particle Filter
        */
        if (resample) {
            mt = arma::median(theta_stored.slice(B-1),1);
            update_Gt(Gt,gain_code, trans_code, mt, ctanh, alpha, ypad.at(t), rho);
            theta = update_at(p,gain_code,trans_code,theta_stored.slice(B-1),Gt,ctanh,alpha,ypad.at(t),rho);
            if (link_code==1) {
                // Exponential link and identity gain
                for (unsigned int i=0; i<N; i++) {
                    lambda.at(i) = mu0 + arma::as_scalar(Ft.t()*theta.col(i));
                }
                lambda.elem(arma::find(lambda>UPBND)).fill(UPBND);
                lambda = arma::exp(lambda);
            } else if (trans_code==1){ // Koyama
                hpsi = psi2hpsi(theta,gain_code,ctanh); // hpsi: p x N
                for (unsigned int i=0; i<N; i++) {
                    lambda.at(i) = mu0 + arma::as_scalar(Ft.t()*hpsi.col(i));
                }
            } else {
                // Koyck or Solow with identity link and different gain functions
                lambda = mu0 + theta.row(1).t();
            }

            for (unsigned int i=0; i<N; i++) {
                w.at(i) = loglike_obs(ypad.at(t+1),lambda.at(i),obs_code,delta_nb,false);
            } // End for loop of i, index of the particles

            if (arma::accu(w)>EPS) { // normalize the particle weights
                w /= arma::accu(w);
                if (1./arma::dot(w,w) > min_eff) {
                    idx_ = Rcpp::sample(N,N,true,Rcpp::as<Rcpp::NumericVector>(Rcpp::wrap(w)));
                    idx = Rcpp::as<arma::uvec>(idx_) - 1;
                    for (unsigned int b=0; b<B; b++) {
                        theta_stored.slice(b) = theta_stored.slice(b).cols(idx);
                    }
                }
            }
        }
        

        /*
        ------ Step 2.1 Propagate ------
        Propagate theta[t,i] to theta[t+1,i] for i=0,1,...,N
        using state evolution distribution P(theta[t+1,i]|theta[t,i]).
        ------
        >> Step 2-1 and 2-2 of Kitagawa and Sato - This is also the state equation
        */
        
        if (t>B) {
            // Is it necessary?
            mt = 0.5*arma::mean(theta_stored.slice(B-1),1) + 0.5*arma::median(theta_stored.slice(B-1),1);
        } else {
            mt = arma::median(theta_stored.slice(B-1),1);
        }
        
        // for (unsigned int i=0; i<N; i++) {
        //     update_Gt(Gt,gain_code, trans_code, 0.5*mt + 0.5*theta_stored.slice(B-1).col(i), ctanh, alpha, ypad.at(t), rho);
        //     theta.col(i) = update_at(p,gain_code,trans_code,theta_stored.slice(B-1).col(i),Gt,ctanh,alpha,ypad.at(t),rho);
        // }
        update_Gt(Gt,gain_code, trans_code, mt, ctanh, alpha, ypad.at(t), rho);
        theta = update_at(p,gain_code,trans_code,theta_stored.slice(B-1),Gt,ctanh,alpha,ypad.at(t),rho);
  
        if (R_IsNA(W)) { // Use discount factor if W is not given
            Wt.at(t) = arma::stddev(theta_stored.slice(B-1).row(0));
            Wsqrt = Wt.at(t);
            if (t>B) {
                Wsqrt *= std::sqrt(1./delta_discount-1.);
            } else {
                Wsqrt *= std::sqrt(1./0.99-1.);
            }
        }
        omega = arma::randn(N) * Wsqrt;
        theta.row(0) += omega.t();

        if (debug) {
            Rcout << "quantiles for hpsi[" << t+1 << "]" << arma::quantile(theta.row(1),Rcpp::as<arma::vec>(qProb));
        }
        
        theta_stored.slices(0,B-2) = theta_stored.slices(1,B-1);
        theta_stored.slice(B-1) = theta;
        /*
        ------ Step 2.1 Propagate ------
        */
        
        

        /*
        ------ Step 2.2 Importance weights ------
        Calculate the importance weight of lambda[t-L:t,i] for i=0,1,...,N, where i is the index of particles
        using likelihood P(y[t]|lambda[t,i]).
        ------
        >> Step 2-3 and 2-4 of Kitagawa and Sato - This is also the observational equation (nonlinear, nongaussian)
        */
        if (link_code==1) {
            // Exponential link and identity gain
            for (unsigned int i=0; i<N; i++) {
                lambda.at(i) = mu0 + arma::as_scalar(Ft.t()*theta.col(i));
            }
            lambda.elem(arma::find(lambda>UPBND)).fill(UPBND);
            lambda = arma::exp(lambda);
        } else if (trans_code==1){ // Koyama
            hpsi = psi2hpsi(theta,gain_code,ctanh); // hpsi: p x N
            for (unsigned int i=0; i<N; i++) {
                lambda.at(i) = mu0 + arma::as_scalar(Ft.t()*hpsi.col(i));
            }
        } else {
            // Koyck or Solow with identity link and different gain functions
            lambda = mu0 + theta.row(1).t();
        }

        if (debug) {
            Rcout << "Quantiles of lambda[" << t+1 << "]: " << arma::quantile(lambda.t(),Rcpp::as<arma::vec>(qProb));
        }

        for (unsigned int i=0; i<N; i++) {
            w.at(i) = loglike_obs(ypad.at(t+1),lambda.at(i),obs_code,delta_nb,false);
        } // End for loop of i, index of the particles


        if (debug) {
            Rcout << "Quantiles of importance weight w[" << t+1 << "]: " << arma::quantile(w.t(),Rcpp::as<arma::vec>(qProb));
        }
        /*
        ------ Step 2.2 Importance weights ------
        */

        /*
        ------ Step 3 Resampling with Replacement ------
        */
        if (arma::accu(w)>EPS) { // normalize the particle weights
            w /= arma::accu(w);
            Meff.at(t) = 1./arma::dot(w,w);
            try{
                idx_ = Rcpp::sample(N,N,true,Rcpp::as<Rcpp::NumericVector>(Rcpp::wrap(w)));
                idx = Rcpp::as<arma::uvec>(idx_) - 1;
                for (unsigned int b=0; b<B; b++) {
                    theta_stored.slice(b) = theta_stored.slice(b).cols(idx);
                }
                resample_status.at(t) = 1;
            } catch(...) {
                // If resampling doesn't work, then just don't resample
                resample_status.at(t) = 0;
            }
        } else {
            resample_status.at(t) = 0;
            Meff.at(t) = 0.;
        }

        if (debug && resample_status.at(t) == 0) {
            Rcout << "Resampling skipped at time t=" << t << std::endl;
            // ::Rf_error("Probabilities must be finite and non-negative!");
        }
        
        
        /*
        ------ Step  Resampling with Replacement ------
        */
        R.row(t) = arma::quantile(theta_stored.slice(0).row(0),Rcpp::as<arma::vec>(qProb));
    }

    /*
    ------ R: an n x 3 matrix ------
        - The first B-2 rows (indices from 0 to B-1) of R are all zeros
        - Shift the the last (B-2):(n-1) rows (n-B+2 in totals) of R to 0:n-B+1
        - For the last B-1 rows (indices from n-B+2 to n), takes values from theta_stored.slice(1,B-1)
    */
    R.rows(0,n-B+1) = R.rows(B-2,n-1);
    for (unsigned int b=0; b<(B-1); b++) {
        R.row(n-B+2+b) = arma::quantile(theta_stored.slice(b+1).row(0),Rcpp::as<arma::vec>(qProb));
    }

    Rcpp::List output;
    output["psi"] = Rcpp::wrap(R); // (n+1) x 3
    output["theta_last"] = Rcpp::wrap(theta_stored.slice(B-1)); // p x N
    output["Wt"] = Rcpp::wrap(Wt);

    if (debug) {
        output["Meff"] = Rcpp::wrap(Meff);
        output["resample_status"] = Rcpp::wrap(resample_status);
    }
    

    return output;
}



/*
---- Method ----
- Two-Filter Smoothing with static parameter W known
- Using the DLM formulation

---- Algorithm ----

---- Model ----
<obs>   y[t] ~ Pois(lambda[t])
<link>  lambda[t] = phi[1]*y[t-1]*exp(psi[t]) + ... + phi[L]*y[t-L]*exp(psi[t-L+1])
<state> psi[t] = psi[t-1] + omega[t]
        omega[t] ~ Normal(0,W)

Unknown parameters: psi[1:n]
Known parameters: W, phi[1:L]
Kwg: Identity link, exp(psi) state space
*/
//' @export
// [[Rcpp::export]]
Rcpp::List tfs_poisson(
    const arma::vec& Y, // n x 1, the observed response
    const arma::uvec& model_code,
	const double W,
    const double rho = 0.9,
    const double alpha = 1.,
    const unsigned int L = 12, // number of lags
    const double mu0 = 2.220446e-16,
    const unsigned int B = 12, // length of the B-lag fixed-lag smoother (Anderson and Moore 1979; Kitagawa and Sato)
	const unsigned int N = 5000, // number of particles
    const Rcpp::Nullable<Rcpp::NumericVector>& m0_prior = R_NilValue,
	const Rcpp::Nullable<Rcpp::NumericMatrix>& C0_prior = R_NilValue,
    const Rcpp::NumericVector& qProb = Rcpp::NumericVector::create(0.025,0.5,0.975),
    const Rcpp::NumericVector& ctanh = Rcpp::NumericVector::create(0.2,0,5.),
    const double delta_nb = 1.,
    const double delta_discount = 0.95,
    const bool verbose = true,
    const bool debug = false){ 
    
    unsigned int tmpi; // store temporary integer value
    const unsigned int n = Y.n_elem; // number of observations
    const double Wsqrt = std::sqrt(W);
    const double min_eff = 0.8*static_cast<double>(N);
    arma::vec ypad(n+1,arma::fill::zeros); ypad.tail(n) = Y;

    if (debug) {
        Rcout << "Evolution variance W=" << W << std::endl;
    }


    /* 
    Dimension of state space depends on type of transfer functions 
    - p: diemsnion of DLM state space
    - Ft: vector for the state-to-observation function
    - Gt: matrix for the state-to-state function
    */
    const unsigned int obs_code = model_code.at(0);
    const unsigned int link_code = model_code.at(1);
    const unsigned int trans_code = model_code.at(2);
    const unsigned int gain_code = model_code.at(3);
    const unsigned int err_code = model_code.at(4);
	unsigned int p, L_;
	arma::vec Ft, Fphi;
    arma::mat Gt;
    init_by_trans(p,L_,Ft,Fphi,Gt,trans_code,L);

    Fphi = arma::pow(Fphi,alpha);
    arma::vec Fy(p,arma::fill::zeros);
    /* Dimension of state space depends on type of transfer functions */




    // theta: DLM state vector
    // p - dimension of state space
    // N - number of particles
    arma::vec lambda(N); // instantaneous intensity
    arma::vec omega(N); // evolution variance
    arma::vec w(N); // importance weight of each particle
    Rcpp::IntegerVector idx_(N);
    arma::uvec idx(N);
    arma::mat w_stored(N,n+1,arma::fill::ones);
    w_stored /= static_cast<double>(N);

        
    arma::mat theta(p,N);
    arma::cube theta_pred(p,N,n+1);
    arma::cube theta_filter(p,N,n+1);
    arma::cube theta_smooth(p,N,n+1);
    arma::mat hpsi;
    if (link_code==1) {
        // Exponential Link
        arma::vec m0(p,arma::fill::zeros);
        arma::mat C0(p,p,arma::fill::eye); C0 *= 3.;
        if (!m0_prior.isNull()) {
		    m0 = Rcpp::as<arma::vec>(m0_prior);
	    }
	    if (!C0_prior.isNull()) {
		    C0 = Rcpp::as<arma::mat>(C0_prior);
            C0 = arma::chol(C0);
	    }

        for (unsigned int i=0; i<N; i++) {
            theta.col(i) = m0 + C0.t() * arma::randn(p);
        }

    } else {
        // Identity Link
        theta = arma::randu(p,N,arma::distr_param(0.,10.)); // Consider it as a flat prior
    }
    theta_pred.slice(0) = theta;
    theta_filter.slice(0) = theta;
    /*
    ------ Step 1. Initialization theta[0,] at time t = 0 ------
    */
    const double c1 = std::pow(1.-rho,static_cast<double>(L_)*alpha);
    double c2 = -rho;

    // const double c1_ = std::pow((1.-rho)*(1.-rho),alpha);
    // const double c2_ = 2.*rho;
    // const double c3_ = -rho*rho;
    
    arma::vec Meff(n,arma::fill::zeros); // Effective sample size (Ref: Lin, 1996; Prado, 2021, page 196)
    arma::vec mt(p);
    arma::vec Wt(n+1,arma::fill::zeros);

    for (unsigned int t=0; t<n; t++) {
        R_CheckUserInterrupt();
        // theta_stored: p x N x B, with B is the lag of the B-lag fixed-lag smoother
        // theta: p x N

        if (trans_code == 1) { // Koyama
            Fy.zeros();
            tmpi = std::min(t,p);
            if (t>0) {
                // Checked Fy - CORRECT
                Fy.head(tmpi) = arma::reverse(ypad.subvec(t+1-tmpi,t));
            }

            Ft = Fphi % Fy; // L(p) x 1
        }
        

        /*
        ------ Step 2.1 Propagate ------
        Propagate theta[t,i] to theta[t+1,i] for i=0,1,...,N
        using state evolution distribution P(theta[t+1,i]|theta[t,i]).
        ------
        >> Step 2-1 and 2-2 of Kitagawa and Sato - This is also the state equation
        */
        
        mt = arma::median(theta_filter.slice(t),1);
        update_Gt(Gt,gain_code, trans_code, mt, ctanh, alpha, ypad.at(t), rho);
        theta = update_at(p,gain_code,trans_code,theta_filter.slice(t),Gt,ctanh,alpha,ypad.at(t),rho);
  
        Wt.at(t+1) = arma::stddev(theta_filter.slice(t).row(0));
        omega = arma::randn(N) * Wt.at(t+1);
        if (t>B) {
            omega *= std::sqrt(1./delta_discount-1.);
        } else {
            omega *= std::sqrt(1./0.99-1.);
        }
        
        theta.row(0) += omega.t();

        if (debug) {
            Rcout << "quantiles for hpsi[" << t+1 << "]" << arma::quantile(theta.row(1),Rcpp::as<arma::vec>(qProb));
        }
        
        theta_pred.slice(t+1) = theta;
        /*
        ------ Step 2.1 Propagate ------
        */
        
        

        /*
        ------ Step 2.2 Importance weights ------
        Calculate the importance weight of lambda[t-L:t,i] for i=0,1,...,N, where i is the index of particles
        using likelihood P(y[t]|lambda[t,i]).
        ------
        >> Step 2-3 and 2-4 of Kitagawa and Sato - This is also the observational equation (nonlinear, nongaussian)
        */
        if (link_code==1) {
            // Exponential link and identity gain
            for (unsigned int i=0; i<N; i++) {
                lambda.at(i) = mu0 + arma::as_scalar(Ft.t()*theta.col(i));
            }
            lambda.elem(arma::find(lambda>UPBND)).fill(UPBND);
            lambda = arma::exp(lambda);
        } else if (trans_code==1){ // Koyama
            hpsi = psi2hpsi(theta,gain_code,ctanh); // hpsi: p x N
            for (unsigned int i=0; i<N; i++) {
                lambda.at(i) = mu0 + arma::as_scalar(Ft.t()*hpsi.col(i));
            }
        } else {
            // Koyck or Solow with identity link and different gain functions
            lambda = mu0 + theta.row(1).t();
        }

        if (debug) {
            Rcout << "Quantiles of lambda[" << t+1 << "]: " << arma::quantile(lambda.t(),Rcpp::as<arma::vec>(qProb));
        }

        for (unsigned int i=0; i<N; i++) {
            w.at(i) = loglike_obs(ypad.at(t+1),lambda.at(i),obs_code,delta_nb,false);
        } // End for loop of i, index of the particles


        if (debug) {
            Rcout << "Quantiles of importance weight w[" << t+1 << "]: " << arma::quantile(w.t(),Rcpp::as<arma::vec>(qProb));
        }
        /*
        ------ Step 2.2 Importance weights ------
        */

        /*
        ------ Step 3 Resampling with Replacement ------
        */
        if (arma::accu(w)>EPS) { // normalize the particle weights
            w /= arma::accu(w);
            Meff.at(t) = 1./arma::dot(w,w);
            idx_ = Rcpp::sample(N,N,true,Rcpp::as<Rcpp::NumericVector>(Rcpp::wrap(w)));
            idx = Rcpp::as<arma::uvec>(idx_) - 1;
            theta_filter.slice(t+1) = theta.cols(idx);
            w_stored.col(t+1) = w.elem(idx);
        } else {
            Meff.at(t) = 0.;
        }
        /*
        ------ Step  Resampling with Replacement ------
        */

    }


    /*
    ------ Kitagawa's Two Filter Smoother ------
    */
    theta_smooth = theta_filter;
    arma::mat delta(N,n+1,arma::fill::ones);
    arma::mat wb_stored(N,n+1,arma::fill::ones);
    delta /= static_cast<double> (N);
    wb_stored /= static_cast<double>(N);
    // wb_stored.col(n) = w_stored.col(n);
    for (unsigned int t=n; t>0; t--) {
        if (t<n) {
            // Evaluate Pr(y[t+1:n] | theta[t,i]) via Monte Carlo Integration.
            for (unsigned int i=0; i<N; i++) {
                delta.at(i,t) = arma::mean(arma::exp(-0.5*Wt.at(t+1)*arma::pow(theta_smooth.slice(t+1).row(0)-theta_pred.slice(t).at(0,i),2.))%wb_stored.col(t+1).t());
            }
        }

        // Update Pr(y[t:n] | theta[t,i])
        wb_stored.col(t) = delta.col(t) % w_stored.col(t);

        // Sample from Pr(theta[t] | y[t:n]) by resampling {theta[t] | y[1:(t-1)]} with weights Pr(y[t:n | theta[t,i]])
        if (arma::accu(wb_stored.col(t)>EPS) && t<n) {
            wb_stored.col(t) /= arma::accu(wb_stored.col(t));
            idx_ = Rcpp::sample(N,N,true,Rcpp::as<Rcpp::NumericVector>(Rcpp::wrap(wb_stored.col(t))));
            idx = Rcpp::as<arma::uvec>(idx_) - 1;
            theta_smooth.slice(t) = theta_pred.slice(t).cols(idx);
        }

        if (verbose) {
			Rcout << "\rProgress: " << (n-t+1) << "/" << n+1;
		}
    }

    // For t = 0
    for (unsigned int i=0; i<N; i++) {
        // Evaluate Pr(y[t+1:n] | theta[t,i]) via Monte Carlo Integration.
        delta.at(i,0) = arma::mean(arma::exp(-0.5*Wt.at(1)*arma::pow(theta_smooth.slice(1).row(0)-theta_pred.slice(0).at(0,i),2.))%wb_stored.col(1).t());
    }
    wb_stored.col(0) = delta.col(0) % w_stored.col(0);
    if (arma::accu(wb_stored.col(0)>EPS)) {
        wb_stored.col(0) /= arma::accu(wb_stored.col(0));
        idx_ = Rcpp::sample(N,N,true,Rcpp::as<Rcpp::NumericVector>(Rcpp::wrap(wb_stored.col(0))));
        idx = Rcpp::as<arma::uvec>(idx_) - 1;
        theta_smooth.slice(0) = theta_pred.slice(0).cols(idx);
    }

    if (verbose) {
		Rcout << "\rProgress: " << n+1 << "/" << n+1;
        Rcout << std::endl;
	}


    Rcpp::List output;
    arma::mat psi(N,n+1);
    arma::mat R(n+1,3);
    psi = theta_pred.row(0);
    R = arma::quantile(psi,Rcpp::as<arma::vec>(qProb),0).t();
    output["psi_pred"] = Rcpp::wrap(R); // (n+1) x 3

    psi = theta_smooth.row(0);
    R = arma::quantile(psi,Rcpp::as<arma::vec>(qProb),0).t();
    output["psi_smooth"] = Rcpp::wrap(R); // (n+1) x 3

    psi = theta_filter.row(0);
    R = arma::quantile(psi,Rcpp::as<arma::vec>(qProb),0).t();
    output["psi_filter"] = Rcpp::wrap(R); // (n+1) x 3

    output["Wt"] = Rcpp::wrap(Wt);
    if (debug) {
        output["Meff"] = Rcpp::wrap(Meff);
    }
    

    return output;
}



/*
Bootstrap filter or Auxiliary Particle Filter
with fixed lag smoothing.
*/
void bf_poisson(
    arma::cube& theta_stored, // p x N x (n+1)
    arma::vec& w_stored, // (n+1) x 1
    arma::mat& Gt,
    const arma::vec& ypad, // (n+1) x 1, the observed response
    const arma::uvec& model_code,
	const double W,
    const double mu0,
    const double rho,
    const double alpha,
    const unsigned int p, // dimension of DLM state space
    const unsigned int L, // number of lags
	const unsigned int N, // number of particles
    const arma::vec& Ft0,
    const arma::vec& Fphi,
    const Rcpp::NumericVector& ctanh,
    const double delta_nb){ 
    
    unsigned int tmpi; // store temporary integer value
    const unsigned int n = ypad.n_elem - 1; // number of observations
    const double N_ = static_cast<double>(N);
    const double Wsqrt = std::sqrt(W);
    const double min_eff = 0.8*static_cast<double>(N);
    const unsigned int B = 20;

    /* 
    Dimension of state space depends on type of transfer functions 
    - p: diemsnion of DLM state space
    - Ft: vector for the state-to-observation function
    - Gt: matrix for the state-to-state function
    */
    // const unsigned int obs_code = model_code.at(0);
    // const unsigned int link_code = model_code.at(1);
    // const unsigned int trans_code = model_code.at(2);
    // const unsigned int gain_code = model_code.at(3);
    // const unsigned int err_code = model_code.at(4);
    arma::vec Ft = Ft0;
    arma::vec Fy(p,arma::fill::zeros);
    /* Dimension of state space depends on type of transfer functions */




    // theta: DLM state vector
    // p - dimension of state space
    // N - number of particles
    arma::vec w(N); // importance weight of each particle
    Rcpp::IntegerVector idx_(N);
    arma::uvec idx(N);

    arma::vec lambda(N); // instantaneous intensity
    arma::vec omega(N); // evolution variance
    arma::mat theta(p,N);
    arma::mat hpsi;
    arma::vec mt(p);
    

    // const double c1_ = std::pow((1.-rho)*(1.-rho),alpha);
    // const double c2_ = 2.*rho;
    // const double c3_ = -rho*rho;
    bool resample = false;
    
    double Meff = 0.; // Effective sample size (Ref: Lin, 1996; Prado, 2021, page 196)

    for (unsigned int t=0; t<n; t++) {
        R_CheckUserInterrupt();
        // theta_stored: p x N x B, with B is the lag of the B-lag fixed-lag smoother
        // theta: p x N

        if (model_code.at(2) == 1) {
            Fy.zeros();
            tmpi = std::min(t,p);
            if (t>0) {
                // Checked Fy - CORRECT
                Fy.head(tmpi) = arma::reverse(ypad.subvec(t+1-tmpi,t));
            }

            Ft = Fphi % Fy; // L(p) x 1
        }

        /*
        ------ Step 1.0 Resample ------
        */
        if (resample) {
            /*Auxiliary Particle Filter*/
            mt = arma::median(theta_stored.slice(t),1);
            update_Gt(Gt,model_code.at(3), model_code.at(2), mt, ctanh, alpha, ypad.at(t), rho);
            theta = update_at(p,model_code.at(3),model_code.at(2),theta_stored.slice(t),Gt,ctanh,alpha,ypad.at(t),rho); // p x N, theta[t+1,]
            if (model_code.at(1)==1) {
                // Exponential link and identity gain
                for (unsigned int i=0; i<N; i++) {
                    lambda.at(i) = mu0 + arma::as_scalar(Ft.t()*theta.col(i));
                }
                lambda.elem(arma::find(lambda>UPBND)).fill(UPBND);
                lambda = arma::exp(lambda);
            } else if (model_code.at(2)==1){ // Koyama
                hpsi = psi2hpsi(theta,model_code.at(3),ctanh); // hpsi: p x N
                for (unsigned int i=0; i<N; i++) {
                    lambda.at(i) = mu0 + arma::as_scalar(Ft.t()*hpsi.col(i));
                }
            } else {
                // Koyck or Solow with identity link and different gain functions
                lambda = mu0 + theta.row(1).t();
            }

            for (unsigned int i=0; i<N; i++) {
                w.at(i) = loglike_obs(ypad.at(t+1),lambda.at(i),model_code.at(0),delta_nb,false);
            } // End for loop of i, index of the particles

            if (arma::accu(w)>EPS) { // normalize the particle weights
                w /= arma::accu(w);
                if (1./arma::dot(w,w) > min_eff) {
                    idx_ = Rcpp::sample(N,N,true,Rcpp::as<Rcpp::NumericVector>(Rcpp::wrap(w)));
                    idx = Rcpp::as<arma::uvec>(idx_) - 1;
                    for (unsigned int b=0; b<=t; b++) {
                        theta_stored.slice(b) = theta_stored.slice(b).cols(idx);
                    }
                }
            }
        }

        /*
        ------ Step 2.1 Propagate ------
        Propagate theta[t,i] to theta[t+1,i] for i=0,1,...,N
        using state evolution distribution P(theta[t+1,i]|theta[t,i]).
        ------
        >> Step 2-1 and 2-2 of Kitagawa and Sato - This is also the state equation
        */
        mt = arma::median(theta_stored.slice(t),1);
        update_Gt(Gt,model_code.at(3), model_code.at(2), mt, ctanh, alpha, ypad.at(t), rho);
        theta = update_at(p,model_code.at(3),model_code.at(2),theta_stored.slice(t),Gt,ctanh,alpha,ypad.at(t),rho); // p x N, theta[t+1,]
        omega = arma::randn(N) * Wsqrt;
        theta.row(0) += omega.t();
        
        // theta_stored.slices(0,B-2) = theta_stored.slices(1,B-1);
        theta_stored.slice(t+1) = theta;
        /*
        ------ Step 2.1 Propagate ------
        */
        
        

        /*
        ------ Step 2.2 Importance weights ------
        Calculate the importance weight of lambda[t-L:t,i] for i=0,1,...,N, where i is the index of particles
        using likelihood P(y[t]|lambda[t,i]).
        ------
        >> Step 2-3 and 2-4 of Kitagawa and Sato - This is also the observational equation (nonlinear, nongaussian)
        */
        if (model_code.at(1)==1) {
            // Exponential link and identity gain
            for (unsigned int i=0; i<N; i++) {
                lambda.at(i) = mu0 + arma::as_scalar(Ft.t()*theta.col(i));
            }
            lambda.elem(arma::find(lambda>UPBND)).fill(UPBND);
            lambda = arma::exp(lambda);
        } else if (model_code.at(2)==1){ // Koyama
            hpsi = psi2hpsi(theta,model_code.at(3),ctanh); // hpsi: p x N
            for (unsigned int i=0; i<N; i++) {
                lambda.at(i) = mu0 + arma::as_scalar(Ft.t()*hpsi.col(i));
            }
        } else {
            // Koyck or Solow with identity link and different gain functions
            lambda = mu0 + theta.row(1).t();
        }

        for (unsigned int i=0; i<N; i++) {
            w.at(i) = loglike_obs(ypad.at(t+1),lambda.at(i),model_code.at(0),delta_nb,false);
        } // End for loop of i, index of the particles

        /*
        ------ Step 2.2 Importance weights ------
        */

        /*
        ------ Step 3 Resampling with Replacement ------
        */
        if (arma::accu(w)>EPS) { 
            w /= arma::accu(w); // normalize the particle weights
            Meff = 1./arma::dot(w,w);
            idx_ = Rcpp::sample(N,N,true,Rcpp::as<Rcpp::NumericVector>(Rcpp::wrap(w)));
            idx = Rcpp::as<arma::uvec>(idx_) - 1;

            for (unsigned int b=std::max((unsigned int)0,t+1-B+1); b<=(t+1); b++) {
                theta_stored.slice(b) = theta_stored.slice(b).cols(idx);
            }
            w = w.elem(idx);
            // theta_stored.slice(t+1) = theta_stored.slice(t+1).cols(idx);

        } else {
            w.fill(1./N_);
        }

        w_stored.at(t+1) = arma::accu(w)/N_;
  
        /*
        ------ Step 3 Resampling with Replacement ------
        */
    }


    return;
}



//' @export
// [[Rcpp::export]]
Rcpp::List ffbs_poisson(
    const arma::vec& Y, // n x 1, the observed response
    const arma::uvec& model_code,
	const double W = NA_REAL,
    const double rho = 0.9,
    const double alpha = 1.,
    const unsigned int L = 12, // number of lags
    const double mu0 = 2.220446e-16,
	const unsigned int N = 5000, // number of particles
    const Rcpp::Nullable<Rcpp::NumericVector>& m0_prior = R_NilValue,
	const Rcpp::Nullable<Rcpp::NumericMatrix>& C0_prior = R_NilValue,
    const Rcpp::NumericVector& qProb = Rcpp::NumericVector::create(0.025,0.5,0.975),
    const Rcpp::NumericVector& ctanh = Rcpp::NumericVector::create(0.2,0,5.),
    const double delta_nb = 1.,
    const double delta_discount = 0.95,
    const bool resample = false, // true = auxiliary particle filtering; false = bootstrap filtering
    const bool smoothing = true, // true = particle smoothing; false = no smoothing
    const bool verbose = false,
    const bool debug = false){ 
    
    unsigned int tmpi; // store temporary integer value
    const unsigned int n = Y.n_elem; // number of observations
    const double N_ = static_cast<double>(N);
    const unsigned int B = std::max((unsigned int)0.1*n,L);
    double Wsqrt;
    if (!R_IsNA(W)) {
        Wsqrt = std::sqrt(W);
    }
    const double min_eff = 0.8*static_cast<double>(N);
    arma::vec ypad(n+1,arma::fill::zeros); ypad.tail(n) = Y;

    if (debug) {
        Rcout << "Evolution variance W=" << W << std::endl;
    }


    /* 
    Dimension of state space depends on type of transfer functions 
    - p: diemsnion of DLM state space
    - Ft: vector for the state-to-observation function
    - Gt: matrix for the state-to-state function
    */
    const unsigned int obs_code = model_code.at(0);
    const unsigned int link_code = model_code.at(1);
    const unsigned int trans_code = model_code.at(2);
    const unsigned int gain_code = model_code.at(3);
    const unsigned int err_code = model_code.at(4);
	unsigned int p, L_;
	arma::vec Ft, Fphi;
    arma::mat Gt;
    init_by_trans(p,L_,Ft,Fphi,Gt,trans_code,L);

    Fphi = arma::pow(Fphi,alpha);
    arma::vec Fy(p,arma::fill::zeros);
    /* Dimension of state space depends on type of transfer functions */




    // theta: DLM state vector
    // p - dimension of state space
    // N - number of particles
    arma::vec lambda(N); // instantaneous intensity
    arma::vec omega(N); // evolution variance
    arma::vec w(N); // importance weight of each particle
    Rcpp::IntegerVector idx_(N);
    arma::uvec idx(N);

        
    arma::mat theta(p,N);
    arma::cube theta_stored(p,N,n+1);
    arma::mat hpsi;
    if (link_code==1) {
        // Exponential Link
        arma::vec m0(p,arma::fill::zeros);
        arma::mat C0(p,p,arma::fill::eye); C0 *= 3.;
        if (!m0_prior.isNull()) {
		    m0 = Rcpp::as<arma::vec>(m0_prior);
	    }
	    if (!C0_prior.isNull()) {
		    C0 = Rcpp::as<arma::mat>(C0_prior);
            C0 = arma::chol(C0);
	    }

        for (unsigned int i=0; i<N; i++) {
            theta.col(i) = m0 + C0.t() * arma::randn(p);
        }

    } else {
        // Identity Link
        theta = arma::randu(p,N,arma::distr_param(0.,10.)); // Consider it as a flat prior
    }
    theta_stored.slice(0) = theta;
    /*
    ------ Step 1. Initialization theta[0,] at time t = 0 ------
    */
    const double c1 = std::pow(1.-rho,static_cast<double>(L_)*alpha);
    double c2 = -rho;

    // const double c1_ = std::pow((1.-rho)*(1.-rho),alpha);
    // const double c2_ = 2.*rho;
    // const double c3_ = -rho*rho;

    
    arma::mat R(n+1,3); // quantiles
    arma::vec mt(p);
    arma::mat Ct(p,p);
    arma::vec Wt(n);

    
    for (unsigned int t=0; t<n; t++) {
        R_CheckUserInterrupt();
        // theta_stored: p x N x B, with B is the lag of the B-lag fixed-lag smoother
        // theta: p x N

        if (model_code.at(2) == 1) {
            Fy.zeros();
            tmpi = std::min(t,p);
            if (t>0) {
                // Checked Fy - CORRECT
                Fy.head(tmpi) = arma::reverse(ypad.subvec(t+1-tmpi,t));
            }

            Ft = Fphi % Fy; // L(p) x 1
        }

        /*
        ------ Step 1.0 Resample ------
        */
        if (resample) {
            /*Auxiliary Particle Filter*/
            mt = arma::median(theta_stored.slice(t),1);
            update_Gt(Gt,model_code.at(3), model_code.at(2), mt, ctanh, alpha, ypad.at(t), rho);
            theta = update_at(p,model_code.at(3),model_code.at(2),theta_stored.slice(t),Gt,ctanh,alpha,ypad.at(t),rho); // p x N, theta[t+1,]
            if (model_code.at(1)==1) {
                // Exponential link and identity gain
                for (unsigned int i=0; i<N; i++) {
                    lambda.at(i) = mu0 + arma::as_scalar(Ft.t()*theta.col(i));
                }
                lambda.elem(arma::find(lambda>UPBND)).fill(UPBND);
                lambda = arma::exp(lambda);
            } else if (model_code.at(2)==1){ // Koyama
                hpsi = psi2hpsi(theta,model_code.at(3),ctanh); // hpsi: p x N
                for (unsigned int i=0; i<N; i++) {
                    lambda.at(i) = mu0 + arma::as_scalar(Ft.t()*hpsi.col(i));
                }
            } else {
                // Koyck or Solow with identity link and different gain functions
                lambda = mu0 + theta.row(1).t();
            }

            for (unsigned int i=0; i<N; i++) {
                w.at(i) = loglike_obs(ypad.at(t+1),lambda.at(i),model_code.at(0),delta_nb,false);
            } // End for loop of i, index of the particles

            if (arma::accu(w)>EPS) { // normalize the particle weights
                w /= arma::accu(w);
                if (1./arma::dot(w,w) > min_eff) {
                    try{
                        idx_ = Rcpp::sample(N,N,true,Rcpp::as<Rcpp::NumericVector>(Rcpp::wrap(w)));
                    } catch(...) {
                        Rcout << "Auxiliary resample: ";
                        w.brief_print();
                        ::Rf_error("sampling error.");
                    }
                    // idx_ = Rcpp::sample(N,N,true,Rcpp::as<Rcpp::NumericVector>(Rcpp::wrap(w)));
                    idx = Rcpp::as<arma::uvec>(idx_) - 1;
                    for (unsigned int b=0; b<=t; b++) {
                        theta_stored.slice(b) = theta_stored.slice(b).cols(idx);
                    }
                }
            }
        }

        /*
        ------ Step 2.1 Propagate ------
        Propagate theta[t,i] to theta[t+1,i] for i=0,1,...,N
        using state evolution distribution P(theta[t+1,i]|theta[t,i]).
        ------
        >> Step 2-1 and 2-2 of Kitagawa and Sato - This is also the state equation
        */
        mt = arma::median(theta_stored.slice(t),1);
        update_Gt(Gt,model_code.at(3), model_code.at(2), mt, ctanh, alpha, ypad.at(t), rho);
        theta = update_at(p,model_code.at(3),model_code.at(2),theta_stored.slice(t),Gt,ctanh,alpha,ypad.at(t),rho); // p x N, theta[t+1,]

        if (R_IsNA(W)) { // Use discount factor if W is not given
            Wt.at(t) = arma::stddev(theta_stored.slice(t).row(0));
            Wsqrt = Wt.at(t);
            if (t>B) {
                Wsqrt *= std::sqrt(1./delta_discount-1.);
            } else {
                Wsqrt *= std::sqrt(1./0.99-1.);
            }
        } else {
            Wt.at(t) = Wsqrt;
        }
        omega = arma::randn(N) * Wsqrt;
        theta.row(0) += omega.t();
        
        // theta_stored.slices(0,B-2) = theta_stored.slices(1,B-1);
        theta_stored.slice(t+1) = theta;
        /*
        ------ Step 2.1 Propagate ------
        */
        
        

        /*
        ------ Step 2.2 Importance weights ------
        Calculate the importance weight of lambda[t-L:t,i] for i=0,1,...,N, where i is the index of particles
        using likelihood P(y[t]|lambda[t,i]).
        ------
        >> Step 2-3 and 2-4 of Kitagawa and Sato - This is also the observational equation (nonlinear, nongaussian)
        */
        if (model_code.at(1)==1) {
            // Exponential link and identity gain
            for (unsigned int i=0; i<N; i++) {
                lambda.at(i) = mu0 + arma::as_scalar(Ft.t()*theta.col(i));
            }
            lambda.elem(arma::find(lambda>UPBND)).fill(UPBND);
            lambda = arma::exp(lambda);
        } else if (model_code.at(2)==1){ // Koyama
            hpsi = psi2hpsi(theta,model_code.at(3),ctanh); // hpsi: p x N
            for (unsigned int i=0; i<N; i++) {
                lambda.at(i) = mu0 + arma::as_scalar(Ft.t()*hpsi.col(i));
            }
        } else {
            // Koyck or Solow with identity link and different gain functions
            lambda = mu0 + theta.row(1).t();
        }

        for (unsigned int i=0; i<N; i++) {
            w.at(i) = loglike_obs(ypad.at(t+1),lambda.at(i),model_code.at(0),delta_nb,false);
        } // End for loop of i, index of the particles

        /*
        ------ Step 2.2 Importance weights ------
        */

        /*
        ------ Step 3 Resampling with Replacement ------
        */
        if (arma::accu(w)>EPS) { 
            w /= arma::accu(w); // normalize the particle weights
            try{
                idx_ = Rcpp::sample(N,N,true,Rcpp::as<Rcpp::NumericVector>(Rcpp::wrap(w)));
            } catch(...) {
                Rcout << "Resample the propagated states: ";
                w.brief_print();
                ::Rf_error("sampling error.");
            }
            
            idx = Rcpp::as<arma::uvec>(idx_) - 1;

            theta_stored.slice(t+1) = theta_stored.slice(t+1).cols(idx); // the original BF
            w = w.elem(idx);
            // theta_stored.slice(t+1) = theta_stored.slice(t+1).cols(idx);

        } else {
            w.fill(1./N_);
        }
        /*
        ------ Step 3 Resampling with Replacement ------
        */
    }


    /*
    ------ Particle Smoothing ------
    */
    Rcpp::NumericVector tmpir(1);
    arma::mat psi_smooth(N,n+1);
    arma::mat psi_filter(N,n+1);
    psi_filter.col(n) = theta_stored.slice(n).row(0).t();
    psi_smooth.col(n) = theta_stored.slice(n).row(0).t();
    for (unsigned int t=n; t>0; t--) {
        psi_filter.col(t-1) = theta_stored.slice(t-1).row(0).t();
        for (unsigned int i=0; i<N; i++) {
            w = -0.5*arma::pow((psi_smooth.at(i,t)-psi_filter.col(t-1))/Wsqrt,2);
            if (w.has_nan() || w.has_inf()) {
                w.brief_print();
                ::Rf_error("step 1");
            }
            w.elem(arma::find(w>UPBND)).fill(UPBND);
            w = arma::exp(w);
            if (w.has_nan() || w.has_inf()) {
                w.brief_print();
                ::Rf_error("step 2");
            }
            if (arma::accu(w)<EPS) {
                w.ones();
            }
            w /= arma::accu(w);
            try{
                tmpir = Rcpp::sample(N,1,true,Rcpp::as<Rcpp::NumericVector>(Rcpp::wrap(w))) - 1;
            } catch(...) {
                Rcout << "Resample for smoothing: ";
                w.brief_print();
                ::Rf_error("sampling error.");
            }
            // tmpir = Rcpp::sample(N,1,true,Rcpp::as<Rcpp::NumericVector>(Rcpp::wrap(w))) - 1;
            psi_smooth.at(i,t-1) = psi_filter.at(tmpir[0],t-1);
        }
    }


    Rcpp::List output;
    R = arma::quantile(psi_filter,Rcpp::as<arma::vec>(qProb),0);
    output["psi_filter"] = Rcpp::wrap(R.t());

    if (smoothing) {
        R = arma::quantile(psi_smooth,Rcpp::as<arma::vec>(qProb),0);
        output["psi"] = Rcpp::wrap(R.t());
    }

    return output;
}



//' @export
// [[Rcpp::export]]
Rcpp::List pmmh_poisson(
    const arma::vec& Y, // n x 1, the observed response
    const arma::uvec& model_code,
	const arma::uvec& eta_select, // 4 x 1, indicator for unknown (=1) or known (=0)
    const arma::vec& eta_init, // 4 x 1, if true/initial values should be provided here
    const arma::uvec& eta_prior_type, // 4 x 1
    const arma::mat& eta_prior_val, // 2 x 4, priors for each element of eta
    const double alpha = 1.,
    const unsigned int L = 2, // number of lags
	const unsigned int N = 1000, // number of particles
    const unsigned int nsample = 100,
    const unsigned int nburnin = 100,
    const unsigned int nthin = 2,
    const Rcpp::Nullable<Rcpp::NumericVector>& m0_prior = R_NilValue,
	const Rcpp::Nullable<Rcpp::NumericMatrix>& C0_prior = R_NilValue,
    const Rcpp::NumericVector& qProb = Rcpp::NumericVector::create(0.025,0.5,0.975),
    const double MH_sd = 1.,
    const double delta_nb = 1.,
    const bool verbose = true,
    const bool debug = false){ 
    
    unsigned int tmpi; // store temporary integer value
    const unsigned int n = Y.n_elem; // number of observations
    const double n_ = static_cast<double>(n);
    const double N_ = static_cast<double>(N);
    const unsigned int ntotal = nburnin + nthin*nsample + 1;
    arma::vec ypad(n+1,arma::fill::zeros); ypad.tail(n) = Y;

    double mu0 = eta_init.at(1);
    double rho = eta_init.at(2);
    const Rcpp::NumericVector ctanh = {1./eta_init.at(3),0.,eta_init.at(3)}; // (1./M, 0, M)

    double W = eta_init.at(0);
    arma::vec W_stored(nsample); W_stored.fill(W);
    arma::vec Wnew_stored(ntotal);
    double a1, a2, a3, aw, bw, Wnew;
    if (eta_select.at(0) == 1) { // W unknown
        switch (eta_prior_type.at(0)) {
            case 0: // Gamma, thus Wtilde = log(W)
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
            case 2: // Inverse-Gamma, thus Wtilde = -log(W)
            {
                aw = eta_prior_val.at(0,0) + 0.5*n_;
                bw = eta_prior_val.at(1,0);
            }
            break;
            default:
            {

            }
        }
    }
    /* 
    Dimension of state space depends on type of transfer functions 
    - p: diemsnion of DLM state space
    - Ft: vector for the state-to-observation function
    - Gt: matrix for the state-to-state function
    */
    const unsigned int obs_code = model_code.at(0);
    const unsigned int link_code = model_code.at(1);
    const unsigned int trans_code = model_code.at(2);
    const unsigned int gain_code = model_code.at(3);
    const unsigned int err_code = model_code.at(4);
	unsigned int p, L_;
	arma::vec Ft, Fphi;
    arma::mat Gt;
    init_by_trans(p,L_,Ft,Fphi,Gt,trans_code,L);
    Fphi = arma::pow(Fphi,alpha);
    arma::vec Fy(p,arma::fill::zeros);
    /* Dimension of state space depends on type of transfer functions */


    // theta: DLM state vector
    // p - dimension of state space
    // N - number of particles
    arma::mat theta0(p,N);
    arma::cube theta_stored(p,N,n+1);
    arma::vec w_stored(n+1,arma::fill::ones);

    arma::mat theta(p,n+1);
    arma::vec logratio_stored(ntotal);
    arma::vec accept(ntotal,arma::fill::zeros);
    double logp_old,logp_new,logq_old,logq_new,logratio;
    arma::mat theta_stored2(n+1,nsample);
    arma::vec w_stored2(nsample,arma::fill::zeros);

    arma::vec m0(p,arma::fill::zeros);
    arma::mat C0(p,p,arma::fill::eye);
    if (link_code==1) {
        // Exponential Link
        C0 *= 3.;
        if (!m0_prior.isNull()) {
		    m0 = Rcpp::as<arma::vec>(m0_prior);
	    }
	    if (!C0_prior.isNull()) {
		    C0 = Rcpp::as<arma::mat>(C0_prior);
            C0 = arma::chol(C0);
	    }
        for (unsigned int i=0; i<N; i++) {
            theta0.col(i) = m0 + C0.t() * arma::randn(p);
        }
    } else {
        // Identity Link
        theta0 = arma::randu(p,N,arma::distr_param(0.,10.)); // Consider it as a flat prior
    }
    theta_stored.slice(0) = theta0;
    
    /* SMC Filtering */
    bf_poisson(
        theta_stored, w_stored, Gt,
        ypad, model_code, W, mu0, rho, alpha, p,  L,  N, 
        Ft, Fphi, ctanh, delta_nb);
    
    for (unsigned int t=0; t<(n+1); t++) {
        theta.col(t) = arma::median(theta_stored.slice(t),1);
    }
    
    logp_old = arma::accu(arma::log(w_stored)); // logarithm of the marginal likelihood
    if (eta_select.at(0) == 1) { // W unknwn
        switch (eta_prior_type.at(0)) {
            case 0: // Gamma
            {
                logq_old = R::dgamma(W,eta_prior_val.at(0,0),1./eta_prior_val.at(1,0),true);
                logp_old += logq_old;
            }
            break;
            case 1: // Half-Cauchy
            {

            }
            break;
            case 2: // Inverse-Gamma
            {
                logq_old = -(eta_prior_val.at(0,0)+1.)*std::log(W)-eta_prior_val.at(1,0)/W;
                logp_old += logq_old;
            }
            break;
            default:
            {

            }
        }
    } // TODO: check it
    /*
    ------ Initialization theta[0,] at time t = 0 ------
    */

    bool saveiter;
    double mh_accept = 0.;
    for (unsigned int b=0; b<ntotal; b++) {
        R_CheckUserInterrupt();
        saveiter = b > nburnin && ((b-nburnin-1)%nthin==0);

        logp_new = 0.;
        if (eta_select.at(0) == 1) { // W unknown
            a3 = 0.5*arma::accu(arma::pow(arma::diff(theta.row(0)),2.));
            switch (eta_prior_type.at(0)) {
                case 0: // Gamma, Wtilde = log(W)
                {
                    coef_W wcoef[1] = {{a1,a2,a3}};
                    aw = optimize_postW_gamma(wcoef[0]); // map of Wtilde
                    bw = postW_deriv2(aw,a2,a3); // second order derivative wsp Wtilde
                    Wnew = std::exp(std::min(R::rnorm(aw,MH_sd*std::sqrt(-1./bw)),UPBND));
                    
                    // Wnew = std::exp(std::min(R::rnorm(std::log(W),MH_sd),UPBND));
                    logp_new = R::dgamma(Wnew,eta_prior_val.at(0,0),1./eta_prior_val.at(1,0),true);
                    logq_new = R::dnorm(std::log(Wnew),aw,MH_sd*std::sqrt(-bw),true);
                }
                break;
                case 1: // Half-Cauchy
                {}
                break;
                case 2: // Inverse-Gamma, Wtilde = -log(W)
                {
                    bw = eta_prior_val.at(1,0) + a3;
                    Wnew = 1./R::rgamma(aw,1./bw);
                    logq_new = -(aw+1.)*std::log(Wnew) - bw/Wnew;
                    // Wnew = std::exp(std::min(-R::rnorm(-std::log(W),MH_sd),UPBND));
                    logp_new = -(eta_prior_val.at(0,0)+1.)*std::log(Wnew)-eta_prior_val.at(1,0)/Wnew;
                }
                break;
                default:
                {}
            }

            Wnew_stored.at(b) = Wnew;
        }

        theta_stored.slice(0) = theta0;
        bf_poisson(
            theta_stored, w_stored, Gt,
            ypad, model_code, Wnew, mu0, rho, alpha, p,  L,  N, 
            Ft, Fphi, ctanh, delta_nb);
        
        logp_new += arma::accu(arma::log(w_stored)); // logarithm of the marginal likelihood

        logratio = std::min(0.,logp_new - logp_old + logq_old - logq_new);
        logratio_stored.at(b) = logratio;
        if (std::log(R::runif(0.,1.)) < logratio) {
            // accept
            W = Wnew;
            logp_old = logp_new;
            logq_old = logq_new;
            for (unsigned int t=0; t<(n+1); t++) {
                theta.col(t) = arma::median(theta_stored.slice(t),1); // p x (n+1)
            }

            if (debug) {
                Rcout << "W: " << Wnew << std::endl;
                theta.row(0).brief_print("psi: ");
            }
            

            mh_accept += 1.;
            accept.at(b) = 1;
        }



        if (saveiter || b==(ntotal-1)) {
            unsigned int idx_run;
			if (saveiter) {
				idx_run = (b-nburnin-1)/nthin;
			} else {
				idx_run = nsample - 1;
			}

            W_stored.at(idx_run) = W;
            theta_stored2.col(idx_run) = theta.row(0).t(); // (n+1) x nsample
        }

        if (verbose) {
			Rcout << "\rProgress: " << b << "/" << ntotal-1;
		}
    }

    if (verbose) {
		Rcout << std::endl;
	}

    arma::mat R(n+1,3); // quantiles
    R = arma::quantile(theta_stored2,Rcpp::as<arma::vec>(qProb),1);

    Rcpp::List output;
    output["psi"] = Rcpp::wrap(R); // (n+1) x 3
    output["W"] = Rcpp::wrap(W_stored);
    output["Wnew"] = Rcpp::wrap(Wnew_stored); // the rejected suitors
    output["mh_accept"] = mh_accept / static_cast<double>(ntotal);
    output["logratio"] = Rcpp::wrap(logratio_stored);
    output["accept"] = Rcpp::wrap(accept);
    output["theta_last"] = Rcpp::wrap(theta_stored); // p x N


    return output;
}




void mcs_poisson(
    arma::mat& R, // (n+1) x 2, (psi,theta)
    const arma::vec& ypad, // (n+1) x 1, the observed response
    const arma::uvec& model_code, // (obs_code,link_code,transfer_code,gain_code,err_code)
	const double W = NA_REAL,
    const double rho = 0.9,
    const double alpha = 1.,
    const unsigned int L = 12, // number of lags
    const double mu0 = 2.220446e-16,
    const unsigned int B = 12, // length of the B-lag fixed-lag smoother (Anderson and Moore 1979; Kitagawa and Sato)
	const unsigned int N = 5000, // number of particles
    const Rcpp::Nullable<Rcpp::NumericVector>& m0_prior = R_NilValue,
	const Rcpp::Nullable<Rcpp::NumericMatrix>& C0_prior = R_NilValue,
    const Rcpp::NumericVector& ctanh = Rcpp::NumericVector::create(0.2,0,5),
    const double delta_nb = 1., // 0: negative binomial DLM; 1: poisson DLM
    const double delta_discount = 0.95){ 

    unsigned int tmpi; // store temporary integer value
    const unsigned int n = ypad.n_elem - 1; // number of observations
    const double min_eff = 0.8*static_cast<double>(N);

    double Wsqrt;
    if (!R_IsNA(W)) {
        Wsqrt = std::sqrt(W);
    }

    /* 
    Dimension of state space depends on type of transfer functions 
    - p: diemsnion of DLM state space
    - Ft: vector for the state-to-observation function
    - Gt: matrix for the state-to-state function
    */
    const unsigned int obs_code = model_code.at(0);
    const unsigned int link_code = model_code.at(1);
    const unsigned int trans_code = model_code.at(2);
    const unsigned int gain_code = model_code.at(3);
    const unsigned int err_code = model_code.at(4);
	unsigned int p, L_;
	arma::vec Ft, Fphi;
    arma::mat Gt;
    init_by_trans(p,L_,Ft,Fphi,Gt,trans_code,L);

    Fphi = arma::pow(Fphi,alpha);
    arma::vec Fy(p,arma::fill::zeros);
    /* Dimension of state space depends on type of transfer functions */


    // theta: DLM state vector
    // p - dimension of state space
    // N - number of particles
    arma::vec lambda(N); // instantaneous intensity
    arma::vec omega(N); // evolution variance
    arma::vec w(N); // importance weight of each particle
    Rcpp::IntegerVector idx_(N);
    arma::uvec idx(N);

    /*
    ------ Step 1. Initialization at time t = -1 ------
        - Sample theta[-1] from the prior of theta[-1]
    
    ------
    The first B-2 rows (indices from 0 to B-2) of R are all zeros, as illustrated as follows:

    At the beginning of t = 0
        - propagate to theta[0]
        - theta_stored.slice(B-2) = theta[-1], theta_stored.slice(B-1) = theta[0]
        - resample theta_stored
        - save theta_stored.slice(0)==ZERO to R.row(0)

    At the beginning of t = 1
        - propagate to theta[1]
        - theta_stored.slice(B-3) = theta[-1], ..., theta_stored.slice(B-1) = theta[1]
        - resample theta_stored
        - save theta_stored.slice(0)==ZERO to R.row(1)

    ...

    At the beginning of t = B-2
        - propagate to theta[B-2]
        - theta_stored.slice(0) = theta[-1], ..., theta_stored.slice(B-1) = theta[B-2]
        - resample theta_stored
        - save theta_stored.slice(0)==theta[-1] to R.row(B-2); *** theta[-1] has been resampled B-1 times

    At the beginning of t = B-1
        - propagate to theta[B-1]
        - theta_stored.slice(0) = theta[0], ..., theta_stored.slice(B-1) = theta[B-1]
        - resample theta_stored
        - save theta_stored.slice(0)==theta[0] to R.row(B-1); *** theta[0] has been resampled B times

    At the beginning of t = n-1, theta_stored.slice(B-1) = theta[n-2]; resample; save theta_stored.slice(0)=theta[n-B-1] to R.row(n-1); propagate to theta[n-1]
        - propagate to theta[n-1]
        - theta_stored.slice(0) = theta[n-B], ..., theta_stored.slice(B-1) = theta[n-1]
        - resample theta_stored
        - save theta_stored.slice(0)==theta[n-B] to R.row(n-1); *** theta[n-B] has been resampled B times
            - theta[n-B+1] has been resample B-1 times
            .....
            - theta[n-2] is resampled 2 times
            - theta[n-1] is resample once
    ------
    >>>>>> Outside of the for loop,
        - The first B-2 rows (indices from 0 to B-1) of R are all zeros
        - Shift the the last (B-2):(n-1) rows (n-B+2 in totals) of R to 0:n-B+1
        - For the last B-1 rows (indices from n-B+2 to n), takes values from theta_stored.slice(1,B-1)


    In order to save the theta resampled B-1 times to R, we just need to save the first slice of theta_stored.
    */
        
    arma::mat theta(p,N);
    arma::cube theta_stored(p,N,B);
    arma::mat hpsi;
    if (link_code==1) {
        // Exponential Link
        arma::vec m0(p,arma::fill::zeros);
        arma::mat C0(p,p,arma::fill::eye); C0 *= 3.;
        if (!m0_prior.isNull()) {
		    m0 = Rcpp::as<arma::vec>(m0_prior);
	    }
	    if (!C0_prior.isNull()) {
		    C0 = Rcpp::as<arma::mat>(C0_prior);
            C0 = arma::chol(C0);
	    }

        for (unsigned int i=0; i<N; i++) {
            theta.col(i) = m0 + C0.t() * arma::randn(p);
        }

    } else {
        // Identity Link
        theta = arma::randu(p,N,arma::distr_param(0.,10.)); // Consider it as a flat prior
    }
    theta_stored.slice(B-1) = theta;
    /*
    ------ Step 1. Initialization at time t = 0 ------
    */
    const double c1 = std::pow(1.-rho,static_cast<double>(L_)*alpha);
    double c2 = -rho;
    // const double c3 = -rho*rho;
    arma::vec mt(p);
    arma::vec Wt(n);
    bool resample = false;

    for (unsigned int t=0; t<n; t++) {
        R_CheckUserInterrupt();
        // theta_stored: p x N x B, with B is the lag of the B-lag fixed-lag smoother
        // theta: p x N
        if (trans_code == 1)  {
            // Koyama
            Fy.zeros();
            tmpi = std::min(t,p);
            if (t>0) {
                // Checked Fy - CORRECT
                Fy.head(tmpi) = arma::reverse(ypad.subvec(t+1-tmpi,t));
            }

            Ft = Fphi % Fy; // L(p) x 1
        }
        /*
        ------ Step 1.0 Resample ------
        Auxiliary Particle Filter
        */
        if (resample && t>B) {
            mt = arma::median(theta_stored.slice(B-1),1);
            update_Gt(Gt,gain_code, trans_code, mt, ctanh, alpha, ypad.at(t), rho);
            theta = update_at(p,gain_code,trans_code,theta_stored.slice(B-1),Gt,ctanh,alpha,ypad.at(t),rho);
            if (link_code==1) {
                // Exponential link and identity gain
                for (unsigned int i=0; i<N; i++) {
                    lambda.at(i) = mu0 + arma::as_scalar(Ft.t()*theta.col(i));
                }
                lambda.elem(arma::find(lambda>UPBND)).fill(UPBND);
                lambda = arma::exp(lambda);
            } else if (trans_code==1){ // Koyama
                hpsi = psi2hpsi(theta,gain_code,ctanh); // hpsi: p x N
                for (unsigned int i=0; i<N; i++) {
                    lambda.at(i) = mu0 + arma::as_scalar(Ft.t()*hpsi.col(i));
                }
            } else {
                // Koyck or Solow with identity link and different gain functions
                lambda = mu0 + theta.row(1).t();
            }

            for (unsigned int i=0; i<N; i++) {
                w.at(i) = loglike_obs(ypad.at(t+1),lambda.at(i),obs_code,delta_nb,false);
            } // End for loop of i, index of the particles

            if (arma::accu(w)>EPS) { // normalize the particle weights
                w /= arma::accu(w);
                if (1./arma::dot(w,w) > min_eff) {
                    idx_ = Rcpp::sample(N,N,true,Rcpp::as<Rcpp::NumericVector>(Rcpp::wrap(w)));
                    idx = Rcpp::as<arma::uvec>(idx_) - 1;
                    for (unsigned int b=0; b<B; b++) {
                        theta_stored.slice(b) = theta_stored.slice(b).cols(idx);
                    }
                }
            }
        }
        
        /*
        ------ Step 2.1 Propagate ------
        Propagate theta[t,i] to theta[t+1,i] for i=0,1,...,N
        using state evolution distribution P(theta[t+1,i]|theta[t,i]).
        ------
        >> Step 2-1 and 2-2 of Kitagawa and Sato - This is also the state equation
        */
        
        mt = arma::median(theta_stored.slice(B-1),1);
        update_Gt(Gt,gain_code, trans_code, mt, ctanh, alpha, ypad.at(t), rho);
        theta = update_at(p,gain_code,trans_code,theta_stored.slice(B-1),Gt,ctanh,alpha,ypad.at(t),rho);

        if (R_IsNA(W)) {
            Wsqrt = arma::stddev(theta_stored.slice(B-1).row(0));
            if (t>B) {
                Wsqrt *= std::sqrt(1./delta_discount-1.);
            } else {
                Wsqrt *= std::sqrt(1./0.99-1.);
            }
        }
        omega = arma::randn(N) * Wsqrt;
        
        // Wt.at(t) = arma::stddev(theta_stored.slice(B-1).row(0));
        // omega = arma::randn(N) * Wt.at(t);
        // if (t>B) {
        //     omega *= std::sqrt(1./delta_discount-1.);
        // } else {
        //     omega *= std::sqrt(1./0.99-1.);
        // }
        theta.row(0) += omega.t();
        
        theta_stored.slices(0,B-2) = theta_stored.slices(1,B-1);
        theta_stored.slice(B-1) = theta;
        /*
        ------ Step 2.1 Propagate ------
        */
        
        

        /*
        ------ Step 2.2 Importance weights ------
        Calculate the importance weight of lambda[t-L:t,i] for i=0,1,...,N, where i is the index of particles
        using likelihood P(y[t]|lambda[t,i]).
        ------
        >> Step 2-3 and 2-4 of Kitagawa and Sato - This is also the observational equation (nonlinear, nongaussian)
        */
        if (link_code == 1) {
            // Exponential link and identity gain
            for (unsigned int i=0; i<N; i++) {
                lambda.at(i) = mu0 + arma::as_scalar(Ft.t()*theta.col(i));
            }
            lambda.elem(arma::find(lambda>UPBND)).fill(UPBND);
            lambda = arma::exp(lambda);
        } else if (trans_code==1){ // Koyama
            hpsi = psi2hpsi(theta,gain_code,ctanh); // hpsi: p x N
            for (unsigned int i=0; i<N; i++) {
                lambda.at(i) = mu0 + arma::as_scalar(Ft.t()*hpsi.col(i));
            }
        } else {
            // Koyck or Solow with identity link and different gain functions
            lambda = mu0 + theta.row(1).t();
        }


        for (unsigned int i=0; i<N; i++) {
            w.at(i) = loglike_obs(ypad.at(t+1),lambda.at(i),obs_code,delta_nb,false);
        } // End for loop of i, index of the particles


        /*
        ------ Step 2.2 Importance weights ------
        */

        /*
        ------ Step 3 Resampling with Replacement ------
        */
        if (arma::accu(w)>EPS) { // normalize the particle weights
            w /= arma::accu(w);
            try{
                idx_ = Rcpp::sample(N,N,true,Rcpp::as<Rcpp::NumericVector>(Rcpp::wrap(w)));
                idx = Rcpp::as<arma::uvec>(idx_) - 1;
                for (unsigned int b=0; b<B; b++) {
                    theta_stored.slice(b) = theta_stored.slice(b).cols(idx);
                }
            } catch(...) {
                // If resampling doesn't work, then just don't resample
            }
        } 

        
        
        /*
        ------ Step  Resampling with Replacement ------
        */
        R.at(t,0) = arma::median(theta_stored.slice(0).row(0)); // psi

        if (trans_code == 1) { // TODO: CHECK THE KOYAMA CASE
            // Koyama
            hpsi = psi2hpsi(theta_stored.slice(0),gain_code,ctanh); // hpsi: p x N
            for (unsigned int i=0; i<N; i++) {
                lambda.at(i) = arma::as_scalar(Ft.t()*hpsi.col(i));
            }
            R.at(t,1) = arma::median(lambda);
        } else {
            // Koyck or Solow
            R.at(t,1) = arma::median(theta_stored.slice(0).row(1)); // theta
        }
    }

    /*
    ------ R: an n x 3 matrix ------
        - The first B-2 rows (indices from 0 to B-1) of R are all zeros
        - Shift the the last (B-2):(n-1) rows (n-B+2 in totals) of R to 0:n-B+1
        - For the last B-1 rows (indices from n-B+2 to n), takes values from theta_stored.slice(1,B-1)
    */
    // R.subvec(0,n-B+1) = R.subvec(B-2,n-1);
    // R.submat(0,0,n-B+1,0) = R.submat(B-2,0,n-1,0);
    R.rows(0,n-B+1) = R.rows(B-2,n-1);
    for (unsigned int b=0; b<(B-1); b++) {
        R.at(n-B+2+b,0) = arma::median(theta_stored.slice(b+1).row(0));

        if (trans_code == 1) { // TODO: CHECK THE KOYAMA CASE
            // Koyama
            Fy.zeros();
            tmpi = std::min(n-B+2+b,p);
            Fy.head(tmpi) = arma::reverse(ypad.subvec(n-B+2+b+1-tmpi,n-B+2+b));
            Ft = Fphi % Fy; // L(p) x 1

            hpsi = psi2hpsi(theta_stored.slice(b+1),gain_code,ctanh); // hpsi: p x N
            for (unsigned int i=0; i<N; i++) {
                lambda.at(i) = arma::as_scalar(Ft.t()*hpsi.col(i));
            }
            R.at(n-B+2+b,1) = arma::median(lambda);
        } else {
            R.at(n-B+2+b,1) = arma::median(theta_stored.slice(b+1).row(1));
        }
    }

    return;
}



/*
Particle Learning
- Reference: Carvalho et al., 2010.

- eta = (W, mu[0], rho, M)
*/
//' @export
// [[Rcpp::export]]
Rcpp::List pl_poisson(
    const arma::vec& Y, // n x 1, the observed response
    const arma::uvec& model_code,
    const arma::uvec& eta_select, // 4 x 1, indicator for unknown (=1) or known (=0)
    const arma::vec& eta_init, // 4 x 1, if true/initial values should be provided here
    const arma::uvec& eta_prior_type, // 4 x 1
    const arma::mat& eta_prior_val, // 2 x 4, priors for each element of eta
    const double alpha = 1.,
    const unsigned int L = 2, // number of lags
	const unsigned int N = 5000, // number of particles
    const Rcpp::Nullable<Rcpp::NumericVector>& m0_prior = R_NilValue,
	const Rcpp::Nullable<Rcpp::NumericMatrix>& C0_prior = R_NilValue,
    const Rcpp::NumericVector& qProb = Rcpp::NumericVector::create(0.025,0.5,0.975),
    const double delta_nb = 1.,
    const bool verbose = true,
    const bool debug = false){ 
    
    unsigned int tmpi; // store temporary integer value
    const unsigned int n = Y.n_elem; // number of observations
    const double n_ = static_cast<double>(n);
    const unsigned int M = 100;
    const unsigned int B = 0.1*n;
    const double min_eff = 0.8*static_cast<double>(N);
    arma::vec ypad(n+1,arma::fill::zeros); ypad.tail(n) = Y;

    double mu0 = eta_init.at(1);
    double rho = eta_init.at(2);
    const Rcpp::NumericVector ctanh = {1./eta_init.at(3),0.,eta_init.at(3)}; // (1./M, 0, M)

    /* 
    Dimension of state space depends on type of transfer functions 
    - p: diemsnion of DLM state space
    - Ft: vector for the state-to-observation function
    - Gt: matrix for the state-to-state function
    */
    const unsigned int obs_code = model_code.at(0);
    const unsigned int link_code = model_code.at(1);
    const unsigned int trans_code = model_code.at(2);
    const unsigned int gain_code = model_code.at(3);
    const unsigned int err_code = model_code.at(4);
	unsigned int p, L_;
	arma::vec Ft, Fphi;
    arma::mat Gt;
    init_by_trans(p,L_,Ft,Fphi,Gt,trans_code,L);

    Fphi = arma::pow(Fphi,alpha);
    arma::vec Fy(p,arma::fill::zeros);
    /* Dimension of state space depends on type of transfer functions */

    // theta: DLM state vector
    // p - dimension of state space
    // N - number of particles
    arma::vec lambda(N); // instantaneous intensity
    arma::vec omega(N); // evolution variance
    arma::vec w(N); // importance weight of each particle
    Rcpp::IntegerVector idx_(N);
    arma::uvec idx(N);
        
    arma::mat theta(p,N);
    arma::cube theta_stored(p,N,n+1);
    arma::mat hpsi;
    if (link_code==1) {
        // Exponential Link
        arma::vec m0(p,arma::fill::zeros);
        arma::mat C0(p,p,arma::fill::eye); C0 *= 3.;
        if (!m0_prior.isNull()) {
		    m0 = Rcpp::as<arma::vec>(m0_prior);
	    }
	    if (!C0_prior.isNull()) {
		    C0 = Rcpp::as<arma::mat>(C0_prior);
            C0 = arma::chol(C0);
	    }

        for (unsigned int i=0; i<N; i++) {
            theta.col(i) = m0 + C0.t() * arma::randn(p);
        }

    } else {
        // Identity Link
        theta = arma::randu(p,N,arma::distr_param(0.,10.)); // Consider it as a flat prior
    }

    theta_stored.slice(0) = theta;
    /*
    ------ Step 1. Initialization of theta[0] at time t = 0 ------
    */
    const double c1 = std::pow(1.-rho,static_cast<double>(L_)*alpha);
    double c2 = -rho;


    arma::mat psi(n+1,N,arma::fill::zeros); // particles
    psi.row(0) = theta.row(0);
    arma::vec Meff(n,arma::fill::zeros); // Effective sample size (Ref: Lin, 1996; Prado, 2021, page 196)
    arma::uvec resample_status(n,arma::fill::zeros);


    arma::vec W(N); W.fill(eta_init.at(0));
    arma::mat W_stored(N,n);
    arma::vec res_stored(n);
    arma::vec Wsqrt = arma::sqrt(W);
    arma::vec res(N,arma::fill::zeros); // Sufficient statistics for W
    arma::vec aw(N);
    arma::vec bw(N);
    double a1, a2, a3;
    if (eta_select.at(0) == 1) { // W unknown
        switch (eta_prior_type.at(0)) {
            case 0: // Gamma, Wtilde = log(W)
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
            case 2: // Inverse-Gamma, Wtilde = -log(W)
            {
                aw.fill(eta_prior_val.at(0,0) + 0.5*n_);
                bw.fill(eta_prior_val.at(1,0));
                // bw = eta_prior_val.at(1,0);
            }
            break;
            default:
            {

            }
        }
    }

    arma::vec mt(p);


    for (unsigned int t=0; t<n; t++) {
        R_CheckUserInterrupt();
        // theta_next: p x N, with B is the lag of the B-lag fixed-lag smoother
        // theta: p x N
        if (trans_code == 1) {
            // Koyama
            Fy.zeros();
            tmpi = std::min(t,p);
            if (t>0) {
                // Checked Fy - CORRECT
                // Fy.head(tmpi) = arma::reverse(Y.subvec(t-tmpi,t-1));
                Fy.head(tmpi) = arma::reverse(ypad.subvec(t-tmpi+1,t));
            }

            Ft = Fphi % Fy; // L(p) x 1
        }
        /*
        ------ Step 2.0 Resample ------
        Resampling theta[t,i] using the predictive likelihood P(y[t+1] | theta[t,i]).
        It doesn't have analytical form so we use P(y[t+1] | theta[t+1,i]_hat = gt(theta[t,i]))
        */
        // E(theta[t+1] | theta[t]), theta[t] is obtained from the previous time point via initialization or SMC
        mt = arma::median(theta_stored.slice(t),1);
        update_Gt(Gt,gain_code, trans_code, mt, ctanh, alpha, ypad.at(t), rho);
        hpsi = update_at(p,gain_code,trans_code,theta_stored.slice(t),Gt,ctanh,alpha,ypad.at(t),rho); // p x N, theta[t+1,]
        if (trans_code == 1) {
            // Koyama
            hpsi = psi2hpsi(hpsi,gain_code,ctanh); // hpsi: p x N
        }

        w.zeros();
        for (unsigned int i=0; i<N; i++) {
            lambda.at(i) = mu0 + arma::as_scalar(Ft.t()*hpsi.col(i));
            if (link_code == 1) {
                lambda.at(i) = std::exp(std::min(lambda.at(i),UPBND));
            }

            w.at(i) = loglike_obs(ypad.at(t+1),lambda.at(i),obs_code,delta_nb,false);            
        } // End for loop of i, index of the particles

        if (arma::accu(w)>EPS) { // normalize the particle weights
            w /= arma::accu(w);
            if (1./arma::dot(w,w) > min_eff) {
                idx_ = Rcpp::sample(N,N,true,Rcpp::as<Rcpp::NumericVector>(Rcpp::wrap(w)));
                idx = Rcpp::as<arma::uvec>(idx_) - 1;
                for (unsigned int b=0; b<=t; b++) {
                    theta_stored.slice(b) = theta_stored.slice(b).cols(idx);
                }
            }
        }

        if (debug) {
            Rcout << "Done step 2.0" << std::endl;
        }
        /*
        ------ Step 2.0 Resample ------
        */

        /*
        ------ Step 2.1 Propagate ------
        Propagate theta[t,i] to theta[t+1,i] for i=0,1,...,N
        using state evolution distribution P(theta[t+1,i]|theta[t,i]).
        ------
        >> Step 2-1 and 2-2 of Kitagawa and Sato - This is also the state equation
        */
        
        if (t>B) {
            Wsqrt = arma::sqrt(W);
            omega = arma::randn(N) % Wsqrt;
        } else {
            omega = arma::randn(N);
            omega *= arma::stddev(theta_stored.slice(t).row(0));
            omega *= std::sqrt(1./0.99-1.);
        }
        

        mt = arma::median(theta_stored.slice(t),1);
        update_Gt(Gt,gain_code, trans_code, mt, ctanh, alpha, ypad.at(t), rho);
        theta = update_at(p,gain_code,trans_code,theta_stored.slice(t),Gt,ctanh,alpha,ypad.at(t),rho);
        theta.row(0) += omega.t();

        theta_stored.slice(t+1) = theta;

        if (debug) {
            Rcout << "Done step 2.1" << std::endl;
        }
        /*
        ------ Step 2.1 Propagate ------
        */
        
        

        /*
        ------ Step 2.2 Importance weights ------
        Calculate the importance weight of lambda[t-L:t,i] for i=0,1,...,N, where i is the index of particles
        using likelihood P(y[t+1]|lambda[t+1,i]).
        ------
        */
        if (link_code==1) {
            // Exponential link and identity gain
            for (unsigned int i=0; i<N; i++) {
                lambda.at(i) = mu0 + arma::as_scalar(Ft.t()*theta.col(i));
            }
            lambda.elem(arma::find(lambda>UPBND)).fill(UPBND);
            lambda = arma::exp(lambda);
        } else if (trans_code==1){ // Koyama
            hpsi = psi2hpsi(theta,gain_code,ctanh); // hpsi: p x N
            for (unsigned int i=0; i<N; i++) {
                lambda.at(i) = mu0 + arma::as_scalar(Ft.t()*hpsi.col(i));
            }
        } else {
            // Koyck or Solow with identity link and different gain functions
            lambda = mu0 + theta.row(1).t();
        }

        for (unsigned int i=0; i<N; i++) {
            w.at(i) = loglike_obs(ypad.at(t+1),lambda.at(i),obs_code,delta_nb,false);
        } // End for loop of i, index of the particles

        if (debug) {
            Rcout << "Done step 2.2" << std::endl;
        }
        /*
        ------ Step 2.2 Importance weights ------
        */

        /*
        ------ Step 3 Resampling with Replacement ------
        */
        if (arma::accu(w)>EPS) { // normalize the particle weights
            w /= arma::accu(w);
            Meff.at(t) = 1./arma::dot(w,w);
            idx_ = Rcpp::sample(N,N,true,Rcpp::as<Rcpp::NumericVector>(Rcpp::wrap(w)));
            idx = Rcpp::as<arma::uvec>(idx_) - 1;
            for (unsigned int b=0; b<=(t+1); b++) {
                theta_stored.slice(b) = theta_stored.slice(b).cols(idx);
            }
            resample_status.at(t) = 1;
        } else {
            resample_status.at(t) = 0;
            Meff.at(t) = 0.;
        }

        if (debug) {
            Rcout << "Done step 3.0" << std::endl;
        }
        /*
        ------ Step 3 Resampling with Replacement ------
        */

        /*
        ------ Step 4 Sufficient Statistics for Global/Static Parameters ------
        ------ Step 5 Sample Global/Static Parameters ------
        */
        if (eta_select.at(0) == 1 && t>B) {
            for (unsigned int i=0; i<N; i++) {
                psi.col(i) = arma::vectorise(theta_stored.tube(0,i));
            }
            res = arma::vectorise(arma::sum(arma::pow(arma::diff(psi.head_rows(t+1),1,0),2),0));
            res_stored.at(t) = arma::median(res);
            // res += arma::vectorise(arma::pow(psi.row(t+1) - psi.row(t),2)); // N x 1; psi: (n+1) x N
            switch (eta_prior_type.at(0)) {
                case 0: // Gamma, Wtilde = log(W)
                {
                    for (unsigned int i=0; i<N; i++) {
                        a3 = 0.5*res.at(i);
                        coef_W wcoef[1] = {{a1,a2,a3}};
                        aw.at(i) = optimize_postW_gamma(wcoef[0]);
                        bw.at(i) = postW_deriv2(aw.at(i),a2,a3);
                        // W_stored.at(i) = std::exp(std::min(R::rnorm(aw.at(i),std::sqrt(-0.1/bw.at(i))),UPBND));
                        W.at(i) = std::exp(std::min(aw.at(i),UPBND));
                    }
                }
                break;
                case 1: // Half-Cauchy
                {
                    ::Rf_error("Half-Cauchy prior for W not implemented yet.");
                }
                break;
                case 2: // Inverse-Gamma
                {
                    bw = eta_prior_val.at(1,0) + 0.5*res;
                    for (unsigned int i=0; i<N; i++) {
                        W.at(i) = 1./R::rgamma(aw.at(i),1./bw.at(i));
                    }
                }
                break;
                default:
                {
                    ::Rf_error("This prior for W is not supported yet.");
                }
            }

            W_stored.col(t) = W;

            if (debug) {
            Rcout << "Done step 4/5" << std::endl;
        }
        }
        
        if (verbose) {
			Rcout << "\rFiltering Progress: " << t+1 << "/" << n;
		}
        
    }

    if (verbose) {
		Rcout << std::endl;
	}


    Rcpp::List output;
    arma::mat R = arma::quantile(psi,Rcpp::as<arma::vec>(qProb),1);
    output["psi"] = Rcpp::wrap(R); // (n+1) x 3
    output["W"] = Rcpp::wrap(arma::pow(W_stored.col(n-1),2.));
    output["Wstored"] = Rcpp::wrap(W_stored);
    output["aw"] = Rcpp::wrap(aw);
    output["bw"] = Rcpp::wrap(bw); 
    output["res"] = Rcpp::wrap(res_stored);

    // if (debug) {
        output["Meff"] = Rcpp::wrap(Meff);
        output["resample_status"] = Rcpp::wrap(resample_status);
    // }
    

    return output;
}

