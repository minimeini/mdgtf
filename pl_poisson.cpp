#include "pl_poisson.h"
using namespace Rcpp;
// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::depends(RcppArmadillo,nloptr)]]



/*
---- Model ----
<obs>   y[t] ~ Pois(lambda[t])
<link>  lambda[t] = theta[t]
<state> theta[t] = 2*rho*theta[t-1] - rho^2*theta[t-2] + (1-rho)^2*x[t-1]*exp(beta[t-1])
        beta[t] = beta[t-1] + omega[t]
        omega[t] ~ Normal(0,W)

Unknown parameters: W, theta[0:n], beta[0:n]
*/
Rcpp::List pl_pois_solow_eye_exp(
    const arma::vec& Y, // n x 1, the observed response
	const arma::vec& X, // n x 1, the exogeneous variable in the transfer function
    const double rho,
	const Rcpp::Nullable<Rcpp::NumericVector>& QPrior = R_NilValue, // (aw, Rw), W ~ IG(aw/2, Rw/2), prior for v[1], the first state/evolution/state error/disturbance.
	const double Q_init = NA_REAL,
	const double Q_true = NA_REAL, // true value of state/evolution error variance
	const unsigned int N = 5000) { // number of particles

    const unsigned int n = Y.n_elem; // number of observations

    const double coef = (1.-rho)*(1.-rho);
    const double rho2 = rho*rho;

    arma::mat theta(N,n+1,arma::fill::zeros); 
    arma::mat beta(N,n+1,arma::fill::zeros);
    arma::vec theta_next(N);
    arma::vec w(N);

    arma::vec aw(N);
    arma::vec bw(N);
	arma::vec Q(N);
	bool Qflag;
	if (R_IsNA(Q_true)) {
		Qflag = true;
		if (QPrior.isNull()) {
			stop("Error: You must provide either true value or prior for Q.");
		}
		arma::vec QPrior_ = Rcpp::as<arma::vec>(QPrior);
		aw.fill(QPrior_.at(0));
		bw.fill(QPrior_.at(1));
		if (R_IsNA(Q_init)) {
			Q.fill(1./R::rgamma(0.5*aw.at(0),2./bw.at(0)));
		} else {
			Q.fill(Q_init);
		}
	} else {
		Qflag = false;
		Q.fill(Q_true);
	}

    Rcpp::IntegerVector idx_(N);
    arma::uvec idx(N);
    arma::vec vec_tmp(N);

    for (unsigned int t=0; t<n; t++) {
        /* 
        1. Resample theta[t,i] for i=0,1,...,N 
        using predictive distribution P(y[t+1]|theta[t,i]) 
        */
        // Calculate theta[t+1,i] as a function of theta[t,i]
        if (t==0) {
            theta_next = 2.*rho*theta.col(t) + coef*X.at(t)*arma::exp(beta.col(t));
        } else {
            theta_next = 2.*rho*theta.col(t) - rho2*theta.col(t-1) + coef*X.at(t)*arma::exp(beta.col(t));
        }
        theta_next.elem(arma::find(theta_next<=0)).fill(arma::datum::eps);
        // P(y[t+1]|theta[t,i]) = P(y[t+1]|theta[t+1,i])
        for (unsigned int i=0; i<N; i++) {
            w.at(i) = R::dpois(Y.at(t),theta_next.at(i),true);
        }
        w.elem(arma::find(w>100.)).fill(100.);
        // w += arma::datum::eps;
        w = arma::exp(w);
        w /= arma::accu(w);

        // Resample
        idx_ = Rcpp::sample(N,N,true,Rcpp::as<Rcpp::NumericVector>(Rcpp::wrap(w)));
        idx = Rcpp::as<arma::uvec>(idx_) - 1;
        vec_tmp = theta.col(t);
        theta.col(t) = vec_tmp.elem(idx);
        vec_tmp = beta.col(t);
        beta.col(t) = vec_tmp.elem(idx);
        Q = Q.elem(idx);

        /* 
        2. Propagate theta[t,i] to theta[t+1,i] for i=0,1,...,N 
        using posterior distribution P(theta[t+1,i]|theta[t,i],y[t+1])
        */
        if (t==0) {
            theta.col(t+1) = 2.*rho*theta.col(t) + coef*X.at(t)*arma::exp(beta.col(t));
        } else {
            theta.col(t+1) = 2.*rho*theta.col(t) - rho2*theta.col(t-1) + coef*X.at(t)*arma::exp(beta.col(t));
        }

        for (unsigned int i=0; i<N; i++) {
            w.at(i) = R::rnorm(0,std::sqrt(Q.at(i)));
        }
        beta.col(t+1) = beta.col(t) + w;

        theta_next = theta.col(t+1);
        theta_next.elem(arma::find(theta_next<=0)).fill(arma::datum::eps);
        theta.col(t+1) = theta_next;
        
        for (unsigned int i=0; i<N; i++) {
            w.at(i) = R::dpois(Y.at(t),theta_next.at(i),true);
        }
        w.elem(arma::find(w>100.)).fill(100.);
        // w += arma::datum::eps;
        w = arma::exp(w);
        w /= arma::accu(w);

        idx_ = Rcpp::sample(N,N,true,Rcpp::as<Rcpp::NumericVector>(Rcpp::wrap(w)));

        idx = Rcpp::as<arma::uvec>(idx_) - 1;
        vec_tmp = theta.col(t+1);
        theta.col(t+1) = vec_tmp.elem(idx);
        vec_tmp = beta.col(t+1);
        beta.col(t+1) = vec_tmp.elem(idx);

        /* 
        3. Update sufficient statistics and update static parameter Q
        */
        if (Qflag) {
            // Update sufficient statistics
            aw = aw + 1.;
            bw = bw + arma::pow(w,2.);
            // Sample static parameter
            for (unsigned int i=0; i<N; i++) {
                Q.at(i) = 1./R::rgamma(0.5*aw.at(i),2./bw.at(i));
            }
        }
    }

    Rcpp::List output;
    output["theta"] = Rcpp::wrap(theta);
    output["beta"] = Rcpp::wrap(beta);
    output["Q"] = Rcpp::wrap(Q);

    return output;
}


/*
---- Method ----
- Particle learning with rho known
- Using the DLM formulation

---- Model ----
<obs>   y[t] ~ Pois(lambda[t])
<link>  lambda[t] = theta[t]
<state> theta[t] = 2*rho*theta[t-1] - rho^2*theta[t-2] + (1-rho)^2*x[t-1]*exp(beta[t-1])
        beta[t] = beta[t-1] + omega[t]
        omega[t] ~ Normal(0,W)

Unknown parameters: theta[0:n], beta[0:n]
Kwg: Identity link, exponential state space
*/
Rcpp::List pl_pois_solow_eye_exp2(
    const arma::vec& Y, // n x 1, the observed response
	const arma::vec& X, // n x 1, the exogeneous variable in the transfer function
    const double rho,
	const Rcpp::Nullable<Rcpp::NumericVector>& QPrior = R_NilValue, // (aw, Rw), W ~ IG(aw/2, Rw/2), prior for v[1], the first state/evolution/state error/disturbance.
	const double Q_init = NA_REAL,
    const double Q_true = NA_REAL,
	const unsigned int N = 5000, // number of particles
    const unsigned int B = 30,
    const bool resample = true) { // Step of particle backshifting

    const unsigned int n = Y.n_elem; // number of observations

    const double coef = (1.-rho)*(1.-rho);
    const double rho2 = rho*rho;

    arma::mat theta(3,N,arma::fill::zeros);
    arma::mat theta_new(3,N,arma::fill::zeros);

    arma::vec theta_next(N);
    arma::mat theta_stored(N,n);
    arma::mat beta_stored(N,n);

    arma::vec aw(N);
    arma::vec bw(N);
	arma::vec Q(N);
    arma::mat err(N,n);
	bool Qflag;
	if (R_IsNA(Q_true)) {
		Qflag = true;
		if (QPrior.isNull()) {
			stop("Error: You must provide either true value or prior for Q.");
		}
		arma::vec QPrior_ = Rcpp::as<arma::vec>(QPrior);
		aw.fill(QPrior_.at(0));
		bw.fill(QPrior_.at(1));
		if (R_IsNA(Q_init)) {
			Q.fill(1./R::rgamma(0.5*aw.at(0),2./bw.at(0)));
		} else {
			Q.fill(Q_init);
		}
        err = arma::randn(N,n) * std::sqrt(Q_init);
	} else {
		Qflag = false;
		Q.fill(Q_true);
        err = arma::randn(N,n) * std::sqrt(Q_true);
	}

    Rcpp::IntegerVector idx_(N);
    arma::uvec idx(N);
    arma::uvec b = {2};
    arma::vec vec_tmp(N);
    arma::vec w(N);

    for (unsigned int t=0; t<n; t++) {
        theta = theta_new;
        /* 
        1. Resample theta[t,i] for i=0,1,...,N 
        using likelihood P(y[t]|theta[t,i]) 
        */
        for (unsigned int i=0; i<N; i++) {
            w.at(i) = R::dpois(Y.at(t),theta.at(0,i),true);
        }
        w.elem(arma::find(w>100.)).fill(100.);
        // w += arma::datum::eps;
        w = arma::exp(w);
        w /= arma::accu(w);

        // Resample
        idx_ = Rcpp::sample(N,N,true,Rcpp::as<Rcpp::NumericVector>(Rcpp::wrap(w)));
        idx = Rcpp::as<arma::uvec>(idx_) - 1;
        theta = theta.cols(idx);

        /* 
        4. Update sufficient statistics and update static parameter Q
        */
        if (Qflag && t>0) {
            // Update sufficient statistics
            b = t-1;
            err.col(t-1) = arma::vectorise(err(idx,b));
            
            aw = aw + 1.;
            bw = bw + arma::pow(err.col(t-1),2.);
            // Sample static parameter
            for (unsigned int i=0; i<N; i++) {
                Q.at(i) = 1./R::rgamma(0.5*aw.at(i),2./bw.at(i));
            }
        }


        /*
        3. Resample again using predictive distribution
        P(y[t+1]|theta[t,i])
        Can be skipped for particle filtering 
        but an essential step for particle learning
        */
        if (resample && t<(n-1)) {
            theta_next = 2.*rho*theta.row(0).t() - rho2*theta.row(1).t() + coef*X.at(t)*arma::exp(theta.row(2).t());
            for (unsigned int i=0; i<N; i++) {
                w.at(i) = R::dpois(Y.at(t+1),theta_next.at(i),true);
            }
            w.elem(arma::find(w>100.)).fill(100.);
            // w += arma::datum::eps;
            w = arma::exp(w);
            w /= arma::accu(w);

            idx_ = Rcpp::sample(N,N,true,Rcpp::as<Rcpp::NumericVector>(Rcpp::wrap(w)));
            idx = Rcpp::as<arma::uvec>(idx_) - 1;
            theta = theta.cols(idx);
            Q = Q.elem(idx);
        }


        theta_stored.col(t) = theta.row(0).t();
        beta_stored.col(t) = theta.row(2).t();


        /* 
        2. Propagate theta[t,i] to theta[t+1,i] for i=0,1,...,N 
        using state evolution distribution P(theta[t+1,i]|theta[t,i])
        */
        for (unsigned int i=0; i<N; i++) {
            err.at(i,t) = R::rnorm(0,std::sqrt(Q.at(i)));
        }
        theta_new.row(0) = 2.*rho*theta.row(0) - rho2*theta.row(1) + coef*X.at(t)*arma::exp(theta.row(2));
        theta_new.row(1) = theta.row(0);
        theta_new.row(2) = theta.row(2) + err.col(t).t();

        b.at(0) = 0;
        arma::uvec tmpp = arma::find(theta_new.row(0)<arma::datum::eps);
        theta_new(b,tmpp).zeros();
        theta_new(b,tmpp) += arma::datum::eps;



    }

    Rcpp::List output;
    output["theta"] = Rcpp::wrap(theta_stored);
    output["beta"] = Rcpp::wrap(beta_stored);
    output["Q"] = Rcpp::wrap(Q);

    return output;
}




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
    const unsigned int ModelCode,
	const double W,
    const double rho = 0.9,
    const double alpha = 1.,
    const unsigned int L = 12, // number of lags
    const double mu0 = 0.,
    const unsigned int B = 12, // length of the B-lag fixed-lag smoother (Anderson and Moore 1979; Kitagawa and Sato)
	const unsigned int N = 5000, // number of particles
    const Rcpp::Nullable<Rcpp::NumericVector>& m0_prior = R_NilValue,
	const Rcpp::Nullable<Rcpp::NumericMatrix>& C0_prior = R_NilValue,
    const Rcpp::NumericVector& qProb = Rcpp::NumericVector::create(0.025,0.5,0.975),
    const Rcpp::NumericVector& ctanh = Rcpp::NumericVector::create(0.3,0,1),
    const double delta_nb = 1.,
    const unsigned int obstype = 1, // 0: negative binomial DLM; 1: poisson DLM
    const bool verbose = false,
    const bool debug = false){ 
    
    const double UPBND = 700.;
    const double EPS = arma::datum::eps;

    unsigned int tmpi; // store temporary integer value
    const unsigned int n = Y.n_elem; // number of observations
    const double Wsqrt = std::sqrt(W);

    if (debug) {
        Rcout << "Evolution variance W=" << W << std::endl;
    }


    /* 
    Dimension of state space depends on type of transfer functions 
    - p: diemsnion of DLM state space
    */
    const unsigned int GainCode = get_gaincode(ModelCode);
    unsigned int TransferCode; // integer indicator for the type of transfer function
	unsigned int p; // dimension of DLM state space
	unsigned int L_;
	get_transcode(TransferCode,p,L_,ModelCode,L);
    /* Dimension of state space depends on type of transfer functions */

    
    /*
    Ft: vector for the state-to-observation function
    Gt: matrix for the state-to-state function
    */
    arma::vec Ft(p,arma::fill::zeros);
    arma::vec Fphi(p);
    arma::vec Fy(p,arma::fill::zeros);
    arma::mat Gt(p,p,arma::fill::zeros);
    switch (TransferCode) {
        case 0: // Koyck
        {
            Ft.at(1) = 1.; // Ft = (0,1)'
        }
        break;
        case 1: // Koyama
        {
            double mu = 2.2204e-16;
		    double m = 4.7;
		    double s = 2.9;
		    Fphi = get_Fphi(p,mu,m,s);
            Fphi = arma::pow(Fphi,alpha);

            Gt.at(0,0) = 1.;
            Gt.diag(-1).ones();

            // Check Fphi and G - CORRECT
        }
        break;
        case 2: // Solow
        {
            Ft.at(1) = 1.; // Ft = (0,1,0)
        }
        break;
        case 3: // Vanilla
        {
            Ft.at(0) = 1.;
        }
        break;
        default:
        {
            ::Rf_error("Not supported transfer function.");
        }
    } // End switch Transfercode


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
    if (ModelCode==6||ModelCode==7||ModelCode==8||ModelCode==9) {
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
    const double c1 = std::pow((1.-rho)*(1.-rho),alpha);
    const double c2 = 2.*rho;
    const double c3 = -rho*rho;
    
    arma::mat R(n+1,3); // quantiles
    arma::vec Meff(n,arma::fill::zeros); // Effective sample size (Ref: Lin, 1996; Prado, 2021, page 196)
    arma::uvec resample_status(n,arma::fill::zeros);

    for (unsigned int t=0; t<n; t++) {
        R_CheckUserInterrupt();
        // theta_stored: p x N x B, with B is the lag of the B-lag fixed-lag smoother
        // theta: p x N
        /*
        ------ Step 2.1 Propagate ------
        Propagate theta[t,i] to theta[t+1,i] for i=0,1,...,N
        using state evolution distribution P(theta[t+1,i]|theta[t,i]).
        ------
        >> Step 2-1 and 2-2 of Kitagawa and Sato - This is also the state equation
        */
        omega = arma::randn(N) * Wsqrt;
        if (TransferCode != 1) {
            hpsi = psi2hpsi(theta_stored.slice(B-1).row(0),GainCode,ctanh); // hpsi: 1 x N
        }
        
        switch (TransferCode) {
            case 0: // Koyck
            {
                theta.row(0) = theta_stored.slice(B-1).row(0) + omega.t();
                theta.row(1) = Y.at(t)*hpsi.row(0) + rho*theta_stored.slice(B-1).row(1);
            }
            break;
            case 1: // Koyama
            {
                for (unsigned int i=0; i<N; i++) {
                    theta.col(i) = Gt * theta_stored.slice(B-1).col(i);
                }
                theta.row(0) += omega.t();
            }
            break;
            case 2: // Solow
            {
                theta.row(0) = theta_stored.slice(B-1).row(0) + omega.t();
                theta.row(2) = theta_stored.slice(B-1).row(1);
                theta.row(1) = c1*Y.at(t)*hpsi.row(0) + c2*theta_stored.slice(B-1).row(1) + c3*theta_stored.slice(B-1).row(2);
            }
            break;
            case 3: // Vanilla
            {
                ::Rf_error("VanillaPois undefined.");
            }
            break;
            default:
            {
                ::Rf_error("Unknown type of model.");
            }
        }


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
        if (ModelCode==6||ModelCode==7||ModelCode==8||ModelCode==9) {
            // Exponential link and identity gain
            for (unsigned int i=0; i<N; i++) {
                lambda.at(i) = mu0 + arma::as_scalar(Ft.t()*theta.col(i));
            }
            lambda.elem(arma::find(lambda>UPBND)).fill(UPBND);
            lambda = arma::exp(lambda);
        } else if (TransferCode==1){ // Koyama
            Fy.zeros();
            tmpi = std::min(t,p);
            if (t>0) {
                // Checked Fy - CORRECT
                Fy.head(tmpi) = arma::reverse(Y.subvec(t-tmpi,t-1));
            }

            Ft = Fphi % Fy; // L(p) x 1
            hpsi = psi2hpsi(theta,GainCode,ctanh); // hpsi: p x N
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
            if (obstype == 0) {
                /*
                Negative-binomial likelihood
                - mean: lambda.at(i)
                - delta_nb: degree of over-dispersion

                sample variance exceeds the sample mean
                */
                w.at(i) = std::exp(R::lgammafn(Y.at(t)+delta_nb) - R::lgammafn(Y.at(t)+1.) - R::lgammafn(delta_nb) + delta_nb*(std::log(delta_nb)-std::log(delta_nb+lambda.at(i))) + Y.at(t)*(std::log(lambda.at(i))-std::log(delta_nb+lambda.at(i))));
                
            } else if (obstype == 1) {
                /*
                Poisson likelihood
                - mean: lambda.at(i)
                - var: lambda.at(i)

                sample variance == sample mean
                */
                w.at(i) = R::dpois(Y.at(t),lambda.at(i),false);
            }
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

    if (debug) {
        output["Meff"] = Rcpp::wrap(Meff);
        output["resample_status"] = Rcpp::wrap(resample_status);
    }
    

    return output;
}




void mcs_poisson(
    arma::vec& R, // (n+1) x 1
    const arma::vec& Y, // n x 1, the observed response
    const unsigned int ModelCode,
	const double W,
    const double rho = 0.9,
    const double alpha = 1.,
    const unsigned int L = 12, // number of lags
    const double mu0 = 0.,
    const unsigned int B = 12, // length of the B-lag fixed-lag smoother (Anderson and Moore 1979; Kitagawa and Sato)
	const unsigned int N = 5000, // number of particles
    const Rcpp::Nullable<Rcpp::NumericVector>& m0_prior = R_NilValue,
	const Rcpp::Nullable<Rcpp::NumericMatrix>& C0_prior = R_NilValue,
    const Rcpp::NumericVector& ctanh = Rcpp::NumericVector::create(0.3,0,1),
    const double delta_nb = 1.,
    const unsigned int obstype = 1){ // 0: negative binomial DLM; 1: poisson DLM
    
    const double UPBND = 700.;
    const double EPS = arma::datum::eps;

    unsigned int tmpi; // store temporary integer value
    const unsigned int n = Y.n_elem; // number of observations
    const double Wsqrt = std::sqrt(W);


    /* 
    Dimension of state space depends on type of transfer functions 
    - p: diemsnion of DLM state space
    */
    const unsigned int GainCode = get_gaincode(ModelCode);
    unsigned int TransferCode; // integer indicator for the type of transfer function
	unsigned int p; // dimension of DLM state space
	unsigned int L_;
	get_transcode(TransferCode,p,L_,ModelCode,L);
    /* Dimension of state space depends on type of transfer functions */

    
    /*
    Ft: vector for the state-to-observation function
    Gt: matrix for the state-to-state function
    */
    arma::vec Ft(p,arma::fill::zeros);
    arma::vec Fphi(p);
    arma::vec Fy(p,arma::fill::zeros);
    arma::mat Gt(p,p,arma::fill::zeros);
    switch (TransferCode) {
        case 0: // Koyck
        {
            Ft.at(1) = 1.; // Ft = (0,1)'
        }
        break;
        case 1: // Koyama
        {
            double mu = 2.2204e-16;
		    double m = 4.7;
		    double s = 2.9;
		    Fphi = get_Fphi(p,mu,m,s);
            Fphi = arma::pow(Fphi,alpha);

            Gt.at(0,0) = 1.;
            Gt.diag(-1).ones();

            // Check Fphi and G - CORRECT
        }
        break;
        case 2: // Solow
        {
            Ft.at(1) = 1.; // Ft = (0,1,0)
        }
        break;
        case 3: // Vanilla
        {
            Ft.at(0) = 1.;
        }
        break;
        default:
        {
            ::Rf_error("Not supported transfer function.");
        }
    } // End switch Transfercode


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
    if (ModelCode==6||ModelCode==7||ModelCode==8||ModelCode==9) {
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
    const double c1 = std::pow((1.-rho)*(1.-rho),alpha);
    const double c2 = 2.*rho;
    const double c3 = -rho*rho;

    for (unsigned int t=0; t<n; t++) {
        R_CheckUserInterrupt();
        // theta_stored: p x N x B, with B is the lag of the B-lag fixed-lag smoother
        // theta: p x N
        /*
        ------ Step 2.1 Propagate ------
        Propagate theta[t,i] to theta[t+1,i] for i=0,1,...,N
        using state evolution distribution P(theta[t+1,i]|theta[t,i]).
        ------
        >> Step 2-1 and 2-2 of Kitagawa and Sato - This is also the state equation
        */
        omega = arma::randn(N) * Wsqrt;
        if (TransferCode != 1) {
            hpsi = psi2hpsi(theta_stored.slice(B-1).row(0),GainCode,ctanh); // hpsi: 1 x N
        }
        
        switch (TransferCode) {
            case 0: // Koyck
            {
                theta.row(0) = theta_stored.slice(B-1).row(0) + omega.t();
                theta.row(1) = Y.at(t)*hpsi.row(0) + rho*theta_stored.slice(B-1).row(1);
            }
            break;
            case 1: // Koyama
            {
                for (unsigned int i=0; i<N; i++) {
                    theta.col(i) = Gt * theta_stored.slice(B-1).col(i);
                }
                theta.row(0) += omega.t();
            }
            break;
            case 2: // Solow
            {
                theta.row(0) = theta_stored.slice(B-1).row(0) + omega.t();
                theta.row(2) = theta_stored.slice(B-1).row(1);
                theta.row(1) = c1*Y.at(t)*hpsi.row(0) + c2*theta_stored.slice(B-1).row(1) + c3*theta_stored.slice(B-1).row(2);
            }
            break;
            case 3: // Vanilla
            {
                ::Rf_error("VanillaPois undefined.");
            }
            break;
            default:
            {
                ::Rf_error("Unknown type of model.");
            }
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
        if (ModelCode==6||ModelCode==7||ModelCode==8||ModelCode==9) {
            // Exponential link and identity gain
            for (unsigned int i=0; i<N; i++) {
                lambda.at(i) = mu0 + arma::as_scalar(Ft.t()*theta.col(i));
            }
            lambda.elem(arma::find(lambda>UPBND)).fill(UPBND);
            lambda = arma::exp(lambda);
        } else if (TransferCode==1){ // Koyama
            Fy.zeros();
            tmpi = std::min(t,p);
            if (t>0) {
                // Checked Fy - CORRECT
                Fy.head(tmpi) = arma::reverse(Y.subvec(t-tmpi,t-1));
            }

            Ft = Fphi % Fy; // L(p) x 1
            hpsi = psi2hpsi(theta,GainCode,ctanh); // hpsi: p x N
            for (unsigned int i=0; i<N; i++) {
                lambda.at(i) = mu0 + arma::as_scalar(Ft.t()*hpsi.col(i));
            }
        } else {
            // Koyck or Solow with identity link and different gain functions
            lambda = mu0 + theta.row(1).t();
        }


        for (unsigned int i=0; i<N; i++) {
            if (obstype == 0) {
                /*
                Negative-binomial likelihood
                - mean: lambda.at(i)
                - delta_nb: degree of over-dispersion

                sample variance exceeds the sample mean
                */
                w.at(i) = std::exp(R::lgammafn(Y.at(t)+delta_nb) - R::lgammafn(Y.at(t)+1.) - R::lgammafn(delta_nb) + delta_nb*(std::log(delta_nb)-std::log(delta_nb+lambda.at(i))) + Y.at(t)*(std::log(lambda.at(i))-std::log(delta_nb+lambda.at(i))));
                
            } else if (obstype == 1) {
                /*
                Poisson likelihood
                - mean: lambda.at(i)
                - var: lambda.at(i)

                sample variance == sample mean
                */
                w.at(i) = R::dpois(Y.at(t),lambda.at(i),false);
            }
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
       R.at(t) = arma::median(theta_stored.slice(0).row(0));
    }

    /*
    ------ R: an n x 3 matrix ------
        - The first B-2 rows (indices from 0 to B-1) of R are all zeros
        - Shift the the last (B-2):(n-1) rows (n-B+2 in totals) of R to 0:n-B+1
        - For the last B-1 rows (indices from n-B+2 to n), takes values from theta_stored.slice(1,B-1)
    */
    R.subvec(0,n-B+1) = R.subvec(B-2,n-1);
    for (unsigned int b=0; b<(B-1); b++) {
        R.at(n-B+2+b) = arma::median(theta_stored.slice(b+1).row(0));
    }

    return;
}



/*
---- Method ----
- Auxiliary Particle filtering with all static parameters (rho,W) known
- Using the DLM formulation

---- Model ----
<obs>   y[t] ~ Pois(lambda[t])
<link>  lambda[t] = theta[t]
<state> theta[t] = 2*rho*theta[t-1] - rho^2*theta[t-2] + (1-rho)^2*x[t-1]*exp(beta[t-1])
        beta[t] = beta[t-1] + omega[t]
        omega[t] ~ Normal(0,W)

Unknown parameters: theta[0:n], beta[0:n]
Kwg: Identity link, exponential state space
*/
Rcpp::List apf_pois_solow_eye_exp(
    const arma::vec& Y, // n x 1, the observed response
	const arma::vec& X, // n x 1, the exogeneous variable in the transfer function
    const double rho,
	const double Q,
	const unsigned int N = 5000, // number of particles
    const unsigned int B = 30) { // Step of particle backshifting

    const unsigned int n = Y.n_elem; // number of observations

    const double coef = (1.-rho)*(1.-rho);
    const double rho2 = rho*rho;

    arma::mat theta(3,N,arma::fill::zeros);
    arma::mat theta_new(3,N,arma::fill::zeros);
    arma::vec theta_next(N);
    arma::mat theta_stored(N,n);
    arma::mat beta_stored(N,n);

    Rcpp::IntegerVector idx_(N);
    arma::uvec idx(N);
    arma::uvec b = {2};
    arma::vec vec_tmp(N);
    arma::vec w(N);

    for (unsigned int t=0; t<n; t++) {
        theta = theta_new;
        /* 
        3. <Resample> theta[t,i] for i=0,1,...,N 
        using likelihood P(y[t]|theta[t,i]) 
        */
        for (unsigned int i=0; i<N; i++) {
            w.at(i) = R::dpois(Y.at(t),theta.at(0,i),true);
        }
        w.elem(arma::find(w>100.)).fill(100.);
        // w += arma::datum::eps;
        w = arma::exp(w);
        w /= arma::accu(w);

        // Resample
        idx_ = Rcpp::sample(N,N,true,Rcpp::as<Rcpp::NumericVector>(Rcpp::wrap(w)));
        idx = Rcpp::as<arma::uvec>(idx_) - 1;
        theta = theta.cols(idx);


        /*
        1. <Resample> again using predictive distribution
        P(y[t+1]|theta[t,i])
        Can be skipped for particle filtering 
        but an essential step for particle learning
        */
        if (t<(n-1)) {
            theta_next = 2.*rho*theta.row(0).t() - rho2*theta.row(1).t() + coef*X.at(t)*arma::exp(theta.row(2).t());
            for (unsigned int i=0; i<N; i++) {
                /* Use of Importance Function in Auxiliary Particle Filtering
                ---
                theta_next is the best guess of theta[t+1,i]
                ---
                */
                w.at(i) = R::dpois(Y.at(t+1),theta_next.at(i),true);
            }
            w.elem(arma::find(w>100.)).fill(100.);
            // w += arma::datum::eps;
            w = arma::exp(w);
            w /= arma::accu(w);

            idx_ = Rcpp::sample(N,N,true,Rcpp::as<Rcpp::NumericVector>(Rcpp::wrap(w)));
            idx = Rcpp::as<arma::uvec>(idx_) - 1;
            theta = theta.cols(idx);
        }


        theta_stored.col(t) = theta.row(0).t();
        beta_stored.col(t) = theta.row(2).t();


        /* 
        2. Propagate theta[t,i] to theta[t+1,i] for i=0,1,...,N 
        using state evolution distribution P(theta[t+1,i]|theta[t,i])
        */
        for (unsigned int i=0; i<N; i++) {
            w.at(i) = R::rnorm(0,std::sqrt(Q));
        }
        theta_new.row(0) = 2.*rho*theta.row(0) - rho2*theta.row(1) + coef*X.at(t)*arma::exp(theta.row(2));
        theta_new.row(1) = theta.row(0);
        theta_new.row(2) = theta.row(2) + w.t();

        b.at(0) = 0;
        arma::uvec tmpp = arma::find(theta_new.row(0)<arma::datum::eps);
        theta_new(b,tmpp).zeros();
        theta_new(b,tmpp) += arma::datum::eps;


    }

    Rcpp::List output;
    output["theta"] = Rcpp::wrap(theta_stored);
    output["beta"] = Rcpp::wrap(beta_stored);

    return output;
}



/*
---- Method ----
- Storvik's Filtering with rho known
- Using the DLM formulation
- Storvik's Filter can be viewed as an extension of 
  bootstrap filter with inference of static parameter W

---- Model ----
<obs>   y[t] ~ Pois(lambda[t])
<link>  lambda[t] = theta[t]
<state> theta[t] = 2*rho*theta[t-1] - rho^2*theta[t-2] + (1-rho)^2*x[t-1]*max(beta[t-1],0)
        beta[t] = beta[t-1] + omega[t]
        omega[t] ~ Normal(0,W)

Unknown parameters: theta[0:n], beta[0:n], W
Kwg: Identity link, max(beta,0) state space
*/
Rcpp::List sf_pois_solow_eye_max(
    const arma::vec& Y, // n x 1, the observed response
	const arma::vec& X, // n x 1, the exogeneous variable in the transfer function
    const double rho,
	const Rcpp::Nullable<Rcpp::NumericVector>& QPrior = R_NilValue, // (aw, Rw), W ~ IG(aw/2, Rw/2), prior for v[1], the first state/evolution/state error/disturbance.
	const double Q_init = NA_REAL,
    const double Q_true = NA_REAL,
	const unsigned int N = 5000, // number of particles
    const unsigned int B = 30) { // Step of particle backshifting

    const unsigned int n = Y.n_elem; // number of observations

    const double coef = (1.-rho)*(1.-rho);
    const double rho2 = rho*rho;

    arma::mat theta(3,N,arma::fill::zeros);
    arma::mat theta_new(3,N,arma::fill::zeros);
    arma::mat theta_stored(N,n);
    arma::mat beta_stored(N,n);

    arma::vec aw(N);
    arma::vec bw(N);
	arma::vec Q(N);

    Rcpp::IntegerVector idx_(N);
    arma::uvec idx(N);
    arma::uvec b = {2};
    arma::vec vec_tmp(N);
    arma::vec w(N);

    /*
    --- Initialization ---
    */
    bool Qflag;
    arma::vec QPrior_;
	if (R_IsNA(Q_true)) {
		Qflag = true;
		if (QPrior.isNull()) {
			stop("Error: You must provide either true value or prior for Q.");
		}
		arma::vec QPrior_ = Rcpp::as<arma::vec>(QPrior);

        /* --- Sufficient Statistics --- */
		aw.fill(QPrior_.at(0));
		bw.fill(QPrior_.at(1));
        /* --- Sufficient Statistics --- */

        /* --- Static Parameters --- */
		if (R_IsNA(Q_init)) {
            for (unsigned int i=0; i<N; i++) {
                Q.at(i) = 1./R::rgamma(0.5*aw.at(i),2./bw.at(i));
            }
		} else {
			Q.fill(Q_init);
		}
        /* --- Static Parameters --- */
	} else {
		Qflag = false;
		Q.fill(Q_true);
	}

    for (unsigned int t=0; t<n; t++) {
        theta = theta_new;
        /* 
        1. Resample theta[t,i] for i=0,1,...,N 
        using likelihood P(y[t]|theta[t,i]) 
        */
        for (unsigned int i=0; i<N; i++) {
            w.at(i) = R::dpois(Y.at(t),theta.at(0,i),true);
        }
        if (w.has_inf()|| w.has_nan()) {
            Rcout << "weights: ";
            Rcout << w.t() << std::endl;
            Rcout << "theta: ";
            Rcout << theta.row(0) << std::endl;
            Rcout << "t = " << t+1 << std::endl;
            stop("w has NaN or infinite values.");
        }
        w.elem(arma::find(w>100.)).fill(100.);
        // w += arma::datum::eps;
        w = arma::exp(w);
        w /= arma::accu(w);

        // Resample
        idx_ = Rcpp::sample(N,N,true,Rcpp::as<Rcpp::NumericVector>(Rcpp::wrap(w)));
        idx = Rcpp::as<arma::uvec>(idx_) - 1;
        theta = theta.cols(idx);
        Q = Q.elem(idx);
        aw = aw.elem(idx);
        bw = bw.elem(idx);

        theta_stored.col(t) = theta.row(0).t();
        beta_stored.col(t) = theta.row(2).t();

        
        if (Qflag) {
            if (t>0) {
                aw += 1.;
                bw += arma::pow((beta_stored.col(t) - beta_stored.col(t-1)),2.);
            }

            if (bw.has_inf() || bw.has_nan()) {
                Rcout << "bw: ";
                Rcout << bw.t() << std::endl;
                Rcout << "t = " << t+1 << std::endl;
                stop("bw has NaN or infinite values.");
            }
            
            for (unsigned int i=0; i<N; i++) {
                Q.at(i) = R::rgamma(0.5*aw.at(i),2./bw.at(i));
            }

            Q.elem(arma::find(Q<arma::datum::eps)).fill(arma::datum::eps);
            Q = 1./Q;

            if (Q.has_inf() || Q.has_nan()) {
                Rcout << "Q: ";
                Rcout << Q.t() << std::endl;
                Rcout << "t = " << t+1 << std::endl;
                stop("Q has NaN or infinite values.");
            }
        }

        

        /* 
        2. Propagate theta[t,i] to theta[t+1,i] for i=0,1,...,N 
        using predictive distribution P(y[t+1]|theta[t,i])
        */
        for (unsigned int i=0; i<N; i++) {
            w.at(i) = R::rnorm(0,std::sqrt(Q.at(i)));
        }
        if (w.has_inf() || w.has_nan()) {
            Rcout << "w: ";
            Rcout << w.t() << std::endl;
            Rcout << "t = " << t+1 << std::endl;
            stop("werr has NaN or infinite values.");
        }
        theta_new.row(1) = theta.row(0);
        theta_new.row(2) = theta.row(2) + w.t();
        b.at(0) = 2;
        {
            arma::uvec tmpp = arma::find(theta_new.row(2)<arma::datum::eps);
            theta_new(b,tmpp).zeros();
            theta_new(b,tmpp) += arma::datum::eps;
        }

        theta_new.row(0) = 2.*rho*theta.row(0) - rho2*theta.row(1) + coef*X.at(t)*theta.row(2);
        if (theta_new.row(0).has_inf()||theta_new.row(0).has_nan()) {
            Rcout << "theta[t]: ";
            Rcout << theta.row(0) << std::endl;
            Rcout << "t = " << t+1 << std::endl;
            stop("theta[t] has NaN or infinite values.");
        }
        if (theta_new.row(1).has_inf()||theta_new.row(1).has_nan()) {
            Rcout << "theta[t-1]: ";
            Rcout << theta.row(1) << std::endl;
            Rcout << "t = " << t+1 << std::endl;
            stop("theta[t-1] has NaN or infinite values.");
        }
        if (theta_new.row(2).has_inf()||theta_new.row(2).has_nan()) {
            Rcout << "beta[t]: ";
            Rcout << theta.row(2) << std::endl;
            Rcout << "t = " << t+1 << std::endl;
            stop("beta[t] has NaN or infinite values.");
        }
        b.at(0) = 0;
        {
            arma::uvec tmpp = arma::find(theta_new.row(0)<arma::datum::eps);
            theta_new(b,tmpp).zeros();
            theta_new(b,tmpp) += arma::datum::eps;
        }
    }

    Rcpp::List output;
    output["theta"] = Rcpp::wrap(theta_stored);
    output["beta"] = Rcpp::wrap(beta_stored);
    output["Q"] = Rcpp::wrap(Q);

    return output;
}

