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
- Bootstrap filtering with static parameter W known
- Using the DLM formulation

---- Model ----
<obs>   y[t] ~ Pois(lambda[t])
<link>  lambda[t] = phi[1]*y[t-1]*max(psi[t],0) + ... + phi[L]*y[t-L]*max(psi[t-L+1],0)
<state> psi[t] = psi[t-1] + omega[t]
        omega[t] ~ Normal(0,W)

Unknown parameters: psi[1:n]
Known parameters: W, phi[1:L]
Kwg: Identity link, max(psi,0) state space
*/
//' @export
// [[Rcpp::export]]
arma::mat bf_pois_koyama_max(
    const arma::vec& Y, // n x 1, the observed response
	const double W,
	const unsigned int N = 5000, // number of particles
    const unsigned int L = 12, // lags
    const unsigned int B = 12, // step of backshift
    const double rho = 34.08792, // parameter for negative binomial likelihood
    const unsigned int obstype = 1){ // 0: negative binomial DLM; 1: poisson DLM

    double tmpd; // store temporary double value
    unsigned int tmpi; // store temporary integer value
    arma::vec tmpv1(1);
    
    const unsigned int n = Y.n_elem; // number of observations

    const double mu = arma::datum::eps;
    const double m = 4.7;
    const double s = 2.9;
    const double sm2 = std::pow(s/m,2);
    const double pk_mu = std::log(m/std::sqrt(1.+sm2));
    const double pk_sg2 = std::log(1.+sm2);
    const double pk_2sg = std::sqrt(2.*pk_sg2);

    const double Wsqrt = std::sqrt(W);

    arma::mat G(L,L,arma::fill::zeros);
    arma::vec Fphi(L,arma::fill::zeros);
    G.at(0,0) = 1.;
    tmpv1.at(0) = pk_mu/pk_2sg;
    Fphi.at(0) = 0.5*arma::as_scalar(arma::erfc(tmpv1));
    for (unsigned int d=1; d<L; d++) {
        G.at(d,d-1) = 1.;
        tmpd = static_cast<double>(d) + 1.;
        tmpv1.at(0) = -(std::log(tmpd)-pk_mu)/pk_2sg;
        Fphi.at(d) = 0.5*arma::as_scalar(arma::erfc(tmpv1));
        tmpd -= 1.;
        tmpv1.at(0) = -(std::log(tmpd)-pk_mu)/pk_2sg;
        Fphi.at(d) -= 0.5*arma::as_scalar(arma::erfc(tmpv1));
    }

    // Check Fphi and G - CORRECT

    arma::vec Fy(L,arma::fill::zeros);
    arma::vec F(L);

    arma::cube theta_stored(L,N,B);
    arma::mat theta = arma::randu(L,N,arma::distr_param(0.,10.));

    arma::vec lambda(N);
    arma::vec omega(N);
    arma::vec w(N); // weight of each particle

    arma::mat R(n,3);
    arma::vec qProb = {0.025,0.5,0.975};

    Rcpp::IntegerVector idx_(N);
    arma::uvec idx(N);

    for (unsigned int t=0; t<n; t++) {
        theta_stored.slice(B-1) = theta;

        Fy.zeros();
        tmpi = std::min(t,L);
        if (t>0) {
            // Checked Fy - CORRECT
            Fy.head(tmpi) = arma::reverse(Y.subvec(t-tmpi,t-1));
        }
        F = Fphi % Fy; // L x 1

        /*
        1. Resample lambda[t,i] for i=0,1,...,N, where i is the index of particles
        using likelihood P(y[t]|lambda[t,i])
        */
        for (unsigned int i=0; i<N; i++) {
            theta.elem(arma::find(theta<arma::datum::eps)).fill(arma::datum::eps);
            lambda.at(i) = mu + arma::as_scalar(F.t()*theta.col(i));

            if (obstype == 0) {
                w.at(i) = std::exp(R::lgammafn(Y.at(t)+(lambda.at(i)/rho))-R::lgammafn(Y.at(t)+1.)-R::lgammafn(lambda.at(i)/rho)+(lambda.at(i)/rho)*std::log(1./(1.+rho))+Y.at(t)*std::log(rho/(1.+rho)));
            } else if (obstype == 1) {
                w.at(i) = R::dpois(Y.at(t),lambda.at(i),false);
            }
            
        }

        idx_ = Rcpp::sample(N,N,true,Rcpp::as<Rcpp::NumericVector>(Rcpp::wrap(w)));
        idx = Rcpp::as<arma::uvec>(idx_) - 1;
        for (unsigned int b=0; b<B; b++) {
            theta_stored.slice(b) = theta_stored.slice(b).cols(idx);
        }
        // theta_stored = theta_stored(arma::span::all,idx,arma::span::all);
        R.row(t) = arma::quantile(theta_stored.slice(0).row(0),qProb);
        // theta = theta.cols(idx);

        /*
        2. Propagate theta[t,i] to theta[t+1,i] for i=0,1,...,N
        using state evolution distribution P(theta[t+1,i]|theta[t,i])
        */
        omega = arma::randn(N) * Wsqrt;
        for (unsigned int i=0; i<N; i++) {
            theta.col(i) = G * theta_stored.slice(B-1).col(i);
        }
        theta.row(0) += omega.t();
        // theta.elem(arma::find(theta<mu)).fill(mu);
        theta_stored.slices(0,B-2) = theta_stored.slices(1,B-1);
    }

    /*
    Up to now, the first L-1 rows of R will be zeros. We need to
    (1) Shift the nonzero elements to the beginning of R
    (2) calculate the quantiles the first L-1 rows of theta
    */
    R.rows(0,n-L) = R.rows(L-1,n-1);
    for (unsigned int d=0; d<(L-1); d++) {
        R.row(n-L+1+d) = arma::quantile(theta_stored.slice(d).row(0),qProb);
    }

    return R;
}




/*
---- Method ----
- Monte Carlo filtering with static parameter W known
- Using the discretized Hawkes formulation
- Reference: Kitagawa and Sato at Ch 9.3.4 of Doucet et al.
- Note: 
    1. Initial version is copied from the `bf_pois_koyama_exp`
    2. This is intended to be the modified Rcpp version of `hawkes_state_space.R`
    3. The difference is that the R version the states are backshifted L times, 
        where L is the maximum transmission delay to be considered.
        To make this a Monte Carlo filtering, we don't backshifting/smoot.

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
arma::mat mcf_pois_koyama_exp(
    const arma::vec& Y, // n x 1, the observed response
	const double W,
    const unsigned int L = 12, // number of lags
	const unsigned int N = 5000, // number of particles
    const double rho = 34.08792, // parameter for negative binomial likelihood
    const unsigned int obstype = 0){ // 0: negative binomial DLM; 1: poisson DLM

    double tmpd; // store temporary double value
    unsigned int tmpi; // store temporary integer value
    arma::vec tmpv1(1);
    
    const unsigned int n = Y.n_elem; // number of observations

    const double mu = arma::datum::eps;
    const double m = 4.7;
    const double s = 2.9;
    const double sm2 = std::pow(s/m,2);
    const double pk_mu = std::log(m/std::sqrt(1.+sm2));
    const double pk_sg2 = std::log(1.+sm2);
    const double pk_2sg = std::sqrt(2.*pk_sg2);

    const double Wsqrt = std::sqrt(W);

    arma::mat G(L,L,arma::fill::zeros); // L x L state transition matrix
    arma::vec Fphi(L,arma::fill::zeros); // L x 1 transmission delay distribution
    G.at(0,0) = 1.;
    tmpv1.at(0) = pk_mu/pk_2sg;
    Fphi.at(0) = 0.5*arma::as_scalar(arma::erfc(tmpv1));
    for (unsigned int d=1; d<L; d++) {
        G.at(d,d-1) = 1.;
        tmpd = static_cast<double>(d) + 1.;
        tmpv1.at(0) = -(std::log(tmpd)-pk_mu)/pk_2sg;
        Fphi.at(d) = 0.5*arma::as_scalar(arma::erfc(tmpv1));
        tmpd -= 1.;
        tmpv1.at(0) = -(std::log(tmpd)-pk_mu)/pk_2sg;
        Fphi.at(d) -= 0.5*arma::as_scalar(arma::erfc(tmpv1));
    }

    // Check Fphi and G - CORRECT
    // Rcout << Fphi.t() << std::endl; 
    // Rcout << G << std::endl;

    arma::vec Fy(L,arma::fill::zeros);
    arma::vec F(L);

    arma::cube theta_stored(L,N,n+1);
    arma::mat theta = arma::randu(L,N,arma::distr_param(0.,10.));
    
    arma::vec lambda(N);
    arma::vec omega(N);
    arma::vec w(N); // weight of each particle

    arma::mat R(n,3);
    arma::vec qProb = {0.025,0.5,0.975};

    Rcpp::IntegerVector idx_(N);
    arma::uvec idx(N);

    for (unsigned int t=0; t<=n; t++) {
        // theta_stored: L x N x B, with B = n
        // theta: L x N
        theta_stored.slice(t) = theta;

        Fy.zeros();
        tmpi = std::min(t,L);
        if (t>0) {
            // Checked Fy - CORRECT
            Fy.head(tmpi) = arma::reverse(Y.subvec(t-tmpi,t-1));
        }
        F = Fphi % Fy; // L x 1

        if (t>0) {
            /*
            ------ RESAMPLING STAGE ------
            1. Resample lambda[t,i] for i=0,1,...,N, where i is the index of particles
            using likelihood P(y[t]|lambda[t,i]).
            ------
            >> Step 2-3 and 2-4 of Kitagawa and Sato - This is also the observational equation (nonlinear, nongaussian)
            */
            for (unsigned int i=0; i<N; i++) {
                /*
                ------ RESAMPLING STAGE ------
                ------ Step 2-3 of Kitagawa and Sato ------
                Namely, Compute the weights using likelihood/observational distribution, 
                p(y[t]|lambda[t,i]), 
                where lambda[t,i] is the i-th particle for lambda[t].
                */ 
                lambda.at(i) = mu + arma::as_scalar(F.t()*arma::exp(theta.col(i)));
                if (obstype == 0) {
                    w.at(i) = std::exp(R::lgammafn(Y.at(t)+(lambda.at(i)/rho))-R::lgammafn(Y.at(t)+1.)-R::lgammafn(lambda.at(i)/rho)+(lambda.at(i)/rho)*std::log(1./(1.+rho))+Y.at(t)*std::log(rho/(1.+rho)));
                } else if (obstype == 1) {
                    w.at(i) = R::dpois(Y.at(t),lambda.at(i),false);
                }   
            }

            /*
            ------ RESAMPLING STAGE (FILTERING) ------
            Generate {lambda[1,i],...,lambda[t-1,i],lambda[t,i] | D[t]} for i = 1,...,N
            by resampling {lambda[i,i],...,lambda[t-1,i],lambda[t,i] | D[t-1]} 
            for i = 1,...,N, using the weights w[t,i] = p(y[t]|lambda]t,i).
            */
            idx_ = Rcpp::sample(N,N,true,Rcpp::as<Rcpp::NumericVector>(Rcpp::wrap(w)));
            idx = Rcpp::as<arma::uvec>(idx_) - 1;
            theta_stored.slice(t) = theta_stored.slice(t).cols(idx);
            R.row(t-1) = arma::quantile(theta_stored.slice(t).row(0),qProb);
        }
        

        /*
        2. Propagate theta[t,i] to theta[t+1,i] for i=0,1,...,N
        using state evolution distribution P(theta[t+1,i]|theta[t,i]).
        ------
        >> Step 2-1 and 2-2 of Kitagawa and Sato - This is also the state equation
        */

        /*
        ------ Step 2-1 of Kitagawa and Sato ------
        Generate N random error following the error distribution
        */
        omega = arma::randn(N) * Wsqrt;

        /*
        ------ Step 2-2 of Kitagawa and Sato ------
        Propagate from t to t+1 using the evolution distribution
        */
        for (unsigned int i=0; i<N; i++) {
            theta.col(i) = G * theta_stored.slice(t).col(i);
        }
        theta.row(0) += omega.t();
        // theta_stored.slices(0,B-2) = theta_stored.slices(1,B-1);
    }

    /*
    Up to now, the first L-1 rows of R will be zeros. We need to
    (1) Shift the nonzero elements to the beginning of R
    (2) calculate the quantiles the first L-1 rows of theta
    */
    return R;
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
    const unsigned int L = 12, // number of lags
    const double mu0 = 0.,
    const unsigned int B = 12, // length of the B-lag fixed-lag smoother (Anderson and Moore 1979; Kitagawa and Sato)
	const unsigned int N = 5000, // number of particles
    const Rcpp::Nullable<Rcpp::NumericVector>& m0_prior = R_NilValue,
	const Rcpp::Nullable<Rcpp::NumericMatrix>& C0_prior = R_NilValue,
    const Rcpp::Nullable<Rcpp::NumericVector>& qProb_ = R_NilValue,
    const double rho_nb = 34.08792, // parameter for negative binomial likelihood
    const double delta_nb = 1.,
    const unsigned int obstype = 1, // 0: negative binomial DLM; 1: poisson DLM
    const bool verbose = false,
    const bool debug = false){ 

    const double UPBND = 700.;
    const double EPS = arma::datum::eps;

    double tmpd; // store temporary double value
    unsigned int tmpi; // store temporary integer value
    arma::vec tmpv1(1);
    
    const unsigned int n = Y.n_elem; // number of observations
    const double Wsqrt = std::sqrt(W);

    if (verbose) {
        Rcout << "Evolution variance W=" << W << std::endl;
    }


    /* 
    Dimension of state space depends on type of transfer functions 
    - p: diemsnion of DLM state space
    */
    const bool is_solow = ModelCode == 2 || ModelCode == 3 || ModelCode == 7 || ModelCode == 12;
	const bool is_koyck = ModelCode == 4 || ModelCode == 5 || ModelCode == 8 || ModelCode == 10;
	const bool is_koyama = ModelCode == 0 || ModelCode == 1 || ModelCode == 6 || ModelCode == 11;
	const bool is_vanilla = ModelCode == 9;
	unsigned int TransferCode; // integer indicator for the type of transfer function
	unsigned int p; // dimension of DLM state space
	unsigned int L_;
	if (is_koyck) { 
		TransferCode = 0; 
		p = 2;
		L_ = 0;
	} else if (is_koyama) { 
        if (L == 0) {
            ::Rf_error("Lag should be greater than 0 for Koyama.");
        }
		TransferCode = 1; 
		p = L;
		L_ = L;
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
    /* Dimension of state space depends on type of transfer functions */

    
    /*
    Ft: vector for the state-to-observation function
    Gt: matrix for the state-to-state function
    */
    arma::vec Ft(p,arma::fill::zeros);
    arma::vec Fphi(p);;
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
    arma::cube theta_stored(p,N,B);

    arma::vec lambda(N); // instantaneous intensity
    arma::vec omega(N); // evolution variance
    arma::vec w(N); // importance weight of each particle

    arma::vec qProb;
    if (qProb_.isNull()) {
        qProb = {0.025,0.5,0.975};
    } else {
        qProb = Rcpp::as<arma::vec>(qProb_);
    }

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

        double hpsi;
        if (TransferCode==1) {
            // Koyama
            for (unsigned int i=0; i<N; i++) {
                theta.col(i) = Gt * theta_stored.slice(B-1).col(i);
            }
            theta.row(0) += omega.t();

        } else {
            switch (ModelCode) {
                case 2: // SolowMax
                {
                    // Identity link + ramp gain
                    for (unsigned int i=0; i<N; i++) {
                        hpsi = std::max(EPS,theta_stored.at(0,i,B-1));
                        theta.at(0,i) = theta_stored.at(0,i,B-1) + omega.at(i);
                        theta.at(1,i) = std::max((1.-rho)*(1.-rho)*Y.at(t)*hpsi + 2.*rho*theta_stored.at(1,i,B-1) - rho*rho*theta_stored.at(2,i,B-1), -mu0+EPS);
                        theta.at(2,i) = theta_stored.at(1,i,B-1);
                    }
                }
                break;
                case 3: // SolowExp
                {
                    // Identity link + exponential gain
                    for (unsigned int i=0; i<N; i++) {
                        hpsi = std::exp(std::min(theta_stored.at(0,i,B-1),UPBND));
                        theta.at(0,i) = theta_stored.at(0,i,B-1) + omega.at(i);
                        theta.at(1,i) = std::max((1.-rho)*(1.-rho)*Y.at(t)*hpsi + 2.*rho*theta_stored.at(1,i,B-1) - rho*rho*theta_stored.at(2,i,B-1),-mu0+EPS);
                        theta.at(2,i) = theta_stored.at(1,i,B-1);
                    }
                }
                break;
                case 4: // KoyckMax
                {
                    // Identity link + ramp gain
                    for (unsigned int i=0; i<N; i++) {
                        hpsi = std::max(EPS,theta_stored.at(0,i,B-1));
                        theta.at(0,i) = theta_stored.at(0,i,B-1) + omega.at(i);
                        theta.at(1,i) = Y.at(t)*hpsi + rho*theta_stored.at(1,i,B-1);
                    }
                }
                break;
                case 5: // KoyckExp
                {
                    // Identity link + exponential gain
                    for (unsigned int i=0; i<N; i++) {
                        hpsi = std::exp(std::min(theta_stored.at(0,i,B-1),UPBND));
                        theta.at(0,i) = theta_stored.at(0,i,B-1) + omega.at(i);
                        theta.at(1,i) = Y.at(t)*hpsi + rho*theta_stored.at(1,i,B-1);
                    }
                }
                break;
                case 7: // SolowEye
                {
                    // Exponential link + identity gain
                    for (unsigned int i=0; i<N; i++) {
                        theta.at(0,i) = theta_stored.at(0,i,B-1) + omega.at(i);
                        theta.at(1,i) = (1.-rho)*(1.-rho)*Y.at(t)*theta_stored.at(0,i,B-1) + 2.*rho*theta_stored.at(1,i,B-1) - rho*rho*theta_stored.at(2,i,B-1);
                        theta.at(2,i) = theta_stored.at(1,i,B-1);
                    }
                }
                break;
                case 8: // KoyckEye
                {
                    // Exponential link + identity gain
                    for (unsigned int i=0; i<N; i++) {
                        theta.at(0,i) = theta_stored.at(0,i,B-1) + omega.at(i);
                        theta.at(1,i) = Y.at(t)*theta_stored.at(0,i,B-1) + rho*theta_stored.at(1,i,B-1);
                    }
                }
                break;
                case 9: // VanillaPois
                {
                    ::Rf_error("VanillaPois undefined.");
                }
                break;
                case 10: // KoyckSoftplus
                {
                    // Identity link + softplus gain
                    for (unsigned int i=0; i<N; i++) {
                        hpsi = std::log(1. + std::exp(std::min(theta_stored.at(0,i,B-1),UPBND)));
                        theta.at(0,i) = theta_stored.at(0,i,B-1) + omega.at(i);
                        theta.at(1,i) = Y.at(t)*hpsi + rho*theta_stored.at(1,i,B-1);
                    }
                }
                break;
                case 12: // SolowSoftplus
                {
                    for (unsigned int i=0; i<N; i++) {
                        hpsi = std::log(1. + std::exp(std::min(theta_stored.at(0,i,B-1),UPBND)));
                        theta.at(0,i) = theta_stored.at(0,i,B-1) + omega.at(i);
                        theta.at(1,i) = std::max((1.-rho)*(1.-rho)*Y.at(t)*hpsi + 2.*rho*theta_stored.at(1,i,B-1) - rho*rho*theta_stored.at(2,i,B-1),-mu0+EPS);
                        theta.at(2,i) = theta_stored.at(1,i,B-1);
                    }
                }
                break;
                default:
                {
                    ::Rf_error("Not supported model type.");
                }
            }
        }

        if (verbose) {
            Rcout << "quantiles for hpsi[" << t+1 << "]" << arma::quantile(theta.row(1),qProb);
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
        if (TransferCode == 1) { // Koyama
            Fy.zeros();
            tmpi = std::min(t,p);
            if (t>0) {
                // Checked Fy - CORRECT
                Fy.head(tmpi) = arma::reverse(Y.subvec(t-tmpi,t-1));
            }

            Ft = Fphi % Fy; // L(p) x 1
        }

        if (ModelCode==6||ModelCode==7||ModelCode==8||ModelCode==9) {
            // Exponential link and identity gain
            for (unsigned int i=0; i<N; i++) {
                lambda.at(i) = mu0 + arma::as_scalar(Ft.t()*theta.col(i));
            }
            lambda.elem(arma::find(lambda>UPBND)).fill(UPBND);
            lambda = arma::exp(lambda);
        } else if (ModelCode==0){
            // KoyamaMax with identity link and ramp gain
            theta.elem(arma::find(theta<EPS)).fill(EPS);
            for (unsigned int i=0; i<N; i++) {
                lambda.at(i) = mu0 + arma::as_scalar(Ft.t()*theta.col(i));
            }
        } else if (ModelCode==1) {
            // KoyamaExp with identity link and exponential gain
            theta.elem(arma::find(theta>UPBND)).fill(UPBND);
            for (unsigned int i=0; i<N; i++) {
                lambda.at(i) = mu0 + arma::as_scalar(Ft.t()*arma::exp(theta.col(i)));
            }
        } else if (ModelCode==11) {
            // KoyamaSoftplus with identity link and softplus gain
            theta.elem(arma::find(theta>UPBND)).fill(UPBND);
            for (unsigned int i=0; i<N; i++) {
                arma::vec htheta = arma::log(1. + arma::exp(theta.col(i)));
                lambda.at(i) = mu0 + arma::as_scalar(Ft.t()*htheta);
            }
        } else {
            // Koyck or Solow with identity link and different gain functions
            // for (unsigned int i=0; i<N; i++) {
            //     lambda.at(i) = mu0 + arma::as_scalar(Ft.t()*theta.col(i));
            // }
            lambda = mu0 + theta.row(1).t();
        }

        if (verbose) {
            Rcout << "Quantiles of lambda[" << t+1 << "]: " << arma::quantile(lambda.t(),qProb);
        }

        for (unsigned int i=0; i<N; i++) {
            if (obstype == 0) {
                /*
                Negative-binomial likelihood
                - mean: lambda.at(i)
                - delta_nb: degree of over-dispersion

                sample variance exceeds the sample mean
                */
                // w.at(i) = std::exp(R::lgammafn(Y.at(t)+(lambda.at(i)/rho_nb))-R::lgammafn(Y.at(t)+1.)-R::lgammafn(lambda.at(i)/rho_nb)+(lambda.at(i)/rho_nb)*std::log(1./(1.+rho_nb))+Y.at(t)*std::log(rho_nb/(1.+rho_nb)));

                // w.at(i) = std::exp(R::lgammafn(Y.at(t)+delta_nb)-R::lgammafn(Y.at(t)+1.)-R::lgammafn(delta_nb)+delta_nb*std::log(delta_nb/(delta_nb+lambda.at(i)))+Y.at(t)*std::log(lambda.at(i)/(delta_nb+lambda.at(i))));
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


        if (verbose) {
            Rcout << "Quantiles of importance weight w[" << t+1 << "]: " << arma::quantile(w.t(),qProb);
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



        R.row(t) = arma::quantile(theta_stored.slice(0).row(0),qProb);
    }

    /*
    ------ R: an n x 3 matrix ------
        - The first B-2 rows (indices from 0 to B-1) of R are all zeros
        - Shift the the last (B-2):(n-1) rows (n-B+2 in totals) of R to 0:n-B+1
        - For the last B-1 rows (indices from n-B+2 to n), takes values from theta_stored.slice(1,B-1)
    */
    R.rows(0,n-B+1) = R.rows(B-2,n-1);
    for (unsigned int b=0; b<(B-1); b++) {
        R.row(n-B+2+b) = arma::quantile(theta_stored.slice(b+1).row(0),qProb);
    }

    Rcpp::List output;
    output["quantiles"] = Rcpp::wrap(R); // (n+1) x 3
    output["theta_last"] = Rcpp::wrap(theta_stored.slice(B-1)); // p x N
    output["Meff"] = Rcpp::wrap(Meff);
    output["resample_status"] = Rcpp::wrap(resample_status);

    return output;
}



/*
---- Method ----
- Bootstrap filtering with all static parameters (rho,W) known
- Using the DLM formulation

---- Model ----
<obs>   y[t] ~ Pois(lambda[t])
<link>  lambda[t] = theta[t]
<state> theta[t] = 2*rho*theta[t-1] - rho^2*theta[t-2] + (1-rho)^2*x[t-1]*max(beta[t-1],0)
        beta[t] = beta[t-1] + omega[t]
        omega[t] ~ Normal(0,W)

Unknown parameters: theta[0:n], beta[0:n]
Kwg: Identity link, max(beta,0) state space
*/
Rcpp::List bf_pois_solow_eye_max(
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

        theta_stored.col(t) = theta.row(0).t();
        beta_stored.col(t) = theta.row(2).t();

        /* 
        2. Propagate theta[t,i] to theta[t+1,i] for i=0,1,...,N 
        using predictive distribution P(y[t+1]|theta[t,i])
        */
        for (unsigned int i=0; i<N; i++) {
            w.at(i) = R::rnorm(0,std::sqrt(Q));
        }
        theta_new.row(2) = theta.row(2) + w.t();
        b.at(0) = 2;
        {
            arma::uvec tmpp = arma::find(theta_new.row(2)<arma::datum::eps);
            theta_new(b,tmpp).zeros();
            theta_new(b,tmpp) += arma::datum::eps;
        }

        theta_new.row(0) = 2.*rho*theta.row(0) - rho2*theta.row(1) + coef*X.at(t)*theta.row(2);
        theta_new.row(1) = theta.row(0);
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

    return output;
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