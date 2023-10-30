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
    const arma::vec &Y, // nt x 1, the observed response
    const arma::uvec &model_code,
    const double W_true = NA_REAL, // Use discount factor if W is not given
    const double rho = 0.9,
    const unsigned int L = 12, // number of lags
    const double mu0 = 2.220446e-16,
    const unsigned int B = 12,                                        // length of the B-lag fixed-lag smoother (Anderson and Moore 1979; Kitagawa and Sato)
    const unsigned int N = 5000,                                      // number of particles
    const Rcpp::Nullable<Rcpp::NumericVector> &m0_prior = R_NilValue, // mean of normal prior for theta0
    const Rcpp::Nullable<Rcpp::NumericMatrix> &C0_prior = R_NilValue, // variance of normal prior for theta0
    const double theta0_upbnd = 2.,                                   // Upper bound of uniform prior for theta0
    const Rcpp::NumericVector &qProb = Rcpp::NumericVector::create(0.025, 0.5, 0.975),
    const double delta_nb = 1.,
    const double delta_discount = 0.95)
{
    const double alpha = 1.;
    const Rcpp::NumericVector ctanh = {1., 0., 1.}; // (1./M, 0, M)

    const unsigned int nt = Y.n_elem; // number of observations
    const double N_ = static_cast<double>(N);
    arma::vec ypad(nt + 1, arma::fill::zeros);
    ypad.tail(nt) = Y;


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

    /* Dimension of state space depends on type of transfer functions */

        
    
    arma::vec m0(p,arma::fill::zeros);
    arma::mat C0(p,p,arma::fill::eye); 
    C0 *= std::pow(theta0_upbnd*0.5,2.);
    if (!m0_prior.isNull())
    {
		m0 = Rcpp::as<arma::vec>(m0_prior);
	}

    C0 = arma::chol(C0);
    arma::mat theta(p, N, arma::fill::zeros);
    for (unsigned int i=0; i<N; i++) {
        theta.col(i) = m0 + C0.t() * arma::randn(p);
    }

    arma::cube theta_stored(p, N, nt + B);
    for (unsigned int b=0; b<B; b++) {
        theta_stored.slice(b).zeros();
    }
    theta_stored.slice(B - 1) = theta;

    arma::vec Wt(nt);
    if (!R_IsNA(W_true))
    {
        Wt.fill(W_true);
    }


    arma::vec Meff(nt, arma::fill::zeros); 
    // Effective sample size (Ref: Lin, 1996; Prado, 2021, page 196)
    arma::uvec resample_status(nt,arma::fill::zeros);
    arma::mat R(nt+1,3,arma::fill::zeros);


    for (unsigned int t=0; t<nt; t++) 
    {
        R_CheckUserInterrupt();

        /*
        ------ Step 2.1 Propagate ------
        */
        arma::mat Theta_old = theta_stored.slice(t+B-1);

        // if (trans_code != 1) {
        //     arma::vec mt;
        //     if (t > B)
        //     {
        //         // Is it necessary?
        //         mt = 0.5 * arma::mean(Theta_old, 1) + 0.5 * arma::median(Theta_old, 1);
        //         // mt: p x 1
        //     }
        //     else
        //     {
        //         mt = arma::median(Theta_old, 1);
        //     }
        //     update_Gt(Gt, gain_code, trans_code, mt, ctanh, 1., ypad.at(t), rho);
        // }

        // // theta_stored: p,N,B
        if (R_IsNA(W_true))
        { // Use discount factor if W is not given
            Wt.at(t) = arma::var(Theta_old.row(0));
            // Wsqrt = std::sqrt(Wt.at(t));
            if (t > B)
            {
                Wt.at(t) *= 1. / delta_discount - 1.;
            }
            else
            {
                Wt.at(t) *= 1. / 0.99 - 1.;
            }
        }
        double Wsqrt = std::sqrt(Wt.at(t));
        

        for (unsigned int i = 0; i < N; i++)
        {
            arma::vec theta_old = Theta_old.col(i);
            arma::vec theta_new = update_at(
                p,gain_code,trans_code,theta_old,Gt,ctanh,1.,ypad.at(t),rho
            );

            double omega_new = R::rnorm(0.,Wsqrt);
            theta_new.at(0) += omega_new;
            theta_stored.slice(t + B).col(i) = theta_new;
        }

        /*
        ------ Step 2.1 Propagate ------
        */
        
        

        /*
        ------ Step 2.2 Importance weights ------
        */
        if (trans_code == 1)
        { // Koyama
            arma::vec Fy(p, arma::fill::zeros);
            unsigned int tmpi = std::min(t, p);
            if (t > 0)
            {
                // Checked Fy - CORRECT
                Fy.head(tmpi) = arma::reverse(ypad.subvec(t + 1 - tmpi, t));
            }

            Ft = Fphi % Fy; // L(p) x 1
        }

        arma::vec weight(N);
        arma::vec lambda(N);
        for (unsigned int i = 0; i < N; i++)
        {
            arma::vec psi = theta_stored.slice(t + B).col(i); // p x 1

            // theta: p x N
            if (link_code == 1)
            {
                // Exponential link and identity gain
                double tmp = arma::as_scalar(Ft.t() * psi);
                lambda.at(i) = std::min(mu0 + tmp, UPBND);
                double tmp2 = std::exp(lambda.at(i));
                lambda.at(i) = tmp2;
            }
            else if (trans_code == 1)
            {
                // Koyama
                arma::vec hpsi = psi2hpsi(psi, gain_code, ctanh);
                lambda.at(i) = mu0 + arma::as_scalar(Ft.t() * hpsi);
            }
            else
            {
                // Koyck or Solow with identity link and different gain functions
                lambda.at(i) = mu0 + psi.at(1);
            }

            weight.at(i) = loglike_obs(
                ypad.at(t + 1), lambda.at(i), obs_code, delta_nb, false
            );
        }




        /*
        ------ Step 2.2 Importance weights ------
        */

        /*
        ------ Step 3 Resampling with Replacement ------
        */
        double wsum = arma::accu(weight);
        bool resample = false;
        if (wsum > EPS)
        {
            // normalize the particle weights
            arma::vec wtmp = weight;
            weight = wtmp / wsum;
            Meff.at(t) = 1. / arma::dot(weight, weight);
            if (Meff.at(t) > std::max(100., 0.1 * N_))
            {
                resample = true;
            }
            else
            {
                resample = false;
            }
        }
        else
        {
            resample = false;
            Meff.at(t) = 0.;
        }


        if (resample) 
        {

            Rcpp::NumericVector w_ = Rcpp::wrap(weight);
            Rcpp::IntegerVector idx_ = Rcpp::sample(N, N, true, w_);
            arma::uvec idx = Rcpp::as<arma::uvec>(idx_) - 1;

            // from t+1, to t+B
            for (unsigned int b = t+1; b < (t+B+1); b++)
            {
                arma::mat ttmp = theta_stored.slice(b);
                theta_stored.slice(b) = ttmp.cols(idx);
            }
        }
        weight.ones();
        resample_status.at(t) = resample;
    }

    Rcpp::List output;
    arma::mat psi_all = theta_stored.row(0); // N x (nt+B)
    arma::mat psi = psi_all.cols(B-1, nt+B-1);
    arma::mat RR = arma::quantile(psi, Rcpp::as<arma::vec>(qProb), 0);
    output["psi"] = Rcpp::wrap(RR.t());

    output["theta"] = Rcpp::wrap(theta_stored); // p x N
    output["Wt"] = Rcpp::wrap(Wt);

    output["Meff"] = Rcpp::wrap(Meff);
    output["resample_status"] = Rcpp::wrap(resample_status);
    
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
        w_stored.at(t+1) = arma::accu(w)/N_; // summation of the unormalized weight
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
    const double theta0_upbnd = 2.,
    const Rcpp::NumericVector& qProb = Rcpp::NumericVector::create(0.025,0.5,0.975),
    const Rcpp::NumericVector& ctanh = Rcpp::NumericVector::create(0.2,0,5.),
    const double delta_nb = 1.,
    const double delta_discount = 0.95,
    const unsigned int npara = 20,
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

        
    arma::mat theta(p,N,arma::fill::zeros);
    arma::cube theta_stored(p,N,n+1);
    arma::mat hpsi;
    // if (link_code==1) {
    //     // Exponential Link
        arma::vec m0(p,arma::fill::zeros);
        arma::mat C0(p,p,arma::fill::eye); C0 *= std::pow(theta0_upbnd*0.5,2.);
        if (!m0_prior.isNull()) {
		    m0 = Rcpp::as<arma::vec>(m0_prior);
	    }
	    // if (!C0_prior.isNull()) {
		//     C0 = Rcpp::as<arma::mat>(C0_prior);
            
	    // }
        C0 = arma::chol(C0);

        // theta.row(0) = m0.at(0) + arma::randn(1,N)*C0.at(0,0);
        // theta.row(1) = m0.at(1) + arma::randn(1,N)*C0.at(1,1);

        for (unsigned int i=0; i<N; i++) {
            theta.col(i) = m0 + C0.t() * arma::randn(p);
        }

    // } else {
    //     // Identity Link
    //     theta = arma::randu(p,N,arma::distr_param(0.,theta0_upbnd)); // Consider it as a flat prior
    // }
    theta_stored.slice(0) = theta;
    /*
    ------ Step 1. Initialization theta[0,] at time t = 0 ------
    */
    const double c1 = std::pow(1.-rho,static_cast<double>(L_)*alpha);
    double c2 = -rho;

    // const double c1_ = std::pow((1.-rho)*(1.-rho),alpha);
    // const double c2_ = 2.*rho;
    // const double c3_ = -rho*rho;

    arma::vec mt(p);
    arma::mat Ct(p,p);
    arma::vec Wt(n);
    arma::vec w_stored(n+1,arma::fill::zeros);

    
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
            Wt.at(t) =std::max(Wsqrt,EPS);
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
        w_stored.at(t+1) = arma::accu(w)/N_; // summation of the unormalized weight, marginal likelihood
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

        // if (verbose)
        {
            Rcout << "\rFiltering Progress: " << t + 1 << "/" << n;
        }
    }

    Rcout << std::endl;


    /*
    ------ Particle Smoothing ------
    */
    Rcpp::NumericVector tmpir(1);
    arma::mat psi_smooth(N,n+1);
    arma::mat psi_filter(N,n+1);
    psi_filter.col(n) = theta_stored.slice(n).row(0).t();
    for (unsigned int t=n; t>0; t--) {
        psi_filter.col(t-1) = theta_stored.slice(t-1).row(0).t();
    }

    psi_smooth.col(n) = theta_stored.slice(n).row(0).t();
    if (smoothing) {
        for (unsigned int t=n; t>0; t--) {   
            for (unsigned int i=0; i<N; i++) {
                w = -0.5*arma::pow((psi_smooth.at(i,t)-psi_filter.col(t-1))/Wt.at(t-1),2);
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

            // if (verbose)
            {
                Rcout << "\rSmoothing Progress: " << n-t+1 << "/" << n;
            }
        }

        Rcout << std::endl;
    }
    
    


    Rcpp::List output;
    

    arma::mat R(n+1,3); // quantiles
    if (smoothing) {
        R = arma::quantile(psi_smooth,Rcpp::as<arma::vec>(qProb),0); // 3 columns, smoothed psi
        output["psi"] = Rcpp::wrap(R.t());
    } else {
        R = arma::quantile(psi_filter,Rcpp::as<arma::vec>(qProb),0);
        output["psi"] = Rcpp::wrap(R.t());
    }

    double log_marg_lik = 0.;
    for (unsigned int t=npara; t<(n+1); t++) {
        log_marg_lik += std::log(std::max(w_stored.at(t),1.e-32));
    }
    output["log_marg_lik"] = log_marg_lik;
    output["marg_lik"] = Rcpp::wrap(w_stored);

    // add RMSE and MAE between lambda[t] and y[t]
    arma::vec hpsiR = psi2hpsi(arma::vectorise(R.row(1)),model_code.at(3),ctanh); // hpsi: p x N
    double theta0 = 0;
    arma::vec lambdaR = hpsi2theta(hpsiR, Y, trans_code, theta0, alpha, L, rho); // n x 1
    output["rmse"] = std::sqrt(arma::as_scalar(arma::mean(arma::pow(lambdaR.tail(n-npara) - Y.tail(n-npara),2.0))));
    output["mae"] = arma::as_scalar(arma::mean(arma::abs(lambdaR.tail(n-npara) - Y.tail(n-npara))));

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
    arma::vec& pmarg_y, // n x 1, marginal likelihood of y
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
    const double theta0_upbnd = 2.,
    const Rcpp::NumericVector& ctanh = Rcpp::NumericVector::create(0.2,0,5),
    const double delta_nb = 1., // 0: negative binomial DLM; 1: poisson DLM
    const double delta_discount = 0.95){ 

    const unsigned int n = ypad.n_elem - 1; // number of observations
    const double N_ = static_cast<double>(N);


    const unsigned int obs_code = model_code.at(0);
    const unsigned int link_code = model_code.at(1);
    const unsigned int trans_code = model_code.at(2);
    const unsigned int gain_code = model_code.at(3);
    const unsigned int err_code = model_code.at(4);
	unsigned int p, L_;
	arma::vec Ft, Fphi;
    arma::mat Gt;
    init_by_trans(p,L_,Ft,Fphi,Gt,trans_code,L);

        
    arma::mat theta(p,N);
    arma::vec m0(p,arma::fill::zeros);
    arma::mat C0(p,p,arma::fill::eye); C0 *= std::pow(theta0_upbnd*0.5,2.);
    if (!m0_prior.isNull()) {
		m0 = Rcpp::as<arma::vec>(m0_prior);
	}

    C0 = arma::chol(C0);
    for (unsigned int i=0; i<N; i++) {
        theta.col(i) = m0 + C0.t() * arma::randn(p);
    }

    arma::cube theta_stored(p, N, n + B);
    for (unsigned int b = 0; b < B; b++)
    {
        theta_stored.slice(b).zeros();
    }
    theta_stored.slice(B-1) = theta;


    arma::vec Wt(n);


    for (unsigned int t=0; t<n; t++) {
        R_CheckUserInterrupt();
        arma::mat Theta_old = theta_stored.slice(t + B - 1);

        
        if (trans_code != 1)
        {
            arma::vec mt = arma::median(Theta_old, 1);
            update_Gt(Gt, gain_code, trans_code, mt, ctanh, alpha, ypad.at(t), rho);
        }


        if (R_IsNA(W))
        { // Use discount factor if W is not given
            Wt.at(t) = arma::var(Theta_old.row(0));
            // Wsqrt = std::sqrt(Wt.at(t));
            if (t > B)
            {
                Wt.at(t) *= 1. / delta_discount - 1.;
            }
            else
            {
                Wt.at(t) *= 1. / 0.99 - 1.;
            }
        }
        double Wsqrt = std::sqrt(Wt.at(t));


        for (unsigned int i=0; i<N; i++) {
            arma::vec theta_old = Theta_old.col(i);
            arma::vec theta_new = update_at(
                p, gain_code, trans_code, theta_old, Gt, ctanh, alpha, ypad.at(t), rho
            );

            double omega_new = R::rnorm(0., Wsqrt);
            theta_new.at(0) += omega_new;

            theta.col(i) = theta_new;
        }
        
        theta_stored.slice(t+B) = theta;
        /*
        ------ Step 2.1 Propagate ------
        */



        /*
        ------ Step 2.2 Importance weights ------
        */
        if (trans_code == 1)
        {
            // Koyama
            arma::vec Fy(p, arma::fill::zeros);
            unsigned int tmpi = std::min(t, p);
            if (t > 0)
            {
                // Checked Fy - CORRECT
                Fy.head(tmpi) = arma::reverse(ypad.subvec(t + 1 - tmpi, t));
            }

            Ft = Fphi % Fy; // L(p) x 1
        }

        arma::vec w(N);
        arma::vec lambda(N);
        for (unsigned int i=0; i<N; i++) 
        {
            arma::vec psi = theta.col(i); // p x 1

            // theta: p x N
            if (link_code == 1) 
            {
                // Exponential link and identity gain
                double tmp = arma::as_scalar(Ft.t() * psi);
                lambda.at(i) = std::min(mu0 + tmp, UPBND);
                double tmp2 = std::exp(lambda.at(i));
                lambda.at(i) = tmp2;
            } else if (trans_code == 1) 
            {
                // Koyama
                arma::vec hpsi = psi2hpsi(psi, gain_code, ctanh);
                lambda.at(i) = mu0 + arma::as_scalar(Ft.t() * hpsi);
            } else {
                // Koyck or Solow with identity link and different gain functions
                lambda.at(i) = mu0 + psi.at(1);
            }

            w.at(i) = loglike_obs(ypad.at(t + 1), lambda.at(i), obs_code, delta_nb, false);
        }
        
        
        /*
        ------ Step 2.2 Importance weights ------
        */

        /*
        ------ Step 3 Resampling with Replacement ------
        */
        double wsum = arma::accu(w);
        bool resample = false;

        if (wsum > EPS)
        {
            // normalize the particle weights
            arma::vec wtmp = w;
            w = wtmp / wsum;
            double Meff = 1. / arma::dot(w, w);
            if (Meff > 0.2 * N_)
            {
                resample = true;
            }
            else
            {
                resample = false;
            }
        }
        else
        {
            resample = false;
        }

        if (resample)
        {

            Rcpp::NumericVector w_ = Rcpp::wrap(w);
            Rcpp::IntegerVector idx_ = Rcpp::sample(N, N, true, w_);
            arma::uvec idx = Rcpp::as<arma::uvec>(idx_) - 1;

            // from t+1, to t+B
            for (unsigned int b = t + 1; b < (t + B + 1); b++)
            {
                arma::mat ttmp = theta_stored.slice(b);
                theta_stored.slice(b) = ttmp.cols(idx);
            }

            pmarg_y.at(t) = std::log(arma::accu(w)) - std::log(N_);
        } else {
            if (w.has_nan())
            {
                w.elem(arma::find_nan(w)).fill(1.e-32);
            }
            pmarg_y.at(t) = std::log(arma::accu(w)) - std::log(N_);
        }
        w.ones();
    }


    R.zeros();
    {
        arma::mat psi_all = theta_stored.row(0);        // N x (nt+B)
        arma::mat psi = psi_all.cols(B - 1, n + B - 1); // N x (nt+1)
        arma::vec RR = arma::median(psi, 0);            // (nt+1)

        R.col(0) = RR; // (nt+1) x 2
    }

    if (trans_code == 1)
    {
        // Koyama
        for (unsigned int t = 0; t < n; t++)
        {
            arma::vec Fy(p, arma::fill::zeros);
            unsigned int tmpi = std::min(t, p);
            if (t > 0)
            {
                // Checked Fy - CORRECT
                Fy.head(tmpi) = arma::reverse(ypad.subvec(t + 1 - tmpi, t));
            }

            Ft = Fphi % Fy; // L(p) x 1
            // theta_stored: p x N x (nt+B)
            arma::mat hpsi = psi2hpsi(theta_stored.slice(B + t), gain_code, ctanh);
            // hpsi: p x N
            arma::vec lambda(N);
            for (unsigned int i = 0; i < N; i++)
            {
                lambda.at(i) = arma::as_scalar(Ft.t() * hpsi.col(i));
            }

            R.at(t + 1, 1) = arma::median(lambda);
        } // end loop over t
    }
    else
    {
        arma::mat ft_all = theta_stored.row(1);       // N x (nt+B)
        arma::mat ft = ft_all.cols(B - 1, n + B - 1); // N x (nt+1)
        arma::vec RR = arma::median(ft, 0);           // (nt+1)
        R.col(1) = RR;
    }


    return;
}




//' @export
// [[Rcpp::export]]
Rcpp::List pl_poisson(
    const arma::vec &Y, // nt x 1, the observed response
    const arma::uvec &model_code,
    const Rcpp::NumericVector &W_prior = Rcpp::NumericVector::create(0.01, 0.01), // IG[aw,bw]
    const double W_true = NA_REAL,
    const unsigned int L = 2,    // number of lags
    const unsigned int N = 5000, // number of particles
    const Rcpp::Nullable<Rcpp::NumericVector> &m0_prior = R_NilValue,
    const Rcpp::Nullable<Rcpp::NumericMatrix> &C0_prior = R_NilValue,
    const Rcpp::NumericVector &qProb = Rcpp::NumericVector::create(0.025, 0.5, 0.975),
    const double mu0 = 0.,
    const double rho = 0.9,
    const double delta_nb = 1.,
    const double theta0_upbnd = 2.)
{
    const double alpha = 1.;
    const Rcpp::NumericVector ctanh = {1., 0., 1.}; // (1./M, 0, M)

    const unsigned int nt = Y.n_elem; // number of observations
    const double N_ = static_cast<double>(N);
    const double nt_ = static_cast<double>(nt);
    arma::vec ypad(nt + 1, arma::fill::zeros);
    ypad.tail(nt) = Y;

    /*
    Define F[t] and G[t].

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
    init_by_trans(p, L_, Ft, Fphi, Gt, trans_code, L);

    /* Dimension of state space depends on type of transfer functions */

    /*
    Initialize latent state theta[0].
    */
    arma::mat theta(p, N);
    arma::vec m0(p, arma::fill::zeros);
    arma::mat C0(p, p, arma::fill::eye);
    C0 *= std::pow(theta0_upbnd * 0.5, 2.);
    if (!m0_prior.isNull())
    {
        m0 = Rcpp::as<arma::vec>(m0_prior);
    }
    C0 = arma::chol(C0);
    for (unsigned int i = 0; i < N; i++)
    {
        theta.col(i) = m0 + C0.t() * arma::randn(p);
    }
    arma::cube theta_stored(p, N, nt + 1);
    theta_stored.slice(0) = theta;

    arma::mat aw(N, nt + 1);
    aw.fill(W_prior[0]);
    arma::mat bw(N, nt + 1);
    bw.fill(W_prior[1]);
    arma::mat Wt(N, nt + 1); // evolution variance
    if (!R_IsNA(W_true))
    {
        Wt.fill(W_true);
    }
    else
    {
        double wtmp = arma::var(theta_stored.slice(0).row(0));
        wtmp *= 1. / 0.99 - 1.;
        Wt.col(0).fill(wtmp);
    }

    arma::vec Meff(nt, arma::fill::zeros);
    // Effective sample size (Ref: Lin, 1996; Prado, 2021, page 196)
    arma::uvec resample_status(nt, arma::fill::zeros);

    for (unsigned int t = 0; t < nt; t++)
    {
        R_CheckUserInterrupt();

        /*
        Propagate
        */
        arma::mat Theta_old = theta_stored.slice(t); // p x N

        for (unsigned int i = 0; i < N; i++)
        {
            arma::vec theta_old = Theta_old.col(i); // p x 1
            arma::vec theta_new = update_at(
                p, gain_code, trans_code, theta_old, Gt, ctanh, 1., ypad.at(t), rho);

            double Wsqrt = std::sqrt(Wt.at(i, t));
            double omega_new = R::rnorm(0., Wsqrt);
            theta_new.at(0) += omega_new;

            theta_stored.slice(t + 1).col(i) = theta_new;

            if (R_IsNA(W_true)) {
                // infer evolution variance W
                double err = theta_stored.at(0,i,t+1) - theta_stored.at(0,i,t);
                double sse = std::pow(err,2.);
                aw.at(i,t+1) = aw.at(i,t) + 0.5;
                bw.at(i,t+1) = bw.at(i,t) + 0.5*sse;
                if (t > std::min(0.1 * nt_,20.))
                {
                    Wt.at(i, t + 1) = 1. / R::rgamma(aw.at(i, t + 1), 1. / bw.at(i, t + 1));
                }
            } else {
                Wt.at(i,t+1) = W_true;
            }
        }

        if (R_IsNA(W_true) && (t <= std::min(0.1 * nt_, 20.)))
        {
            double wtmp = arma::var(theta_stored.slice(t + 1).row(0));
            wtmp *= 1./0.99 - 1.;
            Wt.col(t + 1).fill(wtmp);
        }

        /*
        Resample
        - theta (p x N);
        - a_sigma2, b_sigma2, a_tau2, b_tau2 (N x 2)
        - sigma2, tau2 (N x 2)
        */
        if (trans_code == 1)
        {
            // Update F[t] for Koyama model.
            arma::vec Fy(p, arma::fill::zeros);
            unsigned int tmpi = std::min(t, p);
            if (t > 0)
            {
                // Checked Fy - CORRECT
                Fy.head(tmpi) = arma::reverse(ypad.subvec(t + 1 - tmpi, t));
            }
            Ft = Fphi % Fy; // L(p) x 1
        }
        // Checked OK.

        arma::vec weight(N); // importance weight of each particle
        arma::vec lambda(N);
        for (unsigned int i = 0; i < N; i++)
        {
            arma::vec psi = theta_stored.slice(t + 1).col(i); // p x 1, theta[t]

            if (link_code == 1)
            {
                // Exponential link and identity gain
                double tmp = arma::as_scalar(Ft.t() * psi);
                lambda.at(i) = std::min(mu0 + tmp, UPBND);
                double tmp2 = std::exp(lambda.at(i));
                lambda.at(i) = tmp2;
            }
            else if (trans_code == 1)
            {
                // Koyama
                arma::vec hpsi = psi2hpsi(psi, gain_code, ctanh);
                lambda.at(i) = mu0 + arma::as_scalar(Ft.t() * hpsi);
            }
            else
            {
                // Koyck or Solow with identity link and different gain functions
                lambda.at(i) = mu0 + psi.at(1);
            }

            weight.at(i) = loglike_obs(
                ypad.at(t + 1), lambda.at(i), obs_code, delta_nb, false);
        }
        // Checked. OK

        double wsum = arma::accu(weight);
        bool resample = false;
        if (wsum > EPS)
        {
            // normalize the particle weights
            arma::vec wtmp = weight;
            weight = wtmp / wsum;
            Meff.at(t) = 1. / arma::dot(weight, weight);
            if (Meff.at(t) > std::max(100., 0.1 * N_))
            {
                resample = true;
            }
            else
            {
                resample = false;
            }
        }
        else
        {
            resample = false;
            Meff.at(t) = 0.;
        }

        if (resample)
        {
            Rcpp::NumericVector w_ = Rcpp::wrap(weight);
            Rcpp::IntegerVector idx_ = Rcpp::sample(N, N, true, w_);
            arma::uvec idx = Rcpp::as<arma::uvec>(idx_) - 1;

            arma::mat tttmp = theta_stored.slice(t + 1);
            theta_stored.slice(t + 1) = tttmp.cols(idx);

            arma::vec atmp = aw.col(t + 1);
            aw.col(t + 1) = atmp.elem(idx);

            arma::vec btmp = bw.col(t + 1);
            bw.col(t + 1) = btmp.elem(idx);

            arma::vec stmp = Wt.col(t + 1);
            Wt.col(t + 1) = stmp.elem(idx);
        }
        weight.ones();
        resample_status.at(t) = resample;
    }

    Rcpp::List output;
    arma::mat psi = theta_stored.row(0); // N x (nt+1)
    arma::mat RR = arma::quantile(psi, Rcpp::as<arma::vec>(qProb), 0);
    output["psi"] = Rcpp::wrap(RR.t());         // (n+1) x 3
    output["theta"] = Rcpp::wrap(theta_stored); // p, N, nt + 1

    output["aw"] = Rcpp::wrap(aw.col(nt));
    output["bw"] = Rcpp::wrap(bw.col(nt));
    output["Wt"] = Rcpp::wrap(Wt);

    // if (debug) {
    output["Meff"] = Rcpp::wrap(Meff);
    output["resample_status"] = Rcpp::wrap(resample_status);
    // }

    return output;
}



// Y[t] ~ Normal(Fmat/theta[t],sigma2)
// theta[t] ~ Normal(Gmat*theta[t-1],tau2)
Rcpp::List pl_gaussian_posterior(
    const arma::mat &Y,           // nt x n, the observed response
    const arma::uvec &eta_select, // 2 x 1, indicator for unknown (=1) or known (=0) (sigma2,tau2)
    const unsigned int p = 2,                                              // dimension of the state space
    const unsigned int N = 5000,                                           // number of particles
    const Rcpp::Nullable<Rcpp::NumericMatrix> &eta_prior_val = R_NilValue, // 2 x 2, priors for each element of eta: first column for sigma2, second column for tau2
    const Rcpp::Nullable<Rcpp::NumericVector> &eta_true = R_NilValue,      // 2 x 1, if true/initial values should be provided here
    const Rcpp::Nullable<Rcpp::NumericMatrix> &Fmat = R_NilValue,          // p x n
    const Rcpp::Nullable<Rcpp::NumericMatrix> &Gmat = R_NilValue,          // p x p
    const Rcpp::Nullable<Rcpp::NumericVector> &m0_prior = R_NilValue,
    const Rcpp::Nullable<Rcpp::NumericMatrix> &C0_prior = R_NilValue,
    const Rcpp::NumericVector &qProb = Rcpp::NumericVector::create(0.025, 0.5, 0.975),
    const bool verbose = true,
    const bool debug = false)
{

    const unsigned int nt = Y.n_rows; // number of observations
    const unsigned int n = Y.n_cols; // dimension of Y[t]

    const double n_ = static_cast<double>(n);
    const double min_eff = 0.8 * static_cast<double>(N);

    arma::mat F; // p x n
    if (!Fmat.isNull()) {
        F = Rcpp::as<arma::mat>(Fmat);
    } else {
        // p <= n
        F.set_size(p,n);
        F.diag().ones();
    }

    arma::mat G; // p x p
    if (!Gmat.isNull()){
        G = Rcpp::as<arma::mat>(Gmat);
    } else {
        G.set_size(p,p);
        G.eye();
    }
    
    arma::vec m0; // p x 1
    if (!m0_prior.isNull()) {
        m0 = Rcpp::as<arma::vec>(m0_prior);
    } else {
        m0.set_size(p);
        m0.zeros();
    }

    arma::mat C0; // p x p
    if (!C0_prior.isNull()) {
        C0 = Rcpp::as<arma::mat>(C0_prior);
    } else {
        C0.set_size(p, p);
        C0.eye();
    }
    arma::mat C0_chol = arma::chol(C0);

    double sigma2_true, tau2_true;
    if (!eta_true.isNull()) {
        arma::vec tmp = Rcpp::as<arma::vec>(eta_true);
        if (eta_select.at(0) == 0) { // fix sigma2
            sigma2_true = tmp.at(0);
        }
        if (eta_select.at(1) == 0) { // fix tau2
            tau2_true = tmp.at(1);
        }
    }

    double a_sigma0 = 0.01;
    double b_sigma0 = 0.01;
    double a_tau0 = 0.01;
    double b_tau0 = 0.01;
    if (!eta_prior_val.isNull()) {
        arma::mat tmp = Rcpp::as<arma::mat>(eta_prior_val);
        a_sigma0 = tmp.at(0,0);
        b_sigma0 = tmp.at(1,0);
        a_tau0 = tmp.at(0,1);
        b_tau0 = tmp.at(1,1);
    }

    // first column is [t-1], second column is [t]
    arma::mat a_sigma(N,nt+1); 
    arma::mat b_sigma(N,nt+1);
    arma::mat sigma2(N,nt+1);
    // sigma2.col(0).fill(R::runif(0.,1.));
    // sigma2.col(0) = 1. / arma::randg(N, arma::distr_param(a_sigma0, 1. / b_sigma0));
    for (unsigned int i=0; i<N; i++) {
        a_sigma.at(i,0) = a_sigma0;
        b_sigma.at(i,0) = b_sigma0;
        sigma2.at(i,0) = R::runif(0.,1.);
    }
    // Rcout << sigma2.col(0).t() << std::endl;

    // Rcout << "a_sigma0 = " << a_sigma0 << ", b_sigma0 = " << b_sigma0 
    // sigma2.col(1) = 1. / arma::randg(N, arma::distr_param(a_sigma.at(0, 1), 1. / b_sigma.at(0, 1)));

    arma::mat a_tau(N,nt+1);
    arma::mat b_tau(N,nt+1);
    arma::mat tau2(N,nt+1);
    for (unsigned int i = 0; i < N; i++)
    {
        a_tau.at(i, 0) = a_tau0;
        a_tau.at(i, 0) = b_tau0;
        tau2.at(i, 0) = R::runif(0., 1.);
    }
    // tau2.col(0).fill(R::runif(0., 1.));
    // tau2.col(0) = 1. / arma::randg(N, arma::distr_param(a_tau0, 1. / b_tau0));
    // tau2.col(1) = 1. / arma::randg(N, arma::distr_param(a_tau.at(0, 1), 1. / b_tau.at(0, 1)));

    arma::mat theta(p, N);
    for (unsigned int i = 0; i < N; i++)
    {
        theta.col(i) = m0 + C0_chol.t() * arma::randn(p);
    }
    arma::cube theta_stored(p, N, nt+1);
    theta_stored.slice(0) = theta;

    const arma::mat FFt = F*F.t();
    const arma::mat Ip(p,p,arma::fill::eye);
    const arma::mat FtF = F.t()*F;
    const arma::mat In(n,n,arma::fill::eye);
    const double mvn_cnst = -0.5 * static_cast<double>(n) * std::log(2*arma::datum::pi);
    arma::vec Meff(nt, arma::fill::zeros); // Effective sample size (Ref: Lin, 1996; Prado, 2021, page 196)
    arma::uvec resample_status(nt, arma::fill::zeros);

    for (unsigned int t = 0; t < nt; t++)
    {
        R_CheckUserInterrupt();
        arma::vec Yt = Y.row(t).t();

        /*
        Resample
        - theta (p x N);
        - a_sigma2, b_sigma2, a_tau2, b_tau2 (N x 2)
        - sigma2, tau2 (N x 2)
        */
        arma::vec w(N); // importance weight of each particle
        for (unsigned int i=0; i<N; i++) {
            /*
            Sigy = t(Sigy_chol) * Sigy_chol
            Sigy_inv_chol = inv(Sigy_chol)
            Sigy_inv_chol * t(Sigy_inv_chol) = Sigy_inv
            */
            arma::mat Sigy = FtF * sigma2.at(i, t) * tau2.at(i, t) + In * sigma2.at(i, t);
            arma::mat Sigy_chol = arma::chol(arma::symmatu(Sigy));
            arma::mat Sigy_inv_chol = arma::inv(arma::trimatu(Sigy_chol));
            arma::mat Sigy_inv = arma::symmatu(Sigy_inv_chol * Sigy_inv_chol.t());
            arma::vec muy = F.t() * G * theta.col(i);

            double logdet_val_Sigy_inv;
            double logdet_sign_Sigy_inv;
            bool ok = arma::log_det(logdet_val_Sigy_inv, logdet_sign_Sigy_inv, Sigy_inv);
            arma::vec err3 = Yt - muy;
            double sse = arma::as_scalar(err3.t() * Sigy_inv * err3);

            double loga = mvn_cnst + 0.5 * logdet_val_Sigy_inv - 0.5 * sse;
            double alphat = std::exp(std::min(loga, 700.));
            w.at(i) = alphat;
        }

        if (arma::accu(w) > EPS)
        { // normalize the particle weights
            w /= arma::accu(w);
            Meff.at(t) = 1. / arma::dot(w, w);
            if (Meff.at(t) > 0.1 * static_cast<double>(N))
            {
            resample_status.at(t) = 1;
            }
            else
            {
            resample_status.at(t) = 0;
            }
        }
        else
        {
            resample_status.at(t) = 0;
            Meff.at(t) = 0.;
        }

        if (resample_status.at(t) == 1)
        {
            Rcpp::NumericVector w_ = Rcpp::wrap(w);
            Rcpp::IntegerVector idx_ = Rcpp::sample(N, N, true, w_);
            arma::uvec idx = Rcpp::as<arma::uvec>(idx_) - 1;

            theta_stored.slice(t) = theta.cols(idx);

            arma::vec atmp = a_sigma.col(t);
            a_sigma.col(t) = atmp.elem(idx);

            arma::vec btmp = b_sigma.col(t);
            b_sigma.col(t) = btmp.elem(idx);

            arma::vec stmp = sigma2.col(t);
            sigma2.col(t) = stmp.elem(idx);

            arma::vec atmp2 = a_tau.col(t);
            a_tau.col(t) = atmp2.elem(idx);

            arma::vec btmp2 = b_tau.col(t);
            b_tau.col(t) = btmp2.elem(idx);

            arma::vec ttmp = tau2.col(t);
            tau2.col(t) = ttmp.elem(idx);
        }
        else
        {
            theta_stored.slice(t) = theta;
        }

        /*
        Propagate
        */
        w.ones(); // Equal weight to begin with because of resampling in the last iter.

        for (unsigned int i=0; i<N; i++) {
            arma::vec theta_old = theta_stored.slice(t).col(i);

            arma::mat Sig_inv = FFt/sigma2.at(i,t) + Ip/(sigma2.at(i,t)*tau2.at(i,t));
            arma::mat Sig_inv_chol = arma::chol(arma::symmatu(Sig_inv)); // Sig_inv = t(Sig_inv_chol)*Sig_inv_chol

            arma::mat Sig_chol = arma::inv(arma::trimatu(Sig_inv_chol)); // Sigma = Sig_chol * t(Sig_chol)
            arma::mat Sigma = Sig_chol * Sig_chol.t();

            arma::vec tmp1 = F * Yt / sigma2.at(i,t);
            arma::vec tmp2 = G * theta_old / (sigma2.at(i,t)*tau2.at(i,t));
            arma::vec mu = Sigma * (tmp1 + tmp2);

            arma::vec theta_new = mu + Sig_chol * arma::randn(p);
            theta.col(i) = theta_new;

            arma::vec err1 = Yt - F.t()*theta_new;
            double sse1 = arma::as_scalar(err1.t()*err1);

            arma::vec err2 = theta_new - G*theta_old;
            double sse2 = arma::as_scalar(err2.t()*err2);

            if (eta_select.at(0)==1) {
                // infer sigma2
                a_sigma.at(i, t+1) = a_sigma.at(i, t) + 0.5*static_cast<double>(n) + 0.5*static_cast<double>(p);
                b_sigma.at(i, t+1) = b_sigma.at(i, t) + 0.5*sse1 + 0.5*sse2/tau2.at(i,t);
                sigma2.at(i, t+1) = 1. / R::rgamma(a_sigma.at(i, t+1), 1./b_sigma.at(i, t+1));
                // R::rgamma(shape, scale)
            } else {
                // fix sigma2
                sigma2.at(i,t+1) = sigma2_true;
            }

            if (eta_select.at(1)==1) {
                // infer tau2
                a_tau.at(i, t+1) = a_tau.at(i, t) + 0.5*static_cast<double>(p);
                b_tau.at(i, t+1) = b_tau.at(i, t) + 0.5*sse2/sigma2.at(i,t+1);
                tau2.at(i, t+1) = 1. / R::rgamma(a_tau.at(i, t+1), 1./b_tau.at(i, t+1));
                // R::rgamma(shape, scale)
            } else {
                // fix tau2
                tau2.at(i,t+1) = tau2_true;
            }
        }


        

        if (verbose)
        {
            Rcout << "\rFiltering Progress: " << t + 1 << "/" << nt;
        }
    }

    if (verbose)
    {
        Rcout << std::endl;
    }

    Rcpp::List output;
    // arma::mat R = arma::quantile(psi, Rcpp::as<arma::vec>(qProb), 1);
    output["theta"] = Rcpp::wrap(theta_stored); // (n+1) x 3
    output["a_sigma"] = Rcpp::wrap(a_sigma.col(nt));
    output["b_sigma"] = Rcpp::wrap(b_sigma.col(nt));
    output["sigma2"] = Rcpp::wrap(sigma2.col(nt));
    output["a_tau"] = Rcpp::wrap(a_tau.col(nt));
    output["b_tau"] = Rcpp::wrap(b_tau.col(nt));
    output["tau2"] = Rcpp::wrap(tau2.col(nt));

    // if (debug) {
    output["Meff"] = Rcpp::wrap(Meff);
    output["resample_status"] = Rcpp::wrap(resample_status);
    // }

    return output;
}



Rcpp::List pl_gaussian_evolution(
    const arma::mat &Y,                                                    // nt x n, the observed response
    const arma::uvec &eta_select,                                          // 2 x 1, indicator for unknown (=1) or known (=0) (sigma2,tau2)
    const unsigned int p = 2,                                              // dimension of the state space
    const unsigned int N = 5000,                                           // number of particles
    const Rcpp::Nullable<Rcpp::NumericMatrix> &eta_prior_val = R_NilValue, // 2 x 2, priors for each element of eta: first column for sigma2, second column for tau2
    const Rcpp::Nullable<Rcpp::NumericVector> &eta_true = R_NilValue,      // 2 x 1, if true/initial values should be provided here
    const Rcpp::Nullable<Rcpp::NumericMatrix> &Fmat = R_NilValue,          // p x n
    const Rcpp::Nullable<Rcpp::NumericMatrix> &Gmat = R_NilValue,          // p x p
    const Rcpp::Nullable<Rcpp::NumericVector> &m0_prior = R_NilValue,
    const Rcpp::Nullable<Rcpp::NumericMatrix> &C0_prior = R_NilValue,
    const Rcpp::NumericVector &qProb = Rcpp::NumericVector::create(0.025, 0.5, 0.975),
    const bool verbose = true,
    const bool debug = false)
{

    const unsigned int nt = Y.n_rows; // number of observations
    const unsigned int n = Y.n_cols;  // dimension of Y[t]

    const double n_ = static_cast<double>(n);
    const double min_eff = 0.8 * static_cast<double>(N);

    arma::mat F; // p x n
    if (!Fmat.isNull())
    {
        F = Rcpp::as<arma::mat>(Fmat);
    }
    else
    {
        // p <= n
        F.set_size(p, n);
        F.diag().ones();
    }

    arma::mat G; // p x p
    if (!Gmat.isNull())
    {
        G = Rcpp::as<arma::mat>(Gmat);
    }
    else
    {
        G.set_size(p, p);
        G.eye();
    }

    arma::vec m0; // p x 1
    if (!m0_prior.isNull())
    {
        m0 = Rcpp::as<arma::vec>(m0_prior);
    }
    else
    {
        m0.set_size(p);
        m0.zeros();
    }

    arma::mat C0; // p x p
    if (!C0_prior.isNull())
    {
        C0 = Rcpp::as<arma::mat>(C0_prior);
    }
    else
    {
        C0.set_size(p, p);
        C0.eye();
    }
    arma::mat C0_chol = arma::chol(C0);

    double sigma2_true, tau2_true;
    if (!eta_true.isNull())
    {
        arma::vec tmp = Rcpp::as<arma::vec>(eta_true);
        if (eta_select.at(0) == 0)
        { // fix sigma2
            sigma2_true = tmp.at(0);
        }
        if (eta_select.at(1) == 0)
        { // fix tau2
            tau2_true = tmp.at(1);
        }
    }

    double a_sigma0 = 0.01;
    double b_sigma0 = 0.01;
    double a_tau0 = 0.01;
    double b_tau0 = 0.01;
    if (!eta_prior_val.isNull())
    {
        arma::mat tmp = Rcpp::as<arma::mat>(eta_prior_val);
        a_sigma0 = tmp.at(0, 0);
        b_sigma0 = tmp.at(1, 0);
        a_tau0 = tmp.at(0, 1);
        b_tau0 = tmp.at(1, 1);
    }

    // first column is [t-1], second column is [t]
    arma::mat a_sigma(N, nt + 1);
    arma::mat b_sigma(N, nt + 1);
    arma::mat sigma2(N, nt + 1);
    // sigma2.col(0).fill(R::runif(0.,1.));
    // sigma2.col(0) = 1. / arma::randg(N, arma::distr_param(a_sigma0, 1. / b_sigma0));
    for (unsigned int i = 0; i < N; i++)
    {
        a_sigma.at(i, 0) = a_sigma0;
        b_sigma.at(i, 0) = b_sigma0;
        sigma2.at(i, 0) = R::runif(0., 1.);
    }
    // Rcout << sigma2.col(0).t() << std::endl;

    // Rcout << "a_sigma0 = " << a_sigma0 << ", b_sigma0 = " << b_sigma0
    // sigma2.col(1) = 1. / arma::randg(N, arma::distr_param(a_sigma.at(0, 1), 1. / b_sigma.at(0, 1)));

    arma::mat a_tau(N, nt + 1);
    arma::mat b_tau(N, nt + 1);
    arma::mat tau2(N, nt + 1);
    for (unsigned int i = 0; i < N; i++)
    {
        a_tau.at(i, 0) = a_tau0;
        a_tau.at(i, 0) = b_tau0;
        tau2.at(i, 0) = R::runif(0., 1.);
    }
    // tau2.col(0).fill(R::runif(0., 1.));
    // tau2.col(0) = 1. / arma::randg(N, arma::distr_param(a_tau0, 1. / b_tau0));
    // tau2.col(1) = 1. / arma::randg(N, arma::distr_param(a_tau.at(0, 1), 1. / b_tau.at(0, 1)));

    arma::mat theta(p, N);
    for (unsigned int i = 0; i < N; i++)
    {
        theta.col(i) = m0 + C0_chol.t() * arma::randn(p);
    }
    arma::cube theta_stored(p, N, nt + 1);
    theta_stored.slice(0) = theta;

    const arma::mat FFt = F * F.t();
    const arma::mat Ip(p, p, arma::fill::eye);
    const arma::mat FtF = F.t() * F;
    const arma::mat In(n, n, arma::fill::eye);
    const double mvn_cnst = -0.5 * static_cast<double>(n) * std::log(2 * arma::datum::pi);
    arma::vec Meff(nt, arma::fill::zeros); // Effective sample size (Ref: Lin, 1996; Prado, 2021, page 196)
    arma::uvec resample_status(nt, arma::fill::zeros);

    for (unsigned int t = 0; t < nt; t++)
    {
        R_CheckUserInterrupt();
        arma::vec Yt = Y.row(t).t();
        arma::vec Yt_old(n,arma::fill::zeros);
        if (t>0) {
            Yt_old = Y.row(t-1).t();
        }

        /*
        Propagate
        */
        for (unsigned int i = 0; i < N; i++)
        {
            arma::vec theta_old = theta_stored.slice(t).col(i);
            arma::mat Sigma = Ip * (sigma2.at(i,t)*tau2.at(i,t));
            arma::mat Sig_chol = Ip * std::sqrt(sigma2.at(i,t)*tau2.at(i,t));

            arma::vec mu = G * theta_old;
            arma::vec theta_new = mu + Sig_chol * arma::randn(p);

            theta_stored.slice(t+1).col(i) = theta_new;

            arma::vec err1 = Yt_old - F.t() * theta_old;
            double sse1 = arma::as_scalar(err1.t() * err1);

            arma::vec err2 = theta_new - G * theta_old;
            double sse2 = arma::as_scalar(err2.t() * err2);

            if (eta_select.at(0) == 1)
            {
                // infer sigma2
                a_sigma.at(i, t + 1) = a_sigma.at(i, t) + 0.5 * static_cast<double>(n) + 0.5 * static_cast<double>(p);
                b_sigma.at(i, t + 1) = b_sigma.at(i, t) + 0.5 * sse1 + 0.5 * sse2 / tau2.at(i, t);
                sigma2.at(i, t + 1) = 1. / R::rgamma(a_sigma.at(i, t + 1), 1. / b_sigma.at(i, t + 1));
                // R::rgamma(shape, scale)
            }
            else
            {
                // fix sigma2
                sigma2.at(i, t + 1) = sigma2_true;
            }

            if (eta_select.at(1) == 1)
            {
                // infer tau2
                a_tau.at(i, t + 1) = a_tau.at(i, t) + 0.5 * static_cast<double>(p);
                b_tau.at(i, t + 1) = b_tau.at(i, t) + 0.5 * sse2 / sigma2.at(i, t + 1);
                tau2.at(i, t + 1) = 1. / R::rgamma(a_tau.at(i, t + 1), 1. / b_tau.at(i, t + 1));
                // R::rgamma(shape, scale)
            }
            else
            {
                // fix tau2
                tau2.at(i, t + 1) = tau2_true;
            }
        }

        /*
        Resample
        - theta (p x N);
        - a_sigma2, b_sigma2, a_tau2, b_tau2 (N x 2)
        - sigma2, tau2 (N x 2)
        */
        arma::vec w(N); // importance weight of each particle
        for (unsigned int i = 0; i < N; i++)
        {
            /*
            Sigy = t(Sigy_chol) * Sigy_chol
            Sigy_inv_chol = inv(Sigy_chol)
            Sigy_inv_chol * t(Sigy_inv_chol) = Sigy_inv
            */
            arma::mat Sigy = In * sigma2.at(i, t+1);
            arma::mat Sigy_inv = In * (1./sigma2.at(i,t+1));
            arma::vec muy = F.t() * theta_stored.slice(t+1).col(i);

            double logdet_val_Sigy_inv;
            double logdet_sign_Sigy_inv;
            bool ok = arma::log_det(logdet_val_Sigy_inv, logdet_sign_Sigy_inv, Sigy_inv);
            arma::vec err3 = Yt - muy;
            double sse = arma::as_scalar(err3.t() * Sigy_inv * err3);

            double loga = mvn_cnst + 0.5 * logdet_val_Sigy_inv - 0.5 * sse;
            double alphat = std::exp(std::min(loga, 700.));
            w.at(i) = alphat;
        }

        if (arma::accu(w) > EPS)
        { // normalize the particle weights
            w /= arma::accu(w);
            Meff.at(t) = 1. / arma::dot(w, w);
            if (Meff.at(t) > 0.1 * static_cast<double>(N))
            {
                resample_status.at(t) = 1;
            }
            else
            {
                resample_status.at(t) = 0;
            }
        }
        else
        {
            resample_status.at(t) = 0;
            Meff.at(t) = 0.;
        }

        if (resample_status.at(t) == 1)
        {
            Rcpp::NumericVector w_ = Rcpp::wrap(w);
            Rcpp::IntegerVector idx_ = Rcpp::sample(N, N, true, w_);
            arma::uvec idx = Rcpp::as<arma::uvec>(idx_) - 1;

            arma::mat tttmp = theta_stored.slice(t+1);
            theta_stored.slice(t+1) = tttmp.cols(idx);

            arma::vec atmp = a_sigma.col(t+1);
            a_sigma.col(t+1) = atmp.elem(idx);

            arma::vec btmp = b_sigma.col(t+1);
            b_sigma.col(t+1) = btmp.elem(idx);

            arma::vec stmp = sigma2.col(t+1);
            sigma2.col(t+1) = stmp.elem(idx);

            arma::vec atmp2 = a_tau.col(t+1);
            a_tau.col(t+1) = atmp2.elem(idx);

            arma::vec btmp2 = b_tau.col(t+1);
            b_tau.col(t+1) = btmp2.elem(idx);

            arma::vec ttmp = tau2.col(t+1);
            tau2.col(t+1) = ttmp.elem(idx);
        }
        w.ones();


        if (verbose)
        {
            Rcout << "\rFiltering Progress: " << t + 1 << "/" << nt;
        }
    }

    if (verbose)
    {
        Rcout << std::endl;
    }

    Rcpp::List output;
    // arma::mat R = arma::quantile(psi, Rcpp::as<arma::vec>(qProb), 1);
    output["theta"] = Rcpp::wrap(theta_stored); // (n+1) x 3
    output["a_sigma"] = Rcpp::wrap(a_sigma.col(nt));
    output["b_sigma"] = Rcpp::wrap(b_sigma.col(nt));
    output["sigma2"] = Rcpp::wrap(sigma2.col(nt));
    output["a_tau"] = Rcpp::wrap(a_tau.col(nt));
    output["b_tau"] = Rcpp::wrap(b_tau.col(nt));
    output["tau2"] = Rcpp::wrap(tau2.col(nt));

    // if (debug) {
    output["Meff"] = Rcpp::wrap(Meff);
    output["resample_status"] = Rcpp::wrap(resample_status);
    // }

    return output;
}

/*
Use the conditional posterior distribution as proposal.
*/
Rcpp::List pl_gaussian_1d_posterior(
    const arma::vec &Y,                                                    // nt x 1, the observed response
    const arma::uvec &eta_select,                                          // 2 x 1, indicator for unknown (=1) or known (=0) (sigma2,tau2)
    const unsigned int N = 5000,                                           // number of particles
    const Rcpp::Nullable<Rcpp::NumericMatrix> &eta_prior_val = R_NilValue, // 2 x 2, priors for each element of eta: first column for sigma2, second column for tau2
    const Rcpp::Nullable<Rcpp::NumericVector> &eta_true = R_NilValue,      // 2 x 1, if true/initial values should be provided here
    const double m0 = 0.,
    const double C0 = 10.,
    const Rcpp::NumericVector &qProb = Rcpp::NumericVector::create(0.025, 0.5, 0.975),
    const bool verbose = true,
    const bool debug = false)
{

    const unsigned int nt = Y.n_elem; // number of observations
    const unsigned int n = 1;
    const unsigned int p = 1;
    const double min_eff = 0.8 * static_cast<double>(N);

    double C0_sqrt = std::sqrt(C0);

    double sigma2_true, tau2_true;
    if (!eta_true.isNull())
    {
        arma::vec tmp = Rcpp::as<arma::vec>(eta_true);
        if (eta_select.at(0) == 0)
        { // fix sigma2
            sigma2_true = tmp.at(0);
        }
        if (eta_select.at(1) == 0)
        { // fix tau2
            tau2_true = tmp.at(1);
        }
    }

    double a_sigma0 = 0.01;
    double b_sigma0 = 0.01;
    double a_tau0 = 0.01;
    double b_tau0 = 0.01;
    if (!eta_prior_val.isNull())
    {
        arma::mat tmp = Rcpp::as<arma::mat>(eta_prior_val);
        a_sigma0 = tmp.at(0, 0);
        b_sigma0 = tmp.at(1, 0);
        a_tau0 = tmp.at(0, 1);
        b_tau0 = tmp.at(1, 1);
    }

    // first column is [t-1], second column is [t]
    arma::mat a_sigma(N, nt + 1);
    arma::mat b_sigma(N, nt + 1);
    arma::mat sigma2(N, nt + 1);
    for (unsigned int i = 0; i < N; i++)
    {
        a_sigma.at(i, 0) = a_sigma0;
        b_sigma.at(i, 0) = b_sigma0;
        sigma2.at(i, 0) = R::runif(0., 1.);
    }


    arma::mat a_tau(N, nt + 1);
    arma::mat b_tau(N, nt + 1);
    arma::mat tau2(N, nt + 1);
    for (unsigned int i = 0; i < N; i++)
    {
        a_tau.at(i, 0) = a_tau0;
        a_tau.at(i, 0) = b_tau0;
        tau2.at(i, 0) = R::runif(0., 1.);
    }


    arma::vec theta(N);
    for (unsigned int i = 0; i < N; i++)
    {
        theta.at(i) = m0 + C0_sqrt * R::rnorm(0,1.);
    }
    arma::mat theta_stored(N, nt + 1);
    theta_stored.col(0) = theta;

    
    const double mvn_cnst = -0.5 * std::log(2 * arma::datum::pi);
    arma::vec Meff(nt, arma::fill::zeros); // Effective sample size (Ref: Lin, 1996; Prado, 2021, page 196)
    arma::uvec resample_status(nt, arma::fill::zeros);

    for (unsigned int t = 0; t < nt; t++)
    {
        // t: for y means current [t], for others means past [t-1]

        R_CheckUserInterrupt();
        double Yt = Y.at(t);

        arma::vec w(N,arma::fill::zeros); // importance weight of each particle

        for (unsigned int i=0; i<N; i++) {
            double Sigy = tau2.at(i,t) * sigma2.at(i,t) + sigma2.at(i,t);
            double muy = theta.at(i);

            double Sigy_inv = 1./Sigy;
            double err3 = Yt - muy;
            double sse = std::pow(err3,2.) * Sigy_inv;
            double logdet_val_Sigy_inv = std::log(Sigy_inv);
            double loga = mvn_cnst + 0.5 * logdet_val_Sigy_inv - 0.5 * sse;

            double alphat = std::exp(std::min(loga, 700.));
            w.at(i) = alphat;
        }

        /* Resample */
        if (arma::accu(w) > EPS)
        { // normalize the particle weights
            w /= arma::accu(w);
            Meff.at(t) = 1. / arma::dot(w, w);
            if (Meff.at(t) > 0.1 * static_cast<double>(N))
            {
                resample_status.at(t) = 1;
            }
            else
            {
                resample_status.at(t) = 0;
            }
        }
        else
        {
            resample_status.at(t) = 0;
            Meff.at(t) = 0.;
        }

        if (resample_status.at(t) == 1)
        {
            Rcpp::NumericVector w_ = Rcpp::wrap(w);
            Rcpp::IntegerVector idx_ = Rcpp::sample(N, N, true, w_);
            arma::uvec idx = Rcpp::as<arma::uvec>(idx_) - 1;

            theta_stored.col(t) = theta.elem(idx);

            arma::vec atmp = a_sigma.col(t);
            a_sigma.col(t) = atmp.elem(idx);

            arma::vec btmp = b_sigma.col(t);
            b_sigma.col(t) = btmp.elem(idx);

            arma::vec stmp = sigma2.col(t);
            sigma2.col(t) = stmp.elem(idx);

            arma::vec atmp2 = a_tau.col(t);
            a_tau.col(t) = atmp2.elem(idx);

            arma::vec btmp2 = b_tau.col(t);
            b_tau.col(t) = btmp2.elem(idx);

            arma::vec ttmp = tau2.col(t);
            tau2.col(t) = ttmp.elem(idx);
        }
        else
        {
            theta_stored.col(t) = theta;
        }
        w.ones(); // Equal weight to begin with because of resampling in the last iter.

        for (unsigned int i = 0; i < N; i++)
        {
            
            /* Propagate */
            // theta: N, theta_stored: N x (nt+1)
            double theta_old = theta_stored.at(i,t);
            double Sig_inv = 1./sigma2.at(i,t) + 1./ (sigma2.at(i,t)*tau2.at(i,t));
            double Sigma = 1./Sig_inv;

            double tmp1 = Yt / sigma2.at(i, t);
            double tmp2 = theta_old / (sigma2.at(i, t) * tau2.at(i, t));
            double mu = Sigma * (tmp1 + tmp2);

            double theta_new = mu + std::sqrt(Sigma)*R::rnorm(0.,1.);
            theta.at(i) = theta_new;

            double err1 = std::pow(Yt - theta_new,2.);
            double err2 = std::pow(theta_new - theta_old,2.);

            if (eta_select.at(0) == 1)
            {
                // sample sigma2
                a_sigma.at(i, t + 1) = a_sigma.at(i, t) + 1.;
                b_sigma.at(i, t + 1) = b_sigma.at(i, t) + 0.5*err2/tau2.at(i,t) + 0.5*err1;

                sigma2.at(i, t + 1) = 1. / R::rgamma(a_sigma.at(i, t + 1), 1. / b_sigma.at(i, t + 1));
                // R::rgamma(shape, scale)
            }
            else
            {
                // fix sigma2
                sigma2.at(i, t + 1) = sigma2_true;
            }

            if (eta_select.at(1) == 1)
            {
                // infer tau2
                a_tau.at(i, t + 1) = a_tau.at(i, t) + 0.5;
                b_tau.at(i, t + 1) = b_tau.at(i, t) + 0.5 * err2/sigma2.at(i,t+1);
                tau2.at(i, t + 1) = 1. / R::rgamma(a_tau.at(i, t + 1), 1. / b_tau.at(i, t + 1));
                // R::rgamma(shape, scale)
            }
            else
            {
                // fix tau2
                tau2.at(i, t + 1) = tau2_true;
            }
        }

        

        if (verbose)
        {
            Rcout << "\rFiltering Progress: " << t + 1 << "/" << nt;
        }
    }

    if (verbose)
    {
        Rcout << std::endl;
    }

    Rcpp::List output;
    // arma::mat R = arma::quantile(psi, Rcpp::as<arma::vec>(qProb), 1);
    output["theta"] = Rcpp::wrap(theta_stored); // (n+1) x 3
    output["a_sigma"] = Rcpp::wrap(a_sigma.col(nt));
    output["b_sigma"] = Rcpp::wrap(b_sigma.col(nt));
    output["sigma2"] = Rcpp::wrap(sigma2.col(nt));
    output["a_tau"] = Rcpp::wrap(a_tau.col(nt));
    output["b_tau"] = Rcpp::wrap(b_tau.col(nt));
    output["tau2"] = Rcpp::wrap(tau2.col(nt));

    // if (debug) {
    output["Meff"] = Rcpp::wrap(Meff);
    output["resample_status"] = Rcpp::wrap(resample_status);
    // }

    return output;
}

/*
Use the evolution distribution as proposal distribution
*/
Rcpp::List pl_gaussian_1d_evolution(
    const arma::vec &Y,                                                    // nt x 1, the observed response
    const arma::uvec &eta_select,                                          // 2 x 1, indicator for unknown (=1) or known (=0) (sigma2,tau2)
    const unsigned int N = 5000,                                           // number of particles
    const Rcpp::Nullable<Rcpp::NumericMatrix> &eta_prior_val = R_NilValue, // 2 x 2, priors for each element of eta: first column for sigma2, second column for tau2
    const Rcpp::Nullable<Rcpp::NumericVector> &eta_true = R_NilValue,      // 2 x 1, if true/initial values should be provided here
    const double m0 = 0.,
    const double C0 = 10.,
    const Rcpp::NumericVector &qProb = Rcpp::NumericVector::create(0.025, 0.5, 0.975),
    const bool verbose = true,
    const bool debug = false)
{

    const unsigned int nt = Y.n_elem; // number of observations
    const unsigned int n = 1;
    const unsigned int p = 1;
    const double min_eff = 0.8 * static_cast<double>(N);

    double C0_sqrt = std::sqrt(C0);

    double sigma2_true, tau2_true;
    if (!eta_true.isNull())
    {
        arma::vec tmp = Rcpp::as<arma::vec>(eta_true);
        if (eta_select.at(0) == 0)
        { // fix sigma2
            sigma2_true = tmp.at(0);
        }
        if (eta_select.at(1) == 0)
        { // fix tau2
            tau2_true = tmp.at(1);
        }
    }

    double a_sigma0 = 0.01;
    double b_sigma0 = 0.01;
    double a_tau0 = 0.01;
    double b_tau0 = 0.01;
    if (!eta_prior_val.isNull())
    {
        arma::mat tmp = Rcpp::as<arma::mat>(eta_prior_val);
        a_sigma0 = tmp.at(0, 0);
        b_sigma0 = tmp.at(1, 0);
        a_tau0 = tmp.at(0, 1);
        b_tau0 = tmp.at(1, 1);
    }

    // first column is [t-1], second column is [t]
    arma::mat a_sigma(N, nt + 1);
    arma::mat b_sigma(N, nt + 1);
    arma::mat sigma2(N, nt + 1);
    // sigma2.col(0).fill(R::runif(0.,1.));
    // sigma2.col(0) = 1. / arma::randg(N, arma::distr_param(a_sigma0, 1. / b_sigma0));
    for (unsigned int i = 0; i < N; i++)
    {
        a_sigma.at(i, 0) = a_sigma0;
        b_sigma.at(i, 0) = b_sigma0;
        sigma2.at(i, 0) = R::runif(0., 1.);
    }
    // Rcout << sigma2.col(0).t() << std::endl;

    // Rcout << "a_sigma0 = " << a_sigma0 << ", b_sigma0 = " << b_sigma0
    // sigma2.col(1) = 1. / arma::randg(N, arma::distr_param(a_sigma.at(0, 1), 1. / b_sigma.at(0, 1)));

    arma::mat a_tau(N, nt + 1);
    arma::mat b_tau(N, nt + 1);
    arma::mat tau2(N, nt + 1);
    for (unsigned int i = 0; i < N; i++)
    {
        a_tau.at(i, 0) = a_tau0;
        a_tau.at(i, 0) = b_tau0;
        tau2.at(i, 0) = R::runif(0., 1.);
    }
    // tau2.col(0).fill(R::runif(0., 1.));
    // tau2.col(0) = 1. / arma::randg(N, arma::distr_param(a_tau0, 1. / b_tau0));
    // tau2.col(1) = 1. / arma::randg(N, arma::distr_param(a_tau.at(0, 1), 1. / b_tau.at(0, 1)));

    arma::vec theta(N);
    for (unsigned int i = 0; i < N; i++)
    {
        theta.at(i) = m0 + C0_sqrt * R::rnorm(0, 1.);
    }
    arma::mat theta_stored(N, nt + 1);
    theta_stored.col(0) = theta;

    const double mvn_cnst = -0.5 * std::log(2 * arma::datum::pi);
    arma::vec Meff(nt, arma::fill::zeros); // Effective sample size (Ref: Lin, 1996; Prado, 2021, page 196)
    arma::uvec resample_status(nt, arma::fill::zeros);

    for (unsigned int t = 0; t < nt; t++)
    {
        // t: for y means current [t], for others means past [t-1]

        R_CheckUserInterrupt();
        double Yt = Y.at(t);
        double Yt_old = 0.;
        if (t>0) {
            Yt_old = Y.at(t - 1);
        }
        
        /* Propagate */
        for (unsigned int i = 0; i < N; i++)
        {
            
            // theta: N, theta_stored: N x (nt+1)
            double theta_old = theta_stored.at(i, t);
            double Sigma = sigma2.at(i,t)*tau2.at(i,t);
            double mu = theta_old;

            double theta_new = mu + std::sqrt(Sigma) * R::rnorm(0., 1.);
            theta_stored.at(i,t+1) = theta_new;

            double err1 = std::pow(Yt_old - theta_old, 2.);
            double err2 = std::pow(theta_new - theta_old, 2.);

            if (eta_select.at(0) == 1)
            {
                // sample sigma2
                a_sigma.at(i, t+1) = a_sigma.at(i, t) + 1.;
                b_sigma.at(i, t+1) = b_sigma.at(i, t) + 0.5*err1 + 0.5 * err2 / tau2.at(i, t);
                sigma2.at(i,t+1) = 1./R::rgamma(a_sigma.at(i,t+1), 1. / b_sigma.at(i,t+1));
                // R::rgamma(shape, scale)
            }
            else
            {
                // fix sigma2
                sigma2.at(i, t + 1) = sigma2_true;
            }

            if (eta_select.at(1) == 1)
            {
                // infer tau2
                a_tau.at(i, t+1) = a_tau.at(i, t) + 0.5;
                b_tau.at(i, t+1) = b_tau.at(i, t) + 0.5 * err2 / sigma2.at(i, t + 1);
                tau2.at(i,t+1) = 1./R::rgamma(a_tau.at(i,t+1), 1. / b_tau.at(i,t+1));
                // R::rgamma(shape, scale)
            }
            else
            {
                // fix tau2
                tau2.at(i, t + 1) = tau2_true;
            }
        } // Eng Propagation

        /* Resample */
        arma::vec w(N, arma::fill::zeros); // importance weight of each particle
        for (unsigned int i = 0; i < N; i++)
        {
            double Sigy = sigma2.at(i,t+1);
            double muy = theta_stored.at(i,t+1);

            double Sigy_inv = 1. / Sigy;
            double err3 = Yt - muy;
            double sse = std::pow(err3, 2.) * Sigy_inv;
            double logdet_val_Sigy_inv = std::log(Sigy_inv);
            double loga = mvn_cnst + 0.5 * logdet_val_Sigy_inv - 0.5 * sse;

            double alphat = std::exp(std::min(loga, 700.));
            w.at(i) = alphat;
        }

        
        if (arma::accu(w) > EPS)
        { // normalize the particle weights
            w /= arma::accu(w);
            Meff.at(t) = 1. / arma::dot(w, w);
            if (Meff.at(t) > 0.1 * static_cast<double>(N))
            {
                resample_status.at(t) = 1;
            }
            else
            {
                resample_status.at(t) = 0;
            }
        }
        else
        {
            resample_status.at(t) = 0;
            Meff.at(t) = 0.;
        }

        if (resample_status.at(t) == 1)
        {
            Rcpp::NumericVector w_ = Rcpp::wrap(w);
            Rcpp::IntegerVector idx_ = Rcpp::sample(N, N, true, w_);
            arma::uvec idx = Rcpp::as<arma::uvec>(idx_) - 1;

            arma::mat vvec = theta_stored.col(t+1);
            theta_stored.col(t+1) = vvec.elem(idx);

            arma::vec atmp = a_sigma.col(t+1);
            a_sigma.col(t+1) = atmp.elem(idx);

            arma::vec btmp = b_sigma.col(t+1);
            b_sigma.col(t+1) = btmp.elem(idx);

            arma::vec stmp = sigma2.col(t+1);
            sigma2.col(t+1) = stmp.elem(idx);

            arma::vec atmp2 = a_tau.col(t+1);
            a_tau.col(t+1) = atmp2.elem(idx);

            arma::vec btmp2 = b_tau.col(t+1);
            b_tau.col(t+1) = btmp2.elem(idx);

            arma::vec ttmp = tau2.col(t+1);
            tau2.col(t+1) = ttmp.elem(idx);
        }

        w.ones(); // Equal weight to begin with because of resampling in the last iter.
        // End Resample

        if (verbose)
        {
            Rcout << "\rFiltering Progress: " << t + 1 << "/" << nt;
        }
    }

    if (verbose)
    {
        Rcout << std::endl;
    }

    Rcpp::List output;
    // arma::mat R = arma::quantile(psi, Rcpp::as<arma::vec>(qProb), 1);
    output["theta"] = Rcpp::wrap(theta_stored); // (n+1) x 3
    output["a_sigma"] = Rcpp::wrap(a_sigma.col(nt));
    output["b_sigma"] = Rcpp::wrap(b_sigma.col(nt));
    output["sigma2"] = Rcpp::wrap(sigma2.col(nt));
    output["a_tau"] = Rcpp::wrap(a_tau.col(nt));
    output["b_tau"] = Rcpp::wrap(b_tau.col(nt));
    output["tau2"] = Rcpp::wrap(tau2.col(nt));

    // if (debug) {
    output["Meff"] = Rcpp::wrap(Meff);
    output["resample_status"] = Rcpp::wrap(resample_status);
    // }

    return output;
}
