#include "pl_poisson.h"
using namespace Rcpp;
// [[Rcpp::plugins(cpp17)]]
// [[Rcpp::depends(RcppArmadillo,nloptr)]]

//' @export
// [[Rcpp::export]]
Rcpp::List mcs_poisson0(
    const arma::vec &Y, // n x 1, the observed response
    const arma::uvec &model_code,
    const double W = NA_REAL, // Use discount factor if W is not given
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
    const double delta_discount = 0.95,
    const bool verbose = false,
    const bool debug = false)
{
    const double alpha = 1.;
    const Rcpp::NumericVector ctanh = {1., 0., 1.}; // (1./M, 0, M)
    bool resample = false;

    const unsigned int n = Y.n_elem; // number of observations
    const double min_eff = 0.8 * static_cast<double>(N);
    arma::vec ypad(n + 1, arma::fill::zeros);
    ypad.tail(n) = Y;


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
    init_by_trans(p, L_, Ft, Fphi, Gt, trans_code, L);

    /* Dimension of state space depends on type of transfer functions */

    // theta: DLM state vector
    // p - dimension of state space
    // N - number of particles
    arma::vec lambda(N); // instantaneous intensity
    arma::vec omega(N);  // evolution variance
    arma::vec w(N);      // importance weight of each particle
    Rcpp::IntegerVector idx_(N);
    arma::uvec idx(N);


    arma::mat theta(p, N);
    
    arma::mat hpsi;
    // if (link_code==1) {
    // Exponential Link
    arma::vec m0(p, arma::fill::zeros);
    arma::mat C0(p, p, arma::fill::eye);
    C0 *= std::pow(theta0_upbnd * 0.5, 2.);
    if (!m0_prior.isNull())
    {
        m0 = Rcpp::as<arma::vec>(m0_prior);
    }
    // if (!C0_prior.isNull()) {
    //     C0 = Rcpp::as<arma::mat>(C0_prior);
    // }
    C0 = arma::chol(C0);
    for (unsigned int i = 0; i < N; i++)
    {
        theta.col(i) = m0 + C0.t() * arma::randn(p);
    }

    // } else {
    //     // Identity Link
    //     theta = arma::randu(p,N,arma::distr_param(0.,theta0_upbnd)); // Consider it as a flat prior
    // }
    arma::cube theta_stored(p, N, n + B);
    for (unsigned int b = 0; b < B; b++)
    {
        theta_stored.slice(b).zeros();
    }
    theta_stored.slice(B - 1) = theta;
    /*
    ------ Step 1. Initialization theta[0,] at time t = 0 ------
    */

    arma::vec Meff(n, arma::fill::zeros); // Effective sample size (Ref: Lin, 1996; Prado, 2021, page 196)
    arma::uvec resample_status(n, arma::fill::zeros);
    arma::vec Wt(n);
    if (!R_IsNA(W)) {
        Wt.fill(W);
    }

    for (unsigned int t = 0; t < n; t++)
    {
        R_CheckUserInterrupt();
        // theta_stored: p x N x B, with B is the lag of the B-lag fixed-lag smoother
        // theta: p x N
        arma::mat Theta_old = theta_stored.slice(t + B - 1);


        if (trans_code != 1) {
            arma::vec mt;
            if (t > B)
            {
                // Is it necessary?
                mt = 0.5 * arma::mean(Theta_old, 1) + 0.5 * arma::median(Theta_old, 1);
            }
            else
            {
                mt = arma::median(Theta_old, 1);
            }
            update_Gt(Gt, gain_code, trans_code, mt, ctanh, alpha, ypad.at(t), rho);
        }
        

        theta = update_at(p, gain_code, trans_code, Theta_old, Gt, ctanh, alpha, ypad.at(t), rho);

        
        if (R_IsNA(W))
        { // Use discount factor if W is not given
            Wt.at(t) = arma::var(Theta_old.row(0));
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

        omega = arma::randn(N) * Wsqrt;
        theta.row(0) += omega.t();

        if (debug)
        {
            Rcout << "quantiles for hpsi[" << t + 1 << "]" << arma::quantile(theta.row(1), Rcpp::as<arma::vec>(qProb));
        }

        // theta_stored.slices(0, B - 2) = theta_stored.slices(1, B - 1);
        theta_stored.slice(t+B) = theta;
        /*
        ------ Step 2.1 Propagate ------
        */
        if (trans_code == 1)
        { // Koyama
            arma::vec Fy(p,arma::fill::zeros);
            unsigned int tmpi = std::min(t, p);
            if (t > 0)
            {
                // Checked Fy - CORRECT
                Fy.head(tmpi) = arma::reverse(ypad.subvec(t + 1 - tmpi, t));
            }

            Ft = Fphi % Fy; // L(p) x 1
        }

        if (link_code == 1)
        {
            // Exponential link and identity gain
            for (unsigned int i = 0; i < N; i++)
            {
                lambda.at(i) = mu0 + arma::as_scalar(Ft.t() * theta.col(i));
            }
            lambda.elem(arma::find(lambda > UPBND)).fill(UPBND);
            lambda = arma::exp(lambda);
        }
        else if (trans_code == 1)
        {                                             // Koyama
            hpsi = psi2hpsi(theta, gain_code, ctanh); // hpsi: p x N
            for (unsigned int i = 0; i < N; i++)
            {
                lambda.at(i) = mu0 + arma::as_scalar(Ft.t() * hpsi.col(i));
            }
        }
        else
        {
            // Koyck or Solow with identity link and different gain functions
            lambda = mu0 + theta.row(1).t();
        }

        if (debug)
        {
            Rcout << "Quantiles of lambda[" << t + 1 << "]: " << arma::quantile(lambda.t(), Rcpp::as<arma::vec>(qProb));
        }

        for (unsigned int i = 0; i < N; i++)
        {
            w.at(i) = loglike_obs(ypad.at(t + 1), lambda.at(i), obs_code, delta_nb, false);
        } // End for loop of i, index of the particles

        if (debug)
        {
            Rcout << "Quantiles of importance weight w[" << t + 1 << "]: " << arma::quantile(w.t(), Rcpp::as<arma::vec>(qProb));
        }
        /*
        ------ Step 2.2 Importance weights ------
        */

        /*
        ------ Step 3 Resampling with Replacement ------
        */
        if (arma::accu(w) > EPS)
        { // normalize the particle weights
            w /= arma::accu(w);
            Meff.at(t) = 1. / arma::dot(w, w);
            try
            {
                idx_ = Rcpp::sample(N, N, true, Rcpp::as<Rcpp::NumericVector>(Rcpp::wrap(w)));
                idx = Rcpp::as<arma::uvec>(idx_) - 1;
                for (unsigned int b = t+1; b < (t+B+1); b++)
                {
                    theta_stored.slice(b) = theta_stored.slice(b).cols(idx);
                }
                resample_status.at(t) = 1;
            }
            catch (...)
            {
                // If resampling doesn't work, then just don't resample
                resample_status.at(t) = 0;
            }
        }
        else
        {
            resample_status.at(t) = 0;
            Meff.at(t) = 0.;
        }

        if (debug && resample_status.at(t) == 0)
        {
            Rcout << "Resampling skipped at time t=" << t << std::endl;
            // ::Rf_error("Probabilities must be finite and non-negative!");
        }

        /*
        ------ Step  Resampling with Replacement ------
        */
        // R.row(t+B) = arma::quantile(theta_stored.slice(0).row(0), Rcpp::as<arma::vec>(qProb));
    }


    // R.rows(0, n - B + 1) = R.rows(B - 2, n - 1);
    // for (unsigned int b = 0; b < (B - 1); b++)
    // {
    //     R.row(n - B + 2 + b) = arma::quantile(theta_stored.slice(b + 1).row(0), Rcpp::as<arma::vec>(qProb));
    // }

    Rcpp::List output;
    // output["psi"] = Rcpp::wrap(R);   
    output["theta"] = Rcpp::wrap(theta_stored);                             // (n+1) x 3
    output["theta_last"] = Rcpp::wrap(theta_stored.slice(B - 1)); // p x N
    output["Wt"] = Rcpp::wrap(Wt);

    if (debug)
    {
        output["Meff"] = Rcpp::wrap(Meff);
        output["resample_status"] = Rcpp::wrap(resample_status);
    }

    return output;
}
