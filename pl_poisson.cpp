#include "pl_poisson.h"
using namespace Rcpp;
// [[Rcpp::plugins(cpp17)]]
// [[Rcpp::depends(RcppArmadillo,nloptr)]]





void init_Ft(
    arma::vec &Ft, 
    const arma::vec &ypad, 
    const arma::vec &Fphi,
    const unsigned int &t, 
    const unsigned int &p)
{
    arma::vec Fy(p, arma::fill::zeros);
    unsigned int tmpi = std::min(t, p);
    if (t > 0)
    {
        // Checked Fy - CORRECT
        Fy.head(tmpi) = arma::reverse(ypad.subvec(t + 1 - tmpi, t));
    }

    Ft = Fphi % Fy; // L(p) x 1
}


//' @export
// [[Rcpp::export]]
Rcpp::List mcs_poisson(
    const arma::vec &Y, // nt x 1, the observed response
    const arma::uvec &model_code,
    const double &W = 0.01, // (init, prior type, prior par1, prior par2)
    const Rcpp::NumericVector &obs_par_in = Rcpp::NumericVector::create(0., 30.),
    const Rcpp::NumericVector &lag_par_in = Rcpp::NumericVector::create(0.5, 6), // init/true values of (mu, sg2) or (rho, L)
    const unsigned int &nlag_in = 20,
    const unsigned int &B = 20,                                       // length of the B-lag fixed-lag smoother (Anderson and Moore 1979; Kitagawa and Sato)
    const unsigned int &N = 5000,                                     // number of particles
    const double &theta0_upbnd = 2.,                                  // Upper bound of uniform prior for theta0
    const Rcpp::NumericVector &qProb = Rcpp::NumericVector::create(0.025, 0.5, 0.975),
    const double &delta_discount = 0.88,
    const bool &truncated = true,
    const bool &use_discount = false)
{

    const unsigned int nt = Y.n_elem; // number of observations
    arma::vec ypad(nt + 1, arma::fill::zeros);
    ypad.tail(nt) = Y;

    arma::vec lag_par(lag_par_in.begin(), lag_par_in.length());
    arma::vec obs_par(obs_par_in.begin(), obs_par_in.length());

    double mu0 = obs_par.at(0);
    double delta_nb = obs_par.at(1);

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


    unsigned int nlag = nlag_in;
    unsigned int p = nlag;
    if (!truncated)
    {
        nlag = nt;
        p = (unsigned int) lag_par.at(1) + 1;
    }
    arma::vec Fphi = get_Fphi(nlag, lag_par, trans_code);
    arma::vec Ft = init_Ft(p, trans_code);

    arma::mat Gt;
    init_Gt(Gt, lag_par, p, nlag, truncated);

    /* Dimension of state space depends on type of transfer functions */

        
    
    arma::vec m0(p,arma::fill::zeros);
    double C0_sqrt = theta0_upbnd*0.5;

    arma::vec theta_init = arma::randn(N);
    theta_init.for_each([&m0, &C0_sqrt](arma::vec::elem_type &val){ 
        val = std::abs(m0.at(0) + C0_sqrt * val);});

    arma::cube theta_stored(p, N, nt + B);
    for (unsigned int b=0; b<(nt+B); b++) {
        theta_stored.slice(b).zeros();
    }
    theta_stored.slice(B-1).row(1) = theta_init.t();


    arma::vec Wt(nt); Wt.fill(W);

    arma::vec Meff(nt, arma::fill::zeros); 
    // Effective sample size (Ref: Lin, 1996; Prado, 2021, page 196)
    arma::uvec resample_status(nt,arma::fill::zeros);
    arma::mat R(nt+1,3,arma::fill::zeros);
    arma::mat weights_stored(N,nt+1);
    arma::vec pmarg_y(nt);


    for (unsigned int t=0; t<nt; t++) 
    {
        R_CheckUserInterrupt();

        if (truncated)
        { // truncated
            init_Ft(Ft,ypad,Fphi,t,p);
        }

        bool use_custom_val = false;
        if (t > B)
        {
            use_custom_val = true;
        }
        double y_old = ypad.at(t);
        arma::mat Theta_old = theta_stored.slice(t + B - 1); // p x N

        arma::vec weights(N, arma::fill::ones);
        double wt = Wt.at(t);
        double y_new = ypad.at(t + 1);
        arma::mat Theta_new(p, N, arma::fill::zeros);

        smc_propagate_bootstrap(
            Theta_new, weights, wt, Gt,
            y_new, y_old, Theta_old, Ft, model_code,
            obs_par, lag_par, p, t, nlag, N, delta_discount,
            truncated, use_discount, use_custom_val);

        pmarg_y.at(t) = std::log(arma::accu(weights) + EPS) - std::log(static_cast<double>(N));
        Wt.at(t) = wt;

        theta_stored.slice(t+B) = Theta_new;
        bool resample;
        double meff = 0;
        smc_resample(theta_stored,weights,meff,resample,t,B);

        weights_stored.col(t) = weights;
        resample_status.at(t) = resample;
        Meff.at(t) = meff;
        
    }

    Rcpp::List output;
    arma::mat psi_all = theta_stored.row(0); // N x (nt+B)
    arma::mat psi = psi_all.cols(B-1, nt+B-1); // N x nt
    arma::mat RR = arma::quantile(psi, Rcpp::as<arma::vec>(qProb), 0); // 0: quantile for each column vector
    output["psi"] = Rcpp::wrap(RR.t());
    output["psi_all"] = Rcpp::wrap(psi_all);

    // arma::mat hpsiR = psi2hpsi(RR, gain_code); // hpsi: p x N
    // double theta0 = 0;
    // arma::mat thetaR = hpsi2theta(hpsiR, Y, trans_code, theta0, L, rho); // n x 1

    // output["hpsi"] = Rcpp::wrap(hpsiR);
    // output["theta"] = Rcpp::wrap(thetaR);

    // output["theta"] = Rcpp::wrap(t÷heta_stored); // p x N
    output["Wt"] = Rcpp::wrap(Wt);

    output["Meff"] = Rcpp::wrap(Meff);
    output["resample_status"] = Rcpp::wrap(resample_status);
    // output["weights"] = Rcpp::wrap(weights_stored);
    output["marg_y"] = Rcpp::wrap(pmarg_y);
    
    return output;
}

/**
 * Bootstrap filter or Auxiliary Particle Filter 
 * with fixed lag smoothing.
 * For a single time point.
 * from theta[t] (past) to theta[t+1] (unknown)
 * Relies on `update_at` from `lbe_poisson.h`.
*/
void smc_propagate_bootstrap(
    arma::mat &Theta_new, // p x N
    arma::vec &weights,   // N x 1
    double &wt,
    arma::mat &Gt,
    const double &y_new,
    const double &y_old,        // (n+1) x 1, the observed response
    const arma::mat &Theta_old, // p x N
    const arma::vec &Ft,        // must be already updated if used
    const arma::uvec &model_code,
    const arma::vec &obs_par,
    const arma::vec &lag_par,
    const unsigned int &p, // dimension of DLM state space
    const int &t,
    const unsigned int &nlag,
    const unsigned int &N, // number of particles
    const double &delta_discount,
    const bool &truncated,
    const bool &use_discount,
    const bool &use_custom_val)
{
    const unsigned int obs_code = model_code.at(0);
    const unsigned int link_code = model_code.at(1);
    const unsigned int gain_code = model_code.at(3);

    double mu0 = obs_par.at(0);
    double delta_nb = obs_par.at(1);


    /*
        ------ Step 2.1 Propagate ------
        */
    // // theta_stored: p,N,B
    if (use_discount)
    { // Use discount factor if W is not given
        arma::rowvec psi = Theta_old.row(0);
        double tmp = arma::var(psi);
        if (tmp > EPS)
        {
            wt = tmp;
        }
        else
        {
            wt = 1.;
        }
        // Wsqrt = std::sqrt(Wt.at(t));
        if (use_custom_val)
        {
            wt *= 1. / delta_discount - 1.;
        }
        else
        {
            wt *= 1. / 0.99 - 1.;
        }
    }
    double Wsqrt = std::sqrt(wt) + EPS;
    if (wt < EPS)
    {
        throw std::invalid_argument("smc_propagate_bootstrap: variance wt closed to 0.");
    }

    for (unsigned int i = 0; i < N; i++)
    {
        arma::vec theta_old = Theta_old.col(i); // p x 1
        // update_Gt(Gt, gain_code, trans_code, theta_old, y_old, rho);
        arma::mat theta_new = update_at(
            Gt, theta_old, y_old, lag_par, gain_code, nlag, truncated);

        double omega_new = R::rnorm(0., Wsqrt);
        if (t < p)
        {
            theta_new.at(0) += std::abs(omega_new);
        }
        else
        {
            theta_new.at(0) += omega_new;
        }

        
        Theta_new.col(i) = theta_new;
    }


    /*
    ------ Step 2.2 Importance weights ------
    */
    arma::vec lambda(N);
    for (unsigned int i = 0; i < N; i++)
    {
        arma::vec theta = Theta_new.col(i); // p x 1

        // theta: p x N
        // double lambda = 0.;
        if (link_code == 1)
        {
            // Exponential link and identity gain
            double tmp = arma::as_scalar(Ft.t() * theta);
            lambda.at(i) = std::exp(mu0 + tmp);
        }
        else
        {
            double ft;
            if (truncated)
            {
                arma::vec hpsi = psi2hpsi(theta, gain_code);
                ft = arma::as_scalar(Ft.t() * hpsi);
                // lambda.at(i) = mu0 + ft;
            }
            else
            {
                ft = theta.at(1);
                // Koyck or Solow with identity link and different gain functions
                // lambda.at(i) = mu0 + theta.at(1);
            }

            bound_check(ft,"smc_propagate_bootstrap: ft",false,true);
            lambda.at(i) = mu0 + std::abs(ft);
        }


        double ws = loglike_obs(y_new, lambda.at(i), obs_code, delta_nb, false);
        weights.at(i) = ws;
    }


    return;
}

void smc_resample(
    arma::cube &theta_stored,   // p x N x (nt + B)
    arma::vec &weights,
    double &meff,
    bool &resample,
    const unsigned int &t,
    const unsigned int &B)
{
    unsigned int N = weights.n_elem;
    double N_ = static_cast<double>(N);
    double wsum = arma::accu(weights);
    bound_check(wsum, "smc_resample: wsum");


    resample = wsum > EPS;
    meff = 0.;

    arma::uvec idx;
    if (resample)
    {
        resample = true;
        weights.for_each([&wsum](arma::vec::elem_type &val)
                         { val /= wsum; });
        meff = 1. / arma::dot(weights, weights);
        idx = sample(N, N, weights, true, true);

        for (unsigned int b = t + 1; b < t + B + 1; b++)
        {
            arma::mat tmp_old = theta_stored.slice(b);
            arma::mat tmp_resampled = tmp_old.cols(idx);
            theta_stored.slice(b) = tmp_resampled.cols(idx);
        }
    } else {
        weights.ones();
        weights.for_each([&N_](arma::vec::elem_type &val){ val/= N_;});
    }

    return;
}

//' @export
// [[Rcpp::export]]
Rcpp::List ffbs_poisson(
    const arma::vec &Y, // n x 1, the observed response
    const arma::uvec &model_code,
    const Rcpp::NumericVector &W_par_in = Rcpp::NumericVector::create(0.01, 2, 0.01, 0.01), // (init, prior type, prior par1, prior par2)
    const Rcpp::NumericVector &obs_par_in = Rcpp::NumericVector::create(0., 30.),
    const Rcpp::NumericVector &lag_par_in = Rcpp::NumericVector::create(0.5, 6), // init/true values of (mu, sg2) or (rho, L)
    const unsigned int &nlag_in = 20,
    const unsigned int &N = 5000, // number of particles
    const double &theta0_upbnd = 2.,
    const Rcpp::NumericVector &qProb = Rcpp::NumericVector::create(0.025, 0.5, 0.975),
    const double &delta_discount = 0.95,
    const bool &truncated = true,
    const bool &use_discount = false,
    const bool &smoothing = true)
{

    const unsigned int nt = Y.n_elem; // number of observations
    arma::vec ypad(nt + 1, arma::fill::zeros);
    ypad.tail(nt) = Y;

    arma::vec W_par(W_par_in.begin(), W_par_in.length());
    arma::vec lag_par(lag_par_in.begin(), lag_par_in.length());
    arma::vec obs_par(obs_par_in.begin(), obs_par_in.length());

    double mu0 = obs_par.at(0);
    double delta_nb = obs_par.at(1);
    double m0 = 0.;


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

    double par1 = lag_par[0];
    unsigned int par2 = lag_par[1];

    unsigned int nlag = nlag_in;
    unsigned int p = nlag;
    if (!truncated)
    {
        nlag = nt;
        p = par2 + 1;
    }
    arma::vec Fphi = get_Fphi(nlag, lag_par, trans_code);
    arma::vec Ft = init_Ft(p, trans_code);
    
    unsigned int B = p;

    arma::mat Gt;
    init_Gt(Gt, lag_par, p, nlag, truncated);

    /* Dimension of state space depends on type of transfer functions */

    double C0_sqrt = theta0_upbnd * 0.5;
    arma::vec theta_init = arma::randn(N);
    theta_init.for_each([&m0, &C0_sqrt](arma::vec::elem_type &val)
                        { val = std::abs(m0 + C0_sqrt * val); });

    arma::cube theta_stored(p, N, nt + 1);
    for (unsigned int b = 0; b < (nt + 1); b++)
    {
        theta_stored.slice(b).zeros();
    }
    theta_stored.slice(0).row(1) = theta_init.t();

    double W = W_par[0];
    arma::vec Wt(nt);
    Wt.fill(W);
    /*
    ------ Step 1. Initialization theta[0,] at time t = 0 ------
    */

   arma::vec Meff(nt, arma::fill::zeros);
   // Effective sample size (Ref: Lin, 1996; Prado, 2021, page 196)
   arma::uvec resample_status(nt, arma::fill::zeros);
   arma::mat weights_stored(N, nt + 1);

   for (unsigned int t = 0; t < nt; t++)
   {
        R_CheckUserInterrupt();

        if (nlag < nt)
        { // truncated
            init_Ft(Ft, ypad, Fphi, t, p);
        }

        bool use_custom_val = false;
        if (t > B)
        {
            use_custom_val = true;
        }

        double y_old = ypad.at(t);
        arma::mat Theta_old = theta_stored.slice(t); // p x N

        arma::vec weights(N, arma::fill::ones);
        double wt = Wt.at(t);

        double y_new = ypad.at(t + 1);
        arma::mat Theta_new(p, N, arma::fill::zeros);

        smc_propagate_bootstrap(
            Theta_new, weights, wt, Gt,
            y_new, y_old, Theta_old, Ft, model_code,
            obs_par, lag_par, p, t, nlag, N, delta_discount,
            truncated, use_discount, use_custom_val);

        theta_stored.slice(t + 1) = Theta_new;
        Wt.at(t) = wt;
        bool resample;
        double meff = 0;
        smc_resample(theta_stored, weights, meff, resample, t, 1);

        weights_stored.col(t) = weights;
        resample_status.at(t) = resample;
        Meff.at(t) = meff;

        // if (verbose)
        {
            Rcout << "\rFiltering Progress: " << t + 1 << "/" << nt;
        }
    }

    Rcout << std::endl;


    /*
    ------ Particle Smoothing ------
    */
    arma::mat psi_smooth(N, nt + 1);
    arma::mat psi_filter(N, nt + 1);
    psi_filter.col(nt) = theta_stored.slice(nt).row(0).t();
    for (unsigned int t = nt; t > 0; t--)
    {
        psi_filter.col(t-1) = theta_stored.slice(t-1).row(0).t();
    }

    psi_smooth.col(nt) = theta_stored.slice(nt).row(0).t();
    if (smoothing) {
        for (unsigned int t = nt; t > 0; t--)
        {
            arma::vec w(N);          
            double Wsqrt = std::sqrt(Wt.at(t-1));
            for (unsigned int i=0; i<N; i++) {
                w = -0.5*arma::pow((psi_smooth.at(i,t)-psi_filter.col(t-1))/Wsqrt,2);
                w.elem(arma::find(w>UPBND)).fill(UPBND);
                w = arma::exp(w);
                bound_check(w, "ffbs_poisson: w in smoothing");

                if (arma::accu(w)<EPS) {
                    w.ones();
                }
                w /= arma::accu(w);
                Rcpp::NumericVector tmpir(1);
                tmpir = Rcpp::sample(N, 1, true, Rcpp::as<Rcpp::NumericVector>(Rcpp::wrap(w))) - 1;

                // tmpir = Rcpp::sample(N,1,true,Rcpp::as<Rcpp::NumericVector>(Rcpp::wrap(w))) - 1;
                psi_smooth.at(i,t-1) = psi_filter.at(tmpir[0],t-1);
            }

            // if (verbose)
            {
                Rcout << "\rSmoothing Progress: " << nt - t + 1 << "/" << nt;
            }
        }

        Rcout << std::endl;
    }
    
    


    Rcpp::List output;

    if (smoothing) {
        arma::mat R = arma::quantile(psi_smooth,Rcpp::as<arma::vec>(qProb),0); // 3 columns, smoothed psi
        output["psi"] = Rcpp::wrap(R.t());
        // output["psi_all"] = Rcpp::wrap(psi_smooth);

        // arma::mat hpsiR = psi2hpsi(R, gain_code); // hpsi: p x N
        // double theta0 = 0;
        // arma::mat thetaR = hpsi2theta(hpsiR, Y, trans_code, theta0, L, rho); // n x 1

        // output["hpsi"] = Rcpp::wrap(hpsiR);
        // output["theta"] = Rcpp::wrap(thetaR);
    } 

    arma::mat R2 = arma::quantile(psi_filter, Rcpp::as<arma::vec>(qProb), 0);
    output["psi_filter"] = Rcpp::wrap(R2.t());

    // output["psi_filter_all"] = Rcpp::wrap(psi_filter);
    output["Meff"] = Rcpp::wrap(Meff);
    output["resample_status"] = Rcpp::wrap(resample_status);
    // output["weights"] = Rcpp::wrap(weights_stored);
    output["Wt"] = Rcpp::wrap(Wt);



    // add RMSE and MAE between lambda[t] and y[t]
    
    // output["rmse"] = std::sqrt(arma::as_scalar(arma::mean(arma::pow(thetaR.tail(nt) - Y.tail(nt), 2.0))));
    // output["mae"] = arma::as_scalar(arma::mean(arma::abs(thetaR.tail(nt) - Y.tail(nt))));

    return output;
}

void mcs_poisson(
    arma::mat &R,       // (n+1) x 2, (psi,theta)
    arma::vec &pmarg_y, // n x 1, marginal likelihood of y
    double &W,
    const arma::vec &ypad,        // (n+1) x 1, the observed response
    const arma::uvec &model_code, // (obs_code,link_code,transfer_code,gain_code,err_code)
    const arma::vec &obs_par,
    const arma::vec &lag_par, // init/true values of (mu, sg2) or (rho, L)
    const unsigned int &nlag_in,
    const unsigned int &B, // length of the B-lag fixed-lag smoother
    const unsigned int &N, // number of particles
    const double &theta0_upbnd,
    const double &delta_discount,
    const bool &truncated,
    const bool &use_discount)
{

    const unsigned int nt = ypad.n_elem - 1; // number of observations
    double mu0 = obs_par.at(0);
    double delta_nb = obs_par.at(1);
    double m0 = 0.;


    const unsigned int obs_code = model_code.at(0);
    const unsigned int link_code = model_code.at(1);
    const unsigned int trans_code = model_code.at(2);
    const unsigned int gain_code = model_code.at(3);
    const unsigned int err_code = model_code.at(4);

    unsigned int nlag = nlag_in;
    unsigned int p = nlag;
    if (!truncated)
    {
        nlag = nt;
        p = (unsigned int) lag_par[1] + 1;
    }
    arma::vec Fphi = get_Fphi(nlag, lag_par, trans_code);
    arma::vec Ft = init_Ft(p, trans_code);

    arma::mat Gt;
    init_Gt(Gt, lag_par, p, nlag, truncated);

    double C0_sqrt = theta0_upbnd * 0.5;
    arma::vec theta_init = arma::randn(N);
    theta_init.for_each([&m0, &C0_sqrt](arma::vec::elem_type &val)
                        { val = std::abs(m0 + C0_sqrt * std::abs(val)); });

    arma::cube theta_stored(p, N, nt + B);
    for (unsigned int b = 0; b < (nt + B); b++)
    {
        theta_stored.slice(b).zeros();
    }
    theta_stored.slice(B - 1).row(1) = theta_init.t();

    arma::vec Wt(nt);
    Wt.fill(W);

    for (unsigned int t=0; t<nt; t++) {
        R_CheckUserInterrupt();
        arma::mat Theta_old = theta_stored.slice(t + B - 1);
        arma::vec weights(N, arma::fill::ones);
        double wt = Wt.at(t);
        double y_new = ypad.at(t + 1);
        double y_old = ypad.at(t);

        if (nlag < nt)
        { // truncated
            init_Ft(Ft, ypad, Fphi, t, p);
        }

        arma::mat Theta_new(p, N, arma::fill::zeros);

        bool use_custom_val = false;
        if (t > B)
        {
            use_custom_val = true;
        }

        smc_propagate_bootstrap(
            Theta_new, weights, wt, Gt,
            y_new, y_old, Theta_old, Ft, model_code,
            obs_par, lag_par, p, t, nlag, N, delta_discount,
            truncated, use_discount, use_custom_val);

        theta_stored.slice(t+B) = Theta_new;
        double ymarg = std::log(arma::accu(weights) + EPS8) - std::log(static_cast<double>(N));
        bound_check(ymarg, "mcs_poisson<void>: ymarg");
        pmarg_y.at(t) = ymarg;
        Wt.at(t) = wt;

        bool resample;
        double meff = 0;
        smc_resample(theta_stored, weights, meff, resample, t, B);
        
        
    }

    // W = std::max(W,EPS);



    R.zeros(); // (n+1) x 2
    {
        // theta_stored: p x N x (nt+B)
        arma::mat psi_all = theta_stored.row(0);        // N x (nt+B)
        auto med_psi = arma::median(psi_all, 0); // 0: median for each column
        arma::vec RR = arma::vectorise(med_psi); // (nt+1)
        R.col(0) = RR.tail(R.n_rows); // (nt+1) x 2
    }

    return;
}

//' @export
// [[Rcpp::export]]
Rcpp::List pl_poisson(
    const arma::vec &Y, // nt x 1, the observed response
    const arma::uvec &model_code,
    const Rcpp::IntegerVector &eta_select = Rcpp::IntegerVector::create(1, 0, 0, 0),
    const Rcpp::NumericVector &obs_par_in = Rcpp::NumericVector::create(0., 30.),
    const Rcpp::NumericVector &lag_par_in = Rcpp::NumericVector::create(0.5, 6),
    const Rcpp::NumericVector &W_par_in = Rcpp::NumericVector::create(0.01, 2, 0.01, 0.01), // IG[aw,bw]
    const unsigned int &nlag_in = 0,
    const unsigned int &N = 5000, // number of particles
    const Rcpp::Nullable<Rcpp::NumericVector> &m0_prior = R_NilValue,
    const Rcpp::NumericVector &qProb = Rcpp::NumericVector::create(0.025, 0.5, 0.975),
    const double &theta0_upbnd = 2.,
    const bool &truncated = true,
    const bool &use_discount = false,
    const bool &smoothing = false)
{
    const unsigned int nt = Y.n_elem; // number of observations
    const double N_ = static_cast<double>(N);
    const double nt_ = static_cast<double>(nt);
    arma::vec ypad(nt + 1, arma::fill::zeros);
    ypad.tail(nt) = Y;

    arma::vec W_par(W_par_in.begin(), W_par_in.length());
    arma::vec lag_par(lag_par_in.begin(), lag_par_in.length());
    arma::vec obs_par(obs_par_in.begin(), obs_par_in.length());

    double mu0 = obs_par.at(0);
    double delta_nb = obs_par.at(1);

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

    unsigned int nlag = nlag_in;
    unsigned int p = nlag;
    if (!truncated)
    {
        nlag = nt;
        p = (unsigned int) lag_par[1] + 1;
    }

    arma::vec Fphi = get_Fphi(nlag, lag_par, trans_code);
    arma::vec Ft = init_Ft(p,trans_code);

    arma::mat Gt;
    init_Gt(Gt, lag_par, p, nlag, truncated);

    /* Dimension of state space depends on type of transfer functions */

    /*
    Initialize latent state theta[0].
    */
    arma::mat theta(p, N);
    arma::vec m0(p, arma::fill::zeros);
    arma::mat C0(p, p, arma::fill::eye);
    C0.diag() *= std::pow(theta0_upbnd * 0.5, 2.);
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
    aw.fill(W_par[2]);
    arma::mat bw(N, nt + 1);
    bw.fill(W_par[3]);
    arma::mat Wt(N, nt + 1); // evolution variance
    Wt.fill(W_par[0]);
    bool W_selected = std::abs(eta_select[0] - 1) < EPS;

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
        arma::vec mt = arma::median(Theta_old, 1);
        update_Gt(
            Gt, mt, ypad.at(t), lag_par, 
            gain_code, trans_code, 
            nlag, truncated);

        for (unsigned int i = 0; i < N; i++)
        {
            arma::vec theta_old = Theta_old.col(i); // p x 1
            arma::vec theta_new = update_at(
                Gt, theta_old, ypad.at(t),
                lag_par, gain_code, nlag, truncated);

            double Wsqrt = std::sqrt(Wt.at(i, t));
            double omega_new = R::rnorm(0., Wsqrt);
            theta_new.at(0) += omega_new;

            theta_stored.slice(t + 1).col(i) = theta_new;

            if (W_selected) {
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
                Wt.at(i,t+1) = W_par[0];
            }
        }

        if (W_selected && (t <= std::min(0.1 * nt_, 20.)))
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
        if (truncated)
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
            else if (truncated)
            {
                // truncated
                arma::vec hpsi = psi2hpsi(psi, gain_code);
                lambda.at(i) = mu0 + arma::as_scalar(Ft.t() * hpsi);
            }
            else
            {
                // no truncation: Koyck or Solow with identity link and different gain functions
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

    /*
    ------ Particle Smoothing ------
    */
    double cnst = -0.5*std::log(2*arma::datum::pi);
    unsigned int N_smooth = std::min(1000., 0.1 * N_);
    Rcpp::IntegerVector idx0_ = Rcpp::sample(N, N_smooth, false);
    arma::uvec idx0 = Rcpp::as<arma::uvec>(idx0_) - 1;

    arma::mat psi_smooth(N_smooth, nt + 1);
    arma::vec ptmp = arma::vectorise(theta_stored.slice(nt).row(0));
    psi_smooth.col(nt) = ptmp.elem(idx0);

    // arma::mat aw_smooth(N_smooth, nt + 1);
    // arma::vec atmp = aw.col(nt);
    // aw_smooth.col(nt) = atmp.elem(idx0);

    // arma::mat bw_smooth(N_smooth, nt + 1);
    // arma::vec btmp = bw.col(nt);
    // bw_smooth.col(nt) = btmp.elem(idx0);

    arma::mat Wt_smooth(N_smooth, nt + 1);
    arma::vec wtmp = Wt.col(nt);
    Wt_smooth.col(nt) = wtmp.elem(idx0);

    if (smoothing)
    {
        
        

        for (unsigned int t = nt; t > 1; t--)
        {
            arma::vec psi_filter0 = arma::vectorise(theta_stored.slice(t-1).row(0)); // N x 1
            arma::vec psi_filter = psi_filter0.elem(idx0);

            arma::vec Wt_filter0 = Wt.col(t-1);
            arma::vec Wt_filter = Wt_filter0.elem(idx0);

            for (unsigned int i=0; i<N_smooth; i++) {
                arma::vec psi_diff = arma::vectorise(psi_smooth.at(i, t) - psi_filter); // N x 1
                arma::vec Winv = 1. / Wt_filter;

                arma::vec tmp1 = 0.5 * arma::log(Winv);
                arma::vec tmp2 = arma::pow(psi_diff, 2.);
                arma::vec tmp3 = -0.5 * Winv % tmp2;
                arma::vec logweight = cnst + tmp1 + tmp3;

                logweight.elem(arma::find(logweight > UPBND)).fill(UPBND);
                arma::vec weight = arma::exp(logweight);

                double wsum = arma::accu(weight);
                unsigned int idx;
                if (wsum > EPS)
                {
                    arma::vec wtmp = weight;
                    weight = wtmp / wsum;
                    Rcpp::NumericVector w_ = Rcpp::wrap(weight);
                    Rcpp::IntegerVector idx_ = Rcpp::sample(N_smooth, 1, true, w_);
                    idx = idx_[0] - 1;
                }
                else
                {
                    Rcpp::IntegerVector idx_ = Rcpp::sample(N_smooth, 1, true, R_NilValue);
                    idx = idx_[0] - 1;
                }

                psi_smooth.at(i,t-1) = psi_filter.at(i);
                // aw_smooth.at(i, t - 1) = aw.at(idx,t-1);
                // bw_smooth.at(i, t - 1) = bw.at(idx,t-1);
                Wt_smooth.at(i, t - 1) = Wt_filter.at(idx, t - 1);

                Rcout << "\rSmoothing Progress: " << nt - t + 1 << "/" << nt;

            }
        }
    }

    Rcout << std::endl;

    Rcpp::List output;
    arma::mat psi = theta_stored.row(0); // N x (nt+1)
    arma::mat RR = arma::quantile(psi, Rcpp::as<arma::vec>(qProb), 0);
    output["psi"] = Rcpp::wrap(RR.t());         // (n+1) x 3
    output["theta"] = Rcpp::wrap(theta_stored); // p, N, nt + 1

    arma::mat RR2 = arma::quantile(psi_smooth, Rcpp::as<arma::vec>(qProb), 0);
    output["psi_smooth"] = Rcpp::wrap(RR2.t());        // (n+1) x 3
    output["theta_smooth"] = Rcpp::wrap(psi_smooth); // N, nt + 1

    output["aw"] = Rcpp::wrap(aw.col(nt));
    output["bw"] = Rcpp::wrap(bw.col(nt));
    output["Wt"] = Rcpp::wrap(Wt);
    output["Wt_smooth"] = Rcpp::wrap(Wt_smooth);

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
