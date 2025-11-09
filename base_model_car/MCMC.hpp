#ifndef _MCMC_HPP
#define _MCMC_HPP

#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>

#ifdef DGTF_USE_OPENMP
  #include <omp.h>
#endif

#include <RcppArmadillo.h>
#include <progress.hpp>
#include <progress_bar.hpp>

#include "../core/ApproxDisturbance.hpp"
#include "Model.hpp"

// [[Rcpp::plugins(cpp17)]]
// [[Rcpp::depends(RcppArmadillo,RcppProgress)]]

// Thread-local RNG helpers (OpenMP-safe)
inline double rnorm(double mean, double sd)
{
    thread_local std::mt19937_64 eng(std::random_device{}());
    std::normal_distribution<double> dist(mean, sd);
    return dist(eng);
}

inline double runif()
{
    thread_local std::mt19937_64 eng(std::random_device{}());
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    return dist(eng);
}


class MCMC
{
private:
    unsigned int nsample = 1000;
    unsigned int nburnin = 1000;
    unsigned int nthin = 1;

    unsigned int global_nleapfrog = 10;
    double global_leapfrog_step_size = 0.01;

    double rho_accept_count = 0.0;
    double global_accept_count = 0.0;

    std::vector<std::string> local_params_selected;
    std::vector<std::string> global_params_selected;

    // Prior for disturbance wt
    Prior wt_prior;

    // Priors for local parameters
    Prior rho_prior;
    Prior W_prior;

    // Priors for global parameters
    Prior car_prior;
    Prior lag_par1_prior;
    Prior lag_par2_prior;
    Prior beta_prior;

    // Storage for MCMC samples
    arma::vec lag_par1_stored;
    arma::vec lag_par2_stored;
    arma::vec sp_beta_stored;
    arma::vec car_mu_stored;
    arma::vec car_tau2_stored;
    arma::vec car_rho_stored;

    arma::cube wt_stored; // nS x (nT + 1) x nsample
    arma::mat wt_accept; // nS x (nT + 1)

public:
    
    MCMC(
        const unsigned int &nsample_in = 1000,
        const unsigned int &nburnin_in = 1000,
        const unsigned int &nthin_in = 1,
        const bool &infer_wt_in = false,
        const bool &infer_car_in = true
    )
    {
        nsample = nsample_in;
        nburnin = nburnin_in;
        nthin = nthin_in;

        wt_prior.infer = infer_wt_in;
        wt_prior.mh_sd = 1.0;

        car_prior.infer = infer_car_in;
        car_prior.name = "jeffrey";
        car_prior.mh_sd = 1.0;
        car_prior.par1 = 1.0; // jeffrey_prior_order
        return;
    } // end of constructor

    MCMC(const Rcpp::List &opts)
    {
        nsample = 1000;
        if (opts.containsElementNamed("nsample"))
        {
            nsample = Rcpp::as<unsigned int>(opts["nsample"]);
        }

        nburnin = 1000;
        if (opts.containsElementNamed("nburnin"))
        {
            nburnin = Rcpp::as<unsigned int>(opts["nburnin"]);
        }

        nthin = 1;
        if (opts.containsElementNamed("nthin"))
        {
            nthin = Rcpp::as<unsigned int>(opts["nthin"]);
        }
        
        if (opts.containsElementNamed("wt"))
        {
            Rcpp::List wt_opts = Rcpp::as<Rcpp::List>(opts["wt"]);
            wt_prior.infer = false;
            if (wt_opts.containsElementNamed("infer"))
            {
                wt_prior.infer = Rcpp::as<bool>(wt_opts["infer"]);
            }

            wt_prior.mh_sd = 1.0;
            if (wt_opts.containsElementNamed("mh_sd"))
            {
                wt_prior.mh_sd = Rcpp::as<double>(wt_opts["mh_sd"]);
            }
        }

        if (opts.containsElementNamed("car"))
        {
            Rcpp::List car_opts = Rcpp::as<Rcpp::List>(opts["car"]);
            car_prior.infer = false;
            if (car_opts.containsElementNamed("infer"))
            {
                car_prior.infer = Rcpp::as<bool>(car_opts["infer"]);
            }

            car_prior.mh_sd = 0.1;
            if (car_opts.containsElementNamed("mh_sd"))
            {
                car_prior.mh_sd = Rcpp::as<double>(car_opts["mh_sd"]);
            }

            car_prior.name = "jeffrey";
            car_prior.par1 = 1.0; // jeffrey_prior_order
            if (car_opts.containsElementNamed("par1"))
            {
                car_prior.par1 = Rcpp::as<double>(car_opts["par1"]);
            }
        }

        if (opts.containsElementNamed("rho"))
        {
            Rcpp::List rho_opts = Rcpp::as<Rcpp::List>(opts["rho"]);
            rho_prior.init(rho_opts);
        }
        if (rho_prior.infer)
        {
            local_params_selected.push_back("rho");
        }

        if (opts.containsElementNamed("W"))
        {
            Rcpp::List W_opts = Rcpp::as<Rcpp::List>(opts["W"]);
            W_prior.init(W_opts);
        }
        if (W_prior.infer)
        {
            local_params_selected.push_back("W");
        }

        if (opts.containsElementNamed("lag_par1"))
        {
            Rcpp::List lag_par1_opts = Rcpp::as<Rcpp::List>(opts["lag_par1"]);
            lag_par1_prior.init(lag_par1_opts);
        }
        if (lag_par1_prior.infer)
        {
            global_params_selected.push_back("lag_par1");
        }

        if (opts.containsElementNamed("lag_par2"))
        {
            Rcpp::List lag_par2_opts = Rcpp::as<Rcpp::List>(opts["lag_par2"]);
            lag_par2_prior.init(lag_par2_opts);
        }
        if (lag_par2_prior.infer)
        {
            global_params_selected.push_back("lag_par2");
        }

        if (opts.containsElementNamed("beta"))
        {
            Rcpp::List beta_opts = Rcpp::as<Rcpp::List>(opts["beta"]);
            beta_prior.init(beta_opts);
        }
        if (beta_prior.infer)
        {
            global_params_selected.push_back("beta");
        }

        if (opts.containsElementNamed("global_hmc"))
        {
            Rcpp::List global_hmc_opts = Rcpp::as<Rcpp::List>(opts["global_hmc"]);
            global_nleapfrog = 10;
            if (global_hmc_opts.containsElementNamed("nleapfrog"))
            {
                global_nleapfrog = Rcpp::as<unsigned int>(global_hmc_opts["nleapfrog"]);
            }

            global_leapfrog_step_size = 0.01;
            if (global_hmc_opts.containsElementNamed("leapfrog_step_size"))
            {
                global_leapfrog_step_size = Rcpp::as<double>(global_hmc_opts["leapfrog_step_size"]);
            }
        }

        return;
    } // end of constructor from Rcpp::List

    static Rcpp::List get_default_settings()
    {
        Rcpp::List wt_opts = Rcpp::List::create(
            Rcpp::Named("infer") = false,
            Rcpp::Named("mh_sd") = 1.0
        );

        Rcpp::List car_opts = Rcpp::List::create(
            Rcpp::Named("infer") = false,
            Rcpp::Named("name") = "jeffrey",
            Rcpp::Named("mh_sd") = 0.1,
            Rcpp::Named("par1") = 1.0
        );

        Rcpp::List rho_opts = Rcpp::List::create(
            Rcpp::Named("infer") = false,
            Rcpp::Named("name") = "invgamma",
            Rcpp::Named("par1") = 1.0,
            Rcpp::Named("par2") = 1.0
        );

        Rcpp::List W_opts = Rcpp::List::create(
            Rcpp::Named("infer") = false,
            Rcpp::Named("name") = "invgamma",
            Rcpp::Named("par1") = 2.0,
            Rcpp::Named("par2") = 1.0
        );

        Rcpp::List lag_par1_opts = Rcpp::List::create(
            Rcpp::Named("infer") = false,
            Rcpp::Named("name") = "gaussian",
            Rcpp::Named("par1") = 0.0,
            Rcpp::Named("par2") = 1.0
        );

        Rcpp::List lag_par2_opts = Rcpp::List::create(
            Rcpp::Named("infer") = false,
            Rcpp::Named("name") = "invgamma",
            Rcpp::Named("par1") = 1.0,
            Rcpp::Named("par2") = 1.0
        );

        Rcpp::List beta_opts = Rcpp::List::create(
            Rcpp::Named("infer") = false,
            Rcpp::Named("name") = "gaussian", // normal prior on log(beta)
            Rcpp::Named("par1") = 0.0,
            Rcpp::Named("par2") = 1.0
        );

        Rcpp::List global_hmc_opts = Rcpp::List::create(
            Rcpp::Named("nleapfrog") = 10,
            Rcpp::Named("leapfrog_step_size") = 0.01
        );

        Rcpp::List mcmc_opts = Rcpp::List::create(
            Rcpp::Named("nsample") = 1000,
            Rcpp::Named("nburnin") = 1000,
            Rcpp::Named("nthin") = 1,
            Rcpp::Named("wt") = wt_opts,
            Rcpp::Named("car") = car_opts,
            Rcpp::Named("rho") = rho_opts,
            Rcpp::Named("W") = W_opts,
            Rcpp::Named("lag_par1") = lag_par1_opts,
            Rcpp::Named("lag_par2") = lag_par2_opts,
            Rcpp::Named("beta") = beta_opts,
            Rcpp::Named("global_hmc") = global_hmc_opts
        );

        return mcmc_opts;
    } // end of get_default_settings()

    void update_car(
        SpatialStructure &spatial, 
        double &rho_accept_count,
        const arma::vec &spatial_effects,
        const double &mh_sd = 0.1,
        const double &jeffrey_prior_order = 1.0
    )
    {
        double rho_min = spatial.min_car_rho;
        double rho_max = spatial.max_car_rho;

        double rho_current = spatial.car_rho;
        double log_post_rho_old = spatial.log_posterior_rho(spatial_effects, jeffrey_prior_order);

        double eta = logit(standardize(rho_current, rho_min, rho_max, true));
        double eta_new = eta + R::rnorm(0.0, mh_sd);
        double rho_new = rho_min + logistic(eta_new) * (rho_max - rho_min);
        // double rho_new = rho_current + R::rnorm(0.0, mh_sd);

        /*
        When rho is updated, we also need to update:
        - precision matrix Q
        - one_Q_one: updated in `update_car() -> compute_precision()`
        - post_mu_mean: updated in `log_posterior_rho()`
        - post_mu_prec: updated in `log_posterior_rho()`
        - post_tau2_rate: updated in `log_posterior_rho()`
        */
        // if (rho_new > spatial.min_car_rho && rho_new < spatial.max_car_rho)
        // {
        spatial.update_params(
            spatial.car_mu,
            spatial.car_tau2,
            rho_new);
        double log_post_rho_new = spatial.log_posterior_rho(spatial_effects, jeffrey_prior_order);

        double u_old = standardize(rho_current, rho_min, rho_max, true);
        double u_new = standardize(rho_new, rho_min, rho_max, true);
        double log_jac = (std::log(u_new) + std::log1p(-u_new)) - (std::log(u_old) + std::log1p(-u_old));

        double log_accept_ratio = log_post_rho_new - log_post_rho_old + log_jac;
        if (std::log(R::runif(0.0, 1.0)) < log_accept_ratio)
        {
            rho_accept_count += 1.0;
        }
        else
        {
            // revert
            spatial.update_params(
                spatial.car_mu,
                spatial.car_tau2,
                rho_current);
            double log_post_rho = spatial.log_posterior_rho(spatial_effects, jeffrey_prior_order);
        }
        // }

        /*
        When tau2 is updated, we also need to update:
        - post_mu_prec: update manually here
        */
        double tau2_new = R::rgamma(spatial.post_tau2_shape, 1.0 / spatial.post_tau2_rate);
        spatial.update_params(
            spatial.car_mu,
            tau2_new,
            spatial.car_rho
        );

        double mu_new = R::rnorm(spatial.post_mu_mean, std::sqrt(1.0 / spatial.post_mu_prec));
        spatial.update_params(
            mu_new,
            spatial.car_tau2,
            spatial.car_rho
        );

        return;        
    } // end of update_car()

    void update_wt(
        arma::mat &wt, // nS x (nT + 1)
        arma::mat &wt_accept, // nS x (nT + 1)
        const Model &model, 
        const arma::mat &Y, // nS x (nT + 1),
        const double &mh_sd = 1.0
    )
    {
        const unsigned int nT = Y.n_cols - 1;
        
        #ifdef DGTF_USE_OPENMP
        const bool _omp_enable = (model.nS > 1);
        #pragma omp parallel for if(_omp_enable) schedule(static)
        #endif
        for (unsigned int s = 0; s < model.nS; s++)
        {
            ObsDist dobs(model.dobs, 0.0, model.rho.at(s));
            ApproxDisturbance approx_dlm(Y.n_cols - 1, model.fgain);
            approx_dlm.set_Fphi(model.dlag, model.nP);

            const double prior_sd = std::sqrt(model.W.at(s));
            const double spatial_effect = std::exp(model.log_alpha.at(s));
            const arma::vec neighbor_weights = model.spatial.W.row(s).t();
            const arma::vec cross_region_effects = model.beta * Y.head_cols(nT).t() * neighbor_weights;
            const arma::vec ys = Y.row(s).t();

            arma::vec acc_row(nT + 1, arma::fill::zeros);
            arma::vec ws = wt.row(s).t();
            arma::vec psi_s = arma::cumsum(ws); // (nT + 1) x 1

            arma::vec lam_old = model.compute_intensity_iterative(s, Y, psi_s); // (nT + 1) x 1
            for (unsigned int t = 1; t <= nT; t++)
            {
                double w_ts_old = ws.at(t);
                double logp_old = R::dnorm4(w_ts_old, 0.0, prior_sd, true);
                for (unsigned int i = t; i <= nT; i++)
                {
                    logp_old += ObsDist::loglike(
                        ys.at(i), model.dobs, lam_old.at(i), model.rho.at(s), true
                    );
                }

                /*
                Metropolis-Hastings update for w_ts
                CAVEAT: not taking care of the link function yet
                */
                approx_dlm.update_by_psi(ys, psi_s);
                arma::vec eta_hat = approx_dlm.f0 + approx_dlm.Fn * ws.tail(nT); // approximated self-exciting, nT x 1
                eta_hat += spatial_effect;
                eta_hat += cross_region_effects;

                arma::vec Vt_hat = ApproxDisturbance::func_Vt_approx(eta_hat, dobs, model.flink);
                arma::vec Fnt = approx_dlm.Fn.col(t - 1); // nT x 1
                double mh_prec = arma::accu((Fnt % Fnt) / Vt_hat) + EPS8;
                double mh_step = std::sqrt(1.0 / mh_prec) * mh_sd;

                double w_ts_new = rnorm(w_ts_old, mh_step);
                ws.at(t) = w_ts_new;
                psi_s = arma::cumsum(ws);
                arma::vec lam_new = model.compute_intensity_iterative(s, Y, psi_s);

                double logp_new = R::dnorm4(w_ts_new, 0.0, prior_sd, true);
                for (unsigned int i = t; i <= nT; i++)
                {
                    logp_new += ObsDist::loglike(
                        ys.at(i), model.dobs, lam_new.at(i), model.rho.at(s), true
                    );
                }

                double logratio = std::min(0.0, logp_new - logp_old);
                if (std::log(runif()) < logratio)
                {
                    acc_row.at(t) = 1.0;
                    lam_old = lam_new;
                }
                else
                {
                    ws.at(t) = w_ts_old;
                    psi_s = arma::cumsum(ws);
                }
            } // end of t loop for location s

            wt.row(s) = ws.t();
            wt_accept.row(s) += acc_row.t();
        } // end of s loop
    } // end of update_wt()

    double compute_log_joint(
        const Model &model, 
        const arma::mat &Y, 
        const arma::mat &wt,
        const bool &global_or_local = true // true for global, false for local
    )
    {
        
        double logp = 0.0;
        for (unsigned int s = 0; s < model.nS; s++)
        {
            arma::vec psi_s = arma::cumsum(wt.row(s).t());
            arma::vec lam_s = model.compute_intensity_iterative(s, Y, psi_s);
            for (unsigned int t = 1; t <= Y.n_cols - 1; t++)
            {
                // Compute loglikelihood of y[s, t]
                logp += ObsDist::loglike(
                    Y.at(s, t), model.dobs, lam_s.at(t), model.rho.at(s), true
                );

                if (!global_or_local)
                {
                    // Add log prior for w[s, t]
                    logp += R::dnorm4(wt.at(s, t), 0.0, std::sqrt(model.W.at(s)), true);
                }
            }

            if (!global_or_local)
            {
                // Add local parameter priors
                if (rho_prior.infer)
                {
                    logp += Prior::dprior(model.rho.at(s), rho_prior, true, true);
                }

                if (W_prior.infer)
                {
                    logp += Prior::dprior(model.W.at(s), W_prior, true, true);
                }
            }
        }

        if (global_or_local)
        {
            // Add global parameter priors
            if (lag_par1_prior.infer)
            {
                logp += Prior::dprior(model.dlag.par1, lag_par1_prior, true, true);
            }

            if (lag_par2_prior.infer)
            {
                logp += Prior::dprior(model.dlag.par2, lag_par2_prior, true, true);
            }

            if (beta_prior.infer)
            {
                logp += Prior::dprior(model.beta, beta_prior, true, true);
            }
        }

        return logp;
    }


    /**
     * @brief HMC sampler for global parameters (lag distribution parameters and beta)
     * 
     * @param model 
     * @param Y 
     * @param wt 
     */
    double update_global_params(
        Model &model, 
        const std::vector<std::string> &global_params_selected,
        const arma::mat &Y, 
        const arma::mat &wt,
        const double &leapfrog_step_size,
        const unsigned int &n_leapfrog = 10
    )
    {
        // Precompute hPsi (depends only on wt)
        arma::mat Psi = arma::cumsum(wt, 1);
        arma::mat hPsi = GainFunc::psi2hpsi<arma::mat>(Psi, model.fgain);

        // Potential energy: negative log joint probability.
        double logp_current = compute_log_joint(model, Y, wt, true);
        double energy_current = -logp_current;

        // Get current global parameters in unconstrained space
        arma::vec global_params_current = model.get_global_params_unconstrained(global_params_selected);
        arma::vec q = global_params_current;

        // sample an initial momentum
        arma::vec p = arma::randn(q.n_elem);
        double kinetic_current = 0.5 * arma::dot(p, p);

        arma::vec grad = model.dloglik_dglobal_unconstrained(
            global_params_selected, Y, hPsi, 
            lag_par1_prior, 
            lag_par2_prior, 
            beta_prior
        );

        grad *= -1.0; // Convert to gradient of potential energy

        // Make a half step for momentum at the beginning
        p -= 0.5 * leapfrog_step_size * grad;
        for (unsigned int i = 0; i < n_leapfrog; i++)
        {
            // Make a full step for the position
            q += leapfrog_step_size * p;
            model.update_global_params_unconstrained(global_params_selected, q);

            // Compute the new gradient
            grad = model.dloglik_dglobal_unconstrained(
                global_params_selected, Y, hPsi, 
                lag_par1_prior, 
                lag_par2_prior, 
                beta_prior
            );

            grad *= -1.0; // Convert to gradient of potential energy

            // Make a full step for the momentum, except at the end of trajectory
            if (i != n_leapfrog - 1)
            {
                p -= leapfrog_step_size * grad;
            }
        } // end of leapfrog steps

        p -= 0.5 * leapfrog_step_size * grad; // Make a half step for momentum at the end
        p *= -1; // Negate momentum to make the proposal symmetric

        double logp_proposed = compute_log_joint(model, Y, wt, true);
        double energy_proposed = -logp_proposed;
        double kinetic_proposed = 0.5 * arma::dot(p, p);

        if (!std::isfinite(logp_current) || !std::isfinite(logp_proposed))
        {
            // Reject if either log probability is not finite
            model.update_global_params_unconstrained(global_params_selected, global_params_current);
            return 0.0;
        }
        else
        {
            double log_accept_ratio = energy_current - energy_proposed + kinetic_current - kinetic_proposed;
            if (std::log(runif()) >= log_accept_ratio)
            {
                // Reject: revert to current parameters
                model.update_global_params_unconstrained(global_params_selected, global_params_current);
            }

            double accept_prob = std::min(1.0, std::exp(log_accept_ratio));
            return accept_prob;
        }
    }

    void infer(Model &model, const arma::mat &Y)
    {
        const unsigned int nT = Y.n_cols - 1;
        const unsigned int niter = nburnin + nsample * nthin;

        // Dual averaging state (Hoffman & Gelman 2014)
        const double target_accept = 0.75;
        double mu_da = std::log(10.0 * global_leapfrog_step_size); // bias center
        double log_eps = std::log(global_leapfrog_step_size);
        double log_eps_bar = log_eps;
        double h_bar = 0.0;
        double gamma_da = 0.05;
        double t0_da = 10.0;
        double kappa_da = 0.75;
        unsigned int adapt_count = 0;

        // Clamp helpers
        auto clamp_eps = [](double x)
        { return std::max(1e-3, std::min(0.1, x)); };
        double T_target = 1.0;
        unsigned min_leaps = 3, max_leaps = 128;

        if (car_prior.infer)
        {
            car_mu_stored = arma::vec(nsample, arma::fill::zeros);
            car_tau2_stored = arma::vec(nsample, arma::fill::zeros);
            car_rho_stored = arma::vec(nsample, arma::fill::zeros);

            // model.spatial.car_mu = rnorm()
        }

        if (wt_prior.infer)
        {
            wt_stored = arma::cube(model.nS, nT + 1, nsample, arma::fill::zeros);
            wt_accept = arma::mat(model.nS, nT + 1, arma::fill::zeros);
        }

        if (lag_par1_prior.infer)
        {
            lag_par1_stored = arma::vec(nsample, arma::fill::zeros);
        }

        if (lag_par2_prior.infer)
        {
            lag_par2_stored = arma::vec(nsample, arma::fill::zeros);
        }

        if (beta_prior.infer)
        {
            sp_beta_stored = arma::vec(nsample, arma::fill::zeros);
        }

        arma::mat wt(model.nS, nT + 1, arma::fill::randn);
        wt.each_col() %= arma::sqrt(model.W);

        Progress p(niter, true);
        for (unsigned int iter = 0; iter < niter; ++iter)
        {
            if (wt_prior.infer)
            {
                update_wt(wt, wt_accept, model, Y, wt_prior.mh_sd);
            }

            if (car_prior.infer)
            {
                update_car(
                    model.spatial,
                    rho_accept_count,
                    model.log_alpha,
                    car_prior.mh_sd,
                    car_prior.par1
                );
            }

            if (!global_params_selected.empty())
            {
                double global_hmc_accept_prob = update_global_params(
                    model, global_params_selected, 
                    Y, wt, 
                    global_leapfrog_step_size, 
                    global_nleapfrog
                );
                global_accept_count += global_hmc_accept_prob;

                if (iter < nburnin)
                {
                    adapt_count++;
                    double t = (double)adapt_count;
                    h_bar = (1.0 - 1.0 / (t + t0_da)) * h_bar + (1.0 / (t + t0_da)) * (target_accept - global_hmc_accept_prob);
                    log_eps = mu_da - (std::sqrt(t) / gamma_da) * h_bar;
                    global_leapfrog_step_size = clamp_eps(std::exp(log_eps));
                    double w = std::pow(t, -kappa_da);
                    log_eps_bar = w * std::log(global_leapfrog_step_size) + (1.0 - w) * log_eps_bar;
                }
                else if (iter == nburnin)
                {
                    global_leapfrog_step_size = clamp_eps(std::exp(log_eps_bar));
                    unsigned nlf = (unsigned)std::lround(T_target / global_leapfrog_step_size);
                    global_nleapfrog = std::max(min_leaps, std::min(max_leaps, nlf));
                }
            }



            // Store samples after burn-in and thinning
            if (iter >= nburnin && ((iter - nburnin) % nthin == 0))
            {
                const unsigned int sample_idx = (iter - nburnin) / nthin;
                if (wt_prior.infer)
                {
                    wt_stored.slice(sample_idx) = wt;
                }

                if (car_prior.infer)
                {
                    car_mu_stored(sample_idx) = model.spatial.car_mu;
                    car_tau2_stored(sample_idx) = model.spatial.car_tau2;
                    car_rho_stored(sample_idx) = model.spatial.car_rho;
                }

                if (lag_par1_prior.infer)
                {
                    lag_par1_stored(sample_idx) = model.dlag.par1;
                }
                
                if (lag_par2_prior.infer)
                {
                    lag_par2_stored(sample_idx) = model.dlag.par2;
                }

                if (beta_prior.infer)
                {
                    sp_beta_stored(sample_idx) = model.beta;
                }
            }

            p.increment(); 
        } // end of iter loop
    } // end of infer()

    Rcpp::List get_output() const
    {
        Rcpp::List output;

        if (wt_prior.infer)
        {
            output["wt_samples"] = Rcpp::wrap(wt_stored);
            output["wt_accept_rate"] = wt_accept / (nburnin + nsample * nthin);
        }

        if (car_prior.infer)
        {
            output["car_mu"] = Rcpp::wrap(car_mu_stored);
            output["car_tau2"] = Rcpp::wrap(car_tau2_stored);
            output["car_rho"] = Rcpp::wrap(car_rho_stored);
            output["rho_accept_rate"] = rho_accept_count / (nburnin + nsample * nthin);
        }

        if (!global_params_selected.empty())
        {
            output["global_accept_rate"] = global_accept_count / (nburnin + nsample * nthin);
            output["global_hmc_settings"] = Rcpp::List::create(
                Rcpp::Named("nleapfrog") = global_nleapfrog,
                Rcpp::Named("leapfrog_step_size") = global_leapfrog_step_size
            );

            if (lag_par1_prior.infer)
            {
                output["lag_par1"] = Rcpp::wrap(lag_par1_stored);
            }
            if (lag_par2_prior.infer)
            {
                output["lag_par2"] = Rcpp::wrap(lag_par2_stored);
            }
            if (beta_prior.infer)
            {
                output["beta"] = Rcpp::wrap(sp_beta_stored);
            }
        }

        return output;
    } // end of get_output()
}; // class mcmc


#endif