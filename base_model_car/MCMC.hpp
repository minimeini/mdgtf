#ifndef _MCMC_HPP
#define _MCMC_HPP

#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <algorithm>

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


class MCMC
{
private:
    unsigned int nsample = 1000;
    unsigned int nburnin = 1000;
    unsigned int nthin = 1;

    bool infer_log_alpha = false;

    double car_rho_accept_count = 0.0;

    // HMC settings for global parameters
    std::vector<std::string> global_params_selected;
    bool global_dual_averaging = false;
    bool global_diagnostics = true;
    bool global_verbose = false;
    unsigned int global_nleapfrog = 10;
    double global_leapfrog_step_size = 0.01;
    // global_T_target: integration time T = n_leapfrog * epsilon ~= 1-2 (rough heuristic). 
    // Larger T gives better exploration but higher cost.
    double global_T_target = 1.0;

    // HMC settings for local parameters
    std::vector<std::string> local_params_selected;
    bool local_dual_averaging = false;
    bool local_diagnostics = true;
    bool local_verbose = false;
    unsigned int local_nleapfrog_init = 10;
    double local_leapfrog_step_size_init = 0.01;
    arma::uvec local_nleapfrog; // nS x 1
    arma::vec local_leapfrog_step_size; // nS x 1
    // local_T_target: integration time T = n_leapfrog * epsilon ~= 1-2 (rough heuristic). 
    // Larger T gives better exploration but higher cost.
    double local_T_target = 1.0;

    // HMC settings for log_alpha
    bool logalpha_dual_averaging = false;
    bool logalpha_diagnostics = true;
    bool logalpha_verbose = false;
    unsigned int logalpha_nleapfrog = 10;
    double logalpha_leapfrog_step_size = 0.01;
    // global_T_target: integration time T = n_leapfrog * epsilon ~= 1-2 (rough heuristic). 
    // Larger T gives better exploration but higher cost.
    double logalpha_T_target = 1.0;

    // Prior for disturbance wt
    Prior wt_prior;

    // Priors for local parameters
    Prior rho_prior;
    Prior W_prior;

    // Priors for global parameters
    BYM2Prior bym2_prior;
    Prior lag_par1_prior;
    Prior lag_par2_prior;
    Prior beta_prior;

    // Storage for global parameter samples
    arma::vec lag_par1_stored; // nsample x 1
    arma::vec lag_par2_stored;
    arma::vec sp_beta_stored;

    arma::vec bym2_mu_stored;
    arma::vec bym2_tau_b_stored;
    arma::vec bym2_phi_stored;
    arma::mat log_alpha_stored; // nS x nsample

    // Storage for local parameter samples
    arma::mat rho_stored; // nS x nsample
    arma::mat W_stored; // nS x nsample

    // Storage for disturbance samples
    arma::cube wt_stored; // nS x (nT + 1) x nsample
    arma::mat wt_accept; // nS x (nT + 1)

    // Store global diagnostics
    double global_accept_count = 0.0;
    arma::vec global_energy_diff; // niter x 1
    arma::vec global_grad_norm; // niter x 1
    arma::vec global_nleapfrog_stored; // nburnin x 1
    arma::vec global_leapfrog_step_size_stored; // nburnin x 1

    // Store local diagnostics
    arma::vec local_accept_count; // nS x 1
    arma::mat local_energy_diff; // nS x niter
    arma::mat local_grad_norm; // nS x niter
    arma::mat local_nleapfrog_stored; // nS x nburnin
    arma::mat local_leapfrog_step_size_stored; // nS x nburnin

    // Store log_alpha diagnostics
    double logalpha_accept_count = 0.0;
    arma::vec logalpha_energy_diff; // niter x 1
    arma::vec logalpha_grad_norm; // niter x 1
    arma::vec logalpha_nleapfrog_stored; // nburnin x 1
    arma::vec logalpha_leapfrog_step_size_stored; // nburnin x 1

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

        logalpha_accept_count = 0.0;
        infer_log_alpha = false;
        if (opts.containsElementNamed("log_alpha"))
        {
            Rcpp::List logalpha_opts = Rcpp::as<Rcpp::List>(opts["log_alpha"]);
            if (logalpha_opts.containsElementNamed("infer"))
            {
                infer_log_alpha = Rcpp::as<bool>(logalpha_opts["infer"]);
            }

            if (infer_log_alpha)
            {
                logalpha_nleapfrog = 10;
                logalpha_leapfrog_step_size = 0.01;
                logalpha_dual_averaging = false;
                logalpha_diagnostics = true;
                logalpha_verbose = false;
                logalpha_T_target = 1.0;

                if (logalpha_opts.containsElementNamed("hmc"))
                {
                    Rcpp::List logalpha_hmc_opts = Rcpp::as<Rcpp::List>(logalpha_opts["hmc"]);
                    if (logalpha_hmc_opts.containsElementNamed("nleapfrog"))
                    {
                        logalpha_nleapfrog = Rcpp::as<unsigned int>(logalpha_hmc_opts["nleapfrog"]);
                    }
                    if (logalpha_hmc_opts.containsElementNamed("leapfrog_step_size"))
                    {
                        logalpha_leapfrog_step_size = Rcpp::as<double>(logalpha_hmc_opts["leapfrog_step_size"]);
                    }
                    if (logalpha_hmc_opts.containsElementNamed("dual_averaging"))
                    {
                        logalpha_dual_averaging = Rcpp::as<bool>(logalpha_hmc_opts["dual_averaging"]);
                    }
                    if (logalpha_hmc_opts.containsElementNamed("diagnostics"))
                    {
                        logalpha_diagnostics = Rcpp::as<bool>(logalpha_hmc_opts["diagnostics"]);
                    }
                    if (logalpha_hmc_opts.containsElementNamed("verbose"))
                    {
                        logalpha_verbose = Rcpp::as<bool>(logalpha_hmc_opts["verbose"]);
                    }
                    if (logalpha_hmc_opts.containsElementNamed("T_target"))
                    {
                        logalpha_T_target = Rcpp::as<double>(logalpha_hmc_opts["T_target"]);
                    }
                }
            } // end of HMC settings if (infer_log_alpha)

            if (logalpha_opts.containsElementNamed("bym2"))
            {
                Rcpp::List bym2_opts = Rcpp::as<Rcpp::List>(logalpha_opts["bym2"]);
                bym2_prior = BYM2Prior(bym2_opts);
            }
        } // end of if (opts.containsElementNamed("log_alpha"))
        
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

            global_dual_averaging = false;
            if (global_hmc_opts.containsElementNamed("dual_averaging"))
            {
                global_dual_averaging = Rcpp::as<bool>(global_hmc_opts["dual_averaging"]);
            }

            global_diagnostics = true;
            if (global_hmc_opts.containsElementNamed("diagnostics"))
            {
                global_diagnostics = Rcpp::as<bool>(global_hmc_opts["diagnostics"]);
            }

            global_verbose = false;
            if (global_hmc_opts.containsElementNamed("verbose"))
            {
                global_verbose = Rcpp::as<bool>(global_hmc_opts["verbose"]);
            }

            global_T_target = 1.0;
            if (global_hmc_opts.containsElementNamed("T_target"))
            {
                global_T_target = Rcpp::as<double>(global_hmc_opts["T_target"]);
            }
        }

        if (opts.containsElementNamed("local_hmc"))
        {
            Rcpp::List local_hmc_opts = Rcpp::as<Rcpp::List>(opts["local_hmc"]);

            local_nleapfrog_init = 10;
            if (local_hmc_opts.containsElementNamed("nleapfrog"))
            {
                local_nleapfrog_init = Rcpp::as<unsigned int>(local_hmc_opts["nleapfrog"]);
            }

            local_leapfrog_step_size_init = 0.01;
            if (local_hmc_opts.containsElementNamed("leapfrog_step_size"))
            {
                local_leapfrog_step_size_init = Rcpp::as<double>(local_hmc_opts["leapfrog_step_size"]);
            }

            local_dual_averaging = false;
            if (local_hmc_opts.containsElementNamed("dual_averaging"))
            {
                local_dual_averaging = Rcpp::as<bool>(local_hmc_opts["dual_averaging"]);
            }

            local_diagnostics = true;
            if (local_hmc_opts.containsElementNamed("diagnostics"))
            {
                local_diagnostics = Rcpp::as<bool>(local_hmc_opts["diagnostics"]);
            }

            local_verbose = false;
            if (local_hmc_opts.containsElementNamed("verbose"))
            {
                local_verbose = Rcpp::as<bool>(local_hmc_opts["verbose"]);
            }

            local_T_target = 1.0;
            if (local_hmc_opts.containsElementNamed("T_target"))
            {
                local_T_target = Rcpp::as<double>(local_hmc_opts["T_target"]);
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

        Rcpp::List bym2_opts = Rcpp::List::create(
            Rcpp::Named("infer") = false,
            Rcpp::Named("mh_sd") = 0.1,
            Rcpp::Named("tau_b") = Rcpp::NumericVector::create(1.0, 1.0), // shape and rate for tau_b
            Rcpp::Named("logit_phi") = Rcpp::NumericVector::create(0.0, 1.0) // mean and sd for logit(phi)
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
            Rcpp::Named("leapfrog_step_size") = 0.01,
            Rcpp::Named("dual_averaging") = false,
            Rcpp::Named("diagnostics") = true,
            Rcpp::Named("verbose") = false,
            Rcpp::Named("T_target") = 1.0
        );

        Rcpp::List local_hmc_opts = Rcpp::List::create(
            Rcpp::Named("nleapfrog") = 10,
            Rcpp::Named("leapfrog_step_size") = 0.01,
            Rcpp::Named("dual_averaging") = false,
            Rcpp::Named("diagnostics") = true,
            Rcpp::Named("verbose") = false,
            Rcpp::Named("T_target") = 1.0
        );

        Rcpp::List logalpha_hmc_opts = Rcpp::List::create(
            Rcpp::Named("nleapfrog") = 10,
            Rcpp::Named("leapfrog_step_size") = 0.01,
            Rcpp::Named("dual_averaging") = false,
            Rcpp::Named("diagnostics") = true,
            Rcpp::Named("verbose") = false,
            Rcpp::Named("T_target") = 1.0
        );

        Rcpp::List mcmc_opts = Rcpp::List::create(
            Rcpp::Named("nsample") = 1000,
            Rcpp::Named("nburnin") = 1000,
            Rcpp::Named("nthin") = 1,
            Rcpp::Named("log_alpha") = Rcpp::List::create(
                Rcpp::Named("infer") = false,
                Rcpp::Named("bym2") = bym2_opts,
                Rcpp::Named("hmc") = logalpha_hmc_opts
            ),
            Rcpp::Named("wt") = wt_opts,
            Rcpp::Named("rho") = rho_opts,
            Rcpp::Named("W") = W_opts,
            Rcpp::Named("lag_par1") = lag_par1_opts,
            Rcpp::Named("lag_par2") = lag_par2_opts,
            Rcpp::Named("beta") = beta_opts,
            Rcpp::Named("global_hmc") = global_hmc_opts,
            Rcpp::Named("local_hmc") = local_hmc_opts
        );

        return mcmc_opts;
    } // end of get_default_settings()

    // void update_car(
    //     SpatialStructure &spatial, 
    //     double &rho_accept_count,
    //     const arma::vec &spatial_effects,
    //     const double &mh_sd = 0.1,
    //     const double &jeffrey_prior_order = 1.0
    // )
    // {
    //     double rho_min = spatial.min_car_rho;
    //     double rho_max = spatial.max_car_rho;

    //     double rho_current = spatial.car_rho;
    //     double log_post_rho_old = spatial.log_posterior_rho(spatial_effects, jeffrey_prior_order);

    //     double eta = logit(standardize(rho_current, rho_min, rho_max, true));
    //     double eta_new = eta + R::rnorm(0.0, mh_sd);
    //     double rho_new = rho_min + logistic(eta_new) * (rho_max - rho_min);
    //     // double rho_new = rho_current + R::rnorm(0.0, mh_sd);

    //     /*
    //     When rho is updated, we also need to update:
    //     - precision matrix Q
    //     - one_Q_one: updated in `update_car() -> compute_precision()`
    //     - post_mu_mean: updated in `log_posterior_rho()`
    //     - post_mu_prec: updated in `log_posterior_rho()`
    //     - post_tau2_rate: updated in `log_posterior_rho()`
    //     */
    //     // if (rho_new > spatial.min_car_rho && rho_new < spatial.max_car_rho)
    //     // {
    //     spatial.update_params(
    //         spatial.car_mu,
    //         spatial.car_tau2,
    //         rho_new);
    //     double log_post_rho_new = spatial.log_posterior_rho(spatial_effects, jeffrey_prior_order);

    //     double u_old = standardize(rho_current, rho_min, rho_max, true);
    //     double u_new = standardize(rho_new, rho_min, rho_max, true);
    //     double log_jac = (std::log(u_new) + std::log1p(-u_new)) - (std::log(u_old) + std::log1p(-u_old));

    //     double log_accept_ratio = log_post_rho_new - log_post_rho_old + log_jac;
    //     if (std::log(R::runif(0.0, 1.0)) < log_accept_ratio)
    //     {
    //         rho_accept_count += 1.0;
    //     }
    //     else
    //     {
    //         // revert
    //         spatial.update_params(
    //             spatial.car_mu,
    //             spatial.car_tau2,
    //             rho_current);
    //         double log_post_rho = spatial.log_posterior_rho(spatial_effects, jeffrey_prior_order);
    //     }
    //     // }

    //     /*
    //     When tau2 is updated, we also need to update:
    //     - post_mu_prec: update manually here
    //     */
    //     double tau2_new = R::rgamma(spatial.post_tau2_shape, 1.0 / spatial.post_tau2_rate);
    //     spatial.update_params(
    //         spatial.car_mu,
    //         tau2_new,
    //         spatial.car_rho
    //     );

    //     double mu_new = R::rnorm(spatial.post_mu_mean, std::sqrt(1.0 / spatial.post_mu_prec));
    //     spatial.update_params(
    //         mu_new,
    //         spatial.car_tau2,
    //         spatial.car_rho
    //     );

    //     return;        
    // } // end of update_car()

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

    double compute_log_joint_logalpha(
        const Model &model, 
        const arma::mat &Y, 
        const arma::mat &wt
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
            }
        }

        // Add CAR prior for log_alpha
        logp += model.spatial.log_likelihood(model.log_alpha, false);
        return logp;
    }

    double compute_log_joint_global(
        const Model &model, 
        const arma::mat &Y, 
        const arma::mat &wt
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
            }
        }

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
            logp += Prior::dprior(std::log(std::max(model.beta, EPS)), beta_prior, true, true);
        }

        return logp;
    }

    double compute_log_joint_local(
        const unsigned int &s,
        const Model &model, 
        const arma::vec &y, 
        const arma::vec &lambda,
        const arma::vec &wt
    )
    {
        double logp = 0.0;
        for (unsigned int t = 1; t <= y.n_elem - 1; t++)
        {
            // Compute loglikelihood of y[s, t]
            logp += ObsDist::loglike(
                y.at(t), model.dobs, lambda.at(t), model.rho.at(s), true);

            logp += R::dnorm4(wt.at(t), 0.0, std::sqrt(model.W.at(s)), true);
        }

        // Add local parameter priors
        if (rho_prior.infer)
        {
            logp += Prior::dprior(model.rho.at(s), rho_prior, true, true);
        }

        if (W_prior.infer)
        {
            logp += Prior::dprior(model.W.at(s), W_prior, true, true);
        }

        return logp;
    }

    double update_log_alpha(
        Model &model,
        double &energy_diff,
        double &grad_norm_out,
        const arma::mat &Y, 
        const arma::mat &wt,
        const double &leapfrog_step_size,
        const unsigned int &n_leapfrog = 10
    )
    {
        // Precompute hPsi (depends only on wt)
        arma::mat Psi = arma::cumsum(wt, 1);
        arma::mat hPsi = GainFunc::psi2hpsi<arma::mat>(Psi, model.fgain);
        arma::mat dll_deta = model.dloglik_deta(Y, hPsi); // nS x (nT + 1)

        double logp_current = compute_log_joint_logalpha(model, Y, wt);
        double energy_current = -logp_current;

        arma::vec log_alpha_current = model.log_alpha;
        arma::vec q = log_alpha_current;

        // sample an initial momentum
        arma::vec p = arma::randn(q.n_elem);
        double kinetic_current = 0.5 * arma::dot(p, p);

        arma::vec grad = model.dloglik_dlogalpha(Y, dll_deta);
        grad *= -1.0; // Convert to gradient of potential energy
        grad_norm_out = arma::norm(grad, 2);

        // Make a half step for momentum at the beginning
        p -= 0.5 * leapfrog_step_size * grad;
        for (unsigned int i = 0; i < n_leapfrog; i++)
        {
            // Make a full step for the position
            q += leapfrog_step_size * p;
            model.log_alpha = q;

            // Compute the new gradient
            dll_deta = model.dloglik_deta(Y, hPsi); // nS x (nT + 1)
            grad = model.dloglik_dlogalpha(Y, dll_deta);
            grad *= -1.0; // Convert to gradient of potential energy

            // Make a full step for the momentum, except at the end of trajectory
            if (i != n_leapfrog - 1)
            {
                p -= leapfrog_step_size * grad;
            }
        } // end of leapfrog steps

        p -= 0.5 * leapfrog_step_size * grad; // Make a half step for momentum at the end
        p *= -1; // Negate momentum to make the proposal symmetric

        double logp_proposed = compute_log_joint_logalpha(model, Y, wt);
        double energy_proposed = -logp_proposed;
        double kinetic_proposed = 0.5 * arma::dot(p, p);

        double H_proposed = energy_proposed + kinetic_proposed;
        double H_current = energy_current + kinetic_current;
        energy_diff = H_proposed - H_current;

        if (!std::isfinite(H_current) || !std::isfinite(H_proposed) || std::abs(energy_diff) > 100.0)
        {
            // Reject if either log probability is not finite or HMC diverged
            model.log_alpha = log_alpha_current;
            return 0.0;
        }
        else
        {
            double log_accept_ratio = H_current - H_proposed;
            if (std::log(runif()) >= log_accept_ratio)
            {
                // Reject: revert to current parameters
                model.log_alpha = log_alpha_current;
            }

            double accept_prob = std::min(1.0, std::exp(log_accept_ratio));
            return accept_prob;
        }
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
        double &energy_diff,
        double &grad_norm_out,
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
        double logp_current = compute_log_joint_global(model, Y, wt);
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

        grad_norm_out = arma::norm(grad, 2);

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

        double logp_proposed = compute_log_joint_global(model, Y, wt);
        double energy_proposed = -logp_proposed;
        double kinetic_proposed = 0.5 * arma::dot(p, p);

        double H_proposed = energy_proposed + kinetic_proposed;
        double H_current = energy_current + kinetic_current;
        energy_diff = H_proposed - H_current;

        if (!std::isfinite(H_current) || !std::isfinite(H_proposed) || std::abs(energy_diff) > 100.0)
        {
            // Reject if either log probability is not finite
            model.update_global_params_unconstrained(global_params_selected, global_params_current);
            return 0.0;
        }
        else
        {
            double log_accept_ratio = H_current - H_proposed;
            if (std::log(runif()) < log_accept_ratio)
            {
                // Accept
                if (lag_par1_prior.infer || lag_par2_prior.infer)
                {
                    model.dlag.update_nlag();
                    model.dlag.update_Fphi();
                    if (model.dlag.truncated)
                    {
                        model.nP = model.dlag.nL;
                    }
                }
            }
            else
            {
                // Reject: revert to current parameters
                model.update_global_params_unconstrained(global_params_selected, global_params_current);
            }

            double accept_prob = std::min(1.0, std::exp(log_accept_ratio));
            return accept_prob;
        }
    }

    /**
     * @brief HMC sampler for global parameters (lag distribution parameters and beta)
     * 
     * @param model 
     * @param Y 
     * @param wt 
     */
    double update_local_params(
        Model &model, 
        double &energy_diff,
        double &grad_norm_out,
        const std::vector<std::string> &local_params_selected,
        const unsigned int &s,
        const arma::mat &Y, 
        const arma::mat &wt,
        const double &leapfrog_step_size,
        const unsigned int &n_leapfrog = 10
    )
    {
        // Precompute hPsi (depends only on wt)
        arma::mat Psi = arma::cumsum(wt, 1); // nS x (nT + 1)
        arma::vec lambda = model.compute_intensity_iterative(s, Y, Psi.row(s).t()); // (nT + 1) x 1
        const arma::vec ys = Y.row(s).t(); // (nT + 1) x 1
        const arma::vec ws = wt.row(s).t(); // (nT + 1) x 1

        // Potential energy: negative log joint probability.
        double logp_current = compute_log_joint_local(s, model, ys, lambda, ws);
        double energy_current = -logp_current;

        // Get current global parameters in unconstrained space
        arma::vec local_params_current = model.get_local_params_unconstrained(s, local_params_selected);
        arma::vec q = local_params_current;

        // sample an initial momentum
        arma::vec p = arma::randn(q.n_elem);
        double kinetic_current = 0.5 * arma::dot(p, p);

        arma::vec grad = model.dloglik_dlocal_unconstrained(
            s, local_params_selected, 
            ys, lambda, ws,
            rho_prior, 
            W_prior
        );

        grad *= -1.0; // Convert to gradient of potential energy

        grad_norm_out = arma::norm(grad, 2);

        // Make a half step for momentum at the beginning
        p -= 0.5 * leapfrog_step_size * grad;
        for (unsigned int i = 0; i < n_leapfrog; i++)
        {
            // Make a full step for the position
            q += leapfrog_step_size * p;
            model.update_local_params_unconstrained(s, local_params_selected, q);

            // Compute the new gradient
            grad = model.dloglik_dlocal_unconstrained(
                s, local_params_selected, 
                ys, lambda, ws,
                rho_prior, 
                W_prior
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

        double logp_proposed = compute_log_joint_local(s, model, ys, lambda, ws);
        double energy_proposed = -logp_proposed;
        double kinetic_proposed = 0.5 * arma::dot(p, p);

        double H_proposed = energy_proposed + kinetic_proposed;
        double H_current = energy_current + kinetic_current;
        energy_diff = H_proposed - H_current;

        if (!std::isfinite(H_current) || !std::isfinite(H_proposed) || std::abs(energy_diff) > 100.0)
        {
            // Reject if either log probability is not finite
            model.update_local_params_unconstrained(s, local_params_selected, local_params_current);
            return 0.0;
        }
        else
        {
            double log_accept_ratio = H_current - H_proposed;
            if (std::log(runif()) >= log_accept_ratio)
            {
                // Reject: revert to current parameters
                model.update_local_params_unconstrained(s, local_params_selected, local_params_current);
            }

            double accept_prob = std::min(1.0, std::exp(log_accept_ratio));
            return accept_prob;
        }
    }


    void check_grad_global(Model &model, const arma::mat &Y, const arma::mat &wt,
                    const std::vector<std::string> &names)
    {
        arma::vec q = model.get_global_params_unconstrained(names);
        arma::mat Psi = arma::cumsum(wt, 1);
        arma::mat hPsi = GainFunc::psi2hpsi<arma::mat>(Psi, model.fgain);
        arma::vec g = model.dloglik_dglobal_unconstrained(names, Y, hPsi,
                                                          lag_par1_prior, lag_par2_prior, beta_prior);
        double eps_fd = 1e-5;
        for (unsigned int i = 0; i < q.n_elem; i++)
        {
            arma::vec q1 = q, q2 = q;
            q1[i] += eps_fd;
            q2[i] -= eps_fd;
            model.update_global_params_unconstrained(names, q1);
            double lp1 = compute_log_joint_global(model, Y, wt);
            model.update_global_params_unconstrained(names, q2);
            double lp2 = compute_log_joint_global(model, Y, wt);
            double fd = (lp1 - lp2) / (2 * eps_fd);
            Rcpp::Rcout << "Param " << i << " analytic=" << g[i] << " fd=" << fd
                        << " rel.err=" << std::fabs(g[i] - fd) / std::max(1.0, std::fabs(fd)) << "\n";
            model.update_global_params_unconstrained(names, q); // restore
        }
    }

    void check_grad_local(
        Model &model, 
        const unsigned int &s, 
        const arma::vec &y, 
        const arma::vec &lambda, 
        const arma::vec &wt,
        const std::vector<std::string> &names)
    {
        arma::vec q = model.get_local_params_unconstrained(s, names);
        arma::vec g = model.dloglik_dlocal_unconstrained(s, names, y, lambda, wt, rho_prior, W_prior);
        double eps_fd = 1e-5;
        for (unsigned int i = 0; i < q.n_elem; i++)
        {
            arma::vec q1 = q, q2 = q;
            q1[i] += eps_fd;
            q2[i] -= eps_fd;
            model.update_local_params_unconstrained(s, names, q1);
            double lp1 = compute_log_joint_local(s, model, y, lambda, wt);
            model.update_local_params_unconstrained(s, names, q2);
            double lp2 = compute_log_joint_local(s, model, y, lambda, wt);
            double fd = (lp1 - lp2) / (2 * eps_fd);
            Rcpp::Rcout << "Location " << s << " param " << i << " analytic=" << g[i] << " fd=" << fd
                        << " rel.err=" << std::fabs(g[i] - fd) / std::max(1.0, std::fabs(fd)) << "\n";
            model.update_local_params_unconstrained(s, names, q); // restore
        }
    }

    void check_grad_logalpha(Model &model, const arma::mat &Y, const arma::mat &wt)
    {
        arma::mat Psi = arma::cumsum(wt, 1);
        arma::mat hPsi = GainFunc::psi2hpsi<arma::mat>(Psi, model.fgain);
        arma::mat dll_deta = model.dloglik_deta(Y, hPsi); // nS x (nT + 1)

        arma::vec q = model.log_alpha;
        arma::vec g = model.dloglik_dlogalpha(Y, dll_deta);
        double eps_fd = 1e-5;
        for (unsigned int i = 0; i < q.n_elem; i++)
        {
            arma::vec q1 = q, q2 = q;
            q1[i] += eps_fd;
            q2[i] -= eps_fd;
            model.log_alpha = q1;
            double lp1 = compute_log_joint_logalpha(model, Y, wt);
            model.log_alpha = q2;
            double lp2 = compute_log_joint_logalpha(model, Y, wt);
            double fd = (lp1 - lp2) / (2 * eps_fd);
            Rcpp::Rcout << "log_alpha " << i << " analytic=" << g[i] << " fd=" << fd
                        << " rel.err=" << std::fabs(g[i] - fd) / std::max(1.0, std::fabs(fd)) << "\n";

            model.log_alpha = q; // restore
        }

        return;
    }


    void infer(Model &model, const arma::mat &Y, const arma::mat &wt_in)
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
        unsigned min_leaps = 3, max_leaps = 128;

        if (!global_params_selected.empty() && global_diagnostics)
        {
            global_energy_diff = arma::vec(niter, arma::fill::zeros);
            global_grad_norm = arma::vec(niter, arma::fill::zeros);
            
            if (global_dual_averaging)
            {
                global_nleapfrog_stored = arma::vec(nburnin + 1, arma::fill::zeros);
                global_leapfrog_step_size_stored = arma::vec(nburnin + 1, arma::fill::zeros);
            }
        }

        
        if (!local_params_selected.empty() && local_diagnostics)
        {
            local_accept_count = arma::vec(model.nS, arma::fill::zeros);
            local_nleapfrog = arma::uvec(model.nS, arma::fill::value(local_nleapfrog_init));
            local_leapfrog_step_size = arma::vec(model.nS, arma::fill::value(local_leapfrog_step_size_init));

            local_energy_diff = arma::mat(model.nS, niter, arma::fill::zeros);
            local_grad_norm = arma::mat(model.nS, niter, arma::fill::zeros);

            if (local_dual_averaging)
            {
                local_nleapfrog_stored = arma::mat(model.nS, nburnin + 1, arma::fill::zeros);
                local_leapfrog_step_size_stored = arma::mat(model.nS, nburnin + 1, arma::fill::zeros);
            }
        }

        arma::vec local_mu_da = arma::log(10.0 * local_leapfrog_step_size); // bias center
        arma::vec local_log_eps = arma::log(local_leapfrog_step_size);
        arma::vec local_log_eps_bar = local_log_eps;
        arma::vec local_h_bar(local_mu_da.n_elem, arma::fill::zeros);
        arma::vec local_gamma_da(local_mu_da.n_elem, arma::fill::value(0.05));
        arma::vec local_t0_da(local_mu_da.n_elem, arma::fill::value(10.0));
        arma::vec local_kappa_da(local_mu_da.n_elem, arma::fill::value(0.75));
        arma::uvec local_adapt_count(local_mu_da.n_elem, arma::fill::zeros);

        if (infer_log_alpha)
        {
            log_alpha_stored = arma::mat(model.nS, nsample, arma::fill::zeros);

            if (logalpha_diagnostics)
            {
                logalpha_energy_diff = arma::vec(niter, arma::fill::zeros);
                logalpha_grad_norm = arma::vec(niter, arma::fill::zeros);

                if (logalpha_dual_averaging)
                {
                    logalpha_leapfrog_step_size_stored = arma::vec(nburnin + 1, arma::fill::zeros);
                    logalpha_nleapfrog_stored = arma::vec(nburnin + 1, arma::fill::zeros);
                }
            }

        }

        double logalpha_mu_da = std::log(10.0 * logalpha_leapfrog_step_size); // bias center
        double logalpha_log_eps = std::log(logalpha_leapfrog_step_size);
        double logalpha_log_eps_bar = logalpha_log_eps;
        double logalpha_h_bar = 0.0;
        double logalpha_gamma_da = 0.05;
        double logalpha_t0_da = 10.0;
        double logalpha_kappa_da = 0.75;
        unsigned int logalpha_adapt_count = 0;


        if (bym2_prior.infer)
        {
            bym2_mu_stored = arma::vec(nsample, arma::fill::zeros);
            bym2_tau_b_stored = arma::vec(nsample, arma::fill::zeros);
            bym2_phi_stored = arma::vec(nsample, arma::fill::zeros);

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

        if (rho_prior.infer)
        {
            rho_stored = arma::mat(model.nS, nsample, arma::fill::zeros);
        }

        if (W_prior.infer)
        {
            W_stored = arma::mat(model.nS, nsample, arma::fill::zeros);
        }

        // arma::mat wt(model.nS, nT + 1, arma::fill::randn);
        // wt.each_col() %= arma::sqrt(model.W);
        arma::mat wt = wt_in;

        Progress p(niter, true);
        for (unsigned int iter = 0; iter < niter; ++iter)
        {
            if (wt_prior.infer)
            {
                update_wt(wt, wt_accept, model, Y, wt_prior.mh_sd);
            } // end of wt update

            if (bym2_prior.infer)
            {
                double acc = model.spatial.update_phi_logit_marginal(model.log_alpha, bym2_prior);
                bym2_prior.accept_count += acc;
                model.spatial.update_mu_tau_jointly(
                    model.log_alpha, 
                    bym2_prior.shape_tau, 
                    bym2_prior.rate_tau
                );
                bym2_prior.adapt_phi_proposal_robbins_monro(iter, nburnin);
            } // end of car update

            if (!global_params_selected.empty())
            {
                if (global_verbose && iter % 100 == 0 && iter < nburnin)
                {
                    check_grad_global(model, Y, wt, global_params_selected);
                }


                double energy_diff, grad_norm;
                double global_hmc_accept_prob = update_global_params(
                    model, energy_diff, grad_norm, 
                    global_params_selected, Y, wt, 
                    global_leapfrog_step_size, 
                    global_nleapfrog
                );
                global_accept_count += global_hmc_accept_prob;

                if (global_diagnostics)
                {
                    global_energy_diff(iter) = energy_diff;
                    global_grad_norm(iter) = grad_norm;
                }


                if (global_dual_averaging)
                {
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
                        unsigned nlf = (unsigned)std::lround(global_T_target / global_leapfrog_step_size);
                        global_nleapfrog = std::max(min_leaps, std::min(max_leaps, nlf));
                    }

                    if (global_diagnostics && iter <= nburnin)
                    {
                        global_nleapfrog_stored(iter) = global_nleapfrog;
                        global_leapfrog_step_size_stored(iter) = global_leapfrog_step_size;
                    }
                } // end of global dual averaging
            } // end of global params update

            if (!local_params_selected.empty())
            {
                for (unsigned int s = 0; s < model.nS; s++)
                {
                    if (local_verbose && iter % 100 == 0 && iter < nburnin)
                    {
                        arma::vec ys = Y.row(s).t();
                        arma::vec ws = wt.row(s).t();
                        arma::vec lambda = model.compute_intensity_iterative(s, Y, arma::cumsum(ws));
                        check_grad_local(model, s, ys, lambda, ws, local_params_selected);
                    }

                    double energy_diff, grad_norm;
                    double local_hmc_accept_prob = update_local_params(
                        model, energy_diff, grad_norm, 
                        local_params_selected, s, Y, wt, 
                        local_leapfrog_step_size.at(s), 
                        local_nleapfrog.at(s)
                    );
                    local_accept_count.at(s) += local_hmc_accept_prob;

                    if (local_diagnostics)
                    {
                        local_energy_diff(s, iter) = energy_diff;
                        local_grad_norm(s, iter) = grad_norm;
                    }

                    if (local_dual_averaging)
                    {
                        if (iter < nburnin)
                        {
                            local_adapt_count[s]++;
                            double t = (double)local_adapt_count[s];
                            local_h_bar[s] = (1.0 - 1.0 / (t + local_t0_da[s])) * local_h_bar[s] + (1.0 / (t + local_t0_da[s])) * (target_accept - local_hmc_accept_prob);
                            local_log_eps[s] = local_mu_da[s] - (std::sqrt(t) / local_gamma_da[s]) * local_h_bar[s];
                            local_leapfrog_step_size.at(s) = clamp_eps(std::exp(local_log_eps[s]));
                            double w = std::pow(t, -local_kappa_da[s]);
                            local_log_eps_bar.at(s) = w * std::log(local_leapfrog_step_size.at(s)) + (1.0 - w) * local_log_eps_bar.at(s);
                        }
                        else if (iter == nburnin)
                        {
                            local_leapfrog_step_size.at(s) = clamp_eps(std::exp(local_log_eps_bar.at(s)));
                            unsigned nlf = (unsigned)std::lround(local_T_target / local_leapfrog_step_size.at(s));
                            local_nleapfrog.at(s) = std::max(min_leaps, std::min(max_leaps, nlf));
                        }

                        if (local_diagnostics && iter <= nburnin)
                        {
                            local_nleapfrog_stored(s, iter) = local_nleapfrog.at(s);
                            local_leapfrog_step_size_stored(s, iter) = local_leapfrog_step_size.at(s);
                        }
                    } // end of local dual averaging
                } // end of s loop
            } // end of local params update

            if (infer_log_alpha)
            {
                if (logalpha_verbose && iter % 100 == 0 && iter < nburnin)
                {
                    check_grad_logalpha(model, Y, wt);
                }

                double energy_diff, grad_norm;
                double logalpha_hmc_accept_prob = update_log_alpha(
                    model, energy_diff, grad_norm, Y, wt, 
                    logalpha_leapfrog_step_size, logalpha_nleapfrog
                );
                logalpha_accept_count += logalpha_hmc_accept_prob;

                if (logalpha_diagnostics)
                {
                    logalpha_energy_diff.at(iter) = energy_diff;
                    logalpha_grad_norm.at(iter) = grad_norm;
                }

                if (logalpha_dual_averaging)
                {
                    if (iter < nburnin)
                    {
                        logalpha_adapt_count++;
                        double t = (double)logalpha_adapt_count;
                        logalpha_h_bar = (1.0 - 1.0 / (t + logalpha_t0_da)) * logalpha_h_bar + (1.0 / (t + logalpha_t0_da)) * (target_accept - logalpha_hmc_accept_prob);
                        logalpha_log_eps = logalpha_mu_da - (std::sqrt(t) / logalpha_gamma_da) * logalpha_h_bar;
                        logalpha_leapfrog_step_size = clamp_eps(std::exp(logalpha_log_eps));
                        double w = std::pow(t, -logalpha_kappa_da);
                        logalpha_log_eps_bar = w * std::log(logalpha_leapfrog_step_size) + (1.0 - w) * logalpha_log_eps_bar;
                    }
                    else if (iter == nburnin)
                    {
                        logalpha_leapfrog_step_size = clamp_eps(std::exp(logalpha_log_eps_bar));
                        unsigned nlf = (unsigned)std::lround(logalpha_T_target / logalpha_leapfrog_step_size);
                        logalpha_nleapfrog = std::max(min_leaps, std::min(max_leaps, nlf));
                    }

                    if (logalpha_diagnostics && iter <= nburnin)
                    {
                        logalpha_nleapfrog_stored.at(iter) = logalpha_nleapfrog;
                        logalpha_leapfrog_step_size_stored.at(iter) = logalpha_leapfrog_step_size;
                    }
                } // end of dual averaging
            } // end of infer_log_alpha



            // Store samples after burn-in and thinning
            if (iter >= nburnin && ((iter - nburnin) % nthin == 0))
            {
                const unsigned int sample_idx = (iter - nburnin) / nthin;
                if (wt_prior.infer)
                {
                    wt_stored.slice(sample_idx) = wt;
                }

                if (bym2_prior.infer)
                {
                    bym2_mu_stored(sample_idx) = model.spatial.mu;
                    bym2_tau_b_stored(sample_idx) = model.spatial.tau_b;
                    bym2_phi_stored(sample_idx) = model.spatial.phi;
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

                if (rho_prior.infer)
                {
                    for (unsigned int s = 0; s < model.nS; s++)
                    {
                        rho_stored(s, sample_idx) = model.rho.at(s);
                    }
                }

                if (W_prior.infer)
                {
                    for (unsigned int s = 0; s < model.nS; s++)
                    {
                        W_stored(s, sample_idx) = model.W.at(s);
                    }
                }

                if (infer_log_alpha)
                {
                    log_alpha_stored.col(sample_idx) = model.log_alpha;
                }
            } // end of store samples

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

        if (bym2_prior.infer || infer_log_alpha)
        {
            Rcpp::List spatial_output;
            if (bym2_prior.infer)
            {
                spatial_output["mu"] = Rcpp::wrap(bym2_mu_stored);
                spatial_output["tau_b"] = Rcpp::wrap(bym2_tau_b_stored);
                spatial_output["phi"] = Rcpp::wrap(bym2_phi_stored);
                spatial_output["phi_accept_rate"] = bym2_prior.accept_count / (nburnin + nsample * nthin);
                spatial_output["final_phi_mh_sd"] = bym2_prior.mh_sd;
            } // end of BYM2 output

            if (infer_log_alpha)
            {
                spatial_output["log_alpha"] = Rcpp::wrap(log_alpha_stored);
                spatial_output["logalpha_accept_rate"] = logalpha_accept_count / (nburnin + nsample * nthin);
                spatial_output["logalpha_hmc_settings"] = Rcpp::List::create(
                    Rcpp::Named("nleapfrog") = logalpha_nleapfrog,
                    Rcpp::Named("leapfrog_step_size") = logalpha_leapfrog_step_size);

                if (logalpha_diagnostics && logalpha_dual_averaging)
                {
                    spatial_output["logalpha_diagnostics"] = Rcpp::List::create(
                        Rcpp::Named("energy_diff") = logalpha_energy_diff,
                        Rcpp::Named("grad_norm") = logalpha_grad_norm,
                        Rcpp::Named("n_leapfrog") = logalpha_nleapfrog_stored,
                        Rcpp::Named("step_size") = logalpha_leapfrog_step_size_stored);
                }
                else if (logalpha_diagnostics)
                {
                    spatial_output["logalpha_diagnostics"] = Rcpp::List::create(
                        Rcpp::Named("energy_diff") = logalpha_energy_diff,
                        Rcpp::Named("grad_norm") = logalpha_grad_norm);
                }
            } // end of log_alpha output

            output["spatial"] = spatial_output;
        } // end of spatial output

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

            if (global_diagnostics && global_dual_averaging)
            {
                output["global_diagnostics"] = Rcpp::List::create(
                    Rcpp::Named("energy_diff") = global_energy_diff,
                    Rcpp::Named("grad_norm") = global_grad_norm,
                    Rcpp::Named("n_leapfrog") = global_nleapfrog_stored,
                    Rcpp::Named("step_size") = global_leapfrog_step_size_stored
                );
            }
            else if (global_diagnostics)
            {
                output["global_diagnostics"] = Rcpp::List::create(
                    Rcpp::Named("energy_diff") = global_energy_diff,
                    Rcpp::Named("grad_norm") = global_grad_norm
                );
            }
        } // end of global params output

        if (!local_params_selected.empty())
        {
            output["local_accept_rate"] = Rcpp::wrap(local_accept_count / (nburnin + nsample * nthin));
            output["local_hmc_settings"] = Rcpp::List::create(
                Rcpp::Named("nleapfrog") = local_nleapfrog,
                Rcpp::Named("leapfrog_step_size") = local_leapfrog_step_size
            );

            if (rho_prior.infer)
            {
                output["rho"] = Rcpp::wrap(rho_stored);
            }
            if (W_prior.infer)
            {
                output["W"] = Rcpp::wrap(W_stored);
            }

            if (local_diagnostics && local_dual_averaging)
            {
                output["local_diagnostics"] = Rcpp::List::create(
                    Rcpp::Named("energy_diff") = local_energy_diff,
                    Rcpp::Named("grad_norm") = local_grad_norm,
                    Rcpp::Named("n_leapfrog") = local_nleapfrog_stored,
                    Rcpp::Named("step_size") = local_leapfrog_step_size_stored
                );
            }
            else if (local_diagnostics)
            {
                output["local_diagnostics"] = Rcpp::List::create(
                    Rcpp::Named("energy_diff") = local_energy_diff,
                    Rcpp::Named("grad_norm") = local_grad_norm
                );
            }
        } // end of local params output

        return output;
    } // end of get_output()
}; // class mcmc


#endif