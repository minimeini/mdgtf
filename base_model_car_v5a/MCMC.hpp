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

#include "Model.hpp"

// [[Rcpp::plugins(cpp17)]]
// [[Rcpp::depends(RcppArmadillo,RcppProgress)]]


class MCMC
{
private:
    unsigned int nsample = 1000;
    unsigned int nburnin = 1000;
    unsigned int nthin = 1;

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

    // Prior for disturbance wt
    Prior wt_prior;

    // Priors for local parameters
    Prior rho_prior;

    // Priors for global parameters
    Prior a_intercept_prior;
    Prior a_sigma2_prior;
    Prior coef_self_intercept_prior;
    Prior coef_cross_intercept_prior;

    // Storage for global parameter samples
    arma::vec a_intercept_stored; // nsample x 1
    arma::vec a_sigma2_stored; // nsample x 1
    arma::vec coef_self_intercept_stored; // nsample x 1
    arma::vec coef_cross_intercept_stored; // nsample x 1

    // Storage for local parameter samples
    arma::mat rho_stored; // nS x nsample

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

public:
    
    MCMC(
        const unsigned int &nsample_in = 1000,
        const unsigned int &nburnin_in = 1000,
        const unsigned int &nthin_in = 1,
        const bool &infer_wt_in = false
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
        } // end of log_a options

        if (opts.containsElementNamed("rho"))
        {
            Rcpp::List rho_opts = Rcpp::as<Rcpp::List>(opts["rho"]);
            rho_prior.init(rho_opts);
        }
        if (rho_prior.infer)
        {
            local_params_selected.push_back("rho");
        } // end of rho options

        if (opts.containsElementNamed("a_intercept"))
        {
            Rcpp::List a_intercept_opts = Rcpp::as<Rcpp::List>(opts["a_intercept"]);
            a_intercept_prior.init(a_intercept_opts);
        }
        if (a_intercept_prior.infer)
        {
            global_params_selected.push_back("a_intercept");
        } // end of a_intercept options

        if (opts.containsElementNamed("a_sigma2"))
        {
            Rcpp::List W_opts = Rcpp::as<Rcpp::List>(opts["a_sigma2"]);
            a_sigma2_prior.init(W_opts);
        }
        if (a_sigma2_prior.infer)
        {
            global_params_selected.push_back("a_sigma2");
        } // end of a_sigma2 options

        if (opts.containsElementNamed("coef_self_intercept"))
        {
            Rcpp::List coef_self_intercept_opts = Rcpp::as<Rcpp::List>(opts["coef_self_intercept"]);
            coef_self_intercept_prior.init(coef_self_intercept_opts);
        }
        if (coef_self_intercept_prior.infer)
        {
            global_params_selected.push_back("coef_self_intercept");
        } // end of coef_self_intercept options

        if (opts.containsElementNamed("coef_cross_intercept"))
        {
            Rcpp::List coef_cross_intercept_opts = Rcpp::as<Rcpp::List>(opts["coef_cross_intercept"]);
            coef_cross_intercept_prior.init(coef_cross_intercept_opts);
        }
        if (coef_cross_intercept_prior.infer)
        {
            global_params_selected.push_back("coef_cross_intercept");
        } // end of coef_cross_intercept options

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
        } // end of global_hmc options

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
        } // end of local_hmc options

        return;
    } // end of constructor from Rcpp::List

    static Rcpp::List get_default_settings()
    {
        Rcpp::List wt_opts = Rcpp::List::create(
            Rcpp::Named("infer") = false,
            Rcpp::Named("mh_sd") = 1.0
        );

        Rcpp::List rho_opts = Rcpp::List::create(
            Rcpp::Named("infer") = false,
            Rcpp::Named("prior_name") = "invgamma",
            Rcpp::Named("prior_param") = Rcpp::NumericVector::create(1.0, 1.0)
        );

        Rcpp::List a_intercept_opts = Rcpp::List::create(
            Rcpp::Named("infer") = false,
            Rcpp::Named("prior_name") = "gaussian",
            Rcpp::Named("prior_param") = Rcpp::NumericVector::create(0.0, 10.0)
        );

        Rcpp::List a_sigma2_opts = Rcpp::List::create(
            Rcpp::Named("infer") = false,
            Rcpp::Named("prior_name") = "invgamma",
            Rcpp::Named("prior_param") = Rcpp::NumericVector::create(1.0, 1.0)
        );

        Rcpp::List coef_self_intercept_opts = Rcpp::List::create(
            Rcpp::Named("infer") = false,
            Rcpp::Named("prior_name") = "gaussian",
            Rcpp::Named("prior_param") = Rcpp::NumericVector::create(0.0, 10.0)
        );

        Rcpp::List coef_cross_intercept_opts = Rcpp::List::create(
            Rcpp::Named("infer") = false,
            Rcpp::Named("prior_name") = "gaussian",
            Rcpp::Named("prior_param") = Rcpp::NumericVector::create(0.0, 10.0)
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
            Rcpp::Named("nleapfrog") = 20,
            Rcpp::Named("leapfrog_step_size") = 0.1,
            Rcpp::Named("dual_averaging") = false,
            Rcpp::Named("diagnostics") = true,
            Rcpp::Named("verbose") = false,
            Rcpp::Named("T_target") = 2.0
        );

        Rcpp::List logalpha_hmc_opts = Rcpp::List::create(
            Rcpp::Named("nleapfrog") = 20,
            Rcpp::Named("leapfrog_step_size") = 0.1,
            Rcpp::Named("dual_averaging") = false,
            Rcpp::Named("diagnostics") = true,
            Rcpp::Named("verbose") = false,
            Rcpp::Named("T_target") = 2.0
        );

        Rcpp::List mcmc_opts = Rcpp::List::create(
            Rcpp::Named("nsample") = 1000,
            Rcpp::Named("nburnin") = 1000,
            Rcpp::Named("nthin") = 1,
            Rcpp::Named("wt") = wt_opts,
            Rcpp::Named("a_intercept") = a_intercept_opts,
            Rcpp::Named("a_sigma2") = a_sigma2_opts,
            Rcpp::Named("coef_self_intercept") = coef_self_intercept_opts,
            Rcpp::Named("coef_cross_intercept") = coef_cross_intercept_opts,
            Rcpp::Named("rho") = rho_opts,
            Rcpp::Named("global_hmc") = global_hmc_opts,
            Rcpp::Named("local_hmc") = local_hmc_opts
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
        const double sigma2 = std::max(model.intercept_a.sigma2, EPS);
        const double prior_sd = std::sqrt(sigma2);
        const double log_norm_c = -0.5 * std::log(2.0 * M_PI) - std::log(prior_sd);
        const double coef_self = std::exp(std::min(model.coef_self_b.intercept, UPBND));
        const double coef_cross = std::exp(std::min(model.coef_cross_c.intercept, UPBND));
        
        #ifdef DGTF_USE_OPENMP
        const bool _omp_enable = (model.nS > 1);
        #pragma omp parallel for if(_omp_enable) schedule(static)
        #endif
        for (unsigned int s = 0; s < model.nS; s++)
        {
            const arma::vec neighbor_weights = model.spatial.W.row(s).t();
            const arma::vec ys = Y.row(s).t(); // (nT+1)

            // Precompute self and cross with 0 at t=0
            arma::vec self(nT + 1, arma::fill::zeros);
            self.tail(nT) = coef_self * ys.head(nT);

            arma::vec cross(nT + 1, arma::fill::zeros);
            arma::vec cross_tail = Y.head_cols(nT).t() * neighbor_weights; // nT x 1
            cross.tail(nT) = coef_cross * cross_tail;

            arma::vec acc_row(nT + 1, arma::fill::zeros);
            arma::vec ws = wt.row(s).t();
            arma::vec log_a = arma::cumsum(ws);

            arma::vec lam_old = model.compute_intensity_iterative(s, Y, log_a);

            for (unsigned int t = 1; t <= nT; t++)
            {
                const double w_ts_old = ws.at(t);

                // Recompute exp(a), lambda and Fisher scale at current state
                arma::vec exp_a = arma::exp(model.intercept_a.intercept + log_a); // (nT+1)
                arma::vec lambda_vec = exp_a + self + cross; // (nT+1)
                arma::vec V = lambda_vec % (lambda_vec + model.rho.at(s)) / model.rho.at(s);
                arma::vec F = arma::square(exp_a) / arma::clamp(V, EPS, arma::datum::inf);

                double mh_prec = arma::accu(F.subvec(t, nT)) + 1.0 / sigma2;
                double mh_step = std::sqrt(1.0 / std::max(mh_prec, EPS)) * mh_sd;

                // Old log joint (prior + affected likelihoods)
                double logp_old = log_norm_c - 0.5 * (w_ts_old * w_ts_old) / sigma2;
                for (unsigned int i = t; i <= nT; i++)
                {
                    logp_old += ObsDist::loglike(
                        ys.at(i), model.dobs, lam_old.at(i), model.rho.at(s), true
                    );
                }

                // Propose
                double w_ts_new = rnorm(w_ts_old, mh_step);
                ws.at(t) = w_ts_new;
                log_a = arma::cumsum(ws);
                arma::vec lam_new = model.compute_intensity_iterative(s, Y, log_a);

                double logp_new = log_norm_c - 0.5 * (w_ts_new * w_ts_new) / sigma2;
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
                    log_a = arma::cumsum(ws);
                }
            } // end of loop over t

            wt.row(s) = ws.t();
            wt_accept.row(s) += acc_row.t();
        } // end of loop over s
    } // end of update_wt()


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

        // Prior p(wt | sigma^2) for all s,t>=1 (Gaussian N(0, sigma^2))
        {
            const double sigma2 = std::max(model.intercept_a.sigma2, EPS);
            const double sd = std::sqrt(sigma2);
            const double log_norm_c = -0.5 * std::log(2.0 * M_PI) - std::log(sd);
            for (unsigned int s = 0; s < model.nS; s++)
            {
                for (unsigned int t = 1; t < Y.n_cols; t++)
                {
                    const double w = wt.at(s, t);
                    logp += log_norm_c - 0.5 * (w * w) / sigma2;
                }
            }
        }

        if (a_intercept_prior.infer)
        {
            // normal prior on the intercept (log scale) of the baseline intensity
            logp += Prior::dprior(model.intercept_a.intercept, a_intercept_prior, true, false);
        }

        // Add global parameter priors
        if (a_sigma2_prior.infer)
        {
            // inverse-gamma prior on the variance of the intercept (log scale)
            logp += Prior::dprior(model.intercept_a.sigma2, a_sigma2_prior, true, true);
        }

        if (coef_self_intercept_prior.infer)
        {
            // normal prior on the intercept (log scale) of self-exciting effects
            logp += Prior::dprior(model.coef_self_b.intercept, coef_self_intercept_prior, true, false);
        }

        if (coef_cross_intercept_prior.infer)
        {
            // normal prior on the intercept (log scale) of cross-region effects
            logp += Prior::dprior(model.coef_cross_c.intercept, coef_cross_intercept_prior, true, false);
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
        double sd = std::sqrt(model.intercept_a.sigma2);
        for (unsigned int t = 1; t <= y.n_elem - 1; t++)
        {
            // Compute loglikelihood of y[s, t]
            logp += ObsDist::loglike(
                y.at(t), model.dobs, lambda.at(t), model.rho.at(s), true);

            logp += R::dnorm4(wt.at(t), 0.0, sd, true);
        }

        // Add local parameter priors
        if (rho_prior.infer)
        {
            logp += Prior::dprior(model.rho.at(s), rho_prior, true, true);
        }

        return logp;
    } // end of compute_log_joint_local()


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
            global_params_selected, Y, wt, 
            a_intercept_prior,
            a_sigma2_prior,
            coef_self_intercept_prior,
            coef_cross_intercept_prior
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
                global_params_selected, Y, wt, 
                a_intercept_prior,
                a_sigma2_prior,
                coef_self_intercept_prior,
                coef_cross_intercept_prior
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
        arma::mat log_a = arma::cumsum(wt, 1); // nS x (nT + 1)
        arma::vec lambda = model.compute_intensity_iterative(s, Y, log_a.row(s).t()); // (nT + 1) x 1
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
            rho_prior
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
                rho_prior
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
        arma::vec g = model.dloglik_dglobal_unconstrained(
            names, Y, wt, 
            a_intercept_prior,
            a_sigma2_prior,
            coef_self_intercept_prior,
            coef_cross_intercept_prior
        );
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
        arma::vec g = model.dloglik_dlocal_unconstrained(s, names, y, lambda, wt, rho_prior);
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

        arma::mat wt = wt_in;
        wt.col(0).zeros();
        if (wt_prior.infer)
        {
            wt_stored = arma::cube(model.nS, nT + 1, nsample, arma::fill::zeros);
            wt_accept = arma::mat(model.nS, nT + 1, arma::fill::zeros);
        }

        if (!model.intercept_a.has_intercept)
        {
            a_intercept_prior.infer = false;
        }

        if (a_intercept_prior.infer)
        {
            a_intercept_stored = arma::vec(nsample, arma::fill::zeros);
        }

        if (!model.intercept_a.has_temporal)
        {
            a_sigma2_prior.infer = false;
        }
        if (a_sigma2_prior.infer)
        {
            a_sigma2_stored = arma::vec(nsample, arma::fill::zeros);
        }

        if (!model.coef_self_b.has_intercept)
        {
            coef_self_intercept_prior.infer = false;
        }
        if (coef_self_intercept_prior.infer)
        {
            coef_self_intercept_stored = arma::vec(nsample, arma::fill::zeros);
        }

        if (!model.coef_cross_c.has_intercept)
        {
            coef_cross_intercept_prior.infer = false;
        }
        if (coef_cross_intercept_prior.infer)
        {
            coef_cross_intercept_stored = arma::vec(nsample, arma::fill::zeros);
        }

        if (rho_prior.infer)
        {
            rho_stored = arma::mat(model.nS, nsample, arma::fill::zeros);
        }

        // arma::mat wt(model.nS, nT + 1, arma::fill::randn);
        // wt.each_col() %= arma::sqrt(model.W);
        Progress p(niter, true);
        for (unsigned int iter = 0; iter < niter; ++iter)
        {
            if (wt_prior.infer)
            {
                update_wt(wt, wt_accept, model, Y, wt_prior.mh_sd);
            } // end of wt update

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


            // Store samples after burn-in and thinning
            if (iter >= nburnin && ((iter - nburnin) % nthin == 0))
            {
                const unsigned int sample_idx = (iter - nburnin) / nthin;
                if (wt_prior.infer)
                {
                    wt_stored.slice(sample_idx) = wt;
                }

                if (a_intercept_prior.infer)
                {
                    a_intercept_stored.at(sample_idx) = model.intercept_a.intercept;
                }

                if (a_sigma2_prior.infer)
                {
                    a_sigma2_stored.at(sample_idx) = model.intercept_a.sigma2;
                }
                
                if (coef_self_intercept_prior.infer)
                {
                    coef_self_intercept_stored.at(sample_idx) = model.coef_self_b.intercept;
                }

                if (coef_cross_intercept_prior.infer)
                {
                    coef_cross_intercept_stored.at(sample_idx) = model.coef_cross_c.intercept;
                }

                if (rho_prior.infer)
                {
                    for (unsigned int s = 0; s < model.nS; s++)
                    {
                        rho_stored(s, sample_idx) = model.rho.at(s);
                    }
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

        if (!global_params_selected.empty())
        {
            output["global_accept_rate"] = global_accept_count / (nburnin + nsample * nthin);
            output["global_hmc_settings"] = Rcpp::List::create(
                Rcpp::Named("nleapfrog") = global_nleapfrog,
                Rcpp::Named("leapfrog_step_size") = global_leapfrog_step_size
            );

            if (a_intercept_prior.infer)
            {
                output["a_intercept"] = Rcpp::wrap(a_intercept_stored);
            }
            if (a_sigma2_prior.infer)
            {
                output["a_sigma2"] = Rcpp::wrap(a_sigma2_stored);
            }
            if (coef_self_intercept_prior.infer)
            {
                output["coef_self_intercept"] = Rcpp::wrap(coef_self_intercept_stored);
            }
            if (coef_cross_intercept_prior.infer)
            {
                output["coef_cross_intercept"] = Rcpp::wrap(coef_cross_intercept_stored);
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