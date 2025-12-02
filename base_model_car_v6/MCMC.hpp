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
#include "../inference/hmc.hpp"

// [[Rcpp::plugins(cpp17)]]
// [[Rcpp::depends(RcppArmadillo,RcppProgress)]]


template <typename T> 
struct TemporalPrior
{
    bool infer = false;
    double mh_sd = 1.0;
    T init;

    TemporalPrior() : infer(false), mh_sd(1.0) {}
    TemporalPrior(const Rcpp::List &opts)
    {
        infer = false;
        if (opts.containsElementNamed("infer"))
        {
            infer = Rcpp::as<bool>(opts["infer"]);
        }

        mh_sd = 1.0;
        if (opts.containsElementNamed("mh_sd"))
        {
            mh_sd = Rcpp::as<double>(opts["mh_sd"]);
        }

        if (opts.containsElementNamed("init"))
        {
            init = Rcpp::as<T>(opts["init"]);
        }
    }
};


class MCMC
{
private:
    unsigned int nsample = 1000;
    unsigned int nburnin = 1000;
    unsigned int nthin = 1;


    // Zero-inflation
    bool infer_zero_inflation = false;
    arma::mat z_stored; // nS x (nT + 1)
    arma::mat z_intercept_stored; // nS x nsample
    arma::mat z_slope_stored; // nS x nsample
    HMCOpts_2d zero_hmc_opts;
    HMCDiagnostics_2d zero_hmc_diagnostics;

    
    // psi1_spatial
    BYM2Prior coef_self_bym2_prior;
    arma::vec coef_self_b_car_mu_stored; // nsample x 1
    arma::vec coef_self_b_car_tau_stored; // nsample x 1
    arma::vec coef_self_b_car_phi_stored; // nsample x 1
    bool infer_psi1_spatial = false;
    HMCOpts_1d spatial_coef_self_hmc;
    HMCDiagnostics_1d spatial_coef_self_hmc_diagnostics;
    arma::vec psi1_spatial_init; // nS x 1
    arma::mat psi1_spatial_stored; // nS x nsample

    // Prior for temporal disturbances wt
    TemporalPrior<arma::vec> coef_self_temporal_prior;
    arma::mat psi2_temporal_stored; // (nT + 1) x nsample
    arma::vec psi2_temporal_accept; // nT + 1

    // Global parameters
    HMCOpts_1d global_hmc;
    HMCDiagnostics_1d global_hmc_diagnostics;

    Prior a_intercept_prior;
    arma::vec a_intercept_stored; // nsample x 1

    Prior coef_self_sigma2_prior;
    arma::vec coef_self_sigma2_stored; // nsample x 1, variance of temporal random effects

    Prior lag_par1_prior;
    arma::vec lag_par1_stored; // nsample x 1

    Prior lag_par2_prior;
    arma::vec lag_par2_stored; // nsample x 1

    // Local parameters
    HMCOpts_2d local_hmc;
    HMCDiagnostics_2d local_hmc_diagnostics;
    Prior rho_prior;
    arma::mat rho_stored; // nS x nsample


public:
    
    MCMC(
        const unsigned int &nsample_in = 1000,
        const unsigned int &nburnin_in = 1000,
        const unsigned int &nthin_in = 1
    )
    {
        nsample = nsample_in;
        nburnin = nburnin_in;
        nthin = nthin_in;
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

        infer_zero_inflation = false;
        if (opts.containsElementNamed("zero"))
        {
            Rcpp::List zero_opts = Rcpp::as<Rcpp::List>(opts["zero"]);

            if (zero_opts.containsElementNamed("infer"))
            {
                infer_zero_inflation = Rcpp::as<bool>(zero_opts["infer"]);
            }

            if (zero_opts.containsElementNamed("hmc"))
            {
                Rcpp::List zero_hmc_opts_in = Rcpp::as<Rcpp::List>(zero_opts["hmc"]);
                zero_hmc_opts = HMCOpts_2d(zero_hmc_opts_in);
                zero_hmc_opts.params_selected = {"intercept", "coef"};
            }
        } // end of zero inflation options

        if (opts.containsElementNamed("global_hmc"))
        {
            Rcpp::List global_hmc_opts = Rcpp::as<Rcpp::List>(opts["global_hmc"]);
            global_hmc = HMCOpts_1d(global_hmc_opts);
        } // end of global_hmc options

        if (opts.containsElementNamed("local_hmc"))
        {
            Rcpp::List local_hmc_opts = Rcpp::as<Rcpp::List>(opts["local_hmc"]);
            local_hmc = HMCOpts_2d(local_hmc_opts);
        } // end of local_hmc options

        if (opts.containsElementNamed("rho"))
        {
            Rcpp::List rho_opts = Rcpp::as<Rcpp::List>(opts["rho"]);
            rho_prior.init(rho_opts);
        }
        if (rho_prior.infer)
        {
            local_hmc.params_selected.push_back("rho");
        } // end of rho options

        if (opts.containsElementNamed("lag_par1"))
        {
            Rcpp::List lag_par1_opts = Rcpp::as<Rcpp::List>(opts["lag_par1"]);
            lag_par1_prior.init(lag_par1_opts);
        }
        if (lag_par1_prior.infer)
        {
            global_hmc.params_selected.push_back("lag_par1");
        } // end of lag_par1 options

        if (opts.containsElementNamed("lag_par2"))
        {
            Rcpp::List lag_par2_opts = Rcpp::as<Rcpp::List>(opts["lag_par2"]);
            lag_par2_prior.init(lag_par2_opts);
        }
        if (lag_par2_prior.infer)
        {
            global_hmc.params_selected.push_back("lag_par2");
        } // end of lag_par2 options

        if (opts.containsElementNamed("intercept_a"))
        {
            Rcpp::List a_opts = Rcpp::as<Rcpp::List>(opts["intercept_a"]);
            if (a_opts.containsElementNamed("intercept"))
            {
                Rcpp::List a_intercept_opts = Rcpp::as<Rcpp::List>(a_opts["intercept"]);
                a_intercept_prior.init(a_intercept_opts);
            }
            if (a_intercept_prior.infer)
            {
                global_hmc.params_selected.push_back("a_intercept");
            } // end of a_intercept options
        } // end of a options

        if (opts.containsElementNamed("coef_self"))
        {
            Rcpp::List coef_self_opts = Rcpp::as<Rcpp::List>(opts["coef_self"]);
            if (coef_self_opts.containsElementNamed("sigma2"))
            {
                Rcpp::List coef_self_sigma2_opts = Rcpp::as<Rcpp::List>(coef_self_opts["sigma2"]);
                coef_self_sigma2_prior.init(coef_self_sigma2_opts);
            }
            if (coef_self_sigma2_prior.infer)
            {
                global_hmc.params_selected.push_back("coef_self_sigma2");
            } // end of coef_self_sigma2 options

            if (coef_self_opts.containsElementNamed("temporal"))
            {
                Rcpp::List coef_self_temporal_opts = Rcpp::as<Rcpp::List>(coef_self_opts["temporal"]);
                coef_self_temporal_prior.infer = false;
                if (coef_self_temporal_opts.containsElementNamed("infer"))
                {
                    coef_self_temporal_prior.infer = Rcpp::as<bool>(coef_self_temporal_opts["infer"]);
                }

                coef_self_temporal_prior.mh_sd = 1.0;
                if (coef_self_temporal_opts.containsElementNamed("mh_sd"))
                {
                    coef_self_temporal_prior.mh_sd = Rcpp::as<double>(coef_self_temporal_opts["mh_sd"]);
                }

                if (coef_self_temporal_opts.containsElementNamed("init"))
                {
                    coef_self_temporal_prior.init = Rcpp::as<arma::vec>(coef_self_temporal_opts["init"]);
                }
            } // end of coef_self_temporal options

            if (coef_self_opts.containsElementNamed("spatial"))
            {
                Rcpp::List coef_self_spatial_opts = Rcpp::as<Rcpp::List>(coef_self_opts["spatial"]);
                if (coef_self_spatial_opts.containsElementNamed("infer"))
                {
                    infer_psi1_spatial = Rcpp::as<bool>(coef_self_spatial_opts["infer"]);
                }
                if (coef_self_spatial_opts.containsElementNamed("bym2"))
                {
                    Rcpp::List coef_self_bym2_opts = Rcpp::as<Rcpp::List>(coef_self_spatial_opts["bym2"]);
                    coef_self_bym2_prior = BYM2Prior(coef_self_bym2_opts);
                } // BYM2 prior settings
                if (coef_self_spatial_opts.containsElementNamed("init"))
                {
                    psi1_spatial_init = Rcpp::as<arma::vec>(coef_self_spatial_opts["init"]);
                } // Initial spatial effects
                if (coef_self_spatial_opts.containsElementNamed("hmc"))
                {
                    Rcpp::List coef_self_hmc_opts = Rcpp::as<Rcpp::List>(coef_self_spatial_opts["hmc"]);
                    spatial_coef_self_hmc = HMCOpts_1d(coef_self_hmc_opts);
                } // HMC settings
            } // end of coef_self_spatial_car options
        } // end of coef_self options

        return;
    } // end of constructor from Rcpp::List

    static Rcpp::List get_default_settings()
    {
        Rcpp::List rho_opts = Rcpp::List::create(
            Rcpp::Named("infer") = false,
            Rcpp::Named("prior_name") = "invgamma",
            Rcpp::Named("prior_param") = Rcpp::NumericVector::create(1.0, 1.0)
        );

        Rcpp::List lag_par1_opts = Rcpp::List::create(
            Rcpp::Named("infer") = false,
            Rcpp::Named("prior_name") = "gaussian",
            Rcpp::Named("prior_param") = Rcpp::NumericVector::create(0.0, 10.0)
        );

        Rcpp::List lag_par2_opts = Rcpp::List::create(
            Rcpp::Named("infer") = false,
            Rcpp::Named("prior_name") = "invgamma",
            Rcpp::Named("prior_param") = Rcpp::NumericVector::create(1.0, 1.0)
        );

        Rcpp::List a_opts;
        {
            Rcpp::List a_intercept_opts = Rcpp::List::create(
                Rcpp::Named("infer") = false,
                Rcpp::Named("prior_name") = "gaussian",
                Rcpp::Named("prior_param") = Rcpp::NumericVector::create(0.0, 10.0)
            );

            a_opts = Rcpp::List::create(
                Rcpp::Named("intercept") = a_intercept_opts
            );
        } // end of a options

        Rcpp::List coef_self_opts;
        {
            Rcpp::List coef_self_sigma2_opts = Rcpp::List::create(
                Rcpp::Named("infer") = false,
                Rcpp::Named("prior_name") = "invgamma",
                Rcpp::Named("prior_param") = Rcpp::NumericVector::create(1.0, 1.0));

            Rcpp::List coef_self_temporal_opts = Rcpp::List::create(
                Rcpp::Named("infer") = false,
                Rcpp::Named("mh_sd") = 1.0,
                Rcpp::Named("init") = arma::zeros<arma::vec>(1)
            );

            Rcpp::List coef_self_spatial_car_opts = Rcpp::List::create(
                Rcpp::Named("infer") = false,
                Rcpp::Named("init") = arma::vec({0.5}), // initial value of spatial random effects
                Rcpp::Named("hmc") = Rcpp::List::create(
                    Rcpp::Named("nleapfrog") = 20,
                    Rcpp::Named("leapfrog_step_size") = 0.1,
                    Rcpp::Named("dual_averaging") = false,
                    Rcpp::Named("diagnostics") = true,
                    Rcpp::Named("verbose") = false,
                    Rcpp::Named("T_target") = 2.0),
                Rcpp::Named("bym2") = Rcpp::List::create(
                    Rcpp::Named("infer") = false,
                    Rcpp::Named("mh_sd") = 0.1,
                    Rcpp::Named("tau_b") = Rcpp::NumericVector::create(1.0, 1.0),
                    Rcpp::Named("logit_phi") = Rcpp::NumericVector::create(0.0, 1.0)
                )
            );

            coef_self_opts = Rcpp::List::create(
                Rcpp::Named("sigma2") = coef_self_sigma2_opts,
                Rcpp::Named("temporal") = coef_self_temporal_opts,
                Rcpp::Named("spatial") = coef_self_spatial_car_opts
            );
        } // end of coef_self options


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

        Rcpp::List mcmc_opts = Rcpp::List::create(
            Rcpp::Named("nsample") = 1000,
            Rcpp::Named("nburnin") = 1000,
            Rcpp::Named("nthin") = 1,
            Rcpp::Named("zero") = Rcpp::List::create(
                Rcpp::Named("infer") = false,
                Rcpp::Named("hmc") = local_hmc_opts
            ),
            Rcpp::Named("intercept_a") = a_opts,
            Rcpp::Named("coef_self") = coef_self_opts,
            Rcpp::Named("rho") = rho_opts,
            Rcpp::Named("lag_par1") = lag_par1_opts,
            Rcpp::Named("lag_par2") = lag_par2_opts,
            Rcpp::Named("global_hmc") = global_hmc_opts,
            Rcpp::Named("local_hmc") = local_hmc_opts
        );

        return mcmc_opts;
    } // end of get_default_settings()

    /**
     * @brief Block MH sampler for the BYM2 parameters (tau_b, phi) with no replicate.
     * 
     * @param prior 
     * @param spatial 
     * @param wt 
     * @param iter 
     * @param nburnin 
     */
    void update_car_conditional(
        BYM2Prior &prior,
        BYM2 &spatial, 
        const arma::vec &wt, // nS x 1
        const unsigned int &iter, 
        const unsigned int &nburnin
    )
    {
        double acc = spatial.update_phi_logit_conditional(wt, prior);
        prior.accept_count += acc;
        spatial.update_tau_conditional(wt, prior.shape_tau, prior.rate_tau);
        prior.adapt_phi_proposal_robbins_monro(iter, nburnin);
        return;        
    } // end of update_car_conditional()


    void update_wt(
        arma::vec &wt2_temporal, // (nT + 1) x 1
        arma::vec &wt_accept, // (nT + 1) x 1
        const Model &model, 
        const arma::mat &Y, // nS x (nT + 1),
        const arma::vec &psi1_spatial, // nS x 1
        const double &mh_sd = 1.0
    )
    {
        const unsigned int nT = Y.n_cols - 1;
        const double sigma2 = std::max(model.coef_self_b.sigma2, EPS);
        const double sigma = std::sqrt(sigma2);

        for (unsigned int t = 1; t < Y.n_cols; t++)
        {
            const double wt_old = wt2_temporal.at(t);
            arma::vec psi = arma::cumsum(wt2_temporal); // (nT + 1) x 1

            double mh_prec = 1.0 / std::max(model.coef_self_b.sigma2, EPS);
            double logp_old = R::dnorm4(wt_old, 0.0, sigma, true);
            for (unsigned int s = 0; s < Y.n_rows; s++)
            {
                arma::vec psi_s = psi + psi1_spatial.at(s);
                arma::vec hpsi_s = GainFunc::psi2hpsi<arma::vec>(psi_s, model.fgain);
                arma::vec dhpsi_s = GainFunc::psi2dhpsi<arma::vec>(psi_s, model.fgain);

                arma::vec lam_s = model.compute_intensity_iterative(s, Y, psi1_spatial, psi);
                arma::vec V_s = lam_s % (lam_s + model.rho.at(s)) / model.rho.at(s);
                V_s = arma::clamp(V_s, EPS, std::numeric_limits<double>::infinity());

                ApproxDisturbance approx_dlm(psi, hpsi_s, dhpsi_s, model.fgain);
                approx_dlm.set_Fphi(model.dlag, model.nP);
                arma::vec Fnt = approx_dlm.Fn.col(t - 1); // nT x 1
                mh_prec += arma::accu((Fnt % Fnt) / V_s.tail(nT));
                
                for (unsigned int i = t; i < Y.n_cols; i++)
                {
                    logp_old += ObsDist::loglike(
                        Y.at(s, i), model.dobs, lam_s.at(i), model.rho.at(s), true
                    );
                }
            } // end of location s loop

            if (!std::isfinite(mh_prec) || mh_prec <= 0.0)
            {
                mh_prec = 1.0 / std::max(sigma2, EPS); // conservative fallback
            }
            double mh_step = std::sqrt(1.0 / std::max(mh_prec, EPS)) * mh_sd;

            // Propose
            double wt_new = rnorm(wt_old, mh_step);
            if (!std::isfinite(wt_new))
            {
                // reject immediately
                continue;
            }

            wt2_temporal.at(t) = wt_new;
            psi = arma::cumsum(wt2_temporal); // (nT + 1) x 1
            double logp_new = R::dnorm4(wt_new, 0.0, sigma, true);
            for (unsigned int s = 0; s < Y.n_rows; s++)
            {
                arma::vec lam_s = model.compute_intensity_iterative(s, Y, psi1_spatial, psi);
                for (unsigned int i = t; i < Y.n_cols; i++)
                {
                    logp_new += ObsDist::loglike(
                        Y.at(s, i), model.dobs, lam_s.at(i), model.rho.at(s), true
                    );
                }
            } // end of location s loop

            double diff = logp_new - logp_old;
            if (!std::isfinite(diff))
            {
                // reject non-finite ratio
                wt2_temporal.at(t) = wt_old;
                continue;
            }

            double logratio = std::min(0.0, diff);
            if (std::log(runif()) < logratio)
            {
                // accept
                wt_accept.at(t) += 1.0;
            }
            else
            {
                // reject and revert
                wt2_temporal.at(t) = wt_old;
            }
        } // end of time t loop

        return;
    } // end of update_wt()


    double update_coef_self_spatial(
        arma::vec &psi1_spatial, // nS x 1
        double &energy_diff,
        double &grad_norm_out,
        Model &model,
        const arma::mat &Y, // nS x (nT + 1),
        const arma::vec &wt2_temporal // (nT + 1) x 1
    )
    {
        const arma::vec psi2_temporal = arma::cumsum(wt2_temporal); // (nT + 1) x 1

        // Diagonal preconditioner: M^{-1} ≈ (tau2 * diag(Q))^{-1}
        // Compute the mass matrix (diagonal)
        arma::mat Rphi = model.coef_self_b.spatial.compute_Rphi(model.coef_self_b.spatial.phi);
        const double mass_floor = 1e-6;
        arma::vec inv_mass = arma::clamp(model.coef_self_b.spatial.tau_b * Rphi.diag(), mass_floor, arma::datum::inf);
        arma::vec mass_diag = 1.0 / inv_mass;
        arma::vec sqrt_mass = arma::sqrt(mass_diag);

        arma::mat dll_deta = model.dloglik_deta(Y, psi1_spatial, psi2_temporal); // nS x (nT + 1)

        arma::mat lam_current(Y.n_rows, Y.n_cols, arma::fill::zeros);
        model.compute_intensity_iterative(
            lam_current, 1,
            Y, psi1_spatial, psi2_temporal
        ); // nS x (nT + 1)
        double logp_current = model.coef_self_b.spatial.log_likelihood(psi1_spatial, false);
        for (unsigned int s = 0; s < model.nS; s++)
        {
            for (unsigned int t = 1; t <= Y.n_cols - 1; t++)
            {
                logp_current += ObsDist::loglike(
                    Y.at(s, t), model.dobs, lam_current.at(s, t), model.rho.at(s), true
                );
            }
        }

        double energy_current = -logp_current;

        arma::vec spatial_current = psi1_spatial;
        arma::vec q = spatial_current;

        // sample an initial momentum
        arma::vec p = arma::randn(q.n_elem) % sqrt_mass;
        double kinetic_current = 0.5 * arma::dot(p % inv_mass, p);

        arma::vec grad = model.dloglik_dspatial_coef_self(Y, dll_deta, psi1_spatial, psi2_temporal);
        grad *= -1.0; // Convert to gradient of potential energy
        grad_norm_out = arma::norm(grad, 2);

        // Make a half step for momentum at the beginning
        p -= 0.5 * spatial_coef_self_hmc.leapfrog_step_size * grad;
        for (unsigned int i = 0; i < spatial_coef_self_hmc.nleapfrog; i++)
        {
            // Make a full step for the position
            q += spatial_coef_self_hmc.leapfrog_step_size * (inv_mass % p);
            psi1_spatial = q;

            // Compute the new gradient
            dll_deta = model.dloglik_deta(Y, psi1_spatial, psi2_temporal); // nS x (nT + 1)
            grad = model.dloglik_dspatial_coef_self(Y, dll_deta, psi1_spatial, psi2_temporal);
            grad *= -1.0; // Convert to gradient of potential energy

            // Make a full step for the momentum, except at the end of trajectory
            if (i != spatial_coef_self_hmc.nleapfrog - 1)
            {
                p -= spatial_coef_self_hmc.leapfrog_step_size * grad;
            }
        } // end of leapfrog steps

        p -= 0.5 * spatial_coef_self_hmc.leapfrog_step_size * grad; // Make a half step for momentum at the end
        p *= -1; // Negate momentum to make the proposal symmetric

        model.compute_intensity_iterative(
            lam_current, 1,
            Y, psi1_spatial, psi2_temporal
        );
        double logp_proposed = model.coef_self_b.spatial.log_likelihood(psi1_spatial, false);
        for (unsigned int s = 0; s < model.nS; s++)
        {
            for (unsigned int t = 1; t <= Y.n_cols - 1; t++)
            {
                logp_proposed += ObsDist::loglike(
                    Y.at(s, t), model.dobs, lam_current.at(s, t), model.rho.at(s), true
                );
            }
        }

        double energy_proposed = -logp_proposed;
        double kinetic_proposed = 0.5 * arma::dot(p % inv_mass, p);

        double H_proposed = energy_proposed + kinetic_proposed;
        double H_current = energy_current + kinetic_current;
        energy_diff = H_proposed - H_current;

        if (!std::isfinite(H_current) || !std::isfinite(H_proposed) || std::abs(energy_diff) > 100.0)
        {
            // Reject if either log probability is not finite or HMC diverged
            energy_diff = 100.0;
            psi1_spatial = spatial_current;;
            return 0.0;
        }
        else
        {
            double log_accept_ratio = H_current - H_proposed;
            if (std::log(runif()) >= log_accept_ratio)
            {
                // Reject: revert to current parameters
                psi1_spatial = spatial_current;
            }

            double accept_prob = std::min(1.0, std::exp(log_accept_ratio));
            return accept_prob;
        }
    } // end of update_coef_self_spatial()


    double compute_log_joint_global(
        const Model &model, 
        const arma::mat &Y, // nS x (nT + 1),
        const arma::vec &psi1_spatial, // nS x 1
        const arma::vec &wt2_temporal // (nT + 1) x 1
    )
    {
        double logp = 0.0;
        arma::vec psi2_temporal = arma::cumsum(wt2_temporal);
        for (unsigned int s = 0; s < model.nS; s++)
        {
            arma::vec lam_s = model.compute_intensity_iterative(s, Y, psi1_spatial, psi2_temporal);
            for (unsigned int t = 1; t <= Y.n_cols - 1; t++)
            {
                // Compute loglikelihood of y[s, t]
                if (!model.zero[s].inflated || model.zero[s].z.at(t) > EPS)
                {
                    logp += ObsDist::loglike(
                        Y.at(s, t), model.dobs, lam_s.at(t), model.rho.at(s), true);
                }
            } // end of time t loop
        } // end of location s loop

        if (model.coef_self_b.has_temporal)
        {
            double s2 = std::max(model.coef_self_b.sigma2, EPS);
            double sd = std::sqrt(s2);
            double logc = -0.5*std::log(2.0*M_PI) - std::log(sd);
            for (unsigned int t=1; t<Y.n_cols; ++t)
                logp += logc - 0.5 * (wt2_temporal.at(t)*wt2_temporal.at(t))/s2;
        }

        if (lag_par1_prior.infer)
        {
            // normal prior on the first lag parameter (log scale)
            logp += Prior::dprior(model.dlag.par1, lag_par1_prior, true, false);
        }

        if (lag_par2_prior.infer)
        {
            // inverse-gamma prior on the second lag parameter (log scale)
            logp += Prior::dprior(model.dlag.par2, lag_par2_prior, true, true);
        }

        if (a_intercept_prior.infer)
        {
            // normal prior on the intercept (log scale) of the baseline intensity
            logp += Prior::dprior(model.intercept_a.intercept, a_intercept_prior, true, false);
        }

        if (coef_self_sigma2_prior.infer)
        {
            // inverse-gamma prior on the variance of self-exciting effects (log scale)
            logp += Prior::dprior(model.coef_self_b.sigma2, coef_self_sigma2_prior, true, true);
        }

        return logp;
    }

    double compute_log_joint_local(
        const unsigned int &s,
        const Model &model, 
        const arma::vec &y, 
        const arma::vec &lambda
    )
    {
        double logp = 0.0;
        for (unsigned int t = 1; t <= y.n_elem - 1; t++)
        {
            // Compute loglikelihood of y[s, t]
            if (!model.zero[s].inflated || model.zero[s].z.at(t) > EPS)
            {
                logp += ObsDist::loglike(
                    y.at(t), model.dobs, lambda.at(t), model.rho.at(s), true);
            }
        } // end of time t loop

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
        const arma::vec &psi1_spatial, // nS x 1
        const arma::vec &wt2_temporal, // (nT + 1) x 1
        const double &leapfrog_step_size,
        const unsigned int &n_leapfrog = 10
    )
    {
        // Potential energy: negative log joint probability.
        double logp_current = compute_log_joint_global(
            model, Y, psi1_spatial, wt2_temporal
        );
        double energy_current = -logp_current;

        // Get current global parameters in unconstrained space
        arma::vec global_params_current = model.get_global_params_unconstrained(global_params_selected);
        arma::vec q = global_params_current;

        // sample an initial momentum
        arma::vec p = arma::randn(q.n_elem);
        double kinetic_current = 0.5 * arma::dot(p, p);

        arma::vec grad = model.dloglik_dglobal_unconstrained(
            global_params_selected, Y,
            psi1_spatial, wt2_temporal,
            lag_par1_prior, lag_par2_prior,
            a_intercept_prior,
            coef_self_sigma2_prior
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
                global_params_selected, Y,
                psi1_spatial, wt2_temporal,
                lag_par1_prior, lag_par2_prior,
                a_intercept_prior,
                coef_self_sigma2_prior
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

        double logp_proposed = compute_log_joint_global(
            model, Y, psi1_spatial, wt2_temporal
        );
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
        const arma::vec &psi1_spatial, // nS x 1
        const arma::vec &wt2_temporal, // (nT + 1) x 1
        const double &leapfrog_step_size,
        const unsigned int &n_leapfrog = 10
    )
    {
        // Precompute hPsi (depends only on wt)
        const arma::vec psi2_temporal = arma::cumsum(wt2_temporal); // (nT + 1) x 1

        arma::vec lambda = model.compute_intensity_iterative(
            s, Y, psi1_spatial, psi2_temporal
        ); // (nT + 1) x 1
        const arma::vec ys = Y.row(s).t(); // (nT + 1) x 1

        // Potential energy: negative log joint probability.
        double logp_current = compute_log_joint_local(s, model, ys, lambda);
        double energy_current = -logp_current;

        // Get current global parameters in unconstrained space
        arma::vec local_params_current = model.get_local_params_unconstrained(s, local_params_selected);
        arma::vec q = local_params_current;

        // sample an initial momentum
        arma::vec p = arma::randn(q.n_elem);
        double kinetic_current = 0.5 * arma::dot(p, p);

        arma::vec grad = model.dloglik_dlocal_unconstrained(
            s, local_params_selected, 
            ys, lambda, rho_prior
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
                ys, lambda, rho_prior
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

        double logp_proposed = compute_log_joint_local(s, model, ys, lambda);
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


    void check_grad_global(
        Model &model, 
        const arma::mat &Y, 
        const arma::vec &psi1_spatial, // nS x 1
        const arma::vec &wt2_temporal, // (nT + 1) x 1
        const std::vector<std::string> &names
    )
    {
        arma::vec q = model.get_global_params_unconstrained(names);
        arma::vec g = model.dloglik_dglobal_unconstrained(
            names, Y, 
            psi1_spatial, wt2_temporal,
            lag_par1_prior, lag_par2_prior,
            a_intercept_prior,
            coef_self_sigma2_prior
        );
        double eps_fd = 1e-5;
        for (unsigned int i = 0; i < q.n_elem; i++)
        {
            arma::vec q1 = q, q2 = q;
            q1[i] += eps_fd;
            q2[i] -= eps_fd;
            model.update_global_params_unconstrained(names, q1);
            double lp1 = compute_log_joint_global(
                model, Y, psi1_spatial, wt2_temporal
            );
            model.update_global_params_unconstrained(names, q2);
            double lp2 = compute_log_joint_global(
                model, Y, psi1_spatial, wt2_temporal
            );
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
        const std::vector<std::string> &names)
    {
        arma::vec q = model.get_local_params_unconstrained(s, names);
        arma::vec g = model.dloglik_dlocal_unconstrained(s, names, y, lambda, rho_prior);
        double eps_fd = 1e-5;
        for (unsigned int i = 0; i < q.n_elem; i++)
        {
            arma::vec q1 = q, q2 = q;
            q1[i] += eps_fd;
            q2[i] -= eps_fd;
            model.update_local_params_unconstrained(s, names, q1);
            double lp1 = compute_log_joint_local(s, model, y, lambda);
            model.update_local_params_unconstrained(s, names, q2);
            double lp2 = compute_log_joint_local(s, model, y, lambda);
            double fd = (lp1 - lp2) / (2 * eps_fd);
            Rcpp::Rcout << "Location " << s << " param " << i << " analytic=" << g[i] << " fd=" << fd
                        << " rel.err=" << std::fabs(g[i] - fd) / std::max(1.0, std::fabs(fd)) << "\n";
            model.update_local_params_unconstrained(s, names, q); // restore
        }
    }


    void infer(Model &model, const arma::mat &Y)
    {
        const unsigned int nT = Y.n_cols - 1;
        const unsigned int niter = nburnin + nsample * nthin;
        const double target_accept = 0.75;

        // Clamp helpers
        auto clamp_eps = [](double x)
        { return std::max(1e-3, std::min(0.1, x)); };
        unsigned min_leaps = 3, max_leaps = 128;

        if (!model.zero[0].inflated)
        {
            infer_zero_inflation = false;
            for (unsigned int s = 0; s < model.nS; s++)
            {
                model.zero[s].z = arma::vec(nT + 1, arma::fill::ones);
                model.zero[s].prob = arma::vec(nT + 1, arma::fill::ones);
            }
        }
        else
        {
            for (unsigned int s = 0; s < model.nS; s++)
            {
                model.zero[s].init_Z(Y.row(s).t());
            }
        }

        DualAveraging_2d zero_da_adapter(model.nS, zero_hmc_opts.leapfrog_step_size_init);
        if (infer_zero_inflation)
        {
            zero_hmc_opts.set_size(model.nS);
            zero_hmc_diagnostics = HMCDiagnostics_2d(model.nS, niter, nburnin, zero_hmc_opts.dual_averaging);
            z_stored = arma::mat(model.nS, nT + 1, arma::fill::zeros);
            z_intercept_stored = arma::zeros<arma::mat>(model.nS, nsample);
            z_slope_stored = arma::zeros<arma::mat>(model.nS, nsample);
        }

        // Dual averaging state (Hoffman & Gelman 2014)
        DualAveraging_1d global_hmc_da(global_hmc, target_accept);
        if (!global_hmc.params_selected.empty() && global_hmc.diagnostics)
        {
            global_hmc_diagnostics = HMCDiagnostics_1d(niter, nburnin, global_hmc.dual_averaging);
        }

        if (!model.intercept_a.has_intercept)
        {
            a_intercept_prior.infer = false;
            auto it = std::find(
                global_hmc.params_selected.begin(), 
                global_hmc.params_selected.end(), 
                "a_intercept"
            );
            if (it != global_hmc.params_selected.end())
            {
                global_hmc.params_selected.erase(it);
            }
        }
        if (a_intercept_prior.infer)
        {
            a_intercept_stored = arma::vec(nsample, arma::fill::zeros);
        }

        if (lag_par1_prior.infer)
        {
            lag_par1_stored = arma::vec(nsample, arma::fill::zeros);
        }
        if (lag_par2_prior.infer)
        {
            lag_par2_stored = arma::vec(nsample, arma::fill::zeros);
        }


        if (!model.coef_self_b.has_temporal)
        {
            coef_self_temporal_prior.infer = false;
            coef_self_sigma2_prior.infer = false;
            auto it = std::find(
                global_hmc.params_selected.begin(), 
                global_hmc.params_selected.end(), 
                "coef_self_sigma2"
            );
            if (it != global_hmc.params_selected.end())
            {
                global_hmc.params_selected.erase(it);
            }
        }
        if (coef_self_sigma2_prior.infer)
        {
            coef_self_sigma2_stored = arma::vec(nsample, arma::fill::zeros);
        }
        arma::vec wt2_temporal(nT + 1, arma::fill::zeros);
        psi2_temporal_accept = arma::vec(nT + 1, arma::fill::zeros);
        if (!coef_self_temporal_prior.init.is_empty())
        {
            wt2_temporal = coef_self_temporal_prior.init;
            wt2_temporal.at(0) = 0.0;
        }
        if (coef_self_temporal_prior.infer)
        {
            psi2_temporal_stored = arma::mat(nT + 1, nsample, arma::fill::zeros);
        }


        DualAveraging_1d spatial_coef_self_hmc_da(spatial_coef_self_hmc, target_accept);
        if (!model.coef_self_b.has_spatial)
        {
            infer_psi1_spatial = false;
        }
        if (infer_psi1_spatial)
        {
            if (spatial_coef_self_hmc.diagnostics)
            {
                spatial_coef_self_hmc_diagnostics = HMCDiagnostics_1d(
                    niter, nburnin, spatial_coef_self_hmc.dual_averaging
                );
            }
            psi1_spatial_stored = arma::mat(model.nS, nsample, arma::fill::zeros);
        }
        arma::vec psi1_spatial(model.nS, arma::fill::zeros);
        if (!psi1_spatial_init.is_empty() && psi1_spatial_init.n_elem == model.nS)
        {
            psi1_spatial = psi1_spatial_init;
        }
        if (coef_self_bym2_prior.infer)
        {
            coef_self_b_car_mu_stored = arma::zeros<arma::vec>(nsample);
            coef_self_b_car_tau_stored = arma::zeros<arma::vec>(nsample);
            coef_self_b_car_phi_stored = arma::zeros<arma::vec>(nsample);
        }

        DualAveraging_2d local_hmc_da(
            model.nS, 
            local_hmc.leapfrog_step_size_init, 
            target_accept
        );
        if (!local_hmc.params_selected.empty() && local_hmc.diagnostics)
        {
            local_hmc.set_size(model.nS);
            local_hmc_diagnostics = HMCDiagnostics_2d(model.nS, niter, nburnin, local_hmc.dual_averaging);
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
            if (infer_zero_inflation)
            {
                for (unsigned int s = 0; s < model.nS; s++)
                {
                    arma::vec ys = Y.row(s).t();
                    arma::vec lambda = model.compute_intensity_iterative(
                        s, Y, psi1_spatial,
                        arma::cumsum(wt2_temporal)
                    );
                    model.zero[s].update_zt(ys, lambda, model.dobs, model.rho.at(s));

                    double energy_diff, grad_norm;
                    double accept_prob = model.zero[s].update_params(
                        energy_diff, grad_norm, 0.0, 10.0, 
                        zero_hmc_opts.leapfrog_step_size.at(s), 
                        zero_hmc_opts.nleapfrog.at(s),
                        zero_hmc_opts.mass_diag_est.col(s)
                    );
                    zero_hmc_diagnostics.accept_count.at(s) += accept_prob;

                    if (zero_hmc_opts.dual_averaging)
                    {
                        if (iter < nburnin)
                        {
                            zero_hmc_opts.leapfrog_step_size.at(s) = zero_da_adapter.update_step_size(s,  accept_prob);
                        }
                        else if (iter == nburnin)
                        {
                            zero_da_adapter.finalize_leapfrog_step(
                                zero_hmc_opts.leapfrog_step_size.at(s),
                                zero_hmc_opts.nleapfrog.at(s),
                                s, zero_hmc_opts.T_target
                            );
                        }
                    }
                    
                    if (zero_hmc_opts.diagnostics)
                    {
                        zero_hmc_diagnostics.energy_diff(s, iter) = energy_diff;
                        zero_hmc_diagnostics.grad_norm(s, iter) = grad_norm;

                        if (zero_hmc_opts.dual_averaging && iter <= nburnin)
                        {
                            zero_hmc_diagnostics.nleapfrog_stored(s, iter) = zero_hmc_opts.nleapfrog.at(s);
                            zero_hmc_diagnostics.leapfrog_step_size_stored(s, iter) = zero_hmc_opts.leapfrog_step_size.at(s);
                        }
                    }
                }
            } // end of zero-inflation update


            if (coef_self_temporal_prior.infer)
            {
                update_wt(
                    wt2_temporal, psi2_temporal_accept,
                    model, Y, psi1_spatial, 
                    coef_self_temporal_prior.mh_sd
                );
            }
            if (coef_self_bym2_prior.infer)
            {
                update_car_conditional(
                    coef_self_bym2_prior,
                    model.coef_self_b.spatial,
                    psi1_spatial, iter, nburnin
                );
            }
            if (infer_psi1_spatial)
            {
                double energy_diff, grad_norm;
                double acc_prob = update_coef_self_spatial(
                    psi1_spatial, energy_diff, grad_norm, 
                    model, Y, wt2_temporal
                );
                spatial_coef_self_hmc_diagnostics.accept_count += acc_prob;

                if (spatial_coef_self_hmc.diagnostics)
                {
                    spatial_coef_self_hmc_diagnostics.energy_diff.at(iter) = energy_diff;
                    spatial_coef_self_hmc_diagnostics.grad_norm.at(iter) = grad_norm;
                }

                if (spatial_coef_self_hmc.dual_averaging)
                {
                    if (iter < nburnin)
                    {
                        spatial_coef_self_hmc.leapfrog_step_size = spatial_coef_self_hmc_da.update_step_size(
                            acc_prob
                        );
                    }
                    else if (iter == nburnin)
                    {
                        spatial_coef_self_hmc_da.finalize_leapfrog_step(
                            spatial_coef_self_hmc.leapfrog_step_size,
                            spatial_coef_self_hmc.nleapfrog,
                            spatial_coef_self_hmc.T_target
                        );
                    }

                    if (spatial_coef_self_hmc.diagnostics && iter <= nburnin)
                    {
                        spatial_coef_self_hmc_diagnostics.nleapfrog_stored.at(iter) = spatial_coef_self_hmc.nleapfrog;
                        spatial_coef_self_hmc_diagnostics.leapfrog_step_size_stored.at(iter) = spatial_coef_self_hmc.leapfrog_step_size;
                    }
                } // end of spatial coef self dual averaging
            }

            if (!global_hmc.params_selected.empty())
            {
                if (global_hmc.verbose && iter % 100 == 0 && iter < nburnin)
                {
                    check_grad_global(
                        model, Y, 
                        psi1_spatial, wt2_temporal,
                        global_hmc.params_selected
                    );
                }

                double energy_diff, grad_norm;
                double global_hmc_accept_prob = update_global_params(
                    model, energy_diff, grad_norm, 
                    global_hmc.params_selected, 
                    Y, psi1_spatial, wt2_temporal, 
                    global_hmc.leapfrog_step_size, 
                    global_hmc.nleapfrog
                );
                global_hmc_diagnostics.accept_count += global_hmc_accept_prob;

                if (global_hmc.diagnostics)
                {
                    global_hmc_diagnostics.energy_diff.at(iter) = energy_diff;
                    global_hmc_diagnostics.grad_norm.at(iter) = grad_norm;
                }

                if (global_hmc.dual_averaging)
                {
                    if (iter < nburnin)
                    {
                        global_hmc.leapfrog_step_size = global_hmc_da.update_step_size(
                            global_hmc_accept_prob
                        );
                    }
                    else if (iter == nburnin)
                    {
                        global_hmc_da.finalize_leapfrog_step(
                            global_hmc.leapfrog_step_size,
                            global_hmc.nleapfrog,
                            global_hmc.T_target
                        );
                    }

                    if (global_hmc.diagnostics && iter <= nburnin)
                    {
                        global_hmc_diagnostics.nleapfrog_stored.at(iter) = global_hmc.nleapfrog;
                        global_hmc_diagnostics.leapfrog_step_size_stored.at(iter) = global_hmc.leapfrog_step_size;
                    }
                } // end of global dual averaging
            } // end of global params update

            if (!local_hmc.params_selected.empty())
            {
                for (unsigned int s = 0; s < model.nS; s++)
                {
                    if (local_hmc.verbose && iter % 100 == 0 && iter < nburnin)
                    {
                        arma::vec ys = Y.row(s).t();
                        arma::vec lambda = model.compute_intensity_iterative(
                            s, Y, psi1_spatial,
                            arma::cumsum(wt2_temporal)
                        );
                        check_grad_local(model, s, ys, lambda, local_hmc.params_selected);
                    }

                    double energy_diff, grad_norm;
                    double local_hmc_accept_prob = update_local_params(
                        model, energy_diff, grad_norm, 
                        local_hmc.params_selected, s, Y, 
                        psi1_spatial, wt2_temporal, 
                        local_hmc.leapfrog_step_size.at(s), 
                        local_hmc.nleapfrog.at(s)
                    );
                    local_hmc_diagnostics.accept_count.at(s) += local_hmc_accept_prob;

                    if (local_hmc.diagnostics)
                    {
                        local_hmc_diagnostics.energy_diff.at(s, iter) = energy_diff;
                        local_hmc_diagnostics.grad_norm.at(s, iter) = grad_norm;
                    }

                    if (local_hmc.dual_averaging)
                    {
                        if (iter < nburnin)
                        {
                            local_hmc.leapfrog_step_size.at(s) = local_hmc_da.update_step_size(
                                s, local_hmc_accept_prob
                            );
                        }
                        else if (iter == nburnin)
                        {
                            local_hmc_da.finalize_leapfrog_step(
                                local_hmc.leapfrog_step_size.at(s),
                                local_hmc.nleapfrog.at(s),
                                s,
                                local_hmc.T_target
                            );
                        }

                        if (local_hmc.diagnostics && iter <= nburnin)
                        {
                            local_hmc_diagnostics.nleapfrog_stored.at(s, iter) = local_hmc.nleapfrog.at(s);
                            local_hmc_diagnostics.leapfrog_step_size_stored.at(s, iter) = local_hmc.leapfrog_step_size.at(s);
                        }
                    } // end of local dual averaging
                } // end of s loop
            } // end of local params update


            // Store samples after burn-in and thinning
            if (iter >= nburnin && ((iter - nburnin) % nthin == 0))
            {
                const unsigned int sample_idx = (iter - nburnin) / nthin;

                if (infer_zero_inflation)
                {
                    for (unsigned int s = 0; s < model.nS; s++)
                    {
                        z_intercept_stored(s, sample_idx) = model.zero[s].intercept;
                        z_slope_stored(s, sample_idx) = model.zero[s].coef;
                        z_stored.row(s) += model.zero[s].z.t();
                    }
                }

                if (a_intercept_prior.infer)
                {
                    a_intercept_stored.at(sample_idx) = model.intercept_a.intercept;
                }

                if (coef_self_sigma2_prior.infer)
                {
                    coef_self_sigma2_stored.at(sample_idx) = model.coef_self_b.sigma2;
                }
                if (coef_self_temporal_prior.infer)
                {
                    psi2_temporal_stored.col(sample_idx) = arma::cumsum(wt2_temporal);
                }

                if (coef_self_bym2_prior.infer)
                {
                    coef_self_b_car_mu_stored.at(sample_idx) = model.coef_self_b.spatial.mu;
                    coef_self_b_car_tau_stored.at(sample_idx) = model.coef_self_b.spatial.tau_b;
                    coef_self_b_car_phi_stored.at(sample_idx) = model.coef_self_b.spatial.phi;
                }
                if (infer_psi1_spatial)
                {
                    psi1_spatial_stored.col(sample_idx) = psi1_spatial;
                }


                if (rho_prior.infer)
                {
                    for (unsigned int s = 0; s < model.nS; s++)
                    {
                        rho_stored.at(s, sample_idx) = model.rho.at(s);
                    }
                }

                if (lag_par1_prior.infer)
                {
                    lag_par1_stored.at(sample_idx) = model.dlag.par1;
                }

                if (lag_par2_prior.infer)
                {
                    lag_par2_stored.at(sample_idx) = model.dlag.par2;
                }

            } // end of store samples

            p.increment(); 
        } // end of iter loop
    } // end of infer()

    Rcpp::List get_output() const
    {
        Rcpp::List output;

        if (infer_zero_inflation)
        {
            output["zero"] = Rcpp::List::create(
                Rcpp::Named("z") = Rcpp::wrap(z_stored / (double)nsample),
                Rcpp::Named("intercept") = Rcpp::wrap(z_intercept_stored),
                Rcpp::Named("coef") = Rcpp::wrap(z_slope_stored),
                Rcpp::Named("hmc_accept_rate") = zero_hmc_diagnostics.accept_count / (nburnin + nsample * nthin),
                Rcpp::Named("hmc_settings") = Rcpp::List::create(
                    Rcpp::Named("nleapfrog") = zero_hmc_opts.nleapfrog,
                    Rcpp::Named("leapfrog_step_size") = zero_hmc_opts.leapfrog_step_size
                ),
                Rcpp::Named("hmc_diagnostics") = Rcpp::List::create(
                    Rcpp::Named("energy_diff") = zero_hmc_diagnostics.energy_diff,
                    Rcpp::Named("grad_norm") = zero_hmc_diagnostics.grad_norm,
                    Rcpp::Named("n_leapfrog") = zero_hmc_diagnostics.nleapfrog_stored,
                    Rcpp::Named("step_size") = zero_hmc_diagnostics.leapfrog_step_size_stored)
            );
        }

        if (rho_prior.infer)
        {
            output["rho"] = Rcpp::wrap(rho_stored);
        }

        if (lag_par1_prior.infer)
        {
            output["lag_par1"] = Rcpp::wrap(lag_par1_stored);
        }
        if (lag_par2_prior.infer)
        {
            output["lag_par2"] = Rcpp::wrap(lag_par2_stored);
        }

        if (a_intercept_prior.infer)
        {
            Rcpp::List intercept_a;
            if (a_intercept_prior.infer)
            {
                intercept_a["intercept"] = Rcpp::wrap(a_intercept_stored);
            } // end of intercept_a intercept output
            output["intercept_a"] = intercept_a;
        } // end of intercept_a output

        if (
            coef_self_sigma2_prior.infer || 
            coef_self_temporal_prior.infer || 
            coef_self_bym2_prior.infer || 
            infer_psi1_spatial
        )
        {
            Rcpp::List coef_self_b;
            if (coef_self_sigma2_prior.infer)
            {
                coef_self_b["sigma2"] = Rcpp::wrap(coef_self_sigma2_stored);
            } // end of coef_self_b sigma2 output
            if (coef_self_temporal_prior.infer)
            {
                coef_self_b["psi2_temporal"] = Rcpp::wrap(psi2_temporal_stored);
                coef_self_b["psi2_temporal_accept_rate"] = psi2_temporal_accept / (nburnin + nsample * nthin);
            } // end of coef_self_b temporal output

            Rcpp::List spatial_coef_self_out;
            if (coef_self_bym2_prior.infer)
            {
                spatial_coef_self_out["mu"] = Rcpp::wrap(coef_self_b_car_mu_stored);
                spatial_coef_self_out["tau"] = Rcpp::wrap(coef_self_b_car_tau_stored);
                spatial_coef_self_out["phi"] = Rcpp::wrap(coef_self_b_car_phi_stored);
                spatial_coef_self_out["phi_accept_rate"] = coef_self_bym2_prior.accept_count / (nburnin + nsample * nthin);
                spatial_coef_self_out["final_phi_mh_sd"] = coef_self_bym2_prior.mh_sd;
            }
            if (infer_psi1_spatial)
            {
                spatial_coef_self_out["psi1_spatial"] = Rcpp::wrap(psi1_spatial_stored);
                spatial_coef_self_out["hmc_settings"] = Rcpp::List::create(
                    Rcpp::Named("accept_rate") = spatial_coef_self_hmc_diagnostics.accept_count / (nburnin + nsample * nthin),
                    Rcpp::Named("leapfrog_step_size") = spatial_coef_self_hmc.leapfrog_step_size,
                    Rcpp::Named("nleapfrog") = spatial_coef_self_hmc.nleapfrog
                );
                if (spatial_coef_self_hmc.diagnostics)
                {
                    Rcpp::List hmc_diagnostics = Rcpp::List::create(
                        Rcpp::Named("energy_diff") = spatial_coef_self_hmc_diagnostics.energy_diff,
                        Rcpp::Named("grad_norm") = spatial_coef_self_hmc_diagnostics.grad_norm
                    );

                    if (spatial_coef_self_hmc.dual_averaging)
                    {
                        hmc_diagnostics["n_leapfrog"] = spatial_coef_self_hmc_diagnostics.nleapfrog_stored;
                        hmc_diagnostics["step_size"] = spatial_coef_self_hmc_diagnostics.leapfrog_step_size_stored;
                    }

                    spatial_coef_self_out["hmc_diagnostics"] = hmc_diagnostics;
                }
            } // end of coef_self_b spatial car output

            if (coef_self_bym2_prior.infer || infer_psi1_spatial)
            {
                coef_self_b["spatial"] = spatial_coef_self_out;
            }
            output["coef_self_b"] = coef_self_b;
        } // end of coef_self_b output

        if (!global_hmc.params_selected.empty())
        {
            output["global_accept_rate"] = global_hmc_diagnostics.accept_count / (nburnin + nsample * nthin);
            output["global_hmc_settings"] = Rcpp::List::create(
                Rcpp::Named("nleapfrog") = global_hmc.nleapfrog,
                Rcpp::Named("leapfrog_step_size") = global_hmc.leapfrog_step_size
            );

            if (global_hmc.diagnostics && global_hmc.dual_averaging)
            {
                output["global_diagnostics"] = Rcpp::List::create(
                    Rcpp::Named("energy_diff") = global_hmc_diagnostics.energy_diff,
                    Rcpp::Named("grad_norm") = global_hmc_diagnostics.grad_norm,
                    Rcpp::Named("n_leapfrog") = global_hmc_diagnostics.nleapfrog_stored,
                    Rcpp::Named("step_size") = global_hmc_diagnostics.leapfrog_step_size_stored
                );
            }
            else if (global_hmc.diagnostics)
            {
                output["global_diagnostics"] = Rcpp::List::create(
                    Rcpp::Named("energy_diff") = global_hmc_diagnostics.energy_diff,
                    Rcpp::Named("grad_norm") = global_hmc_diagnostics.grad_norm
                );
            } // end of global diagnostics output
        } // end of global HMC output

        if (!local_hmc.params_selected.empty())
        {
            output["local_accept_rate"] = Rcpp::wrap(local_hmc_diagnostics.accept_count / (nburnin + nsample * nthin));
            output["local_hmc_settings"] = Rcpp::List::create(
                Rcpp::Named("nleapfrog") = local_hmc.nleapfrog,
                Rcpp::Named("leapfrog_step_size") = local_hmc.leapfrog_step_size
            );

            if (local_hmc.diagnostics && local_hmc.dual_averaging)
            {
                output["local_diagnostics"] = Rcpp::List::create(
                    Rcpp::Named("energy_diff") = local_hmc_diagnostics.energy_diff,
                    Rcpp::Named("grad_norm") = local_hmc_diagnostics.grad_norm,
                    Rcpp::Named("n_leapfrog") = local_hmc_diagnostics.nleapfrog_stored,
                    Rcpp::Named("step_size") = local_hmc_diagnostics.leapfrog_step_size_stored
                );
            }
            else if (local_hmc.diagnostics)
            {
                output["local_diagnostics"] = Rcpp::List::create(
                    Rcpp::Named("energy_diff") = local_hmc_diagnostics.energy_diff,
                    Rcpp::Named("grad_norm") = local_hmc_diagnostics.grad_norm
                );
            } // end of local diagnostics output
        } // end of local HMC output

        return output;
    } // end of get_output()
}; // class mcmc


#endif