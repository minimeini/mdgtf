#pragma once
#ifndef MODEL_H
#define MODEL_H

#include <RcppArmadillo.h>
#include <progress.hpp>
#include <progress_bar.hpp>
#ifdef DGTF_USE_OPENMP
    #include <omp.h>
#endif

#include "../core/ErrDist.hpp"
#include "../core/SysEq.hpp"
#include "../core/TransFunc.hpp"
#include "../core/ObsDist.hpp"
#include "../core/LinkFunc.hpp"
#include "../core/Regression.hpp"
#include "../utils/utils.h"
#include "../spatial/bym2.hpp"

// [[Rcpp::plugins(cpp17)]]
// [[Rcpp::depends(RcppArmadillo, RcppProgress)]]


class Model
{
public:
    std::string dobs = "nbinom";
    std::string flink = "identity";
    std::string fgain = "exponential";

    unsigned int nS; // number of locations for spatio-temporal model
    arma::vec rho; // nS x 1
    BYM2 spatial;
    Param intercept_a, coef_self_b, coef_cross_c;

    Model()
    {
        nS = 1;
        spatial = BYM2(nS);

        rho.set_size(nS);
        rho.fill(30.0); // default dispersion for NegBin

        intercept_a.has_intercept = false; // no constant term
        intercept_a.has_temporal = true; // temporal RW
        intercept_a.sigma2 = 0.01;

        coef_self_b.intercept = 0.0; // constant
        coef_cross_c.intercept = 0.0; // constant
        return;
    } // end of Model()

    Model(const Rcpp::List &settings)
    {
        if (settings.containsElementNamed("spatial"))
        {
            Rcpp::List spatial_settings = settings["spatial"];
            if (spatial_settings.containsElementNamed("neighborhood_matrix"))
            {
                arma::mat V_r = Rcpp::as<arma::mat>(spatial_settings["neighborhood_matrix"]);
                arma::mat V = arma::symmatu(V_r); // ensure symmetry
                V.diag().zeros();                 // zero diagonal
                nS = V.n_rows;
                spatial = BYM2(V);
            }
            else
            {
                throw std::invalid_argument("Model::Model - neighborhood matrix 'neighborhood_matrix' is missing.");
            } // end of initialize neighborhood matrix V
        }
        else
        {
            nS = 1;
            spatial = BYM2(nS);
        } // end of initialize spatial settings

        if (settings.containsElementNamed("rho"))
        {
            rho = Rcpp::as<arma::vec>(settings["rho"]);
            if (rho.n_elem != nS)
            {
                throw std::invalid_argument("Model::Model - length of 'rho' should be equal to 'nlocation'.");
            }
        }
        else
        {
            rho.set_size(nS);
            rho.fill(30.0); // default dispersion for NegBin
        } // end of initialize dispersion parameter rho

        if (settings.containsElementNamed("intercept"))
        {
            Rcpp::List opts = settings["intercept"];
            intercept_a = Param(opts, spatial.V);
        }
        else
        {
            intercept_a.has_intercept = false; // no constant term
            intercept_a.has_temporal = true;   // temporal RW
            intercept_a.sigma2 = 0.01;
        }

        if (settings.containsElementNamed("coef_self"))
        {
            Rcpp::List opts = settings["coef_self"];
            coef_self_b = Param(opts, spatial.V);
        }
        else
        {
            coef_self_b.intercept = 0.0; // constant
        }

        if (settings.containsElementNamed("coef_cross"))
        {
            Rcpp::List opts = settings["coef_cross"];
            coef_cross_c = Param(opts, spatial.V);
        }
        else
        {
            coef_cross_c.intercept = 0.0; // constant
        }
    } // end of Model(const Rcpp::List &settings)

    void simulate(
        arma::mat &Y, // nS x (nT + 1)
        arma::mat &Lambda, // nS x (nT + 1)
        arma::vec &psi2_temporal, // (nT + 1) x 1
        arma::vec &psi4_temporal, // (nT + 1) x 1
        arma::vec &psi1_spatial, // nS x 1
        arma::vec &psi3_spatial, // nS x 1
        arma::mat &wt, // nS x (nT + 1)
        arma::mat &a_temporal, // nS x (nT + 1)
        const unsigned int &ntime)
    {
        Y.set_size(nS, ntime + 1);
        Y.zeros();
        Lambda = Y;

        if (coef_self_b.has_spatial)
        {
            psi1_spatial = coef_self_b.spatial.sample_spatial_effects_vec();
        }

        if (coef_self_b.has_temporal)
        {
            psi2_temporal.set_size(ntime + 1);
            psi2_temporal.randn();
            psi2_temporal *= std::sqrt(coef_self_b.sigma2);
            psi2_temporal = arma::cumsum(psi2_temporal);
        }

        if (coef_cross_c.has_spatial)
        {
            psi3_spatial = coef_cross_c.spatial.sample_spatial_effects_vec();
        }

        if (coef_cross_c.has_temporal)
        {
            psi4_temporal.set_size(ntime + 1);
            psi4_temporal.randn();
            psi4_temporal *= std::sqrt(coef_cross_c.sigma2);
            psi4_temporal = arma::cumsum(psi4_temporal);
        }

        if (intercept_a.has_temporal)
        {
            wt.set_size(nS, ntime + 1);
            wt.randn();
            wt *= std::sqrt(intercept_a.sigma2);
            a_temporal = arma::cumsum(wt, 1);
        }
        else
        {
            a_temporal.set_size(nS, ntime + 1);
            a_temporal.zeros();
        }

        const double baseline = intercept_a.has_intercept ? intercept_a.intercept : 0.0;

        for (unsigned int t = 1; t <= ntime; t++)
        {
            // State evolution
            // Construct state vectors at time t for all locations
            for (unsigned int s = 0; s < nS; s++)
            {
                double log_mu = baseline;
                log_mu += intercept_a.has_temporal ? a_temporal.at(s, t) : 0.0;

                double log_coef_self = coef_self_b.intercept;
                log_coef_self += coef_self_b.has_spatial ? psi1_spatial.at(s) : 0.0;
                log_coef_self += coef_self_b.has_temporal ? psi2_temporal.at(t) : 0.0;

                double log_coef_cross = coef_cross_c.intercept;
                log_coef_cross += coef_cross_c.has_spatial ? psi3_spatial.at(s) : 0.0;
                log_coef_cross += coef_cross_c.has_temporal ? psi4_temporal.at(t) : 0.0;

                Lambda.at(s, t) = std::exp(std::min(log_mu, UPBND));
                Lambda.at(s, t) += std::exp(std::min(log_coef_self, UPBND)) * Y.at(s, t - 1);
                Lambda.at(s, t) += std::exp(std::min(log_coef_cross, UPBND)) * arma::dot(
                    spatial.W.row(s).t(), Y.col(t-1)
                );

                Y.at(s, t) = ObsDist::sample(Lambda.at(s, t), rho.at(s), dobs);

            } // end of for s in [0, nS]
        } // end of for t in [1, ntime]

        return;
    } // end of simulate()

    double sample_posterior_predictive_y(
        arma::cube &Y_pred, // nS x (nT + 1) x (nsample * nrep)
        arma::cube &Y_residual, // nS x (nT + 1) x nsample
        const Rcpp::List &output, 
        const Rcpp::List &mcmc_opts,
        const arma::mat &Y, 
        const unsigned int &nsample,
        const unsigned int &nrep = 1
    )
    {
        std::map<std::string, AVAIL::Dist> obs_list = ObsDist::obs_list;
        const unsigned int ntime = Y.n_cols - 1;
        const unsigned int nS = Y.n_rows;

        arma::cube wt_stored(nS, ntime + 1, nsample);
        if (output.containsElementNamed("wt_samples"))
        {
            wt_stored = Rcpp::as<arma::cube>(output["wt_samples"]);
        }
        else if (output.containsElementNamed("wt"))
        {
            wt_stored = Rcpp::as<arma::cube>(output["wt"]);
        }
        else if (mcmc_opts.containsElementNamed("a"))
        {
            Rcpp::List a_opts = mcmc_opts["a"];
            if (a_opts.containsElementNamed("temporal"))
            {
                Rcpp::List a_temporal_opts = a_opts["temporal"];
                if (a_temporal_opts.containsElementNamed("init"))
                {
                    arma::mat wt = Rcpp::as<arma::mat>(a_temporal_opts["init"]);
                    wt_stored.each_slice() = wt;
                }
            }
        }
        else
        {
            wt_stored.zeros();
        }
        
        arma::mat rho_stored;
        if (output.containsElementNamed("rho"))
        {
            rho_stored = Rcpp::as<arma::mat>(output["rho"]);
        }
        else
        {
            rho_stored.set_size(nS, nsample);
            rho_stored.each_col() = rho;
        }

        arma::vec a_intercept_stored;
        if (output.containsElementNamed("intercept_a"))
        {
            Rcpp::List intercept_a_opts = output["intercept_a"];
            if (intercept_a_opts.containsElementNamed("intercept"))
            {
                a_intercept_stored = Rcpp::as<arma::vec>(intercept_a_opts["intercept"]);
            }
            else
            {
                a_intercept_stored = arma::vec(nsample, arma::fill::value(intercept_a.intercept));
            }
        }
        else
        {
            a_intercept_stored = arma::vec(nsample, arma::fill::value(intercept_a.intercept));
        }

        arma::vec a_sigma2_stored;
        if (output.containsElementNamed("a_sigma2"))
        {
            a_sigma2_stored = Rcpp::as<arma::vec>(output["a_sigma2"]);
        }
        else
        {
            a_sigma2_stored = arma::vec(nsample, arma::fill::value(intercept_a.sigma2));
        }

        arma::vec coef_self_intercept_stored(nsample, arma::fill::zeros);
        arma::vec coef_self_sigma2_stored(nsample, arma::fill::zeros);
        arma::mat coef_self_temporal_stored(ntime + 1, nsample, arma::fill::zeros);
        arma::mat coef_self_spatial_stored(nS, nsample, arma::fill::zeros);
        arma::vec coef_self_car_mu_stored(nsample, arma::fill::zeros);
        arma::vec coef_self_car_tau_stored(nsample, arma::fill::zeros);
        arma::vec coef_self_car_phi_stored(nsample, arma::fill::zeros);
        if (output.containsElementNamed("coef_self_b"))
        {
            Rcpp::List coef_self_opts = output["coef_self_b"];
            Rcpp::List coef_self_mcmc_opts = mcmc_opts["coef_self"];
            if (coef_self_opts.containsElementNamed("intercept"))
            {
                coef_self_intercept_stored = Rcpp::as<arma::vec>(coef_self_opts["intercept"]);
            }
            else
            {
                coef_self_intercept_stored = arma::vec(nsample, arma::fill::zeros);
            }

            if (coef_self_opts.containsElementNamed("sigma2"))
            {
                coef_self_sigma2_stored = Rcpp::as<arma::vec>(coef_self_opts["sigma2"]);
            }
            else
            {
                coef_self_sigma2_stored = arma::vec(nsample, arma::fill::value(coef_self_b.sigma2));
            }

            if (coef_self_opts.containsElementNamed("psi2_temporal"))
            {
                coef_self_temporal_stored = Rcpp::as<arma::mat>(coef_self_opts["psi2_temporal"]);
            }
            else if (coef_self_mcmc_opts.containsElementNamed("temporal"))
            {
                Rcpp::List coef_self_temporal_opts = coef_self_mcmc_opts["temporal"];
                if (coef_self_temporal_opts.containsElementNamed("init"))
                {
                    arma::vec coef_self_temporal_init = Rcpp::as<arma::vec>(coef_self_temporal_opts["init"]);
                    coef_self_temporal_stored.each_col() = coef_self_temporal_init;
                }
            }

            if (coef_self_opts.containsElementNamed("spatial"))
            {
                Rcpp::List coef_self_spatial_opts = coef_self_opts["spatial"];
                coef_self_spatial_stored = Rcpp::as<arma::mat>(coef_self_spatial_opts["psi1_spatial"]);
                coef_self_car_mu_stored = Rcpp::as<arma::vec>(coef_self_spatial_opts["mu"]);
                coef_self_car_tau_stored = Rcpp::as<arma::vec>(coef_self_spatial_opts["tau"]);
                coef_self_car_phi_stored = Rcpp::as<arma::vec>(coef_self_spatial_opts["phi"]);
            }
            else if (coef_self_mcmc_opts.containsElementNamed("spatial"))
            {
                Rcpp::List coef_self_spatial_opts = coef_self_mcmc_opts["spatial"];
                if (coef_self_spatial_opts.containsElementNamed("init"))
                {
                    arma::mat coef_self_spatial_init = Rcpp::as<arma::mat>(coef_self_spatial_opts["init"]);
                    coef_self_spatial_stored.each_col() = coef_self_spatial_init;
                }

                coef_self_car_mu_stored.fill(coef_self_b.spatial.mu);
                coef_self_car_tau_stored.fill(coef_self_b.spatial.tau_b);
                coef_self_car_phi_stored.fill(coef_self_b.spatial.phi);
            }
        }


        arma::vec coef_cross_intercept_stored(nsample, arma::fill::zeros);
        arma::vec coef_cross_sigma2_stored(nsample, arma::fill::zeros);
        arma::mat coef_cross_temporal_stored(ntime + 1, nsample, arma::fill::zeros);
        arma::mat coef_cross_spatial_stored(nS, nsample, arma::fill::zeros);
        arma::vec coef_cross_car_mu_stored(nsample, arma::fill::zeros);
        arma::vec coef_cross_car_tau_stored(nsample, arma::fill::zeros);
        arma::vec coef_cross_car_phi_stored(nsample, arma::fill::zeros);
        if (output.containsElementNamed("coef_cross_c"))
        {
            Rcpp::List coef_cross_opts = output["coef_cross_c"];
            Rcpp::List coef_cross_mcmc_opts = mcmc_opts["coef_cross"];
            if (coef_cross_opts.containsElementNamed("intercept"))
            {
                coef_cross_intercept_stored = Rcpp::as<arma::vec>(coef_cross_opts["intercept"]);
            }
            else
            {
                coef_cross_intercept_stored = arma::vec(nsample, arma::fill::zeros);
            }

            if (coef_cross_opts.containsElementNamed("sigma2"))
            {
                coef_cross_sigma2_stored = Rcpp::as<arma::vec>(coef_cross_opts["sigma2"]);
            }
            else
            {
                coef_cross_sigma2_stored = arma::vec(nsample, arma::fill::value(coef_cross_c.sigma2));
            }

            if (coef_cross_opts.containsElementNamed("psi4_temporal"))
            {
                coef_cross_temporal_stored = Rcpp::as<arma::mat>(coef_cross_opts["psi4_temporal"]);
            }
            else if (coef_cross_mcmc_opts.containsElementNamed("temporal"))
            {
                Rcpp::List coef_cross_temporal_opts = coef_cross_mcmc_opts["temporal"];
                if (coef_cross_temporal_opts.containsElementNamed("init"))
                {
                    arma::vec coef_cross_temporal_init = Rcpp::as<arma::vec>(coef_cross_temporal_opts["init"]);
                    coef_cross_temporal_stored.each_col() = coef_cross_temporal_init;
                }
            }

            if (coef_cross_opts.containsElementNamed("spatial"))
            {
                Rcpp::List coef_cross_spatial_opts = coef_cross_opts["spatial"];
                coef_cross_spatial_stored = Rcpp::as<arma::mat>(coef_cross_spatial_opts["psi3_spatial"]);
                coef_cross_car_mu_stored = Rcpp::as<arma::vec>(coef_cross_spatial_opts["mu"]);
                coef_cross_car_tau_stored = Rcpp::as<arma::vec>(coef_cross_spatial_opts["tau"]);
                coef_cross_car_phi_stored = Rcpp::as<arma::vec>(coef_cross_spatial_opts["phi"]);
            }
            else if (coef_cross_mcmc_opts.containsElementNamed("spatial"))
            {
                Rcpp::List coef_cross_spatial_opts = coef_cross_mcmc_opts["spatial"];
                if (coef_cross_spatial_opts.containsElementNamed("init"))
                {
                    arma::mat coef_cross_spatial_init = Rcpp::as<arma::mat>(coef_cross_spatial_opts["init"]);
                    coef_cross_spatial_stored.each_col() = coef_cross_spatial_init;
                }

                coef_cross_car_mu_stored.fill(coef_cross_c.spatial.mu);
                coef_cross_car_tau_stored.fill(coef_cross_c.spatial.tau_b);
                coef_cross_car_phi_stored.fill(coef_cross_c.spatial.phi);
            }
        }

        Y_pred.set_size(nS, ntime + 1, nsample * nrep);
        Y_pred.zeros();
        Y_residual.set_size(nS, ntime + 1, nsample);
        Y_residual.zeros();

        arma::vec chi_sqr(nsample, arma::fill::zeros);
        Progress p(nsample, true);
        #ifdef DGTF_USE_OPENMP
        #pragma omp parallel for num_threads(NUM_THREADS) schedule(static)
        #endif
        for (unsigned int i = 0; i < nsample; i++)
        {
            Model model_i = *this;
            model_i.intercept_a.intercept = a_intercept_stored.at(i);
            model_i.intercept_a.sigma2 = a_sigma2_stored.at(i);
            arma::mat wt = wt_stored.slice(i);
            arma::mat a_temporal = arma::cumsum(wt, 1); // nS x (nT + 1)
            a_temporal.zeros();

            model_i.coef_self_b.intercept = coef_self_intercept_stored.at(i);
            model_i.coef_self_b.sigma2 = coef_self_sigma2_stored.at(i);
            arma::vec psi2_temporal = coef_self_temporal_stored.col(i);
            arma::vec psi1_spatial = coef_self_spatial_stored.col(i);
            model_i.coef_self_b.spatial = BYM2(model_i.coef_self_b.spatial.V);
            model_i.coef_self_b.spatial.mu = coef_self_car_mu_stored.at(i);
            model_i.coef_self_b.spatial.tau_b = coef_self_car_tau_stored.at(i);
            model_i.coef_self_b.spatial.phi = coef_self_car_phi_stored.at(i);

            model_i.coef_cross_c.intercept = coef_cross_intercept_stored.at(i);
            model_i.coef_cross_c.sigma2 = coef_cross_sigma2_stored.at(i);
            arma::vec psi4_temporal = coef_cross_temporal_stored.col(i);
            arma::vec psi3_spatial = coef_cross_spatial_stored.col(i);
            model_i.coef_cross_c.spatial = BYM2(model_i.coef_cross_c.spatial.V);
            model_i.coef_cross_c.spatial.mu = coef_cross_car_mu_stored.at(i);
            model_i.coef_cross_c.spatial.tau_b = coef_cross_car_tau_stored.at(i);
            model_i.coef_cross_c.spatial.phi = coef_cross_car_phi_stored.at(i);

            model_i.rho = rho_stored.col(i);

            const double baseline = model_i.intercept_a.has_intercept ? model_i.intercept_a.intercept : 0.0;

            arma::mat Lambda(nS, ntime + 1, arma::fill::zeros);
            arma::mat ytmp(nS, ntime, arma::fill::zeros);
            for (unsigned int t = 1; t <= ntime; t++)
            {
                for (unsigned int s = 0; s < nS; s++)
                {
                    double log_mu = baseline;
                    log_mu += model_i.intercept_a.has_temporal ? a_temporal.at(s, t) : 0.0;

                    double log_coef_self = model_i.coef_self_b.intercept;
                    log_coef_self += model_i.coef_self_b.has_spatial ? psi1_spatial.at(s) : 0.0;
                    log_coef_self += model_i.coef_self_b.has_temporal ? psi2_temporal.at(t) : 0.0;

                    double log_coef_cross = model_i.coef_cross_c.intercept;
                    log_coef_cross += model_i.coef_cross_c.has_spatial ? psi3_spatial.at(s) : 0.0;
                    log_coef_cross += model_i.coef_cross_c.has_temporal ? psi4_temporal.at(t) : 0.0;

                    double eta = std::exp(std::min(log_mu, UPBND));
                    eta += std::exp(std::min(log_coef_self, UPBND)) * Y.at(s, t - 1);
                    eta += std::exp(std::min(log_coef_cross, UPBND)) * arma::dot(
                        model_i.spatial.W.row(s).t(), Y.col(t-1)
                    );

                    Lambda.at(s, t) = eta;
                    double mean, var;
                    switch (obs_list[model_i.dobs])
                    {
                    case AVAIL::Dist::nbinomm:
                    {
                        mean = Lambda.at(s, t);
                        var = std::abs(nbinomm::var(Lambda.at(s, t), model_i.rho.at(s)));
                        break;
                    }
                    case AVAIL::Dist::nbinomp:
                    {
                        mean = nbinom::mean(Lambda.at(s, t), model_i.rho.at(s));
                        var = nbinom::var(Lambda.at(s, t), model_i.rho.at(s));
                        break;
                    }
                    case AVAIL::Dist::poisson:
                    {
                        mean = Lambda.at(s, t);
                        var = Lambda.at(s, t);
                        break;
                    }
                    default:
                    {
                        throw std::invalid_argument("Unknown observation distribution.");
                        break;
                    }
                    } // end of switch by obs_list

                    Y_residual.at(s, t, i) = Lambda.at(s, t) - Y.at(s, t);
                    double yres2 = 2. * std::log(std::abs(Y_residual.at(s, t, i)) + EPS);
                    double yvar = std::log(var + EPS);
                    ytmp.at(s, t - 1) = std::exp(yres2 - yvar);

                    for (unsigned int r = 0; r < nrep; r++)
                    {
                        Y_pred.at(s, t, i * nrep + r) = ObsDist::sample(
                            Lambda.at(s, t), model_i.rho.at(s), model_i.dobs
                        );
                    } // end of for r in [0, nrep]

                } // end of for s in [0, nS]
            } // end of for t in [1, ntime]

            chi_sqr.at(i) = arma::mean(arma::mean(ytmp));
            p.increment();
        } // end of for i in [0, nsample]

        return arma::mean(chi_sqr);
    } // end of sample_posterior_predictive_y()

    arma::vec compute_intensity_iterative(
        const unsigned int &s,
        const arma::mat &Y, // nS x (nT + 1)
        const arma::vec &psi1_spatial, // nS x 1
        const arma::vec &psi2_temporal, // (nT + 1) x 1
        const arma::vec &psi3_spatial, // nS x 1
        const arma::vec &psi4_temporal, // (nT + 1) x 1
        const arma::mat &log_a // nS x (nT + 1)
    ) const
    {
        const arma::vec neighbor_weights = spatial.W.row(s).t(); // nS x 1
        const double coef_self_base = coef_self_b.intercept + (coef_self_b.has_spatial ? psi1_spatial.at(s) : 0.0);
        const double coef_cross_base = coef_cross_c.intercept + (coef_cross_c.has_spatial ? psi3_spatial.at(s) : 0.0);

        arma::vec Lambda(Y.n_cols, arma::fill::zeros);
        for (unsigned int t = 1; t < Y.n_cols; t++)
        {
            double mu = intercept_a.intercept;
            mu += intercept_a.has_temporal ? log_a.at(s, t) : 0.0;
            mu = std::exp(std::min(mu, UPBND));

            double coef_self = coef_self_base + (coef_self_b.has_temporal ? psi2_temporal.at(t) : 0.0);
            coef_self = std::exp(std::min(coef_self, UPBND));

            double coef_cross = coef_cross_base + (coef_cross_c.has_temporal ? psi4_temporal.at(t) : 0.0);
            coef_cross = std::exp(std::min(coef_cross, UPBND));

            Lambda.at(t) = mu + coef_self * Y.at(s, t - 1) + coef_cross * arma::dot(neighbor_weights, Y.col(t - 1));
        }

        return Lambda; // (nT + 1) x 1
    } // end of compute_intensity_iterative()


    void compute_intensity_iterative(
        arma::mat &Lambda, // nS x (nT + 1)
        const unsigned int &t_start,
        const arma::mat &Y, // nS x (nT + 1)
        const arma::vec &psi1_spatial, // nS x 1
        const arma::vec &psi2_temporal, // (nT + 1) x 1
        const arma::vec &psi3_spatial, // nS x 1
        const arma::vec &psi4_temporal, // (nT + 1) x 1
        const arma::mat &log_a // nS x (nT + 1)
    )
    {
        for (unsigned int t = t_start; t < Y.n_cols; t++)
        {
            const double coef_self_base = coef_self_b.intercept + (coef_self_b.has_temporal ? psi2_temporal.at(t) : 0.0);
            const double coef_cross_base = coef_cross_c.intercept + (coef_cross_c.has_temporal ? psi4_temporal.at(t) : 0.0);

            for (unsigned int s = 0; s < Y.n_rows; s++)
            {
                double mu = intercept_a.intercept + (intercept_a.has_temporal ? log_a.at(s, t) : 0.0);
                mu = std::exp(std::min(mu, UPBND));

                double coef_self = coef_self_base + (coef_self_b.has_spatial ? psi1_spatial.at(s) : 0.0);
                coef_self = std::exp(std::min(coef_self, UPBND));

                double coef_cross = coef_cross_base + (coef_cross_c.has_spatial ? psi3_spatial.at(s) : 0.0);
                coef_cross = std::exp(std::min(coef_cross, UPBND));

                Lambda.at(s, t) = mu + coef_self * Y.at(s, t - 1) + coef_cross * arma::dot(spatial.W.row(s).t(), Y.col(t - 1));
            }
        }

        return; // (nT + 1) x 1
    } // end of compute_intensity_iterative()


    double dloglik_deta(const double &eta, const double &y, const double &obs_par2) const
    {
        return nbinomm::dlogp_dlambda(eta, y, obs_par2);
    } // end of dloglike_deta()

    arma::mat dloglik_deta(
        const arma::mat &Y, // nS x (nT + 1)
        const arma::vec &psi1_spatial, // nS x 1
        const arma::vec &psi2_temporal, // (nT + 1) x 1
        const arma::vec &psi3_spatial, // nS x 1
        const arma::vec &psi4_temporal, // (nT + 1) x 1
        const arma::mat &log_a // nS x (nT + 1)
    ) const
    {
        const unsigned int nT = Y.n_cols - 1;
        arma::mat dll_deta(Y.n_rows, Y.n_cols, arma::fill::zeros);
        for (unsigned int s = 0; s < Y.n_rows; s++)
        {
            const arma::vec neighbor_weights = spatial.W.row(s).t();
            const arma::vec cross_region_effects = Y.head_cols(nT).t() * neighbor_weights; // nT x 1
            const double coef_self_base = coef_self_b.intercept + (coef_self_b.has_spatial ? psi1_spatial.at(s) : 0.0);
            const double coef_cross_base = coef_cross_c.intercept + (coef_cross_c.has_spatial ? psi3_spatial.at(s) : 0.0);

            for (unsigned int t = 1; t < Y.n_cols; t++)
            {
                double mu = intercept_a.intercept + (intercept_a.has_temporal ? log_a.at(s, t) : 0.0);
                mu = std::exp(std::min(mu, UPBND));

                double coef_self = coef_self_base + (coef_self_b.has_temporal ? psi2_temporal.at(t) : 0.0);
                coef_self = std::exp(std::min(coef_self, UPBND));

                double coef_cross = coef_cross_base + (coef_cross_c.has_temporal ? psi4_temporal.at(t) : 0.0);
                coef_cross = std::exp(std::min(coef_cross, UPBND));

                double eta = mu + coef_self * Y.at(s, t - 1) + coef_cross * cross_region_effects.at(t - 1);
                dll_deta.at(s, t) = dloglik_deta(eta, Y.at(s, t), rho.at(s));
            }
        }

        return dll_deta;
    } // end of dloglik_deta()


    double dloglik_dintercept_a(
        const arma::mat &Y,
        const arma::mat &dll_deta,
        const arma::mat &log_a
    )
    {
        double deriv = 0.0;
        for (unsigned int s = 0; s < nS; s++)
        {
            for (unsigned int t = 1; t < Y.n_cols; t++)
            {
                double intercept = std::exp(std::min(intercept_a.intercept + log_a.at(s, t), UPBND));
                deriv += dll_deta.at(s, t) * intercept;
            }
        }

        return deriv;
    }


    double dloglik_dlogsig_a(const arma::mat &Y, const arma::mat &wt)
    {
        double ntime = 0.0;
        double res2 = 0.0;
        for (unsigned int s = 0; s < nS; s++)
        {
            for (unsigned int t = 1; t < Y.n_cols; t++)
            {
                ntime += 1.0;
                res2 += wt.at(s, t) * wt.at(s, t);
            }
        }
        return - 0.5 * ntime + 0.5 * res2 / std::max(intercept_a.sigma2, EPS);
    } // end of dloglik_dlogsig_a()


    double dloglik_dintercept_coef_self(
        const arma::mat &Y,
        const arma::mat &dll_deta,
        const arma::vec &psi1_spatial,
        const arma::vec &psi2_temporal
    )
    {
        double deriv = 0.0;
        for (unsigned int s = 0; s < nS; s++)
        {
            for (unsigned int t = 1; t < Y.n_cols; t++)
            {
                double coef_self = coef_self_b.intercept;
                coef_self += coef_self_b.has_spatial ? psi1_spatial.at(s) : 0.0;
                coef_self += coef_self_b.has_temporal ? psi2_temporal.at(t) : 0.0;
                deriv += dll_deta.at(s, t) * Y.at(s, t - 1) * std::exp(std::min(coef_self, UPBND));
            }
        }

        return deriv;
    } // end of dloglik_dintercept_coef_self()

    arma::vec dloglik_dspatial_coef_self(
        const arma::mat &Y,
        const arma::mat &dll_deta,
        const arma::vec &psi1_spatial,
        const arma::vec &psi2_temporal
    )
    {
        arma::vec grad(nS, arma::fill::zeros);
        for (unsigned int s = 0; s < nS; s++)
        {
            for (unsigned int t = 1; t < Y.n_cols; t++)
            {
                double coef_self = coef_self_b.intercept;
                coef_self += coef_self_b.has_spatial ? psi1_spatial.at(s) : 0.0;
                coef_self += coef_self_b.has_temporal ? psi2_temporal.at(t) : 0.0;
                grad.at(s) += dll_deta.at(s, t) * Y.at(s, t - 1) * std::exp(std::min(coef_self, UPBND));
            }
        }

        grad += coef_self_b.spatial.dloglik_dspatial(psi1_spatial);
        return grad;
    } // end of dloglik_dspatial_coef_self()


    double dloglik_dlogsig_coef_self(const arma::mat &Y, const arma::vec &wt2_temporal)
    {
        double ntime = 0.0;
        double res2 = 0.0;
        for (unsigned int t = 1; t < Y.n_cols; t++)
        {
            ntime += 1.0;
            res2 += wt2_temporal.at(t) * wt2_temporal.at(t);
        }

        return - 0.5 * ntime + 0.5 * res2 / std::max(coef_self_b.sigma2, EPS);
    } // end of dloglik_dlogsig_coef_self()


    double dloglik_dintercept_coef_cross(
        const arma::mat &Y, // nS x (nT + 1)
        const arma::mat &dll_deta, // nS x (nT + 1)
        const arma::vec &psi3_spatial,
        const arma::vec &psi4_temporal
    )
    {
        double deriv = 0.0;
        for (unsigned int s = 0; s < nS; s++)
        {
            for (unsigned int t = 1; t < Y.n_cols; t++)
            {
                double coef_cross = coef_cross_c.intercept;
                coef_cross += coef_cross_c.has_spatial ? psi3_spatial.at(s) : 0.0;
                coef_cross += coef_cross_c.has_temporal ? psi4_temporal.at(t) : 0.0;
                double deta_dcoef_cross = arma::dot(spatial.W.row(s).t(), Y.col(t - 1));
                // double dbeta_dlogbeta = beta;
                deriv += dll_deta.at(s, t) * deta_dcoef_cross * std::exp(std::min(coef_cross, UPBND));
            }
        }

        return deriv;
    } // end of dloglik_dintercept_coef_cross()


    arma::vec dloglik_dspatial_coef_cross(
        const arma::mat &Y, // nS x (nT + 1)
        const arma::mat &dll_deta, // nS x (nT + 1)
        const arma::vec &psi3_spatial,
        const arma::vec &psi4_temporal
    )
    {
        arma::vec grad(nS, arma::fill::zeros);
        for (unsigned int s = 0; s < nS; s++)
        {
            for (unsigned int t = 1; t < Y.n_cols; t++)
            {
                double coef_cross = coef_cross_c.intercept;
                coef_cross += coef_cross_c.has_spatial ? psi3_spatial.at(s) : 0.0;
                coef_cross += coef_cross_c.has_temporal ? psi4_temporal.at(t) : 0.0;
                double deta_dcoef_cross = arma::dot(spatial.W.row(s).t(), Y.col(t - 1));
                grad.at(s) += dll_deta.at(s, t) * deta_dcoef_cross * std::exp(std::min(coef_cross, UPBND));
            }
        }

        grad += coef_cross_c.spatial.dloglik_dspatial(psi3_spatial);
        return grad;
    } // end of dloglik_dspatial_coef_cross()


    double dloglik_dlogsig_coef_cross(const arma::mat &Y, const arma::vec &wt4_temporal)
    {
        double ntime = 0.0;
        double res2 = 0.0;
        for (unsigned int t = 1; t < Y.n_cols; t++)
        {
            ntime += 1.0;
            res2 += wt4_temporal.at(t) * wt4_temporal.at(t);
        }

        return - 0.5 * ntime + 0.5 * res2 / std::max(coef_cross_c.sigma2, EPS);
    } // end of dloglik_dlogsig_coef_cross()


    double dloglik_dlogrho(const unsigned int &s, const arma::vec &y, const arma::vec &lambda)
    {
        double dll_dlogrho = 0.0;
        for (unsigned int t = 1; t < y.n_elem; t++)
        {
            dll_dlogrho += nbinomm::dlogp_dpar2(y.at(t), lambda.at(t), rho.at(s), true);
        }

        return dll_dlogrho;
    } // end of dloglik_dlogrho()


    arma::vec get_global_params_unconstrained(const std::vector<std::string> &global_params)
    {
        std::map<std::string, AVAIL::Param> static_param_list = AVAIL::static_param_list;
        arma::vec unconstrained_params(global_params.size(), arma::fill::zeros);
        for (unsigned int i = 0; i < global_params.size(); i++)
        {
            switch (static_param_list[tolower(global_params[i])])
            {
            case AVAIL::Param::ar_a_intercept:
            {
                unconstrained_params.at(i) = intercept_a.intercept;
                break;
            }
            case AVAIL::Param::ar_a_sigma2:
            {
                unconstrained_params.at(i) = std::log(std::max(intercept_a.sigma2, EPS));
                break;
            }
            case AVAIL::Param::ar_coef_self_intercept:
            {
                unconstrained_params.at(i) = coef_self_b.intercept;
                break;
            }
            case AVAIL::Param::ar_coef_self_sigma2:
            {
                unconstrained_params.at(i) = std::log(std::max(coef_self_b.sigma2, EPS));
                break;
            }
            case AVAIL::Param::ar_coef_cross_intercept:
            {
                unconstrained_params.at(i) = coef_cross_c.intercept;
                break;
            }
            case AVAIL::Param::ar_coef_cross_sigma2:
            {
                unconstrained_params.at(i) = std::log(std::max(coef_cross_c.sigma2, EPS));
                break;
            }
            default:
            {
                throw std::invalid_argument(
                    "Model::get_global_params_unconstrained: unrecognized global parameter name."
                );
            }
            }
        }

        return unconstrained_params;
    } // end of get_global_params_unconstrained()


    void update_global_params_unconstrained(
        const std::vector<std::string> &global_params, 
        const arma::vec &unconstrained_params
    )
    {
        std::map<std::string, AVAIL::Param> static_param_list = AVAIL::static_param_list;
        bool lag_updated = false;
        for (unsigned int i = 0; i < global_params.size(); i++)
        {
            switch (static_param_list[tolower(global_params[i])])
            {
            case AVAIL::Param::ar_a_intercept:
            {
                intercept_a.intercept = std::min(unconstrained_params.at(i), UPBND);
                break;
            }
            case AVAIL::Param::ar_a_sigma2:
            {
                intercept_a.sigma2 = std::exp(std::min(unconstrained_params.at(i), UPBND));
                break;
            }
            case AVAIL::Param::ar_coef_self_intercept:
            {
                coef_self_b.intercept = std::min(unconstrained_params.at(i), UPBND);
                break;
            }
            case AVAIL::Param::ar_coef_self_sigma2:
            {
                coef_self_b.sigma2 = std::exp(std::min(unconstrained_params.at(i), UPBND));
                break;
            }
            case AVAIL::Param::ar_coef_cross_intercept:
            {
                coef_cross_c.intercept = std::min(unconstrained_params.at(i), UPBND);
                break;
            }
            case AVAIL::Param::ar_coef_cross_sigma2:
            {
                coef_cross_c.sigma2 = std::exp(std::min(unconstrained_params.at(i), UPBND));
                break;
            }
            default:
            {
                throw std::invalid_argument(
                    "Model::update_global_params_unconstrained: unrecognized global parameter name."
                );
            }
            }
        }

        return;
    } // end of update_global_params_unconstrained()

    /**
     * @brief Compute the gradient of the log joint probability w.r.t. unconstrained global parameters (lag distribution parameters and beta)
     * 
     * @param global_params 
     * @param Y 
     * @param hPsi 
     * @param lag_par1_prior 
     * @param lag_par2_prior 
     * @param beta_prior 
     * @return arma::vec 
     */
    arma::vec dloglik_dglobal_unconstrained(
        const std::vector<std::string> &global_params,
        const arma::mat &Y,    // nS x (nT + 1)
        const arma::mat &wt, // nS x (nT + 1)
        const arma::vec &psi1_spatial,
        const arma::vec &wt2_temporal,
        const arma::vec &psi3_spatial,
        const arma::vec &wt4_temporal,
        const Prior &a_intercept_prior,
        const Prior &a_sigma2_prior,
        const Prior &coef_self_intercept_prior,
        const Prior &coef_self_sigma2_prior,
        const Prior &coef_cross_intercept_prior,
        const Prior &coef_cross_sigma2_prior
    )
    {
        std::map<std::string, AVAIL::Param> static_param_list = AVAIL::static_param_list;
        arma::vec grad(global_params.size(), arma::fill::zeros);

        const arma::mat log_a = arma::cumsum(wt, 1);
        const arma::vec psi2_temporal = arma::cumsum(wt2_temporal);
        const arma::vec psi4_temporal = arma::cumsum(wt4_temporal);

        arma::mat dll_deta = dloglik_deta(
            Y, psi1_spatial, psi2_temporal, psi3_spatial, psi4_temporal, log_a
        );
        for (unsigned int i = 0; i < global_params.size(); i++)
        {
            switch (static_param_list[tolower(global_params[i])])
            {
            case AVAIL::Param::ar_a_intercept:
            {
                grad.at(i) = dloglik_dintercept_a(Y, dll_deta, log_a);
                grad.at(i) += Prior::dlogprior_dpar(
                    intercept_a.intercept, a_intercept_prior, false
                );
                break;
            }
            case AVAIL::Param::ar_a_sigma2:
            {
                grad.at(i) = dloglik_dlogsig_a(Y, wt);
                grad.at(i) += Prior::dlogprior_dpar(
                    intercept_a.sigma2, a_sigma2_prior, true
                );
                break;
            }
            case AVAIL::Param::ar_coef_self_intercept:
            {
                grad.at(i) = dloglik_dintercept_coef_self(Y, dll_deta, psi1_spatial, psi2_temporal);
                grad.at(i) += Prior::dlogprior_dpar(
                    coef_self_b.intercept, coef_self_intercept_prior, false
                );
                break;
            }
            case AVAIL::Param::ar_coef_self_sigma2:
            {
                grad.at(i) = dloglik_dlogsig_coef_self(Y, wt2_temporal);
                grad.at(i) += Prior::dlogprior_dpar(
                    coef_self_b.sigma2, coef_self_sigma2_prior, true
                );
                break;
            }
            case AVAIL::Param::ar_coef_cross_intercept:
            {
                grad.at(i) = dloglik_dintercept_coef_cross(Y, dll_deta, psi3_spatial, psi4_temporal);
                grad.at(i) += Prior::dlogprior_dpar(
                    coef_cross_c.intercept, 
                    coef_cross_intercept_prior, false
                );
                break;
            }
            case AVAIL::Param::ar_coef_cross_sigma2:
            {
                grad.at(i) = dloglik_dlogsig_coef_cross(Y, wt4_temporal);
                grad.at(i) += Prior::dlogprior_dpar(
                    coef_cross_c.sigma2, 
                    coef_cross_sigma2_prior, true
                );
                break;
            }
            default:
            {
                throw std::invalid_argument(
                    "Model::dloglik_dglobal_unconstrained: unrecognized global parameter name.");
            }
            }
        }

        return grad;
    } // end of dloglik_dglobal_unconstrained()

    arma::vec get_local_params_unconstrained(
        const unsigned int &s, const std::vector<std::string> &local_params
    )
    {
        std::map<std::string, AVAIL::Param> static_param_list = AVAIL::static_param_list;
        arma::vec unconstrained_params(local_params.size(), arma::fill::zeros);
        for (unsigned int i = 0; i < local_params.size(); i++)
        {
            switch (static_param_list[tolower(local_params[i])])
            {
            case AVAIL::Param::rho:
            {
                unconstrained_params.at(i) = std::log(std::max(rho.at(s), EPS));
                break;
            }
            default:
            {
                throw std::invalid_argument(
                    "Model::get_local_params_unconstrained: unrecognized local parameter name.");
            }
            }
        }

        return unconstrained_params;
    } // end of get_local_params_unconstrained()

    void update_local_params_unconstrained(
        const unsigned int &s,
        const std::vector<std::string> &local_params,
        const arma::vec &unconstrained_params
    )
    {
        std::map<std::string, AVAIL::Param> static_param_list = AVAIL::static_param_list;
        for (unsigned int i = 0; i < local_params.size(); i++)
        {
            switch (static_param_list[tolower(local_params[i])])
            {
            case AVAIL::Param::rho:
            {
                rho.at(s) = std::exp(std::min(unconstrained_params.at(i), UPBND));
                break;
            }
            default:
            {
                throw std::invalid_argument(
                    "Model::update_local_params_unconstrained: unrecognized local parameter name.");
            }
            }
        }

        return;
    } // end of update_local_params_unconstrained()


    arma::vec dloglik_dlocal_unconstrained(
        const unsigned int &s,
        const std::vector<std::string> &local_params,
        const arma::vec &y,    // (nT + 1) x 1
        const arma::vec &lambda,   // (nT + 1) x 1
        const arma::vec &wt, // (nT + 1) x 1
        const Prior &rho_prior
    )
    {
        std::map<std::string, AVAIL::Param> static_param_list = AVAIL::static_param_list;
        arma::vec grad(local_params.size(), arma::fill::zeros);
        for (unsigned int i = 0; i < local_params.size(); i++)
        {
            switch (static_param_list[tolower(local_params[i])])
            {
            case AVAIL::Param::rho:
            {
                // dloglik_dlogrho: gradient of loglike w.r.t. log(rho)
                // dlogprior_dpar: gradient of log prior w.r.t. log(rho).
                // The inv-gamma prior is placed on rho
                grad.at(i) = dloglik_dlogrho(s, y, lambda) + Prior::dlogprior_dpar(rho.at(s), rho_prior, true);
                break;
            }
            default:
            {
                throw std::invalid_argument(
                    "Model::dloglik_dlocal_unconstrained: unrecognized local parameter name.");
            }
            }
        }

        return grad;
    } // end of dloglik_dlocal_unconstrained()
};

#endif