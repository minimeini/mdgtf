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
#include "SpatialStructure.hpp"

// [[Rcpp::plugins(cpp17)]]
// [[Rcpp::depends(RcppArmadillo, RcppProgress)]]


struct Param
{
    double intercept = 0.0;
    double sigma2 = 0.01;
    double car_mu = 0.0;
    double car_tau2 = 0.01;
    double car_rho = 0.5;

    bool has_intercept = true;
    bool has_temporal = false;
    bool has_car = false;

    Param()
    {
        has_intercept = true;
        intercept = 0.0;

        has_temporal = false;
        has_car = false;
        return;
    }

    Param(const Rcpp::List &settings)
    {
        if (settings.containsElementNamed("intercept"))
        {
            has_intercept = true;
            intercept = Rcpp::as<double>(settings["intercept"]);
        }
        else
        {
            has_intercept = false;
            intercept = 0.0;
        }

        if (settings.containsElementNamed("sigma2"))
        {
            has_temporal = true;
            sigma2 = Rcpp::as<double>(settings["sigma2"]);
        }
        else
        {
            has_temporal = false;
            sigma2 = 0.01;
        }

        if (settings.containsElementNamed("car_params"))
        {
            has_car = true;
            arma::vec car_params = Rcpp::as<arma::vec>(settings["car_params"]);
            if (car_params.n_elem != 3)
            {
                throw std::invalid_argument("Param::Param - 'car_params' should be a numeric vector of length 3 (car_mu, car_tau2, car_rho).");
            }

            car_mu = car_params.at(0);
            car_tau2 = car_params.at(1);
            car_rho = car_params.at(2);
        }
        else
        {
            has_car = false;
            car_mu = 0.0;
            car_tau2 = 0.01;
            car_rho = 0.5;
        }
    }
};

class Model
{
public:
    std::string dobs = "nbinom";
    std::string flink = "identity";
    std::string fgain = "exponential";

    unsigned int nS; // number of locations for spatio-temporal model
    arma::vec rho;
    SpatialStructure spatial;
    Param intercept_a, coef_self_b, coef_cross_c;

    Model()
    {
        nS = 1;
        spatial = SpatialStructure(nS);

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
            if (spatial_settings.containsElementNamed("nlocation"))
            {
                nS = Rcpp::as<unsigned int>(spatial_settings["nlocation"]);
            }
            else
            {
                throw std::invalid_argument("Model::Model - number of locations 'nlocation' is missing.");
            } // end of initialize number of locations nS

            if (spatial_settings.containsElementNamed("neighborhood_matrix"))
            {
                arma::mat V_r = Rcpp::as<arma::mat>(spatial_settings["neighborhood_matrix"]);
                arma::mat V = arma::symmatu(V_r); // ensure symmetry
                V.diag().zeros();                 // zero diagonal

                if (V.n_rows != nS || V.n_cols != nS)
                {
                    throw std::invalid_argument("Model::Model - dimension of neighborhood matrix is not consistent with 'nlocation'.");
                }
                spatial = SpatialStructure(V);
            }
            else
            {
                throw std::invalid_argument("Model::Model - neighborhood matrix 'neighborhood_matrix' is missing.");
            } // end of initialize neighborhood matrix V
        }
        else
        {
            nS = 1;
            spatial = SpatialStructure(nS);
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
            intercept_a = Param(opts);
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
            coef_self_b = Param(opts);
        }
        else
        {
            coef_self_b.intercept = 0.0; // constant
        }

        if (settings.containsElementNamed("coef_cross"))
        {
            Rcpp::List opts = settings["coef_cross"];
            coef_cross_c = Param(opts);
        }
        else
        {
            coef_cross_c.intercept = 0.0; // constant
        }
    } // end of Model(const Rcpp::List &settings)

    void simulate(
        arma::mat &Y,
        arma::mat &Lambda,
        arma::mat &log_a,
        arma::mat &wt,
        const unsigned int &ntime)
    {
        Y.set_size(nS, ntime + 1);
        Y.zeros();
        Lambda = Y;

        wt.set_size(nS, ntime + 1);
        wt.randn();
        wt *= std::sqrt(intercept_a.sigma2);
        log_a = arma::cumsum(wt, 1);

        const double coef_self = std::exp(std::min(coef_self_b.intercept, UPBND));
        const double coef_cross = std::exp(std::min(coef_cross_c.intercept, UPBND));
        const double baseline = intercept_a.has_intercept ? intercept_a.intercept : 0.0;

        for (unsigned int t = 1; t <= ntime; t++)
        {
            // State evolution
            // Construct state vectors at time t for all locations
            for (unsigned int s = 0; s < nS; s++)
            {
                log_a.at(s, t) = intercept_a.has_temporal ? log_a.at(s, t) : 0.0;
                Lambda.at(s, t) = std::exp(std::min(baseline + log_a.at(s, t), UPBND));
                Lambda.at(s, t) += coef_self * Y.at(s, t - 1);
                Lambda.at(s, t) += coef_cross * arma::dot(
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
        arma::cube &log_a, // nS x (nT + 1) x nsample
        const Rcpp::List &output, 
        const arma::mat &Y, 
        const unsigned int &nrep = 1
    )
    {
        std::map<std::string, AVAIL::Dist> obs_list = ObsDist::obs_list;

        arma::cube wt_stored; // nS x (nT + 1) x nsample
        if (output.containsElementNamed("wt_samples"))
        {
            wt_stored = Rcpp::as<arma::cube>(output["wt_samples"]);
        }
        else if (output.containsElementNamed("wt"))
        {
            wt_stored = Rcpp::as<arma::cube>(output["wt"]);
        }
        else
        {
            throw std::invalid_argument("Model::sample_posterior_predictive_y - 'wt_samples' or 'wt' is missing in the output.");
        }

        const unsigned int nsample = wt_stored.n_slices;
        const unsigned int ntime = wt_stored.n_cols - 1;

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

        arma::vec intercept_intercept_stored;
        if (output.containsElementNamed("intercept_intercept"))
        {
            intercept_intercept_stored = Rcpp::as<arma::vec>(output["intercept_intercept"]);
        }
        else
        {
            intercept_intercept_stored = arma::vec(nsample, arma::fill::value(intercept_a.intercept));
        }

        arma::vec intercept_sigma2_stored;
        if (output.containsElementNamed("intercept_sigma2"))
        {
            intercept_sigma2_stored = Rcpp::as<arma::vec>(output["intercept_sigma2"]);
        }
        else
        {
            intercept_sigma2_stored = arma::vec(nsample, arma::fill::value(intercept_a.sigma2));
        }

        arma::vec coef_self_intercept_stored;
        if (output.containsElementNamed("coef_self_intercept"))
        {
            coef_self_intercept_stored = Rcpp::as<arma::vec>(output["coef_self_intercept"]);
        }
        else
        {
            coef_self_intercept_stored = arma::vec(nsample, arma::fill::value(coef_self_b.intercept));
        }

        arma::vec coef_cross_intercept_stored;
        if (output.containsElementNamed("coef_cross_intercept"))
        {
            coef_cross_intercept_stored = Rcpp::as<arma::vec>(output["coef_cross_intercept"]);
        }
        else
        {
            coef_cross_intercept_stored = arma::vec(nsample, arma::fill::value(coef_cross_c.intercept));
        }

        Y_pred.set_size(nS, ntime + 1, nsample * nrep);
        Y_pred.zeros();
        Y_residual.set_size(nS, ntime + 1, nsample);
        Y_residual.zeros();
        log_a.set_size(nS, ntime + 1, nsample);
        log_a.zeros();

        arma::vec chi_sqr(nsample, arma::fill::zeros);
        Progress p(nsample, true);
        #ifdef DGTF_USE_OPENMP
        #pragma omp parallel for num_threads(NUM_THREADS) schedule(static)
        #endif
        for (unsigned int i = 0; i < nsample; i++)
        {
            Model model_i = *this;
            model_i.intercept_a.intercept = intercept_intercept_stored.at(i);
            model_i.intercept_a.sigma2 = intercept_sigma2_stored.at(i);
            model_i.coef_self_b.intercept = coef_self_intercept_stored.at(i);
            model_i.coef_cross_c.intercept = coef_cross_intercept_stored.at(i);
            model_i.rho = rho_stored.col(i);

            const double coef_self = std::exp(std::min(model_i.coef_self_b.intercept, UPBND));
            const double coef_cross = std::exp(std::min(model_i.coef_cross_c.intercept, UPBND));

            arma::mat Lambda(nS, ntime + 1, arma::fill::zeros);
            log_a.slice(i) = arma::cumsum(wt_stored.slice(i), 1); // nS x (nT + 1)

            arma::mat ytmp(nS, ntime, arma::fill::zeros);
            for (unsigned int t = 1; t <= ntime; t++)
            {
                for (unsigned int s = 0; s < nS; s++)
                {
                    double eta = std::exp(std::min(intercept_a.intercept + log_a.at(s, t, i), UPBND));
                    eta += coef_self * Y.at(s, t - 1);
                    eta += coef_cross * arma::dot(
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
        const arma::vec &log_a // (nT + 1) x 1
    ) const
    {
        arma::vec y = Y.row(s).t();
        arma::vec neighbor_weights = spatial.W.row(s).t(); // nS x 1

        arma::vec baseline = arma::exp(intercept_a.intercept + log_a);
        double coef_self = std::exp(std::min(coef_self_b.intercept, UPBND));
        double coef_cross = std::exp(std::min(coef_cross_c.intercept, UPBND));

        arma::vec Lambda(log_a.n_elem, arma::fill::zeros);
        for (unsigned int t = 1; t < y.n_elem; t++)
        {
            Lambda.at(t) = baseline.at(t) + coef_self * y.at(t - 1);
            Lambda.at(t) += coef_cross * arma::dot(neighbor_weights, Y.col(t - 1));
        }

        return Lambda; // (nT + 1) x 1
    } // end of compute_intensity_iterative()


    double dloglik_deta(const double &eta, const double &y, const double &obs_par2)
    {
        return nbinomm::dlogp_dlambda(eta, y, obs_par2);
    } // end of dloglike_deta()

    arma::mat dloglik_deta(
        const arma::mat &Y, // nS x (nT + 1)
        const arma::mat &log_a // nS x (nT + 1)
    )
    {
        const unsigned int nT = Y.n_cols - 1;
        const double coef_self = std::exp(std::min(coef_self_b.intercept, UPBND));
        const double coef_cross = std::exp(std::min(coef_cross_c.intercept, UPBND));

        arma::mat dll_deta(Y.n_rows, Y.n_cols, arma::fill::zeros);
        for (unsigned int s = 0; s < Y.n_rows; s++)
        {
            const arma::vec ys = Y.row(s).t();
            const arma::vec intercept = arma::exp(intercept_a.intercept + log_a.row(s).t());
            const arma::vec neighbor_weights = spatial.W.row(s).t();
            const arma::vec cross_region_effects = coef_cross * Y.head_cols(nT).t() * neighbor_weights; // nT x 1

            for (unsigned int t = 1; t < Y.n_cols; t++)
            {
                double eta = intercept.at(t) + coef_self * ys.at(t - 1);
                eta += cross_region_effects.at(t - 1);

                dll_deta.at(s, t) = dloglik_deta(eta, ys.at(t), rho.at(s));
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


    double dloglik_dcoef_self(
        const arma::mat &Y,
        const arma::mat &dll_deta
    )
    {
        double deriv = 0.0;
        double coef_self = std::exp(std::min(coef_self_b.intercept, UPBND));
        for (unsigned int s = 0; s < nS; s++)
        {
            for (unsigned int t = 1; t < Y.n_cols; t++)
            {
                deriv += dll_deta.at(s, t) * Y.at(s, t - 1) * coef_self;
            }
        }

        return deriv;
    } // end of dloglik_dcoef_self()


    double dloglik_dcoef_cross(
        const arma::mat &Y, // nS x (nT + 1)
        const arma::mat &dll_deta // nS x (nT + 1)
    )
    {
        double deriv = 0.0;
        double coef_cross = std::exp(std::min(coef_cross_c.intercept, UPBND));
        for (unsigned int s = 0; s < nS; s++)
        {
            for (unsigned int t = 1; t < Y.n_cols; t++)
            {
                double deta_dcoef_cross = arma::dot(spatial.W.row(s).t(), Y.col(t - 1));
                // double dbeta_dlogbeta = beta;
                deriv += dll_deta.at(s, t) * deta_dcoef_cross * coef_cross;
            }
        }

        return deriv;
    } // end of dloglik_dcoef_cross()


    double dloglik_dlogrho(const unsigned int &s, const arma::vec &y, const arma::vec &lambda)
    {
        double dll_dlogrho = 0.0;
        for (unsigned int t = 1; t < y.n_elem; t++)
        {
            dll_dlogrho += nbinomm::dlogp_dpar2(y.at(t), lambda.at(t), rho.at(s), true);
        }

        return dll_dlogrho;
    } // end of dloglik_dlogrho()


    // double dloglik_dlogsig_a(const unsigned int &s, const arma::vec &y, const arma::vec &wt)
    // {
    //     double dll_dlogW = 0.0;
    //     double ntime = 0.0;
    //     double res2 = 0.0;
    //     for (unsigned int t = 1; t < y.n_elem; t++)
    //     {
    //         ntime += 1.0;
    //         res2 += wt.at(t) * wt.at(t);
    //     }

    //     return - 0.5 * ntime + 0.5 * res2 / std::max(W.at(s), EPS);
    // } // end of dloglik_dlogW()
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
    } // end of dloglik_dlogW()


    arma::vec get_global_params_unconstrained(const std::vector<std::string> &global_params)
    {
        std::map<std::string, AVAIL::Param> static_param_list = AVAIL::static_param_list;
        arma::vec unconstrained_params(global_params.size(), arma::fill::zeros);
        for (unsigned int i = 0; i < global_params.size(); i++)
        {
            switch (static_param_list[tolower(global_params[i])])
            {
            case AVAIL::Param::ar_intercept_intercept:
            {
                unconstrained_params.at(i) = intercept_a.intercept;
                break;
            }
            case AVAIL::Param::ar_intercept_sigma2:
            {
                unconstrained_params.at(i) = std::log(std::max(intercept_a.sigma2, EPS));
                break;
            }
            case AVAIL::Param::ar_coef_self_intercept:
            {
                unconstrained_params.at(i) = coef_self_b.intercept;
                break;
            }
            case AVAIL::Param::ar_coef_cross_intercept:
            {
                unconstrained_params.at(i) = coef_cross_c.intercept;
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
            case AVAIL::Param::ar_intercept_intercept:
            {
                intercept_a.intercept = std::min(unconstrained_params.at(i), UPBND);
                break;
            }
            case AVAIL::Param::ar_intercept_sigma2:
            {
                intercept_a.sigma2 = std::exp(std::min(unconstrained_params.at(i), UPBND));
                break;
            }
            case AVAIL::Param::ar_coef_self_intercept:
            {
                coef_self_b.intercept = std::min(unconstrained_params.at(i), UPBND);
                break;
            }
            case AVAIL::Param::ar_coef_cross_intercept:
            {
                coef_cross_c.intercept = std::min(unconstrained_params.at(i), UPBND);
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
        const Prior &intercept_intercept_prior,
        const Prior &intercept_sigma2_prior,
        const Prior &coef_self_intercept_prior,
        const Prior &coef_cross_intercept_prior)
    {
        std::map<std::string, AVAIL::Param> static_param_list = AVAIL::static_param_list;
        arma::vec grad(global_params.size(), arma::fill::zeros);

        arma::mat log_a = arma::cumsum(wt, 1);
        arma::mat dll_deta = dloglik_deta(Y, log_a);
        bool dlag_calculated = false;
        arma::vec dlag_grad;
        for (unsigned int i = 0; i < global_params.size(); i++)
        {
            switch (static_param_list[tolower(global_params[i])])
            {
            case AVAIL::Param::ar_intercept_intercept:
            {
                grad.at(i) = dloglik_dintercept_a(Y, dll_deta, log_a);
                grad.at(i) += Prior::dlogprior_dpar(
                    intercept_a.intercept, intercept_intercept_prior, false
                );
                break;
            }
            case AVAIL::Param::ar_intercept_sigma2:
            {
                grad.at(i) = dloglik_dlogsig_a(Y, wt);
                grad.at(i) += Prior::dlogprior_dpar(
                    intercept_a.sigma2, intercept_sigma2_prior, true
                );
                break;
            }
            case AVAIL::Param::ar_coef_self_intercept:
            {
                grad.at(i) = dloglik_dcoef_self(Y, dll_deta);
                grad.at(i) += Prior::dlogprior_dpar(
                    coef_self_b.intercept, coef_self_intercept_prior, false
                );
                break;
            }
            case AVAIL::Param::ar_coef_cross_intercept:
            {
                grad.at(i) = dloglik_dcoef_cross(Y, dll_deta);
                grad.at(i) += Prior::dlogprior_dpar(
                    coef_cross_c.intercept, 
                    coef_cross_intercept_prior, false
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