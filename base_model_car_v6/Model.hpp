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
#include "../core/LagDist.hpp"
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
    std::string fgain = "softplus";

    bool has_intercept_a = true;
    bool has_coef_self_b = true;

    unsigned int nS; // number of locations for spatio-temporal model
    unsigned int nP;
    arma::vec rho; // nS x 1
    BYM2 spatial;
    Param intercept_a, coef_self_b;

    LagDist dlag;

    Model()
    {
        nS = 1;
        spatial = BYM2(nS);

        rho.set_size(nS);
        rho.fill(30.0); // default dispersion for NegBin

        intercept_a.has_intercept = true; // constant term
        intercept_a.has_temporal = false; // no temporal RW
        intercept_a.has_spatial = false; // no spatial effect
        intercept_a.sigma2 = 0.01;

        coef_self_b.intercept = 0.0; // constant
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

        dlag.init("lognorm", LN_MU, LN_SD2, true); // default lag distribution
        if (settings.containsElementNamed("lag"))
        {
            Rcpp::List lag_settings = settings["lag"];
            std::string dist_name = dlag.name;
            if (lag_settings.containsElementNamed("dist_name"))
            {
                dist_name = Rcpp::as<std::string>(lag_settings["dist_name"]);
            }

            double par1 = dlag.par1;
            double par2 = dlag.par2;
            if (lag_settings.containsElementNamed("dist_param"))
            {
                arma::vec dist_param = Rcpp::as<arma::vec>(lag_settings["dist_param"]);
                if (dist_param.n_elem != 2)
                {
                    throw std::invalid_argument("Model::Model - 'dist_param' should have two elements: (par1, par2).");
                }
                par1 = dist_param.at(0);
                par2 = dist_param.at(1);
            }

            bool truncated = dlag.truncated;
            if (lag_settings.containsElementNamed("truncated"))
            {
                truncated = Rcpp::as<bool>(lag_settings["truncated"]);
            }

            dlag.init(dist_name, par1, par2, truncated);
        }
        nP = LagDist::get_nlag(dlag);
        dlag.Fphi = LagDist::get_Fphi(dlag);

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
            has_coef_self_b = true;
            Rcpp::List opts = settings["coef_self"];
            coef_self_b = Param(opts, spatial.V);
        }
        else
        {
            has_coef_self_b = false;
            coef_self_b.intercept = 0.0; // constant
        }
    } // end of Model(const Rcpp::List &settings)

    void simulate(
        arma::mat &Y, // nS x (nT + 1)
        arma::mat &Lambda, // nS x (nT + 1)
        arma::vec &psi1_spatial, // nS x 1
        arma::vec &psi2_temporal, // (nT + 1) x 1
        arma::vec &wt2_temporal, // (nT + 1) x 1
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
            wt2_temporal.set_size(ntime + 1);
            wt2_temporal.randn();
            wt2_temporal *= std::sqrt(coef_self_b.sigma2);
            psi2_temporal = arma::cumsum(wt2_temporal);
        }

        const double mu = std::exp(std::min(intercept_a.intercept, UPBND));

        for (unsigned int s = 0; s < nS; s++)
        {
            arma::vec psi_s = psi1_spatial.at(s) + psi2_temporal;
            arma::vec hpsi_s = GainFunc::psi2hpsi<arma::vec>(psi_s, fgain);
            arma::vec ys(ntime + 1, arma::fill::zeros);
            for (unsigned int t = 1; t <= ntime; t++)
            {
                double ft = TransFunc::transfer_sliding(
                    t, dlag.nL, ys, dlag.Fphi, hpsi_s
                );
                Lambda.at(s, t) = mu + ft;
                ys.at(t) = ObsDist::sample(Lambda.at(s, t), rho.at(s), dobs);
            } // end of loop over time t

            Y.row(s) = ys.t();
        } // end of loop over locations s

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

        arma::vec lag_par1_stored;
        if (output.containsElementNamed("lag_par1"))
        {
            lag_par1_stored = Rcpp::as<arma::vec>(output["lag_par1"]);
        }
        else
        {
            lag_par1_stored = arma::vec(nsample, arma::fill::value(dlag.par1));
        }

        arma::vec lag_par2_stored;
        if (output.containsElementNamed("lag_par2"))
        {
            lag_par2_stored = Rcpp::as<arma::vec>(output["lag_par2"]);
        }
        else
        {
            lag_par2_stored = arma::vec(nsample, arma::fill::value(dlag.par2));
        }

        arma::vec a_intercept_stored(nsample, arma::fill::value(intercept_a.intercept));
        if (output.containsElementNamed("intercept_a"))
        {
            Rcpp::List intercept_a_opts = output["intercept_a"];
            if (intercept_a_opts.containsElementNamed("intercept"))
            {
                a_intercept_stored = Rcpp::as<arma::vec>(intercept_a_opts["intercept"]);
            }
        } // end of if output contains "intercept_a"

        arma::vec coef_self_sigma2_stored(nsample, arma::fill::zeros);
        arma::mat coef_self_psi2_temporal_stored(ntime + 1, nsample, arma::fill::zeros);
        arma::mat coef_self_psi1_spatial_stored(nS, nsample, arma::fill::zeros);
        arma::vec coef_self_car_mu_stored(nsample, arma::fill::zeros);
        arma::vec coef_self_car_tau_stored(nsample, arma::fill::zeros);
        arma::vec coef_self_car_phi_stored(nsample, arma::fill::zeros);
        if (output.containsElementNamed("coef_self_b"))
        {
            Rcpp::List coef_self_b_opts = output["coef_self_b"];
            Rcpp::List coef_self_b_mcmc_opts = mcmc_opts["coef_self"];
            if (coef_self_b_opts.containsElementNamed("sigma2"))
            {
                coef_self_sigma2_stored = Rcpp::as<arma::vec>(coef_self_b_opts["sigma2"]);
            }
            else
            {
                coef_self_sigma2_stored = arma::vec(nsample, arma::fill::value(coef_self_b.sigma2));
            } // end of if coef_self_b_opts contains "sigma2"

            if (coef_self_b_opts.containsElementNamed("psi2_temporal"))
            {
                coef_self_psi2_temporal_stored = Rcpp::as<arma::mat>(coef_self_b_opts["psi2_temporal"]);
            }
            else if (coef_self_b_mcmc_opts.containsElementNamed("temporal"))
            {
                Rcpp::List coef_self_b_temporal_opts = coef_self_b_mcmc_opts["temporal"];
                if (coef_self_b_temporal_opts.containsElementNamed("init"))
                {
                    arma::vec coef_self_psi2_temporal_init = Rcpp::as<arma::vec>(coef_self_b_temporal_opts["init"]);
                    coef_self_psi2_temporal_stored.each_col() = coef_self_psi2_temporal_init;
                }
            } // end of if coef_self_b_opts contains "temporal"

            if (coef_self_b_opts.containsElementNamed("spatial"))
            {
                Rcpp::List coef_self_b_spatial_opts = coef_self_b_opts["spatial"];
                if (coef_self_b_spatial_opts.containsElementNamed("psi1_spatial"))
                {
                    coef_self_psi1_spatial_stored = Rcpp::as<arma::mat>(coef_self_b_spatial_opts["psi1_spatial"]);
                }
                else if (coef_self_b_mcmc_opts.containsElementNamed("spatial"))
                {
                    Rcpp::List coef_self_b_spatial_mcmc_opts = coef_self_b_mcmc_opts["spatial"];
                    if (coef_self_b_spatial_mcmc_opts.containsElementNamed("init"))
                    {
                        arma::vec coef_self_psi1_spatial_init = Rcpp::as<arma::vec>(coef_self_b_spatial_mcmc_opts["init"]);
                        coef_self_psi1_spatial_stored.each_col() = coef_self_psi1_spatial_init;
                    }
                }

                if (coef_self_b_spatial_opts.containsElementNamed("mu"))
                {
                    coef_self_car_mu_stored = Rcpp::as<arma::vec>(coef_self_b_spatial_opts["mu"]);
                }
                else
                {
                    coef_self_car_mu_stored = arma::vec(nsample, arma::fill::value(coef_self_b.spatial.mu));
                }

                if (coef_self_b_spatial_opts.containsElementNamed("tau"))
                {
                    coef_self_car_tau_stored = Rcpp::as<arma::vec>(coef_self_b_spatial_opts["tau"]);
                }
                else
                {
                    coef_self_car_tau_stored = arma::vec(nsample, arma::fill::value(coef_self_b.spatial.tau_b));
                }

                if (coef_self_b_spatial_opts.containsElementNamed("phi"))
                {
                    coef_self_car_phi_stored = Rcpp::as<arma::vec>(coef_self_b_spatial_opts["phi"]);
                }
                else
                {
                    coef_self_car_phi_stored = arma::vec(nsample, arma::fill::value(coef_self_b.spatial.phi));
                }
            } // end of if coef_self_b_opts contains "spatial"
        } // end of if output contains "coef_self_b"


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
            model_i.coef_self_b.sigma2 = coef_self_sigma2_stored.at(i);
            arma::vec psi2_temporal = coef_self_psi2_temporal_stored.col(i);
            arma::vec psi1_spatial = coef_self_psi1_spatial_stored.col(i);
            model_i.coef_self_b.spatial = BYM2(
                spatial.V, 
                coef_self_car_mu_stored.at(i), 
                coef_self_car_tau_stored.at(i), 
                coef_self_car_phi_stored.at(i)
            );

            model_i.dlag.par1 = lag_par1_stored.at(i);
            model_i.dlag.par2 = lag_par2_stored.at(i);
            model_i.dlag.nL = LagDist::get_nlag(model_i.dlag);
            model_i.dlag.Fphi = LagDist::get_Fphi(model_i.dlag);
            model_i.rho = rho_stored.col(i);

            const double mu = std::exp(std::min(model_i.intercept_a.intercept, UPBND));
            arma::mat ytmp(nS, ntime, arma::fill::zeros);
            for (unsigned int s = 0; s < nS; s++)
            {
                arma::vec psi_s = psi1_spatial.at(s) + psi2_temporal;
                arma::vec hpsi_s = GainFunc::psi2hpsi<arma::vec>(psi_s, fgain);
                for (unsigned int t = 1; t <= ntime; t++)
                {
                    double ft = TransFunc::transfer_sliding(
                        t, model_i.dlag.nL, Y.row(s).t(), model_i.dlag.Fphi, hpsi_s);
                    double lambda = mu + ft;

                    double mean, var;
                    switch (obs_list[model_i.dobs])
                    {
                    case AVAIL::Dist::nbinomm:
                    {
                        mean = lambda;
                        var = std::abs(nbinomm::var(lambda, model_i.rho.at(s)));
                        break;
                    }
                    case AVAIL::Dist::nbinomp:
                    {
                        mean = nbinom::mean(lambda, model_i.rho.at(s));
                        var = nbinom::var(lambda, model_i.rho.at(s));
                        break;
                    }
                    case AVAIL::Dist::poisson:
                    {
                        mean = lambda;
                        var = lambda;
                        break;
                    }
                    default:
                    {
                        throw std::invalid_argument("Unknown observation distribution.");
                        break;
                    }
                    } // end of switch by obs_list

                    Y_residual.at(s, t, i) = lambda - Y.at(s, t);
                    double yres2 = 2. * std::log(std::abs(Y_residual.at(s, t, i)) + EPS);
                    double yvar = std::log(var + EPS);
                    ytmp.at(s, t - 1) = std::exp(yres2 - yvar);

                    for (unsigned int r = 0; r < nrep; r++)
                    {
                        Y_pred.at(s, t, i * nrep + r) = ObsDist::sample(
                            lambda, model_i.rho.at(s), model_i.dobs
                        );
                    } // end of for r in [0, nrep]
                } // end of loop over time t

            } // end of loop over locations s

            chi_sqr.at(i) = arma::mean(arma::mean(ytmp));
            p.increment();
        } // end of for i in [0, nsample]

        return arma::mean(chi_sqr);
    } // end of sample_posterior_predictive_y()

    arma::vec compute_intensity_iterative(
        const unsigned int &s,
        const arma::mat &Y, // nS x (nT + 1)
        const arma::vec &psi1_spatial, // nS x 1
        const arma::vec &psi2_temporal // (nT + 1) x 1
    ) const
    {
        arma::vec Lambda(Y.n_cols, arma::fill::zeros);
        const double mu = std::exp(std::min(intercept_a.intercept, UPBND));
        const arma::vec ys = Y.row(s).t();
        const arma::vec psi_s = psi1_spatial.at(s) + psi2_temporal;
        const arma::vec hpsi_s = GainFunc::psi2hpsi<arma::vec>(psi_s, fgain);
        for (unsigned int t = 1; t < Y.n_cols; t++)
        {
            double ft = TransFunc::transfer_sliding(
                t, dlag.nL, ys, dlag.Fphi, hpsi_s
            );
            Lambda.at(t) = mu + ft;
        }

        return Lambda; // (nT + 1) x 1
    } // end of compute_intensity_iterative()


    void compute_intensity_iterative(
        arma::mat &Lambda, // nS x (nT + 1)
        const unsigned int &t_start,
        const arma::mat &Y, // nS x (nT + 1)
        const arma::vec &psi1_spatial, // nS x 1
        const arma::vec &psi2_temporal // (nT + 1) x 1
    )
    {
        const double mu = std::exp(std::min(intercept_a.intercept, UPBND));
        for (unsigned int s = 0; s < Y.n_rows; s++)
        {
            const arma::vec ys = Y.row(s).t();
            const arma::vec psi_s = psi1_spatial.at(s) + psi2_temporal;
            const arma::vec hpsi_s = GainFunc::psi2hpsi<arma::vec>(psi_s, fgain);
            for (unsigned int t = t_start; t < Y.n_cols; t++)
            {
                double ft = TransFunc::transfer_sliding(
                    t, dlag.nL, ys, dlag.Fphi, hpsi_s
                );
                Lambda.at(s, t) = mu + ft;
            } // end of loop over time t
        } // end of loop over locations s
        return; // (nT + 1) x 1
    } // end of compute_intensity_iterative()


    double dloglik_deta(const double &eta, const double &y, const double &obs_par2) const
    {
        return nbinomm::dlogp_dlambda(eta, y, obs_par2);
    } // end of dloglike_deta()

    arma::mat dloglik_deta(
        const arma::mat &Y, // nS x (nT + 1)
        const arma::vec &psi1_spatial, // nS x 1
        const arma::vec &psi2_temporal // (nT + 1) x 1
    ) const
    {
        const unsigned int nT = Y.n_cols - 1;
        const double mu = std::exp(std::min(intercept_a.intercept, UPBND));
        arma::mat dll_deta(Y.n_rows, Y.n_cols, arma::fill::zeros);
        for (unsigned int s = 0; s < Y.n_rows; s++)
        {
            const arma::vec ys = Y.row(s).t();
            const arma::vec psi_s = psi1_spatial.at(s) + psi2_temporal;
            const arma::vec hpsi_s = GainFunc::psi2hpsi<arma::vec>(psi_s, fgain);
            for (unsigned int t = 1; t < Y.n_cols; t++)
            {
                double ft = TransFunc::transfer_sliding(
                    t, dlag.nL, ys, dlag.Fphi, hpsi_s
                );

                double eta = mu + ft;
                dll_deta.at(s, t) = dloglik_deta(eta, Y.at(s, t), rho.at(s));
            }
        }

        return dll_deta;
    } // end of dloglik_deta()


    double dloglik_dintercept_a(
        const arma::mat &Y,
        const arma::mat &dll_deta
    )
    {
        double deriv = 0.0;
        double intercept = std::exp(std::min(intercept_a.intercept, UPBND));
        for (unsigned int s = 0; s < nS; s++)
        {
            for (unsigned int t = 1; t < Y.n_cols; t++)
            {
                deriv += dll_deta.at(s, t) * intercept;
            }
        }
        return deriv;
    }


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
            const arma::vec ys = Y.row(s).t();
            const arma::vec psi_s = psi1_spatial.at(s) + psi2_temporal;
            const arma::vec dhpsi_s = GainFunc::psi2dhpsi<arma::vec>(psi_s, fgain);
            for (unsigned int t = 1; t < Y.n_cols; t++)
            {
                double deta_dspatial = TransFunc::transfer_sliding(
                    t, dlag.nL, ys, dlag.Fphi, dhpsi_s
                );
                grad.at(s) += dll_deta.at(s, t) * deta_dspatial;
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


    arma::vec dloglik_dlag(
        const arma::mat &Y, // nS x (nT + 1)
        const arma::mat &dll_deta, // nS x (nT + 1)
        const arma::vec &psi1_spatial,
        const arma::vec &psi2_temporal
    )
    {
        arma::mat dFphi_grad = LagDist::get_Fphi_grad(dlag.nL, dlag.name, dlag.par1, dlag.par2);
        arma::vec grad(2, arma::fill::zeros); // gradient w.r.t. par1 and par2
        for (unsigned int s = 0; s < nS; s++)
        {
            const arma::vec ys = Y.row(s).t();
            const arma::vec psi_s = psi1_spatial.at(s) + psi2_temporal;
            const arma::vec hpsi_s = GainFunc::psi2hpsi<arma::vec>(psi_s, fgain);
            for (unsigned int t = 1; t < ys.n_elem; t++)
            {
                double deta_dpar1 = TransFunc::transfer_sliding(t, dlag.nL, ys, dFphi_grad.col(0), hpsi_s);
                double deta_dpar2 = TransFunc::transfer_sliding(t, dlag.nL, ys, dFphi_grad.col(1), hpsi_s);

                grad.at(0) += dll_deta.at(s, t) * deta_dpar1; // dloglik_d[lag_mu]
                grad.at(1) += dll_deta.at(s, t) * deta_dpar2; // dloglik_d[log(lag_sd2)]

            } // end of for t
        } // end of for s

        return grad;
    } // end of dloglik_dlag()


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
            case AVAIL::Param::lag_par1:
            {
                unconstrained_params.at(i) = dlag.par1;
                break;
            }
            case AVAIL::Param::lag_par2:
            {
                unconstrained_params.at(i) = std::log(std::max(dlag.par2, EPS));
                break;
            }
            case AVAIL::Param::ar_a_intercept:
            {
                unconstrained_params.at(i) = intercept_a.intercept;
                break;
            }
            case AVAIL::Param::ar_coef_self_sigma2:
            {
                unconstrained_params.at(i) = std::log(std::max(coef_self_b.sigma2, EPS));
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
            case AVAIL::Param::lag_par1:
            {
                dlag.par1 = unconstrained_params.at(i);
                lag_updated = true;
                break;
            }
            case AVAIL::Param::lag_par2:
            {
                dlag.par2 = std::exp(std::min(unconstrained_params.at(i), UPBND));
                lag_updated = true;
                break;
            }
            case AVAIL::Param::ar_a_intercept:
            {
                intercept_a.intercept = std::min(unconstrained_params.at(i), UPBND);
                break;
            }
            case AVAIL::Param::ar_coef_self_sigma2:
            {
                coef_self_b.sigma2 = std::exp(std::min(unconstrained_params.at(i), UPBND));
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

        if (lag_updated)
        {
            // dlag.update_nlag();
            dlag.update_Fphi();
            // if (dlag.truncated)
            // {
            //     nP = dlag.nL;
            // }
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
        const arma::vec &psi1_spatial,
        const arma::vec &wt2_temporal,
        const Prior &lag_par1_prior,
        const Prior &lag_par2_prior,
        const Prior &a_intercept_prior,
        const Prior &coef_self_sigma2_prior
    )
    {
        std::map<std::string, AVAIL::Param> static_param_list = AVAIL::static_param_list;
        const arma::vec psi2_temporal = arma::cumsum(wt2_temporal);

        arma::vec grad(global_params.size(), arma::fill::zeros);
        arma::mat dll_deta = dloglik_deta(Y, psi1_spatial, psi2_temporal);
        bool dlag_calculated = false;
        arma::vec dlag_grad;
        for (unsigned int i = 0; i < global_params.size(); i++)
        {
            switch (static_param_list[tolower(global_params[i])])
            {
            case AVAIL::Param::lag_par1:
            {
                if (!dlag_calculated)
                {
                    dlag_grad = dloglik_dlag(Y, dll_deta, psi1_spatial, psi2_temporal);
                    dlag_calculated = true;
                }
                grad.at(i) = dlag_grad.at(0) + Prior::dlogprior_dpar(
                    dlag.par1, lag_par1_prior, false
                );
                break;
            }
            case AVAIL::Param::lag_par2:
            {
                if (!dlag_calculated)
                {
                    dlag_grad = dloglik_dlag(Y, dll_deta, psi1_spatial, psi2_temporal);
                    dlag_calculated = true;
                }
                grad.at(i) = dlag_grad.at(1) + Prior::dlogprior_dpar(
                    dlag.par2, lag_par2_prior, true
                );
                break;
            }
            case AVAIL::Param::ar_a_intercept:
            {
                grad.at(i) = dloglik_dintercept_a(Y, dll_deta);
                grad.at(i) += Prior::dlogprior_dpar(
                    intercept_a.intercept, a_intercept_prior, false
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