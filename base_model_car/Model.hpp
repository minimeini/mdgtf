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

class Model
{
public:
    unsigned int nP;
    unsigned int nS; // number of locations for spatio-temporal model
    arma::vec log_alpha, rho, W;
    double beta;

    SpatialStructure spatial;

    std::string fsys = "shift";
    std::string ftrans = "sliding";
    std::string flink = "identity";
    std::string fgain = "softplus";

    std::string dobs = "nbinom";
    std::string derr = "gaussian";
    LagDist dlag;
    Season seas;

    Model()
    {
        nS = 1;

        // no seasonality and no baseline mean in the latent state by default
        flink = "identity";
        fgain = "softplus";
        fsys = "shift";
        ftrans = "sliding";


        rho = arma::vec(nS, arma::fill::ones);
        log_alpha = arma::vec(nS, arma::fill::ones);
        W = arma::vec(nS, arma::fill::ones);
        W *= 0.01;

        dlag.init("lognorm", LN_MU, LN_SD2, true);
        nP = LagDist::get_nlag(dlag);

        spatial = SpatialStructure(nS);

        beta = 1.0;

        seas.init_default();

        return;
    } // end of Model()

    Model(const Rcpp::List &settings)
    {
        Rcpp::List model_settings = settings["model"];
        Rcpp::List param_settings = settings["param"];
        Rcpp::List spatial_settings = settings["spatial"];

        if (model_settings.containsElementNamed("obs_dist"))
        {
            dobs = tolower(Rcpp::as<std::string>(model_settings["obs_dist"]));
        }
        else
        {
            dobs = "nbinom";
        }

        if (model_settings.containsElementNamed("err_dist"))
        {
            derr = tolower(Rcpp::as<std::string>(model_settings["err_dist"]));
        }
        else
        {
            derr = "gaussian";
        }

        if (model_settings.containsElementNamed("sys_eq"))
        {
            fsys = tolower(Rcpp::as<std::string>(model_settings["sys_eq"]));
        }
        else
        {
            fsys = "shift";
        }

        if (model_settings.containsElementNamed("link_func"))
        {
            flink = tolower(Rcpp::as<std::string>(model_settings["link_func"]));
        }
        else
        {
            flink = "identity";
        }

        if (model_settings.containsElementNamed("gain_func"))
        {
            fgain = tolower(Rcpp::as<std::string>(model_settings["gain_func"]));
        }
        else
        {
            fgain = "softplus";
        }

        std::map<std::string, SysEq::Evolution> sys_list = SysEq::sys_list;
        if (sys_list[fsys] == SysEq::Evolution::nbinom)
        {
            ftrans = "iterative";
            dlag.truncated = false;
            dlag.name = "nbinom";
        }
        else if (sys_list[fsys] == SysEq::Evolution::identity)
        {
            ftrans = "sliding";
            dlag.truncated = true;
            dlag.name = "uniform";
        }
        else
        {
            ftrans = "sliding";
            dlag.truncated = true;
            dlag.name = "lognorm";
            if (model_settings.containsElementNamed("lag_dist"))
            {
                dlag.name = tolower(Rcpp::as<std::string>(model_settings["lag_dist"]));
            }
        }

        seas.init_default();


        if (spatial_settings.containsElementNamed("nlocation"))
        {
            nS = Rcpp::as<unsigned int>(spatial_settings["nlocation"]);
        }
        else
        {
            throw std::invalid_argument("Model::Model - number of locations 'nlocation' is missing.");
        }

        if (spatial_settings.containsElementNamed("neighborhood_matrix"))
        {
            arma::mat V_r = Rcpp::as<arma::mat>(spatial_settings["neighborhood_matrix"]);
            arma::mat V = arma::symmatu(V_r); // ensure symmetry
            V.diag().zeros(); // zero diagonal

            if (V.n_rows != nS || V.n_cols != nS)
            {
                throw std::invalid_argument("Model::Model - dimension of neighborhood matrix is not consistent with 'nlocation'.");
            }
            spatial = SpatialStructure(V);
        }
        else
        {
            throw std::invalid_argument("Model::Model - neighborhood matrix 'neighborhood_matrix' is missing.");
        }

        if (spatial_settings.containsElementNamed("car"))
        {
            arma::vec car_param = Rcpp::as<arma::vec>(spatial_settings["car"]);
            if (car_param.n_elem != 3)
            {
                throw std::invalid_argument("Model::Model - CAR parameters 'car' should be a vector of length 3.");
            }
            double car_mu = car_param[0];
            double car_tau2 = car_param[1];
            double car_rho = car_param[2];

            spatial.update_params(car_mu, car_tau2, car_rho);
        }
        else
        {
            spatial.init_params();
        }

        if (param_settings.containsElementNamed("obs"))
        {
            arma::vec rho_in = Rcpp::as<arma::vec>(param_settings["obs"]);
            if (rho_in.n_elem != nS)
            {
                throw std::invalid_argument("Model::Model - dimension of 'obs' is not consistent with 'nlocation'.");
            }
            rho = rho_in;
        }
        else
        {
            rho = (arma::randu<arma::vec>(nS) + 5.0) * 5.0; // (25.0, 30.0]
        }

        arma::vec lag_param {LN_MU, LN_SD2};
        if (param_settings.containsElementNamed("lag"))
        {
            lag_param = Rcpp::as<arma::vec>(param_settings["lag"]);
            if (lag_param.n_elem != 2)
            {
                throw std::invalid_argument("Model::Model - 'lag' should be a vector of length 2.");
            }
        }

        dlag.init(dlag.name, lag_param[0], lag_param[1], dlag.truncated);
        nP = LagDist::get_nlag(dlag);
        dlag.Fphi = LagDist::get_Fphi(dlag);

        if (param_settings.containsElementNamed("W"))
        {
            arma::vec W_in = Rcpp::as<arma::vec>(param_settings["W"]);
            if (W_in.n_elem != nS)
            {
                throw std::invalid_argument("Model::Model - dimension of 'W' is not consistent with 'nlocation'.");
            }
            W = W_in;
        }
        else
        {
            W = arma::vec(nS, arma::fill::ones);
            W *= 0.01;
        }

        if (param_settings.containsElementNamed("beta"))
        {
            beta = Rcpp::as<double>(param_settings["beta"]);
        }
        else
        {
            beta = 1.0;
        }

        if (param_settings.containsElementNamed("spatial_effect"))
        {
            arma::vec log_alpha_in = Rcpp::as<arma::vec>(param_settings["spatial_effect"]);
            if (log_alpha_in.n_elem != nS)
            {
                throw std::invalid_argument("Model::Model - dimension of 'spatial_effect' is not consistent with 'nlocation'.");
            }
            log_alpha = log_alpha_in;
        }
        else
        {
            log_alpha = spatial.prior_sample_spatial_effects_vec();
        }
    } // end of Model(const Rcpp::List &settings)

    void simulate(
        arma::mat &Y, 
        arma::mat &Lambda, 
        arma::mat &wt,
        arma::mat &Psi, 
        const unsigned int &ntime)
    {
        const arma::vec spatial_effects = arma::exp(log_alpha); // nS x 1
        arma::cube Theta(nP, ntime + 1, nS, arma::fill::randn);
        Y.set_size(nS, ntime + 1);
        Y.zeros();
        Lambda.set_size(nS, ntime + 1);

        wt.set_size(nS, ntime + 1);
        wt.randn();
        wt.each_col() %= arma::sqrt(W);
        Psi = arma::cumsum(wt, 1);

        for (unsigned int t = 1; t <= ntime; t++)
        {
            // State evolution
            // Construct state vectors at time t for all locations
            for (unsigned int s = 0; s < nS; s++)
            {
                Theta.slice(s).col(t) = SysEq::func_gt(
                    fsys, fgain, dlag,
                    Theta.slice(s).col(t - 1),
                    Y.at(s, t - 1), 0, false
                );

                Theta.at(0, t, s) = Psi.at(s, t);

                // Spatial effects
                double eta = spatial_effects.at(s);

                // Self-exciting component
                eta += TransFunc::func_ft(
                    ftrans, fgain, dlag, seas, t,
                    Theta.slice(s).col(t), Y.row(s).t()
                );

                // Cross-regional effect
                eta += beta * arma::dot(
                    spatial.W.row(s).t(), Y.col(t-1)
                );

                Lambda.at(s, t) = LinkFunc::ft2mu(eta, flink);
                Y.at(s, t) = ObsDist::sample(Lambda.at(s, t), rho.at(s), dobs);

            } // end of for s in [0, nS]
        } // end of for t in [1, ntime]

        return;
    } // end of simulate()

    double sample_posterior_predictive_y(
        arma::cube &Y_pred,
        arma::cube &Y_residual,
        arma::cube &hPsi,
        const Rcpp::List &output, 
        const arma::mat &Y, 
        const unsigned int &nrep = 1
    )
    {
        std::map<std::string, AVAIL::Dist> obs_list = ObsDist::obs_list;

        arma::cube wt_stored;
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

        arma::mat log_alpha_stored;
        if (output.containsElementNamed("log_alpha"))
        {
            log_alpha_stored = Rcpp::as<arma::mat>(output["log_alpha"]);
        }
        else if (
            output.containsElementNamed("car_mu") && 
            output.containsElementNamed("car_tau2") && 
            output.containsElementNamed("car_rho")
        )
        {
            arma::vec car_mu = Rcpp::as<arma::vec>(output["car_mu"]);
            arma::vec car_tau2 = Rcpp::as<arma::vec>(output["car_tau2"]);
            arma::vec car_rho = Rcpp::as<arma::vec>(output["car_rho"]);

            log_alpha_stored.set_size(nS, nsample);
            for (unsigned int i = 0; i < nsample; i++)
            {
                SpatialStructure spatial_i(spatial.V);
                spatial_i.update_params(car_mu.at(i), car_tau2.at(i), car_rho.at(i));
                log_alpha_stored.col(i) = spatial_i.prior_sample_spatial_effects_vec();
            }
        }
        else
        {
            log_alpha_stored.set_size(nS, nsample);
            log_alpha_stored.each_col() = log_alpha;
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

        arma::vec beta_stored;
        if (output.containsElementNamed("beta"))
        {
            beta_stored = Rcpp::as<arma::vec>(output["beta"]);
        }
        else
        {
            beta_stored = arma::vec(nsample, arma::fill::value(beta));
        }

        Y_pred.set_size(nS, ntime + 1, nsample * nrep);
        Y_pred.zeros();
        Y_residual.set_size(nS, ntime + 1, nsample);
        Y_residual.zeros();
        hPsi.set_size(nS, ntime + 1, nsample);
        hPsi.zeros();

        arma::vec chi_sqr(nsample, arma::fill::zeros);
        Progress p(nsample, true);
        #ifdef DGTF_USE_OPENMP
        #pragma omp parallel for num_threads(NUM_THREADS) schedule(static)
        #endif
        for (unsigned int i = 0; i < nsample; i++)
        {
            Model model_i = *this;
            model_i.log_alpha = log_alpha_stored.col(i);
            model_i.rho = rho_stored.col(i);
            model_i.dlag.par1 = lag_par1_stored.at(i);
            model_i.dlag.par2 = lag_par2_stored.at(i);
            model_i.beta = beta_stored.at(i);

            arma::mat Lambda(nS, ntime + 1, arma::fill::zeros);
            arma::cube Theta(nP, ntime + 1, nS, arma::fill::randn);
            arma::mat Psi = arma::cumsum(wt_stored.slice(i), 1); // nS x (nT + 1)
            hPsi.slice(i) = GainFunc::psi2hpsi<arma::mat>(Psi, model_i.fgain);
            arma::mat ft(Psi.n_rows, Psi.n_cols, arma::fill::zeros);

            arma::mat ytmp(nS, ntime, arma::fill::zeros);
            for (unsigned int t = 1; t <= ntime; t++)
            {
                for (unsigned int s = 0; s < nS; s++)
                {
                    double eta = std::exp(model_i.log_alpha.at(s));

                    arma::vec hpsi = GainFunc::psi2hpsi<arma::vec>(Psi.row(s).t(), model_i.fgain);
                    ft.at(s, t) = TransFunc::func_ft(t, Y.row(s).t(), ft.row(s).t(), hpsi, model_i.dlag, model_i.ftrans);
                    eta += ft.at(s, t);

                    eta += model_i.beta * arma::dot(
                        model_i.spatial.W.row(s).t(), Y.col(t-1)
                    );

                    Lambda.at(s, t) = LinkFunc::ft2mu(eta, model_i.flink);
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
    }

    arma::vec compute_intensity_iterative(
        const unsigned int &s,
        const arma::mat &Y, // nS x (nT + 1)
        const arma::vec &psi // (nT + 1) x 1
    ) const
    {
        arma::vec y = Y.row(s).t();
        arma::vec neighbor_weights = spatial.W.row(s).t(); // nS x 1

        arma::vec hpsi = GainFunc::psi2hpsi<arma::vec>(psi, fgain);
        arma::vec ft(psi.n_elem, arma::fill::zeros);

        arma::vec Lambda(psi.n_elem, arma::fill::zeros);
        const double spatial_effect = std::exp(log_alpha.at(s));
        for (unsigned int t = 1; t < y.n_elem; t++)
        {
            double eta = spatial_effect;

            // Self-exciting component
            ft.at(t) = TransFunc::func_ft(t, y, ft, hpsi, dlag, ftrans);            
            eta += ft.at(t);

            // Cross-regional effect
            eta += beta * arma::dot(neighbor_weights, Y.col(t - 1));
            Lambda.at(t) = LinkFunc::ft2mu(eta, flink);
        }

        return Lambda; // (nT + 1) x 1
    } // end of compute_intensity_iterative()


    double dloglik_deta(const double &eta, const double &y, const double &obs_par2)
    {
        std::map<std::string, AVAIL::Dist> obs_list = ObsDist::obs_list;

        double lambda;
        double dlam_deta = LinkFunc::dlambda_deta(lambda, eta, flink);

        double dloglik_dlam = 0.;
        switch (obs_list[dobs])
        {
        case AVAIL::Dist::nbinomm:
        {
            dloglik_dlam = nbinomm::dlogp_dlambda(lambda, y, obs_par2);
            break;
        }
        case AVAIL::Dist::nbinomp:
        {
            dloglik_dlam = y / lambda - obs_par2 / std::abs(1. - lambda);
            break;
        }
        case AVAIL::Dist::poisson:
        {
            dloglik_dlam = Poisson::dlogp_dlambda(lambda, y);
            break;
        }
        case AVAIL::Dist::gaussian:
        {
            dloglik_dlam = (y - lambda) / obs_par2;
            break;
        }
        default:
        {
            throw std::invalid_argument("Model::dloglik_deta: observation distribution must be nbinomm or poisson.");
        }
        }

        return dloglik_dlam * dlam_deta;
    } // end of dloglike_deta()

    arma::mat dloglik_deta(
        const arma::mat &Y, // nS x (nT + 1)
        const arma::mat &hPsi // nS x (nT + 1)
    )
    {
        const unsigned int nT = Y.n_cols - 1;
        arma::mat dll_deta(Y.n_rows, Y.n_cols, arma::fill::zeros);
        for (unsigned int s = 0; s < Y.n_rows; s++)
        {
            const arma::vec ys = Y.row(s).t();
            const arma::vec hpsi_s = hPsi.row(s).t();
            const double spatial_effect = std::exp(log_alpha.at(s));
            const arma::vec neighbor_weights = spatial.W.row(s).t();
            const arma::vec cross_region_effects = beta * Y.head_cols(nT).t() * neighbor_weights; // nT x 1

            for (unsigned int t = 1; t < Y.n_cols; t++)
            {
                double eta = spatial_effect;
                eta += TransFunc::transfer_sliding(t, dlag.nL, ys, dlag.Fphi, hpsi_s);
                eta += cross_region_effects.at(t - 1);

                dll_deta.at(s, t) = dloglik_deta(eta, ys.at(t), rho.at(s));
            }
        }

        return dll_deta;
    } // end of dloglik_deta()


    arma::vec dloglik_dlag(
        const arma::mat &Y, // nS x (nT + 1)
        const arma::mat &hPsi, // nS x (nT + 1)
        const arma::mat &dll_deta // nS x (nT + 1)
    )
    {
        arma::mat dFphi_grad = LagDist::get_Fphi_grad(dlag.nL, dlag.name, dlag.par1, dlag.par2);
        arma::vec grad(2, arma::fill::zeros); // gradient w.r.t. par1 and par2
        for (unsigned int s = 0; s < nS; s++)
        {
            const arma::vec ys = Y.row(s).t();
            const arma::vec hpsi_s = hPsi.row(s).t();
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


    double dloglik_dlogbeta(
        const arma::mat &Y, // nS x (nT + 1)
        const arma::mat &dll_deta // nS x (nT + 1)
    )
    {
        double dll_dlogbeta = 0.0;
        for (unsigned int s = 0; s < nS; s++)
        {
            for (unsigned int t = 1; t < Y.n_cols; t++)
            {
                double deta_dbeta = arma::dot(spatial.W.row(s).t(), Y.col(t - 1));
                // double dbeta_dlogbeta = beta;
                dll_dlogbeta += dll_deta.at(s, t) * deta_dbeta * beta;
            }
        }

        return dll_dlogbeta;
    } // end of dloglik_dlogbeta()


    double dloglik_dlogrho(const unsigned int &s, const arma::vec &y, const arma::vec &lambda)
    {
        double dll_dlogrho = 0.0;
        for (unsigned int t = 1; t < y.n_elem; t++)
        {
            dll_dlogrho += nbinomm::dlogp_dpar2(y.at(t), lambda.at(t), rho.at(s), true);
        }

        return dll_dlogrho;
    } // end of dloglik_dlogrho()


    arma::vec dloglik_dlogalpha(const arma::mat &Y, const arma::mat &dll_deta)
    {
        // Gradient of the log likelihood w.r.t. log_alpha
        arma::vec grad(nS, arma::fill::zeros);
        for (unsigned int s = 0; s < nS; s++)
        {
            double spatial_effect = std::exp(log_alpha.at(s));
            for (unsigned int t = 1; t < Y.n_cols; t++)
            {
                grad.at(s) += dll_deta.at(s, t) * spatial_effect;
            }
        }

        // Gradient of the CAR prior w.r.t. log_alpha
        grad -= spatial.car_tau2 * spatial.Q * (log_alpha - spatial.car_mu);

        return grad;
    } // end of dloglik_dlogalpha()


    double dloglik_dlogW(const unsigned int &s, const arma::vec &y, const arma::vec &wt)
    {
        double dll_dlogW = 0.0;
        double ntime = 0.0;
        double res2 = 0.0;
        for (unsigned int t = 1; t < y.n_elem; t++)
        {
            ntime += 1.0;
            res2 += wt.at(t) * wt.at(t);
        }

        return - 0.5 * ntime + 0.5 * res2 / std::max(W.at(s), EPS);
    } // end of dloglik_dlogW()


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
            case AVAIL::Param::cnst_beta:
            {
                unconstrained_params.at(i) = std::log(std::max(beta, EPS));
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
            case AVAIL::Param::cnst_beta:
            {
                beta = std::exp(std::min(unconstrained_params.at(i), UPBND));
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
        const arma::mat &hPsi, // nS x (nT + 1)
        const Prior &lag_par1_prior,
        const Prior &lag_par2_prior,
        const Prior &beta_prior)
    {
        std::map<std::string, AVAIL::Param> static_param_list = AVAIL::static_param_list;
        arma::vec grad(global_params.size(), arma::fill::zeros);
        arma::mat dll_deta = dloglik_deta(Y, hPsi);
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
                    dlag_grad = dloglik_dlag(Y, hPsi, dll_deta);
                    dlag_calculated = true;
                }
                // dlag_grad.at(0): gradient of loglike w.r.t. ln_mu
                // dlogprior_dpar: gradient of log prior w.r.t. ln_mu
                // A normal prior is placed on ln_mu
                grad.at(i) = dlag_grad.at(0) + Prior::dlogprior_dpar(dlag.par1, lag_par1_prior);
                break;
            }
            case AVAIL::Param::lag_par2:
            {
                if (!dlag_calculated)
                {
                    dlag_grad = dloglik_dlag(Y, hPsi, dll_deta);
                    dlag_calculated = true;
                }
                // dlag_grad.at(1): gradient of loglike w.r.t. log(ln_sd2)
                // dlogprior_dpar: gradient of log prior w.r.t. log(ln_sd2) when the 3rd argument is true
                // And inv-gamma prior is placed on ln_sd2
                grad.at(i) = dlag_grad.at(1) + Prior::dlogprior_dpar(dlag.par2, lag_par2_prior, true);
                break;
            }
            case AVAIL::Param::cnst_beta:
            {
                // dloglik_dlogbeta: gradient of loglike w.r.t. log(beta)
                // dlogprior_dpar: gradient of log prior w.r.t. log(beta).
                // The normal prior is placed on log(beta)
                double logbeta = std::log(std::max(beta, EPS));
                grad.at(i) = dloglik_dlogbeta(Y, dll_deta) + Prior::dlogprior_dpar(logbeta, beta_prior);
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

    arma::vec get_local_params_unconstrained(const unsigned int &s, const std::vector<std::string> &local_params)
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
            case AVAIL::Param::W:
            {
                unconstrained_params.at(i) = std::log(std::max(W.at(s), EPS));
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
            case AVAIL::Param::W:
            {
                W.at(s) = std::exp(std::min(unconstrained_params.at(i), UPBND));
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
        const Prior &rho_prior,
        const Prior &W_prior
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
            case AVAIL::Param::W:
            {
                // dloglik_dlogW: gradient of loglike w.r.t. log(W)
                // dlogprior_dpar: gradient of log prior w.r.t. log(W).
                // The inv-gamma prior is placed on W
                grad.at(i) = dloglik_dlogW(s, y, wt) + Prior::dlogprior_dpar(W.at(s), W_prior, true);
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