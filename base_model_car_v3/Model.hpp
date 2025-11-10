#pragma once
#ifndef MODEL_H
#define MODEL_H

#include <RcppArmadillo.h>
#include "../core/ErrDist.hpp"
#include "../core/SysEq.hpp"
#include "../core/TransFunc.hpp"
#include "../core/ObsDist.hpp"
#include "../core/LinkFunc.hpp"
#include "../core/Regression.hpp"
#include "../utils/utils.h"
#include "SpatialStructure.hpp"

// [[Rcpp::plugins(cpp17)]]
// [[Rcpp::depends(RcppArmadillo)]]

class Model
{
public:
    unsigned int nP;
    unsigned int nS; // number of locations for spatio-temporal model
    arma::vec log_alpha, log_beta, rho;

    SpatialStructure spatial_alpha;
    SpatialStructure spatial_beta;
    SpatialStructure spatial_wt;

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
        log_beta = arma::vec(nS, arma::fill::ones);

        dlag.init("lognorm", LN_MU, LN_SD2, true);
        nP = LagDist::get_nlag(dlag);

        spatial_alpha = SpatialStructure(nS);
        spatial_beta = SpatialStructure(nS);
        spatial_wt = SpatialStructure(nS);

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
            spatial_alpha = SpatialStructure(V);
            spatial_beta = SpatialStructure(V);
            spatial_wt = SpatialStructure(V);
        }
        else
        {
            throw std::invalid_argument("Model::Model - neighborhood matrix 'neighborhood_matrix' is missing.");
        }

        if (spatial_settings.containsElementNamed("car_alpha"))
        {
            arma::vec car_param = Rcpp::as<arma::vec>(spatial_settings["car_alpha"]);
            if (car_param.n_elem != 3)
            {
                throw std::invalid_argument("Model::Model - CAR parameters 'car' should be a vector of length 3.");
            }
            double car_mu = car_param[0];
            double car_tau2 = car_param[1];
            double car_rho = car_param[2];

            spatial_alpha.update_params(car_mu, car_tau2, car_rho);
        }
        else
        {
            spatial_alpha.init_params();
        }

        if (spatial_settings.containsElementNamed("car_beta"))
        {
            arma::vec car_param = Rcpp::as<arma::vec>(spatial_settings["car_beta"]);
            if (car_param.n_elem != 3)
            {
                throw std::invalid_argument("Model::Model - CAR parameters 'car' should be a vector of length 3.");
            }
            double car_mu = car_param[0];
            double car_tau2 = car_param[1];
            double car_rho = car_param[2];

            spatial_beta.update_params(car_mu, car_tau2, car_rho);
        }
        else
        {
            spatial_beta.init_params();
        }

        if (spatial_settings.containsElementNamed("car_wt"))
        {
            arma::vec car_param = Rcpp::as<arma::vec>(spatial_settings["car_wt"]);
            if (car_param.n_elem != 3)
            {
                throw std::invalid_argument("Model::Model - CAR parameters 'car' should be a vector of length 3.");
            }
            double car_mu = 0.0;
            double car_tau2 = car_param[1];
            double car_rho = car_param[2];

            spatial_wt.update_params(car_mu, car_tau2, car_rho);
        }
        else
        {
            spatial_wt.init_params();
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


        if (param_settings.containsElementNamed("spatial_effect_alpha"))
        {
            arma::vec log_alpha_in = Rcpp::as<arma::vec>(param_settings["spatial_effect_alpha"]);
            if (log_alpha_in.n_elem != nS)
            {
                throw std::invalid_argument("Model::Model - dimension of 'spatial_effect_alpha' is not consistent with 'nlocation'.");
            }
            log_alpha = log_alpha_in;
        }
        else
        {
            log_alpha = spatial_alpha.prior_sample_spatial_effects_vec();
        }

        if (param_settings.containsElementNamed("spatial_effect_beta"))
        {
            arma::vec log_beta_in = Rcpp::as<arma::vec>(param_settings["spatial_effect_beta"]);
            if (log_beta_in.n_elem != nS)
            {
                throw std::invalid_argument("Model::Model - dimension of 'spatial_effect_beta' is not consistent with 'nlocation'.");
            }
            log_beta = log_beta_in;
        }
        else
        {
            log_beta = spatial_beta.prior_sample_spatial_effects_vec();
        }
    } // end of Model(const Rcpp::List &settings)

    void simulate(
        arma::mat &Y, 
        arma::mat &Lambda, 
        arma::mat &wt,
        arma::mat &Psi, 
        const unsigned int &ntime)
    {
        const arma::vec spatial_effects_alpha = arma::exp(log_alpha); // nS x 1
        const arma::vec spatial_effects_beta = arma::exp(log_beta); // nS x 1

        arma::cube Theta(nP, ntime + 1, nS, arma::fill::randn);
        Y.set_size(nS, ntime + 1);
        Y.zeros();
        Lambda.set_size(nS, ntime + 1);

        arma::mat rchol_wt = arma::chol(arma::symmatu(spatial_wt.Q));
        wt.set_size(nS, ntime + 1);
        wt.randn();
        wt = arma::solve(arma::trimatu(rchol_wt), wt);
        wt /= std::sqrt(std::max(spatial_wt.car_tau2, EPS));

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
                double eta = spatial_effects_alpha.at(s);

                // Self-exciting component
                eta += TransFunc::func_ft(
                    ftrans, fgain, dlag, seas, t,
                    Theta.slice(s).col(t), Y.row(s).t()
                );

                // Cross-regional effect
                eta += spatial_effects_beta.at(s) * arma::dot(
                    spatial_beta.W.row(s).t(), Y.col(t-1)
                );

                Lambda.at(s, t) = LinkFunc::ft2mu(eta, flink);
                Y.at(s, t) = ObsDist::sample(Lambda.at(s, t), rho.at(s), dobs);

            } // end of for s in [0, nS]
        } // end of for t in [1, ntime]

        return;
    } // end of simulate()

    arma::vec compute_intensity_iterative(
        const unsigned int &s,
        const arma::mat &Y, // nS x (nT + 1)
        const arma::vec &psi // (nT + 1) x 1
    ) const
    {
        arma::vec y = Y.row(s).t();
        arma::vec neighbor_weights = spatial_beta.W.row(s).t(); // nS x 1
        const double spatial_effect_beta = std::exp(std::min(log_beta.at(s), UPBND));
        const double spatial_effect_alpha = std::exp(std::min(log_alpha.at(s), UPBND));

        arma::vec hpsi = GainFunc::psi2hpsi<arma::vec>(psi, fgain);
        arma::vec ft(psi.n_elem, arma::fill::zeros);

        arma::vec Lambda(psi.n_elem, arma::fill::zeros);
        
        for (unsigned int t = 1; t < y.n_elem; t++)
        {
            double eta = spatial_effect_alpha;

            // Self-exciting component
            ft.at(t) = TransFunc::func_ft(t, y, ft, hpsi, dlag, ftrans);            
            eta += ft.at(t);

            // Cross-regional effect
            eta += spatial_effect_beta * arma::dot(neighbor_weights, Y.col(t - 1));
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
            const double spatial_effect_alpha = std::exp(std::min(log_alpha.at(s), UPBND));
            const double spatial_effect_beta = std::exp(std::min(log_beta.at(s), UPBND));
            const arma::vec neighbor_weights = spatial_beta.W.row(s).t();
            const arma::vec cross_region_effects = spatial_effect_beta * Y.head_cols(nT).t() * neighbor_weights; // nT x 1

            for (unsigned int t = 1; t < Y.n_cols; t++)
            {
                double eta = spatial_effect_alpha;
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


    arma::vec dloglik_dlogbeta(
        const arma::mat &Y, // nS x (nT + 1)
        const arma::mat &dll_deta // nS x (nT + 1)
    )
    {
        // Gradient of the log-likelihood w.r.t. log(beta)
        arma::vec grad(nS, arma::fill::zeros);
        for (unsigned int s = 0; s < nS; s++)
        {
            double spatial_effect_beta = std::exp(std::min(log_beta.at(s), UPBND));
            for (unsigned int t = 1; t < Y.n_cols; t++)
            {
                double deta_dbeta = arma::dot(spatial_beta.W.row(s).t(), Y.col(t - 1));
                grad.at(s) += dll_deta.at(s, t) * deta_dbeta * spatial_effect_beta; // dloglik_dlogbeta
            }
        }

        // Gradient of the CAR prior w.r.t. log(beta)
        grad -= spatial_beta.car_tau2 * spatial_beta.Q * (log_beta - spatial_beta.car_mu);

        return grad;
    } // end of dloglik_dlogbeta()


    arma::vec dloglik_dlogalpha(const arma::mat &Y, const arma::mat &dll_deta)
    {
        // Gradient of the log likelihood w.r.t. log_alpha
        arma::vec grad(nS, arma::fill::zeros);
        for (unsigned int s = 0; s < nS; s++)
        {
            double spatial_effect_alpha = std::exp(std::min(log_alpha.at(s), UPBND));
            for (unsigned int t = 1; t < Y.n_cols; t++)
            {
                grad.at(s) += dll_deta.at(s, t) * spatial_effect_alpha;
            }
        }

        // Gradient of the CAR prior w.r.t. log_alpha
        grad -= spatial_alpha.car_tau2 * spatial_alpha.Q * (log_alpha - spatial_alpha.car_mu);

        return grad;
    } // end of dloglik_dlogalpha()


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
        const Prior &lag_par2_prior)
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