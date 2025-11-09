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

// [[Rcpp::plugins(cpp17)]]
// [[Rcpp::depends(RcppArmadillo)]]


/**
 * @brief Define a dynamic generalized transfer function (DGTF) model using the transfer function form.
 *
 * @param dobs observation distribution characterized by either (mu0, delta_nb) or (mu, sd2)
 *       - (mu0, delta_nb):
 *          (1) mu0: constant mean of the observed time series;
 *       - (mu, sd2): parameters for gaussian observation distribution.
 * @param dlag lag distribution characterized by either (kappa, r) for negative-binomial lags or (mu, sd2) for lognormal lags.
 *
 */
class Model
{
public:
    unsigned int nP;
    Season seas;
    ZeroInflation zero;

    ObsDist dobs;
    LagDist dlag;
    ErrDist derr;

    std::string fsys = "shift";
    std::string ftrans = "sliding";
    std::string flink = "identity";
    std::string fgain = "softplus";
    
    Model()
    {
        // no seasonality and no baseline mean in the latent state by default
        flink = "identity";
        fgain = "softplus";
        fsys = "shift";
        ftrans = "sliding";

        dobs.init_default();
        derr.init_default();
        dlag.init("lognorm", LN_MU, LN_SD2, true);

        seas.init_default();
        nP = get_nP(dlag, seas.period, seas.in_state);

        zero.init_default();
        return;
    }


    Model(const Rcpp::List &settings)
    {
        init(settings);
    }

    void init(const Rcpp::List &settings)
    {
        Rcpp::List model_settings = settings["model"];
        init_model(model_settings);

        Rcpp::List param_settings = settings["param"];
        Rcpp::NumericVector obs_param = Rcpp::NumericVector::create(0., 30.);
        if (param_settings.containsElementNamed("obs"))
        {
            obs_param = Rcpp::as<Rcpp::NumericVector>(param_settings["obs"]);
        }
        dobs.par1 = obs_param[0];
        dobs.par2 = obs_param[1];


        Rcpp::NumericVector lag_param {LN_MU, LN_SD2};
        std::map<std::string, SysEq::Evolution> sys_list = SysEq::sys_list;
        if (sys_list[fsys] == SysEq::Evolution::identity)
        {
            if (!param_settings.containsElementNamed("lag"))
            {
                throw std::invalid_argument("Model::init - true/initial values of autoregressive coefficients are missing.");
            }

            Rcpp::NumericVector tmp = Rcpp::as<Rcpp::NumericVector>(param_settings["lag"]);
            // Use uniform lag distribution (specified in `init_model`)
            lag_param[0] = static_cast<double>(tmp.length()); // order of autoregression
            lag_param[1] = 0.; // no meaning
        }
        else if (param_settings.containsElementNamed("lag"))
        {
            Rcpp::NumericVector tmp = Rcpp::as<Rcpp::NumericVector>(param_settings["lag"]);
            // Lag distribution is specified in `init_model`
            lag_param[0] = tmp[0]; // first param of lag distribution
            lag_param[1] = tmp[1]; // second param of lag distribution
        }
        dlag.init(dlag.name, lag_param[0], lag_param[1], dlag.truncated);

        
        if (settings.containsElementNamed("season"))
        {
            Rcpp::List season_settings = settings["season"];
            seas.init(season_settings);
        }
        else
        {
            seas.init_default();
        }

        nP = get_nP(dlag, seas.period, seas.in_state);

        if (param_settings.containsElementNamed("err"))
        {
            Rcpp::List err_opts = Rcpp::as<Rcpp::List>(param_settings["err"]);
            derr.init(err_opts, nP);
        }


        if (settings.containsElementNamed("zero"))
        {
            Rcpp::List zero_settings = settings["zero"];
            zero.init(zero_settings);
        }
        else
        {
            zero.init_default();
        }

        return;
    }

    void init_model(const Rcpp::List &model_settings)
    {
        Rcpp::List model = model_settings;
        dobs.name = "nbinom";
        if (model.containsElementNamed("obs_dist"))
        {
            dobs.name = tolower(Rcpp::as<std::string>(model["obs_dist"]));
        }

        flink = "identity";
        if (model.containsElementNamed("link_func"))
        {
            flink = tolower(Rcpp::as<std::string>(model["link_func"]));
        }

        fsys = "shift";
        if (model.containsElementNamed("sys_eq"))
        {
            fsys = tolower(Rcpp::as<std::string>(model["sys_eq"]));
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
            if (model.containsElementNamed("lag_dist"))
            {
                dlag.name = tolower(Rcpp::as<std::string>(model["lag_dist"]));
            }
        }

        fgain = "softplus";
        if (model.containsElementNamed("gain_func"))
        {
            fgain = tolower(Rcpp::as<std::string>(model["gain_func"]));
        }

        derr.name = "gaussian";
        if (model.containsElementNamed("err_dist"))
        {
            derr.name = tolower(Rcpp::as<std::string>(model["err_dist"]));
        }
        return;
    }


    static Rcpp::List default_settings()
    {
        Rcpp::List model_settings;
        model_settings["obs_dist"] = "nbinom";
        model_settings["link_func"] = "identity";
        model_settings["gain_func"] = "softplus";
        model_settings["lag_dist"] = "lognorm";
        model_settings["sys_eq"] = "shift";
        model_settings["err_dist"] = "gaussian";

        Rcpp::List param_settings;
        param_settings["obs"] = Rcpp::NumericVector::create(0., 30.);
        param_settings["zero"] = Rcpp::NumericVector::create(0., 0.);
        param_settings["lag"] = Rcpp::NumericVector::create(1.4, 0.3);
        param_settings["err"] = ErrDist::default_settings();

        Rcpp::List settings;
        settings["model"] = model_settings;
        settings["param"] = param_settings;
        settings["season"] = Season::default_settings();
        settings["zero"] = ZeroInflation::default_settings();

        return settings;
    }



    static unsigned int get_nP(
        const LagDist &dlag, 
        const unsigned int &seasonal_period = 0,
        const bool &season_in_state = false)
    {
        unsigned int nP;
        if (dlag.truncated)
        {
            nP = dlag.nL;
        }
        else
        {
            nP = static_cast<unsigned int>(dlag.par2) + 1;
        }

        if (season_in_state)
        {
            nP += seasonal_period;
        }
        
        return nP;
    }


    arma::vec wt2lambda(
        const arma::vec &y, // (nT + 1) x 1
        const arma::vec &wt, // (nT + 1) x 1
        const unsigned int &seasonal_period,
        const arma::mat &X, // period x (nT + 1)
        const arma::vec &seas
    ) // period x 1, checked. ok.
    {
        arma::vec psi = arma::cumsum(wt);
        arma::vec hpsi = GainFunc::psi2hpsi<arma::vec>(psi, fgain);
        
        arma::vec ft(psi.n_elem, arma::fill::zeros);
        arma::vec lambda = ft;
        for (unsigned int t = 1; t < y.n_elem; t++)
        {
            // ft.at(t) = _transfer.func_ft(t, y, ft);
            ft.at(t) = TransFunc::func_ft(t, y, ft, hpsi, dlag, ftrans);
            double eta = ft.at(t);
            if (seasonal_period > 0 && !X.is_empty() && !seas.is_empty())
            {
                eta += arma::dot(X.col(t), seas);
            }
            lambda.at(t) = LinkFunc::ft2mu(eta, flink);
        }

        return lambda;
    }

    static double dloglik_deta(
        const double &eta, 
        const double &yt, 
        const double &obs_par2, 
        const std::string &obs_dist = "nbinomm", 
        const std::string &link_func = "identity")
    {
        std::map<std::string, AVAIL::Dist> obs_list = ObsDist::obs_list;

        double lambda;
        double dlam_deta = LinkFunc::dlambda_deta(lambda, eta, link_func);

        double dloglik_dlam = 0.;
        switch (obs_list[obs_dist])
        {
        case AVAIL::Dist::nbinomm:
        {
            dloglik_dlam = nbinomm::dlogp_dlambda(lambda, yt, obs_par2);
            break;
        }
        case AVAIL::Dist::nbinomp:
        {
            dloglik_dlam = yt / lambda - obs_par2 / std::abs(1. - lambda);
            break;
        }
        case AVAIL::Dist::poisson:
        {
            dloglik_dlam = Poisson::dlogp_dlambda(lambda, yt);
            break;
        }
        case AVAIL::Dist::gaussian:
        {
            dloglik_dlam = (yt - lambda) / obs_par2;
            break;
        }
        default:
        {
            throw std::invalid_argument("Model::dloglik_deta: observation distribution must be nbinomm or poisson.");
        }
        }

        double dloglik_deta = dloglik_dlam * dlam_deta;
        #ifdef DGTF_DO_BOUND_CHECK
            bound_check(dloglik_deta, "Model::dloglik_deta: dloglik_deta");
        #endif
        return dloglik_deta;
    }

    /**
     * @brief This one is for HMC. Parameter must be first mapped to real line.
     * 
     * @param y 
     * @param hpsi 
     * @param nlag 
     * @param lag_dist 
     * @param lag_par1 
     * @param lag_par2 
     * @param dobs 
     * @param seas
     * @param zero
     * @param link_func 
     * @return arma::vec 
     * 
     * @note For iterative transfer function, we must use its EXACT sliding form.
     */
    static arma::vec dloglik_dlag(
        const arma::vec &y, // (ntime + 1) x 1
        const arma::vec &hpsi, // (ntime + 1) x 1
        const unsigned int &nlag,
        const std::string &lag_dist,
        const double &lag_par1,
        const double &lag_par2,
        const ObsDist &dobs,
        const Season &seas,
        const ZeroInflation &zero,
        const std::string &link_func
    )
    {
        arma::vec Fphi = LagDist::get_Fphi(nlag, lag_dist, lag_par1, lag_par2);
        arma::mat dFphi_grad = LagDist::get_Fphi_grad(nlag, lag_dist, lag_par1, lag_par2);

        arma::vec dll_dlag(2, arma::fill::zeros);

        for (unsigned int t = 1; t < y.n_elem; t++)
        {
            if (!zero.inflated || zero.z.at(t) > EPS)
            {
                double eta = TransFunc::transfer_sliding(t, nlag, y, Fphi, hpsi);
                if (seas.period > 0)
                {
                    eta += arma::dot(seas.X.col(t), seas.val);
                }
                double dll_deta = dloglik_deta(eta, y.at(t), dobs.par2, dobs.name, link_func);

                double deta_dpar1 = TransFunc::transfer_sliding(t, nlag, y, dFphi_grad.col(0), hpsi);
                double deta_dpar2 = TransFunc::transfer_sliding(t, nlag, y, dFphi_grad.col(1), hpsi);

                dll_dlag.at(0) += dll_deta * deta_dpar1;
                dll_dlag.at(1) += dll_deta * deta_dpar2;
            }
        }

        #ifdef DGTF_DO_BOUND_CHECK
            bound_check<arma::mat>(dll_dlag, "Model::dloglik_dlag: grad");
        #endif
        return dll_dlag;
    }


    static double dlogp_dpar2_obs(
        const Model &model, 
        const arma::vec &y, 
        const arma::vec &lambda, // (nT + 1) x 1
        const bool &jacobian = true)
    {
        LagDist dlag = model.dlag;
        dlag.Fphi = LagDist::get_Fphi(dlag.nL, dlag.name, dlag.par1, dlag.par2);
        double out = 0.;
        for (unsigned int t = 1; t < y.n_elem; t++)
        {
            if (!(model.zero.inflated && model.zero.z.at(t) < EPS))
            {
                out += nbinomm::dlogp_dpar2(y.at(t), lambda.at(t), model.dobs.par2, jacobian);
            }
        }

        return out;
    }


private:
    double _y0 = 0.;
    double _mu0 = 1.;

    

};

/**
 * @brief Define a dynamic generalized transfer function (DGTF) model using the state space DLM form.
 *
 */
class StateSpace
{
public:
    static void simulate(
        arma::vec &y,
        arma::vec &lambda,
        arma::vec &ft,
        arma::mat &Theta,
        arma::vec &psi, // (ntime + 1) x 1
        Model &model,
        const unsigned int &ntime,
        const double &y0,
        const arma::vec &theta0,
        const bool &full_rank = false)
    {
        std::map<std::string, TransFunc::Transfer> trans_list = TransFunc::trans_list;
        std::map<std::string, AVAIL::Dist> obs_list = ObsDist::obs_list;

        if (!model.dlag.truncated)
        {
            model.dlag.nL = ntime;
        }
        model.dlag.Fphi = LagDist::get_Fphi(model.dlag);

        arma::mat Gmat = SysEq::init_Gt(
            model.nP, model.dlag, model.fsys,
            model.seas.period, model.seas.in_state);

        if (model.seas.period > 0)
        {
            model.seas.X = Season::setX(ntime, model.seas.period, model.seas.P);
        }

        if (model.zero.z.is_empty())
        {
            model.zero.simulateZ(ntime);
        }

        psi = ErrDist::sample(model.derr, ntime, true);

        Theta.set_size(model.nP, ntime + 1);
        Theta.zeros();
        Theta.col(0) = theta0;
        if (trans_list[model.ftrans] == TransFunc::Transfer::iterative)
        {
            Theta.at(0, 0) = psi.at(1);
        }

        y.set_size(ntime + 1);
        y.zeros();
        y.at(0) = y0;
        lambda = y;
        ft = y;

        double npop = 1.;
        if (obs_list[model.dobs.name] == AVAIL::Dist::nbinomp)
        {
            npop = model.dobs.par2;
        }

        for (unsigned int t = 1; t < (ntime + 1); t++)
        {
            unsigned int psi_idx;
            if (trans_list[model.ftrans] == TransFunc::Transfer::iterative && (t < ntime))
            {
                psi_idx = t + 1;
            }
            else
            {
                psi_idx = t;
            }

            Theta.col(t) = SysEq::func_gt(
                model.fsys, model.fgain, model.dlag,
                Theta.col(t - 1), y.at(t - 1), 0, false);

            if ((!model.derr.full_rank) && (model.derr.par1 > EPS))
            {
                // Only update theta if err variance > 0
                Theta.at(0, t) = psi.at(psi_idx);
            }
            else if (model.derr.full_rank)
            {
                arma::vec eps = arma::randn<arma::vec>(Theta.n_rows);
                arma::mat var_chol = arma::chol(model.derr.var);
                Theta.col(t) = Theta.col(t) + var_chol.t() * eps;
                psi.at(psi_idx) = Theta.at(0, t);
            }

            ft.at(t) = TransFunc::func_ft(
                model.ftrans, model.fgain, model.dlag,
                model.seas, t, Theta.col(t), y);

            double eta = ft.at(t);
            lambda.at(t) = LinkFunc::ft2mu(eta, model.flink); // Checked. OK.

            if (std::abs(model.zero.z.at(t) - 1.) < EPS)
            {
                y.at(t) = ObsDist::sample(lambda.at(t), model.dobs.par2, model.dobs.name);
            }
            else
            {
                y.at(t) = 0.;
            }
        }

        return; // Checked. OK.
    }

};

#endif