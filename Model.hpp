#pragma once
#ifndef MODEL_H
#define MODEL_H

#include <RcppArmadillo.h>
#include "ErrDist.hpp"
#include "TransFunc.hpp"
#include "ObsDist.hpp"
#include "LinkFunc.hpp"
#include "Regression.hpp"
#include "utils.h"

// #ifdef _OPENMP
// #include <omp.h>
// #else
// #define omp_get_num_threads() 0
// #define omp_get_thread_num() 0
// #endif

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

    ObsDist dobs;
    LagDist dlag;
    ErrDist derr;

    std::string ftrans;
    std::string flink;
    std::string fgain;
    
    Model()
    {
        // no seasonality and no baseline mean in the latent state by default
        flink = "identity";
        fgain = "softplus";
        ftrans = "sliding";

        dobs.init_default();
        derr.init_default();
        dlag.init("lognorm", LN_MU, LN_SD2, true);

        seas.init_default();
        nP = get_nP(dlag, seas.period, seas.in_state);
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
        Rcpp::NumericVector lag_param, obs_param; //, err_param;
        init_param(obs_param, lag_param, dlag.truncated, param_settings);
        dlag.init(dlag.name, lag_param[0], lag_param[1], dlag.truncated);
        dobs.par1 = obs_param[0];
        dobs.par2 = obs_param[1];

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

        // err_param = Rcpp::NumericVector::create(0.01, 0.);
        // if (param_settings.containsElementNamed("err"))
        // {
        //     err_param = Rcpp::as<Rcpp::NumericVector>(param_settings["err"]);
        // }
        if (param_settings.containsElementNamed("err"))
        {
            Rcpp::List err_opts = Rcpp::as<Rcpp::List>(param_settings["err"]);
            derr.init(err_opts, nP);
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

        ftrans = "sliding";
        if (model.containsElementNamed("trans_func"))
        {
            ftrans = tolower(Rcpp::as<std::string>(model["trans_func"]));
        }

        std::map<std::string, TransFunc::Transfer> trans_list = TransFunc::trans_list;
        if (trans_list[ftrans] == TransFunc::Transfer::iterative)
        {
            dlag.truncated = false;
        }
        else
        {
            dlag.truncated = true;
        }

        fgain = "softplus";
        if (model.containsElementNamed("gain_func"))
        {
            fgain = tolower(Rcpp::as<std::string>(model["gain_func"]));
        }

        dlag.name = "lognorm";
        if (model.containsElementNamed("lag_dist"))
        {
            dlag.name = tolower(Rcpp::as<std::string>(model["lag_dist"]));
        }

        derr.name = "gaussian";
        if (model.containsElementNamed("err_dist"))
        {
            derr.name = tolower(Rcpp::as<std::string>(model["err_dist"]));
        }
        return;
    }

    static void init_param(
        Rcpp::NumericVector &obs,
        Rcpp::NumericVector &lag,
        bool &truncated,
        const Rcpp::List &param_settings)
    {
        Rcpp::List param = param_settings;

        obs = Rcpp::NumericVector::create(0., 30.);
        if (param.containsElementNamed("obs"))
        {
            obs = Rcpp::as<Rcpp::NumericVector>(param["obs"]);
        }

        lag = Rcpp::NumericVector::create(LN_MU, LN_SD2);
        if (param.containsElementNamed("lag"))
        {
            lag = Rcpp::as<Rcpp::NumericVector>(param["lag"]);
        }
    }


    Rcpp::List info()
    {
        Rcpp::List model;
        model["obs_dist"] = dobs.name;
        model["link_func"] = flink;
        model["trans_func"] = ftrans;
        model["gain_func"] = fgain;
        model["lag_dist"] = dlag.name;
        model["err_dist"] = derr.name;

        Rcpp::List param;
        param["obs"] = Rcpp::NumericVector::create(dobs.par1, dobs.par2);
        param["lag"] = Rcpp::NumericVector::create(dlag.par1, dlag.par2);
        param["err"] = Rcpp::NumericVector::create(derr.par1, derr.par2);

        Rcpp::List out;
        out["model"] = model;
        out["param"] = param;

        Rcpp::List season = seas.info();
        out["season"] = season;
        return out;
    }

    static Rcpp::List default_settings()
    {
        Rcpp::List model_settings;
        model_settings["obs_dist"] = "nbinom";
        model_settings["link_func"] = "identity";
        model_settings["trans_func"] = "sliding";
        model_settings["gain_func"] = "softplus";
        model_settings["lag_dist"] = "lognorm";
        model_settings["err_dist"] = "gaussian";
        model_settings["seasonal_period"] = 1;
        model_settings["season_in_state"] = false;

        Rcpp::List param_settings;
        param_settings["obs"] = Rcpp::NumericVector::create(0., 30.);
        param_settings["lag"] = Rcpp::NumericVector::create(1.4, 0.3);
        // param_settings["err"] = Rcpp::NumericVector::create(0.01, 0.);
        param_settings["err"] = ErrDist::default_settings();

        Rcpp::List settings;
        settings["model"] = model_settings;
        settings["param"] = param_settings;
        settings["season"] = Season::default_settings();

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


    /**
     * @brief Forecasting using the transfer function form.
     * 
     * @param y 
     * @param psi_stored 
     * @param W_stored 
     * @param transfer 
     * @param link_func 
     * @param mu0 
     * @param k 
     * @return arma::mat 
     */
    static arma::mat forecast(      // transfer function based
        const arma::vec &y,          // (nT + 1) x 1
        const arma::mat &psi_stored, // (nT + 1) x nsample
        const arma::vec &W_stored,   // nsample x 1
        const LagDist &dlag,
        const Season &seas,
        const std::string &ftrans,
        const std::string &link_func = "identity",
        const std::string &gain_func = "softplus",
        const unsigned int &k = 1)
    {
        std::map<std::string, AVAIL::Dist> obs_list = ObsDist::obs_list;
        unsigned int nsample = psi_stored.n_cols;
        unsigned int nT = y.n_elem - 1;

        arma::mat psi_all(nT + 1 + k, nsample, arma::fill::zeros);
        arma::mat yall(nT + 1 + k, nsample, arma::fill::zeros);
        arma::mat ft(nT + 1 + k, nsample, arma::fill::zeros);
        Season seass = seas;
        seass.X = Season::setX(nT + k, seass.period, seass.P);

        for (unsigned int i = 0; i < nsample; i++)
        {
            arma::vec ft_vec(nT + 1 + k, arma::fill::zeros); // (nT + 1) x 1
            arma::vec psi_vec(nT + 1 + k, arma::fill::zeros);
            psi_vec.head(nT + 1) = psi_stored.col(i);
            arma::vec hpsi = GainFunc::psi2hpsi<arma::vec>(psi_vec, gain_func);

            for (unsigned int t = 1; t < (nT + 1); t++)
            {
                yall.at(t, i) = y.at(t);
                ft_vec.at(t) = TransFunc::func_ft(t, y, ft_vec, hpsi, dlag, ftrans);
            }

            ft.col(i) = ft_vec;
            psi_all.col(i) = psi_vec;
        }


        for (unsigned int i = 0; i < nsample; i++)
        {
            arma::vec psi_vec = psi_all.col(i);
            arma::vec hpsi_vec = GainFunc::psi2hpsi(psi_vec, gain_func);
            arma::vec ft_vec = ft.col(i);
            arma::vec yvec = yall.col(i); // (nT + 1 + k) x 1

            yvec.head(y.n_elem) = y;
            double Wsqrt = std::sqrt(W_stored.at(i));
            for (unsigned int t = 0; t < k; t++)
            {
                /**
                 * @brief `t + nT`-step-ahead forecasting
                 * 
                 */
                unsigned int idx = t + nT; // idx of old
                // psi_vec.at(idx + 1) = psi_vec.at(idx);
                hpsi_vec.at(idx + 1) = hpsi_vec.at(idx);
                ft_vec.at(idx + 1) = TransFunc::func_ft(
                    idx + 1, yvec, ft_vec, hpsi_vec, dlag, ftrans);

                double eta = ft_vec.at(idx + 1);
                if (seass.period > 0)
                {
                    eta += arma::as_scalar(seass.X.col(idx + 1).t() * seass.val);
                }


                double lambda = LinkFunc::ft2mu(eta, link_func);
                yvec.at(idx + 1) = lambda;
            }

            yall.col(i) = yvec;
        }


        arma::mat ycast = yall.tail_rows(k); // k x nsample
        return ycast;
    }


    /**
     * @brief Forecasting using the transfer function form.
     * 
     * @param y 
     * @param psi_stored 
     * @param W_stored 
     * @param model 
     * @param k 
     * @param random_sample 
     * @return Rcpp::List 
     */
    static Rcpp::List forecast( // transfer function based
        const arma::vec &y,          // (nT + 1) x 1
        const arma::mat &psi_stored, // (nT + 1) x nsample
        const arma::vec &W_stored,   // nsample x 1
        const Model &model,
        const unsigned int &k = 1,
        const bool &random_sample = false)
    {
        std::map<std::string, AVAIL::Dist> obs_list = ObsDist::obs_list;
        unsigned int nsample = psi_stored.n_cols;
        unsigned int nT = y.n_elem - 1;

        arma::mat psi_all(nT + 1 + k, nsample, arma::fill::zeros);
        arma::mat yall(nT + 1 + k, nsample, arma::fill::zeros);
        arma::mat ft(nT + 1 + k, nsample, arma::fill::zeros);
        Season seas = model.seas;
        seas.X = Season::setX(nT + k, seas.period, seas.P);

        for (unsigned int i = 0; i < nsample; i++)
        {
            arma::vec ft_vec(nT + 1 + k, arma::fill::zeros); // (nT + 1) x 1
            arma::vec psi_vec(nT + 1 + k, arma::fill::zeros);
            psi_vec.head(nT + 1) = psi_stored.col(i);
            arma::vec hpsi_vec = GainFunc::psi2hpsi<arma::vec>(psi_vec, model.fgain);

            for (unsigned int t = 1; t < (nT + 1); t++)
            {
                yall.at(t, i) = y.at(t);
                ft_vec.at(t) = TransFunc::func_ft(
                    t, y, ft_vec, hpsi_vec, model.dlag, model.ftrans);
            }

            ft.col(i) = ft_vec;
            psi_all.col(i) = psi_vec;
        }

        for (unsigned int i = 0; i < nsample; i++)
        {
            arma::vec psi_vec = psi_all.col(i);
            arma::vec hpsi_vec = GainFunc::psi2hpsi<arma::vec>(psi_vec, model.fgain);
            arma::vec ft_vec = ft.col(i);
            arma::vec yvec = yall.col(i);

            double Wsqrt = std::sqrt(W_stored.at(i));

            for (unsigned int t = 0; t < k; t++)
            {
                unsigned int idx = t + nT; // idx of old
                hpsi_vec.at(idx + 1) = hpsi_vec.at(idx);
                ft_vec.at(idx + 1) = TransFunc::func_ft(
                    idx + 1, yvec, ft_vec, hpsi_vec, 
                    model.dlag, model.ftrans);

                double eta = ft_vec.at(idx + 1);
                if (seas.period > 0)
                {
                    eta += arma::as_scalar(seas.X.col(idx + 1).t() * seas.val);
                }

                double lambda = LinkFunc::ft2mu(eta, model.flink);

                if (random_sample)
                {
                    switch (obs_list[model.dobs.name])
                    {
                    case AVAIL::Dist::nbinomm:
                    {
                        yvec.at(idx + 1) = nbinomm::sample(lambda, model.dobs.par2);
                        break;
                    }
                    case AVAIL::Dist::poisson:
                    {
                        yvec.at(idx + 1) = Poisson::sample(lambda);
                        break;
                    }
                    default:
                    {
                        break;
                    }
                    }
                }
                else
                {
                    yvec.at(idx + 1) = lambda;
                }
                
            }

            yall.col(i) = yvec;
        }

        
        arma::vec qprob = {0.025, 0.5, 0.975};
        arma::mat yqt = arma::quantile(yall, qprob, 1);

        Rcpp::List out;
        out["y"] = Rcpp::wrap(yqt);
        out["yall"] = Rcpp::wrap(yall);
        out["yfit"] = Rcpp::wrap(y);
        out["ypred"] = Rcpp::wrap(yall.tail_rows(k));
        return out;
    }


    static Rcpp::List forecast_error(
        const arma::mat &psi, // (nT + 1) x nsample
        const arma::vec &y,   // (nT + 1) x 1
        const Model &model,
        const std::string &loss_func = "quadratic",
        const unsigned int &k = 1,
        const bool &verbose = VERBOSE,
        const Rcpp::Nullable<unsigned int> &start_time = R_NilValue,
        const Rcpp::Nullable<unsigned int> &end_time = R_NilValue)
    {
        const unsigned int nT = y.n_elem - 1;
        const unsigned int nsample = psi.n_cols;
        arma::cube ycast = arma::zeros<arma::cube>(nT + 1, nsample, k);
        arma::cube y_err_cast = arma::zeros<arma::cube>(nT + 1, nsample, k);
        arma::mat y_cov_cast(nT + 1, k, arma::fill::zeros); // (nT + 1) x k
        arma::mat y_width_cast = y_cov_cast;

        unsigned int tstart = std::max(k, model.nP);
        if (start_time.isNotNull()) 
        {
            tstart = Rcpp::as<unsigned int>(start_time);
        }

        unsigned int tend = nT - k;
        if (end_time.isNotNull())
        {
            tend = Rcpp::as<unsigned int>(end_time);
        }

        for (unsigned int i = 0; i < nsample; i ++)
        {
            arma::vec psi_vec = psi.col(i); // (nT + 1) x 1
            arma::vec hpsi_vec = GainFunc::psi2hpsi<arma::vec>(psi_vec, model.fgain);
            arma::vec ft_vec(nT + 1, arma::fill::zeros);

            for (unsigned int t = 0; t < tstart; t++)
            {
                ft_vec.at(t + 1) = TransFunc::func_ft(
                    t + 1, y, ft_vec, hpsi_vec, model.dlag, model.ftrans);
            }

            for (unsigned int t = tstart; t < tend; t++)
            {
                Rcpp::checkUserInterrupt();

                arma::vec hpsi_tmp = GainFunc::psi2hpsi<arma::vec>(psi_vec, model.fgain);
                arma::vec ft_tmp = ft_vec;
                arma::vec ytmp = y;

                // psi_cast.at(t, i, 0) = psi_tmp.at(t);
                // ft_cast.at(t, i, 0) = ft_tmp.at(t);
                ycast.at(t, i, 0) = ytmp.at(t);

                for (unsigned int j = 1; j <= k; j++)
                {
                    hpsi_tmp.at(t + j) = hpsi_tmp.at(t + j - 1);

                    ft_tmp.at(t + j) = TransFunc::func_ft(
                        t + j, ytmp, ft_tmp, hpsi_tmp, model.dlag, model.ftrans);

                    double eta = ft_tmp.at(t + j);
                    if (model.seas.period > 0)
                    {
                        eta += arma::as_scalar(model.seas.X.col(t + j).t() * model.seas.val);
                    }
                    ytmp.at(t + j) = LinkFunc::ft2mu(eta, model.flink);

                    // psi_cast.at(t, i, j - 1) = psi_tmp.at(t + j);
                    // ft_cast.at(t, i, j - 1) = ft_tmp.at(t + j);
                    ycast.at(t, i, j - 1) = ytmp.at(t + j);
                    y_err_cast.at(t, i, j - 1) = y.at(t + j) - ytmp.at(t + j);
                    // psi_err_cast.at(t, i, j - 1) = psi.at(t + j, i) - psi_tmp.at(t + j);
                }

                if (verbose)
                {
                    Rcpp::Rcout << "\rForecast error: " << t + 1 << "/" << tend;
                }
            } // loop over time

            if (verbose)
            {
                Rcpp::Rcout << std::endl;
            }
        } // loop over nsample

        
        for (unsigned int t = 1; t < nT; t++)
        {
            arma::mat ycast_tmp = ycast.row_as_mat(t); // k x nsample
            arma::vec ymin = arma::vectorise(arma::min(ycast_tmp, 1));
            arma::vec ymax = arma::vectorise(arma::max(ycast_tmp, 1));

            unsigned int ncast = std::min(k, nT - t);
            for (unsigned int j = 0; j < ncast; j ++)
            {
                double ytrue = y.at(t + j + 1);
                double covered = (ytrue >= ymin.at(j) && ytrue <= ymax.at(j)) ? 1. : 0.;
                y_cov_cast.at(t, j) = covered;
                y_width_cast.at(t, j) = std::abs(ymax.at(j) - ymin.at(j));
            }
        }

        
        arma::vec qprob = {0.025, 0.5, 0.975};
        std::map<std::string, AVAIL::Loss> loss_list = AVAIL::loss_list;

        arma::mat y_loss(nT + 1, k, arma::fill::zeros);
        arma::vec y_loss_all(k, arma::fill::zeros);
        arma::vec y_covered_all = y_loss_all;
        arma::vec y_width_all = y_loss_all;
        arma::cube yqt = arma::zeros<arma::cube>(nT + 1, qprob.n_elem, k);

        
        for (unsigned int j = 0; j < k; j ++)
        {
            arma::mat ycast_qt = arma::quantile(ycast.slice(j), qprob, 1);
            yqt.slice(j) = ycast_qt;
            arma::mat y_loss_tmp0 = y_err_cast.slice(j);
            arma::mat y_loss_tmp = y_loss_tmp0.submat(tstart, 0, tend, nsample - 1);
            y_loss_tmp = arma::abs(y_loss_tmp);
            arma::vec ytmp;

            arma::vec ycov_tmp = arma::vectorise(y_cov_cast(arma::span(tstart, tend), arma::span(j)));
            y_covered_all.at(j) = arma::mean(ycov_tmp) * 100.;

            ycov_tmp = arma::vectorise(y_width_cast(arma::span(tstart, tend), arma::span(j)));
            y_width_all.at(j) = arma::mean(ycov_tmp);


            switch (loss_list[tolower(loss_func)])
            {
            case AVAIL::L1: // mae
            {
                ytmp = arma::mean(y_loss_tmp, 1);
                y_loss.submat(tstart, j, tend, j) = ytmp;
                y_loss_all.at(j) = arma::mean(ytmp);

                break;
            }
            case AVAIL::L2: // rmse
            {
                y_loss_tmp = arma::square(y_loss_tmp);

                ytmp = arma::mean(y_loss_tmp, 1);      // (nT - i) x 1
                y_loss.submat(tstart, j, tend, j) = arma::sqrt(ytmp);

                y_loss_all.at(j) = arma::mean(ytmp);
                y_loss_all.at(j) = std::sqrt(y_loss_all.at(j));
                break;
            }
            default:
            {
                break;
            }
            } // switch by loss

        }

        Rcpp::List out;
        out["y_cast"] = Rcpp::wrap(yqt);
        out["y_cast_all"] = Rcpp::wrap(ycast);
        out["y"] = Rcpp::wrap(y);
        out["y_loss"] = Rcpp::wrap(y_loss);
        out["y_loss_all"] = Rcpp::wrap(y_loss_all);
        out["y_covered_all"] = Rcpp::wrap(y_covered_all);
        out["y_width_all"] = Rcpp::wrap(y_width_all);

        
        return out;
    }

    static void forecast_error(
        double &y_loss_all,
        double &y_cover,
        double &y_width,
        const arma::mat &psi, // (nT + 1) x nsample
        const arma::vec &y,   // (nT + 1) x 1
        const Model &model,
        const std::string &loss_func = "quadratic",
        const bool &verbose = VERBOSE)
    {
        const unsigned int nT = y.n_elem - 1;
        const unsigned int nsample = psi.n_cols;

        arma::mat ycast(nT + 1, nsample, arma::fill::zeros);
        arma::mat y_err_cast(nT + 1, nsample, arma::fill::zeros);

        for (unsigned int i = 0; i < nsample; i++)
        {
            Rcpp::checkUserInterrupt();
            arma::vec psi_vec = psi.col(i);
            arma::vec hpsi_vec = GainFunc::psi2hpsi<arma::vec>(psi_vec, model.ftrans); // (nT + 1) x 1
            arma::vec ft_vec(nT + 1, arma::fill::zeros); // (nT + 1) x 1
            ft_vec.at(1) = TransFunc::func_ft(
                1, y, ft_vec, hpsi_vec, model.dlag, model.ftrans);

            for (unsigned int t = 1; t < nT; t++)
            {
                arma::vec hpsi_tmp = hpsi_vec;
                hpsi_tmp.at(t + 1) = hpsi_tmp.at(t);
                // psi_cast.at(t + 1, i) = psi_tmp.at(t + 1);
                // psi_err_cast.at(t + 1, i) = psi.at(t + 1, i) - psi_cast.at(t + 1, i);

                arma::vec ft_tmp = ft_vec;
                ft_tmp.at(t + 1) = TransFunc::func_ft(
                    t + 1, y, ft_tmp, hpsi_tmp, model.dlag, model.ftrans);
                double eta = ft_tmp.at(t + 1);
                if (model.seas.period > 0)
                {
                    eta += arma::as_scalar(model.seas.X.col(t + 1).t() * model.seas.val);
                }
                // ft_cast.at(t + 1, i) = ft_tmp.at(t + 1);

                ycast.at(t + 1, i) = LinkFunc::ft2mu(eta, model.flink);
                y_err_cast.at(t + 1, i) = y.at(t + 1) - ycast.at(t + 1, i);
            }

            if (verbose)
            {
                Rcpp::Rcout << "\rForecast error: " << i + 1 << "/" << nsample;
            }
        }

        if (verbose)
        {
            Rcpp::Rcout << std::endl;
        }

        arma::vec y_cover_cast(nT, arma::fill::zeros);
        arma::vec y_width_cast = y_cover_cast;
        for (unsigned int t = 1; t < nT; t++)
        {
            arma::rowvec ycast_tmp = ycast.row(t + 1); // 1 x nsample
            double ymin = arma::min(ycast_tmp);
            double ymax = arma::max(ycast_tmp);
            double ytrue = y.at(t + 1);

            double covered = (ytrue >= ymin && ytrue <= ymax) ? 1. : 0.;
            y_cover_cast.at(t) = covered;
            y_width_cast.at(t) = std::abs(ymax - ymin);
        }

        y_cover = arma::mean(y_cover_cast.tail(nT - 1)) * 100.;
        y_width = arma::mean(y_width_cast.tail(nT - 1));

        arma::vec y_loss(nT + 1, arma::fill::zeros);

        y_loss_all = 0;

        std::map<std::string, AVAIL::Loss> loss_list = AVAIL::loss_list;
        switch (loss_list[tolower(loss_func)])
        {
        case AVAIL::L1: // mae
        {
            arma::mat ytmp = arma::abs(y_err_cast);
            y_loss = arma::mean(ytmp, 1);
            y_loss_all = arma::mean(y_loss.tail(nT - 1));
            break;
        }
        case AVAIL::L2: // rmse
        {
            arma::mat ytmp = arma::square(y_err_cast);
            y_loss = arma::mean(ytmp, 1);
            y_loss_all = arma::mean(y_loss.tail(nT - 1));
            y_loss = arma::sqrt(y_loss);
            y_loss_all = std::sqrt(y_loss_all);
            break;
        }
        default:
        {
            break;
        }
        }

        return;
    }


    /**
     * @brief Posterior predictive residuals.
     * 
     * @param psi 
     * @param y 
     * @param model 
     * @param loss_func 
     * @param verbose 
     * @return Rcpp::List 
     */
    static Rcpp::List fitted_error(
        const arma::mat &psi, // (nT + 1) x nsample
        const arma::vec &y,      // (nT + 1) x 1
        const Model &model,
        const std::string &loss_func = "quadratic",
        const bool &verbose = VERBOSE)
    {
        const unsigned int nT = y.n_elem - 1;
        const unsigned int nsample = psi.n_cols;
        arma::mat residual(nT + 1, nsample, arma::fill::zeros);
        arma::mat yhat(nT + 1, nsample, arma::fill::zeros);

        for (unsigned int i = 0; i < nsample; i ++)
        {
            Rcpp::checkUserInterrupt();

            arma::vec ft(nT + 1, arma::fill::zeros);
            arma::vec psi_tmp = psi.col(i);
            arma::vec hpsi_tmp = GainFunc::psi2hpsi<arma::vec>(psi_tmp, model.fgain);
            for (unsigned int t = 1; t <= nT; t++)
            {
                ft.at(t) = TransFunc::func_ft(
                    t, y, ft, hpsi_tmp, model.dlag, model.ftrans);
                double eta = ft.at(t);
                if (model.seas.period > 0)
                {
                    eta += arma::as_scalar(model.seas.X.col(t).t() * model.seas.val);
                }

                yhat.at(t, i) = LinkFunc::ft2mu(eta, model.flink);
                residual.at(t, i) = y.at(t) - yhat.at(t, i);
            }

            if (verbose)
            {
                Rcpp::Rcout << "\rFitted error: " << i + 1 << "/" << nsample;
            }
        }

        if (verbose)
        {
            Rcpp::Rcout << std::endl;
        }
        
        arma::vec y_loss(nT + 1, arma::fill::zeros);
        double y_loss_all = 0.;
        std::map<std::string, AVAIL::Loss> loss_list = AVAIL::loss_list;
        switch (loss_list[tolower(loss_func)])
        {
        case AVAIL::L1: // mae
        {
            arma::mat ytmp = arma::abs(residual);
            y_loss = arma::mean(ytmp, 1);
            y_loss_all = arma::mean(y_loss.tail(nT - 1));
            break;
        }
        case AVAIL::L2: // rmse
        {
            arma::mat ytmp = arma::square(residual);
            y_loss = arma::mean(ytmp, 1);
            y_loss_all = arma::mean(y_loss.tail(nT - 1));
            y_loss = arma::sqrt(y_loss);
            y_loss_all = std::sqrt(y_loss_all);
            break;
        }
        default:
        {
            break;
        }
        }


        Rcpp::List out;
        arma::vec qprob = {0.025, 0.5, 0.975};

        arma::mat yhat_qt = arma::quantile(yhat, qprob, 1);
        out["yhat"] = Rcpp::wrap(yhat_qt);
        out["yhat_all"] = Rcpp::wrap(yhat);
        out["residual"] = Rcpp::wrap(residual);
        out["y_loss"] = Rcpp::wrap(y_loss);
        out["y_loss_all"] = y_loss_all;

        return out;
    }

    static void fitted_error(
        double &y_loss_all,
        const arma::mat &psi, // (nT + 1) x nsample
        const arma::vec &y,   // (nT + 1) x 1
        const Model &model,
        const std::string &loss_func = "quadratic",
        const bool &verbose = VERBOSE)
    {
        const unsigned int nT = y.n_elem - 1;
        const unsigned int nsample = psi.n_cols;
        arma::mat residual(nT + 1, nsample, arma::fill::zeros);
        arma::mat yhat(nT + 1, nsample, arma::fill::zeros);

        for (unsigned int i = 0; i < nsample; i++)
        {
            arma::vec ft(nT + 1, arma::fill::zeros);
            arma::vec psi_tmp = psi.col(i);
            arma::vec hpsi_tmp = GainFunc::psi2hpsi<arma::vec>(psi_tmp, model.fgain);
            for (unsigned int t = 1; t <= nT; t++)
            {
                ft.at(t) = TransFunc::func_ft(
                    t, y, ft, hpsi_tmp, model.dlag, model.ftrans);
                double eta = ft.at(t);
                if (model.seas.period > 0)
                {
                    eta += arma::as_scalar(model.seas.X.col(t).t() * model.seas.val);
                }

                yhat.at(t, i) = LinkFunc::ft2mu(eta, model.flink);
                residual.at(t, i) = y.at(t) - yhat.at(t, i);
            }

            if (verbose)
            {
                Rcpp::Rcout << "\rFitted error: " << i + 1 << "/" << nsample;
            }
        }

        if (verbose)
        {
            Rcpp::Rcout << std::endl;
        }

        arma::vec y_loss(nT + 1, arma::fill::zeros);
        y_loss_all = 0.;
        std::map<std::string, AVAIL::Loss> loss_list = AVAIL::loss_list;
        switch (loss_list[tolower(loss_func)])
        {
        case AVAIL::L1: // mae
        {
            arma::mat ytmp = arma::abs(residual);
            y_loss = arma::mean(ytmp, 1);
            y_loss_all = arma::mean(y_loss.tail(nT - 1));
            break;
        }
        case AVAIL::L2: // rmse
        {
            arma::mat ytmp = arma::square(residual);
            y_loss = arma::mean(ytmp, 1);
            y_loss_all = arma::mean(y_loss.tail(nT - 1));
            y_loss = arma::sqrt(y_loss);
            y_loss_all = std::sqrt(y_loss_all);
            break;
        }
        default:
        {
            break;
        }
        }
        return;
    }

    arma::vec wt2lambda(
        const arma::vec &y, // (nT + 1) x 1
        const arma::vec &wt, // (nT + 1) x 1
        const unsigned int &seasonal_period,
        const arma::mat &X, // period x (nT + 1)
        const arma::vec &seas) // period x 1, checked. ok.
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
            if (seasonal_period > 0)
            {
                eta += arma::as_scalar(X.col(t).t() * seas);
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
        case AVAIL::Dist::poisson:
        {
            dloglik_dlam = Poisson::dlogp_dlambda(lambda, yt);
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
     * @param eta 
     * @param psi 
     * @param nlag 
     * @param lag_dist 
     * @param lag_par1 
     * @param lag_par2 
     * @param dobs 
     * @param link_func 
     * @return arma::vec 
     * 
     * @note For iterative transfer function, we must use its EXACT sliding form.
     */
    static arma::vec dloglik_dpar(
        arma::vec &Fphi,
        const arma::vec &y, 
        const arma::vec &hpsi,
        const unsigned int &nlag,
        const std::string &lag_dist,
        const double &lag_par1,
        const double &lag_par2,
        const ObsDist &dobs,
        const Season &seas,
        const std::string &link_func)
    {
        Fphi.clear();
        Fphi = LagDist::get_Fphi(nlag, lag_dist, lag_par1, lag_par2);
        arma::mat dFphi_grad = LagDist::get_Fphi_grad(nlag, lag_dist, lag_par1, lag_par2);

        arma::mat grad(y.n_elem, 2, arma::fill::zeros);
        for (unsigned int t = 1; t < y.n_elem; t ++)
        {
            double eta = TransFunc::transfer_sliding(t, nlag, y, Fphi, hpsi);
            if (seas.period > 0)
            {
                eta += arma::as_scalar(seas.X.col(t).t() * seas.val);
            }
            double dll_deta = dloglik_deta(eta, y.at(t), dobs.par2, dobs.name, link_func);

            double deta_dpar1 = TransFunc::transfer_sliding(t, nlag, y, dFphi_grad.col(0), hpsi);
            double deta_dpar2 = TransFunc::transfer_sliding(t, nlag, y, dFphi_grad.col(1), hpsi);

            grad.at(t, 0) = dll_deta * deta_dpar1;
            grad.at(t, 1) = dll_deta * deta_dpar2;
        }

        #ifdef DGTF_DO_BOUND_CHECK
            bound_check<arma::mat>(grad, "Model::dloglik_dpar: grad");
        #endif
        arma::vec grad_out = arma::vectorise(arma::sum(grad, 0));
        return grad_out;
    }

    /**
     * @brief This one is for HVA.
     *
     * @param y
     * @param hpsi
     * @param model
     * @return arma::vec
     *
     * @note  Parameter must be first mapped to real line. For iterative transfer function, we must use its EXACT sliding form.
     */
    static arma::vec dloglik_dpar(
        const arma::vec &y,
        const arma::vec &hpsi,
        const Model &model)
    {
        std::map<std::string, TransFunc::Transfer> trans_list = TransFunc::trans_list;
        LagDist dlag = model.dlag;
        dlag.nL = (trans_list[model.ftrans] == TransFunc::Transfer::iterative) ? y.n_elem - 1 : dlag.nL;
        dlag.Fphi = LagDist::get_Fphi(dlag);
        arma::mat dFphi_grad = LagDist::get_Fphi_grad(dlag.nL, dlag.name, dlag.par1, dlag.par2);

        arma::mat grad(y.n_elem, 2, arma::fill::zeros);
        for (unsigned int t = 1; t < y.n_elem; t++)
        {
            double eta = TransFunc::transfer_sliding(t, dlag.nL, y, dlag.Fphi, hpsi);
            if (model.seas.period > 0)
            {
                eta += arma::as_scalar(model.seas.X.col(t).t() * model.seas.val);
            }
            double dll_deta = dloglik_deta(eta, y.at(t), model.dobs.par2, model.dobs.name, model.flink);

            // For iterative transfer function, we must use its EXACT sliding form.
            double deta_dpar1 = TransFunc::transfer_sliding(t, dlag.nL, y, dFphi_grad.col(0), hpsi);
            double deta_dpar2 = TransFunc::transfer_sliding(t, dlag.nL, y, dFphi_grad.col(1), hpsi);

            grad.at(t, 0) = dll_deta * deta_dpar1;
            grad.at(t, 1) = dll_deta * deta_dpar2;
        }

        #ifdef DGTF_DO_BOUND_CHECK
            bound_check<arma::mat>(grad, "Model::dloglik_dpar: grad");
        #endif
        arma::vec grad_out = arma::vectorise(arma::sum(grad, 0));
        return grad_out;
    }

    static double dlogp_dpar2_obs(
        const Model &model, 
        const arma::vec &y, 
        const arma::vec &hpsi, 
        const bool &jacobian = true)
    {
        double out = 0.;
        arma::vec ft(y.n_elem, arma::fill::zeros);
        for (unsigned int t = 1; t < y.n_elem; t++)
        {
            ft.at(t) = TransFunc::func_ft(t, y, ft, hpsi, model.dlag, model.ftrans);
            // double eta = TransFunc::transfer_sliding(t, model.dlag.nL, y, model.dlag.Fphi, hpsi);
            double eta = ft.at(t);
            if (model.seas.period > 0)
            {
                eta += arma::as_scalar(model.seas.X.col(t).t() * model.seas.val);
            }
            double lambda = LinkFunc::ft2mu(eta, model.flink);
            out += nbinomm::dlogp_dpar2(y.at(t), lambda, model.dobs.par2, jacobian);
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
    /**
     * @brief Expected state evolution equation for the DLM form model. Expectation of theta[t + 1] = g(theta[t]).
     *
     * @param model
     * @param theta_cur
     * @param ycur
     * @return arma::vec
     * 
     * 
     * @note Checked. OK.
     */
    static arma::vec func_gt(
        const std::string &ftrans,
        const std::string &fgain,
        const LagDist &dlag,
        // const Model &model,
        const arma::vec &theta_cur, // nP x 1, (psi[t], f[t-1], ..., f[t-r])
        const double &ycur,
        const unsigned int &seasonal_period = 0,
        const bool &season_in_state = false
    )
    {
        std::map<std::string, TransFunc::Transfer> trans_list = TransFunc::trans_list;
        const unsigned int nP = theta_cur.n_elem;
        arma::vec theta_next(nP, arma::fill::zeros); // nP x 1
        // theta_next.copy_size(theta_cur);

        unsigned int nr = nP - 1;
        if (season_in_state)
        {
            nr -= seasonal_period;
        }
        

        switch (trans_list[ftrans])
        {
        case TransFunc::Transfer::iterative:
        {
            double hpsi = GainFunc::psi2hpsi(theta_cur.at(0), fgain);
            theta_next.at(0) = theta_cur.at(0); // Expectation of random walk.
            theta_next.at(1) = TransFunc::transfer_iterative(
                theta_cur.subvec(1, nr), // f[t-1], ..., f[t-r]
                hpsi, ycur, dlag.par1, dlag.par2);

            theta_next.subvec(2, nr) = theta_cur.subvec(1, nr - 1);
            break;
        }
        default: // AVAIL::Transfer::sliding
        {
            // theta_next = model.transfer.G0 * theta_cur;
            theta_next.at(0) = theta_cur.at(0);
            theta_next.subvec(1, nr) = theta_cur.subvec(0, nr - 1);

            break;
        }
        }

        if (season_in_state)
        {
            if (seasonal_period == 1)
            {
                theta_next.at(nr + 1) = theta_cur.at(nr + 1);
            }
            else if (seasonal_period == 2)
            {
                // nP - 1 = nr + 2
                theta_next.at(nr + 1) = theta_cur.at(nr + 2);
                theta_next.at(nr + 2) = theta_cur.at(nr + 1);
            }
            else if (seasonal_period > 2)
            {
                theta_next.subvec(nr + 1, nP - 2) = theta_cur.subvec(nr + 2, nP - 1);
                theta_next.at(nP - 1) = theta_cur.at(nr + 1);
            }
        }
        

        #ifdef DGTF_DO_BOUND_CHECK
            bound_check<arma::vec>(theta_next, "func_gt: theta_next");
        #endif
        return theta_next;
    }

    /**
     * @brief A backward propagate from theta[t] to theta[t-1]
     *
     * @param ftrans
     * @param fgain
     * @param dlag
     * @param theta theta[t]
     * @param yprev y[t-1]
     * @return arma::vec, theta[t-1]
     *
     * @note
     * Backward propagation doesn't work for iterative transfer functions.
     * 
     * For an iterative transfer function, theta[t] = (psi[t+1],f[t],...,f[t+1-r]);
     * For a sliding transfer function, theta[t] = (psi[t],...,psi[t+1-nlag]).
     * 
     * The backward propagate probably is not gonna work for iterative transfer function.
     */
    static arma::vec func_backward_gt(
        const std::string &ftrans,
        const std::string &fgain,
        const LagDist &dlag,
        const arma::vec &theta, // nP x 1, theta[t]
        const double &yprev, // y[t-1]
        const double &eps = 0.,
        const unsigned int &seasonal_period = 0,
        const bool &season_in_state = false
    )
    {
        std::map<std::string, TransFunc::Transfer> trans_list = TransFunc::trans_list;
        const unsigned int nP = theta.n_elem;
        unsigned int nstate = nP;
        if (season_in_state)
        {
            nstate -= seasonal_period;
        }

        arma::vec theta_prev(nP, arma::fill::zeros); // nP x 1

        if (trans_list[ftrans] == TransFunc::Transfer::iterative)
        {
            /*
            From theta[t] to theta[t-1]
            <=>
            from (psi[t+1],f[t],...,f[t+1-r]) to (psi[t],f[t-1],...,f[t-r]).

            First, we need to get psi[t] = psi[t+1] + eps;
            Second, we need to get f[t-r]
            */
            theta_prev.subvec(1, nP - 2) = theta.subvec(2, nP - 1);

            arma::vec iter_coef = nbinom::iter_coef(dlag.par1, dlag.par2, true);
            iter_coef = arma::reverse(iter_coef); // r x 1, r = nP - 1
            double ctmp = arma::accu(iter_coef % theta.subvec(1, nP - 1));
            double coef_now = std::pow((dlag.par1 - 1) / dlag.par1, dlag.par2);

            double hpsi_bnd = -ctmp / yprev;
            hpsi_bnd /= coef_now;
            double psi_bnd = GainFunc::hpsi2psi(hpsi_bnd, fgain);
            double eps_bnd = psi_bnd - theta.at(0);

            bool in_bound = (static_cast<int>(dlag.par2) % 2 == 0) ? (eps > eps_bnd) : (eps < eps_bnd);
            if (!in_bound)
            {
                throw std::runtime_error("StateSpace::func_backward_gt: invalid eps leads a negative ft.");
            }

            theta_prev.at(0) = theta.at(0) + eps; // psi[t] = psi[t+1] + eps

            /*
            hpsi_bnd
            It is an lower bound if r is even, an upper bound otherwise.
            */

            double hpsi = GainFunc::psi2hpsi(theta_prev.at(0), fgain);
            double ft_old = ctmp + coef_now * hpsi * yprev;

            if (ft_old < 0)
            {
                throw std::invalid_argument("State::Space::func_backward_gt: ft must be non-negative");
            }

            theta_prev.at(nP - 1) = ft_old;
        }
        else
        {
            // theta[t] = K[t] * theta[t + 1] + w[t]
            theta_prev.subvec(0, nstate - 2) = theta.subvec(1, nstate - 1);
            theta_prev.at(nstate - 1) = theta.at(nstate - 1) + eps;
        }

        if (season_in_state)
        {
            if (seasonal_period == 1)
            {
                theta_prev.at(nstate) = theta.at(nstate);
            }
            else if (seasonal_period > 1)
            {
                theta_prev.at(nstate) = theta.at(nP - 1);
                for (unsigned int i = nstate + 1; i < nP; i++)
                {
                    theta_prev.at(i) = theta.at(i - 1);
                }
            }
        }
        

        return theta_prev;
    }



    static void simulate(
        arma::vec &y,
        arma::vec &lambda,
        arma::vec &ft,
        arma::mat &Theta,
        arma::vec &psi, // (ntime + 1) x 1
        Model &model,
        const unsigned int &ntime,
        const double &y0,
        const bool &full_rank = false)
    {
        std::map<std::string, TransFunc::Transfer> trans_list = TransFunc::trans_list;
        if (!model.dlag.truncated)
        {
            model.dlag.nL = ntime;
        }
        model.dlag.Fphi = LagDist::get_Fphi(model.dlag);
        arma::mat Gmat = TransFunc::init_Gt(
            model.nP, model.dlag, model.ftrans,
            model.seas.period, model.seas.in_state);

        if (model.seas.period > 0)
        {
            model.seas.X = Season::setX(ntime, model.seas.period, model.seas.P);
        }
        Season seas;

        Theta.set_size(model.nP, ntime + 1);
        Theta.zeros();

        y.set_size(ntime + 1);
        y.zeros();
        y.at(0) = y0;
        lambda = y;
        ft = y;

        psi = ErrDist::sample(model.derr, ntime, true);
        if (trans_list[model.ftrans] == TransFunc::Transfer::iterative)
        {
            Theta.at(0, 0) = psi.at(1);
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

            Theta.col(t) = StateSpace::func_gt(
                model.ftrans, model.fgain, model.dlag,
                Theta.col(t - 1), y.at(t - 1), 0, false);

            if (!model.derr.full_rank)
            {
                Theta.at(0, t) = psi.at(psi_idx);
            }
            else
            {
                arma::vec eps = arma::randn<arma::vec>(Theta.n_rows);
                arma::mat var_chol = arma::chol(model.derr.var);
                Theta.col(t) = Theta.col(t) + var_chol.t() * eps;
                psi.at(psi_idx) = Theta.at(0, t);
            }

            ft.at(t) = TransFunc::func_ft(
                model.ftrans, model.fgain, model.dlag,
                seas, t, Theta.col(t), y);

            double eta = ft.at(t);
            if (model.seas.period > 0)
            {
                eta += arma::as_scalar(model.seas.X.col(t).t() * model.seas.val);
            }

            lambda.at(t) = LinkFunc::ft2mu(eta, model.flink); // Checked. OK.
            y.at(t) = ObsDist::sample(lambda.at(t), model.dobs.par2, model.dobs.name);
        }

        return; // Checked. OK.
    }

    static Rcpp::List forecast(
        const arma::vec &y, // (nT + 1)
        const arma::cube &Theta_stored, // p x nsample x (nT + B)
        const arma::vec &W_stored, // nsample x 1
        const Model &model, 
        const unsigned int k = 1
    )
    {
        std::map<std::string, AVAIL::Dist> obs_list = ObsDist::obs_list;
        unsigned int nT = y.n_elem - 1;
        unsigned int nsample = Theta_stored.n_cols;

        arma::cube Theta_all = arma::zeros<arma::cube>(Theta_stored.n_rows, nsample, nT + 1 + k);
        Theta_all.head_slices(nT + 1) = Theta_stored.tail_slices(nT + 1);

        arma::mat yall(nT + 1 + k, nsample, arma::fill::zeros);
        for (unsigned int t = 0; t < nT + 1; t ++)
        {
            yall.row(t).fill(y.at(t));
        }

        Model mod = model;
        mod.seas.X = Season::setX(nT + k, mod.seas.period, mod.seas.P);

        for (unsigned int i = 0; i < nsample; i ++)
        {
            double Wsqrt = std::sqrt(W_stored.at(i));
            arma::vec yvec = yall.col(i);

            for (unsigned int t = 0; t < k; t ++)
            {
                unsigned int idx = t + nT;
                double ynow = yvec.at(idx); // (nT + 1 + k) x 1
                arma::vec theta_now = Theta_all.slice(idx).col(i);
                arma::vec theta_next = StateSpace::func_gt(mod.ftrans, mod.fgain, mod.dlag, theta_now, ynow, mod.seas.period, mod.seas.in_state);
                if (Wsqrt > 0)
                {
                    theta_next.at(0) += R::rnorm(0., Wsqrt);
                }
                // arma::vec theta_next = StateSpace::func_state_propagate(model, theta_now, ynow, Wsqrt, false);
                double ft_next = TransFunc::func_ft(mod.ftrans, mod.fgain, mod.dlag, mod.seas, idx + 1, theta_next, yvec);
                double lambda = LinkFunc::ft2mu(ft_next, mod.flink);
                double ynext = 0.;

                switch (obs_list[mod.dobs.name])
                {
                case AVAIL::Dist::nbinomm:
                {
                    ynext = nbinomm::sample(lambda, mod.dobs.par2);
                    break;
                }
                case AVAIL::Dist::poisson:
                {
                    ynext = Poisson::sample(lambda);
                    break;
                }
                default:
                    break;
                }

                yvec.at(idx + 1) = ynext;
                Theta_all.slice(idx + 1).col(i) = theta_next;
            }

            yall.col(i) = yvec;
        }

        Rcpp::List out;
        out["yall"] = Rcpp::wrap(yall);
        out["yfit"] = Rcpp::wrap(y);
        out["ypred"] = Rcpp::wrap(yall.tail_rows(k));

        // out["Theta_all"] = Rcpp::wrap(Theta_all);
        // out["Theta_pred"] = Rcpp::wrap(Theta_all.tail_slices(k));

        arma::mat psi_all = Theta_all.row_as_mat(0); // (nT + 1 + k) x nsample
        arma::vec qprob = {0.025, 0.5, 0.975};
        arma::mat psi_qt = arma::quantile(psi_all, qprob, 1);

        out["psi_all"] = Rcpp::wrap(psi_qt);
        
        arma::mat psi_pred = psi_qt.tail_rows(k);
        out["psi_pred"] = Rcpp::wrap(psi_pred);
        return out;
    } // func: forecast

    
    static Rcpp::List forecast_error(
        const arma::cube &theta, // p x nsample x (nT + 1)
        const arma::vec &y,   // (nT + 1) x 1
        const Model &model,
        const std::string &loss_func = "quadratic",
        const unsigned int &k = 1,
        const bool &verbose = VERBOSE,
        const Rcpp::Nullable<unsigned int> &start_time = R_NilValue,
        const Rcpp::Nullable<unsigned int> &end_time = R_NilValue)
    {
        const unsigned int nT = y.n_elem - 1;
        const unsigned int p = theta.n_rows;
        const unsigned int nsample = theta.n_cols;
        unsigned int tstart = std::max(k, model.nP);
        if (start_time.isNotNull())
        {
            tstart = Rcpp::as<unsigned int>(start_time);
        }

        unsigned int tend = nT - k;
        if (end_time.isNotNull())
        {
            tend = Rcpp::as<unsigned int>(end_time);
        }

        arma::cube ycast = arma::zeros<arma::cube>(nT + 1, nsample, k);
        arma::cube y_err_cast = arma::zeros<arma::cube>(nT + 1, nsample, k);//(nT+1) x nsample x k
        arma::mat y_cov_cast(nT + 1, k, arma::fill::zeros); // (nT + 1) x k
        arma::mat y_width_cast = y_cov_cast;

        for (unsigned int t = tstart; t < tend; t++)
        {
            Rcpp::checkUserInterrupt();

            for (unsigned int i = 0; i < nsample; i ++)
            {
                arma::vec theta_cur = theta.slice(t).col(i); // p x 1
                arma::vec ytmp = y;
                for (unsigned int j = 1; j <= k; j ++)
                {
                    arma::vec theta_next = func_gt(model.ftrans, model.fgain, model.dlag, theta_cur, ytmp.at(t + j - 1), model.seas.period, model.seas.in_state);
                    double ft_next = TransFunc::func_ft(model.ftrans, model.fgain, model.dlag, model.seas, t + j, theta_next, ytmp);
                    double lambda = LinkFunc::ft2mu(ft_next, model.flink);
                    ycast.at(t, i, j - 1) = lambda;

                    ytmp.at(t + j) = lambda;
                    theta_cur = theta_next;

                    y_err_cast.at(t, i, j - 1) = y.at(t + j) - ytmp.at(t + j);
                }
            }

            for (unsigned int j = 0; j < k; j ++)
            {
                arma::vec ytmp = arma::vectorise(ycast.slice(j).row(t));
                double ymin = arma::min(ytmp);
                double ymax = arma::max(ytmp);
                double ytrue = y.at(t + j + 1);

                double covered = (ytrue >= ymin && ytrue <= ymax) ? 1. : 0.;
                y_cov_cast.at(t, j) = covered;
                y_width_cast.at(t, j) = std::abs(ymax - ymin);
            }


            if (verbose)
            {
                Rcpp::Rcout << "\rForecast error: " << t + 1 << "/" << tend;
            }
        }

        if (verbose)
        {
            Rcpp::Rcout << std::endl;
        }


        arma::vec qprob = {0.025, 0.5, 0.975};

        arma::mat y_loss(nT + 1, k, arma::fill::zeros);
        arma::vec y_loss_all(k, arma::fill::zeros);
        arma::vec y_covered_all = y_loss_all;
        arma::vec y_width_all = y_loss_all;
        arma::cube yqt = arma::zeros<arma::cube>(nT + 1, qprob.n_elem, k);

        std::map<std::string, AVAIL::Loss> loss_list = AVAIL::loss_list;
        for (unsigned int j = 0; j < k; j ++)
        {
            arma::mat ycast_qt = arma::quantile(ycast.slice(j), qprob, 1);
            yqt.slice(j) = ycast_qt;
            arma::mat ytmp = arma::abs(y_err_cast.slice(j)); // (nT + 1) x nsample

            arma::vec ycov_tmp = arma::vectorise(y_cov_cast.submat(arma::span(tstart, tend), arma::span(j)));
            y_covered_all.at(j) = arma::mean(ycov_tmp) * 100.;
            y_covered_all.at(j) *= 100.;

            ycov_tmp = arma::vectorise(y_width_cast.submat(arma::span(tstart, tend), arma::span(j)));
            y_width_all.at(j) = arma::mean(ycov_tmp);

            switch (loss_list[tolower(loss_func)])
            {
            case AVAIL::L1: // mae
            {
                arma::vec y_loss_tmp = arma::mean(ytmp, 1); // (nT + 1) x 1
                y_loss.col(j) = y_loss_tmp;

                arma::vec y_loss_tmp2 = y_loss_tmp.subvec(tstart, tend);
                y_loss_all.at(j) = arma::mean(y_loss_tmp2);
                break;
            }
            case AVAIL::L2: // rmse
            {
                ytmp = arma::square(ytmp); 
                arma::vec y_loss_tmp = arma::mean(ytmp, 1); // (nT + 1) x 1
                arma::vec y_loss_tmp2 = y_loss_tmp.subvec(tstart, tend);
                y_loss_all.at(j) = arma::mean(y_loss_tmp2);
                
                y_loss.col(j) = arma::sqrt(y_loss_tmp);
                y_loss_all.at(j) = std::sqrt(y_loss_all.at(j));
                break;
            }
            default:
            {
                break;
            }
            } // switch by loss
        }
        
        Rcpp::List out;
        out["y_cast"] = Rcpp::wrap(yqt);
        out["y_cast_all"] = Rcpp::wrap(ycast);
        out["y"] = Rcpp::wrap(y);
        out["y_loss"] = Rcpp::wrap(y_loss);

        out["y_loss_all"] = Rcpp::wrap(y_loss_all);
        out["y_covered_all"] = Rcpp::wrap(y_covered_all);
        out["y_width_all"] = Rcpp::wrap(y_width_all);

        return out;
    }

    static void forecast_error(
        double &y_loss_all,
        double &y_cover,
        double &y_width,
        const arma::cube &theta, // p x nsample x (nT + 1)
        const arma::vec &y,      // (nT + 1) x 1
        const Model &model,
        const std::string &loss_func = "quadratic",
        const bool &verbose = VERBOSE)
    {
        const unsigned int nT = y.n_elem - 1;
        const unsigned int p = theta.n_rows;
        const unsigned int nsample = theta.n_cols;

        arma::mat ycast(y.n_elem, nsample, arma::fill::zeros);
        arma::mat y_err_cast(y.n_elem, nsample, arma::fill::zeros);
        arma::vec y_cover_cast(y.n_elem, arma::fill::zeros);
        arma::vec y_width_cast = y_cover_cast;

        for (unsigned int t = 1; t < nT; t++)
        {
            Rcpp::checkUserInterrupt();

            for (unsigned int i = 0; i < nsample; i++)
            {
                arma::vec theta_next = func_gt(model.ftrans, model.fgain, model.dlag, theta.slice(t).col(i), y.at(t), model.seas.period, model.seas.in_state);
                double ft_next = TransFunc::func_ft(model.ftrans, model.fgain, model.dlag, model.seas, t + 1, theta_next, y);
                double lambda = LinkFunc::ft2mu(ft_next, model.flink);
                ycast.at(t + 1, i) = lambda;

                y_err_cast.at(t + 1, i) = y.at(t + 1) - ycast.at(t + 1, i);
            }

            double ymin = arma::min(ycast.row(t + 1));
            double ymax = arma::max(ycast.row(t + 1));
            if (y.at(t + 1) >= ymin && y.at(t + 1) <= ymax)
            {
                y_cover_cast.at(t) = 1.;
            }
            y_width_cast.at(t) = std::abs(ymax - ymin);


            if (verbose)
            {
                Rcpp::Rcout << "\rForecast error: " << t + 1 << "/" << nT;
            }
        }

        y_cover = arma::mean(y_cover_cast.subvec(model.nP, nT - 2)) * 100.;
        y_width = arma::mean(y_width_cast.subvec(model.nP, nT - 2));

        if (verbose)
        {
            Rcpp::Rcout << std::endl;
        }

        arma::vec y_loss(y.n_elem, arma::fill::zeros);

        y_loss_all = 0;

        std::map<std::string, AVAIL::Loss> loss_list = AVAIL::loss_list;
        switch (loss_list[tolower(loss_func)])
        {
        case AVAIL::L1: // mae
        {
            arma::mat ytmp = arma::abs(y_err_cast);
            y_loss = arma::mean(ytmp, 1);
            y_loss_all = arma::mean(y_loss.tail(nT - 1));
            break;
        }
        case AVAIL::L2: // rmse
        {
            arma::mat ytmp = arma::square(y_err_cast);
            y_loss = arma::mean(ytmp, 1);
            y_loss_all = arma::mean(y_loss.tail(nT - 1));
            y_loss = arma::sqrt(y_loss);
            y_loss_all = std::sqrt(y_loss_all);
            break;
        }
        default:
        {
            break;
        }
        }

        return;
    }

    static Rcpp::List fitted_error(
        const arma::cube &theta, // p x nsample x (nT + 1)
        const arma::vec &y,      // (nT + 1) x 1
        const Model &model,
        const std::string &loss_func = "quadratic",
        const bool &verbose = VERBOSE)
    {
        const unsigned int nT = y.n_elem - 1;
        const unsigned int nsample = theta.n_cols;
        arma::mat residual(y.n_elem, nsample, arma::fill::zeros);
        arma::mat yhat(y.n_elem, nsample, arma::fill::zeros);

        for (unsigned int t = 1; t < y.n_elem; t++)
        {
            Rcpp::checkUserInterrupt();

            for (unsigned int i = 0; i < nsample; i ++)
            {                
                arma::vec th = theta.slice(t).col(i); // p x 1
                double ft = TransFunc::func_ft(model.ftrans, model.fgain, model.dlag, model.seas, t, th, y);
                double lambda = LinkFunc::ft2mu(ft, model.flink);
                yhat.at(t, i) = lambda;
                residual.at(t, i) = y.at(t) - yhat.at(t, i);
            }

            if (verbose)
            {
                Rcpp::Rcout << "\rFitted error: " << t << "/" << nT;
            }
        }

        if (verbose)
        {
            Rcpp::Rcout << std::endl;
        }

        arma::vec y_loss(y.n_elem, arma::fill::zeros);
        double y_loss_all = 0.;
        std::map<std::string, AVAIL::Loss> loss_list = AVAIL::loss_list;
        switch (loss_list[tolower(loss_func)])
        {
        case AVAIL::L1: // mae
        {
            arma::mat ytmp = arma::abs(residual);
            y_loss = arma::mean(ytmp, 1);
            y_loss_all = arma::mean(y_loss.tail(nT - 1));
            break;
        }
        case AVAIL::L2: // rmse
        {
            arma::mat ytmp = arma::square(residual);
            y_loss = arma::mean(ytmp, 1);
            y_loss_all = arma::mean(y_loss.tail(nT - 1));
            y_loss = arma::sqrt(y_loss);
            y_loss_all = std::sqrt(y_loss_all);
            break;
        }
        default:
        {
            break;
        }
        }


        Rcpp::List out;
        arma::vec qprob = {0.025, 0.5, 0.975};

        arma::mat yhat_qt = arma::quantile(yhat, qprob, 1);
        out["yhat"] = Rcpp::wrap(yhat_qt);
        out["residual"] = Rcpp::wrap(residual);
        out["y_loss"] = Rcpp::wrap(y_loss);
        out["y_loss_all"] = y_loss_all;

        return out;
    }

    static void fitted_error(
        double &y_loss_all,
        const arma::cube &theta, // p x nsample x (nT + 1)
        const arma::vec &y,      // (nT + 1) x 1
        const Model &model,
        const std::string &loss_func = "quadratic",
        const bool &verbose = VERBOSE)
    {
        const unsigned int nT = y.n_elem - 1;
        const unsigned int nsample = theta.n_cols;
        arma::mat residual(y.n_elem, nsample, arma::fill::zeros);
        arma::mat yhat(y.n_elem, nsample, arma::fill::zeros);

        for (unsigned int t = 1; t < y.n_elem; t++)
        {
            Rcpp::checkUserInterrupt();

            for (unsigned int i = 0; i < nsample; i++)
            {
                arma::vec th = theta.slice(t).col(i); // p x 1
                double ft = TransFunc::func_ft(model.ftrans, model.fgain, model.dlag, model.seas, t, th, y);
                double lambda = LinkFunc::ft2mu(ft, model.flink);
                yhat.at(t, i) = lambda;
                residual.at(t, i) = y.at(t) - yhat.at(t, i);
            }

            if (verbose)
            {
                Rcpp::Rcout << "\rFitted error: " << t << "/" << nT;
            }
        }

        if (verbose)
        {
            Rcpp::Rcout << std::endl;
        }

        arma::vec y_loss(y.n_elem, arma::fill::zeros);
        y_loss_all = 0.;
        std::map<std::string, AVAIL::Loss> loss_list = AVAIL::loss_list;
        switch (loss_list[tolower(loss_func)])
        {
        case AVAIL::L1: // mae
        {
            arma::mat ytmp = arma::abs(residual);
            y_loss = arma::mean(ytmp, 1);
            y_loss_all = arma::mean(y_loss.tail(nT - 1));
            break;
        }
        case AVAIL::L2: // rmse
        {
            arma::mat ytmp = arma::square(residual);
            y_loss = arma::mean(ytmp, 1);
            y_loss_all = arma::mean(y_loss.tail(nT - 1));
            y_loss = arma::sqrt(y_loss);
            y_loss_all = std::sqrt(y_loss_all);
            break;
        }
        default:
        {
            break;
        }
        }

        return;
    }
};

/**
 * @brief Mostly used in MCMC.
 * 
 */
class ApproxDisturbance
{
public:
    ApproxDisturbance()
    {
        nT = 200;
        gain_func = "softplus";

        Fn.set_size(nT, nT); Fn.zeros();
        f0.set_size(nT); f0.zeros();

        Fphi.set_size(nT + 1);
        Fphi.zeros();

        psi = Fphi;
        hpsi = psi;
        dhpsi = psi;

        return;
    }

    ApproxDisturbance(const unsigned int &ntime, const std::string &gain_func_name = "softplus")
    {
        nT = ntime;
        gain_func = gain_func_name;

        Fn.set_size(nT, nT); Fn.zeros();
        f0.set_size(nT); f0.zeros();

        Fphi.set_size(nT + 1);
        Fphi.zeros();

        wt = Fphi;
        psi = Fphi;
        hpsi = psi;
        dhpsi = psi;
    }

    void update_dlag(const LagDist &dlag)
    {
        unsigned int nlag = dlag.Fphi.n_elem;
        set_Fphi(dlag, nlag);
        return;
    }

    void update_dlag(const LagDist &dlag, const arma::vec &y)
    {
        unsigned int nlag = dlag.Fphi.n_elem;
        set_Fphi(dlag, nlag);

        update_f0(y);
        update_Fn(y);
        
        return;
    }

    void set_Fphi(const LagDist &dlag, const unsigned int &nlag) // (nT + 1)
    {
        Fphi.zeros();
        arma::vec tmp = LagDist::get_Fphi(nT, dlag.name, dlag.par1, dlag.par2); // nT x 1
        if (nlag < nT)
        {
            tmp.subvec(nlag, nT - 1).zeros();
        }
        Fphi.tail(nT) = tmp; // fill Fphi[1:nT], leave Fphi[0] = 0.
    }

    void set_Fphi(const arma::vec &Fphi_new)
    {
        unsigned int n_elem = Fphi_new.n_elem;
        Fphi.zeros(); // Fphi: (nT + 1) x 1
        Fphi.subvec(1, n_elem) = Fphi_new;
    }

    void set_psi(const arma::vec &psi_in)
    {
        psi = psi_in;
        hpsi = GainFunc::psi2hpsi(psi, gain_func);
        dhpsi = GainFunc::psi2dhpsi(psi, gain_func);
    }

    void set_wt(const arma::vec &wt_in)
    {
        wt = wt_in;
        psi = arma::cumsum(psi);
        hpsi = GainFunc::psi2hpsi(psi, gain_func);
        dhpsi = GainFunc::psi2dhpsi(psi, gain_func);
    }

    arma::vec getFphi(){return Fphi;}
    arma::mat get_Fn(){return Fn;}
    arma::vec get_f0(){return f0;}
    arma::vec get_psi(){return psi;}
    arma::vec get_hpsi(){return hpsi;}
    arma::vec get_dhpsi(){return dhpsi;}

    void update_by_psi(const arma::vec &y, const arma::vec &psi)
    {
        hpsi = GainFunc::psi2hpsi(psi, gain_func);
        dhpsi = GainFunc::psi2dhpsi(psi, gain_func);
        update_f0(y);
        update_Fn(y);
    }

    void update_by_wt(const arma::vec &y, const arma::vec &wt)
    {
        psi = arma::cumsum(wt);
        hpsi = GainFunc::psi2hpsi(psi, gain_func);
        dhpsi = GainFunc::psi2dhpsi(psi, gain_func);
        update_f0(y);
        update_Fn(y);
    }

    void update_data(const arma::vec &y)
    {
        update_f0(y);
        update_Fn(y);
    }

    /**
     * @brief Need to run `set_psi` before using this function. Fphi must be initialized.
     * 
     * @param t 
     * @param y 
     * @return arma::vec 
     */
    arma::vec get_increment_matrix_byrow( // Checked. OK.
        const unsigned int &t, // t = 1, ..., nT
        const arma::vec &y     // (nT + 1) X 1, y[0], y[1], ..., y[nT], only use the past values before t
        )
    {
        arma::vec phi = Fphi.subvec(1, t);
        // t x 1, phi[1], ..., phi[nlag], phi[nlag + 1], ..., phi[t]
        //      = phi[1], ..., phi[nlag],       0,       ..., 0 (at time t)
        arma::vec yt = y.subvec(0, t - 1);        // y[0], y[1], ..., y[t-1]
        arma::vec dhpsi_tmp = dhpsi.subvec(1, t); // t x 1, h'(psi[1]), ..., h'(psi[t])

        arma::vec increment_row = arma::reverse(yt % dhpsi_tmp); // t x 1
        increment_row = increment_row % phi;

        return increment_row;
    }

    void update_Fn( // Checked. OK.
        const arma::vec &y   // (nT + 1) x 1, y[0], ..., y[nT - 1], y[nT], only use the past values before each t and y[nT] is not used
        )
    {
        Fn.zeros();
        for (unsigned int t = 1; t <= nT; t++)
        {
            arma::vec Fnt = get_increment_matrix_byrow(t, y); // t x 1
            Fnt = arma::cumsum(Fnt);
            Fnt = arma::reverse(Fnt);

            Fn.submat(t - 1, 0, t - 1, t - 1) = Fnt.t();
        }

        #ifdef DGTF_DO_BOUND_CHECK
            bound_check<arma::mat>(Fn, "get_Fn: Fn");
        #endif
        return;
    }

    void update_f0( // Checked. OK.
        const arma::vec &y   // (nT + 1) x 1, only use the past values before each t
        )
    {
        f0.zeros();
        arma::vec h0 = hpsi - dhpsi % psi; // (nT + 1) x 1, h0[0] = 0

        for (unsigned int t = 1; t <= nT; t++)
        {
            arma::vec F0t = get_increment_matrix_byrow(t, y);
            double f0t = arma::accu(F0t);

            f0.at(t - 1) = f0t;
        }

        f0.at(0) = (f0.at(0) < EPS8) ? EPS8 : f0.at(0);
        #ifdef DGTF_DO_BOUND_CHECK
            bound_check<arma::vec>(f0, "get_f0: f0");
        #endif
        return;
    }

    /**
     * @brief Get the regressor eta[1], ..., eta[nT] as a function of {w[t]}, evolution errors of the latent state. If mu = 0, it is equal to only the transfer effect, f[1], ..., f[nT]. Must set Fphi before using this function.
     *
     * @param wt
     * @param y (nT + 1) x 1, only use the past values before each t
     * @return arma::vec, (f[1], ..., f[nT])
     */
    arma::vec get_eta_approx(const Season &seas)
    {
        arma::vec eta = f0 + Fn * wt.tail(nT); // nT x 1
        if (seas.period > 0)
        {
            arma::vec seas_reg = seas.X.t() * seas.val; // (nT + 1) x 1
            eta = eta + seas_reg.tail(eta.n_elem);
        }

        #ifdef DGTF_DO_BOUND_CHECK
            bound_check<arma::vec>(eta, "func_eta_approx: eta");
        #endif
        return eta;
    }


    void get_lambda_eta_approx(arma::vec &lambda, arma::vec &eta, Model &model, const arma::vec &y)
    {
        eta = get_eta_approx(model.seas);
        lambda = LinkFunc::ft2mu<arma::vec>(eta, model.flink, 0.);
        return;
    }



    static arma::vec func_Vt_approx( // Checked. OK.
        const arma::vec &lambda, // (nT + 1) x 1
        const ObsDist &obs_dist,
        const std::string &link_func)
    {
        std::map<std::string, AVAIL::Dist> obs_list = ObsDist::obs_list;
        std::map<std::string, LinkFunc::Func> link_list = LinkFunc::link_list;
        arma::vec Vt = lambda;

        switch (obs_list[obs_dist.name])
        {
        case AVAIL::Dist::poisson:
        {
            switch (link_list[tolower(link_func)])
            {
            case AVAIL::Func::identity:
            {
                Vt = lambda;
                break;
            }
            case AVAIL::Func::exponential:
            {
                // Vt = 1 / lambda = exp( - log(lambda) )
                Vt = -arma::log(arma::abs(lambda) + EPS);
                Vt = arma::exp(Vt);
                break;
            }
            default:
            {
                break;
            }
            }      // switch by link
            break; // Done case poisson
        }
        case AVAIL::Dist::nbinomm:
        {
            switch (link_list[tolower(link_func)])
            {
            case LinkFunc::Func::identity:
            {
                Vt = lambda % (lambda + obs_dist.par2);
                Vt = Vt / obs_dist.par2;
                break;
            }
            case LinkFunc::Func::exponential:
            {
                arma::vec nom = (lambda + obs_dist.par2);
                arma::vec denom = obs_dist.par2 * lambda;
                Vt = nom / denom;
                break;
            }
            default:
            {
                break;
            }
            }      // switch by link
            break; // case nbinom
        }
        default:
        {
        }
        } // switch by observation distribution.

        #ifdef DGTF_DO_BOUND_CHECK
            bound_check<arma::vec>(Vt, "func_Vt_approx: Vt", true, true);
        #endif
        Vt.clamp(EPS8, Vt.max());
        return Vt;
    }

    static double func_Vt_approx( // Checked. OK.
        const double &lambda,      // (nT + 1) x 1
        const ObsDist &obs_dist,
        const std::string &link_func)
    {
        std::map<std::string, AVAIL::Dist> obs_list = ObsDist::obs_list;
        std::map<std::string, LinkFunc::Func> link_list = LinkFunc::link_list;
        double Vt = lambda;

        switch (obs_list[obs_dist.name])
        {
        case AVAIL::Dist::poisson:
        {
            switch (link_list[tolower(link_func)])
            {
            case LinkFunc::Func::identity:
            {
                Vt = lambda;
                break;
            }
            case LinkFunc::Func::exponential:
            {
                // Vt = 1 / lambda = exp( - log(lambda) )
                Vt = -std::log(std::abs(lambda) + EPS);
                Vt = std::exp(Vt);
                break;
            }
            default:
            {
                break;
            }
            } // switch by link
            break; // Done case poisson
        }
        case AVAIL::Dist::nbinomm:
        {
            switch (link_list[tolower(link_func)])
            {
            case AVAIL::Func::identity:
            {
                Vt = lambda * (lambda + obs_dist.par2);
                Vt = Vt / obs_dist.par2;
                break;
            }
            case AVAIL::Func::exponential:
            {
                double nom = (lambda + obs_dist.par2);
                double denom = obs_dist.par2 * lambda;
                Vt = nom / denom;
                break;
            }
            default:
            {
                break;
            }
            } // switch by link
            break; // case nbinom
        }
        default:
        {
        }
        } // switch by observation distribution.

        #ifdef DGTF_DO_BOUND_CHECK
            bound_check(Vt, "func_Vt_approx<double>: Vt", true, true);
        #endif
        Vt = std::max(Vt, EPS);
        return Vt;
    }

    static arma::vec func_yhat(
        const arma::vec &y,
        const std::string &link_func)
    {
        std::map<std::string, LinkFunc::Func> link_list = LinkFunc::link_list;
        arma::vec yhat;

        switch (link_list[tolower(link_func)])
        {
        case LinkFunc::Func::identity:
        {
            yhat = y;
            break;
        }
        case LinkFunc::Func::exponential:
        {
            yhat = arma::log(arma::abs(y) + EPS);
            break;
        }
        default:
        {
            break;
        }
        } // switch by link

        #ifdef DGTF_DO_BOUND_CHECK
            bound_check(yhat, "func_yhat");
        #endif
        return yhat;
    }

private:
    unsigned int nT = 200;
    std::string gain_func = "softplus";
    

    arma::mat Fn;
    arma::vec f0;
    arma::vec Fphi;
    arma::vec wt, psi, hpsi, dhpsi;
};



#endif