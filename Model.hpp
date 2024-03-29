#pragma once
#ifndef MODEL_H
#define MODEL_H

#include <RcppArmadillo.h>
#include "ErrDist.hpp"
#include "TransFunc.hpp"
#include "ObsDist.hpp"
#include "LinkFunc.hpp"
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
    Dim &dim;
    ObsDist &dobs;
    TransFunc &transfer;
    LinkFunc &flink;
    ErrDist &derr;

    Model() : dim(_dim), dobs(_dobs), transfer(_transfer), flink(_flink), derr(_derr)
    {
        _dobs.init_default();
        _flink.init_default();
        _transfer.init_default();
        _derr.init_default();
        _dim.init_default();

        _transfer.dlag.get_Fphi(_dim.nL);

        return;
    }

    Model(
        const Dim &dim_,
        const std::string &obs_dist = "nbinom",
        const std::string &link_func = "identity",
        const std::string &gain_func = "softplus",
        const std::string &lag_dist = "lognorm",
        const std::string &err_dist = "gaussian",
        const Rcpp::NumericVector &obs_param = Rcpp::NumericVector::create(0., 30.),
        const Rcpp::NumericVector &lag_param = Rcpp::NumericVector::create(NB_KAPPA, NB_R),
        const Rcpp::NumericVector &err_param = Rcpp::NumericVector::create(0.01, 0.), // (W, w[0])
        const std::string &trans_func = "sliding") : dim(_dim), dobs(_dobs), transfer(_transfer), flink(_flink), derr(_derr)
    {
        _dim = dim_;
        _dobs.init(obs_dist, obs_param[0], obs_param[1]);
        _flink.init(link_func);
        _transfer.init(dim_, trans_func, gain_func, lag_dist, lag_param);

        _derr.init("gaussian", err_param[0], err_param[1]);
        return;
    }


    Model(const Rcpp::List &settings)  : dim(_dim), dobs(_dobs), transfer(_transfer), flink(_flink), derr(_derr)
    {
        init(settings);
    }

    void init(const Rcpp::List &settings)
    {
        Rcpp::List model_settings = settings["model"];
        std::string obs_dist, link_func, trans_func, gain_func, lag_dist, err_dist;
        init_model(
            obs_dist, link_func, trans_func,
            gain_func, lag_dist, err_dist,
            model_settings);

        Rcpp::List dim_settings = settings["dim"];
        unsigned int nlag, ntime;
        bool truncated, regressor_baseline;
        init_dim(nlag, ntime, truncated, regressor_baseline, dim_settings);

        Rcpp::List param_settings = settings["param"];
        Rcpp::NumericVector lag_param, obs_param, err_param;
        init_param(obs_param, lag_param, err_param, param_settings);

        // if (regressor_baseline)
        // {
        //     obs_param[0] = 0.;
        // }

        _dim.init(ntime, nlag, lag_param[1], regressor_baseline);
        _dobs.init(obs_dist, obs_param[0], obs_param[1]);
        _flink.init(link_func);
        _transfer.init(_dim, trans_func, gain_func, lag_dist, lag_param);
        _derr.init("gaussian", err_param[0], err_param[1]);

        return;
    }

    void init_model(
        std::string &obs_dist,
        std::string &link_func,
        std::string &trans_func,
        std::string &gain_func,
        std::string &lag_dist,
        std::string &err_dist,
        const Rcpp::List &model_settings)
    {
        Rcpp::List model = model_settings;
        if (model.containsElementNamed("obs_dist"))
        {
            obs_dist = tolower(Rcpp::as<std::string>(model["obs_dist"]));
        }
        else
        {
            obs_dist = "nbinom";
        }

        if (model.containsElementNamed("link_func"))
        {
            link_func = tolower(Rcpp::as<std::string>(model["link_func"]));
        }
        else
        {
            link_func = "identity";
        }

        if (model.containsElementNamed("trans_func"))
        {
            trans_func = tolower(Rcpp::as<std::string>(model["trans_func"]));
        }
        else
        {
            trans_func = "sliding";
        }

        if (model.containsElementNamed("gain_func"))
        {
            gain_func = tolower(Rcpp::as<std::string>(model["gain_func"]));
        }
        else
        {
            gain_func = "softplus";
        }

        if (model.containsElementNamed("lag_dist"))
        {
            lag_dist = tolower(Rcpp::as<std::string>(model["lag_dist"]));
        }
        else
        {
            lag_dist = "lognorm";
        }

        if (model.containsElementNamed("err_dist"))
        {
            err_dist = tolower(Rcpp::as<std::string>(model["err_dist"]));
        }
        else
        {
            err_dist = "gaussian";
        }
    }

    void init_dim(
        unsigned int &nlag,
        unsigned int &ntime,
        bool &truncated,
        bool &regressor_baseline,
        const Rcpp::List &dim_settings)
    {
        Rcpp::List dm = dim_settings;

        if (dm.containsElementNamed("nlag"))
        {
            nlag = Rcpp::as<unsigned int>(dm["nlag"]);
        }
        else
        {
            nlag = 10;
        }

        if (dm.containsElementNamed("ntime"))
        {
            ntime = Rcpp::as<unsigned int>(dm["ntime"]);
        }
        else
        {
            ntime = 200;
        }

        if (nlag >= ntime)
        {
            nlag = ntime;
            truncated = false;
        }
        else if (dm.containsElementNamed("truncated"))
        {
            truncated = Rcpp::as<bool>(dm["truncated"]);
        }
        else
        {
            truncated = true;
        }

        if (dm.containsElementNamed("regressor_baseline"))
        {
            regressor_baseline = Rcpp::as<bool>(dm["regressor_baseline"]);
        }
        else
        {
            regressor_baseline = false;
        }
        return;
    }

    void init_param(
        Rcpp::NumericVector &obs,
        Rcpp::NumericVector &lag,
        Rcpp::NumericVector &err,
        const Rcpp::List &param_settings)
    {
        Rcpp::List param = param_settings;
        if (param.containsElementNamed("obs"))
        {
            obs = Rcpp::as<Rcpp::NumericVector>(param["obs"]);
        }
        else
        {
            obs = Rcpp::NumericVector::create(0., 30.);
        }

        if (param.containsElementNamed("lag"))
        {
            lag = Rcpp::as<Rcpp::NumericVector>(param["lag"]);
        }
        else
        {
            lag = Rcpp::NumericVector::create(1.4, 0.3);
        }

        if (param.containsElementNamed("err"))
        {
            err = Rcpp::as<Rcpp::NumericVector>(param["err"]);
        }
        else
        {
            err = Rcpp::NumericVector::create(0.01, 0.);
        }
    }


    Rcpp::List info()
    {
        Rcpp::List model;
        model["obs_dist"] = _dobs.name;
        model["link_func"] = _flink.name;
        model["trans_func"] = _transfer.name;
        model["gain_func"] = _transfer.fgain.name;
        model["lag_dist"] = _transfer.dlag.name;
        model["err_dist"] = _derr.name;

        Rcpp::List param;
        param["obs"] = Rcpp::NumericVector::create(_dobs.par1, _dobs.par2);
        param["lag"] = Rcpp::NumericVector::create(_transfer.dlag.par1, _transfer.dlag.par2);
        param["err"] = Rcpp::NumericVector::create(_derr.par1, _derr.par2);

        Rcpp::List dm;
        dm["nlag"] = _dim.nL;
        dm["ntime"] = _dim.nT;
        dm["truncated"] = _dim.truncated;
        dm["regressor_baseline"] = _dim.regressor_baseline;

        Rcpp::List out;
        out["model"] = model;
        out["param"] = param;
        out["dim"] = dm;

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

        Rcpp::List param_settings;
        param_settings["obs"] = Rcpp::NumericVector::create(0., 30.);
        param_settings["lag"] = Rcpp::NumericVector::create(1.4, 0.3);
        param_settings["err"] = Rcpp::NumericVector::create(0.01, 0.);

        Rcpp::List dim_settings;
        dim_settings["nlag"] = 10;
        dim_settings["ntime"] = 200;
        dim_settings["truncated"] = true;
        dim_settings["regressor_baseline"] = false;

        Rcpp::List settings;
        settings["model"] = model_settings;
        settings["param"] = param_settings;
        settings["dim"] = dim_settings;

        return settings;
    }

    


    void update_dobs(const double &value, const unsigned int &iloc)
    {
        if (iloc == 0)
        {
            // // update mu0
            // if (!_dim.regressor_baseline)
            // {
                _dobs.update_par1(value);
            // }
            // else
            // {
            //     _dobs.update_par1(0.);
            // }
        }
        else
        {
            _dobs.update_par2(value);
        }
    }

    void update_dlag(const double &value, const unsigned int &iloc)
    {
        if (iloc == 0)
        {
            _transfer.dlag.update_par1(value);
        }
        else
        {
            _transfer.dlag.update_par2(value);
        }
    }

    void set_dim(
        const unsigned int &ntime,
        const unsigned int &nlag = 0,
        const bool &add_regressor_baseline = false)
    {
        _dim.init(ntime, nlag, _transfer.dlag.par2, add_regressor_baseline);
        return;
    }


    /**
     * @brief Simulate from the DGTF model from the scratch using the transfer function form.
     * 
     * @return arma::vec 
     */
    arma::vec simulate(const double &y0 = 0.)
    {
        lambda.set_size(_dim.nT + 1);
        lambda.zeros();

        // Sample psi[t].
        _derr.sample(_dim.nT, true);
        _transfer.fgain.update_psi(_derr.psi);

        // Get h(psi[t])
        arma::vec hpsi = GainFunc::psi2hpsi<arma::vec>(
            _transfer.fgain.psi, 
            _transfer.fgain.name); // Checked. OK.
        _transfer.fgain.update_hpsi(hpsi); // Checked. OK.

        // Get phi[1], ..., phi[nL]
        _transfer.dlag.get_Fphi(_dim.nL); // Checked. OK.

        arma::vec y(_dim.nT + 1, arma::fill::zeros);
        y.at(0) = y0;
        for (unsigned int t = 1; t < _dim.nT + 1; t++)
        {
            double ft = _transfer.transfer_sliding(t, y); // Checked. OK.
            double mu = LinkFunc::ft2mu(ft, _flink.name, _dobs.par1);
            lambda.at(t) = mu;
            y.at(t) = _dobs.sample(mu); // Checked. OK.
        }

        return y; // Checked. OK.
    }

    arma::vec lambda;


    /**
     * @brief Simulation via transfer function form
     * 
     * @param psi 
     * @param nlag 
     * @param y0 
     * @param gain_func 
     * @param lag_dist 
     * @param link_func 
     * @param obs_dist 
     * @param lag_param 
     * @param obs_param 
     * @return arma::vec 
     */
    static arma::vec simulate(
        const arma::vec &psi, // (ntime + 1) x 1
        const unsigned int &nlag,
        const double &y0 = 0.,
        const std::string &gain_func = "softplus",
        const std::string &lag_dist = "lognormal",
        const std::string &link_func = "identity",
        const std::string &obs_dist = "nbinom",
        const Rcpp::NumericVector &lag_param = Rcpp::NumericVector::create(1.4, 0.3),
        const Rcpp::NumericVector &obs_param = Rcpp::NumericVector::create(0., 30))
    {
        ObsDist dobs(obs_dist, obs_param[0], obs_param[1]);
        unsigned int ntime = psi.n_elem - 1;
        arma::vec hpsi = GainFunc::psi2hpsi<arma::vec>(psi, gain_func);                 // Checked. OK.
        arma::vec Fphi = LagDist::get_Fphi(nlag, lag_dist, lag_param[0], lag_param[1]); // Checked. OK.

        arma::vec y(ntime + 1, arma::fill::zeros);
        y.at(0) = y0;

        for (unsigned int t = 1; t < (ntime + 1); t++)
        {
            double ftt = TransFunc::transfer_sliding(t, nlag, y, Fphi, hpsi); // Checked. OK.
            double mu = LinkFunc::ft2mu(ftt, link_func, obs_param[0]);        // Checked. OK.
            y.at(t) = ObsDist::sample(mu, obs_param[1], obs_dist);            // Checked. OK.
        }

        return y; // Checked. OK.
    }


    /**
     * @brief Forecasting using the transfer function form.
     * 
     * @param y 
     * @param psi_stored 
     * @param W_stored 
     * @param dim 
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
        const Dim &dim,
        const TransFunc &transfer,
        const std::string &link_func = "identity",
        const double mu0 = 0.,
        const unsigned int &k = 1)
    {
        std::map<std::string, AVAIL::Dist> obs_list = AVAIL::obs_list;
        unsigned int nsample = psi_stored.n_cols;
        unsigned int nT = y.n_elem - 1;


        arma::mat psi_all(nT + 1 + k, nsample, arma::fill::zeros);
        arma::mat yall(nT + 1 + k, nsample, arma::fill::zeros);
        arma::mat ft(nT + 1 + k, nsample, arma::fill::zeros);

        for (unsigned int i = 0; i < nsample; i++)
        {
            arma::vec ft_vec(nT + 1 + k, arma::fill::zeros); // (nT + 1) x 1
            arma::vec psi_vec(nT + 1 + k, arma::fill::zeros);
            psi_vec.head(nT + 1) = psi_stored.col(i);

            for (unsigned int t = 1; t < (nT + 1); t++)
            {
                yall.at(t, i) = y.at(t);
                ft_vec.at(t) = TransFunc::func_ft(
                    t, y, ft_vec, psi_vec, dim,
                    transfer.dlag,
                    transfer.fgain.name,
                    transfer.name);
            }

            ft.col(i) = ft_vec;
            psi_all.col(i) = psi_vec;
        }


        for (unsigned int i = 0; i < nsample; i++)
        {
            arma::vec psi_vec = psi_all.col(i);
            arma::vec ft_vec = ft.col(i);
            arma::vec yvec = yall.col(i); // (nT + 1 + k) x 1

            yvec.head(y.n_elem) = y;

            double Wsqrt = std::sqrt(W_stored.at(i));

            for (unsigned int t = 0; t < k; t++)
            {
                unsigned int idx = t + nT; // idx of old
                psi_vec.at(idx + 1) = psi_vec.at(idx);
                ft_vec.at(idx + 1) = TransFunc::func_ft(
                    idx + 1, yvec, ft_vec, psi_vec, dim,
                    transfer.dlag,
                    transfer.fgain.name,
                    transfer.name);
                
                double lambda = LinkFunc::ft2mu(ft_vec.at(idx + 1), link_func, mu0);
                yvec.at(idx + 1) = lambda;
            }

            yall.col(i) = yvec;
            ft.col(i) = ft_vec;
            psi_all.col(i) = psi_vec;
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
        std::map<std::string, AVAIL::Dist> obs_list = AVAIL::obs_list;
        unsigned int nsample = psi_stored.n_cols;
        unsigned int nT = y.n_elem - 1;

        arma::mat psi_all(nT + 1 + k, nsample, arma::fill::zeros);
        arma::mat yall(nT + 1 + k, nsample, arma::fill::zeros);
        arma::mat ft(nT + 1 + k, nsample, arma::fill::zeros);

        for (unsigned int i = 0; i < nsample; i++)
        {
            arma::vec ft_vec(nT + 1 + k, arma::fill::zeros); // (nT + 1) x 1
            arma::vec psi_vec(nT + 1 + k, arma::fill::zeros);
            psi_vec.head(nT + 1) = psi_stored.col(i);

            for (unsigned int t = 1; t < (nT + 1); t++)
            {
                yall.at(t, i) = y.at(t);
                ft_vec.at(t) = TransFunc::func_ft(
                    t, y, ft_vec, psi_vec, model.dim,
                    model.transfer.dlag,
                    model.transfer.fgain.name,
                    model.transfer.name);
            }

            ft.col(i) = ft_vec;
            psi_all.col(i) = psi_vec;
        }

        for (unsigned int i = 0; i < nsample; i++)
        {
            arma::vec psi_vec = psi_all.col(i);
            arma::vec ft_vec = ft.col(i);
            arma::vec yvec = yall.col(i);

            double Wsqrt = std::sqrt(W_stored.at(i));

            for (unsigned int t = 0; t < k; t++)
            {
                unsigned int idx = t + nT; // idx of old
                if (random_sample)
                {
                    psi_vec.at(idx + 1) = R::rnorm(psi_vec.at(idx), Wsqrt);
                }
                else
                {
                    psi_vec.at(idx + 1) = psi_vec.at(idx);
                }
                
                ft_vec.at(idx + 1) = TransFunc::func_ft(
                    idx + 1, yvec, ft_vec, psi_vec, model.dim,
                    model.transfer.dlag,
                    model.transfer.fgain.name,
                    model.transfer.name);

                double lambda = LinkFunc::ft2mu(ft_vec.at(idx + 1), model.flink.name, model.dobs.par1);

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
            ft.col(i) = ft_vec;
            psi_all.col(i) = psi_vec;
        }

        
        arma::vec qprob = {0.025, 0.5, 0.975};
        arma::mat yqt = arma::quantile(yall, qprob, 1);

        Rcpp::List out;
        out["y"] = Rcpp::wrap(yqt);
        out["yall"] = Rcpp::wrap(yall);
        out["yfit"] = Rcpp::wrap(y);
        out["ypred"] = Rcpp::wrap(yall.tail_rows(k));

        arma::mat psi_qt = arma::quantile(psi_all, qprob, 1);
        out["psi_all"] = Rcpp::wrap(psi_qt);

        arma::mat psi_pred = psi_qt.tail_rows(k);
        out["psi_pred"] = Rcpp::wrap(psi_pred);

        return out;
    }


    static Rcpp::List forecast_error(
        const arma::mat &psi, // (nT + 1) x nsample
        const arma::vec &y,   // (nT + 1) x 1
        const Model &model,
        const std::string &loss_func = "quadratic",
        const unsigned int &k = 1,
        const bool &verbose = VERBOSE)
    {
        unsigned int nsample = psi.n_cols;

        // arma::cube psi_cast = arma::zeros<arma::cube>(model.dim.nT + 1, nsample, k);
        // arma::cube ft_cast = arma::zeros<arma::cube>(model.dim.nT + 1, nsample, k);
        arma::cube ycast = arma::zeros<arma::cube>(model.dim.nT + 1, nsample, k);
        arma::cube y_err_cast = arma::zeros<arma::cube>(model.dim.nT + 1, nsample, k);
        // arma::cube psi_err_cast = arma::zeros<arma::cube>(model.dim.nT + 1, nsample, k);

        // for (unsigned int j = 0; j < k; j ++)
        // {
        //     psi_cast.slice(j).row(1) = psi.row(1); // 1 x nsample
        // }


        for (unsigned int i = 0; i < nsample; i ++)
        {
            arma::vec psi_vec = psi.col(i); // (nT + 1) x 1
            arma::vec ft_vec(model.dim.nT + 1, arma::fill::zeros);

            for (unsigned int t = 0; t < model.dim.nT; t++)
            {
                ft_vec.at(t + 1) = TransFunc::func_ft(
                    t + 1, y, ft_vec, psi_vec, model.dim,
                    model.transfer.dlag,
                    model.transfer.fgain.name,
                    model.transfer.name);
            }

            for (unsigned int t = 1; t < model.dim.nT; t++)
            {
                Rcpp::checkUserInterrupt();

                arma::vec psi_tmp = psi_vec;
                arma::vec ft_tmp = ft_vec;
                arma::vec ytmp = y;

                // psi_cast.at(t, i, 0) = psi_tmp.at(t);
                // ft_cast.at(t, i, 0) = ft_tmp.at(t);
                ycast.at(t, i, 0) = ytmp.at(t);

                unsigned int ncast = std::min(k, model.dim.nT - t);
                for (unsigned int j = 1; j <= ncast; j++)
                {
                    psi_tmp.at(t + j) = psi_tmp.at(t + j - 1);

                    ft_tmp.at(t + j) = TransFunc::func_ft(
                        t + j, ytmp, ft_tmp, psi_tmp, model.dim,
                        model.transfer.dlag,
                        model.transfer.fgain.name,
                        model.transfer.name);
                    ytmp.at(t + j) = LinkFunc::ft2mu(ft_tmp.at(t + j), model.flink.name, model.dobs.par1);

                    // psi_cast.at(t, i, j - 1) = psi_tmp.at(t + j);
                    // ft_cast.at(t, i, j - 1) = ft_tmp.at(t + j);
                    ycast.at(t, i, j - 1) = ytmp.at(t + j);
                    y_err_cast.at(t, i, j - 1) = y.at(t + j) - ytmp.at(t + j);
                    // psi_err_cast.at(t, i, j - 1) = psi.at(t + j, i) - psi_tmp.at(t + j);
                }

                if (verbose)
                {
                    Rcpp::Rcout << "\rForecast error: " << t + 1 << "/" << model.dim.nT;
                }
            } // loop over time

            if (verbose)
            {
                Rcpp::Rcout << std::endl;
            }
        } // loop over nsample

        arma::mat y_cov_cast(model.dim.nT + 1, k, arma::fill::zeros); // (nT + 1) x k
        arma::mat y_width_cast = y_cov_cast;
        for (unsigned int t = 1; t < model.dim.nT; t++)
        {
            arma::mat ycast_tmp = ycast.row_as_mat(t); // k x nsample
            arma::vec ymin = arma::vectorise(arma::min(ycast_tmp, 1));
            arma::vec ymax = arma::vectorise(arma::max(ycast_tmp, 1));

            unsigned int ncast = std::min(k, model.dim.nT - t);
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

        // arma::cube psi_qt = arma::zeros<arma::cube>(model.dim.nT + 1, qprob.n_elem, k);
        // arma::mat psi_loss(model.dim.nT + 1, k, arma::fill::zeros);
        // arma::vec psi_loss_all(k, arma::fill::zeros);

        arma::mat y_loss(model.dim.nT + 1, k, arma::fill::zeros);
        arma::vec y_loss_all(k, arma::fill::zeros);
        arma::vec y_covered_all = y_loss_all;
        arma::vec y_width_all = y_loss_all;
        arma::cube yqt = arma::zeros<arma::cube>(model.dim.nT + 1, qprob.n_elem, k);

        unsigned int tstart = std::max(k, model.dim.nP);
        for (unsigned int i = 0; i < k; i ++)
        {
            unsigned int tend = model.dim.nT - i - 1;

            arma::mat ycast_qt = arma::quantile(ycast.slice(i), qprob, 1);
            yqt.slice(i) = ycast_qt;
            arma::mat y_loss_tmp0 = y_err_cast.slice(i);
            arma::mat y_loss_tmp = y_loss_tmp0.submat(tstart, 0, tend, nsample - 1);
            y_loss_tmp = arma::abs(y_loss_tmp);
            arma::vec ytmp;

            arma::vec ycov_tmp = arma::vectorise(y_cov_cast(arma::span(tstart, tend), arma::span(i)));
            y_covered_all.at(i) = arma::mean(ycov_tmp) * 100.;

            ycov_tmp = arma::vectorise(y_width_cast(arma::span(tstart, tend), arma::span(i)));
            y_width_all.at(i) = arma::mean(ycov_tmp);

            // arma::mat psi_cast_qt = arma::quantile(psi_cast.slice(i), qprob, 1);
            // psi_qt.slice(i) = psi_cast_qt;
            // arma::mat psi_loss_tmp0 = psi_err_cast.slice(i); // (nT + 1) x nsample
            // arma::mat psi_loss_tmp = psi_loss_tmp0.submat(1, 0, model.dim.nT - i - 1, nsample - 1);
            // psi_loss_tmp = arma::abs(psi_loss_tmp);                                     // (nT - i) x nsample
            // arma::vec psi_tmp;
            

            switch (loss_list[tolower(loss_func)])
            {
            case AVAIL::L1: // mae
            {
                // psi_tmp = arma::mean(psi_loss_tmp, 1); // // (nT - i) x 1
                // psi_loss.submat(1, i, model.dim.nT - i - 1, i) = psi_tmp;
                // psi_loss_all.at(i) = arma::mean(psi_tmp);

                ytmp = arma::mean(y_loss_tmp, 1);
                y_loss.submat(tstart, i, tend, i) = ytmp;
                y_loss_all.at(i) = arma::mean(ytmp);
                

                break;
            }
            case AVAIL::L2: // rmse
            {
                // psi_loss_tmp = arma::square(psi_loss_tmp); // (nT - i) x nsample

                // psi_tmp = arma::mean(psi_loss_tmp, 1); // (nT - i) x 1
                // psi_loss.submat(1, i, model.dim.nT - i - 1, i) = arma::sqrt(psi_tmp);

                // psi_loss_all.at(i) = arma::mean(psi_tmp);
                // psi_loss_all.at(i) = std::sqrt(psi_loss_all.at(i));

                y_loss_tmp = arma::square(y_loss_tmp);

                ytmp = arma::mean(y_loss_tmp, 1);      // (nT - i) x 1
                y_loss.submat(tstart, i, tend, i) = arma::sqrt(ytmp);

                y_loss_all.at(i) = arma::mean(ytmp);
                y_loss_all.at(i) = std::sqrt(y_loss_all.at(i));
                break;
            }
            default:
            {
                break;
            }
            } // switch by loss

        }

        Rcpp::List out;
        // out["psi_cast"] = Rcpp::wrap(psi_qt);
        // out["psi_cast_all"] = Rcpp::wrap(psi_cast);
        // out["psi"] = Rcpp::wrap(psi);
        // out["psi_loss"] = Rcpp::wrap(psi_loss);
        // out["psi_loss_all"] = Rcpp::wrap(psi_loss_all);

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
        unsigned int nsample = psi.n_cols;

        // arma::mat psi_cast(model.dim.nT + 1, nsample, arma::fill::zeros);
        // arma::mat ft_cast(model.dim.nT + 1, nsample, arma::fill::zeros);
        arma::mat ycast(model.dim.nT + 1, nsample, arma::fill::zeros);
        arma::mat y_err_cast(model.dim.nT + 1, nsample, arma::fill::zeros);
        
        // arma::mat psi_err_cast(model.dim.nT + 1, nsample, arma::fill::zeros);

        // psi_cast.row(1) = psi.row(1);


        for (unsigned int i = 0; i < nsample; i++)
        {
            Rcpp::checkUserInterrupt();

            arma::vec psi_vec = psi.col(i);                        // (nT + 1) x 1
            arma::vec ft_vec(model.dim.nT + 1, arma::fill::zeros); // (nT + 1) x 1
            ft_vec.at(1) = TransFunc::func_ft(
                1, y, ft_vec, psi_vec, model.dim,
                model.transfer.dlag,
                model.transfer.fgain.name,
                model.transfer.name);

            for (unsigned int t = 1; t < model.dim.nT; t++)
            {
                arma::vec psi_tmp = psi_vec;
                psi_tmp.at(t + 1) = psi_tmp.at(t);
                // psi_cast.at(t + 1, i) = psi_tmp.at(t + 1);
                // psi_err_cast.at(t + 1, i) = psi.at(t + 1, i) - psi_cast.at(t + 1, i);

                arma::vec ft_tmp = ft_vec;
                ft_tmp.at(t + 1) = TransFunc::func_ft(
                    t + 1, y, ft_tmp, psi_tmp, model.dim,
                    model.transfer.dlag,
                    model.transfer.fgain.name,
                    model.transfer.name);
                // ft_cast.at(t + 1, i) = ft_tmp.at(t + 1);

                ycast.at(t + 1, i) = LinkFunc::ft2mu(ft_tmp.at(t + 1), model.flink.name, model.dobs.par1);
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

        arma::vec y_cover_cast(model.dim.nT, arma::fill::zeros);
        arma::vec y_width_cast = y_cover_cast;
        for (unsigned int t = 1; t < model.dim.nT; t++)
        {
            arma::rowvec ycast_tmp = ycast.row(t + 1); // 1 x nsample
            double ymin = arma::min(ycast_tmp);
            double ymax = arma::max(ycast_tmp);
            double ytrue = y.at(t + 1);

            double covered = (ytrue >= ymin && ytrue <= ymax) ? 1. : 0.;
            y_cover_cast.at(t) = covered;
            y_width_cast.at(t) = std::abs(ymax - ymin);
        }

        y_cover = arma::mean(y_cover_cast.tail(model.dim.nT - 1)) * 100.;
        y_width = arma::mean(y_width_cast.tail(model.dim.nT - 1));

        arma::vec y_loss(model.dim.nT + 1, arma::fill::zeros);
        // arma::vec psi_loss(model.dim.nT + 1, arma::fill::zeros);

        y_loss_all = 0;

        std::map<std::string, AVAIL::Loss> loss_list = AVAIL::loss_list;
        switch (loss_list[tolower(loss_func)])
        {
        case AVAIL::L1: // mae
        {
            arma::mat ytmp = arma::abs(y_err_cast);
            y_loss = arma::mean(ytmp, 1);
            y_loss_all = arma::mean(y_loss.tail(model.dim.nT - 1));
            break;
        }
        case AVAIL::L2: // rmse
        {
            arma::mat ytmp = arma::square(y_err_cast);
            y_loss = arma::mean(ytmp, 1);
            y_loss_all = arma::mean(y_loss.tail(model.dim.nT - 1));
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
        const arma::mat &psi, // (nT + 1) x nsample
        const arma::vec &y,      // (nT + 1) x 1
        const Model &model,
        const std::string &loss_func = "quadratic",
        const bool &verbose = VERBOSE)
    {
        unsigned int nsample = psi.n_cols;
        arma::mat residual(model.dim.nT + 1, nsample, arma::fill::zeros);
        arma::mat yhat(model.dim.nT + 1, nsample, arma::fill::zeros);

        for (unsigned int i = 0; i < nsample; i ++)
        {
            Rcpp::checkUserInterrupt();

            arma::vec ft(model.dim.nT + 1, arma::fill::zeros);
            arma::vec psi_tmp = psi.col(i);
            for (unsigned int t = 1; t <= model.dim.nT; t ++)
            {
                ft.at(t) = TransFunc::func_ft(
                    t, y, ft, psi_tmp, model.dim,
                    model.transfer.dlag, 
                    model.transfer.fgain.name,
                    model.transfer.name);

                yhat.at(t, i) = LinkFunc::ft2mu(ft.at(t), model.flink.name, model.dobs.par1);
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
        
        arma::vec y_loss(model.dim.nT + 1, arma::fill::zeros);
        double y_loss_all = 0.;
        std::map<std::string, AVAIL::Loss> loss_list = AVAIL::loss_list;
        switch (loss_list[tolower(loss_func)])
        {
        case AVAIL::L1: // mae
        {
            arma::mat ytmp = arma::abs(residual);
            y_loss = arma::mean(ytmp, 1);
            y_loss_all = arma::mean(y_loss.tail(model.dim.nT - 1));
            break;
        }
        case AVAIL::L2: // rmse
        {
            arma::mat ytmp = arma::square(residual);
            y_loss = arma::mean(ytmp, 1);
            y_loss_all = arma::mean(y_loss.tail(model.dim.nT - 1));
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
        unsigned int nsample = psi.n_cols;
        arma::mat residual(model.dim.nT + 1, nsample, arma::fill::zeros);
        arma::mat yhat(model.dim.nT + 1, nsample, arma::fill::zeros);

        for (unsigned int i = 0; i < nsample; i++)
        {
            arma::vec ft(model.dim.nT + 1, arma::fill::zeros);
            arma::vec psi_tmp = psi.col(i);
            for (unsigned int t = 1; t <= model.dim.nT; t++)
            {
                ft.at(t) = TransFunc::func_ft(
                    t, y, ft, psi_tmp, model.dim,
                    model.transfer.dlag,
                    model.transfer.fgain.name,
                    model.transfer.name);

                yhat.at(t, i) = LinkFunc::ft2mu(ft.at(t), model.flink.name, model.dobs.par1);
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

        arma::vec y_loss(model.dim.nT + 1, arma::fill::zeros);
        y_loss_all = 0.;
        std::map<std::string, AVAIL::Loss> loss_list = AVAIL::loss_list;
        switch (loss_list[tolower(loss_func)])
        {
        case AVAIL::L1: // mae
        {
            arma::mat ytmp = arma::abs(residual);
            y_loss = arma::mean(ytmp, 1);
            y_loss_all = arma::mean(y_loss.tail(model.dim.nT - 1));
            break;
        }
        case AVAIL::L2: // rmse
        {
            arma::mat ytmp = arma::square(residual);
            y_loss = arma::mean(ytmp, 1);
            y_loss_all = arma::mean(y_loss.tail(model.dim.nT - 1));
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
        const arma::vec &wt) // (nT + 1) x 1, checked. ok.
    {
        arma::vec psi = arma::cumsum(wt);
        _transfer.fgain.update_psi(psi);
        _transfer.fgain.psi2hpsi();
        
        arma::vec ft(psi.n_elem, arma::fill::zeros);
        arma::vec lambda = ft;
        for (unsigned int t = 1; t <= dim.nT; t++)
        {
            ft.at(t) = _transfer.func_ft(t, y, ft);
            lambda.at(t) = LinkFunc::ft2mu(ft.at(t), _flink.name, _dobs.par1);
        }


        return lambda;
    }

private:
    ObsDist _dobs; // Observation distribution
    TransFunc _transfer;
    LinkFunc _flink;
    ErrDist _derr;
    Dim _dim;
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
     */
    static arma::vec func_gt( // Checked. OK.
        const Model &model,
        const arma::vec &theta_cur, // nP x 1, (psi[t], f[t-1], ..., f[t-r])
        const double &ycur
    )
    {
        arma::vec theta_next;
        theta_next.copy_size(theta_cur);

        unsigned int nr = model.dim.nP - 1;
        if (model.dim.regressor_baseline)
        {
            nr -= 1; // nP - 2
        }

        switch (model.transfer.trans_list[model.transfer.name])
        {
        case AVAIL::Transfer::iterative:
        {
            theta_next.at(0) = theta_cur.at(0); // Expectation of random walk.
            theta_next.at(1) = TransFunc::transfer_iterative(
                theta_cur.subvec(1, nr), // f[t-1], ..., f[t-r]
                theta_cur.at(0),                       // psi[t]
                ycur,                                  // y[t-1]
                model.transfer.name,
                model.transfer.dlag.par1,
                model.transfer.dlag.par2);

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

        if (model.dim.regressor_baseline)
        {
            theta_next.at(nr + 1) = std::max(theta_cur.at(nr + 1), EPS); // baseline: time-invariant
        }

        bound_check<arma::vec>(theta_next, "func_gt: theta_next");
        return theta_next;
    }

    static arma::vec func_state_propagate(
        const Model &model,
        const arma::vec &theta_now,
        const double &ynow,
        const double &Wsqrt,
        const bool &positive_noise = false)
    {
        arma::vec theta_next = func_gt(model, theta_now, ynow);

        double omega_next = 0.;
        if (Wsqrt > 0)
        {
            omega_next = R::rnorm(0., Wsqrt); // [Input] - Wsqrt
        }
        
        if (positive_noise)                      // t < Theta_now.n_rows
        {
            theta_next.at(0) += std::abs(omega_next);
        }
        else
        {
            theta_next.at(0) += omega_next;
        }
        return theta_next;
    }

    /**
     * @brief f[t]( theta[t] ) - maps state theta[t] to observation-level variable f[t].
     *
     * @param model
     * @param t  time index of theta_cur; yold.tail(nelem) = yall.subvec(t - nelem, t - 1);
     * @param theta_cur theta[t] = (psi[t], ..., psi[t+1 - nL]) or (psi[t+1], f[t], ..., f[t+1-r])
     * @param yall
     * @return double
     */
    static double func_ft(
        const Model &model,
        const int &t,               // t = 0, y[0] = 0, theta[0] = 0; t = 1, y[1], theta[1]; ...;  yold.tail(nelem) = yall.subvec(t - nelem, t - 1);
        const arma::vec &theta_cur, // theta[t] = (psi[t], ..., psi[t+1 - nL]) or (psi[t+1], f[t], ..., f[t+1-r])
        const arma::vec &yall       // We use y[t - nelem], ..., y[t-1]
    )
    {
        double ft_cur;
        if (model.transfer.trans_list[model.transfer.name] == AVAIL::sliding)
        {
            int nelem = std::min(t, (int)model.dim.nL); // min(t,nL)

            arma::vec yold(model.dim.nL, arma::fill::zeros);
            if (nelem > 1)
            {
                yold.tail(nelem) = yall.subvec(t - nelem, t - 1); // 0, ..., 0, y[t - nelem], ..., y[t-1]
            }
            else if (t > 0) // nelem = 1 at t = 1
            {
                yold.at(model.dim.nL - 1) = yall.at(t - 1);
            }

            yold = arma::reverse(yold); // y[t-1], ..., y[t-min(t,nL)]

            arma::vec ft_vec = model.transfer.dlag.Fphi; // nL x 1
            arma::vec th = theta_cur.head(model.dim.nL);
            arma::vec hpsi_cur = GainFunc::psi2hpsi<arma::vec>(th, model.transfer.fgain.name); // (h(psi[t]), ..., h(psi[t+1 - nL])), nL x 1
            arma::vec ftmp = yold % hpsi_cur; // nL x 1
            ft_vec = ft_vec % ftmp;

            ft_cur = arma::accu(ft_vec);
        } // sliding
        else
        {
            ft_cur = theta_cur.at(1);
        } // iterative

        if (model.dim.regressor_baseline && (theta_cur.at(model.dim.nP - 1) > 0.))
        {
            ft_cur += theta_cur.at(model.dim.nP - 1); // constant baseline is the last element of the state vector
        }

        try
        {
            bound_check(ft_cur, "func_ft: ft_cur", false, t > model.dim.nP);
        }
        catch(const std::exception& e)
        {
            std::cout << "\nft_cur = " << ft_cur << ", mu0 = " << theta_cur.at(model.dim.nP - 1) << " at t = " << t << std::endl;
            theta_cur.print("\n theta_cur:");
            throw std::invalid_argument(e.what());
        }
        
        
        return ft_cur;
    }

    static Rcpp::List simulate(
        const Model &model,
        const double &y0 = 0.,
        const Rcpp::Nullable<Rcpp::NumericVector> &theta_init = R_NilValue)
    {
        

        arma::vec wt(model.dim.nT + 1, arma::fill::zeros);
        if (model.derr.par1 > 0)
        {
            wt = ErrDist::sample(model.dim.nT, model.derr.par1, model.derr.par2, false);
        }
        wt.at(0) = model.derr.par2;
        arma::vec psi = arma::cumsum(wt);

        arma::mat wt_ss(model.dim.nP, model.dim.nT + 1, arma::fill::zeros);
        wt_ss.row(0) = wt.t();

        arma::vec theta0(model.dim.nP, arma::fill::zeros);
        if (theta_init.isNotNull())
        {
            theta0 = Rcpp::as<Rcpp::NumericVector>(theta_init);
        }

        arma::mat theta(model.dim.nP, model.dim.nT + 1, arma::fill::zeros);
        theta.col(0) = theta0;

        arma::vec y(model.dim.nT + 1, arma::fill::zeros);
        arma::vec ft(model.dim.nT + 1, arma::fill::zeros);
        arma::vec lambda(model.dim.nT + 1, arma::fill::zeros);

        double mu0 = 0.;
        if (!model.dim.regressor_baseline) { mu0 = model.dobs.par1; }

        for (unsigned int t = 1; t < model.dim.nT + 1; t ++)
        {
            theta.col(t) = func_gt(model, theta.col(t - 1), y.at(t - 1)) + wt_ss.col(t);
            ft.at(t) = func_ft(model, t, theta.col(t), y);
            lambda.at(t) = LinkFunc::ft2mu(ft.at(t), model.flink.name, mu0);
            y.at(t) = ObsDist::sample(lambda.at(t), model.dobs.par2, model.dobs.name);
        }

        Rcpp::List output;
        output["y"] = Rcpp::wrap(y);
        output["theta"] = Rcpp::wrap(theta);
        output["psi"] = Rcpp::wrap(psi);
        output["wt"] = Rcpp::wrap(wt);
        output["lambda"] = Rcpp::wrap(lambda);

        return output; // Checked. OK.
    }

    static Rcpp::List forecast(
        const arma::vec &y, // (nT + 1)
        const arma::cube &Theta_stored, // p x nsample x (nT + B)
        const arma::vec &W_stored, // nsample x 1
        const Model &model, 
        const unsigned int k = 1
    )
    {
        std::map<std::string, AVAIL::Dist> obs_list = AVAIL::obs_list;
        unsigned int nT = y.n_elem - 1;
        unsigned int nsample = Theta_stored.n_cols;

        arma::cube Theta_all = arma::zeros<arma::cube>(Theta_stored.n_rows, nsample, nT + 1 + k);
        Theta_all.head_slices(nT + 1) = Theta_stored.tail_slices(nT + 1);

        arma::mat yall(nT + 1 + k, nsample, arma::fill::zeros);
        for (unsigned int t = 0; t < nT + 1; t ++)
        {
            yall.row(t).fill(y.at(t));
        }

        double mu0 = 0.;
        if (!model.dim.regressor_baseline) { mu0 = model.dobs.par1; }


        for (unsigned int i = 0; i < nsample; i ++)
        {
            double Wsqrt = std::sqrt(W_stored.at(i));
            arma::vec yvec = yall.col(i);

            for (unsigned int t = 0; t < k; t ++)
            {
                unsigned int idx = t + nT;
                double ynow = yvec.at(idx);
                arma::vec theta_now = Theta_all.slice(idx).col(i);

                arma::vec theta_next = StateSpace::func_state_propagate(model, theta_now, ynow, Wsqrt, false);
                double ft_next = StateSpace::func_ft(model, idx + 1, theta_next, yvec);
                double lambda = LinkFunc::ft2mu(ft_next, model.flink.name, mu0);
                double ynext = 0.;
                switch (obs_list[model.dobs.name])
                {
                case AVAIL::Dist::nbinomm:
                {
                    ynext = nbinomm::sample(lambda, model.dobs.par2);
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
        const bool &verbose = VERBOSE)
    {
        unsigned int p = theta.n_rows;
        unsigned int nsample = theta.n_cols;
        unsigned int tstart = std::max(k, model.dim.nP);

        // arma::cube psi_cast = arma::zeros<arma::cube>(model.dim.nT + 1, nsample, k);
        arma::cube ycast = arma::zeros<arma::cube>(model.dim.nT + 1, nsample, k);
        arma::cube y_err_cast = arma::zeros<arma::cube>(model.dim.nT + 1, nsample, k); // (nT + 1) x nsample x k
        arma::mat y_cov_cast(model.dim.nT + 1, k, arma::fill::zeros); // (nT + 1) x k
        // arma::cube psi_err_cast = arma::zeros<arma::cube>(model.dim.nT + 1, nsample, k);
        arma::mat y_width_cast = y_cov_cast;

        double mu0 = 0.;
        if (!model.dim.regressor_baseline) { mu0 = model.dobs.par1; }

        for (unsigned int t = 1; t < model.dim.nT; t++)
        {
            Rcpp::checkUserInterrupt();

            unsigned int ncast = std::min(k, model.dim.nT - t);
            for (unsigned int i = 0; i < nsample; i ++)
            {
                arma::vec theta_cur = theta.slice(t).col(i); // p x 1
                arma::vec ytmp = y;
                for (unsigned int j = 1; j <= ncast; j ++)
                {
                    arma::vec theta_next = func_gt(
                        model, theta_cur, ytmp.at(t + j - 1));
                    // psi_cast.at(t, i, j - 1) = theta_next.at(0);

                    double ft_next = func_ft(model, t + j, theta_next, ytmp);
                    double lambda = LinkFunc::ft2mu(ft_next, model.flink.name, mu0);
                    ycast.at(t, i, j - 1) = lambda;

                    ytmp.at(t + j) = lambda;
                    theta_cur = theta_next;

                    y_err_cast.at(t, i, j - 1) = y.at(t + j) - ytmp.at(t + j);
                    // psi_err_cast.at(t, i, j - 1) = theta.at(0, i, t + j) - theta_next.at(0);
                }
            }

            for (unsigned int j = 0; j < ncast; j ++)
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
                Rcpp::Rcout << "\rForecast error: " << t + 1 << "/" << model.dim.nT;
            }
        }

        if (verbose)
        {
            Rcpp::Rcout << std::endl;
        }


        arma::vec qprob = {0.025, 0.5, 0.975};

        arma::mat y_loss(model.dim.nT + 1, k, arma::fill::zeros);
        arma::vec y_loss_all(k, arma::fill::zeros);
        arma::vec y_covered_all = y_loss_all;
        arma::vec y_width_all = y_loss_all;
        arma::cube yqt = arma::zeros<arma::cube>(model.dim.nT + 1, qprob.n_elem, k);

        // arma::mat psi_loss(model.dim.nT + 1, k, arma::fill::zeros);
        // arma::vec psi_loss_all(k, arma::fill::zeros);
        // arma::cube psi_qt = arma::zeros<arma::cube>(model.dim.nT + 1, qprob.n_elem, k);
        


        std::map<std::string, AVAIL::Loss> loss_list = AVAIL::loss_list;

        for (unsigned int j = 0; j < k; j ++)
        {
            unsigned int tend = model.dim.nT - j - 1;
            arma::mat ycast_qt = arma::quantile(ycast.slice(j), qprob, 1);
            yqt.slice(j) = ycast_qt;
            arma::mat ytmp = arma::abs(y_err_cast.slice(j)); // (nT + 1) x nsample

            // arma::mat psi_cast_qt = arma::quantile(psi_cast.slice(j), qprob, 1);
            // psi_qt.slice(j) = psi_cast_qt;
            // arma::mat psi_tmp = arma::abs(psi_err_cast.slice(j));
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

                // arma::vec psi_loss_tmp = arma::mean(psi_tmp, 1); // (nT + 1) x 1
                // psi_loss.col(j) = psi_loss_tmp;
                // arma::vec psi_loss_tmp2 = psi_loss_tmp.subvec(1, model.dim.nT - j - 1);
                // psi_loss_all.at(j) = arma::mean(psi_loss_tmp2);

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

                // psi_tmp = arma::square(psi_tmp);
                // arma::vec psi_loss_tmp = arma::mean(psi_tmp, 1);
                // arma::vec psi_loss_tmp2 = psi_loss_tmp.subvec(1, model.dim.nT - j - 1);
                // psi_loss_all.at(j) = arma::mean(psi_loss_tmp2);

                // psi_loss.col(j) = arma::sqrt(psi_loss_tmp);
                // psi_loss_all.at(j) = std::sqrt(psi_loss_all.at(j));


                break;
            }
            default:
            {
                break;
            }
            } // switch by loss
        }
        
        Rcpp::List out;
        // out["psi_cast"] = Rcpp::wrap(psi_qt);
        // out["psi_cast_all"] = Rcpp::wrap(psi_cast);
        // arma::mat psi = theta.row_as_mat(0);
        // out["psi"] = Rcpp::wrap(psi);
        // out["psi_loss"] = Rcpp::wrap(psi_loss);
        // out["psi_loss_all"] = Rcpp::wrap(psi_loss_all);

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
        unsigned int p = theta.n_rows;
        unsigned int nsample = theta.n_cols;

        arma::mat ycast(model.dim.nT + 1, nsample, arma::fill::zeros);
        arma::mat y_err_cast(model.dim.nT + 1, nsample, arma::fill::zeros);
        arma::vec y_cover_cast(model.dim.nT, arma::fill::zeros);
        arma::vec y_width_cast = y_cover_cast;

        double mu0 = 0.;
        if (!model.dim.regressor_baseline) { mu0 = model.dobs.par1; }

        for (unsigned int t = 1; t < model.dim.nT; t++)
        {
            Rcpp::checkUserInterrupt();

            for (unsigned int i = 0; i < nsample; i++)
            {
                arma::vec theta_next = func_gt(model, theta.slice(t).col(i), y.at(t));
                double ft_next = func_ft(model, t + 1, theta_next, y);
                double lambda = LinkFunc::ft2mu(ft_next, model.flink.name, mu0);
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
                Rcpp::Rcout << "\rForecast error: " << t + 1 << "/" << model.dim.nT;
            }
        }

        y_cover = arma::mean(y_cover_cast.subvec(model.dim.nP, model.dim.nT - 2)) * 100.;
        y_width = arma::mean(y_width_cast.subvec(model.dim.nP, model.dim.nT - 2));

        if (verbose)
        {
            Rcpp::Rcout << std::endl;
        }

        arma::vec y_loss(model.dim.nT + 1, arma::fill::zeros);

        y_loss_all = 0;

        std::map<std::string, AVAIL::Loss> loss_list = AVAIL::loss_list;
        switch (loss_list[tolower(loss_func)])
        {
        case AVAIL::L1: // mae
        {
            arma::mat ytmp = arma::abs(y_err_cast);
            y_loss = arma::mean(ytmp, 1);
            y_loss_all = arma::mean(y_loss.tail(model.dim.nT - 1));
            break;
        }
        case AVAIL::L2: // rmse
        {
            arma::mat ytmp = arma::square(y_err_cast);
            y_loss = arma::mean(ytmp, 1);
            y_loss_all = arma::mean(y_loss.tail(model.dim.nT - 1));
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
        unsigned int nsample = theta.n_cols;
        arma::mat residual(model.dim.nT + 1, nsample, arma::fill::zeros);
        arma::mat yhat(model.dim.nT + 1, nsample, arma::fill::zeros);

        double mu0 = 0.;
        if (!model.dim.regressor_baseline) { mu0 = model.dobs.par1; }

        
        for (unsigned int t = 1; t <= model.dim.nT; t ++)
        {
            Rcpp::checkUserInterrupt();

            for (unsigned int i = 0; i < nsample; i ++)
            {                
                arma::vec th = theta.slice(t).col(i); // p x 1
                double ft = func_ft(model, t, th, y);
                double lambda = LinkFunc::ft2mu(ft, model.flink.name, mu0);
                yhat.at(t, i) = lambda;
                residual.at(t, i) = y.at(t) - yhat.at(t, i);
            }

            if (verbose)
            {
                Rcpp::Rcout << "\rFitted error: " << t << "/" << model.dim.nT;
            }
        }

        if (verbose)
        {
            Rcpp::Rcout << std::endl;
        }


        arma::vec y_loss(model.dim.nT + 1, arma::fill::zeros);
        double y_loss_all = 0.;
        std::map<std::string, AVAIL::Loss> loss_list = AVAIL::loss_list;
        switch (loss_list[tolower(loss_func)])
        {
        case AVAIL::L1: // mae
        {
            arma::mat ytmp = arma::abs(residual);
            y_loss = arma::mean(ytmp, 1);
            y_loss_all = arma::mean(y_loss.tail(model.dim.nT - 1));
            break;
        }
        case AVAIL::L2: // rmse
        {
            arma::mat ytmp = arma::square(residual);
            y_loss = arma::mean(ytmp, 1);
            y_loss_all = arma::mean(y_loss.tail(model.dim.nT - 1));
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
        unsigned int nsample = theta.n_cols;
        arma::mat residual(model.dim.nT + 1, nsample, arma::fill::zeros);
        arma::mat yhat(model.dim.nT + 1, nsample, arma::fill::zeros);

        double mu0 = 0.;
        if (!model.dim.regressor_baseline) { mu0 = model.dobs.par1; }


        for (unsigned int t = 1; t <= model.dim.nT; t++)
        {
            Rcpp::checkUserInterrupt();

            for (unsigned int i = 0; i < nsample; i++)
            {
                arma::vec th = theta.slice(t).col(i); // p x 1
                double ft = func_ft(model, t, th, y);
                double lambda = LinkFunc::ft2mu(ft, model.flink.name, mu0);
                yhat.at(t, i) = lambda;
                residual.at(t, i) = y.at(t) - yhat.at(t, i);
            }

            if (verbose)
            {
                Rcpp::Rcout << "\rFitted error: " << t << "/" << model.dim.nT;
            }
        }

        if (verbose)
        {
            Rcpp::Rcout << std::endl;
        }

        arma::vec y_loss(model.dim.nT + 1, arma::fill::zeros);
        y_loss_all = 0.;
        std::map<std::string, AVAIL::Loss> loss_list = AVAIL::loss_list;
        switch (loss_list[tolower(loss_func)])
        {
        case AVAIL::L1: // mae
        {
            arma::mat ytmp = arma::abs(residual);
            y_loss = arma::mean(ytmp, 1);
            y_loss_all = arma::mean(y_loss.tail(model.dim.nT - 1));
            break;
        }
        case AVAIL::L2: // rmse
        {
            arma::mat ytmp = arma::square(residual);
            y_loss = arma::mean(ytmp, 1);
            y_loss_all = arma::mean(y_loss.tail(model.dim.nT - 1));
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

    // private:
    //     arma::vec theta; // nP x 1
    //     arma::mat theta_series; // nP x (nT + 1)
    //     arma::vec theta_init;

    //     arma::vec psi; // (nT + 1) x 1

    //     Model model;
};

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

    void set_Fphi(const LagDist &dlag, const unsigned int &nlag)
    {
        Fphi.zeros();
        arma::vec tmp = LagDist::get_Fphi(nT, dlag.name, dlag.par1, dlag.par2);
        if (nlag < nT)
        {
            tmp.subvec(nlag, nT - 1).zeros();
        }
        Fphi.tail(nT) = tmp;
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

        bound_check<arma::mat>(Fn, "get_Fn: Fn");
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
        bound_check<arma::vec>(f0, "get_f0: f0");
        return;
    }

    /**
     * @brief Get the regressor eta[1], ..., eta[nT]. If mu = 0, it is equal to only the transfer effect, f[1], ..., f[nT]. Must set Fphi before using this function.
     *
     * @param wt
     * @param y (nT + 1) x 1, only use the past values before each t
     * @return arma::vec, (f[1], ..., f[nT])
     */
    arma::vec get_eta_approx(const double &mu0 = 0.)
    {
        arma::vec ft = mu0 + f0 + Fn * wt.tail(nT); // nT x 1
        bound_check(ft, "func_eta_approx: ft");
        return ft;
    }

    static arma::vec func_Vt_approx( // Checked. OK.
        const arma::vec &lambda, // (nT + 1) x 1
        const ObsDist &obs_dist,
        const std::string &link_func)
    {
        std::map<std::string, AVAIL::Dist> obs_list = AVAIL::obs_list;
        std::map<std::string, AVAIL::Func> link_list = AVAIL::link_list;
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
            case AVAIL::Func::identity:
            {
                Vt = lambda % (lambda + obs_dist.par2);
                Vt = Vt / obs_dist.par2;
                break;
            }
            case AVAIL::Func::exponential:
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

        bound_check<arma::vec>(Vt, "func_Vt_approx: Vt", true, true);
        Vt.elem(arma::find(Vt < EPS8)).fill(EPS8);
        return Vt;
    }

    static double func_Vt_approx( // Checked. OK.
        const double &lambda,      // (nT + 1) x 1
        const ObsDist &obs_dist,
        const std::string &link_func)
    {
        std::map<std::string, AVAIL::Dist> obs_list = AVAIL::obs_list;
        std::map<std::string, AVAIL::Func> link_list = AVAIL::link_list;
        double Vt = lambda;

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

        bound_check(Vt, "func_Vt_approx<double>: Vt", true, true);
        Vt = std::max(Vt, EPS);
        return Vt;
    }

    static arma::vec func_yhat(
        const arma::vec &y,
        const std::string &link_func)
    {
        std::map<std::string, AVAIL::Func> link_list = AVAIL::link_list;
        arma::vec yhat;

        switch (link_list[tolower(link_func)])
        {
        case AVAIL::Func::identity:
        {
            yhat = y;
            break;
        }
        case AVAIL::Func::exponential:
        {
            yhat = arma::log(arma::abs(y) + EPS);
            break;
        }
        default:
        {
            break;
        }
        } // switch by link

        bound_check(yhat, "func_yhat");
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