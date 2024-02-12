#pragma once
#ifndef MODEL_H
#define MODEL_H

#include <RcppArmadillo.h>
#include "ErrDist.hpp"
#include "TransFunc.hpp"
#include "LinkFunc.hpp"
#include "ObsDist.hpp"
#include "model_utils.h"
// [[Rcpp::plugins(cpp17)]]
// [[Rcpp::depends(RcppArmadillo)]]

/**
 * @brief Define a dynamic generalized transfer function (DGTF) model
 *
 * @param dobs observation distribution characterized by either (mu0, delta_nb) or (mu, sd2)
 *       - (mu0, delta_nb):
 *          (1) mu0: constant mean of the observed time series;
 *          (2) delta_nb: degress of freedom for negative-binomial observation distribution.
 *       - (mu, sd2): parameters for gaussian observation distribution.
 * @param dlag lag distribution characterized by either (kappa, r) for negative-binomial lags or (mu, sd2) for lognormal lags.
 *
 */
class Model
{
public:
    Model() : dobs(_dobs), transfer(_transfer), flink(_flink), derr(_derr)
    {
        _dobs.init_default();
        _flink.init_default();
        _transfer.init_default();
        _derr.init_default();

        return;
    }

    Model(
        const Dim &dim,
        const std::string &obs_dist,
        const std::string &link_func,
        const std::string &gain_func,
        const std::string &lag_dist,
        const Rcpp::NumericVector &obs_param = Rcpp::NumericVector::create(0., 30.),
        const Rcpp::NumericVector &lag_param = Rcpp::NumericVector::create(NB_KAPPA, NB_R),
        const Rcpp::NumericVector &err_param = Rcpp::NumericVector::create(0.01, 0.), // (W, w[0])
        const std::string &trans_func = "sliding") : dobs(_dobs), transfer(_transfer), flink(_flink), derr(_derr)
    {
        _dobs.init(obs_dist, obs_param[0], obs_param[1]);
        _flink.init(link_func, obs_param[0]);
        _transfer.init(dim, trans_func, gain_func, lag_dist, lag_param);
        _derr.init("gaussian", err_param[0], err_param[1]);

        return;
    }

    const ObsDist &dobs;
    const TransFunc &transfer;
    const LinkFunc &flink;
    const ErrDist &derr;

    void update_dobs(const double &value, const unsigned int &iloc)
    {
        if (iloc == 0)
        {
            _dobs.update_par1(value);
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
        const unsigned int &nlag = 0)
    {
        _dim.init(ntime, nlag, _transfer.dlag.par2);
        return;
    }

    arma::vec simulate()
    {
        _derr.sample(_dim.nT, true);

        _transfer.fgain.update_psi(_derr.psi);
        _transfer.fgain.psi2hpsi();

        _transfer.dlag.get_Fphi(_dim.nL);

        arma::vec y(_dim.nT + 1, arma::fill::zeros);
        y.at(0) = _y0;
        for (unsigned int t = 1; t < _dim.nT + 1; t++)
        {
            double ft = _transfer.transfer_sliding(t, y);
            double mu = _flink.ft2mu(ft);
            y.at(t) = _dobs.sample(mu);
        }
        y.at(0) = 0.;

        return y;
    }

    static arma::vec simulate(
        const arma::vec &psi,
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
        arma::vec hpsi = GainFunc::psi2hpsi(psi,gain_func); // Checked. OK.
        arma::vec Fphi = LagDist::get_Fphi(nlag, lag_dist, lag_param[0], lag_param[1]); // Checked. OK.

        arma::vec y(ntime + 1, arma::fill::zeros);
        y.at(0) = y0;
        for (unsigned int t = 1; t < (ntime + 1); t++)
        {
            double ftt = TransFunc::transfer_sliding(t, nlag, y, Fphi, hpsi); // Checked. OK.
            double mu = LinkFunc::ft2mu(ftt, link_func, obs_param[0]);
            y.at(t) = ObsDist::sample(mu, obs_param[1], obs_dist);
            // y.at(t) = dobs.sample(mu);
        }

        return y;
    }

    

private:
    ObsDist _dobs; // Observation distribution
    TransFunc _transfer;
    LinkFunc _flink;
    ErrDist _derr;
    Dim _dim;
    double _y0 = 2;
    double _mu0 = 1.;

};

#endif