#pragma once
#ifndef MODEL_H
#define MODEL_H

#include <RcppArmadillo.h>
#include "ErrDist.hpp"
#include "TransFunc.hpp"
#include "LinkFunc.hpp"
#include "ObsDist.hpp"
#include "utils.h"
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
        const Rcpp::NumericVector &obs_param = Rcpp::NumericVector::create(0., 30.),
        const Rcpp::NumericVector &lag_param = Rcpp::NumericVector::create(NB_KAPPA, NB_R),
        const Rcpp::NumericVector &err_param = Rcpp::NumericVector::create(0.01, 0.), // (W, w[0])
        const std::string &trans_func = "sliding") : dim(_dim), dobs(_dobs), transfer(_transfer), flink(_flink), derr(_derr)
    {
        _dim = dim_;
        _dobs.init(obs_dist, obs_param[0], obs_param[1]);
        _flink.init(link_func, obs_param[0]);
        _transfer.init(dim_, trans_func, gain_func, lag_dist, lag_param);

        _derr.init("gaussian", err_param[0], err_param[1]);
        return;
    }

    const Dim &dim;
    ObsDist &dobs;
    TransFunc &transfer;
    LinkFunc &flink;
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


    /**
     * @brief Simulate from the DGTF model from the scratch.
     * 
     * @return arma::vec 
     */
    arma::vec simulate(const double &y0 = 0.)
    {
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
            double mu = _flink.ft2mu(ft); // Checked. OK.
            y.at(t) = _dobs.sample(mu); // Checked. OK.
        }

        return y; // Checked. OK.
    }

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

    

private:
    ObsDist _dobs; // Observation distribution
    TransFunc _transfer;
    LinkFunc _flink;
    ErrDist _derr;
    Dim _dim;
    double _y0 = 0.;
    double _mu0 = 1.;

};

#endif