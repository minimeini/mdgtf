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


    /**
     * @brief State evolution equation for the DLM form model
     * 
     * @param model 
     * @param theta_cur 
     * @param ycur 
     * @return arma::vec 
     */
    static arma::vec func_gt( // Checked. OK.
        const Model &model,
        const arma::vec &theta_cur, // nP x 1, (psi[t], f[t-1], ..., f[t-r])
        const double &ycur)
    {
        arma::vec theta_next;
        theta_next.copy_size(theta_cur);

        switch (model.transfer.trans_list[model.transfer.name])
        {
        case AVAIL::Transfer::iterative:
        {
            theta_next.at(0) = theta_cur.at(0); // Expectation of random walk.
            theta_next.at(1) = TransFunc::transfer_iterative(
                theta_cur.subvec(1, model.dim.nP - 1), // f[t-1], ..., f[t-r]
                theta_cur.at(0),                       // psi[t]
                ycur,                                  // y[t-1]
                model.transfer.name,
                model.transfer.dlag.par1,
                model.transfer.dlag.par2);
            theta_next.subvec(2, model.dim.nP - 1) = theta_cur.subvec(1, model.dim.nP - 2);
            break;
        }
        default: // AVAIL::Transfer::sliding
        {
            // theta_next = model.transfer.G0 * theta_cur;
            theta_next.at(0) = theta_cur.at(0);
            theta_next.subvec(1, model.dim.nP - 1) = theta_cur.subvec(0, model.dim.nP - 2);
            break;
        }
        }

        bound_check<arma::vec>(theta_next, "func_gt: theta_next");
        return theta_next;
    }

    /**
     * @brief f[t]( theta[t] ) - maps state theta[t] to observation-level variable f[t].
     *
     * @param model
     * @param t
     * @param theta_cur
     * @param yall
     * @return double
     */
    static double func_ft(
        const Model &model,
        const int &t,      // t = 0, y[0] = 0, theta[0] = 0; t = 1, y[1], theta[1]; ...
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
            
            
            yold = arma::reverse(yold);                       // y[t-1], ..., y[t-min(t,nL)]

            arma::vec ft_vec = model.transfer.dlag.Fphi;

            arma::vec hpsi_cur = GainFunc::psi2hpsi(theta_cur, model.transfer.fgain.name); // (h(psi[t]), ..., h(psi[t+1 - nL]))
            arma::vec ftmp = yold % hpsi_cur;
            ft_vec = ft_vec % ftmp;

            ft_cur = arma::accu(ft_vec);
        }
        else
        {
            ft_cur = theta_cur.at(1);
        }

        bound_check(ft_cur, "func_ft: ft_cur");
        return ft_cur;
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