#pragma once
#ifndef TRANSFUNC_H
#define TRANSFUNC_H

#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <RcppArmadillo.h>
#include <nlopt.h>
#include "LagDist.hpp"
#include "GainFunc.hpp"
// [[Rcpp::plugins(cpp17)]]
// [[Rcpp::depends(RcppArmadillo, nloptr)]]

class TransFunc
{
public:
    TransFunc()
    {
        init_default();
        return;
    }
    TransFunc(
        const Dim &dim,
        const std::string &trans_func = "sliding",
        const std::string &gain_func = "softplus",
        const std::string &lag_dist = "nbinom",
        const Rcpp::NumericVector &lag_param = Rcpp::NumericVector::create(NB_KAPPA, NB_R))
    {
        init(dim, trans_func, gain_func, lag_dist, lag_param);
        return;
        
    }

    void init_default()
    {
        Dim dim;
        dim.init_default();

        _trans_list = AVAIL::map_trans_func();
        _name = "sliding";

        dlag.init_default();
        fgain.init_default();
        return;
    }

    void init(
        const Dim &dim,
        const std::string &trans_func = "sliding",
        const std::string &gain_func = "softplus",
        const std::string &lag_dist = "nbinom",
        const Rcpp::NumericVector &lag_param = Rcpp::NumericVector::create(NB_KAPPA, NB_R))
    {
        _trans_list = AVAIL::map_trans_func();

        _nlag = dim.nL;
        _ntime = dim.nT;
        dlag.init(lag_dist, lag_param[0], lag_param[1]);
        fgain.init(gain_func, dim);

        /*
         * We should only use iterative formula for non-truncated negative-binomial lags.
         *
         */
        bool iterative_ok = !dim.truncated && dlag.isnbinom;
        if (!iterative_ok)
        {
            _name = "sliding";
        }

        if (_trans_list[_name] == AVAIL::Transfer::iterative)
        {
            _r = static_cast<unsigned int>(dlag.par2);
            iter_coef = nbinom::iter_coef(dlag.par1, dlag.par2);
            coef_now = coef_now = std::pow(1. - dlag.par1, dlag.par2);

            _ft.set_size(dim.nT + _r);
        }
        else
        {
            _r = 1;
            _ft.set_size(dim.nT + 1);
        }

        _ft.zeros();
        return;
    }


    LagDist dlag;
    GainFunc fgain;



    /**
     * @brief From latent state psi[t] to transfer effect f[t] using the exact formula. Note that psi[t] is the random-walk component of the latent state theta[t].
     * 
     * @param y Observed count data, y = (y[0], y[1], ..., y[nT])'.
     * @param dlag LagDist object that contains Fphi = (phi[1], ..., phi[nL])'.
     * @param fgain GainFunc object that contains hpsi = (h(psi[0]), h(psi[1]), ..., h(psi[nT]))'.
     */
    void update_ft_exact(const arma::vec &y) // y[0], y[1], ..., y[nT]
    {
        for (unsigned int t = 1; t <= _ntime; t++)
        {
            if (_trans_list[_name] == AVAIL::Transfer::iterative)
            {
                _ft.at(t + _r - 1) = transfer_iterative(t, y.at(t-1));
            }
            else
            {
                _ft.at(t) = transfer_sliding(t, y);
            }
        }
    }


    arma::vec get_ft(const bool &no_padding = true)
    {
        arma::vec ft;
        if (no_padding)
        {
            ft = _ft.tail(_nlag + 1);
        }
        else
        {
            ft = _ft;
        }

        return ft;
    }

    /**
     * @brief Exact method that transfers psi[t] to f[t] using the sliding formula: f[t] is a sliding-window weighted-averaged of the past observations y[0:(t-1)] and gains h(psi[0:t]). It is a general methods for all kinds of lag distributions, truncated or not truncated. [Checked. OK.]
     *
     * @param y Observed count data, y = (y[0], y[1], ..., y[nT])'.
     * @param dlag LagDist object that contains Fphi = (phi[1], ..., phi[nL])'.
     * @param fgain GainFunc object that contains hpsi = (h(psi[0]), h(psi[1]), ..., h(psi[nT]))'.
     * 
     * @return double
     */
    double transfer_sliding(
        const unsigned int &t,
        const arma::vec &y) // (nT + 1) x 1: y[0], y[1], ..., y[nT]
    {
        unsigned int nelem = std::min(t, _nlag);
        arma::vec Fphi_t = dlag.Fphi.subvec(0, nelem - 1);      // Fphi[t]      = (phi[1], ..., phi[nL])'

        Fphi_t = arma::reverse(Fphi_t);                    // rev(Fphi[t]) = (phi[nL], ..., phi[1])
        arma::vec Fy_t = y.subvec(t - nelem, t - 1);       // Fy[t]        = (y[t-nL], ..., y[t-1])'

        arma::vec Fhpsi_t = fgain.hpsi.subvec(t + 1 - nelem, t); // Fhpsi[t]     = (h(psi[t+1-nL]), ..., h(psi[t]))'

        arma::vec Fast_t = Fy_t % Fhpsi_t;
        double ft = arma::accu(Fphi_t % Fast_t);
        return ft;
    }

    static double transfer_sliding(
        const unsigned int &t,
        const unsigned int &nlag,
        const arma::vec &y,
        const arma::vec &Fphi,
        const arma::vec &hpsi) // (nT + 1) x 1: y[0], y[1], ..., y[nT]
    {
        unsigned int nelem = std::min(t, nlag);
        

        // arma::vec Fphi_t = Fphi.subvec(0, nelem - 1); // Fphi[t]      = (phi[1], ..., phi[nL])'
        arma::vec Fphi_t;
        try
        {
            Fphi_t = Fphi.subvec(0, nelem - 1);
        }
        catch(...)
        {
            std::cout << "\n Fphi at nlem = " << nelem << ", t = " << t << std::endl;
            throw std::invalid_argument("Fphi");
        }
        
        Fphi_t = arma::reverse(Fphi_t);              // rev(Fphi[t]) = (phi[nL], ..., phi[1])

        // arma::vec Fy_t = y.subvec(t - nelem, t - 1); // Fy[t]        = (y[t-nL], ..., y[t-1])'
        arma::vec Fy_t;
        try
        {
            Fy_t = y.subvec(t - nelem, t - 1);
        }
        catch(const std::exception& e)
        {
            std::cout << "\n Fy at nlem = " << nelem << ", t = " << t << std::endl;
            throw std::invalid_argument("Fy");
        }
        

        // arma::vec Fhpsi_t = hpsi.subvec(t + 1 - nelem, t); // Fhpsi[t]     = (h(psi[t+1-nL]), ..., h(psi[t]))'
        arma::vec Fhpsi_t;
        try
        {
            Fhpsi_t = hpsi.subvec(t + 1 - nelem, t);
        }
        catch (const std::exception &e)
        {
            std::cout << "\n Fhpsi at nlem = " << nelem << ", t = " << t << std::endl;
            throw std::invalid_argument("Fhpsi");
        }

        arma::vec Fast_t = Fy_t % Fhpsi_t;
        double ft = arma::accu(Fphi_t % Fast_t);
        return ft;
    }

    /**
     * @brief Exact method that transfers psi[t] to f[t] using the iterative formula: f[t] is a sum of past transfer effects f[0:(t-1)], and current gain h(psi[t]) + previous observation (ancestor) y[t-1]. Only works for non-truncated negative-binomial lag distribution.
     *
     * @param y Observed count data, y = (y[0], y[1], ..., y[nT])'.
     * @param dlag LagDist object that contains Fphi = (phi[1], ..., phi[nL])'.
     * @param fgain GainFunc object that contains hpsi = (h(psi[0]), h(psi[1]), ..., h(psi[nT]))'.
     *
     * @return double
     */
    double transfer_iterative(
        const unsigned int &t,
        const double &y_prev
    )
    {
        /**
         * @brief _ft: (nT + r) x 1
         * 
         * Indx:     0,     , ..., r-1,  r   , ..., r + nT - 1 
         * _ft = ( f[-(r-1)], ..., f[0], f[1], ..., f[nT] ),
         *         f[-(r-1)] = ... = f[0] = 0.
         * 
         * f[t] is indexed by (t + r - 1).
         */
        arma::vec Fast_t = _ft.subvec(t - 1, t + _r - 2); // f[t-r], ..., f[t-1]
        Fast_t = arma::reverse(Fast_t); // f[t-1], ..., f[t-r]
        double ft = arma::accu(Fast_t % iter_coef);

        double Fast_now = fgain.hpsi.at(t) * y_prev;
        ft += coef_now * Fast_now;

        bound_check(ft, "transfer_iterative: ft");
        return ft;
    }

    void update_ft_approx();

private:
    std::map<std::string, AVAIL::Transfer> _trans_list;
    unsigned int _r;
    unsigned int _nlag;
    unsigned int _ntime;

    std::string _name = "sliding";
    arma::vec iter_coef;
    double coef_now;

    arma::vec _ft;
    arma::vec _Ft;
};

#endif