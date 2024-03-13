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
#include "LinkFunc.hpp"

// [[Rcpp::plugins(cpp17)]]
// [[Rcpp::depends(RcppArmadillo, nloptr)]]

class TransFunc
{
public:
    TransFunc() : F0(_F0), G0(_G0), name(_name), iter_coef(_iter_coef), coef_now(_coef_now)
    {
        init_default();
        return;
    }
    TransFunc(
        const Dim &dim,
        const std::string &trans_func = "sliding",
        const std::string &gain_func = "softplus",
        const std::string &lag_dist = "lognorm",
        const Rcpp::NumericVector &lag_param = Rcpp::NumericVector::create(LN_MU, LN_SD2)) : F0(_F0), G0(_G0), name(_name), iter_coef(_iter_coef), coef_now(_coef_now)
    {
        init(dim, trans_func, gain_func, lag_dist, lag_param);
        return;
    }

    void init_default()
    {
        _dim.init_default();

        trans_list = AVAIL::trans_list;
        _name = "sliding";
        G0_sliding();
        F0_sliding();

        dlag.init_default();
        dlag.get_Fphi(_dim.nL);

        fgain.init_default();

        _r = 1;
        _ft.set_size(_dim.nT + 1);

        _ft.zeros();
        return;
    }

    void init(
        const Dim &dim,
        const std::string &trans_func = "sliding",
        const std::string &gain_func = "softplus",
        const std::string &lag_dist = "nbinom",
        const Rcpp::NumericVector &lag_param = Rcpp::NumericVector::create(NB_KAPPA, NB_R))
    {
        trans_list = AVAIL::trans_list;
        _dim = dim;
        _G0.set_size(_dim.nP, _dim.nP);
        _G0.zeros();

        // _nlag = dim.nL;
        // _ntime = dim.nT;
        dlag.init(lag_dist, lag_param[0], lag_param[1]);
        dlag.get_Fphi(dim.nL);

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
        else
        {
            _name = trans_func;
        }

        if (trans_list[_name] == AVAIL::Transfer::iterative)
        {
            _r = static_cast<unsigned int>(dlag.par2);
            _iter_coef = nbinom::iter_coef(dlag.par1, dlag.par2);
            _coef_now = std::pow(1. - dlag.par1, dlag.par2);

            _ft.set_size(dim.nT + _r);
            G0_iterative();
            F0_iterative();
        }
        else
        {
            _r = 1;
            _ft.set_size(dim.nT + 1);
            G0_sliding();
            F0_sliding();
        }

        _ft.zeros();
        return;
    }

    LagDist dlag;
    GainFunc fgain;
    const arma::mat &G0;
    const arma::vec &F0;
    std::map<std::string, AVAIL::Transfer> trans_list = AVAIL::trans_list;
    const std::string &name;
    const arma::vec &iter_coef;
    const double &coef_now;

    /**
     * @brief From latent state psi[t] to transfer effect f[t] using the exact formula. Note that psi[t] is the random-walk component of the latent state theta[t].
     *
     * @param y Observed count data, y = (y[0], y[1], ..., y[nT])'.
     * @param dlag LagDist object that contains Fphi = (phi[1], ..., phi[nL])'.
     * @param fgain GainFunc object that contains hpsi = (h(psi[0]), h(psi[1]), ..., h(psi[nT]))'.
     */
    void update_ft_exact(const arma::vec &y) // y[0], y[1], ..., y[nT]
    {
        for (unsigned int t = 1; t <= _dim.nT; t++)
        {
            if (trans_list[_name] == AVAIL::Transfer::iterative)
            {
                _ft.at(t + _r - 1) = transfer_iterative(t, y.at(t - 1));
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
            ft = _ft.tail(_dim.nL + 1);
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
        unsigned int nelem = std::min(t, _dim.nL);
        arma::vec Fphi_t = dlag.Fphi.subvec(0, nelem - 1); // Fphi[t] = (phi[1], ..., phi[nL])'
        Fphi_t = arma::reverse(Fphi_t);                    // rev(Fphi[t]) = (phi[nL], ..., phi[1])

        arma::vec Fy_t = y.subvec(t - nelem, t - 1); // Fy[t] = (y[t-nL], ..., y[t-1])'

        arma::vec Fhpsi_t = fgain.hpsi.subvec(t + 1 - nelem, t); // Fhpsi[t] = (h(psi[t+1-nL]), ..., h(psi[t]))'

        arma::vec Fast_t = Fy_t % Fhpsi_t;
        double ft = arma::accu(Fphi_t % Fast_t);
        return ft;
    }

    /**
     * @brief f[t](btheta[t],y[0:t]) calculate: sum Fphi[k] * h(psi[t + 1 - k]) * y[t - k]. This is exact formula for sliding transfer function. The effective number of elements is min(nL, t).
     *
     * @param t
     * @param nlag
     * @param y (nT + 1) x 1: y[0], y[1], ..., y[nT]; we use y[(t - min(nL,t)) : (t - 1)] at time t.
     * @param Fphi Fphi[t] = (phi[1], ..., phi[min(nL, t)])'
     * @param hpsi (nT + 1) x 1: h(psi[0]), h(psi[1]), ..., h(psi[nT])); we have h( btheta[t : (t + 1 - min(nL,t))] ) at time t.
     * @return double
     */
    static double transfer_sliding(
        const unsigned int &t,
        const unsigned int &nlag,
        const arma::vec &y,    // (nT + 1) x 1: y[0], y[1], ..., y[nT]
        const arma::vec &Fphi, // Fphi[t]      = (phi[1], ..., phi[nL])'
        const arma::vec &hpsi) // (nT + 1) x 1: h(psi[0]), h(psi[1]), ..., h(psi[nT])
    {
        unsigned int nelem = std::min(t, nlag);

        arma::vec Fphi_t = Fphi.subvec(0, nelem - 1); // Fphi[t] = (phi[1], ..., phi[nL])'
        Fphi_t = arma::reverse(Fphi_t);

        arma::vec Fy_t = y.subvec(t - nelem, t - 1);       // Fy[t] = (y[t-nL], ..., y[t-1])'
        arma::vec Fhpsi_t = hpsi.subvec(t + 1 - nelem, t); // Fhpsi[t] = (h(psi[t+1-nL]), ..., h(psi[t]))'

        arma::vec Fast_t = Fy_t % Fhpsi_t;
        double ft = arma::accu(Fphi_t % Fast_t);
        return ft;
    }

    void F0_sliding()
    {
        _F0.set_size(_dim.nP);
        _F0.zeros();
        return;
    }

    static arma::vec F0_sliding(const unsigned int &nP)
    {
        arma::vec F0(nP, arma::fill::zeros);
        return F0;
    }

    void G0_sliding()
    {
        // arma::mat G0(_dim.nP, _dim.nP, arma::fill::zeros);
        _G0.set_size(_dim.nP, _dim.nP);
        _G0.zeros();
        _G0.at(0, 0) = 1.;
        for (unsigned int i = 1; i < _dim.nP; i++)
        {
            _G0.at(i, i - 1) = 1.;
        }
        // G0.diag(-1).ones();

        return;
    }

    static arma::mat G0_sliding(const Dim &dim) // Tested. OK.
    {
        arma::mat G0(dim.nP, dim.nP, arma::fill::zeros);
        G0.at(0, 0) = 1.;
        for (unsigned int i = 1; i < dim.nP; i++)
        {
            G0.at(i, i - 1) = 1.;
        }
        // G0.diag(-1).ones();

        return G0;
    }

    /**
     * @brief Exact method that transfers psi[t] to g[t](theta[t]) = f[t](psi[t], f[t-1], ..., f[t-r]) using the iterative formula: f[t] is a sum of past transfer effects f[0:(t-1)], and current gain h(psi[t]) + previous observation (ancestor) y[t-1]. Only works for non-truncated negative-binomial lag distribution.
     *
     * @param y Observed count data, y = (y[0], y[1], ..., y[nT])'.
     * @param dlag LagDist object that contains Fphi = (phi[1], ..., phi[nL])'.
     * @param fgain GainFunc object that contains hpsi = (h(psi[0]), h(psi[1]), ..., h(psi[nT]))'.
     *
     * @return double
     */
    double transfer_iterative(
        const unsigned int &t,
        const double &y_prev)
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
        Fast_t = arma::reverse(Fast_t);                   // f[t-1], ..., f[t-r]
        double ft = arma::accu(Fast_t % _iter_coef);
        // iter_coef: c(r,1)(-kappa)^1, ..., c(r,r)(-kappa)^r

        double Fast_now = fgain.hpsi.at(t) * y_prev;
        ft += _coef_now * Fast_now;

        bound_check(ft, "transfer_iterative: ft");
        return ft;
    }

    /**
     * @brief g[t](\btheta[t-1], y[t-1]), exact formula
     *
     * @param ft_prev_rev btheta[1:(nP-1)] = btheta[1:r] = (f[t-1], ..., f[t-r])
     * @param psi_now psi[t] = btheta[0]
     * @param y_prev y[t-1]
     * @param transfer
     * @return double
     */
    static double transfer_iterative(
        const arma::vec &ft_prev_rev, // (r x 1), f[t-1], ..., f[t-r]
        const double &psi_now,        // psi[t]
        const double &y_prev,         // y[t-1]
        const std::string &gain_func,
        const double &lag_par1,
        const double &lag_par2)
    {
        // iter_coef: c(r,1)(-kappa)^1, ..., c(r,r)(-kappa)^r
        arma::vec iter_coef = nbinom::iter_coef(lag_par1, lag_par2);
        double ft = arma::accu(ft_prev_rev % iter_coef); // sum[k] f[t-k]c(r,k)(-kappa)^k

        double hpsi_now = GainFunc::psi2hpsi(psi_now, gain_func);
        double Fast_now = hpsi_now * y_prev;
        ft += nbinom::coef_now(lag_par1, lag_par2) * Fast_now; // (1-kappa)^r y[t-1] * h(psi[t])

        bound_check(ft, "transfer_iterative: ft");
        return ft;
    }

    void F0_iterative()
    {
        _F0.set_size(_dim.nP);
        _F0.zeros();
        _F0.at(1) = 1.;
        return;
    }

    static arma::vec F0_iterative(const unsigned int &nP)
    {
        arma::vec F0(nP, arma::fill::zeros);
        F0.at(1) = 1.;
        return F0;
    }

    void G0_iterative()
    {
        // arma::mat G0(_dim.nP, _dim.nP, arma::fill::zeros);
        _G0.set_size(_dim.nP, _dim.nP);
        _G0.zeros();
        _G0.at(0, 0) = 1.;
        _G0.at(1, 0) = _coef_now;                          // (1 - kappa)^r
        _G0.submat(1, 1, 1, _dim.nP - 1) = _iter_coef.t(); // c(r,1)(-kappa)^1, ..., c(r,r)(-kappa)^r
        for (unsigned int i = 2; i < _dim.nP; i++)
        {
            _G0.at(i, i - 1) = 1.;
        }

        return;
    }

    static arma::mat G0_iterative(const Dim &dim, const LagDist &dlag) // Tested. OK.
    {
        arma::mat G0(dim.nP, dim.nP, arma::fill::zeros);
        arma::vec iter_coef = nbinom::iter_coef(dlag.par1, dlag.par2);
        double coef_now = std::pow(1. - dlag.par1, dlag.par2);

        G0.at(0, 0) = 1.;
        G0.at(1, 0) = coef_now;                         // (1 - kappa)^r
        G0.submat(1, 1, 1, dim.nP - 1) = iter_coef.t(); // c(r,1)(-kappa)^1, ..., c(r,r)(-kappa)^r
        for (unsigned int i = 2; i < dim.nP; i++)
        {
            G0.at(i, i - 1) = 1.;
        }

        return G0;
    }

    /**
     * @brief Transfer Function effect of the regressor.
     *
     * @param t
     * @param y At least (y[0], y[1], ..., y[t-1]), could be longer including current and future values.
     * @param ft At least (f[0], f[1], ..., f[t-1]), could be longer including current and future values.
     * @param psi At least (psi[0], psi[1], ..., psi[t]), could be longer including future values.
     * @param dim
     * @param dlag
     * @param gain_func
     * @param trans_func
     * @return double
     */
    static double func_ft(
        const unsigned int &t, // 1, ..., nT
        const arma::vec &y,    // At least (y[0], y[1], ..., y[t-1]), could be longer including current and future values.
        const arma::vec &ft,   // At least (f[0], f[1], ..., f[t-1]), could be longer including current and future values.
        const arma::vec &psi,  // At least (psi[0], psi[1], ..., psi[t]), could be longer including future values.
        const Dim &dim,
        const LagDist &dlag,
        const std::string &gain_func,
        const std::string &trans_func)
    {
        double ft_now = 0.;
        std::string trans_func_name = tolower(trans_func);
        std::string gain_func_name = tolower(gain_func);

        std::map<std::string, AVAIL::Transfer> trans_list = AVAIL::trans_list;
        switch (trans_list[trans_func_name])
        {
        case AVAIL::Transfer::sliding:
        {
            /*
            It uses:
            y[0], ..., y[t-1]
            psi[0], ..., psi[t]
            phi[1], ..., phi[nL]
            */
            arma::vec Fphi = LagDist::get_Fphi(dim.nL, dlag.name, dlag.par1, dlag.par2);
            arma::vec hpsi = GainFunc::psi2hpsi<arma::vec>(psi, gain_func_name);
            ft_now = TransFunc::transfer_sliding(t, dim.nL, y, Fphi, hpsi);
            break;
        }
        case AVAIL::Transfer::iterative:
        {
            /*
            It uses:
            f[0], ..., f[t-1]
            y[t-1]
            psi[t]
            */
            unsigned int r = static_cast<unsigned int>(dlag.par2);
            arma::vec ft_prev(r, arma::fill::zeros);
            int idx_s = std::max((int)(t - r), 0);
            int idx_e = std::max((int)(t - 1), 0);
            unsigned int nelem = idx_e - idx_s + 1;
            if (nelem > 1)
            {
                ft_prev.tail(nelem) = ft.subvec(idx_s, idx_e); // 0, ..., 0, f[t-r], ..., f[t-1]
            }
            else
            {
                ft_prev.at(r - 1) = ft.at(idx_e);
            }

            arma::vec ft_prev_rev = arma::reverse(ft_prev);
            ft_now = TransFunc::transfer_iterative(
                ft_prev_rev, psi.at(t), y.at(t - 1),
                gain_func_name, dlag.par1, dlag.par2);
            break;
        }
        default:
            break;
        }

        return ft_now;
    }

    double func_ft(
        const unsigned int &t, // 1, ..., nT
        const arma::vec &y,    // At least (y[0], y[1], ..., y[t-1]), could be longer including current and future values.
        const arma::vec &ft)   // At least (f[0], f[1], ..., f[t-1]), could be longer including current and future values.
    {
        double ft_now = 0.;

        switch (trans_list[name])
        {
        case AVAIL::Transfer::sliding:
        {
            /*
            It uses:
            y[0], ..., y[t-1]
            psi[0], ..., psi[t]
            phi[1], ..., phi[nL]
            */
            ft_now = transfer_sliding(t, y);
            // ft_now = TransFunc::transfer_sliding(t, _dim.nL, y, dlag.Fphi, fgain.hpsi);
            break;
        }
        case AVAIL::Transfer::iterative:
        {
            /*
            It uses:
            f[0], ..., f[t-1]
            y[t-1]
            psi[t]
            */
            unsigned int r = static_cast<unsigned int>(dlag.par2);
            arma::vec ft_prev(r, arma::fill::zeros);

            int idx_s = std::max((int)(t - r), 0);
            int idx_e = std::max((int)(t - 1), 0);
            unsigned int nelem = idx_e - idx_s + 1;
            try
            {
                
                if (nelem > 1)
                {
                    ft_prev.tail(nelem) = ft.subvec(idx_s, idx_e); // 0, ..., 0, f[t-r], ..., f[t-1]
                }
                else
                {
                    ft_prev.at(r - 1) = ft.at(idx_e);
                }
            }
            catch(const std::exception& e)
            {
                std::cout << "\n nelem = " << nelem;
                std::cout << " idx_s = " << idx_s << " idx_e = " << idx_e << " ft = " << ft.n_elem << std::endl;
                std::cerr << e.what() << '\n';
            }
            
            

            arma::vec ft_prev_rev = arma::reverse(ft_prev);
            ft_now = transfer_iterative(t, y.at(t - 1));

            break;
        }
        default:
            break;
        }

        return ft_now;
    }


private:
    Dim _dim;
    unsigned int _r;

    std::string _name = "sliding";
    arma::vec _iter_coef;
    double _coef_now;

    arma::vec _ft; // (nT + r) x 1, r = 1 if using sliding transfer function.
    arma::vec _F0; // nP x 1
    arma::mat _G0; // nP x nP
};

#endif