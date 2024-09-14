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
#include "LagDist.hpp"
#include "GainFunc.hpp"
// #include "LinkFunc.hpp"

// [[Rcpp::plugins(cpp17)]]
// [[Rcpp::depends(RcppArmadillo)]]

class TransFunc
{
public:
    enum Transfer
    {
        sliding,
        iterative,
        none,
        degenerate // Only t(Fphi) * theta[t], {y[t]} is not involved
    };
    static const std::map<std::string, Transfer> trans_list;
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

    /**
     * @brief init_Ft
     *
     * @param nP
     * @param trans_func
     * @param seasonal_period
     * @return F0: arma::vec, nP x 1
     *
     * @note Non-transfer function part, i.e., seasonal component is also added to F0.
     */
    static arma::vec init_Ft(const unsigned int &nP, const std::string &trans_func = "sliding", const unsigned int &seasonal_period = 0, const bool &season_in_state = false)
    {
        std::map<std::string, Transfer> trans_list = TransFunc::trans_list;
        arma::vec F0(nP, arma::fill::zeros);
        if (trans_list[trans_func] == Transfer::iterative)
        {
            F0.at(1) = 1.;
        }

        if (seasonal_period > 0 && season_in_state)
        {
            unsigned int nstate = nP - seasonal_period;
            F0.at(nstate) = 1.;
        }
        return F0;
    }

    /**
     * @brief init_Gt
     * 
     * @param nP 
     * @param dlag 
     * @param trans_func 
     * @param seasonal_period 
     * @return G0: arma::mat, nP x nP
     * 
     * @note Non-transfer function part, i.e., seasonal component is also added to G0.
     */
    static arma::mat init_Gt(const unsigned int &nP, const LagDist &dlag, const std::string &trans_func = "sliding", const unsigned int &seasonal_period = 0, const bool &season_in_state = false)
    {
        unsigned int nstate = nP;
        if (season_in_state)
        {
            nstate -= seasonal_period;
        }
        
        std::map<std::string, Transfer> trans_list = TransFunc::trans_list;
        arma::mat G0(nP, nP, arma::fill::zeros);
        G0.at(0, 0) = 1.;

        if (trans_list[trans_func] == Transfer::iterative)
        {
            arma::vec iter_coef = nbinom::iter_coef(dlag.par1, dlag.par2);
            double coef_now = std::pow(1. - dlag.par1, dlag.par2);

            G0.at(1, 0) = coef_now;                      // (1 - kappa)^r
            G0.submat(1, 1, 1, nstate - 1) = iter_coef.t(); // c(r,1)(-kappa)^1, ..., c(r,r)(-kappa)^r

            for (unsigned int i = 2; i < nstate; i++)
            {
                G0.at(i, i - 1) = 1.;
            }
        }
        else
        {
            for (unsigned int i = 1; i < dlag.nL; i++)
            {
                G0.at(i, i - 1) = 1.;
            }
        }

        if (seasonal_period == 1 && season_in_state)
        {
            // first-order trend
            G0.at(nP - 1, nP - 1) = 1.;
        }
        else if (seasonal_period > 1 && season_in_state)
        {
            // Seasonal permutation matrix
            G0.at(nP - 1, nstate) = 1.;
            for (unsigned int i = 1; i < seasonal_period; i++)
            {
                G0.at(nstate + i - 1, nstate + i) = 1.;
            }
        }

        return G0;
    }

    static arma::mat H0_sliding(const unsigned int &nP) // Tested. OK.
    {
        arma::mat H0(nP, nP, arma::fill::zeros);
        H0.at(nP - 1, nP - 1) = 1.;

        unsigned int nr = nP - 1;
        for (unsigned int i = 1; i <= nr; i++)
        {
            H0.at(i - 1, i) = 1.;
        }
        // G0.diag(-1).ones();

        return H0;
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
        const arma::vec &ft_prev_rev, // (r x 1), f[t-1], ..., f[t-r], _ft.subvec(t - 1, t + _r - 2);
        const double &hpsi_now,        // psi[t]
        const double &y_prev,         // y[t-1]
        const double &lag_par1,
        const double &lag_par2)
    {
        // iter_coef: c(r,1)(-kappa)^1, ..., c(r,r)(-kappa)^r
        arma::vec iter_coef = nbinom::iter_coef(lag_par1, lag_par2);
        double ft = arma::accu(ft_prev_rev % iter_coef); // sum[k] f[t-k]c(r,k)(-kappa)^k

        // double hpsi_now = GainFunc::psi2hpsi(psi_now, gain_func);
        double Fast_now = hpsi_now * y_prev;
        ft += nbinom::coef_now(lag_par1, lag_par2) * Fast_now; // (1-kappa)^r y[t-1] * h(psi[t])

        bound_check(ft, "transfer_iterative: ft");
        return ft;
    }




    /**
     * @brief Transfer Function effect of the regressor.
     *
     * @param t
     * @param y At least (y[0], y[1], ..., y[t-1]), could be longer including current and future values.
     * @param ft At least (f[0], f[1], ..., f[t-1]), could be longer including current and future values.
     * @param psi At least (psi[0], psi[1], ..., psi[t]), could be longer including future values.
     * @param dlag
     * @param gain_func
     * @param trans_func
     * @return double
     * 
     * @note Checked. OK.
     */
    static double func_ft(
        const unsigned int &t, // 1, ..., nT
        const arma::vec &y,    // At least (y[0], y[1], ..., y[t-1]), could be longer including current and future values.
        const arma::vec &ft,   // At least (f[0], f[1], ..., f[t-1]), could be longer including current and future values.
        const arma::vec &hpsi,  // At least (hpsi[0], hpsi[1], ..., hpsi[t]), could be longer including future values.
        const LagDist &dlag,
        const std::string &trans_func)
    {
        double ft_now = 0.;
        std::string trans_func_name = tolower(trans_func);
        std::map<std::string, Transfer> trans_list = TransFunc::trans_list;
        switch (trans_list[trans_func_name])
        {
        case Transfer::sliding:
        {
            /*
            It uses:
            y[0], ..., y[t-1]
            psi[0], ..., psi[t]
            phi[1], ..., phi[nL]
            */
            arma::vec Fphi = LagDist::get_Fphi(dlag.nL, dlag.name, dlag.par1, dlag.par2);
            ft_now = TransFunc::transfer_sliding(t, dlag.nL, y, Fphi, hpsi);
            break;
        }
        case Transfer::iterative:
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
                ft_prev_rev, hpsi.at(t), y.at(t - 1),
                dlag.par1, dlag.par2);
            break;
        }
        default:
            break;
        }

        return ft_now;
    }


    static arma::mat psi2theta(
        const arma::vec &psi, // (nT + 1)
        const arma::vec &y, // (nT + 1) x 1
        const std::string &ftrans,
        const std::string &fgain,
        const LagDist &dlag)
    {
        std::map<std::string, Transfer> trans_list = TransFunc::trans_list;
        unsigned int ntime, nP;
        if (trans_list[ftrans] == Transfer::iterative)
        {
            ntime = psi.n_elem - 1;
            nP = (unsigned int)dlag.par2 + 1;
        }
        else
        {
            ntime = psi.n_elem;
            nP = dlag.nL;
        }

        arma::mat Theta(nP, ntime, arma::fill::zeros);
        if (trans_list[ftrans] == Transfer::sliding)
        {
            Theta.at(0, 0) = psi.at(0);
        }
        else
        {
            Theta.at(0, 0) = psi.at(1);
        }

        arma::vec ft(ntime, arma::fill::zeros);
        arma::vec hpsi = GainFunc::psi2hpsi<arma::vec>(psi, fgain);
        for (unsigned int t = 1; t < ntime; t++)
        {
            // int tlen = t - nlag + 1;
            // unsigned int tstart = (tlen < 0) ? 0 : (unsigned int)tlen;
            ft.at(t) = func_ft(t, y, ft, hpsi, dlag, ftrans);
            Theta.col(t) = psi2theta(t, psi, ft, y, ftrans, dlag, true);

            // if (trans_list[ftrans] == Transfer::sliding)
            // {
            //     arma::vec psi_sub = arma::reverse(psi.subvec(tstart, t));
            //     Theta.submat(0, t, psi_sub.n_elem - 1, t) = psi_sub;
            // }
            // else
            // {
            //     Theta.at(0, t) = psi.at(t + 1);
            //     arma::vec ft_sub = arma::reverse(ft.subvec(tstart, t));
            //     Theta.submat(1, t, ft_sub.n_elem, t) = ft_sub;
            // }
        }

        return Theta;
    }

    /**
     * @brief Return `theta[t]`
     *
     * @param t the time index of theta to be returned
     * @param psi needs `(psi[0],psi[1],..,psi[t],..)` for sliding or `(psi[0],psi[1],...,psi[t+1],...)` for iterative with `retrospective = true`; if `retrospective = false` when using iterative, the first element will be filled with psi[t] instead of psi[t+1].
     * @param ft needs `(f[0],f[1],...,f[t])`
     * @param y (nT + 1) x 1
     * @param ftrans
     * @param dlag dimension of theta is `dlag.nL` for sliding or `dlag.par2 + 1` for iterative
     * @param retrospective should theta[t] use the future psi[t+1] at time t?
     * @return arma::vec
     */
    static arma::vec psi2theta(
        const unsigned int &t,
        const arma::vec &psi, // at least t x 1 for sliding or at least (t+1) x 1 for iterative
        const arma::vec &ft, // at least(f[0],f[1],..,f[t])
        const arma::vec &y,   // (nT + 1) x 1
        const std::string &ftrans,
        const LagDist &dlag,
        const bool &retrospective = true)
    {
        std::map<std::string, Transfer> trans_list = TransFunc::trans_list;
        unsigned int nlag, nP;
        if (trans_list[ftrans] == Transfer::iterative)
        {
            nlag = (unsigned int)dlag.par2;
            nP = nlag + 1;
        }
        else
        {
            nlag = dlag.nL;
            nP = nlag;
            
        }

        arma::vec theta(nP, arma::fill::zeros);
        if (t == 0)
        {
            if (trans_list[ftrans] == Transfer::sliding)
            {
                theta.at(0) = psi.at(0);
            }
            else
            {
                if (retrospective)
                {
                    theta.at(0) = psi.at(1);
                }
                else
                {
                    theta.at(0) = psi.at(0);
                }
            }
        }
        else
        {
            int tlen = t - nlag + 1;
            unsigned int tstart = (tlen < 0) ? 0 : (unsigned int)tlen;
            if (trans_list[ftrans] == Transfer::sliding)
            {
                arma::vec psi_sub = arma::reverse(psi.subvec(tstart, t));
                theta.subvec(0, psi_sub.n_elem - 1) = psi_sub;
            }
            else
            {
                if (retrospective)
                {
                    theta.at(0) = psi.at(t + 1);
                }
                else
                {
                    theta.at(0) = psi.at(t);
                }
                arma::vec ft_sub = arma::reverse(ft.subvec(tstart, t));
                theta.subvec(1, ft_sub.n_elem) = ft_sub;
            }
        }
        
        return theta;
    }

private:
    static std::map<std::string, Transfer> map_trans_func()
    {
        std::map<std::string, Transfer> TRANS_MAP;

        TRANS_MAP["sliding"] = Transfer::sliding;
        TRANS_MAP["slide"] = Transfer::sliding;
        TRANS_MAP["koyama"] = Transfer::sliding;

        TRANS_MAP["iterative"] = Transfer::iterative;
        TRANS_MAP["iter"] = Transfer::iterative;
        TRANS_MAP["solow"] = Transfer::iterative;

        TRANS_MAP["degenerate"] = Transfer::degenerate;
        TRANS_MAP["constant"] = Transfer::degenerate;
        TRANS_MAP["linear"] = Transfer::degenerate;
        return TRANS_MAP;
    }
};

inline const std::map<std::string, TransFunc::Transfer> TransFunc::trans_list = TransFunc::map_trans_func();

#endif