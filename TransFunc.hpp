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
#include "Regression.hpp"
// #include "LinkFunc.hpp"

// [[Rcpp::plugins(cpp17)]]
// [[Rcpp::depends(RcppArmadillo)]]

class TransFunc
{
public:
    enum Transfer
    {
        sliding,
        iterative
    };
    static const std::map<std::string, Transfer> trans_list;

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
        const arma::vec &hpsi, // (nT + 1) x 1: h(psi[0]), h(psi[1]), ..., h(psi[nT])
        const double &y_scale = 1.
    )
    {
        unsigned int nelem = std::min(t, nlag);

        arma::vec Fphi_t = Fphi.subvec(0, nelem - 1); // Fphi[t] = (phi[1], ..., phi[nL])'
        Fphi_t = arma::reverse(Fphi_t);

        arma::vec Fy_t = y.subvec(t - nelem, t - 1) / y_scale;       // Fy[t] = (y[t-nL], ..., y[t-1])'
        arma::vec Fhpsi_t = hpsi.subvec(t + 1 - nelem, t); // Fhpsi[t] = (h(psi[t+1-nL]), ..., h(psi[t]))'

        arma::vec Fast_t = Fy_t % Fhpsi_t;
        double ft = arma::accu(Fphi_t % Fast_t);
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
        const arma::vec &ft_prev_rev, // (r x 1), f[t-1], ..., f[t-r], _ft.subvec(t - 1, t + _r - 2);
        const double &hpsi_now,        // psi[t]
        const double &y_prev,         // y[t-1]
        const double &lag_par1,
        const double &lag_par2,
        const double &y_scale = 1.
    )
    {
        // iter_coef: c(r,1)(-kappa)^1, ..., c(r,r)(-kappa)^r
        arma::vec iter_coef = nbinom::iter_coef(lag_par1, lag_par2);
        double ft = arma::accu(ft_prev_rev % iter_coef); // sum[k] f[t-k]c(r,k)(-kappa)^k

        // double hpsi_now = GainFunc::psi2hpsi(psi_now, gain_func);
        double Fast_now = hpsi_now * y_prev / y_scale;
        ft += nbinom::coef_now(lag_par1, lag_par2) * Fast_now; // (1-kappa)^r y[t-1] * h(psi[t])

        #ifdef DGTF_DO_BOUND_CHECK
        bound_check(ft, "transfer_iterative: ft");
        #endif
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
        const std::string &trans_func,
        const double &y_scale = 1.
    )
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
            ft_now = TransFunc::transfer_sliding(t, dlag.nL, y, Fphi, hpsi, y_scale);
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
                dlag.par1, dlag.par2, y_scale);
            break;
        }
        default:
        {
            throw std::invalid_argument("func_ft: unknown transfer function.");
        }
        }

        return ft_now;
    }


    /**
     * @brief f[t]( theta[t] ) - maps state theta[t] to observation-level variable f[t].
     * 
     * @param ftrans 
     * @param fgain 
     * @param dlag 
     * @param seas 
     * @param t 
     * @param theta_cur 
     * @param yall 
     * @return double 
     */
    static double func_ft(
        const std::string &ftrans,
        const std::string &fgain,
        const LagDist &dlag,
        const Season &seas,
        const int &t,               // t = 0, y[0] = 0, theta[0] = 0; t = 1, y[1], theta[1]; ...;  yold.tail(nelem) = yall.subvec(t - nelem, t - 1);
        const arma::vec &theta_cur, // theta[t] = (psi[t], ..., psi[t+1 - nL]) or (psi[t+1], f[t], ..., f[t+1-r])
        const arma::vec &yall,      // We use y[t - nelem], ..., y[t-1]
        const double &y_scale = 1.
    )
    {
        std::map<std::string, Transfer> trans_list = TransFunc::trans_list;
        double ft_cur;
        if (trans_list[ftrans] == Transfer::sliding)
        {
            int nelem = std::min(t, (int)dlag.nL); // min(t,nL)
            arma::vec yold(dlag.nL, arma::fill::zeros);
            if (nelem > 1)
            {
                yold.tail(nelem) = yall.subvec(t - nelem, t - 1); // 0, ..., 0, y[t - nelem], ..., y[t-1]
            }
            else if (t > 0) // nelem = 1 at t = 1
            {
                yold.at(dlag.nL - 1) = yall.at(t - 1);
            }

            yold = arma::reverse(yold) / y_scale; // y[t-1], ..., y[t-min(t,nL)]

            arma::vec ft_vec = dlag.Fphi; // nL x 1

            arma::vec th = theta_cur.head(dlag.nL);
            arma::vec hpsi_cur = GainFunc::psi2hpsi<arma::vec>(th, fgain); // (h(psi[t]), ..., h(psi[t+1 - nL])), nL x 1
            arma::vec ftmp = yold % hpsi_cur; // nL x 1
            ft_vec = ft_vec % ftmp;
            ft_cur = arma::accu(ft_vec);
        } // sliding
        else
        {
            ft_cur = theta_cur.at(1);
        } // iterative

        if (seas.period > 0)
        {
            // Add the current seasonal level
            if (seas.in_state)
            {
                
                ft_cur += theta_cur.at(theta_cur.n_elem - seas.period);
            }
            else if (!seas.X.is_empty() && !seas.val.is_empty())
            {
                ft_cur += arma::as_scalar(seas.X.col(t).t() * seas.val);
            }
        }

        #ifdef DGTF_DO_BOUND_CHECK
            bound_check(ft_cur, "func_ft: ft_cur");
        #endif
        return ft_cur;
    }


    /**
     * @brief First-order derivative of f[t] w.r.t theta[t].
     *
     * @param model
     * @param t time index of theta_cur
     * @param theta_cur theta[t] = (psi[t], ..., psi[t+1 - nL]) with sliding transfer function or (psi[t+1], f[t], ..., f[t+1-r]) with iterative transfer function
     * @param yall y[0], y[1], ..., y[nT]
     * @return arma::vec
     */
    static arma::vec func_Ft(
        const std::string &ftrans,
        const std::string &fgain,
        const LagDist &dlag,
        const unsigned int &t,      // time index of theta_cur, t = 0, y[0] = 0, theta[0] = 0; t = 1, y[1], theta[1]; ...
        const arma::vec &theta_cur, // nP x 1, theta[t] = (psi[t], ..., psi[t+1 - nL]) or (psi[t+1], f[t], ..., f[t+1-r])
        const arma::vec &yall,      // y[0], y[1], ..., y[nT]
        const unsigned int &seasonal_period = 0,
        const bool &season_in_state = false,
        const double &y_scale = 1.
    )
    {
        std::map<std::string, TransFunc::Transfer> trans_list = TransFunc::trans_list;
        arma::vec Ft(theta_cur.n_elem, arma::fill::zeros);
        if (trans_list[ftrans] == TransFunc::sliding)
        {
            unsigned int nstart = (t > dlag.nL) ? (t - dlag.nL) : 0;
            // unsigned int nstart = std::max((unsigned int)0, t - nL);
            unsigned int nend = std::max(dlag.nL - 1, t - 1);
            unsigned int nelem = nend - nstart + 1;

            arma::vec yold(dlag.nL, arma::fill::zeros);
            yold.tail(nelem) = yall.subvec(nstart, nend);
            yold.elem(arma::find(yold <= EPS)).fill(0.01 / static_cast<double>(dlag.nL));
            yold = arma::reverse(yold) / y_scale;

            arma::vec th = theta_cur.head(dlag.nL); // nL x 1
            arma::vec dhpsi_cur = GainFunc::psi2dhpsi<arma::vec>(th, fgain); // (h'(psi[t]), ..., h'(psi[t+1 - nL]))
            arma::vec Ftmp = yold % dhpsi_cur; // nL x 1
            Ft.head(dlag.nL) = Ftmp % dlag.Fphi;
        }
        else
        {
            Ft.at(1) = 1.;
        }

        if (seasonal_period > 0 && season_in_state)
        {
            unsigned int nstate = theta_cur.n_elem - seasonal_period;
            Ft.at(nstate) = 1.;
        }

        #ifdef DGTF_DO_BOUND_CHECK
        bound_check<arma::vec>(Ft, "func_Ft: Ft");
        #endif
        return Ft;
    }


    static arma::mat psi2theta(
        const arma::vec &psi, // (nT + 1)
        const arma::vec &y, // (nT + 1) x 1
        const std::string &ftrans,
        const std::string &fgain,
        const LagDist &dlag,
        const double &y_scale = 1.
    )
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
            ft.at(t) = func_ft(t, y, ft, hpsi, dlag, ftrans, y_scale);
            Theta.col(t) = psi2theta(t, psi, ft, ftrans, dlag, true);

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
        return TRANS_MAP;
    }
};

inline const std::map<std::string, TransFunc::Transfer> TransFunc::trans_list = TransFunc::map_trans_func();

#endif