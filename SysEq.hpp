#pragma once
#ifndef SYSEQ_H
#define SYSEQ_H

#include <iostream>
#include <iomanip>
#include <map>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <RcppArmadillo.h>
#include "LagDist.hpp"
#include "TransFunc.hpp"
#include "GainFunc.hpp"

// [[Rcpp::plugins(cpp17)]]
// [[Rcpp::depends(RcppArmadillo)]]

class SysEq
{
public:
    enum Evolution
    {
        identity, // autoregression
        shift, // discretized-Hawkes process
        nbinom // distributed lags
    };

    static const std::map<std::string, Evolution> sys_list;


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
    static arma::mat init_Gt(
        const unsigned int &nP, 
        const LagDist &dlag, 
        const std::string &fsys = "shift", 
        const unsigned int &seasonal_period = 0, 
        const bool &season_in_state = false)
    {
        unsigned int nstate = nP;
        if (season_in_state)
        {
            nstate -= seasonal_period;
        }
        
        std::map<std::string, Evolution> sys_list = SysEq::sys_list;
        arma::mat G0(nP, nP, arma::fill::zeros);
        if (sys_list[fsys] == Evolution::nbinom)
        {
            G0.at(0, 0) = 1.;

            arma::vec iter_coef = nbinom::iter_coef(dlag.par1, dlag.par2);
            double coef_now = std::pow(1. - dlag.par1, dlag.par2);

            G0.at(1, 0) = coef_now;                      // (1 - kappa)^r
            G0.submat(1, 1, 1, nstate - 1) = iter_coef.t(); // c(r,1)(-kappa)^1, ..., c(r,r)(-kappa)^r

            for (unsigned int i = 2; i < nstate; i++)
            {
                G0.at(i, i - 1) = 1.;
            }
        }
        else if (sys_list[fsys] == Evolution::shift)
        {
            G0.at(0, 0) = 1.;

            for (unsigned int i = 1; i < dlag.nL; i++)
            {
                G0.at(i, i - 1) = 1.;
            }
        }
        else if (sys_list[fsys] == Evolution::identity)
        {
            G0 = arma::eye<arma::mat>(nP, nP);
        }
        else
        {
            throw std::invalid_argument("SysEq::init_Gt - Unknown system equation.");
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
        const std::string &fsys,
        const std::string &fgain,
        const LagDist &dlag,
        // const Model &model,
        const arma::vec &theta_cur, // nP x 1, (psi[t], f[t-1], ..., f[t-r])
        const double &ycur,
        const unsigned int &seasonal_period = 0,
        const bool &season_in_state = false
    )
    {
        std::map<std::string, Evolution> sys_list = SysEq::sys_list;
        const unsigned int nP = theta_cur.n_elem;
        arma::vec theta_next(nP, arma::fill::zeros); // nP x 1
        // theta_next.copy_size(theta_cur);

        unsigned int nr = nP - 1;
        if (season_in_state)
        {
            nr -= seasonal_period;
        }
        

        switch (sys_list[fsys])
        {
        case Evolution::nbinom:
        {
            double hpsi = GainFunc::psi2hpsi(theta_cur.at(0), fgain);
            theta_next.at(0) = theta_cur.at(0); // Expectation of random walk.
            theta_next.at(1) = TransFunc::transfer_iterative(
                theta_cur.subvec(1, nr), // f[t-1], ..., f[t-r]
                hpsi, ycur, dlag.par1, dlag.par2);

            theta_next.subvec(2, nr) = theta_cur.subvec(1, nr - 1);
            break;
        }
        case Evolution::shift:
        {
            // theta_next = model.transfer.G0 * theta_cur;
            theta_next.at(0) = theta_cur.at(0);
            theta_next.subvec(1, nr) = theta_cur.subvec(0, nr - 1);
            break;
        }
        case Evolution::identity:
        {
            theta_next.subvec(0, nr) = theta_cur.subvec(0, nr);
            break;
        }
        default:
        {
            throw std::invalid_argument("SysEq::func_gt - Unknown system equation.");
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
     * @brief First-order derivative of g[t] w.r.t theta[t]; used in the calculation of R[t] = G[t] C[t-1] t(G[t]) + W[t].
     *
     * @param model
     * @param mt_old
     * @param yold
     * @return arma::mat
     */
    static void func_Gt(
        arma::mat &Gt, // nP x nP, must be already initialized via `TransFunc::init_Gt()`
        const std::string &fsys,
        const std::string &fgain,
        const LagDist &dlag,
        const arma::vec &mt_old,
        const double &yold)
    {
        std::map<std::string, Evolution> sys_list = SysEq::sys_list;
        if (sys_list[fsys] == Evolution::nbinom)
        {
            double coef_now = std::pow(1. - dlag.par1, dlag.par2);
            double dhpsi_now = GainFunc::psi2dhpsi(mt_old.at(0), fgain);
            Gt.at(1, 0) = coef_now * yold * dhpsi_now;
        }

        return;
    }




private:
    static std::map<std::string, Evolution> map_sys_eq()
    {
        std::map<std::string, Evolution> MAP;
        MAP["identity"] = Evolution::identity;
        MAP["shift"] = Evolution::shift;
        MAP["nbinom"] = Evolution::nbinom;
        return MAP;
    }
};

inline const std::map<std::string, SysEq::Evolution> SysEq::sys_list = SysEq::map_sys_eq();

#endif