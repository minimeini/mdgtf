#pragma once
#ifndef LAGDIST_H
#define LAGDIST_H

#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <RcppArmadillo.h>
#include "distributions.hpp"

// [[Rcpp::plugins(cpp17)]]
// [[Rcpp::depends(RcppArmadillo,BH)]]

/**
 * @file LagDist.hpp
 *
 * @brief Define and manipulate lag distributions Fphi. Lag Distributions are also known as transfer functions.
 *
 * @param _name string: name of the lag distribution, must be one of the lag distribution listed in AVAIL (modified by set_lag_dist).
 * @param _par1 (par1, par2) are two parameters that defined the lag distribution: nbinom(kappa, r); lognorm(mu, sd2) (modified by set_lag_dist).
 * @param _par2 (par1, par2) are two parameters that defined the lag distribution: nbinom(kappa, r); lognorm(mu, sd2) (modified by set_lag_dist).
 *
 *
 * @param _Fphi nL x 1 vector: lag distributions, C.D.F of discrete random variables, characterized by two parameters.
 *
 * @return Fphi
 */
class LagDist : public Dist
{
public:
    LagDist() : Dist(), isnbinom(_isnbinom), par1(_par1), par2(_par2), Fphi(_Fphi)
    {
        init_default();
        return;
    }

    LagDist(
        const std::string &name,
        const double &par1,
        const double &par2) : Dist(), isnbinom(_isnbinom), par1(_par1), par2(_par2), Fphi(_Fphi)
    {
        init(name, par1, par2);
        return;
    }

    const bool &isnbinom;
    void init(
        const std::string &name,
        const double &par1,
        const double &par2)
    {
        _name = name;
        _par1 = par1;
        _par2 = par2;
        lag_list = AVAIL::lag_list;
        _isnbinom = (lag_list[_name] == AVAIL::Dist::nbinomp) ? true : false;

        return;
    }

    void init_default()
    {
        _name = "nbinom";
        _par1 = NB_KAPPA;
        _par2 = NB_R;
        lag_list = AVAIL::lag_list;
        _isnbinom = (lag_list[_name] == AVAIL::Dist::nbinomp) ? true : false;
    }

    const double &par1;
    const double &par2;
    const arma::vec &Fphi;
    std::map<std::string, AVAIL::Dist> lag_list = AVAIL::lag_list;
    /**
     * @brief Set Fphi based on number of lags, type of lag distribution and its corresponding parameters
     *
     *
     */
    void get_Fphi(const unsigned int &nlag)
    {
        _Fphi.set_size(nlag);
        _Fphi.zeros();

        switch (lag_list[_name])
        {
        case AVAIL::Dist::lognorm:
            _Fphi = lognorm::dlognorm(nlag, _par1, _par2);
            break;
        case AVAIL::Dist::nbinomp:
            _Fphi = nbinom::dnbinom(nlag, _par1, _par2);
            break;
        default:
            throw std::invalid_argument("Supported lag distributions: 'lognorm', 'nbinom'.");
            break;
        }

        bound_check<arma::vec>(_Fphi, "void get_Fphi: _Fphi");

        return;
    }

    /**
     * @brief Get P.M.F of the lag distribution. [Checked. OK.s]
     *
     * @param nlag unsigned int
     * @param lag_dist nbinom or lognorm
     * @param lag_par1 kappa or mu
     * @param lag_par2 r or var
     * @return arma::vec
     */
    static arma::vec get_Fphi(
        const unsigned int &nlag,
        const std::string &lag_dist,
        const double &lag_par1,
        const double &lag_par2)
    {
        arma::vec Fphi(nlag, arma::fill::zeros);
        std::map<std::string, AVAIL::Dist> lag_list = AVAIL::lag_list;

        switch (lag_list[lag_dist])
        {
        case AVAIL::Dist::lognorm:
            Fphi = lognorm::dlognorm(nlag, lag_par1, lag_par2);
            break;
        case AVAIL::Dist::nbinomp:
            Fphi = nbinom::dnbinom(nlag, lag_par1, lag_par2);
            break;
        default:
            throw std::invalid_argument("Supported lag distributions: 'lognorm', 'nbinom'.");
        }

        bound_check<arma::vec>(Fphi, "arma::vec get_Fphi: Fphi");
        return Fphi;
    }

    /**
     * @brief Get the optim number of lags that satisfy a specific margin of error.
     *
     * @param error_margin double (default = 0.01): the lower bound of 1 - sum(Fphi)
     * @return nlag - unsigned int.
     */
    static unsigned int get_optim_nlag(
        const std::string &lag_dist = "nbinom",
        const double &lag_par1 = NB_KAPPA,
        const double &lag_par2 = NB_R,
        const double &error_margin = 0.01)
    {
        unsigned int nlag = 1;
        bool cont = true;
        while (cont)
        {
            arma::vec Fphi_tmp = get_Fphi(nlag, lag_dist, lag_par1, lag_par2);
            double prob = arma::accu(Fphi_tmp);

            if (1. - prob <= error_margin)
            {
                cont = false;
            }
            else
            {
                nlag += 1;
            }
        }

        return nlag;
    }

private:
    arma::vec _Fphi; // a vector of the lag distribution CDF at desired length _nL.
    bool _isnbinom;
};

#endif