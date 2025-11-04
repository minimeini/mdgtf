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
// [[Rcpp::depends(RcppArmadillo)]]

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
private:
    double prob_thres = 0.995;
public:
    static const std::map<std::string, AVAIL::Dist> lag_list;
    unsigned int nL = 0; // number of lags
    bool truncated = true;
    arma::vec Fphi; // a vector of the lag distribution CDF at desired length _nL.

    LagDist() : Dist()
    {
        name = "nbinom";
        par1 = NB_KAPPA;
        par2 = NB_R;
        nL = 0;
        truncated = false;
        return;
    }

    LagDist(
        const std::string &name,
        const double &par1,
        const double &par2, 
        const bool &truncated) : Dist()
    {
        init(name, par1, par2, truncated);
        return;
    }

    void init(
        const std::string &name_in,
        const double &par1_in,
        const double &par2_in,
        const bool &truncated_in)
    {
        std::map<std::string, AVAIL::Dist> lag_list = LagDist::lag_list;
        name = name_in;
        par1 = par1_in;
        par2 = par2_in;
        truncated = truncated_in;

        if (lag_list[name] == AVAIL::Dist::uniform)
        {
            nL = static_cast<unsigned int>(par1);
        }
        else if (truncated)
        {
            nL = get_nlag(name, par1, par2, prob_thres);
        }
        else
        {
            nL = 0;
        }

        if (truncated)
        {
            Fphi = get_Fphi(nL, name, par1, par2);
        }

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
        std::map<std::string, AVAIL::Dist> lag_list = LagDist::lag_list;

        switch (lag_list[lag_dist])
        {
        case AVAIL::Dist::lognorm:
        {
            Fphi = lognorm::dlognorm(nlag, lag_par1, lag_par2);
            break;
        }
        case AVAIL::Dist::nbinomp:
        {
            Fphi = nbinom::dnbinom(nlag, lag_par1, lag_par2);
            break;
        }
        case AVAIL::Dist::uniform:
        {
            Fphi.ones();
            break;
        }
        default:
            throw std::invalid_argument("Supported lag distributions: 'lognorm', 'nbinom'.");
        }

        #ifdef DGTF_DO_BOUND_CHECK
        bound_check<arma::vec>(Fphi, "arma::vec get_Fphi: Fphi");
        #endif
        
        return Fphi;
    }

    static arma::vec get_Fphi(const LagDist &dlag)
    {
        if (dlag.nL == 0)
        {
            throw std::invalid_argument("LagDist::get_Fphi: dlag.nL must be a positive integer.");
        }
        return get_Fphi(dlag.nL, dlag.name, dlag.par1, dlag.par2);
    }

    /**
     * @brief Fist-order derivative of the P.M.F of the lag distribution w.r.t. its two parameters (mapped to the whole real lines). For non-negative parameters, we take its logarithm. For parameters in (0, 1), we takes its logit.
     * 
     * @return arma::mat (nlag x 2) 
     */
    static arma::mat get_Fphi_grad(
        const unsigned int &nlag,
        const std::string &lag_dist,
        const double &lag_par1,
        const double &lag_par2)
    {
        arma::mat Fphi_deriv(nlag, 2, arma::fill::zeros);
        std::map<std::string, AVAIL::Dist> lag_list = LagDist::lag_list;

        for (unsigned int d = 1; d <= nlag; d++)
        {
            switch (lag_list[lag_dist])
            {
            case AVAIL::Dist::lognorm:
            {
                arma::vec dlag_grad = lognorm::dlag_dpar(d, lag_par1, lag_par2);
                Fphi_deriv.row(d - 1) = dlag_grad.t();
                break;
            }
            case AVAIL::Dist::nbinomp:
            {
                double dlag_dlogitkappa = nbinom::dlag_dlogitkappa(d, lag_par1, lag_par2);
                Fphi_deriv.at(d - 1, 0) = dlag_dlogitkappa;
                break;
            }
            case AVAIL::Dist::uniform:
            {
                Fphi_deriv.zeros();
                break;
            }
            default:
            {
                throw std::invalid_argument("Supported lag distributions: 'lognorm', 'nbinom'.");
            }
            }
        }

        return Fphi_deriv;
    }


    /**
     * @brief If number of lags changes, just run `get_nlag` and then run `get_Fphi` to update lag distribution.
     * 
     * @param lag_dist 
     * @param lag_par1 
     * @param lag_par2 
     * @param prob 
     * @return unsigned int 
     */
    static unsigned int get_nlag(
        const std::string &lag_dist, 
        const double &lag_par1, 
        const double &lag_par2, 
        const double &prob = 0.995,
        const unsigned int &max_lag = 50,
        const unsigned int &min_lag = MIN_LAG)
    {
        if (prob < 0 || prob > 1)
        {
            throw std::invalid_argument("LagDist::get_nlag: probability must in (0, 1).");
        }
        double nlag_ = (double)min_lag;
        std::map<std::string, AVAIL::Dist> lag_list = LagDist::lag_list;

        switch (lag_list[lag_dist])
        {
        case AVAIL::Dist::lognorm:
        {
            nlag_ = R::qlnorm(prob, lag_par1, std::sqrt(lag_par2), 1, 0);
            break;
        }
        case AVAIL::Dist::nbinomp:
        {
            // par1: kappa; par2: r
            nlag_ = R::qnbinom(prob, lag_par2, 1. - lag_par1, true, false);
            break;
        }
        case AVAIL::Dist::nbinomm:
        {
            // double prob_succ = par2 / (par1 + par2);
            // return R::rnbinom(par2, prob_succ);
            double prob_succ = lag_par2 / (lag_par1 + lag_par2);
            nlag_ = R::qnbinom(prob, lag_par2, prob_succ, true, false);
            break;
        }
        case AVAIL::Dist::gamma:
        {
            nlag_ = R::qgamma(prob, lag_par1, 1./lag_par2, true, false);
            break;
        }
        case AVAIL::Dist::uniform:
        {
            nlag_ = lag_par1;
            break;
        }
        default:
        {
            throw std::invalid_argument("LagDist::get_nlag - unknown lag distribution.");
        }
        }

        #ifdef DGTF_DO_BOUND_CHECK
            bound_check(nlag_, "LagDist::update_nlag: nlag_");
        #endif
        unsigned int nlag = static_cast<unsigned int>(nlag_);

        nlag = std::max(nlag, min_lag);
        nlag = std::min(nlag, max_lag);
        return nlag;
    }

    static unsigned int get_nlag(const LagDist &dlag)
    {
        return get_nlag(dlag.name, dlag.par1, dlag.par2);
    }

private:
    static std::map<std::string, AVAIL::Dist> map_lag_dist()
    {
        std::map<std::string, AVAIL::Dist> LAG_MAP;

        LAG_MAP["lognorm"] = AVAIL::Dist::lognorm;
        LAG_MAP["koyama"] = AVAIL::Dist::lognorm;

        LAG_MAP["nbinom"] = AVAIL::Dist::nbinomp;
        LAG_MAP["nbinomp"] = AVAIL::Dist::nbinomp;
        LAG_MAP["solow"] = AVAIL::Dist::nbinomp;

        LAG_MAP["uniform"] = AVAIL::Dist::uniform;
        LAG_MAP["flat"] = AVAIL::Dist::uniform;
        LAG_MAP["identity"] = AVAIL::Dist::uniform;

        return LAG_MAP;
    }
};

inline const std::map<std::string, AVAIL::Dist> LagDist::lag_list = LagDist::map_lag_dist();

#endif