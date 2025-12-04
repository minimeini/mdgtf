#pragma once
#ifndef LAGDIST_EIGEN_H
#define LAGDIST_EIGEN_H

#include <cmath>
#include <map>
#include <string>
#include <stdexcept>
#include <algorithm>
#include <RcppEigen.h>
#include <Eigen/Dense>
#include "../utils/constants.h"
#include "../utils/definition.h"

// [[Rcpp::plugins(cpp17)]]
// [[Rcpp::depends(RcppEigen)]]

/**
 * @brief Eigen-based version of LagDist.hpp (lag/transfer distributions).
 */
class LagDist : public Dist
{
private:
    double prob_thres = 0.995;

public:
    static const std::map<std::string, AVAIL::Dist> lag_list;

    unsigned int nL = 30;       // number of lags
    bool truncated = true;
    bool rescaled = true;
    Eigen::VectorXd Fphi;      // lag distribution CDF at desired length _nL

    LagDist() : Dist()
    {
        name = "lognorm";
        par1 = LN_MU;
        par2 = LN_SD2;
        nL = 30;
        truncated = true;
        rescaled = true;
        Fphi = get_Fphi(nL, name, par1, par2);
        return;
    } // LagDist default constructor

    LagDist(
        const std::string &name_in,
        const double &par1_in,
        const double &par2_in,
        const bool &truncated_in,
        const bool &rescaled_in = true
    ) : Dist()
    {
        const std::map<std::string, AVAIL::Dist> &lag_map = LagDist::lag_list;
        name = name_in;
        par1 = par1_in;
        par2 = par2_in;
        truncated = truncated_in;
        rescaled = rescaled_in;

        const auto it = lag_map.find(name);
        if (it == lag_map.end())
        {
            throw std::invalid_argument("LagDist::init - unknown lag distribution.");
        }

        if (it->second == AVAIL::Dist::uniform)
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
    } // LagDist constructor


    LagDist(const Rcpp::List &opts) : Dist()
    {
        
        if (opts.containsElementNamed("name"))
        {
            name = Rcpp::as<std::string>(opts["name"]);

            if (opts.containsElementNamed("par1"))
            {
                par1 = Rcpp::as<double>(opts["par1"]);
            }
            else if (name == "lognorm")
            {
                par1 = LN_MU;
            }
            else
            {
                throw std::invalid_argument("LagDist::init - opts must contain element 'par1' (lag distribution parameter 1) when lag dist is not lognorm.");
            } // set par1

            if (opts.containsElementNamed("par2"))
            {
                par2 = Rcpp::as<double>(opts["par2"]);
            }
            else if (name == "lognorm")
            {
                par2 = LN_SD2;
            }
            else
            {
                throw std::invalid_argument("LagDist::init - opts must contain element 'par2' (lag distribution parameter 2) when lag dist is not lognorm.");
            } // set par2
        } // if name provided
        else
        {
            if (opts.containsElementNamed("par1") || opts.containsElementNamed("par2"))
            {
                throw std::invalid_argument("LagDist::init - opts must contain element 'name' if 'par1' or 'par2' are provided.");
            }

            name = "lognorm";
            par1 = LN_MU;
            par2 = LN_SD2;
        } // else name not provided

        truncated = true;
        if (opts.containsElementNamed("truncated"))
        {
            truncated = Rcpp::as<bool>(opts["truncated"]);
        }

        rescaled = true;
        if (opts.containsElementNamed("rescaled"))
        {
            rescaled = Rcpp::as<bool>(opts["rescaled"]);
        }

        const std::map<std::string, AVAIL::Dist> &lag_map = LagDist::lag_list;
        const auto it = lag_map.find(name);
        if (it == lag_map.end())
        {
            throw std::invalid_argument("LagDist::init - unknown lag distribution.");
        }

        if (it->second == AVAIL::Dist::uniform)
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
            if (rescaled)
            {
                Fphi /= Fphi.sum();
            }
        }
        return;
    } // LagDist constructor from Rcpp::List


private:

    /**
     * @brief Get P.M.F of the lag distribution.
     */
    static Eigen::VectorXd get_Fphi(
        const unsigned int &nlag,
        const std::string &lag_dist,
        const double &lag_par1,
        const double &lag_par2)
    {
        Eigen::VectorXd Fphi = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(nlag));
        const std::map<std::string, AVAIL::Dist> &lag_map = LagDist::lag_list;

        const auto it = lag_map.find(lag_dist);
        if (it == lag_map.end())
        {
            throw std::invalid_argument("Supported lag distributions: 'lognorm', 'nbinom', 'uniform'.");
        }

        switch (it->second)
        {
        case AVAIL::Dist::lognorm:
        {
            Fphi = dlognorm(nlag, lag_par1, lag_par2);
            break;
        }
        case AVAIL::Dist::nbinomp:
        {
            Fphi = dnbinom(nlag, lag_par1, lag_par2);
            break;
        }
        case AVAIL::Dist::uniform:
        {
            Fphi.setOnes();
            break;
        }
        default:
            throw std::invalid_argument("Supported lag distributions: 'lognorm', 'nbinom', 'uniform'.");
        }

        return Fphi;
    } // get_Fphi

    /**
     * @brief If number of lags changes, run `get_nlag` then `get_Fphi` to update lag distribution.
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

        double nlag_ = static_cast<double>(min_lag);
        const std::map<std::string, AVAIL::Dist> &lag_map = LagDist::lag_list;
        const auto it = lag_map.find(lag_dist);

        if (it == lag_map.end())
        {
            throw std::invalid_argument("LagDist::get_nlag - unknown lag distribution.");
        }

        switch (it->second)
        {
        case AVAIL::Dist::lognorm:
        {
            nlag_ = R::qlnorm(prob, lag_par1, std::sqrt(lag_par2), 1, 0);
            break;
        }
        case AVAIL::Dist::nbinomp:
        {
            nlag_ = R::qnbinom(prob, lag_par2, 1. - lag_par1, true, false);
            break;
        }
        case AVAIL::Dist::nbinomm:
        {
            double prob_succ = lag_par2 / (lag_par1 + lag_par2);
            nlag_ = R::qnbinom(prob, lag_par2, prob_succ, true, false);
            break;
        }
        case AVAIL::Dist::gamma:
        {
            nlag_ = R::qgamma(prob, lag_par1, 1. / lag_par2, true, false);
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

        unsigned int nlag = static_cast<unsigned int>(nlag_);
        nlag = std::max(nlag, min_lag);
        nlag = std::min(nlag, max_lag);
        return nlag;
    } // get_nlag

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
    } // map_lag_dist

    static double plognorm(
        const double d,
        const double mu,
        const double sigmasq)
    {
        const double z = -(std::log(d) - mu) / std::sqrt(2. * sigmasq);
        return 0.5 * std::erfc(z);
    } // plognorm

    static double dlognorm(
        const double &lag,
        const double &mu,
        const double &sd2)
    {
        double output = plognorm(lag, mu, sd2);
        if (lag > 1.)
        {
            output -= plognorm(lag - 1., mu, sd2);
        }

        return output;
    } // dlognorm0

    static Eigen::VectorXd dlognorm(
        const unsigned int &nL,
        const double &mu,
        const double &sd2)
    {
        Eigen::VectorXd output(static_cast<Eigen::Index>(nL));
        for (unsigned int d = 0; d < nL; d++)
        {
            output(static_cast<Eigen::Index>(d)) = dlognorm(static_cast<double>(d) + 1., mu, sd2);
        }
        return output;
    } // dlognorm_vec

    static Eigen::VectorXd dnbinom(
        const unsigned int &nL,
        const double &kappa,
        const double &r)
    {
        Eigen::VectorXd output(static_cast<Eigen::Index>(nL));
        const double c3 = std::pow(1. - kappa, r);
        for (unsigned int d = 0; d < nL; d++)
        {
            output(static_cast<Eigen::Index>(d)) = dnbinom(static_cast<double>(d), kappa, r, c3);
        }
        return output;
    } // dnbinom_vec

    static double dnbinom(
        const double &y,
        const double &kappa,
        const double &r,
        const double &c3)
    {
        const double log_coef = R::lchoose(r + y - 1., y);
        const double log_prob = log_coef + y * std::log(kappa) + std::log(c3);
        return std::exp(log_prob);
    }

};

inline const std::map<std::string, AVAIL::Dist> LagDist::lag_list = LagDist::map_lag_dist();

#endif
