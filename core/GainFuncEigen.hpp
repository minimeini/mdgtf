#pragma once
#ifndef GAINFUNC_EIGEN_H
#define GAINFUNC_EIGEN_H

#include <algorithm>
#include <cmath>
#include <map>
#include <stdexcept>
#include <string>
#include <RcppEigen.h>
#include <Eigen/Dense>
#include "../utils/constants.h"
#include "../utils/definition.h"

// [[Rcpp::plugins(cpp17)]]
// [[Rcpp::depends(RcppEigen)]]

/**
 * @brief Eigen-based version of GainFunc.hpp.
 *
 * psi, hpsi, and dhpsi are typically (nT + 1) length vectors with psi[0] = 0.
 */
class GainFunc
{
private:
    static std::map<std::string, AVAIL::Func> map_gain_func()
    {
        std::map<std::string, AVAIL::Func> GAIN_MAP;

        GAIN_MAP["ramp"] = AVAIL::Func::ramp;
        GAIN_MAP["exponential"] = AVAIL::Func::exponential;
        GAIN_MAP["identity"] = AVAIL::Func::identity;
        GAIN_MAP["softplus"] = AVAIL::Func::softplus;
        GAIN_MAP["logistic"] = AVAIL::Func::logistic;
        return GAIN_MAP;
    }

    static inline double trunc_exp_scalar(const double &x)
    {
        return std::exp(std::min(x, UPBND));
    }

public:
    static const std::map<std::string, AVAIL::Func> gain_list;


    static Eigen::ArrayXXd psi2hpsi(
        const Eigen::ArrayXXd &psi,
        const std::string &gain_func)
    {
        Eigen::ArrayXXd hpsi = psi;

        switch (gain_list.at(gain_func))
        {
        case AVAIL::Func::ramp:
            hpsi = psi.max(EPS);
            break;
        case AVAIL::Func::exponential:
            hpsi = psi.min(UPBND).exp();
            break;
        case AVAIL::Func::identity:
            break;
        case AVAIL::Func::softplus:
            hpsi = (psi.min(UPBND).exp()).log1p();
            break;
        default:
            break;
        }
        return hpsi;
    }


    static Eigen::ArrayXd psi2hpsi(
        const Eigen::ArrayXd &psi,
        const std::string &gain_func)
    {
        Eigen::ArrayXd hpsi = psi;

        std::map<std::string, AVAIL::Func> gain_map = GainFunc::gain_list;
        switch (gain_map[gain_func])
        {
        case AVAIL::Func::ramp:
        {
            hpsi = psi.max(EPS);
            break;
        }
        case AVAIL::Func::exponential:
        {
            hpsi = psi.min(UPBND).exp();
            break;
        }
        case AVAIL::Func::identity:
        {
            break;
        }
        case AVAIL::Func::softplus:
        {
            hpsi = (psi.min(UPBND).exp()).log1p();
            break;
        }
        default:
        {
            break;
        }
        }

        return hpsi;
    }


    static double psi2hpsi(
        const double &psi,
        const std::string &gain_func)
    {
        std::map<std::string, AVAIL::Func> gain_map = GainFunc::gain_list;
        double hpsi = psi;

        switch (gain_map[gain_func])
        {
        case AVAIL::Func::ramp:
        {
            hpsi = std::max(psi, EPS);
            break;
        }
        case AVAIL::Func::exponential:
        {
            hpsi = trunc_exp_scalar(psi);
            break;
        }
        case AVAIL::Func::identity:
        {
            break;
        }
        case AVAIL::Func::softplus:
        {
            double htmp = trunc_exp_scalar(psi);
            hpsi = std::log(1. + htmp);
            break;
        }
        default:
        {
            break;
        }
        }

        return hpsi;
    }

    static double hpsi2psi(
        const double &hpsi,
        const std::string &gain_func)
    {
        std::map<std::string, AVAIL::Func> gain_map = GainFunc::gain_list;
        double psi;

        switch (gain_map[gain_func])
        {
        case AVAIL::Func::ramp:
        {
            psi = hpsi;
            break;
        }
        case AVAIL::Func::exponential:
        {
            psi = std::log(std::abs(hpsi) + EPS);
            break;
        }
        case AVAIL::Func::identity:
        {
            psi = hpsi;
            break;
        }
        case AVAIL::Func::softplus:
        {
            psi = std::log(std::expm1(hpsi));
            break;
        }
        default:
        {
            throw std::invalid_argument("Unknown gain function");
        }
        }

        return psi;
    }

    /**
     * @brief First-order derivative of the gain function, h'(psi[t]).
     */
    static Eigen::ArrayXd psi2dhpsi(
        const Eigen::ArrayXd &psi,
        const std::string &gain_func)
    {
        std::map<std::string, AVAIL::Func> gain_map = GainFunc::gain_list;
        Eigen::ArrayXd dhpsi = psi;

        switch (gain_map[gain_func])
        {
        case AVAIL::Func::ramp:
        {
            throw std::invalid_argument("Ramp function is non-differentiable.");
        }
        case AVAIL::Func::exponential:
        {
            dhpsi = psi.min(UPBND).exp();
            break;
        }
        case AVAIL::Func::identity:
        {
            dhpsi.setOnes();
            break;
        }
        case AVAIL::Func::softplus:
        {
            dhpsi = 1. / (1. + (-psi).min(UPBND).exp());
            break;
        }
        default:
        {
            dhpsi.setOnes();
        }
        }

        return dhpsi;
    }


    static double psi2dhpsi(
        const double &psi,
        const std::string &gain_func)
    {
        std::map<std::string, AVAIL::Func> gain_map = GainFunc::gain_list;
        double dhpsi = psi;

        switch (gain_map[gain_func])
        {
        case AVAIL::Func::ramp:
        {
            throw std::invalid_argument("Ramp function is non-differentiable.");
        }
        case AVAIL::Func::exponential:
        {
            dhpsi = trunc_exp_scalar(psi);
            break;
        }
        case AVAIL::Func::identity:
        {
            dhpsi = 1.;
            break;
        }
        case AVAIL::Func::softplus:
        {
            double tmp = -psi;
            tmp = std::min(tmp, UPBND);
            dhpsi = 1. / (1. + std::exp(tmp));
            break;
        }
        default:
        {
            dhpsi = 1.;
        }
        }

        return dhpsi;
    }

    /**
     * @brief Return ( h(psi[t-L+1]), ..., h(psi[t]) )
     */
    static Eigen::VectorXd Fhpsi(
        const Eigen::VectorXd &hpsi,
        const unsigned int &t,
        const unsigned int &nL,
        const bool &reverse = false)
    {
        unsigned int nelem = std::min(t, nL);
        if (nelem == 0)
        {
            return Eigen::VectorXd(0);
        }

        const Eigen::Index start = static_cast<Eigen::Index>(t - nelem + 1);
        Eigen::VectorXd Fh = hpsi.segment(start, static_cast<Eigen::Index>(nelem));
        if (reverse)
        {
            std::reverse(Fh.begin(), Fh.end());
        }
        return Fh;
    }

    /**
     * @brief Return ( h'(psi[t-L+1]), ..., h'(psi[t]) )
     */
    static Eigen::VectorXd Fdhpsi(
        const Eigen::VectorXd &dhpsi,
        const unsigned int &t,
        const unsigned int &nL,
        const bool &reverse = false)
    {
        unsigned int nelem = std::min(t, nL);
        if (nelem == 0)
        {
            return Eigen::VectorXd(0);
        }

        const Eigen::Index start = static_cast<Eigen::Index>(t - nelem + 1);
        Eigen::VectorXd Fdh = dhpsi.segment(start, static_cast<Eigen::Index>(nelem));
        if (reverse)
        {
            std::reverse(Fdh.begin(), Fdh.end());
        }
        return Fdh;
    }
};

inline const std::map<std::string, AVAIL::Func> GainFunc::gain_list = GainFunc::map_gain_func();

#endif
