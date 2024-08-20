#pragma once
#ifndef GAINFUNC_H
#define GAINFUNC_H

#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <RcppArmadillo.h>
// #include <nlopt.h>
#include "utils.h"
// [[Rcpp::plugins(cpp17)]]
// [[Rcpp::depends(RcppArmadillo)]]

/**
 * @brief Define the gain functions.
 *
 * @param psi (nT + 1) x 1 vector: (psi[0], psi[1], ..., psi[nT]) with psi[0] = 0, the latent random walk, i.e., cumulative white noise.
 * @param hpsi (nT + 1) x 1 vector: (h(psi[0]), h(psi[1]), ..., h(psi[nT])), where h is the gain function.
 * @param dhpsi (nT + 1) x 1 vector: (h'(psi[0]), h'(psi[1]), ..., h'(psi[nT])), where h' is the first-order derivative of h(psi[t]) with respect to psi[t].
 *
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

public:
    static const std::map<std::string, AVAIL::Func> gain_list;

    template <typename T>
    static T psi2hpsi(
        const T &psi, 
        const std::string &gain_func)
    {
        T hpsi = psi;

        std::map<std::string, AVAIL::Func> gain_list = GainFunc::gain_list;
        switch (gain_list[gain_func])
        {
        case AVAIL::Func::ramp:
        {
            hpsi.elem(arma::find(hpsi < EPS)).fill(EPS);
        }
        break;
        case AVAIL::Func::exponential:
        {
            hpsi.elem(arma::find(hpsi > UPBND)).fill(UPBND);
            hpsi = arma::exp(hpsi);
        }
        break;
        case AVAIL::Func::identity:
        {
            // do nothing
        }
        break;
        case AVAIL::Func::softplus:
        {
            hpsi.elem(arma::find(hpsi > UPBND)).fill(UPBND);
            T hpsi_tmp = arma::exp(hpsi);
            hpsi = arma::log(1. + hpsi_tmp);
        }
        break;
        default:
        {
            // Use identity gain: do nothing
        }
        break;
        }

        bound_check<T>(hpsi, "psi2hpsi");

        return hpsi;
    }

    static double psi2hpsi(
        const double &psi,
        const std::string &gain_func)
    {
        std::map<std::string, AVAIL::Func> gain_list = GainFunc::gain_list;
        double hpsi = psi;

        switch (gain_list[gain_func])
        {
        case AVAIL::Func::ramp:
        {
            hpsi = std::max(psi, EPS);
        }
        break;
        case AVAIL::Func::exponential:
        {
            hpsi = std::exp(std::min(psi, UPBND));
        }
        break;
        case AVAIL::Func::identity:
        {
            // do nothing
        }
        break;
        case AVAIL::Func::softplus:
        {
            double htmp = std::exp(std::min(psi, UPBND));
            hpsi = std::log( 1. + htmp);
        }
        break;
        default:
        {
            // Use identity gain: do nothing
        }
        break;
        }

        bound_check(hpsi, "psi2hpsi");
        return hpsi;
    }

    static double hpsi2psi(
        const double &hpsi,
        const std::string &gain_func)
    {
        std::map<std::string, AVAIL::Func> gain_list = GainFunc::gain_list;
        double psi;

        switch (gain_list[gain_func])
        {
        case AVAIL::Func::ramp:
        {
            psi = hpsi;
        }
        break;
        case AVAIL::Func::exponential:
        {
            psi = std::log(std::abs(hpsi) + EPS);
        }
        break;
        case AVAIL::Func::identity:
        {
            // do nothing
            psi = hpsi;
        }
        break;
        case AVAIL::Func::softplus:
        {
            psi = std::log(std::exp(hpsi) - 1.);
        }
        break;
        default:
        {
            // Use identity gain: do nothing
            throw std::invalid_argument("Unknown gain function");
        }
        break;
        }

        bound_check(hpsi, "hpsi2psi");
        return hpsi;
    }

    /**
     * @brief First-order derivative of the gain function, h'(psi[t]).
     * 
     */
    template <typename T>
    static T psi2dhpsi(
        const T &psi,
        const std::string &gain_func
    )
    {
        std::map<std::string, AVAIL::Func> gain_list = GainFunc::gain_list;
        T dhpsi = psi;

        switch (gain_list[gain_func])
        {
        case AVAIL::Func::ramp: // Ramp
        {
            throw std::invalid_argument("Ramp function is non-differentiable.");
        }
        break;
        case AVAIL::Func::exponential: // Exponential
        {
            T tmp = psi;
            tmp.elem(arma::find(tmp > UPBND)).fill(UPBND);
            dhpsi = arma::exp(tmp);
        }
        break;
        case AVAIL::Func::identity: // Identity
        {
            dhpsi.ones();
        }
        break;
        case AVAIL::Func::softplus: // Softplus
        {
            T tmp = -psi;
            tmp.elem(arma::find(tmp > UPBND)).fill(UPBND);
            dhpsi = 1. / (1. + arma::exp(tmp));
        }
        break;
        default:
        {
            dhpsi.ones();
        }
        break;
        }

        bound_check<T>(dhpsi, "psi2dhpsi");

        return dhpsi;
    }

    static double psi2dhpsi(
        const double &psi,
        const std::string &gain_func)
    {
        std::map<std::string, AVAIL::Func> gain_list = GainFunc::gain_list;
        double dhpsi = psi;

        switch (gain_list[gain_func])
        {
        case AVAIL::Func::ramp: // Ramp
        {
            throw std::invalid_argument("Ramp function is non-differentiable.");
        }
        break;
        case AVAIL::Func::exponential: // Exponential
        {
            double tmp = std::min(psi, UPBND);
            dhpsi = std::exp(tmp);
        }
        break;
        case AVAIL::Func::identity: // Identity
        {
            dhpsi = 1.;
        }
        break;
        case AVAIL::Func::softplus: // Softplus
        {
            double tmp = -psi;
            tmp = std::min(tmp, UPBND);
            dhpsi = 1. / (1. + std::exp(tmp));
        }
        break;
        default:
        {
            dhpsi = 1.;
        }
        break;
        }

        bound_check(dhpsi, "psi2dhpsi");
        return dhpsi;
    }

    /**
     * @brief Return ( h(psi[t-L+1]), ..., h(psi[t]) )
     * 
     * @param t 
     * @return arma::vec 
     */
    static arma::vec Fhpsi(
        const arma::vec &hpsi,
        const unsigned int &t, 
        const unsigned int &nL, 
        const bool &reverse = false)
    {
        unsigned int nelem = std::min(t, nL);
        arma::vec Fh = hpsi.subvec(t - nelem + 1, t); // nelem x k
        if (reverse) { Fh = arma::reverse(Fh); }
        return Fh;
    }

    /**
     * @brief Return ( h'(psi[t-L+1]), ..., h'(psi[t]) )
     *
     * @param t
     * @return arma::vec
     */
    static arma::vec Fdhpsi(const arma::vec &dhpsi, const unsigned int &t, const unsigned int &nL, const bool &reverse = false)
    {
        unsigned int nelem = std::min(t, nL);
        arma::vec Fdh = dhpsi.subvec(t - nelem + 1, t); // nelem x k
        if (reverse) { Fdh = arma::reverse(Fdh); }
        return Fdh;
    }
};

inline const std::map<std::string, AVAIL::Func> GainFunc::gain_list = GainFunc::map_gain_func();

#endif