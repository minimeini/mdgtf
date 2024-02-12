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
#include <nlopt.h>
#include "utils.h"
// [[Rcpp::plugins(cpp17)]]
// [[Rcpp::depends(RcppArmadillo,nloptr)]]

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
public:
    GainFunc() : psi(_psi), hpsi(_hpsi), dhpsi(_dhpsi)
    {
        init_default();
        return;
    };

    GainFunc(
        const std::string &name, 
        const Dim &dim) : psi(_psi), hpsi(_hpsi), dhpsi(_dhpsi)
    {
        init(name, dim);
        return;
    };

    void init(
        const std::string &name,
        const Dim &dim)
    {
        _gain_list = AVAIL::map_gain_func();
        _name = name;

        _psi.set_size(dim.nT + 1);
        _psi.zeros();

        _hpsi = _psi;
        _dhpsi = _hpsi;

        return;
    }

    void init_default()
    {
        Dim dim;
        dim.init_default();

        _gain_list = AVAIL::map_gain_func();
        _name = "identity";

        _psi.set_size(dim.nT + 1);
        _psi.zeros();

        _hpsi = _psi;
        _dhpsi = _hpsi;
    }

    const arma::vec &psi;   // Read-only. (nT + 1) x 1 vector: (psi[0], psi[1], ..., psi[nT]) with psi[0] = 0, the latent random walk, i.e., cumulative white noise.
    const arma::vec &hpsi;  // Read-only. (nT + 1) x 1 vector: (h(psi[0]), h(psi[1]), ..., h(psi[nT]))
    const arma::vec &dhpsi; // Read-only. (nT + 1) x 1 vector: (h'(psi[0]), h'(psi[1]), ..., h'(psi[nT])), where h' is the first-order derivative of h(psi[t]) with respect to psi[t].

    void update_psi(const arma::vec &psi_) { _psi = psi_; }
    void update_hpsi(const arma::vec &hpsi_) { _hpsi = hpsi_; }
    void update_dhpsi(const arma::vec &dhpsi_) { _dhpsi = dhpsi_; }



    /**
     * @brief Gain function h(psi[t]). [Checked. OK.]
     * 
     */
    void psi2hpsi()
    {
        _hpsi = _psi;
        switch (_gain_list[_name])
        {
        case AVAIL::Func::ramp:
        {
            _hpsi.elem(arma::find(_hpsi < EPS)).fill(EPS);
        }
        break;
        case AVAIL::Func::exponential:
        {
            _hpsi.elem(arma::find(_hpsi > UPBND)).fill(UPBND);
            _hpsi = arma::exp(_hpsi);
        }
        break;
        case AVAIL::Func::identity:
        {
            // do nothing
        }
        break;
        case AVAIL::Func::softplus:
        {
            _hpsi.elem(arma::find(_hpsi > UPBND)).fill(UPBND);
            arma::vec hpsi_tmp = arma::exp(_hpsi);
            _hpsi = arma::log(1. + hpsi_tmp);
        }
        break;
        default:
        {
            // Use identity gain: do nothing
            _name = "identity";
        }
        break;
        }

        bound_check<arma::vec>(_hpsi, "psi2hpsi");

        return;
    }

    static arma::vec psi2hpsi(
        const arma::vec &psi, 
        const std::string &gain_func)
    {
        std::map<std::string, AVAIL::Func> _gain_list = AVAIL::map_gain_func();
        arma::vec hpsi = psi;

        switch (_gain_list[gain_func])
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
            arma::vec hpsi_tmp = arma::exp(hpsi);
            hpsi = arma::log(1. + hpsi_tmp);
        }
        break;
        default:
        {
            // Use identity gain: do nothing
        }
        break;
        }

        bound_check<arma::vec>(hpsi, "psi2hpsi");

        return hpsi;
    }

    /**
     * @brief First-order derivative of the gain function, h'(psi[t]).
     * 
     */
    void psi2dhpsi()
    {
        _dhpsi.copy_size(_psi);

        switch (_gain_list[_name])
        {
        case AVAIL::Func::ramp: // Ramp
        {
            throw std::invalid_argument("Ramp function is non-differentiable.");
        }
        break;
        case AVAIL::Func::exponential: // Exponential
        {
            arma::mat tmp = _psi;
            tmp.elem(arma::find(tmp > UPBND)).fill(UPBND);
            _dhpsi = arma::exp(tmp);
        }
        break;
        case AVAIL::Func::identity: // Identity
        {
            _dhpsi.ones();
        }
        break;
        case AVAIL::Func::softplus: // Softplus
        {
            arma::mat tmp = -_psi;
            tmp.elem(arma::find(tmp > UPBND)).fill(UPBND);
            _dhpsi = 1. / (1. + arma::exp(tmp));
        }
        break;
        default:
        {
            _dhpsi.ones();
            _name = "identity";
        }
        break;
        }

        bound_check<arma::vec>(_dhpsi, "psi2dhpsi");
    }

    /**
     * @brief Return ( h(psi[t-L+1]), ..., h(psi[t]) )
     * 
     * @param t 
     * @return arma::vec 
     */
    arma::vec Fhpsi(
        const unsigned int &t, 
        const Dim &dim, 
        const bool &reverse = false)
    {
        unsigned int nelem = std::min(t, dim.nL);
        arma::vec Fh = _hpsi.subvec(t - nelem + 1, t); // nelem x k
        if (reverse) { Fh = arma::reverse(Fh); }
        return Fh;
    }

    /**
     * @brief Return ( h'(psi[t-L+1]), ..., h'(psi[t]) )
     *
     * @param t
     * @return arma::vec
     */
    arma::vec Fdhpsi(const unsigned int &t, const Dim &dim, const bool &reverse = false)
    {
        unsigned int nelem = std::min(t, dim.nL);
        arma::vec Fdh = _dhpsi.subvec(t - nelem + 1, t); // nelem x k
        if (reverse) { Fdh = arma::reverse(Fdh); }
        return Fdh;
    }

private:
    std::map<std::string, AVAIL::Func> _gain_list;
    std::string _name = "softplus"; // name of the gain function.
    arma::vec _psi;  // (nT + 1) x 1 vector: (psi[0], psi[1], ..., psi[nT]) with psi[0] = 0, the latent random walk, i.e., cumulative white noise.
    arma::vec _hpsi; // (nT + 1) x 1 vector: (h(psi[0]), h(psi[1]), ..., h(psi[nT]))
    arma::vec _dhpsi; // (nT + 1) x 1 vector: (h'(psi[0]), h'(psi[1]), ..., h'(psi[nT])), where h' is the first-order derivative of h(psi[t]) with respect to psi[t].
};



#endif