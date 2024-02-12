#pragma once
#ifndef LINKFUNC_H
#define LINKFUNC_H

#include <map>
#include <string>
#include "definition.h"
/*
We need

psi2theta
theta2ft: 
(1) Exact method: psi2hpsi + hpsi2ft
(2) Approximate method: F'[t]theta[t]

ft2eta: eta = mu0 + ft

ft2mu: link
mu2ft: inverse link
*/



class LinkFunc // between mean and regressor
{
public:
    LinkFunc()
    {
        init_default();
        return;
    }

    LinkFunc(const std::string &name, const double &mu0 = 0.)
    {
        init(name, mu0);
    }

    void init_default()
    {
        _mu0 = 0.;
        _name = "identity";
        _link_list = AVAIL::map_link_func();
        return;
    }

    void init(const std::string &name, const double &mu0 = 0.)
    {
        _mu0 = mu0;
        _name = name;
        _link_list = AVAIL::map_link_func();
    }

    /**
     * zeta: link function the maps regressor eta[t] to mean mu[t];
     *      zeta(eta[t]) = mu[t]
     * Regressor eta[t] = mu0 + f[t], f[t] is the transfer function regression with zero mean.
     *
     * Exponential link:
     *      mu[t] = exp( eta[t] ) = exp( mu0 + f[t] )
     */
    template <class T>
    T ft2mu(const T &ft)
    {
        T eta = _mu0 + ft;
        T mu;
        switch (_link_list[_name])
        {
        case AVAIL::Func::exponential:
        {
            mu = arma::exp(eta);
            break;
        }
        default:
        {
            // Identity gain
            mu = eta;
            break;
        }
        }
        return mu;
    }

    double ft2mu(const double &ft)
    {
        double eta = _mu0 + ft;
        double mu;
        switch (_link_list[_name])
        {
        case AVAIL::Func::exponential:
        {
            mu = std::exp(eta);
            break;
        }
        default:
        {
            // Identity gain
            mu = eta;
            break;
        }
        }
        return mu;
    }

    static double ft2mu(const double &ft, const std::string &link_func, const double &mu0 = 0.)
    {
        std::map<std::string, AVAIL::Func> _link_list = AVAIL::map_link_func();

        double eta = mu0 + ft;
        double mu;
        switch (_link_list[link_func])
        {
        case AVAIL::Func::exponential:
        {
            mu = std::exp(eta);
            break;
        }
        default:
        {
            // Identity gain
            mu = eta;
            break;
        }
        }
        return mu;
    }

    /**
     * Inverset_zeta:
     *      Inverse of the link function that maps mean mu[t] to eta[t].
     *      inv_zeta(mu[t]) = eta[t].
     *
     * Logarithm - inverse of exponential link:
     *          f[t] = log( mu[t] ) - mu0.
     *
     * It is derived from:
     *          eta[t] = log( mu[t] )
     *      mu0 + f[t] = log( mu[t] )
     *
     */
    template <class T>
    T mu2ft(
        const T &mu)
    {
        T eta;
        switch (_link_list[_name])
        {
        case AVAIL::Func::exponential:
        {
            eta = arma::log(mu);
            break;
        }
        default:
        {
            // Identity link
            eta = mu;
            break;
        }
        }

        T ft = eta - _mu0;

        return ft;
    }

    double mu2ft(
        const double &mu)
    {
        double eta;
        switch (_link_list[_name])
        {
        case AVAIL::Func::exponential:
        {
            eta = std::log(mu);
            break;
        }
        default:
        {
            // Identity link
            eta = mu;
            break;
        }
        }

        double ft = eta - _mu0;

        return ft;
    }

private:
    std::map<std::string, AVAIL::Func> _link_list;
    std::string _name;
    double _mu0;

};

#endif