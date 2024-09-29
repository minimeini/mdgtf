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
    enum Func
    {
        identity,
        exponential,
        logistic
    };

    static const std::map<std::string, LinkFunc::Func> link_list;
    /**
     * zeta: link function the maps regressor eta[t] to mean mu[t];
     *      zeta(eta[t]) = mu[t]
     * Regressor eta[t] = mu0 + f[t], f[t] is the transfer function regression with zero mean.
     *
     * Exponential link:
     *      mu[t] = exp( eta[t] ) = exp( mu0 + f[t] )
     */
    static double ft2mu(const double &ft, const std::string &link_func, const double &mu0 = 0.)
    {
        std::map<std::string, LinkFunc::Func> link_list = LinkFunc::link_list;

        double eta = mu0 + ft;
        double mu;
        switch (link_list[link_func])
        {
        case LinkFunc::Func::exponential:
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

    template <class T>
    static T ft2mu(const T &ft, const std::string &link_func, const double &mu0 = 0.)
    {
        std::map<std::string, LinkFunc::Func> link_list = LinkFunc::link_list;

        T eta = mu0 + ft;
        T mu;
        switch (link_list[tolower(link_func)])
        {
        case LinkFunc::Func::exponential:
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
    static T mu2ft(
        const T &mu,
        const std::string &link_func,
        const double &mu0 = 0.)
    {
        T eta;

        std::map<std::string, LinkFunc::Func> link_list = LinkFunc::link_list;
        switch (link_list[tolower(link_func)])
        {
        case LinkFunc::Func::exponential:
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

        T ft = eta - mu0;

        return ft;
    }

    static double mu2ft(
        const double &mu,
        const std::string &link_func,
        const double &mu0 = 0.)
    {
        double eta = 0.;

        std::map<std::string, LinkFunc::Func> link_list = LinkFunc::link_list;
        switch (link_list[tolower(link_func)])
        {
        case LinkFunc::Func::exponential:
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

        double ft = eta - mu0;

        return ft;
    }


    static double dlambda_deta(double &lambda, const double &eta, const std::string &link_func)
    {
        double deriv = 0.;
        lambda = 0.;
        std::map<std::string, LinkFunc::Func> link_list = LinkFunc::link_list;
        switch (link_list[link_func])
        {
        case LinkFunc::Func::exponential:
        {
            lambda = std::exp(lambda);
            deriv = lambda;
            break;
        }
        case LinkFunc::Func::logistic:
        {
            double tmp = std::exp(eta);
            lambda = tmp / (1. + tmp);
            deriv = tmp / std::pow(1. + tmp, 2.);
        }
        default:
        {
            // Identity link
            lambda = eta;
            deriv = 1.;
            break;
        }
        }

        #ifdef DGTF_DO_BOUND_CHECK
        bound_check(deriv, "LinkFunc::dlambda_deta: deriv");
        #endif
        return deriv;
    }

private:
    static std::map<std::string, Func> map_link_func()
    {
        std::map<std::string, Func> LINK_MAP;

        LINK_MAP["identity"] = Func::identity;
        LINK_MAP["exponential"] = Func::exponential;
        LINK_MAP["logistic"] = Func::logistic;
        return LINK_MAP;
    }
};

inline const std::map<std::string, LinkFunc::Func> LinkFunc::link_list = LinkFunc::map_link_func();

#endif