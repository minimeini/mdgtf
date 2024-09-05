#pragma once
#ifndef REGRESSION_H
#define REGRESSION_H

#include <RcppArmadillo.h>
#include "utils.h"

class Season
{
public:
    bool in_state = false;
    unsigned int period = 0;
    arma::vec val; // period x 1
    arma::mat X; // period x (ntime + 1)
    arma::mat P; // period x period
    double lobnd = 1.;
    double hibnd = 10.;

    Season()
    {
        init_default();
        return;
    }

    Season(const Rcpp::List &settings)
    {
        init(settings);
        return;
    }

    void init_default()
    {
        in_state = false;
        period = 0;
    }

    void init(const Rcpp::List &settings)
    {
        Rcpp::List opts = settings;

        in_state = false;
        if (opts.containsElementNamed("in_state"))
        {
            in_state = Rcpp::as<bool>(opts["in_state"]);
        }

        period = 1;
        if (opts.containsElementNamed("period"))
        {
            period = Rcpp::as<unsigned int>(opts["period"]);
        }

        lobnd = 1.;
        if (opts.containsElementNamed("lobnd"))
        {
            lobnd = Rcpp::as<double>(opts["lobnd"]);
        }

        hibnd = 10.;
        if (opts.containsElementNamed("hibnd"))
        {
            hibnd = Rcpp::as<double>(opts["hibnd"]);
        }

        val.set_size(period);
        val.zeros();
        unsigned int nelem = 0;
        if (opts.containsElementNamed("init"))
        {
            arma::vec init = Rcpp::as<arma::vec>(opts["init"]);
            nelem = (init.n_elem <= period) ? init.n_elem : period;
            val.head(nelem) = init.head(nelem);
        }

        if (nelem < period)
        {
            val.tail(period - nelem) = arma::randu(period - nelem, arma::distr_param(lobnd, hibnd));
        }

        P.set_size(period, period);
        P.zeros();
        P.at(period - 1, 0) = 1.;
        if (period > 1)
        {
            for (unsigned int i = 0; i < period - 1; i++)
            {
                P.at(i, i + 1) = 1.;
            }
        }
    }


    static Rcpp::List default_settings()
    {
        Rcpp::List settings;
        settings["in_state"] = false;
        settings["period"] = 1;
        settings["lobnd"] = 1.;
        settings["hibnd"] = 10.;
        return settings;
    }


    static arma::mat setX(const unsigned int &ntime, const unsigned int &period, const arma::mat &P)
    {
        arma::mat X(period, ntime + 1, arma::fill::zeros);
        X.at(0, 0) = 1.;
        for (unsigned int t = 0; t < ntime; t++)
        {
            X.col(t + 1) = P.t() * X.col(t);
        }
        return X;
    }

    Rcpp::List info()
    {
        Rcpp::List settings;
        settings["period"] = period;
        settings["in_state"] = in_state;
        settings["lobnd"] = lobnd;
        settings["hibnd"] = hibnd;
        settings["X"] = Rcpp::wrap(X);
        settings["val"] = Rcpp::wrap(val.t());
        return settings;
    }
};

#endif