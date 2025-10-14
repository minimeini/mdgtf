#pragma once
#ifndef REGRESSION_H
#define REGRESSION_H

#include <RcppArmadillo.h>
#include "utils.h"


/**
 * @brief Seasonality component to the conditional intensity.
 * 
 */
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


/**
 * @brief Logistic regression of the probability of zero-inflation.
 * 
 */
class ZeroInflation
{
public:
    bool inflated = false;
    arma::vec beta;   // p x 1
    double intercept; // intercept of the logistic regression
    double coef; // AR coef of the logistic regression

    arma::mat X; // p x (ntime + 1), covariates of the logistic regression

    arma::vec prob; // (ntime + 1) x 1, z ~ Bernouli(prob)
    arma::vec z; // (ntime + 1) x 1, indicator, z = 0 for constant 0 and z = 1 for conditional NB/Poisson.

    ZeroInflation()
    {
        init_default();
        return;
    }

    ZeroInflation(const Rcpp::List &settings)
    {
        init(settings);
        return;
    }

    void init_default()
    {
        inflated = false;
        intercept = 0.;
        coef = 0.;

        beta.reset();
        X.reset();
        return;
    }


    void init(const Rcpp::List &settings)
    {
        Rcpp::List opts = settings;

        inflated = false;
        if (opts.containsElementNamed("inflated"))
        {
            inflated = Rcpp::as<bool>(opts["inflated"]);
        }

        intercept = 0.;
        if (opts.containsElementNamed("intercept"))
        {
            intercept = Rcpp::as<double>(opts["intercept"]);
        }

        coef = 0.;
        if (opts.containsElementNamed("coef"))
        {
            coef = Rcpp::as<double>(opts["coef"]);
        }

        beta.reset();
        X.reset();
        if (opts.containsElementNamed("beta"))
        {
            beta = Rcpp::as<arma::vec>(opts["beta"]); //  p x 1
        }

        return;
    }

    void setX(const arma::mat &Xmat) // p x (ntime + 1)
    {
        inflated = true;
        X = Xmat;

        if (beta.is_empty())
        {
            beta = 10. * arma::randn(Xmat.n_rows);
        }
        else if (beta.n_elem != Xmat.n_rows)
        {
            throw std::invalid_argument("Zero: dimension of beta != number of covariates in X.");
        }

        return;
    }

    void setZ(const arma::vec &zvec, const unsigned int &ntime)
    {
        z.set_size(ntime + 1);
        z.ones();
        z.at(0) = 0.;
        z.tail(zvec.n_elem) = zvec;

        prob = z;

        double zsum = arma::accu(arma::abs(z));
        if ((zsum > 1. - EPS) && (zsum < (double)ntime))
        {
            inflated = true;
        }

        return;
    }

    void simulateZ(const unsigned int &ntime)
    {
        z.set_size(ntime + 1);
        z.ones();
        z.at(0) = 0.;
        if (prob.is_empty())
        {
            prob = z;
        }

        if (inflated)
        {
            for (unsigned int t = 1; t < (ntime + 1); t++)
            {
                double val = intercept;
                if (std::abs(z.at(t - 1) - 1.) < EPS)
                {
                    val += coef;
                }

                if (!X.is_empty())
                {
                    val += arma::dot(beta, X.col(t));
                }

                prob.at(t) = logistic(val);
                z.at(t) = (R::runif(0., 1.) < prob.at(t)) ? 1. : 0.;
            }
        }

        return;
    }

    Rcpp::List info()
    {
        Rcpp::List settings;
        settings["inflated"] = inflated;
        settings["intercept"] = intercept;
        settings["coef"] = coef;
        if (beta.is_empty())
        {
            settings["beta"] = NA_REAL;
        }
        else
        {
            settings["beta"] = Rcpp::wrap(beta.t());
        }
        
        return settings;
    }

    static Rcpp::List default_settings()
    {
        Rcpp::List settings;
        settings["inflated"] = true;
        settings["intercept"] = 0.;
        settings["coef"] = 0.;
        settings["beta"] = NA_REAL;
        return settings;
    }


};

#endif