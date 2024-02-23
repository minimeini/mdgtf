#pragma once
#ifndef OBSDIST_H
#define OBSDIST_H


#include <RcppArmadillo.h>
#include "utils.h"
#include "distributions.hpp"

// [[Rcpp::plugins(cpp17)]]
// [[Rcpp::depends(RcppArmadillo)]]

/**
 * @brief observation distribution characterized by either (mu0, delta_nb) or (mu, sd2)
 *       - (mu0, delta_nb):
 *          (1) mu0: constant mean of the observed time series;
 *          (2) delta_nb: degress of freedom for negative-binomial observation distribution.
 *       - (mu, sd2): parameters for gaussian observation distribution.
 *
 */
class ObsDist : public Dist
{
public:
    ObsDist() : Dist(), par1(_par1), par2(_par2) {init_default();}
    ObsDist(
        const std::string &obs_dist,
        const double &par1_,
        const double &par2_) : Dist(), par1(_par1), par2(_par2)
    {
        init(obs_dist, par1_, par2_);
        return;
    }

    const double &par1;
    const double &par2;
    std::map<std::string, AVAIL::Dist> obs_list = AVAIL::obs_list;

    void init(
        const std::string &obs_dist = "nbinom",
        const double &par1 = 0.,
        const double &par2 = 30.)
    {
        obs_list = AVAIL::obs_list;
        _name = obs_dist;
        _par1 = par1;
        _par2 = par2;
        return;
    }

    void init_default()
    {
        obs_list = AVAIL::obs_list;
        _name = "nbinom";
        _par1 = 0.;  // mu0
        _par2 = 30.; // delta_nb
        return;
    }

    /**
     * @brief Draw a single sample from the observation distribution, characterized by two parameters
     *
     * @param lambda first parameter of the observation distribution
     * @param par2 second parameter of the observation distribution
     * @param obs_dist name of the observation distribution
     * @return double
     */
    double sample(const double &lambda)
    {
        double y = 0.;
        switch (obs_list[_name])
        {
        case AVAIL::Dist::nbinomm:
        {
            double prob_succ = par2 / (lambda + par2);
            y = R::rnbinom(par2, prob_succ);
            break;
        }        
        default:
        {
            // Poisson observation distribution
            y = R::rpois(lambda);
            break;
        }
        }


        return y;
    }


    /**
     * @brief Draw a single sample from the observation distribution, characterized by two parameters
     * 
     * @param lambda first parameter of the observation distribution
     * @param par2 second parameter of the observation distribution
     * @param obs_dist name of the observation distribution
     * @return double 
     */
    static double sample(
        const double &lambda, // mean
        const double &par2,
        const std::string &obs_dist)
    {
        std::map<std::string, AVAIL::Dist> obs_list = AVAIL::obs_list;
        double y = 0.;
        switch (obs_list[obs_dist])
        {
        case AVAIL::Dist::nbinomm:
        {
            // double prob_succ = par2 / (lambda + par2);
            // y = R::rnbinom(par2, prob_succ);
            y = nbinomm::sample(lambda, par2);
            break;
        }
        default:
        {
            // Poisson observation distribution
            y = Poisson::sample(lambda);
            // y = R::rpois(lambda);
            break;
        }
        }

        bound_check(y, "static double sample: y", false, true);
        return y;
    }


    static double loglike(
        const double &y, 
        const std::string &obs_dist,
        const double &lambda, 
        const double &par2 = 1.,
        const bool &return_log = true)
    {
        std::map<std::string, AVAIL::Dist> obs_list = AVAIL::obs_list;
        double density;
        // double ys = std::abs(y);
        // double lambda_s = std::max(lambda, EPS);
        double ys = y;
        double lambda_s = lambda;

        switch (obs_list[obs_dist])
        {
        case AVAIL::Dist::nbinomm:
        {
            // density = nbinomm::dnbinomm(ys, lambda_s, par2, return_log);
            density = R::Rf_dnbinom_mu(ys, par2, lambda_s, return_log);
            break;
        }
        case AVAIL::Dist::poisson:
        {
            density = R::dpois(ys, lambda_s, return_log);
            break;
        }
        case AVAIL::Dist::gaussian:
        {
            double sd = std::sqrt(par2);
            density = R::dnorm4(y, lambda, sd, return_log);
        }
        default:
            break;
        }

        bound_check(density, "ObsDist::loglike");
        return density;
    }

    static double dloglike_dlambda(
        const double &y,
        const double &lambda,
        const ObsDist &dobs
    )
    {
        double deriv = 0.;
        std::map<std::string, AVAIL::Dist> obs_list = AVAIL::obs_list;
        switch (obs_list[dobs.name])
        {
        case AVAIL::Dist::poisson:
        {
            deriv = y / lambda;
            deriv -= 1.;
            break;
        }
        case AVAIL::Dist::nbinomm:
        {
            deriv = y / lambda;
            deriv -= (y + dobs.par2) / (lambda + dobs.par2);
            break;
        }
        default:
        {
            break;
        }
        } // switch by obs type

        bound_check(deriv, "dloglike_dlambda: deriv");
        return deriv;
    }

    static double dforecast(
        const double &ynext,
        const std::string &obs_dist,
        const double &obs_par2,
        const double &alpha_next,
        const double &beta_next,
        const bool &return_log = true
    )
    {
        std::map<std::string, AVAIL::Dist> obs_list = AVAIL::obs_list;
        std::string obs_name = tolower(obs_dist);
        double logp_pred = 0.;

        switch (obs_list[obs_name])
        {
        case AVAIL::Dist::poisson:
        {
            // One-step-aheading forecasting for Poisson observations follows a negative-binomial distribution.
            double par1 = alpha_next; // Targeted number of successes.
            double par2 = 1. / (beta_next + 1.); // Probability of failures.
            logp_pred = nbinom::dnbinom(ynext, par2, par1);
            break;
        }
        case AVAIL::Dist::nbinomm:
        {
            double c1 = R::lgammafn(ynext + obs_par2);
            c1 -= R::lgammafn(obs_par2);
            c1 -= R::lgammafn(ynext + 1.);

            double c2 = R::lgammafn(alpha_next + beta_next);
            c2 -= R::lgammafn(alpha_next);
            c2 -= R::lgammafn(beta_next);

            double c3 = R::lgammafn(alpha_next + ynext);
            c3 += R::lgammafn(beta_next + obs_par2);
            c3 -= R::lgammafn(alpha_next + ynext + beta_next + obs_par2);

            logp_pred = c1 + c2 + c3;

            break;
        }
        default:
        {
            break;
        }
        }

        bound_check(logp_pred, "dforecast");

        if (return_log)
        {
            return logp_pred;
        }
        else
        {
            return std::exp(logp_pred);
        }
    }



    
};



#endif