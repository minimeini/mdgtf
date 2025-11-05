#pragma once
#ifndef OBSDIST_H
#define OBSDIST_H


#include <RcppArmadillo.h>
#include "../utils/utils.h"
#include "../utils/distributions.hpp"

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
private:
    static std::map<std::string, AVAIL::Dist> map_obs_dist()
    {
        std::map<std::string, AVAIL::Dist> OBS_MAP;

        OBS_MAP["nbinom"] = AVAIL::Dist::nbinomm; // negative-binomial characterized by mean and location.
        OBS_MAP["nbinomm"] = AVAIL::Dist::nbinomm;

        OBS_MAP["nbinomp"] = AVAIL::Dist::nbinomp;

        OBS_MAP["poisson"] = AVAIL::Dist::poisson;

        OBS_MAP["gaussian"] = AVAIL::Dist::gaussian;
        OBS_MAP["normal"] = AVAIL::Dist::gaussian;
        return OBS_MAP;
    }

public:
    static const std::map<std::string, AVAIL::Dist> obs_list;

    ObsDist() : Dist() {init_default();}
    ObsDist(
        const std::string &obs_dist,
        const double &par1_,
        const double &par2_) : Dist()
    {
        init(obs_dist, par1_, par2_);
        return;
    }

    void init(
        const std::string &obs_dist = "nbinom",
        const double &par1_in = 0.,
        const double &par2_in = 30.)
    {
        std::map<std::string, AVAIL::Dist> obs_list = ObsDist::obs_list;
        name = obs_dist;
        par1 = par1_in;
        par2 = par2_in;

        if (obs_list[obs_dist] == AVAIL::Dist::poisson)
        {
            par2 = 1.;
        }
        return;
    }

    void init_default()
    {
        std::map<std::string, AVAIL::Dist> obs_list = ObsDist::obs_list;
        name = "nbinom";
        par1 = 1.;  // mu0
        par2 = 30.; // delta_nb

        if (obs_list[name] == AVAIL::Dist::poisson)
        {
            par2 = 1.;
        }
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
    static double sample(
        const double &lambda, // mean
        const double &par2,
        const std::string &obs_dist)
    {
        std::map<std::string, AVAIL::Dist> obs_list = ObsDist::obs_list;
        double y = 0.;
        switch (obs_list[obs_dist])
        {
        case AVAIL::Dist::gaussian:
        {
            y = R::rnorm(lambda, std::sqrt(par2));
            break;
        }
        case AVAIL::Dist::nbinomm:
        {
            // double prob_succ = par2 / (lambda + par2);
            // y = R::rnbinom(par2, prob_succ);
            y = nbinomm::sample(lambda, par2);
            break;
        }
        case AVAIL::Dist::nbinomp:
        {
            y = nbinom::sample(lambda, par2);
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
        return y;
    }


    static double loglike(
        const double &y, 
        const std::string &obs_dist,
        const double &lambda, 
        const double &par2 = 1.,
        const bool &return_log = true)
    {
        std::map<std::string, AVAIL::Dist> obs_list = ObsDist::obs_list;
        double density = 0.;
        // double ys = std::abs(y);
        // double lambda_s = std::max(lambda, EPS);
        double ys = y;
        double lambda_s = lambda;

        if (lambda < EPS)
        {
            // std::cout << "Warning<loglike>: negative lambda = " << lambda << "given y = " << y << std::endl;
            // density = -999999;
            // if (return_log)
            // {
            //     density = 0.;
            // }

            // return density;
            lambda_s = EPS;
        }

        switch (obs_list[obs_dist])
        {
        case AVAIL::Dist::nbinomm:
        {
            // density = nbinomm::dnbinomm(ys, lambda_s, par2, return_log);
            density = R::Rf_dnbinom_mu(ys, par2, lambda_s, return_log);
            break;
        }
        case AVAIL::Dist::nbinomp:
        {
            density = R::dnbinom(ys, par2, 1. - lambda, return_log);
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

        #ifdef DGTF_DO_BOUND_CHECK
        bound_check(density, "ObsDist::loglike");
        #endif
        return density;
    }

    static double dloglike_dlambda(
        const double &y,
        const double &lambda,
        const ObsDist &dobs
    )
    {
        double deriv = 0.;
        std::map<std::string, AVAIL::Dist> obs_list = ObsDist::obs_list;
        switch (obs_list[dobs.name])
        {
        case AVAIL::Dist::gaussian:
        {
            deriv = (y - lambda) / dobs.par2;
            break;
        }
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
        case AVAIL::Dist::nbinomp:
        {
            deriv = y / lambda - (y + dobs.par2) / (lambda + 1.);
            break;
        }
        default:
        {
            break;
        }
        } // switch by obs type

        #ifdef DGTF_DO_BOUND_CHECK
        bound_check(deriv, "dloglike_dlambda: deriv");
        #endif
        return deriv;
    }
    
};

inline const std::map<std::string, AVAIL::Dist> ObsDist::obs_list = ObsDist::map_obs_dist();

#endif