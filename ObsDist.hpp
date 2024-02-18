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

    
};



#endif