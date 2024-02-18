#pragma once
#ifndef DEFINITION_H
#define DEFINITION_H

#include <vector>
#include <string>
#include <map>
#include "constants.h"

class AVAIL
{
public:
    enum Algo {
        LinearBayes,
        SMC,
        MCMC,
        VariationBayes
    };

    enum Dist {
        lognorm,
        nbinomm,
        nbinomp,
        poisson,
        gaussian
    };

    enum Func {
        identity,
        exponential,
        softplus,
        ramp
    };

    enum Transfer
    {
        sliding,
        iterative
    };

    static const std::map<std::string, Algo> algo_list;
    static const std::map<std::string, Dist> obs_list;
    static const std::map<std::string, Dist> lag_list;
    static const std::map<std::string, Transfer> trans_list;
    static const std::map<std::string, Func> link_list;
    static const std::map<std::string, Func> gain_list;

    static unsigned int get_gain_code(const std::string &gain_func);
    static unsigned int get_obs_code(const std::string &obs_dist);
    static unsigned int get_link_code(const std::string &link_func);
    static unsigned int get_trans_code(const std::string &lag_dist);
    static unsigned int get_lag_code(const std::string &lag_dist);

private:
    static std::map<std::string, Algo> map_algorithm()
    {
        std::map<std::string, Algo> ALGO_MAP;
        ALGO_MAP["lba"] = Algo::LinearBayes;
        ALGO_MAP["lbe"] = Algo::LinearBayes;
        ALGO_MAP["linearbayes"] = Algo::LinearBayes;
        ALGO_MAP["linear_bayes"] = Algo::LinearBayes;

        ALGO_MAP["smc"] = Algo::SMC;

        ALGO_MAP["mcmc"] = Algo::MCMC;

        ALGO_MAP["variationbayes"] = Algo::VariationBayes;
        ALGO_MAP["variation_bayes"] = Algo::VariationBayes;
        ALGO_MAP["variationalbayes"] = Algo::VariationBayes;
        ALGO_MAP["variational_bayes"] = Algo::VariationBayes;
        ALGO_MAP["vb"] = Algo::VariationBayes;
        ALGO_MAP["vba"] = Algo::VariationBayes;

        return ALGO_MAP;
    }

    static std::map<std::string, Dist> map_lag_dist()
    {
        std::map<std::string, Dist> LAG_MAP;
        LAG_MAP["lognorm"] = Dist::lognorm;
        LAG_MAP["koyama"] = Dist::lognorm;

        LAG_MAP["nbinom"] = Dist::nbinomp;

        LAG_MAP["nbinomp"] = Dist::nbinomp;
        LAG_MAP["solow"] = Dist::nbinomp;
        return LAG_MAP;
    }

    static std::map<std::string, Transfer> map_trans_func()
    {
        std::map<std::string, Transfer> TRANS_MAP;
        TRANS_MAP["sliding"] = Transfer::sliding;
        TRANS_MAP["slide"] = Transfer::sliding;
        TRANS_MAP["koyama"] = Transfer::sliding;

        TRANS_MAP["iterative"] = Transfer::iterative;
        TRANS_MAP["iter"] = Transfer::iterative;
        TRANS_MAP["solow"] = Transfer::iterative;
        return TRANS_MAP;
    }

    static std::map<std::string, Dist> map_obs_dist()
    {
        std::map<std::string, Dist> OBS_MAP;
        OBS_MAP["nbinom"] = Dist::nbinomm; // negative-binomial characterized by mean and location.
        OBS_MAP["nbinomm"] = Dist::nbinomm;
        OBS_MAP["nbinomp"] = Dist::nbinomp;
        OBS_MAP["poisson"] = Dist::poisson;
        OBS_MAP["gaussian"] = Dist::gaussian;
        return OBS_MAP;
    }

    static std::map<std::string, Func> map_link_func()
    {
        std::map<std::string, Func> LINK_MAP;
        LINK_MAP["identity"] = Func::identity;
        LINK_MAP["exponential"] = Func::exponential;
        return LINK_MAP;
    }

    static std::map<std::string, Func> map_gain_func()
    {
        std::map<std::string, Func> GAIN_MAP;
        GAIN_MAP["ramp"] = Func::ramp;
        GAIN_MAP["exponential"] = Func::exponential;
        GAIN_MAP["identity"] = Func::identity;
        GAIN_MAP["softplus"] = Func::softplus;
        return GAIN_MAP;
    }
};

inline const std::map<std::string, AVAIL::Algo> AVAIL::algo_list = AVAIL::map_algorithm();
inline const std::map<std::string, AVAIL::Dist> AVAIL::obs_list = AVAIL::map_obs_dist();
inline const std::map<std::string, AVAIL::Dist> AVAIL::lag_list = AVAIL::map_lag_dist();
inline const std::map<std::string, AVAIL::Transfer> AVAIL::trans_list = AVAIL::map_trans_func();
inline const std::map<std::string, AVAIL::Func> AVAIL::link_list = AVAIL::map_link_func();
inline const std::map<std::string, AVAIL::Func> AVAIL::gain_list = AVAIL::map_gain_func();

inline unsigned int AVAIL::get_gain_code(const std::string &gain_func)
{
    std::map<std::string, AVAIL::Func> gain_list = AVAIL::gain_list;
    unsigned int gain_code;
    switch (gain_list[gain_func])
    {
    case AVAIL::Func::ramp:
        gain_code = 0;
        break;
    case AVAIL::Func::exponential:
        gain_code = 1;
        break;
    case AVAIL::Func::identity:
        gain_code = 2;
        break;
    case AVAIL::Func::softplus:
        gain_code = 3;
        break;
    default:
        throw std::invalid_argument("gain_func: 'ramp', 'exponential'. 'identity', 'softplus'. ");
    }

    return gain_code;
}

inline unsigned int AVAIL::get_obs_code(const std::string &obs_dist)
{
    std::map<std::string, AVAIL::Dist> obs_list = AVAIL::obs_list;
    unsigned int obs_code;
    switch (obs_list[obs_dist])
    {
    case AVAIL::Dist::nbinomm:
        obs_code = 0;
        break;
    case AVAIL::Dist::poisson:
        obs_code = 1;
        break;
    case AVAIL::Dist::nbinomp:
        obs_code = 2;
    case AVAIL::Dist::gaussian:
        obs_code = 3;
    default:
        throw std::invalid_argument("obs_dist: 'nbinom', 'poisson', 'nbinom_p', 'gaussian' ");
        break;
    }
    return obs_code;
}

inline unsigned int AVAIL::get_link_code(const std::string &link_func)
{
    std::map<std::string, AVAIL::Func> link_list = AVAIL::link_list;
    unsigned int link_code;
    switch (link_list[link_func])
    {
    case AVAIL::Func::identity:
        link_code = 0;
        break;
    case AVAIL::Func::exponential:
        link_code = 1;
    default:
        throw std::invalid_argument("link_func: 'identity', 'exponential'. ");
        break;
    }

    return link_code;
}

inline unsigned int AVAIL::get_trans_code(const std::string &lag_dist)
{
    std::map<std::string, AVAIL::Dist> lag_list = AVAIL::lag_list;
    unsigned int trans_code;
    switch (lag_list[lag_dist])
    {
    case AVAIL::Dist::lognorm:
        trans_code = 1;
        break;
    case AVAIL::Dist::nbinomp:
        trans_code = 2;
        break;
    default:
        throw std::invalid_argument("trans_func: 'sliding', 'iterative'. ");
        break;
    }

    return trans_code;
}

inline unsigned int AVAIL::get_lag_code(const std::string &lag_dist)
{
    std::map<std::string, AVAIL::Dist> lag_list = AVAIL::lag_list;
    unsigned int lag_code;
    switch (lag_list[lag_dist])
    {
    case AVAIL::Dist::lognorm:
        lag_code = 1;
        break;
    case AVAIL::Dist::nbinomp:
        lag_code = 2;
        break;
    default:
        throw std::invalid_argument("trans_func: 'sliding', 'iterative'. ");
        break;
    }

    return lag_code;
}

class Dim
{
public:
    unsigned int nT, nL, nP;
    bool truncated;


    /**
     * @brief Create an empty dim object.
     * 
     */
    Dim(){ init_default(); }

    /**
     * @brief Initialize with or without truncation.
     *
     * @param ntime Number of temporal observations.
     * @param nlag Number of lags. The lag distribution is truncated if 1 <= nlag <= (ntime - 1), and non-truncated otherwise.
     * @param nb_r The order of negative-binomial lag distribution (Solow, 1960).
     */
    Dim(
        const unsigned int &ntime, 
        const unsigned int &nlag = 0, 
        const double &nb_r = 0.)
    {
        init(ntime, nlag, nb_r);
    }

    Dim(
        const unsigned int &nlag,
        const unsigned int &np,
        const bool &truncated_,
        const unsigned int &ntime = 0
    )
    {
        nL = nlag;
        nP = np;
        truncated = truncated_;
        nT = ntime;
    }


    /**
     * @brief Initialize by its default setting: non-truncated.
     * 
     */
    void init_default()
    {
        // an non-truncated example
        truncated = true;
        nT = 200;
        nL = 10;
        nP = nL;
        // nP = static_cast<unsigned int>(NB_R) + 1;
    }


    /**
     * @brief Initialize with or without truncation.
     *
     * @param ntime Number of temporal observations.
     * @param nlag Number of lags. The lag distribution is truncated if 1 <= nlag <= (ntime - 1), and non-truncated otherwise.
     * @param nb_r The order of negative-binomial lag distribution (Solow, 1960).
     */
    void init(
        const unsigned int &ntime, 
        const unsigned int &nlag = 0, 
        const double &nb_r = 0)
    {
        nT = ntime;
        if (nlag > 0 && nlag < ntime)
        {
            // truncated if we have a valid nlag, 0 < nlag < ntime
            truncated = true;
            nL = nlag;
            nP = nL;
        }
        else
        {
            // nlag = 0 leads to no truncation.
            truncated = false;
            nL = ntime;
            nP = static_cast<unsigned int>(nb_r) + 1;
        }

        return;
    }


    /**
     * @brief Check if there is any conflicts in the dimension settings.
     * 
     */
    static void validate(
        const Dim &dim, 
        const std::string &dlag_name, 
        const double &dlag_par2)
    {
        std::map<std::string, AVAIL::Dist> _lag_list = AVAIL::lag_list;
        if (_lag_list[dlag_name] == AVAIL::Dist::lognorm && !dim.truncated)
        {
            throw std::invalid_argument("Error: non-truncated form should only be used with a negative-binomial lag distribution");
        } else if (!dim.truncated)
        {
            // A negative-binomial lag distribution with no truncation.
            unsigned int nP_ = static_cast<unsigned int>(dlag_par2) + 1;
            if (dim.nP != nP_)
            {
                throw std::invalid_argument("Error: dimension of non-truncated negative-binomial lags should be (r + 1)");
            }
        }

        return;
    }
};






#endif