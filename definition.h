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

    static std::map<std::string, Dist> map_lag_dist()
    {
        std::map<std::string, Dist> LAG_MAP;
        LAG_MAP["lognorm"] = Dist::lognorm;
        LAG_MAP["nbinom"] = Dist::nbinomp;
        LAG_MAP["nbinomp"] = Dist::nbinomp;
        return LAG_MAP;
    }

    enum Transfer { sliding,
                           iterative };
    static std::map<std::string, Transfer> map_trans_func()
    {
        std::map<std::string, Transfer> TRANS_MAP;
        TRANS_MAP["sliding"] = Transfer::sliding;
        TRANS_MAP["nbinom"] = Transfer::iterative;
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


    /**
     * @brief Initialize by its default setting: non-truncated.
     * 
     */
    void init_default()
    {
        // an non-truncated example
        truncated = false;
        nT = 200;
        nL = 200;
        nP = static_cast<unsigned int>(NB_R) + 1;
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
        std::map<std::string, AVAIL::Dist> _lag_list = AVAIL::map_lag_dist();
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


/**
 * @brief Define a two-parameter distributions.
 *
 * @param name
 * @param par1
 * @param par2
 */
class Dist
{
    // Dist() : name(_name) {}

protected:
    std::string _name;
    double _par1;
    double _par2;

public:
    // const std::string &name;
    void update_par1(const double &par1)
    {
        _par1 = par1;
    }

    void update_par2(const double &par2)
    {
        _par2 = par2;
    }
};



#endif