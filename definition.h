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
        FFBS,
        MCS,
        ParticleLearning,
        MCMC,
        VariationBayes,
        HybridVariation,
        MeanFieldVariation
    };

    enum Dist {
        lognorm,
        nbinomm,
        nbinomp,
        poisson,
        gaussian,
        invgamma,
        gamma,
        halfcauchy,
        uniform,
        constant
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

    enum Param
    {
        W,
        mu0,   // par1 of dobs
        delta, // par2 of dobs
        kappa, // par1 of dlag - nbinom
        r,     // par2 of dlag - nbinom
        mu,    // par1 of dLag - lognorm
        sd,    // par2 of dLag - lognorm
        discount_factor,
        num_backward,
        learning_rate,
        step_size,
        mh_sd
    };

    enum Loss
    {
        L1,
        L2
    };


    static const std::map<std::string, Algo> algo_list;
    static const std::map<std::string, Dist> obs_list;
    static const std::map<std::string, Dist> lag_list;
    static const std::map<std::string, Transfer> trans_list;
    static const std::map<std::string, Func> link_list;
    static const std::map<std::string, Func> gain_list;
    static const std::map<std::string, Dist> err_list;
    static const std::map<std::string, Dist> W_prior_list;
    static const std::map<std::string, Param> static_param_list;
    static const std::map<std::string, Loss> loss_list;
    static const std::map<std::string, Param> tuning_param_list;

    static unsigned int get_gain_code(const std::string &gain_func);
    static unsigned int get_obs_code(const std::string &obs_dist);
    static unsigned int get_link_code(const std::string &link_func);
    static unsigned int get_trans_code(const std::string &lag_dist);
    static unsigned int get_lag_code(const std::string &lag_dist);

    static std::string get_gain_name(const unsigned int &code);
    static std::string get_obs_name(const unsigned int &code);
    static std::string get_link_name(const unsigned int &code);
    static std::string get_trans_name(const unsigned int &code);
    static std::string get_lag_name(const unsigned int &code);

private:
    static std::map<std::string, Algo> map_algorithm()
    {
        std::map<std::string, Algo> ALGO_MAP;

        ALGO_MAP["lba"] = Algo::LinearBayes;
        ALGO_MAP["lbe"] = Algo::LinearBayes;
        ALGO_MAP["linearbayes"] = Algo::LinearBayes;
        ALGO_MAP["linear_bayes"] = Algo::LinearBayes;

        ALGO_MAP["smc"] = Algo::MCS;
        ALGO_MAP["mcs"] = Algo::MCS;

        ALGO_MAP["ffbs"] = Algo::FFBS;

        ALGO_MAP["pl"] = Algo::ParticleLearning;
        ALGO_MAP["particlelearning"] = Algo::ParticleLearning;
        ALGO_MAP["particle_learning"] = Algo::ParticleLearning;

        ALGO_MAP["mcmc"] = Algo::MCMC;

        ALGO_MAP["meanfieldvariation"] = Algo::MeanFieldVariation;
        ALGO_MAP["mean_field_variation"] = Algo::MeanFieldVariation;
        ALGO_MAP["mean-field-variation"] = Algo::MeanFieldVariation;
        ALGO_MAP["meanfieldvb"] = Algo::MeanFieldVariation;
        ALGO_MAP["vb"] = Algo::MeanFieldVariation;

        ALGO_MAP["hybridvariation"] = Algo::HybridVariation;
        ALGO_MAP["hybrid_variation"] = Algo::HybridVariation;
        ALGO_MAP["hybrid-variation"] = Algo::HybridVariation;
        ALGO_MAP["hybridvb"] = Algo::HybridVariation;
        ALGO_MAP["hvb"] = Algo::HybridVariation;
        ALGO_MAP["hva"] = Algo::HybridVariation;

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

        LAG_MAP["uniform"] = Dist::uniform;
        LAG_MAP["flat"] = Dist::uniform;
        LAG_MAP["identity"] = Dist::uniform;

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

    static std::map<std::string, Dist> map_err_dist()
    {
        std::map<std::string, Dist> ERR_MAP;

        ERR_MAP["gaussian"] = Dist::gaussian;
        ERR_MAP["constant"] = Dist::constant;
        return ERR_MAP;
    }

    static std::map<std::string, Dist> map_W_prior()
    {
        std::map<std::string, Dist> map;

        map["invgamma"] = Dist::invgamma;
        map["ig"] = Dist::invgamma;
        map["inv-gamma"] = Dist::invgamma;
        map["inv_gamma"] = Dist::invgamma;

        map["gamma"] = Dist::gamma;

        map["halfcauchy"] = Dist::halfcauchy;
        map["half-cauchy"] = Dist::halfcauchy;
        map["half_cauchy"] = Dist::halfcauchy;

        return map;
    }

    static std::map<std::string, Param> map_static_param()
    {
        std::map<std::string, Param> map;

        map["W"] = Param::W;
        map["w"] = Param::W;
        map["mu0"] = Param::mu0;
        map["delta"] = Param::delta;
        map["kappa"] = Param::kappa;
        map["r"] = Param::r;
        map["mu"] = Param::mu;
        map["sd"] = Param::sd;

        return map;
    }

    static std::map<std::string, Loss> map_loss_func()
    {
        std::map<std::string, Loss> LOSS_MAP;

        LOSS_MAP["l1"] = Loss::L1;
        LOSS_MAP["absolute"] = Loss::L1;
        LOSS_MAP["mae"] = Loss::L1;

        LOSS_MAP["l2"] = Loss::L2;
        LOSS_MAP["quadratic"] = Loss::L2;
        LOSS_MAP["rmse"] = Loss::L2;
        LOSS_MAP["mse"] = Loss::L2;

        return LOSS_MAP;
    }

    static std::map<std::string, Param> map_tuning_param()
    {
        std::map<std::string, Param> maps;
        maps["W"] = Param::W;
        maps["w"] = Param::W;

        maps["discount_factor"] = Param::discount_factor;
        maps["discountfactor"] = Param::discount_factor;
        maps["discount factor"] = Param::discount_factor;

        maps["num_backward"] = Param::num_backward;
        maps["numbackward"] = Param::num_backward;

        maps["learning_rate"] = Param::learning_rate;
        maps["learningrate"] = Param::learning_rate;
        maps["learning rate"] = Param::learning_rate;
        maps["learn_rate"] = Param::learning_rate;
        maps["learnrate"] = Param::learning_rate;
        maps["learn rate"] = Param::learning_rate;

        maps["step_size"] = Param::step_size;
        maps["stepsize"] = Param::step_size;
        maps["step size"] = Param::step_size;
        maps["eps_step_size"] = Param::step_size;
        maps["epsstepsize"] = Param::step_size;
        maps["eps step size"] = Param::step_size;
        maps["eps_step"] = Param::step_size;
        maps["epsstep"] = Param::step_size;
        maps["eps step"] = Param::step_size;
        maps["eps"] = Param::step_size;

        return maps;
    }
}; // class AVAIL

inline const std::map<std::string, AVAIL::Algo> AVAIL::algo_list = AVAIL::map_algorithm();
inline const std::map<std::string, AVAIL::Dist> AVAIL::obs_list = AVAIL::map_obs_dist();
inline const std::map<std::string, AVAIL::Dist> AVAIL::lag_list = AVAIL::map_lag_dist();
inline const std::map<std::string, AVAIL::Transfer> AVAIL::trans_list = AVAIL::map_trans_func();
inline const std::map<std::string, AVAIL::Func> AVAIL::link_list = AVAIL::map_link_func();
inline const std::map<std::string, AVAIL::Func> AVAIL::gain_list = AVAIL::map_gain_func();
inline const std::map<std::string, AVAIL::Dist> AVAIL::err_list = AVAIL::map_err_dist();

inline const std::map<std::string, AVAIL::Dist> AVAIL::W_prior_list = AVAIL::map_W_prior();
inline const std::map<std::string, AVAIL::Param> AVAIL::static_param_list = AVAIL::map_static_param();
inline const std::map<std::string, AVAIL::Loss> AVAIL::loss_list = AVAIL::map_loss_func();
inline const std::map<std::string, AVAIL::Param> AVAIL::tuning_param_list = AVAIL::map_tuning_param();

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

inline std::string AVAIL::get_obs_name(const unsigned int &obs_code)
{
    std::string obs_dist;
    switch (obs_code)
    {
    case 0:
        obs_dist = "nbinom";
        break;
    case 1:
        obs_dist = "poisson";
        break;
    case 2:
        obs_dist = "nbinomp";
    case 3:
        obs_dist = "gaussian";
    default:
        throw std::invalid_argument("obs_dist: 'nbinom', 'poisson', 'nbinom_p', 'gaussian' ");
        break;
    }
    return obs_dist;
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

inline std::string AVAIL::get_link_name(const unsigned int &link_code)
{
    std::string link_func;
    switch (link_code)
    {
    case 0:
        link_func = "identity";
        break;
    case 1:
        link_func = "exponential";
    default:
        throw std::invalid_argument("link_func: 'identity', 'exponential'. ");
        break;
    }

    return link_func;
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


inline std::string AVAIL::get_lag_name(const unsigned int &trans_code)
{
    std::string name;
    switch (trans_code)
    {
    case 1:
        name = "lognorm";
        break;
    case 2:
        name = "nbinom";
        break;
    default:
        throw std::invalid_argument("trans_func: 'sliding', 'iterative'. ");
        break;
    }

    return name;
}

inline std::string AVAIL::get_trans_name(const unsigned int &code)
{
    std::string name;
    switch (code)
    {
    case 1:
        name = "sliding";
        break;
    case 2:
        name = "sliding";
        break;
    default:
        throw std::invalid_argument("trans_func: 'sliding', 'iterative'. ");
        break;
    }

    return name;
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

inline std::string AVAIL::get_gain_name(const unsigned int &code)
{
    std::string name;
    switch (code)
    {
    case 0:
        name = "ramp";
        break;
    case 1:
        name = "exponential";
        break;
    case 2:
        name = "identity";
        break;
    case 3:
        name = "softplus";
        break;
    default:
        throw std::invalid_argument("trans_func: 'sliding', 'iterative'. ");
        break;
    }

    return name;
}

class Dim
{
public:
    unsigned int nT, nL, nP;
    bool truncated = true;
    bool regressor_baseline = false;


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
        const double &nb_r = 0.,
        const bool &add_regressor_baseline = false)
    {
        init(ntime, nlag, nb_r, add_regressor_baseline);
    }

    Dim(
        const unsigned int &nlag,
        const unsigned int &np,
        const bool &truncated_,
        const unsigned int &ntime = 0,
        const bool &add_regressor_baseline = false
    )
    {
        nL = nlag;
        nP = np;
        truncated = truncated_;
        nT = ntime;
        regressor_baseline = add_regressor_baseline;
    }


    /**
     * @brief Initialize by its default setting: non-truncated.
     * 
     */
    void init_default()
    {
        // an non-truncated example
        regressor_baseline = false;
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
        const double &nb_r = 0,
        const bool &add_regressor_baseline = false)
    {
        nT = ntime;
        regressor_baseline = add_regressor_baseline;

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

        if (regressor_baseline)
        {
            nP += 1; // dim = (nL + 1) or (r + 2) if adding constant term to the state vector
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


// struct Discount
// {
//     bool use_discount = false;
//     bool use_custom = false;
//     double custom_value = 0.95;
//     double default_value = 0.99;
// };

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

public:
    std::string name = "undefined";
    double par1 = 0.;
    double par2 = 0.;
    std::string &_name;
    double &_par1;
    double &_par2;


    Dist() : _name(name), _par1(par1), _par2(par2)
    {
        name = "undefined";
        par1 = 0.;
        par2 = 0.;
    }
    Dist(const std::string &name_, const double &par1_, const double &par2_) : _name(name), _par1(par1), _par2(par2)
    {
        init(name_, par1_, par2_);
    }


    void init(const std::string &name_, const double &par1_, const double &par2_)
    {
        name = name_;
        par1 = par1_;
        par2 = par2_;
        return;
    }
    // const std::string &name;
    void update_par1(const double &par1_new)
    {
        par1 = par1_new;
    }

    void update_par2(const double &par2_new)
    {
        par2 = par2_new;
    }

    
};




// struct Sys
// {
//     Discount discount_factor;
//     unsigned int nburnin, nthin, nsample, ntotal, nsmc, nbackward;
// };

#endif