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
        TFS, // Two-filter Smoothing
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
        beta,
        halfcauchy,
        uniform,
        constant
    };

    enum Func {
        identity,
        exponential,
        softplus,
        ramp,
        logistic
    };

    

    enum Param
    {
        W,
        seas,   // par1 of dobs
        rho, // par2 of dobs
        kappa, // par1 of dlag - nbinom
        r,     // par2 of dlag - nbinom
        mu,    // par1 of dLag - lognorm
        sd,    // par2 of dLag - lognorm
        discount_factor,
        num_backward,
        learning_rate,
        step_size,
        k,
        mh_sd,
        lag_par1,
        lag_par2
    };

    enum Loss
    {
        L1,
        L2
    };


    static const std::map<std::string, Algo> algo_list;
    static const std::map<std::string, Dist> dist_list;
    static const std::map<std::string, Param> static_param_list;
    static const std::map<std::string, Loss> loss_list;
    static const std::map<std::string, Param> tuning_param_list;

private:
    static std::map<std::string, Algo> map_algorithm()
    {
        std::map<std::string, Algo> ALGO_MAP;

        ALGO_MAP["lba"] = Algo::LinearBayes;
        ALGO_MAP["lbe"] = Algo::LinearBayes;
        ALGO_MAP["linearbayes"] = Algo::LinearBayes;
        ALGO_MAP["linear_bayes"] = Algo::LinearBayes;

        ALGO_MAP["smc"] = Algo::SMC;
        
        ALGO_MAP["mcs"] = Algo::MCS;

        ALGO_MAP["ffbs"] = Algo::FFBS;

        ALGO_MAP["tfs"] = Algo::TFS;
        ALGO_MAP["two_filter"] = Algo::TFS;
        ALGO_MAP["twofilter"] = Algo::TFS;

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

    

    

    static std::map<std::string, Dist> map_dist()
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

        map["uniform"] = Dist::uniform;
        map["flat"] = Dist::uniform;
        map["identity"] = Dist::uniform;

        map["nbinom"] = Dist::nbinomm; // negative-binomial characterized by mean and location.
        map["nbinomm"] = Dist::nbinomm;

        map["lognorm"] = Dist::lognorm;
        map["koyama"] = Dist::lognorm;

        map["nbinomp"] = Dist::nbinomp;
        map["solow"] = Dist::nbinomp;

        map["poisson"] = Dist::poisson;

        map["gaussian"] = Dist::gaussian;
        map["normal"] = Dist::gaussian;

        map["beta"] = Dist::beta;

        map["constant"] = Dist::constant;

        return map;
    }

    

    
    static std::map<std::string, Param> map_static_param()
    {
        std::map<std::string, Param> map;

        map["W"] = Param::W;
        map["w"] = Param::W;
        map["seas"] = Param::seas;
        map["rho"] = Param::rho;
        map["delta"] = Param::rho;
        map["kappa"] = Param::kappa;
        map["r"] = Param::r;
        map["mu"] = Param::mu;
        map["sd"] = Param::sd;
        map["lag_par1"] = Param::lag_par1;
        map["par1"] = Param::lag_par1;
        map["lag_par2"] = Param::lag_par2;
        map["par2"] = Param::lag_par2;

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

        maps["k"] = Param::k;

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
inline const std::map<std::string, AVAIL::Dist> AVAIL::dist_list = AVAIL::map_dist();
inline const std::map<std::string, AVAIL::Loss> AVAIL::loss_list = AVAIL::map_loss_func();
inline const std::map<std::string, AVAIL::Param> AVAIL::tuning_param_list = AVAIL::map_tuning_param();
inline const std::map<std::string, AVAIL::Param> AVAIL::static_param_list = AVAIL::map_static_param();

/**
 * @brief Define a two-parameter distributions.
 *
 * @param name [std::string | public | "undefined"] name of the distribution
 * @param par1 [double | public | 0.] first parameter of the distribution
 * @param par2 [double | public | 0.] second parameter of the distribution
 * @param infer [bool | public | false] if the corresponding parameter is unknown and to be inferred
 * @param val [double | public | 0.] initial value of the corresponding parameter
 *
 * @param init [function | public] set the values of name, par1, and par2.
 * @param init_param [function | public] set the values of infer and val.
 */
class Dist
{
public:
    std::string name = "undefined"; // name of the distribution
    double par1 = 0.; // first parameter of the distribution
    double par2 = 0.; // second parameter of the distribution

    Dist()
    {
        name = "undefined";
        par1 = 0.;
        par2 = 0.;
    }


    Dist(
        const std::string &name_, // name of the distribution
        const double &par1_, // first parameter of the distribution
        const double &par2_, // second parameter of the distribution
        const bool &infer_ = false, // if the corresponding parameter is unknown
        const double &init_val_ = 0.) // initial value of the corresponding parameter
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

    void init(const Rcpp::List &opts)
    {
        Rcpp::List param_opts = opts;
        if (param_opts.containsElementNamed("prior_name"))
        {
            std::string prior_name = Rcpp::as<std::string>(param_opts["prior_name"]);
            name = prior_name;
        }

        if (param_opts.containsElementNamed("prior_param"))
        {
            Rcpp::NumericVector param = Rcpp::as<Rcpp::NumericVector>(param_opts["prior_param"]);
            par1 = param[0];
            par2 = param[1];
        }


        return;
    }
};


class Prior : public Dist
{
public:
    bool infer = false;

    Prior() : Dist()
    {
        infer = false;
        return;
    }

    Prior(
        const std::string &name_,   // name of the distribution
        const double &par1_,        // first parameter of the distribution
        const double &par2_,        // second parameter of the distribution
        const bool &infer_ = false) : Dist(name_, par1_, par2_)
    {
        infer = infer_;
        return;
    }

    void init(const std::string &name_in, const double &par1_in, const double &par2_in)
    {
        name = name_in;
        par1 = par1_in;
        par2 = par2_in;
        infer = false;
        return;
    }

    void init(const Rcpp::List &opts)
    {
        Rcpp::List param_opts = opts;

        name = "invgamma";
        par1 = 0.01;
        par2 = 0.01;
        Dist::init(param_opts);

        infer = false;
        if (param_opts.containsElementNamed("infer"))
        {
            infer = Rcpp::as<bool>(param_opts["infer"]);
        }

        return;
    }

    static double dprior(const double &val, const Dist &prior, const bool &return_log = true, const double &jacobian = false)
    {
        std::map<std::string, AVAIL::Dist> dist_list = AVAIL::dist_list;
        double out = 0.;
        switch (dist_list[prior.name]) // switch by prior distribution
        {
        case AVAIL::Dist::gamma: // non-negative
        {
            out = R::dgamma(val, prior.par1, 1. / prior.par2, return_log);
            break;
        }
        case AVAIL::Dist::invgamma: // non-negative
        {
            double tau2 = std::abs(1. / val);      // tau2 ~ Gamma
            double logtau2 = std::log(tau2 + EPS); // log jacobian

            out = R::dgamma(tau2, prior.par1, 1. / prior.par2, true);
            if (jacobian) // plus jacobian
            {
                out += logtau2;
            }

            if (!return_log)
            {
                out = std::exp(out);
            }

            break;
        }
        case AVAIL::Dist::gaussian: // whole real line
        {
            out = R::dnorm4(val, prior.par1, prior.par2, return_log);
            break;
        }
        case AVAIL::Dist::nbinomm: // non-negative
        {
            double prob_succ = prior.par2 / (prior.par1 + prior.par2);
            out = R::dnbinom(val, prior.par2, prob_succ, return_log);
            break;
        }
        case AVAIL::Dist::nbinomp: // non-negative
        {
            out = R::dnbinom(val, prior.par2, 1. - prior.par1, return_log);
            break;
        }
        case AVAIL::Dist::lognorm: // non-negative
        {
            out = R::dlnorm(val, prior.par1, std::sqrt(prior.par2), return_log);
            break;
        }
        case AVAIL::Dist::poisson: // non-negative
        {
            out = R::dpois(val, prior.par1, return_log);
            break;
        }
        case AVAIL::Dist::beta:
        {
            out = R::dbeta(val, prior.par1, prior.par2, return_log);
            break;
        }
        default: // uniform or other types
        {
            out = return_log ? 0 : 1;
            break;
        }
        }

        return out;
    }

    /**
     * @brief The derivative of the logarithm of the prior (before mapped to the real line. i.e. `logpi(gamma)`) w.r.t to the parameter, can be either mapped to the real line (`dlogpi(gamma) / dtilde(gamma)`) or not (`dlogpi(gamma) / dgamma`).
     *
     * @param val i.e. `gamma` (before mapped to the real line)
     * @param prior
     * @param jacobian Return `dlogpi(gamma) / dtilde(gamma)` if set to true; otherwise return `dlogpi(gamma) / dgamma`.
     * @return double
     */
    static double dlogprior_dpar(const double &val, const Dist &prior, const double &jacobian = true)
    {
        std::map<std::string, AVAIL::Dist> dist_list = AVAIL::dist_list;
        double out = 0.;
        switch (dist_list[prior.name]) // switch by prior distribution
        {
        case AVAIL::Dist::invgamma: // non-negative
        {
            if (jacobian) // plus jacobian
            {
                out = -prior.par1 + prior.par2 / val;
            }
            else
            {
                out = -(prior.par1 + 1.) / val;
                out += prior.par2 / std::pow(val, 2.);
            }
            break;
        }
        case AVAIL::Dist::gaussian: // whole real line
        {
            out = -val / prior.par2;
            break;
        }
        case AVAIL::Dist::gamma: // non-negative
        {
            if (jacobian)
            {
                out = prior.par1 - prior.par2 * val;
            }
            else
            {
                out = (prior.par1 - 1.) / val;
                out -= prior.par2;
            }
            break;
        }
        case AVAIL::Dist::beta:
        {
            if (jacobian)
            {
                out = prior.par1 - (prior.par1 + prior.par2) * val;
            }
            else
            {
                out = (prior.par1 - 1) * (1. - val);
                out -= (prior.par2 - 1) * val;
            }

            break;
        }
        default: // uniform or other types
        {
            throw std::invalid_argument("Dist::dlogprior_dpar: undefined for " + prior.name + " prior.");
            break;
        }
        }

        return out;
    }

    static double val2real(const double &val, const std::string &dist_name, const bool &inverse_transform = false)
    {
        std::map<std::string, AVAIL::Dist> dist_list = AVAIL::dist_list;
        double out = val;

        switch (dist_list[dist_name]) // switch by prior distribution
        {
        case AVAIL::Dist::gamma: // non-negative
        {
            if (inverse_transform)
            {
                out = std::exp(val);
            }
            else
            {
                out = std::log(std::abs(val) + EPS);
            }
            break;
        }
        case AVAIL::Dist::invgamma: // non-negative
        {
            if (inverse_transform)
            {
                out = std::exp(val);
            }
            else
            {
                out = std::log(std::abs(val) + EPS);
            }
            break;
        }
        case AVAIL::Dist::lognorm: // non-negative
        {
            if (inverse_transform)
            { // exponential
                out = std::exp(val);
            }
            else
            { // logarithm
                out = std::log(std::abs(val) + EPS);
            }
            break;
        }
        case AVAIL::Dist::beta:
        {
            if (inverse_transform)
            { // logistic
                double tmp = std::exp(-val);
                out = 1. / (1. + tmp);
            }
            else
            { // logit
                double odd = std::abs(val / (1. - val));
                out = std::log(odd + EPS);
            }

            break;
        }
        default: // gaussian or other types
        {
            break;
        }
        }

        return out;
    }



};


#endif