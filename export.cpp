#include "Model.hpp"
#include "LinearBayes.hpp"
#include "SequentialMonteCarlo.hpp"
#include "MCMC.hpp"
#include "VariationalBayes.hpp"

using namespace Rcpp;
// [[Rcpp::plugins(cpp17)]]
// [[Rcpp::depends(RcppArmadillo,nloptr, BH)]]

void parse_model_settings(
    std::string &obs_dist,
    std::string &link_func,
    std::string &trans_func,
    std::string &gain_func,
    std::string &lag_dist,
    std::string &err_dist,
    const Rcpp::List &Model)
{
    Rcpp::List model = Model;
    if (model.containsElementNamed("obs_dist"))
    {
        obs_dist = tolower(Rcpp::as<std::string>(model["obs_dist"]));
    }
    else
    {
        obs_dist = "nbinom";
    }

    if (model.containsElementNamed("link_func"))
    {
        link_func = tolower(Rcpp::as<std::string>(model["link_func"]));
    }
    else
    {
        link_func = "identity";
    }

    if (model.containsElementNamed("trans_func"))
    {
        trans_func = tolower(Rcpp::as<std::string>(model["trans_func"]));
    }
    else
    {
        trans_func = "sliding";
    }

    if (model.containsElementNamed("gain_func"))
    {
        gain_func = tolower(Rcpp::as<std::string>(model["gain_func"]));
    }
    else
    {
        gain_func = "softplus";
    }

    if (model.containsElementNamed("lag_dist"))
    {
        lag_dist = tolower(Rcpp::as<std::string>(model["lag_dist"]));
    }
    else
    {
        lag_dist = "lognorm";
    }

    if (model.containsElementNamed("err_dist"))
    {
        err_dist = tolower(Rcpp::as<std::string>(model["err_dist"]));
    }
    else
    {
        err_dist = "gaussian";
    }

    return;
}


void parse_dimension_settings(
    unsigned int &nlag,
    unsigned int &ntime,
    bool &truncated,
    const Rcpp::List &Dimension)
{
    Rcpp::List dim = Dimension;

    if (dim.containsElementNamed("nlag"))
    {
        nlag = Rcpp::as<unsigned int>(dim["nlag"]);
    }
    else
    {
        nlag = 10;
    }

    if (dim.containsElementNamed("ntime"))
    {
        ntime = Rcpp::as<unsigned int>(dim["ntime"]);
    }
    else
    {
        ntime = 200;
    }

    if (nlag >= ntime)
    {
        nlag = ntime;
        truncated = false;
    }
    else if (dim.containsElementNamed("truncated"))
    {
        truncated = Rcpp::as<bool>(dim["truncated"]);
    }
    else
    {
        truncated = true;
    }
}

void parse_param_settings(
    Rcpp::NumericVector &obs,
    Rcpp::NumericVector &lag,
    Rcpp::NumericVector &err,
    const Rcpp::List &Param)
{
    Rcpp::List param = Param;

    if (param.containsElementNamed("obs"))
    {
        obs = Rcpp::as<Rcpp::NumericVector>(param["obs"]);
    }
    else
    {
        obs = Rcpp::NumericVector::create(0., 30.);
    }

    if (param.containsElementNamed("lag"))
    {
        lag = Rcpp::as<Rcpp::NumericVector>(param["lag"]);
    }
    else
    {
        lag = Rcpp::NumericVector::create(1.4, 0.3);
    }

    if (param.containsElementNamed("err"))
    {
        err = Rcpp::as<Rcpp::NumericVector>(param["err"]);
    }
    else
    {
        err = Rcpp::NumericVector::create(0.01, 0.);
    }

    return;
}

//' @export
// [[Rcpp::export]]
Rcpp::List dgtf_default_algo_settings(const std::string &method)
{
    std::map<std::string, AVAIL::Algo> algo_list = AVAIL::algo_list;
    std::string method_name = tolower(method);

    Rcpp::List opts = SMC::SequentialMonteCarlo::default_settings();
    switch (algo_list[method_name])
    {
    case AVAIL::Algo::LinearBayes:
    {
        opts =  LBA::LinearBayes::default_settings();
        break;
    }
    case AVAIL::Algo::MCS:
    {
        opts = SMC::MCS::default_settings();
        break;
    }
    case AVAIL::Algo::FFBS:
    {
        opts = SMC::FFBS::default_settings();
        break;
    }
    case AVAIL::Algo::ParticleLearning:
    {
        opts = SMC::PL::default_settings();
        break;
    }
    case AVAIL::Algo::MCMC:
    {
        opts = MCMC::Disturbance::default_settings();
        break;
    }
    case AVAIL::Algo::HybridVariation:
    {
        opts = VB::Hybrid::default_settings();
    }
    default:
        break;
    }

    return opts;
}

Rcpp::List get_default_model_settings()
{
    Rcpp::List model_settings;
    model_settings["obs_dist"] = "nbinom";
    model_settings["link_func"] = "identity";
    model_settings["trans_func"] = "sliding";
    model_settings["gain_func"] = "softplus";
    model_settings["lag_dist"] = "lognorm";
    model_settings["err_dist"] = "gaussian";

    return model_settings;
}

Rcpp::List get_default_dimensions()
{
    Rcpp::List dim_settings;
    dim_settings["nlag"] = 10;
    dim_settings["ntime"] = 200;
    dim_settings["truncated"] = true;

    return dim_settings;
}

Rcpp::List get_default_params()
{
    Rcpp::List param_settings;
    param_settings["obs"] = Rcpp::NumericVector::create(0., 30.);
    param_settings["lag"] = Rcpp::NumericVector::create(1.4, 0.3);
    param_settings["err"] = Rcpp::NumericVector::create(0.01, 0.);

    return param_settings;
}

//' @export
// [[Rcpp::export]]
Rcpp::List dgtf_default_model()
{
    Rcpp::List settings;
    Rcpp::List model = get_default_model_settings();
    Rcpp::List dim = get_default_dimensions();
    Rcpp::List param = get_default_params();

    settings["model"] = model;
    settings["dim"] = dim;
    settings["param"] = param;

    return settings;
}

//' @export
// [[Rcpp::export]]
arma::uvec dgtf_model_code(const Rcpp::List model)
{
    Rcpp::List opts = model;
    std::string obs_dist, link_func, lag_dist, gain_func, err_dist;
    if (opts.containsElementNamed("obs_dist"))
    {
        obs_dist = Rcpp::as<std::string>(opts["obs_dist"]);
    }
    if (opts.containsElementNamed("link_func"))
    {
        link_func = Rcpp::as<std::string>(opts["link_func"]);
    }
    if (opts.containsElementNamed("lag_dist"))
    {
        lag_dist = Rcpp::as<std::string>(opts["lag_dist"]);
    }
    if (opts.containsElementNamed("gain_func"))
    {
        gain_func = Rcpp::as<std::string>(opts["gain_func"]);
    }
    if (opts.containsElementNamed("err_dist"))
    {
        err_dist = Rcpp::as<std::string>(opts["err_dist"]);
    }

    const unsigned int obs_code = AVAIL::get_obs_code(obs_dist);
    const unsigned int link_code = AVAIL::get_link_code(link_func);
    const unsigned int trans_code = AVAIL::get_trans_code(lag_dist);
    const unsigned int gain_code = AVAIL::get_gain_code(gain_func);
    const unsigned int err_code = 0;

    arma::uvec model_code = {obs_code, link_code, trans_code, gain_code, err_code};
    return model_code;
}

//' @export
// [[Rcpp::export]]
Rcpp::List dgtf_model(
    const std::string &obs_dist = "nbinom",
    const std::string &link_func = "identity",
    const std::string &trans_func = "sliding",
    const std::string &gain_func = "softplus",
    const std::string &lag_dist = "lognorm",
    const std::string &err_dist = "gaussian",
    const unsigned int &nlag = 10,
    const unsigned int &ntime = 200,
    const bool &truncated = true,
    const Rcpp::NumericVector &obs_param = Rcpp::NumericVector::create(0., 30.),  // (mu0, delta_nb)
    const Rcpp::NumericVector &lag_param = Rcpp::NumericVector::create(1.4, 0.3), // (kappa, r) or (mu, sd2)
    const Rcpp::NumericVector &err_param = Rcpp::NumericVector::create(0.01, 0.)  // (W, w[0]))
)
{
    // Consolidate the model definitions into a Rcpp list.s
    Rcpp::List model;
    model["obs_dist"] = tolower(obs_dist);
    model["link_func"] = tolower(link_func);
    model["trans_func"] = tolower(trans_func);
    model["gain_func"] = tolower(gain_func);
    model["lag_dist"] = tolower(lag_dist);
    model["err_dist"] = tolower(err_dist);

    Rcpp::List dim;
    dim["nlag"] = nlag;
    dim["ntime"] = ntime;
    dim["truncated"] = truncated;

    Rcpp::List param;
    param["obs"] = obs_param;
    param["lag"] = lag_param;
    param["truncated"] = truncated;

    Rcpp::List settings;
    settings["model"] = model;
    settings["dim"] = dim;
    settings["param"] = param;

    return settings;
}


Model dgtf_initialize(const Rcpp::List &settings)
{
    Rcpp::List model_settings = settings["model"];
    std::string obs_dist, link_func, trans_func, gain_func, lag_dist, err_dist;
    parse_model_settings(
        obs_dist, link_func, trans_func,
        gain_func, lag_dist, err_dist,
        model_settings);

    Rcpp::List dim_settings = settings["dim"];
    unsigned int nlag, ntime;
    bool truncated;
    parse_dimension_settings(nlag, ntime, truncated, dim_settings);

    Rcpp::List param_settings = settings["param"];
    Rcpp::NumericVector lag_param, obs_param, err_param;
    parse_param_settings(obs_param, lag_param, err_param, param_settings);

    Dim dim(ntime, nlag, lag_param[1]);
    Model model(
        dim, obs_dist, link_func, gain_func, lag_dist, err_dist,
        obs_param, lag_param, err_param, trans_func);

    return model;
}


//' @export
// [[Rcpp::export]]
Rcpp::List dgtf_simulate(const Rcpp::List &settings)
{
    Model model = dgtf_initialize(settings);
    arma::vec ysim = model.simulate();
    arma::vec psi = model.transfer.fgain.psi;
    arma::vec wt = model.derr.wt;

    Rcpp::List output = settings;
    output["y"] = Rcpp::wrap(ysim);
    output["psi"] = Rcpp::wrap(psi);
    output["wt"] = Rcpp::wrap(wt);
    output["lambda"] = Rcpp::wrap(model.lambda);
    return output;
}

//' @export
// [[Rcpp::export]]
Rcpp::List dgtf_infer(
    const Rcpp::List &model_settings, 
    const arma::vec &y,
    const std::string &method,
    const Rcpp::List &method_settings,
    const bool &summarize = true)
{
    Model model = dgtf_initialize(model_settings);
    std::map<std::string, AVAIL::Algo> algo_list = AVAIL::algo_list;
    std::string algo_name = tolower(method);

    // arma::vec ypad(model.dim.nT + 1, arma::fill::zeros);
    // ypad.tail(y.n_elem) = y;

    Rcpp::List output;
    arma::mat psi(model.dim.nT + 1, 3);
    arma::vec ci_prob = {0.025, 0.5, 0.975};

    switch (algo_list[algo_name])
    {
    case AVAIL::Algo::LinearBayes:
    {
        LBA::LinearBayes linear_bayes(model, y);
        linear_bayes.init(method_settings);
        linear_bayes.filter();
        linear_bayes.smoother();
        output = linear_bayes.get_output();

        break;
    } // case Linear Bayes
    case AVAIL::Algo::MCS:
    {
        SMC::MCS mcs(model, y);
        mcs.init(method_settings);
        mcs.infer(model);
        output = mcs.get_output();

        
        break;
    } // case MCS
    case AVAIL::Algo::FFBS:
    {
        SMC::FFBS ffbs(model, y);
        ffbs.init(method_settings);
        ffbs.infer(model);
        output = ffbs.get_output();
        break;
    } // case FFBS
    case AVAIL::Algo::ParticleLearning:
    {
        SMC::PL pl(model, y);
        pl.init(method_settings);
        pl.infer(model);
        output = pl.get_output();
        
        break;
    } // case particle learning
    case AVAIL::Algo::MCMC:
    {
        MCMC::Disturbance mcmc(method_settings, model.dim);
        mcmc.infer(model, y);
        output = mcmc.get_output();
        break;
    }
    case AVAIL::Algo::HybridVariation:
    {
        VB::Hybrid hvb(model, y);
        hvb.init(method_settings);
        hvb.infer(model, y);
        output = hvb.get_output();
        break;
    }
    default:
        break;
    }// switch by algorithm


    return output;
}

