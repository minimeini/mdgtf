#include "Model.hpp"
#include "LinearBayes.hpp"
#include "SequentialMonteCarlo.hpp"
#include "MCMC.hpp"
#include "VariationalBayes.hpp"

using namespace Rcpp;
// [[Rcpp::plugins(cpp17)]]
// [[Rcpp::depends(RcppArmadillo,nloptr, BH)]]


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

//' @export
// [[Rcpp::export]]
Rcpp::List dgtf_default_model()
{
    return Model::default_settings();
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





//' @export
// [[Rcpp::export]]
Rcpp::List dgtf_simulate(const Rcpp::List &settings)
{
    Model model(settings);
    // Model model = dgtf_initialize(settings);
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
    const bool &summarize = true,
    const bool &forecast_error = true,
    const bool &fitted_error = true,
    const std::string &loss_func = "quadratic")
{
    // Model model = dgtf_initialize(model_settings);
    Model model(model_settings);
    std::map<std::string, AVAIL::Algo> algo_list = AVAIL::algo_list;
    std::string algo_name = tolower(method);

    unsigned int nforecast = 0;
    if (method_settings.containsElementNamed("num_step_ahead_forecast"))
    {
        nforecast = Rcpp::as<unsigned int>(method_settings["num_step_ahead_forecast"]);
    }

    // arma::vec ypad(model.dim.nT + 1, arma::fill::zeros);
    // ypad.tail(y.n_elem) = y;

    Rcpp::List output, forecast, error;
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

        if (forecast_error)
        {
            Rcpp::List tmp = linear_bayes.forecast_error(1000, loss_func);
            error["forecast"] = tmp;
        }
        if (fitted_error)
        {
            Rcpp::List tmp = linear_bayes.fitted_error(1000, loss_func);
            error["fitted"] = tmp;
        }

        break;
    } // case Linear Bayes
    case AVAIL::Algo::MCS:
    {
        SMC::MCS mcs(model, y);
        mcs.init(method_settings);
        mcs.infer(model);
        output = mcs.get_output();

        if (nforecast > 0)
        {
            forecast = mcs.forecast(model);
        }

        if (forecast_error)
        {
            Rcpp::List tmp = mcs.forecast_error(model, loss_func);
            error["forecast"] = tmp;
        }
        if (fitted_error)
        {
            Rcpp::List tmp = mcs.fitted_error(model, loss_func);
            error["fitted"] = tmp;
        }

        break;
    } // case MCS
    case AVAIL::Algo::FFBS:
    {
        SMC::FFBS ffbs(model, y);
        ffbs.init(method_settings);
        ffbs.infer(model);
        output = ffbs.get_output();

        if (nforecast > 0)
        {
            forecast = ffbs.forecast(model);
        }

        if (forecast_error)
        {
            Rcpp::List tmp = ffbs.forecast_error(model, loss_func);
            error["forecast"] = tmp;
        }
        if (fitted_error)
        {
            Rcpp::List tmp = ffbs.fitted_error(model, loss_func);
            error["fitted"] = tmp;
        }

        break;
    } // case FFBS
    case AVAIL::Algo::ParticleLearning:
    {
        SMC::PL pl(model, y);
        pl.init(method_settings);
        pl.infer(model);
        output = pl.get_output();

        if (nforecast > 0)
        {
            forecast = pl.forecast(model);
        }

        if (forecast_error)
        {
            Rcpp::List tmp = pl.forecast_error(model, loss_func);
            error["forecast"] = tmp;
        }
        if (fitted_error)
        {
            Rcpp::List tmp = pl.fitted_error(model, loss_func);
            error["fitted"] = tmp;
        }


        break;
    } // case particle learning
    case AVAIL::Algo::MCMC:
    {
        MCMC::Disturbance mcmc(model, y);
        mcmc.init(method_settings);
        mcmc.infer(model);
        output = mcmc.get_output();

        if (nforecast > 0)
        {
            forecast = mcmc.forecast(model);
        }

        if (forecast_error)
        {
            Rcpp::List tmp = mcmc.forecast_error(model, loss_func);
            error["forecast"] = tmp;
        }
        if (fitted_error)
        {
            Rcpp::List tmp = mcmc.fitted_error(model, loss_func);
            error["fitted"] = tmp;
        }
        break;
    }
    case AVAIL::Algo::HybridVariation:
    {
        VB::Hybrid hvb(model, y);
        hvb.init(method_settings);
        hvb.infer(model);
        output = hvb.get_output();

        if (nforecast > 0)
        {
            forecast = hvb.forecast(model);
        }

        if (forecast_error)
        {
            Rcpp::List tmp = hvb.forecast_error(model, loss_func);
            error["forecast"] = tmp;
        }
        if (fitted_error)
        {
            Rcpp::List tmp = hvb.fitted_error(model, loss_func);
            error["fitted"] = tmp;
        }
        break;
    }
    default:
        break;
    }// switch by algorithm

    Rcpp::List out;
    if (nforecast > 0 || forecast_error || fitted_error)
    {
        out["fit"] = output;
        if (nforecast > 0) { out["pred"] = forecast; }
        if (forecast_error || fitted_error)
        {
            out["error"] = error;
        }
    }
    else
    {
        out = output;
    }


    return out;
}


//' @export
// [[Rcpp::export]]
Rcpp::List dgtf_forecast(
    const Rcpp::List &model_settings,
    const arma::vec &y, // (nT + 1) x 1
    const arma::mat &psi, // (nT + 1) x nsample
    const arma::vec &W, // nsample x 1
    const unsigned int &k = 1 // k-step-ahead forecasting
    )
{
    Model model(model_settings);
    Rcpp::List out = Model::forecast(
        y, psi, W, model, k
    );
    return out;
}

//' @export
// [[Rcpp::export]]
Rcpp::List dgtf_evaluate(
    const Rcpp::List &model_settings,
    const arma::vec &y,   // (nT + 1) x 1
    const arma::mat &psi, // (nT + 1) x nsample
    const std::string &loss_func = "quadratic"
)
{
    Model model(model_settings);
    Rcpp::List forecast = Model::forecast_error(psi, y, model, loss_func);
    Rcpp::List fitted = Model::fitted_error(psi, y, model, loss_func);

    Rcpp::List out;
    out["forecast"] = forecast;
    out["fitted"] = fitted;

    return out;
}