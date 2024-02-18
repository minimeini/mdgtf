#include "Model.hpp"
#include "LinearBayes.h"
#include "lbe_poisson.h"

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
        dim, obs_dist, link_func, gain_func, lag_dist,
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

    Rcpp::List output = settings;
    output["y"] = Rcpp::wrap(ysim);
    output["psi"] = Rcpp::wrap(psi);
    return output;
}

//' @export
// [[Rcpp::export]]
Rcpp::List dgtf_infer(
    const Rcpp::List &model_settings, 
    const arma::vec &y)
{
    Model model = dgtf_initialize(model_settings);

    arma::vec ypad(model.dim.nT + 1, arma::fill::zeros);
    ypad.tail(y.n_elem) = y;

    LBA::LinearBayes linear_bayes(model, ypad);
    linear_bayes.filter();
    linear_bayes.smoother();

    Rcpp::List output;

    output["at"] = Rcpp::wrap(linear_bayes.at);
    output["Rt"] = Rcpp::wrap(linear_bayes.Rt);
    output["mt"] = Rcpp::wrap(linear_bayes.mt);
    output["Ct"] = Rcpp::wrap(linear_bayes.Ct);
    output["atilde"] = Rcpp::wrap(linear_bayes.atilde);
    output["Rtilde"] = Rcpp::wrap(linear_bayes.Rtilde);

    arma::mat psi_filter = LBA::LinearBayes::get_psi(linear_bayes.mt, linear_bayes.Ct);
    arma::mat psi_smooth = LBA::LinearBayes::get_psi(linear_bayes.atilde, linear_bayes.Rtilde);
    output["psi_filter"] = Rcpp::wrap(psi_filter);
    output["psi_smooth"] = Rcpp::wrap(psi_smooth);

    return output;
}
