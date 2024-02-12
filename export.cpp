#include "Model.hpp"

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
        obs_dist = Rcpp::as<std::string>(model["obs_dist"]);
    }
    else
    {
        obs_dist = "poisson";
    }

    if (model.containsElementNamed("link_func"))
    {
        link_func = Rcpp::as<std::string>(model["link_func"]);
    }
    else
    {
        link_func = "identity";
    }

    if (model.containsElementNamed("trans_func"))
    {
        trans_func = Rcpp::as<std::string>(model["trans_func"]);
    }
    else
    {
        trans_func = "sliding";
    }

    if (model.containsElementNamed("gain_func"))
    {
        gain_func = Rcpp::as<std::string>(model["gain_func"]);
    }
    else
    {
        gain_func = "identity";
    }

    if (model.containsElementNamed("lag_dist"))
    {
        lag_dist = Rcpp::as<std::string>(model["lag_dist"]);
    }
    else
    {
        lag_dist = "nbinom";
    }

    if (model.containsElementNamed("err_dist"))
    {
        err_dist = Rcpp::as<std::string>(model["err_dist"]);
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
        nlag = 14;
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



//' @export
// [[Rcpp::export]]
Rcpp::List dgtf_simulate(
    const Rcpp::List &model_settings, // list(obs_dist, link_func, trans_func, gain_func, lag_dist, err_dist)
    const Rcpp::List &dim_settings, // list(nlag=<value>, ntime=<value>, truncated=<value>)
    const Rcpp::NumericVector &obs_param = Rcpp::NumericVector::create(0., 30.),
    const Rcpp::NumericVector &lag_param = Rcpp::NumericVector::create(0.395, 6),
    const Rcpp::NumericVector &err_param = Rcpp::NumericVector::create(0.01, 0.) // (W, w[0]))
)
{
    std::string obs_dist, link_func, trans_func, gain_func, lag_dist, err_dist;
    parse_model_settings(
        obs_dist, link_func, trans_func,
        gain_func, lag_dist, err_dist,
        model_settings);


    unsigned int nlag, ntime;
    bool truncated;
    parse_dimension_settings(nlag, ntime, truncated, dim_settings);
    Dim dim(ntime, nlag, lag_param[1]);

    Model model(
        dim, obs_dist, link_func, gain_func, lag_dist,
        obs_param, lag_param, err_param, trans_func);

    arma::vec ysim = model.simulate();

    Rcpp::List param;
    param["obs"] = obs_param;
    param["lag"] = lag_param;
    param["err"] = err_param;

    Rcpp::List output;
    output["y"] = Rcpp::wrap(ysim);
    output["model"] = model_settings;
    output["dim"] = dim_settings;
    output["param"] = param;

    return output;
}

// //' @export
// // [[Rcpp::export]]
// double test_transfer_sliding(
//     const unsigned int &t,
//     const unsigned int &nlag,
//     const arma::vec &y,
//     const arma::vec &Fphi,
//     const arma::vec &hpsi)
// {
//     double ft = TransFunc::transfer_sliding(t, nlag, y, Fphi, hpsi);
//     return ft;
// }

// //' @export
// // [[Rcpp::export]]
// arma::vec test_get_Fphi(
//     const unsigned int &nlag,
//     const arma::vec &lag_par,
//     const std::string &lag_dist)
// {
//     arma::vec Fphi = LagDist::get_Fphi(nlag, lag_dist, lag_par[0], lag_par[1]);
//     return Fphi;
// }

// //' @export
// // [[Rcpp::export]]
// arma::vec test_dnbinom(
//     const unsigned int &nL,
//     const double &kappa,
//     const double &r
// )
// {
//     arma::vec pmf = nbinom::dnbinom(nL, kappa, r);
//     return pmf;
// }

// //' @export
// // [[Rcpp::export]]
// arma::vec test_psi2hpsi(
//     const arma::vec &psi,
//     const std::string &gain_func
// )
// {
//     arma::vec hpsi = GainFunc::psi2hpsi(psi, gain_func);
//     return hpsi;
// }

// //' @export
// // [[Rcpp::export]]
// double test_sample(
//     const double &mu, // mean
//     const std::string &obs_dist,
//     const double &par1,
//     const double &par2)
// {
//     double y = ObsDist::sample(mu, par2, obs_dist);
//     return y;
// }


// //' @export
// // [[Rcpp::export]]
// Rcpp::List test_simulate(
//     const arma::vec &psi,
//     const unsigned int &nlag,
//     const double &y0 = 0.,
//     const std::string &gain_func = "softplus",
//     const std::string &lag_dist = "lognormal",
//     const std::string &link_func = "identity",
//     const std::string &obs_dist = "nbinom",
//     const Rcpp::NumericVector &lag_param = Rcpp::NumericVector::create(1.4, 0.3),
//     const Rcpp::NumericVector &obs_param = Rcpp::NumericVector::create(0., 30))
// {
//     Rcpp::List sim = Model::simulate(
//         psi, nlag, y0, 
//         gain_func, 
//         lag_dist,
//         link_func,
//         obs_dist, 
//         lag_param, 
//         obs_param);

//     return sim;
// }