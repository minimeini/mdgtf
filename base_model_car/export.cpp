#include <chrono>
#include "../core/GainFunc.hpp"
#include "Model.hpp"
#include "MCMC.hpp"

using namespace Rcpp;
// [[Rcpp::plugins(cpp17)]]
// [[Rcpp::depends(RcppArmadillo)]]

//' @export
// [[Rcpp::export]]
arma::mat sample_car(
    const unsigned int &k,
    const arma::mat &V,
    const arma::vec &car_params
)
{
    SpatialStructure spatial(V, car_params[0], car_params[1], car_params[2]);
    return spatial.prior_sample_spatial_effects(k);
}

//' @export
// [[Rcpp::export]]
Rcpp::List mdgtf_default_algo_settings(const std::string &method)
{
    std::map<std::string, AVAIL::Algo> algo_list = AVAIL::algo_list;
    std::string method_name = tolower(method);

    Rcpp::List opts;
    switch (algo_list[method_name])
    {
        case AVAIL::MCMC:
            opts = MCMC::get_default_settings();
            break;
        default:
            throw std::invalid_argument("mdgtf_default_algo_settings - unknown method: " + method);
    }

    return opts;
}


//' @export
// [[Rcpp::export]]
Rcpp::List mdgtf_simulate(
    const Rcpp::List &settings,
    const unsigned int &ntime
)
{
    Model model(settings);
    arma::mat Y, Lambda, wt, Psi;
    model.simulate(Y, Lambda, wt, Psi, ntime);

    return Rcpp::List::create(
        Rcpp::Named("Y") = Y,
        Rcpp::Named("Lambda") = Lambda,
        Rcpp::Named("wt") = wt,
        Rcpp::Named("Psi") = Psi,
        Rcpp::Named("Rt") = GainFunc::psi2hpsi<arma::mat>(Psi, model.fgain),
        Rcpp::Named("model") = settings
    );
}


//' @export
// [[Rcpp::export]]
Rcpp::List mdgtf_infer(
    const Rcpp::List &model_settings,
    const arma::mat &Y_in,
    const arma::mat &wt_in,
    const std::string &method,
    const Rcpp::List &method_settings
)
{
    Model model(model_settings);
    std::map<std::string, AVAIL::Algo> algo_list = AVAIL::algo_list;
    std::string algo_name = tolower(method);

    auto start = std::chrono::high_resolution_clock::now();
    Rcpp::List output;
    switch (algo_list[algo_name])
    {
    case AVAIL::Algo::MCMC:
    {
        MCMC mcmc(method_settings);
        mcmc.infer(model, Y_in, wt_in);
        output = mcmc.get_output();
        break;
    }
    default:
        break;
    }

    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout << "\nElapsed time: " << duration.count() << " microseconds" << std::endl;

    return output;
}