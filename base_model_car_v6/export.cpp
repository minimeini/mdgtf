#include <chrono>
#include "../core/GainFunc.hpp"
#include "Model.hpp"
#include "MCMC.hpp"

using namespace Rcpp;
// [[Rcpp::plugins(cpp17)]]
// [[Rcpp::depends(RcppArmadillo, RcppProgress, RcppEigen)]]


//' @export
// [[Rcpp::export]]
double gelman_rubin_cpp(const arma::mat &samples)
{
    return gelman_rubin(samples);
}


//' @export
// [[Rcpp::export]]
arma::mat sample_car(
    const unsigned int &k,
    const arma::mat &V,
    const arma::vec &car_params
)
{
    BYM2 spatial(V);
    spatial.mu = car_params[0];
    spatial.tau_b = car_params[1];
    spatial.phi = car_params[2];
    return spatial.sample_spatial_effects(k);
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
} // end of mdgtf_default_algo_settings()


//' @export
// [[Rcpp::export]]
Rcpp::List mdgtf_simulate(
    const Rcpp::List &settings,
    const unsigned int &ntime
)
{
    Model model(settings);
    arma::mat Y, Lambda;
    arma::vec psi1_spatial, psi2_temporal, wt2_temporal;
    model.simulate(
        Y, Lambda, 
        psi1_spatial, psi2_temporal, wt2_temporal, 
        ntime
    );

    Rcpp::List output = Rcpp::List::create(
        Rcpp::Named("Y") = Y,
        Rcpp::Named("Lambda") = Lambda,
        Rcpp::Named("psi1_spatial") = psi1_spatial,
        Rcpp::Named("psi2_temporal") = psi2_temporal,
        Rcpp::Named("wt2_temporal") = wt2_temporal,
        Rcpp::Named("model") = settings
    );

    if (model.zero[0].inflated)
    {
        arma::mat Z(Y.n_rows, Y.n_cols, arma::fill::ones);
        arma::mat Z_prob(Y.n_rows, Y.n_cols, arma::fill::ones);
        for (unsigned int s = 0; s < Y.n_rows; s++)
        {
            Z.row(s) = model.zero[s].z.t();
            Z_prob.row(s) = model.zero[s].prob.t();
        }

        output["zero"] = Rcpp::List::create(
            Rcpp::Named("z") = Z,
            Rcpp::Named("prob") = Z_prob
        );
    }

    return output;
} // end of mdgtf_simulate()


//' @export
// [[Rcpp::export]]
Rcpp::List mdgtf_infer(
    const Rcpp::List &model_settings,
    const arma::mat &Y_in,
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
        mcmc.infer(model, Y_in);
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
} // end of mdgtf_infer()


//' @export
// [[Rcpp::export]]
Rcpp::List mdgtf_posterior_predictive(
    const Rcpp::List &output, 
    const Rcpp::List &model_opts, 
    const Rcpp::List &mcmc_opts,
    const arma::mat &Y, // nS x (nT + 1)
    const unsigned int &nsample,
    const unsigned int &nrep = 1
)
{
    Model model(model_opts);
    arma::cube Y_pred, Y_residual, log_a;
    // Y_pred: nS x (nT + 1) x (nsample * nrep)
    // Y_residual: nS x (nT + 1) x nsample
    // hPsi: nS x (nT + 1) x nsample
    double chi_sqr = model.sample_posterior_predictive_y(
        Y_pred, Y_residual, 
        output, mcmc_opts, 
        Y, nsample, nrep
    );

    arma::cube Y_pred_trunc = Y_pred.cols(1, Y_pred.n_cols - 1); // nS x nT x (nsample * nrep)
    arma::mat Y_trunc = Y.cols(1, Y.n_cols - 1); // nS x nT
    arma::vec crps = calculate_crps(Y_trunc, Y_pred_trunc);

    arma::vec prob = {0.025, 0.5, 0.975};
    arma::cube Y_pred_ci(Y_pred.n_rows, Y_pred.n_cols, prob.n_elem);
    arma::vec Y_coverage(Y_pred.n_rows, arma::fill::zeros);
    arma::vec Y_width(Y_pred.n_rows, arma::fill::zeros);
    for (unsigned int i = 0; i < Y_pred.n_rows; i++)
    {
        for (unsigned int j = 0; j < Y_pred.n_cols; j++)
        {
            arma::vec samples = Y_pred.tube(i, j); // (nsample * nrep) x 1
            Y_pred_ci.tube(i, j) = arma::quantile(samples, prob); // 3 x 1
        }

        arma::vec lobnd = Y_pred_ci.slice(0).row(i).t();
        arma::vec upbnd = Y_pred_ci.slice(2).row(i).t();
        arma::vec true_val = Y.row(i).t();

        double covered = arma::accu((true_val >= lobnd) % (true_val <= upbnd));
        Y_coverage.at(i) = covered / static_cast<double>(true_val.n_elem);
        Y_width.at(i) = arma::mean(upbnd - lobnd);
    }

    Rcpp::List stats = Rcpp::List::create(
        Rcpp::Named("Y_pred") = Y_pred_ci, // nS x (nT + 1) x 3
        Rcpp::Named("Y_residual") = Y_residual,
        Rcpp::Named("chi_sqr_avg") = chi_sqr,
        Rcpp::Named("crps") = crps, // nS x 1
        Rcpp::Named("crps_avg") = arma::mean(crps),
        Rcpp::Named("Y_coverage") = Y_coverage, // nS x 1
        Rcpp::Named("Y_coverage_avg") = arma::mean(Y_coverage),
        Rcpp::Named("Y_width") = Y_width, // nS x 1
        Rcpp::Named("Y_width_avg") = arma::mean(Y_width)
    );

    return stats;
}
