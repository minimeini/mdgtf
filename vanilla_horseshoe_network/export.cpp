#include <chrono>
#include <vector>
#include <complex>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <unsupported/Eigen/FFT>
#include "../inference/diagnostics.hpp"
#include "Model.hpp"

using namespace Rcpp;
// [[Rcpp::plugins(cpp17)]]
// [[Rcpp::depends(RcppEigen, RcppProgress)]]


//' @export
// [[Rcpp::export]]
double effective_sample_size(const Eigen::Map<Eigen::VectorXd> &draws)
{
    return effective_sample_size_eigen_impl(draws);
}


//' @export
// [[Rcpp::export]]
Rcpp::List simulate_network_hawkes(
    const Eigen::Index &nt,
    const Eigen::Index &ns,
    const double &mu = 1.0,
    const double &W = 0.001,
    const double &c_sq = 4.0,
    const std::string &fgain = "softplus",
    const Rcpp::Nullable<Rcpp::List> &lagdist_opts = R_NilValue
)
{
    Rcpp::List lagdist_defaults = Rcpp::List::create(
        Rcpp::Named("name") = "lognorm",
        Rcpp::Named("par1") = LN_MU,
        Rcpp::Named("par2") = LN_SD2,
        Rcpp::Named("truncated") = true,
        Rcpp::Named("rescaled") = true
    );

    Rcpp::List lagdist_opts_use = lagdist_opts.isNull() ? lagdist_defaults : Rcpp::as<Rcpp::List>(lagdist_opts);

    Model model(nt, ns, mu, W, c_sq, fgain, lagdist_opts_use);

    return model.simulate();
} // simulate_network_hawkes


//' @export
// [[Rcpp::export]]
Rcpp::List infer_network_hawkes(
    const Eigen::MatrixXd &Y,               // (nt + 1) x ns, observed primary infections
    const double &c_sq = 4.0,
    const std::string &fgain = "softplus",
    const Rcpp::Nullable<Rcpp::List> &lagdist_opts = R_NilValue,
    const unsigned int &nburnin = 1000,
    const unsigned int &nsamples = 1000,
    const unsigned int &nthin = 1,
    const Rcpp::Nullable<Rcpp::List> &mcmc_opts = R_NilValue,
    const bool &sample_augmented_N = false
)
{
    Rcpp::List lagdist_defaults = Rcpp::List::create(
        Rcpp::Named("name") = "lognorm",
        Rcpp::Named("par1") = LN_MU,
        Rcpp::Named("par2") = LN_SD2,
        Rcpp::Named("truncated") = true,
        Rcpp::Named("rescaled") = true
    );
    Rcpp::List lagdist_opts_use = lagdist_opts.isNull() ? lagdist_defaults : Rcpp::as<Rcpp::List>(lagdist_opts);

    Model model(Y, c_sq, fgain, lagdist_opts_use);


    auto start = std::chrono::high_resolution_clock::now();
    Rcpp::List output = model.run_mcmc(
        Y, nburnin, nsamples, nthin, mcmc_opts, sample_augmented_N
    );
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    Rcpp::Rcout << "\nElapsed time: " << duration.count() << " milliseconds" << std::endl;
    output["elapsed_time_ms"] = duration.count();
    return output;
} // infer_network_hawkes
