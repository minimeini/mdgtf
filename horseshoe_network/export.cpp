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
    const Eigen::MatrixXd &dist_matrix,     // ns x ns, pairwise distance matrix
    const Eigen::MatrixXd &mobility_matrix, // ns x ns, pairwise mobility matrix
    const std::string &fgain = "softplus",
    const double &mu = 1.0,
    const double &W = 0.001,
    const Rcpp::Nullable<Rcpp::List> &spatial_opts = R_NilValue,
    const Rcpp::Nullable<Rcpp::List> &lagdist_opts = R_NilValue
)
{
    Rcpp::List spatial_defaults = Rcpp::List::create(
        Rcpp::Named("rho_dist") = 1.0,
        Rcpp::Named("rho_mobility") = 1.0,
        Rcpp::Named("shared_tau") = true
    );
    Rcpp::List lagdist_defaults = Rcpp::List::create(
        Rcpp::Named("name") = "lognorm",
        Rcpp::Named("par1") = LN_MU,
        Rcpp::Named("par2") = LN_SD2,
        Rcpp::Named("truncated") = true,
        Rcpp::Named("rescaled") = true
    );

    Rcpp::List spatial_opts_use = spatial_opts.isNull() ? spatial_defaults : Rcpp::as<Rcpp::List>(spatial_opts);
    Rcpp::List lagdist_opts_use = lagdist_opts.isNull() ? lagdist_defaults : Rcpp::as<Rcpp::List>(lagdist_opts);

    Model model(
        nt,
        dist_matrix,
        mobility_matrix,
        fgain,
        mu,
        W,
        spatial_opts_use,
        lagdist_opts_use
    );

    return model.simulate();
} // simulate_network_hawkes


//' @export
// [[Rcpp::export]]
Rcpp::List infer_network_hawkes(
    const Eigen::MatrixXd &Y,               // (nt + 1) x ns, observed primary infections
    const Eigen::MatrixXd &dist_matrix,     // ns x ns, pairwise distance matrix
    const Eigen::MatrixXd &mobility_matrix, // ns x ns, pairwise mobility matrix
    const std::string &fgain = "softplus",
    const Rcpp::Nullable<Rcpp::List> &lagdist_opts = R_NilValue,
    const unsigned int &nburnin = 1000,
    const unsigned int &nsamples = 1000,
    const unsigned int &nthin = 1,
    const Rcpp::Nullable<Rcpp::List> &mcmc_opts = R_NilValue,
    const bool &shared_tau = true,
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

    Model model(
        Y,
        dist_matrix,
        mobility_matrix,
        fgain,
        lagdist_opts_use,
        shared_tau
    );


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
