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
double compute_future_secondary_infections_sum(
    const Rcpp::NumericVector &N_array, // R array of size ns x ns x (nt + 1) x (nt + 1)
    const unsigned int &s0, // Index of source location
    const unsigned int &t0 // Index of source time point
)
{
    Rcpp::NumericVector N_array_copy(N_array); // make a copy to avoid modifying the original R array
    Eigen::Tensor<double, 4> N = r_to_tensor4(N_array_copy);
    return TemporalTransmission::compute_N_future_sum(N, s0, t0);
}


//' @export
// [[Rcpp::export]]
Eigen::MatrixXd calculate_alpha(
    const Eigen::Map<Eigen::MatrixXd> &dist_scaled,     // ns x ns, pairwise scaled distance matrix
    const Eigen::Map<Eigen::MatrixXd> &mobility_scaled, // ns x ns, pairwise scaled mobility matrix
    const double &rho_dist,
    const double &rho_mobility
)
{
    SpatialNetwork spatial(dist_scaled, mobility_scaled);
    spatial.rho_dist = rho_dist;
    spatial.rho_mobility = rho_mobility;
    spatial.compute_alpha();

    return spatial.alpha;
} // sample_A


//' @export
// [[Rcpp::export]]
Rcpp::List infer_spatial_network(
    const Rcpp::NumericVector &N_array, // R array of size ns x ns x (nt + 1) x (nt + 1)
    const Eigen::Map<Eigen::MatrixXd> &dist_scaled, // ns x ns, pairwise scaled distance matrix
    const Eigen::Map<Eigen::MatrixXd> &mobility_scaled, // ns x ns, pairwise scaled mobility matrix
    const unsigned int &nburnin = 1000,
    const unsigned int &nsamples = 1000,
    const unsigned int &nthin = 1,
    const double &step_size_init = 0.1,
    const unsigned int &n_leapfrog_init = 20
)
{
    SpatialNetwork spatial(dist_scaled, mobility_scaled, true);

    auto start = std::chrono::high_resolution_clock::now();
    Rcpp::List output = spatial.run_mcmc(
        N_array,
        nburnin,
        nsamples,
        nthin,
        step_size_init,
        n_leapfrog_init
    );
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    Rcpp::Rcout << "\nElapsed time: " << duration.count() << " milliseconds" << std::endl;
    return output;
} // infer_spatial_network


//' @export
// [[Rcpp::export]]
Eigen::MatrixXd sample_Rt(
    const unsigned int &nt,
    const unsigned int &ns,
    const double &W,
    const std::string &fgain = "softplus"
)
{
    TemporalTransmission temporal(ns, nt, fgain, W);
    temporal.sample_wt();
    return temporal.compute_Rt();
} // sample_Rt


//' @export
// [[Rcpp::export]]
Rcpp::List infer_temporal_transmission(
    const Rcpp::NumericVector &N_array, // R array of size ns x ns x (nt + 1) x (nt + 1), unobserved secondary infections
    const Eigen::MatrixXd &Y, // (nt + 1) x ns, observed case counts
    const unsigned int &nburnin = 1000,
    const unsigned int &nsamples = 1000,
    const unsigned int &nthin = 1,
    const double &W_init = 0.01,
    const std::string &fgain = "softplus",
    const double &mh_sd = 1.0,
    const double &prior_shape_W = 1.0,
    const double &prior_rate_W = 1.0
)
{
    Eigen::Index nt = Y.rows() - 1;
    Eigen::Index ns = Y.cols();
    TemporalTransmission temporal(ns, nt, fgain, W_init);

    auto start = std::chrono::high_resolution_clock::now();
    Rcpp::List output = temporal.run_mcmc(
        N_array,
        Y,
        nburnin,
        nsamples,
        nthin,
        mh_sd,
        prior_shape_W,
        prior_rate_W
    );
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    Rcpp::Rcout << "\nElapsed time: " << duration.count() << " milliseconds" << std::endl;
    return output;
} // infer_temporal_transmission


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
        Rcpp::Named("kappa") = 1.0
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


void show_vec(const Eigen::VectorXd& v) {
    Rcpp::NumericVector rv = Rcpp::wrap(v);
    Rcpp::Rcout << rv << "\n";        // R-style print
    // or Rcpp::Rcout << v.transpose() << "\n"; // plain Eigen format
}


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
        lagdist_opts_use
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
