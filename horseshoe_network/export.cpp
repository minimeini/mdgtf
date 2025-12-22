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
Rcpp::NumericMatrix standardize_alpha(const Rcpp::NumericMatrix &alpha_in)
{
    Eigen::MatrixXd alpha(Rcpp::as<Eigen::MatrixXd>(alpha_in));
    Eigen::MatrixXd alpha_std = alpha;
    for (Eigen::Index k = 0; k < alpha.cols(); k++)
    {
        double off_diag_sum = alpha.col(k).sum() - alpha(k, k);
        for (Eigen::Index s = 0; s < alpha.rows(); s++)
        {
            if (s != k)
            {
                alpha_std(s, k) = alpha(s, k) / off_diag_sum * (1.0 - alpha(k, k));
            }
        }
    }

    return Rcpp::wrap(alpha_std);
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
    const Rcpp::Nullable<Rcpp::List> &lagdist_opts = R_NilValue,
    const Rcpp::Nullable<Rcpp::NumericMatrix> &dist_matrix = R_NilValue,
    const Rcpp::Nullable<Rcpp::NumericMatrix> &mobility_matrix = R_NilValue,
    const Rcpp::Nullable<Rcpp::List> &spatial_opts = R_NilValue
)
{
    Rcpp::List lagdist_defaults = Rcpp::List::create(
        Rcpp::Named("name") = "lognorm",
        Rcpp::Named("par1") = LN_MU,
        Rcpp::Named("par2") = LN_SD2,
        Rcpp::Named("truncated") = true,
        Rcpp::Named("rescaled") = true
    );

    Rcpp::List spatial_defaults = Rcpp::List::create(
        Rcpp::Named("rho_dist") = dist_matrix.isNull() ? 0.0 : 1.0,
        Rcpp::Named("rho_mobility") = mobility_matrix.isNull() ? 0.0 : 1.0,
        Rcpp::Named("c_sq") = 4.0
    );

    Rcpp::List lagdist_opts_use = lagdist_opts.isNull() ? lagdist_defaults : Rcpp::as<Rcpp::List>(lagdist_opts);
    Rcpp::List spatial_opts_use = spatial_opts.isNull() ? spatial_defaults : Rcpp::as<Rcpp::List>(spatial_opts);

    Model model(
        nt, ns, 
        dist_matrix, mobility_matrix, 
        mu, W, fgain, 
        spatial_opts_use, 
        lagdist_opts_use
    );

    return model.simulate();
} // simulate_network_hawkes


//' @export
// [[Rcpp::export]]
Rcpp::List infer_network_hawkes(
    const Rcpp::NumericVector &Y_in,               // (nt + 1) x ns, observed primary infections or a vector of (nt + 1) length for a single series
    const Rcpp::Nullable<Rcpp::NumericMatrix> &dist_matrix = R_NilValue,     // ns x ns, pairwise distance matrix
    const Rcpp::Nullable<Rcpp::NumericMatrix> &mobility_matrix = R_NilValue, // ns x ns, pairwise mobility matrix
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

    Eigen::MatrixXd Y;
    if (Y_in.hasAttribute("dim"))
    {
        // Already a matrix: convert directly
        Rcpp::NumericMatrix Y_mat(Y_in);
        Y = Rcpp::as<Eigen::MatrixXd>(Y_mat);
    }
    else
    {
        // Vector: make it a (nt + 1) x 1 matrix
        Eigen::VectorXd y_vec = Rcpp::as<Eigen::VectorXd>(Y_in);
        Y.resize(y_vec.size(), 1);
        Y.col(0) = y_vec;
    }

    Model model(Y, dist_matrix, mobility_matrix, c_sq, fgain, lagdist_opts_use);

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
