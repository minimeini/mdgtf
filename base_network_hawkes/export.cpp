#include <chrono>
#include "Model.hpp"

using namespace Rcpp;
// [[Rcpp::plugins(cpp17)]]
// [[Rcpp::depends(RcppEigen, RcppProgress)]]


//' @export
// [[Rcpp::export]]
Eigen::MatrixXd sample_A(
    const Eigen::Map<Eigen::MatrixXd> &dist_scaled,     // ns x ns, pairwise scaled distance matrix
    const Eigen::Map<Eigen::MatrixXd> &mobility_scaled, // ns x ns, pairwise scaled mobility matrix
    const Eigen::Map<Eigen::VectorXd> &kappa,           // ns x 1, node-specific baseline intensities
    const double &rho_dist,
    const double &rho_mobility
)
{
    SpatialNetwork spatial(dist_scaled, mobility_scaled);
    spatial.rho_dist = rho_dist;
    spatial.rho_mobility = rho_mobility;
    spatial.kappa = kappa;

    spatial.sample_A();
    return spatial.A;
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
    const unsigned int &hmc_block = 2, // 2: 2-block (kappa | rho_dist, rho_mobility) HMC; 1: joint HMC
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
        hmc_block,
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
    const double &W = 0.01,
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
    const double &mh_sd_wt = 1.0,
    const double &prior_shape_W = 1.0,
    const double &prior_rate_W = 1.0,
    const unsigned int &spatial_hmc_block = 2, // 2: 2-block (kappa | rho_dist, rho_mobility) HMC; 1: joint HMC
    const double &spatial_hmc_step_size = 0.1,
    const unsigned int &spatial_hmc_nleapfrog = 20
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
        Y,
        nburnin,
        nsamples,
        nthin,
        mh_sd_wt,
        prior_shape_W,
        prior_rate_W,
        spatial_hmc_block,
        spatial_hmc_step_size,
        spatial_hmc_nleapfrog
    );
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    Rcpp::Rcout << "\nElapsed time: " << duration.count() << " milliseconds" << std::endl;
    return output;
} // infer_network_hawkes
