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
} // effective_sample_size


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
} // standardize_alpha


//' @export
// [[Rcpp::export]]
List compute_apparent_R(
    const NumericVector& alpha_array,  // ns x ns x nsample
    const NumericVector& Rt_array,     // nt x ns x nsample
    const NumericMatrix& Y,            // nt x ns
    const Rcpp::Nullable<Rcpp::List> &lagdist_opts = R_NilValue,
    const bool return_quantiles = false,
    const NumericVector quantiles = NumericVector::create(0.025, 0.5, 0.975)
) {
    Rcpp::List lagdist_defaults = Rcpp::List::create(
        Rcpp::Named("name") = "lognorm",
        Rcpp::Named("par1") = LN_MU,
        Rcpp::Named("par2") = LN_SD2,
        Rcpp::Named("truncated") = true,
        Rcpp::Named("rescaled") = true
    );
    Rcpp::List lagdist_opts_use = lagdist_opts.isNull() ? lagdist_defaults : Rcpp::as<Rcpp::List>(lagdist_opts);
    LagDist dlag(lagdist_opts_use);
    Rcpp::NumericVector phi = Rcpp::as<Rcpp::NumericVector>(Rcpp::wrap(dlag.Fphi));

    // Get dimensions
    IntegerVector alpha_dims = alpha_array.attr("dim");
    int ns = alpha_dims[0];
    int nsample = alpha_dims[2];
    
    IntegerVector Rt_dims = Rt_array.attr("dim");
    int nt = Rt_dims[0];
    
    int max_lag = phi.size();
    
    // Output arrays: nt x ns x nsample
    NumericVector R_apparent(nt * ns * nsample);
    NumericVector R_local(nt * ns * nsample);
    NumericVector R_import(nt * ns * nsample);
    
    R_apparent.attr("dim") = IntegerVector::create(nt, ns, nsample);
    R_local.attr("dim") = IntegerVector::create(nt, ns, nsample);
    R_import.attr("dim") = IntegerVector::create(nt, ns, nsample);
    
    // Precompute convolution: conv(t, k) = sum_{l<t} R_{k,l} * phi_{t-l} * y_{k,l}
    // and denominator: denom(t, s) = sum_{l<t} phi_{t-l} * y_{s,l}
    
    // Loop over posterior samples
    for (int m = 0; m < nsample; m++) {
        
        // Extract alpha for this sample: ns x ns matrix
        // alpha_array is stored as [s, k, m] in column-major order
        Eigen::MatrixXd alpha_m(ns, ns);
        for (int k = 0; k < ns; k++) {
            for (int s = 0; s < ns; s++) {
                alpha_m(s, k) = alpha_array[s + k * ns + m * ns * ns];
            }
        }
        
        // Extract Rt for this sample: nt x ns matrix
        // Rt_array is stored as [t, k, m] in column-major order
        Eigen::MatrixXd Rt_m(nt, ns);
        for (int k = 0; k < ns; k++) {
            for (int t = 0; t < nt; t++) {
                Rt_m(t, k) = Rt_array[t + k * nt + m * nt * ns];
            }
        }
        
        // Precompute weighted convolutions for each location k
        // conv_Ry(t, k) = sum_{l < t} R_{k,l} * phi_{t-l} * y_{k,l}
        Eigen::MatrixXd conv_Ry(nt, ns);
        conv_Ry.setZero();
        
        // Precompute denominator for each location s
        // conv_y(t, s) = sum_{l < t} phi_{t-l} * y_{s,l}
        Eigen::MatrixXd conv_y(nt, ns);
        conv_y.setZero();
        
        for (int t = 1; t < nt; t++) {
            for (int k = 0; k < ns; k++) {
                double sum_Ry = 0.0;
                double sum_y = 0.0;
                
                // Sum over lags
                int min_l = std::max(0, t - max_lag);
                for (int l = min_l; l < t; l++) {
                    int lag = t - l;  // lag >= 1
                    if (lag <= max_lag) {
                        double phi_lag = phi[lag - 1];  // phi is 1-indexed in concept
                        double y_kl = Y(l, k);
                        
                        sum_Ry += Rt_m(l, k) * phi_lag * y_kl;
                        sum_y += phi_lag * y_kl;
                    }
                }
                
                conv_Ry(t, k) = sum_Ry;
                conv_y(t, k) = sum_y;
            }
        }
        
        // Now compute R_apparent, R_local, R_import for each (t, s)
        for (int t = 1; t < nt; t++) {
            for (int s = 0; s < ns; s++) {
                
                double denom = conv_y(t, s);
                
                // Local contribution: w_s * conv_Ry(t, s)
                double w_s = alpha_m(s, s);
                double local_num = w_s * conv_Ry(t, s);
                
                // Import contribution: sum_{k != s} alpha_{s,k} * conv_Ry(t, k)
                double import_num = 0.0;
                for (int k = 0; k < ns; k++) {
                    if (k != s) {
                        import_num += alpha_m(s, k) * conv_Ry(t, k);
                    }
                }
                
                // Store results
                int idx = t + s * nt + m * nt * ns;
                
                if (denom > 1e-10) {
                    R_local[idx] = local_num / denom;
                    R_import[idx] = import_num / denom;
                    R_apparent[idx] = (local_num + import_num) / denom;
                } else {
                    R_local[idx] = NA_REAL;
                    R_import[idx] = NA_REAL;
                    R_apparent[idx] = NA_REAL;
                }
            }
        }
        
        // Set t = 0 to NA (no history available)
        for (int s = 0; s < ns; s++) {
            int idx = 0 + s * nt + m * nt * ns;
            R_local[idx] = NA_REAL;
            R_import[idx] = NA_REAL;
            R_apparent[idx] = NA_REAL;
        }
    }

    if (return_quantiles) {
        Rcpp::NumericVector R_apparent_qt(nt * ns * quantiles.size());
        Rcpp::NumericVector R_local_qt(nt * ns * quantiles.size());
        Rcpp::NumericVector R_import_qt(nt * ns * quantiles.size());

        R_apparent_qt.attr("dim") = IntegerVector::create(nt, ns, quantiles.size());
        R_local_qt.attr("dim") = IntegerVector::create(nt, ns, quantiles.size());
        R_import_qt.attr("dim") = IntegerVector::create(nt, ns, quantiles.size());

        // Compute quantiles
        for (int t = 0; t < nt; t++) {
            for (int s = 0; s < ns; s++) {
                // Extract samples for (t, s)
                std::vector<double> R_app_samples(nsample);
                std::vector<double> R_loc_samples(nsample);
                std::vector<double> R_imp_samples(nsample);
                for (int m = 0; m < nsample; m++) {
                    int idx = t + s * nt + m * nt * ns;
                    R_app_samples[m] = R_apparent[idx];
                    R_loc_samples[m] = R_local[idx];
                    R_imp_samples[m] = R_import[idx];
                }

                // Compute quantiles for each requested quantile
                for (int q = 0; q < quantiles.size(); q++) {
                    double q_val = quantiles[q];
                    std::nth_element(R_app_samples.begin(), R_app_samples.begin() + static_cast<int>(q_val * nsample), R_app_samples.end());
                    std::nth_element(R_loc_samples.begin(), R_loc_samples.begin() + static_cast<int>(q_val * nsample), R_loc_samples.end());
                    std::nth_element(R_imp_samples.begin(), R_imp_samples.begin() + static_cast<int>(q_val * nsample), R_imp_samples.end());
                    int qt_idx = t + s * nt + q * nt * ns;
                    R_apparent_qt[qt_idx] = R_app_samples[static_cast<int>(q_val * nsample)];
                    R_local_qt[qt_idx] = R_loc_samples[static_cast<int>(q_val * nsample)];
                    R_import_qt[qt_idx] = R_imp_samples[static_cast<int>(q_val * nsample)];
                }
            }
        }

        return List::create(
            Named("R_apparent") = R_apparent_qt,
            Named("R_local") = R_local_qt,
            Named("R_import") = R_import_qt
        );
    }
    else
    {
        return List::create(
            Named("R_apparent") = R_apparent,
            Named("R_local") = R_local,
            Named("R_import") = R_import
        );
    }
} // compute_apparent_R


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
    const Rcpp::Nullable<Rcpp::List> &spatial_opts = R_NilValue,
    const Rcpp::Nullable<Rcpp::List> &zero_inflation_opts = R_NilValue
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

    Rcpp::List zero_inflation_defaults = Rcpp::List::create(
        Rcpp::Named("inflated") = false,
        Rcpp::Named("beta0") = 0.0,
        Rcpp::Named("beta1") = 0.0
    );

    Rcpp::List lagdist_opts_use = lagdist_opts.isNull() ? lagdist_defaults : Rcpp::as<Rcpp::List>(lagdist_opts);
    Rcpp::List spatial_opts_use = spatial_opts.isNull() ? spatial_defaults : Rcpp::as<Rcpp::List>(spatial_opts);
    Rcpp::List zinfl_opts_use = zero_inflation_opts.isNull() ? zero_inflation_defaults : Rcpp::as<Rcpp::List>(zero_inflation_opts);

    Model model(
        nt, ns, 
        dist_matrix, mobility_matrix, 
        mu, W, fgain, 
        spatial_opts_use, 
        lagdist_opts_use,
        zinfl_opts_use
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
