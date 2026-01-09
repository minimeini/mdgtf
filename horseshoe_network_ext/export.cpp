#include <chrono>
#include <vector>
#include <complex>
#include <numeric>
#include <algorithm>
#include <cmath>

#include <RcppEigen.h>
#include <unsupported/Eigen/FFT>
#include <Eigen/Eigenvalues>

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



//' Compute apparent reproduction number decomposed into local and imported components
//' 
//' @param alpha_array Spatial weight matrix: either ns x ns x nsample (static) 
//'        or ns x ns x nt x nsample (time-varying)
//' @param Rt_array Reproduction numbers: nt x ns x nsample
//' @param Y Observed case counts: nt x ns
//' @param lagdist_opts Optional list of lag distribution parameters
//' @param return_quantiles If TRUE, return quantile summaries instead of all samples
//' @param quantiles Vector of quantiles to compute (default: c(0.025, 0.5, 0.975))
//' @return List with R_apparent, R_local, R_import arrays (nt x ns x nsample or nt x ns x nquantiles)
//' @export
// [[Rcpp::export]]
List compute_apparent_R(
    const NumericVector& alpha_array,  // ns x ns x nsample (static) or ns x ns x nt x nsample (time-varying)
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

    // Get Rt dimensions: nt x ns x nsample
    IntegerVector Rt_dims = Rt_array.attr("dim");
    if (Rt_dims.size() != 3)
    {
        Rcpp::stop("Rt_array must be a 3D array (nt x ns x nsample)");
    }
    int nt = Rt_dims[0];
    int ns = Rt_dims[1];
    int nsample = Rt_dims[2];

    // Parse alpha dimensions to determine if static or time-varying
    IntegerVector alpha_dims = alpha_array.attr("dim");
    bool alpha_time_varying;
    int alpha_nt;

    if (alpha_dims.size() == 3)
    {
        // Static: ns x ns x nsample
        if (alpha_dims[0] != ns || alpha_dims[1] != ns || alpha_dims[2] != nsample)
        {
            Rcpp::stop("Static alpha dimensions must be ns x ns x nsample (got %d x %d x %d, expected %d x %d x %d)",
                       alpha_dims[0], alpha_dims[1], alpha_dims[2], ns, ns, nsample);
        }
        alpha_time_varying = false;
        alpha_nt = 1;
    }
    else if (alpha_dims.size() == 4)
    {
        // Time-varying: ns x ns x nt x nsample
        if (alpha_dims[0] != ns || alpha_dims[1] != ns || alpha_dims[3] != nsample)
        {
            Rcpp::stop("Time-varying alpha dimensions must be ns x ns x nt x nsample");
        }
        alpha_time_varying = true;
        alpha_nt = alpha_dims[2];
        if (alpha_nt != nt)
        {
            Rcpp::stop("Time dimension of alpha (%d) must match Rt (%d)", alpha_nt, nt);
        }
    }
    else
    {
        Rcpp::stop("alpha_array must be 3D (ns x ns x nsample) or 4D (ns x ns x nt x nsample)");
    }
    
    int max_lag = phi.size();
    
    // Output arrays: nt x ns x nsample
    NumericVector R_apparent(nt * ns * nsample);
    NumericVector R_local(nt * ns * nsample);
    NumericVector R_import(nt * ns * nsample);
    
    R_apparent.attr("dim") = IntegerVector::create(nt, ns, nsample);
    R_local.attr("dim") = IntegerVector::create(nt, ns, nsample);
    R_import.attr("dim") = IntegerVector::create(nt, ns, nsample);
    
    // Temporary matrix for alpha at each time point
    Eigen::MatrixXd alpha_t(ns, ns);
    
    // Loop over posterior samples
    for (int m = 0; m < nsample; m++) {
        
        // For static alpha, extract once per sample
        Eigen::MatrixXd alpha_static(ns, ns);
        if (!alpha_time_varying)
        {
            // alpha_array is stored as [s, k, m] in column-major order
            for (int k = 0; k < ns; k++) {
                for (int s = 0; s < ns; s++) {
                    alpha_static(s, k) = alpha_array[s + k * ns + m * ns * ns];
                }
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
            
            // Get alpha for this time point
            if (alpha_time_varying)
            {
                // alpha_array[s, k, t, m] -> index: s + k*ns + t*ns*ns + m*ns*ns*nt
                for (int k = 0; k < ns; k++) {
                    for (int s = 0; s < ns; s++) {
                        int idx = s + k * ns + t * ns * ns + m * ns * ns * nt;
                        alpha_t(s, k) = alpha_array[idx];
                    }
                }
            }
            else
            {
                alpha_t = alpha_static;
            }
            
            for (int s = 0; s < ns; s++) {
                
                double denom = conv_y(t, s);
                
                // Local contribution: w_s(t) * conv_Ry(t, s)
                double w_s = alpha_t(s, s);
                double local_num = w_s * conv_Ry(t, s);
                
                // Import contribution: sum_{k != s} alpha_{s,k}(t) * conv_Ry(t, k)
                double import_num = 0.0;
                for (int k = 0; k < ns; k++) {
                    if (k != s) {
                        import_num += alpha_t(s, k) * conv_Ry(t, k);
                    }
                }
                
                // Store results
                int out_idx = t + s * nt + m * nt * ns;
                
                if (denom > 1e-10) {
                    R_local[out_idx] = local_num / denom;
                    R_import[out_idx] = import_num / denom;
                    R_apparent[out_idx] = (local_num + import_num) / denom;
                } else {
                    R_local[out_idx] = NA_REAL;
                    R_import[out_idx] = NA_REAL;
                    R_apparent[out_idx] = NA_REAL;
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
        int nq = quantiles.size();
        Rcpp::NumericVector R_apparent_qt(nt * ns * nq);
        Rcpp::NumericVector R_local_qt(nt * ns * nq);
        Rcpp::NumericVector R_import_qt(nt * ns * nq);

        R_apparent_qt.attr("dim") = IntegerVector::create(nt, ns, nq);
        R_local_qt.attr("dim") = IntegerVector::create(nt, ns, nq);
        R_import_qt.attr("dim") = IntegerVector::create(nt, ns, nq);

        // Compute quantiles
        std::vector<double> R_app_samples(nsample);
        std::vector<double> R_loc_samples(nsample);
        std::vector<double> R_imp_samples(nsample);

        for (int t = 0; t < nt; t++) {
            for (int s = 0; s < ns; s++) {
                // Extract samples for (t, s)
                for (int m = 0; m < nsample; m++) {
                    int idx = t + s * nt + m * nt * ns;
                    R_app_samples[m] = R_apparent[idx];
                    R_loc_samples[m] = R_local[idx];
                    R_imp_samples[m] = R_import[idx];
                }

                // Sort for proper quantile computation
                std::vector<double> R_app_sorted = R_app_samples;
                std::vector<double> R_loc_sorted = R_loc_samples;
                std::vector<double> R_imp_sorted = R_imp_samples;
                std::sort(R_app_sorted.begin(), R_app_sorted.end());
                std::sort(R_loc_sorted.begin(), R_loc_sorted.end());
                std::sort(R_imp_sorted.begin(), R_imp_sorted.end());

                // Compute each quantile with linear interpolation
                for (int q = 0; q < nq; q++) {
                    double p = quantiles[q];
                    double idx_real = p * (nsample - 1);
                    int idx_low = static_cast<int>(std::floor(idx_real));
                    int idx_high = static_cast<int>(std::ceil(idx_real));
                    
                    int qt_idx = t + s * nt + q * nt * ns;
                    
                    if (idx_low == idx_high || idx_high >= nsample) {
                        R_apparent_qt[qt_idx] = R_app_sorted[idx_low];
                        R_local_qt[qt_idx] = R_loc_sorted[idx_low];
                        R_import_qt[qt_idx] = R_imp_sorted[idx_low];
                    } else {
                        double frac = idx_real - idx_low;
                        R_apparent_qt[qt_idx] = (1.0 - frac) * R_app_sorted[idx_low] + frac * R_app_sorted[idx_high];
                        R_local_qt[qt_idx] = (1.0 - frac) * R_loc_sorted[idx_low] + frac * R_loc_sorted[idx_high];
                        R_import_qt[qt_idx] = (1.0 - frac) * R_imp_sorted[idx_low] + frac * R_imp_sorted[idx_high];
                    }
                }
            }
        }

        // Add quantile names
        Rcpp::CharacterVector q_names(nq);
        for (int q = 0; q < nq; q++) {
            q_names[q] = std::to_string(static_cast<int>(quantiles[q] * 100)) + "%";
        }

        return List::create(
            Named("R_apparent") = R_apparent_qt,
            Named("R_local") = R_local_qt,
            Named("R_import") = R_import_qt,
            Named("quantiles") = quantiles,
            Named("quantile_names") = q_names
        );
    }
    
    return List::create(
        Named("R_apparent") = R_apparent,
        Named("R_local") = R_local,
        Named("R_import") = R_import
    );
} // compute_apparent_R


//' Compute network reproduction number (spectral radius of branching matrix)
//' 
//' @param alpha Spatial weight matrix: either ns x ns x nsample (static) or ns x ns x nt x nsample (time-varying)
//' @param Rt Reproduction numbers: nt x ns x nsample
//' @param quantiles Vector of quantiles to compute (default: c(0.025, 0.5, 0.975))
//' @return List with:
//'   - rho_samples: nt x nsample matrix of spectral radii
//'   - rho_quantiles: nt x nquantiles matrix of quantile summaries
//' @export
// [[Rcpp::export]]
Rcpp::List compute_network_Rt(
    const Rcpp::NumericVector &alpha,
    const Rcpp::NumericVector &Rt,
    Rcpp::Nullable<Rcpp::NumericVector> quantiles_in = R_NilValue
)
{
    // Set up quantiles
    std::vector<double> quantiles;
    if (quantiles_in.isNotNull())
    {
        Rcpp::NumericVector q(quantiles_in);
        quantiles.resize(q.size());
        for (int i = 0; i < q.size(); i++)
        {
            quantiles[i] = q[i];
        }
    }
    else
    {
        quantiles = {0.025, 0.5, 0.975};
    }

    // Parse Rt dimensions: nt x ns x nsample
    Rcpp::IntegerVector Rt_dims = Rt.attr("dim");
    if (Rt_dims.size() != 3)
    {
        Rcpp::stop("Rt must be a 3D array (nt x ns x nsample)");
    }
    const Eigen::Index nt = Rt_dims[0];
    const Eigen::Index ns = Rt_dims[1];
    const Eigen::Index nsample = Rt_dims[2];

    // Parse alpha dimensions
    Rcpp::IntegerVector alpha_dims = alpha.attr("dim");
    bool alpha_time_varying;
    Eigen::Index alpha_nt;

    if (alpha_dims.size() == 3)
    {
        // Static: ns x ns x nsample
        if (alpha_dims[0] != ns || alpha_dims[1] != ns || alpha_dims[2] != nsample)
        {
            Rcpp::stop("Static alpha dimensions must be ns x ns x nsample");
        }
        alpha_time_varying = false;
        alpha_nt = 1;
    }
    else if (alpha_dims.size() == 4)
    {
        // Time-varying: ns x ns x nt x nsample
        if (alpha_dims[0] != ns || alpha_dims[1] != ns || alpha_dims[3] != nsample)
        {
            Rcpp::stop("Time-varying alpha dimensions must be ns x ns x nt x nsample");
        }
        alpha_time_varying = true;
        alpha_nt = alpha_dims[2];
        if (alpha_nt != nt)
        {
            Rcpp::stop("Time dimension of alpha must match Rt");
        }
    }
    else
    {
        Rcpp::stop("alpha must be 3D (static) or 4D (time-varying)");
    }

    // Allocate output matrix for spectral radii
    Eigen::MatrixXd rho_samples(nt, nsample);

    // Temporary matrices for computation
    Eigen::MatrixXd alpha_t(ns, ns);
    Eigen::MatrixXd K_t(ns, ns);
    Eigen::VectorXd Rt_vec(ns);

    // Eigen solver (reuse for efficiency)
    Eigen::EigenSolver<Eigen::MatrixXd> solver;

    // Main computation loop
    for (Eigen::Index m = 0; m < nsample; m++)
    {
        for (Eigen::Index t = 0; t < nt; t++)
        {
            // Extract alpha for this sample (and time if varying)
            if (alpha_time_varying)
            {
                // alpha[s, k, t, m] -> index: s + k*ns + t*ns*ns + m*ns*ns*nt
                for (Eigen::Index k = 0; k < ns; k++)
                {
                    for (Eigen::Index s = 0; s < ns; s++)
                    {
                        Eigen::Index idx = s + k * ns + t * ns * ns + m * ns * ns * nt;
                        alpha_t(s, k) = alpha[idx];
                    }
                }
            }
            else
            {
                // alpha[s, k, m] -> index: s + k*ns + m*ns*ns
                for (Eigen::Index k = 0; k < ns; k++)
                {
                    for (Eigen::Index s = 0; s < ns; s++)
                    {
                        Eigen::Index idx = s + k * ns + m * ns * ns;
                        alpha_t(s, k) = alpha[idx];
                    }
                }
            }

            // Extract Rt for this sample and time
            // Rt[t, k, m] -> index: t + k*nt + m*nt*ns
            for (Eigen::Index k = 0; k < ns; k++)
            {
                Eigen::Index idx = t + k * nt + m * nt * ns;
                Rt_vec(k) = Rt[idx];
            }

            // Compute branching matrix: K_t = alpha_t * diag(Rt)
            // K_t(s, k) = alpha_t(s, k) * Rt(k)
            for (Eigen::Index k = 0; k < ns; k++)
            {
                for (Eigen::Index s = 0; s < ns; s++)
                {
                    K_t(s, k) = alpha_t(s, k) * Rt_vec(k);
                }
            }

            // Compute spectral radius (maximum real part of eigenvalues)
            solver.compute(K_t, /* computeEigenvectors = */ false);
            const auto& eigenvalues = solver.eigenvalues();

            double max_real = eigenvalues(0).real();
            for (Eigen::Index i = 1; i < ns; i++)
            {
                if (eigenvalues(i).real() > max_real)
                {
                    max_real = eigenvalues(i).real();
                }
            }

            rho_samples(t, m) = max_real;
        }
    }

    // Compute quantiles for each time point
    Eigen::Index nq = quantiles.size();
    Eigen::MatrixXd rho_quantiles(nt, nq);
    std::vector<double> sample_vec(nsample);

    for (Eigen::Index t = 0; t < nt; t++)
    {
        // Extract samples for this time point
        for (Eigen::Index m = 0; m < nsample; m++)
        {
            sample_vec[m] = rho_samples(t, m);
        }

        // Sort for quantile computation
        std::sort(sample_vec.begin(), sample_vec.end());

        // Compute each quantile
        for (Eigen::Index q = 0; q < nq; q++)
        {
            double p = quantiles[q];
            double idx_real = p * (nsample - 1);
            Eigen::Index idx_low = static_cast<Eigen::Index>(std::floor(idx_real));
            Eigen::Index idx_high = static_cast<Eigen::Index>(std::ceil(idx_real));
            
            if (idx_low == idx_high || idx_high >= nsample)
            {
                rho_quantiles(t, q) = sample_vec[idx_low];
            }
            else
            {
                // Linear interpolation
                double frac = idx_real - idx_low;
                rho_quantiles(t, q) = (1.0 - frac) * sample_vec[idx_low] + frac * sample_vec[idx_high];
            }
        }
    }

    // Convert to R matrices
    Rcpp::NumericMatrix rho_samples_out = Rcpp::wrap(rho_samples);
    Rcpp::NumericMatrix rho_quantiles_out = Rcpp::wrap(rho_quantiles);

    // Add column names to quantiles
    Rcpp::CharacterVector q_names(nq);
    for (Eigen::Index q = 0; q < nq; q++)
    {
        q_names[q] = std::to_string(static_cast<int>(quantiles[q] * 100)) + "%";
    }
    Rcpp::colnames(rho_quantiles_out) = q_names;

    return Rcpp::List::create(
        Rcpp::Named("rho_samples") = rho_samples_out,
        Rcpp::Named("rho_quantiles") = rho_quantiles_out
    );
} // compute_network_Rt


//' Compute sensitivity of spectral radius to Rt and retention rate
//' 
//' @param alpha Spatial weight matrix at a single time point: ns x ns
//' @param Rt Reproduction numbers at a single time point: ns vector
//' @return List with:
//'   - rho: spectral radius
//'   - d_rho_d_Rt: sensitivity to each Rt (ns vector)
//'   - d_rho_d_w: sensitivity to each retention rate (ns vector)
//'   - left_eigenvec: left eigenvector (ns vector)
//'   - right_eigenvec: right eigenvector (ns vector)
//' @export
// [[Rcpp::export]]
Rcpp::List compute_spectral_sensitivity(
    const Eigen::MatrixXd &alpha,
    const Eigen::VectorXd &Rt
)
{
    const Eigen::Index ns = alpha.rows();
    if (alpha.cols() != ns || Rt.size() != ns)
    {
        Rcpp::stop("Dimension mismatch");
    }

    // Compute branching matrix K = alpha * diag(Rt)
    Eigen::MatrixXd K(ns, ns);
    for (Eigen::Index k = 0; k < ns; k++)
    {
        for (Eigen::Index s = 0; s < ns; s++)
        {
            K(s, k) = alpha(s, k) * Rt(k);
        }
    }

    // Compute eigenvalues and right eigenvectors
    Eigen::EigenSolver<Eigen::MatrixXd> solver(K);
    const auto& eigenvalues = solver.eigenvalues();
    const auto& eigenvectors = solver.eigenvectors();

    // Find dominant eigenvalue (largest real part)
    Eigen::Index dominant_idx = 0;
    double max_real = eigenvalues(0).real();
    for (Eigen::Index i = 1; i < ns; i++)
    {
        if (eigenvalues(i).real() > max_real)
        {
            max_real = eigenvalues(i).real();
            dominant_idx = i;
        }
    }

    double rho = max_real;

    // Right eigenvector (normalize to sum to 1)
    Eigen::VectorXd v(ns);
    for (Eigen::Index i = 0; i < ns; i++)
    {
        v(i) = eigenvectors(i, dominant_idx).real();
    }
    double v_sum = v.sum();
    if (std::abs(v_sum) > 1e-10)
    {
        v /= v_sum;
    }

    // Left eigenvector (from K^T)
    Eigen::EigenSolver<Eigen::MatrixXd> solver_T(K.transpose());
    const auto& eigenvalues_T = solver_T.eigenvalues();
    const auto& eigenvectors_T = solver_T.eigenvectors();

    // Find matching eigenvalue in transpose
    Eigen::Index dominant_idx_T = 0;
    double min_diff = std::abs(eigenvalues_T(0).real() - rho);
    for (Eigen::Index i = 1; i < ns; i++)
    {
        double diff = std::abs(eigenvalues_T(i).real() - rho);
        if (diff < min_diff)
        {
            min_diff = diff;
            dominant_idx_T = i;
        }
    }

    Eigen::VectorXd u(ns);
    for (Eigen::Index i = 0; i < ns; i++)
    {
        u(i) = eigenvectors_T(i, dominant_idx_T).real();
    }

    // Normalize so that u^T v = 1
    double uTv = u.dot(v);
    if (std::abs(uTv) > 1e-10)
    {
        u /= uTv;
    }

    // Compute sensitivities
    // d_rho/d_Rt_k = (v_k * sum_s u_s * alpha_{s,k}) / (u^T v)
    // Since we normalized u^T v = 1, denominator is 1
    Eigen::VectorXd d_rho_d_Rt(ns);
    for (Eigen::Index k = 0; k < ns; k++)
    {
        double sum_u_alpha = 0.0;
        for (Eigen::Index s = 0; s < ns; s++)
        {
            sum_u_alpha += u(s) * alpha(s, k);
        }
        d_rho_d_Rt(k) = v(k) * sum_u_alpha;
    }

    // d_rho/d_w_k = (R_k * v_k) * [u_k - u_bar_{-k}]
    // where u_bar_{-k} = sum_{s != k} u_s * alpha_tilde_{s,k}
    // and alpha_tilde_{s,k} = alpha_{s,k} / (1 - w_k) for s != k
    Eigen::VectorXd d_rho_d_w(ns);
    for (Eigen::Index k = 0; k < ns; k++)
    {
        double w_k = alpha(k, k);  // Diagonal is retention rate
        double one_minus_w = 1.0 - w_k;

        double u_bar_minus_k = 0.0;
        if (std::abs(one_minus_w) > 1e-10)
        {
            for (Eigen::Index s = 0; s < ns; s++)
            {
                if (s != k)
                {
                    double alpha_tilde_sk = alpha(s, k) / one_minus_w;
                    u_bar_minus_k += u(s) * alpha_tilde_sk;
                }
            }
        }

        d_rho_d_w(k) = Rt(k) * v(k) * (u(k) - u_bar_minus_k);
    }

    return Rcpp::List::create(
        Rcpp::Named("rho") = rho,
        Rcpp::Named("d_rho_d_Rt") = d_rho_d_Rt,
        Rcpp::Named("d_rho_d_w") = d_rho_d_w,
        Rcpp::Named("left_eigenvec") = u,
        Rcpp::Named("right_eigenvec") = v
    );
} // compute_spectral_sensitivity


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
    const Rcpp::Nullable<Rcpp::NumericVector> &X_in = R_NilValue,
    const Rcpp::Nullable<Rcpp::NumericVector> &mean_slopes_in = R_NilValue,
    const Rcpp::Nullable<Rcpp::NumericVector> &sd_slopes_in = R_NilValue,
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
        X_in, mean_slopes_in, sd_slopes_in,
        mu, W, fgain, 
        spatial_opts_use, 
        lagdist_opts_use,
        zinfl_opts_use
    );

    return model.simulate();
} // simulate_network_hawkes


//' @export
// [[Rcpp::export]]
Rcpp::List continue_simulation(
    const Rcpp::List &simulation_output,  // Output from simulate_network_hawkes
    const Eigen::Index &k_ahead,
    const Rcpp::Nullable<Rcpp::NumericVector> &X_history = R_NilValue,
    const Rcpp::Nullable<Rcpp::NumericVector> &X_forecast = R_NilValue,
    const Rcpp::Nullable<Rcpp::NumericMatrix> &dist_matrix = R_NilValue,
    const Rcpp::Nullable<Rcpp::NumericMatrix> &mobility_matrix = R_NilValue,
    const double &c_sq = 4.0,
    const std::string &fgain = "softplus"
)
{
    // Extract history from simulation output
    Eigen::MatrixXd Y_history = Rcpp::as<Eigen::MatrixXd>(simulation_output["Y"]);
    Eigen::MatrixXd wt_history = Rcpp::as<Eigen::MatrixXd>(simulation_output["wt"]);
    
    const Eigen::Index T_obs = Y_history.rows();
    const Eigen::Index ns = Y_history.cols();
    
    // Reconstruct model with same parameters
    Rcpp::List params = simulation_output["params"];
    double mu = Rcpp::as<double>(params["mu"]);
    double W = Rcpp::as<double>(params["W"]);
    
    Rcpp::List hs_list = simulation_output["horseshoe"];
    Eigen::MatrixXd theta = Rcpp::as<Eigen::MatrixXd>(hs_list["theta"]);
    Eigen::VectorXd logit_wdiag_intercept = Rcpp::as<Eigen::VectorXd>(hs_list["logit_wdiag_intercept"]);
    
    // Build model and set parameters
    Rcpp::List lagdist_defaults = Rcpp::List::create(
        Rcpp::Named("name") = "lognorm",
        Rcpp::Named("par1") = LN_MU,
        Rcpp::Named("par2") = LN_SD2,
        Rcpp::Named("truncated") = true,
        Rcpp::Named("rescaled") = true
    );
    
    
    Model model(
        Y_history, dist_matrix, mobility_matrix, X_history,
        c_sq, fgain, false, lagdist_defaults
    );
    
    // Set inferred/true parameters
    model.mu = mu;
    model.temporal.W = W;
    model.temporal.wt = wt_history;
    model.spatial.theta = theta;
    model.spatial.logit_wdiag_intercept = logit_wdiag_intercept;
    
    if (hs_list.containsElementNamed("logit_wdiag_slope"))
    {
        model.spatial.logit_wdiag_slope = Rcpp::as<Eigen::VectorXd>(hs_list["logit_wdiag_slope"]);
    }
    if (hs_list.containsElementNamed("rho_dist"))
    {
        model.spatial.rho_dist = Rcpp::as<double>(hs_list["rho_dist"]);
    }
    if (hs_list.containsElementNamed("rho_mobility"))
    {
        model.spatial.rho_mobility = Rcpp::as<double>(hs_list["rho_mobility"]);
    }
    
    // Continue simulation
    return model.simulate_ahead(Y_history, wt_history, k_ahead, X_forecast);
} // continue_simulation


//' @export
// [[Rcpp::export]]
Rcpp::List infer_network_hawkes(
    const Rcpp::NumericVector &Y_in,               // (nt + 1) x ns, observed primary infections or a vector of (nt + 1) length for a single series
    const Rcpp::Nullable<Rcpp::NumericMatrix> &dist_matrix = R_NilValue,     // ns x ns, pairwise distance matrix
    const Rcpp::Nullable<Rcpp::NumericMatrix> &mobility_matrix = R_NilValue, // ns x ns, pairwise mobility matrix
    const Rcpp::Nullable<Rcpp::NumericVector> &X_in = R_NilValue,
    const double &c_sq = 4.0,
    const std::string &fgain = "softplus",
    const Rcpp::Nullable<Rcpp::List> &lagdist_opts = R_NilValue,
    const unsigned int &nburnin = 1000,
    const unsigned int &nsamples = 1000,
    const unsigned int &nthin = 1,
    const Rcpp::Nullable<Rcpp::List> &mcmc_opts = R_NilValue
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

    Model model(Y, dist_matrix, mobility_matrix, X_in, c_sq, fgain, lagdist_opts_use);

    auto start = std::chrono::high_resolution_clock::now();
    Rcpp::List output = model.run_mcmc(
        Y, nburnin, nsamples, nthin, mcmc_opts
    );
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
    Rcpp::Rcout << "\nElapsed time: " << duration.count() << " milliseconds" << std::endl;
    output["elapsed_time_ms"] = duration.count();
    return output;
} // infer_network_hawkes


void parse_mcmc_output(
    Eigen::Tensor<double, 3> &wt_samples,
    Eigen::Tensor<double, 3> &theta_samples,
    Eigen::MatrixXd &logit_wdiag_intercept_samples,
    Eigen::MatrixXd &logit_wdiag_slopes_samples,
    Eigen::VectorXd &mu_samples,
    Eigen::VectorXd &W_samples,
    Eigen::VectorXd &rho_dist_samples,
    Eigen::VectorXd &rho_mobility_samples,
    const Rcpp::List &mcmc_output, 
    const Rcpp::List &true_vals, 
    const Eigen::Index &ns, 
    const Eigen::Index &nt, 
    const Eigen::Index &nsample,
    const bool &use_distance,
    const bool &use_mobility,
    const bool &use_covariates
)
{
    if (mcmc_output.containsElementNamed("wt"))
    {
        Rcpp::NumericVector wt_array = mcmc_output["wt"];
        Rcpp::IntegerVector dim = wt_array.attr("dim");
        if (nsample != dim[2])
        {
            Rcpp::warning("Requested nsample does not match number of samples in MCMC output. Using available samples.");
        }
        wt_samples = r_to_tensor3(wt_array);
    }
    else if (true_vals.containsElementNamed("wt"))
    {
        wt_samples.resize(nt + 1, ns, nsample);
        Rcpp::NumericMatrix wt_mat = true_vals["wt"]; // (nt + 1) x ns
        for (Eigen::Index i = 0; i < nsample; i++)
        {
            for (Eigen::Index s = 0; s < ns; s++)
            {
                for (Eigen::Index t = 0; t < nt + 1; t++)
                {
                    wt_samples(t, s, i) = wt_mat(t, s);
                }
            }
        }
    }
    else
    {
        throw std::invalid_argument("No wt samples in MCMC output.");
    }


    bool found_theta = false;
    if (mcmc_output.containsElementNamed("horseshoe"))
    {
        Rcpp::List hs_list = mcmc_output["horseshoe"];
        if (hs_list.containsElementNamed("theta"))
        {
            Rcpp::NumericVector theta_array = Rcpp::as<Rcpp::NumericVector>(hs_list["theta"]);
            Rcpp::IntegerVector dim = theta_array.attr("dim");
            if (dim[2] != nsample)
            {
                throw std::invalid_argument("Number of theta samples does not match number of wt samples.");
            }
            theta_samples = r_to_tensor3(theta_array);
            found_theta = true;
        }
    }

    if (!found_theta && true_vals.containsElementNamed("horseshoe"))
    {
        Rcpp::List hs_list = true_vals["horseshoe"];
        if (hs_list.containsElementNamed("theta"))
        {
            Rcpp::NumericMatrix theta_mat = Rcpp::as<Rcpp::NumericMatrix>(hs_list["theta"]); // ns x ns
            if (theta_mat.nrow() != theta_mat.ncol())
            {
                throw std::invalid_argument("True theta must be a square matrix.");
            }
            if (theta_mat.nrow() != ns)
            {
                throw std::invalid_argument("Dimension of true theta does not match number of locations in Y_obs.");
            }
            theta_samples.resize(ns, ns, nsample);
            for (Eigen::Index i = 0; i < nsample; i++)
            {
                for (Eigen::Index s = 0; s < ns; s++)
                {
                    for (Eigen::Index k = 0; k < ns; k++)
                    {
                        theta_samples(s, k, i) = theta_mat(s, k);
                    }
                }
            }
            found_theta = true;
        }
    }

    if (!found_theta)
    {
        throw std::invalid_argument("No theta samples in MCMC output or true values.");
    }

    bool found_wdiag_intercept = false;
    if (mcmc_output.containsElementNamed("horseshoe"))
    {
        Rcpp::List hs_list = mcmc_output["horseshoe"];
        if (hs_list.containsElementNamed("logit_wdiag_intercept"))
        {
            Rcpp::NumericMatrix intercept_mat = Rcpp::as<Rcpp::NumericMatrix>(hs_list["logit_wdiag_intercept"]); // ns x nsample
            if (intercept_mat.ncol() != nsample)
            {
                throw std::invalid_argument("Number of logit_wdiag_intercept samples does not match number of wt samples.");
            }
            logit_wdiag_intercept_samples = Rcpp::as<Eigen::MatrixXd>(intercept_mat);
            found_wdiag_intercept = true;
        }
    }

    if (!found_wdiag_intercept && true_vals.containsElementNamed("horseshoe"))
    {
        Rcpp::List hs_list = true_vals["horseshoe"];
        if (hs_list.containsElementNamed("logit_wdiag_intercept"))
        {
            Rcpp::NumericVector intercept_vec = Rcpp::as<Rcpp::NumericVector>(hs_list["logit_wdiag_intercept"]); // ns
            if (intercept_vec.size() != ns)
            {
                throw std::invalid_argument("Dimension of true logit_wdiag_intercept does not match number of locations in Y_obs.");
            }
            logit_wdiag_intercept_samples.resize(ns, nsample);
            for (Eigen::Index i = 0; i < nsample; i++)
            {
                for (Eigen::Index s = 0; s < ns; s++)
                {
                    logit_wdiag_intercept_samples(s, i) = intercept_vec[s];
                }
            }
            found_wdiag_intercept = true;
        }
    }

    if (!found_wdiag_intercept)
    {
        throw std::invalid_argument("No logit_wdiag_intercept samples in MCMC output or true values.");
    }

    bool found_slopes = false;
    if (use_covariates)
    {
        if (mcmc_output.containsElementNamed("horseshoe"))
        {
            Rcpp::List hs_list = mcmc_output["horseshoe"];
            if (hs_list.containsElementNamed("logit_wdiag_slope"))
            {
                Rcpp::NumericMatrix slopes_mat = Rcpp::as<Rcpp::NumericMatrix>(hs_list["logit_wdiag_slope"]); // ns x n_covariates x nsample
                if (slopes_mat.ncol() != nsample)
                {
                    throw std::invalid_argument("Number of logit_wdiag_slope samples does not match number of wt samples.");
                }
                logit_wdiag_slopes_samples = Rcpp::as<Eigen::MatrixXd>(slopes_mat);
                found_slopes = true;
            }
        }
    }

    if (use_covariates && !found_slopes && true_vals.containsElementNamed("horseshoe"))
    {
        Rcpp::List hs_list = true_vals["horseshoe"];
        if (hs_list.containsElementNamed("logit_wdiag_slope"))
        {
            Rcpp::NumericVector slopes_vec = Rcpp::as<Rcpp::NumericVector>(hs_list["logit_wdiag_slope"]); // n_covariates x 1
            Eigen::Index n_covariates = slopes_vec.size();
            logit_wdiag_slopes_samples.resize(n_covariates, nsample);
            for (Eigen::Index i = 0; i < nsample; i++)
            {
                for (Eigen::Index j = 0; j < n_covariates; j++)
                {
                    logit_wdiag_slopes_samples(j, i) = slopes_vec[j];
                }
            }
            found_slopes = true;
        }
    }

    if (use_covariates && !found_slopes)
    {
        throw std::invalid_argument("No logit_wdiag_slope samples in MCMC output or true values.");
    }

    bool found_mu = false;
    if (mcmc_output.containsElementNamed("params"))
    {
        Rcpp::List params_list = mcmc_output["params"];
        if (params_list.containsElementNamed("mu"))
        {
            Rcpp::NumericVector mu_vec = Rcpp::as<Rcpp::NumericVector>(params_list["mu"]); // nsample
            if (mu_vec.size() != nsample)
            {
                throw std::invalid_argument("Number of mu samples does not match number of wt samples.");
            }
            mu_samples = Rcpp::as<Eigen::VectorXd>(mu_vec);
            found_mu = true;
        }
    }

    if (!found_mu && true_vals.containsElementNamed("params"))
    {
        Rcpp::List params_list = true_vals["params"];
        if (params_list.containsElementNamed("mu"))
        {
            double mu_true = Rcpp::as<double>(params_list["mu"]);
            mu_samples.resize(nsample);
            mu_samples.fill(mu_true);
            found_mu = true;
        }
    }

    if (!found_mu)
    {
        throw std::invalid_argument("No mu samples in MCMC output or true values.");
    }

    bool found_W = false;
    if (mcmc_output.containsElementNamed("params"))
    {
        Rcpp::List params_list = mcmc_output["params"];
        if (params_list.containsElementNamed("W"))
        {
            Rcpp::NumericVector W_vec = Rcpp::as<Rcpp::NumericVector>(params_list["W"]); // nsample
            if (W_vec.size() != nsample)
            {
                throw std::invalid_argument("Number of W samples does not match number of wt samples.");
            }
            W_samples = Rcpp::as<Eigen::VectorXd>(W_vec);
            found_W = true;
        }
    }

    if (!found_W && true_vals.containsElementNamed("params"))
    {
        Rcpp::List params_list = true_vals["params"];
        if (params_list.containsElementNamed("W"))
        {
            double W_true = Rcpp::as<double>(params_list["W"]);
            W_samples.resize(nsample);
            W_samples.fill(W_true);
            found_W = true;
        }
    }

    if (!found_W)
    {
        throw std::invalid_argument("No W samples in MCMC output or true values.");
    }

    bool found_rho_dist = false;
    if (use_distance)
    {
        if (mcmc_output.containsElementNamed("params"))
        {
            Rcpp::List params_list = mcmc_output["params"];
            if (params_list.containsElementNamed("rho_dist"))
            {
                Rcpp::NumericVector rho_dist_vec = Rcpp::as<Rcpp::NumericVector>(params_list["rho_dist"]); // nsample
                if (rho_dist_vec.size() != nsample)
                {
                    throw std::invalid_argument("Number of rho_dist samples does not match number of wt samples.");
                }
                rho_dist_samples = Rcpp::as<Eigen::VectorXd>(rho_dist_vec);
                found_rho_dist = true;
            }
        }
    }

    if (use_distance && !found_rho_dist && true_vals.containsElementNamed("horseshoe"))
    {
        Rcpp::List hs_list = true_vals["horseshoe"];
        if (hs_list.containsElementNamed("rho_dist"))
        {
            double rho_dist_true = Rcpp::as<double>(hs_list["rho_dist"]);
            rho_dist_samples.resize(nsample);
            rho_dist_samples.fill(rho_dist_true);
            found_rho_dist = true;
        }
    }

    if (use_distance && !found_rho_dist)
    {
        throw std::invalid_argument("No rho_dist samples in MCMC output or true values.");
    }

    bool found_rho_mobility = false;
    if (use_mobility)
    {
        if (mcmc_output.containsElementNamed("params"))
        {
            Rcpp::List params_list = mcmc_output["params"];
            if (params_list.containsElementNamed("rho_mobility"))
            {
                Rcpp::NumericVector rho_mobility_vec = Rcpp::as<Rcpp::NumericVector>(params_list["rho_mobility"]); // nsample
                if (rho_mobility_vec.size() != nsample)
                {
                    throw std::invalid_argument("Number of rho_mobility samples does not match number of wt samples.");
                }
                rho_mobility_samples = Rcpp::as<Eigen::VectorXd>(rho_mobility_vec);
                found_rho_mobility = true;
            }
        }
    }

    if (use_mobility && !found_rho_mobility && true_vals.containsElementNamed("horseshoe"))
    {
        Rcpp::List hs_list = true_vals["horseshoe"];
        if (hs_list.containsElementNamed("rho_mobility"))
        {
            double rho_mobility_true = Rcpp::as<double>(hs_list["rho_mobility"]);
            rho_mobility_samples.resize(nsample);
            rho_mobility_samples.fill(rho_mobility_true);
            found_rho_mobility = true;
        }
    }

    if (use_mobility && !found_rho_mobility)
    {
        throw std::invalid_argument("No rho_mobility samples in MCMC output or true values.");
    }

    return;
} // parse_mcmc_output


void parse_mcmc_output_for_alpha(
    Eigen::Tensor<double, 3> &theta_samples,
    Eigen::MatrixXd &logit_wdiag_intercept_samples,
    Eigen::MatrixXd &logit_wdiag_slopes_samples,
    Eigen::VectorXd &rho_dist_samples,
    Eigen::VectorXd &rho_mobility_samples,
    const Rcpp::List &mcmc_output, 
    const Rcpp::List &true_vals, 
    const Eigen::Index &ns, 
    const Eigen::Index &nsample,
    const bool &use_distance,
    const bool &use_mobility,
    const bool &use_covariates
)
{
    bool found_theta = false;
    if (mcmc_output.containsElementNamed("horseshoe"))
    {
        Rcpp::List hs_list = mcmc_output["horseshoe"];
        if (hs_list.containsElementNamed("theta"))
        {
            Rcpp::NumericVector theta_array = Rcpp::as<Rcpp::NumericVector>(hs_list["theta"]);
            Rcpp::IntegerVector dim = theta_array.attr("dim");
            if (dim[2] != nsample)
            {
                throw std::invalid_argument("Number of theta samples does not match number of wt samples.");
            }
            theta_samples = r_to_tensor3(theta_array);
            found_theta = true;
        }
    }

    if (!found_theta && true_vals.containsElementNamed("horseshoe"))
    {
        Rcpp::List hs_list = true_vals["horseshoe"];
        if (hs_list.containsElementNamed("theta"))
        {
            Rcpp::NumericMatrix theta_mat = Rcpp::as<Rcpp::NumericMatrix>(hs_list["theta"]); // ns x ns
            if (theta_mat.nrow() != theta_mat.ncol())
            {
                throw std::invalid_argument("True theta must be a square matrix.");
            }
            if (theta_mat.nrow() != ns)
            {
                throw std::invalid_argument("Dimension of true theta does not match number of locations in Y_obs.");
            }
            theta_samples.resize(ns, ns, nsample);
            for (Eigen::Index i = 0; i < nsample; i++)
            {
                for (Eigen::Index s = 0; s < ns; s++)
                {
                    for (Eigen::Index k = 0; k < ns; k++)
                    {
                        theta_samples(s, k, i) = theta_mat(s, k);
                    }
                }
            }
            found_theta = true;
        }
    }

    if (!found_theta)
    {
        throw std::invalid_argument("No theta samples in MCMC output or true values.");
    }

    bool found_wdiag_intercept = false;
    if (mcmc_output.containsElementNamed("horseshoe"))
    {
        Rcpp::List hs_list = mcmc_output["horseshoe"];
        if (hs_list.containsElementNamed("logit_wdiag_intercept"))
        {
            Rcpp::NumericMatrix intercept_mat = Rcpp::as<Rcpp::NumericMatrix>(hs_list["logit_wdiag_intercept"]); // ns x nsample
            if (intercept_mat.ncol() != nsample)
            {
                throw std::invalid_argument("Number of logit_wdiag_intercept samples does not match number of wt samples.");
            }
            logit_wdiag_intercept_samples = Rcpp::as<Eigen::MatrixXd>(intercept_mat);
            found_wdiag_intercept = true;
        }
    }

    if (!found_wdiag_intercept && true_vals.containsElementNamed("horseshoe"))
    {
        Rcpp::List hs_list = true_vals["horseshoe"];
        if (hs_list.containsElementNamed("logit_wdiag_intercept"))
        {
            Rcpp::NumericVector intercept_vec = Rcpp::as<Rcpp::NumericVector>(hs_list["logit_wdiag_intercept"]); // ns
            if (intercept_vec.size() != ns)
            {
                throw std::invalid_argument("Dimension of true logit_wdiag_intercept does not match number of locations in Y_obs.");
            }
            logit_wdiag_intercept_samples.resize(ns, nsample);
            for (Eigen::Index i = 0; i < nsample; i++)
            {
                for (Eigen::Index s = 0; s < ns; s++)
                {
                    logit_wdiag_intercept_samples(s, i) = intercept_vec[s];
                }
            }
            found_wdiag_intercept = true;
        }
    }

    if (!found_wdiag_intercept)
    {
        throw std::invalid_argument("No logit_wdiag_intercept samples in MCMC output or true values.");
    }

    bool found_slopes = false;
    if (use_covariates)
    {
        if (mcmc_output.containsElementNamed("horseshoe"))
        {
            Rcpp::List hs_list = mcmc_output["horseshoe"];
            if (hs_list.containsElementNamed("logit_wdiag_slope"))
            {
                Rcpp::NumericMatrix slopes_mat = Rcpp::as<Rcpp::NumericMatrix>(hs_list["logit_wdiag_slope"]); // ns x n_covariates x nsample
                if (slopes_mat.ncol() != nsample)
                {
                    throw std::invalid_argument("Number of logit_wdiag_slope samples does not match number of wt samples.");
                }
                logit_wdiag_slopes_samples = Rcpp::as<Eigen::MatrixXd>(slopes_mat);
                found_slopes = true;
            }
        }
    }

    if (use_covariates && !found_slopes && true_vals.containsElementNamed("horseshoe"))
    {
        Rcpp::List hs_list = true_vals["horseshoe"];
        if (hs_list.containsElementNamed("logit_wdiag_slope"))
        {
            Rcpp::NumericVector slopes_vec = Rcpp::as<Rcpp::NumericVector>(hs_list["logit_wdiag_slope"]); // n_covariates x 1
            Eigen::Index n_covariates = slopes_vec.size();
            logit_wdiag_slopes_samples.resize(n_covariates, nsample);
            for (Eigen::Index i = 0; i < nsample; i++)
            {
                for (Eigen::Index j = 0; j < n_covariates; j++)
                {
                    logit_wdiag_slopes_samples(j, i) = slopes_vec[j];
                }
            }
            found_slopes = true;
        }
    }

    if (use_covariates && !found_slopes)
    {
        throw std::invalid_argument("No logit_wdiag_slope samples in MCMC output or true values.");
    }

    bool found_rho_dist = false;
    if (use_distance)
    {
        if (mcmc_output.containsElementNamed("params"))
        {
            Rcpp::List params_list = mcmc_output["params"];
            if (params_list.containsElementNamed("rho_dist"))
            {
                Rcpp::NumericVector rho_dist_vec = Rcpp::as<Rcpp::NumericVector>(params_list["rho_dist"]); // nsample
                if (rho_dist_vec.size() != nsample)
                {
                    throw std::invalid_argument("Number of rho_dist samples does not match number of wt samples.");
                }
                rho_dist_samples = Rcpp::as<Eigen::VectorXd>(rho_dist_vec);
                found_rho_dist = true;
            }
        }
    }

    if (use_distance && !found_rho_dist && true_vals.containsElementNamed("horseshoe"))
    {
        Rcpp::List hs_list = true_vals["horseshoe"];
        if (hs_list.containsElementNamed("rho_dist"))
        {
            double rho_dist_true = Rcpp::as<double>(hs_list["rho_dist"]);
            rho_dist_samples.resize(nsample);
            rho_dist_samples.fill(rho_dist_true);
            found_rho_dist = true;
        }
    }

    if (use_distance && !found_rho_dist)
    {
        throw std::invalid_argument("No rho_dist samples in MCMC output or true values.");
    }

    bool found_rho_mobility = false;
    if (use_mobility)
    {
        if (mcmc_output.containsElementNamed("params"))
        {
            Rcpp::List params_list = mcmc_output["params"];
            if (params_list.containsElementNamed("rho_mobility"))
            {
                Rcpp::NumericVector rho_mobility_vec = Rcpp::as<Rcpp::NumericVector>(params_list["rho_mobility"]); // nsample
                if (rho_mobility_vec.size() != nsample)
                {
                    throw std::invalid_argument("Number of rho_mobility samples does not match number of wt samples.");
                }
                rho_mobility_samples = Rcpp::as<Eigen::VectorXd>(rho_mobility_vec);
                found_rho_mobility = true;
            }
        }
    }

    if (use_mobility && !found_rho_mobility && true_vals.containsElementNamed("horseshoe"))
    {
        Rcpp::List hs_list = true_vals["horseshoe"];
        if (hs_list.containsElementNamed("rho_mobility"))
        {
            double rho_mobility_true = Rcpp::as<double>(hs_list["rho_mobility"]);
            rho_mobility_samples.resize(nsample);
            rho_mobility_samples.fill(rho_mobility_true);
            found_rho_mobility = true;
        }
    }

    if (use_mobility && !found_rho_mobility)
    {
        throw std::invalid_argument("No rho_mobility samples in MCMC output or true values.");
    }

    return;
} // parse_mcmc_output_for_alpha



//' Calculate spatial weights alpha from MCMC output
//' 
//' @param mcmc_output List containing MCMC posterior samples
//' @param true_vals List containing true values (used if MCMC samples not available)
//' @param ns Number of spatial locations
//' @param nsample Number of posterior samples
//' @param dist_matrix Optional distance matrix (ns x ns)
//' @param mobility_matrix Optional mobility matrix (ns x ns)
//' @param X_in Optional covariate array (np x ns) or (np x ns x nt+1)
//' @return NumericVector with dimensions:
//'   - If static (no X or X with 1 time point): ns x ns x nsample
//'   - If time-varying (X with nt+1 time points): ns x ns x (nt+1) x nsample
//' @export
// [[Rcpp::export]]
Rcpp::NumericVector calculate_alpha(
    const Rcpp::List &mcmc_output,
    const Rcpp::List &true_vals,
    const Eigen::Index &ns,
    const Eigen::Index &nsample,
    const Rcpp::Nullable<Rcpp::NumericMatrix> &dist_matrix = R_NilValue,
    const Rcpp::Nullable<Rcpp::NumericMatrix> &mobility_matrix = R_NilValue,
    const Rcpp::Nullable<Rcpp::NumericVector> &X_in = R_NilValue
)
{
    const bool use_covariates = X_in.isNotNull();
    const bool use_distance = dist_matrix.isNotNull();
    const bool use_mobility = mobility_matrix.isNotNull();

    // Parse MCMC output
    Eigen::Tensor<double, 3> theta_samples; // ns x ns x nsample
    Eigen::MatrixXd logit_wdiag_intercept_samples; // ns x nsample
    Eigen::MatrixXd logit_wdiag_slopes_samples; // np x nsample
    Eigen::VectorXd rho_dist_samples; // nsample
    Eigen::VectorXd rho_mobility_samples; // nsample

    parse_mcmc_output_for_alpha(
        theta_samples,
        logit_wdiag_intercept_samples,
        logit_wdiag_slopes_samples,
        rho_dist_samples,
        rho_mobility_samples,
        mcmc_output,
        true_vals,
        ns,
        nsample,
        use_distance,
        use_mobility,
        use_covariates
    );

    // Set up distance matrix (centered per column)
    Eigen::MatrixXd dist_centered(ns, ns);
    dist_centered.setZero();
    if (use_distance)
    {
        Rcpp::NumericMatrix dist_mat(dist_matrix);
        Eigen::MatrixXd dist_raw = Rcpp::as<Eigen::MatrixXd>(dist_mat);
        // Center per column (destination)
        for (Eigen::Index k = 0; k < ns; k++)
        {
            double col_mean = 0.0;
            for (Eigen::Index s = 0; s < ns; s++)
            {
                if (s != k) col_mean += dist_raw(s, k);
            }
            col_mean /= (ns - 1);
            for (Eigen::Index s = 0; s < ns; s++)
            {
                dist_centered(s, k) = dist_raw(s, k) - col_mean;
            }
        }
    }

    // Set up log mobility matrix (centered per column)
    Eigen::MatrixXd log_mobility_centered(ns, ns);
    log_mobility_centered.setZero();
    if (use_mobility)
    {
        Rcpp::NumericMatrix mob_mat(mobility_matrix);
        Eigen::MatrixXd mob_raw = Rcpp::as<Eigen::MatrixXd>(mob_mat);
        Eigen::MatrixXd log_mob = mob_raw.array().log().matrix();
        // Center per column (destination)
        for (Eigen::Index k = 0; k < ns; k++)
        {
            double col_mean = 0.0;
            for (Eigen::Index s = 0; s < ns; s++)
            {
                if (s != k) col_mean += log_mob(s, k);
            }
            col_mean /= (ns - 1);
            for (Eigen::Index s = 0; s < ns; s++)
            {
                log_mobility_centered(s, k) = log_mob(s, k) - col_mean;
            }
        }
    }

    // Determine time dimension from covariates
    Eigen::Index nt_plus_1 = 1;  // Default: static (single time point)
    Eigen::Tensor<double, 3> X;
    Eigen::Index np = 0;

    if (use_covariates)
    {
        Rcpp::NumericVector X_vec(X_in);
        if (X_vec.hasAttribute("dim"))
        {
            Rcpp::IntegerVector dims = X_vec.attr("dim");
            np = dims[0];
            
            if (dims.size() == 2)
            {
                // np x ns -> static
                nt_plus_1 = 1;
                X.resize(np, ns, 1);
                Rcpp::NumericMatrix X_mat = Rcpp::as<Rcpp::NumericMatrix>(X_vec);
                for (Eigen::Index p = 0; p < np; p++)
                {
                    for (Eigen::Index k = 0; k < ns; k++)
                    {
                        X(p, k, 0) = X_mat(p, k);
                    }
                }
            }
            else if (dims.size() == 3)
            {
                // np x ns x nt_plus_1
                nt_plus_1 = dims[2];
                X = r_to_tensor3(X_vec);
            }
            else
            {
                Rcpp::stop("X_in must be 2D (np x ns) or 3D (np x ns x nt+1)");
            }
        }
        else
        {
            Rcpp::stop("X_in must have dim attribute");
        }
    }

    // Determine output dimensions
    bool time_varying = (nt_plus_1 > 1);

    // Allocate output array
    Rcpp::NumericVector alpha_out;
    if (time_varying)
    {
        // ns x ns x (nt+1) x nsample
        alpha_out = Rcpp::NumericVector(ns * ns * nt_plus_1 * nsample);
        alpha_out.attr("dim") = Rcpp::IntegerVector::create(ns, ns, nt_plus_1, nsample);
    }
    else
    {
        // ns x ns x nsample
        alpha_out = Rcpp::NumericVector(ns * ns * nsample);
        alpha_out.attr("dim") = Rcpp::IntegerVector::create(ns, ns, nsample);
    }

    // Compute alpha for each posterior sample
    for (Eigen::Index m = 0; m < nsample; m++)
    {
        // Extract parameters for this sample
        double rho_dist_m = use_distance ? rho_dist_samples(m) : 0.0;
        double rho_mobility_m = use_mobility ? rho_mobility_samples(m) : 0.0;

        // For each destination k, compute unnormalized weights u_{s,k}
        // v_{s,k} = rho_mobility * log_mobility_{s,k} - rho_dist * dist_{s,k} + (theta_{s,k} - theta_bar_k)
        // u_{s,k} = exp(v_{s,k})

        Eigen::MatrixXd u_mat(ns, ns);
        Eigen::VectorXd U_k(ns);  // Sum of off-diagonal u for each column k

        for (Eigen::Index k = 0; k < ns; k++)
        {
            // Compute theta_bar_k = mean of off-diagonal theta in column k
            double theta_bar_k = 0.0;
            for (Eigen::Index s = 0; s < ns; s++)
            {
                if (s != k)
                {
                    theta_bar_k += theta_samples(s, k, m);
                }
            }
            theta_bar_k /= (ns - 1);

            // Compute u_{s,k} for all s
            double U_k_sum = 0.0;
            for (Eigen::Index s = 0; s < ns; s++)
            {
                if (s != k)
                {
                    double theta_tilde_sk = theta_samples(s, k, m) - theta_bar_k;
                    double v_sk = rho_mobility_m * log_mobility_centered(s, k) 
                                - rho_dist_m * dist_centered(s, k) 
                                + theta_tilde_sk;
                    u_mat(s, k) = std::exp(v_sk);
                    U_k_sum += u_mat(s, k);
                }
                else
                {
                    u_mat(s, k) = 0.0;  // Diagonal not used in off-diagonal calculation
                }
            }
            U_k(k) = U_k_sum;
        }

        // Compute alpha for each time point
        Eigen::Index n_times = time_varying ? nt_plus_1 : 1;

        for (Eigen::Index t = 0; t < n_times; t++)
        {
            for (Eigen::Index k = 0; k < ns; k++)
            {
                // Compute w_k(t) = logistic(logit_wdiag_intercept_k + X' * slopes)
                double logit_w_k = logit_wdiag_intercept_samples(k, m);
                
                if (use_covariates)
                {
                    Eigen::Index t_idx = time_varying ? t : 0;
                    for (Eigen::Index p = 0; p < np; p++)
                    {
                        logit_w_k += logit_wdiag_slopes_samples(p, m) * X(p, k, t_idx);
                    }
                }
                
                double w_k = 1.0 / (1.0 + std::exp(-logit_w_k));

                // Compute alpha_{s,k}(t) for all s
                for (Eigen::Index s = 0; s < ns; s++)
                {
                    double alpha_sk;
                    if (s == k)
                    {
                        // Diagonal: retention
                        alpha_sk = w_k;
                    }
                    else
                    {
                        // Off-diagonal: normalized cross-regional transmission
                        if (U_k(k) > 1e-10)
                        {
                            alpha_sk = (1.0 - w_k) * u_mat(s, k) / U_k(k);
                        }
                        else
                        {
                            // Fallback: uniform distribution if U_k is zero
                            alpha_sk = (1.0 - w_k) / (ns - 1);
                        }
                    }

                    // Store in output array
                    if (time_varying)
                    {
                        // Index: s + k*ns + t*ns*ns + m*ns*ns*nt_plus_1
                        Eigen::Index idx = s + k * ns + t * ns * ns + m * ns * ns * nt_plus_1;
                        alpha_out[idx] = alpha_sk;
                    }
                    else
                    {
                        // Index: s + k*ns + m*ns*ns
                        Eigen::Index idx = s + k * ns + m * ns * ns;
                        alpha_out[idx] = alpha_sk;
                    }
                }
            }
        }
    }

    return alpha_out;
} // calculate_alpha


//' @export
// [[Rcpp::export]]
Rcpp::List evaluate_posterior_predictive(
    const Rcpp::List &mcmc_output, 
    const Rcpp::List &true_vals, // true parameter values if not inferred in `mcmc_output`
    const Rcpp::NumericVector &Y_in, // (nt + 1) x ns
    const Rcpp::Nullable<Rcpp::NumericMatrix> &Rt_in = R_NilValue, // (nt + 1) x ns
    const Rcpp::Nullable<Rcpp::NumericMatrix> &dist_matrix = R_NilValue, // ns x ns, pairwise distance matrix
    const Rcpp::Nullable<Rcpp::NumericMatrix> &mobility_matrix = R_NilValue, // ns x ns, pairwise mobility matrix
    const Rcpp::Nullable<Rcpp::NumericVector> &X_in = R_NilValue,
    const Eigen::Index &nsample = 1000
)
{
    std::string fgain = "softplus";
    double c_sq = 4.0;
    Rcpp::List lagdist_defaults = Rcpp::List::create(
        Rcpp::Named("name") = "lognorm",
        Rcpp::Named("par1") = LN_MU,
        Rcpp::Named("par2") = LN_SD2,
        Rcpp::Named("truncated") = true,
        Rcpp::Named("rescaled") = true
    );


    Eigen::MatrixXd Y_obs;
    if (Y_in.hasAttribute("dim"))
    {
        // Already a matrix: convert directly
        Rcpp::NumericMatrix Y_mat(Y_in);
        Y_obs = Rcpp::as<Eigen::MatrixXd>(Y_mat);
    }
    else
    {
        // Vector: make it a (nt + 1) x 1 matrix
        Eigen::VectorXd y_vec = Rcpp::as<Eigen::VectorXd>(Y_in);
        Y_obs.resize(y_vec.size(), 1);
        Y_obs.col(0) = y_vec;
    }

    Eigen::MatrixXd Rt_obs;
    if (Rt_in.isNotNull())
    {
        Rcpp::NumericMatrix Rt_mat(Rt_in);
        Rt_obs = Rcpp::as<Eigen::MatrixXd>(Rt_mat);
    }

    const Eigen::Index nt = Y_obs.rows() - 1;
    const Eigen::Index ns = Y_obs.cols();

    const bool use_covariates = X_in.isNotNull();
    const bool use_distance = dist_matrix.isNotNull();
    const bool use_mobility = mobility_matrix.isNotNull();

    Eigen::Tensor<double, 3> wt_samples, theta_samples; // (nt + 1) x ns x nsample
    Eigen::MatrixXd logit_wdiag_intercept_samples, logit_wdiag_slopes_samples; // ns x nsample
    Eigen::VectorXd mu_samples, W_samples, rho_dist_samples, rho_mobility_samples; // nsample
    parse_mcmc_output(
        wt_samples,
        theta_samples,
        logit_wdiag_intercept_samples,
        logit_wdiag_slopes_samples,
        mu_samples,
        W_samples,
        rho_dist_samples,
        rho_mobility_samples,
        mcmc_output,
        true_vals,
        ns,
        nt,
        nsample,
        use_distance,
        use_mobility,
        use_covariates
    );
    

    Eigen::Tensor<double, 3> Y_rep_samples; // (nt + 1) x ns x nsample
    Eigen::Tensor<double, 3> Rt_samples; // (nt + 1) x ns x nsample
    Y_rep_samples.resize(Y_obs.rows(), Y_obs.cols(), nsample);
    Rt_samples.resize(Y_obs.rows(), Y_obs.cols(), nsample);

    for (Eigen::Index i = 0; i < nsample; i++)
    {
        Model model(
            Y_obs, dist_matrix, mobility_matrix, X_in,
            c_sq, fgain, lagdist_defaults
        );

        model.mu = mu_samples(i);

        Eigen::MatrixXd wt_slice(Y_obs.rows(), Y_obs.cols());
        Eigen::MatrixXd theta_slice(Y_obs.cols(), Y_obs.cols());
        for (Eigen::Index s = 0; s < Y_obs.cols(); ++s)
        {
            for (Eigen::Index k = 0; k < Y_obs.cols(); ++k)
            {
                theta_slice(s, k) = theta_samples(s, k, i);
            }

            for (Eigen::Index t = 0; t < Y_obs.rows(); ++t)
            {
                wt_slice(t, s) = wt_samples(t, s, i);
            }
        }

        model.temporal.wt = wt_slice;
        model.temporal.W = W_samples(i);
        model.spatial.theta = theta_slice;
        model.spatial.logit_wdiag_intercept = logit_wdiag_intercept_samples.col(i);
        if (use_covariates)
        {
            model.spatial.logit_wdiag_slope = logit_wdiag_slopes_samples.col(i);
        }
        if (use_distance)
        {
            model.spatial.rho_dist = rho_dist_samples(i);
        }
        if (use_mobility)
        {
            model.spatial.rho_mobility = rho_mobility_samples(i);
        }


        Eigen::MatrixXd u_mat(ns, ns);
        for (Eigen::Index j = 0; j < ns; j++)
        {
            u_mat.col(j) = model.spatial.compute_unnormalized_weight_col(j);
        }
        

        Eigen::MatrixXd Y_rep(Y_obs.rows(), Y_obs.cols());
        Eigen::MatrixXd Rt = model.temporal.compute_Rt(); // (nt + 1) x ns
        for (Eigen::Index t = 1; t < nt + 1; t++)
        {
            for (Eigen::Index s = 0; s < ns; s++)
            {
                Rt_samples(t, s, i) = Rt(t, s);

                double y_ts = R::rpois(std::max(model.mu, EPS));

                for (Eigen::Index k = 0; k < ns; k++)
                {
                    Eigen::VectorXd u_k = u_mat.col(k);
                    for (Eigen::Index l = 0; l < t; l++)
                    {
                        double n_sktl = 0.0;
                        if (t - l <= model.dlag.Fphi.size())
                        {
                            double alpha_skl = model.spatial.compute_alpha_elem(u_k, s, k, l);
                            double lag_prob = model.dlag.Fphi(t - l - 1);
                            double R_kl = Rt(l, k);
                            double lambda_sktl = alpha_skl * R_kl * lag_prob * Y_obs(l, k) + EPS;
                            y_ts += R::rpois(lambda_sktl);
                        }

                    } // for lags l < t
                } // for source locations k

                Y_rep(t, s) = (model.zinfl.Z(t, s) > 0 || !model.zinfl.inflated) ? y_ts : 0.0;
            } // for target locations s
        } // for time t

        Eigen::TensorMap<Eigen::Tensor<const double, 2>> Y_rep_map(Y_rep.data(), Y_rep.rows(), Y_rep.cols());
        Y_rep_samples.chip(i, 2) = Y_rep_map;
    } // for MCMC samples i = 0, ..., nsample - 1


    Eigen::VectorXd crps_vec(ns);
    Eigen::VectorXd chisqr_vec(ns);
    Eigen::VectorXd mae_vec(ns);
    for (Eigen::Index s = 0; s < ns; s++)
    {
        Eigen::MatrixXd Y_rep_s(nt + 1, nsample);
        Eigen::MatrixXd Rt_s(nt + 1, nsample);
        for (Eigen::Index i = 0; i < nsample; i++)
        {
            for (Eigen::Index t = 0; t < nt + 1; t++)
            {
                Y_rep_s(t, i) = Y_rep_samples(t, s, i);
                Rt_s(t, i) = Rt_samples(t, s, i);
            }
        }

        Eigen::VectorXd y_obs_s = Y_obs.col(s);
        crps_vec(s) = calculate_crps(y_obs_s, Y_rep_s);
        chisqr_vec(s) = calculate_chisqr(y_obs_s, Y_rep_s);

        if (Rt_in.isNotNull())
        {
            Eigen::VectorXd Rt_obs_s(nt + 1);
            for (Eigen::Index t = 0; t < nt + 1; t++)
            {
                Rt_obs_s(t) = Rt_obs(t, s);
            }
            mae_vec(s) = calculate_mae(Rt_obs_s, Rt_s, true);
        }
    } // for locations s


    Rcpp::List output = Rcpp::List::create(
        Rcpp::Named("Y_rep_samples") = tensor3_to_r(Y_rep_samples),
        Rcpp::Named("Rt_samples") = tensor3_to_r(Rt_samples),
        Rcpp::Named("crps") = crps_vec,
        Rcpp::Named("chisqr") = chisqr_vec
    );

    if (Rt_in.isNotNull())
    {
        output["mae"] = mae_vec;
    }

    return output;
} // evaluate_posterior_predictive


//' @export
// [[Rcpp::export]]
Rcpp::List forecast_network_hawkes(
    const Eigen::Index &k_step_ahead,
    const Rcpp::List &mcmc_output, 
    const Rcpp::List &true_vals, // true parameter values if not inferred in `mcmc_output`
    const Rcpp::NumericVector &Y_in, // (nt + 1) x ns
    const Rcpp::Nullable<Rcpp::NumericVector> &Y_forecast_true_in = R_NilValue, // (k_step_ahead) x ns
    const Rcpp::Nullable<Rcpp::NumericMatrix> &dist_matrix = R_NilValue, // ns x ns, pairwise distance matrix
    const Rcpp::Nullable<Rcpp::NumericMatrix> &mobility_matrix = R_NilValue, // ns x ns, pairwise mobility matrix
    const Rcpp::Nullable<Rcpp::NumericVector> &X_extended_in = R_NilValue,
    const Eigen::Index &nsample = 1000,
    const bool &sample_disturbances = true
)
{
    std::string fgain = "softplus";
    double c_sq = 4.0;
    Rcpp::List lagdist_defaults = Rcpp::List::create(
        Rcpp::Named("name") = "lognorm",
        Rcpp::Named("par1") = LN_MU,
        Rcpp::Named("par2") = LN_SD2,
        Rcpp::Named("truncated") = true,
        Rcpp::Named("rescaled") = true
    );


    Eigen::MatrixXd Y_obs;
    if (Y_in.hasAttribute("dim"))
    {
        // Already a matrix: convert directly
        Rcpp::NumericMatrix Y_mat(Y_in);
        Y_obs = Rcpp::as<Eigen::MatrixXd>(Y_mat);
    }
    else
    {
        // Vector: make it a (nt + 1) x 1 matrix
        Eigen::VectorXd y_vec = Rcpp::as<Eigen::VectorXd>(Y_in);
        Y_obs.resize(y_vec.size(), 1);
        Y_obs.col(0) = y_vec;
    }

    Eigen::MatrixXd Y_forecast_true;
    if (Y_forecast_true_in.isNotNull())
    {
        Rcpp::NumericVector Y_forecast_in(Y_forecast_true_in);
        if (Y_forecast_in.hasAttribute("dim"))
        {
            // Already a matrix: convert directly
            Rcpp::NumericMatrix Yf_mat(Y_forecast_in);
            Y_forecast_true = Rcpp::as<Eigen::MatrixXd>(Yf_mat);
        }
        else
        {
            // Vector: make it a (k_step_ahead) x 1 matrix
            Eigen::VectorXd yf_vec = Rcpp::as<Eigen::VectorXd>(Y_forecast_in);
            Y_forecast_true.resize(yf_vec.size(), 1);
            Y_forecast_true.col(0) = yf_vec;
        }
    }

    const Eigen::Index nt = Y_obs.rows() - 1;
    const Eigen::Index ns = Y_obs.cols();

    const bool use_covariates = X_extended_in.isNotNull();
    const bool use_distance = dist_matrix.isNotNull();
    const bool use_mobility = mobility_matrix.isNotNull();

    Eigen::Tensor<double, 3> wt_samples, theta_samples; // (nt + 1) x ns x nsample
    Eigen::MatrixXd logit_wdiag_intercept_samples, logit_wdiag_slopes_samples; // ns x nsample
    Eigen::VectorXd mu_samples, W_samples, rho_dist_samples, rho_mobility_samples; // nsample
    parse_mcmc_output(
        wt_samples,
        theta_samples,
        logit_wdiag_intercept_samples,
        logit_wdiag_slopes_samples,
        mu_samples,
        W_samples,
        rho_dist_samples,
        rho_mobility_samples,
        mcmc_output,
        true_vals,
        ns,
        nt,
        nsample,
        use_distance,
        use_mobility,
        use_covariates
    );

    Eigen::Tensor<double, 3> Y_forecast_samples(k_step_ahead, ns, nsample);
    Y_forecast_samples.setZero();
    Eigen::Tensor<double, 3> wt_forecast_samples(k_step_ahead, ns, nsample);
    wt_forecast_samples.setZero();
    Eigen::Tensor<double, 3> lambda_forecast_samples(k_step_ahead, ns, nsample);
    lambda_forecast_samples.setZero();

    for (Eigen::Index i = 0; i < nsample; i++)
    {
        Model model(
            Y_obs, dist_matrix, mobility_matrix, X_extended_in,
            c_sq, fgain, lagdist_defaults
        );

        model.mu = mu_samples(i);

        Eigen::MatrixXd wt_slice(Y_obs.rows(), Y_obs.cols());
        Eigen::MatrixXd theta_slice(Y_obs.cols(), Y_obs.cols());
        for (Eigen::Index s = 0; s < Y_obs.cols(); ++s)
        {
            for (Eigen::Index k = 0; k < Y_obs.cols(); ++k)
            {
                theta_slice(s, k) = theta_samples(s, k, i);
            }

            for (Eigen::Index t = 0; t < Y_obs.rows(); ++t)
            {
                wt_slice(t, s) = wt_samples(t, s, i);
            }
        }

        model.temporal.wt = wt_slice;
        model.temporal.W = W_samples(i);
        double Wsqrt = std::sqrt(W_samples(i));

        model.spatial.theta = theta_slice;
        model.spatial.logit_wdiag_intercept = logit_wdiag_intercept_samples.col(i);
        if (use_covariates)
        {
            model.spatial.logit_wdiag_slope = logit_wdiag_slopes_samples.col(i);
        }
        if (use_distance)
        {
            model.spatial.rho_dist = rho_dist_samples(i);
        }
        if (use_mobility)
        {
            model.spatial.rho_mobility = rho_mobility_samples(i);
        }

        Eigen::MatrixXd u_mat(ns, ns);
        for (Eigen::Index j = 0; j < ns; j++)
        {
            u_mat.col(j) = model.spatial.compute_unnormalized_weight_col(j);
        }

        Eigen::MatrixXd Y_extended(nt + 1 + k_step_ahead, ns);
        Y_extended.setZero();
        Y_extended.block(0, 0, nt + 1, ns) = Y_obs;

        Eigen::MatrixXd wt_extended(nt + 1 + k_step_ahead, ns);
        wt_extended.setZero();
        wt_extended.block(0, 0, nt + 1, ns) = wt_slice;

        for (Eigen::Index t = nt + 1; t < nt + 1 + k_step_ahead; t++)
        {
            for (Eigen::Index s = 0; s < ns; s++)
            {
                if (sample_disturbances)
                {
                    wt_extended(t, s) = R::rnorm(0.0, Wsqrt);
                }

                double lambda_ts = model.mu;
                double y_ts = R::rpois(std::max(model.mu, EPS));
                for (Eigen::Index k = 0; k < ns; k++)
                {
                    Eigen::VectorXd u_k = u_mat.col(k);
                    Eigen::VectorXd wt_cumsum = cumsum_vec(wt_extended.col(k));

                    for (Eigen::Index l = 0; l < t; l++)
                    {
                        double n_sktl = 0.0;
                        if (t - l <= model.dlag.Fphi.size())
                        {
                            double alpha_skl = model.spatial.compute_alpha_elem(u_k, s, k, l);
                            double lag_prob = model.dlag.Fphi(t - l - 1);

                            double R_kl = GainFunc::psi2hpsi(wt_cumsum(l), fgain);
                            double lambda_sktl = alpha_skl * R_kl * lag_prob * Y_extended(l, k) + EPS;

                            lambda_ts += lambda_sktl;
                            y_ts += R::rpois(lambda_sktl);
                        }

                    } // for lags l < t
                } // for source locations k

                Y_extended(t, s) = y_ts;
                lambda_forecast_samples(t - (nt + 1), s, i) = lambda_ts;
                Y_forecast_samples(t - (nt + 1), s, i) = Y_extended(t, s);
                wt_forecast_samples(t - (nt + 1), s, i) = wt_extended(t, s);
            } // for target locations s
        } // for time t

    } // for MCMC samples i = 0, ..., nsample - 1

    Rcpp::List output = Rcpp::List::create(
        Rcpp::Named("Y_forecast_samples") = tensor3_to_r(Y_forecast_samples),
        Rcpp::Named("wt_forecast_samples") = tensor3_to_r(wt_forecast_samples),
        Rcpp::Named("lambda_forecast_samples") = tensor3_to_r(lambda_forecast_samples)
    );

    if (Y_forecast_true_in.isNotNull())
    {
        Eigen::VectorXd crps_y(ns);
        Eigen::VectorXd chisqr_y(ns);
        Eigen::VectorXd crps_lambda(ns);
        Eigen::VectorXd chisqr_lambda(ns);
        for (Eigen::Index s = 0; s < ns; s++)
        {
            Eigen::MatrixXd Y_rep_s(k_step_ahead, nsample);
            Eigen::MatrixXd lambda_rep_s(k_step_ahead, nsample);
            for (Eigen::Index i = 0; i < nsample; i++)
            {
                for (Eigen::Index t = 0; t < k_step_ahead; t++)
                {
                    Y_rep_s(t, i) = Y_forecast_samples(t, s, i);
                    lambda_rep_s(t, i) = lambda_forecast_samples(t, s, i);
                }
            }

            Eigen::VectorXd y_obs_s = Y_forecast_true.col(s);
            crps_y(s) = calculate_crps(y_obs_s, Y_rep_s);
            chisqr_y(s) = calculate_chisqr(y_obs_s, Y_rep_s);
            crps_lambda(s) = calculate_crps(y_obs_s, lambda_rep_s);
            chisqr_lambda(s) = calculate_chisqr(y_obs_s, lambda_rep_s);
        } // for locations s

        output["crps_y"] = crps_y;
        output["chisqr_y"] = chisqr_y;
        output["crps_lambda"] = crps_lambda;
        output["chisqr_lambda"] = chisqr_lambda;
    }

    return output;
}