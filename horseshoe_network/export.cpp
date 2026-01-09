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


void parse_mcmc_output(
    Eigen::Tensor<double, 3> &wt_samples,
    Eigen::Tensor<double, 3> &theta_samples,
    Eigen::MatrixXd &wdiag_samples,
    Eigen::VectorXd &mu_samples,
    Eigen::VectorXd &W_samples,
    Eigen::VectorXd &rho_dist_samples,
    Eigen::VectorXd &rho_mobility_samples,
    Eigen::VectorXd &zinfl_beta0_samples,
    Eigen::VectorXd &zinfl_beta1_samples,
    const Rcpp::List &mcmc_output, 
    const Rcpp::List &true_vals, 
    const Eigen::Index &ns, 
    const Eigen::Index &nt, 
    const Eigen::Index &nsample,
    const bool &use_distance,
    const bool &use_mobility
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

    bool found_wdiag = false;
    if (mcmc_output.containsElementNamed("horseshoe"))
    {
        Rcpp::List hs_list = mcmc_output["horseshoe"];
        if (hs_list.containsElementNamed("wdiag"))
        {
            Rcpp::NumericMatrix intercept_mat = Rcpp::as<Rcpp::NumericMatrix>(hs_list["wdiag"]); // ns x nsample
            if (intercept_mat.ncol() != nsample)
            {
                throw std::invalid_argument("Number of wdiag samples does not match number of wt samples.");
            }
            wdiag_samples = Rcpp::as<Eigen::MatrixXd>(intercept_mat);
            found_wdiag = true;
        }
    }

    if (!found_wdiag && true_vals.containsElementNamed("horseshoe"))
    {
        Rcpp::List hs_list = true_vals["horseshoe"];
        if (hs_list.containsElementNamed("wdiag"))
        {
            Rcpp::NumericVector intercept_vec = Rcpp::as<Rcpp::NumericVector>(hs_list["wdiag"]); // ns
            if (intercept_vec.size() != ns)
            {
                throw std::invalid_argument("Dimension of true wdiag does not match number of locations in Y_obs.");
            }
            wdiag_samples.resize(ns, nsample);
            for (Eigen::Index i = 0; i < nsample; i++)
            {
                for (Eigen::Index s = 0; s < ns; s++)
                {
                    wdiag_samples(s, i) = intercept_vec[s];
                }
            }
            found_wdiag = true;
        }
    }

    if (!found_wdiag)
    {
        throw std::invalid_argument("No wdiag samples in MCMC output or true values.");
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


    if (mcmc_output.containsElementNamed("zero_inflation"))
    {
        bool found_beta0 = false;
        Rcpp::List zinfl_list = mcmc_output["zero_inflation"];
        if (zinfl_list.containsElementNamed("beta0"))
        {
            Rcpp::NumericVector beta0_vec = Rcpp::as<Rcpp::NumericVector>(zinfl_list["beta0"]); // nsample
            if (beta0_vec.size() != nsample)
            {
                throw std::invalid_argument("Number of zinfl beta0 samples does not match number of wt samples.");
            }
            zinfl_beta0_samples = Rcpp::as<Eigen::VectorXd>(beta0_vec);
            found_beta0 = true;
        }

        if (!found_beta0 && true_vals.containsElementNamed("zero_inflation"))
        {
            Rcpp::List zinfl_true_list = true_vals["zero_inflation"];
            if (zinfl_true_list.containsElementNamed("beta0"))
            {
                double beta0_true = Rcpp::as<double>(zinfl_true_list["beta0"]);
                zinfl_beta0_samples.resize(nsample);
                zinfl_beta0_samples.fill(beta0_true);
                found_beta0 = true;
            }
        }

        if (!found_beta0)
        {
            throw std::invalid_argument("No zinfl beta0 samples in MCMC output or true values.");
        }

        bool found_beta1 = false;
        if (zinfl_list.containsElementNamed("beta1"))
        {
            Rcpp::NumericVector beta1_vec = Rcpp::as<Rcpp::NumericVector>(zinfl_list["beta1"]); // nsample
            if (beta1_vec.size() != nsample)
            {
                throw std::invalid_argument("Number of zinfl beta1 samples does not match number of wt samples.");
            }
            zinfl_beta1_samples = Rcpp::as<Eigen::VectorXd>(beta1_vec);
            found_beta1 = true;
        }

        if (!found_beta1 && true_vals.containsElementNamed("zero_inflation"))
        {
            Rcpp::List zinfl_true_list = true_vals["zero_inflation"];
            if (zinfl_true_list.containsElementNamed("beta1"))
            {
                double beta1_true = Rcpp::as<double>(zinfl_true_list["beta1"]);
                zinfl_beta1_samples.resize(nsample);
                zinfl_beta1_samples.fill(beta1_true);
                found_beta1 = true;
            }
        }

        if (!found_beta1)
        {
            throw std::invalid_argument("No zinfl beta1 samples in MCMC output or true values.");
        }
    }

    return;
} // parse_mcmc_output


//' @export
// [[Rcpp::export]]
Rcpp::List evaluate_posterior_predictive(
    const Rcpp::List &mcmc_output, 
    const Rcpp::List &true_vals, // true parameter values if not inferred in `mcmc_output`
    const Rcpp::NumericVector &Y_in, // (nt + 1) x ns
    const Rcpp::Nullable<Rcpp::NumericMatrix> &Rt_in = R_NilValue, // (nt + 1) x ns
    const Rcpp::Nullable<Rcpp::NumericMatrix> &dist_matrix = R_NilValue, // ns x ns, pairwise distance matrix
    const Rcpp::Nullable<Rcpp::NumericMatrix> &mobility_matrix = R_NilValue, // ns x ns, pairwise mobility matrix
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

    const bool use_distance = dist_matrix.isNotNull();
    const bool use_mobility = mobility_matrix.isNotNull();

    Eigen::Tensor<double, 3> wt_samples, theta_samples; // (nt + 1) x ns x nsample
    Eigen::MatrixXd wdiag_samples; // ns x nsample
    Eigen::VectorXd mu_samples, W_samples, rho_dist_samples, rho_mobility_samples, zinfl_beta0_samples, zinfl_beta1_samples; // nsample
    parse_mcmc_output(
        wt_samples,
        theta_samples,
        wdiag_samples,
        mu_samples,
        W_samples,
        rho_dist_samples,
        rho_mobility_samples,
        zinfl_beta0_samples,
        zinfl_beta1_samples,
        mcmc_output,
        true_vals,
        ns,
        nt,
        nsample,
        use_distance,
        use_mobility
    );
    

    Eigen::Tensor<double, 3> Y_rep_samples; // (nt + 1) x ns x nsample
    Eigen::Tensor<double, 3> Rt_samples; // (nt + 1) x ns x nsample
    Y_rep_samples.resize(Y_obs.rows(), Y_obs.cols(), nsample);
    Rt_samples.resize(Y_obs.rows(), Y_obs.cols(), nsample);

    for (Eigen::Index i = 0; i < nsample; i++)
    {
        Model model(
            Y_obs, dist_matrix, mobility_matrix,
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
        Eigen::MatrixXd Rt = model.temporal.compute_Rt(); // (nt + 1) x ns

        model.spatial.theta = theta_slice;
        model.spatial.wdiag = wdiag_samples.col(i);
        if (use_distance)
        {
            model.spatial.rho_dist = rho_dist_samples(i);
        }
        if (use_mobility)
        {
            model.spatial.rho_mobility = rho_mobility_samples(i);
        }
        model.spatial.compute_alpha();

        if (zinfl_beta0_samples.size() > 0 && zinfl_beta1_samples.size() > 0)
        {
            model.zinfl = ZeroInflation(Y_obs, true);
            model.zinfl.beta0 = zinfl_beta0_samples(i);
            model.zinfl.beta1 = zinfl_beta1_samples(i);
        }


        Eigen::MatrixXd u_mat(ns, ns);
        for (Eigen::Index j = 0; j < ns; j++)
        {
            u_mat.col(j) = model.spatial.compute_unnormalized_weight_col(j);
        }
        

        Eigen::MatrixXd Y_rep(Y_obs.rows(), Y_obs.cols());
        for (Eigen::Index t = 1; t < nt + 1; t++)
        {
            for (Eigen::Index s = 0; s < ns; s++)
            {
                Rt_samples(t, s, i) = Rt(t, s);

                double y_ts = R::rpois(std::max(model.mu, EPS));

                for (Eigen::Index k = 0; k < ns; k++)
                {
                    double alpha_sk = model.spatial.alpha(s, k);
                    for (Eigen::Index l = 0; l < t; l++)
                    {
                        double n_sktl = 0.0;
                        if (t - l <= model.dlag.Fphi.size())
                        {
                            double lag_prob = model.dlag.Fphi(t - l - 1);
                            double R_kl = Rt(l, k);
                            double lambda_sktl = alpha_sk * R_kl * lag_prob * Y_obs(l, k) + EPS;
                            y_ts += R::rpois(lambda_sktl);
                        }

                    } // for lags l < t
                } // for source locations k

                // if (model.zinfl.inflated)
                // {
                //     double logit_p_zero = model.zinfl.beta0 + model.zinfl.beta1 * model.zinfl.Z(t - 1, s);
                //     double p_zero = 1.0 / (1.0 + std::exp(- logit_p_zero));
                //     model.zinfl.Z(t, s) = (R::runif(0.0, 1.0) < p_zero) ? 1.0 : 0.0;
                // }
                // else
                // {
                    model.zinfl.Z(t, s) = 1.0;
                // }

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

    const bool use_distance = dist_matrix.isNotNull();
    const bool use_mobility = mobility_matrix.isNotNull();

    Eigen::Tensor<double, 3> wt_samples, theta_samples; // (nt + 1) x ns x nsample
    Eigen::MatrixXd wdiag_samples; // ns x nsample
    Eigen::VectorXd mu_samples, W_samples, rho_dist_samples, rho_mobility_samples, zinfl_beta0_samples, zinfl_beta1_samples; // nsample
    parse_mcmc_output(
        wt_samples,
        theta_samples,
        wdiag_samples,
        mu_samples,
        W_samples,
        rho_dist_samples,
        rho_mobility_samples,
        zinfl_beta0_samples,
        zinfl_beta1_samples,
        mcmc_output,
        true_vals,
        ns,
        nt,
        nsample,
        use_distance,
        use_mobility
    );

    Eigen::Tensor<double, 3> Y_forecast_samples(k_step_ahead, ns, nsample);
    Y_forecast_samples.setZero();
    Eigen::Tensor<double, 3> Z_forecast_samples(k_step_ahead, ns, nsample);
    Z_forecast_samples.setConstant(1.0);
    Eigen::Tensor<double, 3> wt_forecast_samples(k_step_ahead, ns, nsample);
    wt_forecast_samples.setZero();
    Eigen::Tensor<double, 3> lambda_forecast_samples(k_step_ahead, ns, nsample);
    lambda_forecast_samples.setZero();

    for (Eigen::Index i = 0; i < nsample; i++)
    {
        Model model(
            Y_obs, dist_matrix, mobility_matrix,
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
        model.spatial.wdiag = wdiag_samples.col(i);
        if (use_distance)
        {
            model.spatial.rho_dist = rho_dist_samples(i);
        }
        if (use_mobility)
        {
            model.spatial.rho_mobility = rho_mobility_samples(i);
        }
        model.spatial.compute_alpha();

        if (zinfl_beta0_samples.size() > 0 && zinfl_beta1_samples.size() > 0)
        {
            model.zinfl = ZeroInflation(Y_obs, true);
            model.zinfl.beta0 = zinfl_beta0_samples(i);
            model.zinfl.beta1 = zinfl_beta1_samples(i);
        }

        Eigen::MatrixXd u_mat(ns, ns);
        for (Eigen::Index j = 0; j < ns; j++)
        {
            u_mat.col(j) = model.spatial.compute_unnormalized_weight_col(j);
        }

        Eigen::MatrixXd Y_extended(nt + 1 + k_step_ahead, ns);
        Y_extended.setZero();
        Y_extended.block(0, 0, nt + 1, ns) = Y_obs;

        Eigen::MatrixXd Z_extended(nt + 1 + k_step_ahead, ns);
        Z_extended.setOnes();
        Z_extended.block(0, 0, nt + 1, ns) = model.zinfl.Z;

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
                    double alpha_sk = model.spatial.alpha(s, k);
                    Eigen::VectorXd wt_cumsum = cumsum_vec(wt_extended.col(k));

                    for (Eigen::Index l = 0; l < t; l++)
                    {
                        double n_sktl = 0.0;
                        if (t - l <= model.dlag.Fphi.size())
                        {
                            double lag_prob = model.dlag.Fphi(t - l - 1);

                            double R_kl = GainFunc::psi2hpsi(wt_cumsum(l), fgain);
                            double lambda_sktl = alpha_sk * R_kl * lag_prob * Y_extended(l, k) + EPS;

                            lambda_ts += lambda_sktl;
                            y_ts += R::rpois(lambda_sktl);
                        }

                    } // for lags l < t
                } // for source locations k

                // if (model.zinfl.inflated)
                // {
                //     double logit_p_zero = model.zinfl.beta0 + model.zinfl.beta1 * Z_extended(t - 1, s);
                //     double p_zero = 1.0 / (1.0 + std::exp(- logit_p_zero));
                //     Z_extended(t, s) = (R::runif(0.0, 1.0) < p_zero) ? 1.0 : 0.0;
                // }
                // else
                // {
                    Z_extended(t, s) = 1.0;
                // }

                Y_extended(t, s) = (Z_extended(t, s) > 0 || !model.zinfl.inflated) ? y_ts : 0.0;

                wt_forecast_samples(t - (nt + 1), s, i) = wt_extended(t, s);
                lambda_forecast_samples(t - (nt + 1), s, i) = lambda_ts;
                Y_forecast_samples(t - (nt + 1), s, i) = Y_extended(t, s);
                Z_forecast_samples(t - (nt + 1), s, i) = Z_extended(t, s);
            } // for target locations s
        } // for time t

    } // for MCMC samples i = 0, ..., nsample - 1

    Rcpp::List output = Rcpp::List::create(
        Rcpp::Named("Y_forecast_samples") = tensor3_to_r(Y_forecast_samples),
        Rcpp::Named("Z_forecast_samples") = tensor3_to_r(Z_forecast_samples),
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