#pragma once
#ifndef DIAGNOSTICS_HPP
#define DIAGNOSTICS_HPP

#include <vector>
#include <complex>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <limits>
#include <RcppEigen.h>
#include <unsupported/Eigen/FFT>

// [[Rcpp::plugins(cpp17)]]
// [[Rcpp::depends(RcppEigen)]]


inline double effective_sample_size_eigen_impl(const Eigen::Ref<const Eigen::VectorXd> &draws)
{
    // Drop non-finite entries up front
    std::vector<double> vals;
    vals.reserve(static_cast<std::size_t>(draws.size()));
    for (Eigen::Index i = 0; i < draws.size(); ++i)
    {
        double v = draws[i];
        if (std::isfinite(v))
            vals.push_back(v);
    }

    const std::size_t n = vals.size();
    if (n < 2)
        return static_cast<double>(n);

    // Center
    double mean = std::accumulate(vals.begin(), vals.end(), 0.0) / static_cast<double>(n);
    for (double &v : vals)
    {
        v -= mean;
    }

    // Next power-of-two >= 2n for zero-padded FFT
    std::size_t nfft = 1;
    while (nfft < 2 * n)
        nfft <<= 1;

    std::vector<std::complex<double>> x(nfft, std::complex<double>(0.0, 0.0));
    for (std::size_t i = 0; i < n; ++i)
    {
        x[i] = std::complex<double>(vals[i], 0.0);
    }

    Eigen::FFT<double> fft;
    std::vector<std::complex<double>> freq;
    fft.fwd(freq, x);
    for (auto &c : freq)
    {
        c *= std::conj(c);
    }

    std::vector<std::complex<double>> acov_c;
    fft.inv(acov_c, freq);

    std::vector<double> acov(n, 0.0);
    for (std::size_t k = 0; k < n; ++k)
    {
        acov[k] = acov_c[k].real() / static_cast<double>(n - k); // unbiased
    }

    const double gamma0 = acov[0];
    if (!std::isfinite(gamma0) || gamma0 <= 0.0)
        return static_cast<double>(n);

    // Geyer's initial positive sequence
    double sum_rho = 0.0;
    for (std::size_t lag = 1; lag + 1 < n; lag += 2)
    {
        double rho1 = acov[lag] / gamma0;
        double rho2 = acov[lag + 1] / gamma0;
        double pair_sum = rho1 + rho2;
        if (!std::isfinite(pair_sum) || pair_sum < 0.0)
            break;
        sum_rho += pair_sum;
    }

    double ess = static_cast<double>(n) / (1.0 + 2.0 * sum_rho);
    return std::min(static_cast<double>(n), std::max(1.0, ess));
} // effective_sample_size_eigen_impl


// ---- CRPS ----
inline double calculate_crps(
    const Eigen::Ref<const Eigen::VectorXd> &y,
    const Eigen::Ref<const Eigen::MatrixXd> &Y
)
{
    const Eigen::Index nT = y.size();
    if (Y.rows() != nT)
        throw std::invalid_argument("calculate_crps: Number of rows in Y must match length of y");
    const Eigen::Index nsample = Y.cols();
    double total = 0.0;

    for (Eigen::Index t = 0; t < nT; ++t)
    {
        Eigen::RowVectorXd row = Y.row(t);
        const double obs = y(t);

        const double term1 = (row.array() - obs).abs().mean();

        std::vector<double> x(row.data(), row.data() + nsample);
        std::sort(x.begin(), x.end());
        double sum_weighted = 0.0;
        for (Eigen::Index k = 0; k < nsample; ++k)
        {
            sum_weighted += (2.0 * static_cast<double>(k) - (static_cast<double>(nsample) - 1.0)) * x[static_cast<std::size_t>(k)];
        }
        const double mean_abs_diff = 2.0 * sum_weighted / (static_cast<double>(nsample) * static_cast<double>(nsample));

        double crps = term1 - 0.5 * mean_abs_diff;
        if (crps < 0.0)
            crps = 0.0;
        total += crps;
    }
    return total / static_cast<double>(nT);
}


inline double calculate_crps(
    const Rcpp::NumericVector &y, 
    const Rcpp::NumericMatrix &Y
)
{
    Eigen::Map<const Eigen::VectorXd> y_eig(y.begin(), y.size());
    Eigen::Map<const Eigen::MatrixXd> Y_eig(Y.begin(), Y.nrow(), Y.ncol());
    return calculate_crps(y_eig, Y_eig);
}


// ---- MAE ----
inline double calculate_mae(const Eigen::Ref<const Eigen::VectorXd> &y_true,
                            const Eigen::Ref<const Eigen::MatrixXd> &Y,
                            const bool posterior_expected = true)
{
    if (Y.rows() != y_true.size())
        throw std::invalid_argument("calculate_mae: Y must be k x nsample; y_true must have length k.");

    const double k = static_cast<double>(y_true.size());
    if (posterior_expected)
    {
        Eigen::MatrixXd diff = Y.colwise() - y_true;
        Eigen::VectorXd mse_t = diff.array().square().rowwise().mean();
        return mse_t.mean();
    }
    else
    {
        Eigen::VectorXd yhat = Y.rowwise().mean();
        Eigen::VectorXd resid = yhat - y_true;
        return resid.squaredNorm() / k;
    }
}


inline double calculate_mae(
    const Rcpp::NumericVector &y_true,
    const Rcpp::NumericMatrix &Y,
    const bool posterior_expected = true
)
{
    Eigen::Map<const Eigen::VectorXd> y_true_eig(y_true.begin(), y_true.size());
    Eigen::Map<const Eigen::MatrixXd> Y_eig(Y.begin(), Y.nrow(), Y.ncol());
    return calculate_mae(y_true_eig, Y_eig, posterior_expected);
}


// ---- Chi-square ----
inline double calculate_chisqr(
    const Eigen::Ref<const Eigen::VectorXd> &y_true,
    const Eigen::Ref<const Eigen::MatrixXd> &Y,
    const double min_var = 1e-6
)
{
    const Eigen::Index k = Y.rows();
    const Eigen::Index nsample = Y.cols();

    double total = 0.0;
    Eigen::Index valid_k = 0;

    for (Eigen::Index i = 0; i < k; ++i)
    {
        const Eigen::RowVectorXd row = Y.row(i);
        const double mean = row.mean();
        const double var = (row.array() - mean).square().sum() / static_cast<double>(nsample - 1);

        if (var < min_var)
            continue;

        valid_k++;
        const double resid = y_true(i) - mean;
        total += (resid * resid) / var;
    }

    return (valid_k > 0) ? total / static_cast<double>(valid_k) : std::numeric_limits<double>::quiet_NaN();
}


inline double calculate_chisqr(
    const Rcpp::NumericVector &y_true,
    const Rcpp::NumericMatrix &Y,
    const double min_var = 1e-6
)
{
    Eigen::Map<const Eigen::MatrixXd> Y_eig(Y.begin(), Y.nrow(), Y.ncol());
    Eigen::Map<const Eigen::VectorXd> y_true_eig(y_true.begin(), y_true.size());
    return calculate_chisqr(y_true_eig, Y_eig, min_var);
}


// ---- Gelman-Rubin Rhat ----
inline double gelman_rubin(const Eigen::Ref<const Eigen::MatrixXd> &samples)
{
    const Eigen::Index nchain = samples.cols();
    if (nchain < 2)
        throw std::invalid_argument("gelman_rubin: need at least 2 chains (nchain >= 2).");

    std::vector<Eigen::Index> good_rows;
    good_rows.reserve(static_cast<std::size_t>(samples.rows()));
    for (Eigen::Index i = 0; i < samples.rows(); ++i)
    {
        bool all_finite = true;
        for (Eigen::Index j = 0; j < nchain; ++j)
        {
            if (!std::isfinite(samples(i, j)))
            {
                all_finite = false;
                break;
            }
        }
        if (all_finite)
            good_rows.push_back(i);
    }

    if (good_rows.size() < 2)
        throw std::invalid_argument("gelman_rubin: fewer than 2 valid rows after excluding rows with infinite values.");

    Eigen::MatrixXd X(static_cast<Eigen::Index>(good_rows.size()), nchain);
    for (std::size_t r = 0; r < good_rows.size(); ++r)
    {
        X.row(static_cast<Eigen::Index>(r)) = samples.row(good_rows[r]);
    }

    const double nsample = static_cast<double>(X.rows());
    Eigen::RowVectorXd chain_means = X.colwise().mean();

    Eigen::RowVectorXd chain_vars(nchain);
    for (Eigen::Index j = 0; j < nchain; ++j)
    {
        Eigen::VectorXd col = X.col(j).array() - chain_means(j);
        chain_vars(j) = col.squaredNorm() / (nsample - 1.0);
    }

    const double W = chain_vars.mean();
    const double mean_of_means = chain_means.mean();
    const double B = nsample * ((chain_means.array() - mean_of_means).square().mean());

    if (W <= 0.0)
    {
        if (B <= 0.0)
            return 1.0;
        return std::numeric_limits<double>::infinity();
    }

    const double var_hat = ((nsample - 1.0) / nsample) * W + (1.0 / nsample) * B;
    return std::sqrt(var_hat / W);
}


inline double gelman_rubin(const Rcpp::NumericMatrix &samples)
{
    Eigen::Map<const Eigen::MatrixXd> samples_eig(samples.begin(), samples.nrow(), samples.ncol());
    return gelman_rubin(samples_eig);
}


#endif
