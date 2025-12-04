#pragma once
#ifndef MODEL_H
#define MODEL_H

#include <RcppEigen.h>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <unsupported/Eigen/SpecialFunctions>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <vector>

#include <progress.hpp>
#include <progress_bar.hpp>

#include "../core/GainFuncEigen.hpp"
#include "../core/LagDistEigen.hpp"
#include "../inference/hmc.hpp"

// [[Rcpp::plugins(cpp17)]]
// [[Rcpp::depends(RcppEigen,RcppProgress)]]

inline Eigen::Tensor<double, 4> r_to_tensor4(Rcpp::NumericVector &arr)
{
    if (!arr.hasAttribute("dim"))
        Rcpp::stop("need dim");
    Rcpp::IntegerVector dim = arr.attr("dim");
    if (dim.size() != 4)
        Rcpp::stop("need 4D array");

    Eigen::array<Eigen::Index, 4> dims = {dim[0], dim[1], dim[2], dim[3]};
    Eigen::TensorMap<Eigen::Tensor<double, 4>> t(arr.begin(), dims);
    return t;
}; // r_to_tensor4


inline Rcpp::NumericVector tensor4_to_r(const Eigen::Tensor<double, 4> &t)
{
    Eigen::array<Eigen::Index, 4> dims = t.dimensions();
    Rcpp::NumericVector out(dims[0] * dims[1] * dims[2] * dims[3]);
    std::copy(t.data(), t.data() + out.size(), out.begin());
    out.attr("dim") = Rcpp::IntegerVector::create(
        dims[0], dims[1], dims[2], dims[3]);
    return out;
}; // tensor4_to_r


inline Rcpp::NumericVector tensor3_to_r(const Eigen::Tensor<double, 3> &t)
{
    Eigen::array<Eigen::Index, 3> dims = t.dimensions();
    Rcpp::NumericVector out(dims[0] * dims[1] * dims[2]);
    std::copy(t.data(), t.data() + out.size(), out.begin());
    out.attr("dim") = Rcpp::IntegerVector::create(dims[0], dims[1], dims[2]);
    return out;
}; // tensor3_to_r


Eigen::VectorXd cumsum_vec(const Eigen::VectorXd &v) {
    Eigen::VectorXd out(v.size());
    std::partial_sum(v.data(), v.data() + v.size(), out.data());
    return out;
}


/**
 * @brief Cumulative sum down rows for each column
 * 
 * @param M 
 * @return Eigen::MatrixXd 
 */
Eigen::MatrixXd cumsum_rows(const Eigen::MatrixXd &M) {
    Eigen::MatrixXd out(M.rows(), M.cols());
    for (Eigen::Index j = 0; j < M.cols(); ++j) {
        std::partial_sum(M.col(j).data(),
                         M.col(j).data() + M.rows(),
                         out.col(j).data());
    }
    return out;
}


/**
 * @brief Cumulative sum across columns for each row
 * 
 * @param M 
 * @return Eigen::MatrixXd 
 */
Eigen::MatrixXd cumsum_cols(const Eigen::MatrixXd &M) {
    Eigen::MatrixXd out(M.rows(), M.cols());
    for (Eigen::Index i = 0; i < M.rows(); ++i) {
        std::partial_sum(M.row(i).data(),
                         M.row(i).data() + M.cols(),
                         out.row(i).data());
    }
    return out;
}


/**
 * @brief Stochastic spatial network component for discretizeds network Hawkes.
 * 
 */
class SpatialNetwork
{
private:
    Eigen::Index ns = 0; // number of spatial locations
    Eigen::MatrixXd dist_scaled; // ns x ns, pairwise scaled distance matrix
    Eigen::MatrixXd mobility_scaled; // ns x ns, pairwise scaled mobility matrix


    Eigen::VectorXd compute_alpha_col(const Eigen::Index &k) const
    {
        Eigen::VectorXd u(ns); // unnormalized weights
        for (Eigen::Index s = 0; s < ns; s++)
        { // Loop over destinations
            const double mob_safe = std::max(mobility_scaled(s, k), EPS8);
            u(s) = std::exp(-rho_dist * dist_scaled(s, k)) *
                   std::pow(mob_safe, rho_mobility);
        } // for s

        return (kappa(k) * u / u.sum()).array().max(EPS8);
    } // compute_alpha_col


    /**
     * @brief Sample a column of the spatial network A
     * 
     * @param alpha Eigen::VectorXd, Dirichlet parameters for the column
     * @return Eigen::VectorXd 
     */
    static Eigen::VectorXd sample_A_col(
        const Eigen::VectorXd &alpha // Dirichlet parameters for the column
    )
    {
        const Eigen::Index ns = alpha.size();
        Eigen::VectorXd sample(ns);
        for (Eigen::Index s = 0; s < ns; s++)
        {
            sample(s) = R::rgamma(alpha(s), 1.0);
        } // for destination s given source k

        double sample_sum = sample.sum();
        if (sample_sum > 0)
        {
            return sample / sample_sum;
        }
        else
        {
            Eigen::VectorXd uniform_sample(ns);
            uniform_sample.setConstant(1.0 / static_cast<double>(ns));
            return uniform_sample;
        }
    } // sample_A_col


public:
    Eigen::MatrixXd A; // ns x ns, adjacency matrix / stochastic network
    Eigen::VectorXd kappa; // ns x 1, concentration parameters for Dirich
    double rho_dist = 1.0; // distance decay parameter
    double rho_mobility = 1.0; // mobility scaling parameter


    SpatialNetwork()
    {
        return;
    } // SpatialNetwork default constructor


    SpatialNetwork(
        const Eigen::MatrixXd &dist_scaled_in, // ns x ns, pairwise scaled distance matrix
        const Eigen::MatrixXd &mobility_scaled_in, // ns x ns, pairwise scaled mobility matrix
        const bool &random_init = false
    )
    {
        dist_scaled = dist_scaled_in;
        mobility_scaled = mobility_scaled_in;
        ns = dist_scaled.rows();

        A.resize(ns, ns);
        kappa.resize(ns);

        if (random_init)
        {
            for (Eigen::Index k = 0; k < ns; k++)
            { // Loop over source locations
                Eigen::VectorXd alpha = compute_alpha_col(k);
                A.col(k) = sample_A_col(alpha);
                kappa(k) = R::rgamma(2.0, 1.0); // shape=2.0, scale=1.0
            }

            rho_dist = R::runif(0.0, 2.0);
            rho_mobility = R::runif(0.0, 2.0);
        }
        else
        {
            A.setConstant(1.0 / static_cast<double>(ns));
            kappa.setOnes();
        }

        return;
    } // SpatialNetwork constructor


    SpatialNetwork(
        const Eigen::MatrixXd &dist_scaled_in, // ns x ns, pairwise scaled distance matrix
        const Eigen::MatrixXd &mobility_scaled_in, // ns x ns, pairwise scaled mobility matrix
        const Rcpp::List &settings
    )
    {
        dist_scaled = dist_scaled_in;
        mobility_scaled = mobility_scaled_in;
        ns = dist_scaled.rows();

        A.resize(ns, ns);
        A.setOnes();

        kappa.resize(ns);
        kappa.setOnes();

        if (settings.containsElementNamed("rho_dist"))
        {
            rho_dist = Rcpp::as<double>(settings["rho_dist"]);
        } // if rho_dist
        if (settings.containsElementNamed("rho_mobility"))
        {
            rho_mobility = Rcpp::as<double>(settings["rho_mobility"]);
        } // if rho_mobility
        if (settings.containsElementNamed("kappa"))
        {
            Eigen::VectorXd kappa_in = Rcpp::as<Eigen::VectorXd>(settings["kappa"]);
            if (kappa_in.size() < ns)
            {
                kappa.head(kappa_in.size()) = kappa_in;
                kappa.tail(ns - kappa_in.size()).fill(kappa_in.tail(1)(0));
            }
            else
            {
                kappa = kappa_in.head(ns);
            }
        } // if kappa

        return;
    } // SpatialNetwork constructor


    /**
     * @brief Sample adjacency matrix A where each column A[, j] ~ Dirichlet(alpha[, j])
     * 
     */
    void sample_A()
    {
        for (Eigen::Index k = 0; k < ns; k++)
        { // Loop over source locations
            Eigen::VectorXd alpha = compute_alpha_col(k);
            A.col(k) = sample_A_col(alpha);
        }
        return;
    } // sample_A


    /**
     * @brief Gibbs sampler of adjacency matrix A given secondary infection tensor N
     * 
     * @param N 
     */
    void update_A(
        const Eigen::Tensor<double, 4> &N // (ns of destination) x (ns of source) x (nt + 1 of destination) x  (nt + 1 of source)
    )
    {
        for (Eigen::Index k = 0; k < ns; k++)
        { // Loop over source locations
            // Compute alpha (prior parameters)
            Eigen::VectorXd alpha_post = compute_alpha_col(k);

            for (Eigen::Index s = 0; s < ns; s++)
            { // Loop over destination locations
                Eigen::Tensor<double, 0> N_sk = N.chip(s, 0).chip(k, 0).sum();
                alpha_post(s) += N_sk(0);
            }

            // Sample A.col(k) ~ Dirichlet(alpha_post)
            A.col(k) = sample_A_col(alpha_post);
        }

        return;
    } // update_A


    /**
     * @brief Derivative of log-likelihood with respect to kappa.
     * 
     * @param N 4D tensor (ns of destination) x (ns of source) x (nt + 1 of destination) x  (nt + 1 of source)
     * @param add_jacobian Add Jacobian of kappa w.r.t log(kappa) if true
     * @return `Eigen::VectorXd` ns x 1, derivative of log-likelihood w.r.t kappa or log(kappa).
     */
    Eigen::VectorXd dloglike_dkappa(
        const Eigen::Tensor<double, 4> &N,
        const bool &add_jacobian = true)
    {
        Eigen::VectorXd grad(ns);

        for (Eigen::Index k = 0; k < ns; ++k)
        {
            // N_k: total secondaries from source k
            Eigen::Tensor<double, 0> N_k = N.chip(k, 1).sum();

            double term_global = Eigen::numext::digamma(kappa(k)) - Eigen::numext::digamma(kappa(k) + N_k(0));

            Eigen::VectorXd alpha = compute_alpha_col(k);
            double term_alpha = 0.0;
            for (Eigen::Index s = 0; s < ns; ++s)
            {
                Eigen::Tensor<double, 0> N_sk = N.chip(s, 0).chip(k, 0).sum();

                double w_sk = alpha(s) / kappa(k);
                double g_sk = Eigen::numext::digamma(alpha(s) + N_sk(0)) - Eigen::numext::digamma(alpha(s));

                term_alpha += g_sk * w_sk;
            } // for destination s

            grad(k) = term_global + term_alpha;

            if (add_jacobian)
            {
                grad(k) *= kappa(k);
            }
        } // for source location k

        return grad;
    } // dloglike_dkappa


    /**
     * @brief Marginal log-likelihood of secondary infections integrating out adjacency matrix A.
     * 
     * @param N 4D tensor of size (ns of destination) x (ns of source) x (nt + 1 of destination) x  (nt + 1 of source), unobserved secondary infections.
     * @return double 
     */
    double marginal_log_likelihood(
        const Eigen::Tensor<double, 4> &N
    )
    {
        double loglike = 0.0;
        for (Eigen::Index k = 0; k < ns; k++)
        { // Loop over source locations
            Eigen::VectorXd alpha = compute_alpha_col(k);

            Eigen::Tensor<double, 0> N_k = N.chip(k, 1).sum();
            loglike += Eigen::numext::lgamma(kappa(k)) - Eigen::numext::lgamma(kappa(k) + N_k(0));

            for (Eigen::Index s = 0; s < ns; s++)
            { // Loop over destination locations
                Eigen::Tensor<double, 0> N_sk = N.chip(s, 0).chip(k, 0).sum();
                loglike += Eigen::numext::lgamma(alpha(s) + N_sk(0)) - Eigen::numext::lgamma(alpha(s));
            }
        }

        return loglike;
    }


    /**
     * @brief HMC sampler to update kappa. Operate on the unconstrained log(kappa).
     * 
     * @param energy_diff double, output energy difference between proposed and current state
     * @param grad_norm double, output gradient norm at current state
     * @param N 4D tensor of size (ns of destination) x (ns of source) x (nt + 1 of destination) x  (nt + 1 of source), unobserved secondary infections.
     * @param step_size 
     * @param n_leapfrog 
     * @param prior_mean_logkappa double (default 0.0), prior mean of log(kappa)
     * @param prior_sd_logkappa double (default 10.0), prior standard deviation of log(kappa)
     * @return `double` Metropolis acceptance probability
     */
    double update_kappa(
        double &energy_diff,
        double &grad_norm,
        const Eigen::Tensor<double, 4> &N,
        const double &step_size,
        const Eigen::Index &n_leapfrog,
        const double &prior_mean_logkappa = 0.0,
        const double &prior_sd_logkappa = 10.0
    )
    {
        const double prior_var_logkappa = prior_sd_logkappa * prior_sd_logkappa;
        Eigen::VectorXd mass_diag = Eigen::VectorXd::Constant(ns, 1.0);
        Eigen::VectorXd inv_mass = mass_diag.cwiseInverse();
        Eigen::VectorXd sqrt_mass = mass_diag.array().sqrt();

        // Current state
        Eigen::VectorXd current_kappa = kappa;
        Eigen::ArrayXd log_kappa = kappa.array().log();

        // Compute current energy
        double current_loglik = marginal_log_likelihood(N);
        double current_logprior = - 0.5 * (log_kappa - prior_mean_logkappa).square().matrix().sum() / prior_var_logkappa;
        double current_energy = -(current_loglik + current_logprior);

        // Sample momentum
        Eigen::VectorXd momentum = sqrt_mass.array() * Eigen::VectorXd::NullaryExpr(ns, []() { return R::rnorm(0.0, 1.0); }).array();
        double current_kinetic = 0.5 * momentum.cwiseProduct(inv_mass).dot(momentum);

        // Leapfrog integration
        Eigen::VectorXd grad = dloglike_dkappa(N, true);
        grad.array() += - (log_kappa - prior_mean_logkappa) / prior_var_logkappa;
        grad_norm = grad.norm();

        momentum += 0.5 * step_size * grad;
        for (Eigen::Index lf_step = 0; lf_step < n_leapfrog; lf_step++)
        {
            // Update kappa
            log_kappa += step_size * inv_mass.array() * momentum.array();
            kappa = log_kappa.exp();

            // Update gradient
            grad = dloglike_dkappa(N, true);
            grad.array() += - (log_kappa - prior_mean_logkappa) / prior_var_logkappa;

            // Update momentum
            if (lf_step != n_leapfrog - 1)
            {
                momentum += step_size * grad;
            }
        } // for lf_step

        momentum += 0.5 * step_size * grad;
        momentum = -momentum; // Negate momentum to make proposal symmetric

        // Compute proposed energy
        double proposed_loglik = marginal_log_likelihood(N);
        double proposed_logprior = - 0.5 * (log_kappa - prior_mean_logkappa).square().matrix().sum() / prior_var_logkappa;
        double proposed_energy = -(proposed_loglik + proposed_logprior);
        double proposed_kinetic = 0.5 * momentum.cwiseProduct(inv_mass).dot(momentum);

        double H_proposed = proposed_energy + proposed_kinetic;
        double H_current = current_energy + current_kinetic;
        energy_diff = H_proposed - H_current;

        // Metropolis acceptance step
        if (!std::isfinite(energy_diff) || std::abs(energy_diff) > 100.0)
        {
            kappa = current_kappa; // revert to current state
            return 0.0;
        }
        else
        {
            if (std::log(R::runif(0.0, 1.0)) < -energy_diff)
            {
                // accept proposed state
            }
            else
            {
                // reject and revert to current state
                kappa = current_kappa;
            }

            return std::min(1.0, std::exp(-energy_diff)); // acceptance probability
        } // end Metropolis step
    } // update_kappa


    double dloglike_drho_dist(
        const Eigen::Tensor<double, 4> &N,
        const bool &add_jacobian = true
    )
    {
        double deriv = 0.0;
        for (Eigen::Index k = 0; k < ns; k++)
        { // Loop over source locations, indexed by k
            Eigen::VectorXd u_k(ns); // unnormalized weights
            for (Eigen::Index s = 0; s < ns; s++)
            { // Loop over destination locations, indexed by s
                const double mob_safe = std::max(mobility_scaled(s, k), EPS8);
                u_k(s) = std::exp(-rho_dist * dist_scaled(s, k)) * std::pow(mob_safe, rho_mobility);
            }
            double U_k = std::max(u_k.sum(), EPS8);
            Eigen::VectorXd alpha = (kappa(k) * u_k / U_k).array().max(EPS8); // ns x 1, Dirichlet parameters

            double deriv_k = 0.0;
            for (Eigen::Index s = 0; s < ns; s++)
            { // Loop over destination locations, indexed by s
                Eigen::Tensor<double, 0> N_sk = N.chip(s, 0).chip(k, 0).sum();
                
                double dw_drho_dist = - dist_scaled(s, k) * U_k + dist_scaled.col(k).dot(u_k);
                dw_drho_dist *= u_k(s) / (U_k * U_k);
                double dloglike_dalpha = Eigen::numext::digamma(alpha(s) + N_sk(0)) - Eigen::numext::digamma(alpha(s));
                deriv_k += dloglike_dalpha * dw_drho_dist;
            } // for s

            deriv += deriv_k * kappa(k);
        }

        if (add_jacobian)
        {
            deriv *= rho_dist;
        }

        return deriv;
    } // dloglike_drho_dist


    double dloglike_drho_mobility(
        const Eigen::Tensor<double, 4> &N,
        const bool &add_jacobian = true
    )
    {
        double deriv = 0.0;

        for (Eigen::Index k = 0; k < ns; k++)
        { // Loop over source locations, indexed by k
            Eigen::VectorXd u_k(ns);
            for (Eigen::Index s = 0; s < ns; s++)
            { // Loop over destination locations, indexed by s
                const double mob_safe = std::max(mobility_scaled(s, k), EPS8);
                u_k(s) = std::exp(-rho_dist * dist_scaled(s, k)) * std::pow(mob_safe, rho_mobility);
            }
            double U_k = std::max(u_k.sum(), EPS8);
            Eigen::VectorXd w_k = u_k / U_k; // ns x 1, normalized weights
            Eigen::VectorXd alpha = (kappa(k) * w_k).array().max(EPS8); // ns x 1, Dirichlet parameters

            double deriv_k = 0.0;
            for (Eigen::Index s = 0; s < ns; s++)
            { // Loop over destination locations, indexed by s
                Eigen::Tensor<double, 0> N_sk = N.chip(s, 0).chip(k, 0).sum();
                
                double mean_log_mobility = (mobility_scaled.col(k).array().max(EPS8).log() * w_k.array()).matrix().sum();
                double dw_drho_mobility = std::log(std::max(mobility_scaled(s, k), EPS8)) - mean_log_mobility;
                dw_drho_mobility *= w_k(s);
                double dloglike_dalpha = Eigen::numext::digamma(alpha(s) + N_sk(0)) - Eigen::numext::digamma(alpha(s));
                deriv_k += dloglike_dalpha * dw_drho_mobility;
            } // for s

            deriv += deriv_k * kappa(k);
        }

        if (add_jacobian)
        {
            deriv *= rho_mobility;
        }

        return deriv;
    } // dloglike_drho_mobility


    /**
     * @brief HMC sampler to update rho_dist and rho_mobility together. Operate on the unconstrained log(rho_dist) and log(rho_mobility).
     * 
     * @param energy_diff double, output energy difference between proposed and current state
     * @param grad_norm double, output gradient norm at current state
     * @param N 4D tensor of size (ns of destination) x (ns of source) x (nt + 1 of destination) x  (nt + 1 of source), unobserved secondary infections.
     * @param step_size 
     * @param n_leapfrog 
     * @param prior_mean_logrho double (default 0.0), prior mean of (log(rho_dist), log(rho_mobility))
     * @param prior_sd_logrho double (default 10.0), prior standard deviation of (log(rho_dist), log(rho_mobility))
     * @return `double` Metropolis acceptance probability 
     */
    double update_rho(
        double &energy_diff,
        double &grad_norm,
        const Eigen::Tensor<double, 4> &N,
        const double &step_size,
        const unsigned int &n_leapfrog,
        const double &prior_mean_logrho = 0.0,
        const double &prior_sd_logrho = 10.0
    )
    {
        // HMC sampler to update rho_dist and rho_mobility.
        const double prior_var_logrho = prior_sd_logrho * prior_sd_logrho;
        const Eigen::Vector2d mass_diag = Eigen::Vector2d::Constant(1.0);
        const Eigen::Vector2d inv_mass = mass_diag.cwiseInverse();
        const Eigen::Vector2d sqrt_mass = mass_diag.array().sqrt();

        // Current state
        const Eigen::Vector2d rho_current = Eigen::Vector2d(rho_dist, rho_mobility);
        Eigen::ArrayXd log_rho = rho_current.array().log();

        // Compute current energy
        double current_loglik = marginal_log_likelihood(N);
        double current_logprior = - 0.5 * (log_rho - prior_mean_logrho).square().matrix().sum() / prior_var_logrho;
        double current_energy = -(current_loglik + current_logprior);

        // Sample momentum
        Eigen::Vector2d momentum = sqrt_mass.array() * Eigen::Vector2d::NullaryExpr(2, []() { return R::rnorm(0.0, 1.0); }).array();
        double current_kinetic = 0.5 * momentum.cwiseProduct(inv_mass).dot(momentum);

        // Leapfrog integration
        Eigen::Vector2d grad;
        grad(0) = dloglike_drho_dist(N, true);
        grad(1) = dloglike_drho_mobility(N, true);
        grad.array() += - (log_rho - prior_mean_logrho) / prior_var_logrho;
        grad_norm = grad.norm();

        momentum += 0.5 * step_size * grad;
        for (unsigned int lf_step = 0; lf_step < n_leapfrog; lf_step++)
        {
            // Update rho
            log_rho += step_size * inv_mass.array() * momentum.array();
            rho_dist = std::exp(log_rho(0));
            rho_mobility = std::exp(log_rho(1));

            // Update gradient
            grad(0) = dloglike_drho_dist(N, true);
            grad(1) = dloglike_drho_mobility(N, true);
            grad.array() += - (log_rho - prior_mean_logrho) / prior_var_logrho;

            // Update momentum
            if (lf_step != n_leapfrog - 1)
            {
                momentum += step_size * grad;
            }
        } // for lf_step

        momentum += 0.5 * step_size * grad;
        momentum = -momentum; // Negate momentum to make proposal symmetric

        // Compute proposed energy
        double proposed_loglik = marginal_log_likelihood(N);
        double proposed_logprior = - 0.5 * (log_rho - prior_mean_logrho).square().matrix().sum() / prior_var_logrho;
        double proposed_energy = -(proposed_loglik + proposed_logprior);
        double proposed_kinetic = 0.5 * momentum.cwiseProduct(inv_mass).dot(momentum);

        double H_proposed = proposed_energy + proposed_kinetic;
        double H_current = current_energy + current_kinetic;
        energy_diff = H_proposed - H_current;

        // Metropolis acceptance step
        if (!std::isfinite(energy_diff) || std::abs(energy_diff) > 100.0)
        {
            rho_dist = rho_current(0);
            rho_mobility = rho_current(1);
            return 0.0;
        }
        else
        {
            if (std::log(R::runif(0.0, 1.0)) < -energy_diff)
            {
                // accept proposed state
            }
            else
            {
                // reject and revert to current state
                rho_dist = rho_current(0);
                rho_mobility = rho_current(1);
            }
            return std::min(1.0, std::exp(-energy_diff)); // acceptance probability
        } // end Metropolis step
    } // update_rho


    double update_params(
        double &energy_diff,
        double &grad_norm,
        const Eigen::Tensor<double, 4> &N,
        const double &step_size,
        const unsigned int &n_leapfrog,
        const double &prior_mean_logparams = 0.0,
        const double &prior_sd_logparams = 10.0
    )
    {
        // HMC sampler to update kappa, rho_dist, and rho_mobility all together.
        const double prior_var_logparams = prior_sd_logparams * prior_sd_logparams;
        const Eigen::VectorXd mass_diag = Eigen::VectorXd::Constant(ns + 2, 1.0);
        const Eigen::VectorXd inv_mass = mass_diag.cwiseInverse();
        const Eigen::VectorXd sqrt_mass = mass_diag.array().sqrt();

        // Current state
        Eigen::VectorXd params_current(ns + 2);
        params_current.head(ns) = kappa;
        params_current(ns) = rho_dist;
        params_current(ns + 1) = rho_mobility;
        Eigen::ArrayXd log_params = params_current.array().log();

        // Compute current energy
        double current_loglik = marginal_log_likelihood(N);
        double current_logprior = - 0.5 * (log_params - prior_mean_logparams).square().matrix().sum() / prior_var_logparams;
        double current_energy = -(current_loglik + current_logprior);

        // Sample momentum
        Eigen::VectorXd momentum = sqrt_mass.array() * Eigen::VectorXd::NullaryExpr(ns + 2, []() { return R::rnorm(0.0, 1.0); }).array();
        double current_kinetic = 0.5 * momentum.cwiseProduct(inv_mass).dot(momentum);

        // Leapfrog integration
        Eigen::VectorXd grad(ns + 2);
        grad.head(ns) = dloglike_dkappa(N, true);
        grad(ns) = dloglike_drho_dist(N, true);
        grad(ns + 1) = dloglike_drho_mobility(N, true);
        grad.array() += - (log_params - prior_mean_logparams) / prior_var_logparams;
        grad_norm = grad.norm();

        momentum += 0.5 * step_size * grad;
        for (unsigned int lf_step = 0; lf_step < n_leapfrog; lf_step++)
        {
            // Update params
            log_params += step_size * inv_mass.array() * momentum.array();
            kappa = log_params.head(ns).exp();
            rho_dist = std::exp(log_params(ns));
            rho_mobility = std::exp(log_params(ns + 1));

            // Update gradient
            grad.head(ns) = dloglike_dkappa(N, true);
            grad(ns) = dloglike_drho_dist(N, true);
            grad(ns + 1) = dloglike_drho_mobility(N, true);
            grad.array() += - (log_params - prior_mean_logparams) / prior_var_logparams;

            // Update momentum
            if (lf_step != n_leapfrog - 1)
            {
                momentum += step_size * grad;
            }
        } // for lf_step

        momentum += 0.5 * step_size * grad;
        momentum = -momentum; // Negate momentum to make proposal symmetric

        // Compute proposed energy
        double proposed_loglik = marginal_log_likelihood(N);
        double proposed_logprior = - 0.5 * (log_params - prior_mean_logparams).square().matrix().sum() / prior_var_logparams;
        double proposed_energy = -(proposed_loglik + proposed_logprior);
        double proposed_kinetic = 0.5 * momentum.cwiseProduct(inv_mass).dot(momentum);

        double H_proposed = proposed_energy + proposed_kinetic;
        double H_current = current_energy + current_kinetic;
        energy_diff = H_proposed - H_current;

        // Metropolis acceptance step
        if (!std::isfinite(energy_diff) || std::abs(energy_diff) > 100.0)
        {
            kappa = params_current.head(ns);
            rho_dist = params_current(ns);
            rho_mobility = params_current(ns + 1);
            return 0.0;
        }
        else
        {
            if (std::log(R::runif(0.0, 1.0)) < -energy_diff)
            {
                // accept proposed state
            }
            else
            {
                // reject and revert to current state
                kappa = params_current.head(ns);
                rho_dist = params_current(ns);
                rho_mobility = params_current(ns + 1);
            }
            return std::min(1.0, std::exp(-energy_diff)); // acceptance probability
        } // end Metropolis step
    } // update_params


    Rcpp::List run_mcmc(
        const Rcpp::NumericVector &N_array, // R array of size ns x ns x (nt + 1) x (nt + 1)
        const unsigned int &nburnin,
        const unsigned int &nsamples,
        const unsigned int &nthin,
        const unsigned int &hmc_block = 2, // 2: 2-block (kappa | rho_dist, rho_mobility) HMC; 1: joint HMC
        const double &step_size_init = 0.1,
        const unsigned int &n_leapfrog_init = 20
    )
    {
        Eigen::Tensor<double, 4> N = r_to_tensor4(const_cast<Rcpp::NumericVector&>(N_array));
        const Eigen::Index ntotal = static_cast<Eigen::Index>(nburnin + nsamples * nthin);

        HMCOpts_1d hmc_opts, hmc_opts_rho, hmc_opts_kappa;
        HMCDiagnostics_1d hmc_diag, hmc_diag_rho, hmc_diag_kappa;
        DualAveraging_1d da_adapter, da_adapter_rho, da_adapter_kappa;

        if (hmc_block == 1)
        {
            hmc_opts.leapfrog_step_size = step_size_init;
            hmc_opts.nleapfrog = n_leapfrog_init;

            hmc_diag = HMCDiagnostics_1d(static_cast<unsigned int>(ntotal), nburnin, true);
            da_adapter = DualAveraging_1d(hmc_opts);
        }
        else if (hmc_block == 2)
        {
            hmc_opts_rho.leapfrog_step_size = step_size_init;
            hmc_opts_rho.nleapfrog = n_leapfrog_init;
            hmc_diag_rho = HMCDiagnostics_1d(static_cast<unsigned int>(ntotal), nburnin, true);
            da_adapter_rho = DualAveraging_1d(hmc_opts_rho);

            hmc_opts_kappa.leapfrog_step_size = step_size_init;
            hmc_opts_kappa.nleapfrog = n_leapfrog_init;
            hmc_diag_kappa = HMCDiagnostics_1d(static_cast<unsigned int>(ntotal), nburnin, true);
            da_adapter_kappa = DualAveraging_1d(hmc_opts_kappa);
        }
        else
        {
            Rcpp::stop("hmc_block must be 1 or 2.");
        }

        Eigen::Tensor<double, 3> A_samples(ns, ns, nsamples);
        Eigen::MatrixXd kappa_samples(ns, nsamples);
        Eigen::VectorXd rho_dist_samples(nsamples);
        Eigen::VectorXd rho_mobility_samples(nsamples);

        Progress p(static_cast<unsigned int>(ntotal), true);
        for (Eigen::Index iter = 0; iter < ntotal; iter++)
        {
            // Update adjacency matrix A
            update_A(N);

            // Update parameters
            if (hmc_block == 1)
            {
                // Update parameters kappa, rho_dist, rho_mobility
                double energy_diff = 0.0;
                double grad_norm = 0.0;
                double accept_prob = update_params(
                    energy_diff,
                    grad_norm,
                    N,
                    hmc_opts.leapfrog_step_size,
                    hmc_opts.nleapfrog
                );

                hmc_diag.accept_count += accept_prob;
                if (hmc_opts.diagnostics)
                {
                    hmc_diag.energy_diff(iter) = energy_diff;
                    hmc_diag.grad_norm(iter) = grad_norm;
                }

                if (hmc_opts.dual_averaging)
                {
                    if (iter < nburnin)
                    {
                        hmc_opts.leapfrog_step_size = da_adapter.update_step_size(accept_prob);
                    }
                    else if (iter == nburnin)
                    {
                        da_adapter.finalize_leapfrog_step(
                            hmc_opts.leapfrog_step_size,
                            hmc_opts.nleapfrog,
                            hmc_opts.T_target
                        );
                    }

                    if (hmc_opts.diagnostics)
                    {
                        hmc_diag.leapfrog_step_size_stored(iter) = hmc_opts.leapfrog_step_size;
                        hmc_diag.nleapfrog_stored(iter) = hmc_opts.nleapfrog;
                    }
                } // if dual averaging
            } // if hmc_block == 1
            else
            {
                // Update rho_dist and rho_mobility
                double energy_diff_rho = 0.0;
                double grad_norm_rho = 0.0;
                double accept_prob_rho = update_rho(
                    energy_diff_rho,
                    grad_norm_rho,
                    N,
                    hmc_opts_rho.leapfrog_step_size,
                    hmc_opts_rho.nleapfrog
                );

                // (Optional) Update diagnostics and dual averaging for rho
                hmc_diag_rho.accept_count += accept_prob_rho;
                if (hmc_opts_rho.diagnostics)
                {
                    hmc_diag_rho.energy_diff(iter) = energy_diff_rho;
                    hmc_diag_rho.grad_norm(iter) = grad_norm_rho;
                }

                if (hmc_opts_rho.dual_averaging)
                {
                    if (iter < nburnin)
                    {
                        hmc_opts_rho.leapfrog_step_size = da_adapter_rho.update_step_size(accept_prob_rho);
                    }
                    else if (iter == nburnin)
                    {
                        da_adapter_rho.finalize_leapfrog_step(
                            hmc_opts_rho.leapfrog_step_size,
                            hmc_opts_rho.nleapfrog,
                            hmc_opts_rho.T_target
                        );
                    }

                    if (hmc_opts_rho.diagnostics)
                    {
                        hmc_diag_rho.leapfrog_step_size_stored(iter) = hmc_opts_rho.leapfrog_step_size;
                        hmc_diag_rho.nleapfrog_stored(iter) = hmc_opts_rho.nleapfrog;
                    }
                } // if dual averaging for rho

                // Update kappa
                double energy_diff_kappa = 0.0;
                double grad_norm_kappa = 0.0;
                double accept_prob_kappa = update_kappa(
                    energy_diff_kappa,
                    grad_norm_kappa,
                    N,
                    hmc_opts_kappa.leapfrog_step_size,
                    hmc_opts_kappa.nleapfrog
                );

                // (Optional) Update diagnostics and dual averaging for kappa
                hmc_diag_kappa.accept_count += accept_prob_kappa;
                if (hmc_opts_kappa.diagnostics)
                {
                    hmc_diag_kappa.energy_diff(iter) = energy_diff_kappa;
                    hmc_diag_kappa.grad_norm(iter) = grad_norm_kappa;
                }

                if (hmc_opts_kappa.dual_averaging)
                {
                    if (iter < nburnin)
                    {
                        hmc_opts_kappa.leapfrog_step_size = da_adapter_kappa.update_step_size(accept_prob_kappa);
                    }
                    else if (iter == nburnin)
                    {
                        da_adapter_kappa.finalize_leapfrog_step(
                            hmc_opts_kappa.leapfrog_step_size,
                            hmc_opts_kappa.nleapfrog,
                            hmc_opts_kappa.T_target
                        );
                    }

                    if (hmc_opts_kappa.diagnostics)
                    {
                        hmc_diag_kappa.leapfrog_step_size_stored(iter) = hmc_opts_kappa.leapfrog_step_size;
                        hmc_diag_kappa.nleapfrog_stored(iter) = hmc_opts_kappa.nleapfrog;
                    }
                } // if dual averaging for kappa
            } // else hmc_block == 2

            // Store samples after burn-in and thinning
            if (iter >= nburnin && ((iter - nburnin) % nthin == 0))
            {
                Eigen::Index sample_idx = (iter - nburnin) / nthin;
                Eigen::TensorMap<Eigen::Tensor<const double, 2>> A_map(A.data(), A.rows(), A.cols());
                A_samples.chip(sample_idx, 2) = A_map;
                kappa_samples.col(sample_idx) = kappa;
                rho_dist_samples(sample_idx) = rho_dist;
                rho_mobility_samples(sample_idx) = rho_mobility;
            } // if store samples

            p.increment();
        } // for MCMC iter


        Rcpp::List output = Rcpp::List::create(
            Rcpp::Named("A") = tensor3_to_r(A_samples), // ns x ns x nsamples
            Rcpp::Named("kappa") = kappa_samples, // ns x nsamples
            Rcpp::Named("rho_dist") = rho_dist_samples, // nsamples x 1
            Rcpp::Named("rho_mobility") = rho_mobility_samples // nsamples x 1
        );

        if (hmc_block == 1)
        {
            Rcpp::List hmc_stats = Rcpp::List::create(
                Rcpp::Named("acceptance_rate") = hmc_diag.accept_count / static_cast<double>(ntotal),
                Rcpp::Named("leapfrog_step_size") = hmc_opts.leapfrog_step_size,
                Rcpp::Named("n_leapfrog") = hmc_opts.nleapfrog,
                Rcpp::Named("diagnostics") = hmc_diag.to_list()
            );

            output["hmc"] = hmc_stats;
        } // if HMC output for hmc_block == 1
        else
        {
            Rcpp::List hmc_stats_rho = Rcpp::List::create(
                Rcpp::Named("acceptance_rate") = hmc_diag_rho.accept_count / static_cast<double>(ntotal),
                Rcpp::Named("leapfrog_step_size") = hmc_opts_rho.leapfrog_step_size,
                Rcpp::Named("n_leapfrog") = hmc_opts_rho.nleapfrog,
                Rcpp::Named("diagnostics") = hmc_diag_rho.to_list()
            );

            Rcpp::List hmc_stats_kappa = Rcpp::List::create(
                Rcpp::Named("acceptance_rate") = hmc_diag_kappa.accept_count / static_cast<double>(ntotal),
                Rcpp::Named("leapfrog_step_size") = hmc_opts_kappa.leapfrog_step_size,
                Rcpp::Named("n_leapfrog") = hmc_opts_kappa.nleapfrog,
                Rcpp::Named("diagnostics") = hmc_diag_kappa.to_list()
            );

            output["hmc"] = Rcpp::List::create(
                Rcpp::Named("rho") = hmc_stats_rho,
                Rcpp::Named("kappa") = hmc_stats_kappa
            );
        } // else HMC output for hmc_block == 2

        return output;
    } // run_mcmc
}; // class SpatialNetwork


class TemporalTransmission
{
private:
    Eigen::Index ns = 0; // number of spatial locations
    Eigen::Index nt = 0; // number of effective time points


    /**
     * @brief Compute the sum of N_kl(s, t) over all destination locations s and future times t >= l. That is, the sum of secondary infections caused by source location k at source time l to all destination locations s at times t >= l.
     * 
     * @param N 4D tensor of size (ns of destination[s]) x (ns of source[k]) x (nt + 1 of destination[t]) x  (nt + 1 of source[l]), unobserved secondary infections.
     * @param k Index of source location
     * @param l Index of source time
     * @return double 
     */
    static double compute_N_future_sum(
        const Eigen::Tensor<double, 4> &N, // N: (ns of destination[s]) x (ns of source[k]) x (nt + 1 of destination[t]) x  (nt + 1 of source[l])
        const Eigen::Index &k, // Index of source location
        const Eigen::Index &l // Index of source time
    )
    {
        Eigen::Tensor<double, 2> N_kl = N.chip(k, 1).chip(l, 2); // ns x (nt + 1)

        // Sum over destination locations s and time t >= l
        Eigen::array<Eigen::Index, 2> offsets = {0, l}; // use {0, l+1} for t > l
        Eigen::array<Eigen::Index, 2> extents = {N_kl.dimension(0), N_kl.dimension(1) - l}; // use {N_kl.dimension(0), N_kl.dimension(1) - (l + 1)} for t > l
        Eigen::Tensor<double, 0> N_kl_future = N_kl.slice(offsets, extents).sum();

        return N_kl_future(0);
    } // compute_N_future_sum


public:
    std::string fgain = "softplus";
    double W = 0.001; // temporal smoothing parameter
    Eigen::MatrixXd wt; // (nt + 1) x ns, disturbance

    TemporalTransmission()
    {
        return;
    } // TemporalTransmission default constructor


    TemporalTransmission(
        const Eigen::Index &ns_,
        const Eigen::Index &nt_,
        const std::string &fgain_ = "softplus",
        const double &W_ = 0.001
    ) : ns(ns_), nt(nt_), fgain(fgain_), W(W_)
    {
        sample_wt();
        return;
    } // TemporalTransmission constructor


    TemporalTransmission(const Rcpp::List &opts)
    {
        if (opts.containsElementNamed("ns"))
        {
            ns = static_cast<Eigen::Index>(opts["ns"]);
        }
        else
        {
            Rcpp::stop("opts must contain element 'ns' (number of spatial locations).");
        }

        if (opts.containsElementNamed("nt"))
        {
            nt = static_cast<Eigen::Index>(opts["nt"]);
        }
        else
        {
            Rcpp::stop("opts must contain element 'nt' (number of effective time points).");
        }        

        fgain = "softplus";
        if (opts.containsElementNamed("fgain"))
        {
            fgain = Rcpp::as<std::string>(opts["fgain"]);
        }

        W = 0.01;
        if (opts.containsElementNamed("W"))
        {
            W = Rcpp::as<double>(opts["W"]);
        }

        sample_wt();
        return;
    }


    void sample_wt()
    {
        const double Wsqrt = std::sqrt(W);
        wt.resize(nt + 1, ns);
        wt.setZero();
        for (Eigen::Index k = 0; k < ns; k++)
        {
            for (Eigen::Index t = 1; t < nt + 1; t++)
            {
                wt(t, k) = R::rnorm(0.0, Wsqrt);
            }
        }

        return;
    } // sample_wt


    Eigen::MatrixXd compute_Rt()
    {
        Eigen::MatrixXd Rt(nt + 1, ns);
        Rt.setZero();

        for (Eigen::Index k = 0; k < ns; k++)
        {
            // Compute cumulative sum of wt over time to get log Rt
            Eigen::VectorXd wt_cumsum = cumsum_vec(wt.col(k));

            // Then apply fgain function.
            for (Eigen::Index t = 0; t < nt + 1; t++)
            {
                Rt(t, k) = GainFunc::psi2hpsi(wt_cumsum(t), fgain);
            } // for time t
        } // for source location k

        return Rt;
    } // compute_Rt


    /**
     * @brief MH sampler to update wt.
     * 
     * @param N 
     * @param Y 
     * @param mh_sd 
     * @return `Eigen::MatrixXd` acceptance probabilities for each wt(l, k)
     */
    Eigen::MatrixXd update_wt(
        const Eigen::Tensor<double, 4> &N, // (ns of destination[s]) x (ns of source[k]) x (nt + 1 of destination[t]) x  (nt + 1 of source[l]), unobserved secondary infections
        const Eigen::MatrixXd &Y, // (nt + 1) x ns, observed primary infections
        const double &mh_sd = 1.0
    )
    {
        const double W_safe = W + EPS;

        Eigen::MatrixXd accept_prob(nt + 1, ns);
        accept_prob.setZero();
        for (Eigen::Index k = 0; k < ns; k++)
        { // Loop over source locations
            
            for (Eigen::Index l = 1; l < nt + 1; l++)
            { // Loop over source times
                const double wt_old = wt(l, k);
                Eigen::VectorXd wt_cumsum = cumsum_vec(wt.col(k));

                double mh_prec = 1.0 / W_safe;
                double logp_old = - 0.5 * (wt_old * wt_old) / W_safe;

                Eigen::VectorXd N_jk_future(nt + 1 - l);
                for (Eigen::Index j = l; j < nt + 1; j++)
                {
                    double R_jk = GainFunc::psi2hpsi(wt_cumsum(j), fgain); // R_jk = h(r_jk)
                    double lambda_jk = R_jk * Y(j, k) + EPS;

                    double dR_jk = GainFunc::psi2dhpsi(wt_cumsum(j), fgain); // dR_jk/d(r_jk)
                    double beta_jk = dR_jk * Y(j, k);
                    mh_prec += beta_jk * beta_jk / lambda_jk;

                    // N_jk: Total secondary infections produced at location k and time j to all destination locations s in the future times t >= j
                    N_jk_future(j - l) = compute_N_future_sum(N, k, j);
                    logp_old += N_jk_future(j - l) * std::log(lambda_jk) - lambda_jk;
                } // for time t >= l

                double mh_step = std::sqrt(1.0 / mh_prec) * mh_sd;
                double wt_new = R::rnorm(wt_old, mh_step);
                double logq_new_given_old = R::dnorm4(wt_new, wt_old, mh_step, true);

                wt(l, k) = wt_new;
                wt_cumsum = cumsum_vec(wt.col(k));

                double mh_prec_new = 1.0 / W_safe;
                double logp_new = - 0.5 * (wt_new * wt_new) / W_safe;
                for (Eigen::Index j = l; j < nt + 1; j++)
                {
                    double R_jk = GainFunc::psi2hpsi(wt_cumsum(j), fgain); // R_ts = h(r_ts)
                    double lambda_jk = R_jk * Y(j, k) + EPS;

                    double dR_jk = GainFunc::psi2dhpsi(wt_cumsum(j), fgain); // dR_jk/d(r_jk)
                    double beta_jk = dR_jk * Y(j, k);
                    mh_prec_new += beta_jk * beta_jk / lambda_jk;

                    logp_new += N_jk_future(j - l) * std::log(lambda_jk) - lambda_jk;
                } // for time t >= l

                double mh_step_new = std::sqrt(1.0 / mh_prec_new) * mh_sd;
                double logq_old_given_new = R::dnorm4(wt_old, wt_new, mh_step_new, true);

                double logratio = std::min(
                    0.0, logp_new - logp_old + logq_old_given_new - logq_new_given_old
                );
                if (!std::isfinite(logratio))
                {
                    // reject and revert
                    wt(l, k) = wt_old;
                    accept_prob(l, k) = 0.0;
                }
                else
                {
                    if (std::log(R::runif(0.0, 1.0)) < logratio)
                    {
                        // accept
                    }
                    else
                    {
                        // reject and revert
                        wt(l, k) = wt_old;
                    }

                    accept_prob(l, k) = std::min(1.0, std::exp(logratio));
                }
            } // for source time l
        } // for source location k

        return accept_prob;
    } // update_wt


    /**
     * @brief Gibbs sampler to update W.
     * 
     * @param prior_shape 
     * @param prior_rate 
     */
    void update_W(const double &prior_shape = 1.0, const double &prior_rate = 1.0)
    {
        // wt: (nt + 1) x ns, row 0 is fixed (not part of prior/likelihood)
        const double n_effective = static_cast<double>(ns * nt);

        // sum of squares over t = 1..nt only
        double ssq = wt.block(1, 0, nt, ns).array().square().sum();
        // equivalently: double ssq = wt.bottomRows(nt).array().square().sum();

        double posterior_shape = prior_shape + 0.5 * n_effective;
        double posterior_rate = prior_rate + 0.5 * ssq;

        // Inverse-Gamma(shape, rate) via gamma on precision
        W = 1.0 / R::rgamma(posterior_shape, 1.0 / posterior_rate);

        return;
    } // update_W


    Rcpp::List run_mcmc(
        const Rcpp::NumericVector &N_array, // R array of size ns x ns x (nt + 1) x (nt + 1)
        const Eigen::MatrixXd &Y, // matrix of size (nt + 1) x ns
        const unsigned int &nburnin,
        const unsigned int &nsamples,
        const unsigned int &nthin,
        const double &mh_sd = 1.0,
        const double &prior_shape_W = 1.0,
        const double &prior_rate_W = 1.0
    )
    {
        Eigen::Tensor<double, 4> N = r_to_tensor4(const_cast<Rcpp::NumericVector&>(N_array));
        const Eigen::Index ntotal = static_cast<Eigen::Index>(nburnin + nsamples * nthin);

        Eigen::Tensor<double, 3> wt_samples(nt + 1, ns, nsamples);
        Eigen::VectorXd W_samples(nsamples);
        Eigen::MatrixXd wt_accept_prob(nt + 1, ns);

        Progress p(static_cast<unsigned int>(ntotal), true);
        for (Eigen::Index iter = 0; iter < ntotal; iter++)
        {
            // Update wt
            Eigen::MatrixXd accept_prob = update_wt(N, Y, mh_sd);
            wt_accept_prob += accept_prob;

            // Update W
            update_W(prior_shape_W, prior_rate_W);

            // Store samples after burn-in and thinning
            if (iter >= nburnin && ((iter - nburnin) % nthin == 0))
            {
                Eigen::Index sample_idx = (iter - nburnin) / nthin;
                Eigen::TensorMap<Eigen::Tensor<const double, 2>> wt_map(wt.data(), wt.rows(), wt.cols());
                wt_samples.chip(sample_idx, 2) = wt_map;
                W_samples(sample_idx) = W;
            } // if store samples

            p.increment(); 
        } // for MCMC iter

        return Rcpp::List::create(
            Rcpp::Named("wt") = tensor3_to_r(wt_samples), // (nt + 1) x ns x nsamples
            Rcpp::Named("W") = W_samples, // nsamples x 1
            Rcpp::Named("wt_accept_prob") = wt_accept_prob / static_cast<double>(ntotal)
        );
    } // run_mcmc
}; // class TemporalTransmission


class Model
{
private:
    Eigen::Index ns = 0; // number of spatial locations
    Eigen::Index nt = 0; // number of effective time points
public:
    /* Known model properties */
    LagDist dlag;

    /* Unknown model components */
    Eigen::Tensor<double, 4> N; // unobserved secondary infections
    SpatialNetwork spatial; // stochastic spatial network
    TemporalTransmission temporal; // temporal transmission model for reproduction numbers


    Model()
    {
        dlag = LagDist("lognorm", LN_MU, LN_SD2, true);
        return;
    } // Model default constructor


    Model(
        const Eigen::MatrixXd &Y, // (nt + 1) x ns, observed primary infections
        const Eigen::MatrixXd &dist_matrix, // ns x ns, pairwise distance matrix between spatial locations
        const Eigen::MatrixXd &mobility_matrix, // ns x ns, pairwise mobility matrix between spatial locations
        const std::string &fgain_ = "softplus",
        const Rcpp::List &lagdist_opts = Rcpp::List::create(
            Rcpp::Named("name") = "lognorm",
            Rcpp::Named("par1") = LN_MU,
            Rcpp::Named("par2") = LN_SD2,
            Rcpp::Named("truncated") = true,
            Rcpp::Named("rescaled") = true
        )
    )
    {
        ns = dist_matrix.rows();
        nt = Y.rows() - 1;

        dlag = LagDist(lagdist_opts);
        temporal = TemporalTransmission(ns, nt, fgain_);
        spatial = SpatialNetwork(dist_matrix, mobility_matrix, true);

        sample_N(Y);
    } // Model constructor for MCMC inference when model parameters are unknown


    Model(
        const Eigen::Index &nt_,
        const Eigen::MatrixXd &dist_matrix, // ns x ns, pairwise distance matrix between spatial locations
        const Eigen::MatrixXd &mobility_matrix, // ns x ns, pairwise mobility matrix between spatial locations
        const std::string &fgain_ = "softplus",
        const double &W_ = 0.001,
        const Rcpp::List &spatial_opts = Rcpp::List::create(
            Rcpp::Named("rho_dist") = 1.0,
            Rcpp::Named("rho_mobility") = 1.0,
            Rcpp::Named("kappa") = 1.0
        ),
        const Rcpp::List &lagdist_opts = Rcpp::List::create(
            Rcpp::Named("name") = "lognorm",
            Rcpp::Named("par1") = LN_MU,
            Rcpp::Named("par2") = LN_SD2,
            Rcpp::Named("truncated") = true,
            Rcpp::Named("rescaled") = true
        )
    )
    {
        nt = nt_;
        ns = dist_matrix.rows();

        dlag = LagDist(lagdist_opts);
        temporal = TemporalTransmission(ns, nt, fgain_, W_);
        spatial = SpatialNetwork(dist_matrix, mobility_matrix, spatial_opts);

        N.resize(ns, ns, nt + 1, nt + 1);
        N.setZero();
        return;
    } // Model constructor for simulation when model parameters are known but Y is to be simulated


    void sample_N(const Eigen::MatrixXd &Y) // Y: (nt + 1) x ns, observed primary infections
    {
        if (N.size() == 0)
        {
            N.resize(ns, ns, nt + 1, nt + 1);
            N.setZero();
        }

        for (Eigen::Index k = 0; k < ns; k++)
        { // Loop over source locations
            Eigen::VectorXd wt_cumsum = cumsum_vec(temporal.wt.col(k));

            for (Eigen::Index l = 0; l < nt + 1; l++)
            { // Loop over source times
                double R_lk = GainFunc::psi2hpsi(wt_cumsum(l), temporal.fgain); // reproduction number at time l and location k

                for (Eigen::Index s = 0; s < ns; s++)
                { // Loop over destination locations
                    double spatial_weight = spatial.A(s, k);

                    for (Eigen::Index t = l + 1; t < nt + 1; t++)
                    { // Loop over destination times t > l
                        if (t - l < dlag.Fphi.size())
                        {
                            double lag_prob = dlag.Fphi(t - l); // lag probability for lag (t - l)
                            double lambda_sktl = spatial_weight * R_lk * lag_prob * Y(l, k) + EPS;
                            N(s, k, t, l) = R::rpois(lambda_sktl);
                        }
                        else
                        {
                            N(s, k, t, l) = 0.0;
                        }
                    } // for destination time t >= l
                } // for destination location s
            } // for source time l
        } // for source location k

        return;
    } // sample_N



    Rcpp::List simulate()
    {
        if (N.size() == 0)
        {
            N.resize(ns, ns, nt + 1, nt + 1);
        }
        N.setZero();

        spatial.sample_A();
        temporal.sample_wt();
        const Eigen::MatrixXd Rt = temporal.compute_Rt(); // (nt + 1) x ns

        // Simulate primary infections at time t = 0
        Eigen::MatrixXd Y(nt + 1, ns);
        Y.setZero();
        for (Eigen::Index k = 0; k < ns; k++)
        {
            Y(0, k) = R::rpois(spatial.kappa(k) + EPS);
        }

        // Simulate secondary infections N and observed primary infections Y over time
        for (Eigen::Index t = 1; t < nt + 1; t++)
        { // Loop over destination time t

            // Compute new primary infections at time t
            for (Eigen::Index s = 0; s < ns; s++)
            { // Loop over destination location s

                for (Eigen::Index k = 0; k < ns; k++)
                { // Loop over source location k

                    double a_sk = spatial.A(s, k);
                    for (Eigen::Index l = 0; l < t; l++)
                    { // Loop over source time l < t

                        if (t - l < dlag.Fphi.size())
                        {
                            double lag_prob = dlag.Fphi(t - l); // lag probability for lag (t - l)
                            double R_kl = Rt(l, k); // reproduction number at time l and location k
                            double lambda_sktl = a_sk * R_kl * lag_prob * Y(l, k) + EPS;
                            N(s, k, t, l) = R::rpois(std::max(lambda_sktl, 0.1));

                        }
                        else
                        {
                            N(s, k, t, l) = 0.0;
                        }

                        Y(t, s) += N(s, k, t, l);
                    } // for source time l < t
                } // for source location k

            } // for destination location s
        } // for time t

        return Rcpp::List::create(
            Rcpp::Named("Y") = Y, // (nt + 1) x ns
            Rcpp::Named("N") = tensor4_to_r(N), // ns x ns x (nt + 1) x (nt + 1)
            Rcpp::Named("A") = spatial.A, // ns x ns
            Rcpp::Named("wt") = temporal.wt, // (nt + 1) x ns
            Rcpp::Named("Rt") = Rt // (nt + 1) x ns
        );

    } // simulate


    /**
     * @brief Giibs sampler to update unobserved secondary infections N from multinomial distribution.
     * 
     * @param Y 
     */
    void update_N(const Eigen::MatrixXd &Y) // Y: (nt + 1) x ns, observed primary infections
    {
        // Gibbs sampler to update unobserved secondary infections N.
        Eigen::MatrixXd Rt = temporal.compute_Rt(); // (nt + 1) x ns

        for (Eigen::Index s = 0; s < ns; s++)
        { // Loop over destination locations

            for (Eigen::Index t = 1; t < nt + 1; t++)
            { // Loop over destination times

                Eigen::MatrixXd p_st(nt + 1, ns); // (source times l) x (source locations k)
                p_st.setZero();

                double lambda_st = 0.0;
                for (Eigen::Index k = 0; k < ns; k++)
                { // Loop over source locations

                    double a_sk = spatial.A(s, k);
                    for (Eigen::Index l = 0; l < t; l++)
                    { // Loop over source times l < t

                        if (t - l < dlag.Fphi.size())
                        {
                            double lag_prob = dlag.Fphi(t - l); // lag probability for lag (t - l)
                            p_st(l, k) = a_sk * Rt(l, k) * lag_prob * Y(l, k);
                            lambda_st += p_st(l, k);
                        }
                    } // for source time l < t
                } // for source location k

                const double y_st = Y(t, s);
                if (y_st <= 0.0)
                {
                    // no observed cases to allocate for this (s, t)
                    for (Eigen::Index k = 0; k < ns; k++)
                    {
                        for (Eigen::Index l = 0; l < nt + 1; l++)
                        {
                            N(s, k, t, l) = 0.0;
                        }
                    } // for source location k and source time l
                    continue;
                }

                // If total intensity is numerically tiny but y_st > 0, fall back to uniform allocation over valid (k, l < t)
                if (lambda_st < EPS)
                {
                    const double uniform_prob = 1.0 / static_cast<double>(ns * t);
                    for (Eigen::Index k = 0; k < ns; k++)
                    {
                        for (Eigen::Index l = 0; l < t; l++)
                        {
                            p_st(l, k) = uniform_prob;
                        }
                    }
                    lambda_st = 1.0;
                }

                p_st /= lambda_st; // normalize to get probabilities

                // Flatten probabilities into a vector for rmultinom
                const Eigen::Index K = ns * t; // only l < t are valid
                std::vector<double> prob(static_cast<std::size_t>(K), 0.0);
                std::size_t idx = 0;
                for (Eigen::Index k = 0; k < ns; k++)
                {
                    for (Eigen::Index l = 0; l < t; l++)
                    {
                        prob[idx++] = p_st(l, k);
                    }
                } // for source location k and source time l < t

                std::vector<int> counts(K, 0);
                int y_count = static_cast<int>(std::lround(y_st));
                y_count = std::max(0, y_count);
                R::rmultinom(y_count, prob.data(), static_cast<int>(K), counts.data());

                // Write samples back to N and zero out impossible l >= t cells
                idx = 0;
                for (Eigen::Index k = 0; k < ns; k++)
                { // Loop over source locations
                    for (Eigen::Index l = 0; l < nt + 1; l++)
                    { // Loop over source times
                        if (l < t)
                        {
                            N(s, k, t, l) = static_cast<double>(counts[idx++]);
                        }
                        else
                        {
                            N(s, k, t, l) = 0.0;
                        }
                    } // for source time l
                } // for source location k

            } // for destination time t
        } // for destination location s

        return;
    } // update_N


    Rcpp::List run_mcmc(
        const Eigen::MatrixXd &Y, // (nt + 1) x ns, observed primary infections
        const unsigned int &nburnin,
        const unsigned int &nsamples,
        const unsigned int &nthin,
        const double &mh_sd_wt = 1.0,
        const double &prior_shape_W = 1.0,
        const double &prior_rate_W = 1.0,
        const unsigned int &spatial_hmc_block = 2, // 2: 2-block (kappa | rho_dist, rho_mobility) HMC; 1: joint HMC
        const double &spatial_hmc_step_size = 0.1,
        const unsigned int &spatial_hmc_nleapfrog = 20
    )
    {
        const Eigen::Index ntotal = static_cast<Eigen::Index>(nburnin + nsamples * nthin);

        HMCOpts_1d hmc_opts, hmc_opts_rho, hmc_opts_kappa;
        HMCDiagnostics_1d hmc_diag, hmc_diag_rho, hmc_diag_kappa;
        DualAveraging_1d da_adapter, da_adapter_rho, da_adapter_kappa;

        if (spatial_hmc_block == 1)
        {
            hmc_opts.leapfrog_step_size = spatial_hmc_step_size;
            hmc_opts.nleapfrog = spatial_hmc_nleapfrog;

            hmc_diag = HMCDiagnostics_1d(static_cast<unsigned int>(ntotal), nburnin, true);
            da_adapter = DualAveraging_1d(hmc_opts);
        }
        else if (spatial_hmc_block == 2)
        {
            hmc_opts_rho.leapfrog_step_size = spatial_hmc_step_size;
            hmc_opts_rho.nleapfrog = spatial_hmc_nleapfrog;
            hmc_diag_rho = HMCDiagnostics_1d(static_cast<unsigned int>(ntotal), nburnin, true);
            da_adapter_rho = DualAveraging_1d(hmc_opts_rho);

            hmc_opts_kappa.leapfrog_step_size = spatial_hmc_step_size;
            hmc_opts_kappa.nleapfrog = spatial_hmc_nleapfrog;
            hmc_diag_kappa = HMCDiagnostics_1d(static_cast<unsigned int>(ntotal), nburnin, true);
            da_adapter_kappa = DualAveraging_1d(hmc_opts_kappa);
        }
        else
        {
            Rcpp::stop("spatial_hmc_block must be 1 or 2.");
        }

        Eigen::MatrixXd accept_prob_wt(nt + 1, ns);
        accept_prob_wt.setZero();

        Eigen::Tensor<double, 3> A_samples(ns, ns, nsamples);
        Eigen::MatrixXd kappa_samples(ns, nsamples);
        Eigen::VectorXd rho_dist_samples(nsamples);
        Eigen::VectorXd rho_mobility_samples(nsamples);
        Eigen::Tensor<double, 3> wt_samples(nt + 1, ns, nsamples);
        Eigen::VectorXd W_samples(nsamples);

        Progress p(static_cast<unsigned int>(ntotal), true);
        for (Eigen::Index iter = 0; iter < ntotal; iter++)
        {
            // Update unobserved secondary infections N
            update_N(Y);

            // Update temporal transmission components
            Eigen::MatrixXd accept_prob = temporal.update_wt(N, Y, mh_sd_wt);
            accept_prob_wt += accept_prob;

            temporal.update_W(prior_shape_W, prior_rate_W);

            // Update spatial network components
            if (spatial_hmc_block == 1)
            {
                double energy_diff = 0.0;
                double grad_norm = 0.0;
                double accept_prob = spatial.update_params(
                    energy_diff,
                    grad_norm,
                    N,
                    hmc_opts.leapfrog_step_size,
                    hmc_opts.nleapfrog
                );

                hmc_diag.accept_count += accept_prob;
                if (hmc_opts.diagnostics)
                {
                    hmc_diag.energy_diff(iter) = energy_diff;
                    hmc_diag.grad_norm(iter) = grad_norm;
                }

                if (hmc_opts.dual_averaging)
                {
                    if (iter < nburnin)
                    {
                        hmc_opts.leapfrog_step_size = da_adapter.update_step_size(accept_prob);
                    }
                    else if (iter == nburnin)
                    {
                        da_adapter.finalize_leapfrog_step(
                            hmc_opts.leapfrog_step_size,
                            hmc_opts.nleapfrog,
                            hmc_opts.T_target
                        );
                    }

                    if (hmc_opts.diagnostics)
                    {
                        hmc_diag.leapfrog_step_size_stored(iter) = hmc_opts.leapfrog_step_size;
                        hmc_diag.nleapfrog_stored(iter) = hmc_opts.nleapfrog;
                    }
                } // if dual averaging
            } // if spatial_hmc_block == 1
            else
            {
                // Update rho_dist and rho_mobility
                double energy_diff_rho = 0.0;
                double grad_norm_rho = 0.0;
                double accept_prob_rho = spatial.update_rho(
                    energy_diff_rho,
                    grad_norm_rho,
                    N,
                    hmc_opts_rho.leapfrog_step_size,
                    hmc_opts_rho.nleapfrog
                );

                // (Optional) Update diagnostics and dual averaging for rho
                hmc_diag_rho.accept_count += accept_prob_rho;
                if (hmc_opts_rho.diagnostics)
                {
                    hmc_diag_rho.energy_diff(iter) = energy_diff_rho;
                    hmc_diag_rho.grad_norm(iter) = grad_norm_rho;
                }

                if (hmc_opts_rho.dual_averaging)
                {
                    if (iter < nburnin)
                    {
                        hmc_opts_rho.leapfrog_step_size = da_adapter_rho.update_step_size(accept_prob_rho);
                    }
                    else if (iter == nburnin)
                    {
                        da_adapter_rho.finalize_leapfrog_step(
                            hmc_opts_rho.leapfrog_step_size,
                            hmc_opts_rho.nleapfrog,
                            hmc_opts_rho.T_target
                        );
                    }

                    if (hmc_opts_rho.diagnostics)
                    {
                        hmc_diag_rho.leapfrog_step_size_stored(iter) = hmc_opts_rho.leapfrog_step_size;
                        hmc_diag_rho.nleapfrog_stored(iter) = hmc_opts_rho.nleapfrog;
                    }
                } // if dual averaging for rho

                // Update kappa
                double energy_diff_kappa = 0.0;
                double grad_norm_kappa = 0.0;
                double accept_prob_kappa = spatial.update_kappa(
                    energy_diff_kappa,
                    grad_norm_kappa,
                    N,
                    hmc_opts_kappa.leapfrog_step_size,
                    hmc_opts_kappa.nleapfrog
                );

                // (Optional) Update diagnostics and dual averaging for kappa
                hmc_diag_kappa.accept_count += accept_prob_kappa;
                if (hmc_opts_kappa.diagnostics)
                {
                    hmc_diag_kappa.energy_diff(iter) = energy_diff_kappa;
                    hmc_diag_kappa.grad_norm(iter) = grad_norm_kappa;
                }

                if (hmc_opts_kappa.dual_averaging)
                {
                    if (iter < nburnin)
                    {
                        hmc_opts_kappa.leapfrog_step_size = da_adapter_kappa.update_step_size(accept_prob_kappa);
                    }
                    else if (iter == nburnin)
                    {
                        da_adapter_kappa.finalize_leapfrog_step(
                            hmc_opts_kappa.leapfrog_step_size,
                            hmc_opts_kappa.nleapfrog,
                            hmc_opts_kappa.T_target
                        );
                    }

                    if (hmc_opts_kappa.diagnostics)
                    {
                        hmc_diag_kappa.leapfrog_step_size_stored(iter) = hmc_opts_kappa.leapfrog_step_size;
                        hmc_diag_kappa.nleapfrog_stored(iter) = hmc_opts_kappa.nleapfrog;
                    }
                } // if dual averaging for kappa
            } // else spatial_hmc_block == 2


            // Update adjacency matrix A
            spatial.update_A(N);


            // Store samples after burn-in and thinning
            if (iter >= nburnin && ((iter - nburnin) % nthin == 0))
            {
                Eigen::Index sample_idx = (iter - nburnin) / nthin;
                Eigen::TensorMap<Eigen::Tensor<const double, 2>> A_map(spatial.A.data(), spatial.A.rows(), spatial.A.cols());
                A_samples.chip(sample_idx, 2) = A_map;
                kappa_samples.col(sample_idx) = spatial.kappa;
                rho_dist_samples(sample_idx) = spatial.rho_dist;
                rho_mobility_samples(sample_idx) = spatial.rho_mobility;
                Eigen::TensorMap<Eigen::Tensor<const double, 2>> wt_map(temporal.wt.data(), temporal.wt.rows(), temporal.wt.cols());
                wt_samples.chip(sample_idx, 2) = wt_map;
                W_samples(sample_idx) = temporal.W;
            } // if store samples

            p.increment();
        } // for MCMC iter

        Rcpp::List hmc_stats;
        if (spatial_hmc_block == 1)
        {
            hmc_stats = Rcpp::List::create(
                Rcpp::Named("hmc_block") = 1,
                Rcpp::Named("acceptance_rate") = hmc_diag.accept_count / static_cast<double>(ntotal),
                Rcpp::Named("leapfrog_step_size") = hmc_opts.leapfrog_step_size,
                Rcpp::Named("n_leapfrog") = hmc_opts.nleapfrog,
                Rcpp::Named("diagnostics") = hmc_diag.to_list()
            );
        } // if HMC output for hmc_block == 1
        else
        {
            Rcpp::List hmc_stats_rho = Rcpp::List::create(
                Rcpp::Named("acceptance_rate") = hmc_diag_rho.accept_count / static_cast<double>(ntotal),
                Rcpp::Named("leapfrog_step_size") = hmc_opts_rho.leapfrog_step_size,
                Rcpp::Named("n_leapfrog") = hmc_opts_rho.nleapfrog,
                Rcpp::Named("diagnostics") = hmc_diag_rho.to_list()
            );

            Rcpp::List hmc_stats_kappa = Rcpp::List::create(
                Rcpp::Named("acceptance_rate") = hmc_diag_kappa.accept_count / static_cast<double>(ntotal),
                Rcpp::Named("leapfrog_step_size") = hmc_opts_kappa.leapfrog_step_size,
                Rcpp::Named("n_leapfrog") = hmc_opts_kappa.nleapfrog,
                Rcpp::Named("diagnostics") = hmc_diag_kappa.to_list()
            );

            hmc_stats = Rcpp::List::create(
                Rcpp::Named("hmc_block") = 2,
                Rcpp::Named("rho") = hmc_stats_rho,
                Rcpp::Named("kappa") = hmc_stats_kappa
            );
        } // else HMC output for hmc_block == 2


        return Rcpp::List::create(
            Rcpp::Named("spatial") = Rcpp::List::create(
                Rcpp::Named("A") = tensor3_to_r(A_samples), // ns x ns x nsamples
                Rcpp::Named("kappa") = kappa_samples, // ns x nsamples
                Rcpp::Named("rho_dist") = rho_dist_samples, // nsamples x 1
                Rcpp::Named("rho_mobility") = rho_mobility_samples, // nsamples x 1
                Rcpp::Named("hmc_stats") = hmc_stats
            ),
            Rcpp::Named("temporal") = Rcpp::List::create(
                Rcpp::Named("wt") = tensor3_to_r(wt_samples), // (nt + 1) x ns x nsamples
                Rcpp::Named("W") = W_samples, // nsamples x 1
                Rcpp::Named("wt_accept_prob") = accept_prob_wt / static_cast<double>(ntotal) // (nt + 1) x ns
            )
        );

    } // run_mcmc

}; // class Model



#endif
