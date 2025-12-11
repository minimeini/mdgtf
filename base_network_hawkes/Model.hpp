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


inline Rcpp::NumericVector tensor5_to_r(const Eigen::Tensor<double, 5> &t)
{
    Eigen::array<Eigen::Index, 5> dims = t.dimensions();
    Rcpp::NumericVector out(dims[0] * dims[1] * dims[2] * dims[3] * dims[4]);
    std::copy(t.data(), t.data() + out.size(), out.begin());
    out.attr("dim") = Rcpp::IntegerVector::create(
        dims[0], dims[1], dims[2], dims[3], dims[4]);
    return out;
}; // tensor5_to_r


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
        if (ns == 1)
        {
            Eigen::VectorXd sample(1);
            sample(0) = 1.0;
            return sample;
        }
        else
        {
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
        }
    } // sample_A_col


public:
    Eigen::MatrixXd dist_scaled; // ns x ns, pairwise scaled distance matrix
    Eigen::MatrixXd mobility_scaled; // ns x ns, pairwise scaled mobility matrix
    Eigen::MatrixXd alpha; // ns x ns, adjacency matrix / stochastic network
    double rho_dist = 1.0; // distance decay parameter
    double rho_mobility = 1.0; // mobility scaling parameter


    SpatialNetwork()
    {
        ns = 1;
        alpha.resize(ns, ns);
        // kappa.resize(ns);
        for (Eigen::Index k = 0; k < ns; k++)
        { // Loop over source locations
            alpha.col(k) = compute_alpha_col(k);
        }
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

        if (random_init)
        {
            rho_dist = R::runif(0.0, 2.0);
            rho_mobility = R::runif(0.0, 2.0);
        }

        alpha.resize(ns, ns);
        // kappa.resize(ns);
        for (Eigen::Index k = 0; k < ns; k++)
        { // Loop over source locations
            alpha.col(k) = compute_alpha_col(k);
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

        // kappa.resize(ns);
        // kappa.setOnes();

        if (settings.containsElementNamed("rho_dist"))
        {
            rho_dist = Rcpp::as<double>(settings["rho_dist"]);
        } // if rho_dist
        if (settings.containsElementNamed("rho_mobility"))
        {
            rho_mobility = Rcpp::as<double>(settings["rho_mobility"]);
        } // if rho_mobility
        // if (settings.containsElementNamed("kappa"))
        // {
        //     Eigen::VectorXd kappa_in = Rcpp::as<Eigen::VectorXd>(settings["kappa"]);
        //     if (kappa_in.size() < ns)
        //     {
        //         kappa.head(kappa_in.size()) = kappa_in;
        //         kappa.tail(ns - kappa_in.size()).fill(kappa_in.tail(1)(0));
        //     }
        //     else
        //     {
        //         kappa = kappa_in.head(ns);
        //     }
        // } // if kappa

        alpha.resize(ns, ns);
        for (Eigen::Index k = 0; k < ns; k++)
        { // Loop over source locations
            alpha.col(k) = compute_alpha_col(k);
        }
        return;
    } // SpatialNetwork constructor


    Eigen::VectorXd compute_alpha_col(const Eigen::Index &k) const
    {
        if (ns == 1)
        {
            Eigen::VectorXd alpha(1);
            alpha(0) = 1.0;
            return alpha;
        }
        else
        {
            Eigen::VectorXd u(ns); // unnormalized weights
            for (Eigen::Index s = 0; s < ns; s++)
            { // Loop over destinations
                const double mob_safe = std::max(mobility_scaled(s, k), EPS8);
                u(s) = std::exp(-rho_dist * dist_scaled(s, k)) *
                       std::pow(mob_safe, rho_mobility);
            } // for s

            return (u / u.sum()).array().max(EPS8);
        }
    } // compute_alpha_col


    void compute_alpha()
    {
        for (Eigen::Index k = 0; k < ns; k++)
        { // Loop over source locations
            alpha.col(k) = compute_alpha_col(k);
        }
        return;
    }


    double dalpha_drho_dist(
        const Eigen::Index &s,
        const Eigen::Index &k, // source location index
        const Eigen::VectorXd &u_k // unnormalized weights for source k
    ) const
    {
        double U_k = std::max(u_k.sum(), EPS8);
        double deriv = -dist_scaled(s, k) * U_k + dist_scaled.col(k).dot(u_k);
        deriv *= u_k(s) / (U_k * U_k);
        return deriv;
    } // dalpha_drho_dist


    double dalpha_drho_mobility(
        const Eigen::Index &s,
        const Eigen::Index &k
    ) const
    {
        double mean_log_mobility = (mobility_scaled.col(k).array().max(EPS8).log() * alpha.col(k).array()).matrix().sum();
        double deriv = std::log(std::max(mobility_scaled(s, k), EPS8)) - mean_log_mobility;
        deriv *= alpha(s, k);
        return deriv;
    } // dalpha_drho_mobility


    // /**
    //  * @brief Sample adjacency matrix A where each column A[, j] ~ Dirichlet(alpha[, j])
    //  *
    //  */
    // void sample_A()
    // {
    //     if (ns == 1)
    //     {
    //         return;
    //     }
    //     else
    //     {
    //         for (Eigen::Index k = 0; k < ns; k++)
    //         { // Loop over source locations
    //             Eigen::VectorXd alpha = compute_alpha_col(k);
    //             A.col(k) = sample_A_col(alpha);
    //         }
    //         return;
    //     } // Prior samples of A when ns > 1
    // } // sample_A


    // /**
    //  * @brief Gibbs sampler of adjacency matrix A given secondary infection tensor N
    //  * 
    //  * @param N 
    //  * @todo Checked. OK.
    //  */
    // void update_A(
    //     const Eigen::Tensor<double, 4> &N // (ns of destination) x (ns of source) x (nt + 1 of destination) x  (nt + 1 of source)
    // )
    // {
    //     if (ns == 1)
    //     {
    //         return;
    //     }
    //     else
    //     {
    //         for (Eigen::Index k = 0; k < ns; k++)
    //         { // Loop over source locations
    //             // Compute alpha (prior parameters)
    //             Eigen::VectorXd alpha_post = compute_alpha_col(k);

    //             for (Eigen::Index s = 0; s < ns; s++)
    //             { // Loop over destination locations
    //                 Eigen::Tensor<double, 0> N_sk = N.chip(s, 0).chip(k, 0).sum();
    //                 alpha_post(s) += N_sk(0);
    //             }

    //             // Sample A.col(k) ~ Dirichlet(alpha_post)
    //             A.col(k) = sample_A_col(alpha_post);
    //         }
    //         return;
    //     } // Posterior samples of A when ns > 1
    // } // update_A.


    // /**
    //  * @brief Derivative of log-likelihood with respect to kappa.
    //  * 
    //  * @param N 4D tensor (ns of destination) x (ns of source) x (nt + 1 of destination) x  (nt + 1 of source)
    //  * @param add_jacobian Add Jacobian of kappa w.r.t log(kappa) if true
    //  * @return `Eigen::VectorXd` ns x 1, derivative of log-likelihood w.r.t kappa or log(kappa).
    //  */
    // Eigen::VectorXd dloglike_dkappa(
    //     const Eigen::Tensor<double, 4> &N,
    //     const bool &add_jacobian = true)
    // {
    //     Eigen::VectorXd grad(ns);

    //     for (Eigen::Index k = 0; k < ns; ++k)
    //     {
    //         // N_k: total secondaries from source k
    //         Eigen::Tensor<double, 0> N_k = N.chip(k, 1).sum();

    //         double term_global = Eigen::numext::digamma(kappa(k)) - Eigen::numext::digamma(kappa(k) + N_k(0));

    //         Eigen::VectorXd alpha = compute_alpha_col(k);
    //         double term_alpha = 0.0;
    //         for (Eigen::Index s = 0; s < ns; ++s)
    //         {
    //             Eigen::Tensor<double, 0> N_sk = N.chip(s, 0).chip(k, 0).sum();

    //             double w_sk = alpha(s) / kappa(k);
    //             double g_sk = Eigen::numext::digamma(alpha(s) + N_sk(0)) - Eigen::numext::digamma(alpha(s));

    //             term_alpha += g_sk * w_sk;
    //         } // for destination s

    //         grad(k) = term_global + term_alpha;

    //         if (add_jacobian)
    //         {
    //             grad(k) *= kappa(k);
    //         }
    //     } // for source location k

    //     return grad;
    // } // dloglike_dkappa


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
            loglike -= Eigen::numext::lgamma(1.0 + N_k(0));

            for (Eigen::Index s = 0; s < ns; s++)
            { // Loop over destination locations
                Eigen::Tensor<double, 0> N_sk = N.chip(s, 0).chip(k, 0).sum();
                loglike += Eigen::numext::lgamma(alpha(s) + N_sk(0)) - Eigen::numext::lgamma(alpha(s));
            }
        }

        return loglike;
    }


    // /**
    //  * @brief HMC sampler to update kappa. Operate on the unconstrained log(kappa).
    //  * 
    //  * @param energy_diff double, output energy difference between proposed and current state
    //  * @param grad_norm double, output gradient norm at current state
    //  * @param N 4D tensor of size (ns of destination) x (ns of source) x (nt + 1 of destination) x  (nt + 1 of source), unobserved secondary infections.
    //  * @param step_size 
    //  * @param n_leapfrog 
    //  * @param prior_mean_logkappa double (default 0.0), prior mean of log(kappa)
    //  * @param prior_sd_logkappa double (default 10.0), prior standard deviation of log(kappa)
    //  * @return `double` Metropolis acceptance probability
    //  * @todo Precondition mass matrix
    //  */
    // double update_kappa(
    //     double &energy_diff,
    //     double &grad_norm,
    //     const Eigen::Tensor<double, 4> &N,
    //     const double &step_size,
    //     const Eigen::Index &n_leapfrog,
    //     const double &prior_mean_logkappa = 0.0,
    //     const double &prior_sd_logkappa = 10.0
    // )
    // {
    //     const double prior_var_logkappa = prior_sd_logkappa * prior_sd_logkappa;
    //     Eigen::VectorXd mass_diag = Eigen::VectorXd::Constant(ns, 1.0);
    //     Eigen::VectorXd inv_mass = mass_diag.cwiseInverse();
    //     Eigen::VectorXd sqrt_mass = mass_diag.array().sqrt();

    //     // Current state
    //     Eigen::VectorXd current_kappa = kappa;
    //     Eigen::ArrayXd log_kappa = kappa.array().log();

    //     // Compute current energy
    //     double current_loglik = marginal_log_likelihood(N);
    //     double current_logprior = - 0.5 * (log_kappa - prior_mean_logkappa).square().matrix().sum() / prior_var_logkappa;
    //     double current_energy = -(current_loglik + current_logprior);

    //     // Sample momentum
    //     Eigen::VectorXd momentum = sqrt_mass.array() * Eigen::VectorXd::NullaryExpr(ns, []() { return R::rnorm(0.0, 1.0); }).array();
    //     double current_kinetic = 0.5 * momentum.cwiseProduct(inv_mass).dot(momentum);

    //     // Leapfrog integration
    //     Eigen::VectorXd grad = dloglike_dkappa(N, true);
    //     grad.array() += - (log_kappa - prior_mean_logkappa) / prior_var_logkappa;
    //     grad_norm = grad.norm();

    //     momentum += 0.5 * step_size * grad;
    //     for (Eigen::Index lf_step = 0; lf_step < n_leapfrog; lf_step++)
    //     {
    //         // Update kappa
    //         log_kappa += step_size * inv_mass.array() * momentum.array();
    //         kappa = log_kappa.exp();

    //         // Update gradient
    //         grad = dloglike_dkappa(N, true);
    //         grad.array() += - (log_kappa - prior_mean_logkappa) / prior_var_logkappa;

    //         // Update momentum
    //         if (lf_step != n_leapfrog - 1)
    //         {
    //             momentum += step_size * grad;
    //         }
    //     } // for lf_step

    //     momentum += 0.5 * step_size * grad;
    //     momentum = -momentum; // Negate momentum to make proposal symmetric

    //     // Compute proposed energy
    //     double proposed_loglik = marginal_log_likelihood(N);
    //     double proposed_logprior = - 0.5 * (log_kappa - prior_mean_logkappa).square().matrix().sum() / prior_var_logkappa;
    //     double proposed_energy = -(proposed_loglik + proposed_logprior);
    //     double proposed_kinetic = 0.5 * momentum.cwiseProduct(inv_mass).dot(momentum);

    //     double H_proposed = proposed_energy + proposed_kinetic;
    //     double H_current = current_energy + current_kinetic;
    //     energy_diff = H_proposed - H_current;

    //     // Metropolis acceptance step
    //     if (!std::isfinite(energy_diff) || std::abs(energy_diff) > 100.0)
    //     {
    //         kappa = current_kappa; // revert to current state
    //         return 0.0;
    //     }
    //     else
    //     {
    //         if (std::log(R::runif(0.0, 1.0)) < -energy_diff)
    //         {
    //             // accept proposed state
    //         }
    //         else
    //         {
    //             // reject and revert to current state
    //             kappa = current_kappa;
    //         }

    //         return std::min(1.0, std::exp(-energy_diff)); // acceptance probability
    //     } // end Metropolis step
    // } // update_kappa


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
            Eigen::VectorXd alpha = (u_k / U_k).array().max(EPS8); // ns x 1, Dirichlet parameters

            double deriv_k = 0.0;
            for (Eigen::Index s = 0; s < ns; s++)
            { // Loop over destination locations, indexed by s
                Eigen::Tensor<double, 0> N_sk = N.chip(s, 0).chip(k, 0).sum();
                
                double dw_drho_dist = - dist_scaled(s, k) * U_k + dist_scaled.col(k).dot(u_k);
                dw_drho_dist *= u_k(s) / (U_k * U_k);
                double dloglike_dalpha = Eigen::numext::digamma(alpha(s) + N_sk(0)) - Eigen::numext::digamma(alpha(s));
                deriv_k += dloglike_dalpha * dw_drho_dist;
            } // for s

            deriv += deriv_k;
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
            Eigen::VectorXd alpha = (w_k).array().max(EPS8); // ns x 1, Dirichlet parameters

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

            deriv += deriv_k;
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
     * @todo Precondition mass matrix
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


    // /**
    //  * @brief HMC sampler to update kappa, rho_dist, and rho_mobility all together. Operate on the unconstrained log-parameters.
    //  * 
    //  * @param energy_diff 
    //  * @param grad_norm 
    //  * @param N 
    //  * @param step_size 
    //  * @param n_leapfrog 
    //  * @param prior_mean_logparams 
    //  * @param prior_sd_logparams 
    //  * @return double 
    //  * @todo Precondition mass matrix
    //  */
    // double update_params(
    //     double &energy_diff,
    //     double &grad_norm,
    //     const Eigen::Tensor<double, 4> &N,
    //     const double &step_size,
    //     const unsigned int &n_leapfrog,
    //     const double &prior_mean_logparams = 0.0,
    //     const double &prior_sd_logparams = 10.0
    // )
    // {
    //     // HMC sampler to update kappa, rho_dist, and rho_mobility all together.
    //     const double prior_var_logparams = prior_sd_logparams * prior_sd_logparams;
    //     const Eigen::VectorXd mass_diag = Eigen::VectorXd::Constant(ns + 2, 1.0);
    //     const Eigen::VectorXd inv_mass = mass_diag.cwiseInverse();
    //     const Eigen::VectorXd sqrt_mass = mass_diag.array().sqrt();

    //     // Current state
    //     Eigen::VectorXd params_current(ns + 2);
    //     params_current.head(ns) = kappa;
    //     params_current(ns) = rho_dist;
    //     params_current(ns + 1) = rho_mobility;
    //     Eigen::ArrayXd log_params = params_current.array().log();

    //     // Compute current energy
    //     double current_loglik = marginal_log_likelihood(N);
    //     double current_logprior = - 0.5 * (log_params - prior_mean_logparams).square().matrix().sum() / prior_var_logparams;
    //     double current_energy = -(current_loglik + current_logprior);

    //     // Sample momentum
    //     Eigen::VectorXd momentum = sqrt_mass.array() * Eigen::VectorXd::NullaryExpr(ns + 2, []() { return R::rnorm(0.0, 1.0); }).array();
    //     double current_kinetic = 0.5 * momentum.cwiseProduct(inv_mass).dot(momentum);

    //     // Leapfrog integration
    //     Eigen::VectorXd grad(ns + 2);
    //     grad.head(ns) = dloglike_dkappa(N, true);
    //     grad(ns) = dloglike_drho_dist(N, true);
    //     grad(ns + 1) = dloglike_drho_mobility(N, true);
    //     grad.array() += - (log_params - prior_mean_logparams) / prior_var_logparams;
    //     grad_norm = grad.norm();

    //     momentum += 0.5 * step_size * grad;
    //     for (unsigned int lf_step = 0; lf_step < n_leapfrog; lf_step++)
    //     {
    //         // Update params
    //         log_params += step_size * inv_mass.array() * momentum.array();
    //         kappa = log_params.head(ns).exp();
    //         rho_dist = std::exp(log_params(ns));
    //         rho_mobility = std::exp(log_params(ns + 1));

    //         // Update gradient
    //         grad.head(ns) = dloglike_dkappa(N, true);
    //         grad(ns) = dloglike_drho_dist(N, true);
    //         grad(ns + 1) = dloglike_drho_mobility(N, true);
    //         grad.array() += - (log_params - prior_mean_logparams) / prior_var_logparams;

    //         // Update momentum
    //         if (lf_step != n_leapfrog - 1)
    //         {
    //             momentum += step_size * grad;
    //         }
    //     } // for lf_step

    //     momentum += 0.5 * step_size * grad;
    //     momentum = -momentum; // Negate momentum to make proposal symmetric

    //     // Compute proposed energy
    //     double proposed_loglik = marginal_log_likelihood(N);
    //     double proposed_logprior = - 0.5 * (log_params - prior_mean_logparams).square().matrix().sum() / prior_var_logparams;
    //     double proposed_energy = -(proposed_loglik + proposed_logprior);
    //     double proposed_kinetic = 0.5 * momentum.cwiseProduct(inv_mass).dot(momentum);

    //     double H_proposed = proposed_energy + proposed_kinetic;
    //     double H_current = current_energy + current_kinetic;
    //     energy_diff = H_proposed - H_current;

    //     // Metropolis acceptance step
    //     if (!std::isfinite(energy_diff) || std::abs(energy_diff) > 100.0)
    //     {
    //         kappa = params_current.head(ns);
    //         rho_dist = params_current(ns);
    //         rho_mobility = params_current(ns + 1);
    //         return 0.0;
    //     }
    //     else
    //     {
    //         if (std::log(R::runif(0.0, 1.0)) < -energy_diff)
    //         {
    //             // accept proposed state
    //         }
    //         else
    //         {
    //             // reject and revert to current state
    //             kappa = params_current.head(ns);
    //             rho_dist = params_current(ns);
    //             rho_mobility = params_current(ns + 1);
    //         }
    //         return std::min(1.0, std::exp(-energy_diff)); // acceptance probability
    //     } // end Metropolis step
    // } // update_params


    Rcpp::List run_mcmc(
        const Rcpp::NumericVector &N_array, // R array of size ns x ns x (nt + 1) x (nt + 1)
        const unsigned int &nburnin,
        const unsigned int &nsamples,
        const unsigned int &nthin,
        const double &step_size_init = 0.1,
        const unsigned int &n_leapfrog_init = 20
    )
    {
        Eigen::Tensor<double, 4> N = r_to_tensor4(const_cast<Rcpp::NumericVector&>(N_array));
        const Eigen::Index ntotal = static_cast<Eigen::Index>(nburnin + nsamples * nthin);

        HMCOpts_1d hmc_opts_rho;
        HMCDiagnostics_1d hmc_diag_rho;
        DualAveraging_1d da_adapter_rho;

        // if (hmc_block == 1)
        // {
        //     hmc_opts.leapfrog_step_size = step_size_init;
        //     hmc_opts.nleapfrog = n_leapfrog_init;

        //     hmc_diag = HMCDiagnostics_1d(static_cast<unsigned int>(ntotal), nburnin, true);
        //     da_adapter = DualAveraging_1d(hmc_opts);
        // }
        // else if (hmc_block == 2)
        // {
            hmc_opts_rho.leapfrog_step_size = step_size_init;
            hmc_opts_rho.nleapfrog = n_leapfrog_init;
            hmc_diag_rho = HMCDiagnostics_1d(static_cast<unsigned int>(ntotal), nburnin, true);
            da_adapter_rho = DualAveraging_1d(hmc_opts_rho);

        //     hmc_opts_kappa.leapfrog_step_size = step_size_init;
        //     hmc_opts_kappa.nleapfrog = n_leapfrog_init;
        //     hmc_diag_kappa = HMCDiagnostics_1d(static_cast<unsigned int>(ntotal), nburnin, true);
        //     da_adapter_kappa = DualAveraging_1d(hmc_opts_kappa);
        // }
        // else
        // {
        //     Rcpp::stop("hmc_block must be 1 or 2.");
        // }

        // Eigen::Tensor<double, 3> A_samples(ns, ns, nsamples);
        // Eigen::MatrixXd kappa_samples(ns, nsamples);
        Eigen::VectorXd rho_dist_samples(nsamples);
        Eigen::VectorXd rho_mobility_samples(nsamples);

        Progress p(static_cast<unsigned int>(ntotal), true);
        for (Eigen::Index iter = 0; iter < ntotal; iter++)
        {
            // Update adjacency matrix A
            // update_A(N);

            // // Update parameters
            // if (hmc_block == 1)
            // {
            //     // Update parameters kappa, rho_dist, rho_mobility
            //     double energy_diff = 0.0;
            //     double grad_norm = 0.0;
            //     double accept_prob = update_params(
            //         energy_diff,
            //         grad_norm,
            //         N,
            //         hmc_opts.leapfrog_step_size,
            //         hmc_opts.nleapfrog
            //     );

            //     hmc_diag.accept_count += accept_prob;
            //     if (hmc_opts.diagnostics)
            //     {
            //         hmc_diag.energy_diff(iter) = energy_diff;
            //         hmc_diag.grad_norm(iter) = grad_norm;
            //     }

            //     if (hmc_opts.dual_averaging)
            //     {
            //         if (iter < nburnin)
            //         {
            //             hmc_opts.leapfrog_step_size = da_adapter.update_step_size(accept_prob);
            //         }
            //         else if (iter == nburnin)
            //         {
            //             da_adapter.finalize_leapfrog_step(
            //                 hmc_opts.leapfrog_step_size,
            //                 hmc_opts.nleapfrog,
            //                 hmc_opts.T_target
            //             );
            //         }

            //         if (hmc_opts.diagnostics)
            //         {
            //             hmc_diag.leapfrog_step_size_stored(iter) = hmc_opts.leapfrog_step_size;
            //             hmc_diag.nleapfrog_stored(iter) = hmc_opts.nleapfrog;
            //         }
            //     } // if dual averaging
            // } // if hmc_block == 1
            // else
            // {
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

                // // Update kappa
                // double energy_diff_kappa = 0.0;
                // double grad_norm_kappa = 0.0;
                // double accept_prob_kappa = update_kappa(
                //     energy_diff_kappa,
                //     grad_norm_kappa,
                //     N,
                //     hmc_opts_kappa.leapfrog_step_size,
                //     hmc_opts_kappa.nleapfrog
                // );

                // // (Optional) Update diagnostics and dual averaging for kappa
                // hmc_diag_kappa.accept_count += accept_prob_kappa;
                // if (hmc_opts_kappa.diagnostics)
                // {
                //     hmc_diag_kappa.energy_diff(iter) = energy_diff_kappa;
                //     hmc_diag_kappa.grad_norm(iter) = grad_norm_kappa;
                // }

                // if (hmc_opts_kappa.dual_averaging)
                // {
                //     if (iter < nburnin)
                //     {
                //         hmc_opts_kappa.leapfrog_step_size = da_adapter_kappa.update_step_size(accept_prob_kappa);
                //     }
                //     else if (iter == nburnin)
                //     {
                //         da_adapter_kappa.finalize_leapfrog_step(
                //             hmc_opts_kappa.leapfrog_step_size,
                //             hmc_opts_kappa.nleapfrog,
                //             hmc_opts_kappa.T_target
                //         );
                //     }

                //     if (hmc_opts_kappa.diagnostics)
                //     {
                //         hmc_diag_kappa.leapfrog_step_size_stored(iter) = hmc_opts_kappa.leapfrog_step_size;
                //         hmc_diag_kappa.nleapfrog_stored(iter) = hmc_opts_kappa.nleapfrog;
                //     }
                // } // if dual averaging for kappa
            // } // else hmc_block == 2

            // Store samples after burn-in and thinning
            if (iter >= nburnin && ((iter - nburnin) % nthin == 0))
            {
                Eigen::Index sample_idx = (iter - nburnin) / nthin;
                // Eigen::TensorMap<Eigen::Tensor<const double, 2>> A_map(A.data(), A.rows(), A.cols());
                // A_samples.chip(sample_idx, 2) = A_map;
                // kappa_samples.col(sample_idx) = kappa;
                rho_dist_samples(sample_idx) = rho_dist;
                rho_mobility_samples(sample_idx) = rho_mobility;
            } // if store samples

            p.increment();
        } // for MCMC iter


        Rcpp::List output = Rcpp::List::create(
            // Rcpp::Named("A") = tensor3_to_r(A_samples), // ns x ns x nsamples
            // Rcpp::Named("kappa") = kappa_samples, // ns x nsamples
            Rcpp::Named("rho_dist") = rho_dist_samples, // nsamples x 1
            Rcpp::Named("rho_mobility") = rho_mobility_samples // nsamples x 1
        );

        // if (hmc_block == 1)
        // {
        //     Rcpp::List hmc_stats = Rcpp::List::create(
        //         Rcpp::Named("acceptance_rate") = hmc_diag.accept_count / static_cast<double>(ntotal),
        //         Rcpp::Named("leapfrog_step_size") = hmc_opts.leapfrog_step_size,
        //         Rcpp::Named("n_leapfrog") = hmc_opts.nleapfrog,
        //         Rcpp::Named("diagnostics") = hmc_diag.to_list()
        //     );

        //     output["hmc"] = hmc_stats;
        // } // if HMC output for hmc_block == 1
        // else
        // {
            output["hmc"] = Rcpp::List::create(
                Rcpp::Named("acceptance_rate") = hmc_diag_rho.accept_count / static_cast<double>(ntotal),
                Rcpp::Named("leapfrog_step_size") = hmc_opts_rho.leapfrog_step_size,
                Rcpp::Named("n_leapfrog") = hmc_opts_rho.nleapfrog,
                Rcpp::Named("diagnostics") = hmc_diag_rho.to_list()
            );

            // Rcpp::List hmc_stats_kappa = Rcpp::List::create(
            //     Rcpp::Named("acceptance_rate") = hmc_diag_kappa.accept_count / static_cast<double>(ntotal),
            //     Rcpp::Named("leapfrog_step_size") = hmc_opts_kappa.leapfrog_step_size,
            //     Rcpp::Named("n_leapfrog") = hmc_opts_kappa.nleapfrog,
            //     Rcpp::Named("diagnostics") = hmc_diag_kappa.to_list()
            // );

            // output["hmc"] = Rcpp::List::create(
            //     Rcpp::Named("rho") = hmc_stats_rho,
            //     Rcpp::Named("kappa") = hmc_stats_kappa
            // );
        // } // else HMC output for hmc_block == 2

        return output;
    } // run_mcmc
}; // class SpatialNetwork


class TemporalTransmission
{
private:
    Eigen::Index ns = 0; // number of spatial locations
    Eigen::Index nt = 0; // number of effective time points


    static void adapt_phi_proposal_robbins_monro(
        double &mh_sd,
        double &accept_count,
        const int &iter, 
        const int &burn_in, 
        const double &target_rate = 0.4
    )
    {
        if (iter < burn_in && iter > 0 && iter % 50 == 0)
        {
            double accept_rate = accept_count / 50.0;
            accept_count = 0.0;

            // Robbins-Monro update
            double gamma = 1.0 / std::pow(iter / 50.0, 0.6); // Decay rate
            mh_sd *= std::exp(gamma * (accept_rate - target_rate));

            // Keep in reasonable range
            mh_sd = std::max(0.01, std::min(2.0, mh_sd));
        }
        return;
    }

public:
    std::string fgain = "softplus";
    double W = 0.001; // temporal smoothing parameter
    Eigen::MatrixXd wt; // (nt + 1) x ns, centered disturbance.

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

        W = 0.001;
        if (opts.containsElementNamed("W"))
        {
            W = Rcpp::as<double>(opts["W"]);
        }

        sample_wt();
        return;
    }


    /**
     * @brief Compute the sum of N_kl(s, t) over all destination locations s and future times t >= l. That is, the sum of secondary infections caused by source location k at source time l to all destination locations s at times t >= l.
     * 
     * @param N 4D tensor of size (ns of destination[s]) x (ns of source[k]) x (nt + 1 of destination[t]) x  (nt + 1 of source[l]), unobserved secondary infections.
     * @param k Index of source location
     * @param l Index of source time
     * @return double 
     * @todo Adaptive mh_sd via simple Robbins-Monro scheme
     */
    static double compute_N_future_sum(
        const Eigen::Tensor<double, 4> &N, // N: (ns of destination[s]) x (ns of source[k]) x (nt + 1 of destination[t]) x  (nt + 1 of source[l])
        const Eigen::Index &k, // Index of source location
        const Eigen::Index &l // Index of source time
    )
    {
        Eigen::Tensor<double, 2> N_kl = N.chip(k, 1).chip(l, 2); // ns x (nt + 1)

        // Sum over destination locations s and time t >= l
        Eigen::array<Eigen::Index, 2> offsets = {0, l + 1}; // use {0, l+1} for t > l; use {0, l} for t >= l
        Eigen::array<Eigen::Index, 2> extents = {N_kl.dimension(0), N_kl.dimension(1) - (l + 1)}; // use {N_kl.dimension(0), N_kl.dimension(1) - (l + 1)} for t > l; use {N_kl.dimension(0), N_kl.dimension(1) - l} for t >= l
        Eigen::Tensor<double, 0> N_kl_future = N_kl.slice(offsets, extents).sum();

        return N_kl_future(0);
    } // compute_N_future_sum


    void sample_wt()
    {
        const double W_sqrt = std::sqrt(W);
        wt.resize(nt + 1, ns);
        wt.setZero();
        for (Eigen::Index k = 0; k < ns; k++)
        {
            for (Eigen::Index t = 1; t < nt + 1; t++)
            {
                wt(t, k) = R::rnorm(0.0, W_sqrt);
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
                double mh_mean = 0.0;
                double logp_old = - 0.5 * (wt_old * wt_old) / W_safe;

                Eigen::VectorXd N_jk_future(nt + 1 - l);
                for (Eigen::Index j = l; j < nt + 1; j++)
                {
                    // N_jk: Total secondary infections produced at location k and time j to all destination locations s in the future times t >= j
                    N_jk_future(j - l) = compute_N_future_sum(N, k, j);

                    double r_jk = wt_cumsum(j);
                    double R_jk = GainFunc::psi2hpsi(r_jk, fgain); // R_jk = h(r_jk)
                    double dR_jk = GainFunc::psi2dhpsi(r_jk, fgain); // dR_jk/d(r_jk)

                    double lambda_jk = R_jk * Y(j, k) + EPS;
                    double beta_jk = dR_jk * Y(j, k);
                    mh_prec += beta_jk * beta_jk / lambda_jk;

                    double h0_jk = (R_jk - dR_jk * r_jk) * Y(j, k);
                    double d_jk = N_jk_future(j - l) - h0_jk - beta_jk * (r_jk - wt_old);
                    mh_mean += beta_jk * d_jk / lambda_jk;
                    
                    logp_old += N_jk_future(j - l) * std::log(lambda_jk) - lambda_jk;
                } // for time t >= l

                mh_mean /= mh_prec;
                double mh_step = std::sqrt(1.0 / mh_prec) * mh_sd;
                double wt_new = R::rnorm(mh_mean, mh_step);
                double logq_new_given_old = R::dnorm4(wt_new, mh_mean, mh_step, true);

                wt(l, k) = wt_new;
                wt_cumsum = cumsum_vec(wt.col(k));

                mh_prec = 1.0 / W_safe;
                mh_mean = 0.0;
                double logp_new = - 0.5 * (wt_new * wt_new) / W_safe;
                for (Eigen::Index j = l; j < nt + 1; j++)
                {
                    double r_jk = wt_cumsum(j);
                    double R_jk = GainFunc::psi2hpsi(r_jk, fgain); // R_ts = h(r_ts)
                    double dR_jk = GainFunc::psi2dhpsi(r_jk, fgain); // dR_jk/d(r_jk)

                    double lambda_jk = R_jk * Y(j, k) + EPS;                    
                    double beta_jk = dR_jk * Y(j, k);
                    mh_prec += beta_jk * beta_jk / lambda_jk;

                    double h0_jk = (R_jk - dR_jk * r_jk) * Y(j, k);
                    double d_jk = N_jk_future(j - l) - h0_jk - beta_jk * (r_jk - wt_new);
                    mh_mean += beta_jk * d_jk / lambda_jk;

                    logp_new += N_jk_future(j - l) * std::log(lambda_jk) - lambda_jk;
                } // for time t >= l

                mh_mean /= mh_prec;
                mh_step = std::sqrt(1.0 / mh_prec) * mh_sd;
                double logq_old_given_new = R::dnorm4(wt_old, mh_mean, mh_step, true);

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
                        accept_prob(l, k) = 1.0;
                    }
                    else
                    {
                        // reject and revert
                        wt(l, k) = wt_old;
                        accept_prob(l, k) = 0.0;
                    }

                    // accept_prob(l, k) = std::min(1.0, std::exp(logratio));
                }
            } // for source time l
        } // for source location k

        return accept_prob;
    } // update_wt


    /**
     * @brief MH sampler to update eta (noncentered disturbance).
     * 
     * @param N 
     * @param Y 
     * @param mh_sd 
     * @return `Eigen::MatrixXd` acceptance probabilities for each wt(l, k)
     * @todo Adaptive mh_sd via simple Robbins-Monro scheme
     */
    Eigen::MatrixXd update_wt_by_eta(
        const Eigen::Tensor<double, 4> &N, // (ns of destination[s]) x (ns of source[k]) x (nt + 1 of destination[t]) x  (nt + 1 of source[l]), unobserved secondary infections
        const Eigen::MatrixXd &Y, // (nt + 1) x ns, observed primary infections
        const double &mh_sd = 1.0
    )
    {
        const double W_safe = W + EPS;
        const double W_sqrt = std::sqrt(W_safe);
        Eigen::MatrixXd eta = wt / W_sqrt;

        Eigen::MatrixXd accept_prob(nt + 1, ns);
        accept_prob.setZero();
        for (Eigen::Index k = 0; k < ns; k++)
        { // Loop over source locations
            
            for (Eigen::Index l = 1; l < nt + 1; l++)
            { // Loop over source times
                const double eta_old = eta(l, k);
                Eigen::VectorXd eta_cumsum = cumsum_vec(eta.col(k));

                double mh_prec = 0.0;
                double mh_mean = 0.0;
                double logp_old = - 0.5 * eta_old * eta_old;

                Eigen::VectorXd N_jk_future(nt + 1 - l);
                for (Eigen::Index j = l; j < nt + 1; j++)
                {
                    // N_jk: Total secondary infections produced at location k and time j to all destination locations s in the future times t >= j
                    N_jk_future(j - l) = compute_N_future_sum(N, k, j);

                    double r_jk = W_sqrt * eta_cumsum(j);
                    double R_jk = GainFunc::psi2hpsi(r_jk, fgain); // R_jk = h(r_jk)
                    double dR_jk = GainFunc::psi2dhpsi(r_jk, fgain); // dR_jk/d(r_jk)

                    double lambda_jk = R_jk * Y(j, k) + EPS;
                    double beta_jk = dR_jk * Y(j, k);
                    mh_prec += beta_jk * beta_jk / lambda_jk;

                    double h0_jk = (R_jk - dR_jk * r_jk) * Y(j, k);
                    double d_jk = N_jk_future(j - l) - h0_jk - beta_jk * (r_jk - W_sqrt * eta_old);
                    mh_mean += beta_jk * d_jk / lambda_jk;
                    
                    logp_old += N_jk_future(j - l) * std::log(lambda_jk) - lambda_jk;
                } // for time t >= l

                mh_prec *= W_safe;
                mh_prec += 1.0;
                mh_mean /= mh_prec;
                mh_mean *= W_sqrt;
                double mh_step = std::sqrt(1.0 / mh_prec) * mh_sd;
                double eta_new = R::rnorm(mh_mean, mh_step);
                double logq_new_given_old = R::dnorm4(eta_new, mh_mean, mh_step, true);

                eta(l, k) = eta_new;
                eta_cumsum = cumsum_vec(eta.col(k));

                mh_prec = 0.0;
                mh_mean = 0.0;
                double logp_new = - 0.5 * (eta_new * eta_new);
                for (Eigen::Index j = l; j < nt + 1; j++)
                {
                    double r_jk = W_sqrt * eta_cumsum(j);
                    double R_jk = GainFunc::psi2hpsi(r_jk, fgain); // R_ts = h(r_ts)
                    double dR_jk = GainFunc::psi2dhpsi(r_jk, fgain); // dR_jk/d(r_jk)

                    double lambda_jk = R_jk * Y(j, k) + EPS;                    
                    double beta_jk = dR_jk * Y(j, k);
                    mh_prec += beta_jk * beta_jk / lambda_jk;

                    double h0_jk = (R_jk - dR_jk * r_jk) * Y(j, k);
                    double d_jk = N_jk_future(j - l) - h0_jk - beta_jk * (r_jk - W_sqrt * eta_new);
                    mh_mean += beta_jk * d_jk / lambda_jk;

                    logp_new += N_jk_future(j - l) * std::log(lambda_jk) - lambda_jk;
                } // for time t >= l

                mh_prec *= W_safe;
                mh_prec += 1.0;
                mh_mean /= mh_prec;
                mh_mean *= W_sqrt;
                mh_step = std::sqrt(1.0 / mh_prec) * mh_sd;
                double logq_old_given_new = R::dnorm4(eta_old, mh_mean, mh_step, true);

                double logratio = std::min(
                    0.0, logp_new - logp_old + logq_old_given_new - logq_new_given_old
                );
                if (!std::isfinite(logratio))
                {
                    // reject and revert
                    eta(l, k) = eta_old;
                    accept_prob(l, k) = 0.0;
                }
                else
                {
                    if (std::log(R::runif(0.0, 1.0)) < logratio)
                    {
                        // accept
                        accept_prob(l, k) = 1.0;
                        wt(l, k) = W_sqrt * eta(l, k);
                    }
                    else
                    {
                        // reject and revert
                        eta(l, k) = eta_old;
                        accept_prob(l, k) = 0.0;
                    }

                    // accept_prob(l, k) = std::min(1.0, std::exp(logratio));
                }
            } // for source time l
        } // for source location k

        return accept_prob;
    } // update_wt_by_eta


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
        wt_accept_prob.setZero();

        Progress p(static_cast<unsigned int>(ntotal), true);
        for (Eigen::Index iter = 0; iter < ntotal; iter++)
        {
            // Update wt
            Eigen::MatrixXd accept_prob = update_wt(N, Y, mh_sd);
            wt_accept_prob += accept_prob;

            // // Update W
            // update_W(prior_shape_W, prior_rate_W);

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
    double mu = 1.0; // expectation of baseline primary infections
    Eigen::Tensor<double, 4> N; // unobserved secondary infections
    Eigen::MatrixXd N0; // (nt + 1) x ns, baseline primary infections (e.g., imported cases)
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

        N0.resize(nt + 1, ns);
        N0.setOnes();
        return;
    } // Model constructor for MCMC inference when model parameters are unknown


    Model(
        const Eigen::Index &nt_,
        const Eigen::MatrixXd &dist_matrix, // ns x ns, pairwise distance matrix between spatial locations
        const Eigen::MatrixXd &mobility_matrix, // ns x ns, pairwise mobility matrix between spatial locations
        const std::string &fgain_ = "softplus",
        const double &mu_ = 1.0,
        const double &W_ = 0.001,
        const Rcpp::List &spatial_opts = Rcpp::List::create(
            Rcpp::Named("rho_dist") = 1.0,
            Rcpp::Named("rho_mobility") = 1.0
            // Rcpp::Named("kappa") = 1.0
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

        mu = mu_;

        dlag = LagDist(lagdist_opts);
        temporal = TemporalTransmission(ns, nt, fgain_, W_);
        spatial = SpatialNetwork(dist_matrix, mobility_matrix, spatial_opts); // collapsed

        N.resize(ns, ns, nt + 1, nt + 1);
        N.setZero();

        N0.resize(nt + 1, ns);
        N0.setOnes();
        return;
    } // Model constructor for simulation when model parameters are known but Y is to be simulated


    static Eigen::MatrixXd compute_intensity(
        const Eigen::MatrixXd &Y, // (nt + 1) x ns, observed primary infections
        const Eigen::MatrixXd &R_mat, // (nt + 1) x ns
        const Eigen::MatrixXd &alpha, // (ns x ns) spatial weights matrix
        const Eigen::VectorXd &phi, // L x 1, temporal weights
        const double &mu
    )
    {
        Eigen::Index nt = Y.rows() - 1;
        Eigen::Index ns = Y.cols();

        Eigen::MatrixXd lambda_mat(nt + 1, ns);
        for (Eigen::Index t = 0; t < nt + 1; t++)
        { // for destination time t
            for (Eigen::Index s = 0; s < ns; s++)
            { // for destination location s
                double lambda_st = mu;
                for (Eigen::Index k = 0; k < ns; k++)
                {
                    for (Eigen::Index l = 0; l < t; l++)
                    {
                        if (t - l <= static_cast<Eigen::Index>(phi.size()))
                        {
                            lambda_st += alpha(s, k) * R_mat(l, k) * phi(t - l - 1) * Y(l, k);
                        }
                    } // for source time l < t
                }
                lambda_mat(t, s) = std::max(lambda_st, EPS);
            } // for destination location s
        } // for time t
        return lambda_mat;
    } // compute_intensity


    Eigen::VectorXd dloglike_dparams_collapsed(
        double &dloglike,
        const Eigen::MatrixXd &Y, // (nt + 1) x ns, observed primary infections
        const Eigen::MatrixXd &R_mat, // (nt + 1) x ns
        const bool &add_jacobian = true,
        const bool &include_W = false
    )
    {
        Eigen::MatrixXd u_mat(ns, ns);
        for (Eigen::Index s = 0; s < ns; s++)
        {
            for (Eigen::Index k = 0; k < ns; k++)
            {
                double mob_safe = std::max(spatial.mobility_scaled(s, k), EPS8);
                u_mat(s, k) = std::exp(-spatial.rho_dist * spatial.dist_scaled(s, k)) * std::pow(mob_safe, spatial.rho_mobility);
            } // for source location k
        } // for destination location s

        double W_safe = std::max(temporal.W, EPS);
        double W_sqrt = std::sqrt(W_safe);

        double deriv_rho_mobility = 0.0;
        double deriv_rho_dist = 0.0;
        double deriv_mu = 0.0;
        double deriv_W = 0.0;
        dloglike = 0.0;
        for (Eigen::Index t = 0; t < nt + 1; t++)
        { // for destination time t
            for (Eigen::Index s = 0; s < ns; s++)
            { // for destination location s

                double lambda_st = mu;
                double dlambda_st_drho_mobility = 0.0;
                double dlambda_st_drho_dist = 0.0;
                for (Eigen::Index k = 0; k < ns; k++)
                {
                    double dalpha_sk_drho_mobility = spatial.dalpha_drho_mobility(s, k);
                    double dalpha_sk_drho_dist = spatial.dalpha_drho_dist(s, k, u_mat.col(k));
                    for (Eigen::Index l = 0; l < t; l++)
                    {
                        if (t - l <= static_cast<Eigen::Index>(dlag.Fphi.size()))
                        {
                            double coef = R_mat(l, k) * dlag.Fphi(t - l - 1) * Y(l, k);
                            lambda_st += spatial.alpha(s, k) * coef;
                            dlambda_st_drho_mobility += dalpha_sk_drho_mobility * coef;
                            dlambda_st_drho_dist += dalpha_sk_drho_dist * coef;
                        }
                    } // for source time l < t
                } // for source location k

                lambda_st = std::max(lambda_st, EPS);
                double dloglike_dlambda_st = Y(t, s) / lambda_st - 1.0;
                deriv_rho_mobility += dloglike_dlambda_st * dlambda_st_drho_mobility;
                deriv_rho_dist += dloglike_dlambda_st * dlambda_st_drho_dist;
                deriv_mu += dloglike_dlambda_st;
                if (include_W)
                {
                    deriv_W += - 0.5 / W_safe + 0.5 * (temporal.wt(t, s) * temporal.wt(t, s)) / (W_safe * W_safe);
                }

                dloglike += Y(t, s) * std::log(lambda_st) - lambda_st;
                if (include_W)
                {
                    dloglike += R::dnorm4(temporal.wt(t, s), 0.0, W_sqrt, true);
                }
            } // for destination location s
        } // for time t

        if (add_jacobian)
        {
            deriv_rho_mobility *= spatial.rho_mobility;
            deriv_rho_dist *= spatial.rho_dist;
            deriv_mu *= mu;

            if (include_W)
            {
                deriv_W *= temporal.W;
            }
        }

        if (include_W)
        {
            return Eigen::Vector4d(deriv_rho_mobility, deriv_rho_dist, deriv_mu, deriv_W);
        }
        else
        {
            return Eigen::Vector3d(deriv_rho_mobility, deriv_rho_dist, deriv_mu);
        }
    } // dloglike_drho_dist_collapsed


    double update_params_collapsed(
        double &energy_diff,
        double &grad_norm,
        const Eigen::MatrixXd &Y, // (nt + 1) x ns, observed primary infections
        const double &step_size,
        const unsigned int &n_leapfrog,
        const Eigen::VectorXd &mass_diag, // covariance diagonal for momentum, which is proportional to the precision of parameters
        const double &prior_mean = 0.0,
        const double &prior_sd = 10.0,
        const bool &include_W = false
    )
    {
        const double prior_var = prior_sd * prior_sd;
        Eigen::MatrixXd R_mat = temporal.compute_Rt(); // (nt + 1) x ns
        Eigen::VectorXd inv_mass, sqrt_mass;
        if (include_W)
        {
            inv_mass = mass_diag.cwiseInverse();
            sqrt_mass = mass_diag.array().sqrt();
        }
        else
        {
            inv_mass = mass_diag.cwiseInverse();
            sqrt_mass = mass_diag.array().sqrt();
        }

        // Current state
        Eigen::VectorXd params_current;
        if (include_W)
        {
            params_current = Eigen::Vector4d(
                spatial.rho_mobility,
                spatial.rho_dist,
                mu,
                temporal.W
            );
        }
        else
        {
            params_current = Eigen::Vector3d(
                spatial.rho_mobility,
                spatial.rho_dist,
                mu
            );
        }
        Eigen::ArrayXd log_params = params_current.array().log();

        // Compute current energy and gradiant
        double loglike = 0.0;
        Eigen::VectorXd grad = dloglike_dparams_collapsed(
            loglike, Y, R_mat, true, include_W 
        );
        grad.array() += - (log_params - prior_mean) / prior_var;
        grad_norm = grad.norm();
        double current_logprior = - 0.5 * (log_params - prior_mean).square().matrix().sum() / prior_var;
        double current_energy = - (loglike + current_logprior);

        // Sample momentum
        Eigen::VectorXd momentum(params_current.size());
        for (Eigen::Index i = 0; i < momentum.size(); i++)
        {
            momentum(i) = sqrt_mass(i) * R::rnorm(0.0, 1.0);
        }
        double current_kinetic = 0.5 * momentum.cwiseProduct(inv_mass).dot(momentum);

        // Leapfrog integration
        momentum += 0.5 * step_size * grad;
        for (unsigned int lf_step = 0; lf_step < n_leapfrog; lf_step++)
        {
            // Update params
            log_params += step_size * inv_mass.array() * momentum.array();
            spatial.rho_mobility = std::exp(log_params(0));
            spatial.rho_dist = std::exp(log_params(1));
            spatial.compute_alpha();
            mu = std::exp(log_params(2));
            if (include_W)
            {
                temporal.W = std::exp(log_params(3));
            }

            // Update gradient
            grad = dloglike_dparams_collapsed(
                loglike, Y, R_mat, true, include_W
            );
            grad.array() += - (log_params - prior_mean) / prior_var;

            // Update momentum
            if (lf_step != n_leapfrog - 1)
            {
                momentum += step_size * grad;
            }
        } // for leapfrog steps

        momentum += 0.5 * step_size * grad;
        momentum = -momentum; // Negate momentum to make proposal symmetric

        double proposed_logprior = - 0.5 * (log_params - prior_mean).square().matrix().sum() / prior_var;
        double proposed_energy = - (loglike + proposed_logprior);
        double proposed_kinetic = 0.5 * momentum.cwiseProduct(inv_mass).dot(momentum);

        double H_proposed = proposed_energy + proposed_kinetic;
        double H_current = current_energy + current_kinetic;
        energy_diff = H_proposed - H_current;

        // Metropolis acceptance step
        double accept_prob;
        if (std::isfinite(energy_diff) && std::abs(energy_diff) < 100.0 && std::log(R::runif(0.0, 1.0)) < -energy_diff)
        {
            // accept proposed state
            accept_prob = std::min(1.0, std::exp(-energy_diff));
        }
        else
        {
            accept_prob = 0.0;

            // Revert to current state
            spatial.rho_mobility = params_current(0);
            spatial.rho_dist = params_current(1);
            spatial.compute_alpha();
            mu = params_current(2);
            if (include_W)
            {

                temporal.W = params_current(3);
            }
        } // end Metropolis step

        return accept_prob;
    } // update_params_collapsed


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
                    double spatial_weight = spatial.alpha(s, k);

                    for (Eigen::Index t = l + 1; t < nt + 1; t++)
                    { // Loop over destination times t > l
                        if (t - l <= dlag.Fphi.size())
                        {
                            double lag_prob = dlag.Fphi(t - l - 1); // lag probability for lag (t - l)
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

        if (N0.size() == 0)
        {
            N0.resize(nt + 1, ns);
        }

        spatial.compute_alpha();
        temporal.sample_wt();
        const Eigen::MatrixXd Rt = temporal.compute_Rt(); // (nt + 1) x ns

        // Simulate primary infections at time t = 0
        Eigen::MatrixXd Y(nt + 1, ns);
        Y.setZero();
        for (Eigen::Index k = 0; k < ns; k++)
        {
            Y(0, k) = R::rpois(mu + EPS);
        }

        // Simulate secondary infections N and observed primary infections Y over time
        for (Eigen::Index t = 1; t < nt + 1; t++)
        { // Loop over destination time t

            // Compute new primary infections at time t
            for (Eigen::Index s = 0; s < ns; s++)
            { // Loop over destination location s

                N0(t, s) = R::rpois(std::max(mu, EPS));
                Y(t, s) = N0(t, s);

                for (Eigen::Index k = 0; k < ns; k++)
                { // Loop over source location k

                    double a_sk = spatial.alpha(s, k);
                    for (Eigen::Index l = 0; l < t; l++)
                    { // Loop over source time l < t

                        if (t - l <= dlag.Fphi.size())
                        {
                            double lag_prob = dlag.Fphi(t - l - 1); // lag probability for lag (t - l)
                            double R_kl = Rt(l, k); // reproduction number at time l and location k
                            double lambda_sktl = a_sk * R_kl * lag_prob * Y(l, k) + EPS;
                            N(s, k, t, l) = R::rpois(lambda_sktl);

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
            Rcpp::Named("N0") = N0, // (nt + 1) x ns
            Rcpp::Named("alpha") = spatial.alpha, // ns x ns
            Rcpp::Named("wt") = temporal.wt, // (nt + 1) x ns
            Rcpp::Named("Rt") = Rt, // (nt + 1) x ns
            Rcpp::Named("params") = Rcpp::List::create(
                Rcpp::Named("mu") = mu,
                // Rcpp::Named("kappa") = spatial.kappa,
                Rcpp::Named("rho_dist") = spatial.rho_dist,
                Rcpp::Named("rho_mobility") = spatial.rho_mobility,
                Rcpp::Named("W") = temporal.W
            )
        );

    } // simulate


    /**
     * @brief Giibs sampler to update unobserved secondary infections N and baseline primary infections N0 from multinomial distribution.
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

                    double a_sk = spatial.alpha(s, k);
                    for (Eigen::Index l = 0; l < t; l++)
                    { // Loop over source times l < t

                        if (t - l <= dlag.Fphi.size())
                        {
                            double lag_prob = dlag.Fphi(t - l - 1); // lag probability for lag (t - l)
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

                lambda_st += mu; // add baseline primary infection intensity

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
                double p0 = mu / lambda_st; // baseline primary infection probability

                // Flatten probabilities into a vector for rmultinom
                const Eigen::Index K = ns * t; // only l < t are valid
                std::vector<double> prob(static_cast<std::size_t>(K + 1), 0.0);
                prob[0] = p0; // baseline primary infection
                std::size_t idx = 1; // start from 1 to leave prob[0] for p0
                for (Eigen::Index k = 0; k < ns; k++)
                {
                    for (Eigen::Index l = 0; l < t; l++)
                    {
                        prob[idx++] = p_st(l, k);
                    }
                } // for source location k and source time l < t

                std::vector<int> counts(K + 1, 0);
                int y_count = static_cast<int>(std::lround(y_st));
                y_count = std::max(0, y_count);
                R::rmultinom(y_count, prob.data(), static_cast<int>(K + 1), counts.data());

                // Write samples back to N and zero out impossible l >= t cells
                N0(t, s) = static_cast<double>(counts[0]); // baseline primary infections
                idx = 1; // start from 1 to leave counts[0] for baseline primary infections
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


    /**
     * @brief Gibbs sampler to update baseline intensity mu.
     * 
     * @param prior_shape 
     * @param prior_rate 
     */
    void update_mu(const double &prior_shape = 1.0, const double &prior_rate = 1.0)
    {
        double total_N0 = N0.block(1, 0, nt, ns).array().sum();
        double posterior_shape = prior_shape + total_N0;
        double posterior_rate = prior_rate + static_cast<double>(nt * ns);

        mu = R::rgamma(posterior_shape, 1.0 / posterior_rate);
        return;
    } // update_mu


    Eigen::MatrixXd update_wt_by_eta_collapsed(
        const Eigen::MatrixXd &Y, // (nt + 1) x ns, observed primary infections,
        const double &mh_sd = 1.0
    )
    {
        const double W_safe = temporal.W + EPS;
        const double W_sqrt = std::sqrt(W_safe);

        // Convert centered wt to noncentered eta
        Eigen::MatrixXd eta = temporal.wt / W_sqrt;

        // Precompute cumulative sums of eta over time for each location
        Eigen::MatrixXd eta_cumsum(nt + 1, ns);
        for (Eigen::Index k = 0; k < ns; k++)
        {
            eta_cumsum.col(k) = cumsum_vec(eta.col(k));
        }

        // Precompute r, R, dR for all (l, k)
        Eigen::MatrixXd r_mat = W_sqrt * eta_cumsum;
        Eigen::MatrixXd R_mat(nt + 1, ns);
        Eigen::MatrixXd dR_mat(nt + 1, ns);
        for (Eigen::Index k = 0; k < ns; k++)
        {
            for (Eigen::Index l = 0; l < nt + 1; l++)
            {
                R_mat(l, k) = GainFunc::psi2hpsi(r_mat(l, k), temporal.fgain);
                dR_mat(l, k) = GainFunc::psi2dhpsi(r_mat(l, k), temporal.fgain);
            }
        }

        // Precompute intensity lambda for all (s, t)
        Eigen::MatrixXd lambda_mat(nt + 1, ns);
        for (Eigen::Index t = 0; t < nt + 1; t++)
        {
            for (Eigen::Index s = 0; s < ns; s++)
            {
                double lambda_st = mu;
                for (Eigen::Index k = 0; k < ns; k++)
                {
                    for (Eigen::Index l = 0; l < t; l++)
                    {
                        if (t - l <= static_cast<Eigen::Index>(dlag.Fphi.size()))
                        {
                            lambda_st += spatial.alpha(s, k) * R_mat(l, k) * dlag.Fphi(t - l - 1) * Y(l, k);
                        }
                    } // for source time l < t
                }
                lambda_mat(t, s) = std::max(lambda_st, EPS);
            }
        }

        // MH updates for each (k, l)
        Eigen::MatrixXd accept_prob(nt + 1, ns);
        accept_prob.setZero();
        for (Eigen::Index k = 0; k < ns; k++)
        { // Loop over source locations
            
            for (Eigen::Index l = 1; l < nt + 1; l++)
            { // Loop over source times
                const double eta_old = eta(l, k);

                // ===== Step 1: Compute proposal parameters at current state =====
                // Compute c^{t}_{k, l} for all t > l
                double mh_prec = 1.0; // prior precision
                double mh_mean = 0.0;
                double logp_old = - 0.5 * eta_old * eta_old;
                for (Eigen::Index t = l + 1; t < nt + 1; t++)
                {
                    double c_klt = 0.0;
                    for (Eigen::Index i = l; i < t; i++)
                    {
                        if (t - i <= static_cast<Eigen::Index>(dlag.Fphi.size()))
                        {
                            c_klt += dR_mat(i, k) * Y(i, k) * dlag.Fphi(t - i - 1);
                        }
                    } // Calculate c_klt for i >= l and i < t

                    // Sum over all destination locations s
                    for (Eigen::Index s = 0; s < ns; s++)
                    {
                        double coef = W_sqrt * spatial.alpha(s, k) * c_klt;
                        mh_prec += coef * coef / lambda_mat(t, s);

                        double d_st = Y(t, s) - lambda_mat(t, s) + coef * eta_old;
                        mh_mean += coef * d_st / lambda_mat(t, s);

                        // Log Poisson likelihood at current state
                        logp_old += Y(t, s) * std::log(lambda_mat(t, s)) - lambda_mat(t, s);
                    } // for destination location s
                } // for time t > l

                mh_mean /= mh_prec;
                double mh_step = std::sqrt(1.0 / mh_prec) * mh_sd;

                // ===== Step 2: Propose new eta =====
                double eta_new = R::rnorm(mh_mean, mh_step);
                double logq_new_given_old = R::dnorm4(eta_new, mh_mean, mh_step, true);

                // ===== Step 3: Update state temporarily =====
                double delta_eta = eta_new - eta_old;
                eta(l, k) = eta_new;

                // Update cumsum, r, R, dR for indices >= l
                for (Eigen::Index j = l; j < nt + 1; j++)
                {
                    eta_cumsum(j, k) += delta_eta;
                    r_mat(j, k) = W_sqrt * eta_cumsum(j, k);
                    R_mat(j, k) = GainFunc::psi2hpsi(r_mat(j, k), temporal.fgain);
                    dR_mat(j, k) = GainFunc::psi2dhpsi(r_mat(j, k), temporal.fgain);
                } // for j >= l

                // Update lambda_mat for t > l (only contributions from source k changed)
                for (Eigen::Index t = l + 1; t < nt + 1; t++)
                {
                    for (Eigen::Index s = 0; s < ns; s++)
                    {
                        double delta_contrib = 0.0;
                        for (Eigen::Index j = l; j < t; j++)
                        {
                            if (t - j <= static_cast<Eigen::Index>(dlag.Fphi.size()))
                            {
                                // R_old at position j before the update
                                double r_old_j = r_mat(j, k) - W_sqrt * delta_eta;
                                double R_old_j = GainFunc::psi2hpsi(r_old_j, temporal.fgain);
                                delta_contrib += spatial.alpha(s, k) * (R_mat(j, k) - R_old_j) * dlag.Fphi(t - j - 1) * Y(j, k);
                            }
                        } // for j
                        lambda_mat(t, s) = std::max(lambda_mat(t, s) + delta_contrib, EPS);
                    } // for destination location s
                } // for time t > l

                // ===== Step 4: Compute reverse proposal and new log posterior =====
                mh_prec = 1.0; // prior precision
                mh_mean = 0.0;
                double logp_new = - 0.5 * eta_new * eta_new;
                for (Eigen::Index t = l + 1; t < nt + 1; t++)
                {
                    double c_klt = 0.0;
                    for (Eigen::Index i = l; i < t; i++)
                    {
                        if (t - i <= static_cast<Eigen::Index>(dlag.Fphi.size()))
                        {
                            c_klt += dR_mat(i, k) * Y(i, k) * dlag.Fphi(t - i - 1);
                        }
                    } // Calculate c_klt

                    // Sum over all destination locations s
                    for (Eigen::Index s = 0; s < ns; s++)
                    {
                        double coef = W_sqrt * spatial.alpha(s, k) * c_klt;
                        mh_prec += coef * coef / lambda_mat(t, s);

                        double d_st = Y(t, s) - lambda_mat(t, s) + coef * eta_new;
                        mh_mean += coef * d_st / lambda_mat(t, s);

                        // Log Poisson likelihood at current state
                        logp_new += Y(t, s) * std::log(lambda_mat(t, s)) - lambda_mat(t, s);
                    } // for destination location s
                } // for time t > l

                mh_mean /= mh_prec;
                mh_step = std::sqrt(1.0 / mh_prec) * mh_sd;
                double logq_old_given_new = R::dnorm4(eta_old, mh_mean, mh_step, true);

                // ===== Step 5: MH acceptance decision =====
                double logratio = logp_new - logp_old + logq_old_given_new - logq_new_given_old;
                bool accept = false;
                if (std::isfinite(logratio) && std::log(R::runif(0.0, 1.0)) < logratio)
                {
                    accept = true;
                }

                if (accept)
                {
                    // Accept: update wt
                    temporal.wt(l, k) = W_sqrt * eta_new;
                    accept_prob(l, k) = 1.0;
                }
                else
                {
                    // Reject: revert all state changes
                    eta(l, k) = eta_old;

                    // Revert cumsum, r, R, dR
                    for (Eigen::Index j = l; j < nt + 1; j++)
                    {
                        eta_cumsum(j, k) -= delta_eta;
                        r_mat(j, k) = W_sqrt * eta_cumsum(j, k);
                        R_mat(j, k) = GainFunc::psi2hpsi(r_mat(j, k), temporal.fgain);
                        dR_mat(j, k) = GainFunc::psi2dhpsi(r_mat(j, k), temporal.fgain);
                    }

                    // Revert λ_{s,t}
                    for (Eigen::Index t = l + 1; t < nt + 1; t++)
                    {
                        for (Eigen::Index s = 0; s < ns; s++)
                        {
                            double delta_contrib = 0.0;
                            for (Eigen::Index j = l; j < t; j++)
                            {
                                if (t - j <= static_cast<Eigen::Index>(dlag.Fphi.size()))
                                {
                                    // R at the "wrong" state we need to undo
                                    double r_wrong_j = r_mat(j, k) + W_sqrt * delta_eta;
                                    double R_wrong_j = GainFunc::psi2hpsi(r_wrong_j, temporal.fgain);
                                    delta_contrib += spatial.alpha(s, k) * (R_mat(j, k) - R_wrong_j) * dlag.Fphi(t - j - 1) * Y(j, k);
                                }
                            }
                            lambda_mat(t, s) = std::max(lambda_mat(t, s) + delta_contrib, EPS);
                        } // for destination location s
                    } // for time t > l

                    accept_prob(l, k) = 0.0;
                } // if accept

            } // for source time l
        } // for source location k

        return accept_prob;
    } // sample_wt_by_eta_collapsed


    Rcpp::List run_mcmc(
        const Eigen::MatrixXd &Y, // (nt + 1) x ns, observed primary infections
        const unsigned int &nburnin,
        const unsigned int &nsamples,
        const unsigned int &nthin,
        const Rcpp::Nullable<Rcpp::List> &mcmc_opts = R_NilValue,
        const bool &sample_augmented_N = false
    )
    {
        const Eigen::Index ntotal = static_cast<Eigen::Index>(nburnin + nsamples * nthin);

        // Infer all unknown parameters by default
        Prior wt_prior; wt_prior.infer = true; wt_prior.mh_sd = 1.0;
        bool infer_params = true;
        bool hmc_include_W = false;
        double hmc_step_size_init = 0.1;
        unsigned int hmc_nleapfrog_init = 20;
        double prior_mean = 0.0;
        double prior_sd = 10.0;

        if (mcmc_opts.isNotNull())
        {
            // Initialize priors and initial values from mcmc_opts
            Rcpp::List mcmc_opts_list(mcmc_opts);
            if (mcmc_opts_list.containsElementNamed("wt"))
            {
                Rcpp::List wt_opts = mcmc_opts_list["wt"];
                wt_prior = Prior(wt_opts);
                if (wt_opts.containsElementNamed("init"))
                {
                    Rcpp::NumericMatrix wt_mat = wt_opts["init"];
                    temporal.wt = Rcpp::as<Eigen::MatrixXd>(wt_mat);
                }
            } // if wt
            if (mcmc_opts_list.containsElementNamed("params"))
            {
                Rcpp::List params_opts = mcmc_opts_list["params"];
                if (params_opts.containsElementNamed("infer"))
                {
                    infer_params = Rcpp::as<bool>(params_opts["infer"]);
                }
                if (params_opts.containsElementNamed("init"))
                {
                    Rcpp::List init_values = params_opts["init"];
                    if (init_values.containsElementNamed("rho_dist"))
                    {
                        spatial.rho_dist = Rcpp::as<double>(init_values["rho_dist"]);
                    }
                    if (init_values.containsElementNamed("rho_mobility"))                    {
                        spatial.rho_mobility = Rcpp::as<double>(init_values["rho_mobility"]);
                    }
                    spatial.compute_alpha();

                    if (init_values.containsElementNamed("mu"))
                    {
                        mu = Rcpp::as<double>(init_values["mu"]);
                    }
                    if (init_values.containsElementNamed("W"))
                    {
                        temporal.W = Rcpp::as<double>(init_values["W"]);
                    }
                } // if init

                if (params_opts.containsElementNamed("hmc"))
                {
                    Rcpp::List hmc_opts = params_opts["hmc"];
                    if (hmc_opts.containsElementNamed("include_W"))
                    {
                        hmc_include_W = Rcpp::as<bool>(hmc_opts["include_W"]);
                    } // set hmc_include_W
                    if (hmc_opts.containsElementNamed("step_size"))
                    {
                        hmc_step_size_init = Rcpp::as<double>(hmc_opts["step_size"]);
                    } // set hmc_step_size
                    if (hmc_opts.containsElementNamed("n_leapfrog"))
                    {
                        hmc_nleapfrog_init = static_cast<unsigned int>(hmc_opts["n_leapfrog"]);
                    } // set hmc_nleapfrog
                    if (hmc_opts.containsElementNamed("prior_mean"))
                    {
                        prior_mean = Rcpp::as<double>(hmc_opts["prior_mean"]);
                    } // set prior_mean
                    if (hmc_opts.containsElementNamed("prior_sd"))
                    {
                        prior_sd = Rcpp::as<double>(hmc_opts["prior_sd"]);
                    } // set prior_sd
                } // if hmc
            } // if spatial
        } // if mcmc_opts


        // Set up HMC options and diagnostics for spatial parameters if to be inferred
        HMCOpts_1d hmc_opts;
        HMCDiagnostics_1d hmc_diag;
        DualAveraging_1d da_adapter;
        if (infer_params)
        {
            hmc_opts.leapfrog_step_size = hmc_step_size_init;
            hmc_opts.nleapfrog = hmc_nleapfrog_init;
            hmc_diag = HMCDiagnostics_1d(static_cast<unsigned int>(ntotal), nburnin, true);
            da_adapter = DualAveraging_1d(hmc_opts);
        } // infer_params

        Eigen::VectorXd mass_diag_est = Eigen::VectorXd::Ones(hmc_include_W ? 4 : 3);
        MassAdapter mass_adapter;
        mass_adapter.mean = Eigen::VectorXd::Zero(hmc_include_W? 4 : 3);
        mass_adapter.mean(0) = std::log(spatial.rho_mobility);
        mass_adapter.mean(1) = std::log(spatial.rho_dist);
        mass_adapter.mean(2) = std::log(mu);
        if (hmc_include_W)
        {
            mass_adapter.mean(3) = std::log(temporal.W);
        }
        mass_adapter.M2 = Eigen::VectorXd::Zero(mass_adapter.mean.size());

        Eigen::MatrixXd accept_prob_wt(nt + 1, ns);
        accept_prob_wt.setZero();

        Eigen::VectorXd rho_dist_samples(nsamples);
        Eigen::VectorXd rho_mobility_samples(nsamples);
        Eigen::Tensor<double, 3> wt_samples(nt + 1, ns, nsamples);
        Eigen::VectorXd W_samples(nsamples);
        Eigen::VectorXd mu_samples(nsamples);

        Eigen::Tensor<double, 3> N_samples; // (nt + 1) x ns x nsamples, total secondary infections generated by (t, s)
        Eigen::Tensor<double, 3> N0_samples; // (nt + 1) x ns x nsamples, baseline primary infections at (t, s)
        if (sample_augmented_N)
        {
            N_samples.resize(nt + 1, ns, nsamples);
            N0_samples.resize(nt + 1, ns, nsamples);
        }

        Progress p(static_cast<unsigned int>(ntotal), true);
        for (Eigen::Index iter = 0; iter < ntotal; iter++)
        {
            // Update unobserved secondary infections N and baseline primary infections N0
            if (sample_augmented_N)
            {
                update_N(Y);
            }

            // Update temporal transmission components
            if (wt_prior.infer)
            {
                try
                {
                    Eigen::MatrixXd accept_prob = update_wt_by_eta_collapsed(Y, wt_prior.mh_sd);
                accept_prob_wt += accept_prob;
                }
                catch(const std::exception& e)
                {
                    std::cerr << e.what() << '\n';
                    throw std::runtime_error("Error updating wt at iteration " + std::to_string(iter));
                }
                
            }

            // Update spatial network components
            if (infer_params)
            {
                double energy_diff = 0.0;
                double grad_norm = 0.0;
                double accept_prob = update_params_collapsed(
                    energy_diff, grad_norm, Y,
                    hmc_opts.leapfrog_step_size,
                    hmc_opts.nleapfrog, mass_diag_est,
                    prior_mean, prior_sd, hmc_include_W
                );

                // (Optional) Update diagnostics and dual averaging for rho
                hmc_diag.accept_count += accept_prob;
                if (hmc_opts.diagnostics)
                {
                    hmc_diag.energy_diff(iter) = energy_diff;
                    hmc_diag.grad_norm(iter) = grad_norm;
                }

                if (hmc_opts.dual_averaging && iter <= nburnin)
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
                            hmc_opts.T_target);
                    }

                    if (hmc_opts.diagnostics)
                    {
                        hmc_diag.leapfrog_step_size_stored(iter) = hmc_opts.leapfrog_step_size;
                        hmc_diag.nleapfrog_stored(iter) = hmc_opts.nleapfrog;
                    }
                } // if dual averaging


                if (iter < nburnin)
                {
                    // Phase 1 (iter < nburnin/2): Adapt step size with unit mass
                    // Phase 2 (nburnin/2 <= iter < nburnin): Adapt mass matrix

                    Eigen::VectorXd current_log_params(hmc_include_W ? 4 : 3);
                    current_log_params << std::log(spatial.rho_mobility),
                        std::log(spatial.rho_dist),
                        std::log(mu);
                    if (hmc_include_W)
                        current_log_params(3) = std::log(temporal.W);

                    mass_adapter.update(current_log_params);

                    // Only update mass matrix ONCE at the midpoint
                    if (iter == nburnin / 2)
                    {
                        mass_diag_est = mass_adapter.get_mass_diag();

                        // CRITICAL: Reset dual averaging for new geometry
                        da_adapter = DualAveraging_1d(hmc_opts);
                    }
                }
            } // if infer_params

            // Store samples after burn-in and thinning
            if (iter >= nburnin && ((iter - nburnin) % nthin == 0))
            {
                Eigen::Index sample_idx = (iter - nburnin) / nthin;
                rho_dist_samples(sample_idx) = spatial.rho_dist;
                rho_mobility_samples(sample_idx) = spatial.rho_mobility;
                Eigen::TensorMap<Eigen::Tensor<const double, 2>> wt_map(temporal.wt.data(), temporal.wt.rows(), temporal.wt.cols());
                wt_samples.chip(sample_idx, 2) = wt_map;
                W_samples(sample_idx) = temporal.W;
                mu_samples(sample_idx) = mu;

                if (sample_augmented_N)
                {
                    for (Eigen::Index t = 0; t < nt + 1; t++)
                    {
                        for (Eigen::Index s = 0; s < ns; s++)
                        {
                            N_samples(t, s, sample_idx) = temporal.compute_N_future_sum(N, t, s);
                        }
                    }
                    // N_samples.chip(sample_idx, 4) = N;
                    Eigen::TensorMap<Eigen::Tensor<const double, 2>> N0_map(N0.data(), N0.rows(), N0.cols());
                    N0_samples.chip(sample_idx, 2) = N0_map;
                }
            } // if store samples

            p.increment();
        } // for MCMC iter

        Rcpp::List hmc_stats = Rcpp::List::create(
            Rcpp::Named("acceptance_rate") = hmc_diag.accept_count / static_cast<double>(ntotal),
            Rcpp::Named("leapfrog_step_size") = hmc_opts.leapfrog_step_size,
            Rcpp::Named("n_leapfrog") = hmc_opts.nleapfrog,
            Rcpp::Named("diagnostics") = hmc_diag.to_list());

        Rcpp::List output = Rcpp::List::create(
            Rcpp::Named("mu") = mu_samples, // nsamples x 1
            Rcpp::Named("rho_dist") = rho_dist_samples, // nsamples x 1
            Rcpp::Named("rho_mobility") = rho_mobility_samples, // nsamples x 1
            Rcpp::Named("W") = W_samples, // nsamples x 1
            Rcpp::Named("wt") = tensor3_to_r(wt_samples), // (nt + 1) x ns x nsamples
            Rcpp::Named("wt_accept_prob") = accept_prob_wt / static_cast<double>(ntotal), // (nt + 1) x ns
            Rcpp::Named("hmc_stats") = hmc_stats
        );

        if (sample_augmented_N)
        {
            output["N"] = tensor3_to_r(N_samples); // (nt + 1) x ns x nsamples
            output["N0"] = tensor3_to_r(N0_samples); // (nt + 1) x ns x nsamples
        }

        return output;

    } // run_mcmc

}; // class Model



#endif
