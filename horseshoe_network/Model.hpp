#pragma once
#ifndef MODEL_HPP
#define MODEL_HPP

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

#include "../utils/utils2.h"
#include "../core/GainFuncEigen.hpp"
#include "../core/LagDistEigen.hpp"
#include "../inference/hmc.hpp"

// [[Rcpp::plugins(cpp17)]]
// [[Rcpp::depends(RcppEigen,RcppProgress)]]


/**
 * @brief Structure to hold MH proposal parameters for block update of eta_k
 */
struct BlockMHProposal
{
    Eigen::VectorXd mean;           // (nt-1) x 1, proposal mean u_k
    Eigen::MatrixXd prec;           // (nt-1) x (nt-1), precision matrix Omega_k
    Eigen::LLT<Eigen::MatrixXd> chol; // Cholesky decomposition of precision
    double log_det_prec;            // log|Omega_k| = 2 * sum(log(diag(L)))
    bool valid;                     // Whether Cholesky succeeded

    BlockMHProposal() : log_det_prec(0.0), valid(false) {}

    /**
     * @brief Sample from the proposal distribution N(mean, prec^{-1})
     * 
     * Uses: eta = mean + L^{-T} * z, where z ~ N(0, I) and prec = L * L'
     */
    Eigen::VectorXd sample(const double &mh_sd = 1.0) const
    {
        if (!valid)
        {
            Rcpp::stop("Cannot sample from invalid proposal (Cholesky failed)");
        }

        Eigen::Index n = mean.size();
        Eigen::VectorXd z(n);
        for (Eigen::Index i = 0; i < n; i++)
        {
            z(i) = R::rnorm(0.0, mh_sd);
        }

        // Solve L' * x = z  =>  x = L^{-T} * z
        Eigen::VectorXd x = chol.matrixU().solve(z);

        return mean + x;
    }

    /**
     * @brief Compute log density of eta under proposal N(mean, prec^{-1})
     * 
     * log p(eta) = -n/2 * log(2*pi) + 1/2 * log|prec| - 1/2 * (eta - mean)' * prec * (eta - mean)
     */
    double log_density(const Eigen::VectorXd &eta) const
    {
        if (!valid)
        {
            return -std::numeric_limits<double>::infinity();
        }

        Eigen::Index n = mean.size();
        Eigen::VectorXd diff = eta - mean;

        // Quadratic form: diff' * prec * diff = ||L' * diff||^2
        Eigen::VectorXd L_t_diff = chol.matrixL().transpose() * diff;
        double quad_form = L_t_diff.squaredNorm();

        static const double LOG_2PI = std::log(2.0 * M_PI);
        return -0.5 * n * LOG_2PI + 0.5 * log_det_prec - 0.5 * quad_form;
    }
}; // struct BlockMHProposal


/**
 * @brief Sparse spatial network controlled by a regularized horseshoe prior.
 * 
 */
class SpatialNetwork
{
private:
    Eigen::Index ns = 0; // number of spatial locations


    /**
     * @brief Center covariates per-column (excluding diagonal)
     *
     * For each source k, centers the covariate values across destinations s ≠ k
     * This matches the per-column centering of theta
     */
    void center_covariates_per_column()
    {
        if (include_distance && dist.size() > 0)
        {
            for (Eigen::Index k = 0; k < ns; k++)
            {
                // Compute column mean excluding diagonal
                double col_mean = 0.0;
                for (Eigen::Index s = 0; s < ns; s++)
                {
                    if (s != k)
                        col_mean += dist(s, k);
                }
                col_mean /= static_cast<double>(ns - 1);

                // Center
                for (Eigen::Index s = 0; s < ns; s++)
                {
                    if (s != k)
                    {
                        dist(s, k) -= col_mean;
                    }
                }
                dist(k, k) = 0.0; // Diagonal stays zero
            } // for source locations k
        } // centering distance

        if (include_log_mobility && log_mobility.size() > 0)
        {
            for (Eigen::Index k = 0; k < ns; k++)
            {
                double col_mean = 0.0;
                for (Eigen::Index s = 0; s < ns; s++)
                {
                    if (s != k)
                        col_mean += log_mobility(s, k);
                }
                col_mean /= static_cast<double>(ns - 1);

                for (Eigen::Index s = 0; s < ns; s++)
                {
                    if (s != k)
                    {
                        log_mobility(s, k) -= col_mean;
                    }
                }
                log_mobility(k, k) = 0.0;
            } // for source locations k
        } // centering log mobility
    } // center_covariates_per_column

public:
    bool include_distance = false;
    bool include_log_mobility = false;

    Eigen::MatrixXd dist; // ns x ns, pairwise distances between locations
    Eigen::MatrixXd log_mobility; // ns x ns, pairwise log mobility

    Eigen::MatrixXd alpha; // ns x ns, adjacency matrix / stochastic network
    double rho_dist = 0.0; // distance decay parameter
    double rho_mobility = 0.0; // mobility scaling parameter

    Eigen::MatrixXd theta; // ns x ns, horseshoe spatial transmission parameters
    Eigen::MatrixXd epsilon; // ns x ns, noncentered parameterization theta[s,k] = tau[k] * tilde(gamma[s,k]) * epsilon[s,k]
    Eigen::MatrixXd gamma; // ns x ns, horseshoe element-wise variance, gamma[s,k] ~ Half-Cauchy(0,1)
    Eigen::VectorXd tau; // ns x 1, horseshoe column-wise variance, tau[k] ~ Half-Cauchy(0, tau0)
    Eigen::VectorXd wdiag; // ns x 1, self-exciting weight per location

    // Regularized horseshoe slab parameters
    double c_sq = 4.0; // slab variance (c²), controls maximum shrinkage
    double tau0 = 1.0; // scale of the half-Cauchy prior for tau

    SpatialNetwork()
    {
        ns = 1;
        c_sq = 4.0;
        tau0 = 1.0;

        include_distance = false;
        include_log_mobility = false;

        initialize_horseshoe_zero();
        alpha.resize(ns, ns);

        if (ns == 1)
        {
            alpha(0, 0) = 1.0;
            wdiag.setOnes();
        }
        else
        {
            for (Eigen::Index k = 0; k < ns; k++)
            { // Loop over source locations
                alpha.col(k) = compute_alpha_col(k);
            }
        }
        return;
    } // SpatialNetwork default constructor


    SpatialNetwork(
        const unsigned int &ns_in,
        const bool &random_init = false,
        const double &c_sq_in = 4.0,
        const double &tau0_in = 1.0,
        const Rcpp::Nullable<Rcpp::NumericMatrix> &dist_scaled_in = R_NilValue,
        const Rcpp::Nullable<Rcpp::NumericMatrix> &mobility_scaled_in = R_NilValue
    )
    {
        c_sq = c_sq_in;
        tau0 = tau0_in;
        ns = ns_in;

        if (dist_scaled_in.isNotNull())
        {
            Rcpp::NumericMatrix dist_mat(dist_scaled_in);
            if (dist_mat.nrow() != ns || dist_mat.ncol() != ns)
            {
                Rcpp::stop("Dimension mismatch when initializing distance matrix.");
            }
            dist = Rcpp::as<Eigen::MatrixXd>(dist_mat);
            include_distance = true;
        }
        else
        {
            include_distance = false;
        } // if use distance or not

        if (mobility_scaled_in.isNotNull())
        {
            Rcpp::NumericMatrix mobility_mat(mobility_scaled_in);
            if (mobility_mat.nrow() != ns || mobility_mat.ncol() != ns)
            {
                Rcpp::stop("Dimension mismatch when initializing mobility matrix.");
            }
            Eigen::MatrixXd mobility_tmp = Rcpp::as<Eigen::MatrixXd>(mobility_mat);
            log_mobility = mobility_tmp.array().log().matrix();
            include_log_mobility = true;
        }
        else
        {
            include_log_mobility = false;
        } // if use mobility or not

        center_covariates_per_column();

        if (random_init)
        {
            rho_dist = include_distance ? R::runif(0.0, 2.0) : 0.0;
            rho_mobility = include_log_mobility ? R::runif(0.0, 2.0) : 0.0;
            initialize_horseshoe_dominant();
        }
        else
        {
            rho_dist = 0.0;
            rho_mobility = 0.0;
            // initialize_horseshoe_zero();
            initialize_horseshoe_sparse();
        }

        alpha.resize(ns, ns);
        if (ns == 1)
        {
            alpha(0, 0) = 1.0;
            wdiag.setOnes();
        }
        else
        {
            for (Eigen::Index k = 0; k < ns; k++)
            { // Loop over source locations
                alpha.col(k) = compute_alpha_col(k);
            }
        }

        return;
    } // SpatialNetwork constructor for MCMC inference when model parameters are unknown


    SpatialNetwork(
        const unsigned int &ns_in, 
        const Rcpp::List &settings,
        const Rcpp::Nullable<Rcpp::NumericMatrix> &dist_scaled_in = R_NilValue,
        const Rcpp::Nullable<Rcpp::NumericMatrix> &mobility_scaled_in = R_NilValue
    )
    {
        ns = ns_in;

        // Regularized horseshoe slab parameters
        c_sq = 4.0;
        if (settings.containsElementNamed("c_sq"))
        {
            c_sq = Rcpp::as<double>(settings["c_sq"]);
        } // if slab variance specified

        tau0 = 1.0;
        if (settings.containsElementNamed("tau0"))
        {
            tau0 = Rcpp::as<double>(settings["tau0"]);
        }

        if (dist_scaled_in.isNotNull())
        {
            Rcpp::NumericMatrix dist_mat(dist_scaled_in);
            if (dist_mat.nrow() != ns || dist_mat.ncol() != ns)
            {
                Rcpp::stop("Dimension mismatch when initializing distance matrix.");
            }
            dist = Rcpp::as<Eigen::MatrixXd>(dist_mat);
            include_distance = true;
            
            if (settings.containsElementNamed("rho_dist"))
            {
                rho_dist = Rcpp::as<double>(settings["rho_dist"]);
            }
            else
            {
                rho_dist = R::runif(0.0, 2.0);
            }
        }
        else
        {
            rho_dist = 0.0;
            include_distance = false;
        } // if use distance or not

        if (mobility_scaled_in.isNotNull())
        {
            Rcpp::NumericMatrix mobility_mat(mobility_scaled_in);
            if (mobility_mat.nrow() != ns || mobility_mat.ncol() != ns)
            {
                Rcpp::stop("Dimension mismatch when initializing mobility matrix.");
            }
            Eigen::MatrixXd mobility_tmp = Rcpp::as<Eigen::MatrixXd>(mobility_mat);
            log_mobility = mobility_tmp.array().log().matrix();
            include_log_mobility = true;

            if (settings.containsElementNamed("rho_mobility"))
            {
                rho_mobility = Rcpp::as<double>(settings["rho_mobility"]);
            }
            else
            {
                rho_mobility = R::runif(0.0, 2.0);
            }
        }
        else
        {
            rho_mobility = 0.0;
            include_log_mobility = false;
        } // if use mobility or not

        center_covariates_per_column();

        if (settings.containsElementNamed("theta"))
        {
            initialize_horseshoe_zero();
            Eigen::MatrixXd theta_in = Rcpp::as<Eigen::MatrixXd>(settings["theta"]);
            if (theta_in.rows() != ns || theta_in.cols() != ns)
            {
                Rcpp::stop("Dimension mismatch when initializing horseshoe theta.");
            }
            theta = theta_in;
            for (Eigen::Index k = 0; k < ns; k++)
            {
                theta(k, k) = 0.0; // ensure diagonal is zero
            }
        }
        else
        {
            initialize_horseshoe_dominant();
        }

        alpha.resize(ns, ns);
        if (ns == 1)
        {
            alpha(0, 0) = 1.0;
            wdiag.setOnes();
        }
        else
        {
            for (Eigen::Index k = 0; k < ns; k++)
            { // Loop over source locations
                alpha.col(k) = compute_alpha_col(k);
            }
        }

        return;
    } // SpatialNetwork constructor for simulation when model parameters are known but Y is to be simulated


    Rcpp::List to_list() const
    {
        return Rcpp::List::create(
            Rcpp::Named("alpha") = alpha,
            Rcpp::Named("theta") = theta,
            Rcpp::Named("gamma") = gamma,
            Rcpp::Named("tau") = tau,
            Rcpp::Named("wdiag") = wdiag,
            Rcpp::Named("rho_dist") = rho_dist,
            Rcpp::Named("rho_mobility") = rho_mobility,
            Rcpp::Named("dist") = dist,
            Rcpp::Named("log_mobility") = log_mobility,
            Rcpp::Named("c_sq") = c_sq,
            Rcpp::Named("tau0") = tau0
        );
    } // to_list


    /**
     * @brief Initialize horseshoe parameters ensuring diagonal dominance
     *
     * Guarantees: alpha(k, k) > max_{s != k} alpha(s, k) for all k
     */
    void initialize_horseshoe_random(const double &dominance_margin = 0.1)
    {
        const double t0_sq = tau0 * tau0;

        theta.resize(ns, ns);
        epsilon.resize(ns, ns);
        gamma.resize(ns, ns);
        tau.resize(ns);
        wdiag.resize(ns);

        for (Eigen::Index k = 0; k < ns; k++)
        {
            // Sample variance parameters
            double zeta_k = 1.0 / R::rgamma(0.5, t0_sq);
            double tau2_k = std::max(1.0 / R::rgamma(0.5, zeta_k), 1e-6);
            tau(k) = std::sqrt(tau2_k);

            // Sample theta and gamma for off-diagonal elements
            for (Eigen::Index s = 0; s < ns; s++)
            {
                double delta_sk = 1.0 / R::rgamma(0.5, 1.0);
                double gm2_sk = std::max(1.0 / R::rgamma(0.5, delta_sk), 1e-6);
                gamma(s, k) = std::sqrt(gm2_sk);

                if (s == k)
                {
                    theta(s, k) = 0.0;
                }
                else
                {
                    double reg_var = compute_regularized_variance(s, k);
                    epsilon(s, k) = R::rnorm(0.0, 1.0);
                    theta(s, k) = std::sqrt(reg_var) * epsilon(s, k);
                }
            }

            // Compute unnormalized weights
            Eigen::VectorXd u_k(ns);
            u_k.setZero();
            double u_max = 0.0;
            double U_off = 0.0;

            for (Eigen::Index s = 0; s < ns; s++)
            {
                if (s != k)
                {
                    u_k(s) = std::exp(std::min(theta(s, k), UPBND));
                    U_off += u_k(s);
                    u_max = std::max(u_max, u_k(s));
                }
            }

            // Compute minimum wdiag for diagonal dominance
            // alpha(k,k) > alpha(s,k) for all s != k
            // wdiag > u_max / (U_off + u_max)
            double wdiag_min = u_max / (U_off + u_max + EPS);

            // Add margin and ensure wdiag is in valid range
            wdiag_min = std::min(wdiag_min + dominance_margin, 1.0 - EPS);

            // Sample wdiag from truncated Beta(a, b) on [wdiag_min, 1]
            // Simple approach: sample Beta and rescale, or use rejection
            double wdiag_raw = R::rbeta(5.0, 2.0);
            wdiag(k) = wdiag_min + (1.0 - wdiag_min) * wdiag_raw;

            // Ensure valid range
            wdiag(k) = std::min(std::max(wdiag(k), wdiag_min), 1.0 - EPS);
        }
    } // initialize_horseshoe_random


    /**
     * @brief Initialize using a parameterization that guarantees diagonal dominance
     *
     * Key insight: Let r_k = alpha(k,k) / max_{s!=k} alpha(s,k) be the "dominance ratio"
     * We can parameterize r_k > 1 directly using log(r_k - 1) ~ Normal
     */
    void initialize_horseshoe_dominant(
        const double &log_dominance_mean = 1.0, // E[log(r - 1)] ≈ 1 means r ≈ 3.7
        const double &log_dominance_sd = 0.5)
    {
        const double t0_sq = tau0 * tau0;
        theta.resize(ns, ns);
        epsilon.resize(ns, ns);
        gamma.resize(ns, ns);
        tau.resize(ns);
        wdiag.resize(ns);

        for (Eigen::Index k = 0; k < ns; k++)
        {
            // Sample variance parameters
            double zeta_k = 1.0 / R::rgamma(0.5, t0_sq);
            double tau2_k = std::max(1.0 / R::rgamma(0.5, zeta_k), 1e-6);
            tau(k) = std::sqrt(tau2_k);

            // Sample theta for off-diagonal
            double theta_max_k = -std::numeric_limits<double>::infinity();

            for (Eigen::Index s = 0; s < ns; s++)
            {
                double delta_sk = 1.0 / R::rgamma(0.5, 1.0);
                double gm2_sk = std::max(1.0 / R::rgamma(0.5, delta_sk), 1e-6);
                gamma(s, k) = std::sqrt(gm2_sk);

                if (s == k)
                {
                    theta(s, k) = 0.0;
                }
                else
                {
                    double reg_var = compute_regularized_variance(s, k);
                    epsilon(s, k) = R::rnorm(0.0, 1.0);
                    theta(s, k) = std::sqrt(reg_var) * epsilon(s, k);
                    theta_max_k = std::max(theta_max_k, theta(s, k));
                }
            }

            // Compute u_k values
            Eigen::VectorXd u_k(ns);
            u_k.setZero();
            double u_max = 0.0;
            double U_off = 0.0;

            for (Eigen::Index s = 0; s < ns; s++)
            {
                if (s != k)
                {
                    u_k(s) = std::exp(std::min(theta(s, k), UPBND));
                    U_off += u_k(s);
                    u_max = std::max(u_max, u_k(s));
                }
            }

            // Sample dominance ratio r > 1 via log(r - 1) ~ N(mean, sd)
            double log_r_minus_1 = R::rnorm(log_dominance_mean, log_dominance_sd);
            double r = 1.0 + std::exp(log_r_minus_1); // r > 1 guaranteed

            // Solve for wdiag given r = alpha(k,k) / max_offdiag
            // alpha(k,k) = wdiag
            // max_offdiag = (1 - wdiag) * u_max / U_off
            // r = wdiag / ((1 - wdiag) * u_max / U_off)
            // r * (1 - wdiag) * u_max / U_off = wdiag
            // r * u_max / U_off - r * wdiag * u_max / U_off = wdiag
            // r * u_max / U_off = wdiag * (1 + r * u_max / U_off)
            // wdiag = (r * u_max / U_off) / (1 + r * u_max / U_off)
            //       = r * u_max / (U_off + r * u_max)

            double ratio = r * u_max / U_off;
            wdiag(k) = ratio / (1.0 + ratio);

            // Clamp to valid range
            wdiag(k) = std::min(std::max(wdiag(k), EPS), 1.0 - EPS);
        }
    }


    void initialize_horseshoe_zero()
    {
        theta.resize(ns, ns);
        epsilon.resize(ns, ns);
        gamma.resize(ns, ns);
        tau.resize(ns);
        wdiag.resize(ns);

        theta.setZero();
        epsilon.setZero();
        gamma.setOnes();
        tau.setOnes();
        wdiag.setOnes();
        wdiag *= 0.8;
    } // initialize_horseshoe_zero


    /**
     * @brief Initialize horseshoe variables with conservative sparse settings
     *
     * Philosophy: Start with assumption that most transmission is local (wdiag high)
     * and spatial transmission is sparse (theta near 0)
     */
    void initialize_horseshoe_sparse(
        const double &wdiag_init = 0.8, // High = mostly local transmission
        const double &tau_init = 0.1,   // Small = strong global shrinkage
        const double &gamma_init = 1.0 // Moderate local shrinkage
    )
    {
        theta.resize(ns, ns);
        epsilon.resize(ns, ns);
        gamma.resize(ns, ns);
        tau.resize(ns);
        wdiag.resize(ns);

        for (Eigen::Index k = 0; k < ns; k++)
        {
            // Self-excitation weight: start high (local transmission dominates)
            wdiag(k) = wdiag_init;

            // Global shrinkage: start small (sparse network)
            tau(k) = tau_init;

            for (Eigen::Index s = 0; s < ns; s++)
            {
                if (s == k)
                {
                    epsilon(s, k) = 0.0;
                    theta(s, k) = 0.0; // Diagonal fixed at 0
                    gamma(s, k) = 1.0;
                }
                else
                {
                    // Start theta at 0 (no spatial transmission initially)
                    epsilon(s, k) = 0.0;
                    theta(s, k) = 0.0;

                    // Local shrinkage parameters at moderate values
                    gamma(s, k) = gamma_init;
                }
            }
        }
    } // initialize_horseshoe_sparse


    /**
     * @brief Compute regularized horseshoe variance for element (s, k)
     * 
     * Uses the formula: γ̃²_{s,k} = c² γ²_{s,k} / (c² + τ² γ²_{s,k})
     * This bounds the effective variance at c² even when γ_{s,k} → ∞
     */
    double compute_regularized_variance(
        const Eigen::Index &s,
        const Eigen::Index &k
    ) const
    {
        double gamma_sq = gamma(s, k) * gamma(s, k);
        double tau_sq = tau(k) * tau(k);
        
        // Regularized variance: c² γ² / (c² + τ² γ²)
        double numerator = c_sq * gamma_sq;
        double denominator = c_sq + tau_sq * gamma_sq;
        double gm2_tilde = numerator / std::max(denominator, EPS);
        return tau_sq * gm2_tilde;
    } // compute_regularized_variance


    Eigen::MatrixXd center_theta()
    {
        Eigen::MatrixXd theta_centered = theta;
        for (Eigen::Index k = 0; k < ns; k++)
        {
            double col_mean = (theta.col(k).sum() - theta(k, k)) / static_cast<double>(ns - 1);
            for (Eigen::Index s = 0; s < ns; s++)
            {
                if (s != k)
                {
                    theta_centered(s, k) -= col_mean;
                }
            }
        }
        return theta_centered;
    }


    /**
     * @brief Compute regularized variance for entire column k
     */
    Eigen::VectorXd compute_regularized_variance_col(const Eigen::Index &k) const
    {
        Eigen::VectorXd var_col(ns);
        for (Eigen::Index s = 0; s < ns; s++)
        {
            if (s == k)
            {
                var_col(s) = EPS; // diagonal has no variance
            }
            else
            {
                var_col(s) = compute_regularized_variance(s, k);
            }
        }
        return var_col;
    } // compute_regularized_variance_col


    Eigen::VectorXd compute_unnormalized_weight_col(const Eigen::Index &k) const
    {
        double th_mean = (theta.col(k).sum() - theta(k, k)) / static_cast<double>(ns - 1);
        Eigen::VectorXd u_k(ns); // unnormalized weights
        u_k.setZero();
        for (Eigen::Index s = 0; s < ns; s++)
        { // Loop over destinations
            if (s != k)
            {
                double v_sk = theta(s, k) - th_mean;
                if (include_distance)
                {
                    v_sk -= rho_dist * dist(s, k);
                }
                if (include_log_mobility)
                {
                    v_sk += rho_mobility * log_mobility(s, k);
                }
                u_k(s) = std::exp(std::min(v_sk, UPBND));
            }
        } // for s

        return u_k;
    } // compute_unnormalized_weight_col


    Eigen::VectorXd compute_alpha_col(const Eigen::Index &k) const
    {
        Eigen::VectorXd alpha(ns);
        if (ns == 1)
        {
            Eigen::VectorXd alpha(1);
            alpha(0) = 1.0;
            return alpha;
        }
        else
        {
            Eigen::VectorXd u_k = compute_unnormalized_weight_col(k);
            double U_off = u_k.sum(); // u_k(k) == 0
            if (U_off <= EPS8)
            {
                alpha.setZero();
                alpha(k) = 1.0;
                return alpha;
            }

            double w_k = std::min(std::max(wdiag(k), EPS8), 1.0 - EPS8);
            for (Eigen::Index s = 0; s < ns; ++s)
            {
                if (s == k)
                {
                    alpha(s) = w_k;
                }
                else
                {
                    alpha(s) = (1.0 - w_k) * u_k(s) / U_off;
                }
            }
            return alpha;
        }
    } // compute_alpha_col


    void compute_alpha()
    {
        for (Eigen::Index k = 0; k < ns; k++)
        { // Loop over source locations
            alpha.col(k) = compute_alpha_col(k);
        }
        return;
    } // compute_alpha


    double dalpha_dtheta(
        const Eigen::Index &a_s, // destination location index for alpha
        const Eigen::Index &th_j, // destination location index for theta
        const Eigen::Index &k, // source location index for both alpha and theta
        const Eigen::VectorXd &u_k // unnormalized weights for source k
    )
    {
        if (a_s == k)
        {
            return 0.0; // alpha[k, k] does not depend on theta
        }
        else
        {
            const double U_k = u_k.array().sum(); // u_k(k) == 0
            double deriv = 0.0;
            if (a_s == th_j)
            {
                deriv = u_k(a_s) / U_k * (1.0 - u_k(a_s) / U_k);
            }
            else
            {
                deriv = - u_k(a_s) * u_k(th_j) / (U_k * U_k);
            }

            deriv *= 1.0 - wdiag(k);
            return deriv;
        }
    } // dalpha_dtheta


    double dalpha_depsilon(
        const Eigen::Index &a_s, // destination location index for alpha
        const Eigen::Index &eps_j, // destination location index for theta
        const Eigen::Index &k, // source location index for both alpha and theta
        const Eigen::VectorXd &u_k // unnormalized weights for source k
    )
    {
        double reg_gm2 = compute_regularized_variance(eps_j, k);
        double dtheta_deps = std::sqrt(std::max(reg_gm2, EPS));
        return dalpha_dtheta(a_s, eps_j, k, u_k) * dtheta_deps;
    } // dalpha_depsilon


    double dalpha_dgamma(
        const Eigen::Index &a_s, // destination location index for alpha
        const Eigen::Index &gm_j, // destination location index for theta
        const Eigen::Index &k, // source location index for both alpha and theta
        const Eigen::VectorXd &u_k // unnormalized weights for source k
    )
    {
        double dtheta_dreg_gm = tau(k) * epsilon(gm_j, k);
        double c3 = c_sq * std::sqrt(c_sq);
        double denom = c_sq + tau(k) * tau(k) * gamma(gm_j, k) * gamma(gm_j, k);
        double dreg_gm_dgm = c3 / std::pow(denom, 1.5);
        double dalpha_dth = dalpha_dtheta(a_s, gm_j, k, u_k);
        return dalpha_dth * dtheta_dreg_gm * dreg_gm_dgm;
    } // dalpha_dgamma


    double dalpha_dtau(
        const Eigen::Index &a_s, // destination location index for alpha
        const Eigen::Index &k, // source location index for both alpha and tau
        const Eigen::VectorXd &u_k // unnormalized weights for source k
    )
    {
        const double c3 = c_sq * std::sqrt(c_sq);
        Eigen::VectorXd dtheta_dtau(ns);
        for (Eigen::Index s = 0; s < ns; s++)
        {
            if (s == k)
            {
                dtheta_dtau(s) = 0.0;
            }
            else
            {
                double num = c3 * gamma(s, k) * epsilon(s, k);
                double denom = c_sq + tau(k) * tau(k) * gamma(s, k) * gamma(s, k);
                dtheta_dtau(s) = num / (denom * std::sqrt(denom));
            }
        } // for s

        const double U_k = u_k.array().sum(); // u_k(k) == 0
        double dtheta_mean = u_k.dot(dtheta_dtau) / U_k;
        return (1.0 - wdiag(k)) * (u_k(a_s) / U_k) * (dtheta_dtau(a_s) - dtheta_mean);
    } // dalpha_dtau


    double dalpha_dwj(
        const Eigen::Index &s, // destination location index
        const Eigen::Index &j, // source location index
        const Eigen::VectorXd &u_j // unnormalized weights for source j
    )
    {
        double deriv = 0.0;
        if (s == j)
        {
            deriv = 1.0;
        }
        else
        {
            double U_j = u_j.array().sum(); // u_j(j) == 0
            deriv = - u_j(s) / U_j;
        }

        return deriv;
    } // dalpha_dwj


    Eigen::MatrixXd dalpha_drho_dist()
    {
        Eigen::MatrixXd deriv_mat(ns, ns);
        deriv_mat.setZero();
        for (Eigen::Index k = 0; k < ns; k++)
        {
            Eigen::VectorXd u_k = compute_unnormalized_weight_col(k);
            double U_k = u_k.array().sum(); // u_k(k) == 0
            double D_k = u_k.dot(dist.col(k)); // sum_s u_k(s) * dist(s, k), u_k(k) == 0 & dist(k, k) == 0
            for (Eigen::Index s = 0; s < ns; s++)
            {
                if (s != k)
                {
                    deriv_mat(s, k) = - alpha(s, k) * (dist(s, k) - D_k / U_k);
                }
            }
        }

        return deriv_mat;
    } // dalpha_drho_dist


    Eigen::MatrixXd dalpha_drho_mobility()
    {
        Eigen::MatrixXd deriv_mat(ns, ns);
        deriv_mat.setZero();
        for (Eigen::Index k = 0; k < ns; k++)
        {
            Eigen::VectorXd u_k = compute_unnormalized_weight_col(k);
            double U_k = u_k.array().sum(); // u_k(k) == 0
            double M_k = u_k.dot(log_mobility.col(k)); // sum_s u_k(s) * log_mobility(s, k), u_k(k) == 0 & log_mobility(k, k) == 0
            for (Eigen::Index s = 0; s < ns; s++)
            {
                if (s != k)
                {
                    deriv_mat(s, k) = alpha(s, k) * (log_mobility(s, k) - M_k / U_k);
                }
            }
        }

        return deriv_mat;
    } // dalpha_drho_mobility
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
}; // class TemporalTransmission


class ZeroInflation
{
private:
    Eigen::Index ns = 0; // number of spatial locations
    Eigen::Index nt = 0; // number of effective time points
public:
    bool inflated = false;
    double beta0 = 0.0; // intercept for zero-inflation probability
    double beta1 = 0.0; // coefficient for zero-inflation probability

    Eigen::MatrixXd Z; // (nt + 1) x ns, zero-inflation indicators


    ZeroInflation()
    {
        return;
    } // ZeroInflation default constructor


    ZeroInflation(
        const Eigen::Index &ns_, 
        const Eigen::Index &nt_,
        const bool &inflated_in = false
    ) : ns(ns_), nt(nt_)
    {
        inflated = inflated_in;
        beta0 = 0.0;
        beta1 = 0.0;

        Z.resize(nt + 1, ns);
        Z.setOnes();
        return;
    } // ZeroInflation constructor


    ZeroInflation(
        const Eigen::MatrixXd &Y,
        const bool &inflated_in = true
    )
    {
        ns = Y.cols();
        nt = Y.rows() - 1;

        inflated = inflated_in;
        beta0 = 0.0;
        beta1 = 0.0;

        Z.resize(nt + 1, ns);
        for (Eigen::Index t = 0; t < nt + 1; t++)
        {
            for (Eigen::Index s = 0; s < ns; s++)
            {
                if (Y(t, s) < EPS && inflated)
                {
                    Z(t, s) = 0.0;
                }
                else
                {
                    Z(t, s) = 1.0;
                }
            } // for s
        } // for t

        return;
    } // ZeroInflation constructor from data


    ZeroInflation(
        const Eigen::Index &ns_, 
        const Eigen::Index &nt_,
        const Rcpp::List &zi_opts
    )
    {
        ns = ns_;
        nt = nt_;

        inflated = true;
        if (zi_opts.containsElementNamed("inflated"))
        {
            inflated = Rcpp::as<bool>(zi_opts["inflated"]);
        }

        beta0 = 0.0;
        if (zi_opts.containsElementNamed("beta0"))
        {
            beta0 = Rcpp::as<double>(zi_opts["beta0"]);
        }

        beta1 = 0.0;
        if (zi_opts.containsElementNamed("beta1"))
        {
            beta1 = Rcpp::as<double>(zi_opts["beta1"]);
        }

        Z.resize(nt + 1, ns);
        Z.setOnes();
        return;
    }


    Rcpp::List to_list() const
    {
        Rcpp::List zinfo = Rcpp::List::create(
            Rcpp::Named("inflated") = inflated
        );

        if (inflated)
        {
            zinfo["beta0"] = beta0;
            zinfo["beta1"] = beta1;
            zinfo["Z"] = Z;
        }

        return zinfo;
    } // to_list


    void simulate()
    {
        if (!inflated)
        {
            return;
        }

        Z.setOnes();
        for (Eigen::Index s = 0; s < ns; s++)
        {
            for (Eigen::Index t = 1; t < nt + 1; t++)
            {
                double logit_p = beta0 + beta1 * Z(t - 1, s);
                double p_one = inv_logit_stable(logit_p);
                Z(t, s) = R::runif(0.0, 1.0) < p_one ? 1.0 : 0.0;
            } // for t
        } // for s
    } // simulate


    void update_Z(
        const Eigen::MatrixXd &Y, // (nt + 1) x ns, observed primary infections
        const Eigen::MatrixXd &lambda_mat // (nt + 1) x ns, intensity matrix
    )
    {
        const double p11 = inv_logit_stable(beta0 + beta1); // P(Z(t,s)=1 | Z(t-1,s)=1)
        const double p01 = inv_logit_stable(beta0); // P(Z(t,s)=1 | Z(t-1,s)=0)

        for (Eigen::Index s = 0; s < ns; s++)
        {
            // prob_filter: filtering probabilities p(Z(t,s)=1 | Y(1:t,s))
            Eigen::VectorXd prob_filter = Eigen::VectorXd::Zero(nt + 1);

            // Forward filtering
            for (Eigen::Index t = 1; t < nt + 1; t++)
            {
                if (Y(t, s) > 0)
                {
                    Z(t, s) = 1.0;
                    prob_filter(t) = 1.0;
                    // If observed count > 0, then Z(t,s) must be 1
                }
                else
                {
                    double p_y0 = R::dpois(0.0, lambda_mat(t, s), false); // P(Y(t,s)=0 | Z(t,s)=1)

                    /*
                    P(Z(t,s)=1 | Y(1:t,s)) is proportional to:
                        P(Z(t-1,s)=1 | Y(1:t-1,s)) * P(Z(t,s)=1 | Z(t-1,s)=1) * P(Y(t,s)=0 | Z(t,s)=1) +
                        P(Z(t-1,s)=0 | Y(1:t-1,s)) * P(Z(t,s)=1 | Z(t-1,s)=0) * P(Y(t,s)=0 | Z(t,s)=1)
                    */
                    double prob_1 = prob_filter(t - 1) * p11 * p_y0 + (1.0 - prob_filter(t - 1)) * p01 * p_y0;

                    /*
                    P(Z(t,s)=0 | Y(1:t,s)) is proportional to:
                        P(Z(t-1,s)=1 | Y(1:t-1,s)) * P(Z(t,s)=0 | Z(t-1,s)=1) +
                        P(Z(t-1,s)=0 | Y(1:t-1,s)) * P(Z(t,s)=0 | Z(t-1,s)=0)
                    */
                    double prob_0 = prob_filter(t - 1) * (1.0 - p11) + (1.0 - prob_filter(t - 1)) * (1.0 - p01);

                    prob_filter(t) = prob_1 / (prob_1 + prob_0 + EPS);
                } // calculate p(Z(t,s)=1 | Y(1:t,s)) if Y(t,s) == 0
            } // for t - forward filtering

            /*
            Backward sampling from the joint probability:
            p(z[1:nT] | y[1:nT]) = p(z[nT] | y[1:nT]) *
                    prod_{t=2}^{nT} p(z[t-1] | z[t], y[1:t-1])
            */
            Z(nt, s) = (R::runif(0.0, 1.0) < prob_filter(nt)) ? 1.0 : 0.0;
            for (Eigen::Index t = nt - 1; t >= 1; t--)
            {
                /*
                P(Z(t,s)=1 | Z(t+1,s), Y(1:t,s)) is proportional to:
                    P(Z(t+1,s) | Z(t,s)=1) * P(Z(t,s)=1 | Y(1:t,s))

                P(Z(t,s)=0 | Z(t+1,s), Y(1:t,s)) is proportional to:
                    P(Z(t+1,s) | Z(t,s)=0) * P(Z(t,s)=0 | Y(1:t,s))
                */
                double p_ztp1_given_zt_1 = (Z(t + 1, s) == 1.0) ? p11 : (1.0 - p11);
                double p_ztp1_given_zt_0 = (Z(t + 1, s) == 1.0) ? p01 : (1.0 - p01);

                double prob_1 = p_ztp1_given_zt_1 * prob_filter(t);
                double prob_0 = p_ztp1_given_zt_0 * (1.0 - prob_filter(t));
                double prob_zt_1 = prob_1 / (prob_1 + prob_0 + EPS);
                Z(t, s) = (R::runif(0.0, 1.0) < prob_zt_1) ? 1.0 : 0.0;
            } // for t - backward sampling
        } // for s

    } // update_Z


    Eigen::VectorXd dlogprob_dbeta(
        double &logprob,
        const double &prior_mean = 0.0,
        const double &prior_sd = 3.0
    )
    {
        Eigen::VectorXd grad(2);
        grad.setZero();
        logprob = 0.0;

        if (!inflated)
        {
            return grad;
        }

        for (Eigen::Index s = 0; s < ns; s++)
        {
            for (Eigen::Index t = 1; t < nt + 1; t++)
            {
                double logit_p = beta0 + beta1 * Z(t - 1, s);
                double p_one = inv_logit_stable(logit_p);
                logprob += Z(t, s) * std::log(p_one + EPS) + (1.0 - Z(t, s)) * std::log(1.0 - p_one + EPS);

                double dloglike_dlogit_p = Z(t, s) - p_one;
                grad(0) += dloglike_dlogit_p; // derivative w.r.t. beta0
                grad(1) += dloglike_dlogit_p * Z(t - 1, s); // derivative w.r.t. beta1
            } // for t
        } // for s

        // Add priors
        logprob += R::dnorm(beta0, prior_mean, prior_sd, true);
        logprob += R::dnorm(beta1, prior_mean, prior_sd, true);

        double prior_var = prior_sd * prior_sd;
        grad(0) += - (beta0 - prior_mean) / prior_var;
        grad(1) += - (beta1 - prior_mean) / prior_var;
        return grad;
    } // dlogprob_dbeta


    double update_beta(
        double &energy_diff,
        double &grad_norm,
        const double &leapfrog_step_size,
        const unsigned int &n_leapfrog,
        const Eigen::VectorXd &mass_diag = Eigen::VectorXd::Ones(2),
        const double &prior_mean = 0.0,
        const double &prior_sd = 3.0
    )
    {
        Eigen::VectorXd inv_mass = mass_diag.cwiseInverse();
        Eigen::VectorXd sqrt_mass = mass_diag.array().sqrt();

        Eigen::VectorXd beta_old(2);
        beta_old(0) = beta0;
        beta_old(1) = beta1;

        Eigen::VectorXd beta = beta_old;
        double logprob = 0.0;
        Eigen::VectorXd grad = dlogprob_dbeta(logprob, prior_mean, prior_sd);
        grad_norm = grad.norm();
        double current_energy = -logprob;

        // Sample momentum
        Eigen::Vector2d momentum;
        for (Eigen::Index i = 0; i < momentum.size(); i++)
        {
            momentum(i) = sqrt_mass(i) * R::rnorm(0.0, 1.0);
        }
        double current_kinetic = 0.5 * momentum.cwiseProduct(inv_mass).dot(momentum);

        // Leapfrog integration
        momentum += 0.5 * leapfrog_step_size * grad;
        for (unsigned int lf_step = 0; lf_step < n_leapfrog; lf_step++)
        {
            // Update params
            beta += leapfrog_step_size * inv_mass.cwiseProduct(momentum);
            beta0 = beta(0);
            beta1 = beta(1);

            // Compute new gradient
            grad = dlogprob_dbeta(logprob, prior_mean, prior_sd);

            // Update momentum
            if (lf_step != n_leapfrog - 1)
            {
                momentum += leapfrog_step_size * grad;
            }
        } // for leapfrog steps

        momentum += 0.5 * leapfrog_step_size * grad;
        momentum = -momentum; // Negate momentum to make proposal symmetric

        double proposed_energy = - logprob;
        double proposed_kinetic = 0.5 * momentum.cwiseProduct(inv_mass).dot(momentum);

        double H_proposed = proposed_energy + proposed_kinetic;
        double H_current = current_energy + current_kinetic;
        energy_diff = H_proposed - H_current;

        // Metropolis acceptance step
        double accept_prob = 0.0;
        bool accept = false;
        if (std::isfinite(energy_diff) && std::abs(energy_diff) < 100.0)
        {
            accept_prob = std::min(1.0, std::exp(-energy_diff));
            if (std::log(R::runif(0.0, 1.0)) < -energy_diff)
            {
                accept = true;
            }
        }

        if (!accept)
        {
            // Revert to old values
            beta0 = beta_old(0);
            beta1 = beta_old(1);
        }

        return accept_prob;
    } // update_beta
}; // class ZeroInflation


class Model
{
private:
    Eigen::Index ns = 0; // number of spatial locations
    Eigen::Index nt = 0; // number of effective time points
public:
    /* Known model properties */
    LagDist dlag;
    ZeroInflation zinfl; // zero-inflation model

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
        const Rcpp::Nullable<Rcpp::NumericMatrix> &dist_scaled_in = R_NilValue,
        const Rcpp::Nullable<Rcpp::NumericMatrix> &mobility_scaled_in = R_NilValue,
        const double &c_sq = 4.0,
        const double &tau0 = 1.0,
        const std::string &fgain_ = "softplus",
        const bool &zero_inflated = true,
        const Rcpp::List &lagdist_opts = Rcpp::List::create(
            Rcpp::Named("name") = "lognorm",
            Rcpp::Named("par1") = LN_MU,
            Rcpp::Named("par2") = LN_SD2,
            Rcpp::Named("truncated") = true,
            Rcpp::Named("rescaled") = true
        )
    )
    {
        ns = Y.cols();
        nt = Y.rows() - 1;

        dlag = LagDist(lagdist_opts);
        temporal = TemporalTransmission(ns, nt, fgain_);
        spatial = SpatialNetwork(ns, true, c_sq, tau0, dist_scaled_in, mobility_scaled_in);
        zinfl = ZeroInflation(Y, zero_inflated);

        sample_N(Y);

        N0.resize(nt + 1, ns);
        N0.setOnes();
        return;
    } // Model constructor for MCMC inference when model parameters are unknown


    Model(
        const Eigen::Index &nt_,
        const Eigen::Index &ns_,
        const Rcpp::Nullable<Rcpp::NumericMatrix> &dist_scaled_in = R_NilValue,
        const Rcpp::Nullable<Rcpp::NumericMatrix> &mobility_scaled_in = R_NilValue,
        const double &mu_in = 1.0,
        const double &W_in = 0.001,
        const std::string &fgain_ = "softplus",
        const Rcpp::List &spatial_opts = Rcpp::List::create(
            Rcpp::Named("c_sq") = 4.0,
            Rcpp::Named("tau0") = 1.0,
            Rcpp::Named("rho_dist") = 0.0,
            Rcpp::Named("rho_mobility") = 0.0
        ),
        const Rcpp::List &lagdist_opts = Rcpp::List::create(
            Rcpp::Named("name") = "lognorm",
            Rcpp::Named("par1") = LN_MU,
            Rcpp::Named("par2") = LN_SD2,
            Rcpp::Named("truncated") = true,
            Rcpp::Named("rescaled") = true
        ),
        const Rcpp::List &zinfl_opts = Rcpp::List::create(
            Rcpp::Named("inflated") = false,
            Rcpp::Named("beta0") = 0.0,
            Rcpp::Named("beta1") = 0.0
        )
    )
    {
        nt = nt_;
        ns = ns_;

        mu = mu_in;

        dlag = LagDist(lagdist_opts);
        temporal = TemporalTransmission(ns, nt, fgain_, W_in);
        spatial = SpatialNetwork(ns, spatial_opts, dist_scaled_in, mobility_scaled_in);
        zinfl = ZeroInflation(ns, nt, zinfl_opts);

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

    /**
     * @brief Compute gradient of log-likelihood w.r.t. rho_dist
     *
     * This computes the FULL gradient by summing over all (s, t, k), not just one column.
     * Required because rho_dist is a global parameter shared across all columns.
     *
     * @param Y Observed infections (nt+1) x ns
     * @param R_mat Reproduction numbers (nt+1) x ns
     * @return Gradient d(loglike)/d(rho_dist)
     */
    double compute_grad_rho_dist_full(
        const Eigen::MatrixXd &Y,
        const Eigen::MatrixXd &R_mat)
    {
        if (!spatial.include_distance)
            return 0.0;

        // Precompute dalpha/drho_dist matrix (ns x ns)
        Eigen::MatrixXd dalpha_drho = spatial.dalpha_drho_dist();

        double grad = 0.0;
        for (Eigen::Index t = 0; t < nt + 1; t++)
        {
            for (Eigen::Index s = 0; s < ns; s++)
            {
                if (zinfl.inflated && zinfl.Z(t, s) < 1)
                {
                    continue; // Skip if structural zero
                }

                double lambda_st = mu;
                double dlambda_drho = 0.0;

                for (Eigen::Index k = 0; k < ns; k++)
                {
                    double coef_sum = 0.0;
                    for (Eigen::Index l = 0; l < t; l++)
                    {
                        if (t - l <= static_cast<Eigen::Index>(dlag.Fphi.size()))
                        {
                            double coef = R_mat(l, k) * dlag.Fphi(t - l - 1) * Y(l, k);
                            coef_sum += coef;
                            lambda_st += spatial.alpha(s, k) * coef;
                        }
                    }
                    dlambda_drho += dalpha_drho(s, k) * coef_sum;
                }

                lambda_st = std::max(lambda_st, EPS);
                double dloglike_dlambda = Y(t, s) / lambda_st - 1.0;
                grad += dloglike_dlambda * dlambda_drho;
            }
        }
        return grad;
    }

    /**
     * @brief Compute gradient of log-likelihood w.r.t. rho_mobility
     */
    double compute_grad_rho_mobility_full(
        const Eigen::MatrixXd &Y,
        const Eigen::MatrixXd &R_mat)
    {
        if (!spatial.include_log_mobility)
            return 0.0;

        Eigen::MatrixXd dalpha_drho = spatial.dalpha_drho_mobility();

        double grad = 0.0;
        for (Eigen::Index t = 0; t < nt + 1; t++)
        {
            for (Eigen::Index s = 0; s < ns; s++)
            {
                if (zinfl.inflated && zinfl.Z(t, s) < 1)
                {
                    continue; // Skip if structural zero
                }

                double lambda_st = mu;
                double dlambda_drho = 0.0;

                for (Eigen::Index k = 0; k < ns; k++)
                {
                    double coef_sum = 0.0;
                    for (Eigen::Index l = 0; l < t; l++)
                    {
                        if (t - l <= static_cast<Eigen::Index>(dlag.Fphi.size()))
                        {
                            double coef = R_mat(l, k) * dlag.Fphi(t - l - 1) * Y(l, k);
                            coef_sum += coef;
                            lambda_st += spatial.alpha(s, k) * coef;
                        }
                    }
                    dlambda_drho += dalpha_drho(s, k) * coef_sum;
                }

                lambda_st = std::max(lambda_st, EPS);
                double dloglike_dlambda = Y(t, s) / lambda_st - 1.0;
                grad += dloglike_dlambda * dlambda_drho;
            }
        }
        return grad;
    } // compute_grad_rho_mobility_full


    Eigen::Index get_ndim_colvec(
        Eigen::VectorXi & idx_start, // starting index of each parameter block
        Eigen::VectorXi & inverse_idx, // inverse mapping from spatial index to off-diagonal index
        const Eigen::Index &k, // column index
        const bool &include_rho_dist = true,
        const bool &include_rho_mobility = true)
    {
        /*
        Store starting indices for:
        0 - epsilon
        1 - logit(wdiag)
        2 - log(gamma)
        3 - log(tau)
        4 - rho_dist
        5 - rho_mobility
        */
        idx_start.resize(6);
        idx_start.setConstant(-1);

        const Eigen::Index n_offdiag = ns - 1;
        idx_start(0) = 0; // theta start
        idx_start(1) = n_offdiag; // log(wdiag)
        idx_start(2) = n_offdiag + 1; // log(gamma) start
        idx_start(3) = 2 * n_offdiag + 1; // log(tau)

        Eigen::Index ndim = 2 * n_offdiag + 2;
        if (include_rho_dist && spatial.include_distance)
        {
            idx_start(4) = ndim; // log(rho_dist)
            ndim++;
        }
        if (include_rho_mobility && spatial.include_log_mobility)
        {
            idx_start(5) = ndim; // log(rho_mobility)
            ndim++;
        }


        inverse_idx.resize(ns);
        inverse_idx.setConstant(-1);
        std::vector<Eigen::Index> offdiag_idx;
        offdiag_idx.reserve(n_offdiag);
        for (Eigen::Index s = 0; s < ns; s++)
        {
            if (s != k)
            {
                inverse_idx[s] = static_cast<Eigen::Index>(offdiag_idx.size());
                offdiag_idx.push_back(s);
            }
        }

        return ndim;
    } // get_ndim_colvec


    /**
     * @brief Pack parameters into unconstrained vector, optionally including rho
     */
    Eigen::ArrayXd get_unconstrained_colvec(
        const Eigen::Index &k,
        const Eigen::Index &ndim,
        const Eigen::VectorXi &idx_start,
        const Eigen::VectorXi &inverse_idx
    ) const
    {
        const Eigen::Index n_offdiag = ns - 1;

        Eigen::ArrayXd unconstrained(ndim);
        for (Eigen::Index s = 0; s < ns; s++)
        {
            if (s == k)
                continue;
            Eigen::Index i = inverse_idx[s];
            unconstrained(idx_start(0) + i) = spatial.epsilon(s, k); // theta: from 0 to n_offdiag - 1
            unconstrained(idx_start(2) + i) = std::log(spatial.gamma(s, k)); // gamma: from n_offdiag to 2*n_offdiag - 1
        }
        unconstrained(idx_start(3)) = std::log(spatial.tau(k));
        unconstrained(idx_start(1)) = logit_safe(spatial.wdiag(k));

        if (idx_start(4) >= 0)
        {
            unconstrained(idx_start(5)) = std::log(std::max(spatial.rho_dist, EPS));
        }

        if (idx_start(5) >= 0)
        {
            unconstrained(idx_start(6)) = std::log(std::max(spatial.rho_mobility, EPS));
        }

        return unconstrained;
    } // get_unconstrained_colvec


    // Lambda to unpack unconstrained params back to model
    void unpack_unconstrained_colvec(
        const Eigen::ArrayXd &unconstrained_params,
        const Eigen::Index &k,
        const Eigen::VectorXi &idx_start,
        const Eigen::VectorXi &inverse_idx
    )
    {
        spatial.tau(k) = std::exp(clamp_log_scale(unconstrained_params(idx_start(3))));
        for (Eigen::Index s = 0; s < ns; s++)
        {
            if (s == k)
                continue;
            Eigen::Index i = inverse_idx[s];
            spatial.epsilon(s, k) = unconstrained_params(idx_start(0) + i);
            spatial.gamma(s, k) = std::exp(clamp_log_scale(unconstrained_params(idx_start(2) + i)));

            double reg_var = spatial.compute_regularized_variance(s, k);
            spatial.theta(s, k) = spatial.epsilon(s, k) * std::sqrt(reg_var);
        }
        spatial.epsilon(k, k) = 0.0;
        spatial.theta(k, k) = 0.0;

        double w = inv_logit_stable(unconstrained_params(idx_start(1)));
        spatial.wdiag(k) = clamp01(w);

        // Unpack rho parameters
        if (idx_start(4) >= 0)
        {
            spatial.rho_dist = std::exp(clamp_log_scale(unconstrained_params(idx_start(4))));
        }
        if (idx_start(5) >= 0)
        {
            spatial.rho_mobility = std::exp(clamp_log_scale(unconstrained_params(idx_start(5))));
        }

        // Recompute alpha (for ALL columns since rho affects all)
        spatial.compute_alpha();
    } // unpack_unconstrained_colvec


    /**
     * @brief Compute gradient for horseshoe parameters of column k, optionally including rho.
     *
     * Parameter layout in returned gradient vector:
     *   [0, n_offdiag-1]           : epsilon_{s,k} for s != k
     *   [n_offdiag]                : logit(wdiag_k)
     *   [n_offdiag+1, 2*n_offdiag] : log(gamma_{s,k}) for s != k
     *   [2*n_offdiag+1]            : log(tau_k)
     *   [2*n_offdiag+2]            : log(rho_dist)     (if include_rho_dist)
     *   [2*n_offdiag+3]            : log(rho_mobility) (if include_rho_mobility)
     *
     * @param loglike Output: log-likelihood value
     * @param k Column index
     * @param Y Observed infections
     * @param R_mat Reproduction numbers
     * @param include_rho_dist Include rho_dist in the parameter vector
     * @param include_rho_mobility Include rho_mobility in the parameter vector
     * @param beta_prior_a Beta prior parameter for wdiag
     * @param beta_prior_b Beta prior parameter for wdiag
     * @param rho_prior_mean Prior mean for log(rho)
     * @param rho_prior_var Prior variance for log(rho)
     */
    Eigen::VectorXd dlogprob_dcolvec(
        double &loglike,
        const Eigen::Index &k,
        const Eigen::MatrixXd &Y,
        const Eigen::MatrixXd &R_mat,
        const bool &include_rho_dist = true,
        const bool &include_rho_mobility = false,
        const double &beta_prior_a = 5.0,
        const double &beta_prior_b = 2.0,
        const double &rho_prior_mean = 0.0,
        const double &rho_prior_var = 9.0)
    {
        /*
        idx_start:
        0 - epsilon
        1 - logit(wdiag)
        2 - log(gamma)
        3 - log(tau)
        4 - rho_dist
        5 - rho_mobility
        */
        Eigen::VectorXi idx_start, inverse_idx;
        Eigen::Index ndim = get_ndim_colvec(
            idx_start,
            inverse_idx,
            k,
            include_rho_dist,
            include_rho_mobility
        );

        const double w_safe = clamp01(spatial.wdiag(k));
        const double logit_wk = logit_safe(w_safe);
        const double jacobian_wk = w_safe * (1.0 - w_safe);
        Eigen::VectorXd u_k = spatial.compute_unnormalized_weight_col(k);

        loglike = 0.0;
        Eigen::VectorXd grad(ndim);
        grad.setZero();

        // Prior on logit(wdiag)
        grad(idx_start(1)) += beta_prior_a * (1.0 - w_safe) - beta_prior_b * w_safe;
        loglike += beta_prior_a * std::log(w_safe) + beta_prior_b * std::log(1.0 - w_safe);

        // Prior on log(tau(k))
        double tau_scl = std::max(spatial.tau(k) / spatial.tau0, 1.0e-8);
        double t2_scl = tau_scl * tau_scl;
        grad(idx_start(3)) += (1.0 - t2_scl) / (1.0 + t2_scl);
        loglike += std::log(tau_scl) - std::log(1.0 + t2_scl);

        if (idx_start(4) >= 0)
        {
            // Prior on log(rho_dist)
            double log_rho = std::log(std::max(spatial.rho_dist, EPS));
            grad(idx_start(4)) += -(log_rho - rho_prior_mean) / rho_prior_var;
            loglike += -0.5 * (log_rho - rho_prior_mean) * (log_rho - rho_prior_mean) / rho_prior_var;
        }

        if (idx_start(5) >= 0)
        {
            // Prior on log(rho_mobility)
            double log_rho = std::log(std::max(spatial.rho_mobility, EPS));
            grad(idx_start(5)) += -(log_rho - rho_prior_mean) / rho_prior_var;
            loglike += -0.5 * (log_rho - rho_prior_mean) * (log_rho - rho_prior_mean) / rho_prior_var;
        }

        // Loop over destination locations for theta/gamma gradients
        for (Eigen::Index s = 0; s < ns; s++)
        {
            if (s != k)
            {
                Eigen::Index i = inverse_idx[s];
                // epsilon prior (only for off-diagonal)
                grad(i) += -spatial.epsilon(s, k);
                loglike += - 0.5 * spatial.epsilon(s, k) * spatial.epsilon(s, k);
                
                // gamma prior (only for off-diagonal)
                double gm2 = spatial.gamma(s, k) * spatial.gamma(s, k);
                grad(idx_start(2) + i) += (1.0 - gm2) / (1.0 + gm2);
                loglike += std::log(spatial.gamma(s, k)) - std::log(1.0 + gm2);
            }

            // Likelihood contribution from destination (t, s)
            for (Eigen::Index t = 0; t < nt + 1; t++)
            {
                if (zinfl.inflated && zinfl.Z(t, s) < 1)
                {
                    // If zero-inflated and Z(t,s) == 0, skip likelihood contribution
                    continue;
                }

                double lambda_st = mu;
                double dlambda_st_dalpha_sk = 0.0;

                for (Eigen::Index kk = 0; kk < ns; kk++)
                {
                    for (Eigen::Index l = 0; l < t; l++)
                    {
                        if (t - l <= static_cast<Eigen::Index>(dlag.Fphi.size()))
                        {
                            double coef = R_mat(l, kk) * dlag.Fphi(t - l - 1) * Y(l, kk);
                            if (kk == k)
                            {
                                dlambda_st_dalpha_sk += coef;
                            }
                            lambda_st += spatial.alpha(s, kk) * coef;
                        }
                    }
                }

                lambda_st = std::max(lambda_st, EPS);
                loglike += Y(t, s) * std::log(lambda_st) - lambda_st;

                double dloglike_dlambda_st = Y(t, s) / lambda_st - 1.0;
                const double dloglike_dalpha = dloglike_dlambda_st * dlambda_st_dalpha_sk;

                for (Eigen::Index j = 0; j < ns; j++)
                {
                    if (j == k)
                        continue;
                    Eigen::Index i = inverse_idx[j];

                    // Gradient w.r.t. epsilon for s != k
                    double dalpha_sk_depsilon_jk = spatial.dalpha_depsilon(s, j, k, u_k);
                    grad(i) += dloglike_dalpha * dalpha_sk_depsilon_jk;

                    // Gradient w.r.t. log(gamma(s, k)) for s != k
                    double dalpha_sk_dgamma_jk = spatial.dalpha_dgamma(s, j, k, u_k);
                    grad(idx_start(2) + i) += dloglike_dalpha * dalpha_sk_dgamma_jk * spatial.gamma(j, k);
                }


                // Gradient w.r.t. logit(wdiag(k))
                double dalpha_sk_dwk = spatial.dalpha_dwj(s, k, u_k);
                grad(idx_start(1)) += dloglike_dalpha * dalpha_sk_dwk * jacobian_wk;

                // Gradient w.r.t. log(tau(k))
                double dalpha_sk_dtau_k = spatial.dalpha_dtau(s, k, u_k);
                grad(idx_start(3)) += dloglike_dalpha * dalpha_sk_dtau_k * spatial.tau(k);

            } // for time t
        } // for destination locations s

        // Compute FULL gradients for rho parameters (sum over ALL columns)
        if (idx_start(4) >= 0)
        {
            grad(idx_start(4)) += compute_grad_rho_dist_full(Y, R_mat) * spatial.rho_dist;
        }

        if (idx_start(5) >= 0)
        {
            grad(idx_start(5)) += compute_grad_rho_mobility_full(Y, R_mat) * spatial.rho_mobility;
        }

        return grad;
    } // dlogprob_dcolvec


    /**
     * @brief HMC update for horseshoe parameters of column k, including rho parameters
     *
     * @param energy_diff Output: Hamiltonian energy difference
     * @param grad_norm Output: Gradient norm at start
     * @param k Column index
     * @param Y Observed infections
     * @param R_mat Reproduction numbers
     * @param mass_diag Mass matrix diagonal
     * @param step_size HMC step size
     * @param n_leapfrog Number of leapfrog steps
     * @param include_rho_dist Include rho_dist in the update
     * @param include_rho_mobility Include rho_mobility in the update
     * @param beta_prior_a Beta prior parameter for wdiag
     * @param beta_prior_b Beta prior parameter for wdiag
     * @param rho_prior_mean Prior mean for log(rho)
     * @param rho_prior_var Prior variance for log(rho)
     * @return Acceptance probability
     */
    double update_colvec(
        double &energy_diff,
        double &grad_norm,
        const Eigen::Index &k,
        const Eigen::MatrixXd &Y,
        const Eigen::MatrixXd &R_mat,
        const Eigen::VectorXd &mass_diag,
        const double &step_size,
        const unsigned int &n_leapfrog,
        const bool &include_rho_dist = true,
        const bool &include_rho_mobility = false,
        const double &beta_prior_a = 5.0,
        const double &beta_prior_b = 2.0,
        const double &rho_prior_mean = 0.0,
        const double &rho_prior_var = 9.0)
    {
        Eigen::VectorXi idx_start, inverse_idx;
        Eigen::Index ndim = get_ndim_colvec(
            idx_start,
            inverse_idx,
            k,
            include_rho_dist,
            include_rho_mobility
        );

        Eigen::VectorXd inv_mass = mass_diag.cwiseInverse();
        Eigen::VectorXd sqrt_mass = mass_diag.array().sqrt();

        // Get unconstrained parameters
        Eigen::ArrayXd unconstrained_current = get_unconstrained_colvec(k, ndim, idx_start, inverse_idx);
        Eigen::ArrayXd unconstrained_params = unconstrained_current;

        // Compute initial gradient and log-probability
        double current_logprob = 0.0;
        Eigen::VectorXd grad = dlogprob_dcolvec(
            current_logprob, k, Y, R_mat,
            include_rho_dist, 
            include_rho_mobility,
            beta_prior_a, beta_prior_b, 
            rho_prior_mean, rho_prior_var
        );
        grad_norm = grad.norm();
        double current_energy = -current_logprob;

        // Sample momentum
        Eigen::VectorXd momentum(ndim);
        for (Eigen::Index i = 0; i < ndim; i++)
        {
            momentum(i) = sqrt_mass(i) * R::rnorm(0.0, 1.0);
        }
        double current_kinetic = 0.5 * momentum.cwiseProduct(inv_mass).dot(momentum);

        // Leapfrog integration
        momentum += 0.5 * step_size * grad;
        for (unsigned int lf_step = 0; lf_step < n_leapfrog; lf_step++)
        {
            // Update position
            unconstrained_params += step_size * inv_mass.array() * momentum.array();

            // Unpack to model
            unpack_unconstrained_colvec(unconstrained_params, k, idx_start, inverse_idx);

            // Compute new gradient
            grad = dlogprob_dcolvec(
                current_logprob, k, Y, R_mat,
                include_rho_dist, 
                include_rho_mobility,
                beta_prior_a, beta_prior_b, 
                rho_prior_mean, rho_prior_var
            );

            // Update momentum (except last step)
            if (lf_step != n_leapfrog - 1)
            {
                momentum += step_size * grad;
            }
        }

        momentum += 0.5 * step_size * grad;
        momentum = -momentum; // Negate for reversibility

        double proposed_energy = -current_logprob;
        double proposed_kinetic = 0.5 * momentum.cwiseProduct(inv_mass).dot(momentum);

        double H_proposed = proposed_energy + proposed_kinetic;
        double H_current = current_energy + current_kinetic;
        energy_diff = H_proposed - H_current;

        // MH acceptance
        bool accept = false;
        double accept_prob = 0.0;
        if (std::isfinite(energy_diff) && std::abs(energy_diff) < 1.0e4)
        {
            accept_prob = std::min(1.0, std::exp(-energy_diff));
            if (std::log(R::runif(0.0, 1.0)) < -energy_diff)
            {
                accept = true;
            }
        }

        if (!accept)
        {
            // Revert to previous constrained parameters
            unpack_unconstrained_colvec(unconstrained_current, k, idx_start, inverse_idx);
        }

        return accept_prob;
    } // update_colvec


    Eigen::VectorXd dloglike_dparams_collapsed(
        double &loglike,
        const Eigen::MatrixXd &Y, // (nt + 1) x ns, observed primary infections
        const Eigen::MatrixXd &R_mat, // (nt + 1) x ns
        const bool &add_jacobian = true,
        const bool &include_W = false
    )
    {
        double W_safe = std::max(temporal.W, EPS);
        double W_sqrt = std::sqrt(W_safe);

        Eigen::MatrixXd deriv_rho1_mat, deriv_rho2_mat;
        if (spatial.include_distance)
        {
            deriv_rho1_mat = spatial.dalpha_drho_dist();
        }
        if (spatial.include_log_mobility)
        {
            deriv_rho2_mat = spatial.dalpha_drho_mobility();
        }

        double deriv_mu = 0.0;
        double deriv_W = 0.0;
        double deriv_rho_dist = 0.0;
        double deriv_rho_mobility = 0.0;
        loglike = 0.0;
        for (Eigen::Index t = 0; t < nt + 1; t++)
        { // for destination time t
            for (Eigen::Index s = 0; s < ns; s++)
            { // for destination location s
                if (zinfl.inflated && zinfl.Z(t, s) < 1)
                {
                    // If zero-inflated and Z(t,s) == 0, skip likelihood contribution
                    continue;
                }

                double lambda_st = mu;
                double dlambda_st_drho_dist = 0.0;
                double dlambda_st_drho_mobility = 0.0;
                for (Eigen::Index k = 0; k < ns; k++)
                {
                    double coef_sum = 0.0;
                    for (Eigen::Index l = 0; l < t; l++)
                    {
                        if (t - l <= static_cast<Eigen::Index>(dlag.Fphi.size()))
                        {
                            double coef = R_mat(l, k) * dlag.Fphi(t - l - 1) * Y(l, k);
                            coef_sum += coef;
                            lambda_st += spatial.alpha(s, k) * coef;
                        }
                    } // for source time l < t

                    if (spatial.include_distance)
                    {
                        dlambda_st_drho_dist += deriv_rho1_mat(s, k) * coef_sum;
                    }
                    if (spatial.include_log_mobility)
                    {
                        dlambda_st_drho_mobility += deriv_rho2_mat(s, k) * coef_sum;
                    }
                } // for source location k

                lambda_st = std::max(lambda_st, EPS);
                double dloglike_dlambda_st = Y(t, s) / lambda_st - 1.0;
                deriv_mu += dloglike_dlambda_st;

                loglike += Y(t, s) * std::log(lambda_st) - lambda_st;
                if (include_W && t > 0)
                {
                    deriv_W += - 0.5 / W_safe + 0.5 * (temporal.wt(t, s) * temporal.wt(t, s)) / (W_safe * W_safe);
                    loglike += R::dnorm4(temporal.wt(t, s), 0.0, W_sqrt, true);
                }

                if (spatial.include_distance)
                {
                    deriv_rho_dist += dloglike_dlambda_st * dlambda_st_drho_dist;
                }
                if (spatial.include_log_mobility)
                {
                    deriv_rho_mobility += dloglike_dlambda_st * dlambda_st_drho_mobility;
                }
            } // for destination location s
        } // for time t

        if (add_jacobian)
        {
            deriv_mu *= mu;

            if (include_W)
            {
                deriv_W *= temporal.W;
            } // if include_W

            if (spatial.include_distance)
            {
                deriv_rho_dist *= spatial.rho_dist;
            }

            if (spatial.include_log_mobility)
            {
                deriv_rho_mobility *= spatial.rho_mobility;
            }
        } // if add_jacobian

        Eigen::Index n_params = 1;
        if (spatial.include_distance)
        {
            n_params += 1;
        }
        if (spatial.include_log_mobility)
        {
            n_params += 1;
        }
        if (include_W)
        {
            n_params += 1;
        }

        Eigen::VectorXd derivs(n_params);
        Eigen::Index idx = 0;
        derivs(idx) = deriv_mu;
        idx += 1;
        if (spatial.include_distance)
        {
            derivs(idx) = deriv_rho_dist;
            idx += 1;
        }
        if (spatial.include_log_mobility)
        {
            derivs(idx) = deriv_rho_mobility;
            idx += 1;
        }
        if (include_W)
        {
            derivs(idx) = deriv_W;
        }

        return derivs;
    } // dloglike_dparams_collapsed


    Eigen::Index get_ndim_static_params(const bool &include_W = false)
    {
        Eigen::Index n_params = 1; // mu
        if (spatial.include_distance)
        {
            n_params += 1;
        }
        if (spatial.include_log_mobility)
        {
            n_params += 1;
        }
        if (include_W)
        {
            n_params += 1;
        }
        return n_params;
    } // get_ndim_static_params


    Eigen::VectorXd get_constrained_static_params(const bool &include_W = false)
    {
        Eigen::Index n_params = get_ndim_static_params(include_W);
        Eigen::VectorXd constrained_params(n_params);
        Eigen::Index idx = 0;
        constrained_params(idx) = mu;
        idx += 1;
        if (spatial.include_distance)
        {
            constrained_params(idx) = spatial.rho_dist;
            idx += 1;
        }
        if (spatial.include_log_mobility)
        {
            constrained_params(idx) = spatial.rho_mobility;
            idx += 1;
        }
        if (include_W)
        {
            constrained_params(idx) = temporal.W;
        }
        return constrained_params;
    } // get_constrained_static_params


    void unpack_constrained_static_params(
        const Eigen::ArrayXd &constrained_params,
        const bool &include_W = false
    )
    {
        Eigen::Index idx = 0;
        mu = constrained_params(idx);
        idx += 1;
        if (spatial.include_distance)
        {
            spatial.rho_dist = constrained_params(idx);
            idx += 1;
        }
        if (spatial.include_log_mobility)
        {
            spatial.rho_mobility = constrained_params(idx);
            idx += 1;
        }
        if (spatial.include_distance || spatial.include_log_mobility)
        {
            spatial.compute_alpha();
        }
        if (include_W)
        {
            temporal.W = constrained_params(idx);
        }
        return;
    } // unpack_unconstrained_static_params


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
        Eigen::VectorXd inv_mass = mass_diag.cwiseInverse();
        Eigen::VectorXd sqrt_mass = mass_diag.array().sqrt();

        // Current state
        Eigen::VectorXd params_current = get_constrained_static_params(include_W);

        // Compute current energy and gradiant
        Eigen::ArrayXd log_params = params_current.array().log();
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

            Eigen::VectorXd constrained_params = log_params.exp();
            unpack_constrained_static_params(constrained_params, include_W);

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
        double accept_prob = 0.0;
        bool accept = false;
        if (std::isfinite(energy_diff) && std::abs(energy_diff) < 100.0)
        {
            accept_prob = std::min(1.0, std::exp(-energy_diff));
            if (std::log(R::runif(0.0, 1.0)) < -energy_diff)
            {
                accept = true;
            }
        }

        if (!accept)
        {
            // Revert to current state
            unpack_constrained_static_params(params_current, include_W);
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


    /**
     * @brief Simulate data from the model using regularized horseshoe
     */
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
        const Eigen::MatrixXd Rt = temporal.compute_Rt();

        if (zinfl.inflated)
        {
            zinfl.simulate();
        }

        // Simulate primary infections at time t = 0
        Eigen::MatrixXd Y(nt + 1, ns);
        Y.setZero();
        for (Eigen::Index k = 0; k < ns; k++)
        {
            Y(0, k) = R::rpois(mu + EPS);
        }

        // Simulate secondary infections N and observed primary infections Y over time
        for (Eigen::Index t = 1; t < nt + 1; t++)
        {
            for (Eigen::Index s = 0; s < ns; s++)
            {
                N0(t, s) = R::rpois(std::max(mu, EPS));
                double y_ts = N0(t, s);

                for (Eigen::Index k = 0; k < ns; k++)
                {
                    double a_sk = spatial.alpha(s, k);
                    for (Eigen::Index l = 0; l < t; l++)
                    {
                        if (t - l <= dlag.Fphi.size())
                        {
                            double lag_prob = dlag.Fphi(t - l - 1);
                            double R_kl = Rt(l, k);
                            double lambda_sktl = a_sk * R_kl * lag_prob * Y(l, k) + EPS;
                            N(s, k, t, l) = R::rpois(lambda_sktl);
                        }
                        else
                        {
                            N(s, k, t, l) = 0.0;
                        }

                        y_ts += N(s, k, t, l);
                    } // for l < t
                } // for source location k

                Y(t, s) = (zinfl.Z(t, s) > 0 || !zinfl.inflated) ? y_ts : 0.0;
            } // for destination location s
        } // for time t

        Rcpp::List params_list = Rcpp::List::create(
            Rcpp::Named("mu") = mu,
            Rcpp::Named("W") = temporal.W
        );


        return Rcpp::List::create(
            Rcpp::Named("Y") = Y,
            Rcpp::Named("N") = tensor4_to_r(N),
            Rcpp::Named("N0") = N0,
            Rcpp::Named("alpha") = spatial.alpha,
            Rcpp::Named("wt") = temporal.wt,
            Rcpp::Named("Rt") = Rt,
            Rcpp::Named("params") = params_list,
            Rcpp::Named("horseshoe") = spatial.to_list(),
            Rcpp::Named("zero_inflation") = zinfl.to_list()
        );
    } // simulate


    /**
     * @brief Initialize wt using data-driven estimate of R_t
     *
     * Structure: wt is (nt+1) x ns
     *   - wt(0, k) = 0 always (zero-padded, not a parameter)
     *   - wt(t, k) = sqrt(W) * eta_{k,t} for t = 1, ..., nt
     *   - r_{k,l} = sum_{t=1}^{l} wt(t, k) = sqrt(W) * sum_{t=1}^{l} eta_{k,t}
     *   - R_{k,l} = h(r_{k,l})
     */
    void initialize_wt_from_data(
        const Eigen::MatrixXd &Y, // (nt+1) x ns
        const double &R_default = 1.0,
        const double &smoothing = 0.5 // Exponential smoothing parameter
    )
    {
        const Eigen::Index L = static_cast<Eigen::Index>(dlag.Fphi.size());

        temporal.wt.resize(nt + 1, ns);
        temporal.wt.setZero(); // Row 0 stays zero

        for (Eigen::Index k = 0; k < ns; k++)
        {
            // Step 1: Estimate R_t from data for t = 1, ..., nt
            Eigen::VectorXd R_est(nt + 1);
            R_est(0) = R_default; // Not used, but initialize for smoothing

            for (Eigen::Index t = 1; t <= nt; t++)
            {
                // Compute weighted sum of past cases
                double denom = 0.0;
                Eigen::Index l_start = std::max(Eigen::Index(1), t - L);
                for (Eigen::Index l = l_start; l < t; l++)
                {
                    Eigen::Index lag = t - l;
                    if (lag <= L)
                    {
                        denom += dlag.Fphi(lag - 1) * Y(l, k);
                    }
                }

                // Estimate R_t = y_t / (sum_l phi_l * y_{t-l})
                if (denom > EPS)
                {
                    double R_raw = std::max(Y(t, k) - mu, EPS) / denom;
                    // Clamp to reasonable range
                    R_raw = std::max(0.1, std::min(R_raw, 10.0));
                    // Exponential smoothing
                    R_est(t) = smoothing * R_raw + (1.0 - smoothing) * R_est(t - 1);
                }
                else
                {
                    R_est(t) = R_est(t - 1); // Carry forward
                }
            }

            // Step 2: Convert R_t to r_t = h^{-1}(R_t) for t = 1, ..., nt
            // Note: r_0 = 0 by definition (since wt(0, k) = 0)
            Eigen::VectorXd r_est(nt + 1);
            r_est(0) = 0.0; // r_0 = 0
            for (Eigen::Index t = 1; t <= nt; t++)
            {
                r_est(t) = GainFunc::hpsi2psi(R_est(t), temporal.fgain);
            }

            // Step 3: Compute increments
            // wt(t, k) = r_t - r_{t-1}
            // Note: wt(0, k) = 0 is already set by setZero()
            for (Eigen::Index t = 1; t <= nt; t++)
            {
                temporal.wt(t, k) = r_est(t) - r_est(t - 1);
            }
        }
    } // initialize_wt_from_data


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


    void compute_auxiliary(
        Eigen::MatrixXd &C_mat,      // (nt-1) x (nt-1), lower triangular
        Eigen::MatrixXd &mu_vec,     // (nt-1) x ns
        Eigen::MatrixXd &lambda_mat, // (nt-1) x ns
        const Eigen::Index &k,
        const Eigen::MatrixXd &Y,
        const Eigen::MatrixXd &R_mat,
        const Eigen::MatrixXd &dR_mat)
    {
        const Eigen::Index T_obs = nt - 1; // Number of observations t = 2, ..., nt
        const double W_sqrt = std::sqrt(temporal.W + EPS);
        const Eigen::Index L = static_cast<Eigen::Index>(dlag.Fphi.size());

        // Resize outputs to (nt-1) dimensions
        C_mat.resize(T_obs, T_obs);
        C_mat.setZero();
        mu_vec.resize(T_obs, ns);
        lambda_mat.resize(T_obs, ns);

        // Precompute a_l = h'(r_{k,l}) * y_{k,l} for l = 1, ..., nt-1
        Eigen::VectorXd a_vec(T_obs);
        for (Eigen::Index l = 1; l < nt; l++)
        {
            a_vec(l - 1) = dR_mat(l, k) * Y(l, k);
        }

        // =========================================================
        // Step 1: Compute C_k matrix using column recurrence
        // c_{k,j,t} = sum_{l=j}^{t-1} h'(r_{k,l}) * phi_{t-l} * y_{k,l}
        // Recurrence: c_{k,j+1,t} = c_{k,j,t} - a_j * phi_{t-j}
        //
        // Matrix indexing: C_mat(row, col) where
        //   row = t - 2  (t = 2, ..., nt  =>  row = 0, ..., nt-2)
        //   col = j - 1  (j = 1, ..., t-1 =>  col = 0, ..., t-2)
        // =========================================================
        for (Eigen::Index t = 2; t <= nt; t++)
        {
            Eigen::Index row = t - 2;

            // First column (j = 1): compute c_{k,1,t} = sum_{l=1}^{t-1} a_l * phi_{t-l}
            double c_val = 0.0;
            Eigen::Index l_start = std::max(Eigen::Index(1), t - L);
            for (Eigen::Index l = l_start; l < t; l++)
            {
                Eigen::Index lag = t - l;
                c_val += a_vec(l - 1) * dlag.Fphi(lag - 1);
            }
            C_mat(row, 0) = c_val;

            // Remaining columns via recurrence
            for (Eigen::Index j = 2; j < t; j++)
            {
                Eigen::Index col = j - 1;
                Eigen::Index lag = t - (j - 1); // = t - j + 1

                double subtract_term = 0.0;
                if (lag >= 1 && lag <= L)
                {
                    subtract_term = a_vec(j - 2) * dlag.Fphi(lag - 1);
                }
                C_mat(row, col) = C_mat(row, col - 1) - subtract_term;
            }
        }

        // =========================================================
        // Step 2: Compute lambda_{s,t} for t = 2, ..., nt
        // lambda_{s,t} = mu + sum_{kp} sum_{l} alpha_{s,kp} * R_{kp,l} * phi_{t-l} * y_{kp,l}
        // =========================================================
        lambda_mat.setConstant(mu);

        for (Eigen::Index t = 2; t <= nt; t++)
        {
            Eigen::Index row = t - 2;
            Eigen::Index l_start = std::max(Eigen::Index(1), t - L);

            for (Eigen::Index l = l_start; l < t; l++)
            {
                Eigen::Index lag = t - l;
                double phi_lag = dlag.Fphi(lag - 1);

                for (Eigen::Index kp = 0; kp < ns; kp++)
                {
                    double contrib = R_mat(l, kp) * phi_lag * Y(l, kp);

                    for (Eigen::Index s = 0; s < ns; s++)
                    {
                        lambda_mat(row, s) += spatial.alpha(s, kp) * contrib;
                    }
                }
            }
        }

        lambda_mat = lambda_mat.cwiseMax(EPS);

        // =========================================================
        // Step 3: Precompute C_k * eta_k (reused for all destinations s)
        // =========================================================
        Eigen::VectorXd eta_k = temporal.wt.col(k) / W_sqrt; // (nt+1) x 1

        // Extract eta_k[1:nt-1] into (nt-1) x 1 vector
        Eigen::VectorXd eta_k_sub(T_obs);
        for (Eigen::Index j = 1; j < nt; j++)
        {
            eta_k_sub(j - 1) = eta_k(j);
        }

        // C_eta = C_mat * eta_k_sub (lower triangular matrix-vector product)
        Eigen::VectorXd C_eta = Eigen::VectorXd::Zero(T_obs);
        for (Eigen::Index row = 0; row < T_obs; row++)
        {
            for (Eigen::Index col = 0; col <= row; col++)
            {
                C_eta(row) += C_mat(row, col) * eta_k_sub(col);
            }
        }

        // =========================================================
        // Step 4: Compute mu_{s,t} = lambda_{s,t} - sqrt(W) * alpha_{s,k} * (C_k * eta_k)_t
        // =========================================================
        for (Eigen::Index s = 0; s < ns; s++)
        {
            double scale = W_sqrt * spatial.alpha(s, k);

            for (Eigen::Index row = 0; row < T_obs; row++)
            {
                mu_vec(row, s) = std::max(lambda_mat(row, s) - scale * C_eta(row), EPS);
            }
        }
    } // compute_auxiliary


    /**
     * @brief Compute MH proposal parameters for block update of eta_k
     *
     * Computes precision matrix, its Cholesky decomposition, the mean, and log determinant
     */
    void compute_mh_proposal(
        BlockMHProposal &proposal,
        const Eigen::MatrixXd &C_mat,      // (nt-1) x (nt-1), lower triangular
        const Eigen::MatrixXd &mu_vec,     // (nt-1) x ns
        const Eigen::MatrixXd &lambda_mat, // (nt-1) x ns
        const Eigen::MatrixXd &Y,          // (nt+1) x ns
        const Eigen::Index &k)
    {
        const Eigen::Index T_obs = nt - 1;
        const double W = temporal.W + EPS;
        const double W_sqrt = std::sqrt(W);

        // Initialize precision as identity
        proposal.prec = Eigen::MatrixXd::Identity(T_obs, T_obs);
        Eigen::VectorXd canonical_mean = Eigen::VectorXd::Zero(T_obs); // Omega * u

        // Accumulate contributions from each destination
        for (Eigen::Index s = 0; s < ns; s++)
        {
            const double alpha_sk = spatial.alpha(s, k);
            if (std::abs(alpha_sk) < EPS)
                continue;

            const double coef = W_sqrt * alpha_sk;
            const double coef_sq = W * alpha_sk * alpha_sk;

            // d_s = y_s - mu_s
            Eigen::VectorXd d_s = Y.col(s).segment(2, T_obs) - mu_vec.col(s);

            // lambda^{-1} element-wise
            Eigen::VectorXd lambda_inv = lambda_mat.col(s).cwiseInverse();

            // Canonical mean: += coef * C' * (lambda_inv .* d_s)
            Eigen::VectorXd weighted_d = lambda_inv.cwiseProduct(d_s);
            canonical_mean.noalias() += coef * (C_mat.transpose() * weighted_d);

            // Precision: += coef^2 * C' * diag(lambda_inv) * C
            // Use scaled C for efficient computation
            Eigen::MatrixXd C_scaled = lambda_inv.cwiseSqrt().asDiagonal() * C_mat;
            proposal.prec.noalias() += coef_sq * (C_scaled.transpose() * C_scaled);
        }

        // Cholesky decomposition of precision: Omega = L * L'
        proposal.chol.compute(proposal.prec);
        proposal.valid = (proposal.chol.info() == Eigen::Success);

        if (proposal.valid)
        {
            // Log determinant: log|Omega| = 2 * sum(log(diag(L)))
            Eigen::VectorXd chol_diag = proposal.chol.matrixL().toDenseMatrix().diagonal();
            proposal.log_det_prec = 2.0 * chol_diag.array().log().sum();

            // Solve for mean: Omega * u = canonical_mean  =>  u = Omega^{-1} * canonical_mean
            proposal.mean = proposal.chol.solve(canonical_mean);
        }
        else
        {
            proposal.log_det_prec = -std::numeric_limits<double>::infinity();
            proposal.mean = Eigen::VectorXd::Zero(T_obs);
            Rcpp::warning("Cholesky decomposition failed in compute_mh_proposal");
        }
    } // compute_mh_proposal


    /**
     * @brief Compute log prior density for eta_k: N(0, I)
     */
    double log_prior_eta(const Eigen::VectorXd &eta)
    {
        Eigen::Index n = eta.size();
        static const double LOG_2PI = std::log(2.0 * M_PI);
        return -0.5 * n * LOG_2PI - 0.5 * eta.squaredNorm();
    } // log_prior_eta


    /**
     * @brief Compute log likelihood for observations given eta_k
     *
     * sum_{s,t} [y_{s,t} * log(lambda_{s,t}) - lambda_{s,t}]
     */
    double log_likelihood_poisson(
        const Eigen::MatrixXd &Y,         // (nt+1) x ns
        const Eigen::MatrixXd &lambda_mat // (nt-1) x ns, for t = 2, ..., nt
    )
    {
        const Eigen::Index T_obs = lambda_mat.rows();
        double loglik = 0.0;

        for (Eigen::Index s = 0; s < lambda_mat.cols(); s++)
        {
            for (Eigen::Index row = 0; row < T_obs; row++)
            {
                Eigen::Index t = row + 2; // Observation time
                if (zinfl.inflated && zinfl.Z(t, s) < 1)
                {
                    continue;
                }

                double y_st = Y(t, s);
                double lam_st = lambda_mat(row, s);
                loglik += y_st * std::log(lam_st) - lam_st;
            }
        }

        return loglik;
    } // log_likelihood_poisson


    /**
     * @brief Recompute lambda given new eta values (for MH acceptance)
     *
     * lambda_{s,t} = mu_{s,t} + sqrt(W) * alpha_{s,k} * sum_j C(t,j) * eta_j
     */
    void recompute_lambda(
        Eigen::MatrixXd &lambda_new,    // Output: (nt-1) x ns
        const Eigen::MatrixXd &C_mat,   // (nt-1) x (nt-1)
        const Eigen::MatrixXd &mu_vec,  // (nt-1) x ns
        const Eigen::VectorXd &eta_new, // (nt-1) x 1
        const Eigen::Index &k)
    {
        const Eigen::Index T_obs = C_mat.rows();
        const double W_sqrt = std::sqrt(temporal.W + EPS);

        // Compute C * eta_new (lower triangular)
        Eigen::VectorXd C_eta = Eigen::VectorXd::Zero(T_obs);
        for (Eigen::Index row = 0; row < T_obs; row++)
        {
            for (Eigen::Index col = 0; col <= row; col++)
            {
                C_eta(row) += C_mat(row, col) * eta_new(col);
            }
        }

        // lambda_{s,t} = mu_{s,t} + sqrt(W) * alpha_{s,k} * (C * eta)_t
        lambda_new.resize(T_obs, ns);
        for (Eigen::Index s = 0; s < ns; s++)
        {
            double scale = W_sqrt * spatial.alpha(s, k);
            for (Eigen::Index row = 0; row < T_obs; row++)
            {
                lambda_new(row, s) = std::max(mu_vec(row, s) + scale * C_eta(row), EPS);
            }
        }
    } // recompute_lambda


    /**
     * @brief Block MH update for eta_k (all time points for source location k)
     *
     * @return Acceptance probability (0 or 1)
     */
    double update_eta_block_mh(
        const Eigen::MatrixXd &Y,
        const Eigen::Index &k,
        const double &mh_sd = 1.0)
    {
        const Eigen::Index T_obs = nt - 1;
        const double W_sqrt = std::sqrt(temporal.W + EPS);

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

        // Current eta_k (indices 1, ..., nt-1)
        Eigen::VectorXd eta_old = eta.col(k).segment(1, T_obs);

        // =========================================================
        // Step 1: Compute auxiliary quantities at current state
        // =========================================================
        Eigen::MatrixXd C_mat, mu_vec_old, lambda_mat_old;
        compute_auxiliary(C_mat, mu_vec_old, lambda_mat_old, k, Y, R_mat, dR_mat);

        // =========================================================
        // Step 2: Compute proposal at current state and sample
        // =========================================================
        BlockMHProposal proposal_old;
        compute_mh_proposal(proposal_old, C_mat, mu_vec_old, lambda_mat_old, Y, k);

        if (!proposal_old.valid)
        {
            return 0.0; // Reject if Cholesky failed
        }

        Eigen::VectorXd eta_new = proposal_old.sample(mh_sd);

        // =========================================================
        // Step 3: Compute lambda at proposed state
        // =========================================================
        Eigen::MatrixXd lambda_mat_new;
        recompute_lambda(lambda_mat_new, C_mat, mu_vec_old, eta_new, k);

        // Check for numerical issues
        if ((lambda_mat_new.array() <= 0).any() || !lambda_mat_new.allFinite())
        {
            return 0.0;
        }

        // =========================================================
        // Step 4: Compute reverse proposal (linearized at new state)
        // Need to recompute C_mat and mu_vec at new state
        // =========================================================

        // Temporarily update wt to compute R_mat and dR_mat at new state
        Eigen::VectorXd wt_old = temporal.wt.col(k);
        for (Eigen::Index j = 1; j < nt; j++)
        {
            temporal.wt(j, k) = W_sqrt * eta_new(j - 1);
        }

        Eigen::MatrixXd R_mat_new = temporal.compute_Rt();
        Eigen::MatrixXd dR_mat_new(nt + 1, ns);
        for (Eigen::Index kp = 0; kp < ns; kp++)
        {
            Eigen::VectorXd wt_cumsum = cumsum_vec(temporal.wt.col(kp));
            for (Eigen::Index l = 0; l < nt + 1; l++)
            {
                dR_mat_new(l, kp) = GainFunc::psi2dhpsi(wt_cumsum(l), temporal.fgain);
            }
        }

        Eigen::MatrixXd C_mat_new, mu_vec_new, lambda_mat_new_check;
        compute_auxiliary(C_mat_new, mu_vec_new, lambda_mat_new_check, k, Y, R_mat_new, dR_mat_new);

        BlockMHProposal proposal_new;
        compute_mh_proposal(proposal_new, C_mat_new, mu_vec_new, lambda_mat_new_check, Y, k);

        // Restore wt
        temporal.wt.col(k) = wt_old;

        if (!proposal_new.valid)
        {
            return 0.0;
        }

        // =========================================================
        // Step 5: Compute MH acceptance ratio
        // =========================================================
        // log p(eta_new) - log p(eta_old) [prior]
        double log_prior_ratio = log_prior_eta(eta_new) - log_prior_eta(eta_old);

        // log p(y | eta_new) - log p(y | eta_old) [likelihood]
        double log_lik_new = log_likelihood_poisson(Y, lambda_mat_new);
        double log_lik_old = log_likelihood_poisson(Y, lambda_mat_old);
        double log_lik_ratio = log_lik_new - log_lik_old;

        // log q(eta_old | eta_new) - log q(eta_new | eta_old) [proposal]
        double log_q_old_given_new = proposal_new.log_density(eta_old);
        double log_q_new_given_old = proposal_old.log_density(eta_new);
        double log_proposal_ratio = log_q_old_given_new - log_q_new_given_old;

        double log_accept_ratio = log_prior_ratio + log_lik_ratio + log_proposal_ratio;

        // =========================================================
        // Step 6: Accept or reject
        // =========================================================
        double accept_prob = 0.0;
        bool accept = false;

        if (std::isfinite(log_accept_ratio))
        {
            accept_prob = std::min(1.0, std::exp(log_accept_ratio));
            if (std::log(R::runif(0.0, 1.0)) < log_accept_ratio)
            {
                accept = true;
            }
        }

        if (accept)
        {
            // Update wt with accepted values
            for (Eigen::Index j = 1; j < nt; j++)
            {
                temporal.wt(j, k) = W_sqrt * eta_new(j - 1);
            }
        }

        return accept ? 1.0 : 0.0;
    } // update_eta_block_mh


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
        initialize_wt_from_data(Y);
        spatial.rho_dist = R::runif(0.0, 2.0);
        spatial.rho_mobility = R::runif(0.0, 2.0);
        spatial.initialize_horseshoe_sparse();
        spatial.compute_alpha();

        // Infer all unknown parameters by default
        Prior wt_prior; wt_prior.infer = true; wt_prior.mh_sd = 1.0;

        Prior static_params_prior("gaussian", 0.0, 3.0, true); // prior for log(static parameters)
        bool hmc_include_W = false;
        HMCOpts_1d hmc_opts;
        mu = Y.block(1, 0, nt, ns).array().minCoeff() + EPS; // initialize mu to min observed primary infections

        Prior hs_prior("gaussian", 0.0, 3.0, true); // prior for horseshoe per column
        HMCOpts_2d hs_hmc_opts;

        Prior zinfl_prior("gaussian", 0.0, 3.0, false); // prior for zero-inflation parameter
        HMCOpts_1d zinfl_hmc_opts;


        if (mcmc_opts.isNotNull())
        {
            // Initialize priors and initial values from mcmc_opts
            Rcpp::List mcmc_opts_list(mcmc_opts);
            if (mcmc_opts_list.containsElementNamed("params"))
            {
                Rcpp::List params_opts = mcmc_opts_list["params"];
                
                if (params_opts.containsElementNamed("prior_param"))
                {
                    static_params_prior = Prior(params_opts);
                }
                if (params_opts.containsElementNamed("include_W"))
                {
                    hmc_include_W = Rcpp::as<bool>(params_opts["include_W"]);
                }
                if (params_opts.containsElementNamed("hmc"))
                {
                    Rcpp::List hmc_params_opts = params_opts["hmc"];
                    hmc_opts = HMCOpts_1d(hmc_params_opts);
                }
                if (params_opts.containsElementNamed("init"))
                {
                    Rcpp::List init_values = params_opts["init"];
                    spatial.compute_alpha();

                    if (init_values.containsElementNamed("mu"))
                    {
                        mu = Rcpp::as<double>(init_values["mu"]);
                    }
                    if (init_values.containsElementNamed("W"))
                    {
                        temporal.W = Rcpp::as<double>(init_values["W"]);
                        initialize_wt_from_data(Y);
                    }
                    if (init_values.containsElementNamed("rho_dist"))
                    {
                        spatial.rho_dist = Rcpp::as<double>(init_values["rho_dist"]);
                        spatial.compute_alpha();
                    }
                    if (init_values.containsElementNamed("rho_mobility"))
                    {
                        spatial.rho_mobility = Rcpp::as<double>(init_values["rho_mobility"]);
                        spatial.compute_alpha();
                    }
                } // if init
            } // if params

            if (mcmc_opts_list.containsElementNamed("wt"))
            {
                Rcpp::List wt_opts = mcmc_opts_list["wt"];
                wt_prior = Prior(wt_opts);
                if (wt_opts.containsElementNamed("init"))
                {
                    Rcpp::NumericMatrix wt_mat = wt_opts["init"];
                    temporal.wt = Rcpp::as<Eigen::MatrixXd>(wt_mat);
                    temporal.W = temporal.wt.array().square().mean();
                }
            } // if wt

            if (mcmc_opts_list.containsElementNamed("horseshoe"))
            {
                Rcpp::List hs_opts = mcmc_opts_list["horseshoe"];
                hs_prior = Prior(hs_opts);

                if (hs_opts.containsElementNamed("hmc"))
                {
                    Rcpp::List hmc_hs_opts = hs_opts["hmc"];
                    hs_hmc_opts = HMCOpts_2d(hmc_hs_opts);
                }
                if (hs_opts.containsElementNamed("init"))
                {
                    Rcpp::List init_values = hs_opts["init"];
                    if (init_values.containsElementNamed("epsilon"))
                    {
                        spatial.epsilon = Rcpp::as<Eigen::MatrixXd>(init_values["epsilon"]);
                    }
                    if (init_values.containsElementNamed("gamma"))
                    {
                        spatial.gamma = Rcpp::as<Eigen::MatrixXd>(init_values["gamma"]);
                    }
                    if (init_values.containsElementNamed("tau"))
                    {
                        spatial.tau = Rcpp::as<Eigen::VectorXd>(init_values["tau"]);
                    }
                    if (init_values.containsElementNamed("wdiag"))
                    {
                        spatial.wdiag = Rcpp::as<Eigen::VectorXd>(init_values["wdiag"]);
                    }

                    for (Eigen::Index k = 0; k < ns; k++)
                    {
                        for (Eigen::Index s = 0; s < ns; s++)
                        {
                            if (s != k)
                            {
                                spatial.theta(s, k) = spatial.epsilon(s, k) * spatial.gamma(s, k) * spatial.tau(k);
                            }
                            else
                            {
                                spatial.theta(s, k) = 0.0;
                            }
                        } // for s
                    } // for k

                    spatial.compute_alpha();
                } // if init

                if (ns == 1)
                {
                    hs_prior.infer = false; // No horseshoe if only one location
                    spatial.wdiag.setOnes();
                    spatial.alpha(0, 0) = 1.0;
                }
            } // if horseshoe


            if (mcmc_opts_list.containsElementNamed("zero_inflation"))
            {
                Rcpp::List zinfl_opts = mcmc_opts_list["zero_inflation"];
                zinfl_prior = Prior(zinfl_opts, "gaussian", 0.0, 3.0, true);
                zinfl = ZeroInflation(Y, zinfl_prior.infer);

                if (zinfl_opts.containsElementNamed("init"))
                {
                    Rcpp::List initial_values = zinfl_opts["init"];
                    if (initial_values.containsElementNamed("beta0"))
                    {
                        zinfl.beta0 = Rcpp::as<double>(initial_values["beta0"]);
                    }
                    if (initial_values.containsElementNamed("beta1"))
                    {
                        zinfl.beta1 = Rcpp::as<double>(initial_values["beta1"]);
                    }
                    if (initial_values.containsElementNamed("Z"))
                    {
                        Rcpp::NumericMatrix Z_mat = initial_values["Z"];
                        zinfl.Z = Rcpp::as<Eigen::MatrixXd>(Z_mat);
                    }
                } // if init

                if (zinfl_opts.containsElementNamed("hmc"))
                {
                    Rcpp::List hmc_zinfl_opts = zinfl_opts["hmc"];
                    zinfl_hmc_opts = HMCOpts_1d(hmc_zinfl_opts);
                } // if hmc
            } // if zero_inflation
        } // if mcmc_opts
        
        zinfl = ZeroInflation(Y, zinfl_prior.infer); // initialize zero-inflation model


        // Set up HMC options and diagnostics for static parameters if to be inferred
        Eigen::Index n_static_params = get_ndim_static_params(hmc_include_W);
        HMCDiagnostics_1d hmc_diag;
        DualAveraging_1d da_adapter;
        Eigen::VectorXd mass_diag_est = Eigen::VectorXd::Ones(n_static_params);
        MassAdapter mass_adapter;
        if (static_params_prior.infer)
        {
            hmc_opts.leapfrog_step_size = static_params_prior.hmc_step_size_init;
            hmc_opts.nleapfrog = static_params_prior.hmc_nleapfrog_init;
            hmc_diag = HMCDiagnostics_1d(static_cast<unsigned int>(ntotal), nburnin, true);
            da_adapter = DualAveraging_1d(hmc_opts);

            Eigen::VectorXd mass_init = get_constrained_static_params(hmc_include_W);
            mass_adapter.mean = mass_init.array().log().matrix();
            mass_adapter.M2 = Eigen::VectorXd::Zero(mass_adapter.mean.size());
        } // infer static params
        


        Eigen::VectorXi idx_colvec, idx_inverse;
        Eigen::Index n_colvec = get_ndim_colvec(idx_colvec, idx_inverse, 0);
        HMCDiagnostics_2d hs_hmc_diag;
        DualAveraging_2d hs_da_adapter;
        Eigen::MatrixXd mass_diag_hs = Eigen::MatrixXd::Ones(n_colvec, ns);
        MassAdapter_2d mass_adapter_hs;
        if (hs_prior.infer)
        {
            hs_hmc_opts.leapfrog_step_size_init = hs_prior.hmc_step_size_init;
            hs_hmc_opts.nleapfrog_init = hs_prior.hmc_nleapfrog_init;
            hs_hmc_opts.set_size(static_cast<unsigned int>(ns));
            hs_hmc_diag = HMCDiagnostics_2d(
                static_cast<unsigned int>(ns), 
                static_cast<unsigned int>(ntotal), 
                nburnin, 
                true
            );
            hs_da_adapter = DualAveraging_2d(
                static_cast<unsigned int>(ns), 
                hs_prior.hmc_step_size_init
            );

            mass_adapter_hs.mean = Eigen::MatrixXd::Zero(n_colvec, ns);
            mass_adapter_hs.M2 = Eigen::MatrixXd::Zero(n_colvec, ns);
            mass_adapter_hs.count = Eigen::VectorXd::Zero(ns);
            for (Eigen::Index k = 0; k < ns; k++)
            {
                mass_adapter_hs.mean.col(k) = get_unconstrained_colvec(
                    k, n_colvec, idx_colvec, idx_inverse
                );
            }
        } // infer horseshoe


        HMCDiagnostics_1d zinfl_hmc_diag;
        DualAveraging_1d zinfl_da_adapter;
        Eigen::VectorXd mass_diag_zinfl = Eigen::VectorXd::Ones(2);
        MassAdapter mass_adapter_zinfl;
        if (zinfl_prior.infer)
        {
            zinfl_hmc_opts.leapfrog_step_size = zinfl_prior.hmc_step_size_init;
            zinfl_hmc_opts.nleapfrog = zinfl_prior.hmc_nleapfrog_init;
            zinfl_hmc_diag = HMCDiagnostics_1d(static_cast<unsigned int>(ntotal), nburnin, true);
            zinfl_da_adapter = DualAveraging_1d(zinfl_hmc_opts);

            Eigen::VectorXd mass_init(2);
            mass_init(0) = zinfl.beta0;
            mass_init(1) = zinfl.beta1;
            mass_adapter_zinfl.mean = mass_init;
            mass_adapter_zinfl.M2 = Eigen::VectorXd::Zero(mass_adapter_zinfl.mean.size());
        } // HMC objects infer zero-inflation


        Eigen::Index npass = hs_prior.infer ? 5 : 1;
        Eigen::ArrayXd wt_accept_count(ns);
        wt_accept_count.setZero();
        Eigen::ArrayXd wt_mh_sd = Eigen::ArrayXd::Constant(ns, wt_prior.mh_sd);
        Eigen::Tensor<double, 3> wt_samples;
        if (wt_prior.infer)
        {
            wt_samples.resize(temporal.wt.rows(), temporal.wt.cols(), nsamples);
        }

        Eigen::VectorXd mu_samples, W_samples, rho_dist_samples, rho_mobility_samples;
        if (static_params_prior.infer)
        {
            mu_samples.resize(nsamples);
            if (hmc_include_W)
            {
                W_samples.resize(nsamples);
            }
            if (spatial.include_distance)
            {
                rho_dist_samples.resize(nsamples);
            }
            if (spatial.include_log_mobility)
            {
                rho_mobility_samples.resize(nsamples);
            }
        }

        Eigen::Tensor<double, 3> theta_samples, gamma_samples;
        Eigen::MatrixXd tau_samples, wdiag_samples;
        if (hs_prior.infer)
        {
            theta_samples.resize(spatial.theta.rows(), spatial.theta.cols(), nsamples);
            gamma_samples.resize(spatial.gamma.rows(), spatial.gamma.cols(), nsamples);
            tau_samples.resize(spatial.tau.size(), nsamples);
            wdiag_samples.resize(spatial.wdiag.size(), nsamples);
        }

        Eigen::MatrixXd zinfl_Z_average;
        Eigen::VectorXd zinfl_beta0_samples, zinfl_beta1_samples;
        if (zinfl_prior.infer)
        {
            zinfl_Z_average = Eigen::MatrixXd::Zero(zinfl.Z.rows(), zinfl.Z.cols());
            zinfl_beta0_samples.resize(nsamples);
            zinfl_beta1_samples.resize(nsamples);
        }

        Eigen::Tensor<double, 3> alpha_samples(spatial.alpha.rows(), spatial.alpha.cols(), nsamples);
        Eigen::Tensor<double, 3> N_samples; // (nt + 1) x ns x nsamples, total secondary infections generated by (t, s)
        Eigen::Tensor<double, 3> N0_samples; // (nt + 1) x ns x nsamples, baseline primary infections at (t, s)
        if (sample_augmented_N)
        {
            N_samples.resize(nt + 1, ns, nsamples);
            N0_samples.resize(nt + 1, ns, nsamples);
        }


        Eigen::VectorXd post_burnin_accept_sum = Eigen::VectorXd::Zero(ns);
        Eigen::VectorXi post_burnin_accept_count = Eigen::VectorXi::Zero(ns);
        const int post_burnin_window = 50; // Check every 50 iterations
        const double min_step_size = 1e-4;
        const double max_step_size = 1.0;

        Progress p(static_cast<unsigned int>(ntotal), true);
        for (Eigen::Index iter = 0; iter < ntotal; iter++)
        {
            // Update unobserved secondary infections N and baseline primary infections N0
            if (sample_augmented_N)
            {
                update_N(Y);
            }


            if (zinfl_prior.infer)
            {
                Eigen::MatrixXd R_mat = temporal.compute_Rt(); // (nt + 1) x ns
                Eigen::MatrixXd lambda_mat = compute_intensity(Y, R_mat, spatial.alpha, dlag.Fphi, mu);
                zinfl.update_Z(Y, lambda_mat);


                double energy_diff = 0.0;
                double grad_norm = 0.0;
                double accept_prob = zinfl.update_beta(
                    energy_diff, grad_norm,
                    zinfl_hmc_opts.leapfrog_step_size,
                    zinfl_hmc_opts.nleapfrog,
                    mass_diag_zinfl,
                    zinfl_prior.par1, zinfl_prior.par2
                );

                // (Optional) Update diagnostics and dual averaging for rho
                zinfl_hmc_diag.accept_count += accept_prob;
                if (zinfl_hmc_opts.diagnostics)
                {
                    zinfl_hmc_diag.energy_diff(iter) = energy_diff;
                    zinfl_hmc_diag.grad_norm(iter) = grad_norm;
                }

                if (zinfl_hmc_opts.dual_averaging && iter <= nburnin)
                {
                    if (iter < nburnin)
                    {
                        zinfl_hmc_opts.leapfrog_step_size = zinfl_da_adapter.update_step_size(accept_prob);
                    }
                    else if (iter == nburnin)
                    {
                        zinfl_da_adapter.finalize_leapfrog_step(
                            zinfl_hmc_opts.leapfrog_step_size,
                            zinfl_hmc_opts.nleapfrog,
                            zinfl_hmc_opts.T_target);
                    }

                    if (zinfl_hmc_opts.diagnostics)
                    {
                        zinfl_hmc_diag.leapfrog_step_size_stored(iter) = zinfl_hmc_opts.leapfrog_step_size;
                        zinfl_hmc_diag.nleapfrog_stored(iter) = zinfl_hmc_opts.nleapfrog;
                    }
                } // if dual averaging


                if (iter < nburnin)
                {
                    // Phase 1 (iter < nburnin/2): Adapt step size with unit mass
                    // Phase 2 (nburnin/2 <= iter < nburnin): Adapt mass matrix
                    Eigen::Vector2d current_params;
                    current_params(0) = zinfl.beta0;
                    current_params(1) = zinfl.beta1;
                    mass_adapter_zinfl.update(current_params);

                    // Only update mass matrix ONCE at the midpoint
                    if (iter == nburnin / 2)
                    {
                        mass_diag_est = mass_adapter_zinfl.get_mass_diag();

                        // CRITICAL: Reset dual averaging for new geometry
                        zinfl_da_adapter = DualAveraging_1d(zinfl_hmc_opts);
                    }
                } // mass matrix adaptation
            } // if infer zero-inflation


            for (Eigen::Index k = 0; k < ns; k++)
            {
                // Update horseshoe parameters
                if (hs_prior.infer)
                {
                    Eigen::MatrixXd R_mat = temporal.compute_Rt(); // (nt + 1) x ns

                    double energy_diff = 0.0;
                    double grad_norm = 0.0;
                    Eigen::VectorXd mass_diag_vec = mass_diag_hs.col(k);
                    double accept_prob = update_colvec(
                        energy_diff, grad_norm, k, Y,
                        R_mat, mass_diag_vec,
                        hs_hmc_opts.leapfrog_step_size(k),
                        hs_hmc_opts.nleapfrog(k),
                        spatial.include_distance,
                        spatial.include_log_mobility
                    );

                    // (Optional) Update diagnostics and dual averaging for horseshoe theta
                    hs_hmc_diag.accept_count(k) += accept_prob;
                    if (hs_hmc_opts.diagnostics)
                    {
                        hs_hmc_diag.energy_diff(k, iter) = energy_diff;
                        hs_hmc_diag.grad_norm(k, iter) = grad_norm;
                    }
                    if (hs_hmc_opts.dual_averaging && iter <= nburnin)
                    {
                        if (iter < nburnin)
                        {
                            hs_hmc_opts.leapfrog_step_size(k) = hs_da_adapter.update_step_size(k, accept_prob);
                        }
                        else if (iter == nburnin)
                        {
                            double step_size = hs_hmc_opts.leapfrog_step_size(k);
                            hs_da_adapter.finalize_leapfrog_step(
                                step_size,
                                hs_hmc_opts.nleapfrog(k),
                                k, hs_hmc_opts.T_target
                            );
                            hs_hmc_opts.leapfrog_step_size(k) = step_size;
                        }

                        if (hs_hmc_opts.diagnostics)
                        {
                            hs_hmc_diag.leapfrog_step_size_stored(k, iter) = hs_hmc_opts.leapfrog_step_size(k);
                            hs_hmc_diag.nleapfrog_stored(k, iter) = hs_hmc_opts.nleapfrog(k);
                        }
                    } // if dual averaging during burnin

                    if (iter > nburnin)
                    {
                        // Accumulate acceptance probability
                        post_burnin_accept_sum(k) += accept_prob;
                        post_burnin_accept_count(k)++;

                        // Every 'window' iterations, check and adapt
                        if (post_burnin_accept_count(k) >= post_burnin_window)
                        {
                            double recent_accept_rate = post_burnin_accept_sum(k) / post_burnin_accept_count(k);

                            // Target acceptance rate for HMC is ~0.65-0.80
                            // Use gentler bounds to avoid over-adaptation
                            if (recent_accept_rate < 0.4)
                            {
                                // Too many rejections - reduce step size
                                hs_hmc_opts.leapfrog_step_size(k) = std::max(
                                    hs_hmc_opts.leapfrog_step_size(k) * 0.9,
                                    min_step_size);
                                if (hs_hmc_opts.diagnostics)
                                {
                                    Rcpp::Rcout << "Iter " << iter << " col " << k
                                                << ": accept=" << recent_accept_rate
                                                << ", reducing step to " << hs_hmc_opts.leapfrog_step_size(k) << std::endl;
                                }
                            }
                            else if (recent_accept_rate > 0.9)
                            {
                                // Too conservative - can increase step size slightly
                                hs_hmc_opts.leapfrog_step_size(k) = std::min(
                                    hs_hmc_opts.leapfrog_step_size(k) * 1.05,
                                    max_step_size);

                                if (hs_hmc_opts.diagnostics)
                                {
                                    Rcpp::Rcout << "Iter " << iter << " col " << k
                                                << ": accept=" << recent_accept_rate
                                                << ", increasing step to " << hs_hmc_opts.leapfrog_step_size(k) << std::endl;
                                }
                            }

                            // Reset counters for next window
                            post_burnin_accept_sum(k) = 0.0;
                            post_burnin_accept_count(k) = 0;
                        } // if check adaptation
                    } // post-burnin step size adaptation when iter > nburnin
                } // if infer horseshoe

                // Update temporal transmission components
                if (wt_prior.infer)
                {
                    // block update of wt
                    if (iter < nburnin && hs_prior.infer)
                    {
                        double accept_sum = 0.0;
                        for (Eigen::Index pass = 0; pass < nthin; pass++)
                        {
                            double accept = update_eta_block_mh(Y, k, wt_mh_sd(k));
                            accept_sum += accept;
                        }
                        wt_accept_count(k) += accept_sum / static_cast<double>(nthin);
                    }
                    else
                    {
                        double accept = update_eta_block_mh(Y, k, wt_mh_sd(k));
                        wt_accept_count(k) += accept;
                    }

                    if (iter < nburnin && iter > 0 && iter % 50 == 0)
                    {
                        double accept_rate = wt_accept_count(k) / 50.0;
                        wt_accept_count(k) = 0.0;

                        // Robbins-Monro update
                        double gamma = 1.0 / std::pow(iter / 50.0, 0.6); // Decay rate
                        wt_mh_sd(k) *= std::exp(gamma * (accept_rate - 0.6));

                        wt_mh_sd(k) = std::max(0.01, std::min(2.0, wt_mh_sd(k)));
                    }

                } // if infer wt
            } // Loop: update horseshoe and wt for each location k

            
            if (hs_prior.infer)
            {
                if (iter < nburnin)
                {
                    for (Eigen::Index k = 0; k < ns; k++)
                    {
                        // Phase 1 (iter < nburnin/2): Adapt step size with unit mass
                        // Phase 2 (nburnin/2 <= iter < nburnin): Adapt mass matrix
                        Eigen::VectorXi indices_start, indices_inverse;
                        Eigen::Index n_colvec = get_ndim_colvec(indices_start, indices_inverse, k);
                        Eigen::VectorXd current_unconstrained = get_unconstrained_colvec(
                            k, n_colvec, indices_start, indices_inverse
                        );
                        mass_adapter_hs.update(k, current_unconstrained);

                        // Only update mass matrix ONCE at the midpoint
                        if (iter == nburnin / 2)
                        {
                            // double step_size = hs_hmc_opts.leapfrog_step_size(k);
                            // hs_da_adapter.finalize_leapfrog_step(
                            //     step_size,
                            //     hs_hmc_opts.nleapfrog(k),
                            //     k, hs_hmc_opts.T_target);
                            // hs_hmc_opts.leapfrog_step_size(k) = step_size;
                            mass_diag_hs.col(k) = mass_adapter_hs.get_mass_diag(k);
                        }
                    }

                    if (iter == nburnin / 2)
                    {
                        // CRITICAL: Reset dual averaging for new geometry
                        hs_da_adapter = DualAveraging_2d(hs_hmc_opts.leapfrog_step_size);
                    }
                } // mass matrix adaptation during burnin
            } // mass matrix adaptation if infer horseshoe theta


            // Update static parameters ( mu, [W])
            if (static_params_prior.infer)
            {
                double energy_diff = 0.0;
                double grad_norm = 0.0;
                double accept_prob = update_params_collapsed(
                    energy_diff, grad_norm, Y,
                    hmc_opts.leapfrog_step_size,
                    hmc_opts.nleapfrog, mass_diag_est,
                    static_params_prior.par1, static_params_prior.par2,
                    hmc_include_W
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

                    Eigen::VectorXd current_log_params(hmc_include_W ? 2 : 1);
                    current_log_params(0) = std::log(mu);
                    if (hmc_include_W)
                        current_log_params(1) = std::log(temporal.W);

                    mass_adapter.update(current_log_params);

                    // Only update mass matrix ONCE at the midpoint
                    if (iter == nburnin / 2)
                    {
                        mass_diag_est = mass_adapter.get_mass_diag();

                        // CRITICAL: Reset dual averaging for new geometry
                        da_adapter = DualAveraging_1d(hmc_opts);
                    }
                } // mass matrix adaptation
            } // if infer static params


            // Store samples after burn-in and thinning
            if (iter >= nburnin && ((iter - nburnin) % nthin == 0))
            {
                Eigen::Index sample_idx = (iter - nburnin) / nthin;

                if (zinfl_prior.infer)
                {
                    zinfl_Z_average += zinfl.Z;
                    zinfl_beta0_samples(sample_idx) = zinfl.beta0;
                    zinfl_beta1_samples(sample_idx) = zinfl.beta1;
                }

                if (static_params_prior.infer)
                {
                    mu_samples(sample_idx) = mu;
                    if (hmc_include_W)
                    {
                        W_samples(sample_idx) = temporal.W;
                    }
                    if (spatial.include_distance)
                    {
                        rho_dist_samples(sample_idx) = spatial.rho_dist;
                    }
                    if (spatial.include_log_mobility)
                    {
                        rho_mobility_samples(sample_idx) = spatial.rho_mobility;
                    }
                }

                if (wt_prior.infer)
                {
                    // Store wt samples
                    // wt_samples.chip(sample_idx, 2) = temporal.wt;
                    Eigen::TensorMap<Eigen::Tensor<const double, 2>> wt_map(temporal.wt.data(), temporal.wt.rows(), temporal.wt.cols());
                    wt_samples.chip(sample_idx, 2) = wt_map;
                }

                if (hs_prior.infer)
                {
                    // Store horseshoe samples
                    Eigen::MatrixXd theta_centered = spatial.center_theta();
                    Eigen::TensorMap<Eigen::Tensor<const double, 2>> theta_map(theta_centered.data(), theta_centered.rows(), theta_centered.cols());
                    Eigen::TensorMap<Eigen::Tensor<const double, 2>> gamma_map(spatial.gamma.data(), spatial.gamma.rows(), spatial.gamma.cols());
                    Eigen::Map<const Eigen::VectorXd> tau_map(spatial.tau.data(), spatial.tau.size());
                    Eigen::Map<const Eigen::VectorXd> wdiag_map(spatial.wdiag.data(), spatial.wdiag.size());

                    theta_samples.chip(sample_idx, 2) = theta_map;
                    gamma_samples.chip(sample_idx, 2) = gamma_map;
                    tau_samples.col(sample_idx) = tau_map;
                    wdiag_samples.col(sample_idx) = wdiag_map;
                }

                Eigen::TensorMap<Eigen::Tensor<const double, 2>> alpha_map(spatial.alpha.data(), spatial.alpha.rows(), spatial.alpha.cols());
                alpha_samples.chip(sample_idx, 2) = alpha_map;

                if (sample_augmented_N)
                {
                    for (Eigen::Index t = 0; t < nt + 1; t++)
                    {
                        for (Eigen::Index s = 0; s < ns; s++)
                        {
                            N_samples(t, s, sample_idx) = temporal.compute_N_future_sum(N, s, t);
                        }
                    }
                    // N_samples.chip(sample_idx, 4) = N;
                    Eigen::TensorMap<Eigen::Tensor<const double, 2>> N0_map(N0.data(), N0.rows(), N0.cols());
                    N0_samples.chip(sample_idx, 2) = N0_map;
                }
            } // if store samples

            p.increment();
        } // for MCMC iter

        
        Rcpp::List output;

        output["alpha"] = tensor3_to_r(alpha_samples); // ns x ns x nsamples

        if (zinfl_prior.infer)
        {
            Rcpp::List zinfl_list = Rcpp::List::create(
                Rcpp::Named("Z_average") = zinfl_Z_average / static_cast<double>(nsamples), // (nt + 1) x ns
                Rcpp::Named("beta0") = zinfl_beta0_samples, // nsamples x 1
                Rcpp::Named("beta1") = zinfl_beta1_samples  // nsamples x 1
            );

            zinfl_list["hmc"] = Rcpp::List::create(
                Rcpp::Named("acceptance_rate") = zinfl_hmc_diag.accept_count / static_cast<double>(ntotal),
                Rcpp::Named("leapfrog_step_size") = zinfl_hmc_opts.leapfrog_step_size,
                Rcpp::Named("n_leapfrog") = zinfl_hmc_opts.nleapfrog,
                Rcpp::Named("diagnostics") = zinfl_hmc_diag.to_list()
            );

            output["zero_inflation"] = zinfl_list;
        } // if infer zero-inflation

        if (wt_prior.infer)
        {
            output["wt"] = tensor3_to_r(wt_samples); // (nt + 1) x ns x nsamples
            output["wt_mh_sd"] = wt_mh_sd; // ns x 1
            output["wt_accept_prob"] = wt_accept_count / static_cast<double>(ntotal - nburnin); // (nt + 1) x ns
        } // if infer wt

        if (static_params_prior.infer)
        {
            Rcpp::List param_list = Rcpp::List::create(
                Rcpp::Named("mu") = mu_samples // nsamples x 1
            );
            if (hmc_include_W)
            {
                param_list["W"] = W_samples; // nsamples x 1
            }
            if (spatial.include_distance)
            {
                param_list["rho_dist"] = rho_dist_samples; // nsamples x 1
            }
            if (spatial.include_log_mobility)
            {
                param_list["rho_mobility"] = rho_mobility_samples; // nsamples x 1
            }
            param_list["hmc"] = Rcpp::List::create(
            Rcpp::Named("acceptance_rate") = hmc_diag.accept_count / static_cast<double>(ntotal),
            Rcpp::Named("leapfrog_step_size") = hmc_opts.leapfrog_step_size,
            Rcpp::Named("n_leapfrog") = hmc_opts.nleapfrog,
            Rcpp::Named("diagnostics") = hmc_diag.to_list());

            output["params"] = param_list;
        } // if infer static params

        if (hs_prior.infer)
        {
            Rcpp::List hs_list = Rcpp::List::create(
                Rcpp::Named("theta") = tensor3_to_r(theta_samples), // ns x ns x nsamples
                Rcpp::Named("gamma") = tensor3_to_r(gamma_samples), // ns x ns x nsamples
                Rcpp::Named("tau") = tau_samples, // ns x nsamples
                Rcpp::Named("wdiag") = wdiag_samples  // ns x nsamples
            );

            hs_list["hmc"] = Rcpp::List::create(
                Rcpp::Named("acceptance_rate") = hs_hmc_diag.accept_count / static_cast<double>(ntotal),
                Rcpp::Named("leapfrog_step_size") = hs_hmc_opts.leapfrog_step_size,
                Rcpp::Named("n_leapfrog") = hs_hmc_opts.nleapfrog,
                Rcpp::Named("mass_diag") = mass_diag_hs,
                Rcpp::Named("diagnostics") = hs_hmc_diag.to_list()
            );

            output["horseshoe"] = hs_list;
        } // if infer horseshoe theta

        if (sample_augmented_N)
        {
            output["N"] = tensor3_to_r(N_samples); // (nt + 1) x ns x nsamples
            output["N0"] = tensor3_to_r(N0_samples); // (nt + 1) x ns x nsamples
        }

        return output;
    } // run_mcmc

}; // class Model



#endif // MODEL_HPP