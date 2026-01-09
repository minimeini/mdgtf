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
    Eigen::Index ns = 1; // number of spatial locations
    Eigen::Index nt = 0; // number of time points


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
    Eigen::Index np = 0; // number of covariates for retention weights

    bool include_distance = false;
    Eigen::MatrixXd dist; // ns x ns, pairwise distances between locations
    double rho_dist = 0.0; // distance decay parameter


    bool include_log_mobility = false;    
    Eigen::MatrixXd log_mobility; // ns x ns, pairwise log mobility
    double rho_mobility = 0.0; // mobility scaling parameter


    bool include_covariates = false;
    Eigen::Tensor<double, 3> X; // np x ns x (nt + 1), covariate array for the logit of retention weights
    Eigen::VectorXd logit_wdiag_intercept; // ns x 1, intercept of logit retention weights per location
    Eigen::VectorXd logit_wdiag_slope; // np x 1, slope of logit retention weights shared by all locations


    Eigen::MatrixXd theta; // ns x ns, horseshoe component for spatial network
    Eigen::MatrixXd gamma; // ns x ns, horseshoe element-wise variance
    Eigen::MatrixXd delta; // ns x ns, horseshoe local shrinkage parameters
    Eigen::VectorXd tau; // ns x 1, horseshoe column-wise variance
    Eigen::VectorXd zeta; // ns x 1, horseshoe column-wise local shrinkage parameters

    // Regularized horseshoe slab parameters
    double c_sq = 4.0; // slab variance (c²), controls maximum shrinkage


    SpatialNetwork()
    {
        ns = 1;
        nt = 0;
        np = 0;
        c_sq = 4.0;

        include_distance = false;
        include_log_mobility = false;
        include_covariates = false;

        initialize_horseshoe_zero();
        return;
    } // SpatialNetwork default constructor


    SpatialNetwork(
        const unsigned int &ns_in,
        const bool &random_init = false,
        const double &c_sq_in = 4.0,
        const Rcpp::Nullable<Rcpp::NumericMatrix> &dist_scaled_in = R_NilValue,
        const Rcpp::Nullable<Rcpp::NumericMatrix> &mobility_scaled_in = R_NilValue,
        const Rcpp::Nullable<Rcpp::NumericVector> &X_in = R_NilValue,
        const Rcpp::Nullable<Rcpp::NumericVector> &mean_slopes_in = R_NilValue,
        const Rcpp::Nullable<Rcpp::NumericVector> &sd_slopes_in = R_NilValue
    )
    {
        c_sq = c_sq_in;
        ns = ns_in;

        include_covariates = false;
        np = 0;

        initialize_horseshoe_zero();

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

        if (X_in.isNotNull())
        {
            Rcpp::NumericVector X_vec(X_in);
            X = r_to_tensor3(X_vec);
            if (X.dimension(1) != ns)
            {
                Rcpp::stop("Dimension mismatch when initializing covariate array X.");
            }

            include_covariates = true;
            np = X.dimension(0);
            nt = X.dimension(2) - 1;

            // Initialize logit retention weight parameters
            logit_wdiag_intercept.resize(ns);
            logit_wdiag_slope.resize(np);
            logit_wdiag_intercept.setConstant(5.0); // high retention (~=1)
            logit_wdiag_slope.setZero();
        }
        else
        {
            np = 0;
            include_covariates = false;
        } // if add covariates to the retention rate or not


        Eigen::VectorXd mean_slopes, sd_slopes;
        if (mean_slopes_in.isNotNull())
        {
            Rcpp::NumericVector mean_slopes_vec(mean_slopes_in);
            if (mean_slopes_vec.size() != np)
            {
                Rcpp::stop("Dimension mismatch when initializing mean_slopes.");
            }
            mean_slopes = Rcpp::as<Eigen::VectorXd>(mean_slopes_vec);
        }
        else
        {
            mean_slopes.resize(np);
            for (Eigen::Index p = 0; p < np; p++)
            {
                mean_slopes(p) = R::runif(0.2, 0.8);
            }
        }

        if (sd_slopes_in.isNotNull())
        {
            Rcpp::NumericVector sd_slopes_vec(sd_slopes_in);
            if (sd_slopes_vec.size() != np)
            {
                Rcpp::stop("Dimension mismatch when initializing sd_slopes.");
            }
            sd_slopes = Rcpp::as<Eigen::VectorXd>(sd_slopes_vec);
        }
        else
        {
            sd_slopes.resize(np);
            sd_slopes.setConstant(0.3);
        }


        if (random_init)
        {
            rho_dist = include_distance ? R::runif(0.0, 2.0) : 0.0;
            rho_mobility = include_log_mobility ? R::runif(0.0, 2.0) : 0.0;
            initialize_logit_wdiag_slopes(mean_slopes, sd_slopes);
            initialize_horseshoe_dominant();
        }
        else
        {
            rho_dist = 0.0;
            rho_mobility = 0.0;
            // initialize_horseshoe_zero();
            initialize_logit_wdiag_slopes(mean_slopes, sd_slopes);
            initialize_horseshoe_sparse();
        }
        return;
    } // SpatialNetwork constructor for MCMC inference when model parameters are unknown


    SpatialNetwork(
        const unsigned int &ns_in, 
        const Rcpp::List &settings,
        const Rcpp::Nullable<Rcpp::NumericMatrix> &dist_scaled_in = R_NilValue,
        const Rcpp::Nullable<Rcpp::NumericMatrix> &mobility_scaled_in = R_NilValue,
        const Rcpp::Nullable<Rcpp::NumericVector> &X_in = R_NilValue,
        const Rcpp::Nullable<Rcpp::NumericVector> &mean_slopes_in = R_NilValue,
        const Rcpp::Nullable<Rcpp::NumericVector> &sd_slopes_in = R_NilValue
    )
    {
        ns = ns_in;

        // Regularized horseshoe slab parameters
        c_sq = 4.0;
        if (settings.containsElementNamed("c_sq"))
        {
            c_sq = Rcpp::as<double>(settings["c_sq"]);
        } // if slab variance specified

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


        if (X_in.isNotNull())
        {
            Rcpp::NumericVector X_vec(X_in);
            X = r_to_tensor3(X_vec);
            if (X.dimension(1) != ns)
            {
                Rcpp::stop("Dimension mismatch when initializing covariate array X.");
            }

            include_covariates = true;
            np = X.dimension(0);
            nt = X.dimension(2) - 1;

            /*
            Initialize logit retention weight parameters

            - The true values of intercepts and slopes are provided in settings;
            - Otherwise, sample from normal distributions centered at mean_slopes and sd_slopes,
            which can also be specified separately as input arguments.
            */
            logit_wdiag_slope.resize(np);
            if (settings.containsElementNamed("logit_wdiag_slope"))
            {
                Rcpp::NumericVector slope_vec = Rcpp::as<Rcpp::NumericVector>(settings["logit_wdiag_slope"]);
                if (slope_vec.size() != np)
                {
                    Rcpp::stop("Dimension mismatch when initializing logit_wdiag_slope.");
                }
                logit_wdiag_slope = Rcpp::as<Eigen::VectorXd>(slope_vec);
            }
            else
            {
                Eigen::VectorXd mean_slopes, sd_slopes;
                if (mean_slopes_in.isNotNull())
                {
                    Rcpp::NumericVector mean_slopes_vec(mean_slopes_in);
                    if (mean_slopes_vec.size() != np)
                    {
                        Rcpp::stop("Dimension mismatch when initializing mean_slopes.");
                    }
                    mean_slopes = Rcpp::as<Eigen::VectorXd>(mean_slopes_vec);
                }
                else
                {
                    mean_slopes.resize(np);
                    for (Eigen::Index p = 0; p < np; p++)
                    {
                        mean_slopes(p) = R::runif(0.2, 0.8);
                    }
                }

                if (sd_slopes_in.isNotNull())
                {
                    Rcpp::NumericVector sd_slopes_vec(sd_slopes_in);
                    if (sd_slopes_vec.size() != np)
                    {
                        Rcpp::stop("Dimension mismatch when initializing sd_slopes.");
                    }
                    sd_slopes = Rcpp::as<Eigen::VectorXd>(sd_slopes_vec);
                }
                else
                {
                    sd_slopes.resize(np);
                    sd_slopes.setConstant(0.3);
                }

                initialize_logit_wdiag_slopes(mean_slopes, sd_slopes);
            }
        }
        else
        {
            np = 0;
            nt = 0;
            include_covariates = false;
        } // if covariates X is provided or not


        logit_wdiag_intercept.resize(ns);
        if (settings.containsElementNamed("logit_wdiag_intercept"))
        {
            Rcpp::NumericVector intercept_vec = Rcpp::as<Rcpp::NumericVector>(settings["logit_wdiag_intercept"]);
            if (intercept_vec.size() != ns)
            {
                Rcpp::stop("Dimension mismatch when initializing logit_wdiag_intercept.");
            }

            initialize_horseshoe_zero();
            logit_wdiag_intercept = Rcpp::as<Eigen::VectorXd>(intercept_vec);
        }
        else
        {
            initialize_horseshoe_dominant();
        }

        return;
    } // SpatialNetwork constructor for simulation when model parameters are known but Y is to be simulated


    Rcpp::List to_list() const
    {
        return Rcpp::List::create(
            Rcpp::Named("theta") = theta,
            Rcpp::Named("gamma") = gamma,
            Rcpp::Named("delta") = delta,
            Rcpp::Named("tau") = tau,
            Rcpp::Named("zeta") = zeta,
            Rcpp::Named("logit_wdiag_intercept") = logit_wdiag_intercept,
            Rcpp::Named("logit_wdiag_slope") = logit_wdiag_slope,
            Rcpp::Named("rho_dist") = rho_dist,
            Rcpp::Named("rho_mobility") = rho_mobility,
            Rcpp::Named("dist") = dist,
            Rcpp::Named("log_mobility") = log_mobility,
            Rcpp::Named("c_sq") = c_sq
        );
    } // to_list


    void set_X(const Rcpp::NumericVector &X_in)
    {
        Rcpp::NumericVector X_vec(X_in);
        X = r_to_tensor3(X_vec);
        if (X.dimension(1) != ns)
        {
            Rcpp::stop("Dimension mismatch when setting covariate array X.");
        }

        include_covariates = true;
        np = X.dimension(0);
        nt = X.dimension(2) - 1;
    } // set_X


    /**
     * @brief Compute min/max covariate effects over all times for location k
     */
    std::pair<double, double> compute_covariate_effect_range(
        const Eigen::Tensor<double, 3> &X,
        const Eigen::VectorXd &slopes,
        Eigen::Index k)
    {
        if (X.size() == 0 || slopes.size() == 0)
        {
            return {0.0, 0.0};
        }

        Eigen::Index np = X.dimension(0);
        Eigen::Index T = X.dimension(2);

        double min_effect = std::numeric_limits<double>::infinity();
        double max_effect = -std::numeric_limits<double>::infinity();

        for (Eigen::Index t = 0; t < T; t++)
        {
            double effect = 0.0;
            for (Eigen::Index p = 0; p < np; p++)
            {
                effect += slopes(p) * X(p, k, t);
            }
            min_effect = std::min(min_effect, effect);
            max_effect = std::max(max_effect, effect);
        }

        if (min_effect == std::numeric_limits<double>::infinity())
        {
            return {0.0, 0.0};
        }

        return {min_effect, max_effect};
    } // compute_covariate_effect_range


    /**
     * @brief Compute minimum w for diagonal dominance given off-diagonal weights
     * 
     * @param u_col 
     * @return double 
     */
    double compute_w_min_for_dominance(const Eigen::VectorXd &u_col)
    {
        double u_max = u_col.maxCoeff();
        double U_off = u_col.sum();

        if (U_off < EPS)
        {
            return 0.0;
        }

        return u_max / (U_off + u_max);
    } // compute_w_min_for_dominance


    void initialize_logit_wdiag_slopes(
        const Eigen::VectorXd &mean_slopes,
        const Eigen::VectorXd &sd_slopes
    )
    {
        if (include_covariates && np > 0)
        {
            logit_wdiag_slope.resize(np);
            for (Eigen::Index p = 0; p < np; p++)
            {
                logit_wdiag_slope(p) = R::rnorm(mean_slopes(p), sd_slopes(p));
            }
        }
    } // initialize_logit_wdiag_slopes


    /**
     * @brief Initialize horseshoe with RANDOM sampling and covariate effects
     *
     * Samples all parameters from their priors, then adjusts intercepts
     * to ensure diagonal dominance at the worst-case covariate configuration.
     */
    void initialize_horseshoe_random(
        const double &dominance_margin = 0.1,
        const double &baseline_headroom = 0.5
    )
    {
        // Step 1: Run `initialize_logit_wdiag_slopes` if covariates are included or specify `logit_wdiag_slope` manually

        // Resize parameters
        theta.resize(ns, ns);
        gamma.resize(ns, ns);
        delta.resize(ns, ns);
        tau.resize(ns);
        zeta.resize(ns);
        logit_wdiag_intercept.resize(ns);


        // Step 2: Sample horseshoe parameters for each column
        for (Eigen::Index k = 0; k < ns; k++)
        {
            // Global shrinkage (half-Cauchy via inverse gamma)
            zeta(k) = 1.0 / R::rgamma(0.5, 1.0);
            tau(k) = 1.0 / R::rgamma(0.5, zeta(k));

            // Local shrinkage and theta
            for (Eigen::Index s = 0; s < ns; s++)
            {
                delta(s, k) = 1.0 / R::rgamma(0.5, 1.0);
                gamma(s, k) = 1.0 / R::rgamma(0.5, delta(s, k));

                if (s == k)
                {
                    theta(s, k) = 0.0;
                }
                else
                {
                    double reg_var = compute_regularized_variance(s, k);
                    theta(s, k) = R::rnorm(0.0, std::sqrt(reg_var));
                }
            }

            // Compute unnormalized weights
            Eigen::VectorXd u_k = compute_unnormalized_weight_col(k);

            // Minimum w for diagonal dominance
            double w_min = compute_w_min_for_dominance(u_k);

            // Step 3: Find worst-case covariate effect
            auto [min_cov_effect, max_cov_effect] = compute_covariate_effect_range(
                X, logit_wdiag_slope, k
            );

            // Step 4: Set intercept ensuring dominance at worst case
            double target_w = std::min(w_min + dominance_margin, 1.0 - EPS);
            double required_intercept = logit_safe(target_w) - min_cov_effect;

            // Add random headroom
            logit_wdiag_intercept(k) = required_intercept + R::runif(0.0, baseline_headroom);
        } // for each source k
    } // initialize_horseshoe_random


    /**
     * @brief Initialize horseshoe with DOMINANT parameterization and covariate effects
     *
     * Parameterizes initialization by dominance ratio r = α(k,k) / max_{s≠k} α(s,k).
     * Useful for simulation studies where you want to control local dominance.
     */
    void initialize_horseshoe_dominant(
        const double &log_dominance_mean = 1.0,
        const double &log_dominance_sd = 0.5
    )
    {
        // Step 1: Run `initialize_logit_wdiag_slopes` if covariates are included or specify `logit_wdiag_slope` manually

        // Resize parameters
        theta.resize(ns, ns);
        gamma.resize(ns, ns);
        delta.resize(ns, ns);
        tau.resize(ns);
        zeta.resize(ns);
        logit_wdiag_intercept.resize(ns);

        // Step 2: Sample horseshoe parameters
        for (Eigen::Index k = 0; k < ns; k++)
        {
            zeta(k) = 1.0 / R::rgamma(0.5, 1.0);
            tau(k) = 1.0 / R::rgamma(0.5, zeta(k));

            for (Eigen::Index s = 0; s < ns; s++)
            {
                delta(s, k) = 1.0 / R::rgamma(0.5, 1.0);
                gamma(s, k) = 1.0 / R::rgamma(0.5, delta(s, k));

                if (s == k)
                {
                    theta(s, k) = 0.0;
                }
                else
                {
                    double reg_var = compute_regularized_variance(s, k);
                    theta(s, k) = R::rnorm(0.0, std::sqrt(reg_var));
                }
            }

            // Compute u_k
            Eigen::VectorXd u_k = compute_unnormalized_weight_col(k);
            double U_off = u_k.sum();
            double u_max = u_k.maxCoeff();

            // Sample dominance ratio r > 1
            double log_r_minus_1 = R::rnorm(log_dominance_mean, log_dominance_sd);
            double r = 1.0 + std::exp(log_r_minus_1);

            // Solve for w_baseline: r = w / ((1-w) * u_max / U_off)
            double w_baseline;
            if (U_off > EPS)
            {
                double ratio = r * u_max / U_off;
                w_baseline = ratio / (1.0 + ratio);
            }
            else
            {
                w_baseline = 0.9;
            }
            w_baseline = std::max(EPS, std::min(1.0 - EPS, w_baseline));

            // Find worst-case covariate effect
            auto [min_cov_effect, max_cov_effect] = compute_covariate_effect_range(
                X, logit_wdiag_slope, k);

            double w_min = compute_w_min_for_dominance(u_k);

            // Intercept for target baseline, ensuring dominance
            double logit_target = logit_safe(w_baseline);
            double required_min = logit_safe(w_min + 0.05) - min_cov_effect;

            logit_wdiag_intercept(k) = std::max(logit_target, required_min);
        } // for each source k
    } // initialize_horseshoe_dominant


    void initialize_horseshoe_zero()
    {
        theta.resize(ns, ns);
        gamma.resize(ns, ns);
        delta.resize(ns, ns);
        tau.resize(ns);
        zeta.resize(ns);
        logit_wdiag_intercept.resize(ns);
        logit_wdiag_slope.resize(np);

        theta.setZero();
        gamma.setOnes();
        delta.setOnes();
        tau.setOnes();
        zeta.setOnes();
        logit_wdiag_intercept.setConstant(5.0); // high retention (~=1)
        if (np > 0)
        {
            logit_wdiag_slope.setZero();
        }
    } // initialize_horseshoe_zero


    /**
     * @brief Initialize horseshoe with SPARSE deterministic starting point
     *
     * Sets theta=0 (no spatial transmission initially) and high retention.
     * Useful for conservative MCMC starts where you want to discover structure.
     *
     * @param cov_specs Covariate specifications
     * @param w_baseline Target baseline retention when covariates = 0
     * @param tau_init Initial tau (global shrinkage) - smaller = stronger shrinkage
     * @param gamma_init Initial gamma (local shrinkage)
     */
    void initialize_horseshoe_sparse(
        const double &w_baseline = 0.85,
        const double &tau_init = 0.1,
        const double &gamma_init = 1.0
    )
    {
        // Step 1: Run `initialize_logit_wdiag_slopes` if covariates are included or specify `logit_wdiag_slope` manually

        // Resize parameters
        theta.resize(ns, ns);
        gamma.resize(ns, ns);
        delta.resize(ns, ns);
        tau.resize(ns);
        zeta.resize(ns);
        logit_wdiag_intercept.resize(ns);

        // Step 2: Deterministic horseshoe parameters
        theta.setZero();
        gamma.setConstant(gamma_init);
        delta.setOnes();
        tau.setConstant(tau_init);
        zeta.setOnes();

        // Step 3: Set intercepts
        for (Eigen::Index k = 0; k < ns; k++)
        {
            // With theta=0, u_s = 1 for all s ≠ k, so w_min = 1/ns
            double w_min = 1.0 / static_cast<double>(ns);

            // Find worst-case covariate effect
            auto [min_cov_effect, max_cov_effect] = compute_covariate_effect_range(
                X, logit_wdiag_slope, k
            );

            // Set intercept for target baseline, ensuring dominance
            double required_for_target = logit_safe(w_baseline);
            double required_for_dominance = logit_safe(w_min + 0.1) - min_cov_effect;

            logit_wdiag_intercept(k) = std::max(required_for_target, required_for_dominance);
        }
    } // initialize_horseshoe_sparse


    double compute_regularized_gamma(
        const Eigen::Index &s,
        const Eigen::Index &k
    ) const
    {
        double gamma_sq = gamma(s, k);
        
        // Regularized variance: c² γ² / (c² + τ² γ²)
        double numerator = c_sq * gamma_sq;
        double denominator = c_sq + tau(k) * gamma_sq;
        return numerator / std::max(denominator, EPS);
    }


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
        return tau(k) * compute_regularized_gamma(s, k);
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
    }


    double compute_w_elem(const Eigen::Index &k, const Eigen::Index &t) const
    {
        double logit_w_k = logit_wdiag_intercept(k);
        if (np > 0)
        {
            for (Eigen::Index p = 0; p < np; p++)
            {
                logit_w_k += logit_wdiag_slope(p) * X(p, k, t);
            }
        }
        return inv_logit_stable(logit_w_k);
    } // compute_w_elem


    double compute_alpha_elem(
        const Eigen::VectorXd &u_k,
        const Eigen::Index &s,
        const Eigen::Index &k,
        const Eigen::Index &t
    ) const
    {
        if (ns == 1)
        {
            return 1.0;
        }
        else
        {
            double U_off = u_k.sum(); // u_k(k) == 0
            if (U_off <= EPS8)
            {
                if (s == k)
                {
                    return 1.0;
                }
                else
                {
                    return 0.0;
                }
            }

            double w_k = compute_w_elem(k, t);
            if (s == k)
            {
                return w_k;
            }
            else
            {
                return (1.0 - w_k) * u_k(s) / U_off;
            }
        }
    } // compute_alpha_elem


    double dlogprob_dlog_tau_k(
        const unsigned int &k, 
        const bool &add_prior = true
    )
    {
        double dloglike_dtau = 0.0;
        for (unsigned int s = 0; s < ns; s++)
        {
            if (s == k)
            {
                continue; // skip diagonal
            }

            double reg_var = compute_regularized_variance(s, k);
            double denom = c_sq + tau(k) * gamma(s, k);
            double dreg_var_dtau = c_sq * c_sq * gamma(s, k) / (denom * denom);
            double dloglike_dreg_var = - 0.5 * (1.0/reg_var - theta(s, k) * theta(s,k) / (reg_var * reg_var));

            dloglike_dtau += dloglike_dreg_var * dreg_var_dtau;
        }

        double dlogprior_dtau = - 1.5 / tau(k) + 1.0 / (zeta(k) * tau(k) * tau(k));
        double deriv = dloglike_dtau;
        if (add_prior)
        {
            deriv += dlogprior_dtau;
        }

        deriv *= tau(k); // Jacobian adjustment for log-transform
        deriv += 1.0;

        return deriv;
    }


    double dlogprob_dlog_gamma(
        const unsigned int &s, 
        const unsigned int &k, 
        const bool &add_prior = true
    )
    {
        double theta_sq = theta(s, k) * theta(s, k);
        double denom = c_sq + tau(k) * gamma(s, k);
        double gamma_tilde = (c_sq * gamma(s, k)) / std::max(denom, EPS);
        double dgamma_tilde_dgamma = (c_sq * c_sq) / (denom * denom);

        double dloglike_dgamma_tilde = - 0.5 * (1.0 - theta_sq / (tau(k) * gamma_tilde)) / gamma_tilde;
        double deriv = dloglike_dgamma_tilde * dgamma_tilde_dgamma;
        if (add_prior)
        {
            double dlogprior_dgamma = - 1.5 / gamma(s, k) + 1.0 / (delta(s, k) * gamma(s, k) * gamma(s, k));
            deriv += dlogprior_dgamma;
        }
    
        deriv *= gamma(s, k); // Jacobian adjustment for log-transform
        deriv += 1.0;

        return deriv;
    }


    /**
     * @brief Derivative of alpha(a_s, k, t) with respect to theta(th_j, k)
     * 
     * @param u_k 
     * @param a_s 
     * @param th_j 
     * @param k 
     * @param t 
     * @return double 
     */
    double dalpha_dtheta(
        const Eigen::VectorXd &u_k, // unnormalized weights for source k
        const Eigen::Index &k, // source location index for both alpha and theta
        const Eigen::Index &th_j, // destination location index for theta
        const Eigen::Index &a_s, // destination location index for alpha
        const Eigen::Index &t = 0 // time index for alpha (retention weight computation)
    )
    {
        if (a_s == k)
        {
            return 0.0; // alpha[k, k, t] does not depend on theta
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

            deriv *= 1.0 - compute_w_elem(k, t); // multiply by (1 - w_k)
            return deriv;
        }
    } // dalpha_dtheta


    /**
     * @brief Derivative of alpha(a_s, j, t) with respect to w_j (the baseline retention weight for location j in the logit scale).
     * 
     * @param u_j 
     * @param s 
     * @param j 
     * @param t 
     * @return double 
     */
    double dalpha_dwj(
        const Eigen::VectorXd &u_j, // unnormalized weights for source j
        const Eigen::Index &s, // destination location index
        const Eigen::Index &j, // source location index
        const Eigen::Index &t = 0 // time index for alpha (retention weight computation)
    )
    {
        double dalpha_dw_jt = 0.0;
        if (s == j)
        {
            dalpha_dw_jt = 1.0;
        }
        else
        {
            double U_j = u_j.array().sum(); // u_j(j) == 0
            dalpha_dw_jt = - u_j(s) / U_j;
        }

        double w_jt = compute_w_elem(j, t);
        double dw_jt_d_logit_wj = w_jt * (1.0 - w_jt);
        double deriv = dalpha_dw_jt * dw_jt_d_logit_wj;
        return deriv;
    } // dalpha_dwj


    Eigen::VectorXd dalpha_dslope(
        const Eigen::VectorXd &u_j, // unnormalized weights for source j
        const Eigen::Index &s, // destination location index
        const Eigen::Index &j, // source location index
        const Eigen::Index &t = 0 // time index for alpha (retention weight computation)
    )
    {
        double dalpha_dw_jt = 0.0;
        if (s == j)
        {
            dalpha_dw_jt = 1.0;
        }
        else
        {
            double U_j = u_j.array().sum(); // u_j(j) == 0
            dalpha_dw_jt = - u_j(s) / U_j;
        }

        double w_jt = compute_w_elem(j, t);
        double dw_jt_d_logit_wj = w_jt * (1.0 - w_jt);
        Eigen::VectorXd deriv(np);
        for (Eigen::Index p = 0; p < np; p++)
        {
            deriv(p) = dalpha_dw_jt * dw_jt_d_logit_wj * X(p, j, t);
        }

        return deriv;
    } // dalpha_dslope


    double dalpha_dlog_rho_dist(
        const Eigen::VectorXd &u_k, // unnormalized weights for source k
        const Eigen::Index &s, // destination location index
        const Eigen::Index &k, // source location index
        const Eigen::Index &t = 0 // time index for alpha (retention weight computation)
    )
    {
        if (s == k)
        {
            return 0.0; // alpha(k, k, t) does not depend on rho_mobility
        }

        double U_k = u_k.array().sum(); // u_k(k) == 0
        double D_k = u_k.dot(dist.col(k)); // sum_s u_k(s) * dist(s, k), u_k(k) == 0 & dist(k, k) == 0
        double w_kt = compute_w_elem(k, t);
        double alpha_skt = (1.0 - w_kt) * u_k(s) / U_k;
        double deriv = - alpha_skt * (dist(s, k) - D_k / U_k);

        deriv *= rho_dist; // jacobian adjustment for log-transform

        return deriv;
    } // dalpha_drho_dist


    double dalpha_dlog_rho_mobility(
        const Eigen::VectorXd &u_k, // unnormalized weights for source k
        const Eigen::Index &s, // destination location index
        const Eigen::Index &k, // source location index
        const Eigen::Index &t = 0 // time index for alpha (retention weight computation)
    )
    {
        if (s == k)
        {
            return 0.0; // alpha(k, k, t) does not depend on rho_mobility
        }

        double U_k = u_k.array().sum(); // u_k(k) == 0
        double M_k = u_k.dot(log_mobility.col(k)); // sum_s u_k(s) * log_mobility(s, k), u_k(k) == 0 & log_mobility(k, k) == 0
        double w_kt = compute_w_elem(k, t);
        double alpha_skt = (1.0 - w_kt) * u_k(s) / U_k;
        double deriv = alpha_skt * (log_mobility(s, k) - M_k / U_k);

        deriv *= rho_mobility; // jacobian adjustment for log-transform

        return deriv;
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


    /**
     * @brief Compute alpha element handling both historical and forecast time indices
     */
    double compute_alpha_for_time(
        const Eigen::VectorXd &u_k,
        const Eigen::Index s,
        const Eigen::Index k,
        const Eigen::Index t,
        const Eigen::Index T_obs,
        const Eigen::Tensor<double, 3> &X_forecast,
        const bool has_forecast_covariates) const
    {
        const Eigen::Index ns = spatial.theta.cols();

        if (ns == 1)
        {
            return 1.0;
        }

        double U_off = u_k.sum();
        if (U_off <= EPS8)
        {
            return (s == k) ? 1.0 : 0.0;
        }

        // Compute w_{k,t} with appropriate covariates
        double logit_w_k = spatial.logit_wdiag_intercept(k);

        if (spatial.include_covariates && spatial.np > 0)
        {
            if (t < T_obs)
            {
                // Use historical covariates
                for (Eigen::Index p = 0; p < spatial.np; p++)
                {
                    logit_w_k += spatial.logit_wdiag_slope(p) * spatial.X(p, k, t);
                }
            }
            else if (has_forecast_covariates)
            {
                // Use forecast covariates
                Eigen::Index t_forecast = t - T_obs;
                for (Eigen::Index p = 0; p < spatial.np; p++)
                {
                    logit_w_k += spatial.logit_wdiag_slope(p) * X_forecast(p, k, t_forecast);
                }
            }
            // If no forecast covariates provided, use intercept only (assumes covariates = 0)
        }

        double w_k = inv_logit_stable(logit_w_k);

        if (s == k)
        {
            return w_k;
        }
        else
        {
            return (1.0 - w_k) * u_k(s) / U_off;
        }
    } // compute_alpha_for_time


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
        const Rcpp::Nullable<Rcpp::NumericVector> &X_in = R_NilValue,
        const double &c_sq = 4.0,
        const std::string &fgain_ = "softplus",
        const bool &zero_inflated = false,
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
        spatial = SpatialNetwork(
            ns, true, c_sq, 
            dist_scaled_in, 
            mobility_scaled_in, 
            X_in
        );

        zinfl = ZeroInflation(Y, zero_inflated);

        N0.resize(nt + 1, ns);
        N0.setOnes();
        return;
    } // Model constructor for MCMC inference when model parameters are unknown


    Model(
        const Eigen::Index &nt_,
        const Eigen::Index &ns_,
        const Rcpp::Nullable<Rcpp::NumericMatrix> &dist_scaled_in = R_NilValue,
        const Rcpp::Nullable<Rcpp::NumericMatrix> &mobility_scaled_in = R_NilValue,
        const Rcpp::Nullable<Rcpp::NumericVector> &X_in = R_NilValue,
        const Rcpp::Nullable<Rcpp::NumericVector> &mean_slopes_in = R_NilValue,
        const Rcpp::Nullable<Rcpp::NumericVector> &sd_slopes_in = R_NilValue,
        const double &mu_in = 1.0,
        const double &W_in = 0.001,
        const std::string &fgain_ = "softplus",
        const Rcpp::List &spatial_opts = Rcpp::List::create(
            Rcpp::Named("c_sq") = 4.0,
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
        spatial = SpatialNetwork(
            ns, spatial_opts, 
            dist_scaled_in, 
            mobility_scaled_in, 
            X_in, mean_slopes_in, sd_slopes_in
        );
        zinfl = ZeroInflation(ns, nt, zinfl_opts);

        N.resize(ns, ns, nt + 1, nt + 1);
        N.setZero();

        N0.resize(nt + 1, ns);
        N0.setOnes();
        return;
    } // Model constructor for simulation when model parameters are known but Y is to be simulated


    Eigen::MatrixXd 
    
    compute_intensity(
        const Eigen::MatrixXd &Y, // (nt + 1) x ns, observed primary infections
        const Eigen::MatrixXd &R_mat // (nt + 1) x ns
    )
    {
        Eigen::Index nt = Y.rows() - 1;
        Eigen::Index ns = Y.cols();

        Eigen::MatrixXd u_mat(ns, ns);
        for (Eigen::Index k = 0; k < ns; k++)
        {
            u_mat.col(k) = spatial.compute_unnormalized_weight_col(k);
        }

        Eigen::MatrixXd lambda_mat(nt + 1, ns);
        for (Eigen::Index t = 0; t < nt + 1; t++)
        { // for destination time t
            for (Eigen::Index s = 0; s < ns; s++)
            { // for destination location s
                double lambda_st = mu;
                for (Eigen::Index k = 0; k < ns; k++)
                {
                    Eigen::VectorXd u_k = u_mat.col(k);
                    for (Eigen::Index l = 0; l < t; l++)
                    {
                        if (t - l <= static_cast<Eigen::Index>(dlag.Fphi.size()))
                        {
                            double alpha_skl = spatial.compute_alpha_elem(u_k, s, k, l);
                            lambda_st += alpha_skl * R_mat(l, k) * dlag.Fphi(t - l - 1) * Y(l, k);
                        }
                    } // for source time l < t
                }
                lambda_mat(t, s) = std::max(lambda_st, EPS);
            } // for destination location s
        } // for time t
        return lambda_mat;
    } // compute_intensity


    Eigen::Index get_ndim_colvec(
        Eigen::VectorXi & idx_start, // starting index of each parameter block
        Eigen::VectorXi & inverse_idx, // inverse mapping from spatial index to off-diagonal index
        const Eigen::Index &k, // column index
        const bool &include_covariates = true,
        const bool &include_rho_dist = true,
        const bool &include_rho_mobility = true)
    {
        /*
        Store starting indices for:
        0 - theta
        1 - log(gamma)
        2 - log(tau)
        3 - logit(wdiag)
        4 - covariates
        5 - rho_dist
        6 - rho_mobility
        */
        idx_start.resize(7);
        idx_start.setConstant(-1);

        const Eigen::Index n_offdiag = ns - 1;
        idx_start(0) = 0; // theta start
        idx_start(1) = n_offdiag; // log(gamma) start
        idx_start(2) = 2 * n_offdiag; // log(tau)
        idx_start(3) = 2 * n_offdiag + 1; // logit_wdiag_intercept

        Eigen::Index ndim = 2 * n_offdiag + 2;
        if (include_covariates && spatial.include_covariates)
        {
            idx_start(4) = ndim; // logit_wdiag_slope
            ndim += spatial.np;
        }
        if (include_rho_dist && spatial.include_distance)
        {
            idx_start(5) = ndim; // log(rho_dist)
            ndim++;
        }
        if (include_rho_mobility && spatial.include_log_mobility)
        {
            idx_start(6) = ndim; // log(rho_mobility)
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
     * @brief get_unconstrained with rho parameters
        z[k] = (
            theta_{s,k} for s != k, 
            log(gamma_{s,k} for s != k), 
            log(tau_k), 
            logit_wdiag_intercept_k, 
            logit_wdiag_slope, 
            [log(rho_dist)], 
            [log(rho_mobility)]
        )
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
            unconstrained(idx_start(0) + i) = spatial.theta(s, k); // theta: from 0 to n_offdiag - 1
            unconstrained(idx_start(1) + i) = std::log(spatial.gamma(s, k)); // gamma: from n_offdiag to 2*n_offdiag - 1
        }
        unconstrained(idx_start(2)) = std::log(spatial.tau(k));
        unconstrained(idx_start(3)) = spatial.logit_wdiag_intercept(k);
        
        if (idx_start(4) >= 0)
        {
            for (Eigen::Index p = 0; p < spatial.np; p++)
            {
                unconstrained(idx_start(4) + p) = spatial.logit_wdiag_slope(p);
            }
        }

        if (idx_start(5) >= 0)
        {
            unconstrained(idx_start(5)) = std::log(std::max(spatial.rho_dist, EPS));
        }

        if (idx_start(6) >= 0)
        {
            unconstrained(idx_start(6)) = std::log(std::max(spatial.rho_mobility, EPS));
        }

        return unconstrained;
    } // get_unconstrained_colvec


    void unpack_unconstrained_colvec(
        const Eigen::ArrayXd &unconstrained_params,
        const Eigen::Index &k,
        const Eigen::VectorXi &idx_start,
        const Eigen::VectorXi &inverse_idx
    )
    {
        const Eigen::Index n_offdiag = ns - 1;

        for (Eigen::Index s = 0; s < ns; s++)
        {
            if (s == k)
                continue;
            Eigen::Index i = inverse_idx[s];
            spatial.theta(s, k) = unconstrained_params(idx_start(0) + i);
            spatial.gamma(s, k) = std::exp(clamp_log_scale(unconstrained_params(idx_start(1) + i)));
        }
        spatial.theta(k, k) = 0.0;
        spatial.tau(k) = std::exp(clamp_log_scale(unconstrained_params(idx_start(2))));
        spatial.logit_wdiag_intercept(k) = unconstrained_params(idx_start(3));

        if (idx_start(4) >= 0)
        {
            for (Eigen::Index p = 0; p < spatial.np; p++)
            {
                spatial.logit_wdiag_slope(p) = unconstrained_params(idx_start(4) + p);
            }
        }

        if (idx_start(5) >= 0)
        {
            spatial.rho_dist = std::exp(clamp_log_scale(unconstrained_params(idx_start(5))));
        }

        if (idx_start(6) >= 0)
        {
            spatial.rho_mobility = std::exp(clamp_log_scale(unconstrained_params(idx_start(6))));
        }
    } // unpack_unconstrained_colvec


    /**
     * @brief Compute gradient for horseshoe parameters of column k, optionally including rho
     *
     * Parameter layout in returned gradient vector:
     *   [0, n_offdiag-1]                    : theta_{s,k} for s != k
     *   [n_offdiag, 2*n_offdiag-1]          : log(gamma_{s,k}) for s != k
     *   [2*n_offdiag]                       : log(tau_k)
     *   [2*n_offdiag+1]                     : logit(wdiag_k)
     *   [2*n_offdiag+2, 2*n_offdiag+2+np-1] : logit_wdiag_slope (if include_covariates)
     *   [2*n_offdiag+2+np]                  : log(rho_dist) (if include_rho_dist)
     *   [2*n_offdiag+2+np+1]                : log(rho_mobility) (if include_rho_mobility)
     *
     * @param loglike Output: log-likelihood value
     * @param k Column index
     * @param Y Observed infections
     * @param R_mat Reproduction numbers
     * @param include_covariates Include covariate effects in the logit retention weight
     * @param include_rho_dist Include rho_dist in the unnormalized weight computation
     * @param include_rho_mobility Include rho_mobility in the unnormalized weight computation
     * @param beta_prior_a Beta prior parameter for wdiag
     * @param beta_prior_b Beta prior parameter for wdiag
     * @param gauss_prior_mean Prior mean for unconstrained parameters
     * @param gauss_prior_var Prior variance for unconstrained parameters
     */
    Eigen::VectorXd dlogprob_dcolvec(
        double &logprob,
        const Eigen::Index &k,
        const Eigen::MatrixXd &Y,
        const Eigen::MatrixXd &R_mat,
        const bool &include_covariates = true,
        const bool &include_rho_dist = true,
        const bool &include_rho_mobility = true,
        const double &beta_prior_a = 5.0,
        const double &beta_prior_b = 2.0,
        const double &gauss_prior_mean = 0.0,
        const double &gauss_prior_sd = 3.0)
    {
        const double gauss_prior_var = gauss_prior_sd * gauss_prior_sd;
        Eigen::VectorXi idx_start, inverse_idx;
        Eigen::Index ndim = get_ndim_colvec(
            idx_start,
            inverse_idx,
            k,
            include_covariates,
            include_rho_dist,
            include_rho_mobility
        );

        Eigen::MatrixXd u_mat(ns, ns);
        for (Eigen::Index j = 0; j < ns; j++)
        {
            u_mat.col(j) = spatial.compute_unnormalized_weight_col(j);
        }

        logprob = 0.0;
        Eigen::VectorXd grad(ndim);
        grad.setZero();


        /*
        The prior contributions from log(tau[k]) and log(gamma[s,k]) to the gradients are included in `spatial.dlogprob_dlog_tau_k` and `spatial.dlogprob_dlog_gamma`.
        So we only need to add them to the log-likelihood here.
        */
        // tau prior
        grad(idx_start(2)) = spatial.dlogprob_dlog_tau_k(k, true); // include prior in the gradient
        logprob += -0.5 * std::log(spatial.tau(k)) - 1.0 / (spatial.zeta(k) * spatial.tau(k));

        // gamma prior
        for (Eigen::Index s = 0; s < ns; s++)
        {
            if (s == k)
                continue;
            Eigen::Index i = inverse_idx[s];
            logprob += -0.5 * std::log(spatial.gamma(s, k)) - 1.0 / (spatial.delta(s, k) * spatial.gamma(s, k));
        }

        // logit_wdiag_intercept prior (now using a beta prior on wdiag, maybe switch to normal prior on logit_wdiag later?)
        double logit_wk = spatial.logit_wdiag_intercept(k);
        double w_safe = inv_logit_stable(logit_wk);
        grad(idx_start(3)) += beta_prior_a * (1.0 - w_safe) - beta_prior_b * w_safe;
        logprob += beta_prior_a * std::log(w_safe) + beta_prior_b * std::log(1.0 - w_safe);

        // Prior on logit_wdiag_slope
        if (idx_start(4) >= 0)
        {
            for (Eigen::Index p = 0; p < spatial.np; p++)
            {
                double diff = spatial.logit_wdiag_slope(p) - gauss_prior_mean;
                grad(idx_start(4) + p) += -diff / gauss_prior_var;
                logprob += -0.5 * diff * diff / gauss_prior_var;
            }
        }

        // Prior on log(rho_dist)
        if (idx_start(5) >= 0)
        {
            double diff = std::log(std::max(spatial.rho_dist, EPS)) - gauss_prior_mean;
            grad(idx_start(5)) += -diff / gauss_prior_var;
            logprob += -0.5 * diff * diff / gauss_prior_var;
        }

        // Prior on log(rho_mobility)
        if (idx_start(6) >= 0)
        {
            double diff = std::log(std::max(spatial.rho_mobility, EPS)) - gauss_prior_mean;
            grad(idx_start(6)) += -diff / gauss_prior_var;
            logprob += -0.5 * diff * diff / gauss_prior_var;
        }

        // Loop over destination locations for theta/gamma gradients
        for (Eigen::Index s = 0; s < ns; s++)
        {
            // gamma gradient including prior (only for off-diagonal)
            if (s != k)
            {
                Eigen::Index i = inverse_idx[s];
                grad(idx_start(1) + i) = spatial.dlogprob_dlog_gamma(s, k, true); // include prior in the gradient


                // theta prior (only for off-diagonal): theta[s,k] ~ N(0, g[s,k]), where g[s,k] = regularized variance.
                double reg_var = std::max(spatial.compute_regularized_variance(s, k), EPS8);
                grad(i) += -spatial.theta(s, k) / reg_var;
                logprob += -0.5 * std::log(reg_var) - 0.5 * spatial.theta(s, k) * spatial.theta(s, k) / reg_var;
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

                Eigen::VectorXd dlambda_st_dtheta_k(ns - 1);
                dlambda_st_dtheta_k.setZero();

                double dlambda_st_dlogit_wdiag_intercept_k = 0.0;

                Eigen::VectorXd dlambda_st_dlogit_wdiag_slope;
                if (include_covariates && spatial.include_covariates)
                {
                    dlambda_st_dlogit_wdiag_slope = Eigen::VectorXd::Zero(spatial.np);
                }

                double dlambda_st_dlog_rho_dist = 0.0;
                double dlambda_st_dlog_rho_mobility = 0.0;

                for (Eigen::Index i = 0; i < ns; i++)
                {
                    Eigen::VectorXd u_i = u_mat.col(i);
                    for (Eigen::Index l = 0; l < t; l++)
                    {
                        if (t - l <= static_cast<Eigen::Index>(dlag.Fphi.size()))
                        {
                            double alpha_sil = spatial.compute_alpha_elem(u_i, s, i, l);
                            double coef = R_mat(l, i) * dlag.Fphi(t - l - 1) * Y(l, i);
                            lambda_st += alpha_sil * coef;

                            if (i == k)
                            {
                                // alpha(s, k, l) contributes to the gradient w.r.t. theta(1, k), ..., theta(ns, k)
                                for (Eigen::Index j = 0; j < ns; j++)
                                {
                                    if (j == k)
                                        continue;
                                    
                                    Eigen::Index idx_j = inverse_idx[j];
                                    double dalpha_skl_dtheta_jk = spatial.dalpha_dtheta(u_i, k, j, s, l);
                                    dlambda_st_dtheta_k(idx_j) += dalpha_skl_dtheta_jk * coef;
                                } // gradient contribution for all theta(j, k), j != k

                                double dalpha_skl_dlogit_wk = spatial.dalpha_dwj(u_i, s, k, l);
                                dlambda_st_dlogit_wdiag_intercept_k += dalpha_skl_dlogit_wk * coef;
                            } // Only if source location i == k

                            if (idx_start(4) >= 0)
                            {
                                dlambda_st_dlogit_wdiag_slope += spatial.dalpha_dslope(u_i, s, i, l) * coef;
                            }

                            if (idx_start(5) >= 0)
                            {
                                dlambda_st_dlog_rho_dist += spatial.dalpha_dlog_rho_dist(u_i, s, i, l) * coef;
                            }

                            if (idx_start(6) >= 0)
                            {
                                dlambda_st_dlog_rho_mobility += spatial.dalpha_dlog_rho_mobility(u_i, s, i, l) * coef;
                            }
                        }
                    } // for source time l < t
                } // for source location i

                lambda_st = std::max(lambda_st, EPS);
                logprob += Y(t, s) * std::log(lambda_st) - lambda_st;
                double dloglike_dlambda_st = Y(t, s) / lambda_st - 1.0;

                // Gradient w.r.t. theta(j, k) for j != k
                for (Eigen::Index j = 0; j < ns - 1; j++)
                {
                    grad(idx_start(0) + j) += dloglike_dlambda_st * dlambda_st_dtheta_k(j);
                }

                // Gradient w.r.t. the baseline retention weight in the logit scale
                grad(idx_start(3)) += dloglike_dlambda_st * dlambda_st_dlogit_wdiag_intercept_k;


                if (idx_start(4) >= 0)
                {
                    // Gradient w.r.t. the covariate slopes in the logit scale
                    for (Eigen::Index p = 0; p < spatial.np; p++)
                    {
                        grad(idx_start(4) + p) += dloglike_dlambda_st * dlambda_st_dlogit_wdiag_slope(p);
                    }
                }

                if (idx_start(5) >= 0)
                {
                    // Gradient w.r.t. log(rho_dist)
                    grad(idx_start(5)) += dloglike_dlambda_st * dlambda_st_dlog_rho_dist;
                }

                if (idx_start(6) >= 0)
                {
                    // Gradient w.r.t. log(rho_mobility)
                    grad(idx_start(6)) += dloglike_dlambda_st * dlambda_st_dlog_rho_mobility;
                }
            } // for time t
        } // for destination locations s

        return grad;
    } // dloglike_dcolvec


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
        const bool &include_covariates = true,
        const bool &include_rho_dist = true,
        const bool &include_rho_mobility = false,
        const double &beta_prior_a = 5.0,
        const double &beta_prior_b = 2.0,
        const double &gauss_prior_mean = 0.0,
        const double &gauss_prior_sd = 3.0)
    {
        const Eigen::Index n_offdiag = ns - 1;
        Eigen::VectorXi idx_start, inverse_idx;
        Eigen::Index ndim = get_ndim_colvec(
            idx_start,
            inverse_idx,
            k,
            include_covariates,
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
            include_covariates,
            include_rho_dist, 
            include_rho_mobility,
            beta_prior_a, beta_prior_b, 
            gauss_prior_mean, gauss_prior_sd
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
                include_covariates,
                include_rho_dist, 
                include_rho_mobility,
                beta_prior_a, beta_prior_b, 
                gauss_prior_mean, gauss_prior_sd
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

        // Gibbs updates for delta and zeta (conjugate)
        spatial.zeta(k) = 1.0 / R::rgamma(1.0, 1.0 / (1.0 + 1.0 / spatial.tau(k)));
        for (Eigen::Index s = 0; s < ns; s++)
        {
            if (s != k)
            {
                double dl_rate = 1.0 + 1.0 / spatial.gamma(s, k);
                spatial.delta(s, k) = 1.0 / R::rgamma(1.0, 1.0 / dl_rate);
            }
            else
            {
                spatial.delta(s, k) = 0.0;
            }
        }

        return accept_prob;
    } // update_colvec


    Eigen::Index get_ndim_static_params(
        Eigen::VectorXi &idx_start, 
        const bool &include_W = false
    )
    {
        idx_start.resize(5);
        idx_start.setConstant(-1);
        /*
        idx_start layout:
        0 - mu
        1 - W
        2 - rho_dist
        3 - rho_mobility
        4 - covariate slopes
        */

        idx_start(0) = 0; // mu
        Eigen::Index n_params = 1; // mu

        if (include_W)
        {
            idx_start(1) = n_params; // W
            n_params += 1;
        }

        if (spatial.include_distance)
        {
            idx_start(2) = n_params; // rho_dist
            n_params += 1;
        }

        if (spatial.include_log_mobility)
        {
            idx_start(3) = n_params; // rho_mobility
            n_params += 1;
        }

        if (spatial.include_covariates)
        {
            idx_start(4) = n_params; // covariate slopes
            n_params += spatial.np;
        }

        return n_params;
    } // get_ndim_static_params


    Eigen::ArrayXd get_unconstrained_static_params(
        const Eigen::Index &n_params,
        const Eigen::VectorXi &idx_start
    )
    {
        Eigen::ArrayXd unconstrained_params(n_params);

        unconstrained_params(idx_start(0)) = std::log(std::max(mu, EPS));
        
        if (idx_start(1) >= 0)
        {
            unconstrained_params(idx_start(1)) = std::log(std::max(temporal.W, EPS));
        }

        if (idx_start(2) >= 0)
        {
            unconstrained_params(idx_start(2)) = std::log(std::max(spatial.rho_dist, EPS));
        }

        if (idx_start(3) >= 0)
        {
            unconstrained_params(idx_start(3)) = std::log(std::max(spatial.rho_mobility, EPS));
        }

        if (idx_start(4) >= 0)
        {
            for (Eigen::Index p = 0; p < spatial.np; p++)
            {
                unconstrained_params(idx_start(4) + p) = spatial.logit_wdiag_slope(p);
            }
        }
        return unconstrained_params;
    } // get_unconstrained_static_params


    void unpack_unconstrained_static_params(
        const Eigen::ArrayXd &unconstrained_params,
        const Eigen::VectorXi &idx_start
    )
    {
        Eigen::Index idx = 0;
        mu = std::exp(clamp_log_scale(unconstrained_params(idx_start(0))));

        if (idx_start(1) >= 0)
        {
            temporal.W = std::exp(clamp_log_scale(unconstrained_params(idx_start(1))));
        }

        if (idx_start(2) >= 0)
        {
            spatial.rho_dist = std::exp(clamp_log_scale(unconstrained_params(idx_start(2))));
        }

        if (idx_start(3) >= 0)
        {
            spatial.rho_mobility = std::exp(clamp_log_scale(unconstrained_params(idx_start(3))));
        }

        if (idx_start(4) >= 0)
        {
            for (Eigen::Index p = 0; p < spatial.np; p++)
            {
                spatial.logit_wdiag_slope(p) = unconstrained_params(idx_start(4) + p);
            }
        }

        return;
    } // unpack_unconstrained_static_params


    Eigen::VectorXd dlogprob_dstatic_params(
        double &logprob,
        const Eigen::MatrixXd &Y, // (nt + 1) x ns, observed primary infections
        const Eigen::MatrixXd &R_mat, // (nt + 1) x ns
        const bool &include_W = false,
        const double &gauss_prior_mean = 0.0,
        const double &gauss_prior_sd = 3.0
    )
    {
        const double gauss_prior_var = gauss_prior_sd * gauss_prior_sd;

        Eigen::VectorXi idx_start;
        Eigen::Index n_params = get_ndim_static_params(idx_start, include_W);

        Eigen::MatrixXd u_mat(ns, ns);
        for (Eigen::Index j = 0; j < ns; j++)
        {
            u_mat.col(j) = spatial.compute_unnormalized_weight_col(j);
        }

        double W_safe = std::max(temporal.W, EPS);
        double W_sqrt = std::sqrt(W_safe);

        Eigen::VectorXd grad(n_params);
        grad.setZero();
        logprob = 0.0;

        /*
        Add priors to gradient and log-likelihood (jacobian included)
        */
        // Prior on mu (normal on log scale)
        double log_mu_diff = std::log(std::max(mu, EPS)) - gauss_prior_mean;
        grad(0) += -log_mu_diff / gauss_prior_var;
        logprob += -0.5 * log_mu_diff * log_mu_diff / gauss_prior_var;

        if (idx_start(1) >= 0)
        {
            // Prior on W (normal on log scale)
            double log_W_diff = std::log(std::max(temporal.W, EPS)) - gauss_prior_mean;
            grad(1) += -log_W_diff / gauss_prior_var;
            logprob += -0.5 * log_W_diff * log_W_diff / gauss_prior_var;
        }

        if (idx_start(2) >= 0)
        {
            // Prior on rho_dist (normal on log scale)
            double log_rho_dist_diff = std::log(std::max(spatial.rho_dist, EPS)) - gauss_prior_mean;
            grad(idx_start(2)) += -log_rho_dist_diff / gauss_prior_var;
            logprob += -0.5 * log_rho_dist_diff * log_rho_dist_diff / gauss_prior_var;
        }

        if (idx_start(3) >= 0)
        {
            // Prior on rho_mobility (normal on log scale)
            double log_rho_mobility_diff = std::log(std::max(spatial.rho_mobility, EPS)) - gauss_prior_mean;
            grad(idx_start(3)) += -log_rho_mobility_diff / gauss_prior_var;
            logprob += -0.5 * log_rho_mobility_diff * log_rho_mobility_diff / gauss_prior_var;
        }

        if (idx_start(4) >= 0)
        {
            // Prior on covariate slopes (normal)
            for (Eigen::Index p = 0; p < spatial.np; p++)
            {
                double diff = spatial.logit_wdiag_slope(p) - gauss_prior_mean;
                grad(idx_start(4) + p) += -diff / gauss_prior_var;
                logprob += -0.5 * diff * diff / gauss_prior_var;
            }
        }

        // Add likelihood contribution from destination (t, s) to gradient and log-likelihood
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
                double dlambda_st_dlog_rho_dist = 0.0;
                double dlambda_st_dlog_rho_mobility = 0.0;
                Eigen::VectorXd dlambda_st_dlogit_wdiag_slope;
                if (idx_start(4) >= 0)
                {
                    dlambda_st_dlogit_wdiag_slope = Eigen::VectorXd::Zero(spatial.np);
                }

                for (Eigen::Index k = 0; k < ns; k++)
                {
                    Eigen::VectorXd u_k = u_mat.col(k);
                    double coef_sum = 0.0;
                    for (Eigen::Index l = 0; l < t; l++)
                    {
                        if (t - l <= static_cast<Eigen::Index>(dlag.Fphi.size()))
                        {
                            double alpha_skl = spatial.compute_alpha_elem(u_k, s, k, l);
                            double coef = R_mat(l, k) * dlag.Fphi(t - l - 1) * Y(l, k);
                            lambda_st += alpha_skl * coef;

                            if (idx_start(2) >= 0)
                            {
                                dlambda_st_dlog_rho_dist += spatial.dalpha_dlog_rho_dist(u_k, s, k, l) * coef;
                            }

                            if (idx_start(3) >= 0)
                            {
                                dlambda_st_dlog_rho_mobility += spatial.dalpha_dlog_rho_mobility(u_k, s, k, l) * coef;
                            }

                            if (idx_start(4) >= 0)
                            {
                                dlambda_st_dlogit_wdiag_slope += spatial.dalpha_dslope(u_k, s, k, l) * coef;
                            }
                        }
                    } // for source time l < t
                } // for source location k

                lambda_st = std::max(lambda_st, EPS);
                logprob += Y(t, s) * std::log(lambda_st) - lambda_st;

                double dloglike_dlambda_st = Y(t, s) / lambda_st - 1.0;
                grad(idx_start(0)) += dloglike_dlambda_st * mu;
                
                if (idx_start(1) >= 0 && t > 0)
                {
                    logprob += R::dnorm4(temporal.wt(t, s), 0.0, W_sqrt, true);
                    double deriv_W = - 0.5 / W_safe + 0.5 * (temporal.wt(t, s) * temporal.wt(t, s)) / (W_safe * W_safe);
                    grad(idx_start(1)) += deriv_W * W_safe;
                }

                if (idx_start(2) >= 0)
                {
                    grad(idx_start(2)) += dloglike_dlambda_st * dlambda_st_dlog_rho_dist;
                }

                if (idx_start(3) >= 0)
                {
                    grad(idx_start(3)) += dloglike_dlambda_st * dlambda_st_dlog_rho_mobility;
                }

                if (idx_start(4) >= 0)
                {
                    for (Eigen::Index p = 0; p < spatial.np; p++)
                    {
                        grad(idx_start(4) + p) += dloglike_dlambda_st * dlambda_st_dlogit_wdiag_slope(p);
                    }
                }
            } // for destination location s
        } // for time t

        return grad;
    } // dloglike_dstatic_params


    double update_static_params(
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
        Eigen::VectorXi idx_start;
        Eigen::Index n_params = get_ndim_static_params(idx_start, include_W);

        Eigen::MatrixXd R_mat = temporal.compute_Rt(); // (nt + 1) x ns
        Eigen::VectorXd inv_mass = mass_diag.cwiseInverse();
        Eigen::VectorXd sqrt_mass = mass_diag.array().sqrt();

        // Current state
        Eigen::ArrayXd params_current = get_unconstrained_static_params(n_params, idx_start);
        Eigen::ArrayXd unconstrained_params = params_current;

        // Compute current energy and gradiant
        double logprob = 0.0;
        Eigen::VectorXd grad = dlogprob_dstatic_params(
            logprob, Y, R_mat, 
            include_W,
            prior_mean, prior_sd
        );
        grad_norm = grad.norm();
        double current_energy = - logprob;

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
            unconstrained_params += step_size * inv_mass.array() * momentum.array();
            unpack_unconstrained_static_params(unconstrained_params, idx_start);

            // Update gradient
            grad = dlogprob_dstatic_params(
                logprob, Y, R_mat, 
                include_W,
                prior_mean, prior_sd
            );

            // Update momentum
            if (lf_step != n_leapfrog - 1)
            {
                momentum += step_size * grad;
            }
        } // for leapfrog steps

        momentum += 0.5 * step_size * grad;
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
            // Revert to current state
            unpack_unconstrained_static_params(params_current, idx_start);
        } // end Metropolis step

        return accept_prob;
    } // update_static_params


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

        Eigen::MatrixXd u_mat(ns, ns);
        for (Eigen::Index j = 0; j < ns; j++)
        {
            u_mat.col(j) = spatial.compute_unnormalized_weight_col(j);
        }

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


        // Initialize the spatial weight matrix
        Eigen::Tensor<double, 3> alpha(ns, ns, nt + 1);
        alpha.setZero();
        // Simulate secondary infections N and observed primary infections Y over time
        for (Eigen::Index t = 1; t < nt + 1; t++)
        {
            for (Eigen::Index s = 0; s < ns; s++)
            {
                N0(t, s) = R::rpois(std::max(mu, EPS));
                double y_ts = N0(t, s);

                for (Eigen::Index k = 0; k < ns; k++)
                {
                    Eigen::VectorXd u_k = u_mat.col(k);
                    alpha(s, k, t) = spatial.compute_alpha_elem(u_k, s, k, t); // store spatial weights

                    for (Eigen::Index l = 0; l < t; l++)
                    {
                        if (t - l <= dlag.Fphi.size())
                        {
                            double alpha_skl = spatial.compute_alpha_elem(u_k, s, k, l);
                            double lag_prob = dlag.Fphi(t - l - 1);
                            double R_kl = Rt(l, k);
                            double lambda_sktl = alpha_skl * R_kl * lag_prob * Y(l, k) + EPS;
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
            Rcpp::Named("alpha") = tensor3_to_r(alpha),
            Rcpp::Named("N") = tensor4_to_r(N),
            Rcpp::Named("N0") = N0,
            Rcpp::Named("wt") = temporal.wt,
            Rcpp::Named("Rt") = Rt,
            Rcpp::Named("params") = params_list,
            Rcpp::Named("horseshoe") = spatial.to_list(),
            Rcpp::Named("zero_inflation") = zinfl.to_list()
        );

    } // simulate


    /**
     * @brief Continue simulation from existing data for k additional time steps
     *
     * @param Y_history Existing simulated data, (T_obs) x ns matrix
     * @param wt_history Existing disturbance trajectory, (T_obs) x ns matrix
     * @param k_ahead Number of future time steps to simulate
     * @param X_forecast Optional covariates for forecast period, np x ns x k_ahead tensor
     * @return Rcpp::List with Y_forecast, wt_forecast, lambda_forecast, Rt_forecast
     */
    Rcpp::List simulate_ahead(
        const Eigen::MatrixXd &Y_history,
        const Eigen::MatrixXd &wt_history,
        const Eigen::Index k_ahead,
        const Rcpp::Nullable<Rcpp::NumericVector> &X_forecast_in = R_NilValue)
    {
        const Eigen::Index T_obs = Y_history.rows();
        const Eigen::Index ns_check = Y_history.cols();

        if (ns_check != spatial.theta.cols())
        {
            Rcpp::stop("Dimension mismatch: Y_history columns must match number of locations.");
        }
        if (wt_history.rows() != T_obs || wt_history.cols() != ns_check)
        {
            Rcpp::stop("Dimension mismatch: wt_history must match Y_history dimensions.");
        }

        const Eigen::Index ns = ns_check;
        const double W_sqrt = std::sqrt(temporal.W);

        // Parse forecast covariates if provided
        Eigen::Tensor<double, 3> X_forecast;
        bool has_forecast_covariates = false;
        if (X_forecast_in.isNotNull() && spatial.include_covariates)
        {
            Rcpp::NumericVector X_vec(X_forecast_in);
            X_forecast = r_to_tensor3(X_vec);
            if (X_forecast.dimension(0) != spatial.np ||
                X_forecast.dimension(1) != ns ||
                X_forecast.dimension(2) != k_ahead)
            {
                Rcpp::stop("X_forecast must have dimensions (np x ns x k_ahead).");
            }
            has_forecast_covariates = true;
        }

        // Extend Y and wt matrices
        Eigen::MatrixXd Y_extended(T_obs + k_ahead, ns);
        Y_extended.setZero();
        Y_extended.block(0, 0, T_obs, ns) = Y_history;

        Eigen::MatrixXd wt_extended(T_obs + k_ahead, ns);
        wt_extended.setZero();
        wt_extended.block(0, 0, T_obs, ns) = wt_history;

        // Output matrices for forecast period only
        Eigen::MatrixXd Y_forecast(k_ahead, ns);
        Eigen::MatrixXd wt_forecast(k_ahead, ns);
        Eigen::MatrixXd lambda_forecast(k_ahead, ns);
        Eigen::MatrixXd Rt_forecast(k_ahead, ns);

        // Precompute unnormalized spatial weights (static part)
        Eigen::MatrixXd u_mat(ns, ns);
        for (Eigen::Index k = 0; k < ns; k++)
        {
            u_mat.col(k) = spatial.compute_unnormalized_weight_col(k);
        }

        // Simulate forward
        for (Eigen::Index t = T_obs; t < T_obs + k_ahead; t++)
        {
            const Eigen::Index t_forecast = t - T_obs; // Index into forecast arrays

            for (Eigen::Index s = 0; s < ns; s++)
            {
                // Sample new disturbance
                wt_extended(t, s) = R::rnorm(0.0, W_sqrt);
                wt_forecast(t_forecast, s) = wt_extended(t, s);

                // Compute R_{s,t} from cumulative wt
                double wt_cumsum = 0.0;
                for (Eigen::Index tau = 0; tau <= t; tau++)
                {
                    wt_cumsum += wt_extended(tau, s);
                }
                double R_st = GainFunc::psi2hpsi(wt_cumsum, temporal.fgain);
                Rt_forecast(t_forecast, s) = R_st;
            }

            // Now simulate Y for this time step
            for (Eigen::Index s = 0; s < ns; s++)
            {
                double lambda_st = mu; // Baseline

                for (Eigen::Index k = 0; k < ns; k++)
                {
                    Eigen::VectorXd u_k = u_mat.col(k);

                    // Compute cumulative wt for source location k (for R_k,l)
                    Eigen::VectorXd wt_cumsum_k = cumsum_vec(wt_extended.col(k).head(t));

                    for (Eigen::Index l = 0; l < t; l++)
                    {
                        Eigen::Index lag = t - l;
                        if (lag <= static_cast<Eigen::Index>(dlag.Fphi.size()))
                        {
                            // Compute alpha_{s,k,l} - need to handle covariates carefully
                            double alpha_skl = compute_alpha_for_time(
                                u_k, s, k, l, T_obs,
                                X_forecast, has_forecast_covariates);

                            double R_kl = GainFunc::psi2hpsi(wt_cumsum_k(l), temporal.fgain);
                            double phi_lag = dlag.Fphi(lag - 1);

                            lambda_st += alpha_skl * R_kl * phi_lag * Y_extended(l, k);
                        }
                    }
                }

                lambda_forecast(t_forecast, s) = lambda_st;

                // Sample Y from Poisson (or zero-inflated Poisson)
                double y_st = R::rpois(std::max(lambda_st, EPS));

                // Handle zero-inflation if needed
                if (zinfl.inflated)
                {
                    // For forecast, we need to sample Z as well
                    double z_prev = (t > 0) ? ((Y_extended(t - 1, s) > 0) ? 1.0 : 0.0) : 1.0;
                    double logit_p = zinfl.beta0 + zinfl.beta1 * z_prev;
                    double p_one = inv_logit_stable(logit_p);
                    double z_st = (R::runif(0.0, 1.0) < p_one) ? 1.0 : 0.0;

                    if (z_st < 0.5)
                    {
                        y_st = 0.0;
                    }
                }

                Y_extended(t, s) = y_st;
                Y_forecast(t_forecast, s) = y_st;
            }
        }

        return Rcpp::List::create(
            Rcpp::Named("Y_forecast") = Y_forecast,
            Rcpp::Named("wt_forecast") = wt_forecast,
            Rcpp::Named("lambda_forecast") = lambda_forecast,
            Rcpp::Named("Rt_forecast") = Rt_forecast,
            Rcpp::Named("Y_extended") = Y_extended,
            Rcpp::Named("wt_extended") = wt_extended);
    } // simulate_ahead


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
     * @brief Compute C_{s,k} matrix for a single destination s
     *
     * c_{s,k,j,t} = sum_{l=j}^{t-1} alpha_{s,k,l} * h'(r_{k,l}) * phi_{t-l} * y_{k,l}
     *
     * Uses recurrence: c_{s,k,j+1,t} = c_{s,k,j,t} - alpha_{s,k,j} * a_j * phi_{t-j}
     * where a_l = h'(r_{k,l}) * y_{k,l}
     *
     * @param s Destination location index
     * @param k Source location index
     * @param u_k Unnormalized weights for source k (ns x 1)
     * @param a_vec Precomputed h'(r_{k,l}) * y_{k,l} for l = 1..T-1
     * @return Lower triangular matrix C_{s,k} of size (T_obs x T_obs)
     */
    Eigen::MatrixXd compute_C_sk(
        const Eigen::Index &s,
        const Eigen::Index &k,
        const Eigen::VectorXd &u_k,
        const Eigen::VectorXd &a_vec) const
    {
        const Eigen::Index T_obs = nt - 1;
        const Eigen::Index L = static_cast<Eigen::Index>(dlag.Fphi.size());

        Eigen::MatrixXd C_sk = Eigen::MatrixXd::Zero(T_obs, T_obs);

        for (Eigen::Index t = 2; t <= nt; t++)
        {
            Eigen::Index row = t - 2;
            Eigen::Index l_start = std::max(Eigen::Index(1), t - L);

            // First column (j = 1): c_{s,k,1,t} = sum_{l=1}^{t-1} alpha_{s,k,l} * a_l * phi_{t-l}
            double c_val = 0.0;
            for (Eigen::Index l = l_start; l < t; l++)
            {
                Eigen::Index lag = t - l;
                double alpha_skl = spatial.compute_alpha_elem(u_k, s, k, l);
                c_val += alpha_skl * a_vec(l - 1) * dlag.Fphi(lag - 1);
            }
            C_sk(row, 0) = c_val;

            // Remaining columns via recurrence:
            // c_{s,k,j+1,t} = c_{s,k,j,t} - alpha_{s,k,j} * a_j * phi_{t-j}
            for (Eigen::Index j = 2; j < t; j++)
            {
                Eigen::Index col = j - 1;
                Eigen::Index lag = t - (j - 1); // = t - j + 1

                double subtract_term = 0.0;
                if (lag >= 1 && lag <= L)
                {
                    double alpha_sk_jm1 = spatial.compute_alpha_elem(u_k, s, k, j - 1);
                    subtract_term = alpha_sk_jm1 * a_vec(j - 2) * dlag.Fphi(lag - 1);
                }
                C_sk(row, col) = C_sk(row, col - 1) - subtract_term;
            }
        }

        return C_sk;
    } // compute_C_sk


    /**
     * @brief Compute lambda_{s,t} for all destinations and times
     *
     * lambda_{s,t} = mu + sum_{kp} sum_{l=1}^{t-1} alpha_{s,kp,l} * R_{kp,l} * phi_{t-l} * y_{kp,l}
     *
     * @param Y Observations (nt+1) x ns
     * @param u_mat Precomputed unnormalized weights (ns x ns)
     * @param R_mat R values (nt+1) x ns
     * @return lambda_mat of size (T_obs x ns) for t = 2, ..., nt
     */
    Eigen::MatrixXd compute_lambda_mat(
        const Eigen::MatrixXd &Y,
        const Eigen::MatrixXd &u_mat,
        const Eigen::MatrixXd &R_mat) const
    {
        const Eigen::Index T_obs = nt - 1;
        const Eigen::Index L = static_cast<Eigen::Index>(dlag.Fphi.size());

        Eigen::MatrixXd lambda_mat(T_obs, ns);
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
                    Eigen::VectorXd u_kp = u_mat.col(kp);

                    for (Eigen::Index s = 0; s < ns; s++)
                    {
                        double alpha_skpl = spatial.compute_alpha_elem(u_kp, s, kp, l);
                        lambda_mat(row, s) += alpha_skpl * contrib;
                    }
                }
            }
        }

        return lambda_mat.cwiseMax(EPS);
    } // compute_lambda_mat


    /**
     * @brief Compute C_{s,k} * eta for lower triangular C_{s,k}
     */
    Eigen::VectorXd compute_C_eta_product(
        const Eigen::MatrixXd &C_sk,
        const Eigen::VectorXd &eta) const
    {
        const Eigen::Index T_obs = C_sk.rows();
        Eigen::VectorXd C_eta = Eigen::VectorXd::Zero(T_obs);

        for (Eigen::Index row = 0; row < T_obs; row++)
        {
            for (Eigen::Index col = 0; col <= row; col++)
            {
                C_eta(row) += C_sk(row, col) * eta(col);
            }
        }

        return C_eta;
    } // compute_C_eta_product


    /**
     * @brief Compute MH proposal for block update of eta_k with time-varying alpha
     *        Memory-efficient version: computes C_{s,k} per-destination
     *
     * Outputs:
     *   - proposal: BlockMHProposal with precision, mean, Cholesky, log_det
     *   - mu_vec: (T_obs x ns) offset matrix
     *   - lambda_mat: (T_obs x ns) intensity matrix (passed in, used for likelihood)
     *
     * Precision: Omega_k = I + W * sum_s C_{s,k}' * Lambda_s^{-1} * C_{s,k}
     * Mean: u_k = sqrt(W) * Omega_k^{-1} * sum_s C_{s,k}' * Lambda_s^{-1} * d_s
     */
    void compute_mh_proposal(
        BlockMHProposal &proposal,
        Eigen::MatrixXd &mu_vec,
        const Eigen::MatrixXd &lambda_mat,
        const Eigen::Index &k,
        const Eigen::MatrixXd &Y,
        const Eigen::VectorXd &u_k,
        const Eigen::VectorXd &a_vec,
        const Eigen::VectorXd &eta_k_sub)
    {
        const Eigen::Index T_obs = nt - 1;
        const double W = temporal.W + EPS;
        const double W_sqrt = std::sqrt(W);

        // Initialize precision as identity and canonical mean as zero
        proposal.prec = Eigen::MatrixXd::Identity(T_obs, T_obs);
        Eigen::VectorXd canonical_mean = Eigen::VectorXd::Zero(T_obs);
        mu_vec.resize(T_obs, ns);

        // Process one destination at a time (memory efficient)
        for (Eigen::Index s = 0; s < ns; s++)
        {
            // Step 1: Compute C_{s,k} for this destination
            Eigen::MatrixXd C_sk = compute_C_sk(s, k, u_k, a_vec);

            // Step 2: Compute mu_{s,t} = lambda_{s,t} - sqrt(W) * (C_{s,k} * eta_k)_t
            Eigen::VectorXd C_eta = compute_C_eta_product(C_sk, eta_k_sub);
            for (Eigen::Index row = 0; row < T_obs; row++)
            {
                mu_vec(row, s) = std::max(lambda_mat(row, s) - W_sqrt * C_eta(row), EPS);
            }

            // Step 3: Compute d_s = y_s - mu_s and lambda^{-1}
            Eigen::VectorXd d_s = Y.col(s).segment(2, T_obs) - mu_vec.col(s);
            Eigen::VectorXd lambda_inv = lambda_mat.col(s).cwiseInverse();

            // Step 4: Accumulate canonical mean: += sqrt(W) * C_{s,k}' * (lambda_inv .* d_s)
            Eigen::VectorXd weighted_d = lambda_inv.cwiseProduct(d_s);
            canonical_mean.noalias() += W_sqrt * (C_sk.transpose() * weighted_d);

            // Step 5: Accumulate precision: += W * C_{s,k}' * diag(lambda_inv) * C_{s,k}
            //         Using: C_scaled = diag(sqrt(lambda_inv)) * C_{s,k}
            //                prec += W * C_scaled' * C_scaled
            Eigen::MatrixXd C_scaled = lambda_inv.cwiseSqrt().asDiagonal() * C_sk;
            proposal.prec.noalias() += W * (C_scaled.transpose() * C_scaled);

            // C_sk goes out of scope here - memory freed
        }

        // Cholesky decomposition of precision
        proposal.chol.compute(proposal.prec);
        proposal.valid = (proposal.chol.info() == Eigen::Success);

        if (proposal.valid)
        {
            // Log determinant: log|Omega| = 2 * sum(log(diag(L)))
            Eigen::VectorXd chol_diag = proposal.chol.matrixL().toDenseMatrix().diagonal();
            proposal.log_det_prec = 2.0 * chol_diag.array().log().sum();

            // Solve for mean: Omega * u = canonical_mean => u = Omega^{-1} * canonical_mean
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
     * @brief Recompute lambda given new eta values with time-varying alpha
     *        Recomputes C_{s,k} per-destination (same linearization point as proposal)
     *
     * lambda_{s,t}^{new} = mu_{s,t} + sqrt(W) * (C_{s,k} * eta^{new})_t
     *
     * @param lambda_new Output: (T_obs x ns) new intensity matrix
     * @param mu_vec Offset matrix (T_obs x ns) computed at linearization point
     * @param k Source location index
     * @param u_k Unnormalized weights for source k
     * @param a_vec Precomputed h'(r_{k,l}) * y_{k,l} at linearization point
     * @param eta_new New eta values (T_obs x 1)
     */
    void recompute_lambda(
        Eigen::MatrixXd &lambda_new,
        const Eigen::MatrixXd &mu_vec,
        const Eigen::Index &k,
        const Eigen::VectorXd &u_k,
        const Eigen::VectorXd &a_vec,
        const Eigen::VectorXd &eta_new)
    {
        const Eigen::Index T_obs = nt - 1;
        const double W_sqrt = std::sqrt(temporal.W + EPS);

        lambda_new.resize(T_obs, ns);

        for (Eigen::Index s = 0; s < ns; s++)
        {
            // Recompute C_{s,k} (same linearization point, so same a_vec)
            Eigen::MatrixXd C_sk = compute_C_sk(s, k, u_k, a_vec);

            // Compute C_{s,k} * eta_new
            Eigen::VectorXd C_eta = compute_C_eta_product(C_sk, eta_new);

            // lambda_{s,t} = mu_{s,t} + sqrt(W) * (C_{s,k} * eta_new)_t
            for (Eigen::Index row = 0; row < T_obs; row++)
            {
                lambda_new(row, s) = std::max(mu_vec(row, s) + W_sqrt * C_eta(row), EPS);
            }
        }
    } // recompute_lambda

    /**
     * @brief Block MH update for eta_k with time-varying spatial weights
     *        Memory-efficient version using per-destination C_{s,k} computation
     *
     * Algorithm:
     *   1. Compute lambda at current state
     *   2. Compute proposal at current linearization point, sample eta_new
     *   3. Recompute lambda at proposed eta (using same linearization)
     *   4. Compute reverse proposal at new linearization point
     *   5. MH accept/reject
     *
     * @param Y Observations (nt+1) x ns
     * @param k Source location index
     * @param mh_sd Proposal scaling factor (default 1.0)
     * @return Acceptance indicator (0.0 or 1.0)
     */
    double update_eta_block_mh(
        const Eigen::MatrixXd &Y,
        const Eigen::Index &k,
        const double &mh_sd = 1.0)
    {
        const Eigen::Index T_obs = nt - 1;
        const double W_sqrt = std::sqrt(temporal.W + EPS);

        // =========================================================
        // Step 0: Precompute shared quantities
        // =========================================================

        // Convert centered wt to noncentered eta
        Eigen::MatrixXd eta = temporal.wt / W_sqrt;

        // Precompute cumulative sums for r = sqrt(W) * cumsum(eta)
        Eigen::MatrixXd eta_cumsum(nt + 1, ns);
        for (Eigen::Index kp = 0; kp < ns; kp++)
        {
            eta_cumsum.col(kp) = cumsum_vec(eta.col(kp));
        }

        // Precompute u_mat, R_mat, dR_mat
        Eigen::MatrixXd u_mat(ns, ns);
        Eigen::MatrixXd r_mat = W_sqrt * eta_cumsum;
        Eigen::MatrixXd R_mat(nt + 1, ns);
        Eigen::MatrixXd dR_mat(nt + 1, ns);
        for (Eigen::Index kp = 0; kp < ns; kp++)
        {
            u_mat.col(kp) = spatial.compute_unnormalized_weight_col(kp);
            for (Eigen::Index l = 0; l < nt + 1; l++)
            {
                R_mat(l, kp) = GainFunc::psi2hpsi(r_mat(l, kp), temporal.fgain);
                dR_mat(l, kp) = GainFunc::psi2dhpsi(r_mat(l, kp), temporal.fgain);
            }
        }

        // Current eta_k (indices j = 1, ..., T-1)
        Eigen::VectorXd eta_old = eta.col(k).segment(1, T_obs);

        // Precompute a_vec = h'(r_{k,l}) * y_{k,l} for l = 1..T-1 (at current state)
        Eigen::VectorXd a_vec_old(T_obs);
        for (Eigen::Index l = 1; l < nt; l++)
        {
            a_vec_old(l - 1) = dR_mat(l, k) * Y(l, k);
        }

        // Unnormalized weights for source k
        Eigen::VectorXd u_k = u_mat.col(k);

        // =========================================================
        // Step 1: Compute lambda at current state
        // =========================================================
        Eigen::MatrixXd lambda_mat_old = compute_lambda_mat(Y, u_mat, R_mat);

        // =========================================================
        // Step 2: Compute proposal at current state and sample
        // =========================================================
        BlockMHProposal proposal_old;
        Eigen::MatrixXd mu_vec_old;
        compute_mh_proposal(
            proposal_old, mu_vec_old, lambda_mat_old,
            k, Y, u_k, a_vec_old, eta_old
        );

        if (!proposal_old.valid)
        {
            return 0.0; // Reject if Cholesky failed
        }

        Eigen::VectorXd eta_new = proposal_old.sample(mh_sd);

        // =========================================================
        // Step 3: Recompute lambda at proposed eta
        //         (using same linearization point, i.e., same a_vec_old)
        // =========================================================
        Eigen::MatrixXd lambda_mat_new;
        recompute_lambda(
            lambda_mat_new, mu_vec_old,
            k, u_k, a_vec_old, eta_new
        );

        // Check for numerical issues
        if ((lambda_mat_new.array() <= 0).any() || !lambda_mat_new.allFinite())
        {
            return 0.0;
        }

        // =========================================================
        // Step 4: Compute reverse proposal (linearized at new state)
        // =========================================================

        // Temporarily update wt to compute R_mat and dR_mat at new state
        Eigen::VectorXd wt_old = temporal.wt.col(k);
        for (Eigen::Index j = 1; j < nt; j++)
        {
            temporal.wt(j, k) = W_sqrt * eta_new(j - 1);
        }

        // Recompute R_mat and dR_mat at new state (only for column k changes)
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

        // Precompute a_vec at new state
        Eigen::VectorXd a_vec_new(T_obs);
        for (Eigen::Index l = 1; l < nt; l++)
        {
            a_vec_new(l - 1) = dR_mat_new(l, k) * Y(l, k);
        }

        // Compute lambda at new state (for reverse proposal)
        Eigen::MatrixXd lambda_mat_new_full = compute_lambda_mat(Y, u_mat, R_mat_new);

        // Compute reverse proposal
        BlockMHProposal proposal_new;
        Eigen::MatrixXd mu_vec_new;
        compute_mh_proposal(
            proposal_new, mu_vec_new, lambda_mat_new_full,
            k, Y, u_k, a_vec_new, eta_new
        );

        // Restore wt
        temporal.wt.col(k) = wt_old;

        if (!proposal_new.valid)
        {
            return 0.0;
        }

        // =========================================================
        // Step 5: Compute MH acceptance ratio
        // =========================================================

        // log p(eta_new) - log p(eta_old) [prior: eta ~ N(0, I)]
        double log_prior_ratio = log_prior_eta(eta_new) - log_prior_eta(eta_old);

        // log p(y | eta_new) - log p(y | eta_old) [Poisson likelihood]
        double log_lik_new = log_likelihood_poisson(Y, lambda_mat_new);
        double log_lik_old = log_likelihood_poisson(Y, lambda_mat_old);
        double log_lik_ratio = log_lik_new - log_lik_old;

        // log q(eta_old | eta_new) - log q(eta_new | eta_old) [proposal ratio]
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
        const Rcpp::Nullable<Rcpp::List> &mcmc_opts = R_NilValue
    )
    {
        const Eigen::Index ntotal = static_cast<Eigen::Index>(nburnin + nsamples * nthin);
        initialize_wt_from_data(Y);
        spatial.rho_dist = R::runif(0.0, 2.0);
        spatial.rho_mobility = R::runif(0.0, 2.0);
        spatial.initialize_horseshoe_sparse();
        zinfl = ZeroInflation(Y, false); // initialize zero-inflation model

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
                    }
                    if (init_values.containsElementNamed("rho_mobility"))
                    {
                        spatial.rho_mobility = Rcpp::as<double>(init_values["rho_mobility"]);
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
                    if (init_values.containsElementNamed("theta"))
                    {
                        spatial.theta = Rcpp::as<Eigen::MatrixXd>(init_values["theta"]);
                    }
                    if (init_values.containsElementNamed("gamma"))
                    {
                        spatial.gamma = Rcpp::as<Eigen::MatrixXd>(init_values["gamma"]);
                    }
                    if (init_values.containsElementNamed("delta"))
                    {
                        spatial.delta = Rcpp::as<Eigen::MatrixXd>(init_values["delta"]);
                    }
                    if (init_values.containsElementNamed("tau"))
                    {
                        spatial.tau = Rcpp::as<Eigen::VectorXd>(init_values["tau"]);
                    }
                    if (init_values.containsElementNamed("zeta"))
                    {
                        spatial.zeta = Rcpp::as<Eigen::VectorXd>(init_values["zeta"]);
                    }
                    if (init_values.containsElementNamed("logit_wdiag_intercept"))
                    {
                        spatial.logit_wdiag_intercept = Rcpp::as<Eigen::VectorXd>(init_values["logit_wdiag_intercept"]);
                    }
                    if (init_values.containsElementNamed("logit_wdiag_slope"))
                    {
                        spatial.logit_wdiag_slope = Rcpp::as<Eigen::VectorXd>(init_values["logit_wdiag_slope"]);
                    }

                } // if init

                if (ns == 1)
                {
                    hs_prior.infer = false; // No horseshoe if only one location
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
        

        // Set up HMC options and diagnostics for static parameters if to be inferred
        Eigen::VectorXi idx_static_params;
        Eigen::Index n_static_params = get_ndim_static_params(idx_static_params, hmc_include_W);
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

            Eigen::VectorXd mass_init = get_unconstrained_static_params(
                n_static_params, 
                idx_static_params
            );
            mass_adapter.mean = mass_init.matrix();
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
                n_colvec = get_ndim_colvec(idx_colvec, idx_inverse, k);
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

        Eigen::Tensor<double, 3> theta_samples, gamma_samples, delta_samples;
        Eigen::MatrixXd tau_samples, zeta_samples, logit_wdiag_intercept_samples, logit_wdiag_slope_samples;
        if (hs_prior.infer)
        {
            theta_samples.resize(spatial.theta.rows(), spatial.theta.cols(), nsamples);
            gamma_samples.resize(spatial.gamma.rows(), spatial.gamma.cols(), nsamples);
            delta_samples.resize(spatial.delta.rows(), spatial.delta.cols(), nsamples);
            tau_samples.resize(spatial.tau.size(), nsamples);
            zeta_samples.resize(spatial.zeta.size(), nsamples);
            logit_wdiag_intercept_samples.resize(spatial.logit_wdiag_intercept.size(), nsamples);

            if (spatial.include_covariates)
            {
                logit_wdiag_slope_samples.resize(spatial.logit_wdiag_slope.size(), nsamples);
            }
        }

        Eigen::MatrixXd zinfl_Z_average;
        Eigen::VectorXd zinfl_beta0_samples, zinfl_beta1_samples;
        if (zinfl_prior.infer)
        {
            zinfl_Z_average = Eigen::MatrixXd::Zero(zinfl.Z.rows(), zinfl.Z.cols());
            zinfl_beta0_samples.resize(nsamples);
            zinfl_beta1_samples.resize(nsamples);
        }


        Eigen::VectorXd post_burnin_accept_sum = Eigen::VectorXd::Zero(ns);
        Eigen::VectorXi post_burnin_accept_count = Eigen::VectorXi::Zero(ns);
        const int post_burnin_window = 50; // Check every 50 iterations
        const double min_step_size = 1e-4;
        const double max_step_size = 1.0;

        Progress p(static_cast<unsigned int>(ntotal), true);
        for (Eigen::Index iter = 0; iter < ntotal; iter++)
        {
            if (zinfl_prior.infer)
            {
                Eigen::MatrixXd R_mat = temporal.compute_Rt(); // (nt + 1) x ns
                Eigen::MatrixXd lambda_mat = compute_intensity(Y, R_mat);
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
                        mass_diag_zinfl = mass_adapter_zinfl.get_mass_diag();

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
                        spatial.include_covariates,
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
                double accept_prob = update_static_params(
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
                    theta_samples.chip(sample_idx, 2) = theta_map;

                    Eigen::TensorMap<Eigen::Tensor<const double, 2>> gamma_map(
                        spatial.gamma.data(), 
                        spatial.gamma.rows(), 
                        spatial.gamma.cols()
                    );
                    gamma_samples.chip(sample_idx, 2) = gamma_map;

                    Eigen::TensorMap<Eigen::Tensor<const double, 2>> delta_map(
                        spatial.delta.data(), 
                        spatial.delta.rows(), 
                        spatial.delta.cols()
                    );
                    delta_samples.chip(sample_idx, 2) = delta_map;

                    Eigen::Map<const Eigen::VectorXd> tau_map(spatial.tau.data(), spatial.tau.size());
                    tau_samples.col(sample_idx) = tau_map;

                    Eigen::Map<const Eigen::VectorXd> zeta_map(spatial.zeta.data(), spatial.zeta.size());
                    zeta_samples.col(sample_idx) = zeta_map;

                    Eigen::Map<const Eigen::VectorXd> logit_wdiag_intercept_map(
                        spatial.logit_wdiag_intercept.data(), 
                        spatial.logit_wdiag_intercept.size()
                    );
                    logit_wdiag_intercept_samples.col(sample_idx) = logit_wdiag_intercept_map;

                    Eigen::Map<const Eigen::VectorXd> logit_wdiag_slope_map(
                        spatial.logit_wdiag_slope.data(), 
                        spatial.logit_wdiag_slope.size()
                    );
                    logit_wdiag_slope_samples.col(sample_idx) = logit_wdiag_slope_map;
                }

            } // if store samples

            p.increment();
        } // for MCMC iter

        
        Rcpp::List output;

        // output["alpha"] = tensor3_to_r(alpha_samples); // ns x ns x nsamples

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
                Rcpp::Named("delta") = tensor3_to_r(delta_samples), // ns x ns x nsamples
                Rcpp::Named("tau") = tau_samples, // ns x nsamples
                Rcpp::Named("zeta") = zeta_samples, // ns x nsamples
                Rcpp::Named("logit_wdiag_intercept") = logit_wdiag_intercept_samples  // ns x nsamples
            );

            if (spatial.include_covariates)
            {
                hs_list["logit_wdiag_slope"] = logit_wdiag_slope_samples; // ns x nsamples
            }

            hs_list["hmc"] = Rcpp::List::create(
                Rcpp::Named("acceptance_rate") = hs_hmc_diag.accept_count / static_cast<double>(ntotal),
                Rcpp::Named("leapfrog_step_size") = hs_hmc_opts.leapfrog_step_size,
                Rcpp::Named("n_leapfrog") = hs_hmc_opts.nleapfrog,
                Rcpp::Named("mass_diag") = mass_diag_hs,
                Rcpp::Named("diagnostics") = hs_hmc_diag.to_list()
            );

            output["horseshoe"] = hs_list;
        } // if infer horseshoe theta


        return output;
    } // run_mcmc

}; // class Model



#endif // MODEL_HPP