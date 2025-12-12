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
 * @brief Sparse spatial network controlled by a regularized horseshoe prior.
 * 
 */
class SpatialNetwork
{
private:
    Eigen::Index ns = 0; // number of spatial locations
    double mean_dist = 0.0;
    double mean_log_mobility = 0.0;

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


    /**
     * @brief Compute a column of the spatial network alpha
     * 
     * @param k Eigen::Index, source location index
     * @return Eigen::VectorXd 
     */
    void initialize_horseshoe_random()
    {
        theta.resize(ns, ns);
        gamma.resize(ns, ns);
        delta.resize(ns, ns);
        tau.resize(ns);
        zeta.resize(ns);
        wdiag.resize(ns);

        double zz = 1.0 / R::rgamma(0.5, 1.0);
        double tt = 1.0 / R::rgamma(0.5, 1.0 / zz);

        for (Eigen::Index k = 0; k < ns; k++)
        { // Loop over source locations (columns)
            if (shared_tau)
            {
                tau(k) = tt;
                zeta(k) = zz;
            }
            else
            {
                zeta(k) = 1.0 / R::rgamma(0.5, 1.0);
                tau(k) = 1.0 / R::rgamma(0.5, 1.0 / zeta(k));
            }

            wdiag(k) = R::rbeta(1.0, 1.0); // self-exciting weight
            for (Eigen::Index s = 0; s < ns; s++)
            { // Loop over destination locations (rows)
                delta(s, k) = 1.0 / R::rgamma(0.5, 1.0);
                gamma(s, k) = 1.0 / R::rgamma(0.5, 1.0 / delta(s, k));
                if (s == k)
                {
                    theta(s, k) = 0.0;
                }
                else
                {
                    theta(s, k) = R::rnorm(0.0, std::sqrt(tau(k) * gamma(s, k)));
                }
            } // for destination s given source k

            double th_mean = theta.col(k).array().sum() / static_cast<double>(ns - 1);
            theta.col(k) = theta.col(k).array() - th_mean; // center theta column
            theta(k, k) = 0.0; // ensure diagonal is zero
        } // for source k
    } // initialize_horseshoe_random


    void initialize_horseshoe_zero()
    {
        theta.resize(ns, ns);
        gamma.resize(ns, ns);
        delta.resize(ns, ns);
        tau.resize(ns);
        zeta.resize(ns);
        wdiag.resize(ns);

        theta.setZero();
        gamma.setOnes();
        delta.setOnes();
        tau.setOnes();
        zeta.setOnes();
        wdiag.setOnes();
    } // initialize_horseshoe_zero


public:
    Eigen::MatrixXd dist; // ns x ns, pairwise scaled distance matrix
    Eigen::MatrixXd log_mobility; // ns x ns, pairwise scaled mobility matrix

    Eigen::MatrixXd alpha; // ns x ns, adjacency matrix / stochastic network
    double rho_dist = 1.0; // distance decay parameter
    double rho_mobility = 1.0; // mobility scaling parameter

    bool shared_tau = true; // If all columns share the same tau
    Eigen::MatrixXd theta; // ns x ns, horseshoe component for spatial network
    Eigen::MatrixXd gamma; // ns x ns, horseshoe element-wise variance
    Eigen::MatrixXd delta; // ns x ns, horseshoe local shrinkage parameters
    Eigen::VectorXd tau; // ns x 1, horseshoe column-wise variance
    Eigen::VectorXd zeta; // ns x 1, horseshoe column-wise local shrinkage parameters
    Eigen::VectorXd wdiag; // ns x 1, self-exciting weight per location

    // Slab (regularized horseshoe) hyperparameters
    double slab_c2; // c^2 (slab variance)
    double slab_df; // a_c (degrees of freedom parameter)
    double slab_scale; // s (prior scale for slab std dev)

    SpatialNetwork()
    {
        ns = 1;
        shared_tau = true;

        initialize_horseshoe_zero();

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
        const bool &shared_tau_in = true,
        const bool &random_init = false
    )
    {
        shared_tau = shared_tau_in;

        dist = dist_scaled_in;
        log_mobility = mobility_scaled_in.array().max(EPS8).log().matrix();
        ns = dist.rows();

        mean_dist = dist.mean();
        mean_log_mobility = log_mobility.mean();
        dist = dist.array() - mean_dist;
        log_mobility = log_mobility.array() - mean_log_mobility;

        if (random_init)
        {
            rho_dist = R::runif(0.0, 2.0);
            rho_mobility = R::runif(0.0, 2.0);
            initialize_horseshoe_random();
        }
        else
        {
            rho_dist = 1.0;
            rho_mobility = 1.0;
            initialize_horseshoe_zero();
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
        dist = dist_scaled_in;
        log_mobility = mobility_scaled_in.array().max(EPS8).log().matrix();
        ns = dist.rows();

        shared_tau = true;
        if (settings.containsElementNamed("shared_tau"))
        {
            shared_tau = Rcpp::as<bool>(settings["shared_tau"]);
        } // if shared_tau
        if (settings.containsElementNamed("rho_dist"))
        {
            rho_dist = Rcpp::as<double>(settings["rho_dist"]);
        } // if rho_dist
        if (settings.containsElementNamed("rho_mobility"))
        {
            rho_mobility = Rcpp::as<double>(settings["rho_mobility"]);
        } // if rho_mobility


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
                double th_mean = theta.col(k).array().sum() / static_cast<double>(ns - 1);
                theta.col(k) = theta.col(k).array() - th_mean; // center theta column
                theta(k, k) = 0.0; // ensure diagonal is zero
            }
        }
        else
        {
            initialize_horseshoe_random();
        }

        mean_dist = dist.mean();
        mean_log_mobility = log_mobility.mean();
        dist = dist.array() - mean_dist;
        log_mobility = log_mobility.array() - mean_log_mobility;

        alpha.resize(ns, ns);
        for (Eigen::Index k = 0; k < ns; k++)
        { // Loop over source locations
            alpha.col(k) = compute_alpha_col(k);
        }
        return;
    } // SpatialNetwork constructor


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
            Eigen::VectorXd u_k(ns); // unnormalized weights
            u_k.setZero();
            for (Eigen::Index s = 0; s < ns; s++)
            { // Loop over destinations
                u_k(s) = compute_unnormalized_weight(s, k);
            } // for s

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
    }


    /**
     * @brief Calculate unnormalized weight for location pair (s, k) for s != k
     * 
     * @param s 
     * @param k 
     * @return double 
     */
    double compute_unnormalized_weight(
        const Eigen::Index &s,
        const Eigen::Index &k
    ) const
    {
        if (s != k)
        {
            double v_sk = -rho_dist * dist(s, k) +
                          rho_mobility * log_mobility(s, k) +
                          theta(s, k);

            return std::exp(std::min(v_sk, UPBND));
        }
        else
        {
            return 0.0;
        }
    } // compute_unnormalized_weight


    double dalpha_drho_dist(
        const Eigen::Index &s,
        const Eigen::Index &k, // source location index
        const Eigen::VectorXd &u_k // unnormalized weights for source k
    ) const
    {
        if (s != k)
        {
            double U_k = u_k.array().sum(); // u_k(k) == 0
            double deriv = -dist(s, k) * U_k + dist.col(k).dot(u_k);
            deriv *= u_k(s) / (U_k * U_k);
            deriv *= 1.0 - wdiag(k);
            return deriv;
        }
        else
        {
            // alpha[k, k] does not depend on rho_dist
            return 0.0;
        }
    } // dalpha_drho_dist


    double dalpha_drho_mobility(
        const Eigen::Index &s,
        const Eigen::Index &k,
        const Eigen::VectorXd &u_k // unnormalized weights for source k
    ) const
    {
        if (s != k)
        {
            double U_k = u_k.array().sum(); // u_k(k) == 0
            double mean_log_mobility = (log_mobility.col(k).array() * u_k.array()).matrix().sum();
            double deriv = log_mobility(s, k) * U_k - mean_log_mobility;
            deriv *= u_k(s) / (U_k * U_k);
            deriv *= 1.0 - wdiag(k);
            return deriv;
        }
        else
        {
            // alpha[k, k] does not depend on rho_mobility
            return 0.0;
        }
    } // dalpha_drho_mobility


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


    void update_horseshoe_params()
    {
        // Update gamma, delta, tau, zeta, given theta
        double tau_rate_shared = 0.0;
        double zeta_rate_shared = 0.0;
        Eigen::VectorXd tau_rate(ns);
        Eigen::VectorXd zeta_rate(ns);
        tau_rate.setZero();
        zeta_rate.setZero();

        for (Eigen::Index k = 0; k < ns; k++)
        {
            tau_rate(k) = 0.0;
            for (Eigen::Index s = 0; s < ns; s++)
            {
                if (s == k)
                {
                    continue; // skip diagonal
                }

                const double th2 = theta(s, k) * theta(s, k);

                double gm_rate = 1.0 / delta(s, k) + 0.5 * th2 / tau(k);
                gamma(s, k) = 1.0 / R::rgamma(1.0, 1.0 / gm_rate);

                double dl_rate = 1.0 + 1.0 / gamma(s, k);
                delta(s, k) = 1.0 / R::rgamma(1.0, 1.0 / dl_rate);

                tau_rate(k) += th2 / gamma(s, k);
            } // for s

            if (shared_tau)
            {
                tau_rate_shared += tau_rate(k);
            }
            else
            {
                tau_rate(k) *= 0.5;
                tau_rate(k) += 1.0 / zeta(k);
                tau(k) = 1.0 / R::rgamma(0.5 * (ns - 1) + 0.5, 1.0 / tau_rate(k));
                zeta(k) = 1.0 / R::rgamma(1.0, 1.0 / (1.0 + 1.0 / tau(k)));
            }
        } // for k

        if (shared_tau)
        {
            tau_rate_shared *= 0.5;
            tau_rate_shared += 1.0 / zeta(0);
            double tau_shape_shared = 0.5 * ns * (ns - 1) + 0.5;
            double tau_shared = 1.0 / R::rgamma(tau_shape_shared, 1.0 / tau_rate_shared);
            double zeta_shared = 1.0 / R::rgamma(1.0, 1.0 / (1.0 + 1.0 / tau_shared));
            for (Eigen::Index k = 0; k < ns; k++)
            {
                tau(k) = tau_shared;
                zeta(k) = zeta_shared;
            } // for k
        } // if shared_tau
    } // update_horseshoe_params

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
        ),
        const bool &shared_tau = true
    )
    {
        ns = dist_matrix.rows();
        nt = Y.rows() - 1;

        dlag = LagDist(lagdist_opts);
        temporal = TemporalTransmission(ns, nt, fgain_);
        spatial = SpatialNetwork(dist_matrix, mobility_matrix, shared_tau);

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


    Eigen::VectorXd dloglike_dtheta_col(
        double &loglike,
        const Eigen::Index &k, // column index (source location) of theta to compute derivative for
        const Eigen::MatrixXd &Y, // (nt + 1) x ns, observed primary infections
        const Eigen::MatrixXd &R_mat // (nt + 1) x ns
    )
    {
        Eigen::MatrixXd u_mat(ns, ns);
        for (Eigen::Index s = 0; s < ns; s++)
        {
            for (Eigen::Index k = 0; k < ns; k++)
            {
                u_mat(s, k) = spatial.compute_unnormalized_weight(s, k);
            } // for source location k
        } // for destination location s

        double W_safe = std::max(temporal.W, EPS);
        double W_sqrt = std::sqrt(W_safe);

        Eigen::MatrixXd lambda_mat(nt + 1, ns);
        Eigen::MatrixXd dloglike_dlambda_mat(nt + 1, ns);
        loglike = 0.0;
        for (Eigen::Index s = 0; s < ns; s++)
        { // for destination location s
            for (Eigen::Index t = 0; t < nt + 1; t++)
            { // for destination time t
                double lambda_st = mu;
                for (Eigen::Index kk = 0; kk < ns; kk++)
                {
                    for (Eigen::Index l = 0; l < t; l++)
                    {
                        if (t - l <= static_cast<Eigen::Index>(dlag.Fphi.size()))
                        {
                            double coef = R_mat(l, kk) * dlag.Fphi(t - l - 1) * Y(l, kk);
                            lambda_st += spatial.alpha(s, kk) * coef;
                        }
                    } // for source time l < t
                } // for source location k

                lambda_mat(t, s) = std::max(lambda_st, EPS);
                dloglike_dlambda_mat(t, s) = Y(t, s) / lambda_mat(t, s) - 1.0;
                loglike += Y(t, s) * std::log(lambda_mat(t, s)) - lambda_mat(t, s);
            } // for time t
        } // for destination location s

        Eigen::MatrixXd dlambda_st_dalpha_sk(nt + 1, ns);
        dlambda_st_dalpha_sk.setZero();
        for (Eigen::Index s = 0; s < ns; s++)
        { // for destination location s
            for (Eigen::Index t = 0; t < nt + 1; t++)
            { // for destination time t
                for (Eigen::Index l = 0; l < t; l++)
                {
                    if (t - l <= static_cast<Eigen::Index>(dlag.Fphi.size()))
                    {
                        dlambda_st_dalpha_sk(t, s) += R_mat(l, k) * dlag.Fphi(t - l - 1) * Y(l, k);
                    }
                } // for source time l < t
            } // for time t
        } // for destination location s

        Eigen::VectorXd grad(ns);
        grad.setZero();
        for (Eigen::Index j = 0; j < ns; j++)
        { // for destination location j, calculate grad(j): derivative w.r.t. theta[j, k]
            for (Eigen::Index s = 0; s < ns; s++)
            { // loop over destination location s
                double dalpha_sk_dtheta_jk = spatial.dalpha_dtheta(s, j, k, u_mat.col(k));
                for (Eigen::Index t = 0; t < nt + 1; t++)
                {
                    grad(j) += dloglike_dlambda_mat(t, s) * dlambda_st_dalpha_sk(t, s) * dalpha_sk_dtheta_jk;
                }
            } // for destination location s
        } // for destination location j

        return grad;
    } // dloglike_dtheta_col


    // double update_theta_col(
    //     double &energy_diff,
    //     double &grad_norm,
    //     const Eigen::Index &k, // column index (source location) of theta to update
    //     const Eigen::MatrixXd &Y, // (nt + 1) x ns, observed primary infections
    //     const Eigen::MatrixXd &R_mat, // (nt + 1) x ns
    //     const double &step_size,
    //     const unsigned int &n_leapfrog
    // )
    // {
    //     Eigen::VectorXd prior_var = spatial.tau(k) * spatial.gamma.col(k); // ns x 1
    //     Eigen::VectorXd mass_diag(ns);
    //     mass_diag.setOnes();
    //     Eigen::VectorXd inv_mass = mass_diag.cwiseInverse();
    //     Eigen::VectorXd sqrt_mass = mass_diag.array().sqrt();

    //     // Current state
    //     Eigen::VectorXd theta_current = spatial.theta.col(k); // ns x 1
    //     double loglike = 0.0;
    //     Eigen::VectorXd grad = dloglike_dtheta_col(loglike, k, Y, R_mat);
    //     grad += - theta_current.cwiseQuotient(prior_var); // prior contribution
    //     grad_norm = grad.norm();
    //     double current_logprior = - 0.5 * theta_current.cwiseProduct(theta_current.cwiseQuotient(prior_var)).sum();
    //     double current_energy = - (loglike + current_logprior);

    //     // Sample momentum
    //     Eigen::VectorXd momentum(ns);
    //     for (Eigen::Index i = 0; i < momentum.size(); i++)
    //     {
    //         momentum(i) = sqrt_mass(i) * R::rnorm(0.0, 1.0);
    //     }
    //     double current_kinetic = 0.5 * momentum.cwiseProduct(inv_mass).dot(momentum);

    //     // Leapfrog integration
    //     momentum += 0.5 * step_size * grad;
    //     for (unsigned int lf_step = 0; lf_step < n_leapfrog; lf_step++)
    //     {
    //         // Update theta
    //         spatial.theta.col(k) += step_size * momentum.cwiseProduct(inv_mass);
    //         spatial.alpha.col(k) = spatial.compute_alpha_col(k); // update alpha after theta is updated

    //         // Compute new gradient
    //         grad = dloglike_dtheta_col(loglike, k, Y, R_mat);
    //         grad += - spatial.theta.col(k).cwiseQuotient(prior_var); // prior contribution

    //         // Update momentum
    //         if (lf_step != n_leapfrog - 1)
    //         {
    //             momentum += step_size * grad;
    //         }
    //     } // for leapfrog steps

    //     momentum += 0.5 * step_size * grad;
    //     momentum = -momentum; // Negate momentum to make proposal symmetric

    //     double proposed_logprior = - 0.5 * spatial.theta.col(k).cwiseProduct(spatial.theta.col(k).cwiseQuotient(prior_var)).sum();
    //     double proposed_energy = - (loglike + proposed_logprior);
    //     double proposed_kinetic = 0.5 * momentum.cwiseProduct(inv_mass).dot(momentum);

    //     double H_proposed = proposed_energy + proposed_kinetic;
    //     double H_current = current_energy + current_kinetic;
    //     energy_diff = H_proposed - H_current;

    //     // Metropolis-Hastings acceptance step
    //     double accept_prob;
    //     if (std::isfinite(energy_diff) && std::abs(energy_diff) < 100.0 && std::log(R::runif(0.0, 1.0)) < -energy_diff)
    //     {
    //         // accept proposed state
    //         accept_prob = std::min(1.0, std::exp(-energy_diff));
    //     }
    //     else
    //     {
    //         accept_prob = 0.0;

    //         // Revert to current state
    //         spatial.theta.col(k) = theta_current;
    //         spatial.alpha.col(k) = spatial.compute_alpha_col(k);
    //     } // end Metropolis step

    //     return accept_prob;
    // } // update_theta_col

    double update_theta_col(
        double &energy_diff,
        double &grad_norm,
        const Eigen::Index &k,
        const Eigen::MatrixXd &Y,
        const Eigen::MatrixXd &R_mat,
        const double &step_size,
        const unsigned int &n_leapfrog)
    {
        const Eigen::Index S = ns; // number of locations
        // Build list of off-diagonal indices in column k
        std::vector<Eigen::Index> off_idx;
        off_idx.reserve(S - 1);
        for (Eigen::Index s = 0; s < S; ++s)
        {
            if (s == k)
                continue;
            off_idx.push_back(s);
        }
        // Choose reference index r among off-diagonals
        const Eigen::Index r = off_idx[0];

        // Free dimension m = (S-1) - 1 = S-2
        const Eigen::Index m = static_cast<Eigen::Index>(off_idx.size()) - 1;

        // Extract current theta column
        Eigen::VectorXd theta_col = spatial.theta.col(k);
        // Make sure constraint holds (re-center off-diagonals just in case)
        double sum_off = 0.0;
        for (Eigen::Index idx = 0; idx < off_idx.size(); ++idx)
        {
            sum_off += theta_col(off_idx[idx]);
        }
        double mean_off = sum_off / static_cast<double>(off_idx.size());
        for (Eigen::Index idx = 0; idx < off_idx.size(); ++idx)
        {
            theta_col(off_idx[idx]) -= mean_off;
        }
        theta_col(k) = 0.0;

        // Build free vector z from theta_col (drop the reference r)
        Eigen::VectorXd z(m);
        {
            Eigen::Index j = 0;
            for (Eigen::Index idx = 0; idx < off_idx.size(); ++idx)
            {
                Eigen::Index s = off_idx[idx];
                if (s == r)
                    continue;
                z(j++) = theta_col(s);
            }
        }

        // Helper lambdas: map z -> theta_col and compute logpost + grad_z
        auto set_theta_from_z = [&](const Eigen::VectorXd &z_in,
                                    Eigen::VectorXd &theta_out)
        {
            // Fill off-diagonals except r from z
            double sum_except_r = 0.0;
            Eigen::Index j = 0;
            for (Eigen::Index idx = 0; idx < off_idx.size(); ++idx)
            {
                Eigen::Index s = off_idx[idx];
                if (s == r)
                    continue;
                double val = z_in(j++);
                theta_out(s) = val;
                sum_except_r += val;
            }
            // Reference entry chosen to enforce sum zero
            theta_out(r) = -sum_except_r;
            // Diagonal fixed at zero
            theta_out(k) = 0.0;
        };

        auto eval_logpost_and_grad_z =
            [&](const Eigen::VectorXd &z_in,
                double &logpost,
                Eigen::VectorXd &grad_z)
        {
            // 1) Map z -> theta_col
            set_theta_from_z(z_in, theta_col);
            spatial.theta.col(k) = theta_col;
            spatial.alpha.col(k) = spatial.compute_alpha_col(k); // update alpha

            // 2) Likelihood gradient w.r.t. theta (existing code)
            double loglike = 0.0;
            Eigen::VectorXd grad_theta = dloglike_dtheta_col(loglike, k, Y, R_mat);

            // 3) Add horseshoe prior gradient w.r.t. theta
            Eigen::VectorXd prior_var = spatial.tau(k) * spatial.gamma.col(k); // ns
            for (Eigen::Index s = 0; s < S; ++s)
            {
                if (s == k)
                {
                    // diagonal has no horseshoe prior in practice; force 0
                    grad_theta(s) += 0.0;
                }
                else
                {
                    grad_theta(s) += -theta_col(s) / std::max(prior_var(s), EPS);
                }
            }
            double logprior = 0.0;
            for (Eigen::Index s = 0; s < S; ++s)
            {
                if (s == k)
                    continue;
                double v = theta_col(s);
                double var = std::max(prior_var(s), EPS);
                logprior += -0.5 * v * v / var;
            }

            // 4) Chain rule: grad_z = B^T * grad_theta
            // grad_z_j = grad_theta(s_j) - grad_theta(r)
            grad_z.setZero(m);
            {
                Eigen::Index j = 0;
                for (Eigen::Index idx = 0; idx < off_idx.size(); ++idx)
                {
                    Eigen::Index s = off_idx[idx];
                    if (s == r)
                        continue;
                    grad_z(j++) = grad_theta(s) - grad_theta(r);
                }
            }

            grad_norm = grad_z.norm();
            logpost = loglike + logprior;
        };

        // Mass matrix for z
        Eigen::VectorXd mass_diag = Eigen::VectorXd::Ones(m);
        Eigen::VectorXd inv_mass = mass_diag.cwiseInverse();
        Eigen::VectorXd sqrt_mass = mass_diag.array().sqrt();

        // Evaluate at current z
        double current_logpost = 0.0;
        Eigen::VectorXd grad_z(m);
        eval_logpost_and_grad_z(z, current_logpost, grad_z);

        double current_energy = -current_logpost;

        // Sample momentum
        Eigen::VectorXd momentum(m);
        for (Eigen::Index i = 0; i < m; ++i)
        {
            momentum(i) = sqrt_mass(i) * R::rnorm(0.0, 1.0);
        }
        double current_kinetic = 0.5 * momentum.cwiseProduct(inv_mass).dot(momentum);

        // Make a copy of current state
        Eigen::VectorXd z_current = z;
        Eigen::VectorXd mom_current = momentum;

        // Leapfrog: first half-step in momentum
        momentum += 0.5 * step_size * grad_z;

        // Leapfrog integration
        for (unsigned int lf_step = 0; lf_step < n_leapfrog; ++lf_step)
        {
            // Position update
            z += step_size * momentum.cwiseProduct(inv_mass);

            // Gradient at new position (except after last step)
            double logpost_new;
            Eigen::VectorXd grad_z_new(m);
            eval_logpost_and_grad_z(z, logpost_new, grad_z_new);

            if (lf_step != n_leapfrog - 1)
            {
                momentum += step_size * grad_z_new;
            }
            else
            {
                // final half-step
                momentum += 0.5 * step_size * grad_z_new;
            }

            // store for acceptance
            if (lf_step == n_leapfrog - 1)
            {
                grad_z = grad_z_new;
                current_logpost = logpost_new;
            }
        }

        // Negate momentum for symmetry
        momentum = -momentum;

        // Compute proposed energy
        double proposed_energy = -current_logpost;
        double proposed_kinetic = 0.5 * momentum.cwiseProduct(inv_mass).dot(momentum);

        double H_proposed = proposed_energy + proposed_kinetic;
        double H_current = current_energy + current_kinetic;
        energy_diff = H_proposed - H_current;

        double accept_prob = 0.0;
        if (std::isfinite(energy_diff) &&
            std::abs(energy_diff) < 100.0 &&
            std::log(R::runif(0.0, 1.0)) < -energy_diff)
        {
            accept_prob = std::min(1.0, std::exp(-energy_diff));
            // accept: theta_col has already been set via last eval
            spatial.theta.col(k) = theta_col;
            spatial.alpha.col(k) = spatial.compute_alpha_col(k);
        }
        else
        {
            // reject: revert z and theta
            accept_prob = 0.0;
            z = z_current;
            // rebuild theta from old z
            set_theta_from_z(z, theta_col);
            spatial.theta.col(k) = theta_col;
            spatial.alpha.col(k) = spatial.compute_alpha_col(k);
        }

        return accept_prob;
    } // update_theta_col


    Eigen::VectorXd dloglike_dwdiag(
        double &loglike,
        const Eigen::MatrixXd &Y, // (nt + 1) x ns, observed primary infections
        const Eigen::MatrixXd &R_mat, // (nt + 1) x ns
        const bool &add_jacobian = true
    )
    {
        Eigen::MatrixXd u_mat(ns, ns);
        for (Eigen::Index s = 0; s < ns; s++)
        {
            for (Eigen::Index k = 0; k < ns; k++)
            {
                u_mat(s, k) = spatial.compute_unnormalized_weight(s, k);
            } // for source location k
        } // for destination location s

        Eigen::VectorXd deriv_wdiag(ns);
        deriv_wdiag.setZero();
        loglike = 0.0;
        for (Eigen::Index t = 0; t < nt + 1; t++)
        { // for destination time t
            for (Eigen::Index s = 0; s < ns; s++)
            { // for destination location s

                double lambda_st = mu;
                Eigen::VectorXd coef_sum(ns);
                coef_sum.setZero();
                for (Eigen::Index k = 0; k < ns; k++)
                {
                    for (Eigen::Index l = 0; l < t; l++)
                    {
                        if (t - l <= static_cast<Eigen::Index>(dlag.Fphi.size()))
                        {
                            double coef = R_mat(l, k) * dlag.Fphi(t - l - 1) * Y(l, k);
                            coef_sum(k) += coef;
                            lambda_st += spatial.alpha(s, k) * coef;
                        }
                    } // for source time l < t
                } // for source location k

                lambda_st = std::max(lambda_st, EPS);
                double dloglike_dlambda_st = Y(t, s) / lambda_st - 1.0;
                loglike += Y(t, s) * std::log(lambda_st) - lambda_st;

                for (Eigen::Index j = 0; j < ns; j++)
                {
                    double dalpha_sk_dwj = spatial.dalpha_dwj(s, j, u_mat.col(j));
                    deriv_wdiag(j) += dloglike_dlambda_st * dalpha_sk_dwj * coef_sum(j);
                }
            } // for destination location s
        } // for time t

        if (add_jacobian)
        {
            for (Eigen::Index j = 0; j < ns; j++)
            {
                deriv_wdiag(j) *= spatial.wdiag(j) * (1.0 - spatial.wdiag(j)); // Jacobian for logit transform
            }
        } // if add_jacobian

        return deriv_wdiag;
    } // dloglike_dwdiag


    double update_wdiag(
        double &energy_diff,
        double &grad_norm,
        const Eigen::MatrixXd &Y, // (nt + 1) x ns, observed primary infections
        const Eigen::MatrixXd &R_mat, // (nt + 1) x ns
        const double &step_size,
        const unsigned int &n_leapfrog,
        const double &prior_mean = 0.0,
        const double &prior_sd = 3.0
    )
    {
        const double prior_var = prior_sd * prior_sd;

        Eigen::VectorXd mass_diag(ns);
        mass_diag.setOnes();
        Eigen::VectorXd inv_mass = mass_diag.cwiseInverse();
        Eigen::VectorXd sqrt_mass = mass_diag.array().sqrt();

        // Current state
        Eigen::VectorXd wdiag_current = spatial.wdiag; // ns x 1
        Eigen::ArrayXd logit_wdiag = (wdiag_current.array() + EPS).log() - (1.0 - wdiag_current.array() + EPS).log();
        double loglike = 0.0;
        Eigen::VectorXd grad = dloglike_dwdiag(loglike, Y, R_mat, true);
        grad.array() += - (logit_wdiag - prior_mean) / prior_var; // prior contribution
        grad_norm = grad.norm();
        double current_logprior = - 0.5 * (logit_wdiag - prior_mean).square().matrix().sum() / prior_var;
        double current_energy = - (loglike + current_logprior);

        // Sample momentum
        Eigen::VectorXd momentum(ns);
        for (Eigen::Index i = 0; i < momentum.size(); i++)
        {
            momentum(i) = sqrt_mass(i) * R::rnorm(0.0, 1.0);
        }
        double current_kinetic = 0.5 * momentum.cwiseProduct(inv_mass).dot(momentum);

        // Leapfrog integration
        momentum += 0.5 * step_size * grad;
        for (unsigned int lf_step = 0; lf_step < n_leapfrog; lf_step++)
        {
            // Update logit_wdiag
            logit_wdiag += step_size * momentum.cwiseProduct(inv_mass).array();
            spatial.wdiag = (1.0 / (1.0 + (-logit_wdiag).exp())).matrix(); // inverse logit transform
            spatial.compute_alpha(); // update alpha after wdiag is updated

            // Compute new gradient
            grad = dloglike_dwdiag(loglike, Y, R_mat, true);
            grad.array() += - (logit_wdiag - prior_mean) / prior_var; // prior contribution

            // Update momentum
            if (lf_step != n_leapfrog - 1)
            {
                momentum += step_size * grad;
            }
        } // for leapfrog steps

        momentum += 0.5 * step_size * grad;
        momentum = -momentum; // Negate momentum to make proposal symmetric

        double proposed_logprior = - 0.5 * (logit_wdiag - prior_mean).square().matrix().sum() / prior_var;
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
            spatial.wdiag = wdiag_current;
            spatial.compute_alpha();
        } // end Metropolis step

        return accept_prob;
    } // update_wdiag


    Eigen::VectorXd dloglike_dparams_collapsed(
        double &loglike,
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
                u_mat(s, k) = spatial.compute_unnormalized_weight(s, k);
            } // for source location k
        } // for destination location s

        double W_safe = std::max(temporal.W, EPS);
        double W_sqrt = std::sqrt(W_safe);

        double deriv_rho_mobility = 0.0;
        double deriv_rho_dist = 0.0;
        double deriv_mu = 0.0;
        double deriv_W = 0.0;
        loglike = 0.0;
        for (Eigen::Index t = 0; t < nt + 1; t++)
        { // for destination time t
            for (Eigen::Index s = 0; s < ns; s++)
            { // for destination location s

                double lambda_st = mu;
                double dlambda_st_drho_mobility = 0.0;
                double dlambda_st_drho_dist = 0.0;
                for (Eigen::Index k = 0; k < ns; k++)
                {
                    double dalpha_sk_drho_mobility = spatial.dalpha_drho_mobility(s, k, u_mat.col(k));
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

                loglike += Y(t, s) * std::log(lambda_st) - lambda_st;
                if (include_W && t > 0)
                {
                    deriv_W += - 0.5 / W_safe + 0.5 * (temporal.wt(t, s) * temporal.wt(t, s)) / (W_safe * W_safe);
                    loglike += R::dnorm4(temporal.wt(t, s), 0.0, W_sqrt, true);
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
            } // if include_W
        } // if add_jacobian

        if (include_W)
        {
            return Eigen::Vector4d(deriv_rho_mobility, deriv_rho_dist, deriv_mu, deriv_W);
        }
        else
        {
            return Eigen::Vector3d(deriv_rho_mobility, deriv_rho_dist, deriv_mu);
        }
    } // dloglike_dparams_collapsed


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

        Rcpp::List params_list = Rcpp::List::create(
            Rcpp::Named("mu") = mu,
            Rcpp::Named("W") = temporal.W,
            Rcpp::Named("rho_dist") = spatial.rho_dist,
            Rcpp::Named("rho_mobility") = spatial.rho_mobility
        );

        return Rcpp::List::create(
            Rcpp::Named("Y") = Y,                 // (nt + 1) x ns
            Rcpp::Named("N") = tensor4_to_r(N),   // ns x ns x (nt + 1) x (nt + 1)
            Rcpp::Named("N0") = N0,               // (nt + 1) x ns
            Rcpp::Named("alpha") = spatial.alpha, // ns x ns
            Rcpp::Named("wt") = temporal.wt,      // (nt + 1) x ns
            Rcpp::Named("Rt") = Rt,               // (nt + 1) x ns
            Rcpp::Named("params") = params_list,
            Rcpp::Named("horseshoe") = Rcpp::List::create(
                Rcpp::Named("theta") = spatial.theta,
                Rcpp::Named("gamma") = spatial.gamma,
                Rcpp::Named("delta") = spatial.delta,
                Rcpp::Named("tau") = spatial.tau,
                Rcpp::Named("zeta") = spatial.zeta,
                Rcpp::Named("wdiag") = spatial.wdiag
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
        Prior static_params_prior("gaussian", 0.0, 3.0, true); // prior for log(static parameters)
        Prior hs_theta_prior("gaussian", 0.0, 3.0, true); // prior for horseshoe theta
        Prior wdiag_prior("gaussian", 0.0, 3.0, true); // prior for logit(wdiag), self-exciting weight
        bool hmc_include_W = false;

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
                static_params_prior = Prior(params_opts);
                if (params_opts.containsElementNamed("include_W"))
                {
                    hmc_include_W = Rcpp::as<bool>(params_opts["include_W"]);
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
            } // if params
            if (mcmc_opts_list.containsElementNamed("horseshoe"))
            {
                Rcpp::List hs_opts = mcmc_opts_list["horseshoe"];
                hs_theta_prior = Prior(hs_opts);
                spatial.shared_tau = true;
                if (hs_opts.containsElementNamed("shared_tau"))
                {
                    spatial.shared_tau = Rcpp::as<bool>(hs_opts["shared_tau"]);
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
                }
            } // if horseshoe
            if (mcmc_opts_list.containsElementNamed("wdiag"))
            {
                Rcpp::List wdiag_opts = mcmc_opts_list["wdiag"];
                wdiag_prior = Prior(wdiag_opts);
                if (wdiag_opts.containsElementNamed("init"))
                {
                    spatial.wdiag = Rcpp::as<Eigen::VectorXd>(wdiag_opts["init"]);
                }
            } // if wdiag
        } // if mcmc_opts


        // Set up HMC options and diagnostics for spatial parameters if to be inferred
        HMCOpts_1d hmc_opts, wdiag_hmc_opts;
        HMCDiagnostics_1d hmc_diag, wdiag_hmc_diag;
        DualAveraging_1d da_adapter, wdiag_da_adapter;
        if (static_params_prior.infer)
        {
            hmc_opts.leapfrog_step_size = static_params_prior.hmc_step_size_init;
            hmc_opts.nleapfrog = static_params_prior.hmc_nleapfrog_init;
            hmc_diag = HMCDiagnostics_1d(static_cast<unsigned int>(ntotal), nburnin, true);
            da_adapter = DualAveraging_1d(hmc_opts);
        } // infer static params
        if (wdiag_prior.infer)
        {
            wdiag_hmc_opts.leapfrog_step_size = wdiag_prior.hmc_step_size_init;
            wdiag_hmc_opts.nleapfrog = wdiag_prior.hmc_nleapfrog_init;
            wdiag_hmc_diag = HMCDiagnostics_1d(static_cast<unsigned int>(ntotal), nburnin, true);
            wdiag_da_adapter = DualAveraging_1d(wdiag_hmc_opts);
        } // infer wdiag

        HMCOpts_2d hs_th_hmc_opts;
        HMCDiagnostics_2d hs_th_hmc_diag;
        DualAveraging_2d hs_th_da_adapter;
        if (hs_theta_prior.infer)
        {
            hs_th_hmc_opts.leapfrog_step_size_init = hs_theta_prior.hmc_step_size_init;
            hs_th_hmc_opts.nleapfrog_init = hs_theta_prior.hmc_nleapfrog_init;
            hs_th_hmc_opts.set_size(static_cast<unsigned int>(ns));

            hs_th_hmc_diag = HMCDiagnostics_2d(
                static_cast<unsigned int>(ns), 
                static_cast<unsigned int>(ntotal), 
                nburnin, 
                true
            );
            hs_th_da_adapter = DualAveraging_2d(
                static_cast<unsigned int>(ns), 
                hs_theta_prior.hmc_step_size_init
            );
        }

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
        Eigen::Tensor<double, 3> wt_samples;
        if (wt_prior.infer)
        {
            wt_samples.resize(temporal.wt.rows(), temporal.wt.cols(), nsamples);
        }

        Eigen::VectorXd rho_dist_samples, rho_mobility_samples, mu_samples, W_samples;
        if (static_params_prior.infer)
        {
            rho_mobility_samples.resize(nsamples);
            rho_dist_samples.resize(nsamples);
            mu_samples.resize(nsamples);
            if (hmc_include_W)
            {
                W_samples.resize(nsamples);
            }
        }

        Eigen::Tensor<double, 3> theta_samples, gamma_samples, delta_samples;
        Eigen::MatrixXd tau_samples, zeta_samples;
        if (hs_theta_prior.infer)
        {
            theta_samples.resize(spatial.theta.rows(), spatial.theta.cols(), nsamples);
            gamma_samples.resize(spatial.gamma.rows(), spatial.gamma.cols(), nsamples);
            delta_samples.resize(spatial.delta.rows(), spatial.delta.cols(), nsamples);
            tau_samples.resize(spatial.tau.size(), nsamples);
            zeta_samples.resize(spatial.zeta.size(), nsamples);
        }

        Eigen::MatrixXd wdiag_samples;
        if (wdiag_prior.infer)
        {
            wdiag_samples.resize(spatial.wdiag.size(), nsamples);
        }

        Eigen::Tensor<double, 3> alpha_samples(spatial.alpha.rows(), spatial.alpha.cols(), nsamples);

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
            } // if infer wt

            Eigen::MatrixXd R_mat = temporal.compute_Rt(); // (nt + 1) x ns


            // Update horseshoe theta
            if (hs_theta_prior.infer)
            {
                for (Eigen::Index k = 0; k < ns; k++)
                {
                    double energy_diff = 0.0;
                    double grad_norm = 0.0;
                    double accept_prob = update_theta_col(
                        energy_diff, grad_norm, k, Y, R_mat,
                        hs_th_hmc_opts.leapfrog_step_size(k),
                        hs_th_hmc_opts.nleapfrog(k)
                    );

                    // (Optional) Update diagnostics and dual averaging for horseshoe theta
                    hs_th_hmc_diag.accept_count(k) += accept_prob;
                    if (hs_th_hmc_opts.diagnostics)
                    {
                        hs_th_hmc_diag.energy_diff(k, iter) = energy_diff;
                        hs_th_hmc_diag.grad_norm(k, iter) = grad_norm;
                    }
                    if (hs_th_hmc_opts.dual_averaging && iter <= nburnin)
                    {
                        if (iter < nburnin)
                        {
                            hs_th_hmc_opts.leapfrog_step_size(k) = hs_th_da_adapter.update_step_size(k, accept_prob);
                        }
                        else if (iter == nburnin)
                        {
                            double step_size = hs_th_hmc_opts.leapfrog_step_size(k);
                            hs_th_da_adapter.finalize_leapfrog_step(
                                step_size,
                                hs_th_hmc_opts.nleapfrog(k),
                                k, hs_th_hmc_opts.T_target
                            );
                            hs_th_hmc_opts.leapfrog_step_size(k) = step_size;
                        }

                        if (hs_th_hmc_opts.diagnostics)
                        {
                            hs_th_hmc_diag.leapfrog_step_size_stored(k, iter) = hs_th_hmc_opts.leapfrog_step_size(k);
                            hs_th_hmc_diag.nleapfrog_stored(k, iter) = hs_th_hmc_opts.nleapfrog(k);
                        }
                    } // if dual averaging
                } // for location k

                // Update horseshoe scales
                // spatial.update_horseshoe_params();
            } // if infer horseshoe theta


            // Update wdiag
            if (wdiag_prior.infer)
            {
                double energy_diff = 0.0;
                double grad_norm = 0.0;
                double accept_prob = update_wdiag(
                    energy_diff, grad_norm, Y, R_mat,
                    wdiag_hmc_opts.leapfrog_step_size,
                    wdiag_hmc_opts.nleapfrog,
                    wdiag_prior.par1, wdiag_prior.par2
                );

                // (Optional) Update diagnostics and dual averaging for wdiag
                wdiag_hmc_diag.accept_count += accept_prob;
                if (wdiag_hmc_opts.diagnostics)
                {
                    wdiag_hmc_diag.energy_diff(iter) = energy_diff;
                    wdiag_hmc_diag.grad_norm(iter) = grad_norm;
                }
                if (wdiag_hmc_opts.dual_averaging && iter <= nburnin)
                {
                    if (iter < nburnin)
                    {
                        wdiag_hmc_opts.leapfrog_step_size = wdiag_da_adapter.update_step_size(accept_prob);
                    }
                    else if (iter == nburnin)
                    {
                        wdiag_da_adapter.finalize_leapfrog_step(
                            wdiag_hmc_opts.leapfrog_step_size,
                            wdiag_hmc_opts.nleapfrog,
                            wdiag_hmc_opts.T_target);
                    }

                    if (wdiag_hmc_opts.diagnostics)
                    {
                        wdiag_hmc_diag.leapfrog_step_size_stored(iter) = wdiag_hmc_opts.leapfrog_step_size;
                        wdiag_hmc_diag.nleapfrog_stored(iter) = wdiag_hmc_opts.nleapfrog;
                    }
                } // if dual averaging
            } // if infer wdiag


            // Update static parameters (rho_mobility, rho_dist, mu, [W])
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
            } // if infer static params


            // Store samples after burn-in and thinning
            if (iter >= nburnin && ((iter - nburnin) % nthin == 0))
            {
                Eigen::Index sample_idx = (iter - nburnin) / nthin;

                if (static_params_prior.infer)
                {
                    rho_dist_samples(sample_idx) = spatial.rho_dist;
                    rho_mobility_samples(sample_idx) = spatial.rho_mobility;
                    mu_samples(sample_idx) = mu;
                    if (hmc_include_W)
                    {
                        W_samples(sample_idx) = temporal.W;
                    }
                }

                if (wt_prior.infer)
                {
                    // Store wt samples
                    // wt_samples.chip(sample_idx, 2) = temporal.wt;
                    Eigen::TensorMap<Eigen::Tensor<const double, 2>> wt_map(temporal.wt.data(), temporal.wt.rows(), temporal.wt.cols());
                    wt_samples.chip(sample_idx, 2) = wt_map;
                }

                if (hs_theta_prior.infer)
                {
                    // Store horseshoe samples
                    Eigen::TensorMap<Eigen::Tensor<const double, 2>> theta_map(spatial.theta.data(), spatial.theta.rows(), spatial.theta.cols());
                    Eigen::TensorMap<Eigen::Tensor<const double, 2>> gamma_map(spatial.gamma.data(), spatial.gamma.rows(), spatial.gamma.cols());
                    Eigen::TensorMap<Eigen::Tensor<const double, 2>> delta_map(spatial.delta.data(), spatial.delta.rows(), spatial.delta.cols());
                    Eigen::Map<const Eigen::VectorXd> tau_map(spatial.tau.data(), spatial.tau.size());
                    Eigen::Map<const Eigen::VectorXd> zeta_map(spatial.zeta.data(), spatial.zeta.size());

                    theta_samples.chip(sample_idx, 2) = theta_map;
                    gamma_samples.chip(sample_idx, 2) = gamma_map;
                    delta_samples.chip(sample_idx, 2) = delta_map;
                    tau_samples.col(sample_idx) = tau_map;
                    zeta_samples.col(sample_idx) = zeta_map;
                }

                if (wdiag_prior.infer)
                {
                    // Store wdiag samples
                    Eigen::Map<const Eigen::VectorXd> wdiag_map(spatial.wdiag.data(), spatial.wdiag.size());
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

        if (wt_prior.infer)
        {
            output["wt"] = tensor3_to_r(wt_samples); // (nt + 1) x ns x nsamples
            output["wt_accept_prob"] = accept_prob_wt / static_cast<double>(ntotal); // (nt + 1) x ns
        } // if infer wt

        if (static_params_prior.infer)
        {
            Rcpp::List param_list = Rcpp::List::create(
                Rcpp::Named("mu") = mu_samples, // nsamples x 1
                Rcpp::Named("rho_dist") = rho_dist_samples, // nsamples x 1
                Rcpp::Named("rho_mobility") = rho_mobility_samples // nsamples x 1
            );
            if (hmc_include_W)
            {
                param_list["W"] = W_samples; // nsamples x 1
            }
            param_list["hmc"] = Rcpp::List::create(
            Rcpp::Named("acceptance_rate") = hmc_diag.accept_count / static_cast<double>(ntotal),
            Rcpp::Named("leapfrog_step_size") = hmc_opts.leapfrog_step_size,
            Rcpp::Named("n_leapfrog") = hmc_opts.nleapfrog,
            Rcpp::Named("diagnostics") = hmc_diag.to_list());

            output["params"] = param_list;
        } // if infer static params

        if (hs_theta_prior.infer)
        {
            Rcpp::List hs_list = Rcpp::List::create(
                Rcpp::Named("theta") = tensor3_to_r(theta_samples), // ns x ns x nsamples
                Rcpp::Named("gamma") = tensor3_to_r(gamma_samples), // ns x ns x nsamples
                Rcpp::Named("delta") = tensor3_to_r(delta_samples), // ns x ns x nsamples
                Rcpp::Named("tau") = tau_samples, // ns x nsamples
                Rcpp::Named("zeta") = zeta_samples // ns x nsamples
            );
            hs_list["hmc"] = Rcpp::List::create(
                Rcpp::Named("acceptance_rate") = hs_th_hmc_diag.accept_count / static_cast<double>(ntotal),
                Rcpp::Named("leapfrog_step_size") = hs_th_hmc_opts.leapfrog_step_size,
                Rcpp::Named("n_leapfrog") = hs_th_hmc_opts.nleapfrog,
                Rcpp::Named("diagnostics") = hs_th_hmc_diag.to_list()
            );

            output["horseshoe"] = hs_list;
        } // if infer horseshoe theta

        if (wdiag_prior.infer)
        {
            output["wdiag"] = Rcpp::List::create(
                Rcpp::Named("samples") = wdiag_samples, // ns x nsamples
                Rcpp::Named("hmc") = Rcpp::List::create(
                    Rcpp::Named("acceptance_rate") = wdiag_hmc_diag.accept_count / static_cast<double>(ntotal),
                    Rcpp::Named("leapfrog_step_size") = wdiag_hmc_opts.leapfrog_step_size,
                    Rcpp::Named("n_leapfrog") = wdiag_hmc_opts.nleapfrog,
                    Rcpp::Named("diagnostics") = wdiag_hmc_diag.to_list()
                )
            );
        } // if infer wdiag

        if (sample_augmented_N)
        {
            output["N"] = tensor3_to_r(N_samples); // (nt + 1) x ns x nsamples
            output["N0"] = tensor3_to_r(N0_samples); // (nt + 1) x ns x nsamples
        }

        return output;
    } // run_mcmc

}; // class Model



#endif // MODEL_HPP