#pragma once
#ifndef SPATIALSTRUCTURE_H
#define SPATIALSTRUCTURE_H

#include <RcppArmadillo.h>
#include "../utils/utils.h"
// [[Rcpp::plugins(cpp17)]]
// [[Rcpp::depends(RcppArmadillo)]]

class SpatialStructure
{
private:
    mutable arma::mat L_; // Cached lower Cholesky of Q (Q = L_ * L_^T)
    mutable bool chol_ready_ = false;
    bool logpost_rho_calculated_ = false;

    void ensure_chol_() const
    {
        if (chol_ready_) return;

        // Optional numerical jitter if near-singular; keep tiny.
        // Qsym.diag() += 1e-10;

        // Lower-triangular Cholesky
        L_ = arma::chol(arma::symmatu(Q), "lower");   // throws if not SPD
        chol_ready_ = true;
        return;
    }

public:
    unsigned int nS; // number of locations for spatio-temporal model
    arma::mat V; // neighborhood matrix for CAR
    arma::vec neighbors; // row sums: number of neighbors for each location
    arma::mat W; // row-standardized weight matrix for CAR

    arma::mat Q; // precision matrix for CAR
    double one_Q_one;
    double car_mu = 0.0;
    double car_tau2 = 1.0; 
    double car_rho = 0.99;

    double min_car_rho = 0.0;
    double max_car_rho = 1.0;
    double post_mu_mean = 0.0;
    double post_mu_prec = 1.0;
    double post_tau2_shape = 0.01;
    double post_tau2_rate = 0.01;

    SpatialStructure(const unsigned int &nlocation = 1)
    {
        nS = nlocation;
        V = arma::mat(nS, nS, arma::fill::zeros);
        neighbors = arma::vec(nS, arma::fill::zeros);
        W = arma::mat(nS, nS, arma::fill::zeros);
        Q = arma::mat(nS, nS, arma::fill::zeros);
        chol_ready_ = false;
        logpost_rho_calculated_ = false;
        return;
    }

    SpatialStructure(
        const arma::mat &neighborhood_matrix, 
        const double &mu = 0.0, 
        const double &tau2 = 1.0, 
        const double &rho = 0.99)
    {
        V = arma::symmatu(neighborhood_matrix); // ensure symmetry
        nS = V.n_rows;
        V.diag().zeros(); // zero diagonal
        neighbors = arma::sum(V, 1); // row sums

        compute_standardized_weights();

        car_mu = mu;
        car_tau2 = tau2;
        car_rho = rho;
        logpost_rho_calculated_ = false;
        compute_precision();
        return;
    }

    void compute_standardized_weights()
    {
        W = V;
        W.each_col() /= neighbors; // row-standardized weight matrix
        arma::vec eig_vals = arma::eig_sym(W);
        double lam_max = eig_vals.max();
        double lam_min = eig_vals.min();
        if (lam_min < 0)
        {
            min_car_rho = std::max(1.0 / lam_min, -1.0);
            max_car_rho = std::min(1.0 / lam_max, 1.0);
        }
        else
        {
            min_car_rho = 1.0 / lam_max;
            max_car_rho = std::min(1.0 / lam_min, 1.0);
        }
        return;
    }

    void compute_precision()
    {
        Q = - car_rho * V;
        Q.diag() += neighbors;
        Q = arma::symmatu(Q); // ensure symmetry

        arma::vec ones(nS, arma::fill::ones);
        one_Q_one = arma::as_scalar(ones.t() * Q * ones) + EPS;

        chol_ready_ = false;
        return;
    }

    void init_params()
    {
        car_mu = R::rnorm(0.0, 1.0);
        car_tau2 = R::rgamma(1.0, 1.0);
        car_rho = R::runif(0.0, 1.0);
        logpost_rho_calculated_ = false;
        compute_precision();
        return;
    }

    void update_params(const double &mu, const double &tau2, const double &rho)
    {
        double mu_old = car_mu;
        double tau2_old = car_tau2;
        double rho_old = car_rho;

        if (std::abs(rho_old - rho) > EPS)
        {
            // If rho has changed, regardless of tau2 or mu

            /*
            Only recompute if rho has changed:
            - precision matrix Q
            - one_Q_one
            */
            car_mu = mu;
            car_tau2 = tau2;
            car_rho = rho;
            compute_precision();

            /*
            When rho is updated, we also need to update:
            - post_mu_mean
            - post_mu_prec
            - post_tau2_rate

            They depend on `spatial_effects` and are updated in `log_posterior_rho()`
            => flag `logpost_rho_calculated_ = false` to indicate that these need to be recomputed
            */
            logpost_rho_calculated_ = false;
        }
        else
        {
            // If rho is not changed -> not need to recompute Q and tau2 posterior params

            if (std::abs(tau2_old - tau2) > EPS)
            {
                /*
                Only recompute if tau2 has changed:
                - post_mu_prec
                */
                car_tau2 = tau2;
                post_mu_prec = car_tau2 * one_Q_one;
            }

            if (std::abs(mu_old - mu) > EPS)
            {
                /*
                Nothing needs to be recomputed when mu changes
                */
                car_mu = mu;
            }
        }

        return;
    } // end of update_car()

    // Draw k samples from N(mu, Q^{-1}) where mu = car_mu * 1
    const arma::mat prior_sample_spatial_effects(const unsigned int &k = 1)
    {
        ensure_chol_();  // factor once
        arma::mat Z = arma::randn(nS, k); // z ~ N(0, I)
        arma::mat Y = arma::solve(
            arma::trimatu(L_.t()), Z, // y = L^{-T} z
            arma::solve_opts::fast);
        // arma::vec mu = arma::ones<arma::vec>(nS) * car_mu;
        // Y.each_col() += mu; 
        Y += car_mu; // add mean
        return Y;
    }

    const arma::vec prior_sample_spatial_effects_vec()
    {
        return prior_sample_spatial_effects(1).col(0);
    }


    const double log_posterior_rho(
        const arma::vec &spatial_effects,
        const double &jeffrey_prior_order = 1.0
    )
    {
        arma::vec ones(nS, arma::fill::ones);
        double one_Q_spatial = arma::as_scalar(ones.t() * Q * spatial_effects);
        post_mu_mean = one_Q_spatial / one_Q_one;
        post_mu_prec = car_tau2 * one_Q_one;

        arma::vec res = spatial_effects;
        res -= post_mu_mean;
        post_tau2_rate = 0.5 * arma::as_scalar(res.t() * Q * res) + EPS;
        post_tau2_shape = 0.5 * (nS - 1) + jeffrey_prior_order - 1.0;

        double logdet_Q = arma::log_det_sympd(Q);

        logpost_rho_calculated_ = true;
        return 0.5 * logdet_Q - 0.5 * std::log(one_Q_one)
               - post_tau2_shape * std::log(post_tau2_rate);
    } // end of log_posterior_rho()
};

#endif