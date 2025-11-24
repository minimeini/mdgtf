#pragma once
#ifndef BYM2_H
#define BYM2_H

#include <RcppArmadillo.h>
#include "../utils/utils.h"
// [[Rcpp::plugins(cpp17)]]
// [[Rcpp::depends(RcppArmadillo)]]


struct BYM2Prior
{
    bool infer = false;
    double mu_phi = 0.0;
    double sigma_phi = 1.0;
    double shape_tau = 1.0;
    double rate_tau = 1.0;

    // MCMC settings
    double mh_sd = 0.1;
    double accept_count = 0.0;

    BYM2Prior() = default;
    BYM2Prior(const Rcpp::List &opts)
    {
        if (opts.containsElementNamed("infer"))
        {
            infer = Rcpp::as<bool>(opts["infer"]);
        }
        if (opts.containsElementNamed("logit_phi"))
        {
            Rcpp::NumericVector logit_phi_opts = opts["logit_phi"];
            mu_phi = logit_phi_opts[0];
            sigma_phi = logit_phi_opts[1];
        }
        if (opts.containsElementNamed("tau_b"))
        {
            Rcpp::NumericVector tau_b_opts = opts["tau_b"];
            shape_tau = tau_b_opts[0];
            rate_tau = tau_b_opts[1];
        }
        if (opts.containsElementNamed("mh_sd"))
        {
            mh_sd = opts["mh_sd"];
        }
    }

    void adapt_phi_proposal_robbins_monro(
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
    }
};



class SpatialStructure
{
public:
    unsigned int nS; // number of locations for spatio-temporal model
    arma::mat V; // binary neighborhood matrix
    arma::vec neighbors; // number of neighbors for each location (row sums)
    arma::mat W; // row-standardized weight matrix

    arma::mat Q; // precision matrix for BYM2, Q = D - V
    arma::vec eigval_Q; // eigenvalues of Q
    arma::mat eigvec_Q; // eigenvectors of Q
    arma::uvec pos_eig_idx; // indices of positive eigenvalues of Q

    arma::mat Q_scaled_ginv; // scaled generalized inverse of Q
    arma::mat prec; // precision matrix for BYM2

    double scale_factor = 0.0; // scaling factor for BYM2
    double tau_b = 1.0; // overall precision parameter for BYM2
    double phi = 0.5; // mixing parameter for BYM2, between 0 and 1
    double mu = 0.0; // overall mean

    SpatialStructure(const unsigned int &nlocation = 1)
    {
        nS = nlocation;
        V = arma::mat(nS, nS, arma::fill::zeros);
        neighbors = arma::vec(nS, arma::fill::zeros);
        W = arma::mat(nS, nS, arma::fill::zeros);

        compute_precision();
        compute_scale_factor();
        compute_Q_scaled_ginv();
        return;
    } // end of constructor


    SpatialStructure(const arma::mat &neighborhood_matrix)
    {
        // V: binary neighborhood matrix
        V = arma::symmatu(neighborhood_matrix); // ensure symmetry
        nS = V.n_rows;
        V.diag().zeros(); // zero diagonal
        neighbors = arma::sum(V, 1); // row sums

        // W: row-standardized weight matrix
        W = V;
        W.each_col() /= neighbors; // row-standardized weight matrix

        compute_precision();
        compute_scale_factor();
        compute_Q_scaled_ginv();
        return;
    } // end of constructor


    void compute_precision(const double &tol = 1.0e-10)
    {
        // Q: precision matrix for BYM2
        Q = -V;
        Q.diag() += neighbors;

        // Eigendecomposition of ICAR precision
        arma::eig_sym(eigval_Q, eigvec_Q, Q);
        
        // Get positive eigenvalues (excluding the zero eigenvalue)
        pos_eig_idx = arma::find(eigval_Q > tol);
        return;
    } // end of compute_precision()


    void compute_scale_factor() 
    {
        if (pos_eig_idx.n_elem == 0) {
            scale_factor = 1.0;
            return;
        }

        arma::vec pos_eigval = eigval_Q(pos_eig_idx);
        
        // Geometric mean for scaling
        double log_gmean = arma::mean(arma::log(pos_eigval));
        scale_factor = std::exp(0.5 * log_gmean);
    } // end of compute_scale_factor()


    void compute_Q_scaled_ginv()
    {
        if (pos_eig_idx.n_elem == 0) {
            Q_scaled_ginv.zeros(nS, nS);
            return;
        }

        arma::mat V_pos = eigvec_Q.cols(pos_eig_idx);
        arma::vec lambda_pos = eigval_Q(pos_eig_idx);
        double scale2 = scale_factor * scale_factor;
        arma::vec lambda_inv = 1.0 / (lambda_pos * scale2);
        Q_scaled_ginv = V_pos * arma::diagmat(lambda_inv) * V_pos.t();
        return;
    } // end of compute_Q_scaled_ginv()


    arma::vec sample_icar_component()
    {
        if (pos_eig_idx.n_elem == 0) {
            return arma::vec(nS, arma::fill::zeros);
        }

        // Sample independent normals for non-constant eigenvectors
        arma::vec z_prec = arma::sqrt(eigval_Q(pos_eig_idx));
        arma::vec z(pos_eig_idx.n_elem, arma::fill::randn);
        z /= z_prec;

        // Transform back: u = sum_i z_i * v_i
        arma::mat V_pos = eigvec_Q.cols(pos_eig_idx);
        arma::vec u = V_pos * z;

        // Center to sum to zero (project out constant vector)
        u -= arma::mean(u);
        return u;
    } // end of sample_icar_component()


    arma::mat sample_spatial_effects(const unsigned int k = 1) 
    {
        arma::mat samples(nS, k);
        
        for (unsigned int j = 0; j < k; ++j) {
            samples.col(j) = sample_spatial_effects_vec();
        }
        
        return samples;
    } // end of sample_spatial_effects()


    arma::vec sample_spatial_effects_vec() 
    {
        arma::vec v(nS, arma::fill::randn);

        // 2. Sample structured ICAR component u
        arma::vec u = sample_icar_component();

        // 3. Scale the ICAR component to get u*
        arma::vec u_star = u / scale_factor;

        // 4. Combine according to BYM2 formula
        arma::vec b = (1.0 / std::sqrt(tau_b)) *
                      (std::sqrt(1.0 - phi) * v +
                       std::sqrt(phi) * u_star);

        // 5. Add overall mean
        b += mu;

        return b;
    } // end of sample_spatial_effects_vec()


    arma::vec dloglik_dspatial(const arma::vec &b_observed)
    {
        // Gradient of the log-likelihood w.r.t. spatial effects
        arma::mat Rphi_inv = (1.0 - phi) * arma::eye(nS, nS) + phi * Q_scaled_ginv;
        arma::mat Rphi = inverse(Rphi_inv);
        return - tau_b * Rphi * (b_observed - mu);
    } // end of dloglik_dspatial()


    double log_likelihood(const arma::vec &b_observed, const bool &include_constant = false) const
    {
        // Log-likelihood of the spatial effects under the BYM2 prior
        arma::mat Rphi_inv = (1.0 - phi) * arma::eye(nS, nS) + phi * Q_scaled_ginv;
        arma::mat Rphi = inverse(Rphi_inv);
        double quad_form = arma::dot(b_observed - mu, Rphi * (b_observed - mu));
        double deriv = -0.5 * tau_b * quad_form;

        if (include_constant)
        {
            double nS = static_cast<double>(b_observed.n_elem);
            double logdet = arma::log_det_sympd(arma::symmatu(Rphi));
            double cnst = nS * std::log(tau_b);
            deriv += - 0.5 * nS * std::log(2.0 * M_PI) + 0.5 * nS * std::log(tau_b) + 0.5 * logdet;
        }
        
        return deriv;
    } // end of log_likelihood()


    void update_mu_tau_jointly(
        const arma::vec &b_observed, 
        const double &shape_tau = 1.0, 
        const double &rate_tau = 1.0
    )
    {
        // Given phi, (mu, tau_b) have a Normal-Gamma posterior
        arma::mat Rphi_inv = (1.0 - phi) * arma::eye(nS, nS) + phi * Q_scaled_ginv;
        arma::mat Rphi_current = inverse(Rphi_inv);

        arma::vec ones = arma::ones(nS);
        arma::vec Rphi_b = Rphi_current * b_observed;
        arma::vec Rphi_ones = Rphi_current * ones;

        double one_Rphi_one = arma::dot(ones, Rphi_ones);
        double b_Rphi_b = arma::dot(b_observed, Rphi_b);
        double one_Rphi_b = arma::dot(ones, Rphi_b);

        // Sample tau_b first
        double quadratic_form = b_Rphi_b - (one_Rphi_b * one_Rphi_b) / one_Rphi_one;
        double shape_post = shape_tau + 0.5 * static_cast<double>(nS) - 0.5;
        double rate_post = rate_tau + 0.5 * quadratic_form;
        tau_b = R::rgamma(shape_post, 1.0 / rate_post);

        // Then sample mu | tau_b
        double post_mean = one_Rphi_b / one_Rphi_one;
        double post_var = 1.0 / (tau_b * one_Rphi_one);
        mu = R::rnorm(post_mean, std::sqrt(post_var));
    }


    // Normal prior on logit(phi)
    double log_prior_logit_phi(
        const double phi_val, 
        const double mu = 0.0, 
        const double sigma = 1.0
    )
    {
        double logit_phi = logit(phi_val);
        // No need for jacobian if the model is parameterized at the logit scale from the start
        // double jacobian = -std::log(phi_val) - std::log(1.0 - phi_val);
        return R::dnorm4(logit_phi, mu, sigma, true);
    }
    // // beta prior on logit(phi)
    // double log_prior_phi(const double phi_val, const double shape_alpha = 1.0, const double shape_beta = 1.0)
    // {
    //     return R::dbeta(phi_val, shape_alpha, shape_beta, true);
    // }


    double log_post_phi(
        const double &phi_val, // phi on original scale (0, 1)
        const arma::vec &b_observed,
        const BYM2Prior &prior
    )
    {
        if (phi_val <= 0.0 || phi_val >= 1.0)
            return -INFINITY;
        // Cov(b) = (1/tau_b)*[ (1-phi) I + phi * Q_scaled_ginv ]
        arma::mat Rphi_inv = (1.0 - phi_val) * arma::eye(nS, nS) + phi_val * Q_scaled_ginv;
        arma::mat Rphi = inverse(Rphi_inv);
        double logdet = arma::log_det_sympd(arma::symmatu(Rphi));

        arma::vec ones(nS, arma::fill::ones);
        arma::vec Rphi_b = Rphi * b_observed;
        arma::vec Rphi_ones = Rphi * ones;
        double one_Rphi_one = arma::dot(ones, Rphi_ones);
        double b_Rphi_b = arma::dot(b_observed, Rphi_b);
        double one_Rphi_b = arma::dot(b_observed, Rphi_ones);

        double quadratic_form = b_Rphi_b - (one_Rphi_b * one_Rphi_b) / one_Rphi_one;
        double shape_post = prior.shape_tau + 0.5 * static_cast<double>(nS) - 0.5;
        double rate_post = prior.rate_tau + 0.5 * quadratic_form;

        // Log marginal (integrating out tau_b and mu)
        return 0.5 * logdet - shape_post * std::log(rate_post) - 0.5 * std::log(one_Rphi_one) + log_prior_logit_phi(phi_val, prior.mu_phi, prior.sigma_phi);
    }


    double update_phi_logit(
        const arma::vec &b_observed,
        const BYM2Prior &prior
    )
    {
        double logit_phi = logit(phi);
        double prop = logit_phi + R::rnorm(0.0, prior.mh_sd);
        double phi_prop = 1.0 / (1.0 + std::exp(-prop));
        double log_acc = log_post_phi(phi_prop, b_observed, prior) - log_post_phi(phi, b_observed, prior);
        if (std::log(R::runif(0, 1)) < log_acc)
        {
            // accept
            phi = phi_prop;
            return 1.0;
        }
        else
        {
            // reject
            return 0.0;
        }
    }


    // Main MCMC function
    Rcpp::List run_mcmc(
        const arma::vec &b_observed,
        int n_iter = 5000,
        int burn_in = 1000,
        int thin = 1,
        bool verbose = true
    ) {
        BYM2Prior prior;
        int n_save = (n_iter - burn_in) / thin;
        arma::vec mu_samples(n_save, arma::fill::zeros);
        arma::vec tau_b_samples(n_save, arma::fill::zeros);
        arma::vec phi_samples(n_save, arma::fill::zeros);
        
        int save_idx = 0;
        for (int iter = 0; iter < n_iter; ++iter) {
            // Update parameters
            double acc = update_phi_logit(b_observed, prior);
            prior.accept_count += acc;
            update_mu_tau_jointly(b_observed, prior.shape_tau, prior.rate_tau);

            // Adapt proposal during burn-in
            prior.adapt_phi_proposal_robbins_monro(iter, burn_in);
            
            // Save samples after burn-in
            if (iter >= burn_in && (iter - burn_in) % thin == 0) {
                mu_samples(save_idx) = mu;
                tau_b_samples(save_idx) = tau_b;
                phi_samples(save_idx) = phi;
                save_idx++;
            }
            
            // Progress report
            if (verbose && iter % 1000 == 0) {
                Rcpp::Rcout << "Iteration " << iter << "/" << n_iter
                           << " - mu: " << mu 
                           << ", tau_b: " << tau_b
                           << ", phi: " << phi << std::endl;
            }
        }
        
        return Rcpp::List::create(
            Rcpp::Named("mu") = mu_samples,
            Rcpp::Named("tau_b") = tau_b_samples,
            Rcpp::Named("phi") = phi_samples,
            Rcpp::Named("final_mh_sd") = prior.mh_sd
        );
    }
}; // end of class SpatialStructure


#endif