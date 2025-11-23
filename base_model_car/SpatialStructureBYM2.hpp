#pragma once
#ifndef SPATIALSTRUCTUREBYM2_H
#define SPATIALSTRUCTUREBYM2_H

#include <RcppArmadillo.h>
#include "../utils/utils.h"
// [[Rcpp::plugins(cpp17)]]
// [[Rcpp::depends(RcppArmadillo)]]

class SpatialStructureBYM2
{
public:
    unsigned int nS; // number of locations for spatio-temporal model
    arma::mat V; // binary neighborhood matrix
    arma::vec neighbors; // number of neighbors for each location (row sums)
    arma::mat W; // row-standardized weight matrix

    arma::mat Q; // precision matrix for BYM2
    arma::vec eigval_Q; // eigenvalues of Q
    arma::mat eigvec_Q; // eigenvectors of Q
    arma::uvec pos_eig_idx; // indices of positive eigenvalues of Q

    arma::mat Q_scaled_ginv; // scaled generalized inverse of Q
    arma::mat prec; // precision matrix for BYM2

    double scale_factor = 0.0; // scaling factor for BYM2
    double tau_b = 1.0; // overall precision parameter for BYM2
    double phi = 0.5; // mixing parameter for BYM2, between 0 and 1
    double mu = 0.0; // overall mean

    SpatialStructureBYM2(const unsigned int &nlocation = 1)
    {
        nS = nlocation;
        V = arma::mat(nS, nS, arma::fill::zeros);
        neighbors = arma::vec(nS, arma::fill::zeros);
        W = arma::mat(nS, nS, arma::fill::zeros);

        compute_precision();
        compute_scale_factor();
        return;
    } // end of constructor


    SpatialStructureBYM2(
        const arma::mat &neighborhood_matrix)
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
        return;
    } // end of constructor


    void init_params()
    {
        tau_b = R::rgamma(1.0, 1.0);
        phi = R::runif(0.0, 1.0);
        mu = R::rnorm(0.0, 1.0);
        return;
    } // end of init_params()


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


    void compute_scale_factor() {
        
        arma::vec pos_eigval = eigval_Q(pos_eig_idx);
        
        // Geometric mean for scaling
        double log_gmean = arma::mean(arma::log(pos_eigval));
        scale_factor = std::exp(0.5 * log_gmean);
    } // end of compute_scale_factor()


    void compute_Q_scaled_ginv()
    {
        arma::mat V_pos = eigvec_Q.cols(pos_eig_idx);
        arma::vec lambda_pos = eigval_Q(pos_eig_idx);
        double scale2 = scale_factor * scale_factor;
        arma::vec lambda_inv = 1.0 / (lambda_pos * scale2);
        Q_scaled_ginv = V_pos * arma::diagmat(lambda_inv) * V_pos.t();
        return;
    } // end of compute_Q_scaled_ginv()


    void compute_prec()
    {
        double scale2 = scale_factor * scale_factor;
        prec = (1.0 - phi) * arma::eye<arma::mat>(nS, nS) + phi * scale2 * Q;
        return;
    } // end of compute_prec()


    double log_det_prec()
    {
        double logdet = 0.0;
        double scale2 = scale_factor * scale_factor;
        for (unsigned int i = 0; i < pos_eig_idx.n_elem; ++i)
        {
            double lambda = eigval_Q(pos_eig_idx(i));
            logdet += std::log((1.0 - phi) + phi * scale2 * lambda);
        }
        logdet += (nS - pos_eig_idx.n_elem) * std::log(1.0 - phi); // for zero eigenvalues
        return logdet;
    } // end of log_det_prec()


    double compute_quadratic_form(const arma::vec &b)
    {
        arma::vec centered = b - mu;
        return arma::as_scalar(centered.t() * prec * centered);
    } // end of compute_quadratic_form()

    arma::vec sample_icar_component()
    {
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

    arma::mat sample_bym2_prior(const unsigned int k = 1) {
        arma::mat samples(nS, k);
        
        for (unsigned int j = 0; j < k; ++j) {
            // 1. Sample unstructured component v ~ N(0, I)
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
            
            samples.col(j) = b;
        }
        
        return samples;
    } // end of sample_bym2_prior()


}; // end of class SpatialStructureBYM2


class BYM2_MCMC_ObservedEffects {
private:
    // Observed spatial effects
    arma::vec b_observed;
    SpatialStructureBYM2& spatial;
    
    // Current parameters
    double mu;
    double tau_b;
    double phi;
    
    // Precomputed matrices for efficiency
    arma::mat Q_scaled;  // scale^2 * Q
    arma::mat Q_scaled_ginv;  // Generalized inverse of Q_scaled
    
    // PC prior hyperparameters
    double U_tau = 1.0;
    double alpha_tau = 1.0;
    double alpha_phi = 2.0/3.0;  // For PC prior on phi
    
    // MCMC settings
    double mh_sd_phi = 0.1;
    double phi_accept_count = 0.0;
    
    // Storage
    arma::vec mu_samples;
    arma::vec tau_b_samples;
    arma::vec phi_samples;
    
public:
    BYM2_MCMC_ObservedEffects(
        const arma::vec& b_obs, 
        SpatialStructureBYM2& sp
    ) : b_observed(b_obs), spatial(sp) {
        
        // Initialize parameters
        mu = arma::mean(b_observed);
        tau_b = 1.0;
        phi = 0.5;
        
        // Precompute scaled Q matrix
        double scale2 = spatial.scale_factor * spatial.scale_factor;
        Q_scaled = scale2 * spatial.Q;
        
        // Compute generalized inverse using eigendecomposition
        compute_Q_scaled_ginv();
    }
    
    void compute_Q_scaled_ginv() {
        // Use the eigendecomposition already computed in spatial
        arma::mat V_pos = spatial.eigvec_Q.cols(spatial.pos_eig_idx);
        arma::vec lambda_pos = spatial.eigval_Q(spatial.pos_eig_idx);
        
        double scale2 = spatial.scale_factor * spatial.scale_factor;
        arma::vec lambda_inv = 1.0 / (scale2 * lambda_pos);
        
        Q_scaled_ginv = V_pos * arma::diagmat(lambda_inv) * V_pos.t();
    }

    void update_mu_tau_jointly(double shape_tau = 1.0, double rate_tau = 1.0)
    {
        // Given phi, (mu, tau_b) have a Normal-Gamma posterior
        arma::mat Rphi_inv = (1.0 - phi) * arma::eye(spatial.nS, spatial.nS) + phi * Q_scaled_ginv;
        arma::mat Rphi_current = inverse(Rphi_inv);

        arma::vec ones = arma::ones(spatial.nS);
        arma::vec Rphi_b = Rphi_current * b_observed;
        arma::vec Rphi_ones = Rphi_current * ones;

        double one_Rphi_one = arma::dot(ones, Rphi_ones);
        double b_Rphi_b = arma::dot(b_observed, Rphi_b);
        double one_Rphi_b = arma::dot(ones, Rphi_b);

        // Sample tau_b first
        double quadratic_form = b_Rphi_b - (one_Rphi_b * one_Rphi_b) / one_Rphi_one;
        double shape_post = shape_tau + 0.5 * static_cast<double>(spatial.nS) - 0.5;
        double rate_post = rate_tau + 0.5 * quadratic_form;
        tau_b = R::rgamma(shape_post, 1.0 / rate_post);

        // Then sample mu | tau_b
        double post_mean = one_Rphi_b / one_Rphi_one;
        double post_var = 1.0 / (tau_b * one_Rphi_one);
        mu = R::rnorm(post_mean, std::sqrt(post_var));
    }


    // Normal prior on logit(phi)
    double log_prior_logit_phi(const double phi_val, const double mu = 0.0, const double sigma = 1.0)
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
        const double phi_val, // phi on original scale (0, 1)
        const double mu_phi = 0.0, // prior mean of logit(phi)
        const double sigma_phi = 1.0, // prior sd of logit(phi)
        const double shape_tau = 1.0, // prior shape for marginal precision tau_b
        const double rate_tau = 1.0 // prior rate for marginal precision tau_b
    )
    {
        if (phi_val <= 0.0 || phi_val >= 1.0)
            return -INFINITY;
        // Cov(b) = (1/tau_b)*[ (1-phi) I + phi * Q_scaled_ginv ]
        arma::mat Rphi_inv = (1.0 - phi_val) * arma::eye(spatial.nS, spatial.nS) + phi_val * Q_scaled_ginv;
        arma::mat Rphi = inverse(Rphi_inv);
        double logdet = arma::log_det_sympd(arma::symmatu(Rphi));

        arma::vec ones(spatial.nS, arma::fill::ones);
        arma::vec Rphi_b = Rphi * b_observed;
        arma::vec Rphi_ones = Rphi * ones;
        double one_Rphi_one = arma::dot(ones, Rphi_ones);
        double b_Rphi_b = arma::dot(b_observed, Rphi_b);
        double one_Rphi_b = arma::dot(b_observed, Rphi_ones);

        // double shape_post = shape_tau + 0.5 * static_cast<double>(spatial.nS) - 0.5;
        // double rate_post = rate_tau + 0.5 * b_Rphi_b + 0.5 * one_Rphi_b;
        double quadratic_form = b_Rphi_b - (one_Rphi_b * one_Rphi_b) / one_Rphi_one;
        double shape_post = shape_tau + 0.5 * static_cast<double>(spatial.nS) - 0.5;
        double rate_post = rate_tau + 0.5 * quadratic_form;

        // Log marginal (integrating out tau_b and mu)
        return 0.5 * logdet - shape_post * std::log(rate_post) - 0.5 * std::log(one_Rphi_one) + log_prior_logit_phi(phi_val, mu_phi, sigma_phi);
    }


    void update_phi_logit()
    {
        double logit_phi = logit(phi);
        double prop = logit_phi + R::rnorm(0.0, mh_sd_phi);
        double phi_prop = 1.0 / (1.0 + std::exp(-prop));
        double log_acc = log_post_phi(phi_prop) - log_post_phi(phi);
        if (std::log(R::runif(0, 1)) < log_acc)
        {
            phi = phi_prop;
            phi_accept_count += 1.0;
        }
    }

    void adapt_phi_proposal_robbins_monro(int iter, int burn_in, double target_rate = 0.6)
    {
        if (iter < burn_in && iter > 0 && iter % 50 == 0)
        {
            double accept_rate = phi_accept_count / 50.0;
            phi_accept_count = 0.0;

            // Robbins-Monro update
            double gamma = 1.0 / std::pow(iter / 50.0, 0.6); // Decay rate
            mh_sd_phi *= std::exp(gamma * (accept_rate - target_rate));

            // Keep in reasonable range
            mh_sd_phi = std::max(0.01, std::min(2.0, mh_sd_phi));
        }
    }


    // Main MCMC function
    Rcpp::List run_mcmc(
        int n_iter = 5000,
        int burn_in = 1000,
        int thin = 1,
        bool verbose = true
    ) {
        int n_save = (n_iter - burn_in) / thin;
        mu_samples = arma::vec(n_save);
        tau_b_samples = arma::vec(n_save);
        phi_samples = arma::vec(n_save);
        
        int save_idx = 0;
        for (int iter = 0; iter < n_iter; ++iter) {
            // Update parameters
            update_phi_logit();
            update_mu_tau_jointly();

            // Adapt proposal during burn-in
            adapt_phi_proposal_robbins_monro(iter, burn_in);
            
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
            Rcpp::Named("final_mh_sd") = mh_sd_phi
        );
    }
}; // end of class BYM2_MCMC_ObservedEffects


#endif