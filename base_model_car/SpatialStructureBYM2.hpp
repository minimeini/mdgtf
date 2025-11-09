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
    }

    void compute_prec()
    {
        double scale2 = scale_factor * scale_factor;
        prec = (1.0 - phi) * arma::eye<arma::mat>(nS, nS) + phi * scale2 * Q;
        return;
    }

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
    }

    double compute_quadratic_form(const arma::vec &b)
    {
        arma::vec centered = b - mu;
        return arma::as_scalar(centered.t() * prec * centered);
    }

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
    }
}; // end of class SpatialStructureBYM2


#endif