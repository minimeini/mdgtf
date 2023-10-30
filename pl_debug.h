#ifndef _PL_DEBUG_H
#define _PL_DEBUG_H

#include "lbe_poisson.h"

Rcpp::List mcs_poisson0(
    const arma::vec &Y, // n x 1, the observed response
    const arma::uvec &model_code,
    const double W,
    const double rho,
    const unsigned int L, // number of lags
    const double mu0,
    const unsigned int B, // length of the B-lag fixed-lag smoother (Anderson and Moore 1979; Kitagawa and Sato)
    const unsigned int N, // number of particles
    const Rcpp::Nullable<Rcpp::NumericVector> &m0_prior,
    const Rcpp::Nullable<Rcpp::NumericMatrix> &C0_prior,
    const double theta0_upbnd,
    const Rcpp::NumericVector &qProb,
    const double delta_nb,
    const double delta_discount,
    const bool verbose,
    const bool debug);

#endif