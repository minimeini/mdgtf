#pragma once
#ifndef _UTILS_H
#define _UTILS_H

#if !defined(ARMA_USE_BLAS)
#define ARMA_USE_BLAS
#endif

#if !defined(ARMA_USE_LAPACK)
#define ARMA_USE_LAPACK
#endif

#if !defined(ARMA_USE_64BIT_WORD)
#define ARMA_USE_64BIT_WORD
#endif

#if !defined(ARMA_USE_OPENMP)
#define ARMA_USE_OPENMP
#endif

#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <RcppArmadillo.h>
#include "definition.h"

// [[Rcpp::plugins(cpp17)]]
// [[Rcpp::depends(RcppArmadillo)]]

inline void tolower(std::string &S)
{
	for (char &x : S)
	{
		x = tolower(x);
	}
	return;
}

inline std::string tolower(const std::string &S)
{
	std::string SS = S;
	for (char &x : SS)
	{
		x = tolower(x);
	}
	return SS;
}

inline const char *const bool2string(bool b)
{
	return b ? "true" : "false";
}

template <typename T>
inline void bound_check(
	const T &input,
	const std::string &name,
	const bool &check_zero = false,
	const bool &check_negative = false)
{
	// std::type_info tname = typeid(input);
	// bool is_num = tname == typeid(int) || tname == typeid(unsigned int) || tname == typeid(double);
	bool is_infinite = !input.is_finite();
	bool is_nan = input.has_nan();
	if (is_infinite || is_nan)
	{
		// std::cout << name + " = " << std::endl;
		// input.t().print();
		throw std::invalid_argument("bound_check: " + name + " is infinite.");
	}

	if (check_zero)
	{
		bool is_zero = input.is_zero();
	}
	if (check_negative)
	{
		bool is_negative = arma::any(arma::vectorise(input) < -EPS);
	}
	return;
}

inline void bound_check(
	const double &input,
	const std::string &name,
	const bool &check_zero = false,
	const bool &check_negative = false)
{
	bool is_infinite = !std::isfinite(input);
	bool is_nan = is_infinite;
	if (is_infinite || is_nan)
	{
		// std::cout << name + " = " << std::endl;
		// input.t().print();
		throw std::invalid_argument("bound_check: " + name + " is infinite.");
	}

	if (check_zero)
	{
		bool is_zero = std::abs(input) < EPS;
		if (is_zero)
		{
			throw std::invalid_argument("bound_check: " + name + " is zero.");
		}
	}
	if (check_negative)
	{
		bool is_negative = input < -EPS;
		if (is_negative)
		{
			throw std::invalid_argument("bound_check: " + name + " is negative.");
		}
	}
}

inline bool bound_check(
	const double &input,
	const bool &check_zero,
	const bool &check_negative)
{
	bool is_infinite = !std::isfinite(input);
	bool out_of_bound = is_infinite;
	if (check_zero)
	{
		bool is_zero = std::abs(input) < EPS;
		out_of_bound = out_of_bound || is_zero;
	}
	if (check_negative)
	{
		bool is_negative = input < -EPS;
		out_of_bound = out_of_bound || is_negative;
	}

	return out_of_bound;
}

inline void bound_check(
	const double &input,
	const std::string &name,
	const arma::vec &interval)
{
	bool is_infinite = !std::isfinite(input);
	if (is_infinite)
	{
		// std::cout << name + " = " << input << std::endl;
		throw std::invalid_argument("bound_check: " + name + " = " + std::to_string(input) + " is infinite.");
	}

	double lobnd = interval.at(0);
	double upbnd = interval.at(1);

	bool out_of_lobnd = input < lobnd;
	bool out_of_upbnd = input > upbnd;
	if (out_of_lobnd || out_of_upbnd)
	{
		throw std::invalid_argument(
			"bound_check: " + name +
			" out of lobnd = " + bool2string(out_of_lobnd) +
			", out of upbnd = " + bool2string(out_of_upbnd));
	}
}

inline arma::uvec sample(
	const unsigned int n,
	const unsigned int size,
	const arma::vec &weights,
	bool replace,
	bool zero_start)
{
	Rcpp::NumericVector w_ = Rcpp::NumericVector(weights.begin(), weights.end());
	Rcpp::IntegerVector idx_ = Rcpp::sample(n, size, replace, w_);

	arma::uvec idx = Rcpp::as<arma::uvec>(Rcpp::wrap(idx_));
	if (zero_start)
	{
		idx.for_each([](arma::uvec::elem_type &val)
					 { val -= 1; });
	}

	return idx;
}

inline unsigned int sample(
	const int n,
	const arma::vec &weights,
	bool zero_start)
{
	Rcpp::NumericVector w_ = Rcpp::NumericVector(weights.begin(), weights.end());
	Rcpp::IntegerVector idx_ = Rcpp::sample(n, 1, true, w_);

	arma::uvec idx = Rcpp::as<arma::uvec>(Rcpp::wrap(idx_));
	unsigned int idx0 = idx[0];
	if (zero_start)
	{
		idx0 -= 1;
	}

	return idx0;
}

inline unsigned int sample(
	const int n,
	bool zero_start)
{
	arma::vec weights(n, arma::fill::ones);
	Rcpp::NumericVector w_ = Rcpp::NumericVector(weights.begin(), weights.end());
	Rcpp::IntegerVector idx_ = Rcpp::sample(n, 1, true, w_);

	arma::uvec idx = Rcpp::as<arma::uvec>(Rcpp::wrap(idx_));
	unsigned int idx0 = idx[0];
	if (zero_start)
	{
		idx0 -= 1;
	}

	return idx0;
}

inline arma::mat inverse(
	const arma::mat &matrice,
	const bool &force_pseudo = false,
	const bool &try_pseudo = false)
{
	arma::mat mat_inv;
	if (force_pseudo)
	{
		mat_inv = arma::pinv(matrice);
	}
	else
	{
		try
		{
			arma::mat mat_R = arma::chol(arma::symmatu(matrice));
			arma::mat mat_Rinv = arma::inv(arma::trimatu(mat_R));
			mat_inv = mat_Rinv * mat_Rinv.t();
		}
		catch (...)
		{
			if (try_pseudo)
			{
				std::cout << "\nWarning: matrice is not invertible, use pseudo inverse instead.\n\n";
				mat_inv = arma::pinv(matrice);
			}
			else
			{
				throw std::runtime_error("\nError: matrice is not invertible.");
			}
		}
	}

	return mat_inv;
}


inline double logit(const double &p, const double &m = 1.)
{
	double val = p / (1. - p);
	#ifdef DGTF_DO_BOUND_CHECK
	bound_check(val, "logit");
	#endif
	return std::log(val) * m;
}


template <typename T>
inline T logit(const T&x, const double &m = 1.)
{
	T val = x / (1. - x);
	T out = arma::log(val) * m;
	#ifdef DGTF_DO_BOUND_CHECK
	bound_check(val, "logit");
	#endif
	return out;
}

inline double logistic(const double &x, const double &m = 1.)
{
	double val = 1. / (1. + std::exp(- x / m));
	#ifdef DGTF_DO_BOUND_CHECK
	bound_check(val, "logistic", true, true);
	#endif
	return val;
}


template <typename T>
inline T logistic(const T& x, const double &m = 1.)
{
	T val = arma::exp(- x / m) + 1.;
	T out = 1. / val;
	#ifdef DGTF_DO_BOUND_CHECK
	bound_check<T>(out, "logistic", true, true);
	#endif
	return out;
}


inline double dnorm_cpp(double x, double mu, double sd, bool logd=true) {
    const double z = (x - mu) / sd;
    const double logc = -0.5*std::log(2.0*M_PI) - std::log(sd);
    double val = logc - 0.5*z*z;
    return logd ? val : std::exp(val);
}



/**
 * Calculate CRPS for posterior predictive samples
 * 
 * @param y Observations vector (nT x 1)
 * @param Y Posterior predictive samples matrix (nT x nsample)
 * @return Average CRPS over all time points
 */
inline double calculate_crps(const arma::vec& y, const arma::mat& Y) {
    arma::uword nT = y.n_elem, nsample = Y.n_cols;
    if (Y.n_rows != nT) throw std::invalid_argument("Number of rows in Y must match length of y");
    double total = 0.0;

    for (arma::uword t = 0; t < nT; ++t) {
        arma::rowvec row = Y.row(t);
        double obs = y.at(t);

        // term1 = mean |X - y|
        double term1 = arma::mean(arma::abs(row.t() - obs));

        // term2 = mean |Xi - Xj| using sorted row and linear-time formula
        arma::rowvec x = arma::sort(row);
        double sum_weighted = 0.0;
        for (arma::uword k = 0; k < nsample; ++k) {
            // with 0-based k, coefficient = 2*k - (nsample - 1)
            sum_weighted += (2.0 * static_cast<double>(k) - (static_cast<double>(nsample) - 1.0)) * x.at(k);
        }
        double mean_abs_diff = 2.0 * sum_weighted / (static_cast<double>(nsample) * static_cast<double>(nsample));

        double crps = term1 - 0.5 * mean_abs_diff;
        if (crps < 0.0) crps = 0.0; // numerical guard
        total += crps;
    }
    return total / static_cast<double>(nT);
}

inline double calculate_mae(
	const arma::vec &y_true,
	const arma::mat &Y,
	bool posterior_expected = true)
{

	// Y: k x nsample, y_true: k
	if (Y.n_rows != y_true.n_elem)
		throw std::invalid_argument("calculateMAE: Y must be k x nsample; y_true must have length k.");

	if (posterior_expected)
	{
		// E[(Y - y_true)^2 | posterior] per time, then average over time
		arma::mat diff = Y.each_col() - y_true;				 // k x nsample
		arma::vec mse_t = arma::mean(arma::square(diff), 1); // k x 1
		return arma::mean(mse_t);
	}
	else
	{
		// Squared error of posterior mean, averaged over time
		arma::vec yhat = arma::mean(Y, 1); // k x 1
		arma::vec resid = yhat - y_true;
		return arma::mean(arma::square(resid));
	}
}


inline double calculate_chisqr(
	const arma::mat& Y, 
	const arma::vec& y_true, 
	const double eps = 1e-12) {
  // Y: k x nsample posterior samples; y_true: length k
  if (Y.n_rows != y_true.n_elem)
    throw std::invalid_argument("calculate_chisqr: Y must be k x nsample; y_true must have length k.");

  // Row-wise variance across samples (per time point)
  arma::vec var_t = arma::var(Y, /*norm_type=*/1, /*dim=*/1); // length k

  // Squared standardized residuals with epsilon stabilization
  arma::mat diff = Y.each_col() - y_true;                 // k x nsample
  arma::mat num  = arma::square(arma::abs(diff) + eps);   // (|y - mean| + eps)^2
  arma::mat ratio = num.each_col() / (var_t + eps);       // divide by var_t + eps

  // Average over time, then over samples (equivalently overall mean)
  return arma::mean(arma::mean(ratio));
}


#endif