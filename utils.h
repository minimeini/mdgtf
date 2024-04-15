#pragma once
#ifndef _UTILS_H
#define _UTILS_H

#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <RcppArmadillo.h>
#include "definition.h"

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
			arma::mat mat_R = arma::chol(matrice);
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

inline arma::mat inverse(
	arma::mat &Rchol,
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
			Rchol = arma::chol(matrice);
			arma::mat mat_Rinv = arma::inv(arma::trimatu(Rchol));
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

inline void set_seed(double seed)
{
	Rcpp::Environment base_env("package:base");
	Rcpp::Function set_seed_r = base_env["set.seed"];
	set_seed_r(std::floor(std::fabs(seed)));
}

inline arma::vec randdraw(double d, int n)
{
	set_seed(d); // Set a seed for R's RNG library
	// Call Armadillo's RNG procedure that references R's RNG capabilities
	// and change dispersion slightly.
	arma::vec out = std::sqrt(std::fabs(d)) * arma::randn(n);
	return out;
}

inline void init_param(bool &infer, double &init, Dist &prior, const Rcpp::List &opts)
{
	Rcpp::List param_opts = opts;

	if (param_opts.containsElementNamed("infer"))
	{
		infer = Rcpp::as<bool>(param_opts["infer"]);
	}

	if (param_opts.containsElementNamed("init"))
	{
		init = Rcpp::as<double>(param_opts["init"]);
	}

	std::string prior_name = "invgamma";
	if (param_opts.containsElementNamed("prior_name"))
	{
		prior_name = Rcpp::as<std::string>(param_opts["prior_name"]);
		tolower(prior_name);
	}

	Rcpp::NumericVector param = {0.01, 0.01};
	if (param_opts.containsElementNamed("prior_param"))
	{
		param = Rcpp::as<Rcpp::NumericVector>(param_opts["prior_param"]);
	}

	prior.init(prior_name, param[0], param[1]);
	prior.init_param(infer, init);
}

inline void init_dist(Dist &dist, const Rcpp::List &opts)
{
	Rcpp::List param_opts = opts;

	std::string dist_name = "invgamma";
	if (param_opts.containsElementNamed("prior_name"))
	{
		dist_name = Rcpp::as<std::string>(param_opts["prior_name"]);
	}
	else if (param_opts.containsElementNamed("name"))
	{
		dist_name = Rcpp::as<std::string>(param_opts["name"]);
	}
	tolower(dist_name);

	Rcpp::NumericVector param = {0.01, 0.01};
	if (param_opts.containsElementNamed("prior_param"))
	{
		param = Rcpp::as<Rcpp::NumericVector>(param_opts["prior_param"]);
	}
	else if (param_opts.containsElementNamed("param"))
	{
		param = Rcpp::as<Rcpp::NumericVector>(param_opts["param"]);
	}

	dist.init(dist_name, param[0], param[1]);
}

inline void init_prior(Prior &prior, const Rcpp::List &opts)
{
	Rcpp::List param_opts = opts;

	std::string prior_name = "invgamma";
	if (param_opts.containsElementNamed("prior_name"))
	{
		prior_name = Rcpp::as<std::string>(param_opts["prior_name"]);
		tolower(prior_name);
	}

	Rcpp::NumericVector param = {1, 1};
	if (param_opts.containsElementNamed("prior_param"))
	{
		param = Rcpp::as<Rcpp::NumericVector>(param_opts["prior_param"]);
	}

	prior.init(prior_name, param[0], param[1]);

	bool infer = false;
	if (param_opts.containsElementNamed("infer"))
	{
		infer = Rcpp::as<bool>(param_opts["infer"]);
	}

	double init = 0.;
	if (param_opts.containsElementNamed("init"))
	{
		init = Rcpp::as<double>(param_opts["init"]);
	}

	prior.init_param(infer, init);

	return;
}




#endif