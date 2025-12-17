#pragma once
#ifndef UTILS2_H
#define UTILS2_H

#include <RcppEigen.h>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <unsupported/Eigen/SpecialFunctions>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <vector>

// [[Rcpp::plugins(cpp17)]]
// [[Rcpp::depends(RcppEigen)]]


void show_vec(const Eigen::VectorXd& v) {
    Rcpp::NumericVector rv = Rcpp::wrap(v);
    Rcpp::Rcout << rv << "\n";        // R-style print
    // or Rcpp::Rcout << v.transpose() << "\n"; // plain Eigen format
}


inline double clamp01(double w) {
  return std::min(std::max(w, 1.0e-8), 1.0 - 1.0e-8);
}


inline double inv_logit_stable(double x) {
  if (x >= 0) { double e = std::exp(-x); return 1.0 / (1.0 + e); }
  double e = std::exp(x);
  return e / (1.0 + e);
}


inline double logit_safe(double w) {
  w = clamp01(w);
  return std::log(w) - std::log1p(-w);
}


inline double clamp_log_scale(double x) {
  // tune bounds if needed
  return std::min(std::max(x, -30.0), 30.0);
}


inline Eigen::Tensor<double, 4> r_to_tensor4(Rcpp::NumericVector &arr)
{
    if (!arr.hasAttribute("dim"))
        Rcpp::stop("need dim");
    Rcpp::IntegerVector dim = arr.attr("dim");
    if (dim.size() != 4)
        Rcpp::stop("need 4D array");

    Eigen::array<Eigen::Index, 4> dims = {dim[0], dim[1], dim[2], dim[3]};
    Eigen::TensorMap<Eigen::Tensor<double, 4>> t(arr.begin(), dims);
    return t;
}; // r_to_tensor4


inline Rcpp::NumericVector tensor4_to_r(const Eigen::Tensor<double, 4> &t)
{
    Eigen::array<Eigen::Index, 4> dims = t.dimensions();
    Rcpp::NumericVector out(dims[0] * dims[1] * dims[2] * dims[3]);
    std::copy(t.data(), t.data() + out.size(), out.begin());
    out.attr("dim") = Rcpp::IntegerVector::create(
        dims[0], dims[1], dims[2], dims[3]);
    return out;
}; // tensor4_to_r


inline Rcpp::NumericVector tensor3_to_r(const Eigen::Tensor<double, 3> &t)
{
    Eigen::array<Eigen::Index, 3> dims = t.dimensions();
    Rcpp::NumericVector out(dims[0] * dims[1] * dims[2]);
    std::copy(t.data(), t.data() + out.size(), out.begin());
    out.attr("dim") = Rcpp::IntegerVector::create(dims[0], dims[1], dims[2]);
    return out;
}; // tensor3_to_r


inline Rcpp::NumericVector tensor5_to_r(const Eigen::Tensor<double, 5> &t)
{
    Eigen::array<Eigen::Index, 5> dims = t.dimensions();
    Rcpp::NumericVector out(dims[0] * dims[1] * dims[2] * dims[3] * dims[4]);
    std::copy(t.data(), t.data() + out.size(), out.begin());
    out.attr("dim") = Rcpp::IntegerVector::create(
        dims[0], dims[1], dims[2], dims[3], dims[4]);
    return out;
}; // tensor5_to_r


Eigen::VectorXd cumsum_vec(const Eigen::VectorXd &v) {
    Eigen::VectorXd out(v.size());
    std::partial_sum(v.data(), v.data() + v.size(), out.data());
    return out;
}


/**
 * @brief Cumulative sum down rows for each column
 * 
 * @param M 
 * @return Eigen::MatrixXd 
 */
Eigen::MatrixXd cumsum_rows(const Eigen::MatrixXd &M) {
    Eigen::MatrixXd out(M.rows(), M.cols());
    for (Eigen::Index j = 0; j < M.cols(); ++j) {
        std::partial_sum(M.col(j).data(),
                         M.col(j).data() + M.rows(),
                         out.col(j).data());
    }
    return out;
}


/**
 * @brief Cumulative sum across columns for each row
 * 
 * @param M 
 * @return Eigen::MatrixXd 
 */
Eigen::MatrixXd cumsum_cols(const Eigen::MatrixXd &M) {
    Eigen::MatrixXd out(M.rows(), M.cols());
    for (Eigen::Index i = 0; i < M.rows(); ++i) {
        std::partial_sum(M.row(i).data(),
                         M.row(i).data() + M.cols(),
                         out.row(i).data());
    }
    return out;
}


#endif // UTILS2_H