#ifndef _YJTRANS_H
#define _YJTRANS_H

#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <algorithm>
#include <RcppArmadillo.h>

// [[Rcpp::depends(RcppArmadillo)]]

/*
This script is adapted from Loaiza-Maya's Matlalog_det_sympd code.
Below is a reference regarding notations in this script versus notations in Loaiza-Maya's Matlab code and manuscript.

Core VB function is Loaiza-Maya's code: `VB_step.m` and `gradient_compute.m`

theta
    - vector includes all unknown parameters to be learned by HVB.
    - These parameters are already mapped to the real line but before YJ transform.
    - In our code, theta = YJinv(nu) and sometimes refer to as `YJinv`
    - In our code, it is also refer to as `eta_tilde`
nu or phi
    - vector of all unknown parameters to be learned by HVB after YJ transform.
    - Yeo-Johnson transfrom: theta -> nu, i.e., nu = tYJ(theta)
    - `nu` is used in Loaiza-Maya's manuscript.
    - Refer to as `phi` in Loaiza-Maya's Matlab code.
vech
    - vectorization of non-zero elements.
m or q
    - number of unknown parameters to be learned by HVB.
    - defined as `m` in Loaiza-Maya's manuscript.
    - defined as `q` in Loaiza-Maya's Matlab code.
k or p
    - reduced dimension of unknown parameters via latent factor structure.
    - defined as `k` in Loaiza-Maya's manuscript.
    - defined as `p` in Loaiza-Maya's Matlab code

*/

/*
Yeo-Johnson Transformation

- Main Function
- Inverse Transformation
- Gradients
*/


/*
Yeo-Johnson Transform
Ref: `tYJ.m`
*/
inline double tYJ(
    const double theta,
    const double gamma)
{
    double sgn = (theta < 0.) ? -1. : 1.;
    double gmt = (theta < 0.) ? (2. - gamma) : gamma;
    bound_check(gmt, "tYJ: gmt", false, true);

    double nu;
    double tmp = std::abs(theta);
    if (gmt < EPS8)
    {
        nu = std::log(tmp + 1.);
    }
    else
    {
        nu = std::pow(tmp + 1., gmt) - 1.;
        nu /= gmt;
    }
    nu *= sgn;

    bound_check(nu,"tYJ: nu");
    return nu;
} // Status: Checked. OK.

inline arma::vec tYJ(
    const arma::vec &theta, // m x 1
    const arma::vec &gamma)
{ // m x 1
    const unsigned int m = theta.n_elem;
    arma::vec nu(m);
    for (unsigned int i = 0; i < m; i++)
    {
        nu.at(i) = tYJ(theta.at(i), gamma.at(i));
    }

    return nu;
} // Status: Checked. OK.

/*
Inverse of Yeo-Johnson Transformation
Ref: `tYJi.m`
*/
inline double tYJinv(
    const double nu,
    const double gamma)
{
    bound_check(nu,"tYJinv: nu");
    double sgn = (nu < 0.) ? -1. : 1.;
    double gmt = (nu < 0.) ? (2. - gamma) : gamma;
    bound_check(gmt, "tYJinv: gmt", false, true);

    double tmp, theta;
    if (gmt < EPS8)
    {
        tmp = std::abs(nu);
        tmp = std::min(tmp,UPBND);
        theta = std::exp(tmp) - 1.;
    }
    else
    {
        tmp = std::abs(nu*gmt);
        theta = std::pow(1. + tmp, 1. / gmt) - 1.;
    }
    theta *= sgn;

    bound_check(theta,"tYJinv: " + std::to_string(gmt < EPS8) + " theta");
    return theta;
} // Status: Checked. OK.

inline arma::vec tYJinv(
    const arma::vec &nu, // m x 1
    const arma::vec &gamma)
{ // m x 1
    const unsigned int m = nu.n_elem;
    arma::vec theta(m);
    for (unsigned int i = 0; i < m; i++)
    {
        theta.at(i) = tYJinv(nu.at(i), gamma.at(i));
    }
    return theta;
} // Status: Checked. OK.

// Transform gamma in (0,2) to the real line, denoted by tau
// Ref: `eta2tau.m`
inline double gamma2tau(const double gamma)
{
    double output = std::log(gamma + EPS);
    output -= std::log(2. - gamma + EPS);
    bound_check(output,"gamma2tau");
    return output;
} // Status: Checked. OK.

inline arma::vec gamma2tau(const arma::vec &gamma)
{
    const unsigned int m = gamma.n_elem;
    arma::vec tau(m);
    for (unsigned int i = 0; i < m; i++)
    {
        tau.at(i) = gamma2tau(gamma.at(i));
    }
    return tau;
} // Status: Checked. OK.

// Transform tau in real line to (0,2), denoted by gamma
// Ref: `tau2eta.m`
inline double tau2gamma(const double tau)
{
    double neg_tau = std::min(-tau, UPBND);
    double output = 2. / (std::exp(neg_tau) + 1.);
    bound_check(output,"tau2gamma: gamma", 0., 2.);
    return output;
} // Status: Checked. OK.

inline arma::vec tau2gamma(const arma::vec &tau)
{ // m x 1
    const unsigned int m = tau.n_elem;
    arma::vec gamma(m);
    for (unsigned int i = 0; i < m; i++)
    {
        gamma.at(i) = tau2gamma(tau.at(i));
    }
    return gamma;
} // Status: Checked. OK.

// (Element-wise)
// First-order derivative of gamma with respect to tau
// given the value of tau
// Ref: `deta_dtau.m`
inline double dgamma_dtau_tau(const double tau)
{
    double etau = std::min(tau, UPBND);
    etau = std::exp(etau);
    double output = 2. * etau / std::pow(1. + etau, 2.);
    bound_check(output,"dgamma_dtau_tau");
    return output;
} // Status: Checked. OK.

// (Element-wise)
// First-order derivative of gamma with respect to tau
// given the value of gamma
inline double dgm_dtau_gm(const double gamma)
{
    return 0.5 * gamma * (2. - gamma);
}

/*
First-order derivative of `nu = tYJ(theta;gamma)` with respect to theta
given the value of theta and gamma

Ref: `./Derivatives/dtYJ_dtheta.m`
*/
inline double dtYJ_dtheta(
    const double theta,
    const double gamma,
    const bool log = false)
{
    double output;
    double th_abs = std::abs(theta);
    double sgn = (theta < 0.) ? -1 : 1;
    if ((std::abs(gamma) < EPS8) || (std::abs(2 - gamma) < EPS8))
    {
        output = - std::log(th_abs + 1.);
    }
    else
    {
        output = std::log(th_abs + 1.);
        output *= sgn * (gamma - 1.);
    }

    if (!log)
    {
        output = std::min(output,UPBND);
        output = std::exp(output);
    }
    
    bound_check(output,"dtYJ_dtheta");
    return output;
} // Status: Checked. OK.

inline arma::mat dtYJ_dtheta(
    const arma::vec &theta, // m x 1
    const arma::vec &gamma)
{ // m x 1

    const unsigned int m = theta.n_elem;
    arma::mat grad(m, m, arma::fill::zeros);
    for (unsigned int i = 0; i < m; i++)
    {
        grad.at(i, i) = dtYJ_dtheta(theta.at(i), gamma.at(i), false);
    }
    return grad;
} // Status: Checked. OK.

/*
Loaiza-Maya et al., PDF, bottom of p26
*/
inline double dtYJ_dgamma(const double theta, const double gamma)
{
    double sgn = (theta < 0.) ? -1. : 1.;
    double gmt = (theta < 0.) ? (2. - gamma) : gamma;
    bound_check(gmt, "dtYJ_dgamma", false,true);

    double res;
    if (gmt < EPS8)
    {
        res = 0.;
    }
    else
    {
        double th_abs = std::abs(theta);
        res = std::pow(1. + th_abs, gmt);
        res *= std::log(1. + th_abs) * gmt - 1.;
        res += 1.;
        res *= std::pow(gmt, -2.);
    }
    
    bound_check(res,"dtYJ_dgamma: res");
    return res;
} // Check -- Correct

// Ref: line 24-27 of `grad_theta_logq.m`
inline double dlogdYJ_dtheta(const double theta, const double gamma)
{
    double output = (gamma - 1.) / (1. + std::abs(theta));
    bound_check(output,"dlogdYJ_dtheta");
    return output;
}

// Element-wise
inline arma::vec dlogdYJ_dtheta(
    const arma::vec &theta, // m x 1
    const arma::vec &gamma)
{ // m x 1
    const unsigned int m = theta.n_elem;
    arma::vec deriv(m);
    for (unsigned int i = 0; i < m; i++)
    {
        deriv.at(i) = dlogdYJ_dtheta(theta.at(i), gamma.at(i));
    }
    return deriv;
}

/*
First order derivatives of YJinv(theta) with respect to nu
Correspond to `./Derivatives/dtheta_dphi.m`
*/
inline double dYJinv_dnu(const double nu, const double gamma)
{
    double gmt = (nu < 0.) ? (2. - gamma) : gamma;
    bound_check(gmt, "dYJinv_dnu: gmt", false, true);

    double res, tmp;
    if (gmt < EPS8)
    {
        tmp = std::abs(nu);
        tmp = std::min(tmp, UPBND);
        res = std::exp(tmp);
    }
    else
    {
        tmp = std::abs(gmt * nu);
        double pow = (1. / gmt) - 1.;
        res = std::pow(1. + tmp, pow);
    }
    bound_check(res, "dtYJ_dnu: res");
    return res;
} // Status: Checked. OK.

inline arma::mat dYJinv_dnu(
    const arma::vec &nu, // m x 1
    const arma::vec &gamma)
{ // m x 1
    const unsigned int m = nu.n_elem;
    arma::mat grad(m, m, arma::fill::zeros);
    for (unsigned int i = 0; i < m; i++)
    {
        grad.at(i, i) = dYJinv_dnu(nu.at(i), gamma.at(i));
    }
    return grad;
} // Status: Checked. OK.

/*
--- dYJinv_dgamma ---
First order derivatives of YJinv with respect to gamma
Corresponds to `./Derivatives/dtheta_deta.m`

--- Note the following equivalency ---
c = 10
gm = 0.5
nu = tYJ_gm(c,gm)

dYJinv_dgamma(nu,gm)
-dYJ_dgamma(c,gm)/dYJ_dc(c,gm) 
--- Note the following equivalency ---
*/
inline double dYJinv_dgamma(const double nu, const double gamma)
{
    double gmt = (nu < 0.) ? (2. - gamma) : gamma;
    bound_check(gmt, "dYJinv_dgamma: gmt", false, true);

    double res;
    if (std::abs(gmt) < EPS8)
    {
        res = 0.;
    }
    else 
    {
        double tmp = std::abs(gmt * nu);
        res = std::log(1. + tmp) + gmt * nu / (1. + tmp);
        res *= std::pow(gmt, -2.);
        res *= -std::pow(1. + tmp, 1. / gmt);
    }

    bound_check(res, "dYJinv_dgamma: res");
    return res;
} // Status: Checked. OK.

// Ref: `dtheta_dtau.m`
inline double dYJinv_dtau(const double nu, const double gamma)
{
    return dYJinv_dgamma(nu, gamma) * dgm_dtau_gm(gamma);
} // Status: Checked. OK.

inline arma::mat dYJinv_dtau(
    const arma::vec &nu, // m x 1
    const arma::vec &gamma)
{ // m x 1
    const unsigned int m = nu.n_elem;
    arma::mat grad(m, m, arma::fill::zeros);
    for (unsigned int i = 0; i < m; i++)
    {
        grad.at(i, i) = dYJinv_dtau(nu.at(i), gamma.at(i));
    }
    return grad;
} // Status: Checked. OK.


/*
Appendix B.2 equation (ii)
*/
inline arma::mat dYJinv_dB(
    const arma::vec &nu,    // m x 1
    const arma::vec &gamma, // m x 1
    const arma::vec &xi)    // k x 1
{ // k x 1

    const unsigned int m = nu.n_elem;

    arma::mat dtheta_dnu = dYJinv_dnu(nu, gamma); // m x m
    arma::mat Im(m, m, arma::fill::eye);
    arma::mat dnu_dB = arma::kron(xi.t(),Im); // (1 x k) and (m x m)
    arma::mat dtheta_dB = dtheta_dnu * dnu_dB; // m x mk

    return dtheta_dB; // m x mk
} // Status: Checked. OK.

/*
Section B.2 equation (iii)
Ref: `dtheta_dBDelta.m`
*/
inline arma::mat dYJinv_dD(
    const arma::vec &nu,    // m x 1
    const arma::vec &gamma, // m x 1
    const arma::vec &eps)
{ // m x 1

    arma::mat dtheta_dnu = dYJinv_dnu(nu, gamma);     // m x m
    arma::mat dnu_dd = arma::diagmat(eps);
    arma::mat dtheta_dd = dtheta_dnu * dnu_dd; // m x m

    return dtheta_dd;
} // Status: Checked. OK.

inline arma::mat get_sigma_inv(
    const arma::mat &B, // m x k
    const arma::vec &d,
    const unsigned int k)
{
    arma::vec dinv = 1. / arma::pow(d,2.);
    arma::mat Dm2 = arma::diagmat(dinv); // m x m, D^{-2}
    arma::mat Ik(k, k, arma::fill::eye);
    arma::mat tmp = Ik + B.t() * Dm2 * B;                     // k x k
    arma::mat SigInv = Dm2 - Dm2 * B * tmp.i() * B.t() * Dm2; // Woodbury formula
    SigInv = arma::symmatu(SigInv);                           // m x m
    bound_check(SigInv,"get_sigma_inv: SigInv");
    return SigInv;
}



// Ref: `grad_theta_logq.m`
inline arma::vec dlogq_dtheta(
    const arma::mat &SigInv, // m x m
    const arma::vec &nu,     // m x 1
    const arma::vec &theta,  // m x 1
    const arma::vec &gamma,  // m x 1
    const arma::vec &mu)
{ // m x 1

    arma::mat dnu_dtheta = dtYJ_dtheta(theta, gamma);       // m x m
    arma::vec deriv = -dnu_dtheta.t() * SigInv * (nu - mu); // m x 1
    arma::vec deriv2 = dlogdYJ_dtheta(theta, gamma);        // m x 1
    arma::vec output = deriv + deriv2;
    bound_check(output,"dlogq_dtheta: output");
    return output;                                  // m x 1
} // Status: Checked. OK.

inline void rtheta(
    arma::vec &nu,
    arma::vec &theta,
    arma::vec &xi,          // k x 1
    arma::vec &eps,         // m x 1
    const arma::vec &gamma, // m x 1
    const arma::vec &mu,    // m x 1
    const arma::mat &B,     // m x k
    const arma::vec &d)
{ // m x 1

    const unsigned int m = gamma.n_elem;
    const unsigned int k = B.n_cols;

    xi = arma::randn(k, 1);
    eps = arma::randn(m, 1);

    nu = mu + B * xi + d % eps;
    theta = tYJinv(nu, gamma);
    return; // m x 1, static parameters
}



/**
 * Logarithm of the proposal density
*/
inline double logq0(
    const arma::vec &nu,        // m x 1
    const arma::vec &eta_tilde, // m x 1
    const arma::vec &gamma,     // m x 1
    const arma::vec &mu,        // m x 1
    const arma::mat &SigInv,    // m x m
    const unsigned int m)
{
    const double m_ = static_cast<double>(m);
    double logq = -0.5 * m_ * std::log(2 * arma::datum::pi);
    logq += 0.5 * arma::log_det_sympd(arma::symmatu(SigInv));
    logq -= 0.5 * arma::as_scalar((nu.t() - mu.t()) * SigInv * (nu - mu));

    for (unsigned int i = 0; i < m; i++)
    {
        logq += dtYJ_dtheta(eta_tilde.at(i), gamma.at(i), true);
    }
    bound_check(logq,"logq0");
    return logq;
}

#endif