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
inline double tYJ(const double theta, const double gamma)
{
    const double sgn = (theta < 0.) ? -1. : 1.;
    const double gmt = (theta < 0.) ? (2. - gamma) : gamma;
    #ifdef DGTF_DO_BOUND_CHECK
        bound_check(gmt, "tYJ: gmt", false, true);
    #endif

    const double a = std::abs(theta);
    double nu;
    if (gmt < EPS8) {
        // log1p(a)
        nu = std::log1p(a);
    } else {
        // ( (1+a)^gmt - 1 ) / gmt  -> expm1(gmt*log1p(a))/gmt
        nu = std::expm1(gmt * std::log1p(a)) / gmt;
    }
    nu *= sgn;

    #ifdef DGTF_DO_BOUND_CHECK
        bound_check(nu, "tYJ: nu");
    #endif
    return nu;
}

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
inline double tYJinv(const double nu, const double gamma)
{
    const double sgn = (nu < 0.) ? -1. : 1.;
    const double gmt = (nu < 0.) ? (2. - gamma) : gamma;
    #ifdef DGTF_DO_BOUND_CHECK
        bound_check(nu, "tYJinv: nu");
        bound_check(gmt, "tYJinv: gmt", false, true);
    #endif

    double theta;
    if (gmt < EPS8) {
        const double a = std::min(std::abs(nu), UPBND);
        theta = std::expm1(a); // exp(a) - 1
    } else {
        const double a = std::abs(gmt * nu);
        // (1+a)^(1/gmt) - 1  -> expm1(log1p(a)/gmt)
        theta = std::expm1(std::log1p(a) / gmt);
    }
    theta *= sgn;

    #ifdef DGTF_DO_BOUND_CHECK
        bound_check(theta, "tYJinv: theta");
    #endif
    return theta;
}

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
inline arma::vec gamma2tau(const arma::vec &gamma)
{
    return arma::log(gamma + EPS) - arma::log(2.0 - gamma + EPS);
} // Status: Checked. OK.

// Transform tau in real line to (0,2), denoted by gamma
// Ref: `tau2eta.m`
// inline double tau2gamma(const double tau)
// {
//     double neg_tau = std::min(-tau, UPBND);
//     double output = 2. / (std::exp(neg_tau) + 1.);
//     #ifdef DGTF_DO_BOUND_CHECK
//         bound_check(output, "tau2gamma: gamma", 0., 2.);
//     #endif
//     return output;
// } // Status: Checked. OK.

// inline arma::vec tau2gamma(const arma::vec &tau)
// { // m x 1
//     const unsigned int m = tau.n_elem;
//     arma::vec gamma(m);
//     for (unsigned int i = 0; i < m; i++)
//     {
//         gamma.at(i) = tau2gamma(tau.at(i));
//     }
//     return gamma;
// } // Status: Checked. OK.

inline void tau2gamma(const arma::vec &tau, arma::vec &gamma_out)
{
    if (gamma_out.is_empty() || gamma_out.n_elem != tau.n_elem)
    {
        gamma_out.set_size(tau.n_elem);
    }
    arma::vec clamped = arma::clamp(tau, -UPBND, arma::datum::inf);
    gamma_out = 2.0 / (1.0 + arma::exp(-clamped));
    #ifdef DGTF_DO_BOUND_CHECK
    bound_check<arma::vec>(gamma_out, "tau2gamma");
    #endif
}


// (Element-wise)
// First-order derivative of gamma with respect to tau
// given the value of tau
// Ref: `deta_dtau.m`
inline double dgamma_dtau_tau(const double tau)
{
    double etau = std::min(tau, UPBND);
    etau = std::exp(etau);
    double output = 2. * etau / std::pow(1. + etau, 2.);
    #ifdef DGTF_DO_BOUND_CHECK
        bound_check(output, "dgamma_dtau_tau");
    #endif
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
        output = - std::log1p(th_abs);
    }
    else
    {
        output = std::log1p(th_abs);
        output *= sgn * (gamma - 1.);
    }

    if (!log)
    {
        output = std::min(output,UPBND);
        output = std::exp(output);
    }
    
    #ifdef DGTF_DO_BOUND_CHECK
        bound_check(output, "dtYJ_dtheta");
    #endif
    return output;
} // Status: Checked. OK.


inline arma::vec dtYJ_dtheta_diag(const arma::vec& theta, const arma::vec& gamma, bool log=false) {
    const arma::uword m = theta.n_elem;
    arma::vec v(m);
    for (arma::uword i = 0; i < m; ++i) {
        v[i] = dtYJ_dtheta(theta[i], gamma[i], log);
    }
    return v;
}

/*
Loaiza-Maya et al., PDF, bottom of p26
*/
inline double dtYJ_dgamma(const double theta, const double gamma)
{
    const double sgn = (theta < 0.) ? -1. : 1.;
    const double gmt = (theta < 0.) ? (2. - gamma) : gamma;
    #ifdef DGTF_DO_BOUND_CHECK
        bound_check(gmt, "dtYJ_dgamma", false, true);
    #endif

    if (gmt < EPS8) return 0.0;

    const double a = std::abs(theta);
    // ((1+a)^gmt) * (log(1+a)*gmt - 1) + 1
    const double t1 = std::exp(gmt * std::log1p(a));
    const double num = t1 * (std::log1p(a) * gmt - 1.0) + 1.0;
    const double g2inv = 1.0 / (gmt * gmt); // pow(gmt,-2)
    double res = num * g2inv;

    #ifdef DGTF_DO_BOUND_CHECK
        bound_check(res, "dtYJ_dgamma: res");
    #endif
    return res;
}

// Ref: line 24-27 of `grad_theta_logq.m`
inline arma::vec dlogdYJ_dtheta(const arma::vec& theta, const arma::vec& gamma) {
    return (gamma - 1.0) / (1.0 + arma::abs(theta));
}

/*
First order derivatives of YJinv(theta) with respect to nu
Correspond to `./Derivatives/dtheta_dphi.m`
*/
inline double dYJinv_dnu(const double nu, const double gamma)
{
    double gmt = (nu < 0.) ? (2. - gamma) : gamma;
    #ifdef DGTF_DO_BOUND_CHECK
        bound_check(gmt, "dYJinv_dnu: gmt", false, true);
    #endif

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
    
    #ifdef DGTF_DO_BOUND_CHECK
        bound_check(res, "dtYJ_dnu: res");
    #endif
    return res;
} // Status: Checked. OK.

inline arma::vec dYJinv_dnu_diag(const arma::vec& nu, const arma::vec& gamma) {
    const arma::uword m = nu.n_elem;
    arma::vec v(m);
    for (arma::uword i = 0; i < m; ++i) {
        v[i] = dYJinv_dnu(nu[i], gamma[i]);
    }
    return v;
}

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
    const double gmt = (nu < 0.) ? (2. - gamma) : gamma;
    #ifdef DGTF_DO_BOUND_CHECK
        bound_check(gmt, "dYJinv_dgamma: gmt", false, true);
    #endif
    if (std::abs(gmt) < EPS8) return 0.0;

    const double a = std::abs(gmt * nu);
    const double g2inv = 1.0 / (gmt * gmt);
    // [log1p(a) + gmt*nu/(1+a)] * gmt^-2 * -(1+a)^(1/gmt)
    const double bracket = std::log1p(a) + (gmt * nu) / (1.0 + a);
    const double expo = std::exp(std::log1p(a) / gmt);
    double res = - bracket * g2inv * expo;

    #ifdef DGTF_DO_BOUND_CHECK
        bound_check(res, "dYJinv_dgamma: res");
    #endif
    return res;
}

// Ref: `dtheta_dtau.m`
inline double dYJinv_dtau(const double nu, const double gamma)
{
    return dYJinv_dgamma(nu, gamma) * dgm_dtau_gm(gamma);
} // Status: Checked. OK.

inline arma::vec dYJinv_dtau_diag(const arma::vec& nu, const arma::vec& gamma) {
    const arma::uword m = nu.n_elem;
    arma::vec v(m);
    for (arma::uword i = 0; i < m; ++i) {
        v[i] = dYJinv_dtau(nu[i], gamma[i]);
    }
    return v;
}


inline arma::mat get_sigma_inv(
    const arma::mat &B, // m x k
    const arma::vec &d,
    const unsigned int k)
{
    arma::vec dinv = 1. / arma::square(d);
    arma::mat Dm2 = arma::diagmat(dinv); // m x m, D^{-2}
    arma::mat tmp = B.t() * Dm2 * B; // k x k
    tmp.diag() += 1.0;
    arma::mat SigInv = Dm2 - Dm2 * B * tmp.i() * B.t() * Dm2; // Woodbury formula
    
    #ifdef DGTF_DO_BOUND_CHECK
        bound_check<arma::mat>(SigInv, "get_sigma_inv: SigInv");
    #endif
    return arma::symmatu(SigInv);
}


// Ref: `grad_theta_logq.m`
inline arma::vec dlogq_dtheta(
    const arma::mat &SigInv,
    const arma::vec &nu,
    const arma::vec &theta,
    const arma::vec &gamma,
    const arma::vec &mu)
{
    // v = SigInv * (nu - mu)
    arma::vec v = SigInv * (nu - mu);
    // element-wise scale by diagonal of dtYJ_dtheta
    arma::vec d1 = dtYJ_dtheta_diag(theta, gamma, false);
    arma::vec deriv = -(d1 % v);
    arma::vec deriv2 = dlogdYJ_dtheta(theta, gamma);
    return deriv + deriv2;
}


inline void rtheta(
    arma::vec &nu,
    arma::vec &theta,
    arma::vec &xi,          // k x 1 (output)
    arma::vec &eps,         // m x 1 (output)
    const arma::vec &gamma, // m x 1
    const arma::vec &mu,    // m x 1
    const arma::mat &B,     // m x k
    const arma::vec &d)     // m x 1
{
    const arma::uword m = gamma.n_elem;
    const arma::uword k = B.n_cols;

    // ensure sizes; avoids realloc per call if already sized
    if (xi.n_elem != k) xi.set_size(k);
    if (eps.n_elem != m) eps.set_size(m);
    if (nu.n_elem != m)  nu.set_size(m);
    if (theta.n_elem != m) theta.set_size(m);

    // in-place RNG (faster than assigning arma::randn(k,1))
    xi.randn();     // ~ N(0,I_k)
    eps.randn();    // ~ N(0,I_m)

    // nu = mu + B*xi + d%eps  (BLAS gemv + two axpy)
    nu = mu;
    nu += B * xi;
    nu += d % eps;

    // elementwise inverse transform
    theta = tYJinv(nu, gamma);
}


inline arma::mat rtheta_batch(
    const arma::vec &gamma, // m
    const arma::vec &mu,    // m
    const arma::mat &B,     // m x k
    const arma::vec &d,     // m
    const arma::uword S)
{
    const arma::uword m = gamma.n_elem;
    const arma::uword k = B.n_cols;

    arma::mat Theta(m, S, arma::fill::zeros);
    arma::mat Nu(m, S, arma::fill::zeros);
    arma::mat Xi(k, S, arma::fill::randn);
    arma::mat Eps(m, S, arma::fill::randn);

    // Nu = mu*1^T + B*Xi + (d*1^T) % Eps
    Nu.each_col() = mu;
    Nu += B * Xi;                       // gemm
    Nu += arma::repmat(d, 1, S) % Eps;  // broadcast d

    // Columnwise transform (nu_i -> theta_i)
    for (arma::uword s = 0; s < S; ++s)
        Theta.col(s) = tYJinv(Nu.col(s), gamma);
    
    return Theta;
}

#endif