#ifndef _YJTRANS_H
#define _YJTRANS_H

#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <algorithm>
#include "armadillo"

/*
This script is adapted from Loaiza-Maya's Matlalog_det_sympd code.
Below is a reference regarding notations in this script versus notations in Loaiza-Maya's Matlab code and manuscript.

Core VB function is Loaiza-Maya's code: `VB_step.m` and `gradient_compute.m`

theta
    - vector includes all unknown parameters to be learned by HVB.
    - These parameters are already mapped to the real line.
    - In our code, theta = YJinv(nu) and sometimes refer to as `YJinv`
    - In our code, it is also refer to as `eta_tilde`
nu or phi
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

arma::vec mat2vech(const arma::mat &lotri)
{                                                            // m x k
    return lotri.elem(arma::trimatl_ind(arma::size(lotri))); // dim=m*k-(k-1)*k/2
}

/*
Yeo-Johnson Transform
Ref: `tYJ.m`
*/
double tYJ(
    const double theta,
    const double gamma)
{
    double nu;
    if (theta < arma::datum::eps && std::abs(gamma - 2.) > arma::datum::eps)
    {
        // when theta/nu < 0 and gamma != 2
        nu = -(std::pow(-theta + 1., 2. - gamma) - 1.) / (2. - gamma);
    }
    else if (theta < arma::datum::eps)
    {
        // when theta < 0 and gamma == 2
        nu = -std::log(-theta + 1.);
    }
    else if (std::abs(gamma) > arma::datum::eps)
    {
        // when theta >= 0 and gamma != 0
        nu = (std::pow(theta + 1., gamma) - 1.) / gamma;
    }
    else
    {
        // when theta >= 0 and gamma == 0
        nu = std::log(theta + 1.);
    }
    return nu;
} // Status: Checked. OK.

//' @export
// [[Rcpp::export]]
arma::vec tYJ(
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
double tYJinv(
    const double nu,
    const double gamma)
{
    double theta, gmt;
    if (nu < arma::datum::eps && std::abs(gamma - 2.) > arma::datum::eps)
    {
        // when theta/nu < 0 and gamma != 2
        gmt = 2. - gamma;
        theta = 1. - std::pow(1. - gmt * nu, 1. / gmt);
    }
    else if (nu < arma::datum::eps)
    {
        // when theta/nu < 0 and gamma == 2
        theta = 1. - std::exp(-nu);
    }
    else if (std::abs(gamma) > arma::datum::eps)
    {
        // when theta/nu >= 0 and gamma != 0
        gmt = gamma;
        theta = std::pow(1. + gmt * nu, 1. / gmt) - 1.;
    }
    else
    {
        // when theta/nu >= 0 and gamma == 0
        theta = std::exp(nu) - 1.;
    }
    return theta;
} // Status: Checked. OK.

arma::vec tYJinv(
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
double gamma2tau(const double gamma)
{
    return -std::log(2. / gamma - 1.);
} // Status: Checked. OK.

arma::vec gamma2tau(const arma::vec &gamma)
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
double tau2gamma(const double tau)
{
    return 2. / (std::exp(-tau) + 1.);
} // Status: Checked. OK.

arma::vec tau2gamma(const arma::vec &tau)
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
double dgamma_dtau_tau(const double tau)
{
    double etau = std::exp(tau);
    return 2. * etau / std::pow(1. + etau, 2.);
} // Status: Checked. OK.

// (Element-wise)
// First-order derivative of gamma with respect to tau
// given the value of gamma
double dgm_dtau_gm(const double gamma)
{
    return 0.5 * gamma * (2. - gamma);
}

/*
First-order derivative of `nu = tYJ(theta;gamma)` with respect to theta
given the value of theta and gamma

Ref: `./Derivatives/dtYJ_dtheta.m`
*/
double dtYJ_dtheta(
    const double theta,
    const double gamma,
    const bool log = false)
{
    double gamma_ = gamma;
    if (std::abs(gamma_) < 0.0000001)
    {
        gamma_ = 0.0000001;
    }
    if (theta < arma::datum::eps)
    {
        if (log)
        {
            return (1. - gamma_) * std::log(-theta + 1.);
        }
        else
        {
            return std::pow(-theta + 1., 1. - gamma_);
        }
    }
    else
    {
        if (log)
        {
            return (gamma_ - 1.) * std::log(theta + 1.);
        }
        else
        {
            return std::pow(theta + 1., gamma_ - 1.);
        }
    }
} // Status: Checked. OK.

arma::mat dtYJ_dtheta(
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
double dtYJ_dgamma(const double c, const double gamma)
{
    double res, gmt;
    if (c < arma::datum::eps)
    {
        gmt = 2. - gamma;
        res = 1. - std::pow(1. - c, gmt) * (1. - gmt * std::log(1. - c));
        res *= std::pow(gmt, -2.);
    }
    else
    {
        gmt = gamma;
        res = 1. + std::pow(1. + c, gmt) * (std::log(1. + c) * gmt - 1.);
        res *= std::pow(gmt, -2.);
    }
    return res;
} // Check -- Correct

// Ref: line 24-27 of `grad_theta_logq.m`
double dlogdYJ_dtheta(const double theta, const double gamma)
{
    return (gamma - 1.) / (1. + std::abs(theta));
}

// Element-wise
arma::vec dlogdYJ_dtheta(
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
double dYJinv_dnu(const double nu, const double gamma)
{
    double gmt, res;
    double gamma_ = std::max(gamma, 0.0000001);
    if (nu < arma::datum::eps)
    {
        gmt = 2. - gamma_;
        res = std::pow(1. - gmt * nu, (1. - gmt) / gmt);
    }
    else
    {
        gmt = gamma_;
        res = std::pow(1. + gmt * nu, (1. - gmt) / gmt);
    }
    return res;
} // Status: Checked. OK.

arma::mat dYJinv_dnu(
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
double dYJinv_dgamma(const double nu, const double gamma)
{
    double gmt, res;
    if (nu < arma::datum::eps)
    {
        gmt = 2. - gamma;
        res = std::log(1. - gmt * nu) + gmt * nu / (1. - gmt * nu);
        res *= std::pow(gmt, -2.);
        res *= -std::pow(1. - gmt * nu, 1. / gmt);
    }
    else
    {
        gmt = gamma;
        res = -std::log(1. + gmt * nu) + gmt * nu / (1. + gmt * nu);
        res *= std::pow(gmt, -2.);
        res *= std::pow(1. + gmt * nu, 1. / gmt);
    }
    return res;
} // Status: Checked. OK.

// Ref: `dtheta_dtau.m`
double dYJinv_dtau(const double nu, const double gamma)
{
    return dYJinv_dgamma(nu, gamma) * dgm_dtau_gm(gamma);
} // Status: Checked. OK.

arma::mat dYJinv_dtau(
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

// One-dimensional case of `dtheta_dBDelta.m`. We only have D in this case and thus eps.
// Ref: `dtheta_dBDelta.m`
double dYJinv_dBD(const double nu, const double gamma, const double eps)
{
    return dYJinv_dnu(nu, gamma) * eps;
} // Status: Checked. OK.

/*
Appendix B.2 equation (ii)
*/
arma::mat dYJinv_dB(
    const arma::vec &nu,    // m x 1
    const arma::vec &gamma, // m x 1
    const arma::vec &xi)
{ // k x 1

    const unsigned int m = nu.n_elem;

    arma::mat dtheta_dnu = dYJinv_dnu(nu, gamma); // m x m
    arma::mat Im(m, m, arma::fill::eye);
    arma::mat grad = dtheta_dnu * arma::kron(xi.t(), Im); // m x mk

    return grad;
} // Status: Checked. OK.

/*
Appendiex B.2 equation (iii)
*/
arma::mat dYJinv_dD(
    const arma::vec &nu,    // m x 1
    const arma::vec &gamma, // m x 1
    const arma::vec &eps)
{ // m x 1

    arma::mat dtheta_dnu = dYJinv_dnu(nu, gamma);     // m x m
    arma::mat grad = dtheta_dnu * arma::diagmat(eps); // m x m

    return grad;
} // Status: Checked. OK.

/*
First order derivatives of YJ inv with respect to mu, sig, tau
*/
arma::vec dYJinv(
    const double nu, // c = YJinv(nu), nu = mu+sig*eps
    const double eps,
    const double gamma)
{

    arma::vec dd(3);
    dd.at(0) = dYJinv_dnu(nu, gamma);      // == line 20 of ./Derivatives/dtheta_dmu.m
    dd.at(1) = dYJinv_dBD(nu, gamma, eps); // TODO: Check line 26 of ./Derivatives/dtheta_dBDelta.m
    dd.at(2) = dYJinv_dtau(nu, gamma);     // == line 22 of ./Derivatives/dtheta_dtau.m
    return dd;
} // OK.


arma::mat get_sigma_inv(
    const arma::mat &B, // m x k
    const arma::vec &d,
    const unsigned int k)
{

    arma::mat Dm2 = arma::diagmat(1. / arma::pow(d, 2.)); // m x m, D^{-2}
    arma::mat Ik(k, k, arma::fill::eye);
    arma::mat tmp = Ik + B.t() * Dm2 * B;                     // k x k
    arma::mat SigInv = Dm2 - Dm2 * B * tmp.i() * B.t() * Dm2; // Woodbury formula
    SigInv = arma::symmatu(SigInv);                           // m x m

    return SigInv;
}



// Ref: `grad_theta_logq.m`
arma::vec dlogq_dtheta(
    const arma::mat &SigInv, // m x m
    const arma::vec &nu,     // m x 1
    const arma::vec &theta,  // m x 1
    const arma::vec &gamma,  // m x 1
    const arma::vec &mu)
{ // m x 1

    arma::mat dnu_dtheta = dtYJ_dtheta(theta, gamma);       // m x m
    arma::vec deriv = -dnu_dtheta.t() * SigInv * (nu - mu); // m x 1
    arma::vec deriv2 = dlogdYJ_dtheta(theta, gamma);        // m x 1
    return deriv + deriv2;                                  // m x 1
} // Status: Checked. OK.



arma::vec rtheta(
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

    arma::vec nu = mu + B * xi + d % eps;
    return tYJinv(nu, gamma); // m x 1, static parameters
}



/**
 * Logarithm of the proposal density
*/
double logq0(
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
    return logq;
}

#endif