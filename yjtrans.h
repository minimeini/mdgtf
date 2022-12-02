#ifndef _YJTRANS_H
#define _YJTRANS_H

#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <algorithm>
#include <RcppArmadillo.h>


/*
Yeo-Johnson Transformation

- Main Function
- Inverse Transformation
- Gradients
*/


arma::vec mat2vech(const arma::mat& lotri);

/*
Yeo-Johnson Transform
Ref: `tYJ.m`
*/
double tYJ(
    const double theta,
    const double gamma);


arma::vec tYJ(
    const arma::vec& theta, // m x 1
    const arma::vec& gamma);


/*
Inverse of Yeo-Johnson Transformation
Ref: `tYJi.m`
*/
double tYJinv(
    const double nu,
    const double gamma);


arma::vec tYJinv(
    const arma::vec& nu, // m x 1
    const arma::vec& gamma);


// Transform gamma in (0,2) to the real line, denoted by tau
// Ref: `eta2tau.m`
double gamma2tau(const double gamma);

arma::vec gamma2tau(const arma::vec& gamma);



// Transform tau in real line to (0,2), denoted by gamma
// Ref: `tau2eta.m`
double tau2gamma(const double tau);

arma::vec tau2gamma(const arma::vec& tau);


// First-order derivative of gamma with respect to tau
// given the value of tau
// Ref: `deta_detau.m`
double dgamma_dtau_tau(const double tau);


// First-order derivative of gamma with respect to tau
// given the value of gamma
double dgm_dtau_gm(const double gamma);


/* 
First-order derivative of `nu = tYJ(theta;gamma)` with respect to theta
given the value of theta and gamma

Ref: `./Derivatives/dtYJ_dtheta.m`
*/
double dtYJ_dtheta(const double theta, const double gamma);

arma::mat dtYJ_dtheta(
    const arma::vec& theta, // m x 1
    const arma::vec& gamma);

/*
Loaiza-Maya et al., PDF, bottom of p26
*/
double dtYJ_dgamma(const double c, const double gamma);



double dlogdYJ_dtheta(const double theta, const double gamma);

// Element-wise
arma::vec dlogdYJ_dtheta(
    const arma::vec& theta, // m x 1
    const arma::vec& gamma);

/*
First order derivatives of YJinv with respect to nu
Correspond to `./Derivatives/dtheta_dphi.m`
*/
double dYJinv_dnu(const double nu, const double gamma);

arma::mat dYJinv_dnu(
    const arma::vec& nu, // m x 1
    const arma::vec& gamma);

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
double dYJinv_dgamma(const double nu, const double gamma);



// Ref: `dtheta_dtau.m`
double dYJinv_dtau(const double nu, const double gamma);

arma::mat dYJinv_dtau(
    const arma::vec& nu, // m x 1
    const arma::vec& gamma);

// One-dimensional case of `dtheta_dBDelta.m`. We only have D in this case and thus eps.
// Ref: `dtheta_dBDelta.m`
double dYJinv_dBD(const double nu, const double gamma, const double eps);

arma::mat dYJinv_dB(
    const arma::vec& nu, // m x 1
    const arma::vec& gamma, // m x 1
    const arma::vec& xi); // k x 1

arma::mat dYJinv_dD(
    const arma::vec& nu, // m x 1
    const arma::vec& gamma, // m x 1
    const arma::vec& eps); // m x 1

/*
First order derivatives of YJ inv with respect to mu, sig, tau
*/
arma::vec dYJinv(
    const double nu, // c = YJinv(nu), nu = mu+sig*eps
    const double eps,
    const double gamma);



arma::vec dlogq_dtheta(
    const arma::vec& nu, // m x 1
    const arma::vec& theta, // m x 1
    const arma::vec& gamma, // m x 1
    const arma::vec& mu, // m x 1 
    const arma::mat& B, // m x k
    const arma::vec& d);



arma::vec rtheta(
    arma::vec& xi, // k x 1
    arma::vec& eps, // m x 1
    const arma::vec& gamma, // m x 1
    const arma::vec& mu, // m x 1
    const arma::mat& B, // m x k
    const arma::vec& d);


#endif