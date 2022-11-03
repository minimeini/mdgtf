#ifndef _HVA_GRADIENTS_H
#define _HVA_GRADIENTS_H

#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <algorithm>
#include <RcppArmadillo.h>



double tYJ_gm(
    const double c,
    const double gamma);



double tYJinv_gm(
    const double nu,
    const double gamma);



// Transform gamma in (0,2) to the real line, denoted by tau
double gm2tau(const double gamma);



// Transform tau in real line to (0,2), denoted by gamma
double tau2gm(const double tau);



// First-order derivative of gamma with respect to tau
// given the value of tau
double dgm_dtau(const double tau);



// First-order derivative of gamma with respect to tau
// given the value of gamma
double dgm_dtau_gm(const double gamma);



/* 
First-order derivative of tYJ(c;gamma) with respect to c
given the value of c and gamma
Corresponds to ./Derivatives/dtYJ_dtheta.m
*/
double dYJ_dc(const double c, const double gamma);



double dYJ_dgamma(const double c, const double gamma);



double dlogdYJ_dc(const double c, const double gamma);



/*
First order derivatives of YJinv with respect to nu
Correspond to ./Derivatives/dtheta_dphi
*/
double dYJinv_dnu(const double nu, const double gamma);



/*
First order derivatives of YJinv with respect to gamma
Corresponds to ./Derivatives/dtheta_deta.m
*/
double dYJinv_dgamma(const double nu, const double gamma);



/*
First order derivatives of YJ inv with respect to mu, sig, tau
*/
arma::vec dYJinv(
    const double nu, // c = YJinv(nu), nu = mu+sig*eps
    const double eps,
    const double gamma);


double dlogJoint_dc(
    const arma::vec& psi,
    const double c);


double dlogJoint_dc(
    const arma::vec& psi,
    const double c,
    const double aw,
    const double bw);



double dlogVB_dc(
    const double c, 
    const double mu, 
    const double sig,
    const double gamma);



#endif