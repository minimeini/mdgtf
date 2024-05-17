#pragma once
#ifndef CONSTANTS_H
#define CONSTANTS_H

#define ARMA_64BIT_WORD 1
#define ARMA_USE_LAPACK 1
#define ARMA_USE_BLAS 1
#define ARMA_USE_OPENMP 1

inline constexpr double EPS = 2.220446e-16;
inline constexpr double EPS8 = 1.e-8;
inline constexpr double UPBND = 700.;
inline constexpr double LOG2PI = 1.837877;

inline constexpr double covid_m = 4.7;
inline constexpr double covid_s = 2.9;
inline constexpr double LN_MU = 1.386262;
inline constexpr double LN_SD2 = 0.3226017;

inline constexpr double NB_KAPPA = 0.395;
inline constexpr double NB_R = 6;

inline constexpr double NB_LAMBDA = 0.;
inline constexpr double NB_DELTA = 30;

inline constexpr bool DEBUG = false;
inline constexpr bool VERBOSE = true;
inline constexpr unsigned int MAX_ITER = 100;


/**
 * @brief Default settings of linear Bayes
 * 
 */
inline constexpr bool LBA_FILL_ZERO = true;

#endif