#pragma once
#ifndef CONSTANTS_H
#define CONSTANTS_H

#if !defined(ARMA_USE_BLAS)
    #define ARMA_USE_BLAS
#endif

#if !defined(ARMA_USE_LAPACK)
    #define ARMA_USE_LAPACK
#endif

#if !defined(ARMA_64BIT_WORD)
    #define ARMA_64BIT_WORD
#endif

#ifdef _OPENMP
    // #if !defined(DGTF_USE_OPENMP)
    //     #define DGTF_USE_OPENMP
    // #endif

    #if !defined(ARMA_USE_OPENMP)
        #define ARMA_USE_OPENMP
    #endif
#endif

#if !defined(DGTF_DO_BOUND_CHECK)
    #define DGTF_DO_BOUND_CHECK
#endif

// #if !defined(DGTF_DETAILED_OUTPUT)
//     #define DGTF_DETAILED_OUTPUT
// #endif


inline constexpr unsigned int NUM_THREADS = 16;

inline constexpr double EPS = 2.220446e-16;
inline constexpr double EPS8 = 1.e-8;
inline constexpr double UPBND = 700.;
inline constexpr double LOG2PI = 1.837877;

inline constexpr double covid_m = 4.7;
inline constexpr double covid_s = 2.9;
inline constexpr unsigned int MIN_LAG = 2;

inline constexpr double LN_MU = 1.386262;
inline constexpr double LN_SD2 = 0.3226017;
inline constexpr double NB_KAPPA = 0.395;
inline constexpr double NB_R = 6;

inline constexpr double NB_LAMBDA = 0.;
inline constexpr double NB_DELTA = 30;

inline constexpr bool VERBOSE = true;
inline constexpr unsigned int MAX_ITER = 100;
#endif