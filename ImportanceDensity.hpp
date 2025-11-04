#ifndef _IMPORTANCEDENSITY_H_
#define _IMPORTANCEDENSITY_H_

#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <algorithm>
#include <RcppArmadillo.h>
#ifdef DGTF_USE_OPENMP
    #include <omp.h>
#endif
#include "Model.hpp"
#include "LinearBayes.hpp"

// [[Rcpp::plugins(cpp17)]]
// [[Rcpp::depends(RcppArmadillo)]]

/**
 * @brief State only
 * 
 * @param logq 
 * @param model 
 * @param t_new 
 * @param Theta_old 
 * @param y 
 * @return arma::vec 
 */
static arma::vec qforecast0(
    const Model &model,
    const unsigned int &t_new,  // current time t. The following inputs come from time t-1.
    const arma::mat &Theta_old, // p x N, {theta[t-1]}
    const arma::vec &y)
{
    const double y_old = y.at(t_new - 1);
    const double yhat_new = LinkFunc::mu2ft(y.at(t_new), model.flink);
    arma::vec logq(Theta_old.n_cols, arma::fill::zeros);

    #ifdef DGTF_USE_OPENMP
        #pragma omp parallel for num_threads(NUM_THREADS) schedule(runtime)
    #endif
    for (unsigned int i = 0; i < Theta_old.n_cols; i++)
    {
        arma::vec gtheta_old_i = SysEq::func_gt(
            model.fsys, model.fgain, model.dlag, Theta_old.col(i), y_old, 
            model.seas.period, model.seas.in_state); // gt(theta[t-1, i])
        
        double eta = TransFunc::func_ft(
            model.ftrans, model.fgain, model.dlag, model.seas, t_new, gtheta_old_i, y); // ft( gt(theta[t-1,i]) )
        
        double lambda = LinkFunc::ft2mu(eta, model.flink); // (eq 3.10)
        lambda = (t_new == 1 && lambda < EPS) ? 1. : lambda;
        double Vt = ApproxDisturbance::func_Vt_approx(
            lambda, model.dobs, model.flink); // (eq 3.11)
        Vt = std::abs(Vt) + EPS; // guard
        
        // logq.at(i) = R::dnorm4(yhat_new, eta, std::sqrt(Vt), true);
        // logq.at(i) = dnorm_cpp(yhat_new, eta, std::sqrt(Vt), true);
        // Fast normal log-density using variance (no sqrt/divide)
        const double diff = (yhat_new - eta);
        logq.at(i) = -0.5 * (LOG2PI + std::log(Vt) + (diff * diff) / Vt);
    }


    #ifdef DGTF_DO_BOUND_CHECK
        bound_check<arma::vec>(logq, "qforecast");
    #endif
    return logq;
} // func: imp_weights_forecast


struct QForecastFastConsts {
    unsigned int nelem;
    const double* Fphi;      // phi[0..nelem-1]
    const double* yptr;      // y raw pointer
    double seas_off;
    bool link_identity;
    bool obs_nb;
    bool obs_pois;
    double yhat_new;
    // Added: needed to compute Vt correctly
    ObsDist dobs;
    std::string flink;
};

static inline QForecastFastConsts make_qf_consts(
    const Model &model,
    unsigned int t_new,
    const arma::vec &y)
{
    QForecastFastConsts C{};
    C.nelem = std::min(t_new, model.dlag.nL);
    C.Fphi = model.dlag.Fphi.memptr();
    C.yptr = y.memptr();
    C.seas_off = 0.0;
    if (model.seas.period > 0 && !model.seas.in_state &&
        !model.seas.X.is_empty() && !model.seas.val.is_empty())
    {
        C.seas_off = arma::dot(model.seas.X.col(t_new), model.seas.val);
    }
    C.link_identity = (model.flink == "identity");
    C.obs_nb = (model.dobs.name == "nbinom" || model.dobs.name == "nbinomm");
    C.obs_pois = (model.dobs.name == "poisson");
    const double yval_t = C.yptr[t_new];
    C.yhat_new = C.link_identity ? yval_t : LinkFunc::mu2ft(yval_t, model.flink);
    // Added
    C.dobs = model.dobs;
    C.flink = model.flink;
    return C;
}



#endif