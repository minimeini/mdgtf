#ifndef _IMPORTANCEDENSITY_H_
#define _IMPORTANCEDENSITY_H_

#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <algorithm>
#include <RcppArmadillo.h>
#ifdef _OPENMP
    #include <omp.h>
#endif
#include "Model.hpp"
#include "LinearBayes.hpp"

// static arma::vec qlikelihood(
//     const unsigned int &t_next,  // (t + 1)
//     const arma::mat &Theta_next, // p x N
//     const arma::vec &y,
//     const Model &model)
// {
//     unsigned int N = Theta_next.n_cols;
//     arma::vec weights(N, arma::fill::zeros);
//     double mu0 = 0.;

//     for (unsigned int i = 0; i < N; i++)
//     {
//         arma::vec theta_next = Theta_next.col(i);
//         double ft = StateSpace::func_ft(model, t_next, theta_next, y); // use y[t], ..., y[t + 1 - nelem]
//         double lambda = LinkFunc::ft2mu(ft, model.flink.name, mu0);       // conditional mean of the observations

//         weights.at(i) = ObsDist::loglike(
//             y.at(t_next),
//             model.dobs.name,
//             lambda, model.dobs.par2,
//             false);
//     }

//     return weights;
// }

// static arma::vec qlikelihood(
//     const unsigned int &t_next,  // (t + 1)
//     const arma::mat &Theta_next, // p x N
//     const arma::vec &mu0,        // N x 1
//     const arma::vec &yall,
//     const Model &model)
// {
//     unsigned int N = Theta_next.n_cols;
//     arma::vec weights(N, arma::fill::zeros);

//     for (unsigned int i = 0; i < N; i++)
//     {
//         arma::vec theta_next = Theta_next.col(i);
//         double ft = StateSpace::func_ft(model, t_next, theta_next, yall); // use y[t], ..., y[t + 1 - nelem]
//         double lambda = LinkFunc::ft2mu(ft, model.flink.name, mu0.at(i)); // conditional mean of the observations

//         weights.at(i) = ObsDist::loglike(
//             yall.at(t_next),
//             model.dobs.name,
//             lambda, model.dobs.par2,
//             false);
//     }

//     return weights;
// }

static arma::vec qforecast(
    arma::mat &loc,            // p x N
    arma::cube &Prec_chol_inv, // p x p x N
    arma::vec &logq,           // N x 1
    const Model &model,
    const unsigned int &t_new,  // current time t. The following inputs come from time t-1.
    const arma::mat &Theta_old, // p x N, {theta[t-1]}
    const arma::vec &W_old,     // N x 1, {W[t-1]} samples of latent variance
    const arma::mat &param,   // (period + 3) x N, (seasonal components, rho, par1, par2)
    const arma::vec &y,
    const bool &obs_update,
    const bool &lag_update,
    const bool &full_rank = false,
    const unsigned int &max_lag = 30)
{
    const double y_old = y.at(t_new - 1);
    const double yhat_new = LinkFunc::mu2ft(y.at(t_new), model.flink, 0.);

    #ifdef _OPENMP
        #pragma omp parallel for num_threads(NUM_THREADS) schedule(runtime)
    #endif
    for (unsigned int i = 0; i < Theta_old.n_cols; i++)
    {
        Model mod = model;
        if (obs_update)
        {
            mod.seas.val = param.submat(0, i, mod.seas.period - 1, i);
            mod.dobs.par2 = std::exp(param.at(mod.seas.period, i));
        }

        if (lag_update)
        {
            mod.dlag.par1 = param.at(mod.seas.period + 1, i);
            mod.dlag.par2 = std::exp(param.at(mod.seas.period + 2, i));
        }

        arma::vec gtheta_old_i = StateSpace::func_gt(mod.ftrans, mod.fgain, mod.dlag, Theta_old.col(i), y_old, mod.seas.period, mod.seas.in_state);
        double ft_gtheta = StateSpace::func_ft(mod.ftrans, mod.fgain, mod.dlag, mod.seas, t_new, gtheta_old_i, y);
        double eta = ft_gtheta;
        double lambda = LinkFunc::ft2mu(eta, mod.flink); // (eq 3.10)
        lambda = (t_new == 1 && lambda < EPS) ? 1. : lambda;

        double Vt = ApproxDisturbance::func_Vt_approx(
            lambda, mod.dobs, mod.flink); // (eq 3.11)

        if (!full_rank)
        {
            loc.col(i) = gtheta_old_i;
            logq.at(i) = R::dnorm4(yhat_new, eta, std::sqrt(Vt), true);

        } // One-step-ahead predictive density
        else
        {
            arma::vec Ft_gtheta = LBA::func_Ft(mod.ftrans, mod.fgain, mod.dlag, t_new, gtheta_old_i, y, LBA_FILL_ZERO, mod.seas.period, mod.seas.in_state);
            double ft_tilde = ft_gtheta - arma::as_scalar(Ft_gtheta.t() * gtheta_old_i); // (eq 3.8)
            double delta = yhat_new - ft_tilde; // (eq 3.16)

            arma::mat Prec_i = Ft_gtheta * Ft_gtheta.t() / Vt; // nP x nP, function of mu0[i, t]
            Prec_i.diag() += EPS;
            Prec_i.at(0, 0) += 1. / W_old.at(i);                   // (eq 3.21)
            arma::mat Rchol = arma::chol(arma::symmatu(Prec_i));   // Right cholesky of the precision
            arma::mat Rchol_inv = arma::inv(arma::trimatu(Rchol)); // Left cholesky of the variance
            double ldetPrec = arma::accu(arma::log(Rchol.diag())) * 2.;
            Prec_chol_inv.slice(i) = Rchol_inv;

            loc.col(i) = Ft_gtheta * (delta / Vt); // nP x 1
            loc.at(0, i) += gtheta_old_i.at(0) / W_old.at(i);

            double ldetV = std::log(std::abs(Vt) + EPS);
            double ldetW = std::log(std::abs(W_old.at(i)) + EPS);

            double loglik = LOG2PI + ldetV + ldetW + ldetPrec; // (eq 3.24)
            loglik += delta * delta / Vt;
            loglik += std::pow(gtheta_old_i.at(0), 2.) / W_old.at(i);
            loglik -= arma::as_scalar(loc.col(i).t() * Rchol_inv * Rchol_inv.t() * loc.col(i));
            loglik *= -0.5; // (eq 3.24 - 3.25)

            logq.at(i) += loglik;
        } // one-step-ahead predictive density
    }

    double logq_max = logq.max();
    logq.for_each([&logq_max](arma::vec::elem_type &val)
                  { val -= logq_max; });

    arma::vec weights = arma::exp(logq);
    if (DEBUG)
    {
        bound_check<arma::vec>(weights, "qforecast");
    }
    return weights;
} // func: imp_weights_forecast



static arma::vec qforecast(
    arma::mat &loc,            // p x N
    arma::cube &Prec_chol_inv, // p x p x N
    arma::vec &logq,           // N x 1
    const Model &model,
    const unsigned int &t_new,  // current time t. The following inputs come from time t-1.
    const arma::mat &Theta_old, // p x N, {theta[t-1]}
    const arma::mat &Wt,        // p x N, {W[t-1]} samples of latent variance
    const arma::vec &y,         // y[t]
    const bool &full_rank = false)
{
    const double y_old = y.at(t_new - 1);
    const double yhat_new = LinkFunc::mu2ft(y.at(t_new), model.flink, 0.);

    #ifdef _OPENMP
        #pragma omp parallel for num_threads(NUM_THREADS) schedule(runtime)
    #endif
    for (unsigned int i = 0; i < Theta_old.n_cols; i++)
    {
        arma::vec gtheta_old_i = StateSpace::func_gt(model.ftrans, model.fgain, model.dlag, Theta_old.col(i), y_old, model.seas.period, model.seas.in_state); // gt(theta[t-1, i])
        double ft_gtheta = StateSpace::func_ft(model.ftrans, model.fgain, model.dlag, model.seas, t_new, gtheta_old_i, y); // ft( gt(theta[t-1,i]) )
        double eta = ft_gtheta;
        double lambda = LinkFunc::ft2mu(eta, model.flink); // (eq 3.10)
        lambda = (t_new == 1 && lambda < EPS) ? 1. : lambda;

        double Vt = ApproxDisturbance::func_Vt_approx(
            lambda, model.dobs, model.flink); // (eq 3.11)

        if (!full_rank)
        {
            loc.col(i) = gtheta_old_i;
            logq.at(i) = R::dnorm4(yhat_new, eta, std::sqrt(Vt), true);

        } // One-step-ahead predictive density
        else
        {
            arma::vec Ft_gtheta = LBA::func_Ft(model.ftrans, model.fgain, model.dlag, t_new, gtheta_old_i, y, LBA_FILL_ZERO, model.seas.period, model.seas.in_state); // Ft evaluated at a[t_new]
            double ft_tilde = ft_gtheta - arma::as_scalar(Ft_gtheta.t() * gtheta_old_i); // (eq 3.8)
            double delta = yhat_new - model.dobs.par1 - ft_tilde; // (eq 3.16)

            arma::mat Prec_i = Ft_gtheta * Ft_gtheta.t() / Vt; // nP x nP, function of mu0[i, t]
            Prec_i.diag() += EPS;
            Prec_i.at(0, 0) += 1. / Wt.at(0, i); // (eq 3.21)
            arma::mat Rchol = arma::chol(arma::symmatu(Prec_i)); // Right cholesky of the precision
            arma::mat Rchol_inv = arma::inv(arma::trimatu(Rchol)); // Left cholesky of the variance
            Prec_chol_inv.slice(i) = Rchol_inv;

            double ldetPrec = arma::accu(arma::log(Rchol.diag())) * 2.; // ldetSigma = - ldetPrec

            loc.col(i) = Ft_gtheta * (delta / Vt); // nP x 1, location, mean mu = Sigma * loc
            loc.at(0, i) += gtheta_old_i.at(0) / Wt.at(0, i); // location

            double ldetV = std::log(std::abs(Vt) + EPS);
            double ldetW = std::log(std::abs(Wt.at(0, i)) + EPS);

            double loglik = LOG2PI + ldetV + ldetW + ldetPrec; // (eq 3.24)
            loglik += delta * delta / Vt;
            loglik += std::pow(gtheta_old_i.at(0), 2.) / Wt.at(0, i);
            loglik -= arma::as_scalar(loc.col(i).t() * Rchol_inv * Rchol_inv.t() * loc.col(i));
            loglik *= -0.5; // (eq 3.24 - 3.25)

            logq.at(i) += loglik;
        } // one-step-ahead predictive density
    }

    double logq_max = logq.max();
    logq.for_each([&logq_max](arma::vec::elem_type &val)
                  { val -= logq_max; });

    arma::vec weights = arma::exp(logq);
    if (DEBUG)
    {
        bound_check<arma::vec>(weights, "qforecast");
    }
    return weights;
} // func: imp_weights_forecast

static arma::vec qforecast(
    arma::mat &loc,            // p x N
    arma::cube &Prec_chol_inv, // p x p x N
    arma::vec &logq,           // N x 1
    const Model &model,
    const unsigned int &t_new,  // current time t. The following inputs come from time t-1.
    const arma::mat &Theta_old, // p x N, {theta[t-1]}
    const arma::vec &Wt,        // p x 1, {W[t-1]} samples of latent variance
    const arma::vec &y,         // y[t]
    const bool &full_rank = false)
{
    const double y_old = y.at(t_new - 1);
    const double yhat_new = LinkFunc::mu2ft(y.at(t_new), model.flink, 0.);

    #ifdef _OPENMP
        #pragma omp parallel for num_threads(NUM_THREADS) schedule(runtime)
    #endif
    for (unsigned int i = 0; i < Theta_old.n_cols; i++)
    {
        arma::vec gtheta_old_i = StateSpace::func_gt(model.ftrans, model.fgain, model.dlag, Theta_old.col(i), y_old, model.seas.period, model.seas.in_state); // gt(theta[t-1, i])
        double ft_gtheta = StateSpace::func_ft(model.ftrans, model.fgain, model.dlag, model.seas, t_new, gtheta_old_i, y); // ft( gt(theta[t-1,i]) )
        double eta = ft_gtheta;
        double lambda = LinkFunc::ft2mu(eta, model.flink); // (eq 3.10)
        lambda = (t_new == 1 && lambda < EPS) ? 1. : lambda;

        double Vt = ApproxDisturbance::func_Vt_approx(
            lambda, model.dobs, model.flink); // (eq 3.11)

        if (!full_rank)
        {
            // loc.col(i) = gtheta_old_i;
            logq.at(i) = R::dnorm4(yhat_new, eta, std::sqrt(Vt), true);

        } // One-step-ahead predictive density
        else
        {
            arma::vec Ft_gtheta = LBA::func_Ft(model.ftrans, model.fgain, model.dlag, t_new, gtheta_old_i, y, LBA_FILL_ZERO, model.seas.period, model.seas.in_state);
            double ft_tilde = ft_gtheta - arma::as_scalar(Ft_gtheta.t() * gtheta_old_i); // (eq 3.8)
            double delta = yhat_new - model.dobs.par1 - ft_tilde; // (eq 3.16)

            arma::mat Prec_i = Ft_gtheta * Ft_gtheta.t() / Vt; // nP x nP, function of mu0[i, t]
            Prec_i.diag() += EPS;
            Prec_i.at(0, 0) += 1. / Wt.at(0); // (eq 3.21)
            arma::mat Rchol = arma::chol(arma::symmatu(Prec_i)); // Right cholesky of the precision
            arma::mat Rchol_inv = arma::inv(arma::trimatu(Rchol)); // Left cholesky of the variance
            double ldetPrec = arma::accu(arma::log(Rchol.diag())) * 2.; // ldetSigma = - ldetPrec
            Prec_chol_inv.slice(i) = Rchol_inv;

            loc.col(i) = Ft_gtheta * (delta / Vt); // nP x 1, location, mean mu = Sigma * loc
            loc.at(0, i) += gtheta_old_i.at(0) / Wt.at(0); // location

            double ldetV = std::log(std::abs(Vt) + EPS);
            double ldetW = std::log(std::abs(Wt.at(0)) + EPS);

            double loglik = LOG2PI + ldetV + ldetW + ldetPrec; // (eq 3.24)
            loglik += delta * delta / Vt;
            loglik += std::pow(gtheta_old_i.at(0), 2.) / Wt.at(0);
            loglik -= arma::as_scalar(loc.col(i).t() * Rchol_inv * Rchol_inv.t() * loc.col(i));
            loglik *= -0.5; // (eq 3.24 - 3.25)

            logq.at(i) += loglik;
        } // one-step-ahead predictive density
    }

    double logq_max = logq.max();
    logq.for_each([&logq_max](arma::vec::elem_type &val)
                  { val -= logq_max; });

    arma::vec weights = arma::exp(logq);

    if (DEBUG)
    {
        bound_check<arma::vec>(weights, "qforecast");
    }
    return weights;
} // func: imp_weights_forecast

static arma::vec qforecast(
    arma::mat &loc,            // p x N
    arma::cube &Prec_chol_inv, // p x p x N
    arma::vec &logq,           // N x 1
    Model &model,
    const unsigned int &t_new,  // current time t. The following inputs come from time t-1.
    const arma::mat &Theta_old, // p x N, {theta[t-1]}
    const arma::mat &param_filter, // (period + 3) x N
    const arma::vec &Wt,        // p x 1, {W[t-1]} samples of latent variance
    const arma::vec &y,         // y[t]
    const bool &infer_seas = false,
    const bool &full_rank = false)
{
    const double y_old = y.at(t_new - 1);
    const double yhat_new = LinkFunc::mu2ft(y.at(t_new), model.flink, 0.);

    #ifdef _OPENMP
        #pragma omp parallel for num_threads(NUM_THREADS) schedule(runtime)
    #endif
    for (unsigned int i = 0; i < Theta_old.n_cols; i++)
    {
        Model mod = model;
        if (infer_seas)
        {
            mod.seas.val = param_filter.submat(0, i, mod.seas.period - 1, i);
        }

        arma::vec gtheta_old_i = StateSpace::func_gt(mod.ftrans, mod.fgain, mod.dlag, Theta_old.col(i), y_old, mod.seas.period, mod.seas.in_state);           // gt(theta[t-1, i])
        double ft_gtheta = StateSpace::func_ft(mod.ftrans, mod.fgain, mod.dlag, mod.seas, t_new, gtheta_old_i, y);                                            // ft( gt(theta[t-1,i]) )
        double eta = ft_gtheta;
        double lambda = LinkFunc::ft2mu(eta, mod.flink); // (eq 3.10)
        lambda = (t_new == 1 && lambda < EPS) ? 1. : lambda;

        double Vt = ApproxDisturbance::func_Vt_approx(
            lambda, mod.dobs, mod.flink); // (eq 3.11)

        if (!full_rank)
        {
            // loc.col(i) = gtheta_old_i;
            logq.at(i) = R::dnorm4(yhat_new, eta, std::sqrt(Vt), true);

        } // One-step-ahead predictive density
        else
        {
            arma::vec Ft_gtheta = LBA::func_Ft(mod.ftrans, mod.fgain, mod.dlag, t_new, gtheta_old_i, y, LBA_FILL_ZERO, mod.seas.period, mod.seas.in_state);
            double ft_tilde = ft_gtheta - arma::as_scalar(Ft_gtheta.t() * gtheta_old_i); // (eq 3.8)
            double delta = yhat_new - mod.dobs.par1 - ft_tilde;                          // (eq 3.16)

            arma::mat Prec_i = Ft_gtheta * Ft_gtheta.t() / Vt; // nP x nP, function of mu0[i, t]
            Prec_i.diag() += EPS;
            Prec_i.at(0, 0) += 1. / Wt.at(0);                           // (eq 3.21)
            arma::mat Rchol = arma::chol(arma::symmatu(Prec_i));        // Right cholesky of the precision
            arma::mat Rchol_inv = arma::inv(arma::trimatu(Rchol));      // Left cholesky of the variance
            double ldetPrec = arma::accu(arma::log(Rchol.diag())) * 2.; // ldetSigma = - ldetPrec
            Prec_chol_inv.slice(i) = Rchol_inv;

            loc.col(i) = Ft_gtheta * (delta / Vt);         // nP x 1, location, mean mu = Sigma * loc
            loc.at(0, i) += gtheta_old_i.at(0) / Wt.at(0); // location

            double ldetV = std::log(std::abs(Vt) + EPS);
            double ldetW = std::log(std::abs(Wt.at(0)) + EPS);

            double loglik = LOG2PI + ldetV + ldetW + ldetPrec; // (eq 3.24)
            loglik += delta * delta / Vt;
            loglik += std::pow(gtheta_old_i.at(0), 2.) / Wt.at(0);
            loglik -= arma::as_scalar(loc.col(i).t() * Rchol_inv * Rchol_inv.t() * loc.col(i));
            loglik *= -0.5; // (eq 3.24 - 3.25)

            logq.at(i) += loglik;
        } // one-step-ahead predictive density
    }

    double logq_max = logq.max();
    logq.for_each([&logq_max](arma::vec::elem_type &val)
                  { val -= logq_max; });

    arma::vec weights = arma::exp(logq);
    if (DEBUG)
    {
        bound_check<arma::vec>(weights, "qforecast");
    }
    return weights;
} // func: imp_weights_forecast

/**
 * @brief Resampling distribution for normal DLM.
 * 
 * @return arma::vec 
 */
static arma::vec qforecast_normal(
    const Model &model,
    const unsigned int &t_new,   // for y[t] to be predicted
    const arma::mat &Theta_old,  // p x N, theta[t-1]
    const arma::mat &param,      // (sig2, tau2, seasonal component) x N, results from t - 1
    const arma::vec &y,          // (nT + 1) x 1
    const arma::mat &seas) // nseas x (nT + 1)
{
    // model.nP == model.nL: number of latent states excluding static parameters with dynamic models
    // seasonal_period: for seasonal component
    arma::vec Fphi = LagDist::get_Fphi(model.nP, model.dlag.name, model.dlag.par1, model.dlag.par2);
    double FtF = arma::as_scalar(Fphi.t() * Fphi);
    arma::mat Gmat(model.nP, model.nP, arma::fill::zeros);
    Gmat.at(0, 0) = 1.;
    for (unsigned int i = 1; i < model.nP; i++)
    {
        Gmat.at(i, i - 1) = 1.;
    }

    arma::vec logq(Theta_old.n_cols, arma::fill::zeros);
    for (unsigned int i = 0; i < Theta_old.n_cols; i++)
    {
        arma::vec gtheta = Gmat * Theta_old.col(i);
        double yseas = arma::as_scalar(seas.col(t_new).t() * param.submat(2, i, param.n_rows - 1, i));
        double ymean = arma::as_scalar(yseas + Fphi.t() * gtheta);
        double yvar = std::exp(param.at(1, i)) * FtF + std::exp(param.at(0, i));
        double ysd = std::sqrt(yvar);
        logq.at(i) = R::dnorm4(y.at(t_new), ymean, ysd, true);
    }

    double logq_max = logq.max();
    logq.for_each([&logq_max](arma::vec::elem_type &val)
                  { val -= logq_max; });

    return logq;
}

/**
 * @todo Is there a more efficient way to construct artificial priors?
 */
static void prior_forward(
    arma::mat &mu,     // nP x (nT + 1)
    arma::cube &prec,  // nP x nP x (nT + 1)
    const Model &model,
    const arma::vec &Wt, // nP x 1
    const arma::vec &y   // (nT + 1) x 1
)
{
    const unsigned int nT = y.n_elem - 1;

    arma::mat sig = arma::eye<arma::mat>(model.nP, model.nP) * 2.;
    prec.slice(0) = arma::eye<arma::mat>(model.nP, model.nP) * 0.5;

    arma::mat Gt = TransFunc::init_Gt(model.nP, model.dlag, model.ftrans, model.seas.period, model.seas.in_state);
    for (unsigned int t = 1; t <= nT; t++)
    {
        mu.col(t) = StateSpace::func_gt(model.ftrans, model.fgain, model.dlag, mu.col(t - 1), y.at(t - 1), model.seas.period, model.seas.in_state);
        LBA::func_Gt(Gt, model, mu.col(t - 1), y.at(t - 1));
        sig = Gt * sig * Gt.t();
        sig.at(0, 0) += Wt.at(0);
        sig.diag() += EPS;

        prec.slice(t) = inverse(sig);
    }

    return;
}

// /**
//  * @brief Construct artifical priors (for backward filtering) via forward iterations.
//  *
//  */
// static void prior_forward2(
//     arma::mat &v,     // nP x (nT + 1)
//     arma::cube &Vinv, // nP x nP x (nT + 1)
//     arma::cube &VGt,  // nP x nP x (nT + 1)
//     arma::cube &Uinv, // nP x nP x (nT + 1)
//     arma::vec &ldetU, // (nT + 1) x 1
//     arma::vec &cnst_prior, // (nT + 1) x 1
//     const Model &model,
//     const arma::vec &Wt, // nP x 1
//     const arma::vec &y   // (nT + 1) x 1
// )
// {
//     std::map<std::string, AVAIL::Transfer> trans_list = AVAIL::trans_list;
//     const unsigned int nP = Wt.n_elem;
//     const double p_ = static_cast<double>(nP);
//     const unsigned int nT = y.n_elem - 1;

//     arma::mat V = arma::eye<arma::mat>(nP, nP) * 2.;
//     Vinv.slice(0) = arma::eye<arma::mat>(nP, nP) * 0.5;

//     cnst_prior.for_each([&p_](arma::vec::elem_type &c)
//                         { -0.5 * p_ *LOG2PI; });

//     for (unsigned int t = 1; t <= nT; t++)
//     {
//         cnst_prior.at(t - 1) -= 0.5 * arma::log_det_sympd(V);
//         cnst_prior.at(t - 1) -= 0.5 * arma::as_scalar(v.col(t - 1).t() * Vinv.slice(t - 1) * v.col(t - 1));

//         v.col(t) = StateSpace::func_gt(model, v.col(t - 1), y.at(t - 1));
//         arma::mat Gt = LBA::func_Gt(model, v.col(t - 1), y.at(t - 1));

//         VGt.slice(t - 1) = V * Gt.t();
//         arma::mat U = V;

//         V = Gt * V * Gt.t();
//         V.at(0, 0) += Wt.at(0);
//         V.diag() += EPS;

//         arma::mat sig_chol = arma::chol(arma::symmatu(V)); // V = R.t() * R
//         arma::mat sig_chol_inv = arma::inv(arma::trimatu(sig_chol));
//         Vinv.slice(t) = sig_chol_inv * sig_chol_inv.t(); // Vinv = R.i() * R.i().t()

//         arma::mat K(nP, nP, arma::fill::zeros);
//         if (trans_list[model.transfer.name] == AVAIL::sliding)
//         {
//             for (unsigned int i = 0; i < nP; i ++)
//             {
//                 K.at(i, i + 1) = 1.;
//             }
//             K.at(nP - 1, nP - 1) = 1.;

//             U.zeros();
//             Uinv.at(nP - 1, nP - 1, t - 1) = 1. / Wt.at(0);
//             ldetU.at(t - 1) = std::log(Wt.at(0));
//         }
//         else
//         {
//             K = VGt.slice(t - 1) * Vinv.slice(t);
//             U = U - K * V * K.t();
//             sig_chol = arma::chol(arma::symmatu(U));
//             sig_chol_inv = arma::inv(arma::trimatu(sig_chol));
//             Uinv.slice(t - 1) = sig_chol_inv * sig_chol_inv.t();
//             ldetU.at(t - 1) = 2. * arma::accu(arma::log(sig_chol_inv.diag()));
//         }
//         }

//     return;
// }



static void backward_kernel(
    arma::mat &K,
    arma::vec &r,
    arma::mat &Uinv,
    double &ldetU,
    const Model &model,
    const unsigned int &t_cur,
    const arma::mat &vt,  // nP x (nT + 1)
    const arma::cube &Vt_inv, // nP x nP x (nT + 1)
    const arma::vec &Wt,
    const arma::vec &y)
{
    unsigned int nstate = model.nP;
    if (model.seas.in_state)
    {
        nstate -= model.seas.period;
    }
    std::map<std::string, TransFunc::Transfer> trans_list = TransFunc::trans_list;
    arma::mat U_cur = K;
    arma::mat Uprec_cur = K;
    arma::mat Urchol_cur = K;

    if (trans_list[model.ftrans] == TransFunc::sliding)
    {
        for (unsigned int i = 0; i < nstate - 1; i++)
        {
            K.at(i, i + 1) = 1.;
        }
        K.at(nstate - 1, nstate - 1) = 1.;
        if (model.seas.in_state)
        {
            if (model.seas.period == 1)
            {
                K.at(model.nP - 1, model.nP - 1) = 1.;
            }
            else if (model.seas.period > 1)
            {
                K.at(nstate, model.nP - 1) = 1.;
                for (unsigned int i = nstate + 1; i < model.nP; i++)
                {
                    K.at(i, i - 1) = 1.;
                }
            }
        }

        U_cur.at(nstate - 1, nstate - 1) = Wt.at(0);
        Uprec_cur.at(nstate - 1, nstate - 1) = 1. / Wt.at(0);
        Urchol_cur.at(nstate - 1, nstate - 1) = std::sqrt(Wt.at(0));
        ldetU = std::log(Wt.at(0));
    }
    else
    {
        arma::mat G_next = TransFunc::init_Gt(model.nP, model.dlag, model.ftrans, model.seas.period, model.seas.in_state);
        LBA::func_Gt(G_next, model, vt.col(t_cur), y.at(t_cur));

        arma::mat Vchol = arma::chol(Vt_inv.slice(t_cur));
        arma::mat Vchol_inv = arma::inv(arma::trimatu(Vchol));
        arma::mat Vt_cur = Vchol_inv * Vchol_inv.t();

        arma::mat VGt = Vt_cur * G_next.t();
        K = VGt * Vt_inv.slice(t_cur + 1);

        U_cur = arma::symmatu(Vt_cur - VGt * Vt_inv.slice(t_cur + 1) * VGt.t());
        Urchol_cur = arma::chol(U_cur);
        arma::mat Urchol_inv = arma::inv(arma::trimatu(Urchol_cur));
        Uprec_cur = Urchol_inv * Urchol_inv.t();
        ldetU = arma::log_det_sympd(U_cur);
    }

    r = vt.col(t_cur) - K * vt.col(t_cur + 1);
    return;
}

static arma::vec qbackcast(
    arma::mat &loc,          // p x N, mean of the posterior of theta[t_cur] | y[t_cur:nT], theta[t_next], W
    arma::cube &Prec_chol_inv,       // p x p x N, precision of the posterior of theta[t_cur] | y[t_cur:nT], theta[t_next], W
    arma::vec &logq,        // N x 1
    const Model &model,
    const unsigned int &t_cur,   // current time "t". The following inputs come from time t+1. t_next = t + 1; t_prev = t - 1
    const arma::mat &Theta_next, // p x N, {theta[t+1]}
    const arma::mat &vt,         // nP x (nT + 1), v[t], mean of artificial prior for theta[t_cur]
    const arma::cube &Vt_inv,        // nP x nP * (nT + 1), V[t], variance of artificial prior for theta[t_cur]
    const arma::vec &Wt, // p x 1
    const arma::vec &y,
    const bool &full_rank = false
)
{
    std::map<std::string, TransFunc::Transfer> trans_list = TransFunc::trans_list;
    double yhat_cur = LinkFunc::mu2ft(y.at(t_cur), model.flink, 0.);
    unsigned int N = Theta_next.n_cols;
    unsigned int t_next = t_cur + 1;

    arma::vec r_cur(model.nP, arma::fill::zeros);
    arma::mat K_cur(model.nP, model.nP, arma::fill::zeros);
    arma::mat Uprec_cur = K_cur;
    double ldetU = 0.;

    if (trans_list[model.ftrans] == TransFunc::Transfer::sliding || full_rank)
    {
        backward_kernel(K_cur, r_cur, Uprec_cur, ldetU, model, t_cur, vt, Vt_inv, Wt, y);
    }

    // #pragma omp parallel for num_threads(NUM_THREADS) schedule(runtime)
    for (unsigned int i = 0; i < N; i++)
    {
        arma::vec u_cur;
        if (trans_list[model.ftrans] == TransFunc::Transfer::sliding || full_rank)
        {
            u_cur = K_cur * Theta_next.col(i) + r_cur;
        }
        else
        {
            // Iterative transfer function
            // Use point estimate of theta[t] to predict y[t]
            unsigned int cnt = 0;
            u_cur = StateSpace::func_backward_gt(model.ftrans, model.fgain, model.dlag, Theta_next.col(i), y.at(t_cur), 0., model.seas.period, model.seas.in_state);
            
            if (arma::any(u_cur.tail(u_cur.n_elem - 1) < 0))
            {
                Theta_next.col(i).t().print("theta_next");
                u_cur.t().print("u_cur");
                throw std::runtime_error("ft should not be negative.");
            }
            // u_cur = Theta_next.col(i);
        }

        double ft_ut = StateSpace::func_ft(model.ftrans, model.fgain, model.dlag, model.seas, t_cur, u_cur, y);
        double eta = ft_ut;
        double lambda = LinkFunc::ft2mu(eta, model.flink); // (eq 3.58)
        if (lambda < EPS)
        {
            std::cerr << "\n lambda = " << lambda << ", mu0 = " << model.dobs.par1 << ", ft_ut = " << ft_ut << std::endl;
            
            throw std::runtime_error("qbackcast: observation mean lambda should not be zero.\n");
        }
        double Vtilde = ApproxDisturbance::func_Vt_approx(
            lambda, model.dobs, model.flink); // (eq 3.59)
        Vtilde = std::abs(Vtilde) + EPS;

        if (!full_rank)
        {
            // No information from data, degenerates to the backward evolution
            loc.col(i) = u_cur;
            logq.at(i) = R::dnorm4(yhat_cur, eta, std::sqrt(Vtilde), true);
        } // one-step backcasting
        else
        {
            arma::vec F_cur = LBA::func_Ft(model.ftrans, model.fgain, model.dlag, t_cur, u_cur, y, LBA_FILL_ZERO, model.seas.period, model.seas.in_state);
            arma::mat Prec = arma::symmatu(F_cur * F_cur.t() / Vtilde) + Uprec_cur;
            Prec.diag() += EPS;

            arma::mat Rchol = arma::chol(arma::symmatu(Prec));
            arma::mat Rchol_inv = arma::inv(arma::trimatu(Rchol));
            double ldetPrec = arma::accu(arma::log(Rchol.diag())) * 2.;
            Prec_chol_inv.slice(i) = Rchol_inv;

            double delta = yhat_cur - eta;
            delta += arma::as_scalar(F_cur.t() * u_cur);

            loc.col(i) = F_cur * (delta / Vtilde) + Uprec_cur * u_cur;

            double ldetV = std::log(Vtilde);
            double logq_pred = LOG2PI + ldetV + ldetU + ldetPrec; // (eq 3.63)
            logq_pred += delta * delta / Vtilde;
            logq_pred += arma::as_scalar(u_cur.t() * Uprec_cur * u_cur);
            logq_pred -= arma::as_scalar(loc.col(i).t() * Rchol_inv * Rchol_inv.t() * loc.col(i));
            logq_pred *= -0.5;

            logq.at(i) += logq_pred;
        }
    }

    double logq_max = logq.max();
    logq.for_each([&logq_max](arma::vec::elem_type &val)
                  { val -= logq_max; });
    arma::vec weights = arma::exp(logq);
    if (DEBUG)
    {
        bound_check<arma::vec>(weights, "qforecast");
    }
    return weights;
} // func: imp_weights_forecast (no unknown statics)

#endif