#ifndef _IMPORTANCEDENSITY_H_
#define _IMPORTANCEDENSITY_H_

#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <algorithm>
#include <RcppArmadillo.h>
#include <omp.h>
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
    const arma::mat &param,   // 4 x N, (mu0, rho, par1, par2)
    const arma::vec &y,
    const bool &obs_update,
    const bool &lag_update,
    const bool &full_rank = false,
    const unsigned int &max_lag = 30)
{
    double y_old = y.at(t_new - 1);
    double yhat_new = LinkFunc::mu2ft(y.at(t_new), model.flink.name, 0.);

    #pragma omp parallel for num_threads(NUM_THREADS) schedule(runtime)
    for (unsigned int i = 0; i < Theta_old.n_cols; i++)
    {
        ObsDist dobs = model.dobs;
        TransFunc ftrans = model.transfer;
        if (obs_update)
        {
            dobs._par1 = param.at(0, i);
            dobs._par2 = param.at(1, i);
        }
        if (lag_update)
        {
            unsigned int nlag = ftrans.update_dlag(param.at(2, i), param.at(3, i), max_lag, false);
        }
        arma::vec gtheta_old_i = StateSpace::func_gt(ftrans, Theta_old.col(i), y_old);
        double ft_gtheta = StateSpace::func_ft(ftrans, t_new, gtheta_old_i, y);
        double eta = param.at(0, i) + ft_gtheta;
        double lambda = LinkFunc::ft2mu(eta, model.flink.name, 0.); // (eq 3.10)
        lambda = (t_new == 1 && lambda < EPS) ? 1. : lambda;

        double Vt = ApproxDisturbance::func_Vt_approx(
            lambda, dobs, model.flink.name); // (eq 3.11)

        if (!full_rank)
        {
            loc.col(i) = gtheta_old_i;
            logq.at(i) = R::dnorm4(yhat_new, eta, std::sqrt(Vt), true);

        } // One-step-ahead predictive density
        else
        {
            arma::vec Ft_gtheta = LBA::func_Ft(ftrans, t_new, gtheta_old_i, y);
            double ft_tilde = ft_gtheta - arma::as_scalar(Ft_gtheta.t() * gtheta_old_i); // (eq 3.8)
            double delta = yhat_new - param.at(0, i) - ft_tilde;                          // (eq 3.16)

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
    bound_check<arma::vec>(weights, "imp_weights_forecast");

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
    const arma::vec &par,       // m x 1, m = 4
    const arma::vec &y,         // y[t]
    const bool &full_rank = false)
{
    double y_old = y.at(t_new - 1);
    double yhat_new = LinkFunc::mu2ft(y.at(t_new), model.flink.name, 0.);

    #pragma omp parallel for num_threads(NUM_THREADS) schedule(runtime)
    for (unsigned int i = 0; i < Theta_old.n_cols; i++)
    {
        arma::vec gtheta_old_i = StateSpace::func_gt(model.transfer, Theta_old.col(i), y_old); // gt(theta[t-1, i])
        double ft_gtheta = StateSpace::func_ft(model.transfer, t_new, gtheta_old_i, y);        // ft( gt(theta[t-1,i]) )
        double eta = par.at(0) + ft_gtheta;
        double lambda = LinkFunc::ft2mu(eta, model.flink.name, 0.); // (eq 3.10)
        lambda = (t_new == 1 && lambda < EPS) ? 1. : lambda;

        double Vt = ApproxDisturbance::func_Vt_approx(
            lambda, model.dobs, model.flink.name); // (eq 3.11)

        if (!full_rank)
        {
            loc.col(i) = gtheta_old_i;
            logq.at(i) = R::dnorm4(yhat_new, eta, std::sqrt(Vt), true);

        } // One-step-ahead predictive density
        else
        {
            arma::vec Ft_gtheta = LBA::func_Ft(model.transfer, t_new, gtheta_old_i, y);  // Ft evaluated at a[t_new]
            double ft_tilde = ft_gtheta - arma::as_scalar(Ft_gtheta.t() * gtheta_old_i); // (eq 3.8)
            double delta = yhat_new - par.at(0) - ft_tilde; // (eq 3.16)

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

    try
    {
        bound_check<arma::vec>(weights, "imp_weights_forecast");
    }
    catch (const std::exception &e)
    {
        logq.t().brief_print("\n logarithm of weights: ");
        throw std::runtime_error(e.what());
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
    const arma::vec &par,       // m x 1, m = 4
    const arma::vec &y,         // y[t]
    const bool &full_rank = false)
{
    double y_old = y.at(t_new - 1);
    double yhat_new = LinkFunc::mu2ft(y.at(t_new), model.flink.name, 0.);

    #pragma omp parallel for num_threads(NUM_THREADS) schedule(runtime)
    for (unsigned int i = 0; i < Theta_old.n_cols; i++)
    {
        arma::vec gtheta_old_i = StateSpace::func_gt(model.transfer, Theta_old.col(i), y_old); // gt(theta[t-1, i])
        double ft_gtheta = StateSpace::func_ft(model.transfer, t_new, gtheta_old_i, y); // ft( gt(theta[t-1,i]) )
        double eta = par.at(0) + ft_gtheta;
        double lambda = LinkFunc::ft2mu(eta, model.flink.name, 0.); // (eq 3.10)
        lambda = (t_new == 1 && lambda < EPS) ? 1. : lambda;

        double Vt = ApproxDisturbance::func_Vt_approx(
            lambda, model.dobs, model.flink.name); // (eq 3.11)

        if (!full_rank)
        {
            loc.col(i) = gtheta_old_i;
            logq.at(i) = R::dnorm4(yhat_new, eta, std::sqrt(Vt), true);

        } // One-step-ahead predictive density
        else
        {
            arma::vec Ft_gtheta = LBA::func_Ft(model.transfer, t_new, gtheta_old_i, y);
            double ft_tilde = ft_gtheta - arma::as_scalar(Ft_gtheta.t() * gtheta_old_i); // (eq 3.8)
            double delta = yhat_new - par.at(0) - ft_tilde; // (eq 3.16)

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
    bound_check<arma::vec>(weights, "imp_weights_forecast");

    return weights;
} // func: imp_weights_forecast



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
    TransFunc ftrans = model.transfer;
    const unsigned int nP = Wt.n_elem;
    const unsigned int nT = y.n_elem - 1;
    // arma::mat mu_marginal(dim.nP, dim.nT + 1, arma::fill::zeros);
    // arma::cube Sigma_marginal(dim.nP, dim.nP, dim.nT + 1);
    arma::mat sig = arma::eye<arma::mat>(nP, nP) * 2.;
    // arma::cube Prec_marginal = Sigma_marginal;
    prec.slice(0) = arma::eye<arma::mat>(nP, nP) * 0.5;


    for (unsigned int t = 1; t <= nT; t++)
    {
        mu.col(t) = StateSpace::func_gt(ftrans, mu.col(t - 1), y.at(t - 1));
        arma::mat Gt = LBA::func_Gt(model, mu.col(t - 1), y.at(t - 1));
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
    std::map<std::string, AVAIL::Transfer> trans_list = AVAIL::trans_list;
    arma::mat U_cur = K;
    arma::mat Uprec_cur = K;
    arma::mat Urchol_cur = K;

    if (trans_list[model.transfer.name] == AVAIL::sliding)
    {
        for (unsigned int i = 0; i < model.dim.nP - 1; i++)
        {
            K.at(i, i + 1) = 1.;
        }

        K.at(model.dim.nP - 1, model.dim.nP - 1) = 1.;
        U_cur.at(model.dim.nP - 1, model.dim.nP - 1) = Wt.at(0);
        Uprec_cur.at(model.dim.nP - 1, model.dim.nP - 1) = 1. / Wt.at(0);
        Urchol_cur.at(model.dim.nP - 1, model.dim.nP - 1) = std::sqrt(Wt.at(0));
        ldetU = std::log(Wt.at(0));
    }
    else
    {
        arma::mat G_next = LBA::func_Gt(model, vt.col(t_cur), y.at(t_cur));

        arma::mat Vchol = arma::chol(Vt_inv.slice(t_cur));
        arma::mat Vchol_inv = arma::inv(arma::trimatu(Vchol));
        arma::mat Vt_cur = Vchol_inv * Vchol_inv.t();

        arma::mat VGt = Vt_cur * G_next.t();
        K = VGt * Vt_inv.slice(t_cur + 1);
        U_cur = Vt_cur - VGt * Vt_inv.slice(t_cur + 1) * VGt.t();
        U_cur = arma::symmatu(U_cur);
        Uprec_cur = inverse(Urchol_cur, U_cur);
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
    const arma::vec &par, // m x 1, m = 4
    const arma::vec &y,
    const bool &full_rank = false
)
{
    std::map<std::string, AVAIL::Transfer> trans_list = AVAIL::trans_list;

    double yhat_cur = LinkFunc::mu2ft(y.at(t_cur), model.flink.name, 0.);

    unsigned int N = Theta_next.n_cols;
    unsigned int nP = Theta_next.n_rows;
    unsigned int t_next = t_cur + 1;

    arma::vec r_cur(model.dim.nP, arma::fill::zeros);
    arma::mat K_cur(model.dim.nP, model.dim.nP, arma::fill::zeros);
    arma::mat Uprec_cur = K_cur;
    double ldetU = 0.;
    backward_kernel(K_cur, r_cur, Uprec_cur, ldetU, model, t_cur, vt, Vt_inv, Wt, y);
    
    #pragma omp parallel for num_threads(NUM_THREADS) schedule(runtime)
    for (unsigned int i = 0; i < N; i++)
    {
        arma::vec u_cur = K_cur * Theta_next.col(i) + r_cur;
        double ft_ut = StateSpace::func_ft(model.transfer, t_cur, u_cur, y);
        double eta = model.dobs.par1 + ft_ut;
        double lambda = LinkFunc::ft2mu(eta, model.flink.name, 0.); // (eq 3.58)
        if (lambda < EPS)
        {
            std::cerr << "\n lambda = " << lambda << ", mu0 = " << model.dobs.par1 << ", ft_ut = " << ft_ut;
            throw std::runtime_error("\n SMC::SequentialMonteCarlo::im_weights_backast: observation mean lambda should not be zero.\n");
        }
        double Vtilde = ApproxDisturbance::func_Vt_approx(
            lambda, model.dobs, model.flink.name); // (eq 3.59)
        Vtilde = std::abs(Vtilde) + EPS;


        if (!full_rank)
        {
            // No information from data, degenerates to the backward evolution
            loc.col(i) = u_cur;
            logq.at(i) = R::dnorm4(yhat_cur, eta, std::sqrt(Vtilde), true);
        } // one-step backcasting
        else
        {
            arma::vec F_cur = LBA::func_Ft(model.transfer, t_cur, u_cur, y);
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

    try
    {
        bound_check<arma::vec>(weights, "imp_weights_forecast");
    }
    catch (const std::exception &e)
    {
        logq.t().brief_print("\n logarithm of weights: ");
        throw std::runtime_error(e.what());
    }

    return weights;
} // func: imp_weights_forecast (no unknown statics)

#endif