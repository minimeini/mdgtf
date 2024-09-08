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

static arma::vec qforecast0(
    arma::vec &logq,           // N x 1
    const Model &model,
    const unsigned int &t_new,  // current time t. The following inputs come from time t-1.
    const arma::mat &Theta_old, // p x N, {theta[t-1]}
    const arma::vec &y)
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

        logq.at(i) = R::dnorm4(yhat_new, eta, std::sqrt(Vt), true);
    }

    double logq_max = logq.max();
    logq.for_each([&logq_max](arma::vec::elem_type &val)
                  { val -= logq_max; });

    arma::vec weights = arma::exp(logq);

    #ifdef DGTF_DO_BOUND_CHECK
        bound_check<arma::vec>(weights, "qforecast");
    #endif
    return weights;
} // func: imp_weights_forecast




static arma::vec qforecast(
    arma::mat &loc,            // p x N
    arma::cube &Prec_chol_inv, // p x p x N
    arma::vec &logq,           // N x 1
    const Model &model,
    const unsigned int &t_new,  // current time t. The following inputs come from time t-1.
    const arma::mat &Theta_old, // p x N, {theta[t-1]}
    const arma::vec &W, // N x 1, place holder if using discount factor or W not updated
    const arma::mat &param, // (period + 3) x N, (seasonal components, rho, par1, par2), place holder if not updated
    const arma::vec &y, // (nT + 1) x 1
    const bool &update_W = false,
    const bool &update_obs = false,
    const bool &update_lag = false,
    const bool &use_discount = false, // ignored if `update_W = true`
    const double &discount_factor = 0.9) // ignored if `update_W = true`
{
    const double y_old = y.at(t_new - 1);
    const double yhat_new = LinkFunc::mu2ft(y.at(t_new), model.flink);

    arma::mat W_inv(model.nP, model.nP, arma::fill::zeros);
    double ldet_W = 0.;
    if (!update_W)
    {
        if (model.derr.full_rank)
        {
            arma::mat W_chol = arma::chol(arma::symmatu(model.derr.var));
            arma::mat W_chol_inv = arma::inv(arma::trimatu(W_chol));
            W_inv = W_chol_inv * W_chol_inv.t();
            ldet_W = arma::log_det_sympd(model.derr.var);
        }
        else
        {
            W_inv.at(0, 0) = 1. / model.derr.par1;
            ldet_W = std::log(std::abs(model.derr.par1) + EPS);
        }
    }

    #ifdef _OPENMP
        #pragma omp parallel for num_threads(NUM_THREADS) schedule(runtime)
    #endif
    for (unsigned int i = 0; i < Theta_old.n_cols; i++)
    {
        Model mod = model;
        arma::mat Winv = W_inv; // p x p
        double ldetW = ldet_W;

        if (update_obs)
        {
            mod.seas.val = param.submat(0, i, mod.seas.period - 1, i);
            mod.dobs.par2 = std::exp(param.at(mod.seas.period, i));
        }

        if (update_lag)
        {
            mod.dlag.par1 = param.at(mod.seas.period + 1, i);
            mod.dlag.par2 = std::exp(param.at(mod.seas.period + 2, i));
        }

        if (update_W)
        {
            // Checked. OK.
            // discount factor settings are ignored in this case
            // Only applies to univariate rw for now.
            mod.derr.par1 = W.at(i);
            mod.derr.var.at(0, 0) = W.at(i);
            Winv.at(0, 0) = 1. / W.at(i);
            ldetW = std::log(std::abs(W.at(i)) + EPS);
        }
        else if (use_discount && (update_obs || update_lag))
        {
            LBA::LinearBayes lba(use_discount, discount_factor);
            lba.filter(mod, y);
            arma::mat Gt = TransFunc::init_Gt(mod.nP, mod.dlag, mod.ftrans, mod.seas.period, mod.seas.in_state);
            LBA::func_Gt(Gt, mod, lba.mt.col(t_new - 1), y_old);
            arma::mat Pt = Gt * lba.Ct.slice(t_new - 1) * Gt.t();
            arma::mat What = (1. / discount_factor - 1.) * Pt;
            What.diag() += EPS8;

            if (mod.derr.full_rank)
            {
                mod.derr.var = What;
                arma::mat W_chol = arma::chol(arma::symmatu(What));
                arma::mat W_chol_inv = arma::inv(arma::trimatu(W_chol));
                Winv = W_chol_inv * W_chol_inv.t();
                ldetW = arma::log_det_sympd(mod.derr.var);
            }
            else
            {
                mod.derr.par1 = What.at(0, 0);
                mod.derr.var.at(0, 0) = What.at(0, 0);
                Winv.at(0, 0) = 1. / What.at(0, 0);
                ldetW = std::log(std::abs(What.at(0, 0)) + EPS);
            }
        }

        arma::vec gtheta_old_i = StateSpace::func_gt(mod.ftrans, mod.fgain, mod.dlag, Theta_old.col(i), y_old, mod.seas.period, mod.seas.in_state); // gt(theta[t-1, i])
        double ft_gtheta = StateSpace::func_ft(mod.ftrans, mod.fgain, mod.dlag, mod.seas, t_new, gtheta_old_i, y); // ft( gt(theta[t-1,i]) )
        double eta = ft_gtheta;
        double lambda = LinkFunc::ft2mu(eta, mod.flink); // (eq 3.10)
        lambda = (t_new == 1 && lambda < EPS) ? 1. : lambda;

        double Vt = ApproxDisturbance::func_Vt_approx(lambda, mod.dobs, mod.flink); // (eq 3.11)

        if (!mod.derr.full_rank)
        {
            loc.col(i) = gtheta_old_i;
            logq.at(i) = R::dnorm4(yhat_new, eta, std::sqrt(Vt), true);

        } // One-step-ahead predictive density
        else
        {
            arma::vec Ft_gtheta = LBA::func_Ft(mod.ftrans, mod.fgain, mod.dlag, t_new, gtheta_old_i, y, LBA_FILL_ZERO, mod.seas.period, mod.seas.in_state);
            arma::mat Prec_i = Ft_gtheta * Ft_gtheta.t() / Vt + Winv; // nP x nP, function of mu0[i, t]
            Prec_i.diag() += EPS;
            
            arma::mat prec_chol = arma::chol(arma::symmatu(Prec_i)); // Right cholesky of the precision
            arma::mat prec_chol_inv = arma::inv(arma::trimatu(prec_chol)); // Left cholesky of the variance
            double ldetPrec = arma::accu(arma::log(prec_chol.diag())) * 2.; // ldetSigma = - ldetPrec
            Prec_chol_inv.slice(i) = prec_chol_inv;

            double delta = yhat_new - ft_gtheta + arma::as_scalar(Ft_gtheta.t() * gtheta_old_i); // (eq 3.16)
            loc.col(i) = Ft_gtheta * (delta / Vt) + Winv * gtheta_old_i; // (eq 3.20)

            double ldetV = std::log(std::abs(Vt) + EPS);
            double loglik = LOG2PI + ldetV + ldetW + ldetPrec; // (eq 3.24)
            loglik += delta * delta / Vt;
            loglik += arma::as_scalar(gtheta_old_i.t() * Winv * gtheta_old_i);
            loglik -= arma::as_scalar(loc.col(i).t() * prec_chol_inv * prec_chol_inv.t() * loc.col(i));
            loglik *= -0.5; // (eq 3.24 - 3.25)

            logq.at(i) += loglik;
        } // one-step-ahead predictive density
    }

    double logq_max = logq.max();
    logq.for_each([&logq_max](arma::vec::elem_type &val)
                  { val -= logq_max; });

    arma::vec weights = arma::exp(logq);

    #ifdef DGTF_DO_BOUND_CHECK
        bound_check<arma::vec>(weights, "qforecast");
    #endif
    return weights;
} // func: qforecast




/**
 * @todo Is there a more efficient way to construct artificial priors?
 */
static void prior_forward(
    arma::mat &mu,     // nP x (nT + 1)
    arma::cube &prec,  // nP x nP x (nT + 1)
    const Model &model,
    const arma::vec &y,   // (nT + 1) x 1
    const arma::cube &Wt, // p x p x (nT + 1), only initialized if using discount factor
    const bool &use_discount = false
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
        sig.diag() += EPS;
        if (use_discount)
        {
            sig = sig + Wt.slice(t);
        }
        else if (model.derr.full_rank)
        {
            sig = sig + model.derr.var;
        }
        else
        {
            sig.at(0, 0) += model.derr.par1;
        }

        prec.slice(t) = inverse(sig);
    }

    return;
}



/**
 * @brief theta[t] ~ N( r[t] + K[t]theta[t+1], U[t] )
 * 
 * @param K p x p, K[t], transition matrix bring theta[t+1] to theta[t] (slope of the backward kernel).
 * @param r p x 1, r[t], the intercept of the backward kernel.
 * @param Uinv  p x p, inv(U[t]), precision matrix of the backward kernel.
 * @param ldetU scalar, log determinant of U[t]
 * @param model 
 * @param t_cur Time index of theta[t]
 * @param vt p x 1, v[t], mean of the artificial normal prior for theta[t]
 * @param Vt_inv p x p, inv(V[t]), precision matrix of the artificial normal prior for theta[t]
 * @param theta_hat p x p, point of theta[t] for taylor expansion
 * @param y (nT + 1) x 1
 */
static void backward_kernel(
    arma::mat &K,
    arma::vec &r,
    arma::mat &Uinv,
    double &ldetU,
    const Model &model,
    const unsigned int &t_cur,
    const arma::mat &vt,  // nP x (nT + 1)
    const arma::cube &Vt_inv, // nP x nP x (nT + 1)
    const arma::vec &theta_hat, // p x p, point of theta[t] for taylor expansion
    const arma::vec &y)
{
    unsigned int nstate = model.nP;
    if (model.seas.in_state)
    {
        nstate -= model.seas.period;
    }
    std::map<std::string, TransFunc::Transfer> trans_list = TransFunc::trans_list;
    
    if (trans_list[model.ftrans] == TransFunc::sliding && !model.derr.full_rank)
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

        Uinv.set_size(model.nP, model.nP);
        Uinv.zeros();
        Uinv.at(nstate - 1, nstate - 1) = 1. / model.derr.par1;
        ldetU = std::log(model.derr.par1);
        r = vt.col(t_cur) - K * vt.col(t_cur + 1);
    }
    else if (model.derr.full_rank)
    {
        arma::mat G_next = TransFunc::init_Gt(model.nP, model.dlag, model.ftrans, model.seas.period, model.seas.in_state);
        LBA::func_Gt(G_next, model, vt.col(t_cur), y.at(t_cur));

        arma::mat W_chol = arma::chol(arma::symmatu(model.derr.var));
        arma::mat W_chol_inv = arma::inv(arma::trimatu(W_chol));
        arma::mat W_inv = W_chol_inv * W_chol_inv.t();

        Uinv = Vt_inv.slice(t_cur) + G_next.t() * W_inv * G_next; // inv(U[t])
        Uinv.diag() += EPS;
        arma::mat U_inv_chol = arma::chol(arma::symmatu(Uinv));
        arma::mat U_chol = arma::inv(arma::trimatu(U_inv_chol));
        arma::mat U = U_chol * U_chol.t(); // U[t]
        ldetU = arma::log_det_sympd(arma::symmatu(U));

        K = U * G_next.t() * W_inv; // K[t]

        arma::vec ghat = StateSpace::func_gt(
            model.ftrans, model.fgain, model.dlag, 
            theta_hat, y.at(t_cur), 
            model.seas.period, model.seas.in_state);
        arma::vec ht = ghat - G_next * theta_hat;
        r = U * (Vt_inv.slice(t_cur) * vt.col(t_cur) - G_next.t() * W_inv * ht);
    }
    else
    {
        throw std::invalid_argument("backward_kernel: Wt must be either univaraite sliding trans or full-rank (discount is ok).");
    }

    return;
}

static arma::vec qbackcast(
    arma::mat &loc,          // p x N, mean of the posterior of theta[t_cur] | y[t_cur:nT], theta[t_next], W
    arma::cube &Prec_chol_inv,       // p x p x N, precision of the posterior of theta[t_cur] | y[t_cur:nT], theta[t_next], W
    arma::vec &logq,        // N x 1
    const Model &model,
    const unsigned int &t_cur,   // current time "t". The following inputs come from time t+1. t_next = t + 1; t_prev = t - 1
    const arma::mat &Theta_next, // p x N, {theta[t+1]}
    const arma::vec &r_cur, // p x 1, r[t]
    const arma::mat &K_cur, // p x p, K[t]
    const arma::mat &Uprec_cur, // p x p, inv(U[t])
    const double &ldetU,
    const arma::vec &y
)
{
    std::map<std::string, TransFunc::Transfer> trans_list = TransFunc::trans_list;
    double yhat_cur = LinkFunc::mu2ft(y.at(t_cur), model.flink);
    unsigned int N = Theta_next.n_cols;
    unsigned int t_next = t_cur + 1;

    #ifdef _OPENMP
        #pragma omp parallel for num_threads(NUM_THREADS) schedule(runtime)
    #endif
    for (unsigned int i = 0; i < N; i++)
    {
        arma::vec u_cur = K_cur * Theta_next.col(i) + r_cur;
        double ft_ut = StateSpace::func_ft(model.ftrans, model.fgain, model.dlag, model.seas, t_cur, u_cur, y);
        double eta = ft_ut;
        double lambda = LinkFunc::ft2mu(eta, model.flink); // (eq 3.58)
        double Vtilde = ApproxDisturbance::func_Vt_approx(
            lambda, model.dobs, model.flink); // (eq 3.59)
        Vtilde = std::abs(Vtilde) + EPS;

        if (!model.derr.full_rank)
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

            arma::mat prec_chol = arma::chol(arma::symmatu(Prec));
            arma::mat prec_chol_inv = arma::inv(arma::trimatu(prec_chol));
            double ldetPrec = arma::accu(arma::log(prec_chol.diag())) * 2.;
            Prec_chol_inv.slice(i) = prec_chol_inv;

            double delta = yhat_cur - eta + arma::as_scalar(F_cur.t() * u_cur);
            loc.col(i) = F_cur * (delta / Vtilde) + Uprec_cur * u_cur;

            double ldetV = std::log(Vtilde);
            double logq_pred = LOG2PI + ldetV + ldetU + ldetPrec; // (eq 3.63)
            logq_pred += delta * delta / Vtilde;
            logq_pred += arma::as_scalar(u_cur.t() * Uprec_cur * u_cur);
            logq_pred -= arma::as_scalar(loc.col(i).t() * prec_chol_inv * prec_chol_inv.t() * loc.col(i));
            logq_pred *= -0.5;

            logq.at(i) += logq_pred;
        }
    }

    double logq_max = logq.max();
    logq.for_each([&logq_max](arma::vec::elem_type &val)
                  { val -= logq_max; });
    arma::vec weights = arma::exp(logq);

    #ifdef DGTF_DO_BOUND_CHECK
        bound_check<arma::vec>(weights, "qbackcast");
    #endif
    return weights;
} // qbackcast

#endif