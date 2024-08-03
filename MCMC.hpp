#ifndef _MCMC_HPP
#define _MCMC_HPP

#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <algorithm>
// #include <chrono>
#include <RcppArmadillo.h>
#include "Model.hpp"
#include "LinkFunc.hpp"
#include "LinearBayes.hpp"

namespace MCMC
{
    class Posterior
    {
    public:
        /**
         * @brief Particle independent Metropolis-Hastings sampler (Andrieu, Doucet and Holenstein)
         * @todo DEBUGGING, NOT WORKING YET
         *
         * @param psi
         * @param psi_accept
         * @param log_marg
         * @param y
         * @param model
         * @param N
         * @return double
         */
        static void update_psi(
            arma::vec &psi, // (nT + 1)
            double &psi_accept,
            double &log_marg,
            const arma::vec &y,
            const Model &model,
            const unsigned int &N = 100)
        {
            arma::cube Theta = arma::zeros<arma::cube>(model.dim.nP, N, model.dim.nT + 1);
            arma::vec Wt(model.dim.nP, arma::fill::zeros);
            Wt.at(0) = model.derr.par1;

            const bool full_rank = false;
            arma::vec weights(N, arma::fill::zeros);
            arma::vec par = {
                model.dobs.par1, model.dobs.par2,
                model.transfer.dlag.par1, model.transfer.dlag.par2};

            double log_marg_new = 0.;
            const double logN = std::log(static_cast<double>(N));
            for (unsigned int t = 0; t < model.dim.nT; t++)
            {
                Rcpp::checkUserInterrupt();

                arma::vec logq(N, arma::fill::zeros);
                arma::mat loc(model.dim.nP, N, arma::fill::zeros);
                arma::cube prec_chol_inv; // nP x nP x N
                arma::mat Theta_cur = Theta.slice(t);

                if (full_rank)
                {
                    prec_chol_inv = arma::zeros<arma::cube>(model.dim.nP, model.dim.nP, N); // nP x nP x N
                }

                arma::vec tau = qforecast(
                    loc, prec_chol_inv, logq, // sufficient statistics
                    model, t + 1, Theta_cur,  // theta needs to be resampled
                    Wt, par, y);

                tau = weights % tau;
                arma::uvec resample_idx = SMC::SequentialMonteCarlo::get_resample_index(tau);

                Theta_cur = Theta_cur.cols(resample_idx);
                loc = loc.cols(resample_idx);
                if (full_rank)
                {
                    prec_chol_inv = prec_chol_inv.slices(resample_idx);
                }

                logq = logq.elem(resample_idx);

                // Propagate
                arma::mat Theta_new(model.dim.nP, N, arma::fill::zeros);
                arma::vec theta_cur = arma::vectorise(Theta_cur.row(0)); // N x 1
                #pragma omp parallel for num_threads(NUM_THREADS) schedule(runtime)
                for (unsigned int i = 0; i < N; i++)
                {
                    arma::vec theta_new;
                    if (full_rank)
                    {
                        arma::vec eps = arma::randn(Theta_new.n_rows);
                        arma::vec zt = prec_chol_inv.slice(i).t() * loc.col(i) + eps; // shifted
                        theta_new = prec_chol_inv.slice(i) * zt;                      // scaled

                        logq.at(i) += MVNorm::dmvnorm0(zt, loc.col(i), prec_chol_inv.slice(i), true);
                    }
                    else
                    {
                        theta_new = loc.col(i);
                        double eps = R::rnorm(0., std::sqrt(Wt.at(0)));
                        theta_new.at(0) += eps;
                        logq.at(i) += R::dnorm4(eps, 0., std::sqrt(Wt.at(0)), true); // sample from evolution distribution
                    }

                    Theta_new.col(i) = theta_new;

                    double logp = R::dnorm4(theta_new.at(0), theta_cur.at(i), std::sqrt(Wt.at(0)), true);
                    double ft = StateSpace::func_ft(model.transfer, t + 1, theta_new, y);
                    double lambda = LinkFunc::ft2mu(ft, model.flink.name, par.at(0));
                    logp += ObsDist::loglike(y.at(t + 1), model.dobs.name, lambda, model.dobs.par2, true);
                    weights.at(i) = std::exp(logp - logq.at(i));
                }

                double eff = SMC::SequentialMonteCarlo::effective_sample_size(weights);
                Theta.slice(t + 1) = Theta_new;
                if (eff < 0.95 * N)
                {
                    resample_idx = SMC::SequentialMonteCarlo::get_resample_index(weights);
                    Theta.slice(t + 1) = Theta.slice(t + 1).cols(resample_idx);
                    weights.ones();
                }

                log_marg_new += std::log(arma::accu(weights) + EPS) - logN;
            }

            arma::vec psi_new = arma::mean(Theta.row_as_mat(0), 1); // (nT + 1) x 1

            double logratio = log_marg_new - log_marg;
            logratio = std::min(0., logratio);
            if (std::log(R::runif(0., 1.)) < logratio)
            {
                // accept
                psi = psi_new;
                log_marg = log_marg_new;
                psi_accept += 1;
            }

            return;
        }

        static void update_wt( // Checked. OK.
            arma::vec &wt,     // (nT + 1) x 1
            arma::vec &wt_accept,
            ApproxDisturbance &approx_dlm,
            const arma::vec &y, // (nT + 1) x 1
            Model &model,
            const Dist &w0_prior,
            const double &mh_sd = 0.1)
        {
            arma::vec ft(model.dim.nT + 1, arma::fill::zeros);
            double prior_sd = std::sqrt(w0_prior.par2);

            for (unsigned int t = 1; t <= model.dim.nT; t++)
            {
                double wt_old = wt.at(t);
                arma::vec lam = model.wt2lambda(y, wt); // Checked. OK.

                double logp_old = 0.;
                for (unsigned int i = t; i <= model.dim.nT; i++)
                {
                    logp_old += ObsDist::loglike(y.at(i), model.dobs.name, lam.at(i), model.dobs.par2, true);
                } // Checked. OK.

                logp_old += R::dnorm4(wt_old, w0_prior.par1, prior_sd, true);

                /*
                Metropolis-Hastings
                */
                approx_dlm.update_by_wt(y, wt);
                arma::vec eta = approx_dlm.get_eta_approx(model.dobs.par1);               // nT x 1, f0, Fn and psi is updated
                arma::vec lambda = LinkFunc::ft2mu<arma::vec>(eta, model.flink.name, 0.); // nT x 1
                arma::vec Vt_hat = ApproxDisturbance::func_Vt_approx(
                    lambda, model.dobs, model.flink.name); // nT x 1

                arma::mat Fn = approx_dlm.get_Fn(); // nT x nT
                arma::vec Fnt = Fn.col(t - 1);
                arma::vec Fnt2 = Fnt % Fnt;

                arma::vec tmp = Fnt2 / Vt_hat;
                double mh_prec = arma::accu(tmp);
                // mh_prec = std::abs(mh_prec) + 1. / w0_prior.par2 + EPS;

                double Bs = 1. / mh_prec;
                double Btmp = std::sqrt(Bs);
                // double Btmp = prior_sd;
                Btmp *= mh_sd;
                // Btmp = std::min(Btmp, 10.);

                double wt_new = R::rnorm(wt_old, Btmp); // Sample from MH proposal
                // bound_check(wt_new, "Posterior::update_wt: wt_new");
                /*
                Metropolis-Hastings
                */

                wt.at(t) = wt_new;
                lam = model.wt2lambda(y, wt); // Checked. OK.

                double logp_new = 0.;
                for (unsigned int i = t; i <= model.dim.nT; i++)
                {
                    logp_new += ObsDist::loglike(y.at(i), model.dobs.name, lam.at(i), model.dobs.par2, true);
                } // Checked. OK.

                logp_new += R::dnorm4(wt_new, w0_prior.par1, prior_sd, true); // prior

                double logratio = logp_new - logp_old;
                // logratio += logq_old - logq_new;
                logratio = std::min(0., logratio);

                double logps = 0.;
                if (std::log(R::runif(0., 1.)) < logratio)
                {
                    // accept
                    logps = logp_new;
                    wt_accept.at(t) += 1.;
                }
                else
                {
                    // reject
                    logps = logp_old;
                    wt.at(t) = wt_old;
                }
            }
        } // func update_wt

        static double update_W( // Checked. OK.
            double &W_accept,
            const double &W_old,
            const arma::vec &wt,
            const Dist &W_prior,
            const double &mh_sd = 1.)
        {
            double W = W_old;
            double res = arma::accu(arma::pow(wt.tail(wt.n_elem - 2), 2.));

            // double bw_prior = prior_params.at(1); // eta_prior_val.at(1, 0)
            std::map<std::string, AVAIL::Dist> dist_list = AVAIL::dist_list;

            switch (dist_list[W_prior.name])
            {
            case AVAIL::Dist::gamma:
            {
                double aw_new = W_prior.par1;
                double bw_prior = W_prior.par2;

                double logp_old = aw_new * std::log(W_old) - bw_prior * W_old - 0.5 * res / W_old;
                double W_new = std::exp(std::min(R::rnorm(std::log(W_old), mh_sd), UPBND));
                double logp_new = aw_new * std::log(W_new) - bw_prior * W_new - 0.5 * res / W_new;
                double logratio = std::min(0., logp_new - logp_old);
                if (std::log(R::runif(0., 1.)) < logratio)
                { // accept
                    W = W_new;
                    W_accept += 1.;
                }
                break;
            }
            case AVAIL::Dist::invgamma:
            {
                double nSw_new = W_prior.par2 + res;                  // prior_params.at(1) = nSw
                W = 1. / R::rgamma(0.5 * W_prior.par1, 2. / nSw_new); // prior_params.at(0) = nw_new
                W_accept += 1.;
                break;
            }
            default:
            {
                break;
            }
            }

            bound_check(W, "update: W", true, true);
            return W;
        } // func update_W

        static double update_mu0( // flat prior
            double &mu0_accept,
            Model &model,
            const arma::vec &y, // nobs x 1
            const arma::vec &hpsi,
            const double &mh_sd = 1.,
            const double &min_var = 0.1)
        {
            // double mu0_old = mu0;
            double mu0_old = model.dobs.par1;
            double mu0 = mu0_old;

            arma::vec Vt_hat(model.dim.nT + 1, arma::fill::zeros);
            arma::vec lambda(model.dim.nT + 1, arma::fill::zeros);
            arma::vec eta(model.dim.nT + 1, arma::fill::zeros);

            double logp_old = 0.;
            for (unsigned int t = 1; t <= model.dim.nT; t++)
            {
                eta.at(t) = TransFunc::transfer_sliding(t, model.dim.nL, y, model.transfer.dlag.Fphi, hpsi);
                lambda.at(t) = LinkFunc::ft2mu(eta.at(t), model.flink.name, mu0_old);

                logp_old += ObsDist::loglike(y.at(t), model.dobs.name, lambda.at(t), model.dobs.par2, true);
                // Vt_hat.at(t) = ApproxDisturbance::func_Vt_approx(lambda.at(t), model.dobs, model.flink.name);
            }

            Vt_hat = ApproxDisturbance::func_Vt_approx(lambda, model.dobs, model.flink.name);

            arma::vec tmp = 1. / Vt_hat;
            double mu0_prec = arma::accu(tmp);
            double mu0_var = 1. / mu0_prec;
            double mu0_sd = std::sqrt(mu0_var);
            mu0_sd = std::max(mu0_sd, min_var);
            mu0_sd *= mh_sd;
            bound_check(mu0_sd, "MCMC::posterior::update_mu0: mu0_sd", true, true);

            double mu0_new = R::rnorm(mu0_old, mu0_sd);

            // logp_mu0 = logp_old;
            if (mu0_new > -EPS) // non-negative
            {
                mu0_new = std::abs(mu0_new);
                double logp_new = 0.;
                for (unsigned int t = 1; t <= model.dim.nT; t++)
                {
                    lambda.at(t) = LinkFunc::ft2mu(eta.at(t), model.flink.name, mu0_new);
                    logp_old += ObsDist::loglike(y.at(t), model.dobs.name, lambda.at(t), model.dobs.par2, true);
                }

                double logratio = std::min(0., logp_new - logp_old);

                if (std::log(R::runif(0., 1.)) < logratio)
                { // accept
                    model.dobs.update_par1(mu0_new);
                    mu0 = mu0_new;
                    mu0_accept += 1.;
                    // logp_mu0 = logp_new;
                }
            }

            bound_check(mu0, "Posterior::update_mu0");
            return mu0;
        } // func update_mu0

        static double update_dispersion( // Checked. OK.
            double &rho_accept,
            Model &model,
            const arma::vec &y, // nobs x 1
            const arma::vec &hpsi,
            const Dist &rho_prior,
            const double &mh_sd = 1.)
        {
            double rho_old = model.dobs.par2;
            double logp_old = R::dgamma(rho_old, rho_prior.par1, 1. / rho_prior.par2, true);
            arma::vec lambda(model.dim.nT + 1, arma::fill::zeros);
            for (unsigned int t = 1; t <= model.dim.nT; t++)
            {
                double eta = TransFunc::transfer_sliding(t, model.dim.nL, y, model.transfer.dlag.Fphi, hpsi);
                lambda.at(t) = LinkFunc::ft2mu(eta, model.flink.name, model.dobs.par1);

                logp_old += ObsDist::loglike(y.at(t), model.dobs.name, lambda.at(t), rho_old, true);
            }

            // double log_rho_old = std::log(std::abs(rho_old) + EPS);
            // double log_rho_new;
            // bool success = false;
            // unsigned int cnt = 0;
            // while (!success && cnt < MAX_ITER)
            // {
            //     log_rho_new = R::rnorm(log_rho_old, mh_sd);
            //     success = (log_rho_new < UPBND) ? true : false;
            // }

            // double rho_new = std::exp(log_rho_new);

            bool success = false;
            double rho_new;
            unsigned int cnt = 0;
            while (!success && cnt < MAX_ITER)
            {
                rho_new = R::rnorm(rho_old, mh_sd * rho_old); // mh_sd here is the coefficient of variation, i.e., sd/mean.
                success = (rho_new > 0) ? true : false;
                cnt++;
            }

            double logp_new = R::dgamma(rho_new, rho_prior.par1, 1. / rho_prior.par2, true);
            for (unsigned int t = 1; t <= model.dim.nT; t++)
            {
                logp_new += ObsDist::loglike(y.at(t), model.dobs.name, lambda.at(t), rho_new, true);
            }
            double logratio = std::min(0., logp_new - logp_old);

            double rho = rho_old;
            if (std::log(R::runif(0., 1.)) < logratio)
            { // accept
                rho = rho_new;
                model.dobs.update_par2(rho_new);
                rho_accept += 1.;
                // logp_mu0 = logp_new;
            }

            bound_check(rho, "Posterior::update_mu0");
            return rho;
        } // func update_dispersion

        static arma::vec update_dlag(
            double &par1_accept,
            double &par2_accept,
            Model &model,
            const arma::vec &y, // nobs x 1
            const arma::vec &hpsi,
            const Dist &par1_prior,
            const Dist &par2_prior,
            const double &par1_mh_sd = 0.1,
            const double &par2_mh_sd = 0.1,
            const unsigned int &max_lag = 30)
        {
            std::map<std::string, AVAIL::Dist> dist_list = AVAIL::dist_list;
            double par1_old = model.transfer.dlag.par1;
            double par2_old = model.transfer.dlag.par2;

            double loglik_old = 0.;
            for (unsigned int t = 1; t <= model.dim.nT; t++)
            {
                double eta = TransFunc::transfer_sliding(t, model.dim.nL, y, model.transfer.dlag.Fphi, hpsi);
                double lambda = LinkFunc::ft2mu(eta, model.flink.name, model.dobs.par1);
                loglik_old += ObsDist::loglike(y.at(t), model.dobs.name, lambda, model.dobs.par2, true);
            }
            double logp_old = loglik_old;

            double logp_new = 0.;
            double par1_new = par1_old;
            double logprior_par1_old = 0.;
            double logprior_par1_new = 0.;
            if (par1_prior.infer)
            {
                logprior_par1_old = Prior::dprior(par1_old, par1_prior, true, false);
                logp_old += logprior_par1_old;

                bool non_gaussian = !dist_list[par1_prior.name] == AVAIL::Dist::gaussian;
                if (non_gaussian) // non-negative
                {
                    bool success = false;
                    unsigned int cnt = 0;
                    while (!success && cnt < MAX_ITER)
                    {
                        par1_new = R::rnorm(par1_old, par1_mh_sd * par1_old);
                        success = (par1_new > 0) ? true : false;
                        cnt++;
                    }
                }
                else // gaussian
                {
                    par1_new = R::rnorm(par1_old, par1_mh_sd * par1_old);
                }

                logprior_par1_new = Prior::dprior(par1_new, par1_prior, true, false);
                logp_new += logprior_par1_new;
            } // par1

            double par2_new = par2_old;
            double logprior_par2_old = 0.;
            double logprior_par2_new = 0.;
            if (par2_prior.infer)
            {
                logprior_par2_old = Prior::dprior(par2_old, par2_prior, true, false);
                logp_old += logprior_par2_old;

                bool non_gaussian = !dist_list[par2_prior.name] == AVAIL::Dist::gaussian;
                if (non_gaussian) // non-negative
                {
                    bool success = false;
                    unsigned int cnt = 0;
                    while (!success && cnt < MAX_ITER)
                    {
                        par2_new = R::rnorm(par2_old, par2_mh_sd * par2_old);
                        success = (par2_new > 0) ? true : false;
                        cnt++;
                    }
                }
                else // gaussian
                {
                    par2_new = R::rnorm(par2_old, par2_mh_sd * par2_old);
                }

                logprior_par2_new = Prior::dprior(par2_new, par2_prior, true, false);
                logp_new += logprior_par2_new;

            } // par2

            unsigned int nlag = LagDist::update_nlag(model.transfer.dlag.name, par1_new, par2_new, 0.99, max_lag);
            arma::vec Fphi_new = LagDist::get_Fphi(nlag, model.transfer.dlag.name, par1_new, par2_new);

            double loglik_new = 0.;
            for (unsigned int t = 1; t <= model.dim.nT; t++)
            {
                double eta = TransFunc::transfer_sliding(t, nlag, y, Fphi_new, hpsi);
                double lambda = LinkFunc::ft2mu(eta, model.flink.name, model.dobs.par1);
                loglik_new += ObsDist::loglike(y.at(t), model.dobs.name, lambda, model.dobs.par2, true);
            }
            logp_new += loglik_new;

            double logratio = std::min(0., logp_new - logp_old);

            arma::vec par = {par1_old, par2_old};
            if (std::log(R::runif(0., 1.)) < logratio)
            { // accept
                par1_accept += 1;
                par2_accept += 1;

                nlag = model.update_dlag(par1_new, par2_new, max_lag);
                par.at(0) = par1_new;
                par.at(1) = par2_new;

                // logp_mu0 = logp_new;
            }

            return par;

        } // update_lag

        static arma::vec update_dlag_hmc(
            double &par1_accept,
            double &par2_accept,
            Model &model,
            const arma::vec &y,
            const arma::vec &hpsi,
            const Dist &par1_prior,
            const Dist &par2_prior,
            const double &epsilon = 0.01,
            const unsigned int &L = 10,
            const Rcpp::NumericVector &m = Rcpp::NumericVector::create(1., 1.),
            const unsigned int &max_lag = 30)
        {
            std::string lag_dist = model.transfer.dlag.name;
            std::map<std::string, AVAIL::Dist> dist_list = AVAIL::dist_list;

            double par1_old = model.transfer.dlag.par1;
            double par2_old = model.transfer.dlag.par2;

            // log of conditional posterior of lag parameters.
            double logp_old = 0.;
            logp_old += Prior::dprior(par1_old, par1_prior, true, true); // TODO: check it
            logp_old += Prior::dprior(par2_old, par2_prior, true, true); // TODO: check it

            for (unsigned int t = 1; t <= model.dim.nT; t++)
            {
                double eta = TransFunc::transfer_sliding(t, model.dim.nL, y, model.transfer.dlag.Fphi, hpsi);
                double lambda = LinkFunc::ft2mu(eta, model.flink.name, model.dobs.par1);

                logp_old += ObsDist::loglike(y.at(t), model.dobs.name, lambda, model.dobs.par2, true);
            }

            double current_U = -logp_old;

            // map parameters of the lag distribution to the whole real line
            arma::vec q(2, arma::fill::zeros);
            q.at(0) = Prior::val2real(par1_old, par1_prior.name, false);
            q.at(1) = Prior::val2real(par2_old, par2_prior.name, false);

            // sample an initial momentum
            arma::vec p = arma::randn(2);
            p.at(0) *= std::sqrt(m[0]);
            p.at(1) *= std::sqrt(m[1]);

            // Kinetic: negative logprob of the momentum distribution
            double current_K = 0.5 * (std::pow(p.at(0), 2.) / m[0]);
            current_K += 0.5 * (std::pow(p.at(1), 2.) / m[1]);

            // half step update of momentum
            unsigned int nlag = model.dim.nL;
            arma::vec Fphi_new = model.transfer.dlag.Fphi;

            arma::vec grad_U = Model::dloglik_dpar(
                Fphi_new, y, hpsi, nlag,
                lag_dist, par1_old, par2_old,
                model.dobs, model.flink.name);

            grad_U.at(0) += Prior::dlogprior_dpar(par1_old, par1_prior, true);
            grad_U.at(0) *= -1.;
            grad_U.at(1) += Prior::dlogprior_dpar(par2_old, par2_prior, true);
            grad_U.at(1) *= -1;

            double par1_new = 0.;
            double par2_new = 0.;
            p = p - (0.5 * epsilon) * grad_U;

            for (unsigned int i = 1; i <= L; i++)
            {
                // full step update for position
                q = q + epsilon * p;
                par1_new = Prior::val2real(q.at(0), par1_prior.name, true);
                par2_new = Prior::val2real(q.at(1), par2_prior.name, true);

                nlag = LagDist::update_nlag(lag_dist, par1_new, par2_new, 0.99, max_lag);
                grad_U = Model::dloglik_dpar(
                    Fphi_new, y, hpsi, nlag,
                    lag_dist, par1_new, par2_new,
                    model.dobs, model.flink.name);
                grad_U.at(0) += Prior::dlogprior_dpar(par1_new, par1_prior, true);
                grad_U.at(0) *= -1.;
                grad_U.at(1) += Prior::dlogprior_dpar(par2_new, par2_prior, true);
                grad_U.at(1) *= -1;

                // full step update for momentum
                if (i != L)
                {
                    p = p - epsilon * grad_U;
                }
            }

            // half step update for momentum
            p = p - (0.5 * epsilon) * grad_U;
            p = -p; // negate momentum for symmetry

            // log of conditional posterior for the proposed lag parameters.
            double logp_new = 0.;
            for (unsigned int t = 1; t <= model.dim.nT; t++)
            {
                double eta = TransFunc::transfer_sliding(t, nlag, y, Fphi_new, hpsi);
                double lambda = LinkFunc::ft2mu(eta, model.flink.name, model.dobs.par1);

                logp_new += ObsDist::loglike(y.at(t), model.dobs.name, lambda, model.dobs.par2, true);
            }
            logp_new += Prior::dprior(par1_new, par1_prior, true, true);
            logp_new += Prior::dprior(par2_new, par2_prior, true, true);

            double proposed_U = -logp_new;
            double proposed_K = 0.5 * (std::pow(p.at(0), 2.) / m[0]);
            proposed_K += 0.5 * (std::pow(p.at(1), 2.) / m[1]);

            // accept / reject
            double logratio = current_U + current_K;
            logratio -= (proposed_U + proposed_K);

            arma::vec par = {par1_old, par2_old};
            if (std::log(R::runif(0., 1.)) < logratio)
            { // accept
                par1_accept += 1;
                par2_accept += 1;

                nlag = model.update_dlag(par1_new, par2_new, max_lag);
                par.at(0) = par1_new;
                par.at(1) = par2_new;

                // logp_mu0 = logp_new;
            }

            return par;
        }

    }; // class Posterior

    class Disturbance
    {
    public:
        Disturbance()
        {
            dim.init_default();
            y.set_size(dim.nT + 1);
            y.zeros();
        }

        Disturbance(const Model &model, const arma::vec &y_in)
        {
            dim = model.dim;
            y = y_in;
        }

        void init(const Rcpp::List &mcmc_settings)
        {
            Rcpp::List opts = mcmc_settings;

            epsilon = 0.01;
            if (opts.containsElementNamed("epsilon"))
            {
                epsilon = Rcpp::as<double>(opts["epsilon"]);
            }

            L = 10;
            if (opts.containsElementNamed("L"))
            {
                L = Rcpp::as<unsigned int>(opts["L"]);
            }

            m = Rcpp::NumericVector::create(1., 1.);
            if (opts.containsElementNamed("m"))
            {
                m = Rcpp::as<Rcpp::NumericVector>(opts["m"]);
            }

            max_lag = 30;
            if (opts.containsElementNamed("max_lag"))
            {
                max_lag = Rcpp::as<unsigned int>(opts["max_lag"]);
            }

            mh_sd = 0.01;
            if (opts.containsElementNamed("mh_sd"))
            {
                mh_sd = Rcpp::as<double>(opts["mh_sd"]);
            }

            nburnin = 100;
            if (opts.containsElementNamed("nburnin"))
            {
                nburnin = Rcpp::as<unsigned int>(opts["nburnin"]);
            }
            nthin = 1;
            if (opts.containsElementNamed("nthin"))
            {
                nthin = Rcpp::as<unsigned int>(opts["nthin"]);
            }

            nsample = 100;
            if (opts.containsElementNamed("nsample"))
            {
                nsample = Rcpp::as<unsigned int>(opts["nsample"]);
            }

            ntotal = nburnin + nthin * nsample + 1;

            nforecast = 0;
            if (opts.containsElementNamed("num_step_ahead_forecast"))
            {
                nforecast = Rcpp::as<unsigned int>(opts["num_step_ahead_forecast"]);
            }

            nforecast_err = 10; // forecasting for indices (1, ..., ntime-1) has `nforecast` elements
            if (opts.containsElementNamed("num_eval_forecast_error"))
            {
                nforecast_err = Rcpp::as<unsigned int>(opts["num_eval_forecast_error"]);
            }

            tstart_pct = 0.9;
            if (opts.containsElementNamed("tstart_pct"))
            {
                tstart_pct = Rcpp::as<double>(opts["tstart_pct"]);
            }

            W = 0.01;
            bool infer_W = true;
            W_prior.init("invgamma", 0.01, 0.01);
            W_stored.set_size(nsample);
            W_accept = 0.;
            if (opts.containsElementNamed("W"))
            {
                Rcpp::List Wopts = Rcpp::as<Rcpp::List>(opts["W"]);
                init_param(infer_W, W, W_prior, Wopts);
            }

            mu0 = 0.;
            mu0_prior.init("gaussian", 0., 10.);
            mu0_stored.set_size(nsample);
            bool infer_mu0 = false;
            mu0_accept = 0.;
            mu0_mh_sd = 1.;
            if (opts.containsElementNamed("mu0"))
            {
                Rcpp::List param_opts = Rcpp::as<Rcpp::List>(opts["mu0"]);
                init_param(infer_mu0, mu0, mu0_prior, param_opts);

                if (param_opts.containsElementNamed("mh_sd"))
                {
                    mu0_mh_sd = Rcpp::as<double>(param_opts["mh_sd"]);
                }
            }

            rho = 30.;
            rho_stored.set_size(nsample);
            bool infer_rho = false;
            rho_prior.init("gamma", 0.1, 0.1);
            rho_accept = 0.;
            rho_mh_sd = 0.1;
            if (opts.containsElementNamed("rho"))
            {
                Rcpp::List param_opts = Rcpp::as<Rcpp::List>(opts["rho"]);
                init_param(infer_rho, rho, rho_prior, param_opts);

                if (param_opts.containsElementNamed("mh_sd"))
                {
                    rho_mh_sd = Rcpp::as<double>(param_opts["mh_sd"]);
                }
            }

            par1 = 0.;
            par1_stored.set_size(nsample);
            bool infer_par1 = false;
            par1_prior.init("gamma", 0.1, 0.1);
            par1_accept = 0.;
            par1_mh_sd = 0.1;
            if (opts.containsElementNamed("par1"))
            {
                Rcpp::List param_opts = Rcpp::as<Rcpp::List>(opts["par1"]);
                init_param(infer_par1, par1, par1_prior, param_opts);

                if (param_opts.containsElementNamed("mh_sd"))
                {
                    par1_mh_sd = Rcpp::as<double>(param_opts["mh_sd"]);
                }
            }

            par2 = 0.;
            par2_stored.set_size(nsample);
            bool infer_par2 = false;
            par2_prior.init("gamma", 0.1, 0.1);
            par2_accept = 0.;
            par2_mh_sd = 0.1;
            if (opts.containsElementNamed("par2"))
            {
                Rcpp::List param_opts = Rcpp::as<Rcpp::List>(opts["par2"]);
                init_param(infer_par2, par2, par2_prior, param_opts);

                if (param_opts.containsElementNamed("mh_sd"))
                {
                    par2_mh_sd = Rcpp::as<double>(param_opts["mh_sd"]);
                }
            }

            wt = arma::randn(dim.nT + 1) * 0.01;
            wt.at(0) = 0.;
            wt.subvec(1, dim.nP) = arma::abs(wt.subvec(1, dim.nP));
            bound_check(wt, "Disturbance::init");

            wt_stored.set_size(dim.nT + 1, nsample);
            wt_accept.set_size(dim.nT + 1);
            wt_accept.zeros();

            w0_prior.init("gaussian", 0., W);

            double nw = W_prior.par1;
            double nSw = W_prior.par1 * W_prior.par2;
            double nw_new = nw + (double)wt.n_elem - 2.;
            W_prior.init(W_prior.name, nw_new, nSw);

            return;
        }

        static Rcpp::List default_settings()
        {
            Rcpp::List Wopts;
            Wopts["infer"] = true;
            Wopts["init"] = 0.01;
            Wopts["prior_param"] = Rcpp::NumericVector::create(0.01, 0.01);
            Wopts["prior_name"] = "invgamma";

            Rcpp::List mu0_opts;
            mu0_opts["infer"] = false;
            mu0_opts["init"] = 0.;
            mu0_opts["mh_sd"] = 1.;

            Rcpp::List rho_opts;
            rho_opts["infer"] = false;
            rho_opts["init"] = 30;
            rho_opts["mh_sd"] = 1.;
            rho_opts["prior_param"] = Rcpp::NumericVector::create(0.1, 0.1);
            rho_opts["prior_name"] = "gamma";

            Rcpp::List par1_opts;
            par1_opts["infer"] = false;
            par1_opts["init"] = 30;
            par1_opts["mh_sd"] = 1.;
            par1_opts["prior_param"] = Rcpp::NumericVector::create(0.1, 0.1);
            par1_opts["prior_name"] = "gamma";

            Rcpp::List par2_opts;
            par2_opts["infer"] = false;
            par2_opts["init"] = 30;
            par2_opts["mh_sd"] = 1.;
            par2_opts["prior_param"] = Rcpp::NumericVector::create(0.1, 0.1);
            par2_opts["prior_name"] = "gamma";

            Rcpp::List opts;
            opts["W"] = Wopts;
            opts["mu0"] = mu0_opts;
            opts["rho"] = rho_opts;
            opts["par1"] = par1_opts;
            opts["par2"] = par2_opts;

            opts["epsilon"] = 0.01;
            opts["L"] = 10;
            opts["m"] = Rcpp::NumericVector::create(1., 1.);
            opts["max_lag"] = 30;

            opts["mh_sd"] = 0.1;
            opts["nburnin"] = 100;
            opts["nthin"] = 1;
            opts["nsample"] = 100;

            opts["num_step_ahead_forecast"] = 0;
            opts["num_eval_forecast_error"] = 10;
            opts["tstart_pct"] = 0.9;

            return opts;
        }

        Rcpp::List get_output()
        {
            Rcpp::List output;
            output["model"] = Rcpp::wrap(model_info);

            output["wt"] = Rcpp::wrap(wt_stored); // (nT + 1) x nsample

            arma::mat psi_stored = arma::cumsum(wt_stored, 0); // (nT + 1) x nsample
            arma::vec qprob = {0.025, 0.5, 0.975};
            arma::mat psi_quantile = arma::quantile(psi_stored, qprob, 1); // (nT + 1) x 3
            output["psi"] = Rcpp::wrap(psi_quantile);
            output["wt_accept"] = Rcpp::wrap(wt_accept / ntotal);

            output["infer_W"] = W_prior.infer;
            output["W"] = Rcpp::wrap(W_stored);
            output["W_accept"] = W_accept / ntotal;

            output["infer_mu0"] = mu0_prior.infer;
            output["mu0"] = Rcpp::wrap(mu0_stored);
            output["mu0_accept"] = static_cast<double>(mu0_accept / ntotal);

            output["infer_rho"] = rho_prior.infer;
            output["rho"] = Rcpp::wrap(rho_stored);
            output["rho_accept"] = rho_accept / ntotal;

            output["infer_par1"] = par1_prior.infer;
            output["par1"] = Rcpp::wrap(par1_stored);
            output["par1_accept"] = par1_accept / ntotal;

            output["infer_par2"] = par2_prior.infer;
            output["par2"] = Rcpp::wrap(par2_stored);
            output["par2_accept"] = par2_accept / ntotal;

            output["model"] = model_info;

            return output;
        }

        Rcpp::List forecast(const Model &model)
        {
            arma::mat psi_stored = arma::cumsum(wt_stored, 0); // (nT + 1) x nsample
            Rcpp::List out = Model::forecast(
                y, psi_stored, W_stored, model, nforecast);

            return out;
        }


        Rcpp::List fitted_error(const Model &model, const std::string &loss_func = "quadratic")
        {
            arma::mat psi_stored = arma::cumsum(wt_stored, 0); // (nT + 1) x nsample
            return Model::fitted_error(psi_stored, y, model, loss_func);
        }

        void fitted_error(double &err, const Model &model, const std::string &loss_func = "quadratic")
        {
            arma::mat psi_stored = arma::cumsum(wt_stored, 0); // (nT + 1) x nsample
            Model::fitted_error(err, psi_stored, y, model, loss_func);
            return;
        }

        void infer(Model &model, const bool &verbose = VERBOSE)
        {
            model_info = model.info();

            if (!mu0_prior.infer)
            {
                mu0 = model.dobs.par1;
            }
            if (!rho_prior.infer)
            {
                rho = model.dobs.par2;
            }
            if (!par1_prior.infer)
            {
                par1 = model.transfer.dlag.par1;
            }
            if (!par2_prior.infer)
            {
                par2 = model.transfer.dlag.par2;
            }
            if (!W_prior.infer)
            {
                W = model.derr.par1;
            }

            // LBA::LinearBayes linear_bayes(model, y);
            // linear_bayes.filter();
            // linear_bayes.smoother();
            // arma::mat psi_tmp = LBA::get_psi(linear_bayes.atilde, linear_bayes.Rtilde);
            // arma::vec wt_init = arma::diff(psi_tmp.col(1));

            // wt.tail(wt_init.n_elem) = wt_init;

            ApproxDisturbance approx_dlm(model.dim.nT, model.transfer.fgain.name);

            for (unsigned int b = 0; b < ntotal; b++)
            {
                Rcpp::checkUserInterrupt();

                approx_dlm.set_Fphi(model.transfer.dlag, model.dim.nL);
                Posterior::update_wt(wt, wt_accept, approx_dlm, y, model, w0_prior, mh_sd);
                arma::vec psi = arma::cumsum(wt);
                arma::vec hpsi = GainFunc::psi2hpsi<arma::vec>(psi, model.transfer.fgain.name);

                if (W_prior.infer)
                {
                    double W_old = W;
                    W = Posterior::update_W(W_accept, W_old, wt, W_prior, mh_sd);
                    w0_prior.update_par2(W);
                    model.derr.update_par1(W);
                }

                if (par1_prior.infer || par2_prior.infer)
                {
                    arma::vec out = Posterior::update_dlag(
                        par1_accept, par2_accept, model,
                        y, hpsi, par1_prior, par2_prior,
                        par1_mh_sd, par2_mh_sd, max_lag);
                    // arma::vec out = Posterior::update_dlag_hmc(
                    //     par1_accept, par2_accept, model, y, hpsi,
                    //     par1_prior, par2_prior, epsilon, L, m, max_lag);

                    par1 = out.at(0);
                    par2 = out.at(1);
                }

                if (mu0_prior.infer)
                {
                    mu0 = Posterior::update_mu0(mu0_accept, model, y, hpsi, mu0_mh_sd);
                }

                if (rho_prior.infer)
                {
                    double rho_old = rho;
                    rho = Posterior::update_dispersion(rho_accept, model, y, hpsi, rho_prior, rho_mh_sd);
                }

                bool saveiter = b > nburnin && ((b - nburnin - 1) % nthin == 0);
                if (saveiter || b == (ntotal - 1))
                {
                    unsigned int idx_run;
                    if (saveiter)
                    {
                        idx_run = (b - nburnin - 1) / nthin;
                    }
                    else
                    {
                        idx_run = nsample - 1;
                    }

                    wt_stored.col(idx_run) = wt;
                    W_stored.at(idx_run) = W;
                    mu0_stored.at(idx_run) = mu0;
                    rho_stored.at(idx_run) = rho;
                    par1_stored.at(idx_run) = par1;
                    par2_stored.at(idx_run) = par2;
                }

                if (verbose)
                {
                    Rcpp::Rcout << "\rProgress: " << b << "/" << ntotal - 1;
                }

            } // end a single iteration

            if (verbose)
            {
                Rcpp::Rcout << std::endl;
            }

            return;
        }

    private:
        double epsilon = 0.01;
        unsigned int L = 10;
        Rcpp::NumericVector m;

        double mh_sd = 0.1;
        unsigned int nburnin = 100;
        unsigned int nthin = 1;
        unsigned int nsample = 100;
        unsigned int ntotal = 200;
        unsigned int nforecast = 0;
        unsigned int nforecast_err = 10; // forecasting for indices (1, ..., ntime-1) has `nforecast` elements
        double tstart_pct = 0.9;

        unsigned int max_lag = 30;

        Rcpp::List model_info;

        Dim dim;
        arma::vec y;

        Dist w0_prior;
        arma::vec wt;
        arma::vec wt_accept; // nsample x 1
        arma::mat wt_stored; // (nT + 1) x nsample

        Dist mu0_prior;
        double mu0 = 0.;
        arma::vec mu0_stored;
        double mu0_accept = 0.;
        double mu0_mh_sd = 1.;

        Dist rho_prior;
        double rho = 0.;
        arma::vec rho_stored;
        double rho_accept = 0.;
        double rho_mh_sd = 1.;

        Dist par1_prior;
        double par1 = 0.;
        arma::vec par1_stored;
        double par1_accept = 0.;
        double par1_mh_sd = 1.;

        Dist par2_prior;
        double par2 = 0.;
        arma::vec par2_stored;
        double par2_accept = 0.;
        double par2_mh_sd = 1.;

        Dist W_prior;
        double W = 0.01;
        arma::vec W_stored;
        double W_accept = 0.;
    };
}

#endif