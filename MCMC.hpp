#ifndef _MCMC_HPP
#define _MCMC_HPP

#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <algorithm>
#include <RcppArmadillo.h>
#include <pg.h>
#include "Model.hpp"
#include "LinkFunc.hpp"
#include "LinearBayes.hpp"
#include "StaticParams.hpp"

// [[Rcpp::plugins(cpp17)]]
// [[Rcpp::depends(RcppArmadillo,pg)]]

namespace MCMC
{
    class Leapfrog
    {
    public:
        /**
         * @brief Calculate gradient of the potential energery. The potential energy is the negative log joint probability of the model.
         * 
         * @note Duane et al. (1987): https://www.sciencedirect.com/science/article/abs/pii/037026938791197X
         * @note Neal (2011): https://arxiv.org/pdf/1206.1901
         * @note Betancourt (2018): https://arxiv.org/abs/1701.02434
         * @note How people introduce their use of HMC - Patel et al. (2021): https://arxiv.org/pdf/2110.08363
         * @note Practical guide to HMC: https://bjlkeng.io/posts/hamiltonian-monte-carlo/
         * 
         * @param q 
         * @param model 
         * @param param_selected 
         * @param W_prior 
         * @param seas_prior 
         * @param rho_prior 
         * @param par1_prior 
         * @param par2_prior 
         * @param zintercept_infer 
         * @param zzcoef_infer 
         * @return arma::vec 
         */
        static arma::vec grad_U(
            const Model &model,
            const arma::vec &params, 
            const arma::vec &y,
            const arma::vec &hpsi,
            const arma::mat &Theta,
            const std::vector<std::string> &param_selected,
            const Prior &W_prior, 
            const Prior &seas_prior, 
            const Prior &rho_prior, 
            const Prior &par1_prior, 
            const Prior &par2_prior, 
            const bool &zintercept_infer, 
            const bool &zzcoef_infer
        )
        {
            arma::vec lambda(y.n_elem, arma::fill::zeros);
            for (unsigned int t = 1; t < y.n_elem; t++)
            {
                double eta = TransFunc::transfer_sliding(t, model.dlag.nL, y, model.dlag.Fphi, hpsi);
                if (model.seas.period > 0)
                {
                    eta += arma::dot(model.seas.X.col(t), model.seas.val);
                }
                lambda.at(t) = LinkFunc::ft2mu(eta, model.flink);
            }

            arma::vec dloglik_dlag = Model::dloglik_dlag(
                y, hpsi, model.dlag.nL, model.dlag.name, model.dlag.par1, model.dlag.par2,
                model.dobs, model.seas, model.zero, model.flink);

            // dloglik_dlag.t().print("\n Leapfrog dloglik_dlag:");

            arma::vec gd_U = Static::dlogJoint_deta(
                y, Theta, lambda, dloglik_dlag, params, param_selected,
                W_prior, par1_prior, par2_prior, rho_prior, seas_prior, model);
            
            // negate it because U = -log(p) and what we calculate above is the gradient of log(p)
            gd_U.for_each([](arma::vec::elem_type &val)
                          { val *= -1.; });

            return gd_U;
        }
    }; // class Leapfrog

    class Posterior
    {
    public:
        /**
         * @brief Gibbs sampler for zero-inflated indicator z[t]
         * 
         * @param model 
         * @param y 
         * @param wt 
         */
        static void update_zt(
            Model &model, 
            const arma::vec &y, // (nT + 1) x 1
            const arma::vec &wt
        ) // (nT + 1) x 1
        {
            // lambda: (nT + 1) x 1, conditional expectation of y[t] if z[t] = 1
            arma::vec lambda = model.wt2lambda(y, wt, model.seas.period, model.seas.X, model.seas.val);

            arma::vec p01(y.n_elem, arma::fill::zeros); // p(z[t] = 1 | z[t-1] = 0, gamma)
            p01.at(0) = logistic(model.zero.intercept);
            arma::vec p11(y.n_elem, arma::fill::zeros); // p(z[t] = 1 | z[t-1] = 1, gamma)
            p11.at(0) = logistic(model.zero.intercept + model.zero.coef);

            arma::vec prob_filter(y.n_elem, arma::fill::zeros); // p(z[t] = 1 | y[1:t], gamma)
            prob_filter.at(0) = p01.at(0);
            for (unsigned int t = 1; t < y.n_elem; t++)
            {
                double p0 = model.zero.intercept;
                double p1 = model.zero.intercept + model.zero.coef;
                if (!model.zero.X.is_empty())
                {
                    double val = arma::dot(model.zero.X.col(t), model.zero.beta);
                    p0 += val;
                    p1 += val;
                }
                p01.at(t) = logistic(p0); // p(z[t] = 1 | z[t-1] = 0, gamma)
                p11.at(t) = logistic(p1); // p(z[t] = 1 | z[t-1] = 1, gamma)

                if (y.at(t) > EPS)
                {
                    // If y[t] > 0
                    // We must have p(z[t] = 1 | y[t] > 0) = 1
                    prob_filter.at(t) = 1.;
                }
                else
                {
                    double prob_yzero = ObsDist::loglike(
                        0., model.dobs.name, lambda.at(t), model.dobs.par2, false);

                    double pp1 = prob_filter.at(t - 1) * p11.at(t); // p(z[t-1] = 1) * p(z[t] = 1 | z[t-1] = 1)
                    double pp0 = prob_filter.at(t - 1) * std::abs(1. - p11.at(t)); // p(z[t-1] = 1) * p(z[t] = 0 | z[t-1] = 1)

                    pp1 += std::abs(1. - prob_filter.at(t - 1)) * p01.at(t); // p(z[t-1] = 0) * p(z[t] = 1 | z[t-1] = 0)
                    pp0 += std::abs(1. - prob_filter.at(t - 1)) * std::abs(1. - p01.at(t)); // p(z[t-1] = 0) * p(z[t] = 0 | z[t-1] = 0)

                    pp1 *= prob_yzero; // p(y[t] = 0 | z[t] = 1)
                    prob_filter.at(t) = pp1 / (pp1 + pp0 + EPS);
                }
            } // Forward filtering loop

            model.zero.z.at(y.n_elem - 1) = (R::runif(0., 1.) < prob_filter.at(y.n_elem - 1)) ? 1. : 0.;
            for (unsigned int t = y.n_elem - 2; t > 0; t--)
            {
                if (y.at(t) > EPS)
                {
                    model.zero.z.at(t) = 1.;
                }
                else
                {
                    double p1 = model.zero.z.at(t + 1) > EPS ? p11.at(t + 1) : (1. - p11.at(t + 1)); // p(z[t+1] | z[t] = 1)
                    double prob_backward1 = prob_filter.at(t) * std::abs(p1); // p(z[t] = 1 | y[t] = 0) * p(z[t+1] | z[t] = 1)
                    double p0 = model.zero.z.at(t + 1) > EPS ? p01.at(t + 1) : (1. - p01.at(t + 1)); // p(z[t+1] | z[t] = 0)
                    double prob_backward0 = std::abs(1. - prob_filter.at(t)) * std::abs(p0); // p(z[t] = 0 | y[t] = 0) * p(z[t+1] | z[t] = 0)

                    prob_backward1 = prob_backward1 / (prob_backward1 + prob_backward0 + EPS);
                    model.zero.z.at(t) = (R::runif(0., 1.) < prob_backward1) ? 1. : 0.;
                }
            }

        } // update_zt

        static void update_wt( // Checked. OK.
            arma::vec &wt,     // (nT + 1) x 1
            arma::vec &wt_accept,
            ApproxDisturbance &approx_dlm,
            const arma::vec &y, // (nT + 1) x 1
            Model &model,
            const double &mh_sd = 0.1
        )
        {
            arma::vec ft(y.n_elem, arma::fill::zeros);
            double prior_sd = std::sqrt(model.derr.par1);

            for (unsigned int t = 1; t < y.n_elem; t++)
            {
                double wt_old = wt.at(t);
                arma::vec lam = model.wt2lambda(y, wt, model.seas.period, model.seas.X, model.seas.val);

                double logp_old = 0.;
                for (unsigned int i = t; i < y.n_elem; i++)
                {
                    if (!(model.zero.inflated && (model.zero.z.at(i) < EPS)))
                    {
                        // For zero-inflated model, only the y[t] that is not missing taken into account
                        logp_old += ObsDist::loglike(y.at(i), model.dobs.name, lam.at(i), model.dobs.par2, true);
                    }
                } // Checked. OK.

                logp_old += R::dnorm4(wt_old, 0., prior_sd, true);

                /*
                Metropolis-Hastings
                */
                approx_dlm.update_by_wt(y, wt);
                arma::vec eta = approx_dlm.get_eta_approx(model.seas); // nT x 1, f0, Fn and psi is updated
                arma::vec lambda = LinkFunc::ft2mu<arma::vec>(eta, model.flink); // nT x 1
                arma::vec Vt_hat = ApproxDisturbance::func_Vt_approx(
                    lambda, model.dobs, model.flink); // nT x 1

                arma::mat Fn = approx_dlm.get_Fn(); // nT x nT
                arma::vec Fnt = Fn.col(t - 1); // nT x 1
                arma::vec Fnt2 = Fnt % Fnt;

                arma::vec tmp = Fnt2 / Vt_hat; // nT x 1, element-wise division
                if (model.zero.inflated)
                {
                    tmp %= model.zero.z.subvec(1, tmp.n_elem); // If y[t] is missing (z[t] = 0), F[t]*F[t]'/V[t] is removed from the posterior variance of the proposal
                }
                double mh_prec = arma::accu(tmp) + EPS8;
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
                lam = model.wt2lambda(y, wt, model.seas.period, model.seas.X, model.seas.val); // Checked. OK.

                double logp_new = 0.;
                for (unsigned int i = t; i < y.n_elem; i++)
                {
                    if (!(model.zero.inflated && (model.zero.z.at(i) == 0)))
                    {
                        logp_new += ObsDist::loglike(y.at(i), model.dobs.name, lam.at(i), model.dobs.par2, true);
                    }
                } // Checked. OK.

                logp_new += R::dnorm4(wt_new, 0., prior_sd, true); // prior

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

        static void update_W( // Checked. OK.
            double &W_accept,
            Model &model,
            const arma::vec &wt,
            const Dist &W_prior,
            const double &mh_sd = 1.)
        {
            double W_old = model.derr.par1;
            double res = arma::accu(arma::square(wt.tail(wt.n_elem - 2)));

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
                    model.derr.par1 = W_new;
                    W_accept += 1.;
                }
                break;
            }
            case AVAIL::Dist::invgamma:
            {
                double nSw_new = W_prior.par2 + res;                  // prior_params.at(1) = nSw
                model.derr.par1 = 1. / R::rgamma(0.5 * W_prior.par1, 2. / nSw_new); // prior_params.at(0) = nw_new
                // W_accept += 1.;
                break;
            }
            default:
            {
                break;
            }
            }

            #ifdef DGTF_DO_BOUND_CHECK
            bound_check(model.derr.par1, "update: W", true, true);
            #endif
            return;
        } // func update_W

        static void update_seas( // flat prior
            double &seas_accept,
            Model &model,
            const arma::vec &y, // (ntime + 1) x 1
            const arma::vec &hpsi,
            const Prior &seas_prior,
            const double &min_var = 0.1)
        {
            std::map<std::string, AVAIL::Dist> obs_list = ObsDist::obs_list;
            arma::vec seas_old = model.seas.val; // period x 1

            arma::vec Vt_hat(y.n_elem, arma::fill::zeros); // (ntime + 1) x 1
            arma::vec lambda(y.n_elem, arma::fill::zeros); // (ntime + 1) x 1
            arma::vec eta(y.n_elem, arma::fill::zeros);    // (ntime + 1) x 1
            arma::vec ft(y.n_elem, arma::fill::zeros); // (ntime + 1) x 1

            double logp_old = 0.;
            for (unsigned int t = 1; t < y.n_elem; t++)
            {
                ft.at(t) = TransFunc::func_ft(t, y, ft, hpsi, model.dlag, model.ftrans);
                // eta.at(t) = TransFunc::transfer_sliding(t, model.dlag.nL, y, model.dlag.Fphi, hpsi);
                // double ft = eta.at(t);
                eta.at(t) = ft.at(t);
                if (model.seas.period > 0)
                {
                    eta.at(t) += arma::dot(model.seas.X.col(t), seas_old);
                }

                lambda.at(t) = LinkFunc::ft2mu(eta.at(t), model.flink);
                if (y.at(t) < EPS)
                {
                    lambda.at(t) = (lambda.at(t) < EPS) ? EPS : lambda.at(t);
                }

                // if (y.at(t) > 0)
                // {
                    logp_old += ObsDist::loglike(y.at(t), model.dobs.name, lambda.at(t), model.dobs.par2, true);
                // } 
            }

            Vt_hat = ApproxDisturbance::func_Vt_approx(lambda, model.dobs, model.flink); // (ntime + 1) x 1

            arma::vec tmp = 1. / Vt_hat;
            arma::mat seas_prec(model.seas.period, model.seas.period, arma::fill::zeros);
            for (unsigned int t = 1; t < y.n_elem; t++)
            {
                seas_prec = seas_prec + model.seas.X.col(t) * model.seas.X.col(t).t() * tmp.at(t);
            }
            seas_prec.diag() += 1. / seas_prior.par2;

            arma::mat seas_prec_chol = arma::chol(arma::symmatu(seas_prec));
            arma::mat seas_chol = arma::inv(arma::trimatu(seas_prec_chol));
            arma::vec seas_new = seas_old + seas_prior.mh_sd * seas_chol * arma::randn(model.seas.period);


            for (unsigned int t = 1; t < y.n_elem; t++)
            {
                eta.at(t) = ft.at(t);
                if (model.seas.period > 0)
                {
                    eta.at(t) += arma::dot(model.seas.X.col(t), seas_new);
                }
                lambda.at(t) = LinkFunc::ft2mu(eta.at(t), model.flink);
                if (y.at(t) < EPS)
                {
                    lambda.at(t) = (lambda.at(t) < EPS) ? EPS : lambda.at(t);
                }
            }
            bool lambda_in_range = lambda.elem(arma::find(y > 0)).min() > -EPS;

            if (lambda_in_range || obs_list[model.dobs.name] == AVAIL::Dist::gaussian) // non-negative
            {
                double logp_new = 0.;
                for (unsigned int t = 1; t < y.n_elem; t++)
                {
                    // if (y.at(t) > 0)
                    // {
                        logp_new += ObsDist::loglike(y.at(t), model.dobs.name, lambda.at(t), model.dobs.par2, true);
                    // }
                }

                double logratio = std::min(0., logp_new - logp_old);

                if (std::log(R::runif(0., 1.)) < logratio)
                { // accept
                    model.seas.val = seas_new;
                    seas_accept += 1.;
                }
            }

            return;
        } // func update_mu0

        static void update_dispersion( // Checked. OK.
            double &rho_accept,
            Model &model,
            const arma::vec &y, // nobs x 1
            const arma::vec &hpsi,
            const Prior &rho_prior)
        {
            double rho_old = model.dobs.par2;
            double logp_old = R::dgamma(rho_old, rho_prior.par1, 1. / rho_prior.par2, true);
            arma::vec lambda(y.n_elem, arma::fill::zeros);
            arma::vec ft(y.n_elem, arma::fill::zeros);
            for (unsigned int t = 1; t < y.n_elem; t++)
            {
                // double eta = TransFunc::transfer_sliding(t, model.dlag.nL, y, model.dlag.Fphi, hpsi);
                ft.at(t) = TransFunc::func_ft(t, y, ft, hpsi, model.dlag, model.ftrans);
                double eta = ft.at(t);
                if (model.seas.period > 0)
                {
                    eta += arma::dot(model.seas.X.col(t), model.seas.val);
                }
                lambda.at(t) = LinkFunc::ft2mu(eta, model.flink);
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
                rho_new = R::rnorm(rho_old, rho_prior.mh_sd * rho_old); // mh_sd here is the coefficient of variation, i.e., sd/mean.
                success = (rho_new > 0) ? true : false;
                cnt++;
            }

            double logp_new = R::dgamma(rho_new, rho_prior.par1, 1. / rho_prior.par2, true);
            for (unsigned int t = 1; t < y.n_elem; t++)
            {
                logp_new += ObsDist::loglike(y.at(t), model.dobs.name, lambda.at(t), rho_new, true);
            }
            double logratio = std::min(0., logp_new - logp_old);

            if (std::log(R::runif(0., 1.)) < logratio)
            { // accept
                #ifdef DGTF_DO_BOUND_CHECK
                bound_check(rho_new, "Posterior::update_mu0");
                #endif
                
                model.dobs.par2 = rho_new;
                rho_accept += 1.;
                // logp_mu0 = logp_new;
            }

            
            return;
        } // func update_dispersion

        static void update_dlag(
            double &par1_accept,
            double &par2_accept,
            Model &model,
            const arma::vec &y, // nobs x 1
            const arma::vec &hpsi,
            const Prior &par1_prior,
            const Prior &par2_prior,
            const double &par1_mh_sd = 0.1,
            const double &par2_mh_sd = 0.1,
            const unsigned int &max_lag = 50)
        {
            std::map<std::string, AVAIL::Dist> dist_list = AVAIL::dist_list;
            std::map<std::string, TransFunc::Transfer> trans_list = TransFunc::trans_list;
            LagDist dlag_old = model.dlag;

            double loglik_old = 0.;
            arma::vec ft(y.n_elem, arma::fill::zeros);

            for (unsigned int t = 1; t < y.n_elem; t++)
            {
                // double eta = TransFunc::transfer_sliding(t, model.dlag.nL, y, model.dlag.Fphi, hpsi);
                ft.at(t) = TransFunc::func_ft(t, y, ft, hpsi, dlag_old, model.ftrans);
                double eta = ft.at(t);
                if (model.seas.period > 0)
                {
                    eta += arma::dot(model.seas.X.col(t), model.seas.val);
                }
                double lambda = LinkFunc::ft2mu(eta, model.flink);
                loglik_old += ObsDist::loglike(y.at(t), model.dobs.name, lambda, model.dobs.par2, true);
            }
            double logp_old = loglik_old;

            double logp_new = 0.;
            double par1_new = dlag_old.par1;
            double logprior_par1_old = 0.;
            double logprior_par1_new = 0.;
            if (par1_prior.infer)
            {
                logprior_par1_old = Prior::dprior(dlag_old.par1, par1_prior, true, false);
                logp_old += logprior_par1_old;

                bool non_gaussian = !dist_list[par1_prior.name] == AVAIL::Dist::gaussian;
                if (non_gaussian) // non-negative
                {
                    bool success = false;
                    unsigned int cnt = 0;
                    while (!success && cnt < MAX_ITER)
                    {
                        par1_new = R::rnorm(dlag_old.par1, par1_mh_sd * dlag_old.par1);
                        success = (par1_new > 0) ? true : false;
                        cnt++;
                    }
                }
                else // gaussian
                {
                    par1_new = R::rnorm(dlag_old.par1, par1_mh_sd * dlag_old.par1);
                }

                logprior_par1_new = Prior::dprior(par1_new, par1_prior, true, false);
                logp_new += logprior_par1_new;
            } // par1

            double par2_new = dlag_old.par2;
            double logprior_par2_old = 0.;
            double logprior_par2_new = 0.;
            if (par2_prior.infer)
            {
                logprior_par2_old = Prior::dprior(dlag_old.par2, par2_prior, true, false);
                logp_old += logprior_par2_old;

                bool non_gaussian = !dist_list[par2_prior.name] == AVAIL::Dist::gaussian;
                if (non_gaussian) // non-negative
                {
                    bool success = false;
                    unsigned int cnt = 0;
                    while (!success && cnt < MAX_ITER)
                    {
                        par2_new = R::rnorm(dlag_old.par2, par2_mh_sd * dlag_old.par2);
                        success = (par2_new > 0) ? true : false;
                        cnt++;
                    }
                }
                else // gaussian
                {
                    par2_new = R::rnorm(dlag_old.par2, par2_mh_sd * dlag_old.par2);
                }

                logprior_par2_new = Prior::dprior(par2_new, par2_prior, true, false);
                logp_new += logprior_par2_new;

            } // par2

            LagDist dlag_new(dlag_old.name, par1_new, par2_new, dlag_old.truncated);
            if (trans_list[dlag_new.name] == TransFunc::Transfer::iterative)
            {
                dlag_new.truncated = true;
                dlag_new.nL = y.n_elem - 1;
                dlag_new.Fphi = LagDist::get_Fphi(dlag_new);
            }

            double loglik_new = 0.;
            for (unsigned int t = 1; t < y.n_elem; t++)
            {
                // double eta = TransFunc::transfer_sliding(t, nlag, y, Fphi_new, hpsi);
                ft.at(t) = TransFunc::func_ft(t, y, ft, hpsi, dlag_new, model.ftrans);
                double eta = ft.at(t);
                if (model.seas.period > 0)
                {
                    eta += arma::dot(model.seas.X.col(t), model.seas.val);
                }
                double lambda = LinkFunc::ft2mu(eta, model.flink);
                loglik_new += ObsDist::loglike(y.at(t), model.dobs.name, lambda, model.dobs.par2, true);
            }
            logp_new += loglik_new;

            double logratio = std::min(0., logp_new - logp_old);

            if (std::log(R::runif(0., 1.)) < logratio)
            { // accept
                par1_accept += 1;
                par2_accept += 1;

                model.dlag = dlag_new;
                model.nP = Model::get_nP(model.dlag, model.seas.period, model.seas.in_state);
            }

            return;

        } // update_lag

        static Model update_static_hmc(
            double &accept,
            const Model &model,
            const arma::vec &y,
            const arma::vec &psi,
            const std::vector<std::string> &param_selected,
            const Prior &W_prior,
            const Prior &seas_prior,
            const Prior &rho_prior,
            const Prior &par1_prior,
            const Prior &par2_prior,
            const arma::vec &epsilon,
            const bool zintercept_infer = false,
            const bool zzcoef_infer = false,
            const unsigned int &L = 10,
            const double &kinetic_sd = 1.
        )
        {
            std::map<std::string, AVAIL::Dist> obs_list = ObsDist::obs_list;
            arma::mat Theta(model.nP, y.n_elem, arma::fill::zeros);
            Theta.row(0) = psi.t();
            arma::vec hpsi = GainFunc::psi2hpsi<arma::vec>(psi, model.fgain);
            Model mod = model;

            // Calculate log of joint probability
            arma::vec lambda(y.n_elem, arma::fill::zeros);
            for (unsigned int t = 1; t < y.n_elem; t++)
            {
                double eta = TransFunc::transfer_sliding(t, mod.dlag.nL, y, mod.dlag.Fphi, hpsi);
                if (mod.seas.period > 0)
                {
                    eta += arma::dot(mod.seas.X.col(t), mod.seas.val);
                }
                lambda.at(t) = LinkFunc::ft2mu(eta, mod.flink);
            }

            double logp_old = Static::logJoint(
                y, Theta, lambda, W_prior, par1_prior, par2_prior, rho_prior, seas_prior, mod); // Checked. OK.
            
            // Potential energy: negative log joint probability.
            double current_U = -logp_old;

            // map parameters of the lag distribution to the whole real line
            arma::vec params = Static::init_eta(param_selected, mod, true);
            arma::vec q = Static::eta2tilde(
                params, param_selected, W_prior.name, par1_prior.name, 
                mod.dobs.name, mod.seas.period, mod.seas.in_state); // Checked. OK.

            // sample an initial momentum
            arma::vec p = arma::randn(q.n_elem);
            p.for_each([&kinetic_sd](arma::vec::elem_type &val)
                       { val *= kinetic_sd; });

            // Kinetic: negative logprob of the momentum distribution
            double current_K = 0.5 * arma::accu(arma::square(p / kinetic_sd));

            // half step update of momentum
            arma::vec grad_U = Leapfrog::grad_U(
                mod, params, y, hpsi, Theta, param_selected,
                W_prior, seas_prior, rho_prior, par1_prior, par2_prior,
                zintercept_infer, zzcoef_infer); // Checked. OK.

            p -= 0.5 * (epsilon % grad_U);

            for (unsigned int i = 1; i <= L; i++)
            {
                // full step update for position
                q += epsilon % p;

                params = Static::tilde2eta(
                    q, param_selected, W_prior.name, par1_prior.name,
                    mod.dlag.name, mod.dobs.name,
                    mod.seas.period, mod.seas.in_state); // Checked. OK.
                Static::update_params(mod, param_selected, params);

                // Gradient of the potential energy given updated "q"
                grad_U = Leapfrog::grad_U(
                    mod, params, y, hpsi, Theta, param_selected, 
                    W_prior, seas_prior, rho_prior, par1_prior, par2_prior, 
                    zintercept_infer, zzcoef_infer);

                // full step update for momentum
                if (i != L)
                {
                    p -= epsilon % grad_U;
                }
            }

            // Leapfrog: final half step update for momentum
            p -= 0.5 * (epsilon % grad_U);

            // Leapfrog: negate trajectory to make proposal symmetric
            p *= -1.;

            // Calculate log of joint probability with new proposal
            for (unsigned int t = 1; t < y.n_elem; t++)
            {
                double eta = TransFunc::transfer_sliding(t, mod.dlag.nL, y, mod.dlag.Fphi, hpsi);
                if (mod.seas.period > 0)
                {
                    eta += arma::dot(mod.seas.X.col(t), mod.seas.val);
                }
                lambda.at(t) = LinkFunc::ft2mu(eta, mod.flink);
            }

            double logp_new = Static::logJoint(
                y, Theta, lambda, W_prior, par1_prior, par2_prior, rho_prior, seas_prior, mod);
            
            // Potential energy: negative log joint probability
            double proposed_U = -logp_new;

            // Kinetic energy
            double proposed_K = 0.5 * arma::accu(arma::square(p / kinetic_sd));

            // accept / reject
            double logratio = current_U + current_K;
            logratio -= (proposed_U + proposed_K);

            if (std::log(R::runif(0., 1.)) < logratio)
            { // accept
                accept += 1;
                return mod;
                // logp_mu0 = logp_new;
            }
            else
            {
                return model;
            }
        }


        static arma::vec update_pg_psi(
            const Model &model, 
            const arma::vec &y, 
            const arma::vec &psi_old // nP x (nT + 1)
        )
        {
            const unsigned int nP = model.nP;
            const unsigned int nT = y.n_elem - 1;
            const double npop = model.dobs.par2;

            Model normal_dlm = model;
            normal_dlm.dobs.name = "gaussian";
            normal_dlm.flink = "identity";
            normal_dlm.fgain = "identity";

            arma::vec omega(nT + 1, arma::fill::ones);
            arma::mat Theta_old = TransFunc::psi2theta(psi_old, y, model.ftrans, model.fgain, model.dlag);
            for (unsigned int t = 0; t <= nT; t++)
            {
                if (std::abs(model.zero.z.at(t) - 1.) < EPS)
                {
                    // If y[t] is not missing
                    double eta = TransFunc::func_ft(model.ftrans, model.fgain, model.dlag, model.seas, t, Theta_old.col(t), y);
                    omega.at(t) = pg::rpg_scalar_hybrid(y.at(t) + npop, eta);
                }
            }

            arma::vec ktmp = 0.5 * (y - npop) / omega;
            // arma::vec ktmp = y;

            // Forward filtering
            arma::mat mt(nP, nT + 1, arma::fill::zeros);
            arma::cube Ct(nP, nP, nT + 1);
            Ct.slice(0) = 5. * arma::eye<arma::mat>(nP, nP);
            arma::mat at = mt;
            arma::cube Rt = Ct;

            arma::vec Ft = TransFunc::init_Ft(nP, model.ftrans, model.seas.period, model.seas.in_state);
            arma::mat Gt = SysEq::init_Gt(nP, model.dlag, model.fsys, model.seas.period, model.seas.in_state);
            for (unsigned int t = 1; t < nT + 1; t++)
            {
                at.col(t) = SysEq::func_gt(
                    model.fsys, model.fgain, model.dlag,
                    mt.col(t - 1), y.at(t - 1),
                    model.seas.period, model.seas.in_state);
                Rt.slice(t) = LBA::func_Rt(Gt, Ct.slice(t - 1), normal_dlm.derr.par1);

                if (std::abs(model.zero.z.at(t) - 1.) < EPS)
                {
                    // When y[t] is not missing
                    normal_dlm.dobs.par2 = 1. / omega.at(t);
                    double ft_prior = 0.;
                    double qt_prior = 0.;
                    LBA::func_prior_ft(ft_prior, qt_prior, Ft, t, normal_dlm, y, at.col(t), Rt.slice(t));
                    qt_prior += normal_dlm.dobs.par2;

                    double ft_posterior = ktmp.at(t);
                    double qt_posterior = 0.;

                    arma::mat At = LBA::func_At(Rt.slice(t), Ft, qt_prior);
                    mt.col(t) = LBA::func_mt(at.col(t), At, ft_prior, ft_posterior);
                    Ct.slice(t) = LBA::func_Ct(Rt.slice(t), At, qt_prior, qt_posterior);
                }
                else
                {
                    // When y[t] is missing
                    mt.col(t) = at.col(t);
                    Ct.slice(t) = Rt.slice(t);
                }
            }

            arma::mat Theta(nP, nT + 1, arma::fill::zeros);
            // Backward sampling
            {
                arma::mat Ct_chol = arma::chol(arma::symmatu(Ct.slice(nT)));
                Theta.col(nT) = mt.col(nT) + Ct_chol.t() * arma::randn(nP);
            }

            for (unsigned int t = nT - 1; t > 0; t--)
            {
                arma::mat Rt_inv = inverse(Rt.slice(t + 1));
                arma::mat Bt = Ct.slice(t) * Gt.t() * Rt_inv;

                arma::vec ht = mt.col(t) + Bt * (Theta.col(t + 1) - at.col(t + 1));
                arma::mat Ht = Ct.slice(t) - Bt * Gt * Ct.slice(t);
                Ht.diag() += EPS8;

                arma::mat Ht_chol = arma::chol(arma::symmatu(Ht));
                Theta.col(t) = ht + Ht_chol.t() * arma::randn(nP);
            }

            arma::vec psi_new = arma::vectorise(Theta.row(0));
            return psi_new;
        }
    }; // class Posterior

    class Disturbance
    {
    public:
        Disturbance(const Model &model, const Rcpp::List &mcmc_settings)
        {
            Rcpp::List opts = mcmc_settings;

            L = 10;
            if (opts.containsElementNamed("L"))
            {
                L = Rcpp::as<unsigned int>(opts["L"]);
            }

            kinetic_sd = 1.;
            if (opts.containsElementNamed("hmc_kinetic_sd"))
            {
                kinetic_sd = Rcpp::as<double>(opts["hmc_kinetic_sd"]);
            }

            max_lag = 50;
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

            Static::init_prior(
                param_selected, nparam,
                W_prior, seas_prior, rho_prior,
                par1_prior, par2_prior,
                zintercept_infer, zzcoef_infer,
                opts, model);

            if (nparam > 0)
            {
                update_static = true;
                hmc_accept = 0.;
                epsilon.set_size(nparam);
                epsilon.fill(0.01);
                if (opts.containsElementNamed("epsilon"))
                {
                    arma::vec eps = Rcpp::as<arma::vec>(opts["epsilon"]);
                    if (epsilon.n_elem <= eps.n_elem)
                    {
                        epsilon = eps.subvec(0, epsilon.n_elem - 1);
                    }
                    else
                    {
                        epsilon.subvec(0, eps.n_elem - 1) = eps;
                        epsilon.subvec(eps.n_elem, epsilon.n_elem - 1).fill(epsilon.at(eps.n_elem - 1));
                    }
                }
            }
            else
            {
                update_static = false;
            }

            W_stored.set_size(nsample);
            // W_accept = 0.;

            seas_stored.set_size(model.seas.period, nsample);
            // seas_accept = 0.;

            rho_stored.set_size(nsample);
            // rho_accept = 0.;

            par1_stored.set_size(nsample);
            par2_stored.set_size(nsample);
            // lag_accept = 0.;

            zintercept_stored.set_size(nsample);
            zzcoef_stored.set_size(nsample);

            return;
        }

        static Rcpp::List default_settings()
        {
            Rcpp::List opts = Static::default_settings();

            opts["epsilon"] = 0.01;
            opts["L"] = 10;
            opts["hmc_kinetic_sd"] = 1.;
            opts["max_lag"] = 50;

            opts["mh_sd"] = 0.1;
            opts["nburnin"] = 100;
            opts["nthin"] = 1;
            opts["nsample"] = 100;
            return opts;
        }

        Rcpp::List get_output()
        {
            arma::vec qprob = {0.025, 0.5, 0.975};
            Rcpp::List output;

            arma::mat psi_quantile = arma::quantile(psi_stored, qprob, 1); // (nT + 1) x 3
            output["psi_stored"] = Rcpp::wrap(psi_stored);
            output["psi"] = Rcpp::wrap(psi_quantile);
            output["wt_accept"] = Rcpp::wrap(wt_accept / ntotal);

            if (!z_stored.is_empty())
            {
                output["z_stored"] = Rcpp::wrap(arma::vectorise(arma::mean(z_stored, 1)));
            }
            
            // arma::mat psi_quantile = arma::quantile(wt_stored, qprob, 1); // (nT + 1) x 3
            // output["psi"] = Rcpp::wrap(psi_quantile);
            // output["wt_accept"] = Rcpp::wrap(W_accept / ntotal);
            // output["log_marg"] = Rcpp::wrap(log_marg_stored.t());

            if (update_static)
            {
                output["infer_W"] = W_prior.infer;
                output["W"] = Rcpp::wrap(W_stored);
                // output["W_accept"] = W_accept / ntotal;

                output["infer_seas"] = seas_prior.infer;
                output["seas"] = Rcpp::wrap(seas_stored);
                // output["seas_accept"] = static_cast<double>(seas_accept / ntotal);

                output["infer_rho"] = rho_prior.infer;
                output["rho"] = Rcpp::wrap(rho_stored);
                // output["rho_accept"] = rho_accept / ntotal;

                output["infer_par1"] = par1_prior.infer;
                output["par1"] = Rcpp::wrap(par1_stored);

                output["infer_par2"] = par2_prior.infer;
                output["par2"] = Rcpp::wrap(par2_stored);
                // output["lag_accept"] = lag_accept / ntotal;

                output["infer_zintercept"] = zintercept_infer;
                if (zintercept_infer)
                {
                    output["zintercept"] = Rcpp::wrap(zintercept_stored);
                }

                output["infer_zzcoef"] = zzcoef_infer;
                if (zzcoef_infer)
                {
                    output["zzcoef"] = Rcpp::wrap(zzcoef_stored);
                }

                output["hmc_accept"] = hmc_accept / ntotal;
            }

            return output;
        }


        void infer(Model &model, const arma::vec &y, const bool &verbose = VERBOSE)
        {
            const unsigned int nT = y.n_elem - 1;
            model.seas.X = Season::setX(nT, model.seas.period, model.seas.P);
            arma::vec ztmp(nT + 1, arma::fill::ones);
            if (model.zero.inflated)
            {
                z_stored.set_size(nT + 1, nsample);
                z_stored.ones();
                ztmp.zeros();
                for (unsigned int t = 1; t < y.n_elem; t++)
                {
                    if (y.at(t) > EPS)
                    {
                        ztmp.at(t) = 1.;
                    }
                }
            }
            model.zero.setZ(ztmp, nT);

            std::map<std::string, AVAIL::Dist> lag_list = LagDist::lag_list;
            if (!model.dlag.truncated && lag_list[model.dlag.name] == AVAIL::Dist::nbinomp)
            {
                // iterative transfer function
                model.dlag.nL = nT;
                model.dlag.Fphi = LagDist::get_Fphi(model.dlag);
                model.dlag.truncated = true;
            }

            std::map<std::string, AVAIL::Dist> obs_list = ObsDist::obs_list;
            double y_scale = 1.;
            if (obs_list[model.dobs.name] == AVAIL::Dist::nbinomp)
            {
                y_scale = model.dobs.par2;
            }

            wt = arma::randn(nT + 1) * 0.01;
            wt.at(0) = 0.;
            wt.subvec(1, model.nP) = arma::abs(wt.subvec(1, model.nP));
            psi = arma::cumsum(wt);
            wt_accept.set_size(nT + 1);
            wt_accept.zeros();

            #ifdef DGTF_DO_BOUND_CHECK
            bound_check(wt, "Disturbance::init");
            #endif

            psi_stored.set_size(nT + 1, nsample);
            psi_stored.zeros();

            std::map<std::string, AVAIL::Dist> prior_dist = AVAIL::dist_list;
            if ((prior_dist[W_prior.name] == AVAIL::Dist::invgamma) && W_prior.infer)
            {
                double nw = W_prior.par1;
                double nSw = W_prior.par1 * W_prior.par2;
                double nw_new = nw + (double)wt.n_elem - 2.;
                W_prior.par1 = nw_new;
                W_prior.par2 = nSw;
            }

            ApproxDisturbance approx_dlm(nT, model.fgain);
            for (unsigned int b = 0; b < ntotal; b++)
            {
                Rcpp::checkUserInterrupt();

                if (obs_list[model.dobs.name] == AVAIL::Dist::nbinomp)
                {
                    arma::vec psi_old = psi;
                    psi = Posterior::update_pg_psi(model, y, psi_old);
                    wt.subvec(1, nT) = arma::diff(psi);
                }
                else
                {
                    approx_dlm.set_Fphi(model.dlag, model.dlag.nL);
                    Posterior::update_wt(wt, wt_accept, approx_dlm, y, model, mh_sd);
                    psi = arma::cumsum(wt);
                }

                if (model.zero.inflated)
                {
                    Posterior::update_zt(model, y, wt);
                }

                // if (W_prior.infer)
                // {
                //     // arma::vec wt = arma::diff(psi);
                //     Posterior::update_W(W_accept, model, wt, W_prior, mh_sd);
                // }

                if (update_static)
                {
                    // Posterior::update_dlag(
                    //     par1_accept, par2_accept, model,
                    //     y, hpsi, par1_prior, par2_prior,
                    //     par1_mh_sd, par2_mh_sd, max_lag);
                    Model mod = model;
                    model = Posterior::update_static_hmc(
                        hmc_accept, mod, y, psi, param_selected,
                        W_prior, seas_prior, rho_prior, par1_prior, par2_prior,
                        epsilon, zintercept_infer, zzcoef_infer, L, kinetic_sd);
                }

                // if (seas_prior.infer)
                // {
                //     Posterior::update_seas(seas_accept, model, y, hpsi, seas_prior);
                // }

                // if (rho_prior.infer)
                // {
                //     Posterior::update_dispersion(rho_accept, model, y, hpsi, rho_prior);
                // }

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

                    // log_marg_stored.at(idx_run) = log_marg;
                    psi_stored.col(idx_run) = psi;
                    if (model.zero.inflated)
                    {
                        z_stored.col(idx_run) = model.zero.z;
                    }

                    if (update_static)
                    {
                        W_stored.at(idx_run) = model.derr.par1;
                        seas_stored.col(idx_run) = model.seas.val;
                        rho_stored.at(idx_run) = model.dobs.par2;
                        par1_stored.at(idx_run) = model.dlag.par1;
                        par2_stored.at(idx_run) = model.dlag.par2;
                        zintercept_stored.at(idx_run) = model.zero.intercept;
                        zzcoef_stored.at(idx_run) = model.zero.coef;
                    }
                }

                // if (verbose)
                // {
                //     Rcpp::Rcout << "\rProgress: " << b << "/" << ntotal - 1;
                // }

            } // end a single iteration

            // if (verbose)
            // {
            //     Rcpp::Rcout << std::endl;
            // }

            return;
        }

    private:
        arma::vec log_marg_stored;

        arma::vec epsilon;
        unsigned int L = 10;
        double kinetic_sd = 1.;

        double mh_sd = 0.1;
        unsigned int nburnin = 100;
        unsigned int nthin = 1;
        unsigned int nsample = 100;
        unsigned int ntotal = 200;
        unsigned int max_lag = 50;

        bool update_static = true;
        double hmc_accept = 0.;
        unsigned int nparam = 1; // number of unknown static parameters
        std::vector<std::string> param_selected = {"W"};

        arma::mat z_stored; // (nT + 1) x nsample
        bool zintercept_infer = false;
        bool zzcoef_infer = false;
        arma::vec zintercept_stored;
        arma::vec zzcoef_stored;

        arma::vec wt, psi;
        arma::vec wt_accept; // nsample x 1
        arma::mat psi_stored; // (nT + 1) x nsample

        Prior seas_prior;
        arma::mat seas_stored; // period x nsample
        // double seas_accept = 0.;

        Prior rho_prior;
        arma::vec rho_stored;
        // double rho_accept = 0.;

        Prior par1_prior;
        arma::vec par1_stored;
        // double lag_accept = 0.;

        Prior par2_prior;
        arma::vec par2_stored;

        Prior W_prior;
        arma::vec W_stored;
        // double W_accept = 0.;
    };
}

#endif