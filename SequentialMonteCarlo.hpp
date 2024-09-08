#ifndef _SEQUENTIALMONTECARLO_H
#define _SEQUENTIALMONTECARLO_H

#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <algorithm>
// #include <chrono>
#include <RcppArmadillo.h>
#ifdef _OPENMP
    #include <omp.h>
#endif
#include "Model.hpp"
#include "ImportanceDensity.hpp"

// [[Rcpp::depends(RcppArmadillo)]]

/**
 * @brief Sequential Monte Carlo methods.
 * @todo Bugs to be fixed
 *       1. Discount factor is not working.
 * @todo Further improvement on SMC
 *       1. Resample-move step after resamplin: we need to move the particles carefully because we have constrains on the augmented states Theta.
 *       2. Residual resampling, systematic resampling, or stratified resampling.
 *       3. Tampering SMC.
 *       4. Another option for iterative transfer function: change theta to the sliding style in propagation but use the theta in iterative style as the true probability density.
 *
 *       Reference:
 *       1. Notes on sequential Monte Carlo (by N. Kantas);
 *       2. Particle filters and data assimilation (by Fearnhead and Kunsch).
 *       3. Online tutorial for particle MCMC - https://sbfnk.github.io/mfiidd/pmcmc_solution.html#calibrate-the-number-of-particles
 */
namespace SMC
{
    class SequentialMonteCarlo
    {
    public:
        SequentialMonteCarlo(const Model &model, const Rcpp::List &smc_settings)
        {
            Rcpp::List settings = smc_settings;
            N = 1000;
            if (settings.containsElementNamed("num_particle"))
            {
                N = Rcpp::as<unsigned int>(settings["num_particle"]);
            }
            weights.set_size(N);
            weights.ones();
            tau = weights;
            lambda = weights;

            M = N;
            if (settings.containsElementNamed("num_smooth"))
            {
                M = Rcpp::as<unsigned int>(settings["num_smooth"]);
            }

            B = 1;

            nforecast = 0;
            if (settings.containsElementNamed("num_step_ahead_forecast"))
            {
                nforecast = Rcpp::as<unsigned int>(settings["num_step_ahead_forecast"]);
            }


            use_discount = false;
            if (settings.containsElementNamed("use_discount"))
            {
                use_discount = Rcpp::as<bool>(settings["use_discount"]);
            }

            discount_factor = 0.95;
            if (settings.containsElementNamed("discount_factor"))
            {
                discount_factor = Rcpp::as<double>(settings["discount_factor"]);
            }

            smoothing = false;
            if (settings.containsElementNamed("do_smoothing"))
            {
                smoothing = Rcpp::as<bool>(settings["do_smoothing"]);
            }

            return;
        }


        static Rcpp::List default_settings()
        {
            Rcpp::List opts;
            opts["num_particle"] = 1000;
            opts["num_smooth"] = 1000;
            opts["num_step_ahead_forecast"] = 0;
            opts["use_discount"] = false;
            opts["discount_factor"] = 0.95;
            opts["do_smoothing"] = false;

            return opts;
        }


        static arma::vec draw_param_init(
            const Dist &init_dist,
            const unsigned int &N,
            const unsigned int &max_iter = 100)
        {
            std::map<std::string, AVAIL::Dist> dist_list = AVAIL::dist_list;
            arma::vec par_init(N, arma::fill::zeros);

            for (unsigned int i = 0; i < N; i++)
            {
                double val = 0.;
                switch (dist_list[init_dist.name])
                {
                case AVAIL::Dist::invgamma:
                {
                    bool success = false;
                    unsigned int cnt = 0;
                    while (!success && cnt < max_iter)
                    {
                        val = 1. / R::rgamma(init_dist.par1, 1. / init_dist.par2);
                        success = std::isfinite(val) && (val > EPS);
                        cnt++;
                    }

                    break;
                }
                case AVAIL::Dist::gamma:
                {
                    bool success = false;
                    unsigned int cnt = 0;
                    while (!success && cnt < max_iter)
                    {
                        val = R::rgamma(init_dist.par1, 1. / init_dist.par2);
                        success = std::isfinite(val) && (val > EPS);
                        cnt++;
                    }

                    break;
                }
                case AVAIL::Dist::uniform:
                {
                    val = R::runif(init_dist.par1, init_dist.par2);
                    break;
                }
                case AVAIL::Dist::constant:
                {
                    val = init_dist.par1;
                    break;
                }
                default:
                {
                    throw std::invalid_argument("SMC::PL::init_W: unknown prior for W.");
                }
                } // switch by initial distribution

                par_init.at(i) = val;

                #ifdef DGTF_DO_BOUND_CHECK
                bound_check<arma::vec>(par_init, "draw_param_init:: par_init");
                #endif
            }

            return par_init;
        }


        static double discount_W(
            const arma::mat &Theta_now, // p x N
            const double &discount_factor = 0.95)
        {
            double W;
            arma::rowvec psi = Theta_now.row(0);
            double var_psi = arma::var(psi);

            if (var_psi > EPS)
            {
                W = var_psi;
            }
            else
            {
                W = 1.;
            }
            // Wsqrt = std::sqrt(Wt.at(t));
            W *= 1. / discount_factor - 1.;

            #ifdef DGTF_DO_BOUND_CHECK
            bound_check(W, "SequentialMonteCarlo::discount_W", true, true);
            #endif
            return W;
        }

        static double effective_sample_size(const arma::vec &weights)
        {
            arma::vec w2 = arma::square(weights);
            double denom = arma::accu(w2);
            double nom = arma::accu(weights);
            nom = std::pow(nom, 2.);
            double ess = nom / denom;

            #ifdef DGTF_DO_BOUND_CHECK
            bound_check(ess, "effective_sample_size: ess (nom = " + std::to_string(nom) + ", denom = " + std::to_string(denom) + ")");
            #endif

            return ess;
        }



        /**
         * @brief At time t, importance weights of forward filtering particles z[t-1] = (theta[t-1], W[t-1], mu0[t-1]), based on conditional predictive distribution y[t] | z[t-1], y[1:t-1].
         *
         * @param model
         * @param t_new Index of the time that is being predicted. The following inputs come from time t_old = t-1.
         * @param Theta p x N, { theta[t-1, i], i = 1, ..., N }, samples of latent states at t-1
         * @param W N x 1, { W[t-1, i], i = 1, ..., N }, samples of latent variance at t-1
         * @param mu0 N x 1, { mu0[t-1, i], i = 1, ..., N }, samples of baseline at t-1
         * @param mt_old p x N, m[t-1]. Assume no static parameters involved in ft.
         * @param yhat y[t] after transformation.
         * @return arma::vec
         */
        static arma::vec imp_weights_backcast(
            arma::mat &loc, // p x N, mean of the posterior of theta[t_cur] | y[t_cur:nT], theta[t_next], W
            arma::cube &Sigma_chol, // p x p x N, right cholesky of the variance of the posterior of theta[t_cur] | y[t_cur:nT], theta[t_next], W
            arma::vec &logq,        // N x 1
            Model &model,
            const unsigned int &t_cur,   // current time "t". The following inputs come from time t+1. t_next = t + 1; t_prev = t - 1
            const arma::mat &Theta_next, // p x N, {theta[t+1]}
            const arma::mat &Theta_cur, // p x N
            const arma::vec &W,   // N x 1, {inv(W[T])} samples of latent variance
            const arma::mat &param_filter, // (period + 3) x N, {seasonal components, rho, par1, par2} samples of baseline
            const arma::cube &vt,        // nP x (nT + 1) x N, v[t]
            const arma::cube &Vt,        // nP*nP x (nT + 1) x N, V[t]
            const arma::vec &y,
            const bool &full_rank = false,
            const bool &infer_seas = false,
            const bool &infer_obs = false,
            const bool &infer_lag = false
        )
        {
            double yhat_cur = LinkFunc::mu2ft(y.at(t_cur), model.flink);
            unsigned int N = Theta_next.n_cols;

            loc.set_size(model.nP, N);
            loc.zeros();
            Sigma_chol = arma::zeros<arma::cube>(model.nP, model.nP, N);

            for (unsigned int i = 0; i < N; i++)
            {
                if (infer_seas)
                {
                    model.seas.val = param_filter.submat(0, i, model.seas.period - 1, i);
                }
                if (infer_obs)
                {
                    model.dobs.par2 = std::exp(param_filter.at(model.seas.period, i));
                }
                if (infer_lag)
                {
                    model.dlag.par1 = param_filter.at(model.seas.period + 1, i);
                    model.dlag.par2 = std::exp(param_filter.at(model.seas.period + 2, i));
                }

                arma::vec v_cur = vt.slice(i).col(t_cur);
                arma::vec Vtmp = Vt.slice(i).col(t_cur);
                arma::mat V_cur = arma::reshape(Vtmp, model.nP, model.nP);
                arma::mat Vprec_cur = inverse(V_cur);
                arma::vec v_next = vt.slice(i).col(t_cur + 1);

                arma::vec r_cur(model.nP, arma::fill::zeros);
                arma::mat K_cur(model.nP, model.nP, arma::fill::zeros); // evolution matrix
                arma::mat Uprec_cur = K_cur;
                arma::mat Urchol_cur = K_cur;
                double ldetU = 0.;
                backward_kernel(
                    K_cur, r_cur, Uprec_cur, ldetU, model, t_cur,
                    v_cur, v_next, Vprec_cur, Theta_cur.col(i), y);
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
                    arma::mat Prec = arma::symmatu(F_cur * F_cur.t() / Vtilde + Uprec_cur);
                    Prec.diag() += EPS;

                    arma::mat prec_chol = arma::chol(arma::symmatu(Prec));
                    arma::mat prec_chol_inv = arma::inv(arma::trimatu(prec_chol));
                    double ldetPrec = arma::accu(arma::log(prec_chol.diag())) * 2.;
                    Sigma_chol.slice(i) = prec_chol_inv;

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
            } // loop over particles

            double logq_max = logq.max();
            logq.for_each([&logq_max](arma::vec::elem_type &val)
                          { val -= logq_max; });
            arma::vec weights = arma::exp(logq);

            #ifdef DGTF_DO_BOUND_CHECK
            bound_check<arma::vec>(weights, "imp_weights_backcast");
            #endif

            return weights;
        } // func: imp_weights_backcast


        static arma::uvec get_resample_index(const arma::vec &weights)
        {
            unsigned int N = weights.n_elem;
            double wsum = arma::accu(weights);
            arma::uvec indices = arma::regspace<arma::uvec>(0, 1, N - 1);
            if (wsum > EPS)
            {
                arma::vec w = weights / wsum;
                indices = sample(N, N, w, true, true);
            }

            return indices;
        }

        static arma::uvec get_smooth_index(
            const arma::rowvec &psi_smooth_now,  // 1 x M, Theta_smooth.slice(t).row(0)
            const arma::rowvec &psi_filter_prev, // 1 x N, Theta.slice(t - 1).row(0)
            const arma::vec &Wsqrt)              // M x 1
        {
            unsigned int M = psi_smooth_now.n_elem;
            unsigned int N = psi_filter_prev.n_elem;

            arma::uvec smooth_idx = arma::regspace<arma::uvec>(0, 1, M - 1);
            for (unsigned int i = 0; i < M; i++) // loop over M smoothed particles at time t.
            {
                // arma::vec diff = (psi_now.at(i) - psi_old) / Wsqrt.at(i); // N x 1
                // weights = - 0.5 * arma::pow(diff, 2.);
                arma::vec weights(N, arma::fill::zeros);
                for (unsigned int j = 0; j < N; j++)
                {
                    weights.at(j) = R::dnorm(psi_filter_prev.at(j), psi_smooth_now.at(i), Wsqrt.at(i), true);
                }

                double wmax = weights.max();
                weights.for_each([&wmax](arma::vec::elem_type &val)
                                 { val -= wmax; });
                weights = arma::exp(weights);

                double wsum = arma::accu(weights);
                if (wsum < EPS)
                {
                    weights.ones();
                    wsum = static_cast<double>(N);
                }
                weights /= wsum;

                smooth_idx.at(i) = sample(N, weights, true); // draw one sample only
            }

            return smooth_idx;
        }

        static arma::uvec get_smooth_index(
            const arma::mat &theta_now,  // p x M, Theta_smooth.slice(t).row(0)
            const arma::mat &theta_prev, // p x N, Theta.slice(t - 1).row(0)
            const arma::mat &Wt_now) // p x p
        {
            unsigned int M = theta_now.n_cols;
            unsigned int N = theta_prev.n_cols;

            arma::uvec smooth_idx = arma::regspace<arma::uvec>(0, 1, M - 1);
            for (unsigned int i = 0; i < M; i++) // loop over M smoothed particles at time t.
            {
                // arma::vec diff = (psi_now.at(i) - psi_old) / Wsqrt.at(i); // N x 1
                // weights = - 0.5 * arma::pow(diff, 2.);
                arma::vec weights(N, arma::fill::zeros);
                for (unsigned int j = 0; j < N; j++)
                {
                    weights.at(j) = MVNorm::dmvnorm(theta_now.col(i), theta_prev.col(j), Wt_now, true);
                }

                double wmax = weights.max();
                weights.for_each([&wmax](arma::vec::elem_type &val)
                                 { val -= wmax; });
                weights = arma::exp(weights);

                double wsum = arma::accu(weights);
                if (wsum < EPS)
                {
                    weights.ones();
                    wsum = static_cast<double>(N);
                }
                weights /= wsum;

                smooth_idx.at(i) = sample(N, weights, true); // draw one sample only
            }

            return smooth_idx;
        }

        Rcpp::List forecast_error(
            const Model &model,
            const arma::vec &y,
            const std::string &loss_func = "quadratic",
            const unsigned int &k = 1,
            const Rcpp::Nullable<unsigned int> &start_time = R_NilValue,
            const Rcpp::Nullable<unsigned int> &end_time = R_NilValue)
        {
            arma::cube th_filter = Theta.tail_slices(y.n_elem); // p x N x (nT + 1)
            Rcpp::List out2 = StateSpace::forecast_error(th_filter, y, model, loss_func, k, VERBOSE, start_time, end_time);

            return out2;
        }

        void forecast_error(
            double &err,
            double &cov,
            double &width,
            const Model &model,
            const arma::vec &y, 
            const std::string &loss_func = "quadratic")
        {
            arma::cube theta_tmp = Theta.tail_slices(y.n_elem); // p x N x (nT + 1)
            StateSpace::forecast_error(err, cov, width, theta_tmp, y, model, loss_func);
            return;
        }

        Rcpp::List fitted_error(const Model &model, const arma::vec &y, const std::string &loss_func = "quadratic")
        {
            Rcpp::List out3;

            arma::cube theta_tmp = Theta.tail_slices(y.n_elem); // p x N x (nT + 1)
            Rcpp::List out_filter = StateSpace::fitted_error(theta_tmp, y, model, loss_func);
            out3["filter"] = out_filter;

            if (smoothing)
            {
                arma::cube theta_tmp2 = Theta_smooth.tail_slices(y.n_elem); // p x N x (nT + 1)
                Rcpp::List out_smooth = StateSpace::fitted_error(theta_tmp2, y, model, loss_func);
                out3["smooth"] = out_smooth;
            }

            return out3;
        }

        void fitted_error(double &err, const Model &model, const arma::vec &y, const std::string &loss_func = "quadratic")
        {

            arma::cube theta_tmp;
            if (smoothing)
            {
                theta_tmp = Theta_smooth.tail_slices(y.n_elem); // p x N x (nT + 1)
            }
            else
            {
                theta_tmp = Theta.tail_slices(y.n_elem); // p x N x (nT + 1)
            }

            StateSpace::fitted_error(err, theta_tmp, y, model, loss_func);
            return;
        }


        static double auxiliary_filter0(
            arma::cube &Theta, // p x N x (nT + 1)
            Model &model,
            const arma::vec &y, // (nT + 1) x 1
            const unsigned int &N = 1000,
            const bool &initial_resample_all = false,
            const bool &final_resample_by_weights = false)
        {
            const unsigned int nT = y.n_elem - 1;
            const double logN = std::log(static_cast<double>(N));
            const double Wsqrt = std::sqrt(std::abs(model.derr.par1) + EPS);
            arma::vec weights(N, arma::fill::ones);
            double log_cond_marginal = 0.;

            for (unsigned int t = 0; t < nT; t++)
            {
                arma::vec logq(N, arma::fill::zeros);
                arma::vec tau = qforecast0(logq, model, t + 1, Theta.slice(t), y);

                tau = weights % tau;
                if (t > 0)
                {
                    arma::uvec resample_idx = get_resample_index(tau);
                    if (initial_resample_all)
                    {
                        for (unsigned int k = 0; k <= t; k++)
                        {
                            Theta.slice(k) = Theta.slice(k).cols(resample_idx);
                        }
                    }
                    else
                    {
                        Theta.slice(t) = Theta.slice(t).cols(resample_idx);
                    }

                    logq = logq.elem(resample_idx);
                    weights = weights.elem(resample_idx);
                }


                // Propagate
                arma::mat Theta_new(model.nP, N, arma::fill::zeros);
                arma::mat Theta_cur = Theta.slice(t); // nP x N
                #ifdef _OPENMP
                    #pragma omp parallel for num_threads(NUM_THREADS) schedule(runtime)
                #endif
                for (unsigned int i = 0; i < N; i++)
                {
                    double eps = R::rnorm(0., Wsqrt);
                    logq.at(i) += R::dnorm4(eps, 0., Wsqrt, true); // sample from evolution distribution

                    arma::vec theta_new = StateSpace::func_gt(model.ftrans, model.fgain, model.dlag, Theta_cur.col(i), y.at(t), model.seas.period, model.seas.in_state);
                    theta_new.at(0) += eps;
                    Theta_new.col(i) = theta_new;

                    double logp = R::dnorm4(theta_new.at(0), Theta_cur.at(0, i), Wsqrt, true);
                    double ft = StateSpace::func_ft(model.ftrans, model.fgain, model.dlag, model.seas, t + 1, theta_new, y);
                    double lambda = LinkFunc::ft2mu(ft, model.flink);
                    logp += ObsDist::loglike(y.at(t + 1), model.dobs.name, lambda, model.dobs.par2, true);
                    weights.at(i) = logp - logq.at(i);
                }

                Theta.slice(t + 1) = Theta_new;

                double wmax = weights.max();
                weights.for_each([&wmax](arma::vec::elem_type &val)
                                 { val -= wmax; });
                weights = arma::exp(weights);
                if (final_resample_by_weights || t >= nT - 1)
                {
                    double eff = effective_sample_size(weights);
                    if (eff < 0.95 * N)
                    {
                        arma::uvec resample_idx = get_resample_index(weights);
                        Theta.slice(t + 1) = Theta.slice(t + 1).cols(resample_idx);
                        weights.ones();
                    }
                }


                log_cond_marginal += std::log(arma::accu(weights) + EPS) - logN;
            }

            return log_cond_marginal;
        }


        static double auxiliary_filter(
            arma::cube &Theta, // p x N x (nT + 1)
            arma::mat &weights_forward, // (nT + 1) x N
            arma::vec &eff_forward, // (nT + 1) x 1
            arma::cube &Wt, // p x p x (nT + 1), only needs to be initialized if using discount factor.
            Model &model,
            const arma::vec &y, // (nT + 1) x 1
            const unsigned int &N = 1000,
            const bool &initial_resample_all = false,
            const bool &final_resample_by_weights = false,
            const bool &use_discount = false,
            const double &discount_factor = 0.95,
            const bool &verbose = false)
        {
            const unsigned int nT = y.n_elem - 1;
            const double logN = std::log(static_cast<double>(N));

            weights_forward.set_size(y.n_elem, N);
            weights_forward.zeros();
            eff_forward.set_size(y.n_elem);
            eff_forward.zeros();

            arma::vec weights(N, arma::fill::ones);
            double log_cond_marginal = 0.;

            arma::mat Wt_chol(model.nP, model.nP, arma::fill::zeros);
            if (!use_discount)
            {
                if (model.derr.full_rank)
                {
                    Wt_chol = arma::chol(model.derr.var);
                }
                else
                {
                    Wt_chol.at(0, 0) = std::sqrt(model.derr.par1);
                }
            }


            for (unsigned int t = 0; t < nT; t++)
            {
                
                arma::vec logq(N, arma::fill::zeros);
                arma::vec tau = logq;
                arma::mat loc;            // (model.nP, N, arma::fill::zeros);
                arma::cube prec_chol_inv; // nP x nP x N

                if (use_discount)
                {
                    // Update Wt
                    if (model.derr.full_rank)
                    {
                        Wt_chol = arma::chol(Wt.slice(t + 1));
                        model.derr.var = Wt.slice(t + 1);
                    }
                    else
                    {
                        Wt_chol.at(0, 0) = std::sqrt(Wt.at(0, 0, t + 1));
                        model.derr.par1 = Wt.at(0, 0, t + 1);
                        model.derr.var.at(0, 0) = Wt.at(0, 0, t + 1);
                    }
                }


                if (model.derr.full_rank)
                {
                    loc = arma::zeros<arma::mat>(model.nP, N);
                    prec_chol_inv = arma::zeros<arma::cube>(model.nP, model.nP, N); // nP x nP x N
                    arma::mat param;
                    arma::vec W;
                    tau = qforecast(loc, prec_chol_inv, logq, model, t + 1, Theta.slice(t), W, param, y);
                }
                else
                {
                    tau = qforecast0(logq, model, t + 1, Theta.slice(t), y);
                }

                tau = weights % tau;
                weights_forward.row(t) = logq.t();
                

                if (t > 0)
                {
                    arma::uvec resample_idx = get_resample_index(tau);
                    if (initial_resample_all)
                    {
                        for (unsigned int k = 0; k <= t; k++)
                        {
                            Theta.slice(k) = Theta.slice(k).cols(resample_idx);
                            arma::vec wtmp = arma::vectorise(weights_forward.row(k));
                            weights_forward.row(k) = wtmp.elem(resample_idx).t();
                        }
                    }
                    else
                    {
                        Theta.slice(t) = Theta.slice(t).cols(resample_idx);
                        arma::vec wtmp = arma::vectorise(weights_forward.row(t));
                        weights_forward.row(t) = wtmp.elem(resample_idx).t();
                    }

                    if (model.derr.full_rank)
                    {
                        loc = loc.cols(resample_idx);
                        prec_chol_inv = prec_chol_inv.slices(resample_idx);
                    }

                    logq = logq.elem(resample_idx);
                    weights = weights.elem(resample_idx);
                }


                // Propagate
                arma::mat Theta_new(model.nP, N, arma::fill::zeros);
                arma::mat Theta_cur = Theta.slice(t); // nP x N
                #ifdef _OPENMP
                    #pragma omp parallel for num_threads(NUM_THREADS) schedule(runtime)
                #endif
                for (unsigned int i = 0; i < N; i++)
                {
                    arma::vec gtheta = StateSpace::func_gt(model.ftrans, model.fgain, model.dlag, Theta_cur.col(i), y.at(t), model.seas.period, model.seas.in_state);
                    arma::vec eps(model.nP, arma::fill::zeros);
                    arma::vec theta_new;
                    double logp = 0.;
                    if (!use_discount && model.derr.full_rank)
                    {
                        arma::vec eps = arma::randn(Theta_new.n_rows);
                        arma::vec zt = prec_chol_inv.slice(i).t() * loc.col(i) + eps; // shifted
                        theta_new = prec_chol_inv.slice(i) * zt;                      // scaled

                        logq.at(i) += MVNorm::dmvnorm0(zt, loc.col(i), prec_chol_inv.slice(i), true);
                        logp += MVNorm::dmvnorm(theta_new, gtheta, model.derr.var, true);
                    }
                    else
                    {
                        // not full rank or full rank with discount
                        eps = Wt_chol.t() * arma::randn(Theta_new.n_rows);
                        theta_new = gtheta + eps;

                        if (!use_discount)
                        {
                            // not full rank and not use discount
                            logq.at(i) += R::dnorm4(eps.at(0), 0., Wt_chol.at(0, 0), true); // sample from 
                            logp += R::dnorm4(theta_new.at(0), gtheta.at(0), Wt_chol.at(0, 0), true);
                        }
                    }

                    Theta_new.col(i) = theta_new;

                    double ft = StateSpace::func_ft(model.ftrans, model.fgain, model.dlag, model.seas, t + 1, theta_new, y);
                    double lambda = LinkFunc::ft2mu(ft, model.flink);
                    logp += ObsDist::loglike(y.at(t + 1), model.dobs.name, lambda, model.dobs.par2, true);
                    weights.at(i) = logp - logq.at(i);
                }

                double wmax = weights.max();
                weights.for_each([&wmax](arma::vec::elem_type &val)
                                 { val -= wmax; });
                weights = arma::exp(weights);

                Theta.slice(t + 1) = Theta_new;

                if (final_resample_by_weights)
                {
                    eff_forward.at(t + 1) = effective_sample_size(weights);
                    if (eff_forward.at(t + 1) < 0.95 * N || t >= nT - 1)
                    {
                        arma::uvec resample_idx = get_resample_index(weights);
                        Theta.slice(t + 1) = Theta.slice(t + 1).cols(resample_idx);
                        weights.ones();
                    }
                }


                log_cond_marginal += std::log(arma::accu(weights) + EPS) - logN;

                if (verbose)
                {
                    Rcpp::Rcout << "\rForwawrd Filtering: " << t + 1 << "/" << nT;
                }
            }

            if (verbose)
            {
                Rcpp::Rcout << std::endl;
            }

            return log_cond_marginal;
        }

        arma::vec weights, lambda, tau; // N x 1
        arma::cube Theta;        // p x N x (nT + B)
        arma::cube Theta_smooth; // p x M x (nT + B)

        unsigned int N = 1000;
        unsigned int M = 500;
        unsigned int B = 1;
        unsigned int nforecast = 0;

        bool use_discount = false;
        double discount_factor = 0.95;
        bool smoothing = true;

        Rcpp::List output;
        arma::vec ci_prob = {0.025, 0.5, 0.975};

    }; // class Sequential Monte Carlo


    class MCS : public SequentialMonteCarlo
    {
    public:
        MCS(
            const Model &model,
            const Rcpp::List &opts) : SequentialMonteCarlo(model, opts)
        {
            M = N;
            Rcpp::List settings = opts;
            B = 1;
            if (settings.containsElementNamed("num_backward"))
            {
                B = Rcpp::as<unsigned int>(settings["num_backward"]);
            }
        }

        Rcpp::List get_output(const bool &summarize = true)
        {
            return output;
        }


        static Rcpp::List default_settings()
        {
            Rcpp::List opts = SequentialMonteCarlo::default_settings();
            opts["num_backward"] = 10;
            return opts;
        }

        Rcpp::List forecast(const Model &model, const arma::vec &y)
        {
            arma::vec Wtmp(N, arma::fill::zeros);
            Wtmp.fill(model.derr.par1);
            Rcpp::List out = StateSpace::forecast(y, Theta, Wtmp, model, nforecast);
            return out;
        }

        arma::mat optimal_discount_factor(
            Model &model,
            const arma::vec &y, 
            const double &from,
            const double &to,
            const double &delta = 0.01,
            const std::string &loss = "quadratic",
            const bool &verbose = VERBOSE)
        {
            const unsigned int nT = y.n_elem - 1;
            arma::vec grid = arma::regspace<arma::vec>(from, delta, to);
            unsigned int nelem = grid.n_elem;
            arma::mat stats(nelem, 4, arma::fill::zeros);

            use_discount = true;

            for (unsigned int i = 0; i < nelem; i++)
            {
                Rcpp::checkUserInterrupt();

                discount_factor = grid.at(i);
                stats.at(i, 0) = discount_factor;

                infer(model, y);
                arma::cube theta_tmp = Theta.tail_slices(nT + 1);

                double err_forecast = 0.;
                double cov_forecast = 0.;
                double width_forecast = 0.;
                StateSpace::forecast_error(err_forecast, cov_forecast, width_forecast, theta_tmp, y, model, loss, false);

                stats.at(i, 1) = err_forecast;
                stats.at(i, 2) = cov_forecast;
                stats.at(i, 3) = width_forecast;

                if (verbose)
                {
                    Rcpp::Rcout << "\rProgress: " << i + 1 << "/" << nelem;
                }
            }

            if (verbose)
            {
                Rcpp::Rcout << std::endl;
            }

            return stats;
        }

        arma::mat optimal_W(
            Model &model,
            const arma::vec &y,
            const arma::vec &grid,
            const std::string &loss = "quadratic",
            const bool &verbose = VERBOSE)
        {
            unsigned int nelem = grid.n_elem;
            arma::mat stats(nelem, 4, arma::fill::zeros);
\
            use_discount = false;

            for (unsigned int i = 0; i < nelem; i++)
            {
                Rcpp::checkUserInterrupt();

                model.derr.par1 = grid.at(i);
                stats.at(i, 0) = model.derr.par1;

                infer(model, y);
                arma::cube theta_tmp = Theta.tail_slices(y.n_elem);

                double err_forecast = 0.;
                double cov_forecast = 0.;
                double width_forecast = 0.;
                StateSpace::forecast_error(err_forecast, cov_forecast, width_forecast, theta_tmp, y, model, loss, false);
                stats.at(i, 1) = err_forecast;

                stats.at(i, 2) = cov_forecast;
                stats.at(i, 3) = width_forecast;

                if (verbose)
                {
                    Rcpp::Rcout << "\rProgress: " << i + 1 << "/" << nelem;
                }
            }

            if (verbose)
            {
                Rcpp::Rcout << std::endl;
            }

            return stats;
        }

        arma::mat optimal_num_backward(
            Model &model,
            const arma::vec &y,
            const unsigned int &from,
            const unsigned int &to,
            const unsigned int &delta = 1,
            const std::string &loss = "quadratic",
            const bool &verbose = VERBOSE)
        {
            const unsigned int nT = y.n_elem - 1;
            arma::uvec grid = arma::regspace<arma::uvec>(from, delta, to);
            unsigned int nelem = grid.n_elem;
            arma::mat stats(nelem, 4, arma::fill::zeros);

            for (unsigned int i = 0; i < nelem; i++)
            {
                Rcpp::checkUserInterrupt();

                B = grid.at(i);
                stats.at(i, 0) = B;

                Theta.clear();
                Theta.set_size(model.nP, N, nT + B);
                Theta.zeros();

                Theta_smooth.clear();
                Theta_smooth = Theta;

                infer(model, y);
                arma::cube theta_tmp = Theta.tail_slices(nT + 1);

                double err_forecast = 0.;
                double cov_forecast = 0.;
                double width_forecast = 0.;
                StateSpace::forecast_error(err_forecast, cov_forecast, width_forecast, theta_tmp, y, model, loss, false);
                stats.at(i, 1) = err_forecast;

                stats.at(i, 2) = cov_forecast;
                stats.at(i, 3) = width_forecast;

                if (verbose)
                {
                    Rcpp::Rcout << "\rProgress: " << i + 1 << "/" << nelem;
                }
            }

            if (verbose)
            {
                Rcpp::Rcout << std::endl;
            }

            return stats;
        }


        void infer(Model &model, const arma::vec &y, const bool &verbose = VERBOSE)
        {
            const double logN = std::log(static_cast<double>(N));
            const unsigned int nT = y.n_elem - 1;
            arma::mat Wt_chol(model.nP, model.nP, arma::fill::zeros);
            arma::cube Wt;
            if (use_discount)
            {
                LBA::LinearBayes lba(use_discount, discount_factor);
                lba.filter(model, y);
                arma::mat Gt = TransFunc::init_Gt(model.nP, model.dlag, model.ftrans, model.seas.period, model.seas.in_state);

                Wt = arma::zeros<arma::cube>(model.nP, model.nP, y.n_elem);

                if (model.derr.full_rank)
                {
                    Wt.slice(0) = arma::symmatu(lba.Ct.slice(0));
                }
                else
                {
                    Wt.at(0, 0, 0) = std::abs(lba.Ct.at(0, 0, 0)) + EPS;
                }

                for (unsigned int t = 1; t < y.n_elem; t++)
                {
                    LBA::func_Gt(Gt, model, lba.mt.col(t - 1), y.at(t - 1));
                    arma::mat Pt = Gt * lba.Ct.slice(t - 1) * Gt.t();
                    arma::mat Wt_hat = (1. / discount_factor - 1.) * Pt;
                    Wt_hat.diag() += EPS8;

                    if (model.derr.full_rank)
                    {
                        Wt.slice(t) = arma::symmatu(Wt_hat);
                    }
                    else
                    {
                        Wt.at(0, 0, t) = Wt_hat.at(0, 0);
                    }   
                }
            }
            else
            {
                if (model.derr.full_rank)
                {
                    Wt_chol = arma::chol(model.derr.var);
                }
                else
                {
                    Wt_chol.at(0, 0) = std::sqrt(model.derr.par1);
                }
            }


            Theta = arma::randn<arma::cube>(model.nP, N, nT + B);
            Theta_smooth = Theta;

            arma::mat psi_forward = Theta.row_as_mat(0);
            psi_forward.zeros();
            arma::mat psi_smooth = psi_forward;
            arma::vec log_cond_marginal(y.n_elem, arma::fill::zeros);

            for (unsigned int t = 0; t < nT; t++)
            {
                Rcpp::checkUserInterrupt();
                if (use_discount)
                {
                    // Update Wt
                    if (model.derr.full_rank)
                    {
                        Wt_chol = arma::chol(Wt.slice(t + 1));
                        model.derr.var = Wt.slice(t + 1);
                    }
                    else
                    {
                        Wt_chol.at(0, 0) = std::sqrt(Wt.at(0, 0, t + 1));
                        model.derr.par1 = Wt.at(0, 0, t + 1);
                        model.derr.var.at(0, 0) = Wt.at(0, 0, t + 1);
                    }
                }

                arma::mat Theta_new(model.nP, N, arma::fill::zeros);
                bool positive_noise = (t < Theta.n_rows) ? true : false;
                #ifdef _OPENMP
                    #pragma omp parallel for num_threads(NUM_THREADS) schedule(runtime)
                #endif
                for (unsigned int i = 0; i < N; i++)
                {
                    arma::vec gtheta = StateSpace::func_gt(model.ftrans, model.fgain, model.dlag, Theta.slice(t + B - 1).col(i), y.at(t), model.seas.period, model.seas.in_state);
                    arma::vec eps = Wt_chol.t() * arma::randn<arma::vec>(gtheta.n_elem);
                    if (positive_noise)
                    {
                        eps = arma::abs(eps);
                    }
                    arma::vec theta_new = gtheta + eps;
                    Theta_new.col(i) = theta_new;

                    double ft = StateSpace::func_ft(model.ftrans, model.fgain, model.dlag, model.seas, t + 1, theta_new, y);
                    double lambda = LinkFunc::ft2mu(ft, model.flink);
                    weights.at(i) = ObsDist::loglike(y.at(t + 1), model.dobs.name, lambda, model.dobs.par2, true);
                }

                double wmax = weights.max();
                weights.for_each([&wmax](arma::vec::elem_type &val)
                                 { val -= wmax; });
                weights = arma::exp(weights);

                log_cond_marginal.at(t + 1) = std::log(arma::accu(weights) + EPS) - logN;
                arma::uvec resample_idx = get_resample_index(weights);

                Theta.slice(t + B) = Theta_new;
                for (unsigned int b = t + 1; b < t + B + 1; b++)
                {
                    Theta.slice(b) = Theta.slice(b).cols(resample_idx);
                    psi_smooth.row(b) = Theta.slice(b).row(0);
                }

                psi_forward.row(t + B) = Theta.slice(t + B).row(0);

                if (verbose)
                {
                    Rcpp::Rcout << "\rProgress: " << t + 1 << "/" << nT;
                }

            } // loop over time

            if (verbose)
            {
                Rcpp::Rcout << std::endl;
            }

            output["psi_filter"] = Rcpp::wrap(arma::quantile(psi_forward.tail_rows(y.n_elem), ci_prob, 1));
            output["psi"] = Rcpp::wrap(arma::quantile(psi_smooth.tail_rows(y.n_elem), ci_prob, 1));
            output["log_marginal_likelihood"] = arma::accu(log_cond_marginal);
        }
    };

    class FFBS : public SequentialMonteCarlo
    {
    private:
        arma::vec eff_forward; // (nT + 1) x 1
        arma::mat params; // m x N
        arma::cube Wt; // p x p x (nT + 1)

    public:
        FFBS(
            const Model &dgtf_model,
            const Rcpp::List &opts_in) : SequentialMonteCarlo(dgtf_model, opts_in)
        {
            params.set_size(2 + dgtf_model.seas.period, N);
            params.row(0).fill(dgtf_model.dobs.par2);
            params.row(1).fill(dgtf_model.derr.par1);
        }

        Rcpp::List get_output(const bool &summarize = true)
        {
            // arma::vec ci_prob = {0.025, 0.5, 0.975};
            // Rcpp::List output;

            // arma::mat psi_forward = Theta.row_as_mat(0);

            // arma::mat psi_f = arma::quantile(psi_forward, ci_prob, 1);
            // output["psi_filter"] = Rcpp::wrap(psi_f);
            // output["eff_forward"] = Rcpp::wrap(eff_forward);

            // if (smoothing)
            // {
            //     arma::mat psi = arma::quantile(psi_smooth, ci_prob, 1);
            //     output["psi"] = Rcpp::wrap(psi);
            // }

            // output["log_marginal_likelihood"] = marginal_likelihood(log_cond_marginal, true);

            return output;
        }

        Rcpp::List forecast(const Model &model, const arma::vec &y)
        {
            arma::vec Wtmp(N, arma::fill::zeros);
            if (use_discount)
            {
                Wtmp.fill(Wt.at(0, 0, y.n_elem - 1));
            }
            else
            {
                Wtmp.fill(model.derr.par1);
            }

            Rcpp::List out;
            if (smoothing)
            {
                out = StateSpace::forecast(y, Theta_smooth, Wtmp, model, nforecast);
            }
            else
            {
                out = StateSpace::forecast(y, Theta, Wtmp, model, nforecast);
            }
            return out;
        }

        arma::mat optimal_discount_factor(
            Model &model,
            const arma::vec &y,
            const double &from,
            const double &to,
            const double &delta = 0.01,
            const std::string &loss = "quadratic",
            const bool &verbose = VERBOSE)
        {
            const unsigned int nT = y.n_elem - 1;
            arma::vec grid = arma::regspace<arma::vec>(from, delta, to);
            unsigned int nelem = grid.n_elem;
            arma::mat stats(nelem, 4, arma::fill::zeros);

            use_discount = true;

            for (unsigned int i = 0; i < nelem; i++)
            {
                Rcpp::checkUserInterrupt();

                discount_factor = grid.at(i);
                stats.at(i, 0) = discount_factor;

                infer(model, y, false);
                arma::cube theta_tmp = Theta.tail_slices(y.n_elem);

                double err_forecast = 0.;
                double cov_forecast = 0.;
                double width_forecast = 0.;
                StateSpace::forecast_error(err_forecast, cov_forecast, width_forecast, theta_tmp, y, model, loss, false);
                stats.at(i, 1) = err_forecast;
                stats.at(i, 2) = cov_forecast;
                stats.at(i, 3) = width_forecast;


                if (verbose)
                {
                    Rcpp::Rcout << "\rProgress: " << i + 1 << "/" << nelem;
                }
            }

            if (verbose)
            {
                Rcpp::Rcout << std::endl;
            }

            return stats;
        }

        arma::mat optimal_W(
            Model &model,
            const arma::vec &y,
            const arma::vec &grid,
            const std::string &loss = "quadratic",
            const bool &verbose = VERBOSE)
        {
            unsigned int nelem = grid.n_elem;
            arma::mat stats(nelem, 4, arma::fill::zeros);

            use_discount = false;

            for (unsigned int i = 0; i < nelem; i++)
            {
                Rcpp::checkUserInterrupt();

                Wt.tube(0, 0).fill(grid.at(i));
                stats.at(i, 0) = grid.at(i);

                infer(model, y, false);
                arma::cube theta_tmp = Theta.tail_slices(y.n_elem);

                double err_forecast = 0.;
                double cov_forecast = 0.;
                double width_forecast = 0.;
                StateSpace::forecast_error(err_forecast, cov_forecast, width_forecast, theta_tmp, y, model, loss, false);

                stats.at(i, 1) = err_forecast;
                stats.at(i, 2) = cov_forecast;
                stats.at(i, 3) = width_forecast;

                if (verbose)
                {
                    Rcpp::Rcout << "\rProgress: " << i + 1 << "/" << nelem;
                }
            }

            if (verbose)
            {
                Rcpp::Rcout << std::endl;
            }

            return stats;
        }


        void smoother(Model &model, const arma::vec &y, const bool &verbose = VERBOSE)
        {
            const unsigned int nT = y.n_elem - 1;
            arma::uvec idx = sample(N, M, weights, true, true);       // M x 1
            arma::mat theta_last = Theta.slice(nT);         // p x N
            arma::mat theta_sub = theta_last.cols(idx);         // p x M

            Theta_smooth = arma::zeros<arma::cube>(model.nP, M, nT + B);
            Theta_smooth.slice(nT) = theta_sub;

            for (unsigned int t = nT; t > 0; t--)
            {
                Rcpp::checkUserInterrupt();

                // Resampling density for theta[t-1] is: p(theta[t] | theta[t-1], W[t]).
                arma::rowvec psi_smooth_now = Theta_smooth.slice(t).row(0);                       // 1 x M
                arma::rowvec psi_filter_prev = Theta.slice(t - 1).row(0);                         // 1 x N
                arma::uvec smooth_idx;
                if (model.derr.full_rank)
                {
                    if (use_discount)
                    {
                        smooth_idx = get_smooth_index(Theta_smooth.slice(t), Theta.slice(t-1), Wt.slice(t));
                    }
                    else
                    {
                        smooth_idx = get_smooth_index(Theta_smooth.slice(t), Theta.slice(t-1), model.derr.var);
                    }
                }
                else
                {
                    arma::vec Wsqrt(M);
                    if (use_discount)
                    {
                        Wsqrt.fill(std::sqrt(Wt.at(0, 0, t)));
                    }
                    else
                    {
                        Wsqrt.fill(std::sqrt(model.derr.par1));
                    }
                    smooth_idx = get_smooth_index(psi_smooth_now, psi_filter_prev, Wsqrt); // M x 1
                }

                arma::mat theta_next = Theta.slice(t - 1);
                theta_next = theta_next.cols(smooth_idx); // p x M
                Theta_smooth.slice(t - 1) = theta_next;

                if (verbose)
                {
                    Rcpp::Rcout << "\rSmoothing: " << y.n_elem - t << "/" << nT;
                }
            }

            if (verbose)
            {
                Rcpp::Rcout << std::endl;
            }

            arma::mat psi = Theta_smooth.row_as_mat(0);
            output["psi"] = Rcpp::wrap(arma::quantile(psi, ci_prob, 1));
            return;
        }


        void infer(Model &model, const arma::vec &y, const bool &verbose = VERBOSE)
        {
            if (use_discount)
            {
                LBA::LinearBayes lba(use_discount, discount_factor);
                lba.filter(model, y);
                arma::mat Gt = TransFunc::init_Gt(model.nP, model.dlag, model.ftrans, model.seas.period, model.seas.in_state);

                Wt = arma::zeros<arma::cube>(model.nP, model.nP, y.n_elem);

                if (model.derr.full_rank)
                {
                    Wt.slice(0) = arma::symmatu(lba.Ct.slice(0));
                }
                else
                {
                    Wt.at(0, 0, 0) = std::abs(lba.Ct.at(0, 0, 0)) + EPS;
                }

                for (unsigned int t = 1; t < y.n_elem; t++)
                {
                    LBA::func_Gt(Gt, model, lba.mt.col(t - 1), y.at(t - 1));
                    arma::mat Pt = Gt * lba.Ct.slice(t - 1) * Gt.t();
                    arma::mat Wt_hat = (1. / discount_factor - 1.) * Pt;
                    Wt_hat.diag() += EPS8;

                    if (model.derr.full_rank)
                    {
                        Wt.slice(t) = arma::symmatu(Wt_hat);
                    }
                    else
                    {
                        Wt.at(0, 0, t) = Wt_hat.at(0, 0);
                    }   
                }
            }


            Theta = arma::zeros<arma::cube>(model.nP, N, y.n_elem);
            Theta.slice(0) = arma::randn<arma::mat>(model.nP, N);
            if (model.seas.in_state && model.seas.period > 0)
            {
                for (unsigned int i = model.nP - model.seas.period; i < model.nP; i++)
                {
                    Theta.slice(0).row(i) = arma::randu<arma::rowvec>(N, arma::distr_param(model.seas.lobnd, model.seas.hibnd));
                    Theta.slice(1).row(i) = arma::randu<arma::rowvec>(N, arma::distr_param(model.seas.lobnd, model.seas.hibnd));
                }
            }

            arma::mat weights_forward(y.n_elem, N, arma::fill::zeros);
            arma::vec eff_forward(y.n_elem, arma::fill::zeros);
            double log_cond_marg = SMC::SequentialMonteCarlo::auxiliary_filter(
                Theta, weights_forward, eff_forward, Wt,
                model, y, N, false, true, 
                use_discount, discount_factor, verbose);

            arma::mat psi = Theta.row_as_mat(0); // (nT + 1) x N
            output["psi_filter"] = Rcpp::wrap(arma::quantile(psi, ci_prob, 1));
            output["eff_forward"] = Rcpp::wrap(eff_forward.t());
            output["log_marginal_likelihood"] = log_cond_marg;

            if (smoothing)
            {
                smoother(model, y, verbose);
            }

            return;
        }
    };

    /**
     * @brief Two-filter smoothing
     *
     */
    class TFS : public SequentialMonteCarlo
    {
    private:
        arma::cube Wt; // p x p x (nT + 1)
        arma::mat weights_forward;  // (nT + 1) x N
        arma::mat weights_backward; // (nT + 1) x N
        arma::cube Theta_backward; // p x N x (nT + 1)
        bool resample_all = false;

    public:
        TFS(
            const Model &model,
            const Rcpp::List &opts_in) : SequentialMonteCarlo(model, opts_in)
        {
            Rcpp::List opts = opts_in;
            if (opts.containsElementNamed("resample_all"))
            {
                resample_all = Rcpp::as<bool>(opts["resample_all"]);
            }
            return;
        }

        static Rcpp::List default_settings()
        {
            Rcpp::List opts = SequentialMonteCarlo::default_settings();
            opts["resample_all"] = false;
            return opts;
        }

        Rcpp::List get_output(const bool &summarize = true)
        {
            return output;
        }

        Rcpp::List forecast(const Model &model, const arma::vec &y)
        {
            arma::vec Wtmp(N, arma::fill::zeros);
            if (use_discount)
            {
                Wtmp.fill(Wt.at(0, 0, y.n_elem - 1));
            }
            else
            {
                Wtmp.fill(model.derr.par1);
            }

            Rcpp::List out;
            if (smoothing)
            {
                out = StateSpace::forecast(y, Theta_smooth, Wtmp, model, nforecast);
            }
            else
            {
                out = StateSpace::forecast(y, Theta, Wtmp, model, nforecast);
            }
            return out;
        }

        arma::mat optimal_discount_factor(
            Model &model,
            const arma::vec &y,
            const double &from,
            const double &to,
            const double &delta = 0.01,
            const std::string &loss = "quadratic",
            const bool &verbose = VERBOSE)
        {
            arma::vec grid = arma::regspace<arma::vec>(from, delta, to);
            unsigned int nelem = grid.n_elem;
            arma::mat stats(nelem, 4, arma::fill::zeros);

            use_discount = true;

            for (unsigned int i = 0; i < nelem; i++)
            {
                Rcpp::checkUserInterrupt();

                discount_factor = grid.at(i);
                stats.at(i, 0) = discount_factor;

                infer(model, y, false);
                arma::cube theta_tmp = Theta.tail_slices(y.n_elem);

                double err_forecast = 0.;
                double cov_forecast = 0.;
                double width_forecast = 0.;
                StateSpace::forecast_error(err_forecast, cov_forecast, width_forecast, theta_tmp, y, model, loss, false);
                stats.at(i, 1) = err_forecast;
                stats.at(i, 2) = cov_forecast;
                stats.at(i, 3) = width_forecast;


                if (verbose)
                {
                    Rcpp::Rcout << "\rProgress: " << i + 1 << "/" << nelem;
                }
            }

            if (verbose)
            {
                Rcpp::Rcout << std::endl;
            }

            return stats;
        }

        arma::mat optimal_W(
            const Model &model,
            const arma::vec &y, 
            const arma::vec &grid,
            const std::string &loss = "quadratic",
            const bool &verbose = VERBOSE)
        {
            unsigned int nelem = grid.n_elem;
            arma::mat stats(nelem, 4, arma::fill::zeros);
            Model mod = model;

            use_discount = false;

            for (unsigned int i = 0; i < nelem; i++)
            {
                Rcpp::checkUserInterrupt();

                mod.derr.par1 = grid.at(i);
                stats.at(i, 0) = mod.derr.par1;

                infer(mod, y, false);
                arma::cube theta_tmp = Theta.tail_slices(y.n_elem);

                double err_forecast = 0.;
                double cov_forecast = 0.;
                double width_forecast = 0.;
                StateSpace::forecast_error(err_forecast, cov_forecast, width_forecast, theta_tmp, y, model, loss, false);
                stats.at(i, 1) = err_forecast;

                stats.at(i, 2) = cov_forecast;
                stats.at(i, 3) = width_forecast;

                if (verbose)
                {
                    Rcpp::Rcout << "\rProgress: " << i + 1 << "/" << nelem;
                }
            }

            if (verbose)
            {
                Rcpp::Rcout << std::endl;
            }

            return stats;
        }


        void backward_filter(Model &model, const arma::vec &y, const bool &verbose = VERBOSE)
        {
            std::map<std::string, TransFunc::Transfer> trans_list = TransFunc::trans_list;

            arma::vec eff_backward(y.n_elem, arma::fill::zeros);
            Theta_backward = Theta; // p x N x (nT + B)
            weights_backward = weights_forward;

            arma::mat mu_marginal(model.nP, y.n_elem, arma::fill::zeros);
            arma::cube Prec_marginal(model.nP, model.nP, y.n_elem);
            prior_forward(mu_marginal, Prec_marginal, model, y, Wt, use_discount);

            arma::vec log_marg(N, arma::fill::zeros);
            for (unsigned int i = 0; i < N; i++)
            {
                arma::vec theta = Theta_backward.slice(y.n_elem - 1).col(i);
                log_marg.at(i) = MVNorm::dmvnorm2(
                    theta, mu_marginal.col(y.n_elem - 1), Prec_marginal.slice(y.n_elem - 1), true);
            }


            for (unsigned int t = y.n_elem - 2; t > 0; t--)
            {
                Rcpp::checkUserInterrupt();

                arma::vec logq(N, arma::fill::zeros);
                arma::mat loc(model.nP, N, arma::fill::zeros);
                arma::cube prec_chol_inv; // nP x nP x N
                if (model.derr.full_rank)
                {
                    prec_chol_inv = arma::zeros<arma::cube>(model.nP, model.nP, N); // nP x nP x N
                }

                if (use_discount)
                {
                    if (model.derr.full_rank)
                    {
                        model.derr.var = Wt.slice(t + 1);
                    }
                    else
                    {
                        model.derr.var.at(0, 0) = Wt.at(0, 0, t + 1);
                        model.derr.par1 = Wt.at(0, 0, t + 1);
                    }
                }

                arma::mat ut(model.nP, N, arma::fill::zeros);
                arma::cube Uprec = arma::zeros<arma::cube>(model.nP, model.nP, N);
                arma::vec tau = qbackcast(
                    loc, prec_chol_inv, ut, Uprec, logq,
                    model, t, Theta_backward.slice(t + 1), Theta_backward.slice(t),
                    mu_marginal.col(t), mu_marginal.col(t + 1), Prec_marginal.slice(t), y);
                tau = tau % weights;

                if (t < y.n_elem - 2)
                {
                    arma::uvec resample_idx = get_resample_index(tau);

                    Theta_backward.slice(t + 1) = Theta_backward.slice(t + 1).cols(resample_idx);
                    
                    if (model.derr.full_rank)
                    {
                        loc = loc.cols(resample_idx);
                        prec_chol_inv = prec_chol_inv.slices(resample_idx);

                        ut = ut.cols(resample_idx);
                        Uprec = Uprec.slices(resample_idx);
                    }

                    log_marg = log_marg.elem(resample_idx);
                    logq = logq.elem(resample_idx);
                    weights = weights.elem(resample_idx);
                }

                weights_backward.row(t + 1) = logq.t();

                /*
                Propagation
                */
                arma::mat Theta_next = Theta_backward.slice(t + 1); // nP x N;
                arma::mat Theta_cur(model.nP, N, arma::fill::zeros);
                arma::vec mu = mu_marginal.col(t);
                arma::mat Prec = Prec_marginal.slice(t);
                #ifdef _OPENMP
                    #pragma omp parallel for num_threads(NUM_THREADS) schedule(runtime)
                #endif
                for (unsigned int i = 0; i < N; i++)
                {
                    arma::vec theta_cur;
                    double logp = 0.;
                    if (model.derr.full_rank)
                    {
                        arma::vec eps = arma::randn(Theta_cur.n_rows);
                        arma::vec zt = prec_chol_inv.slice(i).t() * loc.col(i) + eps; // shifted
                        theta_cur = prec_chol_inv.slice(i) * zt;
                        logq.at(i) += MVNorm::dmvnorm0(zt, loc.col(i), prec_chol_inv.slice(i), true);

                        logp += MVNorm::dmvnorm2(theta_cur, ut.col(i), Uprec.slice(i), true);
                    }
                    else
                    {
                        double eps = R::rnorm(0., std::sqrt(model.derr.par1));
                        theta_cur = StateSpace::func_backward_gt(model.ftrans, model.fgain, model.dlag, Theta_next.col(i), y.at(t), eps, model.seas.period, model.seas.in_state);

                        if (!use_discount)
                        {
                            logq.at(i) += R::dnorm4(eps, 0, std::sqrt(model.derr.par1), true);
                            logp += R::dnorm4(eps, 0., std::sqrt(model.derr.par1), true);
                        }
                    }

                    Theta_cur.col(i) = theta_cur;

                    double ft_cur = StateSpace::func_ft(model.ftrans, model.fgain, model.dlag, model.seas, t, theta_cur, y);
                    double lambda_cur = LinkFunc::ft2mu(ft_cur, model.dobs.name);

                    logp += ObsDist::loglike(y.at(t), model.dobs.name, lambda_cur, model.dobs.par2, true); // observation density
                    // logp.at(i) -= log_marg.at(i);
                    logp -= log_marg.at(i); // p(theta[t + 1])
                    log_marg.at(i) = MVNorm::dmvnorm2(theta_cur, mu, Prec, true);
                    logp += log_marg.at(i); // p(theta[t])

                    weights.at(i) = logp - logq.at(i); // + logw_next;
                } // loop over i, index of particles

                Theta_backward.slice(t) = Theta_cur;
                double wmax = weights.max();
                weights.for_each([&wmax](arma::vec::elem_type &val)
                                 { val -= wmax; });
                weights = arma::exp(weights);
                eff_backward.at(t + 1) = effective_sample_size(weights);

                if (verbose)
                {
                    Rcpp::Rcout << "\rBackward Filtering: " << y.n_elem - t << "/" << (y.n_elem - 1);
                }

            } // propagate and resample

            if (verbose)
            {
                Rcpp::Rcout << std::endl;
            }

            // psi_backward = Theta_backward.row_as_mat(0); // (nT + 1) x N
            arma::mat psi = Theta_backward.row_as_mat(0);
            output["psi_backward"] = Rcpp::wrap(arma::quantile(psi, ci_prob, 1));
            output["eff_backward"] = Rcpp::wrap(eff_backward.t());
            return;
        }

        void smoother(Model &model, const arma::vec &y, const bool &verbose = VERBOSE)
        {
            const bool full_rank = false;
            Theta_smooth = Theta;

            arma::mat mu_marginal(model.nP, y.n_elem, arma::fill::zeros);
            arma::cube Prec_marginal(model.nP, model.nP, y.n_elem);
            prior_forward(mu_marginal, Prec_marginal, model, y, Wt, use_discount);

            for (unsigned int t = 1; t < (y.n_elem - 1); t++)
            {
                Rcpp::checkUserInterrupt();
                double yhat_cur = LinkFunc::mu2ft(y.at(t), model.flink, 0.);
                arma::mat Theta_cur(model.nP, N, arma::fill::zeros);

                arma::mat Wcur_prec(model.nP, model.nP, arma::fill::zeros);
                arma::mat Wnext_prec = Wcur_prec;
                if (use_discount)
                {
                    if (model.derr.full_rank)
                    {
                        Wcur_prec = arma::inv(Wt.slice(t));
                        Wnext_prec = arma::inv(Wt.slice(t + 1));
                    }
                    else
                    {
                        Wcur_prec.at(0, 0) = 1. / Wt.at(0, 0, t);
                        Wnext_prec.at(0, 0) = 1. / Wt.at(0, 0, t + 1);
                    }
                }
                else
                {
                    if (model.derr.full_rank)
                    {
                        Wcur_prec = arma::inv(model.derr.var);
                        Wnext_prec = Wcur_prec;
                    }
                    else
                    {
                        Wcur_prec.at(0, 0) = 1. / model.derr.par1;
                        Wnext_prec.at(0, 0) = 1. / model.derr.par1;
                    }
                }

                // arma::vec logp(N, arma::fill::zeros);
                // arma::vec logq = arma::vectorise(weights_forward.row(t - 1) + weights_backward.row(t + 1));

                arma::mat Theta_next = Theta_backward.slice(t + 1);
                arma::vec mu = mu_marginal.col(t + 1);
                arma::mat Prec = Prec_marginal.slice(t + 1);
                #ifdef _OPENMP
                    #pragma omp parallel for num_threads(NUM_THREADS) schedule(runtime)
                #endif
                for (unsigned int i = 0; i < N; i++)
                {
                    double logq = weights_forward.at(t - 1, i) + weights_backward.at(t + 1, i);

                    arma::vec gtheta_cur = StateSpace::func_gt(model.ftrans, model.fgain, model.dlag, Theta.slice(t - 1).col(i), y.at(t - 1), model.seas.period, model.seas.in_state); // g(theta[t-1])

                    double ft = StateSpace::func_ft(model.ftrans, model.fgain, model.dlag, model.seas, t, gtheta_cur, y);
                    double eta = ft;
                    double lambda = LinkFunc::ft2mu(eta, model.flink);
                    double Vt = ApproxDisturbance::func_Vt_approx(lambda, model.dobs, model.flink); // (eq 3.11)

                    arma::vec theta_cur;
                    double logp = 0.;
                    

                    if (!model.derr.full_rank)
                    {
                        theta_cur = gtheta_cur;
                        double eps = R::rnorm(0., std::sqrt(1. / Wcur_prec.at(0, 0)));
                        theta_cur.at(0) += eps;

                        logq += R::dnorm4(eps, 0., std::sqrt(1. / Wcur_prec.at(0, 0)), true);
                        logp += R::dnorm4(theta_cur.at(0), gtheta_cur.at(0), std::sqrt(1. / Wcur_prec.at(0, 0)), true); // p(theta[t] | g(theta[t-1]), W[t])
                    }
                    else
                    {
                        arma::vec Ft = LBA::func_Ft(model.ftrans, model.fgain, model.dlag, t, gtheta_cur, y, LBA_FILL_ZERO, model.seas.period, model.seas.in_state);
                        double ft_tilde = ft - arma::as_scalar(Ft.t() * gtheta_cur);
                        double delta = yhat_cur - ft_tilde;
                        arma::mat Gt = TransFunc::init_Gt(model.nP, model.dlag, model.ftrans, model.seas.period, model.seas.in_state);
                        LBA::func_Gt(Gt, model, gtheta_cur, y.at(t));

                        arma::mat prec = Wcur_prec + Gt.t() * Wnext_prec * Gt + Ft * Ft.t() / Vt;
                        arma::mat prec_chol = arma::chol(arma::symmatu(prec));
                        arma::mat prec_chol_inv = arma::inv(arma::trimatu(prec_chol));
                        arma::mat Sigma = prec_chol_inv * prec_chol_inv.t();

                        arma::vec mu = Wcur_prec * gtheta_cur + Gt.t() * Wnext_prec * Theta_next.col(i) + Ft * (delta / Vt);
                        mu = Sigma * mu;

                        theta_cur = mu + prec_chol_inv * arma::randn(model.nP);
                        logq += MVNorm::dmvnorm2(theta_cur, mu, prec, true);
                        logp += MVNorm::dmvnorm2(theta_cur, gtheta_cur, Wcur_prec, true);
                    }

                    Theta_cur.col(i) = theta_cur;
                    arma::vec gtheta_next = StateSpace::func_gt(model.ftrans, model.fgain, model.dlag, theta_cur, y.at(t), model.seas.period, model.seas.in_state);

                    if (!model.derr.full_rank)
                    {
                        logp += R::dnorm4(Theta_next.at(0, i), gtheta_next.at(0), std::sqrt(1. / Wnext_prec.at(0, 0)), true); // p(theta[t+1] | g(theta[t]), W[t+1])
                    }
                    else
                    {
                        logp += MVNorm::dmvnorm2(Theta_next.col(i), gtheta_next, Wnext_prec);
                    }

                    ft = StateSpace::func_ft(model.ftrans, model.fgain, model.dlag, model.seas, t, theta_cur, y);
                    lambda = LinkFunc::ft2mu(ft, model.flink);
                    logp += ObsDist::loglike(y.at(t), model.dobs.name, lambda, model.dobs.par2, true);
                    logp -= MVNorm::dmvnorm2(Theta_next.col(i), mu, Prec, true);

                    weights.at(i) = logp - logq; // + log_forward + log_backward;
                } // loop over particle i

                double wmax = weights.max();
                weights.for_each([&wmax](arma::vec::elem_type &val)
                                 { val -= wmax; });
                weights = arma::exp(weights);

                arma::uvec resample_idx = get_resample_index(weights);
                Theta_smooth.slice(t) = Theta_cur.cols(resample_idx);
                if (verbose)
                {
                    std::cout << "\rSmoothing: " << t + 1 << "/" << (y.n_elem - 1);
                }
            }

            if (verbose)
            {
                std::cout << std::endl;
            }

            arma::mat psi = Theta_smooth.row_as_mat(0);
            output["psi"] = Rcpp::wrap(arma::quantile(psi, ci_prob, 1));
            return;
        }

        void infer(Model &model, const arma::vec &y, const bool &verbose = VERBOSE)
        {
            if (use_discount)
            {
                LBA::LinearBayes lba(use_discount, discount_factor);
                lba.filter(model, y);
                arma::mat Gt = TransFunc::init_Gt(model.nP, model.dlag, model.ftrans, model.seas.period, model.seas.in_state);

                Wt = arma::zeros<arma::cube>(model.nP, model.nP, y.n_elem);

                if (model.derr.full_rank)
                {
                    Wt.slice(0) = arma::symmatu(lba.Ct.slice(0));
                }
                else
                {
                    Wt.at(0, 0, 0) = std::abs(lba.Ct.at(0, 0, 0)) + EPS;
                }

                for (unsigned int t = 1; t < y.n_elem; t++)
                {
                    LBA::func_Gt(Gt, model, lba.mt.col(t - 1), y.at(t - 1));
                    arma::mat Pt = Gt * lba.Ct.slice(t - 1) * Gt.t();
                    arma::mat Wt_hat = (1. / discount_factor - 1.) * Pt;
                    Wt_hat.diag() += EPS8;

                    if (model.derr.full_rank)
                    {
                        Wt.slice(t) = arma::symmatu(Wt_hat);
                    }
                    else
                    {
                        Wt.at(0, 0, t) = Wt_hat.at(0, 0);
                    }   
                }
            }

            // forward filter
            Theta = arma::zeros<arma::cube>(model.nP, N, y.n_elem);
            Theta.slice(0) = arma::randn<arma::mat>(model.nP, N);
            if (model.seas.in_state && model.seas.period > 0)
            {
                for (unsigned int i = model.nP - model.seas.period; i < model.nP; i++)
                {
                    Theta.slice(0).row(i) = arma::randu<arma::rowvec>(N, arma::distr_param(model.seas.lobnd, model.seas.hibnd));
                    Theta.slice(1).row(i) = arma::randu<arma::rowvec>(N, arma::distr_param(model.seas.lobnd, model.seas.hibnd));
                }
            }

            arma::vec eff_forward(y.n_elem, arma::fill::zeros);
            double log_cond_marg = SMC::SequentialMonteCarlo::auxiliary_filter(
                Theta, weights_forward, eff_forward, Wt,
                model, y, N, false, true,
                use_discount, discount_factor, verbose);

            arma::mat psi = Theta.row_as_mat(0); // (nT + 1) x N
            output["eff_forward"] = Rcpp::wrap(eff_forward.t());
            output["log_marginal_likelihood"] = log_cond_marg;

            if (model.seas.period > 0 && model.seas.in_state)
            {
                const unsigned int nstate = model.nP - model.seas.period;
                arma::cube seas = arma::zeros<arma::cube>(model.seas.period, N, Theta.n_slices);
                for (unsigned int i = 0; i < model.seas.period; i++)
                {
                    int j = i + nstate;
                    seas.slice(0).row(i) = Theta.slice(0).row(j);
                    for (unsigned int t = 0; t < Theta.n_slices; t++)
                    {
                        j -= 1;
                        j = (j < nstate) ? model.nP - 1 : j;
                        seas.slice(t).row(i) = Theta.slice(t).row(j);
                    }
                }

                output["seas_forward"] = Rcpp::wrap(seas);
            }

            if (smoothing)
            {
                output["psi_filter"] = Rcpp::wrap(arma::quantile(psi, ci_prob, 1));
                backward_filter(model, y, verbose);
                smoother(model, y, verbose);
            }
            else
            {
                output["psi"] = Rcpp::wrap(arma::quantile(psi, ci_prob, 1));
            }

            return;
        }
    };

    class PL : public SequentialMonteCarlo
    {
    public:
        PL(const Model &model, const Rcpp::List &opts_in) : SequentialMonteCarlo(model, opts_in) 
        {
            Rcpp::List opts = opts_in;
            B = 1;

            max_iter = 10;
            if (opts.containsElementNamed("max_iter"))
            {
                max_iter = Rcpp::as<unsigned int>(opts["max_iter"]);
            }

            {
                prior_W.init("invgamma", 1., 1.);
                if (opts.containsElementNamed("W"))
                {
                    Rcpp::List par_opts = opts["W"];
                    prior_W.init(par_opts);
                }

                aw_forward.set_size(N);
                aw_forward.fill(prior_W.par1);
                bw_forward.set_size(N);
                bw_forward.fill(prior_W.par2);

                W_filter.set_size(N);
                W_filter.ones();

                if (prior_W.infer)
                {
                    std::string par_name = "W_init";
                    Dist dist_W_init;
                    dist_W_init.init("gamma", 1., 1.);
                    if (opts.containsElementNamed("W_init"))
                    {
                        Rcpp::List W_init_opts = Rcpp::as<Rcpp::List>(opts["W_init"]);
                        dist_W_init.init(W_init_opts);
                    }

                    W_filter = draw_param_init(dist_W_init, N);
                    use_discount = false;
                }
            }

            param_filter.set_size(model.seas.period + 3, N);

            {
                prior_seas.init("gaussian", 0., 10.);
                if (opts.containsElementNamed("seas"))
                {
                    Rcpp::List par_opts = opts["seas"];
                    prior_seas.init(par_opts);
                }

                aseas_forward.set_size(model.seas.period, N);
                aseas_forward.fill(prior_seas.par1); // mean
                bseas_forward.set_size(model.seas.period, model.seas.period, N);
                bseas_forward.zeros();

                for (unsigned int i = 0; i < N; i++)
                {
                    bseas_forward.slice(i).diag().fill(prior_seas.par2);
                    if (prior_seas.infer)
                    {
                        param_filter.col(i).head(model.seas.period) = arma::randu<arma::vec>(
                            model.seas.period, arma::distr_param(model.seas.lobnd, model.seas.hibnd));
                    }
                    else
                    {
                        param_filter.col(i).head(model.seas.period) = model.seas.val;
                    }
                }
            }


            {
                prior_rho.init("gaussiaN", 1., 1.);
                if (opts.containsElementNamed("rho"))
                {
                    Rcpp::List opts_tmp = Rcpp::as<Rcpp::List>(opts["rho"]);
                    prior_rho.init(opts_tmp);
                }
                
                if (prior_rho.infer)
                {
                    param_filter.row(model.seas.period) = arma::randu<arma::rowvec>(param_filter.n_cols, arma::distr_param(0, 5));
                }
                else
                {
                    param_filter.row(model.seas.period).fill(std::log(model.dobs.par2));
                }
            }

            obs_update = prior_seas.infer || prior_rho.infer;

            {
                prior_par1.init("gamma", 0.1, 0.1);
                if (opts.containsElementNamed("par1"))
                {
                    Rcpp::List opts_tmp = Rcpp::as<Rcpp::List>(opts["par1"]);
                    prior_par1.init(opts_tmp);
                }
                param_filter.row(model.seas.period + 1).fill(model.dlag.par1);
            }

            {
                prior_par2.init("gamma", 0.1, 0.1);
                if (opts.containsElementNamed("par2"))
                {
                    Rcpp::List opts_tmp = Rcpp::as<Rcpp::List>(opts["par2"]);
                    prior_par2.init(opts_tmp);
                }
                param_filter.row(model.seas.period + 2).fill(std::log(model.dlag.par2));
            }

            lag_update = prior_par1.infer || prior_par2.infer;
            prior_par1.infer = lag_update;
            prior_par2.infer = lag_update;

            param_backward = param_filter;
            param_smooth = param_smooth;

            filter_pass = false;

            return;
        }


        static Rcpp::List default_settings()
        {
            Rcpp::List opts;
            opts = SequentialMonteCarlo::default_settings();
            opts["max_iter"] = 10;

            Rcpp::List W_opts;
            W_opts["infer"] = false;
            W_opts["prior_param"] = Rcpp::NumericVector::create(1., 1.);
            W_opts["prior_name"] = "invgamma";
            opts["W"] = W_opts;

            Rcpp::List rho_opts;
            rho_opts["infer"] = false;
            rho_opts["prior_param"] = Rcpp::NumericVector::create(1., 1.);
            rho_opts["prior_name"] = "invgamma";
            opts["rho"] = rho_opts;

            Rcpp::List par1_opts;
            par1_opts["infer"] = false;
            par1_opts["prior_param"] = Rcpp::NumericVector::create(0., 1.);
            rho_opts["prior_name"] = "gaussian";
            opts["par1"] = par1_opts;

            Rcpp::List par2_opts;
            par2_opts["infer"] = false;
            par2_opts["prior_param"] = Rcpp::NumericVector::create(1., 1.);
            par2_opts["prior_name"] = "invgamma";
            opts["par2"] = par2_opts;

            Rcpp::List seas_opts;
            seas_opts["infer"] = false;
            opts["seas"] = seas_opts;

            return opts;
        }

        Rcpp::List get_output(const bool &summarize = TRUE)
        {
            return output;
        }

        Rcpp::List forecast(const Model &model, const arma::vec &y)
        {

            Rcpp::List out;
            if (smoothing)
            {
                out = StateSpace::forecast(y, Theta_smooth, W_backward, model, nforecast);
            }
            else
            {
                out = StateSpace::forecast(y, Theta, W_filter, model, nforecast);
            }
            return out;
        }


        /**
         * @todo Something wrong with forward filter, comparing to TFS.
         */
        void forward_filter(Model &model, const arma::vec &y, const bool &verbose = VERBOSE)
        {
            const unsigned int nT = y.n_elem - 1;
            const bool full_rank = false;
            const double logN = std::log(static_cast<double>(N));

            const double acoef = 0.5 * (3. * discount_factor - 1.) / discount_factor;
            const double hcoef = std::sqrt(1. - acoef * acoef);
            const unsigned int nelem = (int)prior_rho.infer + (int)prior_par1.infer + (int)prior_par2.infer;
            std::vector<unsigned int> indices_c;
            if (nelem > 0)
            {
                if (prior_rho.infer)
                {
                    indices_c.push_back(model.seas.period);
                }
                if (prior_par1.infer)
                {
                    indices_c.push_back(model.seas.period + 1);
                }
                if (prior_par2.infer)
                {
                    indices_c.push_back(model.seas.period + 2);
                }
            }
            arma::uvec indices(indices_c);

            Theta = arma::randn<arma::cube>(model.nP, N, y.n_elem);
            weights_forward.set_size(y.n_elem, N);
            weights_forward.zeros();
            arma::vec eff_forward(y.n_elem, arma::fill::zeros);
            arma::vec log_cond_marginal = eff_forward;

            std::map<std::string, LinkFunc::Func> link_list = LinkFunc::link_list;
            std::map<std::string, AVAIL::Dist> dist_list = AVAIL::dist_list;
            bool nonnegative_par1 = (dist_list[prior_par1.name] != AVAIL::Dist::gaussian);
            bool withinone_par1 = (dist_list[prior_par1.name] == AVAIL::Dist::beta);

            bool nonnegative_par2 = (dist_list[prior_par2.name] != AVAIL::Dist::gaussian);
            bool withinone_par2 = (dist_list[prior_par2.name] == AVAIL::Dist::beta);

            arma::vec yhat = y; // (nT + 1) x 1
            for (unsigned int t = 0; t < nT; t++)
            {
                yhat.at(t) = LinkFunc::mu2ft(y.at(t), model.flink, 0.);
            }

            

            for (unsigned int t = 0; t < nT; t++)
            {
                Rcpp::checkUserInterrupt();
                bool print_time = (t == nT - 2);
                bool burnin = (t <= std::min(0.1 * nT, 20.)) ? true : false;

                /**
                 * @brief Resampling using conditional one-step-ahead predictive distribution as weights.
                 *
                 */

                arma::vec logq(N, arma::fill::zeros);
                arma::mat loc(model.nP, N, arma::fill::zeros); // nP x N
                arma::cube prec_chol_inv; // nP x nP x N
                if (full_rank)
                {
                    prec_chol_inv = arma::zeros<arma::cube>(model.nP, model.nP, N); // nP x nP x N
                }

                arma::vec tau = qforecast(
                    loc, prec_chol_inv, logq,
                    model, t + 1, Theta.slice(t), W_filter, param_filter, y, 
                    prior_W.infer, obs_update, lag_update, use_discount, discount_factor);

                tau = tau % weights;
                arma::uvec resample_idx = get_resample_index(tau);

                Theta.slice(t) = Theta.slice(t).cols(resample_idx);
                weights = weights.elem(resample_idx);
                logq = logq.elem(resample_idx);
                loc = loc.cols(resample_idx);
                if (full_rank)
                {
                    prec_chol_inv = prec_chol_inv.slices(resample_idx);
                }

                eff_forward.at(t + 1) = effective_sample_size(tau);
                weights_forward.row(t) = logq.t();

                // No need to update static parameters if we already inferred them during forward filtering once with the same data (filter_pass = true).
                if (prior_W.infer)
                {
                    W_filter = W_filter.elem(resample_idx);     // gamma[t]
                    aw_forward = aw_forward.elem(resample_idx); // s[t]
                    bw_forward = bw_forward.elem(resample_idx); // s[t]
                }

                if (obs_update || lag_update)
                {
                    param_filter = param_filter.cols(resample_idx);
                }

                if (prior_seas.infer)
                {
                    aseas_forward = aseas_forward.cols(resample_idx); // s[t]
                    bseas_forward = bseas_forward.slices(resample_idx); // s[t]
                }

                arma::vec param_mean;
                arma::mat param_var, param_var_chol;
                arma::mat par; // m x N
                if (nelem > 0)
                {
                    weights /= arma::accu(weights);

                    param_mean.set_size(nelem);
                    param_mean.zeros();
                    param_var.set_size(nelem, nelem);
                    param_var.zeros();
                    param_var_chol = param_var;

                    par = param_filter.rows(indices); // m x N

                    for (unsigned int i = 0; i < N; i++)
                    {
                        param_mean = param_mean + weights.at(i) * par.col(i);
                    }

                    for (unsigned int i = 0; i < N; i++)
                    {
                        arma::vec tdiff = par.col(i) - param_mean;
                        param_var = param_var + weights.at(i) * tdiff * tdiff.t();
                    }

                    if (t == 0)
                    {
                        param_var.diag() += 0.1;
                    }
                    else
                    {
                        param_var.diag() += EPS8;
                    }

                    param_var = arma::symmatu(param_var);
                    param_var_chol = arma::chol(param_var);
                    param_var_chol.for_each([&hcoef](arma::mat::elem_type &val)
                                            { val *= hcoef; });
                    param_var.for_each([&hcoef](arma::mat::elem_type &val)
                                       { val *= hcoef * hcoef; });
                }


                arma::vec logp(N, arma::fill::zeros);

                // #pragma omp parallel for num_threads(NUM_THREADS) schedule(runtime)
                for (unsigned int i = 0; i < N; i++)
                {
                    if (nelem > 0)
                    {
                        arma::vec param_pred = acoef * par.col(i) + (1. - acoef) * param_mean;
                        arma::vec param_eps = arma::randn(nelem);
                        arma::vec param_new = param_pred + param_var_chol.t() * param_eps;
                        logq.at(i) += MVNorm::dmvnorm(param_new, param_pred, param_var, true);

                        arma::vec tmp = param_filter.col(i);
                        tmp(indices) = param_new;
                        param_filter.col(i) = tmp;

                        if (lag_update)
                        {
                            model.dlag.par1 = param_filter.at(model.seas.period + 1, i);           // mu
                            model.dlag.par2 = std::exp(param_filter.at(model.seas.period + 2, i)); // sig2
                        }

                        if (prior_rho.infer)
                        {
                            model.dobs.par2 = std::exp(param_filter.at(model.seas.period, i)); // rho
                        }
                    }

                    arma::vec theta_new; // nP x 1
                    if (full_rank)
                    {
                        arma::vec eps = arma::randn(Theta.n_rows);
                        arma::vec zt = prec_chol_inv.slice(i).t() * loc.col(i) + eps; // shifted
                        theta_new = prec_chol_inv.slice(i) * zt;
                        logq.at(i) += MVNorm::dmvnorm0(zt, loc.col(i), prec_chol_inv.slice(i), true);
                    }
                    else
                    {
                        theta_new = loc.col(i);
                        double eps = R::rnorm(0., std::sqrt(W_filter.at(i)));
                        theta_new.at(0) += eps;
                        logq.at(i) += R::dnorm4(eps, 0., std::sqrt(W_filter.at(i)), true);
                    }

                    Theta.slice(t + 1).col(i) = theta_new;

                    double wtmp = model.derr.par1;
                    if (filter_pass || (prior_W.infer && !burnin))
                    {
                        // If filter_pass = true, we already have estimates of W
                        // if burnin = false with prior_W.infer = true, it means we have particles of W from previous time
                        wtmp = W_filter.at(i);
                    }
                    else if ((prior_W.infer && burnin) || use_discount)
                    {
                        // If burnin = true with prior_W.infer = true, we generate samples from the discount factor approach
                        // If use_discount = true, we assume W is changing dynamically and be accounted for with a discount factor.
                        wtmp = SequentialMonteCarlo::discount_W(
                            Theta.slice(t), discount_factor);
                    } // else, we have prior_W.infer = false && use_discount = false. In this case we assume the prior value as the "true" value of W.

                    if (prior_W.infer && !filter_pass)
                    {
                        double err = theta_new.at(0) - Theta.at(0, i, t);
                        double sse = std::pow(err, 2.);

                        aw_forward.at(i) += 0.5;
                        bw_forward.at(i) += 0.5 * sse;
                        if (!burnin)
                        {
                            wtmp = InverseGamma::sample(aw_forward.at(i), bw_forward.at(i));
                            logq.at(i) += R::dgamma(1. / wtmp, aw_forward.at(i), 1. / bw_forward.at(i), true);
                        }
                    } // Propagate W
                    W_filter.at(i) = wtmp;

                    if (prior_W.infer)
                    {
                        model.derr.par1 = W_filter.at(i);
                    }

                    if (prior_seas.infer)
                    {
                        model.seas.val = param_filter.submat(0, i, model.seas.period - 1, i);
                    }

                    double ft_new = StateSpace::func_ft(model.ftrans, model.fgain, model.dlag, model.seas, t + 1, theta_new, y); // ft(theta[t+1])
                    double lambda_old = LinkFunc::ft2mu(ft_new, model.flink); // ft_new from time t + 1, mu0_filter from time t (old).

                    {
                        if (prior_seas.infer && !filter_pass)
                        {
                            arma::vec xt = model.seas.X.col(t + 1);
                            double Vt_old = ApproxDisturbance::func_Vt_approx(lambda_old, model.dobs, model.flink);

                            arma::mat bseas_chol = arma::chol(arma::symmatu(bseas_forward.slice(i)));
                            arma::mat bseas_chol_inv = arma::inv(arma::trimatu(bseas_chol));
                            arma::mat bseas_prec_prev = bseas_chol_inv * bseas_chol_inv.t();

                            arma::mat bseas_prec = xt * xt.t() / Vt_old;
                            bseas_prec = bseas_prec + bseas_prec_prev;
                            bseas_chol = arma::chol(arma::symmatu(bseas_prec));
                            bseas_chol_inv = arma::inv(arma::trimatu(bseas_chol));
                            arma::mat bseas_var_cur = bseas_chol_inv * bseas_chol_inv.t();
                            bseas_forward.slice(i) = bseas_var_cur;

                            double diff = yhat.at(t + 1) - ft_new;
                            diff += arma::as_scalar(xt.t() * model.seas.val);
                            diff /= Vt_old;
                            arma::vec seas_loc = bseas_prec_prev * aseas_forward.col(i);
                            seas_loc = seas_loc + xt * diff;

                            aseas_forward.col(i) = bseas_var_cur * seas_loc;

                            arma::vec seas_new = aseas_forward.col(i) + bseas_chol_inv * arma::randn<arma::vec>(model.seas.period);

                            if (link_list[model.flink] == LinkFunc::Func::identity)
                            {
                                unsigned int cnt = 0;
                                while (seas_new.min() < 0 && cnt < max_iter)
                                {
                                    seas_new = aseas_forward.col(i) + bseas_chol_inv * arma::randn<arma::vec>(model.seas.period);
                                    cnt++;
                                }

                                if (seas_new.min() < 0)
                                {
                                    throw std::invalid_argument("SMC::PL::filter - negative varphi when using identity link.");
                                }
                            }

                            param_filter.submat(0, i, model.seas.period - 1, i) = seas_new;
                            logq.at(i) += MVNorm::dmvnorm(seas_new, aseas_forward.col(i), bseas_var_cur, true);

                        } // inference of seasonal components

                        if (prior_seas.infer)
                        {
                            model.seas.val = param_filter.submat(0, i, model.seas.period - 1, i);
                            ft_new = StateSpace::func_ft(model.ftrans, model.fgain, model.dlag, model.seas, t + 1, theta_new, y); // ft(theta[t+1])
                        }
                    } // seasonal component

                    double lambda_new = LinkFunc::ft2mu(ft_new, model.dobs.name);
                    logp.at(i) = ObsDist::loglike(
                        y.at(t + 1), model.dobs.name, lambda_new, 
                        model.dobs.par2, true); // observation density
                    logp.at(i) += R::dnorm4(theta_new.at(0), Theta.at(0, i, t), std::sqrt(W_filter.at(i)), true);

                    weights.at(i) = std::exp(logp.at(i) - logq.at(i)); // + logw_old;
                } // loop over i, index of particles; end of propagation

                #ifdef DGTF_DO_BOUND_CHECK
                bound_check<arma::vec>(weights, "PL::forward_filter: propagation weights at t = " + std::to_string(t));
                #endif

                log_cond_marginal.at(t + 1) = std::log(arma::accu(weights) + EPS) - logN;

                if (verbose)
                {
                    Rcpp::Rcout << "\rForward Filtering: " << t + 1 << "/" << nT;
                }

            } // propagate and resample

            if (verbose)
            {
                Rcpp::Rcout << std::endl;
            }

            // psi_forward = Theta.row_as_mat(0); // (nT + 1) x N
            if (!filter_pass)
            {
                arma::mat psi = Theta.row_as_mat(0);
                output["psi_filter"] = Rcpp::wrap(arma::quantile(psi, ci_prob, 1));
                output["eff_forward"] = Rcpp::wrap(eff_forward.t());
                if (prior_W.infer)
                {
                    output["W"] = Rcpp::wrap(W_filter.t());
                }
                if (prior_seas.infer)
                {
                    output["seas"] = Rcpp::wrap(param_filter.head_rows(model.seas.period));
                }
                if (prior_rho.infer)
                {
                    output["rho"] = Rcpp::wrap(param_filter.row(model.seas.period));
                }
                if (prior_par1.infer)
                {
                    output["par1"] = Rcpp::wrap(param_filter.row(model.seas.period + 1));
                }
                if (prior_par2.infer)
                {
                    output["par2"] = Rcpp::wrap(param_filter.row(model.seas.period + 2));
                }

            }
            else
            {
                output["eff_forward2"] = Rcpp::wrap(eff_forward.t());
            }
            filter_pass = true;
            return;
        }

        void backward_filter(Model &model, const arma::vec &y, const bool &verbose = VERBOSE)
        {
            const unsigned int nT = y.n_elem - 1;
            const bool full_rank = false;
            arma::vec eff_backward(y.n_elem, arma::fill::zeros);
            if (!filter_pass)
            {
                throw std::runtime_error("SMC::PL: you need to run a forward filtering pass before running backward filtering.");
            }

            forward_filter(model, y, VERBOSE);

            Theta_backward = Theta;
            W_backward = W_filter;
            param_backward = param_filter;
            weights_backward = weights_forward;

            // mu0_filter.fill(model.dobs.par1); // N x 1

            arma::mat Sig_init(model.nP, model.nP, arma::fill::eye);
            Sig_init.diag().fill(2.);
            arma::mat Prec_init = Sig_init;
            Prec_init.diag() = 1. / Sig_init.diag();

            mu_marginal = arma::zeros<arma::cube>(model.nP, y.n_elem, N);
            Sigma_marginal = arma::zeros<arma::cube>(model.nP * model.nP, y.n_elem, N);
            Prec_marginal = Sigma_marginal;
            for (unsigned int i = 0; i < N; i++)
            {
                Sigma_marginal.slice(i).col(0) = Sig_init.as_col(); // np^{2} x 1
                Prec_marginal.slice(i).col(0) = Prec_init.as_col();
            }

            arma::mat Gt = TransFunc::init_Gt(model.nP, model.dlag, model.ftrans, model.seas.period, model.seas.in_state);

            for (unsigned int t = 1; t < y.n_elem; t++)
            {
                Rcpp::checkUserInterrupt();

                for (unsigned int i = 0; i < N; i++)
                {
                    mu_marginal.slice(i).col(t) = StateSpace::func_gt(model.ftrans, model.fgain, model.dlag, mu_marginal.slice(i).col(t - 1), y.at(t - 1), model.seas.period, model.seas.in_state);

                    LBA::func_Gt(Gt, model, mu_marginal.slice(i).col(t - 1), y.at(t - 1));
                    arma::mat Vt = arma::reshape(Sigma_marginal.slice(i).col(t - 1), model.nP, model.nP);
                    arma::mat Sig = Gt * Vt * Gt.t();
                    Sig.at(0, 0) += W_backward.at(i);
                    Sig.diag() += EPS;
                    arma::mat Prec = inverse(Sig);

                    Sigma_marginal.slice(i).col(t) = Sig.as_col();
                    Prec_marginal.slice(i).col(t) = Prec.as_col();
                }

                if (verbose)
                {
                    Rcpp::Rcout << "\rArtificial marginal: " << t << "/" << nT;
                }
            }
            Rcpp::Rcout << std::endl;

            arma::vec log_marg(N, arma::fill::zeros);
            for (unsigned int i = 0; i < N; i++)
            {
                arma::vec theta = Theta_backward.slice(nT).col(i);
                arma::mat Prec = arma::reshape(Prec_marginal.slice(i).col(nT), model.nP, model.nP);
                log_marg.at(i) = MVNorm::dmvnorm2(
                    theta, mu_marginal.slice(i).col(nT), Prec, true);
            }


            for (unsigned int t = nT - 1; t > 0; t--)
            {
                Rcpp::checkUserInterrupt();

                unsigned int t_cur = t;
                unsigned int t_next = t + 1;
                arma::mat Theta_next = Theta_backward.slice(t_next); // p x N, theta[t]

                /**
                 * @brief Resampling using conditional one-step-ahead predictive distribution as weights.
                 *
                 */

                arma::vec logq(N, arma::fill::zeros);
                arma::mat mu;                // nP x N
                arma::cube Sigma_chol; // nP x nP x N

                arma::vec tau = imp_weights_backcast(
                    mu, Sigma_chol, logq, model, t_cur, 
                    Theta_next, Theta_backward.slice(t_cur),
                    W_backward, param_backward, mu_marginal, Sigma_marginal, y, full_rank);

                tau = tau % weights;
                arma::uvec resample_idx = get_resample_index(tau);

                Theta_next = Theta_next.cols(resample_idx); // theta[t]
                Theta_backward.slice(t_next) = Theta_next;

                mu = mu.cols(resample_idx);
                Sigma_chol = Sigma_chol.slices(resample_idx);

                log_marg = log_marg.elem(resample_idx);
                logq = logq.elem(resample_idx);
                tau = tau.elem(resample_idx);

                weights_backward.row(t_next) = logq.t();
                eff_backward.at(t_cur) = effective_sample_size(tau);

                if (prior_W.infer)
                {
                    W_backward = W_backward.elem(resample_idx);
                }
                param_backward = param_backward.cols(resample_idx);

                mu_marginal = mu_marginal.slices(resample_idx);
                Sigma_marginal = Sigma_marginal.slices(resample_idx);


                // NEED TO CHANGE PROPAGATE STEP
                // arma::mat Theta_new = propagate(y.at(t_old), Wsqrt, Theta_old, model, positive_noise);
                arma::mat Theta_cur(model.nP, N, arma::fill::zeros);
                arma::vec logp(N, arma::fill::zeros);
                for (unsigned int i = 0; i < N; i++)
                {
                    if (prior_W.infer)
                    {
                        model.derr.par1 = W_backward.at(i);
                    }
                    if (prior_seas.infer)
                    {
                        model.seas.val = param_backward.submat(0, i, model.seas.period - 1, i);
                    }
                    if (prior_rho.infer)
                    {
                        model.dobs.par2 = std::exp(param_backward.at(model.seas.period, i));
                    }
                    if (lag_update)
                    {
                        model.dlag.par1 = param_backward.at(model.seas.period + 1, i);
                        model.dlag.par2 = std::exp(param_backward.at(model.seas.period + 2, i));
                    }
                    arma::vec theta_cur;
                    if (full_rank)
                    {
                        arma::vec eps = arma::randn(Theta_cur.n_rows);
                        arma::vec zt = Sigma_chol.slice(i).t() * mu.col(i) + eps; // shifted
                        theta_cur = Sigma_chol.slice(i) * zt;
                        logq.at(i) += MVNorm::dmvnorm0(zt, mu.col(i), Sigma_chol.slice(i), true);
                    }
                    else
                    {
                        theta_cur = mu.col(i);
                        double eps = R::rnorm(0., std::sqrt(W_backward.at(i)));
                        theta_cur.at(model.nP - 1) += eps;
                        logq.at(i) += R::dnorm4(eps, 0, std::sqrt(W_backward.at(i)), true);
                    }


                    Theta_cur.col(i) = theta_cur;
                    logp.at(i) += R::dnorm4(theta_cur.at(model.nP - 1), Theta_next.at(model.nP - 1, i), std::sqrt(W_backward.at(i)), true);

                    double ft_cur = StateSpace::func_ft(model.ftrans, model.fgain, model.dlag, model.seas, t_cur, theta_cur, y);
                    double lambda_cur = LinkFunc::ft2mu(ft_cur, model.dobs.name);

                    logp.at(i) += ObsDist::loglike(
                        y.at(t_cur), model.dobs.name, lambda_cur, model.dobs.par2, true); // observation density


                    logp.at(i) -= log_marg.at(i);
                    arma::vec Vprec = Prec_marginal.slice(i).col(t_cur);
                    arma::mat Vprec_cur = arma::reshape(Vprec, model.nP, model.nP);
                    arma::vec v_cur = mu_marginal.slice(i).col(t_cur);
                    log_marg.at(i) = MVNorm::dmvnorm2(theta_cur, v_cur, Vprec_cur, true);
                    logp.at(i) += log_marg.at(i);

                    // double logw_next = std::log(weights_prop_backward.at(t_next, i) + EPS);
                    weights.at(i) = std::exp(logp.at(i) - logq.at(i)); // + logw_next;
                } // loop over i, index of particles

                Theta_backward.slice(t_cur) = Theta_cur;
                // weights_prop_backward.row(t_cur) = weights.t();

                if (verbose)
                {
                    Rcpp::Rcout << "\rBackward Filtering: " << nT - t + 1 << "/" << nT;
                }

            } // propagate and resample

            if (verbose)
            {
                Rcpp::Rcout << std::endl;
            }

            arma::mat psi = Theta_backward.row_as_mat(0); // (nT + 1) x N
            output["psi_backward"] = Rcpp::wrap(arma::quantile(psi, ci_prob, 1));
            output["eff_backward"] = Rcpp::wrap(eff_backward.t());
            return;
        }

        void backward_smoother(Model &model, const arma::vec &y, const bool &verbose = VERBOSE)
        {
            const unsigned int nT = y.n_elem - 1;
            weights.ones();
            arma::uvec idx = sample(N, M, weights, false, true); // M x 1

            arma::mat theta_tmp = Theta.slice(nT);        // p x N
            arma::mat theta_tmp2 = theta_tmp.cols(idx);       // p x M
            Theta_smooth.slice(nT) = theta_tmp2;          // p x M

            W_smooth = W_filter.elem(idx);          // M x 1
            arma::vec Wsqrt = arma::sqrt(W_smooth); // M x 1

            for (unsigned int t = nT; t > 0; t--)
            {
                Rcpp::checkUserInterrupt();

                // arma::vec Wtmp0 = W_stored.col(t - 1); // N x 1
                // arma::vec Wtmp = Wtmp0.elem(idx); // M x 1
                // arma::vec Wsqrt = arma::sqrt(W_stored.col(t - 1)); // M x 1

                // arma::uvec smooth_idx = get_smooth_index(t, Wsqrt, idx);
                arma::rowvec psi_smooth_now = Theta_smooth.slice(t).row(0);                       // 1 x M
                arma::rowvec psi_filter_prev = Theta.slice(t - 1).row(0);                         // 1 x N
                arma::uvec smooth_idx = get_smooth_index(psi_smooth_now, psi_filter_prev, Wsqrt); // M x 1

                arma::mat theta_tmp0 = Theta.slice(t - 1); // p x N
                theta_tmp = theta_tmp0.cols(smooth_idx);
                Theta_smooth.slice(t - 1) = theta_tmp;

                if (verbose)
                {
                    Rcpp::Rcout << "\rSmoothing: " << nT - t + 1 << "/" << nT;
                }
            }

            if (verbose)
            {
                Rcpp::Rcout << std::endl;
            }

            arma::mat psi = Theta_smooth.row_as_mat(0);
            output["psi"] = Rcpp::wrap(arma::quantile(psi, ci_prob, 1));
        }

        void two_filter_smoother(Model &model, const arma::vec &y, const bool &verbose = VERBOSE)
        {
            const unsigned int nT = y.n_elem - 1;
            const bool full_rank = false;
            Theta_smooth = Theta;
            for (unsigned int t = 1; t < nT; t++)
            {
                Rcpp::checkUserInterrupt();

                unsigned int t_cur = t;
                unsigned int t_prev = t - 1;
                unsigned int t_next = t + 1;

                double yhat_cur = LinkFunc::mu2ft(y.at(t_cur), model.flink, 0.);

                /**
                 * No resampling here because the resampling were performed in forward and backward filterings and the resampled particles are saved.
                 * 
                 */
                arma::vec logq = arma::vectorise(weights_forward.row(t_prev) + weights_backward.row(t_next));

                arma::mat prec_tmp = Prec_marginal.col_as_mat(t_next); // nP^2 x N
                arma::mat Prec_marg = prec_tmp;                        //.cols(resample_idx); 
                arma::mat mu_marg = mu_marginal.col_as_mat(t_next);    // nP x N

                arma::vec logp(N, arma::fill::zeros);
                arma::mat Theta_cur(model.nP, N, arma::fill::zeros);
                for (unsigned int i = 0; i < N; i++)
                {
                    if (lag_update)
                    {
                        model.dlag.par1 = param_filter.at(model.seas.period + 1, i);
                        model.dlag.par2 = std::exp(param_filter.at(model.seas.period + 2, i));
                        // unsigned int nlag = model.update_dlag(param_filter.at(0, i), param_filter.at(1, i), 30, false);
                    }
                    arma::vec gtheta_prev_fwd = StateSpace::func_gt(model.ftrans, model.fgain, model.dlag, Theta.slice(t_prev).col(i), y.at(t_prev), model.seas.period, model.seas.in_state);

                    if (prior_seas.infer)
                    {
                        model.seas.val = param_backward.submat(0, i, model.seas.period - 1, i);
                    }
                    if (prior_rho.infer)
                    {
                        model.dobs.par2 = std::exp(param_backward.at(model.seas.period, i));
                    }
                    if (lag_update)
                    {
                        model.dlag.par1 = param_backward.at(model.seas.period + 1, i);
                        model.dlag.par2 = std::exp(param_backward.at(model.seas.period + 2, i));
                        // unsigned int nlag = model.update_dlag(param_backward.at(0, i), param_backward.at(1, i), 30, false);
                    }

                    arma::vec gtheta = StateSpace::func_gt(model.ftrans, model.fgain, model.dlag, Theta.slice(t_prev).col(i), y.at(t_prev), model.seas.period, model.seas.in_state);
                    double ft = StateSpace::func_ft(model.ftrans, model.fgain, model.dlag, model.seas, t_cur, gtheta, y);
                    double eta = ft;
                    double lambda = LinkFunc::ft2mu(eta, model.flink);
                    double Vt = ApproxDisturbance::func_Vt_approx(
                        lambda, model.dobs, model.flink); // (eq 3.11)

                    arma::vec theta_cur;
                    if (!full_rank)
                    {
                        theta_cur = gtheta;
                        theta_cur.at(0) += R::rnorm(0., std::sqrt(W_backward.at(i)));
                        logq.at(i) += R::dnorm4(theta_cur.at(0), gtheta.at(0), std::sqrt(W_backward.at(i)), true);
                    }
                    else
                    {
                        arma::vec Ft = LBA::func_Ft(model.ftrans, model.fgain, model.dlag, t_cur, gtheta, y, LBA_FILL_ZERO, model.seas.period, model.seas.in_state);
                        double ft_tilde = ft - arma::as_scalar(Ft.t() * gtheta);
                        arma::mat FFt_norm = Ft * Ft.t() / Vt;

                        double delta = yhat_cur - ft_tilde;

                        arma::mat Gt = TransFunc::init_Gt(model.nP, model.dlag, model.ftrans, model.seas.period, model.seas.in_state);
                        LBA::func_Gt(Gt, model, gtheta, y.at(t_cur));
                        arma::mat Wprec(model.nP, model.nP, arma::fill::zeros);
                        Wprec.at(0, 0) = 1. / W_backward.at(i);
                        arma::mat prec_part1 = Gt.t() * Wprec * Gt;
                        prec_part1.at(0, 0) += 1. / W_backward.at(i);

                        arma::mat prec = prec_part1 + FFt_norm;
                        arma::mat Rchol = arma::chol(arma::symmatu(prec));
                        arma::mat Rchol_inv = arma::inv(arma::trimatu(Rchol));
                        arma::mat Sigma = Rchol_inv * Rchol_inv.t();

                        arma::vec mu_part1 = Gt.t() * Wprec * Theta_backward.slice(t_next).col(i);
                        mu_part1.at(0) += gtheta.at(0) / W_backward.at(i);

                        arma::vec mu = Ft * (delta / Vt);
                        mu = Sigma * (mu_part1 + mu);

                        theta_cur = mu + Rchol.t() * arma::randn(model.nP);
                        logq.at(i) += MVNorm::dmvnorm2(theta_cur, mu, prec, true);
                    }

                    Theta_cur.col(i) = theta_cur;

                    logp.at(i) = R::dnorm4(theta_cur.at(0), gtheta_prev_fwd.at(0), std::sqrt(W_filter.at(i)), true);

                    gtheta = StateSpace::func_gt(model.ftrans, model.fgain, model.dlag, theta_cur, y.at(t_cur), model.seas.period, model.seas.in_state);
                    logp.at(i) += R::dnorm4(Theta_backward.at(0, i, t_next), theta_cur.at(0), std::sqrt(W_backward.at(i)), true);

                    ft = StateSpace::func_ft(model.ftrans, model.fgain, model.dlag, model.seas, t_cur, theta_cur, y);
                    lambda = LinkFunc::ft2mu(ft, model.flink);
                    logp.at(i) += ObsDist::loglike(
                        y.at(t_cur), model.dobs.name, lambda, 
                        model.dobs.par2, true);

                    arma::mat pmarg = arma::reshape(Prec_marg.col(i), model.nP, model.nP);
                    logp.at(i) -= MVNorm::dmvnorm2(Theta_backward.slice(t_next).col(i), mu_marg.col(i), pmarg, true);

                    weights.at(i) = std::exp(logp.at(i) - logq.at(i)); // + log_forward + log_backward;
                } // loop over particle i

                arma::uvec resample_idx = get_resample_index(weights);
                Theta_smooth.slice(t_cur) = Theta_cur.cols(resample_idx);

                if (verbose)
                {
                    Rcpp::Rcout << "\rSmoothing: " << t + 1 << "/" << nT;
                }
            }

            if (verbose)
            {
                Rcpp::Rcout << std::endl;
            }

            arma::mat psi = Theta_smooth.row_as_mat(0);
            output["psi"] = Rcpp::wrap(arma::quantile(psi, ci_prob, 1));
            return;
        }

        void infer(Model &model, const arma::vec &y, const bool &verbose = VERBOSE)
        {
            if (prior_W.infer && use_discount)
            {
                use_discount = false;
            }

            forward_filter(model, y, verbose); // 2,253,382 ms per 1000 particles

            if (smoothing)
            {
                // backward_smoother(model, verbose);
                backward_filter(model, y, verbose);     // 14,600,157 ms per 1000 particles
                two_filter_smoother(model, y, verbose); // 1,431,610 ms per 1000 particles
            } // opts.smoothing
        } // Particle Learning inference

    private:
        bool filter_pass = false;
        bool obs_update = false;
        bool lag_update = false;

        arma::cube mu_marginal;    // nP x (nT + 1) x N
        arma::cube Sigma_marginal; // nP^2 x (nT + 1) x N
        arma::cube Prec_marginal;  // nP^2 x (nT + 1) x N

        arma::mat weights_forward;  // (nT + 1) x N
        arma::mat weights_backward; // (nT + 1) x N
        arma::cube Theta_backward; // p x N x (nT + 1)

        arma::vec aw_forward; // N x 1, shape of IG
        arma::vec bw_forward; // N x 1, scale of IG (i.e. rate of corresponding Gamma)

        arma::vec W_smooth; // N x 1
        arma::vec W_backward; // N x 1
        arma::vec W_filter; // N x 1

        arma::mat param_filter, param_backward, param_smooth;

        arma::mat aseas_forward; // period x N, mean of normal
        arma::cube bseas_forward; // period x period x N, variance of normal
        Prior prior_seas;

        Prior prior_rho;
        Prior prior_par1;
        Prior prior_par2;
        Prior prior_W;

        unsigned int max_iter = 10;
    };
}

#endif