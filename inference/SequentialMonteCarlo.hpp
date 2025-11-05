#ifndef _SEQUENTIALMONTECARLO_H
#define _SEQUENTIALMONTECARLO_H

#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <algorithm>
#include <RcppArmadillo.h>
#ifdef DGTF_USE_OPENMP
#include <omp.h>
#endif
#include "../core/Model.hpp"
#include "LinearBayes.hpp"
#include "ImportanceDensity.hpp"

// Optional: enable with -DDGTF_TIMING_SMC
#ifdef DGTF_TIMING_SMC
#include <chrono>
#define T_NOW() std::chrono::high_resolution_clock::now()
#define T_US(dt) std::chrono::duration_cast<std::chrono::microseconds>(dt).count()
#endif

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

        static double effective_sample_size(const arma::vec &weights)
        {
            const double wsum = arma::accu(weights);
            const double w2sum = arma::dot(weights, weights);
            const double ess = (wsum * wsum) / (w2sum + EPS);

#ifdef DGTF_DO_BOUND_CHECK
            bound_check(ess, "effective_sample_size: ess (nom = " + std::to_string(nom) + ", denom = " + std::to_string(denom) + ")");
#endif

            return ess;
        }

        static inline void gather_cols(
            arma::mat &out,
            const arma::mat &in,
            const arma::uvec &idx)
        {
            const arma::uword N = idx.n_elem;
            out.set_size(in.n_rows, N);
            for (arma::uword j = 0; j < N; ++j)
                out.col(j) = in.col(idx[j]);
        }

        static inline void gather_vec(
            arma::vec &out,
            const arma::vec &in,
            const arma::uvec &idx)
        {
            const arma::uword N = idx.n_elem;
            out.set_size(N);
            for (arma::uword j = 0; j < N; ++j)
                out[j] = in[idx[j]];
        }

        // static arma::uvec get_resample_index(const arma::vec &weights)
        // {
        //     unsigned int N = weights.n_elem;
        //     double wsum = arma::accu(weights);
        //     arma::uvec indices = arma::regspace<arma::uvec>(0, 1, N - 1);
        //     if (wsum > EPS)
        //     {
        //         arma::vec w = weights / wsum;
        //         indices = sample(N, N, w, true, true);
        //     }

        //     return indices;
        // }

        static inline arma::uvec get_resample_index(const arma::vec &weights)
        {
            const unsigned int N = weights.n_elem;
            arma::uvec indices = arma::regspace<arma::uvec>(0, 1, N - 1);

            double wsum = arma::accu(weights);
            if (wsum <= EPS || N == 0)
                return indices;

            // cumulative (unnormalized) weights
            arma::vec cumw = arma::cumsum(weights);
            const double step = wsum / static_cast<double>(N);

            // one uniform in [0, step)
            thread_local std::mt19937_64 rng{0x9E3779B97F4A7C15ULL};
            std::uniform_real_distribution<double> unif(0.0, step);
            double u = unif(rng);

            arma::uvec out(N);
            unsigned int i = 0;
            double threshold = u;

            // two-pointer scan
            for (unsigned int j = 0; j < N; ++j)
            {
                while (i + 1 < N && cumw[i] < threshold)
                    ++i;
                out[j] = i;
                threshold += step;
            }
            // guard numeric edge-case: last threshold may exceed cumw[N-1] by tiny eps
            out[N - 1] = std::min<unsigned int>(out[N - 1], N - 1);

            return out;
        }


        static double auxiliary_filter0(
            arma::mat &Theta_mean, // p x (nT + 1)
            arma::cube &Theta,     // p x N x (nT + 1)
            arma::vec &z_mean,     // (nT + 1)
            Model &model,
            const arma::vec &y, // (nT + 1) x 1
            const unsigned int &N = 1000,
            const bool &initial_resample_all = false,
            const bool &final_resample_by_weights = false,
            const bool &use_discount = false,
            const double &discount_factor = 0.95)
        {
#ifdef DGTF_TIMING_SMC
            auto smc_start = T_NOW();
            // Per-iteration accumulators
            long long us_qforecast = 0;
            long long us_resample = 0;
            long long us_propagate = 0;
            long long us_commit = 0;

            // Optional: sample a few steps (0, 50, 100, 150) like your logs
            auto should_sample_step = [&](unsigned int t)
            {
                return (t == 0 || t == 50 || t == 100 || t == 150);
            };
#endif

            std::map<std::string, SysEq::Evolution> sys_list = SysEq::sys_list;
            const unsigned int nT = y.n_elem - 1;
            arma::vec weights(N, arma::fill::ones);
            double log_cond_marginal = 0.0;

            // Optional dynamic error variance via discount
            arma::cube Wt;
            if (use_discount)
            {
                LBA::LinearBayes lba(use_discount, discount_factor);
                lba.filter(model, y);
                Wt = lba.get_Wt(model, y, discount_factor);
            }

            // Cholesky holder of process noise
            arma::mat Wt_chol(model.nP, model.nP, arma::fill::zeros);
            if (!use_discount)
            {
                if (model.derr.full_rank)
                {
                    Wt_chol = arma::chol(model.derr.var);
                }
                else if (model.derr.par1 > EPS)
                {
                    Wt_chol.at(0, 0) = std::sqrt(model.derr.par1);
                }
            }

            // Ensure output sizes
            if (Theta_mean.n_rows != model.nP)
            {
                Theta_mean.set_size(model.nP, y.n_elem);
            }
            if (Theta.n_rows != model.nP)
            {
                Theta.set_size(model.nP, N, y.n_elem);
            }

            // Initialize state particles at t=0
            if (sys_list[model.fsys] == SysEq::Evolution::identity)
                Theta.slice(0) = arma::randu<arma::mat>(model.nP, N);
            else
                Theta.slice(0) = arma::randn<arma::mat>(model.nP, N);

            // No zero-inflation by default
            arma::mat z = arma::ones<arma::mat>(N, y.n_elem);

            // Reused buffers across time steps
            arma::mat Theta_new(model.nP, N, arma::fill::zeros);
            arma::vec logq(N, arma::fill::zeros);
            arma::vec tau(N, arma::fill::zeros);

            arma::mat eps_mat;
            arma::vec eps1, u;
            if (model.derr.full_rank)
                eps_mat.set_size(model.nP, N);
            else
                eps1.set_size(N);

            if (model.zero.inflated)
                u.set_size(N);

            // Materialization helpers
            arma::mat resample_buf(model.nP, N, arma::fill::none);
            arma::vec zbuf(N, arma::fill::none);
            const arma::uvec idx_id = arma::regspace<arma::uvec>(0, 1, N - 1);
            auto compose = [](const arma::uvec &a, const arma::uvec &b)
            { return a.elem(b); };

            std::vector<arma::uvec> res_aux(nT + 1, idx_id);
            std::vector<arma::uvec> final_idx_slice(y.n_elem, idx_id);
            arma::uvec anc_seed = idx_id;

// Persistent OpenMP team over the entire time loop
#ifdef DGTF_USE_OPENMP
#pragma omp parallel
#endif
            {
                for (unsigned int t = 0; t < nT; ++t)
                {
                    // Shared "fast path" constants container
                    bool fast_ok = false;
                    QForecastFastConsts C;

                    // Per-time ancestry mapping (into Theta.slice(t)/z.col(t))
                    arma::uvec anc;

                    // Per-time constants to share with parallel regions
                    unsigned int nelem = 0;
                    const double *Fphi = nullptr;
                    const double *yptr = nullptr;
                    double seas_off = 0.0;
                    double s_cur = 0.0; // current std of univariate noise (if applicable)

// Single-thread prep section
#ifdef DGTF_USE_OPENMP
#pragma omp single
#endif
                    {
                        anc = anc_seed; // start from previously built ancestry

                        // Dynamic W (discount)
                        if (use_discount)
                        {
                            if (model.derr.full_rank)
                            {
                                Wt_chol = arma::chol(Wt.slice(t + 1));
                                model.derr.var = Wt.slice(t + 1);
                            }
                            else if (model.derr.par1 > EPS)
                            {
                                Wt_chol.at(0, 0) = std::sqrt(Wt.at(0, 0, t + 1));
                                model.derr.par1 = Wt.at(0, 0, t + 1);
                                model.derr.var.at(0, 0) = Wt.at(0, 0, t + 1);
                            }
                        }

                        // Decide fast path once per t
                        fast_ok = (!model.derr.full_rank) &&
                                  (model.ftrans == "sliding") &&
                                  (model.fsys == "shift") &&
                                  (!model.seas.in_state);
                        if (fast_ok)
                            C = make_qf_consts(model, t + 1, y);

                        // Precompute constants used in propagation and qforecast loops
                        nelem = std::min(t + 1, model.dlag.nL);
                        Fphi = model.dlag.Fphi.memptr();
                        yptr = y.memptr();

                        seas_off = 0.0;
                        if (!model.seas.X.is_empty() && !model.seas.val.is_empty())
                            seas_off = arma::dot(model.seas.X.col(t + 1), model.seas.val);

                        // Draw process noise for this step
                        if (model.derr.full_rank)
                        {
                            eps_mat = Wt_chol.t() * arma::randn(model.nP, N);
                        }
                        else
                        {
                            s_cur = Wt_chol.at(0, 0);
                            if (s_cur > EPS)
                            {
                                eps1.randn();
                                eps1 *= s_cur;
                            }
                            else
                            {
                                eps1.zeros();
                            }
                        }

                        if (model.zero.inflated)
                            u.randu();
                    } // omp single

#ifdef DGTF_TIMING_SMC
                    std::chrono::high_resolution_clock::time_point t_qf_beg, t_qf_end;
#ifdef DGTF_USE_OPENMP
#pragma omp single
#endif
                    {
                        t_qf_beg = T_NOW();
                    }
#endif

                    // qforecast: compute logq[i] = log q(y[t+1] | theta[t], z[t+1]=1, gamma)
                    if (fast_ok)
                    {
// Parallel over particles; no nested OpenMP in the kernel
#ifdef DGTF_USE_OPENMP
#pragma omp for schedule(static)
#endif
                        for (unsigned int i = 0; i < N; ++i)
                        {
                            const double *th = Theta.slice(t).colptr(i);

                            // ft(t+1) from theta[t] via shifted/sliding structure
                            double ft = C.seas_off;
                            for (unsigned int j = 0; j < C.nelem; ++j)
                            {
                                const double psi_lag = (j == 0) ? th[0] : th[j - 1];
                                double hpsi;
                                if (psi_lag > 20.0)
                                    hpsi = psi_lag;
                                else if (psi_lag < -20.0)
                                    hpsi = std::exp(psi_lag);
                                else
                                    hpsi = std::log1p(std::exp(psi_lag));
                                const double ylag = C.yptr[(t + 1) - 1 - j];
                                ft += C.Fphi[j] * (hpsi * ylag);
                            }

                            const double mu = C.link_identity ? ft : LinkFunc::ft2mu(ft, C.flink);
                            double Vt = ApproxDisturbance::func_Vt_approx(mu, C.dobs, C.flink);
                            Vt = std::abs(Vt) + EPS;

                            const double diff = (C.yhat_new - ft);
                            logq[i] = -0.5 * (LOG2PI + std::log(Vt) + (diff * diff) / Vt);
                        }
                    }
                    else
                    {
// Fallback generic path executed once (no nested OpenMP)
#ifdef DGTF_USE_OPENMP
#pragma omp single
#endif
                        {
                            logq = qforecast0(model, t + 1, Theta.slice(t), y);
                        }
                    }

#ifdef DGTF_TIMING_SMC
#ifdef DGTF_USE_OPENMP
#pragma omp barrier
#pragma omp single
#endif
                    {
                        t_qf_end = T_NOW();
                        auto us = T_US(t_qf_end - t_qf_beg);
                        us_qforecast += us;
                        if (should_sample_step(t))
                        {
                            std::cout << "    [SMC] Time step " << t << "/" << nT
                                      << " - qforecast took " << us << " microseconds.\n";
                        }
                    }
#endif

                    // Resampling prep timing
#ifdef DGTF_TIMING_SMC
                    std::chrono::high_resolution_clock::time_point t_rs_beg, t_rs_end;
#ifdef DGTF_USE_OPENMP
#pragma omp single
#endif
                    {
                        t_rs_beg = T_NOW();
                    }
#endif

// Turn logq into tau (unnormalized importance weights)
#ifdef DGTF_USE_OPENMP
#pragma omp single
#endif
                    {
                        tau = arma::exp(logq - logq.max());

                        // Zero-inflation adjustment (if used)
                        if (model.zero.inflated)
                        {
                            double val = model.zero.intercept;
                            if (!model.zero.X.is_empty())
                                val += arma::dot(model.zero.X.col(t + 1), model.zero.beta);

                            arma::vec zval = z.col(t) * model.zero.coef + val; // N x 1
                            arma::vec prob = logistic(zval);                   // p(z[t+1] = 1 | z[t], gamma)

                            tau %= prob;
                            if (std::abs(y.at(t + 1)) < EPS)
                                tau += 1. - prob;

                            logq = arma::log(arma::abs(tau) + EPS);
                        }

                        if (t > 0)
                        {
                            // Resample using w[t] * q(y[t+1] | theta[t], ...)
                            tau %= weights;
                            arma::uvec resample_idx = get_resample_index(tau);

                            if (initial_resample_all)
                            {
                                res_aux[t] = resample_idx;
                                anc = anc.elem(resample_idx); // ancestry mapping only; slice(t) remains in original order
                            }
                            else
                            {
                                // Physically permute slice(t) and z(t) to match the resampling
                                gather_cols(resample_buf, Theta.slice(t), resample_idx);
                                Theta.slice(t).swap(resample_buf);

                                if (model.zero.inflated)
                                {
                                    gather_vec(zbuf, z.col(t), resample_idx);
                                    z.col(t) = zbuf;
                                }
                            }

                            // Keep q-density aligned with resampled ancestry
                            logq = logq.elem(resample_idx);
                        }
                        else
                        {
                            // First step: anc starts as identity
                            anc = anc_seed;
                        }
                    } // omp single (resampling prep)

#ifdef DGTF_TIMING_SMC
#ifdef DGTF_USE_OPENMP
#pragma omp single
#endif
                    {
                        t_rs_end = T_NOW();
                        auto us = T_US(t_rs_end - t_rs_beg);
                        us_resample += us;
                    }
#endif

                    // Propagation timing
#ifdef DGTF_TIMING_SMC
                    std::chrono::high_resolution_clock::time_point t_pr_beg, t_pr_end;
#ifdef DGTF_USE_OPENMP
#pragma omp single
#endif
                    {
                        t_pr_beg = T_NOW();
                    }
#endif

// Propagation: theta[t] -> theta[t+1], evaluate weights
#ifdef DGTF_USE_OPENMP
#pragma omp for schedule(static)
#endif
                    for (unsigned int i = 0; i < N; ++i)
                    {
                        const arma::uword parent = anc[i];

                        const double *cur = Theta.slice(t).colptr(parent);
                        double *out = Theta_new.colptr(i);

                        // Shift system for sliding TF (fast path)
                        if (fast_ok)
                        {
                            out[0] = cur[0];
                            const unsigned int nr = model.nP - 1; // season not in-state here
                            for (unsigned int r = 1; r <= nr; ++r)
                                out[r] = cur[r - 1];
                        }
                        else
                        {
                            // Generic propagation
                            arma::vec gtheta = SysEq::func_gt(
                                model.fsys, model.fgain, model.dlag,
                                Theta.slice(t).col(parent), y.at(t),
                                model.seas.period, model.seas.in_state);
                            for (unsigned int r = 0; r < model.nP; ++r)
                                out[r] = gtheta[r];
                        }

                        // Add process noise
                        if (!model.derr.full_rank)
                        {
                            // univariate noise to psi only
                            if (eps1.n_elem > 0)
                                out[0] += eps1[i];
                        }
                        else
                        {
                            const double *e = eps_mat.colptr(i);
                            for (unsigned int r = 0; r < model.nP; ++r)
                                out[r] += e[r];
                        }

                        // Fast ft for likelihood at t+1 (avoid generic func_ft)
                        double ft = 0.0;
                        if (fast_ok)
                        {
                            ft = seas_off;
                            for (unsigned int j = 0; j < nelem; ++j)
                            {
                                const double psi_lag = out[j];
                                double hpsi;
                                if (psi_lag > 20.0)
                                    hpsi = psi_lag;
                                else if (psi_lag < -20.0)
                                    hpsi = std::exp(psi_lag);
                                else
                                    hpsi = std::log1p(std::exp(psi_lag));

                                const double ylag = yptr[t - j];
                                ft += Fphi[j] * (hpsi * ylag);
                            }
                        }
                        else
                        {
                            ft = TransFunc::func_ft(
                                model.ftrans, model.fgain, model.dlag, model.seas,
                                t + 1, arma::vec(out, model.nP, false, true), y);
                        }

                        const double lambda = LinkFunc::ft2mu(ft, model.flink);

                        // Zero-inflated z[t+1] (unused if not zero-inflated)
                        if (model.zero.inflated)
                        {
                            double zval = model.zero.intercept;
                            if (!model.zero.X.is_empty() && !model.zero.beta.is_empty())
                                zval += arma::dot(model.zero.X.col(t + 1), model.zero.beta);
                            const double p1 = 1.0 / (1.0 + std::exp(-zval));
                            z.at(i, t + 1) = (u.at(i) < p1) ? 1.0 : 0.0;
                        }

                        // Likelihood p(y[t+1] | theta[t+1], z[t+1])
                        double val;
                        if (model.zero.inflated && z.at(i, t + 1) < EPS)
                            val = (std::abs(y.at(t + 1)) < EPS) ? 0.0 : -INFINITY;
                        else
                            val = ObsDist::loglike(y.at(t + 1), model.dobs.name, lambda, model.dobs.par2, true);

                        // Incremental weight (log-domain): log p - log q
                        weights.at(i) = val - logq.at(i);
                    } // omp for over particles

#ifdef DGTF_TIMING_SMC
#ifdef DGTF_USE_OPENMP
#pragma omp barrier
#pragma omp single
#endif
                    {
                        t_pr_end = T_NOW();
                        auto us = T_US(t_pr_end - t_pr_beg);
                        us_propagate += us;
                        if (should_sample_step(t))
                        {
                            std::cout << "    [SMC] Time step " << t << " - propagation took "
                                      << us << " microseconds.\n";
                        }
                    }
#endif

                    // Commit timing
#ifdef DGTF_TIMING_SMC
                    std::chrono::high_resolution_clock::time_point t_cm_beg, t_cm_end;
#ifdef DGTF_USE_OPENMP
#pragma omp single
#endif
                    {
                        t_cm_beg = T_NOW();
                    }
#endif

// Commit new slice and resampling-by-weights (optional)
#ifdef DGTF_USE_OPENMP
#pragma omp single
#endif
                    {
                        Theta.slice(t + 1) = Theta_new;

                        // Normalize weights (stable)
                        const double wmax = weights.max();
                        weights = arma::exp(weights - wmax);

                        if (final_resample_by_weights || t >= nT - 1)
                        {
                            const double eff = effective_sample_size(weights);
                            if (eff < 0.95 * N)
                            {
                                arma::uvec resample_idx = get_resample_index(weights);
                                final_idx_slice[t + 1] = resample_idx;  // mapping for slice t+1
                                anc_seed = anc_seed.elem(resample_idx); // ancestry mapping for next t
                                weights.ones();
                            }
                            else
                            {
                                final_idx_slice[t + 1] = idx_id; // identity
                            }
                        }
                        else
                        {
                            final_idx_slice[t + 1] = idx_id;
                        }
                    } // omp single commit

#ifdef DGTF_TIMING_SMC
#ifdef DGTF_USE_OPENMP
#pragma omp single
#endif
                    {
                        t_cm_end = T_NOW();
                        us_commit += T_US(t_cm_end - t_cm_beg);
                    }
#endif
                } // for t
            } // omp parallel (persistent team)

#ifdef DGTF_TIMING_SMC
            auto smc_end = T_NOW();
            auto smc_total = T_US(smc_end - smc_start);
            std::cout << "  [VB] SMC took " << smc_total << " microseconds.\n"
                      << "    [SMC] Total qforecast:   " << us_qforecast << " us\n"
                      << "    [SMC] Total resampling:  " << us_resample << " us\n"
                      << "    [SMC] Total propagation: " << us_propagate << " us\n"
                      << "    [SMC] Total commit:      " << us_commit << " us\n";
#endif

            // Materialize means across particles (histogram + gemv/axpy)
            const double invN = 1.0 / static_cast<double>(N);

            // final slice: weighted mean
            arma::vec final_weights = weights / arma::accu(weights);
            Theta_mean.col(nT) = Theta.slice(nT) * final_weights;
            z_mean.at(nT) = model.zero.inflated ? arma::dot(z.col(nT), final_weights) : 1.0;

            // Reusable buffers
            arma::Col<uint32_t> counts(N, arma::fill::zeros);
            arma::vec wk(N, arma::fill::zeros);
            arma::uvec suffix = idx_id;

            for (int k = static_cast<int>(nT); k >= 0; --k)
            {
                // suffix_k = res_aux[k] ∘ suffix (if we did auxiliary resampling)
                arma::uvec suffix_k = suffix;
                if (initial_resample_all && k >= 1)
                {
                    const arma::uvec &raux = res_aux[static_cast<unsigned int>(k)];
                    if (raux.n_elem == N)
                    {
                        for (arma::uword i = 0; i < N; ++i)
                            suffix_k[i] = raux[suffix[i]];
                    }
                }

                const arma::uvec &fk = final_idx_slice[k];

                // Identity fast-path
                bool trivial = true;
                for (arma::uword i = 0; i < N; ++i)
                {
                    if (fk[suffix_k[i]] != i)
                    {
                        trivial = false;
                        break;
                    }
                }
                if (trivial)
                {
                    Theta_mean.col(k) = arma::mean(Theta.slice(k), 1);
                    z_mean.at(k) = model.zero.inflated ? arma::mean(z.col(k)) : 1.0;
                    suffix = std::move(suffix_k);
                    continue;
                }

                // Histogram of mapping_k = fk ∘ suffix_k
                counts.zeros();
                for (arma::uword i = 0; i < N; ++i)
                    counts[fk[suffix_k[i]]]++;

                arma::uvec nz = arma::find(counts > 0u);
                if (nz.n_elem <= N / 2)
                {
                    // Sparse AXPY over unique columns only
                    Theta_mean.col(k).zeros();
                    for (arma::uword jj = 0; jj < nz.n_elem; ++jj)
                    {
                        const arma::uword j = nz[jj];
                        const double w = static_cast<double>(counts[j]) * invN;
                        Theta_mean.col(k) += w * Theta.slice(k).col(j);
                    }
                    if (model.zero.inflated)
                    {
                        double zm = 0.0;
                        for (arma::uword jj = 0; jj < nz.n_elem; ++jj)
                        {
                            const arma::uword j = nz[jj];
                            const double w = static_cast<double>(counts[j]) * invN;
                            zm += w * z.at(j, k);
                        }
                        z_mean.at(k) = zm;
                    }
                    else
                    {
                        z_mean.at(k) = 1.0;
                    }
                }
                else
                {
                    // Dense BLAS gemv
                    for (arma::uword j = 0; j < N; ++j)
                        wk[j] = static_cast<double>(counts[j]) * invN;
                    Theta_mean.col(k) = Theta.slice(k) * wk;
                    z_mean.at(k) = model.zero.inflated ? arma::dot(z.col(k), wk) : 1.0;
                }

                suffix = std::move(suffix_k);
            }

            return log_cond_marginal;
        }


        arma::vec weights, lambda, tau; // N x 1
        arma::cube Theta;               // p x N x (nT + B)
        arma::cube Theta_smooth;        // p x M x (nT + B)

        // For zero inflated model.
        arma::mat z;        // N x (nT + B)
        arma::mat z_smooth; // N x (nT + B)

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

}

#endif