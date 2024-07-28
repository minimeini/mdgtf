#ifndef _SEQUENTIALMONTECARLO_H
#define _SEQUENTIALMONTECARLO_H

#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <algorithm>
// #include <chrono>
#include <RcppArmadillo.h>
#include "Model.hpp"
#include "ImportanceDensity.hpp"

// [[Rcpp::depends(RcppArmadillo)]]

namespace SMC
{
    class SequentialMonteCarlo
    {
    public:
        SequentialMonteCarlo(
            const Model &model,
            const arma::vec &y_in)
        {
            dim = model.dim;
            y.set_size(dim.nT + 1);
            y.zeros();
            y.tail(y_in.n_elem) = y_in;

            // log_cond_marginal.set_size(dim.nT + 1);
            // log_cond_marginal.zeros();

            par = {
                model.dobs.par1, // mu0
                model.dobs.par2, // rho
                model.transfer.dlag.par1,
                model.transfer.dlag.par2};

            Wt.set_size(dim.nP);
            Wt.zeros();
            Wt.at(0) = model.derr.par1;

            return;
        }

        SequentialMonteCarlo()
        {
            dim.init_default();
            y.set_size(dim.nT + 1);
            y.zeros();

            // log_cond_marginal.set_size(dim.nT + 1);
            // log_cond_marginal.zeros();
        }

        static Rcpp::List default_settings()
        {
            Rcpp::List opts;
            opts["num_particle"] = 1000;
            opts["num_smooth"] = 1000;
            opts["num_backward"] = 1;
            opts["num_step_ahead_forecast"] = 0;
            opts["use_discount"] = false;
            opts["use_custom"] = false;
            opts["custom_discount_factor"] = 0.95;
            opts["do_smoothing"] = false;
            opts["do_backward"] = false;

            // Rcpp::List mu0_opts;
            // mu0_opts["infer"] = false;
            // mu0_opts["init"] = 0.;
            // mu0_opts["prior_name"] = "gamma";
            // mu0_opts["prior_param"] = Rcpp::NumericVector::create(1., 1.);
            // opts["mu0"] = mu0_opts;

            // Rcpp::List W_opts;
            // W_opts["infer"] = false;
            // W_opts["init"] = 0.01;
            // W_opts["prior_name"] = "invgamma";
            // W_opts["prior_param"] = Rcpp::NumericVector::create(1., 1.);
            // opts["W"] = W_opts;

            return opts;
        }

        void infer(const Model &model) { return; }

        void init(const Rcpp::List &smc_settings)
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
            if (settings.containsElementNamed("num_backward"))
            {
                B = Rcpp::as<unsigned int>(settings["num_backward"]);
            }

            nforecast = 0;
            if (settings.containsElementNamed("num_step_ahead_forecast"))
            {
                nforecast = Rcpp::as<unsigned int>(settings["num_step_ahead_forecast"]);
            }

            // psi_smooth.set_size(dim.nT + B, M);
            // psi_smooth.zeros();

            use_discount = false;
            if (settings.containsElementNamed("use_discount"))
            {
                use_discount = Rcpp::as<bool>(settings["use_discount"]);
            }

            use_custom = false;
            if (settings.containsElementNamed("use_custom"))
            {
                use_custom = Rcpp::as<bool>(settings["use_custom"]);
            }

            custom_discount_factor = 0.95;
            if (settings.containsElementNamed("custom_discount_factor"))
            {
                custom_discount_factor = Rcpp::as<double>(settings["custom_discount_factor"]);
            }

            smoothing = false;
            if (settings.containsElementNamed("do_smoothing"))
            {
                smoothing = Rcpp::as<bool>(settings["do_smoothing"]);
            }

            backward = false;
            if (settings.containsElementNamed("do_backward"))
            {
                backward = Rcpp::as<bool>(settings["do_backward"]);
            }

            Theta.set_size(dim.nP, N, dim.nT + B);
            Theta.zeros();
            Theta_smooth.set_size(dim.nP, M, dim.nT + B);
            Theta_smooth.zeros();

            // psi_forward.set_size(dim.nT + B, N);
            // psi_forward.zeros();
            // psi_backward = psi_forward;
            // psi_smooth.set_size(dim.nT + B, M);
            // psi_smooth.zeros();

            std::string par_name = "mu0";
            prior_mu0.init("gamma", 1., 1.);
            prior_mu0.init_param(false, 0.);
            if (settings.containsElementNamed("mu0"))
            {

                init_param_def(settings, par_name, prior_mu0);
            }

            mu0_filter.set_size(N);
            mu0_filter.fill(prior_mu0.val);

            par_name = "W";
            prior_W.init("invgamma", 1., 1.);
            prior_W.init_param(false, 0.01);
            if (settings.containsElementNamed("W"))
            {

                init_param_def(settings, par_name, prior_W);
            }

            W_filter.set_size(N);
            W_filter.fill(prior_W.val);
            // Wt.fill(prior_W.val);

            return;
        }

        static void init_param_def(
            Rcpp::List &opts,
            std::string &par_name,
            Prior &prior)
        {
            std::map<std::string, AVAIL::Dist> dist_list = AVAIL::dist_list;

            prior.init("invgamma", 1., 1.);
            if (opts.containsElementNamed(par_name.c_str()))
            {
                Rcpp::List par_opts = opts[par_name];
                // init_param(infer, init_val, prior, par_opts);
                init_prior(prior, par_opts);
            }
            // prior.init_param(infer, init_val);
            return;
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
                bound_check<arma::vec>(par_init, "draw_param_init:: par_init");
            }

            return par_init;
        }

        void update_np(const unsigned int &np)
        {
            dim.nP = np;
            Theta.reset();
            Theta.set_size(np, N, dim.nT + B);
            Theta.zeros();

            Theta.reset();
            Theta.set_size(np, M, dim.nT + B);
            Theta.zeros();
        }

        arma::mat get_psi_filter()
        {
            arma::mat psi_tmp = Theta.row_as_mat(0);              // (nT + B) x N
            arma::mat psi_filter = psi_tmp.tail_rows(dim.nT + 1); // (nT + 1) x N

            return psi_filter; // (nT + 1) x N
        }

        arma::mat get_psi_smooth()
        {
            arma::mat psi_tmp = Theta_smooth.row_as_mat(0); // (nT + B) x M
            arma::mat psi_smooth = psi_tmp.tail_rows(dim.nT + 1);
            return psi_smooth;
            // return psi_smooth.tail_rows(dim.nT + 1); // (nT + 1) x M
        }

        static double discount_W(
            const arma::mat &Theta_now,
            const double &custom_discount_factor = 0.95,
            const bool &use_custom = false,
            const double &default_discount_factor = 0.99)
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
            if (use_custom)
            {
                W *= 1. / custom_discount_factor - 1.;
            }
            else
            {
                W *= 1. / default_discount_factor - 1.;
            }

            bound_check(W, "SequentialMonteCarlo::discount_W");
            return W;
        }

        static double effective_sample_size(const arma::vec &weights)
        {
            arma::vec w2 = arma::square(weights);
            double denom = arma::accu(w2);
            double nom = arma::accu(weights);
            nom = std::pow(nom, 2.);
            double ess = nom / denom;

            bound_check(ess, "effective_sample_size: ess (nom = " + std::to_string(nom) + ", denom = " + std::to_string(denom) + ")");

            return ess;
        }

        static double log_conditional_marginal(const arma::vec &unnormalised_weights)
        {
            unsigned int N = unnormalised_weights.n_elem; // number of particles
            double N_ = static_cast<double>(N);
            double wsum = arma::accu(arma::abs(unnormalised_weights));
            double log_cond_marg = std::log(wsum + EPS) - std::log(N_);

            return log_cond_marg;
        }

        static double marginal_likelihood(const arma::vec log_cond_marg, const bool &return_log = true)
        {
            double pmarg = arma::accu(log_cond_marg);
            if (!return_log)
            {
                pmarg = std::exp(std::min(pmarg, UPBND));
            }

            return pmarg;
        }

        /**
         * @brief At time t, importance weights of forward filtering particles z[t-1] = (theta[t-1], W[t-1], mu0[t-1]), based on conditional predictive distribution y[t] | z[t-1], y[1:t-1].
         *
         * @param model
         * @param t_new Index of the time that is being predicted, t. The following inputs come from time t_old = t-1.
         * @param Theta p x N, { theta[t-1, i], i = 1, ..., N }, samples of latent states at t-1
         * @param W N x 1, { W[t-1, i], i = 1, ..., N }, samples of latent variance at t-1
         * @param mu0 N x 1, { mu0[t-1, i], i = 1, ..., N }, samples of baseline at t-1
         * @param mt_old p x N, m[t-1]. Assume no static parameters involved in ft.
         * @param yhat y[t] after transformation.
         * @return arma::vec
         */
        static arma::vec imp_weights_forecast(
            arma::mat &mu,          // p x N
            arma::cube &Prec,       // p x p x N
            arma::cube &Sigma_chol, // p x p x N
            arma::vec &logq,        // N x 1
            arma::uvec &updated,
            const Model &model,
            const unsigned int &t_new,  // current time t. The following inputs come from time t-1.
            const arma::mat &Theta_old, // p x N, {theta[t-1]}
            const arma::vec &W_old,     // N x 1, {W[t-1]} samples of latent variance
            const arma::vec &mu0_old,   // N x 1, samples of baseline
            const arma::vec &y,
            const arma::vec &yhat // y[t]
        )
        {
            unsigned int N = Theta_old.n_cols;
            unsigned int nP = Theta_old.n_rows;
            double y_old = y.at(t_new - 1);
            arma::mat Ieps(model.dim.nP, model.dim.nP, arma::fill::eye);
            Ieps.diag().fill(EPS);

            mu.set_size(nP, N);
            mu.zeros();

            Prec = arma::zeros<arma::cube>(nP, nP, N);
            Sigma_chol = Prec;

            updated.set_size(N);
            updated.zeros();

            for (unsigned int i = 0; i < N; i++)
            {
                arma::vec gtheta_old_i = StateSpace::func_gt(model.transfer, Theta_old.col(i), y_old); // gt(theta[t-1, i])
                double ft_gtheta = StateSpace::func_ft(model.transfer, t_new, gtheta_old_i, y);        // ft( gt(theta[t-1,i]) )
                arma::vec Ft_gtheta = LBA::func_Ft(model.transfer, t_new, gtheta_old_i, y);            // Ft evaluated at a[t_new]
                double ft_tilde = ft_gtheta - arma::as_scalar(Ft_gtheta.t() * gtheta_old_i);  // (eq 3.8)

                double eta = mu0_old.at(i) + ft_gtheta;
                double lambda = LinkFunc::ft2mu(eta, model.flink.name, 0.); // (eq 3.10)

                if (t_new == 1 && lambda < EPS)
                {
                    lambda = 1.;
                }

                double Vt = ApproxDisturbance::func_Vt_approx(
                    lambda, model.dobs, model.flink.name); // (eq 3.11)

                double delta = yhat.at(t_new) - mu0_old.at(i) - ft_tilde; // (eq 3.16)
                double delta2 = delta * delta;

                arma::mat FFt_norm = Ft_gtheta * Ft_gtheta.t() / Vt;
                double FFt_det = arma::det(FFt_norm);
                if (FFt_det < EPS8)
                {
                    Ft_gtheta.zeros();
                    FFt_norm.zeros();

                    mu.col(i) = gtheta_old_i;
                    Prec.at(0, 0, i) = 1. / W_old.at(i);
                    Sigma_chol.at(0, 0, i) = std::sqrt(W_old.at(i));

                    logq.at(i) = R::dnorm4(yhat.at(t_new), eta, std::sqrt(Vt), true);

                    updated.at(i) = 0;

                } // One-step-ahead predictive density
                else
                {
                    arma::mat Prec_i = FFt_norm + Ieps;  // nP x nP, function of mu0[i, t]
                    Prec_i.at(0, 0) += 1. / W_old.at(i); // (eq 3.21)

                    Prec_i = arma::symmatu(Prec_i);
                    Prec.slice(i) = Prec_i;
                    double ldetPrec = arma::log_det_sympd(Prec_i);

                    arma::mat Rchol, Sigma_i;
                    try
                    {
                        Sigma_i = inverse(Rchol, Prec_i);
                    }
                    catch (const std::exception &e)
                    {
                        double det_prec = arma::det(Prec_i);
                        std::cout << "\n Determinant of Prec_i is: " << det_prec << "; ";
                        std::cout << "Determinant of Ft * Ft.t() / Vt is: " << FFt_det << std::endl;
                        std::cerr << e.what() << '\n';
                    }

                    Sigma_i = arma::symmatu(Sigma_i);
                    Sigma_chol.slice(i) = Rchol;

                    arma::vec y_scaled = Ft_gtheta * (delta / Vt); // nP x 1
                    y_scaled.at(0) += gtheta_old_i.at(0) / W_old.at(i);
                    // arma::vec g_scaled = precW * gtheta_old_i;
                    // arma::vec mu_i = Sigma_i * (y_scaled + g_scaled); // (eq 3.20)
                    arma::vec mu_i = Sigma_i * y_scaled;

                    mu.col(i) = mu_i;

                    double ldetV = std::log(std::abs(Vt) + EPS);
                    double ldetW = std::log(std::abs(W_old.at(i)) + EPS);

                    double loglik = LOG2PI + ldetV + ldetW + ldetPrec; // (eq 3.24)
                    loglik += delta2 / Vt;
                    loglik += std::pow(gtheta_old_i.at(0), 2.) / W_old.at(i);
                    loglik -= arma::as_scalar(mu_i.t() * Prec_i * mu_i); // minus (-)
                    loglik *= -0.5;                                      // (eq 3.24 - 3.25)

                    try
                    {
                        bound_check(loglik, "imp_weights_forecast: loglik");
                    }
                    catch (const std::exception &e)
                    {
                        std::cout << "\n ldetPrec = " << ldetPrec;
                        std::cout << ", ldetV = " << ldetV;
                        std::cout << ", ldetW = " << ldetW;
                        std::cout << ", delta = " << delta;
                        std::cout << ", Vt = " << Vt;
                        std::cout << ", gWg = " << std::pow(gtheta_old_i.at(0), 2.) / W_old.at(i);
                        std::cout << ", uSu = " << arma::as_scalar(mu_i.t() * Prec_i * mu_i);
                        throw std::runtime_error(e.what());
                    }

                    logq.at(i) += loglik;

                    updated.at(i) = 1;
                } // one-step-ahead predictive density
            }

            double logq_max = logq.max();
            logq.for_each([&logq_max](arma::vec::elem_type &val)
                          { val -= logq_max; });

            // double logm = arma::mean(logq);
            // if (logq.max() > 100)
            // {
            //     double diff = logq.max() - 100;
            //     logm = std::max(logm, diff);
            // }
            // logq.for_each([&logm](arma::vec::elem_type &val){val -= logm;});
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
            arma::mat &mu,          // p x N, mean of the posterior of theta[t_cur] | y[t_cur:nT], theta[t_next], W
            arma::cube &Prec,       // p x p x N, precision of the posterior of theta[t_cur] | y[t_cur:nT], theta[t_next], W
            arma::cube &Sigma_chol, // p x p x N, right cholesky of the variance of the posterior of theta[t_cur] | y[t_cur:nT], theta[t_next], W
            arma::vec &logq,        // N x 1
            arma::uvec &updated,    // N x 1
            const Model &model,
            const unsigned int &t_cur,   // current time "t". The following inputs come from time t+1. t_next = t + 1; t_prev = t - 1
            const arma::mat &Theta_next, // p x N, {theta[t+1]}
            const arma::vec &W_filter,   // N x 1, {inv(W[T])} samples of latent variance
            const arma::vec &mu0_filter, // N x 1, {mu0[T]} samples of baseline
            const arma::cube &vt,        // nP x (nT + 1) x N, v[t]
            const arma::cube &Vt,        // nP*nP x (nT + 1) x N, V[t]
            const arma::vec &y,
            const arma::vec &yhat // y[t]
        )
        {

            std::map<std::string, AVAIL::Transfer> trans_list = AVAIL::trans_list;

            unsigned int N = Theta_next.n_cols;
            unsigned int nP = Theta_next.n_rows;
            unsigned int t_next = t_cur + 1;

            arma::mat Ieps(model.dim.nP, model.dim.nP, arma::fill::eye);
            Ieps.diag().fill(EPS);

            mu.set_size(nP, N);
            mu.zeros();

            Prec = arma::zeros<arma::cube>(nP, nP, N);
            Sigma_chol = Prec;

            updated.set_size(N);
            updated.zeros();

            for (unsigned int i = 0; i < N; i++)
            {
                arma::vec v_cur = vt.slice(i).col(t_cur);
                arma::vec Vtmp = Vt.slice(i).col(t_cur);
                arma::mat V_cur = arma::reshape(Vtmp, model.dim.nP, model.dim.nP);

                arma::vec v_next = vt.slice(i).col(t_next);
                Vtmp = Vt.slice(i).col(t_next);
                arma::mat V_next = arma::reshape(Vtmp, model.dim.nP, model.dim.nP);
                arma::mat Vprec_next = inverse(V_next);

                arma::mat G_next = LBA::func_Gt(model, v_cur, y.at(t_cur));

                arma::vec r_cur(model.dim.nP, arma::fill::zeros);
                arma::mat K_cur(model.dim.nP, model.dim.nP, arma::fill::zeros); // evolution matrix
                arma::mat U_cur = K_cur;                                        // mean of conditional backward evolution
                arma::mat Uprec_cur = K_cur;
                arma::mat Urchol_cur = K_cur;
                double ldetU = 0.;

                if (trans_list[model.transfer.name] == AVAIL::sliding)
                {
                    for (unsigned int i = 0; i < model.dim.nP - 1; i++)
                    {
                        K_cur.at(i, i + 1) = 1.;
                    }

                    K_cur.at(model.dim.nP - 1, model.dim.nP - 1) = 1.;
                    U_cur.at(model.dim.nP - 1, model.dim.nP - 1) = W_filter.at(i);
                    Uprec_cur.at(model.dim.nP - 1, model.dim.nP - 1) = 1. / W_filter.at(i);
                    Urchol_cur.at(model.dim.nP - 1, model.dim.nP - 1) = std::sqrt(W_filter.at(i));
                    ldetU = std::log(W_filter.at(i));
                }
                else
                {
                    K_cur = V_cur * G_next.t() * Vprec_next;
                    U_cur = V_cur - V_cur * G_next.t() * Vprec_next * G_next * V_cur;
                    U_cur = arma::symmatu(U_cur);
                    Uprec_cur = inverse(Urchol_cur, U_cur);
                    ldetU = arma::log_det_sympd(U_cur);
                }
                r_cur = v_cur - K_cur * v_next;

                arma::vec u_cur = K_cur * Theta_next.col(i) + r_cur;
                arma::vec F_cur = LBA::func_Ft(model.transfer, t_cur, u_cur, y);

                double ft_ut = StateSpace::func_ft(model.transfer, t_cur, u_cur, y);
                double eta = mu0_filter.at(i) + ft_ut;
                double lambda = LinkFunc::ft2mu(eta, model.flink.name, 0.); // (eq 3.58)
                double Vtilde = ApproxDisturbance::func_Vt_approx(
                    lambda, model.dobs, model.flink.name); // (eq 3.59)
                Vtilde = std::abs(Vtilde) + EPS;
                double ldetV = std::log(Vtilde);

                double delta = yhat.at(t_cur) - eta;
                delta += arma::as_scalar(F_cur.t() * u_cur);
                double delta2 = delta * delta;

                arma::mat FFt_norm = arma::symmatu(F_cur * F_cur.t() / Vtilde);
                double FFt_det = arma::det(FFt_norm);

                if (FFt_det < EPS8)
                {
                    // No information from data, degenerates to the backward evolution
                    F_cur.zeros();
                    FFt_norm.zeros();

                    mu.col(i) = u_cur;
                    Prec.slice(i) = Uprec_cur;
                    Sigma_chol.slice(i) = Urchol_cur;

                    logq.at(i) = R::dnorm4(yhat.at(t_cur), eta, std::sqrt(Vtilde), true);
                    updated.at(i) = 0;
                } // one-step backcasting
                else
                {
                    Prec.slice(i) = arma::symmatu(FFt_norm + Uprec_cur + Ieps);
                    double ldetPrec;
                    ldetPrec = arma::log_det_sympd(Prec.slice(i));

                    arma::mat Rchol;
                    arma::mat Sig = inverse(Rchol, Prec.slice(i));
                    Sigma_chol.slice(i) = Rchol;

                    arma::vec mu_tmp = F_cur * (delta / Vtilde) + Uprec_cur * u_cur;
                    mu.col(i) = Sig * mu_tmp;

                    double logq_pred = LOG2PI + ldetV + ldetU + ldetPrec; // (eq 3.63)
                    logq_pred += delta2 / Vtilde;
                    logq_pred += arma::as_scalar(u_cur.t() * Uprec_cur * u_cur);
                    logq_pred -= arma::as_scalar(mu.col(i).t() * Prec.slice(i) * mu.col(i));
                    logq_pred *= -0.5;

                    logq.at(i) += logq_pred;
                    updated.at(i) = 1;
                }
            } // loop over particles

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

                weights.elem(arma::find(weights > UPBND)).fill(UPBND);
                weights = arma::exp(weights);
                bound_check<arma::vec>(weights, "SMC::SequentialMonteCarlo::get_smooth_index");

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
            const std::string &loss_func = "quadratic",
            const unsigned int &k = 1)
        {
            arma::cube th_filter = Theta.tail_slices(dim.nT + 1); // p x N x (nT + 1)
            Rcpp::List out2 = StateSpace::forecast_error(th_filter, y, model, loss_func, k);

            return out2;
        }

        void forecast_error(double &err, double &cov, double &width, const Model &model, const std::string &loss_func = "quadratic")
        {
            arma::cube theta_tmp = Theta.tail_slices(dim.nT + 1); // p x N x (nT + 1)
            StateSpace::forecast_error(err, cov, width, theta_tmp, y, model, loss_func);
            return;
        }

        Rcpp::List fitted_error(const Model &model, const std::string &loss_func = "quadratic")
        {
            Rcpp::List out3;

            arma::cube theta_tmp = Theta.tail_slices(dim.nT + 1); // p x N x (nT + 1)
            Rcpp::List out_filter = StateSpace::fitted_error(theta_tmp, y, model, loss_func);
            out3["filter"] = out_filter;

            if (smoothing)
            {
                arma::cube theta_tmp2 = Theta_smooth.tail_slices(dim.nT + 1); // p x N x (nT + 1)
                Rcpp::List out_smooth = StateSpace::fitted_error(theta_tmp2, y, model, loss_func);
                out3["smooth"] = out_smooth;
            }

            return out3;
        }

        void fitted_error(double &err, const Model &model, const std::string &loss_func = "quadratic")
        {

            arma::cube theta_tmp;
            if (smoothing)
            {
                theta_tmp = Theta_smooth.tail_slices(dim.nT + 1); // p x N x (nT + 1)
            }
            else
            {
                theta_tmp = Theta.tail_slices(dim.nT + 1); // p x N x (nT + 1)
            }

            StateSpace::fitted_error(err, theta_tmp, y, model, loss_func);
            return;
        }

        Dim dim;

        arma::vec y;                    // (nT + 1) x 1
        arma::vec weights, lambda, tau; // N x 1
        // arma::vec log_cond_marginal;

        arma::cube Theta;        // p x N x (nT + B)
        arma::cube Theta_smooth; // p x M x (nT + B)
        // arma::mat psi_smooth; // (nT + B) x M

        arma::vec par; // m x 1
        // arma::mat param; // m x N
        arma::vec Wt; // p x 1
        // arma::mat Wt_stored; // p x N

        unsigned int N = 1000;
        unsigned int M = 500;
        unsigned int B = 1;
        unsigned int nforecast = 0;

        // arma::mat psi_forward;  // (nT + B) x N
        // arma::mat psi_backward; // (nT + B) x N
        // arma::mat psi_smooth;   // (nT + B) x M

        Prior prior_W;
        arma::vec W_filter; // N x 1

        Prior prior_mu0;
        arma::vec mu0_filter;

        bool use_discount = false;
        bool use_custom = false;
        double custom_discount_factor = 0.95;
        const double default_discount_factor = 0.99;

        bool smoothing = true;
        bool backward = true;

        Rcpp::List output;
        arma::vec ci_prob = {0.025, 0.5, 0.975};

    }; // class Sequential Monte Carlo

    class MCS : public SequentialMonteCarlo
    {
    public:
        MCS(
            const Model &dgtf_model,
            const arma::vec &y_in) : SequentialMonteCarlo(dgtf_model, y_in) {}

        Rcpp::List get_output(const bool &summarize = true)
        {
            // arma::vec ci_prob = {0.025, 0.5, 0.975};
            // Rcpp::List output;
            // // arma::mat psi_filter = get_psi_filter();
            // // arma::mat psi_smooth = get_psi_smooth();

            // if (DEBUG)
            // {
            //     output["Theta"] = Rcpp::wrap(Theta);
            // }

            // arma::mat psi1 = arma::quantile(psi_forward.tail_rows(dim.nT + 1), ci_prob, 1); // (nT + 1) x 3
            // output["psi_filter"] = Rcpp::wrap(psi1);

            // arma::mat psi2 = arma::quantile(psi_smooth.tail_rows(dim.nT + 1), ci_prob, 1); // (nT + 1) x 3
            // output["psi"] = Rcpp::wrap(psi2);

            // output["log_marginal_likelihood"] = marginal_likelihood(log_cond_marginal, true);

            return output;
        }

        void init(const Rcpp::List &opts_in)
        {
            Rcpp::List opts = opts_in;
            SequentialMonteCarlo::init(opts);

            M = N;

            // Wt.set_size(dim.nT + 1);
            // Wt.fill(prior_W.val);

            Theta_smooth.clear();
            Theta_smooth = Theta;

            // psi_smooth.reset();
            // psi_smooth.set_size(dim.nT + B, N);
            // psi_smooth.zeros();
        }

        static Rcpp::List default_settings()
        {
            Rcpp::List opts = SequentialMonteCarlo::default_settings();
            opts["num_backward"] = 10;
            return opts;
        }

        Rcpp::List forecast(const Model &model)
        {
            arma::vec Wtmp(N, arma::fill::zeros);
            Wtmp.fill(prior_W.val);
            Rcpp::List out = StateSpace::forecast(y, Theta, Wtmp, model, nforecast);
            return out;
        }

        arma::mat optimal_discount_factor(
            const Model &model,
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
            use_custom = true;

            for (unsigned int i = 0; i < nelem; i++)
            {
                Rcpp::checkUserInterrupt();

                custom_discount_factor = grid.at(i);
                stats.at(i, 0) = custom_discount_factor;

                infer(model);
                arma::cube theta_tmp = Theta.tail_slices(model.dim.nT + 1);

                double err_forecast = 0.;
                double cov_forecast = 0.;
                double width_forecast = 0.;
                StateSpace::forecast_error(err_forecast, cov_forecast, width_forecast, theta_tmp, y, model, loss, false);

                // arma::cube theta_tmp2 = Theta_smooth.tail_slices(model.dim.nT + 1);
                // double err_fit = 0.;
                // StateSpace::fitted_error(err_fit, theta_tmp2, y, model, loss, false);

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

                prior_W.val = grid.at(i);
                stats.at(i, 0) = prior_W.val;

                infer(model);
                arma::cube theta_tmp = Theta.tail_slices(model.dim.nT + 1);

                double err_forecast = 0.;
                double cov_forecast = 0.;
                double width_forecast = 0.;
                StateSpace::forecast_error(err_forecast, cov_forecast, width_forecast, theta_tmp, y, model, loss, false);
                stats.at(i, 1) = err_forecast;

                // arma::cube theta_tmp2 = Theta_smooth.tail_slices(model.dim.nT + 1);
                // double err_fit = 0.;
                // StateSpace::fitted_error(err_fit, theta_tmp2, y, model, loss, false);
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
            const Model &model,
            const unsigned int &from,
            const unsigned int &to,
            const unsigned int &delta = 1,
            const std::string &loss = "quadratic",
            const bool &verbose = VERBOSE)
        {
            arma::uvec grid = arma::regspace<arma::uvec>(from, delta, to);
            unsigned int nelem = grid.n_elem;
            arma::mat stats(nelem, 4, arma::fill::zeros);

            for (unsigned int i = 0; i < nelem; i++)
            {
                Rcpp::checkUserInterrupt();

                B = grid.at(i);
                stats.at(i, 0) = B;

                Theta.clear();
                Theta.set_size(model.dim.nP, N, model.dim.nT + B);
                Theta.zeros();

                Theta_smooth.clear();
                Theta_smooth = Theta;

                infer(model);
                arma::cube theta_tmp = Theta.tail_slices(model.dim.nT + 1);

                double err_forecast = 0.;
                double cov_forecast = 0.;
                double width_forecast = 0.;
                StateSpace::forecast_error(err_forecast, cov_forecast, width_forecast, theta_tmp, y, model, loss, false);
                stats.at(i, 1) = err_forecast;

                // arma::cube theta_tmp2 = Theta_smooth.tail_slices(model.dim.nT + 1);
                // double err_fit = 0.;
                // StateSpace::fitted_error(err_fit, theta_tmp2, y, model, loss, false);
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

        // arma::mat optimal_discount_factor(
        //     const Model &model,
        //     const double &from,
        //     const double &to,
        //     const double &delta = 0.01,
        //     const std::string &loss = "quadratic"
        // )
        // {
        //     arma::vec grid = arma::regspace<arma::vec>(from, delta, to);
        //     unsigned int nelem = grid.n_elem;
        //     arma::mat stats(nelem, 3, arma::fill::zeros);

        //     use_discount = true;
        //     use_custom = true;

        //     arma::cube theta_tmp = Theta.tail_slices(model.dim.nT + 1);

        //     for (unsigned int i = 0; i < nelem; i ++)
        //     {
        //         custom_discount_factor = grid.at(i);
        //         stats.at(i, 0) = custom_discount_factor;

        //         infer(model);

        //         double err_forecast = 0.;
        //         StateSpace::forecast_error(err_forecast, theta_tmp, y, model, loss);
        //         stats.at(i, 1) = err_forecast;

        //         double err_fit = 0.;
        //         StateSpace::fitted_error(err_fit, theta_tmp, y, model, loss);
        //         stats.at(i, 2) = err_fit;
        //     }

        //     return stats;
        // }

        // arma::mat optimal_W(
        //     const Model &model,
        //     const arma::vec &grid,
        //     const std::string &loss = "quadratic"
        // )
        // {
        //     unsigned int nelem = grid.n_elem;
        //     arma::mat stats(nelem, 3, arma::fill::zeros);

        //     use_discount = false;

        //     arma::cube theta_tmp = Theta.tail_slices(model.dim.nT + 1);

        //     for (unsigned int i = 0; i < nelem; i++)
        //     {
        //         W = grid.at(i);
        //         stats.at(i, 0) = W;

        //         infer(model);

        //         double err_forecast = 0.;
        //         StateSpace::forecast_error(err_forecast, theta_tmp, y, model, loss);
        //         stats.at(i, 1) = err_forecast;

        //         double err_fit = 0.;
        //         StateSpace::fitted_error(err_fit, theta_tmp, y, model, loss);
        //         stats.at(i, 2) = err_fit;
        //     }

        //     return stats;
        // }

        void infer(const Model &model, const bool &verbose = VERBOSE)
        {
            arma::mat psi_forward(dim.nT + B, N);
            psi_forward.zeros();
            arma::mat psi_smooth = psi_forward;
            arma::vec log_cond_marginal(dim.nT + 1, arma::fill::zeros);

            for (unsigned int t = 0; t < dim.nT; t++)
            {
                Rcpp::checkUserInterrupt();

                if (use_discount)
                { // Use discount factor if W is not given
                    bool use_custom_val = (use_custom && t > B) ? true : false;
                    Wt.at(0) = SequentialMonteCarlo::discount_W(
                        Theta.slice(t + B - 1),
                        custom_discount_factor,
                        use_custom_val,
                        default_discount_factor);
                }

                double Wsqrt = std::sqrt(Wt.at(0) + EPS);
                arma::mat Theta_new(dim.nP, N, arma::fill::zeros);
                bool positive_noise = (t < Theta.n_rows) ? true : false;
                for (unsigned int i = 0; i < N; i++)
                {
                    arma::vec theta_new = StateSpace::func_gt(model.transfer, Theta.slice(t + B - 1).col(i), y.at(t));
                    double eps = R::rnorm(0., Wsqrt);
                    if (positive_noise)
                    {
                        eps = std::abs(eps);
                    }

                    theta_new.at(0) += eps;

                    double ft = StateSpace::func_ft(model.transfer, t + 1, theta_new, y);
                    double lambda = LinkFunc::ft2mu(ft, model.flink.name, par.at(0));

                    weights.at(i) = ObsDist::loglike(y.at(t + 1), model.dobs.name, lambda, par.at(1), false);
                    Theta_new.col(i) = theta_new;
                }

                log_cond_marginal.at(t + 1) = log_conditional_marginal(weights);
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
                    Rcpp::Rcout << "\rProgress: " << t + 1 << "/" << dim.nT;
                }

            } // loop over time

            if (verbose)
            {
                Rcpp::Rcout << std::endl;
            }

            output["psi_filter"] = Rcpp::wrap(arma::quantile(psi_forward.tail_rows(dim.nT + 1), ci_prob, 1));
            output["psi"] = Rcpp::wrap(arma::quantile(psi_smooth.tail_rows(dim.nT + 1), ci_prob, 1));
            output["log_marginal_likelihood"] = marginal_likelihood(log_cond_marginal, true);
        }
    };

    class FFBS : public SequentialMonteCarlo
    {
    private:
        arma::vec eff_forward; // (nT + 1) x 1

    public:
        FFBS(
            const Model &dgtf_model,
            const arma::vec &y_in) : SequentialMonteCarlo(dgtf_model, y_in) {}

        void init(const Rcpp::List &opts_in)
        {
            Rcpp::List opts = opts_in;
            SequentialMonteCarlo::init(opts);

            eff_forward.set_size(dim.nT + 1);
            eff_forward.zeros();
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

        Rcpp::List forecast(const Model &model)
        {
            arma::vec Wtmp(N, arma::fill::zeros);
            Wtmp.fill(Wt.at(0));

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
            const Model &model,
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
            use_custom = true;

            for (unsigned int i = 0; i < nelem; i++)
            {
                Rcpp::checkUserInterrupt();

                custom_discount_factor = grid.at(i);
                stats.at(i, 0) = custom_discount_factor;

                infer(model, false);
                arma::cube theta_tmp = Theta.tail_slices(model.dim.nT + 1);

                double err_forecast = 0.;
                double cov_forecast = 0.;
                double width_forecast = 0.;
                StateSpace::forecast_error(err_forecast, cov_forecast, width_forecast, theta_tmp, y, model, loss, false);
                stats.at(i, 1) = err_forecast;
                stats.at(i, 2) = cov_forecast;
                stats.at(i, 3) = width_forecast;

                // double err_fit = 0.;
                // if (smoothing)
                // {
                //     arma::cube theta_tmp2 = Theta_smooth.tail_slices(model.dim.nT + 1);
                //     StateSpace::fitted_error(err_fit, theta_tmp2, y, model, loss, false);
                // }
                // else
                // {
                //     StateSpace::fitted_error(err_fit, theta_tmp, y, model, loss, false);
                // }

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

                Wt.at(0) = grid.at(i);
                stats.at(i, 0) = Wt.at(0);

                infer(model, false);
                arma::cube theta_tmp = Theta.tail_slices(model.dim.nT + 1);

                double err_forecast = 0.;
                double cov_forecast = 0.;
                double width_forecast = 0.;
                StateSpace::forecast_error(err_forecast, cov_forecast, width_forecast, theta_tmp, y, model, loss, false);
                stats.at(i, 1) = err_forecast;

                // double err_fit = 0.;
                // if (smoothing)
                // {
                //     arma::cube theta_tmp2 = Theta_smooth.tail_slices(model.dim.nT + 1);
                //     StateSpace::fitted_error(err_fit, theta_tmp2, y, model, loss, false);
                // }
                // else
                // {
                //     StateSpace::fitted_error(err_fit, theta_tmp, y, model, loss, false);
                // }

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

        void forward_filter(const Model &model, const bool &verbose = VERBOSE)
        {
            const bool full_rank = false;
            arma::vec eff_forward(dim.nT + 1, arma::fill::zeros);
            arma::vec log_cond_marginal = eff_forward;

            for (unsigned int t = 0; t < dim.nT; t++)
            {
                Rcpp::checkUserInterrupt();

                arma::vec logq(N, arma::fill::zeros);
                arma::mat loc(dim.nP, N, arma::fill::zeros);
                arma::cube prec_chol_inv; // nP x nP x N
                if (full_rank)
                {
                    prec_chol_inv = arma::zeros<arma::cube>(dim.nP, dim.nP, N); // nP x nP x N
                }

                arma::vec tau = qforecast(
                    loc, prec_chol_inv, logq,     // sufficient statistics
                    model, t + 1, Theta.slice(t), // theta needs to be resampled
                    Wt, par, y);

                tau = weights % tau;
                arma::uvec resample_idx = get_resample_index(tau);

                Theta.slice(t) = Theta.slice(t).cols(resample_idx);
                loc = loc.cols(resample_idx);
                if (full_rank)
                {
                    prec_chol_inv = prec_chol_inv.slices(resample_idx);
                }

                logq = logq.elem(resample_idx);
                tau = tau.elem(resample_idx);

                if (use_discount)
                { // Use discount factor if W is not given
                    bool use_custom_val = (use_custom && t > 1) ? true : false;
                    Wt.at(0) = SequentialMonteCarlo::discount_W(
                        Theta.slice(t),
                        custom_discount_factor,
                        use_custom_val,
                        default_discount_factor);
                }

                // Propagate
                arma::vec logp(N, arma::fill::zeros);
                for (unsigned int i = 0; i < N; i++)
                {
                    arma::vec zt, theta_new;
                    if (full_rank)
                    {
                        arma::vec eps = arma::randn(dim.nP);
                        zt = prec_chol_inv.slice(i).t() * loc.col(i) + eps; // shifted
                        theta_new = prec_chol_inv.slice(i) * zt;            // scaled

                        logq.at(i) += MVNorm::dmvnorm0(zt, loc.col(i), prec_chol_inv.slice(i), true);
                        double logp_tmp = R::dnorm4(theta_new.at(0), Theta.at(0, i, t), std::sqrt(Wt.at(0)), true);
                        logp.at(i) = logp_tmp;
                    }
                    else
                    {
                        theta_new = loc.col(i);
                        theta_new.at(0) += R::rnorm(0, std::sqrt(Wt.at(0)));

                        double logp_tmp = R::dnorm4(theta_new.at(0), Theta.at(0, i, t), std::sqrt(Wt.at(0)), true);
                        logq.at(i) += logp_tmp; // sample from evolution distribution
                        logp.at(i) = logp_tmp;
                    }

                    Theta.slice(t + 1).col(i) = theta_new;

                    double ft = StateSpace::func_ft(model.transfer, t + 1, theta_new, y);
                    double lambda = LinkFunc::ft2mu(ft, model.flink.name, par.at(0));
                    logp.at(i) += ObsDist::loglike(y.at(t + 1), model.dobs.name, lambda, model.dobs.par2, true);
                    weights.at(i) = std::exp(logp.at(i) - logq.at(i));
                }

                eff_forward.at(t + 1) = effective_sample_size(weights);
                log_cond_marginal.at(t + 1) = log_conditional_marginal(weights);

                if (eff_forward.at(t + 1) < 0.95 * N || t >= dim.nT - 1)
                {
                    resample_idx = get_resample_index(weights);
                    Theta.slice(t + 1) = Theta.slice(t + 1).cols(resample_idx);
                    weights.ones();
                }

                if (verbose)
                {
                    Rcpp::Rcout << "\rForwawrd Filtering: " << t + 1 << "/" << dim.nT;
                }
            }

            if (verbose)
            {
                Rcpp::Rcout << std::endl;
            }

            arma::mat psi = Theta.row_as_mat(0); // (nT + 1) x N
            output["psi_filter"] = Rcpp::wrap(arma::quantile(psi, ci_prob, 1));
            output["eff_forward"] = Rcpp::wrap(eff_forward.t());
            output["log_marginal_likelihood"] = marginal_likelihood(log_cond_marginal, true);

            return;
        }

        void smoother(const Model &model, const bool &verbose = VERBOSE)
        {
            arma::uvec idx = sample(N, M, weights, true, true); // M x 1
            arma::mat theta_last = Theta.slice(dim.nT);         // p x N
            arma::mat theta_sub = theta_last.cols(idx);         // p x M

            Theta_smooth.slice(dim.nT) = theta_sub;
            // psi_smooth.row(dim.nT) = theta_sub.row(0);

            for (unsigned int t = dim.nT; t > 0; t--)
            {
                Rcpp::checkUserInterrupt();

                double Wsd = std::sqrt(Wt.at(0));
                arma::vec Wsqrt(M);
                Wsqrt.fill(Wsd);

                arma::rowvec psi_smooth_now = Theta_smooth.slice(t).row(0);                       // 1 x M
                arma::rowvec psi_filter_prev = Theta.slice(t - 1).row(0);                         // 1 x N
                arma::uvec smooth_idx = get_smooth_index(psi_smooth_now, psi_filter_prev, Wsqrt); // M x 1

                arma::mat theta_next = Theta.slice(t - 1);
                theta_next = theta_next.cols(smooth_idx); // p x M

                Theta_smooth.slice(t - 1) = theta_next;
                // psi_smooth.row(t - 1) = theta_next.row(0); // 1 x M

                if (verbose)
                {
                    Rcpp::Rcout << "\rSmoothing: " << dim.nT - t + 1 << "/" << dim.nT;
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

        void infer(const Model &model, const bool &verbose = VERBOSE)
        {
            forward_filter(model, verbose);

            if (smoothing)
            {
                smoother(model, verbose);
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
        // arma::vec eff_forward;  // (nT + 1) x 1
        // arma::vec eff_backward; // (nT + 1) x 1

        arma::mat weights_forward;  // (nT + 1) x N
        arma::mat weights_backward; // (nT + 1) x N
        // arma::mat weights_prop_forward;  // (nT + 1) x N
        // arma::mat weights_prop_backward; // (nT + 1) x N
        arma::cube Theta_backward; // p x N x (nT + 1)
    public:
        TFS(
            const Model &dgtf_model,
            const arma::vec &y_in) : SequentialMonteCarlo(dgtf_model, y_in) {}

        void init(const Rcpp::List &opts_in)
        {
            Rcpp::List opts = opts_in;
            SequentialMonteCarlo::init(opts);

            // prior_W.infer = false;
            // prior_mu0.infer = false;
            // B = 1;

            // Wt.set_size(dim.nT + 1);
            // Wt.fill(prior_W.val);

            weights_forward.set_size(dim.nT + 1, N);
            weights_forward.zeros();
            weights_backward = weights_forward;
            // weights_prop_forward = weights_forward;
            // weights_prop_backward = weights_forward;

            Theta_backward = Theta;
            Theta_smooth.clear();
            Theta_smooth = Theta;

            // eff_forward.set_size(dim.nT + 1);
            // eff_forward.zeros();
            // eff_backward = eff_forward;

            return;
        }

        static Rcpp::List default_settings()
        {
            Rcpp::List opts = SequentialMonteCarlo::default_settings();
            return opts;
        }

        Rcpp::List get_output(const bool &summarize = true)
        {
            return output;
        }

        Rcpp::List forecast(const Model &model)
        {
            arma::vec Wtmp(N, arma::fill::zeros);
            Wtmp.fill(prior_W.val);

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
            const Model &model,
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
            use_custom = true;

            for (unsigned int i = 0; i < nelem; i++)
            {
                Rcpp::checkUserInterrupt();

                custom_discount_factor = grid.at(i);
                stats.at(i, 0) = custom_discount_factor;

                infer(model, false);
                arma::cube theta_tmp = Theta.tail_slices(model.dim.nT + 1);

                double err_forecast = 0.;
                double cov_forecast = 0.;
                double width_forecast = 0.;
                StateSpace::forecast_error(err_forecast, cov_forecast, width_forecast, theta_tmp, y, model, loss, false);
                stats.at(i, 1) = err_forecast;
                stats.at(i, 2) = cov_forecast;
                stats.at(i, 3) = width_forecast;

                // double err_fit = 0.;
                // if (smoothing)
                // {
                //     arma::cube theta_tmp2 = Theta_smooth.tail_slices(model.dim.nT + 1);
                //     StateSpace::fitted_error(err_fit, theta_tmp2, y, model, loss, false);
                // }
                // else
                // {
                //     StateSpace::fitted_error(err_fit, theta_tmp, y, model, loss, false);
                // }

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

                prior_W.val = grid.at(i);
                stats.at(i, 0) = prior_W.val;

                infer(model, false);
                arma::cube theta_tmp = Theta.tail_slices(model.dim.nT + 1);

                double err_forecast = 0.;
                double cov_forecast = 0.;
                double width_forecast = 0.;
                StateSpace::forecast_error(err_forecast, cov_forecast, width_forecast, theta_tmp, y, model, loss, false);
                stats.at(i, 1) = err_forecast;

                // double err_fit = 0.;
                // if (smoothing)
                // {
                //     arma::cube theta_tmp2 = Theta_smooth.tail_slices(model.dim.nT + 1);
                //     StateSpace::fitted_error(err_fit, theta_tmp2, y, model, loss, false);
                // }
                // else
                // {
                //     StateSpace::fitted_error(err_fit, theta_tmp, y, model, loss, false);
                // }

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

        void forward_filter(const Model &model, const bool &verbose = VERBOSE)
        {
            const bool full_rank = false;
            arma::vec eff_forward(dim.nT + 1, arma::fill::zeros);
            arma::vec log_cond_marginal = eff_forward;

            for (unsigned int t = 0; t < dim.nT; t++)
            {
                Rcpp::checkUserInterrupt();

                arma::vec logq(N, arma::fill::zeros);
                arma::mat loc(dim.nP, N, arma::fill::zeros);
                arma::cube prec_chol_inv; // nP x nP x N
                if (full_rank)
                {
                    prec_chol_inv = arma::zeros<arma::cube>(dim.nP, dim.nP, N); // nP x nP x N
                }

                arma::vec tau = qforecast(
                    loc, prec_chol_inv, logq,     // sufficient statistics
                    model, t + 1, Theta.slice(t), // theta needs to be resampled
                    Wt, par, y);

                tau = tau % weights;
                weights_forward.row(t) = tau.t();

                arma::uvec resample_idx = get_resample_index(tau);

                for (unsigned int k = 0; k <= t; k++)
                {
                    Theta.slice(k) = Theta.slice(k).cols(resample_idx);
                    arma::vec wtmp = arma::vectorise(weights_forward.row(k));
                    weights_forward.row(k) = wtmp.elem(resample_idx).t();
                }

                loc = loc.cols(resample_idx);
                if (full_rank)
                {
                    prec_chol_inv = prec_chol_inv.slices(resample_idx);
                }
                logq = logq.elem(resample_idx);

                eff_forward.at(t + 1) = effective_sample_size(tau);

                if (use_discount)
                { // Use discount factor if W is not given
                    bool use_custom_val = (use_custom && t > 1) ? true : false;
                    Wt.at(0) = SequentialMonteCarlo::discount_W(
                        Theta.slice(t),
                        custom_discount_factor,
                        use_custom_val,
                        default_discount_factor);
                }

                // Propagate
                arma::vec logp(N, arma::fill::zeros);
                for (unsigned int i = 0; i < N; i++)
                {
                    arma::vec theta_new;
                    if (full_rank)
                    {
                        arma::vec eps = arma::randn(dim.nP);
                        arma::vec zt = prec_chol_inv.slice(i).t() * loc.col(i) + eps; // shifted
                        theta_new = prec_chol_inv.slice(i) * zt;            // scaled

                        logq.at(i) += MVNorm::dmvnorm0(zt, loc.col(i), prec_chol_inv.slice(i), true);
                    }
                    else
                    {
                        theta_new = loc.col(i);
                        double eps = R::rnorm(0., std::sqrt(Wt.at(0)));
                        theta_new.at(0) += eps;
                        logq.at(i) += R::dnorm4(eps, 0., std::sqrt(Wt.at(0)), true); // sample from evolution distribution
                    }

                    Theta.slice(t + 1).col(i) = theta_new;

                    logp.at(i) = R::dnorm4(theta_new.at(0), Theta.at(0, i, t), std::sqrt(Wt.at(0)), true);
                    double ft = StateSpace::func_ft(model.transfer, t + 1, theta_new, y);
                    double lambda = LinkFunc::ft2mu(ft, model.flink.name, par.at(0));
                    logp.at(i) += ObsDist::loglike(y.at(t + 1), model.dobs.name, lambda, model.dobs.par2, true);
                    weights.at(i) = std::exp(logp.at(i) - logq.at(i));
                }

                log_cond_marginal.at(t + 1) = log_conditional_marginal(weights);

                // if (eff_forward.at(t + 1) < 0.95 * N || t >= dim.nT - 1)
                // {
                //     arma::uvec resample_idx = get_resample_index(weights);
                //     Theta.slice(t + 1) = Theta.slice(t + 1).cols(resample_idx);
                //     weights.ones();
                // }
                // else
                // {
                // Theta.slice(t_new) = Theta_new;
                // }

                if (verbose)
                {
                    Rcpp::Rcout << "\rForwawrd Filtering: " << t + 1 << "/" << dim.nT;
                }
            }

            if (verbose)
            {
                Rcpp::Rcout << std::endl;
            }

            arma::mat psi = Theta.row_as_mat(0);
            output["psi_filter"] = Rcpp::wrap(arma::quantile(psi, ci_prob, 1));
            output["eff_forward"] = Rcpp::wrap(eff_forward.t());
            output["log_marginal_likelihood"] = marginal_likelihood(log_cond_marginal, true);

            return;
        }

        void backward_filter(const Model &model, const bool &verbose = VERBOSE)
        {
            const bool full_rank = false;
            arma::vec eff_backward(dim.nT + 1, arma::fill::zeros);
            Theta_backward = Theta; // p x N x (nT + B)

            arma::mat mu_marginal(dim.nP, dim.nT + 1, arma::fill::zeros);
            arma::cube Prec_marginal(dim.nP, dim.nP, dim.nT + 1);
            prior_forward(mu_marginal, Prec_marginal, model, Wt, y);

            arma::vec log_marg(N, arma::fill::zeros);
            for (unsigned int i = 0; i < N; i++)
            {
                arma::vec theta = Theta_backward.slice(dim.nT).col(i);
                log_marg.at(i) = MVNorm::dmvnorm2(
                    theta, mu_marginal.col(dim.nT), Prec_marginal.slice(dim.nT), true);
            }

            for (unsigned int t = dim.nT - 1; t > 0; t--)
            {
                Rcpp::checkUserInterrupt();

                arma::vec logq(N, arma::fill::zeros);
                arma::mat loc(dim.nP, N, arma::fill::zeros);
                arma::cube prec_chol_inv; // nP x nP x N
                if (full_rank)
                {
                    prec_chol_inv = arma::zeros<arma::cube>(dim.nP, dim.nP, N); // nP x nP x N
                }

                arma::vec tau = qbackcast(
                    loc, prec_chol_inv, logq,
                    model, t, Theta_backward.slice(t + 1),
                    mu_marginal, Prec_marginal, Wt, par, y, full_rank);

                tau = tau % weights;
                arma::uvec resample_idx = get_resample_index(tau);

                Theta_backward.slice(t + 1) = Theta_backward.slice(t + 1).cols(resample_idx);
                loc = loc.cols(resample_idx);
                if (full_rank)
                {
                    prec_chol_inv = prec_chol_inv.slices(resample_idx);
                }

                tau = tau.elem(resample_idx);
                weights_backward.row(t + 1) = weights.t();
                eff_backward.at(t) = effective_sample_size(tau);

                log_marg = log_marg.elem(resample_idx);
                logq = logq.elem(resample_idx);
;
                arma::vec logp(N, arma::fill::zeros);
                for (unsigned int i = 0; i < N; i++)
                {
                    arma::vec theta_cur;
                    if (full_rank)
                    {
                        arma::vec eps = arma::randn(dim.nP);
                        arma::vec zt = prec_chol_inv.slice(i).t() * loc.col(i) + eps; // shifted
                        theta_cur = prec_chol_inv.slice(i) * zt;
                        logq.at(i) += MVNorm::dmvnorm0(zt, loc.col(i), prec_chol_inv.slice(i), true);
                    }
                    else
                    {
                        theta_cur = loc.col(i);
                        double eps = R::rnorm(0., std::sqrt(Wt.at(0)));
                        theta_cur.at(dim.nP - 1) += eps;
                        logq.at(i) += R::dnorm4(eps, 0, std::sqrt(Wt.at(0)), true);
                    }

                    logp.at(i) += R::dnorm4(Theta_backward.at(dim.nP - 1, i, t + 1), theta_cur.at(dim.nP - 1), std::sqrt(Wt.at(0)), true);

                    Theta_backward.slice(t).col(i) = theta_cur;

                    double ft_cur = StateSpace::func_ft(model.transfer, t, theta_cur, y);
                    double lambda_cur = LinkFunc::ft2mu(ft_cur, model.dobs.name, par.at(0));

                    logp.at(i) += ObsDist::loglike(
                        y.at(t), model.dobs.name, lambda_cur, model.dobs.par2, true); // observation density
                    // logp.at(i) -= log_marg.at(i);
                    logp.at(i) -= log_marg.at(i);
                    log_marg.at(i) = MVNorm::dmvnorm2(theta_cur, mu_marginal.col(t), Prec_marginal.slice(t), true);
                    logp.at(i) += log_marg.at(i);

                    // double logw_next = std::log(weights_prop_backward.at(t_next, i) + EPS);
                    weights.at(i) = std::exp(logp.at(i) - logq.at(i)); // + logw_next;
                } // loop over i, index of particles

                // double wmax = weights.max();
                // weights.for_each([&wmax](arma::vec::elem_type &val)
                //                  { val -= wmax; });
                // weights = arma::exp(weights);


                if (verbose)
                {
                    Rcpp::Rcout << "\rBackward Filtering: " << dim.nT - t + 1 << "/" << dim.nT;
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

        void smoother(const Model &model, const bool &verbose = VERBOSE)
        {
            
            arma::vec yhat = y; // (nT + 1) x 1
            for (unsigned int t = 0; t < y.n_elem; t++)
            {
                yhat.at(t) = LinkFunc::mu2ft(y.at(t), model.flink.name, 0.);
            }

            // mu0_filter.fill(model.dobs.par1); // N x 1
            Theta_smooth.clear();
            Theta_smooth = Theta;

            arma::mat mu_marginal(dim.nP, dim.nT + 1, arma::fill::zeros);
            arma::cube Prec_marginal(dim.nP, dim.nP, dim.nT + 1);
            prior_forward(mu_marginal, Prec_marginal, model, Wt, y);

            
            for (unsigned int t = 1; t < dim.nT; t++)
            {
                Rcpp::checkUserInterrupt();

                unsigned int t_cur = t;
                unsigned int t_prev = t - 1;
                unsigned int t_next = t + 1;

                arma::vec wfor = arma::vectorise(weights_forward.row(t_prev));
                // arma::uvec resample_idx = get_resample_index(wfor);
                arma::mat Theta_prev = Theta.slice(t_prev); // p x N
                // Theta.slice(t_prev) = Theta_prev.cols(resample_idx);

                arma::vec wback = arma::vectorise(weights_backward.row(t_next));
                // resample_idx = get_resample_index(wback);
                arma::mat Theta_next = Theta_backward.slice(t_next);
                // Theta_backward.slice(t_next) = Theta_next.cols(resample_idx);

                arma::vec logp(N, arma::fill::zeros);
                arma::vec logq = arma::log(wfor + EPS) + arma::log(wback + EPS);
                arma::mat Theta_cur(model.dim.nP, N, arma::fill::zeros);


                for (unsigned int i = 0; i < N; i++)
                {
                    arma::vec gtheta = StateSpace::func_gt(model.transfer, Theta_prev.col(i), y.at(t_prev));
                    arma::vec Ft = LBA::func_Ft(model.transfer, t_cur, gtheta, y);
                    double ft = StateSpace::func_ft(model.transfer, t_cur, gtheta, y);
                    double ft_tilde = ft - arma::as_scalar(Ft.t() * gtheta);

                    double eta = par.at(0) + ft;
                    double lambda = LinkFunc::ft2mu(eta, model.flink.name, 0.);
                    double Vt = ApproxDisturbance::func_Vt_approx(
                        lambda, model.dobs, model.flink.name); // (eq 3.11)

                    double delta = yhat.at(t_cur) - par.at(0) - ft_tilde;
                    double delta2 = delta * delta;

                    arma::mat FFt_norm = Ft * Ft.t() / Vt;
                    double FFt_det = arma::det(FFt_norm);

                    

                    arma::vec theta_cur;
                    if (FFt_det < EPS8)
                    {
                        theta_cur = gtheta;
                        theta_cur.at(0) += R::rnorm(0., std::sqrt(Wt.at(0)));
                        logq.at(i) = R::dnorm4(theta_cur.at(0), gtheta.at(0), std::sqrt(Wt.at(0)), true);
                    }
                    else
                    {
                        arma::mat Gt = LBA::func_Gt(model, gtheta, y.at(t_cur));
                        arma::mat Wprec(model.dim.nP, model.dim.nP, arma::fill::zeros);
                        Wprec.at(0, 0) = 1. / Wt.at(0);
                        arma::mat prec_part1 = Gt.t() * Wprec * Gt;
                        prec_part1.at(0, 0) += 1. / Wt.at(0);

                        arma::mat prec = prec_part1 + FFt_norm;
                        arma::mat Rchol, Sigma;
                        try
                        {
                            Sigma = inverse(Rchol, prec);
                        }
                        catch (const std::exception &e)
                        {
                            prec_part1.brief_print("\n prec_part1: ");
                            arma::vec eigval = arma::eig_sym(prec);
                            eigval.t().brief_print("\n Eigen values of Prec: ");
                            std::cout << "FFt_det = " << FFt_det;
                            throw std::runtime_error(e.what());
                        }

                        arma::vec mu_part1 = Gt.t() * Wprec * Theta_next.col(i);
                        mu_part1.at(0) += gtheta.at(0) / Wt.at(0);

                        arma::vec mu = Ft * (delta / Vt);
                        mu = Sigma * (mu_part1 + mu);

                        theta_cur = mu + Rchol.t() * arma::randn(model.dim.nP);
                        logq.at(i) += MVNorm::dmvnorm2(theta_cur, mu, prec, true);
                    }

                    Theta_cur.col(i) = theta_cur;

                    logp.at(i) = R::dnorm4(theta_cur.at(0), gtheta.at(0), std::sqrt(Wt.at(0)), true);
                    gtheta = StateSpace::func_gt(model.transfer, theta_cur, y.at(t_cur));
                    logp.at(i) += R::dnorm4(Theta_next.at(0, i), theta_cur.at(0), std::sqrt(Wt.at(0)), true);

                    ft = StateSpace::func_ft(model.transfer, t_cur, theta_cur, y);
                    lambda = LinkFunc::ft2mu(ft, model.flink.name, par.at(0));
                    logp.at(i) += ObsDist::loglike(y.at(t_cur), model.dobs.name, lambda, model.dobs.par2, true);

                    logp.at(i) -= MVNorm::dmvnorm2(Theta_next.col(i), mu_marginal.col(t_next), Prec_marginal.slice(t_next), true);

                    // double log_forward = std::log(weights_prop_forward.at(t_prev, i) + EPS);
                    // double log_backward = std::log(weights_prop_backward.at(t_next, i) + EPS);
                    weights.at(i) = logp.at(i) - logq.at(i); // + log_forward + log_backward;
                } // loop over particle i

                

                double wmax = weights.max();
                weights.for_each([&wmax](arma::vec::elem_type &val)
                                 { val -= wmax; });
                weights = arma::exp(weights);
                arma::uvec resample_idx = get_resample_index(weights);
                Theta_smooth.slice(t_cur) = Theta_cur.cols(resample_idx);


                if (verbose)
                {
                    Rcpp::Rcout << "\rSmoothing: " << t + 1 << "/" << dim.nT;
                }
            }

            if (verbose)
            {
                Rcpp::Rcout << std::endl;
            }

            // psi_smooth = Theta_smooth.row_as_mat(0);
            arma::mat psi = Theta_smooth.row_as_mat(0);
            output["psi"] = Rcpp::wrap(arma::quantile(psi, ci_prob, 1));
            return;
        }

        void infer(const Model &model, const bool &verbose = VERBOSE)
        {
            // auto start = std::chrono::high_resolution_clock::now();
            forward_filter(model, verbose);
            // auto stop = std::chrono::high_resolution_clock::now();
            // auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
            // std::cout << "\nForward filter: " << duration.count() << " microseconds" << std::endl;

            if (smoothing)
            {
                // start = std::chrono::high_resolution_clock::now();
                backward_filter(model, verbose);
                // stop = std::chrono::high_resolution_clock::now();
                // duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
                // std::cout << "\nBackward filter: " << duration.count() << " microseconds" << std::endl;

                // start = std::chrono::high_resolution_clock::now();
                smoother(model, verbose);
                // stop = std::chrono::high_resolution_clock::now();
                // duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
                // std::cout << "\nSmoother: " << duration.count() << " microseconds" << std::endl;
            }

            return;
        }
    };

    class PL : public SequentialMonteCarlo
    {
    public:
        PL(const Model &dgtf_model, const arma::vec &y_in) : SequentialMonteCarlo(dgtf_model, y_in) {}

        void init(const Rcpp::List &opts_in)
        {
            Rcpp::List opts = opts_in;
            SequentialMonteCarlo::init(opts);
            B = 1;

            max_iter = 10;
            if (opts.containsElementNamed("max_iter"))
            {
                max_iter = Rcpp::as<unsigned int>(opts["max_iter"]);
            }

            weights_forward.set_size(dim.nT + 1, N);
            weights_forward.zeros();
            weights_backward = weights_forward;
            // weights_prop_forward = weights_forward;
            // weights_prop_backward = weights_forward;

            Theta_backward = Theta;
            Theta_smooth.clear();
            Theta_smooth = Theta;

            // eff_forward.set_size(dim.nT + 1);
            // eff_forward.zeros();
            // eff_filter = eff_forward;
            // eff_backward = eff_forward;

            {
                aw_forward.set_size(N);
                aw_forward.fill(prior_W.par1);
                bw_forward.set_size(N);
                bw_forward.fill(prior_W.par2);

                W_backward.set_size(N);
                W_backward.zeros();
                W_forward = W_backward;

                if (prior_W.infer)
                {
                    std::string par_name = "W_init";
                    Dist dist_W_init;
                    dist_W_init.init("gamma", 1., 1.);
                    if (opts.containsElementNamed("W_init"))
                    {
                        Rcpp::List W_init_opts = Rcpp::as<Rcpp::List>(opts["W_init"]);
                        init_dist(dist_W_init, W_init_opts);
                    }

                    W_filter = draw_param_init(dist_W_init, N);
                    use_discount = false;
                }
            }

            {
                amu_forward.set_size(N);
                amu_forward.fill(prior_mu0.par1);
                bmu_forward.set_size(N);
                bmu_forward.fill(prior_mu0.par2);
                mu0_smooth.set_size(N);
                mu0_smooth.zeros();
                mu0_backward = mu0_smooth;
                mu0_forward = mu0_smooth;

                if (prior_mu0.infer)
                {
                    std::string par_name = "mu0_init";
                    Dist dist_mu0_init;
                    dist_mu0_init.init("gamma", 1., 1.);
                    if (opts.containsElementNamed("mu0_init"))
                    {
                        Rcpp::List mu0_init_opts = Rcpp::as<Rcpp::List>(opts["mu0_init"]);
                        init_dist(dist_mu0_init, mu0_init_opts);
                    }
                    mu0_filter = draw_param_init(dist_mu0_init, N);
                }
            }

            {
                std::string par_name = "rho";
                prior_rho.init("gaussiaN", 1., 1.);
                prior_rho.init_param(false, 0.01);
                rho_mh_sd = 0.1;
                if (opts.containsElementNamed("rho"))
                {
                    init_param_def(opts, par_name, prior_rho);
                    Rcpp::List opts_rho = Rcpp::as<Rcpp::List>(opts["rho"]);
                    if (opts_rho.containsElementNamed("mh_sd"))
                    {
                        rho_mh_sd = Rcpp::as<double>(opts_rho["mh_sd"]);
                    }
                }
                rho_filter.set_size(N);
                rho_filter.fill(prior_rho.val);
                rho_backward = rho_filter;
                rho_forward = rho_filter;
            }

            {
                std::string par_name = "par1";
                prior_par1.init("gamma", 0.1, 0.1);
                prior_par1.init_param(false, 0.01);
                par1_mh_sd = 0.1;
                if (opts.containsElementNamed("par1"))
                {
                    init_param_def(opts, par_name, prior_par1);
                    Rcpp::List opts_tmp = Rcpp::as<Rcpp::List>(opts["par1"]);
                    if (opts_tmp.containsElementNamed("mh_sd"))
                    {
                        par1_mh_sd = Rcpp::as<double>(opts_tmp["mh_sd"]);
                    }
                }
                par1_filter.set_size(N);
                par1_filter.fill(prior_par1.val);
                par1_backward = par1_filter;
                par1_forward = par1_filter;
            }

            {
                std::string par_name = "par2";
                prior_par2.init("gamma", 0.1, 0.1);
                prior_par2.init_param(false, 0.01);
                par2_mh_sd = 0.1;
                if (opts.containsElementNamed("par2"))
                {
                    init_param_def(opts, par_name, prior_par2);
                    Rcpp::List opts_tmp = Rcpp::as<Rcpp::List>(opts["par2"]);
                    if (opts_tmp.containsElementNamed("mh_sd"))
                    {
                        par2_mh_sd = Rcpp::as<double>(opts_tmp["mh_sd"]);
                    }
                }
                par2_filter.set_size(N);
                par2_filter.fill(prior_par2.val);
                par2_backward = par2_filter;
                par2_forward = par2_filter;
            }

            bool infer_lag = prior_par1.infer || prior_par2.infer;
            prior_par1.infer = infer_lag;
            prior_par2.infer = infer_lag;

            mu_marginal.set_size(dim.nP, dim.nT + 1, N);
            mu_marginal.zeros();
            mu_marg_init.set_size(dim.nP);
            mu_marg_init.zeros();
            if (opts.containsElementNamed("mu_marg_init"))
            {
                arma::vec tmp = Rcpp::as<arma::vec>(opts["mu_marg_init"]);
                unsigned int nelem = std::min(tmp.n_elem, dim.nP);
                mu_marg_init.head(nelem) = tmp.head(nelem);
                if (dim.nP > nelem)
                {
                    mu_marg_init.subvec(nelem, dim.nP - 1).fill(tmp.at(nelem - 1));
                }
            }

            Sigma_marginal.set_size(dim.nP * dim.nP, dim.nT + 1, N);
            Sigma_marginal.zeros();
            Prec_marginal = Sigma_marginal;

            Sig_marg_init.set_size(dim.nP, dim.nP);
            Sig_marg_init.eye();
            Sig_marg_init.diag() *= 2.;
            if (opts.containsElementNamed("Sig_diag_marg_init"))
            {
                arma::vec Sig_diag_marg_init(dim.nP, arma::fill::ones);
                Sig_diag_marg_init.fill(2.);

                arma::vec tmp = Rcpp::as<arma::vec>(opts["Sig_diag_marg_init"]);
                unsigned int nelem = std::min(tmp.n_elem, dim.nP);

                Sig_diag_marg_init.head(nelem) = tmp.head(nelem);
                if (dim.nP > nelem)
                {
                    Sig_diag_marg_init.subvec(nelem, dim.nP - 1).fill(tmp.at(nelem - 1));
                }

                Sig_marg_init.diag() = Sig_diag_marg_init;
            }

            if (!Sig_marg_init.is_sympd())
            {
                throw std::invalid_argument("PL::init: initial variance matrix of the artificial marginal should be sympd.");
            }

            filter_pass = false;
            return;
        }

        // static void init_param_stored(
        //     arma::mat &par1,
        //     arma::mat &par2,
        //     arma::mat &par_stored,
        //     arma::vec &par_smooth,
        //     const unsigned int &nT,
        //     const unsigned int N,
        //     const unsigned int M,
        //     const Dist &prior)
        // {
        //     par1.set_size(N, nT + 1); // N x (nT + 1)
        //     par1.fill(prior.par1);
        //     par2 = par1;
        //     par2.fill(prior.par2);

        //     par_stored.set_size(N, nT + 1); // N x (nT + 1)
        //     par_stored.fill(prior.val);

        //     par_smooth.set_size(M); // M x 1
        //     par_smooth.zeros();

        //     return;
        // }

        static Rcpp::List default_settings()
        {
            Rcpp::List opts;
            opts = SequentialMonteCarlo::default_settings();
            opts["max_iter"] = 10;

            arma::vec mu_marg_init(2, arma::fill::zeros);
            arma::vec Sig_diag_marg_init(2, arma::fill::ones);

            opts["mu_marg_init"] = Rcpp::wrap(mu_marg_init.t());
            opts["Sig_diag_marg_init"] = Rcpp::wrap(Sig_diag_marg_init.t());

            Rcpp::List rho_opts;
            rho_opts["infer"] = false;
            rho_opts["init"] = 30;
            rho_opts["mh_sd"] = 0.1;
            rho_opts["prior_param"] = Rcpp::NumericVector::create(0.1, 0.1);
            rho_opts["prior_name"] = "gamma";
            opts["rho"] = rho_opts;

            Rcpp::List par1_opts;
            par1_opts["infer"] = false;
            par1_opts["init"] = 1;
            par1_opts["mh_sd"] = 0.1;
            par1_opts["prior_param"] = Rcpp::NumericVector::create(0.1, 0.1);
            rho_opts["prior_name"] = "invgamma";
            opts["par1"] = par1_opts;

            Rcpp::List par2_opts;
            par2_opts["infer"] = false;
            par2_opts["init"] = 1;
            par2_opts["mh_sd"] = 0.1;
            par2_opts["prior_param"] = Rcpp::NumericVector::create(0.1, 0.1);
            par2_opts["prior_name"] = "invgamma";
            opts["par2"] = par2_opts;

            return opts;
        }

        Rcpp::List get_output(const bool &summarize = TRUE)
        {
            // arma::vec ci_prob = {0.025, 0.5, 0.975};
            // Rcpp::List output;

            // if (summarize)
            // {
            //     arma::mat psi_f = arma::quantile(psi_forward, ci_prob, 1);
            //     output["psi_filter"] = Rcpp::wrap(psi_f);
            // }
            // else
            // {
            //     output["psi_filter"] = Rcpp::wrap(psi_forward);
            // }

            // if (dim.regressor_baseline)
            // {
            //     arma::mat mu0_filter = Theta.row_as_mat(dim.nP - 1);
            //     output["mu0_filter"] = Rcpp::wrap(mu0_filter);
            // }

            // if (prior_W.infer)
            // {
            //     // arma::vec W_filter = W_stored.col(dim.nT);
            //     output["W"] = Rcpp::wrap(W_forward.t());
            //     if (smoothing)
            //     {
            //         output["W_forward2"] = Rcpp::wrap(W_filter.t());
            //         output["W_backward"] = Rcpp::wrap(W_backward.t());
            //     }
            // }

            // if (prior_mu0.infer)
            // {
            //     output["mu0"] = Rcpp::wrap(mu0_forward.t());
            //     if (smoothing)
            //     {
            //         output["mu0_forward2"] = Rcpp::wrap(mu0_filter.t());
            //         output["mu0_backward"] = Rcpp::wrap(mu0_backward.t());
            //     }
            // }

            // if (prior_rho.infer)
            // {
            //     output["rho"] = Rcpp::wrap(rho_forward.t());
            //     if (smoothing)
            //     {
            //         output["rho_forward2"] = Rcpp::wrap(rho_filter.t());
            //         output["rho_backward"] = Rcpp::wrap(rho_backward.t());
            //     }
            // }

            // if (prior_par1.infer || prior_par2.infer)
            // {
            //     output["par1"] = Rcpp::wrap(par1_forward.t());
            //     output["par2"] = Rcpp::wrap(par2_forward.t());
            //     if (smoothing)
            //     {
            //         output["par1_forward2"] = Rcpp::wrap(par1_filter.t());
            //         output["par1_backward"] = Rcpp::wrap(par1_backward.t());

            //         output["par2_forward2"] = Rcpp::wrap(par2_filter.t());
            //         output["par2_backward"] = Rcpp::wrap(par2_backward.t());
            //     }
            // }

            // output["log_marginal_likelihood"] = marginal_likelihood(log_cond_marginal, true);
            // output["eff_forward1"] = Rcpp::wrap(eff_forward.t());

            // if (smoothing)
            // {
            //     arma::mat psi_smooth = get_psi_smooth(); // (nT + 1) x M

            //     if (summarize)
            //     {
            //         arma::mat psi_b = arma::quantile(psi_backward, ci_prob, 1);
            //         output["psi_backward"] = Rcpp::wrap(psi_b);

            //         arma::mat psi = arma::quantile(psi_smooth, ci_prob, 1);
            //         output["psi"] = Rcpp::wrap(psi);
            //     }
            //     else
            //     {
            //         output["psi"] = Rcpp::wrap(psi_smooth);
            //     }

            //     if (dim.regressor_baseline)
            //     {
            //         arma::mat mu0_smooth = Theta_smooth.row_as_mat(dim.nP - 1);
            //         output["mu0"] = Rcpp::wrap(mu0_smooth);
            //     }

            //     output["eff_forward2"] = Rcpp::wrap(eff_filter.t());
            //     output["eff_backward"] = Rcpp::wrap(eff_backward.t());
            // }

            return output;
        }

        Rcpp::List forecast(const Model &model)
        {

            Rcpp::List out;
            if (smoothing)
            {
                out = StateSpace::forecast(y, Theta_smooth, W_smooth, model, nforecast);
            }
            else
            {
                // arma::vec Wtmp = W_stored.col(dim.nT);
                out = StateSpace::forecast(y, Theta, W_filter, model, nforecast);
            }
            return out;
        }


        /**
         * @todo Something wrong with forward filter, comparing to TFS.
         */
        void forward_filter(Model &model, const bool &verbose = VERBOSE)
        {
            const bool full_rank = false;
            arma::vec eff_forward(dim.nT + 1, arma::fill::zeros);
            arma::vec log_cond_marginal = eff_forward;

            // if (arma::any(W_filter < EPS))
            // {
            //     throw std::invalid_argument("PL::forward_filter: W_filter should not be zero.");
            // }

            // if (!prior_W.infer)
            // {
            //     prior_W.val = model.derr.par1;
            //     W_filter.fill(prior_W.val);
            // }

            // if (!prior_mu0.infer)
            // {
            //     prior_mu0.val = model.dobs.par1;
            //     mu0_filter.fill(prior_mu0.val);
            // }

            // if (!prior_rho.infer)
            // {
            //     prior_rho.val = model.dobs.par2;
            //     rho_filter.fill(prior_rho.val);
            // }

            // if (!prior_par1.infer && !prior_par2.infer)
            // {
            //     prior_par1.val = model.transfer.dlag.par1;
            //     prior_par2.val = model.transfer.dlag.par2;

            //     par1_filter.fill(prior_par1.val);
            //     par2_filter.fill(prior_par2.val);
            // }

            std::map<std::string, AVAIL::Func> link_list = AVAIL::link_list;
            std::map<std::string, AVAIL::Dist> dist_list = AVAIL::dist_list;
            bool nonnegative_par1 = (dist_list[prior_par1.name] != AVAIL::Dist::gaussian);
            bool withinone_par1 = (dist_list[prior_par1.name] == AVAIL::Dist::beta);

            bool nonnegative_par2 = (dist_list[prior_par2.name] != AVAIL::Dist::gaussian);
            bool withinone_par2 = (dist_list[prior_par2.name] == AVAIL::Dist::beta);

            arma::vec yhat = y; // (nT + 1) x 1
            for (unsigned int t = 0; t < dim.nT; t++)
            {
                yhat.at(t) = LinkFunc::mu2ft(y.at(t), model.flink.name, 0.);
            }

            for (unsigned int t = 0; t < dim.nT; t++)
            {
                Rcpp::checkUserInterrupt();

                bool print_time = (t == dim.nT - 2);

                bool burnin = (t <= std::min(0.1 * dim.nT, 20.)) ? true : false;
                unsigned int t_old = t;
                unsigned int t_new = t + 1;
                arma::mat Theta_old = Theta.slice(t_old); // p x N, theta[t]

                /**
                 * @brief Resampling using conditional one-step-ahead predictive distribution as weights.
                 *
                 */

                arma::vec logq(N, arma::fill::zeros);
                arma::mat mu;                // nP x N
                arma::cube Prec, Sigma_chol; // nP x nP x N
                arma::uvec updated(N, arma::fill::zeros);

                // auto start = std::chrono::high_resolution_clock::now();
                arma::vec tau = imp_weights_forecast(
                    mu, Prec, Sigma_chol, logq, updated,
                    model, t_new,
                    Theta_old,
                    W_filter, mu0_filter, y, yhat);
                // auto stop = std::chrono::high_resolution_clock::now();
                // auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
                // if (print_time) {
                //     std::cout << "\nimp_weights_forecast: " << duration.count() << " microseconds";
                // }

                tau = tau % weights;
                weights_forward.row(t_old) = tau.t();

                // start = std::chrono::high_resolution_clock::now();
                arma::uvec resample_idx = get_resample_index(tau);

                for (unsigned int k = 0; k <= t_old; k++)
                {
                    Theta.slice(k) = Theta.slice(k).cols(resample_idx);
                    arma::vec wtmp = arma::vectorise(weights_forward.row(k));
                    weights_forward.row(k) = wtmp.elem(resample_idx).t();
                }

                Theta_old = Theta_old.cols(resample_idx); // theta[t]
                // Theta.slice(t_old) = Theta_old;

                // updated = updated.elem(resample_idx);
                logq = logq.elem(resample_idx);
                mu = mu.cols(resample_idx);
                Prec = Prec.slices(resample_idx);
                Sigma_chol = Sigma_chol.slices(resample_idx);

                // tau = tau.elem(resample_idx);
                // weights_forward.row(t_old) = tau.t();
                eff_forward.at(t_new) = effective_sample_size(tau);

                // arma::rowvec wetmp = weights_prop_forward.row(t_old);
                // weights = weights.elem(resample_idx);
                // weights_prop_forward.row(t_old) = weights.t();

                // No need to update static parameters if we already inferred them during forward filtering once with the same data (filter_pass = true).
                if (prior_W.infer)
                {
                    W_filter = W_filter.elem(resample_idx);     // gamma[t]
                    aw_forward = aw_forward.elem(resample_idx); // s[t]
                    bw_forward = bw_forward.elem(resample_idx); // s[t]
                }

                if (prior_mu0.infer)
                {
                    mu0_filter = mu0_filter.elem(resample_idx);   // gamma[t]
                    amu_forward = amu_forward.elem(resample_idx); // s[t]
                    bmu_forward = bmu_forward.elem(resample_idx); // s[t]
                }

                if (prior_rho.infer)
                {
                    rho_filter = rho_filter.elem(resample_idx); // gamma[t]
                }

                if (prior_par1.infer || prior_par2.infer)
                {
                    par1_filter = par1_filter.elem(resample_idx);
                    par2_filter = par2_filter.elem(resample_idx);
                }

                // stop = std::chrono::high_resolution_clock::now();
                // duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
                // if (print_time)
                // {
                //     std::cout << "\nresample: " << duration.count() << " microseconds";
                // }

                // arma::vec Wsqrt = arma::sqrt(W_stored.col(t));
                // arma::vec Wsqrt = arma::sqrt(W_filter);
                bool positive_noise = (t_old < Theta_old.n_rows) ? true : false;

                // NEED TO CHANGE PROPAGATE STEP
                // arma::mat Theta_new = propagate(y.at(t_old), Wsqrt, Theta_old, model, positive_noise);
                // arma::mat Theta_new(model.dim.nP, N, arma::fill::zeros);
                arma::vec logp(N, arma::fill::zeros);

                // start = std::chrono::high_resolution_clock::now();
                for (unsigned int i = 0; i < N; i++)
                {
                    arma::vec theta_new; // nP x 1
                    if (full_rank)
                    {
                        theta_new = mu.col(i) + Sigma_chol.slice(i).t() * arma::randn(model.dim.nP);
                        logq.at(i) += MVNorm::dmvnorm2(theta_new, mu.col(i), Prec.slice(i), true); // sample from posterior
                    }
                    else
                    {
                        theta_new = mu.col(i);
                        double eps = R::rnorm(0., std::sqrt(W_filter.at(i)));
                        theta_new.at(0) += eps;
                        logq.at(i) += R::dnorm4(eps, 0., std::sqrt(W_filter.at(i)), true);
                    }

                    Theta.slice(t_new).col(i) = theta_new;

                    double wtmp = prior_W.val;
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
                            Theta.slice(t),
                            custom_discount_factor,
                            use_custom, default_discount_factor);
                    } // else, we have prior_W.infer = false && use_discount = false. In this case we assume the prior value as the "true" value of W.

                    if (prior_W.infer && !filter_pass)
                    {
                        double err = theta_new.at(0) - Theta_old.at(0, i);
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
                        model.derr._par1 = W_filter.at(i);
                    }

                    if (prior_par1.infer || prior_par2.infer)
                    {
                        unsigned int nlag = model.update_dlag(par1_filter.at(i), par2_filter.at(i), model.dim.nL, false);
                    }
                    double ft_new = StateSpace::func_ft(model.transfer, t_new, theta_new, y); // ft(theta[t+1])
                    double lambda_old = LinkFunc::ft2mu(ft_new, model.flink.name, mu0_filter.at(i)); // ft_new from time t + 1, mu0_filter from time t (old).

                    {
                        if (prior_mu0.infer && !filter_pass)
                        {
                            double Vt_old = ApproxDisturbance::func_Vt_approx(lambda_old, model.dobs, model.flink.name);
                            double amu_scl = amu_forward.at(i) / bmu_forward.at(i);
                            double prec = 1. / bmu_forward.at(i) + 1. / Vt_old;
                            bmu_forward.at(i) = 1. / prec;

                            // double nom = bmu.at(i, t) * amu.at(i, t); // prior precision x mean
                            double eps = yhat.at(t_new) - ft_new;
                            double eps_scl = eps / Vt_old;
                            amu_forward.at(i) = bmu_forward.at(i) * (amu_scl + eps_scl);

                            double sd = std::sqrt(bmu_forward.at(i));
                            double mu0 = R::rnorm(amu_forward.at(i), sd);

                            if (link_list[model.flink.name] == AVAIL::Func::identity)
                            {
                                unsigned int cnt = 0;
                                while (mu0 < 0 && cnt < max_iter)
                                {
                                    mu0 = R::rnorm(amu_forward.at(i), sd);
                                    cnt++;
                                }

                                if (mu0 < 0)
                                {
                                    throw std::invalid_argument("SMC::PL::filter - negative mu0 when using identity link.");
                                }
                            }

                            mu0_filter.at(i) = mu0;
                            logq.at(i) += R::dnorm4(mu0, amu_forward.at(i), sd, true);

                        } // inference of mu0

                        if (prior_mu0.infer)
                        {
                            model.dobs._par1 = mu0_filter.at(i);
                        }
                    } // mu0

                    {
                        if (!prior_rho.infer)
                        {
                            rho_filter.at(i) = prior_rho.val;
                        }
                        else if (!filter_pass)
                        {
                            double rho_old = rho_filter.at(i);
                            rho_filter.at(i) = R::rnorm(rho_old, rho_mh_sd * rho_old);
                            unsigned int cnt = 0;
                            while (rho_filter.at(i) < 0 && cnt < max_iter)
                            {
                                rho_filter.at(i) = R::rnorm(rho_old, rho_mh_sd * rho_old);
                                cnt++;
                            }
                            logq.at(i) += R::dnorm4(rho_filter.at(i), rho_old, rho_mh_sd * rho_old, 1);
                        }

                        if (prior_rho.infer)
                        {
                            model.dobs._par2 = rho_filter.at(i);
                        }
                    } // rho

                    {
                        if (!prior_par1.infer && !prior_par2.infer)
                        {
                            par1_filter.at(i) = prior_par1.val;
                            par2_filter.at(i) = prior_par2.val;
                        }
                        else if (!filter_pass)
                        {
                            double par1_old = par1_filter.at(i);
                            par1_filter.at(i) = R::rnorm(par1_old, par1_mh_sd * par1_old);

                            if (nonnegative_par1)
                            {
                                unsigned int cnt = 0;
                                if (withinone_par1)
                                {
                                    while ((par1_filter.at(i) < 0 || par1_filter.at(i) > 1) && cnt < max_iter)
                                    {
                                        par1_filter.at(i) = R::rnorm(par1_old, par1_mh_sd * par1_old);
                                        cnt++;
                                    }
                                }
                                else
                                {
                                    while (par1_filter.at(i) < 0 && cnt < max_iter)
                                    {
                                        par1_filter.at(i) = R::rnorm(par1_old, par1_mh_sd * par1_old);
                                        cnt++;
                                    }
                                }
                            }

                            logq.at(i) += R::dnorm4(par1_filter.at(i), par1_old, par1_mh_sd * par1_old, 1);

                            double par2_old = par2_filter.at(i);
                            par2_filter.at(i) = R::rnorm(par2_old, par2_mh_sd * par2_old);

                            if (nonnegative_par2)
                            {
                                unsigned int cnt = 0;
                                if (withinone_par2)
                                {
                                    while ((par2_filter.at(i) < 0 || par2_filter.at(i) > 1) && cnt < max_iter)
                                    {
                                        par2_filter.at(i) = R::rnorm(par2_old, par2_mh_sd * par2_old);
                                        cnt++;
                                    }
                                }
                                else
                                {
                                    while (par2_filter.at(i) < 0 && cnt < max_iter)
                                    {
                                        par2_filter.at(i) = R::rnorm(par2_old, par2_mh_sd * par2_old);
                                        cnt++;
                                    }
                                }
                            }

                            logq.at(i) += R::dnorm4(par2_filter.at(i), par2_old, par2_mh_sd * par2_old, 1);
                        }

                        if (prior_par1.infer || prior_par2.infer)
                        {
                            unsigned int nlag = model.update_dlag(par1_filter.at(i), par2_filter.at(i), model.dim.nL, false);
                        }
                    } // lag distribution

                    double lambda_new = LinkFunc::ft2mu(ft_new, model.dobs.name, mu0_filter.at(i));
                    logp.at(i) = ObsDist::loglike(
                        y.at(t_new), model.dobs.name, lambda_new, rho_filter.at(i), true); // observation density
                    logp.at(i) += R::dnorm4(theta_new.at(0), Theta_old.at(0, i), std::sqrt(W_filter.at(i)), true);

                    // double logw_old = std::log(weights_prop_forward.at(t_old, i) + EPS);
                    weights.at(i) = logp.at(i) - logq.at(i); // + logw_old;
                } // loop over i, index of particles; end of propagation

                // stop = std::chrono::high_resolution_clock::now();
                // duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
                // if (print_time)
                // {
                //     std::cout << "\npropagate: " << duration.count() << " microseconds" << std::endl;
                // }

                double wmax = weights.max();
                weights.for_each([&wmax](arma::vec::elem_type &val)
                                 { val -= wmax; });
                weights = arma::exp(weights);
                bound_check<arma::vec>(weights, "PL::forward_filter: propagation weights at t = " + std::to_string(t_old));

                // eff_filter.at(t_new) = effective_sample_size(weights);
                log_cond_marginal.at(t_new) = log_conditional_marginal(weights);

                // Theta.slice(t_new) = Theta_new;

                // if (eff_filter.at(t_new) < 0.5 * N || t_new >= dim.nT - 1)
                // {
                //     arma::uvec resample_idx = get_resample_index(weights);
                //     weights.ones();

                //     for (unsigned int t = 0; t <= t_new; t ++)
                //     {
                //         arma::mat tmp = Theta.slice(t); // p x N
                //         Theta.slice(t) = tmp.cols(resample_idx);
                //     }
                //     // Theta.slice(t_new) = Theta_new.cols(resample_idx);

                //     // Move: psi

                //     if (prior_W.infer)
                //     {
                //         W_filter = W_filter.elem(resample_idx);
                //         aw_forward = aw_forward.elem(resample_idx);
                //         bw_forward = bw_forward.elem(resample_idx);
                //     }

                //     if (prior_mu0.infer)
                //     {
                //         mu0_filter = mu0_filter.elem(resample_idx);
                //         amu_forward = amu_forward.elem(resample_idx);
                //         bmu_forward = bmu_forward.elem(resample_idx);
                //     }

                //     if (prior_rho.infer)
                //     {
                //         rho_filter = rho_filter.elem(resample_idx);
                //     }

                //     if (prior_par1.infer || prior_par2.infer)
                //     {
                //         par1_filter = par1_filter.elem(resample_idx);
                //         par2_filter = par2_filter.elem(resample_idx);
                //     }
                // }

                // weights_prop_forward.row(t_new) = weights.t();

                if (verbose)
                {
                    Rcpp::Rcout << "\rForward Filtering: " << t + 1 << "/" << dim.nT;
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
                if (prior_mu0.infer)
                {
                    output["mu0"] = Rcpp::wrap(mu0_filter.t());
                }
                if (prior_rho.infer)
                {
                    output["rho"] = Rcpp::wrap(rho_filter.t());
                }
                if (prior_par1.infer)
                {
                    output["par1"] = Rcpp::wrap(par1_filter.t());
                }
                if (prior_par2.infer)
                {
                    output["par2"] = Rcpp::wrap(par2_filter.t());
                }

                W_forward = W_filter;
                mu0_forward = mu0_filter;
                rho_forward = rho_filter;
                par1_forward = par1_filter;
                par2_forward = par2_filter;

                // eff_forward = eff_filter;
            }
            else
            {
                output["eff_forward2"] = Rcpp::wrap(eff_forward.t());
            }
            filter_pass = true;
            return;
        }

        void backward_filter(Model &model, const bool &verbose = VERBOSE)
        {
            arma::vec eff_backward(dim.nT + 1, arma::fill::zeros);
            if (!filter_pass)
            {
                throw std::runtime_error("SMC::PL: you need to run a forward filtering pass before running backward filtering.");
            }

            forward_filter(model, VERBOSE);

            Theta_backward = Theta;
            W_backward = W_filter;
            mu0_backward = mu0_filter;
            rho_backward = rho_filter;
            par1_backward = par1_filter;
            par2_backward = par2_filter;

            arma::vec yhat = y; // (nT + 1) x 1
            for (unsigned int t = 0; t < y.n_elem; t++)
            {
                yhat.at(t) = LinkFunc::mu2ft(y.at(t), model.flink.name, 0.);
            }

            // mu0_filter.fill(model.dobs.par1); // N x 1

            arma::mat Prec_marg_init = Sig_marg_init; // nP x nP
            Prec_marg_init.diag() = 1. / Sig_marg_init.diag();
            for (unsigned int i = 0; i < N; i++)
            {
                mu_marginal.slice(i).col(0) = mu_marg_init;              // nP x 1
                Sigma_marginal.slice(i).col(0) = Sig_marg_init.as_col(); // np^{2} x 1
                Prec_marginal.slice(i).col(0) = Prec_marg_init.as_col();
            }

            arma::mat Ieps(dim.nP, dim.nP, arma::fill::eye);
            Ieps.for_each([](arma::mat::elem_type &val)
                          { val *= EPS; });

            arma::vec Wprec = 1. / W_backward;

            for (unsigned int t = 1; t <= dim.nT; t++)
            {
                Rcpp::checkUserInterrupt();

                for (unsigned int i = 0; i < N; i++)
                {
                    mu_marginal.slice(i).col(t) = StateSpace::func_gt(model.transfer, mu_marginal.slice(i).col(t - 1), y.at(t - 1));

                    arma::mat Gt = LBA::func_Gt(model, mu_marginal.slice(i).col(t - 1), y.at(t - 1));
                    arma::mat Vt = arma::reshape(Sigma_marginal.slice(i).col(t - 1), model.dim.nP, model.dim.nP);
                    arma::mat Sig = Gt * Vt * Gt.t();
                    Sig.at(0, 0) += W_backward.at(i);
                    Sig = arma::symmatu(Sig + Ieps);
                    arma::mat Prec = inverse(Sig);

                    Sigma_marginal.slice(i).col(t) = Sig.as_col();
                    Prec_marginal.slice(i).col(t) = Prec.as_col();
                }

                if (verbose)
                {
                    Rcpp::Rcout << "\rArtificial marginal: " << t << "/" << dim.nT;
                }
            }
            Rcpp::Rcout << std::endl;

            arma::vec log_marg(N, arma::fill::zeros);
            for (unsigned int i = 0; i < N; i++)
            {
                arma::vec theta = Theta_backward.slice(dim.nT).col(i);
                arma::mat Prec = arma::reshape(Prec_marginal.slice(i).col(dim.nT), dim.nP, dim.nP);
                log_marg.at(i) = MVNorm::dmvnorm2(
                    theta, mu_marginal.slice(i).col(dim.nT), Prec, true);
            }

            for (unsigned int t = dim.nT - 1; t > 0; t--)
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
                arma::cube Prec, Sigma_chol; // nP x nP x N
                arma::uvec updated(N, arma::fill::zeros);

                arma::vec tau = imp_weights_backcast(
                    mu, Prec, Sigma_chol, logq, updated,
                    model, t_cur, Theta_next,
                    W_backward, mu0_backward, mu_marginal, Sigma_marginal, y, yhat);

                tau = tau % weights;
                // weights_backward.row(t_next) = tau.t();
                arma::uvec resample_idx = get_resample_index(tau);

                // for (unsigned int k = model.dim.nT - 1; k >= t_next; k--)
                // {
                //     Theta_backward.slice(k) = Theta_backward.slice(k).cols(resample_idx);
                //     arma::vec wtmp = arma::vectorise(weights_backward.row(k));
                //     weights_backward.row(k) = wtmp.elem(resample_idx).t();
                // }

                Theta_next = Theta_next.cols(resample_idx); // theta[t]
                Theta_backward.slice(t_next) = Theta_next;

                mu = mu.cols(resample_idx);
                Prec = Prec.slices(resample_idx);
                Sigma_chol = Sigma_chol.slices(resample_idx);

                tau = tau.elem(resample_idx);
                weights_backward.row(t_next) = tau.t();
                eff_backward.at(t_cur) = effective_sample_size(tau);

                // weights = weights.elem(resample_idx);
                // weights_prop_backward.row(t_next) = weights.t();
                // weights_prop_backward should be consistent with the corresponding particles.

                log_marg = log_marg.elem(resample_idx);
                logq = logq.elem(resample_idx);
                updated = updated.elem(resample_idx);

                if (prior_W.infer)
                {
                    W_backward = W_backward.elem(resample_idx);
                }

                if (prior_mu0.infer)
                {
                    mu0_backward = mu0_backward.elem(resample_idx);
                }

                if (prior_rho.infer)
                {
                    rho_backward = rho_backward.elem(resample_idx);
                }

                if (prior_par1.infer || prior_par2.infer)
                {
                    par1_backward = par1_backward.elem(resample_idx);
                    par2_backward = par2_backward.elem(resample_idx);
                }

                mu_marginal = mu_marginal.slices(resample_idx);
                Sigma_marginal = Sigma_marginal.slices(resample_idx);

                // W_filter.fill(Wt.at(t_next));

                // NEED TO CHANGE PROPAGATE STEP
                // arma::mat Theta_new = propagate(y.at(t_old), Wsqrt, Theta_old, model, positive_noise);
                arma::mat Theta_cur(model.dim.nP, N, arma::fill::zeros);
                arma::vec logp(N, arma::fill::zeros);
                for (unsigned int i = 0; i < N; i++)
                {
                    if (prior_W.infer)
                    {
                        model.derr._par1 = W_backward.at(i);
                    }
                    if (prior_mu0.infer)
                    {
                        model.dobs._par1 = mu0_backward.at(i);
                    }
                    if (prior_rho.infer)
                    {
                        model.dobs._par2 = rho_backward.at(i);
                    }
                    if (prior_par1.infer || prior_par2.infer)
                    {
                        unsigned int nlag = model.update_dlag(par1_backward.at(i), par2_backward.at(i), model.dim.nL, false);
                    }
                    arma::vec theta_cur = mu.col(i) + Sigma_chol.slice(i).t() * arma::randn(model.dim.nP);
                    Theta_cur.col(i) = theta_cur;

                    double ft_cur = StateSpace::func_ft(model.transfer, t_cur, theta_cur, y);
                    double lambda_cur = LinkFunc::ft2mu(ft_cur, model.dobs.name, mu0_backward.at(i));

                    if (updated.at(i) == 1)
                    {
                        logq.at(i) += MVNorm::dmvnorm2(theta_cur, mu.col(i), Prec.slice(i), true);
                        logp.at(i) += R::dnorm4(theta_cur.at(0), Theta_next.at(0, i), std::sqrt(W_backward.at(i)), true);
                    }

                    logp.at(i) += ObsDist::loglike(
                        y.at(t_cur), model.dobs.name, lambda_cur, rho_backward.at(i), true); // observation density

                    logp.at(i) -= log_marg.at(i);
                    arma::vec Vprec = Prec_marginal.slice(i).col(t_cur);
                    arma::mat Vprec_cur = arma::reshape(Vprec, model.dim.nP, model.dim.nP);
                    arma::vec v_cur = mu_marginal.slice(i).col(t_cur);
                    log_marg.at(i) = MVNorm::dmvnorm2(theta_cur, v_cur, Vprec_cur, true);
                    logp.at(i) += log_marg.at(i);

                    // double logw_next = std::log(weights_prop_backward.at(t_next, i) + EPS);
                    weights.at(i) = logp.at(i) - logq.at(i); // + logw_next;
                } // loop over i, index of particles

                double wmax = weights.max();
                weights.for_each([&wmax](arma::vec::elem_type &val)
                                 { val -= wmax; });
                weights = arma::exp(weights);
                bound_check<arma::vec>(weights, "PL::backward_filter: propagation weights at t = " + std::to_string(t_cur));

                // eff_backward.at(t_cur) = effective_sample_size(weights);
                // if (eff_backward.at(t_cur) < 0.5 * N || t_cur <= 1)
                // {
                //     // resample
                //     resample_idx = get_resample_index(weights);
                //     Theta_cur = Theta_cur.cols(resample_idx);
                //     log_marg = log_marg.elem(resample_idx);

                //     weights.ones();

                //     arma::rowvec wetmp = weights_backward.row(t_cur);
                //     weights_backward.row(t_cur) = wetmp.elem(resample_idx).t();

                //     if (prior_W.infer)
                //     {
                //         W_backward = W_backward.elem(resample_idx);
                //     }

                //     if (prior_mu0.infer)
                //     {
                //         mu0_backward = mu0_backward.elem(resample_idx);
                //     }

                //     if (prior_rho.infer)
                //     {
                //         rho_backward = rho_backward.elem(resample_idx);
                //     }

                //     if (prior_par1.infer || prior_par2.infer)
                //     {
                //         par1_backward = par1_backward.elem(resample_idx);
                //         par2_backward = par2_backward.elem(resample_idx);
                //     }

                //     mu_marginal = mu_marginal.slices(resample_idx);
                //     Sigma_marginal = Sigma_marginal.slices(resample_idx);

                // }

                Theta_backward.slice(t_cur) = Theta_cur;
                // weights_prop_backward.row(t_cur) = weights.t();

                if (verbose)
                {
                    Rcpp::Rcout << "\rBackward Filtering: " << dim.nT - t + 1 << "/" << dim.nT;
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

        void backward_smoother(const Model &model, const bool &verbose = VERBOSE)
        {
            weights.ones();
            arma::uvec idx = sample(N, M, weights, false, true); // M x 1

            arma::mat theta_tmp = Theta.slice(dim.nT);        // p x N
            arma::mat theta_tmp2 = theta_tmp.cols(idx);       // p x M
            Theta_smooth.slice(dim.nT) = theta_tmp2;          // p x M

            // arma::vec ptmp0 = arma::vectorise(Theta.slice(dim.nT).row(0)); // p x N
            // arma::vec ptmp = ptmp0.elem(idx);

            // psi_smooth.row(dim.nT) = ptmp.t();

            // arma::vec wtmp = W_stored.col(dim.nT);
            // W_smooth.col(dim.nT) = wtmp.elem(idx);
            W_smooth = W_filter.elem(idx);          // M x 1
            arma::vec Wsqrt = arma::sqrt(W_smooth); // M x 1

            for (unsigned int t = dim.nT; t > 0; t--)
            {
                Rcpp::checkUserInterrupt();

                // arma::vec Wtmp0 = W_stored.col(t - 1); // N x 1
                // arma::vec Wtmp = Wtmp0.elem(idx); // M x 1
                // arma::vec Wsqrt = arma::sqrt(W_stored.col(t - 1)); // M x 1

                // arma::uvec smooth_idx = get_smooth_index(t, Wsqrt, idx);
                arma::rowvec psi_smooth_now = Theta_smooth.slice(t).row(0);                       // 1 x M
                arma::rowvec psi_filter_prev = Theta.slice(t - 1).row(0);                         // 1 x N
                arma::uvec smooth_idx = get_smooth_index(psi_smooth_now, psi_filter_prev, Wsqrt); // M x 1

                // if (infer_W)
                // {
                //     W_smooth.col(t - 1) = Wtmp.elem(smooth_idx);
                // }

                arma::mat theta_tmp0 = Theta.slice(t - 1); // p x N
                // arma::mat theta_tmp = theta_tmp0.cols(idx);       // p x M
                theta_tmp = theta_tmp0.cols(smooth_idx);

                Theta_smooth.slice(t - 1) = theta_tmp;
                // psi_smooth.row(t - 1) = theta_tmp.row(0);

                if (verbose)
                {
                    Rcpp::Rcout << "\rSmoothing: " << dim.nT - t + 1 << "/" << dim.nT;
                }
            }

            if (verbose)
            {
                Rcpp::Rcout << std::endl;
            }

            // psi_smooth.clear();
            // psi_smooth = Theta_smooth.row_as_mat(0);
            arma::mat psi = Theta_smooth.row_as_mat(0);
            output["psi"] = Rcpp::wrap(arma::quantile(psi, ci_prob, 1));
        }

        void two_filter_smoother(Model &model, const bool &verbose = VERBOSE)
        {
            arma::vec yhat = y; // (nT + 1) x 1
            for (unsigned int t = 0; t < y.n_elem; t++)
            {
                yhat.at(t) = LinkFunc::mu2ft(y.at(t), model.flink.name, 0.);
            }

            Theta_smooth.clear();
            Theta_smooth = Theta;

            arma::mat Ieps(dim.nP, dim.nP, arma::fill::eye);
            Ieps.for_each([](arma::mat::elem_type &val)
                          { val *= EPS; });

            for (unsigned int t = 1; t < dim.nT; t++)
            {
                Rcpp::checkUserInterrupt();

                unsigned int t_cur = t;
                unsigned int t_prev = t - 1;
                unsigned int t_next = t + 1;

                arma::vec wfor = arma::vectorise(weights_forward.row(t_prev));
                // arma::uvec resample_idx = get_resample_index(wfor);
                arma::mat Theta_prev = Theta.slice(t_prev); // p x N
                // Theta.slice(t_prev) = Theta_prev.cols(resample_idx);
                arma::vec W_filter_new = W_filter;     //.elem(resample_idx);
                arma::vec mu0_filter_new = mu0_filter; // .elem(resample_idx);

                // wfor = wfor.elem(resample_idx);
                // weights_forward.row(t_prev) = wfor.t();

                // arma::vec w_prop = arma::vectorise(weights_prop_forward.row(t_prev));
                // w_prop = w_prop.elem(resample_idx);
                // weights_prop_forward.row(t_prev) = w_prop.t();

                arma::vec wback = arma::vectorise(weights_backward.row(t_next));
                // resample_idx = get_resample_index(wback);
                arma::mat Theta_next = Theta_backward.slice(t_next);
                // Theta_backward.slice(t_next) = Theta_next.cols(resample_idx);
                arma::vec W_backward_new = W_backward;       //.elem(resample_idx);
                arma::vec mu0_backward_new = mu0_backward;   //.elem(resample_idx);
                arma::vec rho_backward_new = rho_backward;   //.elem(resample_idx);
                arma::vec par1_backward_new = par1_backward; //.elem(resample_idx);
                arma::vec par2_backward_new = par2_backward; //.elem(resample_idx);

                // wback = wback.elem(resample_idx);
                // weights_backward.row(t_next) = wback.t();
                // should we reset wfor and wback to one after resampling?

                // w_prop = arma::vectorise(weights_prop_backward.row(t_next));
                // w_prop = w_prop.elem(resample_idx);
                // weights_prop_backward.row(t_next) = w_prop.t();

                arma::mat prec_tmp = Prec_marginal.col_as_mat(t_next); // nP^2 x N
                arma::mat Prec_marg = prec_tmp;                        //.cols(resample_idx);     // nP^2 x N
                arma::mat mu_marg = mu_marginal.col_as_mat(t_next);    // nP x N
                // mu_marg = mu_marg.cols(resample_idx);

                arma::vec logp(N, arma::fill::zeros);
                arma::vec logq = arma::log(wfor + EPS) + arma::log(wback + EPS);
                arma::mat Theta_cur(model.dim.nP, N, arma::fill::zeros);
                for (unsigned int i = 0; i < N; i++)
                {
                    if (prior_W.infer)
                    {
                        model.derr._par1 = W_backward_new.at(i);
                    }
                    if (prior_mu0.infer)
                    {
                        model.dobs._par1 = mu0_backward_new.at(i);
                    }
                    if (prior_rho.infer)
                    {
                        model.dobs._par2 = rho_backward_new.at(i);
                    }

                    arma::vec gtheta = StateSpace::func_gt(model.transfer, Theta_prev.col(i), y.at(t_prev));
                    arma::vec Ft = LBA::func_Ft(model.transfer, t_cur, gtheta, y);
                    double ft = StateSpace::func_ft(model.transfer, t_cur, gtheta, y);
                    double ft_tilde = ft - arma::as_scalar(Ft.t() * gtheta);

                    double eta = mu0_backward_new.at(i) + ft;
                    double lambda = LinkFunc::ft2mu(eta, model.flink.name, 0.);
                    double Vt = ApproxDisturbance::func_Vt_approx(
                        lambda, model.dobs, model.flink.name); // (eq 3.11)

                    double delta = yhat.at(t_cur) - mu0_backward_new.at(i) - ft_tilde;
                    double delta2 = delta * delta;

                    arma::mat FFt_norm = Ft * Ft.t() / Vt;
                    double FFt_det = arma::det(FFt_norm);

                    arma::vec theta_cur;
                    if (FFt_det < EPS8)
                    {
                        theta_cur = gtheta;
                        theta_cur.at(0) += R::rnorm(0., std::sqrt(W_backward_new.at(i)));
                        logq.at(i) += R::dnorm4(theta_cur.at(0), gtheta.at(0), std::sqrt(W_backward_new.at(i)), true);
                    }
                    else
                    {
                        arma::mat Gt = LBA::func_Gt(model, gtheta, y.at(t_cur));
                        arma::mat Wprec(model.dim.nP, model.dim.nP, arma::fill::zeros);
                        Wprec.at(0, 0) = 1. / W_backward_new.at(i);
                        arma::mat prec_part1 = Gt.t() * Wprec * Gt;
                        prec_part1.at(0, 0) += 1. / W_backward_new.at(i);

                        arma::mat prec = prec_part1 + FFt_norm;
                        arma::mat Rchol, Sigma;
                        try
                        {
                            Sigma = inverse(Rchol, prec);
                        }
                        catch (const std::exception &e)
                        {
                            prec_part1.brief_print("\n prec_part1: ");
                            arma::vec eigval = arma::eig_sym(prec);
                            eigval.t().brief_print("\n Eigen values of Prec: ");
                            std::cout << "FFt_det = " << FFt_det;
                            throw std::runtime_error(e.what());
                        }

                        arma::vec mu_part1 = Gt.t() * Wprec * Theta_next.col(i);
                        mu_part1.at(0) += gtheta.at(0) / W_backward_new.at(i);

                        arma::vec mu = Ft * (delta / Vt);
                        mu = Sigma * (mu_part1 + mu);

                        theta_cur = mu + Rchol.t() * arma::randn(model.dim.nP);
                        logq.at(i) += MVNorm::dmvnorm2(theta_cur, mu, prec, true);
                    }

                    Theta_cur.col(i) = theta_cur;

                    logp.at(i) = R::dnorm4(theta_cur.at(0), gtheta.at(0), std::sqrt(W_filter_new.at(i)), true);

                    if (prior_par1.infer || prior_par2.infer)
                    {
                        unsigned int nlag = model.update_dlag(par1_backward_new.at(i), par2_backward_new.at(i), model.dim.nL, false);
                    }
                    gtheta = StateSpace::func_gt(model.transfer, theta_cur, y.at(t_cur));
                    logp.at(i) += R::dnorm4(Theta_next.at(0, i), theta_cur.at(0), std::sqrt(W_backward_new.at(i)), true);

                    ft = StateSpace::func_ft(model.transfer, t_cur, theta_cur, y);
                    lambda = LinkFunc::ft2mu(ft, model.flink.name, mu0_backward_new.at(i));
                    logp.at(i) += ObsDist::loglike(y.at(t_cur), model.dobs.name, lambda, rho_backward_new.at(i), true);

                    arma::mat pmarg = arma::reshape(Prec_marg.col(i), model.dim.nP, model.dim.nP);
                    logp.at(i) -= MVNorm::dmvnorm2(Theta_next.col(i), mu_marg.col(i), pmarg, true);

                    // double log_forward = std::log(weights_prop_forward.at(t_prev, i) + EPS);
                    // double log_backward = std::log(weights_prop_backward.at(t_next, i) + EPS);
                    weights.at(i) = logp.at(i) - logq.at(i); // + log_forward + log_backward;
                } // loop over particle i

                double wmax = weights.max();
                weights.for_each([&wmax](arma::vec::elem_type &val)
                                 { val -= wmax; });
                weights = arma::exp(weights);
                bound_check<arma::vec>(weights, "PL::two_filter_smoother: propagation weights at t = " + std::to_string(t_cur));

                arma::uvec resample_idx = get_resample_index(weights);
                Theta_smooth.slice(t_cur) = Theta_cur.cols(resample_idx);

                if (verbose)
                {
                    Rcpp::Rcout << "\rSmoothing: " << t + 1 << "/" << dim.nT;
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

        void infer(Model &model, const bool &verbose = VERBOSE)
        {
            if (prior_W.infer && use_discount)
            {
                use_discount = false;
            }

            forward_filter(model, verbose); // 2,253,382 ms per 1000 particles

            if (smoothing)
            {
                // backward_smoother(model, verbose);
                backward_filter(model, verbose);     // 14,600,157 ms per 1000 particles
                two_filter_smoother(model, verbose); // 1,431,610 ms per 1000 particles
            } // opts.smoothing
        } // Particle Learning inference

    private:
        bool filter_pass = false;

        arma::vec mu_marg_init;  // nP x 1
        arma::mat Sig_marg_init; // nP x nP

        arma::cube mu_marginal;    // nP x (nT + 1) x N
        arma::cube Sigma_marginal; // nP^2 x (nT + 1) x N
        arma::cube Prec_marginal;  // nP^2 x (nT + 1) x N

        // arma::vec eff_forward;  // (nT + 1) x 1
        // arma::vec eff_backward; // (nT + 1) x 1
        // arma::vec eff_filter;

        arma::mat weights_forward;  // (nT + 1) x N
        arma::mat weights_backward; // (nT + 1) x N
        // arma::mat weights_prop_forward;  // (nT + 1) x N
        // arma::mat weights_prop_backward; // (nT + 1) x N
        arma::cube Theta_backward; // p x N x (nT + 1)

        // arma::mat aw; // N x (nT + 1), shape of IG
        // arma::mat bw; // N x (nT + 1), scale of IG (i.e. rate of corresponding Gamma)
        arma::vec aw_forward; // N x 1, shape of IG
        arma::vec bw_forward; // N x 1, scale of IG (i.e. rate of corresponding Gamma)
        // arma::vec aw_backward; // N x 1, shape of IG
        // arma::vec bw_backward; // N x 1, scale of IG (i.e. rate of corresponding Gamma)
        // arma::mat W_stored; // N x (nT + 1)
        // arma::vec W_backward; // N x 1
        arma::vec W_smooth; // M x 1
        arma::vec W_backward;
        arma::vec W_forward; // N x 4, 1st filter, 2nd filter, backward filter, smoothing

        // arma::mat amu; // N x (nT + 1), mean of normal
        // arma::mat bmu; // N x (nT + 1), precision of normal
        arma::vec amu_forward; // N x 1, mean of normal
        arma::vec bmu_forward; // N x 1, precision of normal
        // arma::vec amu_backward; // N x 1
        // arma::vec bmu_backward; // N x 1
        // arma::mat mu0_stored; // N x (nT + 1)
        // arma::vec mu0_backward; // N x 1
        arma::vec mu0_smooth; // M x 1
        arma::vec mu0_backward;
        arma::vec mu0_forward;

        Prior prior_rho;
        double rho_mh_sd = 1.;
        arma::vec rho_filter; // N x 1
        arma::vec rho_backward;
        arma::vec rho_forward;

        Prior prior_par1;
        double par1_mh_sd = 1.;
        arma::vec par1_filter; // N x 1
        arma::vec par1_backward;
        arma::vec par1_forward;

        Prior prior_par2;
        double par2_mh_sd = 1.;
        arma::vec par2_filter; // N x 1
        arma::vec par2_backward;
        arma::vec par2_forward;

        arma::mat e11; // N x (nT + 1)

        unsigned int max_iter = 10;
    };
}

#endif