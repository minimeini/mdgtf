#ifndef _SEQUENTIALMONTECARLO_H
#define _SEQUENTIALMONTECARLO_H

#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <algorithm>
#include <RcppArmadillo.h>
#include "Model.hpp"

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

            log_cond_marginal.set_size(dim.nT + 1);
            log_cond_marginal.zeros();

            return;
        }

        SequentialMonteCarlo()
        {
            dim.init_default();
            y.set_size(dim.nT + 1);
            y.zeros();

            log_cond_marginal.set_size(dim.nT + 1);
            log_cond_marginal.zeros();
        }

        static Rcpp::List default_settings()
        {
            Rcpp::List opts;
            opts["num_particle"] = 1000;
            opts["num_smooth"] = 500;
            opts["num_backward"] = 1;
            opts["num_step_ahead_forecast"] = 0;
            opts["use_discount"] = false;
            opts["use_custom"] = false;
            opts["custom_discount_factor"] = 0.95;
            opts["do_smoothing"] = false;

            Rcpp::List mu0_opts;
            mu0_opts["infer"] = false;
            mu0_opts["init"] = 0.;
            mu0_opts["prior_name"] = "gamma";
            mu0_opts["prior_param"] = Rcpp::NumericVector::create(1., 1.);
            opts["mu0"] = mu0_opts;

            Rcpp::List W_opts;
            W_opts["infer"] = false;
            W_opts["init"] = 0.01;
            W_opts["prior_name"] = "invgamma";
            W_opts["prior_param"] = Rcpp::NumericVector::create(1., 1.);
            opts["W"] = W_opts;

            return opts;
        }


        void infer(const Model &model){return;}



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
            lambda = weights;

            M = 500;
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

            Theta_stored.set_size(dim.nP, N, dim.nT + B);
            Theta_stored.zeros();

            Theta_smooth.set_size(dim.nP, M, dim.nT + B);
            Theta_smooth.zeros();

            std::string par_name = "mu0";
            prior_mu0.init("gamma", 1., 1.);
            prior_mu0.init_param(false, 0.);
            mu0_filter.set_size(N);
            mu0_filter.fill(prior_mu0.val);
            if (settings.containsElementNamed("mu0"))
            {

                init_param_def(settings, par_name, prior_mu0);
            }

            if (dim.regressor_baseline)
            {
                prior_mu0.infer = false;
                arma::vec mu0 = draw_param_init(prior_mu0, N);
                for (unsigned int t = 0; t < dim.nT + B; t++)
                {
                    Theta_stored.slice(t).row(dim.nP - 1) = mu0.t();
                }
            }

            par_name = "W";
            prior_W.init("invgamma", 1., 1.);
            prior_W.init_param(false, 0.01);
            W_filter.set_size(N);
            W_filter.fill(prior_W.val);
            if (settings.containsElementNamed("W"))
            {

                init_param_def(settings, par_name, prior_W);
            }

            return;
        }


        static void init_param_def(
            Rcpp::List &opts,
            std::string &par_name,
            Prior &prior
        )
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
            const double &const_val = 0.,
            const unsigned int &max_iter = 100)
        {
            std::map<std::string, AVAIL::Dist> dist_list = AVAIL::dist_list;
            arma::vec par_init(N, arma::fill::zeros);

            for (unsigned int i = 0; i < N; i ++)
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
                        success = std::isfinite(val);
                        cnt ++;
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
                        success = std::isfinite(val);
                        cnt ++;
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
            }

            

            return par_init;
        }

        arma::mat get_psi_filter()
        {
            arma::mat psi_tmp = Theta_stored.row_as_mat(0);             // (nT + B) x N
            arma::mat psi_filter = psi_tmp.tail_rows(dim.nT + 1);       // (nT + 1) x N

            return psi_filter; // (nT + 1) x N
        }

        arma::mat get_psi_smooth()
        {
            arma::mat psi_tmp = Theta_smooth.row_as_mat(0); // (nT + B) x M
            arma::mat psi_smooth = psi_tmp.tail_rows(dim.nT + 1);
            return psi_smooth;
            // return psi_smooth.tail_rows(dim.nT + 1); // (nT + 1) x M
        }

        arma::vec get_mu0()
        {
            if (!dim.regressor_baseline)
            {
                throw std::invalid_argument("SMC::SequentialMonteCarlo::get_mu0 - Error: mu0 is not included in the state vector");
            }
            arma::vec mu0 = arma::vectorise(Theta_smooth.slice(Theta_smooth.n_slices - 1).row(Theta_smooth.n_rows - 1));
            return mu0;   
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

        

        

    /**
     * @brief Propagate from now theta[t] to theta[t + 1] following the evolution distribution. Theta_now = Theta_stored.slice(t + B - 1), Theta_new = Theta_stored.slice(t + B). Input: theta[t] (old theta at t), variance W[t] (true value or estimated value at time t), previous observation y[0:t], new observation theta[t + 1]. Output: theta[t + 1] (new theta at t + 1) and corresponding importance weights.
     * 
     * @param ynow y[t], where t is the index the old time.
     * @param Wsqrt N x 1
     * @param model 
     * @return arma::mat theta[t + 1]
     */
        static arma::mat propagate(
            const double &ynow, // y[t]
            const arma::vec &Wsqrt, // N x 1
            const arma::mat &Theta_now, // p x N
            const Model &model,
            const bool &positive_noise = false
        )
        {
            arma::mat Theta_next = Theta_now; // For t + B

            for (unsigned int i = 0; i < Theta_now.n_cols; i++)
            {
                arma::vec theta_now = Theta_now.col(i); // p x 1

                arma::vec theta_next = StateSpace::func_state_propagate(
                    model, theta_now, ynow, Wsqrt.at(i), positive_noise);

                Theta_next.col(i) = theta_next;
            }

            return Theta_next;
        } // func propagate

        static arma::mat propagate(
            const double &ynow,
            const double &Wsqrt,
            const arma::mat &Theta_now, // p x N
            const Model &model,
            const bool &positive_noise = false)
        {
            arma::mat Theta_next = Theta_now; // For t + B

            for (unsigned int i = 0; i < Theta_now.n_cols; i++)
            {
                arma::vec theta_now = Theta_now.col(i); // p x 1

                arma::vec theta_next = StateSpace::func_state_propagate(
                    model, theta_now, ynow, Wsqrt, positive_noise);

                Theta_next.col(i) = theta_next;
            }

            return Theta_next;
        } // func propagate

        static arma::vec imp_weights_likelihood(
            const unsigned int &t_next, // (t + 1)
            const arma::mat &Theta_next, // p x N
            const arma::vec &yall,
            const Model &model
        )
        {
            unsigned int N = Theta_next.n_cols;
            arma::vec weights(N, arma::fill::zeros);
            double mu0 = 0.;
            if (!model.dim.regressor_baseline) { mu0 = model.dobs.par1; }

            for (unsigned int i = 0; i < N; i++)
            {
                arma::vec theta_next = Theta_next.col(i);
                double ft = StateSpace::func_ft(model, t_next, theta_next, yall); // use y[t], ..., y[t + 1 - nelem]
                double lambda = LinkFunc::ft2mu(ft, model.flink.name, mu0); // conditional mean of the observations

                weights.at(i) = ObsDist::loglike(
                    yall.at(t_next),
                    model.dobs.name,
                    lambda, model.dobs.par2,
                    false);
            }

            return weights;            
        }

        static arma::vec imp_weights_likelihood(
            const unsigned int &t_next,  // (t + 1)
            const arma::mat &Theta_next, // p x N
            const arma::vec &mu0, // N x 1
            const arma::vec &yall,
            const Model &model)
        {
            unsigned int N = Theta_next.n_cols;
            arma::vec weights(N, arma::fill::zeros);

            for (unsigned int i = 0; i < N; i++)
            {
                arma::vec theta_next = Theta_next.col(i);
                double ft = StateSpace::func_ft(model, t_next, theta_next, yall); // use y[t], ..., y[t + 1 - nelem]
                double lambda = LinkFunc::ft2mu(ft, model.flink.name, mu0.at(i));       // conditional mean of the observations

                weights.at(i) = ObsDist::loglike(
                    yall.at(t_next),
                    model.dobs.name,
                    lambda, model.dobs.par2,
                    false);
            }

            return weights;
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
         * @param t_new Index of the time that is being predicted. The following inputs come from time t_old = t-1.
         * @param Theta p x N, { theta[t-1, i], i = 1, ..., N }, samples of latent states at t-1
         * @param W N x 1, { W[t-1, i], i = 1, ..., N }, samples of latent variance at t-1
         * @param mu0 N x 1, { mu0[t-1, i], i = 1, ..., N }, samples of baseline at t-1
         * @param mt_old p x N, m[t-1]. Assume no static parameters involved in ft.
         * @param yhat y[t] after transformation.
         * @return arma::vec
         */
        static arma::vec imp_weights_forecast(
            arma::mat &mu, // p x N
            arma::cube &Prec, // p x p x N
            arma::cube &Sigma_chol, // p x p x N
            arma::vec &logq, // N x 1
            arma::uvec &updated,
            const Model &model,
            const unsigned int &t_new, // current time. The following inputs come from time t-1.
            const arma::mat &Theta_old, // p x N, {theta[t-1]}
            const arma::vec &W_old, // N x 1, {W[t-1]} samples of latent variance
            const arma::vec &mu0_old, // N x 1, samples of baseline
            const arma::vec &y,
            const arma::vec &yhat // y[t]
        )
        {
            unsigned int N = Theta_old.n_cols;
            unsigned int nP = Theta_old.n_rows;
            double y_old = y.at(t_new - 1);
            double yhat_new = yhat.at(t_new);
            arma::mat Ieps(model.dim.nP, model.dim.nP, arma::fill::eye);
            Ieps.diag().fill(EPS);

            mu.set_size(nP, N);
            mu.zeros();

            Prec = arma::zeros<arma::cube>(nP, nP, N);
            Sigma_chol = Prec;

            for (unsigned int i = 0; i < N; i++)
            {
                arma::vec gtheta_old_i = StateSpace::func_gt(model, Theta_old.col(i), y_old); // gt(theta[t-1, i])
                double ft_gtheta = StateSpace::func_ft(model, t_new, gtheta_old_i, y); // ft( gt(theta[t-1,i]) )
                arma::vec Ft_gtheta = LBA::func_Ft(model, t_new, gtheta_old_i, y);     // Ft evaluated at a[t_new]
                double ft_tilde = ft_gtheta - arma::as_scalar(Ft_gtheta.t() * gtheta_old_i); // (eq 3.8)

                double eta = mu0_old.at(i) + ft_gtheta;
                double lambda = LinkFunc::ft2mu(eta, model.flink.name, 0.); // (eq 3.10)
                double Vt = ApproxDisturbance::func_Vt_approx(
                    lambda, model.dobs, model.flink.name); // (eq 3.11)

                double delta = yhat_new - mu0_old.at(i) - ft_tilde; // (eq 3.16)
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
                    arma::mat Prec_i = FFt_norm + Ieps; // nP x nP, function of mu0[i, t]
                    Prec_i.at(0, 0) += 1. / W_old.at(i); // (eq 3.21)

                    Prec_i = arma::symmatu(Prec_i);
                    Prec.slice(i) = Prec_i;
                    double ldetPrec = arma::log_det_sympd(Prec_i);

                    arma::mat Rchol, Sigma_i;
                    try
                    {
                        Sigma_i = inverse(Rchol, Prec_i);
                    }
                    catch(const std::exception& e)
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
                    loglik *= -0.5; // (eq 3.24 - 3.25)

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

            double logm = arma::mean(logq);
            if (logq.max() > 100)
            {
                double diff = logq.max() - 100;
                logm = std::max(logm, diff);
            }
            logq.for_each([&logm](arma::vec::elem_type &val){val -= logm;});
            arma::vec weights = arma::exp(logq);

            try
            {
                bound_check<arma::vec>(weights, "imp_weights_forecast");
            }
            catch(const std::exception& e)
            {
                logq.t().brief_print("\n logarithm of weights: ");
                throw std::runtime_error(e.what());
            }
            
            return weights;
        } // func: imp_weights_forecast

        static arma::uvec get_resample_index( const arma::vec &weights_in)
        {
            unsigned int N = weights_in.n_elem;
            double wsum = arma::accu(weights_in);

            arma::uvec resample_idx = arma::regspace<arma::uvec>(0, 1, N - 1);

            if (wsum > EPS)
            {
                arma::vec weights = weights_in / wsum; // N x 1
                resample_idx = sample(N, N, weights, true, true);
            }

            return resample_idx;
        }

        static arma::uvec get_smooth_index(
            const arma::rowvec &psi_smooth_now,  // 1 x M, Theta_smooth.slice(t).row(0)
            const arma::rowvec &psi_filter_prev, // 1 x N, Theta_stored.slice(t - 1).row(0)
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
            arma::cube th_filter = Theta_stored.tail_slices(dim.nT + 1); // p x N x (nT + 1)
            Rcpp::List out2 = StateSpace::forecast_error(th_filter, y, model, loss_func, k);

            return out2;
        }

        void forecast_error(double &err, double &cov, double &width, const Model &model, const std::string &loss_func = "quadratic")
        {
            arma::cube theta_tmp = Theta_stored.tail_slices(dim.nT + 1); // p x N x (nT + 1)
            StateSpace::forecast_error(err, cov, width, theta_tmp, y, model, loss_func);
            return;
        }

        Rcpp::List fitted_error(const Model &model, const std::string &loss_func = "quadratic")
        {
            Rcpp::List out3;

            arma::cube theta_tmp = Theta_stored.tail_slices(dim.nT + 1); // p x N x (nT + 1)
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
                theta_tmp = Theta_stored.tail_slices(dim.nT + 1); // p x N x (nT + 1)
            }

            StateSpace::fitted_error(err, theta_tmp, y, model, loss_func);
            return;
        }

        Dim dim;

        arma::vec y, Wt; // (nT + 1) x 1
        arma::vec weights, lambda; // N x 1
        arma::vec log_cond_marginal;

        arma::cube Theta_stored; // p x N x (nT + B)
        arma::cube Theta_smooth; // p x M x (nT + B)
        // arma::mat psi_smooth; // (nT + B) x M

        unsigned int N = 1000;
        unsigned int M = 500;
        unsigned int B = 10;
        unsigned int nforecast = 0;

        Prior prior_W;
        arma::vec W_filter;

        Prior prior_mu0;
        arma::vec mu0_filter;

        bool use_discount = false;
        bool use_custom = false;
        double custom_discount_factor = 0.95;
        const double default_discount_factor = 0.99;

        bool smoothing = true;

    }; // class Sequential Monte Carlo


    class MCS : public SequentialMonteCarlo
    {
    public:
        MCS(
            const Model &dgtf_model,
            const arma::vec &y_in) : SequentialMonteCarlo(dgtf_model, y_in) { smoothing = false; }
        
        Rcpp::List get_output(const bool &summarize = true)
        {
            arma::vec ci_prob = {0.025, 0.5, 0.975};
            Rcpp::List output;
            arma::mat psi_filter = get_psi_filter();
            arma::mat psi_smooth = get_psi_smooth();

            if (DEBUG)
            {
                output["Theta_filter"] = Rcpp::wrap(Theta_stored);
                output["Theta"] = Rcpp::wrap(Theta_smooth);
            }

            if (summarize)
            {
                arma::mat psi1 = arma::quantile(psi_filter, ci_prob, 1); // (nT + 1) x 3
                output["psi_filter"] = Rcpp::wrap(psi1);

                arma::mat psi2 = arma::quantile(psi_smooth, ci_prob, 1); // (nT + 1) x 3
                output["psi"] = Rcpp::wrap(psi2);
            }
            else
            {
                output["psi_filter"] = Rcpp::wrap(psi_filter);
                output["psi"] = Rcpp::wrap(psi_smooth);
            }

            if (dim.regressor_baseline)
            {
                arma::vec mu0 = arma::vectorise(Theta_smooth.slice(Theta_smooth.n_slices - 1).row(Theta_smooth.n_rows - 1));
                output["mu0"] = Rcpp::wrap(mu0);
            }


            output["log_marginal_likelihood"] = marginal_likelihood(log_cond_marginal, true);

            return output;
        }


        void init(const Rcpp::List &opts_in)
        {
            Rcpp::List opts = opts_in;
            SequentialMonteCarlo::init(opts);

            prior_W.infer = false;
            prior_mu0.infer = false;

            Wt.set_size(dim.nT + 1);
            Wt.fill(prior_W.val);

            smoothing = true;

            Theta_smooth.clear();
            Theta_smooth = Theta_stored;
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
            Rcpp::List out = StateSpace::forecast(y, Theta_stored, Wtmp, model, nforecast);
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
                arma::cube theta_tmp = Theta_stored.tail_slices(model.dim.nT + 1);

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
                arma::cube theta_tmp = Theta_stored.tail_slices(model.dim.nT + 1);

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

                Theta_stored.clear();
                Theta_stored.set_size(model.dim.nP, N, model.dim.nT + B);
                Theta_stored.zeros();

                Theta_smooth.clear();
                Theta_smooth = Theta_stored;

                infer(model);
                arma::cube theta_tmp = Theta_stored.tail_slices(model.dim.nT + 1);

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

        //     arma::cube theta_tmp = Theta_stored.tail_slices(model.dim.nT + 1);

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

        //     arma::cube theta_tmp = Theta_stored.tail_slices(model.dim.nT + 1);

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
            // y: (nT + 1) x 1
            for (unsigned int t = 0; t < dim.nT; t++)
            {
                Rcpp::checkUserInterrupt();

                if (use_discount)
                { // Use discount factor if W is not given
                    bool use_custom_val = (use_custom && t > B) ? true : false;
                    prior_W.val = SequentialMonteCarlo::discount_W(
                        Theta_smooth.slice(t + B - 1),
                        custom_discount_factor,
                        use_custom_val, 
                        default_discount_factor);
                }

                Wt.at(t + 1) = prior_W.val;

                arma::vec Wsqrt(N, arma::fill::zeros);
                Wsqrt.fill(std::sqrt(Wt.at(t + 1)) + EPS);


                bound_check<arma::vec>(Wsqrt, "SMC::propagate: Wsqrt", true, true);

                arma::mat Theta_now = Theta_smooth.slice(t + B - 1);

                bool positive_noise = (t < Theta_now.n_rows) ? true : false;
                arma::mat Theta_next = propagate(y.at(t), Wsqrt, Theta_now, model, positive_noise);
                weights = imp_weights_likelihood(t + 1, Theta_next, y, model);
                log_cond_marginal.at(t + 1) = log_conditional_marginal(weights);

                arma::uvec resample_idx = get_resample_index(weights);

                Theta_smooth.slice(t + B) = Theta_next;

                for (unsigned int b = t + 1; b < t + B + 1; b++)
                {
                    arma::mat theta_tmp = Theta_smooth.slice(b);
                    Theta_smooth.slice(b) = theta_tmp.cols(resample_idx);
                }

                Theta_stored.slice(t + 1) = Theta_next.cols(resample_idx);



                // Theta_stored.slice(t + B) = Theta_next.cols(resample_idx);

                // propagate(t, Wsqrt, model);
                // // arma::uvec resample_idx = resample(t);
                // arma::uvec resample_idx = SequentialMonteCarlo::get_resample_index(weights);
                // resample_theta(t, resample_idx);

                if (verbose)
                {
                    Rcpp::Rcout << "\rProgress: " << t + 1 << "/" << dim.nT;
                }

            } // loop over time

            if (verbose)
            {
                Rcpp::Rcout << std::endl;
            }
        }


    };


    class FFBS : public SequentialMonteCarlo
    {
    public:
        FFBS(
            const Model &dgtf_model,
            const arma::vec &y_in) : SequentialMonteCarlo(dgtf_model, y_in) {}

        void init(const Rcpp::List &opts_in)
        {
            Rcpp::List opts = opts_in;
            SequentialMonteCarlo::init(opts);

            prior_W.infer = false;
            prior_mu0.infer = false;
            B = 1;

            Wt.set_size(dim.nT + 1);
            Wt.fill(prior_W.val);
        }

        Rcpp::List get_output(const bool &summarize = true)
        {
            arma::vec ci_prob = {0.025, 0.5, 0.975};
            Rcpp::List output;
            if (smoothing)
            {
                arma::mat psi_filter = get_psi_filter();
                arma::mat psi_smooth = get_psi_smooth();
                if (summarize)
                {
                    arma::mat psi_f = arma::quantile(psi_filter, ci_prob, 1);
                    arma::mat psi = arma::quantile(psi_smooth, ci_prob, 1);
                    output["psi_filter"] = Rcpp::wrap(psi_f);
                    output["psi"] = Rcpp::wrap(psi);
                }
                else
                {
                    output["psi_filter"] = Rcpp::wrap(psi_filter);
                    output["psi"] = Rcpp::wrap(psi_smooth);
                }

                if (dim.regressor_baseline)
                {
                    arma::mat mu0_filter = Theta_stored.row_as_mat(dim.nP - 1);
                    output["mu0_filter"] = Rcpp::wrap(mu0_filter);

                    arma::mat mu0_smooth = Theta_smooth.row_as_mat(dim.nP - 1);
                    output["mu0"] = Rcpp::wrap(mu0_smooth);
                }
            }
            else
            {
                arma::mat psi = get_psi_filter();
                if (summarize)
                {
                    arma::mat psi_f = arma::quantile(psi, ci_prob, 1);
                    output["psi"] = Rcpp::wrap(psi_f);
                }
                else
                {
                    output["psi"] = Rcpp::wrap(psi);
                }

                if (dim.regressor_baseline)
                {
                    arma::mat mu0_filter = Theta_stored.row_as_mat(dim.nP - 1);
                    output["mu0"] = Rcpp::wrap(mu0_filter);
                }
            }

            output["log_marginal_likelihood"] = marginal_likelihood(log_cond_marginal, true);

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
                out = StateSpace::forecast(y, Theta_stored, Wtmp, model, nforecast);
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
                arma::cube theta_tmp = Theta_stored.tail_slices(model.dim.nT + 1);

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
                arma::cube theta_tmp = Theta_stored.tail_slices(model.dim.nT + 1);

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

        void filter(const Model &model, const bool &verbose = VERBOSE)
        {
            arma::vec yhat = y; // (nT + 1) x 1
            for (unsigned int t = 0; t < dim.nT; t++)
            {
                yhat.at(t) = LinkFunc::mu2ft(y.at(t), model.flink.name, 0.);
            }

            mu0_filter.fill(model.dobs.par1); // N x 1

            if (arma::any(W_filter < EPS))
            {
                throw std::invalid_argument("FFBS::filter: W_filter should not be zero.");
            }

            // y: (nT + 1) x 1

            for (unsigned int t = 0; t < dim.nT; t++)
            {
                Rcpp::checkUserInterrupt();

                unsigned int t_old = t;
                unsigned int t_new = t + 1;
                arma::mat Theta_old = Theta_stored.slice(t_old); // p x N, theta[t]

                arma::vec logq(N, arma::fill::zeros);
                arma::mat mu; // nP x N
                arma::cube Prec, Sigma_chol; // nP x nP x N
                arma::uvec updated(N, arma::fill::zeros);

                weights = imp_weights_forecast(
                    mu, Prec, Sigma_chol, logq, updated, // sufficient statistics
                    model, t_new, 
                    Theta_old, // theta needs to be resampled
                    W_filter, mu0_filter, y, yhat);
                // eff.at(t, 0) = effective_sample_size(weights, false);

                arma::uvec resample_idx = get_resample_index(weights);

                Theta_old = Theta_old.cols(resample_idx);
                Theta_stored.slice(t) = Theta_old;

                // Sufficient statistics for theta
                mu = mu.cols(resample_idx);
                Prec = Prec.slices(resample_idx);
                Sigma_chol = Sigma_chol.slices(resample_idx);
                logq = logq.elem(resample_idx);
                updated = updated.elem(resample_idx);


                if (use_discount)
                { // Use discount factor if W is not given
                    bool use_custom_val = (use_custom && t > 1) ? true : false;
                    prior_W.val = SequentialMonteCarlo::discount_W(
                        Theta_stored.slice(t),
                        custom_discount_factor,
                        use_custom_val,
                        default_discount_factor);
                }

                Wt.at(t + 1) = prior_W.val;
                W_filter.fill(prior_W.val);


                // Propagate
                arma::mat Theta_new(dim.nP, N, arma::fill::zeros);
                arma::vec logp(N, arma::fill::zeros);
                for (unsigned int i = 0; i < N; i ++)
                {
                    arma::vec theta_new = mu.col(i) + Sigma_chol.slice(i).t() * arma::randn(model.dim.nP); // nP
                    Theta_new.col(i) = theta_new;

                    double ft = StateSpace::func_ft(model, t_new, theta_new, y);
                    double lambda = LinkFunc::ft2mu(ft, model.flink.name, mu0_filter.at(i));
                    double logp_tmp = R::dnorm4(theta_new.at(0), Theta_old.at(0, i), std::sqrt(W_filter.at(i)), true);

                    if (updated.at(i) == 1)
                    {
                        logq.at(i) += MVNorm::dmvnorm2(theta_new, mu.col(i), Prec.slice(i), true); // sample from posterior
                    }
                    else
                    {
                        logq.at(i) += logp_tmp; // sample from evolution distribution
                    }

                    logp.at(i) = logp_tmp;
                    logp.at(i) += ObsDist::loglike(y.at(t_new), model.dobs.name, lambda, model.dobs.par2, true);
                    

                    // weights.at(i) = logp.at(i) - logq.at(i);
                    weights.at(i) = logp.at(i);
                    weights.at(i) = std::exp(weights.at(i));
                }

                log_cond_marginal.at(t + 1) = log_conditional_marginal(weights);
                resample_idx = get_resample_index(weights);

                Theta_stored.slice(t + 1) = Theta_new.cols(resample_idx);

                if (verbose)
                {
                    Rcpp::Rcout << "\rFiltering: " << t + 1 << "/" << dim.nT;
                }
            }

            if (verbose)
            {
                Rcpp::Rcout << std::endl;
            }

            return;
        }

        void smoother(const Model &model, const bool &verbose = VERBOSE)
        {
            arma::uvec idx = sample(N, M, weights, true, true); // M x 1
            arma::mat theta_last = Theta_stored.slice(dim.nT);  // p x N
            arma::mat theta_sub = theta_last.cols(idx);         // p x M

            Theta_smooth.slice(dim.nT) = theta_sub;
            // psi_smooth.row(dim.nT) = theta_sub.row(0);

            for (unsigned int t = dim.nT; t > 0; t--)
            {
                Rcpp::checkUserInterrupt();

                double Wnow = Wt.at(t);
                double Wsd = std::sqrt(Wnow);
                arma::vec Wsqrt(M);
                Wsqrt.fill(Wsd);

                arma::rowvec psi_smooth_now = Theta_smooth.slice(t).row(0);                       // 1 x M
                arma::rowvec psi_filter_prev = Theta_stored.slice(t - 1).row(0);                  // 1 x N
                arma::uvec smooth_idx = get_smooth_index(psi_smooth_now, psi_filter_prev, Wsqrt); // M x 1

                arma::mat theta_next = Theta_stored.slice(t - 1);
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
        }
        
        void infer(const Model &model, const bool &verbose = VERBOSE)
        {
            filter(model, verbose);

            if (smoothing)
            {
                smoother(model, verbose);
            }

            return;
        }
    };


    class PL : public SequentialMonteCarlo
    {
    public:

        PL(const Model &dgtf_model, const arma::vec &y_in) : SequentialMonteCarlo(dgtf_model, y_in){}


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

            // mt.set_size(dim.nP, N);
            // mt.zeros();
            // Ct.set_size(dim.nP, dim.nP, N);
            // arma::mat Imat(dim.nP, dim.nP, arma::fill::eye);
            // Imat.for_each([](arma::mat::elem_type &val) {val *= 2;} );
            // Ct.each_slice([&Imat](arma::mat &cmat){ cmat = Imat; });

            // mt_next = mt;
            // Ct_next = Ct;

            init_param_stored(aw, bw, W_filter, W_smooth, dim.nT, N, M, prior_W);
            // W_filter = W_stored.col(0);
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

                W_filter = draw_param_init(dist_W_init, N, prior_W.val);
                // W_stored.col(0) = W_filter;
            }

            if (dim.regressor_baseline)
            {
                prior_mu0.infer = false;
                prior_mu0.val = 0.;
            }

            init_param_stored(amu, bmu, mu0_filter, mu0_smooth, dim.nT, N, M, prior_mu0);
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
                mu0_filter = draw_param_init(dist_mu0_init, N, prior_mu0.val);
            }
            
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

        static void init_param_stored(
            arma::vec &par1,
            arma::vec &par2,
            arma::vec &par_filter,
            arma::vec &par_smooth,
            const unsigned int &nT,
            const unsigned int N,
            const unsigned int M,
            const Dist &prior)
        {
            // par1.set_size(N, nT + 1); // N x (nT + 1)
            // par1.fill(prior.par1);
            // par2 = par1;
            // par2.fill(prior.par2);
            par1.set_size(N); // N x (nT + 1)
            par1.fill(prior.par1);
            par2 = par1;
            par2.fill(prior.par2);

            par_filter.set_size(N); // N x (nT + 1)
            par_filter.fill(prior.val);

            par_smooth.set_size(M); // M x 1
            par_smooth.zeros();

            return;
        }

        static Rcpp::List default_settings()
        {
            Rcpp::List opts;
            opts = SequentialMonteCarlo::default_settings();
            opts["max_iter"] = 10;

            Rcpp::List W_init_opts;
            W_init_opts["prior_name"] = "gamma";
            W_init_opts["prior_param"] = Rcpp::NumericVector::create(1., 1.);

            opts["W_init"] = W_init_opts;

            Rcpp::List mu0_init_opts;
            mu0_init_opts["prior_name"] = "constant";
            mu0_init_opts["prior_param"] = Rcpp::NumericVector::create(0., 0.);

            opts["mu0_init"] = mu0_init_opts;

            return opts;
        }


        Rcpp::List get_output(const bool &summarize = TRUE)
        {
            arma::vec ci_prob = {0.025, 0.5, 0.975};
            Rcpp::List output;

            if (smoothing)
            {
                arma::mat psi_filter = get_psi_filter(); // (nT + 1) x M
                arma::mat psi_smooth = get_psi_smooth(); // (nT + 1) x M
                
                if (summarize)
                {
                    arma::mat psi_f = arma::quantile(psi_filter, ci_prob, 1);
                    arma::mat psi = arma::quantile(psi_smooth, ci_prob, 1);
                    output["psi_filter"] = Rcpp::wrap(psi_f);
                    output["psi"] = Rcpp::wrap(psi);
                }
                else
                {
                    output["psi_filter"] = Rcpp::wrap(psi_filter);
                    output["psi"] = Rcpp::wrap(psi_smooth);
                }

                if (dim.regressor_baseline)
                {
                    arma::mat mu0_filter = Theta_stored.row_as_mat(dim.nP - 1);
                    output["mu0_filter"] = Rcpp::wrap(mu0_filter);

                    arma::mat mu0_smooth = Theta_smooth.row_as_mat(dim.nP - 1);
                    output["mu0"] = Rcpp::wrap(mu0_smooth);
                }
            }
            else
            {
                arma::mat psi = get_psi_filter();
                if (summarize)
                {
                    arma::mat psi_f = arma::quantile(psi, ci_prob, 1);
                    output["psi"] = Rcpp::wrap(psi_f);
                }
                else
                {
                    output["psi"] = Rcpp::wrap(psi);
                }

                if (dim.regressor_baseline)
                {
                    arma::mat mu0_filter = Theta_stored.row_as_mat(dim.nP - 1);
                    output["mu0"] = Rcpp::wrap(mu0_filter);
                }
            }

            if (prior_W.infer)
            {
                // arma::vec W_filter = W_stored.col(dim.nT);
                output["W"] = Rcpp::wrap(W_filter);
            }

            output["mu0"] = Rcpp::wrap(mu0_filter);

            output["log_marginal_likelihood"] = marginal_likelihood(log_cond_marginal, true);

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
                out = StateSpace::forecast(y, Theta_stored, W_filter, model, nforecast);
            }
            return out;
        }


        // /**
        //  * @brief Bugged. The propagation step is wrong. Filetering based on sufficient statistics for state and static parameters.
        //  * 
        //  * @param model 
        //  * @param verbose 
        //  */
        // void filter2(const Model &model, const bool &verbose = VERBOSE)
        // {
        //     // Index t: old
        //     // Index t + 1: to be propagated to.
        //     for (unsigned int t = 0; t < dim.nT; t ++)
        //     {
        //         Rcpp::checkUserInterrupt();

        //         arma::mat at_next(model.dim.nP, N, arma::fill::zeros);  // p x N
        //         arma::cube Rt_next = arma::zeros<arma::cube>(model.dim.nP, model.dim.nP, N); // p x p x N
        //         arma::mat Ft_next(model.dim.nP, N, arma::fill::zeros); // p x N

        //         arma::vec qt_prior_next(N, arma::fill::zeros); // N x 1
        //         arma::vec ft_prior_next(N, arma::fill::zeros); // N x 1

        //         arma::vec alpha_prior_next(N, arma::fill::zeros);
        //         arma::vec beta_prior_next(N, arma::fill::zeros);

        //         arma::vec Wtmp = W_stored.col(t); // N x 1, W[t]

        //         for (unsigned int i = 0; i < N; i ++)
        //         {
        //             at_next.col(i) = StateSpace::func_gt(model, mt.col(i), y.at(t)); // get a[t+1] | y[1:t]
        //             arma::mat Gt_next = LBA::func_Gt(model, mt.col(i), y.at(t)); // G[t + 1] | y[1:t]
        //             Rt_next.slice(i) = LBA::func_Rt(Gt_next, Ct.slice(i), Wtmp.at(i), false); // R[t + 1] | C[t], W[t], y[1:t]

        //             double ftmp = 0.; // ft_prior[t + 1] | y[1:t], a[t + 1]
        //             double qtmp = 0.; // qt_prior[t + 1] | y[1:t], R[t + 1], F[t + 1]
        //             arma::vec Ftmp = model.transfer.F0; // F[t + 1] | y[1:t]
        //             LBA::func_prior_ft(ftmp, qtmp, Ftmp, t + 1, model, y, at_next.col(i), Rt_next.slice(i));
        //             Ft_next.col(i) = Ftmp;
        //             ft_prior_next.at(i) = ftmp;
        //             qt_prior_next.at(i) = qtmp;

        //             double alpha_prior = 0.; // alpha_prior[t + 1] | y[1:t], prior
        //             double beta_prior = 0.;  // alpha_prior[t + 1] | y[1:t], prior
        //             LBA::func_alpha_beta(alpha_prior, beta_prior, model, ftmp, qtmp, 0., false);
        //             alpha_prior_next.at(i) = alpha_prior;
        //             beta_prior_next.at(i) = beta_prior;

        //             if (t > 0)
        //             {
        //                 weights.at(i) = ObsDist::dforecast(
        //                     y.at(t + 1), model.dobs.name, model.dobs.par2,
        //                     alpha_prior, beta_prior, false);
        //             }
        //         } // loop over i, weighted by marginal one-step-ahead forecasting

        //         // [do resample here]
        //         if (t > 0)
        //         {
        //             arma::uvec resample_idx = get_resample_index(weights);

        //             arma::mat Theta_now = Theta_stored.slice(t); // theta[t]
        //             Theta_now = Theta_now.cols(resample_idx);
        //             Theta_stored.slice(t) = Theta_now;

        //             mt = mt.cols(resample_idx); // m[t]
        //             Ct = Ct.slices(resample_idx); // C[t]

        //             at_next = at_next.cols(resample_idx); // a[t + 1]
        //             Rt_next = Rt_next.slices(resample_idx); // R[t + 1]
        //             Ft_next = Ft_next.cols(resample_idx); // F[t + 1]
        //             qt_prior_next = qt_prior_next.elem(resample_idx); // qt_prior[t + 1]
        //             ft_prior_next = ft_prior_next.elem(resample_idx); // ft_prior[t + 1]
        //             alpha_prior_next = alpha_prior_next.elem(resample_idx);
        //             beta_prior_next = beta_prior_next.elem(resample_idx);

                    
        //             if (prior_W.infer)
        //             {
        //                 arma::vec aw_tmp = aw.col(t);
        //                 arma::vec bw_tmp = bw.col(t);
        //                 aw.col(t) = aw_tmp.elem(resample_idx);
        //                 bw.col(t) = bw_tmp.elem(resample_idx);

        //                 W_stored.col(t) = Wtmp.elem(resample_idx); // variance, resampled
        //             }
        //         } // resample


        //         bool burnin = (t <= std::min(0.1 * dim.nT, 20.)) ? true : false;
        //         for (unsigned int i = 0; i < N; i ++)
        //         {
        //             /**
        //              * @brief State propagation
        //              * 
        //              */
        //             arma::mat At = LBA::func_At(Rt_next.slice(i), Ft_next.col(i), qt_prior_next.at(i)); // A[t + 1]

        //             double alpha_posterior_next = alpha_prior_next.at(i) + y.at(t + 1);
        //             double beta_posterior_next = beta_prior_next.at(i) + 1.;

        //             double ft_posterior_next = 0.; // ft_posterior[t + 1] | y[1:t+1]
        //             double qt_posterior_next = 0.; // qt_posterior[t + 1] | y[1:t+1]
        //             LBA::func_posterior_ft(ft_posterior_next, qt_posterior_next, model, alpha_posterior_next, beta_posterior_next);

        //             // Sufficient statistics for the filtering distribution of latent states
        //             mt_next.col(i) = LBA::func_mt(at_next.col(i), At, ft_prior_next.at(i), ft_posterior_next); // m[t + 1] | y[1:t+1]
        //             Ct_next.slice(i) = LBA::func_Ct(Rt_next.slice(i), At, qt_prior_next.at(i), qt_posterior_next);        // C[t + 1] | y[1:t+1]

        //             // arma::mat Ct_chol = arma::chol(arma::symmatu(Ct.slice(i))); // upper triangular
        //             // arma::vec theta = mt.col(i) + Ct_chol.t() * arma::randn(model.dim.nP); // theta[t + 1]
        //             // Theta_stored.slice(t + 1).col(i) = theta;
        //             Theta_stored.slice(t + 1).col(i) = mt_next.col(i);

        //             double err = Theta_stored.at(0, i, t + 1) - Theta_stored.at(0, i, t);
        //             if (prior_W.infer)
        //             {
        //                 double sse = err * err;
        //                 aw.at(i, t + 1) = aw.at(i, t) + 0.5;
        //                 bw.at(i, t + 1) = bw.at(i, t) + 0.5 * sse;
        //             }

        //             double wtmp = prior_W.val;
        //             if ((prior_W.infer && burnin) || use_discount)
        //             {
        //                 wtmp = SequentialMonteCarlo::discount_W(
        //                     Theta_stored.slice(t),
        //                     custom_discount_factor,
        //                     use_custom, default_discount_factor);
        //             }

        //             if (prior_W.infer && !burnin)
        //             {
        //                 // Wt.at(i, t + 1) = 1. / R::rgamma(aw.at(i, t + 1), 1. / bw.at(i, t + 1));
        //                 W_stored.at(i, t + 1) = InverseGamma::sample(aw.at(i, t + 1), bw.at(i, t + 1));
        //             }
        //             else
        //             {
        //                 W_stored.at(i, t + 1) = wtmp;
        //             } // inference of W

        //         } // loop over i

        //         if (verbose)
        //         {
        //             Rcpp::Rcout << "\rFiltering: " << t + 1 << "/" << dim.nT;
        //         }

        //     } // loop over time t

        //     if (verbose)
        //     {
        //         Rcpp::Rcout << std::endl;
        //     }

        //     return;
        // }


        void filter(const Model &model, const bool &verbose = VERBOSE)
        {
            std::map<std::string, AVAIL::Func> link_list = AVAIL::link_list;

            arma::vec yhat = y; // (nT + 1) x 1
            for (unsigned int t = 0; t < dim.nT; t++)
            {
                yhat.at(t) = LinkFunc::mu2ft(y.at(t), model.flink.name, 0.);
            }


            for (unsigned int t = 0; t < dim.nT; t++)
            {
                Rcpp::checkUserInterrupt();

                bool burnin = (t <= std::min(0.1 * dim.nT, 20.)) ? true : false;

                unsigned int t_old = t;
                unsigned int t_new = t + 1;

                // propagate(t, Wsqrt, model); // step 1 (a)
                // arma::uvec resample_idx = resample(t); // step 1 (b)
                arma::mat Theta_old = Theta_stored.slice(t_old); // p x N, theta[t]

                /**
                 * @brief Resampling using conditional one-step-ahead predictive distribution as weights.
                 *
                 */
                
                arma::vec logq(N, arma::fill::zeros);
                arma::mat mu; // nP x N
                arma::cube Prec, Sigma_chol; // nP x nP x N
                arma::uvec updated;
                weights = imp_weights_forecast(
                    mu, Prec, Sigma_chol, logq, updated, 
                    model, t_new, 
                    Theta_old, 
                    W_filter, mu0_filter, y, yhat);
                arma::uvec resample_idx = get_resample_index(weights);

                Theta_old = Theta_old.cols(resample_idx); // theta[t]
                Theta_stored.slice(t_old) = Theta_old;

                updated = updated.elem(resample_idx);
                logq = logq.elem(resample_idx);
                mu = mu.cols(resample_idx);
                Prec = Prec.slices(resample_idx);
                Sigma_chol = Sigma_chol.slices(resample_idx);

                if (prior_W.infer)
                {
                    W_filter = W_filter.elem(resample_idx); // gamma[t]
                    aw = aw.elem(resample_idx);             // s[t]
                    bw = bw.elem(resample_idx);             // s[t]
                }

                if (prior_mu0.infer)
                {
                    mu0_filter = mu0_filter.elem(resample_idx); // gamma[t]
                    amu = amu.elem(resample_idx);               // s[t]
                    bmu = bmu.elem(resample_idx);               // s[t]
                }

                // arma::vec Wsqrt = arma::sqrt(W_stored.col(t));
                arma::vec Wsqrt = arma::sqrt(W_filter);
                bool positive_noise = (t_old < Theta_old.n_rows) ? true : false;


                // NEED TO CHANGE PROPAGATE STEP
                // arma::mat Theta_new = propagate(y.at(t_old), Wsqrt, Theta_old, model, positive_noise);
                arma::mat Theta_new(model.dim.nP, N, arma::fill::zeros);
                arma::vec logp(N, arma::fill::zeros);
                for (unsigned int i = 0; i < N; i ++)
                {
                    arma::vec theta_new = mu.col(i) + Sigma_chol.slice(i).t() * arma::randn(model.dim.nP);
                    logq.at(i) += MVNorm::dmvnorm2(theta_new, mu.col(i), Prec.slice(i), true);
                    Theta_new.col(i) = theta_new;

                    double wtmp = 0.;
                    if ((prior_W.infer && burnin) || use_discount)
                    {
                        wtmp = SequentialMonteCarlo::discount_W(
                            Theta_stored.slice(t),
                            custom_discount_factor,
                            use_custom, default_discount_factor);
                    }

                    if (prior_W.infer)
                    {
                        double err = theta_new.at(0) - Theta_old.at(0, i);
                        double sse = std::pow(err, 2.);

                        aw.at(i) += 0.5;
                        bw.at(i) += 0.5 * sse;
                        if (!burnin)
                        {
                            // Wt.at(i, t + 1) = 1. / R::rgamma(aw.at(i, t + 1), 1. / bw.at(i, t + 1));
                            // W_stored.at(i, t + 1) = InverseGamma::sample(aw.at(i, t + 1), bw.at(i, t + 1));
                            // W_filter.at(i) = InverseGamma::sample(aw.at(i, t + 1), bw.at(i, t + 1));
                            wtmp = InverseGamma::sample(aw.at(i), bw.at(i));
                            logq.at(i) += R::dgamma(1./wtmp, aw.at(i), 1./bw.at(i), true);
                        }                     
                    } // Propagate W
                    W_filter.at(i) = wtmp;

                    double ft_new = StateSpace::func_ft(model, t_new, theta_new, y); // ft(theta[t+1])
                    double lambda_old = LinkFunc::ft2mu(ft_new, model.flink.name, mu0_filter.at(i)); // ft_new from time t + 1, mu0_filter from time t (old).

                    if (prior_mu0.infer)
                    {
                        double Vt_old = ApproxDisturbance::func_Vt_approx(lambda_old, model.dobs, model.flink.name);
                        double amu_scl = amu.at(i) / bmu.at(i);
                        double prec = 1. / bmu.at(i) + 1. / Vt_old;
                        bmu.at(i) = 1. / prec;

                        // double nom = bmu.at(i, t) * amu.at(i, t); // prior precision x mean
                        double eps = yhat.at(t_new) - ft_new;
                        double eps_scl = eps / Vt_old;
                        amu.at(i) = bmu.at(i) * (amu_scl + eps_scl);

                        double sd = std::sqrt(bmu.at(i));
                        double mu0 = R::rnorm(amu.at(i), sd);

                        if (link_list[model.flink.name] == AVAIL::Func::identity)
                        {
                            unsigned int cnt = 0;
                            while (mu0 < 0 && cnt < max_iter)
                            {
                                mu0 = R::rnorm(amu.at(i), sd);
                                cnt++;
                            }

                            if (mu0 < 0)
                            {
                                throw std::invalid_argument("SMC::PL::filter - negative mu0 when using identity link.");
                            }
                        }

                        mu0_filter.at(i) = mu0;
                        logq.at(i) += R::dnorm4(mu0, amu.at(i), sd, true);
                    } // inference of mu0

                    double lambda_new = LinkFunc::ft2mu(ft_new, model.dobs.name, mu0_filter.at(i));
                    logp.at(i) = ObsDist::loglike(
                        y.at(t_new), model.dobs.name, lambda_new, model.dobs.par2, true); // observation density
                    logp.at(i) += R::dnorm4(theta_new.at(0), Theta_old.at(0,i), std::sqrt(W_filter.at(i)), true);
                    
                    weights.at(i) = std::exp(logp.at(i) - logq.at(i));
                } // loop over i, index of particles
                
             

                // NEED TO CHANGE IMPORTANCE WEIGHT

                // weights = imp_weights_likelihood(t_new, Theta_new, mu0_filter, y, model);
                log_cond_marginal.at(t_new) = log_conditional_marginal(weights);
                resample_idx = get_resample_index(weights);
                Theta_new = Theta_new.cols(resample_idx);

                if (prior_W.infer)
                {
                    W_filter = W_filter.elem(resample_idx);
                    aw = aw.elem(resample_idx);
                    bw = bw.elem(resample_idx);
                }

                if (prior_mu0.infer)
                {
                    mu0_filter = mu0_filter.elem(resample_idx);
                    amu = amu.elem(resample_idx);
                    bmu = bmu.elem(resample_idx);
                }


                

                if (verbose)
                {
                    Rcpp::Rcout << "\rFiltering: " << t + 1 << "/" << dim.nT;
                }

            } // propagate and resample

            if (verbose)
            {
                Rcpp::Rcout << std::endl;
            }
        }


        void smoother(const Model &model, const bool &verbose = VERBOSE)
        {
            weights.ones();
            arma::uvec idx = sample(N, M, weights, false, true); // M x 1

            arma::mat theta_tmp = Theta_stored.slice(dim.nT); // p x N
            arma::mat theta_tmp2 = theta_tmp.cols(idx);       // p x M
            Theta_smooth.slice(dim.nT) = theta_tmp2;          // p x M

            // arma::vec ptmp0 = arma::vectorise(Theta_stored.slice(dim.nT).row(0)); // p x N
            // arma::vec ptmp = ptmp0.elem(idx);

            // psi_smooth.row(dim.nT) = ptmp.t();

            // arma::vec wtmp = W_stored.col(dim.nT);
            // W_smooth.col(dim.nT) = wtmp.elem(idx);
            W_smooth = W_filter.elem(idx);              // M x 1
            arma::vec Wsqrt = arma::sqrt(W_smooth); // M x 1

            for (unsigned int t = dim.nT; t > 0; t--)
            {
                Rcpp::checkUserInterrupt();

                // arma::vec Wtmp0 = W_stored.col(t - 1); // N x 1
                // arma::vec Wtmp = Wtmp0.elem(idx); // M x 1
                // arma::vec Wsqrt = arma::sqrt(W_stored.col(t - 1)); // M x 1

                // arma::uvec smooth_idx = get_smooth_index(t, Wsqrt, idx);
                arma::rowvec psi_smooth_now = Theta_smooth.slice(t).row(0);                       // 1 x M
                arma::rowvec psi_filter_prev = Theta_stored.slice(t - 1).row(0);                  // 1 x N
                arma::uvec smooth_idx = get_smooth_index(psi_smooth_now, psi_filter_prev, Wsqrt); // M x 1

                // if (infer_W)
                // {
                //     W_smooth.col(t - 1) = Wtmp.elem(smooth_idx);
                // }

                arma::mat theta_tmp0 = Theta_stored.slice(t - 1); // p x N
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
        }


        void infer(const Model &model, const bool &verbose = VERBOSE)
        {
            if (!prior_mu0.infer)
            {
                prior_mu0.val = model.dobs.par1;
                mu0_filter.fill(prior_mu0.val);
            }

            if (!prior_W.infer)
            {
                prior_W.val = model.derr.par1;
                W_filter.fill(prior_W.val);
            }

            filter(model, verbose);


            if (smoothing)
            {
                smoother(model, verbose);

            } // opts.smoothing
        } // Particle Learning inference

    
    private:
        // arma::mat mt; // p x N
        // arma::cube Ct; // p x p x N
        // arma::mat mt_next; // p x N
        // arma::cube Ct_next; // p x p x N

        // arma::mat aw; // N x (nT + 1), shape of IG
        // arma::mat bw; // N x (nT + 1), scale of IG (i.e. rate of corresponding Gamma)
        arma::vec aw; // N x 1, shape of IG
        arma::vec bw; // N x 1, scale of IG (i.e. rate of corresponding Gamma)
        // arma::mat W_stored; // N x (nT + 1)
        arma::vec W_filter; // N x 1
        arma::vec W_smooth; // M x 1

        // arma::mat amu; // N x (nT + 1), mean of normal
        // arma::mat bmu; // N x (nT + 1), precision of normal
        arma::vec amu; // N x 1, mean of normal
        arma::vec bmu; // N x 1, precision of normal
        // arma::mat mu0_stored; // N x (nT + 1)
        arma::vec mu0_filter; // N x 1
        arma::vec mu0_smooth; // M x 1

        unsigned int max_iter = 10;
    };
}


#endif