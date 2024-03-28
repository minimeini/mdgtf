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
            if (settings.containsElementNamed("W"))
            {

                init_param_def(settings, par_name, prior_W);
            }

            return;
        }


        static void init_param_def(
            Rcpp::List &opts,
            std::string &par_name,
            Dist &prior
        )
        {
            std::map<std::string, AVAIL::Dist> dist_list = AVAIL::dist_list;

            bool infer = false;
            double init_val = 0.01;
            prior.init("invgamma", 1., 1.);
            if (opts.containsElementNamed(par_name.c_str()))
            {
                Rcpp::List par_opts = opts[par_name];
                init_param(infer, init_val, prior, par_opts);
            }
            prior.init_param(infer, init_val);
            return;
        }

        static arma::vec draw_param_init(
            const Dist &prior,
            const unsigned int N)
        {
            std::map<std::string, AVAIL::Dist> dist_list = AVAIL::dist_list;
            arma::vec par_init(N, arma::fill::zeros);

            switch (dist_list[prior.name])
            {
            case AVAIL::Dist::invgamma:
            {
                arma::vec prec = arma::randg(N, arma::distr_param(prior.par1, 1. / prior.par2));
                // par_init = 1. / prec;
                par_init = prec;
                break;
            }
            case AVAIL::Dist::gamma:
            {
                par_init = arma::randg(N, arma::distr_param(prior.par1, 1. / prior.par2));
                break;
            }
            case AVAIL::Dist::uniform:
            {
                par_init = arma::randu(N, arma::distr_param(prior.par1, prior.par2));
                break;
            }
            case AVAIL::Dist::constant:
            {
                par_init.fill(prior.val);
                break;
            }
            default:
            {
                throw std::invalid_argument("SMC::PL::init_W: unknown prior for W.");
            }
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
     * @param t Time from 0 to nT - 1
     * @param Wsqrt N x 1
     * @param model 
     * @return arma::mat 
     */
        static arma::mat propagate(
            const double &ynow, 
            const arma::vec &Wsqrt, 
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

        static arma::vec imp_weights_forecast(
            const arma::mat &Theta_now,
            const arma::vec &Wsqrt,
            const unsigned int &tnow,
            const arma::vec &y,
            const Model &model)
        {
            unsigned int N = Theta_now.n_cols;
            arma::vec weights(N, arma::fill::zeros);
            for (unsigned int i = 0; i < N; i++)
            {
                // arma::vec theta_now = arma::vectorise(Theta_stored.slice(t).col(i));
                // weights.at(i) = MVNorm::dmvnorm(theta_now, lba.mt.col(t), lba.Ct.slice(t));
                arma::vec theta_next = StateSpace::func_gt(model, Theta_now.col(i), y.at(tnow)); // theta[t+1]
                arma::mat Rt(model.dim.nP, model.dim.nP);
                Rt.at(0, 0) = std::pow(Wsqrt.at(i), 2.);

                double ft_next, qt_next;
                arma::vec Ft_next;
                LBA::func_prior_ft(ft_next, qt_next, Ft_next, tnow + 1, model, y, theta_next, Rt);
                // std::cout << "\nf[t+1] = " << ft_next << ", q[t+1] = " << qt_next << std::endl;
                // Ft_next.t().print("\n F[t+1]");
                // theta_next.t().print("\n theta[t+1]");

                double alpha_next, beta_next;
                LBA::func_alpha_beta(alpha_next, beta_next, model, ft_next, qt_next);

                weights.at(i) = ObsDist::dforecast(
                    y.at(tnow + 1),
                    model.dobs.name,
                    model.dobs.par2,
                    alpha_next,
                    beta_next);
            }

            return weights;
        }

        /**
         * @brief Importance weights based on conditional predictive distribution y[t+1] | theta[t], y[1:t]
         *
         * @param Theta_now
         * @param W
         * @param tnow
         * @param y
         * @param model
         * @return arma::vec
         */
        static arma::vec imp_weights_forecast_approx(
            const arma::mat &Theta_now, // p x N
            const arma::vec &W,         // N x 1, samples of latent variance
            const unsigned int &tnow,
            const arma::vec &y,
            const Model &model)
        {
            unsigned int nP = Theta_now.n_rows;
            unsigned int N = Theta_now.n_cols;
            unsigned int t_next = tnow + 1;

            std::map<std::string, AVAIL::Dist> obs_list = AVAIL::obs_list;
            std::map<std::string, AVAIL::Func> link_list = AVAIL::link_list;

            arma::vec weights(N, arma::fill::zeros);
            for (unsigned int i = 0; i < N; i++)
            {
                // mean of eta[t + 1] | z[t], y[1:t]
                arma::vec mean_theta_next = StateSpace::func_gt(model, Theta_now.col(i), y.at(tnow)); // E(theta[t+1]) = g(theta[t])
                double mean_eta_next = StateSpace::func_ft(model, t_next, mean_theta_next, y);
                if (!model.dim.regressor_baseline)
                {
                    mean_eta_next += model.dobs.par1;
                }

                // variance of eta[t + 1] | z[t], y[1:t]
                arma::vec Ft_next = LBA::func_Ft(model, t_next, mean_theta_next, y); // p x 1
                arma::mat Wmat(nP, nP, arma::fill::zeros);                           // p x p
                Wmat.at(0, 0) = W.at(i);
                double var_eta_next = arma::as_scalar(Ft_next.t() * Wmat * Ft_next);

                // mean of y[t + 1] | z[t], y[1:t]
                double mean_yhat_next = mean_eta_next;

                // variance of y[t + 1] | z[t], y[1:t]
                double var_yhat_next = var_eta_next;
                switch (link_list[tolower(model.flink.name)])
                {
                case AVAIL::Func::identity:
                {
                    var_yhat_next += mean_eta_next;
                    if (obs_list[model.dobs.name] == AVAIL::Dist::nbinomm)
                    {
                        double vtmp = var_eta_next;
                        vtmp += mean_eta_next * mean_eta_next;
                        var_yhat_next += vtmp / model.dobs.par2;
                    }
                    break;
                }
                case AVAIL::Func::exponential:
                {
                    var_yhat_next += std::exp(-mean_eta_next);
                    if (obs_list[model.dobs.name] == AVAIL::Dist::nbinomm)
                    {
                        var_yhat_next += 1. / model.dobs.par2;
                    }
                    break;
                }
                default:
                {
                    throw std::invalid_argument("imp_weights_forecast_approx: unknown link function");
                    break;
                }
                } // switch by link

                double sd_yhat_next = std::sqrt(var_yhat_next);
                double yhat_next = LinkFunc::mu2ft(y.at(t_next), model.flink.name, 0.);
                weights.at(i) = R::dnorm4(yhat_next, mean_yhat_next, sd_yhat_next, false);
            }

            bound_check<arma::vec>(weights, "imp_weights_forecast_approx");
            return weights;
        }


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

        Dist prior_W;
        Dist prior_mu0;

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
        
        void infer(const Model &model, const bool &verbose = VERBOSE)
        {
            // y: (nT + 1) x 1
            for (unsigned int t = 0; t < dim.nT; t++)
            {
                Rcpp::checkUserInterrupt();

                arma::mat Theta_now = Theta_stored.slice(t);

                if (t > 0) // Theta_stored.n_rows
                {
                    arma::vec Wtmp(N);
                    Wtmp.fill(Wt.at(t));
                    weights = imp_weights_forecast_approx(Theta_now, Wtmp, t, y, model);
                    // eff.at(t, 0) = effective_sample_size(weights, false);

                    arma::uvec resample_idx = get_resample_index(weights);

                    Theta_now = Theta_now.cols(resample_idx);
                    Theta_stored.slice(t) = Theta_now;
                }


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

                arma::vec Wsqrt(N, arma::fill::zeros);
                Wsqrt.fill(std::sqrt(Wt.at(t + 1)) + EPS);
                bound_check<arma::vec>(Wsqrt, "SMC::propagate: Wt");
                

                bool positive_noise = (t < Theta_now.n_rows) ? true : false;
                arma::mat Theta_next = propagate(y.at(t), Wsqrt, Theta_now, model, positive_noise);
                weights = imp_weights_likelihood(t + 1, Theta_next, y, model);
                log_cond_marginal.at(t + 1) = log_conditional_marginal(weights);

                arma::uvec resample_idx = get_resample_index(weights);

                Theta_stored.slice(t + 1) = Theta_next.cols(resample_idx);

                if (verbose)
                {
                    Rcpp::Rcout << "\rFiltering: " << t + 1 << "/" << dim.nT;
                }
            }

            if (verbose)
            {
                Rcpp::Rcout << std::endl;
            }

            if (smoothing)
            {
                arma::uvec idx = sample(N, M, weights, true, true);   // M x 1
                arma::mat theta_last = Theta_stored.slice(dim.nT); // p x N
                arma::mat theta_sub = theta_last.cols(idx);                     // p x M

                Theta_smooth.slice(dim.nT) = theta_sub;
                // psi_smooth.row(dim.nT) = theta_sub.row(0);

                for (unsigned int t = dim.nT; t > 0; t--)
                {
                    Rcpp::checkUserInterrupt();

                    double Wnow = Wt.at(t);
                    double Wsd = std::sqrt(Wnow);
                    arma::vec Wsqrt(M); 
                    Wsqrt.fill(Wsd);

                    arma::rowvec psi_smooth_now = Theta_smooth.slice(t).row(0); // 1 x M
                    arma::rowvec psi_filter_prev = Theta_stored.slice(t - 1).row(0); // 1 x N
                    arma::uvec smooth_idx = get_smooth_index(psi_smooth_now, psi_filter_prev, Wsqrt); // M x 1

                    arma::mat theta_next = Theta_stored.slice(t - 1);
                    theta_next = theta_next.cols(smooth_idx);  // p x M

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

            init_param_stored(aw, bw, W_stored, W_smooth, dim.nT, N, M, prior_W);

            return;
        }

        static void init_param_stored(
            arma::mat &par1,
            arma::mat &par2,
            arma::mat &par_stored,
            arma::vec &par_smooth,
            const unsigned int &nT,
            const unsigned int N,
            const unsigned int M,
            const Dist &prior)
        {
            par1.set_size(N, nT + 1); // N x (nT + 1)
            par1.fill(prior.par1);
            par2 = par1;
            par2.fill(prior.par2);

            par_stored.set_size(N, nT + 1); // N x (nT + 1)
            if (prior.infer)
            {
                par_stored.zeros();
                par_stored.col(0) = draw_param_init(prior, N);
            }
            else
            {
                par_stored.fill(prior.val);
            }

            par_smooth.set_size(M); // M x 1
            par_smooth.zeros();

            return;
        }

        static Rcpp::List default_settings()
        {
            Rcpp::List opts;
            opts = SequentialMonteCarlo::default_settings();
            
            Rcpp::List Wopts;
            Wopts["init"] = 0.01;
            Wopts["infer"] = true;
            Wopts["prior_name"] = "invgamma";
            Wopts["prior_param"] = Rcpp::NumericVector::create(0.01, 0.01);
            
            opts["W"] = Wopts;
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
                arma::vec W_filter = W_stored.col(dim.nT);
                output["W"] = Rcpp::wrap(W_filter);
            }

            output["log_marginal_likelihood"] = marginal_likelihood(log_cond_marginal, true);

            return output;
        }

     
        Rcpp::List forecast(const Model &model)
        {

            Rcpp::List out;
            if (smoothing)
            {
                arma::vec Wtmp = W_smooth;
                out = StateSpace::forecast(y, Theta_smooth, Wtmp, model, nforecast);
            }
            else
            {
                arma::vec Wtmp = W_stored.col(dim.nT);
                out = StateSpace::forecast(y, Theta_stored, Wtmp, model, nforecast);
            }
            return out;
        }


        void infer(const Model &model, const bool &verbose = VERBOSE)
        {
            for (unsigned int t = 0; t < dim.nT; t++)
            {
                Rcpp::checkUserInterrupt();

                // propagate(t, Wsqrt, model); // step 1 (a)
                // arma::uvec resample_idx = resample(t); // step 1 (b)
                arma::mat Theta_now = Theta_stored.slice(t);
                
                if (t > 0)
                {
                    /**
                     * @brief Resampling using conditional one-step-ahead predictive distribution as weights.
                     *
                     */
                    arma::vec Wtmp = W_stored.col(t); // variance
                    weights = imp_weights_forecast_approx(Theta_now, Wtmp, t, y, model);
                    // eff.at(t, 0) = effective_sample_size(weights, false);

                    arma::uvec resample_idx = get_resample_index(weights);

                    Theta_now = Theta_now.cols(resample_idx);
                    Theta_stored.slice(t) = Theta_now;

                    if (prior_W.infer)
                    {
                        W_stored.col(t) = Wtmp.elem(resample_idx); // variance, resampled

                        arma::vec aw_tmp = aw.col(t);
                        arma::vec bw_tmp = bw.col(t);
                        aw.col(t) = aw_tmp.elem(resample_idx);
                        bw.col(t) = bw_tmp.elem(resample_idx);
                    }
                }


                arma::vec Wsqrt = arma::sqrt(W_stored.col(t));
                bool positive_noise = (t < Theta_now.n_rows) ? true : false;

                arma::mat Theta_next = propagate(y.at(t), Wsqrt, Theta_now, model, positive_noise);
                Theta_stored.slice(t + 1) = Theta_next;
                arma::vec err = arma::vectorise(Theta_next.row(0) - Theta_now.row(0)); // N x 1, for W
                
                weights = imp_weights_likelihood(t + 1, Theta_next, y, model);
                log_cond_marginal.at(t + 1) = log_conditional_marginal(weights);

                arma::uvec resample_idx = get_resample_index(weights);
                Theta_next = Theta_next.cols(resample_idx);
                Theta_stored.slice(t + 1) = Theta_next; // resample for theta

                err = err.elem(resample_idx); // resample for W

                if (prior_W.infer)
                {
                    double wtmp = arma::var(Theta_stored.slice(t + 1).row(0));
                    wtmp *= 1. / default_discount_factor - 1.;

                    for (unsigned int i = 0; i < N; i++)
                    {
                        // double err = theta_stored.at(0, i, t + 1) - theta_stored.at(0, i, t);
                        double sse = std::pow(err.at(i), 2.);
                        aw.at(i, t + 1) = aw.at(i, t) + 0.5;
                        bw.at(i, t + 1) = bw.at(i, t) + 0.5 * sse;
                        if (t > std::min(0.1 * dim.nT, 20.))
                        {
                            // Wt.at(i, t + 1) = 1. / R::rgamma(aw.at(i, t + 1), 1. / bw.at(i, t + 1));
                            W_stored.at(i, t + 1) = InverseGamma::sample(aw.at(i, t + 1), bw.at(i, t + 1));
                        }
                        else
                        {
                            W_stored.at(i, t + 1) = wtmp;
                        }
                    }
                    // propagate_W(aw, bw, W_stored, t, err, wtmp, W_prior_name); // step 1 (c)
                }
                else
                {
                    if (use_discount)
                    { // Use discount factor if W is not given
                        double wtmp = SequentialMonteCarlo::discount_W(
                            Theta_stored.slice(t), 
                            custom_discount_factor,
                            use_custom, default_discount_factor);
                        W_stored.col(t + 1).fill(wtmp);
                    }

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

            /**
             * @todo Actually we should not resample W, but re-draw W from the posterior at every step of smoothing?
             * 
             */
            if (smoothing)
            {
                weights.ones();
                arma::uvec idx = sample(N, M, weights, false, true);   // M x 1

                arma::mat theta_tmp = Theta_stored.slice(dim.nT); // p x N
                arma::mat theta_tmp2 = theta_tmp.cols(idx); // p x M
                Theta_smooth.slice(dim.nT) = theta_tmp2; // p x M

                // arma::vec ptmp0 = arma::vectorise(Theta_stored.slice(dim.nT).row(0)); // p x N
                // arma::vec ptmp = ptmp0.elem(idx);

                
                // psi_smooth.row(dim.nT) = ptmp.t();

                arma::vec wtmp = W_stored.col(dim.nT);
                // W_smooth.col(dim.nT) = wtmp.elem(idx);
                W_smooth = wtmp.elem(idx); // M x 1
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

            } // opts.smoothing
        } // Particle Learning inference

    
    private:
        arma::mat aw, bw; // N x (nT + 1)
        arma::mat W_stored; // N x (nT + 1)
        arma::vec W_filter; // N x 1
        arma::vec W_smooth; // M x 1

        arma::mat amu, bmu;   // N x (nT + 1)
        arma::mat mu0_stored; // N x (nT + 1)
        arma::vec mu0_filter; // n x 1
        arma::vec mu0_smooth; // M x 1
    };
}


#endif