#ifndef _SEQUENTIALMONTECARLO_H
#define _SEQUENTIALMONTECARLO_H

#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <algorithm>
#include <RcppArmadillo.h>
#include "Model.hpp"


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

            meff.set_size(dim.nT + 1);
            meff.zeros();
            log_cond_marginal = meff;

            return;
        }

        SequentialMonteCarlo()
        {
            dim.init_default();
            y.set_size(dim.nT + 1);
            y.zeros();

            meff.set_size(dim.nT + 1);
            meff.zeros();
            log_cond_marginal = meff;
        }

        static Rcpp::List default_settings()
        {
            Rcpp::List opts;
            opts["num_particle"] = 1000;
            opts["num_smooth"] = 500;
            opts["num_backward"] = 1;
            opts["num_step_ahead_forecast"] = 0;
            opts["W"] = 0.01;
            opts["use_discount"] = false;
            opts["use_custom"] = false;
            opts["custom_discount_factor"] = 0.95;
            opts["do_smoothing"] = false;

            Rcpp::List mu0_opts;
            mu0_opts["prior_name"] = "uniform";
            mu0_opts["prior_param"] = Rcpp::NumericVector::create(0., 10.);
            opts["mu0"] = mu0_opts;

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
            Theta_stored.set_size(dim.nP, N, dim.nT + B);
            Theta_stored.zeros();

            Theta_smooth.set_size(dim.nP, M, dim.nT + B);
            Theta_smooth.zeros();

            if (dim.regressor_baseline)
            {
                Rcpp::List mu0_opts;
                if (settings.containsElementNamed("mu0"))
                {
                    mu0_opts = Rcpp::as<Rcpp::List>(settings["mu0"]);
                }
                else
                {
                    mu0_opts["prior_name"] = "uniform";
                    mu0_opts["prior_param"] = Rcpp::NumericVector::create(0., 10.);
                }
                arma::vec mu0 = init_mu0(mu0_opts, N);
                for (unsigned int t = 0; t < dim.nT + B; t++)
                {
                    Theta_stored.slice(t).row(dim.nP - 1) = mu0.t();
                }
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
        }

        static arma::vec init_mu0(const Rcpp::List &mu0_opts, const unsigned int &N)
        {
            Rcpp::List opts = mu0_opts;
            std::string prior_name = "uniform";
            if (opts.containsElementNamed("prior_name"))
            {
                prior_name = Rcpp::as<std::string>(opts["prior_name"]);
                tolower(prior_name);
            }

            Rcpp::NumericVector prior_param = {0., 10.};
            if (opts.containsElementNamed("prior_param"))
            {
                prior_param = Rcpp::as<Rcpp::NumericVector>(opts["prior_param"]);
            }

            arma::vec mu0(N, arma::fill::zeros);
            std::map<std::string, AVAIL::Dist> mu0_prior_list = AVAIL::mu0_prior_list;

            switch (mu0_prior_list[prior_name])
            {
            case AVAIL::Dist::gamma:
            {
                mu0 = arma::randg(N, arma::distr_param(prior_param[0], 1./prior_param[1]));
                break;
            }
            case AVAIL::Dist::invgamma:
            {
                mu0 = 1. / arma::randg(N, arma::distr_param(prior_param[0], 1. / prior_param[1]));
                break;
            }
            case AVAIL::Dist::uniform:
            {
                mu0 = arma::randu(N, arma::distr_param(prior_param[0], prior_param[1]));
                break;
            }
            default:
            {
                mu0 = arma::randu(N, arma::distr_param(0., 10.));
                break;
            }
            } // switch by mu0 prior type

            return mu0;
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

        static arma::uvec get_resample_index( const arma::vec &weights_in)
        {
            unsigned int N = weights_in.n_elem;
            arma::vec weights = weights_in; // N x 1
            double wsum = arma::accu(weights);
            bound_check(wsum, "smc_resample: wsum");

            double meff = 0.;
            bool resample = wsum > EPS;
            arma::uvec resample_idx = arma::regspace<arma::uvec>(0, 1, N - 1);

            if (resample)
            {
                weights.for_each([&wsum](arma::vec::elem_type &val)
                                 { val /= wsum; });
                resample_idx = sample(N, N, weights, true, true);
            }

            return resample_idx;
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

        void forecast_error(double &err, const Model &model, const std::string &loss_func = "quadratic")
        {
            arma::cube theta_tmp = Theta_stored.tail_slices(dim.nT + 1); // p x N x (nT + 1)
            StateSpace::forecast_error(err, theta_tmp, y, model, loss_func);
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
        arma::vec meff, log_cond_marginal;

        arma::cube Theta_stored; // p x N x (nT + B)
        arma::cube Theta_smooth; // p x M x (nT + B)
        // arma::mat psi_smooth; // (nT + B) x M

        unsigned int N = 1000;
        unsigned int M = 500;
        unsigned int B = 10;
        unsigned int nforecast = 0;
        double W = 0.01;

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

            W = 0.01;
            if (opts.containsElementNamed("W"))
            {
                W = Rcpp::as<double>(opts["W"]);
            }
            Wt.set_size(dim.nT + 1);
            Wt.fill(W);

            smoothing = true;

            Theta_smooth.clear();
            Theta_smooth = Theta_stored;
        }

        Rcpp::List forecast(const Model &model)
        {
            arma::vec Wtmp(N, arma::fill::zeros);
            Wtmp.fill(W);
            Rcpp::List out = StateSpace::forecast(y, Theta_stored, Wtmp, model, nforecast);
            return out;
        }


        arma::mat optimal_discount_factor(
            const Model &model,
            const double &from,
            const double &to,
            const double &delta = 0.01,
            const std::string &loss = "quadratic")
        {
            arma::vec grid = arma::regspace<arma::vec>(from, delta, to);
            unsigned int nelem = grid.n_elem;
            arma::mat stats(nelem, 3, arma::fill::zeros);

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
                StateSpace::forecast_error(err_forecast, theta_tmp, y, model, loss, false);
                stats.at(i, 1) = err_forecast;

                arma::cube theta_tmp2 = Theta_smooth.tail_slices(model.dim.nT + 1);
                double err_fit = 0.;
                StateSpace::fitted_error(err_fit, theta_tmp2, y, model, loss, false);
                stats.at(i, 2) = err_fit;

                Rcpp::Rcout << "\rProgress: " << i + 1 << "/" << nelem;
            }

            Rcpp::Rcout << std::endl;
            
            return stats;
        }

        arma::mat optimal_W(
            const Model &model,
            const arma::vec &grid,
            const std::string &loss = "quadratic")
        {
            unsigned int nelem = grid.n_elem;
            arma::mat stats(nelem, 3, arma::fill::zeros);

            use_discount = false;

            for (unsigned int i = 0; i < nelem; i++)
            {
                Rcpp::checkUserInterrupt();

                W = grid.at(i);
                stats.at(i, 0) = W;

                infer(model);
                arma::cube theta_tmp = Theta_stored.tail_slices(model.dim.nT + 1);

                double err_forecast = 0.;
                StateSpace::forecast_error(err_forecast, theta_tmp, y, model, loss, false);
                stats.at(i, 1) = err_forecast;

                arma::cube theta_tmp2 = Theta_smooth.tail_slices(model.dim.nT + 1);
                double err_fit = 0.;
                StateSpace::fitted_error(err_fit, theta_tmp2, y, model, loss, false);
                stats.at(i, 2) = err_fit;

                Rcpp::Rcout << "\rProgress: " << i + 1 << "/" << nelem;
            }

            Rcpp::Rcout << std::endl;

            return stats;
        }

        arma::mat optimal_num_backward(
            const Model &model,
            const unsigned int &from, 
            const unsigned int &to, 
            const unsigned int &delta = 1,
            const std::string &loss = "quadratic")
        {
            arma::uvec grid = arma::regspace<arma::uvec>(from, delta, to);
            unsigned int nelem = grid.n_elem;
            arma::mat stats(nelem, 3, arma::fill::zeros);

            for (unsigned int i = 0; i < nelem; i ++)
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
                StateSpace::forecast_error(err_forecast, theta_tmp, y, model, loss, false);
                stats.at(i, 1) = err_forecast;

                arma::cube theta_tmp2 = Theta_smooth.tail_slices(model.dim.nT + 1);
                double err_fit = 0.;
                StateSpace::fitted_error(err_fit, theta_tmp2, y, model, loss, false);
                stats.at(i, 2) = err_fit;

                Rcpp::Rcout << "\rProgress: " << i + 1 << "/" << nelem;
            }

            Rcpp::Rcout << std::endl;

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
        

        void infer(const Model &model)
        {
            // y: (nT + 1) x 1
            for (unsigned int t = 0; t < dim.nT; t++)
            {
                Rcpp::checkUserInterrupt();

                if (use_discount)
                { // Use discount factor if W is not given
                    bool use_custom_val = (use_custom && t > B) ? true : false;

                    Wt.at(t + 1) = SequentialMonteCarlo::discount_W(
                        Theta_smooth.slice(t + B - 1),
                        custom_discount_factor,
                        use_custom_val, 
                        default_discount_factor);
                }
                else
                {
                    Wt.at(t + 1) = W;
                }


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
            B = 1;

            W = 0.01;
            if (opts.containsElementNamed("W"))
            {
                W = Rcpp::as<double>(opts["W"]);
            }
            Wt.set_size(dim.nT + 1);
            Wt.fill(W);
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

        arma::uvec get_smooth_index(
            const unsigned int t,
            const arma::vec &Wsqrt) // N x 1
        {
            // for t from (nT + B - 1) back to B.
            // For B = 1, t from nT to 1.
            // now is t.
            // old is t - 1

            // arma::vec psi_now = arma::vectorise(psi_smooth.row(t));                // M x 1, smoothed
            arma::vec psi_now = arma::vectorise(Theta_smooth.slice(t).row(0)); // M x 1, smoothed
            arma::vec psi_old = arma::vectorise(Theta_stored.slice(t - 1).row(0)); // N x 1, filtered
            arma::uvec smooth_idx = arma::regspace<arma::uvec>(0, 1, M - 1);

            for (unsigned int i = 0; i < M; i++) // loop over M smoothed particles at time t.
            {
                // arma::vec diff = (psi_now.at(i) - psi_old) / Wsqrt.at(i); // N x 1
                // weights = - 0.5 * arma::pow(diff, 2.);
                for (unsigned int j = 0; j < N; j++)
                {
                    weights.at(j) = R::dnorm(psi_old.at(j), psi_now.at(i), Wsqrt.at(i), true);
                }

                weights.elem(arma::find(weights > UPBND)).fill(UPBND);
                weights = arma::exp(weights);
                bound_check<arma::vec>(weights, "SequentialMonteCarlo::smooth");

                if (arma::accu(weights) < EPS)
                {
                    weights.ones();
                }
                weights /= arma::accu(weights);

                smooth_idx.at(i) = sample(N, weights, true);
            }

            return smooth_idx;
        }

        Rcpp::List forecast(const Model &model)
        {
            arma::vec Wtmp(N, arma::fill::zeros);
            Wtmp.fill(W);

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
            const std::string &loss = "quadratic")
        {
            arma::vec grid = arma::regspace<arma::vec>(from, delta, to);
            unsigned int nelem = grid.n_elem;
            arma::mat stats(nelem, 3, arma::fill::zeros);

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
                StateSpace::forecast_error(err_forecast, theta_tmp, y, model, loss, false);
                stats.at(i, 1) = err_forecast;

                double err_fit = 0.;
                if (smoothing)
                {
                    arma::cube theta_tmp2 = Theta_smooth.tail_slices(model.dim.nT + 1);
                    StateSpace::fitted_error(err_fit, theta_tmp2, y, model, loss, false);
                }
                else
                {
                    StateSpace::fitted_error(err_fit, theta_tmp, y, model, loss, false);
                }

                Rcpp::Rcout << "\rProgress: " << i + 1 << "/" << nelem;
            }

            Rcpp::Rcout << std::endl;

            return stats;
        }

        arma::mat optimal_W(
            const Model &model,
            const arma::vec &grid,
            const std::string &loss = "quadratic")
        {
            unsigned int nelem = grid.n_elem;
            arma::mat stats(nelem, 3, arma::fill::zeros);

            use_discount = false;

            for (unsigned int i = 0; i < nelem; i++)
            {
                Rcpp::checkUserInterrupt();

                W = grid.at(i);
                stats.at(i, 0) = W;

                infer(model, false);
                arma::cube theta_tmp = Theta_stored.tail_slices(model.dim.nT + 1);

                double err_forecast = 0.;
                StateSpace::forecast_error(err_forecast, theta_tmp, y, model, loss, false);
                stats.at(i, 1) = err_forecast;

                double err_fit = 0.;
                if (smoothing)
                {
                    arma::cube theta_tmp2 = Theta_smooth.tail_slices(model.dim.nT + 1);
                    StateSpace::fitted_error(err_fit, theta_tmp2, y, model, loss, false);
                }
                else
                {
                    StateSpace::fitted_error(err_fit, theta_tmp, y, model, loss, false);
                }
                
                stats.at(i, 2) = err_fit;

                Rcpp::Rcout << "\rProgress: " << i + 1 << "/" << nelem;
            }
            Rcpp::Rcout << std::endl;

            return stats;
        }

        void infer(const Model &model, const bool &verbose = true)
        {
            // y: (nT + 1) x 1
            for (unsigned int t = 0; t < dim.nT; t++)
            {
                Rcpp::checkUserInterrupt();

                if (use_discount)
                { // Use discount factor if W is not given
                    bool use_custom_val = (use_custom && t > 1) ? true : false;

                    Wt.at(t + 1) = SequentialMonteCarlo::discount_W(
                        Theta_stored.slice(t),
                        custom_discount_factor,
                        use_custom_val, 
                        default_discount_factor);
                }
                else
                {
                    Wt.at(t + 1) = W;
                }

                arma::vec Wsqrt(N, arma::fill::zeros);
                Wsqrt.fill(std::sqrt(Wt.at(t + 1)) + EPS);
                bound_check<arma::vec>(Wsqrt, "SMC::propagate: Wt");
                
                

                arma::mat Theta_now = Theta_stored.slice(t);

                bool positive_noise = (t < Theta_now.n_rows) ? true : false;
                arma::mat Theta_next = propagate(y.at(t), Wsqrt, Theta_now, model, positive_noise);
                weights = imp_weights_likelihood(t + 1, Theta_next, y, model);
                log_cond_marginal.at(t + 1) = log_conditional_marginal(weights);

                arma::uvec resample_idx = get_resample_index(weights);

                Theta_stored.slice(t + 1) = Theta_next.cols(resample_idx);
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
                    arma::vec Wsqrt(N); 
                    Wsqrt.fill(Wsd);

                    // arma::uvec smooth_idx = get_smooth_index(t, Wsqrt, idx);
                    // arma::mat theta_next0 = Theta_stored.slice(t - 1); // p x N
                    // arma::mat theta_next = theta_next0.cols(idx);      // p x M

                    arma::uvec smooth_idx = get_smooth_index(t, Wsqrt); // M x 1

                    arma::mat theta_next = Theta_stored.slice(t - 1);
                    theta_next = theta_next.cols(smooth_idx);  // p x M

                    Theta_smooth.slice(t - 1) = theta_next;
                    // psi_smooth.row(t - 1) = theta_next.row(0); // 1 x M

                    if (verbose)
                    {
                        Rcpp::Rcout << "\rProgress: " << dim.nT - t + 1 << "/" << dim.nT;
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

            _infer_W = true;
            W_prior_name = "invgamma";

            aw.set_size(N, dim.nT + 1);
            aw.fill(0.01);
            bw = aw;
            W_stored = aw;

            W_smooth.set_size(M, dim.nT + 1);
            W_smooth.zeros();


            if (opts.containsElementNamed("W"))
            {
                Rcpp::List Wopts = Rcpp::as<Rcpp::List>(opts["W"]);

                if (Wopts.containsElementNamed("infer"))
                {
                    _infer_W = Rcpp::as<bool>(Wopts["infer"]);
                }

                if (Wopts.containsElementNamed("init"))
                {
                    double init_W = Rcpp::as<double>(Wopts["init"]);
                    W_stored.fill(init_W);
                }

               
                if (Wopts.containsElementNamed("prior_name"))
                {
                    W_prior_name = Rcpp::as<std::string>(Wopts["prior_name"]);
                }

                if (Wopts.containsElementNamed("prior_param"))
                {
                    Rcpp::NumericVector param = Rcpp::as<Rcpp::NumericVector>(Wopts["prior_param"]);
                    double W_prior_par1 = param[0];
                    double W_prior_par2 = param[1];

                    aw.fill(W_prior_par1);
                    bw.fill(W_prior_par2);
                }

                
            }

            
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

                if (_infer_W)
                {
                    output["W_filter"] = Rcpp::wrap(get_W_filtered());
                    output["W"] = Rcpp::wrap(get_W_smoothed());
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

                if (_infer_W)
                {
                    output["W"] = Rcpp::wrap(get_W_filtered());
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

        /**
         * @brief Constructing the smooth trajectory: for a smoothed/selected particle at time t, we pick one out of N particles at (t - 1) to add to the trajectory.
         *
         * @param t
         */
        arma::uvec get_smooth_index(
            const unsigned int t,
            const arma::vec &Wsqrt, // M x 1
            const arma::uvec &idx0)
        {
            // for t from (nT + B - 1) back to B.
            // For B = 1, t from nT to 1.
            // now is t.
            // old is t - 1

            // psi_smooth: (nT + B) x M
            arma::uvec smooth_idx(M, arma::fill::zeros);

            // arma::vec psi_now = arma::vectorise(psi_smooth.row(t));                // M x 1, smoothed

            arma::vec psi_now = arma::vectorise(Theta_smooth.slice(t).row(0)); // M x 1, smoothed
            arma::vec psi_old = arma::vectorise(Theta_stored.slice(t - 1).row(0)); // N x 1, filtered
            arma::vec psi_filter = psi_old.elem(idx0);                             // M x 1

            for (unsigned int i = 0; i < M; i++) // loop over M smoothed particles at time t.
            {
                // arma::vec diff = (psi_now.at(i) - psi_old) / Wsqrt.at(i); // N x 1
                // weights = - 0.5 * arma::pow(diff, 2.);
                // arma::vec psi_diff = psi_smooth.at(t, i) - psi_filter; // M x 1
                arma::vec psi_diff = Theta_smooth.at(0, i, t) - psi_filter; // M x 1
                // arma::vec psi_diff = psi_smooth.at(t, i) - psi_old; // N x 1
                arma::vec weights_smooth(psi_diff.n_elem, arma::fill::zeros);
                for (unsigned int j = 0; j < psi_diff.n_elem; j++)
                {
                    weights_smooth.at(j) = R::dnorm(psi_diff.at(j), 0., Wsqrt.at(j), true);
                }

                weights_smooth.elem(arma::find(weights_smooth > UPBND)).fill(UPBND);
                weights_smooth = arma::exp(weights_smooth);
                bound_check<arma::vec>(weights_smooth, "SequentialMonteCarlo::smooth");

                if (arma::accu(weights_smooth) < EPS)
                {
                    weights_smooth.ones();
                }
                weights_smooth /= arma::accu(weights_smooth);

                smooth_idx.at(i) = sample(M, weights_smooth, true);
                // psi_smooth.at(t - 1, i) = psi_filter.at(smooth_idx.at(i)); // (nT + B) x M
            }

            return smooth_idx;
        }

        Rcpp::List forecast(const Model &model)
        {

            Rcpp::List out;
            if (smoothing)
            {
                arma::vec Wtmp = W_smooth.col(dim.nT);
                out = StateSpace::forecast(y, Theta_smooth, Wtmp, model, nforecast);
            }
            else
            {
                arma::vec Wtmp = W_stored.col(dim.nT);
                out = StateSpace::forecast(y, Theta_stored, Wtmp, model, nforecast);
            }
            return out;
        }


        void infer(const Model &model, const bool &verbose = true)
        {
            for (unsigned int t = 0; t < dim.nT; t++)
            {
                Rcpp::checkUserInterrupt();

                // propagate(t, Wsqrt, model); // step 1 (a)
                // arma::uvec resample_idx = resample(t); // step 1 (b)
                arma::mat Theta_now = Theta_stored.slice(t);
                arma::vec Wsqrt = arma::sqrt(W_stored.col(t));

                if (t > Theta_stored.n_rows)
                {
                    weights = imp_weights_forecast(Theta_now, Wsqrt, t, y, model);

                    // double meff_tmp = 1. / arma::dot(weights, weights);
                    // if (meff_tmp > 0.1 * static_cast<double>(N))
                    // {
                        arma::uvec resample_idx = get_resample_index(weights);

                        Theta_now = Theta_now.cols(resample_idx);
                        Theta_stored.slice(t) = Theta_now;

                        Wsqrt = Wsqrt.elem(resample_idx);
                        arma::vec Wtmp = W_stored.col(t);
                        W_stored.col(t) = Wtmp.elem(resample_idx);
                    // }
                }



                bool positive_noise = (t < Theta_now.n_rows) ? true : false;
                arma::mat Theta_next = propagate(y.at(t), Wsqrt, Theta_now, model, positive_noise);
                Theta_stored.slice(t + 1) = Theta_next;
                arma::vec err = arma::vectorise(Theta_next.row(0) - Theta_now.row(0)); // N x 1, for W
                
                weights = imp_weights_likelihood(t + 1, Theta_next, y, model);
                log_cond_marginal.at(t + 1) = log_conditional_marginal(weights);

                // meff.at(t) = 1. / arma::dot(weights, weights);

                // if (meff.at(t) > 0.1 * static_cast<double>(N))
                // {
                    arma::uvec resample_idx = get_resample_index(weights);
                    Theta_next = Theta_next.cols(resample_idx);
                    Theta_stored.slice(t + 1) = Theta_next; // resample for theta

                    err = err.elem(resample_idx);                              // resample for W
                // }
                // else
                // {
                    
                // }

                if (_infer_W)
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
                


                
                // if (_infer_W)
                // {
                //     resample_idx = get_resample_index(weights);
                //     arma::vec wnext = W_stored.col(t + 1);
                //     W_stored.col(t + 1) = wnext.elem(resample_idx);
                // }
                

            } // propagate and resample

            /**
             * @todo Actually we should not resample W, but re-draw W from the posterior at every step of smoothing?
             * 
             */
            if (smoothing)
            {
                weights.ones();
                arma::uvec idx = sample(N, M, weights, true, true);   // M x 1

                arma::mat theta_tmp = Theta_stored.slice(dim.nT); // p x N
                arma::mat theta_tmp2 = theta_tmp.cols(idx); // p x M
                Theta_smooth.slice(dim.nT) = theta_tmp2; // p x M

                arma::vec ptmp0 = arma::vectorise(Theta_stored.slice(dim.nT).row(0)); // p x N
                arma::vec ptmp = ptmp0.elem(idx);

                
                // psi_smooth.row(dim.nT) = ptmp.t();



                arma::vec wtmp = W_stored.col(dim.nT);
                W_smooth.col(dim.nT) = wtmp.elem(idx);

                for (unsigned int t = dim.nT; t > 1; t--)
                {
                    Rcpp::checkUserInterrupt();

                    arma::vec Wtmp0 = W_stored.col(t - 1); // N x 1
                    arma::vec Wtmp = Wtmp0.elem(idx); // M x 1
                    arma::vec Wsqrt = arma::sqrt(W_stored.col(t - 1)); // M x 1
                    arma::uvec smooth_idx = get_smooth_index(t, Wsqrt, idx);

                    if (_infer_W)
                    {
                        W_smooth.col(t - 1) = Wtmp.elem(smooth_idx);
                    }

                    arma::mat theta_tmp0 = Theta_stored.slice(t - 1); // p x N
                    arma::mat theta_tmp = theta_tmp0.cols(idx); // p x M
                    theta_tmp = theta_tmp.cols(smooth_idx);

                    Theta_smooth.slice(t - 1) = theta_tmp;
                    // psi_smooth.row(t - 1) = theta_tmp.row(0);

                    if (verbose)
                    {
                        Rcpp::Rcout << "\rProgress: " << dim.nT - t + 1 << "/" << dim.nT;
                    }
                    
                }

                if (verbose)
                {
                    Rcpp::Rcout << std::endl;
                }

            } // opts.smoothing
        } // Particle Learning inference

        arma::mat get_W_filtered(){return W_stored;}
        arma::mat get_W_smoothed(){return W_smooth;}
    
    private:
        bool _infer_W = true;
        std::string W_prior_name = "invgamma";
        arma::mat aw, bw; // N x (nT + 1)
        arma::mat W_stored; // N x (nT + 1)
        arma::mat W_smooth; // M x (nT + 1)

        enum PriorW {
            invgamma,
            gamma
        };

        static std::map<std::string, PriorW> map_w_prior()
        {
            std::map<std::string, PriorW> map;
            map["invgamma"] = PriorW::invgamma;
            map["ig"] = PriorW::invgamma;
            map["inv-gamma"] = PriorW::invgamma;
            map["inv_gamma"] = PriorW::invgamma;

            map["gamma"] = PriorW::gamma;

            return map;
        }
    };
}


#endif