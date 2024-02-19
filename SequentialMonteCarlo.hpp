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
    class Settings
    {
    public:
        Settings()
        {
            init_default();
        }


        Settings(const Rcpp::List &smc_settings)
        {
            init(smc_settings);
        }

        Settings(
            const unsigned int &N_,
            const unsigned int &M_ = 500,
            const unsigned int &B_ = 1,
            const double &W_ = 0.01,
            const bool &use_discount_ = false,
            const bool &use_custom_ = false,
            const double &custom_discount_factor_ = 0.95,
            const bool &do_smoothing = false)
        {
            init(N_, M_,B_, W_, use_discount_, use_custom_, custom_discount_factor_, do_smoothing);
        }

        static Rcpp::List get_default()
        {
            Rcpp::List opts;
            opts["num_particle"] = 1000;
            opts["num_smooth"] = 500;
            opts["num_backward"] = 1;
            opts["W"] = 0.01;
            opts["use_discount"] = false;
            opts["use_custom"] = false;
            opts["custom_discount_factor"] = 0.95;
            opts["do_smoothing"] = false;

            return opts;
        }

        void init_default()
        {
            N = 1000; // number of particles for filtering
            M = 500;  // number of particles for smoothing
            B = 1;    // steps of backward resampling
            W = 0.01;

            use_discount = false;
            use_custom = false;
            custom_discount_factor = 0.95;
            smoothing = false;
        }

        void init(
            const unsigned int &N_,
            const unsigned int &M_ = 500,
            const unsigned int &B_ = 1,
            const double &W_ = 0.01,
            const bool &use_discount_ = false,
            const bool &use_custom_ = false,
            const double &custom_discount_factor_ = 0.95,
            const bool &do_smoothing = false)
        {
            N = N_;
            M = M_;
            B = B_;
            W = W_;

            use_discount = use_discount_;
            use_custom = use_custom_;
            custom_discount_factor = custom_discount_factor_;
            smoothing = do_smoothing;
        }

        void init(const Rcpp::List &smc_settings)
        {
            Rcpp::List settings = smc_settings;
            if (settings.containsElementNamed("num_particle"))
            {
                N = Rcpp::as<unsigned int>(settings["num_particle"]);
            }
            else
            {
                N = 1000;
            }

            if (settings.containsElementNamed("num_smooth"))
            {
                M = Rcpp::as<unsigned int>(settings["num_smooth"]);
            }
            else
            {
                M = 500;
            }

            if (settings.containsElementNamed("num_backward"))
            {
                B = Rcpp::as<unsigned int>(settings["num_backward"]);
            }
            else
            {
                B = 1;
            }

            if (settings.containsElementNamed("W"))
            {
                W = Rcpp::as<double>(settings["W"]);
            }
            else
            {
                W = 0.01;
            }

            if (settings.containsElementNamed("use_discount"))
            {
                use_discount = Rcpp::as<bool>(settings["use_discount"]);
            }
            else
            {
                use_discount = false;
            }

            if (settings.containsElementNamed("use_custom"))
            {
                use_custom = Rcpp::as<bool>(settings["use_custom"]);
            }
            else
            {
                use_custom = false;
            }

            if (settings.containsElementNamed("custom_discount_factor"))
            {
                custom_discount_factor = Rcpp::as<bool>(settings["custom_discount_factor"]);
            }
            else
            {
                custom_discount_factor = 0.95;
            }

            if (settings.containsElementNamed("do_smoothing"))
            {
                smoothing = Rcpp::as<bool>(settings["do_smoothing"]);
            }
            else
            {
                smoothing = false;
            }

        }


        unsigned int N, M, B; // number of particles for filtering
        double W = 0.01;
        double custom_discount_factor = 0.95;        
        bool use_discount, use_custom, smoothing;
        const double default_discount_factor = 0.99;
    };


    class SequentialMonteCarlo
    {
    public:
        SequentialMonteCarlo(
            const Model &model, 
            const arma::vec &y_in, 
            const unsigned int &Nfilter = 1000,
            const unsigned int &Nsmooth = 500,
            const unsigned int &Nbackward = 1,
            const double &W_in = 0.01,
            const bool &use_discount_factor = false,
            const bool &use_custom_value = false,
            const double &custom_discount_factor_value = 0.95,
            const bool &do_smoothing = false)
        {
            opts.init(
                Nfilter, Nsmooth, Nbackward, W_in, 
                use_discount_factor,
                use_custom_value,
                custom_discount_factor_value,
                do_smoothing);

           
            weights.set_size(opts.N);
            weights.ones();
            lambda = weights;

            y.set_size(dim.nT + 1);
            y.zeros();
            y.tail(y_in.n_elem) = y_in;

            Wt.set_size(dim.nT + 1);
            Wt.fill(W_in);

            Theta_stored.set_size(dim.nP, opts.N, dim.nT + opts.B);
            Theta_stored.zeros();

            psi_smooth.set_size(dim.nT + opts.B, opts.M);
            psi_smooth.zeros();

            return;
        }

 
        SequentialMonteCarlo(
            const Model &model,
            const arma::vec &y_in,
            const Rcpp::List &smc_settings)
        {
            opts.init(smc_settings);
            dim = model.dim;


            weights.set_size(opts.N);
            weights.ones();
            lambda = weights;

            y.set_size(dim.nT + 1);
            y.zeros();
            y.tail(y_in.n_elem) = y_in;

            Wt.set_size(dim.nT + 1);
            Wt.fill(opts.W);

            Theta_stored.set_size(dim.nP, opts.N, dim.nT + opts.B);
            Theta_stored.zeros();

            psi_smooth.set_size(dim.nT + opts.B, opts.M);
            psi_smooth.zeros();

            return;
        }

        SequentialMonteCarlo(
            const Model &model,
            const arma::vec &y_in)
        {
            opts.init_default();
            dim = model.dim;

            weights.set_size(opts.N);
            weights.ones();
            lambda = weights;

            y.set_size(dim.nT + 1);
            y.zeros();
            y.tail(y_in.n_elem) = y_in;

            Wt.set_size(dim.nT + 1);
            Wt.fill(opts.W);

            Theta_stored.set_size(dim.nP, opts.N, dim.nT + opts.B);
            Theta_stored.zeros();

            psi_smooth.set_size(dim.nT + opts.B, opts.M);
            psi_smooth.zeros();

            return;
        }


        arma::mat get_psi_filter()
        {
            arma::mat psi_tmp = Theta_stored.row_as_mat(0);             // (nT + B) x N
            arma::mat psi_filter = psi_tmp.tail_rows(dim.nT + 1);       // (nT + 1) x N

            return psi_filter; // (nT + 1) x N
        }

        arma::mat get_psi_smooth()
        {
            return psi_smooth.tail_rows(dim.nT + 1); // (nT + 1) x M
        }

    protected:

        static double discount_W(
            const double &var_psi, 
            const double &custom_discount_factor = 0.95, 
            const bool &use_custom = false, 
            const double &default_discount_factor = 0.99)
        {
            double W;

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
     * @brief Propagate from now (t) to next (t + 1). Theta_now = Theta_stored.slice(t + B - 1), Theta_new = Theta_stored.slice(t + B)
     * 
     * @param t Time from 0 to nT - 1
     * @param W Wt is length nT + 1 or (nT + B)
     * @param Theta_now 
     * @return arma::mat 
     */
        void propagate(const unsigned int &t, const arma::vec &Wsqrt, const Model &model) // p x N
        {
            /*
            Index of now/old: t or t + B - 1;
            Index of new/next: t + 1 or t + B

            t from 0 to (nT - 1)
            */
            arma::mat Theta_now = Theta_stored.slice(t + opts.B - 1); // If B = 1, it is t
            arma::mat Theta_next = Theta_now; // For t + B.

            /*
                ------ Step 2.1 Propagate ------
                */
            // // theta_stored: p,N,B


            for (unsigned int i = 0; i < Theta_now.n_cols; i++)
            {
                arma::vec theta_now = Theta_now.col(i); // p x 1
                // update_Gt(Gt, gain_code, trans_code, theta_old, y_old, rho);
                arma::vec theta_next = Model::func_gt(model, theta_now, y.at(t));
                // ynow (t): y[0 + B - 1], ..., y[nT - 1 + B - 1], from t = 0, ..., nT - 1


                double omega_next = R::rnorm(0., Wsqrt.at(i));
                if (t < dim.nP)
                {
                    theta_next.at(0) += std::abs(omega_next);
                }
                else
                {
                    theta_next.at(0) += omega_next;
                }

                Theta_next.col(i) = theta_next;

                double ft = Model::func_ft(model, t + 1, theta_next, y); // use y[t], ..., y[t + 1 - nelem]


                lambda.at(i) = LinkFunc::ft2mu(ft, model.flink.name, model.dobs.par1);
                weights.at(i) = ObsDist::loglike(
                    y.at(t + 1),
                    model.dobs.name,
                    lambda.at(i), model.dobs.par2,
                    false);
                // ynext (t + 1): y[0 + 1 + B - 1], ..., y[nT - 1 + 1 + B - 1], for t = 0, ..., nT - 1
            }

            Theta_stored.slice(t + opts.B) = Theta_next; // (t + 1) if B = 1.
            return;
        } // func propagate



        /**
         * @brief 
         * 
         * @param t from 0 to nT - 1
         * @return double 
         */
        arma::uvec resample(const unsigned int &t)
        {
            double wsum = arma::accu(weights);
            bound_check(wsum, "smc_resample: wsum");

            double meff = 0.;
            bool resample = wsum > EPS;
            arma::uvec resample_idx = arma::regspace<arma::uvec>(0, 1, opts.N - 1);

            if (resample)
            {
                weights.for_each([&wsum](arma::vec::elem_type &val)
                                 { val /= wsum; });
                meff = 1. / arma::dot(weights, weights);
                resample_idx = sample(opts.N, opts.N, weights, true, true);

                // new sample theta_next is t + B
                for (unsigned int b = t + 1; b < t + opts.B + 1; b++)
                {
                    arma::mat tmp_old = Theta_stored.slice(b);
                    arma::mat tmp_resampled = tmp_old.cols(resample_idx);
                    Theta_stored.slice(b) = tmp_resampled.cols(resample_idx);
                }
            }
            else
            {
                double N_ = static_cast<double>(opts.N);
                weights.ones();
                weights.for_each([&N_](arma::vec::elem_type &val)
                                 { val /= N_; });
            }

            return resample_idx;
        }


        /**
         * @brief Constructing the smooth trajectory: for a smoothed/selected particle at time t, we pick one out of N particles at (t - 1) to add to the trajectory.
         * 
         * @param t 
         */
        arma::uvec smooth(const unsigned int t, const arma::vec &Wsqrt)
        {
            // for t from (nT + B - 1) back to B.
            // For B = 1, t from nT to 1.
            // now is t.
            // old is t - 1                                                                                                    
            arma::vec psi_now = arma::vectorise(psi_smooth.row(t));                // M x 1, smoothed
            arma::vec psi_old = arma::vectorise(Theta_stored.slice(t - 1).row(0)); // N x 1, filtered
            arma::uvec smooth_idx = arma::regspace<arma::uvec>(0, 1, opts.M - 1);

            for (unsigned int i = 0; i < opts.M; i++) // loop over M smoothed particles at time t.
            {
                // arma::vec diff = (psi_now.at(i) - psi_old) / Wsqrt.at(i); // N x 1
                // weights = - 0.5 * arma::pow(diff, 2.);
                for (unsigned int j = 0; j < opts.N; j++)
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

                smooth_idx.at(i) = sample(opts.N, weights, true);
                psi_smooth.at(t - 1, i) = psi_old.at(smooth_idx.at(i)); // (nT + B) x M
            }

            return smooth_idx;
        }


        Dim dim;
        Settings opts;
        arma::vec y, Wt; // (nT + 1) x 1
        arma::vec weights, lambda; // N x 1
        
        arma::cube Theta_stored; // p x N x (nT + B)
        arma::mat psi_smooth; // (nT + B) x M

    }; // class Sequential Monte Carlo


    class MCS : public SequentialMonteCarlo
    {
    public:
        MCS(
            const Model &dgtf_model,
            const arma::vec &y_in,
            const unsigned int &Nfilter = 1000,
            const unsigned int &Nsmooth = 500,
            const unsigned int &Nbackward = 10,
            const double &W_in = 0.01,
            const bool &use_discount_factor = false,
            const bool &use_custom_value = false,
            const double &custom_discount_factor_value = 0.95) : SequentialMonteCarlo(
                dgtf_model, y_in, Nfilter, Nsmooth, Nbackward, W_in, 
                use_discount_factor, use_custom_value, custom_discount_factor_value, false) {

            _meff.set_size(dgtf_model.dim.nT);
            _meff.zeros();
            return;
        }
        
        MCS(
            const Model &dgtf_model,
            const arma::vec &y_in,
            const Rcpp::List &settings) : SequentialMonteCarlo(dgtf_model, y_in, settings) {

            _meff.set_size(dgtf_model.dim.nT);
            _meff.zeros();
            return;
        }

        static Rcpp::List default_settings()
        {
            Rcpp::List opts = Settings::get_default();
            return opts;
        }
        
        void infer(const Model &model)
        {
            // y: (nT + 1) x 1
            for (unsigned int t = 0; t < dim.nT; t++)
            {
                if (opts.use_discount)
                { // Use discount factor if W is not given
                    arma::rowvec psi = Theta_stored.slice(t + opts.B - 1).row(0);
                    double psi_var = arma::var(psi);
                    Wt.at(t + 1) = SequentialMonteCarlo::discount_W(
                        psi_var, opts.custom_discount_factor,
                        opts.use_custom, opts.default_discount_factor);
                }
                else
                {
                    Wt.at(t + 1) = opts.W;
                }

                bound_check(Wt.at(t + 1), "SMC::propagate: Wt", true, true);
                arma::vec Wsqrt(opts.N, arma::fill::zeros);
                Wsqrt.fill(std::sqrt(Wt.at(t + 1)) + EPS);

                propagate(t, Wsqrt, model);
                arma::uvec resample_idx = resample(t);
            }
        }

    
    private:
        arma::vec _meff;
    };


    class FFBS : public SequentialMonteCarlo
    {
    public:
        FFBS(
            const Model &dgtf_model,
            const arma::vec &y_in,
            const unsigned int &Nfilter = 1000,
            const unsigned int &Nsmooth = 500,
            const unsigned int &Nbackward = 10,
            const double &W_in = 0.01,
            const bool &do_smoothing = false,
            const bool &use_discount_factor = false,
            const bool &use_custom_value = false,
            const double &custom_discount_factor_value = 0.95) : SequentialMonteCarlo(dgtf_model, y_in, Nfilter, Nsmooth, Nbackward, W_in,
                                                                                      use_discount_factor, use_custom_value, custom_discount_factor_value,do_smoothing) {}


        FFBS(
            const Model &dgtf_model,
            const arma::vec &y_in,
            const Rcpp::List &settings) : SequentialMonteCarlo(dgtf_model, y_in, settings) {}

        static Rcpp::List default_settings()
        {
            Rcpp::List opts = Settings::get_default();
            return opts;
        }

        void infer(const Model &model)
        {
            // y: (nT + 1) x 1
            for (unsigned int t = 0; t < dim.nT; t++)
            {
                if (opts.use_discount)
                { // Use discount factor if W is not given
                    arma::rowvec psi = Theta_stored.slice(t + opts.B - 1).row(0);
                    double psi_var = arma::var(psi);
                    Wt.at(t + 1) = SequentialMonteCarlo::discount_W(
                        psi_var, opts.custom_discount_factor,
                        opts.use_custom, opts.default_discount_factor);
                }
                else
                {
                    Wt.at(t + 1) = opts.W;
                }

                bound_check(Wt.at(t + 1), "SMC::propagate: Wt", true, true);
                arma::vec Wsqrt(opts.N, arma::fill::zeros);
                Wsqrt.fill(std::sqrt(Wt.at(t + 1)) + EPS);

                propagate(t, Wsqrt, model);
                arma::uvec resample_idx = resample(t);
            }

            if (opts.smoothing)
            {
                arma::uvec idx = sample(opts.N, opts.M, weights, true, true);   // M x 1
                arma::mat theta_last = Theta_stored.slice(dim.nT + opts.B - 1); // p x N
                arma::mat theta_sub = theta_last.cols(idx);                     // p x M
                psi_smooth.row(dim.nT + opts.B - 1) = theta_sub.row(0);

                for (unsigned int t = dim.nT + opts.B - 1; t > opts.B - 1; t--)
                {
                    double Wnow = Wt.at(t);
                    double Wsd = std::sqrt(Wnow);
                    arma::vec Wsqrt(opts.N); 
                    Wsqrt.fill(Wsd);

                    arma::uvec smooth_idx = smooth(t, Wsqrt);
                }
            }

            return;
        }
    };


    class PL : public SequentialMonteCarlo
    {
    public:
        PL(
            const Model &dgtf_model,
            const arma::vec &y_in,
            const unsigned int &Nfilter = 1000,
            const unsigned int &Nsmooth = 500,
            const unsigned int &Nbackward = 10,
            const double &W_in = 0.01,
            const std::string &W_prior_name = "invgamma",
            const Rcpp::NumericVector &W_prior_param = Rcpp::NumericVector::create(0.01, 0.01),
            const bool &inferW = true,
            const bool &do_smoothing = false,
            const bool &use_discount_factor = false,
            const bool &use_custom_value = false,
            const double &custom_discount_factor_value = 0.95) : SequentialMonteCarlo(dgtf_model, y_in, Nfilter, Nsmooth, Nbackward, W_in,
                                                                                      use_discount_factor, use_custom_value, custom_discount_factor_value,
                                                                                      do_smoothing)
        {
            infer_W = inferW;
            W_prior.init(tolower(W_prior_name), W_prior_param[0], W_prior_param[1]);

            aw.set_size(opts.N, dim.nT + 1);
            aw.fill(W_prior.par1);

            bw.set_size(opts.N, dim.nT + 1);
            bw.fill(W_prior.par2);

            W_stored.set_size(opts.N, dim.nT + 1);
            W_stored.fill(opts.W);
            return;
        }

        PL(
            const Model &dgtf_model,
            const arma::vec &y_in,
            const Rcpp::List &settings) : SequentialMonteCarlo(dgtf_model, y_in, settings)
        {
            Rcpp::List pl_settings = settings;
            if (pl_settings.containsElementNamed("infer_W"))
            {
                infer_W = Rcpp::as<bool>(pl_settings["infer_W"]);

                std::string W_prior_name = "invgamma";
                if (pl_settings.containsElementNamed("W_prior_name"))
                {
                    W_prior_name = Rcpp::as<std::string>(pl_settings["W_prior_name"]);
                }

                double W_prior_par1, W_prior_par2;
                if (pl_settings.containsElementNamed("W_prior_param"))
                {
                    Rcpp::NumericVector param = Rcpp::as<Rcpp::NumericVector>(pl_settings["W_prior_param"]);
                    W_prior_par1 = param[0];
                    W_prior_par2 = param[1];
                }
                else
                {
                    W_prior_par1 = 0.01;
                    W_prior_par2 = 0.01;
                }

                W_prior.init(W_prior_name, W_prior_par1, W_prior_par2);
            }
            else
            {
                infer_W = false;
                W_prior.init("invgamma", 0.01, 0.01);
            }

            aw.set_size(opts.N, dim.nT + 1);
            aw.fill(W_prior.par1);

            bw.set_size(opts.N, dim.nT + 1);
            bw.fill(W_prior.par2);

            W_stored.set_size(opts.N, dim.nT + 1);
            W_stored.fill(opts.W);

            W_smooth.set_size(opts.M, dim.nT + 1);
            W_smooth.fill(opts.W);
            return;
        }

        static Rcpp::List default_settings()
        {
            Rcpp::List opts = Settings::get_default();
            opts["infer_W"] = false;
            opts["W_prior_name"] = "invgamma";
            opts["W_prior_param"] = Rcpp::NumericVector::create(0.01, 0.01);

            return opts;
        }

        void propagate_W(const unsigned int &t, const Model &model)
        {
            std::map<std::string, PriorW> w_prior_list = map_w_prior();

            for (unsigned int i = 0; i < opts.N; i++)
            {
                switch (w_prior_list[W_prior.name])
                {
                case PriorW::invgamma:
                {
                    double err = Theta_stored.at(0, i, t + 1) - Theta_stored.at(0, i, t);
                    double sse = err * err;

                    aw.at(i, t + 1) = aw.at(i, t) + 0.5;
                    bw.at(i, t + 1) = bw.at(i, t) + 0.5 * sse;

                    if (t > std::min(0.1 * static_cast<double>(dim.nT), 20.))
                    {
                        W_stored.at(i, t + 1) = InverseGamma::sample(aw.at(i, t + 1), bw.at(i, t + 1));
                    }

                    break;
                }
                case PriorW::gamma:
                {
                    break;
                }
                default:
                {
                    break;
                }
                }

                return;
            }

            if (t <= std::min(0.1 * static_cast<double>(dim.nT), 20.))
            {
                double wtmp = arma::var(Theta_stored.slice(t + 1).row(0));
                wtmp *= 1. / 0.99 - 1.;
                W_stored.col(t + 1).fill(wtmp);
            }

            return;
        }

        void infer(const Model &model)
        {
            for (unsigned int t = 0; t < dim.nT; t++)
            {
                propagate(t, W_stored.col(t), model);

                if (infer_W)
                {
                    propagate_W(t, model);
                }
                else
                {
                    W_stored.col(t + 1).fill(opts.W);
                }

                arma::uvec resample_idx = resample(t);

                if (infer_W) // resample W if it is inferred.
                {
                    arma::vec atmp = aw.col(t + 1);
                    aw.col(t + 1) = atmp.elem(resample_idx);

                    arma::vec btmp = bw.col(t + 1);
                    bw.col(t + 1) = btmp.elem(resample_idx);

                    arma::vec wtmp = W_stored.col(t + 1);
                    W_stored.col(t + 1) = wtmp.elem(resample_idx);
                }
            } // propagate and resample

            if (opts.smoothing)
            {
                arma::uvec idx = sample(opts.N, opts.M, weights, true, true);   // M x 1
                arma::mat theta_last = Theta_stored.slice(dim.nT + opts.B - 1); // p x N
                arma::mat theta_sub = theta_last.cols(idx);                     // p x M
                psi_smooth.row(dim.nT + opts.B - 1) = theta_sub.row(0);
                
                if (infer_W)
                {
                    arma::vec Wlast = W_stored.col(dim.nT + opts.B - 1);
                    W_smooth.col(dim.nT + opts.B - 1) = Wlast.elem(idx);
                }
                else
                {
                    W_smooth.col(dim.nT + opts.B - 1) = W_stored.col(dim.nT + opts.B - 1);
                }
                

                for (unsigned int t = dim.nT + opts.B - 1; t > opts.B - 1; t--)
                {
                    arma::vec Wtmp = W_stored.col(t - 1);
                    arma::vec Wsqrt = arma::sqrt(Wtmp);
                    arma::uvec smooth_idx = smooth(t, Wsqrt);
                    
                    if (infer_W)
                    {
                        W_smooth.col(t - 1) = Wtmp.elem(smooth_idx);
                    }
                }
            } // opts.smoothing
        } // Particle Learning inference

        arma::mat get_W_filtered(){return W_stored;}
        arma::mat get_W_smoothed(){return W_smooth;}
    
    private:
        bool infer_W;
        arma::mat aw, bw, W_stored, W_smooth;
        Dist W_prior;

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