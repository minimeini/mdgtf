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
            pmarg = meff;

            return;
        }

        SequentialMonteCarlo()
        {
            dim.init_default();
            y.set_size(dim.nT + 1);
            y.zeros();

            meff.set_size(dim.nT + 1);
            meff.zeros();
            pmarg = meff;
        }

        static Rcpp::List default_settings()
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
            psi_smooth.set_size(dim.nT + B, M);
            psi_smooth.zeros();

            W = 0.01;
            if (settings.containsElementNamed("W"))
            {
                W = Rcpp::as<double>(settings["W"]);
            }
            Wt.set_size(dim.nT + 1);
            Wt.fill(W);

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

            for (unsigned int i = 0; i < N; i++)
            {
                arma::vec theta_next = Theta_next.col(i);
                double ft = StateSpace::func_ft(model, t_next, theta_next, yall); // use y[t], ..., y[t + 1 - nelem]
                double lambda = LinkFunc::ft2mu(ft, model.flink.name, model.dobs.par1); // conditional mean of the observations

                weights.at(i) = ObsDist::loglike(
                    yall.at(t_next),
                    model.dobs.name,
                    lambda, model.dobs.par2,
                    false);
            }

            return weights;            
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
                std::cout << "\nf[t+1] = " << ft_next << ", q[t+1] = " << qt_next << std::endl;
                Ft_next.t().print("\n F[t+1]");
                theta_next.t().print("\n theta[t+1]");

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

        void resample_theta(const unsigned int &t, const arma::uvec resample_idx)
        {
            // new sample theta_next is t + B
            for (unsigned int b = t + 1; b < t + B + 1; b++)
            {
                arma::mat tmp_old = Theta_stored.slice(b);
                arma::mat tmp_resampled = tmp_old.cols(resample_idx);
                Theta_stored.slice(b) = tmp_resampled.cols(resample_idx);
            }

            return;
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
                psi_smooth.at(t - 1, i) = psi_old.at(smooth_idx.at(i)); // (nT + B) x M
            }

            return smooth_idx;
        }


        Dim dim;

        arma::vec y, Wt; // (nT + 1) x 1
        arma::vec weights, lambda; // N x 1
        arma::vec meff, pmarg;
        
        arma::cube Theta_stored; // p x N x (nT + B)
        arma::mat psi_smooth; // (nT + B) x M

        unsigned int N = 1000;
        unsigned int M = 500;
        unsigned int B = 10;
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
            const arma::vec &y_in) : SequentialMonteCarlo(dgtf_model, y_in) {}


        void infer(const Model &model)
        {
            // y: (nT + 1) x 1
            for (unsigned int t = 0; t < dim.nT; t++)
            {
                if (use_discount)
                { // Use discount factor if W is not given
                    bool use_custom_val = (use_custom && t > B) ? true : false;

                    Wt.at(t + 1) = SequentialMonteCarlo::discount_W(
                        Theta_stored.slice(t + B - 1),
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
                

                arma::mat Theta_now = Theta_stored.slice(t + B - 1);

                bool positive_noise = (t < Theta_now.n_rows) ? true : false;
                arma::mat Theta_next = propagate(y.at(t), Wsqrt, Theta_now, model, positive_noise);
                weights = imp_weights_likelihood(t + 1, Theta_next, y, model);
                arma::uvec resample_idx = get_resample_index(weights);

                Theta_stored.slice(t + B) = Theta_next;
                // for (unsigned int b = t + 1; b < t + B + 1; b++)
                // {
                //     arma::mat theta_tmp = Theta_stored.slice(t + b);
                //     Theta_stored.slice(t + b) = theta_tmp.cols(resample_idx);
                // }

                // Theta_stored.slice(t + B) = Theta_next.cols(resample_idx);

                // propagate(t, Wsqrt, model);
                // // arma::uvec resample_idx = resample(t);
                // arma::uvec resample_idx = SequentialMonteCarlo::get_resample_index(weights);
                resample_theta(t, resample_idx);
            }
        }


    };


    class FFBS : public SequentialMonteCarlo
    {
    public:
        FFBS(
            const Model &dgtf_model,
            const arma::vec &y_in) : SequentialMonteCarlo(dgtf_model, y_in) {}



        void infer(const Model &model)
        {
            // y: (nT + 1) x 1
            for (unsigned int t = 0; t < dim.nT; t++)
            {
                if (use_discount)
                { // Use discount factor if W is not given
                    bool use_custom_val = (use_custom && t > B) ? true : false;

                    Wt.at(t + 1) = SequentialMonteCarlo::discount_W(
                        Theta_stored.slice(t + B - 1),
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
                
                

                arma::mat Theta_now = Theta_stored.slice(t + B - 1);

                bool positive_noise = (t < Theta_now.n_rows) ? true : false;
                arma::mat Theta_next = propagate(y.at(t), Wsqrt, Theta_now, model, positive_noise);
                weights = imp_weights_likelihood(t + 1, Theta_next, y, model);
                arma::uvec resample_idx = get_resample_index(weights);

                Theta_stored.slice(t + B) = Theta_next.cols(resample_idx);
            }

            if (smoothing)
            {
                arma::uvec idx = sample(N, M, weights, true, true);   // M x 1
                arma::mat theta_last = Theta_stored.slice(dim.nT + B - 1); // p x N
                arma::mat theta_sub = theta_last.cols(idx);                     // p x M
                psi_smooth.row(dim.nT + B - 1) = theta_sub.row(0);

                for (unsigned int t = dim.nT + B - 1; t > B - 1; t--)
                {
                    double Wnow = Wt.at(t);
                    double Wsd = std::sqrt(Wnow);
                    arma::vec Wsqrt(N); 
                    Wsqrt.fill(Wsd);

                    arma::uvec smooth_idx = smooth(t, Wsqrt);
                    arma::mat theta_next = Theta_stored.slice(t - 1);
                    theta_next = theta_next.cols(smooth_idx); // p x M
                    psi_smooth.row(t - 1) = theta_next.row(0); // 1 x M
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
            const Rcpp::List &settings) : SequentialMonteCarlo(dgtf_model, y_in){}


        void init(const Rcpp::List &opts_in)
        {
            Rcpp::List opts = opts_in;
            SequentialMonteCarlo::init(opts);

            infer_W = false;
            W_prior.init("invgamma", 0.01, 0.01);
            if (opts.containsElementNamed("W"))
            {
                Rcpp::List Wopts = Rcpp::as<Rcpp::List>(opts["W"]);

                infer_W = Rcpp::as<bool>(opts["infer"]);

                std::string W_prior_name = "invgamma";
                if (opts.containsElementNamed("prior_name"))
                {
                    W_prior_name = Rcpp::as<std::string>(opts["prior_name"]);
                }

                double W_prior_par1 = 0.01;
                double W_prior_par2 = 0.01;
                if (opts.containsElementNamed("prior_param"))
                {
                    Rcpp::NumericVector param = Rcpp::as<Rcpp::NumericVector>(opts["prior_param"]);
                    W_prior_par1 = param[0];
                    W_prior_par2 = param[1];
                }

                W_prior.init(W_prior_name, W_prior_par1, W_prior_par2);
            }

            aw.set_size(N, dim.nT + 1);
            aw.fill(W_prior.par1);

            bw.set_size(N, dim.nT + 1);
            bw.fill(W_prior.par2);

            W_stored.set_size(N, dim.nT + 1);
            W_stored.fill(W);
        }


        static Rcpp::List default_settings()
        {
            Rcpp::List opts;
            opts = SequentialMonteCarlo::default_settings();
            
            Rcpp::List Wopts;
            Wopts["infer"] = false;
            Wopts["prior_name"] = "invgamma";
            Wopts["prior_param"] = Rcpp::NumericVector::create(0.01, 0.01);

            opts["W"] = Wopts;
            return opts;
        }

        void propagate_W(const unsigned int &t, const Model &model)
        {
            std::map<std::string, PriorW> w_prior_list = map_w_prior();

            for (unsigned int i = 0; i < N; i++)
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
                    std::cout << W_stored.col(t + 1).t() << std::endl;

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
            arma::vec Wsqrt(N, arma::fill::ones);
            Wsqrt.fill(std::sqrt(W));

            for (unsigned int t = 0; t < dim.nT; t++)
            {
                // propagate(t, Wsqrt, model); // step 1 (a)
                // arma::uvec resample_idx = resample(t); // step 1 (b)
                if (t > Theta_stored.n_rows)
                {
                    arma::mat Theta_now = Theta_stored.slice(t + B - 1);
                    weights = imp_weights_forecast(Theta_now, Wsqrt, t, y, model);
                    arma::uvec resample_idx = get_resample_index(weights);
                    Theta_stored.slice(t + B - 1) = Theta_now.cols(resample_idx);
                }

                

                arma::mat Theta_now = Theta_stored.slice(t +  - 1);

                bool positive_noise = (t < Theta_now.n_rows) ? true : false;
                arma::mat Theta_next = propagate(y.at(t), Wsqrt, Theta_now, model, positive_noise);
                weights = imp_weights_likelihood(t + 1, Theta_next, y, model);
                arma::uvec resample_idx = get_resample_index(weights);

                Theta_stored.slice(t + B) = Theta_next.cols(resample_idx);

                if (infer_W)
                {
                    propagate_W(t, model); // step 1 (c)
                }
                else
                {
                    if (use_discount)
                    { // Use discount factor if W is not given
                        double wtmp = SequentialMonteCarlo::discount_W(
                            Theta_stored.slice(t + B - 1), 
                            custom_discount_factor,
                            use_custom, default_discount_factor);
                        W_stored.col(t + 1).fill(wtmp);
                    }
                    else
                    {
                        W_stored.col(t + 1).fill(W);
                    }
                }

                Wsqrt = arma::sqrt(W_stored.col(t + 1));

                
                if (infer_W)
                {
                    

                    resample_idx = get_resample_index(weights);
                    arma::vec wnext = W_stored.col(t + 1);
                    W_stored.col(t + 1) = wnext.elem(resample_idx);
                    // std::cout << W_stored.col(t + 1).t() << std::endl;
                }
                

            } // propagate and resample

            if (smoothing)
            {
                arma::uvec idx = sample(N, M, weights, true, true);   // M x 1
                arma::mat theta_last = Theta_stored.slice(dim.nT + B - 1); // p x N
                arma::mat theta_sub = theta_last.cols(idx);                     // p x M
                psi_smooth.row(dim.nT + B - 1) = theta_sub.row(0);
                
                if (infer_W)
                {
                    arma::vec Wlast = W_stored.col(dim.nT + B - 1);
                    W_smooth.col(dim.nT + B - 1) = Wlast.elem(idx);
                }
                else
                {
                    W_smooth.col(dim.nT + B - 1) = W_stored.col(dim.nT + B - 1);
                }
                

                for (unsigned int t = dim.nT + B - 1; t > B - 1; t--)
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
        arma::mat aw, bw; // N x (nT + 1)
        arma::mat W_stored; // N x (nT + 1)
        arma::mat W_smooth; // M x (nT + 1)
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