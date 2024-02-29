#ifndef _MCMC_HPP
#define _MCMC_HPP

#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <algorithm>
#include <RcppArmadillo.h>
#include "Model.hpp"
#include "LinkFunc.hpp"
#include "LinearBayes.hpp"

namespace MCMC
{
    class Posterior
    {
    public:
        static void update_wt( // Checked. OK.
            arma::vec &wt, // (nT + 1) x 1
            arma::vec &wt_accept,
            ApproxDisturbance &approx_dlm, 
            const arma::vec &y,         // (nT + 1) x 1
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
                for (unsigned int i = t; i <= model.dim.nT; i ++)
                {
                    logp_old += ObsDist::loglike(y.at(i), model.dobs.name, lam.at(i), model.dobs.par2, true);
                } // Checked. OK.

                logp_old += R::dnorm4(wt_old, w0_prior.par1, prior_sd, true);
                


                /*
                Metropolis-Hastings
                */
                approx_dlm.update_by_wt(y, wt);
                arma::vec eta = approx_dlm.get_eta_approx(model.dobs.par1); // nT x 1, f0, Fn and psi is updated
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
                for (unsigned int i = t; i <= model.dim.nT; i ++)
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
            std::map<std::string, AVAIL::Dist> W_prior_list = AVAIL::W_prior_list;

            switch (W_prior_list[W_prior.name])
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
                W = 1. / R::rgamma(0.5 *W_prior.par1, 2. / nSw_new); // prior_params.at(0) = nw_new
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

        static void update_mu0(
            double &mu0,
            double &mu0_accept,
            double &logp_mu0,
            const arma::vec &y, // nobs x 1
            const arma::vec &wt,
            const Model &model,
            const double &mh_sd = 1.)
        {
            double mu0_old = mu0;

            ApproxDisturbance approx_dlm(model.dim.nT, model.transfer.fgain.name);
            approx_dlm.set_Fphi(model.transfer.dlag, model.dim.nL);

            approx_dlm.update_by_wt(y, wt);
            arma::vec eta = approx_dlm.get_eta_approx(mu0_old); // f0, Fn and psi is updated
            arma::vec lambda = LinkFunc::ft2mu<arma::vec>(eta, model.flink.name, 0.);

            arma::vec Vt_hat = ApproxDisturbance::func_Vt_approx(lambda, model.dobs, model.flink.name);

            
            double logp_old = 0.;
            for (unsigned int i = 0; i <= model.dim.nT; i++)
            {
                logp_old += ObsDist::loglike(y.at(i), model.dobs.name, lambda.at(i), model.dobs.par2, true);
            }

            arma::vec tmp = 1. / Vt_hat;
            double mu0_prec = arma::accu(tmp);
            double mu0_var = 1. / mu0_prec;
            double mu0_sd = std::sqrt(mu0_var);
            double mu0_new = R::rnorm(mu0_old, mu0_sd * mh_sd);

            logp_mu0 = logp_old;
            if (mu0_new > -EPS) // non-negative
            {
                eta = approx_dlm.get_eta_approx(mu0_new); // f0, Fn and psi is updated
                // arma::vec lambda = LinkFunc::ft2mu(eta, model.flink.name, 0.);
                lambda = eta;
                double logp_new = 0.;
                for (unsigned int i = 0; i <= model.dim.nT; i++)
                {
                    logp_new += ObsDist::loglike(y.at(i), model.dobs.name, lambda.at(i), model.dobs.par2, true);
                }
                double logratio = std::min(0., logp_new - logp_old);
                if (std::log(R::runif(0., 1.)) < logratio)
                { // accept
                    mu0 = mu0_new;
                    mu0_accept += 1.;
                    logp_mu0 = logp_new;
                }
            }
        
            bound_check(mu0, "Posterior::update_mu0");
        } // func update_mu0
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

            W = 0.01;
            infer_W = true;
            W_prior.init("invgamma", 0.01, 0.01);
            W_stored.set_size(nsample);
            if (opts.containsElementNamed("W"))
            {
                Rcpp::List Wopts = Rcpp::as<Rcpp::List>(opts["W"]);
                init_param(infer_W, W, W_prior, Wopts);
            }

            mu0 = 0.;
            mu0_stored.set_size(nsample);
            infer_mu0 = false;
            if (opts.containsElementNamed("mu0"))
            {
                Rcpp::List param_opts = Rcpp::as<Rcpp::List>(opts["mu0"]);
                init_param(infer_mu0, mu0, mu0_prior, param_opts);
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


        static void init_param(bool &infer, double &init, Dist &prior, const Rcpp::List &opts)
        {
            Rcpp::List param_opts = opts;

            if (param_opts.containsElementNamed("infer"))
            {
                infer = Rcpp::as<bool>(param_opts["infer"]);
            }

            if (param_opts.containsElementNamed("init"))
            {
                init = Rcpp::as<double>(param_opts["init"]);
            }

            std::string prior_name = "invgamma";
            if (param_opts.containsElementNamed("prior_name"))
            {
                prior_name = Rcpp::as<std::string>(param_opts["prior_name"]);
            }


            Rcpp::NumericVector param = {0.01, 0.01};
            if (param_opts.containsElementNamed("prior_param"))
            {
                param = Rcpp::as<Rcpp::NumericVector>(param_opts["prior_param"]);
            }

            prior.init(prior_name, param[0], param[1]);
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

            Rcpp::List opts;
            opts["W"] = Wopts;
            opts["mu0"] = mu0_opts;
            
            opts["mh_sd"] = 0.1;
            opts["nburnin"] = 100;
            opts["nthin"] = 1;
            opts["nsample"] = 100;

            opts["num_step_ahead_forecast"] = 0;

            return opts;
        }

        Rcpp::List get_output()
        {
            Rcpp::List output;
            output["wt"] = Rcpp::wrap(wt_stored); // (nT + 1) x nsample

            arma::mat psi_stored = arma::cumsum(wt_stored, 0); // (nT + 1) x nsample
            arma::vec qprob = {0.025, 0.5, 0.975};
            arma::mat psi_quantile = arma::quantile(psi_stored, qprob, 1); // (nT + 1) x 3
            output["psi"] = Rcpp::wrap(psi_quantile);
            output["wt_accept"] = Rcpp::wrap(wt_accept / ntotal);

            output["infer_W"] = infer_W;
            output["W"] = Rcpp::wrap(W_stored);

            output["infer_mu0"] = infer_mu0;
            output["mu0"] = Rcpp::wrap(mu0_stored);

            output["model"] = model_info;

            return output;
        }

        Rcpp::List forecast(const Model &model)
        {
            arma::mat psi_stored = arma::cumsum(wt_stored, 0); // (nT + 1) x nsample
            Rcpp::List out = Model::forecast(
                y, psi_stored, W_stored, model, nforecast
            );

            return out;
        }

        Rcpp::List forecast_error(const Model &model, const std::string &loss_func = "quadratic")
        {
            arma::mat psi_stored = arma::cumsum(wt_stored, 0); // (nT + 1) x nsample
            return Model::forecast_error(psi_stored, y, model, loss_func);
        }

        Rcpp::List fitted_error(const Model &model, const std::string &loss_func = "quadratic")
        {
            arma::mat psi_stored = arma::cumsum(wt_stored, 0); // (nT + 1) x nsample
            return Model::fitted_error(psi_stored, y, model, loss_func);
        }

        void infer(Model &model)
        {
            model_info = model.info();

            // LBA::LinearBayes linear_bayes(model, y);
            // linear_bayes.filter();
            // linear_bayes.smoother();
            // arma::mat psi_tmp = LBA::get_psi(linear_bayes.atilde, linear_bayes.Rtilde);
            // arma::vec wt_init = arma::diff(psi_tmp.col(1));

            // wt.tail(wt_init.n_elem) = wt_init;

            ApproxDisturbance approx_dlm(model.dim.nT, model.transfer.fgain.name);
            approx_dlm.set_Fphi(model.transfer.dlag, model.dim.nL);

            for (unsigned int b = 0; b < ntotal; b++)
            {
                R_CheckUserInterrupt();

                Posterior::update_wt(wt, wt_accept, approx_dlm, y, model, w0_prior, mh_sd);

                if (infer_W)
                {
                    double W_old = W;
                    W = Posterior::update_W(W_accept, W_old, wt, W_prior, mh_sd);
                    w0_prior.update_par2(W);
                    model.derr.update_par1(W);
                }

                if (infer_mu0)
                {
                    double mu0_accept = 0.;
                    double logp_mu0 = 0.;
                    Posterior::update_mu0(mu0, mu0_accept, logp_mu0, y, wt, model, mh_sd);
                    model.dobs.update_par1(mu0);
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
                }


                Rcpp::Rcout << "\rProgress: " << b << "/" << ntotal - 1;
            } // end a single iteration

            Rcpp::Rcout << std::endl;
            return;
        }

        

    private:
        double mh_sd = 0.1;
        unsigned int nburnin = 100;
        unsigned int nthin = 1;
        unsigned int nsample = 100;
        unsigned int ntotal = 200;
        unsigned int nforecast = 0;

        Rcpp::List model_info;

        Dim dim;
        arma::vec y;

        Dist w0_prior;
        arma::vec wt;
        arma::vec wt_accept; // nsample x 1
        arma::mat wt_stored; // (nT + 1) x nsample

        Dist mu0_prior;
        bool infer_mu0 = false;
        double mu0 = 0.;
        arma::vec mu0_stored;

        Dist W_prior;
        bool infer_W = false;
        double W = 0.01;
        arma::vec W_stored;
        double W_accept = 0.;

        
    };
}



#endif