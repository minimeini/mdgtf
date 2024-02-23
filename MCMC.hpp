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
        static void update_wt( // Checked. Consist with `mcmc_disturbance_poisson.cpp`.
            arma::vec &wt, // (nT + 1) x 1
            arma::vec &wt_accept,
            arma::vec &Bs,
            arma::vec &logp,
            const arma::vec &y,         // (nT + 1) x 1
            Model &model,
            const Dist &w0_prior,
            const double &mh_sd = 0.1)
        {
            ApproxDisturbance approx_dlm(model.dim.nT, model.transfer.fgain.name);
            approx_dlm.set_Fphi(model.transfer.dlag, model.dim.nL);
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
                // approx_dlm.update_by_wt(y, wt);
                // arma::vec eta = approx_dlm.get_eta_approx(model.dobs.par1); // nT x 1, f0, Fn and psi is updated
                // arma::vec lambda = LinkFunc::ft2mu<arma::vec>(eta, model.flink.name, 0.); // nT x 1
                // arma::vec Vt_hat = ApproxDisturbance::func_Vt_approx(
                //     lambda, model.dobs, model.flink.name); // nT x 1

                // arma::mat Fn = approx_dlm.get_Fn(); // nT x nT
                // arma::vec Fnt = Fn.col(t - 1);
                // arma::vec Fnt2 = Fnt % Fnt;

                // arma::vec tmp = Fnt2 / Vt_hat;
                // double mh_prec = arma::accu(tmp);
                // // mh_prec = std::abs(mh_prec) + 1. / w0_prior.par2 + EPS;

                // Bs.at(t) = 1. / mh_prec;
                // double Btmp = std::sqrt(Bs.at(t));
                double Btmp = prior_sd;
                Btmp *= mh_sd;
                // Btmp = std::min(Btmp, 10.);

                double wt_new = R::rnorm(wt_old, Btmp); // Sample from MH proposal
                bound_check(wt_new, "Posterior::update_wt: wt_new");
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

                logp.at(t) = logps;
            }
        } // func update_wt

        static double update_W(
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
        Disturbance(){init_default();}
        Disturbance(const Rcpp::List &opts, const Dim &dim){ init(opts, dim); }

        void init_default()
        {
            double mh_sd = 0.1;
            unsigned int nburnin = 100;
            unsigned int nthin = 1;
            unsigned int nsample = 100;
            ntotal = nburnin + nthin * nsample + 1;

            W_prior.init("invgamma", 0.01, 0.01);

            bool infer_wt = true;
            bool infer_W = false;
            bool infer_mu0 = false;

            dim.init_default();
            w0_prior.init("gaussian", 0., 0.1);
            wt_accept.set_size(dim.nT + 1);
            wt_accept.zeros();
            wt = arma::randn(dim.nT + 1) * 0.01;
            wt.at(0) = 0.;
            for (unsigned int t = 1; t <= dim.nP; t++)
            {
                wt.at(t) = std::abs(wt.at(t));
            }
            bound_check(wt, "Disturbance::init_default");


            wt_stored.set_size(dim.nT + 1, nsample);

            W = 0.01;
            W_stored.set_size(nsample);

            mu0 = 0.;
            mu0_stored.set_size(nsample);

        }

        void init(const Rcpp::List &mcmc_settings, const Dim &dim_in)
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

            infer_wt = true;
            wt = arma::randn(dim.nT + 1) * 0.01;
            wt.at(0) = 0.;
            for (unsigned int t = 1; t <= dim.nP; t++)
            {
                wt.at(t) = std::abs(wt.at(t));
            }

            wt_stored.set_size(dim.nT + 1, nsample);

            wt_accept.set_size(dim.nT + 1);
            wt_accept.zeros();

            w0_prior.update_par1(0.01);
            w0_prior.update_par2(0.01);

            bound_check(wt, "Disturbance::init");

            if (opts.containsElementNamed("infer_wt"))
            {
                infer_wt = Rcpp::as<bool>(opts["infer_wt"]);
            }

            if (opts.containsElementNamed("wt"))
            {
                Rcpp::List wt_opts = Rcpp::as<Rcpp::List>(opts["wt"]);
                init_wt(wt_opts);
            }


            W = 0.01;
            infer_W = false;
            W_prior.init("invgamma", 0.01, 0.01);
            if (opts.containsElementNamed("W"))
            {
                Rcpp::List Wopts = Rcpp::as<Rcpp::List>(opts["W"]);
                init_W(Wopts);
            }
            W_stored.set_size(nsample);

            dim = dim_in;


            mu0 = 0.;
            mu0_stored.set_size(nsample);
            infer_mu0 = false;
            if (opts.containsElementNamed("infer_mu0"))
            {
                infer_mu0 = Rcpp::as<bool>(opts["infer_mu0"]);
            }


            return;
        }


        void init_W(const Rcpp::List &W_settings)
        {
            Rcpp::List Wopts = W_settings;
            if (Wopts.containsElementNamed("prior"))
            {
                Rcpp::NumericVector para = Rcpp::as<Rcpp::NumericVector>(Wopts["prior"]);
                W_prior.update_par1(para[0]);
                W_prior.update_par2(para[1]);
            }
            else
            {
                W_prior.update_par1(0.01);
                W_prior.update_par2(0.01);
            }

            if (Wopts.containsElementNamed("init"))
            {
                W = Rcpp::as<double>(Wopts["init"]);
            }
            else
            {
                W = 0.01;
            }

            if (Wopts.containsElementNamed("infer"))
            {
                infer_W = Rcpp::as<bool>(Wopts["infer"]);
            }
            else
            {
                infer_W = false;
            }

            if (Wopts.containsElementNamed("type"))
            {
                std::string prior_name = Rcpp::as<std::string>(Wopts["type"]);
                W_prior.init(prior_name, W_prior.par1, W_prior.par2);
            }
            else
            {
                W_prior.init("invgamma", W_prior.par1, W_prior.par2);
            }
        }

        void init_wt(const Rcpp::List &wt_settings)
        {
            wt = arma::randn(dim.nT + 1) * 0.01;
            for (unsigned int t = 1; t <= dim.nP; t++)
            {
                wt.at(t) = std::abs(wt.at(t));
            }
            wt_stored.set_size(dim.nT + 1, nsample);
            wt_accept.set_size(dim.nT + 1);
            wt_accept.zeros();

            Rcpp::List opts = wt_settings;
            w0_prior.init("gaussian", 0., 0.01);
            if (opts.containsElementNamed("prior"))
            {
                Rcpp::NumericVector para = Rcpp::as<Rcpp::NumericVector>(opts["prior"]);
                w0_prior.update_par1(para[0]);
                w0_prior.update_par2(para[1]);
            }

            wt.at(0) = 0.;
            if (opts.containsElementNamed("init"))
            {
                wt.at(0) = Rcpp::as<double>(opts["init"]);
            }

            infer_wt = true;
            if (opts.containsElementNamed("infer"))
            {
                infer_wt = Rcpp::as<bool>(opts["infer"]);
            }


            bound_check(wt, "Disturbance::init_wt");
        }

        static Rcpp::List default_settings()
        {    
            Rcpp::List Wopts;
            Wopts["infer"] = false;
            Wopts["init"] = 0.01;
            Wopts["prior"] = Rcpp::NumericVector::create(0.01, 0.01);
            Wopts["type"] = "invgamma";

            Rcpp::List wt_opts;
            wt_opts["infer"] = true;
            wt_opts["init"] = 0.;
            wt_opts["prior"] = Rcpp::NumericVector::create(0., 1.);

            Rcpp::List opts;
            opts["W"] = Wopts;
            opts["wt"] = wt_opts;
            
            opts["mh_sd"] = 0.1;
            opts["nburnin"] = 100;
            opts["nthin"] = 1;
            opts["nsample"] = 100;
            opts["infer_wt"] = true;
            opts["infer_W"] = false;

            return opts;
        }

        void save(bool &saveiter, unsigned int &idx_run, const unsigned int &b)
        {
            saveiter = b > nburnin && ((b - nburnin - 1) % nthin == 0);
            saveiter = saveiter || b == (ntotal - 1);

            if (b == (ntotal - 1))
            {
                idx_run = nsample - 1;
            }
            else
            {
                idx_run = (b - nburnin - 1) / nthin;
            }

            return;
        }

        void infer(
            Model &model, 
            const arma::vec &y // (nT + 1) x 1
            )
        {
            LBA::LinearBayes linear_bayes(model, y);
            linear_bayes.filter();
            linear_bayes.smoother();
            arma::mat psi_tmp = LBA::get_psi(linear_bayes.atilde, linear_bayes.Rtilde);
            arma::vec wt_init = arma::diff(psi_tmp.col(1));
            wt.tail(wt_init.n_elem) = wt_init;

            mu0 = model.dobs.par1;
            // W = 0.01;
            w0_prior.update_par2(W);

            for (unsigned int b = 0; b < ntotal; b++)
            {
                R_CheckUserInterrupt();

                arma::vec Bs(dim.nT + 1, arma::fill::zeros);
                arma::vec logp(dim.nT + 1, arma::fill::zeros);
                Posterior::update_wt(wt, wt_accept, Bs, logp, y, model, w0_prior, mh_sd);

                if (infer_W)
                {
                    double W_old = W;
                    W = Posterior::update_W(W_accept, W_old, wt, W_prior, mh_sd);
                    w0_prior.update_par2(W);
                }

                if (infer_mu0)
                {
                    double mu0_accept = 0.;
                    double logp_mu0 = 0.;
                    Posterior::update_mu0(mu0, mu0_accept, logp_mu0, y, wt, model, mh_sd);
                    model.dobs.update_par1(mu0);
                }


                bool saveiter = false;
                unsigned int idx_run = 0;
                save(saveiter, idx_run, b);
                
                if (saveiter)
                {
                    wt_stored.col(idx_run) = wt;
                    W_stored.at(idx_run) = W;
                    mu0_stored.at(idx_run) = mu0;
                }

                Rcpp::Rcout << "\rProgress: " << b << "/" << ntotal - 1;
            } // end a single iteration

            Rcpp::Rcout << std::endl;
            return;
        }

        Rcpp::List get_output()
        {
            Rcpp::List output;
            output["wt_stored"] = Rcpp::wrap(wt_stored); // (nT + 1) x nsample

            arma::mat psi_stored = arma::cumsum(wt_stored, 1); // (nT + 1) x nsample
            arma::vec qprob = {0.025, 0.5, 0.975};
            arma::mat psi_quantile = arma::quantile(psi_stored, qprob, 1); // (nT + 1) x 3
            output["psi"] = Rcpp::wrap(psi_quantile);
            output["wt_accept"] = Rcpp::wrap(wt_accept / ntotal);

            output["infer_W"] = infer_W;
            output["W_stored"] = Rcpp::wrap(W_stored);

            output["infer_mu0"] = infer_mu0;
            output["mu0_stored"] = Rcpp::wrap(mu0_stored);

            return output;
        }

    private:
        double mh_sd = 0.1;
        unsigned int nburnin = 100;
        unsigned int nthin = 1;
        unsigned int nsample = 100;
        unsigned int ntotal = 200;

        Dim dim;

        Dist w0_prior;
        bool infer_wt = true;
        arma::vec wt;
        arma::vec wt_accept;
        arma::mat wt_stored;

        
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