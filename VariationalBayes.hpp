#ifndef _VARIATIONALBAYES_HPP
#define _VARIATIONALBAYES_HPP

#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <algorithm>
#include <RcppArmadillo.h>
#include "Model.hpp"
#include "LinkFunc.hpp"
#include "yjtrans.h"
#include "LinearBayes.hpp"
#include "SequentialMonteCarlo.hpp"

// [[Rcpp::depends(RcppArmadillo)]]

namespace VB
{
    class VariationalBayes
    {
    public:
        unsigned int nsample = 1000;
        unsigned int nthin = 2;
        unsigned int nburnin = 1000;
        unsigned int ntotal = 3001;
        unsigned int nforecast = 0;
        unsigned int nforecast_err = 10; // forecasting for indices (1, ..., ntime-1) has `nforecast` elements
        double tstart_pct = 0.9;

        unsigned int N = 500; // number of SMC particles

        bool update_static = true;
        unsigned int m = 1; // number of unknown static parameters
        std::vector<std::string> param_selected = {"W"};

        Dim dim;

        arma::vec y;

        arma::vec psi;        // (nT + 1) x 1
        arma::mat psi_stored; // (nT + 1) x nsample

        double W = 0.01;
        double mu0 = 0.;
        double rho = 30;
        double par1 = 0.4;
        double par2 = 6;

        Dist W_prior, mu0_prior, rho_prior, par1_prior, par2_prior;

        arma::vec W_stored;    // nsample x 1
        arma::vec mu0_stored;  // nsample x 1
        arma::vec rho_stored;  // nsample x 1
        arma::vec par1_stored; // nsample x 1
        arma::vec par2_stored; // nsample x 1

        VariationalBayes()
        {
            dim.init_default();
            psi.set_size(dim.nT + 1);
            psi.zeros();

            y = psi;
        };

        VariationalBayes(const Model &model, const arma::vec y_in)
        {
            dim = model.dim;
            y = y_in;

            psi.set_size(dim.nT + 1);
            psi.zeros();
        }

        void init(const Rcpp::List &vb_opts)
        {
            Rcpp::List opts = vb_opts;

            nsample = 1000;
            if (opts.containsElementNamed("nsample"))
            {
                nsample = Rcpp::as<unsigned int>(opts["nsample"]);
            }

            nthin = 1000;
            if (opts.containsElementNamed("nthin"))
            {
                nthin = Rcpp::as<unsigned int>(opts["nthin"]);
            }

            nburnin = 1000;
            if (opts.containsElementNamed("nburnin"))
            {
                nburnin = Rcpp::as<unsigned int>(opts["nburnin"]);
            }

            ntotal = nburnin + nthin * nsample + 1;

            N = 500;
            if (opts.containsElementNamed("num_particle"))
            {
                N = Rcpp::as<unsigned int>(opts["num_particle"]);
            }

            nforecast = 0;
            if (opts.containsElementNamed("num_step_ahead_forecast"))
            {
                nforecast = Rcpp::as<unsigned int>(opts["num_step_ahead_forecast"]);
            }

            nforecast_err = 10; // forecasting for indices (1, ..., ntime-1) has `nforecast` elements
            if (opts.containsElementNamed("num_eval_forecast_error"))
            {
                nforecast_err = Rcpp::as<unsigned int>(opts["num_eval_forecast_error"]);
            }

            tstart_pct = 0.9;
            if (opts.containsElementNamed("tstart_pct"))
            {
                tstart_pct = Rcpp::as<double>(opts["tstart_pct"]);
            }

            psi_stored.set_size(psi.n_elem, nsample);
            psi_stored.zeros();

            param_selected.clear();
            m = 0;

            W_stored.set_size(nsample);
            W_stored.zeros();
            bool infer_W = true;
            W_prior.init("invgamma", 0.01, 0.01);
            W = 0.01;
            if (opts.containsElementNamed("W"))
            {
                Rcpp::List param_opts = Rcpp::as<Rcpp::List>(opts["W"]);
                init_param(infer_W, W, W_prior, param_opts);
            }
            if (infer_W)
            {
                param_selected.push_back("W");
                m += 1;
            }

            mu0_stored = W_stored;
            bool infer_mu0 = false;
            mu0_prior.init("gaussian", 0., 10.);
            mu0 = 0.;
            if (opts.containsElementNamed("mu0"))
            {
                Rcpp::List param_opts = Rcpp::as<Rcpp::List>(opts["mu0"]);
                init_param(infer_mu0, mu0, mu0_prior, param_opts);
            }
            if (infer_mu0)
            {
                param_selected.push_back("mu0");
                m += 1;
            }

            rho_stored = W_stored;
            bool infer_rho = false;
            rho_prior.init("gamma", 0.1, 0.1);
            rho = 30.;
            if (opts.containsElementNamed("rho"))
            {
                Rcpp::List param_opts = Rcpp::as<Rcpp::List>(opts["rho"]);
                init_param(infer_rho, rho, rho_prior, param_opts);
            }
            if (rho_prior.infer)
            {
                param_selected.push_back("rho");
                m += 1;
            }

            par1 = 0.;
            par1_stored.set_size(nsample);
            bool infer_par1 = false;
            par1_prior.init("gamma", 0.1, 0.1);
            if (opts.containsElementNamed("par1"))
            {
                Rcpp::List param_opts = Rcpp::as<Rcpp::List>(opts["par1"]);
                init_param(infer_par1, par1, par1_prior, param_opts);
            }
            if (par1_prior.infer)
            {
                param_selected.push_back("par1");
                m += 1;
            }

            par2 = 0.;
            par2_stored.set_size(nsample);
            bool infer_par2 = false;
            par2_prior.init("gamma", 0.1, 0.1);
            if (opts.containsElementNamed("par2"))
            {
                Rcpp::List param_opts = Rcpp::as<Rcpp::List>(opts["par2"]);
                init_param(infer_par2, par2, par2_prior, param_opts);
            }
            if (par2_prior.infer)
            {
                param_selected.push_back("par2");
                m += 1;
            }

            update_static = false;
            if (m > 0)
            {
                update_static = true;
            }
        }

        static Rcpp::List default_settings()
        {
            Rcpp::List W_opts;
            W_opts["infer"] = true;
            W_opts["init"] = 0.01;
            W_opts["prior_name"] = "invgamma";
            W_opts["prior_param"] = Rcpp::NumericVector::create(0.01, 0.01);

            Rcpp::List mu0_opts;
            mu0_opts["infer"] = false;
            mu0_opts["init"] = 0.;
            mu0_opts["prior_name"] = "gaussian";
            mu0_opts["prior_param"] = Rcpp::NumericVector::create(0., 10.);

            Rcpp::List rho_opts;
            rho_opts["infer"] = false;
            rho_opts["init"] = 30;
            rho_opts["prior_param"] = Rcpp::NumericVector::create(0.1, 0.1);
            rho_opts["prior_name"] = "gamma";

            Rcpp::List par1_opts;
            par1_opts["infer"] = false;
            par1_opts["init"] = 30;
            par1_opts["prior_param"] = Rcpp::NumericVector::create(0.1, 0.1);
            par1_opts["prior_name"] = "gamma";

            Rcpp::List par2_opts;
            par2_opts["infer"] = false;
            par2_opts["init"] = 30;
            par2_opts["prior_param"] = Rcpp::NumericVector::create(0.1, 0.1);
            par2_opts["prior_name"] = "gamma";

            Rcpp::List opts;
            opts["nsample"] = 1000;
            opts["nthin"] = 1;
            opts["nburnin"] = 1000;

            opts["num_particle"] = 500;
            opts["num_step_ahead_forecast"] = 0;
            opts["num_eval_forecast_error"] = 10; // forecasting for indices (1, ..., ntime-1) has `nforecast` elements
            opts["tstart_pct"] = 0.9;

            opts["W"] = W_opts;
            opts["mu0"] = mu0_opts;
            opts["rho"] = rho_opts;
            opts["par1"] = par1_opts;
            opts["par2"] = par2_opts;

            return opts;
        }

        Rcpp::List forecast(const Model &model)
        {
            Rcpp::List out = Model::forecast(y, psi_stored, W_stored, model, nforecast);
            return out;
        }

        // Rcpp::List forecast_error(
        //     const Model &model,
        //     const std::string &loss_func = "quadratic",
        //     const unsigned int &k = 1)
        // {
        //     return Model::forecast_error(psi_stored, y, model, loss_func, k);
        // }

        Rcpp::List fitted_error(const Model &model, const std::string &loss_func = "quadratic")
        {
            return Model::fitted_error(psi_stored, y, model, loss_func);
        }

        void fitted_error(double &err, const Model &model, const std::string &loss_func = "quadratic")
        {
            Model::fitted_error(err, psi_stored, y, model, loss_func);
            return;
        }
    };
    /**
     * @brief Gradient ascent.
     *
     */
    class HybridParams
    {
    public:
        HybridParams() : change(par_change)
        {
            learning_rate = 0.01;
            eps_step_size = 1.e-6;
        }

        HybridParams(
            const unsigned int &m_in,
            const double &learn_rate = 0.01,
            const double &eps_size = 1.e-6) : change(par_change)
        {
            init(m_in, learn_rate, eps_size);
        }

        HybridParams(
            const arma::vec &curEg2_init,     // m x 1
            const arma::vec &curEdelta2_init, // m x 1
            const double &learn_rate = 0.01,
            const double &eps_size = 1.e-6) : change(par_change)
        {
            init(curEg2_init, curEdelta2_init, learn_rate, eps_size);
        }

        void init(
            const unsigned int &m_in,
            const double &learn_rate = 0.01,
            const double &eps_size = 1.e-6)
        {
            m = m_in;
            learning_rate = learn_rate;
            eps_step_size = eps_size;

            par_change.set_size(m);
            par_change.zeros();

            curEg2 = par_change;
            curEdelta2 = par_change;
            rho = par_change;
        }

        void init(
            const arma::vec &curEg2_init,     // m x 1
            const arma::vec &curEdelta2_init, // m x 1
            const double &learn_rate = 0.01,
            const double &eps_size = 1.e-6)
        {
            m = curEg2_init.n_elem;
            learning_rate = learn_rate;
            eps_step_size = eps_size;

            curEg2 = curEg2_init;
            curEdelta2 = curEdelta2_init;

            par_change.set_size(m);
            par_change.zeros();
            rho = par_change;
        }

        void update_grad(const arma::vec &dYJinv_dVecPar) // Checked. OK.
        {
            arma::vec oldEg2 = curEg2;

            curEg2 = (1. - learning_rate) * oldEg2 + learning_rate * arma::pow(dYJinv_dVecPar, 2.); // m x 1
            try
            {
                bound_check<arma::vec>(curEg2, "curEg2");
            }
            catch (const std::exception &e)
            {
                oldEg2.t().print("\n oldEg2");
                dYJinv_dVecPar.t().print("\ndYJinv_dVecPar");
                curEg2.t().print("\n curEg2");

                std::cerr << e.what() << '\n';
                throw std::invalid_argument(e.what());
            }

            rho = arma::sqrt(curEdelta2 + eps_step_size) / arma::sqrt(curEg2 + eps_step_size);

            par_change = rho % dYJinv_dVecPar;
            try
            {
                bound_check<arma::vec>(par_change, "update_grad: par_change");
            }
            catch (const std::exception &e)
            {
                dYJinv_dVecPar.t().print("\ndYJinv_dVecPar");
                rho.t().print("\n rho");
                oldEg2.t().print("\n oldEg2");
                std::cout << "\n learn rate = " << learning_rate << " eps = " << eps_step_size << std::endl;
                throw std::invalid_argument(e.what());
            }

            arma::vec oldEdelta2 = curEdelta2;
            curEdelta2 = (1. - learning_rate) * oldEdelta2 + learning_rate * arma::pow(par_change, 2.);

            bound_check<arma::vec>(curEdelta2, "update_grad: curEdelta2");
            return;
        }

        const arma::vec &change;

    private:
        arma::vec curEg2;     // m x 1
        arma::vec curEdelta2; // m x 1
        arma::vec par_change;

        arma::vec rho; // m x 1, step size

        double learning_rate = 0.01;
        double eps_step_size = 1.e-6;
        unsigned int m;
    };

    class Hybrid : public VariationalBayes
    {
    public:
        Hybrid(const Model &model_in, const arma::vec &y_in) : VariationalBayes(model_in, y_in)
        {
            dim = model_in.dim;
            psi.set_size(dim.nT + 1);
            psi.zeros();
            y = y_in;

            m = 1;
            k = 1;
            learning_rate = 0.01;
            eps_step_size = 1.e-6;
        }

        Hybrid() : VariationalBayes() {}

        void init(const Rcpp::List &hvb_opts)
        {
            Rcpp::List opts = hvb_opts;
            VariationalBayes::init(opts);

            learning_rate = 0.01;
            if (opts.containsElementNamed("learning_rate"))
            {
                learning_rate = Rcpp::as<double>(opts["learning_rate"]);
            }

            eps_step_size = 1.e-6;
            if (opts.containsElementNamed("eps_step_size"))
            {
                eps_step_size = Rcpp::as<double>(opts["eps_step_size"]);
            }

            k = 1;
            if (opts.containsElementNamed("k"))
            {
                k = Rcpp::as<unsigned int>(opts["k"]);
            }
            k = std::min(k, m);

            gamma.set_size(m);
            gamma.ones();
            grad_tau.init(m, learning_rate, eps_step_size);

            mu.set_size(m);
            mu.zeros();
            grad_mu.init(m, learning_rate, eps_step_size);

            d.set_size(m);
            d.ones();
            grad_d.init(m, learning_rate, eps_step_size);

            eps = d;

            eta = init_eta(param_selected, W, mu0, rho, par1, par2, update_static); // Checked. OK.
            eta_tilde = eta2tilde(eta, param_selected, W_prior.name, par1_prior.name);

            nu = tYJ(eta_tilde, gamma);

            B.set_size(m, k);
            B.zeros();
            grad_vecB.init(m * k, learning_rate, eps_step_size);

            xi.set_size(k);
            xi.zeros();

            if (m > 1 && k > 1)
            {
                B_uptri_idx = arma::trimatu_ind(arma::size(B), 1);
            }
            else
            {
                B_uptri_idx = {0};
            }

            if (opts.containsElementNamed("smc"))
            {
                smc_opts = Rcpp::as<Rcpp::List>(opts["smc"]);
            }
            else
            {
                // smc_opts = SMC::MCS::default_settings();
                smc_opts = SMC::TFS::default_settings();
            }

            smc_opts["use_discount"] = false;
            smc_opts["num_smooth"] = Rcpp::as<unsigned int>(smc_opts["num_particle"]);
            smc_opts["resample_all"] = true;

            Rcpp::List W_opts = Rcpp::List::create(
                Rcpp::Named("infer") = false,
                Rcpp::Named("init") = W_prior.val);

            unsigned int num_particle = Rcpp::as<unsigned int>(smc_opts["num_particle"]);
            mu0_stored2.set_size(num_particle, nsample);
            mu0_stored2.zeros();
        }

        static Rcpp::List default_settings()
        {
            Rcpp::List opts = VariationalBayes::default_settings();
            opts["learning_rate"] = 0.01;
            opts["eps_step_size"] = 1.e-6;
            opts["k"] = 1;

            // Rcpp::List smc_tmp = SMC::MCS::default_settings();
            // smc_tmp["use_discount"] = false;
            // smc_tmp["num_backward"] = 5;

            Rcpp::List smc_tmp = SMC::TFS::default_settings();
            smc_tmp["resample_all"] = true;
            smc_tmp["use_discount"] = false;

            Rcpp::List W_opts = Rcpp::List::create(
                Rcpp::Named("infer") = false,
                Rcpp::Named("init") = 0.01);

            smc_tmp["W"] = W_opts;

            Rcpp::List mu0_opts = Rcpp::List::create(
                Rcpp::Named("infer") = false,
                Rcpp::Named("init") = 0);

            smc_tmp["mu0"] = mu0_opts;

            opts["smc"] = smc_tmp;

            return opts;
        }

        static arma::vec eta2tilde( // Checked. OK.
            const arma::vec &eta,   // m x 1
            const std::vector<std::string> &param_selected,
            const std::string &W_prior = "invgamma",
            const std::string &lag_par1_prior = "gaussian")
        {
            std::map<std::string, AVAIL::Dist> dist_list = AVAIL::dist_list;
            std::map<std::string, AVAIL::Param> static_param_list = AVAIL::static_param_list;

            arma::vec eta_tilde = eta;
            for (unsigned int i = 0; i < param_selected.size(); i++)
            {
                double val = eta.at(i);
                switch (static_param_list[tolower(param_selected[i])])
                {
                case AVAIL::Param::W:
                {
                    switch (dist_list[tolower(W_prior)])
                    {
                    case AVAIL::Dist::invgamma:
                    {
                        eta_tilde.at(i) = -std::log(std::abs(val) + EPS);
                        break;
                    }
                    case AVAIL::Dist::gamma:
                    {
                        eta_tilde.at(i) = std::log(std::abs(val) + EPS);
                        break;
                    }
                    default:
                    {
                        throw std::invalid_argument("VB::Hybrid::eta2tilde: undefined prior " + W_prior + " for W.");
                        break;
                    }
                    } // switch W prior.
                    break;
                } // W
                case AVAIL::Param::mu0:
                {
                    bound_check(val, "VB::Hybrid::eta2tilde: mu0", false, true);
                    eta_tilde.at(i) = std::log(std::abs(val) + EPS);
                    break;
                } // mu0
                case AVAIL::Param::rho:
                {
                    eta_tilde.at(i) = std::log(std::abs(val) + EPS);
                    break;
                } // rho
                case AVAIL::Param::lag_par1:
                {
                    switch (dist_list[tolower(lag_par1_prior)])
                    {
                    case AVAIL::Dist::gaussian:
                    {
                        eta_tilde.at(i) = val;
                        break;
                    }
                    case AVAIL::Dist::invgamma:
                    {
                        eta_tilde.at(i) = std::log(std::abs(val) + EPS);
                        break;
                    }
                    case AVAIL::Dist::gamma:
                    {
                        eta_tilde.at(i) = std::log(std::abs(val) + EPS);
                        break;
                    }
                    case AVAIL::Dist::beta:
                    {
                        eta_tilde.at(i) = std::log(std::abs(val) + EPS) - std::log(std::abs(1. - val) + EPS);
                        break;
                    }
                    default:
                    {
                        throw std::invalid_argument("VB::Hybrid::eta2tilde: undefined prior " + lag_par1_prior + " for first parameter of lag distribution.");
                        break;
                    }
                    } // switch prior type for lag_par1.
                    break;
                } // lag_par1
                case AVAIL::Param::lag_par2:
                {
                    eta_tilde.at(i) = std::log(std::abs(val) + EPS);
                    break;
                } // lag_par2
                default:
                {
                    throw std::invalid_argument("VB::Hybrid::eta2tilde: undefined static parameter " + param_selected[i]);
                    break;
                }
                } // switch param

                bound_check(eta_tilde.at(i), "eta2tilde: eta_tilde");
            }

            return eta_tilde;
        }

        static arma::vec tilde2eta(     // Checked. OK.
            const arma::vec &eta_tilde, // m x 1
            const std::vector<std::string> &param_selected,
            const std::string &W_prior = "invgamma",
            const std::string &lag_par1_prior = "gaussian")
        {
            std::map<std::string, AVAIL::Dist> dist_list = AVAIL::dist_list;
            std::map<std::string, AVAIL::Param> static_param_list = AVAIL::static_param_list;

            arma::vec eta = eta_tilde;
            for (unsigned int i = 0; i < param_selected.size(); i++)
            {
                double val = eta_tilde.at(i);
                switch (static_param_list[tolower(param_selected[i])])
                {
                case AVAIL::Param::W:
                {
                    switch (dist_list[tolower(W_prior)])
                    {
                    case AVAIL::Dist::invgamma:
                    {
                        val *= -1.;
                        val = std::min(val, UPBND);
                        eta.at(i) = std::exp(val);
                        break;
                    }
                    case AVAIL::Dist::gamma:
                    {
                        val = std::min(val, UPBND);
                        eta.at(i) = std::exp(val);
                        break;
                    }
                    default:
                    {
                        break;
                    }
                    } // switch W prior.
                    break;
                } // W
                case AVAIL::Param::mu0:
                {
                    val = std::min(val, UPBND);
                    eta.at(i) = std::exp(val);
                    break;
                } // mu0
                case AVAIL::Param::rho:
                {
                    val = std::min(val, UPBND);
                    eta.at(i) = std::exp(val);
                    break;
                } // rho
                case AVAIL::Param::lag_par1:
                {
                    switch (dist_list[tolower(lag_par1_prior)])
                    {
                    case AVAIL::Dist::gaussian:
                    {
                        eta.at(i) = val;
                        break;
                    }
                    case AVAIL::Dist::invgamma:
                    {
                        val = std::min(val, UPBND);
                        eta.at(i) = std::exp(val);
                        break;
                    }
                    case AVAIL::Dist::gamma:
                    {
                        val = std::min(val, UPBND);
                        eta.at(i) = std::exp(val);
                        break;
                    }
                    case AVAIL::Dist::beta:
                    {
                        val = std::min(val, UPBND);
                        double tmp = std::exp(val);
                        eta.at(i) = tmp;
                        eta.at(i) /= (1. + tmp);
                        break;
                    }
                    default:
                    {
                        throw std::invalid_argument("VB::Hybrid::tilde2eta: undefined prior " + lag_par1_prior + " for first parameter of lag distribution.");
                        break;
                    }
                    } // switch prior type for lag_par1.
                    break;
                } // lag_par1
                case AVAIL::Param::lag_par2:
                {
                    val = std::min(val, UPBND);
                    eta.at(i) = std::exp(val);
                    break;
                } // lag_par2
                default:
                {
                    throw std::invalid_argument("VB::Hybrid::tilde2eta: undefined static parameter " + param_selected[i]);
                    break;
                }
                } // switch param

                bound_check(eta.at(i), "tilde2eta: eta");
            }

            return eta;
        }

        /*
        ------ dlogJoint_dWtilde ------
        The derivative of the full joint density with respect to Wtilde=log(1/W), where W
        is the evolution variance that affects
            psi[t] | psi[t-1] ~ N(psi[t-1], W)

        Therefore, this function remains unchanged as long as we using the same evolution equation for psi.
        */
        static double dlogJoint_dWtilde( // Checked. OK.
            const arma::vec &psi,        // (n+1) x 1, (psi[0],psi[1],...,psi[n])
            const double G,              // evolution transition matrix
            const double W,              // evolution variance conditional on V
            const Dist &W_prior)
        { // 0 - Gamma(aw=shape,bw=rate); 1 - Half-Cauchy(aw=location=0, bw=scale); 2 - Inverse-Gamma(aw=shape,bw=rate)
            std::map<std::string, AVAIL::Dist> dist_list = AVAIL::dist_list;

            double aw = W_prior.par1;
            double bw = W_prior.par2;

            if (!psi.is_finite() || !std::isfinite(W))
            {
                throw std::invalid_argument("dlogJoint_dWtilde<double>: non-finite psi or W input.");
            }

            const double n = static_cast<double>(psi.n_elem) - 1;
            const unsigned int ni = psi.n_elem - 1;
            double res = 0.;
            for (unsigned int t = 0; t < ni; t++)
            {
                res += (psi.at(t + 1) - G * psi.at(t)) * (psi.at(t + 1) - G * psi.at(t));
            }
            // res = sum((w[t])^2)

            double rdw = std::log(std::abs(res) + EPS) - std::log(std::abs(W) + EPS); // sum(w[t]^2) / W
            rdw = std::min(rdw, UPBND);
            rdw = std::exp(rdw);
            bound_check(rdw, "dlogJoint_dWtilde: rdw");

            double deriv;
            switch (dist_list[W_prior.name])
            {
            case AVAIL::Dist::invgamma:
            {
                /*
                W ~ Inverse-Gamma(aw=shape, bw=rate)
                Wtilde = -log(W)
                (deprecated)
                */
                // deriv = res;
                // deriv *= 0.5;
                // deriv += bw;
                // deriv = -rdw; // TYPO?
                // deriv += aw + 0.5*n;

                double bnew = bw + 0.5 * res;
                double log_bnew_W = std::log(std::abs(bnew) + EPS) - std::log(std::abs(W) + EPS);
                log_bnew_W = std::min(log_bnew_W, UPBND);
                deriv = -std::exp(log_bnew_W);

                double a_new = aw;
                a_new += 0.5 * n;
                a_new += 1.;
                deriv += a_new;
                break;
            }
            case AVAIL::Dist::gamma:
            {
                /*
                W ~ Gamma(aw=shape, bw=rate)
                Wtilde = log(W)
                */
                // deriv = 0.5*n-0.5/W*res - aw + bw*W;

                // deriv = aw - 0.5*n - bw*W + 0.5 * rdw;
                deriv = aw;
                deriv -= 0.5 * n;
                deriv -= bw * W;
                deriv += 0.5 * rdw;
                break;
            }
            case AVAIL::Dist::halfcauchy:
            {
                /*
                sqrt(W) ~ Half-Cauchy(aw=location==0, bw=scale)
                */
                deriv = 0.5 * n - 0.5 * rdw + W / (bw * bw + W) - 0.5;
                break;
            }
            default:
            {
                break;
            }
            } // switch W prior type

            bound_check(deriv, "dlogJoint_dWtilde: deriv");
            return deriv;
        }

        /**
         * @todo Move this to the corresponding distribution.
         * @brief
         *
         * @param Wtilde
         * @param W_prior
         * @return double
         */
        static double logprior_Wtilde( // Checked. OK.
            const double &Wtilde,      // evolution variance conditional on V
            const Dist &W_prior)
        {
            std::map<std::string, AVAIL::Dist> dist_list = AVAIL::dist_list;

            double InvGamma_cnst = W_prior.par1;
            InvGamma_cnst *= std::log(W_prior.par2);
            InvGamma_cnst -= std::lgamma(W_prior.par1);

            double logp = -16.;
            switch (dist_list[W_prior.name])
            {
            case AVAIL::Dist::invgamma:
            {
                /*
                W ~ Inverse-Gamma(aw=shape, bw=rate)
                Wtilde = -log(W)
                */
                logp = InvGamma_cnst;
                logp += W_prior.par1 * Wtilde;
                double Wast = std::min(Wtilde, UPBND);
                logp -= W_prior.par2 * std::exp(Wast);
                break;
            }
            case AVAIL::Dist::gamma:
            {
                /*
                W ~ Gamma(aw=shape, bw=rate)
                Wtilde = log(W)
                */
                logp = W_prior.par1 * std::log(W_prior.par2);
                logp -= std::lgamma(W_prior.par1);
                logp += W_prior.par1 * Wtilde;

                double Wast = std::min(Wtilde, UPBND);
                logp -= W_prior.par2 * std::exp(Wast);
                break;
            }
            default:
            {
                break;
            }
            } // switch by W prior type

            bound_check(logp, "logprior_Wtilde: logp");

            return logp;
        }

        /*
        ------ dlogJoint_dlogmu0 ------
        The derivative of the full joint density with respect to logmu0=log(mu0), where mu0
        is the baseline intensity such that:
            y[t] ~ EFM(lambda[t]=mu[0]+lambda[t])

        mu0 > 0 has a Gamma prior:
            mu[0] ~ Gamma(amu,bmu)
        */
        static double dloglike_dmu0tilde( // Checked. OK.
            const arma::vec &y,           // (nT+1) x 1
            const arma::vec &ft,          // (n+1) x 1, (f[0],f[1],...,f[nT])
            const ObsDist &dobs,
            const std::string &link_func)
        { // 0 - negative binomial; 1 - poisson

            double dy_dlambda = 0.;
            for (unsigned int t = 1; t < y.n_elem; t++)
            {
                double lambda = LinkFunc::ft2mu(ft.at(t), link_func, dobs.par1);
                dy_dlambda += ObsDist::dloglike_dlambda(y.at(t), lambda, dobs);
            }

            double dlambda_dmu = 1.;
            double dmu_dmu0tilde = dobs.par1;
            double deriv = (dy_dlambda * dlambda_dmu) * dmu_dmu0tilde;
            return deriv;
        }

        // double dloglike_dmu0tilde(
        //     const arma::vec &ypad,  // (n+1) x 1
        //     const arma::vec &theta, // (n+1) x 1, (theta[0],theta[1],...,theta[n])
        //     const arma::vec &hpsi_pad, // (n+1) x 1
        //     const arma::vec &lag_par,
        //     const arma::vec &obs_par,
        //     const unsigned int &nlag_in = 20,
        //     const unsigned int &obs_code = 0,
        //     const unsigned int &trans_code = 1,
        //     const bool &truncated = true)
        // { // 0 - negative binomial; 1 - poisson
        //     unsigned int nobs = ypad.n_elem - 1;
        //     double mu0 = obs_par.at(0);
        //     double delta_nb = obs_par.at(1);

        //     double dy_dmu0 = 0.;
        //     for (unsigned int t = 1; t <= nobs; t++)
        //     {
        //         double lambda = mu0 + theta.at(t);
        //         double dy_dlambda = dloglike_dlambda(
        //             ypad.at(t), lambda, delta_nb, obs_code);

        //         unsigned int nelem = theta_nelem(nobs, nlag_in, t, truncated);
        //         arma::vec Fphi_sub; arma::vec hpsi_sub;
        //         theta_subset(
        //             Fphi_sub, hpsi_sub,
        //             hpsi_pad, lag_par,
        //             t, nelem, trans_code);
        //         arma::vec coef = Fphi_times_hpsi(Fphi_sub, hpsi_sub);
        //         double dlambda_dmu0 = 1. - arma::accu(coef);

        //         dy_dmu0 += dy_dlambda * dlambda_dmu0;
        //     }

        //     double dmu0_dmu0tilde = mu0;
        //     double deriv = dy_dmu0 * dmu0_dmu0tilde;
        //     return deriv;
        // }

        static double logprior_mu0tilde(
            const double &logmu0, // evolution variance conditional on V
            const double &amu,
            const double &bmu)
        {

            /*
            mu0 ~ Gamma(aw=shape, bw=rate)
            logmu0 = log(mu0)
            */
            double logp = amu * std::log(bmu) - std::lgamma(amu) + amu * logmu0 - bmu * std::exp(logmu0);
            bound_check(logp, "logprior_logmu0: logp");
            return logp;
        }

        static double logprior_mu0tilde(
            const double &logmu0, // evolution variance conditional on V
            const double &sig2_mu0 = 10.)
        {
            /*
            log(mu0) ~ N(0,sig2_mu0)
            */
            double logp = -logmu0 / sig2_mu0;
            bound_check(logp, "logprior_logmu0: logp");
            return logp;
        }

        /**
         * @brief Derivative of the logarithm of joint probability with respect to parameters mapped to real-line (but before Yeo-Johnson transformation).
         *
         * @param y
         * @param psi
         * @param ft
         * @param eta
         * @param param_selected
         * @param W_prior
         * @param lag_par1_prior
         * @param lag_par2_prior
         * @param rho_prior
         * @param model
         * @return arma::vec
         */
        static arma::vec dlogJoint_deta( // Checked. OK.
            const arma::vec &y,          // (nT + 1) x 1
            const arma::vec &psi,        // (nT + 1) x 1
            const arma::vec &ft,         // (nT + 1) x 1
            const arma::vec &eta,        // m x 1
            const std::vector<std::string> &param_selected,
            const Dist &W_prior,
            const Dist &lag_par1_prior,
            const Dist &lag_par2_prior,
            const Dist &rho_prior,
            const Model &model)
        {
            std::map<std::string, AVAIL::Param> static_param_list = AVAIL::static_param_list;
            arma::vec deriv(param_selected.size());

            bool infer_dlag = lag_par1_prior.infer || lag_par2_prior.infer;
            arma::vec dllk_dpar, hpsi;
            if (infer_dlag || rho_prior.infer)
            {
                hpsi = GainFunc::psi2hpsi(psi, model.transfer.fgain.name);
                if (infer_dlag)
                {
                    dllk_dpar = Model::dloglik_dpar(y, hpsi, model);
                }
            }

            for (unsigned int i = 0; i < param_selected.size(); i++)
            {
                double val = eta.at(i);

                switch (static_param_list[tolower(param_selected[i])])
                {
                case AVAIL::Param::W:
                {
                    deriv.at(i) = dlogJoint_dWtilde(psi, 1., val, W_prior);
                    break;
                }
                case AVAIL::Param::mu0:
                {
                    double mu0tilde = std::log(val + EPS);

                    deriv.at(i) = dloglike_dmu0tilde(
                        y, ft, model.dobs, model.flink.name);
                    deriv.at(i) += logprior_mu0tilde(mu0tilde, 10.);

                    break;
                }
                case AVAIL::Param::rho:
                {
                    deriv.at(i) = Model::dlogp_dpar2_obs(model, y, hpsi, true);
                    deriv.at(i) += Prior::dlogprior_dpar(model.dobs.par2, rho_prior, true);
                    break;
                }
                case AVAIL::Param::lag_par1:
                {
                    deriv.at(i) = dllk_dpar.at(0);
                    deriv.at(i) += Prior::dlogprior_dpar(model.transfer.dlag.par1, lag_par1_prior, true);
                    break;
                }
                case AVAIL::Param::lag_par2:
                {
                    deriv.at(i) = dllk_dpar.at(1);
                    deriv.at(i) += Prior::dlogprior_dpar(model.transfer.dlag.par2, lag_par2_prior, true);
                    break;
                }
                default:
                {
                    break;
                }
                } // switch param
            }

            return deriv;
        }

        static arma::vec init_eta( // Checked. OK.
            const std::vector<std::string> &params_selected,
            const double &W,
            const double &mu0,
            const double &rho,
            const double &par1,
            const double &par2,
            const bool &update_static)
        {
            std::map<std::string, AVAIL::Param> static_param_list = AVAIL::static_param_list;
            arma::vec eta(params_selected.size(), arma::fill::zeros);

            if (!update_static)
            {
                return eta;
            }

            for (unsigned int i = 0; i < params_selected.size(); i++)
            {
                switch (static_param_list[params_selected[i]])
                {
                case AVAIL::Param::W:
                {
                    eta.at(i) = W;
                    break;
                }
                case AVAIL::Param::mu0:
                {
                    eta.at(i) = mu0;
                    break;
                }
                case AVAIL::Param::rho:
                {
                    eta.at(i) = rho;
                    break;
                }
                case AVAIL::Param::lag_par1:
                {
                    eta.at(i) = par1;
                    break;
                }
                case AVAIL::Param::lag_par2:
                {
                    eta.at(i) = par2;
                    break;
                }
                default:
                {
                    break;
                }
                } // switch by param
            }

            return eta;
        }

        static void update_params(
            double &W,
            double &mu0,
            double &rho,
            double &par1,
            double &par2,
            Model &model,
            const std::vector<std::string> &params_selected,
            const arma::vec &eta)
        {
            std::map<std::string, AVAIL::Param> static_param_list = AVAIL::static_param_list;
            bool update_dlag = false;

            for (unsigned int i = 0; i < params_selected.size(); i++)
            {
                double val = eta.at(i);
                switch (static_param_list[params_selected[i]])
                {
                case AVAIL::Param::W: // W is selected
                {
                    W = val;
                    model.derr.update_par1(val);
                    break;
                }
                case AVAIL::Param::mu0: // mu0 is selected
                {
                    mu0 = val;
                    model.dobs.update_par1(val);
                    break;
                }
                case AVAIL::Param::rho: // rho is selected
                {
                    rho = val;
                    model.dobs.update_par2(val);
                    break;
                }
                case AVAIL::Param::lag_par1: // par 1 is selected
                {
                    update_dlag = true;
                    par1 = val;
                    break;
                }
                case AVAIL::Param::lag_par2: // par 2 is selected
                {
                    update_dlag = true;
                    par2 = val;
                    break;
                }
                default:
                {
                    throw std::invalid_argument("VB::Hybrid::update_params: undefined static parameter " + params_selected[i]);
                    break;
                }
                }
            }

            if (update_dlag)
            {
                unsigned int nlag = model.update_dlag(par1, par2, 30, false);
            }

            return;
        }

        arma::mat optimal_learning_rate(
            const Model &model,
            const double &from,
            const double &to,
            const double &delta = 0.01,
            const std::string &loss = "quadratic",
            const bool &verbose = VERBOSE)
        {
            arma::vec grid = arma::regspace(from, delta, to);
            unsigned int nelem = grid.n_elem;
            arma::mat stats(nelem, 4, arma::fill::zeros);

            for (unsigned int i = 0; i < nelem; i++)
            {
                Rcpp::checkUserInterrupt();

                double lrate = grid.at(i);
                stats.at(i, 0) = lrate;

                gamma.set_size(m);
                gamma.ones();
                grad_tau.init(m, lrate, eps_step_size);

                mu.set_size(m);
                mu.zeros();
                grad_mu.init(m, lrate, eps_step_size);

                B.set_size(m, k);
                B.zeros();
                grad_vecB.init(m * k, lrate, eps_step_size);

                d.set_size(m);
                d.ones();
                grad_d.init(m, lrate, eps_step_size);

                infer(model);

                double forecast_loss = 0.;
                double forecast_cover = 0.;
                double forecast_width = 0.;
                Model::forecast_error(forecast_loss, forecast_cover, forecast_width, psi_stored, y, model, loss, false);
                stats.at(i, 1) = forecast_loss;

                // double fit_loss = 0.;
                // Model::fitted_error(fit_loss, psi_stored, y, model, loss, false);
                stats.at(i, 2) = forecast_cover;
                stats.at(i, 3) = forecast_width;

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

        arma::mat optimal_step_size(
            const Model &model,
            const arma::vec &step_size_grid,
            const std::string &loss = "quadratic",
            const bool &verbose = VERBOSE)
        {
            unsigned int nelem = step_size_grid.n_elem;
            arma::mat stats(nelem, 4, arma::fill::zeros);

            for (unsigned int i = 0; i < nelem; i++)
            {
                Rcpp::checkUserInterrupt();

                double step_size = step_size_grid.at(i);
                stats.at(i, 0) = step_size;

                gamma.set_size(m);
                gamma.ones();
                grad_tau.init(m, learning_rate, step_size);

                mu.set_size(m);
                mu.zeros();
                grad_mu.init(m, learning_rate, step_size);

                B.set_size(m, k);
                B.zeros();
                grad_vecB.init(m * k, learning_rate, step_size);

                d.set_size(m);
                d.ones();
                grad_d.init(m, learning_rate, step_size);

                infer(model);

                double forecast_loss = 0.;
                double forecast_cover = 0.;
                double forecast_width = 0.;
                Model::forecast_error(forecast_loss, forecast_cover, forecast_width, psi_stored, y, model, loss, false);
                stats.at(i, 1) = forecast_loss;

                // double fit_loss = 0.;
                // Model::fitted_error(fit_loss, psi_stored, y, model, loss, false);
                stats.at(i, 2) = forecast_cover;
                stats.at(i, 3) = forecast_width;

                if (verbose)
                {
                    Rcpp::Rcout << "\rForecast error: " << i + 1 << "/" << nelem;
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
            arma::mat stats(nelem, 3, arma::fill::zeros);

            for (unsigned int i = 0; i < nelem; i++)
            {
                Rcpp::checkUserInterrupt();

                unsigned int B = grid.at(i);
                stats.at(i, 0) = B;
                smc_opts["num_backward"] = B;

                infer(model);

                double err_forecast = 0.;
                double forecast_cover = 0.;
                double forecast_width = 0.;
                Model::forecast_error(err_forecast, forecast_cover, forecast_width, psi_stored, y, model, loss, false);
                stats.at(i, 1) = err_forecast;

                // double err_fit = 0.;
                // Model::fitted_error(err_fit, psi_stored, y, model, loss, false);
                stats.at(i, 2) = forecast_cover;
                stats.at(i, 3) = forecast_width;

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

        // void forecast_error(double &err, const Model &model, const std::string &loss_func = "quadratic")
        // {
        //     Model::forecast_error(err, psi_stored, y, model, loss_func);
        //     return;
        // }

        /**
         * @brief Perform forecasting on `nforecast_err` time points in time interval [tstart, ntime - kstep].
         *
         * @param model
         * @param loss_func string
         * @param kstep unsigned int
         * @param verbose bool
         * @return Rcpp::List
         */
        Rcpp::List forecast_error(
            const Model &model,
            const std::string &loss_func = "quadratic",
            const unsigned int &kstep = 1,
            const bool &verbose = VERBOSE)
        {
            const unsigned int ntime = model.dim.nT;
            unsigned int tstart = std::max(model.dim.nP, model.dim.nL);
            tstart = std::max(tstart, kstep);
            tstart += 1;
            tstart = std::max(tstart, static_cast<unsigned int>(tstart_pct * ntime));
            unsigned int tend = ntime - kstep;

            if (tstart > tend)
            {
                throw std::invalid_argument("VB::Hybrid::forecast_error: tstart should <= tend.");
            }

            arma::uvec time_indices = arma::regspace<arma::uvec>(tstart, 1, tend);
            /*
            Perform forecasting on `nforecast_err` time points in time interval [tstart, ntime - kstep].
            `time_indices` is a nforecast_err x 1 vector.
            Set `nforecast_err = ntime - kstep - tstart + 1` to perform forecasting at every time point.
            */

            arma::cube ycast = arma::zeros<arma::cube>(ntime + 1, nsample, kstep);
            arma::mat y_cov_cast(ntime + 1, kstep, arma::fill::zeros); // (nT + 1) x k
            arma::mat y_width_cast = y_cov_cast;
            arma::mat y_err_cast = y_cov_cast;

            Rcpp::NumericVector lag_param = {
                model.transfer.dlag.par1,
                model.transfer.dlag.par2};

            arma::vec yall = y;
            arma::vec psi_all = psi;
            arma::mat psi_stored_all = psi_stored;

            arma::uvec success(ntime + 1, arma::fill::zeros);

            for (unsigned int t = 0; t < ntime; t++)
            {
                if (t < tstart || t >= (ntime - kstep))
                {
                    arma::vec ysub(kstep, arma::fill::zeros);
                    unsigned int idxs = t + 1;
                    unsigned int idxe = std::min(t + kstep, ntime);
                    unsigned int nelem = idxe - idxs + 1;
                    ysub.head(nelem) = y.subvec(idxs, idxe);

                    for (unsigned int i = 0; i < nsample; i++)
                    {
                        ycast.tube(t, i) = ysub;
                    }
                }
            }

            for (unsigned int i = 0; i < time_indices.n_elem; i++)
            {
                Rcpp::checkUserInterrupt();

                unsigned int t = time_indices.at(i);

                Model submodel = model;
                submodel.dim.nT = t;
                submodel.dim.init(
                    submodel.dim.nT, submodel.dim.nL,
                    submodel.dobs.par2);

                submodel.transfer.init(
                    submodel.dim, submodel.transfer.name,
                    submodel.transfer.fgain.name,
                    submodel.transfer.dlag.name, lag_param);

                // psi.clear();
                // psi = psi_all.head(t + 1);
                y.clear();
                y = yall.head(t + 1); // (t + 1) x 1, (y[0], y[1], ..., y[t])
                psi_stored.clear();
                psi_stored = psi_stored_all.head_rows(t + 1); // (t + 1) x nsample

                arma::mat ynew(kstep, nsample, arma::fill::zeros);

                try
                {
                    infer(submodel, false);

                    ynew = Model::forecast(
                        y, psi_stored, W_stored,
                        submodel.dim, submodel.transfer,
                        submodel.flink.name,
                        submodel.dobs.par1, kstep); // k x nsample, y[t + 1], ..., y[t + k]

                    success.at(t) = 1;
                }
                catch (const std::exception &e)
                {
                    std::cerr << "\n Forecasting failed at t = " << t << std::endl;
                    success.at(t) = 0;
                }

                for (unsigned int j = 0; j < kstep; j++)
                {
                    arma::vec yest = arma::vectorise(ynew.row(j));
                    double ytrue = yall.at(t + 1 + j);

                    ycast.slice(j).row(t) = yest.t(); // 1 x nsample
                    arma::vec tmp = evaluate(yest, ytrue, loss_func);

                    y_err_cast.at(t, j) = tmp.at(0);
                    y_width_cast.at(t, j) = tmp.at(1);
                    y_cov_cast.at(t, j) = tmp.at(2);
                }

                if (verbose)
                {
                    Rcpp::Rcout << "\rForecast error: " << t + 1 << "/" << tend + 1;
                }
            } // k-step ahead forecasting with information D[t] for each t.

            if (verbose)
            {
                Rcpp::Rcout << std::endl;
            }


            arma::uvec succ_idx = arma::find(success == 1);

            arma::vec qprob = {0.025, 0.5, 0.975};
            arma::cube ycast_qt = arma::zeros<arma::cube>(ntime + 1, 3, kstep);
            std::map<std::string, AVAIL::Loss> loss_list = AVAIL::loss_list;

            arma::vec y_loss_all(kstep, arma::fill::zeros);
            arma::vec y_covered_all = y_loss_all;
            arma::vec y_width_all = y_loss_all;

            for (unsigned int j = 0; j < kstep; j++)
            {
                arma::vec ycov_tmp0 = y_cov_cast.col(j);
                arma::vec ycov_tmp = arma::vectorise(ycov_tmp0.elem(succ_idx));
                y_covered_all.at(j) = arma::mean(ycov_tmp) * 100.;

                ycov_tmp0 = y_width_cast.col(j);
                ycov_tmp = arma::vectorise(ycov_tmp0.elem(succ_idx));
                y_width_all.at(j) = arma::mean(ycov_tmp);

                arma::mat ycast_qtmp = arma::quantile(ycast.slice(j), qprob, 1); // (nT + 1) x nsample x k
                ycast_qt.slice(j) = ycast_qtmp;

                ycov_tmp0 = y_err_cast.col(j);
                ycov_tmp = arma::vectorise(ycov_tmp0.elem(succ_idx));
                y_loss_all.at(j) = arma::mean(ycov_tmp);
                if (loss_list[tolower(loss_func)] == AVAIL::L2)
                {
                    y_loss_all.at(j) = std::sqrt(y_loss_all.at(j)); // RMSE
                }


            } // switch by kstep

            Rcpp::List output;
            output["y_cast"] = Rcpp::wrap(ycast_qt);
            output["y_cast_all"] = Rcpp::wrap(ycast);
            output["y_loss_all"] = Rcpp::wrap(y_loss_all);
            output["y_covered_all"] = Rcpp::wrap(y_covered_all);
            output["y_width_all"] = Rcpp::wrap(y_width_all);

            output["tstart"] = tstart + 1;
            output["tend"] = tend + 1;
            return output;
        } // end of function

        void infer(const Model &model_in, const bool &verbose = VERBOSE)
        {
            Model model = model_in;
            W = model.derr.par1;
            mu0 = model.dobs.par1;
            rho = model.dobs.par2;
            par1 = model.transfer.dlag.par1;
            par2 = model.transfer.dlag.par2;

            // Rcpp::List smc_opts = SMC::MCS::default_settings();
            // smc_opts["num_particle"]  = N;
            // smc_opts["num_backward"] = B;
            // smc_opts["W"] = W;
            // smc_opts["use_discount"] = false;

            // SMC::MCS mcs(model, y);
            // mcs.init(smc_opts);

            SMC::TFS smc(model, y);
            smc.init(smc_opts);

            // arma::vec eta = init_eta(opts.params_selected, W, mu0, kappa, r, opts.update_static); // Checked. OK.
            // arma::vec eta_tilde = eta2tilde(eta, opts.params_selected, W.prior.name);

            for (unsigned int b = 0; b < ntotal; b++)
            {
                bool saveiter = b > nburnin && ((b - nburnin - 1) % nthin == 0);
                Rcpp::checkUserInterrupt();

                // LBA::LinearBayes lba(model, y, W, 0.95, false);
                // lba.filter();
                // lba.smoother();
                // arma::mat psi_tmp = LBA::get_psi(lba.atilde, lba.Rtilde);
                // psi = psi_tmp.col(1);

                smc.prior_W.val = W;
                try
                {
                    smc.infer(model, false);
                }
                catch (const std::exception &e)
                {
                    std::cerr << "W = " << W << ", mu0 = " << mu0 << ", rho = " << rho << std::endl;
                    throw std::runtime_error(e.what());
                }

                arma::mat psi_all;
                if (smc.smoothing)
                {
                    psi_all = smc.Theta.row_as_mat(0); // (nT + B) x N
                }
                else
                {
                    psi_all = smc.Theta_smooth.row_as_mat(0); // (nT + B) x N
                }
                // arma::mat psi_all = mcs.get_psi_smooth(); // (nT + 1) x M
                psi = arma::mean(psi_all.head_rows(model.dim.nT + 1), 1);

                arma::vec ft = psi;
                ft.at(0) = 0.;
                for (unsigned int t = 1; t < ft.n_elem; t++)
                {
                    ft.at(t) = TransFunc::func_ft(
                        t, y, ft, psi, model.dim,
                        model.transfer.dlag,
                        model.transfer.fgain.name,
                        model.transfer.name); // Checked. OK.
                }

                if (update_static)
                {
                    arma::vec dlogJoint = dlogJoint_deta(
                        y, psi, ft, eta, param_selected,
                        W_prior, par1_prior, par2_prior, rho_prior, model); // Checked. OK.

                    arma::mat SigInv = get_sigma_inv(B, d, k);
                    arma::vec dlogq = dlogq_dtheta(SigInv, nu, eta_tilde, gamma, mu);
                    arma::vec ddiff = dlogJoint - dlogq;

                    arma::vec L_mu = dYJinv_dnu(nu, gamma) * ddiff;
                    grad_mu.update_grad(L_mu);
                    mu = mu + grad_mu.change;

                    if (m > 1)
                    {
                        arma::mat dtheta_dB = dYJinv_dB(nu, gamma, xi);             // m x mk
                        arma::mat L_B = arma::reshape(dtheta_dB.t() * ddiff, m, k); // m x k
                        if (k > 1)
                        {
                            L_B.elem(B_uptri_idx).zeros();
                        }

                        arma::vec vecL_B = arma::vectorise(L_B); // mk x 1
                        grad_vecB.update_grad(vecL_B);

                        arma::mat B_change2 = arma::reshape(
                            grad_vecB.change, B.n_rows, B.n_cols); // m x k
                        if (k > 1)
                        {
                            B_change2.elem(B_uptri_idx).zeros();
                        }

                        B = B + B_change2;
                        if (k > 1)
                        {
                            B.elem(B_uptri_idx).zeros();
                        }
                    }

                    // d
                    arma::vec L_d = dYJinv_dD(nu, gamma, eps) * ddiff; // m x 1
                    grad_d.update_grad(L_d);
                    d = d + grad_d.change;

                    // tau
                    arma::vec tau = gamma2tau(gamma);
                    arma::vec L_tau = dYJinv_dtau(nu, gamma) * ddiff;
                    grad_tau.update_grad(L_tau);
                    tau = tau + grad_tau.change;
                    gamma = tau2gamma(tau);

                    rtheta(nu, eta_tilde, xi, eps, gamma, mu, B, d);
                    eta = tilde2eta(eta_tilde, param_selected, W_prior.name, par1_prior.name);

                    update_params(W, mu0, rho, par1, par2, model, param_selected, eta);

                    if (par1_prior.infer || par2_prior.infer)
                    {
                        smc.update_np(model.dim.nP);
                        dim.nL = model.dim.nL;
                        dim.nP = model.dim.nP;
                    }

                } // end update_static

                // bool saveiter = false;
                // unsigned int idx_run = 0;
                // save(saveiter, idx_run, b);
                // if (saveiter)
                // {

                // }

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

                    psi_stored.col(idx_run) = psi;

                    if (W_prior.infer)
                    {
                        W_stored.at(idx_run) = W;
                    }

                    if (mu0_prior.infer)
                    {
                        mu0_stored.at(idx_run) = mu0;
                    }

                    if (rho_prior.infer)
                    {
                        rho_stored.at(idx_run) = rho;
                    }

                    if (par1_prior.infer || par2_prior.infer)
                    {
                        par1_stored.at(idx_run) = par1;
                        par2_stored.at(idx_run) = par2;
                    }
                }

                if (verbose)
                {
                    Rcpp::Rcout << "\rProgress: " << b << "/" << ntotal - 1;
                }

            } // HVB loop

            if (verbose)
            {
                Rcpp::Rcout << std::endl;
            }

        }

        Rcpp::List get_output()
        {
            Rcpp::List output;
            output["psi_stored"] = Rcpp::wrap(psi_stored);
            arma::vec qprob = {0.025, 0.5, 0.975};
            arma::mat psi_quantile = arma::quantile(psi_stored, qprob, 1);
            output["psi"] = Rcpp::wrap(psi_quantile);

            if (W_prior.infer)
            {
                output["W"] = Rcpp::wrap(W_stored);
            }

            if (mu0_prior.infer)
            {
                output["mu0"] = Rcpp::wrap(mu0_stored);
            }

            if (rho_prior.infer)
            {
                output["rho"] = Rcpp::wrap(rho_stored);
            }

            if (par1_prior.infer)
            {
                output["par1"] = Rcpp::wrap(par1_stored);
            }

            if (par2_prior.infer)
            {
                output["par2"] = Rcpp::wrap(par2_stored);
            }

            output["inferred"] = Rcpp::wrap(param_selected);

            return output;
        }

    private:
        double learning_rate = 0.01;
        double eps_step_size = 1.e-6;
        unsigned int k = 1; // rank of unknown static parameters.

        Rcpp::List smc_opts;

        // StaticParam W, mu0, delta, kappa, r;
        HybridParams grad_mu, grad_vecB, grad_d, grad_tau;
        arma::vec mu, d, gamma, nu, eps, eta, eta_tilde; // m x 1
        arma::vec xi;                                    // k x 1
        arma::mat B;                                     // m x k
        arma::vec vecL_B;                                // mk x 1
        arma::uvec B_uptri_idx;

        arma::mat mu0_stored2;
    }; // class Hybrid
}

#endif