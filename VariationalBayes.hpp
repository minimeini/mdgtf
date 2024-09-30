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
        unsigned int N = 500; // number of SMC particles

        bool use_discount = false;
        double discount_factor = 0.95;

        bool update_static = true;
        unsigned int m = 1; // number of unknown static parameters
        std::vector<std::string> param_selected = {"W"};

        Prior W_prior, seas_prior, rho_prior, par1_prior, par2_prior;

        arma::vec W_stored;    // nsample x 1
        arma::mat seas_stored;  // period x nsample
        arma::vec rho_stored;  // nsample x 1
        arma::vec par1_stored; // nsample x 1
        arma::vec par2_stored; // nsample x 1
        arma::mat psi_stored; // (nT + 1) x nsample

        VariationalBayes(const Model &model, const Rcpp::List &vb_opts)
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


            use_discount = false;
            if (opts.containsElementNamed("use_discount"))
            {
                use_discount = Rcpp::as<bool>(opts["use_discount"]);
            }

            discount_factor = 0.95;
            if (opts.containsElementNamed("discount_factor"))
            {
                discount_factor = Rcpp::as<double>(opts["discount_factor"]);
            }

            param_selected.clear();
            m = 0;

            W_stored.set_size(nsample);
            W_stored.zeros();
            W_prior.init("invgamma", 1., 1.);
            W_prior.infer = false;
            if (opts.containsElementNamed("W"))
            {
                Rcpp::List param_opts = Rcpp::as<Rcpp::List>(opts["W"]);
                W_prior.init(param_opts);
            }
            if (W_prior.infer)
            {
                param_selected.push_back("W");
                m += 1;
            }

            seas_stored.set_size(model.seas.period, nsample);
            seas_stored.zeros();
            seas_prior.init("gaussian", 1., 10.);
            seas_prior.infer = false;
            if (opts.containsElementNamed("seas"))
            {
                Rcpp::List param_opts = Rcpp::as<Rcpp::List>(opts["seas"]);
                seas_prior.init(param_opts);
            }
            if (seas_prior.infer)
            {
                param_selected.push_back("seas");
                m += model.seas.period;
            }

            rho_stored = W_stored;
            rho_prior.init("invgamma", 1., 1.);
            rho_prior.infer = false;
            if (opts.containsElementNamed("rho"))
            {
                Rcpp::List param_opts = Rcpp::as<Rcpp::List>(opts["rho"]);
                rho_prior.init(param_opts);
            }
            if (rho_prior.infer)
            {
                param_selected.push_back("rho");
                m += 1;
            }

            par1_stored.set_size(nsample);
            par1_prior.init("gaussian", 0., 1.);
            par1_prior.infer = false;
            if (opts.containsElementNamed("par1"))
            {
                Rcpp::List param_opts = Rcpp::as<Rcpp::List>(opts["par1"]);
                par1_prior.init(param_opts);
            }
            if (par1_prior.infer)
            {
                param_selected.push_back("par1");
                m += 1;
            }

            par2_stored.set_size(nsample);
            par2_prior.init("invgamma", 1., 1.);
            par2_prior.infer = false;
            if (opts.containsElementNamed("par2"))
            {
                Rcpp::List param_opts = Rcpp::as<Rcpp::List>(opts["par2"]);
                par2_prior.init(param_opts);
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
            W_opts["infer"] = false;
            W_opts["prior_name"] = "invgamma";
            W_opts["prior_param"] = Rcpp::NumericVector::create(1., 1.);

            Rcpp::List seas_opts;
            seas_opts["infer"] = false;
            seas_opts["prior_name"] = "gaussian";
            seas_opts["prior_param"] = Rcpp::NumericVector::create(1., 10.);

            Rcpp::List rho_opts;
            rho_opts["infer"] = false;
            rho_opts["prior_param"] = Rcpp::NumericVector::create(1., 1.);
            rho_opts["prior_name"] = "invgamma";

            Rcpp::List par1_opts;
            par1_opts["infer"] = false;
            par1_opts["prior_param"] = Rcpp::NumericVector::create(0., 1.);
            par1_opts["prior_name"] = "gaussian";

            Rcpp::List par2_opts;
            par2_opts["infer"] = false;
            par2_opts["prior_param"] = Rcpp::NumericVector::create(1., 1.);
            par2_opts["prior_name"] = "invgamma";

            Rcpp::List opts;
            opts["nsample"] = 1000;
            opts["nthin"] = 1;
            opts["nburnin"] = 1000;
            opts["num_particle"] = 100;
            opts["use_discount"] = false;
            opts["discount_factor"] = 0.95;

            opts["W"] = W_opts;
            opts["seas"] = seas_opts;
            opts["rho"] = rho_opts;
            opts["par1"] = par1_opts;
            opts["par2"] = par2_opts;

            return opts;
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
            rho = arma::sqrt(curEdelta2 + eps_step_size) / arma::sqrt(curEg2 + eps_step_size);
            par_change = rho % dYJinv_dVecPar;

            arma::vec oldEdelta2 = curEdelta2;
            curEdelta2 = (1. - learning_rate) * oldEdelta2 + learning_rate * arma::pow(par_change, 2.);

            #ifdef DGTF_DO_BOUND_CHECK
                bound_check<arma::vec>(curEg2, "curEg2");
                bound_check<arma::vec>(par_change, "update_grad: par_change");
                bound_check<arma::vec>(curEdelta2, "update_grad: curEdelta2");
            #endif
            
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
        Hybrid(const Model &model_in, const Rcpp::List &hvb_opts) : VariationalBayes(model_in, hvb_opts)
        {
            Rcpp::List opts = hvb_opts;

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

            eta = init_eta(param_selected, model_in, update_static); // Checked. OK.
            eta_tilde = eta2tilde(eta, param_selected, W_prior.name, par1_prior.name, model_in.seas.period, model_in.seas.in_state);

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
        }


        static Rcpp::List default_settings()
        {
            Rcpp::List opts = VariationalBayes::default_settings();
            opts["learning_rate"] = 0.01;
            opts["eps_step_size"] = 1.e-6;
            opts["k"] = 1;
            return opts;
        }

        static arma::vec eta2tilde( // Checked. OK.
            const arma::vec &eta,   // m x 1
            const std::vector<std::string> &param_selected,
            const std::string &W_prior = "invgamma",
            const std::string &lag_par1_prior = "gaussian",
            const unsigned int &seasonal_period = 1,
            const bool &season_in_state = false)
        {
            std::map<std::string, AVAIL::Dist> dist_list = AVAIL::dist_list;
            std::map<std::string, AVAIL::Param> static_param_list = AVAIL::static_param_list;

            arma::vec eta_tilde = eta;
            unsigned int idx = 0;
            for (unsigned int i = 0; i < param_selected.size(); i++)
            {
                double val = eta.at(idx);
                switch (static_param_list[tolower(param_selected[i])])
                {
                case AVAIL::Param::W:
                {
                    switch (dist_list[tolower(W_prior)])
                    {
                    case AVAIL::Dist::invgamma:
                    {
                        eta_tilde.at(idx) = -std::log(std::abs(val) + EPS);
                        break;
                    }
                    case AVAIL::Dist::gamma:
                    {
                        eta_tilde.at(idx) = std::log(std::abs(val) + EPS);
                        break;
                    }
                    default:
                    {
                        throw std::invalid_argument("VB::Hybrid::eta2tilde: undefined prior " + W_prior + " for W.");
                        break;
                    }
                    } // switch W prior.

                    idx += 1;
                    break;
                } // W
                case AVAIL::Param::seas:
                {
                    // bound_check(val, "VB::Hybrid::eta2tilde: mu0", false, true);
                    if (!season_in_state)
                    {
                        arma::vec seas = eta.subvec(idx, idx + seasonal_period - 1);

                        #ifdef DGTF_DO_BOUND_CHECK
                            bound_check<arma::vec>(seas, "VB::Hybrid::eta2tilde: seas", false, true);
                        #endif

                        eta_tilde.subvec(idx, idx + seasonal_period - 1) = arma::log(seas + EPS);
                        idx += seasonal_period;
                    }
                    break;
                } // mu0
                case AVAIL::Param::rho:
                {
                    eta_tilde.at(idx) = std::log(std::abs(val) + EPS);
                    idx += 1;
                    break;
                } // rho
                case AVAIL::Param::lag_par1:
                {
                    switch (dist_list[tolower(lag_par1_prior)])
                    {
                    case AVAIL::Dist::gaussian:
                    {
                        eta_tilde.at(idx) = val;
                        break;
                    }
                    case AVAIL::Dist::invgamma:
                    {
                        eta_tilde.at(idx) = std::log(std::abs(val) + EPS);
                        break;
                    }
                    case AVAIL::Dist::gamma:
                    {
                        eta_tilde.at(idx) = std::log(std::abs(val) + EPS);
                        break;
                    }
                    case AVAIL::Dist::beta:
                    {
                        eta_tilde.at(idx) = std::log(std::abs(val) + EPS) - std::log(std::abs(1. - val) + EPS);
                        break;
                    }
                    default:
                    {
                        throw std::invalid_argument("VB::Hybrid::eta2tilde: undefined prior " + lag_par1_prior + " for first parameter of lag distribution.");
                        break;
                    }
                    } // switch prior type for lag_par1.

                    idx += 1;
                    break;
                } // lag_par1
                case AVAIL::Param::lag_par2:
                {
                    eta_tilde.at(i) = std::log(std::abs(val) + EPS);
                    idx += 1;
                    break;
                } // lag_par2
                default:
                {
                    throw std::invalid_argument("VB::Hybrid::eta2tilde: undefined static parameter " + param_selected[i]);
                    break;
                }
                } // switch param
            }

            #ifdef DGTF_DO_BOUND_CHECK
                bound_check<arma::vec>(eta_tilde, "VB::Hybrid::eta2tilde: eta_tilde");
            #endif
            return eta_tilde;
        }

        static arma::vec tilde2eta(     // Checked. OK.
            const arma::vec &eta_tilde, // m x 1
            const std::vector<std::string> &param_selected,
            const std::string &W_prior = "invgamma",
            const std::string &lag_par1_prior = "gaussian",
            const std::string &lag_dist = "lognorm",
            const unsigned int &seasonal_period = 1,
            const bool &season_in_state = false)
        {
            std::map<std::string, AVAIL::Dist> dist_list = AVAIL::dist_list;
            std::map<std::string, AVAIL::Dist> lag_list = LagDist::lag_list;
            std::map<std::string, AVAIL::Param> static_param_list = AVAIL::static_param_list;

            arma::vec eta = eta_tilde;
            unsigned int idx = 0;
            for (unsigned int i = 0; i < param_selected.size(); i++)
            {
                double val = eta_tilde.at(idx);
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
                        eta.at(idx) = std::exp(val);
                        break;
                    }
                    case AVAIL::Dist::gamma:
                    {
                        val = std::min(val, UPBND);
                        eta.at(idx) = std::exp(val);
                        break;
                    }
                    default:
                    {
                        break;
                    }
                    } // switch W prior.

                    idx += 1;
                    break;
                } // W
                case AVAIL::Param::seas:
                {
                    if (!season_in_state)
                    {
                        arma::vec log_seas = eta_tilde.subvec(idx, idx + seasonal_period - 1);
                        log_seas.clamp(log_seas.min(), UPBND);
                        eta.subvec(idx, idx + seasonal_period - 1) = arma::exp(log_seas);
                        idx += seasonal_period;
                    }
                    break;
                } // mu0
                case AVAIL::Param::rho:
                {
                    val = std::min(val, UPBND);
                    eta.at(idx) = std::exp(val);

                    idx += 1;
                    break;
                } // rho
                case AVAIL::Param::lag_par1:
                {
                    switch (dist_list[tolower(lag_par1_prior)])
                    {
                    case AVAIL::Dist::gaussian:
                    {
                        eta.at(idx) = val;
                        break;
                    }
                    case AVAIL::Dist::invgamma:
                    {
                        val = std::min(val, UPBND);
                        eta.at(idx) = std::exp(val);
                        break;
                    }
                    case AVAIL::Dist::gamma:
                    {
                        val = std::min(val, UPBND);
                        eta.at(idx) = std::exp(val);
                        break;
                    }
                    case AVAIL::Dist::beta:
                    {
                        val = std::min(val, UPBND);
                        double tmp = std::exp(val);
                        eta.at(idx) = tmp;
                        eta.at(idx) /= (1. + tmp);
                        break;
                    }
                    default:
                    {
                        throw std::invalid_argument("VB::Hybrid::tilde2eta: undefined prior " + lag_par1_prior + " for first parameter of lag distribution.");
                        break;
                    }
                    } // switch prior type for lag_par1.

                    idx += 1;
                    break;
                } // lag_par1
                case AVAIL::Param::lag_par2:
                {
                    val = std::min(val, UPBND);
                    eta.at(idx) = std::exp(val);

                    if (lag_list[lag_dist] == AVAIL::Dist::nbinomp)
                    {
                        eta.at(idx) = std::max(std::ceil(eta.at(idx)), 1.);
                    }
                    else
                    idx += 1;
                    break;
                } // lag_par2
                default:
                {
                    throw std::invalid_argument("VB::Hybrid::tilde2eta: undefined static parameter " + param_selected[i]);
                    break;
                }
                } // switch param

                
            }

            #ifdef DGTF_DO_BOUND_CHECK
                bound_check<arma::vec>(eta, "tilde2eta: eta");
            #endif
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
            #ifdef DGTF_DO_BOUND_CHECK
                bound_check(rdw, "dlogJoint_dWtilde: rdw");
            #endif

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

            #ifdef DGTF_DO_BOUND_CHECK
                bound_check(deriv, "dlogJoint_dWtilde: deriv");
            #endif
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

            #ifdef DGTF_DO_BOUND_CHECK
                bound_check(logp, "logprior_Wtilde: logp");
            #endif

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
        static arma::vec dloglike_dlogseas( // Checked. OK.
            const arma::vec &y,           // (nT+1) x 1
            const arma::vec &lambda,          // (n+1) x 1, (f[0],f[1],...,f[nT])
            const ObsDist &dobs,
            const arma::vec &seas,
            const arma::mat &Xseas)
        { // 0 - negative binomial; 1 - poisson

            arma::vec deriv(seas.n_elem, arma::fill::zeros);
            for (unsigned int t = 1; t < y.n_elem; t++)
            {
                double dy_dlambda = ObsDist::dloglike_dlambda(y.at(t), lambda.at(t), dobs);
                arma::vec dlambda_dlogseas = Xseas.col(t) % seas;
                deriv = deriv + dy_dlambda * dlambda_dlogseas;
            }
            return deriv;
        }


        static arma::vec logprior_logseas(
            const arma::vec &logseas, // evolution variance conditional on V
            const double &sig2_mu0 = 10.)
        {
            /*
            log(mu0) ~ N(0,sig2_mu0)
            */
            arma::vec logp = - logseas / sig2_mu0;
            #ifdef DGTF_DO_BOUND_CHECK
                bound_check<arma::vec>(logp, "logprior_logseas: logp");
            #endif
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
            const arma::mat &Theta,        // nP x (nT + 1)
            const arma::vec &lambda,         // (nT + 1) x 1
            const arma::vec &dllk_dpar, // 2 x 1
            const arma::vec &eta,        // m x 1
            const std::vector<std::string> &param_selected,
            const Prior &W_prior,
            const Prior &lag_par1_prior,
            const Prior &lag_par2_prior,
            const Prior &rho_prior,
            const Prior &seas_prior,
            const Model &model)
        {
            std::map<std::string, AVAIL::Param> static_param_list = AVAIL::static_param_list;
            unsigned int nelem = param_selected.size();
            for (unsigned int i = 0; i < param_selected.size(); i++)
            {
                if (static_param_list[tolower(param_selected[i])] == AVAIL::Param::seas)
                {
                    nelem += model.seas.period - 1;
                }
            }
            arma::vec deriv(nelem, arma::fill::zeros);

            unsigned int idx = 0;
            for (unsigned int i = 0; i < param_selected.size(); i++)
            {
                double val = eta.at(idx);

                switch (static_param_list[tolower(param_selected[i])])
                {
                case AVAIL::Param::W:
                {
                    arma::vec psi = arma::vectorise(Theta.row(0));
                    deriv.at(idx) = dlogJoint_dWtilde(psi, 1., val, W_prior);
                    idx += 1;
                    break;
                }
                case AVAIL::Param::seas:
                {
                    arma::vec seas = arma::abs(eta.subvec(idx, idx + model.seas.period - 1));
                    arma::vec logseas = arma::log(seas + EPS);

                    arma::vec dloglik = dloglike_dlogseas(y, lambda, model.dobs, seas, model.seas.X);
                    arma::vec dlogprior = logprior_logseas(logseas, seas_prior.par2);

                    deriv.subvec(idx, idx + model.seas.period - 1) = dloglik + dlogprior;
                    idx += model.seas.period;
                    break;
                }
                case AVAIL::Param::rho:
                {
                    deriv.at(idx) = Model::dlogp_dpar2_obs(model, y, lambda, true);
                    deriv.at(idx) += Prior::dlogprior_dpar(model.dobs.par2, rho_prior, true);
                    idx += 1;
                    break;
                }
                case AVAIL::Param::lag_par1:
                {
                    deriv.at(idx) = dllk_dpar.at(0);
                    deriv.at(idx) += Prior::dlogprior_dpar(model.dlag.par1, lag_par1_prior, true);
                    idx += 1;
                    break;
                }
                case AVAIL::Param::lag_par2:
                {
                    deriv.at(idx) = dllk_dpar.at(1);
                    deriv.at(idx) += Prior::dlogprior_dpar(model.dlag.par2, lag_par2_prior, true);
                    idx += 1;
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
            const Model &model,
            const bool &update_static)
        {
            std::map<std::string, AVAIL::Param> static_param_list = AVAIL::static_param_list;
            unsigned int nelem = params_selected.size();
            for (unsigned int i = 0; i < params_selected.size(); i++)
            {
                if (static_param_list[tolower(params_selected[i])] == AVAIL::Param::seas)
                {
                    nelem += model.seas.period - 1;
                }
            }
            arma::vec eta(nelem, arma::fill::zeros);

            if (!update_static)
            {
                return eta;
            }

            unsigned int idx = 0;
            for (unsigned int i = 0; i < params_selected.size(); i++)
            {
                switch (static_param_list[params_selected[i]])
                {
                case AVAIL::Param::W:
                {
                    eta.at(idx) = model.derr.par1;
                    idx += 1;
                    break;
                }
                case AVAIL::Param::seas:
                {
                    eta.subvec(idx, idx + model.seas.period - 1) = model.seas.val;
                    idx += model.seas.period;
                    break;
                }
                case AVAIL::Param::rho:
                {
                    eta.at(i) = model.dobs.par2;
                    idx += 1;
                    break;
                }
                case AVAIL::Param::lag_par1:
                {
                    eta.at(i) = model.dlag.par1;
                    idx += 1;
                    break;
                }
                case AVAIL::Param::lag_par2:
                {
                    eta.at(i) = model.dlag.par2;
                    idx += 1;
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
            Model &model,
            const std::vector<std::string> &params_selected,
            const arma::vec &eta)
        {
            std::map<std::string, AVAIL::Param> static_param_list = AVAIL::static_param_list;
            bool update_dlag = false;

            unsigned int idx = 0;
            for (unsigned int i = 0; i < params_selected.size(); i++)
            {
                double val = eta.at(idx);
                switch (static_param_list[params_selected[i]])
                {
                case AVAIL::Param::W: // W is selected
                {
                    model.derr.par1 = val;
                    idx += 1;
                    break;
                }
                case AVAIL::Param::seas: // mu0 is selected
                {
                    model.seas.val = eta.subvec(idx, idx + model.seas.period - 1);
                    idx += model.seas.period;
                    break;
                }
                case AVAIL::Param::rho: // rho is selected
                {
                    model.dobs.par2 = val;
                    idx += 1;
                    break;
                }
                case AVAIL::Param::lag_par1: // par 1 is selected
                {
                    update_dlag = true;
                    model.dlag.par1 = val;
                    idx += 1;
                    break;
                }
                case AVAIL::Param::lag_par2: // par 2 is selected
                {
                    update_dlag = true;
                    model.dlag.par2 = val;
                    idx += 1;
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
                model.dlag.nL = LagDist::get_nlag(model.dlag);
                model.dlag.Fphi = LagDist::get_Fphi(model.dlag);
                model.nP = Model::get_nP(model.dlag, model.seas.period, model.seas.in_state);
            }

            return;
        }

       
        void infer(
            Model &model, 
            const arma::vec &y,
            const bool &verbose = VERBOSE)
        {
            std::map<std::string, AVAIL::Algo> algo_list = AVAIL::algo_list;
            const unsigned int nT = y.n_elem - 1;
            psi_stored.set_size(y.n_elem, nsample);
            psi_stored.zeros();

            for (unsigned int b = 0; b < ntotal; b++)
            {
                bool saveiter = b > nburnin && ((b - nburnin - 1) % nthin == 0);
                Rcpp::checkUserInterrupt();

                // You MUST set initial_resample_all = true and final_resample_by_weights = false to make this algorithm work.
                arma::cube Theta_tmp = arma::randu<arma::cube>(model.nP, N, y.n_elem);
                double log_cond_marg = SMC::SequentialMonteCarlo::auxiliary_filter0(
                    Theta_tmp, model, y, N, true, false, use_discount, discount_factor);
                arma::mat Theta = arma::mean(Theta_tmp, 1); // nP x (nT + 1)

                arma::mat dFphi_grad;
                arma::vec dloglik_dlag(2, arma::fill::zeros);
                if (par1_prior.infer || par2_prior.infer)
                {
                    dFphi_grad = LagDist::get_Fphi_grad(
                        model.dlag.nL, model.dlag.name, 
                        model.dlag.par1, model.dlag.par2);
                }

                arma::vec lambda(y.n_elem, arma::fill::zeros);
                for (unsigned int t = 1; t < lambda.n_elem; t++)
                {
                    unsigned int nelem = std::min(t, model.dlag.nL); // min(t,nL)
                    arma::vec yold(model.dlag.nL, arma::fill::zeros);
                    if (nelem > 1)
                    {
                        yold.tail(nelem) = y.subvec(t - nelem, t - 1);
                    }
                    else if (t > 0) // nelem = 1 at t = 1
                    {
                        yold.at(model.dlag.nL - 1) = y.at(t - 1);
                    }
                    yold = arma::reverse(yold); // y[t-1], ..., y[t-min(t,nL)]

                    arma::vec theta = arma::vectorise(Theta.submat(0, t, model.dlag.nL - 1, t));
                    arma::vec htheta = GainFunc::psi2hpsi<arma::vec>(theta, model.fgain);

                    double ft = arma::accu(model.dlag.Fphi % yold % htheta);
                    if ((model.seas.period > 0) && (!model.seas.in_state))
                    {
                        ft += arma::as_scalar(model.seas.X.col(t).t() * model.seas.val);
                    }
                    lambda.at(t) = LinkFunc::ft2mu(ft, model.flink);

                    if (par1_prior.infer || par2_prior.infer)
                    {
                        double dll_deta = Model::dloglik_deta(
                            ft, y.at(t), model.dobs.par2, model.dobs.name, model.flink);
                        double deta_dpar1 = arma::accu(dFphi_grad.col(0) % yold % htheta);
                        double deta_dpar2 = arma::accu(dFphi_grad.col(1) % yold % htheta);

                        dloglik_dlag.at(0) += dll_deta * deta_dpar1;
                        dloglik_dlag.at(1) += dll_deta * deta_dpar2;
                    }
                }


                if (update_static)
                {
                    arma::vec dlogJoint = dlogJoint_deta(
                        y, Theta, lambda, dloglik_dlag, eta, param_selected,
                        W_prior, par1_prior, par2_prior, rho_prior, seas_prior, model); // Checked. OK.


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
                    eta = tilde2eta(
                        eta_tilde, param_selected, 
                        W_prior.name, par1_prior.name, model.dlag.name, 
                        model.seas.period, model.seas.in_state);
                    update_params(model, param_selected, eta);
                } // end update_static


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

                    psi_stored.col(idx_run) = arma::vectorise(Theta.row(0));

                    if (W_prior.infer)
                    {
                        W_stored.at(idx_run) = model.derr.par1;
                        
                    }

                    if (seas_prior.infer)
                    {
                        seas_stored.col(idx_run) = model.seas.val;
                    }

                    if (rho_prior.infer)
                    {
                        rho_stored.at(idx_run) = model.dobs.par2;
                    }

                    if (par1_prior.infer || par2_prior.infer)
                    {
                        par1_stored.at(idx_run) = model.dlag.par1;
                        par2_stored.at(idx_run) = model.dlag.par2;
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
                output["W"] = Rcpp::wrap(W_stored.t());
            }

            if (seas_prior.infer)
            {
                output["seas"] = Rcpp::wrap(seas_stored);
            }

            if (rho_prior.infer)
            {
                output["rho"] = Rcpp::wrap(rho_stored.t());
            }

            if (par1_prior.infer)
            {
                output["par1"] = Rcpp::wrap(par1_stored.t());
            }

            if (par2_prior.infer)
            {
                output["par2"] = Rcpp::wrap(par2_stored.t());
            }

            output["inferred"] = Rcpp::wrap(param_selected);

            return output;
        }

    private:
        double learning_rate = 0.01;
        double eps_step_size = 1.e-6;
        unsigned int k = 1; // rank of unknown static parameters.

        // StaticParam W, mu0, delta, kappa, r;
        HybridParams grad_mu, grad_vecB, grad_d, grad_tau;
        arma::vec mu, d, gamma, nu, eps, eta, eta_tilde; // m x 1
        arma::vec xi;                                    // k x 1
        arma::mat B;                                     // m x k
        arma::vec vecL_B;                                // mk x 1
        arma::uvec B_uptri_idx;
    }; // class Hybrid
}

#endif