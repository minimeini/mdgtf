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
        unsigned int B = 1;

        bool update_static = true;
        unsigned int m = 1; // number of unknown static parameters
        std::vector<std::string> param_selected = {"W"};

        Dim dim;

        arma::vec y;

        arma::vec psi; // (nT + 1) x 1
        arma::mat psi_stored; // (nT + 1) x nsample

        bool infer_W = true;
        bool infer_mu0 = false;
        bool infer_delta = false;
        bool infer_kappa = false;
        bool infer_r = false;

        double W = 0.01;
        double mu0 = 0.;
        double delta = 30;
        double kappa = 0.4;
        double r = 6;

        Dist W_prior, mu0_prior, delta_prior, kappa_prior, r_prior;

        arma::vec W_stored; // nsample x 1
        arma::vec mu0_stored; // nsample x 1
        arma::vec delta_stored; // nsample x 1
        arma::vec kappa_stored; // nsample x 1
        arma::vec r_stored; // nsample x 1

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

        void init(const Rcpp::List & vb_opts)
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

            B = 1;
            if (opts.containsElementNamed("num_backward"))
            {
                B = Rcpp::as<unsigned int>(opts["num_backward"]);
            }

            psi_stored.set_size(psi.n_elem, nsample);
            psi_stored.zeros();

            param_selected.clear();
            m = 0;

            W_stored.set_size(nsample);
            W_stored.zeros();
            infer_W = true;
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
            infer_mu0 = false;
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

            delta_stored = W_stored;
            infer_delta = false;
            delta_prior.init("gaussian", 0., 10.);
            delta = 30.;
            if (opts.containsElementNamed("delta"))
            {
                Rcpp::List param_opts = Rcpp::as<Rcpp::List>(opts["delta"]);
                init_param(infer_delta, delta, delta_prior, param_opts);
            }
            if (infer_delta)
            {
                param_selected.push_back("delta");
                m += 1;
            }

            kappa_stored = W_stored;
            infer_kappa = false;
            kappa_prior.init("beta", 1., 1.);
            kappa = 0.4;
            if (opts.containsElementNamed("kappa"))
            {
                Rcpp::List param_opts = Rcpp::as<Rcpp::List>(opts["kappa"]);
                init_param(infer_kappa, kappa, kappa_prior, param_opts);
            }
            if (infer_kappa)
            {
                param_selected.push_back("kappa");
                m += 1;
            }

            r_stored = W_stored;
            infer_r = false;
            r_prior.init("nbinom", 1., 1.);
            r = 6;
            if (opts.containsElementNamed("r"))
            {
                Rcpp::List param_opts = Rcpp::as<Rcpp::List>(opts["r"]);
                init_param(infer_r, r, r_prior, param_opts);
            }
            if (infer_r)
            {
                param_selected.push_back("r");
                m += 1;
            }

            update_static = false;
            if (m > 0) { update_static = true; }
        }

        static void init_param(bool &infer, double &init, Dist &prior, const Rcpp::List &param_opts)
        {
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

            Rcpp::NumericVector prior_param = {0.01, 0.01};
            if (param_opts.containsElementNamed("prior_param"))
            {
                prior_param = Rcpp::as<Rcpp::NumericVector>(param_opts["prior_param"]);
            }

            prior.init(prior_name, prior_param[0], prior_param[1]);

            return;
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

            Rcpp::List delta_opts;
            delta_opts["infer"] = false;
            delta_opts["init"] = 0.;

            Rcpp::List kappa_opts;
            kappa_opts["infer"] = false;
            kappa_opts["init"] = 0.4;

            Rcpp::List r_opts;
            r_opts["infer"] = false;
            r_opts["init"] = 6.;

            Rcpp::List opts;
            opts["nsample"] = 1000;
            opts["nthin"] = 1;
            opts["nburnin"] = 1000;

            opts["num_particle"] = 500;
            opts["num_backward"] = 1;

            opts["W"] = W_opts;
            opts["mu0"] = mu0_opts;
            opts["delta"] = delta_opts;
            opts["kappa"] = kappa_opts;
            opts["r"] = r_opts;

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
            try
            {
                bound_check<arma::vec>(par_change, "update_grad: par_change");
            }
            catch(const std::exception& e)
            {
                std::cout << e.what() << std::endl;
                dYJinv_dVecPar.t().print("\ndYJinv_dVecPar");
                rho.t().print("\n rho");
                oldEg2.t().print("\n oldEg2");
                std::cout << "\n learn rate = " << learning_rate << " eps = " << eps_step_size << std::endl;
            }
            
            

            arma::vec oldEdelta2 = curEdelta2;
            curEdelta2 = (1. - learning_rate) * oldEdelta2 + learning_rate * arma::pow(par_change, 2.);

            bound_check<arma::vec>(curEg2, "update_grad: curEg2");
            bound_check<arma::vec>(curEdelta2, "update_grad: curEdelta2");
            return;
        }

        const arma::vec &change;

    private:
        arma::vec curEg2; // m x 1
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

        Hybrid(const Model &model, const arma::vec &y_in) : VariationalBayes(model, y_in)
        {
            dim = model.dim;
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


            gamma.set_size(m);
            gamma.ones();
            grad_tau.init(m, learning_rate, eps_step_size);

            mu.set_size(m);
            mu.zeros();
            grad_mu.init(m, learning_rate, eps_step_size);

            B.set_size(m, k);
            B.zeros();
            grad_vecB.init(m * k, learning_rate, eps_step_size);

            d.set_size(m);
            d.ones();
            grad_d.init(m, learning_rate, eps_step_size);


            eta = init_eta(param_selected, W, mu0, kappa, r, update_static); // Checked. OK.
            eta_tilde = eta2tilde(eta, param_selected, W_prior.name);


            nu = tYJ(eta_tilde, gamma);

            xi.set_size(k);
            xi.zeros();
            eps = xi;

            if (m > 1)
            {
                B_uptri_idx = arma::trimatu_ind(arma::size(B), 1);
            }
            else
            {
                B_uptri_idx = {0};
            }


            if (opts.containsElementNamed("mcs"))
            {
                mcs_opts = Rcpp::as<Rcpp::List>(opts["mcs"]);
            }
            else
            {
                mcs_opts = SMC::MCS::default_settings();
            }

            mcs_opts["use_discount"] = false;
            mcs_opts["W"] = W;
        }


        static Rcpp::List default_settings()
        {
            Rcpp::List opts = VariationalBayes::default_settings();
            opts["learning_rate"] = 0.01;
            opts["eps_step_size"] = 1.e-6;
            opts["k"] = 1;

            Rcpp::List mcs_tmp = SMC::MCS::default_settings();
            mcs_tmp["use_discount"] = false;
            opts["mcs"] = mcs_tmp;
            

            return opts;
        }


        static arma::vec eta2tilde( // Checked. OK.
            const arma::vec &eta,         // m x 1
            const std::vector<std::string> &param_selected,
            const std::string &W_prior = "invgamma")
        {
            std::map <std::string, AVAIL::Dist> W_prior_list = AVAIL::W_prior_list;
            std::map <std::string, AVAIL::Param> static_param_list = AVAIL::static_param_list;

            arma::vec eta_tilde = eta;
            for (unsigned int i = 0; i < param_selected.size(); i++)
            {
                double val = eta.at(i);
                switch (static_param_list[tolower(param_selected[i])])
                {
                case AVAIL::Param::W:
                {
                    switch (W_prior_list[tolower(W_prior)])
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
                        break;
                    }
                    } // switch W prior.
                    break;
                }
                case AVAIL::Param::mu0:
                {
                    bound_check(val, "VB::Hybrid::eta2tilde: mu0", false, true);
                    eta_tilde.at(i) = std::log(std::abs(val) + EPS);
                    break;
                }
                case AVAIL::Param::kappa:
                {
                    if (val > 1. || val < 0.)
                    {
                        throw std::invalid_argument("VB::Hybrid::eta2tilde: kappa should be within (0, 1)");
                    }
                    eta_tilde.at(i) = std::log(std::abs(val) + EPS) - std::log(std::abs(1. - val) + EPS);
                    break;
                }
                default:
                {
                    break;
                }
                } // switch param

                bound_check(eta_tilde.at(i), "eta2tilde: eta_tilde");
            }

            return eta_tilde;
        }

        static arma::vec tilde2eta( // Checked. OK.
            const arma::vec &eta_tilde, // m x 1
            const std::vector<std::string> &param_selected,
            const std::string &W_prior = "invgamma")
        {
            std::map<std::string, AVAIL::Dist> W_prior_list = AVAIL::W_prior_list;
            std::map<std::string, AVAIL::Param> static_param_list = AVAIL::static_param_list;

            arma::vec eta = eta_tilde;
            for (unsigned int i = 0; i < param_selected.size(); i++)
            {
                double val = eta_tilde.at(i);
                switch (static_param_list[tolower(param_selected[i])])
                {
                case AVAIL::Param::W:
                {
                    switch (W_prior_list[tolower(W_prior)])
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
                    bound_check(eta.at(i), "VB::Hybrid::tilde2eta: W", true, true);
                    break;
                }
                case AVAIL::Param::mu0:
                {
                    val = std::min(val, UPBND);
                    eta.at(i) = std::exp(val);
                    bound_check(eta.at(i), "VB::Hybrid::tilde2eta: mu0", false, true);
                    break;
                }
                case AVAIL::Param::kappa:
                {
                    val = std::min(val, UPBND);
                    double tmp = std::exp(val);
                    eta.at(i) = tmp;
                    eta.at(i) /= (1. + tmp);
                    if (eta.at(i) > 1. || eta.at(i) < 0.)
                    {
                        throw std::invalid_argument("VB::Hybrid::tilde2eta: kappa should be within (0, 1)");
                    }
                    break;
                }
                default:
                {
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
            const arma::vec &psi, // (n+1) x 1, (psi[0],psi[1],...,psi[n])
            const double G,       // evolution transition matrix
            const double W,       // evolution variance conditional on V
            const Dist &W_prior)
        { // 0 - Gamma(aw=shape,bw=rate); 1 - Half-Cauchy(aw=location=0, bw=scale); 2 - Inverse-Gamma(aw=shape,bw=rate)
            std::map<std::string, AVAIL::Dist> W_prior_list = AVAIL::W_prior_list;
            
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
            switch (W_prior_list[W_prior.name])
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
            const double &Wtilde, // evolution variance conditional on V
            const Dist &W_prior)
        {
            std::map<std::string, AVAIL::Dist> W_prior_list = AVAIL::W_prior_list;

            double InvGamma_cnst = W_prior.par1;
            InvGamma_cnst *= std::log(W_prior.par2);
            InvGamma_cnst -= std::lgamma(W_prior.par1);

            double logp = -16.;
            switch (W_prior_list[W_prior.name])
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
            const arma::vec &y,  // (nT+1) x 1
            const arma::vec &ft, // (n+1) x 1, (f[0],f[1],...,f[nT])
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
         * dlogJoint_dpar1: partial derivative w.r.t rho or mu.
         *
         * @param lag_par: either (rho, L) or (mu, sg2)
         */
        static double dlogJoint_dpar1( // Checked. OK.
            const arma::vec &ypad, // (n+1) x 1
            const arma::mat &R,    // (n+1) x 2, (psi,theta)
            const LagDist &dlag,
            const ObsDist &dobs,
            const std::string &gain_func)
        {

            double aprior = 1.;
            double bprior = 1.;
            double par1 = dlag.par1;

            const unsigned int n = ypad.n_elem - 1;
            unsigned int L = (unsigned int)dlag.par2;
            double L_ = dlag.par2;

            arma::vec hpsi = GainFunc::psi2hpsi<arma::vec>(R.col(0), gain_func);
            arma::vec lambda = dlag.par1 + R.col(1);    // (n+1) x 1

            double c1 = 0.;
            double c2 = 0.;

            for (unsigned int t = 1; t <= n; t++)
            {
                double c10 = ypad.at(t) / lambda.at(t) - (ypad.at(t) + dobs.par2) / (lambda.at(t) + dobs.par2);
                c1 += c10 * hpsi.at(t - 1) * ypad.at(t - 1);

                unsigned int r = std::min(t, L);
                double c20 = 0.;
                double c21 = -par1;
                for (unsigned int k = 1; k <= r; k++)
                {
                    c20 += static_cast<double>(k) * nbinom::binom(L, k) * c21 * R.at(t - k, 1);
                    c21 *= -par1;
                }
                c2 += c10 * c20;
            }

            double deriv = -L_ * std::pow(1. - par1, L_) * par1 * c1 - (1. - par1) * c2;
            deriv += aprior - (bprior + aprior) * par1;
            bound_check(deriv, "dlogJoint_drho: deriv");
            return deriv;
        }

        static double dlogJoint_dkappa2( // Checked. OK.
            const arma::vec &y, // (nT + 1) x 1
            const arma::mat &R, // (n+1) x 2, (psi,theta)
            const ObsDist &dobs,
            const LagDist &dlag,
            const std::string &gain_func,
            const arma::vec &rcomb, // n x 1
            const double akappa = 1.,
            const double bkappa = 1.)
        {
            double kappa = dlag.par1;
            double r = dlag.par2;

            arma::vec hpsi = GainFunc::psi2hpsi<arma::vec>(R.col(0), gain_func);
            arma::vec hy = hpsi % y;                     // (n+1) x 1, (h(psi[0])*y[0], h(psi[1])*y[1], ...,h(psi[n])*y[n])
            arma::vec lambda = dobs.par1 + R.col(1);              // (n+1) x 1

            double c10 = 0.; // d(Loglike) / d(theta[t])
            double c20 = 0.; // first part of d(theta[t]) / d(logit(rho)), l=0,...,t-1
            double c30 = 0.; // second part of d(theta[t]) / d(logit(rho)), l=1,...,t-1
            double c2 = 0.;  // first part of d(Loglike) / d(logit(rho))
            double c3 = 0.;  // second part of d(Loglike) / d(logit(rho))
            double crho = 1.;

            for (unsigned int t = 1; t < y.n_elem; t++)
            {
                crho = 1.;
                c10 = y.at(t) / lambda.at(t) - (y.at(t) + dobs.par2) / (lambda.at(t) + dobs.par2); // d(Loglike) / d(theta[t])
                c20 = rcomb.at(0) * crho * hy.at(t - 1);
                c30 = 0.;

                for (unsigned int l = 1; l < t; l++)
                {
                    crho *= kappa;
                    c20 += rcomb.at(l) * crho * hy.at(t - 1 - l);
                    c30 += static_cast<double>(l) * rcomb.at(l) * crho * hy.at(t - 1 - l);
                }

                c2 += c10 * c20;
                c3 += c10 * c30;
            }

            double deriv = - r * std::pow(1. - kappa, r) * kappa * c2 + std::pow(1. - kappa, r + 1.) * c3;
            deriv += akappa - (akappa + bkappa) * kappa;
            bound_check(deriv, "dlogJoint_drho2: deriv");
            return deriv;
        }

        static double logprior_logitkappa(
            double logitkappa,
            double akappa = 1.,
            double bkappa = 1.)
        {

            double logp = std::lgamma(akappa + bkappa) - std::lgamma(akappa) - std::lgamma(bkappa) + akappa * logitkappa - (akappa + bkappa) * std::log(1. + std::exp(logitkappa));
            bound_check(logp, "logprior_logitrho: logp");
            return logp;
        }

        // eta_tilde = (Wtilde,logmu0)
        static arma::vec dlogJoint_deta( // Checked. OK.
            const arma::vec &y,  // (nT + 1) x 1
            const arma::vec &psi,   // (nT + 1) x 1
            const arma::vec &ft, // (nT + 1) x 1
            const arma::vec &eta,   // m x 1
            const std::vector<std::string> &param_selected,
            const Dist &W_prior,
            const Model &model,
            const arma::vec &rcomb)
        {
            std::map<std::string, AVAIL::Param> static_param_list = AVAIL::static_param_list;
            arma::vec deriv(param_selected.size());

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
                case AVAIL::Param::kappa:
                {
                    deriv.at(i) = dlogJoint_dkappa2(
                        y, ft, model.dobs, 
                        model.transfer.dlag, 
                        model.transfer.fgain.name, 
                        rcomb, 1., 1.);
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
            const double &kappa,
            const double &r,
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
                case AVAIL::Param::kappa:
                {
                    eta.at(i) = kappa;
                    break;
                }
                case AVAIL::Param::r:
                {
                    eta.at(i) = r;
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
            double &kappa,
            double &r,
            Model &model,
            const std::vector<std::string> &params_selected,
            const arma::vec &eta
        )
        {
            std::map<std::string, AVAIL::Param> static_param_list = AVAIL::static_param_list;
            for (unsigned int i = 0; i < params_selected.size(); i++)
            {
                double val = eta.at(i);
                switch (static_param_list[params_selected[i]])
                {
                case AVAIL::Param::W: // W is selected
                {
                    W = val;
                    model.derr.update_par1(val);
                    bound_check(val, "update_params: W", true, true);
                }
                break;
                case AVAIL::Param::mu0: // mu0 is selected
                {
                    mu0 = val;
                    model.dobs.update_par1(val);
                    bound_check(val, "update_params: mu0", false, true);
                }
                break;
                case AVAIL::Param::kappa: // par 1 is selected
                {
                    kappa = val;
                    model.transfer.dlag.update_par1(val);
                    bound_check(val, "update_params: par1", true, true);
                }
                break;
                case AVAIL::Param::r: // par 2 is selected
                {
                    r = val;
                    model.transfer.dlag.update_par2(val);
                    bound_check(val, "update_params: par2", true, true);
                }
                break;
                default:
                {
                    break;
                }
                }
            }

            return;
        }


        void infer(const Model &model_in, const arma::vec &y)
        {
            Model model = model_in;
            W = model.derr.par1;
            mu0 = model.dobs.par1;
            kappa = model.transfer.dlag.par1;
            r = model.transfer.dlag.par2;

            // Rcpp::List mcs_opts = SMC::MCS::default_settings();
            // mcs_opts["num_particle"]  = N;
            // mcs_opts["num_backward"] = B;
            // mcs_opts["W"] = W;
            // mcs_opts["use_discount"] = false;

            SMC::MCS mcs(model, y);
            mcs.init(mcs_opts);


            // arma::vec eta = init_eta(opts.params_selected, W, mu0, kappa, r, opts.update_static); // Checked. OK.
            // arma::vec eta_tilde = eta2tilde(eta, opts.params_selected, W.prior.name);

            arma::vec rcomb(model.dim.nT, arma::fill::zeros);
            for (unsigned int l = 1; l <= model.dim.nT; l++)
            {
                rcomb.at(l - 1) = nbinom::binom((unsigned int)model.transfer.dlag.par2 - 2 + l, l - 1);
            }

            
            for (unsigned int b = 0; b < ntotal; b ++)
            {
                bool saveiter = b > nburnin && ((b - nburnin - 1) % nthin == 0);
                R_CheckUserInterrupt();
                
                mcs.W = W;
                mcs.infer(model);
                arma::mat psi_all = mcs.get_psi_filter(); // (nT + 1) x M
                psi = arma::median(psi_all, 1);


                arma::vec ft  = psi;
                ft.at(0) = 0.;
                for (unsigned int t = 1; t < ft.n_elem; t ++)
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
                        y, psi, ft, eta, 
                        param_selected,
                        W_prior, model, rcomb); // Checked. OK.


                    arma::mat SigInv = get_sigma_inv(B, d, k);
                    arma::vec dlogq = dlogq_dtheta(SigInv, nu, eta_tilde, gamma, mu);
                    arma::vec ddiff = dlogJoint - dlogq;


                    arma::vec L_mu = dYJinv_dnu(nu, gamma) * ddiff;
                    grad_mu.update_grad(L_mu);
                    mu = mu + grad_mu.change;

                    if (m > 1)
                    {
                        arma::mat dtheta_dB = dYJinv_dB(nu, gamma, xi);                  // m x mk
                        arma::mat L_B = arma::reshape(dtheta_dB.t() * ddiff, m, k); // m x k
                        L_B.elem(B_uptri_idx).zeros();
                        arma::vec vecL_B = arma::vectorise(L_B); // mk x 1
                        grad_vecB.update_grad(vecL_B);

                        arma::mat B_change2 = arma::reshape(
                            grad_vecB.change, B.n_rows, B.n_cols); // m x k
                        B_change2.elem(B_uptri_idx).zeros();

                        B = B + B_change2;
                        B.elem(B_uptri_idx).zeros();
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
                    eta = tilde2eta(eta_tilde, param_selected, W_prior.name);


                    update_params(W, mu0, kappa, r, model, param_selected, eta);

                }

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
                    W_stored.at(idx_run) = W;
                    mu0_stored.at(idx_run) = mu0;
                    kappa_stored.at(idx_run) = kappa;
                    r_stored.at(idx_run) = r;
                }

                

                Rcpp::Rcout << "\rProgress: " << b << "/" << ntotal - 1;

            } // HVB loop

            Rcpp::Rcout << std::endl;
        }

        Rcpp::List get_output()
        {
            Rcpp::List output;
            output["psi_stored"] = Rcpp::wrap(psi_stored);
            arma::vec qprob = {0.025, 0.5, 0.975};
            arma::mat psi_quantile = arma::quantile(psi_stored, qprob, 1);
            output["psi"] = Rcpp::wrap(psi_quantile);

            if (infer_W)
            {
                output["W"] = Rcpp::wrap(W_stored);
            }

            if (infer_mu0)
            {
                output["mu0"] = Rcpp::wrap(mu0_stored);
            }

            if (infer_kappa)
            {
                output["kappa"] = Rcpp::wrap(kappa_stored);
            }

            if (infer_r)
            {
                output["r"] = Rcpp::wrap(r_stored);
            }

            output["mcs"] = mcs_opts;

            return output;
        }
    
    private:
        double learning_rate = 0.01;
        double eps_step_size = 1.e-6;
        unsigned int k = 1; // rank of unknown static parameters.

        Rcpp::List mcs_opts;

        // StaticParam W, mu0, delta, kappa, r;
        HybridParams grad_mu, grad_vecB, grad_d, grad_tau;
        arma::vec mu, d, gamma, nu, eps, eta, eta_tilde; // m x 1
        arma::vec xi; // k x 1
        arma::mat B; // m x k
        arma::vec vecL_B;  // mk x 1
        arma::uvec B_uptri_idx;



    }; // class Hybrid
}



#endif