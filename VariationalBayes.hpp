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
#include "StaticParams.hpp"


// #include "MCMC.hpp"

// [[Rcpp::plugins(cpp17)]]
// [[Rcpp::depends(RcppArmadillo)]]

namespace VB
{
    class VariationalBayes
    {
    public:
        unsigned int nsample = 1000;
        // unsigned int nthin = 2;
        unsigned int niter = 1000;
        // unsigned int ntotal = 3001;
        unsigned int N = 500; // number of SMC particles

        bool use_discount = false;
        double discount_factor = 0.95;

        bool update_static = true;
        unsigned int m = 1; // number of unknown static parameters
        std::vector<std::string> param_selected = {"W"};

        Prior W_prior, seas_prior, rho_prior, par1_prior, par2_prior;
        bool zintercept_infer = false;
        bool zzcoef_infer = false;

        arma::vec W_stored;    // nsample x 1
        arma::mat seas_stored;  // period x nsample
        arma::vec rho_stored;  // nsample x 1
        arma::vec par1_stored; // nsample x 1
        arma::vec par2_stored; // nsample x 1
        arma::mat psi_stored; // (nT + 1) x nsample
        arma::mat z_stored; // (nT + 1) x nsample
        arma::mat prob_stored; // (nT + 1) x nsample
        arma::vec zintercept_stored; // nsample x 1
        arma::vec zzcoef_stored; // nsample x 1

        VariationalBayes(const Model &model, const Rcpp::List &vb_opts)
        {
            Rcpp::List opts = vb_opts;

            nsample = 1000;
            if (opts.containsElementNamed("nsample"))
            {
                nsample = Rcpp::as<unsigned int>(opts["nsample"]);
            }

            // nthin = 1000;
            // if (opts.containsElementNamed("nthin"))
            // {
            //     nthin = Rcpp::as<unsigned int>(opts["nthin"]);
            // }

            niter = 5000;
            if (opts.containsElementNamed("niter"))
            {
                niter = Rcpp::as<unsigned int>(opts["niter"]);
            }
            else if (opts.containsElementNamed("nburnin"))
            {
                niter = Rcpp::as<unsigned int>(opts["nburnin"]);
            }

            // ntotal = niter + nthin * nsample + 1;

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

            Static::init_prior(
                param_selected, m,
                W_prior, seas_prior, rho_prior,
                par1_prior, par2_prior,
                zintercept_infer, zzcoef_infer,
                opts, model);

            update_static = false;
            if (m > 0)
            {
                update_static = true;
            }

            if (W_prior.infer)
            {
                W_stored.set_size(nsample);
                W_stored.zeros();
            }

            if (seas_prior.infer)
            {
                seas_stored.set_size(model.seas.period, nsample);
                seas_stored.zeros();
            }

            if (rho_prior.infer)
            {
                rho_stored.set_size(nsample);
                rho_stored.zeros();
            }

            if (par1_prior.infer)
            {
                par1_stored.set_size(nsample);
                par1_stored.zeros();
            }

            if (par2_prior.infer)
            {
                par2_stored.set_size(nsample);
                par2_stored.zeros();
            }

            if (zintercept_infer)
            {
                zintercept_stored.set_size(nsample);
                zintercept_stored.zeros();
            }

            if (zzcoef_infer) {
                zzcoef_stored.set_size(nsample);
                zzcoef_stored.zeros();
            }
            
        }


        static Rcpp::List default_settings()
        {
            Rcpp::List opts = Static::default_settings();
            opts["nsample"] = 1000;
            // opts["nthin"] = 1;
            opts["niter"] = 5000;
            opts["num_particle"] = 500;
            opts["use_discount"] = false;
            opts["discount_factor"] = 0.95;
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
        arma::cube Theta_stored; // nP x (nT + 1) x nsample
        std::string fsys;

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

            marglike_stored.set_size(niter);
            marglike_stored.zeros();

            // condlike_stored.set_size(niter);
            // condlike_stored.zeros();

            // grad_stored.set_size(niter);
            // grad_stored.zeros();

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

            eta = Static::init_eta(param_selected, model_in, update_static); // Checked. OK.
            eta_tilde = Static::eta2tilde(
                eta, param_selected, W_prior.name, 
                par1_prior.name, model_in.dobs.name,
                model_in.seas.period, model_in.seas.in_state);

            nu = tYJ(eta_tilde, gamma);

            B.set_size(m, k);
            B.zeros();
            grad_vecB.init(m * k, learning_rate, eps_step_size);

            xi.set_size(k);
            xi.zeros();
        }


        static Rcpp::List default_settings()
        {
            Rcpp::List opts = VariationalBayes::default_settings();
            opts["learning_rate"] = 0.01;
            opts["eps_step_size"] = 1.e-6;
            opts["k"] = 1;
            return opts;
        }

       
        void infer(
            Model &model, 
            const arma::vec &y,
            const bool &verbose = VERBOSE)
        {
            std::map<std::string, SysEq::Evolution> sys_list = SysEq::sys_list;
            fsys = model.fsys;
            const unsigned int nT = y.n_elem - 1;

            model.zero.z.set_size(y.n_elem);
            model.zero.z.ones();
            model.zero.prob = model.zero.z;


            for (unsigned int b = 0; b < niter; b++)
            {
                // bool saveiter = b > niter && ((b - niter - 1) % nthin == 0);
                Rcpp::checkUserInterrupt();

                // TFS sampler
                // ------------------
                // You MUST set initial_resample_all = true (MCS smoothing) and final_resample_by_weights = false (reduce degeneracy) to make this algorithm work.
                // arma::cube Theta_tmp = arma::zeros<arma::cube>(model.nP, N, y.n_elem);
                arma::cube Theta_tmp = arma::zeros<arma::cube>(model.nP, N, y.n_elem);
                arma::mat ztmp(N, y.n_elem, arma::fill::ones);
                double marg_loglik = SMC::SequentialMonteCarlo::auxiliary_filter0(
                    Theta_tmp, ztmp, model, y, N, 
                    true, false, use_discount, discount_factor);
                arma::mat Theta = arma::mean(Theta_tmp, 1); // nP x (nT + 1)

                marglike_stored.at(b) = marg_loglik;

                if (model.zero.inflated)
                {
                    model.zero.prob = arma::vectorise(arma::mean(ztmp, 0)); // (nT + 1) x 1
                    for (unsigned int t = 0; t < model.zero.z.n_elem; t++)
                    {
                        model.zero.z.at(t) = (R::runif(0., 1.) < model.zero.prob.at(t)) ? 1. : 0.;
                    }
                        
                }
                // ------------------

                // // MCMC sampler
                // // ------------------
                // ApproxDisturbance approx_dlm(nT, model.fgain);
                // approx_dlm.set_Fphi(model.dlag, model.dlag.nL);
                // arma::vec wt_accept(nT + 1, arma::fill::zeros);
                // arma::vec wt = arma::randn<arma::vec>(nT + 1);
                // for (unsigned int k = 0; k < 100; k++)
                // {
                //     MCMC::Posterior::update_wt(wt, wt_accept, approx_dlm, y, model);
                // }

                // arma::vec psi = arma::cumsum(wt); // (nT + 1) x 1
                // arma::mat Theta = TransFunc::psi2theta(psi, y, model.ftrans, model.fgain, model.dlag);
                // // ------------------


                arma::mat dFphi_grad;
                arma::vec dloglik_dlag(2, arma::fill::zeros);
                if (par1_prior.infer || par2_prior.infer)
                {
                    dFphi_grad = LagDist::get_Fphi_grad(
                        model.dlag.nL, model.dlag.name, 
                        model.dlag.par1, model.dlag.par2);
                }

                arma::vec lambda(y.n_elem, arma::fill::zeros);
                for (unsigned int t = 1; t < y.n_elem; t++)
                {
                    if (model.zero.z.at(t) < EPS)
                    {
                        continue;
                    }

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

                    // condlike_stored.at(b) += ObsDist::loglike(
                    //     y.at(t), model.dobs.name, lambda.at(t), model.dobs.par2, true);

                    if (par1_prior.infer || par2_prior.infer)
                    {
                        if (!model.zero.inflated || model.zero.z.at(t) > EPS)
                        {
                            double dll_deta = Model::dloglik_deta(
                                ft, y.at(t), model.dobs.par2, model.dobs.name, model.flink);
                            double deta_dpar1 = arma::accu(dFphi_grad.col(0) % yold % htheta);
                            double deta_dpar2 = arma::accu(dFphi_grad.col(1) % yold % htheta);

                            dloglik_dlag.at(0) += dll_deta * deta_dpar1;
                            dloglik_dlag.at(1) += dll_deta * deta_dpar2;
                        }
                    }
                }

                if (update_static)
                {
                    arma::vec dlogJoint = Static::dlogJoint_deta(
                        y, Theta, lambda, dloglik_dlag, eta, param_selected,
                        W_prior, par1_prior, par2_prior, rho_prior, seas_prior, model); // Checked. OK.

                    arma::mat SigInv = get_sigma_inv(B, d, k);
                    arma::vec dlogq = dlogq_dtheta(SigInv, nu, eta_tilde, gamma, mu);
                    arma::vec ddiff = dlogJoint - dlogq;

                    // mu
                    arma::vec dyji_dnu = dYJinv_dnu_diag(nu, gamma); // m x 1
                    arma::vec L_mu = dyji_dnu % ddiff;
                    grad_mu.update_grad(L_mu);
                    mu += grad_mu.change;

                    // grad_stored.at(b) += arma::accu(arma::abs(grad_mu.change));

                    if (m > 1)
                    {
                        arma::mat L_B(m, k, arma::fill::zeros);
                        if (k > 1)
                        {
                            L_B = L_mu * xi.t(); // m x k, Appendix B.2 equation (ii)

                            // Enforce lower-triangular (or diagonal) constraint
                            for (unsigned int col = 0; col < k; ++col) {
                                for (unsigned int row = 0; row < col; ++row) { // strict upper
                                    L_B(row, col) = 0.0;
                                }
                            }
                            
                        }
                        else
                        {
                            L_B.col(0) = L_mu * xi[0];
                        }


                        // Non‑owning view over memory (no copy)
                        arma::vec L_B_view(const_cast<double*>(L_B.memptr()), m * k, false, true);
                        grad_vecB.update_grad(L_B_view);

                        // Apply update (reshape view of change)
                        arma::mat B_change2(const_cast<double*>(grad_vecB.change.memptr()), m, k, false, true);
                        if (k == 1) {
                            B.col(0) += B_change2.col(0);
                        } else {
                            for (unsigned int col = 0; col < k; ++col) {
                                for (unsigned int row = 0; row < m; ++row) {
                                    if (row < col) {
                                        B(row, col) = 0.0; // strict upper -> zero
                                    } else {
                                        B(row, col) += B_change2(row, col); // lower/diag -> add
                                    }
                                }
                            }
                        }

                        // grad_stored.at(b) += arma::accu(arma::abs(grad_vecB.change));
                    }

                    // d
                    arma::vec L_d = (eps % dyji_dnu) % ddiff; // m x 1, Section B.2 equation (iii), Ref: `dtheta_dBDelta.m`
                    grad_d.update_grad(L_d);
                    d += grad_d.change;
                    // grad_stored.at(b) += arma::accu(arma::abs(grad_d.change));

                    // tau
                    arma::vec tau = gamma2tau(gamma);
                    arma::vec L_tau = dYJinv_dtau_diag(nu, gamma) % ddiff;
                    grad_tau.update_grad(L_tau);
                    tau += grad_tau.change;
                    tau2gamma(tau, gamma);
                    // grad_stored.at(b) += arma::accu(arma::abs(grad_tau.change));

                    rtheta(nu, eta_tilde, xi, eps, gamma, mu, B, d);
                    eta = Static::tilde2eta(
                        eta_tilde, param_selected, 
                        W_prior.name, par1_prior.name, 
                        model.dlag.name, model.dobs.name,
                        model.seas.period, model.seas.in_state);
                    Static::update_params(model, param_selected, eta);
                } // end update_static


                if (verbose)
                {
                    Rcpp::Rcout << "\rProgress: " << b + 1 << "/" << niter;
                }

            } // HVB SGD Loop

            if (verbose)
            {
                Rcpp::Rcout << std::endl;
            }


            if (model.zero.inflated)
            {
                z_stored.set_size(y.n_elem, nsample);
                prob_stored.set_size(y.n_elem, nsample);
            }

            if (sys_list[model.fsys] == SysEq::Evolution::identity)
            {
                psi_stored.set_size(model.nP, nsample);
                Theta_stored = arma::zeros<arma::cube>(model.nP, y.n_elem, nsample);
            }
            else
            {
                psi_stored.set_size(y.n_elem, nsample);
            }

            psi_stored.zeros();

            std::map<std::string, AVAIL::Param> static_param_list = AVAIL::static_param_list;

            // arma::cube Theta_tmp = arma::zeros<arma::cube>(model.nP, nsample, y.n_elem);
            // arma::mat ztmp(nsample, y.n_elem, arma::fill::ones);
            // arma::mat ptmp(nsample, y.n_elem, arma::fill::randu);
            // double log_cond_marg = SMC::SequentialMonteCarlo::auxiliary_filter0(
            //     Theta_tmp, ztmp, ptmp, model, y, nsample,
            //     true, true, false, 1.);
            
            // if (sys_list[model.fsys] != SysEq::Evolution::identity)
            // {
            //     psi_stored = Theta_tmp.row_as_mat(0); // (nT + 1) x nsample
            // }

            arma::mat eta_tilde = rtheta_batch(gamma, mu, B, d, nsample); // m x nsample

            for (unsigned int i = 0; i < nsample; i++)
            {
                eta = Static::tilde2eta(
                    eta_tilde.col(i), param_selected, 
                    W_prior.name, par1_prior.name, 
                    model.dlag.name, model.dobs.name,
                    model.seas.period, model.seas.in_state);
                
                Static::update_params(model, param_selected, eta);

                arma::cube Theta_tmp = arma::zeros<arma::cube>(model.nP, N, y.n_elem);
                arma::mat ztmp(N, y.n_elem, arma::fill::ones);
                double log_cond_marg = SMC::SequentialMonteCarlo::auxiliary_filter0(
                    Theta_tmp, ztmp, model, y, N, 
                    true, false, use_discount, discount_factor);
                arma::mat Theta = arma::mean(Theta_tmp, 1); // nP x (nT + 1)

                if (model.zero.inflated)
                {
                    prob_stored.col(i) = arma::vectorise(arma::mean(ztmp, 0)); // (nT + 1) x 1
                    for (unsigned int t = 0; t < model.zero.z.n_elem; t++)
                    {
                        z_stored.at(t, i) = (R::runif(0., 1.) < prob_stored.at(t, i)) ? 1. : 0.;
                    }
                }

                // ApproxDisturbance approx_dlm(nT, model.fgain);
                // approx_dlm.set_Fphi(model.dlag, model.dlag.nL);
                // arma::vec wt_accept(nT + 1, arma::fill::zeros);
                // arma::vec wt = arma::randn<arma::vec>(nT + 1);
                // for (unsigned int k = 0; k < 100; k++)
                // {
                //     MCMC::Posterior::update_wt(wt, wt_accept, approx_dlm, y, model);
                // }
                    
                // arma::vec psi = arma::cumsum(wt); // (nT + 1) x 1
                // arma::mat Theta = TransFunc::psi2theta(psi, y, model.ftrans, model.fgain, model.dlag);


                if (sys_list[model.fsys] == SysEq::Evolution::identity)
                {
                    psi_stored.col(i) = Theta.col(y.n_elem - 1);
                    Theta_stored.slice(i) = Theta;
                }
                else
                {
                    psi_stored.col(i) = arma::vectorise(Theta.row(0));
                }

                unsigned int idx = 0;
                for (unsigned int k = 0; k < param_selected.size(); k++)
                {
                    /*
                    idx: location of parameter in eta
                    k: location of parameter name in param_selected
                    i: the i-th sample in **_stored
                    */
                    double val = eta.at(idx);
                    switch (static_param_list[param_selected[k]])
                    {
                    case AVAIL::Param::W:
                    {
                        W_stored.at(i) = val;
                        idx += 1;
                        break;
                    }
                    case AVAIL::Param::seas:
                    {
                        if (!model.seas.in_state)
                        {
                            arma::vec seass = eta.subvec(idx, idx + model.seas.period - 1);
                            seas_stored.col(i) = seass;
                            idx += model.seas.period;
                        }
                        break;
                    }
                    case AVAIL::Param::rho:
                    {
                        rho_stored.at(i) = val;
                        idx += 1;
                        break;
                    }
                    case AVAIL::Param::lag_par1:
                    {
                        par1_stored.at(i) = val;
                        idx += 1;
                        break;
                    }
                    case AVAIL::Param::lag_par2:
                    {
                        par2_stored.at(i) = val;
                        idx += 1;
                        break;
                    }
                    case AVAIL::Param::zintercept:
                    {
                        zintercept_stored.at(i) = val;
                        idx += 1;
                        break;
                    }
                    case AVAIL::Param::zzcoef:
                    {
                        zzcoef_stored.at(i) = val;
                        idx += 1;
                        break;
                    }
                    default:
                    {
                        throw std::invalid_argument("VariationalBayes::infer: undefined static parameter " + param_selected[k]);
                    }
                    }
                }

                if (verbose)
                {
                    Rcpp::Rcout << "\rProgress: " << i + 1 << "/" << nsample;
                }
            } // sampling loop

            if (verbose)
            {
                Rcpp::Rcout << std::endl;
            }

        }

        Rcpp::List get_output()
        {
            Rcpp::List output;
            output["marglik"] = Rcpp::wrap(marglike_stored);
            // output["condlik"] = Rcpp::wrap(condlike_stored);
            // output["grad_norm"] = Rcpp::wrap(grad_stored);

            output["psi_stored"] = Rcpp::wrap(psi_stored);
            arma::vec qprob = {0.025, 0.5, 0.975};
            arma::mat psi_quantile = arma::quantile(psi_stored, qprob, 1);
            output["psi"] = Rcpp::wrap(psi_quantile);

            std::map<std::string, SysEq::Evolution> sys_list = SysEq::sys_list;
            if (sys_list[fsys] == SysEq::Evolution::identity)
            {
                output["Theta"] = Rcpp::wrap(Theta_stored);
            }

            if (!z_stored.is_empty() && !prob_stored.is_empty())
            {
                output["z"] = Rcpp::wrap(arma::vectorise(arma::mean(z_stored, 1)));
                output["prob"] = Rcpp::wrap(prob_stored);
            }

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

            if (zintercept_infer)
            {
                output["zintercept"] = Rcpp::wrap(zintercept_stored.t());
            }

            if (zzcoef_infer)
            {
                output["zzcoef"] = Rcpp::wrap(zzcoef_stored.t());
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
        arma::vec marglike_stored; //, condlike_stored, grad_stored; // niter x 1

        arma::vec xi;                                    // k x 1
        arma::mat B;                                     // m x k
    }; // class Hybrid
}

#endif