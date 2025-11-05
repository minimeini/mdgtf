#ifndef _STATICPARAMS_HPP
#define _STATICPARAMS_HPP

#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <algorithm>
#include <RcppArmadillo.h>
#include "../core/LinkFunc.hpp"
#include "../utils/yjtrans.h"
#include "Model.hpp"


// [[Rcpp::depends(RcppArmadillo)]]

namespace Static
{
    inline Rcpp::List default_settings()
    {
        Rcpp::List W_opts;
        W_opts["infer"] = false;
        W_opts["prior_name"] = "invgamma";
        W_opts["prior_param"] = Rcpp::NumericVector::create(1., 1.);

        Rcpp::List seas_opts;
        seas_opts["infer"] = false;
        seas_opts["mh_sd"] = 1.;
        seas_opts["prior_name"] = "gaussian";
        seas_opts["prior_param"] = Rcpp::NumericVector::create(1., 10.);

        Rcpp::List rho_opts;
        rho_opts["infer"] = false;
        rho_opts["mh_sd"] = 1.;
        rho_opts["prior_param"] = Rcpp::NumericVector::create(1., 1.);
        rho_opts["prior_name"] = "invgamma";

        Rcpp::List par1_opts;
        par1_opts["infer"] = false;
        par1_opts["mh_sd"] = 1.;
        par1_opts["prior_param"] = Rcpp::NumericVector::create(0., 1.);
        par1_opts["prior_name"] = "gaussian";

        Rcpp::List par2_opts;
        par2_opts["infer"] = false;
        par2_opts["mh_sd"] = 1.;
        par2_opts["prior_param"] = Rcpp::NumericVector::create(1., 1.);
        par2_opts["prior_name"] = "invgamma";

        Rcpp::List opts;
        opts["W"] = W_opts;
        opts["seas"] = seas_opts;
        opts["rho"] = rho_opts;
        opts["par1"] = par1_opts;
        opts["par2"] = par2_opts;
        opts["zintercept"] = false;
        opts["zzcoef"] = false;

        return opts;
    }

    inline void init_prior(
        std::vector<std::string> &param_selected, // vector of names of unknown params
        unsigned int &m, // total number of unknown static parameters
        Prior &W_prior,
        Prior &seas_prior,
        Prior &rho_prior,
        Prior &par1_prior,
        Prior &par2_prior,
        bool &zintercept_infer,
        bool &zzcoef_infer,
        const Rcpp::List &algo_opts,
        const Model &model
    ) 
    {
        Rcpp::List opts = algo_opts;
        m = 0;
        zintercept_infer = false;
        zzcoef_infer = false;
        param_selected.clear();

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

        if (opts.containsElementNamed("zintercept"))
        {
            zintercept_infer = Rcpp::as<bool>(opts["zintercept"]);
            if (zintercept_infer)
            {
                param_selected.push_back("zintercept");
                m += 1;
            }
        }

        zzcoef_infer = false;
        if (opts.containsElementNamed("zzcoef"))
        {
            zzcoef_infer = Rcpp::as<bool>(opts["zzcoef"]);
            if (zzcoef_infer)
            {
                param_selected.push_back("zzcoef");
                m += 1;
            }
        }
    }


    inline arma::vec init_eta( // Checked. OK.
        const std::vector<std::string> &params_selected,
        const Model &model,
        const bool &update_static = true)
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
                eta.at(idx) = model.dobs.par2;
                idx += 1;
                break;
            }
            case AVAIL::Param::lag_par1:
            {
                eta.at(idx) = model.dlag.par1;
                idx += 1;
                break;
            }
            case AVAIL::Param::lag_par2:
            {
                eta.at(idx) = model.dlag.par2;
                idx += 1;
                break;
            }
            case AVAIL::Param::zintercept:
            {
                eta.at(idx) = model.zero.intercept;
                idx += 1;
                break;
            }
            case AVAIL::Param::zzcoef:
            {
                eta.at(idx) = model.zero.coef;
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

    inline void update_params(
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
                if (model.derr.full_rank)
                {
                    model.derr.var.zeros();
                    model.derr.var.diag().fill(val);
                }
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
            case AVAIL::Param::zintercept:
            {
                model.zero.intercept = val;
                idx += 1;
                break;
            }
            case AVAIL::Param::zzcoef:
            {
                model.zero.coef = val;
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


    inline arma::vec eta2tilde( // Checked. OK.
        const arma::vec &eta,   // m x 1
        const std::vector<std::string> &param_selected,
        const std::string &W_prior = "invgamma",
        const std::string &lag_par1_prior = "gaussian",
        const std::string &obs_dist = "nbinom",
        const unsigned int &seasonal_period = 1,
        const bool &season_in_state = false)
    {
        std::map<std::string, AVAIL::Dist> obs_list = ObsDist::obs_list;
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

                    if (obs_list[obs_dist] == AVAIL::Dist::nbinomm)
                    {
                        eta_tilde.subvec(idx, idx + seasonal_period - 1) = arma::log(arma::abs(seas) + EPS);
                    }
                    else
                    {
                        eta_tilde.subvec(idx, idx + seasonal_period - 1) = seas;
                    }

                    
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
                eta_tilde.at(idx) = std::log(std::abs(val) + EPS);
                idx += 1;
                break;
            } // lag_par2
            case AVAIL::Param::zintercept:
            {
                eta_tilde.at(idx) = val;
                idx += 1;
                break;
            }
            case AVAIL::Param::zzcoef:
            {
                idx += 1;
                eta_tilde.at(idx) = val;
                break;
            }
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


    inline arma::vec tilde2eta(     // Checked. OK.
        const arma::vec &eta_tilde, // m x 1
        const std::vector<std::string> &param_selected,
        const std::string &W_prior = "invgamma",
        const std::string &lag_par1_prior = "gaussian",
        const std::string &lag_dist = "lognorm",
        const std::string &obs_dist = "nbinom",
        const unsigned int &seasonal_period = 1,
        const bool &season_in_state = false)
    {
        std::map<std::string, AVAIL::Dist> dist_list = AVAIL::dist_list;
        std::map<std::string, AVAIL::Dist> lag_list = LagDist::lag_list;
        std::map<std::string, AVAIL::Dist> obs_list = ObsDist::obs_list;
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
                    arma::vec seas_tilde = eta_tilde.subvec(idx, idx + seasonal_period - 1);
                    if (obs_list[obs_dist] == AVAIL::Dist::nbinomm)
                    {
                        eta.subvec(idx, idx + seasonal_period - 1) = arma::trunc_exp(seas_tilde);
                    }
                    else
                    {
                        eta.subvec(idx, idx + seasonal_period - 1) = seas_tilde;
                    }
                    
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
            case AVAIL::Param::zintercept:
            {
                eta.at(idx) = val;
                idx += 1;
                break;
            }
            case AVAIL::Param::zzcoef:
            {
                eta.at(idx) = val;
                idx += 1;
                break;
            }
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
       static double dlogJoint_dWtilde(
        const arma::mat &Theta, // nP x (nT + 1)
        const double W,              // evolution variance conditional on V
        const Dist &W_prior,
        const ZeroInflation &zero,
        const bool &full_rank = false)
    {
        std::map<std::string, AVAIL::Dist> dist_list = AVAIL::dist_list;
        double aw = W_prior.par1;
        double bw = W_prior.par2;
        const double p = full_rank ? static_cast<double>(Theta.n_rows) : 1.;

        double res = 0.;
        double cnt = 0.;
        for (unsigned int t = 1; t < Theta.n_cols; t++)
        {
            if (zero.inflated && zero.z.at(t) < EPS)
            {
                continue;
            }

            cnt += 1.;
            double tmp = 0.;
            if (full_rank)
            {
                arma::vec err = Theta.col(t) - Theta.col(t - 1);
                tmp = arma::dot(err, err);
            }
            else
            {
                double err = Theta.at(0, t) - Theta.at(0, t - 1);
                tmp = err * err;
            }
            res += tmp;
        }
        // res = sum((w[t])^2)

        double deriv;
        if (dist_list[W_prior.name] == AVAIL::Dist::invgamma)
        {
            double bnew = bw + 0.5 * res; // IG prior on W
            double log_bnew_W = std::log(std::abs(bnew) + EPS) - std::log(std::abs(W) + EPS);
            log_bnew_W = std::min(log_bnew_W, UPBND);
            deriv = -std::exp(log_bnew_W);

            double a_new = aw; // IG prior on W
            a_new += 0.5 * cnt * p;
            a_new += 1.;
            deriv += a_new;
        }
        else
        {
            double rdw = std::log(std::abs(res) + EPS) - std::log(std::abs(W) + EPS); // sum(w[t]^2) / W
            rdw = std::min(rdw, UPBND);
            rdw = std::exp(rdw);
            #ifdef DGTF_DO_BOUND_CHECK
            bound_check(rdw, "dlogJoint_dWtilde: rdw");
            #endif

            if (dist_list[W_prior.name] == AVAIL::Dist::gamma)
            {
                deriv = aw;
                deriv -= 0.5 * cnt;
                deriv -= bw * W;
                deriv += 0.5 * rdw;
            }
            else if (dist_list[W_prior.name] == AVAIL::Dist::halfcauchy)
            {
                deriv = 0.5 * cnt - 0.5 * rdw + W / (bw * bw + W) - 0.5;
            }
            else
            {
                throw std::invalid_argument("dlogJoint_dWtilde: prior is not defined.");
            }
        }

        #ifdef DGTF_DO_BOUND_CHECK
            bound_check(deriv, "dlogJoint_dWtilde: deriv");
        #endif
        return deriv;
    }



    /*
    ------ dlogJoint_dseas ------
    The derivative of the full joint density with respect to seasonality term:
        In a linear NB model, it is a derivative w.r.t. log(seas) since seas > 0
        In a nonlinear NB model, it is a derivative w.r.t. seas directly since seas in R.
    */
    inline arma::vec dloglike_dseas( // Checked. OK.
        const arma::vec &y,           // (nT+1) x 1
        const arma::vec &lambda,          // (n+1) x 1, (f[0],f[1],...,f[nT])
        const ObsDist &dobs,
        const ZeroInflation &zero,
        const arma::vec &seas,
        const arma::mat &Xseas,
        const std::string &flink
    )
    { // 0 - negative binomial; 1 - poisson
        std::map<std::string, AVAIL::Dist> obs_list = ObsDist::obs_list;

        arma::vec deriv(seas.n_elem, arma::fill::zeros);
        for (unsigned int t = 1; t < y.n_elem; t++)
        {
            if (zero.inflated && zero.z.at(t) < EPS)
            {
                continue;
            }

            double dy_dlambda = ObsDist::dloglike_dlambda(y.at(t), lambda.at(t), dobs);
            double eta = LinkFunc::mu2ft(lambda.at(t), flink);
            double lam;
            double dlambda_deta = LinkFunc::dlambda_deta(lam, eta, flink);
            arma::vec deta_dseas = Xseas.col(t);
            if (obs_list[dobs.name] == AVAIL::Dist::nbinomm)
            {
                // seas is restricted to be nonnegative
                arma::vec dseas_dlogseas = seas;
                deta_dseas %= dseas_dlogseas;
            }
            deriv = deriv + (dy_dlambda * dlambda_deta) * deta_dseas;
        }
        return deriv;
    }


    inline arma::vec dlogprior_dseas(
        const arma::vec &seas, // evolution variance conditional on V
        const double &sig2_mu0 = 10.)
    {
        /*
        log(mu0) ~ N(0,sig2_mu0)
        */
        arma::vec logp = - seas / sig2_mu0;
        #ifdef DGTF_DO_BOUND_CHECK
            bound_check<arma::vec>(logp, "logprior_logseas: logp");
        #endif
        return logp;
    }

    inline double dlogJoint_dzintercept(
        const ZeroInflation &zero
    ){
        // The derivative of joint probability w.r.t to intercept is
        // sum(z[t] - p[t])
        double grad = 0.;
        for (unsigned int t = 1; t < zero.z.n_elem; t++)
        {
            // logit(p[t]) = a + c * (z[t-1] == 1)
            double val = zero.intercept + zero.coef * zero.z.at(t - 1);
            if (!zero.X.is_empty())
            {
                val += arma::dot(zero.X.col(t), zero.beta);
            }
            double prob = logistic(val); // This is p[t]
            grad += zero.z.at(t) - prob;
        }
        return grad;
    }

    inline double dlogJoint_dzzcoef(const ZeroInflation& zero)
    {
        // The derivative of joint probability w.r.t to slope is
        // sum((z[t] - p[t]) * (z[t-1] == 1))
        double grad = 0.;
        for (unsigned int t = 1; t < zero.z.n_elem; t++)
        {
            if (zero.z.at(t - 1) > EPS)
            {
                // If z[t-1] == 1
                double val = zero.intercept + zero.coef;
                if (!zero.X.is_empty())
                {
                    val += arma::dot(zero.X.col(t), zero.beta);
                }
                double prob = logistic(val); // This is p[t]
                grad += zero.z.at(t) - prob;
            }
        }
        return grad;
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
    inline arma::vec dlogJoint_deta( // Checked. OK.
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
        std::map<std::string, AVAIL::Dist> obs_list = ObsDist::obs_list;
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
                deriv.at(idx) = dlogJoint_dWtilde(
                    Theta, val, W_prior, model.zero, model.derr.full_rank);
                idx += 1;
                break;
            }
            case AVAIL::Param::seas:
            {
                arma::vec seas = arma::abs(eta.subvec(idx, idx + model.seas.period - 1));
                arma::vec seas_tilde = seas;
                if (obs_list[model.dobs.name] == AVAIL::Dist::nbinomm)
                {
                    seas_tilde = arma::log(arma::abs(seas) + EPS);
                }

                arma::vec dloglik = dloglike_dseas(
                    y, lambda, model.dobs, model.zero, seas, model.seas.X, model.flink);
                arma::vec dlogprior = dlogprior_dseas(seas_tilde, seas_prior.par2);

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
            case AVAIL::Param::zintercept:
            {
                deriv.at(idx) = dlogJoint_dzintercept(model.zero);
                idx += 1;
                break;
            }
            case AVAIL::Param::zzcoef:
            {
                deriv.at(idx) = dlogJoint_dzzcoef(model.zero);
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

    inline double logJoint(
        const arma::vec &y,      // (nT + 1) x 1
        const arma::mat &Theta,  // nP x (nT + 1)
        const arma::vec &lambda, // (nT + 1) x 1
        const Prior &W_prior,
        const Prior &par1_prior,
        const Prior &par2_prior,
        const Prior &rho_prior,
        const Prior &seas_prior,
        const Model &model)
    {
        std::map<std::string, AVAIL::Dist> obs_list = ObsDist::obs_list;
        double logp = 0.;
        double Wchol = std::sqrt(model.derr.par1);

        for (unsigned int t = 1; t < y.n_elem; t++)
        {
            // Add likelihood of y[t]
            if (model.zero.inflated && model.zero.z.at(t) < EPS)
            {
                // p(y[t] = 0 | theta[t], z[t] = 0, gamma) = 1
                logp += y.at(t) < EPS ? 0. : -9e16; // numerically exp(-9e16) = 0
            }
            else
            {
                // When z[t] = 1
                // Likelihood p(y[t] | theta[t], z[t] = 1, gamma) is NB(y[t] | lambda[t], rho)
                logp += ObsDist::loglike(y.at(t), model.dobs.name, lambda.at(t), model.dobs.par2, true);
            }

            // Add evolution from theta[t-1] to theta[t]
            if (W_prior.infer && !model.derr.full_rank)
            {
                // logp(theta[t] | theta[t-1], gamma)
                logp += R::dnorm4(Theta.at(0, t), Theta.at(0, t - 1), Wchol, true);
            }

            // Add evolution from z[t-1] to z[t] for zero-inflated model
            if (model.zero.inflated)
            {
                double val = model.zero.intercept + model.zero.coef * model.zero.z.at(t - 1);
                if (!model.zero.X.is_empty())
                {
                    val += arma::dot(model.zero.X.col(t), model.zero.beta);
                }
                double prob = logistic(val); // p(z[t] = 1 | z[t-1], gamma)

                if (model.zero.z.at(t) > EPS)
                {
                    // z[t] = 1 with probability `prob`
                    logp += std::log(prob);
                }
                else
                {
                    // z[t] = 0 with probability `1 - prob`
                    logp += std::log(std::abs(1. - prob) + EPS);
                }
            }
        } // End loop over time `t`

        // Add priors
        if (par1_prior.infer)
        {
            logp += Prior::dprior(model.dlag.par1, par1_prior, true, true); // TODO: check it
        }

        if (par2_prior.infer)
        {
            logp += Prior::dprior(model.dlag.par2, par2_prior, true, true); // TODO: check it
        }

        if (W_prior.infer)
        {
            logp += Prior::dprior(model.derr.par1, W_prior, true, true);
        }

        if (rho_prior.infer)
        {
            logp += Prior::dprior(model.dobs.par2, rho_prior, true, true);
        }

        if (seas_prior.infer)
        {
            for (unsigned int i = 0; i < model.seas.period; i++)
            {
                double seas = model.seas.val.at(i);
                if (obs_list[model.dobs.name] == AVAIL::Dist::nbinomm)
                {
                    seas = std::log(std::abs(seas) + EPS);
                }
                logp += Prior::dprior(seas, seas_prior, true, true);
            }
        }

        return logp;
    }
}

#endif