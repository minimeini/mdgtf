#pragma once
#ifndef REGRESSION_H
#define REGRESSION_H

#include <RcppArmadillo.h>
#include "../utils/utils.h"
#include "../inference/hmc.hpp"
#include "../core/ObsDist.hpp"

/**
 * @brief Seasonality component to the conditional intensity.
 * 
 */
class Season
{
public:
    bool in_state = false;
    unsigned int period = 0;
    arma::vec val; // period x 1
    arma::mat X; // period x (ntime + 1)
    arma::mat P; // period x period
    double lobnd = 1.;
    double hibnd = 10.;

    Season()
    {
        init_default();
        return;
    }

    Season(const Rcpp::List &settings)
    {
        init(settings);
        return;
    }

    void init_default()
    {
        in_state = false;
        period = 0;
    }

    void init(const Rcpp::List &settings)
    {
        Rcpp::List opts = settings;

        in_state = false;
        if (opts.containsElementNamed("in_state"))
        {
            in_state = Rcpp::as<bool>(opts["in_state"]);
        }

        period = 1;
        if (opts.containsElementNamed("period"))
        {
            period = Rcpp::as<unsigned int>(opts["period"]);
        }

        lobnd = 1.;
        if (opts.containsElementNamed("lobnd"))
        {
            lobnd = Rcpp::as<double>(opts["lobnd"]);
        }

        hibnd = 10.;
        if (opts.containsElementNamed("hibnd"))
        {
            hibnd = Rcpp::as<double>(opts["hibnd"]);
        }

        val.set_size(period);
        val.zeros();
        unsigned int nelem = 0;
        if (opts.containsElementNamed("init"))
        {
            arma::vec init = Rcpp::as<arma::vec>(opts["init"]);
            nelem = (init.n_elem <= period) ? init.n_elem : period;
            val.head(nelem) = init.head(nelem);
        }

        if (nelem < period)
        {
            val.tail(period - nelem) = arma::randu(period - nelem, arma::distr_param(lobnd, hibnd));
        }

        P.set_size(period, period);
        P.zeros();
        P.at(period - 1, 0) = 1.;
        if (period > 1)
        {
            for (unsigned int i = 0; i < period - 1; i++)
            {
                P.at(i, i + 1) = 1.;
            }
        }
    }


    static Rcpp::List default_settings()
    {
        Rcpp::List settings;
        settings["in_state"] = false;
        settings["period"] = 1;
        settings["lobnd"] = 1.;
        settings["hibnd"] = 10.;
        return settings;
    }


    static arma::mat setX(const unsigned int &ntime, const unsigned int &period, const arma::mat &P)
    {
        arma::mat X(period, ntime + 1, arma::fill::zeros);
        X.at(0, 0) = 1.;
        for (unsigned int t = 0; t < ntime; t++)
        {
            X.col(t + 1) = P.t() * X.col(t);
        }
        return X;
    }

    Rcpp::List info()
    {
        Rcpp::List settings;
        settings["period"] = period;
        settings["in_state"] = in_state;
        settings["lobnd"] = lobnd;
        settings["hibnd"] = hibnd;
        settings["X"] = Rcpp::wrap(X);
        settings["val"] = Rcpp::wrap(val.t());
        return settings;
    }
};


/**
 * @brief Logistic regression of the probability of zero-inflation.
 * 
 */
class ZeroInflation
{
public:
    bool inflated = false;
    double intercept = 0.0; // intercept of the logistic regression
    double coef = 0.0; // AR coef of the logistic regression

    arma::vec beta;   // p x 1
    arma::mat X; // p x (ntime + 1), covariates of the logistic regression

    arma::vec prob; // (ntime + 1) x 1, z ~ Bernouli(prob)
    arma::vec z; // (ntime + 1) x 1, indicator, z = 0 for constant 0 and z = 1 for conditional NB/Poisson.

    ZeroInflation()
    {
        init_default();
        return;
    }

    ZeroInflation(const Rcpp::List &settings)
    {
        init(settings);
        return;
    }

    ZeroInflation(const double &intercept_, const double &coef_, const bool &inflated_)
    {
        intercept = intercept_;
        coef = coef_;
        inflated = inflated_;

        beta.reset();
        X.reset();
        return;
    }

    void init_default()
    {
        inflated = false;
        intercept = 0.;
        coef = 0.;

        beta.reset();
        X.reset();
        return;
    }


    void init(const Rcpp::List &settings)
    {
        Rcpp::List opts = settings;

        inflated = false;
        if (opts.containsElementNamed("inflated"))
        {
            inflated = Rcpp::as<bool>(opts["inflated"]);
        }

        intercept = 0.;
        if (opts.containsElementNamed("intercept"))
        {
            intercept = Rcpp::as<double>(opts["intercept"]);
        }

        coef = 0.;
        if (opts.containsElementNamed("coef"))
        {
            coef = Rcpp::as<double>(opts["coef"]);
        }

        beta.reset();
        X.reset();
        if (opts.containsElementNamed("beta"))
        {
            beta = Rcpp::as<arma::vec>(opts["beta"]); //  p x 1
        }

        return;
    } // init()


    void setX(const arma::mat &Xmat) // p x (ntime + 1)
    {
        inflated = true;
        X = Xmat;

        if (beta.is_empty())
        {
            beta = 10. * arma::randn(Xmat.n_rows);
        }
        else if (beta.n_elem != Xmat.n_rows)
        {
            throw std::invalid_argument("Zero: dimension of beta != number of covariates in X.");
        }

        return;
    } // setX()


    void setZ(const arma::vec &zvec, const unsigned int &ntime)
    {
        z.set_size(ntime + 1);
        z.ones();
        z.at(0) = 0.;
        z.tail(zvec.n_elem) = zvec;

        prob = z;

        double zsum = arma::accu(arma::abs(z));
        if ((zsum > 1. - EPS) && (zsum < (double)ntime))
        {
            inflated = true;
        }

        return;
    } // setZ()


    void init_Z(const arma::vec &y)
    {
        z = arma::conv_to<arma::vec>::from(y > EPS); // (nT + 1) x 1
        prob = z;
        z.at(0) = 0.;
        return;
    }


    void simulate(const unsigned int &ntime)
    {
        z.set_size(ntime + 1);
        z.ones();
        z.at(0) = 0.;
        if (prob.is_empty())
        {
            prob = z;
        }

        if (inflated)
        {
            for (unsigned int t = 1; t < (ntime + 1); t++)
            {
                double val = intercept;
                if (std::abs(z.at(t - 1) - 1.) < EPS)
                {
                    val += coef;
                }

                if (!X.is_empty())
                {
                    val += arma::dot(beta, X.col(t));
                }

                prob.at(t) = logistic(val);
                z.at(t) = (R::runif(0., 1.) < prob.at(t)) ? 1. : 0.;
            }
        }

        return;
    } // simulateZ()


    /**
     * @brief Estimate the latent indicator z_t for zero-inflation via forward-filtering backward-sampling.
     * 
     * @param y 
     * @param lambda 
     * @param obs_dist \
     * @param obs_par2 
     */
    void update_zt(
        const arma::vec &y,  // (nT + 1) x 1
        const arma::vec &lambda, // (nT + 1) x 1
        const std::string &obs_dist,
        const double &obs_par2
    ) // (nT + 1) x 1
    {
        arma::vec p01(y.n_elem, arma::fill::zeros); // p(z[t] = 1 | z[t-1] = 0, gamma)
        p01.at(0) = logistic(intercept);
        arma::vec p11(y.n_elem, arma::fill::zeros); // p(z[t] = 1 | z[t-1] = 1, gamma)
        p11.at(0) = logistic(intercept + coef);

        arma::vec prob_filter(y.n_elem, arma::fill::zeros); // p(z[t] = 1 | y[1:t], gamma)
        prob_filter.at(0) = p01.at(0);
        for (unsigned int t = 1; t < y.n_elem; t++)
        {
            double p0 = intercept;
            double p1 = intercept + coef;
            if (!X.is_empty())
            {
                double val = arma::dot(X.col(t), beta);
                p0 += val;
                p1 += val;
            }
            p01.at(t) = logistic(p0); // p(z[t] = 1 | z[t-1] = 0, gamma)
            p11.at(t) = logistic(p1); // p(z[t] = 1 | z[t-1] = 1, gamma)

            if (y.at(t) > EPS)
            {
                // If y[t] > 0
                // We must have p(z[t] = 1 | y[t] > 0) = 1
                prob_filter.at(t) = 1.;
            }
            else
            {
                double prob_yzero = ObsDist::loglike(
                    0., obs_dist, lambda.at(t), obs_par2, false);

                double pp1 = prob_filter.at(t - 1) * p11.at(t); // p(z[t-1] = 1) * p(z[t] = 1 | z[t-1] = 1)
                double pp0 = prob_filter.at(t - 1) * std::abs(1. - p11.at(t)); // p(z[t-1] = 1) * p(z[t] = 0 | z[t-1] = 1)

                pp1 += std::abs(1. - prob_filter.at(t - 1)) * p01.at(t); // p(z[t-1] = 0) * p(z[t] = 1 | z[t-1] = 0)
                pp0 += std::abs(1. - prob_filter.at(t - 1)) * std::abs(1. - p01.at(t)); // p(z[t-1] = 0) * p(z[t] = 0 | z[t-1] = 0)

                pp1 *= prob_yzero; // p(y[t] = 0 | z[t] = 1)
                prob_filter.at(t) = pp1 / (pp1 + pp0 + EPS);
            }
        } // Forward filtering loop

        z.at(y.n_elem - 1) = (R::runif(0., 1.) < prob_filter.at(y.n_elem - 1)) ? 1. : 0.;
        for (unsigned int t = y.n_elem - 2; t > 0; t--)
        {
            if (y.at(t) > EPS)
            {
                z.at(t) = 1.;
            }
            else
            {
                double p1 = z.at(t + 1) > EPS ? p11.at(t + 1) : (1. - p11.at(t + 1)); // p(z[t+1] | z[t] = 1)
                double prob_backward1 = prob_filter.at(t) * std::abs(p1);                        // p(z[t] = 1 | y[t] = 0) * p(z[t+1] | z[t] = 1)
                double p0 = z.at(t + 1) > EPS ? p01.at(t + 1) : (1. - p01.at(t + 1)); // p(z[t+1] | z[t] = 0)
                double prob_backward0 = std::abs(1. - prob_filter.at(t)) * std::abs(p0);         // p(z[t] = 0 | y[t] = 0) * p(z[t+1] | z[t] = 0)

                prob_backward1 = prob_backward1 / (prob_backward1 + prob_backward0 + EPS);
                z.at(t) = (R::runif(0., 1.) < prob_backward1) ? 1. : 0.;
            }
        }
    } // update_zt


    /**
     * @brief Log-likelihood of z[t] given the parameters.
     * 
     * @return double 
     */
    double log_likelihood()
    {
        double loglik = 0.;
        for (unsigned int t = 1; t < z.n_elem; t++)
        {
            double logit_p = intercept + (z.at(t - 1) > EPS ? coef : 0.);
            if (!X.is_empty())
            {
                logit_p += arma::dot(X.col(t), beta);
            }
            double prob_t = logistic(logit_p);

            if (z.at(t) > EPS)
            {
                loglik += std::log(prob_t + EPS);
            }
            else
            {
                loglik += std::log(1. - prob_t + EPS);
            }
        }

        return loglik;
    } // log_likelihood()


    /**
     * @brief Gradient of the log-likelihood with respect to the parameters.
     * 
     * @return arma::vec 
     */
    arma::vec dloglik_dparams()
    {
        arma::vec grad(2, arma::fill::zeros); // grad[0] = dloglik/dintercept, grad[1] = dloglik/dcoef

        for (unsigned int t = 1; t < z.n_elem; t++)
        {
            double logit_p = intercept + (z.at(t - 1) > EPS ? coef : 0.);
            if (!X.is_empty())
            {
                logit_p += arma::dot(X.col(t), beta);
            }
            double prob_t = logistic(logit_p);

            double dloglik_dprob = z.at(t) / prob_t - (1. - z.at(t)) / (1. - prob_t);
            double dprob_dlogit = prob_t * (1. - prob_t);
            double common_grad = dloglik_dprob * dprob_dlogit;
            grad.at(0) += common_grad; // dloglik/dintercept
            if (z.at(t - 1) > EPS)
            {
                grad.at(1) += common_grad; // dloglik/dcoef
            }
        }

        return grad; //, grad_coef;
    } // dloglik_dparams()


    double update_params(
        double &energy_diff,
        double &grad_norm_out,
        const double &prior_mean,
        const double &prior_sd,
        const double &leapfrog_step_size,
        const unsigned int &n_leapfrog,
        const arma::vec &mass_diag_est = arma::ones<arma::vec>(2)
    )
    {
        // Current parameters
        arma::vec params(2);
        params.at(0) = intercept;
        params.at(1) = coef;

        arma::vec params_old = params;

        arma::vec mass_diag = arma::max(mass_diag_est, arma::vec(2, arma::fill::value(1e-4)));
        arma::vec inv_mass = 1.0 / mass_diag;
        arma::vec sqrt_mass = arma::sqrt(mass_diag);

        // Compute initial energy
        double current_loglik = log_likelihood();
        double current_logprior = -0.5 * (
            std::pow((intercept - prior_mean) / prior_sd, 2.) +
            std::pow((coef - prior_mean) / prior_sd, 2.)
        );        
        double current_energy = - (current_loglik + current_logprior);

        arma::vec momentum = sqrt_mass % arma::randn<arma::vec>(2);
        double current_kinetic = 0.5 * arma::dot(momentum % inv_mass, momentum);

        // Make a half step for momentum
        arma::vec grad = dloglik_dparams();
        grad.at(0) += -(intercept - prior_mean) / (prior_sd * prior_sd);
        grad.at(1) += -(coef - prior_mean) / (prior_sd * prior_sd);
        grad_norm_out = arma::norm(grad, 2);

        momentum += 0.5 * leapfrog_step_size * grad;        

        // Alternate full steps for position and momentum
        for (unsigned int l = 0; l < n_leapfrog; l++)
        {
            // Full step for position
            params += leapfrog_step_size * (inv_mass % momentum);
            intercept = params.at(0);
            coef = params.at(1);

            grad = dloglik_dparams();
            grad.at(0) += -(params.at(0) - prior_mean) / (prior_sd * prior_sd);
            grad.at(1) += -(params.at(1) - prior_mean) / (prior_sd * prior_sd);

            // Full step for momentum, except at end of trajectory
            if (l != n_leapfrog - 1)
            {
                momentum += leapfrog_step_size * grad;
            }
        }

        // Make a half step for momentum
        momentum += 0.5 * leapfrog_step_size * grad;

        // Negate momentum to make proposal symmetric
        momentum = -momentum;

        // Compute proposed energy
        intercept = params.at(0);
        coef = params.at(1);
        double proposed_loglik = log_likelihood();
        double proposed_logprior = -0.5 * (
            std::pow((intercept - prior_mean) / prior_sd, 2.) +
            std::pow((coef - prior_mean) / prior_sd, 2.)
        );        
        double proposed_energy = - (proposed_loglik + proposed_logprior);
        double proposed_kinetic = 0.5 * arma::dot(momentum % inv_mass, momentum);

        double H_proposed = proposed_energy + proposed_kinetic;
        double H_current = current_energy + current_kinetic;
        energy_diff = H_proposed - H_current;

        if (!std::isfinite(H_current) || !std::isfinite(H_proposed) || std::abs(energy_diff) > 100.0)
        {
            // Reject and revert to previous parameters
            intercept = params_old.at(0);
            coef = params_old.at(1);
            return 0.0;
        }
        else
        {
            if (std::log(runif()) >= -energy_diff)
            {
                // Reject and revert to previous parameters
                intercept = params_old.at(0);
                coef = params_old.at(1);
            }

            // Return acceptance probability
            return std::min(1.0, std::exp(-energy_diff));
        }
    } // update_params()


    Rcpp::List run_mcmc(
        const arma::vec &y,
        const arma::vec &lambda,
        const double &obs_par2,
        const std::string &obs_dist = "nbinom",
        const unsigned int &n_iter = 5000,
        const unsigned int &n_burn = 1000,
        const unsigned int &n_thin = 1,
        const double &prior_mean = 0.0,
        const double &prior_sd = 10.0,
        const double &accept_prob = 0.65,
        const Rcpp::List &hmc_settings = Rcpp::List::create(
            Rcpp::Named("leapfrog_step_size") = 0.1,
            Rcpp::Named("nleapfrog") = 20,
            Rcpp::Named("T_target") = 2.0,
            Rcpp::Named("dual_averaging") = true,
            Rcpp::Named("diagnostics") = true,
            Rcpp::Named("verbose") = false
        )
    )
    {
        inflated = true;
        intercept = rnorm(0.0, 10.0);
        coef = rnorm(0.0, 10.0);

        init_Z(y);
        const unsigned int n_samples = (n_iter - n_burn) / n_thin;
        HMCOpts_1d hmc_opts(hmc_settings);
        DualAveraging_1d da_adapter(hmc_opts, accept_prob);
        HMCDiagnostics_1d hmc_diag(n_iter, n_burn, hmc_opts.dual_averaging);

        arma::vec intercept_samples(n_samples, arma::fill::zeros);
        arma::vec coef_samples(n_samples, arma::fill::zeros);
        arma::mat zt_samples(y.n_elem, n_samples, arma::fill::zeros);

        unsigned int sample_idx = 0;
        for (unsigned int iter = 0; iter < n_iter; iter++)
        {
            // Update z_t
            update_zt(y, lambda, obs_dist, obs_par2);

            // Update parameters via HMC
            {
                double energy_diff = 0.;
                double grad_norm = 0.;
                double accept_prob = update_params(
                    energy_diff,
                    grad_norm,
                    prior_mean,
                    prior_sd,
                    hmc_opts.leapfrog_step_size,
                    hmc_opts.nleapfrog);

                hmc_diag.accept_count += accept_prob;
                if (hmc_opts.diagnostics)
                {
                    const Eigen::Index iter_idx = static_cast<Eigen::Index>(iter);
                    hmc_diag.energy_diff(iter_idx) = energy_diff;
                    hmc_diag.grad_norm(iter_idx) = grad_norm;
                }

                // Adapt step size
                if (hmc_opts.dual_averaging)
                {
                    if (iter < n_burn)
                    {
                        hmc_opts.leapfrog_step_size = da_adapter.update_step_size(accept_prob);
                    }
                    else if (iter == n_burn)
                    {
                        da_adapter.finalize_leapfrog_step(
                            hmc_opts.leapfrog_step_size,
                            hmc_opts.nleapfrog,
                            hmc_opts.T_target);
                    }

                    if (hmc_opts.diagnostics)
                    {
                        const Eigen::Index iter_idx = static_cast<Eigen::Index>(iter);
                        hmc_diag.leapfrog_step_size_stored(iter_idx) = hmc_opts.leapfrog_step_size;
                        hmc_diag.nleapfrog_stored(iter_idx) = hmc_opts.nleapfrog;
                    }
                } // if dual_averaging
            } // end of HMC update

            // Store samples
            if (iter >= n_burn && ((iter - n_burn) % n_thin == 0))
            {
                intercept_samples.at(sample_idx) = intercept;
                coef_samples.at(sample_idx) = coef;
                zt_samples.col(sample_idx) = z;
                sample_idx++;
            }
        } // end of MCMC iterations

        Rcpp::List results = Rcpp::List::create(
            Rcpp::Named("intercept") = intercept_samples,
            Rcpp::Named("coef") = coef_samples,
            Rcpp::Named("zt") = zt_samples,
            Rcpp::Named("accept_rate") = (double)hmc_diag.accept_count / (double)n_iter,
            Rcpp::Named("leapfrog_step_size") = hmc_opts.leapfrog_step_size,
            Rcpp::Named("nleapfrog") = hmc_opts.nleapfrog
        );

        if (hmc_opts.diagnostics)
        {
            Rcpp::List diagnostics_results;
            diagnostics_results["energy_diff"] = hmc_diag.energy_diff;
            diagnostics_results["grad_norm"] = hmc_diag.grad_norm;

            if (hmc_opts.dual_averaging)
            {
                diagnostics_results["leapfrog_step_size"] = hmc_diag.leapfrog_step_size_stored;
                diagnostics_results["nleapfrog"] = hmc_diag.nleapfrog_stored;
            }

            results["diagnostics"] = diagnostics_results;
        }

        return results;
    }
}; // class ZeroInflation


#endif
