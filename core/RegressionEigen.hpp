#pragma once
#ifndef REGRESSION_EIGEN_H
#define REGRESSION_EIGEN_H

#include <RcppEigen.h>
#include <Eigen/Dense>
#include "../utils/utils.h"
#include "../inference/hmc.hpp"
#include "../core/ObsDist.hpp"

// [[Rcpp::plugins(cpp17)]]
// [[Rcpp::depends(RcppEigen)]]


/**
 * @brief Logistic regression of the probability of zero-inflation (Eigen version).
 */
class ZeroInflation
{
public:
    bool inflated = false;
    double intercept = 0.0; // intercept of the logistic regression
    double coef = 0.0;      // AR coef of the logistic regression

    Eigen::VectorXd beta;   // p x 1
    Eigen::MatrixXd X;      // p x (ntime + 1), covariates of the logistic regression

    Eigen::VectorXd prob; // (ntime + 1) x 1, z ~ Bernouli(prob)
    Eigen::VectorXd z;    // (ntime + 1) x 1, indicator, z = 0 for constant 0 and z = 1 for conditional NB/Poisson.

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

    ZeroInflation(
        const double &intercept_, 
        const double &coef_, 
        const bool &inflated_
    )
    {
        intercept = intercept_;
        coef = coef_;
        inflated = inflated_;

        beta.resize(0);
        X.resize(0, 0);
        return;
    }

    void init_default()
    {
        inflated = false;
        intercept = 0.;
        coef = 0.;

        beta.resize(0);
        X.resize(0, 0);
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

        beta.resize(0);
        X.resize(0, 0);
        if (opts.containsElementNamed("beta"))
        {
            beta = Rcpp::as<Eigen::VectorXd>(opts["beta"]); //  p x 1
        }

        return;
    } // init()

    void setX(const Eigen::MatrixXd &Xmat) // p x (ntime + 1)
    {
        inflated = true;
        X = Xmat;

        if (beta.size() == 0)
        {
            beta = 10. * Eigen::VectorXd::NullaryExpr(Xmat.rows(), []() { return R::rnorm(0.0, 1.0); });
        }
        else if (beta.size() != Xmat.rows())
        {
            throw std::invalid_argument("Zero: dimension of beta != number of covariates in X.");
        }

        return;
    } // setX()

    void setZ(const Eigen::VectorXd &zvec, const unsigned int &ntime)
    {
        z.resize(ntime + 1);
        z.setOnes();
        z(0) = 0.;
        const Eigen::Index n_tail = std::min(static_cast<Eigen::Index>(zvec.size()), static_cast<Eigen::Index>(ntime));
        if (n_tail > 0)
        {
            z.tail(n_tail) = zvec.tail(n_tail);
        }

        prob = z;

        double zsum = z.cwiseAbs().sum();
        if ((zsum > 1. - EPS) && (zsum < static_cast<double>(ntime)))
        {
            inflated = true;
        }

        return;
    } // setZ()

    void init_Z(const Eigen::VectorXd &y)
    {
        z = (y.array() > EPS).cast<double>(); // (nT + 1) x 1
        prob = z;
        z(0) = 0.;
        return;
    }

    void simulate(const unsigned int &ntime)
    {
        z.resize(ntime + 1);
        z.setOnes();
        z(0) = 0.;
        if (prob.size() == 0)
        {
            prob = z;
        }

        if (inflated)
        {
            for (unsigned int t = 1; t < (ntime + 1); t++)
            {
                double val = intercept;
                if (std::abs(z(t - 1) - 1.) < EPS)
                {
                    val += coef;
                }

                if (X.size() > 0)
                {
                    val += beta.dot(X.col(static_cast<Eigen::Index>(t)));
                }

                prob(t) = logistic(val);
                z(t) = (R::runif(0., 1.) < prob(t)) ? 1. : 0.;
            }
        }

        return;
    } // simulateZ()

    void update_zt(
        const Eigen::VectorXd &y,      // (nT + 1) x 1
        const Eigen::VectorXd &lambda, // (nT + 1) x 1
        const std::string &obs_dist,
        const double &obs_par2)
    {
        const Eigen::Index n = y.size();
        Eigen::VectorXd p01 = Eigen::VectorXd::Zero(n); // p(z[t] = 1 | z[t-1] = 0, gamma)
        p01(0) = logistic(intercept);
        Eigen::VectorXd p11 = Eigen::VectorXd::Zero(n); // p(z[t] = 1 | z[t-1] = 1, gamma)
        p11(0) = logistic(intercept + coef);

        Eigen::VectorXd prob_filter = Eigen::VectorXd::Zero(n); // p(z[t] = 1 | y[1:t], gamma)
        prob_filter(0) = p01(0);
        for (Eigen::Index t = 1; t < n; t++)
        {
            double p0 = intercept;
            double p1 = intercept + coef;
            if (X.size() > 0)
            {
                double val = X.col(t).dot(beta);
                p0 += val;
                p1 += val;
            }
            p01(t) = logistic(p0); // p(z[t] = 1 | z[t-1] = 0, gamma)
            p11(t) = logistic(p1); // p(z[t] = 1 | z[t-1] = 1, gamma)

            if (y(t) > EPS)
            {
                prob_filter(t) = 1.;
            }
            else
            {
                double prob_yzero = ObsDist::loglike(0., obs_dist, lambda(t), obs_par2, false);

                double pp1 = prob_filter(t - 1) * p11(t);
                double pp0 = prob_filter(t - 1) * std::abs(1. - p11(t));

                pp1 += std::abs(1. - prob_filter(t - 1)) * p01(t);
                pp0 += std::abs(1. - prob_filter(t - 1)) * std::abs(1. - p01(t));

                pp1 *= prob_yzero;
                prob_filter(t) = pp1 / (pp1 + pp0 + EPS);
            }
        } // Forward filtering loop

        z(n - 1) = (R::runif(0., 1.) < prob_filter(n - 1)) ? 1. : 0.;
        for (Eigen::Index t = n - 2; t > 0; t--)
        {
            if (y(t) > EPS)
            {
                z(t) = 1.;
            }
            else
            {
                double p1 = z(t + 1) > EPS ? p11(t + 1) : (1. - p11(t + 1)); // p(z[t+1] | z[t] = 1)
                double prob_backward1 = prob_filter(t) * std::abs(p1);
                double p0 = z(t + 1) > EPS ? p01(t + 1) : (1. - p01(t + 1)); // p(z[t+1] | z[t] = 0)
                double prob_backward0 = std::abs(1. - prob_filter(t)) * std::abs(p0);

                prob_backward1 = prob_backward1 / (prob_backward1 + prob_backward0 + EPS);
                z(t) = (R::runif(0., 1.) < prob_backward1) ? 1. : 0.;
            }
        }
    } // update_zt

    double log_likelihood()
    {
        double loglik = 0.;
        for (Eigen::Index t = 1; t < z.size(); t++)
        {
            double logit_p = intercept + (z(t - 1) > EPS ? coef : 0.);
            if (X.size() > 0)
            {
                logit_p += X.col(t).dot(beta);
            }
            double prob_t = logistic(logit_p);

            if (z(t) > EPS)
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

    Eigen::VectorXd dloglik_dparams()
    {
        Eigen::VectorXd grad = Eigen::VectorXd::Zero(2); // grad[0] = dloglik/dintercept, grad[1] = dloglik/dcoef

        for (Eigen::Index t = 1; t < z.size(); t++)
        {
            double logit_p = intercept + (z(t - 1) > EPS ? coef : 0.);
            if (X.size() > 0)
            {
                logit_p += X.col(t).dot(beta);
            }
            double prob_t = logistic(logit_p);

            double dloglik_dprob = z(t) / prob_t - (1. - z(t)) / (1. - prob_t);
            double dprob_dlogit = prob_t * (1. - prob_t);
            double common_grad = dloglik_dprob * dprob_dlogit;
            grad(0) += common_grad; // dloglik/dintercept
            if (z(t - 1) > EPS)
            {
                grad(1) += common_grad; // dloglik/dcoef
            }
        }

        return grad;
    } // dloglik_dparams()

    double update_params(
        double &energy_diff,
        double &grad_norm_out,
        const double &prior_mean,
        const double &prior_sd,
        const double &leapfrog_step_size,
        const unsigned int &n_leapfrog,
        const Eigen::VectorXd &mass_diag_est = Eigen::VectorXd::Constant(2, 1.0))
    {
        Eigen::VectorXd params(2);
        params(0) = intercept;
        params(1) = coef;

        Eigen::VectorXd params_old = params;

        Eigen::VectorXd mass_diag = mass_diag_est.array().max(1e-4);
        Eigen::VectorXd inv_mass = mass_diag.cwiseInverse();
        Eigen::VectorXd sqrt_mass = mass_diag.array().sqrt();

        double current_loglik = log_likelihood();
        double current_logprior = -0.5 * (
            std::pow((intercept - prior_mean) / prior_sd, 2.) +
            std::pow((coef - prior_mean) / prior_sd, 2.));
        double current_energy = -(current_loglik + current_logprior);

        Eigen::VectorXd momentum = sqrt_mass.array() * Eigen::VectorXd::NullaryExpr(2, []() { return R::rnorm(0.0, 1.0); }).array();
        double current_kinetic = 0.5 * momentum.cwiseProduct(inv_mass).dot(momentum);

        Eigen::VectorXd grad = dloglik_dparams();
        grad(0) += -(intercept - prior_mean) / (prior_sd * prior_sd);
        grad(1) += -(coef - prior_mean) / (prior_sd * prior_sd);
        grad_norm_out = grad.norm();

        momentum += 0.5 * leapfrog_step_size * grad;

        for (unsigned int l = 0; l < n_leapfrog; l++)
        {
            params += leapfrog_step_size * (inv_mass.array() * momentum.array()).matrix();
            intercept = params(0);
            coef = params(1);

            grad = dloglik_dparams();
            grad(0) += -(params(0) - prior_mean) / (prior_sd * prior_sd);
            grad(1) += -(params(1) - prior_mean) / (prior_sd * prior_sd);

            if (l != n_leapfrog - 1)
            {
                momentum += leapfrog_step_size * grad;
            }
        }

        momentum += 0.5 * leapfrog_step_size * grad;
        momentum = -momentum;

        intercept = params(0);
        coef = params(1);
        double proposed_loglik = log_likelihood();
        double proposed_logprior = -0.5 * (
            std::pow((intercept - prior_mean) / prior_sd, 2.) +
            std::pow((coef - prior_mean) / prior_sd, 2.));
        double proposed_energy = -(proposed_loglik + proposed_logprior);
        double proposed_kinetic = 0.5 * momentum.cwiseProduct(inv_mass).dot(momentum);

        double H_proposed = proposed_energy + proposed_kinetic;
        double H_current = current_energy + current_kinetic;
        energy_diff = H_proposed - H_current;

        if (!std::isfinite(H_current) || !std::isfinite(H_proposed) || std::abs(energy_diff) > 100.0)
        {
            intercept = params_old(0);
            coef = params_old(1);
            return 0.0;
        }
        else
        {
            if (std::log(runif()) >= -energy_diff)
            {
                intercept = params_old(0);
                coef = params_old(1);
            }

            return std::min(1.0, std::exp(-energy_diff));
        }
    } // update_params()

    Rcpp::List run_mcmc(
        const Eigen::VectorXd &y,
        const Eigen::VectorXd &lambda,
        const double &obs_par2,
        const std::string &obs_dist = "nbinom",
        const Eigen::Index &n_iter = 5000,
        const Eigen::Index &n_burn = 1000,
        const Eigen::Index &n_thin = 1,
        const double &prior_mean = 0.0,
        const double &prior_sd = 10.0,
        const double &accept_prob = 0.65,
        const Rcpp::List &hmc_settings = Rcpp::List::create(
            Rcpp::Named("leapfrog_step_size") = 0.1,
            Rcpp::Named("nleapfrog") = 20,
            Rcpp::Named("T_target") = 2.0,
            Rcpp::Named("dual_averaging") = true,
            Rcpp::Named("diagnostics") = true,
            Rcpp::Named("verbose") = false))
    {
        inflated = true;
        intercept = rnorm(0.0, 10.0);
        coef = rnorm(0.0, 10.0);

        init_Z(y);
        const Eigen::Index n_samples = (n_iter - n_burn) / n_thin;
        HMCOpts_1d hmc_opts(hmc_settings);
        DualAveraging_1d da_adapter(hmc_opts, accept_prob);
        HMCDiagnostics_1d hmc_diag(n_iter, n_burn, hmc_opts.dual_averaging);

        Eigen::VectorXd intercept_samples = Eigen::VectorXd::Zero(n_samples);
        Eigen::VectorXd coef_samples = Eigen::VectorXd::Zero(n_samples);
        Eigen::MatrixXd zt_samples = Eigen::MatrixXd::Zero(y.size(), n_samples);

        Eigen::Index sample_idx = 0;
        for (Eigen::Index iter = 0; iter < n_iter; iter++)
        {
            update_zt(y, lambda, obs_dist, obs_par2);

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
                    hmc_diag.energy_diff(iter) = energy_diff;
                    hmc_diag.grad_norm(iter) = grad_norm;
                }

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
                        hmc_diag.leapfrog_step_size_stored(iter) = hmc_opts.leapfrog_step_size;
                        hmc_diag.nleapfrog_stored(iter) = hmc_opts.nleapfrog;
                    }
                } // if dual_averaging
            } // end of HMC update

            if (iter >= n_burn && ((iter - n_burn) % n_thin == 0))
            {
                intercept_samples(sample_idx) = intercept;
                coef_samples(sample_idx) = coef;
                zt_samples.col(sample_idx) = z;
                sample_idx++;
            }
        } // end of MCMC iterations

        Rcpp::List results = Rcpp::List::create(
            Rcpp::Named("intercept") = intercept_samples,
            Rcpp::Named("coef") = coef_samples,
            Rcpp::Named("zt") = zt_samples,
            Rcpp::Named("accept_rate") = static_cast<double>(hmc_diag.accept_count) / static_cast<double>(n_iter),
            Rcpp::Named("leapfrog_step_size") = hmc_opts.leapfrog_step_size,
            Rcpp::Named("nleapfrog") = hmc_opts.nleapfrog);

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
}; // class ZeroInflation (Eigen)

#endif
