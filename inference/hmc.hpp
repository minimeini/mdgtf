#pragma once
#ifndef HMC_HPP
#define HMC_HPP

#include <iostream>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <vector>
#include <RcppArmadillo.h>

// [[Rcpp::plugins(cpp17)]]
// [[Rcpp::depends(RcppArmadillo)]]

struct HMCOpts_1d
{
    std::vector<std::string> params_selected;
    bool dual_averaging = false;
    bool diagnostics = true;
    bool verbose = false;
    unsigned int nleapfrog = 20;
    double leapfrog_step_size = 0.1;
    double T_target = 2.0; // integration time T = n_leapfrog * epsilon ~= 1-2 (rough heuristic). Larger T gives better exploration but higher cost.

    HMCOpts_1d() = default;
    HMCOpts_1d(const Rcpp::List &opts)
    {
        nleapfrog = 10;
        if (opts.containsElementNamed("nleapfrog"))
        {
            nleapfrog = Rcpp::as<unsigned int>(opts["nleapfrog"]);
        }

        leapfrog_step_size = 0.01;
        if (opts.containsElementNamed("leapfrog_step_size"))
        {
            leapfrog_step_size = Rcpp::as<double>(opts["leapfrog_step_size"]);
        }

        dual_averaging = false;
        if (opts.containsElementNamed("dual_averaging"))
        {
            dual_averaging = Rcpp::as<bool>(opts["dual_averaging"]);
        }

        diagnostics = true;
        if (opts.containsElementNamed("diagnostics"))
        {
            diagnostics = Rcpp::as<bool>(opts["diagnostics"]);
        }

        verbose = false;
        if (opts.containsElementNamed("verbose"))
        {
            verbose = Rcpp::as<bool>(opts["verbose"]);
        }

        T_target = 1.0;
        if (opts.containsElementNamed("T_target"))
        {
            T_target = Rcpp::as<double>(opts["T_target"]);
        }
    }
};


struct HMCOpts_2d
{
    std::vector<std::string> params_selected;
    bool dual_averaging = false;
    bool diagnostics = true;
    bool verbose = false;
    double T_target = 2.0; // integration time T = n_leapfrog * epsilon ~= 1-2 (rough heuristic). Larger T gives better exploration but higher cost.

    unsigned int nleapfrog_init = 20;
    arma::uvec nleapfrog;
    double leapfrog_step_size_init = 0.1;
    arma::vec leapfrog_step_size;

    HMCOpts_2d() = default;
    HMCOpts_2d(const Rcpp::List &opts)
    {
        nleapfrog_init = 10;
        if (opts.containsElementNamed("nleapfrog"))
        {
            nleapfrog_init = Rcpp::as<unsigned int>(opts["nleapfrog"]);
        }

        leapfrog_step_size_init = 0.01;
        if (opts.containsElementNamed("leapfrog_step_size"))
        {
            leapfrog_step_size_init = Rcpp::as<double>(opts["leapfrog_step_size"]);
        }

        dual_averaging = false;
        if (opts.containsElementNamed("dual_averaging"))
        {
            dual_averaging = Rcpp::as<bool>(opts["dual_averaging"]);
        }

        diagnostics = true;
        if (opts.containsElementNamed("diagnostics"))
        {
            diagnostics = Rcpp::as<bool>(opts["diagnostics"]);
        }

        verbose = false;
        if (opts.containsElementNamed("verbose"))
        {
            verbose = Rcpp::as<bool>(opts["verbose"]);
        }

        T_target = 1.0;
        if (opts.containsElementNamed("T_target"))
        {
            T_target = Rcpp::as<double>(opts["T_target"]);
        }
    }
};


struct DualAveraging_1d
{
    double target_accept = 0.75;
    double mu_da;
    double log_eps_bar;
    double log_eps;
    double h_bar = 0.0;
    double gamma_da = 0.05;
    double t0_da = 10.0;
    double kappa_da = 0.75;
    unsigned int adapt_count = 0;
    unsigned int min_leaps = 3;
    unsigned int max_leaps = 128;

    DualAveraging_1d() = default;
    DualAveraging_1d(const HMCOpts_1d &hmc_opts, const double &target_accept_rate = 0.75)
    {
        target_accept = target_accept_rate;

        mu_da = std::log(10 * hmc_opts.leapfrog_step_size);
        log_eps = std::log(hmc_opts.leapfrog_step_size);
        log_eps_bar = log_eps;
        
        h_bar = 0.0;
        gamma_da = 0.1;
        t0_da = 10.0;
        kappa_da = 0.75;

        adapt_count = 0;

        min_leaps = 3;
        max_leaps = 128;
        return;
    }

    double update_step_size(const double &accept_prob)
    {
        adapt_count++;
        double t = (double)adapt_count;
        h_bar = (1.0 - 1.0 / (t + t0_da)) * h_bar + (1.0 / (t + t0_da)) * (target_accept - accept_prob);
        log_eps = mu_da - (std::sqrt(t) / gamma_da) * h_bar;

        auto clamp_eps = [](double x)
        { return std::max(1e-3, std::min(0.1, x)); };
        double leapfrog_step_size = clamp_eps(std::exp(log_eps));
        double w = std::pow(t, -kappa_da);
        log_eps_bar = w * std::log(leapfrog_step_size) + (1.0 - w) * log_eps_bar;
        return leapfrog_step_size;
    }

    void finalize_leapfrog_step(double &step_size, unsigned int &nleapfrog, const double &T_target)
    {
        auto clamp_eps = [](double x)
        { return std::max(1e-3, std::min(0.1, x)); };
        step_size = clamp_eps(std::exp(log_eps_bar));
        unsigned nlf = (unsigned)std::lround(T_target / step_size);
        nleapfrog = std::max(min_leaps, std::min(max_leaps, nlf));
        return;
    }
};


struct DualAveraging_2d
{
    double target_accept = 0.75;
    arma::vec mu_da;
    arma::vec log_eps_bar;
    arma::vec log_eps;
    arma::vec h_bar;
    arma::vec gamma_da;
    arma::vec t0_da;
    arma::vec kappa_da;
    arma::uvec adapt_count;
    unsigned int min_leaps = 3;
    unsigned int max_leaps = 128;

    DualAveraging_2d()
    {
        mu_da = std::log(10 * 0.01) * arma::ones(1);
        log_eps = std::log(0.01) * arma::ones(1);
        log_eps_bar = log_eps;
        h_bar = arma::zeros(1);
        gamma_da = 0.05 * arma::ones(1);
        t0_da = 10.0 * arma::ones(1);
        kappa_da = 0.75 * arma::ones(1);
        adapt_count = arma::zeros<arma::uvec>(1);
        min_leaps = 3;
        max_leaps = 128;
        return;
    }

    DualAveraging_2d(
        const HMCOpts_2d &hmc_opts, 
        const double &target_accept_rate = 0.75
    )
    {
        target_accept = target_accept_rate;

        mu_da = std::log(10 * hmc_opts.leapfrog_step_size_init) * arma::ones(hmc_opts.params_selected.size());
        log_eps = std::log(hmc_opts.leapfrog_step_size_init) * arma::ones(hmc_opts.params_selected.size());
        log_eps_bar = log_eps;

        h_bar = arma::zeros(hmc_opts.params_selected.size());
        gamma_da = 0.05 * arma::ones(hmc_opts.params_selected.size());
        t0_da = 10.0 * arma::ones(hmc_opts.params_selected.size());
        kappa_da = 0.75 * arma::ones(hmc_opts.params_selected.size());

        adapt_count = arma::zeros<arma::uvec>(hmc_opts.params_selected.size());

        min_leaps = 3;
        max_leaps = 128;
        return;
    }

    double update_step_size(const unsigned int &s, const double &accept_prob)
    {
        adapt_count++;
        double t = (double)adapt_count.at(s);
        h_bar.at(s) = (1.0 - 1.0 / (t + t0_da.at(s))) * h_bar.at(s) + (1.0 / (t + t0_da.at(s))) * (target_accept - accept_prob);
        log_eps.at(s) = mu_da.at(s) - (std::sqrt(t) / gamma_da.at(s)) * h_bar.at(s);

        auto clamp_eps = [](double x)
        { return std::max(1e-3, std::min(0.1, x)); };
        double leapfrog_step_size = clamp_eps(std::exp(log_eps.at(s)));
        double w = std::pow(t, -kappa_da.at(s));
        log_eps_bar.at(s) = w * std::log(leapfrog_step_size) + (1.0 - w) * log_eps_bar.at(s);
        return leapfrog_step_size;
    }

    void finalize_leapfrog_step(
        double &step_size, 
        unsigned int &nleapfrog, 
        const unsigned int &s, 
        const double &T_target
    )
    {
        auto clamp_eps = [](double x)
        { return std::max(1e-3, std::min(0.1, x)); };
        step_size = clamp_eps(std::exp(log_eps_bar.at(s)));
        unsigned nlf = (unsigned)std::lround(T_target / step_size);
        nleapfrog = std::max(min_leaps, std::min(max_leaps, nlf));
        return;
    }
};



struct HMCDiagnostics_1d
{
    double accept_count = 0.0;
    arma::vec energy_diff;
    arma::vec grad_norm;
    arma::vec leapfrog_step_size_stored;
    arma::vec nleapfrog_stored;
    HMCDiagnostics_1d() = default;
    HMCDiagnostics_1d(
        const unsigned int &niter,
        const unsigned int &nburnin = 1,
        const bool &dual_averaging = false
    )
    {
        energy_diff = arma::vec(niter, arma::fill::zeros);
        grad_norm = arma::vec(niter, arma::fill::zeros);

        if (dual_averaging)
        {
            leapfrog_step_size_stored = arma::vec(nburnin + 1, arma::fill::zeros);
            nleapfrog_stored = arma::vec(nburnin + 1, arma::fill::zeros);
        }
        return;
    }
};


struct HMCDiagnostics_2d
{
    arma::vec accept_count;
    arma::mat energy_diff;
    arma::mat grad_norm;
    arma::mat leapfrog_step_size_stored;
    arma::mat nleapfrog_stored;

    HMCDiagnostics_2d() = default;
    HMCDiagnostics_2d(
        const unsigned int &nS,
        const unsigned int &niter,
        const unsigned int &nburnin = 1,
        const bool &dual_averaging = false
    )
    {
        accept_count = arma::vec(nS, arma::fill::zeros);
        energy_diff = arma::mat(nS, niter, arma::fill::zeros);
        grad_norm = arma::mat(nS, niter, arma::fill::zeros);

        if (dual_averaging)
        {
            leapfrog_step_size_stored = arma::mat(nS, nburnin + 1, arma::fill::zeros);
            nleapfrog_stored = arma::mat(nS, nburnin + 1, arma::fill::zeros);
        }
        return;
    }
};


#endif