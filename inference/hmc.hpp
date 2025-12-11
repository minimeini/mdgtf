#pragma once
#ifndef HMC_HPP
#define HMC_HPP

#include <iostream>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <vector>
#include <RcppEigen.h>
#include <Eigen/Dense>

// [[Rcpp::plugins(cpp17)]]
// [[Rcpp::depends(RcppEigen)]]


struct MassAdapter {
    Eigen::VectorXd mean;
    Eigen::VectorXd M2;  // Sum of squared deviations
    unsigned int count = 0;
    unsigned int window_size = 100;
    
    void update(const Eigen::VectorXd& log_params) {
        count++;
        Eigen::VectorXd delta = log_params - mean;
        mean += delta / count;
        M2 += delta.cwiseProduct(log_params - mean);
    }
    
    Eigen::VectorXd get_mass_diag() {
        if (count < 10) return Eigen::VectorXd::Ones(mean.size());
        Eigen::VectorXd var = M2 / (count - 1);

        // setting the momentum covariance to approximate the precision of the parameters
        return (1.0 / var.array()).cwiseMax(0.1).cwiseMin(100.0);
    }
    
    void reset_window() {
        // Optionally reset for new adaptation window
        count = 0;
        M2.setZero();
    }
};


struct HMCOpts_1d
{
    std::vector<std::string> params_selected;
    bool dual_averaging = true;
    bool diagnostics = true;
    bool verbose = false;
    unsigned int nleapfrog = 20;
    double leapfrog_step_size = 0.1;
    double T_target = 2.0; // integration time T = n_leapfrog * epsilon ~= 1-2 (rough heuristic). Larger T gives better exploration but higher cost.

    HMCOpts_1d() = default;
    HMCOpts_1d(const Rcpp::List &opts)
    {
        nleapfrog = 20;
        if (opts.containsElementNamed("nleapfrog"))
        {
            nleapfrog = Rcpp::as<unsigned int>(opts["nleapfrog"]);
        }

        leapfrog_step_size = 0.1;
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

        T_target = 2.0;
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
    Eigen::MatrixXd mass_diag_est; // nvar x nS, estimated mass matrix diagonal

    unsigned int nleapfrog_init = 20;
    Eigen::VectorXi nleapfrog;
    double leapfrog_step_size_init = 0.1;
    Eigen::VectorXd leapfrog_step_size;

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

        if (opts.containsElementNamed("mass_diag_est"))
        {
            mass_diag_est = Rcpp::as<Eigen::MatrixXd>(opts["mass_diag_est"]);
        }
    } // end constructor


    void set_size(const unsigned int &nS)
    {
        Eigen::Index idx_nS = static_cast<Eigen::Index>(nS);
        nleapfrog = Eigen::VectorXi::Constant(idx_nS, static_cast<int>(nleapfrog_init));
        leapfrog_step_size = Eigen::VectorXd::Constant(idx_nS, leapfrog_step_size_init);
        if (mass_diag_est.size() == 0)
        {
            mass_diag_est = Eigen::MatrixXd::Ones(static_cast<Eigen::Index>(params_selected.size()), idx_nS);
        }
        return;
    }
};


struct DualAveraging_1d
{
    double target_accept = 0.75;
    double mu_da = std::log(10 * 0.1);
    double log_eps_bar = std::log(0.1);
    double log_eps = std::log(0.1);
    double h_bar = 0.0;
    double gamma_da = 0.1;
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
        { return std::max(1e-3, std::min(1.0, x)); };
        double leapfrog_step_size = clamp_eps(std::exp(log_eps));
        double w = std::pow(t, -kappa_da);
        log_eps_bar = w * std::log(leapfrog_step_size) + (1.0 - w) * log_eps_bar;
        return leapfrog_step_size;
    }

    void finalize_leapfrog_step(double &step_size, unsigned int &nleapfrog, const double &T_target)
    {
        auto clamp_eps = [](double x)
        { return std::max(1e-3, std::min(1.0, x)); };
        step_size = clamp_eps(std::exp(log_eps_bar));
        unsigned nlf = (unsigned)std::lround(T_target / step_size);
        nleapfrog = std::max(min_leaps, std::min(max_leaps, nlf));
        return;
    }
};


struct DualAveraging_2d
{
    double target_accept = 0.75;
    Eigen::VectorXd mu_da;
    Eigen::VectorXd log_eps_bar;
    Eigen::VectorXd log_eps;
    Eigen::VectorXd h_bar;
    Eigen::VectorXd gamma_da;
    Eigen::VectorXd t0_da;
    Eigen::VectorXd kappa_da;
    Eigen::VectorXi adapt_count;
    unsigned int min_leaps = 3;
    unsigned int max_leaps = 128;

    DualAveraging_2d()
    {
        mu_da = Eigen::VectorXd::Constant(1, std::log(10 * 0.01));
        log_eps = Eigen::VectorXd::Constant(1, std::log(0.01));
        log_eps_bar = log_eps;
        h_bar = Eigen::VectorXd::Zero(1);
        gamma_da = Eigen::VectorXd::Constant(1, 0.05);
        t0_da = Eigen::VectorXd::Constant(1, 10.0);
        kappa_da = Eigen::VectorXd::Constant(1, 0.75);
        adapt_count = Eigen::VectorXi::Zero(1);
        min_leaps = 3;
        max_leaps = 128;
        return;
    }

    DualAveraging_2d(
        const unsigned int &nS,
        const double &leapfrog_step_size_init, 
        const double &target_accept_rate = 0.75
    )
    {
        target_accept = target_accept_rate;
        Eigen::Index idx_nS = static_cast<Eigen::Index>(nS);

        mu_da = Eigen::VectorXd::Constant(idx_nS, std::log(10 * leapfrog_step_size_init));
        log_eps = Eigen::VectorXd::Constant(idx_nS, std::log(leapfrog_step_size_init));
        log_eps_bar = log_eps;

        h_bar = Eigen::VectorXd::Zero(idx_nS);
        gamma_da = Eigen::VectorXd::Constant(idx_nS, 0.05);
        t0_da = Eigen::VectorXd::Constant(idx_nS, 10.0);
        kappa_da = Eigen::VectorXd::Constant(idx_nS, 0.75);

        adapt_count = Eigen::VectorXi::Zero(idx_nS);

        min_leaps = 3;
        max_leaps = 128;
        return;
    }

    double update_step_size(const unsigned int &s, const double &accept_prob)
    {
        Eigen::Index idx = static_cast<Eigen::Index>(s);
        adapt_count(idx)++;
        double t = static_cast<double>(adapt_count(idx));
        h_bar(idx) = (1.0 - 1.0 / (t + t0_da(idx))) * h_bar(idx) + (1.0 / (t + t0_da(idx))) * (target_accept - accept_prob);
        log_eps(idx) = mu_da(idx) - (std::sqrt(t) / gamma_da(idx)) * h_bar(idx);

        auto clamp_eps = [](double x)
        { return std::max(1e-3, std::min(1.0, x)); };
        double leapfrog_step_size = clamp_eps(std::exp(log_eps(idx)));
        double w = std::pow(t, -kappa_da(idx));
        log_eps_bar(idx) = w * std::log(leapfrog_step_size) + (1.0 - w) * log_eps_bar(idx);
        return leapfrog_step_size;
    }

    void finalize_leapfrog_step(
        double &step_size, 
        int &nleapfrog, 
        const unsigned int &s, 
        const double &T_target
    )
    {
        auto clamp_eps = [](double x)
        { return std::max(1e-3, std::min(1.0, x)); };
        step_size = clamp_eps(std::exp(log_eps_bar(static_cast<Eigen::Index>(s))));
        unsigned nlf = (unsigned)std::lround(T_target / step_size);
        nleapfrog = static_cast<int>(std::max(min_leaps, std::min(max_leaps, nlf)));
        return;
    }
};



struct HMCDiagnostics_1d
{
    double accept_count = 0.0;
    Eigen::VectorXd energy_diff;
    Eigen::VectorXd grad_norm;
    Eigen::VectorXd leapfrog_step_size_stored;
    Eigen::VectorXd nleapfrog_stored;
    HMCDiagnostics_1d() = default;
    HMCDiagnostics_1d(
        const unsigned int &niter,
        const unsigned int &nburnin = 1,
        const bool &dual_averaging = false
    )
    {
        energy_diff = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(niter));
        grad_norm = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(niter));

        if (dual_averaging)
        {
            leapfrog_step_size_stored = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(nburnin + 1));
            nleapfrog_stored = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(nburnin + 1));
        }
        return;
    }


    Rcpp::List to_list() const
    {
        Rcpp::List diagnostics_results = Rcpp::List::create(
            Rcpp::Named("energy_diff") = energy_diff,
            Rcpp::Named("grad_norm") = grad_norm
        );

        if (leapfrog_step_size_stored.size() > 0 && std::abs(leapfrog_step_size_stored.sum()) > EPS)
        {
            diagnostics_results["leapfrog_step_size"] = leapfrog_step_size_stored;
            diagnostics_results["nleapfrog"] = nleapfrog_stored;
        }

        return diagnostics_results;
    }
};


struct HMCDiagnostics_2d
{
    Eigen::VectorXd accept_count;
    Eigen::MatrixXd energy_diff;
    Eigen::MatrixXd grad_norm;
    Eigen::MatrixXd leapfrog_step_size_stored;
    Eigen::MatrixXd nleapfrog_stored;

    HMCDiagnostics_2d() = default;
    HMCDiagnostics_2d(
        const unsigned int &nS,
        const unsigned int &niter,
        const unsigned int &nburnin = 1,
        const bool &dual_averaging = false
    )
    {
        Eigen::Index idx_nS = static_cast<Eigen::Index>(nS);
        Eigen::Index idx_niter = static_cast<Eigen::Index>(niter);

        accept_count = Eigen::VectorXd::Zero(idx_nS);
        energy_diff = Eigen::MatrixXd::Zero(idx_nS, idx_niter);
        grad_norm = Eigen::MatrixXd::Zero(idx_nS, idx_niter);

        if (dual_averaging)
        {
            leapfrog_step_size_stored = Eigen::MatrixXd::Zero(idx_nS, static_cast<Eigen::Index>(nburnin + 1));
            nleapfrog_stored = Eigen::MatrixXd::Zero(idx_nS, static_cast<Eigen::Index>(nburnin + 1));
        }
        return;
    }
};


#endif
