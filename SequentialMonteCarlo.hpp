#ifndef _SEQUENTIALMONTECARLO_H
#define _SEQUENTIALMONTECARLO_H

#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <algorithm>
#include <RcppArmadillo.h>
#include <nlopt.h>
#include "nloptrAPI.h"
#include "Model.hpp"



namespace SMC
{
    class SequentialMonteCarlo
    {
    public:
        void smc_propagate_bootstrap(
            arma::mat &Theta_new, // p x N
            arma::vec &weights,   // N x 1
            double &wt,
            arma::mat &Gt,
            const double &y_old,        // (n+1) x 1, the observed response

            const arma::mat &Theta_old, // p x N
            const arma::vec &Ft,        // must be already updated if used
            const arma::uvec &model_code,
            const arma::vec &obs_par,
            const arma::vec &lag_par,
            const unsigned int &p, // dimension of DLM state space
            const int &t,
            const unsigned int &nlag,
            const unsigned int &N, // number of particles
            const double &delta_discount,
            const bool &truncated,
            const bool &use_discount,
            const bool &use_custom_val)
        {
            const unsigned int obs_code = model_code.at(0);
            const unsigned int link_code = model_code.at(1);
            const unsigned int gain_code = model_code.at(3);

            double mu0 = obs_par.at(0);
            double delta_nb = obs_par.at(1);

            /*
                ------ Step 2.1 Propagate ------
                */
            // // theta_stored: p,N,B
            if (use_discount)
            { // Use discount factor if W is not given
                arma::rowvec psi = Theta_old.row(0);
                double tmp = arma::var(psi);
                if (tmp > EPS)
                {
                    wt = tmp;
                }
                else
                {
                    wt = 1.;
                }
                // Wsqrt = std::sqrt(Wt.at(t));
                if (use_custom_val)
                {
                    wt *= 1. / delta_discount - 1.;
                }
                else
                {
                    wt *= 1. / 0.99 - 1.;
                }
            }
            double Wsqrt = std::sqrt(wt) + EPS;
            if (wt < EPS)
            {
                throw std::invalid_argument("smc_propagate_bootstrap: variance wt closed to 0.");
            }

            for (unsigned int i = 0; i < N; i++)
            {
                arma::vec theta_old = Theta_old.col(i); // p x 1
                // update_Gt(Gt, gain_code, trans_code, theta_old, y_old, rho);
                arma::mat theta_new = update_at(
                    Gt, theta_old, y_old, lag_par, gain_code, nlag, truncated);

                double omega_new = R::rnorm(0., Wsqrt);
                if (t < p)
                {
                    theta_new.at(0) += std::abs(omega_new);
                }
                else
                {
                    theta_new.at(0) += omega_new;
                }

                Theta_new.col(i) = theta_new;
            }

            /*
            ------ Step 2.2 Importance weights ------
            */
            arma::vec lambda(N);
            for (unsigned int i = 0; i < N; i++)
            {
                arma::vec theta = Theta_new.col(i); // p x 1

                // theta: p x N
                // double lambda = 0.;
                if (link_code == 1)
                {
                    // Exponential link and identity gain
                    double tmp = arma::as_scalar(Ft.t() * theta);
                    lambda.at(i) = std::exp(mu0 + tmp);
                }
                else
                {
                    double ft;
                    if (truncated)
                    {
                        arma::vec hpsi = psi2hpsi(theta, gain_code);
                        ft = arma::as_scalar(Ft.t() * hpsi);
                        // lambda.at(i) = mu0 + ft;
                    }
                    else
                    {
                        ft = theta.at(1);
                        // Koyck or Solow with identity link and different gain functions
                        // lambda.at(i) = mu0 + theta.at(1);
                    }

                    bound_check(ft, "smc_propagate_bootstrap: ft", false, true);
                    lambda.at(i) = mu0 + std::abs(ft);
                }

            }

            return;
        }
    }
}


#endif