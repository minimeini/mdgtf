#pragma once
#ifndef APPROXDISTURBANCE_H
#define APPROXDISTURBANCE_H

#include <RcppArmadillo.h>
#include "../core/ErrDist.hpp"
#include "../core/SysEq.hpp"
#include "../core/TransFunc.hpp"
#include "../core/ObsDist.hpp"
#include "../core/LinkFunc.hpp"
#include "../core/Regression.hpp"
#include "../utils/utils.h"


/**
 * @brief Mostly used in MCMC.
 * 
 */
class ApproxDisturbance
{
public:
    arma::mat Fn;
    arma::vec f0;

    ApproxDisturbance()
    {
        nT = 200;
        gain_func = "softplus";

        Fn.set_size(nT, nT); Fn.zeros();
        f0.set_size(nT); f0.zeros();

        Fphi.set_size(nT + 1);
        Fphi.zeros();

        psi = Fphi;
        hpsi = psi;
        dhpsi = psi;

        return;
    }

    ApproxDisturbance(const unsigned int &ntime, const std::string &gain_func_name = "softplus")
    {
        nT = ntime;
        gain_func = gain_func_name;

        Fn.set_size(nT, nT); Fn.zeros();
        f0.set_size(nT); f0.zeros();

        Fphi.set_size(nT + 1);
        Fphi.zeros();

        wt = Fphi;
        psi = Fphi;
        hpsi = psi;
        dhpsi = psi;
    }


    void set_Fphi(const LagDist &dlag, const unsigned int &nlag) // (nT + 1)
    {
        Fphi.zeros();
        arma::vec tmp = LagDist::get_Fphi(nT, dlag.name, dlag.par1, dlag.par2); // nT x 1
        if (nlag < nT)
        {
            tmp.subvec(nlag, nT - 1).zeros();
        }
        Fphi.tail(nT) = tmp; // fill Fphi[1:nT], leave Fphi[0] = 0.
    }


    arma::mat get_Fn(){return Fn;}


    void update_by_wt(const arma::vec &y, const arma::vec &wt)
    {
        psi = arma::cumsum(wt);
        hpsi = GainFunc::psi2hpsi(psi, gain_func);
        dhpsi = GainFunc::psi2dhpsi(psi, gain_func);
        update_f0(y);
        update_Fn(y);
    }

    void update_by_psi(const arma::vec &y, const arma::vec &psi_in)
    {
        psi = psi_in;
        hpsi = GainFunc::psi2hpsi(psi, gain_func);
        dhpsi = GainFunc::psi2dhpsi(psi, gain_func);
        update_f0(y);
        update_Fn(y);
    }

    /**
     * @brief Need to run `set_psi` before using this function. Fphi must be initialized.
     * 
     * @param t 
     * @param y 
     * @return arma::vec 
     */
    arma::vec get_increment_matrix_byrow( // Checked. OK.
        const unsigned int &t, // t = 1, ..., nT
        const arma::vec &y     // (nT + 1) X 1, y[0], y[1], ..., y[nT], only use the past values before t
        )
    {
        arma::vec phi = Fphi.subvec(1, t);
        // t x 1, phi[1], ..., phi[nlag], phi[nlag + 1], ..., phi[t]
        //      = phi[1], ..., phi[nlag],       0,       ..., 0 (at time t)
        arma::vec yt = y.subvec(0, t - 1);        // y[0], y[1], ..., y[t-1]
        arma::vec dhpsi_tmp = dhpsi.subvec(1, t); // t x 1, h'(psi[1]), ..., h'(psi[t])

        arma::vec increment_row = arma::reverse(yt % dhpsi_tmp); // t x 1
        increment_row = increment_row % phi;

        return increment_row;
    }

    void update_Fn( // Checked. OK.
        const arma::vec &y   // (nT + 1) x 1, y[0], ..., y[nT - 1], y[nT], only use the past values before each t and y[nT] is not used
        )
    {
        Fn.zeros();
        for (unsigned int t = 1; t <= nT; t++)
        {
            arma::vec Fnt = get_increment_matrix_byrow(t, y); // t x 1
            Fnt = arma::cumsum(Fnt);
            Fnt = arma::reverse(Fnt);

            Fn.submat(t - 1, 0, t - 1, t - 1) = Fnt.t();
        }

        #ifdef DGTF_DO_BOUND_CHECK
            bound_check<arma::mat>(Fn, "get_Fn: Fn");
        #endif
        return;
    }

    void update_f0( // Checked. OK.
        const arma::vec &y   // (nT + 1) x 1, only use the past values before each t
        )
    {
        f0.zeros();
        arma::vec h0 = hpsi - dhpsi % psi; // (nT + 1) x 1, h0[0] = 0

        for (unsigned int t = 1; t <= nT; t++)
        {
            arma::vec F0t = get_increment_matrix_byrow(t, y);
            double f0t = arma::accu(F0t);

            f0.at(t - 1) = f0t;
        }

        f0.at(0) = (f0.at(0) < EPS8) ? EPS8 : f0.at(0);
        #ifdef DGTF_DO_BOUND_CHECK
            bound_check<arma::vec>(f0, "get_f0: f0");
        #endif
        return;
    }

    /**
     * @brief Get the regressor eta[1], ..., eta[nT] as a function of {w[t]}, evolution errors of the latent state. If mu = 0, it is equal to only the transfer effect, f[1], ..., f[nT]. Must set Fphi before using this function.
     *
     * @param wt
     * @param y (nT + 1) x 1, only use the past values before each t
     * @return arma::vec, (f[1], ..., f[nT])
     */
    arma::vec get_eta_approx(const Season &seas)
    {
        arma::vec eta = f0 + Fn * wt.tail(nT); // nT x 1
        if (seas.period > 0)
        {
            arma::vec seas_reg = seas.X.t() * seas.val; // (nT + 1) x 1
            eta = eta + seas_reg.tail(eta.n_elem);
        }

        #ifdef DGTF_DO_BOUND_CHECK
            bound_check<arma::vec>(eta, "func_eta_approx: eta");
        #endif
        return eta;
    }


    static arma::vec func_Vt_approx( // Checked. OK.
        const arma::vec &lambda, // (nT + 1) x 1
        const ObsDist &obs_dist,
        const std::string &link_func)
    {
        std::map<std::string, AVAIL::Dist> obs_list = ObsDist::obs_list;
        std::map<std::string, LinkFunc::Func> link_list = LinkFunc::link_list;
        arma::vec Vt = lambda;

        switch (obs_list[obs_dist.name])
        {
        case AVAIL::Dist::poisson:
        {
            switch (link_list[tolower(link_func)])
            {
            case AVAIL::Func::identity:
            {
                Vt = lambda;
                break;
            }
            case AVAIL::Func::exponential:
            {
                // Vt = 1 / lambda = exp( - log(lambda) )
                Vt = -arma::log(arma::abs(lambda) + EPS);
                Vt = arma::exp(Vt);
                break;
            }
            default:
            {
                break;
            }
            }      // switch by link
            break; // Done case poisson
        }
        case AVAIL::Dist::nbinomm:
        {
            switch (link_list[tolower(link_func)])
            {
            case LinkFunc::Func::identity:
            {
                Vt = lambda % (lambda + obs_dist.par2);
                Vt = Vt / obs_dist.par2;
                break;
            }
            case LinkFunc::Func::exponential:
            {
                arma::vec nom = (lambda + obs_dist.par2);
                arma::vec denom = obs_dist.par2 * lambda;
                Vt = nom / denom;
                break;
            }
            default:
            {
                break;
            }
            }      // switch by link
            break; // case nbinom
        }
        default:
        {
        }
        } // switch by observation distribution.

        #ifdef DGTF_DO_BOUND_CHECK
            bound_check<arma::vec>(Vt, "func_Vt_approx: Vt", true, true);
        #endif
        Vt += EPS8;
        return Vt;
    }

    static double func_Vt_approx( // Checked. OK.
        const double &lambda,      // (nT + 1) x 1
        const ObsDist &obs_dist,
        const std::string &link_func)
    {
        std::map<std::string, AVAIL::Dist> obs_list = ObsDist::obs_list;
        std::map<std::string, LinkFunc::Func> link_list = LinkFunc::link_list;
        double Vt = lambda;

        switch (obs_list[obs_dist.name])
        {
        case AVAIL::Dist::poisson:
        {
            switch (link_list[tolower(link_func)])
            {
            case LinkFunc::Func::identity:
            {
                Vt = lambda;
                break;
            }
            case LinkFunc::Func::exponential:
            {
                // Vt = 1 / lambda = exp( - log(lambda) )
                Vt = -std::log(std::abs(lambda) + EPS);
                Vt = std::exp(Vt);
                break;
            }
            default:
            {
                break;
            }
            } // switch by link
            break; // Done case poisson
        }
        case AVAIL::Dist::nbinomm:
        {
            switch (link_list[tolower(link_func)])
            {
            case AVAIL::Func::identity:
            {
                Vt = lambda * (lambda + obs_dist.par2);
                Vt = Vt / obs_dist.par2;
                break;
            }
            case AVAIL::Func::exponential:
            {
                double nom = (lambda + obs_dist.par2);
                double denom = obs_dist.par2 * lambda;
                Vt = nom / denom;
                break;
            }
            default:
            {
                break;
            }
            } // switch by link
            break; // case nbinom
        }
        default:
        {
        }
        } // switch by observation distribution.

        #ifdef DGTF_DO_BOUND_CHECK
            bound_check(Vt, "func_Vt_approx<double>: Vt", true, true);
        #endif
        Vt = std::max(Vt, EPS);
        return Vt;
    }


private:
    unsigned int nT = 200;
    std::string gain_func = "softplus";

    arma::vec Fphi;
    arma::vec wt, psi, hpsi, dhpsi;
};


#endif