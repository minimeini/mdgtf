#pragma once
#ifndef MODEL_H
#define MODEL_H

#include <RcppArmadillo.h>
#include "ErrDist.hpp"
#include "TransFunc.hpp"
#include "LinkFunc.hpp"
#include "ObsDist.hpp"
#include "utils.h"
// [[Rcpp::plugins(cpp17)]]
// [[Rcpp::depends(RcppArmadillo)]]



/**
 * @brief Define a dynamic generalized transfer function (DGTF) model
 *
 * @param dobs observation distribution characterized by either (mu0, delta_nb) or (mu, sd2)
 *       - (mu0, delta_nb):
 *          (1) mu0: constant mean of the observed time series;
 *          (2) delta_nb: degress of freedom for negative-binomial observation distribution.
 *       - (mu, sd2): parameters for gaussian observation distribution.
 * @param dlag lag distribution characterized by either (kappa, r) for negative-binomial lags or (mu, sd2) for lognormal lags.
 *
 */
class Model
{
public:
    Dim &dim;
    ObsDist &dobs;
    TransFunc &transfer;
    LinkFunc &flink;
    ErrDist &derr;

    Model() : dim(_dim), dobs(_dobs), transfer(_transfer), flink(_flink), derr(_derr)
    {
        _dobs.init_default();
        _flink.init_default();
        _transfer.init_default();
        _derr.init_default();
        _dim.init_default();

        _transfer.dlag.get_Fphi(_dim.nL);

        return;
    }

    Model(
        const Dim &dim_,
        const std::string &obs_dist = "nbinom",
        const std::string &link_func = "identity",
        const std::string &gain_func = "softplus",
        const std::string &lag_dist = "lognorm",
        const std::string &err_dist = "gaussian",
        const Rcpp::NumericVector &obs_param = Rcpp::NumericVector::create(0., 30.),
        const Rcpp::NumericVector &lag_param = Rcpp::NumericVector::create(NB_KAPPA, NB_R),
        const Rcpp::NumericVector &err_param = Rcpp::NumericVector::create(0.01, 0.), // (W, w[0])
        const std::string &trans_func = "sliding") : dim(_dim), dobs(_dobs), transfer(_transfer), flink(_flink), derr(_derr)
    {
        _dim = dim_;
        _dobs.init(obs_dist, obs_param[0], obs_param[1]);
        _flink.init(link_func, obs_param[0]);
        _transfer.init(dim_, trans_func, gain_func, lag_dist, lag_param);

        _derr.init("gaussian", err_param[0], err_param[1]);
        return;
    }

    


    void update_dobs(const double &value, const unsigned int &iloc)
    {
        if (iloc == 0)
        {
            _dobs.update_par1(value);
        }
        else
        {
            _dobs.update_par2(value);
        }
    }

    void update_dlag(const double &value, const unsigned int &iloc)
    {
        if (iloc == 0)
        {
            _transfer.dlag.update_par1(value);
        }
        else
        {
            _transfer.dlag.update_par2(value);
        }
    }

    void set_dim(
        const unsigned int &ntime,
        const unsigned int &nlag = 0)
    {
        _dim.init(ntime, nlag, _transfer.dlag.par2);
        return;
    }


    /**
     * @brief Simulate from the DGTF model from the scratch using the transfer function form.
     * 
     * @return arma::vec 
     */
    arma::vec simulate(const double &y0 = 0.)
    {
        lambda.set_size(_dim.nT + 1);
        lambda.zeros();

        // Sample psi[t].
        _derr.sample(_dim.nT, true);
        _transfer.fgain.update_psi(_derr.psi);

        // Get h(psi[t])
        arma::vec hpsi = GainFunc::psi2hpsi<arma::vec>(
            _transfer.fgain.psi, 
            _transfer.fgain.name); // Checked. OK.
        _transfer.fgain.update_hpsi(hpsi); // Checked. OK.

        // Get phi[1], ..., phi[nL]
        _transfer.dlag.get_Fphi(_dim.nL); // Checked. OK.

        arma::vec y(_dim.nT + 1, arma::fill::zeros);
        y.at(0) = y0;
        for (unsigned int t = 1; t < _dim.nT + 1; t++)
        {
            double ft = _transfer.transfer_sliding(t, y); // Checked. OK.
            double mu = LinkFunc::ft2mu(ft, _flink.name, _dobs.par1);
            lambda.at(t) = mu;
            y.at(t) = _dobs.sample(mu); // Checked. OK.
        }

        return y; // Checked. OK.
    }

    arma::vec lambda;

    static arma::vec simulate(
        const arma::vec &psi, // (ntime + 1) x 1
        const unsigned int &nlag,
        const double &y0 = 0.,
        const std::string &gain_func = "softplus",
        const std::string &lag_dist = "lognormal",
        const std::string &link_func = "identity",
        const std::string &obs_dist = "nbinom",
        const Rcpp::NumericVector &lag_param = Rcpp::NumericVector::create(1.4, 0.3),
        const Rcpp::NumericVector &obs_param = Rcpp::NumericVector::create(0., 30))
    {
        ObsDist dobs(obs_dist, obs_param[0], obs_param[1]);
        unsigned int ntime = psi.n_elem - 1;
        arma::vec hpsi = GainFunc::psi2hpsi<arma::vec>(psi, gain_func);                 // Checked. OK.
        arma::vec Fphi = LagDist::get_Fphi(nlag, lag_dist, lag_param[0], lag_param[1]); // Checked. OK.

        arma::vec y(ntime + 1, arma::fill::zeros);
        y.at(0) = y0;
        for (unsigned int t = 1; t < (ntime + 1); t++)
        {
            double ftt = TransFunc::transfer_sliding(t, nlag, y, Fphi, hpsi); // Checked. OK.
            double mu = LinkFunc::ft2mu(ftt, link_func, obs_param[0]);        // Checked. OK.
            y.at(t) = ObsDist::sample(mu, obs_param[1], obs_dist);            // Checked. OK.
        }

        return y; // Checked. OK.
    }

    arma::vec forecast(
        const arma::mat &psi, // (nT + 1) x nsample
        const unsigned int &nstep = 1,
        const unsigned int &nrep = 1
    )
    {
        const unsigned int nsample = psi.n_cols;
        arma::cube ynew(nstep, nrep, nsample);
        arma::vec psi_new = ErrDist::sample(nstep, _derr.par1, _derr.par2, true, _derr.name); // nstep x 1
        return psi_new;

    }

    arma::vec wt2lambda(const arma::vec &y, const arma::vec &wt) // checked. ok.
    {
        arma::vec psi = arma::cumsum(wt);
        _transfer.fgain.update_psi(psi);
        _transfer.fgain.psi2hpsi<arma::vec>();
        
        arma::vec ft(psi.n_elem, arma::fill::zeros);
        arma::vec lambda = ft;
        for (unsigned int t = 1; t <= dim.nT; t++)
        {
            ft.at(t) = _transfer.func_ft(t, y, ft);
            lambda.at(t) = LinkFunc::ft2mu(ft.at(t), _flink.name, _dobs.par1);
        }

        return lambda;
    }

private:
    ObsDist _dobs; // Observation distribution
    TransFunc _transfer;
    LinkFunc _flink;
    ErrDist _derr;
    Dim _dim;
    double _y0 = 0.;
    double _mu0 = 1.;

    

};

/**
 * @brief State space (DLM) form of the model.
 *
 */
class StateSpace
{
public:
    /**
     * @brief Expected state evolution equation for the DLM form model. Expectation of theta[t + 1] = g(theta[t]).
     *
     * @param model
     * @param theta_cur
     * @param ycur
     * @return arma::vec
     */
    static arma::vec func_gt( // Checked. OK.
        const Model &model,
        const arma::vec &theta_cur, // nP x 1, (psi[t], f[t-1], ..., f[t-r])
        const double &ycur)
    {
        arma::vec theta_next;
        theta_next.copy_size(theta_cur);

        switch (model.transfer.trans_list[model.transfer.name])
        {
        case AVAIL::Transfer::iterative:
        {
            theta_next.at(0) = theta_cur.at(0); // Expectation of random walk.
            theta_next.at(1) = TransFunc::transfer_iterative(
                theta_cur.subvec(1, model.dim.nP - 1), // f[t-1], ..., f[t-r]
                theta_cur.at(0),                       // psi[t]
                ycur,                                  // y[t-1]
                model.transfer.name,
                model.transfer.dlag.par1,
                model.transfer.dlag.par2);
            theta_next.subvec(2, model.dim.nP - 1) = theta_cur.subvec(1, model.dim.nP - 2);
            break;
        }
        default: // AVAIL::Transfer::sliding
        {
            // theta_next = model.transfer.G0 * theta_cur;
            theta_next.at(0) = theta_cur.at(0);
            theta_next.subvec(1, model.dim.nP - 1) = theta_cur.subvec(0, model.dim.nP - 2);
            break;
        }
        }

        bound_check<arma::vec>(theta_next, "func_gt: theta_next");
        return theta_next;
    }

    static arma::vec func_state_propagate(
        const Model &model,
        const arma::vec &theta_now,
        const double &ynow,
        const double &Wsqrt,
        const bool &positive_noise = false)
    {
        arma::vec theta_next = func_gt(model, theta_now, ynow);
        double omega_next = R::rnorm(0., Wsqrt); // [Input] - Wsqrt
        if (positive_noise)                      // t < Theta_now.n_rows
        {
            theta_next.at(0) += std::abs(omega_next);
        }
        else
        {
            theta_next.at(0) += omega_next;
        }
        return theta_next;
    }

    /**
     * @brief f[t]( theta[t] ) - maps state theta[t] to observation-level variable f[t].
     *
     * @param model
     * @param t
     * @param theta_cur
     * @param yall
     * @return double
     */
    static double func_ft(
        const Model &model,
        const int &t,               // t = 0, y[0] = 0, theta[0] = 0; t = 1, y[1], theta[1]; ...
        const arma::vec &theta_cur, // theta[t] = (psi[t], ..., psi[t+1 - nL]) or (psi[t+1], f[t], ..., f[t+1-r])
        const arma::vec &yall       // We use y[t - nelem], ..., y[t-1]
    )
    {
        double ft_cur;
        if (model.transfer.trans_list[model.transfer.name] == AVAIL::sliding)
        {
            int nelem = std::min(t, (int)model.dim.nL); // min(t,nL)

            arma::vec yold(model.dim.nL, arma::fill::zeros);
            if (nelem > 1)
            {
                yold.tail(nelem) = yall.subvec(t - nelem, t - 1); // 0, ..., 0, y[t - nelem], ..., y[t-1]
            }
            else if (t > 0) // nelem = 1 at t = 1
            {
                yold.at(model.dim.nL - 1) = yall.at(t - 1);
            }

            yold = arma::reverse(yold); // y[t-1], ..., y[t-min(t,nL)]

            arma::vec ft_vec = model.transfer.dlag.Fphi;

            arma::vec hpsi_cur = GainFunc::psi2hpsi(theta_cur, model.transfer.fgain.name); // (h(psi[t]), ..., h(psi[t+1 - nL]))
            arma::vec ftmp = yold % hpsi_cur;
            ft_vec = ft_vec % ftmp;

            ft_cur = arma::accu(ft_vec);
        }
        else
        {
            ft_cur = theta_cur.at(1);
        }


        bound_check(ft_cur, "func_ft: ft_cur", false, true);
        return ft_cur;
    }

// private:
//     arma::vec theta; // nP x 1
//     arma::mat theta_series; // nP x (nT + 1)
//     arma::vec theta_init;

//     arma::vec psi; // (nT + 1) x 1

//     Model model;
};

class ApproxDisturbance
{
public:
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

    void set_Fphi(const LagDist &dlag, const unsigned int &nlag)
    {
        Fphi.zeros();
        arma::vec tmp = LagDist::get_Fphi(nT, dlag.name, dlag.par1, dlag.par2);
        if (nlag < nT)
        {
            tmp.subvec(nlag, nT - 1).zeros();
        }
        Fphi.tail(nT) = tmp;
    }

    void set_psi(const arma::vec &psi_in)
    {
        psi = psi_in;
        hpsi = GainFunc::psi2hpsi(psi, gain_func);
        dhpsi = GainFunc::psi2dhpsi(psi, gain_func);
    }

    void set_wt(const arma::vec &wt_in)
    {
        wt = wt_in;
        psi = arma::cumsum(psi);
        hpsi = GainFunc::psi2hpsi(psi, gain_func);
        dhpsi = GainFunc::psi2dhpsi(psi, gain_func);
    }

    arma::vec getFphi(){return Fphi;}
    arma::mat get_Fn(){return Fn;}
    arma::vec get_f0(){return f0;}
    arma::vec get_psi(){return psi;}
    arma::vec get_hpsi(){return hpsi;}
    arma::vec get_dhpsi(){return dhpsi;}

    void update_by_psi(const arma::vec &y, const arma::vec &psi)
    {
        hpsi = GainFunc::psi2hpsi(psi, gain_func);
        dhpsi = GainFunc::psi2dhpsi(psi, gain_func);
        update_f0(y);
        update_Fn(y);
    }

    void update_by_wt(const arma::vec &y, const arma::vec &wt)
    {
        psi = arma::cumsum(wt);
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

        bound_check<arma::mat>(Fn, "get_Fn: Fn");
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
        bound_check<arma::vec>(f0, "get_f0: f0");
        return;
    }

    /**
     * @brief Get the regressor eta[1], ..., eta[nT]. If mu = 0, it is equal to only the transfer effect, f[1], ..., f[nT]. Must set Fphi before using this function.
     *
     * @param wt
     * @param y (nT + 1) x 1, only use the past values before each t
     * @return arma::vec, (f[1], ..., f[nT])
     */
    arma::vec get_eta_approx(const double &mu0 = 0.)
    {
        arma::vec ft = mu0 + f0 + Fn * wt.tail(nT); // nT x 1
        bound_check(ft, "func_eta_approx: ft");
        return ft;
    }

    static arma::vec func_Vt_approx( // Checked. OK.
        const arma::vec lambda, // (nT + 1) x 1
        const ObsDist &obs_dist,
        const std::string &link_func)
    {
        std::map<std::string, AVAIL::Dist> obs_list = AVAIL::obs_list;
        std::map<std::string, AVAIL::Func> link_list = AVAIL::link_list;
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
            case AVAIL::Func::identity:
            {
                Vt = lambda % (lambda + obs_dist.par2);
                Vt = Vt / obs_dist.par2;
                break;
            }
            case AVAIL::Func::exponential:
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

        bound_check(Vt, "func_Vt_approx: Vt", true, true);
        Vt.elem(arma::find(Vt < EPS8)).fill(EPS8);
        return Vt;
    }

    static arma::vec func_yhat(
        const arma::vec &y,
        const std::string &link_func)
    {
        std::map<std::string, AVAIL::Func> link_list = AVAIL::link_list;
        arma::vec yhat;

        switch (link_list[tolower(link_func)])
        {
        case AVAIL::Func::identity:
        {
            yhat = y;
            break;
        }
        case AVAIL::Func::exponential:
        {
            yhat = arma::log(arma::abs(y) + EPS);
            break;
        }
        default:
        {
            break;
        }
        } // switch by link

        bound_check(yhat, "func_yhat");
        return yhat;
    }

private:
    unsigned int nT = 200;
    std::string gain_func = "softplus";

    arma::mat Fn;
    arma::vec f0;
    arma::vec Fphi;
    arma::vec wt, psi, hpsi, dhpsi;
};


#endif