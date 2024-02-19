#ifndef _LINEARBAYES_H
#define _LINEARBAYES_H

#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <algorithm>
#include <RcppArmadillo.h>
#include <nlopt.h>
#include "nloptrAPI.h"
#include "Model.hpp"


/**
 * psi2hpsi: GainFunc.hpp
 * hpsi_deriv: first order derivative
 * get_Fphi: LagDist.hpp
 * 
 * 
 * Q. should they go to class nbinom?
 * trigamma_obj
 * optimize_trigamma
 * 
 * 
 */


namespace LBA
{

    /**
     * @brief First-order derivative of g[t] w.r.t theta[t]; used in the calculation of R[t] = G[t] C[t-1] t(G[t]) + W[t].
     *
     * @param model
     * @param mt_old
     * @param yold
     * @return arma::mat
     */
    static arma::mat func_Gt(
        const Model &model,
        const arma::vec &mt_old,
        const double &yold)
    {
        arma::mat Gt = model.transfer.G0;
        if (model.transfer.trans_list[model.transfer.name] == AVAIL::Transfer::iterative)
        {
            double dhpsi_now = GainFunc::psi2dhpsi(
                mt_old.at(0), // h'(psi[t])
                model.transfer.fgain.name);
            Gt.at(1, 0) = model.transfer.coef_now * yold * dhpsi_now;
        }

        return Gt;
    }

    /**
     * @brief R[t] = G[t] C[t-1] t(G[t]) + W[t] = P[t] + W[t].
     *
     * @param Gt G[t]
     * @param Ct_old C[t-1] = E( theta[t-1] | D[t] )
     * @param W W of W[t] = diag(W, 0, ..., 0).
     * @param use_discount If use discount factor, we have R[t] = P[t] / delta.
     * @param delta_discount The value of discount factor delta.
     * @return arma::mat
     */
    static arma::mat func_Rt(
        const arma::mat &Gt,
        const arma::mat &Ct_old,
        const double &W = 0.01,
        const bool &use_discount = false,
        const double &delta_discount = 0.95)
    {
        arma::mat Rt = Gt * Ct_old * Gt.t();
        if (use_discount)
        {
            // A unknown but general W[t] (could have non-zero off-diagonal values) with discount factor
            Rt.for_each([&delta_discount](arma::mat::elem_type &val)
                        { val /= delta_discount; });
        }
        else
        {
            // W[t] = diag(W, 0, ..., 0)
            Rt.at(0, 0) += W;
        }

        bound_check<arma::mat>(Rt, "func_Rt: Rt");
        return Rt;
    }

 
    /**
     * @brief First-order derivative of f[t] w.r.t theta[t].
     *
     * @param model
     * @param t
     * @param theta_cur
     * @param yall
     * @return arma::vec
     */
    static arma::vec func_Ft(
        const Model &model,
        const unsigned int &t,      // t = 0, y[0] = 0, theta[0] = 0; t = 1, y[1], theta[1]; ...
        const arma::vec &theta_cur, // theta[t] = (psi[t], ..., psi[t+1 - nL]) or (psi[t+1], f[t], ..., f[t+1-r])
        const arma::vec &yall       // y[0], y[1], ..., y[nT]
    )
    {
        arma::vec Ft_cur = model.transfer.F0;
        if (model.transfer.trans_list[model.transfer.name] == AVAIL::sliding)
        {
            unsigned int nelem = std::min(t, model.dim.nL); // min(t,nL)
            arma::vec yold(model.dim.nL, arma::fill::zeros);
            yold.head(nelem) = yall.subvec(t - nelem, t - 1); // y[t - min(t,nL)], ..., y[t-1]
            yold = arma::reverse(yold);                       // y[t-1], ..., y[t-min(t,nL)]

            arma::vec dhpsi_cur = GainFunc::psi2dhpsi(theta_cur, model.transfer.fgain.name); // (h'(psi[t]), ..., h'(psi[t+1 - nL]))
            Ft_cur = yold % dhpsi_cur;
            Ft_cur = Ft_cur % model.transfer.dlag.Fphi;
        }

        bound_check<arma::vec>(Ft_cur, "func_Ft: Ft_cur");

        return Ft_cur;
    }

    /**
     * @brief From (a[t], R[t]) to (f[t], q[t]).
     *
     * @param mean_ft
     * @param var_ft
     * @param t
     * @param model
     * @param at
     * @param Rt
     * @param yall
     */
    static void func_prior_ft(
        double &mean_ft,
        double &var_ft,
        arma::vec &_Ft,
        const unsigned int &t,
        const Model &model,
        const arma::vec &yall,
        const arma::vec &at,
        const arma::mat &Rt)
    {
        mean_ft = Model::func_ft(model, t, at, yall);
        _Ft = func_Ft(model, t, at, yall);
        var_ft = arma::as_scalar(_Ft.t() * Rt * _Ft);
        return;
    }

    /**
     * @brief From (f[t], q[t]) to (alpha[t], beta[t])
     *
     * @param alpha
     * @param beta
     * @param ft
     * @param qt
     */
    static void func_alpha_beta(
        double &alpha,
        double &beta,
        const Model &model,
        const double &ft,
        const double &qt,
        const double &ycur = 0.,
        const bool &get_posterior = true)
    {
        double regressor = model.flink.mu0 + ft;

        switch (model.dobs.obs_list[model.dobs.name])
        {
        case AVAIL::Dist::poisson:
        {
            switch (model.flink.link_list[model.flink.name])
            {
            case AVAIL::Func::identity:
            {
                alpha = std::pow(regressor, 2.) / qt;
                beta = regressor / qt;
                break;
            }
            case AVAIL::Func::exponential:
            {
                alpha = 1. / qt;
                double nom = std::exp(-regressor);
                beta = nom / qt;
                break;
            }
            default:
            {
                throw std::invalid_argument("Unknown link function for Poisson observation distribution.");
            }
            }

            if (get_posterior)
            {
                alpha += ycur;
                beta += 1.;
            }
            break;
        }
        case AVAIL::Dist::nbinomm:
        {
            if (model.flink.link_list[model.flink.name] == AVAIL::Func::identity)
            {
                beta = regressor * (regressor + model.dobs.par2);
                beta /= qt;
                beta += 2.;

                alpha = regressor * (beta - 1.);
                alpha /= model.dobs.par2;
            }
            else
            {
                throw std::invalid_argument("Unknown link function for negative-binomial observation distribution.");
            }

            if (get_posterior)
            {
                alpha += ycur;
                beta += model.dobs.par2;
            }
            break;
        }
        default:
        {
            throw std::invalid_argument("Unknown observation distribution.");
        }
        }

        bound_check(alpha, "func_alpha_beta: alpha");
        bound_check(beta, "func_alpha_beta: beta");

        return;
    }

    static void func_posterior_ft(
        double &mean_ft,
        double &var_ft,
        const Model &model,
        const double &alpha,
        const double &beta)
    {
        switch (model.dobs.obs_list[model.dobs.name])
        {
        case AVAIL::Dist::poisson:
        {
            switch (model.flink.link_list[model.flink.name])
            {
            case AVAIL::Func::identity:
            {
                Poisson::moments_mean(mean_ft, var_ft, alpha, beta, 0.);
                break;
            }
            case AVAIL::Func::exponential:
            {
                mean_ft = Gamma::mean_logGamma(alpha, beta);
                var_ft = Gamma::var_logGamma(alpha, 0.);
                break;
            }
            default:
            {

                throw std::invalid_argument("Unknown link function for Poisson observation distribution.");
            }
            }
            break;
        }
        case AVAIL::Dist::nbinomm:
        {
            if (model.flink.link_list[model.flink.name] == AVAIL::Func::identity)
            {
                nbinomm::moments_mean(mean_ft, var_ft, alpha, beta, model.dobs.par2);
            }
            else
            {
                throw std::invalid_argument("Unknown link function for negative-binomial observation distribution.");
            }
            break;
        }
        default:
        {
            throw std::invalid_argument("Unknown observation distribution.");
        }
        }

        mean_ft -= model.flink.mu0;

        bound_check(mean_ft, "func_posterior_ft: mean_ft");
        bound_check(var_ft, "func_posterior_ft: var_ft");

        return;
    }

    static arma::vec func_At(
        const arma::mat &Rt,
        const arma::vec &Ft,
        const double &qt)
    {
        arma::vec At = Rt * Ft;
        At.for_each([&qt](arma::vec::elem_type &val)
                    { val /= qt; });

        bound_check<arma::vec>(At, "func_At: At");
        return At;
    }

    static arma::vec func_mt(
        const arma::vec &at,
        const arma::vec &At,
        const double &prior_mean_ft,
        const double &posterior_mean_ft)
    {
        double err = posterior_mean_ft - prior_mean_ft;
        arma::vec mt = at + At * err;

        bound_check<arma::vec>(mt, "func_mt: mt");
        return mt;
    }

    static arma::mat func_Ct(
        const arma::mat &Rt,
        const arma::vec &At,
        const double &prior_var_ft,
        const double &posterior_var_ft)
    {
        double err = posterior_var_ft - prior_var_ft;
        arma::mat Ct = Rt + err * (At * At.t());
        bound_check<arma::mat>(Ct, "func_Ct: Ct");
        return Ct;
    }

    static arma::mat get_psi(const arma::mat &mean, const arma::cube &var)
    {
        arma::mat psi(mean.n_cols, 3);
        psi.col(1) = arma::vectorise(mean.row(0));
        arma::vec psi_sd = arma::vectorise(var.tube(0, 0, 0, 0));
        psi_sd = arma::sqrt(arma::abs(psi_sd) + EPS);
        psi.col(0) = psi.col(1) - 2. * psi_sd;
        psi.col(2) = psi.col(1) + 2. * psi_sd;
        return psi; // (nT + 1) x 3
    }

    class LinearBayes
    {
    public:
        // LinearBayes() : at(_at), Rt(_Rt), mt(_mt), Ct(_Ct)
        // {
        //     Model model_;
        //     _W = 0.01;
        //     _use_discount = false;
        //     _discount_factor = 0.95;

        //     _nT = model_.dim.nT;
        //     _nP = model_.dim.nP;

        //     double _ft = 0.;
        //     _ft_prior_mean = _ft;
        //     _ft_prior_var = _ft;
        //     _ft_post_mean = _ft;
        //     _ft_post_var = _ft;
        //     _alphat = _ft;
        //     _betat = _ft;

        //     _At.set_size(_nP);
        //     _At.zeros();

        //     _Ft = model_.transfer.F0; // set F[0]
        //     _Gt = model_.transfer.G0; // set G[0]

        //     _at.set_size(_nP, _nT + 1);
        //     _at.zeros();
        //     _mt = _at;
        //     _atilde_t = _at;
        //     // set m[0]

        //     _Rt.set_size(_nP, _nP, _nT + 1);
        //     _Rt.for_each([](arma::cube::elem_type &val)
        //                  { val = 0.; });
        //     _Rtilde_t = Rt;
        //     _Ct = _Rt;
        //     _Ct.slice(0) = arma::eye<arma::mat>(_nP, _nP);
        //     _Ct.for_each([](arma::cube::elem_type &val)
        //                  { val *= 10.; });
        // }

        LinearBayes(
        const Model &model, 
        const arma::vec &y, // (nT + 1) x 1
        const double &W = 0.01,
        const double &discount_factor = 0.95,
        const bool &use_discount = false,
        const double &theta_upbnd = 1.) : atilde(_atilde_t), Rtilde(_Rtilde_t), at(_at), Rt(_Rt), mt(_mt), Ct(_Ct), _model(model)
        {
            _W = W;
            _discount_factor = discount_factor;
            _use_discount = use_discount;

            _nP = model.dim.nP;
            _nT = model.dim.nT;

            double _ft = 0.;
            _ft_prior_mean = _ft;
            _ft_prior_var = _ft;
            _ft_post_mean = _ft;
            _ft_post_var = _ft;
            _alphat = _ft;
            _betat = _ft;

            _psi_mean.set_size(_nT + 1);
            _psi_mean.zeros();
            _psi_var = _psi_mean;

            _At.set_size(_nP);
            _At.zeros();

            _Ft = model.transfer.F0; // set F[0]
            _Gt = model.transfer.G0; // set G[0]

            _at.set_size(_nP, _nT + 1);
            _at.zeros();
            _mt = _at;
            _atilde_t = _at;
            // set m[0]

            _Rt.set_size(_nP, _nP, _nT + 1);
            _Rt.for_each([](arma::cube::elem_type &val)
                         { val = 0.; });
            _Rtilde_t = _Rt;
            _Ct = _Rt;
            _Ct.slice(0) = arma::eye<arma::mat>(_nP, _nP);
            _Ct.for_each([&theta_upbnd](arma::cube::elem_type &val) { val *= theta_upbnd; });
            // set C[0]

            _y.set_size(model.dim.nT + 1);
            _y.zeros();
            _y.tail(y.n_elem) = y; // (nT + 1) x 1;
            _y.elem(arma::find(_y < EPS)).fill(0.01 / static_cast<double>(_nP));

            return;
        }

        const arma::mat &at;
        const arma::cube &Rt;
        const arma::mat &mt;
        const arma::cube &Ct;
        const arma::mat &atilde;
        const arma::cube &Rtilde;


        void filter()
        {
            for (unsigned int t = 1; t <= _model.dim.nT; t++)
            {
                _at.col(t) = Model::func_gt(_model, _mt.col(t-1), _y.at(t-1));
                _Gt = func_Gt(_model, _mt.col(t-1), _y.at(t-1));
                _Rt.slice(t) = func_Rt(_Gt, _Ct.slice(t - 1), _W, _use_discount, _discount_factor);

                func_prior_ft(
                    _ft_prior_mean, _ft_prior_var, _Ft,
                    t, _model, _y,
                    _at.col(t), _Rt.slice(t));

                func_alpha_beta(_alphat, _betat, _model, _ft_prior_mean, _ft_prior_var, _y.at(t), true);

                _At = func_At(_Rt.slice(t), _Ft, _ft_prior_var);

                func_posterior_ft(_ft_post_mean, _ft_post_var, _model, _alphat, _betat);

                _mt.col(t) = func_mt(_at.col(t), _At, _ft_prior_mean, _ft_post_mean);
                _Ct.slice(t) = func_Ct(_Rt.slice(t), _At, _ft_prior_var, _ft_post_var);
            }

            return;
        }


        void smoother(const bool &use_pseudo = false)
        {
            bool is_iterative = _model.transfer.trans_list[_model.transfer.name] == AVAIL::Transfer::iterative;

            _atilde_t.col(_nT) = _mt.col(_nT);
            _Rtilde_t.slice(_nT) = _Ct.slice(_nT);

            _psi_mean.at(_nT) = _atilde_t.at(0, _nT);
            _psi_var.at(_nT) = _Rtilde_t.at(0, 0, _nT);

            for (unsigned int t = _nT; t > 0; t--)
            {
                if (!_use_discount)
                {
                    arma::mat Rtinv = inverse(_Rt.slice(t));

                    _Gt = func_Gt(_model, _mt.col(t - 1), _y.at(t - 1));
                    arma::mat Bt = (_Ct.slice(t - 1) * _Gt.t()) * Rtinv;

                    arma::vec diff_a = _atilde_t.col(t) - _at.col(t);
                    arma::mat diff_R = _Rtilde_t.slice(t) - _Rt.slice(t);

                    arma::vec _atilde_right = Bt * diff_a;
                    _atilde_t.col(t - 1) = _mt.col(t - 1) + _atilde_right;

                    arma::mat _Rtilde_right = (Bt * diff_R) * Bt.t();
                    _Rtilde_t.slice(t - 1) = _Ct.slice(t - 1) + _Rtilde_right;
                }
                else if (is_iterative || use_pseudo)
                {
                    // use discount factor + iterative transFunc / sliding transFunc with pseudo inverse for Gt
                    _Gt = func_Gt(_model, _mt.col(t - 1), _y.at(t - 1));
                    arma::mat _Gt_inv;
                    if (is_iterative)
                    {
                        // Gt is invertible for iterative transfer function.
                        _Gt_inv = inverse(_Gt);
                    }
                    else
                    {
                        // Gt is not invertible for sliding transfer function, use pseudo inverse instead.
                        _Gt_inv = inverse(_Gt, true);
                    }

                    arma::vec _atilde_left = (1. - _discount_factor) * _mt.col(t - 1);
                    arma::vec _atilde_right = _discount_factor * _Gt_inv * _atilde_t.col(t);
                    _atilde_t.col(t - 1) = _atilde_left + _atilde_right;

                    arma::mat _Rtilde_left = (1. - _discount_factor) * _Ct.slice(t - 1);

                    double delta = _discount_factor;
                    arma::mat _Rtilde_right = (_Gt_inv * _Rtilde_t.slice(t)) * _Gt_inv.t();
                    _Rtilde_right.for_each([&delta](arma::mat::elem_type &val)
                                           { val *= delta * delta; });

                    _Rtilde_t.slice(t - 1) = _Rtilde_left + _Rtilde_right;
                }
                else
                {
                    // use discount factor + sliding transFunc and only consider psi[t], 1st elem of theta[t].
                    _atilde_t.col(t - 1).zeros();
                    _Rtilde_t.slice(t - 1).zeros();

                    double aleft = (1. - _discount_factor) * _mt.at(0, t - 1);
                    double aright = _discount_factor * _atilde_t.at(0, t);
                    _atilde_t.at(0, t - 1) = aleft + aright;

                    double rleft = (1. - _discount_factor) * _Ct.at(0, 0, t - 1);
                    double rright = std::pow(_discount_factor, 2.) * _Rtilde_t.at(0, 0, t);
                    _Rtilde_t.at(0, 0, t - 1) = rleft + rright;
                }

                _psi_mean.at(t - 1) = _atilde_t.at(0, t - 1);
                _psi_var.at(t - 1) = _Rtilde_t.at(0, 0, t - 1);
            }


            bound_check<arma::vec>(_psi_mean, "smoother: _psi_mean");
            bound_check<arma::vec>(_psi_var, "smoother: _psi_var");
        }
    
    private:
        const Model &_model;
        arma::vec _y;

        double _W = 0.01;
        double _discount_factor = 0.95;
        bool _use_discount = false;

        arma::mat _mt, _at, _atilde_t; // nP x (nT + 1)
        arma::cube _Ct, _Rt, _Rtilde_t; // nP x nP x (nT + 1)
        unsigned int _nT, _nP;

        arma::vec _psi_mean; // (nT + 1) x 1
        arma::vec _psi_var; // (nT + 1) x 1

        arma::vec _Ft, _At; // nP x 1
        arma::mat _Gt; // nP x nP
        double _alphat, _betat, _ft, _ft_prior_mean, _ft_prior_var, _ft_post_mean, _ft_post_var;
    };
}


#endif