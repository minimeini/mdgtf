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
        const arma::vec &yall,       // y[0], y[1], ..., y[nT]
        const bool &fill_zero = true
    )
    {
        arma::vec Ft_cur = model.transfer.F0;
        if (model.transfer.trans_list[model.transfer.name] == AVAIL::sliding)
        {
            // unsigned int nelem = std::min(t, model.dim.nL); // min(t,nL)
            int nstart = std::max((int) 0, (int) (t - model.dim.nL));
            unsigned int nend = std::max(model.dim.nL - 1, t - 1);
            unsigned int nelem = nend - (unsigned int) nstart + 1;

            arma::vec yold(model.dim.nL, arma::fill::zeros);
            yold.tail(nelem) = yall.subvec((unsigned int) nstart, nend);
            // yold.head(nelem) = yall.subvec(t - nelem, t - 1); // y[t - min(t,nL)], ..., y[t-1], 0, ..., 0 => 
            // yold = arma::reverse(yold);                       // y[t-1], ..., y[t-min(t,nL)]
            // arma::vec ytmp = yall.subvec(t - nelem, t - 1);

            // if (nelem > 1)
            // {
            //     yold.tail(nelem) = yall.subvec(t - nelem, t - 1);
            // }
            // else
            // {
            //     yold.at(model.dim.nL - 1) = yall.at(t - 1);
            // }

            if (fill_zero)
            {
                yold.elem(arma::find(yold <= EPS)).fill(0.01 / model.dim.nL);
            }
            
            yold = arma::reverse(yold);

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
        const arma::mat &Rt,
        const bool &fill_zero = true)
    {
        mean_ft = StateSpace::func_ft(model, t, at, yall);
        _Ft = func_Ft(model, t, at, yall, fill_zero);
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
        const arma::mat &at;
        const arma::cube &Rt;
        const arma::mat &mt;
        const arma::cube &Ct;
        const arma::mat &atilde;
        const arma::cube &Rtilde;
        const arma::vec &alpha_t;
        const arma::vec &beta_t;
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
        const double &theta_upbnd = 1.) : alpha_t(_alpha_t), beta_t(_beta_t), atilde(_atilde_t), Rtilde(_Rtilde_t), at(_at), Rt(_Rt), mt(_mt), Ct(_Ct), _model(model)
        {
            _W = W;
            _discount_factor = discount_factor;
            _use_discount = use_discount;

            _nP = model.dim.nP;
            _nT = model.dim.nT;

            double _ft = 0.;
            // _ft_prior_mean = _ft;
            // _ft_prior_var = _ft;
            // _ft_post_mean = _ft;
            // _ft_post_var = _ft;
            _alphat = 0.01;
            _betat = 0.01;

            _psi_mean.set_size(_nT + 1);
            _psi_mean.zeros();
            _psi_var = _psi_mean;

            _alpha_t = _psi_mean;
            _beta_t = _psi_mean;
            _alpha_t.at(0) = _alphat;
            _beta_t.at(0) = _betat;

            _ft_prior_mean = _psi_mean;
            _ft_prior_var = _psi_mean;
            _ft_post_mean = _psi_mean;
            _ft_post_var = _psi_mean;


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

        arma::mat optimal_discount_factor(
            const double &from, 
            const double &to, 
            const double &delta = 0.01)
        {
            bool use_discount_old = _use_discount;
            _use_discount = true;

            double discount_old = _discount_factor;


            if (to > 1 || from < 0)
            {
                throw std::invalid_argument("Discount factor range is (0, 1)");
            }
            arma::vec delta_grid = arma::regspace(from, delta, to);
            unsigned int nelem = delta_grid.n_elem;
            arma::mat y_loss(nelem, 2, arma::fill::zeros);

            for (unsigned int i = 0; i < nelem; i ++)
            {
                _discount_factor = delta_grid.at(i);
                y_loss.at(i, 0) = delta_grid.at(i);

                arma::vec ft_prior_mean(_model.dim.nT + 1, arma::fill::zeros);
                arma::vec ft_prior_var(_model.dim.nT + 1, arma::fill::zeros);

                filter_single_iter(1);

                for (unsigned int t = 2; t < _model.dim.nT; t++)
                {
                    _at.col(t) = StateSpace::func_gt(_model, _mt.col(t - 1), _y.at(t - 1)); // Checked. OK.
                    _Gt = func_Gt(_model, _mt.col(t - 1), _y.at(t - 1));
                    _Rt.slice(t) = func_Rt(_Gt, _Ct.slice(t - 1), _W, _use_discount, _discount_factor);

                    double ft_tmp, qt_tmp;
                    
                    func_prior_ft(
                        ft_tmp, qt_tmp, _Ft,
                        t, _model, _y,
                        _at.col(t), _Rt.slice(t), _fill_zero);
                    
                    ft_prior_mean.at(t) = ft_tmp;
                    ft_prior_var.at(t) = qt_tmp;

                    func_alpha_beta(_alphat, _betat, _model, ft_tmp, qt_tmp, _y.at(t), true);

                    _alpha_t.at(t) = _alphat;
                    _beta_t.at(t) = _betat;

                    _At = func_At(_Rt.slice(t), _Ft, qt_tmp);

                    double ft_tmp2, qt_tmp2;

                    func_posterior_ft(ft_tmp2, qt_tmp2, _model, _alphat, _betat);

                    _mt.col(t) = func_mt(_at.col(t), _At, ft_tmp, ft_tmp2);
                    _Ct.slice(t) = func_Ct(_Rt.slice(t), _At, qt_tmp, qt_tmp2);
                }

                y_loss.at(i, 1) = forecast_error(_y, ft_prior_mean, ft_prior_var);

            }

            _use_discount = use_discount_old;
            _discount_factor = discount_old;

            return y_loss;
        }



        void filter_single_iter(const unsigned int &t)
        {
            _at.col(t) = StateSpace::func_gt(_model, _mt.col(t - 1), _y.at(t - 1)); // Checked. OK.
            _Gt = func_Gt(_model, _mt.col(t - 1), _y.at(t - 1));
            _Rt.slice(t) = func_Rt(_Gt, _Ct.slice(t - 1), _W, _use_discount, _discount_factor);

            double ft_tmp = 0.;
            double qt_tmp = 0.;
            func_prior_ft(
                ft_tmp, qt_tmp, _Ft,
                t, _model, _y,
                _at.col(t), _Rt.slice(t), _fill_zero);
            _ft_prior_mean.at(t) = ft_tmp;
            _ft_prior_var.at(t) = qt_tmp;

            func_alpha_beta(_alphat, _betat, _model, ft_tmp, qt_tmp, _y.at(t), true);
            _alpha_t.at(t) = _alphat;
            _beta_t.at(t) = _betat;

            _At = func_At(_Rt.slice(t), _Ft, qt_tmp);

            double ft_tmp2 = 0.;
            double qt_tmp2 = 0.;
            func_posterior_ft(ft_tmp2, qt_tmp2, _model, _alphat, _betat);
            _ft_post_mean.at(t) = ft_tmp2;
            _ft_post_var.at(t) = qt_tmp2;

            _mt.col(t) = func_mt(_at.col(t), _At, ft_tmp, ft_tmp2);
            _Ct.slice(t) = func_Ct(_Rt.slice(t), _At, qt_tmp, qt_tmp2);
        }


        void filter()
        {
            unsigned int tstart = 1;
            if (_do_reference_analysis && (_model.dim.nL < _model.dim.nT))
            {
                filter_single_iter(1);
                for (unsigned int t = 2; t <= _model.dim.nL; t++)
                {
                    _at.col(t) = _at.col(1);
                    _Rt.slice(t) = _Rt.slice(1);
                    _mt.col(t) = _mt.col(1);
                    _Ct.slice(t) = _Ct.slice(1);
                    _alpha_t.at(t) = _alpha_t.at(1);
                    _beta_t.at(t) = _beta_t.at(1);
                }
                tstart = _model.dim.nL + 1;
            }


            for (unsigned int t = tstart; t <= _model.dim.nT; t++)
            {
                // filter_single_iter(t);
                _at.col(t) = StateSpace::func_gt(_model, _mt.col(t - 1), _y.at(t - 1)); // Checked. OK.
                _Gt = func_Gt(_model, _mt.col(t-1), _y.at(t-1));
                _Rt.slice(t) = func_Rt(_Gt, _Ct.slice(t - 1), _W, _use_discount, _discount_factor);

                double ft_tmp = 0.;
                double qt_tmp = 0.;
                func_prior_ft(
                    ft_tmp, qt_tmp, _Ft,
                    t, _model, _y,
                    _at.col(t), _Rt.slice(t), _fill_zero);
                _ft_prior_mean.at(t) = ft_tmp;
                _ft_prior_var.at(t) = qt_tmp;

                func_alpha_beta(_alphat, _betat, _model, ft_tmp, qt_tmp, _y.at(t), true);
                _alpha_t.at(t) = _alphat;
                _beta_t.at(t) = _betat;

                _At = func_At(_Rt.slice(t), _Ft, qt_tmp);

                double ft_tmp2 = 0.;
                double qt_tmp2 = 0.;
                func_posterior_ft(ft_tmp2, qt_tmp2, _model, _alphat, _betat);
                _ft_post_mean.at(t) = ft_tmp2;
                _ft_post_var.at(t) = qt_tmp2;

                _mt.col(t) = func_mt(_at.col(t), _At, ft_tmp, ft_tmp2);
                _Ct.slice(t) = func_Ct(_Rt.slice(t), _At, qt_tmp, qt_tmp2);
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


        /**
         * @brief Filtering fitted distribution: (y[t] | D[t]) ~ (ft post mean, ft post var); smoothing fitted distribution: y[t] | D[nT].
         * 
         * @return Rcpp::List 
         */
        Rcpp::List fitted_error(const unsigned int &nsample = 1000, const std::string &loss_func = "quadratic")
        {
            /*
            Filtering fitted distribution: (y[t] | D[t]) ~ (_ft_post_mean, _ft_post_var);
            
            */
            arma::mat res_filter(_model.dim.nT + 1, nsample, arma::fill::zeros);
            arma::mat yhat_filter(_model.dim.nT + 1, nsample, arma::fill::zeros);
            for (unsigned int t = 1; t <= _model.dim.nT; t ++)
            {
                arma::vec yhat = _ft_post_mean.at(t) + std::sqrt(_ft_post_var.at(t)) * arma::randn(nsample);
                arma::vec res = _y.at(t) - yhat;

                yhat_filter.row(t) = yhat.t();
                res_filter.row(t) = res.t();
            }

            arma::vec y_loss(_model.dim.nT + 1, arma::fill::zeros);
            double y_loss_all = 0.;
            std::map<std::string, AVAIL::Loss> loss_list = AVAIL::loss_list;
            switch (loss_list[tolower(loss_func)])
            {
            case AVAIL::L1: // mae
            {
                arma::mat ytmp = arma::abs(res_filter);
                y_loss = arma::mean(ytmp, 1);
                y_loss_all = arma::mean(y_loss.tail(_model.dim.nT - 1));
                break;
            }
            case AVAIL::L2: // rmse
            {
                arma::mat ytmp = arma::square(res_filter);
                y_loss = arma::mean(ytmp, 1);
                y_loss_all = arma::mean(y_loss.tail(_model.dim.nT - 1));
                y_loss = arma::sqrt(y_loss);
                y_loss_all = std::sqrt(y_loss_all);
                break;
            }
            default:
            {
                break;
            }
            }

            Rcpp::List out_filter;
            arma::vec qprob = {0.025, 0.5, 0.975};
            arma::mat yhat_out = arma::quantile(yhat_filter, qprob, 1);

            out_filter["yhat"] = Rcpp::wrap(yhat_out);
            out_filter["yhat_all"] = Rcpp::wrap(yhat_filter);
            out_filter["residual"] = Rcpp::wrap(res_filter);
            out_filter["y_loss"] = Rcpp::wrap(y_loss);
            out_filter["y_lost_all"] = y_loss_all;

            /*
            Smoothing fitted distribution: (y[t] | D[nT]) = int (y[t] | theta[t]) (theta[t] | D[nT]) d theta[t], where (theta[t] | D[nT]) ~ (atilde[t], Rtilde[t])
            */
            arma::vec psi_mean = arma::vectorise(_atilde_t.row(0)); // (nT + 1) x 1
            arma::vec psi_var = arma::vectorise(_Rtilde_t.tube(0, 0)); // (nT + 1) x 1
            arma::vec psi_sd = arma::sqrt(psi_var);                    // (nT + 1) x 1

            arma::mat psi_sample(_model.dim.nT + 1, nsample, arma::fill::zeros);
            for (unsigned int i = 0; i < nsample; i ++)
            {
                arma::vec psi_tmp = psi_mean + psi_sd % arma::randn(_model.dim.nT + 1); // (nT + 1) x 1
                psi_sample.col(i) = psi_tmp;
            }

            Rcpp::List out_smooth = Model::fitted_error(psi_sample, _y, _model, loss_func);

            Rcpp::List out;
            out["filter"] = out_filter;
            out["smooth"] = out_smooth;
            return out;
        }


        /**
         * @brief Forecasting distribution y[t + 1] | D[t] ~ (ft prior mean, ft prior var)
         * 
         * @return Rcpp::List 
         */
        Rcpp::List forecast_error(const unsigned int &nsample = 1000, const std::string &loss_func = "quadratic")
        {
            arma::mat ycast(_model.dim.nT + 1, nsample, arma::fill::zeros);
            arma::mat y_err_cast(_model.dim.nT + 1, nsample, arma::fill::zeros);
            for (unsigned int t = 2; t <= _model.dim.nT; t ++)
            {
                arma::vec ytmp = _ft_prior_mean.at(t) + std::sqrt(_ft_prior_var.at(t)) * arma::randn(nsample);
                ycast.row(t) = ytmp.t();

                y_err_cast.row(t) = _y.at(t) - ycast.row(t);
            }

            arma::vec y_loss(_model.dim.nT + 1, arma::fill::zeros);
            double y_loss_all = 0;

            std::map<std::string, AVAIL::Loss> loss_list = AVAIL::loss_list;
            switch (loss_list[tolower(loss_func)])
            {
            case AVAIL::L1: // mae
            {
                arma::mat ytmp = arma::abs(y_err_cast);
                y_loss = arma::mean(ytmp, 1);
                y_loss_all = arma::mean(y_loss.tail(_model.dim.nT - 1));
                break;
            }
            case AVAIL::L2: // rmse
            {
                arma::mat ytmp = arma::square(y_err_cast);
                y_loss = arma::mean(ytmp, 1);
                y_loss_all = arma::mean(y_loss.tail(_model.dim.nT - 1));
                y_loss = arma::sqrt(y_loss);
                y_loss_all = std::sqrt(y_loss_all);
                break;
            }
            default:
            {
                break;
            }
            }

            Rcpp::List out;

            arma::vec qprob = {0.025, 0.5, 0.975};
            arma::mat ycast_out = arma::quantile(ycast, qprob, 1);
            out["y_cast"] = Rcpp::wrap(ycast_out);
            out["y_cast_all"] = Rcpp::wrap(ycast);
            out["y"] = Rcpp::wrap(_y);
            out["y_loss"] = Rcpp::wrap(y_loss);
            out["y_loss_all"] = y_loss_all;

            return out;
        }

        static double forecast_error(
            const arma::vec &y,
            const arma::vec &ft_prior_mean, 
            const arma::vec &ft_prior_var,
            const unsigned int &nsample = 1000, 
            const std::string &loss_func = "quadratic")
        {
            unsigned int nT = y.n_elem - 1;
            arma::mat ycast(nT + 1, nsample, arma::fill::zeros);
            arma::mat y_err_cast(nT + 1, nsample, arma::fill::zeros);
            for (unsigned int t = 2; t <= nT; t++)
            {
                arma::vec ytmp = ft_prior_mean.at(t) + std::sqrt(ft_prior_var.at(t)) * arma::randn(nsample);
                ycast.row(t) = ytmp.t();

                y_err_cast.row(t) = y.at(t) - ycast.row(t);
            }

            arma::vec y_loss(nT + 1, arma::fill::zeros);
            double y_loss_all = 0;

            std::map<std::string, AVAIL::Loss> loss_list = AVAIL::loss_list;
            switch (loss_list[tolower(loss_func)])
            {
            case AVAIL::L1: // mae
            {
                arma::mat ytmp = arma::abs(y_err_cast);
                y_loss = arma::mean(ytmp, 1);
                y_loss_all = arma::mean(y_loss.tail(nT - 1));
                break;
            }
            case AVAIL::L2: // rmse
            {
                arma::mat ytmp = arma::square(y_err_cast);
                y_loss = arma::mean(ytmp, 1);
                y_loss_all = arma::mean(y_loss.tail(nT - 1));
                y_loss = arma::sqrt(y_loss);
                y_loss_all = std::sqrt(y_loss_all);
                break;
            }
            default:
            {
                break;
            }
            }

            return y_loss_all;
        }

        void init(const Rcpp::List &opts_in)
        {
            opts = opts_in;
            _W = 0.01;
            if (opts.containsElementNamed("W"))
            {
                _W = Rcpp::as<double>(opts["W"]);
            }

            _use_discount = false;
            if (opts.containsElementNamed("use_discount"))
            {
                _use_discount = Rcpp::as<bool>(opts["use_discount"]);
            }

            _discount_factor = 0.95;
            if (opts.containsElementNamed("custom_discount_factor"))
            {
                _discount_factor = Rcpp::as<double>(opts["custom_discount_factor"]);
            }

            _do_reference_analysis = false;
            if (opts.containsElementNamed("do_reference_analysis"))
            {
                _do_reference_analysis = Rcpp::as<bool>(opts["do_reference_analysis"]);
            }

            _fill_zero = true;
            if (opts.containsElementNamed("fill_zero"))
            {
                _fill_zero = Rcpp::as<bool>(opts["fill_zero"]);
            }
        }

        static Rcpp::List default_settings()
        {
            Rcpp::List opts;

            opts["W"] = 0.01;
            opts["use_discount"] = false;
            opts["use_custom"] = false;
            opts["custom_discount_factor"] = 0.95;
            opts["do_smoothing"] =  true;
            opts["do_reference_analysis"] = false;
            opts["fill_zero"] = true;

            return opts;
        }

        Rcpp::List get_output()
        {
            Rcpp::List output;
            output["opts"] = opts;
            arma::mat psi = get_psi(_atilde_t, _Rtilde_t);
            arma::mat psi_filter = get_psi(_mt, _Ct);
            output["psi"] = Rcpp::wrap(psi);
            output["psi_filter"] = Rcpp::wrap(psi_filter);

            return output;
        }
    
    private:
        const Model &_model;
        Rcpp::List opts;
        arma::vec _y;

        double _W = 0.01;
        double _discount_factor = 0.95;

        bool _use_discount = false;
        bool _do_reference_analysis = false;
        bool _fill_zero = true;

        arma::mat _mt, _at, _atilde_t; // nP x (nT + 1)
        arma::cube _Ct, _Rt, _Rtilde_t; // nP x nP x (nT + 1)
        unsigned int _nT, _nP;

        arma::vec _psi_mean; // (nT + 1) x 1
        arma::vec _psi_var; // (nT + 1) x 1
        arma::vec _alpha_t, _beta_t; // (nT + 1) x 1
        arma::vec _ft_prior_mean; // (nT + 1) x 1, one-step-ahead forecasting distribution of y
        arma::vec _ft_prior_var;  // (nT + 1) x 1, one-step-ahead forecasting distribution of y
        arma::vec _ft_post_mean;  // (nT + 1) x 1, fitted distribution based on filtering states
        arma::vec _ft_post_var; // (nT + 1) x 1, fitted distribution based on filtering states

        arma::vec _Ft, _At; // nP x 1
        arma::mat _Gt; // nP x nP
        double _alphat, _betat, _ft;
        // , _ft_prior_mean, _ft_prior_var, _ft_post_mean, _ft_post_var;
    };
}


#endif