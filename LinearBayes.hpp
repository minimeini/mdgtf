#ifndef _LINEARBAYES_H
#define _LINEARBAYES_H

#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <algorithm>
#include <RcppArmadillo.h>
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
    enum DiscountType
    {
        all_elems,
        all_lag_elems,
        first_elem
    };

    std::map<std::string, DiscountType> map_discount_type()
    {
        std::map<std::string, DiscountType> map;
        map["all_elems"] = DiscountType::all_elems;
        map["all_lag_elems"] = DiscountType::all_lag_elems;
        map["first_elem"] = DiscountType::first_elem;
        return map;
    }

    /**
     * @brief First-order derivative of g[t] w.r.t theta[t]; used in the calculation of R[t] = G[t] C[t-1] t(G[t]) + W[t].
     *
     * @param model
     * @param mt_old
     * @param yold
     * @return arma::mat
     */
    static void func_Gt(
        arma::mat &Gt, // nP x nP, must be already initialized via `TransFunc::init_Gt()`
        const Model &model,
        const arma::vec &mt_old,
        const double &yold)
    {
        std::map<std::string, TransFunc::Transfer> trans_list = TransFunc::trans_list;
        if (trans_list[model.ftrans] == TransFunc::Transfer::iterative)
        {
            double coef_now = std::pow(1. - model.dlag.par1, model.dlag.par2);
            double dhpsi_now = GainFunc::psi2dhpsi(
                mt_old.at(0), // h'(psi[t])
                model.fgain);
            Gt.at(1, 0) = coef_now * yold * dhpsi_now;
        }

        return;
    }


    // static void func_Ht(
    //     arma::mat &Ht, // np x nP, must be initialized via `TransFunc::H0_sliding()`
    //     const Model &model)
    // {
    //     std::map<std::string, AVAIL::Transfer> trans_list = AVAIL::trans_list;
    //     // arma::mat Ht = model.transfer.H0;
    //     if (trans_list[model.ftrans] == AVAIL::Transfer::iterative)
    //     {
    //         throw std::invalid_argument("Ht for iterative transfer function: not defined yet.");
    //     }

    //     return;
    // }


    /**
     * @brief Variance of the one-step-ahead forecasting distribution of latent state, theta[t] | D[t-1].
     *
     * @param Gt nP x nP; First-order derivative of evolution function gt.
     * @param Ct_old nP x nP; Posterior variance of latent state, theta[t-1] | D[t-1]
     * @param W double; Univariate variance of latent random walk process.
     * @param use_discount bool;
     * @param delta_discount double in range (0, 1]
     * @param discount_type string {'first_elem', 'all_elems'}
     * @return arma::mat
     */
    static arma::mat func_Rt(
        const arma::mat &Gt,
        const arma::mat &Ct_old,
        const double &W = 0.01,
        const bool &use_discount = false,
        const double &delta_discount = 0.95,
        const std::string &discount_type = "first_elem")
    {
        arma::mat Rt = Gt * Ct_old * Gt.t(); // Pt
        std::map<std::string, DiscountType> discount_list = map_discount_type();

        if (use_discount)
        {
            switch (discount_list[tolower(discount_type)])
            {
            case DiscountType::first_elem:
            {
                /*
                The discounted Rt is invertible for iterative transfer function (almost guaranteed) and hopefully invertible for sliding transfer function.

                 */
                Rt.at(0, 0) /= delta_discount;
                break;
            }
            case DiscountType::all_elems:
            {
                // A unknown but general W[t] (could have non-zero off-diagonal values) with discount factor
                /*
                The discounted Rt is not invertible for sliding transfer function.
                */
                Rt.for_each([&delta_discount](arma::mat::elem_type &val)
                            { val /= delta_discount; });
                break;
            }
            default:
            {
                throw std::invalid_argument("Unspecified discounting scheme.");
            }
            }
            
        }
        else
        {
            // W[t] = diag(W, 0, ..., 0)
            Rt.at(0, 0) += W;
        }

        bound_check<arma::mat>(Rt, "func_Rt: Rt");
        return Rt;
    }

    static arma::mat Rt2Wt(
        const arma::mat &Rt,
        const double &W = 0.01,
        const bool &use_discount = false,
        const double &delta_discount = 0.95,
        const std::string &discount_type = "all_lag_elems")
    {
        arma::mat Wt(Rt.n_rows, Rt.n_cols, arma::fill::zeros);
        std::map<std::string, DiscountType> discount_list = map_discount_type();

        if (use_discount)
        {
            switch (discount_list[tolower(discount_type)])
            {
            case DiscountType::first_elem:
            {
                Wt.at(0, 0) = (1. - delta_discount) * Rt.at(0, 0);
                break;
            }
            case DiscountType::all_elems:
            {
                // A unknown but general W[t] (could have non-zero off-diagonal values) with discount factor
                Wt = (1. - delta_discount) * Rt;
                break;
            }
            default:
            {
                Wt = (1. - delta_discount) * Rt;
                break;
            }
            }
        }
        else
        {
            // W[t] = diag(W, 0, ..., 0)
            Wt.at(0, 0) = W;
        }

        bound_check<arma::mat>(Wt, "Rt2Wt: Wt");
        return Wt;
    }

    /**
     * @brief First-order derivative of f[t] w.r.t theta[t].
     *
     * @param model
     * @param t time index of theta_cur
     * @param theta_cur theta[t] = (psi[t], ..., psi[t+1 - nL]) with sliding transfer function or (psi[t+1], f[t], ..., f[t+1-r]) with iterative transfer function
     * @param yall y[0], y[1], ..., y[nT]
     * @return arma::vec
     */
    static arma::vec func_Ft(
        const std::string &ftrans,
        const std::string &fgain,
        const LagDist &dlag,
        const unsigned int &t,      // time index of theta_cur, t = 0, y[0] = 0, theta[0] = 0; t = 1, y[1], theta[1]; ...
        const arma::vec &theta_cur, // nP x 1, theta[t] = (psi[t], ..., psi[t+1 - nL]) or (psi[t+1], f[t], ..., f[t+1-r])
        const arma::vec &yall,      // y[0], y[1], ..., y[nT]
        const bool &fill_zero = LBA_FILL_ZERO,
        const unsigned int &seasonal_period = 0,
        const bool &season_in_state = false)
    {
        std::map<std::string, TransFunc::Transfer> trans_list = TransFunc::trans_list;
        arma::vec Ft(theta_cur.n_elem, arma::fill::zeros);
        if (trans_list[ftrans] == TransFunc::sliding)
        {
            unsigned int nstart = (t > dlag.nL) ? (t - dlag.nL) : 0;
            // unsigned int nstart = std::max((unsigned int)0, t - nL);
            unsigned int nend = std::max(dlag.nL - 1, t - 1);
            unsigned int nelem = nend - nstart + 1;

            arma::vec yold(dlag.nL, arma::fill::zeros);
            yold.tail(nelem) = yall.subvec(nstart, nend);

            if (fill_zero)
            {
                yold.elem(arma::find(yold <= EPS)).fill(0.01 / static_cast<double>(dlag.nL));
            }
            
            yold = arma::reverse(yold);

            arma::vec th = theta_cur.head(dlag.nL); // nL x 1
            arma::vec dhpsi_cur = GainFunc::psi2dhpsi<arma::vec>(th, fgain); // (h'(psi[t]), ..., h'(psi[t+1 - nL]))
            arma::vec Ftmp = yold % dhpsi_cur; // nL x 1
            Ft.head(dlag.nL) = Ftmp % dlag.Fphi;
        }
        else
        {
            Ft.at(1) = 1.;
        }

        if (seasonal_period > 0 && season_in_state)
        {
            unsigned int nstate = theta_cur.n_elem - seasonal_period;
            Ft.at(nstate) = 1.;
        }

        bound_check<arma::vec>(Ft, "func_Ft: Ft");
        return Ft;
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
        const bool &fill_zero = LBA_FILL_ZERO)
    {
        mean_ft = StateSpace::func_ft(model.ftrans, model.fgain, model.dlag, model.seas, t, at, yall);
        _Ft = func_Ft(model.ftrans, model.fgain, model.dlag, t, at, yall, fill_zero, model.seas.period, model.seas.in_state);
        var_ft = arma::as_scalar(_Ft.t() * Rt * _Ft);
        bound_check(var_ft, "LBA::func_prior_ft: var_ft", true, true);
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
        std::map<std::string, LinkFunc::Func> link_list = LinkFunc::link_list;
        std::map<std::string, AVAIL::Dist> obs_list = ObsDist::obs_list;

        double regressor = ft;

        switch (obs_list[model.dobs.name])
        {
        case AVAIL::Dist::poisson:
        {
            switch (link_list[model.flink])
            {
            case LinkFunc::Func::identity:
            {
                alpha = std::pow(regressor, 2.) / qt;
                beta = regressor / qt;
                break;
            }
            case LinkFunc::Func::exponential:
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
            if (link_list[model.flink] == AVAIL::Func::identity)
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
        std::map<std::string, LinkFunc::Func> link_list = LinkFunc::link_list;
        std::map<std::string, AVAIL::Dist> obs_list = ObsDist::obs_list;

        switch (obs_list[model.dobs.name])
        {
        case AVAIL::Dist::poisson:
        {
            switch (link_list[model.flink])
            {
            case LinkFunc::Func::identity:
            {
                Poisson::moments_mean(mean_ft, var_ft, alpha, beta, 0.);
                break;
            }
            case LinkFunc::Func::exponential:
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
            if (link_list[model.flink] == AVAIL::Func::identity)
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
        LinearBayes(
            const bool &use_discount_factor = false, 
            const double &discount_factor_value = 0.95)
        {
            discount_factor = discount_factor_value;
            use_discount = use_discount_factor;

            discount_type = "first_elem"; // all_lag_elems, all_elems, first_elem
            do_reference_analysis = false;
            fill_zero = LBA_FILL_ZERO;
            return;
        }

        LinearBayes(const Rcpp::List &opts)
        {
            init(opts);
            return;
        }

        void init(const Rcpp::List &opts_in)
        {
            Rcpp::List opts = opts_in;
            use_discount = false;
            if (opts.containsElementNamed("use_discount"))
            {
                use_discount = Rcpp::as<bool>(opts["use_discount"]);
            }

            discount_factor = 0.95;
            if (opts.containsElementNamed("custom_discount_factor"))
            {
                discount_factor = Rcpp::as<double>(opts["custom_discount_factor"]);
            }

            do_reference_analysis = false;
            if (opts.containsElementNamed("do_reference_analysis"))
            {
                do_reference_analysis = Rcpp::as<bool>(opts["do_reference_analysis"]);
            }

            fill_zero = LBA_FILL_ZERO;
            if (opts.containsElementNamed("fill_zero"))
            {
                fill_zero = Rcpp::as<bool>(opts["fill_zero"]);
            }

            discount_type = "first_elem";
            if (opts.containsElementNamed("discount_type"))
            {
                discount_type = Rcpp::as<std::string>(opts["discount_type"]);
            }
        }

        static Rcpp::List default_settings()
        {
            Rcpp::List opts;

            opts["use_discount"] = false;
            opts["discount_type"] = "first_elem";
            opts["use_custom"] = false;
            opts["custom_discount_factor"] = 0.95;
            opts["do_smoothing"] = true;
            opts["do_reference_analysis"] = false;
            opts["fill_zero"] = LBA_FILL_ZERO;

            return opts;
        }

        Rcpp::List get_output(const Model &model)
        {
            Rcpp::List output;
            arma::mat psi = get_psi(atilde, Rtilde);
            arma::mat psi_filter = get_psi(mt, Ct);
            output["psi"] = Rcpp::wrap(psi);
            output["psi_filter"] = Rcpp::wrap(psi_filter);

            output["nlag"] = model.dlag.nL;
            output["seasonal_period"] = model.seas.period;

            if (model.seas.period > 0 && model.seas.in_state)
            {
                arma::mat seas = atilde.tail_rows(model.seas.period);
                output["seasonality"] = Rcpp::wrap(seas);
            }
            else
            {
                output["seasonality"] = Rcpp::wrap(model.seas.val.t());
            }

            #ifdef DGTF_DETAILED_OUTPUT
            output["mt"] = Rcpp::wrap(mt);
            output["Ct"] = Rcpp::wrap(Ct);
            output["at"] = Rcpp::wrap(at);
            output["Rt"] = Rcpp::wrap(Rt);
            output["atilde"] = Rcpp::wrap(atilde);
            output["Rtilde"] = Rcpp::wrap(Rtilde);
            #endif

            return output;
        }


        arma::cube get_Wt(const Model &model, const arma::vec &y, const double &discount_factor)
        {
            arma::cube Wt = arma::zeros<arma::cube>(model.nP, model.nP, y.n_elem);
            arma::mat Gt = TransFunc::init_Gt(
                model.nP, model.dlag, model.ftrans,
                model.seas.period, model.seas.in_state);

            if (model.derr.full_rank)
            {
                Wt.slice(0) = arma::symmatu(Ct.slice(0));
            }
            else
            {
                Wt.at(0, 0, 0) = std::abs(Ct.at(0, 0, 0)) + EPS;
            }

            for (unsigned int t = 1; t < y.n_elem; t++)
            {
                LBA::func_Gt(Gt, model, mt.col(t - 1), y.at(t - 1));
                arma::mat Pt = Gt * Ct.slice(t - 1) * Gt.t();
                arma::mat Wt_hat = (1. / discount_factor - 1.) * Pt;
                Wt_hat.diag() += EPS8;

                if (model.derr.full_rank)
                {
                    Wt.slice(t) = arma::symmatu(Wt_hat);
                }
                else
                {
                    Wt.at(0, 0, t) = std::abs(Wt_hat.at(0, 0)) + EPS;
                }
            }

            return Wt;
        }

        arma::mat optimal_discount_factor(
            const Model &model,
            const arma::vec &y,
            const double &from, 
            const double &to, 
            const double &delta = 0.01,
            const std::string &loss = "quadratic",
            const bool &verbose = VERBOSE)
        {
            bool use_discount_old = use_discount;
            use_discount = true;

            double discount_old = discount_factor;


            if (to > 1 || from < 0)
            {
                throw std::invalid_argument("Discount factor range is (0, 1)");
            }
            arma::vec delta_grid = arma::regspace(from, delta, to);
            unsigned int nelem = delta_grid.n_elem;
            arma::mat stats(nelem, 4, arma::fill::zeros);


            for (unsigned int i = 0; i < nelem; i ++)
            {
                Rcpp::checkUserInterrupt();

                discount_factor = delta_grid.at(i);
                stats.at(i, 0) = delta_grid.at(i);

                Model mod = model;
                filter(mod, y);
                double forecast_err = 0.;
                double forecast_cover = 0.;
                double forecast_width = 0.;
                forecast_error(mod, y, forecast_err, forecast_cover, forecast_width, 1000, loss);
                stats.at(i, 1) = forecast_err;

                // smoother();

                // double fit_err = 0.;
                // fitted_error(fit_err, 1000, loss);
                stats.at(i, 2) = forecast_cover;
                stats.at(i, 3) = forecast_width;

                if (verbose)
                {
                    Rcpp::Rcout << "\rProgress: " << i + 1 << "/" << nelem;
                }
            }

            if (verbose)
            {
                Rcpp::Rcout << std::endl;
            }

            use_discount = use_discount_old;
            discount_factor = discount_old;

            return stats;
        }

        arma::mat optimal_W(
            const Model &model, 
            const arma::vec &y,
            const arma::vec &grid,
            const std::string &loss = "quadratic",
            const bool &verbose = VERBOSE)
        {
            use_discount = false;
            unsigned int nelem = grid.n_elem;
            arma::mat stats(nelem, 4, arma::fill::zeros);

            for (unsigned int i = 0; i < nelem; i++)
            {
                Rcpp::checkUserInterrupt();
                Model mod = model;
                mod.derr.par1 = grid.at(i);
                stats.at(i, 0) = grid.at(i);

                filter(mod, y);
                double forecast_err = 0.;
                double forecast_cover = 0.;
                double forecast_width = 0.;
                forecast_error(mod, y, forecast_err, forecast_cover, forecast_width, 1000, loss);
                stats.at(i, 1) = forecast_err;

                // smoother();
                // double fit_err = 0.;
                // fitted_error(fit_err, 1000, loss);
                stats.at(i, 2) = forecast_cover;
                stats.at(i, 3) = forecast_width;

                if (verbose)
                {
                    Rcpp::Rcout << "\rProgress: " << i + 1 << "/" << nelem;
                }
            }

            if (verbose)
            {
                Rcpp::Rcout << std::endl;
            }

            return stats;
        }

        static void filter_single_iter(
            arma::vec &mt_new,
            arma::mat &Ct_new,
            arma::vec &at_new,
            arma::mat &Rt_new,
            arma::vec &Ft,
            arma::mat &Gt,
            double &ft_prior,
            double &qt_prior,
            double &ft_posterior,
            double &qt_posterior,
            const unsigned int &t,
            const arma::vec &y,
            const Model &model, 
            const arma::vec &mt_old,
            const arma::mat &Ct_old,
            const bool &use_discount,
            const double &discount_factor,
            const std::string &discount_type)
        {
            at_new = StateSpace::func_gt(
                model.ftrans, model.fgain, model.dlag, 
                mt_old, y.at(t - 1), 
                model.seas.period, model.seas.in_state); // Checked. OK.
            
            func_Gt(Gt, model, mt_old, y.at(t - 1));
            Rt_new = func_Rt(
                Gt, Ct_old, model.derr.par1, 
                use_discount, discount_factor,
                discount_type);

            ft_prior = 0.;
            qt_prior = 0.;
            func_prior_ft(
                ft_prior, qt_prior, Ft,
                t, model, y,
                at_new, Rt_new, LBA_FILL_ZERO);

            double alpha = 0.;
            double beta = 0.;
            func_alpha_beta(alpha, beta, model, ft_prior, qt_prior, y.at(t), true);

            arma::mat At = func_At(Rt_new, Ft, qt_prior);

            ft_posterior = 0.;
            qt_posterior = 0.;
            func_posterior_ft(ft_posterior, qt_posterior, model, alpha, beta);

            mt_new = func_mt(at_new, At, ft_prior, ft_posterior);
            Ct_new = func_Ct(Rt_new, At, qt_prior, qt_posterior);

            return;
        }

        void filter(Model &model, const arma::vec &y, const double &theta_upbnd = 1.)
        {
            const unsigned int nT = y.n_elem - 1;
            arma::vec Ft = TransFunc::init_Ft(model.nP, model.ftrans, model.seas.period, model.seas.in_state);             // set F[0]
            arma::mat Gt = TransFunc::init_Gt(model.nP, model.dlag, model.ftrans, model.seas.period, model.seas.in_state); // set G[0]

            ft_prior_mean.set_size(nT + 1);
            ft_prior_mean.zeros();
            ft_prior_var = ft_prior_mean;
            ft_post_mean = ft_prior_mean;
            ft_post_var = ft_prior_mean;

            at.set_size(model.nP, nT + 1);
            at.zeros();
            if (model.seas.period > 0 && model.seas.in_state)
            {
                at.submat(model.nP - model.seas.period, 0, model.nP - 1, 0) = arma::randu(model.seas.period, arma::distr_param(0, 10));
            }
            mt = at;
            // set m[0]

            Ct = arma::zeros<arma::cube>(model.nP, model.nP, nT + 1);
            Ct.slice(0) = arma::eye<arma::mat>(model.nP, model.nP);
            Ct.for_each([&theta_upbnd](arma::cube::elem_type &val)
                         { val *= theta_upbnd; });

            Rt = arma::zeros<arma::cube>(model.nP, model.nP, nT + 1);

            unsigned int tstart = 1;
            if (do_reference_analysis && (model.dlag.nL < nT))
            {
                arma::vec mt_new, at_new;
                arma::mat Ct_new, Rt_new;
                double ft_prior, qt_prior, ft_posterior, qt_posterior;
                filter_single_iter(
                    mt_new, Ct_new, at_new, Rt_new, Ft, Gt,
                    ft_prior, qt_prior, ft_posterior, qt_posterior,
                    1, y, model, mt.col(0), Ct.slice(0), 
                    use_discount, discount_factor, discount_type);

                at.col(1) = at_new;
                Rt.slice(1) = arma::symmatu(Rt_new);
                mt.col(1) = mt_new;
                Ct.slice(1) = arma::symmatu(Ct_new);

                ft_prior_mean.at(1) = ft_prior;
                ft_prior_var.at(1) = qt_prior;
                ft_post_mean.at(1) = ft_posterior;
                ft_post_var.at(1) = qt_posterior;

                for (unsigned int t = 2; t <= model.dlag.nL; t++)
                {
                    at.col(t) = at.col(1);
                    Rt.slice(t) = Rt.slice(1);
                    mt.col(t) = mt.col(1);
                    Ct.slice(t) = Ct.slice(1);

                    ft_prior_mean.at(t) = ft_prior;
                    ft_prior_var.at(t) = qt_prior;
                    ft_post_mean.at(t) = ft_posterior;
                    ft_post_var.at(t) = qt_posterior;
                }

                tstart = model.dlag.nL + 1;
            }


            for (unsigned int t = tstart; t <= nT; t++)
            {
                arma::vec mt_new, at_new;
                arma::mat Ct_new, Rt_new;
                double ft_prior, qt_prior, ft_posterior, qt_posterior;
                filter_single_iter(
                    mt_new, Ct_new, at_new, Rt_new, Ft, Gt,
                    ft_prior, qt_prior, ft_posterior, qt_posterior,
                    t, y, model, mt.col(t-1), Ct.slice(t-1), 
                    use_discount, discount_factor, discount_type);

                at.col(t) = at_new;
                Rt.slice(t) = arma::symmatu(Rt_new);
                mt.col(t) = mt_new;
                Ct.slice(t) = arma::symmatu(Ct_new);

                ft_prior_mean.at(t) = ft_prior;
                ft_prior_var.at(t) = qt_prior;
                ft_post_mean.at(t) = ft_posterior;
                ft_post_var.at(t) = qt_posterior;
            }

            

            return;
        }


        void smoother(Model &model, const arma::vec &y, const bool &use_pseudo = false)
        {
            const unsigned int nT = y.n_elem - 1;
            arma::vec Ft = TransFunc::init_Ft(model.nP, model.ftrans, model.seas.period, model.seas.in_state);             // set F[0]
            arma::mat Gt = TransFunc::init_Gt(model.nP, model.dlag, model.ftrans, model.seas.period, model.seas.in_state); // set G[0]

            std::map<std::string, TransFunc::Transfer> trans_list = TransFunc::trans_list;
            bool is_iterative = trans_list[model.ftrans] == TransFunc::Transfer::iterative;
            std::map<std::string, DiscountType> discount_list = map_discount_type();
            bool is_first_elem_discount = discount_list[discount_type] == DiscountType::first_elem;

            atilde.set_size(model.nP, nT + 1);
            atilde.zeros();
            atilde.col(nT) = mt.col(nT);

            Rtilde.set_size(model.nP, model.nP, nT + 1);
            Rtilde.zeros();
            Rtilde.slice(nT) = Ct.slice(nT);

            for (unsigned int t = nT; t > 0; t--)
            {
                if (!use_discount)
                {
                    arma::mat Rtinv = inverse(Rt.slice(t));

                    func_Gt(Gt, model, mt.col(t - 1), y.at(t - 1));
                    arma::mat Bt = (Ct.slice(t - 1) * Gt.t()) * Rtinv;

                    arma::vec diff_a = atilde.col(t) - at.col(t);
                    arma::mat diff_R = Rtilde.slice(t) - Rt.slice(t);

                    arma::vec atilde_right = Bt * diff_a;
                    atilde.col(t - 1) = mt.col(t - 1) + atilde_right;

                    arma::mat Rtilde_right = (Bt * diff_R) * Bt.t();
                    Rtilde.slice(t - 1) = Ct.slice(t - 1) + Rtilde_right;
                }
                else if (is_iterative || (use_pseudo && !is_first_elem_discount))
                {
                    // use discount factor + iterative transFunc / sliding transFunc with pseudo inverse for Gt
                    func_Gt(Gt, model, mt.col(t - 1), y.at(t - 1));
                    arma::mat Gt_inv;
                    if (is_iterative)
                    {
                        /*
                        Gt is invertible but asymmetric for iterative transfer function.
                        */
                        Gt_inv = arma::inv(Gt);
                    }
                    else
                    {
                        // Gt is not invertible for sliding transfer function, use pseudo inverse instead.
                        Gt_inv = arma::pinv(Gt);
                    }

                    arma::vec atilde_left = (1. - discount_factor) * mt.col(t - 1);
                    arma::vec atilde_right = discount_factor * Gt_inv * atilde.col(t);
                    atilde.col(t - 1) = atilde_left + atilde_right;

                    arma::mat Rtilde_left = (1. - discount_factor) * Ct.slice(t - 1);

                    double delta = discount_factor;
                    arma::mat Rtilde_right = (Gt_inv * Rtilde.slice(t)) * Gt_inv.t();
                    Rtilde_right.for_each([&delta](arma::mat::elem_type &val)
                                           { val *= delta * delta; });

                    Rtilde.slice(t - 1) = Rtilde_left + Rtilde_right;
                }
                else
                {
                    // use discount factor + sliding transFunc and only consider psi[t], 1st elem of theta[t].
                    atilde.col(t - 1).zeros();
                    Rtilde.slice(t - 1).zeros();

                    double aleft = (1. - discount_factor) * mt.at(0, t - 1);
                    double aright = discount_factor * atilde.at(0, t);
                    atilde.at(0, t - 1) = aleft + aright;
                    
                    double rleft = (1. - discount_factor) * Ct.at(0, 0, t - 1);
                    double rright = std::pow(discount_factor, 2.) * Rtilde.at(0, 0, t);
                    Rtilde.at(0, 0, t - 1) = rleft + rright;
                }

            }
        }


        /**
         * @brief Filtering fitted distribution: (y[t] | D[t]) ~ (ft post mean, ft post var); smoothing fitted distribution: y[t] | D[nT].
         * 
         * @return Rcpp::List 
         */
        Rcpp::List fitted_error(const Model &model, const arma::vec &y, const unsigned int &nsample = 1000, const std::string &loss_func = "quadratic")
        {
            const unsigned int nT = y.n_elem - 1;
            /*
            Filtering fitted distribution: (y[t] | D[t]) ~ (_ft_post_mean, _ft_post_var);

            */
            arma::mat res_filter(nT + 1, nsample, arma::fill::zeros);
            arma::mat yhat_filter(nT + 1, nsample, arma::fill::zeros);
            for (unsigned int t = 1; t <= nT; t ++)
            {
                arma::vec yhat = ft_post_mean.at(t) + std::sqrt(ft_post_var.at(t)) * arma::randn(nsample);
                arma::vec res = y.at(t) - yhat;

                yhat_filter.row(t) = yhat.t();
                res_filter.row(t) = res.t();
            }

            arma::vec y_loss(nT + 1, arma::fill::zeros);
            double y_loss_all = 0.;
            std::map<std::string, AVAIL::Loss> loss_list = AVAIL::loss_list;
            switch (loss_list[tolower(loss_func)])
            {
            case AVAIL::L1: // mae
            {
                arma::mat ytmp = arma::abs(res_filter);
                y_loss = arma::mean(ytmp, 1);
                y_loss_all = arma::mean(y_loss.tail(nT - 1));
                break;
            }
            case AVAIL::L2: // rmse
            {
                arma::mat ytmp = arma::square(res_filter);
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

            Rcpp::List out_filter;
            arma::vec qprob = {0.025, 0.5, 0.975};
            arma::mat yhat_out = arma::quantile(yhat_filter, qprob, 1);

            out_filter["yhat"] = Rcpp::wrap(yhat_out);
            out_filter["yhat_all"] = Rcpp::wrap(yhat_filter);
            out_filter["residual"] = Rcpp::wrap(res_filter);
            out_filter["y_loss"] = Rcpp::wrap(y_loss);
            out_filter["y_loss_all"] = y_loss_all;

            /*
            Smoothing fitted distribution: (y[t] | D[nT]) = int (y[t] | theta[t]) (theta[t] | D[nT]) d theta[t], where (theta[t] | D[nT]) ~ (atilde[t], Rtilde[t])
            */
            arma::vec psi_var = arma::vectorise(Rtilde.tube(0, 0));
            psi_var.elem(arma::find(psi_var < EPS)).fill(EPS8);
            arma::vec psi_sd = arma::sqrt(psi_var); // (nT + 1) x 1

            arma::vec psi_mean = arma::vectorise(atilde.row(0));
            arma::mat psi_sample(nT + 1, nsample, arma::fill::zeros);
            for (unsigned int i = 0; i < nsample; i ++)
            {
                arma::vec psi_tmp = psi_mean + psi_sd % arma::randn(nT + 1); // (nT + 1) x 1
                psi_sample.col(i) = psi_tmp;
            }

            Rcpp::List out_smooth = Model::fitted_error(psi_sample, y, model, loss_func);
            Rcpp::List out;
            out["filter"] = out_filter;
            out["smooth"] = out_smooth;
            return out;
        }

        void fitted_error(const Model &model, const arma::vec &y, double &y_loss_all, const unsigned int &nsample = 1000, const std::string &loss_func = "quadratic")
        {
            const unsigned int nT = y.n_elem - 1;
            /*
            Filtering fitted distribution: (y[t] | D[t]) ~ (_ft_post_mean, _ft_post_var);

            */
            arma::mat res_filter(nT + 1, nsample, arma::fill::zeros);
            arma::mat yhat_filter(nT + 1, nsample, arma::fill::zeros);
            for (unsigned int t = 1; t <= nT; t++)
            {
                arma::vec yhat = ft_post_mean.at(t) + std::sqrt(ft_post_var.at(t)) * arma::randn(nsample);
                arma::vec res = y.at(t) - yhat;

                yhat_filter.row(t) = yhat.t();
                res_filter.row(t) = res.t();
            }

            arma::vec y_loss(nT + 1, arma::fill::zeros);
            y_loss_all = 0.;
            std::map<std::string, AVAIL::Loss> loss_list = AVAIL::loss_list;
            switch (loss_list[tolower(loss_func)])
            {
            case AVAIL::L1: // mae
            {
                arma::mat ytmp = arma::abs(res_filter);
                y_loss = arma::mean(ytmp, 1);
                y_loss_all = arma::mean(y_loss.tail(nT - 1));
                break;
            }
            case AVAIL::L2: // rmse
            {
                arma::mat ytmp = arma::square(res_filter);
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

            return;
        }

        /**
         * @brief k-step-ahead forecasting error.
         * 
         * @return Rcpp::List 
         */
        Rcpp::List forecast_error(
            const Model &model, 
            const arma::vec &y,
            const unsigned int &nsample = 5000,
            const std::string &loss_func = "quadratic",
            const unsigned int &k = 1,
            const Rcpp::Nullable<unsigned int> &start_time = R_NilValue,
            const Rcpp::Nullable<unsigned int> &end_time = R_NilValue)
        {
            const unsigned int nT = y.n_elem - 1;
            arma::cube ycast(nT + 1, nsample, k, arma::fill::zeros);
            arma::cube y_err_cast(nT + 1, nsample, k, arma::fill::zeros); // (nT+1) x nsample x k
            arma::mat y_cov_cast(nT + 1, k,arma::fill::zeros);            // (nT + 1) x k
            arma::mat y_width_cast = y_cov_cast;

            arma::cube at_cast = arma::zeros<arma::cube>(model.nP, nT + 1, k + 1);
            arma::field<arma::cube> Rt_cast(k + 1);
            arma::mat Gt_cast = TransFunc::init_Gt(model.nP, model.dlag, model.ftrans);

            for (unsigned int j = 0; j <= k; j ++)
            {
                Rt_cast.at(j) = arma::zeros<arma::cube>(model.nP, model.nP, nT + 1);
            }

            unsigned int tstart = std::max(k, model.nP);
            if (start_time.isNotNull()) {
                tstart = Rcpp::as<unsigned int>(start_time);
            }

            unsigned int tend = nT - k;
            if (end_time.isNotNull()) {
                tend = Rcpp::as<unsigned int>(end_time);
            }

            for (unsigned int t = tstart; t < tend; t ++)
            {
                arma::vec ytmp = y;

                at_cast.slice(0).col(t) = mt.col(t);
                Rt_cast.at(0).slice(t) = Ct.slice(t);

                arma::mat Wt_onestep(model.nP, model.nP, arma::fill::zeros);
                for (unsigned int j = 1; j <= k; j ++)
                {
                    at_cast.slice(j).col(t) = StateSpace::func_gt(
                        model.ftrans, model.fgain, model.dlag, at_cast.slice(j - 1).col(t), ytmp.at(t + j - 1), model.seas.period, model.seas.in_state);
                    func_Gt(Gt_cast, model, at_cast.slice(j - 1).col(t), ytmp.at(t + j - 1));


                    if (use_discount)
                    {
                        if (j == 1)
                        {
                            arma::mat Rt_onestep = func_Rt(
                                Gt_cast, Rt_cast.at(j - 1).slice(t), model.derr.par1,
                                use_discount, discount_factor,
                                discount_type);
                            
                            Rt_cast.at(j).slice(t) = Rt_onestep;
                            
                            Wt_onestep = Rt2Wt(
                                Rt_onestep, model.derr.par1, use_discount,
                                discount_factor,
                                discount_type);
                        }
                        else
                        {
                            arma::mat Pt = Gt_cast * Rt_cast.at(j - 1).slice(t) * Gt_cast.t();
                            Rt_cast.at(j).slice(t) = Pt + Wt_onestep;
                        }
                    }
                    else
                    {
                        Rt_cast.at(j).slice(t) = func_Rt(
                            Gt_cast, Rt_cast.at(j - 1).slice(t), model.derr.par1,
                            use_discount, discount_factor,
                            discount_type);
                    }
                    
                    
                    arma::vec Ft_cast;
                    double ft_tmp = 0.;
                    double qt_tmp = 0.;
                    func_prior_ft(
                        ft_tmp, qt_tmp, Ft_cast, 
                        t + j, model, ytmp, 
                        at_cast.slice(j).col(t),
                        Rt_cast.at(j).slice(t),
                        fill_zero
                    );

                    ytmp.at(t + j) = LinkFunc::ft2mu(ft_tmp, model.flink, 0.);

                    arma::vec ft_cast_tmp = ft_tmp + std::sqrt(qt_tmp) * arma::randn(nsample);
                    arma::vec lambda_cast_tmp(nsample);
                    for (unsigned int i = 0; i < nsample; i ++)
                    {
                        lambda_cast_tmp.at(i) = LinkFunc::ft2mu(
                            ft_cast_tmp.at(i), model.flink, 0.);
                    }

                    ycast.slice(j - 1).row(t) = lambda_cast_tmp.t();
                    double ymin = arma::min(lambda_cast_tmp);
                    double ymax = arma::max(lambda_cast_tmp);
                    double ytrue = y.at(t + j);

                    y_err_cast.slice(j - 1).row(t) = ytrue - lambda_cast_tmp.t(); // nsample x 1
                    y_width_cast.at(t, j - 1) = std::abs(ymax - ymin);
                    y_cov_cast.at(t, j - 1) = (ytrue >= ymin && ytrue <= ymax) ? 1. : 0.;

                    
                }
            }

            arma::mat y_loss(nT + 1, k, arma::fill::zeros); // (nT + 1) x k
            arma::vec y_loss_all(k, arma::fill::zeros); // k x 1
            arma::vec y_covered_all = y_loss_all;
            arma::vec y_width_all = y_loss_all;

            y_err_cast = arma::abs(y_err_cast);

            std::map<std::string, AVAIL::Loss> loss_list = AVAIL::loss_list;
            

            for (unsigned int j = 0; j < k; j ++)
            {
                arma::vec ycov_tmp = arma::vectorise(y_cov_cast(arma::span(tstart, tend), arma::span(j)));
                y_covered_all.at(j) = arma::mean(ycov_tmp) * 100.;

                ycov_tmp = arma::vectorise(y_width_cast(arma::span(tstart, tend), arma::span(j)));
                y_width_all.at(j) = arma::mean(ycov_tmp);

                switch (loss_list[tolower(loss_func)])
                {
                case AVAIL::L1: // mae
                {
                    arma::mat ytmp = y_err_cast.slice(j);                   // (nT + 1) x nsample
                    arma::vec ymean = arma::vectorise(arma::mean(ytmp, 1)); // (nT + 1) x 1

                    y_loss.col(j) = ymean; // (nT + 1) x 1
                    y_loss_all.at(j) = arma::mean(ymean.subvec(tstart, tend));
                    break;
                }
                case AVAIL::L2: // rmse
                {
                    arma::mat ytmp = y_err_cast.slice(j); // (nT + 1) x nsample
                    ytmp = arma::square(ytmp);

                    arma::vec ymean = arma::vectorise(arma::mean(ytmp, 1));
                    double ymean_all = arma::mean(ymean.subvec(tstart, tend));

                    y_loss.col(j) = arma::sqrt(ymean);
                    y_loss_all.at(j) = std::sqrt(ymean_all);
                    break;
                }
                default:
                {
                    break;
                }
                }
            }
            

            Rcpp::List out;

            arma::vec qprob = {0.025, 0.5, 0.975};
            arma::cube ycast_out(nT + 1, qprob.n_elem, k);
            for (unsigned int j = 0; j < k; j ++)
            {
                ycast_out.slice(j) = arma::quantile(ycast.slice(j), qprob, 1);
            }

            out["y_cast"] = Rcpp::wrap(ycast_out);
            out["y_cast_all"] = Rcpp::wrap(ycast);
            out["y"] = Rcpp::wrap(y);
            out["y_loss"] = Rcpp::wrap(y_loss);
            out["y_loss_all"] = Rcpp::wrap(y_loss_all);
            out["y_covered_all"] = Rcpp::wrap(y_covered_all);
            out["y_width_all"] = Rcpp::wrap(y_width_all);

            return out;
        }

        /**
         * @brief One-step-ahead forecasting error
         * 
         * @param y_loss_all 
         * @param nsample 
         * @param loss_func 
         */
        void forecast_error(
            const Model &model, 
            const arma::vec &y,
            double &y_loss_all,
            double &y_cover,
            double &y_width,
            const unsigned int &nsample = 1000,
            const std::string &loss_func = "quadratic")
        {
            unsigned int nT = y.n_elem - 1;
            arma::mat ycast(nT + 1, nsample, arma::fill::zeros);
            arma::mat y_err_cast(nT + 1, nsample, arma::fill::zeros);
            arma::vec y_cover_cast(nT + 1, arma::fill::zeros);
            arma::vec y_width_cast(nT + 1, arma::fill::zeros);

            for (unsigned int t = 2; t <= nT; t++)
            {
                arma::vec ytmp = ft_prior_mean.at(t) + std::sqrt(ft_prior_var.at(t)) * arma::randn(nsample);
                ycast.row(t) = ytmp.t();

                double ymin = arma::min(ytmp);
                double ymax = arma::max(ytmp);
                double ytrue = y.at(t);

                y_width_cast.at(t) = std::abs(ymax - ymin);
                y_cover_cast.at(t) = (ytrue >= ymin && ytrue <= ymax) ? 1. : 0.;
                y_err_cast.row(t) = ytrue - ycast.row(t);
            }

            y_cover = arma::mean(y_cover_cast.tail(nT - 1)) * 100.;
            y_width = arma::mean(y_width_cast.tail(nT - 1));

            arma::vec y_loss(nT + 1, arma::fill::zeros);
            y_loss_all = 0;

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

            return;
        }
        


        

        arma::mat mt, at, atilde;  // nP x (nT + 1)
        arma::cube Ct, Rt, Rtilde; // nP x nP x (nT + 1)

    
    private:
        double discount_factor = 0.95;
        bool use_discount = false;
        std::string discount_type = "first_elem"; // all_lag_elems, all_elems, first_elem
        bool do_reference_analysis = false;
        bool fill_zero = LBA_FILL_ZERO;

        
        arma::vec ft_prior_mean; // (nT + 1) x 1, one-step-ahead forecasting distribution of y
        arma::vec ft_prior_var;  // (nT + 1) x 1, one-step-ahead forecasting distribution of y
        arma::vec ft_post_mean;  // (nT + 1) x 1, fitted distribution based on filtering states
        arma::vec ft_post_var; // (nT + 1) x 1, fitted distribution based on filtering states
    };
}


#endif