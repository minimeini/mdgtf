#ifndef _LINEARBAYES_H
#define _LINEARBAYES_H

#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <algorithm>
#include <RcppArmadillo.h>
#include "../core/Model.hpp"


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
            if (discount_list[tolower(discount_type)] == DiscountType::first_elem)
            {
                Rt.at(0, 0) /= delta_discount;
            }
            else
            {
                Rt.for_each([&delta_discount](arma::mat::elem_type &val)
                            { val /= delta_discount; });
            }
        }
        else
        {
            // W[t] = diag(W, 0, ..., 0)
            Rt.at(0, 0) += W;
        }

        #ifdef DGTF_DO_BOUND_CHECK
        bound_check<arma::mat>(Rt, "func_Rt: Rt");
        #endif
        return Rt;
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
        const arma::mat &Rt
    )
    {
        mean_ft = TransFunc::func_ft(model.ftrans, model.fgain, model.dlag, model.seas, t, at, yall);
        _Ft = TransFunc::func_Ft(model.ftrans, model.fgain, model.dlag, t, at, yall, model.seas.period, model.seas.in_state);
        var_ft = arma::as_scalar(_Ft.t() * Rt * _Ft);

        #ifdef DGTF_DO_BOUND_CHECK
        bound_check(var_ft, "LBA::func_prior_ft: var_ft", true, true);
        #endif
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

        #ifdef DGTF_DO_BOUND_CHECK
        bound_check(alpha, "func_alpha_beta: alpha");
        bound_check(beta, "func_alpha_beta: beta");
        #endif

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

        #ifdef DGTF_DO_BOUND_CHECK
        bound_check(mean_ft, "func_posterior_ft: mean_ft");
        bound_check(var_ft, "func_posterior_ft: var_ft");
        #endif

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

        #ifdef DGTF_DO_BOUND_CHECK
        bound_check<arma::vec>(At, "func_At: At");
        #endif
        
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

        #ifdef DGTF_DO_BOUND_CHECK
        bound_check<arma::vec>(mt, "func_mt: mt");
        #endif
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

        #ifdef DGTF_DO_BOUND_CHECK
        bound_check<arma::mat>(Ct, "func_Ct: Ct");
        #endif
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
            if (opts.containsElementNamed("discount_factor"))
            {
                discount_factor = Rcpp::as<double>(opts["discount_factor"]);
            }

            do_reference_analysis = false;
            if (opts.containsElementNamed("do_reference_analysis"))
            {
                do_reference_analysis = Rcpp::as<bool>(opts["do_reference_analysis"]);
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
            opts["discount_factor"] = 0.95;
            opts["do_smoothing"] = true;
            opts["do_reference_analysis"] = false;

            return opts;
        }

        Rcpp::List get_output(const Model &model)
        {
            std::map<std::string, SysEq::Evolution> sys_list = SysEq::sys_list;
            Rcpp::List output;

            if (sys_list[model.fsys] == SysEq::Evolution::identity)
            {
                output["mt"] = Rcpp::wrap(mt);
                output["Ct"] = Rcpp::wrap(Ct);
            }
            else
            {
                arma::mat psi = get_psi(atilde, Rtilde);
                arma::mat psi_filter = get_psi(mt, Ct);
                output["psi"] = Rcpp::wrap(psi);
                output["psi_filter"] = Rcpp::wrap(psi_filter);
            }
            

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
            arma::mat Gt = SysEq::init_Gt(
                model.nP, model.dlag, model.fsys,
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
                SysEq::func_Gt(Gt, model.fsys, model.fgain, model.dlag, mt.col(t - 1), y.at(t - 1));
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
            const bool &use_discount = false,
            const double &discount_factor = 0.95,
            const std::string &discount_type = "first_elem")
        {
            std::map<std::string, AVAIL::Dist> obs_list = ObsDist::obs_list;

            at_new = SysEq::func_gt(
                model.fsys, model.fgain, model.dlag,
                mt_old, y.at(t - 1),
                model.seas.period, model.seas.in_state); // Checked. OK.

            SysEq::func_Gt(Gt, model.fsys, model.fgain, model.dlag, mt_old, y.at(t - 1));
            Rt_new = func_Rt(
                Gt, Ct_old, model.derr.par1, 
                use_discount, discount_factor,
                discount_type);
            
            ft_prior = 0.;
            qt_prior = 0.;
            func_prior_ft(ft_prior, qt_prior, Ft, t, model, y, at_new, Rt_new);

            ft_posterior = y.at(t);
            qt_posterior = 0.;

            if (obs_list[model.dobs.name] != AVAIL::Dist::gaussian)
            {
                double alpha = 0.;
                double beta = 0.;
                func_alpha_beta(alpha, beta, model, ft_prior, qt_prior, y.at(t), true);
                func_posterior_ft(ft_posterior, qt_posterior, model, alpha, beta);
            }
            else
            {
                qt_prior += model.dobs.par2;
            }

            arma::mat At = func_At(Rt_new, Ft, qt_prior);
            mt_new = func_mt(at_new, At, ft_prior, ft_posterior);
            Ct_new = func_Ct(Rt_new, At, qt_prior, qt_posterior);
            return;
        }

        void filter(Model &model, const arma::vec &y, const double &theta_upbnd = 1.)
        {
            const unsigned int nT = y.n_elem - 1;
            arma::vec Ft = TransFunc::init_Ft(model.nP, model.ftrans, model.seas.period, model.seas.in_state);             // set F[0]
            arma::mat Gt = SysEq::init_Gt(model.nP, model.dlag, model.fsys, model.seas.period, model.seas.in_state); // set G[0]

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
            arma::mat Gt = SysEq::init_Gt(model.nP, model.dlag, model.fsys, model.seas.period, model.seas.in_state); // set G[0]

            std::map<std::string, TransFunc::Transfer> trans_list = TransFunc::trans_list;
            const bool is_sliding = trans_list[model.ftrans] == TransFunc::Transfer::sliding;

            atilde.set_size(model.nP, nT + 1);
            atilde.zeros();
            atilde.col(nT) = mt.col(nT);

            Rtilde.set_size(model.nP, model.nP, nT + 1);
            Rtilde.zeros();
            Rtilde.slice(nT) = Ct.slice(nT);

            for (unsigned int t = nT; t > 0; t--)
            {
                if (use_discount && is_sliding)
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
                else
                {
                    arma::mat Rtinv = inverse(Rt.slice(t));

                    SysEq::func_Gt(Gt, model.fsys, model.fgain, model.dlag, mt.col(t - 1), y.at(t - 1));
                    arma::mat Bt = (Ct.slice(t - 1) * Gt.t()) * Rtinv;

                    arma::vec diff_a = atilde.col(t) - at.col(t);
                    arma::mat diff_R = Rtilde.slice(t) - Rt.slice(t);

                    arma::vec atilde_right = Bt * diff_a;
                    atilde.col(t - 1) = mt.col(t - 1) + atilde_right;

                    arma::mat Rtilde_right = (Bt * diff_R) * Bt.t();
                    Rtilde.slice(t - 1) = Ct.slice(t - 1) + Rtilde_right;
                }
            }
        }        

        arma::mat mt, at, atilde;  // nP x (nT + 1)
        arma::cube Ct, Rt, Rtilde; // nP x nP x (nT + 1)

    
    private:
        double discount_factor = 0.95;
        bool use_discount = false;
        std::string discount_type = "first_elem"; // all_lag_elems, all_elems, first_elem
        bool do_reference_analysis = false;
        
        arma::vec ft_prior_mean; // (nT + 1) x 1, one-step-ahead forecasting distribution of y
        arma::vec ft_prior_var;  // (nT + 1) x 1, one-step-ahead forecasting distribution of y
        arma::vec ft_post_mean;  // (nT + 1) x 1, fitted distribution based on filtering states
        arma::vec ft_post_var; // (nT + 1) x 1, fitted distribution based on filtering states
    };
}


#endif