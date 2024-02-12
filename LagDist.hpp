#pragma once
#ifndef LAGDIST_H
#define LAGDIST_H

#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <boost/math/special_functions/beta.hpp>
#include <RcppArmadillo.h>
#include "utils.h"
// [[Rcpp::plugins(cpp17)]]
// [[Rcpp::depends(RcppArmadillo,BH)]]

/**
 * @brief Negative-binomial distribution.
 * 
 */
class nbinom
{
public:
    nbinom() : nL(30), kappa(NB_KAPPA), r(NB_R) 
    {
        _r = static_cast<unsigned int>(r);
    };
    nbinom(const unsigned int nL_, const double kappa_, const double r_)
    {
        nL = nL_;
        kappa = kappa_;
        r = r_;
        _r = static_cast<unsigned int>(r);

        return;
    }
    /**
     * @brief Binomial coefficients, i.e., n choose k.
     *
     * @param n
     * @param k
     * @return double
     */
    static double binom(const int &n, const int &k) { return 1. / ((static_cast<double>(n) + 1.) * boost::math::beta(std::max(static_cast<double>(n - k + 1), EPS), std::max(static_cast<double>(k + 1), EPS))); }

    /**
     * @brief Cumulative density function of negative-binomial distribution.
     * [Checked. OK.]
     *
     * @param nL
     * @param kappa
     * @param r
     * @return arma::vec
     */
    static arma::vec dnbinom(
        const unsigned int &nL,
        const double &kappa,
        const double &r)
    {
        // double kappa = param[0];
        // double r = param[1];

        if (nL < 1)
        {
            throw std::invalid_argument("Number of lags, nL, must be positive.");
        }

        arma::vec output(nL, arma::fill::zeros);
        double c3 = std::pow(1. - kappa, r);

        for (unsigned int d = 0; d < nL; d++)
        {
            double lag = static_cast<double>(d) + 1.;
            double a = lag + r - 2.;
            double b = lag - 1.;
            double c1 = binom(a, b);
            double c2 = std::pow(kappa, b);

            output.at(d) = c1 * c2;
            output.at(d) *= c3;
        }

        bound_check<arma::vec>(output, "dnbinom", false, true);
        return output;
    }

    arma::vec dnbinom()
    {
        if (nL < 1)
        {
            throw std::invalid_argument("Number of lags, nL, must be positive.");
        }

        arma::vec output(nL, arma::fill::zeros);
        double c3 = std::pow(1. - kappa, r);

        for (unsigned int d = 0; d < nL; d++)
        {
            double lag = static_cast<double>(d) + 1.;
            double a = lag + r - 2.;
            double b = lag - 1.;
            double c1 = binom(a, b);
            double c2 = std::pow(kappa, b);

            output.at(d) = c1 * c2;
            output.at(d) *= c3;
        }

        bound_check<arma::vec>(output, "dnbinom", false, true);
        return output;
    }


    /**
     * @brief Iterative form as in Solow(1960)
     * 
     * @param kappa 
     * @param r 
     * @return arma::vec 
     */
    static arma::vec iter_coef(const double &kappa, const double &r)
    {
        unsigned int _r = static_cast<unsigned int>(r);
        arma::vec coef(_r, arma::fill::zeros);
        for (unsigned int k = _r; k >= 1; k--)
        {
            double c1 = binom(r, k);
            double c2 = std::pow( - kappa, k);
            coef.at(r - k) = - c1 * c2;
        }

        bound_check<arma::vec>(coef, "nbinom::iter_coef: coef");
        return coef;
    }

private:
    unsigned int nL = 30;
    double kappa = NB_KAPPA;
    double r = NB_R;
    unsigned int _r;

    double dnbinom0(
        const double &lag, // starting from 1
        const double &rho,
        const double &L_order)
    {
        double c3 = std::pow(1. - rho, L_order);
        double a = lag + L_order - 2.;
        double b = lag - 1.;
        double c1 = binom(a, b);
        double c2 = std::pow(rho, b);

        // double c1 = R::dnbinom(k-1,(double)Last,1.-rho);
        // double c2 = std::pow(-1.,k-1.);
        // Fphi.at(d) = c1 * c2;
        double output = (c1 * c2) * c3;
        bound_check(output, "dnbinom0", false, true);
        return output;
    }
};



/**
 * @brief Discretized log-normal distribution characterized by mean and variance.
 *
 */
class lognorm
{
public:
    lognorm(): nL(30), mu(LN_MU), sd2(LN_SD2) {};
    lognorm(
        const unsigned int &nL_,
        const double &mu_,
        const double &sd2_)
    {
        nL = nL_;
        mu = mu_;
        sd2 = sd2_;
    }


    /**
     * @brief P.M.F of discretized log-normal distribution, characterized by mean and variance.
     * @brief General class methods.
     * 
     * @param nL unsigned int: number of lags, defines the length of the returned vector.
     * @param mu double: the mean of the log-normal distribution.
     * @param sd2 double: the variance of the log-normal distribution.
     * 
     * @return arma::vec
     */
    static arma::vec dlognorm(
        const unsigned int &nL,
        const double &mu,
        const double &sd2)
    {
        lognorm lg;
        arma::vec output(nL);
        for (unsigned int d = 0; d < nL; d++)
        {
            output.at(d) = lg.dlognorm0(static_cast<double>(d) + 1., mu, sd2);
        }

        return output;
    }

    /**
     * @brief P.M.F of discretized log-normal distribution, characterized by mean and variance.
     * @brief Member function of a class instance.
     *
     * @param nL unsigned int: number of lags, defines the length of the returned vector.
     * @param mu double: the mean of the log-normal distribution.
     * @param sd2 double: the variance of the log-normal distribution.
     *
     * @return arma::vec
     */
    arma::vec dlognorm()
    {
        arma::vec output(nL);
        for (unsigned int d = 0; d < nL; d++)
        {
            output.at(d) = dlognorm0(static_cast<double>(d) + 1., mu, sd2);
        }

        return output;
    }

private:
    unsigned int nL = 30;
    double mu = LN_MU;
    double sd2 = LN_SD2;

    double Pd(
        const double d,
        const double mu,
        const double sigmasq)
    {
        arma::vec tmpv1(1);
        tmpv1.at(0) = -(std::log(d) - mu) / std::sqrt(2. * sigmasq);
        return arma::as_scalar(0.5 * arma::erfc(tmpv1));
    }

    double dlognorm0(
        const double &lag, // starting from 1
        const double &mu,
        const double &sd2)
    {
        double output = Pd(lag, mu, sd2) - Pd(lag - 1., mu, sd2);
        bound_check(output, "dlognorm0", false, true);
        return output;
    }
};

/**
 * @file LagDist.hpp
 *
 * @brief Define and manipulate lag distributions Fphi. Lag Distributions are also known as transfer functions.
 *
 * @param _name string: name of the lag distribution, must be one of the lag distribution listed in AVAIL (modified by set_lag_dist).
 * @param _par1 (par1, par2) are two parameters that defined the lag distribution: nbinom(kappa, r); lognorm(mu, sd2) (modified by set_lag_dist).
 * @param _par2 (par1, par2) are two parameters that defined the lag distribution: nbinom(kappa, r); lognorm(mu, sd2) (modified by set_lag_dist).
 * 
 * 
 * @param _Fphi nL x 1 vector: lag distributions, C.D.F of discrete random variables, characterized by two parameters.
 * 
 * @return Fphi
 */
class LagDist : public Dist
{
public:
    LagDist() : Dist(), isnbinom(_isnbinom), par1(_par1), par2(_par2), Fphi(_Fphi)
    {
        init_default();
        return;
    }

    LagDist(
        const std::string &name,
        const double &par1,
        const double &par2) : Dist(), isnbinom(_isnbinom), par1(_par1), par2(_par2), Fphi(_Fphi)
    {
        init(name, par1, par2);
        return;
    }

    const bool &isnbinom;
    void init(
        const std::string &name,
        const double &par1,
        const double &par2)
    {
        _name = name;
        _par1 = par1;
        _par2 = par2;
        _lag_list = AVAIL::map_lag_dist();
        _isnbinom = (_lag_list[_name] == AVAIL::Dist::nbinomp) ? true : false;

        return;
    }

    void init_default()
    {
        _name = "nbinom";
        _par1 = NB_KAPPA;
        _par2 = NB_R;
        _lag_list = AVAIL::map_lag_dist();
        _isnbinom = (_lag_list[_name] == AVAIL::Dist::nbinomp) ? true : false;
    }

    const double &par1;
    const double &par2;
    const arma::vec &Fphi;
    /**
     * @brief Set Fphi based on number of lags, type of lag distribution and its corresponding parameters
     *
     *
     */
    void get_Fphi(const unsigned int &nlag)
    {
        _Fphi.set_size(nlag);
        _Fphi.zeros();

        switch (_lag_list[_name])
        {
        case AVAIL::Dist::lognorm:
            _Fphi = lognorm::dlognorm(nlag, _par1, _par2);
            break;
        case AVAIL::Dist::nbinomp:
            _Fphi = nbinom::dnbinom(nlag, _par1, _par2);
            break;
        default:
            throw std::invalid_argument("Supported lag distributions: 'lognorm', 'nbinom'.");
            break;
        }

        bound_check<arma::vec>(_Fphi, "void get_Fphi: _Fphi");

        return;
    }



    /**
     * @brief Get P.M.F of the lag distribution. [Checked. OK.s]
     * 
     * @param nlag unsigned int
     * @param lag_dist nbinom or lognorm
     * @param lag_par1 kappa or mu
     * @param lag_par2 r or var
     * @return arma::vec 
     */
    static arma::vec get_Fphi(
        const unsigned int &nlag, 
        const std::string &lag_dist,
        const double &lag_par1,
        const double &lag_par2)
    {
        arma::vec Fphi(nlag, arma::fill::zeros);
        std::map<std::string, AVAIL::Dist> _lag_list = AVAIL::map_lag_dist();

        switch (_lag_list[lag_dist])
        {
        case AVAIL::Dist::lognorm:
            Fphi = lognorm::dlognorm(nlag, lag_par1, lag_par2);
            break;
        case AVAIL::Dist::nbinomp:
            Fphi = nbinom::dnbinom(nlag, lag_par1, lag_par2);
            break;
        default:
            throw std::invalid_argument("Supported lag distributions: 'lognorm', 'nbinom'.");
        }

        bound_check<arma::vec>(Fphi, "arma::vec get_Fphi: Fphi");
        return Fphi;
    }

    /**
     * @brief Get the optim number of lags that satisfy a specific margin of error.
     *
     * @param error_margin double (default = 0.01): the lower bound of 1 - sum(Fphi)
     * @return nlag - unsigned int.
     */
    static unsigned int get_optim_nlag(
        const std::string &lag_dist = "nbinom",
        const double &lag_par1 = NB_KAPPA,
        const double &lag_par2 = NB_R,
        const double &error_margin = 0.01)
    {
        unsigned int nlag = 1;
        bool cont = true;
        while (cont)
        {
            arma::vec Fphi_tmp = get_Fphi(nlag, lag_dist, lag_par1, lag_par2);
            double prob = arma::accu(Fphi_tmp);

            if (1. - prob <= error_margin)
            {
                cont = false;
            }
            else
            {
                nlag += 1;
            }
        }

        return nlag;
    }

private:
    std::map<std::string, AVAIL::Dist> _lag_list;
    arma::vec _Fphi;      // a vector of the lag distribution CDF at desired length _nL.
    bool _isnbinom;
};

#endif