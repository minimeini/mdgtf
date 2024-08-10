#pragma once
#ifndef ERRDIST_H
#define ERRDIST_H

#include <RcppArmadillo.h>
#include "distributions.hpp"

// [[Rcpp::plugins(cpp17)]]
// [[Rcpp::depends(RcppArmadillo)]]

/**
 * @brief Definition of error distribution
 *
 * @param par1 W
 * @param par2 w[0] the initial value of the error sequence
 * @param wt (nT + 1), vector of normal errors.
 * @param psi (nT + 1), vector of random walk, i.e., cumulative errors.
 *
 * @code{.cpp}
 * arma::vec psi = arma::cumsum(wt)
 * @endcode
 *
 */
class ErrDist : public Dist
{
public:
    ErrDist() : par1(_par1), par2(_par2), wt(_wt), psi(_psi) 
    {
        init_default();
        return;
    }


    ErrDist(
        const std::string &err_dist,
        const double &par1,
        const double &par2
    ) : par1(_par1), par2(_par2), wt(_wt), psi(_psi)
    {
        init(err_dist, par1, par2);
    }


    void init(
        const std::string &err_dist = "gaussian",
        const double &par1 = 0.01, // W
        const double &par2 = 0.) // w[0]
    {
        _name = err_dist;
        _par1 = par1;
        _par2 = par2;

        _err_list = AVAIL::err_list;
        return;
    }


    void init_default()
    {
        _name = "gaussian";
        _par1 = 0.01;  // W
        _par2 = 0.0; // w[0]

        _err_list = AVAIL::err_list;
        return;
    }


    const double &par1;
    const double &par2;
    const arma::vec &wt;
    const arma::vec &psi;


    /**
     * @brief Draw samples from the random-walk process, return either white noise w[t] or the cumulative error psi[t].
     * 
     * @param nT Number of samples we want to draw.
     * @param W Variance of the random-walk process.
     * @param w0 Initial value of the random-walk process.
     * @param cumsum Reterun psi if true; otherwise return wt.
     * @return arma::vec 
     */
    static arma::vec sample(
        const ErrDist &derr,
        const unsigned int &nT,
        const bool &cumsum = true,
        const Rcpp::Nullable<Rcpp::NumericVector> &wt_init = R_NilValue)
    {
        std::map<std::string, AVAIL::Dist> err_list = AVAIL::err_list;
        arma::vec wt(nT + 1, arma::fill::zeros);

        double W = derr.par1;
        double w0 = derr.par2;

        if (WAKEMON_MAKE_FATAL > 0)
        {
            switch (err_list[derr.name])
            {
            case AVAIL::Dist::gaussian:
            {
                double Wsd = std::sqrt(W);
                wt.randn();
                wt.for_each([&Wsd](arma::vec::elem_type &val)
                            { val *= Wsd; });
                wt.at(0) = w0;
                break;
            }
            case AVAIL::Dist::constant:
            {
                wt.zeros();
                if (wt_init.isNull())
                {
                    throw std::invalid_argument("Constant states undefined.");
                }
                else
                {
                    arma::vec _wt_init = Rcpp::as<arma::vec>(wt_init);
                    unsigned int nelem = std::min(_wt_init.n_elem, wt.n_elem);
                    wt.head(nelem) = _wt_init;
                }
                break;
            }
            default:
            {
                throw std::invalid_argument("Undefined.");
                break;
            }
            }
        }

        arma::vec output = wt;
        if (cumsum)
        {
            output = arma::cumsum(wt);
        }

        return output;
    }

    /**
     * @brief Draw samples from the random-walk process, return either white noise w[t] or the cumulative error psi[t].
     *
     * @param nT Number of samples we want to draw.
     * @param W Variance of the random-walk process.
     * @param w0 Initial value of the random-walk process.
     * @param cumsum Reterun psi if true; otherwise return wt.
     * @return arma::vec
     */
    void sample(
        const unsigned int &nT,
        const bool &cumsum = true,
        const Rcpp::Nullable<Rcpp::NumericVector> &wt_init = R_NilValue
    )
    {
        _wt.set_size(nT + 1);
        _wt.zeros();

        if (_par1 > 0)
        {
            switch (_err_list[_name])
            {
            case AVAIL::Dist::gaussian:
            {

                double Wsd = std::sqrt(_par1);
                _wt.randn();
                _wt.for_each([&Wsd](arma::vec::elem_type &val)
                             { val *= Wsd; });
                _wt.at(0) = _par2;
                break;
            }
            case AVAIL::Dist::constant:
            {
                _wt.zeros();
                if (wt_init.isNull())
                {
                    throw std::invalid_argument("Constant states undefined.");
                }
                else
                {
                    _wt_init = Rcpp::as<arma::vec>(wt_init);
                    unsigned int nelem = std::min(_wt_init.n_elem, _wt.n_elem);
                    _wt.head(nelem) = _wt_init;
                }
                break;
            }
            default:
            {
                throw std::invalid_argument("Undefined.");
                break;
            }
            }
        }

        

        if (cumsum)
        {
            _psi = arma::cumsum(_wt);
        }
        return;
    }

private:
    unsigned int _nT;
    arma::vec _wt;
    arma::vec _psi;           // Initial value (at time = 0) of the normal errors.
    std::map<std::string, AVAIL::Dist> _err_list;
    arma::vec _wt_init;
    
};



#endif