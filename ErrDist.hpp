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
    ErrDist()
    {
        init_default();
        return;
    }


    ErrDist(
        const std::string &err_dist,
        const double &par1_in,
        const double &par2_in
    )
    {
        name = err_dist;
        par1 = par1_in;
        par2 = par2_in;
    }


    void init_default()
    {
        name = "gaussian";
        par1 = 0.01;  // W
        par2 = 0.0; // w[0]
        return;
    }


    /**
     * @brief Diagonal variance-covariance matrix Wt at t = 0 (i.e. initial value)
     * 
     * @param nP dimension of Wt
     * @param diagonals optional.
     * @return arma::mat, Wt (nP x nP)
     */
    static arma::mat init_Wt(
        const unsigned int &nP,
        const Rcpp::Nullable<Rcpp::NumericVector> diagonals = R_NilValue
    )
    {
        arma::mat Wt(nP, nP, arma::fill::zeros);
        if (diagonals.isNotNull())
        {
            arma::vec diags = Rcpp::as<arma::vec>(diagonals);
            unsigned int niter = (diags.n_elem < nP) ? diags.n_elem : nP;
            for (unsigned int i = 0; i < niter; i++)
            {
                Wt.at(i, i) = diags.at(i);
            }
        }

        return Wt;
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
    static arma::vec sample(
        const ErrDist &derr,
        const unsigned int &nT,
        const bool &cumsum = true,
        const Rcpp::Nullable<Rcpp::NumericVector> &wt_init = R_NilValue)
    {
        std::map<std::string, AVAIL::Dist> err_list = ErrDist::err_list;
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


    static const std::map<std::string, AVAIL::Dist> err_list;
private:
    static std::map<std::string, AVAIL::Dist> map_err_dist()
    {
        std::map<std::string, AVAIL::Dist> ERR_MAP;

        ERR_MAP["gaussian"] = AVAIL::Dist::gaussian;
        ERR_MAP["normal"] = AVAIL::Dist::gaussian;

        ERR_MAP["constant"] = AVAIL::Dist::constant;
        return ERR_MAP;
    }
};

inline const std::map<std::string, AVAIL::Dist> ErrDist::err_list = ErrDist::map_err_dist();

#endif