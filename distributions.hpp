#ifndef DISTRIBUTIONS_HPP
#define DISTRIBUTIONS_HPP

#include <string>
#include <boost/math/special_functions/beta.hpp>
#include "utils.h"

// [[Rcpp::depends(BH)]]


class MVNorm
{
public:
    MVNorm(const unsigned int nP)
    {
        mu.set_size(nP);
        mu.zeros();

        Sigma.set_size(nP, nP);
        Sigma.eye();

        return;
    }

    void update_mu(const arma::vec &mu_in)
    {
        mu = mu_in;
        return;
    }

    void update_Sigma(const arma::mat &Sigma_in)
    {
        Sigma = Sigma_in;
        return;
    }

    static arma::mat get_precision(const arma::mat &Sigma)
    {
        arma::mat prec = inverse(Sigma, false, true);
        return prec;
    }

    static double dmvnorm(const arma::vec &x, const arma::vec &mu, const arma::mat &Sigma, const bool &return_log = true)
    {
        arma::mat prec = inverse(Sigma, false, true);
        double logdet = arma::log_det_sympd(prec);
        double cnst = static_cast<double>(mu.n_elem) * std::log(2. * arma::datum::pi);
        
        arma::vec diff = x - mu;
        double dist = arma::as_scalar(diff.t() * prec * diff);

        double logprob = - cnst + logdet - dist;
        logprob *= 0.5;

        if (return_log)
        {
            return logprob;
        }
        else
        {
            logprob = std::min(logprob, UPBND);
            return std::exp(logprob);
        }
    }

private:
    arma::vec mu; // mean
    arma::mat Sigma; // Variance-covariance matrix
};


class Beta : public Dist
{
public: 
    Beta() : Dist(), alpha(_par1), beta(_par2)
    {
        _name = "beta";
        _par1 = 1.;
        _par2 = 1.;
    }
    
    Beta(const double &alpha_, const double &beta_) : Dist(), alpha(_par1), beta(_par2)
    {
        _name = "beta";
        _par1 = alpha_;
        _par2 = beta_;
    }

    Beta(const Dist &dist_) : Dist(), alpha(_par1), beta(_par2)
    {
        _name = "beta";
        _par1 = dist_.par1;
        _par2 = dist_.par2;
    }

    const double &alpha;
    const double &beta;

    double mean()
    {
        double a_plus_b = alpha + beta;
        return alpha / a_plus_b;
    }

    static double mean(const double &alpha, const double &beta)
    {
        double a_plus_b = alpha + beta;
        return alpha / a_plus_b;
    }

    double var()
    {
        double a_plus_b = alpha + beta;
        double a_plus_b_sq = std::pow(a_plus_b, 2.);
        double a_times_b = alpha * beta;

        double denom = a_plus_b_sq * (a_plus_b + 1.);
        return a_times_b / denom;
    }


    static double var(const double &alpha, const double &beta)
    {
        double a_plus_b = alpha + beta;
        double a_plus_b_sq = std::pow(a_plus_b, 2.);
        double a_times_b = alpha * beta;

        double denom = a_plus_b_sq * (a_plus_b + 1.);
        return a_times_b / denom;
    }
};


class InverseGamma : public Dist
{
public:
    InverseGamma() : Dist(), alpha(_par1), beta(_par2) 
    {
        _name = "invgamma";
        _par1 = 0.01;
        _par2 = 0.01;
    }

    InverseGamma(const double &shape, const double &rate) : Dist(), alpha(_par1), beta(_par2)
    {
        _name = "invgamma";
        _par1 = shape;
        _par2 = rate;
    }

    const double &alpha; // shape
    const double &beta; // rate

    double mode()
    {
        return mode(alpha, beta);
    }

    static double mode(const double &alpha, const double &beta)
    {
        return beta / (alpha + 1.);
    }

    double mean()
    {
        return mean(alpha, beta);
    }
    
    static double mean(const double &alpha, const double &beta)
    {
        if (alpha > 1.)
        {
            return beta / (alpha - 1.);
        }
        else
        {
            return -1;
        }
    }

    double var()
    {
        return var(alpha, beta);
    }

    static double var(const double &alpha, const double &beta)
    {
        if (alpha > 2.)
        {
            double nom = beta * beta;
            double denom = std::pow(alpha - 1., 2.);
            denom *= alpha - 2.;
            return nom / denom;
        }
        else
        {
            return -1;
        }
    }

    double sample()
    {
        return sample(alpha, beta);
    }

    static double sample(const double &alpha, const double &beta)
    {
        double out = 1. / R::rgamma(alpha, 1. / beta);
        bound_check(out, "InverseGamma::sample: out");
        return out;
    }
};

class Gamma : public Dist
{
public:
    Gamma() : Dist(), alpha(_par1), beta(_par2)
    {
        _name = "gamma";
        _par1 = 1.;
        _par2 = 1.;
    }

    Gamma(const double &shape, const double &rate) : Dist(), alpha(_par1), beta(_par2)
    {
        _name = "gamma";
        _par1 = shape;
        _par2 = rate;
    }

    Gamma(const Dist &dist_) : Dist(), alpha(_par1), beta(_par2)
    {
        _name = "gamma";
        _par1 = dist_.par1;
        _par2 = dist_.par2;
    }

    const double &alpha;
    const double &beta;

    double mean()
    {
        return alpha / beta;
    }

    static double mean(const double &alpha, const double &beta)
    {
        return alpha / beta;
    }

    double var()
    {
        double denom = beta * beta;
        return alpha / denom;
    }


    static double var(const double &alpha, const double &beta)
    {
        double a_plus_b = alpha + beta;
        double a_plus_b_sq = std::pow(a_plus_b, 2.);
        double a_times_b = alpha * beta;

        double denom = beta * beta;
        return alpha / denom;
    }

    double mean_logGamma()
    {
        double da = R::digamma(alpha);
        double logb = std::log(beta);
        return da - logb;
    }

    static double mean_logGamma(const double &alpha, const double &beta)
    {
        double da = R::digamma(alpha);
        double logb = std::log(beta);
        return da - logb;
    }

    double var_logGamma()
    {
        return R::trigamma(alpha);
    }

    static double var_logGamma(const double &alpha, const double &beta = 0.)
    {
        return R::trigamma(alpha);
    }


    double mode_logGamma()
    {
        double loga = std::log(std::abs(alpha) + EPS);
        double logb = std::log(std::abs(beta) + EPS);
        return loga - logb;
    }


    static double mode_logGamma(const double &alpha, const double &beta)
    {
        double loga = std::log(std::abs(alpha) + EPS);
        double logb = std::log(std::abs(beta) + EPS);
        return loga - logb;
    }

    double curvature_logGamma()
    {
        return 1. / std::abs(alpha);
    }

    static double curvature_logGamma(const double &alpha, const double &beta = 0.)
    {
        return 1. / std::abs(alpha);
    }
};

class Poisson : public Dist
{
public:
    Poisson() : Dist(), lambda(_par1)
    {
        _name = "poisson";
        _par1 = 1.;
        _par2 = 0.;
    }

    Poisson(const double &lambda_, const double &par2 = 0.) : Dist(), lambda(_par1)
    {
        _name = "poisson";
        _par1 = lambda_;
        _par2 = 0.;
    }

    Poisson(const Dist &dist_) : Dist(), lambda(_par1)
    {
        _name = "poisson";
        _par1 = dist_.par1;
        _par2 = dist_.par2;
    }

    const double &lambda;

    double sample()
    {
        return R::rpois(lambda);
    }

    static double sample(const double &lambda, const double &delta = 0.)
    {
        return R::rpois(lambda);
    }

    double mean()
    {
        return lambda;
    }

    static double mean(const double &lambda, const double &delta)
    {
        return lambda;
    }

    double var()
    {
        return lambda;
    }

    static double var(const double &lambda, const double &delta)
    {
        return lambda;
    }

    double mean2conj()
    {
        _nu = lambda;
    }

    static double mean2conj(const double &lambda, const double &delta)
    {
        double nu = lambda;
        return lambda;
    }


    static double conj2mean(const double &nu, const double &delta)
    {
        return nu;
    }

    /**
     * @brief Mean and expectation of lambda[t] = nu[t], where nu[t] ~ Gamma(alpha[t], beta[t]).
     *
     * @param lambda_mean
     * @param lambda_var
     * @param Beta
     * @param delta
     */
    static void moments_mean(
        double &lambda_mean,
        double &lambda_var,
        const double &alpha,
        const double &beta,
        const double &delta = 0.)
    {
        lambda_mean = Gamma::mean(alpha, beta);
        lambda_var = Gamma::var(alpha, beta);

        return;
    }


private:
    double _nu;

    // double trigamma_obj(
    //     unsigned n, // not sure what it is for
    //     const double *x,
    //     double *grad,
    //     void *my_func_data)
    // { // extra parameters: q

    //     double *q = (double *)my_func_data;

    //     if (grad)
    //     {
    //         grad[0] = 2 * (R::trigamma(x[0]) - (*q)) * R::psigamma(x[0], 2);
    //     }

    //     return std::pow(R::trigamma(x[0]) - (*q), 2);
    // }

    // double optimize_trigamma(double q)
    // {
    //     nlopt_opt opt;
    //     opt = nlopt_create(NLOPT_LD_MMA, 1);

    //     double lb[1] = {0}; // lower bound
    //     nlopt_set_lower_bounds(opt, lb);
    //     nlopt_set_xtol_rel(opt, 1e-4);
    //     nlopt_set_maxeval(opt, 50);
    //     nlopt_set_maxtime(opt, 5.);
    //     nlopt_set_min_objective(opt, trigamma_obj, (void *)&q);

    //     double x[1] = {1e-6};
    //     double minf;
    //     if (nlopt_optimize(opt, x, &minf) < 0)
    //     {
    //         Rprintf("nlopt failed!\\n");
    //     }

    //     double result = x[0];
    //     nlopt_destroy(opt);
    //     return result;
    // }
};


/**
 * @brief Negative-binomial distribution, characterized by (mean = lambda, dispersion = delta).
 *
 */
class nbinomm : public Dist
{
public:
    nbinomm() : Dist(), nu(_nu), lambda(_par1), delta(_par2)
    {
        _name = "nbinomm";
        _par1 = NB_LAMBDA;
        _par2 = NB_DELTA;
        mean2conj();
        return;
    }

    const double &nu;
    const double &lambda;
    const double &delta;

    nbinomm(const double &lambda, const double &delta) : Dist(), nu(_nu), lambda(_par1), delta(_par2)
    {
        _name = "nbinomm";
        _par1 = lambda;
        _par2 = delta;
        mean2conj();
    }

    static double dnbinomm(
        const double &y, 
        const double &lambda, 
        const double &delta, 
        const bool &log=true)
    {
        // double kappa = lambda / (lambda + delta);
        // double output = nbinom::dnbinom(y, kappa, delta);

        double tmp1 = (R::lgammafn(y + delta) - R::lgammafn(y + 1.)) - R::lgammafn(delta);

        double lognkappa = std::log(delta) - std::log(lambda + delta);
        double tmp2 = delta * lognkappa;

        double logkappa = std::log(lambda) - std::log(lambda + delta);
        double tmp3 = y * logkappa;

        double output = tmp1 + tmp2 + tmp3;
        if (!log)
        {
            output = std::min(output, UPBND);
            output = std::exp(output);
        }

        return output;
    }

    double sample()
    {
        double prob_succ = par2 / (par1 + par2);
        return R::rnbinom(par2, prob_succ);
    }

    static double sample(const double &lambda, const double &delta)
    {
        double prob_succ = delta / (lambda + delta);
        return R::rnbinom(delta, prob_succ);
    }

    double mean()
    {
        return _par1;
    }


    static double mean(const double &lambda, const double &delta)
    {
        return lambda;
    }

    double mean2conj()
    {
        _nu = _par1 / (_par1 + _par2);
    }

    static double mean2conj(const double &lambda, const double &delta)
    {
        double nu = lambda / (lambda + delta);
        return nu;
    }

    static double conj2mean(const double &nu, const double &delta)
    {
        double lambda = delta * nu / (1. - nu);
        return lambda;
    }


    /**
     * @brief Mean and expectation of lambda[t] = delta * nu[t] / (1 - nu[t]), where nu[t] ~ Beta(alpha[t], beta[t]).
     *
     * @param lambda_mean
     * @param lambda_var
     * @param Beta
     * @param delta
     */
    static void moments_mean(
        double &lambda_mean,
        double &lambda_var,
        const double &alpha,
        const double &beta,
        const double &delta
    )
    {
        double nom = delta * alpha;
        double denom = beta - 1.;
        lambda_mean = nom / denom;

        double delta2 = delta * delta;
        nom = alpha + beta - 1.;
        nom *= alpha;
        nom *= delta2;

        denom = beta - 2;
        denom *= std::pow(beta - 1, 2.);

        lambda_var = nom / denom;

        return;
    }


private:
    double _nu;
};

/**
 * @brief Negative-binomial distribution, characterized by `nbinomp`: probability (kappa, r).
 *
 * @param _name "nbinomp"
 * @param _par1 kappa
 * @param _par2 r
 */
class nbinom : public Dist
{
public:
    nbinom() : Dist()
    {
        _r = static_cast<unsigned int>(NB_R);

        _name = "nbinomp";
        _par1 = NB_KAPPA;
        _par2 = NB_R;
        return;
    }


    nbinom(const double kappa, const double r) : Dist()
    {
        _r = static_cast<unsigned int>(r);

        _name = "nbinomp";
        _par1 = kappa;
        _par2 = r;
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


    static double mean(
        const double &kappa, // probability of failures
        const double &r) // number of success
    {
        double prob_succ = 1. - kappa;
        double val = r * (1. - prob_succ);
        val /= prob_succ;
        return val;
    }

    static double var(
        const double &kappa, // probability of failures
        const double &r)  // number of success
    {
        double prob_succ = 1. - kappa;
        double val = r * (1. - prob_succ);
        val /= std::pow(prob_succ, 2.);
        return val;
    }

    static int mode(
        const double &kappa,
        const double &r
    )
    {
        double val = 0.;
        if (r > 1)
        {
            double prob_succ = 1. - kappa;
            val = (r - 1.) * (1. - prob_succ);
            val /= prob_succ;
        }

        return static_cast<int>(val);
    }

        /**
         * @brief P.M.F of negative-binomial distribution.
         * [Checked. OK.]
         *
         * @param nL
         * @return arma::vec
         */
        arma::vec dnbinom(const unsigned int &nL)
    {
        if (nL < 1)
        {
            throw std::invalid_argument("Number of lags, nL, must be positive.");
        }

        arma::vec output(nL, arma::fill::zeros);
        double c3 = std::pow(1. - _par1, _par2);

        for (unsigned int d = 0; d < nL; d++)
        {
            output.at(d) = dnbinom(static_cast<double>(d), _par1, _par2, c3);
        }

        bound_check<arma::vec>(output, "dnbinom", false, true);
        return output;
    }

    /**
     * @brief P.M.F of negative-binomial distribution.
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
            // double lag = static_cast<double>(d) + 1.;
            // double a = lag + r - 2.;
            // double b = lag - 1.;
            // double c1 = binom(a, b);
            // double c2 = std::pow(kappa, b);

            // output.at(d) = c1 * c2;
            // output.at(d) *= c3;

            output.at(d) = dnbinom(static_cast<double>(d), kappa, r, c3);
        }

        bound_check<arma::vec>(output, "dnbinom", false, true);
        return output;
    }


    /**
     * @brief P.M.F of negative-binomial distribution.
     * 
     * @param y Number of failures before r successes
     * @param kappa Probability of failures
     * @param r Number of successes wanted.
     * @return double 
     */
    static double dnbinom(const double &y, const double &kappa, const double &r)
    {
        double c3 = std::pow(1. - kappa, r);

        double a = y + r - 1.;
        double c1 = binom(a, y);
        double c2 = std::pow(kappa, y);
        double output = (c1 * c2) * c3;

        return output;
    }

    /**
     * @brief P.M.F of negative-binomial distribution.
     *
     * @param y Number of failures before r successes
     * @param kappa Probability of failures
     * @param r Number of successes wanted.
     * @param c3 (1-kappa)^r
     * @return double
     */
    static double dnbinom(const double &y, const double &kappa, const double &r, const double &c3)
    {
        double a = y + r - 1.;
        double c1 = binom(a, y);
        double c2 = std::pow(kappa, y);
        double output = (c1 * c2) * c3;

        return output;
    }

    double sample()
    {
        return R::rnbinom(par2, 1. - par1);
    } 

    static double sample(const double &kappa, const double &r)
    {
        return R::rnbinom(r, 1. - kappa);
    }

    /**
     * @brief Iterative form as in Solow(1960): c(r,1)(-kappa)^1, ..., c(r,r)(-kappa)^r
     *
     * @param kappa
     * @param r
     * @return arma::vec
     */
    static arma::vec iter_coef(const double &kappa, const double &r)
    {
        unsigned int _r = static_cast<unsigned int>(r);
        arma::vec coef(_r, arma::fill::zeros);
        for (unsigned int k = 0; k <_r; k ++)
        {
            double c1 = binom(r, k+1);
            double c2 = std::pow(-kappa, k+1);
            coef.at(k) = -c1 * c2; // coef[0]=c(r,1)(-kappa)^1, ..., coef[_r-1]=c(r,r)(-kappa)^r
        }

        bound_check<arma::vec>(coef, "nbinom::iter_coef: coef");
        return coef;
    }

    static double coef_now(const double &kappa, const double &r)
    {
        return std::pow(1. - kappa, r);
    }

private:
    unsigned int _r;
};

/**
 * @brief Discretized log-normal distribution characterized by mean and variance.
 *
 */
class lognorm : public Dist
{
public:
    lognorm() : Dist()
    {
        _name = "lognorm";
        _par1 = LN_MU;
        _par2 = LN_SD2;
    }

    
    lognorm(
        const double &mu,
        const double &sd2) : Dist()
    {
        _name = "lognorm";
        _par1 = mu;
        _par2 = sd2;
    }

    static double mean(const double &mu, const double &sd2)
    {
        double val = mu + 0.5 * sd2;
        return std::exp(val);
    }

    static double var(const double &mu, const double &sd2)
    {
        double val1 = 2. * mu + 0.5 * sd2;
        val1 = std::exp(val1);
        double val2 = std::exp(sd2) - 1.;
        return val1 * val2;

    }


    static double mode(const double &mu, const double &sd2)
    {
        double val = mu - sd2;
        return std::exp(val);
    }

    /**
     * @brief mean of the corresponding serial distribution
     * 
     * @param mu 
     * @param sd2 
     * @return double 
     */
    static double mean_serial(const double &mu, const double &sd2)
    {
        double b2 = std::exp(2 * mu + sd2);
        return std::sqrt(b2);
    }

    static double var_serial(const double &mu, const double &sd2)
    {
        double a = std::exp(sd2) - 1.;
        a *= std::exp(2*mu + sd2);
        return a;
    }

    static Rcpp::NumericVector lognorm2serial(const double &mu, const double &sd2)
    {
        double s2 = lognorm::var_serial(mu, sd2);
        double m = lognorm::mean_serial(mu, sd2);

        Rcpp::NumericVector out = {m, s2};
        return out;
    }

    static Rcpp::NumericVector serial2lognorm(const double &m, const double &s2)
    {
        double m2 = m * m;
        double mu = std::log(m2 / std::sqrt(s2 + m2));
        double sd2 = std::log(1. + s2 / m2);

        Rcpp::NumericVector out = {mu, sd2};
        return out;
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
    arma::vec dlognorm(const unsigned int &nL)
    {
        arma::vec output(nL);
        for (unsigned int d = 0; d < nL; d++)
        {
            output.at(d) = dlognorm0(static_cast<double>(d) + 1., _par1, _par2);
        }

        return output;
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
            output.at(d) = dlognorm0(static_cast<double>(d) + 1., mu, sd2);
        }

        return output;
    }

private:
    /**
     * @brief C.D.F of log-normal distribution. The Pd function in Koyama (2021).
     *
     * @param d
     * @param mu double: the mean of the log-normal distribution.
     * @param sigmasq double: the variance of the log-normal distribution.
     * @return double
     */
    static double plognorm(
        const double d,
        const double mu,
        const double sigmasq)
    {
        arma::vec tmpv1(1);
        tmpv1.at(0) = -(std::log(d) - mu) / std::sqrt(2. * sigmasq);
        return arma::as_scalar(0.5 * arma::erfc(tmpv1));
    }

    /**
     * @brief P.M.F of discretized log-normal distribution. p(d) = Pd(d) - Pd(d - 1).
     *
     * @param lag
     * @param mu
     * @param sd2
     * @return double
     */
    static double dlognorm0(
        const double &lag, // starting from 1
        const double &mu,
        const double &sd2)
    {
        double output = plognorm(lag, mu, sd2) - plognorm(lag - 1., mu, sd2);
        bound_check(output, "dlognorm0", false, true);
        return output;
    }

};

#endif