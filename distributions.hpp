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


    static arma::vec rmvnorm(const arma::vec &loc, const arma::mat &Prec)
    {
        arma::vec epsilon = arma::randn(loc.n_elem);
        arma::mat precision_chol = arma::chol(Prec);
        arma::vec scaled_mu = arma::solve(arma::trimatu(precision_chol.t()), loc);
        arma::vec x = arma::solve(arma::trimatu(precision_chol), epsilon + scaled_mu);

        return x;
    }

    static arma::vec rmvnorm2(const arma::vec &loc, const arma::mat &Prec)
    {

        arma::vec epsilon = arma::randn(loc.n_elem);
        arma::mat precision_chol = arma::chol(Prec);
        arma::mat precision_chol_inv = arma::trans(arma::inv(arma::trimatu(precision_chol)));
        arma::vec x = arma::trans(precision_chol_inv) * (precision_chol_inv * loc + epsilon);

        return x;
    }

    /**
     * @brief A potentially more effective way to calculate multivariate normal density for Bayesian inference.
     *
     * @param z z = prec_rchol_inv.t() * loc + eps, where eps is a sample from a standard multivariate normal; a sample from this multivariate normal is x = prec_rchol_inv * z.
     * @param loc Location, the mean of this multivariate normal is mu = Sigma * loc, where Sigma is variance.
     * @param prec_rchol_inv Left Cholesky of the variance Sigma.
     * @param return_log
     * @return double
     */
    static double dmvnorm0(const arma::vec &z, const arma::vec &loc, const arma::mat &prec_rchol_inv, const bool &return_log = true)
    {
        const double p = static_cast<double>(z.n_elem);
        double c = p * std::log(2. * arma::datum::pi);
        c += 2. * arma::accu(arma::log(prec_rchol_inv.diag())); // determinant of variance [checked. OK.]
        c += arma::dot(z, z);

        arma::vec ll = prec_rchol_inv.t() * loc;
        c -= 2. * arma::dot(z, ll);
        c += arma::dot(ll, ll);

        c *= -0.5;
        if (!return_log)
        {
            c = std::exp(c);
        }

        return c;
    }

    static double dmvnorm(const arma::vec &x, const arma::vec &mu, const arma::mat &Sigma, const bool &return_log = true)
    {
        const double p = static_cast<double>(x.n_elem);
        double c = p * std::log(2. * arma::datum::pi);
        c += arma::log_det_sympd(arma::symmatu(Sigma));

        arma::vec diff = x - mu;
        arma::mat prec = arma::inv(Sigma);

        c += arma::as_scalar(diff.t() * prec * diff);

        c *= -0.5;
        if (!return_log)
        {
            c = std::exp(c);
        }

        return c;
    }

    static double dmvnorm2(const arma::vec &x, const arma::vec &mu, const arma::mat &Prec, const bool &return_log = true)
    {
        double logdet = arma::log_det_sympd(Prec);
        double cnst = static_cast<double>(mu.n_elem) * LOG2PI;

        arma::vec diff = x - mu;
        double dist = arma::as_scalar(diff.t() * Prec * diff);

        double logprob = -cnst + logdet - dist;
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


    static double dmvnorm2(
        const arma::vec &x, 
        const arma::vec &mu, 
        const arma::mat &Prec, 
        const double &logdet_prec, 
        const bool &return_log = true)
    {
        double cnst = static_cast<double>(mu.n_elem) * LOG2PI;

        arma::vec diff = x - mu;
        double dist = arma::as_scalar(diff.t() * Prec * diff);

        double logprob = -cnst + logdet_prec - dist;
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
    Beta() : Dist(), alpha(par1), beta(par2)
    {
        name = "beta";
        par1 = 1.;
        par2 = 1.;
    }
    
    Beta(const double &alpha_, const double &beta_) : Dist(), alpha(par1), beta(par2)
    {
        name = "beta";
        par1 = alpha_;
        par2 = beta_;
    }

    Beta(const Dist &dist_) : Dist(), alpha(par1), beta(par2)
    {
        name = "beta";
        par1 = dist_.par1;
        par2 = dist_.par2;
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
    InverseGamma() : Dist(), alpha(par1), beta(par2) 
    {
        name = "invgamma";
        par1 = 0.01;
        par2 = 0.01;
    }

    InverseGamma(const double &shape, const double &rate) : Dist(), alpha(par1), beta(par2)
    {
        name = "invgamma";
        par1 = shape;
        par2 = rate;
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

    /**
     * @brief Draw one sample from the inverse-gamma distribution.
     * 
     * @param alpha shape parameter
     * @param beta rate parameter of the corresponding gamma distribution; scale parameter for inverse-gamma distribution.
     * @return double 
     */
    static double sample(const double &alpha, const double &beta)
    {
        double out = 1. / R::rgamma(alpha, 1. / beta);
        #ifdef DGTF_DO_BOUND_CHECK
            bound_check(out, "InverseGamma::sample: out");
        #endif
        return out;
    }


    static double dinvgamma(
        const double &x, 
        const double &alpha, 
        const double &beta, 
        const bool &logx = true, // if true, return the density of log(x), IG x Jacobian
        const bool &logp = true) // if true, return the logarithm of the density
    {
        double tau2 = std::abs(1. / x); // tau2 ~ Gamma
        double logtau2 = std::log(tau2 + EPS);

        double out = R::dgamma(tau2, alpha, 1./beta, true);
        if (logx) // plus jacobian
        {
            out += logtau2;
        }
        
        
        if (!logp)
        {
            out = std::exp(out);
        }

        #ifdef DGTF_DO_BOUND_CHECK
            bound_check(out, "InverseGamma::dinvgamma: out");
        #endif
        return out;
    }
};

class Gamma : public Dist
{
public:
    Gamma() : Dist(), alpha(par1), beta(par2)
    {
        name = "gamma";
        par1 = 1.;
        par2 = 1.;
    }

    Gamma(const double &shape, const double &rate) : Dist(), alpha(par1), beta(par2)
    {
        name = "gamma";
        par1 = shape;
        par2 = rate;
    }

    Gamma(const Dist &dist_) : Dist(), alpha(par1), beta(par2)
    {
        name = "gamma";
        par1 = dist_.par1;
        par2 = dist_.par2;
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


    static double dgamma_discrete(const double &lag, const double &alpha, const double &beta)
    {
        if (lag < 0)
        {
            throw std::invalid_argument("Gamma::dgamma_discrete: lag must be nonnegative");
        }

        double out = 0.;
        if (lag > 0)
        {
            out += R::pgamma(lag, alpha, 1./beta, true, false);
        }
        if (lag > 1)
        {
            out -= R::pgamma(lag - 1, alpha, 1./beta, true, false);
        }

        #ifdef DGTF_DO_BOUND_CHECK
            bound_check(out, "Gamma::dgamma_discrete: out");
        #endif
        
        return out;
    }

};

class Poisson : public Dist
{
public:
    Poisson() : Dist(), lambda(par1)
    {
        name = "poisson";
        par1 = 1.;
        par2 = 0.;
    }

    Poisson(const double &lambda_, const double &par2_in = 0.) : Dist(), lambda(par1)
    {
        name = "poisson";
        par1 = lambda_;
        par2 = par2_in;
    }

    Poisson(const Dist &dist_) : Dist(), lambda(par1)
    {
        name = "poisson";
        par1 = dist_.par1;
        par2 = dist_.par2;
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


    static double dlogp_dlambda(const double &lambda, const double &yt)
    {
        double output = (yt / lambda) - 1.;
        #ifdef DGTF_DO_BOUND_CHECK
            bound_check(output, "Poisson::dlogp_dlambda: output");
        #endif
        return output;
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
    nbinomm() : Dist(), nu(_nu), lambda(par1), delta(par2)
    {
        name = "nbinomm";
        par1 = NB_LAMBDA;
        par2 = NB_DELTA;
        mean2conj();
        return;
    }

    const double &nu;
    const double &lambda;
    const double &delta;

    nbinomm(const double &lambda, const double &delta) : Dist(), nu(_nu), lambda(par1), delta(par2)
    {
        name = "nbinomm";
        par1 = lambda;
        par2 = delta;
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
        return par1;
    }

    static double mean(const double &lambda, const double &delta)
    {
        return lambda;
    }

    static double var(const double &lambda, const double &delta)
    {
        return lambda * (1. + lambda / delta);
    }

    double mean2conj()
    {
        _nu = par1 / (par1 + par2);
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


    static double dlogp_dlambda(const double &lambda, const double &yt, const double &par2)
    {
        if (lambda < 0 || par2 < 0)
        {
            throw std::invalid_argument("nbinomm::dlogp_dlambda: lambda and rho must be positive.");
        }
        double c1 = yt / (lambda + EPS);
        double nom = yt + par2;
        double denom = lambda + par2;
        double c2 = nom / denom;

        double output = c1 - c2;
        #ifdef DGTF_DO_BOUND_CHECK
            bound_check(output, "nbinomm::dlogp_dlambda: output");
        #endif
        return output;
    }


    static double dlogp_dpar2(const double &yt, const double &lambda, const double &par2, const bool &jacobian = true)
    {
        double out = R::digamma(yt + par2) - R::digamma(par2);
        out += std::log(par2 / (lambda + par2));
        out += (lambda - yt) / (lambda + par2);

        if (jacobian)
        {
            out *= par2;
        }

        #ifdef DGTF_DO_BOUND_CHECK
            bound_check(out, "nbinomm::dlogp_dpar2: out");
        #endif
        return out;
    }



private:
    double _nu;
};

/**
 * @brief Negative-binomial distribution, characterized by `nbinomp`: probability (kappa, r).
 *
 * @param _name "nbinomp" (1-kappa)^r * kappa^y
 * @param _par1 kappa: probability of failures (y is number of failures)
 * @param _par2 r: number of successes.
 */
class nbinom : public Dist
{
public:
    nbinom() : Dist()
    {
        _r = static_cast<unsigned int>(NB_R);

        name = "nbinomp";
        par1 = NB_KAPPA;
        par2 = NB_R;
        return;
    }


    nbinom(const double kappa, const double r) : Dist()
    {
        _r = static_cast<unsigned int>(r);

        name = "nbinomp";
        par1 = kappa;
        par2 = r;
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

        #ifdef DGTF_DO_BOUND_CHECK
            bound_check<arma::vec>(output, "dnbinom", false, true);
        #endif
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
        double c1 = binom(r + y - 1, y);
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
        double c1 = binom(r + y - 1, y);
        double c2 = std::pow(kappa, y);
        double output = (c1 * c2) * c3;

        return output;
    }

    static double dlag_dlogitkappa(const double &y, const double &kappa, const double &r)
    {
        if (r < 1)
        {
            throw std::invalid_argument("nbinom::dlag_dlogitkappa only valid if r >= 1.");
        }

        double c1 = std::pow(kappa, y - 1);
        double c2 = std::pow(1. - kappa, r - 1);
        double c3 = (1. - kappa) * y - r * kappa;
        double c4 = binom(r + y - 1, y);
        double dlag_dkappa = c1 * c2 * c3 * c4;

        double dkappa_dlogit = kappa * (1. - kappa);
        
        double out = dlag_dkappa * dkappa_dlogit;

        #ifdef DGTF_DO_BOUND_CHECK
            bound_check(out, "nbinom::dlag_dlogitkappa: out");
        #endif
        return out;
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
    static arma::vec iter_coef(const double &kappa, const double &r, const bool &reciprocal = false)
    {
        unsigned int _r = static_cast<unsigned int>(r);
        arma::vec coef(_r, arma::fill::zeros);
        for (unsigned int k = 0; k <_r; k ++)
        {
            double c1 = binom(r, k+1);
            double power = static_cast<double>(k + 1);
            power *= reciprocal ? -1 : 1;
            double c2 = std::pow(-kappa, power);
            coef.at(k) = -c1 * c2; // coef[0]=-c(r,1)(-kappa)^1, ..., coef[_r-1]=-c(r,r)(-kappa)^r
        }

        #ifdef DGTF_DO_BOUND_CHECK
            bound_check<arma::vec>(coef, "nbinom::iter_coef: coef");
        #endif
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
        name = "lognorm";
        par1 = LN_MU;
        par2 = LN_SD2;
    }

    
    lognorm(
        const double &mu,
        const double &sd2) : Dist()
    {
        name = "lognorm";
        par1 = mu;
        par2 = sd2;
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
            output.at(d) = dlognorm0(static_cast<double>(d) + 1., par1, par2);
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
        arma::vec output(nL);
        for (unsigned int d = 0; d < nL; d++)
        {
            output.at(d) = dlognorm0(static_cast<double>(d) + 1., mu, sd2);
        }

        return output;
    }

    /**
     * @brief Derivative of discretized log-normal P.M.F w.r.t v := (log(lag) - mu) / sd.
     *
     * @param lag
     * @param mu
     * @param sd2
     * @return arma::vec
     */
    static arma::vec dlag_dpar(
        const double &lag,
        const double &mu,
        const double &sd2)
    {
        if (lag < EPS)
        {
            throw std::invalid_argument("deriv_dlognorm: lag must be nonnegative.");
        }

        double sig = std::sqrt(sd2 + EPS);
        double dv_dmu = -1. / sig;

        double v = (std::log(lag + EPS) - mu) / sig;
        double dlag_dv = R::dnorm4(v, 0., 1., false); // phi(v(l,mu,sigma))
        double dv_dlogsig2 = -0.5 * v; // dv/dlog(sigma2) = dv/dsigma * dsigma/dlog(sigma2) = - 0.5 * v

        double dlag_dmu = dlag_dv * dv_dmu;
        double dlag_dlogsig2 = dlag_dv * dv_dlogsig2;

        if (lag > 1)
        {
            double lag2 = lag - 1;
            v = (std::log(lag2 + EPS) - mu) / sig;
            dlag_dv = R::dnorm4(v, 0., 1., false);
            dv_dlogsig2 = -0.5 * v;

            dlag_dmu -= dlag_dv * dv_dmu;
            dlag_dlogsig2 -= dlag_dv * dv_dlogsig2;
        }

        arma::vec out = {dlag_dmu, dlag_dlogsig2};

        #ifdef DGTF_DO_BOUND_CHECK
            bound_check<arma::vec>(out, "lognorm::dlag_dpar: out");
        #endif
        return out;
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
        double output = plognorm(lag, mu, sd2);
        if (lag > 1) 
        {
            output -= plognorm(lag - 1., mu, sd2);
        }
        
        #ifdef DGTF_DO_BOUND_CHECK
            bound_check(output, "dlognorm0", false, true);
        #endif
        return output;
    }

};

#endif