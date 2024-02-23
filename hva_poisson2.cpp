#include "pl_poisson.h"
#include "yjtrans.h"
#include "VariationalBayes.hpp"

using namespace Rcpp;
// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::depends(RcppArmadillo)]]


/*
Transform eta = (W,mu0,rho,M) to eta_tilde on the real space R^4
*/
inline void eta2tilde(
    arma::vec &eta_tilde,         // m x 1
    const arma::vec &eta,         // m x 1
    const arma::uvec &idx_select, // m x 1 (m<=4)
    const unsigned int &W_prior_type)
{
    eta_tilde.set_size(idx_select.n_elem);
    for (unsigned int i = 0; i < idx_select.n_elem; i++)
    {
        double val = eta.at(i);
        unsigned int idx = idx_select.at(i);
        // double etmp = eta.at(idx_select.at(i));
        switch(idx) {
            case 0: // W   - Wtilde = log(W)
            {
                bound_check(val, "eta2tilde: W", true, true);
                switch (W_prior_type)
                {
                case 0: // gamma
                    eta_tilde.at(i) = std::log(std::abs(val) + EPS);
                    break;
                case 1:
                    throw std::invalid_argument("Jacobian of Cauchy W is undefined.");
                    break;
                case 2: // inverse gamma
                    eta_tilde.at(i) = -std::log(std::abs(val) + EPS);
                    break;
                default:
                    break;
                }
            }
            break;
            case 1: // mu0 - mu0_tilde = log(mu[0])
            {
                bound_check(val, "eta2tilde: mu0", false, true);
                eta_tilde.at(i) = std::log(std::abs(val) + EPS);
            }
            break;
            case 2: // rho - rho_tilde = log(rho/(1.-rho))
            {
                bound_check(val, "eta2tilde: rho", 0., 1.);
                eta_tilde.at(i) = std::log(std::abs(val) + EPS) - std::log(std::abs(1. - val) + EPS);
            }
            break;
            default:
            {
                 throw std::invalid_argument("eta2tilde - idx_select out of bound.");
            }
        }
    }

    return;
}

inline void tilde2eta(
    arma::vec &eta, // m x 1
    const arma::vec &eta_tilde,   // m x 1
    const arma::uvec &idx_select, // m x 1
    const unsigned int &W_prior_type = 0)
{
    eta.set_size(idx_select.n_elem);
    for (unsigned int i = 0; i < idx_select.n_elem; i++)
    {
        unsigned int idx = idx_select.at(i);
        double val = eta_tilde.at(i);
        switch (idx)
        {
        case 0: // W = exp(Wtilde)
        {
            switch (W_prior_type)
            {
            case 0:
                val = std::min(val, UPBND);
                eta.at(i) = std::exp(val);
                break;
            case 1:
                throw std::invalid_argument("Jacobian of Cauchy W is undefined.");
                break;
            case 2:
                val *= -1.;
                val = std::min(val, UPBND);
                eta.at(i) = std::exp(val);
                break;
            default:
                break;
            }
            bound_check(eta.at(i), "tilde2eta: W", true, true);
        }
        break;
        case 1: // mu0 = exp(mu0_tilde)
        {
            val = std::min(val, UPBND);
            eta.at(i) = std::exp(val);
            bound_check(eta.at(i), "tilde2eta: mu0", false, true);
        }
        break;
        case 2: // rho = exp(rho_tilde) / (1 + exp(rho_tilde))
        {
            val = std::min(val, UPBND);
            double tmp = std::exp(val);
            eta.at(i) = tmp;
            eta.at(i) /= (1. + tmp);
            bound_check(eta.at(i), "tilde2eta: rho", 0., 1.);
        }
        break;
        default:
        {
            throw std::invalid_argument("tilde2eta - idx_select out of bound.");
        }
        }
    }

    return;
}

/*
------ dlogJoint_dWtilde ------
The derivative of the full joint density with respect to Wtilde=log(1/W), where W
is the evolution variance that affects
    psi[t] | psi[t-1] ~ N(psi[t-1], W)

Therefore, this function remains unchanged as long as we using the same evolution equation for psi.
*/
inline double dlogJoint_dWtilde(
    const arma::vec& psi, // (n+1) x 1, (psi[0],psi[1],...,psi[n])
    const double G, // evolution transition matrix
    const double W, // evolution variance conditional on V
    const arma::vec &W_par) { // 0 - Gamma(aw=shape,bw=rate); 1 - Half-Cauchy(aw=location=0, bw=scale); 2 - Inverse-Gamma(aw=shape,bw=rate)

    double aw = W_par.at(2);
    double bw = W_par.at(3);
    unsigned int W_prior_type = W_par.at(1);

    if (!psi.is_finite() || !std::isfinite(W))
    {
        throw std::invalid_argument("dlogJoint_dWtilde<double>: non-finite psi or W input.");
    }
    
    const double n = static_cast<double>(psi.n_elem) - 1;
    const unsigned int ni = psi.n_elem - 1;
    double res = 0.;
    for (unsigned int t=0; t<ni; t++) {
        res += (psi.at(t+1)-G*psi.at(t)) * (psi.at(t+1)-G*psi.at(t));
    }
    // res = sum((w[t])^2)

    double rdw = std::log(res+EPS) - std::log(W+EPS); // sum(w[t]^2) / W
    rdw = std::min(rdw, UPBND);
    rdw = std::exp(rdw);
    bound_check(rdw,"dlogJoint_dWtilde: rdw");

    double deriv;
    if (W_prior_type == 0)
    {
        /*
        W ~ Gamma(aw=shape, bw=rate)
        Wtilde = log(W)
        */
        // deriv = 0.5*n-0.5/W*res - aw + bw*W;
        deriv = aw;
        deriv -= 0.5*n ;
        deriv -= bw*W;
        deriv += 0.5 * rdw;
        // deriv = aw - 0.5*n - bw*W + 0.5 * rdw;
    }
    else if (W_prior_type == 1)
    {
        /*
        sqrt(W) ~ Half-Cauchy(aw=location==0, bw=scale)
        */
        deriv = 0.5*n-0.5 * rdw + W/(bw*bw+W) - 0.5;
    }
    else if (W_prior_type == 2)
    {
        /*
        W ~ Inverse-Gamma(aw=shape, bw=rate)
        Wtilde = -log(W)
        (deprecated)
        */
        // deriv = res;
        // deriv *= 0.5;
        // deriv += bw;
        // deriv = -rdw; // TYPO?
        // deriv += aw + 0.5*n;

        double bnew = bw + 0.5*res;
        double log_bnew_W = std::log(bnew) - std::log(W+EPS);
        log_bnew_W = std::min(log_bnew_W, UPBND);
        deriv = - std::exp(log_bnew_W);

        double a_new = aw;
        a_new += 0.5*n;
        a_new += 1.;
        deriv += a_new;
    }
    else
    {
        throw std::invalid_argument("Unsupported prior for evolution variance W.");
    }

    bound_check(deriv, "dlogJoint_dWtilde: deriv");
    return deriv;
}

inline double logprior_Wtilde(
    const double &Wtilde, // evolution variance conditional on V
    const double &aw = 0.01,
    const double &bw = 0.01,
    const unsigned int &W_prior_type = 2)
{
    double InvGamma_cnst = aw;
    InvGamma_cnst *= std::log(bw);
    InvGamma_cnst -= std::lgamma(aw);

    double logp = -16.;
    if (W_prior_type==0) {
        /*
        W ~ Gamma(aw=shape, bw=rate)
        Wtilde = log(W)
        */
        logp = aw*std::log(bw);
        logp -= std::lgamma(aw);
        logp += aw*Wtilde;

        double Wast = std::min(Wtilde,UPBND);
        logp -= bw * std::exp(Wast);
    }
    else if (W_prior_type == 1)
    {
        /*
        sqrt(W) ~ Half-Cauchy(aw=location==0, bw=scale)
        */
        throw std::invalid_argument("logprior_Wtilde for Half-Cauchy is not implemented yet.");
    }
    else if (W_prior_type == 2)
    {
        /*
        W ~ Inverse-Gamma(aw=shape, bw=rate)
        Wtilde = -log(W)
        */
        logp = InvGamma_cnst;
        logp += aw*Wtilde;
        double Wast = std::min(Wtilde, UPBND);
        logp -= bw * std::exp(Wast);
    }
    else
    {
        throw std::invalid_argument("Unsupported prior for evolution variance W.");
    }

    bound_check(logp,"logprior_Wtilde: logp");

    return logp;
}

/*
------ dlogJoint_dlogmu0 ------
The derivative of the full joint density with respect to logmu0=log(mu0), where mu0
is the baseline intensity such that:
    y[t] ~ EFM(lambda[t]=mu[0]+lambda[t])

mu0 > 0 has a Gamma prior:
    mu[0] ~ Gamma(amu,bmu)
*/
inline double dloglike_dmu0tilde(
    const arma::vec &ypad,  // (n+1) x 1
    const arma::vec &theta, // (n+1) x 1, (theta[0],theta[1],...,theta[n])
    const arma::vec &obs_par,
    const unsigned int obs_code = 0)
{ // 0 - negative binomial; 1 - poisson
    double mu0 = obs_par.at(0);
    double delta_nb = obs_par.at(1);

    double dy_dlambda = 0.;
    for (unsigned int t = 1; t < ypad.n_elem; t++)
    {
        double lambda = mu0 + theta.at(t);
        dy_dlambda += dloglike_dlambda(
            ypad.at(t), lambda, delta_nb, obs_code);
    }

    double dlambda_dmu = 1.;
    double dmu_dmu0tilde = mu0;
    double deriv = (dy_dlambda * dlambda_dmu) * dmu_dmu0tilde;
    return deriv;
}


// double dloglike_dmu0tilde(
//     const arma::vec &ypad,  // (n+1) x 1
//     const arma::vec &theta, // (n+1) x 1, (theta[0],theta[1],...,theta[n])
//     const arma::vec &hpsi_pad, // (n+1) x 1
//     const arma::vec &lag_par,
//     const arma::vec &obs_par,
//     const unsigned int &nlag_in = 20,
//     const unsigned int &obs_code = 0,
//     const unsigned int &trans_code = 1,
//     const bool &truncated = true)
// { // 0 - negative binomial; 1 - poisson
//     unsigned int nobs = ypad.n_elem - 1;
//     double mu0 = obs_par.at(0);
//     double delta_nb = obs_par.at(1);

//     double dy_dmu0 = 0.;
//     for (unsigned int t = 1; t <= nobs; t++)
//     {
//         double lambda = mu0 + theta.at(t);
//         double dy_dlambda = dloglike_dlambda(
//             ypad.at(t), lambda, delta_nb, obs_code);

//         unsigned int nelem = theta_nelem(nobs, nlag_in, t, truncated);
//         arma::vec Fphi_sub; arma::vec hpsi_sub;
//         theta_subset(
//             Fphi_sub, hpsi_sub, 
//             hpsi_pad, lag_par,
//             t, nelem, trans_code);
//         arma::vec coef = Fphi_times_hpsi(Fphi_sub, hpsi_sub);
//         double dlambda_dmu0 = 1. - arma::accu(coef);

//         dy_dmu0 += dy_dlambda * dlambda_dmu0;
//     }

//     double dmu0_dmu0tilde = mu0;
//     double deriv = dy_dmu0 * dmu0_dmu0tilde;
//     return deriv;
// }

inline double logprior_mu0tilde(
    const double &logmu0, // evolution variance conditional on V
    const double &amu,
    const double &bmu) {

    /*
    mu0 ~ Gamma(aw=shape, bw=rate)
    logmu0 = log(mu0)
    */
    double logp = amu*std::log(bmu) - std::lgamma(amu) + amu*logmu0 - bmu*std::exp(logmu0);
    bound_check(logp,"logprior_logmu0: logp");
    return logp;
}

inline double logprior_mu0tilde(
    const double &logmu0, // evolution variance conditional on V
    const double &sig2_mu0 = 10.)
{
    /*
    log(mu0) ~ N(0,sig2_mu0)
    */
    double logp = - logmu0 / sig2_mu0;
    bound_check(logp, "logprior_logmu0: logp");
    return logp;
}

/**
 * dlogJoint_dpar1: partial derivative w.r.t rho or mu.
 * 
 * @param lag_par: either (rho, L) or (mu, sg2)
*/
inline double dlogJoint_dpar1(
    const arma::vec& ypad, // (n+1) x 1
    const arma::mat& R, // (n+1) x 2, (psi,theta)
    const arma::vec& lag_par,
    const double mu0,
    const double delta_nb,
    const unsigned int gain_code) 
{

    double aprior = 1.;
    double bprior = 1.;
    double par1 = lag_par.at(0);

    const unsigned int n = ypad.n_elem - 1;
    unsigned int L = (unsigned int) lag_par.at(1);
    double L_ = lag_par.at(1);

    arma::vec hpsi = psi2hpsi(R.col(0),gain_code); // (n+1) x 1
    arma::vec lambda = lag_par.at(0) + R.col(1);         // (n+1) x 1

    double c1 = 0.;
    double c2 = 0.;

    for (unsigned int t=1; t<=n; t++) {
        double c10 = ypad.at(t)/lambda.at(t) - (ypad.at(t)+delta_nb)/(lambda.at(t)+delta_nb);
        c1 += c10*hpsi.at(t-1)*ypad.at(t-1);

        unsigned int r = std::min(t,L);
        double c20 = 0.;
        double c21 = -par1;
        for (unsigned int k=1; k<=r; k++) {
            c20 += static_cast<double>(k)*binom(L,k)*c21*R.at(t-k,1);
            c21 *= -par1;
        }
        c2 += c10*c20;
    }

    double deriv = -L_ * std::pow(1. - par1, L_) * par1 * c1 - (1. - par1) * c2;
    deriv += aprior - (bprior + aprior) * par1;
    bound_check(deriv,"dlogJoint_drho: deriv");
    return deriv;
}



inline double dlogJoint_drho2(
    const arma::vec& ypad, // (n+1) x 1
    const arma::mat& R, // (n+1) x 2, (psi,theta)
    const unsigned int L,
    const double mu0,
    const double rho,
    const double delta_nb,
    const unsigned int gain_code,
    const arma::vec& rcomb, // n x 1
    const double arho = 1.,
    const double brho = 1.) {

    const unsigned int n = ypad.n_elem - 1;
    const double L_ = static_cast<double>(L);

    arma::vec hpsi = psi2hpsi(R.col(0),gain_code); // (n+1) x 1
    arma::vec hy = hpsi % ypad; // (n+1) x 1, (h(psi[0])*y[0], h(psi[1])*y[1], ...,h(psi[n])*y[n])
    arma::vec lambda = mu0 + R.col(1); // (n+1) x 1

    double c10 = 0.; // d(Loglike) / d(theta[t])
    double c20 = 0.; // first part of d(theta[t]) / d(logit(rho)), l=0,...,t-1
    double c30 = 0.; // second part of d(theta[t]) / d(logit(rho)), l=1,...,t-1
    double c2 = 0.; // first part of d(Loglike) / d(logit(rho))
    double c3 = 0.; // second part of d(Loglike) / d(logit(rho))
    double crho = 1.;

    for (unsigned int t=1; t<=n; t++) {
        crho = 1.;
        c10 = ypad.at(t)/lambda.at(t) - (ypad.at(t)+delta_nb)/(lambda.at(t)+delta_nb); // d(Loglike) / d(theta[t])
        c20 = rcomb.at(0)*crho*hy.at(t-1);
        c30 = 0.;

        for (unsigned int l=1; l<t; l++) {
            crho *= rho;
            c20 += rcomb.at(l) * crho * hy.at(t-1-l);
            c30 += static_cast<double>(l) * rcomb.at(l) * crho * hy.at(t-1-l);
        }

        c2 += c10 * c20;
        c3 += c10 * c30;
    }

    double deriv = -L_*std::pow(1.-rho,L_)*rho*c2 + std::pow(1.-rho,L_+1.)*c3;
    deriv += arho - (arho+brho) * rho;
    bound_check(deriv, "dlogJoint_drho2: deriv");
    return deriv;
}


inline double logprior_logitrho(
    double logitrho,
    double arho = 1.,
    double brho = 1.) {
    
    double logp = std::lgamma(arho+brho) - std::lgamma(arho) - std::lgamma(brho) + arho*logitrho - (arho+brho)*std::log(1.+std::exp(logitrho));
    bound_check(logp,"logprior_logitrho: logp");
    return logp;
}




// eta_tilde = (Wtilde,logmu0)
inline arma::vec dlogJoint_deta(
    const arma::vec& ypad, // (n+1) x 1
    const arma::vec& psi, // (n+1) x 1
    const arma::vec& theta, // (n+1) x 1
    const arma::vec& eta, // m x 1
    const arma::uvec& idx_select, // m x 1,
    const arma::vec &W_par,
    const arma::vec &lag_par,
    const arma::vec& rcomb,
    const double &delta_nb = 30.,
    const unsigned int &obs_code = 0, // 0 - NegBinom; 1 - Poisson
    const unsigned int &gain_code = 3) 
{
    arma::vec deriv(idx_select.n_elem);
    for (unsigned int i = 0; i < idx_select.n_elem; i++)
    {
        double val = eta.at(i);
        unsigned int idx = idx_select.at(i);

        switch (idx)
        {
        case 0: // W, Wtilde = -log(W)
        {
            deriv.at(i) = dlogJoint_dWtilde(psi, 1., val, W_par);
        }
        break;
        case 1: // mu0, mu0tilde = log(mu0)
        {
            double mu0tilde = std::log(val + EPS);
            arma::vec obs_par = {val, delta_nb};

            deriv.at(i) = dloglike_dmu0tilde(
                ypad, theta, obs_par, obs_code);
            deriv.at(i) += logprior_mu0tilde(mu0tilde, 10.);
            // Assume mu0tilde use a flat prior
        }
        break;
        case 2: // rho, rho_tilde = log(rho/(1-rho))
        {
            deriv.at(i) = dlogJoint_drho2(ypad, theta, lag_par.at(1), eta.at(1), eta.at(2), delta_nb, gain_code, rcomb, 1., 1.);
        }
        break;
        default:
        {
            throw std::invalid_argument("dlogJoint_deta - idx_select undefined.");
        }
        }
    }

    return deriv;
}

inline arma::vec update_step_size_adadelta(
    arma::vec &curEg2,
    const arma::vec &curEdelta2,
    const arma::vec &dL_dpar,
    const double &learn_rate = 0.01,
    const double &eps_step = 1.e-6)
{
    arma::vec oldEg2 = curEg2;

    curEg2 = (1. - learn_rate) * oldEg2 + learn_rate * arma::pow(dL_dpar, 2.); // m x 1
    arma::vec rho = arma::sqrt(curEdelta2 + eps_step) / arma::sqrt(curEg2 + eps_step);
    bound_check(rho, "update_step_size_adadelta: rho");

    return rho;
}


inline arma::vec update_vb_param(
    arma::vec &curEg2,
    arma::vec &curEdelta2,
    const arma::vec &dL_dVecPar,
    const double &learn_rate = 0.01,
    const double &eps_step = 1.e-6)
{
    arma::vec rho = update_step_size_adadelta(curEg2, curEdelta2, dL_dVecPar, learn_rate, eps_step);
    arma::vec Change_delta = rho % dL_dVecPar;

    arma::vec oldEdelta2 = curEdelta2;
    curEdelta2 = (1. - learn_rate) * oldEdelta2 + learn_rate * arma::pow(Change_delta, 2.);
    bound_check(curEdelta2, "update_vb_param: curEdelta2");
    return Change_delta;
}



inline void init_eta(
    arma::uvec &idx_select, // m x 1
    arma::vec &eta, // m x 1
    const arma::uvec &eta_select,
    const arma::vec &obs_par,
    const arma::vec &lag_par,
    const arma::vec &W_par,
    const bool &update_static
)
{
    if (update_static)
    {
        idx_select = arma::find(eta_select == 1);
    }
    else
    {
        idx_select.set_size(1);
        idx_select.fill(0);
    }

    eta.set_size(idx_select.n_elem);
    eta.zeros();
    for (unsigned int i = 0; i < idx_select.n_elem; i++)
    {
        unsigned int idx = idx_select.at(i);
        switch (idx)
        {
        case 0: // W is selected
        {
            eta.at(i) = W_par.at(0);
        }
        break;
        case 1: // mu0 is selected
        {
            eta.at(i) = obs_par.at(0);
        }
        break;
        case 2: // par 1 is selected
        {
            eta.at(i) = lag_par.at(0);
        }
        break;
        case 3: // par 2 is selected
        {
            eta.at(i) = lag_par.at(1);
        }
        break;
        default:
        {
            throw std::invalid_argument("HVB: number of unknown static parameters out of bound.");
        }
        }
    }

    return;
}


inline void update_params(
    arma::vec &obs_par,
    arma::vec &lag_par,
    arma::vec &W_par,
    const arma::uvec &idx_select,
    const arma::vec &eta
)
{
    for (unsigned int i = 0; i < idx_select.n_elem; i++)
    {
        unsigned int idx = idx_select.at(i);
        double val = eta.at(i);
        switch (idx)
        {
        case 0: // W is selected
        {
            W_par.at(0) = val;
            bound_check(val, "update_params: W", true, true);
        }
        break;
        case 1: // mu0 is selected
        {
            obs_par.at(0) = val;
            bound_check(val, "update_params: mu0", false, true);
        }
        break;
        case 2: // par 1 is selected
        {
            lag_par.at(0) = val;
            bound_check(val, "update_params: par1", true, true);
        }
        break;
        case 3: // par 2 is selected
        {
            lag_par.at(1) = val;
            bound_check(val, "update_params: par2", true, true);
        }
        break;
        default:
        {
            throw std::invalid_argument("HVB: number of unknown static parameters out of bound.");
        }
        }
    }

    return;
}


