#include "pl_poisson.h"
#include "yjtrans.h"
using namespace Rcpp;
// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::depends(RcppArmadillo)]]


/*
Transform eta = (W,mu0,rho,M) to eta_tilde on the real space R^4
*/
void eta2tilde(
    arma::vec& eta_tilde, // m x 1
    const arma::uvec &eta_prior_type,
    const arma::vec& eta, // 4 x 1
    const arma::uvec& idx_select, // m x 1 (m<=4)
    const unsigned int m){ 
    
    for (unsigned int i=0; i<m; i++) {
        double etmp = eta.at(idx_select.at(i));
        switch(idx_select.at(i)) {
            case 0: // W   - Wtilde = log(W)
            {
                unsigned int Wprior = eta_prior_type.at(0);
                switch (Wprior)
                {
                case 0:
                    eta_tilde.at(i) = std::log(etmp);
                    break;
                case 1:
                    throw std::invalid_argument("Jacobian of Cauchy W is undefined.");
                    break;
                case 2:
                    eta_tilde.at(i) = -std::log(etmp);
                    break;
                default:
                    break;
                }
            }
            break;
            case 1: // mu0 - mu0_tilde = log(mu[0])
            {
                eta_tilde.at(i) = std::log(etmp);
            }
            break;
            case 2: // rho - rho_tilde = log(rho/(1.-rho))
            {
                eta_tilde.at(i) = std::log(etmp) - std::log(1.-etmp);
            }
            break;
            case 3: // M - undefined
            {
                eta_tilde.at(i) = etmp;
            }
            break;
            default:
            {
                 throw std::invalid_argument("eta2tilde - idx_select out of bound.");
            }
        }
    }

    if (!eta_tilde.is_finite())
    {
        throw std::invalid_argument("eta2tilde<void>: non-finite conversion.");
    }
}

void tilde2eta(
    arma::vec &eta, // 4 x 1
    const arma::uvec &eta_prior_type,
    const arma::vec &eta_tilde,   // m x 1
    const arma::uvec &idx_select, // m x 1
    const unsigned int m)
{

    for (unsigned int i=0; i<m; i++) {
        unsigned int itmp = idx_select.at(i);
        double etmp = eta_tilde.at(i);
        switch(itmp) {
            case 0: // W = exp(Wtilde)
            {
                unsigned int Wprior = eta_prior_type.at(0);
                switch (Wprior)
                {
                case 0:
                    etmp = std::min(etmp, UPBND);
                    eta.at(itmp) = std::exp(etmp);
                    break;
                case 1:
                    throw std::invalid_argument("Jacobian of Cauchy W is undefined.");
                    break;
                case 2:
                    etmp *= -1.;
                    etmp = std::min(etmp,UPBND);
                    eta.at(itmp) = std::exp(etmp);
                    break;
                default:
                    break;
                }
            }
            break;
            case 1: // mu0 = exp(mu0_tilde)
            {
                eta.at(itmp) = std::exp(etmp);
            }
            break;
            case 2: // rho = exp(rho_tilde) / (1 + exp(rho_tilde))
            {
                double tmp = std::exp(etmp);
                eta.at(itmp) = tmp;
                eta.at(itmp) /= (1. + tmp);
            }
            break;
            case 3: // M - undefined
            {
                eta.at(itmp) = etmp;
            }
            break;
            default:
            {
                 throw std::invalid_argument("tilde2eta - idx_select out of bound.");
            }
        }
    }

    if (!eta.is_finite())
    {
        throw std::invalid_argument("tilde2eta<void>: non-finite conversion.");
    }
}

/*
------ dlogJoint_dWtilde ------
The derivative of the full joint density with respect to Wtilde=log(1/W), where W
is the evolution variance that affects
    psi[t] | psi[t-1] ~ N(psi[t-1], W)

Therefore, this function remains unchanged as long as we using the same evolution equation for psi.
*/
double dlogJoint_dWtilde(
    const arma::vec& psi, // (n+1) x 1, (psi[0],psi[1],...,psi[n])
    const double G, // evolution transition matrix
    const double W, // evolution variance conditional on V
    const double aw,
    const double bw,
    const unsigned int prior_type = 0) { // 0 - Gamma(aw=shape,bw=rate); 1 - Half-Cauchy(aw=location=0, bw=scale); 2 - Inverse-Gamma(aw=shape,bw=rate)

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

    double rdw = std::exp(std::log(res+EPS) - std::log(W+EPS)); // sum(w[t]^2) / W
    bound_check(rdw,"dlogJoint_dWtilde: rdw");

    double deriv;
    if (prior_type==0) {
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
    } else if (prior_type==1) {
        /*
        sqrt(W) ~ Half-Cauchy(aw=location==0, bw=scale)
        */
        deriv = 0.5*n-0.5 * rdw + W/(bw*bw+W) - 0.5;
    } else if (prior_type==2) {
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
        deriv = - std::exp(log_bnew_W);

        double a_new = aw;
        a_new += 0.5*n;
        a_new += 1.;
        deriv += a_new;
    } else {
         throw std::invalid_argument("Unsupported prior for evolution variance W.");
    }

    bound_check(deriv, "dlogJoint_dWtilde: deriv");
    return deriv;
}

double logprior_Wtilde(
    const double Wtilde, // evolution variance conditional on V
    const double aw,
    const double bw,
    const unsigned int prior_type = 0)
{
    double InvGamma_cnst = aw;
    InvGamma_cnst *= std::log(bw);
    InvGamma_cnst -= std::lgamma(aw);

    double logp = -16.;
    if (prior_type==0) {
        /*
        W ~ Gamma(aw=shape, bw=rate)
        Wtilde = log(W)
        */
        logp = aw*std::log(bw);
        logp -= std::lgamma(aw);
        logp += aw*Wtilde;

        double Wast = std::min(Wtilde,UPBND);
        logp -= bw * std::exp(Wast);
    } else if (prior_type==1) {
        /*
        sqrt(W) ~ Half-Cauchy(aw=location==0, bw=scale)
        */
        throw std::invalid_argument("logprior_Wtilde for Half-Cauchy is not implemented yet.");
    } else if (prior_type==2) {
        /*
        W ~ Inverse-Gamma(aw=shape, bw=rate)
        Wtilde = -log(W)
        */
        logp = InvGamma_cnst;
        logp += aw*Wtilde;
        double Wast = std::min(Wtilde, UPBND);
        logp -= bw * std::exp(Wast);
    } else {
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
double dlogJoint_dlogmu0(
    const arma::vec& ypad, // (n+1) x 1
    const arma::vec& theta, // (n+1) x 1, (theta[0],theta[1],...,theta[n])
    const double mu0, // a realization of mu[0]
    const double amu, // shape parameter of the Gamma prior for mu[0]
    const double bmu, // rate parameter of the Gamma prior for mu[0]
    const double delta_nb = 1.,
    const unsigned int obs_code = 0) { // 0 - negative binomial; 1 - poisson
    /*
    Wtilde = -log(W)
    */
    
    const unsigned int n = ypad.n_elem - 1;
    double res = 0.;
    double lambda;
    for (unsigned int t=1; t<=n; t++) {
        lambda = mu0 + theta.at(t);
        res += ypad.at(t) / lambda;
        if (obs_code == 0) { // negative binomial
            res -= (ypad.at(t)+delta_nb)/(lambda+delta_nb);
        } else if (obs_code == 1) {
            res -= 1.;
        } else {
             throw std::invalid_argument("Unknown likelihood.");
        }
    }

    double deriv = mu0*res + amu - bmu*mu0;
    bound_check(deriv, "dlogJoint_dlogmu0: deriv");
    return deriv;
}


double logprior_logmu0(
    const double logmu0, // evolution variance conditional on V
    const double amu,
    const double bmu) {

    /*
    mu0 ~ Gamma(aw=shape, bw=rate)
    logmu0 = log(mu0)
    */
    double logp = amu*std::log(bmu) - std::lgamma(amu) + amu*logmu0 - bmu*std::exp(logmu0);
    bound_check(logp,"logprior_logmu0: logp");
    return logp;
}


double dlogJoint_drho(
    const arma::vec& ypad, // (n+1) x 1
    const arma::mat& R, // (n+1) x 2, (psi,theta)
    const unsigned int L,
    const double mu0,
    const double rho,
    const double delta_nb,
    const unsigned int gain_code,
    const double arho = 1.,
    const double brho = 1.) {

    const unsigned int n = ypad.n_elem - 1;
    const double L_ = static_cast<double>(L);

    arma::vec hpsi = psi2hpsi(R.col(0),gain_code); // (n+1) x 1
    arma::vec lambda = mu0 + R.col(1); // (n+1) x 1

    unsigned int r;
    double c1 = 0.;
    double c2 = 0.;
    double c20 = 0.;
    double c21 = 0.;
    double c10 = 0.;

    for (unsigned int t=1; t<=n; t++) {
        c10 = ypad.at(t)/lambda.at(t) - (ypad.at(t)+delta_nb)/(lambda.at(t)+delta_nb);
        c1 += c10*hpsi.at(t-1)*ypad.at(t-1);

        r = std::min(t,L);
        c20 = 0.;
        c21 = -rho;
        for (unsigned int k=1; k<=r; k++) {
            c20 += static_cast<double>(k)*binom(L,k)*c21*R.at(t-k,1);
            c21 *= -rho;
        }
        c2 += c10*c20;
    }

    double deriv = -L_*std::pow(1.-rho,L_)*rho*c1 - (1.-rho)*c2;
    deriv += arho - (arho+brho) * rho;
    bound_check(deriv,"dlogJoint_drho: deriv");
    return deriv;
}



double dlogJoint_drho2(
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


double logprior_logitrho(
    double logitrho,
    double arho = 1.,
    double brho = 1.) {
    
    double logp = std::lgamma(arho+brho) - std::lgamma(arho) - std::lgamma(brho) + arho*logitrho - (arho+brho)*std::log(1.+std::exp(logitrho));
    bound_check(logp,"logprior_logitrho: logp");
    return logp;
}


/*
logprior_eta_tilde
------------------------
`eta_prior_val`
    aw  amu   arho    DK 
    bw  bmu   brho    DK
*/
arma::vec logprior_eta_tilde(
    const arma::vec& eta_tilde, // m x 1
    const arma::uvec& idx_select, // m x 1
    const arma::uvec& eta_prior_type, // 4 x 1
    const arma::mat& eta_prior_val, // 2 x 4, priors for each element of eta
    const unsigned int m) {
    
    arma::vec logp(m);

    for (unsigned int i=0; i<m; i++) {
        switch(idx_select.at(i)) {
            case 0: // W   - Wtilde = -log(W)
            {
                logp.at(i) = logprior_Wtilde(eta_tilde.at(i),eta_prior_val.at(0,0),eta_prior_val.at(1,0),eta_prior_type.at(0));
            }
            break;
            case 1: // mu0 - mu0_tilde = log(mu[0])
            {
                logp.at(i) = logprior_logmu0(eta_tilde.at(i),eta_prior_val.at(0,1),eta_prior_val.at(1,1));
            }
            break;
            case 2: // rho - rho_tilde = log(rho/(1.-rho))
            {
                logp.at(i) = logprior_logitrho(eta_tilde.at(i),eta_prior_val.at(0,2),eta_prior_val.at(1,2));
            }
            break;
            case 3: // M - undefined
            {
                 throw std::invalid_argument("logprior_eta_tilde - M not implemented yet.");
            }
            break;
            default:
            {
                 throw std::invalid_argument("logprior_eta_tilde - idx_select out of bound.");
            }
        }
    }

    return logp;
}


// eta_tilde = (Wtilde,logmu0)
arma::vec dlogJoint_deta(
    const arma::vec& ypad, // (n+1) x 1
    const arma::mat& R, // (n+1) x 2, // (psi,theta)
    const arma::vec& eta, // 4 x 1
    const arma::uvec& idx_select, // m x 1
    const arma::uvec& eta_prior_type, // 4 x 1
    const arma::mat& eta_prior_val, // 2 x 4
    const unsigned int m,
    const unsigned int L,
    const arma::vec& rcomb,
    const double delta_nb = 1.,
    const unsigned int obs_code = 0, // 0 - NegBinom; 1 - Poisson
    const unsigned int gain_code = 0) {
    arma::vec deriv(m);
    for (unsigned int i=0; i<m; i++) {
        switch(idx_select.at(i)) {
            case 0: // Wtilde = -log(W)
            {
                deriv.at(i) = dlogJoint_dWtilde(R.col(0),1.,eta.at(0),eta_prior_val.at(0,0),eta_prior_val.at(1,0),eta_prior_type.at(0));
            }
            break;
            case 1: // logmu0 = log(mu0)
            {
                deriv.at(i) = dlogJoint_dlogmu0(ypad,R.col(1),eta.at(1),eta_prior_val.at(0,1),eta_prior_val.at(1,1),delta_nb,obs_code);
            }
            break;
            case 2: // rho_tilde = log(rho/(1-rho))
            {
                deriv.at(i) = dlogJoint_drho2(ypad,R,L,eta.at(1),eta.at(2),delta_nb,gain_code,rcomb,eta_prior_val.at(0,2),eta_prior_val.at(1,2));
            }
            break;
            case 3: // M - undefined
            {
                 throw std::invalid_argument("dlogJoint_deta - Derivative wrt M undefined.");
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




void update_mu(
    arma::vec &mu,
    arma::vec &curEg2_mu, 
    arma::vec &curEdelta2_mu,
    const arma::vec &L_mu,
    const double &learn_rate,
    const double &eps_step)
{
    bound_check(mu,"update_mu: input mu");

    arma::vec oldEg2_mu = curEg2_mu;
    arma::vec oldEdelta2_mu = curEdelta2_mu;

    curEg2_mu = learn_rate * oldEg2_mu + (1. - learn_rate) * arma::pow(L_mu, 2.); // m x 1
    arma::vec Change_delta_mu = arma::sqrt(oldEdelta2_mu + eps_step) / arma::sqrt(curEg2_mu + eps_step) % L_mu;
    bound_check(Change_delta_mu, "update_mu: Change_delta_mu");

    mu = mu + Change_delta_mu;
    curEdelta2_mu = learn_rate * oldEdelta2_mu + (1. - learn_rate) * arma::pow(Change_delta_mu, 2.);
    bound_check(mu,"update_mu: mu");
    bound_check(curEdelta2_mu, "update_mu: curEdelta2_mu");
}

/*
eta = (W,mu[0],rho,M)

`eta_select` 
    Indicates if a global parameter should be inferred by HVA.
    For example, eta_select = (true,true,false,false) means that W and mu[0] will be inferred and we should provide the true values for rho and M. 
`eta_init`
    True (eta_select set to false) or initial (eta_select set to true) values should be provided here.
`eta_prior_type`
    - W: 0 - Gamma(aw,bw); 1 - Half-Cauchy(aw=0,bw); 2 - Inverse-Gamma(aw,bw)
    - mu[0]: 0 - Gamma(amu,bmu)
    - rho: 
    - M
`eta_prio_val`
    aw  amu   arho    DK 
    bw  bmu   brho    DK
*/
//' @export
// [[Rcpp::export]]
Rcpp::List hva_poisson(
    const arma::vec& Y, // n x 1
    const arma::uvec& model_code,
    const arma::uvec& eta_select, // 4 x 1, indicator for unknown (=1) or known (=0)
    const arma::vec& eta_init, // 4 x 1, if true/initial values should be provided here
    const arma::uvec& eta_prior_type, // 4 x 1
    const arma::mat& eta_prior_val, // 2 x 4, priors for each element of eta
    const double L_order = 2,
    const unsigned int nlag_ = 0,
    const Rcpp::Nullable<Rcpp::NumericVector>& m0_prior = R_NilValue,
    const Rcpp::Nullable<Rcpp::NumericMatrix>& C0_prior = R_NilValue,
    const double theta0_upbnd = 2.,
    const Rcpp::Nullable<Rcpp::NumericVector>& psi_init = R_NilValue, // previously `ht_`
    const double Blag_pct = 0.1,
    const double scale_sd = 1.e-16,
    const double learn_rate = 0.95,
    const double eps_step = 1.e-6,
    const unsigned int k = 1, // k <= sum(eta_select)
    const unsigned int nsample = 100,
    const unsigned int nburnin = 100,
    const unsigned int nthin = 2,
    const double delta_nb = 1.,
    const double delta_discount = 0.95,
    const bool summarize_return = false) {


    const unsigned int ntotal = nburnin + nthin*nsample + 1;
    const unsigned int n = Y.n_elem;
    const unsigned int npad = n + 1;
    arma::vec ypad(n+1,arma::fill::zeros); ypad.tail(n) = Y;

    bool update_static = arma::accu(eta_select) > 0;

    const unsigned int obs_code = model_code.at(0);
    const unsigned int link_code = model_code.at(1);
    const unsigned int trans_code = model_code.at(2);
    const unsigned int gain_code = model_code.at(3);
    const unsigned int err_code = model_code.at(4);


    /* ------ Define Local Parameters ------ */
    arma::mat R(n+1,2,arma::fill::zeros); // (psi,theta)
    /* ------ Define Local Parameters ------ */
    
    /* ------ Define Global Parameters ------ */
    arma::vec eta = eta_init; // eta = (W>0,mu0>0,rho in (0,1),M>0)
    arma::vec eta_old = eta;
    double W = eta.at(0);
    double mu0 = eta.at(1);
    double rho = eta.at(2);

    const unsigned int m = std::max(arma::as_scalar<double>(arma::accu(eta_select)), 1.);

    if (k > m) {
        throw std::invalid_argument("k cannot be greater than m, total number of unknowns.");
    }
    /* ------ Define Global Parameters ------ */


    /* ------ Define SMC ------ */
    // const unsigned int Blag = L;
    unsigned int nlag, p, L;
    set_dim(nlag, p, L, n, nlag_, L_order);
    arma::vec Fphi = get_Fphi(nlag, L, rho, trans_code);
    arma::vec Ft = init_Ft(p, trans_code);

    unsigned int Blag;
    if (nlag < n)
    {
        // truncated
        Blag = p;
    }
    else
    {
        Blag = static_cast<unsigned int>(Blag_pct * n); // B-fixed-lags Monte Carlo smoother
    }
    const unsigned int N = 100; // number of particles for SMC
    /* ------ Define SMC ------ */


    /* ------ Define Model ------ */

    arma::vec rcomb(n,arma::fill::zeros);
    for (unsigned int l=1; l<=n; l++) {
        rcomb.at(l-1) = binom(L - 2 + l,l - 1);
    }
    /* ------ Define Model ------ */


    /*  ------ Define LBE ------ */
    arma::mat mt(p,n+1,arma::fill::zeros);
    arma::cube Ct(p,p,n+1,arma::fill::zeros);
    arma::vec m0(p,arma::fill::zeros);
    arma::mat C0(p,p,arma::fill::eye);

    if (!m0_prior.isNull()) {
        m0 = Rcpp::as<arma::vec>(m0_prior);
        mt.col(0) = m0;
    }
	if (!C0_prior.isNull()) {
		C0 = Rcpp::as<arma::mat>(C0_prior);
        Ct.slice(0) = C0;
    } else {
        C0 *= std::pow(theta0_upbnd * 0.5, 2.);
    }

    arma::mat at(p,n+1,arma::fill::zeros);
    arma::cube Rt(p,p,n+1,arma::fill::zeros);
    arma::vec alphat(n+1,arma::fill::ones);
    arma::vec betat(n+1,arma::fill::ones);
    arma::vec Ht(n+1,arma::fill::zeros);

    arma::cube Gt(p,p,npad);
    init_Gt(Gt,rho,p,nlag,n);
    /*  ------ Define LBE ------ */

    /*  ------ Define HVB ------ */
    arma::uvec idx_select; // m x 1
    if (update_static)
    {
        idx_select = arma::find(eta_select == 1);
    }
    else
    {
        idx_select.set_size(m);
        idx_select.fill(1)
;
    }
    arma::vec eta_tilde(m, arma::fill::zeros);           // unknown part of eta which is also transformed to the real line
    arma::vec eta_tilde0(m, arma::fill::zeros);
    eta2tilde(eta_tilde, eta_prior_type, eta, idx_select, m);
    arma::vec gamma(m, arma::fill::ones);
    arma::vec nu = tYJ(eta_tilde, gamma); // m x 1, Yeo-Johnson transform of eta_tilde

    arma::vec tau = gamma2tau(gamma);
    arma::vec mu(m, arma::fill::zeros);
    arma::mat B(m, k, arma::fill::zeros); // Lower triangular part is nonzero
    arma::vec d(m, arma::fill::ones);
    arma::mat SigInv(m, m, arma::fill::zeros);
    double logq;

    arma::vec delta_logJoint(m);
    arma::vec delta_logq(m);
    arma::vec delta_diff(m);
    arma::mat dtheta_dB(m, m * k, arma::fill::zeros);

    arma::vec xi(k, arma::fill::zeros);
    arma::vec xi_old = xi;
    arma::vec eps(m, arma::fill::zeros);
    arma::vec eps_old = eps;

    arma::vec L_mu(m);
    arma::mat L_B(m, k, arma::fill::zeros);
    arma::vec vecL_B(m * k, arma::fill::zeros);
    arma::vec L_d(m);
    arma::vec L_tau(m);

    arma::vec oldEg2_mu(m, arma::fill::zeros);
    arma::vec curEg2_mu(m, arma::fill::zeros);
    arma::vec oldEdelta2_mu(m, arma::fill::zeros);
    arma::vec curEdelta2_mu(m, arma::fill::zeros);
    arma::vec Change_delta_mu(m, arma::fill::zeros);

    arma::vec oldEg2_B(m * k, arma::fill::zeros);
    arma::vec curEg2_B(m * k, arma::fill::zeros);
    arma::vec oldEdelta2_B(m * k, arma::fill::zeros);
    arma::vec curEdelta2_B(m * k, arma::fill::zeros);
    arma::vec Change_delta_B(m * k, arma::fill::zeros);

    arma::vec oldEg2_d(m, arma::fill::zeros);
    arma::vec curEg2_d(m, arma::fill::zeros);
    arma::vec oldEdelta2_d(m, arma::fill::zeros);
    arma::vec curEdelta2_d(m, arma::fill::zeros);
    arma::vec Change_delta_d(m, arma::fill::zeros);

    arma::vec oldEg2_tau(m, arma::fill::zeros);
    arma::vec curEg2_tau(m, arma::fill::zeros);
    arma::vec oldEdelta2_tau(m, arma::fill::zeros);
    arma::vec curEdelta2_tau(m, arma::fill::zeros);
    arma::vec Change_delta_tau(m, arma::fill::zeros);

    /*  ------ Define HVB ------ */


    // arma::mat mu_stored(m,niter);
    // arma::mat d_stored(m,niter);
    // arma::mat gamma_stored(m,niter);
    arma::mat psi_stored(n+1,nsample);
    arma::vec W_stored(nsample);
    arma::vec mu0_stored(nsample);
    arma::vec rho_stored(nsample);

    arma::mat Meff(n,nsample);
    arma::mat resample_status(n,nsample);
    // arma::mat logp_stored(max_iter,nsample);
    // arma::vec logp_iter(max_iter,arma::fill::zeros);

    arma::vec logq_stored(nsample);
    double mh_accept = 0.;
    // arma::mat theta_last(p,nsample);
    // arma::vec theta(p);

    bool saveiter;
    

    for (unsigned int s=0; s<ntotal; s++) {
        R_CheckUserInterrupt();
		saveiter = s > nburnin && ((s-nburnin-1)%nthin==0);

        /*
        Step 2. Sample state parameters via posterior
            - psi: (n+1) x 1 gain factor
            - eta = (W,mu0,rho,M)
        */
        if (eta_select.at(0) == 1) {
            W = eta.at(0);
        }
        if (eta_select.at(1) == 1) {
            mu0 = eta.at(1);
        }
        if (eta_select.at(2) == 1) {
            rho = eta.at(2);
        }

        arma::vec pmarg_y(n, arma::fill::zeros);
        if (eta_select.at(0)==0 || s==0)
        {
            mcs_poisson(
                R,pmarg_y,NA_REAL,ypad,model_code,
                rho,L,nlag,mu0,
                Blag,N,
                m0,C0,
                theta0_upbnd,
                delta_nb,
                delta_discount);
        }
        else 
        {
            mcs_poisson(
                R,pmarg_y,W,ypad,model_code,
                rho,L,nlag,mu0,
                Blag,N,
                m0,C0,
                theta0_upbnd,
                delta_nb,
                delta_discount);

        }

        arma::vec hpsi_tmp = psi2hpsi(R.col(0), gain_code); // (n+1) x 1
        arma::vec theta_tmp(n,arma::fill::zeros);
        hpsi2theta(theta_tmp, hpsi_tmp, ypad, trans_code, L, nlag, rho); // theta
        arma::vec theta_tmp2(n+1,arma::fill::zeros);
        theta_tmp2.tail(n) = theta_tmp;
        R.col(1) = theta_tmp2;


        if (update_static) {
            // Logarithm marginal likelihood of y
            // Dynamic/latent states are integrated out
            // only conditional on static parameters, eta.

            /*
            Step 3. Compute gradient of the log variational distribution (Model Dependent)
            */
            // if (is_vanilla) {
            //     delta_logJoint = dlogJoint_deta(Y,psi,eta.at(2),eta.at(0),aw,bw,prior_type); // 1 x 1
            // } else {
            delta_logJoint = dlogJoint_deta(ypad, R, eta, idx_select, eta_prior_type, eta_prior_val, m, L, rcomb, delta_nb, obs_code, gain_code); // m x 1
            // }

            SigInv = get_sigma_inv(B, d, k);                             // m x m
            delta_logq = dlogq_dtheta(SigInv, nu, eta_tilde, gamma, mu); // m x 1
            delta_diff = delta_logJoint - delta_logq;                    // m x 1

            /*
            Finished gradient calculation

            delta_logJoint: the derivative of (log joint probability) with respect to theta
            dlogq_dtheta: the derivative of (log proposal probability) with respect to theta
            */

            // TODO: transpose or no transpose
            L_mu = dYJinv_dnu(nu, gamma) * delta_diff; // m x 1


            if (m > 1)
            {
                dtheta_dB = dYJinv_dB(nu, gamma, xi);                  // m x mk
                L_B = arma::reshape(dtheta_dB.t() * delta_diff, m, k); // m x k
                L_B.elem(arma::trimatu_ind(arma::size(L_B), 1)).zeros();
                vecL_B = arma::vectorise(L_B);
            }

            L_d = dYJinv_dD(nu, gamma, eps) * delta_diff; // m x 1
            L_tau = dYJinv_dtau(nu, gamma) * delta_diff;


            /*
            Step 4. Update Variational Parameters
            */
            eta_old = eta;

            // mu
            oldEg2_mu = curEg2_mu;
            oldEdelta2_mu = curEdelta2_mu;

            curEg2_mu = learn_rate * oldEg2_mu + (1. - learn_rate) * arma::pow(L_mu, 2.); // m x 1
            Change_delta_mu = arma::sqrt(oldEdelta2_mu + eps_step) / arma::sqrt(curEg2_mu + eps_step) % L_mu;
            mu = mu + Change_delta_mu;
            curEdelta2_mu = learn_rate * oldEdelta2_mu + (1. - learn_rate) * arma::pow(Change_delta_mu, 2.);
            // update_mu(mu,curEg2_mu,curEdelta2_mu,L_mu,learn_rate,eps_step);

            // B
            if (m > 1)
            {
                oldEg2_B = curEg2_B;         // mk x 1
                oldEdelta2_B = curEdelta2_B; // mk x 1

                curEg2_B = learn_rate * oldEg2_B + (1. - learn_rate) * arma::pow(vecL_B, 2.);
                Change_delta_B = arma::sqrt(oldEdelta2_B + eps_step) / arma::sqrt(curEg2_B + eps_step) % vecL_B; // mk x 1

                B = B + arma::reshape(Change_delta_B, m, k);
                curEdelta2_B = learn_rate * oldEdelta2_B + (1. - learn_rate) * arma::pow(Change_delta_B, 2.);
            }

            // d
            oldEg2_d = curEg2_d;
            oldEdelta2_d = curEdelta2_d;

            curEg2_d = learn_rate * oldEg2_d + (1. - learn_rate) * arma::pow(L_d, 2.); // m x 1
            Change_delta_d = arma::sqrt(oldEdelta2_d + eps_step) / arma::sqrt(curEg2_d + eps_step) % L_d;
            d = d + Change_delta_d;
            curEdelta2_d = learn_rate * oldEdelta2_d + (1. - learn_rate) * arma::pow(Change_delta_d, 2.);

            SigInv = get_sigma_inv(B, d, k);

            // tau
            oldEg2_tau = curEg2_tau;
            oldEdelta2_tau = curEdelta2_tau;

            curEg2_tau = learn_rate * oldEg2_tau + (1. - learn_rate) * arma::pow(L_tau, 2.); // m x 1
            Change_delta_tau = arma::sqrt(oldEdelta2_tau + eps_step) / arma::sqrt(curEg2_tau + eps_step) % L_tau;
            tau = tau + Change_delta_tau;
            curEdelta2_tau = learn_rate * oldEdelta2_tau + (1. - learn_rate) * arma::pow(Change_delta_tau, 2.);
            gamma = tau2gamma(tau);



            eta_tilde = rtheta(xi, eps, gamma, mu, B, d);

            // xi_old = xi;
            // eps_old = eps;
            nu = tYJ(eta_tilde, gamma); // recover nu
            tilde2eta(eta, eta_prior_type, eta_tilde, idx_select, m);
            logq = logq0(nu, eta_tilde, gamma, mu, SigInv, m);
            // logp_iter.at(0) = logp_new;
            // for (unsigned int s = 1; s < max_iter; s++)
            // {
            //     eta_tilde0 = rtheta(xi, eps, gamma, mu, B, d);
            //     logprior = logprior_eta_tilde(eta_tilde0, idx_select, eta_prior_type, eta_prior_val, m);
            //     nu = tYJ(eta_tilde0, gamma); // recover nu
            //     tilde2eta(eta, eta_tilde0, idx_select, m);
            //     logq = logq0(nu, eta_tilde0, gamma, mu, SigInv, m);
            //     logp_new = arma::accu(logprior) + logp_y - logq;
            //     logp_iter.at(s) = logp_new;

            //     if (std::log(R::runif(0., 1.)) < std::min(0., logp_iter.at(s) - logp_iter.at(s - 1)))
            //     {
            //         // accept
            //         eta_tilde = eta_tilde0;
            //         xi_old = xi;
            //         eps_old = eps;
            //     }
            // }

            // xi = xi_old;
            // eps = eps_old;


            // nu = tYJ(eta_tilde, gamma); // recover nu
            // tilde2eta(eta, eta_tilde, idx_select, m);

            double Wtmp = eta.at(0);
            bound_check(Wtmp, "hva_poisson: Wtmp at variational step",true,true);
        }



        if (saveiter || s==(ntotal-1)) {
			unsigned int idx_run;
			if (saveiter) {
				idx_run = (s-nburnin-1)/nthin;
			} else {
				idx_run = nsample - 1;
			}

			// mu_stored.col(s) = mu;
            // d_stored.col(s) = d;
            // gamma_stored.col(s) = gamma;
            psi_stored.col(idx_run) = R.col(0);
            W_stored.at(idx_run) = eta.at(0);
            mu0_stored.at(idx_run) = eta.at(1);
            rho_stored.at(idx_run) = eta.at(2);
            // logp_stored.col(idx_run) = logp_iter;
            logq_stored.at(idx_run) = logq;
            // theta_last.col(idx_run) = theta;
		}

		Rcout << "\rProgress: " << s << "/" << ntotal-1;
    }

	Rcout << std::endl;

    Rcpp::List output;

    if (summarize_return) {
        arma::vec qProb = {0.025,0.5,0.975};
        arma::mat RR = arma::quantile(psi_stored,qProb,1); 
        output["psi"] = Rcpp::wrap(RR); // (n+1) x 3

        // arma::mat hpsiR = psi2hpsi(RR, gain_code); // hpsi: p x N
        // double theta0 = 0;
        // arma::mat thetaR = hpsi2theta(hpsiR, Y, trans_code, theta0, L, rho); // n x 1

        // output["hpsi"] = Rcpp::wrap(hpsiR);
        // output["theta"] = Rcpp::wrap(thetaR);

    } else {
        output["psi"] = Rcpp::wrap(psi_stored); // (n+1) x niter
    }

    if (eta_select.at(0)==1) {
        output["W"] = Rcpp::wrap(W_stored); // niter
    }
    if (eta_select.at(1)==1) {
        output["mu0"] = Rcpp::wrap(mu0_stored); // niter
    }
    if (eta_select.at(2)==1) {
        output["rho"] = Rcpp::wrap(rho_stored); // niter
    }

    output["psi_all"] = Rcpp::wrap(psi_stored);

    output["logq"] = Rcpp::wrap(logq_stored);
    // output["logp_diff"] = Rcpp::wrap(logp_stored);
    
    
    
    // output["theta_last"] = Rcpp::wrap(theta_last); dtYJ_dtheta
    // p x niter

    // if (debug) {
    //     output["Meff"] = Rcpp::wrap(Meff);
    //     output["resample_status"] = Rcpp::wrap(resample_status);
    // }
    

    return output;
}

