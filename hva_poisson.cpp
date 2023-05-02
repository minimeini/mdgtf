#include "pl_poisson.h"
#include "yjtrans.h"
using namespace Rcpp;
// [[Rcpp::plugins(cpp17)]]
// [[Rcpp::depends(RcppArmadillo)]]


/*
Transform eta = (W,mu0,rho,M) to eta_tilde on the real space R^4
*/
void eta2tilde(
    arma::vec& eta_tilde, // m x 1
    const arma::vec& eta, // 4 x 1
    const arma::uvec& idx_select, // m x 1 (m<=4)
    const unsigned int m){ 
    
    for (unsigned int i=0; i<m; i++) {
        switch(idx_select.at(i)) {
            case 0: // W   - Wtilde = -log(W)
            {
                eta_tilde.at(i) = -std::log(eta.at(idx_select.at(i)));
            }
            break;
            case 1: // mu0 - mu0_tilde = log(mu[0])
            {
                eta_tilde.at(i) = std::log(eta.at(idx_select.at(i)));
            }
            break;
            case 2: // rho - rho_tilde = log(rho/(1.-rho))
            {
                eta_tilde.at(i) = std::log(eta.at(idx_select.at(i))) - std::log(1.-eta.at(idx_select.at(i)));
            }
            break;
            case 3: // M - undefined
            {
                eta_tilde.at(i) = eta.at(idx_select.at(i));
            }
            break;
            default:
            {
                ::Rf_error("eta2tilde - idx_select out of bound.");
            }
        }
    }
}



void tilde2eta(
    arma::vec& eta, // 4 x 1
    const arma::vec& eta_tilde, // m x 1
    const arma::uvec& idx_select, // m x 1
    const unsigned int m) {
    
    for (unsigned int i=0; i<m; i++) {
        switch(idx_select.at(i)) {
            case 0: // W = exp(-Wtilde)
            {
                eta.at(idx_select.at(i)) = std::exp(-eta_tilde.at(i));
            }
            break;
            case 1: // mu0 = exp(mu0_tilde)
            {
                eta.at(idx_select.at(i)) = std::exp(eta_tilde.at(i));
            }
            break;
            case 2: // rho = exp(rho_tilde) / (1 + exp(rho_tilde))
            {
                eta.at(idx_select.at(i)) = std::exp(eta_tilde.at(i)) / (1. + std::exp(eta_tilde.at(i)));
            }
            break;
            case 3: // M - undefined
            {
                eta.at(idx_select.at(i)) = eta_tilde.at(i);
            }
            break;
            default:
            {
                ::Rf_error("tilde2eta - idx_select out of bound.");
            }
        }
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
    /*
    Wtilde = -log(W)
    */
    
    const double n = static_cast<double>(psi.n_elem) - 1;
    const unsigned int ni = psi.n_elem - 1;
    double res = 0.;
    for (unsigned int t=0; t<ni; t++) {
        res += (psi.at(t+1)-G*psi.at(t)) * (psi.at(t+1)-G*psi.at(t));
    }

    double deriv;
    if (prior_type==0) {
        /*
        W ~ Gamma(aw=shape, bw=rate)
        */
        deriv = 0.5*n-0.5/W*res - aw + bw*W;
    } else if (prior_type==1) {
        /*
        sqrt(W) ~ Half-Cauchy(aw=location==0, bw=scale)
        */
        deriv = 0.5*n-0.5/W*res + W/(bw*bw+W) - 0.5;
    } else if (prior_type==2) {
        /*
        W ~ Inverse-Gamma(aw=shape, bw=rate)
        (deprecated)
        */
        deriv = res;
        deriv *= 0.5;
        deriv += bw;
        deriv = -std::exp(std::log(res) - std::log(W));
        deriv += aw + 0.5*n;
    } else {
        ::Rf_error("Unsupported prior for evolution variance W.");
    }
    return deriv;
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
            ::Rf_error("Unknown likelihood.");
        }
    }

    double deriv = mu0*res + amu - 1. - bmu*mu0;
    return deriv;
}




double dlogJoint_drho(
    const arma::vec& ypad, // (n+1) x 1
    const arma::mat& R, // (n+1) x 2, (psi,theta)
    const unsigned int L,
    const double mu0,
    const double rho,
    const double delta_nb,
    const unsigned int gain_code,
	const Rcpp::NumericVector& coef = Rcpp::NumericVector::create(0.2,0,5.)) {

    const unsigned int n = ypad.n_elem - 1;
    const double L_ = static_cast<double>(L);

    arma::vec hpsi = psi2hpsi(R.col(0),gain_code,coef); // (n+1) x 1
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

    double deriv = -L_*std::pow(1.-rho,L_)*rho*c1 - (1.-rho)*c2 + 1.-2*rho;
    return deriv;
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
    const double delta_nb = 1.,
    const unsigned int obs_code = 0, // 0 - NegBinom; 1 - Poisson
    const unsigned int gain_code = 0,
    const Rcpp::NumericVector& coef = Rcpp::NumericVector::create(0.2,0,5.)) {
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
                deriv.at(i) = dlogJoint_drho(ypad,R,L,eta.at(1),eta.at(2),delta_nb,gain_code,coef);
            }
            break;
            case 3: // M - undefined
            {
                ::Rf_error("dlogJoint_deta - Derivative wrt M undefined.");
            }
            break;
            default:
            {
                ::Rf_error("dlogJoint_deta - idx_select undefined.");
            }
        }
    }

    return deriv;
}


/*
Forward Filtering and Backward Sampling
*/
arma::vec rtheta_ffbs(
    arma::mat& mt, // p x (n+1)
    arma::mat& at, // p x (n+1)
    arma::cube& Ct, // p x p x (n+1)
    arma::cube& Rt, // p x p x (n+1)
    arma::cube& Gt, // p x p x (n+1)
    arma::vec& alphat, // (n+1) x 1
    arma::vec& betat, // (n+1) x 1
    arma::vec& Ht, // (n+1) x 1
    const arma::uvec& model_code,
    const unsigned int n,
    const unsigned int p,
    const arma::vec& ypad, // (n+1) x 1
    const double W,
    const double alpha,
    const Rcpp::NumericVector& ctanh = Rcpp::NumericVector::create(0.2,0,5.),
    const unsigned int L = 0,
    const double rho = 0.9,
    const double delta = 0.9,
    const double mu0 = 0.,
    const double scale_sd = 1.e-16,
    const unsigned int nsample = 1,
    const unsigned int rtheta_type = 0, // 0 - marginal smoothing; 1 - conditional sampling
    const double delta_nb = 1.) {

    const unsigned int obs_code = model_code.at(0);
    const unsigned int link_code = model_code.at(1);
    const unsigned int trans_code = model_code.at(2);
    const unsigned int gain_code = model_code.at(3);
    const unsigned int err_code = model_code.at(4);
    
    arma::vec theta(n+1,arma::fill::zeros);

    forwardFilter(mt,at,Ct,Rt,Gt,alphat,betat,obs_code,link_code,trans_code,gain_code,n,p,ypad,ctanh,alpha,L,rho,mu0,W,NA_REAL,delta_nb,false);
    // Checked. OK.

    if (rtheta_type == 0) {
        backwardSmoother(theta,Ht,n,p,mt,at,Ct,Rt,Gt,W,delta);
    } else if (rtheta_type == 1) {
        backwardSampler(theta,n,p,mt,at,Ct,Rt,Gt,W,scale_sd);
    } else if (rtheta_type == 2) {
        theta = mt.row(0).t();
        Ht = arma::vectorise(Ct.tube(0,0));
    }
    
    // arma::vec theta_tmp = theta;
    // for (unsigned int i=0; i<nsample; i++) {
        
    //     theta += theta_tmp;
    // }
    // theta /= static_cast<double>(nsample);
    // 
    // Checked. OK.
    return theta;
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
    aw  amu   DK    DK 
    bw  bmu   DK    DK
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
    const double L = 2,
    const double alpha = 1.,
    const double delta = NA_REAL, // discount factor
    const Rcpp::Nullable<Rcpp::NumericVector>& m0_prior = R_NilValue,
	const Rcpp::Nullable<Rcpp::NumericMatrix>& C0_prior = R_NilValue,
    const Rcpp::NumericVector& ctanh = Rcpp::NumericVector::create(0.2,0,5.), // 3 x 1, the last one is M will be updated by eta[3] at each step
    const Rcpp::Nullable<Rcpp::NumericVector>& psi_init = R_NilValue, // previously `ht_`
    const unsigned int rtheta_type = 0, // 0 - marginal smoothing; 1 - conditional sampling. Only used for FFBS (sampler_type=0)
    const unsigned int sampler_type = 1, // 0 - FFBS; 1 - SMC
    const double Blag_pct = 0.15,
    const double MH_var = 1.,
    const double scale_sd = 1.e-16,
    const double learn_rate = 0.95,
    const double eps_step = 1.e-6,
    const unsigned int k = 1, // k <= sum(eta_select)
    const unsigned int nsample = 100,
    const unsigned int nburnin = 100,
    const unsigned int nthin = 2,
    const double delta_nb = 1.,
    const bool summarize_return = false,
    const bool verbose = false,
    const bool debug = false) {

    const unsigned int ntotal = nburnin + nthin*nsample + 1;
    const unsigned int n = Y.n_elem;
    const unsigned int npad = n + 1;
    arma::vec ypad(n+1,arma::fill::zeros); ypad.tail(n) = Y;

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
    const unsigned int m = arma::accu(eta_select);
    if (k > m) {
        ::Rf_error("k cannot be greater than m, total number of unknowns.");
    }

    arma::uvec idx_select = arma::find(eta_select == 1); // m x 1
    arma::vec eta_tilde(m,arma::fill::zeros); // unknown part of eta which is also transformed to the real line
    eta2tilde(eta_tilde,eta,idx_select,m);
    arma::vec gamma(m,arma::fill::ones);
    arma::vec nu = tYJ(eta_tilde,gamma); // m x 1, Yeo-Johnson transform of eta_tilde
    /* ------ Define Global Parameters ------ */


    /* ------ Define SMC ------ */
    const unsigned int Blag = static_cast<unsigned int>(Blag_pct*n); // B-fixed-lags Monte Carlo smoother
    const unsigned int N = 100; // number of particles for SMC
    /* ------ Define SMC ------ */


    /* ------ Define Model ------ */
    Rcpp::NumericVector ctanh_ = ctanh;
	unsigned int p, L_;
    init_by_trans(p,L_,trans_code,L);
    /* ------ Define Model ------ */


    /*  ------ Define LBE ------ */
    arma::mat mt(p,n+1,arma::fill::zeros);
    arma::cube Ct(p,p,n+1,arma::fill::zeros);
    if (!m0_prior.isNull()) {
		mt.col(0) = Rcpp::as<arma::vec>(m0_prior);
	}
	if (!C0_prior.isNull()) {
		Ct.slice(0) = Rcpp::as<arma::mat>(C0_prior);
	}

    arma::mat at(p,n+1,arma::fill::zeros);
    arma::cube Rt(p,p,n+1,arma::fill::zeros);
    arma::vec alphat(n+1,arma::fill::ones);
    arma::vec betat(n+1,arma::fill::ones);
    arma::vec Ht(n+1,arma::fill::zeros);

    arma::cube Gt(p,p,npad);
	arma::mat Gt0(p,p,arma::fill::zeros);
	Gt0.at(0,0) = 1.;
	if (trans_code == 0) { // Koyck
		Gt0.at(1,1) = eta.at(2);
	} else if (trans_code == 1) { // Koyama
		Gt0.diag(-1).ones();
	} else if (trans_code == 2) { // Solow
        double coef2 = -eta.at(2);
		Gt0.at(1,1) = -binom(L,1)*coef2;
		for (unsigned int k=2; k<p; k++) {
			coef2 *= -eta.at(2);
			Gt0.at(1,k) = -binom(L,k)*coef2;
			Gt0.at(k,k-1) = 1.;
		}
	} else if (trans_code == 3) { // Vanilla
		Gt0.at(0,0) = eta.at(2);
	}
	for (unsigned int t=0; t<npad; t++) {
		Gt.slice(t) = Gt0;
	}
    /*  ------ Define LBE ------ */


    /*  ------ Define HVB ------ */
    arma::vec tau = gamma2tau(gamma);
    arma::vec mu(m,arma::fill::zeros);
    arma::mat B(m,k,arma::fill::zeros); // Lower triangular part is nonzero
    arma::vec d(m,arma::fill::ones);
    
    arma::vec delta_logJoint(m);
    arma::vec delta_logq(m);
    arma::vec delta_diff(m);
    arma::mat dtheta_dB(m,m*k,arma::fill::zeros);

    arma::vec xi(k);
    arma::vec eps(m);

    arma::vec L_mu(m);
    arma::mat L_B(m,k,arma::fill::zeros);
    arma::vec vecL_B(m*k,arma::fill::zeros);
    arma::vec L_d(m);
    arma::vec L_tau(m);

    arma::vec oldEg2_mu(m,arma::fill::zeros);
    arma::vec curEg2_mu(m,arma::fill::zeros);
    arma::vec oldEdelta2_mu(m,arma::fill::zeros);
    arma::vec curEdelta2_mu(m,arma::fill::zeros);
    arma::vec Change_delta_mu(m,arma::fill::zeros);

    arma::vec oldEg2_B(m*k,arma::fill::zeros);
    arma::vec curEg2_B(m*k,arma::fill::zeros);
    arma::vec oldEdelta2_B(m*k,arma::fill::zeros);
    arma::vec curEdelta2_B(m*k,arma::fill::zeros);
    arma::vec Change_delta_B(m*k,arma::fill::zeros);

    arma::vec oldEg2_d(m,arma::fill::zeros);
    arma::vec curEg2_d(m,arma::fill::zeros);
    arma::vec oldEdelta2_d(m,arma::fill::zeros);
    arma::vec curEdelta2_d(m,arma::fill::zeros);
    arma::vec Change_delta_d(m,arma::fill::zeros);

    arma::vec oldEg2_tau(m,arma::fill::zeros);
    arma::vec curEg2_tau(m,arma::fill::zeros);
    arma::vec oldEdelta2_tau(m,arma::fill::zeros);
    arma::vec curEdelta2_tau(m,arma::fill::zeros);
    arma::vec Change_delta_tau(m,arma::fill::zeros);
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
    arma::mat delta_diff_stored(m,nsample);
    // arma::mat theta_last(p,nsample);
    // arma::vec theta(p);

    const unsigned int max_iter = 10;
    unsigned int cnt = 0;
    bool saveiter;
    for (unsigned int s=0; s<ntotal; s++) {
        R_CheckUserInterrupt();
		saveiter = s > nburnin && ((s-nburnin-1)%nthin==0);

        /*
        Step 2. Sample state parameters via posterior
            - psi: (n+1) x 1 gain factor
            - eta = (W,mu0,rho,M)
        */
        ctanh_[2] = eta.at(3);
        if (s==0 && !psi_init.isNull()) { // Initialization
            R.col(0) = Rcpp::as<arma::vec>(psi_init); // psi
            R.col(1) = psi2hpsi(R.col(0),gain_code,ctanh_); // hpsi
            R.submat(1,1,n,1) = hpsi2theta(R.col(1),Y,trans_code,0.,alpha,L_,eta.at(2)); // theta
            R.at(0,1) = 0.;

        } else if (eta_select.at(0)==1){
            // mcs_poisson(R,ypad,model_code,eta.at(0),eta.at(2),alpha,L,eta.at(1),Blag,N,R_NilValue,R_NilValue,ctanh_,delta_nb);
            mcs_poisson(R,ypad,model_code,eta.at(0),eta.at(2),alpha,L,eta.at(1),Blag,N,R_NilValue,R_NilValue,ctanh_,delta_nb,0.95);
        } else {
            mcs_poisson(R,ypad,model_code,NA_REAL,eta.at(2),alpha,L,eta.at(1),Blag,N,R_NilValue,R_NilValue,ctanh_,delta_nb,0.95);
        }


        /*
        Step 3. Compute gradient of the log variational distribution (Model Dependent)
        */
        // if (is_vanilla) {
        //     delta_logJoint = dlogJoint_deta(Y,psi,eta.at(2),eta.at(0),aw,bw,prior_type); // 1 x 1
        // } else {
        delta_logJoint = dlogJoint_deta(ypad,R,eta,idx_select,eta_prior_type,eta_prior_val,m,L,delta_nb,obs_code,gain_code,ctanh_); // m x 1
        // }

        if (!std::isfinite(delta_logJoint.at(0))) {
            Rcout << "iter=" << s+1 << std::endl;
            Rcout << "delta_logJoint=" << delta_logJoint.at(0) << std::endl;
            Rcout << "W=" << eta.at(0) << std::endl;
            ::Rf_error("delta_logJoint is NA.");
        }
        
        
        delta_logq = dlogq_dtheta(nu,eta_tilde,gamma,mu,B,d); // m x 1
        delta_diff = delta_logJoint - delta_logq; // m x 1
        if (!std::isfinite(delta_logq.at(0))) {
            Rcout << "iter=" << s+1 << std::endl;
            Rcout << "W=" << eta.at(0) << std::endl;
            Rcout << "delta_logq=" << delta_logq.at(0) << std::endl;
            ::Rf_error("delta_logq is NA.");
        }


        // TODO: transpose or no transpose
        L_mu = dYJinv_dnu(nu,gamma) * delta_diff;
        if (!std::isfinite(L_mu.at(0))) {
            Rcout << "iter=" << s+1 << std::endl;
            Rcout << "W=" << eta.at(0) << std::endl;
            Rcout << "L_mu=" << L_mu.at(0) << std::endl;
            ::Rf_error("L_mu is NA.");
        }

        if (m>1) {
            dtheta_dB = dYJinv_dB(nu,gamma,xi); // m x mk
            L_B = arma::reshape(dtheta_dB.t()*delta_diff,m,k); // m x k
            L_B.elem(arma::trimatu_ind(arma::size(L_B),1)).zeros();
            vecL_B = arma::vectorise(L_B);
        }

        L_d = dYJinv_dD(nu,gamma,eps)*delta_diff;
        if (!std::isfinite(L_d.at(0))) {
            Rcout << "iter=" << s+1 << std::endl;
            Rcout << "W=" << eta.at(0) << std::endl;
            Rcout << "L_d=" << L_d.at(0) << std::endl;
            ::Rf_error("L_d is NA.");
        }

        L_tau = dYJinv_dtau(nu,gamma)*delta_diff;
        if (!std::isfinite(L_tau.at(0))) {
            Rcout << "iter=" << s+1 << std::endl;
            Rcout << "W=" << eta.at(0) << std::endl;
            Rcout << "L_tau=" << L_tau.at(0) << std::endl;
            ::Rf_error("L_tau is NA.");
        }

        /*
        Step 4. Update Variational Parameters
        */
        // mu
        oldEg2_mu = curEg2_mu;
        oldEdelta2_mu = curEdelta2_mu;

        curEg2_mu = learn_rate*oldEg2_mu + (1.-learn_rate)*arma::pow(L_mu,2.); // m x 1
        Change_delta_mu = arma::sqrt(oldEdelta2_mu + eps_step)/arma::sqrt(curEg2_mu + eps_step) % L_mu;
        mu = mu + Change_delta_mu;
        curEdelta2_mu = learn_rate*oldEdelta2_mu + (1.-learn_rate)*arma::pow(Change_delta_mu,2.);


        // B
        if (m>1) {
            oldEg2_B = curEg2_B; // mk x 1
            oldEdelta2_B = curEdelta2_B; // mk x 1

            curEg2_B = learn_rate*oldEg2_B + (1.-learn_rate)*arma::pow(vecL_B,2.);
            Change_delta_B = arma::sqrt(oldEdelta2_B + eps_step) / arma::sqrt(curEg2_B + eps_step) % vecL_B; // mk x 1

            B = B + arma::reshape(Change_delta_B,m,k);
            curEdelta2_B = learn_rate*oldEdelta2_B + (1.-learn_rate)*arma::pow(Change_delta_B,2.);
        }


        // d
        oldEg2_d = curEg2_d;
        oldEdelta2_d = curEdelta2_d;

        curEg2_d = learn_rate*oldEg2_d + (1.-learn_rate)*arma::pow(L_d,2.); // m x 1
        Change_delta_d = arma::sqrt(oldEdelta2_d + eps_step)/arma::sqrt(curEg2_d + eps_step) % L_d;
        d = d + Change_delta_d;
        curEdelta2_d = learn_rate*oldEdelta2_d + (1.-learn_rate)*arma::pow(Change_delta_d,2.);


        // tau
        oldEg2_tau = curEg2_tau;
        oldEdelta2_tau = curEdelta2_tau;

        curEg2_tau = learn_rate*oldEg2_tau + (1.-learn_rate)*arma::pow(L_tau,2.); // m x 1
        Change_delta_tau = arma::sqrt(oldEdelta2_tau + eps_step)/arma::sqrt(curEg2_tau + eps_step) % L_tau;
        tau = tau + Change_delta_tau;
        curEdelta2_tau = learn_rate*oldEdelta2_tau + (1.-learn_rate)*arma::pow(Change_delta_tau,2.);
        gamma = tau2gamma(tau);



        /*
        Step 1. Sample static parameters from the variational distribution
        using reparameterisation.
        */
        cnt = 0;
        while(cnt<max_iter) {
            eta_tilde = rtheta(xi,eps,gamma,mu,B,d); // already inverse YJ transformed
            nu = tYJ(eta_tilde,gamma); // recover nu
            tilde2eta(eta,eta_tilde,idx_select,m);
            cnt++;
            if (std::isfinite(eta.at(0))) {break;}
        }
        

        if (!std::isfinite(eta.at(0))) {
            Rcout << "iter=" << s+1 << ", ";
            Rcout << "eta=" << eta_tilde.at(0) << ", ";
            Rcout << "W=" << eta.at(0) << std::endl;
            ::Rf_error("W is NA.");
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
            delta_diff_stored.col(idx_run) = delta_diff;
            // theta_last.col(idx_run) = theta;
		}

        if (verbose) {
			Rcout << "\rProgress: " << s << "/" << ntotal-1;
		}
    }

    if (verbose) {
		Rcout << std::endl;
	}

    Rcpp::List output;
    // output["mu"] = Rcpp::wrap(mu_stored);
    // output["d"] = Rcpp::wrap(d_stored);
    // output["gamma"] = Rcpp::wrap(gamma_stored);
    if (summarize_return) {
        arma::vec qProb = {0.025,0.5,0.975};
        output["psi"] = Rcpp::wrap(arma::quantile(psi_stored,qProb,1)); // (n+1) x 3
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

    output["delta_diff"] = Rcpp::wrap(delta_diff_stored);
    
    
    
    // output["theta_last"] = Rcpp::wrap(theta_last); // p x niter

    // if (debug) {
    //     output["Meff"] = Rcpp::wrap(Meff);
    //     output["resample_status"] = Rcpp::wrap(resample_status);
    // }
    

    return output;
}

