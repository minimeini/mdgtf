#include "lbe_poisson.h"
#include "pl_poisson.h"
#include "yjtrans.h"
using namespace Rcpp;
// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::depends(RcppArmadillo)]]


/*
------ dlogJoint_dWtilde ------
The derivative of the full joint density with respect to Wtilde=log(1/W), where W
is the evolution variance that affects
    psi[t] | psi[t-1] ~ N(psi[t-1], W)

Therefore, this function remains unchanged as long as we using the same evolution equation for psi.
*/
double dlogJoint_dWtilde(
    const arma::vec& theta, // (n+1) x 1, (theta[0],theta[1],...,theta[n])
    const double G, // evolution transition matrix
    const double W, // evolution variance conditional on V
    const double aw,
    const double bw,
    const unsigned int prior_type = 0) {// 0 - Gamma(aw=shape,bw=rate); 1 - Half-Cauchy(aw=location=0, bw=scale); 2 - Inverse-Gamma(aw=shape,bw=rate)
    /*
    Wtilde = -log(W)
    */
    
    const double n = static_cast<double>(theta.n_elem) - 1;
    const unsigned int ni = theta.n_elem - 1;
    double res = 0.;
    for (unsigned int t=0; t<ni; t++) {
        res += (theta.at(t+1)-G*theta.at(t)) * (theta.at(t+1)-G*theta.at(t));
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


// // No corresponding function in Matlab => DOUBLE CHECK IT.
// //' @export
// // [[Rcpp::export]]
// double dlogJoint_dtheta0(
//     const arma::vec& theta, // (n+1) x 1
//     const double G,
//     const double W,
//     const double V,
//     const double m0,
//     const double C0){

//     double res = 2.*theta.at(0)*std::pow(G,2.) - 2.*theta.at(1)*G;
//     res /= -(2.*V*W);
//     res -= 0.5*(2.*theta.at(0)/C0-2.*m0/C0);
//     return res;
// }

// eta = (Vtilde, Wtilde)
arma::vec dlogJoint_deta(
    const arma::vec& y, // n x 1
    const arma::vec& psi, // (n+1) x 1
    const double W,
    const double aw = 0.1,
    const double bw = 0.1,
    const unsigned int prior_type = 0) {
    arma::vec deriv(1);
    deriv.at(0) = dlogJoint_dWtilde(psi,1.,W,aw,bw,prior_type);
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
    const unsigned int ModelCode,
    const unsigned int TransferCode,
    const unsigned int n,
    const unsigned int p,
    const arma::vec& ypad, // (n+1) x 1
    const double W,
    const arma::vec& ctanh, // 3 x 1, coefficients for the hyperbolic tangent gain
    const double alpha = 1.,
    const unsigned int L = 0,
    const double rho = 0.9,
    const double delta = 0.9,
    const double mu0 = 0.,
    const double scale_sd = 1.e-16,
    const unsigned int nsample = 1,
    const unsigned int rtheta_type = 0, // 0 - marginal smoothing; 1 - conditional sampling
    const double delta_nb = 1.,
    const unsigned int obs_type = 1) { // 0 - negative binomial; 1 - poisson
    
    arma::vec theta(n+1,arma::fill::zeros);

    forwardFilter(mt,at,Ct,Rt,Gt,alphat,betat,ModelCode,TransferCode,n,p,ypad,ctanh,alpha,L,rho,mu0,W,NA_REAL,delta_nb,obs_type,false);
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



//' @export
// [[Rcpp::export]]
Rcpp::List hva_poisson2(
    const arma::vec& Y, // n x 1
    const unsigned int ModelCode,
    const double rho_true = 0.9,
    const double L = 0,
    const double alpha = 1.,
    const double delta = NA_REAL, // discount factor
    const double aw = 0.1,
    const double bw = 0.1,
    const Rcpp::Nullable<Rcpp::NumericVector>& m0_prior = R_NilValue,
	const Rcpp::Nullable<Rcpp::NumericMatrix>& C0_prior = R_NilValue,
    const Rcpp::Nullable<Rcpp::NumericVector>& ctanh = R_NilValue,
    const double th0_true = 0.,
    const double psi0_true = 0.,
    const double mu0_true = 0.,
    const Rcpp::Nullable<Rcpp::NumericVector>& ht_ = R_NilValue,
    const double W_init = NA_REAL,
    const unsigned int rtheta_type = 0, // 0 - marginal smoothing; 1 - conditional sampling
    const unsigned int sampler_type = 1, // 0 - FFBS; 1 - SMC
    const unsigned int obs_type = 1, // 0 - negative binomial; 1 - poisson
    const unsigned int prior_type = 0, // prior for W: 0 - Gamma(aw,bw)
    const double MH_var = 1.,
    const double scale_sd = 1.e-16,
    const double learn_rate = 0.95,
    const double eps_step = 1.e-6,
    const unsigned int niter = 100,
    const unsigned int nburnin = 100,
    const unsigned int nthin = 2,
    const unsigned int nsample = 100,
    const double rho_nb = 34.08792,
    const double delta_nb = 1.,
    const bool verbose = false,
    const bool debug = false) {

    const unsigned int n = Y.n_elem;
    const unsigned int npad = n + 1;
    arma::vec ypad(n+1,arma::fill::zeros); ypad.tail(n) = Y;
    arma::vec psi(n+1,arma::fill::zeros); psi.at(0) = psi0_true;

    arma::vec eta_tilde(1,arma::fill::zeros); // ( Wtilde)
    arma::vec nu(1,arma::fill::zeros);

    const unsigned int Blag = static_cast<unsigned int>(0.1*n); // B-fixed-lags Monte Carlo smoother
    const unsigned int N = 100; // number of particles for SMC
    arma::vec qProb = {0.5};

    /*
    ------ MCMC Disturbance Sampler ------
    */
    // arma::vec ht(n+1,arma::fill::zeros); // (h[0],h[1],...,h[n])
	// if (!ht_.isNull()) {
	// 	ht = Rcpp::as<arma::vec>(ht_);
	// }

	// arma::vec Fphi = get_Fphi(L); // L x 1
	// arma::mat Fx = update_Fx(ModelCode,n,Y,rho_true,L,Rcpp::wrap(ht),Rcpp::wrap(Fphi));
	// arma::vec th0tilde = update_theta0(ModelCode,n,Y,th0_true,psi0_true,rho_true,L,Rcpp::wrap(ht),Rcpp::wrap(Fphi));

    // double W;
    // arma::vec wt(n,arma::fill::zeros);
    // if (ModelCode == 9) {
    //     for (unsigned int t=0; t<n; t++) {
    //         wt.at(t) = ht.at(t+1) - rho_true*ht.at(t);
    //     }
    // }
    // arma::vec wt_accept(n,arma::fill::zeros);
    /*
    ------ MCMC Disturbance Sampler ------
    */
    

    /*
    ------ MCMC FFBS Sampler ------
    */
    const bool is_solow = ModelCode == 2 || ModelCode == 3 || ModelCode == 7 || ModelCode == 12 || ModelCode == 15;
	const bool is_koyck = ModelCode == 4 || ModelCode == 5 || ModelCode == 8 || ModelCode == 10 || ModelCode == 13;
	const bool is_koyama = ModelCode == 0 || ModelCode == 1 || ModelCode == 6 || ModelCode == 11 || ModelCode == 14;
	const bool is_vanilla = ModelCode == 9;
	unsigned int TransferCode; // integer indicator for the type of transfer function
	unsigned int p; // dimension of DLM state space
	unsigned int L_;
	if (is_koyck) { 
		TransferCode = 0; 
		p = 2;
		L_ = 0;
	} else if (is_koyama) { 
		TransferCode = 1; 
		p = L;
		L_ = L;
	} else if (is_solow) { 
		TransferCode = 2; 
		p = 3;
		L_ = 0;
	} else if (is_vanilla) {
		TransferCode = 3;
		p = 1;
		L_ = 0;
	} else {
		::Rf_error("Unknown type of model.");
	}

    arma::mat mt(p,n+1,arma::fill::zeros);
    arma::cube Ct(p,p,n+1,arma::fill::zeros);
    if (!m0_prior.isNull()) {
		mt.col(0) = Rcpp::as<arma::vec>(m0_prior);
	}
	if (!C0_prior.isNull()) {
		Ct.slice(0) = Rcpp::as<arma::mat>(C0_prior);
	}
    arma::vec ctanh_ = {0.3,-1.,3.};
	if (!ctanh.isNull()) {
		ctanh_ = Rcpp::as<arma::vec>(ctanh);
	}

    arma::mat at(p,n+1,arma::fill::zeros);
    arma::cube Rt(p,p,n+1,arma::fill::zeros);
    arma::vec alphat(n+1,arma::fill::ones);
    arma::vec betat(n+1,arma::fill::ones);
    arma::vec Ht(n+1,arma::fill::zeros);

    arma::cube Gt(p,p,npad);
	arma::mat Gt0(p,p,arma::fill::zeros);
	Gt0.at(0,0) = 1.;
	if (TransferCode == 0) { // Koyck
		Gt0.at(1,1) = rho_true;
	} else if (TransferCode == 1) { // Koyama
		Gt0.diag(-1).ones();
	} else if (TransferCode == 2) { // Solow
		Gt0.at(1,1) = 2.*rho_true;
		Gt0.at(1,2) = -rho_true*rho_true;
		Gt0.at(2,1) = 1.;
	} else if (TransferCode == 3) { // Vanilla
		Gt0.at(0,0) = rho_true;
	}
	for (unsigned int t=0; t<npad; t++) {
		Gt.slice(t) = Gt0;
	}

    /*
    ------ MCMC FFBS Sampler ------
    */

    double W;

    arma::vec gamma(1,arma::fill::ones);
    arma::vec tau = gamma2tau(gamma);
    arma::vec mu(1,arma::fill::zeros);
    arma::mat B(1,1,arma::fill::zeros); // Lower triangular
    arma::vec d(1,arma::fill::ones);
    
    arma::vec delta_logJoint(1);
    arma::vec delta_logq(1);
    arma::vec delta_diff(1);

    arma::vec xi(1);
    arma::vec eps(1);

    arma::vec L_mu(1);
    arma::mat L_B(1,1);
    arma::vec L_d(1);
    arma::vec L_tau(1);

    arma::vec oldEg2_mu(1,arma::fill::zeros);
    arma::vec curEg2_mu(1,arma::fill::zeros);
    arma::vec oldEdelta2_mu(1,arma::fill::zeros);
    arma::vec curEdelta2_mu(1,arma::fill::zeros);
    arma::vec Change_delta_mu(1,arma::fill::zeros);

    arma::vec oldEg2_d(1,arma::fill::zeros);
    arma::vec curEg2_d(1,arma::fill::zeros);
    arma::vec oldEdelta2_d(1,arma::fill::zeros);
    arma::vec curEdelta2_d(1,arma::fill::zeros);
    arma::vec Change_delta_d(1,arma::fill::zeros);

    arma::vec oldEg2_tau(1,arma::fill::zeros);
    arma::vec curEg2_tau(1,arma::fill::zeros);
    arma::vec oldEdelta2_tau(1,arma::fill::zeros);
    arma::vec curEdelta2_tau(1,arma::fill::zeros);
    arma::vec Change_delta_tau(1,arma::fill::zeros);


    arma::mat mu_stored(1,niter);
    arma::mat d_stored(1,niter);
    arma::mat gamma_stored(1,niter);
    arma::mat psi_stored(n+1,niter);
    arma::vec W_stored(niter);
    arma::mat psi_last(p,niter);

    const unsigned int max_iter = 10;
    unsigned int cnt = 0;

    
    for (unsigned int s=0; s<niter; s++) {
        R_CheckUserInterrupt();

        /*
        Step 1. Sample static parameters from the variational distribution
        using reparameterisation.
        */
        if (s==0 && !R_IsNA(W_init)) {
            // Initial value
            W = W_init;
        } else {
            cnt = 0;
            while(cnt<max_iter) {
                eta_tilde = rtheta(xi,eps,gamma,mu,B,d);
                nu = tYJ(eta_tilde,gamma);
                W = std::exp(-eta_tilde.at(0));
                cnt++;
                if (std::isfinite(W)) {break;}
            }
            
        }
        

        if (!std::isfinite(W)) {
            Rcout << "iter=" << s+1 << ", ";
            Rcout << "eta=" << eta_tilde.at(0) << ", ";
            Rcout << "W=" << W << std::endl;
            ::Rf_error("W is NA.");
        }
        

        /*
        Step 2. Sample state parameters via posterior
            - theta: (n+1) x 1 vector
        */
        if (s==0 && !ht_.isNull()) {
            psi = Rcpp::as<arma::vec>(ht_);
        } else if (sampler_type==0) {
            psi = rtheta_ffbs(mt,at,Ct,Rt,Gt,alphat,betat,Ht,ModelCode,TransferCode,n,p,ypad,W,ctanh_,alpha,L,rho_true,delta,mu0_true,scale_sd,rtheta_type,delta_nb,obs_type);
        } else {
            Rcpp::List mcs_output = mcs_poisson(Y,ModelCode,W,rho_true,alpha,L,mu0_true,Blag,N,R_NilValue,R_NilValue,R_NilValue,ctanh,delta_nb,obs_type,verbose,debug);
            arma::mat psi_mat = Rcpp::as<arma::mat>(mcs_output["quantiles"]);
            psi = psi_mat.col(1);
            psi_last.col(s) = arma::median(Rcpp::as<arma::mat>(mcs_output["theta_last"]),1);
        }
        
        // rtheta_disturbance(wt,wt_accept,Y,ModelCode,W,ht,Fphi,Fx,th0tilde,L,mu0_true,m0,C0,MH_var,false,nburnin,nthin,nsample);
        // if (ModelCode == 9) {
        //     for (unsigned int t=1; t<=n; t++) {
        //         theta.at(t) = rho_true * theta.at(t-1) + wt.at(t-1);
        //     }
        // }


        /*
        Step 3. Compute gradient of the log variational distribution (Model Dependent)
        */
        delta_logJoint = dlogJoint_deta(Y,psi,W,aw,bw,prior_type); // 1 x 1


        if (!std::isfinite(delta_logJoint.at(0))) {
            Rcout << "iter=" << s+1 << std::endl;
            Rcout << "delta_logJoint=" << delta_logJoint.at(0) << std::endl;
            Rcout << "W=" << W << std::endl;
            Rcout << "theta=" << psi.t() << std::endl;
            ::Rf_error("delta_logJoint is NA.");
        }

        
        
        delta_logq = dlogq_dtheta(nu,eta_tilde,gamma,mu,B,d); // 1 x 1
        delta_diff = delta_logJoint - delta_logq; // 1 x 1
        if (!std::isfinite(delta_logq.at(0))) {
            Rcout << "iter=" << s+1 << std::endl;
            Rcout << "W=" << W << std::endl;
            Rcout << "delta_logq=" << delta_logq.at(0) << std::endl;
            ::Rf_error("delta_logq is NA.");
        }

        // TODO: transpose or no transpose
        L_mu = dYJinv_dnu(nu,gamma) * delta_diff;
        if (!std::isfinite(L_mu.at(0))) {
            Rcout << "iter=" << s+1 << std::endl;
            Rcout << "W=" << W << std::endl;
            Rcout << "L_mu=" << L_mu.at(0) << std::endl;
            ::Rf_error("L_mu is NA.");
        }
        // L_B = arma::reshape(arma::inplace_trans(dYJinv_dB(nu,gamma,xi))*delta_diff,2,2);
        // L_B.elem(arma::trimatu_ind(arma::size(L_B),1)).zeros();
        L_d = dYJinv_dD(nu,gamma,eps)*delta_diff;
        if (!std::isfinite(L_d.at(0))) {
            Rcout << "iter=" << s+1 << std::endl;
            Rcout << "W=" << W << std::endl;
            Rcout << "L_d=" << L_d.at(0) << std::endl;
            ::Rf_error("L_d is NA.");
        }
        L_tau = dYJinv_dtau(nu,gamma)*delta_diff;
        if (!std::isfinite(L_tau.at(0))) {
            Rcout << "iter=" << s+1 << std::endl;
            Rcout << "W=" << W << std::endl;
            Rcout << "L_tau=" << L_tau.at(0) << std::endl;
            ::Rf_error("L_tau is NA.");
        }


        /*
        Step 4. Update Variational Parameters
        */
        // mu
        oldEg2_mu = curEg2_mu;
        oldEdelta2_mu = curEdelta2_mu;

        curEg2_mu = learn_rate*oldEg2_mu + (1.-learn_rate)*arma::pow(L_mu,2.); // 2 x 1
        Change_delta_mu = arma::sqrt(oldEdelta2_mu + eps_step)/arma::sqrt(curEg2_mu + eps_step) % L_mu;
        mu = mu + Change_delta_mu;
        curEdelta2_mu = learn_rate*oldEdelta2_mu + (1.-learn_rate)*arma::pow(Change_delta_mu,2.);

        // d
        oldEg2_d = curEg2_d;
        oldEdelta2_d = curEdelta2_d;

        curEg2_d = learn_rate*oldEg2_d + (1.-learn_rate)*arma::pow(L_d,2.); // 2 x 1
        Change_delta_d = arma::sqrt(oldEdelta2_d + eps_step)/arma::sqrt(curEg2_d + eps_step) % L_d;
        d = d + Change_delta_d;
        curEdelta2_d = learn_rate*oldEdelta2_d + (1.-learn_rate)*arma::pow(Change_delta_d,2.);

        // tau
        oldEg2_tau = curEg2_tau;
        oldEdelta2_tau = curEdelta2_tau;

        curEg2_tau = learn_rate*oldEg2_tau + (1.-learn_rate)*arma::pow(L_tau,2.); // 2 x 1
        Change_delta_tau = arma::sqrt(oldEdelta2_tau + eps_step)/arma::sqrt(curEg2_tau + eps_step) % L_tau;
        tau = tau + Change_delta_tau;
        curEdelta2_tau = learn_rate*oldEdelta2_tau + (1.-learn_rate)*arma::pow(Change_delta_tau,2.);
        gamma = tau2gamma(tau);

        mu_stored.col(s) = mu;
        d_stored.col(s) = d;
        gamma_stored.col(s) = gamma;
        psi_stored.col(s) = psi;
        W_stored.at(s) = W;

        if (verbose) {
			Rcout << "\rProgress: " << s+1 << "/" << niter;
		}
    }

    if (verbose) {
		Rcout << std::endl;
	}

    Rcpp::List output;
    output["mu"] = Rcpp::wrap(mu_stored);
    output["d"] = Rcpp::wrap(d_stored);
    output["gamma"] = Rcpp::wrap(gamma_stored);
    output["psi_stored"] = Rcpp::wrap(psi_stored); // (n+1) x niter
    output["W_stored"] = Rcpp::wrap(W_stored); // niter
    output["psi_last"] = Rcpp::wrap(psi_last); // p x niter

    return output;
}