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
    const double bw) {
    
    const double n = static_cast<double>(theta.n_elem) - 1.;
    const unsigned int ni = theta.n_elem - 1;
    double res = 0.;
    for (unsigned int t=0; t<ni; t++) {
        res += (theta.at(t+1)-G*theta.at(t)) * (theta.at(t+1)-G*theta.at(t));
    }
    res *= 0.5;
    res += bw;
    res /= -W;
    res += aw + 0.5*n;
    return res;
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
    const arma::vec& theta, // (n+1) x 1
    const double G,
    const double W,
    const double aw = 0.1,
    const double bw = 0.1) {
    arma::vec deriv(1);
    deriv.at(0) = dlogJoint_dWtilde(theta,G,W,aw,bw);
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

    forwardFilter(mt,at,Ct,Rt,Gt,alphat,betat,ModelCode,TransferCode,n,p,ypad,L,rho,mu0,W,NA_REAL,delta_nb,obs_type,false);
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
Rcpp::List hva_poisson(
    const arma::vec& Y, // n x 1
    const unsigned int ModelCode,
    const double rho_true = 0.9,
    const double L = 0,
    const double delta = NA_REAL, // discount factor
    const double aw = 0.1,
    const double bw = 0.1,
    const Rcpp::Nullable<Rcpp::NumericVector>& m0_prior = R_NilValue,
	const Rcpp::Nullable<Rcpp::NumericMatrix>& C0_prior = R_NilValue,
    const double th0_true = 0.,
    const double psi0_true = 0.,
    const double mu0_true = 0.,
    const Rcpp::Nullable<Rcpp::NumericVector>& ht_ = R_NilValue,
    const double W_init = NA_REAL,
    const unsigned int rtheta_type = 0, // 0 - marginal smoothing; 1 - conditional sampling
    const unsigned int sampler_type = 1, // 0 - FFBS; 1 - SMC
    const unsigned int obs_type = 1, // 0 - negative binomial; 1 - poisson
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
    arma::vec theta(n+1,arma::fill::zeros); theta.at(0) = th0_true;

    arma::vec eta(1,arma::fill::zeros); // ( Wtilde)
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
    const bool is_solow = ModelCode == 2 || ModelCode == 3 || ModelCode == 7 || ModelCode == 12;
	const bool is_koyck = ModelCode == 4 || ModelCode == 5 || ModelCode == 8 || ModelCode == 10;
	const bool is_koyama = ModelCode == 0 || ModelCode == 1 || ModelCode == 6 || ModelCode == 11;
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

	if (!m0_prior.isNull()) {
		mt.col(0) = Rcpp::as<arma::vec>(m0_prior);
	}
	if (!C0_prior.isNull()) {
		Ct.slice(0) = Rcpp::as<arma::mat>(C0_prior);
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
    arma::mat theta_stored(n+1,niter);
    arma::vec W_stored(niter);

    arma::mat Meff(n,niter);
    arma::mat resample_status(n,niter);
    arma::mat theta_last(p,niter);

    
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
            eta = rtheta(xi,eps,gamma,mu,B,d);
            nu = tYJ(eta,gamma);
            W = std::exp(-eta.at(0));
        }
        

        if (!std::isfinite(W)) {
            Rcout << "iter=" << s+1 << ", ";
            Rcout << "eta=" << eta.at(0) << ", ";
            Rcout << "W=" << W << std::endl;
            ::Rf_error("W is NA.");
        }
        

        /*
        Step 2. Sample state parameters via posterior
            - theta: (n+1) x 1 vector
        */
        if (s==0 && !ht_.isNull()) {
            theta = Rcpp::as<arma::vec>(ht_);
        } else if (sampler_type==0) {
            theta = rtheta_ffbs(mt,at,Ct,Rt,Gt,alphat,betat,Ht,ModelCode,TransferCode,n,p,ypad,W,L,rho_true,delta,mu0_true,scale_sd,rtheta_type,delta_nb,obs_type);
        } else {
            Rcpp::List mcs_output = mcs_poisson(Y,ModelCode,W,rho_true,L,mu0_true,Blag,N,R_NilValue,R_NilValue,R_NilValue,rho_nb,delta_nb,obs_type,verbose,debug);
            arma::mat theta_mat = Rcpp::as<arma::mat>(mcs_output["quantiles"]);
            Meff.col(s) = Rcpp::as<arma::vec>(mcs_output["Meff"]);
            resample_status.col(s) = Rcpp::as<arma::vec>(mcs_output["resample_status"]);
            theta = theta_mat.col(1);
            theta_last.col(s) = arma::median(Rcpp::as<arma::mat>(mcs_output["theta_last"]),1);
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
        if (is_vanilla) {
            delta_logJoint = dlogJoint_deta(Y,theta,rho_true,W,aw,bw); // 1 x 1
        } else {
            delta_logJoint = dlogJoint_deta(Y,theta,1.,W,aw,bw); // 1 x 1
        }

        if (!std::isfinite(delta_logJoint.at(0))) {
            Rcout << "iter=" << s+1 << std::endl;
            Rcout << "delta_logJoint=" << delta_logJoint.at(0) << std::endl;
            Rcout << "W=" << W << std::endl;
            Rcout << "theta=" << theta.t() << std::endl;
            ::Rf_error("delta_logJoint is NA.");
        }

        
        
        delta_logq = dlogq_dtheta(nu,eta,gamma,mu,B,d); // 1 x 1
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
        theta_stored.col(s) = theta;
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
    output["theta_stored"] = Rcpp::wrap(theta_stored); // (n+1) x niter
    output["W_stored"] = Rcpp::wrap(W_stored); // niter
    output["theta_last"] = Rcpp::wrap(theta_last); // p x niter
    output["Meff"] = Rcpp::wrap(Meff);
    output["resample_status"] = Rcpp::wrap(resample_status);

    return output;
}


/*
--------------------------------------------------------------------------------------
Below is the MCMC Disturbance MH Sampler Based Hybrid Variational Approximation

NOT RECOMMENDED because

1. Doesn't seem to work
2. No advantage over MCMC disturbance MH sampler in terms of computational efficiency and inference performance.
3. If we use MH for the evolution errors w[t], we cans sample global parameters exactly via MCMC, which definitely outperform HVA.
--------------------------------------------------------------------------------------
*/


double dlogJoint_dWtilde_disturbance(
    const double W, // evolution variance conditional on V
    const arma::vec& wt, // n x 1, (wt[1],...,wt[n])
    const double cw,
    const double dw,
    const unsigned int n) {
    
    const double n_ = static_cast<double>(n);
    double res = 0.;
    for (unsigned int t=1; t<n; t++) {
        res += wt.at(t)*wt.at(t);
    }
    res *= 0.5;
    res += dw;
    res /= -W;
    res += cw + 0.5*(n_-1);
    return res;
}


double dlogJoint_dth0_disturbance(
    const double th0,
    const arma::vec& y, // n x 1
    const arma::vec& lambda,
    const double rho,
    const double m0,
    const double C0,
    const unsigned int n) {
    
    const double n_ = static_cast<double>(n);
    double res = 0.;
    for (unsigned int t=0; t<n; t++) {
        res += (y.at(t)/lambda.at(t) - 1.)*std::pow(rho,static_cast<double>(t+1));
    }
    res -= (th0 - m0) / C0;
    return res;
}


// eta = (Vtilde, Wtilde)
arma::vec dlogJoint_deta_disturbance(
    const arma::vec& y, // n x 1
    const arma::vec& lambda, // n x 1
    const arma::vec& wt, // n x 1
    const double rho,
    const double W,
    const double th0,
    const unsigned int n,
    const double cw = 0.1,
    const double dw = 0.1,
    const double m0 = 0.,
    const double C0 = 1.) {
    arma::vec deriv(1);
    deriv.at(0) = dlogJoint_dWtilde_disturbance(W,wt,cw,dw,n);
    return deriv;
}



void rtheta_disturbance(
    arma::vec& wt, // n x 1, also serves as the initial value
    arma::vec& wt_accept, // n x 1
    const arma::vec& Y, // n x 1
    const unsigned int ModelCode,
    const double W_cur,
    const arma::vec& ht, // (n+1)
    const arma::vec& Fphi,
    const arma::mat& Fx,
    const arma::vec& th0tilde,
    const double L = 0,
    const double mu0_cur = 0.,
    const double aw = 0,
    const double Rw = 100,
    const double MH_var = 1.,
    const bool use_lambda_bound = false,
    const unsigned int niter = 10) {

    const bool use_exp_link = ModelCode==6 || ModelCode==7 || ModelCode==8 || ModelCode==9;
	const unsigned int n = Y.n_elem;
	const double n_ = static_cast<double>(n);

    const double Wrt = std::sqrt(W_cur);
    const double Rwrt = std::sqrt(Rw);

	arma::vec theta(n,arma::fill::zeros); // n x 1
	arma::vec lambda(n,arma::fill::zeros); // n x 1

    double bt,Bt,Btrt;
	arma::vec Ytilde = Y;
	arma::vec Yhat(n);

	double wt_old, wt_new;
	double logp_old,logp_new,logq_old,logq_new,logratio;
	
    arma::mat wt_stored(n,niter,arma::fill::zeros);
    for (unsigned int b=0; b<niter; b++) {
		R_CheckUserInterrupt();

		// [OK] Update evolution/state disturbances/errors, denoted by vt.
		for (unsigned int t=0; t<n; t++) {
			/* Part 1. Checked - OK */
			wt_old = wt.at(t);
		    theta = th0tilde + Fx * wt;
			// Et.elem(arma::find(Et>EBOUND)).fill(EBOUND); //**

			// lambda filled with old values of wt[t]
			if (use_exp_link) {
				// Exponential Link
				// 6 - KoyamaEye, 7 - SolowEye, 8 - KoyckEye
				lambda = arma::exp(mu0_cur+theta);
			} else {
				// Others using identity link.
				lambda = arma::datum::eps+mu0_cur+theta;
			}

			logp_old = 0.;
			for (unsigned int j=t;j<n;j++) {
				logp_old += R::dpois(Y.at(j),lambda.at(j),true);
			}
			if (t==0) {
				logp_old += R::dnorm(wt_old,aw,Rwrt,true);
			} else {
				logp_old += R::dnorm(wt_old,0.,Wrt,true);
			}
			/* Part 1. Checked - OK */

			/* Part 2. */
			// Sample wt[t](new) given wt[t](old)
			if (use_exp_link) {
				// Linearlisation for exponential link
				Ytilde = theta + (Y - lambda) / lambda; 
			} // Otherwise, Ytilde == Y.
			
		    Yhat = Ytilde - theta + Fx.col(t)*wt_old; // TODO - CHECK HERE
		    if (t==0) {
				if (use_exp_link) {
					// Exponential link, Variance V[t]= 1/lambda[t]
					Bt = 1. / (1./Rw + arma::accu(arma::pow(Fx.col(t),2.) % lambda));
        	    	bt = Bt * (aw/Rw + arma::accu(Fx.col(t)%Yhat%lambda));
				} else {
					// Identity link, Variance V[t] = lambda[t]
					Bt = 1. / (1./Rw + arma::accu(arma::pow(Fx.col(t),2.) / lambda));
        	    	bt = Bt * (aw/Rw + arma::accu(Fx.col(t)%Yhat/lambda));
				}
      	    } else {
				if (use_exp_link) {
					// Exponential link, Variance V[t] = 1/lambda[t]
					Bt = 1./(1./W_cur + arma::accu(arma::pow(Fx.col(t),2.) % lambda));
        	    	bt = Bt * (arma::accu(Fx.col(t)%Yhat%lambda));
				} else {
					// Identity link
					Bt = 1./(1./W_cur + arma::accu(arma::pow(Fx.col(t),2.) / lambda));
        	    	bt = Bt * (arma::accu(Fx.col(t)%Yhat/lambda));
				}
      		    
      	    }
			Btrt = std::sqrt(Bt*MH_var);
		    wt_new = R::rnorm(bt, Btrt);
			if (!std::isfinite(wt_new)) {
				// Just reject it and move onto next t.
				wt.at(t) = wt_old;
				continue;
			}

			logq_new = R::dnorm(wt_new,bt,Btrt,true);
			/* Part 2. */

			/* Part 3. */
			wt.at(t) = wt_new;
			theta = th0tilde + Fx * wt;
			// Et.elem(arma::find(Et>EBOUND)).fill(EBOUND);
			// lambda filled with new values of wt[t]
			if (use_exp_link) {
				// Exponential Link
				// 6 - KoyamaEye, 7 - SolowEye, 8 - KoyckEye, 9 - Vanilla
				lambda = arma::exp(mu0_cur+theta);
				if (use_lambda_bound && (lambda.has_inf() || lambda.has_nan())) {
					wt.at(t) = wt_old; // Reject if out of bound
					continue;
				}
			} else {
				// Others using identity link.
				lambda = arma::datum::eps+mu0_cur+theta;
				if (use_lambda_bound && arma::any(lambda<arma::datum::eps)) {
					wt.at(t) = wt_old; // Reject if out of bound
					continue;
				}
			}
			logp_new = 0.;
			for (unsigned int j=t;j<n;j++) {
				logp_new += R::dpois(Y.at(j),lambda.at(j),true);
			}
			if (t==0) {
				logp_new += R::dnorm(wt_new,aw,Rwrt,true);
			} else {
				logp_new += R::dnorm(wt_new,0.,Wrt,true);
			}
			/* Part 3. */

			/* Part 4. */
			// Transition probability to wt[t](old) from wt[t](new)
			if (use_exp_link) {
				// Linearlisation for exponential link
				Ytilde = theta + (Y - lambda) / lambda; 
			} // Otherwise, Ytilde == Y.

		    Yhat = Ytilde - theta + Fx.col(t)*wt_new;
			if (t==0) {
				if (use_exp_link) {
					// Exponential link, Variance V[t]= 1/lambda[t]
					Bt = 1. / (1./Rw + arma::accu(arma::pow(Fx.col(t),2.) % lambda));
        	    	bt = Bt * (aw/Rw + arma::accu(Fx.col(t)%Yhat%lambda));
				} else {
					// Identity link, Variance V[t] = lambda[t]
					Bt = 1. / (1./Rw + arma::accu(arma::pow(Fx.col(t),2.) / lambda));
        	    	bt = Bt * (aw/Rw + arma::accu(Fx.col(t)%Yhat/lambda));
				}
      	    } else {
				if (use_exp_link) {
					// Exponential link, Variance V[t] = 1/lambda[t]
					Bt = 1./(1./W_cur + arma::accu(arma::pow(Fx.col(t),2.) % lambda));
        	    	bt = Bt * (arma::accu(Fx.col(t)%Yhat%lambda));
				} else {
					// Identity link
					Bt = 1./(1./W_cur + arma::accu(arma::pow(Fx.col(t),2.) / lambda));
        	    	bt = Bt * (arma::accu(Fx.col(t)%Yhat/lambda));
				}
      	    }
			Btrt = std::sqrt(Bt*MH_var);
			logq_old = R::dnorm(wt_old,bt,Btrt,true);
			/* Part 4. */

			logratio = std::min(0.,logp_new-logp_old+logq_old-logq_new);
			if (std::log(R::runif(0.,1.)) >= logratio) { // reject
				wt.at(t) = wt_old;
			} else {
				wt_accept.at(t) += 1.;
			}
	    } // Loops

		// store samples after burnin and thinning
		wt_stored.col(b) = wt;
	}

    wt = arma::mean(wt_stored,1);
}



//' @export
// [[Rcpp::export]]
Rcpp::List hva_poisson_disturbance(
    const arma::vec& Y, // n x 1
    const unsigned int ModelCode,
    const double rho_true = 0.9,
    const double L = 0,
    const double cw = 0.1,
    const double dw = 0.1,
    const double aw = 1.,
    const double Rw = 100.,
    const double m0 = 0.,
    const double C0 = 100.,
    const double th0_true = 0.,
    const double psi0_true = 0.,
    const double mu0_true = 0.,
    const Rcpp::Nullable<Rcpp::NumericVector>& ht_ = R_NilValue,
    const double MH_var = 1.,
    const double learn_rate = 0.95,
    const double eps_step = 1.e-6,
    const unsigned int niter = 100,
    const unsigned int nburnin = 100,
    const unsigned int nthin = 2,
    const unsigned int nsample = 100,
    const bool verbose = true) {

    const unsigned int n = Y.n_elem;
    double W = dw / (cw - 1.);
    arma::vec wt(n,arma::fill::zeros);
    arma::vec eta(1,arma::fill::zeros); // ( Wtilde)
    arma::vec nu(1,arma::fill::zeros);

    /*
    ------ MCMC Disturbance Sampler ------
    */
    const bool use_exp_link = ModelCode==6 || ModelCode==7 || ModelCode==8 || ModelCode==9;

    arma::vec ht(n+1,arma::fill::zeros); // (h[0],h[1],...,h[n])
	if (!ht_.isNull()) {
		ht = Rcpp::as<arma::vec>(ht_);
	}

	arma::vec Fphi = get_Fphi(L); // L x 1
	arma::mat Fx = update_Fx(ModelCode,n,Y,rho_true,L,Rcpp::wrap(ht),Rcpp::wrap(Fphi));
	arma::vec th0tilde = update_theta0(ModelCode,n,Y,th0_true,psi0_true,rho_true,L,Rcpp::wrap(ht),Rcpp::wrap(Fphi));

    arma::vec theta(n,arma::fill::zeros);
    arma::vec lambda(n,arma::fill::zeros);

    
    if (ModelCode == 9) {
        for (unsigned int t=0; t<n; t++) {
            wt.at(t) = ht.at(t+1) - rho_true*ht.at(t);
        }
    } else {
        for (unsigned int t=0; t<n; t++) {
            wt.at(t) = ht.at(t+1) - ht.at(t);
        }
    }
    arma::vec wt_accept(n,arma::fill::zeros);
    /*
    ------ MCMC Disturbance Sampler ------
    */
    
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


    arma::mat mu_stored(1,nsample);
    arma::mat d_stored(1,nsample);
    arma::mat gamma_stored(1,nsample);
    arma::mat wt_stored(n,nsample);
    arma::vec W_stored(nsample);

    const unsigned int ntotal = nburnin + nthin*nsample + 1;
    bool saveiter;

    
    for (unsigned int s=0; s<ntotal; s++) {
        R_CheckUserInterrupt();
        saveiter = s > nburnin && ((s-nburnin-1)%nthin==0);
        
        /*
        Step 2. Sample state parameters via posterior
        */
        rtheta_disturbance(wt,wt_accept,Y,ModelCode,W,ht,Fphi,Fx,th0tilde,L,mu0_true,aw,Rw,MH_var,false,niter);
        theta = th0tilde + Fx * wt;
        if (use_exp_link) {
			// Exponential Link
			// 6 - KoyamaEye, 7 - SolowEye, 8 - KoyckEye
			lambda = arma::exp(mu0_true+theta);
		} else {
			// Others using identity link.
			lambda = arma::datum::eps+mu0_true+theta;
		}


        /*
        Step 3. Compute gradient of the log variational distribution
        */
        delta_logJoint = dlogJoint_deta_disturbance(Y,lambda,wt,rho_true,W,th0_true,n,cw,dw,m0,C0); // 1 x 1

        if (!std::isfinite(delta_logJoint.at(0))) {
            Rcout << "iter=" << s+1 << std::endl;
            Rcout << "delta_logJoint=" << delta_logJoint.at(0) << std::endl;
            Rcout << "W=" << W << std::endl;
            ::Rf_error("delta_logJoint is NA.");
        }

        
        
        delta_logq = dlogq_dtheta(nu,eta,gamma,mu,B,d); // 1 x 1
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


        /*
        Step 1. Sample static parameters from the variational distribution
        using reparameterisation.
        */
        eta = rtheta(xi,eps,gamma,mu,B,d);
        nu = tYJ(eta,gamma);
        W = std::exp(-eta.at(0));
        if (!std::isfinite(W)) {
            Rcout << "iter=" << s+1 << std::endl;
            Rcout << "eta=" << eta.at(0) << std::endl;
            Rcout << "W=" << W << std::endl;
            ::Rf_error("W is NA.");
        }

        if (saveiter || s==(ntotal-1)) {
			unsigned int idx_run;
			if (saveiter) {
				idx_run = (s-nburnin-1)/nthin;
			} else {
				idx_run = nsample - 1;
			}

			wt_stored.col(idx_run) = wt;
			W_stored.at(idx_run) = W;
            mu_stored.col(idx_run) = mu;
            d_stored.col(idx_run) = d;
            gamma_stored.col(idx_run) = gamma;
			// rho_stored.at(idx_run) = rho;
			// E0_stored.at(idx_run) = E0;
		}


        if (verbose) {
			Rcout << "\rProgress: " << s+1 << "/" << ntotal;
		}
    }

    if (verbose) {
		Rcout << std::endl;
	}

    Rcpp::List output;
    output["mu"] = Rcpp::wrap(mu_stored);
    output["d"] = Rcpp::wrap(d_stored);
    output["gamma"] = Rcpp::wrap(gamma_stored);
    output["wt"] = Rcpp::wrap(wt_stored);
    output["W"] = Rcpp::wrap(W_stored);

    return output;
}