#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <algorithm>
#include <RcppArmadillo.h>
#include "hva_gradients.h"
using namespace Rcpp;
// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::depends(RcppArmadillo)]]


double Pd(
    const double d,
    const double mu,
    const double sigmasq) {
    arma::vec tmpv1(1);
    tmpv1.at(0) = -(std::log(d)-mu)/std::sqrt(2.*sigmasq);
    return arma::as_scalar(0.5*arma::erfc(tmpv1));
}

double knl(
    const double t,
    const double mu,
    const double sd2) {
    return Pd(t,mu,sd2) - Pd(t-1.,mu,sd2);
}


//' @export
// [[Rcpp::export]]
arma::vec get_Fphi(const unsigned int L){
    double tmpd;

    const double mu = arma::datum::eps;
    const double m = 4.7;
    const double s = 2.9;
    const double sm2 = std::pow(s/m,2);
    const double pk_mu = std::log(m/std::sqrt(1.+sm2));
    const double pk_sg2 = std::log(1.+sm2);

    arma::vec Fphi(L,arma::fill::zeros);
    Fphi.at(0) = knl(1.,pk_mu,pk_sg2);
    for (unsigned int d=1; d<L; d++) {
        tmpd = static_cast<double>(d) + 1.;
        Fphi.at(d) = knl(tmpd,pk_mu,pk_sg2);
    }
    return Fphi;
}


/*
---- Method ----
- Mean-Field Variational Inference
- Ordinary Kalman filtering with all static parameters known
- Using the DLM formulation

---- Model ----
<obs>   y[t] ~ Pois(lambda[t]), t=1,...,n
<link>  lambda[t] = phi[1]*y[t-1]*exp(psi[t]) + ... + phi[L]*y[t-L]*exp(psi[t-L+1]),t=1,...,n
<state> psi[t] = psi[t-1] + omega[t], t=1,...,n
        omega[t] ~ Normal(0,W)

<prior> theta_till[0] ~ Norm(m0, C0)
        W ~ IG(aw,bw)
*/
void forwardFilter(
	const arma::vec& Y, // n x 1, the observation (scalar), n: num of obs
    const arma::mat& G, // L x L, state transition matrix
    const arma::vec& Fphi, // L x 1, state-to-obs nonlinear mapping vector
    const unsigned int Ftype, // 0: exp(psi); 1: max(psi,0.)
	arma::mat& mt, // L x (n+1), t=0 is the mean for initial value theta[0]
	arma::mat& at, // L x (n+1)
	arma::cube& Ct, // L x L x (n+1), t=0 is the var for initial value theta[0]
	arma::cube& Rt, // L x L x (n+1)
    const double W = NA_REAL, // state variance
    const double delta = NA_REAL) { // discount factor

	const unsigned int n = Y.n_elem; // number of observation
    const unsigned int L = Fphi.n_elem;
    const double L_ = static_cast<double>(L);

	double et,ft,Qt,alphat,betat,ft_ast,Qt_ast;
    arma::mat Wtill(L,L,arma::fill::zeros);

    bool dflag;
    if (R_IsNA(W) && !R_IsNA(delta)) {
        dflag = true; // use discount factor
    } else if (!R_IsNA(W) && R_IsNA(delta)) {
        dflag = false;
        Wtill.at(0,0) = W; // use the given W
    } else {
        stop("forwardFilter: can only use either W or delta.");
    }

    

    /* 
    --- Reference Analysis --- 
    !!! This part is essential !!!
    */
    arma::vec Fy = arma::reverse(Y.subvec(0,L-1));
    at.col(1) = G * mt.col(0);
    Rt.slice(1) = G * Ct.slice(0) * G.t();
    if (dflag) { // use discount factor
        Rt.slice(1) /= delta;
    } else { // use given W
        Rt.slice(1) += Wtill;
    }
    
    if (!Rt.slice(1).is_finite()) {
        Rcout << "Current time: " << 1 << std::endl;
        Rcout << "Rt: " << Rt.slice(1) << std::endl;
        Rcout << "C[t-1]: " << Ct.slice(0) << std::endl;
        Rcout << "Wtill: " << Wtill << std::endl;
        Rcout << "G: " << G << std::endl;
        stop("Non-finite values for Rt");
    }

    arma::vec Fa(L);
    if (Ftype==0) {
        Fa = arma::exp(at.col(1));
    } else {
        for (unsigned int j=0; j<L; j++) {
            Fa.at(j) = std::max(at.at(j,1),0.);
        }
    }
    arma::vec Ft = Fa % Fphi % Fy;
    if (!Ft.is_finite()) {
        Rcout << "Current time: " << 1 << std::endl;
        Rcout << "at: " << at.col(1).t() << std::endl;
        Rcout << "m[t-1]: " << mt.col(0).t() << std::endl;
        Rcout << "Q[t-1]: " << Qt << ", f[t-1]: " << ft << std::endl;
        stop("Non-finite values for Ft");
    }
    
    ft = arma::accu(Ft);
    Qt = arma::as_scalar(Ft.t()*Rt.slice(1)*Ft);
    if (!std::isfinite(Qt)) {
        Rcout << "Current time: " << 1 << std::endl;
        Rcout << "Q[t]: " << Qt << ", f[t]: " << ft << std::endl;
        stop("Non-finite value for Qt");
    }
    if (Qt<arma::datum::eps) {
        Rcout << "Current time: " << 1 << std::endl;
        Rcout << "Q[t]: " << Qt << std::endl;
        stop("Zero or negative value for Qt");
    }
    betat = ft / Qt;
    alphat = betat*ft;
    ft_ast = (alphat + Y.at(0)) / (betat + 1.);
    Qt_ast = ft_ast / (betat + 1.);
    et = ft_ast - ft;
	mt.col(1) = at.col(1) + Rt.slice(1)*Ft*et/Qt;
    if (!mt.col(1).is_finite()) {
        Rcout << "Current time: " << 1 << std::endl;
        Rcout << "m[t]: " << mt.col(1).t() << std::endl;
        stop("Non-finite values for mt");
    }
    arma::vec Ft2 = Fa % arma::reverse(Fphi) % Fy;
	Ct.slice(1) = Rt.slice(1) - Rt.slice(1)*Ft*Ft2.t()*Rt.slice(1)*(1.-Qt_ast/Qt)/Qt;
    if (!Ct.slice(1).is_finite()) {
        Rcout << "Current time: " << 1 << std::endl;
        Rcout << "C[t]: " << Ct.slice(1) << std::endl;
        stop("Non-finite values for Ct");
    }

    for (unsigned int t=2; t<=L; t++) {
        at.col(t) = at.col(1);
        Rt.slice(t) = Rt.slice(1);
        mt.col(t) = mt.col(1);
        Ct.slice(t) = Ct.slice(1);
    }
    /* 
    --- Reference Analysis --- 
    */

    
	for (unsigned int t=(L+1); t<=n; t++) {
        /*
        Fy - Checked. Correct.
        */
        Fy = arma::reverse(Y.subvec(t-L-1,t-2));
        Fy.elem(arma::find(Fy<=0)).fill(0.01/L_);
 
		// Prior at time t: theta[t] | D[t-1] ~ (at, Rt)
        at.col(t) = G * mt.col(t-1);
        Rt.slice(t) = G * Ct.slice(t-1) * G.t();
        if (dflag) { // use discount factor
            Rt.slice(t) /= delta;
        } else { // use given W
            Rt.slice(t) += Wtill;
        }
        if (!Rt.slice(t).is_finite()) {
            Rcout << "Current time: " << t << std::endl;
            Rcout << "Rt: " << Rt.slice(t) << std::endl;
            Rcout << "C[t-1]: " << Ct.slice(t-1) << std::endl;
            Rcout << "Wtill: " << Wtill << std::endl;
            Rcout << "G: " << G << std::endl;
            stop("Non-finite values for Rt");
        }
		
		// One-step ahead forecast: Y[t]|D[t-1] ~ (ft, Qt)
        // TODO: check this part, the derivatives of Ft()
        // arma::uvec tmpvL = arma::find(Fy<=0);
        
        if (Ftype==0) {
            Fa = arma::exp(at.col(t));
        } else {
            for (unsigned int j=0; j<L; j++) {
                Fa.at(j) = std::max(at.at(j,t),0.);
            }
        }
        Ft = Fa % Fphi % Fy; // L x 1
        if (!Ft.is_finite()) {
            Rcout << "Current time: " << t << std::endl;
            Rcout << "at: " << at.col(t).t() << std::endl;
            Rcout << "m[t-1]: " << mt.col(t-1).t() << std::endl;
            Rcout << "Q[t-1]: " << Qt << ", f[t-1]: " << ft << std::endl;
            stop("Non-finite values for Ft");
        }
		ft = arma::accu(Ft);
		Qt = arma::as_scalar(Ft.t() * Rt.slice(t) * Ft);
        if (!std::isfinite(Qt)) {
            Rcout << "Current time: " << t << std::endl;
            Rcout << "Q[t]: " << Qt << ", f[t]: " << ft << std::endl;
            stop("Non-finite value for Qt");
        }
        if (Qt<arma::datum::eps) {
            // Qt = arma::datum::eps;
            Rcout << "Current time: " << t << std::endl;
            Rcout << "Q[t]: " << Qt << std::endl;
            Rcout << "Ft: " << Ft.t() << std::endl;
            Rcout << "Rt: " << Rt.slice(t) << std::endl;
            stop("Zero or negative value for Qt");
        }

        betat = ft/Qt;
        alphat = betat*ft;

        ft_ast = (alphat+Y.at(t-1)) / (betat +1.);
        Qt_ast = ft_ast / (betat+1.);

		// Posterior at time t: theta[t] | D[t] ~ N(mt, Ct)
		et = ft_ast - ft;
		mt.col(t) = at.col(t) + Rt.slice(t)*Ft*et/Qt;
        if (!mt.col(t).is_finite()) {
            Rcout << "Current time: " << t << std::endl;
            Rcout << "m[t]: " << mt.col(t).t() << std::endl;
            stop("Non-finite values for mt");
        }
        Ft2 = Fa % arma::reverse(Fphi) % Fy;
		Ct.slice(t) = Rt.slice(t) - Rt.slice(t)*Ft*Ft2.t()*Rt.slice(t)*(1.-Qt_ast/Qt)/Qt;
        if (!Ct.slice(t).is_finite()) {
            Rcout << "Current time: " << t << std::endl;
            Rcout << "C[t]: " << Ct.slice(t) << std::endl;
            stop("Non-finite values for Ct");
        }
	}

}



/*
---- Method ----
- Mean-Field Variational Inference
- Ordinary Kalman filtering with all static parameters known
- Using the DLM formulation

---- Model ----
<obs>   y[t] ~ Pois(lambda[t]), t=1,...,n
<link>  lambda[t] = phi[1]*y[t-1]*exp(psi[t]) + ... + phi[L]*y[t-L]*exp(psi[t-L+1]),t=1,...,n
<state> psi[t] = psi[t-1] + omega[t], t=1,...,n
        omega[t] ~ Normal(0,W)

<prior> theta_till[0] ~ Norm(m0, C0)
        W ~ IG(aw,bw)
*/
void backwardSmoother(
	const arma::mat& G, // L x L, state transition matrix
	const arma::mat& mt, // L x (n+1), t=0 is the mean for initial value theta[0]
	const arma::mat& at, // L x (n+1)
	const arma::cube& Ct, // L x L x (n+1), t=0 is the var for initial value theta[0]
	const arma::cube& Rt, // L x L x (n+1)
	arma::vec& ht, // (n+1) x 1, only keep the first element
    arma::vec& Ht, // (n+1) x 1, only keep the variance of the first element
    const double delta = NA_REAL) { 

	// Use the conditional distribution
	const unsigned int n = ht.n_elem - 1; // num of obs
    const unsigned int L = G.n_rows; 
    // double delta;

	ht.at(n) = mt.at(0,n);
    Ht.at(n) = Ct.at(0,0,n);

	arma::mat Bt(L,L);
	for (unsigned int t=(n-1); t>0; t--) {
		// sample from theta[t] | theta[t+1], D[t]
		// Bt = G * Ct.slice(t+1) * G.t();
        // delta = Bt.at(0,0) / Rt.at(0,0,t+2);
        if (R_IsNA(delta)) {
            Bt = Ct.slice(t) * G.t() * Rt.slice(t+1).i();
		    ht.at(t) = mt.at(0,t) + Bt.at(0,0)*(ht.at(t+1)-at.at(0,t+1));
            Ht.at(t) = Ct.at(0,0,t) - Bt.at(0,0)*(Rt.at(0,0,t+1)-Ht.at(t+1))*Bt.at(0,0);
        } else {
            ht.at(t) = mt.at(0,t) + delta/G.at(0,0)*(ht.at(t+1) - at.at(0,t+1));
            Ht.at(t) = Ct.at(0,0,t) - delta/G.at(0,0)*(Rt.at(0,0,t+1)-Ht.at(t+1));
        }
        
	}

	// t = 0
    // delta = Bt.at(0,0) / Rt.at(0,0,2);
    if (R_IsNA(delta)) {
        Bt = Ct.slice(0) * G.t() * Rt.slice(1).i();
        ht.at(0) = mt.at(0,0) + Bt.at(0,0)*(ht.at(1)-at.at(0,1));
        Ht.at(0) = Ct.at(0,0,0) - Bt.at(0,0)*(Rt.at(0,0,1)-Ht.at(1))*Bt.at(0,0);
    } else {
        ht.at(0) = mt.at(0,0) + delta/G.at(0,0)*(ht.at(1) - at.at(0,1));
        Ht.at(0) = Ct.at(0,0,0) - delta/G.at(0,0)*(Rt.at(0,0,1)-Ht.at(1));
    }
    
}




//' @export
// [[Rcpp::export]]
Rcpp::List va_koyama(
    const arma::vec& Y, // n x 1, the observed response
    const unsigned int L = 12,
    const unsigned int niter = 5000, // number of iterations for variational inference
    const double W0 = NA_REAL, 
    const double delta = 0.88,
    const Rcpp::Nullable<Rcpp::NumericVector>& W_init = R_NilValue, // (aw0:shape, bw0:rate)
    const Rcpp::Nullable<Rcpp::NumericVector>& theta0_init = R_NilValue,
    const unsigned int Ftype = 0) { // 0: exp(psi); 1: max(psi,0.)

    const unsigned int n = Y.n_elem;
    const double n_ = static_cast<double>(n);
    const double ssy2 = arma::accu(Y%Y);

    double tmpd; // store temporary double value
    arma::vec tmpv1(1);
    arma::mat tmpmL(L,L);


    /* ----- Hyperparameter and Initialization ----- */
    double aw0, bw0, aw, bw, W;
    bool Wflag = true;
    if (!R_IsNA(W0)) {
        Wflag = false;
        W = W0;
    } else {
        if (!W_init.isNull()) {
            arma::vec W_init_ = Rcpp::as<arma::vec>(W_init);
            aw0 = W_init_.at(0);
            bw0 = W_init_.at(1);
        } else {
            aw0 = 1.e-3;
            bw0 = 1.e-1;
        }
        aw = aw0;
        bw = bw0;
        W = aw/bw;
    }
    

    arma::vec ht(n+1,arma::fill::zeros); // E(theta[t] | Dn), for t=1,...,n
    arma::vec Ht(n+1,arma::fill::ones);
    arma::mat mt(L,n+1,arma::fill::zeros); // m0
    arma::cube Ct(L,L,n+1); 
    Ct.slice(0).eye(); Ct.slice(0)*= 0.1; // C0
    arma::mat at(L,n+1,arma::fill::zeros);
    arma::cube Rt(L,L,n+1);
    
    for (unsigned int t=1; t<=n; t++) {
        Ct.slice(t).eye();
    }
    for (unsigned int t=0; t<=n; t++) {
        Rt.slice(t).eye();
        // Ht.slice(t).eye();
    }

    if (!theta0_init.isNull()) {
        arma::vec theta0_init_ = Rcpp::as<arma::vec>(theta0_init); 
        mt.col(0).fill(theta0_init_.at(0));
        Ct.slice(0).diag().fill(theta0_init_.at(1));
    }


    const double mu = arma::datum::eps;
    const double m = 4.7;
    const double s = 2.9;
    const double sm2 = std::pow(s/m,2);
    const double pk_mu = std::log(m/std::sqrt(1.+sm2));
    const double pk_sg2 = std::log(1.+sm2);
    // const double pk_2sg = std::sqrt(2.*pk_sg2);

    arma::mat G(L,L,arma::fill::zeros);
    arma::vec Fphi(L,arma::fill::zeros);
    G.at(0,0) = 1.;
    Fphi.at(0) = knl(1.,pk_mu,pk_sg2);
    for (unsigned int d=1; d<L; d++) {
        G.at(d,d-1) = 1.;
        tmpd = static_cast<double>(d) + 1.;
        Fphi.at(d) = knl(tmpd,pk_mu,pk_sg2);
    }
    /*
    Fphi - Checked. Correct.
    G - Checked. Correct.
    */

    arma::vec Exx_th(n+1); // E(theta[t]theta[t]' | Dn), for t=0,1,...,n.
    arma::vec Exy_th(n); // E(theta[t-1]theta[t]' | Dn), for t=1,...,n.

    if (R_IsNA(delta)) {
        forwardFilter(Y,G,Fphi,Ftype,mt,at,Ct,Rt,W,NA_REAL); // use W
        backwardSmoother(G,mt,at,Ct,Rt,ht,Ht,NA_REAL);
    } else {
        forwardFilter(Y,G,Fphi,Ftype,mt,at,Ct,Rt,NA_REAL,delta); // use discount factor
        backwardSmoother(G,mt,at,Ct,Rt,ht,Ht,delta);
    }
    

    for (unsigned int j=1; j<=n; j++) {
        Exx_th.at(j) = ht.at(j)*ht.at(j) + Ht.at(j);
        tmpd = Ht.at(j-1) * G.at(0,0);
        // tmpmL = Ht.slice(j-1)*G.t(); // L x L
        Exy_th.at(j-1) = ht.at(j-1)*ht.at(j) + tmpd;
    }
    Exx_th.at(0) = ht.at(0)*ht.at(0) + Ht.at(0);

    /* ----- Hyperparameter and Initialization ----- */


    /* ----- Storage ----- */
    arma::vec W_stored(niter);
    arma::mat ht_stored(n,niter);
    // arma::mat htL_stored(L,niter);
    arma::mat psit_stored(n,niter);
    /* ----- Storage ----- */

    for (unsigned int i=0; i<niter; i++) {
        R_CheckUserInterrupt();

        // TODO: check this part
        if (Wflag) {
            aw = aw0 + 0.5*n_;
            bw = bw0 + 0.5*(arma::accu(Exx_th.tail(n)) + arma::accu(Exx_th.head(n)) - 2.*arma::accu(Exy_th));
            W = aw/bw;
        }


        if (R_IsNA(delta)) {
            forwardFilter(Y,G,Fphi,Ftype,mt,at,Ct,Rt,W,NA_REAL); // use W
            backwardSmoother(G,mt,at,Ct,Rt,ht,Ht,NA_REAL);
        } else {
            forwardFilter(Y,G,Fphi,Ftype,mt,at,Ct,Rt,NA_REAL,delta); // use discount factor
            backwardSmoother(G,mt,at,Ct,Rt,ht,Ht,delta);
        }

        for (unsigned int j=1; j<=n; j++) {
            Exx_th.at(j) = ht.at(j)*ht.at(j) + Ht.at(j);
            // tmpmL = Ht.slice(j-1)*G.t(); // L x L
            tmpd = Ht.at(j-1) * G.at(0,0);
            Exy_th.at(j-1) = ht.at(j-1)*ht.at(j) + tmpd;
        }
        Exx_th.at(0) = ht.at(0)*ht.at(0) + Ht.at(0);

        W_stored.at(i) = W;
        ht_stored.col(i) = ht.tail(n);
        // htL_stored.col(i) = ht.col(L+1);
        psit_stored.col(i) = mt(0,arma::span(1,n)).t();

        Rcout << "\rProgress: " << i+1 << "/" << niter;
    }


    Rcpp::List output;
    output["W"] = Rcpp::wrap(W_stored);
    output["ht"] = Rcpp::wrap(ht_stored);
    // output["htL"] = Rcpp::wrap(htL_stored);
    output["psi"] = Rcpp::wrap(psit_stored);
    return output;
}



/*
---- Method ----
- Bootstrap filtering with static parameter W known
- Using the DLM formulation

---- Model ----
<obs>   y[t] ~ Pois(lambda[t])
<link>  lambda[t] = phi[1]*y[t-1]*exp(psi[t]) + ... + phi[L]*y[t-L]*exp(psi[t-L+1])
<state> psi[t] = psi[t-1] + omega[t]
        omega[t] ~ Normal(0,W)

Unknown parameters: psi[1:n]
Known parameters: W, phi[1:L]
Kwg: Identity link, exp(psi) state space
*/
arma::mat bf_pois_koyama(
    const arma::vec& Y, // n x 1, the observed response
	const double W,
	const unsigned int N = 5000, // number of particles
    const unsigned int B = 30, // step of backshift
    const double rho = 34.08792, // parameter for negative binomial likelihood
    const unsigned int obstype = 0, // 0: negative binomial DLM; 1: poisson DLM
    const unsigned int Ftype = 0){ // 0: exp(psi); 1: max(psi,0.)

    double tmpd; // store temporary double value
    unsigned int tmpi; // store temporary integer value
    arma::vec tmpv1(1);
    
    const unsigned int n = Y.n_elem; // number of observations
    const unsigned int L = 30; // number of lags

    const double mu = arma::datum::eps;
    const double m = 4.7;
    const double s = 2.9;
    const double sm2 = std::pow(s/m,2);
    const double pk_mu = std::log(m/std::sqrt(1.+sm2));
    const double pk_sg2 = std::log(1.+sm2);
    const double pk_2sg = std::sqrt(2.*pk_sg2);

    const double Wsqrt = std::sqrt(W);

    arma::mat G(L,L,arma::fill::zeros);
    arma::vec Fphi(L,arma::fill::zeros);
    G.at(0,0) = 1.;
    tmpv1.at(0) = pk_mu/pk_2sg;
    Fphi.at(0) = 0.5*arma::as_scalar(arma::erfc(tmpv1));
    for (unsigned int d=1; d<L; d++) {
        G.at(d,d-1) = 1.;
        tmpd = static_cast<double>(d) + 1.;
        tmpv1.at(0) = -(std::log(tmpd)-pk_mu)/pk_2sg;
        Fphi.at(d) = 0.5*arma::as_scalar(arma::erfc(tmpv1));
        tmpd -= 1.;
        tmpv1.at(0) = -(std::log(tmpd)-pk_mu)/pk_2sg;
        Fphi.at(d) -= 0.5*arma::as_scalar(arma::erfc(tmpv1));
    }

    // Check Fphi and G - CORRECT
    // Rcout << Fphi.t() << std::endl; 
    // Rcout << G << std::endl;

    arma::vec Fy(L,arma::fill::zeros);
    arma::vec F(L);

    arma::cube theta_stored(L,N,B);
    arma::mat theta = arma::randu(L,N,arma::distr_param(0.,10.));
    arma::vec lambda(N);
    arma::vec omega(N);
    arma::vec w(N); // weight of each particle

    arma::mat R(n,3);
    arma::vec qProb = {0.025,0.5,0.975};

    Rcpp::IntegerVector idx_(N);
    arma::uvec idx(N);
    arma::vec Fa(L);

    for (unsigned int t=0; t<n; t++) {
        theta_stored.slice(B-1) = theta;

        Fy.zeros();
        tmpi = std::min(t,L);
        if (t>0) {
            // Checked Fy - CORRECT
            Fy.head(tmpi) = arma::reverse(Y.subvec(t-tmpi,t-1));
        }
        F = Fphi % Fy; // L x 1

        /*
        1. Resample lambda[t,i] for i=0,1,...,N, where i is the index of particles
        using likelihood P(y[t]|lambda[t,i])
        */
        for (unsigned int i=0; i<N; i++) {
            if (Ftype == 0) {
                Fa = arma::exp(theta.col(i));
            } else {
                for (unsigned int j=0; j<L; j++) {
                    Fa.at(j) = std::max(theta.at(j,i),0.);
                }
            }
            lambda.at(i) = mu + arma::as_scalar(F.t()*Fa);

            if (obstype == 0) {
                w.at(i) = std::exp(R::lgammafn(Y.at(t)+(lambda.at(i)/rho))-R::lgammafn(Y.at(t)+1.)-R::lgammafn(lambda.at(i)/rho)+(lambda.at(i)/rho)*std::log(1./(1.+rho))+Y.at(t)*std::log(rho/(1.+rho)));
            } else if (obstype == 1) {
                w.at(i) = R::dpois(Y.at(t),lambda.at(i),false);
            }
            
        }

        if (arma::accu(w)>0) {
            idx_ = Rcpp::sample(N,N,true,Rcpp::as<Rcpp::NumericVector>(Rcpp::wrap(w)));
            idx = Rcpp::as<arma::uvec>(idx_) - 1;
            for (unsigned int b=0; b<B; b++) {
                theta_stored.slice(b) = theta_stored.slice(b).cols(idx);
            }
        }
        
        // theta_stored = theta_stored(arma::span::all,idx,arma::span::all);
        R.row(t) = arma::quantile(theta_stored.slice(0).row(0),qProb);
        // theta = theta.cols(idx);

        /*
        2. Propagate theta[t,i] to theta[t+1,i] for i=0,1,...,N
        using state evolution distribution P(theta[t+1,i]|theta[t,i])
        */
        omega = arma::randn(N) * Wsqrt;
        for (unsigned int i=0; i<N; i++) {
            theta.col(i) = G * theta_stored.slice(B-1).col(i);
        }
        theta.row(0) += omega.t();
        theta_stored.slices(0,B-2) = theta_stored.slices(1,B-1);
    }

    /*
    Up to now, the first L-1 rows of R will be zeros. We need to
    (1) Shift the nonzero elements to the beginning of R
    (2) calculate the quantiles the first L-1 rows of theta
    */
    R.rows(0,n-L) = R.rows(L-1,n-1);
    for (unsigned int d=0; d<(L-1); d++) {
        R.row(n-L+1+d) = arma::quantile(theta_stored.slice(d).row(0),qProb);
    }

    return R;
}



/*
---- Method ----
- Hybrid variational approximation
- Using the DLM formulation

---- Model ----
<obs>   y[t] ~ Pois(lambda[t])
<link>  lambda[t] = phi[1]*y[t-1]*exp(psi[t]) + ... + phi[L]*y[t-L]*exp(psi[t-L+1])
<state> psi[t] = psi[t-1] + omega[t]
        omega[t] ~ Normal(0,W)

Unknown parameters: psi[1:n], W
Known parameters: phi[1:L]
Kwg: Identity link, exp(psi) state space
*/
//' @export
// [[Rcpp::export]]
Rcpp::List hva_koyama(
    const arma::vec& Y, // n x 1, the observed response
    const unsigned int niter = 1000, // number of iterations for variational inference
    const unsigned int L = 12,
    const double W0 = NA_REAL, 
    const double delta = 0.88,
    const Rcpp::Nullable<Rcpp::NumericVector>& W_init = R_NilValue,
    const unsigned int Ftype = 0) { // 0: exp(psi); 1: max(psi,0.)

    const unsigned int n = Y.n_elem; // number of observations
    const double CLOBND = -100;

    double tmpd; // store temporary double value
    arma::vec tmpv1(1);
    arma::vec tmpvL(L);
    arma::mat tmpmL(L,L);

    double ada_rho = 0.95;
    double ada_eps_step = 1.e-6;
    double oldEdelta2_tau, oldEg2_tau, Eg2_tau, Change_delta_tau, Edelta2_tau, tau;
    double oldEdelta2_mu, oldEg2_mu, Eg2_mu, Change_delta_mu, Edelta2_mu;
    double oldEdelta2_sig, oldEg2_sig, Eg2_sig, Change_delta_sig, Edelta2_sig;  

    arma::vec zeps = arma::randn(niter); // standard normal error used in step 1 for reparameterisation
    arma::vec mu_hyper(niter+1,arma::fill::zeros);
    mu_hyper.at(0) = 0.;
    arma::vec sig_hyper(niter+1,arma::fill::ones);
    sig_hyper.at(0) = 0.5;
    arma::vec gm_hyper(niter+1,arma::fill::ones);
    gm_hyper.at(0) = 1.;
    double cstar; // cstar = t_gamma(c)
    double c;
    double cp; // cp = the first order derivative of c_star=t_gamma(c)
    double grad_q0,grad_g;
    arma::vec grad_elbo(3);

    double W = 0.01;
    double aw, bw;
    bool Wflag = true;
    if (!R_IsNA(W0)) {
        Wflag = false;
        W = W0;
    } else {
        if (!W_init.isNull()) {
            arma::vec W_init_ = Rcpp::as<arma::vec>(W_init);
            aw = W_init_.at(0);
            bw = W_init_.at(1);
            
        } else {
            aw = 1.e-3;
            bw = 1.e-1;
        }
        W = aw/bw;
    }

    arma::vec psi(n+1,arma::fill::zeros);
    
    arma::mat mt(L,n+1,arma::fill::zeros); // m0
    arma::cube Ct(L,L,n+1); 
    Ct.slice(0).eye(); Ct.slice(0)*= 0.1; // C0
    arma::mat at(L,n+1,arma::fill::zeros);
    arma::cube Rt(L,L,n+1);

    arma::mat theta(L,n+1,arma::fill::zeros);
    arma::mat Bt(L,L,arma::fill::eye);
    arma::vec ht(n+1,arma::fill::zeros); // E(theta[t] | Dn), for t=1,...,n
    arma::vec Ht(n+1,arma::fill::ones);
    
    for (unsigned int t=1; t<=n; t++) {
        Ct.slice(t).eye();
    }
    for (unsigned int t=0; t<=n; t++) {
        Rt.slice(t).eye();
        // Ht.slice(t).eye();
    }

    // if (!theta0_init.isNull()) {
    //     arma::vec theta0_init_ = Rcpp::as<arma::vec>(theta0_init); 
    //     mt.col(0).fill(theta0_init_.at(0));
    //     Ct.slice(0).diag().fill(theta0_init_.at(1));
    // }


    arma::mat G(L,L,arma::fill::zeros);
    G.at(0,0) = 1.;
    for (unsigned int d=1; d<L; d++) {
        G.at(d,d-1) = 1.;
    }


    const double mu = arma::datum::eps;
    const double m = 4.7;
    const double s = 2.9;
    const double sm2 = std::pow(s/m,2);
    const double pk_mu = std::log(m/std::sqrt(1.+sm2));
    const double pk_sg2 = std::log(1.+sm2);
    arma::vec Fphi(L,arma::fill::zeros);
    Fphi.at(0) = knl(1.,pk_mu,pk_sg2);
    for (unsigned int d=1; d<L; d++) {
        tmpd = static_cast<double>(d) + 1.;
        Fphi.at(d) = knl(tmpd,pk_mu,pk_sg2);
    }
    /*
    Fphi - Checked. Correct.
    G - Checked. Correct.
    */



    /* ----- Storage ----- */
    arma::vec W_stored(niter);
    arma::mat theta_stored(n,niter);
    /* ----- Storage ----- */

    for (unsigned int i=0; i<niter; i++) {
        R_CheckUserInterrupt();

        /*
        Step 1. Sample hyperparameter via the variational distribution
            N(mu_hyper,sig_hyper^2)
        using the reparameterisation trick
        */
        if (Wflag) {
            cstar = mu_hyper.at(i) + sig_hyper.at(i)*zeps.at(i);
            c = tYJinv_gm(cstar,gm_hyper.at(i));
            // if (c<CLOBND) {
            //     c = CLOBND;
            //     cstar = tYJ_gm(c,gm_hyper.at(i));
            // }
            W = std::exp(-c);
            
        }
        
        if (!std::isfinite(W)) {
            Rcout << "mu: " << mu_hyper.at(i);
            Rcout << " sig: " << sig_hyper.at(i);
            Rcout << " zeps: " << zeps.at(i);
            Rcout << " gm: " << gm_hyper.at(i) << std::endl;
            Rcout << "cstar: " << cstar;
            Rcout << " c: " << c;
            Rcout << " W: " << W << std::endl;
            stop("Non-finite value for W");
        }


        /*
        Step 2. Sample latent state parameters via conditional posteriors
        using Smoothing distributions
        */
        if (R_IsNA(delta)) {
            forwardFilter(Y,G,Fphi,Ftype,mt,at,Ct,Rt,W,NA_REAL); // use W
            backwardSmoother(G,mt,at,Ct,Rt,ht,Ht,NA_REAL);
        } else {
            forwardFilter(Y,G,Fphi,Ftype,mt,at,Ct,Rt,NA_REAL,delta); // use discount factor
            backwardSmoother(G,mt,at,Ct,Rt,ht,Ht,delta);
        }
        psi = ht;

        /*
        Using FFBS
        */
        // ht = mt.col(n);
        // Ht = Ct.slice(n);
        // Ht = arma::symmatu(Ht);
        // arma::eig_sym(tmpvL,tmpmL,Ht);
        // tmpmL = arma::diagmat(arma::sqrt(arma::abs(tmpvL)))*tmpmL.t();
        // theta.col(n) = ht + tmpmL.t()*arma::randn(L);

        // for (unsigned int t=n-1; t>0; t--) {
        //     Bt = Ct.slice(t)*G.t()*Rt.slice(t+1).i();
        //     ht = mt.col(t) + Bt*(theta.col(t+1) - at.col(t+1));
        //     Ht = Ct.slice(t) - Bt*Rt.slice(t+1)*Bt.t();
        //     Ht = arma::symmatu(Ht);
        //     arma::eig_sym(tmpvL,tmpmL,Ht);
        //     tmpmL = arma::diagmat(arma::sqrt(arma::abs(tmpvL)))*tmpmL.t();
        //     theta.col(t) = ht + tmpmL.t()*arma::randn(L);
        // }

        // Bt = Ct.slice(0)*G.t()*Rt.slice(1).i();
        // ht = mt.col(0) + Bt*(theta.col(1) - at.col(1));
        // Ht = Ct.slice(0) - Bt*Rt.slice(1)*Bt.t();
        // Ht = arma::symmatu(Ht);
        // arma::eig_sym(tmpvL,tmpmL,Ht);
        // tmpmL = arma::diagmat(arma::sqrt(arma::abs(tmpvL)))*tmpmL.t();
        // theta.col(0) = ht + tmpmL.t()*arma::randn(L);
        // psi = theta.row(0).t(); // (n+1) x 1
       

        if (Wflag) {
            /*
            Step 3. Compuate gradient of the log variational distribution
            */
            // gradient of the variational distribution
            // grad_elbo = {mu,sig,tau}
            grad_g = dlogJoint_dc(psi,c,aw,bw);
            // grad_g = dlogJoint_dc(psi,c);
            grad_q0 = dlogVB_dc(c,mu_hyper.at(i),sig_hyper.at(i),gm_hyper.at(i));
            grad_elbo = dYJinv(cstar,zeps.at(i),gm_hyper.at(i));
            grad_elbo *= (grad_g - grad_q0);

            /*
            Step 4. Update Variational coefficients
            */
            // mu - mean of the variational distribution,
            // mu takes value along the whole real line
            oldEdelta2_mu = Edelta2_mu;
            oldEg2_mu = Eg2_mu;

            Eg2_mu = ada_rho*oldEg2_mu + (1.-ada_rho)*std::pow(grad_elbo.at(0),2.);
            Change_delta_mu = std::sqrt(oldEdelta2_mu+ada_eps_step)/std::sqrt(Eg2_mu+ada_eps_step)*grad_elbo.at(0);
            if (mu_hyper.at(i)+Change_delta_mu <= CLOBND) {
                Change_delta_mu = 0.;
            }
            mu_hyper.at(i+1) = mu_hyper.at(i) + Change_delta_mu;
            Edelta2_mu = ada_rho*oldEdelta2_mu + (1.-ada_rho)*std::pow(Change_delta_mu,2.);

            // sig - standard deviation of the variational distribution
            oldEdelta2_sig = Edelta2_sig;
            oldEg2_sig = Eg2_sig;

            Eg2_sig = ada_rho*oldEg2_sig + (1.-ada_rho)*std::pow(grad_elbo.at(1),2.);
            Change_delta_sig = std::sqrt(oldEdelta2_sig+ada_eps_step)/std::sqrt(Eg2_sig+ada_eps_step)*grad_elbo.at(1);
            sig_hyper.at(i+1) = sig_hyper.at(i) + Change_delta_sig;
            Edelta2_sig = ada_rho*oldEdelta2_sig + (1.-ada_rho)*std::pow(Change_delta_sig,2.);


            // gamma
            oldEdelta2_tau = Edelta2_tau;
            oldEg2_tau = Eg2_tau;
            tau = gm2tau(gm_hyper.at(i));

            Eg2_tau = ada_rho*oldEg2_tau + (1.-ada_rho)*std::pow(grad_elbo.at(2),2.);
            Change_delta_tau = std::sqrt(oldEdelta2_tau+ada_eps_step)/std::sqrt(Eg2_tau+ada_eps_step)*grad_elbo.at(2);
            tau = tau + Change_delta_tau;
            gm_hyper.at(i+1) = tau2gm(tau);
            if (std::abs(gm_hyper.at(i+1))<1.e-7) {
                gm_hyper.at(i+1) = 1.e-7;
            }
            Edelta2_tau = ada_rho*oldEdelta2_tau + (1.-ada_rho)*std::pow(Change_delta_tau,2.);
        }
        
        
        
        W_stored.at(i) = W;
        theta_stored.col(i) = psi.tail(n);


        Rcout << "\rProgress: " << i+1 << "/" << niter;
    }


    Rcpp::List output;
    output["W"] = Rcpp::wrap(W_stored);
    output["theta"] = Rcpp::wrap(theta_stored);
    output["mu"] = Rcpp::wrap(mu_hyper);
    output["sig"] = Rcpp::wrap(sig_hyper);
    output["gm"] = Rcpp::wrap(gm_hyper);
    return output;
}







