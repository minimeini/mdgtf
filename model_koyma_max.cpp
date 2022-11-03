#include "model_koyama_max.h"

using namespace Rcpp;
// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::depends(RcppArmadillo,nloptr)]]

/*
------------------------
------ Model Code ------
------------------------

0 - (KoyamaMax) Log-normal transmission delay with maximum thresholding on the reproduction number.
1 - (KoyamaExp) Log-normal transmission delay with exponential function on the reproduction number.
2 - (Solow)
*/

/*
------ ModelCode = 0 -------
----------------------------
------ Koyama's Model ------
----------------------------

------ Discretized Hawkes Form ------
<obs> y[t] | lambda[t] ~ Pois(lambda[t])
<link> lambda[t] = phi[1] max(psi[t],0) y[t-1] + phi[2] max(psi[t-1],0) y[t-2] + ... + phi[L] max(psi[t-L+1],0) y[t-L]
<state> psi[t] = psi[t-1] + omega[t], omega[t] ~ N(0,W)


------ Dynamic Linear Model Form ------
<obs> y[t] | lambda[t] = Ft(theta[t]), 
    where Ft(theta[t]) = phi[1] y[t-1] max(theta[t,1],0) + ... + phi[L] y[t-L] max(theta[t,L],0)
<Link> theta[t] = G theta[t-1] + Omega[t], 
    where G = 1 0 ... 0 0
              1 0 ... 0 0
              . .     . .
              .   .   . .
              .     . . .
              0 0 ... 1 0
        
    and Omega[t] = (omega[t],0,...,0) ~ N(0,W[t]), W[t][1,1] = W and 0 otherwise.

-----------------------
------ Inference ------
-----------------------

1. [x] Linear Bayes Approximation with first order Taylor expansion
2. [x] Linear Bayes Approximation with second order Taylor expansion >> doesn't exist because the Hessian will be exactly zero
3. [x] MCMC Disturbance sampler
4. [x] Sequential Monte Carlo filtering and smoothing
5. [x] Vanila variational Bayes
6. [x] Hybrid variational Bayes

*/



/*
Linear Bayes approximation based filtering for koyama's model with exponential on the transmission delay.
- Observational linear approximation - This function uses 1st order Taylor expansion for the nonlinear Ft at the observational equation.
- Known evolution variance - This function assumes the evolution variance, W, is fixed and known
*/
void forwardFilterW(
	const arma::vec& Y, // n x 1, the observation (scalar), n: num of obs
    const arma::mat& G, // L x L, state transition matrix
    const arma::vec& Fphi, // L x 1, state-to-obs nonlinear mapping vector
	const double W, // state variance
	arma::mat& mt, // L x (n+1), t=0 is the mean for initial value theta[0]
	arma::mat& at, // L x (n+1)
	arma::cube& Ct, // L x L x (n+1), t=0 is the var for initial value theta[0]
	arma::cube& Rt) { // L x L x (n+1)

	const unsigned int n = Y.n_elem; // number of observation
    const unsigned int L = Fphi.n_elem;
    const double L_ = static_cast<double>(L);

	double et,ft,Qt,alphat,betat,ft_ast,Qt_ast;
    arma::mat Wtill(L,L,arma::fill::zeros);
    Wtill.at(0,0) = W;

    /* 
    --- Reference Analysis --- 
    !!! This part is essential !!!
    */
    arma::vec Fy = arma::reverse(Y.subvec(0,L-1));
    at.col(1) = G * mt.col(0);
    Rt.slice(1) = G * Ct.slice(0) * G.t() + Wtill;

    at.elem(arma::find(at<arma::datum::eps)).fill(arma::datum::eps); // max(theta[t],0)
    arma::vec Ft = at.col(1) % Fphi % Fy; // Ft(theta[t]) evaluated at theta[t] = a[t]

    ft = arma::accu(Ft);
    Qt = arma::as_scalar(Ft.t()*Rt.slice(1)*Ft);

    betat = ft / Qt;
    alphat = betat*ft;

    ft_ast = (alphat + Y.at(0)) / (betat + 1.);
    Qt_ast = ft_ast / (betat + 1.);

    et = ft_ast - ft;
	mt.col(1) = at.col(1) + Rt.slice(1)*Ft*et/Qt;
    mt.elem(arma::find(mt<arma::datum::eps)).fill(arma::datum::eps); // max(theta[t],0)
    arma::vec Ft2 = at.col(1) % arma::reverse(Fphi) % Fy;
	Ct.slice(1) = Rt.slice(1) - Rt.slice(1)*Ft*Ft2.t()*Rt.slice(1)*(1.-Qt_ast/Qt)/Qt;

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
        Rt.slice(t) = G * Ct.slice(t-1) * G.t() + Wtill;
		
		// One-step ahead forecast: Y[t]|D[t-1] ~ (ft, Qt)
        // TODO: check this part, the derivatives of Ft()
        // arma::uvec tmpvL = arma::find(Fy<=0);
        
        at.elem(arma::find(at<arma::datum::eps)).fill(arma::datum::eps); // max(theta[t],0)
        Ft = at.col(t) % Fphi % Fy; // L x 1
        if (!Ft.is_finite()) {
            Rcout << "Current time: " << t << std::endl;
            Rcout << "at: " << at.col(t).t() << std::endl;
            Rcout << "m[t-1]: " << mt.col(t-1).t() << std::endl;
            stop("Non-finite values for Ft");
        }
		ft = arma::accu(Ft);
		Qt = arma::as_scalar(Ft.t() * Rt.slice(t) * Ft);

        betat = ft/Qt;
        alphat = betat*ft;

        ft_ast = (alphat+Y.at(t-1)) / (betat +1.);
        Qt_ast = ft_ast / (betat+1.);

		// Posterior at time t: theta[t] | D[t] ~ N(mt, Ct)
		et = ft_ast - ft;
		mt.col(t) = at.col(t) + Rt.slice(t)*Ft*et/Qt;
        mt.elem(arma::find(mt<arma::datum::eps)).fill(arma::datum::eps); // max(theta[t],0)
        if (!mt.col(t).is_finite()) {
            Rcout << "Current time: " << t << std::endl;
            Rcout << "m[t]: " << mt.col(t).t() << std::endl;
            stop("Non-finite values for mt");
        }
        Ft2 = at.col(t) % arma::reverse(Fphi) % Fy;
		Ct.slice(t) = Rt.slice(t) - Rt.slice(t)*Ft*Ft2.t()*Rt.slice(t)*(1.-Qt_ast/Qt)/Qt;
	}

}



/*
Linear Bayes approximation based filtering for koyama's model with exponential on the transmission delay.
- Observational linear approximation - This function uses 1st order Taylor expansion for the nonlinear Ft at the observational equation.
- Evolutional discounting - This function assumes the evolution variance is time-varying, which we are not interested in. Instead, 
we preselect a discount factor, delta, to account for the time-varying evolutional variance.
*/
void forwardFilterWt(
	const arma::vec& Y, // n x 1, the observation (scalar), n: num of obs
    const arma::mat& G, // L x L, state transition matrix
    const arma::vec& Fphi, // L x 1, state-to-obs nonlinear mapping vector
	const double delta, // state variance
	arma::mat& mt, // L x (n+1), t=0 is the mean for initial value theta[0]
	arma::mat& at, // L x (n+1)
	arma::cube& Ct, // L x L x (n+1), t=0 is the var for initial value theta[0]
	arma::cube& Rt) { // L x L x (n+1)

	const unsigned int n = Y.n_elem; // number of observation
    const unsigned int L = Fphi.n_elem;
    const double L_ = static_cast<double>(L);

	double et,ft,Qt,alphat,betat,ft_ast,Qt_ast;
	arma::vec At(L);

    /* 
    --- Reference Analysis --- 
    !!! This part is essential !!!
    */
    arma::vec Fy = arma::reverse(Y.subvec(0,L-1)); // L x 1
    at.col(1) = G * mt.col(0);
    Rt.slice(1) = G * Ct.slice(0) * G.t();
	Rt.at(0,0,1) /= delta;

    at.elem(arma::find(at<arma::datum::eps)).fill(arma::datum::eps); // max(theta[t],0)
    arma::vec Ft = at.col(1) % Fphi % Fy; // L x 1
    ft = arma::accu(Ft);
    Qt = arma::as_scalar(Ft.t()*Rt.slice(1)*Ft);

    betat = ft / Qt;
    alphat = betat*ft;
    ft_ast = (alphat + Y.at(0)) / (betat + 1.);
    Qt_ast = ft_ast / (betat + 1.);

	// TODO: check Ft2 -- should we use `arma::reverse(Fphi)` or not?
    et = ft_ast - ft;
	At = Rt.slice(1)*Ft/Qt; // L x 1
	mt.col(1) = at.col(1) + Rt.slice(1)*Ft*et/Qt;
    mt.elem(arma::find(mt<arma::datum::eps)).fill(arma::datum::eps); // max(theta[t],0)
	Ct.slice(1) = Rt.slice(1) - At * (Qt - Qt_ast) * At.t();
    // arma::vec Ft2 = arma::exp(at.col(1)) % arma::reverse(Fphi) % Fy;
	// Ct.slice(1) = Rt.slice(1) - Rt.slice(1)*Ft*Ft2.t()*Rt.slice(1)*(1.-Qt_ast/Qt)/Qt;

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
		Rt.at(0,0,t) /= delta;
		
		// One-step ahead forecast: Y[t]|D[t-1] ~ (ft, Qt)
        // TODO: check this part, the derivatives of Ft()
        // arma::uvec tmpvL = arma::find(Fy<=0);
        at.elem(arma::find(at<arma::datum::eps)).fill(arma::datum::eps); // max(theta[t],0)
        Ft = at.col(t) % Fphi % Fy; // L x 1
        if (!Ft.is_finite()) {
            Rcout << "Current time: " << t << std::endl;
            Rcout << "at: " << at.col(t).t() << std::endl;
            Rcout << "m[t-1]: " << mt.col(t-1).t() << std::endl;
            stop("Non-finite values for Ft");
        }
		ft = arma::accu(Ft);
		Qt = arma::as_scalar(Ft.t() * Rt.slice(t) * Ft);

        betat = ft/Qt;
        alphat = betat*ft;

        ft_ast = (alphat+Y.at(t-1)) / (betat +1.);
        Qt_ast = ft_ast / (betat+1.);

		// Posterior at time t: theta[t] | D[t] ~ N(mt, Ct)
		et = ft_ast - ft;
		At = Rt.slice(t)*Ft/Qt; // L x 1
		mt.col(t) = at.col(t) + Rt.slice(t)*Ft*et/Qt;
        mt.elem(arma::find(mt<arma::datum::eps)).fill(arma::datum::eps); // max(theta[t],0)
        if (!mt.col(t).is_finite()) {
            Rcout << "Current time: " << t << std::endl;
            Rcout << "m[t]: " << mt.col(t).t() << std::endl;
            stop("Non-finite values for mt");
        }
		Ct.slice(t) = Rt.slice(t) - At * (Qt - Qt_ast) * At.t();
        // Ft2 = arma::exp(at.col(t)) % arma::reverse(Fphi) % Fy;
		// Ct.slice(t) = Rt.slice(t) - Rt.slice(t)*Ft*Ft2.t()*Rt.slice(t)*(1.-Qt_ast/Qt)/Qt;
	}

}



/*
---- Method ----
- Koyama's Discretized Hawkes Process
- Ordinary Kalman filtering with all static parameters known
- Using the DLM formulation
- 1st order linear approximation at the observational equation
- Fixed and known evolutional variance

---- Model ----
<obs>   y[t] ~ Pois(lambda[t]), t=1,...,n
<link>  lambda[t] = phi[1]*y[t-1]*exp(psi[t]) + ... + phi[L]*y[t-L]*exp(psi[t-L+1]),t=1,...,n
<state> psi[t] = psi[t-1] + omega[t], t=1,...,n
        omega[t] ~ Normal(0,W)

<prior> theta_till[0] ~ Norm(m0, C0)
        W ~ IG(aw,bw)
*/
void backwardSmootherW(
	const arma::mat& G, // L x L, state transition matrix
	const arma::mat& mt, // L x (n+1), t=0 is the mean for initial value theta[0]
	const arma::mat& at, // L x (n+1)
	const arma::cube& Ct, // L x L x (n+1), t=0 is the var for initial value theta[0]
	const arma::cube& Rt, // L x L x (n+1)
	arma::mat& ht, // L x (n+1)
    arma::cube& Ht) { // L x L x (n+1)

	// Use the conditional distribution
	const unsigned int n = ht.n_cols - 1; // num of obs
    const unsigned int L = G.n_rows; 
    double delta;

	ht.col(n) = mt.col(n);
    ht.elem(arma::find(ht<arma::datum::eps)).fill(arma::datum::eps); // max(theta[t],0)
    Ht.slice(n) = Ct.slice(n);

	arma::mat Bt(L,L);
	for (unsigned int t=(n-1); t>0; t--) {
		// sample from theta[t] | theta[t+1], D[t]
		// Bt = G * Ct.slice(t+1) * G.t();
        // delta = Bt.at(0,0) / Rt.at(0,0,t+2);
        Bt = Ct.slice(t) * G.t() * Rt.slice(t+1).i();
		ht.col(t) = mt.col(t) + Bt*(ht.col(t+1)-at.col(t+1));
        ht.elem(arma::find(ht<arma::datum::eps)).fill(arma::datum::eps); // max(theta[t],0)
        Ht.slice(t) = Ct.slice(t) - Bt*(Rt.slice(t+1)-Ht.slice(t+1))*Bt.t();
	}

	// t = 0
	Bt = Ct.slice(0) * G.t() * Rt.slice(1).i();
    // delta = Bt.at(0,0) / Rt.at(0,0,2);
    ht.col(0) = mt.col(0) + Bt*(ht.col(1)-at.col(1));
    ht.elem(arma::find(ht<arma::datum::eps)).fill(arma::datum::eps); // max(theta[t],0)
    Ht.slice(0) = Ct.slice(0) - Bt*(Rt.slice(1)-Ht.slice(1))*Bt.t();
}



/*
---- Method ----
- Koyama's Discretized Hawkes Process
- Ordinary Kalman filtering with all static parameters known
- Using the DLM formulation
- 1st order linear approximation at the observational equation
- Discounting for the time-varying and unknown evolutional variance

---- Model ----
<obs>   y[t] ~ Pois(lambda[t]), t=1,...,n
<link>  lambda[t] = phi[1]*y[t-1]*exp(psi[t]) + ... + phi[L]*y[t-L]*exp(psi[t-L+1]),t=1,...,n
<state> psi[t] = psi[t-1] + omega[t], t=1,...,n
        omega[t] ~ Normal(0,W)

<prior> theta_till[0] ~ Norm(m0, C0)
        W ~ IG(aw,bw)
*/
void backwardSmootherWt(
	const arma::mat& G, // L x L, state transition matrix
	const arma::mat& mt, // L x (n+1), t=0 is the mean for initial value theta[0]
	const arma::mat& at, // L x (n+1)
	const arma::cube& Ct, // L x L x (n+1), t=0 is the var for initial value theta[0]
	const arma::cube& Rt, // L x L x (n+1)
	const double delta, // discount factor
	arma::vec& ht, // (n+1) x 1
    arma::vec& Ht) { // (n+1) x 1

	// Use the conditional distribution
	const unsigned int n = ht.n_elem - 1; // num of obs
    const unsigned int L = G.n_rows; 

	ht.at(n) = mt.at(0,n);
    ht.elem(arma::find(ht<arma::datum::eps)).fill(arma::datum::eps); // max(theta[t],0)
    Ht.at(n) = Ct.at(0,0,n);

	for (unsigned int t=(n-1); t>0; t--) {
		// sample from theta[t] | theta[t+1], D[t]
		ht.at(t) = (1.-delta) * mt.at(0,t) + delta*ht.at(t+1);
        ht.elem(arma::find(ht<arma::datum::eps)).fill(arma::datum::eps); // max(theta[t],0)
        Ht.at(t) = (1.-delta) * Ct.at(0,0,t);
	}

	// t = 0
    ht.at(0) = (1.-delta)*mt.at(0,0) + delta*ht.at(1);
    ht.elem(arma::find(ht<arma::datum::eps)).fill(arma::datum::eps); // max(theta[t],0)
    Ht.at(0) = (1.-delta)*Ct.at(0,0,0);
}



/*
---- Method ----
- Koyama's Discretized Hawkes Process
- Ordinary Kalman filtering and smoothing
- with all static parameters known
- Using the DLM formulation

---- Model ----
<obs>   y[t] ~ Pois(lambda[t]), t=1,...,n
<link>  lambda[t] = phi[1]*y[t-1]*max(psi[t],0) + ... + phi[L]*y[t-L]*max(psi[t-L+1],0),t=1,...,n
<state> psi[t] = psi[t-1] + omega[t], t=1,...,n
        omega[t] ~ Normal(0,W)

<prior> theta_till[0] ~ Norm(m0, C0)
        W ~ IG(aw,bw)
*/
//' @export
// [[Rcpp::export]]
Rcpp::List lbe_pois_koyama_max_W(
	const arma::vec& Y, // n x 1, the observed response
    const unsigned int L = 12,
    const double W = 0.01, 
    const Rcpp::Nullable<Rcpp::NumericVector>& theta0_init = R_NilValue) {  // systematic variance

	const unsigned int n = Y.n_elem;
    const double n_ = static_cast<double>(n);
    const double ssy2 = arma::accu(Y%Y);

	double tmpd; // store temporary double value
    arma::vec tmpv1(1);
    arma::mat tmpmL(L,L);

	arma::mat ht(L,n+1,arma::fill::zeros); // E(theta[t] | Dn), for t=1,...,n
    arma::cube Ht(L,L,n+1);
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
    }
    for (unsigned int t=0; t<=n; t++) {
        Ht.slice(t).eye();
    }
    if (!theta0_init.isNull()) {
        arma::vec theta0_init_ = Rcpp::as<arma::vec>(theta0_init); 
        mt.col(0).fill(theta0_init_.at(0));
        Ct.slice(0).diag().fill(theta0_init_.at(1));
    }


    arma::vec Fphi = get_Fphi(L);
    arma::mat G(L,L,arma::fill::zeros);
    G.at(0,0) = 1.;
    for (unsigned int d=1; d<L; d++) {
        G.at(d,d-1) = 1.;
    }
    /*
    Fphi - Checked. Correct.
    G - Checked. Correct.
    */

    forwardFilterW(Y,G,Fphi,W,mt,at,Ct,Rt);
    if (!mt.is_finite() || !at.is_finite() || !Ct.is_finite() || !Rt.is_finite()) {
        stop("Non-finite values for filtering.");
    }
    backwardSmootherW(G,mt,at,Ct,Rt,ht,Ht);
    if (!ht.is_finite() || !Ht.is_finite()) {
        stop("Non-finite values for filtering.");
    }

	Rcpp::List output;
	output["mt"] = Rcpp::wrap(mt);
	output["Ct"] = Rcpp::wrap(Ct);
	output["ht"] = Rcpp::wrap(ht);
	output["Ht"] = Rcpp::wrap(Ht);

	return output;
}



/*
---- Method ----
- Koyama's Discretized Hawkes Process
- Ordinary Kalman filtering and smoothing
- with all static parameters known
- Using the DLM formulation

---- Model ----
<obs>   y[t] ~ Pois(lambda[t]), t=1,...,n
<link>  lambda[t] = phi[1]*y[t-1]*exp(psi[t]) + ... + phi[L]*y[t-L]*exp(psi[t-L+1]),t=1,...,n
<state> psi[t] = psi[t-1] + omega[t], t=1,...,n
        omega[t] ~ Normal(0,Wt)

<prior> theta_till[0] ~ Norm(m0, C0)
        W[0] ~ IG(aw,bw)
*/
//' @export
// [[Rcpp::export]]
Rcpp::List lbe_pois_koyama_max_Wt(
	const arma::vec& Y, // n x 1, the observed response
    const unsigned int L = 12,
    const double delta = 0.9, 
    const Rcpp::Nullable<Rcpp::NumericVector>& theta0_init = R_NilValue) {  // systematic variance

	const unsigned int n = Y.n_elem;
    const double n_ = static_cast<double>(n);
    const double ssy2 = arma::accu(Y%Y);

	double tmpd; // store temporary double value
    arma::vec tmpv1(1);
    arma::mat tmpmL(L,L);

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
    }
    if (!theta0_init.isNull()) {
        arma::vec theta0_init_ = Rcpp::as<arma::vec>(theta0_init); 
        mt.col(0).fill(theta0_init_.at(0));
        Ct.slice(0).diag().fill(theta0_init_.at(1));
    }


    arma::vec Fphi = get_Fphi(L);
    arma::mat G(L,L,arma::fill::zeros);
    G.at(0,0) = 1.;
    for (unsigned int d=1; d<L; d++) {
        G.at(d,d-1) = 1.;
    }
    /*
    Fphi - Checked. Correct.
    G - Checked. Correct.
    */

    forwardFilterWt(Y,G,Fphi,delta,mt,at,Ct,Rt);
    if (!mt.is_finite() || !at.is_finite() || !Ct.is_finite() || !Rt.is_finite()) {
        stop("Non-finite values for filtering.");
    }
    backwardSmootherWt(G,mt,at,Ct,Rt,delta,ht,Ht);
    if (!ht.is_finite() || !Ht.is_finite()) {
        stop("Non-finite values for filtering.");
    }

	Rcpp::List output;
	output["mt"] = Rcpp::wrap(mt);
	output["Ct"] = Rcpp::wrap(Ct);
	output["ht"] = Rcpp::wrap(ht);
	output["Ht"] = Rcpp::wrap(Ht);

	return output;
}




/*
---- Method ----
- Monte Carlo Filtering with static parameter W known
- Using the discretized Hawkes formulation
- Reference: Kitagawa and Sato at Ch 9.3.4 of Doucet et al.
- Note: 
    1. Initial version is copied from the `bf_pois_koyama_exp`
    2. This is intended to be the modified Rcpp version of `hawkes_state_space.R`
    3. The difference is that the R version the states are backshifted L times, 
        where L is the maximum transmission delay to be considered.
        To make this a Monte Carlo filtering, we don't backshifting/smoot.

---- Algorithm ----

1. Generate a random number psi[0](j) ~ an initial distribution.
2. Repeat the following steps for t = 1,...,n:
    2-1. Generate a random number omega[t] ~ Normal(0,W)


---- Model ----
<obs>   y[t] ~ Pois(lambda[t])
<link>  lambda[t] = phi[1]*y[t-1]*exp(psi[t]) + ... + phi[L]*y[t-L]*exp(psi[t-L+1])
<state> psi[t] = psi[t-1] + omega[t]
        omega[t] ~ Normal(0,W)

Unknown parameters: psi[1:n]
Known parameters: W, phi[1:L]
Kwg: Identity link, exp(psi) state space
*/
//' @export
// [[Rcpp::export]]
arma::mat mcf_pois_koyama_max_W(
    const arma::vec& Y, // n x 1, the observed response
	const double W,
    const unsigned int L = 12, // number of lags
	const unsigned int N = 5000, // number of particles
    const double rho = 34.08792, // parameter for negative binomial likelihood
    const unsigned int obstype = 0){ // 0: negative binomial DLM; 1: poisson DLM

    double tmpd; // store temporary double value
    unsigned int tmpi; // store temporary integer value
    arma::vec tmpv1(1);
    
    const unsigned int n = Y.n_elem; // number of observations
    const double Wsqrt = std::sqrt(W);

    arma::mat G(L,L,arma::fill::zeros); // L x L state transition matrix
    G.at(0,0) = 1.;
    for (unsigned int d=1; d<L; d++) {
        G.at(d,d-1) = 1.;
    }
    arma::vec Fphi = get_Fphi(L);

    // Check Fphi and G - CORRECT
    // Rcout << Fphi.t() << std::endl; 
    // Rcout << G << std::endl;

    arma::vec Fy(L,arma::fill::zeros);
    arma::vec F(L);

    arma::cube theta_stored(L,N,n+1);
    arma::mat theta = arma::randu(L,N,arma::distr_param(0.,10.));
    
    const double mu = arma::datum::eps;
    arma::vec lambda(N);
    arma::vec omega(N);
    arma::vec w(N); // weight of each particle

    arma::mat R(n,3);
    arma::vec qProb = {0.025,0.5,0.975};

    Rcpp::IntegerVector idx_(N);
    arma::uvec idx(N);

    for (unsigned int t=0; t<=n; t++) {
        // theta_stored: L x N x B, with B = n
        // theta: L x N
        theta_stored.slice(t) = theta;

        Fy.zeros();
        tmpi = std::min(t,L);
        if (t>0) {
            // Checked Fy - CORRECT
            Fy.head(tmpi) = arma::reverse(Y.subvec(t-tmpi,t-1));
        }
        F = Fphi % Fy; // L x 1

        if (t>0) {
            /*
            ------ RESAMPLING STAGE ------
            1. Resample lambda[t,i] for i=0,1,...,N, where i is the index of particles
            using likelihood P(y[t]|lambda[t,i]).
            ------
            >> Step 2-3 and 2-4 of Kitagawa and Sato - This is also the observational equation (nonlinear, nongaussian)
            */
            for (unsigned int i=0; i<N; i++) {
                /*
                ------ RESAMPLING STAGE ------
                ------ Step 2-3 of Kitagawa and Sato ------
                Namely, Compute the weights using likelihood/observational distribution, 
                p(y[t]|lambda[t,i]), 
                where lambda[t,i] is the i-th particle for lambda[t].
                */ 
                theta.elem(arma::find(theta<arma::datum::eps)).fill(arma::datum::eps);
                lambda.at(i) = mu + arma::as_scalar(F.t()*theta.col(i));
                
                if (obstype == 0) {
                    w.at(i) = std::exp(R::lgammafn(Y.at(t)+(lambda.at(i)/rho))-R::lgammafn(Y.at(t)+1.)-R::lgammafn(lambda.at(i)/rho)+(lambda.at(i)/rho)*std::log(1./(1.+rho))+Y.at(t)*std::log(rho/(1.+rho)));
                } else if (obstype == 1) {
                    w.at(i) = R::dpois(Y.at(t),lambda.at(i),false);
                }   
            }

            /*
            ------ RESAMPLING STAGE (FILTERING) ------
            Generate {lambda[1,i],...,lambda[t-1,i],lambda[t,i] | D[t]} for i = 1,...,N
            by resampling {lambda[i,i],...,lambda[t-1,i],lambda[t,i] | D[t-1]} 
            for i = 1,...,N, using the weights w[t,i] = p(y[t]|lambda]t,i).
            */
            idx_ = Rcpp::sample(N,N,true,Rcpp::as<Rcpp::NumericVector>(Rcpp::wrap(w)));
            idx = Rcpp::as<arma::uvec>(idx_) - 1;
            theta_stored.slice(t) = theta_stored.slice(t).cols(idx);
            R.row(t-1) = arma::quantile(theta_stored.slice(t).row(0),qProb);
        }
        

        /*
        2. Propagate theta[t,i] to theta[t+1,i] for i=0,1,...,N
        using state evolution distribution P(theta[t+1,i]|theta[t,i]).
        ------
        >> Step 2-1 and 2-2 of Kitagawa and Sato - This is also the state equation
        */

        /*
        ------ Step 2-1 of Kitagawa and Sato ------
        Generate N random error following the error distribution
        */
        omega = arma::randn(N) * Wsqrt;

        /*
        ------ Step 2-2 of Kitagawa and Sato ------
        Propagate from t to t+1 using the evolution distribution
        */
        for (unsigned int i=0; i<N; i++) {
            theta.col(i) = G * theta_stored.slice(t).col(i);
        }
        theta.row(0) += omega.t();
        theta.elem(arma::find(theta<arma::datum::eps)).fill(arma::datum::eps);
        // theta_stored.slices(0,B-2) = theta_stored.slices(1,B-1);
    }

    /*
    Up to now, the first L-1 rows of R will be zeros. We need to
    (1) Shift the nonzero elements to the beginning of R
    (2) calculate the quantiles the first L-1 rows of theta
    */
    return R;
}



/*
---- Method ----
- Monte Carlo Smoothing with static parameter W known
- B-lag fixed-lag smoother (Anderson and Moore 1979)
- Using the discretized Hawkes formulation
- Reference: 
    Kitagawa and Sato at Ch 9.3.4 of Doucet et al.
    Check out Algorithm step 2-4L for the explanation of B-lag fixed-lag smoother
- Note: 
    1. Initial version is copied from the `bf_pois_koyama_exp`
    2. This is intended to be the Rcpp version of `hawkes_state_space.R`
    3. The difference is that the R version the states are backshifted L times, where L is the maximum transmission delay to be considered.
        To make this a Monte Carlo smoothing, we need to resample the states n times, where n is the total number of temporal observations.

---- Algorithm ----

1. Generate a random number psi[0](j) ~ an initial distribution.
2. Repeat the following steps for t = 1,...,n:
    2-1. Generate a random number omega[t] ~ Normal(0,W)


---- Model ----
<obs>   y[t] ~ Pois(lambda[t])
<link>  lambda[t] = phi[1]*y[t-1]*exp(psi[t]) + ... + phi[L]*y[t-L]*exp(psi[t-L+1])
<state> psi[t] = psi[t-1] + omega[t]
        omega[t] ~ Normal(0,W)

Unknown parameters: psi[1:n]
Known parameters: W, phi[1:L]
Kwg: Identity link, exp(psi) state space
*/
//' @export
// [[Rcpp::export]]
arma::mat mcs_pois_koyama_max_W(
    const arma::vec& Y, // n x 1, the observed response
	const double W,
    const unsigned int L = 12, // number of lags
    const unsigned int B = 12, // length of the B-lag fixed-lag smoother (Anderson and Moore 1979; Kitagawa and Sato)
	const unsigned int N = 5000, // number of particles
    const double rho = 34.08792, // parameter for negative binomial likelihood
    const unsigned int obstype = 1){ // 0: negative binomial DLM; 1: poisson DLM

    double tmpd; // store temporary double value
    unsigned int tmpi; // store temporary integer value
    arma::vec tmpv1(1);
    
    const unsigned int n = Y.n_elem; // number of observations
    const double Wsqrt = std::sqrt(W);
    arma::vec Fphi = get_Fphi(L);
    arma::mat G(L,L,arma::fill::zeros); // L x L state transition matrix
    G.at(0,0) = 1.;
    for (unsigned int d=1; d<L; d++) {
        G.at(d,d-1) = 1.;
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
    const double mu = arma::datum::eps;

    arma::mat R(n,3);
    arma::vec qProb = {0.025,0.5,0.975};

    Rcpp::IntegerVector idx_(N);
    arma::uvec idx(N);

    for (unsigned int t=0; t<n; t++) {
        // theta_stored: L x N x B, with B is the lag of the B-lag fixed-lag smoother
        // theta: L x N
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
        using likelihood P(y[t]|lambda[t,i]).
        ------
        >> Step 2-3 and 2-4 of Kitagawa and Sato - This is also the observational equation (nonlinear, nongaussian)
        */
        for (unsigned int i=0; i<N; i++) {
            /*
            ------ RESAMPLING STAGE ------
            ------ Step 2-3 of Kitagawa and Sato ------
            Namely, Compute the weights using likelihood/observational distribution, 
            p(y[t]|lambda[t,i]), 
            where lambda[t,i] is the i-th particle for lambda[t].
            */ 
            theta.elem(arma::find(theta<arma::datum::eps)).fill(arma::datum::eps);
            lambda.at(i) = mu + arma::as_scalar(F.t()*theta.col(i));
            if (obstype == 0) {
                // negative binomial observational equation
                w.at(i) = std::exp(R::lgammafn(Y.at(t)+(lambda.at(i)/rho))-R::lgammafn(Y.at(t)+1.)-R::lgammafn(lambda.at(i)/rho)+(lambda.at(i)/rho)*std::log(1./(1.+rho))+Y.at(t)*std::log(rho/(1.+rho)));
            } else if (obstype == 1) {
                // poisson observational equation
                w.at(i) = R::dpois(Y.at(t),lambda.at(i),false);
            }   
        }

        /*
        ------ RESAMPLING STAGE (SMOOTHING) ------
        ------ Step 2-4L of Kitagawa and Sato ------
        Generate {lambda[t-L,i],...,lambda[t-1,i],lambda[t,i] | D[t]} for i = 1,...,N
        by resampling {lambda[t-L,i],...,lambda[t-1,i],particle[t,i] | D[t-1]} 
        for i = 1,...,N, using the weights w[t,i] = p(y[t]|particle[t,i]).
        */
        idx_ = Rcpp::sample(N,N,true,Rcpp::as<Rcpp::NumericVector>(Rcpp::wrap(w)));
        idx = Rcpp::as<arma::uvec>(idx_) - 1;
        for (unsigned int b=0; b<B; b++) {
            theta_stored.slice(b) = theta_stored.slice(b).cols(idx);
        }
        R.row(t) = arma::quantile(theta_stored.slice(0).row(0),qProb);

        /*
        2. Propagate theta[t,i] to theta[t+1,i] for i=0,1,...,N
        using state evolution distribution P(theta[t+1,i]|theta[t,i]).
        ------
        >> Step 2-1 and 2-2 of Kitagawa and Sato - This is also the state equation
        */

        /*
        ------ Step 2-1 of Kitagawa and Sato ------
        Generate N random error following the error distribution
        */
        omega = arma::randn(N) * Wsqrt;

        /*
        ------ Step 2-2 of Kitagawa and Sato ------
        Propagate from t to t+1 using the evolution distribution
        */
        for (unsigned int i=0; i<N; i++) {
            theta.col(i) = G * theta_stored.slice(B-1).col(i);
        }
        theta.row(0) += omega.t();
        theta.elem(arma::find(theta<arma::datum::eps)).fill(arma::datum::eps);
        theta_stored.slices(0,B-2) = theta_stored.slices(1,B-1);
    }

    /*
    ------ R: an n x 3 matrix ------
    Up to now, the first L-1 rows of R will be zeros. We need to
    (1) Shift the nonzero elements to the beginning of R
    (2) calculate the quantiles the first L-1 rows of theta
    */
    R.rows(0,n-L) = R.rows(L-1,n-1);
    for (unsigned int b=0; b<L; b++) {
        R.row(b) = arma::quantile(theta_stored.slice(b).row(0),qProb);
    }

    return R;
}




//' @export
// [[Rcpp::export]]
Rcpp::List mfva_koyama_max_Wt(
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
    arma::mat htMat(L,n+1,arma::fill::zeros);
    arma::cube HtCube(L,L,n+1);

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

    // const double pk_2sg = std::sqrt(2.*pk_sg2);
    arma::vec Fphi = get_Fphi(L);
    arma::mat G(L,L,arma::fill::zeros);
    G.at(0,0) = 1.;
    for (unsigned int d=1; d<L; d++) {
        G.at(d,d-1) = 1.;
    }
    /*
    Fphi - Checked. Correct.
    G - Checked. Correct.
    */

    arma::vec Exx_th(n+1); // E(theta[t]theta[t]' | Dn), for t=0,1,...,n.
    arma::vec Exy_th(n); // E(theta[t-1]theta[t]' | Dn), for t=1,...,n.

    if (R_IsNA(delta)) {
        forwardFilterW(Y,G,Fphi,W,mt,at,Ct,Rt); // use W
        backwardSmootherW(G,mt,at,Ct,Rt,htMat,HtCube);
        ht = htMat.row(0).t();
        Ht = HtCube.tube(0,0);
    } else {
        forwardFilterWt(Y,G,Fphi,delta,mt,at,Ct,Rt); // use discount factor
        backwardSmootherWt(G,mt,at,Ct,Rt,delta,ht,Ht);
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
            forwardFilterW(Y,G,Fphi,W,mt,at,Ct,Rt); // use W
            backwardSmootherW(G,mt,at,Ct,Rt,htMat,HtCube);
            ht = htMat.row(0).t();
            Ht = HtCube.tube(0,0);
        } else {
            forwardFilterWt(Y,G,Fphi,delta,mt,at,Ct,Rt); // use discount factor
            backwardSmootherWt(G,mt,at,Ct,Rt,delta,ht,Ht);
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
- Hybrid variational approximation
- Using the DLM formulation

---- Model ----
<obs>   y[t] ~ Pois(lambda[t])
<link>  lambda[t] = phi[1]*y[t-1]*max(psi[t],0) + ... + phi[L]*y[t-L]*max(psi[t-L+1],0)
<state> psi[t] = psi[t-1] + omega[t]
        omega[t] ~ Normal(0,W)

Unknown parameters: psi[1:n], W
Known parameters: phi[1:L]
Kwg: Identity link, exp(psi) state space
*/
//' @export
// [[Rcpp::export]]
Rcpp::List hva_koyama_max_W(
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
    arma::mat htMat(L,n+1,arma::fill::zeros);
    arma::cube HtCube(L,L,n+1);
    
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
    arma::vec Fphi = get_Fphi(L);
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
            forwardFilterW(Y,G,Fphi,W,mt,at,Ct,Rt); // use W
            backwardSmootherW(G,mt,at,Ct,Rt,htMat,HtCube);
            ht = htMat.row(0).t();
            Ht = HtCube.tube(0,0);
        } else {
            forwardFilterWt(Y,G,Fphi,delta,mt,at,Ct,Rt); // use discount factor
            backwardSmootherWt(G,mt,at,Ct,Rt,delta,ht,Ht);
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







/*
Reference: Gamerman (1998); Alves et al. (2010).
ModelCode = 0
*/
//' @export
// [[Rcpp::export]]
Rcpp::List mcmc_disturbance_pois_koyama_max(
	const arma::vec& Y, // n x 1, the observed response
    const unsigned int L = 12, // lag of transmission delay
    const Rcpp::Nullable<Rcpp::NumericVector>& lambda0_in = R_NilValue, // baseline intensity
	const Rcpp::Nullable<Rcpp::NumericVector>& WPrior = R_NilValue, // (nv,Sv), prior for W~IG(shape=nv/2,rate=nv*Sv/2), W is the evolution error variance
	const Rcpp::Nullable<Rcpp::NumericVector>& w1Prior = R_NilValue, // (aw, Rw), w1 ~ N(aw, Rw), prior for w[1], the first state/evolution/state error/disturbance.
	const double W_init = NA_REAL,
	const Rcpp::Nullable<Rcpp::NumericVector>& wt_init = R_NilValue, // n x 1
	const double W_true = NA_REAL, // true value of state/evolution error variance
	const Rcpp::Nullable<Rcpp::NumericVector>& wt_true = R_NilValue, // n x 1, true value of system/evolution/state error/disturbance
	const unsigned int nburnin = 0,
	const unsigned int nthin = 1,
	const unsigned int nsample = 1) {

    const unsigned int ModelCode = 0;
	const double EBOUND = 700.;

	const unsigned int n = Y.n_elem;
	const double n_ = static_cast<double>(n);
	const unsigned int ntotal = nburnin + nthin*nsample + 1;

	// Hyperparameter
    // Evolution variance
	double W,nv,nSv,nv_new,nSv_new;
	arma::vec W_stored(nsample);
	bool Wflag;
	if (R_IsNA(W_true)) {
		Wflag = true;
		if (WPrior.isNull()) {
			stop("Error: You must provide either true value or prior for W, the evolution variance.");
		}
		arma::vec WPrior_ = Rcpp::as<arma::vec>(WPrior);
		nv = WPrior_.at(0);
		nSv = nv*WPrior_.at(1);
		nv_new = nv + n_ - 1.;
		if (R_IsNA(W_init)) {
			W = 1./R::rgamma(0.5*nv,2./nSv);
		} else {
			W = W_init;
		}
		
	} else {
		Wflag = false;
		W = W_true;
	}
	double Wsqrt = std::sqrt(W);


	bool wtflag = true;
	double aw,Rw,Rwsqrt;
	arma::vec wt(n); 
	arma::vec wt_accept(n);
	arma::mat w_stored(n,nsample,arma::fill::zeros);
	if (!wt_true.isNull()) {
		wtflag = false;
		wt = Rcpp::as<arma::vec>(wt_true);
	} else {
		wtflag = true;
		if (!w1Prior.isNull()) {
			arma::vec w1Prior_ = Rcpp::as<arma::vec>(w1Prior);
			aw = w1Prior_.at(0);
			Rw = w1Prior_.at(1);
			Rwsqrt = std::sqrt(Rw);
		}
		if (!wt_init.isNull()) {
			wt = Rcpp::as<arma::vec>(wt_init);
		} else {
			wt = aw + arma::randn(n)*std::sqrt(Rw);
		}
	}
	
    arma::vec Fphi = get_Fphi(L);
	arma::mat Fx = update_Fx(n,L,Fphi,Y,ModelCode);

    double bt,Bt,Btsqrt;
    arma::vec theta(n);
	arma::vec lambda(n);
	arma::vec Ytilde(n);
	arma::vec Yhat(n);

    arma::vec lambda0(n,arma::fill::zeros);
    if (!lambda0_in.isNull()) {
        lambda0 = Rcpp::as<arma::vec>(lambda0_in);
    }

	double wt_old, wt_new;
	double logp_old,logp_new,logq_old,logq_new,logratio;
	
	bool saveiter;

	for (unsigned int b=0; b<ntotal; b++) {
		R_CheckUserInterrupt();
		saveiter = b > nburnin && ((b-nburnin-1)%nthin==0);

		// [OK] Update evolution/state disturbances/errors, denoted by wt.
		for (unsigned int t=0; t<n; t++) {
            // Metroplis-Hastings for each of the wt
			if (!wtflag) { break; }

			/* Part 1. Target full conditional based on old value */
			wt_old = wt.at(t);
            theta = update_theta(n,Fx,wt,ModelCode);
            lambda = update_lambda(theta, lambda0, ModelCode, true);

			logp_old = 0.;
			for (unsigned int j=t;j<n;j++) {
				logp_old += R::dpois(Y.at(j),lambda.at(j),true);
			}
			if (t==0) {
				logp_old += R::dnorm(wt_old,aw,Rwsqrt,true);
			} else {
				logp_old += R::dnorm(wt_old,0.,Wsqrt,true);
			}
			/* Part 1. Checked - OK */

			/* Part 2. Proposal new | old */
			// Ytilde = Et + (Y - lambda) / lambda; // Linearlisation
		    Yhat = Y - theta + Fx.col(t)*wt_old; // TODO - CHECK HERE
		    if (t==0) {
        	    Bt = 1. / (1./Rw + arma::accu(arma::pow(Fx.col(t),2.) % lambda));
        	    bt = Bt * (aw/Rw + arma::accu(Fx.col(t)%Yhat%lambda));
      	    } else {
      		    Bt = 1./(1./W + arma::accu(arma::pow(Fx.col(t),2.) % lambda));
        	    bt = Bt * (arma::accu(Fx.col(t)%Yhat%lambda));
      	    }
			Btsqrt = std::sqrt(Bt);
		    wt_new = R::rnorm(bt, Btsqrt);
			if (!std::isfinite(wt_new)) {
				Rcout << "(bt=" << bt << ", Bt=" << Bt << ")" << std::endl;
				Rcout << "wt: " << wt.t() << std::endl;
				Rcout << "Yhat: " << Yhat.t() << std::endl;
				Rcout << "Et: " << theta.t() << std::endl;
				Rcout << "lambda: " << lambda.t() << std::endl;
				stop("Non finite vt");
			}
			logq_new = R::dnorm(wt_new,bt,Btsqrt,true);
			/* Part 2. */

			wt.at(t) = wt_new;
            theta = update_theta(n,Fx,wt,ModelCode);
            lambda = update_lambda(theta, lambda0, ModelCode, true);
			logp_new = 0.;
			for (unsigned int j=t;j<n;j++) {
				logp_new += R::dpois(Y.at(j),lambda.at(j),true);
			}
			if (t==0) {
				logp_new += R::dnorm(wt_new,aw,Rwsqrt,true);
			} else {
				logp_new += R::dnorm(wt_new,0.,Wsqrt,true);
			}

			// Ytilde = Et + (Y - lambda) / lambda; // linearlisation: use normal to approximate poisson
		    Yhat = Y - theta + Fx.col(t)*wt_new;
		    if (t==0) {
        	    Bt = 1. / (1./Rw + arma::accu(arma::pow(Fx.col(t),2.)%lambda));
        	    bt = Bt * (aw/Rw + arma::accu(Fx.col(t)%Yhat%lambda));
      	    } else {
      		    Bt = 1./(1./W + arma::accu(arma::pow(Fx.col(t),2.)%lambda));
        	    bt = Bt * (arma::accu(Fx.col(t)%Yhat%lambda));
      	    }
			Btsqrt = std::sqrt(Bt);
			logq_old = R::dnorm(wt_old,bt,Btsqrt,true);

			logratio = std::min(0.,logp_new-logp_old+logq_old-logq_new);
			if (std::log(R::runif(0.,1.)) >= logratio) { // reject
				wt.at(t) = wt_old;
			} else {
				wt_accept.at(t) += 1.;
			}
	    }


		// [OK] Update state/evolution error variance
		if (Wflag) {
			nSv_new = nSv + arma::accu(arma::pow(wt.tail(n-1),2.));
			W = 1./R::rgamma(0.5*nv_new,2./nSv_new);
		}
		
		
		// store samples after burnin and thinning
		if (saveiter || b==(ntotal-1)) {
			unsigned int idx_run;
			if (saveiter) {
				idx_run = (b-nburnin-1)/nthin;
			} else {
				idx_run = nsample - 1;
			}

			w_stored.col(idx_run) = wt;
			W_stored.at(idx_run) = W;
		}

		Rcout << "\rProgress: " << b << "/" << ntotal-1;
	}

	Rcout << std::endl;

	Rcpp::List output;
	output["w"] = Rcpp::wrap(w_stored);
	wt_accept /= static_cast<double>(ntotal);
	output["w_accept"] = Rcpp::wrap(wt_accept);
	output["W"] = Rcpp::wrap(W_stored);
	return output;
}