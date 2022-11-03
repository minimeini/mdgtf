#include "lbe_poisson.h"
using namespace Rcpp;
// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::depends(RcppArmadillo,nloptr)]]



double trigamma_obj(
	unsigned n,
	const double *x, 
	double *grad, 
	void *my_func_data) {

	double *q = (double*)my_func_data;

	if (grad) {
		grad[0] = 2*(R::trigamma(x[0])-(*q))*R::psigamma(x[0],2);
	}

	return std::pow(R::trigamma(x[0])-(*q),2);
}



double optimize_trigamma(double q) {
	nlopt_opt opt;
	opt = nlopt_create(NLOPT_LD_MMA, 1);

	double lb[1] = {0}; // lower bound
	nlopt_set_lower_bounds(opt,lb);
	nlopt_set_xtol_rel(opt,1e-4);
	nlopt_set_min_objective(opt,trigamma_obj,(void *) &q);

	double x[1] = {1e-6};
	double minf;
	if (nlopt_optimize(opt, x, &minf) < 0) {
    	Rprintf("nlopt failed!\\n");
	}
	
	double result = x[0];
	nlopt_destroy(opt);
	return result;
}



void forwardFilter(
	const unsigned int model,
	const arma::vec& Y, // n x 1, the observation (scalar), n: num of obs
	const double G, // state transition matrix
	const double W, // p x p, state error
	arma::vec& mt, // (n+1) x 1, t=0 is the mean for initial value theta[0]
	arma::vec& at, // (n+1) x 1
	arma::vec& Ct, // (n+1) x 1, t=0 is the var for initial value theta[0]
	arma::vec& Rt, // (n+1) x 1
	arma::vec& alphat, // (n+1) x 1
	arma::vec& betat) { // (n+1) x 1

	const unsigned int n = Y.n_elem; // number of observation
    const double F = 1.;

	double et,ft,Qt,At,ft_ast,Qt_ast;

	for (unsigned int t=1; t<=n; t++) {
		// Prior at time t: theta[t] | D[t-1] ~ N(at, Rt)
        if (t==1 && model==1) {
			Rt.at(t) = Ct.at(0); // R1 = Rw and a1 = rho*E0
        } else {
            Rt.at(t) = G * Ct.at(t-1) * G + W;
        }
		at.at(t) = G * mt.at(t-1);
		
		// One-step ahead forecast: Y[t]|D[t-1] ~ N(ft, Qt)
		ft = F * at.at(t);
		Qt = F * Rt.at(t) * F;

        alphat.at(t) = optimize_trigamma(Qt);
		betat.at(t) = std::exp(R::digamma(alphat.at(t)) - ft);

        ft_ast = R::digamma(alphat.at(t)+Y.at(t-1)) - std::log(betat.at(t)+1.);
		Qt_ast = R::trigamma(alphat.at(t)+Y.at(t-1));

		// Posterior at time t: theta[t] | D[t] ~ N(mt, Ct)
		et = ft_ast - ft;
		At = Rt.at(t) * F / Qt;
		mt.at(t) = at.at(t) + At * et;
		Ct.at(t) = Rt.at(t) - At*(Qt-Qt_ast)*At;
	}

}



// void backwardSmoother(
// 	const double G, // k x k, state transition matrix: assume time-invariant
// 	const arma::vec& mt, // (n+1) x 1, t=0 is the mean for initial value theta[0]
// 	const arma::vec& at, // (n+1) x 1
// 	const arma::vec& Ct, // (n+1) x 1, t=0 is the var for initial value theta[0]
// 	const arma::vec& Rt, // (n+1) x 1
// 	arma::vec& ht, // n x 1
// 	arma::vec& Ht) { // n x 1

// 	// Use the conditional distribution
// 	const unsigned int n = ht.n_elem; // num of obs

// 	ht.at(n-1) = mt.at(n);
// 	Ht.at(n-1) = Ct.at(n);

// 	double Bt;
// 	for (unsigned int t=(n-2); t>0; t--) {
// 		Bt = Ct.at(t+1) * G / Rt.at(t+2);
// 		ht.at(t) = mt.at(t+1) + Bt*(ht.at(t+1)-at.at(t+2));
// 		Ht.at(t) = Ct.at(t+1) - Bt*(Rt.at(t+2)-Ht.at(t+1))*Bt;
// 	}

// 	// t = 0
// 	Bt = Ct.at(1) * G / Rt.at(2);
// 	ht.at(0) = mt.at(1) + Bt*(ht.at(1)-at.at(2));
// 	Ht.at(0) = Ct.at(1) - Bt*(Rt.at(2)-Ht.at(1))*Bt;
// }


void forwardFilterX(
	const unsigned int model,
	const arma::vec& Y, // n x 1, the observation (scalar), n: num of obs
	const arma::vec& X, // n x 1
	const double W, // evolution/state error
	arma::mat& Gt, // 2 x 2, state transition matrix
	arma::mat& Wt, // 2 x 2, state error
	arma::mat& mt, // 2 x (n+1), t=0 is the mean for initial value theta[0]
	arma::mat& at, // 2 x (n+1)
	arma::cube& Ct, // 2 x 2 x (n+1), t=0 is the var for initial value theta[0]
	arma::cube& Rt, // 2 x 2 x (n+1)
	arma::vec& alphat, // (n+1) x 1
	arma::vec& betat) { // (n+1) x 1

	const unsigned int n = Y.n_elem; // number of observation
    arma::vec F(2); F.at(0) = 1.; F.at(1) = 0.;

	double et,ft,Qt,ft_ast,Qt_ast;
	arma::vec At(2);

	for (unsigned int t=1; t<=n; t++) {
		R_CheckUserInterrupt();

		Gt.at(0,1) = X.at(t-1);
		Wt.at(1,1) = 1.; 
		Wt.at(0,1) = X.at(t-1);
		Wt.at(1,0) = Wt.at(0,1); 
		Wt.at(0,0) = X.at(t-1)*Wt.at(1,0);
		Wt *= W;
		
		// Prior at time t: theta[t] | D[t-1] ~ N(at, Rt)
        if (t==1 && model==1) {
			Rt.slice(t) = Ct.slice(0); // R1 = Rw and a1 = rho*E0
        } else {
            Rt.slice(t) = Gt * Ct.slice(t-1) * Gt.t() + Wt;
        }
		at.col(t) = Gt * mt.col(t-1);
		
		// One-step ahead forecast: Y[t]|D[t-1] ~ N(ft, Qt)
		ft = arma::as_scalar(F.t() * at.col(t));
		Qt = arma::as_scalar(F.t() * Rt.slice(t) * F);

        alphat.at(t) = optimize_trigamma(Qt);
		betat.at(t) = std::exp(R::digamma(alphat.at(t)) - ft);

        ft_ast = R::digamma(alphat.at(t)+Y.at(t-1)) - std::log(betat.at(t)+1.);
		Qt_ast = R::trigamma(alphat.at(t)+Y.at(t-1));

		// Posterior at time t: theta[t] | D[t] ~ N(mt, Ct)
		et = ft_ast - ft;
		At = Rt.slice(t) * F / Qt;
		mt.col(t) = at.col(t) + At * et;
		Ct.slice(t) = Rt.slice(t) - At*(Qt-Qt_ast)*At.t();
	}

}



void forwardFilterX2(
	const unsigned int model,
	const arma::vec& Y, // n x 1, the observation (scalar), n: num of obs
	const arma::vec& X, // n x 1
	const double delta, // discount factor for evolution/state error
	arma::mat& Gt, // 2 x 2, state transition matrix
	arma::mat& mt, // 2 x (n+1), t=0 is the mean for initial value theta[0]
	arma::mat& at, // 2 x (n+1)
	arma::cube& Ct, // 2 x 2 x (n+1), t=0 is the var for initial value theta[0]
	arma::cube& Rt, // 2 x 2 x (n+1)
	arma::vec& alphat, // (n+1) x 1
	arma::vec& betat) { // (n+1) x 1

	const unsigned int n = Y.n_elem; // number of observation
    arma::vec F(2); F.at(0) = 1.; F.at(1) = 0.;

	double et,ft,Qt,ft_ast,Qt_ast;
	arma::vec At(2);

	for (unsigned int t=1; t<=n; t++) {
		R_CheckUserInterrupt();
		
		Gt.at(0,1) = X.at(t-1);
		
		// Prior at time t: theta[t] | D[t-1] ~ N(at, Rt)
        if (t==1 && model==1) {
			Rt.slice(t) = Ct.slice(0); // R1 = Rw and a1 = rho*E0
        } else {
            Rt.slice(t) = Gt * Ct.slice(t-1) * Gt.t() / delta;
        }
		at.col(t) = Gt * mt.col(t-1);
		
		// One-step ahead forecast: Y[t]|D[t-1] ~ N(ft, Qt)
		ft = arma::as_scalar(F.t() * at.col(t));
		Qt = arma::as_scalar(F.t() * Rt.slice(t) * F);

        alphat.at(t) = optimize_trigamma(Qt);
		betat.at(t) = std::exp(R::digamma(alphat.at(t)) - ft);

        ft_ast = R::digamma(alphat.at(t)+Y.at(t-1)) - std::log(betat.at(t)+1.);
		Qt_ast = R::trigamma(alphat.at(t)+Y.at(t-1));

		// Posterior at time t: theta[t] | D[t] ~ N(mt, Ct)
		et = ft_ast - ft;
		At = Rt.slice(t) * F / Qt;
		mt.col(t) = at.col(t) + At * et;
		Ct.slice(t) = Rt.slice(t) - At*(Qt-Qt_ast)*At.t();
	}

}



/*
---- Method ----
- Koyama's Discretized Hawkes Process
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


/*
Linear Bayes approximation based filtering for koyama's model with exponential on the transmission delay.
- Observational linear approximation - This function uses 1st order Taylor expansion for the nonlinear Ft at the observational equation.
- Known evolution variance - This function assumes the evolution variance, W, is fixed and known
*/
void forwardFilterKoyamaExp(
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
    arma::vec Ft = arma::exp(at.col(1)) % Fphi % Fy;
    ft = arma::accu(Ft);
    Qt = arma::as_scalar(Ft.t()*Rt.slice(1)*Ft);
    betat = ft / Qt;
    alphat = betat*ft;
    ft_ast = (alphat + Y.at(0)) / (betat + 1.);
    Qt_ast = ft_ast / (betat + 1.);
    et = ft_ast - ft;
	mt.col(1) = at.col(1) + Rt.slice(1)*Ft*et/Qt;
    arma::vec Ft2 = arma::exp(at.col(1)) % arma::reverse(Fphi) % Fy;
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
        
        Ft = arma::exp(at.col(t)) % Fphi % Fy; // L x 1
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
        if (!mt.col(t).is_finite()) {
            Rcout << "Current time: " << t << std::endl;
            Rcout << "m[t]: " << mt.col(t).t() << std::endl;
            stop("Non-finite values for mt");
        }
        Ft2 = arma::exp(at.col(t)) % arma::reverse(Fphi) % Fy;
		Ct.slice(t) = Rt.slice(t) - Rt.slice(t)*Ft*Ft2.t()*Rt.slice(t)*(1.-Qt_ast/Qt)/Qt;
	}

}



/*
Linear Bayes approximation based filtering for koyama's model with exponential on the transmission delay.
- Observational linear approximation - This function uses 1st order Taylor expansion for the nonlinear Ft at the observational equation.
- Evolutional discounting - This function assumes the evolution variance is time-varying, which we are not interested in. Instead, 
we preselect a discount factor, delta, to account for the time-varying evolutional variance.
*/
void forwardFilterKoyamaExp2(
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

    arma::vec Ft = arma::exp(at.col(1)) % Fphi % Fy; // L x 1
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
        
        Ft = arma::exp(at.col(t)) % Fphi % Fy; // L x 1
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
void backwardSmootherKoyamaExp(
	const arma::mat& G, // L x L, state transition matrix
	const arma::mat& mt, // L x (n+1), t=0 is the mean for initial value theta[0]
	const arma::mat& at, // L x (n+1)
	const arma::cube& Ct, // L x L x (n+1), t=0 is the var for initial value theta[0]
	const arma::cube& Rt, // L x L x (n+1)
	arma::mat& ht, // L x n
    arma::cube& Ht) { // L x L x n

	// Use the conditional distribution
	const unsigned int n = ht.n_cols; // num of obs
    const unsigned int L = G.n_rows; 
    double delta;

	ht.col(n-1) = mt.col(n);
    Ht.slice(n-1) = Ct.slice(n);

	arma::mat Bt(L,L);
	for (unsigned int t=(n-2); t>0; t--) {
		// sample from theta[t] | theta[t+1], D[t]
		// Bt = G * Ct.slice(t+1) * G.t();
        // delta = Bt.at(0,0) / Rt.at(0,0,t+2);
        Bt = Ct.slice(t+1) * G.t() * Rt.slice(t+2).i();
		ht.col(t) = mt.col(t+1) + Bt*(ht.col(t+1)-at.col(t+2));
        Ht.slice(t) = Ct.slice(t+1) - Bt*(Rt.slice(t+2)-Ht.slice(t+1))*Bt.t();
	}

	// t = 0
	Bt = Ct.slice(1) * G.t() * Rt.slice(2).i();
    // delta = Bt.at(0,0) / Rt.at(0,0,2);
    ht.col(0) = mt.col(1) + Bt*(ht.col(1)-at.col(2));
    Ht.slice(0) = Ct.slice(1) - Bt*(Rt.slice(2)-Ht.slice(1))*Bt.t();
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
void backwardSmootherKoyamaExp2(
	const arma::mat& G, // L x L, state transition matrix
	const arma::mat& mt, // L x (n+1), t=0 is the mean for initial value theta[0]
	const arma::mat& at, // L x (n+1)
	const arma::cube& Ct, // L x L x (n+1), t=0 is the var for initial value theta[0]
	const arma::cube& Rt, // L x L x (n+1)
	const double delta, // discount factor
	arma::vec& ht, // n
    arma::vec& Ht) { // n

	// Use the conditional distribution
	const unsigned int n = ht.n_cols; // num of obs
    const unsigned int L = G.n_rows; 

	ht.at(n-1) = mt.at(0,n);
    Ht.at(n-1) = Ct.at(0,0,n);

	for (unsigned int t=(n-2); t>0; t--) {
		// sample from theta[t] | theta[t+1], D[t]
		ht.at(t) = (1.-delta) * mt.at(0,t+1) + delta*ht.at(t+1);
        Ht.at(t) = (1.-delta) * Ct.at(0,0,t+1);
	}

	// t = 0
    ht.at(0) = (1.-delta)*mt.at(0,1) + delta*ht.at(1);
    Ht.at(0) = (1.-delta)*Ct.at(0,0,1);
}



/*
Solow's Pascal Disctributed Lags with r=2
*/
void forwardFilterSolow2_0(
	const arma::vec& Y, // n x 1, the observation (scalar), n: num of obs
	const arma::vec& X, // (n+1) x 1, x[0],x[1],...,x[n]
	const double W, // evolution error of the beta's
	const double rho,
	arma::mat& Gt, // 3 x 3, state transition matrix
	arma::mat& Wt, // 3 x 3, evolution error matrix
	arma::mat& mt, // 3 x (n+1), t=0 is the mean for initial value theta[0]
	arma::mat& at, // 3 x (n+1)
	arma::cube& Ct, // 3 x 3 x (n+1), t=0 is the var for initial value theta[0]
	arma::cube& Rt, // 3 x 3 x (n+1)
	arma::vec& alphat, // (n+1) x 1
	arma::vec& betat) { // (n+1) x 1

	const unsigned int n = Y.n_elem; // number of observation
    arma::vec F(3,arma::fill::zeros); F.at(0) = 1.;

	double et,ft,Qt,ft_ast,Qt_ast;
	double coef = (1.-rho) * (1.-rho);
	double coef4 = std::pow(coef,2.);
	arma::vec At(3);

	for (unsigned int t=1; t<=n; t++) {
		R_CheckUserInterrupt();
		Gt.at(0,2) = coef * X.at(t-1);

		Wt.zeros();
		Wt.at(0,0) = coef4 * X.at(t-1) * X.at(t-1);
		Wt.at(0,2) = coef * X.at(t-1);
		Wt.at(2,0) = coef * X.at(t-1);
		Wt.at(2,2) = 1.;
		Wt *= W;

		
		// Prior at time t: theta[t] | D[t-1] ~ N(at, Rt)
        Rt.slice(t) = Gt * Ct.slice(t-1) * Gt.t() + Wt;
		at.col(t) = Gt * mt.col(t-1);
		
		// One-step ahead forecast: Y[t]|D[t-1] ~ N(ft, Qt)
		ft = arma::as_scalar(F.t() * at.col(t));
		Qt = arma::as_scalar(F.t() * Rt.slice(t) * F);

        alphat.at(t) = optimize_trigamma(Qt);
		betat.at(t) = std::exp(R::digamma(alphat.at(t)) - ft);

        ft_ast = R::digamma(alphat.at(t)+Y.at(t-1)) - std::log(betat.at(t)+1.);
		Qt_ast = R::trigamma(alphat.at(t)+Y.at(t-1));

		// Posterior at time t: theta[t] | D[t] ~ N(mt, Ct)
		et = ft_ast - ft;
		At = Rt.slice(t) * F / Qt;
		mt.col(t) = at.col(t) + At * et;
		Ct.slice(t) = Rt.slice(t) - At*(Qt-Qt_ast)*At.t();
	}

}



/*
Solow's Pascal Disctributed Lags with r=2
*/
void forwardFilterSolow2(
	const arma::vec& Y, // n x 1, the observation (scalar), n: num of obs
	const arma::vec& X, // (n+1) x 1, x[0],x[1],...,x[n]
	const double delta, // discount factor for evolution/state error, delta = 1 is a constant value with zero variance
	const double rho,
	arma::mat& Gt, // 3 x 3, state transition matrix
	arma::mat& mt, // 3 x (n+1), t=0 is the mean for initial value theta[0]
	arma::mat& at, // 3 x (n+1)
	arma::cube& Ct, // 3 x 3 x (n+1), t=0 is the var for initial value theta[0]
	arma::cube& Rt, // 3 x 3 x (n+1)
	arma::vec& alphat, // (n+1) x 1
	arma::vec& betat) { // (n+1) x 1

	const unsigned int n = Y.n_elem; // number of observation
    arma::vec F(3,arma::fill::zeros); F.at(0) = 1.;

	double et,ft,Qt,ft_ast,Qt_ast;
	double coef = (1.-rho) * (1.-rho);
	arma::vec At(3);

	for (unsigned int t=1; t<=n; t++) {
		R_CheckUserInterrupt();
		Gt.at(0,2) = coef * X.at(t-1);
		
		// Prior at time t: theta[t] | D[t-1] ~ N(at, Rt)
        Rt.slice(t) = Gt * Ct.slice(t-1) * Gt.t() / delta;
		at.col(t) = Gt * mt.col(t-1);
		
		// One-step ahead forecast: Y[t]|D[t-1] ~ N(ft, Qt)
		ft = arma::as_scalar(F.t() * at.col(t));
		Qt = arma::as_scalar(F.t() * Rt.slice(t) * F);

        alphat.at(t) = optimize_trigamma(Qt);
		betat.at(t) = std::exp(R::digamma(alphat.at(t)) - ft);

        ft_ast = R::digamma(alphat.at(t)+Y.at(t-1)) - std::log(betat.at(t)+1.);
		Qt_ast = R::trigamma(alphat.at(t)+Y.at(t-1));

		// Posterior at time t: theta[t] | D[t] ~ N(mt, Ct)
		et = ft_ast - ft;
		At = Rt.slice(t) * F / Qt;
		mt.col(t) = at.col(t) + At * et;
		Ct.slice(t) = Rt.slice(t) - At*(Qt-Qt_ast)*At.t();
	}

}



/*
Solow's Pascal Disctributed Lags with r=2
and an identity link function
*/
void forwardFilterSolow2_Identity(
	const arma::vec& Y, // n x 1, the observation (scalar), n: num of obs
	const arma::vec& X, // n x 1, x[0],x[1],...,x[n]
	const double delta, // discount factor for evolution/state error, delta = 1 is a constant value with zero variance
	const double rho,
	arma::mat& Gt, // 3 x 3, state transition matrix
	arma::mat& mt, // 3 x (n+1), t=0 is the mean for initial value theta[0]
	arma::mat& at, // 3 x (n+1)
	arma::cube& Ct, // 3 x 3 x (n+1), t=0 is the var for initial value theta[0]
	arma::cube& Rt, // 3 x 3 x (n+1)
	arma::vec& alphat, // (n+1) x 1
	arma::vec& betat) { // (n+1) x 1

	const unsigned int n = Y.n_elem; // number of observation
    arma::vec F(3,arma::fill::zeros); F.at(0) = 1.;

	double et,ft,Qt,ft_ast,Qt_ast;
	double coef = (1.-rho) * (1.-rho);
	arma::vec At(3);

	for (unsigned int t=1; t<=n; t++) {
		R_CheckUserInterrupt();
		Gt.at(0,2) = coef * X.at(t-1);
		
		// Prior at time t: theta[t] | D[t-1] ~ N(at, Rt)
        Rt.slice(t) = Gt * Ct.slice(t-1) * Gt.t() / delta;
		at.col(t) = Gt * mt.col(t-1);
		
		// One-step ahead forecast: Y[t]|D[t-1] ~ N(ft, Qt)
		ft = arma::as_scalar(F.t() * at.col(t));
		Qt = arma::as_scalar(F.t() * Rt.slice(t) * F);
        
		betat.at(t) = ft/Qt;
		alphat.at(t) = ft*betat.at(t);

        ft_ast = (alphat.at(t)+Y.at(t-1)) / (betat.at(t)+1.);
		Qt_ast = ft_ast / (betat.at(t)+1.);

		// Posterior at time t: theta[t] | D[t] ~ N(mt, Ct)
		et = ft_ast - ft;
		At = Rt.slice(t) * F / Qt;
		mt.col(t) = at.col(t) + At * et;
		Ct.slice(t) = Rt.slice(t) - At*(Qt-Qt_ast)*At.t();
	}

}



/*
Solow's Pascal Disctributed Lags with r=2
and an identity link function
and nonlinear state space

y[t] ~ Pois(lambda[t])
lambda[t] = theta[t]
theta[t] = 2*rho*theta[t-1] - rho^2*theta[t-2] + (1-rho)^2*x[t-1]*exp(beta[t-1])
beta[t] = beta[t-1] + omega[t]

omega[t] ~ Normal(0,W), where W is modeled by discount factor delta

*/
void forwardFilterSolow2_Identity2(
	const arma::vec& Y, // n x 1, the observation (scalar), n: num of obs
	const arma::vec& X, // n x 1, x[0],x[1],...,x[n]
	const double delta, // discount factor for evolution/state error, delta = 1 is a constant value with zero variance
	const double rho,
	arma::mat& Gt, // 3 x 3, state transition matrix
	arma::mat& mt, // 3 x (n+1), t=0 is the mean for initial value theta[0]
	arma::mat& at, // 3 x (n+1)
	arma::cube& Ct, // 3 x 3 x (n+1), t=0 is the var for initial value theta[0]
	arma::cube& Rt, // 3 x 3 x (n+1)
	arma::vec& alphat, // (n+1) x 1
	arma::vec& betat) { // (n+1) x 1

	const unsigned int n = Y.n_elem; // number of observation
    arma::vec F(3,arma::fill::zeros); F.at(0) = 1.;

	double et,ft,Qt,ft_ast,Qt_ast;
	double coef = (1.-rho) * (1.-rho);
	double rho2 = rho * rho;
	arma::vec At(3);
	arma::vec gt(3);

	for (unsigned int t=1; t<=n; t++) {
		R_CheckUserInterrupt();
		Gt.at(0,2) = coef*X.at(t-1)*std::exp(mt.at(2,t-1));

		gt.at(0) = 2*rho*mt.at(0,t-1) - rho2*mt.at(1,t-1) + Gt.at(0,2);
		gt.at(1) = mt.at(0,t-1);
		gt.at(2) = mt.at(2,t-1);
		
		// Prior at time t: theta[t] | D[t-1] ~ N(at, Rt)
        Rt.slice(t) = Gt * Ct.slice(t-1) * Gt.t() / delta;
		at.col(t) = gt;
		
		// One-step ahead forecast: Y[t]|D[t-1] ~ N(ft, Qt)
		ft = arma::as_scalar(F.t() * at.col(t));
		Qt = arma::as_scalar(F.t() * Rt.slice(t) * F);
        
		betat.at(t) = ft/Qt;
		alphat.at(t) = ft*betat.at(t);

        ft_ast = (alphat.at(t)+Y.at(t-1)) / (betat.at(t)+1.);
		Qt_ast = ft_ast / (betat.at(t)+1.);

		// Posterior at time t: theta[t] | D[t] ~ N(mt, Ct)
		et = ft_ast - ft;
		At = Rt.slice(t) * F / Qt;
		mt.col(t) = at.col(t) + At * et;
		Ct.slice(t) = Rt.slice(t) - At*(Qt-Qt_ast)*At.t();
	}

}




/*
Linear Bayes Estimator - Version 0
Assume  rho, V, W are known;
        E[0] is unknown with prior N(m[0],C[0])
STATUS - Checked and Correct
*/
//' @export
// [[Rcpp::export]]
Rcpp::List lbe_poisson0(
	const arma::vec& Y, // n x 1, obs data
	const double rho, // aka G, state transition matrix: assume time-invariant
    const double W,
    const double m0,
    const double C0) {  // systematic variance

	const unsigned int n = Y.n_elem;
	arma::vec mt(n+1,arma::fill::zeros); mt.at(0) = m0;
    arma::vec Ct(n+1,arma::fill::zeros); Ct.at(0) = C0;
	arma::vec at(n+1,arma::fill::zeros);
    arma::vec Rt(n+1,arma::fill::zeros);
	arma::vec alphat(n+1,arma::fill::zeros);
	arma::vec betat(n+1,arma::fill::zeros);
	// arma::vec ht(n,arma::fill::zeros);
	// arma::vec Ht(n,arma::fill::zeros);

	forwardFilter(0,Y,rho,W,mt,at,Ct,Rt,alphat,betat);
	// backwardSmoother(rho,mt,at,Ct,Rt,ht,Ht);

	Rcpp::List output;
	output["mt"] = Rcpp::wrap(mt);
	output["Ct"] = Rcpp::wrap(Ct);
	// output["ht"] = Rcpp::wrap(ht);
	// output["Ht"] = Rcpp::wrap(Ht);

	return output;
}



/*
Linear Bayes Estimator - Version 1
Assume  rho, V, W, E[0] are known;
STATUS - Checked and Correct
*/
//' @export
// [[Rcpp::export]]
Rcpp::List lbe_poisson1(
	const arma::vec& Y, // n x 1, obs data
	const double rho, // aka G, state transition matrix: assume time-invariant
    const double W,
    const double E0,
    const double Rw) {  // systematic variance

	const unsigned int n = Y.n_elem;
	arma::vec mt(n+1,arma::fill::zeros); mt.at(0) = E0;
    arma::vec Ct(n+1,arma::fill::zeros); Ct.at(0) = Rw;
	arma::vec at(n+1,arma::fill::zeros);
    arma::vec Rt(n+1,arma::fill::zeros);
	arma::vec alphat(n+1,arma::fill::zeros);
	arma::vec betat(n+1,arma::fill::zeros);
	// arma::vec ht(n,arma::fill::zeros);
	// arma::vec Ht(n,arma::fill::zeros);

	forwardFilter(1,Y,rho,W,mt,at,Ct,Rt,alphat,betat);
	// backwardSmoother(rho,mt,at,Ct,Rt,ht,Ht);

	Rcpp::List output;
	output["mt"] = Rcpp::wrap(mt);
	output["Ct"] = Rcpp::wrap(Ct);
	// output["ht"] = Rcpp::wrap(ht);
	// output["Ht"] = Rcpp::wrap(Ht);

	return output;
}



/*
Linear Bayes Estimator with Transfer Function - Version 0
Assume  rho, W are known;
        E[0] and beta[0] is unknown with prior N(m[0],C[0])
STATUS - Checked and Correct
*/
//' @export
// [[Rcpp::export]]
Rcpp::List lbe_poissonX0(
	const arma::vec& Y, // n x 1, obs data
	const arma::vec& X,
	const double rho, // aka G, state transition matrix: assume time-invariant
    const double W,
    const arma::vec& m0, // 2 x 1
    const arma::mat& C0) {  // 2 x 2, systematic variance

	const unsigned int n = Y.n_elem;
	arma::mat Gt(2,2,arma::fill::eye); Gt.at(0,0) = rho;
	arma::mat Wt(2,2,arma::fill::eye);

	arma::mat mt(2,n+1,arma::fill::zeros); mt.col(0) = m0;
    arma::cube Ct(2,2,n+1,arma::fill::zeros); Ct.slice(0) = C0;
	arma::mat at(2,n+1,arma::fill::zeros);
    arma::cube Rt(2,2,n+1,arma::fill::zeros);
	arma::vec alphat(n+1,arma::fill::zeros);
	arma::vec betat(n+1,arma::fill::zeros);
	// arma::vec ht(n,arma::fill::zeros);
	// arma::vec Ht(n,arma::fill::zeros);

	forwardFilterX(0,Y,X,W,Gt,Wt,mt,at,Ct,Rt,alphat,betat);
	// backwardSmoother(rho,mt,at,Ct,Rt,ht,Ht);

	Rcpp::List output;
	output["mt"] = Rcpp::wrap(mt);
	output["Ct"] = Rcpp::wrap(Ct);
	// output["ht"] = Rcpp::wrap(ht);
	// output["Ht"] = Rcpp::wrap(Ht);

	return output;
}



/*
Linear Bayes Estimator with Transfer Function - Version 0
Assume  rho is known;
		use discount factor `delta` for Wt
        E[0] and beta[0] is unknown with prior N(m[0],C[0])
STATUS - Checked and Correct
*/
//' @export
// [[Rcpp::export]]
Rcpp::List lbe_poissonX(
	const arma::vec& Y, // n x 1, obs data
	const arma::vec& X, // n x 1
	const double rho, // aka G, state transition matrix: assume time-invariant
    const double delta, // discount factor
    const arma::vec& m0, // 2 x 1
    const arma::mat& C0) {  // 2 x 2, systematic variance

	const unsigned int n = Y.n_elem;
	arma::mat Gt(2,2,arma::fill::eye); Gt.at(0,0) = rho;

	arma::mat mt(2,n+1,arma::fill::zeros); mt.col(0) = m0;
    arma::cube Ct(2,2,n+1,arma::fill::zeros); Ct.slice(0) = C0;
	arma::mat at(2,n+1,arma::fill::zeros);
    arma::cube Rt(2,2,n+1,arma::fill::zeros);
	arma::vec alphat(n+1,arma::fill::zeros);
	arma::vec betat(n+1,arma::fill::zeros);
	// arma::vec ht(n,arma::fill::zeros);
	// arma::vec Ht(n,arma::fill::zeros);

	forwardFilterX2(0,Y,X,delta,Gt,mt,at,Ct,Rt,alphat,betat);
	// backwardSmoother(rho,mt,at,Ct,Rt,ht,Ht);

	Rcpp::List output;
	output["mt"] = Rcpp::wrap(mt);
	output["Ct"] = Rcpp::wrap(Ct);
	output["alphat"] = Rcpp::wrap(alphat);
	output["betat"] = Rcpp::wrap(betat);
	output["delta"] = delta;
	output["rho"] = rho;
	// output["ht"] = Rcpp::wrap(ht);
	// output["Ht"] = Rcpp::wrap(Ht);

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
        omega[t] ~ Normal(0,W)

<prior> theta_till[0] ~ Norm(m0, C0)
        W ~ IG(aw,bw)
*/
//' @export
// [[Rcpp::export]]
Rcpp::List lbe_poissonKoyamaExp(
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

	arma::mat ht(L,n,arma::fill::zeros); // E(theta[t] | Dn), for t=1,...,n
    arma::cube Ht(L,L,n);
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
    for (unsigned int t=0; t<n; t++) {
        Ht.slice(t).eye();
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

    forwardFilterKoyamaExp(Y,G,Fphi,W,mt,at,Ct,Rt);
    if (!mt.is_finite() || !at.is_finite() || !Ct.is_finite() || !Rt.is_finite()) {
        stop("Non-finite values for filtering.");
    }
    backwardSmootherKoyamaExp(G,mt,at,Ct,Rt,ht,Ht);
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
        omega[t] ~ Normal(0,W)

<prior> theta_till[0] ~ Norm(m0, C0)
        W ~ IG(aw,bw)
*/
//' @export
// [[Rcpp::export]]
Rcpp::List lbe_poissonKoyamaExp2(
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

	arma::vec ht(n,arma::fill::zeros); // E(theta[t] | Dn), for t=1,...,n
    arma::vec Ht(n,arma::fill::ones);
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

    forwardFilterKoyamaExp2(Y,G,Fphi,delta,mt,at,Ct,Rt);
    if (!mt.is_finite() || !at.is_finite() || !Ct.is_finite() || !Rt.is_finite()) {
        stop("Non-finite values for filtering.");
    }
    backwardSmootherKoyamaExp2(G,mt,at,Ct,Rt,delta,ht,Ht);
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
Linear Bayes Estimator with Solow's Transfer Function - Version 0
Assume  rho is known;
		use discount factor `delta` for Wt
        E[0] and beta[0] is unknown with prior N(m[0],C[0])
STATUS - Checked and Correct
*/
//' @export
// [[Rcpp::export]]
Rcpp::List lbe_poissonSolow0(
	const arma::vec& Y, // n x 1, obs data
	const arma::vec& X, // n x 1
	const double rho, // aka G, state transition matrix: assume time-invariant
    const double W, // evolution error variance
    const arma::vec& m0, // 3 x 1
    const arma::mat& C0) {  // 3 x 3, systematic variance

	const unsigned int n = Y.n_elem;
	arma::mat Gt(3,3,arma::fill::zeros); 
	Gt.at(0,0) = 2*rho;
	Gt.at(0,1) = -rho*rho;
	Gt.at(1,0) = 1.;
	Gt.at(2,2) = 1.;
	arma::mat Wt(3,3,arma::fill::zeros);

	arma::vec X_(n+1,arma::fill::zeros); 
	X_.tail(n) = X;

	arma::mat mt(3,n+1,arma::fill::zeros); mt.col(0) = m0;
    arma::cube Ct(3,3,n+1,arma::fill::zeros); Ct.slice(0) = C0;
	arma::mat at(3,n+1,arma::fill::zeros);
    arma::cube Rt(3,3,n+1,arma::fill::zeros);
	arma::vec alphat(n+1,arma::fill::zeros);
	arma::vec betat(n+1,arma::fill::zeros);
	// arma::vec ht(n,arma::fill::zeros);
	// arma::vec Ht(n,arma::fill::zeros);

	forwardFilterSolow2_0(Y,X_,W,rho,Gt,Wt,mt,at,Ct,Rt,alphat,betat);
	// backwardSmoother(rho,mt,at,Ct,Rt,ht,Ht);

	Rcpp::List output;
	output["mt"] = Rcpp::wrap(mt);
	output["Ct"] = Rcpp::wrap(Ct);
	output["alphat"] = Rcpp::wrap(alphat);
	output["betat"] = Rcpp::wrap(betat);
	output["W"] = W;
	output["rho"] = rho;
	// output["ht"] = Rcpp::wrap(ht);
	// output["Ht"] = Rcpp::wrap(Ht);

	return output;
}




/*
Linear Bayes Estimator with Solow's Transfer Function - Version 0
Assume  rho is known;
		use discount factor `delta` for Wt
        E[0] and beta[0] is unknown with prior N(m[0],C[0])
STATUS - Checked and Correct
*/
//' @export
// [[Rcpp::export]]
Rcpp::List lbe_poissonSolow(
	const arma::vec& Y, // n x 1, obs data
	const arma::vec& X, // n x 1
	const double rho, // aka G, state transition matrix: assume time-invariant
    const double delta, // discount factor
    const arma::vec& m0, // 3 x 1
    const arma::mat& C0) {  // 3 x 3, systematic variance

	const unsigned int n = Y.n_elem;
	arma::mat Gt(3,3,arma::fill::zeros); 
	Gt.at(0,0) = 2*rho;
	Gt.at(0,1) = -rho*rho;
	Gt.at(1,0) = 1.;
	Gt.at(2,2) = 1.;

	// arma::vec X_(n+1,arma::fill::zeros); 
	// X_.tail(n) = X;

	arma::mat mt(3,n+1,arma::fill::zeros); mt.col(0) = m0;
    arma::cube Ct(3,3,n+1,arma::fill::zeros); Ct.slice(0) = C0;
	arma::mat at(3,n+1,arma::fill::zeros);
    arma::cube Rt(3,3,n+1,arma::fill::zeros);
	arma::vec alphat(n+1,arma::fill::zeros);
	arma::vec betat(n+1,arma::fill::zeros);
	// arma::vec ht(n,arma::fill::zeros);
	// arma::vec Ht(n,arma::fill::zeros);

	forwardFilterSolow2(Y,X,delta,rho,Gt,mt,at,Ct,Rt,alphat,betat);
	// backwardSmoother(rho,mt,at,Ct,Rt,ht,Ht);

	Rcpp::List output;
	output["mt"] = Rcpp::wrap(mt);
	output["Ct"] = Rcpp::wrap(Ct);
	output["alphat"] = Rcpp::wrap(alphat);
	output["betat"] = Rcpp::wrap(betat);
	output["delta"] = delta;
	output["rho"] = rho;
	// output["ht"] = Rcpp::wrap(ht);
	// output["Ht"] = Rcpp::wrap(Ht);

	return output;
}


/*
Linear Bayes Estimator with Solow's Transfer Function
and an identity link

Assume  rho is known;
		use discount factor `delta` for Wt
        E[0] and beta[0] is unknown with prior N(m[0],C[0])
STATUS - Checked and Correct
*/
//' @export
// [[Rcpp::export]]
Rcpp::List lbe_poissonSolowIdentity(
	const arma::vec& Y, // n x 1, obs data
	const arma::vec& X, // n x 1
	const double rho, // aka G, state transition matrix: assume time-invariant
    const double delta, // discount factor
    const arma::vec& m0, // 3 x 1
    const arma::mat& C0) {  // 3 x 3, systematic variance

	const unsigned int n = Y.n_elem;
	arma::mat Gt(3,3,arma::fill::zeros); 
	Gt.at(0,0) = 2*rho;
	Gt.at(0,1) = -rho*rho;
	Gt.at(1,0) = 1.;
	Gt.at(2,2) = 1.;

	// arma::vec X_(n+1,arma::fill::zeros); 
	// X_.tail(n) = X;

	arma::mat mt(3,n+1,arma::fill::zeros); mt.col(0) = m0;
    arma::cube Ct(3,3,n+1,arma::fill::zeros); Ct.slice(0) = C0;
	arma::mat at(3,n+1,arma::fill::zeros);
    arma::cube Rt(3,3,n+1,arma::fill::zeros);
	arma::vec alphat(n+1,arma::fill::zeros);
	arma::vec betat(n+1,arma::fill::zeros);
	// arma::vec ht(n,arma::fill::zeros);
	// arma::vec Ht(n,arma::fill::zeros);

	forwardFilterSolow2_Identity(Y,X,delta,rho,Gt,mt,at,Ct,Rt,alphat,betat);
	// backwardSmoother(rho,mt,at,Ct,Rt,ht,Ht);

	Rcpp::List output;
	output["mt"] = Rcpp::wrap(mt);
	output["Ct"] = Rcpp::wrap(Ct);
	output["alphat"] = Rcpp::wrap(alphat);
	output["betat"] = Rcpp::wrap(betat);
	output["delta"] = delta;
	output["rho"] = rho;
	// output["ht"] = Rcpp::wrap(ht);
	// output["Ht"] = Rcpp::wrap(Ht);

	return output;
}



/*
Linear Bayes Estimator with Solow's Transfer Function
and an identity link
and nonlinear state space

y[t] ~ Pois(lambda[t])
lambda[t] = theta[t]
theta[t] = 2*rho*theta[t-1] - rho^2*theta[t-2] + (1-rho)^2*x[t-1]*exp(beta[t-1])
beta[t] = beta[t-1] + omega[t]

omega[t] ~ Normal(0,W), where W is modeled by discount factor delta

Assume  rho is known;
		use discount factor `delta` for Wt
        theta[0] and beta[0] is unknown with prior N(m[0],C[0])
STATUS - Checked and Correct
*/
//' @export
// [[Rcpp::export]]
Rcpp::List lbe_poissonSolowIdentity2(
	const arma::vec& Y, // n x 1, obs data
	const arma::vec& X, // n x 1
	const double rho, // aka G, state transition matrix: assume time-invariant
    const double delta, // discount factor
    const arma::vec& m0, // 3 x 1
    const arma::mat& C0) {  // 3 x 3, systematic variance

	const unsigned int n = Y.n_elem;
	arma::mat Gt(3,3,arma::fill::zeros); 
	Gt.at(0,0) = 2*rho;
	Gt.at(0,1) = -rho*rho;
	Gt.at(1,0) = 1.;
	Gt.at(2,2) = 1.;

	// arma::vec X_(n+1,arma::fill::zeros); 
	// X_.tail(n) = X;

	arma::mat mt(3,n+1,arma::fill::zeros); mt.col(0) = m0;
    arma::cube Ct(3,3,n+1,arma::fill::zeros); Ct.slice(0) = C0;
	arma::mat at(3,n+1,arma::fill::zeros);
    arma::cube Rt(3,3,n+1,arma::fill::zeros);
	arma::vec alphat(n+1,arma::fill::zeros);
	arma::vec betat(n+1,arma::fill::zeros);
	// arma::vec ht(n,arma::fill::zeros);
	// arma::vec Ht(n,arma::fill::zeros);

	forwardFilterSolow2_Identity2(Y,X,delta,rho,Gt,mt,at,Ct,Rt,alphat,betat);
	// backwardSmoother(rho,mt,at,Ct,Rt,ht,Ht);

	Rcpp::List output;
	output["mt"] = Rcpp::wrap(mt);
	output["Ct"] = Rcpp::wrap(Ct);
	output["alphat"] = Rcpp::wrap(alphat);
	output["betat"] = Rcpp::wrap(betat);
	output["delta"] = delta;
	output["rho"] = rho;
	// output["ht"] = Rcpp::wrap(ht);
	// output["Ht"] = Rcpp::wrap(Ht);

	return output;
}

