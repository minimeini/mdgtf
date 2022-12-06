cdir = getwd()
repo = "/Users/meinitang/Dropbox/Repository/poisson-dlm"

Rcpp::sourceCpp(file.path(repo,"model_utils.cpp"))

# TODO
# R can only parse default values in the Cpp file not the header file.
# We need a hacky way to adapt to it.

# ------------------------
# ------ Model Code ------
# ------------------------
#
# 0 - (KoyamaMax) Identity link    + LogNorm transmission delay      + ramp function (aka max(x,0)) on gain factor (psi)
# 1 - (KoyamaExp) Identity link    + LogNorm transmission delay      + exponential function on gain factor
# 2 - (SolowMax)  Identity link    + NegBinom transmission delay     + ramp function on gain factor
# 3 - (SolowExp)  Identity link    + NegBinom transmission delay     + exponential function on gain factor
# 4 - (KoyckMax)  Identity link    + Exponential transmission delay  + ramp function on gain factor
# 5 - (KoyckExp)  Identity link    + Exponential transmission delay  + exponential function on gain factor
# 6 - (KoyamaEye) Exponential link + LogNorm transmission delay      + identity function on gain factor
# 7 - (SolowEye)  Exponential link + NegBinom transmission delay     + identity function on gain factor
# 8 - (KoyckEye)  Exponential link + Exponential transmission delay  + identity function on gain factor
# 9 - (Vanilla)   Exponential link + Exponential transmission delay  + No gain



sim_pois_dglm = function(
    n = 200, # number of observations
    ModelCode = 0, # 0 - KoyamaMax; 1 - KoyamaExp; 2 - SolowMax; 3 - SolowExp
    mu0 = 0., # the baseline intensity
    psi0 = 0.,
    theta0 = 0., # initial value for the transfer function block; set it to NULL to sample from a uniform distribution(0,10).
    W = 0.01, # Evolution variance
    L = 0, # length of nonzero transmission delay (Koyama - ModelCode = 0 or 1)
    rho = 0.7, # parameter for negative binomial transmission delay (Solow - ModelCode = 2 or 3)
    rng.seed = NULL) {

    UPBND = 100

    c1 = 2*rho
    c2 = rho^2
    c3 = (1-rho)^2
    
    wt = rep(0,n)
    if (!is.null(rng.seed)) { set.seed(rng.seed)}
    wt[2:n] = rnorm(n-1,0,sqrt(W)) # wt[1] = 0

    if (is.null(psi0)) {
        set.seed(rng.seed)
        psi0 = rnorm(1,0,sd=1)
    }
    psi = cumsum(wt) + psi0
    psi[psi>UPBND] = UPBND
    hpsi = psi

    theta = rep(0,n)
    lambda = rep(0,n)
    y = rep(0,n)

    if (is.null(theta0)) {
        set.seed(rng.seed)
        theta0 = runif(1,0,2)
    }

    if (ModelCode == 0) { # KoyamaMax
        # <obs> y[t] | lambda[t] ~ Pois(lambda[t])
        # <link> lambda[t] = lambda[0] + theta[t]
        # <state> theta[t] = phi[1] max(psi[t],0) y[t-1] + phi[2] max(psi[t-1],0) y[t-2] + ... + phi[L] max(psi[t-L+1],0) y[t-L]
        # <state> psi[t] = psi[t-1] + omega[t], omega[t] ~ N(0,W)
        hpsi[hpsi<.Machine$double.eps] = .Machine$double.eps # Reproduction number after maximum thresholding
        Fphi = get_Fphi(L)
        lambda[1] = mu0

        if (!is.null(rng.seed)) { set.seed(rng.seed)}
        y[1] = rpois(1,lambda[1])
        for (t in 2:n) {
            ytilde = y[(t-1):max(c(1,t-L))]
            nlag = length(ytilde)
            theta[t] = sum(Fphi[1:nlag]*hpsi[t:(t-nlag+1)]*ytilde) # <state>
            lambda[t] = mu0 + theta[t] # <link - intensity>
            y[t] = rpois(1,lambda[t]) # <obs>
        }
    } else if (ModelCode == 1) { # KoyamaExp
        # <obs> y[t] | lambda[t] ~ Pois(lambda[t])
        # <link> lambda[t] = lambda[0] + theta[t]
        # <state> theta[t] = phi[1] exp(psi[t]) y[t-1] + phi[2] exp(psi[t-1]) y[t-2] + ... + phi[L] exp(psi[t-L+1]) y[t-L]
        # <state> psi[t] = psi[t-1] + omega[t], omega[t] ~ N(0,W)
        hpsi = exp(hpsi) # exponentiated reproduction number
        Fphi = get_Fphi(L)
        lambda[1] = mu0

        if (!is.null(rng.seed)) { set.seed(rng.seed)}
        y[1] = rpois(1,lambda[1])
        for (t in 2:n) {
            ytilde = y[(t-1):max(c(1,t-L))]
            nlag = length(ytilde)
            theta[t] = sum(Fphi[1:nlag]*hpsi[t:(t-nlag+1)]*ytilde) # <state>
            lambda[t] = mu0 + theta[t] # <link - intensity>
            y[t] = rpois(1,lambda[t]) # <obs>
        }
    } else if (ModelCode == 2) { # SolowMax
        # <obs> y[t] ~ Pois(lambda[t])
        # <link> lambda[t] = lambda[0] + theta[t]
        # <state> theta[t] = 2*rho*theta[t-1] - rho^2*theta[t-2] + (1-rho)^2*y[t-1]*max(psi[t-1],0)
        # <state> psi[t] = psi[t-1] + omega[t]
        hpsi[hpsi<.Machine$double.eps] = .Machine$double.eps # Reproduction number after maximum thresholding
        lambda[1] = mu0

        if (!is.null(rng.seed)) { set.seed(rng.seed)}
        y[1] = rpois(1,lambda[1])
        for (t in 2:n) {
            if (t==2) {
                theta[t] = c1*theta[1] + c3*y[t-1]*hpsi[t-1]
            } else {
                theta[t] = c1*theta[t-1] - c2*theta[t-2] + c3*y[t-1]*hpsi[t-1]
            }
            lambda[t] = mu0 + theta[t]
            y[t] = rpois(1,lambda[t])
        }

    } else if (ModelCode == 3) { # SolowExp
        # <obs> y[t] ~ Pois(lambda[t])
        # <link> lambda[t] = mu0 + theta[t]
        # <state> theta[t] = 2*rho*theta[t-1] - rho^2*theta[t-2] + (1-rho)^2*y[t-1]*exp(psi[t-1])
        # <state> psi[t] = psi[t-1] + omega[t]
        hpsi = exp(hpsi)
        theta[1] = 2*rho*theta0
        lambda[1] = mu0 + theta[1]
        if (!is.null(rng.seed)) { set.seed(rng.seed)}
        y[1] = rpois(1,lambda[1])

        for (t in 2:n) {
            if (t==2) {
                theta[t] = c1*theta[1] + c3*y[t-1]*hpsi[t-1]
            } else {
                theta[t] = c1*theta[t-1] - c2*theta[t-2] + c3*y[t-1]*hpsi[t-1]
            }
            lambda[t] = mu0 + theta[t]
            y[t] = rpois(1,lambda[t])
        }

    } else if (ModelCode == 4) { # KoyckMax
        # <obs> y[t] ~ Pois(lambda[t])
        # <link> lambda[t] = theta[t]
        # <state> theta[t] = rho*theta[t-1] + y[t-1]*max(psi[t-1],0)
        # <state> psi[t] = psi[t-1] + omega[t]
        hpsi[hpsi<.Machine$double.eps] = .Machine$double.eps # Reproduction number after maximum thresholding
        lambda[1] = mu0

        if (!is.null(rng.seed)) { set.seed(rng.seed)}
        y[1] = rpois(1,lambda[1])
        for (t in 2:n) {
            theta[t] = rho*theta[t-1] + y[t-1]*hpsi[t-1]
            lambda[t] = mu0 + theta[t]
            y[t] = rpois(1,lambda[t])
        }

    } else if (ModelCode == 5) { # KoyckExp
        # <obs> y[t] ~ Pois(lambda[t])
        # <link> lambda[t] = theta[t]
        # <state> theta[t] = rho*theta[t-1] + y[t-1]*exp(psi[t-1])
        # <state> psi[t] = psi[t-1] + omega[t]
        hpsi = exp(hpsi)
        lambda[1] = mu0

        if (!is.null(rng.seed)) { set.seed(rng.seed)}
        y[1] = rpois(1,lambda[1])
        for (t in 2:n) {
            theta[t] = rho*theta[t-1] + y[t-1]*hpsi[t-1]
            lambda[t] = mu0 + theta[t]
            y[t] = rpois(1,lambda[t])
        }
    } else if (ModelCode == 6) { # KoyamaEye
        # <obs> y[t] | lambda[t] ~ Pois(lambda[t])
        # <link> lambda[t] = exp(theta[t])
        # <state> theta[t] = phi[1] psi[t] y[t-1] + phi[2] psi[t-1] y[t-2] + ... + phi[L] psi[t-L+1] y[t-L]
        # <state> psi[t] = psi[t-1] + omega[t], omega[t] ~ N(0,W)
        Fphi = get_Fphi(L)
        theta[1] = .Machine$double.eps
        lambda[1] = exp(min(c(mu0+theta[1],UPBND)))

        if (!is.null(rng.seed)) { set.seed(rng.seed)}
        y[1] = rpois(1,lambda[1])
        for (t in 2:n) {
            ytilde = y[(t-1):max(c(1,t-L))]
            nlag = length(ytilde)
            theta[t] = sum(Fphi[1:nlag]*hpsi[t:(t-nlag+1)]*ytilde) # <state>
            lambda[t] = exp(min(c(mu0+theta[t],UPBND))) # <link - intensity>
            y[t] = rpois(1,lambda[t]) # <obs>
        }
    } else if (ModelCode == 7) { # SolowEye
        # <obs> y[t] ~ Pois(lambda[t])
        # <link> lambda[t] = exp(theta[t])
        # <state> theta[t] = 2*rho*theta[t-1] - rho^2*theta[t-2] + (1-rho)^2*y[t-1]*psi[t-1]
        # <state> psi[t] = psi[t-1] + omega[t]
        theta[1] = .Machine$double.eps
        lambda[1] = exp(min(c(mu0+theta[1],UPBND)))

        if (!is.null(rng.seed)) { set.seed(rng.seed)}
        y[1] = rpois(1,lambda[1])
        for (t in 2:n) {
            if (t==2) {
                theta[t] = c1*theta[1] + c3*y[t-1]*hpsi[t-1]
            } else {
                theta[t] = c1*theta[t-1] - c2*theta[t-2] + c3*y[t-1]*hpsi[t-1]
            }
            lambda[t] =  exp(min(c(mu0+theta[t],UPBND)))
            y[t] = rpois(1,lambda[t])
        }
    } else if (ModelCode == 8) { # KoyckEye
        # <obs> y[t] ~ Pois(lambda[t])
        # <link> lambda[t] = exp(mu0 + theta[t])
        # <state> theta[t] = rho*theta[t-1] + y[t-1]*psi[t-1]
        # <state> psi[t] = psi[t-1] + omega[t]
        theta[1] = .Machine$double.eps
        lambda[1] = exp(min(c(mu0+theta[1],UPBND)))

        if (!is.null(rng.seed)) { set.seed(rng.seed)}
        y[1] = rpois(1,lambda[1])
        for (t in 2:n) {
            theta[t] = rho*theta[t-1] + y[t-1]*hpsi[t-1]
            lambda[t] = exp(min(c(mu0+theta[t],UPBND)))
            y[t] = rpois(1,lambda[t])
        }
    } else if (ModelCode == 9) {
        # <obs> y[t] ~ Pois(lambda[t])
        # <link> lambda[t] = exp(mu0 + theta[t])
        # <state> theta[t] = rho*theta[t-1] + omega[t]
        theta[1] = .Machine$double.eps + rho*theta0 + wt[1]
        lambda[1] = exp(min(c(mu0+theta[1],UPBND)))

        if (!is.null(rng.seed)) { set.seed(rng.seed)}
        y[1] = rpois(1,lambda[1])
        for (t in 2:n) {
            theta[t] = rho*theta[t-1] + wt[t]
            lambda[t] = exp(min(c(mu0+theta[t],UPBND)))
            y[t] = rpois(1,lambda[t])
        }

    } else {
        stop("Not implemented yet.")
    }

    params = list(ModelCode=ModelCode,mu0=mu0,theta0=theta0,W=W,psi0=psi0,rho=rho,L=L)

    return(list(y=y, # Observation
        lambda=lambda, # Intensity
        theta=theta, # Transfer Function Block
        psi=psi, # Gain Factor
        hpsi=hpsi, # Reproduction Number - Function of the Gain Factor
        wt=wt, # Evolution Variance
        params=params)) # Model settings and initial values
}