# Fast Approximate Inference for a General Class of State-Space Models for Count Data


## Dependencies

- C++11 or newer.
- C++ libraries: [Armadillo](https://arma.sourceforge.net), [boost](https://www.boost.org), and [NLopt](https://nlopt.readthedocs.io/en/latest/).
- R packages: RcppArmadillo, nloptr, 
- Benchmarks: EpiEstim (an R package)

Compile in R:

```{r}
Rcpp::sourceCpp("./export.cpp")
```

## Model

This is a model for count time series, $\{y_{t},t=1,\dots\}$, where $y_{t}\in\{0,1,2,\dots\}$.

1. Observation equation:  $y_t| \chi_t,\phi  \sim  \cF(\chi_t,\phi),$ 
		where $\cF(\chi_t,\phi)$ is a distribution in the exponential family with natural parameter $\chi_t$ and scale parameter $\phi >0.$
2. Link Function:  $\mu_t(\chi_t)=\E(y_t| \chi_t,\phi)$ is assumed to be related to the a set of state parameters through a link function $\zeta(\cdot),$ i.e., 
		$$\eta_t=\zeta(\mu_t(\chi_t))=\bx'_{t}\bm{\varphi} + f_t(\by_{t-1},\bm{\theta}_t).$$
3. System Equation: $\bm{\theta}_t = \bm{g}_t(\bm{y}_{t-1},\bm{\theta}_{t-1}) + \bm{w}_t, \;\;\; \bm{w}_t \sim \mathcal{N}(\bzero,\bm{W}_t)$


## Inference

- Method 1. Linear Bayes Filtering and Smoothing: `lbe_poisson.cpp`
- Method 2. MCMC with Univariate MH Proposal via Reparameterisation: `mcmc_disturbance_poisson.cpp`
- Method 3. Particle Filtering and Smoothing: `pl_poisson.cpp`
- Method 4. Variational Inference: `vb_poisson.cpp` and `hva_poisson.cpp`

