# Comparison of Different Inference Methods for Poisson/Negative-Binomial DGLM

## Dependencies

- C++11 or newer.
- C++ libraries: [Armadillo](https://arma.sourceforge.net), [boost](https://www.boost.org), and [NLopt](https://nlopt.readthedocs.io/en/latest/).
- R packages: RcppArmadillo, nloptr, 
- Benchmarks: EpiEstim (an R package)

## Available Models



## Inference

- Method 1. Linear Bayes Filtering and Smoothing: `lbe_poisson.cpp`
- Method 2. MCMC with Univariate MH Proposal via Reparameterisation: `mcmc_disturbance_poisson.cpp`
- Method 3. Particle Filtering and Smoothing: `pl_poisson.cpp`
- Method 4. Variational Inference: `vb_poisson.cpp` and `hva_poisson.cpp`


## Demo

- `script_model_sample_data.R`: Sample data provided by Koyama.
- `script_model_country_data.R`: Country level Covid daily new confirmed cases from March 1, 2020 to Dec 1, 2020.

