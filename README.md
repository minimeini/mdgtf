# Comparison of Different Inference Methods for Poisson/Negative-Binomial DGLM

## Setting up Rcpp Compiler for Apple Silicon

1. Follow [R COMPILER TOOLS FOR RCPP ON MACOS](https://thecoatlessprofessor.com/programming/cpp/r-compiler-tools-for-rcpp-on-macos/)
2. Regarding the `gfortran` part, do NOT follow the first link. Instead, check out [Tools - R for Mac OS X](https://mac.r-project.org/tools/)

Run `xcode-select -v`, you should see

```
xcode-select version 2396.
```

Run `xcode-select -p`, you should see

```
/Library/Developer/CommandLineTools
```

Run `gcc --version`, you should see

```
Apple clang version 14.0.0 (clang-1400.0.29.202)
Target: arm64-apple-darwin22.2.0
Thread model: posix
InstalledDir: /Library/Developer/CommandLineTools/usr/bin
```



## Inference

- Method 1. [Done] Linear Bayes Filtering and Smoothing: `lbe_poisson.cpp`
- Method 2. [Done] MCMC with Univariate MH Proposal via Reparameterisation: `mcmc_disturbance_poisson.cpp`
- Method 3. [Ongoing] Particle Filtering and Smoothing: `pl_poisson.cpp`
- Method 4. [Ongoing] Variational Inference: `vb_poisson.cpp` and `hva_poisson.cpp`


## Visualization

`vis_pois_dlm.R`
