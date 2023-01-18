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

The `R` version I am using is

```
R version 4.2.2 (2022-10-31) -- "Innocent and Trusting"
Copyright (C) 2022 The R Foundation for Statistical Computing
Platform: aarch64-apple-darwin20 (64-bit)
```

The `macOS` version I am using is

```
System Version: macOS 13.1 (22C65)
Kernel Version: Darwin 22.2.0
Processor: Apple M1 Pro
```


## Inference

- Method 1. Linear Bayes Filtering and Smoothing: `lbe_poisson.cpp`
- Method 2. MCMC with Univariate MH Proposal via Reparameterisation: `mcmc_disturbance_poisson.cpp`
- Method 3. Particle Filtering and Smoothing: `pl_poisson.cpp`
- Method 4. Variational Inference: `vb_poisson.cpp` and `hva_poisson.cpp`


## Visualization

`vis_pois_dlm.R`
