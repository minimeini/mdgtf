# Comparison of Different Inference Methods for Poisson/Negative-Binomial DGLM

## Dependencies

- C++ libraries: [Armadillo](https://arma.sourceforge.net) and [NLopt](https://nlopt.readthedocs.io/en/latest/).
- R packages: RcppArmadillo, nloptr

## Setting up Rcpp Compiler for Apple Silicon

1. Follow [R COMPILER TOOLS FOR RCPP ON MACOS](https://thecoatlessprofessor.com/programming/cpp/r-compiler-tools-for-rcpp-on-macos/)
2. Regarding the `gfortran` part, do NOT follow the first link. Instead, check out [Tools - R for Mac OS X](https://mac.r-project.org/tools/)

## Sanity Check

Run `xcode-select -v`, I get

```
xcode-select version 2396.
```

Run `xcode-select -p`, I get

```
/Library/Developer/CommandLineTools
```

Run `gcc --version`, I get

```
Apple clang version 14.0.0 (clang-1400.0.29.202)
Target: arm64-apple-darwin22.2.0
Thread model: posix
InstalledDir: /Library/Developer/CommandLineTools/usr/bin
```

Check `R.version.string` in `R` version, I get

```
> R.version.string
[1] "R version 4.2.2 (2022-10-31)"
```

The `macOS` version I am using is

```
System Version: macOS 13.1 (22C65)
Kernel Version: Darwin 22.2.0
Processor: Apple M1 Pro
```

The R-related headers are location in 

```
> R.home()
[1] "/Library/Frameworks/R.framework/Resources"
> RcppArmadillo:::CxxFlags()
-I"/Library/Frameworks/R.framework/Versions/4.2-arm64/Resources/library/RcppArmadillo/include"
```

## Available Models

| Name           | ID   | Transmission Delay | Gain Function      | Link Function |
|----------------|------|--------------------|--------------------|---------------|
| KoyamaMax      | 0    | Log-normal         | Ramp               | Identity      |
| KoyamaExp      | 1    | Log-normal         | Exponential        | Identity      |
| SolowMax       | 2    | Negative-binomial  | Ramp               | Identity      |
| SolowExp       | 3    | Negative-binomial  | Exponential        | Identity      |
| KoyckMax       | 4    | Exponential        | Ramp               | Identity      |
| KoyckExp       | 5    | Exponential        | Exponential        | Identity      |
| KoyamaEye      | 6    | Log-normal         | Identity           | Exponential   |
| SolowEye       | 7    | Negative-binomial  | Identity           | Exponential   |
| KoyckEye       | 8    | Exponential        | Identity           | Exponential   |
| VanillaPois    | 9    | Exponential        | No                 | Exponential   |
| KoyckSoftplus  | 10   | Exponential        | Softplus           | Identity      |
| KoyamaSoftplus | 11   | Log-normal         | Softplus           | Identity      |
| SolowSoftplus  | 12   | Negative-binomial  | Softplus           | Identity      |
| KoyckTanh      | 13   | Exponential        | Hyperbolic Tangent | Identity      |
| KoyamaTanh     | 14   | Log-normal         | Hyperbolic Tangent | Identity      |
| SolowTanh      | 15   | Negative-binomial  | Hyperbolic Tangent | Identity      |

## Inference

- Method 1. Linear Bayes Filtering and Smoothing: `lbe_poisson.cpp`
- Method 2. MCMC with Univariate MH Proposal via Reparameterisation: `mcmc_disturbance_poisson.cpp`
- Method 3. Particle Filtering and Smoothing: `pl_poisson.cpp`
- Method 4. Variational Inference: `vb_poisson.cpp` and `hva_poisson.cpp`


## Visualization

`vis_pois_dlm.R`
