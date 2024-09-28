### Fast Approximate Inference for a General Class of State-Space Models for Count Data


Dependencies

- C++11 or newer.
- C++ libraries: [Armadillo](https://arma.sourceforge.net), [boost](https://www.boost.org), [NLopt](https://nlopt.readthedocs.io/en/latest/), and [openMP](https://www.openmp.org).
- R packages: RcppArmadillo, nloptr, 
- Benchmarks: EpiEstim (an R package)

Compile in R:

```{r}
Rcpp::sourceCpp("./export.cpp")
```


Documentation is in the [wiki](https://bitbucket.org/minimeini/dgtf/wiki/).

