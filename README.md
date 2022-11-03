# Comparison of Different Inference Methods for Poisson DGLM

## Model Without Transfer Function

For $t=1,...,n$, we consider a univariate Poisson DLM with an AR component:

$$
\begin{aligned}
y_t &\sim\text{Pois}(\lambda_t),\ \log(\lambda_t) = E_t\\
E_t &= \rho E_{t-1} + w_t,\ w_t\sim\mathcal{N}(0,W)
\end{aligned}
$$

### Reparameterisation

$$
E_t = \rho^t E_0 + \sum_{j=1}^t \rho^{t-j} w_j
$$

We can reparameterize all $E_t$, for $t=1,...,n$, in one go using matrix multiplication as follows.

$$
\begin{pmatrix}
E_1 \\
E_2 \\
\vdots \\
E_n
\end{pmatrix}
=
\begin{pmatrix}
\rho \\
\rho^2 \\
\vdots \\
\rho^n
\end{pmatrix} E_0
+
\begin{pmatrix}
1          &            &        &        &   \\
\rho       & 1          &        &        &   \\
\vdots     & \ddots     & \ddots &        &   \\
\vdots     & \vdots     & \ddots & \ddots &  \\
\rho^{n-1} & \rho^{n-2} & \cdots & \rho   & 1
\end{pmatrix}
\begin{pmatrix}
w_1 \\
w_2 \\
\vdots \\
w_n
\end{pmatrix}
$$

Denote this equation by $E = \tilde{E}_0 + F\cdot w$.


## Model With Transfer Function

For $t=1,...,n$, we consider a univariate Poisson DLM with an AR component:

$$
\begin{aligned}
y_t &\sim\text{Pois}(\lambda_t),\ \log(\lambda_t) = E_t\\
E_t &= \rho E_{t-1} + x_t \beta_t\\
\beta_t &= \beta_{t-1} + w_t,\ w_t\sim\mathcal{N}(0,W)
\end{aligned}
$$

### Reparameterisation

$$
E_t = \rho^t E_0 + \sum_{j=1}^t \rho^{t-j} w_j
$$

We can reparameterize all $E_t$, for $t=1,...,n$, in one go using matrix multiplication as follows.

$$
\begin{pmatrix}
E_1 \\
E_2 \\
\vdots \\
E_n
\end{pmatrix}
=
\begin{pmatrix}
\rho \\
\rho^2 \\
\vdots \\
\rho^n
\end{pmatrix} E_0
+
\begin{pmatrix}
x_1        &            &        &        &   \\
\rho x_1 + x_2       & x_2          &        &        &   \\
\vdots     & \ddots     & \ddots &        &   \\
\vdots     & \vdots     & \ddots & \ddots &  \\
\sum_{l=1}^n\rho^{n-l}x_l & \sum_{l=2}^n\rho^{n-l}x_l & \cdots & \rho x_{n-1} + x_n  & x_n
\end{pmatrix}
\begin{pmatrix}
w_1 \\
w_2 \\
\vdots \\
w_n
\end{pmatrix}
$$

Denote this equation by $E = \tilde{E}_0 + F\cdot w$.



## Inference

- Method 1. [Done] Linear Bayes Filtering and Smoothing: `lbe_poisson.cpp`
    - TODO: smoothing
- Method 2. [Not Implemented Yet] Particle Filtering and Smoothing: `pf_poisson.cpp`
- Method 3. [Not Implemented Yet] MCMC with multivariate FFBS proposal: `gibbs_ffbs_poisson.cpp`
- Method 4. [Debugging] MCMC with univariate reparameterised proposal: `gibbs_disturbance_poisson.cpp`



