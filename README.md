# Fast Approximate Inference for a General Class of State-Space Models for Count Data


## Dependencies

- C++11 or newer.
- C++ libraries: [Armadillo](https://arma.sourceforge.net), [boost](https://www.boost.org), and [NLopt](https://nlopt.readthedocs.io/en/latest/).
- R packages: RcppArmadillo, nloptr, 
- Benchmarks: EpiEstim (an R package)

## Models

This is a model for count time series, $\{y_{t},t=1,\dots\}$, where $y_{t}\in\{0,1,2,\dots\}$.

\begin{enumerate}
	\item {\bf Observation equation:} Let $y_t$ denote the number of occurrences at time $t$ (e.g., number of cases or deaths due to a disease at time $t$). Then,
		the observation equation is given by $y_t| \chi_t,\phi  \sim  \cF(\chi_t,\phi),$ 
		where $\cF(\chi_t,\phi)$ is a distribution in the exponential family with natural parameter $\chi_t$ and scale parameter $\phi >0.$
		We further assume that $y_1,\ldots,y_T$ are conditionally independent given $\chi_t$ and $\phi$. In other words, we assume 
		\begin{align}
			p(y_t|\chi_t,\phi) = b(y_t,\phi) \times \exp\left\{ \phi[y_t\chi_t - a(\chi_t)] \right\},
			\label{obs_eq}
		\end{align}
		with $\E(y_t|\chi_t,\phi)=\mu_t(\chi_t)=a'(\chi_t)$ and $\Var(y_t|\chi_t,\phi)= a''(\chi_t)/\phi.$ 

%		Examples of exponential family models commonly used in disease dynamics include: 
%		\begin{itemize}
%			\item {\sl Poisson model:} In this case $y_t| \lambda_t \sim \mbox{Poisson}(\lambda_t),$ i.e., 
%				\begin{eqnarray}
%					p(y_t |\lambda_t,\phi) = \frac{{\lambda_t}^{y_t}}{y_t!} \exp\{-\lambda_t\},
%				\end{eqnarray}
%		and so, $\phi=1,$ $\chi_t=\log(\lambda_t)$, $a(\chi_t)=\exp(\chi_t)=\lambda_t$ and $b(y_t,\phi)=y_t!.$
%	\item {\sl Negative binomial model:} Here it is assumed that $y_t| \lambda_t,\rho \sim \mbox{Neg-Bin}(\lambda_t,\rho),$ 
%		with $\rho >0$ fixed, i.e.,
%		\begin{equation}
%			p(y_t | \lambda_t, \rho) = \frac{\Gamma(y_t + \rho)}{\Gamma(\rho) y_t!} \left( \frac{\rho}{\lambda_t + \rho} \right)^\rho 
%			\left( \frac{\lambda_t}{\lambda_t + \rho} \right)^{y_t}. 
 %           \label{eq-obs-negbinom}
%		\end{equation}
%		Then, $\phi=1,$ $\chi_t=\log\left( \frac{\lambda_t}{\lambda_t + \rho} \right),$ which implies that $\lambda_t = \frac{\rho e^{\chi_t}}{(1-e^{\chi_t})}.$ 
%				Also, 
%				$a(\chi_t)=-\rho \log \left(\rho(1-e^{\chi_t}) \right),$  leading to $\E(y_t |\lambda_t, \rho)=a'(\chi_t)=\lambda_t,$ and 
%				$\Var(y_t|\lambda_t, \rho) = \lambda_t (1 + \lambda_t/\rho).$ Note also that if $\rho \rightarrow \infty$ we obtain the Poisson distribution with parameter $\lambda_t.$ Finally, $b(y_t,\phi)=\Gamma(y_t+\rho)/(\Gamma(\rho) y_t!).$
		% \end{itemize}
	\item {\bf Link function:}  $\mu_t(\chi_t)=\E(y_t| \chi_t,\phi)$ is assumed to be related to the a set of state parameters through a link function $\zeta(\cdot),$ i.e., 
		\begin{equation}
			\eta_t=\zeta(\mu_t(\chi_t))=\bx'_{t}\bm{\varphi} + f_t(\by_{t-1},\btheta_t). \label{link} 
		\end{equation}
    {
        \color{orange} Maybe it is better to use $ \eta_t=\zeta(\mu_t(\chi_t))=\varphi_{t} f_t(\by_{t-1},\btheta_t)$, where $\varphi_{t}=\bx'_{t}\bm{\varphi}$? \color{magenta} let's discuss this when we meet; also, you need to make sure this is accounted for in all the models, algorithms and discussion below. \color{black}
    }
  %$a$ is a static baseline parameter and 
  $\btheta_t$ represents the time-varying state vector. $\bx_{t}$ and $\bm{\varphi}$ denote the regressors and regression coefficients, respectively, which could include a constant intercept, periodic seasonal components, or other exogeneous effects. The function $f_t(\cdot,\cdot)$ may depend solely on the state vector $\btheta_t$, but it can also be a function of $L$ past values of $y_t$ collected in a vector denoted as $\by_{t-1}$, with $\by_{t-1}=(y_{t-1},\ldots,y_{t-L})'$. Whether $f_t(\by_{t-1},\btheta_t)$ is a function of only $\by_{t-1}$, only $\btheta_t$, or a function of both, depends on the model being considered, as illustrated below. In many models the relationship between $\eta_t$ and the state parameters may be fully linear on $\btheta_t,$ and if this is the case we write $f_t(\by_{t-1},\btheta_t)=\bF'_t \btheta_t,$ with $\bF_t$ a vector possibly dependent on $\by_{t-1}.$ 
  %Note that we can also consider $\eta_t = f_t(a,\mathbf{y}_{t-1},\boldsymbol{\theta}_t),$
 % or incorporate $a$ as part of $\boldsymbol{\theta}_t,$ where some components of $\boldsymbol{\theta}_t$ may be static over time (such as $a$) and some may be time-varying. 
	\item {\bf System equation:} This equation describes the dynamics of the state parameters in the model as follows:
		\begin{align}
			\btheta_t = \bg_t(\by_{t-1},\btheta_{t-1}) + \bw_t, \;\;\; \bw_t \sim N(\bzero,\bW_t). \label{state_eq}
		\end{align}
		The function $\bg_t(\cdot,\cdot)$ can be a function of $\btheta_{t-1}$ only, or may also incorporate $L$ past values of $y_t.$
		Many models consider a linear system equation, i.e., $\bg_t(\by_{t-1},\btheta_{t-1})=\bG_t \btheta_{t-1},$ with $\bG_t$ a matrix
		possibly dependent on $\by_{t-1}$ but not on $\btheta_{t-1}.$ 
\end{enumerate}


## Inference

- Method 1. Linear Bayes Filtering and Smoothing: `lbe_poisson.cpp`
- Method 2. MCMC with Univariate MH Proposal via Reparameterisation: `mcmc_disturbance_poisson.cpp`
- Method 3. Particle Filtering and Smoothing: `pl_poisson.cpp`
- Method 4. Variational Inference: `vb_poisson.cpp` and `hva_poisson.cpp`


## Demo

- `script_model_sample_data.R`: Sample data provided by Koyama.
- `script_model_country_data.R`: Country level Covid daily new confirmed cases from March 1, 2020 to Dec 1, 2020.

