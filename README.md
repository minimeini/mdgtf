## Dependencies


- C++11 or newer.
- C++ libraries: [Armadillo](https://arma.sourceforge.net), [boost](https://www.boost.org), [NLopt](https://nlopt.readthedocs.io/en/latest/), and [openMP](https://www.openmp.org).
- R packages: RcppArmadillo, nloptr, 
- Benchmarks: EpiEstim (an R package)

Compile in R:

```{r}
Rcpp::sourceCpp("./export.cpp")
```


## Spatial-Temporal Model Refactoring Strategy


This document outlines a comprehensive strategy to extend the current **temporal-only** DGTF model to support **spatial-temporal** models while maintaining backward compatibility and code reusability.

### Current Model (Temporal)
```
y[t] ~ NBinom(lambda[t], rho)
lambda[t] = x[t]' * varphi + sum(phi[l] * h(psi[t+1-l]) * y[t-l])
psi[t] ~ Normal(psi[t-1], W)
```

### Target Model (Spatial-Temporal)
```
y[s,t] ~ NBinom(lambda[s,t], rho[s])
lambda[s,t] = a[s] + sum(phi[l] * h(psi[s,t+1-l]) * y[s,t-l]) + b * sum(w[s,j] * y[j,t-1])
psi[s,t] ~ Normal(psi[s,t-1], W[s])
```

---

## 1. Design Principles

### 1.1 Core Architectural Goals

1. **Backward Compatibility**: Temporal-only models should continue to work without modification
2. **Separation of Concerns**: Spatial and temporal components should be independently configurable
3. **Template-Based Generics**: Use C++ templates to enable code reuse across dimensions
4. **Minimal Code Duplication**: Share common logic between temporal and spatial-temporal implementations
5. **Performance**: Maintain computational efficiency, especially for large spatial networks
6. **Extensibility**: Easy to add new spatial components (e.g., spatial covariates, different neighborhood structures)

### 1.2 Key Refactoring Strategies

- **Strategy 1**: Introduce dimension-aware data structures
- **Strategy 2**: Create spatial component module
- **Strategy 3**: Refactor distribution classes to support location-specific parameters
- **Strategy 4**: Extend state-space formulation for spatial dimensions
- **Strategy 5**: Add spatial inference algorithms

---

## 2. Architectural Changes

### 2.1 New Directory Structure

```
mdgtf/
├── core/                          # Core temporal components (existing, refactored)
│   ├── ObsDist.hpp
│   ├── LagDist.hpp
│   ├── ErrDist.hpp
│   ├── LinkFunc.hpp
│   ├── GainFunc.hpp
│   └── TransFunc.hpp
│
├── spatial/                       # NEW: Spatial components
│   ├── SpatialStructure.hpp      # Spatial network/weights
│   ├── SpatialParams.hpp         # Location-specific parameters
│   ├── SpatialLag.hpp            # Spatial lag operators
│   └── SpatialPrior.hpp          # Spatial priors (CAR, SAR, etc.)
│
├── models/                        # NEW: Model definitions
│   ├── TemporalModel.hpp         # Refactored temporal model
│   ├── SpatialTemporalModel.hpp  # New spatial-temporal model
│   └── ModelBase.hpp             # Shared base class
│
├── data/                          # NEW: Data structures
│   ├── DataContainer.hpp         # Template for 1D/2D data
│   └── StateArray.hpp            # Template for state storage
│
├── inference/                     # Inference algorithms (existing, refactored)
│   ├── MCMC.hpp
│   ├── VariationalBayes.hpp
│   ├── SequentialMonteCarlo.hpp
│   └── spatial/                  # NEW: Spatial-specific inference
│       ├── SpatialMCMC.hpp
│       └── SpatialVB.hpp
│
└── utils/                         # Utilities
    ├── SpatialUtils.hpp          # NEW: Spatial utility functions
    └── ...
```

---

## 3. Detailed Component Design

### 3.1 Data Container Templates

**File: `data/DataContainer.hpp`**

```cpp
/**
 * @brief Template for handling temporal vs spatial-temporal data
 * @tparam DataType Scalar type (double, int, etc.)
 * @tparam nDim Number of dimensions (1=temporal, 2=spatial-temporal)
 */
template<typename DataType, size_t nDim>
class DataContainer;

// Specialization for temporal data (1D)
template<typename DataType>
class DataContainer<DataType, 1> {
public:
    arma::Col<DataType> data;  // (nT+1) x 1
    unsigned int nT;

    DataType& at(unsigned int t) { return data.at(t); }
    const DataType& at(unsigned int t) const { return data.at(t); }

    void init(unsigned int ntime) {
        nT = ntime;
        data.set_size(nT + 1);
        data.zeros();
    }
};

// Specialization for spatial-temporal data (2D)
template<typename DataType>
class DataContainer<DataType, 2> {
public:
    arma::Mat<DataType> data;  // nS x (nT+1)
    unsigned int nS;  // number of spatial locations
    unsigned int nT;  // number of time points

    DataType& at(unsigned int s, unsigned int t) { return data.at(s, t); }
    const DataType& at(unsigned int s, unsigned int t) const { return data.at(s, t); }

    // Access all locations at time t
    arma::Col<DataType> get_time_slice(unsigned int t) const {
        return data.col(t);
    }

    // Access temporal series for location s
    arma::Col<DataType> get_spatial_slice(unsigned int s) const {
        return data.row(s).t();
    }

    void init(unsigned int nspatial, unsigned int ntime) {
        nS = nspatial;
        nT = ntime;
        data.set_size(nS, nT + 1);
        data.zeros();
    }
};
```

**Benefits:**
- Single interface for both temporal and spatial-temporal data
- Type-safe access patterns
- Compile-time dimension checking

---

### 3.2 Spatial Structure Module

**File: `spatial/SpatialStructure.hpp`**

```cpp
/**
 * @brief Defines spatial neighborhood structure and weights
 */
class SpatialStructure {
public:
    unsigned int nS;              // Number of spatial locations
    arma::sp_mat W;               // nS x nS sparse weight matrix
    arma::uvec neighbors_count;   // Number of neighbors per location
    bool row_normalized = false;  // Whether W is row-normalized

    /**
     * @brief Initialize from adjacency matrix
     */
    void init_from_adjacency(const arma::mat& adjacency, bool normalize = true);

    /**
     * @brief Initialize from edge list
     */
    void init_from_edgelist(const arma::umat& edges,
                            const arma::vec& weights = arma::vec());

    /**
     * @brief Get neighbors of location s
     */
    arma::uvec get_neighbors(unsigned int s) const;

    /**
     * @brief Get weights for location s
     */
    arma::vec get_weights(unsigned int s) const;

    /**
     * @brief Compute spatial lag: W * y
     */
    arma::vec spatial_lag(const arma::vec& y) const;

    /**
     * @brief Compute spatial lag for all time points
     */
    arma::mat spatial_lag(const arma::mat& Y) const;  // Y: nS x nT

    /**
     * @brief Default settings for testing
     */
    static Rcpp::List default_settings();
};

/**
 * @brief Common spatial structures
 */
namespace SpatialStructurePresets {
    // Lattice/grid structure
    SpatialStructure create_lattice(unsigned int nrow, unsigned int ncol,
                                     bool rook = true);

    // Distance-based weights
    SpatialStructure create_distance_weights(const arma::mat& coords,
                                              double threshold);

    // K-nearest neighbors
    SpatialStructure create_knn(const arma::mat& coords, unsigned int k);
}
```

**Key Features:**
- Sparse matrix representation for efficiency
- Multiple initialization methods (adjacency, edge list, presets)
- Built-in spatial lag operators
- Common spatial structures (lattice, distance-based, KNN)

---

### 3.3 Location-Specific Parameters

**File: `spatial/SpatialParams.hpp`**

```cpp
/**
 * @brief Container for location-specific parameters
 * @tparam nDim 1 = shared parameter (scalar), 2 = spatial parameter (vector)
 */
template<size_t nDim>
class SpatialParam;

// Shared parameter (temporal-only or shared across space)
template<>
class SpatialParam<1> {
public:
    double value;

    double get(unsigned int s = 0) const { return value; }
    void set(double val, unsigned int s = 0) { value = val; }
    void init(unsigned int nS = 1) { value = 0.; }
    unsigned int size() const { return 1; }
};

// Location-specific parameter
template<>
class SpatialParam<2> {
public:
    arma::vec values;  // nS x 1
    unsigned int nS;

    double get(unsigned int s) const { return values.at(s); }
    void set(double val, unsigned int s) { values.at(s) = val; }
    void init(unsigned int nspatial) {
        nS = nspatial;
        values.set_size(nS);
        values.zeros();
    }
    unsigned int size() const { return nS; }
};

/**
 * @brief Extended observation distribution with spatial parameters
 */
template<size_t nDim = 1>
class ObsDistSpatial : public ObsDist {
public:
    SpatialParam<nDim> rho;      // Overdispersion: scalar or nS x 1
    SpatialParam<nDim> intercept; // Location intercepts a[s]

    void init(unsigned int nS = 1) {
        ObsDist::init_default();
        rho.init(nS);
        intercept.init(nS);
    }

    // Sample at specific location
    double sample(const double& lambda, unsigned int s = 0) const {
        return ObsDist::sample(lambda, rho.get(s), this->name);
    }

    // Log-likelihood at specific location
    double loglike(const double& y, const double& lambda, unsigned int s = 0) const {
        return ObsDist::loglike(y, this->name, lambda, rho.get(s));
    }
};

/**
 * @brief Extended error distribution with spatial parameters
 */
template<size_t nDim = 1>
class ErrDistSpatial : public ErrDist {
public:
    SpatialParam<nDim> W_variance;  // State evolution variance W[s]

    void init(unsigned int nS = 1, unsigned int nP = 1) {
        ErrDist::init_default();
        W_variance.init(nS);
        for (unsigned int s = 0; s < nS; s++) {
            W_variance.set(this->par1, s);  // Initialize with default
        }
    }

    // Sample innovation for location s
    double sample(unsigned int s = 0) const {
        return R::rnorm(0., std::sqrt(W_variance.get(s)));
    }
};
```

**Benefits:**
- Unified interface for scalar and spatial parameters
- Automatic dimension handling
- Backward compatible (nDim=1 for temporal-only)

---

### 3.4 Spatial Lag Component

**File: `spatial/SpatialLag.hpp`**

```cpp
/**
 * @brief Handles spatial lag calculations: b * sum(w[s,j] * y[j,t])
 */
class SpatialLagComponent {
public:
    double b;                      // Spatial dependence parameter
    SpatialStructure spatial;      // Spatial weights
    bool enabled = false;

    /**
     * @brief Compute spatial lag contribution for location s at time t
     * @param Y nS x (nT+1) matrix of observations
     * @param s Spatial location index
     * @param t Time index
     * @return Spatial lag contribution to lambda[s,t]
     */
    double compute_spatial_lag(const arma::mat& Y, unsigned int s, unsigned int t) const {
        if (!enabled || t == 0) return 0.;

        // Get neighbors of location s
        arma::uvec neighbors = spatial.get_neighbors(s);
        arma::vec weights = spatial.get_weights(s);

        // Compute weighted sum: sum(w[s,j] * y[j,t-1])
        double spatial_effect = 0.;
        for (unsigned int i = 0; i < neighbors.n_elem; i++) {
            unsigned int j = neighbors(i);
            spatial_effect += weights(i) * Y(j, t - 1);
        }

        return b * spatial_effect;
    }

    /**
     * @brief Compute spatial lag for all locations at time t
     */
    arma::vec compute_spatial_lag_all(const arma::mat& Y, unsigned int t) const {
        if (!enabled || t == 0) {
            return arma::zeros(Y.n_rows);
        }

        // Matrix multiplication: b * W * y[t-1]
        arma::vec y_prev = Y.col(t - 1);
        return b * spatial.spatial_lag(y_prev);
    }

    /**
     * @brief Initialize from settings
     */
    void init(const Rcpp::List& settings) {
        if (settings.containsElementNamed("spatial_b")) {
            b = Rcpp::as<double>(settings["spatial_b"]);
            enabled = true;
        } else {
            b = 0.;
            enabled = false;
        }

        if (settings.containsElementNamed("spatial_structure")) {
            Rcpp::List spatial_settings = settings["spatial_structure"];
            // Initialize spatial structure
            spatial.init_from_adjacency(
                Rcpp::as<arma::mat>(spatial_settings["W"])
            );
        }
    }

    static Rcpp::List default_settings() {
        Rcpp::List settings;
        settings["spatial_b"] = 0.0;
        settings["enabled"] = false;
        return settings;
    }
};
```

---

### 3.5 Spatial-Temporal Model Class

**File: `models/SpatialTemporalModel.hpp`**

```cpp
/**
 * @brief Spatial-Temporal DGTF Model
 *
 * Model structure:
 * y[s,t] ~ Nbinom(lambda[s,t], rho[s])
 * lambda[s,t] = a[s] + temporal_component[s,t] + spatial_component[s,t]
 * temporal_component[s,t] = sum(phi[l] * h(psi[s,t+1-l]) * y[s,t-l])
 * spatial_component[s,t] = b * sum(w[s,j] * y[j,t-1])
 * psi[s,t] ~ Normal(psi[s,t-1], W[s])
 */
class SpatialTemporalModel {
public:
    unsigned int nS;  // Number of spatial locations
    unsigned int nT;  // Number of time points
    unsigned int nP;  // State dimension

    // Spatial components
    SpatialStructure spatial_structure;
    SpatialLagComponent spatial_lag;

    // Distribution components (with spatial parameters)
    ObsDistSpatial<2> dobs;    // Location-specific rho[s], a[s]
    ErrDistSpatial<2> derr;    // Location-specific W[s]
    LagDist dlag;              // Temporal lag (shared across space)

    // Regression components (can be location-specific)
    std::vector<Season> seas;  // One per location or shared
    std::vector<ZeroInflation> zero;  // One per location or shared

    // Model specifications (shared across space)
    std::string fsys = "shift";
    std::string ftrans = "sliding";
    std::string flink = "identity";
    std::string fgain = "softplus";

    /**
     * @brief Initialize model
     */
    void init(const Rcpp::List& settings) {
        Rcpp::List model_settings = settings["model"];
        Rcpp::List param_settings = settings["param"];
        Rcpp::List spatial_settings = settings["spatial"];

        // Get dimensions
        nS = Rcpp::as<unsigned int>(spatial_settings["nS"]);
        nT = Rcpp::as<unsigned int>(model_settings["nT"]);

        // Initialize spatial structure
        spatial_structure.init_from_adjacency(
            Rcpp::as<arma::mat>(spatial_settings["W"])
        );

        // Initialize spatial lag
        spatial_lag.init(spatial_settings);

        // Initialize distributions
        dobs.init(nS);
        derr.init(nS, nP);
        dlag.init(dlag.name, dlag.par1, dlag.par2, dlag.truncated);

        // Initialize location-specific parameters
        if (param_settings.containsElementNamed("rho")) {
            arma::vec rho_vec = Rcpp::as<arma::vec>(param_settings["rho"]);
            for (unsigned int s = 0; s < nS; s++) {
                dobs.rho.set(rho_vec(s), s);
            }
        }

        if (param_settings.containsElementNamed("intercept")) {
            arma::vec a_vec = Rcpp::as<arma::vec>(param_settings["intercept"]);
            for (unsigned int s = 0; s < nS; s++) {
                dobs.intercept.set(a_vec(s), s);
            }
        }

        if (param_settings.containsElementNamed("W")) {
            arma::vec W_vec = Rcpp::as<arma::vec>(param_settings["W"]);
            for (unsigned int s = 0; s < nS; s++) {
                derr.W_variance.set(W_vec(s), s);
            }
        }

        // Initialize seasonality (one per location or shared)
        seas.resize(nS);
        for (unsigned int s = 0; s < nS; s++) {
            if (settings.containsElementNamed("season")) {
                seas[s].init(settings["season"]);
            } else {
                seas[s].init_default();
            }
        }

        nP = Model::get_nP(dlag, seas[0].period, seas[0].in_state);
    }

    /**
     * @brief Compute lambda[s,t] for specific location and time
     */
    double compute_lambda(
        const arma::mat& Y,        // nS x (nT+1) observations
        const arma::mat& psi,      // nS x (nT+1) latent states
        const arma::mat& hpsi,     // nS x (nT+1) gain-transformed states
        unsigned int s,
        unsigned int t
    ) const {
        // 1. Location intercept
        double eta = dobs.intercept.get(s);

        // 2. Temporal transfer component
        double temporal_effect = 0.;
        if (t > 0) {
            temporal_effect = TransFunc::func_ft(
                t, Y.row(s).t(), arma::vec(), hpsi.row(s).t(), dlag, ftrans
            );
        }
        eta += temporal_effect;

        // 3. Spatial lag component
        if (spatial_lag.enabled && t > 0) {
            double spatial_effect = spatial_lag.compute_spatial_lag(Y, s, t);
            eta += spatial_effect;
        }

        // 4. Seasonal component (if present)
        if (seas[s].period > 0) {
            eta += arma::dot(seas[s].X.col(t), seas[s].val);
        }

        // 5. Apply link function
        double lambda = LinkFunc::ft2mu(eta, flink);

        return lambda;
    }

    /**
     * @brief Default settings
     */
    static Rcpp::List default_settings() {
        Rcpp::List model_settings = Model::default_settings();

        Rcpp::List spatial_settings;
        spatial_settings["nS"] = 1;  // Single location = temporal only
        spatial_settings["W"] = arma::mat(1, 1, arma::fill::zeros);
        spatial_settings["spatial_b"] = 0.0;

        Rcpp::List settings;
        settings["model"] = model_settings["model"];
        settings["param"] = model_settings["param"];
        settings["spatial"] = spatial_settings;

        return settings;
    }
};
```

---

### 3.6 State Space Simulation

**File: `models/SpatialTemporalModel.hpp` (continued)**

```cpp
/**
 * @brief Simulate spatial-temporal model
 */
class SpatialTemporalStateSpace {
public:
    static void simulate(
        arma::mat& Y,              // nS x (nT+1) observations
        arma::mat& Lambda,         // nS x (nT+1) rates
        arma::mat& Psi,            // nS x (nT+1) latent states
        arma::mat& Hpsi,           // nS x (nT+1) gain-transformed states
        SpatialTemporalModel& model,
        const arma::vec& y0,       // nS x 1 initial observations
        const arma::vec& psi0      // nS x 1 initial states
    ) {
        unsigned int nS = model.nS;
        unsigned int nT = model.nT;

        // Initialize arrays
        Y.set_size(nS, nT + 1);
        Lambda.set_size(nS, nT + 1);
        Psi.set_size(nS, nT + 1);
        Hpsi.set_size(nS, nT + 1);

        Y.zeros();
        Lambda.zeros();
        Psi.zeros();
        Hpsi.zeros();

        // Set initial values
        Y.col(0) = y0;
        Psi.col(0) = psi0;
        Hpsi.col(0) = GainFunc::psi2hpsi(psi0, model.fgain);

        // Simulate forward in time
        for (unsigned int t = 1; t <= nT; t++) {
            // For each spatial location
            for (unsigned int s = 0; s < nS; s++) {
                // 1. Evolve latent state: psi[s,t] ~ Normal(psi[s,t-1], W[s])
                double psi_prev = Psi(s, t - 1);
                double W_s = model.derr.W_variance.get(s);
                Psi(s, t) = R::rnorm(psi_prev, std::sqrt(W_s));

                // 2. Apply gain function
                Hpsi(s, t) = GainFunc::psi2hpsi(Psi(s, t), model.fgain);
            }

            // Compute observations for all locations
            for (unsigned int s = 0; s < nS; s++) {
                // 3. Compute lambda[s,t]
                Lambda(s, t) = model.compute_lambda(Y, Psi, Hpsi, s, t);

                // 4. Sample observation: y[s,t] ~ Nbinom(lambda[s,t], rho[s])
                Y(s, t) = model.dobs.sample(Lambda(s, t), s);
            }
        }
    }
};
```

---

## 4. Refactoring Existing Components

### 4.1 Make TransFunc Spatial-Aware

**Current signature:**
```cpp
static double func_ft(
    unsigned int t,
    const arma::vec &y,        // (nT+1) x 1
    const arma::vec &ft,
    const arma::vec &hpsi,     // (nT+1) x 1
    const LagDist &dlag,
    const std::string &ftrans
);
```

**Add overload for spatial-temporal:**
```cpp
// Location-specific temporal transfer
static double func_ft_spatial(
    unsigned int s,            // spatial location
    unsigned int t,            // time
    const arma::mat &Y,        // nS x (nT+1)
    const arma::mat &Hpsi,     // nS x (nT+1)
    const LagDist &dlag,
    const std::string &ftrans
) {
    // Extract time series for location s
    arma::vec y_s = Y.row(s).t();
    arma::vec hpsi_s = Hpsi.row(s).t();

    // Use existing temporal function
    return func_ft(t, y_s, arma::vec(), hpsi_s, dlag, ftrans);
}
```

### 4.2 Extend Inference Algorithms

**File: `inference/spatial/SpatialMCMC.hpp`**

Key modifications needed:
1. **State sampling**: Sample `psi[s,t]` for each location using forward-filtering backward-sampling
2. **Parameter updates**: Update `rho[s]`, `a[s]`, `W[s]` with location-specific priors
3. **Spatial parameter**: Update `b` (spatial dependence) using Metropolis-Hastings
4. **Computational efficiency**: Exploit spatial sparsity in FFBS

```cpp
class SpatialMCMC {
public:
    /**
     * @brief MCMC inference for spatial-temporal model
     */
    static Rcpp::List infer(
        SpatialTemporalModel& model,
        const arma::mat& Y,              // nS x (nT+1)
        const Rcpp::List& mcmc_settings
    ) {
        unsigned int niter = Rcpp::as<unsigned int>(mcmc_settings["niter"]);
        unsigned int nburn = Rcpp::as<unsigned int>(mcmc_settings["nburn"]);

        // Storage for MCMC samples
        arma::mat Psi_samples(niter, model.nS * (model.nT + 1));
        arma::mat rho_samples(niter, model.nS);
        arma::mat a_samples(niter, model.nS);
        arma::mat W_samples(niter, model.nS);
        arma::vec b_samples(niter);

        // Initialize state
        arma::mat Psi_current(model.nS, model.nT + 1, arma::fill::zeros);

        for (unsigned int iter = 0; iter < niter; iter++) {
            // 1. Update latent states psi[s,t] for each location
            for (unsigned int s = 0; s < model.nS; s++) {
                update_psi_location(Psi_current, Y, model, s);
            }

            // 2. Update observation parameters rho[s], a[s]
            for (unsigned int s = 0; s < model.nS; s++) {
                update_obs_params_location(model, Y, Psi_current, s);
            }

            // 3. Update state variance W[s]
            for (unsigned int s = 0; s < model.nS; s++) {
                update_W_location(model, Psi_current, s);
            }

            // 4. Update spatial dependence parameter b
            if (model.spatial_lag.enabled) {
                update_spatial_b(model, Y, Psi_current);
            }

            // 5. Update temporal lag parameters (shared across space)
            update_lag_params(model, Y, Psi_current);

            // Store samples
            if (iter >= nburn) {
                // Store samples
            }
        }

        return Rcpp::List::create(
            Rcpp::Named("psi") = Psi_samples,
            Rcpp::Named("rho") = rho_samples,
            Rcpp::Named("a") = a_samples,
            Rcpp::Named("W") = W_samples,
            Rcpp::Named("b") = b_samples
        );
    }

private:
    /**
     * @brief Update psi[s,] for location s using FFBS
     */
    static void update_psi_location(
        arma::mat& Psi,
        const arma::mat& Y,
        const SpatialTemporalModel& model,
        unsigned int s
    ) {
        // Extract time series for location s
        arma::vec y_s = Y.row(s).t();
        arma::vec psi_s = Psi.row(s).t();

        // Run FFBS for this location
        // (Can reuse existing temporal FFBS code)
        // ...

        // Update psi for location s
        Psi.row(s) = psi_s.t();
    }

    /**
     * @brief Update spatial parameter b using Metropolis-Hastings
     */
    static void update_spatial_b(
        SpatialTemporalModel& model,
        const arma::mat& Y,
        const arma::mat& Psi
    ) {
        double b_current = model.spatial_lag.b;
        double b_proposal = b_current + R::rnorm(0., 0.1);  // Proposal

        // Compute log-likelihood ratio
        double loglik_current = compute_loglik_spatial(model, Y, Psi, b_current);
        double loglik_proposal = compute_loglik_spatial(model, Y, Psi, b_proposal);

        // Metropolis acceptance
        double log_alpha = loglik_proposal - loglik_current;
        if (std::log(R::runif(0., 1.)) < log_alpha) {
            model.spatial_lag.b = b_proposal;
        }
    }

    static double compute_loglik_spatial(
        const SpatialTemporalModel& model,
        const arma::mat& Y,
        const arma::mat& Psi,
        double b
    ) {
        // Compute log-likelihood with given b
        // ...
        return 0.;
    }
};
```

---

## 5. Backward Compatibility Strategy

### 5.1 Automatic Dimension Detection

```cpp
/**
 * @brief Auto-detect whether model is temporal or spatial-temporal
 */
namespace ModelFactory {
    /**
     * @brief Create appropriate model from settings
     */
    std::unique_ptr<ModelBase> create_model(const Rcpp::List& settings) {
        if (settings.containsElementNamed("spatial")) {
            Rcpp::List spatial_settings = settings["spatial"];
            unsigned int nS = Rcpp::as<unsigned int>(spatial_settings["nS"]);

            if (nS > 1) {
                // Spatial-temporal model
                return std::make_unique<SpatialTemporalModel>();
            }
        }

        // Default: temporal-only model
        return std::make_unique<Model>();
    }
}
```

### 5.2 Wrapper Functions

**File: `export.cpp`**

```cpp
// Existing function signature unchanged
Rcpp::List dgtf_simulate(
    const Rcpp::List& settings,
    unsigned int ntime,
    const Rcpp::NumericVector& y0,
    const Rcpp::NumericVector& z = Rcpp::NumericVector()
) {
    // Auto-detect model type
    if (settings.containsElementNamed("spatial")) {
        return dgtf_simulate_spatial(settings, ntime, y0, z);
    } else {
        return dgtf_simulate_temporal(settings, ntime, y0, z);
    }
}

// New spatial-temporal simulation
Rcpp::List dgtf_simulate_spatial(
    const Rcpp::List& settings,
    unsigned int ntime,
    const Rcpp::NumericVector& y0,
    const Rcpp::NumericVector& z
) {
    SpatialTemporalModel model(settings);

    // ... spatial-temporal simulation

    return Rcpp::List::create(
        Rcpp::Named("y") = Y,
        Rcpp::Named("lambda") = Lambda,
        Rcpp::Named("psi") = Psi
    );
}
```

---

## 6. Implementation Roadmap

### Phase 1: Foundation (Weeks 1-2)
- [ ] Create `data/` directory and implement `DataContainer` templates
- [ ] Create `spatial/` directory structure
- [ ] Implement `SpatialStructure` class
- [ ] Implement `SpatialParam` templates
- [ ] Write unit tests for new data structures

### Phase 2: Spatial Components (Weeks 3-4)
- [ ] Implement `SpatialLagComponent`
- [ ] Implement `ObsDistSpatial` and `ErrDistSpatial`
- [ ] Add spatial structure presets (lattice, distance, KNN)
- [ ] Write unit tests for spatial components

### Phase 3: Model Integration (Weeks 5-6)
- [ ] Create `models/` directory
- [ ] Refactor existing `Model` into `TemporalModel`
- [ ] Implement `SpatialTemporalModel`
- [ ] Implement `SpatialTemporalStateSpace::simulate()`
- [ ] Add model factory for automatic detection
- [ ] Write integration tests

### Phase 4: Inference (Weeks 7-9)
- [ ] Implement `SpatialMCMC` with FFBS for spatial locations
- [ ] Add spatial parameter updates (b, rho[s], a[s], W[s])
- [ ] Implement spatial variational Bayes (optional)
- [ ] Optimize for sparse spatial matrices
- [ ] Write inference tests

### Phase 5: R Interface (Week 10)
- [ ] Add R wrapper functions for spatial-temporal models
- [ ] Update `dgtf_simulate()` to handle spatial data
- [ ] Update `dgtf_infer()` to handle spatial data
- [ ] Add `dgtf_default_spatial_settings()` helper
- [ ] Write R examples and vignettes

### Phase 6: Testing & Documentation (Weeks 11-12)
- [ ] Comprehensive unit tests for all components
- [ ] Integration tests comparing temporal-only vs spatial-temporal
- [ ] Benchmark performance (temporal vs spatial-temporal)
- [ ] Write API documentation
- [ ] Create example workflows
- [ ] Validate against known spatial-temporal datasets

---

## 7. Example Usage

### 7.1 Temporal-Only Model (Unchanged)

```r
# Existing code works without modification
library(mdgtf)

settings <- dgtf_default_model()
settings$model$obs_dist <- "nbinom"
settings$param$obs <- c(0, 30)  # mu0=0, rho=30

result <- dgtf_simulate(settings, ntime = 100, y0 = 1)
```

### 7.2 Spatial-Temporal Model (New)

```r
library(mdgtf)

# Define spatial structure (e.g., 10x10 lattice)
W <- create_lattice_weights(nrow = 10, ncol = 10, rook = TRUE)
nS <- nrow(W)

# Model settings
settings <- dgtf_default_spatial_settings()
settings$model$obs_dist <- "nbinom"
settings$model$link_func <- "identity"
settings$model$gain_func <- "softplus"

# Spatial settings
settings$spatial$nS <- nS
settings$spatial$W <- W
settings$spatial$spatial_b <- 0.3  # Spatial dependence

# Location-specific parameters
settings$param$rho <- rep(30, nS)           # Overdispersion per location
settings$param$intercept <- rnorm(nS, 2, 0.5)  # Random intercepts a[s]
settings$param$W <- rep(0.1, nS)            # State variance per location

# Simulate
y0 <- rpois(nS, 5)
result <- dgtf_simulate(settings, ntime = 100, y0 = y0)

# result$y is now nS x (nT+1) matrix
dim(result$y)  # [1] 100 101

# Inference
fit <- dgtf_infer(
    model_settings = settings,
    y = result$y,
    method = "mcmc",
    method_settings = list(niter = 5000, nburn = 1000)
)

# Extract posterior samples
posterior_b <- fit$samples$b              # Spatial dependence
posterior_rho <- fit$samples$rho          # nS location-specific rho
posterior_psi <- fit$samples$psi          # nS x nT latent states
```

---

## 8. Performance Considerations

### 8.1 Computational Complexity

| Operation | Temporal | Spatial-Temporal | Speedup Strategy |
|-----------|----------|------------------|------------------|
| State simulation | O(nT) | O(nS × nT) | Parallelize over locations |
| MCMC iteration | O(nT) | O(nS × nT + nS × k) | Sparse matrix operations |
| Likelihood evaluation | O(nT) | O(nS × nT) | Vectorization |
| Spatial lag | - | O(nS × k) | Sparse matrices (k = avg neighbors) |

### 8.2 Optimization Strategies

1. **Sparse Matrices**: Use `arma::sp_mat` for spatial weights
2. **Parallel Processing**: Use OpenMP for location-wise operations
   ```cpp
   #pragma omp parallel for
   for (unsigned int s = 0; s < nS; s++) {
       // Update location s independently
   }
   ```
3. **Vectorization**: Batch operations across locations where possible
4. **Cache Efficiency**: Store spatial neighbors in contiguous memory

### 8.3 Memory Usage

- **Temporal**: O(nT) for observations, states
- **Spatial-Temporal**: O(nS × nT) for observations, states; O(nS × k) for sparse weights
- **MCMC Storage**: Consider thinning for large nS × nT

---

## 9. Testing Strategy

### 9.1 Unit Tests

- [ ] `DataContainer` dimension handling
- [ ] `SpatialStructure` neighbor lookups
- [ ] `SpatialParam` get/set operations
- [ ] `SpatialLagComponent` calculations
- [ ] Location-specific parameter updates

### 9.2 Integration Tests

- [ ] Temporal-only model produces same results as before
- [ ] Spatial-temporal model with nS=1 matches temporal model
- [ ] Spatial-temporal model with b=0 has no spatial dependence
- [ ] MCMC converges to known parameters in simulation study

### 9.3 Validation Tests

- [ ] Compare against Stan spatial-temporal model
- [ ] Reproduce published spatial-temporal disease models
- [ ] Verify MCMC diagnostics (Rhat, ESS)

---

## 10. Future Extensions

### 10.1 Advanced Spatial Models

1. **Spatial Random Effects**:
   ```
   a[s] ~ CAR(phi, tau^2, W)  # Conditional autoregressive prior
   ```

2. **Space-Time Interaction**:
   ```
   psi[s,t] ~ Normal(alpha * psi[s,t-1] + beta * sum(w[s,j] * psi[j,t-1]), W)
   ```

3. **Spatial Covariates**:
   ```
   lambda[s,t] = a[s] + X[s,t]' * beta + ...
   ```

### 10.2 Computational Enhancements

- GPU acceleration for large spatial networks
- Approximate Bayesian inference (INLA-style)
- Distributed computing for massive spatial-temporal datasets

---

## 11. Summary

This refactoring strategy provides:

1. **Clean separation** between temporal and spatial-temporal components
2. **Backward compatibility** for existing temporal-only code
3. **Extensible architecture** for future spatial model variants
4. **Performance optimization** through sparse matrices and parallelization
5. **Comprehensive testing** to ensure correctness

The template-based design allows compile-time optimization while maintaining code clarity. The modular structure makes it easy to add new spatial components or extend existing ones.

### Key Files to Create

| Priority | File | Purpose |
|----------|------|---------|
| 🔴 High | `data/DataContainer.hpp` | Template for 1D/2D data |
| 🔴 High | `spatial/SpatialStructure.hpp` | Spatial weights and neighbors |
| 🔴 High | `spatial/SpatialParams.hpp` | Location-specific parameters |
| 🟡 Medium | `spatial/SpatialLag.hpp` | Spatial lag calculations |
| 🟡 Medium | `models/SpatialTemporalModel.hpp` | Main ST model class |
| 🟢 Low | `inference/spatial/SpatialMCMC.hpp` | Spatial MCMC inference |

### Recommended Next Steps

1. **Review** this strategy with team/advisor
2. **Prototype** `DataContainer` and `SpatialStructure`
3. **Validate** design with simple 3x3 lattice example
4. **Iterate** based on performance and usability feedback
