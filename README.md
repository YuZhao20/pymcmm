# pymcmm

**Mixed-Copula Mixture Model (MCMM)** for clustering datasets with mixed continuous, categorical, and ordinal data types.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Features

- **Mixed Data Types**: Handle continuous, categorical, and ordinal variables simultaneously
- **Gaussian Copula**: Capture complex dependencies between variables
- **Missing Values**: Native support for missing data
- **Student-t Marginals**: Robust to outliers with automatic degree of freedom estimation
- **Speedy Mode**: Efficient computation for large datasets using sparse MST/KNN graphs
- **Cython Acceleration**: Optional 35x speedup with Cython (v0.2.0+)

## Installation

### Basic Installation

```bash
pip install pymcmm
```

### With Cython Acceleration (Recommended)

```bash
pip install pymcmm
pip install cython
cd /path/to/pymcmm
python setup.py build_ext --inplace
```

Verify acceleration:
```python
import mcmm
mcmm.check_acceleration()
# ✓ Cython acceleration is enabled (35x faster)
```

## Quick Start

```python
import pandas as pd
from mcmm import MCMMGaussianCopulaSpeedy

# Prepare your data
df = pd.DataFrame({
    'income': [50000, 60000, 75000, ...],      # continuous
    'age': [25, 35, 45, ...],                   # continuous
    'gender': ['M', 'F', 'M', ...],             # categorical
    'satisfaction': [1, 2, 3, 4, 5, ...],       # ordinal
})

# Create and fit model
model = MCMMGaussianCopulaSpeedy(
    n_components=3,           # number of clusters
    cont_marginal='student_t', # robust to outliers
    copula_likelihood='pairwise',
    verbose=1
)

model.fit(
    df,
    cont_cols=['income', 'age'],
    cat_cols=['gender'],
    ord_cols=['satisfaction']
)

# Predict clusters
clusters = model.predict(df)
probabilities = model.predict_proba(df)

# Model evaluation
print(f"BIC: {model.bic_:.2f}")
print(f"Log-likelihood: {model.loglik_:.2f}")
```

## Model Classes

### MCMMGaussianCopula

Full copula model with O(p²) pairwise dependencies.

```python
from mcmm import MCMMGaussianCopula

model = MCMMGaussianCopula(
    n_components=3,
    cont_marginal='student_t',  # 'gaussian' or 'student_t'
    copula_likelihood='pairwise',  # 'full' or 'pairwise'
    max_iter=100,
    verbose=1
)
```

### MCMMGaussianCopulaSpeedy

Optimized for large datasets using sparse graph approximation.

```python
from mcmm import MCMMGaussianCopulaSpeedy

model = MCMMGaussianCopulaSpeedy(
    n_components=3,
    cont_marginal='student_t',
    speedy_graph='mst',        # 'mst' or 'knn'
    corr_subsample=3000,       # subsample size for correlation
    n_jobs=-1,                 # parallel processing
    verbose=1
)
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_components` | 3 | Number of clusters |
| `cont_marginal` | 'student_t' | Marginal for continuous vars: 'gaussian' or 'student_t' |
| `t_nu` | 5.0 | Initial degrees of freedom for Student-t |
| `estimate_nu` | True | Estimate nu from data |
| `ord_marginal` | 'cumlogit' | Ordinal marginal: 'cumlogit' or 'freq' |
| `copula_likelihood` | 'pairwise' | Copula type: 'full' or 'pairwise' |
| `pairwise_weight` | 'abs_rho' | Pairwise weight: 'abs_rho' or 'uniform' |
| `shrink_lambda` | 0.05 | Correlation matrix shrinkage |
| `max_iter` | 100 | Maximum EM iterations |
| `tol` | 1e-4 | Convergence tolerance |
| `n_jobs` | 1 | Number of parallel jobs (-1 for all cores) |
| `random_state` | None | Random seed for reproducibility |
| `verbose` | 0 | Verbosity level |

### Speedy Mode Additional Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `speedy_graph` | 'mst' | Graph type: 'mst' or 'knn' |
| `speedy_k_per_node` | 3 | K for KNN graph |
| `corr_subsample` | 3000 | Subsample size for correlation estimation |
| `e_step_batch` | 4096 | Batch size for E-step |

## Performance

### Speed Comparison (n=500, p=13, K=3)

| Version | Time | Speedup |
|---------|------|---------|
| Pure Python | 65.4s | 1x |
| Cython | 1.9s | **35x** |

### Scalability Guidelines

| Dataset Size | Recommended Mode |
|--------------|------------------|
| n < 1,000 | MCMMGaussianCopula |
| n < 10,000 | MCMMGaussianCopulaSpeedy |
| n > 10,000 | MCMMGaussianCopulaSpeedy + n_jobs=-1 |

## Methods

### Fitting

```python
model.fit(df, cont_cols=None, cat_cols=None, ord_cols=None)
```

### Prediction

```python
# Hard cluster assignment
clusters = model.predict(df)

# Cluster probabilities
proba = model.predict_proba(df)

# Per-sample log-likelihood
log_lik = model.score_samples(df)
```

### Outlier Detection

```python
is_outlier, scores, threshold = model.detect_outliers(df, q=1.0)
```

## Attributes (after fitting)

| Attribute | Description |
|-----------|-------------|
| `pi_` | Cluster mixing proportions (K,) |
| `mu_` | Cluster means for continuous vars (K, p_cont) |
| `sig_` | Cluster stds for continuous vars (K, p_cont) |
| `R_` | Correlation matrices (K, p, p) |
| `fitted_nu_` | Estimated degrees of freedom |
| `loglik_` | Final log-likelihood |
| `bic_` | Bayesian Information Criterion |
| `history_` | Log-likelihood history |

## Example: Customer Segmentation

```python
import pandas as pd
import numpy as np
from mcmm import MCMMGaussianCopulaSpeedy

# Load customer data
df = pd.read_csv('customers.csv')

# Handle missing values (MCMM supports them natively)
# No imputation needed!

# Fit model with multiple K values
results = []
for k in range(2, 8):
    model = MCMMGaussianCopulaSpeedy(
        n_components=k,
        random_state=42,
        verbose=0
    )
    model.fit(df, 
              cont_cols=['income', 'age', 'spending'],
              cat_cols=['region', 'gender'],
              ord_cols=['satisfaction'])
    results.append({'k': k, 'bic': model.bic_, 'loglik': model.loglik_})

# Select best K by BIC
best = min(results, key=lambda x: x['bic'])
print(f"Best K: {best['k']} (BIC: {best['bic']:.2f})")
```

## Changelog

### v0.2.0 (2024-12)
- **NEW**: Cython acceleration (35x speedup)
- **NEW**: Automatic fallback to pure Python when Cython unavailable
- **NEW**: `check_acceleration()` and `run_benchmark()` functions
- Improved numerical stability in copula calculations
- Fixed edge cases in Student-t CDF computation

### v0.1.0 (2024-10)
- Initial release
- MCMMGaussianCopula and MCMMGaussianCopulaSpeedy
- Support for mixed continuous, categorical, and ordinal data
- Missing value handling


## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
