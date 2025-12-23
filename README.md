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
- **Cython Acceleration**: Optional speedup (up to 35x) with Cython

## Installation

### From GitHub (Recommended)

```bash
pip install git+https://github.com/YuZhao20/pymcmm.git
```

### From PyPI

```bash
pip install pymcmm
```

### With Cython Acceleration

```bash
pip install git+https://github.com/YuZhao20/pymcmm.git
pip install cython
cd /path/to/pymcmm
python setup.py build_ext --inplace
```

Verify acceleration:
```python
import mcmm
mcmm.check_acceleration()
```

## Quick Start

```python
import pandas as pd
from mcmm import MCMMGaussianCopulaSpeedy

df = pd.DataFrame({
    'income': [50000, 60000, 75000, 80000],
    'age': [25, 35, 45, 55],
    'gender': ['M', 'F', 'M', 'F'],
    'satisfaction': [1, 2, 3, 4],
})

model = MCMMGaussianCopulaSpeedy(
    n_components=2,
    cont_marginal='student_t',
    copula_likelihood='full',
    verbose=1
)

model.fit(
    df,
    cont_cols=['income', 'age'],
    cat_cols=['gender'],
    ord_cols=['satisfaction']
)

clusters = model.predict(df)
probabilities = model.predict_proba(df)

print(f"BIC: {model.bic_:.2f}")
print(f"Log-likelihood: {model.loglik_:.2f}")
```

## Cython Acceleration

### Overview

pymcmm includes optional Cython-accelerated implementations that provide significant speedups for computationally intensive operations. **Cython is not required** - the package automatically falls back to pure Python implementations if Cython modules are not available.

### What Gets Accelerated

| Component | Pure Python | Cython | Speedup |
|-----------|------------|--------|---------|
| Normal CDF/PPF | scipy.stats | Custom C implementation | up to 10x |
| Student-t CDF | scipy.stats | Incomplete beta function | up to 15x |
| Bivariate copula density | numpy/scipy | Optimized C loops | up to 20x |
| E-step (batch) | Python loops | Parallel Cython | up to 35x |
| M-step (marginals) | Python loops | Vectorized Cython | up to 25x |
| Weighted correlation | numpy | Optimized pairwise | up to 10x |

### Performance Benchmark

Typical speedup for a dataset with n=500, p=13, K=3:

| Mode | Pure Python | Cython | Speedup |
|------|------------|--------|---------|
| MCMMGaussianCopula | ~65s | ~1.9s | up to 35x |
| MCMMGaussianCopulaSpeedy | ~45s | ~1.5s | up to 30x |

*Note: Actual speedup varies depending on hardware and dataset characteristics.*

### Building Cython Extensions

**Prerequisites:**
- C compiler (gcc, clang, or MSVC)
- Cython >= 0.29
- NumPy development headers

**macOS:**
```bash
xcode-select --install
brew install libomp  # Optional: for parallel processing
pip install cython
python setup.py build_ext --inplace
```

**Linux:**
```bash
sudo apt-get install build-essential python3-dev
pip install cython
python setup.py build_ext --inplace
```

**Windows:**
```bash
pip install cython
python setup.py build_ext --inplace
```

### Verification and Benchmarking

```python
import mcmm

# Check if Cython is enabled
mcmm.check_acceleration()

# Run performance benchmark
mcmm.run_benchmark()
```

### Troubleshooting

If Cython compilation fails:

1. **Missing compiler**: Install build tools for your platform
2. **NumPy headers not found**: Reinstall NumPy with `pip install --force-reinstall numpy`
3. **OpenMP errors on macOS**: The library works without OpenMP; parallel loops will be sequential

The package will always work without Cython - just slower.

## Model Classes

### MCMMGaussianCopula

Full copula model with O(p^2) pairwise dependencies.

```python
from mcmm import MCMMGaussianCopula

model = MCMMGaussianCopula(
    n_components=3,
    cont_marginal='student_t',
    copula_likelihood='full',
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
    speedy_graph='mst',
    corr_subsample=3000,
    n_jobs=-1,
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
| `copula_likelihood` | 'full' | Copula type: 'full' or 'pairwise' |
| `pairwise_weight` | 'abs_rho' | Pairwise weight: 'abs_rho' or 'uniform' |
| `dt_mode` | 'mid' | Discretization mode: 'mid' or 'random' |
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

## Methods

### Fitting

```python
model.fit(df, cont_cols=None, cat_cols=None, ord_cols=None)
```

### Prediction

```python
clusters = model.predict(df)
proba = model.predict_proba(df)
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
from mcmm import MCMMGaussianCopulaSpeedy

df = pd.read_csv('customers.csv')

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

best = min(results, key=lambda x: x['bic'])
print(f"Best K: {best['k']} (BIC: {best['bic']:.2f})")
```

## Scalability Guidelines

| Dataset Size | Recommended Mode | Cython |
|--------------|------------------|--------|
| n < 1,000 | MCMMGaussianCopula | Optional |
| n < 10,000 | MCMMGaussianCopulaSpeedy | Recommended |
| n > 10,000 | MCMMGaussianCopulaSpeedy + n_jobs=-1 | Recommended |

## Citation

If you use this package in your research, please cite:

```bibtex
@software{pymcmm,
  author = {Yu Zhao},
  title = {pymcmm: Mixed-Copula Mixture Model for Python},
  institution = {Tokyo University of Science},
  url = {https://github.com/YuZhao20/pymcmm},
  version = {0.3.0},
  year = {2025}
}
```

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
