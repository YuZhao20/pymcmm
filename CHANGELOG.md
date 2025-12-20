# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2024-12-20

### Added
- **Cython acceleration**: 35x speedup for core computations
  - Custom implementations of Student-t CDF/PDF (replacing scipy.stats)
  - Optimized bivariate Gaussian copula density
  - Vectorized logsumexp
  - Fast weighted correlation matrix computation
- Automatic fallback to pure Python when Cython is not available
- `mcmm.check_acceleration()` function to verify Cython status
- `mcmm.run_benchmark()` function for performance testing
- Comprehensive test suite

### Changed
- Improved numerical stability in norm_cdf using erf-based implementation
- Better handling of extreme values in Student-t distribution
- Updated documentation with performance benchmarks

### Fixed
- Edge cases in Student-t CDF for very small degrees of freedom (nu < 3)
- Correlation matrix symmetry enforcement
- Memory efficiency in M-step copula computation

## [0.1.0] - 2024-10-08

### Added
- Initial release
- `MCMMGaussianCopula`: Full copula mixture model
- `MCMMGaussianCopulaSpeedy`: Optimized version with sparse graph approximation
- Support for mixed data types:
  - Continuous variables (Gaussian or Student-t marginals)
  - Categorical variables
  - Ordinal variables (cumulative logit or frequency-based)
- Native missing value handling
- MST and KNN graph options for Speedy mode
- BIC and cBIC model selection criteria
- Outlier detection via log-likelihood scores
- Parallel processing support via joblib
