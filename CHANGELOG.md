# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2024-12-XX

### Added
- Cython implementation for core mathematical functions (`_fast_core.pyx`)
  - Provides up to 35x speedup for computationally intensive operations
  - Automatic fallback to pure Python when Cython is unavailable
- `check_acceleration()` function to verify Cython acceleration status
- Comprehensive test suite (`tests/test_mcmm.py`)
- Modern packaging configuration (`pyproject.toml`)
- GitHub push instructions documentation

### Changed
- Improved numerical stability in correlation matrix computations
- Enhanced error handling for edge cases
- Updated `setup.py` to support optional Cython compilation
- Improved documentation in README

### Performance
- Significant speedup (up to 35x) for core mathematical operations when Cython is available
- Reduced memory footprint in correlation matrix calculations

## [0.1.1] - Previous Version

### Added
- Initial release of MCMM with Gaussian Copula
- Support for mixed data types (continuous, categorical, ordinal)
- Student-t and Gaussian marginal distributions
- Full and pairwise copula likelihood modes
- Speedy mode for large-scale datasets

