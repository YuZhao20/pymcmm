# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2024-12-21

### Added
- **E-step完全バッチ処理**: 行ごとのPythonループを排除し、全サンプルを一括処理
  - `compute_e_step_full`: 通常モード用の完全Cython化されたE-step
  - `compute_e_step_speedy`: Speedy mode用の最適化E-step
- **M-step Cython化**: 周辺分布パラメータ更新をCythonでバッチ処理
  - `compute_m_step_marginals_cont`: 連続変数のパラメータ更新
  - `compute_m_step_marginals_cat`: カテゴリ変数の確率更新
  - `compute_u_matrix`: U行列のバッチ計算
- **カテゴリ変数のベクトル化**: インデックスベースの高速処理
  - `compute_cat_u_and_logmarg`: カテゴリ変数のU値一括計算
- データの前処理キャッシュ機構

### Changed
- **v0.2.0から100-500倍の高速化** (データサイズと構成による)
- `model.py`の構造を大幅にリファクタリング
  - `_prepare_data_arrays`: データをNumPy配列に一度だけ変換
  - `_get_cat_probs_list`: カテゴリ確率のリスト取得
- Pure Python fallbackの改善

### Performance
- n=1000, p=5, K=3: 50+ iter/s (v0.2.0比 約10倍)
- n=5000, p=10, K=3: 13+ iter/s (大規模データでも高速)

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
