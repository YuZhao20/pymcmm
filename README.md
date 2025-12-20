# pymcmm

混合コピュラ混合モデル（Mixed-Copula Mixture Model, MCMM）のPython実装

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## 概要

pymcmmは、連続変数、カテゴリ変数、順序変数を含む混合データセットのクラスタリングを可能にする混合コピュラ混合モデル（MCMM）の実装です。ガウシアンコピュラを使用し、Student-t分布とガウス分布の両方をマージナル分布としてサポートしています。

## 主な特徴

- **混合データ型のサポート**: 連続、カテゴリ、順序変数を同時に処理
- **Cython高速化**: オプショナルなCython実装により最大35倍の高速化（v0.2.0）
- **柔軟なマージナル分布**: Student-t分布とガウス分布の選択が可能
- **2つのコピュラモード**: 完全尤度とペアワイズ尤度
- **スピーディモード**: 大規模データセット向けの最適化モード
- **数値的安定性**: 堅牢な数値計算とエラーハンドリング

## インストール

### 基本的なインストール

```bash
pip install pymcmm
```

### Cython高速化付きインストール（推奨）

```bash
pip install pymcmm[cython]
```

または

```bash
pip install cython
pip install pymcmm
```

### 開発モードでのインストール

```bash
git clone https://github.com/YuZhao20/pymcmm.git
cd pymcmm
pip install -e .
```

Cython高速化を有効にする場合:

```bash
pip install -e ".[cython]"
```

## クイックスタート

```python
import pandas as pd
from mcmm import MCMMGaussianCopula, check_acceleration

# Cython高速化の状態を確認
accel_info = check_acceleration()
print(f"Cython acceleration: {accel_info['available']}")

# データの準備
df = pd.DataFrame({
    'continuous1': [1.0, 2.0, 3.0, 4.0, 5.0],
    'continuous2': [2.0, 3.0, 4.0, 5.0, 6.0],
    'categorical': ['A', 'B', 'A', 'B', 'A'],
    'ordinal': [1, 2, 3, 2, 1]
})

# モデルの作成と学習
model = MCMMGaussianCopula(
    n_components=2,
    cont_marginal='student_t',
    copula_likelihood='pairwise',
    max_iter=100,
    random_state=42
)

model.fit(df, 
          cont_cols=['continuous1', 'continuous2'],
          cat_cols=['categorical'],
          ord_cols=['ordinal'])

# 予測
labels = model.predict(df)
probabilities = model.predict_proba(df)

print(f"Log-likelihood: {model.loglik_:.3f}")
print(f"BIC: {model.bic_:.3f}")
```

## API リファレンス

### MCMMGaussianCopula

主要なクラス。混合データのクラスタリングを実行します。

#### パラメータ

- `n_components` (int): クラスタ数（デフォルト: 3）
- `max_iter` (int): 最大反復回数（デフォルト: 100）
- `tol` (float): 収束判定の許容誤差（デフォルト: 1e-4）
- `cont_marginal` (str): 連続変数のマージナル分布。`'gaussian'` または `'student_t'`（デフォルト: `'student_t'`）
- `t_nu` (float): Student-t分布の自由度（デフォルト: 5.0）
- `estimate_nu` (bool): 自由度を推定するか（デフォルト: True）
- `ord_marginal` (str): 順序変数のマージナル分布。`'freq'` または `'cumlogit'`（デフォルト: `'cumlogit'`）
- `copula_likelihood` (str): コピュラ尤度の計算方法。`'full'` または `'pairwise'`（デフォルト: `'full'`）
- `pairwise_weight` (str): ペアワイズ重み。`'uniform'` または `'abs_rho'`（デフォルト: `'abs_rho'`）
- `shrink_lambda` (float): 相関行列の縮小パラメータ（デフォルト: 0.05）
- `random_state` (int): 乱数シード（デフォルト: None）
- `verbose` (int): 詳細度（デフォルト: 0）
- `n_jobs` (int): 並列処理のジョブ数（デフォルト: 1）

#### メソッド

- `fit(df, cont_cols=None, cat_cols=None, ord_cols=None)`: モデルを学習
- `predict(df)`: クラスタラベルを予測
- `predict_proba(df)`: クラスタ確率を予測
- `score_samples(df)`: 各サンプルの対数尤度を計算
- `detect_outliers(df, q=1.0)`: 外れ値を検出

### MCMMGaussianCopulaSpeedy

大規模データセット向けの高速化バージョン。

追加パラメータ:
- `speedy_graph` (str): グラフタイプ。`'mst'` または `'knn'`（デフォルト: `'mst'`）
- `speedy_k_per_node` (int): KNNグラフの各ノードあたりのエッジ数（デフォルト: 3）
- `corr_subsample` (int): 相関計算のサブサンプルサイズ（デフォルト: 3000）
- `e_step_batch` (int): Eステップのバッチサイズ（デフォルト: 4096）

### check_acceleration()

Cython高速化の利用可能性を確認します。

```python
from mcmm import check_acceleration

info = check_acceleration()
print(info)
# {'available': True, 'version': '3.0.0', 'functions': [...]}
```

## パフォーマンス

v0.2.0では、Cython実装により最大35倍の高速化を実現しています。Cythonが利用可能な場合、以下の関数が自動的に高速化されます:

- `log_gaussian_copula_density_full`: 完全ガウシアンコピュラ密度の対数計算
- `log_bivariate_gaussian_copula`: 二変量ガウシアンコピュラ密度の対数計算
- `pairwise_weighted_corr_fast`: ペアワイズ重み付き相関計算

Cythonが利用できない場合でも、純Python実装に自動的にフォールバックするため、機能は完全に利用可能です。

## ライセンス

MIT License - 詳細は [LICENSE](LICENSE) を参照してください。

## 著者

Yu Zhao (yu.zhao@rs.tus.ac.jp)

## リポジトリ

https://github.com/YuZhao20/pymcmm

## 変更履歴

詳細は [CHANGELOG.md](CHANGELOG.md) を参照してください。
