Mixed-Copula Mixture Model (MCMM)このリポジトリは、論文で提案された混合コピュラ混合モデル（MCMM）のPython実装を提供します。このモデルは、連続、カテゴリ、順序変数が混在するデータセットに対する、柔軟なクラスタリング手法です。主な特徴混合データ型への対応: 連続、カテゴリ、順序変数を統一的なフレームワークで扱います。柔軟な周辺分布: 連続変数には正規分布または外れ値に頑健なスチューデントのt分布を、離散変数にはカテゴリカル分布や累積ロジットモデルを適用します。依存構造のモデル化: ガウシアン・コピュラを用いて、変数間の複雑な依存関係を捉えます。欠損値への頑健性: MAR（Missing At Random）仮定の下で、欠損値を自然に扱うことができます。高速化オプション: 高次元・大規模データに対応するため、計算効率を重視したpairwise（合成尤度）モードや、さらに最適化されたspeedyモードを提供します。並列処理: 計算負荷の高いEステップを並列化し、マルチコアCPUの性能を活かした高速化が可能です。インストールこのパッケージを利用するには、まずリポジトリをクローンし、pip を使ってインストールします。# 1. リポジトリをクローン
git clone [https://github.com/YOUR_USERNAME/pymcmm.git](https://github.com/YOUR_USERNAME/pymcmm.git)
cd pymcmm

# 2. パッケージをインストール
# これにより、mcmmライブラリがあなたのPython環境にインストールされ、
# どのディレクトリからでもインポートできるようになります。
pip install .
開発者向けもし実験コードの実行やライブラリ自体の改修を行う場合は、requirements.txtを使って編集可能モード (-e) でインストールすることをお勧めします。# 編集可能モードでインストールし、実験に必要な追加ライブラリも導入
pip install -r requirements.txt
基本的な使い方mcmmパッケージをインストールすると、どのPythonスクリプトからでもライブラリをインポートして使用できます。import pandas as pd
from mcmm.model import MCMMGaussianCopula

# 1. データの準備
# 例として、顧客データを作成
data = {
    'age': [25, 45, 35, 23, 51, 62, 33, 41],
    'plan': ['Gold', 'Silver', 'Silver', 'Bronze', 'Gold', 'Platinum', 'Silver', 'Gold'],
    'satisfaction': [5, 3, 4, 2, 5, 4, 3, 4]
}
df = pd.DataFrame(data)

# 順序変数のカテゴリ順序を定義
satisfaction_levels = [1, 2, 3, 4, 5]
df['satisfaction'] = pd.Categorical(df['satisfaction'], categories=satisfaction_levels, ordered=True)


# 2. モデルの初期化と学習
# クラスタ数 K=3 でモデルを初期化
mcmm = MCMMGaussianCopula(
    n_components=3,
    cont_marginal='student_t',  # 連続変数はt分布を使用
    estimate_nu=True,           # t分布の自由度も推定
    random_state=42,
    verbose=1,
    n_jobs=-1                   # 利用可能な全てのCPUコアで並列化
)

# カラムの型を指定してモデルを学習
mcmm.fit(
    df,
    cont_cols=['age'],
    cat_cols=['plan'],
    ord_cols=['satisfaction']
)

# 3. 結果の取得
# 各データ点の所属クラスタを取得
labels = mcmm.predict(df)
print("\nクラスタリング結果:")
print(labels)

# 各データ点が各クラスタに所属する確率を取得
probabilities = mcmm.predict_proba(df)
# print("\n所属確率:")
# print(pd.DataFrame(probabilities.round(3), columns=['Cluster 0', 'Cluster 1', 'Cluster 2']))
プロジェクト構成.
├── mcmm/                  # インポート可能なコアライブラリ
│   ├── __init__.py
│   └── model.py           # MCMMのクラス定義
├── experiments/           # 論文の実験を再現するためのスクリプト
│   ├── run_benchmark.py
│   └── utils.py
├── setup.py               # パッケージインストール用の設定ファイル
├── requirements.txt       # 開発環境用の依存関係ファイル
└── README.md              # このファイル
