# =========================================================================
# Utilities for MCMM Benchmark (complete)
# - imputing & evaluation helpers
# - baseline runners (KMeans / K-Prototypes)
# - scenario generators (9 kinds)
# - plotting & scalability helpers
# =========================================================================

import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.api.types import CategoricalDtype
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Optional: K-Prototypes (kmodes)
try:
    from kmodes.kprototypes import KPrototypes

    HAS_KPROTO = True
except Exception:
    HAS_KPROTO = False


# =========================================================================
# ====== Core helpers ======================================================
# =========================================================================

def _impute_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute numeric with mean, and non-numeric with mode (safe handling for all-NaN).
    """
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_numeric_dtype(out[c]):
            out[c] = out[c].fillna(out[c].mean())
        else:
            mode_val = out[c].mode(dropna=True)
            if len(mode_val) > 0:
                out[c] = out[c].fillna(mode_val.iloc[0])
            else:
                out[c] = out[c].fillna("")  # fallback if all-NaN
    return out


def _evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    """
    Return (ARI, NMI). If y_pred contains NaN (shouldn't), sklearn will raise.
    """
    return (
        adjusted_rand_score(y_true, y_pred),
        normalized_mutual_info_score(y_true, y_pred),
    )


def _prep_kmeans_space(df_imp: pd.DataFrame) -> np.ndarray:
    """
    Build a numeric embedding for KMeans/Silhouette:
    - continuous: StandardScaler
    - discrete: OneHotEncoder
    """
    cont_cols = [c for c in df_imp.columns if pd.api.types.is_numeric_dtype(df_imp[c])]
    disc_cols = [c for c in df_imp.columns if not pd.api.types.is_numeric_dtype(df_imp[c])]
    xs = []
    if cont_cols:
        Xc = df_imp[cont_cols].to_numpy(float)
        xs.append(StandardScaler().fit_transform(Xc))
    if disc_cols:
        Xd = OneHotEncoder(sparse_output=False, handle_unknown="ignore").fit_transform(
            df_imp[disc_cols].astype("category")
        )
        xs.append(Xd)
    if not xs:
        xs = [np.zeros((len(df_imp), 1))]
    return np.hstack(xs)


def safe_silhouette(X: np.ndarray, labels: np.ndarray) -> float:
    """
    Silhouette is defined only if 2 <= n_labels <= n_samples-1 and each cluster has >=2 points.
    """
    try:
        n_labels = len(np.unique(labels))
        if n_labels < 2:
            return np.nan
        counts = pd.Series(labels).value_counts().values
        if (counts < 2).any():
            return np.nan
        return float(silhouette_score(X, labels, metric="euclidean"))
    except Exception:
        return np.nan


# =========================================================================
# ====== Baselines =========================================================
# =========================================================================

def run_kmeans(df: pd.DataFrame, k: int):
    """
    Returns: labels, silhouette, inertia, runtime
    """
    df_imp = _impute_dataframe(df)
    X = _prep_kmeans_space(df_imp)
    t0 = time.time()
    km = KMeans(k, n_init=10, random_state=42)
    labels = km.fit_predict(X)
    t1 = time.time()
    sil = safe_silhouette(X, labels)
    inertia = float(getattr(km, "inertia_", np.nan))
    return labels, sil, inertia, t1 - t0


def run_kprototypes(df: pd.DataFrame, k: int):
    """
    Returns: labels, cost, silhouette, runtime
    If kmodes is unavailable, ImportError will bubble up to caller.
    """
    if not HAS_KPROTO:
        raise ImportError("kmodes (K-Prototypes) not installed")

    df_imp = _impute_dataframe(df).copy()
    X = df_imp.to_numpy()
    cat_idx = [
        i for i, c in enumerate(df_imp.columns) if not pd.api.types.is_numeric_dtype(df_imp[c])
    ]

    t0 = time.time()
    kp = KPrototypes(n_clusters=k, init="Huang", n_init=5, verbose=0, random_state=42)
    labels = kp.fit_predict(X, categorical=cat_idx)
    t1 = time.time()

    # Evaluate silhouette in the same embedding as KMeans for comparability
    X_sil = _prep_kmeans_space(df_imp)
    sil = safe_silhouette(X_sil, labels)
    cost = float(getattr(kp, "cost_", np.nan))
    return labels, cost, sil, t1 - t0


# =========================================================================
# ====== Scenario generators (9) ===========================================
# =========================================================================

def _apply_missing(df: pd.DataFrame, rng: np.random.Generator, p=0.03):
    mask = rng.random((len(df), df.shape[1])) < p
    for j, col in enumerate(df.columns):
        df.loc[mask[:, j], col] = np.nan
    return df


def make_scenario_basic(n=1200, seed=0):
    rng = np.random.default_rng(seed)
    K = 3
    z = rng.integers(0, K, size=n)

    mus = np.array([[0, 0], [3, -3], [-3, 3]], float)
    cov = np.array([[1.0, 0.6], [0.6, 1.0]])
    Xc = np.zeros((n, 2))
    for k in range(K):
        idx = z == k
        Xc[idx] = rng.multivariate_normal(mus[k], cov, size=idx.sum())

    cat_levels = np.array(["A", "B", "C"])
    pks = np.array([[0.7, 0.2, 0.1], [0.2, 0.6, 0.2], [0.1, 0.2, 0.7]])
    Xn = np.array([rng.choice(cat_levels, p=pks[k]) for k in z])

    ord_levels = np.array([1, 2, 3, 4])
    qks = np.array([[0.6, 0.25, 0.1, 0.05], [0.2, 0.3, 0.3, 0.2], [0.05, 0.15, 0.3, 0.5]])
    Xo = np.array([rng.choice(ord_levels, p=qks[k]) for k in z])

    df = pd.DataFrame({"x1": Xc[:, 0], "x2": Xc[:, 1], "cat": Xn, "ord": Xo})
    df["ord"] = df["ord"].astype(CategoricalDtype(categories=ord_levels, ordered=True))
    df = _apply_missing(df, rng, p=0.03)
    return df, z, dict(cont=["x1", "x2"], cat=["cat"], ord=["ord"])


def make_scenario_imbalanced(n=1200, seed=1):
    rng = np.random.default_rng(seed)
    K = 3
    pi = np.array([0.7, 0.2, 0.1])
    z = rng.choice(np.arange(K), size=n, p=pi)

    mus = np.array([[0, 0], [3, -3], [-3, 3]], float)
    covs = [
        np.array([[1.2, 0.5], [0.5, 1.2]]),
        np.array([[1.0, 0.3], [0.3, 1.0]]),
        np.array([[1.0, 0.6], [0.6, 1.0]]),
    ]
    Xc = np.zeros((n, 2))
    for k in range(K):
        idx = z == k
        Xc[idx] = rng.multivariate_normal(mus[k], covs[k], size=idx.sum())

    cat_levels = np.array(["A", "B", "C", "D"])
    pks = np.array(
        [
            [0.65, 0.2, 0.1, 0.05],
            [0.25, 0.45, 0.2, 0.1],
            [0.1, 0.2, 0.4, 0.3],
        ]
    )
    Xn = np.array([rng.choice(cat_levels, p=pks[k]) for k in z])

    ord_levels = np.array([1, 2, 3, 4, 5])
    qks = np.array(
        [
            [0.55, 0.25, 0.12, 0.05, 0.03],
            [0.2, 0.25, 0.25, 0.2, 0.1],
            [0.05, 0.1, 0.2, 0.25, 0.4],
        ]
    )
    Xo = np.array([rng.choice(ord_levels, p=qks[k]) for k in z])

    df = pd.DataFrame({"x1": Xc[:, 0], "x2": Xc[:, 1], "cat": Xn, "ord": Xo})
    df["ord"] = df["ord"].astype(CategoricalDtype(categories=ord_levels, ordered=True))
    df = _apply_missing(df, rng, p=0.05)
    return df, z, dict(cont=["x1", "x2"], cat=["cat"], ord=["ord"])


def make_scenario_weaksep(n=1200, seed=2):
    rng = np.random.default_rng(seed)
    K = 3
    z = rng.integers(0, K, size=n)

    mus = np.array([[0, 0], [1.2, -1.2], [-1.2, 1.2]], float)  # close means
    cov = np.array([[1.2, 0.2], [0.2, 1.2]])
    Xc = np.zeros((n, 2))
    for k in range(K):
        idx = z == k
        Xc[idx] = rng.multivariate_normal(mus[k], cov, size=idx.sum())

    cat_levels = np.array(["A", "B", "C"])
    pks = np.array([[0.5, 0.3, 0.2], [0.4, 0.35, 0.25], [0.35, 0.3, 0.35]])
    Xn = np.array([rng.choice(cat_levels, p=pks[k]) for k in z])

    ord_levels = np.array([1, 2, 3, 4])
    qks = np.array(
        [[0.45, 0.3, 0.15, 0.1], [0.4, 0.3, 0.2, 0.1], [0.35, 0.3, 0.2, 0.15]]
    )
    Xo = np.array([rng.choice(ord_levels, p=qks[k]) for k in z])

    df = pd.DataFrame({"x1": Xc[:, 0], "x2": Xc[:, 1], "cat": Xn, "ord": Xo})
    df["ord"] = df["ord"].astype(CategoricalDtype(categories=ord_levels, ordered=True))
    df = _apply_missing(df, rng, p=0.03)
    return df, z, dict(cont=["x1", "x2"], cat=["cat"], ord=["ord"])


def make_scenario_heavytail(n=1200, seed=3):
    rng = np.random.default_rng(seed)
    K = 3
    z = rng.integers(0, K, size=n)

    mus = np.array([[0, 0], [3, -3], [-3, 3]], float)
    nus = [4.0, 6.0, 8.0]
    cov = np.array([[1.0, 0.6], [0.6, 1.0]])
    Xc = np.zeros((n, 2))
    for k in range(K):
        idx = z == k
        m = idx.sum()
        Z = rng.multivariate_normal([0, 0], cov, size=m)
        chi = rng.chisquare(df=nus[k], size=m)
        T = Z / np.sqrt(chi[:, None] / nus[k]) + mus[k]
        Xc[idx] = T

    cat_levels = np.array(["A", "B", "C"])
    pks = np.array([[0.7, 0.2, 0.1], [0.2, 0.6, 0.2], [0.1, 0.2, 0.7]])
    Xn = np.array([rng.choice(cat_levels, p=pks[k]) for k in z])

    ord_levels = np.array([1, 2, 3, 4])
    qks = np.array([[0.6, 0.25, 0.1, 0.05], [0.2, 0.3, 0.3, 0.2], [0.05, 0.15, 0.3, 0.5]])
    Xo = np.array([rng.choice(ord_levels, p=qks[k]) for k in z])

    df = pd.DataFrame({"x1": Xc[:, 0], "x2": Xc[:, 1], "cat": Xn, "ord": Xo})
    df["ord"] = df["ord"].astype(CategoricalDtype(categories=ord_levels, ordered=True))
    df = _apply_missing(df, rng, p=0.05)
    return df, z, dict(cont=["x1", "x2"], cat=["cat"], ord=["ord"])


def make_scenario_highdim(n=1500, seed=4):
    rng = np.random.default_rng(seed)
    K = 3
    z = rng.integers(0, K, size=n)

    cont_d = 10
    mus = np.array([[0] * cont_d, [1.5] * cont_d, [-1.5] * cont_d], float)
    baseR = 0.4
    cov = np.full((cont_d, cont_d), baseR)
    np.fill_diagonal(cov, 1.0)
    Xc = np.zeros((n, cont_d))
    for k in range(K):
        idx = z == k
        Xc[idx] = rng.multivariate_normal(mus[k], cov, size=idx.sum())

    cat1_levels = np.array(list("ABCD"))
    cat2_levels = np.array(list("ABCDE"))
    cat3_levels = np.array(list("ABCDEF"))

    def draw_cat(levels, tilt):
        p = np.ones(len(levels)) / len(levels)
        p = p * (1 - tilt)
        p[0] += tilt
        return rng.choice(levels, p=p)

    Xcat1 = np.array([draw_cat(cat1_levels, [0.2, 0.1, 0.05][k]) for k in z])
    Xcat2 = np.array([draw_cat(cat2_levels, [0.2, 0.1, 0.05][k]) for k in z])
    Xcat3 = np.array([draw_cat(cat3_levels, [0.2, 0.1, 0.05][k]) for k in z])

    ord1_levels = np.array([1, 2, 3, 4])
    ord2_levels = np.array([1, 2, 3, 4, 5])
    ord3_levels = np.array([1, 2, 3, 4])

    def draw_ord(levels, k):
        if k == 0:
            p = np.array([0.5, 0.3, 0.15, 0.05]) if len(levels) == 4 else np.array([0.4, 0.25, 0.2, 0.1, 0.05])
        elif k == 1:
            p = np.ones(len(levels)) / len(levels)
        else:
            p = np.array([0.05, 0.15, 0.3, 0.5]) if len(levels) == 4 else np.array([0.05, 0.1, 0.2, 0.25, 0.4])
        return rng.choice(levels, p=p / p.sum())

    Xo1 = np.array([draw_ord(ord1_levels, k) for k in z])
    Xo2 = np.array([draw_ord(ord2_levels, k) for k in z])
    Xo3 = np.array([draw_ord(ord3_levels, k) for k in z])

    cols = {f"xc{i}": Xc[:, i] for i in range(cont_d)}
    cols.update(dict(cat1=Xcat1, cat2=Xcat2, cat3=Xcat3, ord1=Xo1, ord2=Xo2, ord3=Xo3))
    df = pd.DataFrame(cols)
    df["ord1"] = df["ord1"].astype(CategoricalDtype(categories=ord1_levels, ordered=True))
    df["ord2"] = df["ord2"].astype(CategoricalDtype(categories=ord2_levels, ordered=True))
    df["ord3"] = df["ord3"].astype(CategoricalDtype(categories=ord3_levels, ordered=True))
    df = _apply_missing(df, rng, p=0.04)
    cont_cols = [c for c in df.columns if c.startswith("xc")]
    cat_cols = ["cat1", "cat2", "cat3"]
    ord_cols = ["ord1", "ord2", "ord3"]
    return df, z, dict(cont=cont_cols, cat=cat_cols, ord=ord_cols)


def make_scenario_mnar(n=1400, seed=5):
    rng = np.random.default_rng(seed)
    K = 3
    z = rng.integers(0, K, size=n)

    mus = np.array([[0, 0], [2.5, -2.5], [-2.5, 2.5]], float)
    cov = np.array([[1.1, 0.5], [0.5, 1.1]])
    Xc = np.zeros((n, 2))
    for k in range(K):
        idx = z == k
        Xc[idx] = rng.multivariate_normal(mus[k], cov, size=idx.sum())

    cat_levels = np.array(["A", "B", "C"])
    pks = np.array([[0.65, 0.25, 0.10], [0.3, 0.5, 0.2], [0.1, 0.25, 0.65]])
    Xn = np.array([rng.choice(cat_levels, p=pks[k]) for k in z])

    ord_levels = np.array([1, 2, 3, 4])
    qks = np.array([[0.5, 0.3, 0.15, 0.05], [0.25, 0.3, 0.25, 0.2], [0.05, 0.15, 0.3, 0.5]])
    Xo = np.array([rng.choice(ord_levels, p=qks[k]) for k in z])

    df = pd.DataFrame({"x1": Xc[:, 0], "x2": Xc[:, 1], "cat": Xn, "ord": Xo})
    df["ord"] = df["ord"].astype(CategoricalDtype(categories=ord_levels, ordered=True))

    # MNAR-ish: cluster 0's high x2 more missing; cluster 2 & cat='C' more missing
    prob_x2 = 1 / (1 + np.exp(-0.8 * (df["x2"] - 1.0)))
    m2 = (z == 0) & (np.random.default_rng(seed + 10).random(n) < (prob_x2 * 0.6))
    df.loc[m2, "x2"] = np.nan

    mcat = (z == 2) & (df["cat"] == "C") & (np.random.default_rng(seed + 20).random(n) < 0.4)
    df.loc[mcat, "cat"] = np.nan

    df = _apply_missing(df, rng, p=0.02)
    return df, z, dict(cont=["x1", "x2"], cat=["cat"], ord=["ord"])


def make_scenario_blockcorr(n=1500, seed=6):
    rng = np.random.default_rng(seed)
    K = 3
    z = rng.integers(0, K, size=n)

    cont_d = 6

    def blockR(r):
        R = np.eye(2)
        R[R == 0] = r
        return R

    Rs = [
        np.block(
            [[blockR(0.75), np.zeros((2, 2)), np.zeros((2, 2))],
             [np.zeros((2, 2)), blockR(0.30), np.zeros((2, 2))],
             [np.zeros((2, 2)), np.zeros((2, 2)), blockR(0.10)]]
        ),
        np.block(
            [[blockR(0.30), np.zeros((2, 2)), np.zeros((2, 2))],
             [np.zeros((2, 2)), blockR(0.75), np.zeros((2, 2))],
             [np.zeros((2, 2)), np.zeros((2, 2)), blockR(0.10)]]
        ),
        np.block(
            [[blockR(0.10), np.zeros((2, 2)), np.zeros((2, 2))],
             [np.zeros((2, 2)), blockR(0.30), np.zeros((2, 2))],
             [np.zeros((2, 2)), np.zeros((2, 2)), blockR(0.75)]]
        ),
    ]
    mus = np.array([[0] * cont_d, [1] * cont_d, [-1] * cont_d], float)
    Xc = np.zeros((n, cont_d))
    for k in range(K):
        idx = z == k
        Xc[idx] = rng.multivariate_normal(mus[k], Rs[k], size=idx.sum())

    cat_levels = np.array(["A", "B", "C", "D"])
    pks = np.array(
        [
            [0.55, 0.25, 0.15, 0.05],
            [0.3, 0.4, 0.2, 0.1],
            [0.15, 0.2, 0.35, 0.3],
        ]
    )
    Xn = np.array([rng.choice(cat_levels, p=pks[k]) for k in z])

    ord_levels = np.array([1, 2, 3, 4, 5])
    qks = np.array(
        [
            [0.45, 0.25, 0.15, 0.1, 0.05],
            [0.25, 0.25, 0.2, 0.2, 0.1],
            [0.05, 0.1, 0.2, 0.25, 0.4],
        ]
    )
    Xo = np.array([rng.choice(ord_levels, p=qks[k]) for k in z])

    cols = {f"xc{i}": Xc[:, i] for i in range(cont_d)}
    cols.update(dict(cat=Xn, ord=Xo))
    df = pd.DataFrame(cols)
    df["ord"] = df["ord"].astype(CategoricalDtype(categories=ord_levels, ordered=True))
    df = _apply_missing(df, rng, p=0.03)
    cont_cols = [c for c in df.columns if c.startswith("xc")]
    return df, z, dict(cont=cont_cols, cat=["cat"], ord=["ord"])


def make_scenario_discrete(n=1300, seed=7):
    rng = np.random.default_rng(seed)
    K = 3
    z = rng.integers(0, K, size=n)

    Xc = rng.normal(0, 0.7, size=(n, 2))

    cat1_levels = np.array(list("ABCDE"))
    cat2_levels = np.array(list("ABCDEFG"))
    skew = [
        [0.5, 0.2, 0.1, 0.1, 0.1],
        [0.1, 0.4, 0.2, 0.15, 0.15],
        [0.1, 0.1, 0.2, 0.2, 0.4],
    ]

    def draw_levels(levels, p):
        p = np.array(p, float)
        p = p / p.sum()
        if len(levels) != len(p):
            p = np.ones(len(levels)) / len(levels)
        return rng.choice(levels, p=p)

    Xcat1 = np.array([draw_levels(cat1_levels, skew[k]) for k in z])

    def cond_cat2(c1, k):
        base = np.ones(7) / 7
        base[ord(c1) - ord("A")] += 0.2
        base = base / base.sum()
        return rng.choice(cat2_levels, p=base)

    Xcat2 = np.array([cond_cat2(c1, k) for c1, k in zip(Xcat1, z)])

    ord_levels = np.array([1, 2, 3, 4, 5])
    qks = np.array(
        [
            [0.6, 0.25, 0.1, 0.04, 0.01],
            [0.2, 0.25, 0.25, 0.2, 0.1],
            [0.02, 0.05, 0.1, 0.28, 0.55],
        ]
    )
    Xo = np.array([rng.choice(ord_levels, p=qks[k]) for k in z])

    df = pd.DataFrame(
        {"x1": Xc[:, 0], "x2": Xc[:, 1], "cat1": Xcat1, "cat2": Xcat2, "ord": Xo}
    )
    df["ord"] = df["ord"].astype(CategoricalDtype(categories=ord_levels, ordered=True))
    df = _apply_missing(df, rng, p=0.04)
    return df, z, dict(cont=["x1", "x2"], cat=["cat1", "cat2"], ord=["ord"])


def make_scenario_noise(n=1500, seed=8):
    rng = np.random.default_rng(seed)
    K = 3
    z = rng.integers(0, K, size=n)

    mus = np.array([[0, 0], [3, -3], [-3, 3]], float)
    cov = np.array([[1.0, 0.6], [0.6, 1.0]])
    Xc = np.zeros((n, 2))
    for k in range(K):
        idx = z == k
        Xc[idx] = rng.multivariate_normal(mus[k], cov, size=idx.sum())

    cat_levels = np.array(["A", "B", "C"])
    pks = np.array([[0.7, 0.2, 0.1], [0.2, 0.6, 0.2], [0.1, 0.2, 0.7]])
    Xn = np.array([rng.choice(cat_levels, p=pks[k]) for k in z])

    ord_levels = np.array([1, 2, 3, 4])
    qks = np.array([[0.6, 0.25, 0.1, 0.05], [0.2, 0.3, 0.3, 0.2], [0.05, 0.15, 0.3, 0.5]])
    Xo = np.array([rng.choice(ord_levels, p=qks[k]) for k in z])

    noise_cont = rng.normal(0, 1, size=(n, 8))
    noise_cat = [rng.choice(list("ABCDE"), size=n) for _ in range(5)]
    noise_ord = [rng.choice([1, 2, 3, 4], size=n) for _ in range(3)]

    df = pd.DataFrame({"x1": Xc[:, 0], "x2": Xc[:, 1], "cat": Xn, "ord": Xo})
    for j in range(8):
        df[f"nc{j}"] = noise_cont[:, j]
    for i, col in enumerate(noise_cat):
        df[f"ncat{i}"] = col
    for i, col in enumerate(noise_ord):
        df[f"nord{i}"] = col
        df[f"nord{i}"] = df[f"nord{i}"].astype(CategoricalDtype(categories=[1, 2, 3, 4], ordered=True))

    df["ord"] = df["ord"].astype(CategoricalDtype(categories=ord_levels, ordered=True))
    df = _apply_missing(df, rng, p=0.04)
    cont_cols = ["x1", "x2"] + [f"nc{j}" for j in range(8)]
    cat_cols = ["cat"] + [f"ncat{i}" for i in range(5)]
    ord_cols = ["ord"] + [f"nord{i}" for i in range(3)]
    return df, z, dict(cont=cont_cols, cat=cat_cols, ord=ord_cols)


# Public registry (used by run_benchmark.py)
SCENARIOS = [
    ("basic", make_scenario_basic),
    ("imbalanced", make_scenario_imbalanced),
    ("weak_separation", make_scenario_weaksep),
    ("heavy_tail", make_scenario_heavytail),
    ("high_dim", make_scenario_highdim),
    ("mnar_like", make_scenario_mnar),
    ("block_corr", make_scenario_blockcorr),
    ("mostly_discrete", make_scenario_discrete),
    ("with_noise", make_scenario_noise),
]


# =========================================================================
# ====== Plotting ==========================================================
# =========================================================================

def plot_bar_performance(res_df: pd.DataFrame, metric="ARI", figsize=(12, 6)):
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=figsize)
    order = sorted(res_df["scenario"].unique().tolist())
    sns.barplot(data=res_df, x="scenario", y=metric, hue="model", order=order, errorbar=None)
    plt.title(f"Performance by Scenario ({metric})")
    plt.ylim(0, 1.05)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.show()


def plot_time_tradeoff(res_df: pd.DataFrame, scenario: str = None, metric: str = "ARI", figsize=(8, 7)):
    sns.set_theme(style="whitegrid")
    dfp = res_df.copy()
    if scenario:
        dfp = dfp[dfp["scenario"] == scenario]

    plt.figure(figsize=figsize)
    sns.scatterplot(data=dfp, x="time", y=metric, hue="model", style="model", s=100)
    plt.xlabel("Runtime [s]")
    plt.ylabel(metric)
    title = f"Tradeoff: {metric} vs. Time"
    if scenario:
        title += f" (Scenario: {scenario})"
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True, which="both", ls="--")
    plt.tight_layout()
    plt.show()


def plot_heatmap(res_df: pd.DataFrame, metric="ARI", figsize=(10, 8)):
    pivot = res_df.pivot_table(index="scenario", columns="model", values=metric)
    plt.figure(figsize=figsize)
    sns.heatmap(pivot, annot=True, fmt=".3f", vmin=0, vmax=1, cmap="viridis", linewidths=0.5)
    plt.title(f"Heatmap of {metric}")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()


def plot_scalability(sc_df: pd.DataFrame, metric="time", figsize=(8, 6)):
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=figsize)
    sns.lineplot(data=sc_df, x="n", y=metric, hue="model", marker="o")
    plt.xlabel("Number of Samples (n)")
    ylabel = "Runtime [s]" if metric == "time" else metric
    plt.ylabel(ylabel)
    plt.title(f"Scalability: {ylabel} vs. Number of Samples")
    if metric == "time":
        plt.yscale("log")
        plt.ylabel("Runtime [s] (log scale)")
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.tight_layout()
    plt.show()
