# =========================================================================
# Utilities for MCMM Benchmark
#
# This script contains helper functions for data preprocessing (imputation),
# evaluation metrics (silhouette score), and plotting results from the
# benchmark analysis.
# =========================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import silhouette_score
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from typing import Tuple

def _impute_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Imputes a dataframe with mean for numeric and mode for other types."""
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_numeric_dtype(out[c]):
            out[c] = out[c].fillna(out[c].mean())
        else:
            mode_val = out[c].mode()
            if not mode_val.empty:
                out[c] = out[c].fillna(mode_val.iloc[0])
    return out

def _evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    """Calculates ARI and NMI, handling NaNs in predictions."""
    mask = ~pd.isna(y_pred)
    if not mask.any():
        return np.nan, np.nan
    return adjusted_rand_score(y_true[mask], y_pred[mask]), \
           normalized_mutual_info_score(y_true[mask], y_pred[mask])

def _prep_kmeans_space(df_imp: pd.DataFrame) -> np.ndarray:
    """Prepares a numeric feature space for KMeans/Silhouette (Standardize + OneHot)."""
    cont_cols = [c for c in df_imp.columns if pd.api.types.is_numeric_dtype(df_imp[c])]
    disc_cols = [c for c in df_imp.columns if not pd.api.types.is_numeric_dtype(df_imp[c])]
    xs = []
    if cont_cols:
        xs.append(StandardScaler().fit_transform(df_imp[cont_cols].to_numpy(float)))
    if disc_cols:
        xs.append(OneHotEncoder(sparse_output=False, handle_unknown='ignore').fit_transform(df_imp[disc_cols].astype('category')))
    return np.hstack(xs) if xs else np.zeros((len(df_imp), 1))

def safe_silhouette(X: np.ndarray, labels: np.ndarray) -> float:
    """Safely calculates silhouette score, returning NaN if not possible."""
    try:
        # Silhouette score is defined only if number of labels is 2 <= n_labels <= n_samples - 1
        n_labels = len(np.unique(labels))
        n_samples = len(X)
        if n_labels < 2 or n_labels > n_samples - 1:
            return np.nan
        return float(silhouette_score(X, labels, metric='euclidean'))
    except Exception:
        return np.nan

# =========================================================================
# ====== Visualization Functions ==========================================
# =========================================================================

def plot_bar_performance(res_df: pd.DataFrame, metric='ARI', figsize=(12,6)):
    """Plots a bar chart of model performance for each scenario."""
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=figsize)
    order = sorted(res_df['scenario'].unique().tolist())
    sns.barplot(data=res_df, x='scenario', y=metric, hue='model', order=order)
    plt.title(f'Performance by Scenario ({metric})')
    plt.ylim(0, 1.05)
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.show()

def plot_time_tradeoff(res_df: pd.DataFrame, scenario:str=None, metric:str='ARI', figsize=(8,7)):
    """Plots a scatter plot of performance vs. runtime."""
    sns.set_theme(style="whitegrid")
    dfp = res_df.copy()
    if scenario:
        dfp = dfp[dfp['scenario'] == scenario]
    
    plt.figure(figsize=figsize)
    sns.scatterplot(data=dfp, x='time', y=metric, hue='model', style='model', s=100)
    plt.xlabel('Runtime [s] (log scale)')
    plt.ylabel(metric)
    plt.xscale('log')
    title = f'Tradeoff: {metric} vs. Time'
    if scenario:
        title += f' (Scenario: {scenario})'
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, which="both", ls="--")
    plt.tight_layout()
    plt.show()

def plot_heatmap(res_df: pd.DataFrame, metric='ARI', figsize=(10,8)):
    """Plots a heatmap of model performance across scenarios."""
    pivot = res_df.pivot_table(index='scenario', columns='model', values=metric)
    plt.figure(figsize=figsize)
    sns.heatmap(pivot, annot=True, fmt=".3f", vmin=0, vmax=1, cmap='viridis', linewidths=.5)
    plt.title(f'Heatmap of {metric}')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

def plot_scalability(sc_df: pd.DataFrame, metric='time', figsize=(8,6)):
    """Plots scalability results (metric vs. n_samples)."""
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=figsize)
    sns.lineplot(data=sc_df, x='n', y=metric, hue='model', marker='o')
    plt.xlabel('Number of Samples (n)')
    ylabel = 'Runtime [s]' if metric == 'time' else metric
    plt.ylabel(ylabel)
    plt.title(f'Scalability: {ylabel} vs. Number of Samples')
    if metric == 'time':
        plt.yscale('log')
        plt.ylabel('Runtime [s] (log scale)')
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.tight_layout()
    plt.show()

