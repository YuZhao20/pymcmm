# =========================================================================
# MCMM Benchmark Suite
#
# This script contains the functions to generate synthetic datasets for various
# scenarios and to run the benchmark comparing MCMM against baseline models
# like KMeans and K-Prototypes.
#
# To Run:
#  - Ensure `mcmm` package is installed or in the parent directory.
#  - Ensure `utils.py` is in the same directory.
#  - Run this script directly: `python run_benchmark.py`
# =========================================================================

import numpy as np
import pandas as pd
import time
from typing import List, Dict, Tuple, Optional
from pandas.api.types import CategoricalDtype

# Add parent directory to path to find the 'mcmm' package
import sys
sys.path.append('..')
from mcmm.model import MCMMGaussianCopula, MCMMGaussianCopulaSpeedy
from utils import (
    _impute_dataframe, _evaluate, _prep_kmeans_space, safe_silhouette,
    plot_bar_performance, plot_time_tradeoff, plot_heatmap, plot_scalability
)

# Optional: K-Prototypes
try:
    from kmodes.kprototypes import KPrototypes
    HAS_KPROTO = True
except ImportError:
    HAS_KPROTO = False
    print("Warning: 'kmodes' package not found. K-Prototypes baseline will be skipped.")
    print("         Install with: pip install kmodes")


# =========================================================================
# ====== Data Generation: Synthetic Scenarios =============================
# =========================================================================
# (Ported from the original script)

def _apply_missing(df: pd.DataFrame, rng: np.random.Generator, p=0.03) -> pd.DataFrame:
    """Applies missing values to a dataframe at random."""
    mask = rng.random(df.shape) < p
    df_missing = df.mask(mask)
    return df_missing

def make_scenario_basic(n=1200, seed=0) -> Tuple[pd.DataFrame, np.ndarray, Dict]:
    """Basic, well-separated scenario."""
    rng = np.random.default_rng(seed)
    K = 3
    z = rng.integers(0, K, size=n)
    mus = np.array([[0,0],[3,-3],[-3,3]], float)
    cov = np.array([[1.0, 0.6],[0.6, 1.0]])
    Xc = np.zeros((n,2))
    for k in range(K):
        idx = (z==k)
        if idx.sum() > 0:
            Xc[idx] = rng.multivariate_normal(mus[k], cov, size=idx.sum())
    
    cat_levels = ['A','B','C']
    pks = np.array([[0.7,0.2,0.1],[0.2,0.6,0.2],[0.1,0.2,0.7]])
    Xn = np.array([rng.choice(cat_levels, p=pks[k]) for k in z])
    
    ord_levels = [1,2,3,4]
    qks = np.array([[0.6,0.25,0.1,0.05],[0.2,0.3,0.3,0.2],[0.05,0.15,0.3,0.5]])
    Xo = np.array([rng.choice(ord_levels, p=qks[k]) for k in z])
    
    df = pd.DataFrame({'x1':Xc[:,0], 'x2':Xc[:,1], 'cat':Xn, 'ord':Xo})
    df['ord'] = df['ord'].astype(CategoricalDtype(categories=ord_levels, ordered=True))
    df = _apply_missing(df, rng, p=0.03)
    
    spec = {'cont': ['x1','x2'], 'cat': ['cat'], 'ord': ['ord']}
    return df, z, spec

# ...(Assume all other 8 make_scenario_* functions are here)...
SCENARIOS = [("basic", make_scenario_basic)] # Placeholder for all 9 scenarios

# =========================================================================
# ====== Model Runners ====================================================
# =========================================================================

def run_all_benchmarks():
    """Main function to execute all benchmarks and generate plots."""
    # This would orchestrate the calls to run_benchmark_with_k_selection etc.
    # For now, it's a placeholder.
    print("Running benchmarks...")
    df, z_true, spec = make_scenario_basic()
    
    print("\n--- Testing MCMM Full ---")
    model_full = MCMMGaussianCopula(n_components=3, copula_likelihood='full', verbose=1)
    model_full.fit(df, **spec)
    labels = model_full.predict(df)
    ari, nmi = _evaluate(z_true, labels)
    print(f"ARI: {ari:.4f}, NMI: {nmi:.4f}")
    
    print("\n--- Testing MCMM Speedy ---")
    # model_speedy = MCMMGaussianCopulaSpeedy(n_components=3, verbose=1)
    # model_speedy.fit(df, **spec)
    # labels_speedy = model_speedy.predict(df)
    # ari_s, nmi_s = _evaluate(z_true, labels_speedy)
    # print(f"ARI: {ari_s:.4f}, NMI: {nmi_s:.4f}")

if __name__ == '__main__':
    run_all_benchmarks()

