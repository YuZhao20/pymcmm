# -*- coding: utf-8 -*-
"""
Benchmark suite for MCMM, comparing it against baseline models across
various synthetic data scenarios.
"""
import time
import pandas as pd
import warnings
import numpy as np
from typing import List, Dict

# To run this script, ensure you are in the root directory of the project (pymcmm/)
# and execute: python experiments/run_benchmark.py
from mcmm.model import MCMMGaussianCopula, MCMMGaussianCopulaSpeedy
from experiments.utils import (
    _impute_dataframe, _evaluate, _prep_kmeans_space, safe_silhouette,
    run_kmeans, run_kprototypes,
    plot_bar_performance, plot_time_tradeoff, plot_heatmap, plot_scalability,
    make_scenario_basic, make_scenario_imbalanced, make_scenario_weaksep,
    make_scenario_heavytail, make_scenario_highdim, make_scenario_mnar,
    make_scenario_blockcorr, make_scenario_discrete, make_scenario_noise
)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)

# --- Scenario Registry ---
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

# --- MCMM Execution Helper ---
def run_mcmm(df: pd.DataFrame, spec: Dict[str, List[str]], k: int,
             mode='pairwise', student_t=True, estimate_nu=True,
             max_iter=200, tol=1e-4, verbose=0, **speedy_kwargs):
    
    ModelClass = MCMMGaussianCopulaSpeedy if mode == 'speedy' else MCMMGaussianCopula
    
    model_params = {
        'n_components': k,
        'cont_marginal': 'student_t' if student_t else 'gaussian',
        't_nu': 5.0,
        'estimate_nu': estimate_nu,
        'ord_marginal': 'cumlogit',
        'copula_likelihood': 'pairwise' if mode in ('pairwise', 'speedy') else 'full',
        'pairwise_weight': 'abs_rho',
        'dt_mode': 'random',
        'shrink_lambda': 0.05,
        'max_iter': max_iter,
        'tol': tol,
        'random_state': 42,
        'verbose': verbose
    }

    if mode == 'speedy':
        model_params.update(speedy_kwargs)

    model = ModelClass(**model_params)
    
    t0 = time.time()
    model.fit(df, cont_cols=spec['cont'], cat_cols=spec['cat'], ord_cols=spec['ord'])
    t1 = time.time()
    
    return {
        'labels': model.predict(df),
        'bic': model.bic_,
        'cbic': getattr(model, 'cbic_', np.nan),
        'loglik': model.loglik_,
        'nu': getattr(model, 'fitted_nu_', None),
        'runtime': t1 - t0,
        'model_obj': model
    }

# --- K-Selection Wrappers ---
def select_k_mcmm_bic(df: pd.DataFrame, spec: Dict[str, List[str]],
                      k_grid: List[int], mode='pairwise', **kwargs):
    records = []
    best_res, best_bic = None, np.inf
    
    for k in k_grid:
        res = run_mcmm(df, spec, k, mode=mode, **kwargs)
        bic = res['cbic'] if mode != 'full' and not pd.isna(res['cbic']) else res['bic']
        records.append({'k': k, 'bic': bic, 'loglik': res['loglik'], 'time': res['runtime']})
        if bic < best_bic:
            best_bic = bic
            best_res = {'k': k, **res}
            
    return best_res, pd.DataFrame(records)

def select_k_kmeans(df: pd.DataFrame, k_grid: List[int]):
    df_imp = _impute_dataframe(df)
    X = _prep_kmeans_space(df_imp)
    stats = {}
    for k in k_grid:
        labels, sil, inertia, t = run_kmeans(df, k)
        stats[k] = {'labels': labels, 'silhouette': sil, 'inertia': inertia, 'time': t}
    
    sils = {k: v['silhouette'] for k, v in stats.items() if not pd.isna(v['silhouette'])}
    if sils:
        k_star = max(sils, key=sils.get)
    else:
        k_star = k_grid[0] # Fallback
    return k_star, stats

def select_k_kprototypes(df: pd.DataFrame, k_grid: List[int]):
    stats = {}
    for k in k_grid:
        labels, cost, sil, t = run_kprototypes(df, k)
        stats[k] = {'labels': labels, 'cost': cost, 'silhouette': sil, 'time': t}

    costs = {k: v['cost'] for k, v in stats.items() if not pd.isna(v['cost'])}
    if costs:
        k_star = min(costs, key=costs.get)
    else:
        k_star = k_grid[0] # Fallback
    return k_star, stats

# --- Main Benchmark Runner ---
def run_benchmark(scenarios=SCENARIOS, mcmm_modes=('pairwise', 'full', 'speedy'), k_grid=(2, 3, 4, 5)):
    results = []
    for name, maker in scenarios:
        print(f"\n=== Scenario: {name} ===")
        df, z_true, spec = maker()

        # --- MCMM Models ---
        for mode in mcmm_modes:
            best, _ = select_k_mcmm_bic(df, spec, list(k_grid), mode=mode, verbose=0)
            ari, nmi = _evaluate(z_true, best['labels'])
            print(f"MCMM[{mode}] k*={best['k']} | ARI={ari:.4f} NMI={nmi:.4f} | BIC={best['bic']:.1f} | time={best['runtime']:.2f}s")
            results.append({'scenario': name, 'model': f'MCMM_{mode}', 'k_chosen': best['k'], 'ARI': ari, 'NMI': nmi, 'time': best['runtime']})

        # --- Baseline: KMeans ---
        k_star, stats = select_k_kmeans(df, list(k_grid))
        ari, nmi = _evaluate(z_true, stats[k_star]['labels'])
        print(f"KMeans k*={k_star} | ARI={ari:.4f} NMI={nmi:.4f} | silhouette={stats[k_star]['silhouette']:.3f} | time={stats[k_star]['time']:.2f}s")
        results.append({'scenario': name, 'model': 'KMeans', 'k_chosen': k_star, 'ARI': ari, 'NMI': nmi, 'time': stats[k_star]['time']})

        # --- Baseline: K-Prototypes ---
        try:
            k_star, stats = select_k_kprototypes(df, list(k_grid))
            ari, nmi = _evaluate(z_true, stats[k_star]['labels'])
            print(f"K-Prototypes k*={k_star} | ARI={ari:.4f} NMI={nmi:.4f} | cost={stats[k_star]['cost']:.1f} | time={stats[k_star]['time']:.2f}s")
            results.append({'scenario': name, 'model': 'KPrototypes', 'k_chosen': k_star, 'ARI': ari, 'NMI': nmi, 'time': stats[k_star]['time']})
        except ImportError:
            print("K-Prototypes: Skipped (kmodes library not installed)")

    results_df = pd.DataFrame(results)
    print("\n\n=== Benchmark Summary (Mean over Scenarios) ===")
    print(results_df.groupby('model')[['ARI', 'NMI', 'time']].mean().sort_values('ARI', ascending=False))
    return results_df

if __name__ == '__main__':
    results_df = run_benchmark()
    
    print("\n\n--- Generating Plots ---")
    plot_bar_performance(results_df, metric='ARI')
    plot_heatmap(results_df, metric='ARI')
    plot_time_tradeoff(results_df, metric='ARI')

