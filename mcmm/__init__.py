# -*- coding: utf-8 -*-
"""
pymcmm - Mixed-Copula Mixture Model

A Python library for clustering datasets with mixed continuous, categorical,
and ordinal data types using Gaussian copula-based mixture models.

Usage:
    from mcmm import MCMMGaussianCopula, MCMMGaussianCopulaSpeedy
    
    # For large datasets, use Speedy mode
    model = MCMMGaussianCopulaSpeedy(
        n_components=4,
        cont_marginal='student_t',
        copula_likelihood='pairwise',
        verbose=1
    )
    model.fit(df, cont_cols=['x1', 'x2'], cat_cols=['c1'], ord_cols=['o1'])
    clusters = model.predict(df)
"""

from .model import MCMMGaussianCopula, MCMMGaussianCopulaSpeedy

__version__ = "0.2.0"
__all__ = ["MCMMGaussianCopula", "MCMMGaussianCopulaSpeedy"]

# Cython acceleration status
_CYTHON_AVAILABLE = False

try:
    from ._fast_core import benchmark as _benchmark
    _CYTHON_AVAILABLE = True
except ImportError:
    _benchmark = None


def check_acceleration():
    """
    Check if Cython acceleration is available.
    
    Returns
    -------
    bool
        True if Cython acceleration is enabled, False otherwise.
    
    Examples
    --------
    >>> import mcmm
    >>> mcmm.check_acceleration()
    ✓ Cython acceleration is enabled (35x faster)
    True
    """
    if _CYTHON_AVAILABLE:
        print("✓ Cython acceleration is enabled (35x faster)")
        return True
    else:
        print("✗ Cython acceleration is NOT available (using pure Python)")
        print("  To enable, run: pip install cython && python setup.py build_ext --inplace")
        return False


def run_benchmark():
    """
    Run performance benchmark comparing scipy vs Cython implementations.
    
    Only available when Cython acceleration is enabled.
    """
    if _benchmark is not None:
        _benchmark()
    else:
        print("Benchmark not available. Cython module not compiled.")
        print("To enable, run: pip install cython && python setup.py build_ext --inplace")
