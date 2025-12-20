# mcmm package initializer

from .model import MCMMGaussianCopula, MCMMGaussianCopulaSpeedy

__all__ = [
    "MCMMGaussianCopula",
    "MCMMGaussianCopulaSpeedy",
    "check_acceleration"
]

def check_acceleration():
    """
    Check if Cython acceleration is available.
    
    Returns
    -------
    dict
        Dictionary with acceleration status information:
        - 'available': bool, whether Cython acceleration is available
        - 'version': str or None, Cython version if available
        - 'functions': list, list of accelerated function names
    """
    try:
        from . import _fast_core
        import cython
        return {
            'available': True,
            'version': cython.__version__,
            'functions': [
                'log_gaussian_copula_density_full',
                'log_bivariate_gaussian_copula',
                'pairwise_weighted_corr_fast'
            ]
        }
    except ImportError:
        return {
            'available': False,
            'version': None,
            'functions': []
        }
