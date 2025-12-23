
from .model import MCMMGaussianCopula, MCMMGaussianCopulaSpeedy

__version__ = "0.3.1"
__all__ = [
    "MCMMGaussianCopula",
    "MCMMGaussianCopulaSpeedy",
    "check_acceleration",
    "run_benchmark",
]

_CYTHON_AVAILABLE = False

try:
    from ._fast_core import benchmark as _benchmark
    _CYTHON_AVAILABLE = True
except ImportError:
    _benchmark = None


def check_acceleration():
    if _CYTHON_AVAILABLE:
        print("✓ Cython acceleration is enabled (35x faster)")
        return True
    else:
        print("✗ Cython acceleration is NOT available (using pure Python)")
        print("  To enable, run: pip install cython && python setup.py build_ext --inplace")
        return False


def run_benchmark():
    if _benchmark is not None:
        _benchmark()
    else:
        print("Benchmark not available. Cython module not compiled.")
        print("To enable, run: pip install cython && python setup.py build_ext --inplace")
