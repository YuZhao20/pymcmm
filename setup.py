from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext
import numpy as np
import os

# Try to import Cython
try:
    from Cython.Build import cythonize
    CYTHON_AVAILABLE = True
except ImportError:
    CYTHON_AVAILABLE = False
    cythonize = None

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Define Cython extensions
extensions = []
if CYTHON_AVAILABLE:
    extensions = [
        Extension(
            "mcmm._fast_core",
            ["mcmm/_fast_core.pyx"],
            include_dirs=[np.get_include()],
            extra_compile_args=["-O3"],
        )
    ]
    extensions = cythonize(extensions, compiler_directives={
        'language_level': "3",
        'boundscheck': False,
        'wraparound': False,
    })

setup(
    name="pymcmm",
    version="0.2.0",
    author="Yu Zhao",
    author_email="yu.zhao@rs.tus.ac.jp",
    description="混合コピュラ混合モデル（MCMM）のPython実装",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/YuZhao20/pymcmm",
    packages=find_packages(include=["mcmm", "experiments", "experiments.*"]),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires='>=3.8',
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "scipy",
        "joblib",
    ],
    extras_require={
        "bench": ["kmodes", "matplotlib", "seaborn"],
        "cython": ["cython>=0.29.0"],
    },
    ext_modules=extensions,
    include_dirs=[np.get_include()],
    zip_safe=False,
)
