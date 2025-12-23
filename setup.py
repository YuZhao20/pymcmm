
import sys
import os
from setuptools import setup, Extension, find_packages

USE_CYTHON = False
ext_modules = []

try:
    from Cython.Build import cythonize
    import numpy as np
    USE_CYTHON = True
except ImportError:
    pass

if USE_CYTHON:
    if sys.platform == "darwin":  # macOS
        extra_compile_args = ["-O3"]
        extra_link_args = []
        if os.path.exists("/opt/homebrew/opt/libomp"):
            extra_compile_args.extend(["-Xpreprocessor", "-fopenmp", "-I/opt/homebrew/opt/libomp/include"])
            extra_link_args.extend(["-L/opt/homebrew/opt/libomp/lib", "-lomp"])
        elif os.path.exists("/usr/local/opt/libomp"):
            extra_compile_args.extend(["-Xpreprocessor", "-fopenmp", "-I/usr/local/opt/libomp/include"])
            extra_link_args.extend(["-L/usr/local/opt/libomp/lib", "-lomp"])
    elif sys.platform == "linux":
        extra_compile_args = ["-O3"]
        extra_link_args = []
    elif sys.platform == "win32":
        extra_compile_args = ["/O2"]
        extra_link_args = []
    else:
        extra_compile_args = ["-O3"]
        extra_link_args = []

    extensions = [
        Extension(
            "mcmm._fast_core",
            sources=["mcmm/_fast_core.pyx"],
            include_dirs=[np.get_include()],
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
            define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
            language="c",
        )
    ]
    
    ext_modules = cythonize(
        extensions,
        language_level=3,
        compiler_directives={
            "boundscheck": False,
            "wraparound": False,
            "cdivision": True,
            "initializedcheck": False,
            "nonecheck": False,
        },
    )

long_description = ""
if os.path.exists("README.md"):
    with open("README.md", encoding="utf-8") as f:
        long_description = f.read()

setup(
    name="pymcmm",
    version="0.3.1",
    description="Mixed-Copula Mixture Model for clustering mixed-type data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Yu Zhao",
    url="https://github.com/YuZhao20/pymcmm",
    license="MIT",
    packages=find_packages(),
    ext_modules=ext_modules,
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20",
        "pandas>=1.3",
        "scipy>=1.7",
        "scikit-learn>=1.0",
        "joblib>=1.0",
    ],
    extras_require={
        "dev": [
            "cython>=0.29",
            "pytest>=6.0",
            "pytest-cov>=2.0",
        ],
        "cython": [
            "cython>=0.29",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Cython",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    keywords="clustering, mixture-model, copula, mixed-data, machine-learning",
    zip_safe=False,
)
