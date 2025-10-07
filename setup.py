from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="pymcmm",
    version="0.1.1",
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
        "bench": ["kmodes", "matplotlib", "seaborn"]
    },
)
