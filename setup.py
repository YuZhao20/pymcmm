from setuptools import setup, find_packages

# README.mdの内容をlong_descriptionとして読み込む
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="pymcmm",
    version="0.1.0",
    author="Your Name",  # あなたの名前に変更してください
    author_email="your.email@example.com",  # あなたのメールアドレスに変更してください
    description="混合コピュラ混合モデル（MCMM）のPython実装",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/YOUR_USERNAME/pymcmm",  # あなたのGitHubリポジトリURLに変更してください
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",  # プロジェクトのライセンスに合わせて変更
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
)
