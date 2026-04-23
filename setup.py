from setuptools import find_packages, setup


setup(
    name="flowguard-ids",
    version="0.1.0",
    description="Lightweight two-stage NIDS: CNN-BiLSTM-SE-Transformer backbone + XI2S cascade + SHAP Top-K on CICIDS2017 / UNSW-NB15",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.26",
        "pandas>=2.2",
        "scikit-learn>=1.4",
        "torch>=2.5",
        "pyyaml>=6.0",
        "matplotlib>=3.8",
    ],
)
