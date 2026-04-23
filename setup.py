from setuptools import find_packages, setup


setup(
    name="flowguard-ids",
    version="0.2.0",
    description="CLAN (Contrastive self-supervised NIDS) reproduction and SSL baseline comparison on Lycos2017",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.26",
        "pandas>=2.2",
        "scikit-learn>=1.4",
        "torch>=2.5",
        "pyyaml>=6.0",
        "matplotlib>=3.8",
        "tqdm>=4.66",
        "joblib>=1.3",
    ],
)
