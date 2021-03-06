#!/usr/bin/env python
from setuptools import setup, find_packages

__version__ = "0.1"


setup(
    name="kaggle",
    author="Gregory Rehm",
    version=__version__,
    description="Kaggle repo",
    packages=find_packages(),
    package_data={"*": ["*.html"]},
    install_requires=[
        "numpy",
        "pandas",
        "scipy",
        "matplotlib",
        "scikit-learn",
    ],
    entry_points={
        "console_scripts": [
            "titanic=titanic.model:main",
        ]
    }
)
