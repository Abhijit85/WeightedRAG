#!/usr/bin/env python3
"""Setup script for weighted_rag package."""

from setuptools import setup, find_packages

setup(
    name="weighted_rag",
    version="0.1.0",
    description="Weighted Retrieval Augmented Generation pipeline",
    author="WeightedRAG Team",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.8",
    install_requires=[
        "torch",
        "transformers",
        "sentence-transformers",
        "numpy",
        "pandas",
        "tqdm",
        "faiss-cpu",
        "datasets",
        "beir",
    ],
    extras_require={
        "dev": [
            "pytest",
            "black",
            "flake8",
            "mypy",
        ],
    },
)