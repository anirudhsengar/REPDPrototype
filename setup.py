#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

# Most configuration is in pyproject.toml
# This file exists mainly for backward compatibility

setup(
    name="repd",
    version="0.1.0",
    description="Repository Entry Points Defects (REPD) model implementation",
    author="Anirudh Sengar",
    author_email="anirudhsengar3@gmail.com",
    url="https://github.com/anirudhsengar/REPDPrototype",
    packages=find_packages(exclude=["tests", "tests.*"]),
    include_package_data=True,
    python_requires=">=3.8",
    install_requires=[
        "gitpython>=3.1.0",
        "matplotlib>=3.5.0",
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "networkx>=2.6.0",
        "scikit-learn>=1.0.0",
        "pygments>=2.10.0",
        "tqdm>=4.62.0",
        "rich>=10.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=2.12.0",
            "black>=22.3.0",
            "isort>=5.10.0",
            "flake8>=4.0.0",
            "mypy>=0.910",
            "pre-commit>=2.17.0",
        ],
        "docs": [
            "sphinx>=4.4.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "repd=repd.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Software Development :: Quality Assurance",
        "Topic :: Software Development :: Testing",
    ],
)
