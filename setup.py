#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script d'installation pour le projet Morningstar.

Ce script permet d'installer le projet et ses du00e9pendances.
"""

from setuptools import setup, find_packages
import os
from pathlib import Path

# Lire le contenu du fichier README.md
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

# Lire les du00e9pendances depuis requirements.txt
requirements = []
with open("requirements.txt", "r") as f:
    for line in f.readlines():
        line = line.strip()
        if line and not line.startswith("#") and not line.startswith("##"):
            requirements.append(line)

setup(
    name="morningstar",
    version="2.0.0",
    description="Robot de trading crypto hybride avancu00e9",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Cabrel10",
    author_email="cabrel10@example.com",
    url="https://github.com/Cabrel10/eva001",
    packages=find_packages(),
    install_requires=requirements,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Office/Business :: Financial :: Investment",
    ],
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "morningstar-trading=live.trading_engine:main",
            "morningstar-backtest=run_backtest:main",
            "morningstar-dataset=scripts.datasetbiuld:main",
        ],
    },
    include_package_data=True,
    package_data={
        "morningstar": ["config/*.yaml", "config/*.example"],
    },
)
