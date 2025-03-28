"""
Repository Entry Points Defects (REPD) Model

A bug prediction approach that analyzes repository entry points,
change coupling patterns, and developer activity.

This package provides tools to:
- Identify repository entry points
- Analyze change coupling between files
- Track developer activity patterns
- Calculate defect risk scores
- Visualize results

Author: anirudhsengar
"""

import logging
from importlib.metadata import PackageNotFoundError, version

# Set up logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# Package metadata
__title__ = "repd"
__description__ = "Repository Entry Points Defects (REPD) model implementation"
__author__ = "anirudhsengar"
__created__ = "2025-03-26"

# Version
try:
    __version__ = version("repd")
except PackageNotFoundError:
    __version__ = "0.1.0"

from repd.entry_point_analyzer import EntryPointIdentifier

# Import key components for easier access
from repd.model import REPDModel
from repd.repository import Repository
from repd.risk_calculator import RiskCalculator

__all__ = [
    "REPDModel",
    "EntryPointIdentifier",
    "Repository",
    "DefectRiskCalculator",
]
