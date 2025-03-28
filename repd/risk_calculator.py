#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Risk Calculator Module for REPD Model

This module calculates risk scores for files in a repository based on
various metrics including complexity, churn, coupling, and structural importance.

Author: anirudhsengar
"""

import json
import logging
import os
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np

from repd.repository import Repository
from repd.structure_mapper import StructureMapper

logger = logging.getLogger(__name__)


class RiskCalculator:
    """
    Calculates risk scores for files in a repository based on multiple metrics.

    Risk calculation considers code complexity, change frequency (churn),
    coupling with other files, structural importance, and file age.
    """

    def __init__(self, repository: Repository, structure_mapper: StructureMapper):
        """
        Initialize the risk calculator.

        Args:
            repository: Repository interface to analyze
            structure_mapper: Structure mapper with dependency information
        """
        self.repository = repository
        self.structure_mapper = structure_mapper
        self.risk_scores = {}  # Dictionary mapping file paths to risk scores
        self.risk_factors = {}  # Dictionary mapping file paths to risk factor scores

    def calculate_risk_scores(
        self, weights: Dict[str, float] = None
    ) -> Dict[str, float]:
        """
        Calculate risk scores for all code files in the repository.

        Args:
            weights: Dictionary mapping risk factors to their weights
                    (complexity, churn, coupling, structural, age)

        Returns:
            Dictionary mapping file paths to risk scores
        """
        logger.info("Calculating risk scores")

        # Default weights if not specified
        if weights is None:
            weights = {
                "complexity": 0.25,
                "churn": 0.25,
                "coupling": 0.2,
                "structural": 0.2,
                "age": 0.1,
            }

        # Normalize weights to ensure they sum to 1
        total_weight = sum(weights.values())
        weights = {k: v / total_weight for k, v in weights.items()}

        # Get all code files
        code_files = [
            f
            for f in self.repository.get_all_files()
            if self.repository.is_code_file(f)
        ]

        logger.debug(f"Calculating risk for {len(code_files)} files")

        # Calculate individual risk factors
        complexity_scores = self._analyze_complexity()
        churn_scores = self._analyze_churn()
        coupling_scores = self._analyze_coupling()
        structural_scores = self._analyze_structural_importance()
        age_scores = self._analyze_age()

        # Calculate combined risk scores
        risk_scores = {}
        risk_factors = {}

        for file in code_files:
            # Get individual factor scores, default to 0 if not found
            complexity = complexity_scores.get(file, 0.0)
            churn = churn_scores.get(file, 0.0)
            coupling = coupling_scores.get(file, 0.0)
            structural = structural_scores.get(file, 0.0)
            age = age_scores.get(file, 0.0)

            # Calculate weighted sum
            risk = (
                weights["complexity"] * complexity
                + weights["churn"] * churn
                + weights["coupling"] * coupling
                + weights["structural"] * structural
                + weights["age"] * age
            )

            # Store results
            risk_scores[file] = risk
            risk_factors[file] = {
                "complexity": complexity,
                "churn": churn,
                "coupling": coupling,
                "structural": structural,
                "age": age,
            }

        self.risk_scores = risk_scores
        self.risk_factors = risk_factors

        logger.info(f"Calculated risk scores for {len(risk_scores)} files")

        return risk_scores

    def _analyze_complexity(self) -> Dict[str, float]:
        """
        Analyze code complexity for all files.

        Returns:
            Dictionary mapping file paths to normalized complexity scores
        """
        logger.debug("Analyzing code complexity")

        # Get all code files
        code_files = [
            f
            for f in self.repository.get_all_files()
            if self.repository.is_code_file(f)
        ]

        # Calculate complexity for each file
        raw_scores = {}

        for file in code_files:
            # Get file content
            content = self.repository.get_file_content(file)
            if not content:
                continue

            # Calculate complexity using repository helper method
            complexity = self.repository.calculate_complexity(content)

            # Get file size as an additional complexity factor
            size = self.repository.get_file_size(file)

            # Combined score with more weight to cyclomatic complexity
            raw_scores[file] = (0.7 * complexity) + (0.3 * min(1.0, size / 10000))

        # Normalize scores to 0-1 range
        return self._normalize_scores(raw_scores)

    def _analyze_churn(self) -> Dict[str, float]:
        """
        Analyze code churn (frequency of changes) for all files.

        Returns:
            Dictionary mapping file paths to normalized churn scores
        """
        logger.debug("Analyzing code churn")

        # Get commit history
        commits = self.repository.get_commit_history()

        # Count changes per file
        change_count = defaultdict(int)
        recent_change_count = defaultdict(int)

        # Threshold for recent changes (last 30 days)
        recent_threshold = datetime.now() - timedelta(days=30)

        for commit in commits:
            for file in commit.modified_files:
                change_count[file] += 1

                # Check if change is recent
                if hasattr(commit, "date") and isinstance(commit.date, datetime):
                    if commit.date >= recent_threshold:
                        recent_change_count[file] += 1

        # Calculate churn scores
        raw_scores = {}

        for file, count in change_count.items():
            # Consider both total changes and recent changes
            total_score = min(1.0, count / 10)  # Cap at 10 changes for normalization
            recent_score = min(
                1.0, recent_change_count[file] / 5
            )  # Cap at 5 recent changes

            # Combined score with more weight to recent changes
            raw_scores[file] = (0.4 * total_score) + (0.6 * recent_score)

        # Normalize scores to 0-1 range
        return self._normalize_scores(raw_scores)

    def _analyze_coupling(self) -> Dict[str, float]:
        """
        Analyze coupling with other files.

        Returns:
            Dictionary mapping file paths to normalized coupling scores
        """
        logger.debug("Analyzing code coupling")

        # Use structure mapper's import map if available
        if (
            hasattr(self.structure_mapper, "import_map")
            and self.structure_mapper.import_map
        ):
            import_map = self.structure_mapper.import_map
        else:
            # If no import map, return empty scores
            return {}

        # Calculate coupling scores based on number of imports/dependents
        raw_scores = {}

        for file, imports in import_map.items():
            # Score based on number of imports
            import_count = len(imports)

            # Count files that import this file (dependents)
            dependent_count = sum(1 for f, deps in import_map.items() if file in deps)

            # Combined score - files with many dependents and many imports
            # are often more risky
            raw_scores[file] = (0.3 * min(1.0, import_count / 10)) + (
                0.7 * min(1.0, dependent_count / 5)
            )

        # Normalize scores to 0-1 range
        return self._normalize_scores(raw_scores)

    def _analyze_structural_importance(self) -> Dict[str, float]:
        """
        Analyze structural importance of files in the codebase.

        Returns:
            Dictionary mapping file paths to normalized structural importance scores
        """
        logger.debug("Analyzing structural importance")

        # Get centrality scores if available
        centrality_files = self.structure_mapper.get_central_files()
        if not centrality_files:
            return {}

        # Convert to dictionary
        centrality = dict(centrality_files)

        # Use centrality directly as the structural importance score
        # It should already be normalized by the structure mapper
        return centrality

    def _analyze_age(self) -> Dict[str, float]:
        """
        Analyze file age and maturity.

        Returns:
            Dictionary mapping file paths to normalized age risk scores
        """
        logger.debug("Analyzing file age")

        # Get all code files
        code_files = [
            f
            for f in self.repository.get_all_files()
            if self.repository.is_code_file(f)
        ]

        # Calculate age for each file
        raw_scores = {}
        now = datetime.now()

        for file in code_files:
            try:
                creation_date = self.repository.get_file_creation_date(file)

                if isinstance(creation_date, datetime):
                    # Calculate age in days
                    age_days = (now - creation_date).days

                    # Risk is higher for very new files (< 30 days)
                    # and somewhat elevated for very old files (> 365 days)
                    if age_days < 30:
                        # New files: risk decreases as they age from 1 to 0.5
                        raw_scores[file] = 1.0 - (0.5 * age_days / 30)
                    elif age_days > 365:
                        # Old files: risk increases slightly with age from 0.3 to 0.5
                        raw_scores[file] = 0.3 + (
                            0.2 * min(1.0, (age_days - 365) / 730)
                        )
                    else:
                        # Stable age range: low risk
                        raw_scores[file] = 0.3
            except:
                # If date can't be determined, assume medium risk
                raw_scores[file] = 0.5

        # Normalize scores to 0-1 range
        return self._normalize_scores(raw_scores)

    def get_risk_scores(self, top_n: int = None) -> List[Tuple[str, float]]:
        """
        Get calculated risk scores, optionally limited to top N.

        Args:
            top_n: Number of highest-risk files to return

        Returns:
            List of (file_path, risk_score) tuples, sorted by risk (highest first)
        """
        # Sort files by risk score (descending)
        sorted_scores = sorted(
            self.risk_scores.items(), key=lambda x: x[1], reverse=True
        )

        # Limit to top N if specified
        if top_n is not None:
            return sorted_scores[:top_n]

        return sorted_scores

    def get_risk_factors(self, file_path: str) -> Dict[str, float]:
        """
        Get risk factors for a specific file.

        Args:
            file_path: Path to the file

        Returns:
            Dictionary mapping factor names to scores
        """
        return self.risk_factors.get(file_path, {})

    def get_all_risk_factors(self) -> Dict[str, Dict[str, float]]:
        """
        Get risk factors for all files.

        Returns:
            Dictionary mapping file paths to factor dictionaries
        """
        return self.risk_factors

    def export_risk_data(self, output_file: str) -> None:
        """
        Export risk data to a JSON file.

        Args:
            output_file: Path to the output file
        """
        if not self.risk_scores:
            logger.warning("No risk scores to export")
            return

        # Prepare data for export
        data = {
            "risk_scores": self.risk_scores,
            "risk_factors": self.risk_factors,
            "metadata": {
                "repository": self.repository.get_name(),
                "timestamp": datetime.now().isoformat(),
                "top_risks": [
                    {"file": file, "score": score}
                    for file, score in self.get_risk_scores(top_n=10)
                ],
            },
        }

        # Write to file
        with open(output_file, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Exported risk data to {output_file}")

    def import_risk_data(self, input_file: str) -> None:
        """
        Import risk data from a JSON file.

        Args:
            input_file: Path to the input file
        """
        try:
            with open(input_file, "r") as f:
                data = json.load(f)

            self.risk_scores = data.get("risk_scores", {})
            self.risk_factors = data.get("risk_factors", {})

            logger.info(f"Imported risk data from {input_file}")

        except Exception as e:
            logger.error(f"Error importing risk data: {str(e)}")

    def _normalize_scores(self, raw_scores: Dict[str, float]) -> Dict[str, float]:
        """
        Normalize scores to a 0-1 range.

        Args:
            raw_scores: Dictionary mapping file paths to raw scores

        Returns:
            Dictionary mapping file paths to normalized scores
        """
        if not raw_scores:
            return {}

        # Find min and max values
        values = list(raw_scores.values())
        min_val = min(values)
        max_val = max(values)

        # If all values are the same, return constant score
        if min_val == max_val:
            return {k: 0.5 for k in raw_scores.keys()}

        # Normalize to 0-1 range
        normalized = {}
        range_val = max_val - min_val

        for file, score in raw_scores.items():
            normalized[file] = (score - min_val) / range_val

        return normalized

    def get_highest_risk_factors(self) -> Dict[str, Dict[str, Any]]:
        """
        Identify the highest risk factor for each file.

        Returns:
            Dictionary mapping file paths to highest factor information
        """
        result = {}

        for file, factors in self.risk_factors.items():
            # Find the factor with the highest score
            highest_factor = max(factors.items(), key=lambda x: x[1])
            factor_name, factor_value = highest_factor

            result[file] = {"factor": factor_name, "value": factor_value}

        return result

    def classify_risk(
        self, high_threshold: float = 0.7, medium_threshold: float = 0.4
    ) -> Dict[str, List[str]]:
        """
        Classify files into risk categories based on thresholds.

        Args:
            high_threshold: Minimum score for high risk
            medium_threshold: Minimum score for medium risk

        Returns:
            Dictionary mapping risk levels to lists of files
        """
        high_risk = []
        medium_risk = []
        low_risk = []

        for file, score in self.risk_scores.items():
            if score >= high_threshold:
                high_risk.append(file)
            elif score >= medium_threshold:
                medium_risk.append(file)
            else:
                low_risk.append(file)

        return {
            "high_risk": high_risk,
            "medium_risk": medium_risk,
            "low_risk": low_risk,
        }
