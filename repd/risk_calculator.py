#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Defect Risk Calculator for REPD Model

This module provides risk calculation functionality based on multiple factors
including entry point status, change coupling, developer activity, and
path complexity. It combines these factors to produce an overall risk score.

Author: anirudhsengar
"""

import logging
import math
from typing import Dict, List, Tuple, Optional, Any

import numpy as np

logger = logging.getLogger(__name__)


class DefectRiskCalculator:
    """
    Calculates defect risk scores based on multiple input factors.

    This class combines various risk factors from the REPD model components
    to produce an overall defect risk score for each file. It provides
    flexible weighting options and risk categorization.
    """

    # Risk category thresholds
    RISK_CATEGORIES = {
        "very_high": 0.8,
        "high": 0.6,
        "medium": 0.4,
        "low": 0.2,
        "very_low": 0.0
    }

    def __init__(self):
        """Initialize the risk calculator."""
        pass

    def calculate_risk(
            self,
            entry_point_score: float,
            coupling_score: float,
            dev_expertise_score: float,
            path_complexity_score: float,
            entry_point_weight: float = 0.5,
            coupling_weight: float = 0.3,
            dev_expertise_weight: float = 0.3,
            path_complexity_weight: float = 0.2
    ) -> float:
        """
        Calculate overall risk score based on multiple factors.

        Args:
            entry_point_score: Score indicating entry point importance (0.0-1.0)
            coupling_score: Score indicating change coupling risk (0.0-1.0)
            dev_expertise_score: Score indicating developer expertise risk (0.0-1.0)
            path_complexity_score: Score indicating path complexity risk (0.0-1.0)
            entry_point_weight: Weight for entry point factor
            coupling_weight: Weight for coupling factor
            dev_expertise_weight: Weight for developer expertise factor
            path_complexity_weight: Weight for path complexity factor

        Returns:
            Overall normalized risk score (0.0-1.0)
        """
        # Validate inputs
        entry_point_score = self._validate_score(entry_point_score)
        coupling_score = self._validate_score(coupling_score)
        dev_expertise_score = self._validate_score(dev_expertise_score)
        path_complexity_score = self._validate_score(path_complexity_score)

        # Normalize weights to sum to 1.0
        total_weight = (entry_point_weight + coupling_weight +
                        dev_expertise_weight + path_complexity_weight)

        if total_weight == 0:
            # Default to equal weights if all weights are zero
            entry_point_weight = coupling_weight = dev_expertise_weight = path_complexity_weight = 0.25
        else:
            # Normalize weights
            entry_point_weight /= total_weight
            coupling_weight /= total_weight
            dev_expertise_weight /= total_weight
            path_complexity_weight /= total_weight

        # Calculate weighted sum
        risk_score = (
                entry_point_weight * entry_point_score +
                coupling_weight * coupling_score +
                dev_expertise_weight * dev_expertise_score +
                path_complexity_weight * path_complexity_score
        )

        # Apply non-linear risk adjustment
        # This emphasizes higher risk scores
        adjusted_risk = self._adjust_risk_curve(risk_score)

        return adjusted_risk

    def categorize_risk(self, risk_score: float) -> str:
        """
        Categorize a risk score into one of the predefined categories.

        Args:
            risk_score: Risk score (0.0-1.0)

        Returns:
            Risk category: 'very_high', 'high', 'medium', 'low', 'very_low'
        """
        # Validate input
        risk_score = self._validate_score(risk_score)

        # Determine category based on thresholds
        for category, threshold in sorted(
                self.RISK_CATEGORIES.items(),
                key=lambda x: x[1],
                reverse=True
        ):
            if risk_score >= threshold:
                return category

        # Default case (should never happen if thresholds are set correctly)
        return "very_low"

    def calculate_combined_risk(
            self,
            risks: Dict[str, float],
            weights: Optional[Dict[str, float]] = None
    ) -> float:
        """
        Calculate a combined risk score from multiple component risks.

        Args:
            risks: Dictionary mapping risk factors to scores
            weights: Optional dictionary mapping risk factors to weights

        Returns:
            Combined risk score (0.0-1.0)
        """
        if not risks:
            return 0.0

        # Use equal weights if not specified
        if weights is None:
            weights = {factor: 1.0 / len(risks) for factor in risks}

        # Normalize weights to sum to 1.0
        total_weight = sum(weights.values())
        if total_weight > 0:
            norm_weights = {
                factor: weight / total_weight
                for factor, weight in weights.items()
            }
        else:
            # Equal weights if sum is zero
            norm_weights = {factor: 1.0 / len(risks) for factor in risks}

        # Calculate weighted sum
        weighted_sum = sum(
            risks.get(factor, 0.0) * norm_weights.get(factor, 0.0)
            for factor in set(risks.keys()).union(weights.keys())
        )

        # Validate final score
        return self._validate_score(weighted_sum)

    def risk_factors_contribution(
            self,
            entry_point_score: float,
            coupling_score: float,
            dev_expertise_score: float,
            path_complexity_score: float,
            entry_point_weight: float = 0.5,
            coupling_weight: float = 0.3,
            dev_expertise_weight: float = 0.3,
            path_complexity_weight: float = 0.2
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate contribution of each risk factor to the overall score.

        Args:
            entry_point_score: Score indicating entry point importance (0.0-1.0)
            coupling_score: Score indicating change coupling risk (0.0-1.0)
            dev_expertise_score: Score indicating developer expertise risk (0.0-1.0)
            path_complexity_score: Score indicating path complexity risk (0.0-1.0)
            entry_point_weight: Weight for entry point factor
            coupling_weight: Weight for coupling factor
            dev_expertise_weight: Weight for developer expertise factor
            path_complexity_weight: Weight for path complexity factor

        Returns:
            Dictionary with raw values, normalized weights, weighted scores
        """
        # Validate inputs
        entry_point_score = self._validate_score(entry_point_score)
        coupling_score = self._validate_score(coupling_score)
        dev_expertise_score = self._validate_score(dev_expertise_score)
        path_complexity_score = self._validate_score(path_complexity_score)

        # Normalize weights
        total_weight = (entry_point_weight + coupling_weight +
                        dev_expertise_weight + path_complexity_weight)

        if total_weight == 0:
            norm_entry_weight = norm_coupling_weight = 0.25
            norm_expertise_weight = norm_path_weight = 0.25
        else:
            norm_entry_weight = entry_point_weight / total_weight
            norm_coupling_weight = coupling_weight / total_weight
            norm_expertise_weight = dev_expertise_weight / total_weight
            norm_path_weight = path_complexity_weight / total_weight

        # Calculate weighted scores
        weighted_entry = entry_point_score * norm_entry_weight
        weighted_coupling = coupling_score * norm_coupling_weight
        weighted_expertise = dev_expertise_score * norm_expertise_weight
        weighted_path = path_complexity_score * norm_path_weight

        # Calculate overall risk
        total_risk = self.calculate_risk(
            entry_point_score,
            coupling_score,
            dev_expertise_score,
            path_complexity_score,
            entry_point_weight,
            coupling_weight,
            dev_expertise_weight,
            path_complexity_weight
        )

        # Calculate percentage contributions
        if total_risk > 0:
            entry_contribution = (weighted_entry / total_risk) * 100
            coupling_contribution = (weighted_coupling / total_risk) * 100
            expertise_contribution = (weighted_expertise / total_risk) * 100
            path_contribution = (weighted_path / total_risk) * 100
        else:
            # Equal contribution if total risk is zero
            entry_contribution = coupling_contribution = 25.0
            expertise_contribution = path_contribution = 25.0

        return {
            "raw_scores": {
                "entry_point": entry_point_score,
                "coupling": coupling_score,
                "dev_expertise": dev_expertise_score,
                "path_complexity": path_complexity_score
            },
            "norm_weights": {
                "entry_point": norm_entry_weight,
                "coupling": norm_coupling_weight,
                "dev_expertise": norm_expertise_weight,
                "path_complexity": norm_path_weight
            },
            "weighted_scores": {
                "entry_point": weighted_entry,
                "coupling": weighted_coupling,
                "dev_expertise": weighted_expertise,
                "path_complexity": weighted_path
            },
            "contributions": {
                "entry_point": entry_contribution,
                "coupling": coupling_contribution,
                "dev_expertise": expertise_contribution,
                "path_complexity": path_contribution
            },
            "total_risk": total_risk
        }

    def _validate_score(self, score: float) -> float:
        """
        Ensure a score is within valid range (0.0-1.0).

        Args:
            score: Input score to validate

        Returns:
            Validated score within range (0.0-1.0)
        """
        return max(0.0, min(1.0, float(score)))

    def _adjust_risk_curve(self, risk_score: float) -> float:
        """
        Apply a non-linear adjustment to the risk curve.

        This applies a sigmoid-like function to emphasize medium-to-high risks
        and de-emphasize very low risks.

        Args:
            risk_score: Linear risk score (0.0-1.0)

        Returns:
            Adjusted risk score (0.0-1.0)
        """
        # Apply sigmoid-like function centered at 0.5
        # This makes mid-range risks more pronounced
        adjusted = 1.0 / (1.0 + math.exp(-10 * (risk_score - 0.5)))

        # Blend linear and sigmoid components for a more balanced curve
        blended = 0.7 * adjusted + 0.3 * risk_score

        return self._validate_score(blended)


def normalize_risk_scores(risk_scores: Dict[str, float]) -> Dict[str, float]:
    """
    Normalize risk scores to maintain relative ordering but use full range.

    Args:
        risk_scores: Dictionary mapping items to risk scores

    Returns:
        Dictionary with normalized risk scores
    """
    # Handle empty or single-item dictionaries
    if not risk_scores:
        return {}

    if len(risk_scores) == 1:
        return {k: 0.5 for k, v in risk_scores.items()}

    # Get min and max values
    min_val = min(risk_scores.values())
    max_val = max(risk_scores.values())

    # If all values are the same, return mid-range values
    if min_val == max_val:
        return {k: 0.5 for k, v in risk_scores.items()}

    # Normalize to range [0.1, 0.9] to avoid extremes
    normalized = {}
    range_val = max_val - min_val

    for k, v in risk_scores.items():
        normalized[k] = 0.1 + 0.8 * ((v - min_val) / range_val)

    return normalized


if __name__ == "__main__":
    # Example usage
    calculator = DefectRiskCalculator()

    # Example risk factors
    entry_point = 0.8  # High entry point score
    coupling = 0.6  # Medium-high coupling
    expertise = 0.4  # Medium developer expertise risk
    complexity = 0.3  # Low-medium path complexity

    # Calculate risk with default weights
    risk = calculator.calculate_risk(
        entry_point_score=entry_point,
        coupling_score=coupling,
        dev_expertise_score=expertise,
        path_complexity_score=complexity
    )

    print(f"Overall risk score: {risk:.4f}")
    print(f"Risk category: {calculator.categorize_risk(risk)}")

    # Calculate factor contributions
    contributions = calculator.risk_factors_contribution(
        entry_point_score=entry_point,
        coupling_score=coupling,
        dev_expertise_score=expertise,
        path_complexity_score=complexity
    )

    print("\nRisk factor contributions:")
    for factor, contribution in contributions["contributions"].items():
        print(f"  {factor}: {contribution:.2f}%")