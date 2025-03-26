#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for Risk Calculator in REPD Model

This module contains tests for the risk calculation functionality,
which computes risk scores for files based on various metrics including
complexity, change history, coupling, and structural importance.

Author: anirudhsengar
"""

import os
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from repd.risk_calculator import RiskCalculator
from repd.repository import Repository, Commit
from repd.structure_mapper import StructureMapper


class TestRiskCalculator(unittest.TestCase):
    """Test cases for REPD Risk Calculator."""

    def setUp(self):
        """Set up test environment before each test."""
        # Create a mock repository
        self.mock_repo = MagicMock(spec=Repository)
        self.mock_repo.get_name.return_value = "test-repo"

        # Sample files for testing
        self.sample_files = [
            "src/main.py",
            "src/utils.py",
            "src/complex_module.py",
            "src/stable_module.py",
            "src/new_module.py",
            "src/api/endpoints.py",
        ]

        self.mock_repo.get_all_files.return_value = self.sample_files

        # Mock file content with varying complexity
        def mock_get_file_content(file_path):
            content_map = {
                "src/main.py": "def main():\n    print('Hello')\n    return 0",
                "src/utils.py": "def util1():\n    pass\n\ndef util2():\n    pass",
                "src/complex_module.py": "def complex_func():\n    if x:\n        if y:\n            if z:\n                pass",
                "src/stable_module.py": "def stable():\n    return True",
                "src/new_module.py": "# New module\ndef new_func():\n    pass",
                "src/api/endpoints.py": "@app.route('/api')\ndef api():\n    pass"
            }
            return content_map.get(file_path, "")

        self.mock_repo.get_file_content = mock_get_file_content

        # Mock file size for code volume metrics
        def mock_get_file_size(file_path):
            size_map = {
                "src/main.py": 100,
                "src/utils.py": 200,
                "src/complex_module.py": 500,
                "src/stable_module.py": 50,
                "src/new_module.py": 30,
                "src/api/endpoints.py": 150
            }
            return size_map.get(file_path, 0)

        self.mock_repo.get_file_size = mock_get_file_size

        # Mock file age
        def mock_get_file_creation_date(file_path):
            now = datetime.now()
            date_map = {
                "src/main.py": now - timedelta(days=100),
                "src/utils.py": now - timedelta(days=90),
                "src/complex_module.py": now - timedelta(days=80),
                "src/stable_module.py": now - timedelta(days=200),
                "src/new_module.py": now - timedelta(days=1),
                "src/api/endpoints.py": now - timedelta(days=50)
            }
            return date_map.get(file_path, now)

        self.mock_repo.get_file_creation_date = mock_get_file_creation_date

        # Create sample commits for change history
        base_date = datetime.now() - timedelta(days=30)

        self.sample_commits = [
            # Frequent changes to complex_module.py
            Commit(
                hash="abc123",
                author="dev1",
                date=base_date - timedelta(days=25),
                message="Update complex module",
                modified_files=["src/complex_module.py"]
            ),
            Commit(
                hash="def456",
                author="dev2",
                date=base_date - timedelta(days=20),
                message="Fix complex module",
                modified_files=["src/complex_module.py", "src/utils.py"]
            ),
            Commit(
                hash="ghi789",
                author="dev1",
                date=base_date - timedelta(days=15),
                message="Another complex module fix",
                modified_files=["src/complex_module.py"]
            ),
            # Some changes to main.py and utils.py
            Commit(
                hash="jkl012",
                author="dev3",
                date=base_date - timedelta(days=10),
                message="Update main",
                modified_files=["src/main.py", "src/utils.py"]
            ),
            # Stable module rarely changes
            Commit(
                hash="mno345",
                author="dev2",
                date=base_date - timedelta(days=100),
                message="Initial stable module",
                modified_files=["src/stable_module.py"]
            ),
            # API endpoints change
            Commit(
                hash="pqr678",
                author="dev1",
                date=base_date - timedelta(days=5),
                message="API updates",
                modified_files=["src/api/endpoints.py", "src/main.py"]
            ),
            # New module just added
            Commit(
                hash="stu901",
                author="dev3",
                date=base_date - timedelta(days=1),
                message="Add new module",
                modified_files=["src/new_module.py"]
            )
        ]

        self.mock_repo.get_commit_history.return_value = self.sample_commits

        # Mock structure mapper
        self.mock_structure_mapper = MagicMock(spec=StructureMapper)

        # Mock centrality scores for structural importance
        centrality_scores = [
            ("src/main.py", 0.9),  # High centrality
            ("src/api/endpoints.py", 0.8),  # High centrality
            ("src/utils.py", 0.5),  # Medium centrality
            ("src/complex_module.py", 0.3),  # Lower centrality
            ("src/stable_module.py", 0.2),  # Lower centrality
            ("src/new_module.py", 0.1)  # Low centrality
        ]
        self.mock_structure_mapper.get_central_files.return_value = centrality_scores

        # Create the risk calculator
        self.risk_calculator = RiskCalculator(self.mock_repo, self.mock_structure_mapper)

    def test_init(self):
        """Test calculator initialization."""
        self.assertEqual(self.risk_calculator.repository, self.mock_repo)
        self.assertEqual(self.risk_calculator.structure_mapper, self.mock_structure_mapper)
        self.assertEqual(self.risk_calculator.risk_scores, {})
        self.assertEqual(self.risk_calculator.risk_factors, {})

    def test_calculate_risk_scores(self):
        """Test calculation of risk scores."""
        # Configure default weights
        weights = {
            "complexity": 0.25,
            "churn": 0.25,
            "coupling": 0.2,
            "structural": 0.2,
            "age": 0.1
        }

        # Calculate risk scores
        self.risk_calculator.calculate_risk_scores(weights=weights)

        # Verify risk scores were calculated
        self.assertGreater(len(self.risk_calculator.risk_scores), 0)

        # All files should have scores
        for file in self.sample_files:
            self.assertIn(file, self.risk_calculator.risk_scores)

        # All scores should be between 0 and 1
        for file, score in self.risk_calculator.risk_scores.items():
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)

        # Risk factors should be recorded
        self.assertGreater(len(self.risk_calculator.risk_factors), 0)

    def test_complexity_analysis(self):
        """Test calculation of code complexity metrics."""
        # Call complexity analysis directly
        complexity_scores = self.risk_calculator._analyze_complexity()

        # All files should have complexity scores
        for file in self.sample_files:
            self.assertIn(file, complexity_scores)

        # Complex module should have higher complexity
        self.assertGreater(
            complexity_scores["src/complex_module.py"],
            complexity_scores["src/stable_module.py"]
        )

    def test_churn_analysis(self):
        """Test calculation of code churn metrics."""
        # Call churn analysis directly
        churn_scores = self.risk_calculator._analyze_churn()

        # All files should have churn scores
        for file in self.sample_files:
            self.assertIn(file, churn_scores)

        # Complex module should have higher churn (more commits)
        self.assertGreater(
            churn_scores["src/complex_module.py"],
            churn_scores["src/stable_module.py"]
        )

        # New module should have lower churn (just added)
        self.assertLess(
            churn_scores["src/new_module.py"],
            churn_scores["src/complex_module.py"]
        )

    def test_coupling_analysis(self):
        """Test calculation of coupling risk."""
        # Mock coupling matrix from structure mapper
        coupling_matrix = {
            "src/main.py": {
                "src/utils.py": 0.7,
                "src/api/endpoints.py": 0.8
            },
            "src/utils.py": {
                "src/main.py": 0.7,
                "src/complex_module.py": 0.5
            },
            "src/complex_module.py": {
                "src/utils.py": 0.5
            },
            "src/api/endpoints.py": {
                "src/main.py": 0.8
            },
            "src/stable_module.py": {},
            "src/new_module.py": {}
        }

        # Set up mock return value
        self.mock_structure_mapper.import_map = coupling_matrix

        # Call coupling analysis directly
        coupling_scores = self.risk_calculator._analyze_coupling()

        # All files should have coupling scores
        for file in self.sample_files:
            self.assertIn(file, coupling_scores)

        # Files with more coupling should have higher scores
        self.assertGreater(
            coupling_scores["src/main.py"],
            coupling_scores["src/stable_module.py"]
        )

    def test_structural_analysis(self):
        """Test calculation of structural importance risk."""
        # Call structural analysis directly
        structural_scores = self.risk_calculator._analyze_structural_importance()

        # All files should have structural scores
        for file in self.sample_files:
            self.assertIn(file, structural_scores)

        # Main.py should have highest structural importance
        self.assertEqual(
            max(structural_scores.items(), key=lambda x: x[1])[0],
            "src/main.py"
        )

    def test_age_analysis(self):
        """Test calculation of age-related risk."""
        # Call age analysis directly
        age_scores = self.risk_calculator._analyze_age()

        # All files should have age scores
        for file in self.sample_files:
            self.assertIn(file, age_scores)

        # New module should have highest age-related risk
        self.assertEqual(
            max(age_scores.items(), key=lambda x: x[1])[0],
            "src/new_module.py"
        )

    def test_custom_weights(self):
        """Test risk calculation with custom weights."""
        # Standard weights
        standard_weights = {
            "complexity": 0.25,
            "churn": 0.25,
            "coupling": 0.2,
            "structural": 0.2,
            "age": 0.1
        }

        # Calculate with standard weights
        self.risk_calculator.calculate_risk_scores(weights=standard_weights)
        standard_scores = self.risk_calculator.risk_scores.copy()

        # Reset and calculate with complexity-focused weights
        self.risk_calculator.risk_scores = {}
        complexity_weights = {
            "complexity": 0.6,
            "churn": 0.1,
            "coupling": 0.1,
            "structural": 0.1,
            "age": 0.1
        }

        self.risk_calculator.calculate_risk_scores(weights=complexity_weights)
        complexity_scores = self.risk_calculator.risk_scores.copy()

        # The scores should be different with different weights
        # (at least for some files)
        differences = [
            abs(standard_scores[file] - complexity_scores[file])
            for file in self.sample_files
        ]

        self.assertTrue(any(diff > 0.01 for diff in differences))

    def test_get_risk_scores(self):
        """Test retrieval of risk scores."""
        # Calculate risk scores
        self.risk_calculator.calculate_risk_scores()

        # Get all scores
        all_scores = self.risk_calculator.get_risk_scores()
        self.assertEqual(len(all_scores), len(self.sample_files))

        # Get top N scores
        top_scores = self.risk_calculator.get_risk_scores(top_n=3)
        self.assertEqual(len(top_scores), 3)

        # Check that scores are sorted (highest first)
        self.assertGreaterEqual(top_scores[0][1], top_scores[-1][1])

    def test_get_risk_factors(self):
        """Test retrieval of risk factors."""
        # Calculate risk scores
        self.risk_calculator.calculate_risk_scores()

        # Get risk factors for a specific file
        factors = self.risk_calculator.get_risk_factors("src/main.py")

        # Should include all risk factors
        self.assertIn("complexity", factors)
        self.assertIn("churn", factors)
        self.assertIn("coupling", factors)
        self.assertIn("structural", factors)
        self.assertIn("age", factors)

        # Get risk factors for all files
        all_factors = self.risk_calculator.get_all_risk_factors()
        self.assertEqual(len(all_factors), len(self.sample_files))

    def test_export_import_risk_data(self):
        """Test exporting and importing risk data."""
        # Calculate risk scores
        self.risk_calculator.calculate_risk_scores()

        # Create a temporary directory for test output
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "risk_data.json"

            # Export risk data
            self.risk_calculator.export_risk_data(output_path)

            # Verify file was created
            self.assertTrue(output_path.exists())

            # Create a new calculator
            new_calculator = RiskCalculator(self.mock_repo, self.mock_structure_mapper)

            # Import risk data
            new_calculator.import_risk_data(output_path)

            # Verify data was imported correctly
            self.assertEqual(
                len(new_calculator.risk_scores),
                len(self.risk_calculator.risk_scores)
            )

            # Check specific scores
            for file in self.sample_files:
                self.assertAlmostEqual(
                    new_calculator.risk_scores.get(file, 0),
                    self.risk_calculator.risk_scores.get(file, 0)
                )

    def test_normalize_scores(self):
        """Test score normalization function."""
        # Create some test scores
        raw_scores = {
            "file1": 100,
            "file2": 50,
            "file3": 0,
            "file4": 200
        }

        # Normalize scores
        normalized = self.risk_calculator._normalize_scores(raw_scores)

        # Verify normalization
        self.assertEqual(normalized["file3"], 0.0)  # Min should be 0.0
        self.assertEqual(normalized["file4"], 1.0)  # Max should be 1.0
        self.assertEqual(normalized["file1"], 0.5)  # Should be 0.5

    def test_get_highest_risk_factors(self):
        """Test identification of highest risk factors."""
        # Calculate risk scores
        self.risk_calculator.calculate_risk_scores()

        # Get highest risk factor for each file
        high_factors = self.risk_calculator.get_highest_risk_factors()

        # Each file should have a highest factor identified
        self.assertEqual(len(high_factors), len(self.sample_files))

        # Each entry should contain file, factor name, and factor value
        for file, factor_data in high_factors.items():
            self.assertIn("factor", factor_data)
            self.assertIn("value", factor_data)

    def test_risk_threshold_classification(self):
        """Test classification of files based on risk thresholds."""
        # Calculate risk scores
        self.risk_calculator.calculate_risk_scores()

        # Classify files
        classification = self.risk_calculator.classify_risk(
            high_threshold=0.7,
            medium_threshold=0.4
        )

        # Should have three categories
        self.assertIn("high_risk", classification)
        self.assertIn("medium_risk", classification)
        self.assertIn("low_risk", classification)

        # All files should be classified
        total_files = (
                len(classification["high_risk"]) +
                len(classification["medium_risk"]) +
                len(classification["low_risk"])
        )
        self.assertEqual(total_files, len(self.sample_files))


if __name__ == '__main__':
    unittest.main()