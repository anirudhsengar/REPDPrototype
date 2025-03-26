#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for REPD Model Core Functionality

This module contains tests for the core REPD model functionality,
including repository analysis, risk scoring, and trend identification.

Author: anirudhsengar
"""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from repd.model import REPDModel
from repd.repository import Repository
from repd.structure_mapper import StructureMapper


class TestREPDModel(unittest.TestCase):
    """Test cases for REPD Model functionality."""

    def setUp(self):
        """Set up test environment before each test."""
        # Create a temporary directory for test outputs
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_path = Path(self.temp_dir.name)

        # Mock repository object
        self.mock_repo = MagicMock(spec=Repository)
        self.mock_repo.get_name.return_value = "test-repo"
        self.mock_repo.get_all_files.return_value = [
            "src/main.py",
            "src/utils.py",
            "src/models/model.py",
            "tests/test_main.py"
        ]
        self.mock_repo.is_code_file.return_value = True
        self.mock_repo.get_file_content.return_value = "def test_function():\n    pass"
        self.mock_repo.get_file_size.return_value = 100

        # Create model instance with mock repository
        self.model = REPDModel(self.mock_repo)

    def tearDown(self):
        """Clean up after each test."""
        self.temp_dir.cleanup()

    def test_init(self):
        """Test model initialization."""
        self.assertEqual(self.model.repository, self.mock_repo)
        self.assertIsNone(self.model.structure_mapper)
        self.assertEqual(self.model.config, {})
        self.assertEqual(self.model.results, {})

    def test_configure_default(self):
        """Test model configuration with defaults."""
        self.model.configure()

        # Check default configuration values
        self.assertIn("risk_weights", self.model.config)
        self.assertIn("max_files", self.model.config)
        self.assertIn("min_history_days", self.model.config)

    def test_configure_custom(self):
        """Test model configuration with custom parameters."""
        custom_config = {
            "risk_weights": {
                "complexity": 0.5,
                "churn": 0.3,
                "coupling": 0.2
            },
            "max_files": 500,
            "min_history_days": 60
        }

        self.model.configure(**custom_config)

        # Check custom configuration values
        self.assertEqual(self.model.config["risk_weights"]["complexity"], 0.5)
        self.assertEqual(self.model.config["max_files"], 500)
        self.assertEqual(self.model.config["min_history_days"], 60)

    @patch.object(StructureMapper, 'map_structure')
    def test_analyze_structure(self, mock_map_structure):
        """Test repository structure analysis."""
        # Configure mock structure mapper
        mock_map_structure.return_value = MagicMock()

        # Run analysis
        self.model.configure()
        self.model.analyze_structure()

        # Verify structure mapper was created and used
        self.assertIsNotNone(self.model.structure_mapper)
        mock_map_structure.assert_called_once()

    def test_calculate_risk_scores(self):
        """Test risk score calculation."""
        # Setup necessary components
        self.model.configure()
        self.model.structure_mapper = MagicMock(spec=StructureMapper)
        self.model.structure_mapper.get_central_files.return_value = [
            ("src/main.py", 0.8),
            ("src/utils.py", 0.5)
        ]

        # Mock complexity analysis
        mock_complexity = {
            "src/main.py": {"complexity": 0.7, "loc": 150},
            "src/utils.py": {"complexity": 0.4, "loc": 80}
        }
        self.model._analyze_complexity = MagicMock(return_value=mock_complexity)

        # Mock change history analysis
        mock_history = {
            "src/main.py": {"churn": 0.9, "commits": 20},
            "src/utils.py": {"churn": 0.3, "commits": 5}
        }
        self.model._analyze_change_history = MagicMock(return_value=mock_history)

        # Run risk score calculation
        self.model.calculate_risk_scores()

        # Verify risk scores were calculated
        self.assertIn("risk_scores", self.model.results)
        self.assertIn("src/main.py", self.model.results["risk_scores"])
        self.assertIn("src/utils.py", self.model.results["risk_scores"])

        # Verify risk factors were stored
        self.assertIn("risk_factors", self.model.results)

        # Check that risk scores are between 0 and 1
        for file, score in self.model.results["risk_scores"].items():
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)

    @patch("repd.visualization.visualize_results")
    def test_visualize(self, mock_visualize):
        """Test visualization generation."""
        # Mock visualization results
        mock_visualize.return_value = {
            "risk": str(self.output_path / "risk_scores.png"),
            "coupling": str(self.output_path / "change_coupling_network.png")
        }

        # Add some mock results
        self.model.results = {
            "risk_scores": {"src/main.py": 0.8, "src/utils.py": 0.4},
            "coupling_matrix": {"src/main.py": {"src/utils.py": 0.6}}
        }

        # Run visualization
        viz_files = self.model.visualize(output_dir=self.output_path)

        # Verify visualization was called with correct parameters
        mock_visualize.assert_called_once()
        self.assertEqual(viz_files, mock_visualize.return_value)

    def test_identify_hotspots(self):
        """Test hotspot identification."""
        # Add mock risk scores
        self.model.results = {
            "risk_scores": {
                "src/main.py": 0.9,
                "src/utils.py": 0.4,
                "src/models/model.py": 0.8,
                "tests/test_main.py": 0.2
            }
        }

        # Run hotspot identification
        hotspots = self.model.identify_hotspots(threshold=0.7)

        # Verify hotspots identified correctly
        self.assertEqual(len(hotspots), 2)
        self.assertIn("src/main.py", [file for file, _ in hotspots])
        self.assertIn("src/models/model.py", [file for file, _ in hotspots])

    @pytest.mark.parametrize("file_list,expected", [
        (["src/main.py", "src/utils.py"], True),  # Files exist
        (["non_existent.py"], False),  # File doesn't exist
        ([], True)  # Empty list (vacuously true)
    ])
    def test_validate_files(self, file_list, expected):
        """Test file validation with parameterized inputs."""
        # Configure mock repository response
        self.mock_repo.file_exists = lambda f: f in [
            "src/main.py", "src/utils.py", "src/models/model.py", "tests/test_main.py"
        ]

        # Run validation
        result = self.model.validate_files(file_list)

        # Check result
        self.assertEqual(result, expected)

    def test_save_and_load_results(self):
        """Test saving and loading analysis results."""
        # Add mock results
        self.model.results = {
            "risk_scores": {"src/main.py": 0.8, "src/utils.py": 0.4},
            "analyzed_at": "2025-03-26 06:59:59"
        }

        # Save results
        save_path = self.output_path / "results.json"
        self.model.save_results(save_path)

        # Verify file was created
        self.assertTrue(save_path.exists())

        # Create new model instance
        new_model = REPDModel(self.mock_repo)

        # Load results
        new_model.load_results(save_path)

        # Verify results loaded correctly
        self.assertEqual(
            new_model.results["risk_scores"]["src/main.py"],
            self.model.results["risk_scores"]["src/main.py"]
        )

    def test_analyze_trend(self):
        """Test trend analysis between two sets of results."""
        # Create two sets of results
        previous_results = {
            "risk_scores": {
                "src/main.py": 0.7,
                "src/utils.py": 0.3,
                "src/models/model.py": 0.6
            },
            "analyzed_at": "2025-03-25 06:59:59"
        }

        current_results = {
            "risk_scores": {
                "src/main.py": 0.8,
                "src/utils.py": 0.2,
                "src/models/model.py": 0.6,
                "src/new_file.py": 0.5
            },
            "analyzed_at": "2025-03-26 06:59:59"
        }

        # Run trend analysis
        trends = self.model.analyze_trend(previous_results, current_results)

        # Verify trend analysis results
        self.assertIn("improving", trends)
        self.assertIn("worsening", trends)
        self.assertIn("unchanged", trends)
        self.assertIn("new", trends)
        self.assertIn("removed", trends)

        # Check specific files
        self.assertIn("src/main.py", trends["worsening"])
        self.assertIn("src/utils.py", trends["improving"])
        self.assertIn("src/models/model.py", trends["unchanged"])
        self.assertIn("src/new_file.py", trends["new"])


if __name__ == '__main__':
    unittest.main()