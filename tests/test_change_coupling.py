#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for Change Coupling Analysis in REPD Model

This module contains tests for the change coupling analysis functionality,
which identifies files that frequently change together in commits and may
indicate hidden dependencies or architectural issues.

Author: anirudhsengar
"""

import os
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import networkx as nx
import numpy as np
import pytest

from repd.change_coupling_analyzer import ChangeCouplingAnalyzer
from repd.repository import Commit, Repository


class TestChangeCouplingAnalyzer(unittest.TestCase):
    """Test cases for Change Coupling Analysis."""

    def setUp(self):
        """Set up test environment before each test."""
        # Create a mock repository
        self.mock_repo = MagicMock(spec=Repository)
        self.mock_repo.get_name.return_value = "test-repo"

        # Create sample files for the mock repository
        self.sample_files = [
            "src/main.py",
            "src/utils.py",
            "src/api/endpoints.py",
            "src/models/user.py",
            "src/models/product.py",
            "src/templates/index.html",
            "src/static/styles.css",
            "tests/test_main.py",
        ]

        self.mock_repo.get_all_files.return_value = self.sample_files

        # Create sample commits for the mock repository
        base_date = datetime.now() - timedelta(days=30)

        self.sample_commits = [
            # Commit 1: Changes to main.py and utils.py
            Commit(
                hash="abc123",
                author="dev1",
                date=base_date,
                message="Initial setup",
                modified_files=["src/main.py", "src/utils.py"],
            ),
            # Commit 2: Changes to API and models
            Commit(
                hash="def456",
                author="dev2",
                date=base_date + timedelta(days=1),
                message="Implement API endpoints",
                modified_files=["src/api/endpoints.py", "src/models/user.py"],
            ),
            # Commit 3: Changes to API, models and templates
            Commit(
                hash="ghi789",
                author="dev1",
                date=base_date + timedelta(days=2),
                message="User API implementation",
                modified_files=[
                    "src/api/endpoints.py",
                    "src/models/user.py",
                    "src/templates/index.html",
                ],
            ),
            # Commit 4: Changes to main and API
            Commit(
                hash="jkl012",
                author="dev3",
                date=base_date + timedelta(days=3),
                message="Connect main to API",
                modified_files=["src/main.py", "src/api/endpoints.py"],
            ),
            # Commit 5: Changes to models only
            Commit(
                hash="mno345",
                author="dev2",
                date=base_date + timedelta(days=5),
                message="Product model updates",
                modified_files=["src/models/product.py"],
            ),
            # Commit 6: Multiple file changes
            Commit(
                hash="pqr678",
                author="dev1",
                date=base_date + timedelta(days=7),
                message="Frontend updates",
                modified_files=[
                    "src/templates/index.html",
                    "src/static/styles.css",
                    "src/main.py",
                ],
            ),
            # Commit 7: More model changes with API
            Commit(
                hash="stu901",
                author="dev3",
                date=base_date + timedelta(days=8),
                message="API enhancements",
                modified_files=[
                    "src/api/endpoints.py",
                    "src/models/user.py",
                    "src/models/product.py",
                ],
            ),
            # Commit 8: Test updates
            Commit(
                hash="vwx234",
                author="dev2",
                date=base_date + timedelta(days=10),
                message="Add tests",
                modified_files=["tests/test_main.py", "src/main.py"],
            ),
        ]

        self.mock_repo.get_commit_history.return_value = self.sample_commits

        # Create the change coupling analyzer
        self.analyzer = ChangeCouplingAnalyzer(self.mock_repo)

    def test_init(self):
        """Test analyzer initialization."""
        self.assertEqual(self.analyzer.repository, self.mock_repo)
        self.assertIsNone(self.analyzer.coupling_matrix)
        self.assertIsNone(self.analyzer.coupling_graph)

    def test_analyze_coupling(self):
        """Test basic change coupling analysis."""
        # Run analysis
        self.analyzer.analyze_coupling()

        # Verify coupling matrix was created
        self.assertIsNotNone(self.analyzer.coupling_matrix)
        self.assertIsNotNone(self.analyzer.coupling_graph)

        # Check some expected couplings based on commit history
        # Files that changed together in commits should have coupling scores

        # src/api/endpoints.py and src/models/user.py changed together multiple times
        self.assertGreater(
            self.analyzer.get_coupling_score(
                "src/api/endpoints.py", "src/models/user.py"
            ),
            0,
        )

        # src/main.py and src/utils.py changed together
        self.assertGreater(
            self.analyzer.get_coupling_score("src/main.py", "src/utils.py"), 0
        )

    def test_get_coupling_score(self):
        """Test retrieval of coupling scores between files."""
        # Run analysis
        self.analyzer.analyze_coupling()

        # Get coupling score for files that changed together
        score = self.analyzer.get_coupling_score(
            "src/api/endpoints.py", "src/models/user.py"
        )

        # Score should be between 0 and 1
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 1)

        # Test for files that never changed together
        no_coupling_score = self.analyzer.get_coupling_score(
            "src/static/styles.css", "tests/test_main.py"
        )
        self.assertEqual(no_coupling_score, 0)

        # Test for invalid file
        invalid_score = self.analyzer.get_coupling_score(
            "src/main.py", "non_existent_file.py"
        )
        self.assertEqual(invalid_score, 0)

    def test_get_coupled_files(self):
        """Test getting files coupled with a specific file."""
        # Run analysis
        self.analyzer.analyze_coupling()

        # Get files coupled with API endpoints
        coupled_files = self.analyzer.get_coupled_files(
            "src/api/endpoints.py", min_score=0
        )

        # Should include user.py and product.py
        self.assertIn("src/models/user.py", coupled_files)

        # Test with threshold
        strongly_coupled = self.analyzer.get_coupled_files(
            "src/api/endpoints.py", min_score=0.7
        )

        # There may or may not be strongly coupled files depending on scoring algorithm
        # But the count should be less than or equal to total coupled files
        self.assertLessEqual(len(strongly_coupled), len(coupled_files))

    def test_get_coupled_clusters(self):
        """Test identification of coupled file clusters."""
        # Run analysis with cluster detection
        self.analyzer.analyze_coupling()
        clusters = self.analyzer.get_coupled_clusters(min_coupling=0.1)

        # Verify clusters were found
        self.assertGreater(len(clusters), 0)

        # Each cluster should be a list or set of files
        for cluster in clusters:
            self.assertGreater(len(cluster), 1)  # Each cluster has at least 2 files

        # Check if API and model files are in the same cluster as they change together
        api_file = "src/api/endpoints.py"
        model_file = "src/models/user.py"

        # Find if they're in the same cluster
        are_clustered = False
        for cluster in clusters:
            if api_file in cluster and model_file in cluster:
                are_clustered = True
                break

        self.assertTrue(are_clustered)

    def test_get_coupling_matrix(self):
        """Test retrieval of the full coupling matrix."""
        # Run analysis
        self.analyzer.analyze_coupling()

        # Get coupling matrix
        matrix = self.analyzer.get_coupling_matrix()

        # Should be a dictionary
        self.assertIsInstance(matrix, dict)

        # Should contain entries for all files
        for file in self.sample_files:
            self.assertIn(file, matrix)

    def test_get_high_coupling_pairs(self):
        """Test identification of highly coupled file pairs."""
        # Run analysis
        self.analyzer.analyze_coupling()

        # Get highly coupled pairs
        pairs = self.analyzer.get_high_coupling_pairs(threshold=0.3)

        # Verify return format
        for pair in pairs:
            self.assertEqual(len(pair), 3)  # (file1, file2, score)
            self.assertGreaterEqual(pair[2], 0.3)  # Score >= threshold

    def test_export_coupling_data(self):
        """Test exporting coupling data to a file."""
        # Run analysis
        self.analyzer.analyze_coupling()

        # Create a temporary directory for test output
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "coupling_data.json"

            # Export coupling data
            self.analyzer.export_coupling_data(output_path)

            # Verify file was created
            self.assertTrue(output_path.exists())

            # Verify file content (basic check)
            with open(output_path, "r") as f:
                content = f.read()
                self.assertIn("coupling_matrix", content)

    def test_analyze_with_time_window(self):
        """Test coupling analysis with a time window."""
        # Run analysis with 5-day window
        self.analyzer.analyze_coupling(days=5)
        recent_matrix = self.analyzer.get_coupling_matrix()

        # Reset and run with all-time window
        self.analyzer = ChangeCouplingAnalyzer(self.mock_repo)
        self.analyzer.analyze_coupling()
        all_time_matrix = self.analyzer.get_coupling_matrix()

        # The recent matrix should have fewer or equal entries
        # since it only considers a subset of commits
        recent_pairs = sum(len(files) for files in recent_matrix.values())
        all_time_pairs = sum(len(files) for files in all_time_matrix.values())

        self.assertLessEqual(recent_pairs, all_time_pairs)

    def test_normalized_scores(self):
        """Test that coupling scores are properly normalized."""
        # Run analysis
        self.analyzer.analyze_coupling(normalize=True)

        # Check all scores are between 0 and 1
        for file1, couplings in self.analyzer.coupling_matrix.items():
            for file2, score in couplings.items():
                self.assertGreaterEqual(score, 0)
                self.assertLessEqual(score, 1)

    def test_graph_metrics(self):
        """Test calculation of graph-based coupling metrics."""
        # Run analysis
        self.analyzer.analyze_coupling()

        # Calculate coupling density
        density = self.analyzer.calculate_coupling_density()

        # Density should be between 0 and 1
        self.assertGreaterEqual(density, 0)
        self.assertLessEqual(density, 1)

        # Calculate coupling centrality
        centrality = self.analyzer.calculate_coupling_centrality()

        # Should return values for all files
        for file in self.sample_files:
            self.assertIn(file, centrality)

    def test_temporal_coupling_decay(self):
        """Test temporal coupling with decay factor."""
        # Run analysis with decay factor
        self.analyzer.analyze_coupling(temporal_decay=0.9)
        decayed_matrix = self.analyzer.get_coupling_matrix()

        # Reset and run without decay
        self.analyzer = ChangeCouplingAnalyzer(self.mock_repo)
        self.analyzer.analyze_coupling(temporal_decay=None)
        regular_matrix = self.analyzer.get_coupling_matrix()

        # With decay, older couplings should have less impact
        # This is hard to assert without knowing exact algorithm details
        # But we can check the API works without errors
        self.assertIsNotNone(decayed_matrix)
        self.assertIsNotNone(regular_matrix)

    def test_filter_by_file_type(self):
        """Test filtering coupling analysis by file type."""
        # Run analysis with file type filter
        self.analyzer.analyze_coupling(file_extensions=[".py"])
        py_matrix = self.analyzer.get_coupling_matrix()

        # Check that only Python files are in the matrix
        for file in py_matrix:
            self.assertTrue(file.endswith(".py"))

        # No HTML or CSS files should be present
        self.assertNotIn("src/templates/index.html", py_matrix)
        self.assertNotIn("src/static/styles.css", py_matrix)


if __name__ == "__main__":
    unittest.main()
