#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for Entry Point Detection in REPD Model

This module contains tests for the entry point detection functionality,
which identifies files that serve as primary interfaces or entry points
to the codebase.

Author: anirudhsengar
"""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import networkx as nx
import pytest

from repd.entry_point_detector import EntryPointDetector
from repd.repository import Repository
from repd.structure_mapper import StructureMapper


class TestEntryPointDetector(unittest.TestCase):
    """Test cases for REPD Entry Point Detection."""

    def setUp(self):
        """Set up test environment before each test."""
        # Create a mock repository
        self.mock_repo = MagicMock(spec=Repository)
        self.mock_repo.get_name.return_value = "test-repo"
        self.mock_repo.get_all_files.return_value = [
            "src/main.py",
            "src/app.py",
            "src/api/endpoints.py",
            "src/utils/helpers.py",
            "src/models/user.py",
            "tests/test_main.py"
        ]
        self.mock_repo.is_code_file.return_value = True
        self.mock_repo.normalize_path = lambda path: path

        # Mock file contents for different entry point patterns
        def mock_get_file_content(file_path):
            content_map = {
                "src/main.py": 'if __name__ == "__main__":\n    main()',
                "src/app.py": 'app = Flask(__name__)\n@app.route("/")\ndef index():\n    return "Hello"',
                "src/api/endpoints.py": '@api.route("/users")\ndef get_users():\n    pass',
                "src/utils/helpers.py": 'def helper():\n    pass',
                "src/models/user.py": 'class User:\n    pass',
                "tests/test_main.py": 'def test_main():\n    pass'
            }
            return content_map.get(file_path, "")

        self.mock_repo.get_file_content = mock_get_file_content

        # Create a mock structure mapper with dependency graph
        self.mock_structure_mapper = MagicMock(spec=StructureMapper)

        # Create a dependency graph for testing
        self.dependency_graph = nx.DiGraph()
        self.dependency_graph.add_nodes_from([
            "src/main.py",
            "src/app.py",
            "src/api/endpoints.py",
            "src/utils/helpers.py",
            "src/models/user.py",
            "tests/test_main.py"
        ])

        # Add edges (dependencies)
        self.dependency_graph.add_edges_from([
            ("src/main.py", "src/utils/helpers.py"),
            ("src/main.py", "src/models/user.py"),
            ("src/app.py", "src/api/endpoints.py"),
            ("src/app.py", "src/models/user.py"),
            ("src/api/endpoints.py", "src/utils/helpers.py"),
            ("src/api/endpoints.py", "src/models/user.py"),
            ("tests/test_main.py", "src/main.py")
        ])

        self.mock_structure_mapper.dependency_graph = self.dependency_graph

        # Create the entry point detector
        self.detector = EntryPointDetector(self.mock_repo, self.mock_structure_mapper)

    def test_init(self):
        """Test detector initialization."""
        self.assertEqual(self.detector.repository, self.mock_repo)
        self.assertEqual(self.detector.structure_mapper, self.mock_structure_mapper)
        self.assertEqual(self.detector.entry_points, {})

    def test_pattern_detection(self):
        """Test detection of entry points based on file patterns."""
        # Run detection
        self.detector.detect_entry_points()
        entry_points = self.detector.get_entry_points()

        # Verify that main.py is detected as an entry point
        self.assertIn("src/main.py", entry_points)
        self.assertGreater(entry_points["src/main.py"], 0)

        # Verify that app.py with Flask routes is detected
        self.assertIn("src/app.py", entry_points)
        self.assertGreater(entry_points["src/app.py"], 0)

    def test_api_endpoint_detection(self):
        """Test detection of API endpoints."""
        # Run detection
        self.detector.detect_entry_points()
        entry_points = self.detector.get_entry_points()

        # Verify that API endpoints are detected
        self.assertIn("src/api/endpoints.py", entry_points)
        self.assertGreater(entry_points["src/api/endpoints.py"], 0)

    def test_dependency_based_detection(self):
        """Test entry point detection based on dependency graph structure."""
        # Configure mock in-degree calculations
        in_degree = {
            "src/main.py": 1,  # Only test_main.py imports it
            "src/app.py": 0,  # Nothing imports it
            "src/api/endpoints.py": 1,
            "src/utils/helpers.py": 3,
            "src/models/user.py": 3,
            "tests/test_main.py": 0
        }

        # Apply in-degree to mock graph
        for node, degree in in_degree.items():
            self.dependency_graph.nodes[node]["in_degree"] = degree

        # Run detection with dependency analysis
        self.detector.detect_entry_points(use_dependencies=True)
        entry_points = self.detector.get_entry_points()

        # Files with 0 in-degree should be considered entry points
        self.assertIn("src/app.py", entry_points)
        self.assertIn("tests/test_main.py", entry_points)

        # Files with high in-degree (imported by many) should not be entry points
        # or should have lower scores
        if "src/utils/helpers.py" in entry_points:
            self.assertLess(
                entry_points["src/utils/helpers.py"],
                entry_points["src/app.py"]
            )

    def test_filename_convention_detection(self):
        """Test detection based on filename conventions."""
        # Add mock files with conventional entry point names
        additional_files = {
            "index.js": 'console.log("Hello World")',
            "server.py": 'server.start()',
            "cli.py": 'def main():\n    pass',
            "run.sh": '#!/bin/bash\necho "Running"'
        }

        all_files = self.mock_repo.get_all_files() + list(additional_files.keys())
        self.mock_repo.get_all_files.return_value = all_files

        original_get_file_content = self.mock_repo.get_file_content

        # Update mock get_file_content to include new files
        def extended_get_file_content(file_path):
            if file_path in additional_files:
                return additional_files[file_path]
            return original_get_file_content(file_path)

        self.mock_repo.get_file_content = extended_get_file_content

        # Run detection
        self.detector = EntryPointDetector(self.mock_repo, self.mock_structure_mapper)
        self.detector.detect_entry_points()
        entry_points = self.detector.get_entry_points()

        # Check that files with conventional entry point names are detected
        self.assertIn("index.js", entry_points)
        self.assertIn("server.py", entry_points)
        self.assertIn("cli.py", entry_points)

    def test_get_top_entry_points(self):
        """Test retrieval of top entry points."""
        # Set predefined entry points
        self.detector.entry_points = {
            "src/main.py": 0.9,
            "src/app.py": 0.85,
            "src/api/endpoints.py": 0.7,
            "index.js": 0.8,
            "cli.py": 0.6
        }

        # Get top 3 entry points
        top_entry_points = self.detector.get_top_entry_points(3)

        # Verify correct number and order
        self.assertEqual(len(top_entry_points), 3)
        self.assertEqual(top_entry_points[0][0], "src/main.py")
        self.assertEqual(top_entry_points[1][0], "src/app.py")
        self.assertEqual(top_entry_points[2][0], "index.js")

    def test_export_entry_points(self):
        """Test exporting entry points to a file."""
        # Set predefined entry points
        self.detector.entry_points = {
            "src/main.py": 0.9,
            "src/app.py": 0.85
        }

        # Create a temporary directory for test output
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "entry_points.json"

            # Export entry points
            self.detector.export_entry_points(output_path)

            # Verify file was created
            self.assertTrue(output_path.exists())

            # Verify file content (basic check)
            with open(output_path, 'r') as f:
                content = f.read()
                self.assertIn("src/main.py", content)
                self.assertIn("src/app.py", content)

    @patch.object(EntryPointDetector, '_analyze_file_content')
    def test_scan_for_signatures(self, mock_analyze):
        """Test scanning files for entry point signatures."""
        # Setup mock to return predefined scores
        mock_analyze.side_effect = lambda file: {
            "src/main.py": 0.9,
            "src/app.py": 0.8,
            "src/api/endpoints.py": 0.7,
            "src/utils/helpers.py": 0.1,
            "src/models/user.py": 0.0,
            "tests/test_main.py": 0.3
        }.get(file, 0.0)

        # Run scan
        self.detector._scan_files_for_signatures()

        # Verify call count
        self.assertEqual(mock_analyze.call_count, 6)  # One for each file

        # Verify entry point scores
        self.assertEqual(self.detector.entry_points["src/main.py"], 0.9)
        self.assertEqual(self.detector.entry_points["src/app.py"], 0.8)
        self.assertEqual(self.detector.entry_points["src/utils/helpers.py"], 0.1)

    def test_combine_scores(self):
        """Test combining different entry point detection methods."""
        # Set up pattern-based scores
        pattern_scores = {
            "src/main.py": 0.9,
            "src/app.py": 0.8,
            "src/api/endpoints.py": 0.7
        }

        # Set up dependency-based scores
        dependency_scores = {
            "src/main.py": 0.7,
            "src/app.py": 0.9,
            "index.js": 0.8
        }

        # Combine scores
        combined = self.detector._combine_scores(
            pattern_scores,
            dependency_scores,
            pattern_weight=0.6,
            dependency_weight=0.4
        )

        # Check combined scores
        self.assertAlmostEqual(combined["src/main.py"], 0.9 * 0.6 + 0.7 * 0.4)
        self.assertAlmostEqual(combined["src/app.py"], 0.8 * 0.6 + 0.9 * 0.4)
        self.assertAlmostEqual(combined["src/api/endpoints.py"], 0.7 * 0.6)
        self.assertAlmostEqual(combined["index.js"], 0.8 * 0.4)


if __name__ == '__main__':
    unittest.main()