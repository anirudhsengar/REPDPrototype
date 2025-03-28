#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for Command Line Interface in REPD Model

This module contains tests for the REPD CLI functionality,
including command parsing, argument handling, and execution
of analysis operations through the command line.

Author: anirudhsengar
"""

import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

from repd.cli import main, parse_args, run_analysis, visualize_results
from repd.model import REPDModel
from repd.repository import Repository, GitRepository, LocalRepository


class TestCLI(unittest.TestCase):
    """Test cases for REPD Command Line Interface."""

    def setUp(self):
        """Set up test environment before each test."""
        # Create temporary directory for test outputs
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_dir = Path(self.temp_dir.name)

        # Create a mock repository path for tests
        self.repo_path = self.output_dir / "test_repo"
        self.repo_path.mkdir(exist_ok=True)

        # Create some mock files
        (self.repo_path / "file1.py").write_text("print('hello')")
        (self.repo_path / "file2.py").write_text("def test(): pass")

    def tearDown(self):
        """Clean up after each test."""
        self.temp_dir.cleanup()

    def test_parse_args_local(self):
        """Test argument parsing for local repository analysis."""
        # Test basic local repository analysis
        args = parse_args(
            [
                "analyze",
                "--local",
                str(self.repo_path),
                "--output",
                str(self.output_dir),
            ]
        )

        self.assertEqual(args.command, "analyze")
        self.assertEqual(args.local, str(self.repo_path))
        self.assertEqual(args.output, str(self.output_dir))
        self.assertFalse(args.git)

    def test_parse_args_git(self):
        """Test argument parsing for Git repository analysis."""
        # Test Git repository analysis
        args = parse_args(
            [
                "analyze",
                "--git",
                "https://github.com/user/repo.git",
                "--output",
                str(self.output_dir),
            ]
        )

        self.assertEqual(args.command, "analyze")
        self.assertEqual(args.git, "https://github.com/user/repo.git")
        self.assertEqual(args.output, str(self.output_dir))
        self.assertFalse(args.local)

    def test_parse_args_visualize(self):
        """Test argument parsing for visualization command."""
        # Test visualization command
        args = parse_args(
            [
                "visualize",
                "--input",
                str(self.output_dir / "results.json"),
                "--output",
                str(self.output_dir / "viz"),
            ]
        )

        self.assertEqual(args.command, "visualize")
        self.assertEqual(args.input, str(self.output_dir / "results.json"))
        self.assertEqual(args.output, str(self.output_dir / "viz"))

    def test_parse_args_config(self):
        """Test argument parsing with configuration options."""
        # Test configuration options
        args = parse_args(
            [
                "analyze",
                "--local",
                str(self.repo_path),
                "--output",
                str(self.output_dir),
                "--config",
                str(self.output_dir / "config.json"),
                "--max-files",
                "100",
                "--history-days",
                "30",
            ]
        )

        self.assertEqual(args.config, str(self.output_dir / "config.json"))
        self.assertEqual(args.max_files, 100)
        self.assertEqual(args.history_days, 30)

    def test_parse_args_report(self):
        """Test argument parsing for report generation."""
        # Test report command
        args = parse_args(
            [
                "report",
                "--input",
                str(self.output_dir / "results.json"),
                "--output",
                str(self.output_dir / "report.html"),
                "--template",
                "default",
            ]
        )

        self.assertEqual(args.command, "report")
        self.assertEqual(args.input, str(self.output_dir / "results.json"))
        self.assertEqual(args.output, str(self.output_dir / "report.html"))
        self.assertEqual(args.template, "default")

    @patch("repd.cli.LocalRepository")
    @patch("repd.cli.REPDModel")
    def test_run_analysis_local(self, mock_model_class, mock_repo_class):
        """Test running analysis on local repository."""
        # Setup mocks
        mock_repo = MagicMock()
        mock_repo_class.return_value = mock_repo

        mock_model = MagicMock()
        mock_model_class.return_value = mock_model

        # Create args for local analysis
        args = MagicMock()
        args.local = str(self.repo_path)
        args.git = None
        args.output = str(self.output_dir)
        args.max_files = 500
        args.history_days = 90
        args.config = None
        args.skip_viz = False

        # Run analysis
        run_analysis(args)

        # Verify repository was created correctly
        mock_repo_class.assert_called_once_with(self.repo_path)

        # Verify model was created with the repository
        mock_model_class.assert_called_once_with(mock_repo)

        # Verify analysis methods were called
        mock_model.configure.assert_called_once()
        mock_model.analyze_structure.assert_called_once()
        mock_model.calculate_risk_scores.assert_called_once()
        mock_model.save_results.assert_called_once()
        mock_model.visualize.assert_called_once()

    @patch("repd.cli.GitRepository")
    @patch("repd.cli.REPDModel")
    def test_run_analysis_git(self, mock_model_class, mock_repo_class):
        """Test running analysis on Git repository."""
        # Setup mocks
        mock_repo = MagicMock()
        mock_repo_class.return_value = mock_repo

        mock_model = MagicMock()
        mock_model_class.return_value = mock_model

        # Create args for Git analysis
        args = MagicMock()
        args.local = None
        args.git = "https://github.com/user/repo.git"
        args.output = str(self.output_dir)
        args.max_files = 500
        args.history_days = 90
        args.config = None
        args.skip_viz = True  # Skip visualization

        # Run analysis
        run_analysis(args)

        # Verify repository was created correctly
        mock_repo_class.assert_called_once_with("https://github.com/user/repo.git")

        # Verify model was created with the repository
        mock_model_class.assert_called_once_with(mock_repo)

        # Verify analysis methods were called
        mock_model.configure.assert_called_once()
        mock_model.analyze_structure.assert_called_once()
        mock_model.calculate_risk_scores.assert_called_once()
        mock_model.save_results.assert_called_once()

        # Visualization should be skipped
        mock_model.visualize.assert_not_called()

    @patch("repd.cli.json.load")
    @patch("repd.cli.REPDModel")
    def test_run_analysis_with_config(self, mock_model_class, mock_json_load):
        """Test running analysis with custom configuration file."""
        # Setup mocks
        mock_model = MagicMock()
        mock_model_class.return_value = mock_model

        # Setup mock config
        mock_config = {
            "risk_weights": {
                "complexity": 0.4,
                "churn": 0.3,
                "coupling": 0.2,
                "structural": 0.1,
            },
            "max_files": 200,
        }
        mock_json_load.return_value = mock_config

        # Create args with config
        args = MagicMock()
        args.local = str(self.repo_path)
        args.git = None
        args.output = str(self.output_dir)
        args.config = str(self.output_dir / "config.json")
        args.max_files = 500  # Should be overridden by config
        args.history_days = 90
        args.skip_viz = False

        # Create mock config file
        config_file = self.output_dir / "config.json"
        config_file.write_text(
            '{"risk_weights": {"complexity": 0.4, "churn": 0.3, "coupling": 0.2, "structural": 0.1}, "max_files": 200}'
        )

        # Run analysis
        with patch(
            "repd.cli.open", unittest.mock.mock_open(read_data=config_file.read_text())
        ):
            run_analysis(args)

        # Verify configure was called with the config settings
        mock_model.configure.assert_called_once()
        call_kwargs = mock_model.configure.call_args[1]
        self.assertEqual(call_kwargs["max_files"], 200)  # From config
        self.assertEqual(call_kwargs["risk_weights"]["complexity"], 0.4)  # From config

    @patch("repd.cli.json.load")
    @patch("repd.cli.visualize_results")
    def test_visualize_command(self, mock_visualize_func, mock_json_load):
        """Test running the visualize command."""
        # Setup mock results
        mock_results = {"risk_scores": {"file1.py": 0.8, "file2.py": 0.4}}
        mock_json_load.return_value = mock_results

        # Create args for visualization
        args = MagicMock()
        args.input = str(self.output_dir / "results.json")
        args.output = str(self.output_dir / "viz")
        args.viz_types = ["risk", "coupling"]

        # Create mock results file
        results_file = self.output_dir / "results.json"
        results_file.write_text('{"risk_scores": {"file1.py": 0.8, "file2.py": 0.4}}')

        # Run visualization
        with patch(
            "repd.cli.open", unittest.mock.mock_open(read_data=results_file.read_text())
        ):
            visualize_results(args)

        # Verify visualization function was called correctly
        mock_visualize_func.assert_called_once()
        call_args = mock_visualize_func.call_args[0]
        self.assertEqual(call_args[0], mock_results)
        self.assertEqual(call_args[1], Path(args.output))
        self.assertEqual(call_args[2], args.viz_types)

    @patch("repd.cli.run_analysis")
    def test_main_analyze(self, mock_run_analysis):
        """Test main function with analyze command."""
        # Setup test arguments
        test_args = ["analyze", "--local", str(self.repo_path)]

        # Run main with test arguments
        with patch("sys.argv", ["repd_cli.py"] + test_args):
            main()

        # Verify run_analysis was called
        mock_run_analysis.assert_called_once()

    @patch("repd.cli.visualize_results")
    def test_main_visualize(self, mock_visualize):
        """Test main function with visualize command."""
        # Setup test arguments
        test_args = ["visualize", "--input", str(self.output_dir / "results.json")]

        # Run main with test arguments
        with patch("sys.argv", ["repd_cli.py"] + test_args):
            main()

        # Verify visualize_results was called
        mock_visualize.assert_called_once()

    @patch("repd.cli.generate_report")
    def test_main_report(self, mock_report):
        """Test main function with report command."""
        # Setup test arguments
        test_args = ["report", "--input", str(self.output_dir / "results.json")]

        # Run main with test arguments
        with patch("sys.argv", ["repd_cli.py"] + test_args):
            main()

        # Verify generate_report was called
        mock_report.assert_called_once()

    def test_main_invalid_command(self):
        """Test main function with invalid command."""
        # Setup test arguments with invalid command
        test_args = ["invalid", "--local", str(self.repo_path)]

        # Run main with invalid command - should exit with error
        with patch("sys.argv", ["repd_cli.py"] + test_args):
            with self.assertRaises(SystemExit):
                main()

    def test_create_output_directory(self):
        """Test creation of output directory."""
        # Remove the output directory
        new_output_dir = self.output_dir / "new_output"

        # Create args with non-existent output directory
        args = MagicMock()
        args.local = str(self.repo_path)
        args.git = None
        args.output = str(new_output_dir)
        args.max_files = 500
        args.history_days = 90
        args.config = None
        args.skip_viz = True

        # Run analysis with patched classes to prevent actual execution
        with patch("repd.cli.LocalRepository"), patch("repd.cli.REPDModel"):
            run_analysis(args)

        # Verify directory was created
        self.assertTrue(new_output_dir.exists())
        self.assertTrue(new_output_dir.is_dir())


if __name__ == "__main__":
    unittest.main()
