#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Command Line Interface for REPD Model

This module provides a command-line interface for the REPD model,
allowing users to analyze repositories, visualize results, and generate reports.

Author: anirudhsengar
Date: 2025-03-26 08:42:30
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from repd.model import REPDModel
from repd.repository import GitRepository, LocalRepository, Repository
from repd.visualization import visualize_results

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args(args=None):
    """
    Parse command-line arguments.

    Args:
        args: Command-line arguments (defaults to sys.argv[1:])

    Returns:
        Parsed argument namespace
    """
    parser = argparse.ArgumentParser(
        description="REPD: Repository Engineering and Project Dynamics Analysis Tool",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    subparsers.required = True

    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze a repository")
    repo_group = analyze_parser.add_mutually_exclusive_group(required=True)
    repo_group.add_argument("--local", help="Path to local repository")
    repo_group.add_argument("--git", help="URL of Git repository")
    analyze_parser.add_argument(
        "--output", required=True, help="Output directory for results"
    )
    analyze_parser.add_argument("--config", help="Path to configuration JSON file")
    analyze_parser.add_argument(
        "--max-files", type=int, default=500, help="Maximum files to analyze"
    )
    analyze_parser.add_argument(
        "--history-days", type=int, default=90, help="Days of history to analyze"
    )
    analyze_parser.add_argument(
        "--skip-viz", action="store_true", help="Skip visualization generation"
    )
    analyze_parser.add_argument(
        "--report", help="Generate HTML report at specified path"
    )

    # Visualize command
    visualize_parser = subparsers.add_parser(
        "visualize", help="Visualize analysis results"
    )
    visualize_parser.add_argument(
        "--input", required=True, help="Path to analysis results JSON file"
    )
    visualize_parser.add_argument(
        "--output", required=True, help="Output directory for visualizations"
    )
    visualize_parser.add_argument(
        "--types",
        nargs="+",
        dest="viz_types",
        help="Types of visualizations to generate",
    )

    # Report command
    report_parser = subparsers.add_parser(
        "report", help="Generate report from analysis results"
    )
    report_parser.add_argument(
        "--input", required=True, help="Path to analysis results JSON file"
    )
    report_parser.add_argument("--output", required=True, help="Output path for report")
    report_parser.add_argument(
        "--template", default="default", help="Report template to use"
    )

    return parser.parse_args(args)


def run_analysis(args):
    """
    Run repository analysis based on command-line arguments.

    Args:
        args: Parsed command-line arguments
    """
    logger.info("Starting repository analysis")

    # Create output directory if it doesn't exist
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize repository
    if args.local:
        logger.info(f"Analyzing local repository at {args.local}")
        repository = LocalRepository(args.local)
    else:  # args.git
        logger.info(f"Analyzing Git repository at {args.git}")
        repository = GitRepository(args.git)

    # Create and configure model
    model = REPDModel(repository)

    # Load configuration from file if specified
    config = {}
    if args.config:
        try:
            with open(args.config, "r") as f:
                config = json.load(f)
                logger.info(f"Loaded configuration from {args.config}")
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")

    # Configure model
    model.configure(
        max_files=config.get("max_files", args.max_files),
        history_days=config.get("history_days", args.history_days),
        risk_weights=config.get("risk_weights"),
        coupling_threshold=config.get("coupling_threshold"),
        entry_point_min_score=config.get("entry_point_min_score"),
        exclude_patterns=config.get("exclude_patterns"),
    )

    # Run analysis
    logger.info("Running comprehensive analysis")
    model.analyze_structure()
    model.calculate_risk_scores()

    # Save results
    results_file = output_dir / "results.json"
    model.save_results(results_file)
    logger.info(f"Analysis results saved to {results_file}")

    # Generate visualizations if not skipped
    if not args.skip_viz:
        viz_dir = output_dir / "visualizations"
        viz_files = model.visualize(viz_dir)
        logger.info(f"Generated {len(viz_files)} visualizations in {viz_dir}")

    # Generate report if requested
    if args.report:
        report_path = Path(args.report)
        model.generate_report(report_path)
        logger.info(f"Generated report at {report_path}")

    logger.info("Analysis completed successfully")


def visualize_results(args):
    """
    Visualize analysis results based on command-line arguments.

    Args:
        args: Parsed command-line arguments
    """
    logger.info(f"Visualizing results from {args.input}")

    # Load results
    try:
        with open(args.input, "r") as f:
            results = json.load(f)
    except Exception as e:
        logger.error(f"Error loading results: {str(e)}")
        return

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate visualizations
    viz_files = visualize_results(results, output_dir, args.viz_types)

    logger.info(f"Generated {len(viz_files)} visualizations in {args.output}")
    for viz_type, path in viz_files.items():
        logger.info(f"- {viz_type}: {path}")


def generate_report(args):
    """
    Generate report based on command-line arguments.

    Args:
        args: Parsed command-line arguments
    """
    logger.info(f"Generating report from {args.input}")

    # Load results
    try:
        with open(args.input, "r") as f:
            results = json.load(f)
    except Exception as e:
        logger.error(f"Error loading results: {str(e)}")
        return

    # Create output directory if needed
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Create temporary model instance for report generation
    # We need to create a mock repository since we don't have the actual one
    class MockRepository(Repository):
        def __init__(self, name):
            super().__init__(name)

        def get_name(self):
            return results.get("metadata", {}).get("repository_name", "unknown")

        def get_all_files(self):
            return []

        def file_exists(self, file_path):
            return False

        def get_file_content(self, file_path):
            return None

        def get_file_size(self, file_path):
            return 0

        def get_file_creation_date(self, file_path):
            return None

        def get_commit_history(self, days=None, author=None, file_path=None):
            return []

        def list_directory(self, directory_path):
            return []

        def get_file_attributes(self, file_path):
            return {}

    # Create model with mock repository
    mock_repo = MockRepository(
        results.get("metadata", {}).get("repository_name", "unknown")
    )
    model = REPDModel(mock_repo)

    # Set results directly
    model.results = results

    # Generate report
    model.generate_report(output_path, template=args.template)

    logger.info(f"Generated report at {args.output}")


def main():
    """
    Main entry point for the command-line interface.
    """
    args = parse_args()

    try:
        if args.command == "analyze":
            run_analysis(args)
        elif args.command == "visualize":
            visualize_results(args)
        elif args.command == "report":
            generate_report(args)
        else:
            logger.error(f"Unknown command: {args.command}")
            sys.exit(1)
    except Exception as e:
        logger.error(f"Error executing command: {str(e)}")
        logger.debug("Exception details:", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
