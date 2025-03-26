#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Command-line interface for the REPD model.

This module provides a command-line interface for running REPD analysis,
generating visualizations, comparing with other models, and more.

Author: anirudhsengar
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any

from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn

from repd.model import REPDModel
from repd.repository import Repository
from repd.visualization import visualize_results

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)]
)
logger = logging.getLogger("repd")
console = Console()


def setup_analyze_parser(subparsers) -> None:
    """Set up the parser for the analyze command."""
    parser = subparsers.add_parser(
        "analyze",
        help="Run REPD analysis on a Git repository",
        description="Analyze a Git repository using the REPD model to identify risky files",
    )

    # Required arguments
    parser.add_argument(
        "--repo", "-r",
        help="Path to the Git repository",
        required=True
    )

    # Optional arguments
    parser.add_argument(
        "--lookback", "-l",
        help="Number of commits to analyze (default: %(default)s)",
        type=int,
        default=1000
    )
    parser.add_argument(
        "--output", "-o",
        help="Path to save the analysis results (default: %(default)s)",
        default="repd_results.json"
    )
    parser.add_argument(
        "--entry-weight", "-e",
        help="Weight for entry point importance (default: %(default)s)",
        type=float,
        default=0.5
    )
    parser.add_argument(
        "--coupling-threshold", "-c",
        help="Threshold for change coupling significance (default: %(default)s)",
        type=float,
        default=0.3
    )
    parser.add_argument(
        "--verbose", "-v",
        help="Enable verbose output",
        action="store_true"
    )

    parser.set_defaults(func=cmd_analyze)


def setup_visualize_parser(subparsers) -> None:
    """Set up the parser for the visualize command."""
    parser = subparsers.add_parser(
        "visualize",
        help="Generate visualizations from analysis results",
        description="Create visualizations based on REPD analysis results",
    )

    # Required arguments
    parser.add_argument(
        "--results", "-r",
        help="Path to REPD analysis results file",
        required=True
    )

    # Optional arguments
    parser.add_argument(
        "--output", "-o",
        help="Directory to save visualizations (default: %(default)s)",
        default="visualizations"
    )
    parser.add_argument(
        "--format", "-f",
        help="Output format for visualizations (default: %(default)s)",
        choices=["png", "pdf", "svg"],
        default="png"
    )
    parser.add_argument(
        "--types", "-t",
        help="Types of visualizations to generate (comma-separated)",
        default="risk,coupling,entry_points,network"
    )

    parser.set_defaults(func=cmd_visualize)


def setup_compare_parser(subparsers) -> None:
    """Set up the parser for the compare command."""
    parser = subparsers.add_parser(
        "compare",
        help="Compare REPD with other models",
        description="Compare REPD model results with other defect prediction models",
    )

    # Required arguments
    parser.add_argument(
        "--repd", "-r",
        help="Path to REPD analysis results file",
        required=True
    )

    # Optional arguments
    parser.add_argument(
        "--fixcache", "-f",
        help="Path to FixCache results file",
        default=None
    )
    parser.add_argument(
        "--output", "-o",
        help="Path to save comparison results (default: %(default)s)",
        default="comparison_results.json"
    )
    parser.add_argument(
        "--visualize", "-v",
        help="Generate visualization of comparison",
        action="store_true"
    )

    parser.set_defaults(func=cmd_compare)


def setup_optimize_parser(subparsers) -> None:
    """Set up the parser for the optimize command."""
    parser = subparsers.add_parser(
        "optimize",
        help="Optimize REPD model parameters",
        description="Find optimal parameters for the REPD model on a given repository",
    )

    # Required arguments
    parser.add_argument(
        "--repo", "-r",
        help="Path to the Git repository",
        required=True
    )

    # Optional arguments
    parser.add_argument(
        "--validation-data", "-v",
        help="Path to validation data (known defects)",
        required=False
    )
    parser.add_argument(
        "--output", "-o",
        help="Path to save optimized parameters (default: %(default)s)",
        default="repd_params.json"
    )
    parser.add_argument(
        "--iterations", "-i",
        help="Number of optimization iterations (default: %(default)s)",
        type=int,
        default=100
    )

    parser.set_defaults(func=cmd_optimize)


def cmd_analyze(args) -> int:
    """Analyze a repository using the REPD model."""
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    try:
        # Validate repository path
        repo_path = Path(args.repo)
        if not repo_path.exists():
            logger.error(f"Repository path does not exist: {args.repo}")
            return 1

        # Set up the progress indicator
        with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                transient=True,
        ) as progress:

            # Initialize repository
            progress.add_task(description="Initializing repository...", total=None)
            repository = Repository(str(repo_path))

            # Initialize model
            progress.add_task(description="Setting up REPD model...", total=None)
            model = REPDModel(
                repository=repository,
                lookback_commits=args.lookback,
                entry_point_weight=args.entry_weight,
                coupling_threshold=args.coupling_threshold
            )

            # Run analysis
            progress.add_task(description="Running analysis...", total=None)
            results = model.analyze()

            # Generate output
            progress.add_task(description="Generating results...", total=None)
            output_path = Path(args.output)

            # Ensure directory exists
            output_dir = output_path.parent
            if not output_dir.exists():
                output_dir.mkdir(parents=True)

            # Add metadata
            results["metadata"] = {
                "timestamp": datetime.now().isoformat(),
                "repository": str(repo_path),
                "lookback_commits": args.lookback,
                "entry_point_weight": args.entry_weight,
                "coupling_threshold": args.coupling_threshold,
                "repd_version": model.version
            }

            # Save results
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2)

        # Print summary
        console.print(f"\n[bold green]Analysis complete![/bold green]")
        console.print(f"Results saved to: [bold]{args.output}[/bold]")
        console.print(f"\nTop 5 riskiest files:")

        for i, (file, score) in enumerate(sorted(results["risk_scores"].items(),
                                                 key=lambda x: x[1], reverse=True)[:5], 1):
            console.print(f"{i}. {file} ([bold]{score:.4f}[/bold])")

        console.print(f"\nTo visualize these results, run:")
        console.print(f"[bold]repd visualize --results {args.output}[/bold]")

        return 0

    except KeyboardInterrupt:
        console.print("\n[yellow]Analysis cancelled by user.[/yellow]")
        return 130
    except Exception as e:
        logger.exception(f"Error during analysis: {str(e)}")
        return 1


def cmd_visualize(args) -> int:
    """Generate visualizations from REPD analysis results."""
    try:
        # Load results
        with open(args.results, "r") as f:
            results = json.load(f)

        # Create output directory
        output_dir = Path(args.output)
        if not output_dir.exists():
            output_dir.mkdir(parents=True)

        # Parse visualization types
        viz_types = [t.strip() for t in args.types.split(",")]

        # Generate visualizations
        console.print(f"Generating {args.format} visualizations in [bold]{args.output}[/bold]...")

        with Progress() as progress:
            task = progress.add_task("Creating visualizations...", total=len(viz_types))

            # Call visualization module
            visualize_results(
                results=results,
                output_dir=output_dir,
                viz_types=viz_types,
                file_format=args.format,
                progress_callback=lambda: progress.update(task, advance=1)
            )

        console.print("[bold green]Visualizations complete![/bold green]")
        return 0

    except FileNotFoundError:
        logger.error(f"Results file not found: {args.results}")
        return 1
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON in results file: {args.results}")
        return 1
    except Exception as e:
        logger.exception(f"Error generating visualizations: {str(e)}")
        return 1


def cmd_compare(args) -> int:
    """Compare REPD results with other models."""
    try:
        # Load REPD results
        with open(args.repd, "r") as f:
            repd_results = json.load(f)

        comparison_results = {
            "repd": repd_results["metadata"],
            "comparisons": {}
        }

        if args.fixcache:
            console.print(f"Comparing with FixCache model...")
            try:
                with open(args.fixcache, "r") as f:
                    fixcache_results = json.load(f)

                # Compare top risky files
                repd_top_files = sorted(repd_results["risk_scores"].items(),
                                        key=lambda x: x[1], reverse=True)[:10]
                fixcache_top_files = sorted(fixcache_results.get("risky_files", {}).items(),
                                            key=lambda x: x[1], reverse=True)[:10]

                overlap = set([x[0] for x in repd_top_files]) & set([x[0] for x in fixcache_top_files])

                comparison_results["comparisons"]["fixcache"] = {
                    "top_10_overlap": len(overlap),
                    "top_10_overlap_percentage": len(overlap) * 10,
                    "repd_unique": list(set([x[0] for x in repd_top_files]) - set([x[0] for x in fixcache_top_files])),
                    "fixcache_unique": list(
                        set([x[0] for x in fixcache_top_files]) - set([x[0] for x in repd_top_files]))
                }

                console.print(
                    f"Overlap in top 10 risky files: [bold]{len(overlap)} files[/bold] ({len(overlap) * 10}%)")

            except Exception as e:
                logger.error(f"Error comparing with FixCache: {str(e)}")
                console.print(f"[red]Error comparing with FixCache: {str(e)}[/red]")

        # Save comparison results
        with open(args.output, "w") as f:
            json.dump(comparison_results, f, indent=2)

        console.print(f"[bold green]Comparison complete![/bold green]")
        console.print(f"Results saved to: [bold]{args.output}[/bold]")

        return 0

    except FileNotFoundError as e:
        logger.error(f"File not found: {str(e)}")
        return 1
    except Exception as e:
        logger.exception(f"Error during comparison: {str(e)}")
        return 1


def cmd_optimize(args) -> int:
    """Optimize REPD model parameters."""
    try:
        console.print("Parameter optimization not yet implemented")
        console.print("This feature will be available in a future version")
        return 0
    except Exception as e:
        logger.exception(f"Error during optimization: {str(e)}")
        return 1


def main(argv: Optional[List[str]] = None) -> int:
    """Main entry point for the CLI."""
    # Parse arguments
    parser = argparse.ArgumentParser(
        prog="repd",
        description="Repository Entry Points Defects (REPD) model tool"
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"repd {REPDModel.version}"
    )
    parser.add_argument(
        "--debug",
        help="Enable debug output",
        action="store_true"
    )

    # Create subparsers for commands
    subparsers = parser.add_subparsers(
        dest="command",
        help="Command to run"
    )
    subparsers.required = True

    # Set up command parsers
    setup_analyze_parser(subparsers)
    setup_visualize_parser(subparsers)
    setup_compare_parser(subparsers)
    setup_optimize_parser(subparsers)

    # Parse arguments
    args = parser.parse_args(argv)

    # Set debug mode if requested
    if args.debug:
        logger.setLevel(logging.DEBUG)

    # Execute the appropriate command
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())