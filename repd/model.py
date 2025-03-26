#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
REPD Model Core Implementation

This module contains the main implementation of the Repository Entry Points
Defects (REPD) model for bug prediction in software repositories.

Author: anirudhsengar
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional, Set, Union

import networkx as nx
import numpy as np

from repd.entry_point_analyzer import EntryPointIdentifier
from repd.change_coupling import ChangeCouplingAnalyzer
from repd.developer_activity import DeveloperActivityTracker
from repd.repository import Repository
from repd.risk_calculator import DefectRiskCalculator

logger = logging.getLogger(__name__)


class REPDModel:
    """
    Repository Entry Points Defects (REPD) Model

    A bug prediction approach that analyzes repository entry points,
    change coupling patterns, and developer activity to predict
    defect-prone files in software repositories.

    Key features:
    - Identifies critical repository entry points
    - Analyzes change coupling between files
    - Studies developer expertise and contribution patterns
    - Maps structural and dependency relationships
    """

    version = "0.1.0"

    def __init__(
            self,
            repository: Union[Repository, str],
            lookback_commits: int = 1000,
            entry_point_weight: float = 0.5,
            coupling_threshold: float = 0.3,
            min_change_count: int = 5,
            dev_expertise_weight: float = 0.3,
            path_complexity_weight: float = 0.2,
    ):
        """
        Initialize the REPD model.

        Args:
            repository: Repository object or path to repository
            lookback_commits: Number of commits to analyze
            entry_point_weight: Weight for entry point importance
            coupling_threshold: Threshold for change coupling significance
            min_change_count: Minimum changes for file consideration
            dev_expertise_weight: Weight for developer expertise factor
            path_complexity_weight: Weight for path complexity factor
        """
        # Initialize repository
        if isinstance(repository, str):
            self.repository = Repository(repository)
        else:
            self.repository = repository

        # Store configuration parameters
        self.lookback_commits = lookback_commits
        self.entry_point_weight = entry_point_weight
        self.coupling_threshold = coupling_threshold
        self.min_change_count = min_change_count
        self.dev_expertise_weight = dev_expertise_weight
        self.path_complexity_weight = path_complexity_weight

        # Initialize component analyzers
        self.entry_point_analyzer = EntryPointIdentifier(self.repository)
        self.coupling_analyzer = ChangeCouplingAnalyzer(
            self.repository,
            self.coupling_threshold,
            self.min_change_count
        )
        self.dev_activity_tracker = DeveloperActivityTracker(self.repository)
        self.risk_calculator = DefectRiskCalculator()

        # Initialize results containers
        self.entry_points: Dict[str, float] = {}
        self.coupling_matrix: Dict[str, Dict[str, float]] = {}
        self.dev_activity_stats: Dict[str, Dict[str, Any]] = {}
        self.risk_scores: Dict[str, float] = {}
        self.dependency_graph = nx.DiGraph()

        # Analysis status
        self.analyzed = False
        self.analysis_timestamp = None

    def analyze(self) -> Dict[str, Any]:
        """
        Perform complete REPD analysis on the repository.

        Returns:
            Dict containing analysis results:
            - entry_points: Files identified as entry points with scores
            - coupling_matrix: Change coupling relationships between files
            - risk_scores: Calculated risk scores for each file
            - dependency_graph: Graph representation of file relationships
            - metadata: Analysis metadata
        """
        logger.info("Starting REPD analysis")
        start_time = datetime.now()

        # Step 1: Identify entry points
        logger.info("Identifying repository entry points")
        self.entry_points = self.entry_point_analyzer.identify_entry_points(
            weight_factor=self.entry_point_weight
        )

        # Step 2: Analyze change coupling
        logger.info("Analyzing change coupling patterns")
        self.coupling_matrix = self.coupling_analyzer.analyze_coupling(
            lookback=self.lookback_commits
        )

        # Step 3: Track developer activity
        logger.info("Tracking developer activity")
        self.dev_activity_stats = self.dev_activity_tracker.track_activity(
            lookback=self.lookback_commits
        )

        # Step 4: Build dependency graph
        logger.info("Building file dependency graph")
        self.dependency_graph = self._build_dependency_graph()

        # Step 5: Calculate risk scores
        logger.info("Calculating defect risk scores")
        self.risk_scores = self._calculate_risk_scores()

        # Mark as analyzed and record timestamp
        self.analyzed = True
        self.analysis_timestamp = datetime.now()

        analysis_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"Analysis completed in {analysis_time:.2f} seconds")

        # Return results dictionary
        return {
            "entry_points": self.entry_points,
            "coupling_matrix": self.coupling_matrix,
            "dev_activity": self.dev_activity_stats,
            "risk_scores": self.risk_scores,
            "metadata": {
                "timestamp": self.analysis_timestamp.isoformat(),
                "repository": self.repository.path,
                "lookback_commits": self.lookback_commits,
                "entry_point_weight": self.entry_point_weight,
                "coupling_threshold": self.coupling_threshold,
                "analyzed_files": len(self.risk_scores),
                "elapsed_time": analysis_time,
            }
        }

    def get_top_risky_files(self, limit: int = 10) -> List[Tuple[str, float]]:
        """
        Get the top risky files based on calculated scores.

        Args:
            limit: Maximum number of files to return

        Returns:
            List of (filename, risk_score) tuples for top risky files
        """
        if not self.analyzed:
            logger.warning("Repository not yet analyzed, run analyze() first")
            return []

        return sorted(
            self.risk_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:limit]

    def visualize(self, output_path: str = "visualizations") -> None:
        """
        Generate visualizations from analysis results.

        Args:
            output_path: Directory to save visualizations
        """
        if not self.analyzed:
            logger.warning("Repository not yet analyzed, run analyze() first")
            return

        # Import here to avoid circular imports
        from repd.visualization import visualize_results

        # Create results dictionary for visualization
        results = {
            "entry_points": self.entry_points,
            "coupling_matrix": self.coupling_matrix,
            "dev_activity": self.dev_activity_stats,
            "risk_scores": self.risk_scores,
            "metadata": {
                "timestamp": self.analysis_timestamp.isoformat(),
                "repository": self.repository.path,
            }
        }

        # Generate visualizations
        output_dir = Path(output_path)
        output_dir.mkdir(exist_ok=True, parents=True)

        visualize_results(
            results=results,
            output_dir=output_dir,
            viz_types=["risk", "coupling", "entry_points", "network"],
            file_format="png"
        )

    def analyze_file(self, filepath: str) -> Dict[str, Any]:
        """
        Analyze a specific file for defect risk.

        Args:
            filepath: Path to the file to analyze

        Returns:
            Dictionary with file analysis results
        """
        if not self.analyzed:
            logger.warning("Repository not yet analyzed, run analyze() first")
            return {}

        # Normalize path to match repository format
        filepath = self.repository.normalize_path(filepath)

        # Check if file exists in analyzed files
        if filepath not in self.risk_scores:
            logger.warning(f"File {filepath} not found in analysis results")
            return {}

        # Get related files through coupling
        related_files = {}
        if filepath in self.coupling_matrix:
            related_files = {
                filename: score
                for filename, score in self.coupling_matrix[filepath].items()
                if score > self.coupling_threshold
            }

        # Get entry point status
        entry_point_score = self.entry_points.get(filepath, 0)

        # Get developer activity
        dev_activity = self.dev_activity_stats.get(filepath, {})

        # Return detailed file analysis
        return {
            "file": filepath,
            "risk_score": self.risk_scores.get(filepath, 0),
            "is_entry_point": entry_point_score > 0,
            "entry_point_score": entry_point_score,
            "related_files": related_files,
            "developer_count": dev_activity.get("developer_count", 0),
            "change_frequency": dev_activity.get("change_frequency", 0),
            "last_modified": dev_activity.get("last_modified", ""),
            "riskiest_related_file": max(related_files.items(), key=lambda x: x[1])[0]
            if related_files else None,
        }

    def _build_dependency_graph(self) -> nx.DiGraph:
        """
        Build a dependency graph from repository structure and coupling data.

        Returns:
            NetworkX directed graph representing file dependencies
        """
        graph = nx.DiGraph()

        # Add nodes for all files
        all_files = set()
        for file in self.repository.get_all_files():
            all_files.add(file)
            graph.add_node(file, type="file")

        # Add edges for change coupling
        for source, targets in self.coupling_matrix.items():
            for target, weight in targets.items():
                if weight >= self.coupling_threshold:
                    graph.add_edge(source, target, weight=weight, type="coupling")

        # Add edges for import dependencies if available
        # This would require language-specific parsing, placeholder for now

        # Add entry point attributes
        for file, score in self.entry_points.items():
            if file in graph:
                graph.nodes[file]["entry_point"] = True
                graph.nodes[file]["entry_score"] = score

        # Add risk score attributes if calculated
        if self.risk_scores:
            for file, score in self.risk_scores.items():
                if file in graph:
                    graph.nodes[file]["risk_score"] = score

        return graph

    def _calculate_risk_scores(self) -> Dict[str, float]:
        """
        Calculate defect risk scores for all files using the REPD model.

        Returns:
            Dictionary mapping filenames to risk scores (0.0-1.0)
        """
        # Get all files from repository
        all_files = set(self.repository.get_all_files())

        # Add files from coupling matrix and entry points
        all_files.update(self.coupling_matrix.keys())
        all_files.update(self.entry_points.keys())

        # Initialize risk scores dictionary
        risk_scores = {}

        # Process each file
        for filename in all_files:
            # Skip directories and non-code files
            if not self.repository.is_code_file(filename):
                continue

            # Calculate entry point factor (0.0-1.0)
            entry_point_factor = self.entry_points.get(filename, 0.0)

            # Calculate coupling factor (0.0-1.0)
            coupling_factor = self._calculate_coupling_factor(filename)

            # Calculate developer expertise factor (0.0-1.0)
            dev_expertise_factor = self._calculate_dev_expertise_factor(filename)

            # Calculate path complexity factor (0.0-1.0)
            path_complexity_factor = self._calculate_path_complexity_factor(filename)

            # Combine factors with weights
            risk_score = self.risk_calculator.calculate_risk(
                entry_point_score=entry_point_factor,
                coupling_score=coupling_factor,
                dev_expertise_score=dev_expertise_factor,
                path_complexity_score=path_complexity_factor,
                entry_point_weight=self.entry_point_weight,
                dev_expertise_weight=self.dev_expertise_weight,
                path_complexity_weight=self.path_complexity_weight
            )

            risk_scores[filename] = risk_score

        return risk_scores

    def _calculate_coupling_factor(self, filename: str) -> float:
        """
        Calculate the coupling factor for a file.

        Args:
            filename: The file to calculate coupling factor for

        Returns:
            Coupling factor (0.0-1.0)
        """
        # Check if file has coupling data
        if filename not in self.coupling_matrix:
            return 0.0

        # Get coupling values for this file
        couplings = self.coupling_matrix[filename]

        # If no couplings, return 0
        if not couplings:
            return 0.0

        # Calculate the average coupling strength
        avg_coupling = sum(couplings.values()) / len(couplings)

        # Calculate connectivity degree (normalized by number of files)
        connectivity = min(1.0, len(couplings) / 20)  # Cap at 20 files

        # Combine average coupling and connectivity
        coupling_factor = 0.7 * avg_coupling + 0.3 * connectivity

        return coupling_factor

    def _calculate_dev_expertise_factor(self, filename: str) -> float:
        """
        Calculate the developer expertise factor for a file.

        Args:
            filename: The file to calculate developer expertise for

        Returns:
            Developer expertise factor (0.0-1.0)
        """
        # Check if file has developer activity data
        if filename not in self.dev_activity_stats:
            return 0.5  # Default to medium risk

        stats = self.dev_activity_stats[filename]

        # Calculate developer count factor (more developers = higher risk)
        dev_count = stats.get("developer_count", 1)
        dev_count_factor = min(1.0, dev_count / 5)  # Normalize, cap at 5 developers

        # Calculate change frequency factor
        change_freq = stats.get("change_frequency", 0)
        change_freq_factor = min(1.0, change_freq / 20)  # Normalize, cap at 20 changes

        # Calculate expertise factor (lower expertise = higher risk)
        expertise = stats.get("expertise_level", 0.5)
        expertise_factor = 1.0 - expertise  # Invert so higher is riskier

        # Calculate ownership factor (lower ownership = higher risk)
        ownership = stats.get("ownership", 1.0)
        ownership_factor = 1.0 - ownership  # Invert so higher is riskier

        # Combine factors
        dev_expertise_factor = (
                0.3 * dev_count_factor +
                0.3 * change_freq_factor +
                0.2 * expertise_factor +
                0.2 * ownership_factor
        )

        return dev_expertise_factor

    def _calculate_path_complexity_factor(self, filename: str) -> float:
        """
        Calculate the path complexity factor for a file.

        Args:
            filename: The file to calculate path complexity for

        Returns:
            Path complexity factor (0.0-1.0)
        """
        # Simple path complexity based on file path depth
        path_depth = len(Path(filename).parts)

        # Normalize depth (deeper paths are riskier)
        # Cap at depth 10 to avoid extreme values
        normalized_depth = min(path_depth / 10, 1.0)

        # Check if file is in a test directory (lower risk)
        is_test = "test" in filename.lower() or "spec" in filename.lower()
        test_factor = 0.7 if is_test else 1.0

        # Calculate path complexity factor
        path_complexity_factor = normalized_depth * test_factor

        return path_complexity_factor


if __name__ == "__main__":
    # Example usage
    import sys

    # Configure logging
    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) < 2:
        print("Usage: python model.py /path/to/repository")
        sys.exit(1)

    # Initialize model
    repo_path = sys.argv[1]
    model = REPDModel(repo_path)

    # Run analysis
    results = model.analyze()

    # Print top risky files
    print("\nTop 10 risky files:")
    for file, score in model.get_top_risky_files(10):
        print(f"{file}: {score:.4f}")