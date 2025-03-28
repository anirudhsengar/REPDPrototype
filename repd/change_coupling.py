#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Change Coupling Analyzer for REPD Model

This module analyzes change coupling patterns in a repository, identifying
files that tend to change together frequently, which can indicate
dependencies and potential defect propagation paths.

Author: anirudhsengar
"""

import logging
import time
from collections import defaultdict
from datetime import datetime
from typing import Any, DefaultDict, Dict, List, Optional, Set, Tuple

import numpy as np
from tqdm import tqdm

from repd.repository import Repository

logger = logging.getLogger(__name__)


class ChangeCouplingAnalyzer:
    """
    Analyzes change coupling patterns in a repository.

    Change coupling refers to files that tend to change together in commits,
    indicating logical or structural dependencies between them. Files with
    high change coupling are more likely to propagate defects between them.
    """

    def __init__(
        self,
        repository: Repository,
        coupling_threshold: float = 0.3,
        min_change_count: int = 5,
        max_files_per_commit: int = 100,
    ):
        """
        Initialize the change coupling analyzer.

        Args:
            repository: Repository object to analyze
            coupling_threshold: Minimum coupling score to consider significant
            min_change_count: Minimum number of changes required for a file to be considered
            max_files_per_commit: Ignore commits with more than this many files
        """
        self.repository = repository
        self.coupling_threshold = coupling_threshold
        self.min_change_count = min_change_count
        self.max_files_per_commit = max_files_per_commit

        # Internal data structures
        self.file_changes: DefaultDict[str, int] = defaultdict(int)
        self.co_changes: DefaultDict[str, DefaultDict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        self.coupling_matrix: Dict[str, Dict[str, float]] = {}

    def analyze_coupling(
        self, lookback: int = 1000, branch: str = "HEAD"
    ) -> Dict[str, Dict[str, float]]:
        """
        Analyze change coupling patterns in the repository.

        Args:
            lookback: Number of commits to analyze
            branch: Git branch to analyze

        Returns:
            Dictionary mapping filenames to dictionaries of coupled files and scores
        """
        start_time = time.time()
        logger.info(
            f"Analyzing change coupling patterns (lookback: {lookback} commits)"
        )

        # Analyze commits to build co-change matrix
        commits = self._extract_commits_for_analysis(lookback, branch)
        self._build_co_change_matrix(commits)

        # Filter out files with too few changes
        self._filter_low_frequency_files()

        # Compute coupling scores from co-change data
        self.coupling_matrix = self._compute_coupling_scores()

        # Filter by threshold
        self._filter_by_threshold()

        elapsed = time.time() - start_time
        logger.info(f"Change coupling analysis completed in {elapsed:.2f} seconds")

        # Total number of significant couplings found
        total_couplings = sum(
            len(couplings) for couplings in self.coupling_matrix.values()
        )
        logger.info(
            f"Identified {total_couplings} significant change couplings "
            f"among {len(self.coupling_matrix)} files"
        )

        return self.coupling_matrix

    def get_top_coupled_files(
        self, filename: str, limit: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Get the top files coupled with the given file.

        Args:
            filename: Target file to find coupled files for
            limit: Maximum number of results to return

        Returns:
            List of (filename, coupling_score) tuples for top coupled files
        """
        if filename not in self.coupling_matrix:
            return []

        couplings = self.coupling_matrix[filename]
        return sorted(couplings.items(), key=lambda x: x[1], reverse=True)[:limit]

    def get_coupling_hotspots(self, limit: int = 10) -> List[Tuple[str, int]]:
        """
        Get files with the most coupling relationships.

        Args:
            limit: Maximum number of results to return

        Returns:
            List of (filename, num_couplings) tuples for most coupled files
        """
        file_coupling_counts = [
            (file, len(couplings)) for file, couplings in self.coupling_matrix.items()
        ]
        return sorted(file_coupling_counts, key=lambda x: x[1], reverse=True)[:limit]

    def _extract_commits_for_analysis(
        self, lookback: int, branch: str
    ) -> List[Dict[str, Any]]:
        """
        Extract relevant commits for analysis.

        Args:
            lookback: Number of commits to extract
            branch: Git branch to analyze

        Returns:
            List of commit data dictionaries
        """
        logger.debug(f"Extracting up to {lookback} commits from branch '{branch}'")

        # Extract commits from repository
        commits = self.repository.get_commit_history(lookback, branch)

        # Filter out commits without useful information
        filtered_commits = []
        for commit in commits:
            # Skip commits with too many files (likely automated changes)
            if len(commit["files"]) > self.max_files_per_commit:
                continue

            # Skip commits with no files
            if not commit["files"]:
                continue

            # Skip merge commits
            if commit.get("is_merge", False):
                continue

            filtered_commits.append(commit)

        logger.debug(
            f"Using {len(filtered_commits)} commits for analysis after filtering"
        )
        return filtered_commits

    def _build_co_change_matrix(self, commits: List[Dict[str, Any]]) -> None:
        """
        Build the co-change matrix from commit data.

        Args:
            commits: List of commit data dictionaries
        """
        logger.debug("Building co-change matrix")

        # Process each commit
        for commit in tqdm(commits, desc="Analyzing commits", unit="commit"):
            files = commit["files"]

            # Skip commits with only one file
            if len(files) < 2:
                continue

            # Update file change counts
            for file in files:
                self.file_changes[file] += 1

            # Update co-change counts for each pair of files
            for i, file1 in enumerate(files):
                for file2 in files[i + 1 :]:
                    if file1 != file2:
                        self.co_changes[file1][file2] += 1
                        self.co_changes[file2][file1] += 1

        logger.debug(
            f"Processed {len(self.file_changes)} unique files in co-change matrix"
        )

    def _filter_low_frequency_files(self) -> None:
        """
        Filter out files with too few changes from the co-change matrix.
        """
        # Find files with enough changes
        valid_files = {
            file
            for file, count in self.file_changes.items()
            if count >= self.min_change_count
        }

        # Filter out low frequency files from co-changes
        filtered_co_changes = defaultdict(lambda: defaultdict(int))

        for file1, couplings in self.co_changes.items():
            if file1 not in valid_files:
                continue

            for file2, count in couplings.items():
                if file2 in valid_files:
                    filtered_co_changes[file1][file2] = count

        self.co_changes = filtered_co_changes

        # Log the filtering results
        original_count = len(self.file_changes)
        filtered_count = len(valid_files)
        logger.debug(
            f"Filtered out {original_count - filtered_count} files with < "
            f"{self.min_change_count} changes, keeping {filtered_count} files"
        )

    def _compute_coupling_scores(self) -> Dict[str, Dict[str, float]]:
        """
        Compute coupling scores based on co-change data.

        Returns:
            Dictionary mapping filenames to dictionaries of coupled files and scores
        """
        coupling_scores = {}

        for file1, couplings in self.co_changes.items():
            file1_changes = self.file_changes[file1]
            file_scores = {}

            for file2, co_change_count in couplings.items():
                file2_changes = self.file_changes[file2]

                # Calculate coupling score using Support / (Support_A + Support_B - Support)
                # This is similar to the Jaccard coefficient but accounts for relative frequency
                denominator = file1_changes + file2_changes - co_change_count
                if denominator > 0:
                    score = co_change_count / denominator
                    file_scores[file2] = score

            if file_scores:
                coupling_scores[file1] = file_scores

        return coupling_scores

    def _filter_by_threshold(self) -> None:
        """
        Filter coupling matrix to only include scores above the threshold.
        """
        filtered_matrix = {}

        for file, couplings in self.coupling_matrix.items():
            filtered_couplings = {
                coupled_file: score
                for coupled_file, score in couplings.items()
                if score >= self.coupling_threshold
            }

            if filtered_couplings:
                filtered_matrix[file] = filtered_couplings

        self.coupling_matrix = filtered_matrix

        # Count the number of couplings
        total_couplings = sum(
            len(couplings) for couplings in self.coupling_matrix.values()
        )
        logger.debug(
            f"After filtering by threshold {self.coupling_threshold}, "
            f"kept {total_couplings} couplings"
        )

    def visualize_coupling_network(
        self, output_file: str = "coupling_network.png", top_n_files: int = 30
    ) -> None:
        """
        Visualize the change coupling network.

        Args:
            output_file: Path to save the visualization
            top_n_files: Number of most coupled files to include
        """
        try:
            import matplotlib.pyplot as plt
            import networkx as nx

            # Create graph
            G = nx.Graph()

            # Get top files by total coupling score
            file_total_scores = {}
            for file, couplings in self.coupling_matrix.items():
                file_total_scores[file] = sum(couplings.values())

            top_files = sorted(
                file_total_scores.items(), key=lambda x: x[1], reverse=True
            )[:top_n_files]
            top_file_names = {file for file, _ in top_files}

            # Add nodes and edges for top files
            for file in top_file_names:
                G.add_node(file)

                if file in self.coupling_matrix:
                    for coupled_file, score in self.coupling_matrix[file].items():
                        if coupled_file in top_file_names:
                            G.add_edge(file, coupled_file, weight=score)

            # Create visualization
            plt.figure(figsize=(12, 10))

            # Calculate node sizing based on coupling degree
            sizes = [100 + 200 * G.degree(node, weight="weight") for node in G.nodes()]

            # Calculate edge thickness based on weight
            edge_weights = [G[u][v]["weight"] * 2 for u, v in G.edges()]

            # Create layout
            pos = nx.spring_layout(G, k=0.2, iterations=50)

            # Draw the graph
            nx.draw_networkx(
                G,
                pos=pos,
                node_size=sizes,
                node_color="skyblue",
                font_size=8,
                width=edge_weights,
                edge_color="gray",
                alpha=0.8,
                with_labels=True,
                font_weight="bold",
            )

            plt.title("Change Coupling Network", fontsize=16)
            plt.axis("off")

            # Save figure
            plt.tight_layout()
            plt.savefig(output_file, dpi=300, bbox_inches="tight")
            plt.close()

            logger.info(f"Coupling network visualization saved to {output_file}")

        except ImportError:
            logger.warning(
                "Could not create visualization. Required packages: matplotlib, networkx"
            )


if __name__ == "__main__":
    # Example usage
    import sys

    # Configure logging
    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) < 2:
        print("Usage: python change_coupling.py /path/to/repository")
        sys.exit(1)

    # Initialize repository and analyzer
    repo_path = sys.argv[1]
    repository = Repository(repo_path)
    analyzer = ChangeCouplingAnalyzer(repository)

    # Run analysis
    coupling_matrix = analyzer.analyze_coupling()

    # Print coupling hotspots
    print("\nTop coupling hotspots (files with most coupling relationships):")
    for file, count in analyzer.get_coupling_hotspots(10):
        print(f"{file}: {count} coupling relationships")

    # Print a sample coupling relationship for the first hotspot
    if analyzer.get_coupling_hotspots():
        hotspot = analyzer.get_coupling_hotspots()[0][0]
        print(f"\nTop files coupled with {hotspot}:")
        for file, score in analyzer.get_top_coupled_files(hotspot):
            print(f"  - {file}: {score:.4f}")
