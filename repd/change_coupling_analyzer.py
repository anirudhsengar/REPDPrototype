#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Change Coupling Analysis Module for REPD Model

This module analyzes commit history to identify files that frequently change
together, which can indicate hidden dependencies or architectural issues.

Author: anirudhsengar
"""

import json
import logging
import os
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import networkx as nx
import numpy as np
from sklearn.cluster import DBSCAN

from repd.repository import Commit, Repository

logger = logging.getLogger(__name__)


class ChangeCouplingAnalyzer:
    """
    Analyzes change coupling between files in a repository.

    Change coupling refers to files that frequently change together in commits,
    which may indicate hidden dependencies or architectural issues.
    """

    def __init__(self, repository: Repository):
        """
        Initialize the change coupling analyzer.

        Args:
            repository: Repository interface to analyze
        """
        self.repository = repository
        self.coupling_matrix = None  # Will be populated during analysis
        self.coupling_graph = None  # NetworkX graph representation of coupling
        self.file_changes = None  # Count of changes per file
        self.commit_count = 0  # Total number of commits analyzed

    def analyze_coupling(
        self,
        days: int = None,
        normalize: bool = True,
        temporal_decay: float = None,
        file_extensions: List[str] = None,
    ) -> Dict[str, Dict[str, float]]:
        """
        Analyze change coupling in the repository.

        Args:
            days: Number of days to include in the analysis (None for all history)
            normalize: Whether to normalize the coupling scores (0-1)
            temporal_decay: Decay factor for older commits (None to disable)
            file_extensions: List of file extensions to include (None for all)

        Returns:
            Dictionary mapping file pairs to coupling scores
        """
        logger.info("Analyzing change coupling")

        # Get commit history
        commits = self.repository.get_commit_history(days=days)
        self.commit_count = len(commits)

        logger.debug(f"Analyzing {self.commit_count} commits")

        if self.commit_count == 0:
            logger.warning("No commits found, coupling analysis skipped")
            self.coupling_matrix = {}
            self.file_changes = {}
            return {}

        # Initialize data structures
        file_changes = defaultdict(int)  # Count changes per file
        file_co_changes = defaultdict(lambda: defaultdict(int))  # Count co-changes

        # Weight for temporal decay if enabled
        current_date = datetime.now()

        # Process each commit
        for idx, commit in enumerate(commits):
            # Get modified files
            modified_files = commit.modified_files

            # Filter by extension if specified
            if file_extensions:
                modified_files = [
                    f
                    for f in modified_files
                    if any(f.endswith(ext) for ext in file_extensions)
                ]

            # Skip commits with too many files (likely large refactorings)
            if len(modified_files) > 20:
                logger.debug(
                    f"Skipping commit {commit.hash} with {len(modified_files)} files"
                )
                continue

            # Skip commits with only one file (no coupling)
            if len(modified_files) <= 1:
                # Still count the file change
                for file in modified_files:
                    file_changes[file] += 1
                continue

            # Calculate commit weight if using temporal decay
            weight = 1.0
            if temporal_decay is not None and hasattr(commit, "date"):
                # Calculate days since commit
                if isinstance(commit.date, datetime):
                    days_since = (current_date - commit.date).days
                    weight = temporal_decay**days_since

            # Update file change counts
            for file in modified_files:
                file_changes[file] += weight

            # Update co-change counts for all file pairs in this commit
            for i, file1 in enumerate(modified_files):
                for file2 in modified_files[i + 1 :]:
                    if file1 == file2:
                        continue
                    file_co_changes[file1][file2] += weight
                    file_co_changes[file2][file1] += weight

        # Calculate coupling scores
        coupling_matrix = {}

        for file1, co_changes in file_co_changes.items():
            coupling_matrix[file1] = {}

            for file2, count in co_changes.items():
                # Calculate coupling score using Jaccard similarity:
                # count of commits with both files / min of the individual counts
                if file_changes[file1] > 0 and file_changes[file2] > 0:
                    if normalize:
                        # Normalized score (0-1)
                        coupling = count / min(file_changes[file1], file_changes[file2])
                    else:
                        # Raw co-change count
                        coupling = count

                    coupling_matrix[file1][file2] = coupling

        self.coupling_matrix = coupling_matrix
        self.file_changes = dict(file_changes)

        # Create a graph representation
        self._create_coupling_graph()

        logger.info(
            f"Analyzed change coupling for {len(file_changes)} files, "
            f"found {sum(len(v) for v in coupling_matrix.values())} coupling relationships"
        )

        return coupling_matrix

    def get_coupling_score(self, file1: str, file2: str) -> float:
        """
        Get the coupling score between two files.

        Args:
            file1: First file path
            file2: Second file path

        Returns:
            Coupling score (0-1) or 0 if no coupling
        """
        if not self.coupling_matrix:
            return 0.0

        if file1 not in self.coupling_matrix or file2 not in self.coupling_matrix:
            return 0.0

        return self.coupling_matrix[file1].get(file2, 0.0)

    def get_coupled_files(
        self, file_path: str, min_score: float = 0.3
    ) -> Dict[str, float]:
        """
        Get files coupled with a specific file.

        Args:
            file_path: Path to the file
            min_score: Minimum coupling score to include

        Returns:
            Dictionary mapping file paths to coupling scores
        """
        if not self.coupling_matrix or file_path not in self.coupling_matrix:
            return {}

        return {
            f: score
            for f, score in self.coupling_matrix[file_path].items()
            if score >= min_score
        }

    def get_coupled_clusters(
        self, min_coupling: float = 0.3, min_cluster_size: int = 2
    ) -> List[Set[str]]:
        """
        Identify clusters of coupled files.

        Args:
            min_coupling: Minimum coupling score to consider
            min_cluster_size: Minimum size of a cluster

        Returns:
            List of file clusters (each a set of file paths)
        """
        if not self.coupling_graph:
            return []

        # Create a copy of the graph with only edges above threshold
        g = nx.Graph()
        for u, v, data in self.coupling_graph.edges(data=True):
            if data["weight"] >= min_coupling:
                g.add_edge(u, v, weight=data["weight"])

        # Extract connected components - these are the clusters
        clusters = list(nx.connected_components(g))

        # Filter by size
        clusters = [c for c in clusters if len(c) >= min_cluster_size]

        # Sort by size (largest first)
        clusters.sort(key=len, reverse=True)

        return clusters

    def get_coupling_matrix(self) -> Dict[str, Dict[str, float]]:
        """
        Get the full coupling matrix.

        Returns:
            Dictionary mapping file pairs to coupling scores
        """
        return self.coupling_matrix or {}

    def get_high_coupling_pairs(
        self, threshold: float = 0.7
    ) -> List[Tuple[str, str, float]]:
        """
        Get file pairs with coupling above a threshold.

        Args:
            threshold: Minimum coupling score

        Returns:
            List of (file1, file2, score) tuples
        """
        if not self.coupling_matrix:
            return []

        high_coupling = []
        seen_pairs = set()

        for file1, couplings in self.coupling_matrix.items():
            for file2, score in couplings.items():
                if score < threshold:
                    continue

                # Avoid duplicate pairs
                pair = tuple(sorted([file1, file2]))
                if pair in seen_pairs:
                    continue

                seen_pairs.add(pair)
                high_coupling.append((file1, file2, score))

        # Sort by coupling score (highest first)
        high_coupling.sort(key=lambda x: x[2], reverse=True)

        return high_coupling

    def export_coupling_data(self, output_file: str) -> None:
        """
        Export coupling data to a JSON file.

        Args:
            output_file: Path to the output file
        """
        if not self.coupling_matrix:
            logger.warning("No coupling data to export")
            return

        # Prepare data for export
        data = {
            "coupling_matrix": self.coupling_matrix,
            "file_changes": self.file_changes,
            "metadata": {
                "commit_count": self.commit_count,
                "file_count": len(self.file_changes),
                "coupling_count": sum(len(v) for v in self.coupling_matrix.values()),
                "top_coupled_pairs": [
                    {"file1": f1, "file2": f2, "score": score}
                    for f1, f2, score in self.get_high_coupling_pairs()[:10]
                ],
                "coupled_clusters": [
                    list(cluster) for cluster in self.get_coupled_clusters()[:5]
                ],
            },
        }

        # Write to file
        with open(output_file, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Exported coupling data to {output_file}")

    def _create_coupling_graph(self) -> None:
        """
        Create a graph representation of the coupling matrix.
        """
        if not self.coupling_matrix:
            return

        # Create undirected graph
        g = nx.Graph()

        # Add nodes
        g.add_nodes_from(self.coupling_matrix.keys())

        # Add edges
        for file1, couplings in self.coupling_matrix.items():
            for file2, score in couplings.items():
                g.add_edge(file1, file2, weight=score)

        self.coupling_graph = g

    def calculate_coupling_density(self) -> float:
        """
        Calculate the overall coupling density.

        Returns:
            Coupling density (0-1)
        """
        if not self.coupling_graph or self.coupling_graph.number_of_nodes() <= 1:
            return 0.0

        # Calculate density using NetworkX
        return nx.density(self.coupling_graph)

    def calculate_coupling_centrality(self) -> Dict[str, float]:
        """
        Calculate centrality measures for the coupling graph.

        Returns:
            Dictionary mapping file paths to centrality scores
        """
        if not self.coupling_graph or self.coupling_graph.number_of_nodes() <= 1:
            return {}

        try:
            # Calculate eigenvector centrality (importance based on connections)
            centrality = nx.eigenvector_centrality_numpy(
                self.coupling_graph, weight="weight"
            )
        except:
            try:
                # Fall back to degree centrality if eigenvector fails
                centrality = nx.degree_centrality(self.coupling_graph)
            except:
                # If all else fails, use a simple count of connections
                centrality = {
                    node: len(list(self.coupling_graph.neighbors(node)))
                    / (self.coupling_graph.number_of_nodes() - 1)
                    for node in self.coupling_graph.nodes()
                }

        return centrality
