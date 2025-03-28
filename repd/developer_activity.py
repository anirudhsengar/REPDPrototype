#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Developer Activity Tracker for REPD Model

This module tracks and analyzes developer activity patterns in a repository,
including expertise levels, file ownership, change frequency, and developer
interaction patterns.

Author: anirudhsengar
"""

import logging
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from typing import Any, DefaultDict, Dict, List, Optional, Set, Tuple

import numpy as np
from tqdm import tqdm

from repd.repository import Repository

logger = logging.getLogger(__name__)


class DeveloperActivityTracker:
    """
    Tracks and analyzes developer activity patterns in a repository.

    This class examines developer interactions with the codebase, including:
    - Developer expertise levels based on contribution history
    - File ownership (which developer has made the most changes)
    - Change frequency statistics
    - Developer interaction patterns (e.g., how many developers touch a file)
    """

    def __init__(self, repository: Repository):
        """
        Initialize the developer activity tracker.

        Args:
            repository: Repository object to analyze
        """
        self.repository = repository

        # Internal data structures
        self.file_changes: DefaultDict[str, int] = defaultdict(int)
        self.file_owners: Dict[str, str] = {}
        self.file_last_modified: Dict[str, datetime] = {}
        self.file_developers: DefaultDict[str, Set[str]] = defaultdict(set)
        self.developer_changes: DefaultDict[str, DefaultDict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        self.developer_total_changes: DefaultDict[str, int] = defaultdict(int)
        self.developer_expertise: Dict[str, float] = {}

        # Activity statistics
        self.activity_stats: Dict[str, Dict[str, Any]] = {}

    def track_activity(self, lookback: int = 1000) -> Dict[str, Dict[str, Any]]:
        """
        Track developer activity patterns in the repository.

        Args:
            lookback: Number of commits to analyze

        Returns:
            Dictionary mapping filenames to developer activity statistics
        """
        logger.info(
            f"Tracking developer activity patterns (lookback: {lookback} commits)"
        )

        # Extract commits for analysis
        commits = self.repository.get_commit_history(lookback)

        # Process commits to track developer activity
        self._process_commits(commits)

        # Calculate developer expertise levels
        self._calculate_developer_expertise()

        # Calculate file ownership
        self._calculate_file_ownership()

        # Compile activity statistics for each file
        self._compile_activity_stats()

        # Log summary statistics
        num_developers = len(self.developer_total_changes)
        num_files = len(self.file_changes)
        avg_developers_per_file = sum(
            len(devs) for devs in self.file_developers.values()
        ) / max(1, num_files)

        logger.info(
            f"Analyzed activity of {num_developers} developers across {num_files} files"
        )
        logger.info(f"Average of {avg_developers_per_file:.2f} developers per file")

        return self.activity_stats

    def get_file_activity(self, filename: str) -> Dict[str, Any]:
        """
        Get activity statistics for a specific file.

        Args:
            filename: Path to the file

        Returns:
            Dictionary with file activity statistics
        """
        return self.activity_stats.get(filename, {})

    def get_developer_files(
        self, developer: str, min_ownership: float = 0.5
    ) -> List[str]:
        """
        Get files primarily owned by a specific developer.

        Args:
            developer: Developer identifier (email or name)
            min_ownership: Minimum ownership threshold (0.0-1.0)

        Returns:
            List of files owned by the developer
        """
        owned_files = []
        for file, stats in self.activity_stats.items():
            owner = stats.get("primary_owner", "")
            ownership = stats.get("ownership_ratio", 0.0)
            if owner == developer and ownership >= min_ownership:
                owned_files.append(file)

        return owned_files

    def get_top_developers(self, limit: int = 10) -> List[Tuple[str, int]]:
        """
        Get top developers by total number of changes.

        Args:
            limit: Maximum number of developers to return

        Returns:
            List of (developer, change_count) tuples
        """
        return sorted(
            self.developer_total_changes.items(), key=lambda x: x[1], reverse=True
        )[:limit]

    def get_hot_files(self, limit: int = 10) -> List[Tuple[str, int]]:
        """
        Get files with the most frequent changes.

        Args:
            limit: Maximum number of files to return

        Returns:
            List of (filename, change_count) tuples
        """
        return sorted(self.file_changes.items(), key=lambda x: x[1], reverse=True)[
            :limit
        ]

    def _process_commits(self, commits: List[Dict[str, Any]]) -> None:
        """
        Process commits to extract developer activity data.

        Args:
            commits: List of commit data dictionaries
        """
        logger.debug("Processing commits to track developer activity")

        # Process each commit
        for commit in tqdm(commits, desc="Processing commits", unit="commit"):
            author = commit["author"]
            timestamp = commit["timestamp"]
            files = commit["files"]

            # Update file changes and developer information
            for file in files:
                # Skip non-code files
                if not self.repository.is_code_file(file):
                    continue

                self.file_changes[file] += 1
                self.file_developers[file].add(author)
                self.file_last_modified[file] = timestamp
                self.developer_changes[author][file] += 1
                self.developer_total_changes[author] += 1

    def _calculate_developer_expertise(self) -> None:
        """
        Calculate expertise level for each developer based on contribution history.
        """
        logger.debug("Calculating developer expertise levels")

        # Get the total number of changes across all developers
        total_changes = sum(self.developer_total_changes.values())

        # Calculate raw expertise scores based on total changes
        raw_scores = {
            dev: changes / max(1, total_changes)
            for dev, changes in self.developer_total_changes.items()
        }

        # Apply normalization to get expertise scores between 0 and 1
        max_raw_score = max(raw_scores.values()) if raw_scores else 1.0

        for developer, raw_score in raw_scores.items():
            # Apply sigmoid-like function to balance scores
            normalized = raw_score / max_raw_score
            expertise = 1.0 / (1.0 + np.exp(-10 * (normalized - 0.5)))
            self.developer_expertise[developer] = expertise

    def _calculate_file_ownership(self) -> None:
        """
        Determine primary owner and ownership ratio for each file.
        """
        logger.debug("Calculating file ownership patterns")

        for file, developers in self.file_developers.items():
            # Skip files with no developers
            if not developers:
                continue

            # Get change counts for this file by developer
            dev_changes = {}
            for dev in developers:
                dev_changes[dev] = self.developer_changes[dev][file]

            # Determine primary owner (developer with most changes)
            primary_owner = max(dev_changes.items(), key=lambda x: x[1])
            self.file_owners[file] = primary_owner[0]

            # Calculate ownership ratio (primary owner changes / total changes)
            total_file_changes = sum(dev_changes.values())
            ownership_ratio = (
                primary_owner[1] / total_file_changes if total_file_changes > 0 else 0.0
            )

            # Store information in the file's activity stats
            if file not in self.activity_stats:
                self.activity_stats[file] = {}

            self.activity_stats[file]["primary_owner"] = primary_owner[0]
            self.activity_stats[file]["ownership_ratio"] = ownership_ratio

    def _compile_activity_stats(self) -> None:
        """
        Compile comprehensive activity statistics for each file.
        """
        logger.debug("Compiling comprehensive activity statistics")

        now = datetime.now()

        for file, changes in self.file_changes.items():
            # Skip files that don't exist in the current repository state
            if not self.repository.file_exists(file):
                continue

            # Calculate days since last modification
            last_modified = self.file_last_modified.get(file, now)
            days_since_modified = (now - last_modified).days

            # Get developers who have touched this file
            developers = self.file_developers.get(file, set())
            developer_count = len(developers)

            # Calculate average developer expertise for this file
            dev_expertise_values = [
                self.developer_expertise.get(dev, 0.5) for dev in developers
            ]
            avg_dev_expertise = sum(dev_expertise_values) / max(
                1, len(dev_expertise_values)
            )

            # Get primary owner and ownership ratio
            primary_owner = self.file_owners.get(file, "")
            ownership_ratio = 0.0
            if file in self.activity_stats:
                ownership_ratio = self.activity_stats[file].get("ownership_ratio", 0.0)

            # Calculate change frequency (changes per week)
            # Assume lookback period is approximately 6 months
            weeks_in_period = 26
            change_frequency = changes / weeks_in_period

            # Calculate recency factor (higher values for recently modified files)
            recency_factor = (
                np.exp(-days_since_modified / 30) if days_since_modified > 0 else 1.0
            )

            # Calculate expertise factor (lower values indicate higher risk)
            # This inverts expertise so higher risk is represented by higher values
            expertise_factor = 1.0 - avg_dev_expertise

            # Calculate ownership factor (lower ownership indicates higher risk)
            # This inverts ownership so higher risk is represented by higher values
            ownership_factor = 1.0 - ownership_ratio

            # Store all statistics
            self.activity_stats[file] = {
                "developer_count": developer_count,
                "primary_owner": primary_owner,
                "ownership_ratio": ownership_ratio,
                "change_count": changes,
                "change_frequency": change_frequency,
                "last_modified": last_modified.isoformat(),
                "days_since_modified": days_since_modified,
                "recency_factor": recency_factor,
                "expertise_level": avg_dev_expertise,
                "expertise_factor": expertise_factor,
                "ownership_factor": ownership_factor,
            }


if __name__ == "__main__":
    # Example usage
    import sys

    # Configure logging
    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) < 2:
        print("Usage: python developer_activity.py /path/to/repository")
        sys.exit(1)

    # Initialize repository and tracker
    repo_path = sys.argv[1]
    repository = Repository(repo_path)
    tracker = DeveloperActivityTracker(repository)

    # Track activity
    activity_stats = tracker.track_activity()

    # Print top developers
    print("\nTop 5 developers by contribution:")
    for dev, changes in tracker.get_top_developers(5):
        print(f"{dev}: {changes} changes")

    # Print hot files
    print("\nTop 5 most frequently changed files:")
    for file, changes in tracker.get_hot_files(5):
        print(f"{file}: {changes} changes")

    # Print a detailed analysis for the hottest file
    if tracker.get_hot_files():
        hot_file = tracker.get_hot_files()[0][0]
        stats = tracker.get_file_activity(hot_file)

        print(f"\nDetailed activity for {hot_file}:")
        print(f"  Changes: {stats.get('change_count', 0)}")
        print(f"  Change frequency: {stats.get('change_frequency', 0):.2f} per week")
        print(f"  Developers: {stats.get('developer_count', 0)}")
        print(f"  Primary owner: {stats.get('primary_owner', 'None')}")
        print(f"  Ownership ratio: {stats.get('ownership_ratio', 0):.2f}")
        print(f"  Last modified: {stats.get('last_modified', 'Unknown')}")
        print(f"  Days since modified: {stats.get('days_since_modified', 0)}")
        print(f"  Avg. developer expertise: {stats.get('expertise_level', 0):.2f}")
