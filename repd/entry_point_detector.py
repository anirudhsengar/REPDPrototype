#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Entry Point Detection Module for REPD Model

This module analyzes a repository to identify files that serve as entry points
to the codebase, such as main scripts, API endpoints, and interfaces.

Author: anirudhsengar
"""

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx

from repd.repository import Repository
from repd.structure_mapper import StructureMapper

logger = logging.getLogger(__name__)


class EntryPointDetector:
    """
    Detects entry points in a repository's codebase.

    Entry points are files that serve as interfaces or starting points for the codebase,
    such as main scripts, API endpoints, command handlers, etc.
    """

    def __init__(self, repository: Repository, structure_mapper: StructureMapper):
        """
        Initialize the entry point detector.

        Args:
            repository: Repository interface to analyze
            structure_mapper: Structure mapper with dependency information
        """
        self.repository = repository
        self.structure_mapper = structure_mapper
        self.entry_points = {}  # Dictionary mapping file paths to entry point scores

    def detect_entry_points(
        self,
        use_dependencies: bool = True,
        pattern_weight: float = 0.6,
        dependency_weight: float = 0.4,
    ) -> Dict[str, float]:
        """
        Detect entry points in the repository.

        Args:
            use_dependencies: Whether to use dependency information for detection
            pattern_weight: Weight to assign to pattern-based detection (0-1)
            dependency_weight: Weight to assign to dependency-based detection (0-1)

        Returns:
            Dictionary mapping file paths to entry point scores
        """
        logger.info("Detecting entry points")

        # Pattern-based detection
        pattern_scores = {}
        self._scan_files_for_signatures()
        pattern_scores = self.entry_points.copy()

        # Dependency-based detection if enabled
        dependency_scores = {}
        if (
            use_dependencies
            and self.structure_mapper
            and self.structure_mapper.dependency_graph
        ):
            dependency_scores = self._detect_by_dependencies()

            # Combine the scores
            self.entry_points = self._combine_scores(
                pattern_scores,
                dependency_scores,
                pattern_weight=pattern_weight,
                dependency_weight=dependency_weight,
            )
        else:
            self.entry_points = pattern_scores

        logger.info(
            f"Detected {len([s for s in self.entry_points.values() if s > 0.5])} "
            f"likely entry points out of {len(self.entry_points)} files analyzed"
        )

        return self.entry_points

    def get_entry_points(self, min_score: float = 0) -> Dict[str, float]:
        """
        Get detected entry points with scores above a threshold.

        Args:
            min_score: Minimum score to include (0-1)

        Returns:
            Dictionary mapping file paths to entry point scores
        """
        return {
            file: score
            for file, score in self.entry_points.items()
            if score >= min_score
        }

    def get_top_entry_points(self, n: int = 10) -> List[Tuple[str, float]]:
        """
        Get the top N entry points by score.

        Args:
            n: Number of entry points to return

        Returns:
            List of (file_path, score) tuples, sorted by score (descending)
        """
        return sorted(self.entry_points.items(), key=lambda x: x[1], reverse=True)[:n]

    def export_entry_points(self, output_file: str) -> None:
        """
        Export entry point data to a JSON file.

        Args:
            output_file: Path to the output file
        """
        data = {
            "entry_points": self.entry_points,
            "metadata": {
                "repository": self.repository.get_name(),
                "top_entry_points": [
                    {"file": file, "score": score}
                    for file, score in self.get_top_entry_points()
                ],
            },
        }

        with open(output_file, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Exported entry point data to {output_file}")

    def _scan_files_for_signatures(self) -> None:
        """
        Scan files for signatures that indicate entry points.
        """
        logger.debug("Scanning files for entry point signatures")

        # Get all code files
        files = [
            f
            for f in self.repository.get_all_files()
            if self.repository.is_code_file(f)
        ]

        # Dictionary to track entry point scores from pattern matching
        pattern_scores = {}

        # Analyze each file for entry point signatures
        for file in files:
            score = self._analyze_file_content(file)
            pattern_scores[file] = score

        self.entry_points = pattern_scores

    def _analyze_file_content(self, file_path: str) -> float:
        """
        Analyze a file's content for entry point signatures.

        Args:
            file_path: Path to the file to analyze

        Returns:
            Entry point score (0-1)
        """
        # Get file content
        content = self.repository.get_file_content(file_path)
        if not content:
            return 0.0

        # File name based scoring (conventional entry points)
        filename_score = self._score_by_filename(os.path.basename(file_path))

        # Language-specific pattern detection
        if file_path.endswith(".py"):
            pattern_score = self._analyze_python_file(content)
        elif file_path.endswith((".js", ".ts", ".jsx", ".tsx")):
            pattern_score = self._analyze_js_file(content)
        elif file_path.endswith((".java")):
            pattern_score = self._analyze_java_file(content)
        elif file_path.endswith((".go")):
            pattern_score = self._analyze_go_file(content)
        elif file_path.endswith((".rb")):
            pattern_score = self._analyze_ruby_file(content)
        elif file_path.endswith((".php")):
            pattern_score = self._analyze_php_file(content)
        elif file_path.endswith((".c", ".cpp", ".cc")):
            pattern_score = self._analyze_cpp_file(content)
        else:
            pattern_score = 0.0

        # Combine scores, with more weight to pattern detection
        combined_score = 0.7 * pattern_score + 0.3 * filename_score

        # Normalize to 0-1 range
        return min(1.0, combined_score)

    def _score_by_filename(self, filename: str) -> float:
        """
        Score a file based on its name (conventional entry points).

        Args:
            filename: File name to score

        Returns:
            Score based on file name conventions (0-1)
        """
        filename = filename.lower()

        # High-confidence entry point file names
        high_confidence = [
            "main.py",
            "main.js",
            "main.java",
            "main.go",
            "app.py",
            "app.js",
            "server.py",
            "server.js",
            "index.js",
            "index.php",
            "cli.py",
            "run.py",
            "start.py",
            "api.py",
            "application.java",
        ]

        if filename in high_confidence:
            return 0.9

        # Medium-confidence patterns
        medium_patterns = [
            r"^main\..*$",
            r"^app\..*$",
            r"^index\..*$",
            r"^server\..*$",
            r"^start\..*$",
            r"^run\..*$",
            r"^cli\..*$",
        ]

        for pattern in medium_patterns:
            if re.match(pattern, filename):
                return 0.6

        # Low-confidence patterns
        low_patterns = [
            r"^.*service\..*$",
            r"^.*controller\..*$",
            r"^.*handler\..*$",
            r"^.*endpoint\..*$",
            r"^.*application\..*$",
            r"^.*launcher\..*$",
            r"^.*bootstrap\..*$",
        ]

        for pattern in low_patterns:
            if re.match(pattern, filename):
                return 0.3

        return 0.0

    def _analyze_python_file(self, content: str) -> float:
        """
        Analyze a Python file for entry point patterns.

        Args:
            content: File content

        Returns:
            Entry point score (0-1)
        """
        score = 0.0

        # Check for if __name__ == "__main__" pattern
        if re.search(r'if\s+__name__\s*==\s*(\'|")__main__(\'|")', content):
            score += 0.9

        # Check for Flask/FastAPI app
        if re.search(r"app\s*=\s*(?:Flask|FastAPI)", content):
            score += 0.8

        # Check for route definitions
        if re.search(r"@.*\.route", content):
            score += 0.7

        # Check for click/argparse - CLI apps
        if re.search(
            r"(?:import\s+click|from\s+click\s+import|import\s+argparse|from\s+argparse\s+import)",
            content,
        ):
            score += 0.5

        # Check for Django management commands
        if re.search(r"from\s+django\.core\.management", content):
            score += 0.7

        # Check for setup.py entry points
        if re.search(r"entry_points\s*=", content):
            score += 0.6

        return min(1.0, score)

    def _analyze_js_file(self, content: str) -> float:
        """
        Analyze a JavaScript file for entry point patterns.

        Args:
            content: File content

        Returns:
            Entry point score (0-1)
        """
        score = 0.0

        # Check for Express app
        if re.search(r"(?:const|let|var)\s+app\s*=\s*express\(\)", content):
            score += 0.8

        # Check for app.listen pattern (server start)
        if re.search(r"(?:app|server)\.listen", content):
            score += 0.7

        # Check for React index file
        if re.search(r"ReactDOM\.render", content):
            score += 0.7

        # Check for route definitions
        if re.search(r"app\.(get|post|put|delete|patch)", content):
            score += 0.6

        # Check for commander/yargs (CLI)
        if re.search(r'require\([\'"](?:commander|yargs)[\'"]\)', content):
            score += 0.5

        # Check for exports = module.exports
        if not re.search(r"(?:exports|module\.exports)\s*=", content):
            # If it doesn't export anything, more likely to be an entry point
            score += 0.3

        return min(1.0, score)

    def _analyze_java_file(self, content: str) -> float:
        """
        Analyze a Java file for entry point patterns.

        Args:
            content: File content

        Returns:
            Entry point score (0-1)
        """
        score = 0.0

        # Check for main method
        if re.search(r"public\s+static\s+void\s+main\s*\(\s*String", content):
            score += 0.9

        # Check for Spring Boot application
        if re.search(r"@SpringBootApplication", content):
            score += 0.8

        # Check for REST controllers
        if re.search(r"@(?:Rest)?Controller", content):
            score += 0.7

        # Check for servlet
        if re.search(r"extends\s+(?:Http)?Servlet", content):
            score += 0.6

        return min(1.0, score)

    def _analyze_go_file(self, content: str) -> float:
        """
        Analyze a Go file for entry point patterns.

        Args:
            content: File content

        Returns:
            Entry point score (0-1)
        """
        score = 0.0

        # Check for main package
        if re.search(r"package\s+main", content):
            score += 0.5

        # Check for main function
        if re.search(r"func\s+main\s*\(\s*\)", content):
            score += 0.9

        # Check for HTTP handlers
        if re.search(r"func\s+\w+\s*\(\s*\w+\s+[*]?http\.ResponseWriter", content):
            score += 0.7

        # Check for Gin/Echo/other web framework routes
        if re.search(r"\.\s*(?:GET|POST|PUT|DELETE|Handle)\s*\(", content):
            score += 0.6

        return min(1.0, score)

    def _analyze_ruby_file(self, content: str) -> float:
        """
        Analyze a Ruby file for entry point patterns.

        Args:
            content: File content

        Returns:
            Entry point score (0-1)
        """
        score = 0.0

        # Check for shebang line
        if content.startswith("#!/"):
            score += 0.7

        # Check for Sinatra/Rails patterns
        if re.search(
            r"class\s+\w+\s*<\s*(?:Sinatra::Base|Rails::Application)", content
        ):
            score += 0.8

        # Check for command-line execution
        if re.search(r"if\s+__FILE__\s*==\s*\$0", content):
            score += 0.9

        # Check for route definitions
        if re.search(r'(?:get|post|put|delete|patch)\s+[\'"]', content):
            score += 0.6

        # Check for CLI tool
        if re.search(r'require\s+[\'"]thor[\'"]', content):
            score += 0.5

        return min(1.0, score)

    def _analyze_php_file(self, content: str) -> float:
        """
        Analyze a PHP file for entry point patterns.

        Args:
            content: File content

        Returns:
            Entry point score (0-1)
        """
        score = 0.0

        # Check if file starts with <?php
        if content.startswith("<?php"):
            score += 0.3

        # Check if it's a controller
        if re.search(r"class\s+\w+Controller", content):
            score += 0.7

        # Check for route definitions
        if re.search(r"Route::(get|post|put|delete|patch)", content):
            score += 0.6

        # Check if it extends a framework class
        if re.search(r"extends\s+(?:Controller|Command|Console)", content):
            score += 0.5

        return min(1.0, score)

    def _analyze_cpp_file(self, content: str) -> float:
        """
        Analyze a C/C++ file for entry point patterns.

        Args:
            content: File content

        Returns:
            Entry point score (0-1)
        """
        score = 0.0

        # Check for main function
        if re.search(r"int\s+main\s*\(", content):
            score += 0.9

        return min(1.0, score)

    def _detect_by_dependencies(self) -> Dict[str, float]:
        """
        Detect entry points based on dependency structure.

        Returns:
            Dictionary mapping file paths to entry point scores
        """
        logger.debug("Detecting entry points from dependency structure")

        dependency_scores = {}
        g = self.structure_mapper.dependency_graph

        if not g:
            return dependency_scores

        # Calculate in-degree (number of files importing this file)
        in_degree = dict(g.in_degree())

        # Calculate out-degree (number of files this file imports)
        out_degree = dict(g.out_degree())

        # Calculate total number of nodes for normalization
        total_nodes = len(g.nodes())
        max_in = max(in_degree.values()) if in_degree else 1

        for node in g.nodes():
            # Files with low in-degree (few/no imports) are potential entry points
            in_deg = in_degree.get(node, 0)
            out_deg = out_degree.get(node, 0)

            # Score based on in-degree (reversed and normalized)
            in_score = 1.0 - (in_deg / max_in) if max_in > 0 else 1.0

            # Adjust by out-degree (entry points often import many modules)
            # But cap the effect to avoid bias towards files that import everything
            out_factor = min(1.0, out_deg / 10) if total_nodes > 10 else 0.5

            # Combined score - weighted towards low in-degree with some boost from out-degree
            score = (0.8 * in_score) + (0.2 * out_factor)

            dependency_scores[node] = score

        return dependency_scores

    def _combine_scores(
        self,
        pattern_scores: Dict[str, float],
        dependency_scores: Dict[str, float],
        pattern_weight: float = 0.6,
        dependency_weight: float = 0.4,
    ) -> Dict[str, float]:
        """
        Combine scores from different detection methods.

        Args:
            pattern_scores: Scores from pattern-based detection
            dependency_scores: Scores from dependency-based detection
            pattern_weight: Weight to assign to pattern scores (0-1)
            dependency_weight: Weight to assign to dependency scores (0-1)

        Returns:
            Dictionary with combined scores
        """
        # Normalize weights
        total_weight = pattern_weight + dependency_weight
        pattern_weight /= total_weight
        dependency_weight /= total_weight

        # Get all files from both dictionaries
        all_files = set(pattern_scores.keys()) | set(dependency_scores.keys())

        # Combine scores
        combined_scores = {}
        for file in all_files:
            p_score = pattern_scores.get(file, 0)
            d_score = dependency_scores.get(file, 0)

            # Weighted combination
            combined_scores[file] = (p_score * pattern_weight) + (
                d_score * dependency_weight
            )

        return combined_scores
