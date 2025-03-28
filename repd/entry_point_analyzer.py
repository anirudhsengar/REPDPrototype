#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Entry Point Analyzer for REPD Model

This module identifies and scores entry points in a repository.
Entry points are files that serve as interfaces to the outside world
(APIs, UIs, exported functionality) and are critical in defect analysis.

Author: anirudhsengar
"""

import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from pygments import lexers
from pygments.util import ClassNotFound

from repd.repository import Repository

logger = logging.getLogger(__name__)


class EntryPointIdentifier:
    """
    Identifies entry points in a repository based on various heuristics.

    Entry points are files that:
    - Serve as APIs or public interfaces
    - Contain exported functionality
    - Are referenced by external systems
    - Serve as UI components or controllers
    - Represent configuration entry points
    """

    # Patterns that suggest entry points in various languages
    ENTRY_POINT_PATTERNS = {
        # APIs and Controllers
        "api_patterns": [
            r"(^|/)api/",
            r"(^|/)controller[s]?/",
            r"(^|/)endpoint[s]?/",
            r"(^|/)route[s]?/",
            r"(^|/)rest/",
            r"(^|/)graphql/",
            r"(^|/)public/",
            r"(^|/)external/",
        ],
        # UI related entry points
        "ui_patterns": [
            r"(^|/)ui/",
            r"(^|/)view[s]?/",
            r"(^|/)template[s]?/",
            r"(^|/)component[s]?/",
            r"(^|/)page[s]?/",
            r"(^|/)screen[s]?/",
            r"(^|/)dialog[s]?/",
        ],
        # Configuration entry points
        "config_patterns": [
            r"(^|/)config/",
            r"(^|/)settings/",
            r"\.config\.",
            r"\.conf$",
            r"\.ini$",
            r"\.json$",
            r"\.ya?ml$",
            r"\.toml$",
        ],
        # Main and entry files
        "main_patterns": [
            r"main\.([a-zA-Z]+)$",
            r"index\.([a-zA-Z]+)$",
            r"app\.([a-zA-Z]+)$",
            r"start\.([a-zA-Z]+)$",
            r"bootstrap\.([a-zA-Z]+)$",
            r"server\.([a-zA-Z]+)$",
            r"run\.([a-zA-Z]+)$",
            r"handler\.([a-zA-Z]+)$",
        ],
        # Plugin, hook, and extension points
        "plugin_patterns": [
            r"(^|/)plugin[s]?/",
            r"(^|/)extension[s]?/",
            r"(^|/)hook[s]?/",
            r"(^|/)middleware/",
            r"(^|/)callback[s]?/",
        ],
    }

    # Language-specific entry point indicators
    LANGUAGE_ENTRY_PATTERNS = {
        # Python
        "python": [
            r"def\s+main\s*\(",
            r"if\s+__name__\s*==\s*['\"]__main__['\"]",
            r"@app\.route",
            r"@api\.",
            r"class\s+.*\(.*View\)",
            r"class\s+.*\(.*Controller\)",
            r"class\s+.*\(.*Resource\)",
            r"class\s+.*Plugin\(",
        ],
        # JavaScript/TypeScript
        "javascript": [
            r"export\s+default",
            r"module\.exports",
            r"createApp\(",
            r"ReactDOM\.render",
            r"addEventListener\(['\"]load['\"]",
            r"new\s+Vue\(",
            r"angular\.module\(",
            r"app\.get\(",
            r"app\.post\(",
            r"app\.use\(",
            r"function\s+main\s*\(",
            r"const\s+router",
        ],
        # Java
        "java": [
            r"public\s+static\s+void\s+main\(",
            r"@RestController",
            r"@Controller",
            r"@RequestMapping",
            r"@GetMapping",
            r"@PostMapping",
            r"implements\s+.*Plugin",
            r"extends\s+.*Activity",
            r"extends\s+.*Application",
        ],
        # C/C++
        "c": [
            r"int\s+main\s*\(",
            r"void\s+main\s*\(",
            r"APIENTRY\s+wWinMain",
            r"DllMain\(",
        ],
        # Go
        "go": [
            r"func\s+main\(",
            r"func\s+init\(",
            r"http\.HandleFunc\(",
            r"http\.Handle\(",
            r"mux\.HandleFunc\(",
        ],
        # Ruby
        "ruby": [
            r"Rails\.application",
            r"get\s+['\"]/",
            r"post\s+['\"]/",
            r"namespace\s+:api",
        ],
        # PHP
        "php": [
            r"namespace\s+App\\Controller",
            r"class\s+.*Controller",
            r"function\s+__construct",
            r"\$app->get\(",
            r"\$router->add\(",
        ],
    }

    def __init__(self, repository: Repository):
        """
        Initialize the entry point identifier.

        Args:
            repository: Repository object to analyze
        """
        self.repository = repository

    def identify_entry_points(self, weight_factor: float = 0.5) -> Dict[str, float]:
        """
        Identify and score entry points in the repository.

        Args:
            weight_factor: Weight factor for entry point scoring (0.0-1.0)

        Returns:
            Dictionary mapping filenames to entry point scores (0.0-1.0)
        """
        logger.info("Identifying repository entry points")

        # Get all relevant files in repository
        files = self.repository.get_all_files()

        # Track entry points and their scores
        entry_points: Dict[str, float] = {}

        # Analyze each file
        for filename in files:
            # Skip non-code files
            if not self.repository.is_code_file(filename):
                continue

            # Calculate entry point score based on patterns
            pattern_score = self._calculate_pattern_score(filename)

            # Calculate content score if pattern score is non-zero
            content_score = 0.0
            if pattern_score > 0:
                content_score = self._calculate_content_score(filename)

            # Calculate reference score (how much this file is imported/referenced)
            reference_score = self._calculate_reference_score(filename)

            # Calculate final score weighted by importance
            final_score = self._calculate_weighted_score(
                pattern_score, content_score, reference_score, weight_factor
            )

            # If final score meets threshold, add to entry points
            if final_score > 0.1:  # Minimum threshold
                entry_points[filename] = min(final_score, 1.0)  # Cap at 1.0

        logger.info(f"Identified {len(entry_points)} entry points")
        return entry_points

    def _calculate_pattern_score(self, filename: str) -> float:
        """
        Calculate entry point score based on filename patterns.

        Args:
            filename: Path to file

        Returns:
            Pattern match score (0.0-1.0)
        """
        # Initialize score
        pattern_score = 0.0

        # Check against all pattern categories
        for category, patterns in self.ENTRY_POINT_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, filename, re.IGNORECASE):
                    # Different weights based on category
                    if category == "api_patterns":
                        pattern_score += 0.4
                    elif category == "main_patterns":
                        pattern_score += 0.35
                    elif category == "plugin_patterns":
                        pattern_score += 0.3
                    elif category == "ui_patterns":
                        pattern_score += 0.25
                    elif category == "config_patterns":
                        pattern_score += 0.2

                    # No need to check remaining patterns in this category
                    break

        # Special case: files in the repository root are often important
        if "/" not in filename and "\\" not in filename:  # No directory separators
            pattern_score += 0.15

        # Cap at 1.0
        return min(pattern_score, 1.0)

    def _calculate_content_score(self, filename: str) -> float:
        """
        Calculate entry point score based on file content.

        Args:
            filename: Path to file

        Returns:
            Content match score (0.0-1.0)
        """
        # Get file content
        try:
            content = self.repository.get_file_content(filename)
            if not content:
                return 0.0
        except Exception as e:
            logger.debug(f"Error reading file content for {filename}: {str(e)}")
            return 0.0

        # Determine language for language-specific patterns
        language = self._detect_language(filename)

        # Initialize score
        content_score = 0.0

        # Check language-specific patterns
        if language in self.LANGUAGE_ENTRY_PATTERNS:
            patterns = self.LANGUAGE_ENTRY_PATTERNS[language]
            for pattern in patterns:
                if re.search(pattern, content, re.MULTILINE):
                    content_score += 0.25
                    # Cap at 0.75 for language patterns
                    if content_score >= 0.75:
                        break

        # Check general entry point indicators in content
        general_indicators = [
            r"@api",
            r"@public",
            r"@expose",
            r"@export",
            r"\bpublic\s+interface\b",
            r"\bpublic\s+class\b",
            r"\bexport\b",
            r"\bprovide\b",
        ]

        for indicator in general_indicators:
            if re.search(indicator, content, re.MULTILINE | re.IGNORECASE):
                content_score += 0.15
                # Cap at 1.0
                if content_score >= 1.0:
                    break

        # Adjust based on file size
        size_factor = min(1.0, len(content) / (50 * 1024))  # 50KB cap
        content_score *= 0.7 + 0.3 * size_factor

        return min(content_score, 1.0)

    def _calculate_reference_score(self, filename: str) -> float:
        """
        Calculate score based on how much this file is referenced by others.

        Args:
            filename: Path to file

        Returns:
            Reference score (0.0-1.0)
        """
        # This is a simplification - in a real implementation this would:
        # 1. Analyze import/include statements across the codebase
        # 2. Track how many files reference this file
        # 3. Calculate a normalized score

        # For now, use a heuristic approach based on filename characteristics

        # Extract the base name without extension
        base_name = Path(filename).stem

        # Score based on common names that are often imported
        common_imported_names = [
            "util",
            "utils",
            "helper",
            "common",
            "core",
            "base",
            "api",
            "client",
            "server",
            "index",
            "constants",
            "config",
            "types",
            "model",
            "shared",
        ]

        # Check if the base name is one of the commonly imported names
        base_score = 0.0
        for name in common_imported_names:
            if name in base_name.lower():
                base_score = 0.3
                break

        # Check file extension for additional score
        extension = Path(filename).suffix.lower()
        if extension in [".h", ".hpp", ".d.ts", ".py", ".rb"]:
            base_score += 0.1

        # Calculate final score based on directory depth
        # Files closer to root are typically more referenced
        parts = Path(filename).parts
        depth_factor = max(0, 1.0 - (len(parts) - 1) * 0.1)  # Decrease by 0.1 per level

        # Combine scores
        reference_score = base_score * (0.7 + 0.3 * depth_factor)

        return min(reference_score, 1.0)

    def _calculate_weighted_score(
        self,
        pattern_score: float,
        content_score: float,
        reference_score: float,
        weight_factor: float,
    ) -> float:
        """
        Calculate weighted entry point score.

        Args:
            pattern_score: Score from filename patterns (0.0-1.0)
            content_score: Score from file content (0.0-1.0)
            reference_score: Score from references (0.0-1.0)
            weight_factor: Weight factor for entry point scoring (0.0-1.0)

        Returns:
            Weighted entry point score (0.0-1.0)
        """
        # Apply weights to each score component
        weighted_score = (
            0.4 * pattern_score + 0.4 * content_score + 0.2 * reference_score
        )

        # Apply overall weight factor
        final_score = weighted_score * weight_factor

        return final_score

    def _detect_language(self, filename: str) -> str:
        """
        Detect the programming language of a file.

        Args:
            filename: Path to the file

        Returns:
            Detected language name or empty string if unknown
        """
        # Get file extension
        _, ext = os.path.splitext(filename)

        # Simple mapping for common extensions
        simple_map = {
            ".py": "python",
            ".js": "javascript",
            ".jsx": "javascript",
            ".ts": "javascript",  # TypeScript is handled like JavaScript for our purposes
            ".tsx": "javascript",
            ".java": "java",
            ".c": "c",
            ".cpp": "c",
            ".cc": "c",
            ".h": "c",
            ".hpp": "c",
            ".go": "go",
            ".rb": "ruby",
            ".php": "php",
            ".rs": "rust",
            ".swift": "swift",
            ".kt": "kotlin",
            ".cs": "csharp",
        }

        # Check simple map first
        if ext.lower() in simple_map:
            return simple_map[ext.lower()]

        # Use Pygments for more sophisticated detection
        try:
            lexer = lexers.get_lexer_for_filename(filename)
            lang = lexer.name.lower()

            # Map Pygments language to our language categories
            if "python" in lang:
                return "python"
            elif "javascript" in lang or "typescript" in lang or "jsx" in lang:
                return "javascript"
            elif "java" in lang:
                return "java"
            elif "c++" in lang or "c" == lang or "objective c" in lang:
                return "c"
            elif "go" in lang:
                return "go"
            elif "ruby" in lang:
                return "ruby"
            elif "php" in lang:
                return "php"
        except (ClassNotFound, Exception) as e:
            logger.debug(f"Could not detect language for {filename}: {str(e)}")

        # Default to empty string if language could not be determined
        return ""


if __name__ == "__main__":
    # Example usage
    import sys

    # Configure logging
    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) < 2:
        print("Usage: python entry_point_analyzer.py /path/to/repository")
        sys.exit(1)

    # Initialize repository and analyzer
    repo_path = sys.argv[1]
    repository = Repository(repo_path)
    analyzer = EntryPointIdentifier(repository)

    # Identify entry points
    entry_points = analyzer.identify_entry_points()

    # Print top entry points
    print("\nTop 10 entry points:")
    for file, score in sorted(entry_points.items(), key=lambda x: x[1], reverse=True)[
        :10
    ]:
        print(f"{file}: {score:.4f}")
