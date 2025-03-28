#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Structure Mapping Module for REPD Model

This module analyzes the structure of a repository to build a dependency graph
that represents relationships between files, including imports, inheritance,
and other forms of coupling.

Author: anirudhsengar
"""

import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import matplotlib.pyplot as plt
import networkx as nx

from repd.repository import Repository

logger = logging.getLogger(__name__)


class StructureMapper:
    """Maps the structure of a repository by analyzing file dependencies."""

    def __init__(self, repository: Repository):
        """
        Initialize the structure mapper with a repository.

        Args:
            repository: Repository interface to analyze
        """
        self.repository = repository
        self.dependency_graph = nx.DiGraph()
        self.import_map = {}  # Map of file to its imports
        self.inheritance_map = {}  # Map of class to its parent classes
        self.centrality_scores = {}  # Centrality measures for files
        self.entry_points = {}  # Entry points to the codebase

    def map_structure(self, max_files: int = None) -> nx.DiGraph:
        """
        Map the structure of the repository by analyzing dependencies.

        Args:
            max_files: Maximum number of files to analyze (None for all)

        Returns:
            Directed graph representing the dependency structure
        """
        logger.info("Mapping repository structure")

        # Get all code files in the repository
        files = [
            f
            for f in self.repository.get_all_files()
            if self.repository.is_code_file(f)
        ]

        if max_files and len(files) > max_files:
            logger.warning(
                f"Limiting analysis to {max_files} files (out of {len(files)})"
            )
            files = files[:max_files]

        # Create nodes for each file
        for file in files:
            self.dependency_graph.add_node(file)

        # Analyze dependencies between files
        for file in files:
            self._analyze_file_dependencies(file)

        # Calculate centrality measures
        self._calculate_centrality_measures()

        logger.info(
            f"Mapped structure with {self.dependency_graph.number_of_nodes()} nodes "
            f"and {self.dependency_graph.number_of_edges()} edges"
        )

        return self.dependency_graph

    def _analyze_file_dependencies(self, file_path: str) -> None:
        """
        Analyze dependencies in a file and add them to the graph.

        Args:
            file_path: Path to the file to analyze
        """
        logger.debug(f"Analyzing dependencies in {file_path}")

        # Get file content
        content = self.repository.get_file_content(file_path)
        if not content:
            logger.warning(f"Could not get content for {file_path}")
            return

        # Detect language
        language = self._detect_language(file_path)
        if not language:
            logger.warning(f"Could not detect language for {file_path}")
            return

        # Extract imports based on language
        imports = self._extract_imports(content, language)
        self.import_map[file_path] = imports

        # Resolve imports to actual files
        for imported_item in imports:
            resolved_file = self._resolve_import_to_file(
                imported_item, file_path, language
            )
            if resolved_file and resolved_file in self.dependency_graph:
                self.dependency_graph.add_edge(file_path, resolved_file)

        # For object-oriented languages, extract inheritance relationships
        if language in ["python", "java", "cpp", "csharp"]:
            inheritance = self._extract_inheritance(content, language)
            self.inheritance_map[file_path] = inheritance

            # Resolve parent classes to files
            for class_name in inheritance:
                resolved_file = self._resolve_class_to_file(
                    class_name, file_path, language
                )
                if resolved_file and resolved_file in self.dependency_graph:
                    self.dependency_graph.add_edge(file_path, resolved_file)

    def _extract_imports(self, content: str, language: str) -> List[str]:
        """
        Extract import statements from file content based on language.

        Args:
            content: File content
            language: Programming language

        Returns:
            List of imported modules or packages
        """
        imports = []

        if language == "python":
            # Regular import statements
            import_pattern = r"^\s*import\s+([a-zA-Z0-9_.,\s]+)"
            from_import_pattern = r"^\s*from\s+([a-zA-Z0-9_.]+)\s+import"

            # Find all imports
            for line in content.split("\n"):
                import_match = re.match(import_pattern, line)
                if import_match:
                    modules = import_match.group(1).split(",")
                    for module in modules:
                        module = module.strip()
                        if module:
                            imports.append(module)

                from_match = re.match(from_import_pattern, line)
                if from_match:
                    imports.append(from_match.group(1))

        elif language == "javascript":
            # ES6 style imports
            import_pattern = r'import\s+.*\s+from\s+[\'"]([^\'"]+)[\'"]'
            require_pattern = r'require\s*\(\s*[\'"]([^\'"]+)[\'"]\s*\)'

            # Find all imports
            for match in re.finditer(import_pattern, content):
                imports.append(match.group(1))

            for match in re.finditer(require_pattern, content):
                imports.append(match.group(1))

        elif language == "java":
            # Java imports
            import_pattern = r"^\s*import\s+([a-zA-Z0-9_.]+(\.[*])?);\s*$"

            # Find all imports
            for line in content.split("\n"):
                match = re.match(import_pattern, line)
                if match:
                    imports.append(match.group(1))

        elif language == "cpp":
            # C++ includes
            include_pattern = r'#include\s+[<"]([^>"]+)[>"]'

            # Find all includes
            for match in re.finditer(include_pattern, content):
                imports.append(match.group(1))

        elif language == "go":
            # Go imports
            import_pattern = r"import\s+\(\s*(.*?)\s*\)"
            single_import_pattern = r'import\s+"([^"]+)"'

            # Find multi-line imports
            matches = re.findall(import_pattern, content, re.DOTALL)
            for match in matches:
                for line in match.split("\n"):
                    stripped = line.strip()
                    if stripped and stripped.startswith('"') and stripped.endswith('"'):
                        imports.append(stripped[1:-1])  # Remove quotes

            # Find single imports
            for match in re.finditer(single_import_pattern, content):
                imports.append(match.group(1))

        elif language == "ruby":
            # Ruby requires
            require_pattern = r'require\s+[\'"]([^\'"]+)[\'"]'

            # Find all requires
            for match in re.finditer(require_pattern, content):
                imports.append(match.group(1))

        elif language == "php":
            # PHP includes and requires
            include_pattern = (
                r'(include|require|include_once|require_once)\s+[\'"]([^\'"]+)[\'"]'
            )
            namespace_pattern = r"use\s+([a-zA-Z0-9_\\]+)"

            # Find all includes
            for match in re.finditer(include_pattern, content):
                imports.append(match.group(2))

            # Find all namespace imports
            for match in re.finditer(namespace_pattern, content):
                imports.append(match.group(1))

        return imports

    def _extract_inheritance(self, content: str, language: str) -> List[str]:
        """
        Extract inheritance relationships from file content.

        Args:
            content: File content
            language: Programming language

        Returns:
            List of parent classes
        """
        inheritance = []

        if language == "python":
            # Python class definitions with inheritance
            class_pattern = r"^\s*class\s+([a-zA-Z0-9_]+)\s*\(([^)]+)\)"

            # Find all class definitions with inheritance
            for match in re.finditer(class_pattern, content, re.MULTILINE):
                parents = match.group(2).split(",")
                for parent in parents:
                    parent = parent.strip()
                    if parent and parent not in ["object", "Exception"]:
                        inheritance.append(parent)

        elif language == "java":
            # Java class/interface definitions with extends/implements
            extends_pattern = r"class\s+[a-zA-Z0-9_]+\s+extends\s+([a-zA-Z0-9_]+)"
            implements_pattern = (
                r"(class|interface)\s+[a-zA-Z0-9_]+\s+implements\s+([^{]+)"
            )

            # Find all extends relationships
            for match in re.finditer(extends_pattern, content):
                inheritance.append(match.group(1))

            # Find all implements relationships
            for match in re.finditer(implements_pattern, content):
                interfaces = match.group(2).split(",")
                for interface in interfaces:
                    interface = interface.strip()
                    if interface:
                        inheritance.append(interface)

        elif language == "cpp":
            # C++ class definitions with inheritance
            class_pattern = (
                r"class\s+[a-zA-Z0-9_]+\s*:\s*(?:public|protected|private)?\s*([^{,]+)"
            )

            # Find all class definitions with inheritance
            for match in re.finditer(class_pattern, content):
                parent = match.group(1).strip()
                if parent:
                    inheritance.append(parent)

        elif language == "csharp":
            # C# class definitions with inheritance
            class_pattern = r"class\s+[a-zA-Z0-9_]+\s*:\s*([^{,]+)"

            # Find all class definitions with inheritance
            for match in re.finditer(class_pattern, content):
                parents = match.group(1).split(",")
                for parent in parents:
                    parent = parent.strip()
                    if parent:
                        inheritance.append(parent)

        return inheritance

    def _resolve_import_to_file(
        self, import_name: str, source_file: str, language: str
    ) -> Optional[str]:
        """
        Resolve an import statement to the actual file it refers to.

        Args:
            import_name: The imported module or package
            source_file: The file containing the import
            language: The programming language

        Returns:
            Path to the imported file, or None if not found
        """
        # Get repository file list (limit the search to code files)
        all_files = [
            f
            for f in self.repository.get_all_files()
            if self.repository.is_code_file(f)
        ]

        # Handle language-specific import resolution
        if language == "python":
            # Convert import to potential file paths
            potential_paths = []

            # Convert dot notation to directory structure
            path_parts = import_name.split(".")
            path_with_init = os.path.join(*path_parts, "__init__.py")
            path_with_extension = f"{os.path.join(*path_parts)}.py"

            potential_paths.append(path_with_init)
            potential_paths.append(path_with_extension)

            # Handle relative imports
            if import_name.startswith("."):
                source_dir = os.path.dirname(source_file)
                rel_import = import_name.lstrip(".")
                rel_parts = rel_import.split(".")

                # Go up one directory for each dot
                up_count = len(import_name) - len(rel_import)
                for _ in range(up_count):
                    source_dir = os.path.dirname(source_dir)

                rel_path_with_init = os.path.join(source_dir, *rel_parts, "__init__.py")
                rel_path_with_extension = f"{os.path.join(source_dir, *rel_parts)}.py"

                potential_paths.append(rel_path_with_init)
                potential_paths.append(rel_path_with_extension)

            # Check if any potential path exists in the repository
            for path in potential_paths:
                normalized_path = self.repository.normalize_path(path)
                if normalized_path in all_files:
                    return normalized_path

        elif language == "javascript":
            # Handle JavaScript/TypeScript imports
            potential_paths = []

            # Direct path with extension
            for ext in [".js", ".jsx", ".ts", ".tsx"]:
                potential_paths.append(f"{import_name}{ext}")

            # Directory with index file
            for ext in [".js", ".jsx", ".ts", ".tsx"]:
                potential_paths.append(os.path.join(import_name, f"index{ext}"))

            # Handle relative imports
            source_dir = os.path.dirname(source_file)
            if import_name.startswith("./") or import_name.startswith("../"):
                for ext in [".js", ".jsx", ".ts", ".tsx"]:
                    rel_path = os.path.normpath(
                        os.path.join(source_dir, f"{import_name}{ext}")
                    )
                    potential_paths.append(rel_path)

                    # Directory with index file for relative imports
                    rel_dir_path = os.path.normpath(
                        os.path.join(source_dir, import_name, f"index{ext}")
                    )
                    potential_paths.append(rel_dir_path)

            # Check if any potential path exists in the repository
            for path in potential_paths:
                normalized_path = self.repository.normalize_path(path)
                if normalized_path in all_files:
                    return normalized_path

        elif language == "java":
            # Handle Java imports
            # Convert package notation to directory structure
            if import_name.endswith(".*"):
                # This is a package import, look for any file in the package
                package = import_name[:-2]
                package_path = package.replace(".", "/")

                # Find any file in this package
                for file in all_files:
                    if file.startswith(package_path + "/") and file.endswith(".java"):
                        return file
            else:
                # This is a specific class import
                class_path = import_name.replace(".", "/") + ".java"
                normalized_path = self.repository.normalize_path(class_path)

                if normalized_path in all_files:
                    return normalized_path

        elif language == "cpp":
            # Handle C++ includes
            # Check in include paths
            include_paths = ["include", "src", ""]  # Common include directories

            for base in include_paths:
                potential_path = os.path.join(base, import_name)
                normalized_path = self.repository.normalize_path(potential_path)

                if normalized_path in all_files:
                    return normalized_path

        elif language == "ruby":
            # Handle Ruby requires
            potential_paths = []

            # Try direct path with .rb extension
            potential_paths.append(f"{import_name}.rb")

            # Try with lib directory
            potential_paths.append(os.path.join("lib", f"{import_name}.rb"))

            # Check if any potential path exists in the repository
            for path in potential_paths:
                normalized_path = self.repository.normalize_path(path)
                if normalized_path in all_files:
                    return normalized_path

        elif language == "php":
            # Handle PHP includes
            potential_paths = []

            # Direct path
            potential_paths.append(import_name)

            # Path with .php extension
            if not import_name.endswith(".php"):
                potential_paths.append(f"{import_name}.php")

            # Path relative to source directory
            source_dir = os.path.dirname(source_file)
            potential_paths.append(os.path.join(source_dir, import_name))

            # Check if any potential path exists in the repository
            for path in potential_paths:
                normalized_path = self.repository.normalize_path(path)
                if normalized_path in all_files:
                    return normalized_path

        # No match found
        return None

    def _resolve_class_to_file(
        self, class_name: str, source_file: str, language: str
    ) -> Optional[str]:
        """
        Resolve a class name to the file that defines it.

        Args:
            class_name: The class name to resolve
            source_file: The file containing the reference to the class
            language: The programming language

        Returns:
            Path to the file defining the class, or None if not found
        """
        # Get repository file list (limit the search to code files)
        all_files = [
            f
            for f in self.repository.get_all_files()
            if self.repository.is_code_file(f)
        ]

        # Handle language-specific class resolution
        if language == "python":
            # Look for class definition in each Python file
            for file in all_files:
                if not file.endswith(".py"):
                    continue

                content = self.repository.get_file_content(file)
                if not content:
                    continue

                # Look for class definition
                class_pattern = r"^\s*class\s+" + re.escape(class_name) + r"\s*(\(|:)"
                if re.search(class_pattern, content, re.MULTILINE):
                    return file

        elif language == "java":
            # In Java, class name typically matches file name
            potential_file = class_name + ".java"

            # Check all possible locations
            for file in all_files:
                if file.endswith("/" + potential_file):
                    return file

        elif language == "cpp":
            # Look for class definition in header files
            for file in all_files:
                if not file.endswith((".h", ".hpp")):
                    continue

                content = self.repository.get_file_content(file)
                if not content:
                    continue

                # Look for class definition
                class_pattern = (
                    r"(class|struct)\s+" + re.escape(class_name) + r"\s*[{:]"
                )
                if re.search(class_pattern, content):
                    return file

        elif language == "csharp":
            # In C#, class name might match file name
            potential_file = class_name + ".cs"

            # Check all possible locations
            for file in all_files:
                if file.endswith("/" + potential_file):
                    return file

                # If not found by name, look for class definition
                content = self.repository.get_file_content(file)
                if not content:
                    continue

                # Look for class definition
                class_pattern = (
                    r"(public|internal|private)?\s+class\s+"
                    + re.escape(class_name)
                    + r"\s*[{:]"
                )
                if re.search(class_pattern, content):
                    return file

        # No match found
        return None

    def _detect_language(self, file_path: str) -> Optional[str]:
        """
        Detect the programming language of a file based on extension and content.

        Args:
            file_path: Path to the file

        Returns:
            Language identifier or None if not detected
        """
        # Map extensions to languages
        extension_map = {
            ".py": "python",
            ".js": "javascript",
            ".jsx": "javascript",
            ".ts": "javascript",
            ".tsx": "javascript",
            ".java": "java",
            ".c": "cpp",
            ".cpp": "cpp",
            ".cc": "cpp",
            ".h": "cpp",
            ".hpp": "cpp",
            ".go": "go",
            ".rb": "ruby",
            ".php": "php",
            ".cs": "csharp",
            ".html": "html",
            ".css": "css",
        }

        # Get file extension
        _, ext = os.path.splitext(file_path.lower())

        # Check if extension is in the map
        if ext in extension_map:
            return extension_map[ext]

        # If extension doesn't match, try content-based detection
        content = self.repository.get_file_content(file_path)
        if not content:
            return None

        # Check for language-specific patterns
        if re.search(r"^\s*package\s+[a-zA-Z0-9_.]+;", content):
            return "java"
        if re.search(r"^\s*<\?php", content):
            return "php"
        if re.search(
            r"^\s*(import\s+[a-zA-Z0-9_.]+|from\s+[a-zA-Z0-9_.]+\s+import)", content
        ):
            return "python"
        if re.search(r"^\s*(let|const|var|function|import)\s+", content):
            return "javascript"

        # Unknown language
        return None

    def get_dependencies(self, file_path: str) -> List[str]:
        """
        Get direct dependencies of a file.

        Args:
            file_path: Path to the file

        Returns:
            List of files that the given file depends on
        """
        if file_path not in self.dependency_graph:
            return []

        return list(self.dependency_graph.successors(file_path))

    def get_dependents(self, file_path: str) -> List[str]:
        """
        Get files that directly depend on a file.

        Args:
            file_path: Path to the file

        Returns:
            List of files that depend on the given file
        """
        if file_path not in self.dependency_graph:
            return []

        return list(self.dependency_graph.predecessors(file_path))

    def get_recursive_dependencies(self, file_path: str) -> Set[str]:
        """
        Get all dependencies of a file recursively.

        Args:
            file_path: Path to the file

        Returns:
            Set of all files that the given file depends on (directly or indirectly)
        """
        if file_path not in self.dependency_graph:
            return set()

        return self._get_recursive_dependencies(file_path, set())

    def _get_recursive_dependencies(
        self, file_path: str, visited: Set[str]
    ) -> Set[str]:
        """
        Helper method to get recursive dependencies with cycle detection.

        Args:
            file_path: Path to the file
            visited: Set of already visited files to detect cycles

        Returns:
            Set of all dependencies
        """
        if file_path in visited:
            return set()

        result = set()
        visited.add(file_path)

        for dependency in self.get_dependencies(file_path):
            result.add(dependency)
            result.update(self._get_recursive_dependencies(dependency, visited.copy()))

        return result

    def get_recursive_dependents(self, file_path: str) -> Set[str]:
        """
        Get all files that depend on a file recursively.

        Args:
            file_path: Path to the file

        Returns:
            Set of all files that depend on the given file (directly or indirectly)
        """
        if file_path not in self.dependency_graph:
            return set()

        return self._get_recursive_dependents(file_path, set())

    def _get_recursive_dependents(self, file_path: str, visited: Set[str]) -> Set[str]:
        """
        Helper method to get recursive dependents with cycle detection.

        Args:
            file_path: Path to the file
            visited: Set of already visited files to detect cycles

        Returns:
            Set of all dependents
        """
        if file_path in visited:
            return set()

        result = set()
        visited.add(file_path)

        for dependent in self.get_dependents(file_path):
            result.add(dependent)
            result.update(self._get_recursive_dependents(dependent, visited.copy()))

        return result

    def get_central_files(self, top_n: int = None) -> List[Tuple[str, float]]:
        """
        Get the most central files in the repository based on centrality measures.

        Args:
            top_n: Number of files to return (None for all)

        Returns:
            List of (file_path, centrality_score) tuples, sorted by centrality
        """
        if not self.centrality_scores:
            self._calculate_centrality_measures()

        # Sort files by centrality score (descending)
        sorted_files = sorted(
            self.centrality_scores.items(), key=lambda x: x[1], reverse=True
        )

        # Return top N if specified
        if top_n:
            return sorted_files[:top_n]
        return sorted_files

    def _calculate_centrality_measures(self) -> None:
        """
        Calculate various centrality measures for the dependency graph.
        """
        logger.info("Calculating centrality measures")

        if not self.dependency_graph.nodes():
            logger.warning("No nodes in graph, skipping centrality calculation")
            return

        # Calculate different centrality measures
        try:
            # Betweenness centrality - how often a node acts as a bridge
            betweenness = nx.betweenness_centrality(self.dependency_graph)

            # Degree centrality - number of connections
            in_degree = nx.in_degree_centrality(self.dependency_graph)
            out_degree = nx.out_degree_centrality(self.dependency_graph)

            # PageRank - importance of a node based on connections
            pagerank = nx.pagerank(self.dependency_graph)

            # Combine centrality measures with weights
            weights = {
                "betweenness": 0.35,
                "in_degree": 0.3,
                "out_degree": 0.15,
                "pagerank": 0.2,
            }

            # Calculate combined centrality score
            for node in self.dependency_graph.nodes():
                self.centrality_scores[node] = (
                    betweenness.get(node, 0) * weights["betweenness"]
                    + in_degree.get(node, 0) * weights["in_degree"]
                    + out_degree.get(node, 0) * weights["out_degree"]
                    + pagerank.get(node, 0) * weights["pagerank"]
                )

            # Store individual centrality measures as node attributes
            nx.set_node_attributes(self.dependency_graph, betweenness, "betweenness")
            nx.set_node_attributes(self.dependency_graph, in_degree, "in_degree")
            nx.set_node_attributes(self.dependency_graph, out_degree, "out_degree")
            nx.set_node_attributes(self.dependency_graph, pagerank, "pagerank")
            nx.set_node_attributes(
                self.dependency_graph, self.centrality_scores, "centrality"
            )

            logger.info("Centrality measures calculated successfully")

        except Exception as e:
            logger.error(f"Error calculating centrality measures: {str(e)}")
            # Initialize with empty values if calculation fails
            self.centrality_scores = {
                node: 0.0 for node in self.dependency_graph.nodes()
            }

    def visualize_dependency_graph(
        self, output_path: str, max_nodes: int = 100
    ) -> None:
        """
        Visualize the dependency graph.

        Args:
            output_path: Path to save the visualization
            max_nodes: Maximum number of nodes to include in the visualization
        """
        if not self.dependency_graph.nodes():
            logger.warning("No nodes in graph, skipping visualization")
            return

        # Create a copy of the graph for visualization
        if len(self.dependency_graph) > max_nodes:
            logger.info(f"Limiting visualization to {max_nodes} most central nodes")

            # Get top central nodes
            top_files = [f for f, _ in self.get_central_files(max_nodes)]

            # Create subgraph
            viz_graph = self.dependency_graph.subgraph(top_files)
        else:
            viz_graph = self.dependency_graph

        # Create figure
        plt.figure(figsize=(12, 10))

        # Use spring layout for node positioning
        pos = nx.spring_layout(viz_graph, k=0.3, seed=42)

        # Get centrality for node sizing
        centrality = self.centrality_scores or nx.degree_centrality(viz_graph)
        node_sizes = [
            5000 * centrality.get(node, 0.1) + 100 for node in viz_graph.nodes()
        ]

        # Draw the graph
        nx.draw(
            viz_graph,
            pos,
            with_labels=False,
            node_size=node_sizes,
            node_color="skyblue",
            edge_color="gray",
            alpha=0.8,
            arrows=True,
        )

        # Add labels for most central nodes
        if len(viz_graph) > 20:
            # Only label top 20 nodes by centrality
            top_files = [f for f, _ in self.get_central_files(20)]
            label_dict = {
                node: self._get_short_name(node)
                for node in top_files
                if node in viz_graph
            }
        else:
            # Label all nodes
            label_dict = {
                node: self._get_short_name(node) for node in viz_graph.nodes()
            }

        nx.draw_networkx_labels(viz_graph, pos, labels=label_dict, font_size=8)

        # Add title
        plt.title(f"Dependency Graph for {self.repository.get_name()}", fontsize=16)

        # Save figure
        plt.savefig(output_path, bbox_inches="tight", dpi=300)
        plt.close()

        logger.info(f"Dependency graph visualization saved to {output_path}")

    def _get_short_name(self, file_path: str) -> str:
        """
        Get a shortened version of the file name for display.

        Args:
            file_path: Full file path

        Returns:
            Shortened file name
        """
        # Get basename
        name = os.path.basename(file_path)

        # If path has directory components, add the parent directory
        if "/" in file_path:
            parent = os.path.basename(os.path.dirname(file_path))
            return f"{parent}/{name}"

        return name
