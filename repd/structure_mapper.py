#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Structure Mapper for REPD Model

This module analyzes and maps the structural relationships between files
in a repository, including import dependencies, inheritance hierarchies,
and package structures to identify critical components.

Author: anirudhsengar
"""

import logging
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple, DefaultDict, Optional, Any

import networkx as nx
from tqdm import tqdm

from repd.repository import Repository

logger = logging.getLogger(__name__)


class StructureMapper:
    """
    Maps structural relationships between files in a repository.

    This class analyzes codebases to extract dependency information,
    identify import relationships, map class hierarchies, and build
    a comprehensive graph of the codebase structure.
    """

    # Language-specific import patterns for dependency extraction
    IMPORT_PATTERNS = {
        # Python imports
        "python": [
            r"^\s*import\s+([a-zA-Z0-9_.,\s]+)$",  # import module
            r"^\s*from\s+([a-zA-Z0-9_.]+)\s+import\s+",  # from module import
            r"^\s*import\s+([a-zA-Z0-9_.]+)\s+as\s+",  # import module as alias
        ],
        # JavaScript/TypeScript imports
        "javascript": [
            r"^\s*import\s+.*\s+from\s+['\"]([^'\"]+)['\"]",  # import x from 'module'
            r"^\s*import\s+['\"]([^'\"]+)['\"]",  # import 'module'
            r"^\s*require\s*\(\s*['\"]([^'\"]+)['\"]",  # require('module')
        ],
        # Java imports
        "java": [
            r"^\s*import\s+([a-zA-Z0-9_.]+)(\.[\*\w]+)?;",  # import package.Class
            r"^\s*import\s+static\s+([a-zA-Z0-9_.]+)(\.[\*\w]+)?;",  # import static
        ],
        # C/C++ includes
        "c": [
            r'^\s*#\s*include\s*[<"]([\w./]+)[>"]',  # #include <header> or "header"
        ],
        # Go imports
        "go": [
            r'^\s*import\s+[(_]?\s*"([^"]+)"',  # import "package"
            r'^\s*import\s+\(\s*"([^"]+)"',  # import ("package"...)
        ],
        # Ruby requires
        "ruby": [
            r'^\s*require\s+[\'"]([^\'"]+)[\'"]',  # require 'module'
            r'^\s*require_relative\s+[\'"]([^\'"]+)[\'"]',  # require_relative 'module'
        ],
        # PHP includes
        "php": [
            r'^\s*(?:require|include)(?:_once)?\s*\(\s*[\'"]([^\'"]+)[\'"]',  # require('file')
            r'^\s*use\s+([a-zA-Z0-9_\\]+)',  # use Namespace\Class
        ],
    }

    # Inheritance patterns to detect class relationships
    INHERITANCE_PATTERNS = {
        # Python class inheritance
        "python": [
            r"^\s*class\s+(\w+)\s*\(([^)]+)\):",  # class Name(Parent):
        ],
        # JavaScript class inheritance
        "javascript": [
            r"^\s*class\s+(\w+)\s+extends\s+(\w+)",  # class Child extends Parent
        ],
        # Java class inheritance
        "java": [
            r"^\s*(?:public|private|protected)?\s*(?:abstract|final)?\s*class\s+(\w+)\s+extends\s+(\w+)",
            # class Child extends Parent
            r"^\s*(?:public|private|protected)?\s*(?:abstract|final)?\s*class\s+(\w+).*implements\s+([a-zA-Z0-9_,\s]+)",
            # class Name implements Interface
        ],
        # C++ class inheritance
        "c": [
            r"^\s*class\s+(\w+)\s*:\s*(?:public|private|protected)?\s*(\w+)",  # class Child : public Parent
        ],
    }

    def __init__(self, repository: Repository):
        """
        Initialize the structure mapper.

        Args:
            repository: Repository object to analyze
        """
        self.repository = repository
        self.dependency_graph = nx.DiGraph()
        self.import_map: Dict[str, Set[str]] = defaultdict(set)
        self.inheritance_map: Dict[str, List[str]] = defaultdict(list)
        self.module_map: Dict[str, Set[str]] = defaultdict(set)
        self.centrality_scores: Dict[str, float] = {}

    def map_structure(self, file_limit: Optional[int] = None) -> nx.DiGraph:
        """
        Map the structural relationships between files in the repository.

        Args:
            file_limit: Optional limit on the number of files to analyze

        Returns:
            NetworkX DiGraph representing file dependencies
        """
        logger.info("Mapping codebase structure and dependencies")

        # Start with a fresh graph
        self.dependency_graph = nx.DiGraph()

        # Get all code files
        all_files = [f for f in self.repository.get_all_files()
                     if self.repository.is_code_file(f)]

        # Apply optional file limit
        if file_limit:
            all_files = all_files[:file_limit]

        # Build file mappings for module resolution
        self._build_file_mappings(all_files)

        # Process each file to extract dependencies
        for filename in tqdm(all_files, desc="Analyzing file dependencies"):
            self._analyze_file_dependencies(filename)

        # Calculate various centrality measures
        self._calculate_centrality_measures()

        logger.info(f"Mapped structure with {len(self.dependency_graph.nodes())} nodes "
                    f"and {len(self.dependency_graph.edges())} edges")

        return self.dependency_graph

    def get_central_files(self, limit: int = 10, method: str = "betweenness") -> List[Tuple[str, float]]:
        """
        Get the most central files in the codebase structure.

        Args:
            limit: Maximum number of files to return
            method: Centrality measure to use ('betweenness', 'degree', 'closeness', 'eigenvector')

        Returns:
            List of (filename, centrality_score) tuples for the most central files
        """
        if not self.centrality_scores:
            self._calculate_centrality_measures()

        # Filter scores based on requested method
        method_key = f"{method}_centrality"
        scores = [(node, data.get(method_key, 0))
                  for node, data in self.dependency_graph.nodes(data=True)]

        # Sort by centrality score (descending)
        return sorted(scores, key=lambda x: x[1], reverse=True)[:limit]

    def get_dependencies(
            self,
            filename: str,
            recursive: bool = False,
            depth: int = 1
    ) -> List[str]:
        """
        Get dependencies for a specific file.

        Args:
            filename: Path to the file
            recursive: Whether to get dependencies recursively
            depth: Maximum recursion depth if recursive is True

        Returns:
            List of dependencies
        """
        # Normalize the filename
        filename = self.repository.normalize_path(filename)

        # Check if file exists in the graph
        if filename not in self.dependency_graph:
            return []

        if not recursive or depth <= 0:
            # Get direct dependencies
            return list(self.dependency_graph.successors(filename))
        else:
            # Get recursive dependencies
            deps = set()
            self._get_recursive_dependencies(filename, deps, depth)
            return list(deps)

    def get_dependents(
            self,
            filename: str,
            recursive: bool = False,
            depth: int = 1
    ) -> List[str]:
        """
        Get files that depend on a specific file.

        Args:
            filename: Path to the file
            recursive: Whether to get dependents recursively
            depth: Maximum recursion depth if recursive is True

        Returns:
            List of dependent files
        """
        # Normalize the filename
        filename = self.repository.normalize_path(filename)

        # Check if file exists in the graph
        if filename not in self.dependency_graph:
            return []

        if not recursive or depth <= 0:
            # Get direct dependents
            return list(self.dependency_graph.predecessors(filename))
        else:
            # Get recursive dependents
            deps = set()
            self._get_recursive_dependents(filename, deps, depth)
            return list(deps)

    def detect_cycles(self) -> List[List[str]]:
        """
        Detect dependency cycles in the codebase.

        Returns:
            List of cycles, where each cycle is a list of filenames
        """
        cycles = list(nx.simple_cycles(self.dependency_graph))

        # Sort cycles by length (shorter cycles first)
        return sorted(cycles, key=len)

    def get_critical_paths(self) -> List[List[str]]:
        """
        Get critical dependency paths in the codebase.

        These are paths that, if broken, would disconnect many parts of the system.

        Returns:
            List of critical paths, where each path is a list of filenames
        """
        # Look for longest paths in the graph
        paths = []

        # Use longest path in directed acyclic subgraphs
        try:
            # Make a copy of the graph and remove cycles
            acyclic = nx.DiGraph(self.dependency_graph)

            # Remove edges that create cycles
            cycles = list(nx.simple_cycles(acyclic))
            for cycle in cycles:
                if len(cycle) >= 2:
                    acyclic.remove_edge(cycle[-1], cycle[0])

            # Find all simple paths between entry points and leaf nodes
            entry_points = [n for n in acyclic.nodes() if acyclic.in_degree(n) == 0]
            leaf_nodes = [n for n in acyclic.nodes() if acyclic.out_degree(n) == 0]

            # Limit the search to avoid excessive computation
            for source in entry_points[:5]:  # Limit to first 5 entry points
                for target in leaf_nodes[:5]:  # Limit to first 5 leaf nodes
                    # Limit path length to 10 nodes
                    for path in nx.all_simple_paths(acyclic, source, target, cutoff=10):
                        paths.append(path)

            # Sort paths by length (longest paths first)
            paths.sort(key=len, reverse=True)

        except Exception as e:
            logger.warning(f"Error finding critical paths: {str(e)}")

        return paths[:10]  # Return top 10 paths

    def visualize_structure(
            self,
            output_path: str = "dependency_graph.png",
            max_nodes: int = 100
    ) -> None:
        """
        Visualize the codebase structure.

        Args:
            output_path: Path to save the visualization
            max_nodes: Maximum number of nodes to include in visualization
        """
        try:
            import matplotlib.pyplot as plt

            # Create a subgraph with the most important nodes
            if len(self.dependency_graph) > max_nodes:
                # Use eigenvector centrality to select important nodes
                central_nodes = [n for n, _ in self.get_central_files(max_nodes, "eigenvector")]
                subgraph = self.dependency_graph.subgraph(central_nodes)
            else:
                subgraph = self.dependency_graph

            # Set up the figure
            plt.figure(figsize=(12, 10))

            # Define node colors based on file types
            node_colors = []
            for node in subgraph.nodes():
                if node.endswith('.py'):
                    node_colors.append('skyblue')
                elif node.endswith(('.js', '.ts')):
                    node_colors.append('yellow')
                elif node.endswith(('.java', '.kt')):
                    node_colors.append('orange')
                elif node.endswith(('.c', '.cpp', '.h')):
                    node_colors.append('green')
                else:
                    node_colors.append('gray')

            # Calculate node sizes based on centrality
            node_sizes = []
            for node in subgraph.nodes():
                # Use degree centrality for node size
                centrality = subgraph.nodes[node].get('degree_centrality', 0.1)
                node_sizes.append(300 * (centrality + 0.1))

            # Create layout
            pos = nx.spring_layout(subgraph, k=0.15, iterations=50, seed=42)

            # Draw the graph
            nx.draw_networkx(
                subgraph,
                pos=pos,
                with_labels=True,
                node_color=node_colors,
                node_size=node_sizes,
                font_size=8,
                alpha=0.8,
                width=0.5,
                edge_color='gray',
                arrows=True,
                arrowsize=10
            )

            plt.title('Codebase Structure and Dependencies', fontsize=16)
            plt.axis('off')

            # Save figure
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()

            logger.info(f"Structure visualization saved to {output_path}")

        except ImportError:
            logger.warning("Could not create visualization. Required package: matplotlib")

    def _build_file_mappings(self, files: List[str]) -> None:
        """
        Build mappings between module names and files.

        Args:
            files: List of files to process
        """
        # This mapping helps resolve imports to actual files
        for filename in files:
            # Extract possible module paths
            module_path = self._file_to_module_path(filename)
            if module_path:
                self.module_map[module_path].add(filename)

                # Add parent modules as well
                parts = module_path.split('.')
                for i in range(1, len(parts)):
                    parent = '.'.join(parts[:-i])
                    if parent:
                        self.module_map[parent].add(filename)

    def _file_to_module_path(self, filename: str) -> str:
        """
        Convert a file path to a potential module import path.

        Args:
            filename: Path to the file

        Returns:
            Potential module import path
        """
        # Handle Python files
        if filename.endswith('.py'):
            # Remove extension and convert path separators to dots
            module_path = filename[:-3].replace('/', '.')
            # Handle __init__.py files
            if module_path.endswith('.__init__'):
                module_path = module_path[:-9]
            return module_path

        # Handle JavaScript/TypeScript files
        elif filename.endswith(('.js', '.jsx', '.ts', '.tsx')):
            # For JS files, use path without extension
            base = os.path.splitext(filename)[0]
            return base.replace('/', '.')

        # Handle Java files
        elif filename.endswith('.java'):
            # Extract package from file content
            content = self.repository.get_file_content(filename)
            if content:
                package_match = re.search(r'^\s*package\s+([a-zA-Z0-9_.]+);', content, re.MULTILINE)
                if package_match:
                    package = package_match.group(1)
                    class_name = os.path.splitext(os.path.basename(filename))[0]
                    return f"{package}.{class_name}"

            # Fallback to path-based module name
            base = os.path.splitext(filename)[0]
            return base.replace('/', '.')

        # Default: use normalized path without extension
        base = os.path.splitext(filename)[0]
        return base.replace('/', '.')

    def _analyze_file_dependencies(self, filename: str) -> None:
        """
        Analyze a file to extract its dependencies.

        Args:
            filename: Path to the file
        """
        # Add the file to the graph
        if filename not in self.dependency_graph:
            self.dependency_graph.add_node(filename, type="file")

        # Skip further analysis if file is too large
        file_size = self.repository.get_file_size(filename)
        if file_size > 1000000:  # Skip files larger than 1MB
            logger.debug(f"Skipping large file: {filename} ({file_size} bytes)")
            return

        # Get file content
        content = self.repository.get_file_content(filename)
        if not content:
            logger.debug(f"Could not read file content: {filename}")
            return

        # Determine file language
        language = self._detect_language(filename)
        if not language:
            return

        # Extract imports
        imports = self._extract_imports(content, language)
        for imp in imports:
            self._resolve_and_add_dependency(filename, imp, language)

        # Extract class inheritance
        inheritance = self._extract_inheritance(content, language)
        for cls, parents in inheritance.items():
            self.inheritance_map[cls].extend(parents)

            # Also add class dependencies to the graph if we can resolve them
            for parent in parents:
                parent_files = self._resolve_symbol_to_files(parent, language)
                for parent_file in parent_files:
                    if parent_file != filename:  # Avoid self-dependencies
                        self.dependency_graph.add_edge(filename, parent_file, type="inheritance")

    def _detect_language(self, filename: str) -> Optional[str]:
        """
        Detect the programming language of a file based on its extension.

        Args:
            filename: Path to the file

        Returns:
            Detected language or None if not recognized
        """
        extension = os.path.splitext(filename)[1].lower()

        if extension in ['.py']:
            return 'python'
        elif extension in ['.js', '.jsx', '.ts', '.tsx']:
            return 'javascript'
        elif extension in ['.java']:
            return 'java'
        elif extension in ['.c', '.cpp', '.cc', '.h', '.hpp']:
            return 'c'
        elif extension in ['.go']:
            return 'go'
        elif extension in ['.rb']:
            return 'ruby'
        elif extension in ['.php']:
            return 'php'

        # Default to None for unsupported file types
        return None

    def _extract_imports(self, content: str, language: str) -> Set[str]:
        """
        Extract imports from file content based on language.

        Args:
            content: File content
            language: Programming language

        Returns:
            Set of imported module names
        """
        imports = set()

        # Skip if language not supported or no patterns defined
        if language not in self.IMPORT_PATTERNS:
            return imports

        # Get patterns for the language
        patterns = self.IMPORT_PATTERNS[language]

        # Process each line of content
        for line in content.split('\n'):
            # Try each import pattern
            for pattern in patterns:
                matches = re.findall(pattern, line)
                if matches:
                    for match in matches:
                        # Handle tuple matches (multiple capture groups)
                        if isinstance(match, tuple):
                            match = match[0]  # Use first capture group

                        # Clean up and normalize import
                        cleaned = match.strip()

                        # Handle comma-separated imports (Python's "import x, y, z")
                        if language == "python" and "," in cleaned:
                            for submodule in cleaned.split(","):
                                submodule = submodule.strip()
                                if submodule:
                                    imports.add(submodule)
                        else:
                            imports.add(cleaned)

        return imports

    def _analyze_file_dependencies(self, filename: str) -> None:
        """
        Analyze a file to extract its dependencies.

        Args:
            filename: Path to the file
        """
        # Add the file to the graph
        if filename not in self.dependency_graph:
            self.dependency_graph.add_node(filename, type="file")

        # Skip further analysis if file is too large
        file_size = self.repository.get_file_size(filename)
        if file_size > 1_000_000:  # Skip files larger than 1MB
            logger.debug(f"Skipping large file: {filename} ({file_size} bytes)")
            return

        # Get file content
        content = self.repository.get_file_content(filename)
        if not content:
            return

        # Detect language
        language = self._detect_language(filename)
        if not language:
            return

        # Extract imports
        imports = self._extract_imports(content, language)

        # Resolve imports to files
        for import_name in imports:
            resolved_files = self._resolve_import_to_file(import_name, filename, language)
            for resolved_file in resolved_files:
                self.dependency_graph.add_edge(filename, resolved_file)
                self.import_map[filename].add(resolved_file)

        # Extract inheritance relationships
        self._extract_inheritance(content, language, filename)

    def _extract_imports(self, content: str, language: str) -> Set[str]:
        """
        Extract imports from file content based on language.

        Args:
            content: File content
            language: Programming language

        Returns:
            Set of imported module names
        """
        imports = set()

        # Skip if language not supported or no patterns defined
        if language not in self.IMPORT_PATTERNS:
            return imports

        # Get patterns for the language
        patterns = self.IMPORT_PATTERNS[language]

        # Process each line of content
        for line in content.split('\n'):
            # Try each import pattern
            for pattern in patterns:
                matches = re.findall(pattern, line)
                if matches:
                    for match in matches:
                        # Handle tuple matches (multiple capture groups)
                        if isinstance(match, tuple):
                            match = match[0]  # Use first capture group

                        # Clean up and normalize import
                        cleaned = match.strip()

                        # Handle comma-separated imports (Python's "import x, y, z")
                        if language == "python" and "," in cleaned:
                            for submodule in cleaned.split(","):
                                submodule = submodule.strip()
                                if submodule:
                                    imports.add(submodule)
                        else:
                            imports.add(cleaned)

        return imports

    def _extract_inheritance(self, content: str, language: str, filename: str) -> None:
        """
        Extract class inheritance relationships from file content.

        Args:
            content: File content
            language: Programming language
            filename: Current file being analyzed
        """
        # Skip if language not supported
        if language not in self.INHERITANCE_PATTERNS:
            return

        # Get patterns for the language
        patterns = self.INHERITANCE_PATTERNS[language]

        # Extract class definitions and their parent classes
        for line in content.split('\n'):
            for pattern in patterns:
                match = re.search(pattern, line)
                if match:
                    # Extract class name and parent(s)
                    class_name = match.group(1).strip()
                    parent_classes = match.group(2).strip()

                    # Language-specific parsing of parent classes
                    if language == "python":
                        # Handle multiple inheritance
                        parents = [p.strip() for p in parent_classes.split(',')]
                    elif language == "java" or language == "javascript":
                        # Java/JavaScript single inheritance
                        parents = [parent_classes]
                    elif language == "c":
                        # C++ can have multiple inheritance but need more complex parsing
                        # This is simplified for now
                        parents = [parent_classes]
                    else:
                        parents = [parent_classes]

                    # Store inheritance relationships
                    for parent in parents:
                        if parent and parent.lower() not in ("object", "none"):
                            self.inheritance_map[class_name].append(parent)

                            # Try to resolve the parent class to a file
                            parent_files = self._resolve_class_to_file(parent, language)
                            if parent_files and filename not in parent_files:
                                for parent_file in parent_files:
                                    # Add inheritance dependency to graph
                                    self.dependency_graph.add_edge(filename, parent_file,
                                                                   type="inheritance")

    def _resolve_import_to_file(
            self,
            import_name: str,
            current_file: str,
            language: str
    ) -> Set[str]:
        """
        Resolve an import name to actual files in the repository.

        Args:
            import_name: Name of the imported module
            current_file: Path to the file containing the import
            language: Programming language

        Returns:
            Set of files that match the import
        """
        resolved_files = set()

        # Handle language-specific import resolution
        if language == "python":
            # Python relative imports
            if import_name.startswith('.'):
                # Resolve relative import based on current file's package
                current_dir = os.path.dirname(current_file)
                rel_levels = 0

                # Count how many dots for relative import
                while import_name.startswith('.'):
                    import_name = import_name[1:]
                    rel_levels += 1

                # Go up directories based on relative levels
                package_path = current_dir
                for _ in range(rel_levels - 1):
                    package_path = os.path.dirname(package_path)

                # Construct potential target path
                if import_name:
                    target_path = os.path.join(package_path, import_name.replace('.', '/'))
                else:
                    target_path = package_path

                # Check for .py file or __init__.py in directory
                py_file = f"{target_path}.py"
                init_file = os.path.join(target_path, "__init__.py")

                if self.repository.file_exists(py_file):
                    resolved_files.add(py_file)
                if self.repository.file_exists(init_file):
                    resolved_files.add(init_file)
            else:
                # Absolute imports - check the module map
                resolved_files.update(self.module_map.get(import_name, set()))

                # Try with added .py extension
                py_import = f"{import_name.replace('.', '/')}.py"
                if self.repository.file_exists(py_import):
                    resolved_files.add(py_import)

                # Try as directory with __init__.py
                init_path = f"{import_name.replace('.', '/')}/__init__.py"
                if self.repository.file_exists(init_path):
                    resolved_files.add(init_path)

        elif language in ["javascript", "typescript"]:
            # Handle JavaScript/TypeScript imports
            # Strip quotes if present
            clean_import = import_name.strip('\'"')

            # Handle relative imports
            if clean_import.startswith('./') or clean_import.startswith('../'):
                base_dir = os.path.dirname(current_file)
                abs_path = os.path.normpath(os.path.join(base_dir, clean_import))

                # Check various extensions
                for ext in ['.js', '.jsx', '.ts', '.tsx', '/index.js', '/index.ts']:
                    candidate = f"{abs_path}{ext}"
                    if self.repository.file_exists(candidate):
                        resolved_files.add(candidate)
            else:
                # Non-relative imports, try various common patterns
                # This is simplified - a real implementation would check node_modules, etc.
                for base_dir in ['src', 'lib', 'app', '']:
                    for ext in ['.js', '.jsx', '.ts', '.tsx', '/index.js', '/index.ts']:
                        candidate = f"{base_dir}/{clean_import}{ext}"
                        if self.repository.file_exists(candidate):
                            resolved_files.add(candidate)

        elif language == "java":
            # Java imports
            # Convert package dots to slashes and add .java
            java_path = f"{import_name.replace('.', '/')}.java"
            if self.repository.file_exists(java_path):
                resolved_files.add(java_path)

            # Check for wildcard imports (e.g., import java.util.*)
            if import_name.endswith("*"):
                base_package = import_name[:-2]  # Remove the .*
                package_dir = base_package.replace('.', '/')
                # Find all Java files in this package
                package_files = [f for f in self.repository.list_directory(package_dir)
                                 if f.endswith('.java')]
                resolved_files.update(package_files)

        # For other languages, use a more generic approach
        if not resolved_files:
            # Try direct file match (for C/C++ includes, etc.)
            if self.repository.file_exists(import_name):
                resolved_files.add(import_name)

        return resolved_files

    def _resolve_class_to_file(self, class_name: str, language: str) -> Set[str]:
        """
        Try to resolve a class name to the files where it might be defined.

        Args:
            class_name: Name of the class
            language: Programming language

        Returns:
            Set of files that might contain the class definition
        """
        # This is a simplified implementation that could be improved
        # based on language-specific class resolution rules
        files = set()

        # For most languages, the class name often matches the file name
        if language == "python":
            # Check for files named like the class
            potential_files = [f for f in self.repository.get_all_files()
                               if f.endswith('.py') and
                               os.path.basename(f).lower() == f"{class_name.lower()}.py"]
            files.update(potential_files)

        elif language in ["java", "kotlin"]:
            # In Java/Kotlin, class typically matches filename
            potential_files = [f for f in self.repository.get_all_files()
                               if f.endswith('.java') or f.endswith('.kt') and
                               os.path.basename(f).lower() == f"{class_name.lower()}.java" or
                               os.path.basename(f).lower() == f"{class_name.lower()}.kt"]
            files.update(potential_files)

        elif language in ["javascript", "typescript"]:
            # Check for JS/TS files with class name
            for ext in ['.js', '.jsx', '.ts', '.tsx']:
                potential_files = [f for f in self.repository.get_all_files()
                                   if f.endswith(ext) and
                                   os.path.basename(f).lower() == f"{class_name.lower()}{ext}"]
                files.update(potential_files)

        # This approach has limitations - in a real implementation,
        # we might need to parse all files and build a class->file index

        return files

    def _detect_language(self, filename: str) -> Optional[str]:
        """
        Detect the programming language based on file extension.

        Args:
            filename: Path to the file

        Returns:
            Language identifier or None if not recognized
        """
        extension = os.path.splitext(filename)[1].lower()

        if extension in ['.py']:
            return "python"
        elif extension in ['.js', '.jsx', '.ts', '.tsx']:
            return "javascript"
        elif extension in ['.java']:
            return "java"
        elif extension in ['.c', '.cpp', '.h', '.hpp', '.cc']:
            return "c"
        elif extension in ['.go']:
            return "go"
        elif extension in ['.rb']:
            return "ruby"
        elif extension in ['.php']:
            return "php"

        return None

    def _get_recursive_dependencies(
            self,
            filename: str,
            visited: Set[str],
            depth: int = -1
    ) -> None:
        """
        Recursively get dependencies for a file.

        Args:
            filename: Path to the file
            visited: Set of already visited files to avoid cycles
            depth: Maximum recursion depth (-1 for unlimited)
        """
        if filename in visited or (depth == 0):
            return

        visited.add(filename)

        # Get direct dependencies
        dependencies = list(self.dependency_graph.successors(filename))

        # Recursively get dependencies of dependencies
        if depth != 1:  # Continue recursion if not at max depth
            for dependency in dependencies:
                self._get_recursive_dependencies(dependency, visited, depth - 1)

    def _get_recursive_dependents(
            self,
            filename: str,
            visited: Set[str],
            depth: int = -1
    ) -> None:
        """
        Recursively get dependents for a file.

        Args:
            filename: Path to the file
            visited: Set of already visited files to avoid cycles
            depth: Maximum recursion depth (-1 for unlimited)
        """
        if filename in visited or (depth == 0):
            return

        visited.add(filename)

        # Get direct dependents
        dependents = list(self.dependency_graph.predecessors(filename))

        # Recursively get dependents of dependents
        if depth != 1:  # Continue recursion if not at max depth
            for dependent in dependents:
                self._get_recursive_dependents(dependent, visited, depth - 1)

    def _calculate_centrality_measures(self) -> None:
        """
        Calculate various centrality measures for files in the dependency graph.

        This helps identify the most important files in the codebase structure.
        """
        logger.info("Calculating centrality measures for files")

        try:
            # Skip if graph is empty
            if len(self.dependency_graph) == 0:
                return

            # Calculate degree centrality (most connected files)
            degree_centrality = nx.degree_centrality(self.dependency_graph)
            nx.set_node_attributes(self.dependency_graph, degree_centrality, "degree_centrality")

            # Try to calculate betweenness centrality (files that bridge different parts)
            try:
                betweenness_centrality = nx.betweenness_centrality(self.dependency_graph)
                nx.set_node_attributes(self.dependency_graph, betweenness_centrality, "betweenness_centrality")
            except Exception as e:
                logger.warning(f"Could not calculate betweenness centrality: {str(e)}")
                # Set default values
                default_betweenness = {node: 0.0 for node in self.dependency_graph.nodes()}
                nx.set_node_attributes(self.dependency_graph, default_betweenness, "betweenness_centrality")

            # Calculate closeness centrality (files central to the structure)
            try:
                # Use the largest connected component for closeness
                largest_cc = max(nx.weakly_connected_components(self.dependency_graph), key=len)
                subgraph = self.dependency_graph.subgraph(largest_cc)
                closeness_centrality = nx.closeness_centrality(subgraph)

                # Set centrality for nodes in the subgraph
                closeness_dict = {node: 0.0 for node in self.dependency_graph.nodes()}
                closeness_dict.update(closeness_centrality)
                nx.set_node_attributes(self.dependency_graph, closeness_dict, "closeness_centrality")
            except Exception as e:
                logger.warning(f"Could not calculate closeness centrality: {str(e)}")
                # Set default values
                default_closeness = {node: 0.0 for node in self.dependency_graph.nodes()}
                nx.set_node_attributes(self.dependency_graph, default_closeness, "closeness_centrality")

            # Calculate eigenvector centrality (files connected to important files)
            try:
                eigenvector_centrality = nx.eigenvector_centrality_numpy(self.dependency_graph)
                nx.set_node_attributes(self.dependency_graph, eigenvector_centrality, "eigenvector_centrality")
            except Exception as e:
                logger.warning(f"Could not calculate eigenvector centrality: {str(e)}")
                # Set default values
                default_eigenvector = {node: 0.0 for node in self.dependency_graph.nodes()}
                nx.set_node_attributes(self.dependency_graph, default_eigenvector, "eigenvector_centrality")

            logger.info("Finished calculating centrality measures")

        except Exception as e:
            logger.error(f"Error calculating centrality measures: {str(e)}")