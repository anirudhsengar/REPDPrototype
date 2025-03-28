#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
REPD (Repository Engineering and Project Dynamics) Model

This is the main model class that orchestrates the analysis of repositories
using various components: structure mapping, risk calculation, entry point
detection, and change coupling analysis.

Author: anirudhsengar
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import networkx as nx

from repd.change_coupling_analyzer import ChangeCouplingAnalyzer
from repd.entry_point_detector import EntryPointDetector
from repd.repository import Repository
from repd.risk_calculator import RiskCalculator
from repd.structure_mapper import StructureMapper
from repd.visualization import visualize_results

logger = logging.getLogger(__name__)


class REPDModel:
    """
    Main model class for the REPD (Repository Engineering and Project Dynamics) system.

    This class orchestrates the analysis of repositories to identify risk patterns,
    structural dependencies, and project dynamics.
    """

    def __init__(self, repository: Repository):
        """
        Initialize the REPD model with a repository.

        Args:
            repository: Repository interface to analyze
        """
        self.repository = repository

        # Initialize components
        self.structure_mapper = StructureMapper(repository)

        # These components will be initialized after structure mapping
        self.risk_calculator = None
        self.entry_point_detector = None
        self.change_coupling_analyzer = None

        # Configuration parameters
        self.config = {
            "max_files": None,  # Maximum number of files to analyze (None for all)
            "history_days": 90,  # Days of history to consider
            "risk_weights": {  # Weights for risk calculation
                "complexity": 0.25,
                "churn": 0.25,
                "coupling": 0.2,
                "structural": 0.2,
                "age": 0.1,
            },
            "coupling_threshold": 0.5,  # Threshold for change coupling
            "entry_point_min_score": 0.6,  # Minimum score for entry points
            "exclude_patterns": [  # Patterns to exclude from analysis
                ".git/",
                "venv/",
                "env/",
                "node_modules/",
                "__pycache__/",
                "*.pyc",
                "*.pyo",
                "*.pyd",
                "*.so",
                "*.o",
                "*.a",
                "*.lib",
                "*.dll",
            ],
        }

        # Results storage
        self.results = {}

    def configure(
        self,
        max_files: int = None,
        history_days: int = None,
        risk_weights: Dict[str, float] = None,
        coupling_threshold: float = None,
        entry_point_min_score: float = None,
        exclude_patterns: List[str] = None,
    ) -> None:
        """
        Configure analysis parameters.

        Args:
            max_files: Maximum number of files to analyze
            history_days: Days of history to consider
            risk_weights: Weights for different risk factors
            coupling_threshold: Threshold for change coupling
            entry_point_min_score: Minimum score for entry points
            exclude_patterns: Patterns to exclude from analysis
        """
        logger.info("Configuring REPD model")

        # Update configuration parameters if specified
        if max_files is not None:
            self.config["max_files"] = max_files

        if history_days is not None:
            self.config["history_days"] = history_days

        if risk_weights is not None:
            self.config["risk_weights"].update(risk_weights)

        if coupling_threshold is not None:
            self.config["coupling_threshold"] = coupling_threshold

        if entry_point_min_score is not None:
            self.config["entry_point_min_score"] = entry_point_min_score

        if exclude_patterns is not None:
            self.config["exclude_patterns"] = exclude_patterns

    def analyze_structure(self) -> Dict[str, Any]:
        """
        Analyze the structure of the repository.

        This maps the dependency structure of the repository and initializes
        other analysis components.

        Returns:
            Dictionary with structure analysis results
        """
        logger.info("Starting structure analysis")

        # Map dependency structure
        dependency_graph = self.structure_mapper.map_structure(
            max_files=self.config["max_files"]
        )

        # Calculate centrality measures
        central_files = self.structure_mapper.get_central_files()

        # Initialize other components now that we have structure information
        self.risk_calculator = RiskCalculator(self.repository, self.structure_mapper)
        self.entry_point_detector = EntryPointDetector(
            self.repository, self.structure_mapper
        )
        self.change_coupling_analyzer = ChangeCouplingAnalyzer(self.repository)

        # Store results
        structure_results = {
            "dependency_graph": dependency_graph,
            "central_files": dict(central_files),
            "file_count": len(dependency_graph.nodes()),
            "edge_count": len(dependency_graph.edges()),
            "analysis_timestamp": datetime.now().isoformat(),
        }

        self.results["structure"] = structure_results

        # Detect entry points
        entry_points = self.entry_point_detector.detect_entry_points()
        self.results["entry_points"] = entry_points

        # Analyze change coupling
        coupling_matrix = self.change_coupling_analyzer.analyze_coupling(
            days=self.config["history_days"]
        )
        self.results["coupling_matrix"] = coupling_matrix

        logger.info("Structure analysis completed")
        return structure_results

    def calculate_risk_scores(self) -> Dict[str, float]:
        """
        Calculate risk scores for files in the repository.

        Returns:
            Dictionary mapping file paths to risk scores
        """
        logger.info("Calculating risk scores")

        # Ensure structure has been analyzed first
        if not self.risk_calculator:
            logger.warning("Structure analysis not performed, running it now")
            self.analyze_structure()

        # Calculate risk scores using configured weights
        risk_scores = self.risk_calculator.calculate_risk_scores(
            weights=self.config["risk_weights"]
        )

        # Store results
        self.results["risk_scores"] = risk_scores
        self.results["risk_factors"] = self.risk_calculator.risk_factors

        # Categorize files by risk level
        risk_categories = self.risk_calculator.classify_risk()
        self.results["risk_categories"] = risk_categories

        # Find highest risk factors for each file
        highest_factors = self.risk_calculator.get_highest_risk_factors()
        self.results["highest_risk_factors"] = highest_factors

        logger.info(
            f"Risk calculation complete. Found {len(risk_categories['high_risk'])} "
            f"high-risk files out of {len(risk_scores)} total."
        )

        return risk_scores

    def analyze_entry_points(self) -> Dict[str, float]:
        """
        Analyze entry points to the codebase.

        Returns:
            Dictionary mapping file paths to entry point scores
        """
        logger.info("Analyzing entry points")

        # Ensure structure has been analyzed first
        if not self.entry_point_detector:
            logger.warning("Structure analysis not performed, running it now")
            self.analyze_structure()

        # Analyze entry points if not already done
        if "entry_points" not in self.results:
            entry_points = self.entry_point_detector.detect_entry_points()
            self.results["entry_points"] = entry_points

        # Get significant entry points
        significant_entry_points = {
            path: score
            for path, score in self.results["entry_points"].items()
            if score >= self.config["entry_point_min_score"]
        }

        self.results["significant_entry_points"] = significant_entry_points

        logger.info(
            f"Identified {len(significant_entry_points)} significant entry points"
        )

        return significant_entry_points

    def analyze_coupling(self) -> Dict[str, Dict[str, float]]:
        """
        Analyze change coupling between files.

        Returns:
            Dictionary with change coupling matrix
        """
        logger.info("Analyzing change coupling")

        # Ensure change coupling analyzer is initialized
        if not self.change_coupling_analyzer:
            logger.warning("Structure analysis not performed, running it now")
            self.analyze_structure()

        # Analyze coupling if not already done
        if "coupling_matrix" not in self.results:
            coupling_matrix = self.change_coupling_analyzer.analyze_coupling(
                days=self.config["history_days"]
            )
            self.results["coupling_matrix"] = coupling_matrix

        # Get high coupling pairs
        high_coupling_pairs = self.change_coupling_analyzer.get_high_coupling_pairs(
            threshold=self.config["coupling_threshold"]
        )
        self.results["high_coupling_pairs"] = high_coupling_pairs

        # Get coupled clusters
        coupled_clusters = self.change_coupling_analyzer.get_coupled_clusters(
            min_coupling=self.config["coupling_threshold"]
        )
        self.results["coupled_clusters"] = [
            list(cluster) for cluster in coupled_clusters
        ]

        logger.info(
            f"Identified {len(high_coupling_pairs)} high coupling pairs "
            f"and {len(coupled_clusters)} coupled clusters"
        )

        return self.results["coupling_matrix"]

    def analyze_all(self) -> Dict[str, Any]:
        """
        Perform all analyses on the repository.

        Returns:
            Dictionary with all analysis results
        """
        logger.info("Starting comprehensive repository analysis")

        # Run all analyses
        self.analyze_structure()
        self.calculate_risk_scores()
        self.analyze_entry_points()
        self.analyze_coupling()

        # Add metadata
        self.results["metadata"] = {
            "repository_name": self.repository.get_name(),
            "analysis_timestamp": datetime.now().isoformat(),
            "configuration": self.config,
        }

        logger.info("Comprehensive analysis completed")

        return self.results

    def save_results(self, output_path: Union[str, Path]) -> None:
        """
        Save analysis results to a JSON file.

        Args:
            output_path: Path to save results to
        """
        logger.info(f"Saving results to {output_path}")

        # Ensure output directory exists
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert Graph objects to serializable format
        serializable_results = self._prepare_for_serialization(self.results)

        # Add metadata if not already present
        if "metadata" not in serializable_results:
            serializable_results["metadata"] = {
                "repository_name": self.repository.get_name(),
                "analysis_timestamp": datetime.now().isoformat(),
                "configuration": self.config,
            }

        # Write to file
        with open(output_path, "w") as f:
            json.dump(serializable_results, f, indent=2)

        logger.info(f"Results saved to {output_path}")

    def visualize(
        self, output_dir: Union[str, Path], viz_types: List[str] = None
    ) -> Dict[str, str]:
        """
        Generate visualizations of analysis results.

        Args:
            output_dir: Directory to save visualizations to
            viz_types: Types of visualizations to generate (None for all)

        Returns:
            Dictionary mapping visualization types to output file paths
        """
        logger.info(f"Generating visualizations in {output_dir}")

        # Create output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate visualizations using the visualization module
        viz_files = visualize_results(self.results, output_dir, viz_types)

        logger.info(f"Generated {len(viz_files)} visualizations")

        return viz_files

    def identify_hotspots(self, threshold: float = 0.7) -> List[Dict[str, Any]]:
        """
        Identify high-risk hotspots in the codebase.

        Hotspots are files with high risk that are also central to the codebase
        or are entry points.

        Args:
            threshold: Risk score threshold for hotspots

        Returns:
            List of hotspot information dictionaries
        """
        logger.info(f"Identifying hotspots with threshold {threshold}")

        # Ensure we have risk scores
        if "risk_scores" not in self.results:
            logger.warning("Risk scores not calculated, running calculation now")
            self.calculate_risk_scores()

        risk_scores = self.results["risk_scores"]

        # Get centrality if available
        centrality = {}
        if "structure" in self.results and "central_files" in self.results["structure"]:
            centrality = self.results["structure"]["central_files"]
        elif hasattr(self.structure_mapper, "centrality_scores"):
            centrality = self.structure_mapper.centrality_scores

        # Get entry points if available
        entry_points = self.results.get("entry_points", {})

        # Identify hotspots
        hotspots = []

        for file, risk_score in risk_scores.items():
            if risk_score < threshold:
                continue

            # Create hotspot entry
            hotspot = {
                "file": file,
                "risk_score": risk_score,
                "is_central": centrality.get(file, 0) > 0.5,
                "centrality": centrality.get(file, 0),
                "is_entry_point": entry_points.get(file, 0) > 0.5,
                "entry_point_score": entry_points.get(file, 0),
            }

            # Add risk factors if available
            if "risk_factors" in self.results and file in self.results["risk_factors"]:
                hotspot["risk_factors"] = self.results["risk_factors"][file]

            # Add coupling information if available
            if (
                "coupling_matrix" in self.results
                and file in self.results["coupling_matrix"]
            ):
                coupled_files = self.change_coupling_analyzer.get_coupled_files(file)
                hotspot["coupled_files"] = len(coupled_files)

            hotspots.append(hotspot)

        # Sort hotspots by risk score
        hotspots.sort(key=lambda x: x["risk_score"], reverse=True)

        logger.info(f"Identified {len(hotspots)} hotspots")

        return hotspots

    def generate_report(
        self, output_path: Union[str, Path], template: str = "default"
    ) -> None:
        """
        Generate a human-readable report of analysis results.

        Args:
            output_path: Path to save the report to
            template: Report template to use
        """
        logger.info(f"Generating report using {template} template")

        # Ensure output directory exists
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Perform all analyses if not already done
        if not self.results:
            logger.warning("No analysis results found, running comprehensive analysis")
            self.analyze_all()

        # Generate HTML report
        if template == "default":
            self._generate_default_report(output_path)
        else:
            logger.warning(f"Unknown template: {template}, falling back to default")
            self._generate_default_report(output_path)

        logger.info(f"Report generated at {output_path}")

    def _generate_default_report(self, output_path: Path) -> None:
        """
        Generate default HTML report.

        Args:
            output_path: Path to save the report to
        """
        # Basic HTML report template
        html_start = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>REPD Analysis Report</title>
            <style>
                body { font-family: Arial, sans-serif; line-height: 1.6; margin: 30px; }
                h1 { color: #2c3e50; }
                h2 { color: #3498db; margin-top: 30px; }
                table { border-collapse: collapse; width: 100%; margin: 20px 0; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
                tr:nth-child(even) { background-color: #f9f9f9; }
                .risk-high { color: #e74c3c; }
                .risk-medium { color: #f39c12; }
                .risk-low { color: #27ae60; }
                .summary { background-color: #eef; padding: 15px; border-radius: 5px; }
            </style>
        </head>
        <body>
            <h1>REPD Analysis Report</h1>
        """

        repo_name = self.repository.get_name()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        html_summary = f"""
            <div class="summary">
                <h2>Summary</h2>
                <p><strong>Repository:</strong> {repo_name}</p>
                <p><strong>Analysis Date:</strong> {timestamp}</p>
        """

        # Add summary statistics
        if "risk_categories" in self.results:
            risk_categories = self.results["risk_categories"]
            html_summary += f"""
                <p><strong>High Risk Files:</strong> {len(risk_categories['high_risk'])}</p>
                <p><strong>Medium Risk Files:</strong> {len(risk_categories['medium_risk'])}</p>
                <p><strong>Low Risk Files:</strong> {len(risk_categories['low_risk'])}</p>
            """

        if "structure" in self.results:
            structure = self.results["structure"]
            html_summary += f"""
                <p><strong>Total Files:</strong> {structure.get('file_count', 'N/A')}</p>
                <p><strong>Dependencies:</strong> {structure.get('edge_count', 'N/A')}</p>
            """

        html_summary += """
            </div>
        """

        # Top high risk files
        html_risk = """
            <h2>Top High Risk Files</h2>
            <table>
                <tr>
                    <th>File</th>
                    <th>Risk Score</th>
                    <th>Highest Risk Factor</th>
                </tr>
        """

        # Add risk data if available
        if "risk_scores" in self.results and "highest_risk_factors" in self.results:
            risk_scores = sorted(
                self.results["risk_scores"].items(), key=lambda x: x[1], reverse=True
            )[
                :20
            ]  # Top 20

            highest_factors = self.results["highest_risk_factors"]

            for file, score in risk_scores:
                risk_class = (
                    "risk-high"
                    if score >= 0.7
                    else ("risk-medium" if score >= 0.4 else "risk-low")
                )

                factor = highest_factors.get(file, {}).get("factor", "unknown")
                factor_value = highest_factors.get(file, {}).get("value", 0)

                html_risk += f"""
                    <tr>
                        <td>{file}</td>
                        <td class="{risk_class}">{score:.2f}</td>
                        <td>{factor} ({factor_value:.2f})</td>
                    </tr>
                """

        html_risk += """
            </table>
        """

        # Entry points
        html_entry_points = """
            <h2>Key Entry Points</h2>
            <table>
                <tr>
                    <th>File</th>
                    <th>Confidence</th>
                </tr>
        """

        # Add entry point data if available
        if "entry_points" in self.results:
            entry_points = sorted(
                self.results["entry_points"].items(), key=lambda x: x[1], reverse=True
            )[
                :10
            ]  # Top 10

            for file, score in entry_points:
                if score < 0.5:
                    continue  # Skip low confidence entry points

                html_entry_points += f"""
                    <tr>
                        <td>{file}</td>
                        <td>{score:.2f}</td>
                    </tr>
                """

        html_entry_points += """
            </table>
        """

        # Coupled files
        html_coupled = """
            <h2>Highly Coupled Files</h2>
            <p>Files that frequently change together may indicate hidden dependencies.</p>
            <table>
                <tr>
                    <th>File 1</th>
                    <th>File 2</th>
                    <th>Coupling Strength</th>
                </tr>
        """

        # Add coupling data if available
        if "high_coupling_pairs" in self.results:
            high_coupling_pairs = self.results["high_coupling_pairs"][:15]  # Top 15

            for file1, file2, strength in high_coupling_pairs:
                html_coupled += f"""
                    <tr>
                        <td>{file1}</td>
                        <td>{file2}</td>
                        <td>{strength:.2f}</td>
                    </tr>
                """

        html_coupled += """
            </table>
        """

        # Central files
        html_central = """
            <h2>Central Files</h2>
            <p>Files with high centrality are architectural hotspots.</p>
            <table>
                <tr>
                    <th>File</th>
                    <th>Centrality</th>
                </tr>
        """

        # Add centrality data if available
        centrality = {}
        if "structure" in self.results and "central_files" in self.results["structure"]:
            centrality = self.results["structure"]["central_files"]
        elif hasattr(self.structure_mapper, "centrality_scores"):
            centrality = self.structure_mapper.centrality_scores

        if centrality:
            central_files = sorted(
                centrality.items(), key=lambda x: x[1], reverse=True
            )[
                :10
            ]  # Top 10

            for file, score in central_files:
                html_central += f"""
                    <tr>
                        <td>{file}</td>
                        <td>{score:.2f}</td>
                    </tr>
                """

        html_central += """
            </table>
        """

        # End of HTML
        html_end = """
        </body>
        </html>
        """

        # Combine all sections
        html_content = (
            html_start
            + html_summary
            + html_risk
            + html_entry_points
            + html_coupled
            + html_central
            + html_end
        )

        # Write to file
        with open(output_path, "w") as f:
            f.write(html_content)

    def _prepare_for_serialization(self, data: Any) -> Any:
        """
        Prepare data for JSON serialization by converting non-serializable objects.

        Args:
            data: Data to prepare

        Returns:
            Serializable version of the data
        """
        if isinstance(data, dict):
            return {k: self._prepare_for_serialization(v) for k, v in data.items()}

        elif isinstance(data, list):
            return [self._prepare_for_serialization(item) for item in data]

        elif isinstance(data, set):
            return list(data)

        elif isinstance(data, nx.Graph) or isinstance(data, nx.DiGraph):
            # Convert networkx graph to adjacency list
            return {
                "nodes": list(data.nodes()),
                "edges": [(u, v, dict(attrs)) for u, v, attrs in data.edges(data=True)],
            }

        elif isinstance(data, datetime):
            return data.isoformat()

        elif hasattr(data, "__dict__"):
            # Convert custom objects to dictionaries
            return {
                k: self._prepare_for_serialization(v)
                for k, v in data.__dict__.items()
                if not k.startswith("_")
            }

        else:
            return data
