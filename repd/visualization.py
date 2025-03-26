#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualization Module for REPD Model

This module provides visualization functionality for REPD analysis results,
including risk score visualizations, change coupling networks, entry point
maps, and developer activity patterns.

Author: anirudhsengar
Date: 2025-03-26 06:37:46
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any, Callable, Union

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
import seaborn as sns
from matplotlib.ticker import MaxNLocator
import networkx as nx
from matplotlib.figure import Figure
from matplotlib.axes import Axes

logger = logging.getLogger(__name__)


def visualize_results(
        results: Dict[str, Any],
        output_dir: Union[str, Path],
        viz_types: List[str] = None,
        file_format: str = "png",
        max_items: int = 20,
        figsize: Tuple[int, int] = (10, 8),
        dpi: int = 300,
        cmap: str = "viridis",
        progress_callback: Callable = None
) -> Dict[str, str]:
    """
    Generate visualizations from REPD analysis results.

    Args:
        results: Dictionary with REPD analysis results
        output_dir: Directory to save visualizations
        viz_types: List of visualization types to generate
                   (options: risk, coupling, entry_points, network, activity, treemap, heatmap)
        file_format: Output file format (png, pdf, svg)
        max_items: Maximum number of items to include in visualizations
        figsize: Figure size as (width, height) tuple
        dpi: DPI for output files
        cmap: Colormap name for visualizations
        progress_callback: Optional callback function to report progress

    Returns:
        Dictionary mapping visualization types to output file paths
    """
    logger.info(f"Generating visualizations in {output_dir}")

    # Create output directory if it doesn't exist
    if isinstance(output_dir, str):
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Default visualization types
    if viz_types is None:
        viz_types = ["risk", "coupling", "entry_points", "network", "activity"]

    # Set default style
    plt.style.use("seaborn-v0_8-darkgrid")

    # Validation
    if "risk_scores" not in results:
        logger.warning("No risk scores found in results, some visualizations may be skipped")

    # Track output files
    output_files = {}

    # Generate visualizations
    for viz_type in viz_types:
        try:
            if viz_type == "risk":
                output_file = _visualize_risk_scores(
                    results, output_dir, file_format, max_items, figsize, dpi, cmap)

            elif viz_type == "coupling":
                output_file = _visualize_change_coupling(
                    results, output_dir, file_format, max_items, figsize, dpi, cmap)

            elif viz_type == "entry_points":
                output_file = _visualize_entry_points(
                    results, output_dir, file_format, max_items, figsize, dpi, cmap)

            elif viz_type == "network":
                output_file = _visualize_dependency_network(
                    results, output_dir, file_format, max_items, figsize, dpi, cmap)

            elif viz_type == "activity":
                output_file = _visualize_developer_activity(
                    results, output_dir, file_format, max_items, figsize, dpi, cmap)

            elif viz_type == "treemap":
                output_file = _visualize_risk_treemap(
                    results, output_dir, file_format, max_items, figsize, dpi, cmap)

            elif viz_type == "heatmap":
                output_file = _visualize_risk_heatmap(
                    results, output_dir, file_format, max_items, figsize, dpi, cmap)

            else:
                logger.warning(f"Unknown visualization type: {viz_type}")
                output_file = None

            if output_file:
                output_files[viz_type] = output_file
                logger.info(f"Generated {viz_type} visualization: {output_file}")

            # Report progress if callback is provided
            if progress_callback:
                progress_callback()

        except Exception as e:
            logger.exception(f"Error generating {viz_type} visualization: {str(e)}")

    return output_files


def _visualize_risk_scores(
        results: Dict[str, Any],
        output_dir: Path,
        file_format: str,
        max_items: int,
        figsize: Tuple[int, int],
        dpi: int,
        cmap: str
) -> str:
    """
    Visualize risk scores as a horizontal bar chart.

    Args:
        results: Results dictionary
        output_dir: Output directory
        file_format: Output file format
        max_items: Maximum items to display
        figsize: Figure size
        dpi: DPI for output
        cmap: Colormap name

    Returns:
        Path to output file
    """
    # Extract risk scores
    risk_scores = results.get("risk_scores", {})

    if not risk_scores:
        logger.warning("No risk scores found, skipping risk score visualization")
        return None

    # Sort by risk score (descending)
    sorted_items = sorted(risk_scores.items(), key=lambda x: x[1], reverse=True)[:max_items]

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Extract data for plotting
    files = [_simplify_path(item[0]) for item in sorted_items]
    scores = [item[1] for item in sorted_items]

    # Create colormap for risk levels
    norm = mcolors.Normalize(vmin=0, vmax=1)
    colors = plt.cm.get_cmap(cmap)(norm(scores))

    # Plot horizontal bars
    y_pos = np.arange(len(files))
    bars = ax.barh(y_pos, scores, align='center', color=colors)

    # Add risk categories
    for i, score in enumerate(scores):
        category = _get_risk_category(score)
        ax.text(
            score + 0.01,  # Slight offset
            y_pos[i],
            category,
            va='center',
            fontsize=9,
            alpha=0.8
        )

    # Set labels and title
    ax.set_yticks(y_pos)
    ax.set_yticklabels(files)
    ax.invert_yaxis()  # Highest risk at the top
    ax.set_xlabel('Risk Score')
    ax.set_title('Top Risky Files')

    # Set x-axis limit slightly beyond 1 to make room for category labels
    ax.set_xlim(0, 1.2)

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=plt.cm.get_cmap(cmap), norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Risk Level')

    # Add grid lines
    ax.grid(axis='x', linestyle='--', alpha=0.7)

    # Adjust layout
    plt.tight_layout()

    # Save figure
    output_file = output_dir / f"risk_scores.{file_format}"
    plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
    plt.close(fig)

    return str(output_file)


def _visualize_change_coupling(
        results: Dict[str, Any],
        output_dir: Path,
        file_format: str,
        max_items: int,
        figsize: Tuple[int, int],
        dpi: int,
        cmap: str
) -> str:
    """
    Visualize change coupling as a network graph.

    Args:
        results: Results dictionary
        output_dir: Output directory
        file_format: Output file format
        max_items: Maximum items to display
        figsize: Figure size
        dpi: DPI for output
        cmap: Colormap name

    Returns:
        Path to output file
    """
    # Extract coupling matrix
    coupling_matrix = results.get("coupling_matrix", {})
    risk_scores = results.get("risk_scores", {})

    if not coupling_matrix:
        logger.warning("No coupling matrix found, skipping coupling visualization")
        return None

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Create network graph
    G = nx.Graph()

    # Get top files by risk score or by coupling degree
    if risk_scores:
        top_files = [file for file, _ in sorted(risk_scores.items(),
                                                key=lambda x: x[1],
                                                reverse=True)[:max_items]]
    else:
        # Use files with most coupling relationships
        file_coupling_counts = {}
        for file, couplings in coupling_matrix.items():
            file_coupling_counts[file] = len(couplings)

        top_files = [file for file, _ in sorted(file_coupling_counts.items(),
                                                key=lambda x: x[1],
                                                reverse=True)[:max_items]]

    # Add nodes and edges
    for file in top_files:
        # Add node with risk score if available
        node_attrs = {}
        if file in risk_scores:
            node_attrs["risk"] = risk_scores[file]

        G.add_node(file, **node_attrs)

        # Add edges for coupling relationships
        if file in coupling_matrix:
            for coupled_file, coupling_score in coupling_matrix[file].items():
                if coupled_file in top_files:
                    G.add_edge(file, coupled_file, weight=coupling_score)

    # Calculate node sizes based on risk or degree
    node_sizes = []
    node_colors = []

    for node in G.nodes:
        # Size based on risk score or degree centrality
        if "risk" in G.nodes[node]:
            size = 300 + 1000 * G.nodes[node]["risk"]
            color = G.nodes[node]["risk"]
        else:
            size = 300 + 100 * G.degree(node)
            color = 0.5  # Default color

        node_sizes.append(size)
        node_colors.append(color)

    # Calculate edge widths based on coupling strength
    edge_widths = [G[u][v]["weight"] * 5 for u, v in G.edges]

    # Simplify node labels
    node_labels = {node: _simplify_path(node) for node in G.nodes}

    # Create layout
    pos = nx.spring_layout(G, k=0.3, seed=42)

    # Draw network
    nx.draw_networkx_nodes(
        G, pos,
        ax=ax,
        node_size=node_sizes,
        node_color=node_colors,
        cmap=plt.cm.get_cmap(cmap),
        alpha=0.8
    )

    nx.draw_networkx_edges(
        G, pos,
        ax=ax,
        width=edge_widths,
        edge_color="gray",
        alpha=0.6
    )

    # Draw labels with appropriate font sizes
    # Adjust font size based on node size
    for node, (x, y) in pos.items():
        node_idx = list(G.nodes).index(node)
        size = node_sizes[node_idx]
        fontsize = min(12, max(8, size / 100))

        ax.text(
            x, y,
            node_labels[node],
            fontsize=fontsize,
            ha='center',
            va='center',
            bbox=dict(facecolor='white', alpha=0.7, boxstyle="round,pad=0.2")
        )

    # Add colorbar for risk scores
    if risk_scores:
        sm = plt.cm.ScalarMappable(cmap=plt.cm.get_cmap(cmap), norm=plt.Normalize(0, 1))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('Risk Score')

    # Set title and remove axes
    ax.set_title('Change Coupling Network')
    ax.axis('off')

    # Save figure
    output_file = output_dir / f"change_coupling_network.{file_format}"
    plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
    plt.close(fig)

    return str(output_file)


def _visualize_entry_points(
        results: Dict[str, Any],
        output_dir: Path,
        file_format: str,
        max_items: int,
        figsize: Tuple[int, int],
        dpi: int,
        cmap: str
) -> str:
    """
    Visualize entry points as a scatter plot.

    Args:
        results: Results dictionary
        output_dir: Output directory
        file_format: Output file format
        max_items: Maximum items to display
        figsize: Figure size
        dpi: DPI for output
        cmap: Colormap name

    Returns:
        Path to output file
    """
    # Extract entry points and risk scores
    entry_points = results.get("entry_points", {})
    risk_scores = results.get("risk_scores", {})

    if not entry_points:
        logger.warning("No entry points found, skipping entry point visualization")
        return None

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Sort entry points by score
    sorted_entry_points = sorted(entry_points.items(), key=lambda x: x[1], reverse=True)[:max_items]

    # Extract data
    files = [_simplify_path(item[0]) for item in sorted_entry_points]
    entry_scores = [item[1] for item in sorted_entry_points]

    # Get corresponding risk scores if available
    if risk_scores:
        risk_values = [risk_scores.get(item[0], 0) for item in sorted_entry_points]
    else:
        risk_values = [0.5] * len(files)  # Default value

    # Create scatter plot
    scatter = ax.scatter(
        entry_scores,
        np.arange(len(files)),
        c=risk_values,
        s=200,
        cmap=cmap,
        alpha=0.8,
        edgecolors='w'
    )

    # Add labels
    for i, file in enumerate(files):
        ax.text(
            entry_scores[i] + 0.02,
            i,
            file,
            va='center',
            fontsize=10
        )

    # Set labels and title
    ax.set_yticks([])  # Hide y-tick labels
    ax.set_xlabel('Entry Point Score')
    ax.set_title('Repository Entry Points')

    # Add colorbar for risk scores
    if any(score > 0 for score in risk_values):
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Risk Score')

    # Add grid lines
    ax.grid(axis='x', linestyle='--', alpha=0.7)

    # Adjust layout
    plt.tight_layout()

    # Save figure
    output_file = output_dir / f"entry_points.{file_format}"
    plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
    plt.close(fig)

    return str(output_file)


def _visualize_dependency_network(
        results: Dict[str, Any],
        output_dir: Path,
        file_format: str,
        max_items: int,
        figsize: Tuple[int, int],
        dpi: int,
        cmap: str
) -> str:
    """
    Visualize dependency network with entry points highlighted.

    Args:
        results: Results dictionary
        output_dir: Output directory
        file_format: Output file format
        max_items: Maximum items to display
        figsize: Figure size
        dpi: DPI for output
        cmap: Colormap name

    Returns:
        Path to output file
    """
    # Extract required data
    risk_scores = results.get("risk_scores", {})
    entry_points = results.get("entry_points", {})
    coupling_matrix = results.get("coupling_matrix", {})

    if not risk_scores or not coupling_matrix:
        logger.warning("Missing required data for dependency network visualization")
        return None

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Create network graph
    G = nx.DiGraph()

    # Select top risky files
    top_files = [file for file, _ in sorted(risk_scores.items(),
                                            key=lambda x: x[1],
                                            reverse=True)[:max_items]]

    # Add nodes
    for file in top_files:
        is_entry = file in entry_points
        entry_score = entry_points.get(file, 0)
        risk_score = risk_scores.get(file, 0)

        G.add_node(
            file,
            risk=risk_score,
            is_entry=is_entry,
            entry_score=entry_score
        )

    # Add edges
    for source in top_files:
        if source in coupling_matrix:
            for target, weight in coupling_matrix[source].items():
                if target in top_files:
                    G.add_edge(source, target, weight=weight)

    # Calculate node attributes
    node_sizes = []
    node_colors = []
    node_shapes = []

    for node in G.nodes:
        # Size based on risk and entry score
        if G.nodes[node]["is_entry"]:
            size = 300 + 1000 * G.nodes[node]["entry_score"]
            shape = "o"  # Circle for entry points
        else:
            size = 300 + 500 * G.nodes[node]["risk"]
            shape = "s"  # Square for non-entry points

        # Color based on risk score
        color = G.nodes[node]["risk"]

        node_sizes.append(size)
        node_colors.append(color)
        node_shapes.append(shape)

    # Calculate edge widths and colors
    edge_widths = []
    edge_colors = []

    for u, v in G.edges:
        weight = G[u][v]["weight"]
        target_risk = G.nodes[v]["risk"]

        edge_widths.append(weight * 3)
        edge_colors.append(target_risk)  # Color based on target risk

    # Create layout with entry points emphasized
    pos = nx.spring_layout(G, k=0.3, seed=42)

    # Draw network with different node shapes
    # Draw circles (entry points)
    circle_nodes = [node for node, shape in zip(G.nodes, node_shapes) if shape == "o"]
    circle_idx = [i for i, shape in enumerate(node_shapes) if shape == "o"]

    if circle_nodes:
        nx.draw_networkx_nodes(
            G, pos,
            ax=ax,
            nodelist=circle_nodes,
            node_size=[node_sizes[i] for i in circle_idx],
            node_color=[node_colors[i] for i in circle_idx],
            cmap=plt.cm.get_cmap(cmap),
            edgecolors='white',
            linewidths=2,
            alpha=0.9
        )

    # Draw squares (non-entry points)
    square_nodes = [node for node, shape in zip(G.nodes, node_shapes) if shape == "s"]
    square_idx = [i for i, shape in enumerate(node_shapes) if shape == "s"]

    if square_nodes:
        nx.draw_networkx_nodes(
            G, pos,
            ax=ax,
            nodelist=square_nodes,
            node_shape="s",
            node_size=[node_sizes[i] for i in square_idx],
            node_color=[node_colors[i] for i in square_idx],
            cmap=plt.cm.get_cmap(cmap),
            edgecolors='black',
            linewidths=1,
            alpha=0.8
        )

    # Draw edges
    nx.draw_networkx_edges(
        G, pos,
        ax=ax,
        width=edge_widths,
        edge_color=edge_colors,
        edge_cmap=plt.cm.get_cmap(cmap),
        alpha=0.6,
        arrows=True,
        arrowsize=15,
        arrowstyle='->'
    )

    # Simplify node labels and draw them
    node_labels = {node: _simplify_path(node) for node in G.nodes}

    # Draw labels with appropriate font sizes
    for node, (x, y) in pos.items():
        node_idx = list(G.nodes).index(node)
        size = node_sizes[node_idx]
        fontsize = min(12, max(8, size / 120))

        ax.text(
            x, y,
            node_labels[node],
            fontsize=fontsize,
            ha='center',
            va='center',
            bbox=dict(facecolor='white', alpha=0.7, boxstyle="round,pad=0.2")
        )

    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', label='Entry Points',
                   markerfacecolor='gray', markersize=15, markeredgecolor='white', markeredgewidth=2),
        plt.Line2D([0], [0], marker='s', color='w', label='Other Files',
                   markerfacecolor='gray', markersize=12, markeredgecolor='black')
    ]

    ax.legend(handles=legend_elements, loc='upper right')

    # Add colorbar for risk scores
    sm = plt.cm.ScalarMappable(cmap=plt.cm.get_cmap(cmap), norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Risk Score')

    # Set title and remove axes
    ax.set_title('Dependency Network with Risk Indicators')
    ax.axis('off')

    # Save figure
    output_file = output_dir / f"dependency_network.{file_format}"
    plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
    plt.close(fig)

    return str(output_file)


def _visualize_developer_activity(
        results: Dict[str, Any],
        output_dir: Path,
        file_format: str,
        max_items: int,
        figsize: Tuple[int, int],
        dpi: int,
        cmap: str
) -> str:
    """
    Visualize developer activity trends.

    Args:
        results: Results dictionary
        output_dir: Output directory
        file_format: Output file format
        max_items: Maximum items to display
        figsize: Figure size
        dpi: DPI for output
        cmap: Colormap name

    Returns:
        Path to output file
    """
    # Extract developer activity data
    developer_data = results.get("developer_data", {})
    risk_scores = results.get("risk_scores", {})

    if not developer_data:
        logger.warning("No developer data found, skipping developer activity visualization")
        return None

    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, gridspec_kw={'height_ratios': [2, 1]})

    # Extract top developers by number of commits
    developer_commits = {dev: data.get("commits", 0) for dev, data in developer_data.items()}
    top_devs = sorted(developer_commits.items(), key=lambda x: x[1], reverse=True)[:max_items]

    # Extract data for plotting
    dev_names = [dev[0] for dev in top_devs]
    dev_commits = [dev[1] for dev in top_devs]

    # Plot 1: Developer commits
    bars = ax1.bar(dev_names, dev_commits, color=plt.cm.get_cmap(cmap)(np.linspace(0, 1, len(dev_names))))

    # Add commit count labels on the bars
    for bar, count in zip(bars, dev_commits):
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.,
            height + 0.1,
            f"{count}",
            ha='center',
            va='bottom',
            fontsize=9
        )

    # Calculate risk statistics per developer
    dev_risk_stats = {}
    for dev, data in developer_data.items():
        modified_files = data.get("modified_files", [])
        if modified_files:
            dev_risks = [risk_scores.get(file, 0) for file in modified_files if file in risk_scores]
            if dev_risks:
                avg_risk = sum(dev_risks) / len(dev_risks)
                max_risk = max(dev_risks)
                dev_risk_stats[dev] = {
                    "avg_risk": avg_risk,
                    "max_risk": max_risk
                }

    # Plot 2: Developer risk scores
    if dev_risk_stats:
        top_dev_names = dev_names  # Use same developers as in commit chart

        # Extract average and max risk for selected developers
        avg_risks = [dev_risk_stats.get(dev, {"avg_risk": 0})["avg_risk"] for dev in top_dev_names]
        max_risks = [dev_risk_stats.get(dev, {"max_risk": 0})["max_risk"] for dev in top_dev_names]

        # Create width for bars
        width = 0.35
        x = np.arange(len(top_dev_names))

        # Plot average and max risk bars
        ax2.bar(x - width / 2, avg_risks, width, label='Avg Risk', color='cornflowerblue')
        ax2.bar(x + width / 2, max_risks, width, label='Max Risk', color='salmon')

        ax2.set_xticks(x)
        ax2.set_xticklabels(top_dev_names, rotation=45, ha='right')
        ax2.set_ylim(0, 1.0)
        ax2.legend(loc='upper right')
    else:
        ax2.text(
            0.5, 0.5,
            "No risk data available for developers",
            ha='center',
            va='center',
            fontsize=12
        )

    # Set labels and titles
    ax1.set_ylabel('Number of Commits')
    ax1.set_title('Developer Activity')
    ax2.set_ylabel('Risk Score')
    ax2.set_title('Developer Risk Profile')

    # Set x-tick positions and labels for developer chart
    ax1.set_xticklabels(dev_names, rotation=45, ha='right')

    # Add grid lines
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    ax2.grid(axis='y', linestyle='--', alpha=0.7)

    # Adjust layout
    plt.tight_layout()

    # Save figure
    output_file = output_dir / f"developer_activity.{file_format}"
    plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
    plt.close(fig)

    return str(output_file)


def _visualize_risk_treemap(
        results: Dict[str, Any],
        output_dir: Path,
        file_format: str,
        max_items: int,
        figsize: Tuple[int, int],
        dpi: int,
        cmap: str
) -> str:
    """
    Visualize risk scores as a treemap by directory structure.

    Args:
        results: Results dictionary
        output_dir: Output directory
        file_format: Output file format
        max_items: Maximum items to display
        figsize: Figure size
        dpi: DPI for output
        cmap: Colormap name

    Returns:
        Path to output file
    """
    try:
        # This visualization requires squarify package
        import squarify

        # Extract risk scores
        risk_scores = results.get("risk_scores", {})

        if not risk_scores:
            logger.warning("No risk scores found, skipping treemap visualization")
            return None

        # Group files by directory
        dir_risks = {}
        for file, risk in risk_scores.items():
            directory = os.path.dirname(file) or "/"  # Use "/" for root directory
            if directory not in dir_risks:
                dir_risks[directory] = {"total_risk": 0, "count": 0, "max_risk": 0}

            dir_risks[directory]["total_risk"] += risk
            dir_risks[directory]["count"] += 1
            dir_risks[directory]["max_risk"] = max(dir_risks[directory]["max_risk"], risk)

        # Calculate average risk for each directory
        for dir_name, stats in dir_risks.items():
            stats["avg_risk"] = stats["total_risk"] / stats["count"]

        # Sort directories by total risk and limit to max items
        top_dirs = sorted(
            dir_risks.items(),
            key=lambda x: x[1]["total_risk"],
            reverse=True
        )[:max_items]

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Extract data for plotting
        dir_names = [_simplify_path(item[0]) for item in top_dirs]
        dir_sizes = [item[1]["count"] for item in top_dirs]  # Size by number of files
        dir_colors = [item[1]["avg_risk"] for item in top_dirs]  # Color by average risk
        dir_max_risks = [item[1]["max_risk"] for item in top_dirs]  # For labeling

        # Create treemap
        squarify.plot(
            sizes=dir_sizes,
            label=[f"{name}\nMax: {risk:.2f}" for name, risk in zip(dir_names, dir_max_risks)],
            alpha=0.8,
            color=plt.cm.get_cmap(cmap)(dir_colors),
            ax=ax
        )

        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=plt.cm.get_cmap(cmap), norm=plt.Normalize(0, 1))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('Average Risk Score')

        # Set title and remove axes
        ax.set_title('Risk Distribution by Directory')
        ax.axis('off')

        # Save figure
        output_file = output_dir / f"risk_treemap.{file_format}"
        plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
        plt.close(fig)

        return str(output_file)

    except ImportError:
        logger.warning("Treemap visualization requires squarify package. Skipping.")
        return None


def _visualize_risk_heatmap(
        results: Dict[str, Any],
        output_dir: Path,
        file_format: str,
        max_items: int,
        figsize: Tuple[int, int],
        dpi: int,
        cmap: str
) -> str:
    """
    Visualize risk factors as a heatmap.

    Args:
        results: Results dictionary
        output_dir: Output directory
        file_format: Output file format
        max_items: Maximum items to display
        figsize: Figure size
        dpi: DPI for output
        cmap: Colormap name

    Returns:
        Path to output file
    """
    # Extract risk factors and scores
    risk_factors = results.get("risk_factors", {})
    risk_scores = results.get("risk_scores", {})

    if not risk_factors or not risk_scores:
        logger.warning("No risk factors or scores found, skipping heatmap visualization")
        return None

    # Get top risky files
    top_files = [file for file, _ in sorted(risk_scores.items(),
                                            key=lambda x: x[1],
                                            reverse=True)[:max_items]]

    # Create data matrix for heatmap
    # Collect all factor names and find which ones are available
    all_factors = set()
    for file, factors in risk_factors.items():
        all_factors.update(factors.keys())

    factor_list = sorted(list(all_factors))

    # Build data matrix
    data_matrix = []
    file_labels = []

    for file in top_files:
        file_labels.append(_simplify_path(file))

        if file in risk_factors:
            file_factors = risk_factors[file]
            row_data = [file_factors.get(factor, 0) for factor in factor_list]
        else:
            row_data = [0] * len(factor_list)

        data_matrix.append(row_data)

    # Convert to numpy array for heatmap
    data_array = np.array(data_matrix)

    # Create figure (wider figure for factor labels)
    fig, ax = plt.subplots(figsize=(figsize[0] + 2, figsize[1]))

    # Create heatmap with Seaborn
    heatmap = sns.heatmap(
        data_array,
        cmap=cmap,
        annot=True,
        fmt=".2f",
        linewidths=.5,
        ax=ax,
        yticklabels=file_labels,
        xticklabels=factor_list,
        cbar_kws={'label': 'Risk Factor Value'}
    )

    # Rotate x labels for better readability
    plt.xticks(rotation=45, ha='right')

    # Set title and labels
    ax.set_title('Risk Factors Heatmap')

    # Adjust layout
    plt.tight_layout()

    # Save figure
    output_file = output_dir / f"risk_heatmap.{file_format}"
    plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
    plt.close(fig)

    return str(output_file)


def _simplify_path(path: str, max_length: int = 30) -> str:
    """
    Simplify file path for display in visualizations.

    Args:
        path: File path
        max_length: Maximum length for the simplified path

    Returns:
        Simplified path
    """
    # Handle None inputs
    if not path:
        return ""

    # Use os.path for proper path handling
    basename = os.path.basename(path)
    dirname = os.path.dirname(path)

    if len(path) <= max_length:
        return path

    # Keep the basename and abbreviate the path
    if len(basename) > max_length - 4:
        # If basename alone is too long, truncate it
        return f"...{basename[-(max_length - 4):]}"
    else:
        # Keep full basename, abbreviate directory
        available_length = max_length - len(basename) - 4  # 4 for ".../"
        if available_length <= 0:
            return f".../{basename}"

        # Take the end of the dirname
        return f"...{dirname[-available_length:]}/{basename}"


def _get_risk_category(risk_score: float) -> str:
    """
    Get risk category label based on risk score.

    Args:
        risk_score: Risk score (0.0 to 1.0)

    Returns:
        Risk category label
    """
    if risk_score >= 0.8:
        return "Critical"
    elif risk_score >= 0.6:
        return "High"
    elif risk_score >= 0.4:
        return "Medium"
    elif risk_score >= 0.2:
        return "Low"
    else:
        return "Minimal"