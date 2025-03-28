#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualization Module for REPD Model

This module contains functions for visualizing repository analysis results,
including risk scores, dependency networks, change coupling, and developer
activity patterns.

Author: anirudhsengar
"""

import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns
import squarify  # For treemap visualization
from matplotlib.dates import DateFormatter

logger = logging.getLogger(__name__)


def visualize_results(
    results: Dict, output_dir: Path, viz_types: List[str] = None
) -> Dict[str, str]:
    """
    Visualize repository analysis results.

    Args:
        results: Analysis results dictionary
        output_dir: Directory to save visualizations
        viz_types: List of visualization types to generate
                  (defaults to all available types)

    Returns:
        Dictionary mapping visualization types to output file paths
    """
    logger.info(f"Generating visualizations in {output_dir}")

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Default visualization types
    all_viz_types = [
        "risk",
        "coupling",
        "dependency",
        "activity",
        "risk_treemap",
        "risk_heatmap",
    ]

    # Use specified visualization types or all available types
    viz_types = viz_types or all_viz_types

    # Output file paths
    output_files = {}

    # Generate each requested visualization type
    for viz_type in viz_types:
        try:
            if viz_type == "risk" and "risk_scores" in results:
                output_file = output_dir / "risk_scores.png"
                _visualize_risk_scores(results["risk_scores"], output_file)
                output_files["risk"] = str(output_file)

            elif viz_type == "coupling" and "coupling_matrix" in results:
                output_file = output_dir / "change_coupling_network.png"
                _visualize_coupling_network(results["coupling_matrix"], output_file)
                output_files["coupling"] = str(output_file)

            elif viz_type == "dependency" and "dependency_graph" in results:
                output_file = output_dir / "dependency_network.png"
                _visualize_dependency_network(results["dependency_graph"], output_file)
                output_files["dependency"] = str(output_file)

            elif viz_type == "activity" and "commit_history" in results:
                output_file = output_dir / "developer_activity.png"
                _visualize_developer_activity(results["commit_history"], output_file)
                output_files["activity"] = str(output_file)

            elif viz_type == "risk_treemap" and "risk_scores" in results:
                output_file = output_dir / "risk_treemap.png"
                _visualize_risk_treemap(
                    results["risk_scores"], results.get("risk_factors", {}), output_file
                )
                output_files["risk_treemap"] = str(output_file)

            elif viz_type == "risk_heatmap" and "risk_scores" in results:
                output_file = output_dir / "risk_heatmap.png"
                _visualize_risk_heatmap(results["risk_scores"], output_file)
                output_files["risk_heatmap"] = str(output_file)

        except Exception as e:
            logger.error(f"Error generating {viz_type} visualization: {str(e)}")

    logger.info(f"Generated {len(output_files)} visualizations")
    return output_files


def _visualize_risk_scores(risk_scores: Dict[str, float], output_file: Path) -> None:
    """
    Visualize risk scores as a bar chart.

    Args:
        risk_scores: Dictionary mapping file paths to risk scores
        output_file: Path to save the visualization
    """
    logger.debug(f"Generating risk score visualization with {len(risk_scores)} files")

    # Sort files by risk score (descending)
    sorted_items = sorted(risk_scores.items(), key=lambda x: x[1], reverse=True)

    # Limit to top N files for readability
    top_n = 30
    if len(sorted_items) > top_n:
        logger.debug(f"Limiting visualization to top {top_n} high-risk files")
        sorted_items = sorted_items[:top_n]

    # Extract data for plotting
    file_paths = [_simplify_path(path) for path, _ in sorted_items]
    scores = [score for _, score in sorted_items]

    # Create color map based on risk levels
    colors = [_get_risk_color(score) for score in scores]

    # Create figure with enough height for the bars
    fig, ax = plt.subplots(figsize=(10, max(8, len(sorted_items) * 0.3)))

    # Create horizontal bar chart
    y_pos = np.arange(len(file_paths))
    ax.barh(y_pos, scores, color=colors)

    # Add labels and formatting
    ax.set_yticks(y_pos)
    ax.set_yticklabels(file_paths)
    ax.invert_yaxis()  # Display highest risk at the top
    ax.set_xlabel("Risk Score")
    ax.set_title("File Risk Scores")

    # Add risk level guidelines
    ax.axvline(x=0.7, color="red", linestyle="--", alpha=0.7)
    ax.axvline(x=0.4, color="orange", linestyle="--", alpha=0.7)
    ax.text(
        0.71, len(file_paths) - 1, "High Risk", color="red", verticalalignment="bottom"
    )
    ax.text(
        0.41,
        len(file_paths) - 1,
        "Medium Risk",
        color="orange",
        verticalalignment="bottom",
    )

    # Tight layout to ensure all labels are visible
    plt.tight_layout()

    # Save figure
    plt.savefig(output_file)
    plt.close()

    logger.debug(f"Risk score visualization saved to {output_file}")


def _visualize_coupling_network(
    coupling_matrix: Dict[str, Dict[str, float]],
    output_file: Path,
    threshold: float = 0.5,
    max_nodes: int = 50,
) -> None:
    """
    Visualize change coupling as a network.

    Args:
        coupling_matrix: Change coupling matrix
        output_file: Path to save the visualization
        threshold: Minimum coupling strength to include
        max_nodes: Maximum number of nodes to display
    """
    logger.debug(f"Generating change coupling visualization with threshold {threshold}")

    # Create graph
    G = nx.Graph()

    # Add edges for files with coupling above threshold
    edges = []
    for file1, couplings in coupling_matrix.items():
        for file2, strength in couplings.items():
            if strength >= threshold:
                edges.append((file1, file2, {"weight": strength}))

    # If too many edges, increase threshold adaptively
    adaptive_threshold = threshold
    while len(edges) > 200 and adaptive_threshold < 0.95:
        adaptive_threshold += 0.05
        edges = [e for e in edges if e[2]["weight"] >= adaptive_threshold]

    # Add edges to graph
    G.add_edges_from(edges)

    # If no edges, nothing to visualize
    if not G.edges():
        logger.warning(
            f"No coupling relationships above threshold {adaptive_threshold}"
        )
        plt.figure(figsize=(8, 6))
        plt.text(
            0.5,
            0.5,
            f"No coupling relationships above threshold {adaptive_threshold}",
            horizontalalignment="center",
            verticalalignment="center",
        )
        plt.savefig(output_file)
        plt.close()
        return

    # If too many nodes, limit to most connected ones
    if len(G) > max_nodes:
        # Get most connected nodes by degree centrality
        centrality = nx.degree_centrality(G)
        top_nodes = sorted(
            centrality.keys(), key=lambda x: centrality[x], reverse=True
        )[:max_nodes]
        G = G.subgraph(top_nodes)

    # Create figure
    plt.figure(figsize=(12, 10))

    # Layout
    pos = nx.spring_layout(G, k=0.3, seed=42)

    # Get edge weights for line thickness
    edge_weights = [G[u][v]["weight"] * 3 for u, v in G.edges()]

    # Get node degrees for node size
    node_size = [300 * (0.1 + nx.degree(G)[node]) for node in G.nodes()]

    # Draw the network
    nx.draw(
        G,
        pos,
        with_labels=False,
        node_color="skyblue",
        node_size=node_size,
        edge_color="gray",
        width=edge_weights,
        alpha=0.7,
    )

    # Add labels for nodes, but only for the most connected ones to avoid clutter
    if len(G) > 15:
        # Label only top connected nodes
        degrees = nx.degree(G)
        top_nodes = sorted(G.nodes(), key=lambda x: degrees[x], reverse=True)[:15]
        label_dict = {node: _simplify_path(node) for node in top_nodes}
    else:
        # Label all nodes
        label_dict = {node: _simplify_path(node) for node in G.nodes()}

    nx.draw_networkx_labels(G, pos, labels=label_dict, font_size=8)

    # Add title with threshold information
    plt.title(
        f"Change Coupling Network (threshold: {adaptive_threshold:.2f})", fontsize=16
    )

    # Save figure
    plt.savefig(output_file)
    plt.close()

    logger.debug(f"Change coupling visualization saved to {output_file}")


def _visualize_dependency_network(
    dependency_graph: nx.DiGraph, output_file: Path, max_nodes: int = 50
) -> None:
    """
    Visualize dependency network from the structure mapper.

    Args:
        dependency_graph: Directed dependency graph
        output_file: Path to save the visualization
        max_nodes: Maximum number of nodes to display
    """
    logger.debug(f"Generating dependency network visualization")

    # Create a copy for visualization
    G = dependency_graph.copy()

    # If no nodes, nothing to visualize
    if not G.nodes():
        logger.warning("No dependencies to visualize")
        plt.figure(figsize=(8, 6))
        plt.text(
            0.5,
            0.5,
            "No dependencies found",
            horizontalalignment="center",
            verticalalignment="center",
        )
        plt.savefig(output_file)
        plt.close()
        return

    # If too many nodes, limit to most central ones
    if len(G) > max_nodes:
        # Get centrality measures
        try:
            # Try to use existing centrality if available in node attributes
            centrality = nx.get_node_attributes(G, "centrality")
            if not centrality:
                # Calculate centrality
                centrality = nx.betweenness_centrality(G)
        except:
            # Fallback to degree centrality if betweenness fails
            centrality = nx.degree_centrality(G)

        # Get top central nodes
        top_nodes = sorted(
            centrality.keys(), key=lambda x: centrality.get(x, 0), reverse=True
        )[:max_nodes]
        G = G.subgraph(top_nodes)

    # Create figure
    plt.figure(figsize=(12, 10))

    # Layout - hierarchical for directed graph
    try:
        pos = nx.nx_agraph.graphviz_layout(G, prog="dot")
    except:
        # Fallback if graphviz not available
        pos = nx.spring_layout(G, k=0.3, seed=42)

    # Get node color based on in-degree (number of files that import this)
    in_degree = dict(G.in_degree())
    max_in = max(in_degree.values()) if in_degree else 1
    node_colors = [cm.viridis(in_degree.get(node, 0) / max_in) for node in G.nodes()]

    # Node size based on centrality
    try:
        centrality = nx.get_node_attributes(
            G, "centrality"
        ) or nx.betweenness_centrality(G)
    except:
        centrality = nx.degree_centrality(G)

    node_size = [3000 * (0.1 + centrality.get(node, 0)) for node in G.nodes()]

    # Draw the network
    nx.draw(
        G,
        pos,
        with_labels=False,
        node_size=node_size,
        node_color=node_colors,
        edge_color="gray",
        arrowsize=10,
        arrows=True,
        alpha=0.7,
    )

    # Add labels for nodes, but only for the most important ones to avoid clutter
    if len(G) > 15:
        # Label only top important nodes
        top_nodes = sorted(G.nodes(), key=lambda x: centrality.get(x, 0), reverse=True)[
            :15
        ]
        label_dict = {node: _simplify_path(node) for node in top_nodes}
    else:
        # Label all nodes
        label_dict = {node: _simplify_path(node) for node in G.nodes()}

    nx.draw_networkx_labels(G, pos, labels=label_dict, font_size=8)

    # Add title
    plt.title("Code Dependency Network", fontsize=16)

    # Add legend for node colors
    sm = plt.cm.ScalarMappable(cmap=cm.viridis, norm=plt.Normalize(0, max_in))
    sm.set_array([])
    cbar = plt.colorbar(sm)
    cbar.set_label("Import Count (In-Degree)")

    # Save figure
    plt.savefig(output_file)
    plt.close()

    logger.debug(f"Dependency network visualization saved to {output_file}")


def _visualize_developer_activity(
    commit_history: List[Dict], output_file: Path, days: int = 90
) -> None:
    """
    Visualize developer activity over time.

    Args:
        commit_history: List of commit objects
        output_file: Path to save the visualization
        days: Number of days to include in the visualization
    """
    logger.debug(
        f"Generating developer activity visualization for the past {days} days"
    )

    # Filter commits to the specified time range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    # Group commits by date and author
    date_range = [start_date + timedelta(days=i) for i in range(days + 1)]
    date_str = [d.strftime("%Y-%m-%d") for d in date_range]

    # Extract commit dates and authors
    filtered_commits = []
    for commit in commit_history:
        if isinstance(commit, dict):
            commit_date = commit.get("date")
            commit_author = commit.get("author")
            if commit_date and commit_author:
                if not isinstance(commit_date, datetime):
                    try:
                        commit_date = datetime.strptime(
                            commit_date, "%Y-%m-%d %H:%M:%S"
                        )
                    except:
                        continue
                if start_date <= commit_date <= end_date:
                    filtered_commits.append(
                        {"date": commit_date, "author": commit_author}
                    )
        else:
            # Handle case where commit is an object with different attributes
            commit_date = getattr(commit, "date", None)
            commit_author = getattr(commit, "author", None)
            if commit_date and commit_author:
                if start_date <= commit_date <= end_date:
                    filtered_commits.append(
                        {"date": commit_date, "author": commit_author}
                    )

    # Count commits by day and author
    author_commits = {}
    for commit in filtered_commits:
        day_str = commit["date"].strftime("%Y-%m-%d")
        author = commit["author"]
        if author not in author_commits:
            author_commits[author] = {d: 0 for d in date_str}
        author_commits[author][day_str] = author_commits[author].get(day_str, 0) + 1

    # If no data, return empty visualization
    if not author_commits:
        logger.warning("No commit data found for the specified time range")
        plt.figure(figsize=(10, 6))
        plt.text(
            0.5,
            0.5,
            "No commit data found for the specified time range",
            horizontalalignment="center",
            verticalalignment="center",
        )
        plt.savefig(output_file)
        plt.close()
        return

    # Limit to top N authors by commit count
    top_n = 10
    author_totals = {
        author: sum(commits.values()) for author, commits in author_commits.items()
    }
    top_authors = sorted(author_totals.items(), key=lambda x: x[1], reverse=True)[
        :top_n
    ]

    # Create figure
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 10), gridspec_kw={"height_ratios": [3, 1]}
    )

    # Prepare data for stacked area chart
    dates = [datetime.strptime(d, "%Y-%m-%d") for d in date_str]
    commit_data = []
    authors = []

    for author, _ in top_authors:
        authors.append(author)
        commit_data.append([author_commits[author][d] for d in date_str])

    # Create stacked area chart - binned by week
    ax1.stackplot(dates, commit_data, labels=authors, alpha=0.7)

    # Format x-axis to show dates nicely
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax1.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    plt.setp(ax1.get_xticklabels(), rotation=45, ha="right")

    # Set labels and title
    ax1.set_ylabel("Number of Commits")
    ax1.set_title(f"Developer Activity (Last {days} Days)")

    # Add legend
    ax1.legend(loc="upper left")

    # Add commit heatmap by weekday/hour
    weekday_hour = {}
    for commit in filtered_commits:
        weekday = commit["date"].weekday()
        hour = commit["date"].hour
        key = (weekday, hour)
        weekday_hour[key] = weekday_hour.get(key, 0) + 1

    # Create heatmap data
    heatmap_data = np.zeros((7, 24))
    for (weekday, hour), count in weekday_hour.items():
        heatmap_data[weekday, hour] = count

    # Plot heatmap
    weekday_names = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]
    ax2.imshow(heatmap_data, aspect="auto", cmap="YlGnBu")

    # Configure heatmap axes
    ax2.set_yticks(np.arange(7))
    ax2.set_yticklabels(weekday_names)
    ax2.set_xticks(np.arange(0, 24, 2))
    ax2.set_xticklabels([f"{h:02d}:00" for h in range(0, 24, 2)])
    ax2.set_title("Commit Activity by Weekday and Hour")
    ax2.set_xlabel("Hour of Day (UTC)")

    # Add colorbar
    cbar = plt.colorbar(plt.cm.ScalarMappable(cmap="YlGnBu"), ax=ax2)
    cbar.set_label("Number of Commits")

    # Adjust layout
    plt.tight_layout()

    # Save figure
    plt.savefig(output_file)
    plt.close()

    logger.debug(f"Developer activity visualization saved to {output_file}")


def _visualize_risk_treemap(
    risk_scores: Dict[str, float],
    risk_factors: Dict[str, Dict[str, float]],
    output_file: Path,
    min_risk: float = 0.3,
) -> None:
    """
    Visualize risk scores as a treemap, grouping files by directory.

    Args:
        risk_scores: Dictionary mapping file paths to risk scores
        risk_factors: Dictionary mapping files to their risk factor breakdowns
        output_file: Path to save the visualization
        min_risk: Minimum risk score to include in visualization
    """
    logger.debug(f"Generating risk treemap visualization with {len(risk_scores)} files")

    # Filter to files with risk above threshold
    filtered_scores = {
        path: score for path, score in risk_scores.items() if score >= min_risk
    }

    if not filtered_scores:
        logger.warning(f"No files with risk score >= {min_risk}")
        plt.figure(figsize=(8, 6))
        plt.text(
            0.5,
            0.5,
            f"No files with risk score >= {min_risk}",
            horizontalalignment="center",
            verticalalignment="center",
        )
        plt.savefig(output_file)
        plt.close()
        return

    # Group files by directory
    dir_structure = {}
    for path, score in filtered_scores.items():
        parts = path.split("/")
        if len(parts) > 1:
            # Has directory
            directory = "/".join(parts[:-1])
            filename = parts[-1]
        else:
            # In root directory
            directory = "(root)"
            filename = path

        if directory not in dir_structure:
            dir_structure[directory] = {}

        dir_structure[directory][filename] = score

    # Prepare data for treemap
    treemap_data = []

    # First add directories
    for directory, files in dir_structure.items():
        # Calculate total risk and file count for directory
        total_risk = sum(files.values())
        file_count = len(files)

        # Add directory entry
        treemap_data.append(
            {
                "name": directory,
                "value": total_risk * 100,  # Scale for better visualization
                "color": "lightgray",
                "alpha": 0.7,
                "is_dir": True,
            }
        )

        # Add entries for high-risk files
        for filename, score in sorted(files.items(), key=lambda x: x[1], reverse=True):
            if len(treemap_data) > 100:  # Limit entries for readability
                break

            # Add file entry
            treemap_data.append(
                {
                    "name": filename,
                    "dir": directory,
                    "value": score * 100,  # Scale for better visualization
                    "color": _get_risk_color(score),
                    "alpha": 0.8,
                    "is_dir": False,
                }
            )

    # Create figure
    plt.figure(figsize=(12, 10))

    # Get data in the format required by squarify
    values = [item["value"] for item in treemap_data]
    colors = [item["color"] for item in treemap_data]
    alphas = [item["alpha"] for item in treemap_data]

    # Create treemap
    squarify.plot(
        sizes=values,
        color=colors,
        alpha=alphas,
        label=[item["name"] for item in treemap_data],
        text_kwargs={"fontsize": 8, "wrap": True},
    )

    # Set title and remove axes
    plt.title("Code Risk Treemap (by Directory)", fontsize=16)
    plt.axis("off")

    # Add legend
    high_patch = plt.Rectangle(
        (0, 0), 1, 1, color=_get_risk_color(0.8), alpha=0.8, label="High Risk"
    )
    medium_patch = plt.Rectangle(
        (0, 0), 1, 1, color=_get_risk_color(0.5), alpha=0.8, label="Medium Risk"
    )
    low_patch = plt.Rectangle(
        (0, 0), 1, 1, color=_get_risk_color(0.3), alpha=0.8, label="Low Risk"
    )
    dir_patch = plt.Rectangle(
        (0, 0), 1, 1, color="lightgray", alpha=0.7, label="Directories"
    )

    plt.legend(
        handles=[high_patch, medium_patch, low_patch, dir_patch], loc="upper right"
    )

    # Save figure
    plt.savefig(output_file)
    plt.close()

    logger.debug(f"Risk treemap visualization saved to {output_file}")


def _visualize_risk_heatmap(
    risk_scores: Dict[str, float], output_file: Path, group_by_directory: bool = True
) -> None:
    """
    Visualize risk scores as a heatmap showing distribution across the codebase.

    Args:
        risk_scores: Dictionary mapping file paths to risk scores
        output_file: Path to save the visualization
        group_by_directory: Whether to group and aggregate by directory
    """
    logger.debug(f"Generating risk heatmap visualization")

    if not risk_scores:
        logger.warning("No risk scores to visualize")
        plt.figure(figsize=(8, 6))
        plt.text(
            0.5,
            0.5,
            "No risk score data available",
            horizontalalignment="center",
            verticalalignment="center",
        )
        plt.savefig(output_file)
        plt.close()
        return

    # Parse paths and prepare data structure
    if group_by_directory:
        # Group by directory
        dir_risks = {}
        dir_counts = {}

        for path, score in risk_scores.items():
            parts = path.split("/")
            if len(parts) > 1:
                # Has directory
                directory = "/".join(parts[:-1])
            else:
                # In root directory
                directory = "(root)"

            if directory not in dir_risks:
                dir_risks[directory] = 0
                dir_counts[directory] = 0

            dir_risks[directory] += score
            dir_counts[directory] += 1

        # Calculate average risk by directory
        dir_avg_risk = {d: dir_risks[d] / dir_counts[d] for d in dir_risks}

        # Create entries for visualization
        labels = []
        values = []

        for d, risk in sorted(dir_avg_risk.items(), key=lambda x: x[1], reverse=True):
            labels.append(d)
            values.append(risk)

        # Limit number of entries for readability
        if len(labels) > 30:
            labels = labels[:30]
            values = values[:30]

        # Create figure
        plt.figure(figsize=(12, 10))

        # Create colormap
        cmap = plt.cm.get_cmap("YlOrRd")

        # Create horizontal bars with color based on risk
        y_pos = np.arange(len(labels))
        colors = [cmap(val) for val in values]

        plt.barh(y_pos, values, color=colors)
        plt.yticks(y_pos, labels)
        plt.xlabel("Average Risk Score")
        plt.title("Directory Risk Heatmap")

        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap)
        sm.set_array(values)
        cbar = plt.colorbar(sm)
        cbar.set_label("Risk Level")

        # Add risk level indicators
        plt.axvline(x=0.7, color="red", linestyle="--", alpha=0.7)
        plt.axvline(x=0.4, color="orange", linestyle="--", alpha=0.7)
        plt.text(
            0.71, len(labels) - 1, "High Risk", color="red", verticalalignment="bottom"
        )
        plt.text(
            0.41,
            len(labels) - 1,
            "Medium Risk",
            color="orange",
            verticalalignment="bottom",
        )

    else:
        # Create a hierarchical path structure
        path_structure = {}

        for path, score in risk_scores.items():
            parts = path.split("/")
            current = path_structure

            # Build the path hierarchy
            for i, part in enumerate(parts):
                if i == len(parts) - 1:
                    # Leaf node (file)
                    current[part] = score
                else:
                    # Directory node
                    if part not in current:
                        current[part] = {}
                    current = current[part]

        # Create figure
        plt.figure(figsize=(12, 12))

        # Use recursive function to plot hierarchy (implementation omitted for brevity)
        # This would require a more complex visualization library like plotly or custom sunburst chart

        # For simplicity, we'll use a scatter plot with directories as x and files as y
        x = []
        y = []
        s = []  # sizes
        c = []  # colors

        for path, score in risk_scores.items():
            parts = path.split("/")
            if len(parts) > 1:
                # Has directory
                directory = parts[0]
                filename = "/".join(parts[1:])
            else:
                # In root directory
                directory = "(root)"
                filename = path

            x.append(directory)
            y.append(filename)
            s.append(score * 100)  # Scale for visibility
            c.append(score)

        # Create scatter plot
        unique_dirs = sorted(set(x))
        dir_indices = {d: i for i, d in enumerate(unique_dirs)}

        plt.scatter([dir_indices[d] for d in x], y, s=s, c=c, cmap="YlOrRd", alpha=0.6)
        plt.xticks(range(len(unique_dirs)), unique_dirs, rotation=90)
        plt.title("File Risk Heatmap")
        plt.colorbar(label="Risk Score")
        plt.tight_layout()

    # Save figure
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

    logger.debug(f"Risk heatmap visualization saved to {output_file}")


def _simplify_path(path: str) -> str:
    """
    Simplify a file path for display in visualizations.

    Args:
        path: Full file path

    Returns:
        Simplified file path for display
    """
    # If path is short, return it as is
    if len(path) < 30:
        return path

    # Extract meaningful parts
    parts = path.split("/")

    if len(parts) <= 2:
        # Just filename or directory/filename
        return path

    elif len(parts) == 3:
        # Simple case like "src/module/file.py"
        return path

    else:
        # More complex path
        # Keep first directory, use ellipsis for middle, and keep last 2 parts
        first = parts[0]
        last_two = "/".join(parts[-2:])
        return f"{first}/.../{last_two}"


def _get_risk_color(risk_score: float) -> str:
    """
    Get color based on risk score.

    Args:
        risk_score: Risk score (0-1)

    Returns:
        Color string
    """
    if risk_score >= 0.7:
        return "#d73027"  # Red
    elif risk_score >= 0.4:
        return "#fc8d59"  # Orange
    else:
        return "#91bfdb"  # Blue


def _get_risk_category(risk_score: float) -> str:
    """
    Get risk category based on risk score.

    Args:
        risk_score: Risk score (0-1)

    Returns:
        Risk category string
    """
    if risk_score >= 0.7:
        return "High"
    elif risk_score >= 0.4:
        return "Medium"
    else:
        return "Low"
