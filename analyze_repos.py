#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to analyze the 3 locally installed repositories using the REPD model
"""

import os
import time
from pathlib import Path

from repd.repository import LocalRepository
from repd.model import REPDModel

REPOSITORIES = [r"C:\Users\aniru\openjdk"]  # Path to your repositories

# Output directory for analysis results
OUTPUT_DIR = "./repd_analysis_results"


def analyze_repository(repo_path, output_dir):
    """Analyze a single repository and save results"""
    start_time = time.time()
    print(f"\n\n{'=' * 80}")
    print(f"Analyzing repository: {repo_path}")
    print(f"{'=' * 80}")

    # Create repository object
    repo = LocalRepository(repo_path)

    # Create model
    model = REPDModel(repo)

    # Configure model
    model.configure(
        max_files=500,  # Analyze up to 500 files
        history_days=90,  # Look at last 90 days of history
        risk_weights={  # Custom risk weights if desired
            "complexity": 0.25,
            "churn": 0.25,
            "coupling": 0.2,
            "structural": 0.2,
            "age": 0.1,
        },
    )

    # Run all analyses
    print("Running comprehensive analysis...")
    results = model.analyze_all()

    # Save results
    results_file = Path(output_dir) / "results.json"
    model.save_results(results_file)
    print(f"Results saved to {results_file}")

    # Generate visualizations
    viz_dir = Path(output_dir) / "visualizations"
    viz_files = model.visualize(viz_dir)
    print(f"Generated {len(viz_files)} visualizations in {viz_dir}")

    # Generate report
    report_file = Path(output_dir) / "report.html"
    model.generate_report(report_file)
    print(f"Generated report at {report_file}")

    end_time = time.time()
    print(f"Analysis completed in {end_time - start_time:.2f} seconds")

    # Print summary of findings
    print("\nAnalysis Summary:")
    print("-----------------")

    # Risk categories
    if "risk_categories" in results:
        categories = results["risk_categories"]
        print(f"High Risk Files: {len(categories.get('high_risk', []))}")
        print(f"Medium Risk Files: {len(categories.get('medium_risk', []))}")
        print(f"Low Risk Files: {len(categories.get('low_risk', []))}")

    # Entry points
    if "significant_entry_points" in results:
        entry_points = results["significant_entry_points"]
        print(f"Significant Entry Points: {len(entry_points)}")

    # Coupling
    if "high_coupling_pairs" in results:
        coupling_pairs = results["high_coupling_pairs"]
        print(f"High Coupling Pairs: {len(coupling_pairs)}")

    # Hotspots
    hotspots = model.identify_hotspots()
    print(f"Critical Hotspots: {len(hotspots)}")
    if hotspots:
        print("\nTop 5 Critical Hotspots:")
        for i, hotspot in enumerate(hotspots[:5]):
            print(f"  {i + 1}. {hotspot['file']} (Risk: {hotspot['risk_score']:.2f})")


def main():
    """Main function to analyze all repositories"""
    # Create base output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Analyze each repository
    for repo_path in REPOSITORIES:
        repo_name = os.path.basename(repo_path)
        repo_output_dir = os.path.join(OUTPUT_DIR, repo_name)
        os.makedirs(repo_output_dir, exist_ok=True)

        try:
            analyze_repository(repo_path, repo_output_dir)
        except Exception as e:
            print(f"Error analyzing repository {repo_path}: {str(e)}")
            import traceback

            traceback.print_exc()


if __name__ == "__main__":
    main()
