# REPDPrototype

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![Status](https://img.shields.io/badge/status-prototype-orange.svg)

A Python implementation of the Repository Entry Points Defects (REPD) model for bug prediction in software repositories.

## About REPD Model

The REPD (Repository Entry Points Defects) model is a bug prediction approach that leverages:

1. **Entry Point Analysis**: Identifies critical repository entry points (APIs, UI components, exported functionality)
2. **Change Coupling**: Analyzes which files tend to change together
3. **Developer Activity Patterns**: Studies developer expertise and contribution patterns
4. **Structural Relationships**: Maps dependency and inheritance relationships

Unlike traditional bug prediction models that rely solely on historical bug data, REPD focuses on structural relationships and developer interaction patterns, which can be more effective at predicting bugs in evolving codebases.

## Installation

### Prerequisites
- Python 3.8 or higher
- Git

### Installing from source

```bash
# Clone the repository
git clone https://github.com/anirudhsengar/REPDPrototype.git
cd REPDPrototype

# Install the package in development mode
pip install -e .

# Install development dependencies (optional)
pip install -e ".[dev]"
```

## Usage

### Basic Analysis

```bash
# Run REPD analysis on a repository
repd analyze --repo /path/to/repository --lookback 1000

# Generate visualizations
repd visualize --results /path/to/results.json --output visualizations/

# Compare with other models (if implemented)
repd compare --repd /path/to/repd/results.json --fixcache /path/to/fixcache/results.json
```

### Python API

```python
from repd.model import REPDModel

# Initialize the model
model = REPDModel(
    repo_path="/path/to/repository",
    lookback_commits=1000,
    entry_point_weight=0.5,
    coupling_threshold=0.3
)

# Run analysis
results = model.analyze()

# Get risky files
risky_files = model.get_top_risky_files(limit=20)

# Generate visualization
model.visualize(output_path="visualizations/")
```

## Project Structure

```
REPDPrototype/
├── .github/
│   └── workflows/
│       └── python-tests.yml       # GitHub Actions CI workflow
├── docs/
│   ├── README.md                  # Documentation main page
│   ├── model_description.md       # REPD model technical description
│   └── comparison_results.md      # Future comparison with FixCache
├── repd/
│   ├── __init__.py                # Package initialization
│   ├── cli.py                     # Command-line interface
│   ├── model.py                   # Main REPD model class
│   ├── entry_point_analyzer.py    # Repository entry point analysis
│   ├── entry_point_detector.py    # Repository entry point detection
│   ├── change_coupling.py         # Change coupling analysis
│   ├── change_coupling_analyzer.py# Change coupling analysis helper
│   ├── developer_activity.py      # Developer activity tracking
│   ├── risk_calculator.py         # Defect risk calculation
│   ├── repository.py              # Repository access and analysis
│   ├── structure_mapper.py        # Repository structure mapping
│   └── visualization.py           # Visualization utilities
├── tests/
│   ├── __init__.py
│   ├── test_model.py              # Tests for REPD model
│   ├── test_entry_points.py       # Tests for entry point detection
│   ├── test_change_coupling.py    # Tests for change coupling
│   ├── test_risk_calculator.py    # Tests for risk calculation
│   ├── test_repository.py         # Tests for repository analysis
│   └── test_cli.py                # Tests for CLI
├── test_results/                  # Directory for test results and visualizations
├── .gitignore                     # Git ignore file
├── LICENSE                        # License file
├── README.md                      # Project README
├── requirements.txt               # Python dependencies
├── setup.py                       # Package setup script
└── pyproject.toml                 # Python project metadata

```

## Comparison with Other Models

The REPD model will be compared with:

- **FixCache Algorithm**: A bug prediction approach that uses a cache of bug-prone files
- **Traditional metrics-based approaches**: LOC, complexity, churn, etc.

Results will be documented in `docs/comparison_results.md`.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- This is a prototype implementation for the GlitchWitcher GSOC 2025 project
- The REPD model is based on research in software defect prediction and repository mining

---

Created by [Anirudh Sengar](https://github.com/anirudhsengar)