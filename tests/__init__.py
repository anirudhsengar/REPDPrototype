#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Suite for REPD Model

This package contains tests for all components of the REPD model,
including repository analysis, structure mapping, risk scoring,
and visualization.

Author: anirudhsengar
"""

import logging
import os
import sys
from pathlib import Path

# Add the parent directory to sys.path to allow importing the main package
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure logging for tests
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

# Suppress excessive logging during tests
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('PIL').setLevel(logging.WARNING)