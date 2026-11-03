"""Shared data paths for the retained experiment and submission scripts."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from housing_affordability.features import DATA, read_table

TARGET = 'AffordabilityPercentageTrue'
