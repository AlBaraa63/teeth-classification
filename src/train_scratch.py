"""
train_scratch.py - Training Script for Custom CNN (From Scratch)

This script trains the custom TeethClassifierImproved model built from scratch.
"""

import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config

# Override config to use scratch model
config.MODEL_TYPE = 'scratch'

# Import and run training
from train import *

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("TRAINING CUSTOM MODEL (FROM SCRATCH)")
    print("=" * 60)
    print(f"Model: TeethClassifierImproved")
    print("=" * 60)
