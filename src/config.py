"""
config.py - Centralized Configuration

This file contains all the settings, paths, and hyperparameters for the project.
Import this file in other scripts to access these values.
"""

import os
import torch

# ============================================================
# PATHS
# ============================================================

# Project Root (one level up from src)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Data Directory
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

# Output Directories
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
MODELS_DIR = os.path.join(OUTPUT_DIR, "models")

# Ensure output directories exist
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


# ============================================================
# DATASET CONSTANTS
# ============================================================

# The 7 classes we're classifying
CLASSES = ['CaS', 'CoS', 'Gum', 'MC', 'OC', 'OLP', 'OT']

# Friendly names for display
CLASS_FULL_NAMES = {
    'CaS': 'Calculus (Tartar)',
    'CoS': 'Caries (Cavities)',
    'Gum': 'Gum Disease',
    'MC': 'Mouth Cancer',
    'OC': 'Oral Candidiasis',
    'OLP': 'Oral Lichen Planus',
    'OT': 'Oral Trauma'
}

# ImageNet statistics for normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ============================================================
# COMPUTE SETTINGS
# ============================================================

# Use GPU if available
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Number of workers for DataLoader
# On Windows, set to 0 to avoid multiprocessing errors
NUM_WORKERS = 0 if os.name == 'nt' else 4


# ============================================================
# TRAINING HYPERPARAMETERS
# ============================================================

BATCH_SIZE = 32
IMAGE_SIZE = 224
LEARNING_RATE = 0.001
NUM_EPOCHS = 50
PATIENCE = 10  # Early stopping patience
DROPOUT_RATE = 0.5
