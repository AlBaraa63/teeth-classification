"""
train_transfer.py - Training Script for Transfer Learning Models

This script trains a transfer learning model (ResNet18/50, EfficientNet).
You can configure which model to use in config.py.
"""

import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config

# Override config to use transfer learning
config.MODEL_TYPE = 'transfer'
config.FREEZE_FEATURES = False  # Unfreeze all layers for fine-tuning
config.LEARNING_RATE = 0.0001   # Lower LR to prevent destroying pre-trained weights

# Import and run training
import train

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("TRAINING TRANSFER LEARNING MODEL")
    print("=" * 60)
    print(f"Model: {config.TRANSFER_MODEL_NAME}")
    print(f"Freeze Features: {config.FREEZE_FEATURES}")
    print("=" * 60)
    
    train.main()
