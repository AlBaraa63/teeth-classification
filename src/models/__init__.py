"""
models package - Contains all model architectures

Available models:
- scratch_model: Custom CNNs built from scratch
- transfer_model: Pre-trained models for transfer learning
"""

from .scratch_model import (
    TeethClassifier,
    TeethClassifierImproved,
    ConvBlock,
    ResidualBlock,
    count_parameters,
    model_summary
)

from .transfer_model import (
    get_transfer_model,
    freeze_feature_extractor,
    unfreeze_all_layers
)

__all__ = [
    # Scratch models
    'TeethClassifier',
    'TeethClassifierImproved',
    'ConvBlock',
    'ResidualBlock',
    
    # Transfer learning
    'get_transfer_model',
    'freeze_feature_extractor',
    'unfreeze_all_layers',
    
    # Utilities
    'count_parameters',
    'model_summary'
]
