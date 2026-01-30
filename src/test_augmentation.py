"""
test_augmentation.py - Prove augmentation happens on-the-fly

This script shows that:
- Training images are DIFFERENT each time (due to random augmentation)
- Validation images are IDENTICAL each time (no augmentation)
"""

import torch
import sys
import os

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset import get_train_transforms, get_val_transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from PIL import Image


def test_augmentation_simple():
    """
    Simple test without DataLoader (avoids multiprocessing issues on Windows)
    """
    
    print("=" * 60)
    print("AUGMENTATION PROOF TEST")
    print("=" * 60)
    
    # Find a sample image
    data_dir = "data/Training/MC"
    images = [f for f in os.listdir(data_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    
    if not images:
        print("❌ No images found!")
        return
    
    img_path = os.path.join(data_dir, images[0])
    print(f"\n📷 Using image: {img_path}")
    
    # Load image
    img = Image.open(img_path)
    
    # Get transforms
    train_transform = get_train_transforms()
    val_transform = get_val_transforms()
    
    # === TEST TRAINING TRANSFORMS ===
    print("\n" + "-" * 60)
    print("TRAINING TRANSFORM (with random augmentation)")
    print("-" * 60)
    print("Applying same transform 5 times to same image:\n")
    
    train_results = []
    for i in range(5):
        transformed = train_transform(img)
        mean_val = transformed.mean().item()
        std_val = transformed.std().item()
        min_val = transformed.min().item()
        max_val = transformed.max().item()
        train_results.append(mean_val)
        print(f"  Run {i+1}: mean={mean_val:.4f}, std={std_val:.4f}, min={min_val:.4f}, max={max_val:.4f}")
    
    # Check if values are different
    all_same = all(abs(x - train_results[0]) < 0.0001 for x in train_results)
    if all_same:
        print("\n⚠️ Values are the same - augmentation might not be working!")
    else:
        print("\n✅ Values are DIFFERENT each time = Augmentation is working!")
    
    # === TEST VALIDATION TRANSFORMS ===
    print("\n" + "-" * 60)
    print("VALIDATION TRANSFORM (no augmentation)")
    print("-" * 60)
    print("Applying same transform 5 times to same image:\n")
    
    val_results = []
    for i in range(5):
        transformed = val_transform(img)
        mean_val = transformed.mean().item()
        std_val = transformed.std().item()
        min_val = transformed.min().item()
        max_val = transformed.max().item()
        val_results.append(mean_val)
        print(f"  Run {i+1}: mean={mean_val:.4f}, std={std_val:.4f}, min={min_val:.4f}, max={max_val:.4f}")
    
    # Check if values are same
    all_same = all(abs(x - val_results[0]) < 0.0001 for x in val_results)
    if all_same:
        print("\n✅ Values are IDENTICAL each time = No augmentation (correct for validation)")
    else:
        print("\n⚠️ Values differ - something unexpected!")
    
    # === SUMMARY ===
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("""
    Training transforms:  RANDOM each time
    ├── RandomHorizontalFlip (50% chance)
    ├── RandomRotation (±15°)
    ├── ColorJitter (brightness, contrast, etc.)
    └── RandomAffine (shift, scale)
    
    Validation transforms: DETERMINISTIC (always same)
    ├── Resize
    └── Normalize
    
    This means during training:
    - Epoch 1: Model sees image_001.jpg (flipped, rotated 5°, darker)
    - Epoch 2: Model sees image_001.jpg (not flipped, rotated -3°, brighter)
    - Epoch 3: Model sees image_001.jpg (flipped, rotated 10°, normal)
    - ... and so on
    
    The model learns the CONCEPT, not the exact pixels!
    """)


if __name__ == '__main__':
    test_augmentation_simple()