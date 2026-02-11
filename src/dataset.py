"""
dataset.py - Data Loading and Augmentation

This module handles:
1. Loading images from folders
2. Applying augmentations (training only)
3. Preparing data for the model

KEY CONCEPT FROM ALEXNET:
Data augmentation artificially increases dataset size by creating
variations of existing images. This helps prevent overfitting.
"""

import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image
import matplotlib.pyplot as plt


# === CONFIGURATION ===
from . import config

# === CONFIGURATION ===
# Using constants from config.py
IMAGE_SIZE = config.IMAGE_SIZE
BATCH_SIZE = config.BATCH_SIZE
NUM_WORKERS = config.NUM_WORKERS

# ImageNet statistics (used for normalization)
IMAGENET_MEAN = config.IMAGENET_MEAN
IMAGENET_STD = config.IMAGENET_STD


def get_train_transforms():
    """
    Transforms for TRAINING data.
    
    Includes augmentation to artificially increase dataset variety.
    Each time an image is loaded, random transforms are applied,
    so the model sees slightly different versions each epoch.
    """
    return transforms.Compose([
        # 1. RESIZE: Make all images the same size
        #    Why 224? It's the standard from AlexNet, works well with most architectures
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        
        # 2. RANDOM HORIZONTAL FLIP: 50% chance to mirror the image
        #    Why? Teeth conditions look the same flipped horizontally
        #    This doubles our effective dataset size
        transforms.RandomHorizontalFlip(p=0.5),
        
        # 3. RANDOM ROTATION: Slight rotation (-15 to +15 degrees)
        #    Why? Camera angle varies in real photos
        #    Why only 15°? Medical images shouldn't be rotated too much
        transforms.RandomRotation(degrees=15),
        
        # 4. COLOR JITTER: Randomly adjust brightness, contrast, saturation
        #    Why? Lighting conditions vary in dental photos
        #    Why small values? We don't want to change colors drastically
        #    (color is diagnostic in medical images!)
        transforms.ColorJitter(
            brightness=0.2,  # ±20% brightness
            contrast=0.2,    # ±20% contrast
            saturation=0.2,  # ±20% saturation
            hue=0.1          # ±10% hue (very subtle - preserve tooth color)
        ),
        
        # 5. RANDOM AFFINE: Small translations and scaling
        #    Why? Teeth might not always be centered perfectly
        transforms.RandomAffine(
            degrees=0,           # No additional rotation (already did above)
            translate=(0.1, 0.1), # Shift up to 10% in any direction
            scale=(0.9, 1.1)     # Scale between 90% and 110%
        ),
        
        # 6. CONVERT TO TENSOR: PIL Image → PyTorch Tensor
        #    Changes shape from (H, W, C) to (C, H, W)
        #    Changes values from [0, 255] to [0.0, 1.0]
        transforms.ToTensor(),
        
        # 7. NORMALIZE: Standardize pixel values
        #    Why? Helps the network learn faster and more stably
        #    Formula: (pixel - mean) / std
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])


def get_val_transforms():
    """
    Transforms for VALIDATION and TESTING data.
    
    NO augmentation here! We want consistent, reproducible evaluation.
    Only resize, convert to tensor, and normalize.
    """
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])


def create_dataloaders(data_dir, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS):
    """
    Create DataLoaders for training, validation, and testing.
    
    DataLoader handles:
    - Batching: Groups images together
    - Shuffling: Randomizes order each epoch (training only)
    - Parallel loading: Uses multiple CPU cores
    
    Returns:
        dict with 'train', 'val', 'test' DataLoaders
        class_names: list of class names
    """
    
    # Paths to each split
    train_dir = os.path.join(data_dir, "Training")
    val_dir = os.path.join(data_dir, "Validation")
    test_dir = os.path.join(data_dir, "Testing")
    
    # Create datasets using ImageFolder
    # ImageFolder automatically assigns labels based on subfolder names!
    train_dataset = datasets.ImageFolder(
        root=train_dir,
        transform=get_train_transforms()
    )
    
    val_dataset = datasets.ImageFolder(
        root=val_dir,
        transform=get_val_transforms()
    )
    
    test_dataset = datasets.ImageFolder(
        root=test_dir,
        transform=get_val_transforms()
    )
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,       # Randomize order each epoch
        num_workers=num_workers,
        pin_memory=True     # Faster GPU transfer
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,      # No shuffling for validation
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Get class names from folder structure
    class_names = train_dataset.classes
    
    print(f"✓ Training samples: {len(train_dataset)}")
    print(f"✓ Validation samples: {len(val_dataset)}")
    print(f"✓ Test samples: {len(test_dataset)}")
    print(f"✓ Classes: {class_names}")
    
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }, class_names


# === VISUALIZATION: Before/After Augmentation ===

def denormalize(tensor):
    """
    Reverse the normalization to display images properly.
    
    The normalized image looks weird (blue/green tint) because
    values are centered around 0. This function restores original colors.
    """
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return tensor * std + mean


def visualize_augmentations(data_dir, class_name="MC", num_versions=5):
    """
    Show the same image with different random augmentations.
    
    THIS IS YOUR SECOND DELIVERABLE!
    Shows how augmentation creates variety from a single image.
    """
    # Load one image
    class_path = os.path.join(data_dir, "Training", class_name)
    images = [f for f in os.listdir(class_path) 
              if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    img_path = os.path.join(class_path, images[0])
    
    # Load original image
    original_img = Image.open(img_path)
    
    # Get transforms
    train_transform = get_train_transforms()
    val_transform = get_val_transforms()
    
    # Create figure
    fig, axes = plt.subplots(2, num_versions + 1, figsize=(16, 7))
    
    # Row 1: Original + Augmented versions
    axes[0, 0].imshow(original_img)
    axes[0, 0].set_title("Original", fontsize=10)
    axes[0, 0].axis('off')
    
    for i in range(num_versions):
        augmented = train_transform(original_img)
        augmented_display = denormalize(augmented).permute(1, 2, 0).clamp(0, 1)
        
        axes[0, i + 1].imshow(augmented_display)
        axes[0, i + 1].set_title(f"Augmented #{i+1}", fontsize=10)
        axes[0, i + 1].axis('off')
    
    # Row 2: Show what validation transform looks like (no randomness)
    val_transformed = val_transform(original_img)
    val_display = denormalize(val_transformed).permute(1, 2, 0).clamp(0, 1)
    
    axes[1, 0].imshow(original_img)
    axes[1, 0].set_title("Original", fontsize=10)
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(val_display)
    axes[1, 1].set_title("Validation Transform\n(No Augmentation)", fontsize=10)
    axes[1, 1].axis('off')
    
    # Hide remaining subplots in row 2
    for i in range(2, num_versions + 1):
        axes[1, i].axis('off')
    
    plt.suptitle(f"Data Augmentation Visualization - Class: {class_name}", fontsize=14)
    plt.tight_layout()
    
    plt.tight_layout()
    
    # Save to figures directory
    save_path = os.path.join(config.FIGURES_DIR, "augmentation_comparison.png")
    plt.savefig(save_path, dpi=150)
    plt.show()
    
    print(f"✓ Saved to {save_path}")


# === MAIN EXECUTION ===
if __name__ == "__main__":
    print("=" * 50)
    print("STEP 2: PREPROCESSING & AUGMENTATION")
    print("=" * 50)
    
    DATA_DIR = config.DATA_DIR
    
    # Visualize augmentations
    print("\n🖼️ Visualizing augmentations...")
    visualize_augmentations(DATA_DIR, class_name="MC")
    
    # Test the dataloaders
    print("\n📦 Creating DataLoaders...")
    loaders, class_names = create_dataloaders(DATA_DIR)
    
    # Show one batch
    print("\n🔍 Testing one batch...")
    images, labels = next(iter(loaders['train']))
    print(f"Batch shape: {images.shape}")  # Should be [32, 3, 224, 224]
    print(f"Labels shape: {labels.shape}")  # Should be [32]
    print(f"Label values: {labels[:5]}")    # Class indices
    
    print("\n✅ Preprocessing complete!")