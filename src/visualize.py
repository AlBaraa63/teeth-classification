"""
visualize.py - Understanding our dental dataset

GOAL: See what we're working with before training anything
"""

import os
import matplotlib.pyplot as plt
from PIL import Image
from collections import Counter

# === CONFIGURATION ===
# Using raw strings (r"...") to handle Windows backslashes
DATA_DIR = r"data/Training"  # Change this to your actual path

# The 7 classes we're classifying
CLASSES = ["CaS", "CoS", "Gum", "MC", "OC", "OLP", "OT"]


def count_images_per_class(data_dir):
    """
    Count how many images are in each class folder.
    
    WHY THIS MATTERS:
    - If one class has 1000 images and another has 50, the model
      will be biased toward the larger class (imbalanced data problem)
    - We need to know this BEFORE training
    """
    counts = {}
    
    for class_name in CLASSES:
        class_path = os.path.join(data_dir, class_name)
        
        if os.path.exists(class_path):
            # Count only image files (not hidden files or folders)
            images = [f for f in os.listdir(class_path) 
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            counts[class_name] = len(images)
        else:
            print(f"Warning: {class_path} not found!")
            counts[class_name] = 0
    
    return counts


def plot_class_distribution(counts, save_path="outputs/class_distribution.png"):
    """
    Create a bar chart showing images per class.
    
    THIS IS YOUR FIRST DELIVERABLE for the internship!
    """
    plt.figure(figsize=(10, 6))
    
    classes = list(counts.keys())
    values = list(counts.values())
    
    # Color bars based on count (visual indicator of imbalance)
    colors = ['#2ecc71' if v > 100 else '#e74c3c' for v in values]
    
    bars = plt.bar(classes, values, color=colors, edgecolor='black')
    
    # Add count labels on top of each bar
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                str(value), ha='center', va='bottom', fontsize=12)
    
    plt.xlabel("Tooth Condition Class", fontsize=12)
    plt.ylabel("Number of Images", fontsize=12)
    plt.title("Class Distribution in Training Set", fontsize=14)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Create outputs folder if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.show()
    
    print(f"✓ Saved to {save_path}")


def show_sample_images(data_dir, num_samples=1):
    """
    Display one sample image from each class.
    
    WHY THIS MATTERS:
    - See what the model will actually see
    - Check for image quality issues
    - Understand the visual differences between classes
    """
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for idx, class_name in enumerate(CLASSES):
        class_path = os.path.join(data_dir, class_name)
        
        # Get first image in the folder
        images = [f for f in os.listdir(class_path) 
                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if images:
            img_path = os.path.join(class_path, images[0])
            img = Image.open(img_path)
            
            axes[idx].imshow(img)
            axes[idx].set_title(f"{class_name}\nSize: {img.size}", fontsize=10)
            axes[idx].axis('off')
    
    # Hide the 8th subplot (we only have 7 classes)
    axes[7].axis('off')
    
    plt.suptitle("Sample Images from Each Class", fontsize=14)
    plt.tight_layout()
    plt.savefig("outputs/sample_images.png", dpi=150)
    plt.show()


# === MAIN EXECUTION ===
if __name__ == "__main__":
    print("=" * 50)
    print("STEP 1: DATASET EXPLORATION")
    print("=" * 50)
    
    # Count images
    print("\n📊 Counting images per class...")
    counts = count_images_per_class(DATA_DIR)
    
    print("\nImage counts:")
    for class_name, count in counts.items():
        print(f"  {class_name}: {count} images")
    
    print(f"\nTotal: {sum(counts.values())} images")
    
    # Plot distribution
    print("\n📈 Creating class distribution chart...")
    plot_class_distribution(counts)
    
    # Show samples
    print("\n🖼️ Displaying sample images...")
    show_sample_images(DATA_DIR)
    
    print("\n✅ Exploration complete! Check the 'outputs' folder.")