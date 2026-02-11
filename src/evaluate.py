"""
evaluate.py - Evaluate the Trained Model

This script:
1. Loads the best trained model
2. Tests on the test set (unseen data)
3. Creates confusion matrix
4. Shows per-class accuracy
5. Displays sample predictions
"""

import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image

from . import config
from .dataset import create_dataloaders, denormalize
from .model import TeethClassifierImproved as TeethClassifier


# ============================================================
# CONFIGURATION
# ============================================================

CONFIG = {
    'data_dir': config.DATA_DIR,
    'model_path': os.path.join(config.MODELS_DIR, 'best_model.pth'),
    'device': config.DEVICE,
    'batch_size': config.BATCH_SIZE,
    'save_dir': config.FIGURES_DIR, # Saving plots to outputs/figures/
}

CLASS_NAMES = config.CLASSES
CLASS_FULL_NAMES = config.CLASS_FULL_NAMES


# ============================================================
# LOAD MODEL
# ============================================================

def load_model(model_path, num_classes, device):
    """
    Load a trained model from checkpoint.
    """
    print(f"📂 Loading model from: {model_path}")
    
    # Create model architecture
    model = TeethClassifier(num_classes=num_classes)
    
    # Load trained weights
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Move to device and set to evaluation mode
    model = model.to(device)
    model.eval()
    
    val_acc = checkpoint.get('val_acc', None)
    print(f"✓ Model loaded! (Val Acc: {f'{val_acc:.2f}%' if val_acc is not None else 'N/A'})")
    
    return model


# ============================================================
# EVALUATION FUNCTIONS
# ============================================================

def evaluate_model(model, dataloader, device):
    """
    Evaluate model on a dataset.
    
    Returns:
        all_predictions: predicted class indices
        all_labels: true class indices
        all_probs: prediction probabilities
    """
    
    model.eval()
    
    all_predictions = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc='Evaluating'):
            images = images.to(device)
            
            # Forward pass
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
    
    return np.array(all_predictions), np.array(all_labels), np.array(all_probs)


def calculate_metrics(predictions, labels, class_names):
    """
    Calculate various metrics.
    """
    
    # Overall accuracy
    accuracy = 100 * np.mean(predictions == labels)
    
    # Per-class accuracy
    class_correct = {}
    class_total = {}
    
    for cls_idx, cls_name in enumerate(class_names):
        mask = labels == cls_idx
        if mask.sum() > 0:
            class_correct[cls_name] = (predictions[mask] == labels[mask]).sum()
            class_total[cls_name] = mask.sum()
        else:
            class_correct[cls_name] = 0
            class_total[cls_name] = 0
    
    return accuracy, class_correct, class_total


# ============================================================
# CONFUSION MATRIX
# ============================================================

def plot_confusion_matrix(predictions, labels, class_names, save_path=None):
    if save_path is None:
        save_path = os.path.join(CONFIG['save_dir'], 'confusion_matrix.png')
    """
    Create and plot a confusion matrix.
    
    Rows: True labels
    Columns: Predicted labels
    
    Diagonal = correct predictions (should be high)
    Off-diagonal = mistakes (should be low)
    """
    
    num_classes = len(class_names)
    
    # Build confusion matrix
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for true, pred in zip(labels, predictions):
        cm[true][pred] += 1
    
    # Normalize by row (percentage of each true class)
    cm_normalized = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # --- Raw counts ---
    im1 = axes[0].imshow(cm, cmap='Blues')
    axes[0].set_xticks(range(num_classes))
    axes[0].set_yticks(range(num_classes))
    axes[0].set_xticklabels(class_names, rotation=45, ha='right')
    axes[0].set_yticklabels(class_names)
    axes[0].set_xlabel('Predicted', fontsize=12)
    axes[0].set_ylabel('True', fontsize=12)
    axes[0].set_title('Confusion Matrix (Counts)', fontsize=14)
    
    # Add text annotations
    for i in range(num_classes):
        for j in range(num_classes):
            color = 'white' if cm[i, j] > cm.max() / 2 else 'black'
            axes[0].text(j, i, str(cm[i, j]), ha='center', va='center', color=color, fontsize=10)
    
    plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
    
    # --- Normalized (percentages) ---
    im2 = axes[1].imshow(cm_normalized, cmap='Blues', vmin=0, vmax=100)
    axes[1].set_xticks(range(num_classes))
    axes[1].set_yticks(range(num_classes))
    axes[1].set_xticklabels(class_names, rotation=45, ha='right')
    axes[1].set_yticklabels(class_names)
    axes[1].set_xlabel('Predicted', fontsize=12)
    axes[1].set_ylabel('True', fontsize=12)
    axes[1].set_title('Confusion Matrix (Percentages)', fontsize=14)
    
    # Add text annotations
    for i in range(num_classes):
        for j in range(num_classes):
            color = 'white' if cm_normalized[i, j] > 50 else 'black'
            axes[1].text(j, i, f'{cm_normalized[i, j]:.1f}%', ha='center', va='center', color=color, fontsize=9)
    
    plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"✓ Confusion matrix saved to {save_path}")
    
    return cm


# ============================================================
# PER-CLASS ACCURACY
# ============================================================

def plot_per_class_accuracy(class_correct, class_total, save_path=None):
    if save_path is None:
        save_path = os.path.join(CONFIG['save_dir'], 'per_class_accuracy.png')
    """
    Plot accuracy for each class.
    """
    
    classes = list(class_correct.keys())
    accuracies = [100 * class_correct[c] / class_total[c] if class_total[c] > 0 else 0 for c in classes]
    totals = [class_total[c] for c in classes]
    
    # Sort by accuracy
    sorted_indices = np.argsort(accuracies)[::-1]
    classes = [classes[i] for i in sorted_indices]
    accuracies = [accuracies[i] for i in sorted_indices]
    totals = [totals[i] for i in sorted_indices]
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = ['#2ecc71' if acc >= 70 else '#f39c12' if acc >= 50 else '#e74c3c' for acc in accuracies]
    
    bars = ax.barh(range(len(classes)), accuracies, color=colors, edgecolor='black')
    
    # Add labels
    for i, (acc, total) in enumerate(zip(accuracies, totals)):
        ax.text(acc + 1, i, f'{acc:.1f}% ({total} samples)', va='center', fontsize=10)
    
    ax.set_yticks(range(len(classes)))
    ax.set_yticklabels([f"{c}\n({CLASS_FULL_NAMES.get(c, c)})" for c in classes])
    ax.set_xlabel('Accuracy (%)', fontsize=12)
    ax.set_title('Per-Class Accuracy (Test Set)', fontsize=14)
    ax.set_xlim(0, 110)
    ax.axvline(x=50, color='gray', linestyle='--', alpha=0.5, label='50% baseline')
    ax.legend()
    ax.grid(True, axis='x', alpha=0.3)
    
    # Color legend
    ax.text(0.98, 0.02, '🟢 ≥70%  🟡 50-70%  🔴 <50%', transform=ax.transAxes, 
            ha='right', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"✓ Per-class accuracy saved to {save_path}")


# ============================================================
# SAMPLE PREDICTIONS
# ============================================================

def show_sample_predictions(model, dataloader, class_names, device, 
                            num_samples=12, save_path=None):
    if save_path is None:
        save_path = os.path.join(CONFIG['save_dir'], 'sample_predictions.png')
    """
    Show sample predictions with images.
    
    Green border = correct
    Red border = wrong
    """
    
    model.eval()
    
    # Get one batch
    images, labels = next(iter(dataloader))
    images_gpu = images.to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(images_gpu)
        probs = torch.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs, 1)
    
    # Plot
    num_samples = min(num_samples, len(images))
    cols = 4
    rows = (num_samples + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
    axes = axes.flatten()
    
    for i in range(num_samples):
        # Denormalize image for display
        img = denormalize(images[i]).permute(1, 2, 0).clamp(0, 1).numpy()
        
        true_label = labels[i].item()
        pred_label = predicted[i].item()
        confidence = probs[i][pred_label].item() * 100
        
        true_name = class_names[true_label]
        pred_name = class_names[pred_label]
        
        # Determine if correct
        is_correct = true_label == pred_label
        border_color = 'green' if is_correct else 'red'
        
        # Plot image
        axes[i].imshow(img)
        axes[i].axis('off')
        
        # Add colored border
        for spine in axes[i].spines.values():
            spine.set_edgecolor(border_color)
            spine.set_linewidth(4)
            spine.set_visible(True)
        
        # Title
        if is_correct:
            title = f"✓ {pred_name}\n({confidence:.1f}%)"
            title_color = 'green'
        else:
            title = f"✗ Pred: {pred_name}\nTrue: {true_name}\n({confidence:.1f}%)"
            title_color = 'red'
        
        axes[i].set_title(title, fontsize=10, color=title_color)
    
    # Hide unused subplots
    for i in range(num_samples, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle('Sample Predictions (Green=Correct, Red=Wrong)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"✓ Sample predictions saved to {save_path}")


# ============================================================
# MOST CONFUSED PAIRS
# ============================================================

def show_most_confused(cm, class_names, top_n=5):
    """
    Show which class pairs are most confused.
    """
    
    print("\n" + "=" * 60)
    print("MOST CONFUSED CLASS PAIRS")
    print("=" * 60)
    
    # Find off-diagonal elements (mistakes)
    confusions = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j and cm[i][j] > 0:
                confusions.append({
                    'true': class_names[i],
                    'pred': class_names[j],
                    'count': cm[i][j]
                })
    
    # Sort by count
    confusions.sort(key=lambda x: x['count'], reverse=True)
    
    print(f"\nTop {top_n} confusions:\n")
    for i, conf in enumerate(confusions[:top_n]):
        true_full = CLASS_FULL_NAMES.get(conf['true'], conf['true'])
        pred_full = CLASS_FULL_NAMES.get(conf['pred'], conf['pred'])
        print(f"  {i+1}. {conf['true']} → {conf['pred']}: {conf['count']} times")
        print(f"      ({true_full} confused as {pred_full})")
        print()


# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == "__main__":
    
    print("\n" + "=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)
    
    device = torch.device(CONFIG['device'])
    print(f"\n🖥️ Device: {device}")
    
    # --- Load Data ---
    print("\n📂 Loading test data...")
    loaders, class_names = create_dataloaders(
        CONFIG['data_dir'],
        batch_size=CONFIG['batch_size']
    )
    test_loader = loaders['test']
    print(f"   Test samples: {len(test_loader.dataset)}")
    print(f"   Classes: {class_names}")
    
    # --- Load Model ---
    print("\n🧠 Loading trained model...")
    model = load_model(CONFIG['model_path'], num_classes=len(class_names), device=device)
    
    # --- Evaluate ---
    print("\n📊 Evaluating on test set...")
    predictions, labels, probs = evaluate_model(model, test_loader, device)
    
    # --- Calculate Metrics ---
    accuracy, class_correct, class_total = calculate_metrics(predictions, labels, class_names)
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"\n🎯 Overall Test Accuracy: {accuracy:.2f}%")
    print(f"   ({np.sum(predictions == labels)} / {len(labels)} correct)")
    
    # --- Per-class results ---
    print("\n📋 Per-Class Accuracy:")
    print("-" * 40)
    for cls_name in class_names:
        if class_total[cls_name] > 0:
            cls_acc = 100 * class_correct[cls_name] / class_total[cls_name]
            print(f"   {cls_name}: {cls_acc:.1f}% ({class_correct[cls_name]}/{class_total[cls_name]})")
    
    # --- Visualizations ---
    print("\n📈 Creating visualizations...")
    
    # Confusion Matrix
    cm = plot_confusion_matrix(predictions, labels, class_names)
    
    # Per-class accuracy
    plot_per_class_accuracy(class_correct, class_total)
    
    # Sample predictions
    show_sample_predictions(model, test_loader, class_names, device)
    
    # Most confused pairs
    show_most_confused(cm, class_names)
    
    # --- Final Summary ---
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE!")
    print("=" * 60)
    print(f"""
    📊 Results Summary:
    ─────────────────────────────────────
    Test Accuracy:     {accuracy:.2f}%
    Best Class:        {max(class_correct, key=lambda k: class_correct[k]/class_total[k] if class_total[k] > 0 else 0)}
    Needs Improvement: {min(class_correct, key=lambda k: class_correct[k]/class_total[k] if class_total[k] > 0 else 0)}
    
    📁 Files saved to '{CONFIG['save_dir']}/':
       - confusion_matrix.png
       - per_class_accuracy.png
       - sample_predictions.png
    """)