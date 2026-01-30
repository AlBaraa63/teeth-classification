"""
train.py - IMPROVED VERSION

Improvements over original:
1. More epochs (50 instead of 20)
2. Learning rate scheduler (reduces LR when stuck)
3. Early stopping (stops if no improvement)
4. Better monitoring and logging
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
from tqdm import tqdm

# Import our modules
from dataset import create_dataloaders, denormalize
from model import TeethClassifierImproved as TeethClassifier  # Use improved model


# ============================================================
# CONFIGURATION - IMPROVED SETTINGS
# ============================================================

CONFIG = {
    'data_dir': 'data',
    'num_epochs': 50,           # Increased from 20
    'learning_rate': 0.001,     # Starting learning rate
    'batch_size': 32,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'save_dir': 'outputs',
    'patience': 10,             # Early stopping: stop if no improvement for 10 epochs
}


# ============================================================
# TRAINING FUNCTIONS
# ============================================================

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    Train the model for ONE epoch.
    
    Steps:
    1. Set model to training mode
    2. For each batch:
       - Forward pass (predict)
       - Calculate loss (how wrong?)
       - Backward pass (calculate gradients)
       - Update weights (optimizer step)
    
    Returns:
        average loss, accuracy for this epoch
    """
    
    model.train()  # Training mode: dropout ON, batchnorm updates
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc='Training', leave=False)
    
    for images, labels in pbar:
        # Move to device (GPU/CPU)
        images = images.to(device)
        labels = labels.to(device)
        
        # Zero gradients (fresh start each batch)
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        
        # Calculate loss
        loss = criterion(outputs, labels)
        
        # Backward pass (calculate gradients)
        loss.backward()
        
        # Update weights
        optimizer.step()
        
        # Track statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Update progress bar
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    """
    Validate the model on unseen data.
    
    No weight updates here - just measuring performance.
    """
    
    model.eval()  # Evaluation mode: dropout OFF, batchnorm frozen
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():  # No gradient calculation needed
        for images, labels in tqdm(dataloader, desc='Validating', leave=False):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc


# ============================================================
# IMPROVED TRAINING LOOP
# ============================================================

def train_model(model, train_loader, val_loader, criterion, optimizer, 
                scheduler, num_epochs, device, save_dir):
    """
    Complete training loop with improvements:
    - Learning rate scheduler
    - Early stopping
    - Best model saving
    - Comprehensive logging
    """
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'lr': []
    }
    
    best_val_acc = 0.0
    patience_counter = 0
    
    print("=" * 60)
    print("TRAINING STARTED (IMPROVED VERSION)")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Max Epochs: {num_epochs}")
    print(f"Batch Size: {CONFIG['batch_size']}")
    print(f"Initial Learning Rate: {CONFIG['learning_rate']}")
    print(f"Early Stopping Patience: {CONFIG['patience']}")
    print("=" * 60)
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"\nEpoch {epoch + 1}/{num_epochs} (LR: {current_lr:.6f})")
        print("-" * 40)
        
        # --- TRAIN ---
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # --- VALIDATE ---
        val_loss, val_acc = validate(
            model, val_loader, criterion, device
        )
        
        # --- UPDATE SCHEDULER ---
        # Reduces LR if validation loss doesn't improve
        scheduler.step(val_loss)
        
        # --- RECORD HISTORY ---
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['lr'].append(current_lr)
        
        # --- PRINT RESULTS ---
        epoch_time = time.time() - epoch_start
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        print(f"Time: {epoch_time:.1f}s")
        
        # --- SAVE BEST MODEL ---
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0  # Reset patience
            
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
            }, os.path.join(save_dir, 'best_model.pth'))
            
            print(f"✓ New best model saved! (Val Acc: {val_acc:.2f}%)")
        else:
            patience_counter += 1
            print(f"  No improvement for {patience_counter}/{CONFIG['patience']} epochs")
        
        # --- EARLY STOPPING ---
        if patience_counter >= CONFIG['patience']:
            print(f"\n" + "=" * 60)
            print(f"⚠️ EARLY STOPPING at epoch {epoch + 1}")
            print(f"   No improvement for {CONFIG['patience']} consecutive epochs")
            print("=" * 60)
            break
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print(f"Total Epochs Run: {len(history['train_loss'])}")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    
    return history


# ============================================================
# VISUALIZATION
# ============================================================

def plot_training_history(history, save_path='outputs/training_history.png'):
    """
    Plot training curves with 3 graphs:
    1. Loss over time
    2. Accuracy over time
    3. Learning rate over time
    """
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # --- Plot 1: Loss ---
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Training', linewidth=2)
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Validation', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Loss Over Time', fontsize=14)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    axes[0].annotate('Lower is better ↓', xy=(0.7, 0.9), xycoords='axes fraction',
                     fontsize=10, color='green')
    
    # --- Plot 2: Accuracy ---
    axes[1].plot(epochs, history['train_acc'], 'b-', label='Training', linewidth=2)
    axes[1].plot(epochs, history['val_acc'], 'r-', label='Validation', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[1].set_title('Accuracy Over Time', fontsize=14)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    axes[1].annotate('Higher is better ↑', xy=(0.7, 0.1), xycoords='axes fraction',
                     fontsize=10, color='green')
    
    # --- Plot 3: Learning Rate ---
    axes[2].plot(epochs, history['lr'], 'g-', linewidth=2, marker='o', markersize=3)
    axes[2].set_xlabel('Epoch', fontsize=12)
    axes[2].set_ylabel('Learning Rate', fontsize=12)
    axes[2].set_title('Learning Rate Schedule', fontsize=14)
    axes[2].grid(True, alpha=0.3)
    axes[2].set_yscale('log')  # Log scale to see changes better
    axes[2].annotate('Decreases when stuck', xy=(0.5, 0.9), xycoords='axes fraction',
                     fontsize=10, color='green')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\n✓ Training history saved to {save_path}")


def print_training_summary(history):
    """Print a nice summary of training results."""
    
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    
    print(f"""
    📊 Final Results:
    ─────────────────────────────────────
    Final Train Accuracy:  {history['train_acc'][-1]:.2f}%
    Final Val Accuracy:    {history['val_acc'][-1]:.2f}%
    Best Val Accuracy:     {max(history['val_acc']):.2f}%
    
    📉 Loss:
    ─────────────────────────────────────
    Starting Loss:         {history['train_loss'][0]:.4f}
    Final Loss:            {history['train_loss'][-1]:.4f}
    
    📈 Learning Rate:
    ─────────────────────────────────────
    Starting LR:           {history['lr'][0]:.6f}
    Final LR:              {history['lr'][-1]:.6f}
    
    ⏱️ Epochs:
    ─────────────────────────────────────
    Total Epochs:          {len(history['train_loss'])}
    """)


# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == "__main__":
    
    print("\n" + "=" * 60)
    print("TEETH CLASSIFICATION - IMPROVED TRAINING")
    print("=" * 60)
    
    # --- Create output directory ---
    os.makedirs(CONFIG['save_dir'], exist_ok=True)
    
    # --- Set device ---
    device = torch.device(CONFIG['device'])
    print(f"\n🖥️ Using device: {device}")
    
    if device.type == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("   ⚠️ Running on CPU - this will be slower!")
    
    # --- Create data loaders ---
    print("\n📂 Loading data...")
    loaders, class_names = create_dataloaders(
        CONFIG['data_dir'], 
        batch_size=CONFIG['batch_size']
    )
    print(f"   Classes: {class_names}")
    
    # --- Create model ---
    print("\n🧠 Creating improved model...")
    model = TeethClassifier(num_classes=len(class_names))
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    # --- Loss function ---
    criterion = nn.CrossEntropyLoss()
    print(f"\n📉 Loss function: CrossEntropyLoss")
    
    # --- Optimizer ---
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    print(f"🎯 Optimizer: Adam (lr={CONFIG['learning_rate']})")
    
    # --- Learning Rate Scheduler ---
    # Reduces LR by 0.5 if val_loss doesn't improve for 5 epochs
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',        # Minimize loss
        factor=0.5,        # New LR = old LR * 0.5
        patience=5,        # Wait 5 epochs before reducing
        min_lr=1e-7        # Don't go below this
    )
    print(f"📉 Scheduler: ReduceLROnPlateau (factor=0.5, patience=5)")
    
    # --- Configuration Summary ---
    print("\n" + "-" * 60)
    print("Configuration:")
    print("-" * 60)
    for key, value in CONFIG.items():
        print(f"   {key}: {value}")
    print("-" * 60)
    
    # --- Start Training ---
    input("\n🚀 Press Enter to start training...")
    
    history = train_model(
        model=model,
        train_loader=loaders['train'],
        val_loader=loaders['val'],
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=CONFIG['num_epochs'],
        device=device,
        save_dir=CONFIG['save_dir']
    )
    
    # --- Plot Results ---
    plot_training_history(history)
    
    # --- Print Summary ---
    print_training_summary(history)
    
    # --- Save Final Model ---
    final_model_path = os.path.join(CONFIG['save_dir'], 'final_model.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'class_names': class_names,
        'history': history,
        'config': CONFIG,
    }, final_model_path)
    
    print(f"\n✅ Models saved to '{CONFIG['save_dir']}/':")
    print("   📁 best_model.pth  (highest validation accuracy)")
    print("   📁 final_model.pth (after all epochs)")
    print("\n🎉 Training complete! Ready for evaluation.")