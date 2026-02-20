"""
transfer_model.py - Transfer Learning Models using Pre-trained Networks

This module provides transfer learning functionality using pre-trained models
from torchvision (ResNet, EfficientNet, etc.).

Transfer Learning Benefits:
- Faster convergence (pre-trained features)
- Better performance with limited data
- Leverages knowledge from ImageNet
"""

import torch
import torch.nn as nn
from torchvision import models


def get_transfer_model(model_name='resnet18', num_classes=7, pretrained=True, freeze_features=True):
    """
    Create a transfer learning model.
    
    Args:
        model_name: Name of the pre-trained model ('resnet18', 'resnet50', 'efficientnet_b0')
        num_classes: Number of output classes
        pretrained: Whether to load pre-trained ImageNet weights
        freeze_features: Whether to freeze the feature extractor (only train classifier)
    
    Returns:
        model: PyTorch model ready for training
    """
    
    if model_name == 'resnet18':
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)
        
    elif model_name == 'resnet50':
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)
        
    elif model_name == 'efficientnet_b0':
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
        num_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_features, num_classes)
        
    else:
        raise ValueError(f"Unknown model: {model_name}. Choose from: resnet18, resnet50, efficientnet_b0")
    
    # Freeze feature extractor if requested
    if freeze_features and pretrained:
        freeze_feature_extractor(model, model_name)
    
    return model


def freeze_feature_extractor(model, model_name):
    """
    Freeze all layers except the final classifier.
    
    This is useful for initial training - we only train the new classifier head
    while keeping the pre-trained features frozen.
    """
    if 'resnet' in model_name:
        # Freeze all layers except fc
        for name, param in model.named_parameters():
            if 'fc' not in name:
                param.requires_grad = False
    
    elif 'efficientnet' in model_name:
        # Freeze all layers except classifier
        for name, param in model.named_parameters():
            if 'classifier' not in name:
                param.requires_grad = False


def unfreeze_all_layers(model):
    """
    Unfreeze all layers for fine-tuning.
    
    After initial training with frozen features, you can unfreeze all layers
    and continue training with a lower learning rate.
    """
    for param in model.parameters():
        param.requires_grad = True


def count_parameters(model):
    """Count total and trainable parameters."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


# ============================================================
# MAIN - TEST TRANSFER MODELS
# ============================================================

if __name__ == "__main__":
    
    print("\n" + "=" * 60)
    print("TESTING TRANSFER LEARNING MODELS")
    print("=" * 60)
    
    models_to_test = ['resnet18', 'resnet50', 'efficientnet_b0']
    
    for model_name in models_to_test:
        print(f"\n📦 {model_name.upper()}")
        print("-" * 40)
        
        # Create model with frozen features
        model = get_transfer_model(model_name, num_classes=7, pretrained=True, freeze_features=True)
        total, trainable = count_parameters(model)
        
        print(f"Total parameters:     {total:,}")
        print(f"Trainable parameters: {trainable:,}")
        print(f"Frozen parameters:    {total - trainable:,}")
        print(f"Trainable ratio:      {100 * trainable / total:.2f}%")
        
        # Test forward pass
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            output = model(x)
        print(f"Output shape:         {list(output.shape)}")
    
    print("\n" + "=" * 60)
    print("✅ All transfer models work correctly!")
    print("=" * 60)
