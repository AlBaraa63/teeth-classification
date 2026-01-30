"""
model.py - CNN Architectures for Teeth Classification

Contains:
1. TeethClassifier - Original simple CNN
2. TeethClassifierImproved - With residual connections (like ResNet!)

The improved version uses skip connections from your ResNet paper!
"""

import torch
import torch.nn as nn


# ============================================================
# ORIGINAL SIMPLE MODEL
# ============================================================

class ConvBlock(nn.Module):
    """Simple Conv → BatchNorm → ReLU → Pool block."""
    
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
    
    def forward(self, x):
        return self.block(x)


class TeethClassifier(nn.Module):
    """Original simple CNN (4 conv blocks)."""
    
    def __init__(self, num_classes=7, dropout_rate=0.5):
        super(TeethClassifier, self).__init__()
        
        self.features = nn.Sequential(
            ConvBlock(3, 32),     # 224 → 112
            ConvBlock(32, 64),    # 112 → 56
            ConvBlock(64, 128),   # 56 → 28
            ConvBlock(128, 256),  # 28 → 14
        )
        
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, num_classes)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        x = self.features(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        return x


# ============================================================
# IMPROVED MODEL WITH RESIDUAL CONNECTIONS
# ============================================================

class ResidualBlock(nn.Module):
    """
    Residual block with skip connection (from ResNet!).
    
    Instead of learning H(x) directly, we learn F(x) = H(x) - x
    Then output = F(x) + x
    
    This is the KEY INNOVATION from the ResNet paper you studied!
    
    Why it works:
    - Gradients flow directly through skip connection
    - Easier to learn "refinements" than full transformation
    - Enables training of much deeper networks
    """
    
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        
        # Main path: Conv → BN → ReLU → Conv → BN
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, 
            kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Skip connection (shortcut)
        # If dimensions change, we need a 1x1 conv to match them
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        # Save input for skip connection
        identity = x
        
        # Main path
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Add skip connection (THIS IS THE MAGIC!)
        out += self.shortcut(identity)
        
        # Final activation
        out = self.relu(out)
        
        return out


class TeethClassifierImproved(nn.Module):
    """
    Improved CNN with residual connections.
    
    Architecture inspired by ResNet:
    - Initial convolution to extract basic features
    - 4 stages of residual blocks (progressively deeper features)
    - Global average pooling (reduces parameters dramatically)
    - Dropout + final classification layer
    
    Key differences from simple model:
    - Skip connections help gradients flow
    - Deeper network (more capacity to learn)
    - Better weight initialization
    """
    
    def __init__(self, num_classes=7, dropout_rate=0.5):
        super(TeethClassifierImproved, self).__init__()
        
        # === INITIAL CONVOLUTION ===
        # Quickly reduce spatial size while extracting basic features
        self.initial = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        # Input: 224×224 → Output: 56×56
        
        # === RESIDUAL STAGES ===
        # Each stage has multiple residual blocks
        # Channels increase, spatial size decreases
        
        self.stage1 = self._make_stage(
            in_channels=32, out_channels=64, 
            num_blocks=2, stride=1
        )  # 56×56 → 56×56
        
        self.stage2 = self._make_stage(
            in_channels=64, out_channels=128,
            num_blocks=2, stride=2
        )  # 56×56 → 28×28
        
        self.stage3 = self._make_stage(
            in_channels=128, out_channels=256,
            num_blocks=2, stride=2
        )  # 28×28 → 14×14
        
        self.stage4 = self._make_stage(
            in_channels=256, out_channels=512,
            num_blocks=2, stride=2
        )  # 14×14 → 7×7
        
        # === CLASSIFIER ===
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))  # 7×7 → 1×1
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc = nn.Linear(512, num_classes)
        
        # Initialize weights properly
        self._initialize_weights()
    
    def _make_stage(self, in_channels, out_channels, num_blocks, stride):
        """
        Create a stage with multiple residual blocks.
        
        First block may downsample (stride=2), rest keep same size.
        """
        layers = []
        
        # First block (may change dimensions)
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        
        # Remaining blocks (same dimensions)
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, stride=1))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """
        Initialize weights using Kaiming initialization.
        
        This is important for training deep networks!
        - Conv layers: Kaiming normal (designed for ReLU)
        - BatchNorm: weight=1, bias=0
        - Linear: Kaiming normal
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Input:  (batch, 3, 224, 224)
        Output: (batch, num_classes)
        """
        # Initial convolution
        x = self.initial(x)      # (B, 32, 56, 56)
        
        # Residual stages
        x = self.stage1(x)       # (B, 64, 56, 56)
        x = self.stage2(x)       # (B, 128, 28, 28)
        x = self.stage3(x)       # (B, 256, 14, 14)
        x = self.stage4(x)       # (B, 512, 7, 7)
        
        # Classifier
        x = self.global_pool(x)  # (B, 512, 1, 1)
        x = x.view(x.size(0), -1)  # (B, 512)
        x = self.dropout(x)
        x = self.fc(x)           # (B, num_classes)
        
        return x


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def count_parameters(model):
    """Count total trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_summary(model, input_size=(3, 224, 224)):
    """Print a summary of the model architecture."""
    
    print("=" * 60)
    print("MODEL SUMMARY")
    print("=" * 60)
    
    print(f"\nModel: {model.__class__.__name__}")
    print(f"Parameters: {count_parameters(model):,}")
    
    # Test forward pass
    x = torch.randn(1, *input_size)
    
    print(f"\nInput shape:  {list(x.shape)}")
    
    with torch.no_grad():
        output = model(x)
    
    print(f"Output shape: {list(output.shape)}")
    print("=" * 60)


# ============================================================
# MAIN - TEST BOTH MODELS
# ============================================================

if __name__ == "__main__":
    
    print("\n" + "=" * 60)
    print("TESTING BOTH MODEL ARCHITECTURES")
    print("=" * 60)
    
    # Test original model
    print("\n📦 ORIGINAL MODEL (Simple CNN)")
    print("-" * 40)
    model_simple = TeethClassifier(num_classes=7)
    model_summary(model_simple)
    
    # Test improved model
    print("\n📦 IMPROVED MODEL (With Residual Connections)")
    print("-" * 40)
    model_improved = TeethClassifierImproved(num_classes=7)
    model_summary(model_improved)
    
    # Compare
    params_simple = count_parameters(model_simple)
    params_improved = count_parameters(model_improved)
    
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)
    print(f"""
    Original Model:     {params_simple:,} parameters
    Improved Model:     {params_improved:,} parameters
    
    Improved model is {params_improved / params_simple:.1f}x larger,
    but has skip connections that help it learn better!
    
    For reference:
    - AlexNet:  ~60,000,000 parameters
    - ResNet-18: ~11,000,000 parameters
    - Our model: ~{params_improved:,} parameters (much smaller!)
    """)
    
    print("✅ Both models work correctly!")