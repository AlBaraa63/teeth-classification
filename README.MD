# 🦷 Teeth Classification using Deep Learning

A computer vision solution for classifying dental conditions into 7 categories using a custom CNN built from scratch with residual connections.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2-orange)
![Accuracy](https://img.shields.io/badge/Accuracy-97.67%25-green)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Results](#results)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Training Details](#training-details)
- [References](#references)
- [Future Work](#future-work)

---

## 🎯 Overview

This project develops a comprehensive teeth classification solution that can accurately classify dental images into **7 distinct categories**. The model is built from scratch using PyTorch, incorporating techniques from landmark papers including **AlexNet** and **ResNet**.

### Key Features

- Custom CNN architecture with **residual connections** (inspired by ResNet)
- **Data augmentation** pipeline for improved generalization
- **Learning rate scheduling** for optimal training
- Comprehensive evaluation with confusion matrix and per-class metrics

### Dental Conditions Classified

| Code | Condition | Description |
|------|-----------|-------------|
| CaS | Calculus | Tartar buildup on teeth |
| CoS | Caries | Tooth decay / Cavities |
| Gum | Gum Disease | Periodontal conditions |
| MC | Mouth Cancer | Oral malignancies |
| OC | Oral Candidiasis | Fungal infection |
| OLP | Oral Lichen Planus | Inflammatory condition |
| OT | Oral Trauma | Physical injuries |

---

## 📊 Results

### Performance Summary

| Metric | Value |
|--------|-------|
| **Validation Accuracy** | 97.67% |
| **Test Accuracy** | 97%+ |
| **Total Parameters** | ~2.7M |
| **Training Time** | ~45 epochs |

### Per-Class Accuracy

| Class | Accuracy | Samples |
|-------|----------|---------|
| CoS (Caries) | 100.0% | 149 |
| MC (Mouth Cancer) | 99.4% | 180 |
| OT (Oral Trauma) | 97.7% | 131 |
| CaS (Calculus) | 97.5% | 160 |
| Gum (Gum Disease) | 97.5% | 120 |
| OLP (Oral Lichen Planus) | 93.9% | 180 |
| OC (Oral Candidiasis) | 92.6% | 108 |

### Training Curves

![Training History](outputs/figures/training_history2.png)

### Confusion Matrix

![Confusion Matrix](outputs/figures/confusion_matrix.png)

### Sample Predictions

![Sample Predictions](outputs/figures/sample_predictions.png)

---

## 📁 Dataset

### Structure

```
data/
├── Training/       # 3,087 images
│   ├── CaS/
│   ├── CoS/
│   ├── Gum/
│   ├── MC/
│   ├── OC/
│   ├── OLP/
│   └── OT/
├── Validation/     # 1,028 images
│   └── (same structure)
└── Testing/        # 1,028 images
    └── (same structure)
```

### Class Distribution

![Class Distribution](outputs/figures/class_distribution.png)

### Preprocessing Pipeline

1. **Resize**: All images resized to 224×224 pixels
2. **Augmentation** (training only):
   - Random horizontal flip
   - Random rotation (±15°)
   - Color jitter (brightness, contrast, saturation)
   - Random affine transformations
3. **Normalization**: ImageNet mean and std

---

## 🏗️ Model Architecture

### Overview

The model uses a **ResNet-inspired architecture** built from scratch with:

- Initial convolution layer (7×7, stride 2)
- 4 stages of residual blocks
- Global average pooling
- Dropout (0.5) for regularization
- Final fully connected layer (7 classes)

### Architecture Diagram

```
Input (3, 224, 224)
        │
        ▼
┌─────────────────────┐
│  Initial Conv 7×7   │  → (32, 56, 56)
│  + MaxPool          │
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  Stage 1            │  → (64, 56, 56)
│  2× Residual Blocks │
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  Stage 2            │  → (128, 28, 28)
│  2× Residual Blocks │
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  Stage 3            │  → (256, 14, 14)
│  2× Residual Blocks │
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  Stage 4            │  → (512, 7, 7)
│  2× Residual Blocks │
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  Global Avg Pool    │  → (512)
│  Dropout (0.5)      │
│  FC Layer           │  → (7)
└─────────────────────┘
        │
        ▼
    Output (7 classes)
```

### Residual Block

```
Input ──────────────────────┐
   │                        │
   ▼                        │
┌──────────┐                │
│ Conv 3×3 │                │
│ BatchNorm│                │
│ ReLU     │                │
└──────────┘                │
   │                        │
   ▼                        │
┌──────────┐                │
│ Conv 3×3 │                │
│ BatchNorm│                │
└──────────┘                │
   │                        │
   ▼                        │
   + ◄──────────────────────┘  (Skip Connection)
   │
   ▼
  ReLU
   │
   ▼
Output
```

### Key Techniques Used

| Technique | Source | Purpose |
|-----------|--------|---------|
| ReLU Activation | AlexNet | Faster training |
| Dropout (0.5) | AlexNet | Prevent overfitting |
| Batch Normalization | ResNet | Stable training |
| Residual Connections | ResNet | Enable deeper networks |
| Global Average Pooling | ResNet | Reduce parameters |
| Data Augmentation | AlexNet | Increase dataset variety |

---

## 🚀 Installation

### Requirements

- Python 3.10+
- NVIDIA GPU with CUDA support (recommended)

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/teeth-classification.git
cd teeth-classification
```

2. **Create virtual environment (optional)**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

### requirements.txt

```
torch>=2.0.0
torchvision>=0.15.0
matplotlib>=3.7.0
numpy>=1.24.0
pillow>=9.4.0
tqdm>=4.65.0
```

### Install PyTorch with CUDA

For GPU support, install PyTorch with CUDA:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

---

## 💻 Usage

### Training

```bash
python src/train.py
```

**Configuration** (edit in `train.py`):
```python
CONFIG = {
    'data_dir': 'data',
    'num_epochs': 50,
    'learning_rate': 0.001,
    'batch_size': 32,
    'patience': 10,  # Early stopping
}
```

### Evaluation

```bash
python src/evaluate.py
```

This generates:
- Test accuracy report
- Confusion matrix
- Per-class accuracy chart
- Sample predictions visualization

### Visualize Dataset

```bash
python src/dataset.py
```

This generates:
- Class distribution chart
- Augmentation examples
- Sample batch visualization

---

## 📂 Project Structure

```
teeth-classification/
│
├── data/                       # Dataset directory
│   ├── Training/
│   ├── Validation/
│   └── Testing/
│
├── src/                        # Source code
│   ├── config.py               # Configuration & constants
│   ├── dataset.py              # Data loading & augmentation
│   ├── model.py                # CNN architecture
│   ├── train.py                # Training loop
│   └── evaluate.py             # Evaluation & visualization
│
├── outputs/                    # Generated outputs
│   ├── figures/                # Plots and visualizations
│   │   ├── training_history.png
│   │   └── ...
│   ├── models/                 # Saved models
│   │   ├── best_model.pth
│   │   └── final_model.pth
│
├── requirements.txt            # Dependencies
└── README.md                   # This file
```

---

## 📈 Training Details

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| Initial Learning Rate | 0.001 |
| LR Scheduler | ReduceLROnPlateau |
| LR Reduction Factor | 0.5 |
| LR Patience | 5 epochs |
| Batch Size | 32 |
| Epochs | 50 (with early stopping) |
| Dropout Rate | 0.5 |
| Early Stopping Patience | 10 epochs |

### Data Augmentation

```python
transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, 
                          saturation=0.2, hue=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), 
                           scale=(0.9, 1.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])
```

### Training Progress

| Epoch | Train Acc | Val Acc | Notes |
|-------|-----------|---------|-------|
| 1 | 25% | 30% | Starting |
| 10 | 60% | 65% | Learning |
| 20 | 75% | 80% | Improving |
| 30 | 88% | 90% | Good progress |
| 45 | 95% | 97.67% | Best model saved |

---

## 📚 References

### Research Papers

1. **AlexNet** - Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). *ImageNet Classification with Deep Convolutional Neural Networks.* NeurIPS.
   - Key contributions: ReLU, Dropout, GPU training, Data augmentation

2. **ResNet** - He, K., Zhang, X., Ren, S., & Sun, J. (2016). *Deep Residual Learning for Image Recognition.* CVPR.
   - Key contributions: Skip connections, Batch normalization, Very deep networks

### Frameworks & Libraries

- [PyTorch](https://pytorch.org/) - Deep learning framework
- [torchvision](https://pytorch.org/vision/) - Computer vision utilities
- [matplotlib](https://matplotlib.org/) - Visualization

---

## 🔮 Future Work

Potential improvements for future iterations:

1. **Transfer Learning**: Use pretrained ResNet/EfficientNet for potentially higher accuracy
2. **Cross-Validation**: Implement k-fold cross-validation for more robust evaluation
3. **Class Imbalance**: Apply weighted loss or oversampling for minority classes
4. **Model Ensemble**: Combine multiple models for improved predictions
5. **Grad-CAM Visualization**: Show which regions the model focuses on
6. **Web Deployment**: Create a Flask/FastAPI app for real-time predictions
7. **Mobile Deployment**: Convert to ONNX/TensorFlow Lite for mobile use

---

## 👤 Author

**AlBaraa AlOlabi**

Computer Vision Engineering Intern at Cellula Technologies

---

## 📄 License

This project is part of an internship program at Cellula Technologies.

---

## 🙏 Acknowledgments

- Cellula Technologies for the internship opportunity
- Mentor guidance throughout the project
- Dataset providers for the dental images