# 👕 Fashion-MNIST Clothing Classifier

[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![TensorFlow 2.x](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Accuracy](https://img.shields.io/badge/Accuracy-83%25-brightgreen.svg)]()

A deep learning-based image classifier for the Fashion-MNIST dataset, achieving **83% test accuracy** with a custom CNN architecture. This project demonstrates fundamental computer vision concepts including data preprocessing, data augmentation, batch normalization, regularization techniques, and comprehensive model evaluation.

## 📋 Table of Contents
- [Dataset](#dataset)
- [Features](#features)
- [Architecture](#architecture)
- [Performance](#performance)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Technical Deep Dive](#technical-deep-dive)
- [Future Improvements](#future-improvements)
- [License](#license)

## 📊 Dataset

**Fashion-MNIST** is a drop-in replacement for the classic MNIST dataset, containing 70,000 grayscale images of clothing items:

| Split | Size | Description |
|-------|------|-------------|
| **Training** | 60,000 images | 28×28 grayscale |
| **Testing** | 10,000 images | 28×28 grayscale |
| **Classes** | 10 categories | Clothing types |

### Class Labels
0: T-shirt/top 1: Trouser 2: Pullover 3: Dress 4: Coat
5: Sandal 6: Shirt 7: Sneaker 8: Bag 9: Ankle boot


## ✨ Features

- **Complete CNN Pipeline**: End-to-end from data loading to evaluation
- **Data Augmentation**: Horizontal flips and rotations for better generalization
- **Regularization**: Batch normalization + Dropout (0.5) to prevent overfitting
- **Training Optimization**: Early stopping and learning rate scheduling
- **Comprehensive Evaluation**: Confusion matrix, training curves, per-class analysis
- **Visualization**: Sample predictions with true vs predicted labels

## 🏗️ Architecture



**Total Parameters**: ~1.2M

## 📈 Performance

### Key Metrics
- **Test Accuracy**: 83.0%
- **Human Baseline**: 83.5% (Zalando research)
- **Training Time**: ~5 minutes on GPU (30 epochs)
- **Inference Speed**: <1ms per image

### Per-Class Performance

| Class | Typical Accuracy | Common Confusions |
|-------|-----------------|-------------------|
| T-shirt/top | 85% | Shirt |
| Trouser | 98% | - |
| Pullover | 80% | Coat, Shirt |
| Dress | 88% | Coat |
| Coat | 82% | Pullover, Shirt |
| Sandal | 96% | - |
| Shirt | 72% | T-shirt, Pullover, Coat |
| Sneaker | 95% | Ankle boot |
| Bag | 97% | - |
| Ankle boot | 94% | Sneaker |

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/fashion-mnist-classifier.git
cd fashion-mnist-classifier

```
2. **Create virtual environment (optional but recommended)**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
3. **Install dependencies**
```bash
pip install -r requirements.txt
```

### 📊 Results

**Accuracy and Loss**
<img width="452" height="193" alt="image" src="https://github.com/user-attachments/assets/72151a4c-f69e-4d35-8888-ee174d123d69" />

**Confusion Matrix**
<img width="452" height="399" alt="image" src="https://github.com/user-attachments/assets/be3cf1c9-4fac-4e4e-be05-2153a54d78d7" />

**Model Summary and Test Accuracy**
<img width="452" height="330" alt="image" src="https://github.com/user-attachments/assets/6cf91595-3eda-4c53-b395-40e3327d316c" />



