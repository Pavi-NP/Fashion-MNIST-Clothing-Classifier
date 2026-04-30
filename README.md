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
