# Fingerprint Verification System

**Advanced Fingerprint Matching using Siamese / Triplet-Loss Network**

A complete Deep Learning project for **fingerprint verification** built with TensorFlow/Keras, featuring a professional GUI and comprehensive evaluation tools.

![Project Banner](https://via.placeholder.com/800x300/00D4AA/0F1117?text=Fingerprint+Verification+System)  
*(Replace with actual screenshot later)*

## 🎯 Project Overview

This project implements a **Siamese Neural Network** (with Triplet Loss) to verify whether two fingerprints belong to the same person. The system achieves strong performance on the **SOCOFing** dataset and includes both CLI and a beautiful desktop application.

## ✨ Features

- **🔍 Fingerprint Verification**: Compare any two fingerprints with similarity score
- **📊 Full Model Evaluation**: 500–1000 test pairs with balanced sampling
- **📈 Visualization**: Confusion Matrix, ROC Curve, and detailed results
- **🖥️ Professional GUI**: Built with **PySide6** (beautiful dark theme)
- **📁 Dual Support**: Works with both `.keras` and `.h5` models
- **⚡ Fast Inference**: Batch embedding computation
- **CLI + GUI**: Flexible usage options

## 🛠️ Technologies Used

- **Deep Learning**: TensorFlow / Keras
- **Computer Vision**: OpenCV
- **GUI**: PySide6 (with PyQt6 fallback)
- **Visualization**: Matplotlib + Seaborn style
- **Metrics**: scikit-learn
- **Data**: NumPy

## 📂 Dataset

- **SOCOFing** (Sokoto Coventry Fingerprint Dataset)
- 600 subjects, ~6000 fingerprint images
- Includes gender, hand, finger, and image quality metadata

## 🚀 Installation & Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/fingerprint-verification-system.git
cd fingerprint-verification-system
