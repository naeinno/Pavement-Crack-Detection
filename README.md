# Pavement Crack Detection
A hybrid deep learning project using Vision Transformer and CNN architectures to detect concrete and pavement surface cracks.

## Overview
This repository contains an end-to-end computer vision pipeline for automated crack detection on concrete and pavement surfaces. The project implements a hybrid deep learning architecture combining Convolutional Neural Networks (CNN) and Vision Transformer (ViT) concepts with multi-head self-attention mechanisms in TensorFlow to classify surface imagery into crack and non-crack categories.

Model interpretability is incorporated using SHAP to visualize feature attributions and highlight regions of interest during inference.

## Key Capabilities
- Hybrid architecture leveraging CNN feature extraction and Vision Transformer self-attention.
- High-throughput image processing pipeline for binary classification.
- Automated hyperparameter optimization using KerasTuner.
- Explainable AI integration with SHAP for spatial feature visualization.
- Automated dataset downloading and integration via the Kagglehub API.

## Dataset
This project uses the Concrete and Pavement Crack Images dataset.
- Dataset size: 30,000 RGB images
- Class distribution: 15,000 Positive (Crack) and 15,000 Negative (Non-crack)
- Resolution: 128x128 pixels
- Split ratio: 80% Training and 20% Testing

## Environment and Dependencies
The project is built using Python 3 and TensorFlow. Primary libraries required:
- tensorflow
- keras-tuner
- shap
- visualkeras
- opencv-python
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- kagglehub

## Project Structure and Workflow
1. Data Acquisition: Images are fetched directly from Kaggle via kagglehub.
2. Preprocessing: Resizing images to 128x128 and normalizing pixel intensity values.
3. Architecture: CNN feature extraction layers, Image Patch Projection, Positional Embedding, Multi-Head Self-Attention layers, Dense MLP classification head.
4. Optimization: Hyperparameter tuning via KerasTuner RandomSearch.
5. Evaluation and Explainability: Performance evaluation via confusion matrix and classification reports, followed by SHAP feature attribution generation.

## Model Baseline Parameters
- Input Shape: 128 x 128 x 3
- Batch Size: 64
- Projection Dimension: 64
- Attention Heads: 4
- Transformer Layers: 8
- MLP Head Units: 2048, 1024
- Optimizer: AdamW with Learning Rate 0.001 and Weight Decay 0.0001
