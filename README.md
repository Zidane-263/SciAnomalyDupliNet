# SciAnomalyDupliNet: Scientific Image Forgery Detection

SciAnomalyDupliNet is a state-of-the-art hybrid Vision Transformer and CNN architecture specifically engineered to detect complex **copy-move** and **repetition-based forgeries** in scientific research images.

## 🚀 Overview

This framework utilizes a dual-domain (spatial + frequency) anomaly detection approach. By fusing classical signal processing spectral techniques (Phase Correlation & Autocorrelation) with modern deep learning feature extraction, it pinpoints replicated image regions that standard computer vision approaches often fail to detect.

## 🏗️ Core Architecture
- **Dual-Domain Input**: Computes 3 RGB channels fused with 2 computed Spectral Prior channels.
- **Encoder Backbone**: MobileViT (`mobilevit_xs`) for efficient spatial feature extraction.
- **Novel RepAnom Cross-Attention Module**: A custom transformer layer that queries frequency domain tokens to uncover spatial discrepancies.
- **Anomaly-Guided Decoder**: Modified U-Net architecture featuring skip connections that are explicitly weighted by classical spectral anomaly maps.

## 📁 Repository Structure
- `config.py`: Centralized control hub for training hyperparameters, paths, and configurations.
- `dataset.py`: Custom PyTorch `Dataset` logic that computes spectral priors on-the-fly.
- `losses.py`: Custom implemented loss functions including high-weight Dice implementations.
- `model.py`: The full defining architecture of SciAnomalyDupliNet.
- `predict.py`: Inference script to process unseen images and output binary predictions.
- `spectral_priors.py`: Core mathematical signal processing codes including 2D FFT logic.
- `train.py`: Primary orchestration script outlining the training steps and stability loops.
- `visualize.py`: Visual plotting and output saving logic.

## 🛠️ Getting Started

### 1. Installation
Clone the repository and install the dependencies:
```bash
git clone https://github.com/Zidane-263/SciAnomalyDupliNet.git
cd SciAnomalyDupliNet
pip install -r requirements.txt
```

### 2. Training
Check `config.py` to ensure dataset paths align with your setup, then start your run:
```bash
python train.py
```

### 3. Inference
To generate masks and forgery heatmaps on new images:
```bash
python predict.py
```

## 📊 Evaluation & Stability Details
The network is uniquely configured with an aggressive Dice loss ratio (e.g., `LOSS_DICE_WEIGHT=3.0`, `LEARNING_RATE=1e-4`) which has proven highly successful in guarding against "metric collapse"—a common phenomenon in scientific image forgery where the network naturally defaults to predicting empty masks due to imbalanced authentic background ratios.

---
*Created for robust open-source scientific integrity evaluation.*
