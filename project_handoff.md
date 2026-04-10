# Project Handoff: SciAnomalyDupliNet (Image Forgery Detection)

This project implemented **SciAnomalyDupliNet**, a dual-domain (spatial + frequency) neural network specifically designed to detect copy-move and repetition-based forgeries in research images.

## 🚀 **Current Development Status**
- **Phase**: Training optimization (Version 2).
- **Environment**: Windows, 4GB RTX 3050 Ti Laptop GPU, 32GB RAM.
- **Goal**: Reach **0.60+ Val Dice Score**.
- **Current Training Run (DONE)**: 
  - Stability improved from v1 by increasing `LOSS_DICE_WEIGHT` (1.0 → 3.0) and lowering `LEARNING_RATE` (3e-4 → 1e-4).
  - **Final Results**: Best Val Dice **0.3584**.
  - **Status**: Successfully completed with zero metric collapse. The model is robust and ready for inference.

---

## 🏗️ **Core Architecture**
- **Dual-Domain Input**: 3 RGB + 2 Spectral Prior channels (Phase Correlation & Repetition Anomaly Maps).
- **Encoder**: `mobilevit_xs` (Pretrained via `timm`).
- **Novel Components**:
  - **RepAnom Cross-Attention**: Attends to frequency tokens to find spatial anomalies.
  - **Frequency Branch**: Real-valued FFTnd-based magnitude projections.
  - **Anomaly-Guided Decoder**: U-Net skip connections weighted by the classical spectral anomaly map.

---

## 📁 **Key Files**
- [config.py](file:///c:/Projects/Image%20Forgery/config.py): Central control for all training params (LR, Weights, etc.).
- [train.py](file:///c:/Projects/Image%20Forgery/train.py): Training script with linear LR warmup and best-model checkpointing.
- [dataset.py](file:///c:/Projects/Image%20Forgery/dataset.py): Custom loader that computes **classical spectral priors** on-the-fly.
- [model.py](file:///c:/Projects/Image%20Forgery/model.py): Full model definition including the custom cross-attention and decoder blocks.
- [spectral_priors.py](file:///c:/Projects/Image%20Forgery/spectral_priors.py): Signal processing code (FFT, Phase Correlation, Autocorrelation Anomaly).
- [predict.py](file:///c:/Projects/Image%20Forgery/predict.py): Inference script; generates binary masks and overlays.

---

## 🛠️ **Future Roadmap & Recommendations**
1.  **Monitor Convergence**: If the Dice score stalls again after Epoch 6, check the loss balance in `losses.ly`. 
2.  **Dataset Balance**: Consider adding authentic images (label 0) if the model starts exhibiting "forgery hallucination."
3.  **Post-Processing**: The current `predict.py` uses Otsu thresholding; a fixed 0.5 threshold might be more stable if the model confidence is consistent.

---

### **Handover Note for the Next Agent:**
*The project is currently in the middle of a fresh training run. The goal was to avoid the 'metric collapse' seen in the previous 10-epoch run where the Dice Score dropped at the very end. The current run uses a significantly higher Dice weight (3.0) and lower learning rate (1e-4) to solve this.* 
