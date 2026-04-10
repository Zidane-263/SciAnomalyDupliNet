# SciAnomalyDupliNet — Google Colab Setup Guide

Since this model is lightweight (~8M params), it runs perfectly on Colab's T4 GPU (16GB VRAM), which is much faster than an RTX 3050.

---

## Step 1: Prepare Your Google Drive
1. Create a folder in your Google Drive named `SciAnomaly`.
2. Upload the `IF-Data` folder into `SciAnomaly/`.
3. Upload all the Python files (`*.py` and `requirements.txt`) into `SciAnomaly/`.

Your Drive structure should look like this:
```
My Drive/
└── SciAnomaly/
    ├── IF-Data/
    ├── checkpoints/  (will be created automatically)
    ├── config.py
    ├── dataset.py
    ├── model.py
    ├── losses.py
    ├── spectral_priors.py
    ├── train.py
    └── requirements.txt
```

---

## Step 2: Open a New Colab Notebook
1. Go to [colab.research.google.com](https://colab.research.google.com).
2. Click **New Notebook**.
3. Change Runtime to GPU: **Runtime > Change runtime type > T4 GPU**.

---

## Step 3: Run These Cells

### 3.1 Mount Google Drive
```python
from google.colab import drive
drive.mount('/content/drive')
```

### 3.2 Navigate to Project Folder
```python
import os
os.chdir('/content/drive/MyDrive/SciAnomaly')
!ls
```

### 3.3 Install Dependencies
```python
!pip install -r requirements.txt
```

### 3.4 Start Training
You can run the training directly as a shell command. 

```python
# Optional: Edit config.py directly in Colab by double-clicking it in the file browser


```

---

## Important Tips for Colab

### 1. Increase Batch Size
Since Colab's T4 has 16GB VRAM (4x more than your 3050), you can increase `BATCH_SIZE` in `config.py` to **16** or even **24**. This will make training much faster and more stable.

### 2. Disconnection Prevention
Colab might disconnect if you leave the tab. To prevent this, press `Ctrl+Shift+I` (Inspect), go to **Console**, and paste this:
```javascript
function ClickConnect(){
    console.log("Working"); 
    document.querySelector("colab-toolbar-button#connect").click() 
}
setInterval(ClickConnect, 60000)
```

### 3. Using `last_model.pth`
If Colab disconnects, just re-run the training cell. Because we implemented **Auto-Resume**, it will automatically find `checkpoints/last_model.pth` on your Drive and pick up exactly where it left off!

---

## 4. Hybrid Training: Move between Colab and Local
You can **absolutely** start training on Colab and finish locally (or vice-versa).

### To move from Colab to Local:
1. Download `checkpoints/best_model.pth` (or `last_model.pth`) from your Google Drive.
2. Put it in the `checkpoints/` folder on your computer.
3. Run `python predict.py` to test it locally.
4. Run `python train.py` to continue training locally.

### To move from Local to Colab:
1. Upload your local `checkpoints/` folder to your Google Drive (`SciAnomaly/checkpoints/`).
2. Run the Colab notebook. It will find the checkpoint and continue.

**Note:** The model architecture settings in `config.py` (like `ENCODER_NAME`, `FREQ_DIM`, etc.) MUST be the same on both machines for the checkpoint to load correctly. `BATCH_SIZE` and `IMG_SIZE` can be different.

---
