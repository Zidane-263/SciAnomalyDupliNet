"""
eda_dataset.py
--------------
Exploratory Data Analysis script to verify the dataset before training.
- Check image/mask counts
- Check mask shapes and values
- Visualise a few samples (image + mask + spectral prior)
"""

import os
import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

from dataset import ForgedImageDataset


def main():
    root = "IF-Data"
    print(f"--- EDA for {root} ---")

    # 1. Basic Counts
    auth_imgs = glob.glob(os.path.join(root, "train_images", "authentic", "*.png"))
    forged_imgs = glob.glob(os.path.join(root, "train_images", "forged", "*.png"))
    masks = glob.glob(os.path.join(root, "train_masks", "*.npy"))
    
    print(f"Authentic images: {len(auth_imgs)}")
    print(f"Forged images:    {len(forged_imgs)}")
    print(f"Total masks:      {len(masks)}")

    # 2. Dataset Test
    ds = ForgedImageDataset(root, split="train", img_size=512)
    if len(ds) == 0:
        print("❌ Dataset is empty! Check folder structure.")
        return

    # 3. Sample Statistics
    print("\n--- Analysing Mask Content ---")
    pos_pixel_ratios = []
    shapes = []
    
    # Sample 100 images
    indices = np.random.choice(len(ds), min(len(ds), 100), replace=False)
    for idx in tqdm(indices):
        item = ds[idx]
        mask = item["mask"].numpy()
        pos_pixel_ratios.append(np.mean(mask))
        shapes.append(item["image"].shape[1:])

    print(f"Avg forgery area: {np.mean(pos_pixel_ratios)*100:.2f}% of pixels")
    print(f"Typical shape:    {shapes[0]}")

    # 4. Visualisation
    print("\nGenerating sample visualisations (eda_samples.png)...")
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    
    for i in range(3):
        # Fresh item for each row
        item = ds[np.random.randint(len(ds))]
        img = item["image"].permute(1, 2, 0).numpy()
        # Denormalise for vis
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img = np.clip(img, 0, 1)
        
        mask = item["mask"].squeeze().numpy()
        prior_pc = item["spectral_prior"][0].numpy()
        prior_anom = item["spectral_prior"][1].numpy()

        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f"Image ({item['img_id']})")
        
        axes[i, 1].imshow(mask, cmap="gray")
        axes[i, 1].set_title("GT Mask")
        
        axes[i, 2].imshow(prior_pc, cmap="hot")
        axes[i, 2].set_title("Spectral Prior: Phase Corr")
        
        axes[i, 3].imshow(prior_anom, cmap="jet")
        axes[i, 3].set_title("Spectral Prior: Anomaly Score")

    plt.tight_layout()
    plt.savefig("eda_samples.png")
    print("Done. Saved eda_samples.png")


if __name__ == "__main__":
    main()
