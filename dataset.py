"""
dataset.py
----------
PyTorch Dataset for Scientific Image Forgery Detection.

Directory layout expected:
  IF-Data/
    train_images/
      authentic/   *.png   (label=0, no mask needed)
      forged/      *.png   (label=1, mask in train_masks/<id>.npy)
    train_masks/   *.npy
    supplemental_images/  *.png  (optional extra, treated as forged if mask exists)
    test_images/   *.png
"""

import os
import glob
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

from spectral_priors import compute_spectral_prior


# ─────────────────────────────────────────────────────────────────────────────
# Augmentation pipelines
# ─────────────────────────────────────────────────────────────────────────────

def get_train_transforms(img_size: int = 512):
    return A.Compose([
        A.LongestMaxSize(max_size=img_size),
        A.PadIfNeeded(min_height=img_size, min_width=img_size, border_mode=cv2.BORDER_CONSTANT),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.Rotate(limit=15, border_mode=cv2.BORDER_CONSTANT, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, p=0.3),
        A.GaussNoise(std_range=(0.01, 0.05), p=0.2),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def get_val_transforms(img_size: int = 512):
    return A.Compose([
        A.LongestMaxSize(max_size=img_size),
        A.PadIfNeeded(min_height=img_size, min_width=img_size, border_mode=cv2.BORDER_CONSTANT),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# Dataset class
# ─────────────────────────────────────────────────────────────────────────────

class ForgedImageDataset(Dataset):
    """
    Returns:
        image:          (3, H, W) float32 — normalized RGB
        spectral_prior: (2, H, W) float32 — phase corr + anomaly
        mask:           (1, H, W) float32 — binary forgery mask (0 or 1)
        img_id:         str
    """

    def __init__(self,
                 root: str = "IF-Data",
                 split: str = "train",
                 img_size: int = 512,
                 use_supplemental: bool = True,
                 val_ids: list = None):
        super().__init__()
        self.root = root
        self.img_size = img_size
        self.split = split

        mask_dir = os.path.join(root, "train_masks")

        # ---- Collect forged samples (image + mask pairs) ----
        samples = []
        forged_dir = os.path.join(root, "train_images", "forged")
        for img_path in sorted(glob.glob(os.path.join(forged_dir, "*.png"))):
            img_id = os.path.splitext(os.path.basename(img_path))[0]
            mask_path = os.path.join(mask_dir, f"{img_id}.npy")
            if os.path.exists(mask_path):
                samples.append((img_path, mask_path, img_id))

        # ---- Supplemental images (if mask exists) ----
        if use_supplemental:
            sup_dir = os.path.join(root, "supplemental_images")
            if os.path.isdir(sup_dir):
                for img_path in sorted(glob.glob(os.path.join(sup_dir, "*.png"))):
                    img_id = os.path.splitext(os.path.basename(img_path))[0]
                    mask_path = os.path.join(mask_dir, f"{img_id}.npy")
                    if os.path.exists(mask_path):
                        samples.append((img_path, mask_path, img_id))

        # ---- Train / val split ----
        if val_ids is not None:
            if split == "val":
                samples = [s for s in samples if s[2] in val_ids]
            else:
                samples = [s for s in samples if s[2] not in val_ids]

        self.samples = samples
        self.transforms = (get_train_transforms(img_size)
                           if split == "train"
                           else get_val_transforms(img_size))

        print(f"[Dataset] split={split}, samples={len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path, img_id = self.samples[idx]

        # ---- Load image ----
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            raise FileNotFoundError(f"Cannot load image: {img_path}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # ---- Compute spectral prior (on resized copy, fast) ----
        # Use original BGR image; compute_spectral_prior handles resize internally
        spectral_prior = compute_spectral_prior(img_bgr, out_size=(self.img_size, self.img_size))
        # spectral_prior: (2, H, W) float32

        # ---- Load and preprocess mask ----
        mask_raw = np.load(mask_path, allow_pickle=True)
        mask = self._process_mask(mask_raw, img_bgr.shape[:2])

        # ---- Spatial augmentations (image + mask together) ----
        augmented = self.transforms(image=img_rgb, mask=mask)
        img_tensor = augmented["image"]       # (3, H, W)
        mask_tensor = augmented["mask"]       # (H, W)

        mask_tensor = mask_tensor.unsqueeze(0).float()           # (1, H, W)
        spectral_prior_tensor = torch.from_numpy(spectral_prior) # (2, H, W)

        return {
            "image": img_tensor,
            "spectral_prior": spectral_prior_tensor,
            "mask": mask_tensor,
            "img_id": img_id,
        }

    def _process_mask(self, mask_raw: np.ndarray, orig_shape: tuple) -> np.ndarray:
        """Normalise mask to uint8 H×W in {0,1}."""
        mask = np.asarray(mask_raw)

        # Flatten extra dims if needed
        if mask.ndim == 3:
            mask = mask[..., 0]   # take first channel
        elif mask.ndim > 3:
            mask = mask.reshape(mask.shape[-2], mask.shape[-1])

        # Convert to float, threshold
        mask = mask.astype(np.float32)
        if mask.max() > 1.0:
            mask = mask / 255.0
        mask = (mask > 0.5).astype(np.uint8)

        # Resize to match image
        H_orig, W_orig = orig_shape
        if mask.shape != (H_orig, W_orig):
            mask = cv2.resize(mask, (W_orig, H_orig), interpolation=cv2.INTER_NEAREST)

        return mask


# ─────────────────────────────────────────────────────────────────────────────
# Test dataset (no masks)
# ─────────────────────────────────────────────────────────────────────────────

class TestImageDataset(Dataset):
    def __init__(self, root: str = "IF-Data", img_size: int = 512):
        self.img_size = img_size
        test_dir = os.path.join(root, "test_images")
        self.paths = sorted(glob.glob(os.path.join(test_dir, "*.png")))
        self.transforms = get_val_transforms(img_size)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img_path = self.paths[idx]
        img_id = os.path.splitext(os.path.basename(img_path))[0]
        img_bgr = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        spectral_prior = compute_spectral_prior(img_bgr, out_size=(self.img_size, self.img_size))
        augmented = self.transforms(image=img_rgb)
        return {
            "image": augmented["image"],
            "spectral_prior": torch.from_numpy(spectral_prior),
            "img_id": img_id,
        }
