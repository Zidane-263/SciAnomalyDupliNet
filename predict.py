"""
predict.py
----------
Inference script for SciAnomalyDupliNet.

Loads a trained checkpoint, runs inference on test images,
applies post-processing, and saves binary masks as .npy files.

Usage:
  python predict.py [--data IF-Data] [--checkpoint checkpoints/best_model.pth]
                    [--output predictions] [--threshold auto]
"""

import argparse
import os
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
from scipy import ndimage

from dataset import TestImageDataset
from model import SciAnomalyDupliNet


# ─────────────────────────────────────────────────────────────────────────────
# Post-processing
# ─────────────────────────────────────────────────────────────────────────────

def postprocess_mask(prob_map: np.ndarray,
                     threshold: float = None,
                     min_blob_px: int = 50) -> np.ndarray:
    """
    Args:
        prob_map:    (H, W) float32 in [0, 1]
        threshold:   if None, use Otsu's method
        min_blob_px: remove connected components smaller than this

    Returns:
        binary_mask: (H, W) uint8 {0, 1}
    """
    # Convert to uint8 for OpenCV
    prob_u8 = (prob_map * 255).astype(np.uint8)

    # Otsu threshold or fixed
    if threshold is None:
        t_val, binary = cv2.threshold(prob_u8, 0, 255,
                                      cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        t_val = int(threshold * 255)
        _, binary = cv2.threshold(prob_u8, t_val, 255, cv2.THRESH_BINARY)

    binary = (binary > 0).astype(np.uint8)

    # Morphological opening (remove small noise)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    # Remove small connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    clean = np.zeros_like(binary)
    for lbl in range(1, num_labels):
        area = stats[lbl, cv2.CC_STAT_AREA]
        if area >= min_blob_px:
            clean[labels == lbl] = 1

    return clean


# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────

def predict_single(model: torch.nn.Module,
                   img_bgr: np.ndarray,
                   device: torch.device,
                   img_size: int = 512,
                   threshold: float = None) -> np.ndarray:
    """
    Run model on a single BGR image, return binary mask at original resolution.
    """
    from dataset import get_val_transforms
    from spectral_priors import compute_spectral_prior

    H_orig, W_orig = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Spectral prior
    sp = compute_spectral_prior(img_bgr, out_size=(img_size, img_size))
    sp_tensor = torch.from_numpy(sp).unsqueeze(0).to(device)   # (1, 2, H, W)

    # Image tensor
    transforms = get_val_transforms(img_size)
    aug = transforms(image=img_rgb)
    img_tensor = aug["image"].unsqueeze(0).to(device)           # (1, 3, H, W)

    with torch.no_grad(), autocast():
        logits = model(img_tensor, sp_tensor)                   # (1, 1, H, W)
        prob = torch.sigmoid(logits).squeeze().float().cpu().numpy()    # (H, W)

    # Resize back to original resolution
    prob_orig = cv2.resize(prob, (W_orig, H_orig), interpolation=cv2.INTER_LINEAR)

    # Post-process
    binary = postprocess_mask(prob_orig, threshold=threshold)
    return binary


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SciAnomalyDupliNet Inference")
    parser.add_argument("--data",       default="IF-Data")
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pth")
    parser.add_argument("--output",     default="predictions")
    parser.add_argument("--img-size",   type=int,   default=512)
    parser.add_argument("--threshold",  type=float, default=None,
                        help="Fixed threshold; None = auto (Otsu)")
    parser.add_argument("--min-blob",   type=int,   default=50)
    parser.add_argument("--input",      default=None,
                        help="Single image path (overrides --data)")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Predict] device={device}")

    # ── Load model ────────────────────────────────────────────────────────
    ckpt = torch.load(args.checkpoint, map_location=device)
    model = SciAnomalyDupliNet(img_size=args.img_size, pretrained=False).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"[Predict] Loaded checkpoint (val Dice={ckpt.get('val_dice', 'N/A'):.4f})")

    # ── Single image mode ─────────────────────────────────────────────────
    if args.input is not None:
        img_bgr = cv2.imread(args.input)
        if img_bgr is None:
            raise FileNotFoundError(f"Cannot read: {args.input}")
        img_id = os.path.splitext(os.path.basename(args.input))[0]
        binary = predict_single(model, img_bgr, device,
                                 args.img_size, args.threshold)
        out_path = os.path.join(args.output, f"{img_id}.npy")
        np.save(out_path, binary)
        print(f"[Predict] Saved → {out_path}  (forged px: {binary.sum()})")
        return

    # ── Batch test-set mode ───────────────────────────────────────────────
    import glob
    test_imgs = sorted(glob.glob(os.path.join(args.data, "test_images", "*.png")))
    print(f"[Predict] Found {len(test_imgs)} test images")

    for img_path in test_imgs:
        img_id  = os.path.splitext(os.path.basename(img_path))[0]
        img_bgr = cv2.imread(img_path)
        binary  = predict_single(model, img_bgr, device,
                                  args.img_size, args.threshold)
        out_path = os.path.join(args.output, f"{img_id}.npy")
        np.save(out_path, binary)
        print(f"  {img_id}: forged_px={binary.sum()}")

    print(f"[Predict] Done. Masks saved to {args.output}/")


if __name__ == "__main__":
    main()
