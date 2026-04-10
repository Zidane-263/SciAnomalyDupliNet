"""
spectral_priors.py
------------------
Classical signal-processing module for SciAnomalyDupliNet.
Produces a 2-channel spectral prior (Phase Correlation + Repetition Anomaly Score)
that is fed as extra input channels to the network.

No learned weights — runs fast on CPU during data loading.
"""

import numpy as np
import cv2
from scipy.ndimage import uniform_filter


# ─────────────────────────────────────────────────────────────────────────────
# 1. Fourier Phase Correlation Map
# ─────────────────────────────────────────────────────────────────────────────

def phase_correlation_map(img_gray: np.ndarray) -> np.ndarray:
    """
    Compute the normalized cross-power spectrum (phase correlation) of an image
    with itself, shifted by all possible offsets.  Peaks in this map indicate
    repeated regions (copy-move evidence).

    Args:
        img_gray: (H, W) float32 in [0, 1]

    Returns:
        pc_map: (H, W) float32 in [0, 1]  — higher = stronger periodicity / duplication
    """
    H, W = img_gray.shape

    # Hann window to suppress border artefacts
    win = np.outer(np.hanning(H), np.hanning(W)).astype(np.float32)
    f = np.fft.fft2(img_gray * win)

    # Normalized cross-power spectrum (self-correlation)
    eps = 1e-8
    G = f * np.conj(f)
    G /= (np.abs(G) + eps)

    # Inverse FFT → phase correlation surface
    pc = np.abs(np.fft.ifft2(G)).astype(np.float32)
    pc = np.fft.fftshift(pc)          # centre at (H/2, W/2)

    # Zero out the DC peak (self = no shift)
    cy, cx = H // 2, W // 2
    r = max(3, min(H, W) // 64)
    pc[cy - r: cy + r + 1, cx - r: cx + r + 1] = 0.0

    # Normalise to [0, 1]
    pc_min, pc_max = pc.min(), pc.max()
    if pc_max > pc_min:
        pc = (pc - pc_min) / (pc_max - pc_min + 1e-8)

    return pc


# ─────────────────────────────────────────────────────────────────────────────
# 2. Local Autocorrelation / Repetition Anomaly Score
# ─────────────────────────────────────────────────────────────────────────────

def local_autocorrelation_anomaly(img_gray: np.ndarray,
                                  window: int = 32,
                                  stride: int = 16) -> np.ndarray:
    """
    Compute a per-pixel Repetition Anomaly Score.

    For each local window, compute the autocorrelation magnitude at the first
    off-DC peak.  If a region has been copied, it will share an autocorrelation
    fingerprint with another region — but the *local* autocorrelation may deviate
    from the *global* expected periodicity of the image.

    Strategy:
      1. Compute the global autocorrelation 'template' (IFT of |FFT(img)|²).
      2. Compute local autocorrelation for each patch.
      3. Anomaly = |local_AC - global_AC_template| (L2, then spatially smooth).

    Args:
        img_gray: (H, W) float32 in [0, 1]
        window:   Patch size (default 32)
        stride:   Stride for sliding window (default 16)

    Returns:
        anomaly_map: (H, W) float32 in [0, 1]
    """
    H, W = img_gray.shape

    # ---- Global autocorrelation template ----
    F = np.fft.fft2(img_gray)
    global_power = np.abs(F) ** 2
    global_ac = np.abs(np.fft.ifft2(global_power)).astype(np.float32)
    global_ac = np.fft.fftshift(global_ac)
    # Crop to window size for comparison
    cy, cx = global_ac.shape[0] // 2, global_ac.shape[1] // 2
    half = window // 2
    global_ac_patch = global_ac[cy - half: cy + half, cx - half: cx + half]
    global_ac_patch_norm = global_ac_patch / (global_ac_patch.max() + 1e-8)

    # ---- Sliding window local AC ----
    score_map = np.zeros((H, W), dtype=np.float32)
    count_map = np.zeros((H, W), dtype=np.float32)

    for y in range(0, H - window + 1, stride):
        for x in range(0, W - window + 1, stride):
            patch = img_gray[y: y + window, x: x + window]
            fp = np.fft.fft2(patch)
            local_power = np.abs(fp) ** 2
            local_ac = np.abs(np.fft.ifft2(local_power)).astype(np.float32)
            local_ac = np.fft.fftshift(local_ac)
            local_ac_norm = local_ac / (local_ac.max() + 1e-8)

            # Deviation from global template
            diff = np.mean(np.abs(local_ac_norm - global_ac_patch_norm))

            score_map[y: y + window, x: x + window] += diff
            count_map[y: y + window, x: x + window] += 1.0

    # Average overlapping patches
    mask_valid = count_map > 0
    score_map[mask_valid] /= count_map[mask_valid]

    # Gaussian-smooth the anomaly map
    score_map = cv2.GaussianBlur(score_map, (0, 0), sigmaX=window // 4)

    # Normalize to [0, 1]
    s_min, s_max = score_map.min(), score_map.max()
    if s_max > s_min:
        score_map = (score_map - s_min) / (s_max - s_min + 1e-8)

    return score_map


# ─────────────────────────────────────────────────────────────────────────────
# 3. Combined: build the 2-channel spectral prior
# ─────────────────────────────────────────────────────────────────────────────

def compute_spectral_prior(img_bgr: np.ndarray,
                           out_size: tuple = (512, 512)) -> np.ndarray:
    """
    Given a BGR uint8 image, return the 2-channel spectral prior:
      Channel 0 — Phase Correlation Map
      Channel 1 — Repetition Anomaly Score

    Args:
        img_bgr:  (H, W, 3) uint8
        out_size: target (H, W) for the output prior

    Returns:
        prior: (2, out_H, out_W) float32
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

    # Resize to a fixed working resolution (faster + consistent)
    work_h, work_w = 256, 256
    gray_small = cv2.resize(gray, (work_w, work_h), interpolation=cv2.INTER_AREA)

    pc_map = phase_correlation_map(gray_small)
    ra_map = local_autocorrelation_anomaly(gray_small, window=32, stride=16)

    # Upscale to target size
    pc_map = cv2.resize(pc_map, (out_size[1], out_size[0]), interpolation=cv2.INTER_LINEAR)
    ra_map = cv2.resize(ra_map, (out_size[1], out_size[0]), interpolation=cv2.INTER_LINEAR)

    prior = np.stack([pc_map, ra_map], axis=0)   # (2, H, W)
    return prior.astype(np.float32)
