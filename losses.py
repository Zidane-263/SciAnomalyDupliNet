"""
losses.py
---------
Loss functions for SciAnomalyDupliNet:

  L = BCE + DiceLoss + 0.4 * FocalLoss + λ * PeriodicityConsistencyLoss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Dice Loss
# ─────────────────────────────────────────────────────────────────────────────

class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        pred:   (B, 1, H, W) — sigmoid probabilities
        target: (B, 1, H, W) — binary mask
        """
        pred_flat   = pred.view(-1)
        target_flat = target.view(-1)
        intersection = (pred_flat * target_flat).sum()
        dice = (2.0 * intersection + self.smooth) / (
            pred_flat.sum() + target_flat.sum() + self.smooth
        )
        return 1.0 - dice


# ─────────────────────────────────────────────────────────────────────────────
# Focal Loss
# ─────────────────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: float = 0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        logits: (B, 1, H, W) — raw (pre-sigmoid) logits
        target: (B, 1, H, W) — binary mask
        """
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        prob  = torch.sigmoid(logits)
        p_t   = prob * target + (1 - prob) * (1 - target)
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        return (focal_weight * bce).mean()


# ─────────────────────────────────────────────────────────────────────────────
# Periodicity Consistency Loss  (novel physics-informed term)
# ─────────────────────────────────────────────────────────────────────────────

class PeriodicityConsistencyLoss(nn.Module):
    """
    Encourages the model to predict forgery where the repetition anomaly
    score is high.

    For predicted forged pixels (pred > threshold):
        target anomaly = 1.0  (anomaly should be high there)
    For predicted authentic pixels:
        target anomaly = 0.0

    Loss = MSE(anomaly_map_at_pred_pixels, target_anomaly)

    This acts as physics-grounded self-supervision.
    """

    def __init__(self, threshold: float = 0.5):
        super().__init__()
        self.threshold = threshold

    def forward(self,
                pred: torch.Tensor,
                spectral_prior: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        """
        pred:           (B, 1, H, W) sigmoid probabilities
        spectral_prior: (B, 2, H, W)  — channel 1 is anomaly map
        target:         (B, 1, H, W) ground truth mask
        """
        anomaly_map = spectral_prior[:, 1:2, :, :]   # (B, 1, H, W)

        # Only compute on GT-positive pixels to avoid trivial solution
        gt_mask = target.bool()

        if gt_mask.sum() == 0:
            return torch.tensor(0.0, device=pred.device)

        # At forged locations, anomaly should be high
        loss_pos = F.mse_loss(
            anomaly_map[gt_mask],
            torch.ones_like(anomaly_map[gt_mask]),
        )

        # At authentic locations, anomaly should be low
        gt_neg = ~gt_mask
        if gt_neg.sum() > 0:
            loss_neg = F.mse_loss(
                anomaly_map[gt_neg],
                torch.zeros_like(anomaly_map[gt_neg]),
            )
        else:
            loss_neg = torch.tensor(0.0, device=pred.device)

        return 0.7 * loss_pos + 0.3 * loss_neg


# ─────────────────────────────────────────────────────────────────────────────
# Combined Loss
# ─────────────────────────────────────────────────────────────────────────────

class SciAnomalyLoss(nn.Module):
    def __init__(self,
                 bce_weight:    float = 1.0,
                 dice_weight:   float = 1.0,
                 focal_weight:  float = 0.4,
                 period_weight: float = 0.1):
        super().__init__()
        self.bce_w    = bce_weight
        self.dice_w   = dice_weight
        self.focal_w  = focal_weight
        self.period_w = period_weight

        self.bce   = nn.BCEWithLogitsLoss()
        self.dice  = DiceLoss()
        self.focal = FocalLoss(gamma=2.0)
        self.period = PeriodicityConsistencyLoss()

    def forward(self,
                logits: torch.Tensor,
                target: torch.Tensor,
                spectral_prior: torch.Tensor) -> torch.Tensor:
        """
        logits:         (B, 1, H, W) raw logits
        target:         (B, 1, H, W) binary mask
        spectral_prior: (B, 2, H, W)
        """
        pred = torch.sigmoid(logits)

        loss_bce   = self.bce(logits, target)
        loss_dice  = self.dice(pred, target)
        loss_focal = self.focal(logits, target)
        loss_period = self.period(pred, spectral_prior, target)

        total = (self.bce_w    * loss_bce
               + self.dice_w   * loss_dice
               + self.focal_w  * loss_focal
               + self.period_w * loss_period)

        return total, {
            "bce":    loss_bce.item(),
            "dice":   loss_dice.item(),
            "focal":  loss_focal.item(),
            "period": loss_period.item(),
        }
