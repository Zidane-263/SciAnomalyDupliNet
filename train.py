"""
train.py
--------
Training script for SciAnomalyDupliNet.

Usage:
  python train.py [--data IF-Data] [--epochs 30] [--batch 4] [--img-size 512]
                  [--lr 3e-4] [--output checkpoints] [--debug]
"""

import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# Fix: duplicate OpenMP DLL on Windows (Conda + PyTorch collision)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import config as CFG
from dataset import ForgedImageDataset
from model import SciAnomalyDupliNet
from losses import SciAnomalyLoss


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def dice_score(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    """Compute batch-level Dice score."""
    pred_bin = (torch.sigmoid(pred) > threshold).float()
    smooth = 1.0
    intersection = (pred_bin * target).sum()
    return ((2.0 * intersection + smooth) /
            (pred_bin.sum() + target.sum() + smooth)).item()


def warmup_lr(optimizer, epoch: int, warmup_epochs: int, base_lr: float):
    if epoch < warmup_epochs:
        lr = base_lr * (epoch + 1) / warmup_epochs
        for pg in optimizer.param_groups:
            pg["lr"] = lr


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, loss_fn, scaler, device, epoch):
    model.train()
    total_loss = 0.0
    total_dice = 0.0

    for i, batch in enumerate(loader):
        img    = batch["image"].to(device)
        sp     = batch["spectral_prior"].to(device)
        mask   = batch["mask"].to(device)

        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type="cuda"):
            logits = model(img, sp)
            loss, loss_dict = loss_fn(logits, mask, sp)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        total_dice += dice_score(logits.detach(), mask)

        if i % 10 == 0:
            breakdown = " | ".join(f"{k}={v:.4f}" for k, v in loss_dict.items())
            print(f"  [E{epoch} {i}/{len(loader)}] loss={loss.item():.4f} | {breakdown}")

    n = len(loader)
    return total_loss / n, total_dice / n


@torch.no_grad()
def validate(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    total_dice = 0.0

    for batch in loader:
        img    = batch["image"].to(device)
        sp     = batch["spectral_prior"].to(device)
        mask   = batch["mask"].to(device)

        with autocast(device_type="cuda"):
            logits = model(img, sp)
            loss, _ = loss_fn(logits, mask, sp)

        total_loss += loss.item()
        total_dice += dice_score(logits, mask)

    n = len(loader)
    return total_loss / n, total_dice / n


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train SciAnomalyDupliNet")
    # All defaults come from config.py — override on the command line if needed
    parser.add_argument("--data",      default=CFG.DATA_ROOT)
    parser.add_argument("--epochs",    type=int,   default=CFG.EPOCHS)
    parser.add_argument("--batch",     type=int,   default=CFG.BATCH_SIZE)
    parser.add_argument("--img-size",  type=int,   default=CFG.IMG_SIZE)
    parser.add_argument("--lr",        type=float, default=CFG.LEARNING_RATE)
    parser.add_argument("--wd",        type=float, default=CFG.WEIGHT_DECAY)
    parser.add_argument("--warmup",    type=int,   default=CFG.WARMUP_EPOCHS)
    parser.add_argument("--val-frac",  type=float, default=CFG.VAL_FRACTION)
    parser.add_argument("--output",    default=CFG.OUTPUT_DIR)
    parser.add_argument("--workers",   type=int,   default=CFG.NUM_WORKERS)
    parser.add_argument("--debug",     action="store_true",
                        help="Single-batch smoke test (1 epoch)")
    parser.add_argument("--resume",    default=None,
                        help="Explicit checkpoint path. Leave blank for auto-resume.")
    parser.add_argument("--no-resume", action="store_true",
                        help="Force fresh training even if last_model.pth exists.")
    parser.add_argument("--finetune",  action="store_true",
                        help="If set, only loads model weights from resume checkpoint and resets optimizer/LR.")
    args = parser.parse_args()

    set_seed(CFG.SEED)
    os.makedirs(args.output, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] device={device}  img_size={args.img_size}  batch={args.batch}  epochs={args.epochs}")

    # ── Auto-resume: pick up last_model.pth automatically ─────────────────
    auto_ckpt = os.path.join(args.output, "last_model.pth")
    if args.resume is None and not args.no_resume and os.path.exists(auto_ckpt):
        args.resume = auto_ckpt
        print(f"[Train] 🔄 Auto-resuming from {auto_ckpt}")
    elif args.no_resume:
        print("[Train] Starting fresh (--no-resume flag set).")

    # ── Build val ids ─────────────────────────────────────────────────────
    import glob
    forged_ids = sorted([
        os.path.splitext(os.path.basename(p))[0]
        for p in glob.glob(os.path.join(args.data, "train_images", "forged", "*.png"))
    ])
    n_val = max(1, int(len(forged_ids) * args.val_frac))
    random.seed(CFG.SEED)
    random.shuffle(forged_ids)
    val_ids = set(forged_ids[:n_val])
    print(f"[Train] train={len(forged_ids)-n_val}  val={n_val}")

    # ── Datasets & DataLoaders ────────────────────────────────────────────
    train_ds = ForgedImageDataset(args.data, split="train",
                                  img_size=args.img_size, val_ids=val_ids)
    val_ds   = ForgedImageDataset(args.data, split="val",
                                  img_size=args.img_size, val_ids=val_ids)

    if args.debug:
        from torch.utils.data import Subset
        train_ds = Subset(train_ds, list(range(min(4, len(train_ds)))))
        val_ds   = Subset(val_ds,   list(range(min(2, len(val_ds)))))
        args.epochs = 1
        print("[DEBUG] Smoke-test mode: 1 epoch, tiny dataset")

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=args.workers, pin_memory=True,
                              drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False,
                              num_workers=args.workers, pin_memory=True)

    # ── Model ─────────────────────────────────────────────────────────────
    model = SciAnomalyDupliNet(img_size=args.img_size, pretrained=True).to(device)

    # ── Optimizer & Schedule ───────────────────────────────────────────────
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, args.epochs - args.warmup), eta_min=1e-6)
    scaler    = GradScaler(device="cuda")
    loss_fn   = SciAnomalyLoss(
        bce_weight=CFG.LOSS_BCE_WEIGHT,
        dice_weight=CFG.LOSS_DICE_WEIGHT,
        focal_weight=CFG.LOSS_FOCAL_WEIGHT,
        period_weight=CFG.LOSS_PERIOD_WEIGHT,
    )

    # ── Load checkpoint (auto or manual resume) ─────────────────────
    start_epoch = 1
    best_dice   = 0.0
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        
        if not args.finetune:
            optimizer.load_state_dict(ckpt["optimizer"])
            start_epoch = ckpt["epoch"] + 1
            best_dice   = ckpt.get("val_dice", 0.0)
            print(f"[Train] ✅ Resumed from epoch {ckpt['epoch']}  "
                  f"(best dice so far: {best_dice:.4f}, continuing to epoch {args.epochs})")
            # Restore scheduler position so LR continues correctly
            for _ in range(ckpt["epoch"] - args.warmup):
                scheduler.step()
        else:
            print(f"[Train] 🎯 Fine-tuning from checkpoint epoch {ckpt['epoch']}. Optimizer/LR reset.")

    best_path = os.path.join(args.output, "best_model.pth")

    # ── Training loop ─────────────────────────────────────────────────────
    for epoch in range(start_epoch, args.epochs + 1):
        # Warm-up LR
        if epoch <= args.warmup:
            warmup_lr(optimizer, epoch - 1, args.warmup, args.lr)

        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{args.epochs}   LR={optimizer.param_groups[0]['lr']:.2e}")

        train_loss, train_dice = train_one_epoch(
            model, train_loader, optimizer, loss_fn, scaler, device, epoch)

        val_loss, val_dice = validate(model, val_loader, loss_fn, device)

        if epoch > args.warmup:
            scheduler.step()

        print(f"\n[E{epoch}] Train loss={train_loss:.4f} dice={train_dice:.4f} | "
              f"Val loss={val_loss:.4f} dice={val_dice:.4f}")

        # Log VRAM usage
        if device.type == "cuda":
            used = torch.cuda.max_memory_allocated() / 1e9
            print(f"         Peak VRAM: {used:.2f} GB")
            torch.cuda.reset_peak_memory_stats()

        # Save checkpoint
        state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "val_dice": val_dice,
            "args": vars(args),
        }
        torch.save(state, os.path.join(args.output, "last_model.pth"))

        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(state, best_path)
            print(f"  ✅ New best val Dice: {best_dice:.4f}  → saved to {best_path}")

    print(f"\n[Train] Done. Best val Dice = {best_dice:.4f}")


if __name__ == "__main__":
    main()
