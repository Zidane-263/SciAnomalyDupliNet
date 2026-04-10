"""
config.py
---------
Central configuration for SciAnomalyDupliNet.
Edit this file to change ANY training parameter.
Then just run:  python train.py
"""

# ─────────────────────────────────────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────────────────────────────────────
DATA_ROOT       = "IF-Data"          # path to dataset folder
IMG_SIZE        = 512                # input resolution (try 384 to save VRAM)
USE_SUPPLEMENTAL = True              # use supplemental_images/ as extra train data
VAL_FRACTION    = 0.15               # fraction of forged images held out for validation

# ─────────────────────────────────────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────────────────────────────────────
EPOCHS          = 40                # total epochs to train (v3 extension)
BATCH_SIZE      = 6                 # ↓ to 2 if you get OOM, ↑ to 8 if you have headroom
NUM_WORKERS     = 4                 # 0 = safest on Windows; set 2-4 if on Linux

# ─────────────────────────────────────────────────────────────────────────────
# OPTIMIZER
# ─────────────────────────────────────────────────────────────────────────────
LEARNING_RATE   = 1e-4               # initial learning rate
WEIGHT_DECAY    = 1e-4
WARMUP_EPOCHS   = 3                  # linear LR warmup for first N epochs
GRAD_CLIP       = 1.0                # gradient clipping max norm

# ─────────────────────────────────────────────────────────────────────────────
# LOSS WEIGHTS
# ─────────────────────────────────────────────────────────────────────────────
LOSS_BCE_WEIGHT    = 0.5
LOSS_DICE_WEIGHT   = 3.0
LOSS_FOCAL_WEIGHT  = 1.5
LOSS_PERIOD_WEIGHT = 0.1             # Periodicity Consistency Loss weight

# ─────────────────────────────────────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────────────────────────────────────
ENCODER_NAME    = "mobilevit_xs"     # timm model name
PRETRAINED      = True               # use ImageNet pretrained weights
FREQ_DIM        = 128                # frequency token embedding dim
NUM_FREQ_TOKENS = 64                 # number of frequency tokens
ATTN_HEADS      = 4                  # RepAnom cross-attention heads
GRAD_CHECKPOINT = True               # gradient checkpointing (saves VRAM)

# ─────────────────────────────────────────────────────────────────────────────
# CHECKPOINTING & OUTPUT
# ─────────────────────────────────────────────────────────────────────────────
OUTPUT_DIR      = "checkpoints"      # where to save checkpoints
AUTO_RESUME     = False              # automatically resume from last_model.pth if it exists
SEED            = 42
