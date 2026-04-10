"""
model.py
--------
SciAnomalyDupliNet — Dual-Domain Forgery Segmentation Network

Architecture:
  Input: 5ch  (3 RGB + 2 spectral prior)
  → MobileViT-XS encoder  (spatial features)
  → Parallel FFT frequency branch
  → RepAnom Cross-Attention block (NOVEL)
  → Anomaly-Guided Decoder (U-Net style)
  → 1ch sigmoid mask
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


# ─────────────────────────────────────────────────────────────────────────────
# Helper blocks
# ─────────────────────────────────────────────────────────────────────────────

class ConvBnRelu(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, s, p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DepthwiseSeparable(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch, bias=False)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.pw(self.dw(x))))


# ─────────────────────────────────────────────────────────────────────────────
# RepAnom Cross-Attention Block  (the novel component)
# ─────────────────────────────────────────────────────────────────────────────

class RepAnomCrossAttention(nn.Module):
    """
    Spatial patches → Query
    Frequency tokens + Repetition Anomaly embeddings → Key / Value

    Forces the model to attend to where duplication breaks natural periodicity.
    """

    def __init__(self, spatial_dim: int, freq_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert spatial_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = spatial_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Project spatial features → Q
        self.q_proj = nn.Linear(spatial_dim, spatial_dim)
        # Project frequency tokens → K, V
        self.k_proj = nn.Linear(freq_dim, spatial_dim)
        self.v_proj = nn.Linear(freq_dim, spatial_dim)

        self.out_proj = nn.Linear(spatial_dim, spatial_dim)
        self.norm = nn.LayerNorm(spatial_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, spatial_feat: torch.Tensor, freq_tokens: torch.Tensor) -> torch.Tensor:
        """
        spatial_feat: (B, C_s, H, W)
        freq_tokens:  (B, N_f, C_f)
        returns:      (B, C_s, H, W)
        """
        B, C, H, W = spatial_feat.shape
        N_s = H * W

        # Flatten spatial → (B, N_s, C_s)
        x_flat = spatial_feat.view(B, C, N_s).permute(0, 2, 1)

        Q = self.q_proj(x_flat)                    # (B, N_s, C_s)
        K = self.k_proj(freq_tokens)               # (B, N_f, C_s)
        V = self.v_proj(freq_tokens)               # (B, N_f, C_s)

        # Multi-head split
        def split_heads(t):
            b, n, c = t.shape
            return t.view(b, n, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        Q, K, V = split_heads(Q), split_heads(K), split_heads(V)

        # Scaled dot-product attention
        attn = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # (B, heads, N_s, N_f)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, V)                                # (B, heads, N_s, head_dim)
        out = out.permute(0, 2, 1, 3).reshape(B, N_s, C)
        out = self.out_proj(out)

        # Residual + norm
        out = self.norm(x_flat + out)

        # Reshape back to spatial
        out = out.permute(0, 2, 1).view(B, C, H, W)
        return out


# ─────────────────────────────────────────────────────────────────────────────
# Frequency Branch
# ─────────────────────────────────────────────────────────────────────────────

class FrequencyBranch(nn.Module):
    """
    Takes the deepest spatial feature map, computes its 2D FFT,
    projects to frequency tokens.
    """

    def __init__(self, in_ch: int, freq_dim: int = 256, num_tokens: int = 64):
        super().__init__()
        self.num_tokens = num_tokens
        self.freq_dim = freq_dim
        # Project flattened FFT magnitude → fixed-size token set
        self.proj = nn.Sequential(
            nn.Linear(in_ch, freq_dim),
            nn.ReLU(inplace=True),
            nn.Linear(freq_dim, freq_dim),
        )
        self.pool = nn.AdaptiveAvgPool2d((int(num_tokens ** 0.5), int(num_tokens ** 0.5)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W)
        returns: (B, N_tokens, freq_dim)
        """
        B, C, H, W = x.shape

        # Real FFT → log-magnitude spectrum
        fft_out = torch.fft.rfft2(x, norm="ortho")           # (B, C, H, W//2+1)
        magnitude = torch.abs(fft_out)
        log_mag = torch.log1p(magnitude)                      # (B, C, H, W//2+1)

        # Spatially pool to fixed N_tokens spatial grid
        sq = int(self.num_tokens ** 0.5)
        log_mag_pooled = F.adaptive_avg_pool2d(log_mag, (sq, sq))  # (B, C, sq, sq)

        # Flatten spatial → tokens
        tokens = log_mag_pooled.view(B, C, sq * sq).permute(0, 2, 1)  # (B, N, C)
        tokens = self.proj(tokens)                                      # (B, N, freq_dim)
        return tokens


# ─────────────────────────────────────────────────────────────────────────────
# Anomaly-Guided Decoder Block
# ─────────────────────────────────────────────────────────────────────────────

class AnomalyGuidedDecoderBlock(nn.Module):
    """
    U-Net style decoder block with anomaly gating.
    
    The anomaly score (from spectral prior channel 1) is used as a spatial gate:
    gate = sigmoid(anomaly_score)
    skip = skip * (1 + gate)   ← boost features where anomaly is high
    """

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        merged_ch = in_ch // 2 + skip_ch
        self.conv = nn.Sequential(
            ConvBnRelu(merged_ch + 1, out_ch),  # +1 for anomaly channel
            DepthwiseSeparable(out_ch, out_ch),
        )
        # Anomaly gate: single-channel feature map
        self.anomaly_gate = nn.Sequential(
            nn.Conv2d(1, 1, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor, anomaly: torch.Tensor) -> torch.Tensor:
        """
        x:       (B, in_ch, H, W)
        skip:    (B, skip_ch, 2H, 2W)
        anomaly: (B, 1, 2H, 2W)  — the repetition anomaly score
        """
        x = self.up(x)

        # Align size in case of odd dims
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)

        # Anomaly gating on skip connection
        gate = self.anomaly_gate(anomaly)       # (B, 1, H, W)
        skip = skip * (1.0 + gate)              # boost forged-region features

        x = torch.cat([x, skip, anomaly], dim=1)
        return self.conv(x)


# ─────────────────────────────────────────────────────────────────────────────
# Main Model
# ─────────────────────────────────────────────────────────────────────────────

class SciAnomalyDupliNet(nn.Module):
    """
    Full SciAnomalyDupliNet model.

    Input:   (B, 5, H, W)  — 3 RGB + 2 spectral prior channels
    Output:  (B, 1, H, W)  — binary forgery mask logits (pre-sigmoid)
    """

    ENCODER_NAME = "mobilevit_xs"

    def __init__(self,
                 img_size: int = 512,
                 encoder_name: str = ENCODER_NAME,
                 pretrained: bool = True,
                 freq_dim: int = 128,
                 num_freq_tokens: int = 64,
                 attn_heads: int = 4,
                 use_gradient_checkpointing: bool = True):
        super().__init__()
        self.img_size = img_size

        # ── Patch the stem to accept 5 input channels ──────────────────────
        # Load backbone with default 3ch first, then modify stem
        self.encoder = timm.create_model(
            encoder_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=(0, 1, 2, 3),
        )

        # Identify the first conv and expand from 3 → 5 channels
        self._patch_stem_channels(in_channels=5)

        if use_gradient_checkpointing:
            # Enable gradient checkpointing on encoder children if supported
            for module in self.encoder.children():
                if hasattr(module, "gradient_checkpointing_enable"):
                    module.gradient_checkpointing_enable()

        # Get encoder feature channel counts via a dummy forward pass
        with torch.no_grad():
            dummy = torch.zeros(1, 5, img_size, img_size)
            feats = self.encoder(dummy)
            self.feat_channels = [f.shape[1] for f in feats]

        print(f"[Model] Encoder feature channels: {self.feat_channels}")

        C4 = self.feat_channels[-1]   # deepest feature map
        C3 = self.feat_channels[-2]
        C2 = self.feat_channels[-3]
        C1 = self.feat_channels[-4]

        # ── Frequency Branch ───────────────────────────────────────────────
        self.freq_branch = FrequencyBranch(in_ch=C4, freq_dim=freq_dim, num_tokens=num_freq_tokens)

        # ── RepAnom Cross-Attention ────────────────────────────────────────
        self.rep_anom_attn = RepAnomCrossAttention(
            spatial_dim=C4,
            freq_dim=freq_dim,
            num_heads=attn_heads,
        )

        # ── Anomaly gate projection (spectral prior → anomaly map at each scale) ─
        # We project the 2-ch prior down to 1ch for gating
        self.anomaly_gate_proj = nn.Conv2d(2, 1, 1)

        # ── Decoder ───────────────────────────────────────────────────────
        dec_ch = [256, 128, 64, 32]
        self.dec4 = AnomalyGuidedDecoderBlock(C4, C3, dec_ch[0])
        self.dec3 = AnomalyGuidedDecoderBlock(dec_ch[0], C2, dec_ch[1])
        self.dec2 = AnomalyGuidedDecoderBlock(dec_ch[1], C1, dec_ch[2])

        # Final upsample to original resolution (no skip at this stage)
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(dec_ch[2], dec_ch[3], kernel_size=2, stride=2),
            ConvBnRelu(dec_ch[3], dec_ch[3]),
        )
        self.head = nn.Conv2d(dec_ch[3], 1, kernel_size=1)

    # ── Stem patching ─────────────────────────────────────────────────────

    def _patch_stem_channels(self, in_channels: int = 5):
        """Replace the first conv layer to accept `in_channels` instead of 3."""
        # Find the first Conv2d in the encoder
        first_conv = None
        first_name = None
        for name, m in self.encoder.named_modules():
            if isinstance(m, nn.Conv2d):
                first_conv = m
                first_name = name
                break

        if first_conv is None:
            raise RuntimeError("Could not find first Conv2d in encoder.")

        # Build replacement conv
        new_conv = nn.Conv2d(
            in_channels,
            first_conv.out_channels,
            kernel_size=first_conv.kernel_size,
            stride=first_conv.stride,
            padding=first_conv.padding,
            bias=first_conv.bias is not None,
        )

        # Copy pretrained weights for RGB channels; init extra channels with small noise
        with torch.no_grad():
            new_conv.weight[:, :3, :, :] = first_conv.weight.clone()
            if in_channels > 3:
                nn.init.normal_(new_conv.weight[:, 3:, :, :], mean=0.0, std=0.01)
            if first_conv.bias is not None:
                new_conv.bias.copy_(first_conv.bias)

        # Set the replacement (handle nested attribute paths)
        parts = first_name.split(".")
        parent = self.encoder
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], new_conv)

    # ── Forward ───────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor, spectral_prior: torch.Tensor) -> torch.Tensor:
        """
        x:              (B, 3, H, W)
        spectral_prior: (B, 2, H, W)
        returns:        (B, 1, H, W) logits
        """
        B, _, H, W = x.shape

        # Concat RGB + spectral prior → 5-channel input
        inp = torch.cat([x, spectral_prior], dim=1)   # (B, 5, H, W)

        # ── Encoder ──────────────────────────────────────────────────────
        feats = self.encoder(inp)          # list of 4 feature maps
        f1, f2, f3, f4 = feats            # low → high resolution

        # ── Frequency branch on deepest feature ──────────────────────────
        freq_tokens = self.freq_branch(f4)             # (B, N, freq_dim)

        # ── RepAnom Cross-Attention ───────────────────────────────────────
        f4_enriched = self.rep_anom_attn(f4, freq_tokens)   # (B, C4, h4, w4)

        # ── Anomaly gate map (downsample spectral prior to each decoder scale) ──
        def get_anomaly(target_hw):
            sp = F.interpolate(spectral_prior, size=target_hw, mode="bilinear", align_corners=False)
            return self.anomaly_gate_proj(sp)   # (B, 1, h, w)

        # ── Decoder ──────────────────────────────────────────────────────
        d4 = self.dec4(f4_enriched, f3, get_anomaly(f3.shape[2:]))
        d3 = self.dec3(d4, f2, get_anomaly(f2.shape[2:]))
        d2 = self.dec2(d3, f1, get_anomaly(f1.shape[2:]))

        # Final upsample + head
        out = self.final_up(d2)
        if out.shape[2:] != (H, W):
            out = F.interpolate(out, size=(H, W), mode="bilinear", align_corners=False)
        logits = self.head(out)     # (B, 1, H, W)
        return logits


# ─────────────────────────────────────────────────────────────────────────────
# Quick parameter count
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    model = SciAnomalyDupliNet(img_size=512, pretrained=False)
    total = sum(p.numel() for p in model.parameters()) / 1e6
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"Total params:     {total:.2f}M")
    print(f"Trainable params: {trainable:.2f}M")

    x = torch.randn(2, 3, 512, 512)
    sp = torch.randn(2, 2, 512, 512)
    out = model(x, sp)
    print(f"Output shape: {out.shape}")
