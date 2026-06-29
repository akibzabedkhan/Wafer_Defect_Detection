"""
SpatialModel.py
U-Net for per-die spatial segmentation with multi-task output.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

sys.path.append(os.path.dirname(__file__))
from SpatialConfig import (
    number_of_classes,
    learning_rate,
    weight_decay,
    device,
    Dropout_amount,
)

num_defect_classes = number_of_classes - 1  # 8 — excludes None


# ─────────────────────────────────────────────────────────────────
# Building blocks
# ─────────────────────────────────────────────────────────────────
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=Dropout_amount):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch,  out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
        )

    def forward(self, x):
        return self.block(x)


class Down(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=Dropout_amount):
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch, dropout),
        )

    def forward(self, x):
        return self.block(x)


class Up(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=Dropout_amount):
        super().__init__()
        self.up   = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_ch, out_ch, dropout)

    def forward(self, x, skip):
        x = self.up(x)

        # Pad for odd spatial dimensions e.g. 52→26→13→6
        if x.shape != skip.shape:
            x = F.pad(x, [0, skip.shape[3] - x.shape[3],
                          0, skip.shape[2] - x.shape[2]])

        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


# ─────────────────────────────────────────────────────────────────
# Spatial U-Net
# ─────────────────────────────────────────────────────────────────
class SpatialUNet(nn.Module):
    """
    Input  : [B, 1, 52, 52]

    Outputs:
        seg_logits : [B, num_defect_classes, 52, 52]
        cls_logits : [B, number_of_classes]
        bottleneck : [B, 256, 6, 6]
    """
    def __init__(self, dropout=Dropout_amount):
        super().__init__()

        # ── Encoder ─────────────────────────────────────────────
        self.enc1       = DoubleConv(1,   32,  dropout)  # [B,   1, 52, 52] → [B,  32, 52, 52]
        self.enc2       = Down(32,  64,  dropout)         # [B,  32, 52, 52] → [B,  64, 26, 26]
        self.enc3       = Down(64,  128, dropout)         # [B,  64, 26, 26] → [B, 128, 13, 13]

        # ── Bottleneck ───────────────────────────────────────────
        self.bottleneck = Down(128, 256, dropout)         # [B, 128, 13, 13] → [B, 256,  6,  6]

        # ── Decoder ─────────────────────────────────────────────
        self.dec3       = Up(256 + 128, 128, dropout)    # [B, 384,  6,  6] → [B, 128, 13, 13]
        self.dec2       = Up(128 + 64,   64, dropout)    # [B, 192, 13, 13] → [B,  64, 26, 26]
        self.dec1       = Up(64  + 32,   32, dropout)    # [B,  96, 26, 26] → [B,  32, 52, 52]

        # ── Segmentation head ────────────────────────────────────
        self.seg_head   = nn.Conv2d(32, num_defect_classes, kernel_size=1)

        # ── Classification head ──────────────────────────────────
        self.cls_head   = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, number_of_classes),
        )

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)           # [B,  32, 52, 52]
        e2 = self.enc2(e1)          # [B,  64, 26, 26]
        e3 = self.enc3(e2)          # [B, 128, 13, 13]

        # Bottleneck
        b  = self.bottleneck(e3)    # [B, 256,  6,  6]

        # Decoder
        d3 = self.dec3(b,  e3)     # [B, 128, 13, 13]
        d2 = self.dec2(d3, e2)     # [B,  64, 26, 26]
        d1 = self.dec1(d2, e1)     # [B,  32, 52, 52]

        # Outputs
        seg_logits = self.seg_head(d1)  # [B, num_defect_classes, 52, 52]
        cls_logits = self.cls_head(b)   # [B, number_of_classes]

        return seg_logits, cls_logits, b


# ─────────────────────────────────────────────────────────────────
# Build function
# ─────────────────────────────────────────────────────────────────
def buildSpatialModel(lr=learning_rate, wd=weight_decay, dropout=Dropout_amount):
    """
    Returns: model, seg_criterion, cls_criterion, optimizer
    """
    model = SpatialUNet(dropout=dropout).to(device)

    seg_criterion = nn.BCEWithLogitsLoss(reduction='none')
    cls_criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr           = lr,
        weight_decay = wd,
    )

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"SpatialUNet built — trainable parameters: {total_params:,}")

    return model, seg_criterion, cls_criterion, optimizer