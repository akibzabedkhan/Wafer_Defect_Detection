"""
SpatialYield.py
Yield loss computation from segmentation model outputs.

For each wafer computes:
    - Dies lost per defect class
    - Defect Area Ratio (DAR) per class
    - Total yield loss (union of all defective dies)
    - Overlap dies (attributed to 2+ classes)

Aggregates across a full dataset to produce a summary report.
"""

import torch
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(__file__))
from SpatialConfig import (
    WM38K_classes,
    number_of_classes,
)

num_defect_classes = number_of_classes - 1  # 8
THRESHOLD          = 0.5


# ─────────────────────────────────────────────────────────────────
# Per-wafer yield computation
# ─────────────────────────────────────────────────────────────────
def compute_yield_loss(logits, wafer_maps, threshold=THRESHOLD):
    """
    Compute per-class yield loss for each wafer in a batch.

    
    """
    wafer_maps = wafer_maps.to(logits.device)
    
    probs        = torch.sigmoid(logits)                        # [B, num_defect_classes, H, W]
    preds        = (probs > threshold).float()                  # [B, num_defect_classes, H, W]

    # Valid die mask — excludes background (pixel == 0)
    valid_mask   = (wafer_maps != 0).float()                    # [B, H, W]
    valid_exp    = valid_mask.unsqueeze(1).expand_as(preds)     # [B, num_defect_classes, H, W]

    # Only count predictions on real die positions
    preds_valid  = preds * valid_exp                            # [B, num_defect_classes, H, W]

    results = []

    for b in range(logits.shape[0]):
        total_valid = int(valid_mask[b].sum().item())

        if total_valid == 0:
            results.append({
                'dies_lost':        {name: 0 for name in WM38K_classes[:num_defect_classes]},
                'DAR':              {name: 0.0 for name in WM38K_classes[:num_defect_classes]},
                'total_dies_lost':  0,
                'total_yield_loss': 0.0,
                'overlap_dies':     0,
                'total_valid_dies': 0,
            })
            continue

        # Per-class dies lost and DAR
        dies_lost = {}
        DAR       = {}
        for c, name in enumerate(WM38K_classes[:num_defect_classes]):
            count         = int(preds_valid[b, c].sum().item())
            dies_lost[name] = count
            DAR[name]       = count / total_valid

        # Union — dies killed by any class
        any_defect       = preds_valid[b].any(dim=0).float()   # [H, W]
        total_dies_lost  = int(any_defect.sum().item())
        total_yield_loss = total_dies_lost / total_valid

        # Overlap — dies attributed to 2+ classes
        class_count  = preds_valid[b].sum(dim=0)               # [H, W]
        overlap_dies = int((class_count >= 2).sum().item())

        results.append({
            'dies_lost':        dies_lost,
            'DAR':              DAR,
            'total_dies_lost':  total_dies_lost,
            'total_yield_loss': total_yield_loss,
            'overlap_dies':     overlap_dies,
            'total_valid_dies': total_valid,
        })

    return results


# ─────────────────────────────────────────────────────────────────
# Dataset-level aggregation
# ─────────────────────────────────────────────────────────────────
def batch_yield_summary(results):
    """
    Aggregate yield results across all wafers in a dataset.

    Returns:
        summary dict:
            'total_dies_lost_per_class' : {class_name: int}
            'total_valid_dies'          : int
            'total_dies_lost'           : int
            'total_overlap_dies'        : int
            'DAR_per_class'             : {class_name: float}
            'total_yield_loss'          : float
            'pct_of_total_loss'         : {class_name: float}
    """
    total_valid        = 0
    total_lost         = 0
    total_overlap      = 0
    dies_lost_per_class = {name: 0 for name in WM38K_classes[:num_defect_classes]}

    for r in results:
        total_valid   += r['total_valid_dies']
        total_lost    += r['total_dies_lost']
        total_overlap += r['overlap_dies']
        for name in WM38K_classes[:num_defect_classes]:
            dies_lost_per_class[name] += r['dies_lost'][name]

    # DAR per class across full dataset
    DAR_per_class = {
        name: dies_lost_per_class[name] / max(total_valid, 1)
        for name in WM38K_classes[:num_defect_classes]
    }
    #Computing the percentage loss per defect type to show usability of wafers with present defects
    #

    # Each class's share of total dies lost
    pct_of_total = {
        name: (dies_lost_per_class[name] / max(total_lost, 1)) * 100
        for name in WM38K_classes[:num_defect_classes]
    }

    return {
        'dies_lost_per_class': dies_lost_per_class,
        'DAR_per_class':       DAR_per_class,
        'pct_of_total_loss':   pct_of_total,
        'total_valid_dies':    total_valid,
        'total_dies_lost':     total_lost,
        'total_overlap_dies':  total_overlap,
        'total_yield_loss':    total_lost / max(total_valid, 1),
    }


# ─────────────────────────────────────────────────────────────────
# Report printing
# ─────────────────────────────────────────────────────────────────
def print_yield_report(summary):
    """
    Print formatted yield loss report from batch_yield_summary output.
    """
    print("\n── Yield Loss Report ───────────────────────────────")
    print(f"  {'Class':<12} {'Dies Lost':>10} {'DAR':>8} {'% of Total Loss':>16}")
    print(f"  {'─'*50}")

    for name in WM38K_classes[:num_defect_classes]:
        dies = summary['dies_lost_per_class'][name]
        dar  = summary['DAR_per_class'][name]
        pct  = summary['pct_of_total_loss'][name]
        print(f"  {name:<12} {dies:>10,} {dar*100:>7.2f}% {pct:>15.1f}%")

    print(f"  {'─'*50}")
    print(f"  {'Total lost':<12} {summary['total_dies_lost']:>10,} "
          f"{summary['total_yield_loss']*100:>7.2f}%")
    print(f"  {'Overlap':<12} {summary['total_overlap_dies']:>10,} "
          f"  (attributed to 2+ classes)")
    print(f"  {'Valid dies':<12} {summary['total_valid_dies']:>10,}")
    print(f"  {'─'*50}")