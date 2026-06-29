import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch
import sys
import os

"""Calls the makePseudolabel function f the saved stage1 dataset and maps them with colors"""

sys.path.append(os.path.dirname(__file__))
from SpatialConfig import WM38K_classes, WM38K_data, device
from SpatialDataload import make_pseudo_labels
from SpatialModel import buildSpatialModel
from SpatialConfig import checkpoint_dir

num_defect_classes = 8

CLASS_COLORS = [
    '#e41a1c',  # Center  
    '#377eb8',  # Donut    
    '#4daf4a',  # Edge-Loc  
    '#ff7f00',  # Edge-Ring 
    '#984ea3',  # Loc      
    '#a65628',  # Near-Full
    '#f781bf',  # Scratch
    '#999999',  # Random
]
BACKGROUND_COLOR = '#000000'
GOOD_DIE_COLOR   = '#ffffff'
OVERLAP_COLOR    = '#ffff00'


def colorize_wafer(wafer_map, pseudo_mask):
    """
    wafer_map   : [H, W]  values {0=background, 1=good, 2=defective}
    pseudo_mask : [8, H, W] values {-1=background, 0=good, 1=defect}

    Returns the color_map
    """
    H, W = wafer_map.shape
    #Defaults the map to be white with np.ones
    color_map = np.ones((H, W, 3))

    #Background dies
    bg = wafer_map == 0
    color_map[bg] = [0, 0, 0]

    # Count how many classes are active per pixel
    class_hits = (pseudo_mask == 1).sum(axis=0)  # [H, W]

    #Dealing with single classes
    for c in range(num_defect_classes):
        single_class = (pseudo_mask[c] == 1) & (class_hits == 1)
        color = np.array([int(CLASS_COLORS[c][i : i + 2], 16) / 255
                          for i in (1, 3, 5)])
        color_map[single_class] = color

    #Overlap when the model triggers 2 or more defects on a single die
    overlap = class_hits >= 2
    color_map[overlap] = [1.0, 1.0, 0.0]

    return color_map


def label_combo_name(label_vector):
    """Returns the mixed names"""
    active = [WM38K_classes[i] for i in range(num_defect_classes)
              if label_vector[i] == 1]
    return ' + '.join(active) if active else 'None'


def visualize_pseudo_labels(model=None, checkpoint=None, examples_per_combo=10, save_dir='visualizations/pseudo_labels/'):
    """
    Each mixed defect type has 10 examples displayed for the generated wafer mask
    """
    os.makedirs(save_dir, exist_ok=True)

    # ── Load model ───────────────────────────────────────────────
    if model is None:
        model, _, _, _ = buildSpatialModel()
        ckpt = checkpoint or os.path.join(checkpoint_dir, 'best_spatial_model.pt')
        model.load_state_dict(torch.load(ckpt, map_location=device))
        print(f"Loaded checkpoint: {ckpt}")
    model.eval()

    # ── Load mixed wafers ────────────────────────────────────────
    data = np.load(WM38K_data)
    raw_images = data['arr_0']
    raw_labels = data['arr_1']

    no_defect = (raw_labels.sum(axis=1, keepdims=True) == 0).astype(np.float32)
    labels = np.concatenate([raw_labels, no_defect], axis = 1)

    defect_count = raw_labels.sum(axis = 1)
    mixed_index = np.where(defect_count >= 2)[0]

    # ── Group by label combination ───────────────────────────────
    combos = {}
    for idx in mixed_index:
        key = tuple(labels[idx, :num_defect_classes].astype(int))
        combos.setdefault(key, []).append(idx)

    print(f"Found {len(combos)} unique mixed-defect combinations")

    # ── Generate pseudo masks for all mixed wafers at once ───────
    print("Generating pseudo masks...")
    pseudo_masks = make_pseudo_labels(model, raw_images[mixed_index], labels[mixed_index])
    # Build lookup: original dataset index → pseudo mask
    index_to_mask = {idx: pseudo_masks[i] for i, idx in enumerate(mixed_index)}

    # ── Plot each combination ────────────────────────────────────
    for combo_key, indices in combos.items():
        combo_name = label_combo_name(np.array(combo_key))
        chosen = indices[:examples_per_combo]
        n = len(chosen)

        fig, axes = plt.subplots(1, n, figsize=(3 * n, 3.5))
        if n == 1:
            axes = [axes]

        fig.suptitle(f'Pseudo Labels: {combo_name}  ({n} examples)',
                     fontsize=11, fontweight='bold')

        for ax, idx in zip(axes, chosen):
            wafer_map = raw_images[idx]
            pseudo_mask = index_to_mask[idx]

            overlap_count = (pseudo_mask == 1).sum(axis=0)
            print(f"Wafer {idx}: {int((overlap_count >= 2).sum())} overlap pixels out of {int((wafer_map != 0).sum())} valid dies")

            rgb = colorize_wafer(wafer_map, pseudo_mask)
            ax.imshow(rgb, interpolation = 'nearest')
            ax.set_title(f'Wafer {idx}', fontsize = 7)
            ax.axis('off')

        # ── Legend ───────────────────────────────────────────────
        legend_handles = [
            mpatches.Patch(color = GOOD_DIE_COLOR, label = 'Good die', ec = 'grey'),
            mpatches.Patch(color = BACKGROUND_COLOR, label = 'Background'),
            mpatches.Patch(color = OVERLAP_COLOR, label = 'Overlap (2+ classes)'),
        ]
        active_classes = [i for i, v in enumerate(combo_key) if v == 1]
        for c in active_classes:
            legend_handles.append(
                mpatches.Patch(color = CLASS_COLORS[c],
                               label = WM38K_classes[c])
            )

        fig.legend(handles = legend_handles, loc = 'lower center',
                   ncol = min(len(legend_handles), 5),
                   fontsize = 7, bbox_to_anchor = (0.5, -0.05))

        # Save — use combo name as filename, sanitise special chars
        safe_name = combo_name.replace(' ', '_').replace('+', '-')
        save_path = os.path.join(save_dir, f'{safe_name}.png')
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
        plt.close()
        print(f"Saved: {save_path}")

    print(f"\nDone — all figures saved to {save_dir}")


if __name__ == '__main__':
    visualize_pseudo_labels(
        examples_per_combo=10,
        save_dir='visualizations/pseudo_labels/'
    )
    