"""
Multi-Planar Reconstruction (MPR) 3D Visualizer for MedNeXt-3D Predictions
Displays Axial, Coronal, and Sagittal cross-sections side by side.
"""

import os
import argparse
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from mednext_v1 import create_mednext_v1
from losses_3d import compute_dice_score_3d


def resolve_path(rel_path):
    candidates = [
        os.path.abspath(rel_path),
        os.path.join(os.path.dirname(__file__), rel_path),
        os.path.join(os.path.dirname(__file__), "..", rel_path),
        os.path.join("E:/LIRDC", rel_path),
        os.path.join("E:/LIRDC/Chest-CT-Scan-ML-Health-360", rel_path),
    ]
    for c in candidates:
        if os.path.exists(c):
            return os.path.abspath(c)
    return os.path.abspath(candidates[0])


def visualize_3d_prediction(args):
    cubes_dir = resolve_path("mednext-3d-segmentation/cubes_3d")
    meta_csv = os.path.join(cubes_dir, "meta_3d.csv")
    checkpoint_path = resolve_path(args.checkpoint)

    if not os.path.exists(meta_csv):
        print(f"Error: {meta_csv} not found! Please run 'prepare_3d_dataset.py' first.")
        return

    df = pd.read_csv(meta_csv)
    nodule_df = df[df["is_clean"] == False]

    if len(nodule_df) == 0:
        print("Error: No nodule volumes found in metadata.")
        return

    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    print(f"Using device: {device}")

    # Load Model
    model = create_mednext_v1(
        num_input_channels=1,
        num_classes=1,
        model_id=args.model_size.upper(),
        kernel_size=args.kernel_size,
        deep_supervision=False
    ).to(device)

    if os.path.exists(checkpoint_path):
        try:
            ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        except TypeError:
            ckpt = torch.load(checkpoint_path, map_location=device)
        state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
        model.load_state_dict(state_dict)
        print(f"Loaded trained weights from: {checkpoint_path}")
    else:
        print(f"Notice: Checkpoint '{checkpoint_path}' not found. Using untrained weights for demonstration.")

    model.eval()

    # Pick a sample nodule
    sample = nodule_df.sample(1, random_state=args.seed).iloc[0]
    cube_path = os.path.join(cubes_dir, sample["filepath"])
    print(f"Selected sample: {sample['cube_id']} (Patient: {sample['patient_id']}, Malignancy: {sample['malignancy']})")

    data = np.load(cube_path)
    img_vol = data["image"]  # (D, H, W)
    mask_vol = data["mask"]  # (D, H, W)

    # Inference
    x_tensor = torch.from_numpy(img_vol).unsqueeze(0).unsqueeze(0).float().to(device)
    with torch.no_grad():
        out_logits = model(x_tensor)
        pred_prob = torch.sigmoid(out_logits).squeeze().cpu().numpy()
        pred_mask = (pred_prob > args.threshold).astype(np.uint8)

    sample_dice = compute_dice_score_3d(out_logits, torch.from_numpy(mask_vol).unsqueeze(0).unsqueeze(0).to(device))
    print(f"3D Volumetric Dice Score for this sample: {sample_dice:.4f}")

    # Find Centroid coordinates for orthogonal slices
    pos = np.argwhere(mask_vol > 0)
    if pos.size > 0:
        cz, cy, cx = pos.mean(axis=0).astype(int)
    else:
        cz, cy, cx = [s // 2 for s in img_vol.shape]

    # Boundaries check
    cz = min(max(cz, 0), img_vol.shape[0] - 1)
    cy = min(max(cy, 0), img_vol.shape[1] - 1)
    cx = min(max(cx, 0), img_vol.shape[2] - 1)

    # Multi-Planar Orthogonal Slices
    views = [
        ("Axial (Transverse)", img_vol[cz, :, :], mask_vol[cz, :, :], pred_mask[cz, :, :]),
        ("Coronal (Frontal)", img_vol[:, cy, :], mask_vol[:, cy, :], pred_mask[:, cy, :]),
        ("Sagittal (Side)", img_vol[:, :, cx], mask_vol[:, :, cx], pred_mask[:, :, cx]),
    ]

    fig, axes = plt.subplots(3, 3, figsize=(14, 12))
    plt.suptitle(
        f"MedNeXt-3D Multi-Planar Reconstruction (MPR)\n"
        f"Sample: {sample['cube_id']} | 3D Dice: {sample_dice:.4f} | Malignancy: {sample['malignancy']}",
        fontsize=15,
        fontweight="bold"
    )

    for col_idx, (view_name, img_slice, gt_slice, pred_slice) in enumerate(views):
        # Row 1: Raw CT Volume
        axes[0, col_idx].imshow(img_slice, cmap="gray")
        axes[0, col_idx].set_title(f"{view_name}\nRaw CT", fontsize=12)
        axes[0, col_idx].axis("off")

        # Row 2: Ground Truth Mask Overlay
        axes[1, col_idx].imshow(img_slice, cmap="gray")
        axes[1, col_idx].imshow(gt_slice, cmap="Reds", alpha=0.5 if np.any(gt_slice > 0) else 0.0)
        axes[1, col_idx].set_title(f"Ground Truth Consensus", fontsize=12)
        axes[1, col_idx].axis("off")

        # Row 3: MedNeXt-3D Prediction Overlay
        axes[2, col_idx].imshow(img_slice, cmap="gray")
        axes[2, col_idx].imshow(pred_slice, cmap="Blues", alpha=0.5 if np.any(pred_slice > 0) else 0.0)
        axes[2, col_idx].set_title(f"MedNeXt-3D Prediction", fontsize=12)
        axes[2, col_idx].axis("off")

    plt.tight_layout()
    out_img = os.path.join(os.path.dirname(__file__), "mednext3d_sample_prediction.png")
    plt.savefig(out_img, dpi=300, bbox_inches="tight")
    print(f"Visualization saved to: {out_img}")
    if args.show:
        plt.show()
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Multi-Planar 3D Reconstruction for MedNeXt")
    parser.add_argument('--checkpoint', type=str, default="mednext-3d-segmentation/best_mednext3d.pth")
    parser.add_argument('--model-size', type=str, default="S", choices=['S', 'B', 'M', 'L'])
    parser.add_argument('--kernel-size', type=int, default=3)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--show', action='store_true', default=False, help="Display window")
    args = parser.parse_args()
    visualize_3d_prediction(args)

