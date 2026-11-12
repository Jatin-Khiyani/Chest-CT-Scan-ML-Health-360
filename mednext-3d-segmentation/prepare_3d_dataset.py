"""
3D Nodule Volume Extractor for MedNeXt-3D

Groups consecutive 2D CT slices per nodule into standardized 3D cubes
(default D=32, H=64, W=64) centered on radiologist consensus masks.
"""

import os
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm


def resolve_path(rel_path):
    candidates = [
        os.path.join("E:/LIRDC", rel_path),
        os.path.join("E:/LIRDC/Chest-CT-Scan-ML-Health-360", rel_path),
        os.path.abspath(rel_path),
        os.path.join(os.path.dirname(__file__), "..", rel_path),
        os.path.join(os.path.dirname(__file__), "..", "..", rel_path),
    ]
    for c in candidates:
        if os.path.exists(c):
            if os.path.isdir(c):
                # Verify directory has actual data, not just .gitkeep
                items = [f for f in os.listdir(c) if not f.startswith('.')]
                if len(items) > 0:
                    return os.path.abspath(c)
            else:
                return os.path.abspath(c)
    return os.path.abspath(candidates[0])


def crop_and_pad_3d(img_vol, mask_vol, target_shape=(32, 64, 64)):
    """
    Crops and symmetrically pads a 3D volume around the mask centroid to target_shape.
    """
    D_tgt, H_tgt, W_tgt = target_shape
    D_cur, H_cur, W_cur = img_vol.shape

    # Find mask centroid
    pos = np.argwhere(mask_vol > 0)
    if pos.size > 0:
        cz, cy, cx = pos.mean(axis=0).astype(int)
    else:
        cz, cy, cx = D_cur // 2, H_cur // 2, W_cur // 2

    # Spatial cropping coordinates (H, W)
    y1 = max(0, cy - H_tgt // 2)
    y2 = min(H_cur, y1 + H_tgt)
    if y2 - y1 < H_tgt:
        y1 = max(0, y2 - H_tgt)

    x1 = max(0, cx - W_tgt // 2)
    x2 = min(W_cur, x1 + W_tgt)
    if x2 - x1 < W_tgt:
        x1 = max(0, x2 - W_tgt)

    img_spatial = img_vol[:, y1:y2, x1:x2]
    mask_spatial = mask_vol[:, y1:y2, x1:x2]

    # Depth padding / cropping (D)
    out_img = np.zeros(target_shape, dtype=np.float32)
    out_mask = np.zeros(target_shape, dtype=np.uint8)

    if D_cur >= D_tgt:
        z1 = max(0, cz - D_tgt // 2)
        z2 = min(D_cur, z1 + D_tgt)
        if z2 - z1 < D_tgt:
            z1 = max(0, z2 - D_tgt)
        out_img = img_spatial[z1:z1 + D_tgt]
        out_mask = mask_spatial[z1:z1 + D_tgt]
    else:
        pad_z = (D_tgt - D_cur) // 2
        out_img[pad_z:pad_z + D_cur] = img_spatial
        out_mask[pad_z:pad_z + D_cur] = mask_spatial

    return out_img, out_mask


def extract_3d_cubes(args):
    meta_csv = resolve_path("LIDC-IDRI-Preprocessing-master/data/Meta/meta_info.csv")
    image_dir = resolve_path("LIDC-IDRI-Preprocessing-master/data/Image")
    mask_dir = resolve_path("LIDC-IDRI-Preprocessing-master/data/Mask")
    clean_dir = resolve_path("LIDC-IDRI-Preprocessing-master/data/Clean/Image")

    output_dir = os.path.join(os.path.dirname(__file__), "cubes_3d")
    nodule_out_dir = os.path.join(output_dir, "nodules")
    clean_out_dir = os.path.join(output_dir, "clean")
    Path(nodule_out_dir).mkdir(parents=True, exist_ok=True)
    Path(clean_out_dir).mkdir(parents=True, exist_ok=True)

    print(f"Loading metadata from: {meta_csv}")
    df = pd.read_csv(meta_csv)

    target_shape = (args.depth, args.size, args.size)
    print(f"Target 3D cube dimensions: {target_shape} (D x H x W)")

    meta_3d_records = []
    meta_3d_csv = os.path.join(output_dir, "meta_3d.csv")

    # Load existing metadata if resuming
    existing_cube_ids = set()
    if args.resume and os.path.exists(meta_3d_csv):
        try:
            existing_df = pd.read_csv(meta_3d_csv)
            meta_3d_records = existing_df.to_dict('records')
            existing_cube_ids = set(existing_df['cube_id'].tolist())
            print(f"Resuming: Loaded {len(existing_cube_ids)} existing 3D cubes.")
        except Exception as e:
            print(f"Notice: Could not load existing meta_3d.csv: {e}")

    # 1. Process Nodule volumes
    nodule_df = df[df["is_clean"] == False]
    nodule_groups = nodule_df.groupby(["patient_id", "nodule_no"])
    print(f"Found {len(nodule_groups)} candidate 3D nodules to process.")

    saved_nodules = 0
    for (pid, nno), group in tqdm(nodule_groups, desc="Extracting 3D Nodules"):
        pid_str = f"LIDC-IDRI-{str(pid).zfill(4)}"
        cube_id = f"{pid_str}_N{nno}"

        if args.resume and cube_id in existing_cube_ids:
            continue

        slices = group.sort_values("slice_no")
        imgs = []
        masks = []
        skip_nodule = False

        for _, row in slices.iterrows():
            img_p = os.path.join(image_dir, pid_str, str(row["original_image"]) + ".npy")
            mask_p = os.path.join(mask_dir, pid_str, str(row["mask_image"]) + ".npy")

            if not os.path.exists(img_p) or not os.path.exists(mask_p):
                skip_nodule = True
                break

            imgs.append(np.load(img_p))
            masks.append(np.load(mask_p))

        if skip_nodule or len(imgs) == 0:
            continue

        vol_img = np.stack(imgs, axis=0).astype(np.float32)
        vol_mask = np.stack(masks, axis=0).astype(np.uint8)

        # Skip if mask is empty
        mask_voxel_count = int(np.sum(vol_mask))
        if mask_voxel_count == 0:
            continue

        # Normalize slice intensity to [0, 1]
        v_min, v_max = vol_img.min(), vol_img.max()
        vol_img = (vol_img - v_min) / (v_max - v_min + 1e-6)

        # 3D Centered Crop & Pad
        cube_img, cube_mask = crop_and_pad_3d(vol_img, vol_mask, target_shape=target_shape)

        # Save compressed npz
        out_path = os.path.join(nodule_out_dir, f"{cube_id}.npz")
        np.savez_compressed(out_path, image=cube_img, mask=cube_mask)

        first_row = slices.iloc[0]
        meta_3d_records.append({
            "cube_id": cube_id,
            "patient_id": pid_str,
            "nodule_no": nno,
            "is_clean": False,
            "is_cancer": first_row.get("is_cancer", "Unknown"),
            "malignancy": first_row.get("malignancy", 0),
            "orig_depth": len(slices),
            "mask_voxels": mask_voxel_count,
            "filepath": os.path.relpath(out_path, output_dir)
        })
        saved_nodules += 1

    # 2. Process Clean / Negative lung volumes (optional sample)
    if args.include_clean and os.path.exists(clean_dir):
        clean_patients = [p for p in os.listdir(clean_dir) if os.path.isdir(os.path.join(clean_dir, p))]
        print(f"Sampling clean volumes from {len(clean_patients)} non-nodule patients...")

        for c_pid in tqdm(clean_patients[:args.max_clean], desc="Extracting Clean 3D Cubes"):
            cube_id = f"{c_pid}_CLEAN"
            if args.resume and cube_id in existing_cube_ids:
                continue

            c_folder = os.path.join(clean_dir, c_pid)
            c_slices = sorted([f for f in os.listdir(c_folder) if f.endswith(".npy")])
            if len(c_slices) < 8:
                continue

            imgs = [np.load(os.path.join(c_folder, f)) for f in c_slices[:args.depth]]
            vol_img = np.stack(imgs, axis=0).astype(np.float32)
            vol_mask = np.zeros_like(vol_img, dtype=np.uint8)

            v_min, v_max = vol_img.min(), vol_img.max()
            vol_img = (vol_img - v_min) / (v_max - v_min + 1e-6)

            cube_img, cube_mask = crop_and_pad_3d(vol_img, vol_mask, target_shape=target_shape)

            out_path = os.path.join(clean_out_dir, f"{cube_id}.npz")
            np.savez_compressed(out_path, image=cube_img, mask=cube_mask)

            meta_3d_records.append({
                "cube_id": cube_id,
                "patient_id": c_pid,
                "nodule_no": -1,
                "is_clean": True,
                "is_cancer": "False",
                "malignancy": 0,
                "orig_depth": len(c_slices),
                "mask_voxels": 0,
                "filepath": os.path.relpath(out_path, output_dir)
            })

    # Save meta_3d.csv
    out_df = pd.DataFrame(meta_3d_records)
    out_df.to_csv(meta_3d_csv, index=False)
    print(f"\nSuccessfully extracted and indexed {len(out_df)} 3D cubes in: {output_dir}")
    print(f"Summary saved to: {meta_3d_csv}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extract 3D Nodule Cubes for MedNeXt-3D")
    parser.add_argument('--depth', type=int, default=32, help="Volumetric depth (default: 32)")
    parser.add_argument('--size', type=int, default=64, help="Spatial height & width (default: 64)")
    parser.add_argument('--include-clean', action='store_true', default=True, help="Include clean non-nodule cubes")
    parser.add_argument('--max-clean', type=int, default=150, help="Max clean patient volumes to sample")
    parser.add_argument('--resume', action='store_true', default=True, help="Resume extraction without overwriting")
    args = parser.parse_args()
    extract_3d_cubes(args)
