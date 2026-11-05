import os
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from skimage.transform import resize
from tqdm import tqdm

def resolve_path(rel_path):
    if os.path.exists(rel_path):
        return os.path.abspath(rel_path)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.abspath(rel_path),
        os.path.join(base_dir, rel_path),
        os.path.join(base_dir, "..", rel_path),
        os.path.join(base_dir, "..", "..", rel_path),
        os.path.join("E:/LIRDC", rel_path),
        os.path.join("E:/LIRDC/Chest-CT-Scan-ML-Health-360", rel_path),
    ]
    for c in candidates:
        if os.path.exists(c):
            return os.path.abspath(c)
    cur = base_dir
    for _ in range(5):
        probe = os.path.join(cur, rel_path)
        if os.path.exists(probe):
            return os.path.abspath(probe)
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    return os.path.abspath(candidates[0])

def extract(args):
    meta_csv = resolve_path("LIDC-IDRI-Preprocessing-master/data/Meta/meta_info.csv")
    image_dir = resolve_path("LIDC-IDRI-Preprocessing-master/data/Image")
    mask_dir = resolve_path("LIDC-IDRI-Preprocessing-master/data/Mask")
    output_dir = os.path.join(os.path.dirname(__file__), "patches")
    patch_size = (64, 64)
    padding = 8

    df = pd.read_csv(meta_csv)
    df = df[df["is_cancer"] != "Ambiguous"]
    df = df[df["is_clean"] == False]

    Path(f"{output_dir}/benign").mkdir(parents=True, exist_ok=True)
    Path(f"{output_dir}/malignant").mkdir(parents=True, exist_ok=True)

    print(f"Extracting patches from {len(df)} candidate slices into {output_dir}...")
    patch_id = 0

    for _, row in tqdm(df.iterrows(), total=len(df)):
        pid = f"LIDC-IDRI-{str(row.patient_id).zfill(4)}"
        image_path = os.path.join(image_dir, pid, str(row.original_image) + ".npy")
        mask_path = os.path.join(mask_dir, pid, str(row.mask_image) + ".npy")

        if not os.path.exists(image_path) or not os.path.exists(mask_path):
            continue

        image = np.load(image_path)
        mask = np.load(mask_path)

        if np.sum(mask) == 0:
            continue

        y_indices, x_indices = np.where(mask > 0)
        if image.ndim == 3:
            image = image[0]

        height, width = image.shape
        x_min = max(x_indices.min() - padding, 0)
        x_max = min(x_indices.max() + padding, width)
        y_min = max(y_indices.min() - padding, 0)
        y_max = min(y_indices.max() + padding, height)

        patch = image[y_min:y_max, x_min:x_max]
        patch_resized = resize(patch, patch_size, mode="reflect", anti_aliasing=True)

        label = "malignant" if str(row.is_cancer).lower() == "true" else "benign"
        pid_num = int(row.patient_id) if str(row.patient_id).isdigit() else 0
        out_path = os.path.join(output_dir, label, f"{pid_num:04d}_{row.original_image}.npy")
        np.save(out_path, patch_resized.astype(np.float32))
        patch_id += 1

    print(f"Done! Successfully extracted {patch_id} nodule patches.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extract Nodule Patches for Classification")
    args = parser.parse_args()
    extract(args)

