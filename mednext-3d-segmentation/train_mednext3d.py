"""
MedNeXt-3D Volumetric Segmentation Training Pipeline
Supports MedNeXt-S, MedNeXt-B, MedNeXt-M, MedNeXt-L with 3D Convolutions
"""

import os
import argparse
import pandas as pd
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from mednext_v1 import create_mednext_v1
from dataset_3d import LIDC3DDataset
from losses_3d import HybridDiceBCELoss3D, compute_dice_score_3d


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


def train(args):
    cubes_dir = resolve_path("mednext-3d-segmentation/cubes_3d")
    meta_csv = os.path.join(cubes_dir, "meta_3d.csv")

    print("=" * 70)
    print("      MedNeXt-3D Volumetric Segmentation Training")
    print("=" * 70)
    print(f"Metadata: {meta_csv}")
    print(f"Cubes Directory: {cubes_dir}")

    df = None
    if os.path.exists(meta_csv) and os.path.getsize(meta_csv) > 0:
        try:
            df = pd.read_csv(meta_csv)
        except Exception as e:
            print(f"Notice: Reading {meta_csv} failed ({e}). Auto-rebuilding index...")

    if df is None or len(df) == 0:
        print("Auto-rebuilding 3D metadata index from .npz volume files on disk...")
        records = []
        nodules_dir = os.path.join(cubes_dir, "nodules")
        clean_dir = os.path.join(cubes_dir, "clean")

        if os.path.exists(nodules_dir):
            for fname in sorted(os.listdir(nodules_dir)):
                if fname.endswith(".npz"):
                    cube_id = fname[:-4]
                    pid = cube_id.split("_")[0]
                    records.append({
                        "cube_id": cube_id,
                        "patient_id": pid,
                        "nodule_no": 0,
                        "is_clean": False,
                        "is_cancer": "Unknown",
                        "malignancy": 0,
                        "filepath": os.path.join("nodules", fname)
                    })

        if os.path.exists(clean_dir):
            for fname in sorted(os.listdir(clean_dir)):
                if fname.endswith(".npz"):
                    cube_id = fname[:-4]
                    pid = cube_id.replace("_CLEAN", "")
                    records.append({
                        "cube_id": cube_id,
                        "patient_id": pid,
                        "nodule_no": -1,
                        "is_clean": True,
                        "is_cancer": "False",
                        "malignancy": 0,
                        "filepath": os.path.join("clean", fname)
                    })

        df = pd.DataFrame(records)
        df.to_csv(meta_csv, index=False)
        print(f"Successfully generated and saved {len(df)} records to: {meta_csv}")

    print(f"Total 3D volumes available: {len(df)} ({len(df[df['is_clean'] == False])} nodules, {len(df[df['is_clean'] == True])} clean)")

    # Patient-level split to prevent data leakage
    unique_patients = df['patient_id'].unique()
    train_patients, val_patients = train_test_split(unique_patients, test_size=0.2, random_state=42)

    train_df = df[df['patient_id'].isin(train_patients)].reset_index(drop=True)
    val_df = df[df['patient_id'].isin(val_patients)].reset_index(drop=True)
    print(f"Train set: {len(train_df)} volumes | Validation set: {len(val_df)} volumes")

    # Compute foreground pos_weight from a representative sample
    sample_df = train_df[train_df['is_clean'] == False].sample(min(200, len(train_df)), random_state=42)
    total_pos, total_neg = 0, 0
    for _, row in sample_df.iterrows():
        p = os.path.join(cubes_dir, row['filepath'])
        if os.path.exists(p):
            m = np.load(p)['mask']
            pos_count = np.sum(m > 0)
            total_pos += pos_count
            total_neg += (m.size - pos_count)

    pos_weight = min(float(total_neg / (total_pos + 1e-6)), 50.0)  # capped for stable gradients
    print(f"Class Balance: Estimated positive-to-negative voxel ratio: {pos_weight:.2f}")

    # Datasets & Loaders
    train_ds = LIDC3DDataset(train_df, cubes_dir=cubes_dir, is_train=True, augment=True)
    val_ds = LIDC3DDataset(val_df, cubes_dir=cubes_dir, is_train=False, augment=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=max(1, args.batch_size // 2),
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True
    )

    # Initialize MedNeXt Model
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    print(f"Target compute device: {device}")
    if device.type == 'cuda':
        print(f"Device name: {torch.cuda.get_device_name(0)}")

    print(f"Building MedNeXt-{args.model_size.upper()} (Kernel size: {args.kernel_size}x{args.kernel_size}x{args.kernel_size})...")
    model = create_mednext_v1(
        num_input_channels=1,
        num_classes=1,
        model_id=args.model_size.upper(),
        kernel_size=args.kernel_size,
        deep_supervision=False
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,}")

    # Loss, Optimizer & Scheduler
    loss_fn = HybridDiceBCELoss3D(pos_weight=pos_weight, dice_weight=0.5, bce_weight=0.5)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=3,
        min_lr=1e-6
    )

    # Mixed precision scaler
    use_amp = args.amp and (device.type == 'cuda')
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    best_val_dice = 0.0
    epochs_no_improve = 0
    save_dir = os.path.dirname(__file__)
    best_model_path = os.path.join(save_dir, "best_mednext3d.pth")
    last_model_path = os.path.join(save_dir, "last_mednext3d.pth")
    history_csv = os.path.join(save_dir, "training_history_mednext3d.csv")

    history = []

    print(f"\nStarting 3D training for {args.epochs} epochs with batch size {args.batch_size}...")

    for epoch in range(1, args.epochs + 1):
        # === Training Phase ===
        model.train()
        running_loss = 0.0
        train_dices = []

        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [Train]"):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            optimizer.zero_grad()

            with torch.amp.autocast('cuda', enabled=use_amp):
                outputs = model(images)
                loss = loss_fn(outputs, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            train_dices.append(compute_dice_score_3d(outputs, masks))

        train_loss = running_loss / len(train_loader)
        train_dice = np.mean(train_dices)

        # === Validation Phase ===
        model.eval()
        val_running_loss = 0.0
        val_dices = []

        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)

                with torch.amp.autocast('cuda', enabled=use_amp):
                    outputs = model(images)
                    loss = loss_fn(outputs, masks)

                val_running_loss += loss.item()
                val_dices.append(compute_dice_score_3d(outputs, masks))

        val_loss = val_running_loss / len(val_loader)
        val_dice = np.mean(val_dices)

        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_dice)

        print(
            f"[Epoch {epoch:02d}/{args.epochs:02d}] "
            f"Train Loss: {train_loss:.4f} | Train Dice: {train_dice:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Dice: {val_dice:.4f} | LR: {current_lr:.6f}"
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_dice": train_dice,
            "val_loss": val_loss,
            "val_dice": val_dice,
            "lr": current_lr
        })
        pd.DataFrame(history).to_csv(history_csv, index=False)

        # === Checkpointing & Early Stopping ===
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            epochs_no_improve = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_dice': val_dice,
                'model_size': args.model_size,
                'kernel_size': args.kernel_size,
            }, best_model_path)
            print(f"  ★ Saved new best MedNeXt-3D model (Val Dice: {val_dice:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"\n[Notice] Early stopping triggered after {epochs_no_improve} epochs without improvement.")
                break

    # Save last checkpoint
    torch.save(model.state_dict(), last_model_path)
    print(f"\nTraining completed! Best Validation Dice: {best_val_dice:.4f}")
    print(f"Best checkpoint saved to: {best_model_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train MedNeXt-3D Nodule Segmentation")
    parser.add_argument('--model-size', type=str, default='S', choices=['S', 'B', 'M', 'L'], help="MedNeXt variant")
    parser.add_argument('--kernel-size', type=int, default=3, choices=[3, 5], help="Kernel size (3 or 5)")
    parser.add_argument('--epochs', type=int, default=30, help="Number of training epochs")
    parser.add_argument('--batch-size', type=int, default=8, help="Batch size (e.g. 8 or 16)")
    parser.add_argument('--lr', type=float, default=1e-3, help="Initial AdamW learning rate")
    parser.add_argument('--weight-decay', type=float, default=1e-4, help="Weight decay")
    parser.add_argument('--patience', type=int, default=10, help="Early stopping patience")
    parser.add_argument('--workers', type=int, default=0, help="DataLoader num_workers (0 for Windows compatibility)")
    parser.add_argument('--amp', action='store_true', default=True, help="Enable automatic mixed precision")
    parser.add_argument('--device', type=str, default='cuda', help="Target device")
    args = parser.parse_args()
    train(args)

