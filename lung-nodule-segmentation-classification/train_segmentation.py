import os
import argparse
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from monai.networks.nets import UNet
from monai.metrics import DiceMetric
from monai.transforms import Resize
from monai.losses import DiceLoss
import torch.nn as nn

# === Dataset Class ===
class LIDCSegmentationDataset(Dataset):
    def __init__(self, dataframe, image_dir, mask_dir, transform=None, patch_size=(256, 256), filter_empty=True):
        self.df = dataframe.reset_index(drop=True)
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.patch_size = patch_size
        self.filter_empty = filter_empty

        if self.filter_empty:
            self.df = self.df[self.df.apply(self._has_foreground, axis=1)].reset_index(drop=True)

    def _has_foreground(self, row):
        pid = f"LIDC-IDRI-{str(row.patient_id).zfill(4)}"
        mask_path = os.path.join(self.mask_dir, pid, str(row.mask_image) + ".npy")
        if not os.path.exists(mask_path):
            return False
        mask = np.load(mask_path)
        return np.sum(mask) > 0

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        pid = f"LIDC-IDRI-{str(row.patient_id).zfill(4)}"
        img_path = os.path.join(self.image_dir, pid, str(row.original_image) + ".npy")
        mask_path = os.path.join(self.mask_dir, pid, str(row.mask_image) + ".npy")

        image = np.load(img_path).astype(np.float32)
        mask = np.load(mask_path).astype(np.uint8)

        image = (image - image.min()) / (image.max() - image.min() + 1e-6)
        image = np.expand_dims(image, axis=0)
        mask = np.expand_dims(mask, axis=0)

        image, mask = self._crop_patch_around_mask(image, mask)

        if self.transform:
            image, mask = self.transform(image, mask)

        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).float()
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask).float()

        return image, mask

    def _crop_patch_around_mask(self, image, mask):
        _, H, W = image.shape
        ph, pw = self.patch_size
        pos = np.argwhere(mask[0] > 0)
        if pos.size == 0:
            center_y, center_x = H // 2, W // 2
        else:
            y_min, x_min = pos.min(axis=0)
            y_max, x_max = pos.max(axis=0)
            center_y = (y_min + y_max) // 2
            center_x = (x_min + x_max) // 2

        y1 = max(center_y - ph // 2, 0)
        y2 = min(y1 + ph, H)
        x1 = max(center_x - pw // 2, 0)
        x2 = min(x1 + pw, W)

        y1 = y2 - ph if y2 - y1 < ph else y1
        x1 = x2 - pw if x2 - x1 < pw else x1

        return image[:, y1:y2, x1:x2], mask[:, y1:y2, x1:x2]

# === Loss ===
class WeightedDiceBCELoss(nn.Module):
    def __init__(self, pos_weight=1.0, dice_weight=0.5, bce_weight=0.5):
        super().__init__()
        self.dice = DiceLoss(sigmoid=True)
        self.register_buffer('pos_weight', torch.tensor([pos_weight]))
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight

    def forward(self, pred, target):
        bce_loss_fn = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight.to(pred.device))
        bce_loss = bce_loss_fn(pred, target)
        dice_loss = self.dice(pred, target)
        return self.dice_weight * dice_loss + self.bce_weight * bce_loss

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
    # Walk upward looking for directory containing rel_path
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

def train(args):
    meta_csv = resolve_path("LIDC-IDRI-Preprocessing-master/data/Meta/meta_info.csv")
    image_dir = resolve_path("LIDC-IDRI-Preprocessing-master/data/Image")
    mask_dir = resolve_path("LIDC-IDRI-Preprocessing-master/data/Mask")

    print(f"Loading metadata from: {meta_csv}")
    df = pd.read_csv(meta_csv)
    df = df[df["is_cancer"] != "Ambiguous"]
    df = df[df["is_clean"] == False]

    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    print(f"Dataset split: {len(train_df)} train slices, {len(val_df)} validation slices")

    # Compute pos_weight from sample
    total_pos, total_neg = 0, 0
    sample_df = train_df.sample(min(1000, len(train_df)), random_state=42)
    for _, row in sample_df.iterrows():
        pid = f"LIDC-IDRI-{str(row.patient_id).zfill(4)}"
        mask_path = os.path.join(mask_dir, pid, str(row.mask_image) + ".npy")
        if os.path.exists(mask_path):
            mask = np.load(mask_path)
            total_pos += np.sum(mask)
            total_neg += mask.size - np.sum(mask)

    pos_weight = total_neg / (total_pos + 1e-6)
    print(f"Recommended pos_weight: {pos_weight:.2f}")

    def monai_transforms(image, mask):
        image = torch.tensor(image)
        mask = torch.tensor(mask)
        resize = Resize(spatial_size=(256, 256))
        return resize(image), resize(mask)

    train_ds = LIDCSegmentationDataset(train_df, image_dir, mask_dir, transform=monai_transforms)
    val_ds = LIDCSegmentationDataset(val_df, image_dir, mask_dir, transform=monai_transforms)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=1)

    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    print(f"Using device: {device}")

    model = UNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(device)

    loss_fn = WeightedDiceBCELoss(pos_weight=pos_weight, dice_weight=0.5, bce_weight=0.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    dice_metric = DiceMetric(include_background=False, reduction="mean")

    best_dice = 0.0
    epochs_no_improve = 0

    print(f"Starting UNet segmentation training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = loss_fn(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)

        # Validation
        model.eval()
        dice_metric.reset()
        with torch.no_grad():
            for val_images, val_labels in val_loader:
                val_images, val_labels = val_images.to(device), val_labels.to(device)
                val_outputs = torch.sigmoid(model(val_images))
                val_outputs = (val_outputs > 0.5).float()
                dice_metric(y_pred=val_outputs, y=val_labels)

        val_dice = dice_metric.aggregate().item()
        print(f"[Epoch {epoch+1}/{args.epochs}] Train Loss: {avg_loss:.4f} | Val Dice: {val_dice:.4f}")
        scheduler.step(val_dice)

        if val_dice > best_dice:
            best_dice = val_dice
            epochs_no_improve = 0
            out_model = os.path.join(os.path.dirname(__file__), "best_model.pth")
            torch.save(model.state_dict(), out_model)
            print(f"  --> Saved new best model (Dice: {val_dice:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train UNet Lung Nodule Segmentation")
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    train(args)

