import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import torchvision.models as models
from torchvision import transforms

class NPYImageFolder(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.classes = ["benign", "malignant"]
        self.filepaths = []
        self.labels = []

        for label_idx, class_name in enumerate(self.classes):
            class_path = os.path.join(root_dir, class_name)
            if not os.path.exists(class_path):
                continue
            for fname in os.listdir(class_path):
                if fname.endswith(".npy"):
                    self.filepaths.append(os.path.join(class_path, fname))
                    self.labels.append(label_idx)

        self.transform = transform

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        img = np.load(self.filepaths[idx]).astype(np.float32)
        img = (img - img.min()) / (img.max() - img.min() + 1e-6)
        img = np.stack([img] * 3, axis=0)  # (3, H, W)
        img = torch.tensor(img)

        if self.transform:
            img = self.transform(img)

        label = self.labels[idx]
        return img, label

def train(args):
    patch_dir = os.path.join(os.path.dirname(__file__), "patches")
    candidates = [
        patch_dir,
        os.path.abspath("patches"),
        os.path.abspath("lung-nodule-segmentation-classification/patches"),
        os.path.join("..", "patches"),
        os.path.join("E:/LIRDC", "lung-nodule-segmentation-classification-main", "lung-nodule-segmentation-classification-main", "patches"),
    ]
    for c in candidates:
        if os.path.exists(c) and (os.path.exists(os.path.join(c, "benign")) or os.path.exists(os.path.join(c, "malignant"))):
            patch_dir = os.path.abspath(c)
            break

    if not os.path.exists(patch_dir):
        print(f"Error: {patch_dir} does not exist. Run extract_patches.py first!")
        return

    resize_transform = transforms.Compose([
        transforms.Resize((224, 224))
    ])

    dataset = NPYImageFolder(patch_dir, transform=resize_transform)
    if len(dataset) == 0:
        print(f"Error: No patches found in {patch_dir}. Run extract_patches.py first!")
        return

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    print(f"Dataset: {len(dataset)} total patches ({train_size} train, {val_size} val). Using device: {device}")

    try:
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    except Exception:
        model = models.resnet18(pretrained=True)

    model.fc = nn.Linear(model.fc.in_features, 2)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    best_val_acc = 0.0
    epochs_no_improve = 0

    print(f"Starting ResNet-18 classifier training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)

        train_acc = correct_train / (total_train + 1e-6)
        avg_loss = running_loss / len(train_loader)

        # Validation
        model.eval()
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                correct_val += (preds == labels).sum().item()
                total_val += labels.size(0)

        val_acc = correct_val / (total_val + 1e-6)
        scheduler.step(val_acc)

        print(f"[Epoch {epoch+1}/{args.epochs}] Train Loss: {avg_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            out_model = os.path.join(os.path.dirname(__file__), "best_classifier.pth")
            torch.save(model.state_dict(), out_model)
            print(f"  --> Saved new best classifier (Val Acc: {val_acc:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train ResNet Nodule Malignancy Classifier")
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    train(args)

