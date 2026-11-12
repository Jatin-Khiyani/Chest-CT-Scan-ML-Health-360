"""
3D Volumetric Segmentation Losses & Evaluation Metrics for MedNeXt-3D
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftDiceLoss3D(nn.Module):
    """
    Volumetric Soft Dice Loss with Laplace smoothing.
    Expects raw logits or sigmoid probabilities.
    """
    def __init__(self, smooth: float = 1e-5, apply_sigmoid: bool = True):
        super().__init__()
        self.smooth = smooth
        self.apply_sigmoid = apply_sigmoid

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.apply_sigmoid:
            probs = torch.sigmoid(logits)
        else:
            probs = logits

        # Flatten spatial dims: (B, C, D, H, W) -> (B, -1)
        probs_flat = probs.view(probs.size(0), -1)
        targets_flat = targets.view(targets.size(0), -1).float()

        intersection = (probs_flat * targets_flat).sum(dim=1)
        cardinality = probs_flat.sum(dim=1) + targets_flat.sum(dim=1)

        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - dice.mean()


class HybridDiceBCELoss3D(nn.Module):
    """
    Hybrid 3D Loss: Weighted Dice + BCE with optional pos_weight for class imbalance.
    """
    def __init__(
        self,
        pos_weight: float = 1.0,
        dice_weight: float = 0.5,
        bce_weight: float = 0.5,
        smooth: float = 1e-5
    ):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.dice_loss = SoftDiceLoss3D(smooth=smooth, apply_sigmoid=True)

        if pos_weight is not None and pos_weight > 1.0:
            pw_tensor = torch.tensor([pos_weight], dtype=torch.float32)
            self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=pw_tensor)
        else:
            self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Move pos_weight to same device as logits if needed
        if hasattr(self.bce_loss, 'pos_weight') and self.bce_loss.pos_weight is not None:
            if self.bce_loss.pos_weight.device != logits.device:
                self.bce_loss.pos_weight = self.bce_loss.pos_weight.to(logits.device)

        dice = self.dice_loss(logits, targets)
        bce = self.bce_loss(logits, targets.float())
        return self.dice_weight * dice + self.bce_weight * bce


def compute_dice_score_3d(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    smooth: float = 1e-5
) -> float:
    """
    Computes volumetric Dice similarity coefficient between predicted logits and target masks.
    """
    with torch.no_grad():
        preds = (torch.sigmoid(logits) > threshold).float()
        targets = (targets > threshold).float()

        preds_flat = preds.view(preds.size(0), -1)
        targets_flat = targets.view(targets.size(0), -1)

        intersection = (preds_flat * targets_flat).sum(dim=1)
        cardinality = preds_flat.sum(dim=1) + targets_flat.sum(dim=1)

        # For samples where both pred and target are completely empty, dice is 1.0
        both_empty = (cardinality == 0)
        dice = (2.0 * intersection + smooth) / (cardinality + smooth)
        dice[both_empty] = 1.0

        return dice.mean().item()

