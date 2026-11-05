# Lung Nodule Segmentation & Malignancy Classification

This module implements an end-to-end deep learning pipeline for pulmonary nodule analysis on thoracic CT scans from the LIDC-IDRI dataset.

## Pipeline Architecture

```
                       Preprocessed 2D Slices
                                 │
           ┌─────────────────────┴─────────────────────┐
           ▼                                           ▼
   Stage 1: UNet Segmentation                  Stage 2: Patch Extraction
   (Detects nodule contours)                   (Crops 64x64 nodule regions)
           │                                           │
           ▼                                           ▼
    best_model.pth                             patches/{benign,malignant}
                                                       │
                                                       ▼
                                               Stage 3: ResNet-18 Classifier
                                               (Predicts Benign vs. Malignant)
                                                       │
                                                       ▼
                                               best_classifier.pth
```

---

## 1. Stage 1: UNet Nodule Segmentation

Segments nodule contours from thoracic CT slices using a 2D UNet with residual units and hybrid Weighted Dice + BCE Loss.

### Run Training:
```bash
python train_segmentation.py --epochs 30 --batch-size 16 --device cuda
```

### CLI Arguments:
| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--epochs` | `int` | `50` | Total number of training epochs |
| `--batch-size` | `int` | `8` | Training batch size (increase to 16/32 if GPU VRAM allows) |
| `--lr` | `float` | `1e-3` | Initial Adam learning rate |
| `--patience` | `int` | `10` | Early stopping patience (epochs without validation improvement) |
| `--device` | `str` | `cuda` | Target compute device (`cuda` or `cpu`) |

### Key Features:
- **Foreground Balancing (`pos_weight`)**: Automatically estimates positive-to-negative pixel ratio to counteract class imbalance.
- **Dynamic Scheduler**: `ReduceLROnPlateau` halves learning rate when validation Dice plateaus.
- **Best Model Checkpointing**: Saves `best_model.pth` based on highest validation Dice score.

---

## 2. Stage 2: Nodule Patch Extraction

Extracts $64 \times 64$ centered nodule crops from preprocessed scans according to ground truth consensus masks.

### Run Extraction:
```bash
python extract_patches.py
```

### Output:
- `patches/benign/`: Normalized $64 \times 64$ numpy patches from nodules scored with malignancy $\le 2$.
- `patches/malignant/`: Normalized $64 \times 64$ numpy patches from nodules scored with malignancy $\ge 4$.
*(Ambiguous nodules with score 3 are excluded from binary classification).*

---

## 3. Stage 3: ResNet-18 Malignancy Classification

Classifies nodule crops as **Benign** vs. **Malignant** using a fine-tuned ResNet-18 backbone.

### Run Training:
```bash
python train_classifier.py --epochs 25 --batch-size 32 --device cuda
```

### CLI Arguments:
| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--epochs` | `int` | `30` | Maximum training epochs |
| `--batch-size` | `int` | `32` | Batch size for training and validation |
| `--lr` | `float` | `1e-3` | Initial learning rate |
| `--patience` | `int` | `5` | Early stopping patience |
| `--device` | `str` | `cuda` | Target compute device (`cuda` or `cpu`) |

### Output:
- Saves `best_classifier.pth` upon achieving improved validation accuracy.

---

## 4. Interactive Jupyter Notebook

For step-by-step experimentation and multi-panel visual inference:
- Open [`lung-nodule-segmentation-classification.ipynb`](./lung-nodule-segmentation-classification.ipynb)
- Run cells sequentially to visualize CT slices, ground truth masks, predicted segmentation masks, and malignancy classification outputs side by side.

