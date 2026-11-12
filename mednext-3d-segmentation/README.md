# MedNeXt-3D: Volumetric ConvNeXt Architecture for Lung Nodule Segmentation

[![Paper](https://img.shields.io/badge/MICCAI'23-Paper-red.svg)](https://arxiv.org/abs/2303.09975)
[![Framework](https://img.shields.io/badge/PyTorch-2.8%2Bcu128-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Hardware](https://img.shields.io/badge/GPU-CUDA%20Accelerated-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-zone)

This module provides a standalone, fully 3D implementation of **MedNeXt** (*Roy et al., MICCAI 2023*) applied to pulmonary nodule segmentation on the LIDC-IDRI dataset.

---

## 🏗️ Architecture Highlights

MedNeXt brings modern ConvNeXt design principles to volumetric medical imaging:

```
Input Volume: (B, 1, 32, 64, 64)
           │
           ▼
[Stem: 1x1x1 Conv3d -> 32 channels]
           │
 ┌─────────┴─────────┐
 ▼                   ▼
Encoder Stages      Decoder Stages
 (3D Depthwise Conv) (Transposed 3D Upsampling)
 (Inverted Bottleneck)(Skip Additions)
 (GroupNorm + GELU)  (Residual Connections)
 └─────────┬─────────┘
           ▼
[OutBlock: 1x1x1 Conv3d -> Binary Nodule Mask]
```

### Architectural Features:
* **Fully 3D Depthwise Separable Convolutions:** Spatial mixing via $3\times3\times3$ or $5\times5\times5$ kernels with GroupNorm.
* **Inverted Bottlenecks:** Channels expand by factor $R$ (e.g. $R=2$ or $R=4$) before non-linearity and compress back.
* **Residual Resampling:** Downsampling and upsampling blocks feature residual connections for optimal gradient flow.
* **Variants:**
  * **MedNeXt-S (Small):** 5.56M parameters — lightweight, fast training.
  * **MedNeXt-B (Base):** 10.56M parameters — compound scaling `[2, 3, 4, 4, 4, 4, 4, 3, 2]`.
  * **MedNeXt-M / MedNeXt-L (Medium/Large):** Deeper staging with gradient checkpointing.

---

## 🚀 Quickstart Workflow

### Step 1: Extract 3D Nodule Cubes
Preprocesses and groups consecutive CT slices into standardized $32 \times 64 \times 64$ volumetric cubes centered on radiologist consensus masks:

```bash
python prepare_3d_dataset.py --depth 32 --size 64 --resume
```

* **Output:** Saves `.npz` cubes to `cubes_3d/nodules/` and clean volumes to `cubes_3d/clean/`.
* **Index:** Generates `cubes_3d/meta_3d.csv` tracking metadata and voxel counts.

---

### Step 2: Train MedNeXt-3D on GPU
Launch volumetric segmentation training with automatic mixed-precision (`torch.amp.autocast`) and hybrid 3D Dice + BCE loss:

```bash
python train_mednext3d.py --model-size S --kernel-size 3 --epochs 30 --batch-size 8 --device cuda
```

#### CLI Options:
| Flag | Default | Description |
| :--- | :--- | :--- |
| `--model-size` | `S` | Variant: `S` (Small), `B` (Base), `M` (Medium), `L` (Large) |
| `--kernel-size`| `3` | Kernel size: `3` ($3\times3\times3$) or `5` (UpKern $5\times5\times5$) |
| `--batch-size` | `8` | 3D batch size (increase to 16 for RTX PRO 5000 48GB) |
| `--epochs` | `30` | Number of training epochs |
| `--lr` | `1e-3` | Initial AdamW learning rate |
| `--patience` | `10` | Early stopping patience |
| `--amp` | `True` | Mixed precision FP16/BF16 acceleration |

* **Output:** Saves best checkpoint to `best_mednext3d.pth` and metric curves to `training_history_mednext3d.csv`.

---

### Step 3: Multi-Planar Orthogonal Reconstruction (MPR)
Inspect predictions across **Axial**, **Coronal**, and **Sagittal** cross-sections side by side:

```bash
python visualize_3d.py --checkpoint best_mednext3d.pth --model-size S --device cuda
```

* **Output:** Generates `mednext3d_sample_prediction.png` showing raw CT, ground truth, and predicted masks across all 3 spatial planes.

---

## 📊 File Organization

```text
mednext-3d-segmentation/
├── mednext_v1.py                  # Standalone MedNeXt-3D PyTorch architecture
├── prepare_3d_dataset.py          # 3D volume cube extractor from LIDC dataset
├── dataset_3d.py                  # PyTorch 3D dataset with volumetric augmentations
├── losses_3d.py                   # 3D Hybrid Dice + BCE Loss & metric functions
├── train_mednext3d.py             # CUDA-accelerated training pipeline with AMP
├── visualize_3d.py                # Multi-planar 3D visualization tool
├── README.md                      # Documentation & parameter reference
└── cubes_3d/                      # Generated 3D data directory (git-ignored)
    ├── nodules/                   # Nodule .npz cubes
    ├── clean/                     # Non-nodule .npz cubes
    └── meta_3d.csv                # 3D dataset index
```

