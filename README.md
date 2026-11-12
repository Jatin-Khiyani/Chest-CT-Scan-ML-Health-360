# MLHealth360: Chest CT Scan Preprocessing & Nodule Analysis Pipeline

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.8%2Bcu128-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![MONAI](https://img.shields.io/badge/MONAI-1.3%2B-5B2C6F)](https://monai.io/)
[![Dataset](https://img.shields.io/badge/Dataset-LIDC--IDRI-blue)](https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI)
[![Hardware](https://img.shields.io/badge/GPU-CUDA%20Accelerated-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-zone)

This repository provides an end-to-end medical deep learning workflow for **pulmonary nodule preprocessing, segmentation, and malignancy classification** using thoracic CT scans from the **LIDC-IDRI** (Lung Image Database Consortium) dataset.

---

## 📑 Table of Contents
- [Overview & Architecture](#-overview--architecture)
- [Repository Structure](#-repository-structure)
- [Hardware & Environment Setup](#-hardware--environment-setup)
- [Dataset Preprocessing Pipeline](#-dataset-preprocessing-pipeline)
- [Model Training & Inference](#-model-training--inference)
  - [Stage 1: UNet Nodule Segmentation](#stage-1-unet-nodule-segmentation)
  - [Stage 2: Patch Extraction](#stage-2-nodule-patch-extraction)
  - [Stage 3: ResNet-18 Malignancy Classification](#stage-3-resnet-18-malignancy-classification)
  - [Stage 4: Interactive Visual Inference](#stage-4-interactive-visual-inference)
- [Data Directory Conventions](#-data-directory-conventions)
- [Team Collaboration & Branching](#-team-collaboration--branching)

---

## 🔬 Overview & Architecture

The pipeline consists of two primary subsystems:

```
[Raw DICOM Scans (LIDC-IDRI)]
           │
           ▼
[Preprocessing Subsystem]
 ├── pylidc annotation consensus (≥50% agreement)
 ├── Lung parenchyma segmentation (Hounsfield thresholding + morphological cleaning)
 ├── Nodule centering and 2D slice extraction
 └── Metadata indexing (is_cancer, malignancy, clean slices)
           │
           ├──────────────────────────────┬──────────────────────────────┐
           ▼                              ▼                              ▼
 [Segmented 2D Images]           [Consensus 2D Masks]           [Metadata CSV]
           │                              │                              │
           └──────────────┬───────────────┘                              │
                          │                                              │
                          ▼                                              ▼
           [Stage 1: UNet Segmentation]                    [Stage 2: Patch Extraction]
            • Residual UNet backbone                        • 64×64 nodule crops
            • Weighted Dice + BCE Loss                      • Benign (≤2) vs Malignant (≥4)
            • pos_weight foreground balancing                             │
            • Early stopping & checkpointing                              ▼
                          │                                [Stage 3: ResNet-18 Classifier]
                          │                                 • Transfer learning backbone
                          │                                 • Binary malignancy prediction
                          │                                               │
                          └───────────────────────┬───────────────────────┘
                                                  ▼
                                   [Stage 4: Multi-Panel Visual Inference]
```

---

## 📂 Repository Structure

```text
Chest-CT-Scan-ML-Health-360/
├── LIDC-IDRI-Preprocessing-master/       # Preprocessing Subsystem
│   ├── prepare_dataset.py                # DICOM to 2D numpy slice preprocessor
│   ├── config_file_create.py             # Generates pylidc and path configuration
│   ├── utils.py                          # Lung segmentation & visualization utilities
│   ├── pylidc.conf                       # pylidc configuration file
│   ├── lung.conf                         # Preprocessing directory settings
│   ├── PREPROCESSING_README.md           # Preprocessing module documentation
│   ├── data/                             # Processed dataset output directories
│   │   ├── Image/                        # Nodule CT slices (.npy)
│   │   ├── Mask/                         # Consensus nodule masks (.npy)
│   │   ├── Clean/Image/                  # Healthy / non-nodule CT slices (.npy)
│   │   ├── Clean/Mask/                   # All-zero masks (.npy)
│   │   └── Meta/meta_info.csv            # Master slice metadata index (14,684 records)
│   ├── figures/                          # Visual verification outputs
│   └── notebook/make_label.ipynb         # Nodule labeling & exploration notebook
│
├── lung-nodule-segmentation-classification/ # 2D Segmentation & Classification
│   ├── train_segmentation.py             # 2D UNet segmentation training CLI
│   ├── extract_patches.py                # 64x64 nodule patch extractor CLI
│   ├── train_classifier.py               # ResNet-18 malignancy classifier CLI
│   ├── lung-nodule-segmentation-classification.ipynb # Interactive notebook
│   ├── output.png                        # Sample segmentation & classification result
│   ├── README.md                         # 2D pipeline architecture guide
│   └── patches/                          # Extracted nodule crops
│
├── mednext-3d-segmentation/              # MedNeXt (3D) Volumetric SOTA Segmentation
│   ├── mednext_v1.py                     # Standalone MedNeXt-3D architecture (MICCAI 2023)
│   ├── prepare_3d_dataset.py             # 3D volume cube extractor (32x64x64)
│   ├── dataset_3d.py                     # PyTorch 3D dataset with volumetric augmentations
│   ├── losses_3d.py                      # 3D Hybrid Dice + BCE Loss & metrics
│   ├── train_mednext3d.py                # CUDA training with mixed-precision AMP
│   ├── visualize_3d.py                   # Multi-planar 3D visualizer (Axial/Coronal/Sagittal)
│   ├── README.md                         # 3D architecture & hyperparameter guide
│   └── cubes_3d/                         # Extracted 3D cubes (.npz) & meta_3d.csv
│
├── requirements.txt                      # Project-wide dependency specification
├── .gitignore                            # Rules for Python, PyTorch, and medical data
└── README.md                             # Repository master documentation
```

---

## ⚙ Hardware & Environment Setup

### 1. Clone Repository & Create Virtual Environment
```bash
git clone https://github.com/Jatin-Khiyani/Chest-CT-Scan-ML-Health-360.git
cd Chest-CT-Scan-ML-Health-360

# Create virtual environment
python -m venv .venv

# Activate environment
# On Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# On Linux/macOS:
source .venv/bin/activate
```

> **Windows PowerShell Execution Policy Note:** If activation is restricted, run:
> ```powershell
> Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
> ```

### 2. Install PyTorch with CUDA (Recommended)
For NVIDIA GPUs (e.g. RTX 40/50 series, RTX PRO Blackwell, Ada, Ampere):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

### 3. Install All Remaining Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure `pylidc`
Create a file named `.pylidcrc` in your user home directory (`~/.pylidcrc` on Linux or `C:\Users\<username>\.pylidcrc` on Windows) containing:
```ini
[dicom]
path = /absolute/path/to/raw/lidc_idri
warn = False
```

---

## 🔄 Dataset Preprocessing Pipeline

To extract 2D segmented CT slices and consensus masks from raw DICOM series:

```bash
cd LIDC-IDRI-Preprocessing-master
python prepare_dataset.py --resume
```

### Key Capabilities:
- **Resilient Resume (`--resume`)**: Preserves already-processed patients in `data/Meta/meta_info.csv` and skips re-processing completed series.
- **Slice Alignment Validation**: Automatically validates that DICOM volume depths match XML annotation coordinates (`shape[2] == mask.shape[2]`) to prevent crashes from incomplete TCIA downloads.
- **Consensus Filtering**: Leverages `pylidc.consensus` with 50% radiologist agreement level (`c_level=0.5`).
- **Clean Slice Generation**: Slices non-nodule lung volumes (`is_clean=True`) for negative-sample balancing.

---

## 🚀 Model Training & Inference

All deep learning training scripts are located in `lung-nodule-segmentation-classification/` and can be run directly from terminal or via Jupyter.

### Stage 1: UNet Nodule Segmentation

Trains a 2D UNet with residual units using hybrid Weighted Dice + Binary Cross-Entropy Loss:

```bash
cd lung-nodule-segmentation-classification
python train_segmentation.py --epochs 30 --batch-size 16 --device cuda
```

- **Early Stopping:** Automatically stops training if validation Dice score does not improve after 10 epochs.
- **Output:** Saves the checkpoint with the highest validation Dice score as `best_model.pth`.

---

### Stage 2: Nodule Patch Extraction

Extracts $64 \times 64$ nodule regions centered around the radiologist consensus bounding box:

```bash
python extract_patches.py
```

- Slices with `malignancy <= 2` are saved to `patches/benign/`.
- Slices with `malignancy >= 4` are saved to `patches/malignant/`.
- Ambiguous nodules (`malignancy == 3`) are safely excluded.

---

### Stage 3: ResNet-18 Malignancy Classification

Fine-tunes a pretrained ResNet-18 model to classify nodule patches as Benign or Malignant:

```bash
python train_classifier.py --epochs 25 --batch-size 32 --device cuda
```

- **Metric:** Tracks top-1 validation accuracy and Cross-Entropy Loss.
- **Output:** Saves the highest accuracy checkpoint as `best_classifier.pth`.

---

### Stage 4: Interactive Visual Inference

Launch the interactive Jupyter notebook to inspect slice-level segmentations and classification predictions:

```bash
jupyter notebook lung-nodule-segmentation-classification.ipynb
```

The notebook provides:
- Raw CT scan visualization with radiologist consensus contour overlays.
- Dynamic pos-weight class balancing calculation.
- Side-by-side comparison of **Original Slice**, **Ground Truth Mask**, **UNet Predicted Segmentation**, and **ResNet Malignancy Prediction**.

---

## 📁 Data Directory Conventions

Due to size constraints (100+ GB raw DICOM, 15+ GB preprocessed numpy arrays), large binary data files are excluded from git via `.gitignore`. 

The folder structure is preserved via `.gitkeep`:
```
LIDC-IDRI-Preprocessing-master/data/
├── Image/             # Populated by prepare_dataset.py
├── Mask/              # Populated by prepare_dataset.py
├── Clean/Image/       # Populated by prepare_dataset.py
├── Clean/Mask/        # Populated by prepare_dataset.py
└── Meta/
    └── meta_info.csv  # Included in git as reference index (14,684 records)
```

---

## 👥 Team Collaboration & Branching

This project is actively developed by the MLHealth360 team using dedicated feature/member branches:

```text
origin/main
│
├── origin/Jatin      (Preprocessing & Model Architecture Pipeline)
├── origin/Prarambh   (Data Exploration & Validation)
├── origin/Archit     (Segmentation Tuning)
└── origin/Kashika    (Classification & Evaluation)
```

### Git Workflow:
1. Always pull latest changes from your working branch:
   ```bash
   git pull origin <your-branch-name>
   ```
2. Commit your code modifications (checkpoints and data arrays will be automatically ignored by `.gitignore`):
   ```bash
   git add <modified-files>
   git commit -m "Your descriptive commit message"
   ```
3. Push to your branch and create a Pull Request to `main` for code review.
