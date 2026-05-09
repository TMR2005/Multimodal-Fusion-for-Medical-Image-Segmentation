# Vision-Language Multimodal Brain Tumor Segmentation

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)
![BraTS 2020](https://img.shields.io/badge/Dataset-BraTS%202020-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## Table of Contents

1. [Overview](#overview)
2. [Model Architecture](#model-architecture)
   - [2.5D Segmentation Backbone](#25d-segmentation-backbone)
   - [Vision-Language Bottleneck](#vision-language-bottleneck)
   - [Loss Formulation](#loss-formulation)
3. [Training Procedure](#training-procedure)
4. [Evaluation Metrics](#evaluation-metrics)
5. [Results](#results)
6. [Usage](#usage)
7. [Hardware Support](#hardware-support)

---

## Overview

This repository implements a **vision-language multimodal segmentation network** that fuses **FLAIR MRI volumes** with **Clinical Text Reports** to produce high-resolution brain tumor segmentations. By aligning the spatial latent space of the visual encoder with the semantic embedding space of a pre-trained language model, the network leverages radiological context to improve anatomical boundary delineation.

### Key Contributions

- **2.5D Spatial Context** — Efficiently bypasses 3D memory constraints and disk I/O bottlenecks by dynamically extracting the optimal 3-slice context stack per volume.
- **Latent Contrastive Alignment** — Maps the U-Net bottleneck features directly into a 512-dimensional CLIP-style text embedding space.
- **Hardware-Agnostic Design** — Fully optimized for macOS Apple Silicon (Metal Performance Shaders — MPS) alongside standard CUDA environments.
- High-performance segmentation on the **BraTS 2020** dataset.

---

## Model Architecture

### 2.5D Segmentation Backbone

The visual pathway utilizes a modified U-Net architecture adapted for 2.5D inputs. Instead of processing full 3D volumes or single 2D slices, the network ingests a 3-channel stack representing `[z-1, z, z+1]`, providing local 3D spatial context with 2D computational efficiency.

| Component | Architecture | Output Channels |
|---|---|---|
| **Encoder** | 3-stage Convolutional (3×3) with Max Pooling | 3 → 32 → 128 → 256 |
| **Bottleneck** | Deep Feature Extractor (Latent Projection Source) | 256 → 512 → 256 |
| **Decoder** | 3-stage Transpose Convolutions with Skip Connections | 256 → 128 → 64 → 3 |

### Vision-Language Bottleneck

To inject semantic context into the visual pipeline, the architecture employs a frozen `sentence-transformers/all-MiniLM-L6-v2` encoder.

1. **Text Encoding** — Clinical reports are tokenized and processed through the frozen transformer to generate a stable 512-dimensional semantic embedding.
2. **Visual Projection** — The 256-channel spatial bottleneck from the U-Net is passed through an Adaptive Average Pool and MLP projection head to map it into the same 512-dimensional space.
3. **Alignment** — A contrastive loss function forces the projected visual features to align with their corresponding clinical text embeddings within the shared latent space.

### Loss Formulation

The network is optimized using a dual-objective loss function:

- **Segmentation Loss** — Standard CrossEntropy Loss on the final spatial output mask.
- **Contrastive Loss** — Symmetric CrossEntropy (Image-to-Text and Text-to-Image) applied to the cosine similarity matrix of the projected embeddings, scaled by a temperature hyperparameter ($\tau = 0.07$).
- **Total Loss:**

$$L_{\text{total}} = L_{\text{seg}} + \lambda \cdot L_{\text{contrastive}} \quad (\lambda = 0.1)$$

---

## Training Procedure

### 1. Pre-processing

MRI arrays (`.npy`) and text reports (`.txt`) are dynamically mapped:

- Volumes are scanned to locate the slice index ($z$) containing the maximum tumor area.
- The $z-1$, $z$, $z+1$ slices are extracted, Min-Max normalized, and resized to $256 \times 256$ using bilinear interpolation.

### 2. Optimization

| Hyperparameter | Value |
|---|---|
| Optimizer | Adam |
| Learning Rate | `1e-4` |
| Batch Size | 16 |
| Hardware | Apple M-Series (MPS) / NVIDIA (CUDA) |

### 3. Training Schedule

- **15 epochs** total.
- Model checkpoints are evaluated at the end of each epoch.
- The optimal state dictionary is saved based on the highest **Validation Dice Score**.

---

## Evaluation Metrics

| Metric | Description |
|---|---|
| **Dice Similarity Coefficient (DSC)** | Primary metric for target overlap, ignoring the background class. |
| **Intersection-over-Union (IoU)** | Stricter penalty for boundary misalignment (Jaccard Index). |
| **Hausdorff Distance 95 (HD95)** | Measures worst-case spatial outlier errors between predicted and ground truth boundaries. |

---

## Results

Evaluation on the held-out validation split (**80 patient volumes**) using optimal weights from **Epoch 12**:

| Metric | Score |
|---|---|
| **Average Dice Score** | **0.8280** |
| **Average IoU Score** | 0.7080 |
| **Average HD95** | 61.63 |

> The results demonstrate highly competitive structural accuracy, successfully leveraging the multimodal bottleneck to achieve **>0.82 Dice** on the BraTS 2020 target structures.

---

## Usage

### Directory Structure

Ensure your data matches the following hierarchy before execution:

```
project_root/
├── data/
│   ├── FLAIR_BRATS2020_split/
│   │   ├── train/
│   │   │   ├── images/     # image_0.npy, image_1.npy, ...
│   │   │   └── masks/      # mask_0.npy,  mask_1.npy,  ...
│   │   └── val/
│   └── TextBRats/
│       └── TextBraTSData/
│           └── BraTS20_Training_001/   # 001.txt
├── train.py
└── results/
```

### Installation & Execution

```bash
# Install dependencies
pip install torch torchvision transformers scipy scikit-learn matplotlib tqdm pillow

# Execute the training pipeline
python train.py
```

The script will automatically:

1. Detect the optimal hardware accelerator.
2. Execute the 15-epoch training loop.
3. Save the best weights to `./results/best_model_weights.pt`.
4. Generate overlay `.png` visualizations for the top and bottom predictions.

---

## Hardware Support

This codebase utilizes an abstraction layer for PyTorch device placement and is guaranteed to run natively on the following platforms:

| Platform | Backend | Notes |
|---|---|---|
| **macOS** | Apple Silicon (M1/M2/M3/M4) via MPS | Multiprocessing (`num_workers`) is safely disabled by default to prevent macOS `spawn` context crashes. |
| **Linux / Windows** | NVIDIA GPUs via CUDA | Full multi-worker DataLoader support. |
| **Fallback** | CPU | Standard execution on any machine. |