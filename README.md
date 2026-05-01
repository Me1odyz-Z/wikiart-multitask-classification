# WikiArt Multi-Task Classification

Multi-task deep learning for artistic style and genre classification on the WikiArt dataset, comparing CNN (ResNet-50, EfficientNet-B4) and Transformer (ViT-B/16) architectures.

**Course Project for EE559 (USC, Spring 2026)**

**Team:** Yiming Zheng, Siyang Wang

## Overview

We build a multi-task classification system that simultaneously predicts the artistic style (27 classes) and genre (11 classes) of a painting. Three pre-trained backbones share a common feature extractor and branch into two task-specific heads, trained end-to-end with a weighted sum of cross-entropy losses.

## Dataset

- **Source:** [huggan/wikiart](https://huggingface.co/datasets/huggan/wikiart) on HuggingFace
- **Size:** 81,444 artwork images
- **Labels:** 27 styles, 11 genres
- **Challenge:** Severe class imbalance (133x ratio between most and least frequent style)

## Results

### Main Comparison (Test Set, 5 epochs)

| Model | Style Top-1 | Style Top-5 | Style Macro-F1 | Genre Top-1 | Genre Macro-F1 |
|-------|-------------|-------------|----------------|-------------|----------------|
| ResNet-50 | 59.13% | 93.89% | 0.5601 | 67.76% | 0.6640 |
| **EfficientNet-B4** | **60.30%** | **94.37%** | **0.5888** | 67.85% | 0.6603 |
| ViT-B/16 | 57.60% | 92.71% | 0.5297 | 65.87% | 0.6503 |

### Ablation 1: Single-task vs Multi-task (ResNet-50, 3 epochs)

| Setting | Val Acc |
|---------|---------|
| Single-task (style only) | 58.47% |
| Multi-task (style + genre) | **60.07%** |

Multi-task learning improves style accuracy by 1.6%, supporting the hypothesis that style and genre share complementary visual features.

### Ablation 2: Class Weighting (ResNet-50, 3 epochs)

| Setting | Val Acc | Macro-F1 |
|---------|---------|----------|
| No weights (5 epoch baseline) | 59.16% | 0.5610 |
| Inverse frequency | 51.78% | 0.4976 |
| **Square-root inverse** | **58.56%** | **0.5532** |

Aggressive inverse-frequency reweighting destabilizes training, while smoother sqrt weighting recovers near-baseline performance.

## Setup

### Requirements
- Python 3.11
- PyTorch 2.0+ with CUDA
- 8GB+ GPU memory (RTX 3070 used)

### Install

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers datasets timm pillow matplotlib scikit-learn pandas tqdm seaborn jupyter
```

### Reproduce

Run the notebooks in this order:
1. `wikiart_exploration.ipynb` - Data exploration
2. `train_resnet.ipynb` - ResNet-50 training
3. `train_efficientnet.ipynb` - EfficientNet-B4 training
4. `train_vit.ipynb` - ViT-B/16 training
5. `evaluate_models.ipynb` - Evaluate on test set
6. `ablation_single_task.ipynb` - Ablation 1
7. `ablation_class_weights.ipynb` - Ablation 2

## Authors

- Yiming Zheng
- Siyang Wang
