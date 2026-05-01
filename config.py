"""
Shared configuration for all three models.
This ensures fair comparison between ResNet-50, EfficientNet-B4, and ViT.
"""
import os

# ── MUST be set BEFORE importing datasets ──────────────
os.environ["HF_HOME"] = "E:\\huggingface_cache"
os.environ["HF_DATASETS_CACHE"] = "E:\\huggingface_cache\\datasets"

import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from datasets import load_dataset

# ── Random seed for reproducibility ─────────────────────
SEED = 42

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ── Shared training hyperparameters ─────────────────────
NUM_EPOCHS    = 5
LEARNING_RATE = 1e-4
LOSS_WEIGHTS  = {'style': 0.7, 'genre': 0.3}

# ── Class counts ────────────────────────────────────────
NUM_STYLES = 27
NUM_GENRES = 11

# ── Data split ratios ───────────────────────────────────
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15

# ── Save path ───────────────────────────────────────────
SAVE_DIR = "E:\\wikiart_project"

# ── Get shared train/val/test indices ───────────────────
def get_split_indices(total_size, seed=SEED):
    """Returns shuffled indices split into train/val/test."""
    set_seed(seed)
    indices = list(range(total_size))
    random.shuffle(indices)

    train_end = int(TRAIN_RATIO * total_size)
    val_end   = int((TRAIN_RATIO + VAL_RATIO) * total_size)

    return {
        'train': indices[:train_end],
        'val':   indices[train_end:val_end],
        'test':  indices[val_end:],
    }

# ── Image transforms (only image_size differs per model) ─
def get_transforms(image_size):
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    return train_transform, eval_transform

# ── Shared Dataset class ────────────────────────────────
class WikiArtDataset(Dataset):
    def __init__(self, hf_dataset, indices, transform):
        self.data      = hf_dataset
        self.indices   = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        sample = self.data[self.indices[idx]]
        image  = sample['image'].convert('RGB')
        image  = self.transform(image)
        style  = sample['style']
        genre  = sample['genre']
        return image, style, genre