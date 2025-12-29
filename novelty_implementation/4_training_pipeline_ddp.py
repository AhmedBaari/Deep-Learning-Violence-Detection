# %% [markdown]
# ## CELL 1: Imports & DDP Initialization
# 

# %%
# ============================================================
# CELL 1: IMPORTS & DDP INITIALIZATION
# ============================================================

"""
This cell:
1. Imports all required libraries
2. Initializes PyTorch Distributed Data Parallel (DDP) OR single-GPU mode
3. Verifies GPUs are available
4. Sets up process rank and world size

MODES:
- DDP Mode: Launch with `torchrun --nproc_per_node=8 script.py`
- Single-GPU Mode: Run directly in Jupyter (for testing)
"""

import os
import sys
import json
import pickle
import random
from pathlib import Path
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler, Subset
from torch.cuda.amp import autocast, GradScaler

# Vision
import torchvision
from torchvision import transforms
from PIL import Image

# Data
import numpy as np
import pandas as pd

# Timm for ViT
try:
    import timm
except ImportError:
    print("Installing timm library...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "timm"])
    import timm

# ============================================================
# INITIALIZE DISTRIBUTED TRAINING (OR SINGLE-GPU MODE)
# ============================================================

def setup_ddp():
    """
    Initialize PyTorch Distributed Data Parallel OR single-GPU mode.
    
    Automatically detects if running with torchrun (DDP) or Jupyter (single-GPU).
    
    Returns:
        rank: GPU rank (0 for single-GPU, 0-7 for DDP)
        world_size: Total number of GPUs (1 for single-GPU, 8 for DDP)
        use_ddp: Boolean indicating if DDP is active
    """
    # Check if DDP environment variables are set (torchrun sets these)
    use_ddp = 'RANK' in os.environ and 'WORLD_SIZE' in os.environ
    
    if use_ddp:
        # DDP mode (launched with torchrun)
        dist.init_process_group(backend='nccl')
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        torch.cuda.set_device(rank)
    else:
        # Single-GPU mode (Jupyter notebook)
        rank = 0
        world_size = 1
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
    
    return rank, world_size, use_ddp

# Initialize (auto-detect mode)
rank, world_size, use_ddp = setup_ddp()
device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')

# Print info only on rank 0 (master process)
if rank == 0:
    print("="*80)
    print("NOTEBOOK 4: DISTRIBUTED TRAINING PIPELINE")
    print("="*80)
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    
    if use_ddp:
        print(f"✓ MODE: DISTRIBUTED (DDP) - Multi-GPU Training")
        print(f"  NCCL Backend: {dist.is_nccl_available()}")
        print(f"  World Size: {world_size} GPUs")
        print(f"  Backend: NCCL")
        
        # Verify 8 GPUs for production training
        if world_size != 8:
            print(f"  ⚠ WARNING: Expected 8 GPUs for production, got {world_size}")
        else:
            print(f"  ✓ All 8 H200 GPUs detected")
    else:
        print(f"✓ MODE: SINGLE-GPU (Jupyter) - Testing/Development")
        print(f"  Device: {device}")
        print(f"  NOTE: For production training with 8 GPUs, use:")
        print(f"        torchrun --nproc_per_node=8 training_script.py")
    
    # Print GPU info
    if torch.cuda.is_available():
        print(f"\nGPU Information:")
        num_gpus = torch.cuda.device_count()
        for i in range(num_gpus):
            props = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {props.name} - {props.total_memory / 1e9:.1f}GB")
    else:
        print("\n⚠ WARNING: No CUDA GPUs detected! Training will be very slow.")
    
    print("="*80)

# Set random seeds for reproducibility
RANDOM_SEED = 42 + rank  # Different seed per GPU for augmentation diversity
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_SEED)

if rank == 0:
    print(f"\n✓ Random seeds set (base=42, rank offset={rank})")
    print(f"✓ Initialization complete")
    print(f"✓ Mode: {'DDP' if use_ddp else 'Single-GPU'}")
    print(f"✓ Device: {device}")


# %% [markdown]
# ## CELL 2: Load Configuration & Data Splits
# 
# Load saved data from Notebook 1.
# 

# %%
# ============================================================
# CELL 2: LOAD CONFIGURATION & DATA SPLITS
# ============================================================

"""
Load all necessary data from Notebook 1:
- Configuration
- Train/val/test indices
- Class mappings
- Dataset samples
"""

base_dir = Path('./novelty_files')

# ============================================================
# LOAD CONFIGURATION
# ============================================================

if rank == 0:
    print("\n" + "="*80)
    print("LOADING CONFIGURATION & DATA")
    print("="*80)

config_path = base_dir / 'configs' / 'notebook_01_config.json'
with open(config_path, 'r') as f:
    CONFIG = json.load(f)

if rank == 0:
    print(f"✓ Loaded configuration from {config_path}")

# ============================================================
# LOAD CLASS DISTRIBUTION
# ============================================================

dist_path = base_dir / 'splits' / 'class_distribution.json'
with open(dist_path, 'r') as f:
    dist_data = json.load(f)

class_to_idx = dist_data['class_to_idx']
idx_to_class = {int(k): v for k, v in dist_data['idx_to_class'].items()}

if rank == 0:
    print(f"✓ Loaded class mappings ({len(class_to_idx)} classes)")

# ============================================================
# LOAD TRAIN/VAL/TEST SPLITS
# ============================================================

with open(base_dir / 'splits' / 'train_indices.pkl', 'rb') as f:
    train_indices = pickle.load(f)
with open(base_dir / 'splits' / 'val_indices.pkl', 'rb') as f:
    val_indices = pickle.load(f)
with open(base_dir / 'splits' / 'test_indices.pkl', 'rb') as f:
    test_indices = pickle.load(f)

if rank == 0:
    print(f"✓ Loaded splits:")
    print(f"  Train: {len(train_indices):,} samples")
    print(f"  Val:   {len(val_indices):,} samples")
    print(f"  Test:  {len(test_indices):,} samples")

# ============================================================
# RELOAD DATASET SAMPLES
# ============================================================

class HMDB51FightDataset(Dataset):
    def __init__(self, root_dir: str, split: str, class_to_idx: Dict[str, int]):
        self.root_dir = Path(root_dir)
        self.split = split
        self.class_to_idx = class_to_idx
        self.samples = []
        
        split_dir = self.root_dir / split
        for class_name, class_idx in class_to_idx.items():
            class_dir = split_dir / class_name
            if class_dir.exists():
                image_files = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png'))
                for img_path in image_files:
                    self.samples.append({
                        'path': str(img_path),
                        'label': class_idx,
                        'class_name': class_name
                    })
    
    def __len__(self):
        return len(self.samples)

dataset_path = CONFIG['dataset_path']
train_dataset_loader = HMDB51FightDataset(dataset_path, 'train', class_to_idx)
test_dataset_loader = HMDB51FightDataset(dataset_path, 'test', class_to_idx)

all_samples = train_dataset_loader.samples + test_dataset_loader.samples

if rank == 0:
    print(f"✓ Reloaded {len(all_samples):,} samples")
    print("="*80)

# %% [markdown]
# ## CELL 3: MixUp & CutMix Augmentation Functions
# 
# Implement advanced augmentation techniques for improved generalization.
# 

# %%
# ============================================================
# CELL 3: MIXUP & CUTMIX AUGMENTATION
# ============================================================

"""
Implement MixUp and CutMix augmentation for improved generalization.

MixUp: Linear interpolation between two images
  mixed_img = λ * img1 + (1-λ) * img2
  mixed_label = λ * label1 + (1-λ) * label2

CutMix: Cut and paste patches between images
  Cut rectangular patch from img2, paste into img1
  Label mixing based on patch area ratio
"""

def mixup_data(x, y, alpha=1.0):
    """
    Apply MixUp augmentation.
    
    Args:
        x: Input images tensor (batch_size, 3, 224, 224)
        y: Labels tensor (batch_size,)
        alpha: MixUp interpolation strength (default: 1.0)
    
    Returns:
        mixed_x: Mixed images
        y_a, y_b: Original labels
        lam: Mixing coefficient
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam


def cutmix_data(x, y, alpha=1.0):
    """
    Apply CutMix augmentation.
    
    Args:
        x: Input images tensor (batch_size, 3, 224, 224)
        y: Labels tensor (batch_size,)
        alpha: CutMix interpolation strength (default: 1.0)
    
    Returns:
        mixed_x: Mixed images with cutout patches
        y_a, y_b: Original labels
        lam: Mixing coefficient (based on cut area)
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    # Get image dimensions
    _, _, H, W = x.shape
    
    # Calculate cut dimensions
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    
    # Uniform random position for cut
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    
    # Calculate bounding box
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    
    # Apply CutMix
    mixed_x = x.clone()
    mixed_x[:, :, bby1:bby2, bbx1:bbx2] = x[index, :, bby1:bby2, bbx1:bbx2]
    
    # Adjust lambda based on actual cut area
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
    
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """
    Compute mixed loss for MixUp/CutMix.
    
    Args:
        criterion: Loss function (e.g., CrossEntropyLoss)
        pred: Model predictions
        y_a, y_b: Original labels
        lam: Mixing coefficient
    
    Returns:
        Mixed loss
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


if rank == 0:
    print("\n" + "="*80)
    print("AUGMENTATION FUNCTIONS DEFINED")
    print("="*80)
    print("✓ MixUp: Linear interpolation between images")
    print("✓ CutMix: Cut-and-paste patch augmentation")
    print("✓ Mixed criterion: Weighted loss combination")
    print("="*80)


# %% [markdown]
# ## CELL 4: Create Distributed DataLoaders
# 
# Create DataLoaders with DistributedSampler for multi-GPU training.
# 

# %%
# ============================================================
# CELL 4: CREATE DISTRIBUTED DATALOADERS
# ============================================================

"""
Create DataLoaders with DistributedSampler.

Each GPU gets a unique subset of data via DistributedSampler.
Batch size per GPU: 64
Total effective batch: 64 × 8 GPUs × 4 accumulation = 2048
"""

class HMDB51Dataset(Dataset):
    def __init__(self, samples, indices, transform=None):
        self.samples = [samples[i] for i in indices]
        self.transform = transform
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        img = Image.open(sample['path']).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
        
        return img, sample['label']

# Define transforms
vit_mean = [0.485, 0.456, 0.406]
vit_std = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=vit_mean, std=vit_std),
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=vit_mean, std=vit_std),
])

# Create datasets
train_dataset = HMDB51Dataset(all_samples, train_indices, transform=train_transform)
val_dataset = HMDB51Dataset(all_samples, val_indices, transform=val_transform)

# Create DistributedSampler (critical for DDP!)
train_sampler = DistributedSampler(
    train_dataset,
    num_replicas=world_size,
    rank=rank,
    shuffle=True,
    seed=42
)

val_sampler = DistributedSampler(
    val_dataset,
    num_replicas=world_size,
    rank=rank,
    shuffle=False
)

# Create DataLoaders
BATCH_SIZE = 64  # Per GPU
NUM_WORKERS = 4

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    sampler=train_sampler,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    drop_last=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    sampler=val_sampler,
    num_workers=NUM_WORKERS,
    pin_memory=True
)

if rank == 0:
    print("\n" + "="*80)
    print("DISTRIBUTED DATALOADERS CREATED")
    print("="*80)
    print(f"Batch Size per GPU: {BATCH_SIZE}")
    print(f"Number of GPUs: {world_size}")
    print(f"Gradient Accumulation Steps: 4")
    print(f"Effective Batch Size: {BATCH_SIZE * world_size * 4} = {BATCH_SIZE}×{world_size}×4")
    print(f"\nTrain: {len(train_loader)} batches/GPU × {world_size} GPUs")
    print(f"Val:   {len(val_loader)} batches/GPU × {world_size} GPUs")
    print("="*80)

# %% [markdown]
# ## CELL 5: Load ViT Model & Wrap in DDP
# 
# Load pretrained ViT from Notebook 2 and wrap in DistributedDataParallel.
# 

# %%
# ============================================================
# CELL 5: LOAD MODEL & WRAP IN DDP (IF ACTIVE)
# ============================================================

"""
Load ViT-Base model and wrap in DistributedDataParallel (if using DDP).

Options:
1. Load from Notebook 2 baseline checkpoint (if exists)
2. Load fresh ViT-Base from timm

DDP wrapping only happens if launched with torchrun.
"""

# Load ViT-Base model
model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=8)

# Try to load Notebook 2 checkpoint if it exists
vit_baseline_path = base_dir / 'checkpoints' / 'vit_baseline.pt'
if vit_baseline_path.exists():
    if rank == 0:
        print(f"\nLoading ViT baseline from: {vit_baseline_path}")
    checkpoint = torch.load(vit_baseline_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    if rank == 0:
        print(f"✓ Loaded checkpoint (val_acc={checkpoint['val_accuracy']:.2f}%)")
else:
    if rank == 0:
        print("\nNo baseline checkpoint found, using ImageNet pretrained weights")

# Move to device BEFORE wrapping in DDP
model = model.to(device)

# Wrap in DDP ONLY if running in distributed mode
if use_ddp:
    model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=False)
    if rank == 0:
        print("✓ Model wrapped in DistributedDataParallel")
else:
    if rank == 0:
        print("✓ Model running in single-GPU mode")

if rank == 0:
    print("\n" + "="*80)
    print("MODEL SETUP COMPLETE")
    print("="*80)
    # Access model correctly (DDP wraps the model in .module)
    param_model = model.module if use_ddp else model
    total_params = sum(p.numel() for p in param_model.parameters())
    print(f"Model: ViT-Base/16")
    print(f"Parameters: {total_params/1e6:.1f}M")
    print(f"Mode: {'DDP (multi-GPU)' if use_ddp else 'Single-GPU'}")
    print(f"Device: {device}")
    print("="*80)


# %% [markdown]
# ## CELL 6: Main DDP Training Loop (Resume-Safe)
# 
# Full training loop with gradient accumulation, MixUp/CutMix, and checkpointing.  
# **This is the core training cell - runs for ~3 hours on 8 GPUs**
# 

# %%
# ============================================================
# CELL 6: MAIN DDP TRAINING LOOP
# ============================================================

"""
Full distributed training with:
- Gradient accumulation (4 steps)
- MixUp/CutMix (50% probability each)
- Cosine annealing LR schedule
- Gradient clipping
- Checkpoint save/resume
"""

# Training hyperparameters
NUM_EPOCHS = 50
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
GRAD_ACCUM_STEPS = 4
MAX_GRAD_NORM = 1.0
MIXUP_PROB = 0.5
CUTMIX_PROB = 0.5

# Setup optimizer and scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
criterion = nn.CrossEntropyLoss()

# Check for existing checkpoint (resume-safe)
best_checkpoint_path = base_dir / 'checkpoints' / 'ddp_best_model.pt'
start_epoch = 0
best_val_acc = 0.0

if best_checkpoint_path.exists() and rank == 0:
    print(  f"\nFound existing checkpoint: {best_checkpoint_path}")
    print("To resume training, load checkpoint here.")
    # checkpoint = torch.load(best_checkpoint_path)
    # model.module.load_state_dict(checkpoint['model_state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # start_epoch = checkpoint['epoch']
    # best_val_acc = checkpoint['best_val_acc']

if rank == 0:
    print("\n" + "="*80)
    print("STARTING DISTRIBUTED TRAINING")
    print("="*80)
    print(f"Epochs: {NUM_EPOCHS}")
    print(f"Learning Rate: {LEARNING_RATE}")
    print(f"Batch Size (per GPU): {BATCH_SIZE}")
    print(f"Gradient Accumulation: {GRAD_ACCUM_STEPS} steps")
    print(f"Effective Batch Size: {BATCH_SIZE * world_size * GRAD_ACCUM_STEPS}")
    print(f"MixUp Probability: {MIXUP_PROB}")
    print(f"CutMix Probability: {CUTMIX_PROB}")
    print("="*80)

# Training loop
for epoch in range(start_epoch, NUM_EPOCHS):
    # Set epoch for distributed sampler (important!)
    train_sampler.set_epoch(epoch)
    
    # TRAINING PHASE
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    
    if rank == 0:
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS} ")
        print("-" * 60)
    
    optimizer.zero_grad()
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        # Apply augmentation randomly
        use_mixup = np.random.random() < MIXUP_PROB
        use_cutmix = np.random.random() < CUTMIX_PROB and not use_mixup
        
        if use_mixup:
            images, labels_a, labels_b, lam = mixup_data(images, labels, alpha=1.0)
            outputs = model(images)
            loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
        elif use_cutmix:
            images, labels_a, labels_b, lam = cutmix_data(images, labels, alpha=1.0)
            outputs = model(images)
            loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
        
        # Scale loss for gradient accumulation
        loss = loss / GRAD_ACCUM_STEPS
        loss.backward()
        
        # Accumulate gradients
        if (batch_idx + 1) % GRAD_ACCUM_STEPS == 0:
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()
            optimizer.zero_grad()
        
        # Statistics
        train_loss += loss.item() * GRAD_ACCUM_STEPS
        _, predicted = outputs.max(1)
        train_total += labels.size(0)
        train_correct += predicted.eq(labels).sum().item()
        
        # Print progress (rank 0 only, every 100 batches)
        if rank == 0 and (batch_idx + 1) % 100 == 0:
            print(f"  Batch {batch_idx+1}/{len(train_loader)}: Loss={loss.item()*GRAD_ACCUM_STEPS:.4f}")
    
    # Compute training metrics
    avg_train_loss = train_loss / len(train_loader)
    train_acc = 100.0 * train_correct / train_total
    
    # Gather metrics from all GPUs
    train_loss_tensor = torch.tensor([avg_train_loss], device=device)
    train_acc_tensor = torch.tensor([train_acc], device=device)
    dist.all_reduce(train_loss_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(train_acc_tensor, op=dist.ReduceOp.SUM)
    avg_train_loss = (train_loss_tensor / world_size).item()
    train_acc = (train_acc_tensor / world_size).item()
    
    # VALIDATION PHASE (every epoch)
    model.eval()
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            outputs = model(images)
            _, predicted = outputs.max(1)
            val_total += labels.size(0)
            val_correct += predicted.eq(labels).sum().item()
    
    # Gather validation metrics
    val_correct_tensor = torch.tensor([val_correct], device=device)
    val_total_tensor = torch.tensor([val_total], device=device)
    dist.all_reduce(val_correct_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(val_total_tensor, op=dist.ReduceOp.SUM)
    val_acc = 100.0 * val_correct_tensor.item() / val_total_tensor.item()
    
    # Update scheduler
    scheduler.step()
    
    # Print metrics (rank 0 only)
    if rank == 0:
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Acc: {val_acc:.2f}%")
        print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save checkpoint if best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_acc': best_val_acc,
                'config': {
                    'batch_size': BATCH_SIZE,
                    'learning_rate': LEARNING_RATE,
                    'num_epochs': NUM_EPOCHS,
                }
            }
            torch.save(checkpoint, best_checkpoint_path)
            print(f"  ✓ Best model saved! (val_acc={val_acc:.2f}%)")
        
        # Save periodic checkpoint
        if (epoch + 1) % 10 == 0:
            periodic_path = base_dir / 'checkpoints' / f'ddp_epoch_{epoch+1}.pt'
            torch.save(checkpoint, periodic_path)
            print(f"  ✓ Periodic checkpoint saved: {periodic_path.name}")

# Cleanup
if rank == 0:
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"Expected: 80-85% (ViT + Augmentation)")
    print(f"Checkpoint: {best_checkpoint_path}")
    print("="*80)

dist.destroy_process_group()


