#!/usr/bin/env python3
"""
=============================================================================
COMPLETE ABLATION STUDY RUNNER
=============================================================================
This script trains all missing ablation configurations for the HMDB51
violence detection project.

Usage:
    python run_complete_ablations.py [--config CONFIG_NAME] [--gpu GPU_ID]
    
Examples:
    python run_complete_ablations.py                    # Run all missing
    python run_complete_ablations.py --config 04       # Run config 04 only
    python run_complete_ablations.py --gpu 0           # Use GPU 0

Author: Ablation Implementation for Journal Submission
Date: December 2025
=============================================================================
"""

import os
import sys
import json
import pickle
import random
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

import torchvision
from torchvision import transforms
from PIL import Image
import numpy as np

try:
    import timm
except ImportError:
    print("Installing timm...")
    os.system("pip install timm")
    import timm


# =============================================================================
# CONFIGURATION
# =============================================================================

RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# Paths
BASE_DIR = Path('./novelty_files')
CHECKPOINTS_DIR = BASE_DIR / 'checkpoints' / 'ablation'
LOGS_DIR = BASE_DIR / 'logs' / 'ablation'
RESULTS_DIR = BASE_DIR / 'metrics'
DATA_DIR = Path('./fight_dataset/actions (2)/actions')

# Create directories
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# DATASET
# =============================================================================

class HMDB51Dataset(Dataset):
    """HMDB51 Dataset for ablation studies."""
    
    def __init__(self, root_dir: str, indices: List[int], class_to_idx: Dict, 
                 transform=None, return_neighbors: bool = False, 
                 neighbor_indices: Optional[np.ndarray] = None):
        self.root_dir = Path(root_dir)
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.return_neighbors = return_neighbors
        self.neighbor_indices = neighbor_indices
        
        # Build file list
        self.samples = []
        for class_name, class_idx in class_to_idx.items():
            class_dir = self.root_dir / class_name
            if class_dir.exists():
                for img_path in sorted(class_dir.glob('*.jpg')):
                    self.samples.append((str(img_path), class_idx))
        
        # Filter by indices
        if indices is not None:
            self.samples = [self.samples[i] for i in indices if i < len(self.samples)]
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        if self.return_neighbors and self.neighbor_indices is not None:
            neighbors = self.neighbor_indices[idx]
            return image, label, neighbors
        
        return image, label


# =============================================================================
# TRANSFORMS
# =============================================================================

def get_base_transform():
    """Base transform without augmentation."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def get_basic_aug_transform():
    """Basic augmentation: flip, rotation, color jitter."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


# =============================================================================
# AUGMENTATION FUNCTIONS
# =============================================================================

def mixup_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 0.4):
    """MixUp augmentation."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam

def rand_bbox(size, lam):
    """Generate random bounding box for CutMix."""
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    
    return bbx1, bby1, bbx2, bby2

def cutmix_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0):
    """CutMix augmentation."""
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    
    mixed_x = x.clone()
    mixed_x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size(-1) * x.size(-2)))
    
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Compute MixUp/CutMix loss."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# =============================================================================
# NSL LOSS FUNCTIONS
# =============================================================================

def virtual_adversarial_loss(model, x: torch.Tensor, logits: torch.Tensor, 
                              xi: float = 1e-6, eps: float = 2.0, 
                              num_iters: int = 1):
    """Virtual Adversarial Training (VAT) loss."""
    d = torch.randn_like(x, requires_grad=False)
    d = d / (torch.norm(d.view(d.size(0), -1), dim=1, keepdim=True).unsqueeze(-1).unsqueeze(-1) + 1e-8)
    
    for _ in range(num_iters):
        d = d.clone().detach().requires_grad_(True)
        pred_hat = model(x + xi * d)
        
        logp = F.log_softmax(pred_hat, dim=1)
        p = F.softmax(logits.detach(), dim=1)
        kl = F.kl_div(logp, p, reduction='batchmean')
        
        kl.backward()
        d = d.grad.data.clone()
        d = d / (torch.norm(d.view(d.size(0), -1), dim=1, keepdim=True).unsqueeze(-1).unsqueeze(-1) + 1e-8)
        model.zero_grad()
    
    r_adv = eps * d.detach()
    pred_hat = model(x + r_adv)
    
    logp_hat = F.log_softmax(pred_hat, dim=1)
    p = F.softmax(logits.detach(), dim=1)
    vat_loss = F.kl_div(logp_hat, p, reduction='batchmean')
    
    return vat_loss


# =============================================================================
# PGD ADVERSARIAL TRAINING
# =============================================================================

def pgd_attack(model, images: torch.Tensor, labels: torch.Tensor,
               eps: float = 8/255, alpha: float = 2/255, 
               num_steps: int = 7, random_start: bool = True):
    """PGD adversarial attack."""
    images = images.clone().detach()
    adv_images = images.clone().detach()
    
    if random_start:
        adv_images = adv_images + torch.empty_like(adv_images).uniform_(-eps, eps)
        adv_images = torch.clamp(adv_images, 0, 1)
    
    for _ in range(num_steps):
        adv_images.requires_grad = True
        outputs = model(adv_images)
        loss = F.cross_entropy(outputs, labels)
        
        grad = torch.autograd.grad(loss, adv_images, retain_graph=False, create_graph=False)[0]
        
        adv_images = adv_images.detach() + alpha * grad.sign()
        delta = torch.clamp(adv_images - images, -eps, eps)
        adv_images = torch.clamp(images + delta, 0, 1)
    
    return adv_images.detach()


# =============================================================================
# CBAM ATTENTION MODULE
# =============================================================================

class ChannelAttention(nn.Module):
    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        b, c, _, _ = x.size()
        avg_out = self.mlp(self.avg_pool(x).view(b, c))
        max_out = self.mlp(self.max_pool(x).view(b, c))
        attention = self.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        return x * attention

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = x.mean(dim=1, keepdim=True)
        max_out = x.max(dim=1, keepdim=True)[0]
        combined = torch.cat([avg_out, max_out], dim=1)
        attention = self.sigmoid(self.conv(combined))
        return x * attention

class CBAM(nn.Module):
    def __init__(self, in_channels: int, reduction: int = 16, kernel_size: int = 7):
        super().__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)
    
    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


# =============================================================================
# MODEL ARCHITECTURES
# =============================================================================

def create_vit_model(num_classes: int = 8, pretrained: bool = True):
    """Create ViT-Base/16 model."""
    model = timm.create_model('vit_base_patch16_224', pretrained=pretrained)
    model.head = nn.Linear(model.head.in_features, num_classes)
    return model

def create_vgg16_model(num_classes: int = 8, pretrained: bool = True):
    """Create VGG-16 model."""
    model = torchvision.models.vgg16(pretrained=pretrained)
    model.classifier[-1] = nn.Linear(4096, num_classes)
    return model

class ViTWithCBAM(nn.Module):
    """ViT with CBAM attention module."""
    
    def __init__(self, num_classes: int = 8, pretrained: bool = True):
        super().__init__()
        
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=pretrained)
        hidden_dim = self.vit.head.in_features
        self.vit.head = nn.Identity()
        
        self.cbam = CBAM(hidden_dim, reduction=16)
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        features = self.vit.forward_features(x)
        cls_token = features[:, 0]
        
        cls_reshaped = cls_token.unsqueeze(-1).unsqueeze(-1)
        attended = self.cbam(cls_reshaped)
        attended = attended.squeeze(-1).squeeze(-1)
        
        out = self.classifier(attended)
        return out
    
    def get_embeddings(self, x):
        features = self.vit.forward_features(x)
        cls_token = features[:, 0]
        cls_reshaped = cls_token.unsqueeze(-1).unsqueeze(-1)
        attended = self.cbam(cls_reshaped)
        return attended.squeeze(-1).squeeze(-1)


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

def train_one_epoch(model, dataloader, optimizer, criterion, device,
                    use_mixup: bool = False, use_cutmix: bool = False,
                    mixup_alpha: float = 0.4, cutmix_alpha: float = 1.0,
                    use_nsl: bool = False, nsl_weight: float = 0.1,
                    use_pgd: bool = False, pgd_weight: float = 0.5):
    """Train for one epoch with configurable components."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, batch_data in enumerate(dataloader):
        if len(batch_data) == 3:
            images, labels, neighbors = batch_data
        else:
            images, labels = batch_data
            neighbors = None
        
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        apply_mixup = use_mixup and np.random.random() < 0.5
        apply_cutmix = use_cutmix and not apply_mixup and np.random.random() < 0.5
        
        if apply_mixup:
            images, labels_a, labels_b, lam = mixup_data(images, labels, mixup_alpha)
            outputs = model(images)
            loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
        elif apply_cutmix:
            images, labels_a, labels_b, lam = cutmix_data(images, labels, cutmix_alpha)
            outputs = model(images)
            loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
        
        if use_nsl:
            try:
                nsl = virtual_adversarial_loss(model, images, outputs)
                loss = loss + nsl_weight * nsl
            except Exception as e:
                pass  # Skip NSL if error
        
        if use_pgd:
            model.eval()
            adv_images = pgd_attack(model, images, labels)
            model.train()
            adv_outputs = model(adv_images)
            adv_loss = criterion(adv_outputs, labels)
            loss = (1 - pgd_weight) * loss + pgd_weight * adv_loss
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if not apply_mixup and not apply_cutmix:
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        else:
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels_a).sum().item() * lam
            correct += predicted.eq(labels_b).sum().item() * (1 - lam)
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100.0 * correct / total if total > 0 else 0.0
    
    return avg_loss, accuracy

def validate(model, dataloader, criterion, device):
    """Validate model."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100.0 * correct / total
    
    return avg_loss, accuracy


# =============================================================================
# INDIVIDUAL ABLATION TRAINERS
# =============================================================================

def train_ablation(config_name: str, device, train_indices, val_indices, 
                   class_to_idx, neighbor_indices=None):
    """Generic ablation training function."""
    
    checkpoint_path = CHECKPOINTS_DIR / f"{config_name}.pt"
    
    # Check for existing checkpoint
    if checkpoint_path.exists():
        print(f"✓ Found existing checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device)
        return ckpt.get('val_accuracy', None)
    
    # Configuration-specific settings
    configs = {
        '01_VGG16_Baseline': {
            'model_type': 'vgg16', 'epochs': 20, 'batch_size': 32,
            'use_basicaug': False, 'use_mixup': False, 'use_cutmix': False,
            'use_nsl': False, 'use_cbam': False, 'use_pgd': False
        },
        '03_ViT_BasicAug': {
            'model_type': 'vit', 'epochs': 15, 'batch_size': 64,
            'use_basicaug': True, 'use_mixup': False, 'use_cutmix': False,
            'use_nsl': False, 'use_cbam': False, 'use_pgd': False
        },
        '04_ViT_MixUp': {
            'model_type': 'vit', 'epochs': 15, 'batch_size': 64,
            'use_basicaug': False, 'use_mixup': True, 'use_cutmix': False,
            'use_nsl': False, 'use_cbam': False, 'use_pgd': False
        },
        '05_ViT_CutMix': {
            'model_type': 'vit', 'epochs': 15, 'batch_size': 64,
            'use_basicaug': False, 'use_mixup': False, 'use_cutmix': True,
            'use_nsl': False, 'use_cbam': False, 'use_pgd': False
        },
        '06_ViT_MixUp_CutMix': {
            'model_type': 'vit', 'epochs': 15, 'batch_size': 64,
            'use_basicaug': False, 'use_mixup': True, 'use_cutmix': True,
            'use_nsl': False, 'use_cbam': False, 'use_pgd': False
        },
        '07_ViT_NSL': {
            'model_type': 'vit', 'epochs': 15, 'batch_size': 64,
            'use_basicaug': False, 'use_mixup': False, 'use_cutmix': False,
            'use_nsl': True, 'use_cbam': False, 'use_pgd': False
        },
        '08_ViT_Aug_NSL': {
            'model_type': 'vit', 'epochs': 15, 'batch_size': 64,
            'use_basicaug': False, 'use_mixup': True, 'use_cutmix': True,
            'use_nsl': True, 'use_cbam': False, 'use_pgd': False
        },
        '10_ViT_CBAM': {
            'model_type': 'vit_cbam', 'epochs': 15, 'batch_size': 64,
            'use_basicaug': False, 'use_mixup': False, 'use_cutmix': False,
            'use_nsl': False, 'use_cbam': True, 'use_pgd': False
        },
        '11_ViT_CBAM_Aug': {
            'model_type': 'vit_cbam', 'epochs': 15, 'batch_size': 64,
            'use_basicaug': False, 'use_mixup': True, 'use_cutmix': True,
            'use_nsl': False, 'use_cbam': True, 'use_pgd': False
        },
        '12_ViT_PGD': {
            'model_type': 'vit', 'epochs': 10, 'batch_size': 32,
            'use_basicaug': False, 'use_mixup': False, 'use_cutmix': False,
            'use_nsl': False, 'use_cbam': False, 'use_pgd': True
        },
        '14_ViT_CBAM_Aug_NSL_PGD': {
            'model_type': 'vit_cbam', 'epochs': 15, 'batch_size': 32,
            'use_basicaug': False, 'use_mixup': True, 'use_cutmix': True,
            'use_nsl': True, 'use_cbam': True, 'use_pgd': True
        },
    }
    
    if config_name not in configs:
        print(f"Unknown configuration: {config_name}")
        return None
    
    cfg = configs[config_name]
    
    print(f"\n{'='*80}")
    print(f"TRAINING: {config_name}")
    print(f"{'='*80}")
    print(f"Model: {cfg['model_type']}")
    print(f"Epochs: {cfg['epochs']}, Batch Size: {cfg['batch_size']}")
    print(f"MixUp: {cfg['use_mixup']}, CutMix: {cfg['use_cutmix']}")
    print(f"NSL: {cfg['use_nsl']}, CBAM: {cfg['use_cbam']}, PGD: {cfg['use_pgd']}")
    print(f"{'='*80}\n")
    
    # Transforms
    if cfg['use_basicaug']:
        train_transform = get_basic_aug_transform()
    else:
        train_transform = get_base_transform()
    val_transform = get_base_transform()
    
    # Datasets
    train_dataset = HMDB51Dataset(
        DATA_DIR / 'train', train_indices, class_to_idx, 
        train_transform, return_neighbors=cfg['use_nsl'], 
        neighbor_indices=neighbor_indices
    )
    val_dataset = HMDB51Dataset(
        DATA_DIR / 'test', val_indices, class_to_idx, val_transform
    )
    
    train_loader = DataLoader(train_dataset, batch_size=cfg['batch_size'], 
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg['batch_size'], 
                            shuffle=False, num_workers=4, pin_memory=True)
    
    # Model
    if cfg['model_type'] == 'vgg16':
        model = create_vgg16_model(num_classes=8, pretrained=True)
    elif cfg['model_type'] == 'vit_cbam':
        model = ViTWithCBAM(num_classes=8, pretrained=True)
    else:
        model = create_vit_model(num_classes=8, pretrained=True)
    
    model = model.to(device)
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg['epochs'])
    
    best_val_acc = 0.0
    history = {'train_loss': [], 'val_acc': []}
    
    for epoch in range(cfg['epochs']):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            use_mixup=cfg['use_mixup'], use_cutmix=cfg['use_cutmix'],
            use_nsl=cfg['use_nsl'], use_pgd=cfg['use_pgd']
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()
        
        history['train_loss'].append(train_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch+1}/{cfg['epochs']}: "
              f"Train Loss={train_loss:.4f}, Val Acc={val_acc:.2f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': val_acc,
                'history': history,
                'config': cfg
            }, checkpoint_path)
            print(f"  ✓ Saved best model (val_acc={val_acc:.2f}%)")
    
    print(f"\n✓ {config_name} complete. Best Val Acc: {best_val_acc:.2f}%")
    return best_val_acc


# =============================================================================
# MAIN RUNNER
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Run Ablation Studies')
    parser.add_argument('--config', type=str, default=None,
                        help='Specific config to run (e.g., "04" or "04_ViT_MixUp")')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID to use')
    args = parser.parse_args()
    
    # Device
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
        print(f"Using GPU {args.gpu}: {torch.cuda.get_device_name(args.gpu)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    # Load configuration
    config_path = BASE_DIR / 'configs' / 'notebook_01_config.json'
    with open(config_path) as f:
        CONFIG = json.load(f)
    
    dist_path = BASE_DIR / 'splits' / 'class_distribution.json'
    with open(dist_path) as f:
        dist_data = json.load(f)
    
    class_to_idx = dist_data['class_to_idx']
    
    with open(BASE_DIR / 'splits' / 'train_indices.pkl', 'rb') as f:
        train_indices = pickle.load(f)
    with open(BASE_DIR / 'splits' / 'val_indices.pkl', 'rb') as f:
        val_indices = pickle.load(f)
    
    print(f"Train: {len(train_indices):,} samples, Val: {len(val_indices):,} samples")
    
    # Load neighbor indices for NSL
    neighbor_path = BASE_DIR / 'graphs' / 'train_neighbors.npy'
    neighbor_indices = None
    if neighbor_path.exists():
        neighbor_indices = np.load(neighbor_path)
        print(f"Loaded neighbor indices: {neighbor_indices.shape}")
    
    # All configurations to run
    all_configs = [
        '01_VGG16_Baseline',
        '03_ViT_BasicAug',
        '04_ViT_MixUp',
        '05_ViT_CutMix',
        '06_ViT_MixUp_CutMix',
        '07_ViT_NSL',
        '08_ViT_Aug_NSL',
        '10_ViT_CBAM',
        '11_ViT_CBAM_Aug',
        '12_ViT_PGD',
        '14_ViT_CBAM_Aug_NSL_PGD',
    ]
    
    # Filter configs if specified
    if args.config:
        all_configs = [c for c in all_configs if args.config in c]
        if not all_configs:
            print(f"No matching configurations for: {args.config}")
            return
    
    # Run ablations
    results = {}
    results_path = RESULTS_DIR / 'ablation_results_complete.json'
    
    # Load existing results
    if results_path.exists():
        with open(results_path) as f:
            results = json.load(f)
    
    for config_name in all_configs:
        if config_name in results and results[config_name] is not None:
            print(f"\n✓ {config_name}: Already completed ({results[config_name]:.2f}%)")
            continue
        
        try:
            val_acc = train_ablation(
                config_name, device, train_indices, val_indices,
                class_to_idx, neighbor_indices
            )
            
            if val_acc is not None:
                results[config_name] = float(val_acc)
                
                # Save after each completion
                with open(results_path, 'w') as f:
                    json.dump(results, f, indent=2)
        
        except Exception as e:
            print(f"\n✗ {config_name}: Failed with error: {e}")
            import traceback
            traceback.print_exc()
    
    # Final summary
    print("\n" + "="*80)
    print("ABLATION STUDY SUMMARY")
    print("="*80)
    
    for config, acc in sorted(results.items(), key=lambda x: -x[1] if x[1] else 0):
        print(f"{config:40s}: {acc:.2f}%")
    
    print(f"\nResults saved to: {results_path}")


if __name__ == '__main__':
    main()
