"""
train_coral_focal_v2.py
========================
Fixes vs v1
-----------
1.  METRIC "INVERSION" EXPLAINED (not a bug — see comment below).
2.  Augmentation verified at startup (calls preprocess_v2.verify_augmentation).
3.  OneCycleLR replaces CosineAnnealingLR: built-in warmup prevents early
    unstable epochs that caused the QWK zigzag pattern.
4.  Gradient clipping (max_norm=1.0): stabilises CORAL threshold learning.
5.  Only denseblock1 is frozen (v1 froze 1+2 — too aggressive).
6.  Label smoothing via CORALModule_v2 (smooth=0.05).
7.  Top-3 model checkpointing: keeps the 3 best QWK checkpoints and
    ensembles their predictions at the end for a stabler final result.
8.  TTA (Test-Time Augmentation) validation: averages logits over
    horizontal flip, vertical flip, and original → more stable QWK.

WHY TRAIN LOSS > VAL LOSS (and val acc > train acc) — THIS IS EXPECTED:
------------------------------------------------------------------------
    Training batches are drawn by WeightedRandomSampler → 20% per class (hard).
    Validation uses the natural distribution → ~73% Grade 0 (easy).
    A model even weakly biased toward Grade 0 gets high val accuracy for free.
    Augmentation makes each training image harder (higher loss per sample).
    BatchNorm uses noisy batch statistics during training vs. smooth running
    statistics during eval, which also lowers val loss.
    >>> The gap is a sign of regularisation working, NOT classic overfitting.
    Classic overfitting would show val LOSS RISING while train loss falls.
    Both curves are declining together in the v1 plot — that is healthy.
    Focus on QWK, not on this loss gap.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import heapq
from pathlib import Path
from PIL import Image
import cv2
import warnings
warnings.filterwarnings('ignore')

from preprocess_v2 import (
    preprocess_and_cache,
    get_train_augmentation,
    get_val_augmentation,
    verify_augmentation,
)
from utils.losses_v2 import CORALModule_v2
from utils.models_v2 import DenseNet121withCORALFocal_v2, freeze_early_layers

# ============================================================================
# CONFIGURATION
# ============================================================================
CONFIG = {
    'model_name': 'densenet121_coral_focal_v2',
    'num_classes': 5,
    'input_size': 224,
    'batch_size': 16,
    'num_epochs': 30,
    'learning_rate': 3e-4,      # OneCycleLR peak LR (scheduler handles warmup/decay)
    'weight_decay': 1e-3,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'data_dir': './data/raw',
    'output_dir': './results/exp_coral_focal_v2',
    'resume_from_checkpoint': False,
    'preprocessing_method': 'hybrid',
    'gamma': 2.0,
    'label_smooth': 0.05,       # ordinal label smoothing (new in v2)
    'pretrained': True,
    'cache_dir': './data/preprocessed_hybrid',
    'force_repreprocess': False,
    'grad_clip': 1.0,           # gradient clipping max norm (new in v2)
    'top_k_checkpoints': 3,     # save top-K best QWK models for ensemble (new in v2)
    'tta': True,                # test-time augmentation at validation (new in v2)
}

os.makedirs(CONFIG['output_dir'], exist_ok=True)
os.makedirs(CONFIG['cache_dir'], exist_ok=True)


# ============================================================================
# DATASET
# ============================================================================

class PreprocessedDRDataset(Dataset):
    def __init__(self, cache_dir, labels_df, augmentation=None):
        self.cache_dir = cache_dir
        self.labels_df = labels_df.reset_index(drop=True)
        self.augmentation = augmentation

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        row = self.labels_df.iloc[idx]
        img_path = os.path.join(self.cache_dir, f"{row['id_code']}.png")

        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Cache miss: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.augmentation:
            image = self.augmentation(image=image)['image']

        # Safety clip after augmentation (some transforms can shift values slightly)
        image = np.clip(image, 0, 255).astype(np.uint8)

        image = torch.from_numpy(image).float() / 255.0
        image = image.permute(2, 0, 1)

        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image = (image - mean) / std

        return image, int(row['diagnosis'])


# ============================================================================
# TTA HELPERS
# ============================================================================

def _tta_flip(image_batch):
    """Return list of [original, h-flip, v-flip] for a batch tensor."""
    return [
        image_batch,
        torch.flip(image_batch, dims=[3]),   # horizontal flip
        torch.flip(image_batch, dims=[2]),   # vertical flip
    ]


# ============================================================================
# TRAINING
# ============================================================================

def train_epoch(model, train_loader, optimizer, scheduler, device, epoch, num_epochs):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        logits = model(images)
        coral_targets = model.coral_label_transform(labels)
        loss = model.loss(logits, coral_targets)

        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping — prevents large updates from destabilising thresholds
        torch.nn.utils.clip_grad_norm_(
            filter(lambda p: p.requires_grad, model.parameters()),
            max_norm=CONFIG['grad_clip']
        )

        optimizer.step()
        scheduler.step()   # OneCycleLR steps every BATCH, not every epoch

        total_loss += loss.item()
        predictions = model.predict(logits.detach())
        total += labels.size(0)
        correct += (predictions == labels).sum().item()

        if (batch_idx + 1) % 50 == 0:
            lr_now = scheduler.get_last_lr()[0]
            print(f"  Epoch [{epoch+1}/{num_epochs}] Batch [{batch_idx+1}/{len(train_loader)}] "
                  f"Loss: {loss.item():.4f}  LR: {lr_now:.2e}")

    return total_loss / len(train_loader), 100.0 * correct / total


def validate(model, val_loader, device, use_tta=False):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            if use_tta:
                # Average sigmoid probabilities across TTA variants
                tta_probs = None
                for aug_images in _tta_flip(images):
                    logits = model(aug_images)
                    probs = torch.sigmoid(logits)
                    tta_probs = probs if tta_probs is None else tta_probs + probs
                tta_probs /= 3
                # Derive pseudo-logits from averaged probs for loss (informational only)
                logits = torch.log(tta_probs / (1 - tta_probs + 1e-8))
                predictions = (tta_probs > 0.5).sum(dim=1)
            else:
                logits = model(images)
                predictions = model.predict(logits)

            coral_targets = model.coral_label_transform(labels)
            loss = model.loss(logits, coral_targets)
            total_loss += loss.item()

            total += labels.size(0)
            correct += (predictions == labels).sum().item()
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(val_loader)
    accuracy = 100.0 * correct / total
    qwk = cohen_kappa_score(all_labels, all_preds, weights='quadratic')
    return avg_loss, accuracy, qwk, all_preds, all_labels


# ============================================================================
# TOP-K CHECKPOINT MANAGER
# ============================================================================

class TopKCheckpoints:
    """Keeps the top-K model checkpoints by QWK score."""

    def __init__(self, k, save_dir):
        self.k = k
        self.save_dir = save_dir
        self._heap = []   # min-heap of (qwk, path)

    def update(self, qwk, epoch, model):
        path = os.path.join(self.save_dir, f'model_qwk{qwk:.4f}_ep{epoch+1}.pth')
        torch.save(model.state_dict(), path)

        heapq.heappush(self._heap, (qwk, path))

        if len(self._heap) > self.k:
            worst_qwk, worst_path = heapq.heappop(self._heap)
            if os.path.exists(worst_path):
                os.remove(worst_path)
            print(f"  Removed checkpoint {os.path.basename(worst_path)} "
                  f"(QWK={worst_qwk:.4f})")

        print(f"  Saved checkpoint: {os.path.basename(path)}")

    def best_paths(self):
        """Return paths sorted best-first."""
        return [p for _, p in sorted(self._heap, reverse=True)]


def ensemble_from_checkpoints(model, checkpoint_paths, val_loader, device):
    """Average CORAL probabilities from top-K saved checkpoints."""
    print(f"\nEnsembling {len(checkpoint_paths)} checkpoints...")
    all_probs = None
    all_labels = []

    for ckpt_path in checkpoint_paths:
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()
        batch_probs = []
        batch_labels = []

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                logits = model(images)
                probs = torch.sigmoid(logits).cpu()
                batch_probs.append(probs)
                if not all_labels:
                    batch_labels.extend(labels.numpy())

        epoch_probs = torch.cat(batch_probs, dim=0)
        all_probs = epoch_probs if all_probs is None else all_probs + epoch_probs
        if not all_labels:
            all_labels = batch_labels

    all_probs /= len(checkpoint_paths)
    ensemble_preds = (all_probs > 0.5).sum(dim=1).numpy()
    qwk = cohen_kappa_score(all_labels, ensemble_preds, weights='quadratic')
    print(f"Ensemble QWK ({len(checkpoint_paths)} models): {qwk:.4f}")
    return qwk, ensemble_preds, all_labels


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 80)
    print("TRAINING v2: DenseNet121 + CORAL + Focal Loss  (fixed pipeline)")
    print("=" * 80)
    print(f"Device: {CONFIG['device']}")
    print(f"Label smoothing: {CONFIG['label_smooth']}")
    print(f"Grad clip: {CONFIG['grad_clip']}")
    print(f"TTA at validation: {CONFIG['tta']}")
    print()

    # ----------------------------------------------------------------
    # Data split
    # ----------------------------------------------------------------
    train_df = pd.read_csv(os.path.join(CONFIG['data_dir'], 'train.csv'))
    train_df, val_df = train_test_split(
        train_df, test_size=0.2, random_state=42, stratify=train_df['diagnosis']
    )
    print(f"Train: {len(train_df)}  Val: {len(val_df)}")
    print(f"Class distribution (train):\n{train_df['diagnosis'].value_counts().sort_index()}\n")

    # ----------------------------------------------------------------
    # Preprocess + cache
    # ----------------------------------------------------------------
    preprocess_and_cache(
        raw_dir=os.path.join(CONFIG['data_dir'], 'train_images'),
        cache_dir=CONFIG['cache_dir'],
        df=pd.concat([train_df, val_df]).reset_index(drop=True),
        method=CONFIG['preprocessing_method'],
        verbose=True,
    )

    # ----------------------------------------------------------------
    # AUGMENTATION VERIFICATION — confirms transforms are actually running
    # ----------------------------------------------------------------
    print("\nVerifying augmentation pipeline...")
    verify_augmentation(
        cache_dir=CONFIG['cache_dir'],
        save_dir=os.path.join(CONFIG['output_dir'], 'aug_samples'),
    )

    # ----------------------------------------------------------------
    # WeightedRandomSampler
    # ----------------------------------------------------------------
    class_counts = np.bincount(train_df['diagnosis'].values, minlength=CONFIG['num_classes'])
    print(f"\nClass counts (train): {dict(enumerate(class_counts))}")
    sample_weights = [1.0 / class_counts[label] for label in train_df['diagnosis'].values]
    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights), replacement=True
    )

    # ----------------------------------------------------------------
    # Datasets
    # ----------------------------------------------------------------
    train_dataset = PreprocessedDRDataset(
        CONFIG['cache_dir'], train_df, augmentation=get_train_augmentation()
    )
    val_dataset = PreprocessedDRDataset(
        CONFIG['cache_dir'], val_df, augmentation=get_val_augmentation()
    )

    train_loader = DataLoader(
        train_dataset, batch_size=CONFIG['batch_size'], sampler=sampler,
        num_workers=0, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=CONFIG['batch_size'], shuffle=False,
        num_workers=0, pin_memory=True,
    )

    # ----------------------------------------------------------------
    # Model
    # ----------------------------------------------------------------
    print("\nInitialising DenseNet121 with CORAL + Focal head (v2)...")
    model = DenseNet121withCORALFocal_v2(
        num_classes=CONFIG['num_classes'],
        gamma=CONFIG['gamma'],
        pretrained=CONFIG['pretrained'],
        label_smooth=CONFIG['label_smooth'],
    ).to(CONFIG['device'])

    # Freeze only denseblock1 (v1 froze 1+2 — too aggressive)
    freeze_early_layers(model, blocks=('denseblock1',))

    # CORAL alpha weights
    print("\nComputing CORAL alpha weights...")
    model.compute_alpha(torch.tensor(train_df['diagnosis'].values, dtype=torch.long))
    print("Alpha weights computed ✓")

    # ----------------------------------------------------------------
    # Optimizer + OneCycleLR
    # OneCycleLR provides:
    #   - Linear warmup for the first ~30% of steps (prevents early spikes)
    #   - Cosine decay for the remaining steps
    #   - Avoids the need for manual LR tuning
    # ----------------------------------------------------------------
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=CONFIG['learning_rate'] / 25,   # OneCycleLR starts here and ramps up
        weight_decay=CONFIG['weight_decay'],
    )

    total_steps = len(train_loader) * CONFIG['num_epochs']
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=CONFIG['learning_rate'],
        total_steps=total_steps,
        pct_start=0.1,          # 10% warmup
        anneal_strategy='cos',
        div_factor=25,          # start LR = max_lr / 25
        final_div_factor=1e4,   # end LR = start_lr / 1e4
    )
    print(f"\nOneCycleLR: {total_steps} total steps, peak LR={CONFIG['learning_rate']:.1e}")

    # ----------------------------------------------------------------
    # Training loop
    # ----------------------------------------------------------------
    top_k = TopKCheckpoints(k=CONFIG['top_k_checkpoints'], save_dir=CONFIG['output_dir'])

    history = {k: [] for k in ('train_loss', 'train_acc', 'val_loss', 'val_acc', 'val_qwk')}
    best_qwk = 0.0
    best_epoch = 0
    best_preds = None
    best_labels = None

    print("\n" + "=" * 80)
    print("TRAINING STARTED")
    print("=" * 80)

    for epoch in range(CONFIG['num_epochs']):
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, scheduler,
            CONFIG['device'], epoch, CONFIG['num_epochs']
        )

        val_loss, val_acc, val_qwk, val_preds, val_labels = validate(
            model, val_loader, CONFIG['device'], use_tta=CONFIG['tta']
        )

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_qwk'].append(val_qwk)

        print(f"\nEpoch {epoch+1}/{CONFIG['num_epochs']} Summary:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}% | "
              f"Val QWK: {val_qwk:.4f}")

        # Track best + top-K
        top_k.update(val_qwk, epoch, model)
        if val_qwk > best_qwk:
            best_qwk = val_qwk
            best_epoch = epoch
            best_preds = val_preds
            best_labels = val_labels
            torch.save(model.state_dict(),
                       os.path.join(CONFIG['output_dir'], 'best_model.pth'))
            print(f"  ✓ Best model saved (QWK={val_qwk:.4f})")

    # ----------------------------------------------------------------
    # Ensemble predictions from top-K checkpoints
    # ----------------------------------------------------------------
    ens_qwk, ens_preds, ens_labels = ensemble_from_checkpoints(
        model, top_k.best_paths(), val_loader, CONFIG['device']
    )

    # Use ensemble if it beats best single-model
    if ens_qwk > best_qwk:
        print(f"\nEnsemble QWK ({ens_qwk:.4f}) > best single ({best_qwk:.4f}) — using ensemble")
        best_qwk = ens_qwk
        best_preds = ens_preds
        best_labels = ens_labels
    else:
        print(f"\nBest single model ({best_qwk:.4f}) >= ensemble ({ens_qwk:.4f}) — using single")

    # ----------------------------------------------------------------
    # Results
    # ----------------------------------------------------------------
    print(f"\n{'='*80}")
    print(f"TRAINING COMPLETE  |  Best QWK: {best_qwk:.4f}  |  Epoch: {best_epoch+1}")
    print(f"{'='*80}")

    cm = confusion_matrix(best_labels, best_preds)
    class_report = classification_report(
        best_labels, best_preds,
        target_names=['Grade 0', 'Grade 1', 'Grade 2', 'Grade 3', 'Grade 4'],
        output_dict=True,
    )

    # Training curves
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(history['train_loss'], label='Train', marker='o')
    axes[0].plot(history['val_loss'],   label='Val',   marker='s')
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
    axes[0].set_title('CORAL + Focal Loss (v2)')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(history['train_acc'], label='Train', marker='o')
    axes[1].plot(history['val_acc'],   label='Val',   marker='s')
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Accuracy Curve\n(Val > Train is EXPECTED — see header comment)')
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    axes[2].plot(history['val_qwk'], label='Val QWK', marker='o', color='green')
    axes[2].axhline(y=best_qwk, color='r', linestyle='--', label=f'Best: {best_qwk:.4f}')
    axes[2].set_xlabel('Epoch'); axes[2].set_ylabel('QWK')
    axes[2].set_title('Quadratic Weighted Kappa')
    axes[2].legend(); axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG['output_dir'], 'training_curves.png'), dpi=150)
    print(f"✓ Saved: training_curves.png")
    plt.close()

    # Confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['G0','G1','G2','G3','G4'],
                yticklabels=['G0','G1','G2','G3','G4'])
    plt.title(f'Confusion Matrix (QWK={best_qwk:.4f})')
    plt.ylabel('True Label'); plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG['output_dir'], 'confusion_matrix.png'), dpi=150)
    print(f"✓ Saved: confusion_matrix.png")
    plt.close()

    report_text = classification_report(
        best_labels, best_preds,
        target_names=['Grade 0', 'Grade 1', 'Grade 2', 'Grade 3', 'Grade 4'],
    )
    with open(os.path.join(CONFIG['output_dir'], 'classification_report.txt'), 'w') as f:
        f.write(report_text)
    print(f"✓ Saved: classification_report.txt")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    summary = {
        'model': 'DenseNet121 with CORAL + Focal Loss (v2)',
        'best_qwk': float(best_qwk),
        'ensemble_qwk': float(ens_qwk),
        'best_epoch': best_epoch + 1,
        'label_smooth': CONFIG['label_smooth'],
        'grad_clip': CONFIG['grad_clip'],
        'tta': CONFIG['tta'],
        'total_params': int(total_params),
        'trainable_params': int(trainable_params),
        'config': CONFIG,
        'per_class_metrics': {
            str(i): {k: float(v)
                     for k, v in class_report.get(f'Grade {i}', {}).items()
                     if k != 'support'}
            for i in range(5)
        },
    }
    with open(os.path.join(CONFIG['output_dir'], 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Saved: summary.json")

    print(f"\n{'='*80}")
    print("KEY METRICS")
    print(f"{'='*80}")
    print(f"  Best single-model QWK : {best_qwk:.4f}  (epoch {best_epoch+1})")
    print(f"  Ensemble QWK          : {ens_qwk:.4f}")
    for i in range(5):
        metrics = class_report.get(f'Grade {i}', {})
        print(f"  Grade {i} Recall: {metrics.get('recall', 0):.2%}")


if __name__ == '__main__':
    main()
