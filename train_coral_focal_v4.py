"""
train_coral_focal_v3.py
========================
Improvements vs v2
------------------
1. Uses preprocess_v3.py — research-backed pipeline with Ben Graham + LAB-CLAHE.
2. Augmentation now includes Normalize + ToTensorV2 → outputs ready-to-use tensors.
3. Uses create_balanced_train_dataframe() from preprocess_v3 for class balancing.
4. get_train_augmentation_strong() for minority classes, mild() available if needed.
5. All v2 improvements retained:
   - OneCycleLR with built-in warmup
   - Gradient clipping (max_norm=1.0)
   - Label smoothing via CORALModule_v2
   - Top-3 model checkpointing + ensemble
   - TTA (Test-Time Augmentation) at validation

KEY PIPELINE CHANGE:
    preprocess_v3 augmentations (get_train_augmentation_strong, get_val_augmentation)
    now include:
        - Geometric: HFlip, VFlip, Rotate, Affine, CircleMask
        - Normalize(ImageNet stats)
        - ToTensorV2()  ← returns torch.Tensor, channels-first, normalized
    So the dataset receives tensors directly — no manual normalization needed.
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

from preprocess_v3_1 import (
    preprocess_and_cache,
    create_balanced_train_dataframe,
    get_train_augmentation_mild,
    get_train_augmentation_moderate,
    get_train_augmentation_strong,
    get_val_augmentation,
    verify_augmentation,
)
from utils.losses_v2 import CORALModule_v2
from utils.models_v2 import DenseNet121withCORALFocal_v2, freeze_early_layers

# ============================================================================
# CONFIGURATION
# ============================================================================
CONFIG = {
    'model_name': 'densenet121_coral_focal_v3',
    'num_classes': 5,
    'input_size': 512,          # preprocess_v3 default (better lesion preservation)
    'batch_size': 8,
    'num_epochs': 30,
    'learning_rate': 3e-4,      # OneCycleLR peak LR (scheduler handles warmup/decay)
    'weight_decay': 1e-3,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'data_dir': './data/raw',
    'output_dir': './results/exp_coral_focal_v3',
    'resume_from_checkpoint': False,
    'preprocessing_method': 'hybrid',  # Ben Graham + LAB-CLAHE (recommended)
    'gamma': 2.0,
    'label_smooth': 0.05,       # ordinal label smoothing
    'pretrained': True,
    'cache_dir': './data/processed/hybrid_512',
    'force_repreprocess': False,
    'grad_clip': 1.0,
    'top_k_checkpoints': 3,     # save top-K best QWK models for ensemble
    'tta': True,                # test-time augmentation at validation
    'target_per_class': 800,    # balanced training target (for create_balanced_train_dataframe)
}

os.makedirs(CONFIG['output_dir'], exist_ok=True)
os.makedirs(CONFIG['cache_dir'], exist_ok=True)


# ============================================================================
# DATASET
# ============================================================================

class PreprocessedDRDataset(Dataset):
    """
    Loads preprocessed images from cache and applies grade-aware augmentation.

    Grade-aware dispatch (the key fix from the augmentation analysis):
        Grade 0           → mild aug  (no lesions to protect, but keep it light)
        Grade 1           → mild aug  (microaneurysms only — 3-6px dots that
                                       blur/dropout/sat-shift reliably erase)
        Grade 2           → moderate  (multiple lesion types, can take more jitter
                                       but blur still softens haemorrhage edges)
        Grade 3, Grade 4  → strong    (dense lesion fields survive conservative
                                       blur and small dropout safely)

    All augmentation pipelines from preprocess_v3 end with
    Normalize(ImageNet) + ToTensorV2(), so __getitem__ returns a
    normalized float32 torch.Tensor (C, H, W) — no manual norm needed.
    """
    def __init__(self, cache_dir, labels_df,
                 mild_aug=None, moderate_aug=None, strong_aug=None,
                 val_aug=None, is_train=True):
        self.cache_dir    = cache_dir
        self.labels_df    = labels_df.reset_index(drop=True)
        self.mild_aug     = mild_aug
        self.moderate_aug = moderate_aug
        self.strong_aug   = strong_aug
        self.val_aug      = val_aug
        self.is_train     = is_train

    def __len__(self):
        return len(self.labels_df)

    def _pick_aug(self, grade: int):
        if not self.is_train:
            return self.val_aug
        if grade <= 1:
            return self.mild_aug      # protect microaneurysms
        if grade == 2:
            return self.moderate_aug  # moderate jitter, no blur/dropout
        return self.strong_aug        # grades 3-4 can tolerate more

    def __getitem__(self, idx):
        row = self.labels_df.iloc[idx]
        img_path = os.path.join(self.cache_dir, f"{row['id_code']}.png")

        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Cache miss: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        label = int(row['diagnosis'])
        aug = self._pick_aug(label)

        # All preprocess_v3 augs end with Normalize + ToTensorV2.
        # Output is torch.Tensor (C, H, W), already normalized — do NOT
        # apply manual normalization again.
        if aug is not None:
            image = aug(image=image)['image']
        else:
            # Fallback: manual conversion (should not be reached in practice)
            image = torch.from_numpy(image).float() / 255.0
            image = image.permute(2, 0, 1)
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            image = (image - mean) / std

        return image, label


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
    print("TRAINING v3: DenseNet121 + CORAL + Focal Loss  (preprocess_v3 pipeline)")
    print("=" * 80)
    print(f"Device: {CONFIG['device']}")
    print(f"Input size: {CONFIG['input_size']}")
    print(f"Preprocessing: {CONFIG['preprocessing_method']}")
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
    # Preprocess + cache (using preprocess_v3)
    # ----------------------------------------------------------------
    print("Step 1: Preprocessing and caching images (preprocess_v3)...")
    preprocess_and_cache(
        raw_dir=os.path.join(CONFIG['data_dir'], 'train_images'),
        cache_dir=CONFIG['cache_dir'],
        df=pd.concat([train_df, val_df]).reset_index(drop=True),
        method=CONFIG['preprocessing_method'],
        image_size=CONFIG['input_size'],
        verbose=True,
    )

    # ----------------------------------------------------------------
    # Balance training set using preprocess_v3
    # ----------------------------------------------------------------
    print("\nStep 2: Balancing training dataset...")
    train_df_balanced = create_balanced_train_dataframe(
        train_df=train_df,
        strategy='oversample',
        target_per_class=CONFIG['target_per_class'],
    )
    print(f"Balanced train: {len(train_df_balanced)} samples")
    print(f"Class distribution (balanced train):\n{train_df_balanced['diagnosis'].value_counts().sort_index()}\n")

    # ----------------------------------------------------------------
    # AUGMENTATION VERIFICATION — confirms transforms are actually running
    # ----------------------------------------------------------------
    print("Step 3: Verifying augmentation pipeline...")
    verify_augmentation(
        cache_dir=CONFIG['cache_dir'],
        save_dir=os.path.join(CONFIG['output_dir'], 'aug_samples'),
    )

    # ----------------------------------------------------------------
    # Datasets (use balanced train, natural val)
    # ----------------------------------------------------------------
    print(f"\nStep 4: Creating datasets...")
    # Grade-aware augmentation pipelines (built once, reused per sample).
    # See PreprocessedDRDataset._pick_aug() for the dispatch logic.
    mild_aug     = get_train_augmentation_mild()
    moderate_aug = get_train_augmentation_moderate()
    strong_aug   = get_train_augmentation_strong()
    val_aug      = get_val_augmentation()

    train_dataset = PreprocessedDRDataset(
        cache_dir=CONFIG['cache_dir'],
        labels_df=train_df_balanced,
        mild_aug=mild_aug,
        moderate_aug=moderate_aug,
        strong_aug=strong_aug,
        val_aug=val_aug,
        is_train=True,
    )
    val_dataset = PreprocessedDRDataset(
        cache_dir=CONFIG['cache_dir'],
        labels_df=val_df,
        mild_aug=mild_aug,
        moderate_aug=moderate_aug,
        strong_aug=strong_aug,
        val_aug=val_aug,
        is_train=False,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=CONFIG['batch_size'], shuffle=True,
        num_workers=0, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=CONFIG['batch_size'], shuffle=False,
        num_workers=0, pin_memory=True,
    )

    # ----------------------------------------------------------------
    # Model
    # ----------------------------------------------------------------
    print("\nStep 5: Initialising DenseNet121 with CORAL + Focal head...")
    model = DenseNet121withCORALFocal_v2(
        num_classes=CONFIG['num_classes'],
        gamma=CONFIG['gamma'],
        pretrained=CONFIG['pretrained'],
        label_smooth=CONFIG['label_smooth'],
    ).to(CONFIG['device'])

    # Freeze only denseblock1
    freeze_early_layers(model, blocks=('denseblock1',))

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.1f}%)")

    # CORAL alpha weights
    print("\nComputing CORAL alpha weights...")
    model.compute_alpha(torch.tensor(train_df_balanced['diagnosis'].values, dtype=torch.long))
    print("Alpha weights computed ✓")

    # ----------------------------------------------------------------
    # Optimizer + OneCycleLR
    # ----------------------------------------------------------------
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=CONFIG['learning_rate'] / 25,   # init LR (will be ramped up by scheduler)
        weight_decay=CONFIG['weight_decay'],
    )

    total_steps = len(train_loader) * CONFIG['num_epochs']
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=CONFIG['learning_rate'],
        total_steps=total_steps,
        pct_start=0.1,          # 10% of steps for warmup
        anneal_strategy='cos',
        div_factor=25,          # initial LR = max_lr / 25
        final_div_factor=1e4,   # final LR = max_lr / (25 * 1e4)
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
            model, train_loader, optimizer, scheduler, CONFIG['device'], epoch, CONFIG['num_epochs']
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
        print(f"  Train: Loss={train_loss:.4f}, Acc={train_acc:.2f}%")
        print(f"  Val:   Loss={val_loss:.4f}, Acc={val_acc:.2f}%, QWK={val_qwk:.4f}")

        if val_qwk > best_qwk:
            best_qwk = val_qwk
            best_epoch = epoch
            best_preds = val_preds
            best_labels = val_labels
            print(f"  ✓ Best QWK so far: {val_qwk:.4f}")

        # Save top-K
        top_k.update(val_qwk, epoch, model)

    # ================================================================
    # Post-training: ensemble predictions from top-K checkpoints
    # ================================================================
    print("\n" + "=" * 80)
    print("TRAINING COMPLETED")
    print("=" * 80)
    print(f"\nBest single-model QWK: {best_qwk:.4f} at Epoch {best_epoch+1}")

    best_ckpt_paths = top_k.best_paths()
    if len(best_ckpt_paths) > 1:
        ensemble_qwk, ensemble_preds, ensemble_labels = ensemble_from_checkpoints(
            model, best_ckpt_paths, val_loader, CONFIG['device']
        )
        print(f"Single-model QWK:  {best_qwk:.4f}")
        print(f"Ensemble QWK:      {ensemble_qwk:.4f}  ({len(best_ckpt_paths)} models)")
        final_preds = ensemble_preds
        final_labels = ensemble_labels
    else:
        final_preds = best_preds
        final_labels = best_labels

    # ================================================================
    # Results
    # ================================================================
    cm = confusion_matrix(final_labels, final_preds)
    class_report = classification_report(
        final_labels, final_preds,
        target_names=['Grade 0', 'Grade 1', 'Grade 2', 'Grade 3', 'Grade 4'],
        output_dict=True
    )

    # Training curves
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(history['train_loss'], label='Train', marker='o')
    axes[0].plot(history['val_loss'], label='Val', marker='s')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('CORAL + Focal Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history['train_acc'], label='Train', marker='o')
    axes[1].plot(history['val_acc'], label='Val', marker='s')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Accuracy Curve')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(history['val_qwk'], label='Val QWK', marker='o', color='green')
    axes[2].axhline(y=best_qwk, color='r', linestyle='--', label=f'Best: {best_qwk:.4f}')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('QWK')
    axes[2].set_title('Quadratic Weighted Kappa')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG['output_dir'], 'training_curves.png'), dpi=150)
    print(f"✓ Saved: training_curves.png")
    plt.close()

    # Confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['G0', 'G1', 'G2', 'G3', 'G4'],
                yticklabels=['G0', 'G1', 'G2', 'G3', 'G4'])
    plt.title(f'Confusion Matrix (Best Epoch {best_epoch+1}, QWK={best_qwk:.4f})')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG['output_dir'], 'confusion_matrix.png'), dpi=150)
    print(f"✓ Saved: confusion_matrix.png")
    plt.close()

    # Classification report
    report_text = classification_report(
        final_labels, final_preds,
        target_names=['Grade 0', 'Grade 1', 'Grade 2', 'Grade 3', 'Grade 4']
    )
    with open(os.path.join(CONFIG['output_dir'], 'classification_report.txt'), 'w') as f:
        f.write(report_text)
    print(f"✓ Saved: classification_report.txt")

    # Summary JSON
    summary = {
        'model': 'DenseNet121 with CORAL + Focal Loss (v3 preprocess_v3)',
        'best_epoch': best_epoch + 1,
        'best_qwk': float(best_qwk),
        'ensemble_qwk': float(ensemble_qwk) if len(best_ckpt_paths) > 1 else float(best_qwk),
        'best_val_loss': float(history['val_loss'][best_epoch]),
        'best_val_acc': float(history['val_acc'][best_epoch]),
        'total_params': int(total_params),
        'trainable_params': int(trainable_params),
        'preprocessing': CONFIG['preprocessing_method'],
        'input_size': CONFIG['input_size'],
        'config': CONFIG,
        'per_class_metrics': {
            str(i): {
                'precision': float(class_report.get(str(i), {}).get('precision', 0.0)),
                'recall': float(class_report.get(str(i), {}).get('recall', 0.0)),
                'f1-score': float(class_report.get(str(i), {}).get('f1-score', 0.0)),
            }
            for i in range(5)
        }
    }
    with open(os.path.join(CONFIG['output_dir'], 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Saved: summary.json")

    print(f"\n{'='*80}")
    print("ALL RESULTS SAVED")
    print(f"{'='*80}")
    print(f"\n📁 Results folder: {CONFIG['output_dir']}/")
    print(f"\n📊 Key Metrics:")
    print(f"  Best QWK: {best_qwk:.4f}")
    if len(best_ckpt_paths) > 1:
        print(f"  Ensemble QWK: {ensemble_qwk:.4f}")
    print(f"  Best Epoch: {best_epoch + 1}")
    for i in range(5):
        metrics = class_report.get(str(i), {})
        recall = metrics.get('recall', 0.0)
        print(f"  Grade {i} Recall: {recall:.2%}")


if __name__ == '__main__':
    main()
