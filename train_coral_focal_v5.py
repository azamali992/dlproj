"""
train_coral_focal_v5.py
========================
Changes vs train_coral_focal_correct_cache.py
---------------------------------------------

1. INDEPENDENT PER-THRESHOLD PROJECTIONS (losses_v5 / models_v5)
   The shared weight vector in v2 was a single Linear(1024→1) producing one
   number that all 4 CORAL thresholds relied on. Features that separate G0/G1
   (microaneurysm presence) are not the same features that separate G3/G4
   (neovascularisation). v3 uses 4 independent Linear(1024→1) projections,
   one per boundary, plus an ordinal consistency regularization term to keep
   predictions rank-monotone.

2. ALPHA ON ORIGINAL IMBALANCED LABELS
   v2 called compute_alpha(train_df_balanced) — after oversampling all classes
   to 800, alpha was near-uniform [≈0.79, 0.87, 0.87, 1.0] and did nothing.
   v5 calls compute_alpha(train_df_original) — before balancing, giving real
   weights [≈1.00, 1.07, 1.23, 1.25] that push 25% more gradient toward the
   G2/G3 and G3/G4 thresholds where the model is weakest.

3. UNFREEZE DENSEBLOCK3
   Previous run (claude Code edit) froze denseblock1+2+3, leaving only
   denseblock4 + CORAL head trainable (~35% of params). denseblock3 learns
   lesion morphology features (haemorrhage shapes, exudate boundaries) that
   are fundus-specific and cannot be inherited from ImageNet. Freezing it
   directly caused G3 recall to drop to 0.38. v5 freezes denseblock1 ONLY.

4. All previous fixes retained:
   - Correct cache path (data/processed/hybrid_224)
   - label_smooth=0.0
   - Grade-aware augmentation (mild/moderate/strong per class)
   - Windows-safe pathlib checkpoints
   - CORAL bias diagnostic
   - TTA + top-3 ensemble
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
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
import cv2
import warnings
warnings.filterwarnings('ignore')

from preprocess_grade_aware_aug import (
    preprocess_and_cache,
    create_balanced_train_dataframe,
    get_train_augmentation_mild,
    get_train_augmentation_moderate,
    get_train_augmentation_strong,
    get_val_augmentation,
    verify_augmentation,
)
from utils.losses_v5 import CORALModule_v5
from utils.models_v5 import DenseNet121withCORALFocal_v5, freeze_early_layers


# ============================================================================
# CONFIGURATION
# ============================================================================
CONFIG = {
    'model_name':             'densenet121_coral_v5',
    'num_classes':            5,
    'input_size':             224,
    'batch_size':             16,
    'num_epochs':             30,
    'learning_rate':          3e-4,
    'weight_decay':           1e-3,
    'device':                 'cuda' if torch.cuda.is_available() else 'cpu',
    'data_dir':               './data/raw',
    'output_dir':             os.path.join('results', 'exp_v5'),
    'resume_from_checkpoint': False,
    'preprocessing_method':   'hybrid',
    'gamma':                  2.0,
    'lambda_ord':             0.1,    # ordinal consistency regularization weight
    'pretrained':             True,
    'cache_dir':              os.path.join('data', 'processed', 'hybrid_224'),
    'grad_clip':              1.0,
    'top_k_checkpoints':      3,
    'tta':                    True,
    'target_per_class':       800,
}

os.makedirs(CONFIG['output_dir'], exist_ok=True)
os.makedirs(CONFIG['cache_dir'],  exist_ok=True)


# ============================================================================
# DATASET
# ============================================================================

class PreprocessedDRDataset(Dataset):
    """
    Grade-aware augmentation dispatch:
        Grade 0, 1  → mild     (protect microaneurysms)
        Grade 2     → moderate  (no blur/dropout)
        Grade 3, 4  → strong    (dense lesions survive conservative aug)
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
            return self.mild_aug
        if grade == 2:
            return self.moderate_aug
        return self.strong_aug

    def __getitem__(self, idx):
        row      = self.labels_df.iloc[idx]
        img_path = os.path.join(self.cache_dir, f"{row['id_code']}.png")

        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Cache miss: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        label = int(row['diagnosis'])
        aug   = self._pick_aug(label)

        if aug is not None:
            image = aug(image=image)['image']
        else:
            image = torch.from_numpy(image).float() / 255.0
            image = image.permute(2, 0, 1)
            mean  = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std   = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            image = (image - mean) / std

        return image, label


# ============================================================================
# TTA
# ============================================================================

def _tta_flip(image_batch):
    return [
        image_batch,
        torch.flip(image_batch, dims=[3]),
        torch.flip(image_batch, dims=[2]),
    ]


# ============================================================================
# TRAIN / VALIDATE
# ============================================================================

def train_epoch(model, train_loader, optimizer, scheduler, device, epoch, num_epochs):
    model.train()
    total_loss = 0.0
    correct    = 0
    total      = 0

    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        logits        = model(images)
        coral_targets = model.coral_label_transform(labels)
        loss          = model.loss(logits, coral_targets)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            filter(lambda p: p.requires_grad, model.parameters()),
            max_norm=CONFIG['grad_clip'],
        )
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        predictions = model.predict(logits.detach())
        total      += labels.size(0)
        correct    += (predictions == labels).sum().item()

        if (batch_idx + 1) % 50 == 0:
            lr_now = scheduler.get_last_lr()[0]
            print(f"  Epoch [{epoch+1}/{num_epochs}] "
                  f"Batch [{batch_idx+1}/{len(train_loader)}] "
                  f"Loss: {loss.item():.4f}  LR: {lr_now:.2e}")

    return total_loss / len(train_loader), 100.0 * correct / total


def validate(model, val_loader, device, use_tta=False):
    model.eval()
    total_loss = 0.0
    correct    = 0
    total      = 0
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            if use_tta:
                tta_probs = None
                for aug_images in _tta_flip(images):
                    probs     = torch.sigmoid(model(aug_images))
                    tta_probs = probs if tta_probs is None else tta_probs + probs
                tta_probs   /= 3
                tta_clamped  = tta_probs.clamp(1e-6, 1 - 1e-6)
                logits       = torch.log(tta_clamped / (1 - tta_clamped))
                predictions  = (tta_probs > 0.5).sum(dim=1)
            else:
                logits      = model(images)
                predictions = model.predict(logits)

            coral_targets = model.coral_label_transform(labels)
            loss          = model.loss(logits, coral_targets)
            total_loss   += loss.item()

            total   += labels.size(0)
            correct += (predictions == labels).sum().item()
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(val_loader)
    accuracy = 100.0 * correct / total
    qwk      = cohen_kappa_score(all_labels, all_preds, weights='quadratic')
    return avg_loss, accuracy, qwk, all_preds, all_labels


# ============================================================================
# TOP-K CHECKPOINTS
# ============================================================================

class TopKCheckpoints:
    def __init__(self, k, save_dir):
        self.k        = k
        self.save_dir = Path(save_dir)
        self._heap    = []

    def update(self, qwk, epoch, model):
        path = self.save_dir / f'model_qwk{qwk:.4f}_ep{epoch+1}.pth'
        torch.save(model.state_dict(), path)
        heapq.heappush(self._heap, (qwk, str(path)))

        if len(self._heap) > self.k:
            worst_qwk, worst_path = heapq.heappop(self._heap)
            p = Path(worst_path)
            if p.exists():
                p.unlink()
            print(f"  Removed checkpoint {p.name} (QWK={worst_qwk:.4f})")

        print(f"  Saved checkpoint: {path.name}")

    def best_paths(self):
        return [p for _, p in sorted(self._heap, reverse=True)]


def ensemble_from_checkpoints(model, checkpoint_paths, val_loader, device):
    print(f"\nEnsembling {len(checkpoint_paths)} checkpoints...")
    all_probs  = None
    all_labels = []

    for ckpt_path in checkpoint_paths:
        model.load_state_dict(
            torch.load(ckpt_path, map_location=device, weights_only=True)
        )
        model.eval()
        batch_probs  = []
        batch_labels = []

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                probs  = torch.sigmoid(model(images)).cpu()
                batch_probs.append(probs)
                if not all_labels:
                    batch_labels.extend(labels.numpy())

        epoch_probs = torch.cat(batch_probs, dim=0)
        all_probs   = epoch_probs if all_probs is None else all_probs + epoch_probs
        if not all_labels:
            all_labels = batch_labels

    all_probs      /= len(checkpoint_paths)
    ensemble_preds  = (all_probs > 0.5).sum(dim=1).numpy()
    qwk = cohen_kappa_score(all_labels, ensemble_preds, weights='quadratic')
    print(f"Ensemble QWK ({len(checkpoint_paths)} models): {qwk:.4f}")
    return qwk, ensemble_preds, all_labels


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 80)
    print("TRAINING v5: DenseNet121 + independent CORAL projections")
    print("=" * 80)
    print(f"Device         : {CONFIG['device']}")
    print(f"Input size     : {CONFIG['input_size']}px")
    print(f"Batch size     : {CONFIG['batch_size']}")
    print(f"Cache dir      : {CONFIG['cache_dir']}")
    print(f"lambda_ord     : {CONFIG['lambda_ord']}  (ordinal consistency reg)")
    print(f"Frozen blocks  : denseblock1 ONLY  (denseblock2/3/4 all trainable)")
    print(f"Alpha labels   : ORIGINAL imbalanced distribution")
    print()

    # ── Data split ──────────────────────────────────────────────────────────
    full_df  = pd.read_csv(os.path.join(CONFIG['data_dir'], 'train.csv'))
    train_df, val_df = train_test_split(
        full_df, test_size=0.2, random_state=42, stratify=full_df['diagnosis']
    )
    print(f"Train: {len(train_df)}  Val: {len(val_df)}")
    print(f"Class distribution (train, original):\n"
          f"{train_df['diagnosis'].value_counts().sort_index()}\n")

    # ── Step 1: preprocess & cache ───────────────────────────────────────────
    print("Step 1: Preprocessing and caching...")
    preprocess_and_cache(
        raw_dir=os.path.join(CONFIG['data_dir'], 'train_images'),
        cache_dir=CONFIG['cache_dir'],
        df=pd.concat([train_df, val_df]).reset_index(drop=True),
        method=CONFIG['preprocessing_method'],
        image_size=CONFIG['input_size'],
        verbose=True,
    )

    # ── Step 2: balance (for training only) ──────────────────────────────────
    # NOTE: balance AFTER saving train_df so compute_alpha gets original dist.
    print("\nStep 2: Balancing training dataset...")
    train_df_balanced = create_balanced_train_dataframe(
        train_df=train_df,
        strategy='oversample',
        target_per_class=CONFIG['target_per_class'],
    )
    print(f"Balanced train: {len(train_df_balanced)} samples\n")

    # ── Step 3: verify aug ───────────────────────────────────────────────────
    print("Step 3: Verifying augmentation pipeline...")
    verify_augmentation(
        cache_dir=CONFIG['cache_dir'],
        save_dir=os.path.join(CONFIG['output_dir'], 'aug_samples'),
    )

    # ── Step 4: datasets ─────────────────────────────────────────────────────
    print("\nStep 4: Creating datasets...")
    mild_aug     = get_train_augmentation_mild()
    moderate_aug = get_train_augmentation_moderate()
    strong_aug   = get_train_augmentation_strong()
    val_aug      = get_val_augmentation()

    train_dataset = PreprocessedDRDataset(
        cache_dir=CONFIG['cache_dir'], labels_df=train_df_balanced,
        mild_aug=mild_aug, moderate_aug=moderate_aug,
        strong_aug=strong_aug, val_aug=val_aug, is_train=True,
    )
    val_dataset = PreprocessedDRDataset(
        cache_dir=CONFIG['cache_dir'], labels_df=val_df,
        mild_aug=mild_aug, moderate_aug=moderate_aug,
        strong_aug=strong_aug, val_aug=val_aug, is_train=False,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=CONFIG['batch_size'],
        shuffle=True, num_workers=0, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=CONFIG['batch_size'],
        shuffle=False, num_workers=0, pin_memory=True,
    )

    # ── Step 5: model ────────────────────────────────────────────────────────
    print("\nStep 5: Initialising DenseNet121 with independent CORAL projections...")
    model = DenseNet121withCORALFocal_v5(
        num_classes=CONFIG['num_classes'],
        gamma=CONFIG['gamma'],
        pretrained=CONFIG['pretrained'],
        lambda_ord=CONFIG['lambda_ord'],
    ).to(CONFIG['device'])

    # Freeze ONLY denseblock1 — denseblock2/3 must train for lesion features
    freeze_early_layers(model, blocks=('denseblock1',))

    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters     : {total_params:,}")
    print(f"Trainable parameters : {trainable_params:,} "
          f"({100*trainable_params/total_params:.1f}%)")

    # ── Step 6: alpha on ORIGINAL labels (key fix) ───────────────────────────
    # Pass train_df (before balancing) so alpha reflects true class imbalance.
    # This gives meaningful weights ~[1.0, 1.07, 1.23, 1.25] for thresholds
    # 0-3, pushing more gradient toward the G2/G3 and G3/G4 boundaries.
    print("\nComputing CORAL alpha on ORIGINAL imbalanced labels...")
    model.compute_alpha(
        torch.tensor(train_df['diagnosis'].values, dtype=torch.long)
    )

    # ── Optimizer + OneCycleLR ───────────────────────────────────────────────
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=CONFIG['learning_rate'] / 25,
        weight_decay=CONFIG['weight_decay'],
    )
    total_steps = len(train_loader) * CONFIG['num_epochs']
    scheduler   = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=CONFIG['learning_rate'],
        total_steps=total_steps,
        pct_start=0.1,
        anneal_strategy='cos',
        div_factor=25,
        final_div_factor=1e4,
    )
    print(f"\nOneCycleLR: {total_steps} steps, peak LR={CONFIG['learning_rate']:.1e}")

    # ── Training loop ────────────────────────────────────────────────────────
    top_k   = TopKCheckpoints(k=CONFIG['top_k_checkpoints'], save_dir=CONFIG['output_dir'])
    history = {k: [] for k in ('train_loss', 'train_acc', 'val_loss', 'val_acc', 'val_qwk')}
    best_qwk    = 0.0
    best_epoch  = 0
    best_preds  = None
    best_labels = None

    print("\n" + "=" * 80)
    print("TRAINING STARTED")
    print("=" * 80)

    for epoch in range(CONFIG['num_epochs']):
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, scheduler,
            CONFIG['device'], epoch, CONFIG['num_epochs'],
        )
        val_loss, val_acc, val_qwk, val_preds, val_labels = validate(
            model, val_loader, CONFIG['device'], use_tta=CONFIG['tta'],
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
            best_qwk    = val_qwk
            best_epoch  = epoch
            best_preds  = val_preds
            best_labels = val_labels
            print(f"  ✓ New best QWK: {val_qwk:.4f}")

        top_k.update(val_qwk, epoch, model)

    # ── CORAL diagnostics ────────────────────────────────────────────────────
    print(f"\nCORAL threshold diagnostics:")
    biases = [layer.bias.data.item() for layer in model.model.classifier.fc]
    print(f"  Per-threshold biases : {[round(b, 4) for b in biases]}")
    monotonic = all(biases[k] >= biases[k+1] for k in range(len(biases)-1))
    print(f"  Monotonic            : {'✓ YES' if monotonic else '✗ NO'}")
    print(f"  Alpha weights        : "
          f"{[round(a, 4) for a in model.model.classifier.alpha.tolist()]}")

    # ── Ensemble ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("TRAINING COMPLETED")
    print("=" * 80)
    print(f"\nBest single-model QWK: {best_qwk:.4f} at Epoch {best_epoch+1}")

    best_ckpt_paths = top_k.best_paths()
    ensemble_qwk    = best_qwk
    if len(best_ckpt_paths) > 1:
        ensemble_qwk, ensemble_preds, ensemble_labels = ensemble_from_checkpoints(
            model, best_ckpt_paths, val_loader, CONFIG['device'],
        )
        print(f"Single-model QWK : {best_qwk:.4f}")
        print(f"Ensemble QWK     : {ensemble_qwk:.4f}  ({len(best_ckpt_paths)} models)")
        final_preds  = ensemble_preds
        final_labels = ensemble_labels
    else:
        if best_preds is None:
            _, _, _, best_preds, best_labels = validate(
                model, val_loader, CONFIG['device'], use_tta=False,
            )
        final_preds  = best_preds
        final_labels = best_labels

    # ── Results ──────────────────────────────────────────────────────────────
    cm = confusion_matrix(final_labels, final_preds)
    class_report = classification_report(
        final_labels, final_preds,
        target_names=['Grade 0', 'Grade 1', 'Grade 2', 'Grade 3', 'Grade 4'],
        output_dict=True,
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(history['train_loss'], label='Train', marker='o')
    axes[0].plot(history['val_loss'],   label='Val',   marker='s')
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
    axes[0].set_title('CORAL + Focal Loss (v5)')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(history['train_acc'], label='Train', marker='o')
    axes[1].plot(history['val_acc'],   label='Val',   marker='s')
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Accuracy Curve')
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    axes[2].plot(history['val_qwk'], label='Val QWK', marker='o', color='green')
    axes[2].axhline(y=best_qwk, color='r', linestyle='--',
                    label=f'Best: {best_qwk:.4f}')
    axes[2].set_xlabel('Epoch'); axes[2].set_ylabel('QWK')
    axes[2].set_title('Quadratic Weighted Kappa')
    axes[2].legend(); axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG['output_dir'], 'training_curves.png'), dpi=150)
    print("✓ Saved: training_curves.png")
    plt.close()

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['G0', 'G1', 'G2', 'G3', 'G4'],
                yticklabels=['G0', 'G1', 'G2', 'G3', 'G4'])
    plt.title(f'Confusion Matrix (Best Epoch {best_epoch+1}, QWK={best_qwk:.4f})')
    plt.ylabel('True Label'); plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG['output_dir'], 'confusion_matrix.png'), dpi=150)
    print("✓ Saved: confusion_matrix.png")
    plt.close()

    report_text = classification_report(
        final_labels, final_preds,
        target_names=['Grade 0', 'Grade 1', 'Grade 2', 'Grade 3', 'Grade 4'],
    )
    with open(os.path.join(CONFIG['output_dir'], 'classification_report.txt'), 'w') as f:
        f.write(report_text)
    print("✓ Saved: classification_report.txt")

    summary = {
        'model':             'DenseNet121 + independent CORAL projections (v5)',
        'changes_vs_v4': [
            'independent per-threshold Linear projections (losses_v5/models_v5)',
            'alpha computed on original imbalanced labels (not balanced)',
            'ordinal consistency regularization lambda_ord=0.1',
            'denseblock1 frozen ONLY (denseblock2/3 unfrozen)',
        ],
        'best_epoch':        best_epoch + 1,
        'best_qwk':          float(best_qwk),
        'ensemble_qwk':      float(ensemble_qwk),
        'coral_biases':      [round(b, 4) for b in biases],
        'coral_monotonic':   monotonic,
        'alpha_weights':     [round(a, 4) for a in
                              model.model.classifier.alpha.tolist()],
        'best_val_loss':     float(history['val_loss'][best_epoch]),
        'best_val_acc':      float(history['val_acc'][best_epoch]),
        'total_params':      int(total_params),
        'trainable_params':  int(trainable_params),
        'config':            {k: str(v) if isinstance(v, Path) else v
                              for k, v in CONFIG.items()},
        'per_class_metrics': {
            str(i): {
                'precision': float(
                    class_report.get(f'Grade {i}', {}).get('precision', 0.0)),
                'recall':    float(
                    class_report.get(f'Grade {i}', {}).get('recall', 0.0)),
                'f1-score':  float(
                    class_report.get(f'Grade {i}', {}).get('f1-score', 0.0)),
            } for i in range(5)
        },
    }
    with open(os.path.join(CONFIG['output_dir'], 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print("✓ Saved: summary.json")

    print(f"\n{'='*80}\nALL RESULTS SAVED\n{'='*80}")
    print(f"\n📁 Results : {CONFIG['output_dir']}/")
    print(f"\n🎯 Key Metrics:")
    print(f"  Best single-model QWK : {best_qwk:.4f}  (epoch {best_epoch+1})")
    print(f"  Ensemble QWK          : {ensemble_qwk:.4f}")
    print(f"  CORAL monotonic       : {'✓' if monotonic else '✗'}")
    for i in range(5):
        recall = class_report.get(f'Grade {i}', {}).get('recall', 0.0)
        print(f"  Grade {i} Recall       : {recall:.2%}")


if __name__ == '__main__':
    main()
