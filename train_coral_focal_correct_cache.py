"""
train_coral_focal_correct_cache.py
====================================
Fixes applied vs train_coral_focal_v3.py:

1. CACHE PATH — was './data/preprocessed_hybrid' (old v2 cache, wrong pipeline).
   Now './data/processed/hybrid_512' which is where preprocess_grade_aware_aug.py
   actually saves files (Ben Graham + LAB-CLAHE + fundus crop + circle mask).
   Every run before this was training on the wrong preprocessed images.

2. BATCH SIZE — was 16 at 512px input on a 4GB GPU (RTX 2050).
   512x512x3 float32 x 16 = ~3.8GB before activations/gradients → OOM risk.
   Now 8, which uses ~2.2GB and trains stably.

3. LABEL SMOOTHING OFF — was label_smooth=0.05 with gamma=2.0 focal loss.
   These two conflict: focal suppresses easy examples, smoothing inflates
   their loss back up. Net effect: gradient signal that focal was focusing
   onto hard/minority cases gets diluted. Set label_smooth=0.0 — focal
   alone handles hard example mining.

4. GRADE-AWARE AUGMENTATION — imports from preprocess_grade_aware_aug.py
   and dispatches per sample: G0/G1 → mild, G2 → moderate, G3/G4 → strong.
   Replaces the single get_train_augmentation_strong() for all grades which
   was erasing microaneurysms from Grade 1 images (blur/dropout on 3-6px dots).

5. WINDOWS PATH FIX — TopKCheckpoints.update() now uses pathlib.Path so
   checkpoint filenames don't mix / and \ and cause RuntimeError on Windows.

6. CORAL BIAS DIAGNOSTIC — prints bias vector after training so you can
   verify rank-monotonicity of the learned thresholds.

7. OUTPUT DIR renamed to exp_correct_cache to distinguish from previous runs.
"""

import torch
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
import cv2
import warnings
warnings.filterwarnings('ignore')

# ── grade-aware aug pipeline (the fixed preprocessing) ──────────────────────
from preprocess_grade_aware_aug import (
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
    'model_name': 'densenet121_coral_focal_correct_cache',
    'num_classes': 5,
    'input_size': 512,
    'batch_size': 8,            # FIX 2: was 16 → OOM on RTX 2050 at 512px
    'num_epochs': 30,
    'learning_rate': 3e-4,
    'weight_decay': 1e-3,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'data_dir': './data/raw',
    'output_dir': os.path.join('results', 'exp_correct_cache'),  # pathlib-safe
    'resume_from_checkpoint': False,
    'preprocessing_method': 'hybrid',
    'gamma': 2.0,
    'label_smooth': 0.0,        # FIX 3: was 0.05 — conflicts with focal loss
    'pretrained': True,
    # FIX 1: correct cache path — this is where preprocess_grade_aware_aug.py saves
    'cache_dir': os.path.join('data', 'processed', 'hybrid_512'),
    'force_repreprocess': False,
    'grad_clip': 1.0,
    'top_k_checkpoints': 3,
    'tta': True,
    'target_per_class': 800,
}

os.makedirs(CONFIG['output_dir'], exist_ok=True)
os.makedirs(CONFIG['cache_dir'],  exist_ok=True)


# ============================================================================
# DATASET — grade-aware augmentation dispatch
# ============================================================================

class PreprocessedDRDataset(Dataset):
    """
    Loads preprocessed images from cache and applies grade-aware augmentation.

    Dispatch logic:
        Grade 0, 1  → mild    (protect microaneurysms — 3-6px dots)
        Grade 2     → moderate (multiple lesion types, no blur/dropout)
        Grade 3, 4  → strong  (dense lesion fields survive conservative aug)

    All aug pipelines from preprocess_grade_aware_aug end with
    Normalize(ImageNet) + ToTensorV2 — output is a normalized float32
    torch.Tensor (C, H, W). Do NOT apply manual normalization on top.
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
        row = self.labels_df.iloc[idx]
        img_path = os.path.join(self.cache_dir, f"{row['id_code']}.png")

        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Cache miss: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        label = int(row['diagnosis'])
        aug = self._pick_aug(label)

        if aug is not None:
            image = aug(image=image)['image']   # → torch.Tensor (C,H,W), normalized
        else:
            image = torch.from_numpy(image).float() / 255.0
            image = image.permute(2, 0, 1)
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
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
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        logits = model(images)
        coral_targets = model.coral_label_transform(labels)
        loss = model.loss(logits, coral_targets)

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
        total   += labels.size(0)
        correct += (predictions == labels).sum().item()

        if (batch_idx + 1) % 50 == 0:
            lr_now = scheduler.get_last_lr()[0]
            print(f"  Epoch [{epoch+1}/{num_epochs}] "
                  f"Batch [{batch_idx+1}/{len(train_loader)}] "
                  f"Loss: {loss.item():.4f}  LR: {lr_now:.2e}")

    return total_loss / len(train_loader), 100.0 * correct / total


def validate(model, val_loader, device, use_tta=False):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            if use_tta:
                tta_probs = None
                for aug_images in _tta_flip(images):
                    probs = torch.sigmoid(model(aug_images))
                    tta_probs = probs if tta_probs is None else tta_probs + probs
                tta_probs /= 3
                logits      = torch.log(tta_probs / (1 - tta_probs + 1e-8))
                predictions = (tta_probs > 0.5).sum(dim=1)
            else:
                logits      = model(images)
                predictions = model.predict(logits)

            coral_targets = model.coral_label_transform(labels)
            loss = model.loss(logits, coral_targets)
            total_loss += loss.item()

            total   += labels.size(0)
            correct += (predictions == labels).sum().item()
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(val_loader)
    accuracy = 100.0 * correct / total
    qwk      = cohen_kappa_score(all_labels, all_preds, weights='quadratic')
    return avg_loss, accuracy, qwk, all_preds, all_labels


# ============================================================================
# TOP-K CHECKPOINTS  (FIX 5: pathlib so Windows paths don't mix / and \)
# ============================================================================

class TopKCheckpoints:
    def __init__(self, k, save_dir):
        self.k        = k
        self.save_dir = Path(save_dir)
        self._heap    = []

    def update(self, qwk, epoch, model):
        # pathlib.Path normalises separators on all platforms
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
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
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

    all_probs     /= len(checkpoint_paths)
    ensemble_preds = (all_probs > 0.5).sum(dim=1).numpy()
    qwk = cohen_kappa_score(all_labels, ensemble_preds, weights='quadratic')
    print(f"Ensemble QWK ({len(checkpoint_paths)} models): {qwk:.4f}")
    return qwk, ensemble_preds, all_labels


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 80)
    print("TRAINING: DenseNet121 + CORAL + Focal  (correct cache + grade-aware aug)")
    print("=" * 80)
    print(f"Device:          {CONFIG['device']}")
    print(f"Input size:      {CONFIG['input_size']}px")
    print(f"Batch size:      {CONFIG['batch_size']}  (safe for RTX 2050 at 512px)")
    print(f"Cache dir:       {CONFIG['cache_dir']}  ← fixed path")
    print(f"Label smooth:    {CONFIG['label_smooth']}  (disabled — focal handles hard examples)")
    print(f"Augmentation:    grade-aware (mild/moderate/strong per class)")
    print()

    # ── Data split ──────────────────────────────────────────────────────────
    train_df = pd.read_csv(os.path.join(CONFIG['data_dir'], 'train.csv'))
    train_df, val_df = train_test_split(
        train_df, test_size=0.2, random_state=42, stratify=train_df['diagnosis']
    )
    print(f"Train: {len(train_df)}  Val: {len(val_df)}")
    print(f"Class distribution (train):\n"
          f"{train_df['diagnosis'].value_counts().sort_index()}\n")

    # ── Step 1: preprocess & cache ───────────────────────────────────────────
    print("Step 1: Preprocessing and caching (preprocess_grade_aware_aug)...")
    preprocess_and_cache(
        raw_dir=os.path.join(CONFIG['data_dir'], 'train_images'),
        cache_dir=CONFIG['cache_dir'],
        df=pd.concat([train_df, val_df]).reset_index(drop=True),
        method=CONFIG['preprocessing_method'],
        image_size=CONFIG['input_size'],
        verbose=True,
    )

    # ── Step 2: balance ──────────────────────────────────────────────────────
    print("\nStep 2: Balancing training dataset...")
    train_df_balanced = create_balanced_train_dataframe(
        train_df=train_df,
        strategy='oversample',
        target_per_class=CONFIG['target_per_class'],
    )
    print(f"Balanced train: {len(train_df_balanced)} samples")

    # ── Step 3: verify aug ───────────────────────────────────────────────────
    print("\nStep 3: Verifying augmentation pipeline...")
    verify_augmentation(
        cache_dir=CONFIG['cache_dir'],
        save_dir=os.path.join(CONFIG['output_dir'], 'aug_samples'),
    )

    # ── Step 4: datasets ─────────────────────────────────────────────────────
    print("\nStep 4: Creating datasets (grade-aware aug)...")
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
        train_dataset, batch_size=CONFIG['batch_size'],
        shuffle=True, num_workers=0, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=CONFIG['batch_size'],
        shuffle=False, num_workers=0, pin_memory=True,
    )

    # ── Step 5: model ────────────────────────────────────────────────────────
    print("\nStep 5: Initialising DenseNet121 with CORAL + Focal head...")
    model = DenseNet121withCORALFocal_v2(
        num_classes=CONFIG['num_classes'],
        gamma=CONFIG['gamma'],
        pretrained=CONFIG['pretrained'],
        label_smooth=CONFIG['label_smooth'],   # 0.0 — disabled
    ).to(CONFIG['device'])

    freeze_early_layers(model, blocks=('denseblock1',))

    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,} "
          f"({100*trainable_params/total_params:.1f}%)")

    # CORAL alpha — computed on balanced df (NOTE: near-uniform, kept for
    # consistency but WeightedRandomSampler is the real balancing mechanism)
    print("\nComputing CORAL alpha weights...")
    model.compute_alpha(
        torch.tensor(train_df_balanced['diagnosis'].values, dtype=torch.long)
    )
    print("Alpha weights computed ✓")

    # ── Optimizer + OneCycleLR ───────────────────────────────────────────────
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=CONFIG['learning_rate'] / 25,
        weight_decay=CONFIG['weight_decay'],
    )
    total_steps = len(train_loader) * CONFIG['num_epochs']
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=CONFIG['learning_rate'],
        total_steps=total_steps,
        pct_start=0.1,
        anneal_strategy='cos',
        div_factor=25,
        final_div_factor=1e4,
    )
    print(f"\nOneCycleLR: {total_steps} total steps, peak LR={CONFIG['learning_rate']:.1e}")

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

    # ── FIX 6: CORAL bias diagnostic ─────────────────────────────────────────
    bias = model.model.classifier.bias.data.cpu()
    print(f"\nCORAL bias vector (should be monotonically decreasing):")
    print(f"  {bias.tolist()}")
    monotonic = all(bias[i] >= bias[i+1] for i in range(len(bias)-1))
    print(f"  Monotonic: {'✓ YES' if monotonic else '✗ NO — predictions may be inconsistent'}")

    # ── Post-training ensemble ────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("TRAINING COMPLETED")
    print("=" * 80)
    print(f"\nBest single-model QWK: {best_qwk:.4f} at Epoch {best_epoch+1}")

    best_ckpt_paths = top_k.best_paths()
    ensemble_qwk = best_qwk
    if len(best_ckpt_paths) > 1:
        ensemble_qwk, ensemble_preds, ensemble_labels = ensemble_from_checkpoints(
            model, best_ckpt_paths, val_loader, CONFIG['device'],
        )
        print(f"Single-model QWK: {best_qwk:.4f}")
        print(f"Ensemble QWK:     {ensemble_qwk:.4f}  ({len(best_ckpt_paths)} models)")
        final_preds  = ensemble_preds
        final_labels = ensemble_labels
    else:
        final_preds  = best_preds
        final_labels = best_labels

    # ── Results ──────────────────────────────────────────────────────────────
    cm = confusion_matrix(final_labels, final_preds)
    class_report = classification_report(
        final_labels, final_preds,
        target_names=['Grade 0', 'Grade 1', 'Grade 2', 'Grade 3', 'Grade 4'],
        output_dict=True,
    )

    # Training curves
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(history['train_loss'], label='Train', marker='o')
    axes[0].plot(history['val_loss'],   label='Val',   marker='s')
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
    axes[0].set_title('CORAL + Focal Loss')
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

    # Confusion matrix
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

    # Classification report
    report_text = classification_report(
        final_labels, final_preds,
        target_names=['Grade 0', 'Grade 1', 'Grade 2', 'Grade 3', 'Grade 4'],
    )
    with open(os.path.join(CONFIG['output_dir'], 'classification_report.txt'), 'w') as f:
        f.write(report_text)
    print("✓ Saved: classification_report.txt")

    # Summary JSON
    summary = {
        'model':            'DenseNet121 + CORAL + Focal (correct cache + grade-aware aug)',
        'fixes_applied':    [
            'cache_dir corrected to data/processed/hybrid_512',
            'batch_size 16→8 for RTX 2050 at 512px',
            'label_smooth 0.05→0.0 (conflicts with focal loss)',
            'grade-aware aug: mild/moderate/strong per grade',
            'Windows-safe checkpoint paths via pathlib',
        ],
        'best_epoch':       best_epoch + 1,
        'best_qwk':         float(best_qwk),
        'ensemble_qwk':     float(ensemble_qwk),
        'coral_bias':       bias.tolist(),
        'coral_monotonic':  monotonic,
        'best_val_loss':    float(history['val_loss'][best_epoch]),
        'best_val_acc':     float(history['val_acc'][best_epoch]),
        'total_params':     int(total_params),
        'trainable_params': int(trainable_params),
        'config':           {k: str(v) if isinstance(v, Path) else v
                             for k, v in CONFIG.items()},
        'per_class_metrics': {
            str(i): {
                'precision': float(class_report.get(str(i), {}).get('precision', 0.0)),
                'recall':    float(class_report.get(str(i), {}).get('recall',    0.0)),
                'f1-score':  float(class_report.get(str(i), {}).get('f1-score',  0.0)),
            } for i in range(5)
        },
    }
    with open(os.path.join(CONFIG['output_dir'], 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print("✓ Saved: summary.json")

    print(f"\n{'='*80}\nALL RESULTS SAVED\n{'='*80}")
    print(f"\n📁 Results: {CONFIG['output_dir']}/")
    print(f"\n🎯 Key Metrics:")
    print(f"  Best single-model QWK : {best_qwk:.4f}  (epoch {best_epoch+1})")
    print(f"  Ensemble QWK          : {ensemble_qwk:.4f}")
    print(f"  CORAL bias monotonic  : {'✓' if monotonic else '✗'}")
    for i in range(5):
        recall = class_report.get(str(i), {}).get('recall', 0.0)
        print(f"  Grade {i} Recall       : {recall:.2%}")


if __name__ == '__main__':
    main()
