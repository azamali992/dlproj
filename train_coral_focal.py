"""
Training Script: CORAL + Focal Loss with Preprocessing
=======================================================

Uses:
1. preprocess.py - CLAHE/Ben Graham preprocessing
2. utils.losses.py - CORAL + Focal Loss Module
3. utils.models.py - ResNet18 with CORAL head
4. Proper data handling and class weighting

This script implements ordinal regression for diabetic retinopathy grading,
respecting the natural ordering: Grade 0 < 1 < 2 < 3 < 4
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from pathlib import Path
from PIL import Image
import cv2
import warnings
warnings.filterwarnings('ignore')

# Import preprocessing
from preprocess import (
    preprocess_and_cache,
    get_train_augmentation,
    get_val_augmentation
)

# Import your model and losses
from utils.losses import CORALModule
from utils.models import DenseNet121withCORALFocal

# ============================================================================
# CONFIGURATION
# ============================================================================
CONFIG = {
    'model_name': 'densenet121_coral_focal',
    'num_classes': 5,
    'input_size': 224,
    'batch_size': 16,
    'num_epochs': 30,
    'learning_rate': 0.0001,  # Reduced 10x (was 0.001 - caused spike)
    'weight_decay': 1e-3,     # Increased 10x (was 1e-4 - reduce overfitting)
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'data_dir': './data/raw',
    'output_dir': './results/exp_coral_focal',
    'resume_from_checkpoint': False,
    
    # Preprocessing choice: 'ben_graham', 'clahe', or 'hybrid'
    'preprocessing_method': 'hybrid',
    
    # CORAL + Focal parameters
    'gamma': 2.0,  # Focal loss focusing parameter
    'pretrained': True,

    # Cache preprocessed images to disk (preprocess once, load fast every epoch)
    'cache_dir': './data/preprocessed_hybrid',
    'force_repreprocess': False,  # Set True to redo even if cache exists

    # Class-balanced augmentation
    # Minority classes get augmented up to the majority class count
    'use_augmentation': True,
    'force_reaugment': True,    # Force redo with new target_per_class
    'target_per_class': 800,   # Reduced from 1444 (auto) — prevents over-augmentation
}

os.makedirs(CONFIG['output_dir'], exist_ok=True)
os.makedirs(CONFIG['cache_dir'], exist_ok=True)

# ============================================================================
# PREPROCESSING CACHE (Preprocess once, load fast every epoch)
# ============================================================================
# ============================================================================
# DATASET (Loads from cache — fast!)
# ============================================================================
class PreprocessedDRDataset(Dataset):
    def __init__(self, cache_dir, labels_df, transform=None, augmentation=None):
        self.cache_dir = cache_dir
        self.labels_df = labels_df.reset_index(drop=True)
        self.transform = transform
        self.augmentation = augmentation  # Albumentations

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        row = self.labels_df.iloc[idx]
        img_path = os.path.join(self.cache_dir, f"{row['id_code']}.png")

        # Load image as numpy array
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply albumentations augmentation (if provided)
        if self.augmentation:
            augmented = self.augmentation(image=image)
            image = augmented['image']

        # Convert to tensor
        image = torch.from_numpy(image).float() / 255.0
        image = image.permute(2, 0, 1)
        
        # Normalize
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image = (image - mean) / std

        label = int(row['diagnosis'])
        return image, label

# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================
def train_epoch(model, train_loader, optimizer, device, epoch, num_epochs):
    """Train one epoch with CORAL + Focal loss"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        logits = model(images)
        
        # Transform labels to CORAL ordinal encoding
        coral_targets = model.coral_label_transform(labels)
        
        # Compute CORAL + Focal loss
        loss = model.loss(logits, coral_targets)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Metrics
        total_loss += loss.item()
        predictions = model.predict(logits)
        total += labels.size(0)
        correct += (predictions == labels).sum().item()
        
        # Progress
        if (batch_idx + 1) % 50 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] Batch [{batch_idx+1}/{len(train_loader)}] "
                  f"Loss: {loss.item():.4f}")
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy


def validate(model, val_loader, device):
    """Validate model with CORAL predictions"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            logits = model(images)
            
            # Loss
            coral_targets = model.coral_label_transform(labels)
            loss = model.loss(logits, coral_targets)
            total_loss += loss.item()
            
            # Predictions
            predictions = model.predict(logits)
            total += labels.size(0)
            correct += (predictions == labels).sum().item()
            
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(val_loader)
    accuracy = 100 * correct / total
    qwk = cohen_kappa_score(all_labels, all_preds, weights='quadratic')
    
    return avg_loss, accuracy, qwk, all_preds, all_labels


# ============================================================================
# CHECKPOINT FUNCTIONS
# ============================================================================
def save_checkpoint(checkpoint_path, epoch, model, optimizer, history,
                    best_qwk, best_epoch, best_preds, best_labels):
    """Save training checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'history': history,
        'best_qwk': best_qwk,
        'best_epoch': best_epoch,
        'best_preds': best_preds,
        'best_labels': best_labels,
    }
    torch.save(checkpoint, checkpoint_path)


def load_checkpoint(checkpoint_path, model, optimizer, device):
    """Load training checkpoint"""
    print(f"\n{'='*80}")
    print(f"RESUMING FROM CHECKPOINT")
    print(f"{'='*80}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    
    start_epoch = checkpoint['epoch']
    history = checkpoint['history']
    best_qwk = checkpoint['best_qwk']
    best_epoch = checkpoint['best_epoch']
    best_preds = checkpoint['best_preds']
    best_labels = checkpoint['best_labels']
    
    print(f"✓ Resumed from epoch {start_epoch + 1}")
    print(f"✓ Previous best QWK: {best_qwk:.4f} (at epoch {best_epoch + 1})")
    print(f"{'='*80}\n")
    
    return start_epoch, history, best_qwk, best_epoch, best_preds, best_labels


# ============================================================================
# MAIN TRAINING
# ============================================================================
def main():
    print("="*80)
    print("TRAINING: ResNet18 with CORAL + Focal Loss")
    print("="*80)
    print(f"Preprocessing method: {CONFIG['preprocessing_method']}")
    print(f"Device: {CONFIG['device']}")
    print()
    
    # Load data
    print("Loading dataset...")
    train_df = pd.read_csv(os.path.join(CONFIG['data_dir'], 'train.csv'))
    
    # Split
    from sklearn.model_selection import train_test_split
    train_df, val_df = train_test_split(
        train_df,
        test_size=0.2,
        random_state=42,
        stratify=train_df['diagnosis']
    )
    
    print(f"Training samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    print(f"Class distribution:\n{train_df['diagnosis'].value_counts().sort_index()}\n")
    
    # ================================================================
    # STEP 1: Preprocess and cache all images
    # ================================================================
    preprocess_and_cache(
        raw_dir=os.path.join(CONFIG['data_dir'], 'train_images'),
        cache_dir=CONFIG['cache_dir'],
        df=pd.concat([train_df, val_df]).reset_index(drop=True),
        method=CONFIG['preprocessing_method'],
        verbose=True
    )

    # ================================================================
    # STEP 2: Build WeightedRandomSampler for equal class sampling
    # ================================================================
    class_counts = np.bincount(train_df['diagnosis'].values, minlength=CONFIG['num_classes'])
    print(f"\nClass counts (train): {dict(enumerate(class_counts))}")
    class_weights = 1.0 / class_counts.astype(float)
    sample_weights = [float(class_weights[label]) for label in train_df['diagnosis'].values]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    print(f"WeightedRandomSampler: {len(sample_weights)} samples, equal class probability ✓")

    # ================================================================
    # STEP 3: Create datasets with augmentation at dataset level
    # ================================================================
    print(f"\nCreating datasets with augmentation...")
    train_dataset = PreprocessedDRDataset(
        CONFIG['cache_dir'],
        train_df,
        augmentation=get_train_augmentation()
    )
    val_dataset = PreprocessedDRDataset(
        CONFIG['cache_dir'],
        val_df,
        augmentation=get_val_augmentation()
    )

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG['batch_size'],
        sampler=sampler,        # replaces shuffle=True; sampler draws equal class batches
        num_workers=0,          # Must be 0 on Windows (cv2 multiprocessing issue)
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    # Model
    print("\nInitializing DenseNet121 with CORAL + Focal head...")
    model = DenseNet121withCORALFocal(
        num_classes=CONFIG['num_classes'],
        gamma=CONFIG['gamma'],
        pretrained=CONFIG['pretrained']
    )
    model = model.to(CONFIG['device'])

    # Freeze early DenseNet blocks (denseblock1, denseblock2) to reduce overfitting
    print("\nFreezing early backbone layers (denseblock1, denseblock2)...")
    for name, param in model.model.named_parameters():
        if 'denseblock1' in name or 'denseblock2' in name:
            param.requires_grad = False
    print("✓ denseblock1 and denseblock2 frozen")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.1f}%)")
    print(f"Frozen parameters:    {total_params - trainable_params:,} ({100*(total_params-trainable_params)/total_params:.1f}%)")

    # Compute CORAL alpha weights (imbalance handling)
    print("\nComputing CORAL alpha weights for class imbalance...")
    model.compute_alpha(torch.tensor(train_df['diagnosis'].values, dtype=torch.long))
    print(f"Alpha weights computed ✓")

    # Optimizer (only trainable params)
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=CONFIG['learning_rate'],
        weight_decay=CONFIG['weight_decay']
    )

    # Cosine annealing scheduler (smoother than ReduceLROnPlateau — prevents spikes)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=CONFIG['num_epochs'],
        eta_min=1e-6
    )
    
    # Checkpoint
    checkpoint_path = os.path.join(CONFIG['output_dir'], 'checkpoint.pth')
    
    start_epoch = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'val_qwk': []}
    best_qwk = 0
    best_epoch = 0
    best_preds = None
    best_labels = None
    
    if CONFIG['resume_from_checkpoint'] and os.path.exists(checkpoint_path):
        start_epoch, history, best_qwk, best_epoch, best_preds, best_labels = \
            load_checkpoint(checkpoint_path, model, optimizer, CONFIG['device'])
    else:
        print(f"\n{'='*80}")
        print(f"STARTING NEW TRAINING")
        print(f"{'='*80}\n")
    
    # Training loop
    print("="*80)
    print("TRAINING STARTED")
    print("="*80)
    
    for epoch in range(start_epoch, CONFIG['num_epochs']):
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, CONFIG['device'], epoch, CONFIG['num_epochs']
        )
        
        val_loss, val_acc, val_qwk, val_preds, val_labels = validate(
            model, val_loader, CONFIG['device']
        )
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_qwk'].append(val_qwk)
        
        print(f"\nEpoch {epoch+1}/{CONFIG['num_epochs']} Summary:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | Val QWK: {val_qwk:.4f}")
        
        # Save best model
        if val_qwk > best_qwk:
            best_qwk = val_qwk
            best_epoch = epoch
            best_preds = val_preds
            best_labels = val_labels
            torch.save(model.state_dict(), os.path.join(CONFIG['output_dir'], 'best_model.pth'))
            print(f"  ✓ Best model saved! (QWK: {val_qwk:.4f})")
        
        # Save checkpoint
        save_checkpoint(checkpoint_path, epoch, model, optimizer, history,
                       best_qwk, best_epoch, best_preds, best_labels)
        print(f"  💾 Checkpoint saved")
        
        scheduler.step()  # CosineAnnealingLR steps every epoch
    
    # Results
    print(f"\n{'='*80}")
    print("TRAINING COMPLETED")
    print(f"{'='*80}")
    print(f"Best QWK: {best_qwk:.4f} at Epoch {best_epoch+1}")
    
    # Confusion matrix
    cm = confusion_matrix(best_labels, best_preds)
    
    # Classification report
    class_report = classification_report(
        best_labels, best_preds,
        target_names=['Grade 0', 'Grade 1', 'Grade 2', 'Grade 3', 'Grade 4'],
        output_dict=True
    )
    
    # Plot training curves
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
        best_labels, best_preds,
        target_names=['Grade 0', 'Grade 1', 'Grade 2', 'Grade 3', 'Grade 4']
    )
    
    with open(os.path.join(CONFIG['output_dir'], 'classification_report.txt'), 'w') as f:
        f.write(report_text)
    print(f"✓ Saved: classification_report.txt")
    
    # Summary
    summary = {
        'model': 'ResNet18 with CORAL + Focal Loss',
        'approach': 'Ordinal regression with preprocessing',
        'preprocessing': CONFIG['preprocessing_method'],
        'best_epoch': best_epoch + 1,
        'best_qwk': float(best_qwk),
        'best_val_loss': float(history['val_loss'][best_epoch]),
        'best_val_acc': float(history['val_acc'][best_epoch]),
        'total_params': int(total_params),
        'trainable_params': int(trainable_params),
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
    
    # Cleanup
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print(f"✓ Cleanup: Removed checkpoint.pth (training complete)")
    
    # Final Summary
    print(f"\n{'='*80}")
    print("ALL RESULTS SAVED")
    print(f"{'='*80}")
    print(f"\n📁 Results folder: {CONFIG['output_dir']}/")
    print(f"\n📊 Main Results:")
    print(f"  ✓ best_model.pth (trained weights)")
    print(f"  ✓ training_curves.png (loss & accuracy)")
    print(f"  ✓ confusion_matrix.png")
    print(f"  ✓ classification_report.txt")
    print(f"  ✓ summary.json")
    print(f"\n🎯 Key Metrics:")
    print(f"  Best QWK: {best_qwk:.4f}")
    print(f"  Best Epoch: {best_epoch + 1}")
    print(f"  Preprocessing: {CONFIG['preprocessing_method']}")
    for i in range(5):
        metrics = class_report.get(str(i), {})
        recall = metrics.get('recall', 0.0)
        print(f"  Grade {i} Recall: {recall:.2%}")


if __name__ == '__main__':
    main()