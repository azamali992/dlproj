"""
run_ensemble_v7.py
==================
Run the ensemble step for exp_v7 without re-training.
Uses the top-3 checkpoints already saved in results/exp_v7/.

Usage:
  python run_ensemble_v7.py
"""

import os, json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import cohen_kappa_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from pathlib import Path
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from preprocess_grade_aware_aug import (
    get_val_augmentation, IMAGENET_MEAN, IMAGENET_STD)
from utils.losses_v5 import CORALModule_v5
from utils.models_v5 import DenseNet121withCORALFocal_v5
from train_coral_focal_v7 import PreprocessedDRDataset, build_dataframes

CONFIG = {
    'num_classes':          5,
    'input_size':           224,
    'batch_size':           16,
    'device':               'cuda' if torch.cuda.is_available() else 'cpu',
    'data_dir':             './data/raw',
    'output_dir':           os.path.join('results', 'exp_v7'),
    'gamma':                2.0,
    'lambda_ord':           0.1,
    'cache_dir':            os.path.join('data', 'processed', 'hybrid_224'),
    'supplement_csv':       os.path.join('data', 'eyepacs_supplement', 'supplement.csv'),
    'supplement_cache_dir': os.path.join('data', 'eyepacs_supplement', 'processed', 'hybrid_224'),
}

def main():
    device = CONFIG['device']
    out    = CONFIG['output_dir']

    # Find saved checkpoints
    ckpt_paths = sorted(
        Path(out).glob('model_qwk*.pth'),
        key=lambda p: float(p.stem.split('qwk')[1].split('_')[0]),
        reverse=True,
    )
    if not ckpt_paths:
        raise FileNotFoundError(f"No checkpoints found in {out}/")
    print(f"Found {len(ckpt_paths)} checkpoints:")
    for p in ckpt_paths:
        print(f"  {p.name}")

    # Build validation set (same split as training)
    _, _, val_df, _ = build_dataframes()
    val_aug     = get_val_augmentation()
    val_dataset = PreprocessedDRDataset(
        val_df, val_aug=val_aug, is_train=False)
    val_loader  = DataLoader(val_dataset, batch_size=CONFIG['batch_size'],
                             shuffle=False, num_workers=0)

    # Load model skeleton
    model = DenseNet121withCORALFocal_v5(
        num_classes=CONFIG['num_classes'],
        gamma=CONFIG['gamma'],
        pretrained=False,
        lambda_ord=CONFIG['lambda_ord'],
    ).to(device)

    # Ensemble
    print(f"\nEnsembling {len(ckpt_paths)} checkpoints ...")
    all_probs  = None
    all_labels = []
    for ckpt in ckpt_paths:
        model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
        model.eval()
        batch_probs = []
        collect     = not all_labels
        with torch.no_grad():
            for images, labels in val_loader:
                probs = torch.sigmoid(model(images.to(device))).cpu()
                batch_probs.append(probs)
                if collect:
                    all_labels.extend(labels.numpy())
        ep        = torch.cat(batch_probs)
        all_probs = ep if all_probs is None else all_probs + ep

    all_probs /= len(ckpt_paths)
    preds = (all_probs > 0.5).sum(dim=1).numpy()
    qwk   = cohen_kappa_score(all_labels, preds, weights='quadratic')
    print(f"\nEnsemble QWK ({len(ckpt_paths)} models): {qwk:.4f}")

    # Classification report
    grade_names = ['Grade 0 (No DR)', 'Grade 1 (Mild)',
                   'Grade 2 (Moderate)', 'Grade 3 (Severe)',
                   'Grade 4 (Proliferative)']
    print("\n" + classification_report(all_labels, preds, target_names=grade_names))

    # Confusion matrix
    cm = confusion_matrix(all_labels, preds)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['G0','G1','G2','G3','G4'],
                yticklabels=['G0','G1','G2','G3','G4'])
    plt.title(f'exp_v7 ensemble confusion matrix  (QWK={qwk:.4f})')
    plt.ylabel('True'); plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(os.path.join(out, 'confusion_matrix_ensemble.png'), dpi=150)
    plt.close()
    print(f"Saved: {out}/confusion_matrix_ensemble.png")

    # Training curves from history.json
    history_path = Path(out) / 'history.json'
    if history_path.exists():
        with open(history_path) as f:
            history = json.load(f)

        epochs = range(1, len(history['train_loss']) + 1)
        best_qwk_val = max(history['val_qwk'])
        best_ep      = history['val_qwk'].index(best_qwk_val) + 1

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        # Loss
        axes[0].plot(epochs, history['train_loss'], marker='o', ms=3, label='train loss')
        axes[0].plot(epochs, history['val_loss'],   marker='o', ms=3, label='val loss')
        axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
        axes[0].set_title('Loss'); axes[0].legend(); axes[0].grid(True, alpha=0.3)

        # Accuracy
        axes[1].plot(epochs, history['train_acc'], marker='o', ms=3, label='train acc')
        axes[1].plot(epochs, history['val_acc'],   marker='o', ms=3, label='val acc')
        axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Accuracy (%)')
        axes[1].set_title('Accuracy'); axes[1].legend(); axes[1].grid(True, alpha=0.3)

        # QWK
        axes[2].plot(epochs, history['val_qwk'], marker='o', ms=3,
                     label='val QWK', color='green')
        axes[2].axhline(best_qwk_val, color='r', ls='--',
                        label=f'best single {best_qwk_val:.4f} (ep {best_ep})')
        axes[2].axhline(qwk, color='purple', ls='-.',
                        label=f'ensemble {qwk:.4f}')
        axes[2].set_xlabel('Epoch'); axes[2].set_ylabel('QWK')
        axes[2].set_title('Quadratic Weighted Kappa')
        axes[2].legend(); axes[2].grid(True, alpha=0.3)

        fig.suptitle('exp_v7 training curves', fontsize=11)
        plt.tight_layout()
        plt.savefig(os.path.join(out, 'training_curves.png'), dpi=150)
        plt.close(fig)
        print(f"Saved: {out}/training_curves.png")
    else:
        print(f"[WARN] {history_path} not found — skipping training curves.")

    # Patch summary.json with the correct ensemble QWK
    summary_path = Path(out) / 'summary.json'
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)
        summary['ensemble_qwk'] = float(qwk)
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Updated summary.json  ensemble_qwk={qwk:.4f}")

if __name__ == '__main__':
    main()
