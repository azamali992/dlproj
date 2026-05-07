"""
Evaluation Metrics for Diabetic Retinopathy Classification
===========================================================

Primary metric: Quadratic Weighted Kappa (QWK)

Reference: Cohen (1968) - "Weighted kappa: nominal scale agreement
provision for scaled disagreement or partial credit"
"""

import numpy as np
from sklearn.metrics import (
    cohen_kappa_score,
    confusion_matrix,
    classification_report,
    accuracy_score,
)
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from datetime import datetime


CLASS_NAMES = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative']


def quadratic_weighted_kappa(y_true, y_pred):
    """Primary metric for APTOS 2019 competition."""
    return cohen_kappa_score(y_true, y_pred, weights='quadratic')


def compute_all_metrics(y_true, y_pred):
    """Compute comprehensive metrics."""
    return {
        'qwk': quadratic_weighted_kappa(y_true, y_pred),
        'accuracy': accuracy_score(y_true, y_pred),
        'confusion_matrix': confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3, 4]),
        'report': classification_report(
            y_true, y_pred,
            target_names=CLASS_NAMES,
            digits=4,
            output_dict=True,
        ),
    }


def print_metrics(metrics):
    """Pretty-print metrics to console."""
    print("=" * 60)
    print(f"  QWK:      {metrics['qwk']:.4f}  (PRIMARY METRIC)")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print("=" * 60)

    report = metrics['report']
    print(f"\n{'Class':<16} {'Precision':<11} {'Recall':<11} {'F1':<11} {'Support':<8}")
    print("-" * 57)
    for name in CLASS_NAMES:
        r = report[name]
        print(f"{name:<16} {r['precision']:<11.4f} {r['recall']:<11.4f} "
              f"{r['f1-score']:<11.4f} {int(r['support']):<8}")
    print("-" * 57)


def save_confusion_matrix(cm, save_path, title='Confusion Matrix'):
    """Save a publication-quality confusion matrix figure."""
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
        linewidths=0.5, linecolor='gray', ax=ax,
    )
    ax.set_xlabel('Predicted Grade', fontweight='bold')
    ax.set_ylabel('True Grade', fontweight='bold')
    ax.set_title(title, fontweight='bold', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()


def save_training_curves(history, save_path):
    """Save loss + QWK training curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    axes[0].plot(history['train_loss'], label='Train', linewidth=2)
    axes[0].plot(history['val_loss'], label='Validation', linewidth=2)
    axes[0].set_xlabel('Epoch', fontweight='bold')
    axes[0].set_ylabel('Loss', fontweight='bold')
    axes[0].set_title('Loss Curves', fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # QWK
    axes[1].plot(history['val_qwk'], linewidth=2, color='green', label='Val QWK')
    if 'train_qwk' in history:
        axes[1].plot(history['train_qwk'], linewidth=2, color='blue',
                     label='Train QWK', alpha=0.7)
    axes[1].set_xlabel('Epoch', fontweight='bold')
    axes[1].set_ylabel('QWK', fontweight='bold')
    axes[1].set_title('Quadratic Weighted Kappa', fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()


def log_experiment(config, metrics, save_dir):
    """Save experiment results as JSON for easy comparison."""
    log = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'config': config,
        'results': {
            'qwk': float(metrics['qwk']),
            'accuracy': float(metrics['accuracy']),
        },
    }

    log_path = os.path.join(save_dir, 'experiment_log.json')
    plt.close('all')

    with open(log_path, 'w') as f:
        json.dump(log, f, indent=2)

    print(f"Experiment logged to: {log_path}")