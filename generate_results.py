"""
generate_results.py
===================
Re-generate all post-training outputs from saved checkpoint files.

Use this if training was interrupted after some checkpoints were already
saved, or to regenerate results for any experiment at any time.

Usage
-----
    python generate_results.py --exp exp_v6

It will:
  - Find all model_qwk*.pth files in the experiment folder
  - Run ensemble evaluation on the validation set
  - Generate Grad-CAM analysis
  - Save confusion matrix, classification report, summary.json

All paths and model settings are read from the experiment's summary.json
if it exists; otherwise sensible defaults for exp_v6 are used.
"""

import os, json, argparse, warnings
import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import (cohen_kappa_score, confusion_matrix,
                             classification_report)
from sklearn.model_selection import train_test_split
from pathlib import Path
warnings.filterwarnings('ignore')

from preprocess_grade_aware_aug import (
    get_val_augmentation,
    IMAGENET_MEAN,
    IMAGENET_STD,
)
from utils.models_v5 import DenseNet121withCORALFocal_v5


# ── Dataset (val-only, no augmentation) ───────────────────────────────────────
class ValDataset(Dataset):
    def __init__(self, cache_dir, labels_df, val_aug):
        self.cache_dir = cache_dir
        self.df        = labels_df.reset_index(drop=True)
        self.aug       = val_aug

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        img   = cv2.imread(os.path.join(self.cache_dir, f"{row['id_code']}.png"))
        if img is None:
            raise FileNotFoundError(os.path.join(self.cache_dir,
                                                  f"{row['id_code']}.png"))
        img   = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img   = self.aug(image=img)['image']
        return img, int(row['diagnosis'])


# ── Helpers ───────────────────────────────────────────────────────────────────
def tensor_to_uint8(t):
    arr = t.cpu().numpy()
    if arr.ndim == 3:
        arr = arr.transpose(1, 2, 0)
    arr = arr * np.array(IMAGENET_STD) + np.array(IMAGENET_MEAN)
    return np.clip(arr * 255, 0, 255).astype(np.uint8)


def run_ensemble(model, ckpt_paths, loader, device):
    print(f"Ensembling {len(ckpt_paths)} checkpoint(s) ...")
    all_probs  = None
    all_labels = None   # collected once from the first checkpoint pass

    for ckpt in ckpt_paths:
        model.load_state_dict(torch.load(ckpt, map_location=device,
                                         weights_only=True))
        model.eval()
        batch_probs, batch_labels = [], []
        with torch.no_grad():
            for images, labels in loader:
                probs = torch.sigmoid(model(images.to(device))).cpu()
                batch_probs.append(probs)
                batch_labels.extend(labels.numpy())

        ep = torch.cat(batch_probs)
        all_probs  = ep if all_probs is None else all_probs + ep
        if all_labels is None:
            all_labels = batch_labels   # save ground-truth labels once

    all_probs /= len(ckpt_paths)
    preds = (all_probs > 0.5).sum(dim=1).numpy()
    qwk   = cohen_kappa_score(all_labels, preds, weights='quadratic')
    return qwk, preds, all_labels


def run_gradcam(model, val_dataset, val_labels, val_preds, device,
                output_dir, n_samples=3):
    gradcam_dir = os.path.join(output_dir, 'gradcam')
    os.makedirs(gradcam_dir, exist_ok=True)

    grade_names  = {0:'G0_NoDR', 1:'G1_Mild', 2:'G2_Moderate',
                    3:'G3_Severe', 4:'G4_Proliferative'}
    boundary_desc= {0:'T0: No DR -> any DR',
                    1:'T0: No DR -> any DR',
                    2:'T1: Mild -> Moderate',
                    3:'T2: Moderate -> Severe',
                    4:'T3: Severe -> Proliferative'}

    model.eval()
    all_rows = []

    for grade in range(5):
        grade_out = os.path.join(gradcam_dir, grade_names[grade])
        os.makedirs(grade_out, exist_ok=True)

        correct_idx = [i for i, (t, p) in enumerate(zip(val_labels, val_preds))
                       if t == grade and p == grade][:n_samples]
        if not correct_idx:
            correct_idx = [i for i, t in enumerate(val_labels)
                           if t == grade][:n_samples]

        row_imgs = []
        for k, idx in enumerate(correct_idx):
            image_tensor, _ = val_dataset[idx]
            image_tensor    = image_tensor.unsqueeze(0).to(device)
            logits          = model(image_tensor)
            cam             = model.generate_gradcam(logits, target_class=grade)
            cam_np          = cv2.resize(cam.squeeze().cpu().numpy(), (224, 224))
            orig_np         = tensor_to_uint8(image_tensor.squeeze(0))
            heatmap         = cv2.cvtColor(
                cv2.applyColorMap((cam_np * 255).astype(np.uint8),
                                  cv2.COLORMAP_JET),
                cv2.COLOR_BGR2RGB)
            overlay = np.clip(
                0.55 * orig_np.astype(float) + 0.45 * heatmap.astype(float),
                0, 255).astype(np.uint8)

            for name, img in [('orig', orig_np),
                               ('heatmap', heatmap),
                               ('overlay', overlay)]:
                cv2.imwrite(os.path.join(grade_out, f'img{k}_{name}.png'),
                            cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            row_imgs.append((orig_np, heatmap, overlay))
        all_rows.append((grade, row_imgs))

    # Summary grid
    n_cols = n_samples * 3
    fig, axes = plt.subplots(5, n_cols, figsize=(n_cols * 2.0, 5 * 2.2))
    for r, (grade, row_imgs) in enumerate(all_rows):
        for k, (orig, heat, over) in enumerate(row_imgs):
            for c_off, img in enumerate([orig, heat, over]):
                ax = axes[r, k * 3 + c_off]
                ax.imshow(img); ax.axis('off')
        axes[r, 0].set_ylabel(f'{grade_names[grade]}\n{boundary_desc[grade]}',
                               fontsize=7, rotation=0, labelpad=60, va='center')
    fig.suptitle('Grad-CAM -- boundary-specific saliency per DR grade', fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(gradcam_dir, 'gradcam_grid.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Grad-CAM saved: {gradcam_dir}/gradcam_grid.png")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', default='exp_v6',
                        help='Experiment folder name under results/')
    parser.add_argument('--data-dir', default='./data/raw')
    parser.add_argument('--cache-dir',
                        default=os.path.join('data', 'processed', 'hybrid_224'))
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--gradcam-samples', type=int, default=3)
    parser.add_argument('--no-gradcam', action='store_true')
    args = parser.parse_args()

    exp_dir = os.path.join('results', args.exp)
    device  = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Find checkpoints
    ckpt_paths = sorted(
        Path(exp_dir).glob('model_qwk*.pth'),
        key=lambda p: float(p.stem.split('qwk')[1].split('_')[0]),
        reverse=True,
    )
    if not ckpt_paths:
        raise FileNotFoundError(
            f"No model_qwk*.pth files found in {exp_dir}.\n"
            "Make sure training has run for at least one epoch.")

    print(f"Found {len(ckpt_paths)} checkpoint(s):")
    for p in ckpt_paths:
        print(f"  {p.name}")

    # Rebuild the same val split used during training
    full_df = pd.read_csv(os.path.join(args.data_dir, 'train.csv'))
    _, val_df = train_test_split(full_df, test_size=0.2, random_state=42,
                                 stratify=full_df['diagnosis'])

    val_aug     = get_val_augmentation()
    val_dataset = ValDataset(args.cache_dir, val_df, val_aug)
    val_loader  = DataLoader(val_dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=0, pin_memory=True)

    print(f"\nValidation set: {len(val_df)} images")
    print(val_df['diagnosis'].value_counts().sort_index().to_string())

    # Load model (architecture identical to exp_v5/v6)
    model = DenseNet121withCORALFocal_v5(
        num_classes=5, gamma=2.0, pretrained=False, lambda_ord=0.1
    ).to(device)

    # Ensemble
    ensemble_qwk, final_preds, final_labels = run_ensemble(
        model, [str(p) for p in ckpt_paths], val_loader, device)

    # Save best_model.pth
    import shutil
    best_model_path = os.path.join(exp_dir, 'best_model.pth')
    shutil.copy(str(ckpt_paths[0]), best_model_path)
    print(f"Best model saved: {best_model_path}")

    # Training curves (loaded from history.json written each epoch)
    history_path = os.path.join(exp_dir, 'history.json')
    if os.path.exists(history_path):
        with open(history_path) as f:
            history = json.load(f)
        epochs_ran = len(history['val_qwk'])
        best_qwk   = max(history['val_qwk'])
        best_epoch = history['val_qwk'].index(best_qwk)

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        x = range(1, epochs_ran + 1)

        axes[0].plot(x, history['train_loss'], label='Train', marker='o', ms=3)
        axes[0].plot(x, history['val_loss'],   label='Val',   marker='s', ms=3)
        axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
        axes[0].set_title('CORAL + Focal Loss')
        axes[0].legend(); axes[0].grid(True, alpha=0.3)

        axes[1].plot(x, history['train_acc'], label='Train', marker='o', ms=3)
        axes[1].plot(x, history['val_acc'],   label='Val',   marker='s', ms=3)
        axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Accuracy (%)')
        axes[1].set_title('Accuracy')
        axes[1].legend(); axes[1].grid(True, alpha=0.3)

        axes[2].plot(x, history['val_qwk'], label='Val QWK',
                     marker='o', ms=3, color='green')
        axes[2].axhline(best_qwk, color='r', ls='--',
                        label=f'Best {best_qwk:.4f} (ep {best_epoch+1})')
        axes[2].axhline(ensemble_qwk, color='blue', ls=':',
                        label=f'Ensemble {ensemble_qwk:.4f}')
        axes[2].set_xlabel('Epoch'); axes[2].set_ylabel('QWK')
        axes[2].set_title('Quadratic Weighted Kappa')
        axes[2].legend(); axes[2].grid(True, alpha=0.3)

        if epochs_ran < 30:
            fig.suptitle(
                f'Training curves  --  {epochs_ran}/30 epochs completed',
                fontsize=10, color='darkorange')

        plt.tight_layout()
        plt.savefig(os.path.join(exp_dir, 'training_curves.png'),
                    dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Training curves saved  ({epochs_ran} epochs).")
    else:
        print("No history.json found -- training curves skipped."
              " Re-run training to generate them.")

    # Grad-CAM (load best checkpoint)
    if not args.no_gradcam:
        print("\nRunning Grad-CAM analysis ...")
        model.load_state_dict(
            torch.load(str(ckpt_paths[0]), map_location=device, weights_only=True))
        run_gradcam(model, val_dataset, final_labels, final_preds,
                    device, exp_dir, n_samples=args.gradcam_samples)

    # Confusion matrix
    cm = confusion_matrix(final_labels, final_preds)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['G0','G1','G2','G3','G4'],
                yticklabels=['G0','G1','G2','G3','G4'])
    plt.title(f'Confusion matrix  --  ensemble QWK = {ensemble_qwk:.4f}')
    plt.ylabel('True'); plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(os.path.join(exp_dir, 'confusion_matrix.png'), dpi=150)
    plt.close()
    print(f"Confusion matrix saved.")

    # Classification report
    grade_names = ['Grade 0 (No DR)', 'Grade 1 (Mild)',
                   'Grade 2 (Moderate)', 'Grade 3 (Severe)',
                   'Grade 4 (Proliferative)']
    report_str = classification_report(
        final_labels, final_preds, target_names=grade_names)
    val_dist = val_df['diagnosis'].value_counts().sort_index()

    report_full = (
        f"Classification Report -- {args.exp}\n"
        "=" * 60 + "\n"
        "NOTE: 'support' shows the true validation-set count per class\n"
        "(20% stratified split, random_state=42). It is fixed across\n"
        "experiments and reflects the real-world class imbalance.\n"
        "Augmentation stats are in summary.json -> balanced_train_dist.\n"
        "=" * 60 + "\n\n"
        + report_str
    )

    with open(os.path.join(exp_dir, 'classification_report.txt'), 'w') as f:
        f.write(report_full)
    print("\n" + report_full)

    # Summary JSON
    class_report_dict = classification_report(
        final_labels, final_preds,
        target_names=grade_names, output_dict=True)

    summary = {
        'experiment':       args.exp,
        'checkpoints_used': [p.name for p in ckpt_paths],
        'ensemble_qwk':     float(ensemble_qwk),
        'best_model_path':  best_model_path,
        'val_dist':         {str(g): int(n) for g, n in val_dist.items()},
        'per_class_metrics': {
            str(i): {
                'precision': float(class_report_dict[grade_names[i]]['precision']),
                'recall':    float(class_report_dict[grade_names[i]]['recall']),
                'f1-score':  float(class_report_dict[grade_names[i]]['f1-score']),
                'support':   int(class_report_dict[grade_names[i]]['support']),
            } for i in range(5)
        },
    }

    # Merge with existing summary.json if present
    existing = os.path.join(exp_dir, 'summary.json')
    if os.path.exists(existing):
        with open(existing) as f:
            old = json.load(f)
        old.update(summary)
        summary = old

    with open(existing if os.path.exists(existing)
              else os.path.join(exp_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"summary.json saved.")
    print(f"\nDone. Ensemble QWK = {ensemble_qwk:.4f}")


if __name__ == '__main__':
    main()
