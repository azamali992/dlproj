"""
train_coral_focal_v6.py
=======================
Re-run of exp_v5 with:

1. New preprocessing cache  (data/processed/hybrid_224)
   - pad_to_square preserves aspect ratio (fixes oval fundus)
   - Pixel-wise fundus mask blacks out background exactly
   - BORDER_REPLICATE padding prevents Ben Graham halos
   - HSV-CLAHE (V channel only) -- uniform colour matching old pipeline

2. Grade-aware augmentation without CoarseDropout (preprocess_grade_aware_aug v2)
   - Mild  -> Grade 0/1 : geometry + tiny colour jitter, no blur
   - Moderate -> Grade 2 : fuller geometry, moderate colour jitter, no blur
   - Strong -> Grade 3/4 : full geometry + colour + 3-px Gaussian blur

3. Grad-CAM analysis after training
   For each DR grade: show the boundary-specific saliency map overlaid
   on 3 correctly-classified validation images.

4. Augmentation visualisation saved from real training batches
   (so the saved samples reflect what the model actually sees, not just
   verify_augmentation output)

5. Enhanced summary with both training (augmented) and validation class
   distributions, so it is clear augmentation took effect.

All model/loss/training decisions identical to exp_v5:
- DenseNet121 + CORALModule_v5 (4 independent projections)
- Alpha computed on ORIGINAL imbalanced labels
- Ordinal consistency regularisation lambda_ord=0.1
- Freeze denseblock1 ONLY
- OneCycleLR, grad clip 1.0, TTA, top-3 ensemble
"""

import os, json, heapq, warnings
import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import (cohen_kappa_score, confusion_matrix,
                             classification_report)
from sklearn.model_selection import train_test_split
from pathlib import Path
warnings.filterwarnings('ignore')

from preprocess_grade_aware_aug import (
    preprocess_and_cache,
    create_balanced_train_dataframe,
    get_train_augmentation_mild,
    get_train_augmentation_moderate,
    get_train_augmentation_strong,
    get_val_augmentation,
    IMAGENET_MEAN,
    IMAGENET_STD,
)
from utils.losses_v5 import CORALModule_v5
from utils.models_v5 import DenseNet121withCORALFocal_v5, freeze_early_layers


# -- CONFIG --------------------------------------------------------------------
CONFIG = {
    'model_name':           'densenet121_coral_v6',
    'num_classes':          5,
    'input_size':           224,
    'batch_size':           16,
    'num_epochs':           30,
    'learning_rate':        3e-4,
    'weight_decay':         1e-3,
    'device':               'cuda' if torch.cuda.is_available() else 'cpu',
    'data_dir':             './data/raw',
    'output_dir':           os.path.join('results', 'exp_v6'),
    'preprocessing_method': 'hybrid',
    'gamma':                2.0,
    'lambda_ord':           0.1,
    'pretrained':           True,
    'cache_dir':            os.path.join('data', 'processed', 'hybrid_224'),
    'grad_clip':            1.0,
    'top_k_checkpoints':    3,
    'tta':                  True,
    'target_per_class':     800,
    'gradcam_samples':      3,   # val images per grade for Grad-CAM
}

os.makedirs(CONFIG['output_dir'], exist_ok=True)


# -- DATASET -------------------------------------------------------------------
class PreprocessedDRDataset(Dataset):
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

    def _pick_aug(self, grade):
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
        image    = cv2.imread(img_path)
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
            mean  = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
            std   = torch.tensor(IMAGENET_STD).view(3, 1, 1)
            image = (image - mean) / std
        return image, label


# -- HELPERS -------------------------------------------------------------------
def _tta_flip(batch):
    return [batch,
            torch.flip(batch, dims=[3]),
            torch.flip(batch, dims=[2])]


def tensor_to_uint8(t):
    arr = t.cpu().numpy()
    if arr.ndim == 3:  # CHW
        arr = arr.transpose(1, 2, 0)
    arr = arr * np.array(IMAGENET_STD) + np.array(IMAGENET_MEAN)
    return np.clip(arr * 255, 0, 255).astype(np.uint8)


# -- TRAINING / VALIDATION -----------------------------------------------------
def train_epoch(model, loader, optimizer, scheduler, device, epoch, total):
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    for i, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)
        logits        = model(images)
        coral_targets = model.coral_label_transform(labels)
        loss          = model.loss(logits, coral_targets)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            filter(lambda p: p.requires_grad, model.parameters()),
            CONFIG['grad_clip'])
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
        preds = model.predict(logits.detach())
        n     += labels.size(0)
        correct += (preds == labels).sum().item()
        if (i + 1) % 50 == 0:
            print(f"  Epoch [{epoch+1}/{total}] "
                  f"Batch [{i+1}/{len(loader)}] "
                  f"Loss={loss.item():.4f} LR={scheduler.get_last_lr()[0]:.2e}")
    return total_loss / len(loader), 100.0 * correct / n


def validate(model, loader, device, use_tta=False):
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    preds_all, labels_all  = [], []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if use_tta:
                probs = None
                for aug in _tta_flip(images):
                    p    = torch.sigmoid(model(aug))
                    probs = p if probs is None else probs + p
                probs   /= 3
                clamped  = probs.clamp(1e-6, 1 - 1e-6)
                logits   = torch.log(clamped / (1 - clamped))
                preds    = (probs > 0.5).sum(dim=1)
            else:
                logits = model(images)
                preds  = model.predict(logits)
            coral_targets = model.coral_label_transform(labels)
            total_loss += model.loss(logits, coral_targets).item()
            n          += labels.size(0)
            correct    += (preds == labels).sum().item()
            preds_all.extend(preds.cpu().numpy())
            labels_all.extend(labels.cpu().numpy())
    qwk = cohen_kappa_score(labels_all, preds_all, weights='quadratic')
    return total_loss / len(loader), 100.0 * correct / n, qwk, preds_all, labels_all


# -- TOP-K CHECKPOINTS ---------------------------------------------------------
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
            _, worst = heapq.heappop(self._heap)
            p = Path(worst)
            if p.exists():
                p.unlink()
        print(f"  Saved checkpoint: {path.name}")

    def best_paths(self):
        return [p for _, p in sorted(self._heap, reverse=True)]


def ensemble_from_checkpoints(model, ckpt_paths, loader, device):
    print(f"\nEnsembling {len(ckpt_paths)} checkpoints ...")
    all_probs, all_labels = None, []
    for ckpt in ckpt_paths:
        model.load_state_dict(torch.load(ckpt, map_location=device,
                                         weights_only=True))
        model.eval()
        batch_probs = []
        with torch.no_grad():
            for images, labels in loader:
                probs = torch.sigmoid(model(images.to(device))).cpu()
                batch_probs.append(probs)
                if not all_labels:
                    all_labels.extend(labels.numpy())
        ep = torch.cat(batch_probs)
        all_probs = ep if all_probs is None else all_probs + ep
    all_probs /= len(ckpt_paths)
    preds = (all_probs > 0.5).sum(dim=1).numpy()
    qwk   = cohen_kappa_score(all_labels, preds, weights='quadratic')
    print(f"Ensemble QWK ({len(ckpt_paths)} models): {qwk:.4f}")
    return qwk, preds, all_labels


# -- AUG VISUALISATION (from real training batches) ----------------------------
def save_aug_samples(train_dataset, output_dir, n_per_grade=2):
    """
    Pull real augmented images from the training dataset and save them
    alongside their pre-aug cached originals.  These reflect what the model
    actually sees -- not verify_augmentation() output.
    """
    save_dir = os.path.join(output_dir, 'aug_samples')
    os.makedirs(save_dir, exist_ok=True)

    grade_labels = {0: 'G0_NoDR', 1: 'G1_Mild', 2: 'G2_Moderate',
                    3: 'G3_Severe', 4: 'G4_Proliferative'}
    aug_strength = {0: 'mild', 1: 'mild', 2: 'moderate', 3: 'strong', 4: 'strong'}

    collected = {g: [] for g in range(5)}
    for idx in range(len(train_dataset)):
        row   = train_dataset.labels_df.iloc[idx]
        grade = int(row['diagnosis'])
        if len(collected[grade]) >= n_per_grade:
            continue

        # Load original cached image
        img_path = os.path.join(CONFIG['cache_dir'], f"{row['id_code']}.png")
        orig_bgr = cv2.imread(img_path)
        if orig_bgr is None:
            continue
        orig_rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)

        # Get augmented tensor
        aug_tensor, _ = train_dataset[idx]
        aug_rgb       = tensor_to_uint8(aug_tensor)

        collected[grade].append((orig_rgb, aug_rgb))
        if all(len(v) >= n_per_grade for v in collected.values()):
            break

    fig, axes = plt.subplots(5, n_per_grade * 2,
                             figsize=(n_per_grade * 4, 5 * 2.5))
    for g in range(5):
        for k, (orig, aug) in enumerate(collected[g][:n_per_grade]):
            c = k * 2
            axes[g, c].imshow(orig)
            axes[g, c].set_title(f'{grade_labels[g]}\noriginal', fontsize=7)
            axes[g, c].axis('off')
            axes[g, c + 1].imshow(aug)
            axes[g, c + 1].set_title(f'{aug_strength[g]} aug', fontsize=7)
            axes[g, c + 1].axis('off')

    fig.suptitle('Training images -- original (cached) vs augmented (model input)',
                 fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_aug_grid.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Aug samples saved to {save_dir}/training_aug_grid.png")


# -- GRAD-CAM ------------------------------------------------------------------
def run_gradcam_analysis(model, val_dataset, val_labels, val_preds, device, output_dir):
    """
    For each DR grade, find up to CONFIG['gradcam_samples'] correctly-classified
    validation images and generate Grad-CAM using the most discriminative
    CORAL threshold for that grade.

    Saves: results/exp_v6/gradcam/grade_N/img_K_{orig|heatmap|overlay}.png
           results/exp_v6/gradcam/gradcam_grid.png
    """
    gradcam_dir = os.path.join(output_dir, 'gradcam')
    os.makedirs(gradcam_dir, exist_ok=True)

    # CORAL threshold most discriminative for each grade
    # threshold_k = max(0, grade - 1)  (built into generate_gradcam default)
    grade_names = {0: 'G0_NoHR', 1: 'G1_Mild', 2: 'G2_Moderate',
                   3: 'G3_Severe', 4: 'G4_Proliferative'}
    boundary_desc = {0: 'T0: No DR -> any DR',
                     1: 'T0: No DR -> any DR',
                     2: 'T1: Mild -> Moderate',
                     3: 'T2: Moderate -> Severe',
                     4: 'T3: Severe -> Proliferative'}

    n_samples = CONFIG['gradcam_samples']
    model.eval()

    all_rows = []   # (grade, orig_img, heatmap, overlay, label_str)

    for grade in range(5):
        grade_out = os.path.join(gradcam_dir, grade_names[grade])
        os.makedirs(grade_out, exist_ok=True)

        # Collect correctly-classified validation indices for this grade
        correct_idx = [i for i, (t, p) in enumerate(zip(val_labels, val_preds))
                       if t == grade and p == grade][:n_samples]

        if not correct_idx:
            # Fall back to any images of this grade
            correct_idx = [i for i, t in enumerate(val_labels)
                           if t == grade][:n_samples]

        row_imgs = []
        for k, idx in enumerate(correct_idx):
            image_tensor, label = val_dataset[idx]
            image_tensor = image_tensor.unsqueeze(0).to(device)

            # Forward pass WITH gradients for Grad-CAM
            logits = model(image_tensor)

            cam = model.generate_gradcam(logits, target_class=grade)
            cam_np = cam.squeeze().cpu().numpy()
            cam_np = cv2.resize(cam_np, (224, 224))

            orig_np = tensor_to_uint8(image_tensor.squeeze(0))

            heatmap = cv2.applyColorMap(
                (cam_np * 255).astype(np.uint8), cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

            overlay = np.clip(
                0.55 * orig_np.astype(float) + 0.45 * heatmap.astype(float),
                0, 255).astype(np.uint8)

            # Save individual files
            for name, img in [('orig', orig_np),
                               ('heatmap', heatmap),
                               ('overlay', overlay)]:
                cv2.imwrite(
                    os.path.join(grade_out, f'img{k}_{name}.png'),
                    cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

            row_imgs.append((orig_np, heatmap, overlay))

        all_rows.append((grade, row_imgs))

    # -- Summary grid: rows = grades, cols = orig / heatmap / overlay --------
    n_rows = 5
    n_cols = n_samples * 3
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 2.0, n_rows * 2.2))

    col_header = []
    for k in range(n_samples):
        col_header += [f'sample {k+1}\norig',
                       f'sample {k+1}\nheatmap',
                       f'sample {k+1}\noverlay']

    for r, (grade, row_imgs) in enumerate(all_rows):
        for k, (orig, heat, over) in enumerate(row_imgs):
            for c_off, img in enumerate([orig, heat, over]):
                ax = axes[r, k * 3 + c_off]
                ax.imshow(img)
                ax.axis('off')
                if r == 0:
                    ax.set_title(col_header[k * 3 + c_off], fontsize=6)
        axes[r, 0].set_ylabel(
            f'{grade_names[grade]}\n{boundary_desc[grade]}',
            fontsize=7, rotation=0, labelpad=60, va='center')

    fig.suptitle('Grad-CAM -- boundary-specific saliency per DR grade\n'
                 'Heatmap = where the model looks to decide the relevant CORAL threshold',
                 fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(gradcam_dir, 'gradcam_grid.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Grad-CAM grid saved to {gradcam_dir}/gradcam_grid.png")


# -- MAIN ----------------------------------------------------------------------
def main():
    print("=" * 80)
    print("TRAINING v6: new preprocessing + grade-aware aug + Grad-CAM")
    print("=" * 80)
    print(f"Device     : {CONFIG['device']}")
    print(f"Cache dir  : {CONFIG['cache_dir']}")
    print(f"Output dir : {CONFIG['output_dir']}")
    print()

    # -- 1. Data split --------------------------------------------------------
    full_df  = pd.read_csv(os.path.join(CONFIG['data_dir'], 'train.csv'))
    train_df, val_df = train_test_split(
        full_df, test_size=0.2, random_state=42,
        stratify=full_df['diagnosis'])

    print("Original training class distribution:")
    orig_dist = train_df['diagnosis'].value_counts().sort_index()
    for g, n in orig_dist.items():
        print(f"  Grade {g}: {n:4d}")

    # -- 2. Verify cache exists -----------------------------------------------
    cache = Path(CONFIG['cache_dir'])
    n_cached = len(list(cache.glob('*.png')))
    print(f"\nCache: {CONFIG['cache_dir']}  ({n_cached} images)")
    if n_cached < len(full_df):
        print("Cache incomplete -- running preprocess_and_cache ...")
        from preprocess_grade_aware_aug import preprocess_and_cache
        preprocess_and_cache(
            raw_dir=os.path.join(CONFIG['data_dir'], 'train_images'),
            cache_dir=CONFIG['cache_dir'],
            df=full_df,
            method=CONFIG['preprocessing_method'],
            image_size=CONFIG['input_size'],
            verbose=True,
        )

    # -- 3. Balance training set ----------------------------------------------
    print("\nBalancing training set ...")
    train_df_balanced = create_balanced_train_dataframe(
        train_df=train_df,
        strategy='oversample',
        target_per_class=CONFIG['target_per_class'],
    )
    bal_dist = train_df_balanced['diagnosis'].value_counts().sort_index()
    print("Balanced training distribution:")
    for g, n in bal_dist.items():
        print(f"  Grade {g}: {n:4d}")

    val_dist = val_df['diagnosis'].value_counts().sort_index()
    print("\nValidation distribution (unchanged -- reflects real-world imbalance):")
    for g, n in val_dist.items():
        print(f"  Grade {g}: {n:4d}")

    # -- 4. Datasets & loaders ------------------------------------------------
    mild_aug     = get_train_augmentation_mild()
    moderate_aug = get_train_augmentation_moderate()
    strong_aug   = get_train_augmentation_strong()
    val_aug      = get_val_augmentation()

    train_dataset = PreprocessedDRDataset(
        CONFIG['cache_dir'], train_df_balanced,
        mild_aug=mild_aug, moderate_aug=moderate_aug,
        strong_aug=strong_aug, val_aug=val_aug, is_train=True)
    val_dataset = PreprocessedDRDataset(
        CONFIG['cache_dir'], val_df,
        mild_aug=mild_aug, moderate_aug=moderate_aug,
        strong_aug=strong_aug, val_aug=val_aug, is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'],
                              shuffle=True, num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=CONFIG['batch_size'],
                              shuffle=False, num_workers=0, pin_memory=True)

    # -- 5. Save aug samples from real training batches -----------------------
    print("\nSaving augmentation samples from training dataset ...")
    save_aug_samples(train_dataset, CONFIG['output_dir'])

    # -- 6. Model -------------------------------------------------------------
    print("\nInitialising model ...")
    model = DenseNet121withCORALFocal_v5(
        num_classes=CONFIG['num_classes'],
        gamma=CONFIG['gamma'],
        pretrained=CONFIG['pretrained'],
        lambda_ord=CONFIG['lambda_ord'],
    ).to(CONFIG['device'])

    freeze_early_layers(model, blocks=('denseblock1',))

    # Alpha on ORIGINAL imbalanced labels (key fix from v5)
    print("\nComputing alpha on original imbalanced labels ...")
    model.compute_alpha(
        torch.tensor(train_df['diagnosis'].values, dtype=torch.long))

    # -- 7. Optimiser + scheduler ----------------------------------------------
    optimizer   = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=CONFIG['learning_rate'] / 25,
        weight_decay=CONFIG['weight_decay'])
    total_steps = len(train_loader) * CONFIG['num_epochs']
    scheduler   = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=CONFIG['learning_rate'],
        total_steps=total_steps,
        pct_start=0.1,
        anneal_strategy='cos',
        div_factor=25,
        final_div_factor=1e4)

    print(f"\nOneCycleLR: {total_steps} steps, peak LR={CONFIG['learning_rate']:.1e}")

    # -- 8. Training loop -----------------------------------------------------
    top_k   = TopKCheckpoints(CONFIG['top_k_checkpoints'], CONFIG['output_dir'])
    history = {k: [] for k in ('train_loss', 'train_acc', 'val_loss',
                                'val_acc', 'val_qwk')}
    best_qwk, best_epoch, best_preds, best_labels = 0.0, 0, None, None

    print("\n" + "=" * 80)
    print("TRAINING")
    print("=" * 80)

    for epoch in range(CONFIG['num_epochs']):
        tr_loss, tr_acc = train_epoch(
            model, train_loader, optimizer, scheduler,
            CONFIG['device'], epoch, CONFIG['num_epochs'])
        vl_loss, vl_acc, vl_qwk, vl_preds, vl_labels = validate(
            model, val_loader, CONFIG['device'], use_tta=CONFIG['tta'])

        history['train_loss'].append(tr_loss)
        history['train_acc'].append(tr_acc)
        history['val_loss'].append(vl_loss)
        history['val_acc'].append(vl_acc)
        history['val_qwk'].append(vl_qwk)

        # Save after every epoch so training curves survive an interruption
        with open(os.path.join(CONFIG['output_dir'], 'history.json'), 'w') as _f:
            json.dump(history, _f)

        print(f"\nEpoch {epoch+1}/{CONFIG['num_epochs']}  "
              f"train loss={tr_loss:.4f} acc={tr_acc:.1f}%  "
              f"val loss={vl_loss:.4f} acc={vl_acc:.1f}% QWK={vl_qwk:.4f}")

        if vl_qwk > best_qwk:
            best_qwk, best_epoch = vl_qwk, epoch
            best_preds, best_labels = vl_preds, vl_labels
            print(f"  *** New best QWK: {vl_qwk:.4f}")

        top_k.update(vl_qwk, epoch, model)

    # -- 9. CORAL diagnostics -------------------------------------------------
    biases = [layer.bias.data.item()
              for layer in model.model.classifier.fc]
    monotonic = all(biases[k] >= biases[k+1] for k in range(len(biases)-1))
    print(f"\nCORAL biases    : {[round(b,4) for b in biases]}")
    print(f"Monotonic       : {'YES' if monotonic else 'NO'}")

    # -- 10. Ensemble ----------------------------------------------------------
    print("\n" + "=" * 80)
    print("ENSEMBLE")
    best_ckpt_paths = top_k.best_paths()

    # Save the single best checkpoint as best_model.pth for future use
    import shutil
    best_model_path = os.path.join(CONFIG['output_dir'], 'best_model.pth')
    shutil.copy(best_ckpt_paths[0], best_model_path)
    print(f"Best model saved: {best_model_path}  (QWK={best_qwk:.4f})")

    ensemble_qwk    = best_qwk
    if len(best_ckpt_paths) > 1:
        ensemble_qwk, ensemble_preds, ensemble_labels = ensemble_from_checkpoints(
            model, best_ckpt_paths, val_loader, CONFIG['device'])
        final_preds, final_labels = ensemble_preds, ensemble_labels
    else:
        final_preds, final_labels = best_preds, best_labels

    # -- 11. Grad-CAM (load best single model) --------------------------------
    print("\n" + "=" * 80)
    print("GRAD-CAM ANALYSIS")
    model.load_state_dict(
        torch.load(best_ckpt_paths[0], map_location=CONFIG['device'],
                   weights_only=True))
    run_gradcam_analysis(
        model, val_dataset, best_labels, best_preds,
        CONFIG['device'], CONFIG['output_dir'])

    # -- 12. Plots -------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, key, ylabel in [
        (axes[0], ('train_loss', 'val_loss'), 'Loss'),
        (axes[1], ('train_acc',  'val_acc'),  'Accuracy (%)'),
        (axes[2], ('val_qwk',),              'QWK'),
    ]:
        for k in key:
            ax.plot(history[k], label=k.replace('_', ' '), marker='o', ms=3)
        if ylabel == 'QWK':
            ax.axhline(best_qwk, color='r', ls='--',
                       label=f'best {best_qwk:.4f}')
        ax.set_xlabel('Epoch'); ax.set_ylabel(ylabel)
        ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG['output_dir'], 'training_curves.png'), dpi=150)
    plt.close()

    cm = confusion_matrix(final_labels, final_preds)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['G0','G1','G2','G3','G4'],
                yticklabels=['G0','G1','G2','G3','G4'])
    plt.title(f'Confusion matrix (ensemble QWK={ensemble_qwk:.4f})')
    plt.ylabel('True'); plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG['output_dir'], 'confusion_matrix.png'), dpi=150)
    plt.close()

    # -- 13. Classification report ---------------------------------------------
    grade_names = ['Grade 0 (No DR)', 'Grade 1 (Mild)',
                   'Grade 2 (Moderate)', 'Grade 3 (Severe)',
                   'Grade 4 (Proliferative)']
    report_str = classification_report(
        final_labels, final_preds, target_names=grade_names)

    report_full = (
        "Classification Report -- exp_v6\n"
        "=" * 60 + "\n"
        "NOTE: 'support' shows the true validation-set count per class\n"
        "(20% stratified split, random_state=42). It is fixed across\n"
        "experiments and reflects the real-world class imbalance.\n"
        "=" * 60 + "\n\n"
        + report_str
        + "\nTraining distribution after oversampling (see summary.json):\n"
    )
    for g, n in bal_dist.items():
        aug = 'mild' if g <= 1 else ('moderate' if g == 2 else 'strong')
        report_full += f"  Grade {g}: {n:5d} samples  [{aug} aug]\n"

    with open(os.path.join(CONFIG['output_dir'], 'classification_report.txt'), 'w') as f:
        f.write(report_full)
    print("\nClassification Report:")
    print(report_full)

    # -- 14. Summary JSON ------------------------------------------------------
    class_report_dict = classification_report(
        final_labels, final_preds, target_names=grade_names,
        output_dict=True)

    summary = {
        'model':          'DenseNet121 + independent CORAL projections (v6)',
        'experiment':     'exp_v6',
        'changes_vs_v5':  [
            'New preprocessing: pad_to_square + pixel-wise mask + HSV-CLAHE',
            'Ben Graham sees BORDER_REPLICATE padding -- no halo artefacts',
            'CoarseDropout removed from strong augmentation',
            'Grad-CAM analysis included in results',
        ],
        'best_epoch':     best_epoch + 1,
        'best_qwk':       float(best_qwk),
        'ensemble_qwk':   float(ensemble_qwk),
        'coral_biases':   [round(b, 4) for b in biases],
        'coral_monotonic': monotonic,
        'alpha_weights':  [round(a, 4) for a in
                           model.model.classifier.alpha.tolist()],
        'original_train_dist':  {str(g): int(n) for g, n in orig_dist.items()},
        'balanced_train_dist':  {str(g): int(n) for g, n in bal_dist.items()},
        'val_dist':             {str(g): int(n) for g, n in val_dist.items()},
        'per_class_metrics': {
            str(i): {
                'precision': float(class_report_dict[grade_names[i]]['precision']),
                'recall':    float(class_report_dict[grade_names[i]]['recall']),
                'f1-score':  float(class_report_dict[grade_names[i]]['f1-score']),
                'support':   int(class_report_dict[grade_names[i]]['support']),
            } for i in range(5)
        },
        'best_model_path': best_model_path,
        'config': {k: str(v) if isinstance(v, Path) else v
                   for k, v in CONFIG.items()},
    }
    with open(os.path.join(CONFIG['output_dir'], 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)
    print(f"  Best single QWK : {best_qwk:.4f}  (epoch {best_epoch+1})")
    print(f"  Ensemble QWK    : {ensemble_qwk:.4f}")
    print(f"  CORAL monotonic : {'YES' if monotonic else 'NO'}")
    for i in range(5):
        r = class_report_dict[grade_names[i]]['recall']
        print(f"  Grade {i} recall  : {r:.1%}")
    print(f"\nResults saved to: {CONFIG['output_dir']}/")


if __name__ == '__main__':
    main()
