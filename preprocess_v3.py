"""
preprocess_v3.py — APTOS 2019 Blindness Detection preprocessing pipeline
=========================================================================

Research-backed pipeline. Replaces v2 which had several bugs (wrong Ben
Graham formula, color-destroying z-score normalization, no fundus crop,
HSV-CLAHE instead of the literature-standard LAB-CLAHE, 224x224 too small
to preserve microaneurysms).

Pipeline order (each step is documented with its source):

    raw RGB image
        │
        ▼
    1. crop_fundus()        — remove black border around the retina
        │                     (Biomedical & Pharmacology J., 2017;
        │                      every APTOS paper does this first)
        ▼
    2. resize INTER_LANCZOS4 — preserves small lesion detail better
        │                      than INTER_LINEAR (V5.2 working code)
        ▼
    3. ben_graham()         — Graham, Kaggle DR 2015 winner.
        │                     `cv2.addWeighted(img,4,blur,-4,128)`
        │                     Subtracts local mean colour, maps to mid grey.
        │                     Macsik et al. (IET Image Proc., 2024);
        │                     DR-NASNet (MDPI Diagnostics, 2023).
        ▼
    4. clahe_lab()          — CLAHE on L channel of CIELAB.
        │                     Preserves colour, normalises perceived
        │                     contrast. Macsik et al. 2024;
        │                     filipmu Kaggle reference impl.
        ▼
    5. apply_circle_mask()  — zero out corners outside the FOV so the
                              network never sees border artefacts.

This is the same pipeline as your friend's V5.2 (which reached raw QWK
0.85). The augmentation block is also taken from V5.2 with the broken
albumentations 2.x calls fixed.

Inputs / outputs
----------------
    data/raw/train_images/*.png      (original APTOS files)
    data/raw/train.csv               (id_code, diagnosis)

    data/processed/train_images/*.png   (preprocessed, on disk)
    data/processed/train_balanced.csv   (oversampled label list)

Usage
-----
    from preprocess_v3 import (
        preprocess_and_cache,
        create_balanced_train_dataframe,
        get_train_augmentation_mild,
        get_train_augmentation_strong,
        get_val_augmentation,
        verify_preprocessing,
        verify_augmentation,
    )

    # one-time: preprocess the whole train set to data/processed/
    preprocess_and_cache(
        raw_dir='data/raw/train_images',
        cache_dir='data/processed/train_images',
        df=pd.read_csv('data/raw/train.csv'),
        method='hybrid',
        image_size=512,
    )

    # build a balanced label CSV for the sampler
    bal_df = create_balanced_train_dataframe(
        train_df=pd.read_csv('data/raw/train.csv'),
        strategy='oversample',
        target_per_class=1000,
    )
    bal_df.to_csv('data/processed/train_balanced.csv', index=False)

    # in your Dataset.__getitem__, load from data/processed/ and apply:
    #   img = get_train_augmentation_strong()(image=img)['image']
"""

from __future__ import annotations

import os
import cv2
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path
from typing import Optional


# ============================================================================
#  DEFAULT PATHS — edit if your layout differs
# ============================================================================

DEFAULT_RAW_DIR        = 'data/raw/train_images'
DEFAULT_RAW_CSV        = 'data/raw/train.csv'
DEFAULT_PROCESSED_DIR  = 'data/processed/train_images'
DEFAULT_BALANCED_CSV   = 'data/processed/train_balanced.csv'

DEFAULT_IMAGE_SIZE     = 512    # 512 strongly recommended for grading.
                                # Use 384 if VRAM-limited, never below 256.

# ImageNet stats for transfer-learning normalisation (ResNet/EffNet/ViT)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ============================================================================
#  CORE PREPROCESSING FUNCTIONS
#  Each is small and unit-testable on its own.
# ============================================================================

def crop_fundus(img: np.ndarray, threshold: int = 7) -> np.ndarray:
    """
    Crop the black background around the circular retina.

    Uses the green channel (which has the strongest fundus signal — see
    Stage-Aware DR ordinal regression paper, arXiv 2511.14398, 2025) and
    finds the tight bounding box around pixels brighter than `threshold`.

    If the image is somehow all dark, returns it unchanged.
    """
    if img.ndim != 3:
        return img
    gray = img[:, :, 1]                       # green channel
    mask = gray > threshold
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any() or not cols.any():
        return img
    r0, r1 = np.where(rows)[0][[0, -1]]
    c0, c1 = np.where(cols)[0][[0, -1]]
    return img[r0:r1 + 1, c0:c1 + 1]


def make_circle_mask(h: int, w: int) -> np.ndarray:
    """Boolean mask True inside the inscribed circle (the camera FOV)."""
    cy, cx = h / 2.0, w / 2.0
    r = min(cy, cx) - 1
    Y, X = np.ogrid[:h, :w]
    return (X - cx) ** 2 + (Y - cy) ** 2 <= r ** 2


def apply_circle_mask(img: np.ndarray, fill: int = 0) -> np.ndarray:
    """Zero out pixels outside the circular FOV. Removes corner artefacts."""
    h, w = img.shape[:2]
    out = img.copy()
    out[~make_circle_mask(h, w)] = fill
    return out


def ben_graham(img: np.ndarray, sigmaX: int = 10) -> np.ndarray:
    """
    Ben Graham's contrast enhancement (Kaggle DR 2015 competition winner).

    `enhanced = 4*img - 4*Gaussian(img, sigma) + 128`

    This subtracts the local mean colour and re-centres at mid grey,
    which standardises lighting across cameras and amplifies fine-grained
    lesions (microaneurysms, hard exudates).

    Reference implementation: every paper in the lit review uses this exact
    formula — see DR-NASNet (Diagnostics 2023), Macsik et al. (IET Image
    Processing 2024), Sensors 2023 (PMC10301863).
    """
    blurred = cv2.GaussianBlur(img, (0, 0), sigmaX)
    return cv2.addWeighted(img, 4, blurred, -4, 128)


def clahe_lab(img: np.ndarray, clip_limit: float = 2.0,
              tile: int = 8) -> np.ndarray:
    """
    CLAHE on the L channel of CIELAB colour space.

    Why LAB and not HSV/RGB:
    - L is perceptual lightness (roughly luminance), so CLAHE on L behaves
      like CLAHE on a grayscale image while a/b channels keep colour intact.
    - CIELAB was designed so equal numerical changes ≈ equal perceived
      changes, making the contrast boost look natural rather than garish.
    - Per Macsik et al. (IET Image Processing, 2024), Lab-CLAHE was the
      single best-performing CLAHE variant on APTOS 2019 in their ensemble
      ablation.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile, tile))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB)


def preprocess_full(img_bgr: np.ndarray, size: int = DEFAULT_IMAGE_SIZE,
                    sigmaX: int = 10, clahe_clip: float = 2.0) -> np.ndarray:
    """
    The full pipeline:  crop → resize → Ben Graham → LAB-CLAHE → circle mask.
    Returns RGB uint8 in (size, size, 3).
    """
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img = crop_fundus(img)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LANCZOS4)
    img = ben_graham(img, sigmaX=sigmaX)
    img = np.clip(img, 0, 255).astype(np.uint8)
    img = clahe_lab(img, clip_limit=clahe_clip)
    img = apply_circle_mask(img)
    return img


# ============================================================================
#  PREPROCESSOR CLASSES — drop-in replacements for v2 classes
# ============================================================================

class BenGrahamPreprocessor:
    """Crop + resize + correct Ben Graham + circle mask. No CLAHE."""

    def __init__(self, image_size: int = DEFAULT_IMAGE_SIZE, sigmaX: int = 10):
        self.image_size = image_size
        self.sigmaX = sigmaX

    def process(self, image_path: str) -> np.ndarray:
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            raise FileNotFoundError(f'cv2 failed to read: {image_path}')
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img = crop_fundus(img)
        img = cv2.resize(img, (self.image_size, self.image_size),
                         interpolation=cv2.INTER_LANCZOS4)
        img = ben_graham(img, sigmaX=self.sigmaX)
        img = np.clip(img, 0, 255).astype(np.uint8)
        return apply_circle_mask(img)


class CLAHEPreprocessor:
    """Crop + resize + LAB-CLAHE + circle mask. No Ben Graham."""

    def __init__(self, image_size: int = DEFAULT_IMAGE_SIZE,
                 clip_limit: float = 2.0, tile_size: int = 8):
        self.image_size = image_size
        self.clip_limit = clip_limit
        self.tile_size  = tile_size

    def process(self, image_path: str) -> np.ndarray:
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            raise FileNotFoundError(f'cv2 failed to read: {image_path}')
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img = crop_fundus(img)
        img = cv2.resize(img, (self.image_size, self.image_size),
                         interpolation=cv2.INTER_LANCZOS4)
        img = clahe_lab(img, clip_limit=self.clip_limit, tile=self.tile_size)
        return apply_circle_mask(img)


class HybridPreprocessor:
    """
    RECOMMENDED. Crop + resize + Ben Graham + LAB-CLAHE + circle mask.

    This is the pipeline used in DR-NASNet (Diagnostics 2023) which reports
    that Ben Graham followed by CLAHE gives better classification than
    either alone, because Ben Graham normalises lighting first, then CLAHE
    refines local contrast on the already-balanced image.
    """

    def __init__(self, image_size: int = DEFAULT_IMAGE_SIZE,
                 sigmaX: int = 10, clip_limit: float = 2.0, tile_size: int = 8):
        self.image_size = image_size
        self.sigmaX     = sigmaX
        self.clip_limit = clip_limit
        self.tile_size  = tile_size

    def process(self, image_path: str) -> np.ndarray:
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            raise FileNotFoundError(f'cv2 failed to read: {image_path}')
        return preprocess_full(img_bgr, size=self.image_size,
                               sigmaX=self.sigmaX,
                               clahe_clip=self.clip_limit)


# ============================================================================
#  AUGMENTATION PIPELINES
#  Two strengths (mild for healthy classes, strong for minority/diseased).
#  All include Normalize + ToTensorV2 so the output is a model-ready tensor.
# ============================================================================

class CircleMaskTransform(A.ImageOnlyTransform):
    """Re-applies the circle mask after geometric transforms shift content."""

    def __init__(self, p: float = 1.0):
        super().__init__(p=p)

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        return apply_circle_mask(img)

    def get_transform_init_args_names(self):
        return ()


def _spatial_block(rotate_p: float = 0.8, scale: float = 0.10):
    """
    Geometric augmentations common to all training augs.

    NB: written to be compatible with albumentations >= 2.0 — the
    `value=` kwarg and `ShiftScaleRotate` are deprecated there. We use
    Affine (their replacement) and the new `fill=` kwarg.
    """
    return [
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Rotate(limit=360, border_mode=cv2.BORDER_CONSTANT, fill=0,
                 p=rotate_p),
        A.Affine(
            translate_percent=(-0.02, 0.02),
            scale=(1.0 - scale, 1.0 + scale),
            rotate=0,
            border_mode=cv2.BORDER_CONSTANT,
            fill=0,
            p=0.5,
        ),
        CircleMaskTransform(p=1.0),
    ]


def _normalize_block():
    return [
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ]


def get_train_augmentation_mild():
    """
    Light augmentation — appropriate for the majority class (No DR) where
    you don't want to risk distorting normal anatomy too much.
    """
    return A.Compose(
        _spatial_block(rotate_p=0.6, scale=0.05) + [
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.10,
                                           contrast_limit=0.10, p=1.0),
                A.HueSaturationValue(hue_shift_limit=8,
                                     sat_shift_limit=12,
                                     val_shift_limit=8, p=1.0),
            ], p=0.5),
            A.GaussianBlur(blur_limit=(3, 3), p=0.2),
            CircleMaskTransform(p=1.0),
        ] + _normalize_block()
    )


def get_train_augmentation_strong():
    """
    Heavy augmentation — for minority classes (Severe, Proliferative) where
    extra variety helps and the lesion-rich images can tolerate more noise.

    Notes vs. v2:
    - NO `p=` at the Compose level (your v2 had p=0.8 which silently skipped
      the whole pipeline 20 % of the time).
    - NO ElasticTransform/GaussNoise with old `var_limit=` API — those raise
      under albumentations >= 2.0. Replaced with the >=2.0-compatible calls.
    - Photometric jitter is wrapped in OneOf so transforms compose more
      realistically (real cameras vary in either lighting OR colour, rarely
      everything at once).
    """
    return A.Compose(
        _spatial_block(rotate_p=0.9, scale=0.10) + [
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.20,
                                           contrast_limit=0.20, p=1.0),
                A.HueSaturationValue(hue_shift_limit=15,
                                     sat_shift_limit=25,
                                     val_shift_limit=15, p=1.0),
                A.RGBShift(r_shift_limit=15, g_shift_limit=15,
                           b_shift_limit=15, p=1.0),
            ], p=0.75),
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                A.Sharpen(alpha=(0.15, 0.40), p=1.0),
                A.MotionBlur(blur_limit=5, p=1.0),
            ], p=0.40),
            A.CoarseDropout(
                num_holes_range=(2, 6),
                hole_height_range=(10, 20),
                hole_width_range=(10, 20),
                fill=0,
                p=0.30,
            ),
            CircleMaskTransform(p=1.0),
        ] + _normalize_block()
    )


def get_val_augmentation():
    """Validation/test transform: only normalise + tensor. NO randomness."""
    return A.Compose(_normalize_block())


# ============================================================================
#  PREPROCESS-AND-CACHE  (data/raw  →  data/processed)
# ============================================================================

def preprocess_and_cache(raw_dir: str = DEFAULT_RAW_DIR,
                         cache_dir: str = DEFAULT_PROCESSED_DIR,
                         df: Optional[pd.DataFrame] = None,
                         csv_path: Optional[str] = None,
                         method: str = 'hybrid',
                         image_size: int = DEFAULT_IMAGE_SIZE,
                         file_ext: str = 'png',
                         skip_existing: bool = True,
                         verbose: bool = True) -> str:
    """
    Apply the chosen preprocessing pipeline to every image listed in `df`
    (or in `csv_path`) and save the result as PNG in `cache_dir`.

    Args
    ----
    raw_dir       : folder of original images, e.g. 'data/raw/train_images'
    cache_dir     : output folder, e.g. 'data/processed/train_images'
    df            : DataFrame with at least an 'id_code' column. If None,
                    `csv_path` is read instead.
    csv_path      : path to a CSV (id_code, diagnosis). Used if df is None.
    method        : 'hybrid' (recommended) | 'ben_graham' | 'clahe'
    image_size    : output side length, default 512
    file_ext      : extension of the raw files (APTOS = 'png')
    skip_existing : if True, files already in cache_dir are not redone
    """
    if df is None:
        if csv_path is None:
            raise ValueError('Provide either df or csv_path')
        df = pd.read_csv(csv_path)

    if 'id_code' not in df.columns:
        # try lowercase fallback
        df.columns = [c.strip().lower() for c in df.columns]
        if 'id_code' not in df.columns:
            raise ValueError("DataFrame must have an 'id_code' column")

    os.makedirs(cache_dir, exist_ok=True)

    factories = {
        'hybrid':     HybridPreprocessor,
        'ben_graham': BenGrahamPreprocessor,
        'clahe':      CLAHEPreprocessor,
    }
    if method not in factories:
        raise ValueError(f"method must be one of {list(factories)}")
    preprocessor = factories[method](image_size=image_size)

    print('=' * 72)
    print(f'Preprocessing & caching  (method={method}, size={image_size})')
    print(f'  raw_dir   : {raw_dir}')
    print(f'  cache_dir : {cache_dir}')
    print(f'  total     : {len(df)} images')
    print('=' * 72)

    successful = 0
    skipped    = 0
    failed     = 0

    for idx, row in df.iterrows():
        id_code    = str(row['id_code'])
        in_path    = os.path.join(raw_dir,   f'{id_code}.{file_ext}')
        out_path   = os.path.join(cache_dir, f'{id_code}.png')

        if skip_existing and os.path.exists(out_path):
            skipped += 1
            successful += 1
            continue

        if not os.path.exists(in_path):
            failed += 1
            if verbose and failed <= 5:
                print(f'  missing raw: {in_path}')
            continue

        try:
            img_rgb = preprocessor.process(in_path)
            cv2.imwrite(out_path, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
            successful += 1
        except Exception as e:
            failed += 1
            if verbose and failed <= 5:
                print(f'  failed {id_code}: {str(e)[:80]}')

        if verbose and (idx + 1) % 500 == 0:
            print(f'  processed {idx + 1}/{len(df)}'
                  f'  (ok={successful}, skip={skipped}, fail={failed})')

    print('-' * 72)
    print(f'Done. ok={successful}  skipped_existing={skipped}  failed={failed}')
    print('=' * 72)
    return cache_dir


# ============================================================================
#  CLASS-BALANCING (DataFrame-level oversampling for the WeightedRandomSampler
#                   or for a simple shuffle+train loop)
# ============================================================================

def create_balanced_train_dataframe(train_df: pd.DataFrame,
                                    strategy: str = 'oversample',
                                    target_per_class: int = 1000,
                                    label_col: str = 'diagnosis',
                                    random_state: int = 42
                                    ) -> pd.DataFrame:
    """
    Build a balanced label list by resampling rows of `train_df`.

    Strategies
    ----------
    'oversample'  : duplicate minority-class rows up to target_per_class.
                    Combine with online augmentation so duplicates are NOT
                    seen identically by the model.
    'undersample' : drop majority rows down to the smallest class size.

    APTOS 2019 raw distribution (train set, 3662 imgs):
        Grade 0 (No DR)        : 1805
        Grade 1 (Mild)         :  370
        Grade 2 (Moderate)     :  999
        Grade 3 (Severe)       :  193
        Grade 4 (Proliferative):  295

    target_per_class=1000 is a reasonable default — every class roughly
    matches the moderate count without exploding the dataset size.
    """
    if label_col not in train_df.columns:
        raise ValueError(f"'{label_col}' not in DataFrame columns: "
                         f"{list(train_df.columns)}")

    counts = train_df[label_col].value_counts().sort_index()
    print('=' * 72)
    print(f'Class balancing  (strategy={strategy}, target={target_per_class})')
    print('-' * 72)
    print('Before:')
    for grade, count in counts.items():
        print(f'  Grade {grade}: {count:5d}')

    balanced_parts = []

    if strategy == 'oversample':
        for grade in sorted(counts.index):
            sub = train_df[train_df[label_col] == grade]
            balanced_parts.append(sub)
            n_needed = max(0, target_per_class - len(sub))
            if n_needed > 0:
                balanced_parts.append(
                    sub.sample(n=n_needed, replace=True,
                               random_state=random_state)
                )
    elif strategy == 'undersample':
        target = min(target_per_class, int(counts.min()))
        for grade in sorted(counts.index):
            sub = train_df[train_df[label_col] == grade]
            balanced_parts.append(
                sub.sample(n=target, replace=False, random_state=random_state)
            )
    else:
        raise ValueError("strategy must be 'oversample' or 'undersample'")

    balanced = (pd.concat(balanced_parts, ignore_index=True)
                  .sample(frac=1, random_state=random_state)
                  .reset_index(drop=True))

    print('After:')
    for grade, count in (balanced[label_col].value_counts()
                                            .sort_index().items()):
        print(f'  Grade {grade}: {count:5d}')
    print(f'Total: {len(train_df)}  →  {len(balanced)}')
    print('=' * 72)
    return balanced


# ============================================================================
#  VERIFICATION HELPERS — run these once to confirm the pipeline behaves
# ============================================================================

def verify_preprocessing(raw_dir: str = DEFAULT_RAW_DIR,
                         cache_dir: str = DEFAULT_PROCESSED_DIR,
                         save_path: Optional[str] = None,
                         n_samples: int = 4):
    """
    Compare a few raw vs. preprocessed images side-by-side. Saves a PNG
    grid you can eyeball to confirm Ben Graham + CLAHE actually fired.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib not available, skipping visual verification')
        return

    raw_files = sorted(Path(raw_dir).glob('*.png'))[:n_samples]
    if not raw_files:
        print(f'No PNGs found in {raw_dir}')
        return

    fig, axes = plt.subplots(n_samples, 2, figsize=(8, 4 * n_samples))
    if n_samples == 1:
        axes = axes.reshape(1, 2)

    for i, raw_path in enumerate(raw_files):
        cache_path = Path(cache_dir) / raw_path.name
        raw = cv2.cvtColor(cv2.imread(str(raw_path)), cv2.COLOR_BGR2RGB)
        axes[i, 0].imshow(raw)
        axes[i, 0].set_title(f'raw  {raw_path.name}'); axes[i, 0].axis('off')
        if cache_path.exists():
            proc = cv2.cvtColor(cv2.imread(str(cache_path)), cv2.COLOR_BGR2RGB)
            axes[i, 1].imshow(proc)
            axes[i, 1].set_title('preprocessed')
        else:
            axes[i, 1].text(0.5, 0.5, 'NOT PREPROCESSED YET',
                            ha='center', va='center')
        axes[i, 1].axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
        print(f'Comparison saved to {save_path}')
    plt.show()


def verify_augmentation(cache_dir: str = DEFAULT_PROCESSED_DIR,
                        save_dir: Optional[str] = None,
                        n_samples: int = 4):
    """
    Apply the strong training augmentation 4× to a single cached image.
    Reports mean pixel diff (>3 means augmentation is firing) and
    optionally writes the original + 4 augmented versions to disk.

    REMINDER: cached files are NOT augmented — augmentation runs in
    Dataset.__getitem__ at training time. To eyeball augmentation, this
    function applies it manually and saves the result.
    """
    files = list(Path(cache_dir).glob('*.png'))
    if not files:
        print(f'No PNGs found in {cache_dir}')
        return

    aug = get_train_augmentation_strong()
    img_path = files[0]
    original = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)

    print(f'Augmentation check on: {img_path.name}')
    print(f'  original mean={original.mean():.1f}  std={original.std():.1f}')

    diffs = []
    for k in range(8):
        out = aug(image=original)['image']
        # `out` is now a torch.Tensor (C,H,W) normalized — convert back to
        # uint8 image space for diff comparison.
        out_np = out.permute(1, 2, 0).cpu().numpy()
        out_np = (out_np * np.array(IMAGENET_STD) + np.array(IMAGENET_MEAN))
        out_np = np.clip(out_np * 255, 0, 255).astype(np.uint8)
        d = np.abs(original.astype(float) - out_np.astype(float)).mean()
        diffs.append(d)
        print(f'  aug {k+1}: mean pixel diff = {d:.2f}')

    avg = np.mean(diffs)
    print(f'  AVERAGE diff = {avg:.2f}')
    print('  ✓ augmentation is active' if avg > 3.0
          else '  ✗ WARNING: augmentation appears inactive')

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        cv2.imwrite(os.path.join(save_dir, 'original.png'),
                    cv2.cvtColor(original, cv2.COLOR_RGB2BGR))
        for k in range(min(n_samples, 4)):
            t = aug(image=original)['image']
            arr = t.permute(1, 2, 0).cpu().numpy()
            arr = (arr * np.array(IMAGENET_STD) + np.array(IMAGENET_MEAN))
            arr = np.clip(arr * 255, 0, 255).astype(np.uint8)
            cv2.imwrite(os.path.join(save_dir, f'aug_{k+1}.png'),
                        cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
        print(f'  examples written to {save_dir}/')


# ============================================================================
#  CLI ENTRYPOINT
# ============================================================================

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='APTOS 2019 preprocessing — data/raw → data/processed')
    parser.add_argument('--raw-dir',   default=DEFAULT_RAW_DIR)
    parser.add_argument('--csv',       default=DEFAULT_RAW_CSV)
    parser.add_argument('--cache-dir', default=DEFAULT_PROCESSED_DIR)
    parser.add_argument('--method',    default='hybrid',
                        choices=['hybrid', 'ben_graham', 'clahe'])
    parser.add_argument('--size',      type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument('--balance',   action='store_true',
                        help='also write data/processed/train_balanced.csv')
    parser.add_argument('--target',    type=int, default=1000,
                        help='target images per class for balancing')
    parser.add_argument('--verify',    action='store_true',
                        help='show raw vs processed comparison after')
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    preprocess_and_cache(
        raw_dir=args.raw_dir,
        cache_dir=args.cache_dir,
        df=df,
        method=args.method,
        image_size=args.size,
    )

    if args.balance:
        bal = create_balanced_train_dataframe(
            df, strategy='oversample', target_per_class=args.target)
        bal.to_csv(DEFAULT_BALANCED_CSV, index=False)
        print(f'Balanced CSV → {DEFAULT_BALANCED_CSV}')

    if args.verify:
        verify_preprocessing(args.raw_dir, args.cache_dir,
                             save_path='data/processed/_verify.png')
