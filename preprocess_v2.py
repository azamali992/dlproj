"""
preprocess_v2.py — Fixed augmentation pipeline
=================================================

Changes from v1:
1. Removed `p=0.8` from A.Compose (was silently skipping ALL augmentations 20% of the time)
2. Removed A.GaussNoise and A.ElasticTransform (breaking API changes in albumentations >= 2.0)
3. Added medically-appropriate augmentations: ColorJitter, CLAHE as aug, GaussianBlur
4. Added ShiftScaleRotate for zoom/scale invariance (critical for varying fundus FOV)
5. Added verify_augmentation() so you can confirm it is actually running

NOTE on "augmentation not visible in cache":
    Cached images are RAW preprocessed images with NO augmentation applied.
    Augmentation happens in-memory at dataset __getitem__ time during training.
    If you open ./data/preprocessed_hybrid/*.png, you will NOT see augmentation.
    To see augmented examples, call verify_augmentation() below.
"""

import cv2
import numpy as np
import albumentations as A
import os
import pandas as pd
from pathlib import Path


# ============================================================================
# PREPROCESSING: Ben Graham + CLAHE  (unchanged from v1)
# ============================================================================

class BenGrahamPreprocessor:
    def __init__(self, image_size: int = 224, kernel_size: int = 31):
        self.image_size = image_size
        self.kernel_size = kernel_size

    def process(self, image_path: str) -> np.ndarray:
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.image_size, self.image_size),
                           interpolation=cv2.INTER_LINEAR)

        img_float = image.astype(np.float32)
        blurred = cv2.GaussianBlur(img_float, (self.kernel_size, self.kernel_size), 0)
        enhanced = img_float + (img_float - blurred) * 0.5
        image = np.clip(enhanced, 0, 255).astype(np.uint8)

        img_float = image.astype(np.float32) / 255.0
        for i in range(3):
            ch = img_float[:, :, i]
            std = ch.std()
            if std > 0:
                ch = (ch - ch.mean()) / std
                ch = (ch - ch.min()) / (ch.max() - ch.min() + 1e-8)
                img_float[:, :, i] = ch
        return (img_float * 255).astype(np.uint8)


class CLAHEPreprocessor:
    def __init__(self, clip_limit: float = 2.0, tile_size: int = 8, image_size: int = 224):
        self.clip_limit = clip_limit
        self.tile_size = tile_size
        self.image_size = image_size

    def process(self, image_path: str) -> np.ndarray:
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.image_size, self.image_size),
                           interpolation=cv2.INTER_LINEAR)

        clahe = cv2.createCLAHE(clipLimit=self.clip_limit,
                                tileGridSize=(self.tile_size, self.tile_size))
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)
        v = clahe.apply(v)
        image = cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2RGB)
        return image


class HybridPreprocessor:
    def __init__(self, image_size: int = 224):
        self.bg = BenGrahamPreprocessor(image_size=image_size)

    def process(self, image_path: str) -> np.ndarray:
        image = self.bg.process(image_path)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)
        v = clahe.apply(v)
        return cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2RGB)


# ============================================================================
# AUGMENTATION TRANSFORMS — FIXED
# ============================================================================

def get_train_augmentation():
    """
    Training augmentation pipeline.

    Key fixes vs v1:
    - Removed p=0.8 from Compose: the entire pipeline is now always entered.
      Individual transforms still have their own probabilities.
    - Removed A.GaussNoise: var_limit API changed in albumentations >= 2.0.
    - Removed A.ElasticTransform: distorts lesion morphology and broke in >= 2.0.
    - Added ShiftScaleRotate: handles scale variation across fundus cameras.
    - Added HueSaturationValue + CLAHE: critical for fundus colour normalisation.
    - Added GaussianBlur: simulates different image sharpness levels.
    """
    return A.Compose([
        # Geometric — simulate camera/patient positioning variation
        A.ShiftScaleRotate(
            shift_limit=0.05,
            scale_limit=0.15,   # ±15% zoom — models varying fundus FOV
            rotate_limit=30,
            border_mode=cv2.BORDER_REFLECT,
            p=0.8,
        ),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),

        # Photometric — simulate different camera/lighting conditions
        A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.7),
        A.HueSaturationValue(
            hue_shift_limit=10,
            sat_shift_limit=25,
            val_shift_limit=25,
            p=0.5,
        ),
        # CLAHE as augmentation: randomly re-enhances contrast
        A.CLAHE(clip_limit=3.0, tile_grid_size=(8, 8), p=0.3),

        # Blur — simulates focus variation; forces model to use larger receptive field
        A.GaussianBlur(blur_limit=(3, 7), p=0.3),
    ])
    # NOTE: no p= at the Compose level → the pipeline is ALWAYS entered.
    # Individual p= values control each transform independently.


def get_val_augmentation():
    """No augmentation for validation — deterministic transforms only."""
    return A.Compose([])


# ============================================================================
# AUGMENTATION VERIFICATION — run this to confirm transforms are working
# ============================================================================

def verify_augmentation(cache_dir: str, sample_id: str = None, save_dir: str = None):
    """
    Load one cached image, apply the training pipeline N times, and compare.
    Prints mean pixel difference (should be > 5 if augmentation is active).

    Args:
        cache_dir: directory containing preprocessed .png files
        sample_id: filename without extension (auto-picks first file if None)
        save_dir: if provided, saves original + 4 augmented versions there
    """
    aug = get_train_augmentation()

    if sample_id is None:
        files = list(Path(cache_dir).glob("*.png"))
        if not files:
            print("ERROR: no .png files found in cache_dir")
            return
        img_path = str(files[0])
    else:
        img_path = os.path.join(cache_dir, f"{sample_id}.png")

    original = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    print(f"\nAugmentation verification on: {os.path.basename(img_path)}")
    print(f"Original image stats — mean: {original.mean():.1f}, std: {original.std():.1f}")

    diffs = []
    for i in range(8):
        augmented = aug(image=original)['image']
        diff = np.abs(original.astype(float) - augmented.astype(float)).mean()
        diffs.append(diff)
        print(f"  Augmented {i+1}: mean pixel diff = {diff:.2f}")

    avg_diff = np.mean(diffs)
    print(f"\nAverage pixel diff: {avg_diff:.2f}")
    if avg_diff > 3.0:
        print("✓ Augmentation is WORKING correctly")
    else:
        print("✗ WARNING: augmentation may not be applying — check albumentations version")

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        cv2.imwrite(os.path.join(save_dir, "original.png"),
                    cv2.cvtColor(original, cv2.COLOR_RGB2BGR))
        for i in range(4):
            aug_img = aug(image=original)['image']
            cv2.imwrite(os.path.join(save_dir, f"aug_{i+1}.png"),
                        cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR))
        print(f"✓ Saved original + 4 augmented examples to: {save_dir}")


# ============================================================================
# PREPROCESSING CACHE  (unchanged from v1)
# ============================================================================

def preprocess_and_cache(raw_dir: str = None, cache_dir: str = None,
                         df: pd.DataFrame = None, img_dir: str = None,
                         method: str = 'hybrid', verbose: bool = True):
    if raw_dir is None and img_dir is None:
        raise ValueError("Must provide either raw_dir or img_dir")
    if raw_dir is None:
        raw_dir = img_dir
    os.makedirs(cache_dir, exist_ok=True)

    preprocessor_map = {
        'hybrid': HybridPreprocessor(image_size=224),
        'ben_graham': BenGrahamPreprocessor(image_size=224),
        'clahe': CLAHEPreprocessor(image_size=224),
    }
    preprocessor = preprocessor_map.get(method, preprocessor_map['hybrid'])

    print(f"\n{'='*80}")
    print(f"PREPROCESSING & CACHING  (method={method})")
    print(f"{'='*80}")

    successful = 0
    failed = 0

    for idx, row in df.iterrows():
        id_code = row['id_code']
        img_path = os.path.join(raw_dir, f"{id_code}.png")
        cache_path = os.path.join(cache_dir, f"{id_code}.png")

        if os.path.exists(cache_path):
            successful += 1
            continue

        try:
            image = preprocessor.process(img_path)
            cv2.imwrite(cache_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            successful += 1
        except Exception as e:
            failed += 1
            if verbose and failed <= 5:
                print(f"  Failed {id_code}: {str(e)[:60]}")

        if verbose and (idx + 1) % 500 == 0:
            print(f"  Processed {idx + 1}/{len(df)}...")

    print(f"✓ Preprocessing complete: {successful} success, {failed} failed")
    return cache_dir


# ============================================================================
# CLASS BALANCING  (unchanged from v1)
# ============================================================================

def create_balanced_train_dataframe(train_df: pd.DataFrame,
                                    strategy: str = 'oversample',
                                    target_per_class: int = 700) -> pd.DataFrame:
    print(f"\n{'='*80}")
    print(f"CLASS BALANCING (via DataFrame resampling, strategy={strategy})")
    print(f"{'='*80}")

    class_counts = train_df['diagnosis'].value_counts().sort_index()
    print(f"\nBEFORE:")
    for grade, count in class_counts.items():
        needed = max(0, target_per_class - count)
        mult = target_per_class / count if count > 0 else 0
        print(f"  Grade {grade}: {count:4d} (need {needed:4d}, x{mult:.1f})")

    balanced_rows = []

    if strategy == 'oversample':
        for grade in sorted(class_counts.index):
            grade_df = train_df[train_df['diagnosis'] == grade]
            balanced_rows.append(grade_df)
            n_needed = max(0, target_per_class - len(grade_df))
            if n_needed > 0:
                balanced_rows.append(
                    grade_df.sample(n=n_needed, replace=True, random_state=42)
                )
    elif strategy == 'undersample':
        target = min(target_per_class, class_counts.min())
        for grade in sorted(class_counts.index):
            balanced_rows.append(
                train_df[train_df['diagnosis'] == grade]
                .sample(n=target, replace=False, random_state=42)
            )

    balanced_df = pd.concat(balanced_rows, ignore_index=True).reset_index(drop=True)

    print(f"\nAFTER:")
    for grade, count in balanced_df['diagnosis'].value_counts().sort_index().items():
        print(f"  Grade {grade}: {count:4d}")
    print(f"\nTotal: {len(train_df)} -> {len(balanced_df)}")

    return balanced_df


if __name__ == '__main__':
    print("preprocess_v2.py loaded.")
    print("Run verify_augmentation(cache_dir) to confirm augmentation is active.")
