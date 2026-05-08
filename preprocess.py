"""
SIMPLE APPROACH: Augmentation happens at dataset/dataloader level
NOT at filesystem level. This is more reliable and standard in deep learning.
"""

import cv2
import numpy as np
import albumentations as A
import os
import pandas as pd
from pathlib import Path

# ============================================================================
# PREPROCESSING: Ben Graham + CLAHE
# ============================================================================

class BenGrahamPreprocessor:
    """Ben Graham preprocessing"""
    
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
                ch = (ch - ch.min()) / (ch.max() - ch.min())
                img_float[:, :, i] = ch
        image = (img_float * 255).astype(np.uint8)
        
        return image


class CLAHEPreprocessor:
    """CLAHE"""
    
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
        hsv = cv2.merge([h, s, v])
        image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        
        return image


class HybridPreprocessor:
    """Ben Graham → CLAHE"""
    
    def __init__(self, image_size: int = 224):
        self.bg = BenGrahamPreprocessor(image_size=image_size)
        self.clahe = CLAHEPreprocessor(image_size=image_size)

    def process(self, image_path: str) -> np.ndarray:
        image = self.bg.process(image_path)
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)
        v = clahe.apply(v)
        hsv = cv2.merge([h, s, v])
        image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        
        return image


# ============================================================================
# AUGMENTATION TRANSFORMS (applied at dataset level during training)
# ============================================================================

def get_train_augmentation():
    """Augmentation for TRAINING set"""
    return A.Compose([
        A.Rotate(limit=15, p=0.7, border_mode=cv2.BORDER_REFLECT),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.6),
        A.GaussNoise(p=0.3, var_limit=(10, 50)),
        A.ElasticTransform(alpha=1, sigma=50, p=0.3, border_mode=cv2.BORDER_REFLECT),
    ], p=0.8)


def get_val_augmentation():
    """NO augmentation for validation"""
    return A.Compose([], p=1.0)


# ============================================================================
# SIMPLE PREPROCESSING CACHE (no complex augmentation, just cache images)
# ============================================================================

def preprocess_and_cache(raw_dir: str = None, cache_dir: str = None, df: pd.DataFrame = None,
                         img_dir: str = None, method: str = 'hybrid', verbose: bool = True):
    """
    Simple: Just preprocess and cache all images.
    Augmentation happens at dataset level during training.
    
    Args:
        raw_dir or img_dir: Directory with original images
        cache_dir: Where to save preprocessed images
        df: DataFrame with id_code and diagnosis columns
        method: 'hybrid', 'ben_graham', or 'clahe'
        verbose: Print progress
    """
    # Handle both raw_dir and img_dir parameter names
    if raw_dir is None and img_dir is None:
        raise ValueError("Must provide either raw_dir or img_dir")
    if raw_dir is None:
        raw_dir = img_dir
    os.makedirs(cache_dir, exist_ok=True)
    
    if method == 'hybrid':
        preprocessor = HybridPreprocessor(image_size=224)
    elif method == 'ben_graham':
        preprocessor = BenGrahamPreprocessor(image_size=224)
    else:
        preprocessor = CLAHEPreprocessor(image_size=224)
    
    print(f"\n{'='*80}")
    print(f"PREPROCESSING & CACHING")
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
        
        if (idx + 1) % 500 == 0:
            print(f"  Processed {idx + 1}/{len(df)}...")
    
    print(f"✓ Preprocessing complete: {successful} success, {failed} failed")
    return cache_dir


# ============================================================================
# CLASS BALANCING: Simple resampling approach
# ============================================================================

def create_balanced_train_dataframe(train_df: pd.DataFrame, 
                                   strategy: str = 'oversample',
                                   target_per_class: int = 700) -> pd.DataFrame:
    """
    Create balanced training DataFrame using simple resampling.
    
    No file creation - just duplicate rows in the DataFrame.
    Albumentations will provide diversity via augmentation.
    
    Args:
        train_df: Original training DataFrame
        strategy: 'oversample' (duplicate) or 'undersample' (remove)
        target_per_class: Target count per class
    
    Returns:
        Balanced DataFrame (ready for dataset)
    """
    
    print(f"\n{'='*80}")
    print(f"CLASS BALANCING (via DataFrame resampling)")
    print(f"{'='*80}")
    
    class_counts = train_df['diagnosis'].value_counts().sort_index()
    
    print(f"\nBEFORE:")
    for grade, count in class_counts.items():
        needed = max(0, target_per_class - count)
        mult = target_per_class / count if count > 0 else 0
        print(f"  Grade {grade}: {count:4d} (need {needed:4d}, ×{mult:.1f})")
    
    balanced_rows = []
    
    if strategy == 'oversample':
        # Oversample minority to target
        for grade in sorted(class_counts.index):
            grade_df = train_df[train_df['diagnosis'] == grade]
            n_orig = len(grade_df)
            n_needed = max(0, target_per_class - n_orig)
            
            # Add originals
            balanced_rows.append(grade_df)
            
            # Oversample by repeating
            if n_needed > 0:
                # Randomly sample with replacement
                oversampled = grade_df.sample(n=n_needed, replace=True, random_state=42)
                balanced_rows.append(oversampled)
    
    elif strategy == 'undersample':
        # Undersample majority to target
        min_count = class_counts.min()
        target = min(target_per_class, min_count)
        
        for grade in sorted(class_counts.index):
            grade_df = train_df[train_df['diagnosis'] == grade]
            sampled = grade_df.sample(n=target, replace=False, random_state=42)
            balanced_rows.append(sampled)
    
    balanced_df = pd.concat(balanced_rows, ignore_index=True).reset_index(drop=True)
    
    print(f"\nAFTER:")
    new_counts = balanced_df['diagnosis'].value_counts().sort_index()
    for grade, count in new_counts.items():
        print(f"  Grade {grade}: {count:4d}")
    
    print(f"\nTotal samples: {len(train_df)} → {len(balanced_df)}")
    
    return balanced_df


# ============================================================================
# QUICK TEST
# ============================================================================

if __name__ == '__main__':
    print("Module loaded successfully!")
    print("\nUsage in train_coral_focal.py:")
    print("""
    from preprocess import (
        preprocess_and_cache,
        create_balanced_train_dataframe,
        get_train_augmentation,
        get_val_augmentation,
        HybridPreprocessor
    )
    
    # Step 1: Preprocess and cache
    preprocess_and_cache(
        raw_dir='./data/train_images',
        cache_dir='./data/preprocessed_hybrid',
        df=pd.concat([train_df, val_df]),
        method='hybrid'
    )
    
    # Step 2: Balance training set (resampling, no file creation)
    train_df_balanced = create_balanced_train_dataframe(
        train_df,
        strategy='oversample',
        target_per_class=700
    )
    
    # Step 3: Create dataset with augmentation
    train_dataset = PreprocessedDRDataset(
        cache_dir='./data/preprocessed_hybrid',
        df=train_df_balanced,
        augmentation=get_train_augmentation()  # ← Augmentation at dataset level
    )
    val_dataset = PreprocessedDRDataset(
        cache_dir='./data/preprocessed_hybrid',
        df=val_df,
        augmentation=get_val_augmentation()  # No augmentation for val
    )
    """)