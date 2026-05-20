"""
generate_aug_analysis.py
========================
Generates results/aug_analysis/ with pre- and post-augmentation images for
every DR grade, using the grade-appropriate augmentation pipeline.

Preprocessing is applied on-the-fly from raw images so the output always
reflects the current pipeline (including the pad_to_square fix).

Output structure
----------------
results/aug_analysis/
    grade_0_mild/
        original.png        <- preprocessed image, no augmentation
        aug_1.png ... aug_4.png
    grade_1_mild/   ...
    grade_2_moderate/ ...
    grade_3_strong/ ...
    grade_4_strong/ ...
    summary_grid.png        <- 5x5 grid: rows=grades, cols=orig+4 augs

Run
---
    python generate_aug_analysis.py
"""

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

from preprocess_grade_aware_aug import (
    preprocess_full,
    get_train_augmentation_mild,
    get_train_augmentation_moderate,
    get_train_augmentation_strong,
    IMAGENET_MEAN,
    IMAGENET_STD,
)

# ── Configuration ─────────────────────────────────────────────────────────────
RAW_DIR    = os.path.join('data', 'raw', 'train_images')
CSV_PATH   = os.path.join('data', 'raw', 'train.csv')
OUTPUT_DIR = os.path.join('results', 'aug_analysis')
N_AUG      = 4       # augmented variants saved per grade
GRID_DPI   = 150     # resolution of the summary grid PNG
IMG_SIZE   = 224

GRADE_AUG = {
    0: ('mild',     get_train_augmentation_mild),
    1: ('mild',     get_train_augmentation_mild),
    2: ('moderate', get_train_augmentation_moderate),
    3: ('strong',   get_train_augmentation_strong),
    4: ('strong',   get_train_augmentation_strong),
}

GRADE_LABELS = {
    0: 'Grade 0 - No DR',
    1: 'Grade 1 - Mild',
    2: 'Grade 2 - Moderate',
    3: 'Grade 3 - Severe',
    4: 'Grade 4 - Proliferative',
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def tensor_to_uint8(t) -> np.ndarray:
    """Undo ImageNet normalisation and convert to uint8 HWC RGB."""
    arr = t.permute(1, 2, 0).cpu().numpy()
    arr = arr * np.array(IMAGENET_STD) + np.array(IMAGENET_MEAN)
    return np.clip(arr * 255, 0, 255).astype(np.uint8)


def save_rgb(img: np.ndarray, path: str) -> None:
    cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load grade labels from CSV
    df = pd.read_csv(CSV_PATH)
    df.columns = [c.strip().lower() for c in df.columns]

    # Group raw image paths by grade, pick one per grade
    raw_dir = Path(RAW_DIR)
    grade_to_path: dict = {}
    for _, row in df.iterrows():
        g = int(row['diagnosis'])
        if g not in grade_to_path:
            p = raw_dir / f"{row['id_code']}.png"
            if p.exists():
                grade_to_path[g] = p
        if len(grade_to_path) == 5:
            break

    print('Raw images selected for analysis:')
    for g, p in sorted(grade_to_path.items()):
        print(f'  Grade {g}: {p.name}')

    # Build the grid: rows = grades, cols = original + N_AUG augmented
    grid_imgs   = []
    grid_titles = []

    for grade in range(5):
        if grade not in grade_to_path:
            print(f'WARNING: no raw image found for Grade {grade}, skipping')
            continue

        strength, aug_fn = GRADE_AUG[grade]
        aug = aug_fn()

        # Preprocess the raw image on-the-fly using the current pipeline
        raw_bgr = cv2.imread(str(grade_to_path[grade]))
        original = preprocess_full(raw_bgr, size=IMG_SIZE)   # uint8 RGB

        grade_dir = os.path.join(OUTPUT_DIR, f'grade_{grade}_{strength}')
        os.makedirs(grade_dir, exist_ok=True)

        # Save pre-augmentation (preprocessed, not yet augmented)
        save_rgb(original, os.path.join(grade_dir, 'original.png'))

        row_imgs   = [original]
        row_titles = [f'{GRADE_LABELS[grade]}\noriginal (pre-aug)']

        # Save post-augmentation images
        for k in range(N_AUG):
            t = aug(image=original)['image']
            aug_img = tensor_to_uint8(t)
            save_rgb(aug_img, os.path.join(grade_dir, f'aug_{k+1}.png'))
            row_imgs.append(aug_img)
            row_titles.append(f'{strength} aug #{k+1}')

        grid_imgs.append(row_imgs)
        grid_titles.append(row_titles)

        mean_diff = np.mean([
            np.abs(original.astype(float) -
                   tensor_to_uint8(aug(image=original)['image']).astype(float)).mean()
            for _ in range(4)
        ])
        print(f'Grade {grade} ({strength}): saved to {grade_dir}/  '
              f'[mean px diff={mean_diff:.1f}]')

    # Summary grid
    n_rows = len(grid_imgs)
    n_cols = 1 + N_AUG

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * 2.8, n_rows * 2.8),
        gridspec_kw={'hspace': 0.05, 'wspace': 0.05},
    )
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    for r, (row_imgs, row_titles) in enumerate(zip(grid_imgs, grid_titles)):
        for c, (img, title) in enumerate(zip(row_imgs, row_titles)):
            ax = axes[r, c]
            ax.imshow(img)
            ax.axis('off')
            fontsize = 7 if c > 0 else 8
            weight   = 'bold' if c == 0 else 'normal'
            ax.set_title(title, fontsize=fontsize, fontweight=weight, pad=2)

    fig.suptitle(
        'Pre-augmentation (col 1)  vs  Post-augmentation (cols 2-5)\n'
        'Each row uses the grade-appropriate pipeline: '
        'mild (G0-G1) / moderate (G2) / strong (G3-G4)',
        fontsize=9, y=1.01,
    )

    grid_path = os.path.join(OUTPUT_DIR, 'summary_grid.png')
    plt.savefig(grid_path, dpi=GRID_DPI, bbox_inches='tight')
    plt.close(fig)

    print(f'\nSummary grid saved: {grid_path}')
    print(f'All per-grade folders: {OUTPUT_DIR}/')


if __name__ == '__main__':
    main()
