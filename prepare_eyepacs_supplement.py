"""prepare_eyepacs_supplement.py
================================
Extracts 50 images per DR grade (0-4) from a (partial) EyePACS zip download
and prepares them as a supplement to the APTOS training data.

REVERSIBLE: all output goes to data/eyepacs_supplement/.
To undo: delete that folder.  Original APTOS data is never touched.

Output layout:
  data/eyepacs_supplement/
    raw_images/           <- raw EyePACS images (named eyepacs_<stem>)
    supplement.csv        <- id_code, diagnosis  (same format as train.csv)
    processed/
      hybrid_224/         <- same preprocessing pipeline as v6

Expected inputs in data/eyepacs_supplement/_download_tmp/:
  eyepacs.zip          <- full or partial Kaggle download
  trainLabels.csv      <- labels file  (columns: image, level)

Usage:
  python prepare_eyepacs_supplement.py
"""

import os
import sys
import struct
import zlib
import io
import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

# ── Config ─────────────────────────────────────────────────────────────────────
IMAGES_PER_CLASS = 250   # max ~450 for grade 4 given current partial download; hard cap is 708
RANDOM_SEED      = 42
SUPP_ROOT        = Path('data/eyepacs_supplement')
RAW_IMG_DIR      = SUPP_ROOT / 'raw_images'
SUPP_CSV         = SUPP_ROOT / 'supplement.csv'
CACHE_DIR        = SUPP_ROOT / 'processed' / 'hybrid_224'
DOWNLOAD_TMP     = SUPP_ROOT / '_download_tmp'
EXISTING_ZIP     = DOWNLOAD_TMP / 'eyepacs.zip'
IMAGE_SIZE       = 224

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ── ZIP64-aware local file header parser ──────────────────────────────────────
LFH_SIG = b'PK\x03\x04'
LFH_FMT = '<HHHHHIIIHH'   # 26 bytes after signature
LFH_LEN = struct.calcsize(LFH_FMT)
IMG_EXTS = {'.jpg', '.jpeg', '.png'}


def _parse_zip64_extra(extra: bytes, comp_sz_flag: bool, uncomp_sz_flag: bool):
    """
    Read real compressed/uncompressed sizes from a ZIP64 extended info block.
    Returns (real_comp_sz, real_uncomp_sz) or (None, None) if not found.
    """
    pos = 0
    while pos + 4 <= len(extra):
        tag, size = struct.unpack_from('<HH', extra, pos)
        data = extra[pos + 4: pos + 4 + size]
        pos += 4 + size
        if tag != 0x0001:
            continue
        # Fields present in order: uncomp_sz (if flag), comp_sz (if flag)
        offset = 0
        real_uncomp = None
        real_comp   = None
        if uncomp_sz_flag and offset + 8 <= len(data):
            real_uncomp = struct.unpack_from('<Q', data, offset)[0]
            offset += 8
        if comp_sz_flag and offset + 8 <= len(data):
            real_comp = struct.unpack_from('<Q', data, offset)[0]
            offset += 8
        return real_comp, real_uncomp
    return None, None


def _find_next_sig(f, start_pos: int, file_size: int, sig: bytes = LFH_SIG) -> int:
    """Scan forward from start_pos for the next occurrence of sig. Returns -1 if not found."""
    BLOCK = 1 << 20
    pos = start_pos
    while pos < file_size:
        f.seek(pos)
        block = f.read(min(BLOCK, file_size - pos))
        if not block:
            break
        idx = block.find(sig)
        if idx != -1:
            return pos + idx
        pos += len(block) - (len(sig) - 1)
    return -1


def _load_labels(tmp_dir: Path) -> dict:
    """
    Load the labels CSV from tmp_dir.
    Accepts any CSV with one column matching image stems and one for level/grade.
    Returns {stem: grade} dict.
    """
    csvs = list(tmp_dir.glob('*.csv'))
    if not csvs:
        raise FileNotFoundError(
            f"No CSV found in {tmp_dir}. "
            "Download the labels file (e.g. trainLabels.csv) from Kaggle "
            "and put it in that folder.")

    csv_path = csvs[0]
    print(f"Loading labels from: {csv_path.name}")
    df = pd.read_csv(csv_path)
    cols = [c.lower() for c in df.columns]

    id_col  = next((df.columns[i] for i, c in enumerate(cols)
                    if c in ('image', 'id_code', 'filename', 'id', 'name')), None)
    lbl_col = next((df.columns[i] for i, c in enumerate(cols)
                    if c in ('level', 'diagnosis', 'label', 'grade')), None)

    if id_col is None or lbl_col is None:
        raise ValueError(f"Cannot detect id/label columns in {csv_path}. "
                         f"Columns found: {list(df.columns)}")

    mapping = {str(row[id_col]).strip(): int(row[lbl_col])
               for _, row in df.iterrows()}
    print(f"  Loaded {len(mapping)} label entries  "
          f"(id_col={id_col!r}, lbl_col={lbl_col!r})")
    dist = pd.Series(list(mapping.values())).value_counts().sort_index()
    for g, n in dist.items():
        print(f"  Grade {g}: {n}")
    return mapping


# ── Streaming extractor ────────────────────────────────────────────────────────
def stream_extract(zip_path: Path, raw_img_dir: Path,
                   label_map: dict, images_per_class: int = 50):
    """
    Scan a ZIP file (including partial downloads) by reading local file headers
    directly.  Handles ZIP64 extended sizes in the extra field.

    label_map: {image_stem: grade}
    Returns list of {'id_code': str, 'diagnosis': int} rows.
    """
    raw_img_dir.mkdir(parents=True, exist_ok=True)
    file_size = zip_path.stat().st_size

    # Per-class reservoir: keep a fixed random sample
    grade_counts  = {g: 0 for g in range(5)}
    rows          = []

    print(f"\nScanning {zip_path.name} ({file_size / (1 << 30):.1f} GB) ...")
    print(f"Stops as soon as every grade has collected {images_per_class} images.")

    with open(zip_path, 'rb') as f:
        pos = 0

        while True:
            pos = _find_next_sig(f, pos, file_size)
            if pos == -1:
                break

            # ── Parse local file header ────────────────────────────────────
            f.seek(pos + 4)
            raw_hdr = f.read(LFH_LEN)
            if len(raw_hdr) < LFH_LEN:
                break

            (ver, flags, method, mtime, mdate, crc32,
             comp_sz, uncomp_sz,
             fname_len, extra_len) = struct.unpack(LFH_FMT, raw_hdr)

            fname_bytes = f.read(fname_len)
            extra_bytes = f.read(extra_len)
            data_pos    = pos + 4 + LFH_LEN + fname_len + extra_len

            try:
                filename = fname_bytes.decode('utf-8', errors='replace').replace('\\', '/')
            except Exception:
                pos = data_pos
                continue

            # ── Resolve ZIP64 sizes ────────────────────────────────────────
            need_comp   = (comp_sz   == 0xFFFFFFFF)
            need_uncomp = (uncomp_sz == 0xFFFFFFFF)
            if need_comp or need_uncomp:
                rc, ru = _parse_zip64_extra(extra_bytes, need_comp, need_uncomp)
                if rc is not None:
                    comp_sz   = rc
                if ru is not None:
                    uncomp_sz = ru

            # Skip directories and entries with unknown/zero compressed size
            if filename.endswith('/') or comp_sz == 0:
                pos = data_pos
                continue

            lower    = filename.lower()
            is_image = any(lower.endswith(ext) for ext in IMG_EXTS)

            if not is_image:
                pos = data_pos + comp_sz
                continue

            # ── Look up grade ──────────────────────────────────────────────
            stem  = Path(filename).stem          # e.g. "10003_left"
            grade = label_map.get(stem)
            if grade is None:
                pos = data_pos + comp_sz
                continue

            if grade_counts[grade] >= images_per_class:
                pos = data_pos + comp_sz
                continue

            # ── Extract and decode ─────────────────────────────────────────
            f.seek(data_pos)
            comp_data = f.read(comp_sz)
            if len(comp_data) < comp_sz:
                print(f"  Reached end of partial file at {filename}")
                break

            try:
                if method == 0:
                    img_data = comp_data
                elif method == 8:
                    img_data = zlib.decompress(comp_data, -15)
                else:
                    pos = data_pos + comp_sz
                    continue

                arr = np.frombuffer(img_data, dtype=np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if img is None:
                    pos = data_pos + comp_sz
                    continue
            except Exception as e:
                pos = data_pos + comp_sz
                continue

            dest_name = f"eyepacs_{stem}"
            dest_path = raw_img_dir / f"{dest_name}.png"
            cv2.imwrite(str(dest_path), img)

            grade_counts[grade] += 1
            rows.append({'id_code': dest_name, 'diagnosis': grade})

            total = sum(grade_counts.values())
            if total % 25 == 0:
                print(f"  Collected: {dict(grade_counts)}  (total={total})")

            if all(v >= images_per_class for v in grade_counts.values()):
                print(f"\n  All {images_per_class} per class reached — stopping early.")
                break

            pos = data_pos + comp_sz

    print(f"\nExtraction complete: {grade_counts}")
    return rows


# ── Preprocessing ──────────────────────────────────────────────────────────────
def _run_preprocessing():
    supp_df  = pd.read_csv(SUPP_CSV)
    n_total  = len(supp_df)
    n_cached = sum(1 for _, r in supp_df.iterrows()
                   if (CACHE_DIR / f"{r['id_code']}.png").exists())

    if n_cached >= n_total:
        print(f"\n[SKIP] Cache complete ({n_cached}/{n_total} in {CACHE_DIR}).")
        return

    print(f"\nPreprocessing {n_total - n_cached} new images -> {CACHE_DIR} ...")
    from preprocess_grade_aware_aug import preprocess_and_cache
    preprocess_and_cache(
        raw_dir=str(RAW_IMG_DIR),
        cache_dir=str(CACHE_DIR),
        df=supp_df,
        method='hybrid',
        image_size=IMAGE_SIZE,
        file_ext='png',
        skip_existing=True,
        verbose=True,
    )
    n_done   = len(list(CACHE_DIR.glob('*.png')))
    missing  = [r['id_code'] for _, r in supp_df.iterrows()
                if not (CACHE_DIR / f"{r['id_code']}.png").exists()]
    print(f"Preprocessing done: {n_done}/{n_total} images.")
    if missing:
        print(f"[WARN] {len(missing)} missing: {missing[:5]}")


# ── Main ──────────────────────────────────────────────────────────────────────
def prepare():
    print("=" * 70)
    print("EyePACS Supplement Preparation")
    print(f"Target: {IMAGES_PER_CLASS} images per class  "
          f"(5 classes = {IMAGES_PER_CLASS * 5} total)")
    print("=" * 70)

    SUPP_ROOT.mkdir(parents=True, exist_ok=True)
    RAW_IMG_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if SUPP_CSV.exists():
        df = pd.read_csv(SUPP_CSV)
        print(f"\n[SKIP] supplement.csv already exists ({len(df)} rows).")
        print("Delete it and raw_images/ to redo.\n")
        _run_preprocessing()
        return

    if not EXISTING_ZIP.exists():
        raise FileNotFoundError(
            f"No zip file found at {EXISTING_ZIP}.\n"
            "Download the EyePACS dataset from Kaggle first:\n"
            "  kaggle datasets download dreamer07/eyepacs -p data/eyepacs_supplement/_download_tmp/")

    # Load labels CSV (must be manually placed in _download_tmp)
    label_map = _load_labels(DOWNLOAD_TMP)

    # Stream-extract images
    rows = stream_extract(EXISTING_ZIP, RAW_IMG_DIR, label_map, IMAGES_PER_CLASS)

    if not rows:
        raise RuntimeError(
            "No images extracted.\n"
            "The partial zip may not contain images for all grades yet.\n"
            "Check that trainLabels.csv image stems match the zip filenames.")

    supp_df = pd.DataFrame(rows)
    dist    = supp_df['diagnosis'].value_counts().sort_index()
    print("\nExtracted class distribution:")
    for g in range(5):
        n    = dist.get(g, 0)
        warn = " [INCOMPLETE — need more of the zip]" if n < IMAGES_PER_CLASS else ""
        print(f"  Grade {g}: {n:3d}{warn}")

    supp_df.to_csv(SUPP_CSV, index=False)
    print(f"\nSaved: {SUPP_CSV}  ({len(supp_df)} rows)")

    _run_preprocessing()


def print_reversal_instructions():
    print("\n" + "=" * 70)
    print("HOW TO REVERT (restore original APTOS-only setup):")
    print("  Delete: data/eyepacs_supplement/")
    print("  Run:    train_coral_focal_v6.py  instead of v7")
    print("  Original data/raw/ and data/processed/ are UNTOUCHED.")
    print("=" * 70)


if __name__ == '__main__':
    try:
        prepare()
        print_reversal_instructions()
        print("\nNext step: python train_coral_focal_v7.py")
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(1)
