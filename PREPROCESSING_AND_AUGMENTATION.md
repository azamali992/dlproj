# Preprocessing and Augmentation Pipeline

**File:** `preprocess_grade_aware_aug.py`  
**Used by:** `train_coral_focal_v5.py` (and all experiments from v4 onward)

---

## Overview

Every raw fundus image goes through two distinct stages before reaching the model:

```
Raw PNG (variable size, variable lighting)
    │
    ▼
PREPROCESSING  (done once, result saved to disk as a cached PNG)
    │
    ▼
Cached PNG (224×224, standardised contrast and lighting)
    │
    ▼
AUGMENTATION   (done every time an image is loaded during training)
    │
    ▼
Normalised Tensor (3×224×224, float32, ImageNet-normalised)
    │
    ▼
Model (DenseNet121 + CORAL head)
```

Preprocessing is a one-time operation. The result is saved to `data/processed/hybrid_224/` so it never runs twice for the same image. Augmentation is random and runs fresh every time a training batch is constructed — so even if an image appears multiple times (due to oversampling), the model sees a different version each time.

---

## Part 1 — Preprocessing

The preprocessing pipeline is called the **Hybrid** pipeline because it chains two independent contrast techniques: Ben Graham enhancement followed by LAB-CLAHE. Each step is explained below in the order it runs.

### Step 1 — Crop the fundus (`crop_fundus`)

**What it does:**  
APTOS images are fundus photographs — a circular retinal image surrounded by a black background. The size of that black border varies significantly between imaging devices: some images are nearly full-frame, others have thick black borders. Before resizing, the border is stripped away.

The green channel is used to find the extent of the retina (it carries the strongest signal for fundus tissue; the optic disc and vessels are bright green, the lesions are dark green). Any row or column where all green-channel pixels are ≤ 7 is considered border and discarded. The result is a tight bounding box around the retinal disc.

**Why it matters:**  
Without this step, two images of the same eye at the same severity can appear very different just because of camera framing. After cropping, the retina fills the frame consistently, so the resize step in Step 2 maps equivalent anatomical regions to equivalent pixel positions.

**Parameters:**  
- `threshold = 7` — pixel values ≤ 7 in the green channel are treated as background. This is robust to minor JPEG compression noise at the border.

---

### Step 2 — Resize to 224×224 (`cv2.resize` with INTER_LANCZOS4)

**What it does:**  
The cropped image (which is now roughly square, but variable in size) is resized to exactly 224×224 pixels.

**Why LANCZOS4 and not bilinear?**  
LANCZOS4 uses an 8-tap sinc-based kernel. When downsampling a large fundus image (e.g. 1500×1500) to 224×224, bilinear interpolation averages nearby pixels in a way that blurs sharp boundaries. LANCZOS4 applies a higher-order filter that better preserves edges — meaning microaneurysms (which are 3–6px at this resolution) survive the resize rather than being smeared into the background.

---

### Step 3 — Ben Graham contrast enhancement (`ben_graham`)

**What it does:**  
Applies the formula:

```
enhanced = 4 × image − 4 × GaussianBlur(image, σ=10) + 128
```

The Gaussian blur with σ=10 computes a smoothed version of the image that represents the *local mean illumination* — the slow, large-scale variation in brightness caused by the angle of the illumination source and the curvature of the retina. Subtracting 4× this from 4× the original removes that large-scale variation. Adding 128 re-centres the result at mid-grey.

**Why this matters:**  
Fundus images from different cameras and patients have very different global brightness and colour balance — some are reddish-orange, some greenish, some nearly desaturated. The Ben Graham transform removes all of this large-scale variation and re-centres every image at the same mid-grey baseline. After this step, the *fine-grained local structure* (microaneurysm dots, vessel patterns, hard exudates) stands out much more clearly against the uniform background, because the large-scale illumination gradient is gone.

The result is clipped to [0, 255] and cast back to uint8 before the next step.

**Parameters:**  
- `sigmaX = 10` — the Gaussian blur radius. This is large enough to capture the slow illumination gradient but small enough not to blur vessel patterns.

---

### Step 4 — LAB-CLAHE contrast enhancement (`clahe_lab`)

**What it does:**  
Converts the image from RGB to CIELAB colour space, applies CLAHE (Contrast-Limited Adaptive Histogram Equalization) only to the L (lightness) channel, then converts back to RGB.

**What CLAHE does:**  
CLAHE divides the image into an 8×8 grid of tiles (at 224px, each tile is 28×28 pixels). Within each tile it computes a histogram of pixel values and remaps them to equalise the histogram — boosting the contrast of detail in that local region. The "contrast-limited" part caps how aggressively any histogram bin can be boosted, preventing noise amplification in very dark or very uniform regions.

**Why the L channel of LAB and not HSV or RGB?**  
- If you apply CLAHE to all three RGB channels independently, you change the colour balance of the image (shifting saturation and hue) — lesion colours like the deep red of haemorrhages or the yellow-white of exudates can shift to unrecognisable values.
- HSV's V channel is perceived brightness, but it is not perceptually linear — a unit change in V at high brightness is much less visible than the same unit change at low brightness. This makes CLAHE's clip limit behave inconsistently across the image.
- CIELAB's L channel is *perceptually uniform lightness* — it is specifically designed so that equal numerical changes correspond to equal perceived changes in brightness. CLAHE on L therefore applies consistent contrast enhancement everywhere, never over-boosting already bright regions, and the a/b colour channels (which carry lesion colours) are completely untouched.

**Parameters:**  
- `clip_limit = 2.0` — moderate clip. Higher values boost contrast more aggressively but amplify noise; 2.0 is the standard value used in medical imaging.
- `tile = 8` — 8×8 tile grid (28×28px tiles at 224px).

---

### Step 5 — Circle mask (`apply_circle_mask`)

**What it does:**  
The fundus image is circular but the image file is square. After preprocessing, the four corners of the square (outside the circular camera field-of-view) are set to exactly 0 (pure black).

The mask is the largest circle that fits inside the square: centred at (h/2, w/2) with radius = min(h/2, w/2) − 1.

**Why it matters:**  
Without masking, the corner regions contain residual artefacts from the camera lens or from the resize/contrast steps. These artefacts are different for every image and every imaging device. If the model is allowed to see them, it can learn to use corner texture as a proxy for image source, which is a spurious feature not related to DR severity. Setting corners to black removes this possibility entirely. It also makes the circular boundary consistent across all images — the model always sees a circle of retinal tissue on a black background, regardless of the original camera framing.

---

### Complete pipeline summary

| Step | Operation | Key parameter | Purpose |
|------|-----------|---------------|---------|
| 1 | Crop black border | green threshold = 7 | Consistent framing |
| 2 | Resize 224×224 | INTER_LANCZOS4 | Preserve fine detail |
| 3 | Ben Graham | σ = 10 | Remove global illumination gradient |
| 4 | LAB-CLAHE | clip=2.0, tile=8 | Boost local lesion contrast |
| 5 | Circle mask | fill = 0 | Remove corner artefacts |

The output is a 224×224 uint8 RGB image saved as a PNG to disk. This cached file is what the dataset class loads during training.

---

## Part 2 — Augmentation

Augmentation runs **inside the training loop**, not during preprocessing. Every time a cached image is loaded for a training batch, a random augmentation is applied on the fly. The validation set receives **no augmentation** — only normalisation.

### Grade-aware dispatch

The key design decision is that different DR grades should not receive the same augmentation. The dataset class chooses the augmentation pipeline based on the grade of each image:

```
Grade 0 (No DR)        → mild augmentation
Grade 1 (Mild NPDR)    → mild augmentation
Grade 2 (Moderate NPDR) → moderate augmentation
Grade 3 (Severe NPDR)  → strong augmentation
Grade 4 (Proliferative DR) → strong augmentation
```

The reason for this split is explained in the pipeline descriptions below.

---

### Common geometric block (shared by all three pipelines)

All three pipelines start with the same set of geometric transforms, with slightly different parameters depending on the pipeline strength.

| Transform | What it does | Parameters (mild / moderate / strong) |
|-----------|-------------|--------------------------------------|
| HorizontalFlip | Mirrors the image left-right | p=0.5 for all |
| VerticalFlip | Mirrors the image top-bottom | p=0.5 for all |
| RandomRotate90 | Rotates by 0°, 90°, 180°, or 270° | p=0.5 for all |
| Rotate | Rotates by any angle 0–360° | p=0.7 / 0.85 / 0.90 |
| Affine | Zoom and translate | scale ±5% / ±8% / ±10%, translate ±2% |
| CircleMaskTransform | Re-applies circle mask | p=1.0 for all |

**Why full 360° rotation is safe for fundus images:**  
A retinal photograph has no natural orientation — the retina is a sphere and the camera can be positioned at any angle. Severe DR lesions (haemorrhages, neovascularisation) appear at any position on the fundus. Allowing the model to see any rotation teaches it that grade is independent of orientation, which is clinically correct.

**Why CircleMaskTransform runs after geometric transforms:**  
Rotation and affine translation shift image content relative to the frame. If the original circle mask is still applied, it may now cut into retinal content that has been rotated into a corner, or leave corner artefacts visible that were rotated into the circle. Re-applying the mask after the geometric transform ensures the circle boundary is always clean.

---

### Pipeline 1 — Mild (Grades 0 and 1)

**Photometric transforms (applied with 50% probability, one chosen at random):**

| Transform | Parameters | Purpose |
|-----------|-----------|---------|
| RandomBrightnessContrast | brightness ±0.08, contrast ±0.08 | Simulate different cameras |
| HueSaturationValue | hue ±5, sat ±10, val ±8 | Small colour shift |

**Deliberately excluded transforms:**

| Transform | Why excluded |
|-----------|-------------|
| GaussianBlur | A blur kernel larger than 3px can erase a microaneurysm (3–6px red dot). Even 3px softens the boundary enough to reduce detectability. |
| Sharpen | Mild sharpening is safe but adds unnecessary risk for no clear gain at Grade 0 (which has no lesions to sharpen). |
| RGBShift | Larger channel shifts can collapse the red-on-orange contrast that distinguishes microaneurysms from background. |

**Why Grade 1 requires special treatment:**  
Grade 1 (Mild NPDR) is defined by microaneurysms *only* — typically 1 to 5 small red dots, 3–6 pixels wide at 224px resolution. These are the smallest diagnostic features in the entire dataset. If any photometric transform shifts the colour enough to reduce red-orange contrast, or if any blur transform softens the dot boundaries, the model receives a Grade 1 label on an image that visually looks identical to Grade 0. This poisons the training signal for the first CORAL threshold (the boundary between No DR and any DR). By restricting augmentation to very conservative limits, the microaneurysm signal is preserved.

---

### Pipeline 2 — Moderate (Grade 2)

**Photometric transforms (applied with 65% probability, one chosen at random):**

| Transform | Parameters | Purpose |
|-----------|-----------|---------|
| RandomBrightnessContrast | brightness ±0.12, contrast ±0.12 | Wider range than mild |
| HueSaturationValue | hue ±8, sat ±15, val ±10 | Moderate colour shift |
| RGBShift | r/g/b ±8 | Independent channel shift |

**Additional transform:**

| Transform | Parameters | Probability |
|-----------|-----------|-------------|
| Sharpen | alpha 0.10–0.25 | 25% |

**Why sharpen is included here but not in mild:**  
Grade 2 (Moderate NPDR) has multiple lesion types: microaneurysms, dot and blot haemorrhages, hard exudates, and sometimes soft exudates. These are larger features with distinct boundaries. Mild sharpening (alpha 0.10–0.25) makes haemorrhage edges and exudate borders slightly crisper — it enhances the features rather than erasing them.

**Deliberately excluded from moderate:**

| Transform | Why excluded |
|-----------|-------------|
| GaussianBlur | Haemorrhage boundaries are diagnostically important for distinguishing G2 from G3 (which has many more and denser haemorrhages). Blur softens these boundaries. |

---

### Pipeline 3 — Strong (Grades 3 and 4)

**Photometric transforms (applied with 70% probability, one chosen at random):**

| Transform | Parameters | Purpose |
|-----------|-----------|---------|
| RandomBrightnessContrast | brightness ±0.15, contrast ±0.15 | Full range |
| HueSaturationValue | hue ±10, sat ±15, val ±12 | Full colour variation |
| RGBShift | r/g/b ±10 | Full channel shift |

**Blur/Sharpen (applied with 30% probability, one chosen at random):**

| Transform | Parameters | Why safe at G3/G4 |
|-----------|-----------|-------------------|
| GaussianBlur | kernel (3,3) — 3px max | G3/G4 lesions are tens of pixels wide; a 3px blur cannot erase them |
| Sharpen | alpha 0.15–0.35 | Enhances vessel boundaries and haemorrhage morphology |

**Why Grade 3 and 4 can tolerate stronger augmentation:**  
Severe (Grade 3) and Proliferative (Grade 4) DR are characterised by extensive, large-scale lesions: dense haemorrhage fields, venous beading, and new vessel formation (neovascularisation). These structures cover large areas of the fundus and are robust to moderate photometric variation. A 3px Gaussian blur on a 50px haemorrhage barely changes its appearance. Wider colour shifts simulate the range of imaging conditions the model may encounter at deployment.

---

### Normalisation (applied to all splits including validation)

After the augmentation transforms (or with no augmentation for validation), every image is normalised using **ImageNet statistics**:

```
mean = [0.485, 0.456, 0.406]   (R, G, B)
std  = [0.229, 0.224, 0.225]   (R, G, B)

normalised = (pixel / 255 − mean) / std
```

This is applied because DenseNet121 was pre-trained on ImageNet with these exact statistics. The backbone's learned weights expect inputs in this range. Applying the same normalisation ensures the pre-trained feature activations are in their calibrated range when training starts, rather than requiring the model to adapt from scratch.

The output of this step is a `torch.Tensor` of shape `(3, 224, 224)` with float32 values (typically in the range −2.5 to +2.5).

---

### Validation augmentation

During validation and test-time, **no random transforms are applied**. The only operation is normalisation:

```
Cached PNG  →  Normalize(ImageNet mean/std)  →  Tensor
```

This ensures validation metrics are deterministic and comparable across epochs. Any randomness in validation would make QWK scores noisy and harder to interpret.

---

## Part 3 — Test-Time Augmentation (TTA)

At inference time (after training is complete), each validation image is passed through the model three times with different fixed transforms:

1. Original image (no flip)
2. Horizontal flip
3. Vertical flip

The model outputs 4 CORAL logits for each of the three versions. The three sigmoid probability vectors are averaged, and the predicted grade is computed from the averaged probabilities. This reduces prediction variance — flipping is geometrically valid for fundus images (the grade does not change with orientation), so the three predictions should agree on grade but may differ on borderline cases, and averaging resolves those borderline cases more reliably than any single pass.

---

## Part 4 — Full data flow diagram

```
data/raw/train_images/
    <id_code>.png  (original, variable size ~1500×1500)
          │
          ▼  [preprocess_and_cache — done ONCE]
          │
    1. crop_fundus()           remove black border
    2. resize(224, LANCZOS4)   fixed size, high-quality downscale
    3. ben_graham(σ=10)        remove illumination gradient
    4. clahe_lab(clip=2, t=8)  boost local contrast in L channel
    5. apply_circle_mask()     zero corners outside FOV
          │
          ▼
data/processed/hybrid_224/
    <id_code>.png  (224×224, standardised, cached)
          │
          ▼  [Dataset.__getitem__ — done EVERY batch]
          │
    Grade 0–1 → mild aug    │
    Grade 2   → moderate    │  random transforms (see above)
    Grade 3–4 → strong aug  │
          │
    Normalize(ImageNet)      always, for all grades
    ToTensorV2()             HWC uint8 → CHW float32
          │
          ▼
    Tensor shape: (3, 224, 224)
    dtype: float32
    value range: approx −2.5 to +2.5
          │
          ▼
    DenseNet121 backbone → CORAL head → 4 threshold logits
```
