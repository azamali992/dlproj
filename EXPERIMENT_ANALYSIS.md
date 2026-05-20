# APTOS 2019 DR Grading — Full Experiment Analysis
**Project:** DenseNet121 + CORAL ordinal regression for Diabetic Retinopathy grading  
**Last experiment:** exp_v5 (QWK 0.8823 single / 0.8901 ensemble)  
**Written:** 2026-05-18

---

## Table of Contents
1. [What CORAL is and why it matters](#1-what-coral-is-and-why-it-matters)
2. [Baseline vs. v5 — side-by-side comparison](#2-baseline-vs-v5--side-by-side-comparison)
3. [Every change made, version by version](#3-every-change-made-version-by-version)
4. [Augmentation deep dive — the black patches problem](#4-augmentation-deep-dive--the-black-patches-problem)
5. [Class imbalance — current state and options](#5-class-imbalance--current-state-and-options)
6. [What still can be done (prioritised)](#6-what-still-can-be-done-prioritised)

---

## 1. What CORAL is and why it matters

DR grading is not a classification problem in the standard sense — it is *ordinal regression*. The five grades are not arbitrary categories like "cat vs. dog"; they form a strict clinical severity progression:

```
Grade 0 (No DR) < Grade 1 (Mild) < Grade 2 (Moderate) < Grade 3 (Severe) < Grade 4 (Proliferative)
```

A model that predicts Grade 2 when the true label is Grade 3 makes a much smaller error than a model that predicts Grade 0 for Grade 3. The official APTOS metric, **Quadratic Weighted Kappa (QWK)**, penalises predictions in proportion to the *square* of how far off they are — so a 2-grade error costs 4× as much as a 1-grade error.

CORAL (Consistent Ordinal Regression via Adjacency Loss) exploits this structure by framing grading as a cascade of binary questions:

| Threshold | Question | Binary target |
|-----------|----------|---------------|
| k=0 | Is grade > 0? (anything beyond normal?) | 1 for G1–G4, 0 for G0 |
| k=1 | Is grade > 1? (moderate or worse?) | 1 for G2–G4, 0 for G0–G1 |
| k=2 | Is grade > 2? (severe or worse?) | 1 for G3–G4, 0 for G0–G2 |
| k=3 | Is grade > 3? (proliferative?) | 1 for G4, 0 for G0–G3 |

The model outputs 4 logits — one per threshold. The predicted grade = count of how many thresholds exceed 0.5. This is a fundamentally better fit for a metric that cares about distance from the truth.

---

## 2. Baseline vs. v5 — side-by-side comparison

### Numbers

| Metric | exp_coral_focal (baseline CORAL) | exp_v5 (latest) | Delta |
|--------|----------------------------------|-----------------|-------|
| Best QWK (single model) | 0.8089 | **0.8823** | +0.0734 (+9.1%) |
| Ensemble QWK | — | **0.8901** | — |
| Best epoch | 27 | 17 | |
| Val accuracy | 67.8% | 76.4% | +8.6pp |
| Trainable params | 5,700,164 | 6,622,916 | +922k |
| Grade 0 recall | ~0.0 (metrics broken in log) | 93.4% | |
| Grade 1 recall | ~0.0 | **77.0%** | |
| Grade 2 recall | ~0.0 | 62.5% | |
| Grade 3 recall | ~0.0 | 66.7% | |
| Grade 4 recall | ~0.0 | 39.0% | |

> Note: exp_coral_focal's per-class metrics in summary.json are all 0.0 due to a logging bug in the v1 training script — the QWK of 0.8089 is the correct value. The best comparable v1 per-class data comes from exp_coral_focal_v2 which is the first version with correct logging.

### Architecture differences

| Component | Baseline CORAL (v1) | v5 |
|-----------|--------------------|----|
| Backbone | DenseNet121 (pretrained) | DenseNet121 (pretrained) |
| CORAL head | 1× Linear(1024→1, no bias) + 4 learned biases | 4× independent Linear(1024→1, bias=True) |
| Loss | CORAL BCE + Focal (γ=2) | CORAL BCE + Focal (γ=2) + ordinal reg (λ=0.1) |
| Label smoothing | None | None (0.0, explicit) |
| Alpha weighting | On balanced df (near-uniform, useless) | On original imbalanced df (meaningful weights) |
| Frozen blocks | denseblock1 + denseblock2 | denseblock1 only |
| Scheduler | CosineAnnealingLR (no warmup) | OneCycleLR (10% warmup, cosine decay) |
| Gradient clipping | None | max_norm = 1.0 |
| Augmentation | Standard single pipeline | Grade-aware: mild / moderate / strong |
| Checkpointing | Single best model | Top-3 by QWK |
| Ensemble | No | Average of top-3 checkpoints |
| TTA | No | H-flip + V-flip + original (3 variants) |
| Cache directory | ./data/preprocessed_hybrid | data/processed/hybrid_224 |

---

## 3. Every change made, version by version

### v1 — train_coral_focal.py (Baseline CORAL)
**QWK: 0.8089**

This is the starting point. DenseNet121 with a CORAL head. The head design is:
```python
self.fc   = nn.Linear(in_features, 1, bias=False)   # shared weight vector
self.bias = nn.Parameter(torch.zeros(4))             # 4 threshold biases
output    = self.fc(x) + self.bias                   # broadcast to [B, 4]
```
All four thresholds share a single linear projection. The CORAL paper proves this preserves rank-monotonicity under plain BCE, but that proof breaks down the moment you add focal weighting.

**Problems in v1:**
- Learning rate 1e-4 is too conservative; CosineAnnealingLR has no warmup so the first few batches see full LR and can cause unstable threshold updates.
- denseblock1 + denseblock2 frozen — this is too aggressive. denseblock2 and denseblock3 are where DenseNet learns medium-level texture features (lesion shapes, haemorrhage boundaries). Freezing them transfers no fundus-specific knowledge.
- No gradient clipping — CORAL's shared weight can receive conflicting signals from different thresholds and oscillate.
- Single best checkpoint saved — no ensemble diversity.
- No TTA.
- Alpha computed on balanced dataframe — after oversampling all classes to ~800, the weights are nearly uniform [0.79, 0.87, 0.87, 1.0] and do almost nothing useful.

---

### v2 — train_coral_focal_v2.py
**QWK: 0.8946 (best single model in the whole project)**

This was the biggest single jump — a 10.6% relative improvement over v1. The key changes:

**1. OneCycleLR scheduler**  
CosineAnnealingLR starts the model at the peak learning rate immediately. For CORAL, this means the threshold biases can make large jumps in the first few batches before the backbone has stabilised, causing the optimizer to overshoot. OneCycleLR starts at LR/25 = 1.2e-5, ramps to peak at 10% of training, then cosine-decays to LR/25000 = 1.2e-8. The warmup phase gives the model time to settle before the backbone updates start compounding with the threshold updates. This eliminated the zigzag pattern in the early training loss that was present in v1.

**2. Label smoothing (smooth=0.05)**  
CORAL's targets are hard binary labels (0 or 1 per threshold). When the model drives a logit to +8 or −8, the BCE gradient becomes essentially zero — the model has "saturated" at that threshold and stops learning from those examples. Label smoothing replaces hard targets with soft ones: 1→0.95, 0→0.025. This keeps logits from going to ±∞ and maintains a small but nonzero gradient everywhere. Concretely, it prevents the G0 threshold (which is easy — 1805 samples) from saturating while the G3/G4 threshold (hard — only 236 samples) is still trying to learn.

**3. Gradient clipping (max_norm=1.0)**  
A batch that contains many Grade 3 samples can produce large gradients on the G2/G3 threshold while producing tiny gradients on the G0/G1 threshold. Without clipping, the optimizer takes a huge step on the shared weight vector in response to G3 examples, which then destabilises the G0/G1 boundary. Clipping caps the total gradient norm at 1.0, preventing any single batch from taking a step large enough to break a threshold that was previously well-calibrated.

**4. Frozen only denseblock1**  
denseblock1 learns low-level edges and gradients — features directly transferable from ImageNet. denseblock2 and 3 learn mid- and high-level textures (lesion morphology, vessel patterns) that require fundus-specific training. Unfreezing them gave the backbone the ability to specialise its representations for DR.

**5. Top-3 checkpointing + ensemble**  
The three best QWK checkpoints during training (not necessarily from consecutive epochs) are saved. Ensemble averages their sigmoid probabilities before applying the 0.5 threshold. This reduces variance — each checkpoint may have a slightly different confusion boundary due to the random order of batches.

**6. TTA (Test-Time Augmentation)**  
At validation/test time, three versions of each image are passed through the model: the original, a horizontal flip, and a vertical flip. A retinal image of any grade looks structurally the same regardless of orientation (the optic disc position changes, but lesion distribution does not), so averaging all three logit sets reduces prediction noise.

---

### v3 — train_coral_focal_v3.py + preprocess_v3.py
**QWK: 0.8920 (single) / 0.8848 (ensemble)**

The main change was moving to a research-backed preprocessing pipeline and 512×512 input.

**Preprocessing pipeline improvements:**
- *Ben Graham correction:* v1 used `img - blur + 128` (additive). The correct formula is `4*img - 4*blur + 128` (scaled subtraction). The 4× coefficient amplifies the high-frequency content (lesions, vessels) relative to the low-frequency background illumination gradient.
- *LAB-CLAHE vs. HSV-CLAHE:* CLAHE in HSV space operates on the V (value/brightness) channel, which is not perceptually uniform — a unit change in V at high brightness is less visible than the same change at low brightness. CIELAB's L channel is perceptually linearised: equal L differences correspond to equal perceived lightness differences. This makes CLAHE's clip limit behave consistently across different regions of the image, preventing over-amplification of noise in bright areas.
- *Circle mask:* After preprocessing, pixels outside the circular camera FOV are zeroed. This removes black/dark border artefacts that vary wildly between imaging devices — without this, the model can learn to use border texture as a proxy for image quality/equipment, which is not a DR signal.
- *512×512 input:* Microaneurysms (the defining lesion of Grade 1 DR) are 3–6px at 224px resolution and 7–13px at 512px. Preserving them at training time is directly tied to Grade 1 recall.

**Why ensemble QWK dropped from v2:**  
512px images require batch size 8 (vs. 16 at 224px on RTX 2050). Smaller batches produce higher-variance gradient estimates, meaning the top-3 checkpoints show more variation in what they got right and wrong — useful for ensemble diversity in theory, but in practice the lower batch size also makes training less stable, and the 3 checkpoints may be correlated to the same local minima.

---

### v4 — train_coral_focal_v4.py + preprocess_grade_aware_aug.py
**QWK: not separately logged (incorporated into correct_cache)**

The key insight: a single augmentation pipeline cannot be optimal for all grades simultaneously.

**Why this matters — the microaneurysm problem:**  
Grade 1 DR is defined by microaneurysms *only*. At 224px, these are 3–6px red dots. The previous "strong" augmentation pipeline included:
- GaussianBlur(kernel up to 7×7): a 7px kernel can fully erase a 5px dot
- CoarseDropout with 20×20 holes: with only 2–3 microaneurysms in the image, one 20×20 hole can remove the *only* diagnostic feature
- HueSaturationValue with sat±25: microaneurysms are distinguished by their red-on-orange contrast; large saturation shifts collapse this contrast

When these transforms run on a Grade 1 image, the model receives a Grade 1 label attached to an image that *looks* identical to Grade 0. This poisons the training signal for the k=0 threshold (G0 vs. G1+).

**Grade-aware dispatch:**
```
Grade 0–1 → mild aug:   geometry only + tiny photometric jitter (no blur, no dropout)
Grade 2   → moderate:   fuller geometry + moderate color jitter (no blur, no dropout)
Grade 3–4 → strong aug: full geometry + color jitter + capped GaussianBlur(3,3) + CoarseDropout(8–12px)
```
For Grade 3–4, lesions are large haemorrhages and neovascularisation covering tens of pixels. A 12px CoarseDropout hole on a Grade 4 image erases a small portion of many lesions — the remaining lesion signal is more than sufficient to identify the grade. For Grade 1, the same hole would erase the entire diagnostic content.

---

### exp_correct_cache — train_coral_focal_correct_cache.py
**QWK: 0.8900 (single) / 0.8932 (ensemble)**

This was a fix-and-consolidate pass that addressed several accumulated bugs without changing the model architecture:

**Bug 1 — wrong cache directory:**  
v3 saved preprocessed images to `./data/preprocessed_hybrid` (the v1 path). The training script was configured to read from `data/processed/hybrid_224` — so it was reading v1-preprocessed (incorrect Ben Graham formula, HSV-CLAHE, no circle mask) images while believing it was reading v3-preprocessed images. This silently negated all the preprocessing improvements.

**Bug 2 — label smoothing conflicts with focal loss:**  
Focal loss already down-weights easy examples (high pt → small (1−pt)^γ) and up-weights hard examples (low pt → large (1−pt)^γ). Label smoothing prevents probabilities from reaching exactly 0 or 1, which modulates the focal weight. The two mechanisms interact: label smoothing increases pt for hard examples slightly, which reduces their focal weight slightly, which reduces the intended amplification. Setting smooth=0.0 lets focal loss work as designed.

**Bug 3 — alpha on balanced dataframe:**  
After oversampling to 800 per class, the alpha weights were computed on the balanced df. With equal class counts, S_k ≈ N/2 for all thresholds, M_k ≈ N/2, and all alpha ≈ 1.0. The per-threshold imbalance correction was effectively disabled.

---

### v5 — train_coral_focal_v5.py + losses_v5.py + models_v5.py
**QWK: 0.8823 (single) / 0.8901 (ensemble)**

The architectural innovation: independent per-threshold projections.

**Motivation — the shared weight limitation:**  
In v2's CORAL head, a single weight vector w ∈ R^1024 produces the scalar logit for all four thresholds via `w·x + b_k`. The four thresholds are trying to differentiate:
- k=0: Microaneurysm presence (tiny red dots, 3–6px)
- k=1: Multiple microaneurysms + possibly haemorrhages (medium-scale features)
- k=2: Haemorrhage patterns, venous beading (larger-scale morphology)
- k=3: Neovascularisation, fibrous proliferation (vessel architecture)

These are categorically different visual features. A microaneurysm detector (good for k=0) has low correlation with a neovascularisation detector (needed for k=3). A single weight vector w must compromise — it will be suboptimal for at least some thresholds.

**The fix — 4 independent projections:**
```python
self.fc = nn.ModuleList([
    nn.Linear(1024, 1, bias=True)  # each learns its own feature combination
    for _ in range(4)
])
```
Each projection has its own 1024 weights and 1 bias. The k=0 projection can learn to activate on red-channel localised features; the k=3 projection can learn vessel architecture features. They don't interfere with each other.

**Ordinal consistency regularisation:**  
The CORAL shared-weight theorem guarantees rank-monotonicity (P(y>k) ≥ P(y>k+1)) for free, because the same w produces both logits and only the biases differ. With independent projections, this guarantee is lost — it's theoretically possible for k=1 to fire more confidently than k=0 on some input. The regularisation term penalises this:
```
reg = Σ_k ReLU(logit[k+1] − logit[k])
```
This is zero when thresholds are monotone and positive when a higher threshold fires more confidently than a lower one. λ=0.1 adds this to the main loss.

**Alpha on original imbalanced labels:**  
The raw APTOS distribution before balancing:
```
G0: 1805, G1: 370, G2: 999, G3: 193, G4: 295
```
Threshold weights from these counts:
```
threshold 0 (G0 vs rest): α ≈ 0.743
threshold 1 (G0+G1 vs rest): α ≈ 0.804
threshold 2 (G0–G2 vs G3+G4): α ≈ 0.971
threshold 3 (G0–G3 vs G4): α ≈ 1.000
```
Thresholds 2 and 3 (the hardest clinical boundaries) get ~25–35% more gradient weight. This is what the alpha mechanism was designed to do.

**Why v5 QWK (0.8823) is lower than v2 (0.8946) despite more sophisticated architecture:**  
1. More parameters to learn: 4×1024 = 4096 weights vs. 1×1024 = 1024 weights in the CORAL head. With only ~2900 training samples, the larger head has higher variance.
2. The ordinal regularisation (λ=0.1) adds a constraint on the loss landscape that may be slightly too strong, making some correct predictions that violate ordinality get penalised unnecessarily.
3. Unfreezing denseblock3 (which was correctly frozen in the broken run that showed G3 recall=0.38, and correctly unfrozen in v5) means more parameters are updating — more training epochs or a warmup phase tuned to the larger trainable parameter count may be needed.
4. v5 is architecturally more promising at scale, but v2 was already at a better point on the bias-variance tradeoff for the current dataset size.

---

## 4. Augmentation deep dive — the black patches problem

### What the aug_3.png image shows

Looking at the aug_3.png sample in `results/exp_v5/aug_samples/`, you see a circular fundus image with 1–2 small black rectangular patches *inside* the retinal area. These are not artefacts, imaging errors, or preprocessing bugs. They are **intentional CoarseDropout holes**.

### Root cause: CoarseDropout with fill=0

In `get_train_augmentation_strong()`:
```python
A.CoarseDropout(
    num_holes_range=(1, 3),
    hole_height_range=(8, 12),
    hole_width_range=(8, 12),
    fill=0,           # <-- pure black, same value as the circular mask background
    p=0.20,
)
```

CoarseDropout is the augmentation equivalent of Cutout (DeVries & Taylor, 2017). It randomly removes rectangular patches of the image and fills them with a constant value, forcing the model to make predictions based on partial information. This prevents over-reliance on any single spatial region — a known failure mode for dense lesion images where the model always looks at the same quadrant.

**Why fill=0 is the problem:**  
The retinal image already has a large black region — the circular mask background (everything outside the camera FOV). When CoarseDropout fills holes with 0 (pure black), the holes are visually indistinguishable from the background. A patch of black inside the circle looks like the circular mask boundary shifted inward. This is arguably fine during training (the model sees the context: the black patch is surrounded by retinal tissue, so it cannot be background), but it creates two subtle issues:

1. **The model may learn that black is OK inside the circle.** Real retinal images never have pure black regions inside the FOV — a black patch inside the retina would indicate a imaging artifact or data corruption. By training with fill=0, you're teaching the model that this is a normal pattern for severe DR images.

2. **The verify_augmentation function always uses get_train_augmentation_strong().** The aug_samples you see were generated by `verify_augmentation()` which picks the first image from the cache directory and applies the *strong* augmentation to it — regardless of that image's actual grade. So aug_3.png might be showing a Grade 0 or Grade 2 image with CoarseDropout applied, which would never happen in actual training (because Grade 0–2 images get mild/moderate aug which excludes CoarseDropout). The aug samples are misleading as a representation of what actually happens in training.

### Secondary source: Rotate with border fill

The `_spatial_block()` function includes:
```python
A.Rotate(limit=360, border_mode=cv2.BORDER_CONSTANT, fill=0, p=rotate_p)
```

After a large rotation (say 45°), the square image corners that were previously outside the circle are now rotated into new positions, and new corner regions are created that fall outside the original image boundary. These get filled with 0. The `CircleMaskTransform(p=1.0)` applied immediately after re-zeroes the entire region outside the inscribed circle — which handles this correctly for most rotation angles. However, if the rotation is large and the image has any content near the corners of the circle (the inscribed circle reaches the edges of the square), thin black slivers can appear at the boundary of the circle mask.

This is a minor issue and less visually prominent than CoarseDropout.

### Why this is less damaging than it looks

In actual training (not just verification):
- Grade 0 and Grade 1 images get `get_train_augmentation_mild()` → **no CoarseDropout at all**
- Grade 2 images get `get_train_augmentation_moderate()` → **no CoarseDropout**
- Only Grade 3 and Grade 4 images get the strong aug with CoarseDropout

Grade 3 and Grade 4 have dense, large-scale lesions covering significant fractions of the image. An 8–12px hole removes a tiny portion of many lesions simultaneously — the remaining signal is more than sufficient. In this context, CoarseDropout serves its intended purpose of preventing the model from fixating on a single lesion cluster.

### What should be fixed

**Fix 1 — Change CoarseDropout fill value:**
```python
# Instead of fill=0 (indistinguishable from background):
A.CoarseDropout(
    num_holes_range=(1, 3),
    hole_height_range=(8, 12),
    hole_width_range=(8, 12),
    fill=128,    # mid-grey: clearly different from both black background and red lesions
    p=0.20,
)
```
Using 128 makes the holes visually distinct from the circular mask, preventing the model from confusing dropped regions with the background boundary.

**Fix 2 — Fix verify_augmentation to be grade-aware:**
```python
# In verify_augmentation(), select images per grade and apply the correct aug:
for grade, aug_fn in [(0, get_train_augmentation_mild),
                      (2, get_train_augmentation_moderate),
                      (4, get_train_augmentation_strong)]:
    grade_files = [f for f in files if grade_labels.get(f.stem) == grade]
    if grade_files:
        show_augmented(grade_files[0], aug_fn(), save_dir, prefix=f'grade{grade}')
```
This way the aug_samples folder shows the actual augmentation each grade will experience.

**Fix 3 — Reduce or remove CoarseDropout:**  
p=0.20 is already conservative (80% of Grade 3/4 images never see it). Given that Grade 4 recall is already the weakest point (39.0%) and the model needs to learn neovascularisation features, adding dropout on those rare images adds variance without clear benefit. Consider reducing to p=0.10 or removing it.

---

## 5. Class imbalance — current state and options

### The distribution problem

Raw APTOS 2019 training set:
```
Grade 0: 1805 images  (49.3% of dataset)
Grade 1:  370 images  (10.1%)
Grade 2:  999 images  (27.3%)
Grade 3:  193 images   (5.3%)
Grade 4:  295 images   (8.1%)
```

Grade 0 outnumbers Grade 3 by **9.4:1**. The model naturally optimises accuracy, which can be maximised by correctly classifying Grade 0 at the cost of ignoring Grade 3.

### What has already been tried

| Technique | Where | Effect |
|-----------|-------|--------|
| Focal loss (γ=2.0) | All versions | Down-weights easy examples (Grade 0 dominance), up-weights hard examples (Grade 3). Partial mitigation. |
| Alpha weighting | All versions | Adds extra gradient weight to thresholds with higher imbalance. Near-uniform in v1–v4 (applied to balanced df). Meaningful in v5. |
| Oversampling to 800/class | v3+ | Duplicate Grade 1/3/4 images in the training dataframe. Augmentation means duplicates look different at each epoch. |
| Grade-aware augmentation | v4+ | Mild aug for Grade 0/1 (prevents Grade 0 from being too easy), strong aug for Grade 3/4 (more variety for rare classes). |

### Current per-class performance (exp_v5)

| Grade | Precision | Recall | F1 | Clinical meaning |
|-------|-----------|--------|----|------------------|
| 0 (Normal) | 99.1% | 93.4% | 96.1% | Almost perfect |
| 1 (Mild) | 45.6% | **77.0%** | 57.3% | Recall improved massively vs v2 (40.5%) |
| 2 (Moderate) | 77.2% | 62.5% | 69.1% | Decent but overcautious |
| 3 (Severe) | 33.8% | **66.7%** | 44.8% | Recall improved vs v2 (46.2%) |
| 4 (Proliferative) | 79.3% | **39.0%** | 52.3% | Grade 4 recall degraded vs v2 (55.9%) |

The independent projections in v5 improved Grade 1 recall dramatically (40.5% → 77.0%) and Grade 3 recall (46.2% → 66.7%), but at the cost of Grade 4 recall (55.9% → 39.0%). This trade-off suggests the k=3 projection (G3 vs G4 boundary) is not yet calibrated — possibly because Grade 4 has so few samples (295 raw, ~236 in the training split after 80/20 split).

### What can still be done about imbalance (in order of impact)

**Option A — Collect / augment more Grade 3 and Grade 4 data**  
This is what you already identified as the most direct fix. Grade 3 (193 images) and Grade 4 (295 images) are critically underrepresented. Options:
- Download the full Kaggle 2019 APTOS dataset including the test set labels (if available)
- Use the older Kaggle 2015 DR dataset (88,702 images) and cross-map the 5-class labels
- Apply domain-specific synthetic augmentation (Mixup within grade, or CycleGAN-based disease progression synthesis — though this is complex)

**Option B — Increase oversample target for rare classes**  
Currently oversample to 800/class. Grade 3 goes from 193 → 800 (4.1× duplication). With aggressive but correct augmentation, even 1500/class could be tried. The risk is overfitting to the small set of real Grade 3 images — the model memorises which specific eyes have Grade 3 rather than learning the visual features.

**Option C — Class-weighted loss (already partially done via alpha)**  
The alpha weighting in v5 is meaningful but not extreme. A more aggressive approach: set alpha proportional to 1/frequency rather than the square-root formula. This would give Grade 3/4 thresholds ~3–5× more weight than Grade 0/1. The risk: training becomes unstable because rare-class batches dominate the loss.

**Option D — Threshold tuning post-training**  
Instead of using 0.5 as the uniform threshold for all CORAL boundaries, tune each threshold independently on the validation set. For instance, lower the k=2 decision threshold from 0.5 to 0.4 to capture more Grade 3/4 predictions. This can be done after training with a simple grid search.

**Option E — Ordinal-aware sampling**  
Instead of random oversampling, use boundary-aware sampling: oversample specifically the cases near the decision boundary (model-predicted probability between 0.3 and 0.7). These are the hardest examples and training on more of them improves boundary calibration. Implement by running inference on the training set, computing per-sample confidence, and building a new sampling distribution from the confidence scores.

---

## 6. What still can be done (prioritised)

### Tier 1 — Quick wins (minimal code changes, likely to help)

**1. Fix CoarseDropout fill value (augmentation)**  
Change `fill=0` to `fill=128` in `get_train_augmentation_strong()`. This separates dropped regions from the circular mask background and prevents the model from conflating dropout holes with image boundaries. One-line change in `preprocess_grade_aware_aug.py`.

**2. Fix verify_augmentation to be grade-aware**  
Currently saves strong aug on arbitrary images. Change to save one sample per grade using the correct aug for that grade. This gives an accurate picture of what each grade actually experiences in training.

**3. Tune CoarseDropout probability downward**  
Reduce p=0.20 to p=0.10 for Grade 3/4. Grade 4 recall dropped from 55.9% (v2) to 39.0% (v5), possibly because v5 simultaneously has more augmentation variation (from unfreezing more blocks + new head) AND CoarseDropout removing some Grade 4 signal. Reducing dropout gives the model more signal from rare Grade 4 samples.

**4. Post-training threshold tuning**  
Run a sweep on the validation set: for each of the 4 CORAL thresholds, try decision boundaries in [0.3, 0.4, 0.45, 0.5, 0.55, 0.6] and pick the combination that maximises validation QWK. This requires no retraining — just loading the best checkpoint and sweeping thresholds.

### Tier 2 — Moderate changes (new training run required)

**5. Train for more epochs (40–50 instead of 30)**  
v5's best epoch was 17/30 — suggesting the model found its optimum early and then slightly overfit. Increasing epochs with a longer warmup (15% instead of 10%) may allow the larger trainable parameter space (denseblock2+3 now unfrozen) to converge properly.

**6. Increase target_per_class to 1200 for rare grades**  
Specifically for Grade 1 (370 raw → 1200, 3.2× duplication), Grade 3 (193 → 1200, 6.2× duplication). Combined with grade-aware mild augmentation, this means each "duplicate" of a Grade 3 image receives a different strong augmentation every epoch, producing genuine variety.

**7. Try EffNet or ConvNeXt backbone**  
DenseNet121 was top-tier in 2017. EfficientNet-B4 (from 2019, same era as APTOS) and ConvNeXt-Small (2022) provide stronger feature extractors with similar parameter counts. The `timm` library (already in requirements.txt) makes this a 5-line swap.

**8. Tune lambda_ord in v5**  
The ordinal regularisation at λ=0.1 may be too strong, preventing the independent projections from specialising. Try λ ∈ {0.0, 0.01, 0.05, 0.1} to find the sweet spot. λ=0.0 gives pure independent projections (no monotonicity guarantee), λ=0.1 is the current setting.

### Tier 3 — Significant research (multi-week effort)

**9. Get more data (Grade 3 and 4)**  
The single highest-leverage improvement. The 2015 Kaggle DR dataset has 88,702 images across 5 grades with a different but mappable class schema. Even if the preprocessing pipeline and label noise are not perfect, 5× more training data for rare classes will outweigh any architectural improvement.

**10. Semi-supervised learning with unlabelled data**  
If additional unlabelled fundus images are available, a self-supervised pre-training step (SimCLR or DINO on fundus images) will produce a backbone that understands fundus structure before the CORAL head is attached. The CORAL fine-tuning then requires fewer labelled examples to reach the same accuracy.

**11. Mixup between adjacent grades**  
Standard Mixup (interpolate two images and their labels) works poorly for ordinal regression because fractional CORAL targets are ambiguous. Ordinal Mixup (interpolate two Grade k images and two Grade k+1 images at ratio α) creates "soft" examples near the boundary. This is theoretically the most targeted fix for the boundary confusion problem.

**12. Multi-scale ensemble**  
Train two models: one at 224px (current) and one at 512px (v3). Ensemble their predictions. The 224px model learns global structure; the 512px model captures microaneurysm detail. Their errors should be partially uncorrelated, giving a higher QWK ensemble than two same-scale models.

---

## Summary of results across all experiments

| Experiment | Model | Best QWK | Ensemble QWK | Key contribution |
|------------|-------|----------|--------------|-----------------|
| exp_coral_focal | DenseNet121 + CORAL v1 | 0.8089 | — | CORAL baseline |
| exp_coral_focal_resnet18 | ResNet18 + CORAL | 0.8207 | — | Weaker backbone |
| exp_finetuned_resnet18 | ResNet18 fine-tuned | 0.8236 | — | Freeze + finetune |
| **exp_coral_focal_v2** | DenseNet121 + CORAL v2 | **0.8946** | 0.8946 | OneCycleLR, TTA, ensemble, label smoothing |
| exp_coral_focal_v3 | DenseNet121 + CORAL v2 (512px) | 0.8920 | 0.8848 | Research preprocessing |
| exp_correct_cache | DenseNet121 + CORAL v2 (fixed) | 0.8900 | **0.8932** | All bugs fixed, best ensemble |
| **exp_v5** | DenseNet121 + CORAL v5 | 0.8823 | 0.8901 | Independent projections, meaningful alpha |

**Best single model:** exp_coral_focal_v2 (QWK 0.8946)  
**Best ensemble:** exp_correct_cache (QWK 0.8932)  
**Most promising architecture:** exp_v5 — lower current QWK but fundamentally more capable head once the dataset is larger or training is longer  
**Most important unfixed bug:** Grade 4 recall degradation in v5 (39.0% vs 55.9% in v2) — likely fixable with threshold tuning + reduced CoarseDropout on Grade 4 images
