"""
Grad-CAM Analysis for CORAL + Focal Loss Model
===============================================

Problem with standard Grad-CAM on CORAL:
- Standard Grad-CAM backprops through a single class logit
- CORAL outputs K-1 threshold logits, NOT K class scores
- So we need a custom approach:
  For grade G, backprop through the CUMULATIVE activation
  i.e. sum of threshold logits up to G (captures "is this at least grade G?")

This script:
1. Loads your trained CORAL model
2. Applies Grad-CAM with ordinal-aware backprop
3. Visualizes attention per grade
4. Generates clinical alignment report
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import json
import pandas as pd
from PIL import Image
from pathlib import Path
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Import your model
from utils.models import Resnet18withCORALFocal

# Import preprocessing
from preprocess import HybridPreprocessor, BenGrahamPreprocessor, CLAHEPreprocessor

# ============================================================================
# CONFIGURATION
# ============================================================================
CONFIG = {
    'model_path': './results/exp_coral_focal/best_model.pth',
    'data_dir': './data/raw',
    'output_dir': './results/exp_coral_focal/gradcam',
    'num_classes': 5,
    'gamma': 2.0,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'samples_per_grade': 3,
    'preprocessing_method': 'hybrid',  # Match what was used in training
}

os.makedirs(CONFIG['output_dir'], exist_ok=True)

# ============================================================================
# GRAD-CAM FOR CORAL (Ordinal-Aware)
# ============================================================================
class CoralGradCAM:
    """
    Grad-CAM adapted for CORAL ordinal regression.

    CORAL outputs K-1 threshold logits: P(y > 0), P(y > 1), ..., P(y > K-2)

    For grade G, we want to know what the model attends to when it thinks
    the image is "grade G or above". We compute this as:

    target_score = sum of threshold logits from 0 to G-1
                   (how strongly does model think this is at least grade G?)

    For grade 0 (No DR), we flip: negate threshold 0 logit
    (how strongly does the model think this is NOT above grade 0?)
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.forward_handle = self.target_layer.register_forward_hook(forward_hook)
        self.backward_handle = self.target_layer.register_backward_hook(backward_hook)

    def generate(self, input_tensor, target_grade):
        """
        Generate Grad-CAM heatmap for a specific grade.

        Args:
            input_tensor: [1, 3, H, W] preprocessed image tensor
            target_grade: 0-4, which DR grade to visualize

        Returns:
            cam: [H, W] normalized heatmap
        """
        self.model.eval()

        # Forward pass - get CORAL threshold logits [1, K-1]
        logits = self.model(input_tensor)  # [1, 4]

        self.model.zero_grad()

        # ORDINAL-AWARE TARGET SCORE
        if target_grade == 0:
            # Grade 0 = No DR: negate first threshold
            target_score = -logits[0, 0]
        else:
            # Grade G: sum of thresholds 0 to G-1
            target_score = logits[0, :target_grade].sum()

        # Backprop
        target_score.backward()

        # Grad-CAM weights
        gradients = self.gradients[0].cpu().numpy()    # [C, H, W]
        activations = self.activations[0].cpu().numpy() # [C, H, W]

        weights = np.mean(gradients, axis=(1, 2))  # [C]

        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i, :, :]

        cam = np.maximum(cam, 0)
        cam = cam / (cam.max() + 1e-8)

        return cam

    def remove_hooks(self):
        self.forward_handle.remove()
        self.backward_handle.remove()


# ============================================================================
# LOAD MODEL
# ============================================================================
def load_model(model_path, device):
    """Load trained CORAL model"""
    print(f"Loading model from: {model_path}")

    model = Resnet18withCORALFocal(
        num_classes=CONFIG['num_classes'],
        gamma=CONFIG['gamma'],
        pretrained=False
    )

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    print(f"✓ Model loaded successfully")
    return model


# ============================================================================
# IMAGE LOADING + PREPROCESSING
# ============================================================================
def load_and_preprocess(image_path, method='hybrid', device='cuda'):
    """Load image, preprocess and convert to tensor"""

    if method == 'hybrid':
        preprocessor = HybridPreprocessor(image_size=224)
        image_np = preprocessor.both_methods(image_path)
    elif method == 'ben_graham':
        preprocessor = BenGrahamPreprocessor(image_size=224)
        image_np = preprocessor.process(image_path)
    elif method == 'clahe':
        preprocessor = CLAHEPreprocessor(image_size=224)
        image_np = preprocessor.process(image_path)
    else:
        image_np = np.array(Image.open(image_path).convert('RGB').resize((224, 224)))

    img_pil = Image.fromarray(image_np)

    img_tensor = torch.from_numpy(image_np).float() / 255.0
    img_tensor = img_tensor.permute(2, 0, 1)  # HWC to CHW

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img_tensor = (img_tensor - mean) / std

    img_tensor = img_tensor.unsqueeze(0).to(device)  # [1, 3, H, W]

    return img_tensor, img_pil


# ============================================================================
# SELECT SAMPLES
# ============================================================================
def select_samples(data_dir, n_per_grade=3):
    """Select n images per grade from validation set"""

    train_df = pd.read_csv(os.path.join(data_dir, 'train.csv'))

    _, val_df = train_test_split(
        train_df,
        test_size=0.2,
        random_state=42,
        stratify=train_df['diagnosis']
    )
    val_df = val_df.reset_index(drop=True)

    samples = {}
    for grade in range(5):
        grade_rows = val_df[val_df['diagnosis'] == grade]
        n = min(n_per_grade, len(grade_rows))
        selected = grade_rows.sample(n=n, random_state=42)
        samples[grade] = selected

    return samples


# ============================================================================
# GENERATE ALL GRAD-CAM
# ============================================================================
def generate_all_gradcam(model, samples, data_dir, device, preprocessing_method):
    """Generate Grad-CAM for all selected samples"""

    # Hook onto layer4 (last conv layer in ResNet18)
    target_layer = model.model.layer4[-1]
    grad_cam = CoralGradCAM(model, target_layer)

    results = {}

    grade_names = [
        'Grade 0: No DR',
        'Grade 1: Mild NPDR',
        'Grade 2: Moderate NPDR',
        'Grade 3: Severe NPDR',
        'Grade 4: Proliferative DR'
    ]

    for grade in range(5):
        results[grade] = []
        grade_df = samples[grade]

        print(f"\nProcessing {grade_names[grade]} ({len(grade_df)} samples)...")

        for _, row in grade_df.iterrows():
            img_path = os.path.join(data_dir, 'train_images', f"{row['id_code']}.png")

            img_tensor, img_pil = load_and_preprocess(
                img_path, method=preprocessing_method, device=device
            )

            # Get prediction
            with torch.no_grad():
                logits = model(img_tensor)
                pred = model.predict(logits).item()

            # Generate Grad-CAM for the TRUE grade
            heatmap = grad_cam.generate(img_tensor, grade)

            results[grade].append({
                'id_code': row['id_code'],
                'image': img_pil,
                'heatmap': heatmap,
                'true_grade': grade,
                'predicted_grade': pred,
                'correct': (pred == grade)
            })

            print(f"  {row['id_code']}: True=G{grade}, Pred=G{pred} "
                  f"({'✓' if pred == grade else '✗'})")

    grad_cam.remove_hooks()
    return results


# ============================================================================
# VISUALIZATIONS
# ============================================================================
def plot_clinical_summary(results, output_dir):
    """One row per grade, 3 columns. Best figure for your report."""

    grade_names = [
        'Grade 0: No DR',
        'Grade 1: Mild NPDR',
        'Grade 2: Moderate NPDR',
        'Grade 3: Severe NPDR',
        'Grade 4: Proliferative DR'
    ]

    clinical_features = [
        'Uniform attention — healthy disc/macula',
        'Microaneurysms (tiny red dots near macula)',
        'Hemorrhages (red) + Hard exudates (yellow)',
        'Widespread damage, cotton-wool spots, IRMA',
        'Neovascularization (disc/peripheral new vessels)'
    ]

    fig, axes = plt.subplots(5, 3, figsize=(14, 22))
    fig.suptitle(
        'Grad-CAM Clinical Analysis\nResNet18 + CORAL Ordinal Regression',
        fontsize=15, fontweight='bold', y=1.01
    )

    for grade in range(5):
        sample = results[grade][0]
        img = np.array(sample['image'])
        heatmap = sample['heatmap']

        heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        heatmap_colored = cv2.applyColorMap(
            (heatmap_resized * 255).astype(np.uint8), cv2.COLORMAP_JET
        )
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

        # Column 0: Original
        axes[grade, 0].imshow(img)
        axes[grade, 0].set_title(
            f'{grade_names[grade]}\n{clinical_features[grade]}',
            fontsize=8, pad=4
        )
        axes[grade, 0].axis('off')

        # Column 1: Overlay
        overlay = (img * 0.5 + heatmap_colored * 0.5).astype(np.uint8)
        pred = sample['predicted_grade']
        correct = '✓' if sample['correct'] else '✗'
        axes[grade, 1].imshow(overlay)
        axes[grade, 1].set_title(f'Grad-CAM Overlay\nPredicted: G{pred} {correct}', fontsize=8, pad=4)
        axes[grade, 1].axis('off')

        # Column 2: Heatmap
        im = axes[grade, 2].imshow(heatmap_resized, cmap='jet', vmin=0, vmax=1)
        axes[grade, 2].set_title('Attention Map\n(Red=High, Blue=Low)', fontsize=8, pad=4)
        axes[grade, 2].axis('off')

    plt.colorbar(im, ax=axes[:, 2], orientation='vertical',
                 fraction=0.03, pad=0.04, label='Attention Intensity')
    plt.tight_layout()

    out_path = os.path.join(output_dir, 'gradcam_clinical_summary.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved: gradcam_clinical_summary.png  ← USE THIS IN REPORT")
    plt.close()


def plot_per_grade_detail(results, output_dir):
    """Detailed per-grade analysis with all samples"""

    grade_names = [
        'Grade 0: No DR', 'Grade 1: Mild NPDR', 'Grade 2: Moderate NPDR',
        'Grade 3: Severe NPDR', 'Grade 4: Proliferative DR'
    ]

    for grade in range(5):
        grade_samples = results[grade]
        n = len(grade_samples)

        fig, axes = plt.subplots(n, 3, figsize=(12, 5 * n))
        if n == 1:
            axes = axes.reshape(1, -1)

        fig.suptitle(f'Detailed Grad-CAM: {grade_names[grade]}', fontsize=13, fontweight='bold')

        for row_idx, sample in enumerate(grade_samples):
            img = np.array(sample['image'])
            heatmap = sample['heatmap']
            heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
            heatmap_colored = cv2.applyColorMap(
                (heatmap_resized * 255).astype(np.uint8), cv2.COLORMAP_JET
            )
            heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
            overlay = (img * 0.5 + heatmap_colored * 0.5).astype(np.uint8)

            pred = sample['predicted_grade']
            correct = '✓ Correct' if sample['correct'] else '✗ Wrong'

            axes[row_idx, 0].imshow(img)
            axes[row_idx, 0].set_title(f"Sample {row_idx+1} | {sample['id_code']}")
            axes[row_idx, 0].axis('off')

            axes[row_idx, 1].imshow(overlay)
            axes[row_idx, 1].set_title(f"Overlay | Pred: G{pred} {correct}")
            axes[row_idx, 1].axis('off')

            im = axes[row_idx, 2].imshow(heatmap_resized, cmap='jet')
            axes[row_idx, 2].set_title("Attention Map")
            axes[row_idx, 2].axis('off')

        plt.colorbar(im, ax=axes[:, 2])
        plt.tight_layout()

        out_path = os.path.join(output_dir, f'gradcam_grade_{grade}_detail.png')
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved: gradcam_grade_{grade}_detail.png")
        plt.close()


# ============================================================================
# CLINICAL ALIGNMENT REPORT
# ============================================================================
def generate_clinical_report(results, output_dir):
    """Clinical alignment report template"""

    clinical_structures = [
        {
            'structures': 'None (healthy retina)',
            'attention': 'Uniform/diffuse — optic disc, macula area',
            'note': 'Model should show no focused lesion attention'
        },
        {
            'structures': 'Microaneurysms (tiny red/dark dots)',
            'attention': 'Focal spots, often near macula',
            'note': 'Microaneurysms are the earliest sign of DR (ADA 2023)'
        },
        {
            'structures': 'Hard exudates (yellow deposits), retinal hemorrhages',
            'attention': 'Scattered bright spots + red regions',
            'note': 'Hard exudates indicate fluid leakage from damaged vessels'
        },
        {
            'structures': 'Cotton-wool spots, IRMA, venous beading',
            'attention': 'Widespread across multiple quadrants',
            'note': 'IRMA = Intraretinal microvascular abnormalities (severe ischemia)'
        },
        {
            'structures': 'Neovascularization (NVD/NVE), vitreous hemorrhage',
            'attention': 'Disc region (NVD) + peripheral areas (NVE)',
            'note': 'New vessel growth is hallmark of PDR — requires urgent treatment'
        }
    ]

    grade_accuracy = {}
    for grade in range(5):
        s = results[grade]
        n_correct = sum(1 for x in s if x['correct'])
        grade_accuracy[grade] = f"{n_correct}/{len(s)} ({100*n_correct/len(s):.0f}%)"

    report = f"""GRAD-CAM CLINICAL ALIGNMENT REPORT
=====================================
Model: ResNet18 + CORAL Ordinal Regression + Focal Loss
Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

METHODOLOGY
-----------
Standard Grad-CAM is incompatible with CORAL (different output structure).
This script uses ORDINAL-AWARE backprop:

  Grade 0: backprop through -threshold_logit[0]
             (evidence the image has NO DR)

  Grade G: backprop through sum(threshold_logits[0..G-1])
             (cumulative evidence the image is at least grade G)

This matches clinical interpretation: "how severe is this image?"
Cite: Selvaraju et al. (2017) + Cao et al. (2020)

SAMPLE ACCURACY ON ANALYZED IMAGES
------------------------------------
"""
    for g in range(5):
        report += f"  Grade {g}: {grade_accuracy[g]}\n"

    report += "\n"

    grade_names = [
        'Grade 0 (No DR)', 'Grade 1 (Mild NPDR)', 'Grade 2 (Moderate NPDR)',
        'Grade 3 (Severe NPDR)', 'Grade 4 (Proliferative DR)'
    ]

    for grade in range(5):
        info = clinical_structures[grade]
        samples = results[grade]
        n_correct = sum(1 for s in samples if s['correct'])

        report += f"""
────────────────────────────────────────────────────────
{grade_names[grade]}
────────────────────────────────────────────────────────
Expected Structures: {info['structures']}
Expected Attention:  {info['attention']}
Clinical Note:       {info['note']}
Prediction:          {n_correct}/{len(samples)} correct

Observations (fill in after viewing gradcam_grade_{grade}_detail.png):
  Q1. Does model focus on expected structures?  [YES / NO / PARTIAL]
  Q2. Attention pattern (localized/diffuse)?    [FILL IN]
  Q3. Misclassified samples — what does attention show?  [FILL IN]
  Q4. Clinical alignment:  [ALIGNED / MISALIGNED / PARTIALLY]

"""

    report += """
=====================================
SUMMARY TABLE
=====================================

| Grade | Structures Expected    | Attention Pattern | Alignment |
|-------|------------------------|-------------------|-----------|
| G0    | [FILL IN]              | [FILL IN]         | [FILL IN] |
| G1    | [FILL IN]              | [FILL IN]         | [FILL IN] |
| G2    | [FILL IN]              | [FILL IN]         | [FILL IN] |
| G3    | [FILL IN]              | [FILL IN]         | [FILL IN] |
| G4    | [FILL IN]              | [FILL IN]         | [FILL IN] |

KEY FINDINGS:
  [FILL IN]

CONCLUSION:
  [FILL IN — see example template below]

  Example:
  "The CORAL model demonstrated clinically meaningful attention at Grades 0 and 2,
   focusing on [structure]. Grade 1 attention was [diffuse/concentrated], suggesting
   the model [relies on / correctly identifies] [feature]. Grade 3 and 4 patterns
   [aligned / did not align] with expected neovascularization regions as per 
   International Clinical Diabetic Retinopathy Severity Scale (ICDRS)."

REFERENCES:
  - Selvaraju et al. (2017). Grad-CAM. ICCV. https://arxiv.org/abs/1610.02055
  - Cao et al. (2020). CORAL. Pattern Recognit. Lett. https://arxiv.org/abs/1901.04667
  - ADA (2023). Standards of Care: Microvascular Complications.
  - ICDRS: International Clinical Diabetic Retinopathy Severity Scale
"""

    out_path = os.path.join(output_dir, 'clinical_alignment_report.txt')
    with open(out_path, 'w') as f:
        f.write(report)

    print(f"✓ Saved: clinical_alignment_report.txt")


# ============================================================================
# MAIN
# ============================================================================
def main():
    print("="*80)
    print("GRAD-CAM ANALYSIS: ResNet18 + CORAL Ordinal Regression")
    print("="*80)

    device = CONFIG['device']
    print(f"Device: {device}")

    # Load model
    model = load_model(CONFIG['model_path'], device)

    # Select validation samples
    print(f"\nSelecting {CONFIG['samples_per_grade']} samples per grade from validation set...")
    samples = select_samples(CONFIG['data_dir'], CONFIG['samples_per_grade'])

    # Generate Grad-CAM
    print("\nGenerating Grad-CAM (ordinal-aware backprop)...")
    results = generate_all_gradcam(
        model=model,
        samples=samples,
        data_dir=CONFIG['data_dir'],
        device=device,
        preprocessing_method=CONFIG['preprocessing_method']
    )

    # Save all visualizations
    print("\nSaving visualizations...")
    plot_clinical_summary(results, CONFIG['output_dir'])
    plot_per_grade_detail(results, CONFIG['output_dir'])
    generate_clinical_report(results, CONFIG['output_dir'])

    print(f"\n{'='*80}")
    print("DONE")
    print(f"{'='*80}")
    print(f"\n📁 Output: {CONFIG['output_dir']}/")
    print(f"\n🏥 Files for your report:")
    print(f"  ✓ gradcam_clinical_summary.png   ← Best single figure")
    print(f"  ✓ gradcam_grade_X_detail.png     ← Per-grade detail")
    print(f"  ✓ clinical_alignment_report.txt  ← Fill in observations")
    print(f"\n⚠️  IMPORTANT — Why CORAL Grad-CAM is different:")
    print(f"  Standard Grad-CAM backprops through class logit (won't work for CORAL)")
    print(f"  This script backprops through cumulative threshold activation instead")
    print(f"  Grade 0 → -threshold[0]")
    print(f"  Grade G → sum(thresholds[0..G-1])")
    print(f"  This correctly captures ordinal severity attention")


if __name__ == '__main__':
    main()
