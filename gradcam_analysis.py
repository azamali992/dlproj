"""
GRAD-CAM ANALYSIS FOR DIABETIC RETINOPATHY
============================================
Generates Grad-CAM heatmaps for each experiment's best model.
Shows which retinal structures the model attends to at each severity level.

Reference: Selvaraju et al. (2017) "Grad-CAM: Visual Explanations from Deep
           Networks via Gradient-based Localization" ICCV

Clinical features to verify:
  Grade 0 (No DR):        Uniform attention, no specific focus
  Grade 1 (Mild):         Microaneurysms — tiny red dots
  Grade 2 (Moderate):     Hemorrhages — red blotches, some hard exudates
  Grade 3 (Severe):       Cotton wool spots, extensive hemorrhages
  Grade 4 (Proliferative): Neovascularization — abnormal blood vessel growth

Usage:
  python gradcam_analysis.py

  This will generate Grad-CAM visualizations for ALL experiments that have
  a best_model.pth in their results folder.

Requires: pip install grad-cam
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cv2
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from sklearn.model_selection import train_test_split

# Grad-CAM library
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# ============================================================================
# CONFIGURATION
# ============================================================================
DATA_DIR = 'data/raw'
TRAIN_CSV = os.path.join(DATA_DIR, 'train.csv')
TRAIN_IMAGES = os.path.join(DATA_DIR, 'train_images')
IMAGE_SIZE = 224
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SAMPLES_PER_GRADE = 3  # How many images to show per grade

GRADE_NAMES = {
    0: 'Grade 0: No DR',
    1: 'Grade 1: Mild',
    2: 'Grade 2: Moderate',
    3: 'Grade 3: Severe',
    4: 'Grade 4: Proliferative'
}

# Clinical features expected at each grade
CLINICAL_FEATURES = {
    0: 'Healthy retina — model should show uniform/diffuse attention, no specific lesion focus',
    1: 'Microaneurysms (tiny red dots) — model should focus on small scattered spots',
    2: 'Hemorrhages (red blotches) + hard exudates (yellow spots) — focal attention on lesion areas',
    3: 'Cotton wool spots + extensive hemorrhages — widespread attention across retina',
    4: 'Neovascularization (new abnormal vessels) — attention on vessel abnormalities, often near optic disc'
}

# ============================================================================
# EXPERIMENTS TO ANALYZE (UPDATED WITH FINE-TUNED MODEL)
# ============================================================================
EXPERIMENTS = {
    'baseline': {
        'name': 'Exp1: ResNet18 (Baseline) + CE',
        'model_path': 'results/baseline/best_model.pth',
        'model_type': 'resnet18',
        'output_dir': 'results/baseline/gradcam',
        'description': 'Training from scratch with standard cross-entropy loss'
    },
    'finetuned_resnet': {
        'name': 'Exp Finetuned: ResNet18 (Fine-Tuned Backbone) + CE',
        'model_path': 'results/exp_finetuned_resnet18/best_model.pth',
        'model_type': 'resnet18',
        'is_finetuned': True,  # ← Special flag for fine-tuned model
        'output_dir': 'results/exp_finetuned_resnet18/gradcam',
        'description': 'Fine-tuned with frozen backbone, improved for minority classes'
    },
    'resnet50': {
        'name': 'Exp2: ResNet50 + CE',
        'model_path': 'results/exp2_resnet50/best_model.pth',
        'model_type': 'resnet50',
        'output_dir': 'results/exp2_resnet50/gradcam',
        'description': 'Deeper architecture, 50 layers'
    },
    'efficientnet': {
        'name': 'Exp3: EfficientNet-B0 + CE',
        'model_path': 'results/exp3_efficientnet/best_model.pth',
        'model_type': 'efficientnet',
        'output_dir': 'results/exp3_efficientnet/gradcam',
        'description': 'Efficient architecture with compound scaling'
    },
    'weighted_ce': {
        'name': 'Exp4: ResNet18 + Weighted CE',
        'model_path': 'results/exp4_weighted_ce/best_model.pth',
        'model_type': 'resnet18',
        'output_dir': 'results/exp4_weighted_ce/gradcam',
        'description': 'Addresses class imbalance with weighted loss'
    },
    'grayscale': {
        'name': 'Exp5: ResNet18 + CE + Grayscale',
        'model_path': 'results/exp5_grayscale/best_model.pth',
        'model_type': 'resnet18',
        'output_dir': 'results/exp5_grayscale/gradcam',
        'description': 'Grayscale input to verify color importance'
    },
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_model(model_type, model_path, num_classes=5, is_finetuned=False):
    """Load a trained model and return model + target layer for Grad-CAM
    
    Args:
        model_type: 'resnet18', 'resnet50', 'efficientnet'
        model_path: path to model weights
        num_classes: number of output classes
        is_finetuned: True if this is a fine-tuned model with backbone wrapper
    """

    if model_type == 'resnet18':
        if is_finetuned:
            # Fine-tuned model has ResNet18FineTuned wrapper with backbone + fc
            model = models.resnet18(weights=None)
            model.fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(model.fc.in_features, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, num_classes)
            )
            target_layer = [model.layer4[-1]]  # Last conv block
            
            # Load and adapt state dict (remove 'backbone.' prefix)
            state_dict = torch.load(model_path, map_location=DEVICE)
            new_state_dict = {}
            for k, v in state_dict.items():
                # Remove 'backbone.' prefix from keys
                if k.startswith('backbone.'):
                    new_k = k.replace('backbone.', '')
                    new_state_dict[new_k] = v
                else:
                    new_state_dict[k] = v
            model.load_state_dict(new_state_dict)
        else:
            # Standard baseline model
            model = models.resnet18(weights=None)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
            target_layer = [model.layer4[-1]]
            model.load_state_dict(torch.load(model_path, map_location=DEVICE))

    elif model_type == 'resnet50':
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        target_layer = [model.layer4[-1]]
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))

    elif model_type == 'efficientnet':
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        target_layer = [model.features[-1]]  # Last feature block
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model = model.to(DEVICE)
    model.eval()

    return model, target_layer


def load_and_preprocess(img_path, grayscale=False):
    """Load image, return both raw (for display) and preprocessed (for model)"""

    # Raw image for display
    raw_img = cv2.imread(img_path)
    raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
    raw_img = cv2.resize(raw_img, (IMAGE_SIZE, IMAGE_SIZE))
    raw_img_float = raw_img.astype(np.float32) / 255.0  # [0, 1] for overlay

    # Preprocessed for model
    pil_img = Image.open(img_path).convert('RGB')

    if grayscale:
        preprocess = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        preprocess = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    input_tensor = preprocess(pil_img).unsqueeze(0).to(DEVICE)

    return raw_img, raw_img_float, input_tensor


def generate_gradcam(model, target_layer, input_tensor, target_class=None):
    """Generate Grad-CAM heatmap"""
    cam = GradCAM(model=model, target_layers=target_layer)

    if target_class is not None:
        targets = [ClassifierOutputTarget(target_class)]
    else:
        targets = None  # Use predicted class

    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    return grayscale_cam[0, :]


def create_grade_grid(model, target_layer, df, image_dir, output_dir, exp_name,
                      grayscale=False, samples_per_grade=3):
    """
    Create a grid showing Grad-CAM for multiple samples per grade.
    This is the MAIN visualization for the report.
    """

    fig = plt.figure(figsize=(5 * samples_per_grade, 25))
    gs = gridspec.GridSpec(5, samples_per_grade * 2, hspace=0.3, wspace=0.05)

    for grade in range(5):
        grade_df = df[df['diagnosis'] == grade]

        # Sample random images for this grade
        n_samples = min(samples_per_grade, len(grade_df))
        sampled = grade_df.sample(n=n_samples, random_state=42)

        for idx, (_, row) in enumerate(sampled.iterrows()):
            img_path = os.path.join(image_dir, f"{row['id_code']}.png")
            raw_img, raw_float, input_tensor = load_and_preprocess(img_path, grayscale)

            # Generate Grad-CAM
            cam_mask = generate_gradcam(model, target_layer, input_tensor, target_class=grade)
            overlay = show_cam_on_image(raw_float, cam_mask, use_rgb=True)

            # Original image
            ax1 = fig.add_subplot(gs[grade, idx * 2])
            ax1.imshow(raw_img)
            ax1.axis('off')
            if idx == 0:
                ax1.set_title(GRADE_NAMES[grade], fontsize=11, fontweight='bold', loc='left')

            # Grad-CAM overlay
            ax2 = fig.add_subplot(gs[grade, idx * 2 + 1])
            ax2.imshow(overlay)
            ax2.axis('off')

    plt.suptitle(f'Grad-CAM Analysis — {exp_name}', fontsize=16, fontweight='bold', y=1.01)
    plt.savefig(os.path.join(output_dir, 'gradcam_grid.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Grad-CAM grid saved")


def create_single_grade_detail(model, target_layer, df, image_dir, output_dir,
                               grade, grayscale=False):
    """
    Create detailed Grad-CAM for ONE grade with clinical annotation.
    Shows original, heatmap, overlay side by side.
    """

    grade_df = df[df['diagnosis'] == grade]
    sampled = grade_df.sample(n=min(3, len(grade_df)), random_state=42)

    fig, axes = plt.subplots(len(sampled), 3, figsize=(15, 5 * len(sampled)))
    if len(sampled) == 1:
        axes = axes.reshape(1, -1)

    for idx, (_, row) in enumerate(sampled.iterrows()):
        img_path = os.path.join(image_dir, f"{row['id_code']}.png")
        raw_img, raw_float, input_tensor = load_and_preprocess(img_path, grayscale)

        # Grad-CAM for true class
        cam_mask = generate_gradcam(model, target_layer, input_tensor, target_class=grade)
        overlay = show_cam_on_image(raw_float, cam_mask, use_rgb=True)

        # Heatmap only
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_mask), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        axes[idx, 0].imshow(raw_img)
        axes[idx, 0].set_title('Original', fontsize=12)
        axes[idx, 0].axis('off')

        axes[idx, 1].imshow(heatmap)
        axes[idx, 1].set_title('Heatmap', fontsize=12)
        axes[idx, 1].axis('off')

        axes[idx, 2].imshow(overlay)
        axes[idx, 2].set_title('Overlay', fontsize=12)
        axes[idx, 2].axis('off')

    plt.suptitle(
        f'{GRADE_NAMES[grade]}\n{CLINICAL_FEATURES[grade]}',
        fontsize=13, fontweight='bold', y=1.02, wrap=True
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, f'gradcam_grade_{grade}_detail.png'),
        dpi=200, bbox_inches='tight'
    )
    plt.close()


def create_clinical_summary(model, target_layer, df, image_dir, output_dir,
                            exp_name, grayscale=False):
    """
    Create a single summary image showing one representative sample per grade
    with clinical feature annotations. BEST for mid-report.
    """

    fig, axes = plt.subplots(5, 3, figsize=(15, 25))

    for grade in range(5):
        grade_df = df[df['diagnosis'] == grade]
        sample = grade_df.sample(n=1, random_state=42).iloc[0]
        img_path = os.path.join(image_dir, f"{sample['id_code']}.png")
        raw_img, raw_float, input_tensor = load_and_preprocess(img_path, grayscale)

        cam_mask = generate_gradcam(model, target_layer, input_tensor, target_class=grade)
        overlay = show_cam_on_image(raw_float, cam_mask, use_rgb=True)
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_mask), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        # Get model prediction
        with torch.no_grad():
            output = model(input_tensor)
            pred = output.argmax(1).item()
            confidence = torch.softmax(output, dim=1)[0, pred].item()

        axes[grade, 0].imshow(raw_img)
        axes[grade, 0].set_title(f'{GRADE_NAMES[grade]}', fontsize=11, fontweight='bold')
        axes[grade, 0].axis('off')

        axes[grade, 1].imshow(heatmap)
        axes[grade, 1].set_title(f'Heatmap', fontsize=11)
        axes[grade, 1].axis('off')

        axes[grade, 2].imshow(overlay)
        axes[grade, 2].set_title(f'Pred: Grade {pred} ({confidence:.1%})', fontsize=11)
        axes[grade, 2].axis('off')

        # Add clinical note on the left
        axes[grade, 0].text(
            0.02, 0.02, CLINICAL_FEATURES[grade][:60] + '...',
            transform=axes[grade, 0].transAxes,
            fontsize=7, color='white',
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.7),
            verticalalignment='bottom'
        )

    plt.suptitle(
        f'Grad-CAM Clinical Analysis — {exp_name}\n'
        f'Red regions = high model attention | Blue = low attention',
        fontsize=14, fontweight='bold', y=1.01
    )
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'gradcam_clinical_summary.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Clinical summary saved")


def write_clinical_report(output_dir, exp_name):
    """Write a text file with clinical analysis template"""
    report = f"""GRAD-CAM CLINICAL ANALYSIS REPORT
===================================
Experiment: {exp_name}
Reference: Selvaraju et al. (2017) "Grad-CAM: Visual Explanations from Deep Networks" ICCV

CLINICAL FEATURE ALIGNMENT ANALYSIS
------------------------------------

Grade 0 (No DR):
  Expected: Uniform/diffuse attention across healthy retina
  Observed: [FILL IN after viewing heatmaps]
  Alignment: [Does model focus match clinical expectation? YES/NO/PARTIAL]

Grade 1 (Mild NPDR):
  Expected: Focus on microaneurysms (tiny red dots, often near macula)
  Observed: [FILL IN]
  Alignment: [YES/NO/PARTIAL]

Grade 2 (Moderate NPDR):
  Expected: Focus on hemorrhages (red blotches) and hard exudates (yellow spots)
  Observed: [FILL IN]
  Alignment: [YES/NO/PARTIAL]

Grade 3 (Severe NPDR):
  Expected: Widespread attention on cotton wool spots, extensive hemorrhages
  Observed: [FILL IN]
  Alignment: [YES/NO/PARTIAL]

Grade 4 (Proliferative DR):
  Expected: Focus on neovascularization (abnormal vessel growth), often near optic disc
  Observed: [FILL IN]
  Alignment: [YES/NO/PARTIAL]

SUMMARY
-------
[Does the model generally attend to clinically relevant features?]
[Which grades show the strongest clinical alignment?]
[Any concerns about model attention patterns?]
[How does this model compare to baseline for clinical interpretability?]

NOTE: This analysis follows the methodology of Quellec et al. (2019)
who showed kappa=0.76 agreement between Grad-CAM and ophthalmologist annotations.
"""
    with open(os.path.join(output_dir, 'clinical_analysis_report.txt'), 'w') as f:
        f.write(report)
    print(f"  ✓ Clinical report template saved")


# ============================================================================
# MAIN
# ============================================================================
def main():
    print("="*80)
    print("GRAD-CAM ANALYSIS FOR DIABETIC RETINOPATHY")
    print("="*80)
    print(f"Device: {DEVICE}")
    print(f"Processing {len(EXPERIMENTS)} experiments\n")

    # Load data
    df = pd.read_csv(TRAIN_CSV)
    _, val_df = train_test_split(
        df, test_size=0.2, stratify=df['diagnosis'], random_state=42
    )
    print(f"Using {len(val_df)} validation images\n")

    # Process each experiment
    completed = 0
    skipped = 0

    for exp_key, exp_config in EXPERIMENTS.items():

        # Check if model exists
        if not os.path.exists(exp_config['model_path']):
            print(f"⚠ Skipping {exp_config['name']} — no model found at {exp_config['model_path']}")
            skipped += 1
            continue

        print(f"\n{'─'*80}")
        print(f"Processing: {exp_config['name']}")
        print(f"Description: {exp_config['description']}")
        print(f"{'─'*80}")

        # Create output directory
        os.makedirs(exp_config['output_dir'], exist_ok=True)

        # Load model
        is_grayscale = (exp_key == 'grayscale')
        is_finetuned = exp_config.get('is_finetuned', False)  # Get flag, default False
        model, target_layer = load_model(
            exp_config['model_type'],
            exp_config['model_path'],
            is_finetuned=is_finetuned  # Pass the flag
        )
        print(f"  ✓ Model loaded from {exp_config['model_path']}")

        # 1. Generate main grid (all grades, multiple samples)
        print(f"  Generating Grad-CAM grid...")
        create_grade_grid(
            model, target_layer, val_df, TRAIN_IMAGES,
            exp_config['output_dir'], exp_config['name'],
            grayscale=is_grayscale, samples_per_grade=SAMPLES_PER_GRADE
        )

        # 2. Generate detailed view per grade
        print(f"  Generating per-grade detail views...")
        for grade in range(5):
            create_single_grade_detail(
                model, target_layer, val_df, TRAIN_IMAGES,
                exp_config['output_dir'], grade, grayscale=is_grayscale
            )
        print(f"  ✓ All 5 grade details saved")

        # 3. Generate clinical summary (BEST for report)
        print(f"  Generating clinical summary...")
        create_clinical_summary(
            model, target_layer, val_df, TRAIN_IMAGES,
            exp_config['output_dir'], exp_config['name'],
            grayscale=is_grayscale
        )

        # 4. Write clinical report template
        write_clinical_report(exp_config['output_dir'], exp_config['name'])

        # Cleanup
        del model
        torch.cuda.empty_cache()
        
        completed += 1

    print(f"\n{'='*80}")
    print("GRAD-CAM ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\nResults:")
    print(f"  ✓ Completed: {completed} experiments")
    print(f"  ⚠ Skipped: {skipped} experiments (models not found)")
    print(f"\nGenerated files per experiment:")
    print(f"  gradcam_grid.png              — Grid: all grades, multiple samples")
    print(f"  gradcam_grade_X_detail.png    — Detailed view per grade (0-4)")
    print(f"  gradcam_clinical_summary.png  — Summary with predictions (BEST FOR REPORT)")
    print(f"  clinical_analysis_report.txt  — Template to fill in observations")
    print(f"\nNote: Fine-tuned model included in Experiments!")
    print(f"      Compare gradcam_clinical_summary.png between baseline and fine-tuned")
    print(f"      to verify improved clinical alignment on minority classes.")


if __name__ == '__main__':
    main()