"""
Fine-tuning ResNet18 with Grad-CAM Analysis - CORRECTED VERSION
Unfreezes deep layers, adds class weighting, reduces dropout
Results organized into: results/exp_finetuned_resnet18/
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from PIL import Image
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================
CONFIG = {
    'model_name': 'resnet18_finetuned',
    'num_classes': 5,
    'input_size': 224,
    'batch_size': 16,
    'num_epochs': 30,
    'learning_rate': 0.001,
    'weight_decay': 1e-4,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'data_dir': './data/raw',
    'output_dir': './results/exp_finetuned_resnet18',
    'gradcam_samples_per_grade': 3,
    'resume_from_checkpoint': False,
}

os.makedirs(CONFIG['output_dir'], exist_ok=True)

# ============================================================================
# DATASET
# ============================================================================
class DRDataset(Dataset):
    def __init__(self, img_dir, labels_df, transform=None, img_paths_dict=None):
        self.img_dir = img_dir
        self.labels_df = labels_df.reset_index(drop=True)
        self.transform = transform
        self.img_paths_dict = img_paths_dict or {}
    
    def __len__(self):
        return len(self.labels_df)
    
    def __getitem__(self, idx):
        row = self.labels_df.iloc[idx]
        img_path = os.path.join(self.img_dir, f"{row['id_code']}.png")
        
        self.img_paths_dict[idx] = img_path
        
        image = Image.open(img_path).convert('RGB')
        label = int(row['diagnosis'])
        
        if self.transform:
            image = self.transform(image)
        
        return image, label, img_path

# ============================================================================
# MODEL: Unfreeze deep layers, reduced dropout
# ============================================================================
class ResNet18FineTuned(nn.Module):
    def __init__(self, num_classes=5, freeze_early_only=True):
        super(ResNet18FineTuned, self).__init__()
        
        # Load pretrained ResNet18
        self.backbone = models.resnet18(pretrained=True)
        
        # Get number of input features for classifier
        num_features = self.backbone.fc.in_features
        
        # Replace final classifier with REDUCED dropout
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.3),  # Reduced from 0.5
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.2),  # Reduced from 0.3
            nn.Linear(512, num_classes)
        )
        
        # Freeze only early layers, unfreeze deep layers
        if freeze_early_only:
            for name, param in self.backbone.named_parameters():
                # Freeze only layer1 and layer2 (basic features)
                if 'layer1' in name or 'layer2' in name:
                    param.requires_grad = False
                # Unfreeze layer3, layer4, and fc (domain-specific)
                else:
                    param.requires_grad = True
    
    def forward(self, x):
        return self.backbone(x)
    
    def get_features(self, x):
        """Get activations before final classifier"""
        return self.backbone.avgpool(self.backbone.layer4(self.backbone.layer3(
            self.backbone.layer2(self.backbone.layer1(self.backbone.relu(
                self.backbone.bn1(self.backbone.conv1(x))))))))

# ============================================================================
# GRAD-CAM
# ============================================================================
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_handles = []
        self._register_hooks()
    
    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        forward_handle = self.target_layer.register_forward_hook(forward_hook)
        backward_handle = self.target_layer.register_backward_hook(backward_hook)
        self.hook_handles.append(forward_handle)
        self.hook_handles.append(backward_handle)
    
    def generate(self, input_tensor, target_class):
        """Generate Grad-CAM heatmap"""
        self.model.eval()
        
        output = self.model(input_tensor)
        
        self.model.zero_grad()
        
        target_score = output[0, target_class]
        target_score.backward()
        
        gradients = self.gradients[0].cpu().numpy()
        activations = self.activations[0].cpu().numpy()
        
        weights = np.mean(gradients, axis=(1, 2))
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        
        for i, w in enumerate(weights):
            cam += w * activations[i, :, :]
        
        cam = np.maximum(cam, 0)
        cam = cam / (cam.max() + 1e-8)
        
        return cam
    
    def remove_hooks(self):
        for handle in self.hook_handles:
            handle.remove()

# ============================================================================
# TRAINING LOOP
# ============================================================================
def train_epoch(model, train_loader, criterion, optimizer, device, epoch, num_epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, batch in enumerate(train_loader):
        images, labels = batch[0], batch[1]
        images, labels = images.to(device), labels.to(device)
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        if (batch_idx + 1) % 50 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] Batch [{batch_idx+1}/{len(train_loader)}] "
                  f"Loss: {loss.item():.4f}")
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy

def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in val_loader:
            images, labels = batch[0], batch[1]
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(val_loader)
    accuracy = 100 * correct / total
    qwk = cohen_kappa_score(all_labels, all_preds, weights='quadratic')
    
    return avg_loss, accuracy, qwk, all_preds, all_labels

# ============================================================================
# CHECKPOINT LOADING & SAVING
# ============================================================================
def load_checkpoint(checkpoint_path, model, optimizer, device):
    """Load checkpoint and return starting epoch + previous state"""
    print(f"\n{'='*80}")
    print(f"RESUMING FROM CHECKPOINT")
    print(f"{'='*80}")
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    
    start_epoch = checkpoint['epoch']
    history = checkpoint['history']
    best_qwk = checkpoint['best_qwk']
    best_epoch = checkpoint['best_epoch']
    best_preds = checkpoint['best_preds']
    best_labels = checkpoint['best_labels']
    
    print(f"✓ Resumed from epoch {start_epoch + 1}")
    print(f"✓ Previous best QWK: {best_qwk:.4f} (at epoch {best_epoch + 1})")
    print(f"{'='*80}\n")
    
    return start_epoch, history, best_qwk, best_epoch, best_preds, best_labels

def save_checkpoint(checkpoint_path, epoch, model, optimizer, history, 
                    best_qwk, best_epoch, best_preds, best_labels):
    """Save checkpoint for resuming later"""
    checkpoint = {
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'history': history,
        'best_qwk': best_qwk,
        'best_epoch': best_epoch,
        'best_preds': best_preds,
        'best_labels': best_labels,
    }
    torch.save(checkpoint, checkpoint_path)

# ============================================================================
# GRAD-CAM ANALYSIS
# ============================================================================
def generate_gradcam_analysis(model, val_dataset, val_df, device, output_dir, 
                              num_samples_per_grade=3):
    """Generate Grad-CAM visualizations for all grades"""
    
    print("\n" + "="*80)
    print("GENERATING GRAD-CAM ANALYSIS")
    print("="*80)
    
    model.eval()
    
    target_layer = model.backbone.layer4[-1]
    grad_cam = GradCAM(model, target_layer)
    
    samples_per_grade = {}
    for grade in range(5):
        grade_indices = np.where(val_df['diagnosis'].values == grade)[0]
        selected = np.random.choice(grade_indices, 
                                   size=min(num_samples_per_grade, len(grade_indices)), 
                                   replace=False)
        samples_per_grade[grade] = selected
    
    gradcam_data = {}
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    for grade in range(5):
        gradcam_data[grade] = {}
        
        for sample_idx in samples_per_grade[grade]:
            img_path = os.path.join(CONFIG['data_dir'], 'train_images', 
                                   f"{val_df.iloc[sample_idx]['id_code']}.png")
            img_pil = Image.open(img_path).convert('RGB')
            img_tensor = val_transform(img_pil).unsqueeze(0).to(device)
            
            with torch.no_grad():
                output = model(img_tensor)
                pred = output.argmax(1).item()
            
            heatmap = grad_cam.generate(img_tensor, grade)
            
            gradcam_data[grade][sample_idx] = {
                'image': img_pil,
                'heatmap': heatmap,
                'prediction': pred,
                'correct': (pred == grade)
            }
    
    grad_cam.remove_hooks()
    
    # Clinical Summary
    fig, axes = plt.subplots(5, 3, figsize=(15, 20))
    fig.suptitle('Grad-CAM Analysis: Clinical Summary\n(One sample per grade)', 
                 fontsize=16, fontweight='bold')
    
    grade_names = ['Grade 0: No DR', 'Grade 1: Mild', 'Grade 2: Moderate', 
                   'Grade 3: Severe', 'Grade 4: Proliferative']
    
    for grade in range(5):
        sample_idx = list(samples_per_grade[grade])[0]
        data = gradcam_data[grade][sample_idx]
        
        axes[grade, 0].imshow(data['image'])
        axes[grade, 0].set_title(f"{grade_names[grade]}\n(True label)")
        axes[grade, 0].axis('off')
        
        axes[grade, 1].imshow(data['image'])
        im = axes[grade, 1].imshow(data['heatmap'], cmap='jet', alpha=0.5)
        pred_str = f"Pred: G{data['prediction']}"
        correct_str = "✓" if data['correct'] else "✗"
        axes[grade, 1].set_title(f"Grad-CAM Overlay\n{pred_str} {correct_str}")
        axes[grade, 1].axis('off')
        
        axes[grade, 2].imshow(data['heatmap'], cmap='jet')
        axes[grade, 2].set_title("Heatmap (Attention)")
        axes[grade, 2].axis('off')
    
    plt.colorbar(im, ax=axes, orientation='vertical', pad=0.02, aspect=50)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'gradcam', 'gradcam_clinical_summary.png'), 
                dpi=150, bbox_inches='tight')
    print(f"✓ Saved: gradcam_clinical_summary.png")
    plt.close()
    
    # Grid
    fig, axes = plt.subplots(5, num_samples_per_grade * 2, figsize=(20, 20))
    fig.suptitle('Grad-CAM Analysis: All Samples\n(Image | Heatmap Overlay)', 
                 fontsize=16, fontweight='bold')
    
    for grade in range(5):
        col = 0
        for sample_idx in samples_per_grade[grade]:
            data = gradcam_data[grade][sample_idx]
            
            axes[grade, col].imshow(data['image'])
            axes[grade, col].axis('off')
            
            axes[grade, col + 1].imshow(data['image'])
            axes[grade, col + 1].imshow(data['heatmap'], cmap='jet', alpha=0.5)
            pred_str = f"G{data['prediction']}"
            correct_str = "✓" if data['correct'] else "✗"
            axes[grade, col + 1].set_title(f"{pred_str} {correct_str}", fontsize=10)
            axes[grade, col + 1].axis('off')
            
            col += 2
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'gradcam', 'gradcam_grid.png'), 
                dpi=150, bbox_inches='tight')
    print(f"✓ Saved: gradcam_grid.png")
    plt.close()
    
    # Per-Grade Detail
    for grade in range(5):
        fig, axes = plt.subplots(num_samples_per_grade, 3, figsize=(15, 5*num_samples_per_grade))
        if num_samples_per_grade == 1:
            axes = axes.reshape(1, -1)
        
        fig.suptitle(f'{grade_names[grade]} - Detailed Grad-CAM Analysis', 
                     fontsize=14, fontweight='bold')
        
        for row, sample_idx in enumerate(samples_per_grade[grade]):
            data = gradcam_data[grade][sample_idx]
            
            axes[row, 0].imshow(data['image'])
            axes[row, 0].set_title(f"Sample {row+1}: Original Image")
            axes[row, 0].axis('off')
            
            axes[row, 1].imshow(data['image'])
            im = axes[row, 1].imshow(data['heatmap'], cmap='jet', alpha=0.6)
            pred_str = f"Predicted: Grade {data['prediction']}"
            correct_str = "✓ Correct" if data['correct'] else "✗ Wrong"
            axes[row, 1].set_title(f"{pred_str}\n{correct_str}")
            axes[row, 1].axis('off')
            
            axes[row, 2].imshow(data['heatmap'], cmap='jet')
            axes[row, 2].set_title(f"Attention Heatmap\n(Model focus areas)")
            axes[row, 2].axis('off')
        
        plt.colorbar(im, ax=axes[:, 2])
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'gradcam', f'gradcam_grade_{grade}_detail.png'), 
                    dpi=150, bbox_inches='tight')
        print(f"✓ Saved: gradcam_grade_{grade}_detail.png")
        plt.close()
    
    # Clinical Report Template
    clinical_report = f"""GRAD-CAM CLINICAL ANALYSIS REPORT
=====================================

Model: ResNet18 Fine-Tuned (Corrected)
Task: Diabetic Retinopathy Grading (Grade 0-4)
Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

METHODOLOGY
-----------
Grad-CAM (Gradient-weighted Class Activation Maps) visualizes which regions
of the fundus image the model attends to when making predictions.

CLINICAL FEATURES BY GRADE
---------------------------

Grade 0 (No DR):
  Expected: Uniform, diffuse attention across healthy retina
  Observed: [FILL IN after viewing heatmaps]
  Alignment: [YES/NO/PARTIAL]

Grade 1 (Mild NPDR):
  Expected: Focus on microaneurysms (tiny red dots)
  Observed: [FILL IN]
  Alignment: [YES/NO/PARTIAL]

Grade 2 (Moderate NPDR):
  Expected: Focus on hemorrhages (red) and hard exudates (yellow)
  Observed: [FILL IN]
  Alignment: [YES/NO/PARTIAL]

Grade 3 (Severe NPDR):
  Expected: Widespread attention on cotton wool spots, hemorrhages
  Observed: [FILL IN]
  Alignment: [YES/NO/PARTIAL]

Grade 4 (Proliferative DR):
  Expected: Focus on neovascularization (abnormal vessels)
  Observed: [FILL IN]
  Alignment: [YES/NO/PARTIAL]

KEY OBSERVATIONS
----------------
[Fill in your observations after analysis]

CONCLUSION
----------
[Summary of whether model behavior aligns with clinical expectations]
"""
    
    with open(os.path.join(output_dir, 'gradcam', 'clinical_analysis_report.txt'), 'w') as f:
        f.write(clinical_report)
    
    print(f"✓ Saved: clinical_analysis_report.txt (template for your observations)")

# ============================================================================
# COMPUTE CLASS WEIGHTS
# ============================================================================
def compute_class_weights(labels):
    """Compute class weights inversely proportional to class frequency"""
    classes, counts = np.unique(labels, return_counts=True)
    weights = 1.0 / counts
    weights = weights / weights.sum() * len(classes)
    
    class_weights = torch.zeros(5)
    for cls, weight in zip(classes, weights):
        class_weights[cls] = weight
    
    return class_weights.to(CONFIG['device'])

# ============================================================================
# MAIN TRAINING
# ============================================================================
def main():
    # Load data
    print("Loading dataset...")
    train_df = pd.read_csv(os.path.join(CONFIG['data_dir'], 'train.csv'))
    
    # Split into train/val
    from sklearn.model_selection import train_test_split
    train_df, val_df = train_test_split(
        train_df, 
        test_size=0.2, 
        random_state=42, 
        stratify=train_df['diagnosis']
    )
    
    # Data augmentation (CORRECTED - no harmful flips)
    train_transform = transforms.Compose([
        transforms.RandomRotation(5),  # Small rotation OK
        transforms.ColorJitter(brightness=0.1, contrast=0.1),  # Color OK
        transforms.Resize((CONFIG['input_size'], CONFIG['input_size'])),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((CONFIG['input_size'], CONFIG['input_size'])),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Datasets
    img_paths = {}
    train_dataset = DRDataset(
        os.path.join(CONFIG['data_dir'], 'train_images'),
        train_df,
        transform=train_transform,
        img_paths_dict=img_paths
    )
    val_dataset = DRDataset(
        os.path.join(CONFIG['data_dir'], 'train_images'),
        val_df,
        transform=val_transform
    )
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Model (CORRECTED - unfreezes deep layers)
    print("\nInitializing model...")
    model = ResNet18FineTuned(
        num_classes=CONFIG['num_classes'],
        freeze_early_only=True  # Only freeze layer1, layer2
    )
    model = model.to(CONFIG['device'])
    
    # Count trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.1f}%)")
    print(f"Frozen parameters: {total_params - trainable_params:,} ({100*(total_params-trainable_params)/total_params:.1f}%)")
    
    # Loss with CLASS WEIGHTING (CORRECTED)
    class_weights = compute_class_weights(train_df['diagnosis'].values)
    print(f"\nClass Weights: {class_weights}")
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Optimizer
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=CONFIG['learning_rate'],
        weight_decay=CONFIG['weight_decay']
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=3,
        verbose=True
    )
    
    # Check for checkpoint
    start_epoch = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'val_qwk': []}
    best_qwk = 0
    best_epoch = 0
    best_preds = None
    best_labels = None
    
    checkpoint_path = os.path.join(CONFIG['output_dir'], 'checkpoint.pth')
    
    if CONFIG['resume_from_checkpoint'] and os.path.exists(checkpoint_path):
        start_epoch, history, best_qwk, best_epoch, best_preds, best_labels = \
            load_checkpoint(checkpoint_path, model, optimizer, CONFIG['device'])
    else:
        if CONFIG['resume_from_checkpoint']:
            print("\n" + "="*80)
            print("NO CHECKPOINT FOUND - STARTING FROM SCRATCH")
            print("="*80 + "\n")
        else:
            print("\n" + "="*80)
            print("CHECKPOINT/RESUME DISABLED - STARTING FROM SCRATCH")
            print("="*80 + "\n")
    
    # Training loop
    print("="*80)
    print("TRAINING STARTED")
    print("="*80)
    
    for epoch in range(start_epoch, CONFIG['num_epochs']):
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer,
            CONFIG['device'], epoch, CONFIG['num_epochs']
        )
        
        val_loss, val_acc, val_qwk, val_preds, val_labels = validate(
            model, val_loader, criterion, CONFIG['device']
        )
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_qwk'].append(val_qwk)
        
        print(f"\nEpoch {epoch+1}/{CONFIG['num_epochs']} Summary:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | Val QWK: {val_qwk:.4f}")
        
        # Save best model
        if val_qwk > best_qwk:
            best_qwk = val_qwk
            best_epoch = epoch
            best_preds = val_preds
            best_labels = val_labels
            torch.save(model.state_dict(), os.path.join(CONFIG['output_dir'], 'best_model.pth'))
            print(f"  ✓ Best model saved! (QWK: {val_qwk:.4f})")
        
        # Save checkpoint
        save_checkpoint(checkpoint_path, epoch, model, optimizer, history,
                       best_qwk, best_epoch, best_preds, best_labels)
        print(f"  💾 Checkpoint saved for resuming")
        
        scheduler.step(val_qwk)
    
    # Results
    print("\n" + "="*80)
    print("TRAINING COMPLETED")
    print("="*80)
    print(f"Best QWK: {best_qwk:.4f} at Epoch {best_epoch+1}")
    
    # Confusion matrix
    cm = confusion_matrix(best_labels, best_preds)
    
    # Classification report
    class_report = classification_report(
        best_labels, best_preds,
        target_names=['Grade 0', 'Grade 1', 'Grade 2', 'Grade 3', 'Grade 4'],
        output_dict=True
    )
    
    # Plot training curves
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    axes[0].plot(history['train_loss'], label='Train', marker='o')
    axes[0].plot(history['val_loss'], label='Val', marker='s')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss Curve')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(history['train_acc'], label='Train', marker='o')
    axes[1].plot(history['val_acc'], label='Val', marker='s')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Accuracy Curve')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(history['val_qwk'], label='Val QWK', marker='o', color='green')
    axes[2].axhline(y=best_qwk, color='r', linestyle='--', label=f'Best: {best_qwk:.4f}')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('QWK')
    axes[2].set_title('Quadratic Weighted Kappa')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG['output_dir'], 'training_curves.png'), dpi=150)
    print(f"✓ Saved: training_curves.png")
    plt.close()
    
    # Confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['G0', 'G1', 'G2', 'G3', 'G4'],
                yticklabels=['G0', 'G1', 'G2', 'G3', 'G4'])
    plt.title(f'Confusion Matrix (Best Epoch {best_epoch+1}, QWK={best_qwk:.4f})')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG['output_dir'], 'confusion_matrix.png'), dpi=150)
    print(f"✓ Saved: confusion_matrix.png")
    plt.close()
    
    # Classification report (text)
    report_text = classification_report(
        best_labels, best_preds,
        target_names=['Grade 0', 'Grade 1', 'Grade 2', 'Grade 3', 'Grade 4']
    )
    
    with open(os.path.join(CONFIG['output_dir'], 'classification_report.txt'), 'w') as f:
        f.write(report_text)
    print(f"✓ Saved: classification_report.txt")
    
    # Summary
    summary = {
        'model': 'ResNet18 Fine-Tuned (Corrected)',
        'approach': 'Unfreeze layers 3-4, train with class weighting',
        'best_epoch': best_epoch + 1,
        'best_qwk': float(best_qwk),
        'best_val_loss': float(history['val_loss'][best_epoch]),
        'best_val_acc': float(history['val_acc'][best_epoch]),
        'total_params': int(total_params),
        'trainable_params': int(trainable_params),
        'frozen_params': int(total_params - trainable_params),
        'config': CONFIG,
        'per_class_metrics': {
            str(i): {
                'precision': float(class_report.get(str(i), {}).get('precision', 0.0)),
                'recall': float(class_report.get(str(i), {}).get('recall', 0.0)),
                'f1-score': float(class_report.get(str(i), {}).get('f1-score', 0.0)),
            }
            for i in range(5)
        }
    }
    
    with open(os.path.join(CONFIG['output_dir'], 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Saved: summary.json")
    
    # Cleanup
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print(f"✓ Cleanup: Removed checkpoint.pth (training complete)")
    
    # Final Summary
    print(f"\n" + "="*80)
    print("ALL RESULTS SAVED")
    print("="*80)
    print(f"\n📁 Results folder: {CONFIG['output_dir']}/")
    print(f"\n📊 Main Results:")
    print(f"  ✓ best_model.pth (trained weights)")
    print(f"  ✓ training_curves.png (loss & accuracy)")
    print(f"  ✓ confusion_matrix.png")
    print(f"  ✓ classification_report.txt")
    print(f"  ✓ summary.json")
    print(f"\n🎯 Key Metrics:")
    print(f"  Best QWK: {best_qwk:.4f}")
    print(f"  Best Epoch: {best_epoch + 1}")
    for i in range(5):
        metrics = class_report.get(str(i), {})
        recall = metrics.get('recall', 0.0)
        print(f"  Grade {i} Recall: {recall:.2%}")

if __name__ == '__main__':
    main()