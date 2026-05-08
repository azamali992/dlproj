"""
EXPERIMENT 5: ResNet18 + CE + Grayscale Preprocessing
=======================================================
Change from Exp1: RGB → Grayscale (3-channel grayscale for ResNet compatibility)
Everything else SAME as baseline.

Purpose: Does converting to grayscale help or hurt?
         Some papers use grayscale, but color contains diagnostic info
         (hemorrhages are RED, exudates are YELLOW).
         This experiment tests that hypothesis.
"""

import os, random, numpy as np, pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image

from sklearn.metrics import cohen_kappa_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ============================================================================
# CONFIG
# ============================================================================
class Config:
    DATA_DIR = 'data/raw'
    TRAIN_CSV = os.path.join(DATA_DIR, 'train.csv')
    TRAIN_IMAGES = os.path.join(DATA_DIR, 'train_images')
    IMAGE_SIZE = 224
    BATCH_SIZE = 16
    NUM_CLASSES = 5
    NUM_EPOCHS = 10
    LEARNING_RATE = 1e-4
    VAL_SPLIT = 0.2
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    RESULTS_DIR = 'results/exp5_grayscale'
    EXP_NAME = 'Exp5: ResNet18 + CE + Grayscale'

config = Config()
os.makedirs(config.RESULTS_DIR, exist_ok=True)

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
set_seed(42)

# ============================================================================
# DATA — GRAYSCALE TRANSFORM
# ============================================================================
transform = transforms.Compose([
    transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
    transforms.Grayscale(num_output_channels=3),  # ← Convert to grayscale but keep 3 channels for ResNet
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class DRDataset(Dataset):
    def __init__(self, df, image_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        img_id = self.df.loc[idx, 'id_code']
        label = self.df.loc[idx, 'diagnosis']
        img = Image.open(os.path.join(self.image_dir, f'{img_id}.png')).convert('RGB')
        if self.transform: img = self.transform(img)
        return img, label

df = pd.read_csv(config.TRAIN_CSV)
train_df, val_df = train_test_split(df, test_size=config.VAL_SPLIT, stratify=df['diagnosis'], random_state=42)

train_loader = DataLoader(DRDataset(train_df, config.TRAIN_IMAGES, transform), batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
val_loader = DataLoader(DRDataset(val_df, config.TRAIN_IMAGES, transform), batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

# ============================================================================
# MODEL
# ============================================================================
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, config.NUM_CLASSES)
model = model.to(config.DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

print(f"\n{'='*60}")
print(f"{config.EXP_NAME}")
print(f"{'='*60}")
print(f"Device: {config.DEVICE}")
print(f"Preprocessing: Grayscale (3-channel)")
print(f"Train: {len(train_df)}, Val: {len(val_df)}")

# ============================================================================
# TRAINING
# ============================================================================
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train(); running_loss = 0.0
    pbar = tqdm(loader, desc='Training')
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward(); optimizer.step()
        running_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    return running_loss / len(loader)

def validate(model, loader, criterion, device):
    model.eval(); running_loss = 0.0; all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in tqdm(loader, desc='Validation'):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            running_loss += criterion(outputs, labels).item()
            all_preds.extend(outputs.argmax(1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    y_true, y_pred = np.array(all_labels), np.array(all_preds)
    return running_loss / len(loader), {
        'qwk': cohen_kappa_score(y_true, y_pred, weights='quadratic'),
        'accuracy': (y_true == y_pred).mean(),
        'confusion_matrix': confusion_matrix(y_true, y_pred, labels=[0,1,2,3,4])
    }, y_pred, y_true

history = {'train_loss': [], 'val_loss': [], 'val_qwk': [], 'val_accuracy': []}
best_qwk = 0.0

for epoch in range(config.NUM_EPOCHS):
    print(f"\nEpoch {epoch+1}/{config.NUM_EPOCHS}\n" + "-"*60)
    train_loss = train_one_epoch(model, train_loader, criterion, optimizer, config.DEVICE)
    val_loss, metrics, preds, labels = validate(model, val_loader, criterion, config.DEVICE)
    print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val QWK: {metrics['qwk']:.4f} | Val Acc: {metrics['accuracy']:.4f}")
    history['train_loss'].append(train_loss); history['val_loss'].append(val_loss)
    history['val_qwk'].append(metrics['qwk']); history['val_accuracy'].append(metrics['accuracy'])
    if metrics['qwk'] > best_qwk:
        best_qwk = metrics['qwk']
        torch.save(model.state_dict(), os.path.join(config.RESULTS_DIR, 'best_model.pth'))
        print(f"✓ New best model (QWK: {best_qwk:.4f})")

# ============================================================================
# FINAL EVALUATION & SAVE
# ============================================================================
model.load_state_dict(torch.load(os.path.join(config.RESULTS_DIR, 'best_model.pth')))
_, metrics, preds, labels = validate(model, val_loader, criterion, config.DEVICE)

print(f"\n{'='*60}\n{config.EXP_NAME} — FINAL RESULTS\n{'='*60}")
print(f"Best QWK: {metrics['qwk']:.4f} | Accuracy: {metrics['accuracy']:.4f}")
report = classification_report(labels, preds, target_names=['Grade 0','Grade 1','Grade 2','Grade 3','Grade 4'], digits=4)
print(report)

with open(os.path.join(config.RESULTS_DIR, 'classification_report.txt'), 'w') as f:
    f.write(f"{config.EXP_NAME}\n\nBest QWK: {metrics['qwk']:.4f}\nAccuracy: {metrics['accuracy']:.4f}\n\n{report}")

fig, axes = plt.subplots(1, 2, figsize=(15, 5))
axes[0].plot(history['train_loss'], label='Train', lw=2); axes[0].plot(history['val_loss'], label='Val', lw=2)
axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss'); axes[0].set_title('Loss Curves'); axes[0].legend(); axes[0].grid(True, alpha=0.3)
axes[1].plot(history['val_qwk'], lw=2, color='green')
axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('QWK'); axes[1].set_title('Validation QWK'); axes[1].grid(True, alpha=0.3)
plt.suptitle(config.EXP_NAME, fontweight='bold'); plt.tight_layout()
plt.savefig(os.path.join(config.RESULTS_DIR, 'training_curves.png'), dpi=150)

cm = metrics['confusion_matrix']
plt.figure(figsize=(10, 8)); plt.imshow(cm, cmap='Blues'); plt.colorbar()
for i in range(5):
    for j in range(5):
        plt.text(j, i, cm[i,j], ha='center', va='center', color='white' if cm[i,j] > cm.max()/2 else 'black', fontsize=14, fontweight='bold')
plt.xlabel('Predicted Grade', fontsize=12); plt.ylabel('True Grade', fontsize=12)
plt.title(f'Confusion Matrix — {config.EXP_NAME}', fontsize=14); plt.xticks([0,1,2,3,4]); plt.yticks([0,1,2,3,4])
plt.tight_layout(); plt.savefig(os.path.join(config.RESULTS_DIR, 'confusion_matrix.png'), dpi=150)

with open(os.path.join(config.RESULTS_DIR, 'config.txt'), 'w') as f:
    f.write(f"Experiment: {config.EXP_NAME}\nModel: ResNet18 (pretrained)\nLoss: CrossEntropyLoss\n")
    f.write(f"Preprocessing: Grayscale (3-ch) + Resize + Normalize\nAugmentation: None\n")
    f.write(f"Image Size: {config.IMAGE_SIZE}\nBatch Size: {config.BATCH_SIZE}\nLR: {config.LEARNING_RATE}\nBest QWK: {best_qwk:.4f}\n")

print(f"\n✅ Results saved to: {config.RESULTS_DIR}")
