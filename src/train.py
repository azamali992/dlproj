import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from augmentation import get_train_transforms, get_valid_transforms

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)


def resolve_project_path(path_value):
    """Resolve a path relative to project root unless it's already absolute."""
    if os.path.isabs(path_value):
        return path_value
    return os.path.abspath(os.path.join(PROJECT_ROOT, path_value))


class APTOSDataset(Dataset):
    def __init__(self, df, img_dir, transforms=None):
        self.df = df
        self.img_dir = img_dir
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_id = self.df.iloc[idx]['id_code']
        img_path = os.path.join(self.img_dir, f"{img_id}.png")

        import cv2
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        label = self.df.iloc[idx]['diagnosis']

        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented['image']

        return image, torch.tensor(label, dtype=torch.long)


def prepare_data(csv_path, img_dir, batch_size=16):
    """
    Creates stratified splits and DataLoaders.
    """
    csv_path = resolve_project_path(csv_path)
    img_dir = resolve_project_path(img_dir)

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    if not os.path.isdir(img_dir):
        raise FileNotFoundError(f"Image directory not found: {img_dir}")

    df = pd.read_csv(csv_path)

    train_df, temp_df = train_test_split(
        df, test_size=0.30, stratify=df['diagnosis'], random_state=42
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, stratify=temp_df['diagnosis'], random_state=42
    )

    train_dataset = APTOSDataset(train_df, img_dir, transforms=get_train_transforms())
    val_dataset = APTOSDataset(val_df, img_dir, transforms=get_valid_transforms())
    test_dataset = APTOSDataset(test_df, img_dir, transforms=get_valid_transforms())

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader, test_loader


import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from metrics import quadratic_weighted_kappa
from models import APTOSModel


def train_baseline(csv_path, img_dir, num_epochs=15, batch_size=4, patience=5, accum_steps=4):
    torch.cuda.empty_cache()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    train_loader, val_loader, _ = prepare_data(csv_path, img_dir, batch_size)
    model = APTOSModel('efficientnet-b0', num_classes=5).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=3e-4)

    amp_enabled = device.type == "cuda"
    scaler = GradScaler("cuda", enabled=amp_enabled)

    best_qwk = -1.0
    patience_counter = 0

    checkpoint_dir = os.path.join(PROJECT_ROOT, "results", "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, "best_baseline.pth")

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()

        for i, (images, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")):
            images, labels = images.to(device), labels.to(device)

            with autocast("cuda", enabled=amp_enabled):
                outputs = model(images)
                loss = criterion(outputs, labels) / accum_steps

            scaler.scale(loss).backward()

            if (i + 1) % accum_steps == 0 or (i + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            train_loss += loss.item() * accum_steps

        model.eval()
        val_loss = 0.0
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
                images, labels = images.to(device), labels.to(device)

                with autocast("cuda", enabled=amp_enabled):
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                val_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)

                val_preds.extend(preds.cpu().numpy())
                val_targets.extend(labels.cpu().numpy())

        epoch_qwk = quadratic_weighted_kappa(val_targets, val_preds)
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        print(
            f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | Val QWK: {epoch_qwk:.4f}"
        )

        if epoch_qwk > best_qwk:
            best_qwk = epoch_qwk
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved best model with QWK: {best_qwk:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break


if __name__ == "__main__":
    CSV_PATH = "data/raw/train.csv"
    IMG_DIR = "data/processed/train_images_512"
    train_baseline(CSV_PATH, IMG_DIR, batch_size=4)