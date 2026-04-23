import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from tqdm import tqdm
import os

from models import APTOSModel
from train import prepare_data
from metrics import quadratic_weighted_kappa

def evaluate_model(csv_path, img_dir, model_path, batch_size=4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Evaluating on device: {device}")
    
    # load test data
    _, _, test_loader = prepare_data(csv_path, img_dir, batch_size)
    
    # init and load model
    model = APTOSModel('efficientnet-b0', num_classes=5).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    all_preds = []
    all_targets = []
    
    # inference loop
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images = images.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.numpy())
            
    # calc metrics
    acc = accuracy_score(all_targets, all_preds)
    qwk = quadratic_weighted_kappa(all_targets, all_preds)
    
    print("\n" + "="*50)
    print(f"Overall Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print(f"Quadratic Weighted Kappa: {qwk:.4f}")
    print("="*50 + "\n")
    
    # class wise report
    print("Classification Report (Checks Class Imbalance):")
    print(classification_report(all_targets, all_preds, digits=4))
    
    # gen confusion matrix
    cm = confusion_matrix(all_targets, all_preds)
    
    # plot and save cm
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix (QWK: {qwk:.4f})')
    plt.ylabel('True Grade')
    plt.xlabel('Predicted Grade')
    
    # fix save path to be relative to project root
    os.makedirs('results/figures', exist_ok=True)
    save_path = 'results/figures/confusion_matrix_baseline.png'
    plt.savefig(save_path)
    print(f"\nConfusion Matrix saved to: {save_path}")

if __name__ == "__main__":
    # update paths (removed ../)
    CSV_PATH = "data/raw/train.csv"
    IMG_DIR = "data/processed/train_images_512"
    MODEL_PATH = "results/checkpoints/best_baseline.pth"
    
    evaluate_model(CSV_PATH, IMG_DIR, MODEL_PATH)