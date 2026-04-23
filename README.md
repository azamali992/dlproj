# Diabetic Retinopathy Grading - Implementation Log

Yeh repository APTOS 2019 dataset par based Diabetic Retinopathy grading ke liye banayi gayi hai. Is README ka maqsad project ka ab tak ka progress log, run kiye gaye commands, aur updated files ko ek jagah document karna hai, taake koi bhi team member asaani se setup samajh kar apne system par project chala sake.

## Setup & Prerequisites

Sab se pehle virtual environment activate kiya gaya aur required dependencies install ki gayi hain.

```bash
pip install -r requirements.txt
pip install kagglehub
```

## Step-by-Step Execution Log

### Step 1: Project Structure Creation

Standard ICML-style project structure set ki gayi hai, jisme `src/`, `data/`, `notebooks/`, `results/`, aur `configs/` jaisay folders shamil hain. Is se code modular aur maintainable rehta hai.

### Step 2: Data Download

Kaggle se APTOS 2019 dataset download karne ke liye `kagglehub` use kiya gaya hai. Downloaded data `data/raw/` folder mein copy hota hai.

File: `data/download_data.py`

```bash
python data/download_data.py
```

### Step 3: Data Preprocessing (Ben Graham Method)

Raw fundus images se black borders crop karne aur local average color subtract karne ke liye preprocessing script chalayi gayi hai. Processed images `data/processed/train_images_512/` mein save hoti hain.

File: `data/preprocess.py`

```bash
python data/preprocess.py
```

### Step 4: Exploratory Data Analysis (EDA)

Dataset distribution, class imbalance, aur preprocessing ke raw vs processed results visualize karne ke liye Jupyter notebook use ki gayi hai. Is step se confirm hota hai ke dataset severely imbalanced hai aur Grade 0 samples sab se zyada hain.

File: `notebooks/01_EDA.ipynb`

Action: Is notebook ko VS Code ya Jupyter mein open karke tamam cells run karein.

### Step 5: Augmentation & Dataset Setup

`albumentations` use karke training augmentation pipeline banayi gayi hai, jisme flips, rotations, aur brightness adjustments shamil hain. Saath hi stratified split `70/15/15` ke saath PyTorch `Dataset` aur `DataLoader` setup kiya gaya hai.

Files Updated:

- `src/augmentation.py` - transforms logic
- `src/train.py` - `APTOSDataset` class aur `prepare_data` function

### Step 6: Baseline Model & Training Loop

EfficientNet-B0 ko baseline architecture ke taur par initialize kiya gaya hai. Cross-Entropy loss aur Quadratic Weighted Kappa (QWK) metric ke saath complete training loop, early stopping, aur model checkpointing implement ki gayi hai.

Files Updated:

- `src/models.py` - EfficientNet-B0 architecture setup
- `src/metrics.py` - QWK metric calculation function
- `src/train.py` - main training loop

```bash
python src/train.py
```

### Step 7: Training Stability & Runtime Fixes

Baseline training script ko update kiya gaya taake woh different working directories se reliably run ho sake aur runtime issues avoid hon.

Key improvements:

- Project-root based path resolution add ki gayi (`resolve_project_path`) for CSV/image loading.
- Checkpoint save path ko absolute project path par shift kiya gaya.
- `results/checkpoints/` directory auto-create karne ke liye `os.makedirs(..., exist_ok=True)` add kiya gaya.
- Mixed precision training ko new PyTorch AMP API par migrate kiya gaya (`torch.amp.autocast`, `torch.amp.GradScaler`).
- Gradient accumulation (`accum_steps=4`) add ki gayi to reduce GPU memory pressure.

Files Updated:

- `src/train.py`

Command Run:

```bash
python src/train.py
```

Observed:

- Training successfully start hui on CUDA.
- Epoch 1 validation QWK ~ `0.7998` observe hua.

### Step 8: Evaluation Pipeline Added

Model evaluation ke liye separate script add/update ki gayi jo test split par inference run karti hai aur detailed metrics report karti hai.

What evaluation does:

- Best checkpoint load karta hai (`results/checkpoints/best_baseline.pth`).
- Test split par predictions generate karta hai.
- Accuracy + QWK compute karta hai.
- Classification report print karta hai (class-wise precision/recall/F1).
- Confusion matrix plot karke save karta hai.

Files Updated:

- `src/evaluate.py`

Command Run:

```bash
python src/evaluate.py
```

Artifacts:

- Checkpoint: `results/checkpoints/best_baseline.pth`
- Confusion Matrix Figure: `results/figures/confusion_matrix_baseline.png`

## Current Status & Next Steps

### In Progress

Baseline training + evaluation pipeline functional hai. Team ab class imbalance mitigation aur QWK optimization phase mein hai.

### Next Target

Ab next iteration mein severe class imbalance handle karne ke liye `src/losses.py` mein Weighted Cross-Entropy, Focal Loss, ya Ordinal Regression loss implement kiya jayega. Saath hi threshold tuning / ensembling evaluate ki jayegi. Target Quadratic Weighted Kappa (QWK) `> 0.82` achieve karna hai.

## Collaboration Note

Yeh README continuously update hoti rahegi. Har nayi milestone ke saath neeche:

- kya kiya gaya
- kaunsi file update hui
- kaunsa command run hua
- result kya nikla

document kiya jayega, taake team members same workflow follow kar saken.