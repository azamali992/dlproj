import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

def get_train_transforms(image_size=512):
    return A.Compose([
        # Randomly flip images horizontally and vertically
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        
        # Randomly rotate between -30 and 30 degrees
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=30, p=0.5),
        
        # Adjust brightness and contrast
        A.RandomBrightnessContrast(p=0.5),
        
        # Normalize using ImageNet mean and std
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        
        # Convert to PyTorch Tensor format
        ToTensorV2()
    ])

def get_valid_transforms(image_size=512):
    return A.Compose([
        # Only normalize and convert to tensor for validation
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2()
    ])