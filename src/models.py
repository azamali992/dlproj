import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet

class APTOSModel(nn.Module):
    def __init__(self, model_name='efficientnet-b0', num_classes=5):
        super().__init__()
        # Load pretrained model
        self.model = EfficientNet.from_pretrained(model_name)
        
        # Replace final fully connected layer for 5 classes
        in_features = self.model._fc.in_features
        self.model._fc = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        return self.model(x)