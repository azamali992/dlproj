from torchvision import models
from torch import nn
from utils.losses import CORALModule


class Resnet18withCORALFocal(nn.Module):
    def __init__(self, num_classes=5, gamma=2.0, pretrained=True):
        super().__init__()

        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        self.model = models.resnet18(weights=weights)

        # Get feature size
        num_features = self.model.fc.in_features

        # Replace final FC with CORAL module
        self.model.fc = CORALModule(
            in_features=num_features,
            num_classes=num_classes,
            gamma=gamma
        )

    def forward(self, x):
        return self.model(x)

    def compute_alpha(self, labels):
        return self.model.fc.compute_alpha(labels)

    def coral_label_transform(self, y):
        return self.model.fc.coral_label_transform(y)

    def loss(self, logits, targets):
        return self.model.fc.loss(logits, targets)

    def predict(self, logits):
        return self.model.fc.predict(logits)

    def threshold_probs(self, logits):
        return self.model.fc.threshold_probs(logits)



from torchvision import models
from torch import nn
from utils.losses import CORALModule


class DenseNet121withCORALFocal(nn.Module):
    def __init__(self, num_classes=5, gamma=2.0, pretrained=True):
        super().__init__()

        weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
        self.model = models.densenet121(weights=weights)

        num_features = self.model.classifier.in_features

        self.model.classifier = CORALModule(
            in_features=num_features,
            num_classes=num_classes,
            gamma=gamma
        )

    def forward(self, x):
        return self.model(x)

    def compute_alpha(self, labels):
        return self.model.classifier.compute_alpha(labels)

    def coral_label_transform(self, y):
        return self.model.classifier.coral_label_transform(y)

    def loss(self, logits, targets):
        return self.model.classifier.loss(logits, targets)

    def predict(self, logits):
        return self.model.classifier.predict(logits)

    def threshold_probs(self, logits):
        return self.model.classifier.threshold_probs(logits)