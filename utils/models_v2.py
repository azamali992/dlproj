"""
utils/models_v2.py — DenseNet121 with CORAL head + Grad-CAM support
=====================================================================

Changes from v1:
1. Uses CORALModule_v2 (label smoothing support).
2. Registers forward/backward hooks on features.denseblock4 for Grad-CAM.
3. Exposes generate_gradcam(logits, target_class) method.
4. Layer freezing recommendation: only freeze denseblock1 (v1 froze 1+2,
   which was too aggressive and slowed ordinal boundary learning).

Grad-CAM usage:
    model.eval()
    logits = model(image_batch)
    cam = model.generate_gradcam(logits, target_class=3)  # [B,1,H,W] in [0,1]
    # Resize to input resolution and overlay on image.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from utils.losses_v2 import CORALModule_v2


class DenseNet121withCORALFocal_v2(nn.Module):

    def __init__(self, num_classes=5, gamma=2.0, pretrained=True, label_smooth=0.05):
        super().__init__()

        weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
        self.model = models.densenet121(weights=weights)

        num_features = self.model.classifier.in_features
        self.model.classifier = CORALModule_v2(
            in_features=num_features,
            num_classes=num_classes,
            gamma=gamma,
            label_smooth=label_smooth,
        )

        # Grad-CAM storage (populated by hooks)
        self._activations = None
        self._gradients = None

        # Hook the last dense block — output shape (B, 1024, 7, 7) for 224×224 input
        target = self.model.features.denseblock4
        target.register_forward_hook(self._hook_activation)
        target.register_full_backward_hook(self._hook_gradient)

    # ------------------------------------------------------------------
    # Hook callbacks
    # ------------------------------------------------------------------

    def _hook_activation(self, module, input, output):
        self._activations = output  # keep gradient graph intact

    def _hook_gradient(self, module, grad_input, grad_output):
        self._gradients = grad_output[0]

    # ------------------------------------------------------------------
    # Standard CORAL interface (identical to v1)
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Grad-CAM
    # ------------------------------------------------------------------

    def generate_gradcam(self, logits, target_class=None):
        """
        Generate a Grad-CAM saliency map.

        Args:
            logits:       [B, K-1] CORAL logits (output of forward()).
                          Must have been computed with requires_grad active
                          (i.e. do NOT call inside torch.no_grad()).
            target_class: integer 0-4. If None, uses the predicted class
                          for the first element in the batch.

        Returns:
            cam: [B, 1, H, W] float tensor in [0, 1].
                 Upsampled to input resolution with F.interpolate before overlaying.

        Example:
            model.eval()  # hooks still fire in eval mode
            image = image.unsqueeze(0).to(device).requires_grad_(False)
            logits = model(image)
            cam = model.generate_gradcam(logits, target_class=2)
            cam_resized = F.interpolate(cam, size=(224, 224), mode='bilinear',
                                        align_corners=False)
        """
        if target_class is None:
            target_class = int(self.predict(logits.detach())[0].item())

        # For CORAL, backprop through the threshold logit that corresponds
        # to distinguishing target_class from target_class-1.
        # Threshold k separates "grade > k" from "grade <= k".
        # For grade g, the most discriminative threshold is k = g-1 (if g>0) or k=0.
        k = max(0, target_class - 1)

        # Zero existing gradients and backpropagate
        if logits.grad_fn is None:
            raise RuntimeError(
                "generate_gradcam requires logits with a gradient graph. "
                "Do not wrap forward() in torch.no_grad()."
            )

        self.model.zero_grad()
        # Sum over batch so gradient flows for all examples
        logits[:, k].sum().backward(retain_graph=True)

        gradients = self._gradients   # [B, C, H, W]
        activations = self._activations  # [B, C, H, W]

        if gradients is None or activations is None:
            raise RuntimeError("Grad-CAM hooks did not fire. "
                               "Ensure the model performed a forward pass first.")

        # Global average pool gradients → channel importance weights
        weights = gradients.mean(dim=(2, 3), keepdim=True)  # [B, C, 1, 1]
        cam = (weights * activations).sum(dim=1, keepdim=True)  # [B, 1, H, W]
        cam = F.relu(cam)

        # Normalize each map independently to [0, 1]
        B = cam.shape[0]
        cam_flat = cam.view(B, -1)
        cam_min = cam_flat.min(dim=1)[0].view(B, 1, 1, 1)
        cam_max = cam_flat.max(dim=1)[0].view(B, 1, 1, 1)
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)

        return cam.detach()


# ------------------------------------------------------------------
# Freezing helper — call after model construction
# ------------------------------------------------------------------

def freeze_early_layers(model: DenseNet121withCORALFocal_v2, blocks=('denseblock1',)):
    """
    Freeze specified DenseNet blocks.

    v1 froze denseblock1 + denseblock2.  Freezing two blocks was too aggressive:
    denseblock2 learns mid-level texture features (vessel patterns, microaneurysms)
    that are specific to fundus images and need fine-tuning.

    Recommended: freeze only denseblock1 (low-level edges/colours — ImageNet
    features transfer well and freezing reduces overfitting without hurting QWK).
    """
    frozen = 0
    for name, param in model.model.named_parameters():
        if any(block in name for block in blocks):
            param.requires_grad = False
            frozen += 1

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Frozen blocks: {blocks}")
    print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")
    return model
