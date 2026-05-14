"""
utils/models_v5.py — DenseNet121 with independent CORAL head + Grad-CAM
========================================================================

Changes from v2
---------------
1. Uses CORALModule_v5 (independent per-threshold projections, alpha on
   original labels, ordinal consistency regularization).

2. Freeze recommendation updated:
   v2 recommended denseblock1 only but Claude Code froze 1+2+3 which was
   too aggressive — denseblock3 learns lesion morphology features critical
   for G2/G3 separation.
   v3 freeze_early_layers() defaults to denseblock1 ONLY.

3. Grad-CAM updated for independent projections:
   With shared weights, all thresholds shared one gradient source.
   With independent projections, each threshold has its own gradient.
   generate_gradcam() now accepts threshold_k to select which boundary's
   gradient to visualize (default: the boundary most relevant to target_class).

4. compute_alpha() interface updated — must pass ORIGINAL (pre-balance)
   labels. The training script is responsible for passing train_df before
   create_balanced_train_dataframe() is called.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from utils.losses_v5 import CORALModule_v5


class DenseNet121withCORALFocal_v5(nn.Module):

    def __init__(self, num_classes=5, gamma=2.0, pretrained=True, lambda_ord=0.1):
        super().__init__()

        weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
        self.model = models.densenet121(weights=weights)

        num_features = self.model.classifier.in_features  # 1024
        self.model.classifier = CORALModule_v5(
            in_features=num_features,
            num_classes=num_classes,
            gamma=gamma,
            lambda_ord=lambda_ord,
        )

        # Grad-CAM storage
        self._activations = None
        self._gradients   = None

        # Hook denseblock4 — highest-level spatial features
        target = self.model.features.denseblock4
        target.register_forward_hook(self._hook_activation)
        target.register_full_backward_hook(self._hook_gradient)

    # ------------------------------------------------------------------
    # Hook callbacks
    # ------------------------------------------------------------------

    def _hook_activation(self, module, input, output):
        self._activations = output

    def _hook_gradient(self, module, grad_input, grad_output):
        self._gradients = grad_output[0]

    # ------------------------------------------------------------------
    # Standard CORAL interface — identical to v2, training script unchanged
    # ------------------------------------------------------------------

    def forward(self, x):
        return self.model(x)

    def compute_alpha(self, labels):
        """
        Pass ORIGINAL imbalanced labels (train_df BEFORE balancing).
        This is the key change from v2 — v2 received balanced labels and
        computed near-uniform alpha. v3 needs the raw distribution.
        """
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
    # Grad-CAM (updated for independent projections)
    # ------------------------------------------------------------------

    def generate_gradcam(self, logits, target_class=None, threshold_k=None):
        """
        Generate a Grad-CAM saliency map.

        Args:
            logits:       [B, K-1] CORAL logits from forward().
                          Must NOT be inside torch.no_grad().
            target_class: integer 0-4. If None, uses predicted class for
                          the first batch element.
            threshold_k:  which threshold's gradient to backprop through.
                          If None, uses max(0, target_class - 1) which is
                          the boundary most discriminative for target_class.

        Returns:
            cam: [B, 1, H, W] float tensor in [0, 1].
        """
        if target_class is None:
            target_class = int(self.predict(logits.detach())[0].item())

        if threshold_k is None:
            threshold_k = max(0, target_class - 1)

        if logits.grad_fn is None:
            raise RuntimeError(
                "generate_gradcam requires logits with a gradient graph. "
                "Do not wrap forward() in torch.no_grad()."
            )

        self.model.zero_grad()
        logits[:, threshold_k].sum().backward(retain_graph=True)

        gradients  = self._gradients    # [B, C, H, W]
        activations = self._activations # [B, C, H, W]

        if gradients is None or activations is None:
            raise RuntimeError("Grad-CAM hooks did not fire. "
                               "Ensure model performed a forward pass first.")

        weights = gradients.mean(dim=(2, 3), keepdim=True)          # [B, C, 1, 1]
        cam     = (weights * activations).sum(dim=1, keepdim=True)   # [B, 1, H, W]
        cam     = F.relu(cam)

        B = cam.shape[0]
        cam_flat = cam.view(B, -1)
        cam_min  = cam_flat.min(dim=1)[0].view(B, 1, 1, 1)
        cam_max  = cam_flat.max(dim=1)[0].view(B, 1, 1, 1)
        cam      = (cam - cam_min) / (cam_max - cam_min + 1e-8)

        return cam.detach()


# ------------------------------------------------------------------
# Freezing helper
# ------------------------------------------------------------------

def freeze_early_layers(model: DenseNet121withCORALFocal_v5,
                        blocks: tuple = ('denseblock1',)):
    """
    Freeze specified DenseNet feature blocks.

    v3 default: freeze denseblock1 ONLY.

    Block roles in fundus imaging:
        denseblock1 — edges, color gradients, basic textures.
                      ImageNet features transfer perfectly. Safe to freeze.
        denseblock2 — vessel patterns, circular structures.
                      Partially transfers. Can freeze if training is slow.
        denseblock3 — lesion shape, boundary sharpness, texture gradients.
                      FUNDUS-SPECIFIC. Must be trained. Freezing this was
                      the cause of G3 recall dropping to 0.38 in v2.
        denseblock4 — high-level semantic features (what the CORAL head reads).
                      Always trained.

    Previous run froze denseblock1+2+3 — equivalent to training only
    denseblock4 + CORAL head (~35% params). That prevented the model from
    learning what moderate vs severe haemorrhages look like at the feature
    level, directly causing the G2/G3 confusion.
    """
    for name, param in model.model.named_parameters():
        if any(block in name for block in blocks):
            param.requires_grad = False

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen    = total - trainable

    print(f"Frozen blocks    : {blocks}")
    print(f"Trainable params : {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")
    print(f"Frozen params    : {frozen:,} ({100*frozen/total:.1f}%)")
    return model
