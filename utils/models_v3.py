"""
utils/models_v3.py — EfficientNet-B3 + CBAM + Multi-scale fusion + Dual heads
===============================================================================

Hardware target: RTX 2050 4GB VRAM, Core i5 12th gen, 8GB RAM.
Profiled safe config:
    backbone  : EfficientNet-B3   (10.7M params)
    image size: 384x384
    batch size: 8 (+ gradient accumulation x2 → effective BS 16)
    VRAM use  : ~3.2-3.6 GB

Architecture
------------
    EfficientNet-B3 (stage 0 frozen)
         |      |      |
      stage2  stage3  stage4
         |      |      |
       CBAM   CBAM   CBAM     ← channel attention (Frontiers Med 2026)
         |      |      |
        GAP    GAP    GAP
         └──────┴──────┘
              concat (568-d)
                 |
           FusionHead (256-d)
            /          \
       CoralHead    RegressionHead
     (4 thresholds)  (scalar 0->4)   ← regression regularises G2/G3 boundary

Why EfficientNet-B3 over DenseNet121:
    Springer Nature 2025: EfficientNetB0+MSAG achieves QWK 0.923 vs
    DenseNet121 QWK 0.908 on APTOS 2019. B3 is larger than B0.

Why CBAM on stages 2-4:
    Stage 0/1 = low-level edges. Stages 2-4 carry lesion semantics
    (haemorrhages, exudates, neovascularisation).

Why multi-scale fusion:
    Grade 1/2 lesions show in stage 2/3. Grade 4 shows in stage 4.
    Fusing all three gives the CORAL head both signals.
    Source: MSTNet, Medical Image Analysis 2025.

Why dual heads:
    MSE on a continuous 0-4 target smooths the G2->G3 decision
    boundary — the 50-sample leak seen in the v3 confusion matrix.
    Source: Diagnostics 2023, regression-based EfficientNet for DR.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


# ============================================================================
#  CHANNEL ATTENTION (CBAM channel gate)
# ============================================================================

class ChannelAttention(nn.Module):
    """CBAM channel gate. ~1K params per module at r=16."""
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, mid, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.sigmoid(self.fc(self.avg_pool(x)) + self.fc(self.max_pool(x)))


# ============================================================================
#  FUSION HEAD
# ============================================================================

class FusionHead(nn.Module):
    """
    Fuses 568-d concatenated GAP features into a 256-d embedding.
    SiLU activation matches EfficientNet's internal activation.
    """
    def __init__(self, in_features: int = 568, hidden: int = 256, dropout: float = 0.4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.BatchNorm1d(hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ============================================================================
#  CORAL HEAD
# ============================================================================

class CoralHead(nn.Module):
    """K-1 binary logits for ordinal regression over K grades."""
    def __init__(self, in_features: int = 256, num_classes: int = 5):
        super().__init__()
        self.fc   = nn.Linear(in_features, num_classes - 1, bias=False)
        self.bias = nn.Parameter(torch.zeros(num_classes - 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x) + self.bias


# ============================================================================
#  REGRESSION HEAD
# ============================================================================

class RegressionHead(nn.Module):
    """Auxiliary scalar head. Only used during training (loss regulariser)."""
    def __init__(self, in_features: int = 256):
        super().__init__()
        self.fc = nn.Linear(in_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x).squeeze(-1)


# ============================================================================
#  MAIN MODEL
# ============================================================================

class EfficientNetB3CORAL(nn.Module):
    """
    EfficientNet-B3 + CBAM + multi-scale fusion + CORAL + regression head.
    RTX 2050 safe: image_size=384, batch_size=8, grad_checkpointing=True.
    """

    STAGE_CHANNELS = {2: 48, 3: 136, 4: 384}   # profiled from timm B3

    def __init__(
        self,
        num_classes: int = 5,
        pretrained: bool = True,
        dropout: float = 0.4,
        gamma: float = 2.0,
        label_smooth: float = 0.05,
        reg_weight: float = 0.3,
    ):
        super().__init__()
        self.num_classes  = num_classes
        self.gamma        = gamma
        self.label_smooth = label_smooth
        self.reg_weight   = reg_weight

        # Backbone — features_only returns list of stage outputs
        self.backbone = timm.create_model(
            'efficientnet_b3',
            pretrained=pretrained,
            features_only=True,
            out_indices=(2, 3, 4),
        )
        # Gradient checkpointing: halves activation memory, ~20% slower
        self.backbone.set_grad_checkpointing(enable=True)

        # CBAM modules
        self.cbam2 = ChannelAttention(self.STAGE_CHANNELS[2])
        self.cbam3 = ChannelAttention(self.STAGE_CHANNELS[3])
        self.cbam4 = ChannelAttention(self.STAGE_CHANNELS[4])

        self.gap = nn.AdaptiveAvgPool2d(1)

        fused_dim    = sum(self.STAGE_CHANNELS.values())   # 568
        self.fusion  = FusionHead(fused_dim, 256, dropout)
        self.coral   = CoralHead(256, num_classes)
        self.reg     = RegressionHead(256)

        self.register_buffer('alpha', torch.ones(num_classes - 1) / (num_classes - 1))

    # ── Freeze helper ──────────────────────────────────────────────────────

    def freeze_stage0(self):
        """Freeze conv_stem + bn1 (stage 0). Call once before training."""
        for name, param in self.backbone.named_parameters():
            if name.startswith('conv_stem') or name.startswith('bn1'):
                param.requires_grad = False
        frozen = sum(not p.requires_grad for p in self.backbone.parameters())
        print(f"  Stage 0 frozen ({frozen} params)")

    # ── CORAL utilities ────────────────────────────────────────────────────

    def compute_alpha(self, labels: torch.Tensor):
        """Per-threshold frequency weights. Call after dataset split."""
        counts = torch.bincount(labels, minlength=self.num_classes).float()
        alpha  = torch.zeros(self.num_classes - 1)
        for k in range(self.num_classes - 1):
            minority = min(counts[:k+1].sum(), counts[k+1:].sum())
            alpha[k] = 1.0 / (minority + 1e-6)
        alpha = alpha / alpha.sum()
        self.alpha.copy_(alpha)
        print(f"  CORAL alpha: {alpha.numpy().round(4)}")

    def coral_label_transform(self, labels: torch.Tensor) -> torch.Tensor:
        """Integer grade → CORAL binary target. Shape: (B, num_classes-1)."""
        B = labels.size(0)
        t = torch.zeros(B, self.num_classes - 1, device=labels.device)
        for i in range(B):
            t[i, :labels[i]] = 1.0
        return t

    # ── Loss ───────────────────────────────────────────────────────────────

    def coral_focal_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ls = self.label_smooth
        t  = targets * (1 - ls) + ls / 2.0
        p  = torch.sigmoid(logits).clamp(1e-7, 1 - 1e-7)
        bce = -(t * (1-p)**self.gamma * torch.log(p)
                + (1-t) * p**self.gamma * torch.log(1-p))
        return (bce * self.alpha.unsqueeze(0)).mean()

    def total_loss(
        self,
        logits: torch.Tensor,
        coral_targets: torch.Tensor,
        reg_pred: torch.Tensor,
        float_labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (total, coral_loss, reg_loss) for logging."""
        coral_loss = self.coral_focal_loss(logits, coral_targets)
        reg_loss   = F.mse_loss(reg_pred, float_labels)
        total      = coral_loss + self.reg_weight * reg_loss
        return total, coral_loss, reg_loss

    @staticmethod
    def predict(logits: torch.Tensor) -> torch.Tensor:
        """CORAL logits → integer grade (0-4)."""
        return (torch.sigmoid(logits) > 0.5).sum(dim=1)

    # ── Forward ────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (coral_logits, reg_pred)."""
        s2, s3, s4 = self.backbone(x)

        s2 = self.cbam2(s2)
        s3 = self.cbam3(s3)
        s4 = self.cbam4(s4)

        f  = torch.cat([self.gap(s2).flatten(1),
                         self.gap(s3).flatten(1),
                         self.gap(s4).flatten(1)], dim=1)   # (B, 568)

        embed    = self.fusion(f)          # (B, 256)
        logits   = self.coral(embed)       # (B, 4)
        reg_pred = self.reg(embed)         # (B,)

        return logits, reg_pred


# ── convenience alias used in training script ──────────────────────────────

def freeze_early_layers(model: EfficientNetB3CORAL, **_):
    """Drop-in replacement for the v2 freeze_early_layers call."""
    model.freeze_stage0()
