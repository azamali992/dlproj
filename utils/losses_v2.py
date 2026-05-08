"""
utils/losses_v2.py — CORAL + Focal Loss with label smoothing
=============================================================

Changes from v1:
1. coral_label_transform() accepts a `smooth` parameter (default 0.1).
   Ordinal label smoothing prevents the model from predicting extreme
   logit values, which destabilises CORAL threshold learning.
   e.g. Grade 3 target [1,1,1,0] becomes [0.95, 0.95, 0.95, 0.05].
2. compute_alpha() explicitly moves result to CPU — avoids device mismatch
   when labels arrive on GPU.
3. No other logic changes — the CORAL + Focal formulation is correct.
"""

import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F


class CORALModule_v2(nn.Module):
    """
    CORAL + Focal Loss + class-imbalance alpha weighting.
    Adds ordinal label smoothing vs v1.
    """

    def __init__(self, in_features, num_classes=5, gamma=2.0, label_smooth=0.05):
        super().__init__()

        self.num_classes = num_classes
        self.num_thresholds = num_classes - 1
        self.gamma = gamma
        self.label_smooth = label_smooth

        # Shared weight vector + per-threshold bias (CORAL Theorem 1)
        self.fc = nn.Linear(in_features, 1, bias=False)
        self.bias = nn.Parameter(torch.zeros(self.num_thresholds))

        self.alpha = None

    # ------------------------------------------------------------------
    def forward(self, x):
        # [B, in_features] → [B, K-1]
        return self.fc(x) + self.bias

    # ------------------------------------------------------------------
    def coral_label_transform(self, y, smooth=None):
        """
        y: [B] integer labels 0 … K-1
        Returns: [B, K-1] ordinal binary matrix, optionally smoothed.

        Label smoothing: instead of hard 0/1 targets, use
            1 → (1 - smooth)   and   0 → smooth
        This prevents the model from driving logits to ±∞ to satisfy
        hard targets, which destabilises threshold learning.
        """
        if smooth is None:
            smooth = self.label_smooth

        batch_size = y.size(0)
        K = self.num_thresholds

        y_coral = torch.zeros(batch_size, K, device=y.device)
        for i in range(K):
            y_coral[:, i] = (y > i).float()

        if smooth > 0.0:
            # Soft targets: 1→(1-smooth), 0→smooth
            y_coral = y_coral * (1.0 - smooth) + smooth * 0.5

        return y_coral

    # ------------------------------------------------------------------
    def compute_alpha(self, labels):
        """
        Threshold-level imbalance weights per Cao et al. (2019) Eq. 7:
            λ^(k) = sqrt(M_k) / max_i(sqrt(M_i))
            M_k = max(S_k, N − S_k),  S_k = #{i : y_i > r_k}

        Result stored on CPU (moved to device inside loss()).
        """
        labels = labels.cpu()
        N = labels.size(0)
        alpha_threshold = []

        for k in range(self.num_thresholds):
            S_k = (labels > k).sum().float()
            M_k = torch.max(S_k, torch.tensor(float(N)) - S_k)
            alpha_threshold.append(M_k.sqrt())

        alpha = torch.stack(alpha_threshold)
        self.alpha = alpha / alpha.max()
        return self.alpha

    # ------------------------------------------------------------------
    def loss(self, logits, targets):
        """
        logits:  [B, K-1]
        targets: [B, K-1]  (already smoothed via coral_label_transform)
        """
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

        probs = torch.sigmoid(logits)
        # p_t: probability assigned to the correct target
        pt = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - pt) ** self.gamma

        loss = focal_weight * bce

        if self.alpha is not None:
            alpha = self.alpha.to(logits.device).view(1, -1)
            loss = loss * alpha
        else:
            warnings.warn(
                "CORALModule_v2.loss called without alpha weights. "
                "Call compute_alpha(labels) before training.",
                UserWarning,
                stacklevel=2,
            )

        return loss.mean()

    # ------------------------------------------------------------------
    def predict(self, logits):
        """Sum of thresholds exceeded → predicted ordinal class."""
        probs = torch.sigmoid(logits)
        return (probs > 0.5).sum(dim=1)

    # ------------------------------------------------------------------
    def threshold_probs(self, logits):
        """P(y > k) for each threshold k."""
        return torch.sigmoid(logits)
