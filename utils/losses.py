import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F


class CORALModule(nn.Module):
    """
    Complete CORAL + Focal Loss + Imbalance Handling Module
    for ordinal regression (e.g., Diabetic Retinopathy grading)
    """

    def __init__(self, in_features, num_classes=5, gamma=2.0):
        super().__init__()

        self.num_classes = num_classes
        self.num_thresholds = num_classes - 1
        self.gamma = gamma

        # CORAL prediction head: one shared weight vector, K-1 independent biases.
        # This enforces rank-monotonicity (Theorem 1 in Cao et al., 2019).
        self.fc = nn.Linear(in_features, 1, bias=False)   # shared weights
        self.bias = nn.Parameter(torch.zeros(self.num_thresholds))  # per-threshold biases

        # alpha will be computed later from dataset
        self.alpha = None

    # =========================================================
    # 1. CORAL HEAD FORWARD
    # =========================================================
    def forward(self, x):
        # x: [B, in_features]
        # fc(x): [B, 1] + bias: [K-1] → [B, K-1] via broadcasting
        return self.fc(x) + self.bias

    # =========================================================
    # 2. LABEL TRANSFORM (CORAL ENCODING)
    # =========================================================
    def coral_label_transform(self, y):
        """
        y: [B] scalar labels (0 ... K-1)
        returns: [B, K-1] ordinal binary matrix
        """
        batch_size = y.size(0)
        K = self.num_classes - 1

        y_coral = torch.zeros(batch_size, K, device=y.device)

        for i in range(K):
            y_coral[:, i] = (y > i).float()

        return y_coral

    # =========================================================
    # 3. ALPHA COMPUTATION (IMBALANCE HANDLING)
    # =========================================================
    def compute_alpha(self, labels):
        """
        labels: full dataset labels [N]
        returns threshold-level alpha weights per Eq. 7 of Cao et al. (2019):
            λ^(k) = sqrt(M_k) / max_i(sqrt(M_i))
        where M_k = max(S_k, N - S_k) and S_k = #{i : y_i > r_k}.
        """
        N = labels.size(0)
        alpha_threshold = []

        for k in range(self.num_thresholds):
            S_k = (labels > k).sum().float()
            M_k = torch.max(S_k, torch.tensor(N, dtype=torch.float) - S_k)
            alpha_threshold.append(M_k.sqrt())

        alpha = torch.stack(alpha_threshold)
        self.alpha = alpha / alpha.max()

        return self.alpha

    # =========================================================
    # 4. CORAL FOCAL LOSS
    # =========================================================
    def loss(self, logits, targets):
        """
        logits:  [B, K-1]
        targets: [B, K-1]
        """

        # BCE per threshold
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )

        probs = torch.sigmoid(logits)

        # p_t (correct class probability)
        pt = probs * targets + (1 - probs) * (1 - targets)

        # focal modulation
        focal_weight = (1 - pt) ** self.gamma

        loss = focal_weight * bce

        # alpha weighting
        if self.alpha is not None:
            alpha = self.alpha.to(logits.device).view(1, -1)
            loss = loss * alpha
        else:
            warnings.warn(
                "CORALModule.loss called without alpha weights. "
                "Call compute_alpha(labels) before training to enable imbalance handling.",
                UserWarning,
                stacklevel=2,
            )

        return loss.mean()

    # =========================================================
    # 5. PREDICTION DECODING
    # =========================================================
    def predict(self, logits):
        """
        Convert CORAL outputs → class labels
        """
        probs = torch.sigmoid(logits)
        return (probs > 0.5).sum(dim=1)

    # =========================================================
    # 6. OPTIONAL: GET THRESHOLD PROBABILITIES
    # =========================================================
    def threshold_probs(self, logits):
        """
        Returns P(y > k) for each threshold k — one value per threshold, not per class.
        """
        return torch.sigmoid(logits)