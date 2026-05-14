"""
utils/losses_v5.py — CORAL with independent per-threshold projections
======================================================================

Changes from v2
---------------
1. INDEPENDENT PER-THRESHOLD PROJECTIONS (the main fix)
   v2 used a single shared weight vector:
       self.fc = nn.Linear(in_features, 1, bias=False)   # [1024 → 1]
       output  = self.fc(x) + self.bias                  # [B, 4]

   The CORAL paper proves rank-monotonicity is preserved with shared weights
   under UNWEIGHTED BCE. The moment you add focal weighting and alpha
   weighting, that proof no longer holds — the shared vector gets pulled in
   contradictory directions by different thresholds.

   More importantly, the features that separate G0/G1 (presence of tiny
   microaneurysms) are DIFFERENT from the features that separate G3/G4
   (neovascularisation vs severe haemorrhages). A single linear projection
   cannot be optimal for all four boundaries simultaneously.

   v3 uses K-1 independent projections, one per threshold:
       self.fc = nn.ModuleList([
           nn.Linear(in_features, 1, bias=True) for _ in range(K-1)
       ])
   Each projection learns which feature dimensions matter for ITS boundary.
   The bias is now absorbed into each Linear (bias=True), so the separate
   self.bias Parameter is removed.

   Ordinal consistency (rank-monotonicity of predictions) is enforced via
   a regularization term added to the loss:
       reg = sum(ReLU(logit[k+1] - logit[k]))  for k in 0..K-3
   This penalises any batch where a higher threshold fires more confidently
   than a lower threshold — pushing the projections to respect ordinality
   without requiring the CORAL shared-weight theorem.

2. ALPHA ON ORIGINAL IMBALANCED LABELS
   v2 computed alpha AFTER oversampling — all classes were ~800, so alpha
   was nearly uniform [0.79, 0.87, 0.87, 1.0] and did almost nothing.

   v3 stores the original label distribution and computes alpha on it.
   On the raw APTOS distribution (G0=1444, G1=296, G2=799, G3=154, G4=236):
       threshold 0 (G0 vs rest): S0=1485, M0=max(1485,2177)=2177, α≈1.00
       threshold 1 (G0+G1 vs rest): S1=1189, M1=max(1189,2473)=2473, α≈1.07
       threshold 2 (G0-G2 vs G3+G4): S2=390, M2=max(390,3272)=3272, α≈1.23
       threshold 3 (G0-G3 vs G4): S3=236, M3=max(236,3426)=3426, α≈1.25
   Thresholds 2 and 3 (the G2/G3 and G3/G4 boundaries) now get ~25% more
   gradient weight — exactly where the model is weakest.

3. LABEL SMOOTHING REMOVED (was already 0.0 in training, now explicit)

4. ORDINAL CONSISTENCY REGULARIZATION
   lambda_ord=0.1 controls the regularization strength.
   Set to 0.0 to disable (pure independent projections, no reg).
"""

import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F


class CORALModule_v5(nn.Module):
    """
    CORAL + Focal Loss with:
    - Independent per-threshold linear projections
    - Alpha computed on original imbalanced label distribution
    - Ordinal consistency regularization
    """

    def __init__(self, in_features, num_classes=5, gamma=2.0, lambda_ord=0.1):
        super().__init__()

        self.num_classes    = num_classes
        self.num_thresholds = num_classes - 1   # K-1 = 4
        self.gamma          = gamma
        self.lambda_ord     = lambda_ord

        # Independent projection per threshold — each learns its own
        # feature combination for its specific clinical boundary.
        # bias=True absorbs the old self.bias Parameter.
        self.fc = nn.ModuleList([
            nn.Linear(in_features, 1, bias=True)
            for _ in range(self.num_thresholds)
        ])

        # Initialize biases to spread thresholds evenly in logit space
        # so training starts with reasonable ordinal ordering.
        # logit(-0.6) ≈ -0.4, logit(0.6) ≈ 0.4 — gives non-degenerate start.
        init_biases = torch.linspace(1.5, -1.5, self.num_thresholds)
        for k, layer in enumerate(self.fc):
            nn.init.xavier_uniform_(layer.weight)
            layer.bias.data.fill_(init_biases[k].item())

        self.alpha = None   # set by compute_alpha()

    # ------------------------------------------------------------------
    def forward(self, x):
        """
        x: [B, in_features]
        Returns: [B, K-1] — one logit per threshold
        """
        return torch.cat([self.fc[k](x) for k in range(self.num_thresholds)], dim=1)

    # ------------------------------------------------------------------
    def coral_label_transform(self, y):
        """
        y: [B] integer labels 0 … K-1
        Returns: [B, K-1] ordinal binary matrix.
            y_coral[i, k] = 1 if y[i] > k, else 0

        No label smoothing — focal loss handles hard/easy example weighting.
        """
        batch_size = y.size(0)
        K = self.num_thresholds
        y_coral = torch.zeros(batch_size, K, device=y.device)
        for k in range(K):
            y_coral[:, k] = (y > k).float()
        return y_coral

    # ------------------------------------------------------------------
    def compute_alpha(self, labels):
        """
        Compute threshold-level imbalance weights on the ORIGINAL
        imbalanced label distribution (before any oversampling).

        Per Cao et al. (2019) Eq. 7:
            S_k = #{i : y_i > k}
            M_k = max(S_k, N - S_k)
            alpha_k = sqrt(M_k) / max_j(sqrt(M_j))

        Pass train_df['diagnosis'] (BEFORE balancing) to get meaningful
        weights. If you pass the balanced df, all classes are ~equal and
        alpha is near-uniform — doing nothing.

        Prints the computed weights so you can verify they're not uniform.
        """
        labels = labels.cpu()
        N = labels.size(0)
        alpha_threshold = []

        print("  Computing alpha on original imbalanced distribution:")
        for k in range(self.num_thresholds):
            S_k = (labels > k).sum().float()
            M_k = torch.max(S_k, torch.tensor(float(N)) - S_k)
            alpha_threshold.append(M_k.sqrt())
            print(f"    threshold {k}: S_k={int(S_k)}, M_k={M_k:.0f}, "
                  f"sqrt(M_k)={M_k.sqrt():.2f}")

        alpha = torch.stack(alpha_threshold)
        self.alpha = alpha / alpha.max()
        print(f"  Alpha weights (normalised): {self.alpha.tolist()}")
        return self.alpha

    # ------------------------------------------------------------------
    def _ordinal_consistency_reg(self, logits):
        """
        Penalise violations of ordinal monotonicity:
            P(y > k) should be >= P(y > k+1)
        In logit space: logit[k] should be >= logit[k+1]

        Penalty = sum_k ReLU(logit[k+1] - logit[k])
        This is 0 when thresholds are monotone (correct) and positive
        when a higher threshold fires more confidently than a lower one.
        """
        reg = 0.0
        for k in range(self.num_thresholds - 1):
            # logit[k+1] > logit[k] means P(y>k+1) > P(y>k) — violation
            violation = F.relu(logits[:, k+1] - logits[:, k])
            reg = reg + violation.mean()
        return reg

    # ------------------------------------------------------------------
    def loss(self, logits, targets):
        """
        logits:  [B, K-1]
        targets: [B, K-1] from coral_label_transform (hard 0/1)

        Loss = mean(focal_weight * BCE) * alpha  +  lambda_ord * consistency_reg
        """
        # Focal-weighted BCE
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        probs = torch.sigmoid(logits)
        pt = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - pt) ** self.gamma
        loss = focal_weight * bce

        # Alpha weighting (meaningful only if compute_alpha was called on
        # original imbalanced labels)
        if self.alpha is not None:
            alpha = self.alpha.to(logits.device).view(1, -1)
            loss = loss * alpha
        else:
            warnings.warn(
                "CORALModule_v5.loss called without alpha weights. "
                "Call compute_alpha(original_train_labels) before training.",
                UserWarning, stacklevel=2,
            )

        main_loss = loss.mean()

        # Ordinal consistency regularization
        if self.lambda_ord > 0.0:
            reg = self._ordinal_consistency_reg(logits)
            return main_loss + self.lambda_ord * reg

        return main_loss

    # ------------------------------------------------------------------
    def predict(self, logits):
        """
        Sum of thresholds exceeded → predicted ordinal class.
        Identical interface to v2 — training script unchanged.
        """
        probs = torch.sigmoid(logits)
        return (probs > 0.5).sum(dim=1)

    # ------------------------------------------------------------------
    def threshold_probs(self, logits):
        """P(y > k) for each threshold k."""
        return torch.sigmoid(logits)
