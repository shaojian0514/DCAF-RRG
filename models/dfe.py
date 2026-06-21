"""
Disease Classifier for Disease-aware Feature Extraction (DFE)

Implements:
1. Per-disease binary Softmax classification (Eq. 4)
2. Focal loss for handling severe label imbalance (Eq. 5)
3. Disease-aware feature extraction: F_d = ŷ · Φ + F_c (Eq. after 5)

The auxiliary disease classifier leverages fused features F_c to predict
disease annotations, and the predicted distribution is used to extract
disease-aware features F_d that emphasize disease-relevant visual patterns.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for multi-label disease classification.

    Mitigates severe label imbalance in multi-label disease classification.

    Reference:
        Lin et al., "Focal loss for dense object detection." ICCV 2017.

    Args:
        alpha: weighting factor for positive class (default 0.25)
        gamma: focusing parameter (default 2.0)
        reduction: 'mean' | 'sum' | 'none'
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred:   (B, d, 2) predicted logits per disease (binary)
            target: (B, d, 2) one-hot ground-truth labels
        Returns:
            scalar focal loss
        """
        # Compute softmax probabilities
        prob = F.softmax(pred, dim=-1)  # (B, d, 2)

        # Cross-entropy: -log(p_t)
        log_prob = F.log_softmax(pred, dim=-1)  # (B, d, 2)
        ce_loss = -(target * log_prob).sum(dim=-1)  # (B, d)

        # p_t: probability of the true class
        p_t = (prob * target).sum(dim=-1)  # (B, d)

        # Focal weight: (1 - p_t)^gamma
        focal_weight = (1.0 - p_t) ** self.gamma

        # Alpha weighting: alpha for positive class (target[:,:,1]==1), else (1-alpha)
        alpha_t = self.alpha * target[:, :, 1] + (1 - self.alpha) * target[:, :, 0]

        focal_loss = alpha_t * focal_weight * ce_loss  # (B, d)

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class DiseaseClassifier(nn.Module):
    """
    Disease Classifier for Disease-aware Feature Extraction (DFE).

    Given fused cross-modal features F_c ∈ R^{d×e}, predicts disease
    probabilities ŷ ∈ R^{d×2} via per-disease binary Softmax, and
    extracts disease-aware features F_d = ŷ · Φ + F_c.

    Eq. 4:
        ŷ = Softmax(F_c · Φ^T / sqrt(e))

    Eq. 5:
        L_cls = Focal(ŷ, y)

    Disease-aware feature:
        F_d = ŷ · Φ + F_c

    Args:
        num_diseases: number of disease categories d (default 14)
        embed_dim: embedding dimension e (default 256)
        focal_alpha: focal loss alpha
        focal_gamma: focal loss gamma
    """

    def __init__(
        self,
        num_diseases: int = 14,
        embed_dim: int = 256,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
    ):
        super().__init__()
        self.num_diseases = num_diseases
        self.embed_dim = embed_dim

        # Learnable disease embedding matrix Φ ∈ R^{2×e}
        # (binary: positive/negative per disease)
        self.disease_embedding = nn.Parameter(
            torch.randn(2, embed_dim) * 0.02
        )

        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)

    def forward(
        self,
        Fc: torch.Tensor,
        disease_labels: torch.Tensor = None,
    ) -> tuple:
        """
        Args:
            Fc: (B, d, e) fused cross-modal features (d = num_diseases, e = embed_dim)
            disease_labels: (B, d) binary ground-truth labels {0, 1} (optional, for training)
        Returns:
            Fd: (B, d, e) disease-aware features
            y_hat: (B, d, 2) predicted disease probabilities
            loss_cls: scalar focal loss (None if disease_labels not provided)
        """
        B, d, e = Fc.shape
        assert d == self.num_diseases, \
            f"Expected num_diseases={self.num_diseases}, got {d}"

        # Φ: (2, e) -> compute logits: (B, d, 2)
        # logit_{b,i,c} = F_c[b,i,:] · Φ[c,:] / sqrt(e)
        Phi = self.disease_embedding  # (2, e)
        logits = torch.matmul(Fc, Phi.T) / (e ** 0.5)  # (B, d, 2)

        # Per-disease binary Softmax (Eq. 4)
        y_hat = F.softmax(logits, dim=-1)  # (B, d, 2)

        # Disease-aware features: F_d = ŷ · Φ + F_c
        # ŷ: (B, d, 2), Φ: (2, e) -> (B, d, e)
        Fd = torch.matmul(y_hat, Phi) + Fc  # (B, d, e)

        # Compute focal loss if labels provided
        loss_cls = None
        if disease_labels is not None:
            # Convert labels to one-hot: (B, d) -> (B, d, 2)
            labels_onehot = self._to_onehot(disease_labels, Fc.device)
            loss_cls = self.focal_loss(logits, labels_onehot)

        return Fd, y_hat, loss_cls

    def _to_onehot(self, labels: torch.Tensor, device: torch.device) -> torch.Tensor:
        """
        Convert binary labels to one-hot format.

        Args:
            labels: (B, d) binary labels {0, 1}
        Returns:
            (B, d, 2) one-hot labels
        """
        B, d = labels.shape
        onehot = torch.zeros(B, d, 2, device=device)
        # labels == 1 -> class 1 (positive), labels == 0 -> class 0 (negative)
        # Handle uncertain labels (2) and blank (3) as negative
        pos_mask = (labels == 1)
        onehot[:, :, 1] = pos_mask.float()
        onehot[:, :, 0] = (~pos_mask).float()
        return onehot

    def get_disease_probs(self, y_hat: torch.Tensor) -> torch.Tensor:
        """
        Extract positive class probabilities for each disease.

        Args:
            y_hat: (B, d, 2) predicted disease probabilities
        Returns:
            (B, d) positive class probabilities
        """
        return y_hat[:, :, 1]  # probability of positive class
