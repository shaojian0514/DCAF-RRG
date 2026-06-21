"""
Disease-Consistent Retrieval Enhancer (DRE)

Implements:
1. Disease-prior Multi-positive Contrastive Learning (DMCL)
2. Disease-matching Constrained Image-Text Retrieval (DCR)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiseaseAwareProjection(nn.Module):
    """
    Learnable disease-aware projection layer.
    Projects image/text features into a shared disease-aware embedding space.
    Output shape: (batch, num_diseases, embed_dim)
    """

    def __init__(self, input_dim: int, num_diseases: int, embed_dim: int):
        super().__init__()
        self.num_diseases = num_diseases
        self.embed_dim = embed_dim
        # Project to (num_diseases * embed_dim) then reshape
        self.proj = nn.Linear(input_dim, num_diseases * embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, input_dim)
        Returns:
            (batch, num_diseases, embed_dim)
        """
        out = self.proj(x)  # (B, d*e)
        out = out.view(x.size(0), self.num_diseases, self.embed_dim)
        out = self.norm(out)
        return out


class MultiPositiveContrastiveLoss(nn.Module):
    """
    Disease-prior Multi-positive Contrastive Learning (DMCL).

    For each image in the batch, reports sharing identical disease labels
    are treated as positive samples; others as negatives.

    Loss (Eq. 1 in paper):
        L_con^D = -log( sum_{j in P_i} exp(sim(F_I, F_Tj)/tau) /
                        sum_{j in P_i U N_i} exp(sim(F_I, F_Tj)/tau) )
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        image_feats: torch.Tensor,
        text_feats: torch.Tensor,
        image_labels: torch.Tensor,
        text_labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Multi-positive contrastive loss (DMCL, Eq. 1).

        Positive pairs: image i and text j share at least one common positive
        disease label (multi-positive setting). This is more practical than
        exact-match, which would yield almost no positives in a batch.

        Args:
            image_feats: (B, d, e) disease-aware image embeddings
            text_feats:  (B, d, e) disease-aware text embeddings
            image_labels: (B, d) binary disease labels for images  {0, 1}
            text_labels:  (B, d) binary disease labels for texts   {0, 1}
        Returns:
            scalar contrastive loss
        """
        B = image_feats.size(0)

        # Global pooling over disease dimension -> (B, e)
        img_emb = F.normalize(image_feats.mean(dim=1), dim=-1)   # (B, e)
        txt_emb = F.normalize(text_feats.mean(dim=1), dim=-1)    # (B, e)

        # Similarity matrix: (B, B)
        sim_matrix = torch.matmul(img_emb, txt_emb.T) / self.temperature

        # Build positive mask (multi-positive):
        # image i and text j are positive if they share at least one positive disease.
        # image_labels: (B, d), text_labels: (B, d)
        img_lbl = image_labels.float()  # (B, d)
        txt_lbl = text_labels.float()   # (B, d)

        # pos_mask[i, j] = 1 if dot(img_lbl[i], txt_lbl[j]) > 0
        # i.e., they share at least one common positive disease
        pos_mask = (torch.matmul(img_lbl, txt_lbl.T) > 0).float()  # (B, B)

        # If a row has no positives (e.g., "no finding" only), treat diagonal as positive
        no_pos = (pos_mask.sum(dim=-1) == 0)
        if no_pos.any():
            eye = torch.eye(B, device=image_feats.device)
            pos_mask[no_pos] = eye[no_pos]

        # Numerical stability
        sim_matrix = sim_matrix - sim_matrix.max(dim=-1, keepdim=True)[0].detach()

        exp_sim = torch.exp(sim_matrix)  # (B, B)

        # For each image i: sum over positives / sum over all
        pos_sum = (exp_sim * pos_mask).sum(dim=-1)   # (B,)
        all_sum = exp_sim.sum(dim=-1)                 # (B,)

        # Avoid log(0)
        pos_sum = pos_sum.clamp(min=1e-8)

        loss = -torch.log(pos_sum / all_sum).mean()
        return loss


class DiseaseMatchingConstraint(nn.Module):
    """
    Disease-matching Constrained Image-Text Retrieval (DCR).

    Given top-k retrieved text embeddings, enforces that their predicted
    disease annotations match those of the input image.

    Constraint (Eq. 2 in paper):
        gamma = (1/k) * sum_{i=1}^{k} L_CE(y^I, y_i^T)
    """

    def __init__(self, embed_dim: int, num_diseases: int):
        super().__init__()
        # Small MLP to predict disease labels from text embeddings
        self.disease_predictor = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, num_diseases),
        )

    def forward(
        self,
        retrieved_text_feats: torch.Tensor,
        image_disease_labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            retrieved_text_feats: (B, k, embed_dim) top-k retrieved text features
            image_disease_labels: (B, num_diseases) ground-truth binary labels for images
        Returns:
            scalar constraint loss gamma
        """
        B, k, e = retrieved_text_feats.shape

        # Predict disease labels for each retrieved text
        # (B*k, e) -> (B*k, num_diseases)
        preds = self.disease_predictor(retrieved_text_feats.view(B * k, e))
        preds = preds.view(B, k, -1)  # (B, k, num_diseases)

        # Expand image labels to match: (B, 1, num_diseases) -> (B, k, num_diseases)
        targets = image_disease_labels.float().unsqueeze(1).expand_as(preds)

        # Binary cross-entropy per disease, averaged over k and diseases
        gamma = F.binary_cross_entropy_with_logits(preds, targets)
        return gamma


class DiseaseConsistentRetrievalEnhancer(nn.Module):
    """
    Disease-Consistent Retrieval Enhancer (DRE).

    Combines:
    - DiseaseAwareProjection for image and text
    - MultiPositiveContrastiveLoss (DMCL)
    - DiseaseMatchingConstraint (DCR)
    """

    def __init__(
        self,
        image_feat_dim: int,
        text_feat_dim: int,
        num_diseases: int = 14,
        embed_dim: int = 256,
        temperature: float = 0.07,
    ):
        super().__init__()
        self.num_diseases = num_diseases
        self.embed_dim = embed_dim

        self.image_proj = DiseaseAwareProjection(image_feat_dim, num_diseases, embed_dim)
        self.text_proj = DiseaseAwareProjection(text_feat_dim, num_diseases, embed_dim)

        self.contrastive_loss = MultiPositiveContrastiveLoss(temperature=temperature)
        self.matching_constraint = DiseaseMatchingConstraint(embed_dim, num_diseases)

    def encode_image(self, image_avg_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image_avg_feat: (B, image_feat_dim) global average-pooled image features
        Returns:
            (B, num_diseases, embed_dim) disease-aware image embeddings
        """
        return self.image_proj(image_avg_feat)

    def encode_text(self, text_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            text_feat: (B, text_feat_dim) global text features
        Returns:
            (B, num_diseases, embed_dim) disease-aware text embeddings
        """
        return self.text_proj(text_feat)

    def compute_contrastive_loss(
        self,
        image_feats: torch.Tensor,
        text_feats: torch.Tensor,
        image_labels: torch.Tensor,
        text_labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute DMCL loss."""
        return self.contrastive_loss(image_feats, text_feats, image_labels, text_labels)

    def compute_matching_constraint(
        self,
        retrieved_text_feats: torch.Tensor,
        image_disease_labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute DCR constraint loss."""
        return self.matching_constraint(retrieved_text_feats, image_disease_labels)

    def retrieve_top_k(
        self,
        image_feats: torch.Tensor,
        text_feat_bank: torch.Tensor,
        k: int,
    ) -> tuple:
        """
        Retrieve top-k most similar text features from the bank.

        Args:
            image_feats: (B, num_diseases, embed_dim) disease-aware image embeddings
            text_feat_bank: (N, num_diseases, embed_dim) all text embeddings in bank
            k: number of top retrievals
        Returns:
            top_k_feats: (B, k, embed_dim) retrieved text features (mean over diseases)
            top_k_indices: (B, k) indices into text_feat_bank
        """
        # Global pooling over disease dimension
        img_emb = F.normalize(image_feats.mean(dim=1), dim=-1)   # (B, e)
        txt_emb = F.normalize(text_feat_bank.mean(dim=1), dim=-1)  # (N, e)

        # Similarity: (B, N)
        sim = torch.matmul(img_emb, txt_emb.T)

        # Top-k
        top_k_scores, top_k_indices = sim.topk(k, dim=-1)  # (B, k)

        # Gather top-k text features (mean over diseases)
        top_k_feats = txt_emb[top_k_indices]  # (B, k, e)

        return top_k_feats, top_k_indices
