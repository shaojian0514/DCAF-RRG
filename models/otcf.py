"""
Optimal Transport-Enhanced Cross-Modal Fusion (OTCF)

Implements Multi-Prompt Sinkhorn Attention (MPSA) for visual-text cross-attention,
grounded in optimal transport (OT) theory.

Reference:
    Kim et al., "OtSeg: Multi-prompt Sinkhorn attention for zero-shot semantic
    segmentation." ECCV 2024.

The module selectively integrates clinically relevant textual knowledge conditioned
on visual evidence, while suppressing irrelevant content.

Eq. 3 in paper:
    F_c = MPSA(QK^T) V
    F_c = f_proj(LayerNorm(F_c + F_v))
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def sinkhorn_log_exp_sum(
    C: torch.Tensor,
    mu: torch.Tensor,
    nu: torch.Tensor,
    epsilon: float = 0.05,
    max_iter: int = 100,
    thresh: float = 1e-6,
) -> torch.Tensor:
    """
    Sinkhorn algorithm in log-exp-sum form (numerically stable).

    Solves the entropy-regularised optimal transport problem:
        T* = argmin_{T in Pi(mu,nu)} <C, T> - epsilon * H(T)

    Directly ported from REVTAF / OtSeg implementation.

    Args:
        C:       (..., M, N) cost matrix  (non-negative, e.g. 1 - cosine_sim)
        mu:      (..., M)    source marginal (uniform: 1/M)
        nu:      (..., N)    target marginal (uniform: 1/N)
        epsilon: regularisation strength (default 0.05)
        max_iter: maximum Sinkhorn iterations
        thresh:  convergence threshold on u update
    Returns:
        T: (..., M, N) transport plan (doubly-stochastic up to marginals)
    """

    def _log_boltzmann_kernel(u, v, C):
        # kernel_{m,n} = (-C_{m,n} + u_m + v_n) / epsilon
        return (-C + u.unsqueeze(-1) + v.unsqueeze(-2)) / epsilon

    u = torch.zeros_like(mu)
    v = torch.zeros_like(nu)

    for _ in range(max_iter):
        u0 = u
        K = _log_boltzmann_kernel(u, v, C)
        u_ = torch.log(mu + 1e-8) - torch.logsumexp(K, dim=-1)
        u = epsilon * u_ + u

        K_t = _log_boltzmann_kernel(u, v, C).transpose(-2, -1).contiguous()
        v_ = torch.log(nu + 1e-8) - torch.logsumexp(K_t, dim=-1)
        v = epsilon * v_ + v

        err = (u - u0).abs().mean()
        if err.item() < thresh:
            break

    K = _log_boltzmann_kernel(u, v, C)
    T = torch.exp(K)
    return T


class MultiPromptSinkhornAttention(nn.Module):
    """
    Multi-Prompt Sinkhorn Attention (MPSA).

    Replaces standard multi-head cross-attention with OT-based globally
    constrained attention, enabling selective integration of clinically
    relevant textual knowledge.

    Implementation follows REVTAF / OtSeg:
      1. L2-normalise Q and K.
      2. Compute cosine similarity sim = Q @ K^T.
      3. Use cost C = 1 - sim (non-negative).
      4. Solve OT with Sinkhorn to get transport plan T.
      5. Output = T @ V  (weighted sum of values).

    Args:
        embed_dim: feature dimension
        num_heads: number of attention heads
        dropout: dropout rate
        sinkhorn_eps: Sinkhorn regularisation epsilon (default 0.05)
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        sinkhorn_eps: float = 0.05,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.sinkhorn_eps = sinkhorn_eps

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.attn_drop = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            query: (B, Lq, embed_dim)  visual patch features F_v
            key:   (B, Lk, embed_dim)  textual features F_g
            value: (B, Lk, embed_dim)  textual features F_g
        Returns:
            (B, Lq, embed_dim) fused features
        """
        B, Lq, _ = query.shape
        Lk = key.shape[1]
        H = self.num_heads
        d = self.head_dim

        # Project
        Q = self.q_proj(query)   # (B, Lq, D)
        K = self.k_proj(key)     # (B, Lk, D)
        V = self.v_proj(value)   # (B, Lk, D)

        # Reshape to multi-head: (B*H, L, d)
        Q = Q.view(B, Lq, H, d).permute(0, 2, 1, 3).reshape(B * H, Lq, d)
        K = K.view(B, Lk, H, d).permute(0, 2, 1, 3).reshape(B * H, Lk, d)
        V = V.view(B, Lk, H, d).permute(0, 2, 1, 3).reshape(B * H, Lk, d)

        # L2 normalise Q and K (following OtSeg / REVTAF)
        Q = F.normalize(Q, dim=-1, p=2)   # (B*H, Lq, d)
        K = F.normalize(K, dim=-1, p=2)   # (B*H, Lk, d)

        # Cosine similarity: (B*H, Lq, Lk)
        sim = torch.bmm(Q, K.transpose(-2, -1))

        # Cost matrix: C = 1 - sim  (non-negative since sim in [-1,1])
        C = 1.0 - sim   # (B*H, Lq, Lk)

        # Uniform marginals
        BH = B * H
        mu = torch.full((BH, Lq), 1.0 / Lq, dtype=sim.dtype, device=sim.device)
        nu = torch.full((BH, Lk), 1.0 / Lk, dtype=sim.dtype, device=sim.device)

        # Sinkhorn transport plan: (B*H, Lq, Lk)
        T = sinkhorn_log_exp_sum(C, mu, nu, epsilon=self.sinkhorn_eps)
        T = self.attn_drop(T)

        # Weighted sum of values: (B*H, Lq, d)
        out = torch.bmm(T, V)

        # Merge heads: (B, Lq, D)
        out = out.view(B, H, Lq, d).permute(0, 2, 1, 3).contiguous().view(B, Lq, self.embed_dim)
        out = self.out_proj(out)

        return out


class OptimalTransportCrossModalFusion(nn.Module):
    """
    Optimal Transport-Enhanced Cross-Modal Fusion (OTCF) module.

    Given visual features F_v and global textual features F_g from the
    most relevant retrieved report, produces fused features F_c via MPSA,
    followed by residual connection and layer normalization.

    Eq. 3 in paper:
        F_c = MPSA(QK^T) V
        F_c = f_proj(LayerNorm(F_c + F_v))

    Args:
        visual_dim: dimension of input visual features (before projection)
        text_dim:   dimension of input text features (before projection)
        embed_dim:  internal embedding dimension
        num_heads:  number of attention heads
        dropout:    dropout rate
        sinkhorn_eps: Sinkhorn regularisation epsilon
    """

    def __init__(
        self,
        visual_dim: int,
        text_dim: int,
        embed_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1,
        sinkhorn_eps: float = 0.05,
        # kept for backward-compat with callers that pass sinkhorn_iters
        sinkhorn_iters: int = 3,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        # Project visual and text features to common embed_dim
        self.visual_proj = nn.Linear(visual_dim, embed_dim)
        self.text_proj = nn.Linear(text_dim, embed_dim)

        # MPSA cross-attention (OT-based)
        self.mpsa = MultiPromptSinkhornAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            sinkhorn_eps=sinkhorn_eps,
        )

        # Post-fusion: residual + LayerNorm + projection (Eq. 3)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.f_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(
        self,
        visual_feats: torch.Tensor,
        text_feats: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            visual_feats: (B, Lv, visual_dim) patch-level visual features F_v
            text_feats:   (B, Lt, text_dim)   retrieved text features F_g
        Returns:
            F_c: (B, Lv, embed_dim) fused cross-modal features
        """
        # Project to common dimension
        Fv = self.visual_proj(visual_feats)   # (B, Lv, embed_dim)
        Fg = self.text_proj(text_feats)        # (B, Lt, embed_dim)

        # MPSA: Q=Fv (visual queries), K=V=Fg (text keys/values)
        Fc = self.mpsa(query=Fv, key=Fg, value=Fg)

        # Residual connection + LayerNorm + projection (Eq. 3)
        Fc = self.f_proj(self.layer_norm(Fc + Fv))

        return Fc
