"""
Personalized IPS Loss — the core contribution of the paper.

Loss formulation:
    weight(u, i, t) = 1 / (e_pop(i)^α(u)  ×  e_trend(i, t)^(1−β(u)))
    L               = weight(u, i, t) × (−log P(i_{t+1}))

GPU optimisations vs the original:
  - alpha_tensor and beta_tensor are pre-built from identity_scorer.scores
    and registered as buffers so they move to the correct device with .to(device).
  - pop_tensor was already a buffer.
  - _compute_weights() replaces the Python for-loop with vectorised tensor ops.
    The only remaining Python work is building the e_trend batch tensor (a sparse
    dict; cannot be pre-materialised without huge memory cost).
  - All pow(), mul(), clamp() operations run on GPU.

Ablation switches:
    use_alpha=False → α(u) = 0  → pop correction disabled
    use_beta=False  → β(u) = 1  → trend correction disabled (exponent = 0)
"""
from __future__ import annotations
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.features import EPSILON, IdentityScorer


class PersonalizedIPSLoss(nn.Module):
    def __init__(
        self,
        e_pop: Dict[int, float],
        e_trend: Dict[int, Dict[int, float]],
        identity_scorer: IdentityScorer,
        num_items: int,
        max_weight: float = 10.0,
        use_alpha: bool = True,
        use_beta: bool = True,
    ) -> None:
        super().__init__()
        self.e_trend = e_trend
        self.num_items = num_items
        self.max_weight = max_weight
        self.use_alpha = use_alpha
        self.use_beta = use_beta

        # ── pop exposure buffer  [num_items+1] ───────────────────────────────
        pop_tensor = torch.full((num_items + 1,), EPSILON)
        for item_idx, ep in e_pop.items():
            if 0 < item_idx <= num_items:
                pop_tensor[item_idx] = max(ep, EPSILON)
        self.register_buffer("pop_tensor", pop_tensor)

        # ── alpha / beta buffers indexed by user_idx ─────────────────────────
        # Default 0.5 for any user not in scorer (safe fallback).
        if identity_scorer.scores:
            max_user = max(identity_scorer.scores.keys())
            alpha_t = torch.full((max_user + 1,), 0.5)
            beta_t = torch.full((max_user + 1,), 0.5)
            for uid, (a, b) in identity_scorer.scores.items():
                alpha_t[uid] = float(a)
                beta_t[uid] = float(b)
        else:
            alpha_t = torch.tensor([0.5])
            beta_t = torch.tensor([0.5])
        self.register_buffer("alpha_tensor", alpha_t)
        self.register_buffer("beta_tensor", beta_t)

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        logits: torch.Tensor,
        batch: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        target = batch["target_item"]          # [B]
        user_ids = batch["user_idx"]           # [B]
        time_bins = batch["target_time_bin"]   # [B]

        logits = logits.clone()
        logits[:, 0] = float("-inf")

        ce_loss = F.cross_entropy(logits, target, reduction="none")   # [B]

        weights = self._compute_weights(target, user_ids, time_bins)  # [B]
        weights = weights / (weights.mean() + 1e-8)

        return (weights * ce_loss).mean()

    # ── vectorised weight computation ─────────────────────────────────────────

    def _compute_weights(
        self,
        target_items: torch.Tensor,   # [B]  on device
        user_ids: torch.Tensor,       # [B]  on device
        time_bins: torch.Tensor,      # [B]  on device
    ) -> torch.Tensor:
        device = target_items.device

        # Alpha and beta — safe clamp handles user_idx > tensor size
        uid_clamped = user_ids.clamp(max=self.alpha_tensor.size(0) - 1)
        alpha = self.alpha_tensor[uid_clamped]   # [B]
        beta = self.beta_tensor[uid_clamped]     # [B]

        # Pop exposure from buffer
        ep = self.pop_tensor[target_items]       # [B]

        # Trend exposure — sparse dict lookup, result moved to device
        et_list = [
            max(self.e_trend.get(item.item(), {}).get(tb.item(), EPSILON), EPSILON)
            for item, tb in zip(target_items, time_bins)
        ]
        et = torch.tensor(et_list, dtype=torch.float32, device=device)   # [B]

        # Vectorised weight formula
        if self.use_alpha:
            pop_term = ep.pow(alpha)
        else:
            pop_term = torch.ones(target_items.size(0), dtype=torch.float32, device=device)

        if self.use_beta:
            trend_exp = 1.0 - beta
        else:
            trend_exp = torch.zeros(target_items.size(0), dtype=torch.float32, device=device)

        trend_term = et.pow(trend_exp)

        denom = (pop_term * trend_term).clamp(min=EPSILON)
        return (1.0 / denom).clamp(max=self.max_weight)
