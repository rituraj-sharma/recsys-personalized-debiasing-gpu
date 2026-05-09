"""
Personalized IPS Loss — the core contribution of the paper.

Loss formulation:
    weight(u, i, t) = 1 / (e_pop(i)^α(u)  ×  e_trend(i, t)^(1−β(u)))
    L               = weight(u, i, t) × (−log P(i_{t+1}))

Two variants selected automatically via identity_scorer.is_learnable():

    Formula variant (FormulaIdentityScorer):
        α and β are fixed per user, pre-computed from training data.
        Weight computation is fully vectorised — no gradients into scorer.

    Learned variant (LearnedIdentityScorer):
        α and β come from an MLP forward pass each step.
        Gradients flow back into the MLP weights.

Ablation switches:
    use_alpha=False → α(u) = 0  → pop correction disabled
    use_beta=False  → β(u) = 1  → trend correction disabled (exponent = 0)
"""
from __future__ import annotations
from typing import Dict, Tuple

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
        self.identity_scorer = identity_scorer
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

        # ── alpha / beta buffers indexed by user_idx (formula variant) ───────
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

        # ── per-epoch weight stats (reset each epoch by Trainer) ─────────────
        self._weight_sum = 0.0
        self._weight_sq_sum = 0.0
        self._weight_max = 0.0
        self._weight_n = 0

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        logits: torch.Tensor,
        batch: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        target = batch["target_item"]          # [B]
        user_ids = batch["user_idx"]           # [B]
        time_bins = batch["target_time_bin"]   # [B]
        device = logits.device

        logits = logits.clone()
        logits[:, 0] = float("-inf")

        ce_loss = F.cross_entropy(logits, target, reduction="none")   # [B]

        weights = self._compute_weights(target, user_ids, time_bins, device)  # [B]

        # accumulate raw weight stats before normalisation
        with torch.no_grad():
            self._weight_sum += weights.sum().item()
            self._weight_sq_sum += (weights ** 2).sum().item()
            self._weight_max = max(self._weight_max, weights.max().item())
            self._weight_n += weights.numel()

        weights = weights / (weights.mean() + 1e-8)
        return (weights * ce_loss).mean()

    # ── weight computation ────────────────────────────────────────────────────

    def _compute_weights(
        self,
        target_items: torch.Tensor,   # [B]
        user_ids: torch.Tensor,       # [B]
        time_bins: torch.Tensor,      # [B]
        device: torch.device,
    ) -> torch.Tensor:
        if self.identity_scorer.is_learnable():
            return self._compute_weights_learned(target_items, user_ids, time_bins, device)
        return self._compute_weights_formula(target_items, user_ids, time_bins, device)

    def _compute_weights_formula(
        self,
        target_items: torch.Tensor,
        user_ids: torch.Tensor,
        time_bins: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        # Vectorised — no gradients
        uid_clamped = user_ids.clamp(max=self.alpha_tensor.size(0) - 1)
        alpha = self.alpha_tensor[uid_clamped]   # [B]
        beta = self.beta_tensor[uid_clamped]     # [B]

        ep = self.pop_tensor[target_items]       # [B]

        et_list = [
            max(self.e_trend.get(item.item(), {}).get(tb.item(), EPSILON), EPSILON)
            for item, tb in zip(target_items, time_bins)
        ]
        et = torch.tensor(et_list, dtype=torch.float32, device=device)   # [B]

        B = target_items.size(0)
        pop_term = ep.pow(alpha) if self.use_alpha else torch.ones(B, dtype=torch.float32, device=device)
        trend_exp = (1.0 - beta) if self.use_beta else torch.zeros(B, dtype=torch.float32, device=device)
        trend_term = et.pow(trend_exp)

        denom = (pop_term * trend_term).clamp(min=EPSILON)
        return (1.0 / denom).clamp(max=self.max_weight)

    def _compute_weights_learned(
        self,
        target_items: torch.Tensor,
        user_ids: torch.Tensor,
        time_bins: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        # MLP forward — gradients flow into MLP
        feat = self.identity_scorer.get_feature_batch(user_ids.tolist(), device)  # [B, 4]
        ab = self.identity_scorer.mlp(feat)   # [B, 2]
        alpha = ab[:, 0]   # [B]
        beta = ab[:, 1]    # [B]

        ep = self.pop_tensor[target_items]   # [B] reuse buffer; no grad needed here

        et_list = [
            max(self.e_trend.get(item.item(), {}).get(tb.item(), EPSILON), EPSILON)
            for item, tb in zip(target_items, time_bins)
        ]
        et = torch.tensor(et_list, dtype=torch.float32, device=device)   # [B]

        B = target_items.size(0)
        pop_term = ep.pow(alpha) if self.use_alpha else torch.ones(B, device=device)
        trend_term = et.pow(1.0 - beta) if self.use_beta else torch.ones(B, device=device)

        denom = (pop_term * trend_term).clamp(min=EPSILON)
        return (1.0 / denom).clamp(max=self.max_weight)

    # ── epoch stats API (called by Trainer) ───────────────────────────────────

    def reset_epoch_stats(self) -> None:
        self._weight_sum = 0.0
        self._weight_sq_sum = 0.0
        self._weight_max = 0.0
        self._weight_n = 0

    def get_epoch_weight_stats(self) -> Dict[str, float]:
        if self._weight_n == 0:
            return {}
        mean = self._weight_sum / self._weight_n
        var = max(self._weight_sq_sum / self._weight_n - mean ** 2, 0.0)
        return {
            "weight_mean": mean,
            "weight_std": var ** 0.5,
            "weight_max": self._weight_max,
        }
