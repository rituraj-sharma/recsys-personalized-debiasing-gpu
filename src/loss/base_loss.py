"""Standard cross-entropy loss — no debiasing."""
from __future__ import annotations
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class StandardCELoss(nn.Module):
    def __init__(self, num_items: int) -> None:
        super().__init__()
        self.num_items = num_items

    def forward(
        self,
        logits: torch.Tensor,
        batch: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        target = batch["target_item"]
        logits = logits.clone()
        logits[:, 0] = float("-inf")
        return F.cross_entropy(logits, target)
