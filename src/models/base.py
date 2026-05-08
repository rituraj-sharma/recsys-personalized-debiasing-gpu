"""
Abstract base class for sequential recommendation models.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict

import torch
import torch.nn as nn


class SequentialModel(ABC, nn.Module):
    def __init__(self, num_items: int) -> None:
        super().__init__()
        self.num_items = num_items

    @abstractmethod
    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        pass

    def predict_top_k(
        self,
        batch: Dict[str, torch.Tensor],
        k: int,
        seen_items: Dict[int, set],
    ) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            logits = self.forward(batch)

        logits[:, 0] = float("-inf")

        user_indices = batch["user_idx"].tolist()
        for row, user_idx in enumerate(user_indices):
            for seen_item in seen_items.get(user_idx, set()):
                if seen_item < logits.size(1):
                    logits[row, seen_item] = float("-inf")

        _, top_k = torch.topk(logits, k, dim=-1)
        return top_k
