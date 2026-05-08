"""
GRU4Rec — recurrent sequential recommendation model.

Architecture:
  item_embedding : Embedding(num_items+1, embed_dim, padding_idx=0)
  gru            : GRU(embed_dim, hidden_dim, num_layers, batch_first=True)
  dropout        : Dropout(dropout)
  output_layer   : Linear(hidden_dim, num_items+1)
"""
from __future__ import annotations
from typing import Dict

import torch
import torch.nn as nn

from src.models.base import SequentialModel


class GRU4Rec(SequentialModel):
    def __init__(
        self,
        num_items: int,
        embed_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.2,
    ) -> None:
        super().__init__(num_items)

        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.item_embedding = nn.Embedding(num_items + 1, embed_dim, padding_idx=0)
        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(hidden_dim, num_items + 1)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.item_embedding.weight[1:])
        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        input_seq = batch["input_seq"]   # [B, L]
        seq_len = batch["seq_len"]       # [B]

        embedded = self.item_embedding(input_seq)      # [B, L, E]
        gru_output, _ = self.gru(embedded)             # [B, L, H]

        last_idx = (seq_len - 1).clamp(min=0)
        idx = last_idx.view(-1, 1, 1).expand(-1, 1, self.hidden_dim)
        last_hidden = gru_output.gather(1, idx).squeeze(1)  # [B, H]

        last_hidden = self.dropout(last_hidden)
        logits = self.output_layer(last_hidden)        # [B, num_items+1]

        return logits
