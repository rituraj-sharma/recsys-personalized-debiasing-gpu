"""
MovieLens-1M data loading and PyTorch Dataset.

Key design decisions:
  - min_rating threshold converts explicit ratings to implicit interactions.
  - Multi-genre handled with fractional counting (1/n per genre for n-genre item).
  - 0 is reserved as the padding token; all real IDs are 1-indexed.
  - time_bin = timestamp // 604800  (weekly bins, consistent with exposure.py).
"""
from __future__ import annotations
import os
from typing import Dict, List, Tuple, Optional

import pandas as pd
import torch
from torch.utils.data import Dataset


class ML1MLoader:
    """
    Loads and preprocesses the MovieLens-1M dataset.

    Usage::

        loader = ML1MLoader("data/ml-1m", min_rating=4.0).load()
        sequences = loader.get_user_sequences()   # {user_idx: [(item_idx, time_bin), ...]}
    """

    SECONDS_PER_WEEK: int = 7 * 24 * 3600

    def __init__(
        self,
        data_path: str,
        min_rating: float = 4.0,
        min_interactions: int = 5,
    ) -> None:
        self.data_path = data_path
        self.min_rating = min_rating
        self.min_interactions = min_interactions

        self.ratings_df: Optional[pd.DataFrame] = None
        self.movies_df: Optional[pd.DataFrame] = None

        self.user2idx: Dict[int, int] = {}
        self.idx2user: Dict[int, int] = {}
        self.item2idx: Dict[int, int] = {}
        self.idx2item: Dict[int, int] = {}

        self.item_genres: Dict[int, List[str]] = {}
        self._all_genres: Optional[List[str]] = None

    def load(self) -> "ML1MLoader":
        self._load_raw()
        self._filter()
        self._remap_ids()
        self._parse_genres()
        return self

    @property
    def num_users(self) -> int:
        return len(self.user2idx)

    @property
    def num_items(self) -> int:
        return len(self.item2idx)

    @property
    def all_genres(self) -> List[str]:
        assert self._all_genres is not None, "Call load() first."
        return self._all_genres

    @property
    def num_genres(self) -> int:
        return len(self.all_genres)

    def get_user_sequences(self) -> Dict[int, List[Tuple[int, int]]]:
        assert self.ratings_df is not None, "Call load() first."
        sequences: Dict[int, List[Tuple[int, int]]] = {}
        for user_idx, grp in self.ratings_df.groupby("user_idx"):
            grp = grp.sort_values("timestamp")
            sequences[int(user_idx)] = list(
                zip(grp["item_idx"].tolist(), grp["time_bin"].tolist())
            )
        return sequences

    def _load_raw(self) -> None:
        self.ratings_df = pd.read_csv(
            os.path.join(self.data_path, "ratings.dat"),
            sep="::",
            names=["user_id", "item_id", "rating", "timestamp"],
            engine="python",
        )
        self.movies_df = pd.read_csv(
            os.path.join(self.data_path, "movies.dat"),
            sep="::",
            names=["item_id", "title", "genres"],
            engine="python",
            encoding="latin-1",
        )

    def _filter(self) -> None:
        df = self.ratings_df
        df = df[df["rating"] >= self.min_rating].copy()
        counts = df.groupby("user_id").size()
        valid = counts[counts >= self.min_interactions].index
        df = df[df["user_id"].isin(valid)].copy()
        self.ratings_df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)

    def _remap_ids(self) -> None:
        df = self.ratings_df
        users = sorted(df["user_id"].unique())
        items = sorted(df["item_id"].unique())

        self.user2idx = {u: i + 1 for i, u in enumerate(users)}
        self.idx2user = {i + 1: u for i, u in enumerate(users)}
        self.item2idx = {it: i + 1 for i, it in enumerate(items)}
        self.idx2item = {i + 1: it for i, it in enumerate(items)}

        df["user_idx"] = df["user_id"].map(self.user2idx)
        df["item_idx"] = df["item_id"].map(self.item2idx)
        df["time_bin"] = df["timestamp"] // self.SECONDS_PER_WEEK
        self.ratings_df = df

    def _parse_genres(self) -> None:
        all_genres: set = set()
        for _, row in self.movies_df.iterrows():
            raw_id = row["item_id"]
            if raw_id not in self.item2idx:
                continue
            idx = self.item2idx[raw_id]
            genres = [g.strip() for g in str(row["genres"]).split("|")]
            self.item_genres[idx] = genres
            all_genres.update(genres)
        self._all_genres = sorted(all_genres)


Example = Tuple[int, List[int], int, int]


class SequenceDataset(Dataset):
    """
    PyTorch Dataset wrapping a list of sequential recommendation examples.

    Each example is a 4-tuple::

        (user_idx, input_item_sequence, target_item_idx, target_time_bin)

    The sequence is left-padded with zeros to ``max_seq_len``.
    """

    def __init__(
        self,
        examples: List[Example],
        max_seq_len: int = 50,
    ) -> None:
        self.examples = examples
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        user_idx, item_seq, target_item, target_time_bin = self.examples[idx]

        item_seq = item_seq[-self.max_seq_len:]
        seq_len = len(item_seq)

        pad_len = self.max_seq_len - seq_len
        padded = [0] * pad_len + item_seq

        return {
            "user_idx": torch.tensor(user_idx, dtype=torch.long),
            "input_seq": torch.tensor(padded, dtype=torch.long),
            "seq_len": torch.tensor(seq_len, dtype=torch.long),
            "target_item": torch.tensor(target_item, dtype=torch.long),
            "target_time_bin": torch.tensor(target_time_bin, dtype=torch.long),
        }
