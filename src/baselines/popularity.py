"""Popularity-based recommender — non-personalised baseline."""
from __future__ import annotations
from collections import Counter
from typing import Dict, List, Set, Tuple


class PopularityRecommender:
    def __init__(self, num_items: int) -> None:
        self.num_items = num_items
        self._sorted_items: List[int] = []
        self._item_popularity: Dict[int, int] = {}

    def fit(self, train_seqs: Dict[int, List[Tuple[int, int]]]) -> "PopularityRecommender":
        counter: Counter = Counter()
        for seq in train_seqs.values():
            for item_idx, _ in seq:
                counter[item_idx] += 1
        self._item_popularity = dict(counter)
        self._sorted_items = [item for item, _ in counter.most_common()]
        return self

    def predict_top_k(self, user_idx: int, seen_items: Set[int], k: int) -> List[int]:
        recs: List[int] = []
        for item in self._sorted_items:
            if item not in seen_items:
                recs.append(item)
            if len(recs) == k:
                break
        return recs
