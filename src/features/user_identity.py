"""
User Identity Scoring — computes α(u) and β(u) for every user.

FormulaIdentityScorer  ← fully implemented
    α(u) = H(u) / log(K)   normalized Shannon entropy over genre distribution
    β(u) = mean e_trend(i, t) over user's training history

LearnedIdentityScorer  ← PLACEHOLDER (not yet implemented)
"""
from __future__ import annotations
import math
import os
import pickle
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Dict, List, Tuple

from src.features.exposure import get_trend_exposure, EPSILON


class IdentityScorer(ABC):
    def __init__(self) -> None:
        self._scores: Dict[int, Tuple[float, float]] = {}

    @abstractmethod
    def compute(
        self,
        train_seqs: Dict[int, List[Tuple[int, int]]],
        item_genres: Dict[int, List[str]],
        e_trend: Dict[int, Dict[int, float]],
    ) -> "IdentityScorer":
        pass

    @property
    def scores(self) -> Dict[int, Tuple[float, float]]:
        return self._scores

    def get(self, user_idx: int) -> Tuple[float, float]:
        return self._scores.get(user_idx, (0.5, 0.5))

    def is_learnable(self) -> bool:
        return False


class FormulaIdentityScorer(IdentityScorer):
    """
    Computes α and β analytically from training data.

    α(u) — normalized Shannon entropy over fractional genre distribution
    β(u) — mean trending exposure over user's training history
    """

    def compute(
        self,
        train_seqs: Dict[int, List[Tuple[int, int]]],
        item_genres: Dict[int, List[str]],
        e_trend: Dict[int, Dict[int, float]],
    ) -> "FormulaIdentityScorer":
        all_genres: set = set()
        for genres in item_genres.values():
            all_genres.update(genres)
        K = len(all_genres)
        log_K = math.log(K) if K > 1 else 1.0

        for user_idx, seq in train_seqs.items():
            alpha = self._compute_alpha(seq, item_genres, log_K)
            beta = self._compute_beta(seq, e_trend)
            self._scores[user_idx] = (alpha, beta)

        return self

    @staticmethod
    def _compute_alpha(
        seq: List[Tuple[int, int]],
        item_genres: Dict[int, List[str]],
        log_K: float,
    ) -> float:
        genre_weight: Dict[str, float] = defaultdict(float)

        for item_idx, _ in seq:
            genres = item_genres.get(item_idx, [])
            if not genres:
                continue
            w = 1.0 / len(genres)
            for g in genres:
                genre_weight[g] += w

        total = sum(genre_weight.values())
        if total == 0:
            return 0.0

        entropy = 0.0
        for w in genre_weight.values():
            p = w / total
            if p > 0:
                entropy -= p * math.log(p)

        return min(entropy / log_K, 1.0)

    @staticmethod
    def _compute_beta(
        seq: List[Tuple[int, int]],
        e_trend: Dict[int, Dict[int, float]],
    ) -> float:
        if not seq:
            return 0.0
        total = sum(
            get_trend_exposure(e_trend, item_idx, time_bin)
            for item_idx, time_bin in seq
        )
        return total / len(seq)


class LearnedIdentityScorer(IdentityScorer):
    """PLACEHOLDER — not yet implemented. See original codebase for full spec."""

    def __init__(self) -> None:
        raise NotImplementedError(
            "LearnedIdentityScorer is not yet implemented."
        )

    def compute(self, *args, **kwargs) -> "LearnedIdentityScorer":
        raise NotImplementedError

    def is_learnable(self) -> bool:
        return True


def load_or_compute_identity(
    scorer: IdentityScorer,
    train_seqs: Dict[int, List[Tuple[int, int]]],
    item_genres: Dict[int, List[str]],
    e_trend: Dict[int, Dict[int, float]],
    cache_dir: str = "data/cache",
    scorer_tag: str = "formula",
    force: bool = False,
) -> IdentityScorer:
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"identity_{scorer_tag}.pkl")

    if force and os.path.exists(cache_path):
        os.remove(cache_path)
        print(f"[identity] Removed cache: {cache_path}")

    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            scores = pickle.load(f)
        scorer._scores = scores
        print(f"[identity] Loaded {scorer_tag} scores from {cache_path}")
    else:
        print(f"[identity] Computing {scorer_tag} identity scores ...")
        scorer.compute(train_seqs, item_genres, e_trend)
        with open(cache_path, "wb") as f:
            pickle.dump(scorer._scores, f)
        print(f"[identity] Saved to {cache_path}")

    return scorer
