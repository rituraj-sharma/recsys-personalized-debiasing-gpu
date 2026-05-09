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

import torch
import torch.nn as nn

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


class UserIdentityMLP(nn.Module):
    """
    Small MLP that maps user feature vector → (α, β).

    Input features (4):
        [0] entropy          — normalised Shannon entropy (same as formula α)
        [1] avg_popularity   — mean e_pop of items in user history
        [2] profile_size     — normalised number of interactions
        [3] trending_corr    — mean e_trend (same as formula β)

    Architecture:
        Linear(4, 32) → ReLU → Linear(32, 2) → Sigmoid

    Output:
        [α, β] both in (0, 1)
    """
    def __init__(self, hidden_dim: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : [B, 4]  batch of user feature vectors

        Returns
        -------
        [B, 2]  where [:, 0] = α,  [:, 1] = β
        """
        return self.net(x)


class LearnedIdentityScorer(IdentityScorer):
    """
    MLP-based identity scorer jointly trained with the sequential model.

    Key difference from FormulaIdentityScorer:
        - α and β are NOT fixed — they change every training step
          as the MLP weights update via backprop
        - is_learnable() returns True → Trainer adds MLP params to optimizer

    Usage:
        scorer = LearnedIdentityScorer(formula_scorer=existing_formula_scorer)
        scorer.compute(train_seqs, item_genres, e_trend)
        # scorer.mlp  → the nn.Module (pass to optimizer)
    """

    def __init__(
        self,
        hidden_dim: int = 32,
        formula_scorer: "FormulaIdentityScorer | None" = None,
    ) -> None:
        super().__init__()
        self.mlp: "UserIdentityMLP | None" = None
        self.hidden_dim = hidden_dim
        self._features: Dict[int, torch.Tensor] = {}
        # reuse an already-computed formula scorer to avoid recomputing
        self._formula_scorer: "FormulaIdentityScorer | None" = formula_scorer

    def compute(
        self,
        train_seqs: Dict[int, List[Tuple[int, int]]],
        item_genres: Dict[int, List[str]],
        e_trend: Dict[int, Dict[int, float]],
    ) -> "LearnedIdentityScorer":
        """
        Precompute the 4 input features for every user.
        Instantiate the MLP (weights are random at this point —
        they are learned during training via backprop).
        """
        if self._formula_scorer is None:
            self._formula_scorer = FormulaIdentityScorer()
            self._formula_scorer.compute(train_seqs, item_genres, e_trend)

        # Compute e_pop per item for avg_popularity feature
        from src.features.exposure import compute_pop_exposure
        e_pop = compute_pop_exposure(train_seqs)

        # Max interactions for profile_size normalisation
        profile_sizes = [len(seq) for seq in train_seqs.values()]
        max_profile = max(profile_sizes) if profile_sizes else 1

        for user_idx, seq in train_seqs.items():
            alpha_formula, beta_formula = self._formula_scorer.get(user_idx)

            # avg_popularity: mean e_pop of items in user history
            pop_vals = [e_pop.get(item, EPSILON) for item, _ in seq]
            avg_pop = sum(pop_vals) / len(pop_vals) if pop_vals else 0.0

            # profile_size: normalised [0, 1]
            profile_size = len(seq) / max_profile

            # Feature vector: [entropy, avg_pop, profile_size, trending_corr]
            feat = torch.tensor(
                [alpha_formula, avg_pop, profile_size, beta_formula],
                dtype=torch.float32,
            )
            self._features[user_idx] = feat

        # Populate _scores with formula values for interface compatibility
        # (used by PersonalizedIPSLoss.__init__ to init alpha/beta buffers)
        self._scores = dict(self._formula_scorer._scores)

        # Instantiate MLP — weights are random, learned during training
        self.mlp = UserIdentityMLP(hidden_dim=self.hidden_dim)

        return self

    def get_features(self, user_idx: int) -> torch.Tensor:
        """
        Return feature vector for a user.
        Falls back to zero vector for unseen users.
        """
        return self._features.get(
            user_idx, torch.zeros(4, dtype=torch.float32)
        )

    def get_feature_batch(
        self,
        user_ids: List[int],
        device: torch.device,
    ) -> torch.Tensor:
        """
        Return feature matrix for a batch of users.

        Parameters
        ----------
        user_ids : list of int, length B
        device   : torch device

        Returns
        -------
        [B, 4] tensor on the correct device
        """
        feats = torch.stack(
            [self.get_features(u) for u in user_ids], dim=0
        )
        return feats.to(device)

    def get(self, user_idx: int) -> Tuple[float, float]:
        """
        Return current (α, β) from MLP forward pass.
        Used only when MLP is not yet on a device (e.g. during caching).
        Falls back to formula scores.
        """
        # During actual training, _compute_weights() calls get_feature_batch()
        # and does a proper batched forward pass. This method is kept for
        # interface compatibility only.
        if self._formula_scorer is not None:
            return self._formula_scorer.get(user_idx)
        return (0.5, 0.5)

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
