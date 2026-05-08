"""Accuracy metrics for top-K recommendation: Recall@K and NDCG@K."""
from __future__ import annotations
import math
from typing import Dict, List


def recall_at_k(recommended: List[int], ground_truth: int) -> float:
    return 1.0 if ground_truth in recommended else 0.0


def ndcg_at_k(recommended: List[int], ground_truth: int) -> float:
    if ground_truth not in recommended:
        return 0.0
    rank = recommended.index(ground_truth) + 1
    return 1.0 / math.log2(rank + 1)


def compute_ranking_metrics(
    recommendations: Dict[int, List[int]],
    ground_truths: Dict[int, int],
    k: int = 10,
) -> Dict[str, float]:
    recalls, ndcgs = [], []
    for user_idx, target in ground_truths.items():
        recs = recommendations.get(user_idx, [])
        recalls.append(recall_at_k(recs, target))
        ndcgs.append(ndcg_at_k(recs, target))

    return {
        f"recall@{k}": sum(recalls) / len(recalls) if recalls else 0.0,
        f"ndcg@{k}": sum(ndcgs) / len(ndcgs) if ndcgs else 0.0,
    }
