"""
Fairness and diversity metrics following Abdollahpouri et al. (RecSys 2019).

Metrics:
  ΔGAP per group  — change in Group Average Popularity
  Avg Popularity  — mean popularity of recommended items
  Coverage        — fraction of catalogue recommended
  Long-Tail Ratio — fraction of recommendations from bottom-80% items
"""
from __future__ import annotations
from collections import defaultdict
from typing import Dict, List, Set, Tuple

import numpy as np


def assign_user_groups(
    train_seqs: Dict[int, List[Tuple[int, int]]],
    e_pop: Dict[int, float],
    popular_threshold: float = 0.8,
) -> Dict[int, str]:
    sorted_items = sorted(e_pop.items(), key=lambda x: x[1], reverse=True)
    popular_items: Set[int] = set()
    cumulative = 0.0
    for item_idx, ep in sorted_items:
        popular_items.add(item_idx)
        cumulative += ep
        if cumulative >= popular_threshold:
            break

    ratios: Dict[int, float] = {}
    for user_idx, seq in train_seqs.items():
        if not seq:
            ratios[user_idx] = 0.0
            continue
        pop_count = sum(1 for item, _ in seq if item in popular_items)
        ratios[user_idx] = pop_count / len(seq)

    all_ratios = sorted(ratios.values())
    p20 = np.percentile(all_ratios, 20)
    p80 = np.percentile(all_ratios, 80)

    groups: Dict[int, str] = {}
    for user_idx, ratio in ratios.items():
        if ratio <= p20:
            groups[user_idx] = "niche"
        elif ratio >= p80:
            groups[user_idx] = "blockbuster"
        else:
            groups[user_idx] = "diverse"

    return groups


def _avg_pop_of_list(items: List[int], e_pop: Dict[int, float]) -> float:
    if not items:
        return 0.0
    return sum(e_pop.get(i, 0.0) for i in items) / len(items)


def compute_delta_gap(
    recommendations: Dict[int, List[int]],
    train_seqs: Dict[int, List[Tuple[int, int]]],
    e_pop: Dict[int, float],
    user_groups: Dict[int, str],
) -> Dict[str, float]:
    group_profile_pops: Dict[str, List[float]] = defaultdict(list)
    group_rec_pops: Dict[str, List[float]] = defaultdict(list)

    for user_idx in recommendations:
        group = user_groups.get(user_idx, "diverse")
        seq = train_seqs.get(user_idx, [])
        profile_items = [item for item, _ in seq]

        group_profile_pops[group].append(_avg_pop_of_list(profile_items, e_pop))
        group_rec_pops[group].append(_avg_pop_of_list(recommendations[user_idx], e_pop))

    result: Dict[str, float] = {}
    for group in ("niche", "diverse", "blockbuster"):
        profile_avg = (
            sum(group_profile_pops[group]) / len(group_profile_pops[group])
            if group_profile_pops[group] else 0.0
        )
        rec_avg = (
            sum(group_rec_pops[group]) / len(group_rec_pops[group])
            if group_rec_pops[group] else 0.0
        )
        result[f"delta_gap_{group}"] = rec_avg - profile_avg

    return result


def compute_avg_popularity(
    recommendations: Dict[int, List[int]],
    e_pop: Dict[int, float],
) -> float:
    all_pops = [
        e_pop.get(item, 0.0)
        for recs in recommendations.values()
        for item in recs
    ]
    return sum(all_pops) / len(all_pops) if all_pops else 0.0


def compute_coverage(
    recommendations: Dict[int, List[int]],
    num_items: int,
) -> float:
    unique_recommended = {item for recs in recommendations.values() for item in recs}
    return len(unique_recommended) / num_items if num_items > 0 else 0.0


def compute_long_tail_ratio(
    recommendations: Dict[int, List[int]],
    e_pop: Dict[int, float],
    long_tail_threshold: float = 0.8,
) -> float:
    sorted_items = sorted(e_pop.items(), key=lambda x: x[1], reverse=True)
    popular_items: Set[int] = set()
    cumulative = 0.0
    for item_idx, ep in sorted_items:
        if cumulative >= long_tail_threshold:
            break
        popular_items.add(item_idx)
        cumulative += ep

    total = 0
    long_tail_count = 0
    for recs in recommendations.values():
        for item in recs:
            total += 1
            if item not in popular_items:
                long_tail_count += 1

    return long_tail_count / total if total > 0 else 0.0


def compute_fairness_metrics(
    recommendations: Dict[int, List[int]],
    train_seqs: Dict[int, List[Tuple[int, int]]],
    e_pop: Dict[int, float],
    user_groups: Dict[int, str],
    num_items: int,
) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    metrics.update(compute_delta_gap(recommendations, train_seqs, e_pop, user_groups))
    metrics["avg_popularity"] = compute_avg_popularity(recommendations, e_pop)
    metrics["coverage"] = compute_coverage(recommendations, num_items)
    metrics["long_tail_ratio"] = compute_long_tail_ratio(recommendations, e_pop)
    return metrics
