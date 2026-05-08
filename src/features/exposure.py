"""
Exposure precomputation — computed once from training data only.

Two exposure signals:
  e_pop(i)            — global normalized interaction frequency
  e_trend(i, time_bin)— local normalized frequency in a sliding window of W days

Both cached to disk as pickle so they are computed only once per dataset /
window-size combination.
"""
from __future__ import annotations
import os
import pickle
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np


EPSILON: float = 1e-8


def compute_pop_exposure(
    train_seqs: Dict[int, List[Tuple[int, int]]]
) -> Dict[int, float]:
    freq: Dict[int, int] = defaultdict(int)
    for seq in train_seqs.values():
        for item_idx, _ in seq:
            freq[item_idx] += 1

    total = sum(freq.values())
    if total == 0:
        return {}
    return {item: count / total for item, count in freq.items()}


def compute_trend_exposure(
    train_seqs: Dict[int, List[Tuple[int, int]]],
    trend_window_days: int = 30,
) -> Dict[int, Dict[int, float]]:
    w_bins: int = max(1, trend_window_days // 7)

    raw: Dict[int, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
    for seq in train_seqs.values():
        for item_idx, time_bin in seq:
            raw[item_idx][time_bin] += 1

    all_bins = sorted({tb for seq in train_seqs.values() for _, tb in seq})

    e_trend: Dict[int, Dict[int, float]] = defaultdict(dict)

    for bin_t in all_bins:
        window_start = bin_t - w_bins
        windowed: Dict[int, int] = defaultdict(int)
        for item_idx, bin_counts in raw.items():
            for b, cnt in bin_counts.items():
                if window_start <= b <= bin_t:
                    windowed[item_idx] += cnt

        total = sum(windowed.values())
        if total == 0:
            continue

        for item_idx, cnt in windowed.items():
            e_trend[item_idx][bin_t] = cnt / total

    return dict(e_trend)


def get_trend_exposure(
    e_trend: Dict[int, Dict[int, float]],
    item_idx: int,
    time_bin: int,
) -> float:
    return e_trend.get(item_idx, {}).get(time_bin, EPSILON)


def load_or_compute_exposures(
    train_seqs: Dict[int, List[Tuple[int, int]]],
    trend_window_days: int,
    cache_dir: str = "data/cache",
) -> Tuple[Dict[int, float], Dict[int, Dict[int, float]]]:
    os.makedirs(cache_dir, exist_ok=True)
    pop_path = os.path.join(cache_dir, "e_pop.pkl")
    trend_path = os.path.join(cache_dir, f"e_trend_W{trend_window_days}.pkl")

    if os.path.exists(pop_path):
        with open(pop_path, "rb") as f:
            e_pop = pickle.load(f)
        print(f"[exposure] Loaded e_pop from {pop_path}")
    else:
        print("[exposure] Computing e_pop ...")
        e_pop = compute_pop_exposure(train_seqs)
        with open(pop_path, "wb") as f:
            pickle.dump(e_pop, f)
        print(f"[exposure] Saved e_pop to {pop_path}")

    if os.path.exists(trend_path):
        with open(trend_path, "rb") as f:
            e_trend = pickle.load(f)
        print(f"[exposure] Loaded e_trend (W={trend_window_days}) from {trend_path}")
    else:
        print(f"[exposure] Computing e_trend (W={trend_window_days} days) ...")
        e_trend = compute_trend_exposure(train_seqs, trend_window_days)
        with open(trend_path, "wb") as f:
            pickle.dump(e_trend, f)
        print(f"[exposure] Saved e_trend to {trend_path}")

    return e_pop, e_trend
