"""
run_sensitivity.py — Sensitivity analysis on trending window W.

Sweeps W ∈ {7, 14, 30, 60} days for GRU4Rec + Personalised Formula IPS.
Each W gets its own subfolder under a timestamped experiment directory.
A sensitivity_summary.csv is written at the end.

Usage
─────
    cd recsys_debiasing_gpu
    python experiments/run_sensitivity.py --config configs/default.yaml

    # Custom W values:
    python experiments/run_sensitivity.py --windows 7 14 30 60
"""
from __future__ import annotations
import argparse
import csv
import json
import os
import sys
from datetime import datetime
from typing import List

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils import ExperimentConfig, RunLogger
from src.data import ML1MLoader, make_datasets, split_user_sequences
from src.features import (
    load_or_compute_exposures,
    FormulaIdentityScorer,
    load_or_compute_identity,
)
from src.models import GRU4Rec
from src.loss import PersonalizedIPSLoss
from src.training import Trainer, build_seen_items
from src.metrics import assign_user_groups


DEFAULT_WINDOWS: List[int] = [7, 14, 30, 60]


def resolve_device(cfg: ExperimentConfig) -> None:
    if cfg.training.device == "cuda" and not torch.cuda.is_available():
        print("[WARNING] CUDA not available — falling back to CPU.")
        cfg.training.device = "cpu"
        cfg.training.use_amp = False
        cfg.training.pin_memory = False


def setup_base_data(cfg: ExperimentConfig, force_recompute: bool = False) -> dict:
    """Load and split data; compute W-independent features (e_pop, α)."""
    print("=" * 60)
    print("Loading ML-1M dataset ...")
    loader = ML1MLoader(
        cfg.data.data_path,
        min_rating=cfg.data.min_rating,
        min_interactions=cfg.data.min_interactions,
    ).load()
    print(f"  Users: {loader.num_users}  Items: {loader.num_items}  "
          f"Genres: {loader.num_genres}")

    user_sequences = loader.get_user_sequences()
    train_seqs, val_data, test_data = split_user_sequences(user_sequences)
    train_ds, val_ds, test_ds, _ = make_datasets(user_sequences, cfg.model.max_seq_len)

    seen_train = build_seen_items(train_seqs)
    seen_all = build_seen_items(train_seqs, include_val=val_data)

    val_gt = {u: tgt for u, (_, tgt, _) in val_data.items()}
    test_gt = {u: tgt for u, (_, tgt, _) in test_data.items()}

    cache_dir = os.path.join(cfg.data.data_path, "cache")
    e_pop, _ = load_or_compute_exposures(
        train_seqs, trend_window_days=cfg.debiasing.trend_window_days, cache_dir=cache_dir,
        force=force_recompute,
    )
    user_groups = assign_user_groups(train_seqs, e_pop)

    return dict(
        loader=loader,
        train_seqs=train_seqs,
        val_data=val_data,
        test_data=test_data,
        train_ds=train_ds,
        val_ds=val_ds,
        test_ds=test_ds,
        seen_train=seen_train,
        seen_all=seen_all,
        val_gt=val_gt,
        test_gt=test_gt,
        e_pop=e_pop,
        user_groups=user_groups,
        force_recompute=force_recompute,
    )


def run_sensitivity_variant(
    cfg: ExperimentConfig,
    base_data: dict,
    experiment_dir: str,
    window_days: int,
) -> None:
    run_cfg = cfg.override(
        **{
            "debiasing.loss_type": "personalized",
            "debiasing.trend_window_days": window_days,
        }
    )

    run_name = f"window_W{window_days}"
    run_dir = os.path.join(experiment_dir, run_name)
    checkpoint_dir = os.path.join(run_dir, "checkpoints")
    logger = RunLogger(run_dir)
    logger.log_config(run_cfg)

    device = cfg.training.device
    loader_obj = base_data["loader"]
    train_seqs = base_data["train_seqs"]
    cache_dir = os.path.join(cfg.data.data_path, "cache")
    pin = cfg.training.pin_memory and "cuda" in device

    print(f"\n{'─'*60}")
    print(f"  Sensitivity run: W = {window_days} days")

    # W-specific e_trend and β(u)
    _, e_trend_w = load_or_compute_exposures(
        train_seqs, trend_window_days=window_days, cache_dir=cache_dir,
        force=base_data.get("force_recompute", False),
    )
    scorer_w = FormulaIdentityScorer()
    load_or_compute_identity(
        scorer_w, train_seqs, loader_obj.item_genres, e_trend_w,
        cache_dir=cache_dir, scorer_tag=f"formula_W{window_days}",
        force=base_data.get("force_recompute", False),
    )

    model = GRU4Rec(
        num_items=loader_obj.num_items,
        embed_dim=cfg.model.embed_dim,
        hidden_dim=cfg.model.hidden_dim,
        num_layers=cfg.model.num_layers,
        dropout=cfg.model.dropout,
    )
    loss_fn = PersonalizedIPSLoss(
        e_pop=base_data["e_pop"],
        e_trend=e_trend_w,
        identity_scorer=scorer_w,
        num_items=loader_obj.num_items,
        max_weight=cfg.debiasing.max_weight,
        use_alpha=True,
        use_beta=True,
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay,
    )

    train_loader = DataLoader(
        base_data["train_ds"], batch_size=cfg.training.batch_size,
        shuffle=True, num_workers=cfg.training.num_workers, pin_memory=pin,
    )
    val_loader = DataLoader(
        base_data["val_ds"], batch_size=cfg.training.batch_size,
        shuffle=False, num_workers=cfg.training.num_workers, pin_memory=pin,
    )
    test_loader = DataLoader(
        base_data["test_ds"], batch_size=cfg.training.batch_size,
        shuffle=False, num_workers=cfg.training.num_workers, pin_memory=pin,
    )

    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device,
        eval_k=cfg.training.eval_k,
        patience=cfg.training.patience,
        checkpoint_dir=checkpoint_dir,
        logger=logger,
        use_amp=cfg.training.use_amp,
    )

    trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=cfg.training.epochs,
        seen_items_train=base_data["seen_train"],
        val_ground_truths=base_data["val_gt"],
    )

    all_metrics = trainer.evaluate_test(
        test_loader=test_loader,
        seen_items_all=base_data["seen_all"],
        test_ground_truths=base_data["test_gt"],
        train_seqs=train_seqs,
        e_pop=base_data["e_pop"],
        user_groups=base_data["user_groups"],
        num_items=loader_obj.num_items,
    )

    print(f"\n  Results for W={window_days}:")
    for k, v in sorted(all_metrics.items()):
        print(f"    {k:30s}: {v:.4f}")


def write_sensitivity_summary(experiment_dir: str, windows: List[int]) -> None:
    rows = []
    for w in windows:
        metrics_path = os.path.join(experiment_dir, f"window_W{w}", "final_metrics.json")
        if not os.path.isfile(metrics_path):
            continue
        with open(metrics_path) as f:
            metrics = json.load(f)
        rows.append({"window_days": w, **metrics})

    if not rows:
        return

    out_path = os.path.join(experiment_dir, "sensitivity_summary.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSummary written to: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--windows", nargs="+", type=int, default=DEFAULT_WINDOWS,
        help="Trend window sizes in days to sweep (default: 7 14 30 60)",
    )
    parser.add_argument("--force-recompute", action="store_true",
                        help="Delete and recompute all feature caches (e_pop, e_trend, alpha/beta)")
    args = parser.parse_args()

    cfg = ExperimentConfig.from_yaml(args.config)
    resolve_device(cfg)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    experiment_dir = os.path.join("results", "run_sensitivity", timestamp)
    os.makedirs(experiment_dir, exist_ok=True)
    print(f"\nResults will be saved to: {experiment_dir}")

    base_data = setup_base_data(cfg, force_recompute=args.force_recompute)

    print(f"\n{'='*60}")
    print(f"Sweeping W ∈ {args.windows} days")
    print("=" * 60)

    for i, W in enumerate(args.windows, start=1):
        print(f"\n[{i}/{len(args.windows)}] W = {W} days")
        run_sensitivity_variant(cfg, base_data, experiment_dir, window_days=W)

    write_sensitivity_summary(experiment_dir, args.windows)
    print("\n" + "=" * 60)
    print("Sensitivity experiment complete.")
    print(f"Key metric to plot: test ndcg@{cfg.training.eval_k} vs window_days")


if __name__ == "__main__":
    main()
