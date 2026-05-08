"""
run_ablation.py — Ablation experiment.

Tests the contribution of α (popularity correction) and β (trending correction)
in PersonalizedIPSLoss. Comment/uncomment entries in ABLATION_RUNS to select
which variants to run.

Each execution creates a timestamped folder under results/run_ablation/.
Each variant gets its own subfolder. An ablation_summary.csv is written at end.

Usage
─────
    cd recsys_debiasing_gpu
    python experiments/run_ablation.py --config configs/default.yaml
"""
from __future__ import annotations
import argparse
import csv
import json
import os
import sys
from datetime import datetime

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


# Each tuple: (run_name, use_alpha, use_beta, description)
ABLATION_RUNS = [
    ("alpha_only", True,  False, "Popularity correction only (β disabled)"),
    ("beta_only",  False, True,  "Trending correction only (α disabled)"),
    # ("both",       True,  True,  "Full method — both α and β active"),
]


def resolve_device(cfg: ExperimentConfig) -> None:
    if cfg.training.device == "cuda" and not torch.cuda.is_available():
        print("[WARNING] CUDA not available — falling back to CPU.")
        cfg.training.device = "cpu"
        cfg.training.use_amp = False
        cfg.training.pin_memory = False


def setup_data(cfg: ExperimentConfig) -> dict:
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
    e_pop, e_trend = load_or_compute_exposures(
        train_seqs, cfg.debiasing.trend_window_days, cache_dir=cache_dir,
    )
    scorer = FormulaIdentityScorer()
    load_or_compute_identity(
        scorer, train_seqs, loader.item_genres, e_trend,
        cache_dir=cache_dir, scorer_tag="formula",
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
        e_trend=e_trend,
        scorer=scorer,
        user_groups=user_groups,
    )


def run_ablation_variant(
    cfg: ExperimentConfig,
    data: dict,
    experiment_dir: str,
    run_name: str,
    use_alpha: bool,
    use_beta: bool,
    description: str,
) -> None:
    run_cfg = cfg.override(
        **{
            "debiasing.loss_type": "personalized",
            "debiasing.use_alpha": use_alpha,
            "debiasing.use_beta": use_beta,
        }
    )

    run_dir = os.path.join(experiment_dir, run_name)
    checkpoint_dir = os.path.join(run_dir, "checkpoints")
    logger = RunLogger(run_dir)
    logger.log_config(run_cfg)

    device = cfg.training.device
    loader_obj = data["loader"]
    pin = cfg.training.pin_memory and "cuda" in device

    print(f"\n{'─'*60}")
    print(f"  Ablation: {run_name}  |  {description}")
    print(f"  use_alpha={use_alpha}  use_beta={use_beta}")

    model = GRU4Rec(
        num_items=loader_obj.num_items,
        embed_dim=cfg.model.embed_dim,
        hidden_dim=cfg.model.hidden_dim,
        num_layers=cfg.model.num_layers,
        dropout=cfg.model.dropout,
    )
    loss_fn = PersonalizedIPSLoss(
        e_pop=data["e_pop"],
        e_trend=data["e_trend"],
        identity_scorer=data["scorer"],
        num_items=loader_obj.num_items,
        max_weight=cfg.debiasing.max_weight,
        use_alpha=use_alpha,
        use_beta=use_beta,
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay,
    )

    train_loader = DataLoader(
        data["train_ds"], batch_size=cfg.training.batch_size,
        shuffle=True, num_workers=cfg.training.num_workers, pin_memory=pin,
    )
    val_loader = DataLoader(
        data["val_ds"], batch_size=cfg.training.batch_size,
        shuffle=False, num_workers=cfg.training.num_workers, pin_memory=pin,
    )
    test_loader = DataLoader(
        data["test_ds"], batch_size=cfg.training.batch_size,
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
        seen_items_train=data["seen_train"],
        val_ground_truths=data["val_gt"],
    )

    all_metrics = trainer.evaluate_test(
        test_loader=test_loader,
        seen_items_all=data["seen_all"],
        test_ground_truths=data["test_gt"],
        train_seqs=data["train_seqs"],
        e_pop=data["e_pop"],
        user_groups=data["user_groups"],
        num_items=loader_obj.num_items,
    )

    print(f"\n  Results for '{run_name}':")
    for k, v in sorted(all_metrics.items()):
        print(f"    {k:30s}: {v:.4f}")


def write_ablation_summary(experiment_dir: str) -> None:
    rows = []
    for entry in sorted(os.listdir(experiment_dir)):
        metrics_path = os.path.join(experiment_dir, entry, "final_metrics.json")
        if not os.path.isfile(metrics_path):
            continue
        with open(metrics_path) as f:
            metrics = json.load(f)
        rows.append({"variant": entry, **metrics})

    if not rows:
        return

    out_path = os.path.join(experiment_dir, "ablation_summary.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSummary written to: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    cfg = ExperimentConfig.from_yaml(args.config)
    resolve_device(cfg)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    experiment_dir = os.path.join("results", "run_ablation", timestamp)
    os.makedirs(experiment_dir, exist_ok=True)
    print(f"\nResults will be saved to: {experiment_dir}")

    data = setup_data(cfg)

    active = [r for r in ABLATION_RUNS]
    print(f"\n{'='*60}")
    print(f"Running {len(active)} ablation variants")
    print("=" * 60)

    for i, (run_name, use_alpha, use_beta, description) in enumerate(active, start=1):
        print(f"\n[{i}/{len(active)}] {run_name}")
        run_ablation_variant(
            cfg=cfg,
            data=data,
            experiment_dir=experiment_dir,
            run_name=run_name,
            use_alpha=use_alpha,
            use_beta=use_beta,
            description=description,
        )

    write_ablation_summary(experiment_dir)
    print("\n" + "=" * 60)
    print("Ablation experiment complete.")


if __name__ == "__main__":
    main()
