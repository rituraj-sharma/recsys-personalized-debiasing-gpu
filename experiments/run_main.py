"""
run_main.py — Main comparison experiment.

Runs up to four methods; comment/uncomment blocks in main() to select which ones.
Each execution creates a timestamped folder under results/run_main/.
Each method gets its own subfolder with config, training log, checkpoint, and
final metrics. A comparison_summary.csv is written at the end covering whichever
methods actually ran.

Usage
─────
    cd recsys_debiasing_gpu
    python experiments/run_main.py --config configs/default.yaml

    tmux new -s lm
    conda activate lmcf
    CUDA_VISIBLE_DEVICES=0 python train_v2.py

    Detach: Ctrl+B then D — job keeps running
    Re-attach later: tmux attach -t lm
    conda deactivate

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
from src.loss import StandardCELoss, UniformIPSLoss, PersonalizedIPSLoss
from src.baselines import PopularityRecommender
from src.training import Trainer, build_seen_items
from src.metrics import compute_ranking_metrics, assign_user_groups, compute_fairness_metrics


# ─────────────────────────────────────────────────────────────────────────────
# Device helper
# ─────────────────────────────────────────────────────────────────────────────

def resolve_device(cfg: ExperimentConfig) -> str:
    if cfg.training.device == "cuda" and not torch.cuda.is_available():
        print("[WARNING] CUDA not available — falling back to CPU. "
              "Set use_amp=false in config if issues arise.")
        cfg.training.device = "cpu"
        cfg.training.use_amp = False
        cfg.training.pin_memory = False
    return cfg.training.device


# ─────────────────────────────────────────────────────────────────────────────
# Shared data setup
# ─────────────────────────────────────────────────────────────────────────────

def setup_data(cfg: ExperimentConfig, force_recompute: bool = False) -> dict:
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
        force=force_recompute,
    )

    scorer = FormulaIdentityScorer()
    load_or_compute_identity(
        scorer, train_seqs, loader.item_genres, e_trend,
        cache_dir=cache_dir, scorer_tag="formula", force=force_recompute,
    )

    from src.features.user_identity import LearnedIdentityScorer
    learned_scorer = LearnedIdentityScorer(formula_scorer=scorer)
    learned_scorer.compute(train_seqs, loader.item_genres, e_trend)

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
        learned_scorer=learned_scorer,
        user_groups=user_groups,
    )



# ─────────────────────────────────────────────────────────────────────────────
# Per-method runners
# ─────────────────────────────────────────────────────────────────────────────

def run_popularity(cfg: ExperimentConfig, data: dict, experiment_dir: str) -> None:
    run_dir = os.path.join(experiment_dir, "popularity")
    logger = RunLogger(run_dir)
    logger.log_config(cfg)

    rec = PopularityRecommender(data["loader"].num_items)
    rec.fit(data["train_seqs"])

    recommendations = {
        user_idx: rec.predict_top_k(user_idx, data["seen_all"].get(user_idx, set()), cfg.training.eval_k)
        for user_idx, (_, _, _) in data["test_data"].items()
    }

    acc = compute_ranking_metrics(recommendations, data["test_gt"], cfg.training.eval_k)
    fairness = compute_fairness_metrics(
        recommendations, data["train_seqs"], data["e_pop"],
        data["user_groups"], data["loader"].num_items,
    )
    all_metrics = {**acc, **fairness}
    logger.log_final_metrics(all_metrics)
    _print_results("popularity", all_metrics)


def run_gru4rec(
    cfg: ExperimentConfig,
    data: dict,
    experiment_dir: str,
    loss_type: str,
    method_name: str,
    use_alpha: bool = True,
    use_beta: bool = True,
) -> None:
    run_cfg = cfg.override(
        **{
            "debiasing.loss_type": loss_type,
            "debiasing.use_alpha": use_alpha,
            "debiasing.use_beta": use_beta,
        }
    )

    run_dir = os.path.join(experiment_dir, method_name)
    checkpoint_dir = os.path.join(run_dir, "checkpoints")
    logger = RunLogger(run_dir)
    logger.log_config(run_cfg)

    device = cfg.training.device
    loader_obj = data["loader"]
    pin = cfg.training.pin_memory and "cuda" in device

    model = GRU4Rec(
        num_items=loader_obj.num_items,
        embed_dim=cfg.model.embed_dim,
        hidden_dim=cfg.model.hidden_dim,
        num_layers=cfg.model.num_layers,
        dropout=cfg.model.dropout,
    )

    if loss_type == "ce":
        loss_fn = StandardCELoss(loader_obj.num_items)
    elif loss_type == "ips":
        loss_fn = UniformIPSLoss(
            e_pop=data["e_pop"],
            num_items=loader_obj.num_items,
            max_weight=cfg.debiasing.max_weight,
        )
    elif loss_type == "personalized":
        loss_fn = PersonalizedIPSLoss(
            e_pop=data["e_pop"],
            e_trend=data["e_trend"],
            identity_scorer=data["scorer"],
            num_items=loader_obj.num_items,
            max_weight=cfg.debiasing.max_weight,
            use_alpha=use_alpha,
            use_beta=use_beta,
        )
    elif loss_type == "personalized_learned":
        loss_fn = PersonalizedIPSLoss(
            e_pop=data["e_pop"],
            e_trend=data["e_trend"],
            identity_scorer=data["learned_scorer"],   # <-- note: learned_scorer
            num_items=loader_obj.num_items,
            max_weight=cfg.debiasing.max_weight,
            use_alpha=use_alpha,
            use_beta=use_beta,
        )
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")

    # optimizer = torch.optim.Adam(
    #     model.parameters(),
    #     lr=cfg.training.lr,
    #     weight_decay=cfg.training.weight_decay,
    # )

    if loss_type == "personalized_learned":
        optimizer = torch.optim.Adam(
            [
                {"params": model.parameters()},
                {"params": loss_fn.identity_scorer.mlp.parameters()},
            ],
            lr=cfg.training.lr,
            weight_decay=cfg.training.weight_decay,
        )
    else:
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=cfg.training.lr,
            weight_decay=cfg.training.weight_decay,
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
    _print_results(method_name, all_metrics)


# ─────────────────────────────────────────────────────────────────────────────
# Summary writer
# ─────────────────────────────────────────────────────────────────────────────

def write_comparison_summary(experiment_dir: str) -> None:
    rows = []
    for entry in sorted(os.listdir(experiment_dir)):
        metrics_path = os.path.join(experiment_dir, entry, "final_metrics.json")
        if not os.path.isfile(metrics_path):
            continue
        with open(metrics_path) as f:
            metrics = json.load(f)
        rows.append({"method": entry, **metrics})

    if not rows:
        return

    fieldnames = list(rows[0].keys())
    out_path = os.path.join(experiment_dir, "comparison_summary.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSummary written to: {out_path}")


def _print_results(name: str, metrics: dict) -> None:
    print(f"\n{'-'*50}")
    print(f"  Run: {name}")
    for k, v in sorted(metrics.items()):
        print(f"    {k:30s}: {v:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--force-recompute", action="store_true",
                        help="Delete and recompute all feature caches (e_pop, e_trend, alpha/beta)")
    args = parser.parse_args()

    cfg = ExperimentConfig.from_yaml(args.config)
    resolve_device(cfg)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    experiment_dir = os.path.join("results", "run_main", timestamp)
    os.makedirs(experiment_dir, exist_ok=True)
    print(f"\nResults will be saved to: {experiment_dir}")

    data = setup_data(cfg, force_recompute=args.force_recompute)

    # ── Run 1: Popularity baseline ────────────────────────────────────────────
    print("\n[1/4] Running: popularity baseline")
    run_popularity(cfg, data, experiment_dir)

    # ── Run 2: Plain GRU4Rec (CE loss) ───────────────────────────────────────
    print("\n[2/4] Running: gru4rec (CE loss)")
    run_gru4rec(cfg, data, experiment_dir, loss_type="ce", method_name="gru4rec_ce")

    # ── Run 3: GRU4Rec + Uniform IPS ─────────────────────────────────────────
    print("\n[3/4] Running: gru4rec + uniform IPS")
    run_gru4rec(cfg, data, experiment_dir, loss_type="ips", method_name="gru4rec_ips")

    # ── Run 4: GRU4Rec + Personalised IPS (formula) ──────────────────────────
    print("\n[4/4] Running: gru4rec + personalised IPS (formula)")
    run_gru4rec(cfg, data, experiment_dir, loss_type="personalized", method_name="gru4rec_personalized")

    # ── Run 5: GRU4Rec + Personalised IPS (learned MLP) ──────────────────────
    print("\n[5/5] Running: gru4rec + personalised IPS (learned MLP)")
    run_gru4rec(cfg, data, experiment_dir, loss_type="personalized_learned", run_name="gru4rec_pers_learned")

    write_comparison_summary(experiment_dir)


if __name__ == "__main__":
    main()
