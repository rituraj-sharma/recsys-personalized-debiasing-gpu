"""
run_hptuning.py — Automated hyperparameter tuning with Optuna.

Tunes GRU4Rec + personalised IPS (formula). Objective: val_ndcg@10.
Each execution creates a timestamped study folder under results/hptuning/.
Each Optuna trial gets its own subfolder with config, training log, and checkpoint.
At the end writes: study_summary.csv, best_config.yaml.

Usage
─────
    cd recsys_debiasing_gpu
    pip install optuna

    # Default: 30 trials
    python experiments/run_hptuning.py --config configs/default.yaml

    # Fast mode — 20% users, 15 trials, 5 epochs
    python experiments/run_hptuning.py --fast --n_trials 15 --trial_epochs 5
"""
from __future__ import annotations
import argparse
import csv
import json
import logging
import os
import sys
from datetime import datetime
from typing import Dict

import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logging.getLogger("optuna").setLevel(logging.WARNING)

try:
    import optuna
    from optuna.samplers import TPESampler
except ImportError:
    print("Optuna not installed. Run: pip install optuna")
    sys.exit(1)

from src.utils import ExperimentConfig, RunLogger
from src.data import (
    ML1MLoader, make_datasets, split_user_sequences,
    SequenceDataset, build_train_examples, build_eval_examples,
)
from src.features import (
    load_or_compute_exposures,
    FormulaIdentityScorer,
    load_or_compute_identity,
)
from src.models import GRU4Rec
from src.loss import PersonalizedIPSLoss
from src.training import Trainer, build_seen_items
from src.metrics import compute_ranking_metrics, assign_user_groups


# ─────────────────────────────────────────────────────────────────────────────
# Device helper
# ─────────────────────────────────────────────────────────────────────────────

def resolve_device(cfg: ExperimentConfig) -> None:
    if cfg.training.device == "cuda" and not torch.cuda.is_available():
        print("[WARNING] CUDA not available — falling back to CPU.")
        cfg.training.device = "cpu"
        cfg.training.use_amp = False
        cfg.training.pin_memory = False


# ─────────────────────────────────────────────────────────────────────────────
# Search space
# ─────────────────────────────────────────────────────────────────────────────

SEARCH_SPACE = {
    "embed_dim":    [64, 128, 256],
    "hidden_dim":   [128, 256, 512],
    "num_layers":   [1, 2],
    "dropout":      [0.1, 0.2, 0.3, 0.5],
    "lr":           [1e-4, 1e-3, 1e-2],
    "weight_decay": [1e-5, 1e-4, 1e-3],
    "batch_size":   [128, 256, 512],
    "max_weight":   [5.0, 10.0, 20.0],
}


def sample_hparams(trial: optuna.Trial) -> Dict:
    return {
        "embed_dim":    trial.suggest_categorical("embed_dim",    SEARCH_SPACE["embed_dim"]),
        "hidden_dim":   trial.suggest_categorical("hidden_dim",   SEARCH_SPACE["hidden_dim"]),
        "num_layers":   trial.suggest_categorical("num_layers",   SEARCH_SPACE["num_layers"]),
        "dropout":      trial.suggest_categorical("dropout",      SEARCH_SPACE["dropout"]),
        "lr":           trial.suggest_categorical("lr",           SEARCH_SPACE["lr"]),
        "weight_decay": trial.suggest_categorical("weight_decay", SEARCH_SPACE["weight_decay"]),
        "batch_size":   trial.suggest_categorical("batch_size",   SEARCH_SPACE["batch_size"]),
        "max_weight":   trial.suggest_categorical("max_weight",   SEARCH_SPACE["max_weight"]),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Single trial
# ─────────────────────────────────────────────────────────────────────────────

def run_trial(
    trial: optuna.Trial,
    base_cfg: ExperimentConfig,
    data: dict,
    trial_epochs: int,
    study_dir: str,
) -> float:
    hparams = sample_hparams(trial)
    trial_dir = os.path.join(study_dir, f"trial_{trial.number:03d}")
    checkpoint_dir = os.path.join(trial_dir, "checkpoints")

    cfg = base_cfg.override(
        **{
            "model.embed_dim":       hparams["embed_dim"],
            "model.hidden_dim":      hparams["hidden_dim"],
            "model.num_layers":      hparams["num_layers"],
            "model.dropout":         hparams["dropout"],
            "training.lr":           hparams["lr"],
            "training.weight_decay": hparams["weight_decay"],
            "training.batch_size":   hparams["batch_size"],
            "training.epochs":       trial_epochs,
            "debiasing.max_weight":  hparams["max_weight"],
        }
    )

    logger = RunLogger(trial_dir)
    logger.log_config(cfg)

    device = base_cfg.training.device
    loader_obj = data["loader"]
    pin = base_cfg.training.pin_memory and "cuda" in device

    model = GRU4Rec(
        num_items=loader_obj.num_items,
        embed_dim=hparams["embed_dim"],
        hidden_dim=hparams["hidden_dim"],
        num_layers=hparams["num_layers"],
        dropout=hparams["dropout"],
    )
    loss_fn = PersonalizedIPSLoss(
        e_pop=data["e_pop"],
        e_trend=data["e_trend"],
        identity_scorer=data["scorer"],
        num_items=loader_obj.num_items,
        max_weight=hparams["max_weight"],
        use_alpha=True,
        use_beta=True,
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=hparams["lr"],
        weight_decay=hparams["weight_decay"],
    )

    train_loader = DataLoader(
        data["train_ds"], batch_size=hparams["batch_size"],
        shuffle=True, num_workers=base_cfg.training.num_workers, pin_memory=pin,
    )
    val_loader = DataLoader(
        data["val_ds"], batch_size=hparams["batch_size"],
        shuffle=False, num_workers=base_cfg.training.num_workers, pin_memory=pin,
    )

    tuning_patience = min(base_cfg.training.patience, 2)

    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device,
        eval_k=base_cfg.training.eval_k,
        patience=tuning_patience,
        checkpoint_dir=checkpoint_dir,
        logger=logger,
        use_amp=base_cfg.training.use_amp,
    )

    trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=trial_epochs,
        seen_items_train=data["seen_train"],
        val_ground_truths=data["val_gt"],
    )

    val_recs = trainer._get_recommendations(val_loader, data["seen_train"])
    val_metrics = compute_ranking_metrics(val_recs, data["val_gt"], base_cfg.training.eval_k)
    val_ndcg = val_metrics[f"ndcg@{base_cfg.training.eval_k}"]

    print(
        f"  Trial {trial.number:4d} | "
        f"embed={hparams['embed_dim']} hidden={hparams['hidden_dim']} "
        f"layers={hparams['num_layers']} dropout={hparams['dropout']} "
        f"lr={hparams['lr']:.0e} bs={hparams['batch_size']} "
        f"maxw={hparams['max_weight']} "
        f"-> val_ndcg@{base_cfg.training.eval_k}={val_ndcg:.4f}"
    )

    trial.report(val_ndcg, step=0)
    if trial.should_prune():
        raise optuna.TrialPruned()

    return val_ndcg


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


# ─────────────────────────────────────────────────────────────────────────────
# Fast mode: subsample users
# ─────────────────────────────────────────────────────────────────────────────

def subsample_data(data: dict, ratio: float, seed: int) -> dict:
    import random, copy
    rng = random.Random(seed)
    all_users = list(data["train_seqs"].keys())
    n_sample = max(1, int(len(all_users) * ratio))
    sampled_users = set(rng.sample(all_users, n_sample))

    print(f"  [fast mode] Subsampling {n_sample}/{len(all_users)} users "
          f"({ratio*100:.0f}%) for tuning ...")

    sub_train_seqs = {u: v for u, v in data["train_seqs"].items() if u in sampled_users}
    sub_val_data = {u: v for u, v in data["val_data"].items() if u in sampled_users}

    sub_train_ds = SequenceDataset(
        build_train_examples(sub_train_seqs), max_seq_len=data["train_ds"].max_seq_len,
    )
    sub_val_ds = SequenceDataset(
        build_eval_examples(sub_val_data), max_seq_len=data["val_ds"].max_seq_len,
    )

    sub_seen_train = {u: v for u, v in data["seen_train"].items() if u in sampled_users}
    sub_val_gt = {u: tgt for u, (_, tgt, _) in sub_val_data.items()}

    sub_data = copy.copy(data)
    sub_data["train_seqs"] = sub_train_seqs
    sub_data["val_data"] = sub_val_data
    sub_data["train_ds"] = sub_train_ds
    sub_data["val_ds"] = sub_val_ds
    sub_data["seen_train"] = sub_seen_train
    sub_data["val_gt"] = sub_val_gt
    return sub_data


# ─────────────────────────────────────────────────────────────────────────────
# Output writers
# ─────────────────────────────────────────────────────────────────────────────

def write_study_config(study_dir: str, args, base_cfg: ExperimentConfig) -> None:
    study_cfg = {
        "n_trials": args.n_trials,
        "trial_epochs": args.trial_epochs,
        "fast_mode": args.fast,
        "fast_ratio": args.fast_ratio if args.fast else None,
        "seed": args.seed,
        "dataset": base_cfg.data.data_path,
        "patience_during_tuning": min(base_cfg.training.patience, 2),
    }
    with open(os.path.join(study_dir, "study_config.yaml"), "w") as f:
        yaml.dump(study_cfg, f, default_flow_style=False, sort_keys=False)

    with open(os.path.join(study_dir, "search_space.yaml"), "w") as f:
        yaml.dump(SEARCH_SPACE, f, default_flow_style=False, sort_keys=False)


def write_study_summary(study_dir: str, study: optuna.Study, eval_k: int) -> None:
    rows = []
    for trial in study.trials:
        if trial.state != optuna.trial.TrialState.COMPLETE:
            continue
        row = {"trial_id": trial.number, f"val_ndcg@{eval_k}": round(trial.value, 6)}
        row.update(trial.params)
        row["status"] = trial.state.name
        rows.append(row)

    rows.sort(key=lambda x: x[f"val_ndcg@{eval_k}"], reverse=True)
    for i, row in enumerate(rows):
        row["rank"] = i + 1

    if not rows:
        return

    fieldnames = ["rank", "trial_id", f"val_ndcg@{eval_k}"] + list(SEARCH_SPACE.keys()) + ["status"]
    out_path = os.path.join(study_dir, "study_summary.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nStudy summary written to: {out_path}")


def write_best_config(study_dir: str, base_cfg: ExperimentConfig, best_params: Dict) -> str:
    best_cfg = {
        "data": {
            "data_path":        base_cfg.data.data_path,
            "min_rating":       base_cfg.data.min_rating,
            "min_interactions": base_cfg.data.min_interactions,
        },
        "model": {
            "embed_dim":   best_params["embed_dim"],
            "hidden_dim":  best_params["hidden_dim"],
            "num_layers":  best_params["num_layers"],
            "dropout":     best_params["dropout"],
            "max_seq_len": base_cfg.model.max_seq_len,
        },
        "training": {
            "batch_size":   best_params["batch_size"],
            "lr":           best_params["lr"],
            "weight_decay": best_params["weight_decay"],
            "epochs":       base_cfg.training.epochs,
            "patience":     base_cfg.training.patience,
            "eval_k":       base_cfg.training.eval_k,
            "device":       base_cfg.training.device,
            "num_workers":  base_cfg.training.num_workers,
            "pin_memory":   base_cfg.training.pin_memory,
            "use_amp":      base_cfg.training.use_amp,
        },
        "debiasing": {
            "loss_type":         "ce",
            "trend_window_days": base_cfg.debiasing.trend_window_days,
            "max_weight":        best_params["max_weight"],
            "use_alpha":         True,
            "use_beta":          True,
        },
    }
    out_path = os.path.join(study_dir, "best_config.yaml")
    with open(out_path, "w") as f:
        yaml.dump(best_cfg, f, default_flow_style=False, sort_keys=False)
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",       default="configs/default.yaml")
    parser.add_argument("--n_trials",     type=int, default=30)
    parser.add_argument("--trial_epochs", type=int, default=20)
    parser.add_argument("--seed",         type=int, default=42)
    parser.add_argument("--force-recompute", action="store_true",
                        help="Delete and recompute all feature caches (e_pop, e_trend, alpha/beta)")
    parser.add_argument("--fast",         action="store_true",
                        help="Subsample users for faster tuning")
    parser.add_argument("--fast_ratio",   type=float, default=0.2)
    args = parser.parse_args()

    base_cfg = ExperimentConfig.from_yaml(args.config)
    resolve_device(base_cfg)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    study_dir = os.path.join("results", "hptuning", f"study_{timestamp}")
    os.makedirs(study_dir, exist_ok=True)
    print(f"\nStudy results will be saved to: {study_dir}")

    data = setup_data(base_cfg, force_recompute=args.force_recompute)

    if args.fast:
        tuning_data = subsample_data(data, args.fast_ratio, args.seed)
    else:
        tuning_data = data

    write_study_config(study_dir, args, base_cfg)

    print("\n" + "=" * 60)
    print("Optuna hyperparameter search")
    print(f"  Method      : gru4rec_pers_formula")
    print(f"  Trials      : {args.n_trials}")
    print(f"  Epochs/trial: {args.trial_epochs}  (patience={min(base_cfg.training.patience, 2)})")
    print(f"  Objective   : val_ndcg@{base_cfg.training.eval_k}  (maximise)")
    if args.fast:
        n_users = len(tuning_data["train_seqs"])
        print(f"  Fast mode   : ON  ({args.fast_ratio*100:.0f}% → {n_users} users)")
    else:
        print(f"  Fast mode   : OFF (full dataset)")
    print("=" * 60 + "\n")

    sampler = TPESampler(seed=args.seed)
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        study_name="recsys_debiasing_hptuning",
    )

    study.optimize(
        lambda trial: run_trial(trial, base_cfg, tuning_data, args.trial_epochs, study_dir),
        n_trials=args.n_trials,
        catch=(Exception,),
    )

    best_trial = study.best_trial
    print("\n" + "=" * 60)
    print("Hyperparameter search complete.")
    print(f"  Best trial      : #{best_trial.number}")
    print(f"  Best val NDCG@{base_cfg.training.eval_k}: {best_trial.value:.4f}")
    print("\n  Best hyperparameters:")
    for k, v in best_trial.params.items():
        print(f"    {k:20s}: {v}")

    write_study_summary(study_dir, study, base_cfg.training.eval_k)
    best_config_path = write_best_config(study_dir, base_cfg, best_trial.params)

    print(f"\nBest config saved to: {best_config_path}")
    print("\nNext step — run final comparison with best config:")
    print(f"  python experiments/run_main.py --config {best_config_path}")

    print("\n  Hyperparameter importances:")
    try:
        importances = optuna.importance.get_param_importances(study)
        for param, importance in importances.items():
            bar = "█" * int(importance * 40)
            print(f"    {param:20s}: {bar}  {importance:.3f}")
    except Exception:
        print("    (importance analysis requires more completed trials)")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
