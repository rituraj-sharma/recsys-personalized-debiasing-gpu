"""
Local run logger — replaces MLflow for storing per-run results to disk.

Writes three artefacts into run_dir:
  config.yaml        — full experiment config, written once at run start
  train_log.csv      — one row per epoch, appended incrementally
  final_metrics.json — test-set results, written once after evaluation
"""
from __future__ import annotations
import csv
import json
import os
from dataclasses import asdict
from typing import Dict

import yaml


class RunLogger:
    """
    Lightweight logger that writes training artefacts to a local directory.

    Parameters
    ----------
    run_dir:
        Directory for this run's artefacts. Created on init if absent.
    """

    def __init__(self, run_dir: str) -> None:
        self.run_dir = run_dir
        os.makedirs(run_dir, exist_ok=True)

    # ── public API ────────────────────────────────────────────────────────────

    def log_config(self, cfg) -> None:
        """Write the experiment config to config.yaml."""
        config_dict = {
            "data": asdict(cfg.data),
            "model": asdict(cfg.model),
            "training": asdict(cfg.training),
            "debiasing": asdict(cfg.debiasing),
        }
        out_path = os.path.join(self.run_dir, "config.yaml")
        with open(out_path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        val_metrics: Dict[str, float],
        lr: float,
        epoch_time_sec: float,
        grad_norm: float = None,
        **extra_stats,
    ) -> None:
        """Append one row to train_log.csv.  Safe to call after a crash (append mode)."""
        row: Dict = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "grad_norm": round(grad_norm, 6) if grad_norm is not None else "",
        }
        for k, v in extra_stats.items():
            row[k] = round(float(v), 6) if v is not None else ""
        row["lr"] = lr
        row["epoch_time_sec"] = round(epoch_time_sec, 1)
        for k, v in val_metrics.items():
            row[f"val_{k}"] = round(float(v), 6)

        log_path = os.path.join(self.run_dir, "train_log.csv")
        file_exists = os.path.exists(log_path) and os.path.getsize(log_path) > 0

        with open(log_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

    def log_final_metrics(self, metrics: Dict[str, float]) -> None:
        """Write test-set metrics to final_metrics.json."""
        out_path = os.path.join(self.run_dir, "final_metrics.json")
        with open(out_path, "w") as f:
            json.dump({k: round(float(v), 6) for k, v in metrics.items()}, f, indent=2)
