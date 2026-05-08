"""
Trainer — handles the full training lifecycle for sequential models.

GPU optimisations vs the original:
  - Mixed precision training via torch.amp.autocast + GradScaler
    (enabled when use_amp=True and device is CUDA).
  - torch.backends.cudnn.benchmark = True for fixed-size GRU inputs.
  - non_blocking=True on batch device transfers (pairs with pin_memory DataLoader).
  - RunLogger replaces MLflow for all metric and checkpoint logging.
"""
from __future__ import annotations
import os
import time
from typing import Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models import SequentialModel
from src.metrics import compute_ranking_metrics, compute_fairness_metrics
from src.utils.local_logger import RunLogger


class Trainer:
    def __init__(
        self,
        model: SequentialModel,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: str = "cpu",
        eval_k: int = 10,
        patience: int = 5,
        checkpoint_dir: str = "checkpoints",
        max_grad_norm: float = 5.0,
        lr_scheduler_factor: float = 0.5,
        lr_scheduler_patience: int = 5,
        logger: Optional[RunLogger] = None,
        use_amp: bool = False,
    ) -> None:
        self.model = model.to(device)
        self.loss_fn = loss_fn.to(device)
        self.optimizer = optimizer
        self.device = device
        self.eval_k = eval_k
        self.patience = patience
        self.checkpoint_dir = checkpoint_dir
        self.max_grad_norm = max_grad_norm
        self.logger = logger

        # AMP only makes sense on CUDA
        self.use_amp = use_amp and "cuda" in str(device)
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None

        if "cuda" in str(device):
            torch.backends.cudnn.benchmark = True

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=lr_scheduler_factor,
            patience=lr_scheduler_patience,
        )
        os.makedirs(checkpoint_dir, exist_ok=True)

    # ── public entry point ────────────────────────────────────────────────────

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        seen_items_train: Dict[int, Set[int]],
        val_ground_truths: Dict[int, int],
    ) -> str:
        """
        Run the training loop.

        Returns
        -------
        Path to the best checkpoint file.
        """
        checkpoint_path = os.path.join(self.checkpoint_dir, "best_model.pt")
        resume_path = os.path.join(self.checkpoint_dir, "resume.pt")

        best_ndcg = -1.0
        epochs_without_improvement = 0
        start_epoch = 1

        if os.path.exists(resume_path):
            state = torch.load(resume_path, map_location=self.device)
            self.model.load_state_dict(state["model"])
            self.optimizer.load_state_dict(state["optimizer"])
            self.scheduler.load_state_dict(state["scheduler"])
            if self.use_amp and self.scaler and "scaler" in state:
                self.scaler.load_state_dict(state["scaler"])
            best_ndcg = state["best_ndcg"]
            epochs_without_improvement = state["epochs_without_improvement"]
            start_epoch = state["epoch"] + 1
            tqdm.write(
                f"[resume] Resuming from epoch {start_epoch} "
                f"(best NDCG so far: {best_ndcg:.4f})"
            )

        for epoch in range(start_epoch, epochs + 1):
            t0 = time.time()
            train_loss = self._train_epoch(train_loader, epoch)
            val_metrics = self._evaluate(val_loader, seen_items_train, val_ground_truths)
            epoch_time = time.time() - t0

            ndcg = val_metrics[f"ndcg@{self.eval_k}"]
            self.scheduler.step(ndcg)
            current_lr = self.optimizer.param_groups[0]["lr"]

            if self.logger:
                self.logger.log_epoch(epoch, train_loss, val_metrics, current_lr, epoch_time)

            tqdm.write(
                f"Epoch {epoch:3d} | loss {train_loss:.4f} | "
                f"Recall@{self.eval_k} {val_metrics[f'recall@{self.eval_k}']:.4f} | "
                f"NDCG@{self.eval_k} {ndcg:.4f} | "
                f"lr {current_lr:.2e} | {epoch_time:.1f}s"
            )

            if ndcg > best_ndcg:
                best_ndcg = ndcg
                epochs_without_improvement = 0
                torch.save(self.model.state_dict(), checkpoint_path)
            else:
                epochs_without_improvement += 1

            resume_state = {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "epoch": epoch,
                "best_ndcg": best_ndcg,
                "epochs_without_improvement": epochs_without_improvement,
            }
            if self.use_amp and self.scaler:
                resume_state["scaler"] = self.scaler.state_dict()
            torch.save(resume_state, resume_path)

            if epochs_without_improvement >= self.patience:
                tqdm.write(f"Early stopping at epoch {epoch}.")
                break

        if os.path.exists(resume_path):
            os.remove(resume_path)

        self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        return checkpoint_path

    def evaluate_test(
        self,
        test_loader: DataLoader,
        seen_items_all: Dict[int, Set[int]],
        test_ground_truths: Dict[int, int],
        train_seqs: Dict,
        e_pop: Dict[int, float],
        user_groups: Dict[int, str],
        num_items: int,
    ) -> Dict[str, float]:
        acc_metrics = self._evaluate(test_loader, seen_items_all, test_ground_truths)

        recommendations = self._get_recommendations(test_loader, seen_items_all)

        fairness = compute_fairness_metrics(
            recommendations, train_seqs, e_pop, user_groups, num_items
        )

        all_metrics = {**acc_metrics, **fairness}

        if self.logger:
            self.logger.log_final_metrics(all_metrics)

        print("\n=== Test Results ===")
        for k, v in sorted(all_metrics.items()):
            print(f"  {k:30s}: {v:.4f}")

        return all_metrics

    # ── private helpers ───────────────────────────────────────────────────────

    def _train_epoch(self, loader: DataLoader, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in tqdm(loader, desc=f"Epoch {epoch}", leave=False):
            # non_blocking pairs with pin_memory=True on the DataLoader
            batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}

            self.optimizer.zero_grad()

            if self.use_amp:
                with torch.amp.autocast(device_type="cuda"):
                    logits = self.model(batch)
                    loss = self.loss_fn(logits, batch)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits = self.model(batch)
                loss = self.loss_fn(logits, batch)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    def _evaluate(
        self,
        loader: DataLoader,
        seen_items: Dict[int, Set[int]],
        ground_truths: Dict[int, int],
    ) -> Dict[str, float]:
        recommendations = self._get_recommendations(loader, seen_items)
        return compute_ranking_metrics(recommendations, ground_truths, self.eval_k)

    def _get_recommendations(
        self,
        loader: DataLoader,
        seen_items: Dict[int, Set[int]],
    ) -> Dict[int, List[int]]:
        self.model.eval()
        recommendations: Dict[int, List[int]] = {}

        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
                top_k = self.model.predict_top_k(batch, self.eval_k, seen_items)
                user_ids = batch["user_idx"].tolist()
                for user_idx, recs in zip(user_ids, top_k.tolist()):
                    recommendations[user_idx] = recs

        return recommendations


# ─────────────────────────────────────────────────────────────────────────────
# Utility: build seen-item sets
# ─────────────────────────────────────────────────────────────────────────────

def build_seen_items(
    train_seqs: Dict[int, List[Tuple[int, int]]],
    include_val: Optional[Dict[int, Tuple[List[int], int, int]]] = None,
) -> Dict[int, Set[int]]:
    seen: Dict[int, Set[int]] = {}
    for user_idx, seq in train_seqs.items():
        seen[user_idx] = {item for item, _ in seq}

    if include_val is not None:
        for user_idx, (val_input, val_target, _) in include_val.items():
            seen.setdefault(user_idx, set()).add(val_target)

    return seen
