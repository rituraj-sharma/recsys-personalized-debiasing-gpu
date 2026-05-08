"""
Dataclass-based experiment configuration.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any, Dict
import yaml


@dataclass
class DataConfig:
    data_path: str = "data/ml-1m"
    min_rating: float = 4.0
    min_interactions: int = 5


@dataclass
class ModelConfig:
    embed_dim: int = 64
    hidden_dim: int = 128
    num_layers: int = 1
    dropout: float = 0.2
    max_seq_len: int = 50


@dataclass
class TrainingConfig:
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-5
    epochs: int = 50
    patience: int = 5
    eval_k: int = 10
    device: str = "cuda"
    num_workers: int = 4        # DataLoader workers; use 0 on Windows
    pin_memory: bool = True     # pinned memory for faster GPU transfer
    use_amp: bool = True        # mixed precision; auto-disabled on CPU


@dataclass
class DebiasingConfig:
    loss_type: str = "ce"
    trend_window_days: int = 30
    max_weight: float = 10.0
    use_alpha: bool = True
    use_beta: bool = True


@dataclass
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    debiasing: DebiasingConfig = field(default_factory=DebiasingConfig)

    def to_flat_dict(self) -> Dict[str, Any]:
        flat: Dict[str, Any] = {}
        for section_name in ("data", "model", "training", "debiasing"):
            section = getattr(self, section_name)
            for k, v in asdict(section).items():
                flat[f"{section_name}.{k}"] = v
        return flat

    @classmethod
    def from_yaml(cls, path: str) -> "ExperimentConfig":
        with open(path, "r") as f:
            raw = yaml.safe_load(f)
        return cls(
            data=DataConfig(**raw.get("data", {})),
            model=ModelConfig(**raw.get("model", {})),
            training=TrainingConfig(**raw.get("training", {})),
            debiasing=DebiasingConfig(**raw.get("debiasing", {})),
        )

    def override(self, **kwargs) -> "ExperimentConfig":
        import copy
        cfg = copy.deepcopy(self)
        for key, val in kwargs.items():
            section, attr = key.split(".", 1)
            setattr(getattr(cfg, section), attr, val)
        return cfg
