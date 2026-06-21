"""Config loading from YAML or environment variables."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class FlockConfig:
    task_id: str = ""
    api_key: str = ""
    mock: bool = True
    encrypt: bool = True


@dataclass
class ModelConfig:
    input_dim: int = 32
    hidden: int = 64
    dropout: float = 0.3


@dataclass
class TrainingConfig:
    lr: float = 1e-3
    epochs: int = 5
    pos_weight: float = 10.0
    use_fedprox: bool = False
    fedprox_mu: float = 0.01


@dataclass
class PrivacyConfig:
    dp_noise_scale: float = 0.01
    min_participants: int = 5


@dataclass
class FIFNConfig:
    insurer_id: str = "insurer-local"
    data_path: str = "data/raw/claims.parquet"
    model_dir: str = "models"
    flock: FlockConfig = field(default_factory=FlockConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    privacy: PrivacyConfig = field(default_factory=PrivacyConfig)


def load_config(path: Optional[Path] = None) -> FIFNConfig:
    """Load config from YAML file, falling back to defaults. Env vars override file values."""
    raw: dict = {}
    if path and Path(path).exists():
        with open(path) as f:
            raw = yaml.safe_load(f) or {}

    flock_raw = raw.get("flock", {})
    api_key = os.environ.get("FLOCK_API_KEY", flock_raw.get("api_key", ""))
    if isinstance(api_key, str) and api_key.startswith("${"):
        api_key = os.environ.get(api_key[2:-1], "")

    return FIFNConfig(
        insurer_id=raw.get("insurer_id", "insurer-local"),
        data_path=raw.get("data_path", "data/raw/claims.parquet"),
        model_dir=raw.get("model_dir", "models"),
        flock=FlockConfig(
            task_id=flock_raw.get("task_id", ""),
            api_key=api_key,
            mock=flock_raw.get("mock", True),
            encrypt=flock_raw.get("encrypt", True),
        ),
        model=ModelConfig(**raw["model"]) if "model" in raw else ModelConfig(),
        training=TrainingConfig(**raw["training"]) if "training" in raw else TrainingConfig(),
        privacy=PrivacyConfig(**raw["privacy"]) if "privacy" in raw else PrivacyConfig(),
    )
