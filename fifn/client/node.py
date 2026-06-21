from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from fifn.model.fraud_net import FraudNet, LocalTrainer, save_model, load_model
from fifn.federation.flock_client import FlockBackend
from fifn.data.loader import load_claims_features

logger = logging.getLogger(__name__)


def _weights_to_bytes(model: FraudNet) -> bytes:
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.getvalue()


def _bytes_to_weights(model: FraudNet, raw: bytes) -> None:
    buf = io.BytesIO(raw)
    model.load_state_dict(torch.load(buf, weights_only=True))


def _add_dp_noise(model: FraudNet, noise_scale: float = 0.01) -> None:
    with torch.no_grad():
        for param in model.parameters():
            param.add_(torch.randn_like(param) * noise_scale)


class FIFNNode:
    """One insurer's federated node. Call run_round() each federation round."""

    def __init__(
        self,
        insurer_id: str,
        data_path: Path,
        flock: FlockBackend,
        input_dim: int,
        hidden: int = 64,
        dp_noise_scale: float = 0.01,
        use_fedprox: bool = False,
        fedprox_mu: float = 0.01,
        model_dir: Optional[Path] = None,
    ):
        self.insurer_id = insurer_id
        self.data_path = Path(data_path)
        self.flock = flock
        self.dp_noise_scale = dp_noise_scale
        self.use_fedprox = use_fedprox
        self.fedprox_mu = fedprox_mu
        self.model_dir = Path(model_dir) if model_dir else None
        self.model = FraudNet(input_dim=input_dim, hidden=hidden)
        self.trainer = LocalTrainer(self.model)
        self._last_auc: Optional[float] = None

    @classmethod
    def from_config(cls, cfg) -> "FIFNNode":
        """Construct a node from a FIFNConfig object."""
        from fifn.federation.flock_client import FlockSDKBackend, MockFlockBackend

        if cfg.flock.mock:
            flock = MockFlockBackend(min_participants=cfg.privacy.min_participants)
        else:
            flock = FlockSDKBackend(
                task_id=cfg.flock.task_id,
                api_key=cfg.flock.api_key,
                encrypt=cfg.flock.encrypt,
            )

        node = cls(
            insurer_id=cfg.insurer_id,
            data_path=cfg.data_path,
            flock=flock,
            input_dim=cfg.model.input_dim,
            hidden=cfg.model.hidden,
            dp_noise_scale=cfg.privacy.dp_noise_scale,
            use_fedprox=cfg.training.use_fedprox,
            fedprox_mu=cfg.training.fedprox_mu,
            model_dir=cfg.model_dir,
        )
        node.trainer = LocalTrainer(
            node.model,
            lr=cfg.training.lr,
            epochs=cfg.training.epochs,
            pos_weight=cfg.training.pos_weight,
        )

        # Restore persisted model weights if available
        if node.model_dir:
            checkpoint = node.model_dir / "global_model.pt"
            if checkpoint.exists():
                node.model = load_model(checkpoint)
                node.trainer.model = node.model
                logger.info("[%s] Restored model from %s", cfg.insurer_id, checkpoint)

        return node

    def run_round(self) -> dict:
        round_info = self.flock.current_round()
        round_id = round_info["round_id"]

        global_bytes = self.flock.get_global_model(round_id)
        global_state = torch.load(io.BytesIO(global_bytes), weights_only=True)
        self.model.load_state_dict(global_state)
        logger.info("[%s] Pulled global model for %s", self.insurer_id, round_id)

        X, y = load_claims_features(self.data_path)

        if self.use_fedprox:
            metrics = self.trainer.train_fedprox(X, y, global_state, mu=self.fedprox_mu)
            logger.info("[%s] FedProx training on %d samples", self.insurer_id, metrics["n_samples"])
        else:
            metrics = self.trainer.train(X, y)
            logger.info("[%s] FedAvg training on %d samples", self.insurer_id, metrics["n_samples"])

        auc = self.trainer.evaluate_auc(X, y)
        metrics["auc"] = auc
        self._last_auc = auc
        logger.info("[%s] Local AUC: %.4f", self.insurer_id, auc)

        if self.dp_noise_scale > 0:
            _add_dp_noise(self.model, self.dp_noise_scale)

        weight_bytes = _weights_to_bytes(self.model)
        tx_hash = self.flock.submit_update(round_id, weight_bytes, metrics["n_samples"])
        logger.info("[%s] Submitted update — tx: %s…", self.insurer_id, tx_hash[:16])

        return {"round_id": round_id, "tx_hash": tx_hash, **metrics}

    def await_round(self, round_id: str, timeout: int = 3600) -> dict:
        """Wait for aggregation, pull the new global model, and optionally persist it."""
        next_round_id = self.flock.await_aggregation(round_id, timeout=timeout)
        global_bytes = self.flock.get_global_model(next_round_id)
        _bytes_to_weights(self.model, global_bytes)
        logger.info("[%s] Received aggregated model for %s", self.insurer_id, next_round_id)

        if self.model_dir:
            save_model(self.model, self.model_dir / "global_model.pt")

        return {"round_id": next_round_id}

    def score(self, X: np.ndarray) -> np.ndarray:
        """Returns fraud probability in [0, 1] for each claim row."""
        return self.trainer.score(X)

    @property
    def last_auc(self) -> Optional[float]:
        return self._last_auc

    def status(self) -> dict:
        """Return a summary dict suitable for the /status API endpoint."""
        round_info = self.flock.current_round()
        return {
            "insurer_id": self.insurer_id,
            "round": round_info,
            "last_auc": self._last_auc,
            "model_params": sum(p.numel() for p in self.model.parameters()),
            "use_fedprox": self.use_fedprox,
            "dp_noise_scale": self.dp_noise_scale,
        }
