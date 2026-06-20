from __future__ import annotations
import io
from pathlib import Path

import numpy as np
import torch

from fifn.model.fraud_net import FraudNet, LocalTrainer
from fifn.federation.flock_client import FlockBackend
from fifn.data.loader import load_claims_features


def _weights_to_bytes(model: FraudNet) -> bytes:
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.getvalue()


def _bytes_to_weights(model: FraudNet, raw: bytes):
    buf = io.BytesIO(raw)
    model.load_state_dict(torch.load(buf, weights_only=True))


def _add_dp_noise(model: FraudNet, noise_scale: float = 0.01):
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
    ):
        self.insurer_id = insurer_id
        self.data_path = data_path
        self.flock = flock
        self.dp_noise_scale = dp_noise_scale
        self.model = FraudNet(input_dim=input_dim, hidden=hidden)
        self.trainer = LocalTrainer(self.model)

    def run_round(self) -> dict:
        round_info = self.flock.current_round()
        round_id = round_info["round_id"]

        global_bytes = self.flock.get_global_model(round_id)
        _bytes_to_weights(self.model, global_bytes)
        print(f"[{self.insurer_id}] Pulled global model for {round_id}")

        X, y = load_claims_features(self.data_path)
        metrics = self.trainer.train(X, y)
        print(
            f"[{self.insurer_id}] Trained on {metrics['n_samples']} samples, "
            f"final loss: {metrics['loss_per_epoch'][-1]:.4f}"
        )

        if self.dp_noise_scale > 0:
            _add_dp_noise(self.model, self.dp_noise_scale)

        weight_bytes = _weights_to_bytes(self.model)
        tx_hash = self.flock.submit_update(round_id, weight_bytes, metrics["n_samples"])
        print(f"[{self.insurer_id}] Submitted update — tx: {tx_hash[:16]}…")

        return {"round_id": round_id, "tx_hash": tx_hash, **metrics}

    def score(self, X: np.ndarray) -> np.ndarray:
        """Returns fraud probability in [0, 1] for each claim row."""
        return self.trainer.score(X)
