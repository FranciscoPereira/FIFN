"""
Flock.io integration layer.

Real SDK:  pip install flock-sdk  (https://docs.flock.io/sdk)
Mock backend available for local dev — set flock.mock: true in config.yaml.
"""
from __future__ import annotations

import hashlib
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class FlockBackend(Protocol):
    def get_global_model(self, round_id: str) -> bytes: ...
    def submit_update(self, round_id: str, weights: bytes, n_samples: int) -> str: ...
    def current_round(self) -> dict: ...
    def await_aggregation(self, round_id: str, timeout: int = 3600) -> str: ...


class FlockSDKBackend:
    """Thin wrapper around the official flock-sdk."""

    def __init__(self, task_id: str, api_key: str, encrypt: bool = True):
        from flock_sdk import FlockClient  # pip install flock-sdk

        self._client = FlockClient(api_key=api_key)
        self._task_id = task_id
        self._encrypt = encrypt

    def current_round(self) -> dict:
        return self._client.tasks.get_current_round(self._task_id)

    def get_global_model(self, round_id: str) -> bytes:
        cid = self._client.tasks.get_global_model_cid(self._task_id, round_id)
        return self._client.ipfs.download(cid)

    def submit_update(self, round_id: str, weights: bytes, n_samples: int) -> str:
        if self._encrypt:
            weights = self._client.crypto.encrypt(weights)
        cid = self._client.ipfs.upload(weights)
        tx = self._client.tasks.submit_model(
            task_id=self._task_id,
            round_id=round_id,
            model_cid=cid,
            data_size=n_samples,
        )
        return tx["transaction_hash"]

    def await_aggregation(self, round_id: str, timeout: int = 3600) -> str:
        return self._client.tasks.wait_for_aggregation(
            self._task_id, round_id, timeout=timeout
        )


class MockFlockBackend:
    """
    In-memory mock performing plain FedAvg locally.
    Auto-aggregates once ≥ min_participants submissions arrive.
    """

    def __init__(self, min_participants: int = 2):
        self._round_num = 0
        self._min_participants = min_participants
        self._submissions: list[tuple[bytes, int]] = []
        self._global_weights: bytes | None = None

    @property
    def _round_id(self) -> str:
        return f"round-{self._round_num}"

    def current_round(self) -> dict:
        return {
            "round_id": self._round_id,
            "status": "open",
            "min_participants": self._min_participants,
        }

    def get_global_model(self, round_id: str) -> bytes:
        if self._global_weights is None:
            raise RuntimeError("No global model yet — complete at least one round.")
        return self._global_weights

    def submit_update(self, round_id: str, weights: bytes, n_samples: int) -> str:
        self._submissions.append((weights, n_samples))
        if len(self._submissions) >= self._min_participants:
            self._aggregate()
        return hashlib.sha256(weights[:64]).hexdigest()

    def await_aggregation(self, round_id: str, timeout: int = 3600) -> str:
        if len(self._submissions) < self._min_participants:
            raise RuntimeError(
                f"Need {self._min_participants} submissions, have {len(self._submissions)}"
            )
        self._aggregate()
        self._round_num += 1
        self._submissions.clear()
        return self._round_id

    def _aggregate(self):
        import io
        import torch

        total = sum(n for _, n in self._submissions)
        avg_state: dict | None = None

        for weights_bytes, n in self._submissions:
            state = torch.load(io.BytesIO(weights_bytes), weights_only=True)
            scale = n / total
            if avg_state is None:
                avg_state = {k: v.float() * scale for k, v in state.items()}
            else:
                for k in avg_state:
                    avg_state[k] += state[k].float() * scale

        buf = io.BytesIO()
        torch.save(avg_state, buf)
        self._global_weights = buf.getvalue()
