import io
import numpy as np
import torch
import pytest

from fifn.model.fraud_net import FraudNet
from fifn.federation.flock_client import MockFlockBackend


def _random_weights_bytes(input_dim: int = 16) -> tuple[bytes, int]:
    model = FraudNet(input_dim=input_dim)
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.getvalue(), 500


def test_mock_aggregates_after_min_participants():
    backend = MockFlockBackend(min_participants=2)
    w1, n1 = _random_weights_bytes()
    w2, n2 = _random_weights_bytes()

    round_id = backend.current_round()["round_id"]
    backend.submit_update(round_id, w1, n1)
    backend.submit_update(round_id, w2, n2)

    # global model should now be available
    global_weights = backend.get_global_model(round_id)
    assert isinstance(global_weights, bytes)
    assert len(global_weights) > 0


def test_mock_returns_initial_model_before_round():
    """Backend auto-initialises a fresh model for round 0 so nodes can start training."""
    backend = MockFlockBackend(min_participants=2)
    weights = backend.get_global_model("round-0")
    assert isinstance(weights, bytes) and len(weights) > 0


def test_await_aggregation_advances_round():
    backend = MockFlockBackend(min_participants=2)
    w1, n1 = _random_weights_bytes()
    w2, n2 = _random_weights_bytes()

    round_id = backend.current_round()["round_id"]
    backend.submit_update(round_id, w1, n1)
    backend.submit_update(round_id, w2, n2)

    next_round = backend.await_aggregation(round_id)
    assert next_round != round_id
