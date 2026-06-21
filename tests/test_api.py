"""Tests for the FastAPI scoring API."""
import io
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from fifn.model.fraud_net import FraudNet
from fifn.federation.flock_client import MockFlockBackend
from fifn.client.node import FIFNNode


INPUT_DIM = 32  # matches MockFlockBackend default input_dim


def _make_node() -> FIFNNode:
    flock = MockFlockBackend(min_participants=1, input_dim=INPUT_DIM)
    return FIFNNode(
        insurer_id="test-insurer",
        data_path=Path("/tmp/nonexistent.parquet"),
        flock=flock,
        input_dim=INPUT_DIM,
    )


@pytest.fixture()
def client():
    """TestClient that injects a test node AFTER the lifespan starts."""
    from fifn.api import scoring as api_module

    node = _make_node()
    with TestClient(api_module.app) as c:
        # Override whatever the lifespan set with our controlled test node
        api_module._node = node
        yield c
    api_module._node = None


def test_health_with_node(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["node_ready"] is True
    assert body["insurer_id"] == "test-insurer"


def test_health_without_node():
    from fifn.api import scoring as api_module

    with TestClient(api_module.app) as c:
        api_module._node = None  # force degraded after lifespan
        r = c.get("/health")
    assert r.status_code == 200
    assert r.json()["node_ready"] is False


def test_score_returns_probabilities(client):
    features = np.random.randn(4, INPUT_DIM).tolist()
    r = client.post("/score", json={"features": features})
    assert r.status_code == 200
    body = r.json()
    assert len(body["fraud_scores"]) == 4
    assert len(body["flags"]) == 4
    assert all(0.0 <= s <= 1.0 for s in body["fraud_scores"])


def test_score_custom_threshold(client):
    features = [[0.0] * INPUT_DIM]
    r = client.post("/score", json={"features": features, "threshold": 0.0})
    assert r.json()["flags"] == [True]

    r = client.post("/score", json={"features": features, "threshold": 1.0})
    assert r.json()["flags"] == [False]


def test_status_endpoint(client):
    r = client.get("/status")
    assert r.status_code == 200
    body = r.json()
    assert body["insurer_id"] == "test-insurer"
    assert "round" in body
    assert "model_params" in body


def test_metrics_endpoint(client):
    r = client.get("/metrics")
    assert r.status_code == 200
    body = r.json()
    assert "last_auc" in body
    assert "model_params" in body
    assert body["model_params"] > 0


def test_score_503_without_node():
    from fifn.api import scoring as api_module

    with TestClient(api_module.app) as c:
        api_module._node = None  # force degraded after lifespan
        r = c.post("/score", json={"features": [[1.0] * INPUT_DIM]})
    assert r.status_code == 503
