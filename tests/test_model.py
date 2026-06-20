import numpy as np
import pytest
import torch

from fifn.model.fraud_net import FraudNet, LocalTrainer


def make_data(n: int = 200, input_dim: int = 16):
    X = np.random.randn(n, input_dim).astype(np.float32)
    y = (np.random.rand(n) > 0.9).astype(np.float32)  # ~10% fraud
    return X, y


def test_fraud_net_forward():
    model = FraudNet(input_dim=16)
    x = torch.randn(8, 16)
    out = model(x)
    assert out.shape == (8,)
    assert (out >= 0).all() and (out <= 1).all()


def test_local_trainer_runs():
    model = FraudNet(input_dim=16)
    trainer = LocalTrainer(model, epochs=2)
    X, y = make_data()
    metrics = trainer.train(X, y)
    assert len(metrics["loss_per_epoch"]) == 2
    assert metrics["n_samples"] == 200


def test_score_shape():
    model = FraudNet(input_dim=16)
    trainer = LocalTrainer(model, epochs=1)
    X, y = make_data()
    trainer.train(X, y)
    scores = trainer.score(X[:10])
    assert scores.shape == (10,)
    assert np.all((scores >= 0) & (scores <= 1))
