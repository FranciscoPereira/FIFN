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


def test_fedprox_training():
    model = FraudNet(input_dim=16)
    trainer = LocalTrainer(model, epochs=2)
    X, y = make_data()
    global_state = {k: v.clone() for k, v in model.state_dict().items()}
    metrics = trainer.train_fedprox(X, y, global_state, mu=0.01)
    assert len(metrics["loss_per_epoch"]) == 2
    assert metrics["n_samples"] == 200


def test_evaluate_auc():
    model = FraudNet(input_dim=16)
    trainer = LocalTrainer(model, epochs=3)
    X, y = make_data(n=400)
    trainer.train(X, y)
    auc = trainer.evaluate_auc(X, y)
    assert 0.0 <= auc <= 1.0


def test_model_save_load(tmp_path):
    from fifn.model.fraud_net import save_model, load_model

    model = FraudNet(input_dim=16)
    path = tmp_path / "model.pt"
    save_model(model, path)
    loaded = load_model(path)
    x = torch.randn(4, 16)
    model.eval()
    loaded.eval()
    with torch.no_grad():
        orig_out = model(x)
        loaded_out = loaded(x)
    assert torch.allclose(orig_out, loaded_out)


def test_score_shape():
    model = FraudNet(input_dim=16)
    trainer = LocalTrainer(model, epochs=1)
    X, y = make_data()
    trainer.train(X, y)
    scores = trainer.score(X[:10])
    assert scores.shape == (10,)
    assert np.all((scores >= 0) & (scores <= 1))
