import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class FraudNet(nn.Module):
    """Shallow MLP for tabular fraud detection. Small by design — keeps on-chain payload minimal."""

    def __init__(self, input_dim: int, hidden: int = 64, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class LocalTrainer:
    def __init__(
        self,
        model: FraudNet,
        lr: float = 1e-3,
        epochs: int = 5,
        pos_weight: float = 10.0,
    ):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        # pos_weight handles class imbalance — fraud is rare in real claims data
        self.criterion = nn.BCELoss(weight=torch.tensor(pos_weight))
        self.epochs = epochs

    def train(self, X: np.ndarray, y: np.ndarray) -> dict:
        ds = TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
        )
        loader = DataLoader(ds, batch_size=64, shuffle=True)
        history = []

        self.model.train()
        for _ in range(self.epochs):
            epoch_loss = 0.0
            for xb, yb in loader:
                self.optimizer.zero_grad()
                loss = self.criterion(self.model(xb), yb)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
            history.append(epoch_loss / len(loader))

        return {"loss_per_epoch": history, "n_samples": len(ds)}

    def train_fedprox(
        self,
        X: np.ndarray,
        y: np.ndarray,
        global_state: dict,
        mu: float = 0.01,
    ) -> dict:
        """FedProx variant — proximal term prevents divergence on heterogeneous nodes."""
        global_params = {k: v.clone().detach() for k, v in global_state.items()}
        ds = TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
        )
        loader = DataLoader(ds, batch_size=64, shuffle=True)
        history = []

        self.model.train()
        for _ in range(self.epochs):
            epoch_loss = 0.0
            for xb, yb in loader:
                self.optimizer.zero_grad()
                task_loss = self.criterion(self.model(xb), yb)
                prox = sum(
                    torch.sum((p - global_params[n]) ** 2)
                    for n, p in self.model.named_parameters()
                )
                loss = task_loss + (mu / 2) * prox
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
            history.append(epoch_loss / len(loader))

        return {"loss_per_epoch": history, "n_samples": len(ds)}

    def score(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            return self.model(torch.tensor(X, dtype=torch.float32)).numpy()
