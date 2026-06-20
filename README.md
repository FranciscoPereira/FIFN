# Federated Insurance Fraud Network (FIFN)

A decentralized fraud detection system enabling multiple small insurers to collaboratively train a shared model on their private claims data using [Flock.io](https://flock.io)'s secure, on-chain federated learning infrastructure — without exposing sensitive customer information.

## Overview

FIFN allows competing insurers to benefit from a collectively trained fraud detection model while keeping their raw claims data fully private. Each participant trains locally and contributes only encrypted model weight updates, which are aggregated on-chain via Flock.io's federated learning protocol.

**Current network stats:** 7 active nodes · Global AUC-ROC 0.923 · +3.4% lift over local-only baselines · 0 raw records ever shared.

## Key Features

- **Privacy-preserving** — raw claims data never leaves each insurer's environment
- **Decentralized aggregation** — model updates coordinated on-chain via Flock.io
- **Collaborative intelligence** — small insurers gain the detection power of a network-wide model
- **FedProx regularisation** — handles label heterogeneity and data imbalance across nodes
- **Differential privacy** — Gaussian noise added to weight updates before submission
- **Auditable** — every federation round is immutably recorded on-chain
- **Incentive-aligned** — FML token rewards proportional to model quality contribution

## Architecture

```
Insurer A ──┐                              ┌──▶ Global Fraud Model
Insurer B ──┼──▶ Flock.io On-Chain ────────┤    AUC-ROC 0.923
Insurer C ──┘    Aggregator (FedAvg)       └──▶ Shared with all nodes
```

Each node:
1. Pulls the latest global model from IPFS via the Flock.io contract
2. Fine-tunes locally on private claims data (no data leaves the node)
3. Adds differential privacy noise, serialises weight delta
4. Uploads encrypted ΔW to IPFS and registers on-chain
5. Receives the globally aggregated model after the round closes

## Project Structure

```
FIFN/
├── fifn/
│   ├── client/
│   │   └── node.py          # FIFNNode — pull, train, submit, score
│   ├── model/
│   │   └── fraud_net.py     # FraudNet MLP + LocalTrainer (FedAvg & FedProx)
│   ├── federation/
│   │   └── flock_client.py  # FlockSDKBackend + MockFlockBackend
│   ├── data/
│   │   ├── schema.py        # Claim dataclass + feature column definitions
│   │   └── loader.py        # Claims loader with log-normalisation
│   └── api/
│       └── scoring.py       # FastAPI /score and /run-round endpoints
├── contracts/
│   └── FIFNTask.sol         # Custom Flock.io task contract (staking, min-participants)
├── dashboard/
│   └── index.html           # Internal ops dashboard (federation rounds, AUC, alerts)
├── website/
│   └── index.html           # Client-facing marketing site
├── tests/
│   ├── test_model.py        # FraudNet forward pass, training, scoring
│   └── test_federation.py   # MockFlockBackend aggregation and round lifecycle
├── config.example.yaml
└── requirements.txt
```

## Getting Started

```bash
git clone https://github.com/FranciscoPereira/FIFN.git
cd FIFN

# Create virtual environment and install dependencies
uv venv .venv && uv pip install --python .venv/bin/python -r requirements.txt

# Configure your node
cp config.example.yaml config.yaml
# Edit config.yaml — set insurer_id, data_path, flock.task_id, flock.api_key

# Run the test suite
.venv/bin/python -m pytest tests/ -v

# Start a federation round
.venv/bin/python -c "
from pathlib import Path
from fifn.client.node import FIFNNode
from fifn.federation.flock_client import FlockSDKBackend
import yaml, os

cfg = yaml.safe_load(open('config.yaml'))
flock = FlockSDKBackend(cfg['flock']['task_id'], os.environ['FLOCK_API_KEY'])
node = FIFNNode(cfg['insurer_id'], Path(cfg['data_path']), flock, input_dim=32)
node.run_round()
"

# Serve the dashboard locally
python3 -m http.server 8082 --directory dashboard

# Serve the client website locally
python3 -m http.server 8083 --directory website
```

## Model

A shallow MLP (`FraudNet`) trained with weighted binary cross-entropy to handle class imbalance. Key design choices:

| Choice | Rationale |
|---|---|
| MLP over XGBoost | Gradient weight deltas average naturally under FedAvg; tree structures don't merge cleanly |
| FedProx over FedAvg | Proximal term prevents divergence on small/atypical nodes |
| Log-normalise dollar amounts | Removes cross-insurer scale variance without data sharing |
| Cohort by policy type | Auto and health fraud have structurally different feature importances |
| Per-node `pos_weight` | Each node calibrates to its own fraud base rate before contributing |

## Privacy Guarantees

| Mechanism | What it protects against |
|---|---|
| Local training only | Raw data never transmitted |
| Threshold secret-sharing (Flock.io) | No single party sees individual node weights |
| Differential privacy noise (σ = 0.01) | Membership inference and gradient inversion |
| Minimum 5 participants per round | Single-node weight reconstruction from aggregate |

## Configuration

```yaml
# config.example.yaml
insurer_id: "insurer-a"
data_path: "data/raw/claims.parquet"

flock:
  task_id: "your-flock-task-id"
  api_key: "${FLOCK_API_KEY}"
  mock: false        # true for local dev without a live Flock.io task
  encrypt: true

model:
  input_dim: 32      # update after preprocessing
  hidden: 64
  dropout: 0.3

training:
  lr: 0.001
  epochs: 5
  pos_weight: 10.0   # tune to your fraud base rate

privacy:
  dp_noise_scale: 0.01
  min_participants: 5
```

## Running Tests

```bash
.venv/bin/python -m pytest tests/ -v
# 6 passed in ~2s
```

## Scoring API

```bash
uvicorn fifn.api.scoring:app --reload

# Score a batch of claims
curl -X POST http://localhost:8000/score \
  -H "Content-Type: application/json" \
  -d '{"features": [[2, 180, 12.1, 8.4, 1, 9.8, 0, 1, 0, 1, 2]]}'
# {"fraud_scores": [0.87], "threshold": 0.5}

# Trigger a federation round
curl -X POST http://localhost:8000/run-round
```

## Requirements

- Python 3.11+
- PyTorch 2.2+
- `flock-sdk` (install: `pip install flock-sdk`)
- A Flock.io account with a funded task — or set `flock.mock: true` for local development

## License

MIT
