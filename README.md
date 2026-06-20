# Federated Insurance Fraud Network (FIFN)

A decentralized fraud detection system enabling multiple small insurers to collaboratively train a shared model on their private claims data using [Flock.io](https://flock.io)'s secure, on-chain federated learning infrastructure—without exposing sensitive customer information.

## Overview

FIFN allows competing insurers to benefit from a collectively trained fraud detection model while keeping their raw claims data fully private. Each participant trains locally and contributes only model updates, which are aggregated on-chain via Flock.io's federated learning protocol.

## Key Features

- **Privacy-preserving** — raw claims data never leaves each insurer's environment
- **Decentralized aggregation** — model updates are coordinated on-chain via Flock.io
- **Collaborative intelligence** — small insurers gain the detection power of a network-wide model
- **Auditable** — all federation rounds are recorded on-chain for transparency

## Architecture

```
Insurer A ──┐
Insurer B ──┼──▶ Flock.io On-Chain Aggregator ──▶ Global Fraud Model
Insurer C ──┘
```

Each node:
1. Trains a local model on its private claims data
2. Submits encrypted model weights to the Flock.io contract
3. Receives the globally aggregated model after each federation round

## Getting Started

```bash
# Clone the repo
git clone <repo-url>
cd FIFN

# Install dependencies
pip install -r requirements.txt

# Configure your node
cp config.example.yaml config.yaml
# Edit config.yaml with your Flock.io credentials and data paths

# Run a federation round
python fifn/train.py
```

## Project Structure

```
FIFN/
├── fifn/
│   ├── client/        # Local training logic per insurer node
│   ├── model/         # Fraud detection model architecture
│   ├── federation/    # Flock.io integration and aggregation hooks
│   └── data/          # Data loading and preprocessing utilities
├── contracts/         # On-chain smart contracts (if any custom logic)
├── tests/
├── config.example.yaml
└── requirements.txt
```

## License

MIT
