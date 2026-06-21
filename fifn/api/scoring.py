"""FastAPI scoring + federation API."""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)

_node = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _node
    config_path = Path(os.environ.get("FIFN_CONFIG", "config.yaml"))
    try:
        from fifn.config import load_config
        from fifn.client.node import FIFNNode

        cfg = load_config(config_path if config_path.exists() else None)
        _node = FIFNNode.from_config(cfg)
        logger.info("FIFN node initialised: %s", cfg.insurer_id)
    except Exception as exc:
        logger.warning("Node init failed (%s) — running in degraded mode", exc)
    yield
    _node = None


app = FastAPI(title="FIFN Fraud Scorer", version="1.0.0", lifespan=lifespan)


# ── Request / Response models ─────────────────────────────────────────────────

class ClaimFeatures(BaseModel):
    features: list[list[float]]
    threshold: float = 0.5


class ScoreResponse(BaseModel):
    fraud_scores: list[float]
    flags: list[bool]
    threshold: float


class RoundResult(BaseModel):
    round_id: str
    tx_hash: str
    n_samples: int
    loss_per_epoch: list[float]
    auc: Optional[float] = None


class HealthResponse(BaseModel):
    status: str
    node_ready: bool
    insurer_id: Optional[str] = None


class StatusResponse(BaseModel):
    insurer_id: str
    round: dict
    last_auc: Optional[float]
    model_params: int
    use_fedprox: bool
    dp_noise_scale: float


class MetricsResponse(BaseModel):
    last_auc: Optional[float]
    model_params: int


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        node_ready=_node is not None,
        insurer_id=_node.insurer_id if _node else None,
    )


@app.get("/status", response_model=StatusResponse)
def status():
    if _node is None:
        raise HTTPException(status_code=503, detail="Node not initialised")
    return StatusResponse(**_node.status())


@app.get("/metrics", response_model=MetricsResponse)
def metrics():
    if _node is None:
        raise HTTPException(status_code=503, detail="Node not initialised")
    return MetricsResponse(
        last_auc=_node.last_auc,
        model_params=sum(p.numel() for p in _node.model.parameters()),
    )


@app.post("/score", response_model=ScoreResponse)
def score_claims(payload: ClaimFeatures):
    if _node is None:
        raise HTTPException(status_code=503, detail="Node not initialised")
    X = np.array(payload.features, dtype=np.float32)
    scores = _node.score(X).tolist()
    flags = [s >= payload.threshold for s in scores]
    return ScoreResponse(fraud_scores=scores, flags=flags, threshold=payload.threshold)


@app.post("/run-round", response_model=RoundResult)
def trigger_round():
    if _node is None:
        raise HTTPException(status_code=503, detail="Node not initialised")
    result = _node.run_round()
    return RoundResult(
        round_id=result["round_id"],
        tx_hash=result["tx_hash"],
        n_samples=result["n_samples"],
        loss_per_epoch=result["loss_per_epoch"],
        auc=result.get("auc"),
    )


@app.post("/await-round/{round_id}")
def await_round(round_id: str, timeout: int = 3600):
    if _node is None:
        raise HTTPException(status_code=503, detail="Node not initialised")
    return _node.await_round(round_id, timeout=timeout)
