"""Minimal FastAPI scoring endpoint. Wire _node at application startup."""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

if TYPE_CHECKING:
    from fifn.client.node import FIFNNode

app = FastAPI(title="FIFN Fraud Scorer")

_node: FIFNNode | None = None


class ClaimFeatures(BaseModel):
    features: list[list[float]]


class ScoreResponse(BaseModel):
    fraud_scores: list[float]
    threshold: float = 0.5


@app.post("/score", response_model=ScoreResponse)
def score_claims(payload: ClaimFeatures):
    if _node is None:
        raise HTTPException(status_code=503, detail="Node not initialised")
    X = np.array(payload.features, dtype=np.float32)
    scores = _node.score(X).tolist()
    return ScoreResponse(fraud_scores=scores)


@app.post("/run-round")
def trigger_round():
    if _node is None:
        raise HTTPException(status_code=503, detail="Node not initialised")
    return _node.run_round()
