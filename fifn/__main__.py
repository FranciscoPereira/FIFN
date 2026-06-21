"""python -m fifn <command> [options]"""
import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def cmd_run_round(args):
    from fifn.config import load_config
    from fifn.client.node import FIFNNode

    cfg = load_config(Path(args.config))
    node = FIFNNode.from_config(cfg)
    result = node.run_round()
    print(f"Round {result['round_id']} complete — tx: {result['tx_hash'][:16]}…")
    print(f"  Samples: {result['n_samples']}  AUC: {result.get('auc', 'n/a'):.4f}")
    if args.await_agg:
        agg = node.await_round(result["round_id"])
        print(f"  New global model available at {agg['round_id']}")


def cmd_generate_data(args):
    from fifn.data.synthetic import generate_claims

    out = Path(args.output)
    generate_claims(
        n_samples=args.n_samples,
        fraud_rate=args.fraud_rate,
        seed=args.seed,
        output_path=out,
    )
    print(f"Generated {args.n_samples} synthetic claims → {out}")


def cmd_serve(args):
    import uvicorn

    uvicorn.run(
        "fifn.api.scoring:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


def cmd_score(args):
    import json
    import numpy as np
    from fifn.config import load_config
    from fifn.client.node import FIFNNode

    cfg = load_config(Path(args.config))
    node = FIFNNode.from_config(cfg)

    features = json.loads(args.features)
    X = np.array(features, dtype=np.float32)
    if X.ndim == 1:
        X = X[np.newaxis, :]
    scores = node.score(X)
    for i, s in enumerate(scores):
        flag = "FRAUD" if s >= args.threshold else "clean"
        print(f"  [{i}] score={s:.4f}  [{flag}]")


def main():
    parser = argparse.ArgumentParser(prog="python -m fifn", description="FIFN CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    # run-round
    p_round = sub.add_parser("run-round", help="Execute one federation round")
    p_round.add_argument("--config", default="config.yaml")
    p_round.add_argument("--await", dest="await_agg", action="store_true",
                         help="Wait for aggregation and pull new global model")

    # generate-data
    p_data = sub.add_parser("generate-data", help="Generate synthetic claims parquet")
    p_data.add_argument("--output", default="data/raw/claims.parquet")
    p_data.add_argument("--n-samples", type=int, default=1000)
    p_data.add_argument("--fraud-rate", type=float, default=0.08)
    p_data.add_argument("--seed", type=int, default=42)

    # serve
    p_serve = sub.add_parser("serve", help="Start the FastAPI scoring server")
    p_serve.add_argument("--host", default="0.0.0.0")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.add_argument("--reload", action="store_true")

    # score
    p_score = sub.add_parser("score", help="Score a single claim from JSON features")
    p_score.add_argument("features", help='JSON array, e.g. "[[2, 180, ...]]"')
    p_score.add_argument("--config", default="config.yaml")
    p_score.add_argument("--threshold", type=float, default=0.5)

    args = parser.parse_args()
    dispatch = {
        "run-round": cmd_run_round,
        "generate-data": cmd_generate_data,
        "serve": cmd_serve,
        "score": cmd_score,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
