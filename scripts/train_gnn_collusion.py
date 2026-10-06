#!/usr/bin/env python3
"""
scripts/train_gnn_collusion.py
==============================
CLI training script for the ClaimGuard AI heterogeneous GNN collusion detector.

Pipeline
--------
1.  Load claims data  (CSV path or PostgreSQL via DATABASE_URL)
2.  Build heterogeneous PyG graph  (claimant / garage / medical nodes)
3.  Train the R-GCN model          (configurable via CLI flags)
4.  Evaluate on full graph         (loss / accuracy / precision / recall / F1)
5.  Write collusion scores → Neo4j (batched MERGE on :Claimant nodes)
6.  Export a JSON score file       (for inspection / downstream use)

Usage
-----
    # Minimal (uses data/sample_claims.csv, no Neo4j write)
    python scripts/train_gnn_collusion.py

    # Full run with Neo4j export
    python scripts/train_gnn_collusion.py \\
        --claims-csv data/sample_claims.csv \\
        --epochs 100 \\
        --hidden-dim 64 \\
        --lr 1e-3 \\
        --write-neo4j \\
        --scores-out data/gnn_scores.json

    # With PostgreSQL source
    python scripts/train_gnn_collusion.py \\
        --db-url postgresql://user:pass@localhost/claimguard \\
        --epochs 80 \\
        --write-neo4j

Exit codes
----------
0  success
1  fatal error (missing data, training failure)
2  PyG / torch not installed  (soft exit — not a CI blocker)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

# Ensure repo root is on sys.path so src.* imports work from any cwd
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(_REPO_ROOT / ".env")
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("train_gnn_collusion")


# ---------------------------------------------------------------------------
# Availability checks
# ---------------------------------------------------------------------------

def _check_pyg() -> bool:
    try:
        import torch                         # noqa: F401
        from torch_geometric.data import HeteroData  # noqa: F401
        return True
    except (ImportError, OSError):
        # OSError covers broken DLL installs on Windows
        return False


def _check_pandas() -> bool:
    try:
        import pandas as pd  # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_from_csv(path: Path) -> "pd.DataFrame":
    import pandas as pd
    if not path.exists():
        raise FileNotFoundError(f"Claims CSV not found: {path}")
    df = pd.read_csv(path)
    logger.info("Loaded %d claims from %s", len(df), path)
    return df


def _load_from_postgres(db_url: str) -> "pd.DataFrame":
    """Load claims table via asyncpg-style DSN using psycopg2 (sync)."""
    try:
        import psycopg2             # type: ignore
        import pandas as pd
        conn = psycopg2.connect(db_url, connect_timeout=10)
        df = pd.read_sql("SELECT * FROM claims LIMIT 50000", conn)
        conn.close()
        logger.info("Loaded %d claims from PostgreSQL", len(df))
        return df
    except Exception as exc:
        raise RuntimeError(f"PostgreSQL load failed: {exc}") from exc


def load_claims(
    csv_path: Optional[Path] = None,
    db_url:   Optional[str]  = None,
) -> "pd.DataFrame":
    """
    Load claims data — PostgreSQL when db_url is given, CSV otherwise.

    Falls back to data/sample_claims.csv if neither is supplied.
    """
    if db_url:
        return _load_from_postgres(db_url)

    if csv_path:
        return _load_from_csv(csv_path)

    # Default: repo sample data
    default = _REPO_ROOT / "data" / "sample_claims.csv"
    return _load_from_csv(default)


# ---------------------------------------------------------------------------
# Data validation
# ---------------------------------------------------------------------------

_REQUIRED_COLS = {"claimant_id", "claim_amount", "fraud_label"}
_OPTIONAL_COLS = {
    "days_since_policy_start", "num_prior_claims", "claim_type",
    "claim_severity", "repair_shop_id", "medical_provider_id",
}


def validate_dataframe(df: "pd.DataFrame") -> None:
    """Raise ValueError if required columns are absent."""
    missing = _REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Claims DataFrame missing required columns: {missing}")

    # Fill optional columns with sensible defaults so feature builders don't crash
    import pandas as pd
    for col in _OPTIONAL_COLS:
        if col not in df.columns:
            logger.warning("Column '%s' absent — filling with 0/NA.", col)
            if col in ("days_since_policy_start", "num_prior_claims"):
                df[col] = 0
            else:
                df[col] = pd.NA

    logger.info(
        "DataFrame: %d rows | fraud rate %.1f%% | garages %d | providers %d",
        len(df),
        df["fraud_label"].mean() * 100,
        df["repair_shop_id"].nunique() if "repair_shop_id" in df.columns else 0,
        df["medical_provider_id"].nunique() if "medical_provider_id" in df.columns else 0,
    )


# ---------------------------------------------------------------------------
# Score export
# ---------------------------------------------------------------------------

def export_scores(scores: dict, path: Path) -> None:
    """Write {claimant_id: score} dict to a JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # Add summary statistics alongside raw scores
    import statistics
    vals = list(scores.values())
    output = {
        "_meta": {
            "n_claimants":       len(vals),
            "mean_score":        round(statistics.mean(vals), 4)   if vals else 0,
            "flagged_count":     sum(1 for v in vals if v >= 0.5),
            "flag_rate_pct":     round(100 * sum(1 for v in vals if v >= 0.5) / max(len(vals), 1), 2),
        },
        "scores": scores,
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    logger.info("Scores exported to %s (%d claimants)", path, len(scores))


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(args: argparse.Namespace) -> int:
    """
    Execute the full training pipeline.

    Returns
    -------
    int  exit code (0 = success, 1 = error, 2 = PyG absent)
    """
    # ── 0. Dependency check ──────────────────────────────────────────────────
    if not _check_pandas():
        logger.error("pandas is not installed.  pip install pandas")
        return 1

    if not _check_pyg():
        logger.warning(
            "torch or torch_geometric is not installed.\n"
            "Install with:\n"
            "  pip install torch --index-url https://download.pytorch.org/whl/cpu\n"
            "  pip install torch_geometric\n"
            "GNN training skipped — ring detector will use NetworkX fallback."
        )
        return 2   # soft exit: not a CI blocker

    from src.gnn_collusion import (
        TrainingConfig,
        build_hetero_graph,
        load_model,
        score_claimants,
        write_scores_to_neo4j,
        CollusionGNNTrainer,
    )

    t_start = time.monotonic()

    # ── 1. Load data ─────────────────────────────────────────────────────────
    try:
        df = load_claims(
            csv_path=Path(args.claims_csv) if args.claims_csv else None,
            db_url=args.db_url or os.getenv("DATABASE_URL"),
        )
        validate_dataframe(df)
    except Exception as exc:
        logger.error("Data loading failed: %s", exc)
        return 1

    # ── 2. Build heterogeneous graph ─────────────────────────────────────────
    logger.info("Building heterogeneous graph…")
    data, meta = build_hetero_graph(df)

    if data is None:
        logger.error("Graph construction returned None — check PyG installation.")
        return 1

    n_c = len(meta.claimant_ids)
    n_g = len(meta.garage_ids)
    n_m = len(meta.medical_ids)

    # Count edges from HeteroData
    def _n_edges(et):
        try:
            return data[et[0], et[1], et[2]].edge_index.shape[1]
        except Exception:
            return 0

    from src.gnn_collusion import EDGE_TYPES
    total_edges = sum(_n_edges(et) for et in EDGE_TYPES)

    logger.info(
        "Graph: %d claimants | %d garages | %d medical providers | %d edges",
        n_c, n_g, n_m, total_edges,
    )

    if n_c == 0:
        logger.error("No claimant nodes found — cannot train.")
        return 1

    # ── 3. Configure and train ───────────────────────────────────────────────
    model_path = Path(args.model_out or _REPO_ROOT / "data" / "models" / "gnn_collusion.pt")

    cfg = TrainingConfig(
        hidden_dim            = args.hidden_dim,
        dropout               = args.dropout,
        lr                    = args.lr,
        weight_decay          = args.weight_decay,
        epochs                = args.epochs,
        early_stop_patience   = args.patience,
        pos_weight            = args.pos_weight,
        device                = args.device,
        model_save_path       = model_path,
    )

    if args.resume and model_path.exists():
        logger.info("Resuming from checkpoint: %s", model_path)
        model = load_model(model_path, hidden_dim=cfg.hidden_dim, dropout=cfg.dropout)
    else:
        model = None

    logger.info(
        "Training R-GCN: epochs=%d  hidden=%d  lr=%.0e  device=%s",
        cfg.epochs, cfg.hidden_dim, cfg.lr, cfg.device,
    )
    trainer = CollusionGNNTrainer(config=cfg)
    trained_model = trainer.train(data, verbose=not args.quiet)

    t_train = time.monotonic() - t_start
    logger.info("Training complete in %.1f s", t_train)

    # ── 4. Evaluate ──────────────────────────────────────────────────────────
    metrics = trainer.evaluate(trained_model, data)
    if metrics:
        logger.info(
            "Evaluation — loss: %.4f | acc: %.3f | precision: %.3f | recall: %.3f | F1: %.3f",
            metrics.get("loss",      0),
            metrics.get("accuracy",  0),
            metrics.get("precision", 0),
            metrics.get("recall",    0),
            metrics.get("f1",        0),
        )
        if args.scores_out:
            # Append metrics to a separate file
            metrics_path = Path(args.scores_out).with_suffix(".metrics.json")
            metrics_path.parent.mkdir(parents=True, exist_ok=True)
            with metrics_path.open("w") as f:
                json.dump({"training_seconds": round(t_train, 2), **metrics}, f, indent=2)
            logger.info("Metrics saved to %s", metrics_path)

    # ── 5. Score all claimants ────────────────────────────────────────────────
    logger.info("Running inference on full graph…")
    scores = score_claimants(trained_model, data, meta=meta, device=args.device)

    flagged = sum(1 for v in scores.values() if v >= 0.5)
    logger.info(
        "Scored %d claimants — %d flagged (%.1f%%)",
        len(scores), flagged, 100 * flagged / max(len(scores), 1),
    )

    # Print top-10 highest-risk claimants to stdout
    if not args.quiet and scores:
        top10 = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:10]
        print("\n── Top-10 highest collusion scores ─────────────────────")
        print(f"{'Claimant ID':<20}  {'Score':>8}  {'Flag':>6}")
        print("─" * 42)
        for cid, s in top10:
            flag = "🚨" if s >= 0.5 else "  "
            print(f"{cid:<20}  {s:>8.4f}  {flag}")
        print()

    # ── 6. Export scores to JSON ─────────────────────────────────────────────
    if args.scores_out:
        export_scores(scores, Path(args.scores_out))

    # ── 7. Write scores to Neo4j (optional) ─────────────────────────────────
    if args.write_neo4j:
        logger.info("Writing collusion scores to Neo4j…")
        n_updated = write_scores_to_neo4j(
            scores,
            uri=args.neo4j_uri      or os.getenv("NEO4J_URI"),
            user=args.neo4j_user    or os.getenv("NEO4J_USER"),
            password=args.neo4j_pwd or os.getenv("NEO4J_PASSWORD"),
            batch_size=args.neo4j_batch,
        )
        if n_updated > 0:
            logger.info("Neo4j: updated %d Claimant nodes.", n_updated)
        else:
            logger.warning(
                "Neo4j write returned 0 updates — "
                "check NEO4J_URI / credentials / server status."
            )

    total_elapsed = time.monotonic() - t_start
    logger.info("Pipeline finished in %.1f s  ✓", total_elapsed)
    return 0


# ---------------------------------------------------------------------------
# CLI argument parser
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="train_gnn_collusion",
        description="Train a heterogeneous R-GCN to detect claim-ring collusion.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Data ─────────────────────────────────────────────────────────────────
    data_g = p.add_argument_group("Data")
    data_g.add_argument(
        "--claims-csv", metavar="PATH",
        default=None,
        help="Path to claims CSV.  Defaults to data/sample_claims.csv.",
    )
    data_g.add_argument(
        "--db-url", metavar="DSN",
        default=None,
        help="PostgreSQL DSN (overrides --claims-csv and DATABASE_URL env var).",
    )

    # ── Model ────────────────────────────────────────────────────────────────
    model_g = p.add_argument_group("Model")
    model_g.add_argument("--hidden-dim",   type=int,   default=64,   help="GNN hidden dimension.")
    model_g.add_argument("--dropout",      type=float, default=0.3,  help="Dropout probability.")
    model_g.add_argument("--epochs",       type=int,   default=100,  help="Maximum training epochs.")
    model_g.add_argument("--lr",           type=float, default=1e-3, help="Adam learning rate.")
    model_g.add_argument("--weight-decay", type=float, default=1e-4, help="Adam weight decay.")
    model_g.add_argument("--patience",     type=int,   default=15,   help="Early-stopping patience.")
    model_g.add_argument("--pos-weight",   type=float, default=3.0,  help="BCEWithLogitsLoss positive class weight.")
    model_g.add_argument("--device",       default="cpu",            help="'cpu' or 'cuda'.")
    model_g.add_argument(
        "--resume", action="store_true",
        help="Resume training from an existing checkpoint at --model-out.",
    )
    model_g.add_argument(
        "--model-out", metavar="PATH",
        default=None,
        help="Where to save the trained model.  Default: data/models/gnn_collusion.pt",
    )

    # ── Output ────────────────────────────────────────────────────────────────
    out_g = p.add_argument_group("Output")
    out_g.add_argument(
        "--scores-out", metavar="PATH",
        default=None,
        help="Write claimant scores to this JSON file.",
    )
    out_g.add_argument(
        "--quiet", action="store_true",
        help="Suppress per-epoch progress and top-10 table.",
    )

    # ── Neo4j ─────────────────────────────────────────────────────────────────
    neo_g = p.add_argument_group("Neo4j")
    neo_g.add_argument(
        "--write-neo4j", action="store_true",
        help="Write collusion scores back to Neo4j after training.",
    )
    neo_g.add_argument(
        "--neo4j-uri",  default=None,
        help="Bolt URI (overrides NEO4J_URI env var).",
    )
    neo_g.add_argument(
        "--neo4j-user", default=None,
        help="Username (overrides NEO4J_USER env var).",
    )
    neo_g.add_argument(
        "--neo4j-pwd",  default=None,
        help="Password (overrides NEO4J_PASSWORD env var).",
    )
    neo_g.add_argument(
        "--neo4j-batch", type=int, default=200,
        help="Nodes per Neo4j write transaction.",
    )

    return p


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = _build_parser()
    args   = parser.parse_args()
    sys.exit(run_pipeline(args))
