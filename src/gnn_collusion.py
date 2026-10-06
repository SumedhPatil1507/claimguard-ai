"""
src/gnn_collusion.py
====================
Heterogeneous Graph Neural Network (GNN) for claim-ring collusion detection.

Architecture
------------
Graph schema  (heterogeneous, three node types, four edge types):

  Node types
  ──────────
  claimant      One node per unique claimant_id.
                Features: [claim_amount_norm, days_since_policy_start_norm,
                           num_prior_claims_norm, claim_type_enc,
                           claim_severity_enc, fraud_label]   dim = 6

  garage        One node per unique repair_shop_id.
                Features: [avg_claim_amount_norm, claim_count_norm,
                           fraud_rate, severity_high_rate]    dim = 4

  medical       One node per unique medical_provider_id.
                Features: [avg_claim_amount_norm, claim_count_norm,
                           fraud_rate, severity_high_rate]    dim = 4

  Edge types  (all edges are also added in reverse for message passing)
  ──────────
  (claimant, filed_at_garage, garage)
  (claimant, treated_by, medical)
  (garage,   co_used_by, garage)      ← two garages co-used by ≥2 claimants
  (medical,  co_used_by, medical)     ← same for medical providers

Model
-----
Relational Graph Convolutional Network (R-GCN)  using PyTorch Geometric
`HeteroConv` wrapping individual `SAGEConv` layers per relation.

  Input  → HeteroConv(SAGEConv, each relation)  → ReLU + Dropout
         → HeteroConv(SAGEConv, each relation)  → ReLU
         → claimant embeddings
         → Linear(hidden_dim → 1)  → sigmoid
         → collusion_score ∈ (0, 1)

A score > 0.5 is treated as a collusion signal.  Rings containing
high-scoring claimants are promoted to higher severity.

Graceful degradation
--------------------
Every import from torch / torch_geometric is wrapped in try/except.
If PyTorch or PyG is absent:
  - HAS_PYG = False
  - CollusionGNNModel returns None on forward()
  - CollusionGNNTrainer.train() returns an untrained-model sentinel
  - score_claimants() returns an empty dict {}
The existing NetworkX / Neo4j ring detector in graph_collusion.py
is always used as the primary backend; GNN scores are an optional
second pass that re-ranks severity.

Usage
-----
    from src.gnn_collusion import (
        build_hetero_graph,
        CollusionGNNModel,
        CollusionGNNTrainer,
        score_claimants,
    )
    data = build_hetero_graph(claims_df)
    trainer = CollusionGNNTrainer()
    model = trainer.train(data, epochs=50)
    scores = score_claimants(model, data)   # {claimant_id: float}
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional dependency guards
# ---------------------------------------------------------------------------

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except (ImportError, OSError):
    # OSError covers DLL load failures on Windows (broken torch install)
    HAS_TORCH = False
    torch = None  # type: ignore[assignment]
    nn = None     # type: ignore[assignment]
    F = None      # type: ignore[assignment]

try:
    from torch_geometric.data import HeteroData                    # type: ignore
    from torch_geometric.nn import HeteroConv, SAGEConv            # type: ignore
    HAS_PYG = True
except (ImportError, OSError):
    HAS_PYG = False
    HeteroData = None   # type: ignore[assignment,misc]
    HeteroConv = None   # type: ignore[assignment,misc]
    SAGEConv = None     # type: ignore[assignment,misc]

try:
    import pandas as pd   # type: ignore
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

# ---------------------------------------------------------------------------
# Graph schema constants
# ---------------------------------------------------------------------------

NODE_TYPES = ("claimant", "garage", "medical")

EDGE_TYPES: List[Tuple[str, str, str]] = [
    ("claimant", "filed_at_garage",  "garage"),
    ("claimant", "treated_by",       "medical"),
    ("garage",   "co_used_by",       "garage"),
    ("medical",  "co_used_by",       "medical"),
]

# Reverse edge types (added for bidirectional message passing)
REV_EDGE_TYPES: List[Tuple[str, str, str]] = [
    ("garage",   "rev_filed_at_garage", "claimant"),
    ("medical",  "rev_treated_by",      "claimant"),
    ("garage",   "rev_co_used_by",      "garage"),
    ("medical",  "rev_co_used_by",      "medical"),
]

ALL_EDGE_TYPES = EDGE_TYPES + REV_EDGE_TYPES

# Feature dimensions per node type
CLAIMANT_FEAT_DIM = 6
GARAGE_FEAT_DIM   = 4
MEDICAL_FEAT_DIM  = 4

NODE_FEAT_DIMS: Dict[str, int] = {
    "claimant": CLAIMANT_FEAT_DIM,
    "garage":   GARAGE_FEAT_DIM,
    "medical":  MEDICAL_FEAT_DIM,
}

# Threshold above which a claimant is flagged as colluding
COLLUSION_THRESHOLD = 0.5

# Minimum co-usage count for a garage/medical co_used_by edge
_CO_USAGE_MIN = 2

# ---------------------------------------------------------------------------
# Encoding helpers
# ---------------------------------------------------------------------------

_CLAIM_TYPE_MAP   = {"auto": 0, "health": 1, "property": 2, "life": 3, "motor": 0}
_SEVERITY_MAP     = {"low": 0, "medium": 1, "high": 2}


def _safe_enc(mapping: dict, value: Any, default: int = 0) -> float:
    if value is None:
        return float(default)
    return float(mapping.get(str(value).lower(), default))


def _normalise(arr: np.ndarray) -> np.ndarray:
    """Min-max normalise; returns zeros if range is zero."""
    mn, mx = arr.min(), arr.max()
    if mx == mn:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - mn) / (mx - mn)).astype(np.float32)


# ---------------------------------------------------------------------------
# Node feature builders
# ---------------------------------------------------------------------------

def build_claimant_features(df: "pd.DataFrame") -> Tuple[np.ndarray, List[str]]:
    """
    Build claimant node feature matrix.

    Returns
    -------
    feats : np.ndarray  shape (N_claimants, CLAIMANT_FEAT_DIM)
    ids   : list of claimant_id strings in row order
    """
    grouped = df.groupby("claimant_id", sort=False)
    ids = list(grouped.groups.keys())
    rows = []
    for cid in ids:
        sub = grouped.get_group(cid)
        row0 = sub.iloc[0]
        rows.append({
            "claim_amount":              float(sub["claim_amount"].mean()) if "claim_amount" in sub else 0.0,
            "days_since_policy_start":   float(row0.get("days_since_policy_start", 0) or 0),
            "num_prior_claims":          float(row0.get("num_prior_claims", 0) or 0),
            "claim_type_enc":            _safe_enc(_CLAIM_TYPE_MAP, row0.get("claim_type", "auto")),
            "claim_severity_enc":        _safe_enc(_SEVERITY_MAP,   row0.get("claim_severity", "low")),
            "fraud_label":               float(row0.get("fraud_label", 0) or 0),
        })
    arr = np.array([[
        r["claim_amount"],
        r["days_since_policy_start"],
        r["num_prior_claims"],
        r["claim_type_enc"],
        r["claim_severity_enc"],
        r["fraud_label"],
    ] for r in rows], dtype=np.float32)
    # Normalise continuous columns only (first three)
    if len(arr) > 1:
        arr[:, :3] = np.column_stack([_normalise(arr[:, i]) for i in range(3)])
    return arr, ids


def _build_entity_features(df: "pd.DataFrame", id_col: str) -> Tuple[np.ndarray, List[str]]:
    """
    Build shared feature matrix for garage / medical provider node types.

    Features: [avg_claim_amount_norm, claim_count_norm, fraud_rate, severity_high_rate]
    """
    valid = df[df[id_col].notna() & (df[id_col].astype(str) != "nan") & (df[id_col].astype(str) != "")]
    if valid.empty:
        return np.zeros((0, GARAGE_FEAT_DIM), dtype=np.float32), []

    grouped = valid.groupby(id_col, sort=False)
    ids = list(grouped.groups.keys())
    rows = []
    for eid in ids:
        sub = grouped.get_group(eid)
        n = len(sub)
        rows.append([
            float(sub["claim_amount"].mean()) if "claim_amount" in sub else 0.0,
            float(n),
            float(sub["fraud_label"].mean())  if "fraud_label" in sub else 0.0,
            float((sub.get("claim_severity", pd.Series([])) == "high").sum()) / max(n, 1),
        ])
    arr = np.array(rows, dtype=np.float32)
    if len(arr) > 1:
        arr[:, :2] = np.column_stack([_normalise(arr[:, i]) for i in range(2)])
    return arr, [str(i) for i in ids]


def build_garage_features(df: "pd.DataFrame") -> Tuple[np.ndarray, List[str]]:
    return _build_entity_features(df, "repair_shop_id")


def build_medical_features(df: "pd.DataFrame") -> Tuple[np.ndarray, List[str]]:
    return _build_entity_features(df, "medical_provider_id")


# ---------------------------------------------------------------------------
# Edge index builders
# ---------------------------------------------------------------------------

def _build_bipartite_edges(
    df: "pd.DataFrame",
    src_ids: List[str],
    dst_ids: List[str],
    src_col: str,
    dst_col: str,
) -> np.ndarray:
    """
    Return a (2, E) int64 edge-index array for bipartite edges
    src_col → dst_col using the provided id lists as index maps.
    """
    src_idx = {v: i for i, v in enumerate(src_ids)}
    dst_idx = {v: i for i, v in enumerate(dst_ids)}
    srcs, dsts = [], []
    for _, row in df.iterrows():
        s = str(row.get(src_col, ""))
        d = str(row.get(dst_col, ""))
        if s in src_idx and d in dst_idx:
            srcs.append(src_idx[s])
            dsts.append(dst_idx[d])
    if not srcs:
        return np.zeros((2, 0), dtype=np.int64)
    return np.array([srcs, dsts], dtype=np.int64)


def _build_co_usage_edges(
    df: "pd.DataFrame",
    entity_ids: List[str],
    entity_col: str,
    claimant_col: str = "claimant_id",
    min_co_usage: int = _CO_USAGE_MIN,
) -> np.ndarray:
    """
    Build co-usage edges between two entity nodes that were both used by
    at least *min_co_usage* common claimants.

    Returns (2, E) int64 edge-index (undirected — each edge appears once).
    """
    if not entity_ids:
        return np.zeros((2, 0), dtype=np.int64)

    eid_idx = {v: i for i, v in enumerate(entity_ids)}
    valid = df[df[entity_col].notna() & (df[entity_col].astype(str) != "nan")]
    # Map: entity_id → set of claimant_ids
    entity_claimants: Dict[str, set] = {}
    for _, row in valid.iterrows():
        eid = str(row.get(entity_col, ""))
        cid = str(row.get(claimant_col, ""))
        if eid in eid_idx:
            entity_claimants.setdefault(eid, set()).add(cid)

    eids = list(entity_claimants.keys())
    srcs, dsts = [], []
    for i in range(len(eids)):
        for j in range(i + 1, len(eids)):
            shared = len(entity_claimants[eids[i]] & entity_claimants[eids[j]])
            if shared >= min_co_usage:
                si, di = eid_idx[eids[i]], eid_idx[eids[j]]
                srcs += [si, di]   # undirected: add both directions
                dsts += [di, si]
    if not srcs:
        return np.zeros((2, 0), dtype=np.int64)
    return np.array([srcs, dsts], dtype=np.int64)


# ---------------------------------------------------------------------------
# HeteroData graph builder
# ---------------------------------------------------------------------------

@dataclass
class GraphMetadata:
    """Stores id-to-index maps so callers can map scores back to entity ids."""
    claimant_ids: List[str] = field(default_factory=list)
    garage_ids:   List[str] = field(default_factory=list)
    medical_ids:  List[str] = field(default_factory=list)


def build_hetero_graph(
    df: "pd.DataFrame",
) -> Tuple[Optional[Any], GraphMetadata]:
    """
    Construct a PyG HeteroData object from a claims DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns from the ClaimGuard synthetic dataset:
        claimant_id, claim_amount, days_since_policy_start,
        num_prior_claims, claim_type, claim_severity, fraud_label,
        repair_shop_id (optional), medical_provider_id (optional).

    Returns
    -------
    (HeteroData | None, GraphMetadata)
        Returns (None, metadata) when PyG is not installed.
    """
    # Always build metadata (used even without PyG)
    c_feats, c_ids = build_claimant_features(df)
    g_feats, g_ids = build_garage_features(df)
    m_feats, m_ids = build_medical_features(df)
    meta = GraphMetadata(claimant_ids=c_ids, garage_ids=g_ids, medical_ids=m_ids)

    if not HAS_PYG or not HAS_TORCH:
        logger.warning("gnn_collusion: torch_geometric not installed; returning None graph.")
        return None, meta

    data = HeteroData()

    # ── Node features ────────────────────────────────────────────────────────
    data["claimant"].x = torch.tensor(c_feats, dtype=torch.float)
    data["claimant"].y = torch.tensor(
        c_feats[:, -1].astype(np.int64), dtype=torch.long
    )  # fraud_label as node-level supervision signal

    data["garage"].x  = (
        torch.tensor(g_feats, dtype=torch.float) if len(g_feats) > 0
        else torch.zeros((0, GARAGE_FEAT_DIM), dtype=torch.float)
    )
    data["medical"].x = (
        torch.tensor(m_feats, dtype=torch.float) if len(m_feats) > 0
        else torch.zeros((0, MEDICAL_FEAT_DIM), dtype=torch.float)
    )

    # ── Edge indices ─────────────────────────────────────────────────────────
    valid_garage  = df[df["repair_shop_id"].notna() & (df["repair_shop_id"].astype(str) != "nan")]
    valid_medical = df[df["medical_provider_id"].notna() & (df["medical_provider_id"].astype(str) != "nan")]

    # claimant → garage
    cg_ei = _build_bipartite_edges(valid_garage, c_ids, g_ids, "claimant_id", "repair_shop_id")
    data["claimant", "filed_at_garage", "garage"].edge_index = torch.tensor(cg_ei, dtype=torch.long)

    # claimant → medical
    cm_ei = _build_bipartite_edges(valid_medical, c_ids, m_ids, "claimant_id", "medical_provider_id")
    data["claimant", "treated_by", "medical"].edge_index = torch.tensor(cm_ei, dtype=torch.long)

    # garage co-usage
    gg_ei = _build_co_usage_edges(valid_garage, g_ids, "repair_shop_id")
    data["garage", "co_used_by", "garage"].edge_index = torch.tensor(gg_ei, dtype=torch.long)

    # medical co-usage
    mm_ei = _build_co_usage_edges(valid_medical, m_ids, "medical_provider_id")
    data["medical", "co_used_by", "medical"].edge_index = torch.tensor(mm_ei, dtype=torch.long)

    # ── Reverse edges (bidirectional message passing) ─────────────────────────
    data["garage",   "rev_filed_at_garage", "claimant"].edge_index = torch.tensor(cg_ei[[1, 0]], dtype=torch.long) if cg_ei.shape[1] > 0 else torch.zeros((2, 0), dtype=torch.long)
    data["medical",  "rev_treated_by",      "claimant"].edge_index = torch.tensor(cm_ei[[1, 0]], dtype=torch.long) if cm_ei.shape[1] > 0 else torch.zeros((2, 0), dtype=torch.long)
    data["garage",   "rev_co_used_by",      "garage"  ].edge_index = torch.tensor(gg_ei[[1, 0]], dtype=torch.long) if gg_ei.shape[1] > 0 else torch.zeros((2, 0), dtype=torch.long)
    data["medical",  "rev_co_used_by",      "medical" ].edge_index = torch.tensor(mm_ei[[1, 0]], dtype=torch.long) if mm_ei.shape[1] > 0 else torch.zeros((2, 0), dtype=torch.long)

    return data, meta


# ---------------------------------------------------------------------------
# R-GCN model
# ---------------------------------------------------------------------------

if HAS_PYG and HAS_TORCH:

    class CollusionGNNModel(nn.Module):
        """
        Relational Graph Convolutional Network for claimant collusion scoring.

        Two HeteroConv layers (SAGEConv per relation) aggregate neighbourhood
        information across all edge types.  The claimant embeddings from the
        second layer are passed through a linear head to produce a collusion
        score in (0, 1).

        Parameters
        ----------
        hidden_dim   : width of GNN hidden layers (default 64)
        dropout      : dropout probability applied after first conv (default 0.3)
        """

        def __init__(self, hidden_dim: int = 64, dropout: float = 0.3) -> None:
            super().__init__()
            self.dropout_p = dropout

            # ── Input projections (per node type → hidden_dim) ───────────────
            self.proj_claimant = nn.Linear(CLAIMANT_FEAT_DIM, hidden_dim)
            self.proj_garage   = nn.Linear(GARAGE_FEAT_DIM,   hidden_dim)
            self.proj_medical  = nn.Linear(MEDICAL_FEAT_DIM,  hidden_dim)

            # ── Layer 1: HeteroConv(SAGEConv) ────────────────────────────────
            self.conv1 = HeteroConv(
                {
                    et: SAGEConv((hidden_dim, hidden_dim), hidden_dim)
                    for et in ALL_EDGE_TYPES
                },
                aggr="mean",
            )

            # ── Layer 2: HeteroConv(SAGEConv) ────────────────────────────────
            self.conv2 = HeteroConv(
                {
                    et: SAGEConv((hidden_dim, hidden_dim), hidden_dim)
                    for et in ALL_EDGE_TYPES
                },
                aggr="mean",
            )

            # ── Output head (claimant nodes only) ─────────────────────────────
            self.head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1),
            )

        def forward(self, x_dict: dict, edge_index_dict: dict) -> dict:
            """
            Forward pass.

            Parameters
            ----------
            x_dict          : {node_type: Tensor(N, feat_dim)}
            edge_index_dict : {edge_type_tuple: Tensor(2, E)}

            Returns
            -------
            dict with key 'claimant' → Tensor(N_claimants, 1) of logits.
            """
            # Project all node types to hidden_dim
            h: dict = {
                "claimant": F.relu(self.proj_claimant(x_dict["claimant"])),
                "garage":   F.relu(self.proj_garage(x_dict["garage"])),
                "medical":  F.relu(self.proj_medical(x_dict["medical"])),
            }

            # Layer 1
            h = self.conv1(h, edge_index_dict)
            h = {k: F.relu(v) for k, v in h.items()}
            h = {k: F.dropout(v, p=self.dropout_p, training=self.training) for k, v in h.items()}

            # Layer 2
            h = self.conv2(h, edge_index_dict)
            h = {k: F.relu(v) for k, v in h.items()}

            # Collusion head over claimant nodes
            logits = self.head(h["claimant"])   # (N_claimants, 1)
            return {"claimant": logits}

        def predict_scores(self, x_dict: dict, edge_index_dict: dict) -> torch.Tensor:
            """Sigmoid-activated collusion scores ∈ (0, 1)."""
            self.eval()
            with torch.no_grad():
                logits = self.forward(x_dict, edge_index_dict)["claimant"]
                return torch.sigmoid(logits).squeeze(1)   # (N_claimants,)

else:
    # ── Stub when torch / PyG are absent ─────────────────────────────────────
    class CollusionGNNModel:   # type: ignore[no-redef]
        """No-op placeholder used when torch_geometric is not installed."""

        def __init__(self, hidden_dim: int = 64, dropout: float = 0.3) -> None:
            logger.warning(
                "gnn_collusion: CollusionGNNModel is a stub — "
                "install torch and torch_geometric to enable GNN inference."
            )

        def forward(self, *args, **kwargs):
            return None

        def predict_scores(self, *args, **kwargs):
            return None

        def train(self, mode: bool = True):  # noqa: A003
            return self

        def eval(self):
            return self

        def parameters(self):
            return iter([])

        def state_dict(self):
            return {}

        def load_state_dict(self, *args, **kwargs):
            pass


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------

@dataclass
class TrainingConfig:
    """Hyperparameters for the R-GCN training loop."""
    hidden_dim:       int   = 64
    dropout:          float = 0.3
    lr:               float = 1e-3
    weight_decay:     float = 1e-4
    epochs:           int   = 100
    early_stop_patience: int = 15
    pos_weight:       float = 3.0    # upweight fraud positives (class imbalance)
    device:           str   = "cpu"  # 'cuda' if GPU available
    model_save_path:  Path  = Path("data/models/gnn_collusion.pt")


class CollusionGNNTrainer:
    """
    Trains a CollusionGNNModel on a HeteroData graph.

    The training objective is binary cross-entropy on the fraud_label of
    claimant nodes, using the GNN-derived collusion score as the prediction.
    This means the model learns that claimants embedded near fraudulent
    repair shops / medical providers are more likely to collude.

    Parameters
    ----------
    config : TrainingConfig  (optional — defaults are sensible)
    """

    def __init__(self, config: Optional[TrainingConfig] = None) -> None:
        self.config = config or TrainingConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(
        self,
        data: Any,
        verbose: bool = True,
    ) -> CollusionGNNModel:
        """
        Train the R-GCN model and return it.

        Parameters
        ----------
        data    : HeteroData built by build_hetero_graph()
        verbose : print per-epoch loss when True

        Returns
        -------
        Trained CollusionGNNModel.  When PyG is absent, returns the no-op stub.
        """
        if not HAS_PYG or not HAS_TORCH or data is None:
            logger.warning("gnn_collusion: training skipped (PyG unavailable or no data).")
            return CollusionGNNModel()

        cfg  = self.config
        dev  = torch.device(cfg.device)
        data = data.to(dev)

        model = CollusionGNNModel(hidden_dim=cfg.hidden_dim, dropout=cfg.dropout).to(dev)

        # Positive-class weight for imbalanced fraud labels
        pos_w = torch.tensor([cfg.pos_weight], dtype=torch.float, device=dev)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)

        optimiser = torch.optim.Adam(
            model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimiser, patience=5, factor=0.5, min_lr=1e-5
        )

        x_dict         = {k: data[k].x          for k in NODE_TYPES if hasattr(data[k], "x")}
        edge_index_dict = {
            et: data[et[0], et[1], et[2]].edge_index
            for et in ALL_EDGE_TYPES
            if hasattr(data[et[0], et[1], et[2]], "edge_index")
        }
        y = data["claimant"].y.float()

        best_loss     = float("inf")
        patience_cnt  = 0
        best_state    = None

        for epoch in range(1, cfg.epochs + 1):
            model.train()
            optimiser.zero_grad()

            logits = model(x_dict, edge_index_dict)["claimant"].squeeze(1)
            loss   = criterion(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimiser.step()
            scheduler.step(loss)

            loss_val = loss.item()
            if verbose and (epoch % 10 == 0 or epoch == 1):
                acc = _binary_accuracy(torch.sigmoid(logits), y)
                logger.info(
                    "gnn_collusion epoch %d/%d  loss=%.4f  acc=%.3f",
                    epoch, cfg.epochs, loss_val, acc,
                )

            # Early stopping
            if loss_val < best_loss - 1e-4:
                best_loss    = loss_val
                patience_cnt = 0
                best_state   = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                patience_cnt += 1
                if patience_cnt >= cfg.early_stop_patience:
                    logger.info("gnn_collusion: early stopping at epoch %d.", epoch)
                    break

        # Restore best weights
        if best_state is not None:
            model.load_state_dict(best_state)

        # Persist model
        _save_model(model, cfg.model_save_path)
        logger.info("gnn_collusion: model saved to %s.", cfg.model_save_path)

        return model

    def evaluate(
        self,
        model: CollusionGNNModel,
        data: Any,
    ) -> Dict[str, float]:
        """
        Evaluate model on full graph.

        Returns
        -------
        dict with keys: loss, accuracy, precision, recall, f1
        """
        if not HAS_PYG or not HAS_TORCH or data is None:
            return {}

        cfg = self.config
        dev = torch.device(cfg.device)
        data = data.to(dev)

        x_dict = {k: data[k].x for k in NODE_TYPES if hasattr(data[k], "x")}
        edge_index_dict = {
            et: data[et[0], et[1], et[2]].edge_index
            for et in ALL_EDGE_TYPES
            if hasattr(data[et[0], et[1], et[2]], "edge_index")
        }
        y = data["claimant"].y.float()

        pos_w = torch.tensor([cfg.pos_weight], dtype=torch.float, device=dev)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)

        model.eval()
        with torch.no_grad():
            logits = model(x_dict, edge_index_dict)["claimant"].squeeze(1)
            loss   = criterion(logits, y).item()
            probs  = torch.sigmoid(logits)
            preds  = (probs >= COLLUSION_THRESHOLD).float()

        tp = ((preds == 1) & (y == 1)).sum().item()
        fp = ((preds == 1) & (y == 0)).sum().item()
        fn = ((preds == 0) & (y == 1)).sum().item()
        tn = ((preds == 0) & (y == 0)).sum().item()

        acc       = (tp + tn) / max(tp + fp + fn + tn, 1)
        precision = tp / max(tp + fp, 1)
        recall    = tp / max(tp + fn, 1)
        f1        = 2 * precision * recall / max(precision + recall, 1e-8)

        return {
            "loss":      round(loss, 4),
            "accuracy":  round(acc, 4),
            "precision": round(precision, 4),
            "recall":    round(recall, 4),
            "f1":        round(f1, 4),
        }


# ---------------------------------------------------------------------------
# Inference helper
# ---------------------------------------------------------------------------

def score_claimants(
    model: CollusionGNNModel,
    data: Any,
    meta: Optional[GraphMetadata] = None,
    device: str = "cpu",
) -> Dict[str, float]:
    """
    Run inference and return a mapping of claimant_id → collusion_score.

    Parameters
    ----------
    model  : trained CollusionGNNModel (or the no-op stub)
    data   : HeteroData from build_hetero_graph()
    meta   : GraphMetadata — if None, keys are integer indices
    device : 'cpu' or 'cuda'

    Returns
    -------
    dict  {claimant_id: score ∈ (0, 1)}
    Empty dict when PyG is absent or data is None.
    """
    if not HAS_PYG or not HAS_TORCH or data is None:
        return {}

    try:
        dev  = torch.device(device)
        data = data.to(dev)
        x_dict = {k: data[k].x for k in NODE_TYPES if hasattr(data[k], "x")}
        edge_index_dict = {
            et: data[et[0], et[1], et[2]].edge_index
            for et in ALL_EDGE_TYPES
            if hasattr(data[et[0], et[1], et[2]], "edge_index")
        }
        scores = model.predict_scores(x_dict, edge_index_dict)
        if scores is None:
            return {}

        scores_np = scores.cpu().numpy().tolist()
        if meta and meta.claimant_ids:
            return {cid: float(s) for cid, s in zip(meta.claimant_ids, scores_np)}
        return {str(i): float(s) for i, s in enumerate(scores_np)}

    except Exception as exc:
        logger.error("gnn_collusion.score_claimants: %s", exc)
        return {}


# ---------------------------------------------------------------------------
# Model persistence
# ---------------------------------------------------------------------------

def _save_model(model: CollusionGNNModel, path: Path) -> None:
    if not HAS_TORCH:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), path)
    except Exception as exc:
        logger.warning("gnn_collusion: could not save model — %s", exc)


def load_model(
    path: Path,
    hidden_dim: int = 64,
    dropout: float = 0.3,
) -> CollusionGNNModel:
    """
    Load a saved CollusionGNNModel from *path*.

    Returns a fresh (untrained) model if *path* does not exist or PyG is absent.
    """
    model = CollusionGNNModel(hidden_dim=hidden_dim, dropout=dropout)
    if not HAS_TORCH:
        return model
    if not Path(path).exists():
        logger.warning("gnn_collusion.load_model: %s not found; returning untrained model.", path)
        return model
    try:
        state = torch.load(path, map_location="cpu")
        model.load_state_dict(state)
        logger.info("gnn_collusion: loaded model from %s.", path)
    except Exception as exc:
        logger.warning("gnn_collusion: could not load model from %s — %s", path, exc)
    return model


# ---------------------------------------------------------------------------
# Neo4j score writer
# ---------------------------------------------------------------------------

def write_scores_to_neo4j(
    scores: Dict[str, float],
    uri:      Optional[str] = None,
    user:     Optional[str] = None,
    password: Optional[str] = None,
    batch_size: int = 200,
) -> int:
    """
    Write GNN collusion scores back to Neo4j as a property on Claimant nodes.

    Creates or updates a ``collusionScore`` float property and a boolean
    ``gnnCollusionFlag`` property on each ``:Claimant`` node.

    Parameters
    ----------
    scores     : {claimant_id: score} dict from score_claimants()
    uri        : Bolt URI (defaults to NEO4J_URI env var)
    user       : username (defaults to NEO4J_USER env var)
    password   : password (defaults to NEO4J_PASSWORD env var)
    batch_size : number of nodes to update per transaction

    Returns
    -------
    int — number of nodes successfully updated.
        Returns 0 when neo4j package is absent or the server is unreachable.
    """
    try:
        from neo4j import GraphDatabase as _GDB   # type: ignore
    except ImportError:
        logger.warning("gnn_collusion.write_scores_to_neo4j: neo4j package not installed.")
        return 0

    _uri  = uri      or os.getenv("NEO4J_URI",      "bolt://localhost:7687")
    _user = user     or os.getenv("NEO4J_USER",     "neo4j")
    _pwd  = password or os.getenv("NEO4J_PASSWORD", "neo4j")

    if not scores:
        return 0

    try:
        driver = _GDB.driver(_uri, auth=(_user, _pwd))
        driver.verify_connectivity()
    except Exception as exc:
        logger.warning("gnn_collusion.write_scores_to_neo4j: cannot connect — %s", exc)
        return 0

    items = list(scores.items())
    total_updated = 0

    # Cypher: MERGE on claimant_id, SET score + flag
    _CYPHER = (
        "UNWIND $rows AS row "
        "MERGE (c:Claimant {claimantId: row.cid}) "
        "SET c.collusionScore    = row.score, "
        "    c.gnnCollusionFlag  = row.flag, "
        "    c.scoreUpdatedAt    = datetime() "
        "RETURN count(c) AS updated"
    )

    try:
        with driver.session() as session:
            for start in range(0, len(items), batch_size):
                batch = items[start : start + batch_size]
                rows  = [
                    {
                        "cid":   cid,
                        "score": round(float(score), 6),
                        "flag":  float(score) >= COLLUSION_THRESHOLD,
                    }
                    for cid, score in batch
                ]
                result = session.run(_CYPHER, rows=rows)
                record = result.single()
                total_updated += record["updated"] if record else len(batch)
    except Exception as exc:
        logger.error("gnn_collusion.write_scores_to_neo4j: write failed — %s", exc)
    finally:
        driver.close()

    logger.info(
        "gnn_collusion.write_scores_to_neo4j: updated %d/%d Claimant nodes.",
        total_updated,
        len(items),
    )
    return total_updated


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _binary_accuracy(probs: "torch.Tensor", targets: "torch.Tensor") -> float:
    preds = (probs >= COLLUSION_THRESHOLD).float()
    return (preds == targets).float().mean().item()
