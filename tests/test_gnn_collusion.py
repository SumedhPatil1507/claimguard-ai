"""
tests/test_gnn_collusion.py
============================
Unit tests for the heterogeneous GNN collusion detection stack:

  src/gnn_collusion.py      — graph schema, feature builders, R-GCN model,
                               trainer, inference, Neo4j writer
  src/graph_collusion.py    — two-stage detector (structural + GNN)
  scripts/train_gnn_collusion.py — CLI pipeline helpers

Design constraints
------------------
* No live Neo4j, PostgreSQL, or Redis — all external I/O is mocked.
* No GPU — all torch operations run on CPU.
* Tests pass whether or not torch / torch_geometric are installed:
  - When PyG IS present  → test real forward passes and training.
  - When PyG is absent   → test graceful stub behaviour.
* Synthetic DataFrames are constructed inline — no CSV files required.

Test groups
-----------
A  Graph schema constants
B  Node feature builders
C  Edge index builders
D  build_hetero_graph() — full graph construction
E  CollusionGNNModel   — forward pass, predict_scores (PyG required)
F  CollusionGNNTrainer — train(), evaluate()          (PyG required)
G  score_claimants()   — inference helper
H  write_scores_to_neo4j() — batched Cypher MERGE (mocked)
I  load_model() / _save_model()
J  GraphCollusionDetector — two-stage pipeline
K  _apply_gnn_scores     — severity upgrade rules
L  CLI pipeline helpers  (scripts/train_gnn_collusion.py)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List
from unittest.mock import MagicMock, call, patch

import numpy as np
import pandas as pd
import pytest

# Repo root on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

# ---------------------------------------------------------------------------
# Availability flags
# ---------------------------------------------------------------------------

try:
    import torch
    from torch_geometric.data import HeteroData
    HAS_PYG = True
except (ImportError, OSError):
    # OSError covers broken DLL installs on Windows
    HAS_PYG = False

# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------

def _make_claims(n: int = 20, seed: int = 42) -> pd.DataFrame:
    """
    Build a minimal synthetic claims DataFrame with the columns that
    feature builders, edge builders, and the ring detector all expect.
    """
    rng = np.random.default_rng(seed)

    n_claimants  = max(5, n // 4)
    n_garages    = max(2, n // 8)
    n_providers  = max(2, n // 8)

    claimant_ids = [f"CLT{i:04d}" for i in range(n_claimants)]
    garage_ids   = [f"RS{i:03d}"  for i in range(n_garages)]
    provider_ids = [f"MP{i:03d}"  for i in range(n_providers)]

    rows = []
    for i in range(n):
        rows.append({
            "claim_id":                f"CLM{i:05d}",
            "claimant_id":             claimant_ids[i % n_claimants],
            "claim_amount":            float(rng.integers(10_000, 500_000)),
            "days_since_policy_start": int(rng.integers(0, 1000)),
            "num_prior_claims":        int(rng.integers(0, 6)),
            "claim_type":              rng.choice(["motor", "health", "property"]),
            "claim_severity":          rng.choice(["low", "medium", "high"]),
            "fraud_label":             int(rng.random() < 0.20),
            "repair_shop_id":          garage_ids[i % n_garages] if rng.random() > 0.2 else None,
            "medical_provider_id":     provider_ids[i % n_providers] if rng.random() > 0.5 else None,
            "witness_ids":             "",
        })
    return pd.DataFrame(rows)


def _make_small_claims() -> pd.DataFrame:
    """Minimal 6-row DataFrame guaranteed to produce collusion rings."""
    return pd.DataFrame([
        # Three claimants all using the same garage RS001 (ring trigger)
        {"claim_id": "C1", "claimant_id": "CLT01", "claim_amount": 100_000, "days_since_policy_start": 30,
         "num_prior_claims": 1, "claim_type": "motor", "claim_severity": "high", "fraud_label": 1,
         "repair_shop_id": "RS001", "medical_provider_id": "MP001", "witness_ids": ""},
        {"claim_id": "C2", "claimant_id": "CLT02", "claim_amount": 120_000, "days_since_policy_start": 45,
         "num_prior_claims": 2, "claim_type": "motor", "claim_severity": "high", "fraud_label": 1,
         "repair_shop_id": "RS001", "medical_provider_id": "MP001", "witness_ids": ""},
        {"claim_id": "C3", "claimant_id": "CLT03", "claim_amount": 90_000, "days_since_policy_start": 20,
         "num_prior_claims": 3, "claim_type": "motor", "claim_severity": "medium", "fraud_label": 1,
         "repair_shop_id": "RS001", "medical_provider_id": "MP001", "witness_ids": ""},
        # Two claimants sharing RS001 again (adds to ring count)
        {"claim_id": "C4", "claimant_id": "CLT04", "claim_amount": 80_000, "days_since_policy_start": 60,
         "num_prior_claims": 0, "claim_type": "health", "claim_severity": "low", "fraud_label": 0,
         "repair_shop_id": "RS001", "medical_provider_id": None, "witness_ids": ""},
        {"claim_id": "C5", "claimant_id": "CLT05", "claim_amount": 50_000, "days_since_policy_start": 200,
         "num_prior_claims": 0, "claim_type": "health", "claim_severity": "low", "fraud_label": 0,
         "repair_shop_id": "RS002", "medical_provider_id": None, "witness_ids": ""},
        # Independent claim
        {"claim_id": "C6", "claimant_id": "CLT06", "claim_amount": 30_000, "days_since_policy_start": 500,
         "num_prior_claims": 0, "claim_type": "property", "claim_severity": "low", "fraud_label": 0,
         "repair_shop_id": None, "medical_provider_id": None, "witness_ids": ""},
    ])


# ===========================================================================
# A — Graph schema constants
# ===========================================================================

class TestGraphSchemaConstants:
    def test_node_types_tuple(self) -> None:
        from src.gnn_collusion import NODE_TYPES
        assert set(NODE_TYPES) == {"claimant", "garage", "medical"}

    def test_edge_types_count(self) -> None:
        from src.gnn_collusion import EDGE_TYPES
        assert len(EDGE_TYPES) == 4

    def test_all_edge_types_include_reverse(self) -> None:
        from src.gnn_collusion import ALL_EDGE_TYPES, EDGE_TYPES, REV_EDGE_TYPES
        assert len(ALL_EDGE_TYPES) == len(EDGE_TYPES) + len(REV_EDGE_TYPES) == 8

    def test_feature_dims_correct(self) -> None:
        from src.gnn_collusion import (
            CLAIMANT_FEAT_DIM, GARAGE_FEAT_DIM, MEDICAL_FEAT_DIM, NODE_FEAT_DIMS
        )
        assert CLAIMANT_FEAT_DIM == 6
        assert GARAGE_FEAT_DIM   == 4
        assert MEDICAL_FEAT_DIM  == 4
        assert NODE_FEAT_DIMS["claimant"] == CLAIMANT_FEAT_DIM
        assert NODE_FEAT_DIMS["garage"]   == GARAGE_FEAT_DIM
        assert NODE_FEAT_DIMS["medical"]  == MEDICAL_FEAT_DIM

    def test_collusion_threshold_in_01(self) -> None:
        from src.gnn_collusion import COLLUSION_THRESHOLD
        assert 0.0 < COLLUSION_THRESHOLD < 1.0


# ===========================================================================
# B — Node feature builders
# ===========================================================================

class TestNodeFeatureBuilders:
    def test_claimant_features_shape(self) -> None:
        from src.gnn_collusion import build_claimant_features, CLAIMANT_FEAT_DIM
        df = _make_claims(20)
        feats, ids = build_claimant_features(df)
        assert feats.shape == (len(ids), CLAIMANT_FEAT_DIM)
        assert len(ids) == df["claimant_id"].nunique()

    def test_claimant_features_no_nan(self) -> None:
        from src.gnn_collusion import build_claimant_features
        df = _make_claims(30)
        feats, _ = build_claimant_features(df)
        assert not np.any(np.isnan(feats)), "NaN in claimant features"
        assert not np.any(np.isinf(feats)), "Inf in claimant features"

    def test_claimant_continuous_cols_normalised(self) -> None:
        """First three continuous columns should be in [0, 1] after normalisation."""
        from src.gnn_collusion import build_claimant_features
        df = _make_claims(50)
        feats, _ = build_claimant_features(df)
        for col in range(3):
            assert feats[:, col].min() >= -1e-6
            assert feats[:, col].max() <= 1.0 + 1e-6

    def test_claimant_fraud_label_binary(self) -> None:
        from src.gnn_collusion import build_claimant_features
        df = _make_claims(20)
        feats, _ = build_claimant_features(df)
        fraud_col = feats[:, -1]
        assert set(np.unique(fraud_col)).issubset({0.0, 1.0})

    def test_garage_features_shape(self) -> None:
        from src.gnn_collusion import build_garage_features, GARAGE_FEAT_DIM
        df = _make_claims(30)
        feats, ids = build_garage_features(df)
        n_garages = df["repair_shop_id"].dropna().nunique()
        assert feats.shape == (n_garages, GARAGE_FEAT_DIM)
        assert len(ids) == n_garages

    def test_garage_features_no_nan(self) -> None:
        from src.gnn_collusion import build_garage_features
        df = _make_claims(30)
        feats, _ = build_garage_features(df)
        assert not np.any(np.isnan(feats))

    def test_medical_features_empty_when_all_null(self) -> None:
        from src.gnn_collusion import build_medical_features
        df = _make_claims(10)
        df["medical_provider_id"] = None
        feats, ids = build_medical_features(df)
        assert len(ids) == 0
        assert feats.shape[0] == 0

    def test_feature_ids_are_strings(self) -> None:
        from src.gnn_collusion import build_garage_features
        df = _make_claims(20)
        _, ids = build_garage_features(df)
        assert all(isinstance(i, str) for i in ids)


# ===========================================================================
# C — Edge index builders
# ===========================================================================

class TestEdgeIndexBuilders:
    def test_bipartite_edge_shape(self) -> None:
        from src.gnn_collusion import (
            build_claimant_features, build_garage_features,
            _build_bipartite_edges,
        )
        df = _make_claims(30)
        _, c_ids = build_claimant_features(df)
        _, g_ids = build_garage_features(df)
        valid = df[df["repair_shop_id"].notna()]
        ei = _build_bipartite_edges(valid, c_ids, g_ids, "claimant_id", "repair_shop_id")
        assert ei.shape[0] == 2
        assert ei.dtype == np.int64

    def test_bipartite_indices_in_bounds(self) -> None:
        from src.gnn_collusion import (
            build_claimant_features, build_garage_features,
            _build_bipartite_edges,
        )
        df = _make_claims(40)
        _, c_ids = build_claimant_features(df)
        _, g_ids = build_garage_features(df)
        valid = df[df["repair_shop_id"].notna()]
        ei = _build_bipartite_edges(valid, c_ids, g_ids, "claimant_id", "repair_shop_id")
        if ei.shape[1] > 0:
            assert ei[0].max() < len(c_ids)
            assert ei[1].max() < len(g_ids)

    def test_bipartite_empty_when_no_match(self) -> None:
        from src.gnn_collusion import _build_bipartite_edges
        df = pd.DataFrame({"claimant_id": ["A"], "repair_shop_id": ["X"]})
        ei = _build_bipartite_edges(df, ["B"], ["Y"], "claimant_id", "repair_shop_id")
        assert ei.shape == (2, 0)

    def test_co_usage_edges_symmetric(self) -> None:
        """Co-usage edges should be undirected (appear in both directions)."""
        from src.gnn_collusion import _build_co_usage_edges
        # Two garages each used by 3 claimants, 2 claimants in common → edge
        df = pd.DataFrame({
            "claimant_id":   ["C1", "C1", "C2", "C2", "C3"],
            "repair_shop_id": ["RS1", "RS2", "RS1", "RS2", "RS1"],
        })
        ei = _build_co_usage_edges(df, ["RS1", "RS2"], "repair_shop_id",
                                   min_co_usage=2)
        # Undirected: (0,1) and (1,0) both present
        pairs = set(zip(ei[0].tolist(), ei[1].tolist()))
        assert (0, 1) in pairs and (1, 0) in pairs

    def test_co_usage_no_edges_below_threshold(self) -> None:
        from src.gnn_collusion import _build_co_usage_edges
        # Only 1 shared claimant → below min_co_usage=2
        df = pd.DataFrame({
            "claimant_id":    ["C1", "C2"],
            "repair_shop_id": ["RS1", "RS2"],
        })
        ei = _build_co_usage_edges(df, ["RS1", "RS2"], "repair_shop_id", min_co_usage=2)
        assert ei.shape[1] == 0


# ===========================================================================
# D — build_hetero_graph()
# ===========================================================================

class TestBuildHeteroGraph:
    def test_returns_metadata_always(self) -> None:
        from src.gnn_collusion import build_hetero_graph, GraphMetadata
        df = _make_claims(20)
        _, meta = build_hetero_graph(df)
        assert isinstance(meta, GraphMetadata)
        assert len(meta.claimant_ids) > 0

    def test_metadata_claimant_count_matches_df(self) -> None:
        from src.gnn_collusion import build_hetero_graph
        df = _make_claims(20)
        _, meta = build_hetero_graph(df)
        assert len(meta.claimant_ids) == df["claimant_id"].nunique()

    def test_metadata_garage_count_matches_df(self) -> None:
        from src.gnn_collusion import build_hetero_graph
        df = _make_claims(30)
        _, meta = build_hetero_graph(df)
        expected = df["repair_shop_id"].dropna().nunique()
        assert len(meta.garage_ids) == expected

    @pytest.mark.skipif(not HAS_PYG, reason="torch_geometric not installed")
    def test_graph_node_features_shape(self) -> None:
        from src.gnn_collusion import (
            build_hetero_graph, CLAIMANT_FEAT_DIM, GARAGE_FEAT_DIM
        )
        df = _make_claims(30)
        data, meta = build_hetero_graph(df)
        assert data is not None
        assert data["claimant"].x.shape == (len(meta.claimant_ids), CLAIMANT_FEAT_DIM)
        assert data["garage"].x.shape[1]  == GARAGE_FEAT_DIM

    @pytest.mark.skipif(not HAS_PYG, reason="torch_geometric not installed")
    def test_graph_labels_binary(self) -> None:
        from src.gnn_collusion import build_hetero_graph
        df = _make_claims(20)
        data, _ = build_hetero_graph(df)
        assert data["claimant"].y is not None
        assert set(data["claimant"].y.tolist()).issubset({0, 1})

    @pytest.mark.skipif(not HAS_PYG, reason="torch_geometric not installed")
    def test_graph_edge_indices_dtype(self) -> None:
        from src.gnn_collusion import build_hetero_graph, EDGE_TYPES
        df = _make_claims(30)
        data, _ = build_hetero_graph(df)
        for et in EDGE_TYPES:
            ei = data[et[0], et[1], et[2]].edge_index
            assert ei.dtype == torch.long, f"Edge {et} has wrong dtype"
            assert ei.shape[0] == 2

    @pytest.mark.skipif(not HAS_PYG, reason="torch_geometric not installed")
    def test_graph_reverse_edges_present(self) -> None:
        from src.gnn_collusion import build_hetero_graph, REV_EDGE_TYPES
        df = _make_claims(30)
        data, _ = build_hetero_graph(df)
        for et in REV_EDGE_TYPES:
            ei = data[et[0], et[1], et[2]].edge_index
            assert ei.shape[0] == 2

    def test_build_with_no_garages_returns_empty_garage_nodes(self) -> None:
        from src.gnn_collusion import build_hetero_graph
        df = _make_claims(10)
        df["repair_shop_id"] = None
        _, meta = build_hetero_graph(df)
        assert len(meta.garage_ids) == 0

    def test_build_safe_on_empty_dataframe(self) -> None:
        from src.gnn_collusion import build_hetero_graph
        empty = pd.DataFrame(columns=["claimant_id", "claim_amount", "fraud_label",
                                       "days_since_policy_start", "num_prior_claims",
                                       "claim_type", "claim_severity",
                                       "repair_shop_id", "medical_provider_id"])
        _, meta = build_hetero_graph(empty)  # must not raise
        assert isinstance(meta.claimant_ids, list)


# ===========================================================================
# E — CollusionGNNModel forward pass
# ===========================================================================

@pytest.mark.skipif(not HAS_PYG, reason="torch_geometric not installed")
class TestCollusionGNNModel:
    @pytest.fixture(scope="class")
    def graph_data(self):
        from src.gnn_collusion import build_hetero_graph
        return build_hetero_graph(_make_claims(40))

    def test_forward_returns_claimant_logits(self, graph_data) -> None:
        from src.gnn_collusion import CollusionGNNModel, NODE_TYPES, ALL_EDGE_TYPES
        data, meta = graph_data
        model = CollusionGNNModel(hidden_dim=16, dropout=0.0)
        model.eval()

        x_dict = {k: data[k].x for k in NODE_TYPES if hasattr(data[k], "x")}
        ei_dict = {
            et: data[et[0], et[1], et[2]].edge_index
            for et in ALL_EDGE_TYPES
            if hasattr(data[et[0], et[1], et[2]], "edge_index")
        }
        with torch.no_grad():
            out = model(x_dict, ei_dict)

        assert "claimant" in out
        assert out["claimant"].shape == (len(meta.claimant_ids), 1)

    def test_predict_scores_in_01(self, graph_data) -> None:
        from src.gnn_collusion import CollusionGNNModel, NODE_TYPES, ALL_EDGE_TYPES
        data, meta = graph_data
        model = CollusionGNNModel(hidden_dim=16, dropout=0.0)
        x_dict = {k: data[k].x for k in NODE_TYPES if hasattr(data[k], "x")}
        ei_dict = {
            et: data[et[0], et[1], et[2]].edge_index
            for et in ALL_EDGE_TYPES
            if hasattr(data[et[0], et[1], et[2]], "edge_index")
        }
        scores = model.predict_scores(x_dict, ei_dict)
        assert scores.shape[0] == len(meta.claimant_ids)
        assert float(scores.min()) >= 0.0
        assert float(scores.max()) <= 1.0

    def test_forward_output_changes_with_different_weights(self, graph_data) -> None:
        """Two freshly initialised models should produce different outputs."""
        from src.gnn_collusion import CollusionGNNModel, NODE_TYPES, ALL_EDGE_TYPES
        data, _ = graph_data
        x_dict = {k: data[k].x for k in NODE_TYPES if hasattr(data[k], "x")}
        ei_dict = {
            et: data[et[0], et[1], et[2]].edge_index
            for et in ALL_EDGE_TYPES
            if hasattr(data[et[0], et[1], et[2]], "edge_index")
        }
        torch.manual_seed(0)
        m1 = CollusionGNNModel(hidden_dim=16, dropout=0.0)
        torch.manual_seed(99)
        m2 = CollusionGNNModel(hidden_dim=16, dropout=0.0)
        with torch.no_grad():
            o1 = m1(x_dict, ei_dict)["claimant"]
            o2 = m2(x_dict, ei_dict)["claimant"]
        assert not torch.allclose(o1, o2), "Two random models gave identical outputs"

    def test_dropout_only_active_in_train_mode(self, graph_data) -> None:
        from src.gnn_collusion import CollusionGNNModel, NODE_TYPES, ALL_EDGE_TYPES
        data, _ = graph_data
        x_dict = {k: data[k].x for k in NODE_TYPES if hasattr(data[k], "x")}
        ei_dict = {
            et: data[et[0], et[1], et[2]].edge_index
            for et in ALL_EDGE_TYPES
            if hasattr(data[et[0], et[1], et[2]], "edge_index")
        }
        model = CollusionGNNModel(hidden_dim=16, dropout=0.9)
        model.eval()
        with torch.no_grad():
            o1 = model(x_dict, ei_dict)["claimant"]
            o2 = model(x_dict, ei_dict)["claimant"]
        # In eval mode, outputs must be deterministic
        assert torch.allclose(o1, o2)


# ===========================================================================
# F — CollusionGNNTrainer
# ===========================================================================

@pytest.mark.skipif(not HAS_PYG, reason="torch_geometric not installed")
class TestCollusionGNNTrainer:
    @pytest.fixture(scope="class")
    def trained_model_and_data(self, tmp_path_factory):
        from src.gnn_collusion import TrainingConfig, CollusionGNNTrainer, build_hetero_graph
        tmp = tmp_path_factory.mktemp("models")
        df   = _make_claims(40)
        data, meta = build_hetero_graph(df)
        cfg  = TrainingConfig(
            epochs=5,
            hidden_dim=16,
            early_stop_patience=3,
            model_save_path=tmp / "gnn_collusion_test.pt",
        )
        trainer = CollusionGNNTrainer(config=cfg)
        model   = trainer.train(data, verbose=False)
        return model, data, meta, trainer

    def test_train_returns_model(self, trained_model_and_data) -> None:
        from src.gnn_collusion import CollusionGNNModel
        model, _, _, _ = trained_model_and_data
        assert isinstance(model, CollusionGNNModel)

    def test_model_file_saved(self, trained_model_and_data, tmp_path) -> None:
        from src.gnn_collusion import TrainingConfig, CollusionGNNTrainer, build_hetero_graph
        save_path = tmp_path / "saved.pt"
        df   = _make_claims(20)
        data, _ = build_hetero_graph(df)
        cfg  = TrainingConfig(epochs=3, hidden_dim=16, model_save_path=save_path)
        trainer = CollusionGNNTrainer(config=cfg)
        trainer.train(data, verbose=False)
        assert save_path.exists()

    def test_evaluate_returns_dict_with_expected_keys(self, trained_model_and_data) -> None:
        model, data, _, trainer = trained_model_and_data
        metrics = trainer.evaluate(model, data)
        for key in ("loss", "accuracy", "precision", "recall", "f1"):
            assert key in metrics, f"Missing key: {key}"

    def test_evaluate_metrics_in_range(self, trained_model_and_data) -> None:
        model, data, _, trainer = trained_model_and_data
        metrics = trainer.evaluate(model, data)
        assert 0.0 <= metrics["accuracy"]  <= 1.0
        assert 0.0 <= metrics["precision"] <= 1.0
        assert 0.0 <= metrics["recall"]    <= 1.0
        assert 0.0 <= metrics["f1"]        <= 1.0
        assert metrics["loss"] >= 0.0

    def test_train_with_no_data_returns_stub(self) -> None:
        from src.gnn_collusion import CollusionGNNTrainer
        trainer = CollusionGNNTrainer()
        model   = trainer.train(None, verbose=False)
        # Should return stub without raising
        assert model is not None

    def test_evaluate_with_no_data_returns_empty(self) -> None:
        from src.gnn_collusion import CollusionGNNTrainer, CollusionGNNModel
        trainer = CollusionGNNTrainer()
        model   = CollusionGNNModel()
        result  = trainer.evaluate(model, None)
        assert result == {}


# ===========================================================================
# G — score_claimants()
# ===========================================================================

class TestScoreClaimants:
    def test_returns_empty_when_pyg_absent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import src.gnn_collusion as gc
        monkeypatch.setattr(gc, "HAS_PYG",   False)
        monkeypatch.setattr(gc, "HAS_TORCH",  False)
        from src.gnn_collusion import score_claimants, CollusionGNNModel
        result = score_claimants(CollusionGNNModel(), None)
        assert result == {}

    def test_returns_empty_when_data_is_none(self) -> None:
        from src.gnn_collusion import score_claimants, CollusionGNNModel
        result = score_claimants(CollusionGNNModel(), None)
        assert result == {}

    @pytest.mark.skipif(not HAS_PYG, reason="torch_geometric not installed")
    def test_scores_keyed_by_claimant_id(self) -> None:
        from src.gnn_collusion import (
            build_hetero_graph, CollusionGNNModel, score_claimants
        )
        df    = _make_claims(20)
        data, meta = build_hetero_graph(df)
        model = CollusionGNNModel(hidden_dim=16, dropout=0.0)
        scores = score_claimants(model, data, meta=meta)
        assert set(scores.keys()) == set(meta.claimant_ids)

    @pytest.mark.skipif(not HAS_PYG, reason="torch_geometric not installed")
    def test_scores_in_01_range(self) -> None:
        from src.gnn_collusion import (
            build_hetero_graph, CollusionGNNModel, score_claimants
        )
        df    = _make_claims(20)
        data, meta = build_hetero_graph(df)
        model = CollusionGNNModel(hidden_dim=16, dropout=0.0)
        scores = score_claimants(model, data, meta=meta)
        for cid, s in scores.items():
            assert 0.0 <= s <= 1.0, f"Score out of range for {cid}: {s}"

    @pytest.mark.skipif(not HAS_PYG, reason="torch_geometric not installed")
    def test_scores_use_integer_keys_when_no_meta(self) -> None:
        from src.gnn_collusion import build_hetero_graph, CollusionGNNModel, score_claimants
        df    = _make_claims(10)
        data, _ = build_hetero_graph(df)
        model   = CollusionGNNModel(hidden_dim=16, dropout=0.0)
        scores  = score_claimants(model, data, meta=None)
        # Keys should be stringified integers
        assert all(k.isdigit() for k in scores.keys())


# ===========================================================================
# H — write_scores_to_neo4j()
# ===========================================================================

class TestWriteScoresToNeo4j:
    def _mock_driver(self, n_updated: int = 3):
        """Build a mock neo4j driver chain."""
        mock_record  = MagicMock()
        mock_record.__getitem__ = MagicMock(return_value=n_updated)
        mock_result  = MagicMock()
        mock_result.single.return_value = mock_record
        mock_session = MagicMock()
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__  = MagicMock(return_value=False)
        mock_session.run.return_value = mock_result
        mock_driver  = MagicMock()
        mock_driver.session.return_value = mock_session
        mock_driver.verify_connectivity.return_value = None
        return mock_driver

    def test_returns_zero_when_neo4j_absent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import builtins
        real_import = builtins.__import__
        def _block_neo4j(name, *args, **kwargs):
            if name == "neo4j":
                raise ImportError("blocked")
            return real_import(name, *args, **kwargs)
        monkeypatch.setattr(builtins, "__import__", _block_neo4j)
        from src.gnn_collusion import write_scores_to_neo4j
        result = write_scores_to_neo4j({"CLT01": 0.9})
        assert result == 0

    def test_returns_zero_on_empty_scores(self) -> None:
        from src.gnn_collusion import write_scores_to_neo4j
        result = write_scores_to_neo4j({})
        assert result == 0

    def test_calls_cypher_merge_with_correct_structure(self) -> None:
        from src.gnn_collusion import write_scores_to_neo4j
        scores = {"CLT01": 0.9, "CLT02": 0.3, "CLT03": 0.7}
        mock_driver = self._mock_driver(n_updated=3)
        neo4j_mock = MagicMock()
        neo4j_mock.GraphDatabase.driver.return_value = mock_driver

        with patch.dict("sys.modules", {"neo4j": neo4j_mock}):
            result = write_scores_to_neo4j(
                scores, uri="bolt://localhost:7687", user="neo4j", password="pw"
            )

        # Driver constructed with the right URI and credentials
        neo4j_mock.GraphDatabase.driver.assert_called_once_with(
            "bolt://localhost:7687", auth=("neo4j", "pw")
        )
        # verify_connectivity was called
        mock_driver.verify_connectivity.assert_called_once()

    def test_batches_large_score_dict(self) -> None:
        """Scores exceeding batch_size must be written in multiple transactions."""
        from src.gnn_collusion import write_scores_to_neo4j
        scores = {f"CLT{i:04d}": float(i) / 1000 for i in range(250)}

        mock_record  = MagicMock(); mock_record.__getitem__ = MagicMock(return_value=100)
        mock_result  = MagicMock(); mock_result.single.return_value = mock_record
        mock_session = MagicMock()
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__  = MagicMock(return_value=False)
        mock_session.run.return_value = mock_result
        mock_driver  = MagicMock()
        mock_driver.session.return_value = mock_session
        mock_driver.verify_connectivity.return_value = None

        neo4j_mock = MagicMock()
        neo4j_mock.GraphDatabase.driver.return_value = mock_driver

        with patch.dict("sys.modules", {"neo4j": neo4j_mock}):
            write_scores_to_neo4j(scores, batch_size=100)

        # 250 scores / batch 100 = 3 calls to session.run
        assert mock_session.run.call_count == 3

    def test_flag_set_true_above_threshold(self) -> None:
        """Claimants with score ≥ COLLUSION_THRESHOLD must have flag=True."""
        from src.gnn_collusion import write_scores_to_neo4j, COLLUSION_THRESHOLD
        captured_rows = []

        def _mock_run(cypher, **kwargs):
            rows = kwargs.get("rows", [])
            captured_rows.extend(rows)
            m = MagicMock(); m.single.return_value = {"updated": len(rows)}
            return m

        mock_session = MagicMock()
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__  = MagicMock(return_value=False)
        mock_session.run.side_effect = _mock_run
        mock_driver  = MagicMock()
        mock_driver.session.return_value = mock_session
        mock_driver.verify_connectivity.return_value = None
        neo4j_mock = MagicMock()
        neo4j_mock.GraphDatabase.driver.return_value = mock_driver

        scores = {"A": COLLUSION_THRESHOLD + 0.1, "B": COLLUSION_THRESHOLD - 0.1}
        with patch.dict("sys.modules", {"neo4j": neo4j_mock}):
            write_scores_to_neo4j(scores)

        flag_map = {r["cid"]: r["flag"] for r in captured_rows}
        assert flag_map["A"] is True
        assert flag_map["B"] is False

    def test_returns_zero_when_connection_fails(self) -> None:
        from src.gnn_collusion import write_scores_to_neo4j
        neo4j_mock = MagicMock()
        mock_driver = MagicMock()
        mock_driver.verify_connectivity.side_effect = ConnectionError("refused")
        neo4j_mock.GraphDatabase.driver.return_value = mock_driver

        with patch.dict("sys.modules", {"neo4j": neo4j_mock}):
            result = write_scores_to_neo4j({"CLT01": 0.8})
        assert result == 0


# ===========================================================================
# I — load_model() / _save_model()
# ===========================================================================

class TestModelPersistence:
    @pytest.mark.skipif(not HAS_PYG, reason="torch_geometric not installed")
    def test_save_and_load_roundtrip(self, tmp_path: Path) -> None:
        from src.gnn_collusion import (
            CollusionGNNModel, _save_model, load_model,
            build_hetero_graph, NODE_TYPES, ALL_EDGE_TYPES
        )
        df    = _make_claims(20)
        data, meta = build_hetero_graph(df)
        path  = tmp_path / "test_model.pt"

        model = CollusionGNNModel(hidden_dim=16, dropout=0.0)
        torch.manual_seed(7)

        x_dict = {k: data[k].x for k in NODE_TYPES if hasattr(data[k], "x")}
        ei_dict = {
            et: data[et[0], et[1], et[2]].edge_index
            for et in ALL_EDGE_TYPES
            if hasattr(data[et[0], et[1], et[2]], "edge_index")
        }
        with torch.no_grad():
            out_before = model(x_dict, ei_dict)["claimant"].clone()

        _save_model(model, path)
        assert path.exists()

        loaded = load_model(path, hidden_dim=16)
        with torch.no_grad():
            out_after = loaded(x_dict, ei_dict)["claimant"]

        assert torch.allclose(out_before, out_after, atol=1e-5), \
            "Loaded model output differs from saved model"

    def test_load_returns_fresh_model_when_file_absent(self, tmp_path: Path) -> None:
        from src.gnn_collusion import load_model
        model = load_model(tmp_path / "does_not_exist.pt")
        assert model is not None

    def test_save_no_op_when_torch_absent(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        import src.gnn_collusion as gc
        monkeypatch.setattr(gc, "HAS_TORCH", False)
        from src.gnn_collusion import _save_model, CollusionGNNModel
        path = tmp_path / "should_not_exist.pt"
        _save_model(CollusionGNNModel(), path)   # must not raise
        assert not path.exists()


# ===========================================================================
# J — GraphCollusionDetector two-stage pipeline
# ===========================================================================

class TestGraphCollusionDetector:
    def test_returns_list_always(self) -> None:
        from src.graph_collusion import GraphCollusionDetector
        df = _make_small_claims()
        rings = GraphCollusionDetector(use_gnn=False).analyze(df)
        assert isinstance(rings, list)

    def test_detects_at_least_one_ring(self) -> None:
        from src.graph_collusion import GraphCollusionDetector
        df = _make_small_claims()
        rings = GraphCollusionDetector(use_gnn=False).analyze(df)
        assert len(rings) >= 1, "Expected at least one collusion ring"

    def test_ring_has_required_fields(self) -> None:
        from src.graph_collusion import GraphCollusionDetector, CollusionRing
        df = _make_small_claims()
        rings = GraphCollusionDetector(use_gnn=False).analyze(df)
        for ring in rings:
            assert isinstance(ring, CollusionRing)
            assert ring.ring_id
            assert len(ring.claimant_ids) >= 2
            assert ring.severity in ("low", "medium", "high")
            assert isinstance(ring.gnn_scores, dict)
            assert isinstance(ring.max_gnn_score, float)

    def test_centrality_score_positive(self) -> None:
        from src.graph_collusion import GraphCollusionDetector
        df = _make_small_claims()
        rings = GraphCollusionDetector(use_gnn=False).analyze(df)
        for ring in rings:
            assert ring.centrality_score >= 0

    def test_returns_empty_for_empty_df(self) -> None:
        from src.graph_collusion import GraphCollusionDetector
        empty = pd.DataFrame(columns=["claimant_id", "repair_shop_id",
                                       "medical_provider_id", "witness_ids"])
        rings = GraphCollusionDetector(use_gnn=False).analyze(empty)
        assert rings == []

    def test_gnn_rescore_skipped_when_use_gnn_false(self) -> None:
        from src.graph_collusion import GraphCollusionDetector
        df = _make_small_claims()
        detector = GraphCollusionDetector(use_gnn=False)
        rings = detector.analyze(df)
        # With use_gnn=False, gnn_scores must be empty
        for ring in rings:
            assert ring.gnn_scores == {}
            assert ring.max_gnn_score == 0.0

    def test_analyze_tolerates_missing_columns(self) -> None:
        """Columns repair_shop_id / medical_provider_id absent → no crash."""
        from src.graph_collusion import GraphCollusionDetector
        df = pd.DataFrame([
            {"claimant_id": "A", "claim_amount": 100, "fraud_label": 1},
            {"claimant_id": "B", "claim_amount": 200, "fraud_label": 0},
        ])
        rings = GraphCollusionDetector(use_gnn=False).analyze(df)
        assert isinstance(rings, list)

    def test_analyze_returns_empty_without_pandas(self, monkeypatch) -> None:
        import src.graph_collusion as gc
        monkeypatch.setattr(gc, "HAS_PANDAS", False)
        from src.graph_collusion import GraphCollusionDetector
        rings = GraphCollusionDetector(use_gnn=False).analyze(_make_small_claims())
        assert rings == []

    def test_gnn_rescore_degrades_when_torch_absent(self, monkeypatch) -> None:
        import src.graph_collusion as gc
        monkeypatch.setattr(gc, "_HAS_GNN", False)
        from src.graph_collusion import GraphCollusionDetector
        df = _make_small_claims()
        detector = GraphCollusionDetector(use_gnn=True)
        rings = detector.analyze(df)
        # Stage 1 should still work
        assert isinstance(rings, list)
        for ring in rings:
            assert ring.gnn_scores == {}

    def test_rings_sorted_by_max_gnn_score_desc(self) -> None:
        """When GNN scores are attached, highest-risk rings should come first."""
        from src.graph_collusion import GraphCollusionDetector, CollusionRing, _apply_gnn_scores

        rings = [
            CollusionRing(ring_id="r1", claimant_ids=["A", "B"],
                         shared_entities=["e1"], centrality_score=0.5, severity="medium"),
            CollusionRing(ring_id="r2", claimant_ids=["C", "D"],
                         shared_entities=["e2"], centrality_score=0.5, severity="medium"),
        ]
        scores = {"A": 0.3, "B": 0.4, "C": 0.9, "D": 0.85}
        r1 = _apply_gnn_scores(rings[0], scores)
        r2 = _apply_gnn_scores(rings[1], scores)
        result = sorted([r1, r2], key=lambda r: r.max_gnn_score, reverse=True)
        assert result[0].ring_id == "r2"


# ===========================================================================
# K — _apply_gnn_scores severity upgrade rules
# ===========================================================================

class TestApplyGnnScores:
    def _make_ring(self, severity: str, cids: List[str]) -> "CollusionRing":
        from src.graph_collusion import CollusionRing
        return CollusionRing(
            ring_id=f"ring-{severity}",
            claimant_ids=cids,
            shared_entities=["e1"],
            centrality_score=0.5,
            severity=severity,
        )

    def test_upgrades_low_to_medium_above_threshold(self) -> None:
        from src.graph_collusion import _apply_gnn_scores, GNN_SEVERITY_THRESHOLD
        ring   = self._make_ring("low", ["C1", "C2"])
        scores = {"C1": GNN_SEVERITY_THRESHOLD + 0.01, "C2": 0.1}
        result = _apply_gnn_scores(ring, scores)
        assert result.severity == "medium"

    def test_upgrades_to_high_above_high_threshold(self) -> None:
        from src.graph_collusion import _apply_gnn_scores, GNN_HIGH_THRESHOLD
        ring   = self._make_ring("low", ["C1", "C2"])
        scores = {"C1": GNN_HIGH_THRESHOLD + 0.01, "C2": 0.1}
        result = _apply_gnn_scores(ring, scores)
        assert result.severity == "high"

    def test_medium_ring_not_downgraded(self) -> None:
        from src.graph_collusion import _apply_gnn_scores, GNN_SEVERITY_THRESHOLD
        ring   = self._make_ring("medium", ["C1", "C2"])
        scores = {"C1": GNN_SEVERITY_THRESHOLD - 0.1, "C2": 0.1}
        result = _apply_gnn_scores(ring, scores)
        # Should not downgrade from medium to low
        assert result.severity == "medium"

    def test_max_gnn_score_is_max_of_ring_claimants(self) -> None:
        from src.graph_collusion import _apply_gnn_scores
        ring   = self._make_ring("low", ["C1", "C2", "C3"])
        scores = {"C1": 0.3, "C2": 0.9, "C3": 0.5, "C99": 0.99}
        result = _apply_gnn_scores(ring, scores)
        assert abs(result.max_gnn_score - 0.9) < 1e-6

    def test_gnn_scores_only_contains_ring_claimants(self) -> None:
        from src.graph_collusion import _apply_gnn_scores
        ring   = self._make_ring("low", ["C1", "C2"])
        scores = {"C1": 0.7, "C2": 0.4, "C99": 0.99}
        result = _apply_gnn_scores(ring, scores)
        assert set(result.gnn_scores.keys()) == {"C1", "C2"}

    def test_missing_claimants_produce_zero_max_score(self) -> None:
        from src.graph_collusion import _apply_gnn_scores
        ring   = self._make_ring("low", ["C1", "C2"])
        result = _apply_gnn_scores(ring, {})   # no scores for ring's claimants
        assert result.max_gnn_score == 0.0

    def test_severity_unchanged_when_scores_all_below_threshold(self) -> None:
        from src.graph_collusion import _apply_gnn_scores, GNN_SEVERITY_THRESHOLD
        ring   = self._make_ring("medium", ["C1", "C2"])
        scores = {"C1": GNN_SEVERITY_THRESHOLD - 0.2, "C2": 0.05}
        result = _apply_gnn_scores(ring, scores)
        assert result.severity == "medium"


# ===========================================================================
# L — CLI pipeline helpers
# ===========================================================================

class TestCLIPipelineHelpers:
    def test_validate_dataframe_passes_valid(self) -> None:
        from scripts.train_gnn_collusion import validate_dataframe
        df = _make_claims(10)
        validate_dataframe(df)   # must not raise

    def test_validate_dataframe_raises_missing_columns(self) -> None:
        from scripts.train_gnn_collusion import validate_dataframe
        df = pd.DataFrame({"claim_id": ["C1"]})   # missing required cols
        with pytest.raises(ValueError, match="missing required columns"):
            validate_dataframe(df)

    def test_validate_fills_optional_columns(self) -> None:
        from scripts.train_gnn_collusion import validate_dataframe
        df = _make_claims(10)[["claimant_id", "claim_amount", "fraud_label"]]
        validate_dataframe(df)
        assert "days_since_policy_start" in df.columns
        assert "num_prior_claims" in df.columns

    def test_load_claims_from_csv(self, tmp_path: Path) -> None:
        from scripts.train_gnn_collusion import load_claims
        csv_path = tmp_path / "test_claims.csv"
        _make_claims(10).to_csv(csv_path, index=False)
        df = load_claims(csv_path=csv_path)
        assert len(df) == 10

    def test_load_claims_raises_on_missing_file(self, tmp_path: Path) -> None:
        from scripts.train_gnn_collusion import load_claims
        with pytest.raises(FileNotFoundError):
            load_claims(csv_path=tmp_path / "nonexistent.csv")

    def test_export_scores_creates_json(self, tmp_path: Path) -> None:
        from scripts.train_gnn_collusion import export_scores
        scores = {"CLT01": 0.9, "CLT02": 0.3}
        out    = tmp_path / "scores.json"
        export_scores(scores, out)
        assert out.exists()
        data   = json.loads(out.read_text())
        assert "scores" in data
        assert "_meta" in data
        assert data["_meta"]["n_claimants"] == 2

    def test_export_scores_meta_flag_rate(self, tmp_path: Path) -> None:
        from scripts.train_gnn_collusion import export_scores
        scores = {"A": 0.8, "B": 0.6, "C": 0.2, "D": 0.1}
        out    = tmp_path / "s.json"
        export_scores(scores, out)
        meta = json.loads(out.read_text())["_meta"]
        assert meta["flagged_count"] == 2    # A and B ≥ 0.5
        assert meta["flag_rate_pct"] == 50.0

    def test_run_pipeline_returns_2_when_pyg_absent(self, tmp_path: Path) -> None:
        """exit code 2 = PyG not installed / broken (soft exit, not a CI blocker)."""
        import argparse
        from scripts.train_gnn_collusion import run_pipeline

        # Simulate PyG absent by patching _check_pyg to return False
        with patch("scripts.train_gnn_collusion._check_pyg", return_value=False):
            args = argparse.Namespace(
                claims_csv   = None,
                db_url       = None,
                epochs       = 5,
                hidden_dim   = 16,
                dropout      = 0.3,
                lr           = 1e-3,
                weight_decay = 1e-4,
                patience     = 3,
                pos_weight   = 3.0,
                device       = "cpu",
                resume       = False,
                model_out    = None,
                scores_out   = None,
                quiet        = True,
                write_neo4j  = False,
                neo4j_uri    = None,
                neo4j_user   = None,
                neo4j_pwd    = None,
                neo4j_batch  = 200,
            )
            code = run_pipeline(args)
        assert code == 2

    def test_build_parser_defaults(self) -> None:
        from scripts.train_gnn_collusion import _build_parser
        parser = _build_parser()
        args   = parser.parse_args([])
        assert args.epochs == 100
        assert args.hidden_dim == 64
        assert args.device == "cpu"
        assert args.write_neo4j is False
        assert args.quiet is False
