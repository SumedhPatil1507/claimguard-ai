"""
src/graph_collusion.py
======================
Claim-ring collusion detection for ClaimGuard AI.

Detection pipeline
------------------
Stage 1 — Structural ring detection (always runs)
    Build a bipartite graph of claimants and shared entities
    (repair shops / garages, medical providers, witnesses).
    Identify clusters of claimants that share ≥ MIN_SHARED_ENTITIES
    suspicious entities, using either:
      a) Neo4j  (primary — requires a running server)
      b) NetworkX  (in-process fallback)
    Returns a list of CollusionRing objects with initial severity.

Stage 2 — GNN re-scoring (optional — requires torch + torch_geometric)
    Load (or train on-the-fly if no saved model exists) the R-GCN from
    src/gnn_collusion.py.  Run inference to obtain per-claimant collusion
    scores, then upgrade ring severity for rings whose claimants score
    above GNN_SEVERITY_THRESHOLD.

    GNN scoring never blocks stage 1:
      - If torch / PyG are absent             → skip silently
      - If the saved model file doesn't exist  → train a quick model
      - If training or inference fails          → skip silently

Graceful degradation summary
-----------------------------
  Neo4j unavailable            → NetworkX fallback
  NetworkX unavailable         → empty list
  torch / PyG unavailable      → stage 1 result returned as-is
  GNN model file absent        → quick on-the-fly training attempted
  GNN training / infer error   → stage 1 result returned as-is
  Any other exception           → empty list, never raises

Public API
----------
    from src.graph_collusion import GraphCollusionDetector, CollusionRing
    rings = GraphCollusionDetector().analyze(claims_df)
    rings = GraphCollusionDetector(use_gnn=False).analyze(claims_df)  # stage 1 only

    Each CollusionRing has:
        ring_id         – unique hex identifier
        claimant_ids    – sorted list of claimant id strings
        shared_entities – sorted list of shared entity id strings
        centrality_score – float; len(shared_entities)/len(claimants)
        severity        – 'low' | 'medium' | 'high'
        gnn_scores      – dict {claimant_id: float} (empty when GNN skipped)
        max_gnn_score   – float max of gnn_scores values (0.0 when skipped)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional dependency guards
# ---------------------------------------------------------------------------

try:
    import pandas as pd   # type: ignore
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    import networkx as nx  # type: ignore
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False

try:
    from neo4j import GraphDatabase   # type: ignore
    HAS_NEO4J = True
except ImportError:
    HAS_NEO4J = False

# GNN availability is tested lazily inside _gnn_rescore so that
# importing this module never fails even without torch/PyG.
_HAS_GNN: Optional[bool] = None   # resolved on first use


def _gnn_available() -> bool:
    global _HAS_GNN
    if _HAS_GNN is None:
        try:
            import torch                              # noqa: F401
            from torch_geometric.data import HeteroData  # noqa: F401
            _HAS_GNN = True
        except (ImportError, OSError):
            # OSError covers broken DLL installs on Windows
            _HAS_GNN = False
    return _HAS_GNN


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_NEO4J_URI      = os.getenv("NEO4J_URI",      "bolt://localhost:7687")
_NEO4J_USER     = os.getenv("NEO4J_USER",     "neo4j")
_NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "neo4j")

_DEFAULT_MODEL_PATH = Path(__file__).parent.parent / "data" / "models" / "gnn_collusion.pt"

# Stage 1 thresholds
_MIN_ENTITY_CLAIM_COUNT = 3   # entity must appear in ≥ N claims
_MIN_SHARED_ENTITIES    = 2   # pair of claimants must share ≥ N entities

# Stage 2 GNN thresholds
GNN_SEVERITY_THRESHOLD  = 0.60  # claimant GNN score above which ring is upgraded
GNN_HIGH_THRESHOLD      = 0.80  # score above which ring is forced to 'high'


# ---------------------------------------------------------------------------
# Pydantic v2 schema
# ---------------------------------------------------------------------------


class CollusionRing(BaseModel):
    """A detected group of claimants suspected of coordinated fraud."""

    model_config = ConfigDict(str_strip_whitespace=True)

    ring_id:          str
    claimant_ids:     List[str]
    shared_entities:  List[str]
    centrality_score: float
    severity:         str              # 'low' | 'medium' | 'high'
    gnn_scores:       Dict[str, float] = Field(default_factory=dict)
    max_gnn_score:    float            = 0.0


# ---------------------------------------------------------------------------
# Main detector
# ---------------------------------------------------------------------------


class GraphCollusionDetector:
    """
    Two-stage collusion detector.

    Parameters
    ----------
    use_gnn : bool
        Enable GNN re-scoring (stage 2).  Default True.
        Set to False to skip GNN regardless of availability.
    model_path : Path | None
        Path to saved R-GCN weights.  Defaults to
        data/models/gnn_collusion.pt.  If the file is absent,
        a quick on-the-fly model is trained when use_gnn=True.
    gnn_epochs : int
        Epochs for on-the-fly GNN training.  Default 30 (fast).
    """

    def __init__(
        self,
        use_gnn:     bool              = True,
        model_path:  Optional[Path]    = None,
        gnn_epochs:  int               = 30,
    ) -> None:
        self.use_gnn    = use_gnn
        self.model_path = Path(model_path) if model_path else _DEFAULT_MODEL_PATH
        self.gnn_epochs = gnn_epochs

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def analyze(self, claims_df: "pd.DataFrame") -> List[CollusionRing]:
        """
        Run the full two-stage pipeline and return CollusionRing objects.

        Parameters
        ----------
        claims_df : pd.DataFrame
            Must contain at minimum: claimant_id, claim_amount, fraud_label,
            repair_shop_id (optional), medical_provider_id (optional),
            witness_ids (optional).

        Returns
        -------
        List[CollusionRing]  — sorted by max_gnn_score descending, then
                               centrality_score descending.
        """
        if not HAS_PANDAS:
            return []

        # ── Stage 1: structural ring detection ──────────────────────────
        rings = self._structural_detect(claims_df)

        if not rings:
            return []

        # ── Stage 2: GNN re-scoring ─────────────────────────────────────
        if self.use_gnn and _gnn_available():
            rings = self._gnn_rescore(claims_df, rings)

        # Sort: GNN-scored rings first, then by centrality
        rings.sort(
            key=lambda r: (r.max_gnn_score, r.centrality_score),
            reverse=True,
        )
        return rings

    # ------------------------------------------------------------------
    # Stage 1 helpers
    # ------------------------------------------------------------------

    def _structural_detect(self, df: "pd.DataFrame") -> List[CollusionRing]:
        """Delegate to Neo4j if available, otherwise NetworkX."""
        if HAS_NEO4J:
            try:
                return self._analyze_neo4j(df)
            except Exception as exc:
                logger.warning(
                    "graph_collusion: Neo4j unavailable (%s); falling back to NetworkX.", exc
                )

        if HAS_NETWORKX:
            try:
                return self._analyze_networkx(df)
            except Exception as exc:
                logger.error("graph_collusion: NetworkX analysis failed — %s", exc)
                return []

        logger.warning("graph_collusion: no graph backend available (install networkx).")
        return []

    # ── Neo4j backend ───────────────────────────────────────────────────────

    def _analyze_neo4j(self, df: "pd.DataFrame") -> List[CollusionRing]:
        driver = GraphDatabase.driver(
            _NEO4J_URI, auth=(_NEO4J_USER, _NEO4J_PASSWORD)
        )
        driver.verify_connectivity()

        with driver.session() as session:
            session.run("MATCH (n:TempClaimant) DETACH DELETE n")
            session.run("MATCH (n:TempEntity)   DETACH DELETE n")

            for _, row in df.iterrows():
                cid      = str(row.get("claimant_id", ""))
                entities = _extract_entities(row)
                for entity in entities:
                    session.run(
                        "MERGE (c:TempClaimant {id: $cid}) "
                        "MERGE (e:TempEntity   {id: $eid}) "
                        "MERGE (c)-[:LINKED_TO]->(e)",
                        cid=cid, eid=entity,
                    )

            result = session.run(
                "MATCH (c:TempClaimant)-[:LINKED_TO]->(e:TempEntity) "
                "WITH e, collect(DISTINCT c.id) AS claimants "
                "WHERE size(claimants) >= $min_count "
                "RETURN e.id AS entity, claimants, size(claimants) AS degree",
                min_count=_MIN_ENTITY_CLAIM_COUNT,
            )
            suspicious: dict = {}
            for record in result:
                suspicious[record["entity"]] = set(record["claimants"])

        driver.close()
        return _build_rings_from_suspicious(suspicious)

    # ── NetworkX backend ────────────────────────────────────────────────────

    def _analyze_networkx(self, df: "pd.DataFrame") -> List[CollusionRing]:
        G = nx.Graph()

        for _, row in df.iterrows():
            cid      = f"claimant:{row.get('claimant_id', '')}"
            entities = _extract_entities(row)
            for entity in entities:
                eid = f"entity:{entity}"
                G.add_node(cid, node_type="claimant")
                G.add_node(eid, node_type="entity")
                G.add_edge(cid, eid)

        if G.number_of_nodes() == 0:
            return []

        suspicious: dict = {}
        for node, data in G.nodes(data=True):
            if data.get("node_type") == "entity":
                neighbours = [
                    n for n in G.neighbors(node)
                    if G.nodes[n].get("node_type") == "claimant"
                ]
                if len(neighbours) >= _MIN_ENTITY_CLAIM_COUNT:
                    entity_name = node.replace("entity:", "")
                    suspicious[entity_name] = {
                        n.replace("claimant:", "") for n in neighbours
                    }

        return _build_rings_from_suspicious(suspicious)

    # ------------------------------------------------------------------
    # Stage 2 — GNN re-scoring
    # ------------------------------------------------------------------

    def _gnn_rescore(
        self,
        df: "pd.DataFrame",
        rings: List[CollusionRing],
    ) -> List[CollusionRing]:
        """
        Run the R-GCN, attach per-claimant scores to each ring, and
        optionally upgrade severity based on GNN_SEVERITY_THRESHOLD.

        Never raises — returns rings unchanged on any error.
        """
        try:
            from src.gnn_collusion import (       # type: ignore
                TrainingConfig,
                build_hetero_graph,
                load_model,
                score_claimants,
                CollusionGNNTrainer,
            )
        except ImportError as exc:
            logger.warning("graph_collusion: gnn_collusion import failed — %s", exc)
            return rings

        try:
            # Build graph (metadata always populated even without PyG)
            data, meta = build_hetero_graph(df)

            # ── Load or train model ───────────────────────────────────────
            if self.model_path.exists():
                model = load_model(self.model_path)
                logger.info(
                    "graph_collusion: loaded GNN model from %s.", self.model_path
                )
            else:
                logger.info(
                    "graph_collusion: no saved GNN model at %s — "
                    "training on-the-fly for %d epochs.",
                    self.model_path,
                    self.gnn_epochs,
                )
                cfg = TrainingConfig(
                    epochs=self.gnn_epochs,
                    early_stop_patience=5,
                    model_save_path=self.model_path,
                )
                trainer = CollusionGNNTrainer(config=cfg)
                model   = trainer.train(data, verbose=False)

            # ── Inference ─────────────────────────────────────────────────
            scores: Dict[str, float] = score_claimants(model, data, meta=meta)

            if not scores:
                return rings

            logger.info(
                "graph_collusion: GNN scored %d claimants (flagged: %d).",
                len(scores),
                sum(1 for v in scores.values() if v >= GNN_SEVERITY_THRESHOLD),
            )

            # ── Attach scores to rings and upgrade severity ───────────────
            return [
                _apply_gnn_scores(ring, scores)
                for ring in rings
            ]

        except Exception as exc:
            logger.error(
                "graph_collusion: GNN re-scoring failed (%s); "
                "returning stage-1 results.", exc,
            )
            return rings


# ---------------------------------------------------------------------------
# Shared ring-building logic
# ---------------------------------------------------------------------------


def _extract_entities(row: "pd.Series") -> List[str]:
    """Extract non-null entity identifiers from a single claims row."""
    entities: List[str] = []

    shop = row.get("repair_shop_id", "")
    if shop and str(shop) not in ("nan", "", "None"):
        entities.append(f"shop:{shop}")

    prov = row.get("medical_provider_id", "")
    if prov and str(prov) not in ("nan", "", "None"):
        entities.append(f"provider:{prov}")

    wits = row.get("witness_ids", "")
    if wits and str(wits) not in ("nan", "", "None"):
        for w in str(wits).split(","):
            w = w.strip()
            if w:
                entities.append(f"witness:{w}")

    return entities


def _build_rings_from_suspicious(suspicious: dict) -> List[CollusionRing]:
    """
    Group claimants that share ≥ _MIN_SHARED_ENTITIES suspicious entities
    into CollusionRing objects with initial severity based on ring size.
    """
    if not suspicious:
        return []

    # claimant_id → set of suspicious entities it belongs to
    claimant_entities: dict = {}
    for entity, claimants in suspicious.items():
        for cid in claimants:
            claimant_entities.setdefault(cid, set()).add(entity)

    # Greedy grouping: merge claimants that share ≥ _MIN_SHARED_ENTITIES
    rings: List[dict] = []
    for cid, ents in claimant_entities.items():
        merged = False
        for ring in rings:
            overlap = ents & ring["shared_entities"]
            if len(overlap) >= _MIN_SHARED_ENTITIES:
                ring["claimant_ids"].add(cid)
                ring["shared_entities"] |= ents
                merged = True
                break
        if not merged:
            rings.append({"claimant_ids": {cid}, "shared_entities": set(ents)})

    result: List[CollusionRing] = []
    for ring in rings:
        n = len(ring["claimant_ids"])
        if n < 2:
            continue

        centrality = round(len(ring["shared_entities"]) / max(n, 1), 4)
        severity   = "high" if n >= 5 else "medium" if n >= 3 else "low"

        result.append(
            CollusionRing(
                ring_id          = f"ring-{uuid4().hex[:8]}",
                claimant_ids     = sorted(ring["claimant_ids"]),
                shared_entities  = sorted(ring["shared_entities"]),
                centrality_score = centrality,
                severity         = severity,
            )
        )

    return result


def _apply_gnn_scores(
    ring: CollusionRing,
    all_scores: Dict[str, float],
) -> CollusionRing:
    """
    Attach GNN collusion scores to a ring and optionally upgrade its severity.

    Severity upgrade rules
    ----------------------
    - Any claimant score ≥ GNN_HIGH_THRESHOLD  → ring severity = 'high'
    - Any claimant score ≥ GNN_SEVERITY_THRESHOLD (and ring is 'low')
                                               → ring severity = 'medium'
    - Otherwise severity is unchanged from stage 1.
    """
    ring_scores   = {
        cid: all_scores[cid]
        for cid in ring.claimant_ids
        if cid in all_scores
    }
    max_score = max(ring_scores.values(), default=0.0)

    # Determine upgraded severity
    current  = ring.severity
    upgraded = current
    if max_score >= GNN_HIGH_THRESHOLD:
        upgraded = "high"
    elif max_score >= GNN_SEVERITY_THRESHOLD and current == "low":
        upgraded = "medium"

    if upgraded != current:
        logger.info(
            "graph_collusion: ring %s upgraded %s → %s (max_gnn=%.3f).",
            ring.ring_id, current, upgraded, max_score,
        )

    # Return a new CollusionRing with GNN fields filled in
    return CollusionRing(
        ring_id          = ring.ring_id,
        claimant_ids     = ring.claimant_ids,
        shared_entities  = ring.shared_entities,
        centrality_score = ring.centrality_score,
        severity         = upgraded,
        gnn_scores       = ring_scores,
        max_gnn_score    = round(max_score, 6),
    )
