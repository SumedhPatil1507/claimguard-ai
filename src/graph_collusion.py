"""
Graph-based claim-ring collusion detection for ClaimGuard AI.

Detects claimant rings by building a bipartite graph of claimants and shared
entities (repair shops, medical providers, witnesses), then identifying clusters
of claimants who share multiple high-centrality entities.

Primary backend: Neo4j (NEO4J_URI env var, default bolt://localhost:7687).
First fallback: NetworkX (in-process, no server required).
Final fallback: returns [] — never raises.
"""

from __future__ import annotations

import os
from typing import List
from uuid import uuid4

from pydantic import BaseModel, ConfigDict

try:
    import pandas as pd  # type: ignore

    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    import networkx as nx  # type: ignore

    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False

try:
    from neo4j import GraphDatabase  # type: ignore

    HAS_NEO4J = True
except ImportError:
    HAS_NEO4J = False

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
_NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
_NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
_NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "neo4j")

# An entity must appear in at least this many distinct claims to be "suspicious".
_MIN_ENTITY_CLAIM_COUNT = 3
# Two claimants must share at least this many suspicious entities to be in a ring.
_MIN_SHARED_ENTITIES = 2


# ---------------------------------------------------------------------------
# Pydantic v2 schema
# ---------------------------------------------------------------------------


class CollusionRing(BaseModel):
    """A detected group of claimants suspected of coordinated fraud."""

    model_config = ConfigDict(str_strip_whitespace=True)

    ring_id: str
    claimant_ids: List[str]
    shared_entities: List[str]
    centrality_score: float
    severity: str  # 'low' | 'medium' | 'high'


# ---------------------------------------------------------------------------
# Main detector
# ---------------------------------------------------------------------------


class GraphCollusionDetector:
    """
    Analyses a claims DataFrame to surface collusion rings.

    Usage:
        detector = GraphCollusionDetector()
        rings = detector.analyze(df)
    """

    def analyze(self, claims_df: "pd.DataFrame") -> List[CollusionRing]:
        """
        Return a (possibly empty) list of CollusionRing objects.

        Order of preference:
          1. Neo4j graph query (requires running server)
          2. NetworkX in-process graph analysis
          3. Empty list (both backends unavailable)
        """
        if not HAS_PANDAS:
            return []

        try:
            if HAS_NEO4J:
                return self._analyze_neo4j(claims_df)
        except Exception:
            pass  # Neo4j server not running — fall through to NetworkX.

        if HAS_NETWORKX:
            try:
                return self._analyze_networkx(claims_df)
            except Exception:
                return []

        return []

    # ------------------------------------------------------------------
    # Neo4j backend
    # ------------------------------------------------------------------

    def _analyze_neo4j(self, df: "pd.DataFrame") -> List[CollusionRing]:
        """
        Build a transient bipartite graph in Neo4j, compute degree centrality,
        and return collusion rings.

        Falls back to NetworkX on any connection error.
        """
        driver = GraphDatabase.driver(
            _NEO4J_URI,
            auth=(_NEO4J_USER, _NEO4J_PASSWORD),
        )
        # Verify connectivity — raises if server unreachable.
        driver.verify_connectivity()

        with driver.session() as session:
            # Clear previous run's transient data.
            session.run("MATCH (n:TempClaimant) DETACH DELETE n")
            session.run("MATCH (n:TempEntity) DETACH DELETE n")

            for _, row in df.iterrows():
                cid = str(row.get("claimant_id", ""))
                entities = _extract_entities(row)
                for entity in entities:
                    session.run(
                        "MERGE (c:TempClaimant {id: $cid}) "
                        "MERGE (e:TempEntity {id: $eid}) "
                        "MERGE (c)-[:LINKED_TO]->(e)",
                        cid=cid,
                        eid=entity,
                    )

            # Find entities shared across >= _MIN_ENTITY_CLAIM_COUNT claimants.
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

    # ------------------------------------------------------------------
    # NetworkX backend
    # ------------------------------------------------------------------

    def _analyze_networkx(self, df: "pd.DataFrame") -> List[CollusionRing]:
        """
        Build a bipartite NetworkX graph and detect collusion rings without
        requiring an external database server.
        """
        G = nx.Graph()

        for _, row in df.iterrows():
            cid = f"claimant:{row.get('claimant_id', '')}"
            entities = _extract_entities(row)
            for entity in entities:
                eid = f"entity:{entity}"
                G.add_node(cid, node_type="claimant")
                G.add_node(eid, node_type="entity")
                G.add_edge(cid, eid)

        if G.number_of_nodes() == 0:
            return []

        # Degree centrality on the full graph.
        degree_centrality = nx.degree_centrality(G)

        # Identify entity nodes whose degree (number of linked claimants)
        # meets the suspicious threshold.
        suspicious: dict = {}
        for node, data in G.nodes(data=True):
            if data.get("node_type") == "entity":
                neighbours = [n for n in G.neighbors(node) if G.nodes[n].get("node_type") == "claimant"]
                if len(neighbours) >= _MIN_ENTITY_CLAIM_COUNT:
                    entity_name = node.replace("entity:", "")
                    suspicious[entity_name] = {n.replace("claimant:", "") for n in neighbours}

        return _build_rings_from_suspicious(suspicious)


# ---------------------------------------------------------------------------
# Shared ring-building logic (used by both backends)
# ---------------------------------------------------------------------------


def _extract_entities(row: "pd.Series") -> List[str]:
    """Extract non-null entity identifiers from a single claims row."""
    entities: List[str] = []

    if "repair_shop_id" in row and row["repair_shop_id"] and str(row["repair_shop_id"]) not in ("nan", ""):
        entities.append(f"shop:{row['repair_shop_id']}")

    if (
        "medical_provider_id" in row
        and row["medical_provider_id"]
        and str(row["medical_provider_id"]) not in ("nan", "")
    ):
        entities.append(f"provider:{row['medical_provider_id']}")

    if "witness_ids" in row and row["witness_ids"] and str(row["witness_ids"]) not in ("nan", ""):
        for w in str(row["witness_ids"]).split(","):
            w = w.strip()
            if w:
                entities.append(f"witness:{w}")

    return entities


def _build_rings_from_suspicious(suspicious: dict) -> List[CollusionRing]:
    """
    Group claimants that share >= _MIN_SHARED_ENTITIES suspicious entities
    into CollusionRing objects.
    """
    if not suspicious:
        return []

    # Build an entity-membership map: claimant_id → set of suspicious entity names.
    claimant_entities: dict = {}
    for entity, claimants in suspicious.items():
        for cid in claimants:
            claimant_entities.setdefault(cid, set()).add(entity)

    # Union-Find / simple greedy grouping:
    # Start a new ring for each claimant; merge if they share >= _MIN_SHARED_ENTITIES.
    rings: List[dict] = []  # each dict: {claimant_ids: set, shared_entities: set}

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
            continue  # A ring needs at least two claimants.

        centrality = round(len(ring["shared_entities"]) / max(n, 1), 4)

        if n >= 5:
            severity = "high"
        elif n >= 3:
            severity = "medium"
        else:
            severity = "low"

        result.append(
            CollusionRing(
                ring_id=f"ring-{uuid4().hex[:8]}",
                claimant_ids=sorted(ring["claimant_ids"]),
                shared_entities=sorted(ring["shared_entities"]),
                centrality_score=centrality,
                severity=severity,
            )
        )

    return result
