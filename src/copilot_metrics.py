"""
Prometheus observability metrics for the ClaimGuard AI Policy Copilot.

All metrics live under the claimguard_copilot_* namespace and are
Grafana-exportable.  When prometheus_client is not installed every metric is
replaced by a no-op stub so the rest of the codebase never needs an import guard.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Optional Prometheus import
# ---------------------------------------------------------------------------

try:
    from prometheus_client import Counter, Gauge, Histogram  # type: ignore

    HAS_PROMETHEUS = True
except ImportError:
    HAS_PROMETHEUS = False


# ---------------------------------------------------------------------------
# No-op stub classes (used when prometheus_client is absent)
# ---------------------------------------------------------------------------


class _NoOpCounter:
    """Drop-in replacement for prometheus_client.Counter."""

    def labels(self, **kwargs):  # noqa: ANN001
        return self

    def inc(self, amount: float = 1) -> None:  # noqa: ARG002
        pass


class _NoOpHistogram:
    """Drop-in replacement for prometheus_client.Histogram."""

    def labels(self, **kwargs):  # noqa: ANN001
        return self

    def observe(self, amount: float) -> None:  # noqa: ARG002
        pass


class _NoOpGauge:
    """Drop-in replacement for prometheus_client.Gauge."""

    def labels(self, **kwargs):  # noqa: ANN001
        return self

    def set(self, value: float) -> None:  # noqa: ARG002
        pass

    def inc(self, amount: float = 1) -> None:  # noqa: ARG002
        pass

    def dec(self, amount: float = 1) -> None:  # noqa: ARG002
        pass


# ---------------------------------------------------------------------------
# Metric definitions
# ---------------------------------------------------------------------------

if HAS_PROMETHEUS:
    # Number of agent pipeline invocations, labelled by agent name.
    AGENT_RUN_COUNT = Counter(
        "claimguard_copilot_agent_run_total",
        "Total number of agent node executions",
        ["agent_name"],
    )

    # Latency per agent node (seconds).  Tracks p95/p99 via default buckets.
    AGENT_LATENCY = Histogram(
        "claimguard_copilot_agent_latency_seconds",
        "Agent node execution latency in seconds",
        ["agent_name"],
    )

    # Number of tool calls issued by ToolAgent, labelled by tool name.
    TOOL_CALL_COUNT = Counter(
        "claimguard_copilot_tool_call_total",
        "Total number of tool calls issued by the ToolAgent",
        ["tool_name"],
    )

    # Number of successful vector store retrievals.
    RETRIEVER_HIT_COUNT = Counter(
        "claimguard_copilot_retriever_hit_total",
        "Total number of policy document retriever hits",
    )

    # Number of structured decisions drafted by the WriterAgent.
    DECISIONS_DRAFTED = Counter(
        "claimguard_copilot_decisions_drafted_total",
        "Total number of claim/underwriting decisions drafted by WriterAgent",
    )

    # Current depth of the HITL review queue (pending items).
    HITL_QUEUE_DEPTH = Gauge(
        "claimguard_copilot_hitl_queue_depth",
        "Current number of items pending human analyst review",
    )

    # Graph collusion query count.
    GRAPH_QUERY_COUNT = Counter(
        "claimguard_copilot_graph_query_total",
        "Total number of graph collusion queries executed",
    )

else:
    AGENT_RUN_COUNT = _NoOpCounter()      # type: ignore[assignment]
    AGENT_LATENCY = _NoOpHistogram()       # type: ignore[assignment]
    TOOL_CALL_COUNT = _NoOpCounter()       # type: ignore[assignment]
    RETRIEVER_HIT_COUNT = _NoOpCounter()   # type: ignore[assignment]
    DECISIONS_DRAFTED = _NoOpCounter()     # type: ignore[assignment]
    HITL_QUEUE_DEPTH = _NoOpGauge()        # type: ignore[assignment]
    GRAPH_QUERY_COUNT = _NoOpCounter()     # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Helper functions — each wraps the raw metric in try/except
# ---------------------------------------------------------------------------


def record_agent_run(agent_name: str, duration_seconds: float) -> None:
    """Increment run counter and record latency for *agent_name*."""
    try:
        AGENT_RUN_COUNT.labels(agent_name=agent_name).inc()
        AGENT_LATENCY.labels(agent_name=agent_name).observe(duration_seconds)
    except Exception:
        pass


def record_tool_call(tool_name: str) -> None:
    """Increment the tool call counter for *tool_name*."""
    try:
        TOOL_CALL_COUNT.labels(tool_name=tool_name).inc()
    except Exception:
        pass


def record_retriever_hit() -> None:
    """Increment the retriever hit counter."""
    try:
        RETRIEVER_HIT_COUNT.inc()
    except Exception:
        pass


def record_decision_drafted() -> None:
    """Increment the decisions drafted counter."""
    try:
        DECISIONS_DRAFTED.inc()
    except Exception:
        pass


def set_hitl_queue_depth(n: int) -> None:
    """Set the HITL queue depth gauge to *n*."""
    try:
        HITL_QUEUE_DEPTH.set(float(n))
    except Exception:
        pass


def record_graph_query() -> None:
    """Increment the graph query counter."""
    try:
        GRAPH_QUERY_COUNT.inc()
    except Exception:
        pass
