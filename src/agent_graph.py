"""
LangGraph-powered Policy Copilot pipeline for ClaimGuard AI.

Pipeline nodes (run in sequence):
  retriever_node  → Semantic search over ChromaDB policy/IRDAI vector store.
  tool_node       → Calls the underwriting or claims-fraud scoring engine.
  writer_node     → Synthesises a structured decision draft (Groq LLM or rule-based).
  hitl_router_node → Enqueues draft for mandatory human analyst review (IRDAI mandate).

Graceful-fallback chain:
  - langgraph absent  → run_copilot_fallback() executes nodes linearly.
  - chromadb/ST absent → retriever returns []; pipeline continues.
  - GROQ_API_KEY unset → writer uses rule-based narrative template.
  - ML engines missing → model_result is {}; writer notes unavailability.
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional: LangGraph
# ---------------------------------------------------------------------------
try:
    from langgraph.graph import StateGraph, END  # type: ignore

    HAS_LANGGRAPH = True
except ImportError:
    HAS_LANGGRAPH = False

# ---------------------------------------------------------------------------
# Optional: Groq / LangChain
# ---------------------------------------------------------------------------
try:
    from langchain_groq import ChatGroq  # type: ignore
    from langchain_core.messages import HumanMessage  # type: ignore

    HAS_GROQ = True
except ImportError:
    HAS_GROQ = False

# ---------------------------------------------------------------------------
# CopilotState TypedDict
# ---------------------------------------------------------------------------


class CopilotState(TypedDict):
    query: str
    context_type: str          # 'underwriting' | 'claims'
    features: dict
    retrieved_docs: List[dict]
    model_result: dict
    decision_draft: str
    requires_human_review: bool
    session_id: str
    timestamp: str


# ---------------------------------------------------------------------------
# Lazy singletons — catch every import/init error so startup always succeeds
# ---------------------------------------------------------------------------

try:
    from src.underwriting import UnderwritingEngine, UnderwritingFeatures

    _underwriting_engine: Optional[Any] = UnderwritingEngine()
except Exception:
    _underwriting_engine = None
    UnderwritingFeatures = None  # type: ignore[assignment,misc]

try:
    from src.claims_fraud import FraudDetectionEngine, ClaimFeatures

    _fraud_engine: Optional[Any] = FraudDetectionEngine()
except Exception:
    _fraud_engine = None
    ClaimFeatures = None  # type: ignore[assignment,misc]

try:
    from src.vector_store import policy_store
except Exception:
    policy_store = None  # type: ignore[assignment]

try:
    from src.hitl import hitl_queue, HITLItem
except Exception:
    hitl_queue = None  # type: ignore[assignment]
    HITLItem = None  # type: ignore[assignment,misc]

try:
    from src.copilot_metrics import (
        record_agent_run,
        record_tool_call,
        record_retriever_hit,
        record_decision_drafted,
        set_hitl_queue_depth,
    )
except Exception:

    def record_agent_run(*a: Any, **k: Any) -> None:  # type: ignore[misc]
        pass

    def record_tool_call(*a: Any, **k: Any) -> None:  # type: ignore[misc]
        pass

    def record_retriever_hit(*a: Any, **k: Any) -> None:  # type: ignore[misc]
        pass

    def record_decision_drafted(*a: Any, **k: Any) -> None:  # type: ignore[misc]
        pass

    def set_hitl_queue_depth(*a: Any, **k: Any) -> None:  # type: ignore[misc]
        pass

# ---------------------------------------------------------------------------
# Optional: Guardrails AI for IRDAI compliance validation
# ---------------------------------------------------------------------------
try:
    from src.guardrails_config import get_irdai_validator
    HAS_GUARDRAILS = True
except Exception:
    HAS_GUARDRAILS = False
    get_irdai_validator = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Optional: Enterprise System Prompt
# ---------------------------------------------------------------------------
try:
    from src.policy_copilot_system_prompt import get_system_prompt
    HAS_SYSTEM_PROMPT = True
except Exception:
    HAS_SYSTEM_PROMPT = False

    def get_system_prompt():  # type: ignore[misc]
        return "You are ClaimGuard AI Policy Copilot, an insurance analysis assistant."

# ---------------------------------------------------------------------------
# Optional: OpenTelemetry instrumentation
# ---------------------------------------------------------------------------
try:
    from opentelemetry import trace
    from opentelemetry.trace import Status, StatusCode
    HAS_OPENTELEMETRY = True
    _tracer = trace.get_tracer(__name__)
except ImportError:
    HAS_OPENTELEMETRY = False
    _tracer = None  # type: ignore[assignment]
    Status = None  # type: ignore[assignment]
    StatusCode = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Node: retriever_node
# ---------------------------------------------------------------------------


def retriever_node(state: CopilotState) -> dict:
    """Semantic search over the policy/IRDAI vector store."""
    t0 = time.monotonic()
    results: List[dict] = []

    if HAS_OPENTELEMETRY and _tracer:
        with _tracer.start_as_current_span("agent_graph.retriever_node") as span:
            span.set_attribute("query", state.get("query", "")[:200])
            span.set_attribute("session_id", state.get("session_id", ""))
            try:
                if policy_store is not None:
                    results = policy_store.search(state["query"], top_k=3)
                    if results:
                        record_retriever_hit()
                        span.set_attribute("retrieved_count", len(results))
                else:
                    span.set_status(Status(StatusCode.ERROR, "policy_store not available"))
            except Exception as exc:
                logger.warning("retriever_node: search failed — %s", exc)
                span.set_status(Status(StatusCode.ERROR, str(exc)))
                results = []
    else:
        try:
            if policy_store is not None:
                results = policy_store.search(state["query"], top_k=3)
                if results:
                    record_retriever_hit()
        except Exception as exc:
            logger.warning("retriever_node: search failed — %s", exc)
            results = []

    record_agent_run("retriever", time.monotonic() - t0)
    return {"retrieved_docs": results}


# ---------------------------------------------------------------------------
# Node: tool_node
# ---------------------------------------------------------------------------


def tool_node(state: CopilotState) -> dict:
    """Call the appropriate ML scoring engine based on context_type."""
    t0 = time.monotonic()

    if HAS_OPENTELEMETRY and _tracer:
        with _tracer.start_as_current_span("agent_graph.tool_node") as span:
            span.set_attribute("context_type", state.get("context_type", ""))
            span.set_attribute("session_id", state.get("session_id", ""))
            try:
                context = state.get("context_type", "")
                raw_features = state.get("features", {})

                if context == "underwriting" and _underwriting_engine is not None and UnderwritingFeatures is not None:
                    try:
                        features = UnderwritingFeatures(**raw_features)
                        result = _underwriting_engine.predict(features)
                        record_tool_call("underwriting")
                        span.set_attribute("tool_type", "underwriting")
                        return {"model_result": result.model_dump()}
                    except Exception as exc:
                        logger.warning("tool_node: underwriting scoring failed — %s", exc)
                        span.set_status(Status(StatusCode.ERROR, str(exc)))
                        return {"model_result": {}}

                if context == "claims" and _fraud_engine is not None and ClaimFeatures is not None:
                    try:
                        features = ClaimFeatures(**raw_features)
                        result = _fraud_engine.predict(features)
                        record_tool_call("fraud_scoring")
                        span.set_attribute("tool_type", "fraud_scoring")
                        return {"model_result": result.model_dump()}
                    except Exception as exc:
                        logger.warning("tool_node: fraud scoring failed — %s", exc)
                        span.set_status(Status(StatusCode.ERROR, str(exc)))
                        return {"model_result": {}}

                # No matching engine or unknown context type.
                span.set_status(Status(StatusCode.ERROR, "No matching engine"))
                return {"model_result": {}}

            except Exception as exc:
                logger.warning("tool_node: unexpected error — %s", exc)
                span.set_status(Status(StatusCode.ERROR, str(exc)))
                return {"model_result": {}}
            finally:
                record_agent_run("tool", time.monotonic() - t0)
    else:
        try:
            context = state.get("context_type", "")
            raw_features = state.get("features", {})

            if context == "underwriting" and _underwriting_engine is not None and UnderwritingFeatures is not None:
                try:
                    features = UnderwritingFeatures(**raw_features)
                    result = _underwriting_engine.predict(features)
                    record_tool_call("underwriting")
                    return {"model_result": result.model_dump()}
                except Exception as exc:
                    logger.warning("tool_node: underwriting scoring failed — %s", exc)
                    return {"model_result": {}}

            if context == "claims" and _fraud_engine is not None and ClaimFeatures is not None:
                try:
                    features = ClaimFeatures(**raw_features)
                    result = _fraud_engine.predict(features)
                    record_tool_call("fraud_scoring")
                    return {"model_result": result.model_dump()}
                except Exception as exc:
                    logger.warning("tool_node: fraud scoring failed — %s", exc)
                    return {"model_result": {}}

            # No matching engine or unknown context type.
            return {"model_result": {}}

        except Exception as exc:
            logger.warning("tool_node: unexpected error — %s", exc)
            return {"model_result": {}}
        finally:
            record_agent_run("tool", time.monotonic() - t0)


# ---------------------------------------------------------------------------
# Node: _rule_based_narrative (internal helper)
# ---------------------------------------------------------------------------


def _rule_based_narrative(state: CopilotState) -> str:
    """
    Assemble a plain-text decision narrative from retrieved docs and model output.
    Used when GROQ_API_KEY is absent or the LLM call fails.
    """
    lines: List[str] = [
        "=== ClaimGuard AI — Policy Copilot Decision Draft ===",
        "",
        f"Query: {state.get('query', 'N/A')}",
        f"Context type: {state.get('context_type', 'N/A').upper()}",
        "",
    ]

    # --- Retrieved policy / IRDAI evidence ---
    docs = state.get("retrieved_docs", [])
    if docs:
        lines.append("--- Relevant Policy / Regulatory Evidence ---")
        for i, doc in enumerate(docs[:3], start=1):
            source = doc.get("source", "unknown")
            score = doc.get("score", 0.0)
            chunk_id = doc.get("chunk_id", "unknown")
            snippet = doc.get("content", "")[:300].replace("\n", " ")
            lines.append(f"[{i}] [Chunk ID: {chunk_id}] Source: {source}  (relevance: {score:.3f})")
            lines.append(f"    \"{snippet}\"")
        lines.append("")
    else:
        lines.append("No matching policy / regulatory documents retrieved.")
        lines.append("")

    # --- Model scoring evidence ---
    model = state.get("model_result", {})
    if model:
        lines.append("--- Model Scoring Evidence ---")
        context = state.get("context_type", "")

        if context == "underwriting":
            lines.append(f"Risk Score : {model.get('risk_score', 'N/A')}")
            lines.append(f"Risk Tier  : {model.get('risk_tier', 'N/A').upper()}")
            lines.append(f"Premium Adj: {model.get('premium_adjustment', 'N/A')}")
        elif context == "claims":
            lines.append(f"Fraud Score      : {model.get('fraud_score', 'N/A')}")
            lines.append(f"Fraud Flag       : {model.get('fraud_flag', 'N/A')}")
            lines.append(f"Confidence Tier  : {model.get('confidence_tier', 'N/A').upper()}")

        shap_drivers = model.get("shap_drivers", [])
        if shap_drivers:
            lines.append("Top SHAP Drivers:")
            for driver in shap_drivers[:5]:
                feat = driver.get("feature", "?")
                val = driver.get("shap_value", 0.0)
                direction = "↑ risk" if val > 0 else "↓ risk"
                lines.append(f"  • {feat}: {val:+.4f} ({direction})")
        lines.append("")
    else:
        lines.append("Model scoring data unavailable for this request.")
        lines.append("")

    lines.append(
        "This draft has been queued for mandatory human analyst review per IRDAI guidelines."
    )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Node: writer_node
# ---------------------------------------------------------------------------


def writer_node(state: CopilotState) -> dict:
    """Synthesise a structured decision draft using Groq LLM or rule-based fallback."""
    t0 = time.monotonic()
    draft = ""

    if HAS_OPENTELEMETRY and _tracer:
        with _tracer.start_as_current_span("agent_graph.writer_node") as span:
            span.set_attribute("context_type", state.get("context_type", ""))
            span.set_attribute("session_id", state.get("session_id", ""))
            span.set_attribute("has_groq", HAS_GROQ and bool(os.environ.get("GROQ_API_KEY")))
            try:
                if HAS_GROQ and os.environ.get("GROQ_API_KEY"):
                    span.set_attribute("generation_method", "llm")
                    # Build a structured prompt using enterprise system prompt
                    docs = state.get("retrieved_docs", [])[:2]
                    doc_snippets = "\n".join(
                        f"[Chunk ID: {d.get('chunk_id', 'unknown')}]\n"
                        f"Source: {d.get('source', 'unknown')}\n"
                        f"Relevance: {d.get('score', 0.0):.2f}\n"
                        f"Content: {d.get('content', '')[:500]}"
                        for d in docs
                    )
                    model_result = state.get("model_result", {})

                    # Use enterprise system prompt if available
                    system_prompt = get_system_prompt() if HAS_SYSTEM_PROMPT else (
                        "You are the ClaimGuard AI Policy Copilot, an expert insurance analyst assistant."
                    )

                    prompt = (
                        f"{system_prompt}\n\n"
                        f"QUERY: {state.get('query', '')}\n\n"
                        f"CONTEXT TYPE: {state.get('context_type', '').upper()}\n\n"
                        "RETRIEVED POLICY / IRDAI EVIDENCE:\n"
                        f"{doc_snippets if doc_snippets else 'No relevant documents retrieved.'}\n\n"
                        "MODEL SCORING EVIDENCE:\n"
                        f"{model_result if model_result else 'No model result available.'}\n\n"
                        "Generate a structured decision draft following the output format specified in the system prompt. "
                        "Ensure you include chunk IDs for all citations and set appropriate confidence scores."
                    )

                    llm = ChatGroq(model="llama3-8b-8192", temperature=0)
                    response = llm.invoke([HumanMessage(content=prompt)])
                    draft = response.content
                else:
                    span.set_attribute("generation_method", "rule_based")
                    draft = _rule_based_narrative(state)

            except Exception as exc:
                logger.warning("writer_node: LLM call failed (%s), using rule-based fallback.", exc)
                span.set_status(Status(StatusCode.ERROR, str(exc)))
                draft = _rule_based_narrative(state)

            # ---------------------------------------------------------------------------
            # IRDAI Compliance Validation with Guardrails AI
            # ---------------------------------------------------------------------------
            if HAS_GUARDRAILS and get_irdai_validator is not None:
                try:
                    # Get validator with claim type
                    model_result = state.get("model_result", {})
                    features = state.get("features", {})
                    claim_type = str(
                        model_result.get("claim_type")
                        or features.get("claim_type")
                        or model_result.get("insurance_type")
                        or features.get("insurance_type")
                        or "motor"
                    ).lower()
                    validator = get_irdai_validator(claim_type=claim_type)

                    # Validate with retrieved docs and model result
                    validation_result = validator.validate_draft(
                        draft,
                        retrieved_docs=state.get("retrieved_docs"),
                        model_result=state.get("model_result")
                    )

                    span.set_attribute("guardrails_enabled", True)
                    span.set_attribute("guardrails_is_valid", validation_result["is_valid"])
                    span.set_attribute("guardrails_hitl_required", validation_result.get("hitl_required", False))

                    if not validation_result["is_valid"]:
                        logger.warning(
                            "writer_node: IRDAI compliance validation failed for draft. "
                            "Validation result: %s",
                            validation_result["validation_result"]
                        )
                        # Append compliance warning to the draft
                        draft = str(validation_result.get("filtered_output") or draft)
                        draft += (
                            "\n\n[COMPLIANCE HOLD: This draft did not pass validation and "
                            "must not be used for an automated decision. Route to a human analyst. "
                            f"Details: {validation_result.get('hitl_reason', 'Validation failed')} ]"
                        )
                    else:
                        logger.info("writer_node: IRDAI compliance validation passed")

                    # Update state with HITL requirement from validation
                    if validation_result.get("hitl_required"):
                        state["requires_human_review"] = True

                except Exception as exc:
                    logger.warning("writer_node: Guardrails validation failed: %s", exc)
                    span.set_status(Status(StatusCode.ERROR, f"Guardrails validation failed: {exc}"))
                    state["requires_human_review"] = True
                    draft += (
                        "\n\n[COMPLIANCE HOLD: Validation could not be completed. "
                        "Do not use this draft for an automated decision; route to a human analyst.]"
                    )

            record_decision_drafted()
            record_agent_run("writer", time.monotonic() - t0)
            return {"decision_draft": draft}
    else:
        try:
            if HAS_GROQ and os.environ.get("GROQ_API_KEY"):
                # Build a structured prompt using enterprise system prompt
                docs = state.get("retrieved_docs", [])[:2]
                doc_snippets = "\n".join(
                    f"[Chunk ID: {d.get('chunk_id', 'unknown')}]\n"
                    f"Source: {d.get('source', 'unknown')}\n"
                    f"Relevance: {d.get('score', 0.0):.2f}\n"
                    f"Content: {d.get('content', '')[:500]}"
                    for d in docs
                )
                model_result = state.get("model_result", {})

                # Use enterprise system prompt if available
                system_prompt = get_system_prompt() if HAS_SYSTEM_PROMPT else (
                    "You are the ClaimGuard AI Policy Copilot, an expert insurance analyst assistant."
                )

                prompt = (
                    f"{system_prompt}\n\n"
                    f"QUERY: {state.get('query', '')}\n\n"
                    f"CONTEXT TYPE: {state.get('context_type', '').upper()}\n\n"
                    "RETRIEVED POLICY / IRDAI EVIDENCE:\n"
                    f"{doc_snippets if doc_snippets else 'No relevant documents retrieved.'}\n\n"
                    "MODEL SCORING EVIDENCE:\n"
                    f"{model_result if model_result else 'No model result available.'}\n\n"
                    "Generate a structured decision draft following the output format specified in the system prompt. "
                    "Ensure you include chunk IDs for all citations and set appropriate confidence scores."
                )

                llm = ChatGroq(model="llama3-8b-8192", temperature=0)
                response = llm.invoke([HumanMessage(content=prompt)])
                draft = response.content
            else:
                draft = _rule_based_narrative(state)

        except Exception as exc:
            logger.warning("writer_node: LLM call failed (%s), using rule-based fallback.", exc)
            draft = _rule_based_narrative(state)

        # ---------------------------------------------------------------------------
        # IRDAI Compliance Validation with Guardrails AI
        # ---------------------------------------------------------------------------
        if HAS_GUARDRAILS and get_irdai_validator is not None:
            try:
                # Get validator with claim type
                model_result = state.get("model_result", {})
                features = state.get("features", {})
                claim_type = str(
                    model_result.get("claim_type")
                    or features.get("claim_type")
                    or model_result.get("insurance_type")
                    or features.get("insurance_type")
                    or "motor"
                ).lower()
                validator = get_irdai_validator(claim_type=claim_type)

                # Validate with retrieved docs and model result
                validation_result = validator.validate_draft(
                    draft,
                    retrieved_docs=state.get("retrieved_docs"),
                    model_result=state.get("model_result")
                )

                if not validation_result["is_valid"]:
                    logger.warning(
                        "writer_node: IRDAI compliance validation failed for draft. "
                        "Validation result: %s",
                        validation_result["validation_result"]
                    )
                    # Append compliance warning to the draft
                    draft = str(validation_result.get("filtered_output") or draft)
                    draft += (
                        "\n\n[COMPLIANCE HOLD: This draft did not pass validation and "
                        "must not be used for an automated decision. Route to a human analyst. "
                        f"Details: {validation_result.get('hitl_reason', 'Validation failed')} ]"
                    )
                else:
                    logger.info("writer_node: IRDAI compliance validation passed")

                # Update state with HITL requirement from validation
                if validation_result.get("hitl_required"):
                    state["requires_human_review"] = True

            except Exception as exc:
                logger.warning("writer_node: Guardrails validation failed: %s", exc)
                state["requires_human_review"] = True
                draft += (
                    "\n\n[COMPLIANCE HOLD: Validation could not be completed. "
                    "Do not use this draft for an automated decision; route to a human analyst.]"
                )

        record_decision_drafted()
        record_agent_run("writer", time.monotonic() - t0)
        return {"decision_draft": draft}


# ---------------------------------------------------------------------------
# Node: hitl_router_node
# ---------------------------------------------------------------------------


def hitl_router_node(state: CopilotState) -> dict:
    """Enqueue the decision draft for mandatory human analyst review."""
    if HAS_OPENTELEMETRY and _tracer:
        with _tracer.start_as_current_span("agent_graph.hitl_router_node") as span:
            span.set_attribute("context_type", state.get("context_type", ""))
            span.set_attribute("session_id", state.get("session_id", ""))
            try:
                if hitl_queue is not None and HITLItem is not None:
                    item = HITLItem(
                        session_id=state.get("session_id", uuid.uuid4().hex),
                        context_type=state.get("context_type", "unknown"),
                        decision_draft=state.get("decision_draft", ""),
                        model_result=state.get("model_result", {}),
                    )
                    hitl_queue.enqueue(item)
                    span.set_attribute("enqueued", True)
                else:
                    span.set_status(Status(StatusCode.ERROR, "hitl_queue not available"))
                    span.set_attribute("enqueued", False)

                depth = hitl_queue.queue_depth() if hitl_queue is not None else 0
                set_hitl_queue_depth(depth)
                span.set_attribute("queue_depth", depth)

            except Exception as exc:
                logger.warning("hitl_router_node: failed to enqueue HITL item — %s", exc)
                span.set_status(Status(StatusCode.ERROR, str(exc)))

            return {"requires_human_review": True}
    else:
        try:
            if hitl_queue is not None and HITLItem is not None:
                item = HITLItem(
                    session_id=state.get("session_id", uuid.uuid4().hex),
                    context_type=state.get("context_type", "unknown"),
                    decision_draft=state.get("decision_draft", ""),
                    model_result=state.get("model_result", {}),
                )
                hitl_queue.enqueue(item)

            depth = hitl_queue.queue_depth() if hitl_queue is not None else 0
            set_hitl_queue_depth(depth)

        except Exception as exc:
            logger.warning("hitl_router_node: failed to enqueue HITL item — %s", exc)

        return {"requires_human_review": True}


# ---------------------------------------------------------------------------
# LangGraph graph wiring
# ---------------------------------------------------------------------------


def _build_graph():  # type: ignore[return]
    """Compile the LangGraph StateGraph for the Policy Copilot pipeline."""
    from langgraph.graph import StateGraph, END  # noqa: F811  (re-import for clarity)

    graph = StateGraph(CopilotState)
    graph.add_node("retriever", retriever_node)
    graph.add_node("tool", tool_node)
    graph.add_node("writer", writer_node)
    graph.add_node("hitl_router", hitl_router_node)

    graph.set_entry_point("retriever")
    graph.add_edge("retriever", "tool")
    graph.add_edge("tool", "writer")
    graph.add_edge("writer", "hitl_router")
    graph.add_edge("hitl_router", END)

    return graph.compile()


if HAS_LANGGRAPH:
    try:
        _compiled_graph = _build_graph()
    except Exception as exc:
        logger.warning("agent_graph: LangGraph compile failed (%s); will use fallback.", exc)
        _compiled_graph = None
else:
    _compiled_graph = None


# ---------------------------------------------------------------------------
# Fallback: linear pipeline (no LangGraph required)
# ---------------------------------------------------------------------------


def run_copilot_fallback(state: dict) -> dict:
    """
    Execute all four pipeline nodes sequentially, merging their update dicts
    back into *state* after each step.  Always returns the full state dict.
    """
    nodes = [retriever_node, tool_node, writer_node, hitl_router_node]
    for node_fn in nodes:
        try:
            update = node_fn(state)  # type: ignore[arg-type]
            state.update(update)
        except Exception as exc:
            logger.warning("run_copilot_fallback: node %s raised %s", node_fn.__name__, exc)
    return state


# ---------------------------------------------------------------------------
# Public entry-point
# ---------------------------------------------------------------------------


def run_copilot(
    query: str,
    context_type: str,
    features: dict,
    session_id: str | None = None,
) -> dict:
    """
    Run the Policy Copilot pipeline for a single query.

    Parameters
    ----------
    query        : Natural-language question or case description.
    context_type : 'underwriting' or 'claims'.
    features     : Dict of model input features (will be validated inside tool_node).
    session_id   : Optional caller-supplied session identifier.

    Returns
    -------
    Full CopilotState dict with all fields populated (or gracefully degraded).
    """
    state: CopilotState = {
        "query": query,
        "context_type": context_type,
        "features": features,
        "retrieved_docs": [],
        "model_result": {},
        "decision_draft": "",
        "requires_human_review": False,
        "session_id": session_id or uuid.uuid4().hex,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    if HAS_OPENTELEMETRY and _tracer:
        with _tracer.start_as_current_span("agent_graph.run_copilot") as span:
            span.set_attribute("query", query[:200])
            span.set_attribute("context_type", context_type)
            span.set_attribute("session_id", state["session_id"])
            span.set_attribute("has_langgraph", _compiled_graph is not None)

            if _compiled_graph is not None:
                try:
                    result = _compiled_graph.invoke(state)
                    span.set_attribute("execution_mode", "langgraph")
                    return dict(result)
                except Exception as exc:
                    logger.warning(
                        "run_copilot: LangGraph invoke failed (%s); falling back to linear pipeline.", exc
                    )
                    span.set_status(Status(StatusCode.ERROR, f"LangGraph failed: {exc}"))
                    span.set_attribute("execution_mode", "fallback")
            else:
                span.set_attribute("execution_mode", "fallback")

            return run_copilot_fallback(state)
    else:
        if _compiled_graph is not None:
            try:
                result = _compiled_graph.invoke(state)
                return dict(result)
            except Exception as exc:
                logger.warning(
                    "run_copilot: LangGraph invoke failed (%s); falling back to linear pipeline.", exc
                )

        return run_copilot_fallback(state)
