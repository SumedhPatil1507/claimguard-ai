"""Quick smoke check for src/decision_json.py — run: python scripts/_smoke_decision.py"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.decision_json import (  # noqa: E402
    build_batch_decision,
    build_copilot_decision,
    build_fraud_decision,
    build_underwriting_decision,
)

d1 = build_fraud_decision({
    "claim_id": "CLM-1",
    "fraud_score": 0.82,
    "fraud_flag": True,
    "confidence_tier": "high",
    "shap_drivers": [{"feature": "claim_amount", "shap_value": 0.41}],
    "claim_amount": 450000,
})
d2 = build_underwriting_decision({"risk_tier": "low", "risk_score": 0.22,
                                  "premium_adjustment": 0.95, "shap_drivers": []})
d3 = build_batch_decision("claims", processed=100, flagged=3, errors=1)
d4 = build_copilot_decision({"coverage_status": "requires_review",
                             "confidence_score": 0.72,
                             "requires_human_review": True,
                             "recommended_payout": 0.0},
                             retrieved_docs=[{"chunk_id": "pol_1", "score": 0.91}])

for name, d in [("fraud", d1), ("underwriting", d2), ("batch", d3), ("copilot", d4)]:
    payload = d.model_dump()
    assert set(payload) == {"verdict", "reasoning", "recommendation", "next_steps"}, name
    assert len(payload["next_steps"]) >= 3, name
    print(f"[{name}] OK -> {payload['verdict']}")

print(json.dumps(d1.model_dump(), indent=2, ensure_ascii=False))
