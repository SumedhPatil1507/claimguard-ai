// ─── Auth ────────────────────────────────────────────────────────────────────
export type Role = "admin" | "analyst" | "viewer";

export interface TokenResponse {
  access_token: string;
  token_type: string;
  role: Role;
  expires_in: number;
}

export interface AuthUser {
  username: string;
  role: Role;
  token: string;
}

// ─── Task queue ──────────────────────────────────────────────────────────────
export type TaskStatus = "PENDING" | "STARTED" | "SUCCESS" | "FAILURE" | "RETRY";

export interface TaskEnqueueResponse {
  task_id: string;
  status: TaskStatus;
  status_url: string;
  message: string;
}

export interface TaskStatusResponse {
  task_id: string;
  status: TaskStatus;
  result?: UnderwritingResult | FraudScoringResult;
  error?: string;
  error_type?: string;
  progress?: Record<string, unknown>;
}

// ─── Underwriting ────────────────────────────────────────────────────────────
export interface UnderwritingFeatures {
  age: number;
  annual_income: number;
  credit_score: number;
  sum_insured: number;
  coverage_type: string;
  num_dependents: number;
  prior_claims_count: number;
  region: string;
  occupation: string;
}

export interface ShapDriver {
  feature: string;
  shap_value: number;
  direction: "increases_risk" | "decreases_risk";
}

export interface UnderwritingResult {
  risk_tier: "low" | "medium" | "high";
  risk_score: number;
  premium_adjustment: number;
  shap_drivers: ShapDriver[];
  model_version: string;
  timestamp: string;
}

// ─── Fraud scoring ───────────────────────────────────────────────────────────
export interface ClaimFeatures {
  claim_id: string;
  claimant_id: string;
  policy_id: string;
  claim_amount: number;
  days_since_policy_start: number;
  num_prior_claims: number;
  claim_type: string;
  claim_severity: string;
  repair_shop_id?: string;
  medical_provider_id?: string;
}

export interface FraudScoringResult {
  claim_id: string;
  fraud_score: number;
  fraud_flag: boolean;
  confidence_tier: "low" | "medium" | "high";
  shap_drivers: ShapDriver[];
  model_version: string;
  timestamp: string;
}

// ─── Graph ───────────────────────────────────────────────────────────────────
export interface CollusionRing {
  ring_id: string;
  claimant_ids: string[];
  shared_entities: string[];
  centrality_score: number;
  severity: "low" | "medium" | "high";
  gnn_scores: Record<string, number>;
  max_gnn_score: number;
}

// ─── HITL ────────────────────────────────────────────────────────────────────
export type HITLStatus = "pending" | "approved" | "rejected" | "escalated";

export interface HITLItem {
  item_id: string;
  session_id: string;
  context_type: "underwriting" | "claims";
  decision_draft: string;
  model_result: Record<string, unknown>;
  analyst_review: string | null;
  status: HITLStatus;
  created_at: string;
  reviewed_at: string | null;
}

// ─── Explorer ────────────────────────────────────────────────────────────────
export interface ClaimRecord {
  claim_id: string;
  policy_id: string;
  claimant_id: string;
  claim_amount: number;
  claim_date: string;
  claim_type: string;
  num_prior_claims: number;
  claim_severity: string;
  fraud_label: 0 | 1;
  repair_shop_id?: string;
  medical_provider_id?: string;
}
