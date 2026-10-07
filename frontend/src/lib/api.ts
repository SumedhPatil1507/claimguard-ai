/**
 * ClaimGuard AI — Typed API Client for Next.js Frontend
 *
 * Automatically handles JWT Bearer authentication, polling,
 * backend health checks, and fallback states.
 */

import type {
  TokenResponse,
  TaskEnqueueResponse,
  TaskStatusResponse,
  UnderwritingFeatures,
  ClaimFeatures,
  CollusionRing,
  HITLItem,
  ClaimRecord,
} from "@/types";

const BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
const TOKEN_KEY = "cg_access_token";

// ── Token Storage ────────────────────────────────────────────────────────────

export function saveToken(token: string): void {
  if (typeof window !== "undefined") sessionStorage.setItem(TOKEN_KEY, token);
}

export function getToken(): string | null {
  if (typeof window === "undefined") return null;
  return sessionStorage.getItem(TOKEN_KEY);
}

export function clearToken(): void {
  if (typeof window !== "undefined") sessionStorage.removeItem(TOKEN_KEY);
}

// ── Core Fetch Wrapper ───────────────────────────────────────────────────────

async function apiFetch<T>(
  path: string,
  options: RequestInit = {},
  authenticated = true,
): Promise<T> {
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    ...(options.headers as Record<string, string> | undefined),
  };

  if (authenticated) {
    const token = getToken();
    if (!token) throw new ApiError(401, "Not authenticated. Please log in.");
    headers["Authorization"] = `Bearer ${token}`;
  }

  try {
    const res = await fetch(`${BASE}${path}`, { ...options, headers });

    if (!res.ok) {
      const body = await res.json().catch(() => ({ detail: res.statusText }));
      throw new ApiError(res.status, body?.detail ?? res.statusText);
    }

    if (res.status === 204) return undefined as T;
    return res.json() as Promise<T>;
  } catch (err: unknown) {
    if (err instanceof ApiError) throw err;
    const msg = err instanceof Error ? err.message : "Network error";
    throw new ApiError(503, `API connection failed: ${msg}`);
  }
}

export class ApiError extends Error {
  constructor(
    public readonly status: number,
    message: string,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

// ── Auth ─────────────────────────────────────────────────────────────────────

export async function login(username: string, password: string): Promise<TokenResponse> {
  return apiFetch<TokenResponse>(
    "/auth/token",
    { method: "POST", body: JSON.stringify({ username, password }) },
    false,
  );
}

export async function getMe(): Promise<{ role: string; authenticated: boolean }> {
  return apiFetch("/auth/me");
}

// ── Health ───────────────────────────────────────────────────────────────────

export interface HealthResponse {
  status: string;
  version?: string;
  timestamp?: string;
  subsystems?: Record<string, string>;
}

export async function getHealth(): Promise<HealthResponse> {
  try {
    return await apiFetch<HealthResponse>("/health", {}, false);
  } catch {
    return { status: "offline", version: "unknown", timestamp: new Date().toISOString() };
  }
}

// ── Underwriting ─────────────────────────────────────────────────────────────

export async function enqueueUnderwriting(
  features: UnderwritingFeatures,
): Promise<TaskEnqueueResponse> {
  return apiFetch<TaskEnqueueResponse>("/underwrite", {
    method: "POST",
    body: JSON.stringify(features),
  });
}

// ── Claims / Fraud ────────────────────────────────────────────────────────────

export async function enqueueFraudScore(
  features: ClaimFeatures,
): Promise<TaskEnqueueResponse> {
  return apiFetch<TaskEnqueueResponse>("/claims/score", {
    method: "POST",
    body: JSON.stringify(features),
  });
}

// ── Task Polling ─────────────────────────────────────────────────────────────

export async function getTaskStatus(taskId: string): Promise<TaskStatusResponse> {
  return apiFetch<TaskStatusResponse>(`/tasks/${taskId}`);
}

/** Poll every `intervalMs` until status is SUCCESS or FAILURE (max `maxAttempts`). */
export async function pollTask(
  taskId: string,
  intervalMs = 1500,
  maxAttempts = 40,
): Promise<TaskStatusResponse> {
  for (let i = 0; i < maxAttempts; i++) {
    const status = await getTaskStatus(taskId);
    if (status.status === "SUCCESS" || status.status === "FAILURE") return status;
    await new Promise((r) => setTimeout(r, intervalMs));
  }
  throw new ApiError(408, "Task timed out after polling.");
}

// ── Graph ────────────────────────────────────────────────────────────────────

export async function getCollusionRings(): Promise<{ rings: CollusionRing[] }> {
  return apiFetch<{ rings: CollusionRing[] }>("/graph/collusion-rings");
}

// ── HITL ─────────────────────────────────────────────────────────────────────

export async function getHITLQueue(): Promise<{ items: HITLItem[] }> {
  return apiFetch<{ items: HITLItem[] }>("/hitl/queue").catch(() => ({ items: [] }));
}

export async function reviewHITLItem(
  itemId: string,
  decision: "approved" | "rejected" | "escalated",
  notes: string,
): Promise<void> {
  return apiFetch(`/hitl/review/${itemId}`, {
    method: "POST",
    body: JSON.stringify({ decision, analyst_notes: notes }),
  });
}

// ── Policy Copilot & Compliance ──────────────────────────────────────────────

export async function runCopilotDecide(
  query: string,
  context_type: "underwriting" | "claims",
  features: Record<string, unknown> = {},
): Promise<Record<string, unknown>> {
  return apiFetch("/copilot/decide", {
    method: "POST",
    body: JSON.stringify({ query, context_type, features }),
  });
}

export async function getComplianceReport(): Promise<Record<string, unknown>> {
  return apiFetch("/compliance");
}

// ── Explorer Data ────────────────────────────────────────────────────────────

export async function getClaimsData(): Promise<ClaimRecord[]> {
  try {
    const res = await fetch("/api/data/claims");
    if (!res.ok) return [];
    return res.json() as Promise<ClaimRecord[]>;
  } catch {
    return [];
  }
}
