"use client";

import { useState } from "react";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { z } from "zod";
import { AlertTriangle, Loader2, XCircle, CheckCircle2, RefreshCw, ShieldAlert, ShieldCheck } from "lucide-react";
import { enqueueFraudScore } from "@/lib/api";
import { useTaskPoller } from "@/hooks/useTaskPoller";
import { ScoreGauge } from "@/components/charts/score-gauge";
import { ShapChart } from "@/components/charts/shap-chart";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import type { FraudScoringResult } from "@/types";

const schema = z.object({
  claim_id:                z.string().min(1),
  claimant_id:             z.string().min(1),
  policy_id:               z.string().min(1),
  claim_amount:            z.coerce.number().positive(),
  days_since_policy_start: z.coerce.number().int().min(0),
  num_prior_claims:        z.coerce.number().int().min(0),
  claim_type:              z.enum(["motor", "health", "property", "life"]),
  claim_severity:          z.enum(["low", "medium", "high"]),
  repair_shop_id:          z.string().optional(),
  medical_provider_id:     z.string().optional(),
});
type FormValues = z.infer<typeof schema>;

const F = "w-full bg-background border border-border rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary/50 transition";

export default function FraudPage() {
  const [enqueueError, setEnqueueError] = useState<string | null>(null);
  const { taskResult, taskError, isPolling, startPolling, reset } = useTaskPoller();

  const { register, handleSubmit, formState: { errors, isSubmitting } } = useForm<FormValues>({
    resolver: zodResolver(schema),
    defaultValues: {
      claim_id: "CLM-001", claimant_id: "CLT-001", policy_id: "POL-001",
      claim_amount: 75000, days_since_policy_start: 45, num_prior_claims: 2,
      claim_type: "motor", claim_severity: "high",
      repair_shop_id: "SHOP-001",
    },
  });

  async function onSubmit(data: FormValues) {
    setEnqueueError(null);
    reset();
    try {
      const resp = await enqueueFraudScore(data);
      startPolling(resp.task_id);
    } catch (e: unknown) {
      setEnqueueError(e instanceof Error ? e.message : "Enqueue failed");
    }
  }

  const result = taskResult?.result as FraudScoringResult | undefined;
  const isFraud  = result?.fraud_flag;
  const tier     = result?.confidence_tier;

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold flex items-center gap-2">
          <AlertTriangle className="w-6 h-6 text-primary" /> Claims Fraud Detection
        </h1>
        <p className="text-muted-foreground text-sm mt-1">
          Score claims at filing time for fraud probability
        </p>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
        {/* Form */}
        <div className="bg-card border border-border rounded-xl p-6">
          <h2 className="text-sm font-semibold mb-5">Claim Features</h2>
          <form onSubmit={handleSubmit(onSubmit)} className="space-y-4">
            <div className="grid grid-cols-2 gap-3">
              {(["claim_id", "claimant_id", "policy_id"] as const).map((f) => (
                <div key={f} className={f === "claim_id" ? "col-span-2" : ""}>
                  <label className="text-xs text-muted-foreground mb-1 block capitalize">{f.replace(/_/g, " ")}</label>
                  <input {...register(f)} className={cn(F, errors[f] && "border-destructive")} />
                </div>
              ))}
              <div>
                <label className="text-xs text-muted-foreground mb-1 block">Claim Amount (₹)</label>
                <input type="number" {...register("claim_amount")} className={cn(F, errors.claim_amount && "border-destructive")} />
              </div>
              <div>
                <label className="text-xs text-muted-foreground mb-1 block">Days Since Policy Start</label>
                <input type="number" {...register("days_since_policy_start")} className={F} />
              </div>
              <div>
                <label className="text-xs text-muted-foreground mb-1 block">Prior Claims</label>
                <input type="number" {...register("num_prior_claims")} className={F} />
              </div>
              <div>
                <label className="text-xs text-muted-foreground mb-1 block">Claim Type</label>
                <select {...register("claim_type")} className={F}>
                  {["motor", "health", "property", "life"].map((o) => <option key={o}>{o}</option>)}
                </select>
              </div>
              <div>
                <label className="text-xs text-muted-foreground mb-1 block">Claim Severity</label>
                <select {...register("claim_severity")} className={F}>
                  {["low", "medium", "high"].map((o) => <option key={o}>{o}</option>)}
                </select>
              </div>
              <div>
                <label className="text-xs text-muted-foreground mb-1 block">Repair Shop ID</label>
                <input {...register("repair_shop_id")} placeholder="Optional" className={F} />
              </div>
              <div>
                <label className="text-xs text-muted-foreground mb-1 block">Medical Provider ID</label>
                <input {...register("medical_provider_id")} placeholder="Optional" className={F} />
              </div>
            </div>

            {enqueueError && (
              <div className="flex items-center gap-2 text-red-400 bg-red-400/10 border border-red-400/20 rounded-lg px-3 py-2 text-sm">
                <XCircle className="w-4 h-4 shrink-0" />{enqueueError}
              </div>
            )}

            <button
              type="submit"
              disabled={isSubmitting || isPolling}
              className="w-full py-2.5 rounded-lg bg-primary text-primary-foreground font-semibold text-sm hover:bg-primary/90 transition disabled:opacity-60 flex items-center justify-center gap-2"
            >
              {(isSubmitting || isPolling) ? <><Loader2 className="w-4 h-4 animate-spin" />{isPolling ? "Scoring…" : "Enqueueing…"}</> : "🚨 Score Claim"}
            </button>
          </form>
        </div>

        {/* Result */}
        <div className="bg-card border border-border rounded-xl p-6 flex flex-col">
          <div className="flex items-center justify-between mb-5">
            <h2 className="text-sm font-semibold">Result</h2>
            {taskResult && (
              <button onClick={reset} className="text-xs text-muted-foreground hover:text-foreground flex items-center gap-1.5">
                <RefreshCw className="w-3 h-3" /> Reset
              </button>
            )}
          </div>

          {!isPolling && !taskResult && !taskError && (
            <div className="flex-1 flex items-center justify-center text-muted-foreground text-sm">Submit the form to score a claim</div>
          )}

          {isPolling && (
            <div className="flex-1 flex flex-col items-center justify-center gap-3 text-muted-foreground">
              <div className="w-12 h-12 rounded-full border-2 border-primary/30 border-t-primary animate-spin" />
              <p className="text-sm">Running fraud detection…</p>
            </div>
          )}

          {taskError && (
            <div className="flex items-center gap-2 text-red-400 bg-red-400/10 border border-red-400/20 rounded-lg px-4 py-3 text-sm">
              <XCircle className="w-4 h-4 shrink-0" />{taskError}
            </div>
          )}

          {result && !taskError && (
            <div className="space-y-5 animate-fade-in">
              {/* Fraud alert banner */}
              <div className={cn(
                "flex items-center gap-3 rounded-xl px-4 py-3 border",
                isFraud
                  ? "bg-red-500/10 border-red-500/30 text-red-400"
                  : "bg-emerald-500/10 border-emerald-500/30 text-emerald-400"
              )}>
                {isFraud
                  ? <ShieldAlert className="w-5 h-5 shrink-0" />
                  : <ShieldCheck className="w-5 h-5 shrink-0" />}
                <div>
                  <p className="font-semibold text-sm">{isFraud ? "⚠️ HIGH FRAUD RISK DETECTED" : "✅ Claim Appears Legitimate"}</p>
                  <p className="text-xs opacity-80">Confidence: {tier?.toUpperCase()} · All decisions require HITL review</p>
                </div>
              </div>

              {/* Gauge + stats */}
              <div className="flex items-center gap-6">
                <ScoreGauge
                  score={result.fraud_score}
                  label="Fraud Probability"
                  tier={result.fraud_score > 0.6 ? "high" : result.fraud_score > 0.3 ? "medium" : "low"}
                />
                <div className="space-y-2.5">
                  <div>
                    <p className="text-xs text-muted-foreground">Fraud Flag</p>
                    <div className="flex items-center gap-1.5 mt-0.5">
                      {isFraud
                        ? <><XCircle className="w-4 h-4 text-red-400" /> <span className="text-red-400 font-semibold text-sm">Flagged</span></>
                        : <><CheckCircle2 className="w-4 h-4 text-emerald-400" /> <span className="text-emerald-400 font-semibold text-sm">Clean</span></>
                      }
                    </div>
                  </div>
                  <div>
                    <p className="text-xs text-muted-foreground">Confidence</p>
                    <Badge variant="severity" severity={tier!}>{tier}</Badge>
                  </div>
                  <div>
                    <p className="text-xs text-muted-foreground">Model</p>
                    <p className="text-xs font-mono text-muted-foreground">v{result.model_version}</p>
                  </div>
                </div>
              </div>

              {/* SHAP */}
              {result.shap_drivers.length > 0 && (
                <div>
                  <p className="text-xs font-medium text-muted-foreground mb-2">SHAP Feature Drivers</p>
                  <ShapChart drivers={result.shap_drivers} />
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
