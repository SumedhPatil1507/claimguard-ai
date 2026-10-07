"use client";

import { useState } from "react";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { z } from "zod";
import { Shield, Loader2, RefreshCw, CheckCircle2, XCircle } from "lucide-react";
import { enqueueUnderwriting } from "@/lib/api";
import { useTaskPoller } from "@/hooks/useTaskPoller";
import { ScoreGauge } from "@/components/charts/score-gauge";
import { ShapChart } from "@/components/charts/shap-chart";
import { Badge } from "@/components/ui/badge";
import { formatCurrency } from "@/lib/utils";
import { cn } from "@/lib/utils";
import type { UnderwritingResult } from "@/types";

const schema = z.object({
  age:                z.coerce.number().int().min(18).max(85),
  annual_income:      z.coerce.number().positive(),
  credit_score:       z.coerce.number().int().min(300).max(900),
  sum_insured:        z.coerce.number().positive(),
  coverage_type:      z.enum(["motor", "health", "property", "life"]),
  num_dependents:     z.coerce.number().int().min(0).max(10),
  prior_claims_count: z.coerce.number().int().min(0),
  region:             z.enum(["north", "south", "east", "west", "central"]),
  occupation:         z.enum(["salaried", "self-employed", "business", "retired", "student"]),
});
type FormValues = z.infer<typeof schema>;

const FIELD_CLASS = "w-full bg-background border border-border rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary/50 transition";
const SELECT_OPTS: Record<string, string[]> = {
  coverage_type: ["motor", "health", "property", "life"],
  region:        ["north", "south", "east", "west", "central"],
  occupation:    ["salaried", "self-employed", "business", "retired", "student"],
};

export default function UnderwritingPage() {
  const [enqueueError, setEnqueueError] = useState<string | null>(null);
  const { taskResult, taskError, isPolling, startPolling, reset } = useTaskPoller();

  const { register, handleSubmit, formState: { errors, isSubmitting } } = useForm<FormValues>({
    resolver: zodResolver(schema),
    defaultValues: {
      age: 35, annual_income: 800000, credit_score: 720, sum_insured: 1000000,
      coverage_type: "motor", num_dependents: 2, prior_claims_count: 0,
      region: "north", occupation: "salaried",
    },
  });

  async function onSubmit(data: FormValues) {
    setEnqueueError(null);
    reset();
    try {
      const resp = await enqueueUnderwriting(data);
      startPolling(resp.task_id);
    } catch (e: unknown) {
      setEnqueueError(e instanceof Error ? e.message : "Enqueue failed");
    }
  }

  const result = taskResult?.result as UnderwritingResult | undefined;
  const tier = result?.risk_tier;

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold flex items-center gap-2">
          <Shield className="w-6 h-6 text-primary" /> Underwriting Risk Scorer
        </h1>
        <p className="text-muted-foreground text-sm mt-1">
          Score applicants at policy-issuance time · Results run asynchronously via Celery
        </p>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
        {/* Form */}
        <div className="bg-card border border-border rounded-xl p-6">
          <h2 className="text-sm font-semibold mb-5">Applicant Features</h2>
          <form onSubmit={handleSubmit(onSubmit)} className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              {/* Age */}
              <div>
                <label className="text-xs text-muted-foreground mb-1 block">Age</label>
                <input type="number" {...register("age")} className={cn(FIELD_CLASS, errors.age && "border-destructive")} />
                {errors.age && <p className="text-destructive text-[10px] mt-0.5">{errors.age.message}</p>}
              </div>
              {/* Credit score */}
              <div>
                <label className="text-xs text-muted-foreground mb-1 block">Credit Score</label>
                <input type="number" {...register("credit_score")} className={cn(FIELD_CLASS, errors.credit_score && "border-destructive")} />
                {errors.credit_score && <p className="text-destructive text-[10px] mt-0.5">{errors.credit_score.message}</p>}
              </div>
              {/* Annual income */}
              <div>
                <label className="text-xs text-muted-foreground mb-1 block">Annual Income (₹)</label>
                <input type="number" {...register("annual_income")} className={cn(FIELD_CLASS, errors.annual_income && "border-destructive")} />
              </div>
              {/* Sum insured */}
              <div>
                <label className="text-xs text-muted-foreground mb-1 block">Sum Insured (₹)</label>
                <input type="number" {...register("sum_insured")} className={cn(FIELD_CLASS, errors.sum_insured && "border-destructive")} />
              </div>
              {/* Dependents */}
              <div>
                <label className="text-xs text-muted-foreground mb-1 block">Dependents</label>
                <input type="number" {...register("num_dependents")} className={FIELD_CLASS} />
              </div>
              {/* Prior claims */}
              <div>
                <label className="text-xs text-muted-foreground mb-1 block">Prior Claims</label>
                <input type="number" {...register("prior_claims_count")} className={FIELD_CLASS} />
              </div>
              {/* Selects */}
              {(["coverage_type", "region", "occupation"] as const).map((f) => (
                <div key={f} className="col-span-2 sm:col-span-1">
                  <label className="text-xs text-muted-foreground mb-1 block capitalize">{f.replace("_", " ")}</label>
                  <select {...register(f)} className={FIELD_CLASS}>
                    {SELECT_OPTS[f].map((o) => <option key={o} value={o}>{o}</option>)}
                  </select>
                </div>
              ))}
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
              {(isSubmitting || isPolling) ? <><Loader2 className="w-4 h-4 animate-spin" />{isPolling ? "Running inference…" : "Enqueueing…"}</> : "🔍 Score Risk"}
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

          {/* Idle */}
          {!isPolling && !taskResult && !taskError && (
            <div className="flex-1 flex items-center justify-center text-muted-foreground text-sm">
              Submit the form to score an applicant
            </div>
          )}

          {/* Polling */}
          {isPolling && (
            <div className="flex-1 flex flex-col items-center justify-center gap-3 text-muted-foreground">
              <div className="relative">
                <div className="w-12 h-12 rounded-full border-2 border-primary/30 border-t-primary animate-spin" />
              </div>
              <p className="text-sm">Running inference via Celery…</p>
            </div>
          )}

          {/* Error */}
          {taskError && (
            <div className="flex items-center gap-2 text-red-400 bg-red-400/10 border border-red-400/20 rounded-lg px-4 py-3 text-sm">
              <XCircle className="w-4 h-4 shrink-0" />{taskError}
            </div>
          )}

          {/* Success */}
          {result && !taskError && (
            <div className="space-y-5 animate-fade-in">
              {/* Tier + icon */}
              <div className="flex items-center gap-3">
                <CheckCircle2 className="w-5 h-5 text-primary shrink-0" />
                <div>
                  <p className="text-xs text-muted-foreground">Risk Tier</p>
                  <div className="flex items-center gap-2 mt-0.5">
                    <Badge variant="severity" severity={tier!}>{tier?.toUpperCase()}</Badge>
                    <span className="text-xs text-muted-foreground">v{result.model_version}</span>
                  </div>
                </div>
              </div>

              {/* Gauge + stats */}
              <div className="flex items-center gap-6">
                <ScoreGauge score={result.risk_score} label="Risk Score" tier={tier} />
                <div className="space-y-2.5">
                  <div>
                    <p className="text-xs text-muted-foreground">Premium Adjustment</p>
                    <p className="text-xl font-bold">{result.premium_adjustment.toFixed(2)}×</p>
                  </div>
                  <div>
                    <p className="text-xs text-muted-foreground">Raw Score</p>
                    <p className="font-mono text-sm">{(result.risk_score * 100).toFixed(2)}%</p>
                  </div>
                </div>
              </div>

              {/* SHAP drivers */}
              {result.shap_drivers.length > 0 && (
                <div>
                  <p className="text-xs font-medium text-muted-foreground mb-2">SHAP Feature Drivers</p>
                  <ShapChart drivers={result.shap_drivers} />
                  <p className="text-[10px] text-muted-foreground mt-1">🟥 increases risk · 🟩 decreases risk</p>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
