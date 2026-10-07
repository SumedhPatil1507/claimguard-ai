"use client";

import { useCallback, useEffect, useState } from "react";
import {
  Users, CheckCircle2, XCircle, ArrowUpCircle, RefreshCw, Inbox, Loader2, Clock,
} from "lucide-react";
import { getHITLQueue, reviewHITLItem, ApiError } from "@/lib/api";
import { Badge } from "@/components/ui/badge";
import { StatCard } from "@/components/ui/stat-card";
import { cn } from "@/lib/utils";
import type { HITLItem, HITLStatus } from "@/types";

const STATUS_STYLE: Record<HITLStatus, string> = {
  pending:   "bg-amber-500/10  text-amber-400  border-amber-500/30",
  approved:  "bg-emerald-500/10 text-emerald-400 border-emerald-500/30",
  rejected:  "bg-red-500/10    text-red-400    border-red-500/30",
  escalated: "bg-purple-500/10 text-purple-400  border-purple-500/30",
};

function ReviewCard({ item, onAction }: { item: HITLItem; onAction: () => void }) {
  const [notes, setNotes] = useState("");
  const [acting, setActing] = useState<string | null>(null);
  const [error, setError]  = useState<string | null>(null);

  async function act(decision: "approved" | "rejected" | "escalated") {
    setActing(decision);
    setError(null);
    try {
      await reviewHITLItem(item.item_id, decision, notes);
      onAction();
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "Action failed");
      setActing(null);
    }
  }

  const isPending = item.status === "pending";

  return (
    <div className={cn(
      "bg-card border rounded-xl p-5 transition",
      isPending ? "border-amber-500/30" : "border-border opacity-70",
    )}>
      {/* Header */}
      <div className="flex items-start justify-between gap-3 mb-3">
        <div>
          <p className="text-xs font-mono text-muted-foreground">{item.item_id.slice(0, 16)}…</p>
          <div className="flex items-center gap-2 mt-1">
            <span className={cn("text-xs px-2 py-0.5 rounded-full border font-medium", STATUS_STYLE[item.status])}>
              {item.status}
            </span>
            <Badge className={item.context_type === "underwriting" ? "bg-blue-500/20 text-blue-400 border-blue-500/30" : "bg-orange-500/20 text-orange-400 border-orange-500/30"}>
              {item.context_type}
            </Badge>
          </div>
        </div>
        <div className="text-right">
          <p className="text-xs text-muted-foreground flex items-center gap-1 justify-end">
            <Clock className="w-3 h-3" />
            {new Date(item.created_at).toLocaleString()}
          </p>
          {item.reviewed_at && (
            <p className="text-[10px] text-muted-foreground mt-0.5">
              Reviewed: {new Date(item.reviewed_at).toLocaleString()}
            </p>
          )}
        </div>
      </div>

      {/* Decision draft */}
      <div className="bg-background rounded-lg p-3 text-xs text-muted-foreground mb-3 max-h-28 overflow-y-auto border border-border/50">
        {item.decision_draft || "No decision draft."}
      </div>

      {/* Model result snippet */}
      {Object.keys(item.model_result).length > 0 && (
        <div className="flex flex-wrap gap-3 mb-3">
          {["risk_score", "fraud_score"].map((key) => {
            const val = item.model_result[key];
            if (val === undefined) return null;
            return (
              <div key={key} className="bg-secondary rounded-lg px-3 py-1.5">
                <p className="text-[10px] text-muted-foreground">{key.replace("_", " ")}</p>
                <p className="text-sm font-bold">{(Number(val) * 100).toFixed(1)}%</p>
              </div>
            );
          })}
        </div>
      )}

      {isPending && (
        <>
          <textarea
            value={notes}
            onChange={(e) => setNotes(e.target.value)}
            placeholder="Analyst notes (optional)…"
            rows={2}
            className="w-full bg-background border border-border rounded-lg px-3 py-2 text-xs resize-none focus:outline-none focus:ring-2 focus:ring-primary/50 transition mb-3"
          />
          {error && <p className="text-red-400 text-xs mb-2">{error}</p>}
          <div className="grid grid-cols-3 gap-2">
            <button
              onClick={() => act("approved")}
              disabled={!!acting}
              className="flex items-center justify-center gap-1.5 py-2 rounded-lg bg-emerald-500/10 text-emerald-400 border border-emerald-500/30 text-xs font-semibold hover:bg-emerald-500/20 transition disabled:opacity-50"
            >
              {acting === "approved" ? <Loader2 className="w-3 h-3 animate-spin" /> : <CheckCircle2 className="w-3.5 h-3.5" />}
              Approve
            </button>
            <button
              onClick={() => act("rejected")}
              disabled={!!acting}
              className="flex items-center justify-center gap-1.5 py-2 rounded-lg bg-red-500/10 text-red-400 border border-red-500/30 text-xs font-semibold hover:bg-red-500/20 transition disabled:opacity-50"
            >
              {acting === "rejected" ? <Loader2 className="w-3 h-3 animate-spin" /> : <XCircle className="w-3.5 h-3.5" />}
              Reject
            </button>
            <button
              onClick={() => act("escalated")}
              disabled={!!acting}
              className="flex items-center justify-center gap-1.5 py-2 rounded-lg bg-purple-500/10 text-purple-400 border border-purple-500/30 text-xs font-semibold hover:bg-purple-500/20 transition disabled:opacity-50"
            >
              {acting === "escalated" ? <Loader2 className="w-3 h-3 animate-spin" /> : <ArrowUpCircle className="w-3.5 h-3.5" />}
              Escalate
            </button>
          </div>
        </>
      )}

      {!isPending && item.analyst_review && (
        <p className="text-xs text-muted-foreground italic">"{item.analyst_review}"</p>
      )}
    </div>
  );
}

export default function HITLPage() {
  const [items, setItems]   = useState<HITLItem[]>([]);
  const [loading, setLoading] = useState(false);
  const [filter, setFilter] = useState<"all" | HITLStatus>("all");

  const fetchQueue = useCallback(async () => {
    setLoading(true);
    try {
      const data = await getHITLQueue();
      setItems(data.items ?? []);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { fetchQueue(); }, [fetchQueue]);

  const pending   = items.filter((i) => i.status === "pending");
  const reviewed  = items.filter((i) => i.status !== "pending");
  const displayed = filter === "all" ? items : items.filter((i) => i.status === filter);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-start justify-between">
        <div>
          <h1 className="text-2xl font-bold flex items-center gap-2">
            <Users className="w-6 h-6 text-primary" /> HITL Review Queue
          </h1>
          <p className="text-muted-foreground text-sm mt-1">
            All AI-drafted decisions require human analyst review — no auto-approval
          </p>
        </div>
        <button
          onClick={fetchQueue}
          disabled={loading}
          className="flex items-center gap-2 px-4 py-2 rounded-lg border border-border text-sm hover:bg-secondary/50 transition disabled:opacity-50"
        >
          <RefreshCw className={cn("w-4 h-4", loading && "animate-spin")} />
          Refresh
        </button>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard label="Pending Reviews"   value={pending.length}  icon={Clock}         highlight={pending.length > 0} />
        <StatCard label="Approved"          value={items.filter((i) => i.status === "approved").length}  icon={CheckCircle2} />
        <StatCard label="Rejected"          value={items.filter((i) => i.status === "rejected").length}  icon={XCircle} />
        <StatCard label="Escalated"         value={items.filter((i) => i.status === "escalated").length} icon={ArrowUpCircle} />
      </div>

      {/* Filter tabs */}
      <div className="flex items-center gap-2 border-b border-border pb-3">
        {(["all", "pending", "approved", "rejected", "escalated"] as const).map((f) => (
          <button
            key={f}
            onClick={() => setFilter(f)}
            className={cn(
              "text-xs px-3 py-1.5 rounded-lg capitalize transition",
              filter === f ? "bg-primary/10 text-primary font-semibold" : "text-muted-foreground hover:text-foreground hover:bg-secondary/50",
            )}
          >
            {f} {f === "all" ? `(${items.length})` : `(${items.filter((i) => i.status === f).length})`}
          </button>
        ))}
      </div>

      {/* Empty state */}
      {!loading && displayed.length === 0 && (
        <div className="flex flex-col items-center justify-center gap-3 h-48 text-muted-foreground">
          <Inbox className="w-10 h-10 opacity-30" />
          <p className="text-sm">
            {filter === "pending" ? "Queue is empty — all reviews complete ✅" : `No ${filter} items.`}
          </p>
          <p className="text-xs text-center max-w-xs">
            Use the Policy Copilot or submit claims/policies to generate decisions for review.
          </p>
        </div>
      )}

      {/* Cards */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {displayed.map((item) => (
          <ReviewCard key={item.item_id} item={item} onAction={fetchQueue} />
        ))}
      </div>
    </div>
  );
}
