"use client";

import { useEffect, useState } from "react";
import { Network, RefreshCw, AlertTriangle, Loader2 } from "lucide-react";
import { getCollusionRings, ApiError } from "@/lib/api";
import { CollusionGraph } from "@/components/graph/collusion-graph";
import { StatCard } from "@/components/ui/stat-card";
import { Badge } from "@/components/ui/badge";
import { getSeverityBadgeClass } from "@/lib/utils";
import type { CollusionRing } from "@/types";

export default function GraphPage() {
  const [rings, setRings]   = useState<CollusionRing[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError]   = useState<string | null>(null);
  const [selected, setSelected] = useState<CollusionRing | null>(null);

  async function fetchRings() {
    setLoading(true);
    setError(null);
    try {
      const data = await getCollusionRings();
      setRings(data.rings ?? []);
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "Failed to load collusion data");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => { fetchRings(); }, []);

  const highRings = rings.filter((r) => r.severity === "high").length;
  const gnnScored = rings.some((r) => r.max_gnn_score > 0);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-start justify-between">
        <div>
          <h1 className="text-2xl font-bold flex items-center gap-2">
            <Network className="w-6 h-6 text-primary" /> Graph Intel
          </h1>
          <p className="text-muted-foreground text-sm mt-1">
            Two-stage collusion ring detection · Structural analysis + R-GCN GNN re-scoring
          </p>
        </div>
        <button
          onClick={fetchRings}
          disabled={loading}
          className="flex items-center gap-2 px-4 py-2 rounded-lg border border-border text-sm hover:bg-secondary/50 transition disabled:opacity-50"
        >
          <RefreshCw className={`w-4 h-4 ${loading ? "animate-spin" : ""}`} />
          Refresh
        </button>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard label="Rings Detected"   value={rings.length}  icon={Network} />
        <StatCard label="High Severity"    value={highRings}     icon={AlertTriangle} highlight={highRings > 0} />
        <StatCard label="Claimants"        value={[...new Set(rings.flatMap((r) => r.claimant_ids))].length} icon={Network} />
        <StatCard label="GNN Scored"       value={gnnScored ? "✅ Yes" : "⬜ No"} icon={Network} />
      </div>

      {/* Error */}
      {error && (
        <div className="bg-red-500/10 border border-red-500/30 rounded-xl px-4 py-3 text-red-400 text-sm flex items-center gap-2">
          <AlertTriangle className="w-4 h-4 shrink-0" />
          {error.includes("503") || error.includes("PostgreSQL")
            ? "PostgreSQL is required for the graph endpoint. Run Docker Compose or connect a database."
            : error}
        </div>
      )}

      {/* Loading */}
      {loading && !rings.length && (
        <div className="flex items-center justify-center h-40 text-muted-foreground gap-2">
          <Loader2 className="w-5 h-5 animate-spin text-primary" />
          Analysing collusion rings…
        </div>
      )}

      {/* Graph canvas */}
      {!loading && rings.length > 0 && (
        <>
          <div>
            <p className="text-xs text-muted-foreground mb-3">
              ⚪ Claimant nodes · 🟡 Shared entity nodes · Animated edges = high severity · Node size ∝ GNN risk score
            </p>
            <CollusionGraph rings={rings} />
          </div>

          {/* Ring summary table */}
          <div className="bg-card border border-border rounded-xl overflow-hidden">
            <div className="px-5 py-4 border-b border-border">
              <h3 className="text-sm font-semibold">Ring Summary</h3>
            </div>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-border text-muted-foreground text-xs">
                    {["Ring ID", "Severity", "Claimants", "Shared Entities", "Centrality", "Max GNN Score"].map((h) => (
                      <th key={h} className="text-left px-4 py-3 font-medium">{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {rings.map((ring) => (
                    <tr
                      key={ring.ring_id}
                      onClick={() => setSelected(selected?.ring_id === ring.ring_id ? null : ring)}
                      className="border-b border-border/50 hover:bg-secondary/20 cursor-pointer transition"
                    >
                      <td className="px-4 py-2.5 font-mono text-xs text-muted-foreground">{ring.ring_id}</td>
                      <td className="px-4 py-2.5">
                        <Badge variant="severity" severity={ring.severity}>{ring.severity}</Badge>
                      </td>
                      <td className="px-4 py-2.5">{ring.claimant_ids.length}</td>
                      <td className="px-4 py-2.5">{ring.shared_entities.length}</td>
                      <td className="px-4 py-2.5 font-mono text-xs">{ring.centrality_score.toFixed(3)}</td>
                      <td className="px-4 py-2.5">
                        {ring.max_gnn_score > 0 ? (
                          <span className={ring.max_gnn_score >= 0.8 ? "text-red-400 font-bold" : ring.max_gnn_score >= 0.6 ? "text-amber-400 font-semibold" : "text-muted-foreground"}>
                            {(ring.max_gnn_score * 100).toFixed(1)}%
                          </span>
                        ) : "—"}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Selected ring detail */}
          {selected && (
            <div className="bg-card border border-primary/30 rounded-xl p-5 animate-fade-in">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-sm font-semibold">Ring Detail · <span className="font-mono text-muted-foreground">{selected.ring_id}</span></h3>
                <button onClick={() => setSelected(null)} className="text-muted-foreground hover:text-foreground text-xs">✕ Close</button>
              </div>
              <div className="grid grid-cols-2 gap-6">
                <div>
                  <p className="text-xs font-medium text-muted-foreground mb-2">Claimants</p>
                  <ul className="space-y-1.5">
                    {selected.claimant_ids.map((cid) => {
                      const score = selected.gnn_scores?.[cid];
                      return (
                        <li key={cid} className="flex items-center justify-between text-xs">
                          <span className="font-mono text-muted-foreground">{cid}</span>
                          {score !== undefined && (
                            <span className={score >= 0.5 ? "text-red-400 font-semibold" : "text-emerald-400"}>
                              {score >= 0.5 ? "🚨" : "✅"} {(score * 100).toFixed(1)}%
                            </span>
                          )}
                        </li>
                      );
                    })}
                  </ul>
                </div>
                <div>
                  <p className="text-xs font-medium text-muted-foreground mb-2">Shared Entities</p>
                  <ul className="space-y-1.5">
                    {selected.shared_entities.map((eid) => (
                      <li key={eid} className="text-xs text-amber-400 font-mono">{eid}</li>
                    ))}
                  </ul>
                </div>
              </div>
            </div>
          )}
        </>
      )}

      {!loading && !rings.length && !error && (
        <div className="flex items-center justify-center h-40 text-muted-foreground text-sm">
          No collusion rings detected in the current dataset.
        </div>
      )}
    </div>
  );
}
