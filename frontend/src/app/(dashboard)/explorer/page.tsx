"use client";

import { useEffect, useMemo, useState } from "react";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, Legend, ScatterChart, Scatter, ZAxis,
} from "recharts";
import { BarChart3, TrendingUp, FileText, AlertCircle } from "lucide-react";
import { getClaimsData } from "@/lib/api";
import { formatCurrency } from "@/lib/utils";
import { StatCard } from "@/components/ui/stat-card";
import { Badge } from "@/components/ui/badge";
import type { ClaimRecord } from "@/types";

const COLORS = { legitimate: "#10b981", fraud: "#ef4444" };
const CHART_STYLE = {
  background: "hsl(222 47% 10%)",
  border: "1px solid hsl(217 32% 17%)",
  borderRadius: 8,
  fontSize: 12,
};

export default function ExplorerPage() {
  const [claims, setClaims]   = useState<ClaimRecord[]>([]);
  const [loading, setLoading] = useState(true);
  const [search, setSearch]   = useState("");
  const [page, setPage]       = useState(0);
  const PAGE_SIZE = 10;

  useEffect(() => {
    getClaimsData().then(setClaims).finally(() => setLoading(false));
  }, []);

  // ── Computed stats ─────────────────────────────────────────────────────────
  const stats = useMemo(() => {
    const fraudCount = claims.filter((c) => c.fraud_label === 1).length;
    const total      = claims.length;
    const avgAmount  = claims.reduce((s, c) => s + c.claim_amount, 0) / (total || 1);
    return { total, fraudCount, fraudRate: fraudCount / (total || 1), avgAmount };
  }, [claims]);

  // ── Chart data ─────────────────────────────────────────────────────────────
  const typeData = useMemo(() => {
    const map: Record<string, { legitimate: number; fraud: number }> = {};
    claims.forEach((c) => {
      if (!map[c.claim_type]) map[c.claim_type] = { legitimate: 0, fraud: 0 };
      if (c.fraud_label) map[c.claim_type].fraud++;
      else map[c.claim_type].legitimate++;
    });
    return Object.entries(map).map(([name, v]) => ({ name, ...v }));
  }, [claims]);

  const pieData = useMemo(() => [
    { name: "Legitimate", value: stats.total - stats.fraudCount, color: COLORS.legitimate },
    { name: "Fraud",      value: stats.fraudCount,               color: COLORS.fraud },
  ], [stats]);

  const severityData = useMemo(() => {
    const map: Record<string, number> = {};
    claims.forEach((c) => { map[c.claim_severity] = (map[c.claim_severity] ?? 0) + 1; });
    return Object.entries(map).map(([name, count]) => ({ name, count }));
  }, [claims]);

  const scatterData = useMemo(() =>
    claims.slice(0, 300).map((c) => ({
      x: c.num_prior_claims,
      y: c.claim_amount,
      z: c.fraud_label ? 8 : 5,
      fraud: c.fraud_label,
    })), [claims]);

  // ── Table ─────────────────────────────────────────────────────────────────
  const filtered = useMemo(() => {
    const q = search.toLowerCase();
    return claims.filter((c) =>
      !q || c.claim_id.toLowerCase().includes(q) || c.claimant_id.toLowerCase().includes(q) || c.claim_type.toLowerCase().includes(q),
    );
  }, [claims, search]);
  const paged     = filtered.slice(page * PAGE_SIZE, (page + 1) * PAGE_SIZE);
  const totalPages = Math.ceil(filtered.length / PAGE_SIZE);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="flex items-center gap-3 text-muted-foreground">
          <BarChart3 className="w-5 h-5 animate-pulse text-primary" />
          Loading claims data…
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold flex items-center gap-2">
          <BarChart3 className="w-6 h-6 text-primary" /> Data Explorer
        </h1>
        <p className="text-muted-foreground text-sm mt-1">
          Explore {stats.total.toLocaleString()} claims from the synthetic dataset
        </p>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard label="Total Claims"   value={stats.total.toLocaleString()}       icon={FileText}    />
        <StatCard label="Fraud Rate"     value={`${(stats.fraudRate * 100).toFixed(1)}%`} icon={AlertCircle} highlight />
        <StatCard label="Avg Claim (₹)"  value={formatCurrency(stats.avgAmount)}    icon={TrendingUp}  />
        <StatCard label="Fraud Cases"    value={stats.fraudCount.toLocaleString()}  icon={AlertCircle} />
      </div>

      {/* Charts row 1 */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* Claim type breakdown */}
        <div className="bg-card border border-border rounded-xl p-5">
          <h3 className="text-sm font-semibold mb-4">Claims by Type (Fraud vs Legitimate)</h3>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={typeData} margin={{ left: -10 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="hsl(217 32% 17%)" />
              <XAxis dataKey="name" tick={{ fill: "hsl(215 20% 65%)", fontSize: 11 }} axisLine={false} tickLine={false} />
              <YAxis tick={{ fill: "hsl(215 20% 65%)", fontSize: 11 }} axisLine={false} tickLine={false} />
              <Tooltip contentStyle={CHART_STYLE} />
              <Legend wrapperStyle={{ fontSize: 12 }} />
              <Bar dataKey="legitimate" name="Legitimate" fill={COLORS.legitimate} radius={[3, 3, 0, 0]} />
              <Bar dataKey="fraud"      name="Fraud"      fill={COLORS.fraud}      radius={[3, 3, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Fraud pie */}
        <div className="bg-card border border-border rounded-xl p-5">
          <h3 className="text-sm font-semibold mb-4">Fraud Distribution</h3>
          <ResponsiveContainer width="100%" height={220}>
            <PieChart>
              <Pie data={pieData} cx="50%" cy="50%" innerRadius={60} outerRadius={90} paddingAngle={3} dataKey="value">
                {pieData.map((d, i) => <Cell key={i} fill={d.color} />)}
              </Pie>
              <Tooltip contentStyle={CHART_STYLE} formatter={(v: number) => [v.toLocaleString(), ""]} />
              <Legend wrapperStyle={{ fontSize: 12 }} />
            </PieChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Charts row 2 */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* Severity */}
        <div className="bg-card border border-border rounded-xl p-5">
          <h3 className="text-sm font-semibold mb-4">Claim Severity Distribution</h3>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={severityData} margin={{ left: -10 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="hsl(217 32% 17%)" />
              <XAxis dataKey="name" tick={{ fill: "hsl(215 20% 65%)", fontSize: 11 }} axisLine={false} tickLine={false} />
              <YAxis tick={{ fill: "hsl(215 20% 65%)", fontSize: 11 }} axisLine={false} tickLine={false} />
              <Tooltip contentStyle={CHART_STYLE} />
              <Bar dataKey="count" fill="#14b8a6" radius={[3, 3, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Scatter: prior claims vs amount */}
        <div className="bg-card border border-border rounded-xl p-5">
          <h3 className="text-sm font-semibold mb-4">Prior Claims vs Claim Amount</h3>
          <ResponsiveContainer width="100%" height={200}>
            <ScatterChart margin={{ left: -10, right: 8 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="hsl(217 32% 17%)" />
              <XAxis dataKey="x" name="Prior Claims" tick={{ fill: "hsl(215 20% 65%)", fontSize: 11 }} axisLine={false} tickLine={false} label={{ value: "Prior claims", position: "insideBottom", offset: -2, fill: "hsl(215 20% 65%)", fontSize: 10 }} />
              <YAxis dataKey="y" name="Amount (₹)" tick={{ fill: "hsl(215 20% 65%)", fontSize: 10 }} axisLine={false} tickLine={false} tickFormatter={(v) => `${(v / 1000).toFixed(0)}k`} />
              <ZAxis dataKey="z" range={[20, 80]} />
              <Tooltip contentStyle={CHART_STYLE} formatter={(v: number, name: string) => [name === "Amount (₹)" ? formatCurrency(v) : v, name]} />
              <Scatter
                data={scatterData.filter((d) => d.fraud === 0)}
                fill={COLORS.legitimate}
                fillOpacity={0.6}
                name="Legitimate"
              />
              <Scatter
                data={scatterData.filter((d) => d.fraud === 1)}
                fill={COLORS.fraud}
                fillOpacity={0.7}
                name="Fraud"
              />
            </ScatterChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Table */}
      <div className="bg-card border border-border rounded-xl overflow-hidden">
        <div className="px-5 py-4 border-b border-border flex items-center justify-between gap-4">
          <h3 className="text-sm font-semibold">Claims Records</h3>
          <input
            type="text"
            placeholder="Search by ID, claimant, type…"
            value={search}
            onChange={(e) => { setSearch(e.target.value); setPage(0); }}
            className="bg-background border border-border rounded-lg px-3 py-1.5 text-sm w-60 focus:outline-none focus:ring-2 focus:ring-primary/50"
          />
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-border text-muted-foreground">
                {["Claim ID", "Claimant", "Amount", "Type", "Severity", "Prior Claims", "Status"].map((h) => (
                  <th key={h} className="text-left px-4 py-3 font-medium text-xs">{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {paged.map((c, i) => (
                <tr key={c.claim_id} className={`border-b border-border/50 hover:bg-secondary/20 transition ${i % 2 === 0 ? "" : "bg-secondary/5"}`}>
                  <td className="px-4 py-2.5 font-mono text-xs text-muted-foreground">{c.claim_id}</td>
                  <td className="px-4 py-2.5 text-xs">{c.claimant_id}</td>
                  <td className="px-4 py-2.5 font-medium">{formatCurrency(c.claim_amount)}</td>
                  <td className="px-4 py-2.5 capitalize">{c.claim_type}</td>
                  <td className="px-4 py-2.5">
                    <Badge variant="severity" severity={c.claim_severity as "low" | "medium" | "high"}>{c.claim_severity}</Badge>
                  </td>
                  <td className="px-4 py-2.5 text-center">{c.num_prior_claims}</td>
                  <td className="px-4 py-2.5">
                    <span className={`inline-flex items-center gap-1 text-xs font-medium ${c.fraud_label ? "text-red-400" : "text-emerald-400"}`}>
                      <span className={`w-1.5 h-1.5 rounded-full ${c.fraud_label ? "bg-red-400" : "bg-emerald-400"}`} />
                      {c.fraud_label ? "Fraud" : "Legitimate"}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        {/* Pagination */}
        <div className="px-5 py-3 border-t border-border flex items-center justify-between text-xs text-muted-foreground">
          <span>Showing {page * PAGE_SIZE + 1}–{Math.min((page + 1) * PAGE_SIZE, filtered.length)} of {filtered.length}</span>
          <div className="flex items-center gap-2">
            <button disabled={page === 0} onClick={() => setPage((p) => p - 1)} className="px-3 py-1 rounded-lg border border-border disabled:opacity-40 hover:bg-secondary/50 transition">Prev</button>
            <span>{page + 1} / {totalPages || 1}</span>
            <button disabled={page + 1 >= totalPages} onClick={() => setPage((p) => p + 1)} className="px-3 py-1 rounded-lg border border-border disabled:opacity-40 hover:bg-secondary/50 transition">Next</button>
          </div>
        </div>
      </div>
    </div>
  );
}
