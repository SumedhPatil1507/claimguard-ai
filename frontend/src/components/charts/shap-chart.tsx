"use client";

import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Cell,
} from "recharts";
import type { ShapDriver } from "@/types";

interface ShapChartProps {
  drivers: ShapDriver[];
}

export function ShapChart({ drivers }: ShapChartProps) {
  if (!drivers.length) return <p className="text-muted-foreground text-sm text-center py-4">No SHAP drivers available.</p>;

  const data = [...drivers]
    .sort((a, b) => Math.abs(b.shap_value) - Math.abs(a.shap_value))
    .slice(0, 8)
    .reverse();

  return (
    <ResponsiveContainer width="100%" height={Math.max(200, data.length * 36)}>
      <BarChart data={data} layout="vertical" margin={{ left: 8, right: 40, top: 4, bottom: 4 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="hsl(217 32% 17%)" horizontal={false} />
        <XAxis type="number" tick={{ fill: "hsl(215 20% 65%)", fontSize: 11 }} tickLine={false} axisLine={false} />
        <YAxis
          type="category"
          dataKey="feature"
          tick={{ fill: "hsl(215 20% 65%)", fontSize: 11 }}
          tickLine={false}
          axisLine={false}
          width={130}
        />
        <Tooltip
          contentStyle={{ background: "hsl(222 47% 10%)", border: "1px solid hsl(217 32% 17%)", borderRadius: 8, fontSize: 12 }}
          formatter={(v: number) => [v.toFixed(4), "SHAP value"]}
        />
        <Bar dataKey="shap_value" radius={[0, 4, 4, 0]}>
          {data.map((d, i) => (
            <Cell key={i} fill={d.direction === "increases_risk" ? "#ef4444" : "#10b981"} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}
