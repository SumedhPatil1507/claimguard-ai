"use client";

import { RadialBarChart, RadialBar, ResponsiveContainer, PolarAngleAxis } from "recharts";
import { getRiskColor } from "@/lib/utils";

interface ScoreGaugeProps {
  score: number;          // 0–1
  label: string;
  tier?: "low" | "medium" | "high";
  size?: number;
}

export function ScoreGauge({ score, label, tier, size = 180 }: ScoreGaugeProps) {
  const pct   = Math.round(score * 100);
  const color = tier ? getRiskColor(tier) : score > 0.6 ? "#ef4444" : score > 0.3 ? "#f59e0b" : "#10b981";
  const data  = [{ value: pct, fill: color }];

  return (
    <div className="flex flex-col items-center gap-2">
      <div style={{ width: size, height: size / 1.6 }}>
        <ResponsiveContainer width="100%" height="100%">
          <RadialBarChart
            cx="50%"
            cy="100%"
            innerRadius="70%"
            outerRadius="100%"
            barSize={16}
            data={data}
            startAngle={180}
            endAngle={0}
          >
            <PolarAngleAxis type="number" domain={[0, 100]} angleAxisId={0} tick={false} />
            <RadialBar
              background={{ fill: "hsl(217 32% 17%)" }}
              dataKey="value"
              angleAxisId={0}
              cornerRadius={8}
            />
          </RadialBarChart>
        </ResponsiveContainer>
      </div>
      <div className="text-center -mt-2">
        <p className="text-3xl font-bold" style={{ color }}>
          {(score * 100).toFixed(1)}%
        </p>
        <p className="text-xs text-muted-foreground mt-0.5">{label}</p>
      </div>
    </div>
  );
}
