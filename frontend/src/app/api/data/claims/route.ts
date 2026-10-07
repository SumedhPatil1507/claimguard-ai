import { NextResponse } from "next/server";
import path from "path";
import fs from "fs";

/** Serve sample_claims.csv as JSON with multi-path resolution and synthetic fallback. */
export async function GET() {
  try {
    const candidatePaths = [
      path.resolve(process.cwd(), "..", "data", "sample_claims.csv"),
      path.resolve(process.cwd(), "data", "sample_claims.csv"),
      path.resolve(process.cwd(), "public", "sample_claims.csv"),
      "/app/data/sample_claims.csv",
    ];

    let foundPath: string | null = null;
    for (const p of candidatePaths) {
      if (fs.existsSync(p)) {
        foundPath = p;
        break;
      }
    }

    if (foundPath) {
      const text = fs.readFileSync(foundPath, "utf-8");
      const [header, ...rows] = text.trim().split("\n");
      const keys = header.split(",").map((k) => k.trim());
      const records = rows.filter(Boolean).map((row) => {
        const vals = row.split(",");
        return Object.fromEntries(
          keys.map((k, i) => {
            const v = vals[i]?.trim() ?? "";
            const n = Number(v);
            return [k, isNaN(n) || v === "" ? v : n];
          }),
        );
      });
      return NextResponse.json(records);
    }

    // Fallback: generate realistic sample records if CSV file isn't on disk
    const fallbackTypes = ["motor", "health", "property", "life"] as const;
    const fallbackSeverities = ["low", "medium", "high"] as const;
    const records = Array.from({ length: 150 }, (_, i) => ({
      claim_id: `CLM-${(i + 1).toString().padStart(4, "0")}`,
      claimant_id: `CLT-${((i % 40) + 1).toString().padStart(3, "0")}`,
      policy_id: `POL-${((i % 50) + 1).toString().padStart(3, "0")}`,
      claim_amount: Math.round(15000 + Math.random() * 450000),
      days_since_policy_start: Math.floor(Math.random() * 700) + 5,
      num_prior_claims: Math.floor(Math.random() * 4),
      claim_type: fallbackTypes[i % fallbackTypes.length],
      claim_severity: fallbackSeverities[i % fallbackSeverities.length],
      fraud_label: Math.random() > 0.85 ? 1 : 0,
      repair_shop_id: `SHOP-${((i % 8) + 1).toString().padStart(3, "0")}`,
      medical_provider_id: `MED-${((i % 6) + 1).toString().padStart(3, "0")}`,
    }));

    return NextResponse.json(records);
  } catch {
    return NextResponse.json([], { status: 200 });
  }
}
