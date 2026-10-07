import { NextResponse } from "next/server";
import path from "path";
import fs from "fs";

/** Serve sample_claims.csv as JSON — only used in dev / Streamlit-less mode. */
export async function GET() {
  try {
    // Walk up from frontend/ to repo root, then into data/
    const csvPath = path.resolve(process.cwd(), "..", "data", "sample_claims.csv");
    if (!fs.existsSync(csvPath)) {
      return NextResponse.json([], { status: 200 });
    }
    const text = fs.readFileSync(csvPath, "utf-8");
    const [header, ...rows] = text.trim().split("\n");
    const keys = header.split(",").map((k) => k.trim());
    const records = rows.map((row) => {
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
  } catch {
    return NextResponse.json([], { status: 200 });
  }
}
