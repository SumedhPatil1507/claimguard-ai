import { type ClassValue, clsx } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function formatCurrency(value: number, currency = "INR"): string {
  return new Intl.NumberFormat("en-IN", {
    style: "currency",
    currency,
    maximumFractionDigits: 0,
  }).format(value);
}

export function formatPercent(value: number, decimals = 1): string {
  return `${(value * 100).toFixed(decimals)}%`;
}

export function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

export function getRiskColor(tier: "low" | "medium" | "high"): string {
  return { low: "#10b981", medium: "#f59e0b", high: "#ef4444" }[tier];
}

export function getSeverityBadgeClass(severity: "low" | "medium" | "high"): string {
  return {
    low:    "bg-emerald-500/20 text-emerald-400 border-emerald-500/30",
    medium: "bg-amber-500/20  text-amber-400  border-amber-500/30",
    high:   "bg-red-500/20    text-red-400    border-red-500/30",
  }[severity];
}
