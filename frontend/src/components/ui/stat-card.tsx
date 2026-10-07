import { cn } from "@/lib/utils";
import type { LucideIcon } from "lucide-react";

interface StatCardProps {
  label: string;
  value: string | number;
  icon?: LucideIcon;
  trend?: { value: string; positive: boolean };
  className?: string;
  highlight?: boolean;
}

export function StatCard({ label, value, icon: Icon, trend, className, highlight }: StatCardProps) {
  return (
    <div className={cn(
      "bg-card border border-border rounded-xl p-5 flex items-start gap-4 transition hover:border-primary/30",
      highlight && "border-primary/30 bg-primary/5",
      className,
    )}>
      {Icon && (
        <div className={cn("w-9 h-9 rounded-lg flex items-center justify-center shrink-0",
          highlight ? "bg-primary/20" : "bg-secondary")}>
          <Icon className={cn("w-4.5 h-4.5", highlight ? "text-primary" : "text-muted-foreground")} />
        </div>
      )}
      <div>
        <p className="text-xs text-muted-foreground font-medium">{label}</p>
        <p className="text-2xl font-bold mt-0.5">{value}</p>
        {trend && (
          <p className={cn("text-xs mt-1 font-medium", trend.positive ? "text-emerald-400" : "text-red-400")}>
            {trend.positive ? "↑" : "↓"} {trend.value}
          </p>
        )}
      </div>
    </div>
  );
}
