import { cn } from "@/lib/utils";
import { getSeverityBadgeClass } from "@/lib/utils";

interface BadgeProps {
  children: React.ReactNode;
  variant?: "default" | "severity" | "status";
  severity?: "low" | "medium" | "high";
  className?: string;
}

export function Badge({ children, variant = "default", severity, className }: BadgeProps) {
  return (
    <span className={cn(
      "inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium border",
      variant === "severity" && severity ? getSeverityBadgeClass(severity) : "bg-secondary text-secondary-foreground border-border",
      className,
    )}>
      {children}
    </span>
  );
}
