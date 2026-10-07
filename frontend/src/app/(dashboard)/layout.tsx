"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import {
  BarChart3,
  Shield,
  AlertTriangle,
  Network,
  Users,
  LogOut,
  Activity,
  ChevronRight,
  Wifi,
  WifiOff,
} from "lucide-react";
import { useAuth } from "@/hooks/useAuth";
import { getHealth, type HealthResponse } from "@/lib/api";
import { cn } from "@/lib/utils";

const NAV = [
  { href: "/explorer",    label: "Explorer",        icon: BarChart3,    roles: ["admin", "analyst", "viewer"] },
  { href: "/underwriting", label: "Underwriting",   icon: Shield,       roles: ["admin", "analyst"] },
  { href: "/fraud",       label: "Fraud Scoring",   icon: AlertTriangle, roles: ["admin", "analyst"] },
  { href: "/graph",       label: "Graph Intel",     icon: Network,      roles: ["admin", "analyst"] },
  { href: "/hitl",        label: "HITL Review",     icon: Users,        roles: ["admin", "analyst"] },
];

const ROLE_BADGE: Record<string, string> = {
  admin:   "bg-purple-500/20 text-purple-400 border-purple-500/30",
  analyst: "bg-blue-500/20   text-blue-400   border-blue-500/30",
  viewer:  "bg-gray-500/20   text-gray-400   border-gray-500/30",
};

export default function DashboardLayout({ children }: { children: React.ReactNode }) {
  const { user, isLoading, logout } = useAuth();
  const router = useRouter();
  const pathname = usePathname();
  const [apiHealth, setApiHealth] = useState<HealthResponse | null>(null);

  useEffect(() => {
    if (!isLoading && !user) router.replace("/login");
  }, [user, isLoading, router]);

  // Periodic health check polling
  useEffect(() => {
    let mounted = true;
    async function check() {
      const h = await getHealth();
      if (mounted) setApiHealth(h);
    }
    check();
    const interval = setInterval(check, 15000);
    return () => {
      mounted = false;
      clearInterval(interval);
    };
  }, []);

  if (isLoading || !user) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-background">
        <Activity className="w-8 h-8 text-primary animate-pulse" />
      </div>
    );
  }

  const visibleNav = NAV.filter((n) => n.roles.includes(user.role));
  const isOnline = apiHealth?.status === "ok" || apiHealth?.status === "healthy";

  return (
    <div className="flex min-h-screen bg-background">
      {/* ── Sidebar ── */}
      <aside className="w-60 shrink-0 flex flex-col border-r border-border bg-card/50">
        {/* Logo & Platform Info */}
        <div className="px-5 py-5 border-b border-border">
          <div className="flex items-center gap-2.5">
            <div className="flex items-center justify-center w-8 h-8 rounded-lg bg-primary/10 border border-primary/20">
              <Shield className="w-4 h-4 text-primary" />
            </div>
            <div>
              <p className="text-sm font-bold leading-none">ClaimGuard AI</p>
              <p className="text-[10px] text-muted-foreground mt-0.5">InsurTech Next.js 14</p>
            </div>
          </div>
        </div>

        {/* Backend API Connection Status Chip */}
        <div className="px-4 py-2.5 mx-3 mt-3 rounded-lg bg-secondary/30 border border-border/50 flex items-center justify-between text-xs">
          <div className="flex items-center gap-2">
            <span className="relative flex h-2 w-2">
              <span className={cn("animate-ping absolute inline-flex h-full w-full rounded-full opacity-75", isOnline ? "bg-emerald-400" : "bg-amber-400")} />
              <span className={cn("relative inline-flex rounded-full h-2 w-2", isOnline ? "bg-emerald-500" : "bg-amber-500")} />
            </span>
            <span className="text-[11px] text-muted-foreground font-medium">
              API: {isOnline ? "Online" : "Fallback/Local"}
            </span>
          </div>
          {isOnline ? (
            <Wifi className="w-3.5 h-3.5 text-emerald-400" />
          ) : (
            <WifiOff className="w-3.5 h-3.5 text-amber-400" />
          )}
        </div>

        {/* Nav links */}
        <nav className="flex-1 px-3 py-4 space-y-0.5">
          {visibleNav.map((item) => {
            const active = pathname.startsWith(item.href);
            return (
              <Link
                key={item.href}
                href={item.href}
                className={cn(
                  "flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-all group",
                  active
                    ? "bg-primary/10 text-primary"
                    : "text-muted-foreground hover:text-foreground hover:bg-secondary/50",
                )}
              >
                <item.icon className={cn("w-4 h-4 shrink-0 transition", active ? "text-primary" : "group-hover:text-foreground")} />
                {item.label}
                {active && <ChevronRight className="w-3 h-3 ml-auto text-primary/60" />}
              </Link>
            );
          })}
        </nav>

        {/* User section */}
        <div className="px-3 py-4 border-t border-border">
          <div className="flex items-center gap-3 px-3 py-2.5 rounded-lg bg-secondary/30">
            <div className="w-7 h-7 rounded-full bg-primary/20 flex items-center justify-center text-xs font-bold text-primary">
              {user.username[0].toUpperCase()}
            </div>
            <div className="flex-1 min-w-0">
              <p className="text-xs font-medium truncate">{user.username}</p>
              <span className={cn("text-[10px] px-1.5 py-0.5 rounded-full border font-medium", ROLE_BADGE[user.role])}>
                {user.role}
              </span>
            </div>
            <button
              onClick={logout}
              className="text-muted-foreground hover:text-destructive transition p-1 rounded"
              title="Sign out"
            >
              <LogOut className="w-3.5 h-3.5" />
            </button>
          </div>
        </div>
      </aside>

      {/* ── Main content ── */}
      <main className="flex-1 min-w-0 overflow-auto">
        <div className="p-6 animate-fade-in">{children}</div>
      </main>
    </div>
  );
}
