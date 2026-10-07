"use client";

import { useEffect } from "react";
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
} from "lucide-react";
import { useAuth } from "@/hooks/useAuth";
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

  useEffect(() => {
    if (!isLoading && !user) router.replace("/login");
  }, [user, isLoading, router]);

  if (isLoading || !user) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-background">
        <Activity className="w-8 h-8 text-primary animate-pulse" />
      </div>
    );
  }

  const visibleNav = NAV.filter((n) => n.roles.includes(user.role));

  return (
    <div className="flex min-h-screen bg-background">
      {/* ── Sidebar ── */}
      <aside className="w-60 shrink-0 flex flex-col border-r border-border bg-card/50">
        {/* Logo */}
        <div className="px-5 py-5 border-b border-border">
          <div className="flex items-center gap-2.5">
            <div className="flex items-center justify-center w-8 h-8 rounded-lg bg-primary/10">
              <Shield className="w-4 h-4 text-primary" />
            </div>
            <div>
              <p className="text-sm font-bold leading-none">ClaimGuard AI</p>
              <p className="text-[10px] text-muted-foreground mt-0.5">InsurTech Platform</p>
            </div>
          </div>
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
