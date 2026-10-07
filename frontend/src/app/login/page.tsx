"use client";

import { useState } from "react";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { z } from "zod";
import { Shield, Loader2, AlertCircle } from "lucide-react";
import { useAuth } from "@/hooks/useAuth";
import { ApiError } from "@/lib/api";
import { cn } from "@/lib/utils";

const schema = z.object({
  username: z.string().min(1, "Username is required"),
  password: z.string().min(1, "Password is required"),
});
type FormValues = z.infer<typeof schema>;

const DEMO_CREDS = [
  { label: "Admin",   username: "admin",   password: "admin123" },
  { label: "Analyst", username: "analyst", password: "analyst123" },
  { label: "Viewer",  username: "viewer",  password: "viewer123" },
];

export default function LoginPage() {
  const { login } = useAuth();
  const [serverError, setServerError] = useState<string | null>(null);

  const {
    register,
    handleSubmit,
    setValue,
    formState: { errors, isSubmitting },
  } = useForm<FormValues>({ resolver: zodResolver(schema) });

  async function onSubmit(data: FormValues) {
    setServerError(null);
    try {
      await login(data.username, data.password);
    } catch (err) {
      setServerError(err instanceof ApiError ? err.message : "Login failed");
    }
  }

  return (
    <div className="min-h-screen flex items-center justify-center bg-background p-4">
      {/* Decorative glow */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <div className="absolute -top-40 left-1/2 -translate-x-1/2 w-96 h-96 bg-primary/10 rounded-full blur-3xl" />
      </div>

      <div className="relative w-full max-w-md animate-fade-in">
        {/* Logo */}
        <div className="text-center mb-8">
          <div className="inline-flex items-center justify-center w-16 h-16 rounded-2xl bg-primary/10 border border-primary/20 mb-4">
            <Shield className="w-8 h-8 text-primary" />
          </div>
          <h1 className="text-2xl font-bold text-foreground">ClaimGuard AI</h1>
          <p className="text-sm text-muted-foreground mt-1">Agentic InsurTech Platform</p>
        </div>

        {/* Card */}
        <div className="bg-card border border-border rounded-2xl p-8 shadow-2xl">
          <h2 className="text-lg font-semibold mb-6">Sign in to your account</h2>

          <form onSubmit={handleSubmit(onSubmit)} className="space-y-4">
            {/* Username */}
            <div>
              <label className="block text-sm font-medium text-muted-foreground mb-1.5">
                Username
              </label>
              <input
                {...register("username")}
                autoComplete="username"
                placeholder="admin"
                className={cn(
                  "w-full px-3 py-2.5 rounded-lg bg-background border text-sm focus:outline-none focus:ring-2 focus:ring-primary/50 transition",
                  errors.username ? "border-destructive" : "border-border",
                )}
              />
              {errors.username && (
                <p className="text-destructive text-xs mt-1">{errors.username.message}</p>
              )}
            </div>

            {/* Password */}
            <div>
              <label className="block text-sm font-medium text-muted-foreground mb-1.5">
                Password
              </label>
              <input
                {...register("password")}
                type="password"
                autoComplete="current-password"
                placeholder="••••••••"
                className={cn(
                  "w-full px-3 py-2.5 rounded-lg bg-background border text-sm focus:outline-none focus:ring-2 focus:ring-primary/50 transition",
                  errors.password ? "border-destructive" : "border-border",
                )}
              />
              {errors.password && (
                <p className="text-destructive text-xs mt-1">{errors.password.message}</p>
              )}
            </div>

            {/* Server error */}
            {serverError && (
              <div className="flex items-center gap-2 text-destructive bg-destructive/10 border border-destructive/20 rounded-lg px-3 py-2.5 text-sm">
                <AlertCircle className="w-4 h-4 shrink-0" />
                {serverError}
              </div>
            )}

            <button
              type="submit"
              disabled={isSubmitting}
              className="w-full py-2.5 rounded-lg bg-primary text-primary-foreground font-semibold text-sm hover:bg-primary/90 active:scale-[0.98] transition disabled:opacity-60 flex items-center justify-center gap-2"
            >
              {isSubmitting && <Loader2 className="w-4 h-4 animate-spin" />}
              {isSubmitting ? "Signing in…" : "Sign in"}
            </button>
          </form>

          {/* Demo credentials */}
          <div className="mt-6 pt-6 border-t border-border">
            <p className="text-xs text-muted-foreground text-center mb-3">
              Demo credentials — click to auto-fill
            </p>
            <div className="grid grid-cols-3 gap-2">
              {DEMO_CREDS.map((c) => (
                <button
                  key={c.label}
                  type="button"
                  onClick={() => {
                    setValue("username", c.username);
                    setValue("password", c.password);
                  }}
                  className="text-xs px-2 py-2 rounded-lg border border-border hover:border-primary/50 hover:bg-primary/5 transition text-muted-foreground hover:text-foreground"
                >
                  {c.label}
                </button>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
