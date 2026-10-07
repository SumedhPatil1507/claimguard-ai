"use client";

import React, {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useState,
} from "react";
import { useRouter } from "next/navigation";
import { login as apiLogin, clearToken, saveToken, getToken } from "@/lib/api";
import type { AuthUser, Role } from "@/types";

interface AuthContextValue {
  user: AuthUser | null;
  isLoading: boolean;
  login: (username: string, password: string) => Promise<void>;
  logout: () => void;
}

const AuthContext = createContext<AuthContextValue | undefined>(undefined);

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const router = useRouter();
  const [user, setUser] = useState<AuthUser | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  // Rehydrate from sessionStorage on mount
  useEffect(() => {
    const token = getToken();
    if (token) {
      try {
        // Decode JWT payload (no signature verification — server already validated)
        const [, payload] = token.split(".");
        const decoded = JSON.parse(atob(payload.replace(/-/g, "+").replace(/_/g, "/")));
        if (decoded.exp * 1000 > Date.now()) {
          setUser({ username: decoded.sub, role: decoded.role as Role, token });
        } else {
          clearToken();
        }
      } catch {
        clearToken();
      }
    }
    setIsLoading(false);
  }, []);

  const login = useCallback(
    async (username: string, password: string) => {
      const data = await apiLogin(username, password);
      saveToken(data.access_token);
      setUser({ username, role: data.role, token: data.access_token });
      router.replace("/explorer");
    },
    [router],
  );

  const logout = useCallback(() => {
    clearToken();
    setUser(null);
    router.replace("/login");
  }, [router]);

  return (
    <AuthContext.Provider value={{ user, isLoading, login, logout }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth(): AuthContextValue {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used within AuthProvider");
  return ctx;
}
