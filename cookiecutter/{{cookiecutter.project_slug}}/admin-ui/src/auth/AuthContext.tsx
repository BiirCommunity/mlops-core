import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from "react";
import { trainingApi } from "@/api/client";
import {
  clearAccessToken,
  getAccessToken,
  setAccessToken,
} from "@/auth/token";

type AuthContextValue = {
  token: string | null;
  tokenRequired: boolean;
  loading: boolean;
  login: (username: string, password: string) => Promise<void>;
  logout: () => void;
};

const AuthContext = createContext<AuthContextValue | null>(null);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [token, setToken] = useState<string | null>(() => getAccessToken());
  const [tokenRequired, setTokenRequired] = useState(true);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function init() {
      try {
        const status = await trainingApi.authStatus();
        setTokenRequired(status.token_required);
        const stored = getAccessToken();
        if (status.token_required && stored) {
          try {
            await trainingApi.verifyToken(stored);
          } catch {
            clearAccessToken();
            setToken(null);
          }
        }
      } finally {
        setLoading(false);
      }
    }
    void init();
  }, []);

  const login = useCallback(async (username: string, password: string) => {
    const result = await trainingApi.login(username, password);
    await trainingApi.verifyToken(result.token);
    setAccessToken(result.token);
    setToken(result.token);
  }, []);

  const logout = useCallback(() => {
    clearAccessToken();
    setToken(null);
  }, []);

  const value = useMemo(
    () => ({ token, tokenRequired, loading, login, logout }),
    [token, tokenRequired, loading, login, logout],
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used within AuthProvider");
  return ctx;
}
