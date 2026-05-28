import { useState } from "react";
import { Navigate, useNavigate } from "react-router-dom";
import { login } from "@/api/client";
import { useAuth } from "@/auth/AuthContext";
import { getToken } from "@/auth/token";

export function LoginPage() {
  const navigate = useNavigate();
  const { token, login: saveToken } = useAuth();
  const [username, setUsername] = useState("admin");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  if (token ?? getToken()) {
    return <Navigate to="/" replace />;
  }

  async function onSubmit(event: React.FormEvent) {
    event.preventDefault();
    setLoading(true);
    setError(null);
    try {
      const result = await login(username, password);
      saveToken(result.token);
      navigate("/");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Login failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="flex min-h-screen items-center justify-center p-4">
      <form onSubmit={onSubmit} className="card w-full max-w-md p-6 space-y-4">
        <div>
          <h1 className="text-2xl font-semibold">MLOps Chat</h1>
          <p className="mt-2 text-sm text-[var(--muted)]">
            Вход через auth-service. TTT-сессия сохраняется в диалоге.
          </p>
        </div>
        <input
          className="input"
          placeholder="Username"
          value={username}
          onChange={(e) => setUsername(e.target.value)}
          autoComplete="username"
        />
        <input
          className="input"
          type="password"
          placeholder="Password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          autoComplete="current-password"
        />
        {error ? <div className="text-sm text-red-300">{error}</div> : null}
        <button className="btn btn-primary w-full" disabled={loading} type="submit">
          {loading ? "Вход…" : "Войти"}
        </button>
      </form>
    </div>
  );
}
