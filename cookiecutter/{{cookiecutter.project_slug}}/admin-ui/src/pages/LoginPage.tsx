import { FormEvent, useState } from "react";
import { Navigate, useLocation, useNavigate } from "react-router-dom";
import { KeyRound, ShieldCheck } from "lucide-react";
import { useAuth } from "@/auth/AuthContext";

export function LoginPage() {
  const { token, tokenRequired, login } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();
  const [username, setUsername] = useState("admin");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [pending, setPending] = useState(false);

  if (!tokenRequired) {
    return <Navigate to="/" replace />;
  }

  if (token) {
    return <Navigate to={(location.state as { from?: string })?.from ?? "/"} replace />;
  }

  async function onSubmit(event: FormEvent) {
    event.preventDefault();
    setPending(true);
    setError(null);
    try {
      await login(username.trim(), password);
      navigate((location.state as { from?: string })?.from ?? "/");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Ошибка авторизации");
    } finally {
      setPending(false);
    }
  }

  return (
    <div className="flex min-h-screen items-center justify-center px-4">
      <form onSubmit={onSubmit} className="card w-full max-w-md p-8">
        <div className="mb-6 flex items-center gap-3">
          <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-[var(--accent-soft)] text-[var(--accent)]">
            <ShieldCheck size={22} />
          </div>
          <div>
            <div className="text-xl font-semibold">Admin Studio</div>
            <div className="text-sm text-[var(--muted)]">Вход в control plane</div>
          </div>
        </div>
        <p className="mb-6 text-sm text-[var(--muted)]">
          Вход через auth-service (username / password).
        </p>
        <label className="mb-4 block">
          <span className="mb-2 block text-sm text-[var(--muted)]">Username</span>
          <input
            className="input"
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            autoComplete="username"
            required
          />
        </label>
        <label className="mb-4 block">
          <span className="mb-2 block text-sm text-[var(--muted)]">Password</span>
          <input
            className="input"
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            autoComplete="current-password"
            required
          />
        </label>
        {error ? (
          <div className="mb-4 rounded-xl border border-red-500/30 bg-red-500/10 px-4 py-3 text-sm text-red-200">
            {error}
          </div>
        ) : null}
        <button type="submit" className="btn btn-primary w-full" disabled={pending}>
          <KeyRound size={18} />
          Войти
        </button>
      </form>
    </div>
  );
}
