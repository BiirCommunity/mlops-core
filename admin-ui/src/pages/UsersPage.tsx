import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { LoaderCircle, Trash2, UserPlus } from "lucide-react";
import { useState } from "react";
import { authServiceApi, ApiError } from "@/api/client";
import { formatTimestamp } from "@/lib/format";

export function UsersPage() {
  const queryClient = useQueryClient();
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [active, setActive] = useState(true);
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [newToken, setNewToken] = useState<string | null>(null);

  const usersQuery = useQuery({
    queryKey: ["auth-users"],
    queryFn: authServiceApi.listUsers,
    refetchInterval: 15000,
    retry: false,
  });

  const usersError =
    usersQuery.error instanceof Error ? usersQuery.error.message : null;
  const needsRelogin =
    usersQuery.error instanceof ApiError && usersQuery.error.status === 401;

  const createMutation = useMutation({
    mutationFn: authServiceApi.createUser,
    onSuccess: async (user) => {
      setError(null);
      setMessage(`Пользователь ${user.username} создан.`);
      setUsername("");
      setPassword("");
      setActive(true);
      queryClient.invalidateQueries({ queryKey: ["auth-users"] });
      try {
        const key = await authServiceApi.createApiKey({
          user_id: user.id,
          name: "default",
          scopes: ["inference"],
        });
        setNewToken(key.token);
        setMessage(
          `Пользователь ${user.username} создан. API token для Chat UI скопируйте ниже.`,
        );
        queryClient.invalidateQueries({ queryKey: ["auth-api-keys"] });
      } catch {
        setMessage(`Пользователь ${user.username} создан (без auto API key).`);
      }
    },
    onError: (err: Error) => {
      setMessage(null);
      setNewToken(null);
      setError(err.message);
    },
  });

  const toggleMutation = useMutation({
    mutationFn: ({ id, active: nextActive }: { id: number; active: boolean }) =>
      authServiceApi.updateUser(id, { active: nextActive }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["auth-users"] });
    },
    onError: (err: Error) => setError(err.message),
  });

  const deleteMutation = useMutation({
    mutationFn: authServiceApi.deleteUser,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["auth-users"] });
      setMessage("Пользователь удалён.");
    },
    onError: (err: Error) => setError(err.message),
  });

  const keysQuery = useQuery({
    queryKey: ["auth-api-keys"],
    queryFn: authServiceApi.listApiKeys,
    refetchInterval: 15000,
  });

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-3xl font-semibold">Пользователи</h1>
        <p className="mt-2 text-[var(--muted)]">
          Управление учётными записями auth-service для Chat UI и Admin Studio.
          Scope <code>inference</code> — чат, <code>admin</code> — control plane.
        </p>
      </div>

      {message ? (
        <div className="rounded-xl border border-green-500/30 bg-green-500/10 px-4 py-3 text-sm text-green-100">
          {message}
        </div>
      ) : null}
      {error || usersError ? (
        <div className="rounded-xl border border-red-500/30 bg-red-500/10 px-4 py-3 text-sm text-red-200">
          {error ?? usersError}
          {needsRelogin ? (
            <div className="mt-2">
              Нажмите «Выйти» слева и войдите снова через auth-service (admin /
              AUTH_BOOTSTRAP_PASSWORD).
            </div>
          ) : null}
        </div>
      ) : null}
      {newToken ? (
        <div className="card space-y-2 p-4">
          <div className="text-sm text-[var(--muted)]">Новый API token (показывается один раз)</div>
          <code className="block break-all rounded-lg bg-[#0d1219] p-3 text-sm">{newToken}</code>
        </div>
      ) : null}

      <section className="card space-y-4 p-6">
        <h2 className="text-xl font-semibold">Создать пользователя</h2>
        <div className="grid gap-4 md:grid-cols-2">
          <label className="block">
            <span className="mb-2 block text-sm text-[var(--muted)]">Username</span>
            <input
              className="input"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              placeholder="operator"
              minLength={2}
            />
          </label>
          <label className="block">
            <span className="mb-2 block text-sm text-[var(--muted)]">Password</span>
            <input
              className="input"
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="минимум 6 символов"
              minLength={6}
            />
          </label>
        </div>
        <label className="flex items-center gap-2 text-sm">
          <input
            type="checkbox"
            checked={active}
            onChange={(e) => setActive(e.target.checked)}
          />
          Активен
        </label>
        <button
          type="button"
          className="btn btn-primary"
          disabled={
            createMutation.isPending ||
            username.trim().length < 2 ||
            password.length < 6
          }
          onClick={() =>
            createMutation.mutate({
              username: username.trim(),
              password,
              active,
            })
          }
        >
          {createMutation.isPending ? (
            <LoaderCircle className="animate-spin" size={18} />
          ) : (
            <UserPlus size={18} />
          )}
          Создать
        </button>
      </section>

      <section className="space-y-4">
        <h2 className="text-xl font-semibold">Список пользователей</h2>
        <div className="table-wrap">
          <table className="data-table">
            <thead>
              <tr>
                <th>ID</th>
                <th>Username</th>
                <th>Status</th>
                <th>Created</th>
                <th></th>
              </tr>
            </thead>
            <tbody>
              {(usersQuery.data?.users ?? []).map((user) => (
                <tr key={user.id}>
                  <td>{user.id}</td>
                  <td>{user.username}</td>
                  <td>
                    <span
                      className={`badge ${user.active ? "badge-completed" : "badge-failed"}`}
                    >
                      {user.active ? "active" : "disabled"}
                    </span>
                  </td>
                  <td>{formatTimestamp(user.created_at)}</td>
                  <td className="space-x-2">
                    <button
                      type="button"
                      className="btn btn-secondary !px-3 !py-2 text-sm"
                      disabled={toggleMutation.isPending}
                      onClick={() =>
                        toggleMutation.mutate({ id: user.id, active: !user.active })
                      }
                    >
                      {user.active ? "Disable" : "Enable"}
                    </button>
                    <button
                      type="button"
                      className="btn btn-secondary !px-3 !py-2 text-sm"
                      disabled={deleteMutation.isPending}
                      onClick={() => {
                        if (window.confirm(`Удалить ${user.username}?`)) {
                          deleteMutation.mutate(user.id);
                        }
                      }}
                    >
                      <Trash2 size={16} />
                    </button>
                  </td>
                </tr>
              ))}
              {!usersQuery.data?.users.length ? (
                <tr>
                  <td colSpan={5} className="text-[var(--muted)]">
                    {usersQuery.isLoading ? "Загрузка…" : "Нет пользователей"}
                  </td>
                </tr>
              ) : null}
            </tbody>
          </table>
        </div>
      </section>

      <section className="space-y-4">
        <h2 className="text-xl font-semibold">API keys</h2>
        <p className="text-sm text-[var(--muted)]">
          Ключи для programmatic access. При создании пользователя автоматически
          выпускается inference key.
        </p>
        <div className="table-wrap">
          <table className="data-table">
            <thead>
              <tr>
                <th>ID</th>
                <th>User</th>
                <th>Name</th>
                <th>Scopes</th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody>
              {(keysQuery.data?.api_keys ?? []).map((key) => (
                <tr key={key.id}>
                  <td>{key.id}</td>
                  <td>{key.username}</td>
                  <td>{key.name}</td>
                  <td>{key.scopes.join(", ")}</td>
                  <td>
                    <span
                      className={`badge ${key.active ? "badge-completed" : "badge-failed"}`}
                    >
                      {key.active ? "active" : "revoked"}
                    </span>
                  </td>
                </tr>
              ))}
              {!keysQuery.data?.api_keys.length ? (
                <tr>
                  <td colSpan={5} className="text-[var(--muted)]">
                    {keysQuery.isLoading ? "Загрузка…" : "Нет ключей"}
                  </td>
                </tr>
              ) : null}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
}
