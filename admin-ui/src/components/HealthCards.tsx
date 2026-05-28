import { ExternalLink } from "lucide-react";
import type { TrainingHealth } from "@/types";

type Props = {
  health?: TrainingHealth;
  loading?: boolean;
};

function ServiceCard({
  title,
  ok,
  message,
  href,
}: {
  title: string;
  ok?: boolean;
  message?: string;
  href?: string;
}) {
  return (
    <div className="card p-5">
      <div className="mb-3 flex items-center justify-between gap-3">
        <div className="text-sm text-[var(--muted)]">{title}</div>
        <span
          className={`badge ${ok ? "badge-completed" : "badge-failed"}`}
        >
          {ok ? "online" : "degraded"}
        </span>
      </div>
      <div className="text-sm">{message ?? "—"}</div>
      {href ? (
        <a
          href={href}
          target="_blank"
          rel="noreferrer"
          className="mt-4 inline-flex items-center gap-2 text-sm text-[#93c5fd] hover:underline"
        >
          Open console
          <ExternalLink size={14} />
        </a>
      ) : null}
    </div>
  );
}

export function HealthCards({ health, loading }: Props) {
  if (loading) {
    return (
      <div className="grid gap-4 md:grid-cols-3">
        {[1, 2, 3].map((item) => (
          <div key={item} className="card h-32 p-5">
            <div className="skeleton mb-3 h-4 w-24" />
            <div className="skeleton h-8 w-full" />
          </div>
        ))}
      </div>
    );
  }

  return (
    <div className="grid gap-4 md:grid-cols-3">
      <ServiceCard
        title="Platform"
        ok={health?.status === "ok"}
        message={
          health?.status === "ok"
            ? "Admin Studio API, MLflow и MinIO доступны."
            : "Один или несколько сервисов недоступны."
        }
      />
      <ServiceCard
        title="MLflow"
        ok={health?.mlflow.ok}
        message={health?.mlflow.message}
        href={health?.mlflow_ui}
      />
      <ServiceCard
        title="MinIO"
        ok={health?.minio.ok}
        message={health?.minio.message}
        href={health?.minio_console || undefined}
      />
    </div>
  );
}
