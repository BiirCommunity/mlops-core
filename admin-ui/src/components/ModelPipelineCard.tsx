import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  AlertTriangle,
  ArrowRight,
  Box,
  CloudUpload,
  Database,
  RefreshCw,
  Rocket,
} from "lucide-react";
import { Link } from "react-router-dom";
import { trainingApi } from "@/api/client";
import type { ModelPipelineStatus } from "@/types";

function pipelineLabel(status: ModelPipelineStatus["pipeline_status"]) {
  switch (status) {
    case "ready":
      return "Готово";
    case "restart_required":
      return "Нужен restart";
    case "dvc_sync_required":
      return "Нужен DVC sync";
    case "missing_checkpoint":
      return "Нет checkpoint";
    default:
      return "Требует внимания";
  }
}

function pipelineBadge(status: ModelPipelineStatus["pipeline_status"]) {
  switch (status) {
    case "ready":
      return "badge-completed";
    case "restart_required":
      return "badge-running";
    case "missing_checkpoint":
      return "badge-failed";
    default:
      return "badge-queued";
  }
}

function dvcLabel(status: ModelPipelineStatus["dvc"]["status"]) {
  switch (status) {
    case "in_sync":
      return "DVC ✓";
    case "out_of_sync":
      return "DVC ≠ disk";
    case "remote_missing":
      return "DVC remote missing";
    case "untracked":
      return "DVC не tracked";
    case "missing_checkpoint":
      return "нет файла";
    default:
      return status;
  }
}

function formatTs(value?: number | null) {
  if (!value) return "—";
  return new Date(value * 1000).toLocaleString("ru-RU");
}

function formatSize(bytes?: number | null) {
  if (!bytes) return "—";
  if (bytes >= 1_073_741_824) return `${(bytes / 1_073_741_824).toFixed(2)} GB`;
  if (bytes >= 1_048_576) return `${(bytes / 1_048_576).toFixed(1)} MB`;
  return `${bytes} B`;
}

type Props = {
  status?: ModelPipelineStatus;
  loading?: boolean;
  compact?: boolean;
  onSyncDvc?: () => void;
  syncingDvc?: boolean;
};

export function ModelPipelineCard({
  status,
  loading,
  compact = false,
  onSyncDvc,
  syncingDvc = false,
}: Props) {
  const queryClient = useQueryClient();
  const statusQuery = useQuery({
    queryKey: ["model-pipeline"],
    queryFn: trainingApi.getModelStatus,
    refetchInterval: 15000,
    enabled: !status && !loading,
  });

  const syncMutation = useMutation({
    mutationFn: trainingApi.syncDvc,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["model-pipeline"] });
      queryClient.invalidateQueries({ queryKey: ["inference-model"] });
      queryClient.invalidateQueries({ queryKey: ["overview"] });
    },
  });

  const data = status ?? statusQuery.data;
  const isLoading = loading ?? statusQuery.isLoading;
  const syncDvc = onSyncDvc ?? (() => syncMutation.mutate());
  const syncing = syncingDvc || syncMutation.isPending;

  if (isLoading) {
    return (
      <div className={`card ${compact ? "p-4" : "p-5"}`}>
        <div className="skeleton mb-3 h-4 w-40" />
        <div className="skeleton h-24 w-full" />
      </div>
    );
  }

  if (!data) return null;

  const { inference, dvc, actions } = data;
  const versionLabel = inference.loaded_version
    ? `v${inference.loaded_version}`
    : inference.deployed_version
      ? `v${inference.deployed_version} (disk)`
      : "—";

  return (
    <div className={`card ${compact ? "p-4" : "p-5"} space-y-4`}>
      <div className="flex items-center justify-between gap-3">
        <div className="flex items-center gap-2 text-sm text-[var(--muted)]">
          <Box size={16} />
          Model pipeline
        </div>
        <span className={`badge ${pipelineBadge(data.pipeline_status)}`}>
          {pipelineLabel(data.pipeline_status)}
        </span>
      </div>

      <div className={`${compact ? "text-xl" : "text-2xl"} font-semibold`}>
        {inference.registry_name} {versionLabel}
      </div>

      <div className="grid gap-3 md:grid-cols-3">
        <div className="rounded-xl border border-[var(--border)] bg-[#0d1219] p-3">
          <div className="mb-1 flex items-center gap-2 text-xs uppercase tracking-wide text-[var(--muted)]">
            <Rocket size={14} />
            MLflow → disk
          </div>
          <div className="text-sm">
            Deploy: {formatTs(inference.deployed_at)}
          </div>
          <div className="mt-1 text-xs text-[var(--muted)]">
            {inference.object_name ?? "manifest не создан"}
          </div>
        </div>

        <div className="rounded-xl border border-[var(--border)] bg-[#0d1219] p-3">
          <div className="mb-1 flex items-center gap-2 text-xs uppercase tracking-wide text-[var(--muted)]">
            <Database size={14} />
            Disk checkpoint
          </div>
          <div className="text-sm">{formatSize(dvc.disk_size)}</div>
          <div className="mt-1 truncate font-mono text-xs text-[var(--muted)]">
            md5 {dvc.disk_md5?.slice(0, 12) ?? "—"}…
          </div>
        </div>

        <div className="rounded-xl border border-[var(--border)] bg-[#0d1219] p-3">
          <div className="mb-1 flex items-center gap-2 text-xs uppercase tracking-wide text-[var(--muted)]">
            <CloudUpload size={14} />
            DVC / MinIO
          </div>
          <div className="text-sm">{dvcLabel(dvc.status)}</div>
          <div className="mt-1 truncate text-xs text-[var(--muted)]">
            {dvc.remote_uri}
          </div>
        </div>
      </div>

      <div className="flex flex-wrap items-center gap-2 text-sm text-[var(--muted)]">
        <span className="rounded-lg bg-[#0d1219] px-2 py-1">Registry</span>
        <ArrowRight size={14} />
        <span className="rounded-lg bg-[#0d1219] px-2 py-1">Disk</span>
        <ArrowRight size={14} />
        <span className="rounded-lg bg-[#0d1219] px-2 py-1">DVC</span>
        <ArrowRight size={14} />
        <span className="rounded-lg bg-[#0d1219] px-2 py-1">Inference GPU</span>
      </div>

      {actions.restart_app ? (
        <div className="flex items-start gap-2 text-sm text-amber-200">
          <RefreshCw size={14} className="mt-0.5 shrink-0" />
          На диске v{inference.deployed_version}, в памяти v
          {inference.loaded_version ?? "?"} — требуется перезапуск сервиса.
        </div>
      ) : null}

      {actions.sync_dvc ? (
        <div className="flex flex-col gap-3 rounded-xl border border-amber-500/30 bg-amber-500/10 p-3 sm:flex-row sm:items-center sm:justify-between">
          <div className="flex items-start gap-2 text-sm text-amber-100">
            <AlertTriangle size={14} className="mt-0.5 shrink-0" />
            Checkpoint на диске не совпадает с DVC remote — нужен sync после Deploy.
          </div>
          <button
            type="button"
            className="btn btn-secondary shrink-0"
            disabled={syncing}
            onClick={syncDvc}
          >
            {syncing ? (
              <RefreshCw className="animate-spin" size={16} />
            ) : (
              <CloudUpload size={16} />
            )}
            Sync DVC
          </button>
        </div>
      ) : null}

      {!compact ? (
        <Link to="/models" className="inline-block text-sm text-[#93c5fd] hover:underline">
          Model registry
        </Link>
      ) : null}
    </div>
  );
}
