import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Link, useParams } from "react-router-dom";
import { LoaderCircle, OctagonX } from "lucide-react";
import { trainingApi } from "@/api/client";
import type { TrainingCheckpoint } from "@/types";
import { formatLoss, formatTimestamp, statusBadgeClass } from "@/lib/format";

function collectCheckpoints(job: {
  progress?: Record<string, unknown>;
  result?: Record<string, unknown>;
}): TrainingCheckpoint[] {
  const fromProgress = job.progress?.checkpoints;
  if (Array.isArray(fromProgress) && fromProgress.length) {
    return fromProgress as TrainingCheckpoint[];
  }
  const fromResult = job.result?.checkpoints;
  if (Array.isArray(fromResult) && fromResult.length) {
    return fromResult as TrainingCheckpoint[];
  }
  return [];
}

export function JobDetailPage() {
  const { jobId = "" } = useParams();
  const queryClient = useQueryClient();

  const jobQuery = useQuery({
    queryKey: ["job", jobId],
    queryFn: () => trainingApi.getJob(jobId),
    enabled: Boolean(jobId),
    refetchInterval: (query) => {
      const status = query.state.data?.status;
      return status === "running" || status === "queued" ? 3000 : false;
    },
  });

  const cancelMutation = useMutation({
    mutationFn: () => trainingApi.cancelJob(jobId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["job", jobId] });
      queryClient.invalidateQueries({ queryKey: ["jobs"] });
    },
  });

  const job = jobQuery.data;
  const checkpoints = job ? collectCheckpoints(job) : [];
  const canCancel = job?.status === "running" || job?.status === "queued";

  if (jobQuery.isLoading) {
    return <div className="skeleton h-40 w-full" />;
  }

  if (!job) {
    return (
      <div className="card p-6">
        <p>Job не найден.</p>
        <Link to="/jobs" className="mt-4 inline-block text-[#93c5fd]">
          Back to jobs
        </Link>
      </div>
    );
  }

  function handleCancel() {
    if (
      !window.confirm(
        "Остановить дообучение? Уже сохранённые чекпоинты эпох останутся доступны.",
      )
    ) {
      return;
    }
    cancelMutation.mutate();
  }

  return (
    <div className="space-y-6">
      <div className="flex flex-col gap-3 lg:flex-row lg:items-center lg:justify-between">
        <div>
          <div className="text-sm text-[var(--muted)]">Training job</div>
          <h1 className="mono text-3xl font-semibold">{job.id}</h1>
        </div>
        <div className="flex flex-wrap items-center gap-3">
          <span className={statusBadgeClass(job.status)}>{job.status}</span>
          {canCancel ? (
            <button
              type="button"
              className="btn btn-secondary"
              disabled={cancelMutation.isPending || job.cancel_requested}
              onClick={handleCancel}
            >
              {cancelMutation.isPending ? (
                <LoaderCircle className="animate-spin" size={18} />
              ) : (
                <OctagonX size={18} />
              )}
              {job.cancel_requested ? "Останавливается…" : "Остановить дообучение"}
            </button>
          ) : null}
        </div>
      </div>

      <div className="grid gap-4 md:grid-cols-4">
        <div className="card p-4">
          <div className="text-sm text-[var(--muted)]">Run name</div>
          <div className="mt-2 font-medium">{job.config.run_name}</div>
        </div>
        <div className="card p-4">
          <div className="text-sm text-[var(--muted)]">Loss</div>
          <div className="mt-2 font-medium">{formatLoss(job)}</div>
        </div>
        <div className="card p-4">
          <div className="text-sm text-[var(--muted)]">Started</div>
          <div className="mt-2 font-medium">{formatTimestamp(job.started_at)}</div>
        </div>
        <div className="card p-4">
          <div className="text-sm text-[var(--muted)]">Finished</div>
          <div className="mt-2 font-medium">{formatTimestamp(job.finished_at)}</div>
        </div>
      </div>

      {job.error ? (
        <div
          className={`rounded-xl border px-4 py-3 text-sm ${
            job.status === "cancelled"
              ? "border-amber-500/30 bg-amber-500/10 text-amber-100"
              : "border-red-500/30 bg-red-500/10 text-red-200"
          }`}
        >
          {job.error}
        </div>
      ) : null}

      {cancelMutation.error ? (
        <div className="rounded-xl border border-red-500/30 bg-red-500/10 px-4 py-3 text-sm text-red-200">
          {cancelMutation.error.message}
        </div>
      ) : null}

      {checkpoints.length ? (
        <div className="card p-5">
          <h2 className="mb-3 text-lg font-semibold">Чекпоинты по эпохам</h2>
          <p className="mb-4 text-sm text-[var(--muted)]">
            После каждой эпохи сохраняются LoRA adapter и merged model локально и в
            MinIO.
          </p>
          <div className="table-wrap">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Epoch</th>
                  <th>Local adapter</th>
                  <th>MinIO adapter</th>
                  <th>MinIO model</th>
                </tr>
              </thead>
              <tbody>
                {checkpoints.map((checkpoint) => (
                  <tr key={checkpoint.epoch}>
                    <td>{checkpoint.epoch}</td>
                    <td className="mono text-xs text-[var(--muted)]">
                      {checkpoint.adapter_path}
                    </td>
                    <td className="mono text-xs text-[var(--muted)]">
                      {checkpoint.minio_adapter_uri ?? "—"}
                    </td>
                    <td className="mono text-xs text-[var(--muted)]">
                      {checkpoint.minio_model_uri ?? "—"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      ) : null}

      <div className="grid gap-6 xl:grid-cols-2">
        <div className="card p-5">
          <h2 className="mb-3 text-lg font-semibold">Config</h2>
          <pre className="overflow-auto rounded-xl bg-[#0b1015] p-4 text-xs text-[#cbd5e1]">
            {JSON.stringify(job.config, null, 2)}
          </pre>
        </div>
        <div className="card p-5">
          <h2 className="mb-3 text-lg font-semibold">Progress / Result</h2>
          <pre className="overflow-auto rounded-xl bg-[#0b1015] p-4 text-xs text-[#cbd5e1]">
            {JSON.stringify(
              { progress: job.progress, result: job.result },
              null,
              2,
            )}
          </pre>
        </div>
      </div>
    </div>
  );
}
