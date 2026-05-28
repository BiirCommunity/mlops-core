import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Link } from "react-router-dom";
import {
  AlertTriangle,
  ArrowRight,
  BellRing,
  PlayCircle,
  Rocket,
} from "lucide-react";
import { trainingApi } from "@/api/client";
import { HealthCards } from "@/components/HealthCards";
import { ModelPipelineCard } from "@/components/ModelPipelineCard";
import { formatLoss } from "@/lib/format";
import type { ModelPipelineStatus } from "@/types";

export function DashboardPage() {
  const queryClient = useQueryClient();
  const healthQuery = useQuery({
    queryKey: ["health"],
    queryFn: trainingApi.getHealth,
    refetchInterval: 15000,
  });
  const overviewQuery = useQuery({
    queryKey: ["overview"],
    queryFn: trainingApi.getOverview,
    refetchInterval: 10000,
  });
  const alertsQuery = useQuery({
    queryKey: ["drift-alerts", 5],
    queryFn: () => trainingApi.getDriftAlerts(5),
    refetchInterval: 15000,
  });
  const jobsQuery = useQuery({
    queryKey: ["jobs", 5],
    queryFn: () => trainingApi.listJobs(5),
    refetchInterval: 5000,
  });

  const reportsQuery = useQuery({
    queryKey: ["drift-reports", 1],
    queryFn: () => trainingApi.listDriftReports(1),
    refetchInterval: 30000,
  });

  const posttrainMutation = useMutation({
    mutationFn: () =>
      trainingApi.startJob({
        run_name: `posttrain-${new Date().toISOString().slice(0, 19)}`,
        epochs: 3,
        batch_size: 2,
        learning_rate: 0.0002,
        max_seq_len: 512,
        lora_rank: 8,
        lora_alpha: 16,
        gradient_accumulation_steps: 1,
        seed: 42,
        dataset_path: null,
      }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["jobs"] });
    },
  });

  const interactions = overviewQuery.data?.interactions as
    | { total?: number; anomalous?: number }
    | undefined;
  const alerts = alertsQuery.data?.alerts ?? [];
  const latestReport = reportsQuery.data?.reports[0];

  return (
    <div className="space-y-8">
      <section className="flex flex-col gap-4 xl:flex-row xl:items-end xl:justify-between">
        <div>
          <h1 className="text-3xl font-semibold tracking-tight">Dashboard</h1>
          <p className="mt-2 max-w-2xl text-[var(--muted)]">
            Admin Studio — мониторинг Q&A, drift, экспериментов и LoRA post-train.
          </p>
        </div>
        <div className="flex flex-wrap gap-3">
          <button
            type="button"
            className="btn btn-primary"
            disabled={posttrainMutation.isPending}
            onClick={() => posttrainMutation.mutate()}
          >
            <Rocket size={18} />
            Запустить дообучение
          </button>
          <Link to="/jobs/new" className="btn btn-secondary">
            <PlayCircle size={18} />
            Настроить run
          </Link>
        </div>
      </section>

      {alerts.length ? (
        <section className="rounded-2xl border border-amber-500/30 bg-amber-500/10 p-4">
          <div className="mb-3 flex items-center gap-2 font-medium text-amber-100">
            <BellRing size={18} />
            Drift notifications ({alerts.length})
          </div>
          <div className="space-y-2">
            {alerts.slice(0, 3).map((alert, index) => (
              <div key={`${alert.source}-${index}`} className="text-sm text-amber-50/90">
                <AlertTriangle size={14} className="mr-2 inline" />
                {alert.message ?? alert.summary}
              </div>
            ))}
          </div>
          <Link to="/drift" className="mt-3 inline-block text-sm text-[#fde68a] hover:underline">
            Все alerts
          </Link>
        </section>
      ) : null}

      <HealthCards health={healthQuery.data} loading={healthQuery.isLoading} />

      <section className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
        <ModelPipelineCard
          status={overviewQuery.data?.inference_model as ModelPipelineStatus | undefined}
          loading={overviewQuery.isLoading}
          compact
        />
        <div className="card p-5">
          <div className="text-sm text-[var(--muted)]">Q&A total</div>
          <div className="mt-2 text-3xl font-semibold">{interactions?.total ?? 0}</div>
          <Link to="/conversations" className="mt-3 inline-block text-sm text-[#93c5fd]">
            История
          </Link>
        </div>
        <div className="card p-5">
          <div className="text-sm text-[var(--muted)]">Anomalies</div>
          <div className="mt-2 text-3xl font-semibold text-[#fca5a5]">
            {interactions?.anomalous ?? 0}
          </div>
          <Link
            to="/conversations"
            className="mt-3 inline-block text-sm text-[#93c5fd]"
          >
            Смотреть flagged
          </Link>
        </div>
        <div className="card p-5">
          <div className="text-sm text-[var(--muted)]">Drift alerts</div>
          <div className="mt-2 text-3xl font-semibold">{alertsQuery.data?.count ?? 0}</div>
          <Link to="/drift" className="mt-3 inline-block text-sm text-[#93c5fd]">
            Подробнее
          </Link>
        </div>
        <div className="card p-5">
          <div className="text-sm text-[var(--muted)]">Latest drift report</div>
          <div className="mt-2 text-lg font-semibold">
            {latestReport?.severity ?? "—"}
          </div>
          <p className="mt-2 line-clamp-2 text-sm text-[var(--muted)]">
            {latestReport?.summary ?? "Отчётов пока нет"}
          </p>
          <Link to="/drift" className="mt-3 inline-block text-sm text-[#93c5fd]">
            Все отчёты
          </Link>
        </div>
      </section>

      <section className="grid gap-6 xl:grid-cols-2">
        <div className="card p-5">
          <div className="mb-4 flex items-center justify-between">
            <h2 className="text-lg font-semibold">Recent jobs</h2>
            <Link to="/jobs" className="text-sm text-[#93c5fd] hover:underline">
              View all
            </Link>
          </div>
          <div className="table-wrap">
            <table className="data-table">
              <thead>
                <tr>
                  <th>ID</th>
                  <th>Status</th>
                  <th>Run</th>
                  <th>Loss</th>
                </tr>
              </thead>
              <tbody>
                {(jobsQuery.data?.jobs ?? []).map((job) => (
                  <tr key={job.id}>
                    <td>
                      <Link
                        to={`/jobs/${job.id}`}
                        className="mono text-[#93c5fd] hover:underline"
                      >
                        {job.id}
                      </Link>
                    </td>
                    <td>
                      <span className={`badge badge-${job.status}`}>{job.status}</span>
                    </td>
                    <td>{job.config.run_name}</td>
                    <td>{formatLoss(job)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        <div className="card p-5">
          <div className="mb-4 flex items-center justify-between">
            <h2 className="text-lg font-semibold">Experiments</h2>
            <Link to="/experiments" className="text-sm text-[#93c5fd] hover:underline">
              Open
            </Link>
          </div>
          <div className="space-y-3">
            {(jobsQuery.data?.jobs ?? []).slice(0, 4).map((job) => (
              <div
                key={job.id}
                className="flex items-center justify-between rounded-xl border border-[var(--border)] bg-[#0d1219] px-4 py-3"
              >
                <div>
                  <div className="font-medium">{job.config.run_name}</div>
                  <div className="text-sm text-[var(--muted)]">{job.status}</div>
                </div>
                <ArrowRight size={16} className="text-[var(--muted)]" />
              </div>
            ))}
          </div>
        </div>
      </section>

      {posttrainMutation.isSuccess ? (
        <div className="rounded-xl border border-green-500/30 bg-green-500/10 px-4 py-3 text-sm text-green-100">
          Job {posttrainMutation.data.id} поставлен в очередь.
        </div>
      ) : null}
      {posttrainMutation.error ? (
        <div className="rounded-xl border border-red-500/30 bg-red-500/10 px-4 py-3 text-sm text-red-200">
          {posttrainMutation.error.message}
        </div>
      ) : null}
    </div>
  );
}
