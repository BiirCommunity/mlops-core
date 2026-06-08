import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { LoaderCircle, Rocket, UploadCloud } from "lucide-react";
import { trainingApi } from "@/api/client";
import { ModelPipelineCard } from "@/components/ModelPipelineCard";
import { statusBadgeClass } from "@/lib/format";

export function ModelsPage() {
  const queryClient = useQueryClient();
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const modelsQuery = useQuery({
    queryKey: ["models", 50],
    queryFn: () => trainingApi.listModels(50),
    refetchInterval: 15000,
  });

  const checkpointsQuery = useQuery({
    queryKey: ["training-checkpoints"],
    queryFn: trainingApi.listTrainingCheckpoints,
    refetchInterval: 15000,
  });

  const pipelineQuery = useQuery({
    queryKey: ["model-pipeline"],
    queryFn: trainingApi.getModelStatus,
    refetchInterval: 15000,
  });

  const deployMutation = useMutation({
    mutationFn: (version: string) => trainingApi.deployModel(version),
    onSuccess: (result) => {
      setError(null);
      const dvcOk = result.dvc_sync?.status === "synced";
      if (result.pending_reload) {
        setMessage(
          `v${result.version} на диске${dvcOk ? ", DVC sync ✓" : ""}. Перезапустите inference: docker compose restart app`,
        );
      } else if (dvcOk) {
        setMessage(`v${result.version} задеплоена, DVC и inference синхронизированы.`);
      } else {
        setMessage(
          `v${result.version} на диске. DVC: ${result.dvc_sync?.status ?? "unknown"} — при необходимости нажмите Sync DVC.`,
        );
      }
      queryClient.invalidateQueries({ queryKey: ["models"] });
      queryClient.invalidateQueries({ queryKey: ["model-pipeline"] });
      queryClient.invalidateQueries({ queryKey: ["overview"] });
    },
    onError: (err: Error) => {
      setMessage(null);
      setError(err.message);
    },
  });

  const registerMutation = useMutation({
    mutationFn: ({ jobId, epoch }: { jobId: string; epoch: number }) =>
      trainingApi.registerTrainingCheckpoint(jobId, epoch),
    onSuccess: (result) => {
      setError(null);
      if (result.status === "already_registered") {
        setMessage(`Checkpoint уже в registry: v${result.model_version}`);
      } else {
        setMessage(
          `Checkpoint зарегистрирован как v${result.model_version}. Теперь можно Deploy.`,
        );
      }
      queryClient.invalidateQueries({ queryKey: ["models"] });
      queryClient.invalidateQueries({ queryKey: ["training-checkpoints"] });
    },
    onError: (err: Error) => {
      setMessage(null);
      setError(err.message);
    },
  });

  const pendingCheckpoints = (checkpointsQuery.data?.checkpoints ?? []).filter(
    (checkpoint) => !checkpoint.registered_version,
  );

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-3xl font-semibold">Model registry</h1>
        <p className="mt-2 text-[var(--muted)]">
          Единый pipeline: MLflow Registry → disk → DVC (MinIO) → inference GPU.
          Deploy автоматически обновляет DVC; после смены версии перезапустите app.
        </p>
      </div>

      {message ? (
        <div className="rounded-xl border border-green-500/30 bg-green-500/10 px-4 py-3 text-sm text-green-100">
          {message}
        </div>
      ) : null}
      {error ? (
        <div className="rounded-xl border border-red-500/30 bg-red-500/10 px-4 py-3 text-sm text-red-200">
          {error}
        </div>
      ) : null}

      <ModelPipelineCard
        status={pipelineQuery.data}
        loading={pipelineQuery.isLoading}
      />

      <section className="space-y-4">
        <div>
          <h2 className="text-xl font-semibold">Checkpoints дообучения</h2>
          <p className="mt-1 text-sm text-[var(--muted)]">
            Сохранённые эпохи из cancelled/failed/completed jobs, ещё не в
            registry.
          </p>
        </div>
        <div className="table-wrap">
          <table className="data-table">
            <thead>
              <tr>
                <th>Job</th>
                <th>Epoch</th>
                <th>Status</th>
                <th>Run</th>
                <th>Local</th>
                <th></th>
              </tr>
            </thead>
            <tbody>
              {pendingCheckpoints.map((checkpoint) => (
                <tr key={`${checkpoint.job_id}-${checkpoint.epoch}`}>
                  <td className="mono text-sm">{checkpoint.job_id}</td>
                  <td>{checkpoint.epoch}</td>
                  <td>
                    <span className={statusBadgeClass(checkpoint.job_status)}>
                      {checkpoint.job_status}
                    </span>
                  </td>
                  <td>{checkpoint.run_name}</td>
                  <td>{checkpoint.available_locally ? "yes" : "no"}</td>
                  <td>
                    <button
                      className="btn btn-secondary !px-3 !py-2 text-sm"
                      disabled={
                        registerMutation.isPending ||
                        !checkpoint.available_locally
                      }
                      onClick={() =>
                        registerMutation.mutate({
                          jobId: checkpoint.job_id,
                          epoch: checkpoint.epoch,
                        })
                      }
                    >
                      {registerMutation.isPending ? (
                        <LoaderCircle className="animate-spin" size={16} />
                      ) : (
                        <UploadCloud size={16} />
                      )}
                      Register
                    </button>
                  </td>
                </tr>
              ))}
              {!pendingCheckpoints.length ? (
                <tr>
                  <td colSpan={6} className="text-[var(--muted)]">
                    Нет незарегистрированных checkpoints
                  </td>
                </tr>
              ) : null}
            </tbody>
          </table>
        </div>
      </section>

      <section className="space-y-4">
        <div>
          <h2 className="text-xl font-semibold">MLflow Registry</h2>
          <p className="mt-1 text-sm text-[var(--muted)]">
            Deploy: MinIO → /models/model.pt + DVC sync. Restart app для загрузки в GPU.
          </p>
        </div>
        <div className="table-wrap">
          <table className="data-table">
            <thead>
              <tr>
                <th>Name</th>
                <th>Version</th>
                <th>Stage</th>
                <th>Run</th>
                <th>Status</th>
                <th></th>
              </tr>
            </thead>
            <tbody>
              {(modelsQuery.data?.versions ?? []).map((model) => {
                const active = pipelineQuery.data?.inference;
                const isLoaded = active?.loaded_version === model.version;
                const isDeployedPending =
                  active?.pending_reload && active?.deployed_version === model.version;

                return (
                <tr key={`${model.name}-${model.version}`}>
                  <td>{model.name}</td>
                  <td>
                    v{model.version}
                    {isLoaded ? (
                      <span className="ml-2 badge badge-completed">in memory</span>
                    ) : null}
                    {isDeployedPending ? (
                      <span className="ml-2 badge badge-running">on disk</span>
                    ) : null}
                  </td>
                  <td>{model.stage || "None"}</td>
                  <td className="mono text-sm">{model.run_id.slice(0, 12)}</td>
                  <td>{model.status}</td>
                  <td>
                    <button
                      className="btn btn-secondary !px-3 !py-2 text-sm"
                      disabled={deployMutation.isPending}
                      onClick={() => deployMutation.mutate(model.version)}
                    >
                      {deployMutation.isPending ? (
                        <LoaderCircle className="animate-spin" size={16} />
                      ) : (
                        <Rocket size={16} />
                      )}
                      Deploy
                    </button>
                  </td>
                </tr>
                );
              })}
              {!modelsQuery.data?.versions.length ? (
                <tr>
                  <td colSpan={6} className="text-[var(--muted)]">
                    В registry пока нет версий
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
