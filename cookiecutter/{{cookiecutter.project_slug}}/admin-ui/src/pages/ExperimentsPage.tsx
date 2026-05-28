import { useQuery } from "@tanstack/react-query";
import { ExternalLink, FlaskConical } from "lucide-react";
import { trainingApi } from "@/api/client";

export function ExperimentsPage() {
  const experimentsQuery = useQuery({
    queryKey: ["experiments"],
    queryFn: () => trainingApi.listExperiments(50),
    refetchInterval: 30000,
  });

  const runsQuery = useQuery({
    queryKey: ["experiment-runs"],
    queryFn: () => trainingApi.listExperimentRuns(undefined, 50),
    refetchInterval: 15000,
  });

  return (
    <div className="space-y-6">
      <div className="flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
        <div>
          <h1 className="text-3xl font-semibold">Experiments</h1>
          <p className="mt-2 text-[var(--muted)]">
            MLflow experiments и последние training runs.
          </p>
        </div>
        {experimentsQuery.data?.experiments[0] ? (
          <a
            href={import.meta.env.VITE_MLFLOW_UI_URL ?? "http://{{ cookiecutter.mlops_registry }}"}
            target="_blank"
            rel="noreferrer"
            className="btn btn-secondary"
          >
            <ExternalLink size={16} />
            Open MLflow UI
          </a>
        ) : null}
      </div>

      <div className="card p-5">
        <h2 className="mb-4 text-lg font-semibold">Experiments</h2>
        <div className="table-wrap">
          <table className="data-table">
            <thead>
              <tr>
                <th>Name</th>
                <th>ID</th>
                <th>Stage</th>
              </tr>
            </thead>
            <tbody>
              {(experimentsQuery.data?.experiments ?? []).map((item) => (
                <tr key={item.experiment_id}>
                  <td>{item.name}</td>
                  <td className="mono text-sm">{item.experiment_id}</td>
                  <td>{item.lifecycle_stage}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="card p-5">
        <div className="mb-4 flex items-center gap-2">
          <FlaskConical size={18} />
          <h2 className="text-lg font-semibold">Recent runs</h2>
        </div>
        <div className="table-wrap">
          <table className="data-table">
            <thead>
              <tr>
                <th>Run</th>
                <th>Status</th>
                <th>Loss</th>
                <th>Params</th>
              </tr>
            </thead>
            <tbody>
              {(runsQuery.data?.runs ?? []).map((run) => (
                <tr key={String(run.run_id)}>
                  <td>
                    <div className="font-medium">{String(run.run_name ?? run.run_id)}</div>
                    <div className="mono text-xs text-[var(--muted)]">
                      {String(run.run_id).slice(0, 12)}
                    </div>
                  </td>
                  <td>{String(run.status)}</td>
                  <td>
                    {typeof (run.metrics as Record<string, number> | undefined)?.final_loss ===
                    "number"
                      ? (run.metrics as Record<string, number>).final_loss.toFixed(4)
                      : "—"}
                  </td>
                  <td className="max-w-sm truncate text-sm text-[var(--muted)]">
                    {JSON.stringify(run.params)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
