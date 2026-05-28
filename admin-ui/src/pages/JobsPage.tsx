import { useQuery } from "@tanstack/react-query";
import { Link } from "react-router-dom";
import { trainingApi } from "@/api/client";
import { formatLoss, formatTimestamp, statusBadgeClass } from "@/lib/format";

export function JobsPage() {
  const jobsQuery = useQuery({
    queryKey: ["jobs", 50],
    queryFn: () => trainingApi.listJobs(50),
    refetchInterval: 5000,
  });

  return (
    <div className="space-y-6">
      <div className="flex items-end justify-between gap-4">
        <div>
          <h1 className="text-3xl font-semibold">Jobs дообучения</h1>
          <p className="mt-2 text-[var(--muted)]">
            Активные и завершённые LoRA post-train jobs в Admin Studio.
          </p>
        </div>
        <Link to="/jobs/new" className="btn btn-primary">
          New run
        </Link>
      </div>

      <div className="table-wrap">
        <table className="data-table">
          <thead>
            <tr>
              <th>ID</th>
              <th>Status</th>
              <th>Run</th>
              <th>Dataset</th>
              <th>Loss</th>
              <th>Created</th>
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
                  <span className={statusBadgeClass(job.status)}>
                    {job.status}
                  </span>
                </td>
                <td>{job.config.run_name}</td>
                <td className="max-w-xs truncate mono text-sm text-[var(--muted)]">
                  {job.config.dataset_path}
                </td>
                <td>{formatLoss(job)}</td>
                <td>{formatTimestamp(job.created_at)}</td>
              </tr>
            ))}
            {!jobsQuery.data?.jobs.length ? (
              <tr>
                <td colSpan={6} className="text-[var(--muted)]">
                  Jobs пока нет
                </td>
              </tr>
            ) : null}
          </tbody>
        </table>
      </div>
    </div>
  );
}
