import { useEffect, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { BellRing, FileText, Radar } from "lucide-react";
import { DriftReportView } from "@/components/DriftReportView";
import { trainingApi } from "@/api/client";

function severityClass(severity: string) {
  if (severity === "red") return "badge-failed";
  if (severity === "yellow") return "badge-running";
  return "badge-completed";
}

function severityBorder(severity: string) {
  if (severity === "red") return "border-red-500/40 ring-1 ring-red-500/20";
  if (severity === "yellow") return "border-amber-500/40 ring-1 ring-amber-500/20";
  return "border-green-500/30";
}

function formatReportTime(value: string) {
  const parsed = Date.parse(value);
  if (Number.isNaN(parsed)) return value;
  return new Date(parsed).toLocaleString("ru-RU");
}

function shortReportId(reportId: string) {
  const match = reportId.match(/drift-(\d{8}T\d{6})/);
  return match ? match[1].replace("T", " ") : reportId.slice(0, 24);
}

export function DriftAlertsPage() {
  const [tab, setTab] = useState<"reports" | "live">("reports");
  const [selectedReportId, setSelectedReportId] = useState<string | null>(null);

  const alertsQuery = useQuery({
    queryKey: ["drift-alerts"],
    queryFn: () => trainingApi.getDriftAlerts(50),
    refetchInterval: 15000,
  });

  const reportsQuery = useQuery({
    queryKey: ["drift-reports"],
    queryFn: () => trainingApi.listDriftReports(50),
    refetchInterval: 30000,
  });

  const reports = reportsQuery.data?.reports ?? [];
  const selectedReport =
    reports.find((item) => item.report_id === selectedReportId) ??
    reports[0] ??
    null;

  useEffect(() => {
    if (!selectedReportId && reports[0]?.report_id) {
      setSelectedReportId(reports[0].report_id);
    }
  }, [reports, selectedReportId]);

  const snapshot = alertsQuery.data?.live_snapshot;

  return (
    <div className="space-y-6">
      <div className="flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
        <div>
          <h1 className="text-3xl font-semibold">Drift</h1>
          <p className="mt-2 text-[var(--muted)]">
            Отчёты из <code className="mono text-[#93c5fd]">reports/drift/</code>{" "}
            и live-мониторинг Admin Studio.
          </p>
        </div>
        <div className="flex gap-2 rounded-xl border border-[var(--border)] bg-[#0d1219] p-1">
          <button
            type="button"
            className={`rounded-lg px-4 py-2 text-sm font-medium transition ${
              tab === "reports"
                ? "bg-[var(--accent-soft)] text-[#dbeafe]"
                : "text-[var(--muted)] hover:text-white"
            }`}
            onClick={() => setTab("reports")}
          >
            <FileText size={14} className="mr-2 inline" />
            Отчёты ({reports.length})
          </button>
          <button
            type="button"
            className={`rounded-lg px-4 py-2 text-sm font-medium transition ${
              tab === "live"
                ? "bg-[var(--accent-soft)] text-[#dbeafe]"
                : "text-[var(--muted)] hover:text-white"
            }`}
            onClick={() => setTab("live")}
          >
            <BellRing size={14} className="mr-2 inline" />
            Live
          </button>
        </div>
      </div>

      {tab === "reports" ? (
        <div className="grid gap-6 xl:grid-cols-[320px_1fr]">
          <aside className="space-y-3">
            <div className="text-sm font-medium text-[var(--muted)]">
              История отчётов
            </div>
            <div className="max-h-[70vh] space-y-2 overflow-auto pr-1">
              {reports.map((report) => {
                const active = selectedReport?.report_id === report.report_id;
                return (
                  <button
                    key={report.report_id}
                    type="button"
                    onClick={() => setSelectedReportId(report.report_id)}
                    className={[
                      "w-full rounded-2xl border bg-[#0d1219] p-4 text-left transition hover:bg-white/5",
                      active
                        ? `${severityBorder(report.severity)} bg-[var(--accent-soft)]`
                        : "border-[var(--border)]",
                    ].join(" ")}
                  >
                    <div className="mb-2 flex items-center justify-between gap-2">
                      <span className={`badge ${severityClass(report.severity)}`}>
                        {report.severity}
                      </span>
                      <span className="text-xs text-[var(--muted)]">
                        {report.status}
                      </span>
                    </div>
                    <div className="mono text-sm font-medium text-[#dbeafe]">
                      {shortReportId(report.report_id)}
                    </div>
                    <div className="mt-1 text-xs text-[var(--muted)]">
                      {formatReportTime(report.generated_at)}
                    </div>
                    <div className="mt-3 grid grid-cols-3 gap-2 text-center text-xs">
                      <div className="rounded-lg bg-black/20 px-2 py-1">
                        <div className="text-[var(--muted)]">data</div>
                        <div>{Number(report.data_drift.score ?? 0).toFixed(2)}</div>
                      </div>
                      <div className="rounded-lg bg-black/20 px-2 py-1">
                        <div className="text-[var(--muted)]">concept</div>
                        <div>
                          {Number(report.concept_drift.score ?? 0).toFixed(2)}
                        </div>
                      </div>
                      <div className="rounded-lg bg-black/20 px-2 py-1">
                        <div className="text-[var(--muted)]">target</div>
                        <div>
                          {Number(report.target_drift.score ?? 0).toFixed(2)}
                        </div>
                      </div>
                    </div>
                  </button>
                );
              })}
              {!reports.length ? (
                <div className="card p-6 text-sm text-[var(--muted)]">
                  {reportsQuery.isLoading
                    ? "Загрузка отчётов..."
                    : "В reports/drift пока нет JSON-отчётов."}
                </div>
              ) : null}
            </div>
          </aside>

          <div>
            {selectedReport ? (
              <DriftReportView report={selectedReport} />
            ) : (
              <div className="card flex min-h-[420px] items-center justify-center p-8 text-[var(--muted)]">
                Выберите отчёт слева
              </div>
            )}
          </div>
        </div>
      ) : (
        <div className="space-y-6">
          {snapshot ? (
            <div className="grid gap-4 md:grid-cols-3">
              {(
                [
                  ["data_drift_score", "Data drift"],
                  ["concept_drift_score", "Concept drift"],
                  ["target_drift_score", "Target drift"],
                ] as const
              ).map(([key, label]) => (
                <div key={key} className="card p-4">
                  <div className="text-sm text-[var(--muted)]">{label}</div>
                  <div className="mt-2 text-3xl font-semibold">
                    {Number(snapshot[key] ?? 0).toFixed(3)}
                  </div>
                </div>
              ))}
            </div>
          ) : null}

          <section className="space-y-3">
            <h2 className="flex items-center gap-2 text-lg font-semibold">
              <Radar size={18} />
              Alerts
            </h2>
            {(alertsQuery.data?.alerts ?? []).map((alert, index) => (
              <div
                key={`${alert.source}-${alert.report_id ?? alert.kind}-${index}`}
                className="card p-5"
              >
                <div className="mb-2 flex items-center gap-3">
                  <span className={`badge ${severityClass(alert.severity)}`}>
                    {alert.severity}
                  </span>
                  <span className="text-sm text-[var(--muted)]">{alert.source}</span>
                </div>
                <div className="text-sm">
                  {alert.message ?? alert.summary ?? "Drift threshold exceeded"}
                </div>
              </div>
            ))}
            {!alertsQuery.data?.alerts.length ? (
              <div className="card flex items-center gap-3 p-8 text-[var(--muted)]">
                <BellRing size={18} />
                Активных drift alerts нет.
              </div>
            ) : null}
          </section>
        </div>
      )}
    </div>
  );
}
