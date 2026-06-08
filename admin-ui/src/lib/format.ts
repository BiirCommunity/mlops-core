export function formatTimestamp(value?: number | null) {
  if (!value) return "—";
  return new Date(value * 1000).toLocaleString("ru-RU");
}

export function formatLoss(job: {
  result?: Record<string, unknown>;
  progress?: Record<string, unknown>;
}) {
  const finalLoss = job.result?.final_loss;
  if (typeof finalLoss === "number") return finalLoss.toFixed(4);
  const progressLoss = job.progress?.loss ?? job.progress?.epoch_loss;
  if (typeof progressLoss === "number") return progressLoss.toFixed(4);
  return "—";
}

export function statusBadgeClass(status: string) {
  return `badge badge-${status}`;
}
