type DistributionCompareProps = {
  title: string;
  baseline: Record<string, number>;
  current: Record<string, number>;
};

function total(values: Record<string, number>) {
  return Object.values(values).reduce((sum, value) => sum + value, 0) || 1;
}

export function DistributionCompare({
  title,
  baseline,
  current,
}: DistributionCompareProps) {
  const keys = Array.from(
    new Set([...Object.keys(baseline), ...Object.keys(current)]),
  ).sort((a, b) => (current[b] ?? 0) - (current[a] ?? 0));

  if (!keys.length) return null;

  const baselineTotal = total(baseline);
  const currentTotal = total(current);
  const maxShare = Math.max(
    ...keys.map((key) =>
      Math.max(
        (baseline[key] ?? 0) / baselineTotal,
        (current[key] ?? 0) / currentTotal,
      ),
    ),
    0.01,
  );

  return (
    <div className="rounded-2xl border border-[var(--border)] bg-[#0d1219] p-4">
      <h4 className="mb-4 text-sm font-semibold text-[#dbeafe]">{title}</h4>
      <div className="mb-2 flex gap-6 text-xs text-[var(--muted)]">
        <span className="inline-flex items-center gap-2">
          <span className="h-2 w-2 rounded-full bg-[#64748b]" />
          Baseline
        </span>
        <span className="inline-flex items-center gap-2">
          <span className="h-2 w-2 rounded-full bg-[var(--accent)]" />
          Current window
        </span>
      </div>
      <div className="space-y-3">
        {keys.map((key) => {
          const baseValue = baseline[key] ?? 0;
          const currentValue = current[key] ?? 0;
          const basePct = (baseValue / baselineTotal / maxShare) * 100;
          const currentPct = (currentValue / currentTotal / maxShare) * 100;
          return (
            <div key={key}>
              <div className="mb-1 flex items-center justify-between text-xs">
                <span className="font-medium text-[#cbd5e1]">{key}</span>
                <span className="text-[var(--muted)]">
                  {baseValue} → {currentValue}
                </span>
              </div>
              <div className="space-y-1">
                <div className="h-2 overflow-hidden rounded-full bg-[#1a2431]">
                  <div
                    className="h-full rounded-full bg-[#64748b] transition-all"
                    style={{ width: `${basePct}%` }}
                  />
                </div>
                <div className="h-2 overflow-hidden rounded-full bg-[#1a2431]">
                  <div
                    className="h-full rounded-full bg-[var(--accent)] transition-all"
                    style={{ width: `${currentPct}%` }}
                  />
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
