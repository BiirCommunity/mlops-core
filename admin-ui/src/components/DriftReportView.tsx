import {
  Activity,
  BarChart3,
  Gauge,
  Languages,
  MessageSquare,
  Star,
  Target,
} from "lucide-react";
import { DistributionCompare } from "@/components/DistributionCompare";
import type { DriftReport } from "@/types";

function formatReportTime(value: string) {
  const parsed = Date.parse(value);
  if (Number.isNaN(parsed)) return value;
  return new Date(parsed).toLocaleString("ru-RU");
}

function scoreTone(score: number) {
  if (score >= 0.7) return "danger";
  if (score >= 0.4) return "warning";
  return "success";
}

const toneStyles = {
  danger: {
    bar: "bg-[var(--danger)]",
    text: "text-[#fca5a5]",
    ring: "stroke-[var(--danger)]",
    glow: "shadow-[0_0_40px_rgba(239,68,68,0.15)]",
    border: "border-red-500/30",
    bg: "from-red-500/10 to-transparent",
  },
  warning: {
    bar: "bg-[var(--warning)]",
    text: "text-[#fde68a]",
    ring: "stroke-[var(--warning)]",
    glow: "shadow-[0_0_40px_rgba(245,158,11,0.12)]",
    border: "border-amber-500/30",
    bg: "from-amber-500/10 to-transparent",
  },
  success: {
    bar: "bg-[var(--success)]",
    text: "text-[#86efac]",
    ring: "stroke-[var(--success)]",
    glow: "shadow-[0_0_40px_rgba(34,197,94,0.12)]",
    border: "border-green-500/30",
    bg: "from-green-500/10 to-transparent",
  },
} as const;

function ScoreGauge({
  label,
  score,
  icon: Icon,
}: {
  label: string;
  score: number;
  icon: typeof Gauge;
}) {
  const tone = scoreTone(score);
  const style = toneStyles[tone];
  const pct = Math.min(Math.max(score * 100, 0), 100);
  const radius = 42;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference - (pct / 100) * circumference;

  return (
    <div className={`card p-5 ${style.glow}`}>
      <div className="mb-4 flex items-center gap-2 text-sm text-[var(--muted)]">
        <Icon size={16} />
        {label}
      </div>
      <div className="flex items-center gap-5">
        <div className="relative h-28 w-28 shrink-0">
          <svg className="h-28 w-28 -rotate-90" viewBox="0 0 100 100">
            <circle
              cx="50"
              cy="50"
              r={radius}
              fill="none"
              stroke="#1a2431"
              strokeWidth="8"
            />
            <circle
              cx="50"
              cy="50"
              r={radius}
              fill="none"
              className={style.ring}
              strokeWidth="8"
              strokeLinecap="round"
              strokeDasharray={circumference}
              strokeDashoffset={offset}
            />
          </svg>
          <div className="absolute inset-0 flex items-center justify-center">
            <span className={`text-2xl font-semibold ${style.text}`}>
              {score.toFixed(2)}
            </span>
          </div>
        </div>
        <div className="min-w-0 flex-1">
          <div className="mb-2 h-2 overflow-hidden rounded-full bg-[#1a2431]">
            <div
              className={`h-full rounded-full ${style.bar}`}
              style={{ width: `${pct}%` }}
            />
          </div>
          <p className="text-xs leading-relaxed text-[var(--muted)]">
            {tone === "danger"
              ? "Высокий drift — стоит проверить данные и поведение модели."
              : tone === "warning"
                ? "Умеренный drift — мониторинг и возможное дообучение."
                : "Drift в норме относительно baseline."}
          </p>
        </div>
      </div>
    </div>
  );
}

function MetricRow({
  label,
  value,
  warn,
  critical,
}: {
  label: string;
  value: number;
  warn: number;
  critical: number;
}) {
  const level =
    value >= critical ? "critical" : value >= warn ? "warn" : "ok";
  const colors = {
    ok: "text-[#86efac]",
    warn: "text-[#fde68a]",
    critical: "text-[#fca5a5]",
  };

  return (
    <div className="flex items-center justify-between gap-4 rounded-xl border border-[var(--border)] bg-[#0b1015] px-4 py-3">
      <span className="text-sm text-[var(--muted)]">{label}</span>
      <span className={`font-mono text-sm font-semibold ${colors[level]}`}>
        {value.toFixed(4)}
      </span>
    </div>
  );
}

function distributionBlock(
  distributions: DriftReport["distributions"],
  key: string,
) {
  const block = distributions as Record<
    string,
    Record<string, Record<string, number>>
  >;
  return {
    baseline: block.baseline?.[key] ?? {},
    current: block.current?.[key] ?? {},
  };
}

export function DriftReportView({ report }: { report: DriftReport }) {
  const tone = scoreTone(report.data_drift.score ?? 0);
  const headerStyle = toneStyles[tone];
  const warn = report.thresholds.psi_warn ?? 0.2;
  const critical = report.thresholds.psi_critical ?? 0.5;

  const promptLang = distributionBlock(report.distributions, "prompt_languages");
  const responseLang = distributionBlock(
    report.distributions,
    "response_languages",
  );
  const userRatings = distributionBlock(report.distributions, "user_ratings");
  const toxicity = distributionBlock(report.distributions, "toxicity_tiers");
  const promptLengths = distributionBlock(report.distributions, "prompt_lengths");
  const responseLengths = distributionBlock(
    report.distributions,
    "response_lengths",
  );

  return (
    <article className={`card overflow-hidden border ${headerStyle.border}`}>
      <div className={`bg-gradient-to-r ${headerStyle.bg} px-6 py-5`}>
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div>
            <div className="mb-2 flex flex-wrap items-center gap-2">
              <span className={`badge badge-${report.severity === "red" ? "failed" : report.severity === "yellow" ? "running" : "completed"}`}>
                {report.severity}
              </span>
              <span className="badge badge-queued">{report.status}</span>
            </div>
            <h2 className="mono text-lg font-semibold text-[#dbeafe]">
              {report.report_id}
            </h2>
            <p className="mt-1 text-sm text-[var(--muted)]">
              {formatReportTime(report.generated_at)}
            </p>
          </div>
          <div className="grid grid-cols-2 gap-3 text-sm sm:grid-cols-4">
            <div className="rounded-xl bg-black/20 px-3 py-2">
              <div className="text-[var(--muted)]">Baseline</div>
              <div className="font-semibold">{report.windows.baseline_samples}</div>
            </div>
            <div className="rounded-xl bg-black/20 px-3 py-2">
              <div className="text-[var(--muted)]">Window</div>
              <div className="font-semibold">{report.windows.window_samples}</div>
            </div>
            <div className="rounded-xl bg-black/20 px-3 py-2">
              <div className="text-[var(--muted)]">Locked</div>
              <div className="font-semibold">
                {report.windows.baseline_locked ? "yes" : "no"}
              </div>
            </div>
            <div className="rounded-xl bg-black/20 px-3 py-2">
              <div className="text-[var(--muted)]">PSI warn</div>
              <div className="font-semibold">{warn}</div>
            </div>
          </div>
        </div>
        <p className="mt-4 max-w-4xl text-sm leading-relaxed text-[#cbd5e1]">
          {report.summary}
        </p>
      </div>

      <div className="space-y-6 p-6">
        <section className="grid gap-4 xl:grid-cols-3">
          <ScoreGauge
            label="Data drift"
            score={Number(report.data_drift.score ?? 0)}
            icon={BarChart3}
          />
          <ScoreGauge
            label="Concept drift"
            score={Number(report.concept_drift.score ?? 0)}
            icon={Activity}
          />
          <ScoreGauge
            label="Target drift"
            score={Number(report.target_drift.score ?? 0)}
            icon={Target}
          />
        </section>

        <section className="grid gap-4 lg:grid-cols-3">
          <div className="space-y-3">
            <h3 className="flex items-center gap-2 text-sm font-semibold uppercase tracking-wide text-[var(--muted)]">
              <BarChart3 size={14} />
              Data metrics
            </h3>
            <MetricRow
              label="Embedding distance"
              value={Number(report.data_drift.embedding_distance ?? 0)}
              warn={0.1}
              critical={0.2}
            />
            <MetricRow
              label="Prompt length PSI"
              value={Number(report.data_drift.prompt_length_psi ?? 0)}
              warn={warn}
              critical={critical}
            />
            <MetricRow
              label="Language PSI"
              value={Number(report.data_drift.language_psi ?? 0)}
              warn={warn}
              critical={critical}
            />
          </div>

          <div className="space-y-3">
            <h3 className="flex items-center gap-2 text-sm font-semibold uppercase tracking-wide text-[var(--muted)]">
              <MessageSquare size={14} />
              Concept metrics
            </h3>
            <MetricRow
              label="Response embedding distance"
              value={Number(report.concept_drift.response_embedding_distance ?? 0)}
              warn={0.1}
              critical={0.2}
            />
            <MetricRow
              label="Toxicity PSI"
              value={Number(report.concept_drift.toxicity_psi ?? 0)}
              warn={warn}
              critical={critical}
            />
            <MetricRow
              label="Response length PSI"
              value={Number(report.concept_drift.response_length_psi ?? 0)}
              warn={warn}
              critical={critical}
            />
          </div>

          <div className="space-y-3">
            <h3 className="flex items-center gap-2 text-sm font-semibold uppercase tracking-wide text-[var(--muted)]">
              <Star size={14} />
              Target metrics
            </h3>
            <MetricRow
              label="User rating PSI"
              value={Number(report.target_drift.user_rating_psi ?? 0)}
              warn={warn}
              critical={critical}
            />
            <MetricRow
              label="Toxicity tier PSI"
              value={Number(report.target_drift.toxicity_tier_psi ?? 0)}
              warn={warn}
              critical={critical}
            />
            <MetricRow
              label="Response language PSI"
              value={Number(report.target_drift.response_language_psi ?? 0)}
              warn={warn}
              critical={critical}
            />
          </div>
        </section>

        <section>
          <h3 className="mb-4 flex items-center gap-2 text-lg font-semibold">
            <Languages size={18} />
            Распределения: baseline vs current
          </h3>
          <div className="grid gap-4 xl:grid-cols-2">
            <DistributionCompare
              title="Prompt languages"
              baseline={promptLang.baseline}
              current={promptLang.current}
            />
            <DistributionCompare
              title="Response languages"
              baseline={responseLang.baseline}
              current={responseLang.current}
            />
            <DistributionCompare
              title="User ratings"
              baseline={userRatings.baseline}
              current={userRatings.current}
            />
            <DistributionCompare
              title="Toxicity tiers"
              baseline={toxicity.baseline}
              current={toxicity.current}
            />
            <DistributionCompare
              title="Prompt lengths"
              baseline={promptLengths.baseline}
              current={promptLengths.current}
            />
            <DistributionCompare
              title="Response lengths"
              baseline={responseLengths.baseline}
              current={responseLengths.current}
            />
          </div>
        </section>
      </div>
    </article>
  );
}
