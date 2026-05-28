import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { AlertTriangle, MessageSquareWarning } from "lucide-react";
import { trainingApi } from "@/api/client";
import { formatTimestamp } from "@/lib/format";

const flagLabels: Record<string, string> = {
  high_toxicity: "High toxicity",
  toxicity_critical: "Critical toxicity",
  language_mismatch: "Language mismatch",
  invalid_json: "Invalid JSON",
  low_rating: "Low rating",
  empty_response: "Empty response",
  error_response: "Error response",
  very_long_response: "Very long response",
  very_short_prompt: "Very short prompt",
};

export function ConversationsPage() {
  const [anomaliesOnly, setAnomaliesOnly] = useState(false);
  const query = useQuery({
    queryKey: ["interactions", anomaliesOnly],
    queryFn: () =>
      trainingApi.listInteractions({
        limit: 200,
        anomalies_only: anomaliesOnly,
      }),
    refetchInterval: 10000,
  });

  return (
    <div className="space-y-6">
      <div className="flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
        <div>
          <h1 className="text-3xl font-semibold">История Q&A</h1>
          <p className="mt-2 text-[var(--muted)]">
            Полная история запросов и ответов с флагами аномалий.
          </p>
        </div>
        <label className="flex items-center gap-2 text-sm text-[var(--muted)]">
          <input
            type="checkbox"
            checked={anomaliesOnly}
            onChange={(e) => setAnomaliesOnly(e.target.checked)}
          />
          Только аномалии
        </label>
      </div>

      <div className="grid gap-4 md:grid-cols-3">
        <div className="card p-4">
          <div className="text-sm text-[var(--muted)]">Всего</div>
          <div className="mt-2 text-2xl font-semibold">
            {(query.data?.stats.total as number | undefined) ?? "—"}
          </div>
        </div>
        <div className="card p-4">
          <div className="text-sm text-[var(--muted)]">С аномалиями</div>
          <div className="mt-2 text-2xl font-semibold text-[#fca5a5]">
            {(query.data?.stats.anomalous as number | undefined) ?? "—"}
          </div>
        </div>
        <div className="card p-4">
          <div className="text-sm text-[var(--muted)]">Показано</div>
          <div className="mt-2 text-2xl font-semibold">{query.data?.count ?? "—"}</div>
        </div>
      </div>

      <div className="space-y-4">
        {(query.data?.interactions ?? []).map((item) => (
          <article key={item.id} className="card p-5">
            <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
              <div>
                <div className="mono text-sm text-[#93c5fd]">{item.id}</div>
                <div className="text-sm text-[var(--muted)]">
                  {formatTimestamp(item.created_at)} · {item.conversation_id ?? item.session_id ?? "—"}
                </div>
              </div>
              <div className="flex flex-wrap gap-2">
                {item.anomaly_flags.length ? (
                  item.anomaly_flags.map((flag) => (
                    <span key={flag} className="badge badge-failed">
                      <AlertTriangle size={12} />
                      {flagLabels[flag] ?? flag}
                    </span>
                  ))
                ) : (
                  <span className="badge badge-completed">ok</span>
                )}
              </div>
            </div>
            <div className="grid gap-4 lg:grid-cols-2">
              <div>
                <div className="mb-2 text-xs uppercase tracking-wide text-[var(--muted)]">
                  Prompt ({item.prompt_lang})
                </div>
                <div className="rounded-xl bg-[#0b1015] p-4 text-sm">{item.prompt}</div>
              </div>
              <div>
                <div className="mb-2 text-xs uppercase tracking-wide text-[var(--muted)]">
                  Response ({item.response_lang}) · toxicity {item.toxicity.toFixed(3)}
                </div>
                <div className="rounded-xl bg-[#0b1015] p-4 text-sm">{item.response}</div>
              </div>
            </div>
          </article>
        ))}
        {!query.data?.interactions.length ? (
          <div className="card flex items-center justify-center gap-3 p-10 text-[var(--muted)]">
            <MessageSquareWarning size={18} />
            История пока пуста — отправьте запросы в chat API.
          </div>
        ) : null}
      </div>
    </div>
  );
}
