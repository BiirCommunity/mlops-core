import { FormEvent, useState } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import { useNavigate } from "react-router-dom";
import { LoaderCircle, RefreshCw, Rocket } from "lucide-react";
import { ApiError, trainingApi } from "@/api/client";
import { useAuth } from "@/auth/AuthContext";
import type { StartTrainingPayload } from "@/types";

const defaultForm: StartTrainingPayload = {
  dataset_path: "",
  run_name: "lora-posttrain",
  epochs: 3,
  batch_size: 2,
  learning_rate: 0.0002,
  max_seq_len: 512,
  lora_rank: 8,
  lora_alpha: 16,
  gradient_accumulation_steps: 1,
  seed: 42,
};

function formatBytes(size: number): string {
  if (size < 1024) return `${size} B`;
  if (size < 1024 * 1024) return `${(size / 1024).toFixed(1)} KB`;
  return `${(size / (1024 * 1024)).toFixed(1)} MB`;
}

export function NewJobPage() {
  const navigate = useNavigate();
  const { token, tokenRequired } = useAuth();
  const [form, setForm] = useState(defaultForm);
  const [error, setError] = useState<string | null>(null);

  const datasetsQuery = useQuery({
    queryKey: ["datasets"],
    queryFn: trainingApi.listDatasets,
    enabled: !tokenRequired || Boolean(token),
    staleTime: 0,
    retry: (failureCount, queryError) => {
      if (
        queryError instanceof ApiError &&
        (queryError.status >= 502 || queryError.status === 503)
      ) {
        return failureCount < 6;
      }
      return failureCount < 1;
    },
    retryDelay: (attempt) => Math.min(1000 * 2 ** attempt, 8000),
  });

  const mutation = useMutation({
    mutationFn: trainingApi.startJob,
    onSuccess: (job) => navigate(`/jobs/${job.id}`),
    onError: (err: Error) => setError(err.message),
  });

  function updateField<K extends keyof StartTrainingPayload>(
    key: K,
    value: StartTrainingPayload[K],
  ) {
    setForm((current) => ({ ...current, [key]: value }));
  }

  function onSubmit(event: FormEvent) {
    event.preventDefault();
    setError(null);
    mutation.mutate({
      ...form,
      dataset_path: form.dataset_path?.trim() ? form.dataset_path : null,
    });
  }

  const defaultLabel =
    datasetsQuery.data?.default_dataset.label ?? "lora_posttrain_sample.jsonl";
  const minioDatasets = datasetsQuery.data?.datasets ?? [];

  return (
    <div className="mx-auto max-w-4xl space-y-6">
      <div>
        <h1 className="text-3xl font-semibold">Новый LoRA run</h1>
        <p className="mt-2 text-[var(--muted)]">
          Настройте post-train job в Admin Studio. Выберите JSONL из MinIO или
          example dataset на backend.
        </p>
      </div>

      <form onSubmit={onSubmit} className="card space-y-6 p-6">
        <div className="grid gap-4 md:grid-cols-2">
          <label className="block md:col-span-2">
            <span className="mb-2 block text-sm text-[var(--muted)]">
              Run name
            </span>
            <input
              className="input"
              value={form.run_name}
              onChange={(e) => updateField("run_name", e.target.value)}
              required
            />
          </label>

          <label className="block md:col-span-2">
            <span className="mb-2 block text-sm text-[var(--muted)]">
              Dataset
            </span>
            <select
              className="input mono"
              value={form.dataset_path ?? ""}
              onChange={(e) => updateField("dataset_path", e.target.value)}
              disabled={datasetsQuery.isLoading}
            >
              <option value="">
                Example — {defaultLabel}
              </option>
              {minioDatasets.map((dataset) => (
                <option key={dataset.uri} value={dataset.uri}>
                  {dataset.filename} ({formatBytes(dataset.size_bytes)})
                </option>
              ))}
            </select>
            {datasetsQuery.isFetching && !datasetsQuery.isLoading ? (
              <p className="mt-2 text-xs text-[var(--muted)]">
                Обновление списка датасетов…
              </p>
            ) : null}
            {datasetsQuery.isError ? (
              <div className="mt-2 flex flex-wrap items-center gap-2 text-xs text-amber-300">
                <span>
                  {datasetsQuery.error instanceof Error
                    ? datasetsQuery.error.message
                    : "Не удалось загрузить список датасетов"}
                  . Доступен example dataset.
                </span>
                <button
                  type="button"
                  className="btn btn-secondary px-2 py-1 text-xs"
                  onClick={() => datasetsQuery.refetch()}
                >
                  <RefreshCw size={14} />
                  Повторить
                </button>
              </div>
            ) : null}
            {!datasetsQuery.isLoading &&
            !datasetsQuery.isError &&
            minioDatasets.length === 0 ? (
              <p className="mt-2 text-xs text-[var(--muted)]">
                В MinIO пока нет JSONL. Загрузите файл на странице Datasets.
              </p>
            ) : null}
          </label>

          <label className="block">
            <span className="mb-2 block text-sm text-[var(--muted)]">
              Epochs
            </span>
            <input
              className="input"
              type="number"
              min={1}
              max={100}
              value={form.epochs}
              onChange={(e) => updateField("epochs", Number(e.target.value))}
            />
          </label>

          <label className="block">
            <span className="mb-2 block text-sm text-[var(--muted)]">
              Batch size
            </span>
            <input
              className="input"
              type="number"
              min={1}
              max={128}
              value={form.batch_size}
              onChange={(e) => updateField("batch_size", Number(e.target.value))}
            />
          </label>

          <label className="block">
            <span className="mb-2 block text-sm text-[var(--muted)]">
              Learning rate
            </span>
            <input
              className="input"
              type="number"
              step="0.00001"
              value={form.learning_rate}
              onChange={(e) =>
                updateField("learning_rate", Number(e.target.value))
              }
            />
          </label>

          <label className="block">
            <span className="mb-2 block text-sm text-[var(--muted)]">
              Max seq len
            </span>
            <input
              className="input"
              type="number"
              min={64}
              max={4096}
              value={form.max_seq_len}
              onChange={(e) =>
                updateField("max_seq_len", Number(e.target.value))
              }
            />
          </label>

          <label className="block">
            <span className="mb-2 block text-sm text-[var(--muted)]">
              LoRA rank
            </span>
            <input
              className="input"
              type="number"
              min={1}
              max={128}
              value={form.lora_rank}
              onChange={(e) => updateField("lora_rank", Number(e.target.value))}
            />
          </label>

          <label className="block">
            <span className="mb-2 block text-sm text-[var(--muted)]">
              LoRA alpha
            </span>
            <input
              className="input"
              type="number"
              min={1}
              max={256}
              value={form.lora_alpha}
              onChange={(e) =>
                updateField("lora_alpha", Number(e.target.value))
              }
            />
          </label>
        </div>

        {error ? (
          <div className="rounded-xl border border-red-500/30 bg-red-500/10 px-4 py-3 text-sm text-red-200">
            {error}
          </div>
        ) : null}

        <div className="flex gap-3">
          <button type="submit" className="btn btn-primary" disabled={mutation.isPending}>
            {mutation.isPending ? (
              <LoaderCircle className="animate-spin" size={18} />
            ) : (
              <Rocket size={18} />
            )}
            Запустить дообучение
          </button>
        </div>
      </form>
    </div>
  );
}
