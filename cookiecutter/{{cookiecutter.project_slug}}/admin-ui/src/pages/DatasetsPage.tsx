import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { Download, LoaderCircle, Upload } from "lucide-react";
import { trainingApi } from "@/api/client";

export function DatasetsPage() {
  const queryClient = useQueryClient();
  const [file, setFile] = useState<File | null>(null);
  const [uploadResult, setUploadResult] = useState<string>("");
  const [error, setError] = useState<string | null>(null);
  const [downloading, setDownloading] = useState(false);

  const uploadMutation = useMutation({
    mutationFn: trainingApi.uploadDataset,
    onSuccess: (result) => {
      setError(null);
      setUploadResult(JSON.stringify(result, null, 2));
      queryClient.invalidateQueries({ queryKey: ["datasets"] });
    },
    onError: (err: Error) => {
      setUploadResult("");
      setError(err.message);
    },
  });

  async function handleDownloadExample() {
    setDownloading(true);
    setError(null);
    try {
      await trainingApi.downloadExampleDataset();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Не удалось скачать example dataset");
    } finally {
      setDownloading(false);
    }
  }

  return (
    <div className="mx-auto max-w-4xl space-y-6">
      <div>
        <h1 className="text-3xl font-semibold">Datasets</h1>
        <p className="mt-2 text-[var(--muted)]">
          JSONL с полем <code>messages</code>. Последнее сообщение должно быть от
          assistant.
        </p>
      </div>

      <div className="card space-y-5 p-6">
        <div>
          <div className="mb-2 text-sm text-[var(--muted)]">Example dataset</div>
          <button
            type="button"
            className="btn btn-secondary"
            disabled={downloading}
            onClick={handleDownloadExample}
          >
            {downloading ? (
              <LoaderCircle className="animate-spin" size={18} />
            ) : (
              <Download size={18} />
            )}
            Скачать lora_posttrain_sample.jsonl
          </button>
        </div>

        <div>
          <div className="mb-2 text-sm text-[var(--muted)]">Upload JSONL</div>
          <input
            className="input"
            type="file"
            accept=".jsonl,application/jsonl"
            onChange={(event) => setFile(event.target.files?.[0] ?? null)}
          />
        </div>

        {error ? (
          <div className="rounded-xl border border-red-500/30 bg-red-500/10 px-4 py-3 text-sm text-red-200">
            {error}
          </div>
        ) : null}

        <button
          className="btn btn-primary"
          disabled={!file || uploadMutation.isPending}
          onClick={() => file && uploadMutation.mutate(file)}
        >
          {uploadMutation.isPending ? (
            <LoaderCircle className="animate-spin" size={18} />
          ) : (
            <Upload size={18} />
          )}
          Upload в MinIO
        </button>

        {uploadResult ? (
          <pre className="overflow-auto rounded-xl bg-[#0b1015] p-4 text-xs text-[#cbd5e1]">
            {uploadResult}
          </pre>
        ) : null}
      </div>

      <div className="card p-6">
        <h2 className="mb-3 text-lg font-semibold">Format</h2>
        <pre className="overflow-auto rounded-xl bg-[#0b1015] p-4 text-xs text-[#cbd5e1]">
{`{"messages": [
  {"role": "user", "content": "What about you?"},
  {"role": "assistant", "content": "I'm doing well, thanks for asking."}
]}`}
        </pre>
      </div>
    </div>
  );
}
