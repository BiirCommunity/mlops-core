import type {
  AuthApiKey,
  AuthUser,
  DatasetListResponse,
  DeployableCheckpoint,
  DeployModelResponse,
  DriftAlert,
  DriftReport,
  ModelPipelineStatus,
  InteractionRecord,
  ModelVersion,
  RegisterCheckpointResponse,
  StartTrainingPayload,
  TrainingHealth,
  TrainingJob,
  UploadDatasetResponse,
} from "@/types";
import { authHeaders, getAccessToken } from "@/auth/token";

const TRAINING_API_BASE = import.meta.env.VITE_API_BASE_URL ?? "/v1/training";
const AUTH_API_BASE = "/v1";

class ApiError extends Error {
  status: number;

  constructor(message: string, status: number) {
    super(message);
    this.status = status;
  }
}

async function requestAt<T>(
  base: string,
  path: string,
  init?: RequestInit,
): Promise<T> {
  const response = await fetch(`${base}${path}`, {
    ...init,
    headers: {
      ...authHeaders(),
      ...(init?.headers ?? {}),
    },
  });
  const payload = await response.json().catch(() => ({}));
  if (response.status === 401) {
    const detail = typeof payload.detail === "string" ? payload.detail : null;
    if (!getAccessToken()) {
      throw new ApiError("Требуется вход (username / password)", 401);
    }
    throw new ApiError(
      detail ?? "Сессия недействительна — выйдите и войдите снова",
      401,
    );
  }
  if (!response.ok) {
    if (response.status >= 502) {
      throw new ApiError(
        "Backend недоступен. Дождитесь запуска app и нажмите «Повторить».",
        response.status,
      );
    }
    const detail =
      typeof payload.detail === "string"
        ? payload.detail
        : response.statusText || "Request failed";
    throw new ApiError(detail, response.status);
  }
  return payload as T;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  return requestAt(TRAINING_API_BASE, path, init);
}

async function authRequest<T>(path: string, init?: RequestInit): Promise<T> {
  return requestAt(AUTH_API_BASE, path, init);
}

export const trainingApi = {
  authStatus: () =>
    request<{ token_required: boolean }>("/auth/status", {
      headers: {},
    }),

  verifyToken: (token: string) =>
    fetch(`${AUTH_API_BASE}/auth/verify`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${token}`,
      },
      body: JSON.stringify({ scope: "admin" }),
    }).then(async (response) => {
      if (!response.ok) {
        throw new ApiError("Invalid token", response.status);
      }
    }),

  login: (username: string, password: string) =>
    fetch(`${AUTH_API_BASE}/auth/login`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ username, password }),
    }).then(async (response) => {
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) {
        const detail =
          typeof payload.detail === "string"
            ? payload.detail
            : "Invalid credentials";
        throw new ApiError(detail, response.status);
      }
      return payload as { token: string };
    }),

  getHealth: () => request<TrainingHealth>("/health"),

  getOverview: () => request<Record<string, unknown>>("/admin/overview"),

  listInteractions: (params?: {
    limit?: number;
    anomalies_only?: boolean;
    conversation_id?: string;
  }) => {
    const query = new URLSearchParams();
    if (params?.limit) query.set("limit", String(params.limit));
    if (params?.anomalies_only) query.set("anomalies_only", "true");
    if (params?.conversation_id) {
      query.set("conversation_id", params.conversation_id);
    }
    const suffix = query.toString() ? `?${query}` : "";
    return request<{ count: number; interactions: InteractionRecord[]; stats: Record<string, unknown> }>(
      `/admin/interactions${suffix}`,
    );
  },

  getDriftAlerts: (limit = 20) =>
    request<{ count: number; alerts: DriftAlert[]; live_snapshot: Record<string, unknown> | null }>(
      `/admin/drift/alerts?limit=${limit}`,
    ),

  listDriftReports: (limit = 50) =>
    request<{ count: number; reports: DriftReport[] }>(
      `/admin/drift/reports?limit=${limit}`,
    ),

  getLatestDriftReport: () =>
    request<DriftReport>("/admin/drift/reports/latest"),

  getDriftReport: (reportId: string) =>
    request<DriftReport>(`/admin/drift/reports/${reportId}`),

  listExperiments: (limit = 50) =>
    request<{ count: number; experiments: Record<string, string>[] }>(
      `/admin/experiments?limit=${limit}`,
    ),

  listExperimentRuns: (experimentName?: string, limit = 50) => {
    const query = new URLSearchParams({ limit: String(limit) });
    if (experimentName) query.set("experiment_name", experimentName);
    return request<{ count: number; runs: Record<string, unknown>[]; experiment_name: string }>(
      `/admin/experiments/runs?${query}`,
    );
  },

  listJobs: (limit = 50) =>
    request<{ count: number; jobs: TrainingJob[] }>(`/jobs?limit=${limit}`),

  getJob: (jobId: string) => request<TrainingJob>(`/jobs/${jobId}`),

  cancelJob: (jobId: string) =>
    request<TrainingJob>(`/jobs/${jobId}/cancel`, {
      method: "POST",
    }),

  startJob: (payload: StartTrainingPayload) =>
    request<TrainingJob>("/jobs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }),

  listDatasets: () => request<DatasetListResponse>("/datasets"),

  uploadDataset: (file: File) => {
    const formData = new FormData();
    formData.append("file", file);
    return request<UploadDatasetResponse>("/datasets/upload", {
      method: "POST",
      body: formData,
    });
  },

  downloadExampleDataset: async () => {
    const response = await fetch(`${TRAINING_API_BASE}/datasets/example`, {
      headers: authHeaders(),
    });
    if (response.status === 401) {
      throw new ApiError("Требуется access token", 401);
    }
    if (!response.ok) {
      const payload = await response.json().catch(() => ({}));
      const detail =
        typeof payload.detail === "string"
          ? payload.detail
          : response.statusText || "Download failed";
      throw new ApiError(detail, response.status);
    }
    const blob = await response.blob();
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = "lora_posttrain_sample.jsonl";
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(url);
  },

  listModels: (limit = 20) =>
    request<{ count: number; versions: ModelVersion[] }>(
      `/models?limit=${limit}`,
    ),

  getActiveModel: () => request<ModelPipelineStatus>("/models/active"),

  getModelStatus: () => request<ModelPipelineStatus>("/models/status"),

  syncDvc: () =>
    request<ModelPipelineStatus>("/models/dvc/sync", { method: "POST" }),

  listTrainingCheckpoints: () =>
    request<{ count: number; checkpoints: DeployableCheckpoint[] }>(
      "/models/checkpoints",
    ),

  registerTrainingCheckpoint: (jobId: string, epoch: number) =>
    request<RegisterCheckpointResponse>("/models/checkpoints/register", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ job_id: jobId, epoch }),
    }),

  deployModel: (version: string) =>
    request<DeployModelResponse>("/models/deploy", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ version }),
    }),
};

export const authServiceApi = {
  listUsers: () => authRequest<{ count: number; users: AuthUser[] }>("/users"),

  createUser: (payload: {
    username: string;
    password: string;
    active?: boolean;
  }) =>
    authRequest<AuthUser>("/users", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }),

  updateUser: (
    userId: number,
    payload: { username?: string; password?: string; active?: boolean },
  ) =>
    authRequest<AuthUser>(`/users/${userId}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }),

  deleteUser: (userId: number) =>
    authRequest<{ status: string; id: string }>(`/users/${userId}`, {
      method: "DELETE",
    }),

  listApiKeys: () =>
    authRequest<{ count: number; api_keys: AuthApiKey[] }>("/api-keys"),

  createApiKey: (payload: {
    user_id: number;
    name: string;
    scopes: string[];
  }) =>
    authRequest<{ token: string; api_key: AuthApiKey }>("/api-keys", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }),
};

export { ApiError };
