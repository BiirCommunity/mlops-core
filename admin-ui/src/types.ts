export type ServiceHealth = {
  ok: boolean;
  message: string;
};

export type TrainingHealth = {
  status: "ok" | "degraded";
  minio: ServiceHealth;
  mlflow: ServiceHealth;
  mlflow_tracking_uri: string;
  mlflow_ui: string;
  minio_console: string;
  admin_ui: string;
  token_required: boolean;
};

export type TrainingJob = {
  id: string;
  status: "queued" | "running" | "completed" | "failed" | "cancelled";
  config: {
    dataset_path: string;
    run_name: string;
    job_id?: string | null;
    epochs: number;
    batch_size: number;
    learning_rate: number;
    max_seq_len: number;
    lora_rank: number;
    lora_alpha: number;
    gradient_accumulation_steps: number;
    seed: number;
  };
  created_at: number;
  started_at?: number | null;
  finished_at?: number | null;
  error?: string | null;
  cancel_requested?: boolean;
  progress: Record<string, number | string | unknown[] | Record<string, unknown> | null>;
  result: Record<string, unknown>;
};

export type TrainingCheckpoint = {
  epoch: number;
  adapter_path: string;
  model_path: string;
  minio_adapter_uri?: string;
  minio_model_uri?: string;
};

export type ModelVersion = {
  name: string;
  version: string;
  stage: string;
  run_id: string;
  status: string;
  description: string;
  tags: Record<string, string>;
};

export type StartTrainingPayload = {
  dataset_path?: string | null;
  run_name: string;
  epochs: number;
  batch_size: number;
  learning_rate: number;
  max_seq_len: number;
  lora_rank: number;
  lora_alpha: number;
  gradient_accumulation_steps: number;
  seed: number;
};

export type DatasetListItem = {
  object_name: string;
  filename: string;
  uri: string;
  size_bytes: number;
  last_modified: string | null;
};

export type DatasetListResponse = {
  count: number;
  datasets: DatasetListItem[];
  default_dataset: {
    label: string;
    local_path: string | null;
  };
};

export type UploadDatasetResponse = {
  local_path: string;
  minio_uri: string;
  summary: {
    path: string;
    examples: number;
    roles: Record<string, number>;
  };
};

export type InferenceModelStatus = {
  registry_name: string;
  checkpoint_path: string;
  model_revision: string;
  loaded_version: string | null;
  deployed_version: string | null;
  pending_reload: boolean;
  deployed_at?: number | null;
  object_name?: string | null;
  checkpoint_on_disk?: { size_bytes: number; mtime: number } | null;
  checkpoint_loaded_at_startup?: { size_bytes: number; mtime: number } | null;
  status: "loaded" | "reload_required" | "unknown" | "missing_checkpoint";
};

export type DvcModelStatus = {
  checkpoint_exists: boolean;
  remote_uri: string;
  sidecar_path?: string;
  sidecar?: {
    sidecar_path: string;
    md5: string;
    size: number;
    path: string;
  } | null;
  disk_md5: string | null;
  disk_size: number | null;
  tracked_md5: string | null;
  remote_present: boolean | null;
  remote_object?: string | null;
  in_sync: boolean;
  status:
    | "in_sync"
    | "out_of_sync"
    | "remote_missing"
    | "untracked"
    | "missing_checkpoint"
    | "unknown";
};

export type ModelPipelineStatus = {
  pipeline_status:
    | "ready"
    | "restart_required"
    | "dvc_sync_required"
    | "missing_checkpoint"
    | "attention";
  inference: InferenceModelStatus;
  dvc: DvcModelStatus;
  actions: {
    restart_app: boolean;
    sync_dvc: boolean;
  };
};

export type DeployModelResponse = {
  version: string;
  local_path: string;
  object_name: string;
  pending_reload?: boolean;
  inference_status?: string;
  dvc_sync?: {
    status: string;
    md5?: string;
    remote_uri?: string;
    message?: string;
  };
  pipeline?: ModelPipelineStatus;
};

export type DeployableCheckpoint = {
  job_id: string;
  epoch: number;
  run_name: string;
  job_status: string;
  model_path: string;
  adapter_path: string;
  minio_model_uri?: string | null;
  minio_adapter_uri?: string | null;
  available_locally: boolean;
  registered_version?: string | null;
};

export type RegisterCheckpointResponse = {
  status: "registered" | "already_registered";
  job_id: string;
  epoch: number;
  model_version?: string;
  registry_name?: string;
  minio_model_uri?: string;
  minio_adapter_uri?: string | null;
};

export type InteractionRecord = {
  id: string;
  created_at: number;
  prompt: string;
  response: string;
  prompt_lang: string;
  response_lang: string;
  toxicity: number;
  json_valid: boolean;
  status: string;
  anomaly_flags: string[];
  session_id?: string | null;
  conversation_id?: string | null;
  user_rating?: number | null;
  prompt_tokens: number;
  completion_tokens: number;
  model: string;
};

export type DriftAlert = {
  source: string;
  kind?: string;
  severity: string;
  score?: number;
  message?: string;
  report_id?: string;
  generated_at?: string;
  summary?: string;
  scores?: Record<string, number | undefined>;
};

export type DriftReport = {
  report_id: string;
  generated_at: string;
  status: string;
  severity: string;
  summary: string;
  windows: Record<string, number | boolean>;
  data_drift: Record<string, number>;
  concept_drift: Record<string, number>;
  target_drift: Record<string, number | boolean>;
  distributions: Record<string, unknown>;
  thresholds: Record<string, number>;
};

export type AuthUser = {
  id: number;
  username: string;
  active: boolean;
  created_at: number;
};

export type AuthApiKey = {
  id: number;
  user_id: number;
  username: string;
  name: string;
  scopes: string[];
  active: boolean;
  created_at: number;
};
