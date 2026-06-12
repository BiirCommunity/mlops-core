import { authHeaders } from "@/auth/token";

export type ChatMessage = {
  role: "user" | "assistant";
  content: string;
};

export type ChatCompletionResponse = {
  id: string;
  choices: Array<{
    message: { role: string; content: string };
  }>;
  usage?: {
    prompt_tokens?: number;
    completion_tokens?: number;
  };
};

export async function login(username: string, password: string) {
  const response = await fetch("/v1/auth/login", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ username, password }),
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(typeof payload.detail === "string" ? payload.detail : "Login failed");
  }
  return payload as { token: string; username?: string; scopes?: string[] };
}

export async function sendChat(params: {
  messages: ChatMessage[];
  sessionId: string;
  maxTokens?: number;
}) {
  const response = await fetch("/v1/chat/completions", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      ...authHeaders(),
    },
    body: JSON.stringify({
      model: "local",
      messages: params.messages,
      session_id: params.sessionId,
      max_tokens: params.maxTokens ?? 256,
      temperature: 0.8,
    }),
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    if (response.status === 502 || response.status === 503) {
      throw new Error(
        "Inference недоступен — модель ещё загружается. Подождите 1–2 мин и повторите.",
      );
    }
    const detail =
      typeof payload.detail === "string"
        ? payload.detail
        : `Inference error ${response.status}`;
    throw new Error(detail);
  }
  return payload as ChatCompletionResponse;
}

export async function sendFeedback(completionId: string, rating: number) {
  const response = await fetch("/v1/feedback", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      ...authHeaders(),
    },
    body: JSON.stringify({ completion_id: completionId, rating }),
  });
  if (!response.ok) {
    throw new Error("Failed to send feedback");
  }
}
