import { useMemo, useState } from "react";
import { sendChat, sendFeedback, type ChatMessage } from "@/api/client";
import { useAuth } from "@/auth/AuthContext";
import { createSessionId } from "@/lib/session";

type UiMessage = ChatMessage & {
  completionId?: string;
  userRating?: number;
  ratingPending?: boolean;
};

export function ChatPage() {
  const { logout } = useAuth();
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState<UiMessage[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const sessionId = useMemo(() => createSessionId(), []);

  async function onSend(event: React.FormEvent) {
    event.preventDefault();
    const text = input.trim();
    if (!text || loading) return;

    const nextMessages: UiMessage[] = [...messages, { role: "user", content: text }];
    setMessages(nextMessages);
    setInput("");
    setLoading(true);
    setError(null);

    try {
      const response = await sendChat({
        messages: nextMessages,
        sessionId,
        maxTokens: 256,
      });
      const assistant = response.choices[0]?.message?.content?.trim() || "…";
      setMessages([
        ...nextMessages,
        { role: "assistant", content: assistant, completionId: response.id },
      ]);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Inference failed");
    } finally {
      setLoading(false);
    }
  }

  async function rate(completionId: string, rating: number) {
    const alreadyRated = messages.some(
      (message) => message.completionId === completionId && message.userRating != null,
    );
    if (alreadyRated) return;

    setMessages((prev) =>
      prev.map((message) =>
        message.completionId === completionId
          ? { ...message, ratingPending: true }
          : message,
      ),
    );
    setError(null);

    try {
      await sendFeedback(completionId, rating);
      setMessages((prev) =>
        prev.map((message) =>
          message.completionId === completionId
            ? { ...message, userRating: rating, ratingPending: false }
            : message,
        ),
      );
    } catch {
      setMessages((prev) =>
        prev.map((message) =>
          message.completionId === completionId
            ? { ...message, ratingPending: false }
            : message,
        ),
      );
      setError("Не удалось отправить оценку");
    }
  }

  function ratingLabel(value: number): string {
    if (value <= 2) return "плохо";
    if (value === 3) return "нейтрально";
    if (value === 4) return "хорошо";
    return "отлично";
  }

  return (
    <div className="mx-auto flex min-h-screen max-w-4xl flex-col p-4">
      <header className="mb-4 flex items-center justify-between gap-3">
        <div>
          <h1 className="text-2xl font-semibold">MLOps Chat</h1>
          <p className="text-sm text-[var(--muted)]">session: {sessionId}</p>
        </div>
        <button className="btn btn-secondary" onClick={logout} type="button">
          Выйти
        </button>
      </header>

      <div className="card flex flex-1 flex-col gap-3 overflow-y-auto p-4 min-h-[60vh]">
        {!messages.length ? (
          <p className="text-[var(--muted)]">Напишите сообщение модели llm-lora.</p>
        ) : null}
        {messages.map((message, index) => (
          <div key={`${index}-${message.role}`} className="flex flex-col gap-2">
            <div
              className={`max-w-[85%] rounded-2xl px-4 py-3 whitespace-pre-wrap ${
                message.role === "user" ? "message-user" : "message-assistant"
              }`}
            >
              {message.content}
            </div>
            {message.role === "assistant" && message.completionId ? (
              <div className="flex flex-col gap-1">
                {message.userRating != null ? (
                  <p className="text-sm text-green-300">
                    ✓ Оценка {message.userRating}/5 ({ratingLabel(message.userRating)})
                  </p>
                ) : (
                  <>
                    <p className="text-xs text-[var(--muted)]">
                      Оцените ответ по шкале 1–5 (1 — плохо, 5 — отлично)
                    </p>
                    <div className="flex flex-wrap gap-2">
                      {[1, 2, 3, 4, 5].map((value) => (
                        <button
                          key={value}
                          type="button"
                          className="btn btn-secondary !min-w-9 !py-1 !px-3"
                          disabled={message.ratingPending}
                          onClick={() => rate(message.completionId!, value)}
                        >
                          {message.ratingPending ? "…" : value}
                        </button>
                      ))}
                    </div>
                  </>
                )}
              </div>
            ) : null}
          </div>
        ))}
        {loading ? <div className="text-sm text-[var(--muted)]">Думаю…</div> : null}
      </div>

      {error ? <div className="mt-3 text-sm text-red-300">{error}</div> : null}

      <form onSubmit={onSend} className="mt-4 flex gap-3">
        <input
          className="input flex-1"
          placeholder="Сообщение…"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          disabled={loading}
        />
        <button className="btn btn-primary" disabled={loading || !input.trim()} type="submit">
          Отправить
        </button>
      </form>
    </div>
  );
}
