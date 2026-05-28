const TOKEN_KEY = "{{ cookiecutter.auth_token_prefix }}chat_token";

export function getChatToken(): string | null {
  return localStorage.getItem(TOKEN_KEY);
}

export function setChatToken(token: string): void {
  localStorage.setItem(TOKEN_KEY, token);
}

export function clearChatToken(): void {
  localStorage.removeItem(TOKEN_KEY);
}
