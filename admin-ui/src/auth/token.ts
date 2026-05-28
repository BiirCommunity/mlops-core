const TOKEN_STORAGE_KEY = "mlops_access_token";

export function getAccessToken(): string | null {
  return sessionStorage.getItem(TOKEN_STORAGE_KEY);
}

export function setAccessToken(token: string): void {
  sessionStorage.setItem(TOKEN_STORAGE_KEY, token);
}

export function clearAccessToken(): void {
  sessionStorage.removeItem(TOKEN_STORAGE_KEY);
}

export function authHeaders(): HeadersInit {
  const token = getAccessToken();
  if (!token) return {};
  return {
    Authorization: `Bearer ${token}`,
    "X-Access-Token": token,
  };
}
