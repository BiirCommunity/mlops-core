import { useEffect, useState } from "react";

export default function App() {
  const [health, setHealth] = useState<string>("…");

  useEffect(() => {
    fetch("/api/health")
      .then((r) => r.json())
      .then((d) => setHealth(d.status ?? JSON.stringify(d)))
      .catch(() => setHealth("offline"));
  }, []);

  return (
    <main className="page">
      <h1>{{ cookiecutter.project_name }}</h1>
      <p>Chat UI</p>
      <p>API: {health}</p>
    </main>
  );
}
