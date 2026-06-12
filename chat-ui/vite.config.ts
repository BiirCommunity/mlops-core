import { fileURLToPath, URL } from "node:url";
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

export default defineConfig({
  base: "/chat/",
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: {
      "@": fileURLToPath(new URL("./src", import.meta.url)),
    },
  },
  server: {
    port: 5174,
    proxy: {
      "/v1/chat/completions": {
        target: "http://localhost:8080",
        changeOrigin: true,
      },
      "/v1/feedback": {
        target: "http://localhost:8080",
        changeOrigin: true,
      },
      "/v1/auth": {
        target: "http://localhost:8090",
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/v1\/auth/, "/auth"),
      },
    },
  },
});
