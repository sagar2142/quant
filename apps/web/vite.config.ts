import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// MASTER_PLAN 13.7: bind to 127.0.0.1, never 0.0.0.0. This surface holds a
// kill switch; remote access is Tailscale''s job, not the dev server''s.
export default defineConfig({
  plugins: [react()],
  server: {
    host: "127.0.0.1",
    port: 5173,
    proxy: { "/api": { target: "http://127.0.0.1:8000", rewrite: (p) => p.replace(/^\/api/, "") } },
  },
  build: { outDir: "dist", sourcemap: true },
});
