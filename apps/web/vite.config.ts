import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// MASTER_PLAN 13.7: bind to 127.0.0.1, never 0.0.0.0. This surface holds a
// kill switch; remote access is Tailscale's job, not the dev server's.
//
// The API port is read from NEUTRON_API_PORT because 8000 is a popular port
// and another project holding it makes the console proxy to the wrong
// service — which presents as a console that loads but shows nothing, rather
// than as a connection error.
const apiPort = process.env.NEUTRON_API_PORT ?? "8000";

// The dev server injects the token so it never reaches browser JavaScript.
// A token in frontend code is readable by any script on the page and ends up
// in devtools, in a screenshot, and in the bundle — the proxy is the only
// layer that can hold it and still be useful.
const apiToken = process.env.NEUTRON_API_TOKEN ?? "";

export default defineConfig({
  plugins: [react()],
  server: {
    host: "127.0.0.1",
    port: 5173,
    proxy: {
      "/api": {
        target: `http://127.0.0.1:${apiPort}`,
        rewrite: (p) => p.replace(/^\/api/, ""),
        headers: apiToken ? { Authorization: `Bearer ${apiToken}` } : undefined,
      },
    },
  },
  build: { outDir: "dist", sourcemap: true },
});
