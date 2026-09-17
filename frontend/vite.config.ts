// frontend/vite.config.ts
//
// The dev server proxies /api to the backend so the browser sees a single origin during
// development. That is not just convenience: it means the app runs against a same-origin API
// in dev exactly as it does in production (where FastAPI serves this build from its own
// origin), so cookie and CORS behaviour cannot differ between the two.
//
// `changeOrigin` is off because the target is localhost and the backend does not care about
// the Host header. `/api/runs/*/events` is a server-sent-events stream, so buffering must not
// be introduced here - Vite's proxy streams by default, and the backend already sends
// `X-Accel-Buffering: no` for whatever sits in front of it in production.
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const BACKEND = process.env.VITE_API_TARGET || "http://127.0.0.1:8000";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": { target: BACKEND, changeOrigin: false },
    },
  },
  build: {
    // FastAPI mounts /assets and falls back to index.html for everything else, which is what
    // these defaults produce. Changing assetsDir would require changing backend/app.py too.
    outDir: "dist",
    assetsDir: "assets",
    sourcemap: true,
  },
});
