import path from "path"
import tailwindcss from "@tailwindcss/vite"
import react from "@vitejs/plugin-react"
import { defineConfig } from "vite"

// The production build lands directly in the FastAPI package so that
// `pdfplay serve` works straight after `pip install`, with no node step.
export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: { alias: { "@": path.resolve(__dirname, "./src") } },
  build: {
    outDir: path.resolve(__dirname, "../src/pdfplay/server/static"),
    emptyOutDir: true,
  },
  server: {
    // `npm run dev` proxies API calls to a `pdfplay serve` on :8000.
    proxy: { "/api": "http://127.0.0.1:8000" },
  },
})
