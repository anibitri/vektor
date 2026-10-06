import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

// With `npm run dev`, API calls go to vektor-server on port 8080. In production
// the server itself serves the built files (vektor-server --ui ui/dist).
const server = "http://localhost:8080";

export default defineConfig({
  plugins: [react()],
  server: { proxy: { "/stats": server, "/rag": server, "/health": server } },
  test: { environment: "jsdom" },
});
