import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// base "./" damit der Build auch ueber file:// (Electron) laedt
export default defineConfig({
  plugins: [react()],
  base: "./",
  server: { port: 5173, strictPort: true },
});
