import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  clearScreen: false,
  server: {
    port: 5199,
    strictPort: false,
  },
  build: {
    target: "es2022",
    rollupOptions: {
      output: {
        /* 界面库占产物八成体积，与运行时、自有代码分开发，
           改表达式内核或调画法不必让用户重下一份 antd */
        manualChunks(id) {
          if (!id.includes("node_modules")) return;
          const ui = /[/\\](antd|@ant-design|rc-[^/\\]+|@rc-component)[/\\]/.test(id);
          return ui ? "ui" : "runtime";
        },
      },
    },
  },
});
