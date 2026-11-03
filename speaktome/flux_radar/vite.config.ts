import { defineConfig } from "vite";
import { resolve } from "node:path";

export default defineConfig({
  build: {
    emptyOutDir: true,
    outDir: "webgl",
    lib: {
      entry: resolve(__dirname, "src/webgl_renderer.ts"),
      name: "FluxRadarWebGL",
      formats: ["es"],
      fileName: () => "webgl_renderer.js",
    },
    sourcemap: false,
    minify: "oxc",
  },
});
