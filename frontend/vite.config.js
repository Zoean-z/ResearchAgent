import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
export default defineConfig({
    base: "./",
    plugins: [react()],
    build: {
        rollupOptions: {
            input: {
                app: decodeURI(new URL("./index.html", import.meta.url).pathname),
                demo: decodeURI(new URL("./demo.html", import.meta.url).pathname),
            },
        },
    },
    server: {
        host: "127.0.0.1",
        port: 5173,
        proxy: {
            "/api": "http://127.0.0.1:8000",
            "/health": "http://127.0.0.1:8000",
        },
    },
});
