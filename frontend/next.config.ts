import type { NextConfig } from "next";

/**
 * The Python backend is proxied under this app's own origin at /api/*, the
 * same pattern semantic-va's web app uses, so the backend needs no CORS.
 *
 * Ports are chosen so mini-reid runs beside semantic-va without clashing:
 * semantic-va uses 3000 (web) and 8080 (API); mini-reid uses 3001 (web) and
 * 8000 (API, as in docker-compose.yml and .env.example).
 */
const API = process.env.MINI_REID_API ?? "http://127.0.0.1:8000";

const nextConfig: NextConfig = {
  cacheComponents: true,
  partialPrefetching: true,
  turbopack: {
    rules: {
      "*.css": {
        loaders: ["@tailwindcss/turbopack"],
        as: "*.css",
      },
    },
  },
  async rewrites() {
    return { fallback: [{ source: "/api/:path*", destination: `${API}/:path*` }] };
  },
};

export default nextConfig;
