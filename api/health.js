// /api/health.js — Vercel serverless function (Node.js runtime)
// Lightweight health & diagnostics endpoint for Duki

export const config = { runtime: "nodejs" };

const BOT_NAME = process.env.BOT_NAME || "Duki";

export default async function handler(req, res) {
  /* ---------- CORS + preflight ---------- */
  const origin = req.headers.origin || "*";

  res.setHeader("Access-Control-Allow-Origin", origin);
  res.setHeader("Access-Control-Allow-Credentials", "true");
  res.setHeader("Access-Control-Allow-Methods", "GET, OPTIONS");
  res.setHeader(
    "Access-Control-Allow-Headers",
    "Content-Type, X-Session-ID, X-SessionID, X-Client-Session"
  );
  res.setHeader("Cache-Control", "no-store");

  if (req.method === "OPTIONS") {
    return res.status(204).end();
  }

  if (req.method !== "GET") {
    res.setHeader("Allow", ["GET", "OPTIONS"]);
    return res.status(405).json({ ok: false, error: "Method Not Allowed" });
  }

  /* ---------- Health payload ---------- */
  const now = Date.now();

  return res.status(200).json({
    ok: true,
    service: "duki",
    bot: BOT_NAME,
    ts: now,
    uptime: typeof process.uptime === "function" ? process.uptime() : null,
    env: process.env.VERCEL_ENV || "production",
    // Simple diagnostic flags (do not expose secrets)
    diagnostics: {
      googleApiKeyPresent: !!process.env.GOOGLE_API_KEY,
      redisConfigured:
        !!process.env.UPSTASH_REDIS_REST_URL &&
        !!process.env.UPSTASH_REDIS_REST_TOKEN,
      embeddingModel: process.env.EMBEDDING_MODEL || "text-embedding-004",
      generationModel: process.env.GENERATION_MODEL || "gemini-2.5-flash",
    },
  });
}
