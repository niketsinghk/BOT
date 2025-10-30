// /api/health.js — Vercel serverless function (Node.js runtime)
export const config = { runtime: "nodejs" };

const BOT_NAME = process.env.BOT_NAME || "Duki";

export default async function handler(req, res) {
  // ---- CORS + preflight ----
  const origin = req.headers.origin || "*";
  res.setHeader("Access-Control-Allow-Origin", origin);
  res.setHeader("Access-Control-Allow-Credentials", "true");
  res.setHeader("Access-Control-Allow-Methods", "GET, OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type, X-Session-ID");
  res.setHeader("Cache-Control", "no-store");

  if (req.method === "OPTIONS") {
    return res.status(204).end();
  }
  if (req.method !== "GET") {
    res.setHeader("Allow", ["GET", "OPTIONS"]);
    return res.status(405).json({ ok: false, error: "Method Not Allowed" });
  }

  // ---- Healthy response ----
  return res.status(200).json({
    ok: true,
    service: "duki",
    bot: BOT_NAME,
    ts: Date.now(),
    uptime: process.uptime(),
    env: process.env.VERCEL_ENV || "production",
  });
}
