// /api/reset.js — Vercel serverless (Node.js runtime)
export const config = { runtime: "nodejs" };

// Keep a per-cold-start in-memory session cache (safe on serverless)
const SESSIONS =
  globalThis.__DUKEJIA_SESSIONS__ ?? (globalThis.__DUKEJIA_SESSIONS__ = new Map());

const BOT_NAME = process.env.BOT_NAME || "Duki";

export default async function handler(req, res) {
  /* ---------- CORS + preflight ---------- */
  const origin = req.headers.origin || "*";
  res.setHeader("Access-Control-Allow-Origin", origin);
  res.setHeader("Access-Control-Allow-Credentials", "true");
  res.setHeader("Access-Control-Allow-Methods", "POST, OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type, X-Session-ID");
  res.setHeader("Cache-Control", "no-store");

  if (req.method === "OPTIONS") return res.status(204).end();
  if (req.method !== "POST") {
    res.setHeader("Allow", ["POST", "OPTIONS"]);
    return res.status(405).json({ ok: false, error: "Method Not Allowed" });
  }

  /* ---------- Parse body safely ---------- */
  let body = {};
  try {
    body = typeof req.body === "string" ? JSON.parse(req.body) : (req.body || {});
  } catch {
    body = {};
  }

  /* ---------- Extract session ID ---------- */
  const cookie = req.headers.cookie || "";
  const fromCookie =
    cookie.match(/(?:^|;\s*)sid=([^;]+)/)?.[1] ||
    cookie.match(/(?:^|;\s*)dukejia_sid=([^;]+)/)?.[1] ||
    cookie.match(/(?:^|;\s*)hca_sid=([^;]+)/)?.[1];

  const sid =
    body.sessionId ||
    req.headers["x-session-id"] ||
    fromCookie ||
    "anon";

  /* ---------- Reset in-memory session ---------- */
  try {
    SESSIONS.delete(String(sid));
  } catch {
    // ignore
  }

  // Optional: clear cookie for client
  // res.setHeader("Set-Cookie", "sid=; Max-Age=0; Path=/; SameSite=Lax");

  /* ---------- Respond ---------- */
  return res.status(200).json({
    ok: true,
    sessionId: String(sid),
    bot: BOT_NAME,
    message: "Session cleared successfully",
  });
}
