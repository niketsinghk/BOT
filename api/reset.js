// /api/reset.js — Vercel serverless (Node.js runtime)
// Clears server-side session state for Duki (in-memory SESSIONS map).
// Optionally used by frontend "Reset chat" button.

export const config = { runtime: "nodejs" };

// Shared in-memory session map (same pattern as /api/ask.js and /api/session.js)
const SESSIONS =
  globalThis.__DUKEJIA_SESSIONS__ ??
  globalThis.__HCA_SESSIONS__ ??
  (globalThis.__DUKEJIA_SESSIONS__ = new Map());

const BOT_NAME = process.env.BOT_NAME || "Duki";

export default async function handler(req, res) {
  /* ---------- CORS + preflight ---------- */
  const origin = req.headers.origin || "*";
  res.setHeader("Access-Control-Allow-Origin", origin);
  res.setHeader("Access-Control-Allow-Credentials", "true");
  res.setHeader("Access-Control-Allow-Methods", "POST, OPTIONS");
  res.setHeader(
    "Access-Control-Allow-Headers",
    "Content-Type, X-Session-ID, X-SessionID, X-Client-Session"
  );
  res.setHeader("Cache-Control", "no-store");

  if (req.method === "OPTIONS") {
    return res.status(204).end();
  }

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

  /* ---------- Extract session ID (body, header, cookie) ---------- */
  const cookie = req.headers.cookie || "";
  const fromCookie =
    cookie.match(/(?:^|;\s*)sid=([^;]+)/)?.[1] ||
    cookie.match(/(?:^|;\s*)dukejia_sid=([^;]+)/)?.[1] ||
    cookie.match(/(?:^|;\s*)hca_sid=([^;]+)/)?.[1];

  const fromHeader =
    req.headers["x-session-id"] ||
    req.headers["x-sessionid"] ||
    req.headers["x-client-session"];

  const sid =
    body.sessionId ||
    fromHeader ||
    fromCookie ||
    "anon";

  /* ---------- Reset in-memory session ---------- */
  try {
    SESSIONS.delete(String(sid));
  } catch {
    // ignore errors
  }

  // If you also want to clear client cookie, uncomment:
  // res.setHeader("Set-Cookie", "sid=; Max-Age=0; Path=/; SameSite=Lax");

  /* ---------- Respond ---------- */
  return res.status(200).json({
    ok: true,
    bot: BOT_NAME,
    sessionId: String(sid),
    message: "Session cleared successfully",
  });
}
