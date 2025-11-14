// /api/session.js — Debug / inspection endpoint for Duki sessions
// Shows what the server remembers for a given sessionId.
//
// Notes:
// - Shares the same in-memory SESSIONS map as /api/ask.js
// - If you're also using Redis for long-term history, this endpoint
//   will still show the in-process snapshot (what this cold-start
//   instance has seen so far).

export const config = { runtime: "nodejs" };

// Shared in-memory map per cold start (serverless-safe).
// Prefer Dukejia map; fall back to legacy HCA map if present.
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

  /* ---------- Extract session ID (header or cookies) ---------- */
  const cookie = req.headers.cookie || "";
  const cookieSid =
    cookie.match(/(?:^|;\s*)sid=([^;]+)/)?.[1] ||
    cookie.match(/(?:^|;\s*)dukejia_sid=([^;]+)/)?.[1] ||
    cookie.match(/(?:^|;\s*)hca_sid=([^;]+)/)?.[1];

  const headerSid =
    req.headers["x-session-id"] ||
    req.headers["x-sessionid"] ||
    req.headers["x-client-session"];

  const sid = headerSid || cookieSid || "anon";

  /* ---------- Read session safely from SESSIONS map ---------- */
  const sess = SESSIONS.get(String(sid)) || {};
  const history = Array.isArray(sess.history) ? sess.history : [];

  // Optional: short preview of first few messages (to eyeball quickly)
  const preview = history.slice(-5).map((m) => ({
    ts: m.ts,
    role: m.role,
    text: typeof m.text === "string" ? m.text.slice(0, 200) : m.text,
  }));

  /* ---------- Response ---------- */
  return res.status(200).json({
    ok: true,
    bot: BOT_NAME,
    sessionId: String(sid),
    historyLength: history.length || 0,
    messages: history,
    preview, // last few turns, truncated text
    createdAt: sess.createdAt || null,
    lastSeen: sess.lastSeen || null,
    hits: sess.hits || 0,
    env: process.env.VERCEL_ENV || "production",
  });
}
