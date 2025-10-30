// api/ask.js — Vercel serverless, continuity + dynamic-entity hybrid RAG + strict fallback
import fs from "fs";
import path from "path";
import { GoogleGenerativeAI } from "@google/generative-ai";
import { Redis } from "@upstash/redis";

/* ─────────────────────────── Paths & Config ─────────────────────────── */
const DATA_DIR = path.join(process.cwd(), "data");
const EMB_PATH = path.join(DATA_DIR, "index.json");

const TOP_K            = parseInt(process.env.TOP_K || "6", 10);
const GENERATION_MODEL = process.env.GENERATION_MODEL || "gemini-2.5-flash";
const EMBEDDING_MODEL  = process.env.EMBEDDING_MODEL  || "text-embedding-004";
// soften slightly; hybrid/entity-lock prevents spurious fallbacks
const MIN_OK_SCORE     = parseFloat(process.env.MIN_OK_SCORE || "0.16");

const BOT_NAME        = process.env.BOT_NAME || "Duki";
const BRAND_NAME      = "Dukejia";
const FRONTEND_GREETS = (process.env.FRONTEND_GREETS ?? "true") !== "false"; // default true

if (!process.env.GOOGLE_API_KEY) {
  throw new Error("Missing GOOGLE_API_KEY env on Vercel");
}

/* ─────────────────────────── Google Gemini SDK ───────────────────────── */
const genAI    = new GoogleGenerativeAI(process.env.GOOGLE_API_KEY);
const embedder = genAI.getGenerativeModel({ model: EMBEDDING_MODEL });
const llm      = genAI.getGenerativeModel({ model: GENERATION_MODEL });

/* ──────────────────────────── Optional Redis ─────────────────────────── */
const redis = process.env.UPSTASH_REDIS_REST_URL && process.env.UPSTASH_REDIS_REST_TOKEN
  ? new Redis({ url: process.env.UPSTASH_REDIS_REST_URL, token: process.env.UPSTASH_REDIS_REST_TOKEN })
  : null;
const SESSION_TTL = parseInt(process.env.SESSION_TTL_SECONDS || "3600", 10);

/* ───────────────────────────── Helpers ──────────────────────────────── */
function getSessionId(req) {
  const h = req.headers["x-session-id"] || req.headers["x-sessionid"] || req.headers["x-client-session"];
  if (h) return String(h).slice(0, 100);
  const ck = (req.headers.cookie || "").match(/sid=([^;]+)/)?.[1];
  if (ck) return ck.slice(0, 100);
  return (req.headers["x-forwarded-for"] || req.socket?.remoteAddress || "anon") + ":" + (req.headers["user-agent"] || "");
}
async function loadHistory(sessionId, k = 10) {
  if (!redis) return [];
  const key = `duki:chat:${sessionId}`;
  const rows = await redis.lrange(key, -k, -1);
  return rows?.map(r => JSON.parse(r)) || [];
}
async function saveTurn(sessionId, role, text) {
  if (!redis) return;
  const key = `duki:chat:${sessionId}`;
  await redis.rpush(key, JSON.stringify({ ts: Date.now(), role, text }));
  await redis.expire(key, SESSION_TTL);
}

function detectResponseMode(q = "") {
  const text = q.toLowerCase();
  if (/[\u0900-\u097F]/.test(text)) return "hinglish";
  const tokens = [
    "hai","hain","tha","thi","the","kya","kyu","kyun","kyunki","kisi","kis","kaun","kab","kaha","kahaan","kaise",
    "nahi","nahin","ka","ki","ke","mein","me","mai","mei","hum","ap","aap","tum","kr","kar","karo","karna","chahiye",
    "bhi","sirf","jaldi","kitna","ho","hoga","hogaya","krdo","pls","plz","yaar","shukriya","dhanyavaad","dhanyavad"
  ];
  let score = 0;
  for (const t of tokens) {
    if (text.includes(` ${t} `) || text.startsWith(t + " ") || text.endsWith(" " + t) || text === t) score += 1;
  }
  const chatCues = (text.match(/[:)(!?]{2,}|\.{3,}|😂|👍|🙏/g) || []).length;
  score += chatCues >= 1 ? 0.5 : 0;
  return score >= 2 ? "hinglish" : "english";
}

const EN_STOP = new Set(`a about above after again against all am an and any are aren't as at
be because been before being below between both but by
can't cannot could couldn't did didn't do does doesn't doing don't down during
each few for from further
had hadn't has hasn't have haven't having he he'd he'll he's her here here's hers herself him himself his how how's
i i'd i'll i'm i've if in into is isn't it it's its itself let's
me more most mustn't my myself
no nor not of off on once only or other ought our ours ourselves out over own
same shan't she she'd she'll she's should shouldn't so some such
than that that's the their theirs them themselves then there there's these they they'd they'll they're they've this those through to too
under until up very
was wasn't we we'd we'll we're we've were weren't what what's when when's where where's which while who who's whom why why's with won't would wouldn't
you you'd you'll you're you've your yours yourself yourselves`.trim().split(/\s+/));

const PROTECTED_TOKENS = new Set([
  // Core company / brands / entities
  "hari","chand","anand","anil","hca","hari-chand-anand","duke","duke-jia","dukejia","duki","contact","call","email","address",
  "head office","factory","website","whatsapp","phone","brand","features","specification","model","models","application","id",
  // Regions / domains
  "delhi","india","bangladesh","ethiopia","automation","garment","leather","mattress","perforation","embroidery","quilting","sewing","upholstery","pattern",
  // Attachments / techniques
  "sequin","sequins","bead","beads","cording","coiling","taping","rhinestone","chenille","chainstitch","cap","tubular",
  // Control systems / file formats
  "dahao","a18","dst","tajima","usb","u-disk","lcd","touchscreen","network",
  // Features / mechanics
  "auto-trimming","automatic-trimming","auto-color-change","automatic-color-change","thread-break-detection","power-failure-recovery",
  "servo","servo-motor","36v","oil-mist","dust-clean","wide-voltage","270-cap-frame"
]);

function cleanForEmbedding(s = "") {
  const lower = s.toLowerCase();
  const stripped = lower.replace(/[^a-z0-9\u0900-\u097F\s-]/g, " ");
  return stripped
    .split(/\s+/)
    .filter(Boolean)
    .filter(t => {
      if (PROTECTED_TOKENS.has(t)) return true;
      if (EN_STOP.has(t)) return false;
      return true;
    })
    .join(" ")
    .trim();
}

function cosineSim(a, b) {
  let dot = 0, na = 0, nb = 0;
  const n = Math.min(a.length, b.length);
  for (let i = 0; i < n; i++) { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
  return dot / (Math.sqrt(na) * Math.sqrt(nb) || 1);
}

function loadVectors() {
  if (!fs.existsSync(EMB_PATH)) throw new Error(`Embeddings not found at ${EMB_PATH}.`);
  const raw = JSON.parse(fs.readFileSync(EMB_PATH, "utf8"));
  if (!raw?.vectors?.length) throw new Error("Embeddings file has no vectors.");
  return raw.vectors;
}

// Load once per cold start
let VECTORS = [];
try { VECTORS = loadVectors(); } catch (e) { console.warn(e.message); }

/* ───────────── Entity & Hybrid Retrieval (dynamic, KB-driven) ───────────── */
/**
 * We dynamically learn model tokens/aliases from the KB (VECTORS) instead of hardcoding.
 * Also includes a generic pattern fallback to catch unseen-but-valid model strings.
 */
let MODEL_SET = new Set();     // normalized tokens (e.g., "dy-1206hc", "cs3000")
let MODEL_ALIASES = new Map(); // norm -> Set of alias strings found in KB text
let MODEL_INIT_DONE = false;

// Normalize a potential model token to a canonical form
function normModelToken(s = "") {
  return s
    .toLowerCase()
    .replace(/\s+/g, "")           // remove spaces
    .replace(/_/g, "-")            // underscores to dashes
    .replace(/–/g, "-")            // en-dash to dash
    .replace(/[()]+/g, "")         // strip parens
    .replace(/[^a-z0-9.+-]/g, "")  // keep letters, digits, dot, plus, dash
    .replace(/^-+|-+$/g, "");      // trim dashes
}

// A permissive "model-like" matcher with at least one digit
const GENERIC_MODEL_RE = /\b([a-z]{1,6}[\w.+-]*\d[\w.+-]*|[a-z]*\d[\w.+-]{1,15})\b/gi;
// Optional brand/prefix bias: prioritize these prefixes when mining
const LIKELY_PREFIX = /^(dy|dukejia|duke|cs|sk|pe|halo|es|tajima|highlead|juki|dyk|dy\-|dy_)/i;

// Build model index from VECTORS content (run once after VECTORS load)
function initModelIndexOnce() {
  if (MODEL_INIT_DONE) return;
  MODEL_INIT_DONE = true;

  const localSet = new Set();
  const aliasMap = new Map();

  for (const v of VECTORS || []) {
    const txt = String(v.text_original || v.text_cleaned || v.text || "");
    if (!txt) continue;

    // Prefer explicit metadata if present
    const metaModel = v.model || v.meta?.model || v.meta?.Model || null;
    if (metaModel) {
      const n = normModelToken(String(metaModel));
      if (n && /\d/.test(n)) {
        localSet.add(n);
        if (!aliasMap.has(n)) aliasMap.set(n, new Set([n]));
      }
    }

    // Mine tokens that look like model names from the text
    const seenHere = new Set();
    let m;
    while ((m = GENERIC_MODEL_RE.exec(txt)) !== null) {
      const raw = m[1];
      if (!raw) continue;

      if (!/\d/.test(raw)) continue;
      if (raw.length < 3 || raw.length > 24) continue;

      const n = normModelToken(raw);
      if (!n || n.length < 3) continue;

      // Heuristic boosts: plausible family prefix, or contains dash/plus/dot, or letters+digits start
      const goodShape = LIKELY_PREFIX.test(raw) || /[-+.]/.test(raw) || /^[a-z]+\d/.test(raw);
      if (!goodShape) continue;

      seenHere.add(n);
    }

    for (const n of seenHere) {
      localSet.add(n);
      if (!aliasMap.has(n)) aliasMap.set(n, new Set());
      aliasMap.get(n).add(n);
    }

    // Add simple alias variations for tokens from this chunk
    for (const n of seenHere) {
      const variants = new Set([n, n.replace(/-/g, ""), n.replace(/\./g, ""), n.replace(/\+/g, "")]);
      const withSpaces = n.replace(/-/g, " ").replace(/\./g, " ").replace(/\+/g, " ");
      variants.add(withSpaces);
      const bucket = aliasMap.get(n) || new Set();
      for (const v of variants) bucket.add(v);
      aliasMap.set(n, bucket);
    }
  }

  MODEL_SET = localSet;
  MODEL_ALIASES = aliasMap;
}

// Extract entities from current text + short history using dynamic sets.
// Also catch valid unseen tokens via a generic fallback.
function extractEntities(text = "", history = []) {
  initModelIndexOnce();

  const combined = [text, ...history.map(h => h.text || "")].join(" ");
  const found = new Set();

  // 1) Alias match against known KB-derived models
  if (MODEL_SET.size > 0) {
    const lower = combined.toLowerCase();

    for (const canonical of MODEL_SET) {
      const aliases = MODEL_ALIASES.get(canonical) || new Set([canonical]);
      for (const alias of aliases) {
        if (!alias || alias.length < 3) continue;
        const needle = alias.toLowerCase();
        if (lower.includes(needle)) {
          found.add(canonical);
          break;
        }
      }
    }
  }

  // 2) Generic fallback for unseen-but-valid model strings
  let m;
  const localUnseen = new Set();
  GENERIC_MODEL_RE.lastIndex = 0;
  while ((m = GENERIC_MODEL_RE.exec(combined)) !== null) {
    const raw = m[1];
    if (!raw) continue;
    if (raw.length < 3 || raw.length > 24) continue;
    if (!/\d/.test(raw)) continue;
    const n = normModelToken(raw);
    if (!n) continue;

    // Filter out tiny generic tokens like v2/v3 unless prefixed (dy-v2 ok)
    if (/^v\d{1,2}$/.test(n)) continue;

    if (LIKELY_PREFIX.test(raw) || /[-+.]/.test(raw) || /^[a-z]+\d/.test(raw)) {
      localUnseen.add(n);
    }
  }
  for (const x of localUnseen) found.add(x);

  return [...found];
}

// Keyword bonus: match either canonical or any alias quickly
function kwScoreFor(v, kwList) {
  if (!kwList?.length) return 0;
  const t = (v.text_original || v.text_cleaned || v.text || "").toLowerCase();
  let s = 0;

  // Build alias bag for queried keywords (so “dy1206h”, “dy-1206h”, “dy 1206h” all count)
  const aliasBag = new Set();
  for (const kw of kwList) {
    const n = normModelToken(kw);
    aliasBag.add(n);
    aliasBag.add(n.replace(/-/g, ""));
    aliasBag.add(n.replace(/\./g, ""));
    aliasBag.add(n.replace(/\+/g, ""));
    aliasBag.add(n.replace(/-/g, " ").replace(/\./g, " ").replace(/\+/g, " "));
    if (MODEL_ALIASES.has(n)) {
      for (const al of MODEL_ALIASES.get(n)) aliasBag.add(al.toLowerCase());
    }
  }
  for (const alias of aliasBag) {
    if (!alias || alias.length < 3) continue;
    if (t.includes(alias)) s += 1;
  }
  return s;
}

/* ──────────────────────── Small-talk (ported) ───────────────────────── */
function getISTGreeting(now = new Date()) {
  const hour = Number(
    new Intl.DateTimeFormat("en-GB", { timeZone: "Asia/Kolkata", hour: "2-digit", hour12: false }).format(now)
  );
  if (hour < 5)  return "Good night";
  if (hour < 12) return "Good morning";
  if (hour < 17) return "Good afternoon";
  if (hour < 21) return "Good evening";
  return "Good night";
}
function buildMinimalAssist(mode) {
  return mode === "hinglish" ? "Kaise madad kar sakta hoon?" : "How can I assist you?";
}
function makeSmallTalkReply(kind, mode) {
  const en = {
    hello: ["Hi! How can I help today?","How can I help you?"],
    morning: ["Good morning! How can I help today?"],
    afternoon: ["Good afternoon! How can I help today?"],
    evening: ["Good evening! Need help with machines or spares?"],
    thanks: ["You’re welcome! Anything else I can do?","Happy to help! Need brochures or a sales connect?"],
    bye: ["Take care! I’m here if you need me.","Bye! Have a great day."],
    help: ["Ask about flagship lines, suggestions by application, or spares."],
    ack: ["Got it! What would you like next?"],
  };
  const hi = {
    hello: ["Namaste 👋 Duki se related kya madad chahiye?","Hello ji 👋 Main madad ke liye hoon—puchhiye."],
    morning: ["Good morning! Aaj kis cheez mein help chahiye?"],
    afternoon: ["Good afternoon! Duki ke baare mein kya jaana hai?"],
    evening: ["Good evening! Machines/spares par madad chahiye to batayein."],
    thanks: ["Shukriya! Aur kuch chahiye to pooch lijiye.","Welcome ji! Brochure chahiye ya sales connect karu?"],
    bye: ["Theek hai, milte hain! Jab chahein ping kar dijiyega.","Bye! Din shubh rahe."],
    help: ["Try: “Flagship features”, “Application-wise machine suggestion”, “Spares info”."],
    ack: ["Thik hai! Ab kya puchhna hai?"],
  };
  const bank = mode === "hinglish" ? hi : en;
  const pick = (arr) => arr[Math.floor(Math.random() * arr.length)];
  switch (kind) {
    case "hello":     return pick(bank.hello);
    case "morning":   return pick(bank.morning);
    case "afternoon": return pick(bank.afternoon);
    case "evening":   return pick(bank.evening);
    case "thanks":    return pick(bank.thanks);
    case "bye":       return pick(bank.bye);
    case "help":      return pick(bank.help);
    case "ack":       return pick(bank.ack);
    default:          return pick(bank.hello);
  }
}
function smallTalkMatch(q) {
  const t = (q || "").trim();
  const patterns = [
    { kind: "hello",     re: /^(hi+|h[iy]+|hello+|hey( there)?|hlo+|yo+|hola|namaste|namaskar|salaam|salam|👋|🙏)\b/i },
    { kind: "morning",   re: /^(good\s*morning|gm)\b/i },
    { kind: "afternoon", re: /^(good\s*afternoon|ga)\b/i },
    { kind: "evening",   re: /^(good\s*evening|ge)\b/i },
    { kind: "ack",       re: /^(ok+|okay+|okk+|hmm+|haan+|ha+|sure|done|great|nice|cool|perfect|thik|theek|fine)\b/i },
    { kind: "thanks",    re: /^(thanks|thank\s*you|thx|tnx|ty|much\s*(appreciated|thanks)|appreciate(d)?|shukriya|dhanyavaad|dhanyavad)\b/i },
    { kind: "bye",       re: /^(bye|bb|good\s*bye|goodbye|see\s*ya|see\s*you|take\s*care|tc|ciao|gn)\b/i },
    { kind: "help",      re: /(who\s*are\s*you|what\s*can\s*you\s*do|help|menu|options|how\s*to\s*use)\b/i },
  ];
  for (const p of patterns) if (p.re.test(t)) return p.kind;
  return null;
}
function handleSmallTalkAll(q, { isFirstTurn = false } = {}) {
  if (!q) return null;
  const mode = detectResponseMode(q);
  const trimmed = q.trim();

  const short = trimmed.toLowerCase().replace(/[^a-z]/g, "");
  const HELLO_SHORT  = new Set(["hi","hey","yo","sup"]);
  const BYE_SHORT    = new Set(["bye","bb","ciao","gn"]);
  const THANKS_SHORT = new Set(["ty","thx","tnx","tx"]);
  const GM_SHORT     = new Set(["gm"]);
  const GA_SHORT     = new Set(["ga"]);
  const GE_SHORT     = new Set(["ge"]);

  let quickKind =
    (HELLO_SHORT.has(short)  && "hello")    ||
    (BYE_SHORT.has(short)    && "bye")      ||
    (THANKS_SHORT.has(short) && "thanks")   ||
    (GM_SHORT.has(short)     && "morning")  ||
    (GA_SHORT.has(short)     && "afternoon")||
    (GE_SHORT.has(short)     && "evening")  ||
    null;

  const isBlank = trimmed.replace(/[?.!\s]/g, "") === "";
  const isGreetingWord = /^(hi+|hello+|hey( there)?|hlo+|namaste|namaskar|salaam|gm|ga|ge|👋|🙏)$/i.test(trimmed);

  if (quickKind) {
    if (isFirstTurn && FRONTEND_GREETS && ["hello","morning","afternoon","evening"].includes(quickKind)) {
      return { text: buildMinimalAssist(mode) };
    }
    return { text: makeSmallTalkReply(quickKind, mode) };
  }

  const kind = smallTalkMatch(trimmed);
  if (kind) {
    if (isFirstTurn && FRONTEND_GREETS && ["hello","morning","afternoon","evening"].includes(kind)) {
      return { text: buildMinimalAssist(mode) };
    }
    if (["morning","afternoon","evening"].includes(kind)) {
      const sal = getISTGreeting();
      const line = mode === "hinglish"
        ? (kind === "morning" ? "Good morning! Aaj kis cheez mein help chahiye?"
            : kind === "afternoon" ? "Good afternoon! Duki ke baare mein kya jaana hai?"
            : "Good evening! Machines/spares par madad chahiye to batayein.")
        : `${sal}! How can I help today?`;
      return { text: line };
    }
    return { text: makeSmallTalkReply(kind, mode) };
  }

  if (isFirstTurn && FRONTEND_GREETS && (isBlank || isGreetingWord)) {
    return { text: buildMinimalAssist(mode) };
  }
  return null;
}

/* ───────────────────────────── Handler ───────────────────────────────── */
export default async function handler(req, res) {
  // CORS preflight
  if (req.method === "OPTIONS") {
    res.setHeader("Access-Control-Allow-Origin", req.headers.origin || "*");
    res.setHeader("Access-Control-Allow-Methods", "POST, OPTIONS");
    res.setHeader("Access-Control-Allow-Headers", "Content-Type, X-Session-ID, X-SessionID, X-Client-Session");
    res.setHeader("Access-Control-Allow-Credentials", "true");
    return res.status(204).end();
  }
  if (req.method !== "POST") {
    res.setHeader("Allow", ["POST", "OPTIONS"]);
    return res.status(405).json({ error: "Method Not Allowed" });
  }

  // Basic CORS
  res.setHeader("Access-Control-Allow-Origin", req.headers.origin || "*");
  res.setHeader("Access-Control-Allow-Credentials", "true");

  try {
    // Accept both message|question; allow optional isFirstTurn
    const body = typeof req.body === "string" ? JSON.parse(req.body) : (req.body || {});
    const q = (body.message ?? body.question ?? "").toString().trim();
    const isFirstTurn = !!body.isFirstTurn;

    if (!q && !isFirstTurn) {
      return res.status(400).json({ error: "Missing 'message' or 'question'." });
    }

    const sessionId = getSessionId(req);
    const history = await loadHistory(sessionId, 10);

    // Full small-talk
    const st = handleSmallTalkAll(q, { isFirstTurn });
    if (st && st.text) {
      await saveTurn(sessionId, "user", q || "");
      await saveTurn(sessionId, "assistant", st.text);
      return res.status(200).json({ answer: st.text, citations: [], mode: detectResponseMode(q || ""), bot: BOT_NAME });
    }

    // RAG guard
    if (!VECTORS.length) {
      const msg = "Embeddings not loaded on server. Add data/index.json (npm run embed) and redeploy.";
      await saveTurn(sessionId, "user", q || "");
      await saveTurn(sessionId, "assistant", msg);
      return res.status(500).json({ error: msg });
    }

    const mode = detectResponseMode(q);
    const cleaned = cleanForEmbedding(q) || q.toLowerCase();

    // Sticky entities from current + recent history
    const stickyEntities = extractEntities(q, history);
    const kwList = [...stickyEntities];

    // Embed the query
    const embRes = await embedder.embedContent({ content: { parts: [{ text: cleaned }] } });
    const qVec = embRes?.embedding?.values || embRes?.embeddings?.[0]?.values || [];
    if (!qVec.length) {
      const msg = "Embedding failed";
      await saveTurn(sessionId, "user", q || "");
      await saveTurn(sessionId, "assistant", msg);
      return res.status(500).json({ error: msg });
    }

    // Hybrid retrieval: cosine + keyword bonus
    const HYBRID_BONUS = 0.12;
    const candidates = VECTORS.map(v => {
      const cos = cosineSim(qVec, v.embedding);
      const kw = kwScoreFor(v, kwList);
      const score = cos + (kw > 0 ? HYBRID_BONUS * Math.min(kw, 3) : 0);
      return { ...v, score, cos, kw };
    }).sort((a,b) => b.score - a.score).slice(0, TOP_K);

    // Entity lock & passability
    const topHit = candidates[0];
    const modelFoundInTop = stickyEntities.length > 0 && candidates.some(c => {
      const txt = (c.text_original || c.text_cleaned || c.text || "").toLowerCase();
      return stickyEntities.some(m => txt.includes(m));
    });
    const passable = (topHit?.score ?? 0) >= (MIN_OK_SCORE - 0.04) || modelFoundInTop;

    if (!passable) {
      // STRICT: do NOT leak contact early; ask for specificity instead
      const tip = mode === "hinglish"
        ? "Is topic par Dukejia knowledge base mein clear info nahi mil rahi. Thoda specific likhiye—jaise 'DY-CS3000 specs' ya '1206H applications'."
        : "I couldn’t find clear context in the Dukejia knowledge base for that. Try being more specific—for example, 'DY-CS3000 specs' or '1206H applications'.";
      await saveTurn(sessionId, "user", q || "");
      await saveTurn(sessionId, "assistant", tip);
      return res.status(200).json({ answer: tip, citations: [], mode, bot: BOT_NAME });
    }

    // Build context + recent chat
    const context = candidates.map((s,i) => `【${i+1}】 ${s.text_original || s.text_cleaned || s.text}`).join("\n\n");
    const recentChat = history.slice(-6)
      .map(h => (h.role === "user" ? `User: ${h.text}` : `Assistant: ${h.text}`))
      .join("\n");

    // System guardrails — STRICT: only show contact if truly not in context
    const languageGuide = mode === "hinglish"
      ? `REPLY LANGUAGE: Hinglish (Hindi in Latin script). Do NOT use Devanagari.`
      : `REPLY LANGUAGE: English. Professional and concise.`;

    const systemInstruction = `
You are ${BOT_NAME}, Dukejia’s assistant.
Answer STRICTLY and ONLY from the provided CONTEXT.
If the requested details are clearly NOT present in CONTEXT, reply exactly:
"Please contact our sales team at 
Whatsapp: +91 9350513789 
Embroidery@grouphca.com"
Rules:
- Do not invent or add external knowledge.
- Be concise and factual.
- Prefer details about these STICKY ENTITIES if present: ${stickyEntities.join(", ") || "none"}.
- Use recent chat for continuity if it helps resolve the user’s intent.
- Absolutely do NOT include contact details unless the answer is not present in CONTEXT.
- ${languageGuide}
`.trim();

    const prompt = `
${systemInstruction}

RECENT CHAT (for continuity):
${recentChat || "(none)"}

QUESTION:
${q}

CONTEXT (numbered blocks):
${context}

Format:
- Direct answer grounded in CONTEXT (bullets/tables ok).
- If NOT found in CONTEXT: “Please contact our sales team at 
Whatsapp: +91 9350513789 
Embroidery@grouphca.com”
- Use the reply language specified above.
`.trim();

    const result = await llm.generateContent({ contents: [{ role: "user", parts: [{ text: prompt }] }] });
    let text = result?.response?.text?.() || "";

    // Final safety: block premature contact drop if we actually had passable context
    const contactLine = "Please contact our sales team";
    const containsContact = text.includes(contactLine);
    const contextLooksNonEmpty = Boolean(context && context.trim().length > 0);

    if (containsContact && contextLooksNonEmpty) {
      // Replace with a polite “not in context” nudge instead (no contact shared early)
      text = mode === "hinglish"
        ? "Mujhe context mein exact details nahi mili. Agar aap model/feature thoda aur specific batayenge to main exact specs dikha sakta hoon."
        : "I didn’t see the exact details in the context. If you specify the model/feature a bit more, I can pull precise specs.";
    }
    if (!text) {
      // As a last resort, still avoid early contact unless truly empty context
      text = contextLooksNonEmpty
        ? (mode === "hinglish"
            ? "Main context se details nikal raha hoon—please model/feature thoda aur specific batayein."
            : "I’m using the knowledge base—please specify the model/feature you need.")
        : "Please contact our sales team at \nWhatsapp: +91 9350513789 \nEmbroidery@grouphca.com";
    }

    await saveTurn(sessionId, "user", q || "");
    await saveTurn(sessionId, "assistant", text);

    return res.status(200).json({
      answer: text,
      mode,
      bot: BOT_NAME,
      citations: candidates.map((s,i) => ({
        idx: i + 1,
        score: Number(s.score.toFixed(4)),
        kw: s.kw,
        cos: Number(s.cos.toFixed(4))
      })),
    });
  } catch (err) {
    console.error("ask error:", err);
    return res.status(err?.status || 500).json({
      error: err?.message || "Server error",
      details: { status: err?.status || 500, statusText: err?.statusText || null, type: err?.name || null }
    });
  }
}
