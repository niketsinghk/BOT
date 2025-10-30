// server.mjs — Dukejia-bot (first-turn greeting trimmed to a single minimal line)
import "dotenv/config";
import express from "express";
import cors from "cors";
import fs from "fs";
import path from "path";
import cookieParser from "cookie-parser";
import { v4 as uuidv4 } from "uuid";
import { GoogleGenerativeAI } from "@google/generative-ai";
import { fileURLToPath } from "url";

/* ─────────────────────────── Paths & Config ─────────────────────────── */
const __filename = fileURLToPath(import.meta.url);
const __dirname  = path.dirname(__filename);

const PORT         = parseInt(process.env.PORT || "5173", 10);
const TOP_K        = parseInt(process.env.TOP_K || "6", 10);
const DATA_DIR     = path.join(__dirname, "data");
const EMB_PATH     = path.join(DATA_DIR, "index.json");

const GENERATION_MODEL = process.env.GENERATION_MODEL || "gemini-2.5-flash";
const EMBEDDING_MODEL  = process.env.EMBEDDING_MODEL  || "text-embedding-004";
const BOT_NAME         = process.env.BOT_NAME || "Duki";

// NEW: point-wise reply toggle (server-side)
const POINTWISE_MODE   = process.env.POINTWISE_MODE !== "false"; // default true
console.log("POINTWISE_MODE:", process.env.POINTWISE_MODE, "=>", POINTWISE_MODE);

// Optional: if your frontend already shows the greeting bubble, keep backend minimal on first 'hi'
const FRONTEND_GREETS  = true;

if (!process.env.GOOGLE_API_KEY) {
  console.error("❌ Missing GOOGLE_API_KEY in .env");
  process.exit(1);
}

/* ───────────────────────────── Express App ───────────────────────────── */
const app = express();
app.set("trust proxy", 1);

app.use(
  cors({
    origin: true,
    credentials: true,
  })
);
app.use(express.json({ limit: "4mb" }));
app.use(cookieParser());

// Static files (optional)
app.use(express.static(path.join(__dirname, "public")));
app.use("/images", express.static(path.join(__dirname, "images")));

/* ─────────────────────────── Google Gemini SDK ───────────────────────── */
const genAI    = new GoogleGenerativeAI(process.env.GOOGLE_API_KEY);
const embedder = genAI.getGenerativeModel({ model: EMBEDDING_MODEL });
const llm      = genAI.getGenerativeModel({ model: GENERATION_MODEL });

/* ─────────────────────────── In-Memory Sessions ──────────────────────── */
const sessions = new Map(); // sid -> { history:[], createdAt, lastSeen, hits }

/** Time-of-day greeting in IST (kept for other uses if needed) */
function getISTGreeting(now = new Date()) {
  const hour = Number(
    new Intl.DateTimeFormat("en-GB", {
      timeZone: "Asia/Kolkata",
      hour: "2-digit",
      hour12: false,
    }).format(now)
  );
  if (hour < 5)  return "Good night";
  if (hour < 12) return "Good morning";
  if (hour < 17) return "Good afternoon";
  if (hour < 21) return "Good evening";
  return "Good night";
}

/** First-turn minimal line ONLY when user greets and frontend already introduced the bot */
function buildMinimalAssist(mode) {
  return mode === "hinglish" ? "Kaise madad kar sakta hoon?" : "How can I assist you?";
}

/** Build a full greeting (not used on first user 'hi' anymore) */
function buildGreeting(mode) {
  const base = getISTGreeting();
  if (mode === "hinglish") {
    return `${base}! Main ${BOT_NAME} hoon. How can I help you today?`;
  }
  return `${base}! I’m ${BOT_NAME}. How can I help you today?`;
}

/** Attach or create session, echo X-Session-ID for debugging */
function sessionMiddleware(req, res, next) {
  let sid = req.get("X-Session-ID") || req.body?.sessionId || req.cookies?.sid;
  if (!sid || typeof sid !== "string" || sid.length > 200) {
    sid = uuidv4();
    res.cookie("sid", sid, {
      httpOnly: true,
      sameSite: "Lax",
      secure: !!process.env.COOKIE_SECURE,
      maxAge: 1000 * 60 * 60 * 24 * 30,
    });
  }
  res.setHeader("X-Session-ID", sid);

  const now = Date.now();
  if (!sessions.has(sid)) {
    sessions.set(sid, { history: [], createdAt: now, lastSeen: now, hits: 0 });
  } else {
    sessions.get(sid).lastSeen = now;
  }
  sessions.get(sid).hits += 1;

  req.sid = sid;
  req.session = sessions.get(sid);
  next();
}

/* ───────────────────── Language Mode (EN / Hinglish) ─────────────────── */
function detectResponseMode(q) {
  const text = (q || "").toLowerCase();
  if (/[\u0900-\u097F]/.test(text)) return "hinglish";
  const hinglishTokens = [
    "hai","hain","tha","thi","the","kya","kyu","kyun","kyunki","kisi","kis","kaun","kab","kaha","kahaan","kaise",
    "nahi","nahin","ka","ki","ke","mein","me","mai","mei","hum","ap","aap","tum","kr","kar","karo","karna","chahiye",
    "bhi","sirf","jaldi","kitna","ho","hoga","hogaya","krdo","pls","plz","yaar","shukriya","dhanyavaad","dhanyavad"
  ];
  let score = 0;
  for (const t of hinglishTokens) {
    if (text.includes(` ${t} `) || text.startsWith(t + " ") || text.endsWith(" " + t) || text === t) score += 1;
  }
  const chatCues = (text.match(/[:)(!?]{2,}|\.{3,}|😂|👍|🙏/g) || []).length;
  score += chatCues >= 1 ? 0.5 : 0;
  return score >= 2 ? "hinglish" : "english";
}

/* ───────────────────── Stopwords / Cleaner for Embeds ────────────────── */
const EN_STOPWORDS = new Set(`a about above after again against all am an and any are aren't as at
be because been before being below between both but by
can't cannot could couldn't did didn't do does doesn't doing don't down during
each few for from further
had hadn't has hasn't have haven't having he he'd he'll he's her here here's hers herself him himself his how how's
i i'd i'll i'm i've if in into is isn't it it's its itself
let's
me more most mustn't my myself
no nor not of off on once only or other ought our ours ourselves out over own
same shan't she she'd she'll she's should shouldn't so some such
than that that's the their theirs them themselves then there there's these they they'd they'll they're they've this those through to too
under until up very
was wasn't we we'd we'll we're we've were weren't what what's when when's where where's which while who who's whom why why's with won't would wouldn't
you you'd you'll you're you've your yours yourself yourselves`.trim().split(/\s+/));

const HINGLISH_STOPWORDS = new Set([
  "hai","hain","ho","hona","hoga","hogi","honge","hote","hota","thi","tha","the",
  "kya","kyu","kyun","kyunki","kisi","kis","kaun","konsa","kab","kaha","kahaan","kaise",
  "nahi","nahin","na","mat","bas","sirf","bhi","hi","to","tho","ab","abhi","phir","fir",
  "ye","yeh","vo","woh","aisa","waisa","jab","tab","agar","lekin","magar","par","per","ya","aur",
  "ka","ki","ke","mein","me","mai","mei","mujhe","mujhko","hume","humko","tumhe","aap","ap","hum","tum",
  "se","ko","tak","pe","par","liye","ke","liye",
  "kr","kar","karo","karna","karke","krke","krna","ho gya","hogaya","chahiye","chahie","krdo","kardo","de","do","lo","le","dena","lena"
]);

const HINDI_STOPWORDS = new Set([
  "है","हैं","हो","होना","होगा","होगी","होंगे","होते","होता","था","थी","थे",
  "क्या","क्यों","क्योंकि","किसी","कौन","कौनसा","कब","कहाँ","कैसे",
  "नहीं","मत","बस","सिर्फ","भी","ही","तो","अब","अभी","फिर",
  "यह","ये","वह","वो","जब","तब","अगर","लेकिन","मगर","या","और",
  "का","की","के","में","मे","मुझे","हमें","तुम्हें","आप","हम","तुम",
  "से","को","तक","पर","लिए","चाहिए","कर","करो","करना","करके","कर दें","कर लो"
]);

const PROTECTED_TOKENS = new Set([
  "hari","chand","anand","anil","hca","duke","duke-jia","dukejia","duki",
  "delhi","india","automation","garment","leather","perforation","embroidery","quilting","sewing",
  "pattern","dst","tajima","servo","36v",

  // (models trimmed for brevity; add as needed)
  "duke", "duke-jia", "dukejia", "duki","contact","call","email","address","Branches","Headquarters", 
  "Head Office","Factory","Works","WhatsApp","Whatsapp","Whats app","Phone", "Brand","Brands","names","name","features",
  "feature","specification","specifications","specs","model","models","type","types", "descriptions","description","Appllications",
  "application","Machine_id","Machine ID","ID","needle","niddle","heads","head","speed", "rpm","Embroidery Area","phase","phases",

  // ─── Regions ─── //
  "delhi", "india", "bangladesh", "ethiopia", 

  // ─── Domains / industries ─── //
  "automation", "solution", "solutions", "garment", "leather", "mattress", "perforation", "embroidery", "quilting", "sewing", "upholstery","pattern", 

  // ─── Attachments / techniques ─── 
  "sequin", "sequins", "bead", "beads", "cording", "coiling", "taping", "rhinestone", "chenille", "chainstitch", "cap", "tubular",

  // ─── Control systems / file formats ─── 
  "dahao", "a18", "dst", "tajima", "usb", "u-disk", "lcd", "touchscreen", "network", 

  // ─── Features / safety / mechanics ───
  "auto-trimming", "automatic-trimming", "auto-color-change", "automatic-color-change", "thread-break-detection", "power-failure-recovery", 
  "servo", "servo-motor", "36v", "36v-dc", "oil-mist", "dust-clean", "wide-voltage", "270-cap-frame", 

  // ─── Machine models (embroidery & related) ─── 
  "es-1300","es 1300","dy pe750x600","halo-100", "dy-601ctm","dy sk d2-2.0rh", "dy-606", "dy-606h", "dy-606hc", "dy-606l", "dy-606xl",
  "dy-606s", "dy-606+1ct", "dy-606+1pd", "dy-602", "dy-602h", "dy-602hc", "dy-602l", "dy-602xl", "dy-602s", "dy-602+1ct", "dy-602+1pd",
  "dy 601ctm", "dy 606+6", "dy602+2", "dy-606+6","dy 908","dy 912","dy 915-120","dy 918-120", "dy 1201", "dy 1201l", "dy 1201h", "dy 1201xl", "dy 1201s",
  "dy-1201", "dy-1201l", "dy-1201h", "dy-1201xl", "dy-1201s","dy Halo-100", "dy-1201+1ct", "dy-1201+1pd","dy 1204","dy 1206","dy 1206h","dy 1206hc", "dy-1204",
  "dy-1206", "dy-1206h", "dy-1206hc","dy-1202h","dy 1202h", "dy-1202", "dy-1202l", "dy-1202h", "dy-1202xl", "dy-1202s","dy 918","dy 915", "dy 1502","dy-1502",
  "dy-1202hc", "dy-1203h", "dy-1204", "dy-1206", "dy-1206h", "dy-1502", "dy-908", "dy-912", "dy915-120", "dy918-120","dy cs3000", "duke-single-head","duke multi-head",
  "duke multi head","duke multihead", "dukejia-single-head","dukejia multi-head","dukejia multi head","dukejia multihead", 

  // ─── Non-embroidery models (brand-relevant) ───
  "dy-cs3000", "dy-pe750x600", "dy-sk-d2-2.0rh"
]);

function cleanForEmbedding(s) {
  if (!s) return "";
  const lower = s.toLowerCase();
  const stripped = lower.replace(/[^a-z0-9\u0900-\u097F\s-]/g, " ");
  const tokens = stripped.split(/\s+/).filter(Boolean);
  const kept = tokens.filter((t) => {
    if (PROTECTED_TOKENS.has(t)) return true;
    if (EN_STOPWORDS.has(t)) return false;
    if (HINGLISH_STOPWORDS.has(t)) return false;
    if (HINDI_STOPWORDS.has(t)) return false;
    return true;
  });
  return kept.join(" ").trim();
}

/* ────────────────────────── Embeddings (RAG) ─────────────────────────── */
function cosineSim(a, b) {
  let dot = 0, na = 0, nb = 0;
  const n = Math.min(a.length, b.length);
  for (let i = 0; i < n; i++) { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
  return dot / (Math.sqrt(na) * Math.sqrt(nb) || 1);
}
function loadVectors() {
  if (!fs.existsSync(EMB_PATH)) throw new Error(`Embeddings not found at ${EMB_PATH}. Run "npm run embed" first.`);
  const raw = JSON.parse(fs.readFileSync(EMB_PATH, "utf8"));
  if (!raw?.vectors?.length) throw new Error("Embeddings file has no vectors.");
  return raw.vectors;
}
let VECTORS = [];
try { VECTORS = loadVectors(); console.log(`🗂️  Loaded ${VECTORS.length} vectors`); }
catch (err) { console.warn("⚠️", err.message); }
try {
  fs.watch(EMB_PATH, { persistent: false }, () => {
    try { VECTORS = loadVectors(); console.log(`♻️  Reloaded ${VECTORS.length} vectors`); }
    catch (e) { console.warn("⚠️ Reload failed:", e?.message || e); }
  });
} catch { /* ignore */ }

/* ─────────────────────── Point-wise (server-side) ────────────────────── */
function isListLike(text = "") {
  return /^\s*([-*•]|\d+\.)\s+/m.test(text);
}
function toPointWise(text = "") {
  if (!POINTWISE_MODE) return text;
  if (!text || isListLike(text)) return text;

  // Normalize whitespace
  const norm = text.replace(/\r/g, "").replace(/[ \t]+\n/g, "\n").trim();

  // Split on: blank lines | sentence boundaries | semicolons | bullets | " - " separators
  let parts = norm
    .split(/\n{2,}|(?<=[.!?])\s+(?=[A-Z(0-9])|[;•]|(?:\s+-\s+)/)
    .map(s => s.trim())
    .filter(Boolean);

  // Fallback: single newlines if we still didn't separate
  if (parts.length < 2) {
    const byLine = norm.split(/\n+/).map(s => s.trim()).filter(Boolean);
    if (byLine.length >= 2) parts = byLine;
  }
  if (parts.length < 2) return text;

  return parts
    .map(p => p.replace(/^[•*\-]\s+/, ""))  // strip accidental bullet
    .map(p => p.replace(/\s*\.\s*$/, ""))   // drop trailing dot
    .map(p => `- ${p}`)
    .join("\n");
}

/* ───────────────────────── Small Talk Replies ───────────────────────── */
function makeSmallTalkReply(kind, mode) {
  const en = {
    hello: [
      "Hi! How can I help today?",
      "How can I help with Duki today?",
    ],
    morning:   ["Good morning! How can I help today?"],
    afternoon: ["Good afternoon! How can I help today?"],
    evening:   ["Good evening! Need help with machines or spares?"],
    thanks: [
      "You’re welcome! Anything else I can do?",
      "Happy to help! Need brochures or a sales connect?",
    ],
    bye: [
      "Take care! I’m here if you need me.",
      "Bye! Have a great day.",
    ],
    help: [
      "Ask about flagship lines, suggestions by application, or spares.",
    ],
    ack: [
      "Got it! What would you like next?",
    ],
  };

  const hi = {
    hello: [
      "Namaste 👋 Duki se related kya madad chahiye?",
      "Hello ji 👋 Main madad ke liye hoon—puchhiye.",
    ],
    morning:   ["Good morning! Aaj kis cheez mein help chahiye?"],
    afternoon: ["Good afternoon! Duki ke baare mein kya jaana hai?"],
    evening:   ["Good evening! Machines/spares par madad chahiye to batayein."],
    thanks: [
      "Shukriya! Aur kuch chahiye to pooch lijiye.",
      "Welcome ji! Brochure chahiye ya sales connect karu?",
    ],
    bye: [
      "Theek hai, milte hain! Jab chahein ping kar dijiyega.",
      "Bye! Din shubh rahe.",
    ],
    help: [
      "Try: “Flagship features”, “Application-wise machine suggestion”, “Spares info”.",
    ],
    ack: [
      "Thik hai! Ab kya puchhna hai?",
    ],
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

/* ───────────────────────── Health & Utility APIs ─────────────────────── */
app.get("/api/health", (_, res) => res.json({ ok: true, bot: BOT_NAME, ts: Date.now() }));

app.post("/api/reset", sessionMiddleware, (req, res) => {
  req.session.history = [];
  res.json({ sessionId: req.sid, cleared: true });
});

app.get("/api/session", sessionMiddleware, (req, res) => {
  res.json({
    sessionId: req.sid,
    historyLength: req.session.history.length,
    createdAt: req.session.createdAt,
    lastSeen: req.session.lastSeen,
    hits: req.session.hits,
    bot: BOT_NAME,
  });
});

app.get("/api/history", sessionMiddleware, (req, res) => {
  const n = Math.max(0, Math.min(100, parseInt(req.query.n || "20", 10)));
  const last = req.session.history.slice(-n);
  res.json({ sessionId: req.sid, items: last });
});

/* ───────────────────────────── Ask Endpoint ──────────────────────────── */
app.post("/api/ask", sessionMiddleware, async (req, res) => {
  try {
    const question = (req.body?.question ?? req.body?.message ?? "").toString();
    if (!question || typeof question !== "string") {
      return res.status(400).json({ error: "Missing 'question' (or 'message') string" });
    }

    const q = question.trim();
    const mode = detectResponseMode(q);
    const isFirstTurn = (req.session.history.length === 0);

    // We greet minimally on first user greeting if frontend already introduced the bot.
    const isBlank = q.replace(/[?.!\s]/g, "") === "";
    const isGreetingWord = /^(hi+|hello+|hey( there)?|hlo+|namaste|namaskar|salaam|gm|ga|ge|👋|🙏)$/i.test(q.trim());
    const shouldGreetFirstTurn = isFirstTurn && (isBlank || isGreetingWord);

    // Single-token small talk quick path
    const short = q.toLowerCase().trim().replace(/[^a-z]/g, "");
    const HELLO_SHORT  = new Set(["hi","hey","yo","sup"]);
    const BYE_SHORT    = new Set(["bye","bb","ciao","gn"]);
    const THANKS_SHORT = new Set(["ty","thx","tnx","tx"]);
    const GM_SHORT     = new Set(["gm"]);
    const GA_SHORT     = new Set(["ga"]);
    const GE_SHORT     = new Set(["ge"]);

    const quickKind =
      (HELLO_SHORT.has(short)  && "hello")    ||
      (BYE_SHORT.has(short)    && "bye")      ||
      (THANKS_SHORT.has(short) && "thanks")   ||
      (GM_SHORT.has(short)     && "morning")  ||
      (GA_SHORT.has(short)     && "afternoon")||
      (GE_SHORT.has(short)     && "evening")  ||
      null;

    if (quickKind) {
      let reply;
      if (shouldGreetFirstTurn && FRONTEND_GREETS && ["hello","morning","afternoon","evening"].includes(quickKind)) {
        reply = buildMinimalAssist(mode); // minimal one-liner only
      } else {
        reply = makeSmallTalkReply(quickKind, mode);
      }
      const final = POINTWISE_MODE ? toPointWise(reply) : reply;
      req.session.history.push({ role: "user", content: q, ts: Date.now() });
      req.session.history.push({ role: "assistant", content: final, ts: Date.now() });
      return res.json({ answer: final, reply: final, sessionId: req.sid, mode, citations: [] });
    }

    // Regex small-talk path
    const kind = smallTalkMatch(q);
    if (kind) {
      let reply;
      if (shouldGreetFirstTurn && FRONTEND_GREETS && ["hello","morning","afternoon","evening"].includes(kind)) {
        reply = buildMinimalAssist(mode); // minimal one-liner only
      } else {
        reply = makeSmallTalkReply(kind, mode);
      }
      const final = POINTWISE_MODE ? toPointWise(reply) : reply;
      req.session.history.push({ role: "user", content: q, ts: Date.now() });
      req.session.history.push({ role: "assistant", content: final, ts: Date.now() });
      return res.json({ answer: final, reply: final, sessionId: req.sid, mode, citations: [] });
    }

    // RAG: require vectors
    if (!VECTORS.length) {
      let fallback = mode === "hinglish"
        ? "Reference data abhi load nahi hai. Server par `npm run embed` chalayen, phir dobara poochhiye."
        : "Reference data isn’t loaded yet. Please run `npm run embed` on the server and try again.";
      const final = POINTWISE_MODE ? toPointWise(fallback) : fallback;
      req.session.history.push({ role: "user", content: q, ts: Date.now() });
      req.session.history.push({ role: "assistant", content: final, ts: Date.now() });
      return res.json({ answer: final, reply: final, sessionId: req.sid, mode, citations: [] });
    }

    // Clean query → embed
    const cleanedQuery = cleanForEmbedding(q) || q.toLowerCase();
    const embRes = await embedder.embedContent({ content: { parts: [{ text: cleanedQuery }] } });
    const qVec =
      embRes?.embedding?.values ||
      embRes?.embeddings?.[0]?.values ||
      [];

    // Retrieve top K by cosine
    const scored = VECTORS
      .map((v) => ({ ...v, score: cosineSim(qVec, v.embedding) }))
      .sort((a, b) => b.score - a.score)
      .slice(0, TOP_K);

    const MIN_OK_SCORE = 0.18;
    if (scored.length === 0 || (scored?.[0]?.score ?? 0) < MIN_OK_SCORE) {
      const tip = mode === "hinglish"
        ? "Mujhe is par kaafi specifics nahi mil pa rahe. Kripya thoda specific likhiye—jaise 'Dukejia E+P key features' ya 'Highlead 269 applications'."
        : "I couldn’t find enough details on that. Please try rephrasing or be more specific—like 'Dukejia E+P key features' or 'Highlead 269 applications'.";
      const finalTip = POINTWISE_MODE ? toPointWise(tip) : tip;
      req.session.history.push({ role: "user", content: q, ts: Date.now() });
      req.session.history.push({ role: "assistant", content: finalTip, ts: Date.now() });
      return res.json({ answer: finalTip, reply: finalTip, sessionId: req.sid, mode, citations: [] });
    }

    // Build contextual prompt
    const contextBlocks = scored
      .map((s, i) => `【${i + 1}】 ${s.text_original || s.text_cleaned || s.text}`)
      .join("\n\n");

    const languageGuide =
      mode === "hinglish"
        ? `REPLY LANGUAGE: Hinglish (Hindi in Latin script).`
        : `REPLY LANGUAGE: English. Professional and concise.`;

    const systemInstruction = `
You are ${BOT_NAME}, Dukejia’s assistant. Answer STRICTLY and ONLY from the provided CONTEXT (the Dukejia knowledge base).
If the answer is not present in the CONTEXT, reply exactly:
"Please contact our sales team at 
Whatsapp: +91 9350513789 
Embroidery@grouphca.com"

Rules:
- Do not invent or add external knowledge.
- Be concise and factual.
- ${languageGuide}
`.trim();

    const prompt = `
${systemInstruction}

QUESTION:
${q}

CONTEXT (numbered blocks):
${contextBlocks}

Format:
- Direct answer grounded in context.
- If not found: “Please contact our sales team at 
Whatsapp: +91 9350513789 
Embroidery@grouphca.com
”
- Use the reply language specified above.
`.trim();

    // Call LLM
    req.session.history.push({ role: "user", content: q, ts: Date.now() });
    const result = await llm.generateContent({ contents: [{ role: "user", parts: [{ text: prompt }] }] });
    let text = result.response.text();

    // Server-side bulletization for main answers
    if (POINTWISE_MODE) {
      text = toPointWise(text);
    }

    // IMPORTANT: No greeting prepend on the first turn; UI already greeted
    req.session.history.push({ role: "assistant", content: text, ts: Date.now() });

    res.json({
      answer: text,
      reply: text,
      mode,
      sessionId: req.sid,
      bot: BOT_NAME,
      citations: scored.map((s, i) => ({ idx: i + 1, score: s.score })),
    });
  } catch (err) {
    console.error("Ask error:", err);
    const status = err?.status || 500;
    const msg    = err?.message || err?.statusText || "Generation failed";
    res.status(status).json({
      error: msg,
      details: { status, statusText: err?.statusText || null, type: err?.name || null },
    });
  }
});

/* ───────────────────────────── Start Server ──────────────────────────── */
app.listen(PORT, () => {
  console.log(`${BOT_NAME} running on http://localhost:${PORT}`);
  console.log(` Static UI at http://localhost:${PORT}/`);
});
