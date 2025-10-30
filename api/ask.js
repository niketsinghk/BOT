// api/ask.js — Vercel serverless chatbot for Duke-Jia Assistant (Full version with all tokens + pointwise output)
import fs from "fs";
import path from "path";
import { GoogleGenerativeAI } from "@google/generative-ai";

/* ─────────────────────────── Paths & Config ─────────────────────────── */
const DATA_DIR = path.join(process.cwd(), "data");
const EMB_PATH = path.join(DATA_DIR, "index.json");

const TOP_K = parseInt(process.env.TOP_K || "6", 10);
const GENERATION_MODEL = process.env.GENERATION_MODEL || "gemini-2.5-flash";
const EMBEDDING_MODEL = process.env.EMBEDDING_MODEL || "text-embedding-004";
const MIN_OK_SCORE = 0.18;

const BOT_NAME = process.env.BOT_NAME || "Duki";
const FRONTEND_GREETS = (process.env.FRONTEND_GREETS ?? "true") !== "false";

const CONTACT_WHATSAPP = process.env.CONTACT_WHATSAPP || "+91 93505 13789";
const CONTACT_EMAIL = process.env.CONTACT_EMAIL || "Embroidery@grouphca.com";

if (!process.env.GOOGLE_API_KEY) throw new Error("Missing GOOGLE_API_KEY");

/* ─────────────────────────── Google Gemini SDK ───────────────────────── */
const genAI = new GoogleGenerativeAI(process.env.GOOGLE_API_KEY);
const embedder = genAI.getGenerativeModel({ model: EMBEDDING_MODEL });
const llm = genAI.getGenerativeModel({ model: GENERATION_MODEL });

/* ───────────────────────────── Stopwords & Tokens ─────────────────────────────── */
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
  // Brands / org
  "hari","chand","anand","anil","hca","hari-chand-anand","duke","duke-jia","dukejia","duki",
  "contact","whatsapp","email","brand","website","brochure",

  // Divisions / domain
  "embroidery","quilting","perforation","sewing","pattern","machine","machines","spec","specs",
  "specification","specifications","model","models","single","multi","flagship","application","applications",
  "technical","leather","mattress","garment","automation",

  // Key terms we don’t want removed
  "head","heads","needle","needles","area","mm","configuration","description","details",

  // Models (extend as needed)
  "dy-1201","dy-1201h","dy-1201l","dy-1202","dy-1202h","dy-1202hc","dy-1203h","dy-1204","dy-1206","dy-1206h","dy-1502",
  "dy-908","dy-912","dy-915-120","dy-918-120","dy-606+6","dy-cs3000","dy-pe750x600","dy-sk-d2-2.0rh",
  "pe750x600","dy 1201","dy 1201h","dy 1201l","dy 1206","dy cs3000","dy pe750x600","dy sk d2-2.0rh"
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

/* ───────────────────────────── Utilities ─────────────────────────────── */
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
    if (text.includes(` ${t} `) || text.startsWith(t + " ") || text.endsWith(" " + t)) score += 1;
  }
  const chatCues = (text.match(/[:)(!?]{2,}|\.{3,}|😂|👍|🙏/g) || []).length;
  score += chatCues >= 1 ? 0.5 : 0;
  return score >= 2 ? "hinglish" : "english";
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

let VECTORS = [];
try { VECTORS = loadVectors(); } catch (e) { console.warn(e.message); }

/* ─────────────────────────── Small-talk Logic ─────────────────────────── */
function buildMinimalAssist(mode) {
  return mode === "hinglish" ? "Kaise madad kar sakta hoon?" : "How can I assist you?";
}

function makeSmallTalkReply(kind, mode) {
  const HELP_MESSAGE = `
**Hi! I'm Duki — The Duke-Jia Assistant**

I help you explore our main divisions:
• **Embroidery** — Flagship, Single-Head & Multi-Head  
• **Quilting** — Computerized & Mechanical  
• **Perforation** — Leather / Foam / Technical Fabrics  

**What I can do**
• Recommend machines by your use-case  
• Share model specs & applications  
• Connect you to sales when needed  

**Sales Contact**
WhatsApp: ${CONTACT_WHATSAPP}  
Email: ${CONTACT_EMAIL}
`;

  const en = {
    hello: ["Hi! How can I help today?"],
    help: [HELP_MESSAGE],
    thanks: ["You're welcome! Anything else I can do?"],
    bye: ["Take care! I'm here if you need me."],
  };

  const hi = {
    hello: ["Namaste 👋 Kaise madad kar sakta hoon?"],
    help: [HELP_MESSAGE],
    thanks: ["Shukriya! Aur kuch chahiye to pooch lijiye."],
    bye: ["Theek hai, milte hain! Jab chahein ping kar dijiyega."],
  };

  const bank = mode === "hinglish" ? hi : en;
  return bank[kind]?.[0] || bank.hello[0];
}

function smallTalkMatch(q) {
  const t = (q || "").trim().toLowerCase();
  if (/^(hi|hello|hey|namaste|👋|🙏)$/.test(t)) return "hello";
  if (/thank|thanks|shukriya/.test(t)) return "thanks";
  if (/bye|goodbye|see you/.test(t)) return "bye";
  if (/help|about|what can you do|who are you/.test(t)) return "help";
  return null;
}

function handleSmallTalkAll(q, { isFirstTurn = false } = {}) {
  const kind = smallTalkMatch(q);
  if (!kind) return null;
  const mode = detectResponseMode(q);
  if (isFirstTurn && ["hello"].includes(kind)) return { text: buildMinimalAssist(mode) };
  return { text: makeSmallTalkReply(kind, mode) };
}

/* ─────────────────────── Machine Formatting ─────────────────────── */
function formatMachineResponse(rawText) {
  if (!rawText) return rawText;
  const lower = rawText.toLowerCase();

  const isEmb = /(embroidery|needle|head|color change|stitch)/i.test(lower);
  const isQuilt = /(quilting|pattern|padding|mattress)/i.test(lower);
  const isPerf = /(perforation|punching|leather|hole|stitching)/i.test(lower);

  const modelMatch = rawText.match(/dy[-\s]?[a-z0-9.-]+/i);
  const model = modelMatch ? modelMatch[0].toUpperCase() : "Model N/A";

  const area = rawText.match(/(\d{2,4}\s?[×x]\s?\d{2,4})/);
  const headNeedle = rawText.match(/(\d+)\s?(head|heads).+?(\d+)\s?(needle|needles)/i);

  if (isEmb) {
    return `
**${model} — Embroidery Machine**

**Description:** ${rawText.split(".")[0]}

**Configuration:**
• ${headNeedle ? `${headNeedle[1]} Heads, ${headNeedle[3]} Needles` : "Head & needle info on request"}  
• ${area ? `Embroidery Area: ${area[0]} mm` : "Embroidery area available on request"}

**Next:**
• Say “show features” or “full specs” for details
`;
  }

  if (isQuilt) {
    return `
**${model} — Quilting Machine**

**Description:** ${rawText.split(".")[0]}

**Application:**
• Pattern stitching for quilts, mattress covers, and layered fabrics

**Next:**
• Say “show features” or “full specs” for details
`;
  }

  if (isPerf) {
    return `
**${model} — Perforation / Stitching Machine**

**Description:** ${rawText.split(".")[0]}

**Application:**
• Leather, foam, or technical textiles needing punching + stitching precision

**Next:**
• Say “show features” or “full specs” for details
`;
  }

  return rawText;
}

/* ────────────────── Point-wise Output Normalizer ────────────────── */
function hasList(text) {
  return /(^|\n)\s*(?:•|-|\d+\.)\s+/.test(text);
}

function extractContact(text) {
  const lines = text.split(/\r?\n/);
  const contact = [];
  const rest = [];
  for (const ln of lines) {
    if (/whatsapp\s*:|email\s*:/i.test(ln)) contact.push(ln);
    else rest.push(ln);
  }
  return { rest: rest.join("\n").trim(), contact: contact.join("\n").trim() };
}

function toBullets(plain) {
  // Split into sentences conservatively; keep 4–8 meaningful points
  const sentences = plain
    .replace(/\r?\n+/g, " ")
    .split(/(?<=[.!?])\s+(?=[A-Z(“"']|\d)/)
    .map(s => s.trim())
    .filter(s => s && !/^[-•\d.]/.test(s))
    .slice(0, 8);
  if (!sentences.length) return plain;
  return sentences.map(s => `• ${s}`).join("\n");
}

/**
 * If the text is a simple paragraph (not already a list or machine block),
 * convert it into clean bullet points. Preserve any contact lines at the end.
 */
function enforcePointwise(text) {
  if (!text || hasList(text)) return text;

  // Keep machine blocks (they already have sections)
  if (/^\s*[<>]/.test(text)) return text;

  // Keep pure contact-only replies
  const contactOnly = /^please contact/i.test(text.trim());
  if (contactOnly) return text;

  const { rest, contact } = extractContact(text);
  const bullets = toBullets(rest);
  return contact ? `${bullets}\n\n${contact}` : bullets;
}

/* ───────────────────────────── Handler ───────────────────────────────── */
export default async function handler(req, res) {
  // CORS preflight
  if (req.method === "OPTIONS") {
    res.setHeader("Access-Control-Allow-Origin", req.headers.origin || "*");
    res.setHeader("Access-Control-Allow-Methods", "POST, OPTIONS");
    res.setHeader("Access-Control-Allow-Headers", "Content-Type, X-Session-ID");
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
    const body = typeof req.body === "string" ? JSON.parse(req.body) : (req.body || {});
    const q = (body.message ?? body.question ?? "").toString().trim();
    const isFirstTurn = !!body.isFirstTurn;
    const pageHints = body.pageHints || {};

    if (!q && !isFirstTurn)
      return res.status(400).json({ error: "Missing 'message' or 'question'." });

    // Small talk
    const st = handleSmallTalkAll(q, { isFirstTurn });
    if (st && st.text)
      return res.status(200).json({ answer: st.text, citations: [], mode: detectResponseMode(q || ""), bot: BOT_NAME });

    // RAG ready?
    if (!VECTORS.length)
      return res.status(500).json({ error: "Embeddings not loaded. Add data/index.json and redeploy." });

    const mode = detectResponseMode(q);
    const cleaned = cleanForEmbedding(q);
    const embRes = await embedder.embedContent({ content: { parts: [{ text: cleaned }] } });
    const qVec = embRes?.embedding?.values || embRes?.embeddings?.[0]?.values || [];
    if (!qVec.length) return res.status(500).json({ error: "Embedding failed" });

    const top = VECTORS.map(v => ({ ...v, score: cosineSim(qVec, v.embedding) }))
      .sort((a, b) => b.score - a.score)
      .slice(0, TOP_K);

    if (!top.length || (top[0].score ?? 0) < MIN_OK_SCORE) {
      const tip = mode === "hinglish"
        ? "Is topic par Duke-Jia knowledge base mein clear info nahi mil rahi. Thoda specific likhiye—model number ya segment (Flagship/Single/Multi/Quilting/Perforation)."
        : "I couldn’t find clear info in the Duke-Jia knowledge base. Please be specific—e.g., a model name or segment (Flagship/Single/Multi/Quilting/Perforation).";
      return res.status(200).json({ answer: tip, citations: [], mode, bot: BOT_NAME });
    }

    const context = top.map((s, i) => `【${i + 1}】 ${s.text_original || s.text_cleaned || s.text}`).join("\n\n");
    const seg = (pageHints.segment || "").toLowerCase();

    const systemInstruction = `
You are ${BOT_NAME}, the Duke-Jia Assistant (Hari Chand Anand & Co.).
Help users understand Duke-Jia machines across 3 divisions:
1) Embroidery   2) Quilting   3) Perforation

${seg === "embroidery" ? "Focus on embroidery (Flagship, Single-head, Multi-head)." :
seg === "quilting" ? "Focus on quilting machines and stitching applications." :
seg === "perforation" ? "Focus on perforation & punching machines for leather or foam." :
"Respond only from Duke-Jia context."}

STRICT RULES:
- Use ONLY the CONTEXT. If info is missing/unclear, reply exactly:
  "Please contact our sales team at
  WhatsApp: ${CONTACT_WHATSAPP}
  Email: ${CONTACT_EMAIL}"
- Be concise and factual (sales-engineer tone).
- Prefer short lines or bullet points when listing information.
`.trim();

    const prompt = `
${systemInstruction}

QUESTION:
${q}

CONTEXT (knowledge base excerpts):
${context}

Output:
- Direct, grounded answer. No external info or guesses.
- Keep it readable; lists and short lines preferred.
`.trim();

    const result = await llm.generateContent({ contents: [{ role: "user", parts: [{ text: prompt }] }] });
    let text = result?.response?.text?.() || `Please contact our sales team at
WhatsApp: ${CONTACT_WHATSAPP}
Email: ${CONTACT_EMAIL}`;

    // Prefer machine block if detectable, then normalize other replies to point-wise
    text = formatMachineResponse(text);
    text = enforcePointwise(text);

    return res.status(200).json({
      answer: text,
      mode,
      bot: BOT_NAME,
      citations: top.map((s, i) => ({ idx: i + 1, score: s.score })),
    });
  } catch (err) {
    console.error("ask error:", err);
    return res.status(err?.status || 500).json({
      error: err?.message || "Server error",
      details: { status: err?.status || 500, type: err?.name || null },
    });
  }
}
