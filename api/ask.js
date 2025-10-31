// api/ask.js — contact intent + category intent (lexical fallback) + optional Redis + dynamic-entity hybrid RAG + strict fallback + resilient Gemini retries
import fs from "fs";
import path from "path";
import { GoogleGenerativeAI } from "@google/generative-ai";

/* ─────────────────────────── Paths & Config ─────────────────────────── */
const DATA_DIR = path.join(process.cwd(), "data");
const EMB_PATH = path.join(DATA_DIR, "index.json");

const TOP_K            = parseInt(process.env.TOP_K || "6", 10);
const GENERATION_MODEL = process.env.GENERATION_MODEL || "gemini-2.5-flash";
const FALLBACK_MODEL   = process.env.FALLBACK_MODEL   || "gemini-1.5-flash";
const EMBEDDING_MODEL  = process.env.EMBEDDING_MODEL  || "text-embedding-004";
const MIN_OK_SCORE     = parseFloat(process.env.MIN_OK_SCORE || "0.16");

const BOT_NAME        = process.env.BOT_NAME || "Duki";
const FRONTEND_GREETS = (process.env.FRONTEND_GREETS ?? "true") !== "false";

// Contact fallbacks
const CONTACT_WHATSAPP = process.env.CONTACT_WHATSAPP || "+91 9350513789";
const CONTACT_EMAIL    = process.env.CONTACT_EMAIL    || "Embroidery@grouphca.com";

if (!process.env.GOOGLE_API_KEY) throw new Error("Missing GOOGLE_API_KEY env on Vercel");

/* ─────────────────────────── Google Gemini SDK ───────────────────────── */
const genAI    = new GoogleGenerativeAI(process.env.GOOGLE_API_KEY);
let   llm      = genAI.getGenerativeModel({ model: GENERATION_MODEL });
const embedder = genAI.getGenerativeModel({ model: EMBEDDING_MODEL });

/* ───────── Optional Redis (lazy import so missing pkg won’t crash) ───── */
const SESSION_TTL = parseInt(process.env.SESSION_TTL_SECONDS || "3600", 10);
const WANT_REDIS  = !!(process.env.UPSTASH_REDIS_REST_URL && process.env.UPSTASH_REDIS_REST_TOKEN);
let redisClient = null;

async function getRedis() {
  if (!WANT_REDIS) return null;
  if (redisClient) return redisClient;
  try {
    const mod = await import("@upstash/redis");
    redisClient = new mod.Redis({
      url: process.env.UPSTASH_REDIS_REST_URL,
      token: process.env.UPSTASH_REDIS_REST_TOKEN,
    });
    return redisClient;
  } catch (e) {
    console.warn("Redis not available, running without memory:", e?.message || e);
    return null;
  }
}

function getSessionId(req) {
  const h = req.headers["x-session-id"] || req.headers["x-sessionid"] || req.headers["x-client-session"];
  if (h) return String(h).slice(0, 100);
  const ck = (req.headers.cookie || "").match(/sid=([^;]+)/)?.[1];
  if (ck) return ck.slice(0, 100);
  return (req.headers["x-forwarded-for"] || req.socket?.remoteAddress || "anon") + ":" + (req.headers["user-agent"] || "");
}
async function loadHistory(sessionId, k = 10) {
  const redis = await getRedis();
  if (!redis) return [];
  const key = `duki:chat:${sessionId}`;
  const rows = await redis.lrange(key, -k, -1);
  return rows?.map(r => JSON.parse(r)) || [];
}
async function saveTurn(sessionId, role, text) {
  const redis = await getRedis();
  if (!redis) return;
  const key = `duki:chat:${sessionId}`;
  await redis.rpush(key, JSON.stringify({ ts: Date.now(), role, text }));
  await redis.expire(key, SESSION_TTL);
}

/* ───────────────────────────── Text Utils ────────────────────────────── */
function detectResponseMode(q = "") {
  const text = q.toLowerCase();
  if (/[\u0900-\u097F]/.test(text)) return "hinglish";
  const toks = ["hai","hain","tha","thi","the","kya","kyun","kyunki","kisi","kis","kaun","kab","kaha","kahaan","kaise","nahi","nahin","ka","ki","ke","mein","me","mai","hum","aap","kr","kar","chahiye","bhi","sirf","jaldi","kitna","hoga","hogaya","pls","plz","yaar","shukriya","dhanyavaad"];
  let score = 0;
  for (const t of toks) if (text.includes(` ${t} `) || text.startsWith(t+" ") || text.endsWith(" "+t) || text===t) score++;
  const cues = (text.match(/[:)(!?]{2,}|\.{3,}|😂|👍|🙏/g) || []).length;
  if (cues) score += 0.5;
  return score >= 2 ? "hinglish" : "english";
}
const EN_STOP = new Set(`a about above after again against all am an and any are aren't as at
be because been before being below between both but by
can't cannot could couldn't did didn't do does doesn't doing don't down during
each few for from further had hadn't has hasn't have haven't having he he'd he'll he's her here here's hers herself him himself his how how's
i i'd i'll i'm i've if in into is isn't it it's its itself let's me more most mustn't my myself
no nor not of off on once only or other ought our ours ourselves out over own
same shan't she she'd she'll she's should shouldn't so some such than that that's the their theirs them themselves then there there's these they they'd they'll they're they've this those through to too
under until up very was wasn't we we'd we'll we're we've were weren't what what's when when's where where's which while who who's whom why why's with won't would wouldn't
you you'd you'll you're you've your yours yourself yourselves`.trim().split(/\s+/));
const PROTECTED_TOKENS = new Set(["hari","chand","anand","anil","hca","duke","duke-jia","dukejia","duki","contact","email","address","website","whatsapp","brand","features","specification","model","models","application","id","delhi","india","automation","garment","leather","mattress","perforation","embroidery","quilting","sewing","upholstery","pattern","sequin","bead","cording","coiling","taping","rhinestone","chenille","chainstitch","cap","tubular","dahao","a18","dst","tajima","usb","lcd","touchscreen","network","auto-trimming","auto-color-change","thread-break-detection","power-failure-recovery","servo","oil-mist","dust-clean","wide-voltage","270-cap-frame"]);
function cleanForEmbedding(s="") {
  const lower = s.toLowerCase();
  const stripped = lower.replace(/[^a-z0-9\u0900-\u097F\s-]/g, " ");
  return stripped.split(/\s+/).filter(Boolean).filter(t => PROTECTED_TOKENS.has(t) || !EN_STOP.has(t)).join(" ").trim();
}
function cosineSim(a,b){let d=0,na=0,nb=0;const n=Math.min(a.length,b.length);for(let i=0;i<n;i++){d+=a[i]*b[i];na+=a[i]*a[i];nb+=b[i]*b[i]}return d/(Math.sqrt(na)*Math.sqrt(nb)||1);}

/* ─────────────────────── Load KB Vectors ─────────────────────────────── */
function loadVectors() {
  if (!fs.existsSync(EMB_PATH)) throw new Error(`Embeddings not found at ${EMB_PATH}.`);
  const raw = JSON.parse(fs.readFileSync(EMB_PATH, "utf8"));
  if (!raw?.vectors?.length) throw new Error("Embeddings file has no vectors.");
  return raw.vectors;
}
let VECTORS=[]; try{ VECTORS = loadVectors(); }catch(e){ console.warn(e.message); }

/* ───────── KB Contact Extraction (once at cold start) ───────── */
const PHONE_RE    = /(\+?\d[\d\s-]{7,}\d)/g;
const EMAIL_RE    = /[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}/gi;
const ADDRESS_HINTS = /(head\s*office|office|branch|showroom|address|factory|works|warehouse|hq|headquarters)[:\-]?\s*/i;
const CONTACT_CACHE = { whatsapp: null, phone: null, email: null, address: null };

(function initContactsFromKB() {
  try {
    for (const v of VECTORS || []) {
      const t = String(v.text_original || v.text_cleaned || v.text || "");
      if (!t) continue;
      if (!CONTACT_CACHE.email)    { const em = t.match(EMAIL_RE); if (em?.length) CONTACT_CACHE.email = em[0]; }
      if (!CONTACT_CACHE.phone || !CONTACT_CACHE.whatsapp) {
        const ph = t.match(PHONE_RE);
        if (ph?.length) {
          const first = ph[0].replace(/\s+/g, " ").trim();
          if (!CONTACT_CACHE.phone)    CONTACT_CACHE.phone = first;
          if (!CONTACT_CACHE.whatsapp) CONTACT_CACHE.whatsapp = first;
        }
      }
      if (!CONTACT_CACHE.address) {
        const line = t.split(/\r?\n/).find(l => ADDRESS_HINTS.test(l));
        if (line) CONTACT_CACHE.address = line.trim();
      }
      if (CONTACT_CACHE.email && CONTACT_CACHE.whatsapp && CONTACT_CACHE.address) break;
    }
  } catch(e){ console.warn("KB contact parse error:", e?.message || e); }
  if (!CONTACT_CACHE.whatsapp) CONTACT_CACHE.whatsapp = CONTACT_WHATSAPP;
  if (!CONTACT_CACHE.phone)    CONTACT_CACHE.phone    = CONTACT_PHONE;
  if (!CONTACT_CACHE.email)    CONTACT_CACHE.email    = CONTACT_EMAIL;
  if (!CONTACT_CACHE.address)  CONTACT_CACHE.address  = CONTACT_HO_ADDR;
})();

/* ────────── Contact Intent Detector (explicit requests only) ────────── */
function isContactIntent(q = "") {
  const t = q.toLowerCase();
  return /\b(contact|contact\s*us|phone|call|whats\s*app|whatsapp|mail|email|address|location|where|service|support|sales|helpdesk|showroom|branch|head\s*office|office)\b/.test(t);
}
function contactReply(mode = "english") {
  const lines = [
    "Here are our contact details:",
    `• WhatsApp: ${CONTACT_CACHE.whatsapp}`,
    `• Email: ${CONTACT_CACHE.email}`,
  ];
  if (mode === "hinglish") {
    return [
      "Yeh rahe hamare contact details:",
      `• WhatsApp: ${CONTACT_CACHE.whatsapp}`,
      `• Email: ${CONTACT_CACHE.email}`,
    ].join("\n");
  }
  return lines.join("\n");
}

/* ───────────── Entity & Hybrid Retrieval (dynamic, KB-driven) ────────── */
let MODEL_SET = new Set();
let MODEL_ALIASES = new Map();
let MODEL_INIT_DONE = false;
function normModelToken(s=""){return s.toLowerCase().replace(/\s+/g,"").replace(/_/g,"-").replace(/–/g,"-").replace(/[()]+/g,"").replace(/[^a-z0-9.+-]/g,"").replace(/^-+|-+$/g,"");}
const GENERIC_MODEL_RE = /\b([a-z]{1,6}[\w.+-]*\d[\w.+-]*|[a-z]*\d[\w.+-]{1,15})\b/gi;
const LIKELY_PREFIX = /^(dy|dukejia|duke|cs|sk|pe|halo|es|tajima|highlead|juki|dyk|dy\-|dy_)/i;

function initModelIndexOnce(){
  if (MODEL_INIT_DONE) return; MODEL_INIT_DONE = true;
  const localSet=new Set(); const aliasMap=new Map();
  for (const v of VECTORS||[]) {
    const txt = String(v.text_original || v.text_cleaned || v.text || "");
    if (!txt) continue;
    const metaModel = v.model || v.meta?.model || v.meta?.Model || null;
    if (metaModel) {
      const n = normModelToken(String(metaModel));
      if (n && /\d/.test(n)) { localSet.add(n); if(!aliasMap.has(n)) aliasMap.set(n,new Set([n])); }
    }
    const seenHere = new Set(); let m;
    while ((m = GENERIC_MODEL_RE.exec(txt)) !== null) {
      const raw = m[1]; if (!raw) continue;
      if (!/\d/.test(raw)) continue; if (raw.length<3 || raw.length>24) continue;
      const n = normModelToken(raw); if (!n || n.length<3) continue;
      const good = LIKELY_PREFIX.test(raw) || /[-+.]/.test(raw) || /^[a-z]+\d/.test(raw);
      if (!good) continue; seenHere.add(n);
    }
    for (const n of seenHere) { localSet.add(n); if(!aliasMap.has(n)) aliasMap.set(n,new Set()); aliasMap.get(n).add(n); }
    for (const n of seenHere) {
      const variants = new Set([n, n.replace(/-/g,""), n.replace(/\./g,""), n.replace(/\+/g,""), n.replace(/-/g," ").replace(/\./g," ").replace(/\+/g," ")]);
      const bucket = aliasMap.get(n) || new Set(); for (const v of variants) bucket.add(v); aliasMap.set(n,bucket);
    }
  }
  MODEL_SET=localSet; MODEL_ALIASES=aliasMap;
}
function extractEntities(text="", history=[]){
  initModelIndexOnce();
  const combined=[text,...history.map(h=>h.text||"")].join(" ");
  const found=new Set();
  if (MODEL_SET.size>0){
    const lower=combined.toLowerCase();
    for (const canonical of MODEL_SET){
      const aliases = MODEL_ALIASES.get(canonical) || new Set([canonical]);
      for (const alias of aliases){
        if (!alias || alias.length<3) continue;
        if (lower.includes(alias.toLowerCase())) { found.add(canonical); break; }
      }
    }
  }
  let m; const unseen=new Set(); GENERIC_MODEL_RE.lastIndex=0;
  while ((m = GENERIC_MODEL_RE.exec(combined)) !== null) {
    const raw=m[1]; if (!raw) continue; if (raw.length<3 || raw.length>24) continue; if (!/\d/.test(raw)) continue;
    const n=normModelToken(raw); if (!n) continue; if (/^v\d{1,2}$/.test(n)) continue;
    if (LIKELY_PREFIX.test(raw) || /[-+.]/.test(raw) || /^[a-z]+\d/.test(raw)) unseen.add(n);
  }
  for (const x of unseen) found.add(x);
  return [...found];
}
function kwScoreFor(v, kwList){
  if (!kwList?.length) return 0;
  const t=(v.text_original||v.text_cleaned||v.text||"").toLowerCase(); let s=0;
  const bag=new Set();
  for (const kw of kwList){
    const n=normModelToken(kw); bag.add(n); bag.add(n.replace(/-/g,"")); bag.add(n.replace(/\./g,"")); bag.add(n.replace(/\+/g,"")); bag.add(n.replace(/-/g," ").replace(/\./g," ").replace(/\+/g," "));
    if (MODEL_ALIASES.has(n)) for (const al of MODEL_ALIASES.get(n)) bag.add(al.toLowerCase());
  }
  for (const alias of bag){ if (!alias || alias.length<3) continue; if (t.includes(alias)) s++; }
  return s;
}

/* ───────────── Category Intent + Lexical Fallback (for broad asks) ───── */
const CATEGORY_DEFS = {
  embroidery: {
    kw: ["embroidery", "embroidery machine", "single head", "multi head", "dukejia", "dy-"],
    promptHint: "Embroidery machines overview, use specs/features if present.",
  },
  perforation: {
    kw: ["perforation", "leather punching", "punching", "pe750", "pe750x600", "punch"],
    promptHint: "Leather perforation/punching machines overview.",
  },
  quilting: {
    kw: ["quilting", "cs3000", "quilting machine", "mattress", "quilt"],
    promptHint: "Quilting machines overview (CS3000 etc.).",
  },
  pattern: {
    kw: ["pattern sewing", "programmable", "pattern", "tacking", "bartack"],
    promptHint: "Programmable pattern sewing overview.",
  },
};

function detectCategoryIntent(q = "") {
  const t = q.toLowerCase();
  for (const [cat, def] of Object.entries(CATEGORY_DEFS)) {
    if (def.kw.some(k => t.includes(k))) return cat;
  }
  // very broad intents
  if (/^what\s+is\s+embroidery( machine)?\b/i.test(q)) return "embroidery";
  if (/^i\s+want\s+embroidery( machine)?\b/i.test(q))   return "embroidery";
  return null;
}

// simple lexical scorer (BM25-lite)
function lexicalTop(vectors, keywords, k = 8) {
  const toks = keywords
    .flatMap(w => [w, w.replace(/\s+/g, "-"), w.replace(/\s+/g, "")])
    .map(w => w.toLowerCase());
  const scored = [];
  for (const v of vectors) {
    const txt = String(v.text_original || v.text_cleaned || v.text || "").toLowerCase();
    if (!txt) continue;
    let s = 0;
    for (const w of toks) {
      if (!w || w.length < 3) continue;
      const m = txt.match(new RegExp(`\\b${escapeRegExp(w)}\\b`, "g"));
      s += (m?.length || 0);
      // also partial contains bonus
      if (!m?.length && txt.includes(w)) s += 0.3;
    }
    if (s > 0) scored.push({ ...v, lex: s, score: s });
  }
  return scored.sort((a, b) => b.score - a.score).slice(0, k);
}
function escapeRegExp(s){return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");}

/* ─────────────────────── Small-talk (same as before) ─────────────────── */
function getISTGreeting(now=new Date()){
  const hour=Number(new Intl.DateTimeFormat("en-GB",{timeZone:"Asia/Kolkata",hour:"2-digit",hour12:false}).format(now));
  if (hour<5) return "Good night"; if (hour<12) return "Good morning"; if (hour<17) return "Good afternoon"; if (hour<21) return "Good evening"; return "Good night";
}
function buildMinimalAssist(mode){return mode==="hinglish"?"Kaise madad kar sakta hoon?":"How can I assist you?";}
function makeSmallTalkReply(kind,mode){
  const en={hello:["Hi! How can I help today?","How can I help you?"],morning:["Good morning! How can I help today?"],afternoon:["Good afternoon! How can I help today?"],evening:["Good evening! Need help with machines or spares?"],thanks:["You’re welcome! Anything else I can do?","Happy to help! Need brochures or a sales connect?"],bye:["Take care! I’m here if you need me.","Bye! Have a great day."],help:["Ask about flagship lines, suggestions by application, or spares."],ack:["Got it! What would you like next?"]};
  const hi={hello:["Namaste 👋 Duki se related kya madad chahiye?","Hello ji 👋 Main madad ke liye hoon—puchhiye."],morning:["Good morning! Aaj kis cheez mein help chahiye?"],afternoon:["Good afternoon! Duki ke baare mein kya jaana hai?"],evening:["Good evening! Machines/spares par madad chahiye to batayein."],thanks:["Shukriya! Aur kuch chahiye to pooch lijiye.","Welcome ji! Brochure chahiye ya sales connect karu?"],bye:["Theek hai, milte hain! Jab chahein ping kar dijiyega.","Bye! Din shubh rahe."],help:["Try: “Flagship features”, “Application-wise machine suggestion”, “Spares info”."],ack:["Thik hai! Ab kya puchhna hai?"]};
  const bank=mode==="hinglish"?hi:en; const pick=a=>a[Math.floor(Math.random()*a.length)];
  switch(kind){case"hello":return pick(bank.hello);case"morning":return pick(bank.morning);case"afternoon":return pick(bank.afternoon);case"evening":return pick(bank.evening);case"thanks":return pick(bank.thanks);case"bye":return pick(bank.bye);case"help":return pick(bank.help);case"ack":return pick(bank.ack);default:return pick(bank.hello);}
}
function smallTalkMatch(q){
  const t=(q||"").trim(); const p=[{kind:"hello",re:/^(hi+|h[iy]+|hello+|hey( there)?|hlo+|yo+|hola|namaste|namaskar|salaam|salam|👋|🙏)\b/i},{kind:"morning",re:/^(good\s*morning|gm)\b/i},{kind:"afternoon",re:/^(good\s*afternoon|ga)\b/i},{kind:"evening",re:/^(good\s*evening|ge)\b/i},{kind:"ack",re:/^(ok+|okay+|okk+|hmm+|haan+|ha+|sure|done|great|nice|cool|perfect|thik|theek|fine)\b/i},{kind:"thanks",re:/^(thanks|thank\s*you|thx|tnx|ty|much\s*(appreciated|thanks)|appreciate(d)?|shukriya|dhanyavaad|dhanyavad)\b/i},{kind:"bye",re:/^(bye|bb|good\s*bye|goodbye|see\s*ya|see\s*you|take\s*care|tc|ciao|gn)\b/i},{kind:"help",re:/(who\s*are\s*you|what\s*can\s*you\s*do|help|menu|options|how\s*to\s*use)\b/i}];
  for (const r of p) if (r.re.test(t)) return r.kind; return null;
}
function handleSmallTalkAll(q,{isFirstTurn=false}={}){
  if(!q) return null; const mode=detectResponseMode(q); const trimmed=q.trim();
  const short=trimmed.toLowerCase().replace(/[^a-z]/g,""); const HELLO=new Set(["hi","hey","yo","sup"]); const BYE=new Set(["bye","bb","ciao","gn"]); const THANKS=new Set(["ty","thx","tnx","tx"]); const GM=new Set(["gm"]); const GA=new Set(["ga"]); const GE=new Set(["ge"]);
  let quick=(HELLO.has(short)&&"hello")||(BYE.has(short)&&"bye")||(THANKS.has(short)&&"thanks")||(GM.has(short)&&"morning")||(GA.has(short)&&"afternoon")||(GE.has(short)&&"evening")||null;
  const isBlank=trimmed.replace(/[?.!\s]/g,"")===""; const isGreeting=/^(hi+|hello+|hey( there)?|hlo+|namaste|namaskar|salaam|gm|ga|ge|👋|🙏)$/i.test(trimmed);
  if (quick){ if (isFirstTurn && FRONTEND_GREETS && ["hello","morning","afternoon","evening"].includes(quick)) return {text:buildMinimalAssist(mode)}; return {text:makeSmallTalkReply(quick,mode)}; }
  const kind=smallTalkMatch(trimmed);
  if (kind){ if (isFirstTurn && FRONTEND_GREETS && ["hello","morning","afternoon","evening"].includes(kind)) return {text:buildMinimalAssist(mode)};
    if (["morning","afternoon","evening"].includes(kind)){ const sal=getISTGreeting(); const line=mode==="hinglish"?(kind==="morning"?"Good morning! Aaj kis cheez mein help chahiye?":kind==="afternoon"?"Good afternoon! Duki ke baare mein kya jaana hai?":"Good evening! Machines/spares par madad chahiye to batayein."):`${sal}! How can I help today?`; return {text:line}; }
    return {text:makeSmallTalkReply(kind,mode)};
  }
  if (isFirstTurn && FRONTEND_GREETS && (isBlank || isGreeting)) return {text:buildMinimalAssist(mode)};
  return null;
}

/* ──────────────── Gemini guarded call (retry + model fallback) ─────────────── */
async function tryGeminiText(prompt, opts = {}) {
  const {
    retries = 2,
    backoffMs = 1200,
    modelFallback = FALLBACK_MODEL
  } = opts;

  let lastErr;
  for (let i = 0; i < Math.max(1, retries); i++) {
    try {
      const res = await llm.generateContent({ contents: [{ role: "user", parts: [{ text: prompt }] }] });
      const txt = res?.response?.text?.() || "";
      if (txt) return txt;
      throw new Error("Empty response from model");
    } catch (e) {
      lastErr = e;
      const msg = String(e?.message || e);
      const isOverload = msg.includes("503") || msg.toLowerCase().includes("overloaded") || msg.toLowerCase().includes("service unavailable");
      const isQuota    = msg.toLowerCase().includes("quota") || msg.toLowerCase().includes("api key");

      if (i === 0 && modelFallback && GENERATION_MODEL !== modelFallback) {
        console.warn("Switching to fallback model:", modelFallback);
        llm = genAI.getGenerativeModel({ model: modelFallback });
        continue;
      }
      if (isOverload && i < retries - 1) {
        const wait = backoffMs * (i + 1);
        console.warn(`Gemini overloaded — retrying in ${wait}ms`);
        await new Promise(r => setTimeout(r, wait));
        continue;
      }
      if (isQuota) return "There seems to be a temporary issue with my AI engine. Please try again shortly.";
      if (isOverload) return "Server is a bit busy right now ⏳ — please try again in a few seconds.";
      return "Sorry, I ran into a technical issue fetching that. Please try again.";
    }
  }
  console.error("Gemini final error:", lastErr?.message || lastErr);
  return "Sorry, I ran into a technical issue fetching that. Please try again.";
}

/* ───────────────────────────── Handler ───────────────────────────────── */
export default async function handler(req, res) {
  if (req.method === "OPTIONS") {
    res.setHeader("Access-Control-Allow-Origin", req.headers.origin || "*");
    res.setHeader("Access-Control-Allow-Methods", "POST, OPTIONS");
    res.setHeader("Access-Control-Allow-Headers", "Content-Type, X-Session-ID, X-SessionID, X-Client-Session");
    res.setHeader("Access-Control-Allow-Credentials", "true");
    return res.status(204).end();
  }
  if (req.method !== "POST") {
    res.setHeader("Allow", ["POST","OPTIONS"]);
    return res.status(405).json({ error: "Method Not Allowed" });
  }
  res.setHeader("Access-Control-Allow-Origin", req.headers.origin || "*");
  res.setHeader("Access-Control-Allow-Credentials", "true");

  try {
    const body = typeof req.body === "string" ? JSON.parse(req.body) : (req.body || {});
    const q = (body.message ?? body.question ?? "").toString().trim();
    const isFirstTurn = !!body.isFirstTurn;
    if (!q && !isFirstTurn) return res.status(400).json({ error: "Missing 'message' or 'question'." });

    const sessionId = getSessionId(req);
    const history = await loadHistory(sessionId, 10);

    // 0) Small talk
    const st = handleSmallTalkAll(q, { isFirstTurn });
    if (st && st.text) {
      await saveTurn(sessionId, "user", q || "");
      await saveTurn(sessionId, "assistant", st.text);
      return res.status(200).json({ answer: st.text, citations: [], mode: detectResponseMode(q || ""), bot: BOT_NAME });
    }

    const mode = detectResponseMode(q);

    // 1) Explicit CONTACT intent → reply immediately (no LLM)
    if (isContactIntent(q)) {
      const msg = contactReply(mode);
      await saveTurn(sessionId, "user", q || "");
      await saveTurn(sessionId, "assistant", msg);
      return res.status(200).json({ answer: msg, citations: [], mode, bot: BOT_NAME });
    }

    // 2) Category intent (broad ask) → lexical fallback context first
    const cat = detectCategoryIntent(q);

    if (!VECTORS.length) {
      const msg = "Embeddings not loaded on server. Add data/index.json (npm run embed) and redeploy.";
      await saveTurn(sessionId, "user", q || ""); await saveTurn(sessionId, "assistant", msg);
      return res.status(500).json({ error: msg });
    }

    const cleaned = cleanForEmbedding(q) || q.toLowerCase();
    const stickyEntities = extractEntities(q, history);
    const kwList = [...stickyEntities];

    // Embedding guard
    let qVec = [];
    try {
      const embRes = await embedder.embedContent({ content: { parts: [{ text: cleaned }] } });
      qVec = embRes?.embedding?.values || embRes?.embeddings?.[0]?.values || [];
    } catch (e) {
      console.error("Embedding error:", e?.message || e);
    }

    // If we have a category intent, build a lexical candidate set right away.
    let catCandidates = [];
    if (cat) {
      catCandidates = lexicalTop(VECTORS, CATEGORY_DEFS[cat].kw, TOP_K * 2);
    }

    // Hybrid retrieval (cosine + entity bonus)
    let hybridCandidates = [];
    if (qVec.length) {
      const HYBRID_BONUS = 0.12;
      hybridCandidates = VECTORS.map(v => {
        const cos = cosineSim(qVec, v.embedding);
        const kw  = kwScoreFor(v, kwList);
        const score = cos + (kw > 0 ? HYBRID_BONUS * Math.min(kw, 3) : 0);
        return { ...v, score, cos, kw };
      }).sort((a,b)=>b.score-a.score).slice(0, TOP_K);
    }

    // Merge candidates: prefer hybrid if strong, otherwise lexical
    let candidates = hybridCandidates;
    const topHit = candidates[0];
    const modelFoundInTop = stickyEntities.length>0 && candidates.some(c => {
      const txt=(c.text_original||c.text_cleaned||c.text||"").toLowerCase();
      return stickyEntities.some(m => txt.includes(m));
    });
    let passable = (topHit?.score ?? 0) >= (MIN_OK_SCORE - 0.04) || modelFoundInTop;

    if (!passable && catCandidates.length) {
      candidates = catCandidates.slice(0, TOP_K);
      passable = candidates.length > 0;
    }

    if (!passable) {
      const tip = mode === "hinglish"
        ? "Is topic par Dukejia knowledge base mein clear info nahi mil rahi. Thoda specific likhiye—jaise 'DY-CS3000 specs' ya '1206H applications'."
        : "I couldn’t find clear context in the Dukejia knowledge base for that. Try being more specific—for example, 'DY-CS3000 specs' or '1206H applications'.";
      await saveTurn(sessionId,"user",q||""); await saveTurn(sessionId,"assistant",tip);
      return res.status(200).json({ answer: tip, citations: [], mode, bot: BOT_NAME });
    }

    // Build context + recent chat
    const context = candidates.map((s,i)=>`【${i+1}】 ${s.text_original || s.text_cleaned || s.text}`).join("\n\n");
    const recentChat = history.slice(-6).map(h => (h.role==="user"?`User: ${h.text}`:`Assistant: ${h.text}`)).join("\n");

    const languageGuide = mode==="hinglish" ? `REPLY LANGUAGE: Hinglish (Hindi in Latin script). Do NOT use Devanagari.` : `REPLY LANGUAGE: English. Professional and concise.`;

    const catHint = cat ? CATEGORY_DEFS[cat].promptHint : "";
    const systemInstruction = `
You are ${BOT_NAME}, Dukejia’s assistant.
Answer STRICTLY and ONLY from the provided CONTEXT.
If the requested details are clearly NOT present in CONTEXT, reply exactly:
"Please contact our sales team at 
Whatsapp: ${CONTACT_CACHE.whatsapp} 
${CONTACT_CACHE.email}"
Rules:
- Do not invent or add external knowledge.
- Be concise and factual.
- Use recent chat for continuity if it helps resolve the user’s intent.
- Do NOT include contact details unless the answer is not present in CONTEXT (explicit contact requests are handled before this step).
- ${languageGuide}
${cat ? `- The user intent category is **${cat}**. Hint: ${catHint}` : ""}
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
Whatsapp: ${CONTACT_CACHE.whatsapp} 
${CONTACT_CACHE.email}”
- Use the reply language specified above.
`.trim();

    // Robust model call
    let text = await tryGeminiText(prompt, { retries: 2, backoffMs: 1200, modelFallback: FALLBACK_MODEL });

    // Safety: block premature contact drop if we had passable context
    const contactLine = "Please contact our sales team";
    const containsContact = text.includes(contactLine);
    const contextLooksNonEmpty = Boolean(context && context.trim().length > 0);
    if (containsContact && contextLooksNonEmpty) {
      text = mode==="hinglish"
        ? "Mujhe context mein exact details nahi mili. Agar aap model/feature thoda aur specific batayenge to main exact specs dikha sakta hoon."
        : `Please contact our sales team at \nWhatsapp: ${CONTACT_CACHE.whatsapp} \n${CONTACT_CACHE.email}`;
    }
    if (!text) {
      text = contextLooksNonEmpty
        ? (mode==="hinglish" ? "Main context se details nikal raha hoon—please model/feature thoda aur specific batayein." : "I’m using the knowledge base—please specify the model/feature you need.")
        : `Please contact our sales team at \nWhatsapp: ${CONTACT_CACHE.whatsapp} \n${CONTACT_CACHE.email}`;
    }

    await saveTurn(sessionId,"user",q||"");
    await saveTurn(sessionId,"assistant",text);

    return res.status(200).json({
      answer: text,
      mode,
      bot: BOT_NAME,
      citations: candidates.map((s,i)=>({ idx:i+1, score: Number((s.score ?? s.lex ?? 0).toFixed(4)), kw: s.kw, cos: s.cos != null ? Number(s.cos.toFixed(4)) : undefined })),
    });
  } catch (err) {
    console.error("ask error:", err);
    return res.status(err?.status || 500).json({
      error: err?.message || "Server error",
      details: { status: err?.status || 500, statusText: err?.statusText || null, type: err?.name || null }
    });
  }
}
