// scripts/embed.mjs — Multi-PDF embedding with Poppler fallback (Dukejia)

/* ===================== Imports ===================== */
import path from "node:path";
import fs from "node:fs";
import { fileURLToPath } from "node:url";
import dotenv from "dotenv";
import { GoogleGenerativeAI } from "@google/generative-ai";
import { execFileSync } from "node:child_process";

/* ---- pdf-parse: robust ESM/CJS loader ---- */
import { createRequire } from "node:module";
const require = createRequire(import.meta.url);

let pdfParse;
try {
  const m = await import("pdf-parse");
  pdfParse = m?.default ?? m;
} catch {
  const m2 = require("pdf-parse");
  pdfParse = m2?.default ?? m2;
}
if (typeof pdfParse !== "function") {
  throw new TypeError("pdf-parse export resolution failed. Try `npm i pdf-parse@1`.");
}

/* ===================== Env Bootstrap ===================== */
const __filename = fileURLToPath(import.meta.url);
const __dirname  = path.dirname(__filename);
const ENV_PATH   = path.resolve(__dirname, "../.env");
dotenv.config({ path: ENV_PATH });

const API_KEY =
  process.env.GOOGLE_API_KEY ||
  process.env.GEMINI_API_KEY ||
  process.env.GENAI_API_KEY;

if (!API_KEY) {
  console.error("❌ Missing GOOGLE_API_KEY (or GEMINI_API_KEY/GENAI_API_KEY) in:", ENV_PATH);
  process.exit(1);
}

console.log("📄 Using .env:  ", ENV_PATH);
console.log("🔑 API key set: ", "yes");

/* ===================== Paths & Config ===================== */
const PROJECT_ROOT = path.resolve(__dirname, "..");
const DATA_DIR     = path.resolve(PROJECT_ROOT, "data");

function resolveToRoot(p) {
  const norm = String(p || "").replace(/\\+/g, "/");
  return path.isAbsolute(norm) ? norm : path.resolve(PROJECT_ROOT, norm);
}

// Prefer multi-PDF via PDF_PATHS (comma-separated).
// Fallback order: QUESTION_PATH, then PDF_PATH, then defaults.
let CANDIDATE_PDFS = [];
if (process.env.PDF_PATHS) {
  CANDIDATE_PDFS = process.env.PDF_PATHS.split(",").map(s => resolveToRoot(s.trim())).filter(Boolean);
} else {
  const qp = process.env.QUESTION_PATH;
  const sp = process.env.PDF_PATH;
  if (qp) CANDIDATE_PDFS.push(resolveToRoot(qp));
  if (sp) CANDIDATE_PDFS.push(resolveToRoot(sp));
  if (CANDIDATE_PDFS.length === 0) {
    CANDIDATE_PDFS = [
      resolveToRoot("./data/Question.pdf"),   // 👈 default new file
          resolveToRoot("./data/contact.pdf"),
            resolveToRoot("./data/Query.pdf"),
              resolveToRoot("./data/UserQuery.pdf"),
                resolveToRoot("./data/knowledge.pdf"),
                resolveToRoot("./data/Question.pdf"),
                resolveToRoot("./data/embroidery.pdf"),
                resolveToRoot("./data/Difference.pdf"),

    ];
  }
}

// Optional: auto-scan /data for all PDFs when PDF_SCAN=1
if (String(process.env.PDF_SCAN || "0").trim() === "1" && fs.existsSync(DATA_DIR)) {
  const found = fs.readdirSync(DATA_DIR, { withFileTypes: true })
    .filter(d => d.isFile() && d.name.toLowerCase().endsWith(".pdf"))
    .map(d => path.join(DATA_DIR, d.name));
  const set = new Set([...CANDIDATE_PDFS, ...found.map(resolveToRoot)]);
  CANDIDATE_PDFS = [...set];
}

const OUT_PATH = resolveToRoot(process.env.OUT_PATH || path.join(DATA_DIR, "index.json"));

const GENERATION_MODEL = process.env.GENERATION_MODEL || "gemini-2.5-flash";
const EMBEDDING_MODEL  = process.env.EMBEDDING_MODEL  || "text-embedding-004";
const CHUNK_SIZE       = parseInt(process.env.CHUNK_SIZE    || "1200", 10);
const CHUNK_OVERLAP    = parseInt(process.env.CHUNK_OVERLAP || "200", 10);

/* ===================== Client ===================== */
const genAI    = new GoogleGenerativeAI(API_KEY);
const embedder = genAI.getGenerativeModel({ model: EMBEDDING_MODEL });

/* ===================== Utilities ===================== */
function chunkText(text, size, overlap) {
  const chunks = [];
  let start = 0;
  while (start < text.length) {
    const end = Math.min(start + size, text.length);
    const chunk = text.slice(start, end).trim();
    if (chunk) chunks.push(chunk);
    if (end === text.length) break;
    start = Math.max(0, end - overlap);
  }
  return chunks;
}

function normalizeWhitespace(s) {
  return (s || "")
    .replace(/\r/g, "")
    .replace(/[ \t]+\n/g, "\n")
    .replace(/\n{3,}/g, "\n\n");
}

// Try Poppler when pdf-parse returns too little (fonts/encoding issues)
function tryPopplerText(pdfPath) {
  try {
    const args = ["-layout", "-enc", "UTF-8", pdfPath, "-"]; // stdout
    const out = execFileSync("pdftotext", args, { encoding: "utf8", stdio: ["ignore", "pipe", "pipe"] });
    return (out || "").trim();
  } catch {
    return "";
  }
}

/* ===================== Stopwords / Protected Tokens ===================== */
const EN_STOPWORDS = new Set(`
a about above after again against all am an and any are aren't as at
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
you you'd you'll you're you've your yours yourself yourselves
`.trim().split(/\s+/));

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
  "hari","chand","anand","anil","hca",
  "hari-chand-anand","hari-chand-anand-&-co","hari-chand-anand-and-co",
  "duke","duke-jia","dukejia","duki","contact","call","email","address","Branches","Headquarters",
  "Head Office","Factory","Works","Website","WhatsApp","Whatsapp","Whats app","Phone",
  "Brand","Brands","names","name","features","feature","specification","specifications","specs","model","models","type","types",
  "descriptions","description","Appllications","application","Machine_id","Machine ID","ID",
  "delhi","india","bangladesh","ethiopia",
  "automation","solution","solutions","garment","leather","mattress","perforation","embroidery","quilting","sewing","upholstery","pattern",
  "sequin","sequins","bead","beads","cording","coiling","taping","rhinestone","chenille","chainstitch","cap","tubular",
  "dahao","a18","dst","tajima","usb","u-disk","lcd","touchscreen","network",
  "auto-trimming","automatic-trimming","auto-color-change","automatic-color-change","thread-break-detection","power-failure-recovery",
  "servo","servo-motor","36v","36v-dc","oil-mist","dust-clean","wide-voltage","270-cap-frame",
  "es-1300","es 1300","dy pe750x600","halo-100","dy-601ctm","dy sk d2-2.0rh",
  "dy-606","dy-606h","dy-606hc","dy-606l","dy-606xl","dy-606s","dy-606+1ct","dy-606+1pd",
  "dy-602","dy-602h","dy-602hc","dy-602l","dy-602xl","dy-602s","dy-602+1ct","dy-602+1pd","dy 601ctm",
  "dy 606+6","dy602+2","dy-606+6","dy 908","dy 912","dy 915-120","dy 918-120",
  "dy 1201","dy 1201l","dy 1201h","dy 1201xl","dy 1201s","dy-1201","dy-1201l","dy-1201h","dy-1201xl","dy-1201s","dy Halo-100",
  "dy-1201+1ct","dy-1201+1pd","dy 1204","dy 1206","dy 1206h","dy 1206hc","dy-1204","dy-1206","dy-1206h","dy-1206hc",
  "dy-1202","dy-1202l","dy-1202h","dy-1202xl","dy-1202s","dy 918","dy 915",
  "dy 1502","dy-1502","dy-1202hc","dy-1203h","dy-1204","dy-1206","dy-1206h",
  "dy-1502","dy-908","dy-912","dy915-120","dy918-120","dy cs3000",
  "duke-single-head","duke multi-head","duke multi head","duke multihead",
  "dukejia-single-head","dukejia multi-head","dukejia multi head","dukejia multihead",
  "dy-cs3000","dy-pe750x600","dy-sk-d2-2.0rh"
]);

// Keep units/symbols useful in specs: . , / + - × ° % " '
const KEEP_REGEX = /[^a-z0-9\u0900-\u097F\s\.\,\+\/\-×°%"']/g;

function cleanForEmbedding(s) {
  if (!s) return "";
  const lower = s.toLowerCase();
  const stripped = lower.replace(KEEP_REGEX, " ");
  const tokens = stripped.split(/\s+/).filter(Boolean);
  const kept = tokens.filter((t) => {
    if (PROTECTED_TOKENS.has(t)) return true;
    if (EN_STOPWORDS.has(t)) return false;
    if (HINGLISH_STOPWORDS.has(t)) return false;
    if (HINDI_STOPWORDS.has(t)) return false;
    return true;
  });
  let out = kept.join(" ").trim();
  // If cleaning nuked too much (tables/numbers), use lightly-normalized original
  if (out.replace(/\s+/g, "").length < 16) {
    out = normalizeWhitespace(lower).slice(0, 4000);
  }
  return out;
}

/* ===================== Loaders ===================== */
async function readPdfText(filePath) {
  if (!filePath) return { ok: false, why: "no path" };
  if (!fs.existsSync(filePath)) return { ok: false, why: "missing", filePath };

  const buf = fs.readFileSync(filePath);
  if (!buf?.length) return { ok: false, why: "empty", filePath };

  // 1) primary: pdf-parse
  let text = "";
  try {
    const parsed = await pdfParse(buf);
    text = parsed?.text || "";
  } catch {
    text = "";
  }

  // 2) Poppler fallback if too little text
  if (!text || text.replace(/\s+/g, "").length < 50) {
    const poppler = tryPopplerText(filePath);
    if (poppler && poppler.replace(/\s+/g, "").length >= 50) {
      text = poppler;
      console.log("ℹ️  Used Poppler fallback for:", path.basename(filePath));
    }
  }

  text = normalizeWhitespace(text).trim();
  if (!text) return { ok: false, why: "no-extract", filePath };
  return { ok: true, text, filePath };
}

/* ===================== Runner ===================== */
async function main() {
  console.log("📂 CWD:          ", process.cwd());
  console.log("📂 PROJECT_ROOT: ", PROJECT_ROOT);
  console.log("📂 DATA_DIR:     ", DATA_DIR);
  console.log("📄 Candidate PDFs:", CANDIDATE_PDFS.join(" | "));
  console.log("📦 OUT_PATH:     ", OUT_PATH);
  console.log("🧩 CHUNK/OVERLAP:", CHUNK_SIZE, CHUNK_OVERLAP);
  console.log("🧠 MODEL(emb):   ", EMBEDDING_MODEL);

  const sources = [];
  for (const p of CANDIDATE_PDFS) {
    const res = await readPdfText(p);
    if (res.ok) {
      sources.push({ kind: "pdf", name: path.basename(p), text: res.text });
      console.log("✅ Loaded:", p);
    } else {
      console.log("⚠️ Skipped:", p, "reason:", res.why);
    }
  }

  if (sources.length === 0) {
    console.error("❌ No PDF sources found. Use PDF_PATHS or set QUESTION_PATH/PDF_PATH.");
    console.error("   Example .env:");
    console.error("   PDF_PATHS=./data/Difference.pdf");
    process.exit(1);
  }

  console.log("✂️  Chunking sources…");
  const allChunks = [];
  for (const src of sources) {
    const chunks = chunkText(src.text, CHUNK_SIZE, CHUNK_OVERLAP);
    chunks.forEach((c, idx) => {
      allChunks.push({
        source: src.name,
        kind: src.kind,
        chunk_index: idx,
        text_original: c,
      });
    });
    console.log(`   • ${src.name}: ${chunks.length} chunks`);
  }

  console.log(`🧠 Embedding ${allChunks.length} chunks…`);
  const batchSize = 64;
  const vectors = [];

  for (let i = 0; i < allChunks.length; i += batchSize) {
    const batch = allChunks.slice(i, i + batchSize);

    const cleaned = batch.map((b) => {
      const c = cleanForEmbedding(b.text_original);
      return c && c.trim() ? c : (normalizeWhitespace(b.text_original).slice(0, 4000) || " ");
    });

    const res = await embedder.batchEmbedContents({
      requests: cleaned.map((text) => ({ content: { parts: [{ text }] } })),
    });

    const emb = res.embeddings || [];
    for (let j = 0; j < emb.length; j++) {
      const src = batch[j];
      vectors.push({
        id: i + j,
        source: src.source,
        kind: src.kind,
        chunk_index: src.chunk_index,
        text_original: src.text_original,
        text_cleaned: cleaned[j],
        embedding: emb[j].values,
      });
    }
    console.log(`   → ${Math.min(i + batchSize, allChunks.length)}/${allChunks.length}`);
  }

  if (!fs.existsSync(DATA_DIR)) fs.mkdirSync(DATA_DIR, { recursive: true });
  const payload = {
    createdAt: new Date().toISOString(),
    model: EMBEDDING_MODEL,
    generation_model: GENERATION_MODEL,
    stopwords: "EN+Hinglish+Hindi",
    meta: {
      sources: sources.map((s) => ({ name: s.name, kind: s.kind })),
      chunk_size: CHUNK_SIZE,
      chunk_overlap: CHUNK_OVERLAP,
    },
    vectors,
  };
  fs.writeFileSync(OUT_PATH, JSON.stringify(payload, null, 2));
  console.log("✅ Saved embeddings to:", OUT_PATH);
  console.log(`📊 Total vectors: ${vectors.length}`);
}

main().catch((err) => {
  console.error("⚠️ Embed error:", err?.stack || err?.message || err);
  process.exit(1);
});
