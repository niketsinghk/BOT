import fs from "node:fs";
import path from "node:path";

// === CONFIG ===
// Names must match the `source` field stored in vectors (usually the PDF filename)
const SOURCES_TO_REMOVE = [
  // e.g. "HCA.pdf", "model_list.pdf"
  "Difference.pdf"
];

const DATA_DIR  = path.resolve(process.cwd(), "..", "data");
const INDEX_IN  = path.join(DATA_DIR, "index.json");
const INDEX_OUT = INDEX_IN; // overwrite in place; change if you want a new file

if (!fs.existsSync(INDEX_IN)) {
  console.error("❌ index.json not found at:", INDEX_IN);
  process.exit(1);
}

// Backup first
const backupPath = INDEX_IN.replace(/\.json$/, `.backup.${Date.now()}.json`);
fs.copyFileSync(INDEX_IN, backupPath);
console.log("🧯 Backup created:", backupPath);

// Load
const raw = JSON.parse(fs.readFileSync(INDEX_IN, "utf8"));
const origCount = (raw.vectors || []).length;

// If you want to see available sources:
const sourcesPresent = new Set((raw.vectors || []).map(v => v.source));
console.log("📚 Sources in index:", [...sourcesPresent].join(", ") || "(none)");

// Filter vectors
const removeSet = new Set(SOURCES_TO_REMOVE.map(s => s.trim().toLowerCase()));
const kept = (raw.vectors || []).filter(v => !removeSet.has(String(v.source).toLowerCase()));
const removed = origCount - kept.length;

// Renumber IDs to keep them dense
kept.forEach((v, i) => { v.id = i; });

// Update meta.sources (optional but tidy)
if (raw.meta?.sources) {
  raw.meta.sources = raw.meta.sources.filter(s => !removeSet.has(String(s.name).toLowerCase()));
}

// Save
raw.vectors = kept;
fs.writeFileSync(INDEX_OUT, JSON.stringify(raw, null, 2));
console.log(`✅ Pruned ${removed} vector(s).`);
console.log(`📊 Remaining: ${kept.length} vector(s).`);
console.log("💾 Saved:", INDEX_OUT);
