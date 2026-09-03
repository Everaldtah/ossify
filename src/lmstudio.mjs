// LM Studio driver: server lifecycle, model resolution, load/unload, benchmark.
import { LMStudioClient } from "@lmstudio/sdk";
import { spawn } from "node:child_process";
import { existsSync, statSync, readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { lmstudio as lmsInfo, LMS_DIR } from "./sys.mjs";

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

export async function serverUp(port) {
  try {
    const ac = new AbortController(); const t = setTimeout(() => ac.abort(), 2500);
    const r = await fetch(`http://127.0.0.1:${port}/api/v1/models`, { signal: ac.signal });
    clearTimeout(t);
    return r.ok;
  } catch { return false; }
}

export async function ensureServer(log = () => {}) {
  const info = lmsInfo();
  if (await serverUp(info.port)) return info;
  if (!info.installed) throw new Error(`LM Studio CLI not found at ${info.lmsExe}. Install LM Studio (https://lmstudio.ai) and run it once.`);
  log(`LM Studio server not reachable on :${info.port} - starting it with lms...`);
  await new Promise((resolve) => {
    const p = spawn(info.lmsExe, ["server", "start", "--port", String(info.port)], { stdio: "ignore", windowsHide: true });
    p.on("exit", resolve); p.on("error", resolve);
  });
  for (let i = 0; i < 40; i++) { if (await serverUp(info.port)) return info; await sleep(1000); }
  throw new Error("LM Studio server did not come up within 40s");
}

export function client(port) {
  return new LMStudioClient({ baseUrl: `ws://127.0.0.1:${port}` });
}

function* walkGguf(dir, depth = 0) {
  if (depth > 5 || !existsSync(dir)) return;
  for (const e of readdirSync(dir, { withFileTypes: true })) {
    const p = join(dir, e.name);
    if (e.isDirectory()) yield* walkGguf(p, depth + 1);
    else if (/\.gguf$/i.test(e.name)) yield p;
  }
}

/** Resolve a model key ("openai/gpt-oss-20b") to its downloaded info and absolute GGUF path. */
export async function resolveModel(lms, key, modelsDir) {
  const all = await lms.system.listDownloadedModels();
  const llms = all.filter((m) => m.type === "llm");
  const norm = (s) => s.toLowerCase();
  let m = llms.find((x) => norm(x.modelKey) === norm(key)) || llms.find((x) => norm(x.modelKey).startsWith(norm(key) + "@"))
    || llms.find((x) => norm(x.path) === norm(key)) || llms.find((x) => norm(x.modelKey).includes(norm(key)) || norm(x.path).includes(norm(key)));
  if (!m) throw new Error(`No downloaded LLM matches "${key}". Available: ${llms.map((x) => x.modelKey).join(", ")}`);
  let file = join(modelsDir, m.path);
  if (!/\.gguf$/i.test(file) || !existsSync(file)) {
    // Hub-style virtual model ("openai/gpt-oss-20b"): follow the hub manifest to the concrete GGUF repo.
    let hit = null;
    const manifest = join(LMS_DIR, "hub", "models", ...m.path.split("/"), "manifest.json");
    if (existsSync(manifest)) {
      try {
        const keys = JSON.parse(readFileSync(manifest, "utf8")).dependencies?.flatMap((d) => d.modelKeys ?? []) ?? [];
        const ggufs = [...walkGguf(modelsDir)];
        for (const k of keys) {
          const found = ggufs.filter((p) => norm(p.replace(/\\/g, "/")).includes(norm(k).replace(/-gguf$/, "")));
          if (found.length) { hit = found.sort((a, b) => statSync(b).size - statSync(a).size)[0]; break; }
        }
      } catch { /* fall through */ }
    }
    if (!hit) { // last resort: match by size (manifest bytes make it inexact)
      hit = [...walkGguf(modelsDir)].find((p) => { try { return Math.abs(statSync(p).size - m.sizeBytes) < m.sizeBytes * 0.005; } catch { return false; } });
    }
    if (!hit) throw new Error(`Could not locate the GGUF file for ${m.modelKey} under ${modelsDir}`);
    file = hit;
  }
  // LM Studio also loads a sibling vision projector (mmproj-*.gguf) onto the GPU for VLMs.
  let mmprojBytes = 0;
  try {
    const dir = file.slice(0, Math.max(file.lastIndexOf("\\"), file.lastIndexOf("/")));
    for (const e of readdirSync(dir)) if (/^mmproj.*\.gguf$/i.test(e)) mmprojBytes += statSync(join(dir, e)).size;
  } catch { /* ignore */ }
  return { ...m, file, format: m.format, mmprojBytes };
}

export async function listLoaded(lms) {
  const loaded = await lms.llm.listLoaded();
  return Promise.all(loaded.map(async (h) => ({ identifier: h.identifier, modelKey: h.modelKey, path: h.path, handle: h })));
}

export async function unloadAll(lms, log = () => {}) {
  for (const h of await lms.llm.listLoaded()) { log(`Unloading ${h.identifier}`); await lms.llm.unload(h.identifier); }
  for (const h of await lms.embedding.listLoaded()) { log(`Unloading embedding ${h.identifier}`); await lms.embedding.unload(h.identifier); }
}

// --- raw KV field injection -------------------------------------------------------------------
// @lmstudio/sdk 1.5.0 drops `gpu.numCpuExpertLayersRatio` (and a few other experimental fields)
// when it converts LLMLoadModelConfig to the wire format, even though LM Studio's server accepts
// them (verified against LM Studio 0.4.2 server logs). We patch the outgoing "loadModel" message
// and append the fields ourselves. Keys follow LM Studio's llm.load schematics.
let pendingExtraFields = [];
let patched = false;
async function installWirePatch() {
  if (patched) return; patched = true;
  const protos = [];
  try { const { createRequire } = await import("node:module"); const req = createRequire(import.meta.url); protos.push(req("ws").WebSocket.prototype); } catch { /* sdk may use global WebSocket */ }
  if (globalThis.WebSocket) protos.push(globalThis.WebSocket.prototype);
  for (const p of protos) {
    const orig = p.send;
    p.send = function (data) {
      if (pendingExtraFields.length && typeof data === "string" && data.includes('"endpoint":"loadModel"')) {
        try {
          const msg = JSON.parse(data);
          const layer = msg.creationParameter?.loadConfigStack?.layers?.find((l) => l.layerName === "apiOverride");
          if (layer) {
            const have = new Set(layer.config.fields.map((f) => f.key));
            for (const f of pendingExtraFields) if (!have.has(f.key)) layer.config.fields.push(f);
            data = JSON.stringify(msg);
          }
        } catch { /* send untouched */ }
      }
      return orig.call(this, data);
    };
  }
}

export function extraLoadFields(cfg) {
  const f = [];
  if (cfg.gpu?.numCpuExpertLayersRatio !== undefined) f.push({ key: "llm.load.numCpuExpertLayersRatio", value: cfg.gpu.numCpuExpertLayersRatio });
  if (cfg.numParallelSessions !== undefined) f.push({ key: "llm.load.numParallelSessions", value: cfg.numParallelSessions });
  if (cfg.offloadKVCacheToGpu !== undefined) f.push({ key: "llm.load.offloadKVCacheToGpu", value: cfg.offloadKVCacheToGpu });
  if (cfg.evalBatchSize !== undefined) f.push({ key: "llm.load.llama.evalBatchSize", value: cfg.evalBatchSize });
  if (cfg.llamaKCacheQuantizationType) f.push({ key: "llm.load.llama.kCacheQuantizationType", value: { checked: true, value: cfg.llamaKCacheQuantizationType } });
  if (cfg.llamaVCacheQuantizationType) f.push({ key: "llm.load.llama.vCacheQuantizationType", value: { checked: true, value: cfg.llamaVCacheQuantizationType } });
  return f;
}

export async function load(lms, modelKey, config, { identifier, ttl, onProgress } = {}) {
  await installWirePatch();
  pendingExtraFields = extraLoadFields(config);
  try {
    return await lms.llm.load(modelKey, { identifier: identifier ?? modelKey, config, ttl, onProgress, verbose: false });
  } finally { pendingExtraFields = []; }
}

const LOREM = `function reconcile(prev, next, path = []) {
  const changes = [];
  for (const key of new Set([...Object.keys(prev ?? {}), ...Object.keys(next ?? {})])) {
    const a = prev?.[key], b = next?.[key];
    if (typeof a === "object" && typeof b === "object" && a && b) changes.push(...reconcile(a, b, [...path, key]));
    else if (a !== b) changes.push({ path: [...path, key].join("."), from: a, to: b });
  }
  return changes;
}
`;

/** Build a prompt of roughly N tokens (code + prose; ~3.6 chars/token). Unique per call so prefix caching cannot skew results. */
export function benchPrompt(tokens = 1800) {
  const target = tokens * 3.6; let s = `// bench-${Date.now()}-${Math.random().toString(36).slice(2)}\n`; let i = 0;
  while (s.length < target) s += LOREM.replaceAll("reconcile", `reconcile${i++}`) + `// Note ${i}: the helper above diffs nested objects and reports dotted paths for every changed leaf.\n`;
  return s + "\nExplain what this module does and list three edge cases it does not handle. Be concise.";
}

/** Run one prediction and derive prompt-processing and generation speeds from LM Studio's stats. */
export async function bench(model, { promptTokens = 1800, maxTokens = 160 } = {}) {
  const prompt = benchPrompt(promptTokens);
  const t0 = performance.now();
  const res = await model.respond([{ role: "user", content: prompt }], { maxTokens, temperature: 0, reasoning: "low" }).catch(async (e) => {
    if (/reasoning/i.test(String(e))) return model.respond([{ role: "user", content: prompt }], { maxTokens, temperature: 0 });
    throw e;
  });
  const wall = (performance.now() - t0) / 1000;
  const st = res.stats || {};
  const ttft = st.timeToFirstTokenSec ?? wall / 2;
  const pTok = st.promptTokensCount ?? promptTokens;
  const gTok = st.predictedTokensCount ?? maxTokens;
  return {
    ppTps: pTok / Math.max(ttft, 0.001),
    tgTps: st.tokensPerSecond ?? gTok / Math.max(wall - ttft, 0.001),
    ttftSec: ttft, promptTokens: pTok, genTokens: gTok, wallSec: wall, numGpuLayers: st.numGpuLayers, stopReason: st.stopReason,
  };
}
