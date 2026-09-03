#!/usr/bin/env node
// Ossify CLI - plan, tune, load and supervise a local LM Studio model for Claude Code.
//
//   ossify up      [--model KEY] [--ctx N] [--ttl SEC] [--retune] [--quick] [--keep-others]
//   ossify tune    [--model KEY] [--ctx N] [--deep]
//   ossify plan    [--model KEY] [--ctx N]
//   ossify status | unload | bench | doctor
import { mkdirSync, readFileSync, writeFileSync, existsSync } from "node:fs";
import { join } from "node:path";
import { readGguf, summarize } from "./gguf.mjs";
import { snapshot, fmtGB, fmtMiB, MiB, GiB, OSSIFY_DIR } from "./sys.mjs";
import { candidates, toLmsConfig, estimateVram, estimateRam, turnSeconds } from "./plan.mjs";
import { ensureServer, client, resolveModel, listLoaded, unloadAll, load, bench, serverUp } from "./lmstudio.mjs";

const DEFAULT_MODEL = "openai/gpt-oss-20b";
const VRAM_MARGIN_MIB = 450;   // leave room for the desktop compositor, browsers, Discord overlays

const args = process.argv.slice(2);
const cmd = args.shift() ?? "help";
const flag = (n) => args.includes(`--${n}`);
const opt = (n, d) => { const i = args.indexOf(`--${n}`); return i >= 0 && args[i + 1] !== undefined ? args[i + 1] : d; };
const log = (...m) => console.error("[ossify]", ...m);
const modelKey = opt("model", process.env.OSSIFY_MODEL || DEFAULT_MODEL);
const ctxTarget = parseInt(opt("ctx", process.env.OSSIFY_CTX || "65536"), 10);
const ttl = parseInt(opt("ttl", process.env.OSSIFY_TTL || "1800"), 10);
// RAM headroom we refuse to plan into. 4 GB keeps the desktop snappy; big models may need to lower it.
const RAM_MARGIN = Math.max(1, parseFloat(opt("ram-margin", process.env.OSSIFY_RAM_MARGIN_GB || "4"))) * GiB;

mkdirSync(OSSIFY_DIR, { recursive: true });
const tunedPath = (key) => join(OSSIFY_DIR, `tuned-${key.replace(/[^a-z0-9.-]+/gi, "_")}.json`);
const currentPath = join(OSSIFY_DIR, "current.json");

function fingerprint(sys, s, ctx) {
  return `${sys.gpu?.name ?? "cpu"}|${sys.gpu?.totalMiB ?? 0}|${sys.lmstudio.backend}|${s.totalBytes}|ctx${ctx}`;
}

function budgets(ctx) {
  ctx.sys = snapshot();
  ctx.budgetBytes = ctx.sys.gpu ? Math.max(0, (ctx.sys.gpu.freeMiB - VRAM_MARGIN_MIB) * MiB) : 0;
  ctx.ramFreeBytes = Math.max(0, ctx.sys.ram.freeBytes - RAM_MARGIN);
  return ctx;
}

// After an unload LM Studio takes several seconds to hand 10-20 GB back to Windows. Planning
// against a snapshot taken during that window under-budgets RAM and picks a bad placement.
async function settleMemory(maxSec = 25) {
  let last = snapshot().ram.freeBytes;
  for (let i = 0; i < maxSec; i++) {
    await new Promise((r) => setTimeout(r, 1000));
    const now = snapshot().ram.freeBytes;
    if (Math.abs(now - last) < 150 * MiB && i >= 2) break;
    last = now;
  }
}

async function analyze() {
  const info = await ensureServer(log);
  const lms = client(info.port);
  const sys = snapshot();
  const m = await resolveModel(lms, modelKey, sys.lmstudio.modelsDir);
  const s = summarize(await readGguf(m.file));
  s.mmprojBytes = m.mmprojBytes || 0;
  return budgets({ sys, info, lms, m, s });
}

function printPlan(ctx, cands) {
  console.log(`\nModel  ${ctx.m.modelKey}  (${ctx.s.arch}, ${ctx.s.nLayer} layers, ${ctx.s.nExpert ? `${ctx.s.nExpert} experts / ${ctx.s.nExpertUsed} active, ` : ""}${fmtGB(ctx.s.totalBytes)})`);
  console.log(`       dense ${fmtGB(ctx.s.denseBytes + ctx.s.embedBytes)}  experts ${fmtGB(ctx.s.expertBytes)}  (${fmtGB(ctx.s.expertBytesPerLayer)}/layer)`);
  console.log(`GPU    ${ctx.sys.gpu ? `${ctx.sys.gpu.name}  ${fmtMiB(ctx.sys.gpu.freeMiB)} free of ${fmtMiB(ctx.sys.gpu.totalMiB)}  -> budget ${fmtGB(ctx.budgetBytes)}` : "none detected"}`);
  console.log(`RAM    ${fmtGB(ctx.sys.ram.freeBytes)} free of ${fmtGB(ctx.sys.ram.totalBytes)}  -> budget ${fmtGB(ctx.ramFreeBytes)}`);
  console.log(`CPU    ${ctx.sys.cpu.model}  ${ctx.sys.cpu.physical}c/${ctx.sys.cpu.logical}t\n`);
  for (const c of cands) {
    const e = c.est;
    console.log(`  ${c.id.padEnd(34)} vram~${fmtGB(e.total).padStart(8)}  kv ${fmtGB(e.kv)}  experts-on-gpu ${e.expertLayersOnGpu ?? 0}  ram~${fmtGB(estimateRam(ctx.s, c.cfg, e))}`);
  }
}

async function unloadOthers(ctx, keepKey) {
  const loaded = await listLoaded(ctx.lms);
  for (const h of loaded) {
    if (h.modelKey === keepKey || h.identifier === keepKey) continue;
    log(`Unloading ${h.identifier} to free memory`);
    await ctx.lms.llm.unload(h.identifier);
  }
  for (const h of await ctx.lms.embedding.listLoaded()) { log(`Unloading embedding ${h.identifier}`); await ctx.lms.embedding.unload(h.identifier); }
}

async function loadWith(ctx, cand, { ttl: t } = {}) {
  const cfg = toLmsConfig(cand.cfg);
  let last = -1;
  const t0 = Date.now();
  const model = await load(ctx.lms, ctx.m.modelKey, cfg, {
    ttl: t, onProgress: (p) => { const pct = Math.floor(p * 10) * 10; if (pct !== last) { last = pct; process.stderr.write(`\r[ossify] loading ${cand.id} ${pct}%   `); } },
  });
  process.stderr.write(`\r[ossify] loaded ${cand.id} in ${((Date.now() - t0) / 1000).toFixed(1)}s        \n`);
  return model;
}

function needMoreRam(ctx) {
  const minimal = candidates(ctx.s, { budgetBytes: ctx.budgetBytes, ramFreeBytes: Infinity, ctxTarget })[0];
  const need = estimateRam(ctx.s, minimal.cfg, minimal.est) + RAM_MARGIN;
  return `Not enough free RAM for ${ctx.m.modelKey}: needs ~${fmtGB(need)} free (incl. ${fmtGB(RAM_MARGIN)} headroom), have ${fmtGB(ctx.sys.ram.freeBytes)}. Close ~${fmtGB(need - ctx.sys.ram.freeBytes)} of apps (browsers, Discord, Notion, games) and retry, or lower --ram-margin.`;
}

async function runTune(ctx, { deep = false } = {}) {
  await unloadAll(ctx.lms, log);
  await settleMemory();
  budgets(ctx);
  const cands = candidates(ctx.s, { budgetBytes: ctx.budgetBytes, ramFreeBytes: ctx.ramFreeBytes, ctxTarget, deep });
  if (!cands.length) throw new Error(needMoreRam(ctx));
  printPlan(ctx, cands);
  const results = [];
  for (const cand of cands) {
    try {
      const model = await loadWith(ctx, cand);
      // Spill guard: if the card is (nearly) full the driver is paging VRAM over PCIe - every number
      // after this point would be garbage (3 tok/s territory) and RAM balloons. Skip the benchmark.
      const g = snapshot().gpu;
      if (g && g.totalMiB - g.usedMiB < 200) {
        console.log(`  ${cand.id.padEnd(34)} SPILLED: ${fmtMiB(g.usedMiB)} of ${fmtMiB(g.totalMiB)} VRAM in use after load - skipped`);
        results.push({ id: cand.id, strategy: cand.strategy, cfg: cand.cfg, est: cand.est, error: "vram spill", score: Infinity });
        await ctx.lms.llm.unload(model.identifier); continue;
      }
      const r = snapshot().ram;
      if (r.freeBytes < 1.5 * GiB) {
        console.log(`  ${cand.id.padEnd(34)} RAM CRITICAL: only ${fmtGB(r.freeBytes)} free after load - skipped`);
        results.push({ id: cand.id, strategy: cand.strategy, cfg: cand.cfg, est: cand.est, error: "ram exhausted", score: Infinity });
        await ctx.lms.llm.unload(model.identifier); continue;
      }
      await model.respond([{ role: "user", content: "Say OK." }], { maxTokens: 4, temperature: 0 }); // warm-up: graph + weights paged in
      const b1 = await bench(model, { promptTokens: 1800, maxTokens: 160 });
      const b2 = await bench(model, { promptTokens: 1800, maxTokens: 160 });
      const b = { ppTps: Math.max(b1.ppTps, b2.ppTps), tgTps: Math.max(b1.tgTps, b2.tgTps), ttftSec: Math.min(b1.ttftSec, b2.ttftSec), numGpuLayers: b2.numGpuLayers, promptTokens: b2.promptTokens };
      const gpuAfter = snapshot().gpu;
      // q4_0 KV cache trades accuracy over long contexts for speed: only let it win when clearly faster.
      // Tight VRAM headroom (< 450 MiB) risks spilling during long prefills: prefer a safer placement unless it is much slower.
      const headroomMiB = gpuAfter ? gpuAfter.totalMiB - gpuAfter.usedMiB : Infinity;
      const score = turnSeconds(b) * (cand.strategy.includes("kvq4") ? 1.15 : 1) * (headroomMiB < 450 ? 1.12 : 1);
      results.push({ id: cand.id, strategy: cand.strategy, cfg: cand.cfg, est: cand.est, bench: b, score, vramUsedMiB: gpuAfter?.usedMiB });
      console.log(`  ${cand.id.padEnd(34)} prefill ${b.ppTps.toFixed(0).padStart(5)} tok/s   gen ${b.tgTps.toFixed(1).padStart(5)} tok/s   turn ${score.toFixed(1)}s   vram-used ${gpuAfter ? fmtMiB(gpuAfter.usedMiB) : "?"}`);
      await ctx.lms.llm.unload(model.identifier);
    } catch (e) {
      console.log(`  ${cand.id.padEnd(34)} FAILED: ${String(e.message || e).split("\n")[0]}`);
      results.push({ id: cand.id, strategy: cand.strategy, cfg: cand.cfg, est: cand.est, error: String(e.message || e), score: Infinity });
      await unloadAll(ctx.lms).catch(() => {});
    }
  }
  const ok = results.filter((r) => Number.isFinite(r.score)).sort((a, b) => a.score - b.score);
  if (!ok.length) throw new Error("Every candidate failed to load. Run `ossify doctor`.");
  const chosen = ok[0];
  const rec = { modelKey: ctx.m.modelKey, file: ctx.m.file, fingerprint: fingerprint(ctx.sys, ctx.s, ctxTarget), tunedAt: new Date().toISOString(), chosen, results };
  writeFileSync(tunedPath(ctx.m.modelKey), JSON.stringify(rec, null, 2));
  console.log(`\nWinner: ${chosen.id}  (${chosen.bench.tgTps.toFixed(1)} tok/s generation, ${chosen.bench.ppTps.toFixed(0)} tok/s prefill). Saved to ${tunedPath(ctx.m.modelKey)}`);
  return rec;
}

function readTuned(ctx) {
  const p = tunedPath(ctx.m.modelKey);
  if (!existsSync(p)) return null;
  try { const t = JSON.parse(readFileSync(p, "utf8")); return t.fingerprint === fingerprint(ctx.sys, ctx.s, ctxTarget) ? t : null; } catch { return null; }
}

const PROXY_PORT = parseInt(process.env.OSSIFY_PROXY_PORT || "20130", 10);

async function proxyUp() {
  try { const r = await fetch(`http://127.0.0.1:${PROXY_PORT}/ossify/health`, { signal: AbortSignal.timeout(1500) }); return r.ok; } catch { return false; }
}

// The shim proxy (src/proxy.mjs) rewrites the few request shapes LM Studio's Anthropic endpoint
// rejects. It is started detached once and survives across Claude Code sessions.
async function ensureProxy(ctx) {
  if (await proxyUp()) return;
  const { spawn } = await import("node:child_process");
  const { fileURLToPath } = await import("node:url");
  const script = fileURLToPath(new URL("./proxy.mjs", import.meta.url));
  const child = spawn(process.execPath, [script, "--port", String(PROXY_PORT), "--target", String(ctx.info.port)], { detached: true, stdio: "ignore", windowsHide: true });
  child.unref();
  for (let i = 0; i < 20; i++) { if (await proxyUp()) { log(`shim proxy started on :${PROXY_PORT} (pid ${child.pid})`); return; } await new Promise((r) => setTimeout(r, 250)); }
  throw new Error(`shim proxy did not start on :${PROXY_PORT}`);
}

function writeCurrent(ctx, cand, extra = {}) {
  const cur = { modelKey: ctx.m.modelKey, identifier: ctx.m.modelKey, port: ctx.info.port, lmsUrl: `http://127.0.0.1:${ctx.info.port}`, baseUrl: `http://127.0.0.1:${PROXY_PORT}`, contextLength: cand.cfg.contextLength, strategy: cand.strategy ?? cand.id, config: toLmsConfig(cand.cfg), bench: cand.bench ?? null, loadedAt: new Date().toISOString(), ...extra };
  writeFileSync(currentPath, JSON.stringify(cur, null, 2));
  return cur;
}

const commands = {
  async plan() {
    const ctx = await analyze();
    const cands = candidates(ctx.s, { budgetBytes: ctx.budgetBytes, ramFreeBytes: ctx.ramFreeBytes, ctxTarget, deep: flag("deep") });
    printPlan(ctx, cands);
    if (!cands.length) console.log("  (nothing fits)  " + needMoreRam(ctx));
  },

  async tune() { const ctx = await analyze(); await runTune(ctx, { deep: flag("deep") }); },

  async up() {
    const ctx = await analyze();
    const loaded = await listLoaded(ctx.lms);
    const mine = loaded.find((h) => h.modelKey === ctx.m.modelKey || h.identifier === ctx.m.modelKey);
    if (mine && !flag("retune") && existsSync(currentPath)) {
      const cur = JSON.parse(readFileSync(currentPath, "utf8"));
      if (cur.modelKey === ctx.m.modelKey && cur.contextLength === ctxTarget) {
        log(`${ctx.m.modelKey} already loaded (${cur.strategy}, ctx ${cur.contextLength}). Reusing.`);
        await ensureProxy(ctx);
        writeCurrent(ctx, { cfg: { contextLength: cur.contextLength, ...cur.config }, strategy: cur.strategy, bench: cur.bench }, { reused: true });
        return;
      }
    }
    if (!flag("keep-others") && loaded.length) { await unloadOthers(ctx, null); await settleMemory(); }
    budgets(ctx);
    let tuned = flag("retune") ? null : readTuned(ctx);
    let cand;
    if (tuned) { cand = tuned.chosen; log(`Using tuned profile ${cand.id} (${cand.bench.tgTps.toFixed(1)} tok/s gen, tuned ${tuned.tunedAt.slice(0, 10)})`); }
    else if (flag("quick")) {
      cand = candidates(ctx.s, { budgetBytes: ctx.budgetBytes, ramFreeBytes: ctx.ramFreeBytes, ctxTarget })[0];
      if (!cand) throw new Error(needMoreRam(ctx));
      log(`No tuned profile - using planner default ${cand.id}`);
    }
    else { log("No tuned profile for this machine/model/context yet - running the auto-tuner once (a few minutes). Use --quick to skip."); tuned = await runTune(ctx); cand = tuned.chosen; }
    // Preflight against the *current* free memory, not the tuning-time numbers.
    const est = estimateVram(ctx.s, cand.cfg);
    const needRam = estimateRam(ctx.s, cand.cfg, est);
    if (needRam > ctx.ramFreeBytes) throw new Error(`Refusing to load: needs ~${fmtGB(needRam)} RAM but only ${fmtGB(ctx.sys.ram.freeBytes)} is free (keeping ${fmtGB(RAM_MARGIN)} headroom). Close some apps first.`);
    if (ctx.sys.gpu && est.total > ctx.budgetBytes) log(`Warning: plan wants ${fmtGB(est.total)} VRAM, budget is ${fmtGB(ctx.budgetBytes)} - the GPU is busier than at tuning time; expect slower speed.`);
    await loadWith(ctx, cand, { ttl: ttl > 0 ? ttl : undefined });
    await ensureProxy(ctx);
    const after = snapshot();
    const cur = writeCurrent(ctx, cand);
    console.log(`\n  ${ctx.m.modelKey}  ready on ${cur.baseUrl}   ctx ${cur.contextLength}   ${cand.strategy}`);
    if (cand.bench) console.log(`  expected: ~${cand.bench.tgTps.toFixed(0)} tok/s generation, ~${cand.bench.ppTps.toFixed(0)} tok/s prefill`);
    console.log(`  memory:   VRAM ${after.gpu ? `${fmtMiB(after.gpu.usedMiB)} used / ${fmtMiB(after.gpu.totalMiB)}` : "n/a"}   RAM free ${fmtGB(after.ram.freeBytes)}${ttl > 0 ? `   auto-unload after ${Math.round(ttl / 60)} min idle` : ""}`);
  },

  async status() {
    const sys = snapshot();
    const up = await serverUp(sys.lmstudio.port);
    console.log(`LM Studio server: ${up ? `up on :${sys.lmstudio.port}` : "down"}   backend ${sys.lmstudio.backend}   shim proxy :${PROXY_PORT} ${(await proxyUp()) ? "up" : "down"}`);
    console.log(`GPU: ${sys.gpu ? `${sys.gpu.name}  ${fmtMiB(sys.gpu.usedMiB)} used / ${fmtMiB(sys.gpu.totalMiB)}` : "none"}   RAM free ${fmtGB(sys.ram.freeBytes)} / ${fmtGB(sys.ram.totalBytes)}`);
    if (up) {
      const loaded = await listLoaded(client(sys.lmstudio.port));
      console.log(loaded.length ? `Loaded: ${loaded.map((h) => h.identifier).join(", ")}` : "Loaded: nothing");
    }
    if (existsSync(currentPath)) { const c = JSON.parse(readFileSync(currentPath, "utf8")); console.log(`Last ossify load: ${c.modelKey}  ctx ${c.contextLength}  ${c.strategy}  at ${c.loadedAt}`); }
    const tp = tunedPath(modelKey);
    if (existsSync(tp)) { const t = JSON.parse(readFileSync(tp, "utf8")); console.log(`Tuned profile: ${t.chosen.id}  gen ${t.chosen.bench.tgTps.toFixed(1)} tok/s  prefill ${t.chosen.bench.ppTps.toFixed(0)} tok/s  (${t.tunedAt.slice(0, 10)})`); }
  },

  async unload() { const info = await ensureServer(log); await unloadAll(client(info.port), log); console.log("All models unloaded."); },

  async bench() {
    const ctx = await analyze();
    const loaded = await listLoaded(ctx.lms);
    const h = loaded.find((x) => x.modelKey === ctx.m.modelKey) ?? loaded[0];
    if (!h) throw new Error("Nothing loaded. Run `ossify up` first.");
    const b = await bench(h.handle, { promptTokens: parseInt(opt("tokens", "1800"), 10), maxTokens: 200 });
    console.log(`${h.identifier}: prefill ${b.ppTps.toFixed(0)} tok/s (${b.promptTokens} tok, ttft ${b.ttftSec.toFixed(2)}s)   generation ${b.tgTps.toFixed(1)} tok/s (${b.genTokens} tok)   gpu layers ${b.numGpuLayers ?? "?"}`);
  },

  async doctor() {
    const sys = snapshot();
    console.log(JSON.stringify({ ...sys, ossifyDir: OSSIFY_DIR, node: process.version }, null, 2));
    const up = await serverUp(sys.lmstudio.port);
    console.log(`server: ${up ? "reachable" : "NOT reachable"}`);
    if (up) {
      const r = await fetch(`http://127.0.0.1:${sys.lmstudio.port}/v1/messages`, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ model: "__probe__", max_tokens: 1, messages: [{ role: "user", content: "x" }] }) });
      console.log(`anthropic /v1/messages endpoint: HTTP ${r.status} ${r.status === 404 ? "(MISSING - LM Studio too old, need >= 0.3.x with Anthropic compat)" : "(present)"}`);
    }
  },

  help() {
    console.log(`ossify <up|tune|plan|status|unload|bench|doctor> [--model KEY] [--ctx N] [--ttl SEC] [--deep] [--quick] [--retune] [--keep-others]\n  default model ${DEFAULT_MODEL}, ctx ${ctxTarget}, ttl ${ttl}s`);
  },
};

try {
  if (!commands[cmd]) { commands.help(); process.exit(2); }
  await commands[cmd]();
  process.exit(0);
} catch (e) {
  console.error(`[ossify] error: ${e.message || e}`);
  process.exit(1);
}
