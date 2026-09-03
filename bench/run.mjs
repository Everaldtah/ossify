#!/usr/bin/env node
// Ossify benchmark: score a local model's coding and tool-calling ability through the real
// Claude Code harness, so the number reflects what you actually get when you type `gptoss`.
//
//   node bench/run.mjs                      run every task against the loaded model
//   node bench/run.mjs --group tools        only the tool-calling tasks (tools|coding|instruction)
//   node bench/run.mjs --only code-bugfix   one task by id
//   node bench/run.mjs --repeat 3           run each task N times (these models are not deterministic)
//   node bench/run.mjs --timeout 900        per-task seconds (default 600)
//
// Each task runs as its OWN Claude Code session, so tasks cannot contaminate each other. That
// means every task pays the full system-prompt prefill, which is the honest cost of a first turn.
import { readFileSync, writeFileSync, mkdirSync, existsSync, rmSync, cpSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { spawn } from "node:child_process";
import { homedir } from "node:os";

const HERE = dirname(fileURLToPath(import.meta.url));
const REPO = dirname(HERE);
const args = process.argv.slice(2);
const opt = (n, d) => { const i = args.indexOf(`--${n}`); return i >= 0 && args[i + 1] !== undefined ? args[i + 1] : d; };
const flag = (n) => args.includes(`--${n}`);

const REPEAT = parseInt(opt("repeat", "1"), 10);
const TIMEOUT_MS = parseInt(opt("timeout", "600"), 10) * 1000;
const GROUP = opt("group", null);
const ONLY = opt("only", null);
const KEEP = flag("keep");

const currentPath = join(homedir(), ".ossify", "current.json");
if (!existsSync(currentPath)) { console.error("No model loaded. Run `gptoss --oss-status` or `ossify up` first."); process.exit(2); }
const cur = JSON.parse(readFileSync(currentPath, "utf8"));

const claudeExe = process.env.OSSIFY_CLAUDE || join(homedir(), ".local", "bin", "claude.exe");
if (!existsSync(claudeExe)) { console.error(`claude.exe not found at ${claudeExe}. Set OSSIFY_CLAUDE.`); process.exit(2); }

const pythonExe = process.env.OSSIFY_PYTHON || "python";
const outRoot = join(REPO, "bench", ".work");
if (!KEEP && existsSync(outRoot)) rmSync(outRoot, { recursive: true, force: true });
mkdirSync(outRoot, { recursive: true });

// Same provider wiring the gptoss/qwen35 launchers use, applied to the child only.
function childEnv() {
  const compact = Math.max(16384, (cur.contextLength ?? 65536) - 8192);
  const e = { ...process.env };
  for (const k of ["ANTHROPIC_API_KEY", "CLAUDE_CODE_USE_OPENAI", "OPENAI_BASE_URL", "OPENAI_API_KEY", "CLAUDE_CODE_USE_BEDROCK", "CLAUDE_CODE_USE_VERTEX"]) delete e[k];
  return {
    ...e,
    ANTHROPIC_BASE_URL: cur.baseUrl,
    ANTHROPIC_AUTH_TOKEN: "lm-studio",
    ANTHROPIC_MODEL: cur.identifier,
    ANTHROPIC_DEFAULT_OPUS_MODEL: cur.identifier,
    ANTHROPIC_DEFAULT_SONNET_MODEL: cur.identifier,
    ANTHROPIC_DEFAULT_HAIKU_MODEL: cur.identifier,
    ANTHROPIC_SMALL_FAST_MODEL: cur.identifier,
    CLAUDE_CODE_AUTO_COMPACT_WINDOW: String(compact),
    CLAUDE_CODE_MAX_OUTPUT_TOKENS: "8192",
    API_TIMEOUT_MS: String(TIMEOUT_MS),
    CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC: "1",
    DISABLE_TELEMETRY: "1",
    DISABLE_ERROR_REPORTING: "1",
  };
}

function runClaude(prompt, allowedTools, cwd) {
  return new Promise((resolve) => {
    const a = ["-p", prompt, "--output-format", "json"];
    if (allowedTools) a.push("--allowedTools", allowedTools);
    const t0 = Date.now();
    const child = spawn(claudeExe, a, { cwd, env: childEnv(), stdio: ["ignore", "pipe", "pipe"], windowsHide: true });
    let out = "", err = "";
    child.stdout.on("data", (d) => (out += d));
    child.stderr.on("data", (d) => (err += d));
    const timer = setTimeout(() => { child.kill(); }, TIMEOUT_MS + 30000);
    child.on("close", () => {
      clearTimeout(timer);
      const secs = (Date.now() - t0) / 1000;
      // Claude Code prints one JSON object on stdout. Parse the whole thing first; only if that
      // fails fall back to the first top-level brace (never the last - that lands mid-object).
      let json = null;
      const text = out.trim();
      try { json = JSON.parse(text); } catch { /* try harder below */ }
      if (!json) { const i = text.indexOf("{"); if (i >= 0) { try { json = JSON.parse(text.slice(i)); } catch { /* give up */ } } }
      resolve({ secs, json, raw: out, err });
    });
  });
}

function runPython(code, cwd) {
  return new Promise((resolve) => {
    const file = join(cwd, "__ossify_check.py");
    writeFileSync(file, code + "\nprint('OSSIFY_CHECK_OK')\n");
    const child = spawn(pythonExe, [file], { cwd, stdio: ["ignore", "pipe", "pipe"], windowsHide: true });
    let out = "", err = "";
    child.stdout.on("data", (d) => (out += d));
    child.stderr.on("data", (d) => (err += d));
    child.on("error", () => resolve({ ok: false, detail: "python not runnable" }));
    child.on("close", () => resolve({ ok: out.includes("OSSIFY_CHECK_OK"), detail: (err.trim().split(/\r?\n/).pop() || "").slice(0, 160) }));
  });
}

async function grade(task, res, workDir) {
  const fails = [];
  if (!res.json) return { pass: false, fails: ["no JSON result from claude"], answer: "" };
  if (res.json.is_error) return { pass: false, fails: [`api error: ${String(res.json.result).slice(0, 120)}`], answer: "" };
  const answer = String(res.json.result ?? "");
  const c = task.check ?? {};
  if (c.answerMatches && !new RegExp(c.answerMatches, "i").test(answer)) fails.push(`answer did not match /${c.answerMatches}/`);
  if (c.answerNotMatches && new RegExp(c.answerNotMatches, "i").test(answer)) fails.push(`answer matched forbidden /${c.answerNotMatches}/`);
  if (c.fileExists && !existsSync(join(workDir, c.fileExists))) fails.push(`did not create ${c.fileExists}`);
  if (c.python) {
    const r = await runPython(c.python, workDir);
    if (!r.ok) fails.push(`python check failed: ${r.detail || "assertion"}`);
  }
  return { pass: fails.length === 0, fails, answer };
}

const all = JSON.parse(readFileSync(join(HERE, "tasks.json"), "utf8")).tasks;
const tasks = all.filter((t) => (!GROUP || t.group === GROUP) && (!ONLY || t.id === ONLY));
if (!tasks.length) { console.error(`No tasks matched. ids: ${all.map((t) => t.id).join(", ")}`); process.exit(2); }

console.log(`\nOssify benchmark`);
console.log(`  model    ${cur.identifier}   ctx ${cur.contextLength}   ${cur.strategy}`);
console.log(`  endpoint ${cur.baseUrl}`);
console.log(`  tasks    ${tasks.length}${REPEAT > 1 ? ` x ${REPEAT} runs` : ""}   (each is a fresh session, so each pays a full first-turn prefill)\n`);

const results = [];
for (const task of tasks) {
  for (let rep = 0; rep < REPEAT; rep++) {
    const label = REPEAT > 1 ? `${task.id}#${rep + 1}` : task.id;
    // Each task gets its own directory AS its working directory, with a private copy of the
    // fixtures inside it. Prompts then use short relative paths ("fixtures/inventory.py",
    // "inventory.py"): absolute Windows paths in a prompt get mangled by smaller models, which
    // scores a correct solution as a failure. The private copy also means a task that edits a
    // fixture in place cannot affect any other task.
    const workDir = join(outRoot, label.replace(/[^a-z0-9-]+/gi, "_"));
    mkdirSync(workDir, { recursive: true });
    cpSync(join(HERE, "fixtures"), join(workDir, "fixtures"), { recursive: true, filter: (s) => !s.includes("__pycache__") });
    process.stdout.write(`  ${label.padEnd(22)} running... `);
    const res = await runClaude(task.prompt, task.allowedTools ?? "", workDir);
    const g = await grade(task, res, workDir);
    const turns = res.json?.num_turns ?? 0;
    console.log(`${g.pass ? "PASS" : "FAIL"}  ${res.secs.toFixed(0).padStart(4)}s  ${String(turns).padStart(2)} turns${g.pass ? "" : "   " + g.fails[0]}`);
    results.push({ id: task.id, group: task.group, weight: task.weight ?? 1, rep: rep + 1, pass: g.pass, fails: g.fails, secs: res.secs, turns, answer: g.answer.slice(0, 400) });
  }
}

const byGroup = {};
for (const r of results) {
  const g = (byGroup[r.group] ??= { got: 0, max: 0, passed: 0, n: 0, secs: 0 });
  g.max += r.weight; g.got += r.pass ? r.weight : 0; g.passed += r.pass ? 1 : 0; g.n++; g.secs += r.secs;
}
const tot = { got: 0, max: 0, passed: 0, n: 0, secs: 0 };
for (const g of Object.values(byGroup)) { tot.got += g.got; tot.max += g.max; tot.passed += g.passed; tot.n += g.n; tot.secs += g.secs; }

console.log(`\n  ${"group".padEnd(14)} ${"passed".padEnd(10)} ${"weighted".padEnd(12)} avg time`);
for (const [name, g] of Object.entries(byGroup)) {
  console.log(`  ${name.padEnd(14)} ${`${g.passed}/${g.n}`.padEnd(10)} ${`${g.got}/${g.max}`.padEnd(12)} ${(g.secs / g.n).toFixed(0)}s`);
}
console.log(`  ${"TOTAL".padEnd(14)} ${`${tot.passed}/${tot.n}`.padEnd(10)} ${`${tot.got}/${tot.max}`.padEnd(12)} ${(tot.secs / tot.n).toFixed(0)}s`);
console.log(`\n  score ${((tot.got / tot.max) * 100).toFixed(0)}%   total wall time ${(tot.secs / 60).toFixed(1)} min`);

const failed = results.filter((r) => !r.pass);
if (failed.length) {
  console.log(`\n  failures:`);
  for (const f of failed) console.log(`    ${f.id.padEnd(22)} ${f.fails.join("; ").slice(0, 150)}`);
}

const stamp = new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19);
const outFile = join(REPO, "bench", `result-${cur.identifier.replace(/[^a-z0-9.-]+/gi, "_")}-${stamp}.json`);
writeFileSync(outFile, JSON.stringify({ model: cur.identifier, contextLength: cur.contextLength, strategy: cur.strategy, bench: cur.bench, ranAt: new Date().toISOString(), score: tot.got / tot.max, byGroup, results }, null, 2));
console.log(`\n  saved ${outFile}\n`);
process.exit(failed.length ? 1 : 0);
