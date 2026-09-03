// Hardware / environment probe. Everything here is read-only.
import os from "node:os";
import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { execFileSync, spawnSync } from "node:child_process";

export const HOME = os.homedir();
export const LMS_DIR = join(HOME, ".lmstudio");
export const LMS_EXE = join(LMS_DIR, "bin", process.platform === "win32" ? "lms.exe" : "lms");
export const OSSIFY_DIR = join(HOME, ".ossify");
export const MiB = 1024 * 1024;
export const GiB = 1024 * MiB;

export function gpu() {
  try {
    const out = execFileSync("nvidia-smi", [
      "--query-gpu=name,memory.total,memory.used,memory.free,driver_version",
      "--format=csv,noheader,nounits",
    ], { encoding: "utf8", timeout: 8000, windowsHide: true });
    const line = out.trim().split(/\r?\n/)[0];
    if (!line) return null;
    const [name, total, used, free, driver] = line.split(",").map((s) => s.trim());
    return { vendor: "nvidia", name, totalMiB: +total, usedMiB: +used, freeMiB: +free, driver };
  } catch {
    return null; // no NVIDIA GPU / nvidia-smi missing -> CPU-only planning
  }
}

export function ram() {
  return { totalBytes: os.totalmem(), freeBytes: os.freemem() };
}

let cpuCache;
export function cpu() {
  if (cpuCache) return cpuCache;
  const logical = os.cpus().length;
  let physical = Math.max(1, Math.floor(logical / 2));
  if (process.platform === "win32") {
    try {
      const r = spawnSync("powershell", ["-NoProfile", "-NonInteractive", "-Command",
        "(Get-CimInstance Win32_Processor | Measure-Object -Sum NumberOfCores).Sum"],
        { encoding: "utf8", timeout: 15000, windowsHide: true });
      const n = parseInt((r.stdout || "").trim(), 10);
      if (n > 0) physical = n;
    } catch { /* keep heuristic */ }
  }
  cpuCache = { model: os.cpus()[0]?.model?.trim() ?? "cpu", logical, physical };
  return cpuCache;
}

export function lmstudio() {
  const settingsPath = join(LMS_DIR, "settings.json");
  let modelsDir = join(LMS_DIR, "models");
  let guardrails = null, defaultCtx = null;
  if (existsSync(settingsPath)) {
    try {
      const s = JSON.parse(readFileSync(settingsPath, "utf8"));
      if (s.downloadsFolder) modelsDir = s.downloadsFolder;
      guardrails = s.modelLoadingGuardrails?.mode ?? null;
      defaultCtx = s.defaultContextLength?.value ?? null;
    } catch { /* ignore */ }
  }
  let port = 1234;
  const httpCfg = join(LMS_DIR, ".internal", "http-server-config.json");
  if (existsSync(httpCfg)) {
    try { port = JSON.parse(readFileSync(httpCfg, "utf8")).port ?? port; } catch { /* ignore */ }
  }
  let backend = null;
  const be = join(LMS_DIR, ".internal", "backend-preferences-v1.json");
  if (existsSync(be)) {
    try { const b = JSON.parse(readFileSync(be, "utf8"))[0]; if (b) backend = `${b.name}@${b.version}`; } catch { /* ignore */ }
  }
  return { installed: existsSync(LMS_EXE), lmsExe: LMS_EXE, modelsDir, port, guardrails, defaultCtx, backend };
}

export function snapshot() {
  return { gpu: gpu(), ram: ram(), cpu: cpu(), lmstudio: lmstudio(), platform: `${process.platform} ${os.release()}` };
}

export const fmtGB = (b) => `${(b / GiB).toFixed(2)} GB`;
export const fmtMiB = (m) => `${Math.round(m)} MiB`;
