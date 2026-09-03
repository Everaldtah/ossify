// VRAM planner: turns (GGUF summary + hardware) into concrete LM Studio load configs.
//
// The key idea for MoE models on small GPUs: keep every dense tensor (attention,
// norms, router, output head) plus the KV cache on the GPU, and push the expert
// blocks - the bulk of the bytes but only 4/32 touched per token - into system RAM.
// llama.cpp exposes this as "n-cpu-moe"; LM Studio as gpu.numCpuExpertLayersRatio.
import { kvBytes } from "./gguf.mjs";
import { MiB, GiB } from "./sys.mjs";

export const KV_BYTES = { f16: 2, q8_0: 1.0625, q4_0: 0.5625 };

// Fixed costs on the GPU that are not weights or KV. Calibrated against LM Studio 0.4.2 /
// llama.cpp engine 2.14 on an RTX 3050 (gpt-oss-20b: CUDA0 compute buffer 398 MiB, ~110 MiB
// runtime context; n_ubatch is pinned at 512 so the compute buffer does not grow with evalBatchSize).
const CUDA_CONTEXT = 160 * MiB;
const MOE_STAGING = 350 * MiB;    // transient scratch while CPU-resident experts are streamed through the GPU during prefill
const computeBuffer = () => 420 * MiB;

export function estimateVram(s, cfg) {
  const kv = cfg.offloadKVCacheToGpu === false ? 0 : kvBytes(s, cfg.contextLength, KV_BYTES[cfg.kvType], KV_BYTES[cfg.kvType]);
  const gpuLayers = Math.round(cfg.gpuRatio * s.nLayer);
  const denseOnGpu = s.denseBytesPerLayer * gpuLayers + (cfg.gpuRatio >= 1 ? s.embedBytes * 0.5 : 0); // output head ~ half of embedBytes
  const expertLayersOnGpu = Math.max(0, gpuLayers - Math.round(cfg.cpuExpertRatio * s.nLayer));
  const experts = s.expertBytesPerLayer * expertLayersOnGpu;
  const staging = expertLayersOnGpu < s.nLayer ? MOE_STAGING : 0;
  const total = CUDA_CONTEXT + denseOnGpu + experts + kv + computeBuffer(cfg.evalBatchSize) + staging;
  return { total, kv, denseOnGpu, experts, expertLayersOnGpu, gpuLayers, staging, compute: computeBuffer(cfg.evalBatchSize) };
}

export function estimateRam(s, cfg, est) {
  // Weights not on the GPU are mmapped into RAM; add the input embedding table and a working margin.
  const cpuWeights = s.totalBytes - est.denseOnGpu - est.experts;
  return cpuWeights + 1.5 * GiB;
}

function ratioFor(layers, nLayer) { return Math.min(1, Math.max(0, layers / nLayer)); }

/**
 * Build candidate configs, best-guess first.
 * budgetBytes: VRAM we are allowed to use (free VRAM minus a safety margin).
 */
export function candidates(s, { budgetBytes, ramFreeBytes, ctxTarget = 65536, minCtx = 16384, deep = false }) {
  const out = [];
  const ctxLadder = [ctxTarget, 98304, 65536, 49152, 32768, 24576, 16384].filter((c, i, a) => c >= minCtx && c <= (s.ctxTrain || 1e9) && c <= ctxTarget && a.indexOf(c) === i).sort((a, b) => b - a);
  const isMoe = s.nExpert > 1 && s.expertBytes > 0;

  const base = { gpuRatio: 1, cpuExpertRatio: 0, contextLength: ctxTarget, kvType: "q8_0", flashAttention: true, evalBatchSize: 512, offloadKVCacheToGpu: true };

  const fits = (cfg) => {
    const est = estimateVram(s, cfg);
    return { ok: est.total <= budgetBytes && estimateRam(s, cfg, est) <= ramFreeBytes, est };
  };

  // Strategy A: everything dense on GPU, minimum number of expert layers on CPU that fits.
  // Deep mode also tries one more layer on the CPU (safer VRAM headroom) and q4_0 KV (more experts on GPU).
  if (isMoe) {
    for (const ctx of ctxLadder) {
      let found = null;
      for (let cpuLayers = 0; cpuLayers <= s.nLayer; cpuLayers++) {
        const cfg = { ...base, contextLength: ctx, cpuExpertRatio: ratioFor(cpuLayers, s.nLayer) };
        const f = fits(cfg);
        if (f.ok) { found = { cfg, est: f.est, cpuLayers }; break; }
      }
      if (!found) continue;
      out.push({ id: `experts-cpu${found.cpuLayers}/${s.nLayer}-c${ctx / 1024}k`, strategy: "experts-cpu", ...found });
      if (deep && found.cpuLayers < s.nLayer) {
        const cfg = { ...base, contextLength: ctx, cpuExpertRatio: ratioFor(found.cpuLayers + 1, s.nLayer) };
        out.push({ id: `experts-cpu${found.cpuLayers + 1}/${s.nLayer}-c${ctx / 1024}k`, strategy: "experts-cpu", cfg, est: estimateVram(s, cfg), cpuLayers: found.cpuLayers + 1 });
      }
      if (deep) {
        for (let cpuLayers = 0; cpuLayers <= s.nLayer; cpuLayers++) {
          const cfg = { ...base, contextLength: ctx, kvType: "q4_0", cpuExpertRatio: ratioFor(cpuLayers, s.nLayer) };
          const f = fits(cfg);
          if (f.ok) { if (cpuLayers < found.cpuLayers) out.push({ id: `experts-cpu${cpuLayers}/${s.nLayer}-kvq4-c${ctx / 1024}k`, strategy: "experts-cpu-kvq4", cfg, est: f.est, cpuLayers }); break; }
        }
      }
      break; // biggest context that fits wins
    }
  }

  // Strategy B: classic partial layer offload (whole layers incl. experts), same ctx as A if possible.
  const ctxB = out[0]?.cfg.contextLength ?? ctxLadder.at(-1);
  for (let layers = s.nLayer; layers >= 0; layers--) {
    const cfg = { ...base, contextLength: ctxB, gpuRatio: ratioFor(layers, s.nLayer), cpuExpertRatio: 0 };
    const f = fits(cfg);
    if (f.ok) { out.push({ id: `layers-${layers}/${s.nLayer}-b512-c${ctxB / 1024}k`, strategy: "layer-split", cfg, est: f.est, gpuLayers: layers }); break; }
  }

  // Strategy C (deep): KV cache in RAM, buys room for more experts on GPU. Usually slower, but measure.
  if (deep && isMoe) {
    for (let cpuLayers = 0; cpuLayers <= s.nLayer; cpuLayers++) {
      const cfg = { ...base, contextLength: ctxB, offloadKVCacheToGpu: false, cpuExpertRatio: ratioFor(cpuLayers, s.nLayer) };
      const f = fits(cfg);
      if (f.ok) { out.push({ id: `experts-cpu${cpuLayers}/${s.nLayer}-kvram-c${ctxB / 1024}k`, strategy: "experts-cpu-kvram", cfg, est: f.est, cpuLayers }); break; }
    }
  }

  // Strategy D: CPU-only fallback (no/insufficient GPU). Still subject to the RAM budget - an
  // empty result means "does not fit in RAM", and the caller must refuse rather than swap.
  if (!out.length) {
    for (const ctx of ctxLadder.length ? ctxLadder : [minCtx]) {
      const cfg = { ...base, gpuRatio: 0, cpuExpertRatio: 0, contextLength: ctx, offloadKVCacheToGpu: false };
      const est = estimateVram(s, cfg);
      if (estimateRam(s, cfg, est) + kvBytes(s, ctx, KV_BYTES[cfg.kvType], KV_BYTES[cfg.kvType]) <= ramFreeBytes) {
        out.push({ id: `cpu-only-c${ctx / 1024}k`, strategy: "cpu-only", cfg, est }); break;
      }
    }
  }
  return out;
}

/** Translate a planner cfg into LM Studio's LLMLoadModelConfig. */
export function toLmsConfig(cfg) {
  const c = {
    gpu: { ratio: cfg.gpuRatio >= 1 ? "max" : cfg.gpuRatio <= 0 ? "off" : cfg.gpuRatio },
    contextLength: cfg.contextLength,
    flashAttention: cfg.flashAttention,
    evalBatchSize: cfg.evalBatchSize,
    offloadKVCacheToGpu: cfg.offloadKVCacheToGpu,
    tryMmap: true,
    keepModelInMemory: cfg.keepModelInMemory ?? false,
  };
  if (cfg.cpuExpertRatio > 0) c.gpu.numCpuExpertLayersRatio = cfg.cpuExpertRatio >= 1 ? "max" : cfg.cpuExpertRatio;
  if (cfg.kvType && cfg.kvType !== "f16") { c.llamaKCacheQuantizationType = cfg.kvType; c.llamaVCacheQuantizationType = cfg.kvType; }
  return c;
}

/** Score = seconds for a representative Claude Code turn: N fresh prompt tokens + M generated. Lower is better. */
export function turnSeconds(bench, promptTokens = 6000, genTokens = 400) {
  if (!bench || !bench.tgTps) return Infinity;
  const pp = bench.ppTps || 1;
  return promptTokens / pp + genTokens / bench.tgTps;
}
