// Minimal GGUF v2/v3 header reader: metadata + tensor table (no tensor data).
// Used to size dense vs. expert weights and the KV cache before loading a model.
import { open } from "node:fs/promises";

const T = { UINT8: 0, INT8: 1, UINT16: 2, INT16: 3, UINT32: 4, INT32: 5, FLOAT32: 6, BOOL: 7, STRING: 8, ARRAY: 9, UINT64: 10, INT64: 11, FLOAT64: 12 };

// ggml type id -> [blockSize, bytesPerBlock]
const GGML = {
  0: [1, 4], 1: [1, 2], 2: [32, 18], 3: [32, 20], 6: [32, 22], 7: [32, 24], 8: [32, 34],
  9: [32, 36], 10: [256, 84], 11: [256, 110], 12: [256, 144], 13: [256, 176], 14: [256, 210],
  15: [256, 292], 16: [256, 66], 17: [256, 74], 18: [256, 56], 19: [256, 110], 20: [256, 50],
  21: [256, 82], 22: [256, 54], 23: [256, 137], 24: [1, 1], 25: [1, 2], 26: [1, 4], 27: [1, 8],
  28: [1, 8], 29: [256, 90], 30: [1, 2], 34: [128, 50], 35: [128, 64], 36: [128, 80], 37: [128, 144],
  38: [128, 192], 39: [32, 17],
};

class Reader {
  constructor(fd) { this.fd = fd; this.pos = 0n; this.buf = Buffer.alloc(1 << 20); this.bufStart = 0n; this.bufLen = 0; }
  async ensure(n) {
    const rel = Number(this.pos - this.bufStart);
    if (rel >= 0 && rel + n <= this.bufLen) return rel;
    const { bytesRead } = await this.fd.read(this.buf, 0, this.buf.length, Number(this.pos));
    this.bufStart = this.pos; this.bufLen = bytesRead;
    if (bytesRead < n) throw new Error("gguf: unexpected EOF");
    return 0;
  }
  async u8() { const o = await this.ensure(1); this.pos += 1n; return this.buf.readUInt8(o); }
  async u16() { const o = await this.ensure(2); this.pos += 2n; return this.buf.readUInt16LE(o); }
  async u32() { const o = await this.ensure(4); this.pos += 4n; return this.buf.readUInt32LE(o); }
  async i32() { const o = await this.ensure(4); this.pos += 4n; return this.buf.readInt32LE(o); }
  async u64() { const o = await this.ensure(8); this.pos += 8n; return this.buf.readBigUInt64LE(o); }
  async i64() { const o = await this.ensure(8); this.pos += 8n; return this.buf.readBigInt64LE(o); }
  async f32() { const o = await this.ensure(4); this.pos += 4n; return this.buf.readFloatLE(o); }
  async f64() { const o = await this.ensure(8); this.pos += 8n; return this.buf.readDoubleLE(o); }
  async str() {
    const len = Number(await this.u64());
    if (len > this.buf.length) { this.pos += BigInt(len); return `<${len} bytes>`; }
    const o = await this.ensure(len); this.pos += BigInt(len);
    return this.buf.toString("utf8", o, o + len);
  }
  async value(type) {
    switch (type) {
      case T.UINT8: return this.u8();
      case T.INT8: return ((await this.u8()) << 24) >> 24;
      case T.UINT16: return this.u16();
      case T.INT16: return ((await this.u16()) << 16) >> 16;
      case T.UINT32: return this.u32();
      case T.INT32: return this.i32();
      case T.FLOAT32: return this.f32();
      case T.BOOL: return (await this.u8()) !== 0;
      case T.STRING: return this.str();
      case T.UINT64: return Number(await this.u64());
      case T.INT64: return Number(await this.i64());
      case T.FLOAT64: return this.f64();
      case T.ARRAY: {
        const et = await this.u32(); const n = Number(await this.u64());
        if (n > 4096) { for (let i = 0; i < n; i++) await this.value(et); return `<array ${n}>`; }
        const out = []; for (let i = 0; i < n; i++) out.push(await this.value(et)); return out;
      }
      default: throw new Error(`gguf: unknown value type ${type}`);
    }
  }
}

export async function readGguf(path) {
  const fd = await open(path, "r");
  try {
    const r = new Reader(fd);
    const magic = await r.u32();
    if (magic !== 0x46554747) throw new Error("not a GGUF file");
    const version = await r.u32();
    if (version < 2) throw new Error(`gguf v${version} unsupported`);
    const nTensors = Number(await r.u64());
    const nKv = Number(await r.u64());
    const meta = {};
    for (let i = 0; i < nKv; i++) { const k = await r.str(); const t = await r.u32(); meta[k] = await r.value(t); }
    const tensors = [];
    for (let i = 0; i < nTensors; i++) {
      const name = await r.str(); const nd = await r.u32();
      let elems = 1n; for (let d = 0; d < nd; d++) elems *= await r.u64();
      const type = await r.u32(); await r.u64();
      const [bs, ts] = GGML[type] ?? [1, 2];
      tensors.push({ name, type, bytes: (Number(elems) / bs) * ts });
    }
    return { version, meta, tensors };
  } finally { await fd.close(); }
}

export function summarize({ meta, tensors }) {
  const arch = meta["general.architecture"];
  const g = (k) => meta[`${arch}.${k}`];
  const isExpert = (n) => /_exps\./.test(n);
  let expertBytes = 0, denseBytes = 0, embedBytes = 0;
  for (const t of tensors) {
    if (isExpert(t.name)) expertBytes += t.bytes;
    else if (/^token_embd|^output\.weight|^output_norm/.test(t.name)) embedBytes += t.bytes;
    else denseBytes += t.bytes;
  }
  const nLayer = g("block_count");
  const nHead = g("attention.head_count");
  const nHeadKvRaw = g("attention.head_count_kv") ?? nHead;
  const nHeadKv = Array.isArray(nHeadKvRaw) ? nHeadKvRaw[0] : nHeadKvRaw;
  const nEmbd = g("embedding_length");
  const keyLen = g("attention.key_length") ?? Math.floor(nEmbd / nHead);
  const valLen = g("attention.value_length") ?? keyLen;
  const swa = g("attention.sliding_window") ?? 0;
  const swaPattern = g("attention.sliding_window_pattern") ?? (swa ? 2 : 0);
  let fullLayers = nLayer, swaLayers = 0;
  if (swa && swaPattern) { swaLayers = Math.floor(nLayer / swaPattern); fullLayers = nLayer - swaLayers; }
  // Hybrid models (Qwen3.5 / Qwen3-Next): only every Nth layer is full attention; the rest are
  // linear-attention (SSM/DeltaNet) layers with a tiny fixed recurrent state instead of a KV cache.
  const fai = g("full_attention_interval");
  if (fai && fai > 1) { fullLayers = Math.floor(nLayer / fai); swaLayers = 0; }
  return {
    arch, name: meta["general.name"], nLayer, nExpert: g("expert_count") ?? 0, nExpertUsed: g("expert_used_count") ?? 0,
    ctxTrain: g("context_length"), nEmbd, nHead, nHeadKv, keyLen, valLen, swa, fullLayers, swaLayers,
    expertBytes, denseBytes, embedBytes, totalBytes: expertBytes + denseBytes + embedBytes,
    expertBytesPerLayer: nLayer ? expertBytes / nLayer : 0, denseBytesPerLayer: nLayer ? denseBytes / nLayer : 0,
    fileType: meta["general.file_type"],
  };
}

// KV cache bytes for a context; bytes/elem: 2 = f16, 1.0625 = q8_0, 0.5625 = q4_0.
// swaFull: LM Studio applies a legacy `swa_full=true` for sliding-window archs (gpt-oss), which
// allocates the sliding-window layers at full context size too (verified in its server log).
export function kvBytes(s, ctx, kB = 2, vB = 2, swaFull = true) {
  const perTok = s.nHeadKv * (s.keyLen * kB + s.valLen * vB);
  const full = s.fullLayers * perTok * ctx;
  const swa = s.swaLayers * perTok * (swaFull ? ctx : Math.min(ctx, (s.swa || ctx) + 256));
  return full + swa;
}

const self = process.argv[1] && `file:///${process.argv[1].replace(/\\/g, "/")}`;
if (import.meta.url === self) {
  const g = await readGguf(process.argv[2]);
  const s = summarize(g);
  const gb = (b) => (b / 1e9).toFixed(3) + " GB";
  console.log(JSON.stringify({ ...s, expertBytes: gb(s.expertBytes), denseBytes: gb(s.denseBytes), embedBytes: gb(s.embedBytes), totalBytes: gb(s.totalBytes), expertBytesPerLayer: gb(s.expertBytesPerLayer), denseBytesPerLayer: gb(s.denseBytesPerLayer), kv32k_f16: gb(kvBytes(s, 32768)), kv64k_f16: gb(kvBytes(s, 65536)), kv64k_q8: gb(kvBytes(s, 65536, 1.0625, 1.0625)), kv128k_q8: gb(kvBytes(s, 131072, 1.0625, 1.0625)) }, null, 2));
  const byType = {}; for (const t of g.tensors) byType[t.type] = (byType[t.type] ?? 0) + t.bytes;
  console.log("bytes by ggml type:", Object.fromEntries(Object.entries(byType).map(([k, v]) => [k, gb(v)])));
  console.log("layer-0 tensors:\n  " + g.tensors.filter(t => /^blk\.0\./.test(t.name)).map(t => `${t.name}:${t.type}:${(t.bytes / 1e6).toFixed(1)}MB`).join("\n  "));
}
