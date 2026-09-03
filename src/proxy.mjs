#!/usr/bin/env node
// Ossify shim proxy: Claude Code  ->  http://127.0.0.1:<port>  ->  LM Studio /v1/messages
//
// LM Studio's Anthropic-compatible endpoint is strict about the request shape. Claude Code
// 2.1.x sends a few things it rejects or does not know:
//   - a `role: "system"` message inside `messages` (LM Studio: "Invalid discriminator value")
//   - `thinking`, `context_management`, `output_config` top-level fields
//   - POST /v1/messages/count_tokens
// This proxy rewrites those and streams everything else through untouched.
//
//   node src/proxy.mjs [--port 20130] [--target 1234]
import http from "node:http";

const args = process.argv.slice(2);
const opt = (n, d) => { const i = args.indexOf(`--${n}`); return i >= 0 && args[i + 1] !== undefined ? args[i + 1] : d; };
const PORT = parseInt(opt("port", process.env.OSSIFY_PROXY_PORT || "20130"), 10);
const TARGET = parseInt(opt("target", process.env.OSSIFY_LMS_PORT || "1234"), 10);
const DROP_FIELDS = ["thinking", "context_management", "output_config", "betas"];
// Bump when the rewriting logic changes: `ossify up` replaces a running proxy whose version differs.
export const PROXY_VERSION = 3;

// --- conversation isolation ------------------------------------------------------------------
// LM Studio's HTTP API cannot pin a conversation to a prompt-cache slot: its engine logs
// `slot selection: session_id=<empty> server-selected (LCP/LRU)` and then picks a slot by
// longest-common-prefix similarity (threshold 0.100, scored against the CACHED prompt length).
// Claude Code sends the same ~20k-token system prompt in every conversation, so a brand-new
// conversation scores ~0.5-0.99 against a slot still holding a DIFFERENT conversation, takes it,
// and rolls that KV cache back to where the two diverge. Neither model here can shift a KV cache
// (gpt-oss uses sliding-window attention, Qwen3.5 has recurrent SSM layers; the engine logs
// "shifting is not supported for this context"), so those rollbacks depend on context checkpoints
// and can leave recurrent state that does not match the tokens. That is what surfaces as text
// from an unrelated older conversation.
//
// Fix: give each conversation its own prefix. One short marker at the very front of the system
// prompt makes the common prefix between two different conversations a handful of tokens out of
// ~20k, which is far below the 0.100 threshold, so the engine takes a free slot and resets it
// instead of inheriting. Within one conversation the marker is constant, so caching is untouched.
function conversationKey(body) {
  try {
    const uid = body.metadata?.user_id;
    if (typeof uid === "string") {
      const m = uid.match(/"session_id"\s*:\s*"([^"]+)"/);
      if (m) return m[1].slice(0, 16);
      if (uid.length <= 64 && uid.trim()) return uid.trim().slice(0, 16);
    }
  } catch { /* fall through */ }
  // No usable metadata: derive a stable id from the opening user turn, which is fixed for the
  // life of a conversation and differs between conversations.
  const first = Array.isArray(body.messages) ? body.messages.find((m) => m.role === "user") : null;
  const text = typeof first?.content === "string" ? first.content
    : Array.isArray(first?.content) ? first.content.map((b) => b?.text ?? "").join(" ") : "";
  if (!text) return null;
  let h = 0x811c9dc5;
  for (let i = 0; i < text.length; i++) { h ^= text.charCodeAt(i); h = Math.imul(h, 0x01000193) >>> 0; }
  return `fnv${h.toString(36)}`;
}

function isolateConversation(body) {
  const key = conversationKey(body);
  if (!key) return body;
  const marker = `[conversation ${key}]\n`;
  if (typeof body.system === "string") body.system = marker + body.system;
  else if (Array.isArray(body.system) && body.system.length) {
    const head = body.system[0];
    if (head && head.type === "text" && typeof head.text === "string" && !head.text.startsWith("[conversation ")) {
      body.system = [{ ...head, text: marker + head.text }, ...body.system.slice(1)];
    }
  } else if (body.system === undefined) body.system = [{ type: "text", text: marker.trim() }];
  return body;
}

function normalizeMessages(body) {
  if (!Array.isArray(body.messages)) return body;
  const asBlocks = (c) => (typeof c === "string" ? [{ type: "text", text: c }] : Array.isArray(c) ? c : []);
  const out = [];
  for (const m of body.messages) {
    let role = m.role;
    let content = m.content;
    if (role !== "user" && role !== "assistant") {
      // Mid-conversation system/developer notes: keep their position, deliver as a user-side reminder.
      role = "user";
      content = asBlocks(content).map((b) => (b.type === "text" ? { ...b, text: `<system-reminder>\n${b.text}\n</system-reminder>` } : b));
    }
    const prev = out[out.length - 1];
    if (prev && prev.role === role) {
      prev.content = [...asBlocks(prev.content), ...asBlocks(content)]; // strict alternation for chat templates
    } else {
      out.push({ ...m, role, content });
    }
  }
  body.messages = out;
  for (const f of DROP_FIELDS) delete body[f];
  return body;
}

function estimateTokens(body) {
  const text = JSON.stringify(body.system ?? "") + JSON.stringify(body.messages ?? []) + JSON.stringify(body.tools ?? []);
  return Math.ceil(text.length / 3.6);
}

const server = http.createServer((req, res) => {
  if (req.url === "/ossify/health") { res.writeHead(200, { "content-type": "application/json" }); return res.end(JSON.stringify({ ok: true, target: TARGET, pid: process.pid, version: PROXY_VERSION })); }
  if (req.url === "/ossify/quit") { res.writeHead(200, { "content-type": "application/json" }); res.end(JSON.stringify({ ok: true })); return setTimeout(() => process.exit(0), 50); }
  const chunks = [];
  req.on("data", (c) => chunks.push(c));
  req.on("end", () => {
    let bodyBuf = Buffer.concat(chunks);
    const url = req.url ?? "/";
    if (req.method === "POST" && /\/v1\/messages\/count_tokens/.test(url)) {
      let n = 0; try { n = estimateTokens(JSON.parse(bodyBuf.toString("utf8"))); } catch { /* ignore */ }
      res.writeHead(200, { "content-type": "application/json" });
      return res.end(JSON.stringify({ input_tokens: n }));
    }
    if (req.method === "POST" && /\/v1\/messages/.test(url) && bodyBuf.length) {
      try { bodyBuf = Buffer.from(JSON.stringify(isolateConversation(normalizeMessages(JSON.parse(bodyBuf.toString("utf8"))))), "utf8"); } catch { /* forward as-is */ }
    }
    const headers = { ...req.headers, host: `127.0.0.1:${TARGET}`, "content-length": String(bodyBuf.length) };
    delete headers["accept-encoding"]; // keep SSE plain
    const up = http.request({ host: "127.0.0.1", port: TARGET, method: req.method, path: url, headers }, (r) => {
      res.writeHead(r.statusCode ?? 502, r.headers);
      r.pipe(res);
    });
    up.on("error", (e) => {
      res.writeHead(502, { "content-type": "application/json" });
      res.end(JSON.stringify({ type: "error", error: { type: "api_error", message: `ossify proxy: LM Studio unreachable on :${TARGET} (${e.message})` } }));
    });
    // Abort upstream only when the CLIENT goes away mid-response. Note: `req.on("close")` is wrong
    // here - Node fires it as soon as the request body has been fully read, which would kill the
    // upstream socket before LM Studio ever replies ("socket hang up").
    res.on("close", () => { if (!res.writableFinished && !up.destroyed) up.destroy(); });
    up.end(bodyBuf);
  });
});
server.keepAliveTimeout = 0;
server.headersTimeout = 0;
server.requestTimeout = 0;

// Only listen when run directly - cli.mjs imports this module just to read PROXY_VERSION.
const self = process.argv[1] && `file:///${process.argv[1].replace(/\\/g, "/")}`;
if (import.meta.url === self) {
  server.listen(PORT, "127.0.0.1", () => console.log(`[ossify-proxy] listening on http://127.0.0.1:${PORT} -> LM Studio :${TARGET}`));
}
