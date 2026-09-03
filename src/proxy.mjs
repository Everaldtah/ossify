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
export const PROXY_VERSION = 2;

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
      try { bodyBuf = Buffer.from(JSON.stringify(normalizeMessages(JSON.parse(bodyBuf.toString("utf8")))), "utf8"); } catch { /* forward as-is */ }
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
