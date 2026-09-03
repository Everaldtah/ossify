#!/usr/bin/env node
// Ossify web tools - a stdio MCP server giving a LOCAL model real internet access inside
// Claude Code.
//
// Why this exists: Claude Code's built-in WebSearch is executed server-side by Anthropic, so it
// returns nothing when ANTHROPIC_BASE_URL points at LM Studio. MCP tools are different: Claude
// Code runs them on this machine and feeds the result back as tool output, so they work with any
// backend, local models included.
//
// Design notes for small models:
//  - Results are trimmed hard. Every token fed back costs prefill time (gpt-oss does ~290 tok/s,
//    so 5k tokens of page text is ~17s before the model even starts answering).
//  - `web_research` does search + fetch in ONE call. Each extra tool round-trip on a local model
//    costs a full turn, so collapsing four calls into one is a large win.
//
// Search backends, first available wins:
//   BRAVE_API_KEY -> Brave Search API      (best quality)
//   TAVILY_API_KEY -> Tavily               (built for LLMs)
//   SERPER_API_KEY -> Serper (Google)
//   none -> DuckDuckGo HTML (keyless, default)
import { createInterface } from "node:readline";

const UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0 Safari/537.36";
const FETCH_TIMEOUT = Number(process.env.OSSIFY_WEB_TIMEOUT_MS || 20000);
const MAX_BYTES = 3_000_000; // stop reading a page after this, before parsing

const log = (...m) => console.error("[ossify-web]", ...m);

/* ------------------------------------------------------------------ http */

async function httpGet(url, { accept = "text/html,application/xhtml+xml,*/*;q=0.8", headers = {} } = {}) {
  const res = await fetch(url, {
    redirect: "follow",
    signal: AbortSignal.timeout(FETCH_TIMEOUT),
    headers: { "user-agent": UA, accept, "accept-language": "en-US,en;q=0.9", ...headers },
  });
  const type = (res.headers.get("content-type") || "").toLowerCase();
  const reader = res.body?.getReader();
  let bytes = 0;
  const chunks = [];
  if (reader) {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      bytes += value.length;
      chunks.push(value);
      if (bytes > MAX_BYTES) { await reader.cancel(); break; }
    }
  }
  const buf = Buffer.concat(chunks.map((c) => Buffer.from(c)));
  return { status: res.status, ok: res.ok, type, url: res.url, body: buf.toString("utf8"), bytes };
}

/* ------------------------------------------------------- html -> readable text */

const ENTITIES = { amp: "&", lt: "<", gt: ">", quot: '"', apos: "'", nbsp: " ", "#39": "'", "#x27": "'", "#x2F": "/", "#47": "/", mdash: "-", ndash: "-", hellip: "...", rsquo: "'", lsquo: "'", ldquo: '"', rdquo: '"' };
function decodeEntities(s) {
  return s.replace(/&(#x?[0-9a-f]+|[a-z]+);/gi, (m, e) => {
    const k = e.toLowerCase();
    if (ENTITIES[k]) return ENTITIES[k];
    if (k[0] === "#") {
      const n = k[1] === "x" ? parseInt(k.slice(2), 16) : parseInt(k.slice(1), 10);
      if (Number.isFinite(n) && n > 0 && n < 0x10ffff) { try { return String.fromCodePoint(n); } catch { return m; } }
    }
    return m;
  });
}

/** Strip a page down to the text a reader would actually see. No dependencies on purpose. */
export function htmlToText(html) {
  let s = html;
  const title = (s.match(/<title[^>]*>([\s\S]*?)<\/title>/i)?.[1] || "").trim();
  // Prefer the main content region when the page marks one.
  const main = s.match(/<(?:main|article)\b[^>]*>([\s\S]*?)<\/(?:main|article)>/i);
  if (main && main[1].length > 500) s = main[1];
  s = s.replace(/<!--[\s\S]*?-->/g, "");
  s = s.replace(/<(script|style|noscript|svg|canvas|iframe|form|template)\b[^>]*>[\s\S]*?<\/\1>/gi, " ");
  s = s.replace(/<(nav|header|footer|aside)\b[^>]*>[\s\S]*?<\/\1>/gi, " ");
  s = s.replace(/<\/(p|div|section|li|tr|h[1-6]|blockquote|pre)>/gi, "\n");
  s = s.replace(/<br\s*\/?>/gi, "\n");
  s = s.replace(/<li\b[^>]*>/gi, "\n- ");
  s = s.replace(/<h([1-6])\b[^>]*>/gi, (m, n) => "\n" + "#".repeat(Number(n)) + " ");
  s = s.replace(/<[^>]+>/g, " ");
  s = decodeEntities(s);
  s = s.replace(/[ \t ]+/g, " ").replace(/ *\n */g, "\n").replace(/\n{3,}/g, "\n\n").trim();
  // Tag stripping strands list markers on their own line, so a nav menu reads as alternating
  // bullets and words and escapes the run detector below. Reattach each marker to its text.
  s = s.replace(/(^|\n)-[ \t]*\n+/g, "$1- ");
  return { title: decodeEntities(title), text: denoise(s) };
}

/**
 * Drop navigation chrome. Menus survive tag-stripping as long runs of very short lines with no
 * sentence punctuation ("Blog", "Dev", "Pricing", "- About"), which would otherwise eat the
 * context budget a local model badly needs for actual content.
 */
export function denoise(text) {
  const lines = text.split("\n");
  const isShort = (l) => {
    const t = l.replace(/^[-#\s]+/, "").trim();
    return t.length > 0 && t.length < 32 && !/[.!?:;]$/.test(t) && t.split(/\s+/).length <= 4;
  };
  const keep = new Array(lines.length).fill(true);
  for (let i = 0; i < lines.length;) {
    if (lines[i].startsWith("#") || !isShort(lines[i])) { i++; continue; }
    let j = i;
    while (j < lines.length && !lines[j].startsWith("#") && (isShort(lines[j]) || !lines[j].trim())) j++;
    const run = lines.slice(i, j).filter((l) => l.trim()).length;
    if (run >= 4) for (let k = i; k < j; k++) keep[k] = false; // a menu, not prose
    i = j > i ? j : i + 1;
  }
  const out = [];
  let prev = null;
  for (let i = 0; i < lines.length; i++) {
    if (!keep[i]) continue;
    const l = lines[i];
    if (l.trim() && l.trim() === prev) continue; // repeated boilerplate
    if (/^\s*(accept all cookies|subscribe|advertisement|sign in|log in|share this)\s*$/i.test(l)) continue;
    out.push(l);
    if (l.trim()) prev = l.trim();
  }
  return out.join("\n").replace(/\n{3,}/g, "\n\n").trim();
}

/* ---------------------------------------------------------------- search */

function ddgUnwrap(href) {
  // DuckDuckGo wraps results as //duckduckgo.com/l/?uddg=<encoded>&rut=...
  const m = href.match(/[?&]uddg=([^&]+)/);
  if (m) { try { return decodeURIComponent(m[1]); } catch { /* fall through */ } }
  if (href.startsWith("//")) return "https:" + href;
  return href;
}

async function searchDuckDuckGo(query, count) {
  const url = `https://html.duckduckgo.com/html/?q=${encodeURIComponent(query)}`;
  const r = await httpGet(url);
  const out = [];
  const re = /<a[^>]+class="[^"]*result__a[^"]*"[^>]+href="([^"]+)"[^>]*>([\s\S]*?)<\/a>/gi;
  let m;
  while ((m = re.exec(r.body)) && out.length < count) {
    const link = ddgUnwrap(m[1]);
    if (!/^https?:\/\//i.test(link)) continue;
    const title = decodeEntities(m[2].replace(/<[^>]+>/g, "")).trim();
    // The snippet sits in the next result__snippet block after this anchor.
    const rest = r.body.slice(m.index, m.index + 4000);
    const sm = rest.match(/class="[^"]*result__snippet[^"]*"[^>]*>([\s\S]*?)<\/a>/i);
    const snippet = sm ? decodeEntities(sm[1].replace(/<[^>]+>/g, "")).replace(/\s+/g, " ").trim() : "";
    if (title) out.push({ title, url: link, snippet });
  }
  if (out.length) return out;
  // Fallback: the lite endpoint has a different, simpler shape.
  const lr = await httpGet(`https://lite.duckduckgo.com/lite/?q=${encodeURIComponent(query)}`);
  const lre = /<a[^>]+href="([^"]+)"[^>]*class="[^"]*result-link[^"]*"[^>]*>([\s\S]*?)<\/a>/gi;
  while ((m = lre.exec(lr.body)) && out.length < count) {
    const link = ddgUnwrap(m[1]);
    if (!/^https?:\/\//i.test(link)) continue;
    out.push({ title: decodeEntities(m[2].replace(/<[^>]+>/g, "")).trim(), url: link, snippet: "" });
  }
  return out;
}

async function searchBrave(query, count) {
  const r = await fetch(`https://api.search.brave.com/res/v1/web/search?q=${encodeURIComponent(query)}&count=${count}`, {
    headers: { accept: "application/json", "x-subscription-token": process.env.BRAVE_API_KEY },
    signal: AbortSignal.timeout(FETCH_TIMEOUT),
  });
  if (!r.ok) throw new Error(`Brave ${r.status}`);
  const j = await r.json();
  return (j.web?.results ?? []).slice(0, count).map((x) => ({ title: x.title, url: x.url, snippet: (x.description || "").replace(/<[^>]+>/g, "") }));
}

async function searchTavily(query, count) {
  const r = await fetch("https://api.tavily.com/search", {
    method: "POST", headers: { "content-type": "application/json" }, signal: AbortSignal.timeout(FETCH_TIMEOUT),
    body: JSON.stringify({ api_key: process.env.TAVILY_API_KEY, query, max_results: count, search_depth: "basic" }),
  });
  if (!r.ok) throw new Error(`Tavily ${r.status}`);
  const j = await r.json();
  return (j.results ?? []).slice(0, count).map((x) => ({ title: x.title, url: x.url, snippet: x.content ?? "" }));
}

async function searchSerper(query, count) {
  const r = await fetch("https://google.serper.dev/search", {
    method: "POST", headers: { "content-type": "application/json", "X-API-KEY": process.env.SERPER_API_KEY }, signal: AbortSignal.timeout(FETCH_TIMEOUT),
    body: JSON.stringify({ q: query, num: count }),
  });
  if (!r.ok) throw new Error(`Serper ${r.status}`);
  const j = await r.json();
  return (j.organic ?? []).slice(0, count).map((x) => ({ title: x.title, url: x.link, snippet: x.snippet ?? "" }));
}

export function searchBackend() {
  if (process.env.BRAVE_API_KEY) return { name: "brave", fn: searchBrave };
  if (process.env.TAVILY_API_KEY) return { name: "tavily", fn: searchTavily };
  if (process.env.SERPER_API_KEY) return { name: "serper", fn: searchSerper };
  return { name: "duckduckgo", fn: searchDuckDuckGo };
}

export async function search(query, count = 5) {
  const b = searchBackend();
  try { return { backend: b.name, results: await b.fn(query, count) }; }
  catch (e) {
    if (b.name !== "duckduckgo") { log(`${b.name} failed (${e.message}); falling back to duckduckgo`); return { backend: "duckduckgo", results: await searchDuckDuckGo(query, count) }; }
    throw e;
  }
}

/* ----------------------------------------------------------------- fetch */

export async function readPage(url, { maxChars = 5000, offset = 0 } = {}) {
  if (!/^https?:\/\//i.test(url)) throw new Error("url must start with http:// or https://");
  const r = await httpGet(url);
  if (!r.ok) throw new Error(`HTTP ${r.status} fetching ${url}`);
  let title = "", text = "";
  if (r.type.includes("json")) { title = url; text = r.body; }
  else if (r.type.includes("html") || r.body.trimStart().startsWith("<")) ({ title, text } = htmlToText(r.body));
  else if (r.type.includes("text") || r.type.includes("markdown") || r.type.includes("xml")) { title = url; text = r.body; }
  else throw new Error(`unsupported content type "${r.type || "unknown"}" - this tool reads web pages and text, not binaries or PDFs`);
  const total = text.length;
  const slice = text.slice(offset, offset + maxChars);
  return { url: r.url, title, text: slice, total, offset, truncated: offset + slice.length < total };
}

/* ------------------------------------------------------------ MCP plumbing */

const TOOLS = [
  {
    name: "web_search",
    description: "Search the web and get a numbered list of results (title, URL, snippet). Use this to find pages; use web_fetch to read one.",
    inputSchema: {
      type: "object",
      properties: {
        query: { type: "string", description: "What to search for." },
        count: { type: "integer", description: "How many results, 1-10. Default 5.", minimum: 1, maximum: 10 },
      },
      required: ["query"],
      additionalProperties: false,
    },
  },
  {
    name: "web_fetch",
    description: "Fetch one web page and return its readable text with the HTML stripped. Use after web_search, or on any URL the user gives you.",
    inputSchema: {
      type: "object",
      properties: {
        url: { type: "string", description: "Full URL starting with http:// or https://" },
        max_chars: { type: "integer", description: "Characters of page text to return, 500-20000. Default 5000.", minimum: 500, maximum: 20000 },
        offset: { type: "integer", description: "Start this many characters in, to continue reading a long page. Default 0.", minimum: 0 },
      },
      required: ["url"],
      additionalProperties: false,
    },
  },
  {
    name: "web_research",
    description: "Search the web AND read the top pages in a single step, returning their combined text. Prefer this over separate web_search and web_fetch calls when you need facts from the internet - it is much faster.",
    inputSchema: {
      type: "object",
      properties: {
        query: { type: "string", description: "What to research." },
        pages: { type: "integer", description: "How many top results to read, 1-4. Default 2.", minimum: 1, maximum: 4 },
        chars_per_page: { type: "integer", description: "Characters to keep from each page, 500-10000. Default 3000.", minimum: 500, maximum: 10000 },
      },
      required: ["query"],
      additionalProperties: false,
    },
  },
];

async function callTool(name, a = {}) {
  if (name === "web_search") {
    const count = Math.min(10, Math.max(1, a.count ?? 5));
    const { backend, results } = await search(String(a.query ?? ""), count);
    if (!results.length) return `No results for "${a.query}" (backend: ${backend}). Try different words.`;
    return `Search results for "${a.query}" (via ${backend}):\n\n` +
      results.map((r, i) => `${i + 1}. ${r.title}\n   ${r.url}${r.snippet ? `\n   ${r.snippet.slice(0, 300)}` : ""}`).join("\n\n") +
      `\n\nUse web_fetch on a URL above to read the full page.`;
  }

  if (name === "web_fetch") {
    const p = await readPage(String(a.url ?? ""), {
      maxChars: Math.min(20000, Math.max(500, a.max_chars ?? 5000)),
      offset: Math.max(0, a.offset ?? 0),
    });
    return `# ${p.title || p.url}\nURL: ${p.url}\n` +
      `(${p.text.length} of ${p.total} characters${p.offset ? `, starting at ${p.offset}` : ""})\n\n${p.text}` +
      (p.truncated ? `\n\n[truncated - call web_fetch again with offset=${p.offset + p.text.length} to continue]` : "");
  }

  if (name === "web_research") {
    const pages = Math.min(4, Math.max(1, a.pages ?? 2));
    const per = Math.min(10000, Math.max(500, a.chars_per_page ?? 3000));
    const { backend, results } = await search(String(a.query ?? ""), Math.max(pages + 2, 5));
    if (!results.length) return `No results for "${a.query}" (backend: ${backend}).`;
    const out = [`Research for "${a.query}" (via ${backend})`];
    let read = 0;
    for (const r of results) {
      if (read >= pages) break;
      try {
        const p = await readPage(r.url, { maxChars: per });
        read++;
        out.push(`\n---\n## Source ${read}: ${p.title || r.title}\nURL: ${p.url}\n\n${p.text}${p.truncated ? "\n[page truncated]" : ""}`);
      } catch (e) {
        out.push(`\n---\n## (skipped ${r.url}: ${e.message})`);
      }
    }
    if (!read) return out.join("\n") + "\n\nCould not read any of the results.";
    const rest = results.slice(0, 6).map((r, i) => `${i + 1}. ${r.title} - ${r.url}`).join("\n");
    return out.join("\n") + `\n\n---\nOther results not read:\n${rest}`;
  }

  throw new Error(`unknown tool: ${name}`);
}

/* ------------------------------------------------------- JSON-RPC over stdio */

function send(msg) { process.stdout.write(JSON.stringify(msg) + "\n"); }
const reply = (id, result) => send({ jsonrpc: "2.0", id, result });
const fail = (id, code, message) => send({ jsonrpc: "2.0", id, error: { code, message } });

async function handle(msg) {
  const { id, method, params } = msg;
  if (method === "initialize") {
    return reply(id, {
      // Echo the client's protocol version when it names one, so we stay compatible as it moves.
      protocolVersion: params?.protocolVersion || "2025-06-18",
      capabilities: { tools: {} },
      serverInfo: { name: "ossify-web", version: "1.0.0" },
    });
  }
  if (method === "notifications/initialized" || method === "notifications/cancelled") return;
  if (method === "ping") return reply(id, {});
  if (method === "tools/list") return reply(id, { tools: TOOLS });
  if (method === "resources/list") return reply(id, { resources: [] });
  if (method === "prompts/list") return reply(id, { prompts: [] });
  if (method === "tools/call") {
    const name = params?.name;
    try {
      const text = await callTool(name, params?.arguments ?? {});
      return reply(id, { content: [{ type: "text", text }] });
    } catch (e) {
      // Report tool failures as tool results, not protocol errors, so the model can react.
      return reply(id, { content: [{ type: "text", text: `${name} failed: ${e.message}` }], isError: true });
    }
  }
  if (id !== undefined) fail(id, -32601, `method not found: ${method}`);
}

const selfUrl = process.argv[1] && `file:///${process.argv[1].replace(/\\/g, "/")}`;
if (import.meta.url === selfUrl) {
  if (process.argv[2] === "--selftest") {
    const q = process.argv[3] || "what is the rust borrow checker";
    console.log(`backend: ${searchBackend().name}`);
    const r = await callTool("web_search", { query: q, count: 3 });
    console.log(r.slice(0, 900));
    console.log("\n--- web_research ---");
    console.log((await callTool("web_research", { query: q, pages: 1, chars_per_page: 700 })).slice(0, 1200));
    process.exit(0);
  }
  createInterface({ input: process.stdin }).on("line", async (line) => {
    const s = line.trim();
    if (!s) return;
    let msg;
    try { msg = JSON.parse(s); } catch { return; }
    try { await handle(msg); } catch (e) { if (msg?.id !== undefined) fail(msg.id, -32603, String(e.message || e)); }
  });
  log(`ready (search backend: ${searchBackend().name})`);
}
