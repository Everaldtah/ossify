# Ossify

**Claude Code on your local LM Studio models, sized for your GPU.**
Type `gptoss` and Claude Code runs against **gpt-oss-20b**; type `qwen35` and it runs against
**Qwen3.5-35B-A3B**. Both models are far larger than a 6 GB card, yet Ossify runs them at
usable speed without freezing the PC, because it places the model deliberately instead of
letting the default "auto offload" spill VRAM into system RAM.

```
PS> gptoss                      # load gpt-oss-20b (tuned for this PC) and start Claude Code on it
PS> qwen35 -p "explain src/"    # same for Qwen3.5-35B-A3B; normal claude args pass through
PS> gptoss --oss-status         # server / VRAM / RAM / loaded model / tuned profile
PS> gptoss --oss-unload         # free VRAM and RAM
```

Plain `claude` is untouched: the provider environment is set only for the child process and
restored afterwards.

## Why a loader at all

Mixture-of-experts models such as gpt-oss-20b (12 GB, 24 layers x 32 experts, 4 active) and
Qwen3.5-35B-A3B (21 GB, 40 layers x 256 experts, 8 active) are mostly *expert* weights that are
touched sparsely: only 4 of 32 (or 8 of 256) experts run per token. Everything else - attention,
norms, routers, the output head, the KV cache - runs every token and wants to be on the GPU.

On a 6 GB card the default LM Studio load puts *whole layers* on the GPU until it is "full",
which in practice over-commits VRAM. The driver then pages VRAM over PCIe and generation drops
to ~3 tok/s while RAM balloons (measured here: 19 GB working set for an 11 GB model, 2 GB RAM
left). Ossify instead:

1. **Reads the GGUF header** (`src/gguf.mjs`) and sizes dense vs. expert tensors and the KV
   cache exactly for the requested context, including LM Studio quirks (full-size sliding-window
   KV for gpt-oss, hybrid attention layers for Qwen3.5).
2. **Plans** (`src/plan.mjs`) against *live* free VRAM and RAM: all dense weights + KV on GPU,
   the minimum number of expert layers pushed to system RAM (llama.cpp `n-cpu-moe`, LM Studio
   `numCpuExpertLayersRatio`), q8_0 KV cache, flash attention, mmap.
3. **Auto-tunes once** (`ossify tune`): loads each candidate placement, checks for VRAM spill and
   RAM exhaustion, benchmarks prompt-processing and generation speed, and saves the winner per
   machine / model / context in `~/.ossify/`.
4. **Refuses to load** when the plan would eat into the RAM headroom (default 4 GB), and tells
   you how much to free instead of freezing the machine.
5. **Hands the model to Claude Code** through LM Studio's native Anthropic-compatible
   `/v1/messages` endpoint, behind a 100-line shim proxy (`src/proxy.mjs`, port 20130) that
   fixes the few request shapes LM Studio rejects: Claude Code 2.1.x puts a `role: "system"`
   reminder inside `messages` (LM Studio answers `Invalid discriminator value`), sends
   `thinking` / `context_management` / `output_config`, and calls `/v1/messages/count_tokens`.
   The proxy folds system-role messages into user-side `<system-reminder>` blocks (keeping strict
   user/assistant alternation for chat templates), drops the unknown fields, estimates token
   counts, and streams everything else through untouched.

### The SDK gotcha this project works around

`@lmstudio/sdk` 1.5.0 silently drops `gpu.numCpuExpertLayersRatio` when it serialises the load
config (confirmed by capturing the websocket message: only contextLength, offloadRatio,
gpuSplitConfig and flashAttention were sent). LM Studio 0.4.2 itself supports the field
(llama.cpp engine >= 1.46). `src/lmstudio.mjs` appends the raw KV fields
(`llm.load.numCpuExpertLayersRatio`, KV-cache quantisation, batch size) to the outgoing
`loadModel` message, and the server log then shows `Num CPU Expert Layers: 20`.

## Measured on the reference machine

Ryzen 5 5500 (6c/12t), 32 GB DDR4-3200, RTX 3050 6 GB, LM Studio 0.4.2, llama.cpp CUDA 12 engine 2.14.

| Model | Placement | VRAM used | RAM used | Prefill | Generation |
|---|---|---|---|---|---|
| gpt-oss-20b, 64k ctx, LM Studio default auto-offload | 25/25 layers on GPU (spills) | 5.9 / 6.1 GB | ~19 GB | 137 tok/s | **3.2 tok/s** |
| gpt-oss-20b, 64k ctx, classic layer split (5/24) | whole layers | 3.6 GB | ~10 GB | 275 tok/s | 14.1 tok/s |
| gpt-oss-20b, 64k ctx, **Ossify** | dense + 3 expert layers on GPU, 21 expert layers in RAM, q8 KV | 5.0 GB | ~10 GB | **290 tok/s** | **20.1 tok/s** |
| gpt-oss-20b, 64k ctx, Ossify `--oss-deep` q4 KV | dense + 5 expert layers on GPU | 5.1 GB | ~10 GB | 304 tok/s | 21.9 tok/s (quality tax, not chosen) |
| Qwen3.5-35B-A3B, 32k ctx, classic layer split (5/40) | whole layers | 4.7 GB | ~19 GB | 131 tok/s | 11.4 tok/s |
| Qwen3.5-35B-A3B, 32k ctx, **Ossify** | dense + vision projector + 3 expert layers on GPU, 37 expert layers in RAM, q8 KV | 5.6 GB | ~19 GB | **147 tok/s** | **27.7 tok/s** |

Qwen generates faster than gpt-oss despite being twice the size: 8 of 256 experts per token read
fewer bytes from RAM than gpt-oss's 4 of 32. Prefill is the opposite - CPU-resident experts are
streamed through the GPU per 512-token micro-batch, so bigger expert tables cost more per batch.
Claude Code's first turn (~20k tokens of system prompt and tools) therefore takes about a
minute; later turns reuse LM Studio's prompt cache and only process the delta.

`gptoss --oss-status` prints the numbers measured on *your* machine.

## Install

Requirements: Windows, Node 20+, [Claude Code](https://claude.com/claude-code), [LM Studio](https://lmstudio.ai)
0.4.x with the models downloaded (`openai/gpt-oss-20b`, `qwen/qwen3.5-35b-a3b`), NVIDIA GPU (CPU-only also plans, slowly).

```powershell
git clone https://github.com/Everaldtah/ossify.git $env:USERPROFILE\ossify
powershell -ExecutionPolicy Bypass -File $env:USERPROFILE\ossify\install.ps1
```

The installer runs `npm install`, dot-sources `bin/ossify.ps1` from your PowerShell profile
(backup kept), and drops `gptoss.cmd` / `qwen35.cmd` into `~\.local\bin` for cmd.exe and other
terminals. Open a new terminal afterwards.

## Commands

| Command | What it does |
|---|---|
| `gptoss` / `qwen35` | ensure LM Studio server, unload other models, load the tuned placement, start Claude Code |
| `... --oss-status` | server state, GPU/RAM usage, loaded model, tuned profile |
| `... --oss-tune` / `--oss-deep` | re-run the auto-tuner (deep = more candidates: q4 KV, KV-in-RAM, classic layer split) |
| `... --oss-plan` | print the placement plan without loading |
| `... --oss-bench` | benchmark whatever is loaded |
| `... --oss-unload` | unload everything |
| `... --oss-ctx N` | different context length (default 65536 for gpt-oss, 32768 for Qwen) |
| `... --oss-ttl SEC` | idle auto-unload (default 1800 s; `0` = keep loaded) |
| `... --oss-quick` | first run without tuning (planner default) |
| `... --oss-retune` | ignore the saved profile |
| `... --oss-doctor` | environment dump |

Anything else is passed to `claude` unchanged (`-p`, `--continue`, `--resume`, ...).

Direct CLI: `node src/cli.mjs <up|tune|plan|status|unload|bench|doctor> [--model KEY] [--ctx N] [--ttl S] [--ram-margin GB] [--deep] [--quick] [--retune]`.

### Adding another model

Any GGUF in LM Studio works: `node src/cli.mjs plan --model <lmstudio-key>`. Dense models get the
classic layer split; MoE models get expert offload. To add a launcher, copy one line in
`bin/ossify.ps1`:

```powershell
function mymodel { Start-OssifyClaude 'mymodel' 'publisher/model-key' 32768 4 $args }
#                                     name       LM Studio model key   ctx   RAM headroom GB
```

## How Claude Code is wired

The launcher sets, for the child process only:

- `ANTHROPIC_BASE_URL=http://127.0.0.1:1234`, `ANTHROPIC_AUTH_TOKEN=lm-studio`
- `ANTHROPIC_MODEL` and every `ANTHROPIC_DEFAULT_*_MODEL` / `CLAUDE_CODE_SUBAGENT_MODEL` to the loaded model's identifier (so LM Studio never JIT-loads a second copy with default settings)
- `CLAUDE_CODE_AUTO_COMPACT_WINDOW = context - 8192`, `CLAUDE_CODE_MAX_OUTPUT_TOKENS=8192`, `API_TIMEOUT_MS=3600000`
- clears `ANTHROPIC_API_KEY`, `CLAUDE_CODE_USE_OPENAI`, `OPENAI_*`, Bedrock/Vertex switches

## Keeping the machine usable

- VRAM budget = live free VRAM - 450 MiB (compositor, browsers, overlays).
- RAM budget = live free RAM - headroom (4 GB default, 2 GB for the Qwen launcher). The loader
  refuses rather than swaps.
- Models auto-unload after 30 min idle (`--oss-ttl`), so a game launched later gets its VRAM back.
- The tuner never benchmarks a spilled load: if fewer than 200 MiB of VRAM remain after loading,
  the candidate is rejected immediately.

## Layout

```
bin/ossify.ps1   gptoss / qwen35 launcher functions (dot-sourced from the profile)
bin/run.ps1      entry for the .cmd shims
src/cli.mjs      up | tune | plan | status | unload | bench | doctor
src/plan.mjs     VRAM/RAM planner and candidate generator
src/gguf.mjs     GGUF header reader (tensor sizes, KV geometry)
src/lmstudio.mjs LM Studio driver (server, resolve, load with raw-field injection, benchmark)
src/sys.mjs      hardware probe (nvidia-smi, RAM, CPU, LM Studio settings)
install.ps1      installer
```

MIT.
