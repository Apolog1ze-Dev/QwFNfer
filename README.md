<div align="center">
  <img alt="QwFNfer: Qwen Four Inference Engine, big models running on small hardware" src="docs/img/header.jpg" width="100%">
  <p><b>Qwen3.8-Flash-Next, a 125B mixture-of-experts model with 512 experts, 111 GB on disk, at 160K context on one 16 GB GPU, 30 GB of RAM and an NVMe.</b></p>
</div>

<p align="center">
| <a href="#getting-started"><b>Getting Started</b></a> | <a href="#results"><b>Results</b></a> | <a href="#how-it-works"><b>How it works</b></a> | <a href="#built-around-the-qwen4-architecture"><b>Qwen4</b></a> | <a href="#faq"><b>FAQ</b></a> | <a href="https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF"><b>Model (Unsloth GGUF)</b></a> |
</p>

Run a **125B** open-weight MoE on the gaming PC you already own, at interactive speed: **15–25 tok/s**.

## Updates

**2026-09-11** — (Experimental)The console now tunes itself to the machine it runs on: it measures the drive, sweeps the CPU thread count live on a long-context run, sizes the RAM tier from the memory the server really needs, defaults the KV cache to q8_0 wherever the plan affords it, and its Stats page is live (prefill progress, input / cached / output tokens). The numbers below were re-measured today with those defaults, and an OpenCode agentic-coding run was added.
Several optimizations and tweaks were introduced including fixing a MTP and vision bug.

**2026-09-11** — Claude Code runs on it: the server now serves the Anthropic Messages API (`POST /v1/messages`, streamed, with `count_tokens`) next to the OpenAI one, so `ANTHROPIC_BASE_URL=http://127.0.0.1:8080` is all it takes. Thinking and tool calls stream as Anthropic blocks, and a replayed conversation continues the engine's prefix as before. **Claude Desktop** runs on it too: `scripts/claude-desktop.sh` (`qwfnfer-claude-desktop` from the bundle) starts the engine if it is not up and launches an instance of Claude Desktop.


## About

qwfnfer is a purpose-built inference engine for Qwen3.8-Flash-Next (GGUF architecture `qwen4exp`): 48 layers, 512 routed experts with top-10 routing, DeltaNet recurrent layers and Qwen Sparse Attention. It is not a llama.cpp fork. It uses ggml's quantized kernels and CUDA backend and llama.cpp's tokenizer, and owns everything above them: the model graph, the memory hierarchy, the expert cache, prefill, the server and the console. Its core features:

- **Three-tier expert runtime**: experts live in VRAM, in pinned RAM and on the NVMe. Reads happen at an expert's natural 0.6–1.2 MB size over io_uring/O_DIRECT (23× the bandwidth of 4 KiB demand paging on the same disk), VRAM-resident experts compute inside replayed CUDA graphs, and the next layer's routing is predicted from the residual and prefetched while the current layer computes.
- **Long context that stays flat**: 163,840 tokens with q8_0 KV on 16 GB (262,144 with the attention state in pinned RAM). Decode attention costs the same 0.55 ms per layer at 4K and at 160K (sparse attention over pooled block keys), and a layer-major prefill streams experts through VRAM at 350–590 tok/s depending on the batch instead of paging them.
- **OpenAI-compatible server, and the Anthropic Messages API for Claude Code**: streaming, thinking with `reasoning_effort` and a thinking budget, tool calling, vision, `/props`, `/stats`, `/slots`, `/metrics` and llama.cpp-style `timings`. Works with Unsloth Studio, Open WebUI or any OpenAI client, and `POST /v1/messages` lets Claude Code run on it with one environment variable.
- **Console**: a local page that finds the downloaded quants (the Hugging Face cache and any folder you add), sizes the flags for *your* GPU and RAM behind four tiers (Chat at 32K context, Agentic coding at 128K, Agentic coding+ at 256K, and a Custom tier you save), auto-tunes them on your hardware (the drive's read rate, a thread sweep on the running server, the KV precision the GPU has room for, the memory the RAM tier can take, then a measured verification), starts and stops the server, chats with it, and shows it live: prefill progress, input / cached / output tokens, tokens/s, the last request and the session's totals.
- **Measured, not projected**: the forward pass is validated bit-exact against llama.cpp, and every number here is a real run on the reference machine, same file, same settings.

## Results

<div align="center">
  <img alt="measured results" src="docs/img/results.png" width=100%>
</div>

Reference machine: RTX 4080 SUPER 16 GB, 30 GB RAM, one NVMe. 163,840-token context and the console's plan for it (KV q8_0, batch 8192, indexer and KV cache in pinned RAM, speculative block, draft head and vision on), with a 15 GB RAM tier asked, 8 CPU threads and 768 MB of VRAM reserve; one run each through the server, measured 2026-09-11. The engine clamps the RAM tier to the memory the machine has: with the draft head's 2.7 GB of pinned experts it built 13.0 GB on Q4 (VRAM expert tier 8.1 GB, 2,591 experts) and 14.1 GB on Q3 (8.7 GB, 3,859 experts), and free memory went down to 1.0 and 0.4 GB at the worst point of the 155K-token prefill. The console's own sizing keeps 3 GB of headroom instead, a tier about 2 GB smaller (a GB of RAM tier is worth about 3% of decode).

| | UD-Q4_K_XL (111 GB) | UD-Q3_K_XL (90 GB) |
|---|---:|---:|
| short chat, decode (thinking on) | 13.2–15.7 tok/s | 19.8–21.2 tok/s |
| 155K-token document: prefill | 355 tok/s (7.3 min) | 348 tok/s (7.4 min) |
| 155K-token document: decode, grounded answer | 12.9 tok/s | 17.7 tok/s |
| one 8,247-token answer at 160K context, sustained | 13.8 tok/s | not measured |
| llama.cpp on the same machine and file (measured 2026-09-06) | 1.4–2.2 tok/s chat, 3 tok/s prefill (Unsloth Studio's defaults) | 4.6–6.3 tok/s at long context |

The prefill runs at the batch the console picks for the context: it streams every expert once per batch and computes per token, so a bigger batch is a faster prefill (on a 43K-token document 4096 gives ~300 tok/s, 8192 ~480, 16384 ~720), and the console takes the largest batch whose VRAM the expert tier can lend while a prompt streams (8192 at 160K on 16 GB). Decode is unchanged either way.

The two options the console exposes, on the same Q4 file and settings; both are on by default:

| UD-Q4_K_XL, 160K context | short chat | 155K-token document, decode |
|---|---:|---:|
| speculative block and draft head off | 13.3–13.5 tok/s | 11.7 tok/s |
| speculative block on, draft head off | 13.4–14.6 tok/s | 13.2 tok/s |
| both on (the console's defaults) | 13.2–15.7 tok/s | 12.9 tok/s |

The speculative block predicts the next layer's experts by running that layer's own block on the residual (95–96% right, against 80% for the router alone) and fetches them a layer early; the output is exact and it is worth 13% on the long document. The draft head is the checkpoint's own next-token predictor, verified by the trunk, so its output is the trunk's own; it pays in chat and in agentic coding (95% of drafts accepted in the OpenCode run below), while on a 155K-token document the tier it costs (0.35 GB of VRAM and 2.7 GB of pinned RAM, which here also meant a 13 GB arena instead of 15) cancels its gain. GPU decode varies about 5% run to run on identical work.

While it serves Q4 with the 13 GB arena: 29 of 31 GB of RAM in use system-wide (the engine 14.8 GB), 15.3 of 16 GB of VRAM, the GPU 50% busy, the engine on 2 of 16 threads.

**Agentic coding through OpenCode.** The same Q4 file driving a real coding harness: [OpenCode](https://opencode.ai) (`opencode run --auto`, tool calls auto-approved) is given a 40-line Python inventory module with two bugs and one missing method and its seven-case test suite, four cases failing, in a throwaway git repository, with this prompt:

> Some tests in test_inventory.py fail. Run the test suite, fix inventory.py so that every test passes without changing the tests, run the suite again to confirm, then reply with one line saying what you changed.

Server flags for this run: 262,144-token context, KV q4_0, a 15 GB RAM tier, 8 CPU threads, batch 16384, 768 MB of VRAM reserve, thinking `xhigh` with a 10,000-token budget, skip-miss off, speculative block on, draft head on, vision on, indexer and KV cache in pinned RAM. The engine built the full 15 GB arena and an 8.5 GB VRAM expert tier (2,730 experts). One run, measured 2026-09-11:

| OpenCode, UD-Q4_K_XL at 256K context | |
|---|---:|
| task solved (7 of 7 tests pass) | yes, in 5 steps and 6 tool calls (bash, bash, read, read, edit, bash) |
| wall time, prompt to final reply | 123 s |
| decode | 947 tokens at 14.0 tok/s |
| prefill | 10,209 new tokens at 188 tok/s; the 7,301-token first turn at 590 tok/s, every later step continuing the cached prefix (895 new tokens on 7,392 reused, 20 on 9,894 …) |
| draft head | 497 verify steps, 88.5% of drafts accepted |
| expert cache | 97.3% hit, 60.4% served from VRAM |

The harness reads the numbers from the server's own `/stats` around the run. What a coding step costs is mostly its prefill: a tool result of a few hundred tokens takes 7–10 s through the batched path, a longer one a full expert sweep (~12 s), and the decode of a 100–300-token tool call 7–20 s on top.


## Getting Started

Linux x86_64 with an NVIDIA GPU (driver 580 or newer) and Python 3. Windows is not supported yet: the engine reads the NVMe through io_uring, and that layer needs a port first.

**1. Install.** One command:

```bash
curl -fsSL https://raw.githubusercontent.com/Apolog1ze-Dev/QwFN/master/scripts/install.sh | bash
```

Or download `qwfnfer-linux-x86_64-cuda.zip` from the [Releases](https://github.com/Apolog1ze-Dev/QwFN/releases) page, unzip it anywhere and run `./qwfnfer`. The bundle carries the engine, the console and every library it needs (ggml, the CUDA runtime, liburing); only the NVIDIA driver comes from your system. The installer puts it under `~/.local/share/qwfnfer` and links `~/.local/bin/qwfnfer`; delete those two paths to uninstall.

**2. Get the model.** The console finds Qwen3.8-Flash-Next GGUFs in your Hugging Face cache. UD-Q4_K_XL is the quality choice, UD-Q3_K_XL the faster one; the `mmproj` file adds vision.

```bash
hf download unsloth/Qwen3.8-Flash-Next-GGUF --include "UD-Q4_K_XL/*" "mmproj-F16.gguf"
```

**3. Run it.**

```bash
qwfnfer
```

It opens http://127.0.0.1:8090. Pick a downloaded quant and a tier: **Chat** (32K context), **Agentic coding** (128K), **Agentic coding+** (256K, the model's full trained context) or **Custom** (anything you set under *Advanced settings* and save). Press **Auto-tune & start**: the console measures the drive under the model (random 2 MiB reads, the pattern of an expert miss), plans every flag for your GPU and RAM with that rate (context; the KV cache at q8_0 whenever the plan can afford it, q4_0 only where it would not fit; the expert tiers, the prefill batch the tier can lend, the reserve, where the attention caches live; vision on when the `mmproj` file is next to the model, the draft head on when its file is there), starts the server, verifies it on a short chat and a 16K–32K-token document with a passphrase planted in it (prefill and decode tokens/s, and whether the answer found the passphrase), sweeps the CPU thread count live on that document's context (the physical cores unless another count measures over 3% faster), and measures the memory the server needs besides its RAM tier through the run, then re-sizes the tier to leave exactly the headroom you set (3 GB by default; a GB of tier is about 3% of decode) and restarts with it. About five minutes; the result is saved per model, the tier card then shows the measured speed instead of the prediction, and every tier for that model uses the measured thread count and drive rate from then on. **Start server** starts with the plan alone; **Self-test** measures a running server. The banner names the model, the tier and every flag it is running with; the Chat, Stats and Log tabs talk to it. Stats is live at one second: the prefill's progress inside a batch with the time left, input / cached / output tokens for the running request, the last request in full (how much of its prompt was reused, prefill and decode speed, why it finished), the session's totals, the cache hit rate and the endpoint. *Model locations* under the model list adds any folder that holds the shards. `qwfnfer --start` starts the last served model and tier as the console comes up. Stop it from the same page.

<div align="center">
  <img alt="qwfn console" src="docs/img/console-serve.png" width=92%>
</div>

**4. Point your tools at it.** The console shows the endpoint, `http://127.0.0.1:8080/v1` by default; any OpenAI-compatible client works with any API key (Unsloth Studio as a custom provider, Open WebUI, your own scripts). What the server accepts:

- Chat completions with streaming; the model id is `qwen3.8-flash-next`. Images go in as OpenAI content parts (base64 `data:` URLs) when vision is on, up to 4,096 image tokens each; the projector runs on the CPU so it takes no VRAM; a 1400×1000 screenshot is 1,364 tokens and encodes in about 15 s on 8 cores, a 1280×720 one in 7-8 s.
- Thinking is `xhigh` by default; change it per request with `reasoning_effort` (`xhigh` | `medium` | `low` | `off`) or `reasoning_budget`, or with `/think` and `/no_think` in a message. Reasoning comes back separately in `reasoning_content`.
- Sampling presets follow the model card (thinking and non-thinking) unless you pass `temperature`, `top_p`, `top_k`, `min_p` or the penalties; tool calling follows the OpenAI `tools` / `tool_choice` shape, and a call streams as `tool_calls` deltas while the model is still writing it, so a harness sees the code arrive instead of a minutes-long silence (Unsloth Studio drops a stream after 300 s without bytes; the server also sends an SSE keepalive whenever nothing else has gone out for 15 s).
- Every response carries llama.cpp-style `timings`; `/stats` is what the console's live panel reads.
- One request at a time: the engine keeps a single context, and a conversation that continues the previous one only prefills its new turn.

**Claude Code** talks to it natively: the server also serves the Anthropic Messages API (`POST /v1/messages`, streamed, and `count_tokens`), so point Claude Code at the server itself rather than at `/v1`:

```bash
ANTHROPIC_BASE_URL=http://127.0.0.1:8080 ANTHROPIC_AUTH_TOKEN=local ANTHROPIC_MODEL=qwen3.8-flash-next claude
```

Thinking comes back as `thinking` blocks and tool calls as `tool_use` blocks, streamed as the model writes them; usage reports input, cached and output tokens; a replayed conversation continues the engine's prefix like any other. Claude Code's settings map onto the reasoning effort: `thinking` `disabled` is off, `/effort low` and `medium` are those levels, `high` (its default) is the level the server was started with, `max` is xhigh, and a `budget_tokens` caps under the server's own thinking budget. Its first request is about 17K tokens of system prompt and tool definitions, prefilled once (~25 s on the Q4 file at 680 tok/s) and continued from then on, so use the Agentic coding tier or larger. The model name is echoed, not checked.

**Claude Desktop** has a third-party inference mode that takes the same server. `scripts/claude-desktop.sh` (`qwfnfer-claude-desktop` from the bundle) starts the engine through the console if it is not running, writes a Claude Desktop profile whose inference provider is the server, and launches Claude Desktop on that profile: a second instance, next to the one signed into your claude.ai account, with Chat and Code on the local model. The profile lives in `~/.config/Claude-qwfnfer` and `~/.config/Claude-qwfnfer-3p`; `--restart` relaunches it after a change, `--stop` quits it, `--reset` deletes it. The app only accepts model ids that look like Claude models, so the profile sends `claude-sonnet-5` and labels it with the model the server really serves; the server ignores the name. Doing it by hand instead: Help → Troubleshooting → Enable Developer Mode, then Developer → Configure Third-Party Inference, provider *gateway*, base URL `http://127.0.0.1:8080`, any API key, and a model entry named `claude-sonnet-5`.

**Or additive, in the app you already use.** Claude Code takes one base URL, so `tools/qwfn_router.py` listens on it and forwards each request by the model it names: the local model's id goes to the engine, everything else goes to `api.anthropic.com` as it came, headers and body untouched, so the claude.ai login, prompt caching and the beta features keep working. `python3 tools/qwfn_router.py --configure` points Claude Code at it (`~/.claude/settings.json`: `env.ANTHROPIC_BASE_URL` and a `modelPicker` entry, a backup kept; `--unconfigure` reverts) and `--install-service` keeps it running as a systemd user service. The local model then shows in `/model` next to the Anthropic ones, in the same app and the same list of sessions, and each session picks. Side requests (titles, summaries) use the session's small model, so they go to Anthropic and leave the engine's prefix alone.

**Building from source** (only if you want to change the engine). Needs CMake, Ninja, CUDA, liburing and a built [llama.cpp](https://github.com/unslothai/llama.cpp) tree for the ggml backends and the tokenizer (default `~/.unsloth/llama.cpp`, override with `-DLLAMA_CPP_ROOT`):

```bash
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release && cmake --build build -j && scripts/console.sh
```

`scripts/package.sh` builds the relocatable bundle the installer downloads (`-DQWFN_PORTABLE=ON`: baseline x86-64-v3 code, libraries next to the binaries, the glibc floor of the machine it is built on, 2.35 from the release workflow), and `.github/workflows/release.yml` does the same on a tag push and attaches the zip to the release.


## How it works

Per decoded token the model touches about 1.1 GB of expert weights (48 layers × 10 experts); at 15 tok/s that is 16 GB/s, more than the NVMe delivers. The engine arranges for most of it to never leave the GPU:

1. **Expert slices are read whole**, 0.6–1.2 MB at a time over io_uring/O_DIRECT, instead of being demand-paged 4 KiB at a time through mmap.
2. **Three tiers with one policy**: a VRAM tier of ~2,600–3,900 experts at 160K context with the attention state in pinned RAM (8–9 GB on a 16 GB GPU), a pinned RAM arena sized from the memory the server leaves (13–15 GB on the reference machine, worth about +3% of decode per GB) and the NVMe; 95–99% of lookups hit and 62–77% are served from VRAM, depending on the quant and on the draft head's tier step.
3. **In-graph MoE**: VRAM-resident experts are computed inside each layer's replayed CUDA graph, with residency looked up on the device; experts fetched late are folded into the next graph.
4. **Speculative prefetch**: the next layer's routing is predicted by running that layer's own DeltaNet or attention block on the current residual inside the current layer's graph, state writes suppressed, and its experts are fetched while the current layer computes; the prediction is right for 95-96% of the next layer's experts (80% with the router alone), only the reads a token actually needs are waited for, and the output is exact.
5. **Flat decode at any context**: sparse attention over pooled block keys, F16 keys, q4_0 KV, which gives the same per-layer cost at 4K and at 160K.
6. **Layer-major prefill**: each layer's experts stream through VRAM in 16 MB chunks and sweep the whole batch; the tier lends the memory and takes it back.
7. **Measured against the reference**: bit-exact forward pass, in-process numeric checks, real chat workloads and replay files; GPU decode is nondeterministic, so nothing is judged on a single token diff.

## Built around the Qwen4 architecture

Qwen3.8-Flash-Next ships the Qwen4-generation design, `qwen4exp` in the GGUF, and the engine is shaped by what that checkpoint actually contains rather than by its parameter count:

- **48 layers, 36 Gated DeltaNet + 12 sparse attention** (every fourth layer), a residual of 4 hyper-connected streams, 512 routed experts with top-10 routing plus one shared expert, a lightning indexer (4 × 128, top-2048) with 4-way pooled keys, and a 51B-parameter per-layer n-gram embedding table (PLE), 28.8 GB on its own.
- **Placement follows the shape.** The dense core (about 5 GB) is resident in VRAM. The routed experts are the only weights that need bandwidth, so they get the three-tier cache. The PLE table stays on the NVMe: a token reads 16 rows of 90 bytes from it (181 µs), so the largest tensor in the file costs no RAM at all.
- **The hybrid layer mix is what makes decode flat.** DeltaNet layers carry a fixed recurrent state and no KV, so their decode graphs reference nothing that changes with position and replay as CUDA graphs; the 12 attention layers select over pooled block keys, so their cost does not grow with context up to the trained 262K.
- **The MoE's routing skew is what makes a small GPU enough.** With 512 fine-grained experts and 10 active, routing is far from uniform, so a VRAM tier of ~2,600–3,900 experts serves 62–77% of lookups at 160K context, and the next layer's routing can be computed a layer early by running its block on the residual stream and prefetched while the current layer runs.
- **The model card is followed** for the thinking template, the tool-call format, mrope for images and the sampling presets; the forward pass is checked node by node against llama.cpp's `qwen4exp`, which matters because this architecture is unusually sensitive to accumulation order.

**What this means for the next Qwen releases.** Every hyper-parameter the engine uses is read from the GGUF metadata (layer count and interval, expert count and top-k, indexer geometry, DeltaNet sizes, PLE geometry, context and rope). Nothing is hard-coded to this checkpoint. A future checkpoint built from the same blocks at a different size (more experts, more layers, a bigger PLE, a longer context) is a metadata change; a new block is a graph change, validated against the reference the same way. What the engine needs from the hardware is set by the *active* path per token and by the tiers you can afford, not by the file size: the dense core and the KV/indexer state must fit in VRAM, and everything else streams through cache tiers sized to the GPU and RAM present. The console's cost model does that sizing for whatever machine it finds. Two things are still on the list: the checkpoint's multi-token-prediction head (the *Draft head* setting: the trunk verifies every draft, so the output is its own) pays in chat and in agentic coding but not yet on a 155K-token document, and the tiers have only been measured on the 16 GB / 30 GB reference machine; the console's auto-tune is what carries the sizing to other machines.

## FAQ

**Does it have to be this exact machine?** No — what matters is the shape, not the model numbers. VRAM has to hold the dense core (about 5 GB), and what is left over becomes the expert tier, which is what sets decode speed. The context's caches mostly do not compete for it: from 128K up the console keeps the KV and indexer caches in pinned RAM and gathers them over PCIe, because their VRAM is worth more as expert tier (about 4% of decode per GB at 131K, and at 256K the difference between having a tier and not). At 160K with q8_0 KV that moves 2.5 of the 2.8 GB of attention state off the device and leaves the pooled block keys and the DeltaNet state, under 0.3 GB. Below 64K it all stays in VRAM, where it is small anyway (0.6 GB at 32K); in between, only the indexer cache moves. What 16 GB has left at 160K is an 8.1–8.7 GB expert tier (2,591 experts on Q4, 3,859 on Q3; 62–77% of lookups served straight from VRAM). A smaller GPU gets a smaller tier and the console sizes for it; where there is no room for a tier at all, every expert comes from RAM or the NVMe, about 25% slower by the cost model. Every number in this README is one machine — RTX 4080 SUPER 16 GB, 30 GB of RAM, one NVMe — and the auto-tune is what carries the sizing to a different one.

**How much RAM do I need?** Mostly as a cache tier rather than as a floor. The engine clamps that tier to what the machine really has free (0.75 × MemAvailable), and the console sizes it from the memory the server actually needed during the tune, leaving 3 GB of headroom — 13–15 GB here. A GB of tier is worth about 3% of decode, so less RAM costs speed rather than the ability to run. What is not elastic is the pinned memory the plan puts there on purpose: the attention caches at long context (2.5 GB at 160K) and the draft head's 2.7 GB of pinned experts when it is on — the console counts both before it sizes the tier.

**Does the model have to sit on an NVMe?** In practice yes. A decoded token touches about 1.1 GB of expert weights; the tiers serve 95–99% of the lookups and the drive covers the rest, so decode follows the drive's *random* read rate, not its sequential one. The console probes it before planning (random 2 MiB O_DIRECT reads, the pattern of a miss) and warns below 2.5 GB/s; on a spinning disk it tells you to move the file. The engine only ever reads the model — no writes, no conversion step — and reads it with O_DIRECT, falling back to buffered reads on a filesystem that refuses it.

**How much disk space?** The file itself: 111 GB for UD-Q4_K_XL, 90 GB for UD-Q3_K_XL. Vision adds the 0.9 GB `mmproj-F16.gguf` and the draft head the 2.6 GB `MTP/mtp-*.gguf`; the console picks both up when they sit next to the shards (`hf download unsloth/Qwen3.8-Flash-Next-GGUF --include "MTP/*"`). Nothing is unpacked or converted — the GGUF shards are read in place.

**Q4 or Q3?** Q4 for quality, Q3 for speed: 13.2–15.7 against 19.8–21.2 tok/s in chat, 12.9 against 17.7 on a 155K-token document, same machine and same plan. Q3's expert blocks are 2.27 MB against Q4's 3.13, so more of them fit in the same tiers and every miss reads less; that is most of the difference.

**Windows?** In the works, not there yet. Today it is Linux x86_64 with an NVIDIA GPU, driver 580 or newer: the NVMe path is io_uring, so that layer is what has to be ported first — nothing above it is Linux-specific.

**AMD or Intel GPU? Two GPUs?** One NVIDIA GPU: the dense core, the replayed graphs and the VRAM expert tier run on ggml's CUDA backend, and the engine builds for a single device. `--cpu` runs everything on the CPU path — the one the forward pass is validated bit-exact against — but that path exists for validation, not for use.

**Can it run other models?** No. It reads the architecture out of the GGUF and refuses anything that is not `qwen4exp`; it implements that graph, not a general one. Its hyper-parameters all come from the file's metadata, so a future checkpoint built from the same blocks at another size is a metadata change, while a new block is a graph change.

**Is this a llama.cpp fork?** No. It uses ggml's quantized kernels and CUDA backend, llama.cpp's tokenizer, and llama.cpp's own `qwen4exp` implementation as the reference the forward pass is checked against. The model graph, the memory hierarchy, the expert cache, the prefill and the server are about 12K lines of its own C++, with the console on top of them in Python.

**Then why is it 7–10× faster than llama.cpp on the same file and GPU?** I/O granularity, mostly. The mmap path demand-pages experts 4 KiB at a time — 4.4 million reads per pass at 0.3 GB/s on this drive, essentially all of decode spent in page faults — while the same NVMe asked for whole 0.6–1.2 MB expert slices at queue depth 4 gives 7 GB/s. The tiers, the MoE inside the CUDA graph and the layer-ahead prefetch are all built on top of that factor of 23.

**Do the speed options change what the model writes?** The two that are on by default do not. The speculative block only decides which experts to fetch early; whatever a token actually routes to is waited for, and the output is bit-identical with the block off. The draft head's tokens are verified by the trunk, so what is accepted is what the trunk itself would have produced. *Skip missed experts* (`--skip-miss`) is the one that trades: +16% decode at 4K and +22–29% at 128K for a measured quality cost (NLL +0.046 at 4K, within noise at 128K), and it is off by default. Independently of all three, GPU decode varies about 5% run to run on identical work.

**Can several people share one server?** No — one engine, one request at a time, and a second request waits for the first. That is the model's shape rather than a missing feature: a single KV cache plus the DeltaNet recurrent state and the short-conv history are sequential accumulations over one sequence, and interleaving two conversations would corrupt both. What it does instead is continue the previous conversation's prefix, so a harness that replays the whole thread every turn only prefills the new turn.

**Is it safe to put on the network?** It binds `127.0.0.1` and has no authentication — any API key works because none is checked, and CORS is open so the console page can reach it. `--host` will bind it wider; put something that authenticates in front of it before you do.

**Why is the first request so much slower than the rest?** The tiers start cold: they fill from the routing of the text going through them, and that first prefill reads every expert it touches off the drive. A long system prompt is where it shows — Claude Code's ~17K-token first request costs about 25 s on the Q4 file, and every turn after it continues that prefix.

**How long does the auto-tune take, and is it once?** About five minutes, and the result is saved per model: the measured thread count and drive rate then apply to every tier for that model, and the tier card shows the measured speed instead of the prediction. Re-run it when the hardware or the model file changes; **Start server** uses the saved plan without re-measuring.

**Decode is slower than the numbers here — what do I check?** The Log tab prints what the engine actually built: the VRAM expert tier and its block count, the RAM arena, and a VRAM audit of what is left on the device after init. The usual causes are a RAM tier clamped down because the machine had less free memory than the plan assumed, another process holding VRAM (the console measures the desktop's use and subtracts it, but it measures it once), a drive slower at random 2 MiB reads than the plan assumed, and a cache that is simply still cold. `/stats` shows the hit rate and the VRAM-served share while it runs.

**Does it phone home, and does it work offline?** Only the model download needs the internet. The engine, the server and the console page are local, and nothing is sent anywhere — `tools/qwfn_router.py` is the one exception, and only for the requests you point at Anthropic yourself.

**How do I uninstall it?** Delete `~/.local/share/qwfnfer` and `~/.local/bin/qwfnfer`. The model files are yours and stay where you downloaded them.

## Acknowledgment

Built on [ggml](https://github.com/ggml-org/ggml) (quantized kernels, CUDA backend) and [llama.cpp](https://github.com/ggml-org/llama.cpp) (tokenizer, and the bit-exact reference the forward pass is validated against). Model: Qwen3.8-Flash-Next by the [Qwen](https://huggingface.co/Qwen) team; quantized GGUFs by [Unsloth](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF), whose Studio served as the harness for testing.

## License

[Apache License 2.0](LICENSE).
