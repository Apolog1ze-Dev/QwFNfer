<div align="center">
  <h1>qwfnfer</h1>
  <p><b>Qwen3.8-Flash-Next — a 56B mixture-of-experts model, 512 experts, 111 GB on disk — at 160K context on one 16 GB GPU, 30 GB of RAM and an NVMe.</b></p>
</div>

<p align="center">
| <a href="#getting-started"><b>Getting Started</b></a> | <a href="#results"><b>Results</b></a> | <a href="#how-it-works"><b>How it works</b></a> | <a href="#built-around-the-qwen4-architecture"><b>Qwen4</b></a> | <a href="https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF"><b>Model (Unsloth GGUF)</b></a> |
</p>

Run a **125B** open-weight MoE on the gaming PC you already own, at interactive speed: **10–11 tok/s** in chat, 10 tok/s answering questions about a 155K-token document it read at 240 tok/s — with the desktop still usable next to it.

## About

qwfnfer is a purpose-built inference engine for Qwen3.8-Flash-Next (GGUF architecture `qwen4exp`): 48 layers, 512 routed experts with top-10 routing, DeltaNet recurrent layers and Qwen Sparse Attention. It is not a llama.cpp fork. It uses ggml's quantized kernels and CUDA backend and llama.cpp's tokenizer, and owns everything above them: the model graph, the memory hierarchy, the expert cache, prefill, the server and the console. Its core features:

- **Three-tier expert runtime**: experts live in VRAM, in pinned RAM and on the NVMe. Reads happen at an expert's natural 0.6–1.2 MB size over io_uring/O_DIRECT (23× the bandwidth of 4 KiB demand paging on the same disk), VRAM-resident experts compute inside replayed CUDA graphs, and the next layer's routing is predicted from the residual and prefetched while the current layer computes.
- **Long context that stays flat**: 163,840 tokens with q4_0 KV on 16 GB. Decode attention costs the same 0.55 ms per layer at 4K and at 160K (sparse attention over pooled block keys), and a layer-major prefill streams experts through VRAM at 240–295 tok/s instead of paging them.
- **OpenAI-compatible server**: streaming, thinking with `reasoning_effort` and a thinking budget, tool calling, vision, `/props`, `/stats`, `/slots`, `/metrics` and llama.cpp-style `timings` — works with Unsloth Studio, Open WebUI or any OpenAI client.
- **Console**: a local page that lists the downloaded quants, sizes the flags for *your* GPU and RAM behind three presets (Chat at 32K context, Agentic coding at 128K, Agentic coding+ at 256K, each with its predicted speed), starts and stops the server, self-tests it on your hardware, chats with it, and shows live tokens/s, prefill speed, cache hit rate and the endpoint.
- **Measured, not projected**: the forward pass is validated bit-exact against llama.cpp, and every number here is a real run on the reference machine, same file, same settings.

## Results

<div align="center">
  <img alt="measured results" src="docs/img/results.png" width=100%>
</div>

Reference machine: RTX 4080 SUPER 16 GB, 30 GB RAM, one NVMe. 163,840-token context, KV q4_0, the console's flags (speculative block and vision on, 1 GB of VRAM reserved for the desktop), one run each through the server, measured 2026-09-06.

| | UD-Q4_K_XL (111 GB) | UD-Q3_K_XL (90 GB) |
|---|---:|---:|
| short chat, decode | 10.2–10.8 tok/s | 15.4–16.2 tok/s |
| 155K-token document: prefill | 240 tok/s (11 min) | 295 tok/s (9 min) |
| 155K-token document: decode, grounded answer | 10.0 tok/s | 14.9 tok/s |
| one 8,300-token answer at 160K context, sustained | 10.3 tok/s | — |
| llama.cpp on the same machine and file | 1.4–2.2 tok/s chat, 3 tok/s prefill (Unsloth Studio's defaults) | 4.6–6.3 tok/s at long context |

The two options the console exposes, on the same Q4 file and settings:

| UD-Q4_K_XL, 160K context | short chat | 155K-token document, decode |
|---|---:|---:|
| speculative next-layer block off | 9.0–10.3 tok/s | 8.3 tok/s |
| speculative block on (the default) | 10.2–10.8 tok/s | 10.0 tok/s |
| speculative block + draft head (MTP) | 9.6–11.4 tok/s (70% of drafts accepted) | 10.0 tok/s (no drafts on a streamed prompt) |

The speculative block predicts the next layer's experts by running that layer's own block on the residual (95–96% right, against 80% for the router alone) and fetches them a layer early; the output is exact. The draft head is the checkpoint's own next-token predictor, verified by the trunk, so its output is the trunk's own; it only drafts when the prompt fits in one batch, so a streamed 155K-token document decodes without it. GPU decode varies about 5% run to run on identical work.

While it serves Q4: 22 of 31 GB of RAM in use system-wide, 14.7 of 16 GB of VRAM, the GPU 45% busy, the engine on 3 of 16 threads. A browser with a video and a Discord or Teams call run alongside it without touching the token rate.

## Getting Started

**1. Get the model.** The console finds Qwen3.8-Flash-Next GGUFs in your Hugging Face cache. UD-Q4_K_XL is the quality choice, UD-Q3_K_XL the faster one; the `mmproj` file adds vision.

```bash
hf download unsloth/Qwen3.8-Flash-Next-GGUF --include "UD-Q4_K_XL/*" "mmproj-F16.gguf"
```

**2. Build once.** Needs CMake, Ninja, CUDA and a built [llama.cpp](https://github.com/ggml-org/llama.cpp) for the ggml backends and the tokenizer (default `~/.unsloth/llama.cpp`, override with `-DLLAMA_CPP_ROOT`):

```bash
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
```

**3. Open the console.**

```bash
scripts/console.sh
```

It opens http://127.0.0.1:8090. Pick a downloaded quant and a preset — **Chat** (32K context), **Agentic coding** (128K) or **Agentic coding+** (256K, the model's full trained context) — and press **Start server**. The console sizes every flag for your GPU and RAM (context, KV type, expert tiers, reserve; vision on when the `mmproj` file is next to the model, unless a preset needs the projector's VRAM to keep an expert tier at all, and the card says so) and shows the decode speed to expect from each preset; the table behind *Advanced settings* shows what every context size costs, and each setting has a tooltip. **Self-test** starts the server if needed and measures it on your machine: a short chat, then a 16K–32K-token document with a passphrase planted in it, reporting prefill and decode tokens/s against the prediction and whether the answer found the passphrase. The banner names the model and every flag it is running with; the Chat, Stats and Log tabs talk to it, show live tokens/s, prefill speed, tokens served, cache hit rate and the endpoint, and follow the server log. Stop it from the same page.

<div align="center">
  <img alt="qwfn console" src="docs/img/console-serve.png" width=92%>
</div>

**4. Point your tools at it.** The console shows the endpoint, `http://127.0.0.1:8080/v1` by default; any OpenAI-compatible client works with any API key (Unsloth Studio as a custom provider, Open WebUI, your own scripts). What the server accepts:

- Chat completions with streaming; the model id is `qwen3.8-flash-next`. Images go in as OpenAI content parts (base64 `data:` URLs) when vision is on, up to 4,096 image tokens each; a 1400×1000 screenshot is 1,364 tokens and encodes in 0.7 s.
- Thinking is `xhigh` by default; change it per request with `reasoning_effort` (`xhigh` | `medium` | `low` | `off`) or `reasoning_budget`, or with `/think` and `/no_think` in a message. Reasoning comes back separately in `reasoning_content`.
- Sampling presets follow the model card (thinking and non-thinking) unless you pass `temperature`, `top_p`, `top_k`, `min_p` or the penalties; tool calling follows the OpenAI `tools` / `tool_choice` shape, and a call streams as `tool_calls` deltas while the model is still writing it, so a harness sees the code arrive instead of a minutes-long silence (Unsloth Studio drops a stream after 300 s without bytes; the server also sends an SSE keepalive whenever nothing else has gone out for 15 s).
- Every response carries llama.cpp-style `timings`; `/stats` is what the console's live panel reads.
- One request at a time: the engine keeps a single context, and a conversation that continues the previous one only prefills its new turn.

## How it works

Per decoded token the model touches about 1.1 GB of expert weights (48 layers × 10 experts); at 15 tok/s that is 16 GB/s, more than the NVMe delivers. The engine arranges for most of it to never leave the GPU:

1. **Expert slices are read whole**, 0.6–1.2 MB at a time over io_uring/O_DIRECT, instead of being demand-paged 4 KiB at a time through mmap.
2. **Three tiers with one policy**: a VRAM tier of ~1,650–2,400 experts at 160K context (up to 3,700 at short context), a 12 GB pinned RAM arena and the NVMe; 91–97% of lookups hit and 57–86% are served from VRAM, depending on the quant and the tier the context leaves.
3. **In-graph MoE**: VRAM-resident experts are computed inside each layer's replayed CUDA graph, with residency looked up on the device; experts fetched late are folded into the next graph.
4. **Speculative prefetch**: the next layer's routing is predicted by running that layer's own DeltaNet or attention block on the current residual inside the current layer's graph, state writes suppressed, and its experts are fetched while the current layer computes; the prediction is right for 95-96% of the next layer's experts (80% with the router alone), only the reads a token actually needs are waited for, and the output is exact.
5. **Flat decode at any context**: sparse attention over pooled block keys, F16 keys, q4_0 KV — the same per-layer cost at 4K and at 160K.
6. **Layer-major prefill**: each layer's experts stream through VRAM in 16 MB chunks and sweep the whole batch; the tier lends the memory and takes it back.
7. **Measured against the reference**: bit-exact forward pass, in-process numeric checks, real chat workloads and replay files; GPU decode is nondeterministic, so nothing is judged on a single token diff.

## Built around the Qwen4 architecture

Qwen3.8-Flash-Next ships the Qwen4-generation design, `qwen4exp` in the GGUF, and the engine is shaped by what that checkpoint actually contains rather than by its parameter count:

- **48 layers, 36 Gated DeltaNet + 12 sparse attention** (every fourth layer), a residual of 4 hyper-connected streams, 512 routed experts with top-10 routing plus one shared expert, a lightning indexer (4 × 128, top-2048) with 4-way pooled keys, and a 51B-parameter per-layer n-gram embedding table (PLE) — 28.8 GB on its own.
- **Placement follows the shape.** The dense core (about 5 GB) is resident in VRAM. The routed experts are the only weights that need bandwidth, so they get the three-tier cache. The PLE table stays on the NVMe: a token reads 16 rows of 90 bytes from it (181 µs), so the largest tensor in the file costs no RAM at all.
- **The hybrid layer mix is what makes decode flat.** DeltaNet layers carry a fixed recurrent state and no KV, so their decode graphs reference nothing that changes with position and replay as CUDA graphs; the 12 attention layers select over pooled block keys, so their cost does not grow with context up to the trained 262K.
- **The MoE's routing skew is what makes a small GPU enough.** With 512 fine-grained experts and 10 active, routing is far from uniform, so a VRAM tier of ~1,650–2,400 experts serves 57–68% of lookups at 160K context (~3,700 serve 86% at short context), and the next layer's routing can be computed a layer early by running its block on the residual stream and prefetched while the current layer runs.
- **The model card is followed** for the thinking template, the tool-call format, mrope for images and the sampling presets; the forward pass is checked node by node against llama.cpp's `qwen4exp`, which matters because this architecture is unusually sensitive to accumulation order.

**What this means for the next Qwen releases.** Every hyper-parameter the engine uses is read from the GGUF metadata — layer count and interval, expert count and top-k, indexer geometry, DeltaNet sizes, PLE geometry, context and rope — nothing is hard-coded to this checkpoint. A future checkpoint built from the same blocks at a different size (more experts, more layers, a bigger PLE, a longer context) is a metadata change; a new block is a graph change, validated against the reference the same way. What the engine needs from the hardware is set by the *active* path per token and by the tiers you can afford, not by the file size: the dense core and the KV/indexer state must fit in VRAM, and everything else streams through cache tiers sized to the GPU and RAM present — the console's cost model does that sizing for whatever machine it finds. Two things are still on the list: the checkpoint's multi-token-prediction head is wired in (the *Draft head* setting: the trunk verifies every draft, so the output is its own) but only pays at short context so far, and the tiers have only been measured on the 16 GB / 30 GB reference machine.

## Acknowledgment

Built on [ggml](https://github.com/ggml-org/ggml) (quantized kernels, CUDA backend) and [llama.cpp](https://github.com/ggml-org/llama.cpp) (tokenizer, and the bit-exact reference the forward pass is validated against). Model: Qwen3.8-Flash-Next by the [Qwen](https://huggingface.co/Qwen) team; quantized GGUFs by [Unsloth](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF), whose Studio served as the harness for testing.

## License

To be added.
