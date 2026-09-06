# qwfnfer — post drafts

Images: `docs/img/results.png` (the numbers and the how), `docs/img/console-serve.png`,
`docs/img/console-chat.png`, `docs/img/console-stats.png`, `docs/img/console-log.png`.

---

## Hugging Face community post (long form)

**Running Qwen3.8-Flash-Next UD-Q4_K_XL (125B, 111 GB, 512 experts) with 160K context on a single 16 GB GPU — a purpose-built engine, not a llama.cpp fork**

Qwen3.8-Flash-Next is the Qwen4-architecture preview: 48 layers, 512 routed experts (top-10
plus one shared), gated DeltaNet on 36 layers and sparse attention with an indexer on 12,
hyper-connections, a per-layer n-gram embedding table, 262K context. At UD-Q3_K_XL the
checkpoint is 90 GB: 56 GB of routed experts, 29 GB of n-gram table, and a 5 GB dense core.
Only ~6.7B parameters are active per token, but on a 16 GB GPU with 30 GB of RAM none of
the usual paths fit it: the experts have to come off the NVMe.

llama.cpp runs this model correctly, and on this machine it decodes the Q4 file at 1.4–2.2
tok/s in chat and prefills at 3 tok/s (Unsloth Studio's bundled llama-server, its defaults; the
lighter Q3 file: 4.6–6.3 tok/s at long context). Not because of its math — because of I/O granularity: its mmap path demand-pages
experts 4 KiB at a time (measured: 4.4 million reads per pass at 0.3 GB/s, ~100% of decode
time in page faults). The same NVMe, asked for 640 KB slices at queue depth 4, does 7 GB/s.
An expert slice is 627–921 KB. That factor of 23 is the whole thesis; everything else follows.

**qwfnfer** is a small engine (about 8K lines of C++) built on ggml as the kernel library
and llama.cpp's tokenizer, that owns everything above the kernels: the `qwen4exp` forward
graph, the memory hierarchy, the expert cache, the prefill, the server and a console. Its
forward pass is validated bit-exact against llama.cpp's layer by layer.

What it does:

- **Expert slices read at their natural size over io_uring/O_DIRECT** — 6.4 GB/s where mmap
  paging gave 0.28.
- **A three-tier expert cache**: ~7 GB of VRAM (about 3,400 experts), 12 GB of pinned RAM,
  the NVMe behind them, with hybrid frequency/recency eviction. Routing is heavily skewed
  (Zipf α≈1.2), so 95–96% of lookups hit and 85–90% are served straight from VRAM.
- **Routed experts computed inside each layer's CUDA graph**: the router's ids index a
  per-layer residency table on the device and one `mul_mat_id` runs over the VRAM tier, no
  host round trip; the dense-core graphs are captured once and replayed.
- **Cross-layer prefetch**: the next layer's routing is predicted from the current residual
  (83.6% of its experts) and its reads go out while the current layer computes; a fetch
  waits only for the reads that token needs.
- **Decode attention flat at any context** — Qwen Sparse Attention with pooled block keys,
  top-k block selection and gathered cells: 550 µs per layer at 4K and at 160K alike.
- **Layer-major streamed prefill**: each layer's experts are streamed in 16 MB O_DIRECT
  chunks into VRAM and swept over the whole batch — 2.6 → 272–298 tok/s on a 133K-token
  document, 372 tok/s on a 3.8K prompt.
- **VRAM budgeting for long context**: the prefill's memory is lent by the expert tier
  and returned; the embedding table gathers on the host; KV at q4_0; F16 pooled keys —
  the tier at 160K context went from 3.5 to 7 GB.
- **An OpenAI-compatible server** with streaming, tool calling (mapped to the model's own
  `<tool_call>` template), thinking control and a thinking budget, live `timings`, `/stats`,
  `/props` and Prometheus `/metrics`; and a **console** that lists the downloaded quants,
  reads the hardware, sizes the flags for it behind three presets (chat at 32K, agentic
  coding at 128K and 256K, each with its predicted speed), starts and stops the server,
  self-tests it on the machine, chats with it, and watches it live.

Where it stands on the reference machine (RTX 4080 SUPER 16 GB, 30 GB RAM, one NVMe),
163,840-token context, KV q4_0, measured 2026-09-05:

| | UD-Q4_K_XL (111 GB) | UD-Q3_K_XL (90 GB) |
|---|---:|---:|
| short chat, decode | 11.8 tok/s | 16.8 tok/s |
| 155K-token document: prefill | 228 tok/s (11 min) | 273 tok/s |
| 155K-token document: decode, grounded answer | 9.3 tok/s | 14.5 tok/s |
| one 14,000-token answer at 160K context, sustained | 11.7 tok/s | — |
| expert cache: hit / served from VRAM | 95% / 82% | 95% / 85% |

llama.cpp on the same machine, same files, measured the same day. The Q4 numbers are Unsloth
Studio's bundled llama-server (build 10715) at its own defaults — 126K context, KV q4_0,
`--fit on`, four slots with a unified KV, speculative decoding on:

| llama.cpp | UD-Q4_K_XL | UD-Q3_K_XL |
|---|---:|---:|
| short chat, decode (three runs) | 1.4–2.2 tok/s | 4.6–6.3 tok/s at long context (earlier session, mmap) |
| prefill of the same document | 3.0 tok/s — stopped after 4,096 tokens (22.7 min); 117K tokens would take ~11 h | — |
| decode grounded on a 4K-token document | 2.7 tok/s | — |

The gap is the memory hierarchy, not the kernels (they are the same ggml kernels): llama.cpp
pages experts in 4 KiB units through mmap, so its prefill of a long document is bound by page
faults (158 MB/s off the NVMe, the GPU 15–30% busy) and its decode by what the page cache can
hold — about a quarter of the Q4 expert set on a 30 GB machine.

Q3 is the faster file because its expert blocks are 42% smaller — every gigabyte of VRAM holds
30% more experts and every miss reads less; Q4 is the quality choice, and it is what I run.

What the machine looks like while it serves Q4: 22 of 31 GB of RAM in use system-wide (the
engine itself 15 GB: a 12 GB pinned expert arena plus staging), 13.3 of 16 GB of VRAM with
1 GB kept in reserve, the GPU 70% busy during decode, and the engine on about 3 of 16 CPU
threads. That is the point I want to make to anyone with a similar desktop: **the machine
stays usable while it serves** — 8 GB of RAM, most of the CPU and the GPU's reserve are left
over, and in practice a browser with a video or a Discord/Teams call runs alongside without
touching the token rate.

Things that measured as no better and were kept off, because the engineering log keeps score: a
learned routing predictor (a linear head cannot see the next layer's attention output; it
transfers badly across text types), a whole-token-ahead prefetch (too diffuse for the
NVMe's bandwidth), more RAM tier (the marginal experts are cold), deeper prediction at long
context (saturates the disk), and the 1-bit cold tier (neutral once its I/O bug was fixed).

Everything above is measured on real workloads and written up in the repo with the numbers,
including the things that did not work; the roadmap lists what is left (the MTP head the
model ships separately is the obvious next lever).

---

## LinkedIn post (short form)

I built a small inference engine that runs Qwen3.8-Flash-Next at UD-Q4_K_XL — a 111 GB
checkpoint of a 125B-parameter MoE with 512 experts — on one 16 GB GPU with 30 GB of RAM,
with 160K tokens of context in use: 12 tok/s in chat, 9 tok/s answering questions about a
155K-token document it just read at 228 tok/s (the lighter Q3 file: 17 and 14.5).

The checkpoint is 111 GB. On this machine llama.cpp (Unsloth Studio's bundled build, its
defaults) manages about 2 tok/s in chat and 3 tok/s of prefill with it — 4.6–6.3 tok/s at long
context with the lighter Q3 file — and the reason is not its math: it pages experts off the
NVMe 4 KiB at a time. Read at
their natural 600–900 KB size, the same disk is 23× faster. The engine (about 8K lines of
C++, built on ggml, validated bit-exact against llama.cpp) is organised around that:

• expert slices over io_uring/O_DIRECT, a three-tier VRAM/RAM/NVMe expert cache with
  95–96% hit rate
• the VRAM-resident experts computed inside each layer's CUDA graph, next-layer routing
  predicted and prefetched while the current layer runs
• decode attention that costs the same at 160K context as at 4K
• prefill of a 133K-token document at ~280 tok/s instead of 2.6
• an OpenAI-compatible server (streaming, tool calling, thinking budget, live stats) and a
  console with three presets sized for your hardware and a self-test

And the machine stays usable while it serves: 22 of 31 GB of RAM in use, 13 of 16 GB of
VRAM, the GPU 70% busy, the engine on 3 of 16 CPU threads — a video, a Discord or Teams
call and the desktop run alongside. Every number is in the repo, including the things that
did not work.

#LLM #MoE #Qwen #inference #CUDA #ggml
