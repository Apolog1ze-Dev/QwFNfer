#!/usr/bin/env python3
"""Render docs/img/results.svg/.png from measured numbers: python3 docs/img/make_results.py numbers.json"""
import json, sys, subprocess
n = json.load(open(sys.argv[1]))
GREEN, RED, FG, DIM, TXT = "#6ee7b7", "#f87171", "#e6e9ee", "#8b95a3", "#c9d1db"
def bar(x, y, w, color, label, opacity=1.0):
    return f'<rect x="{x}" y="{y}" width="{max(2, w):.0f}" height="22" rx="3" fill="{color}" opacity="{opacity}"/><text x="{x + max(2, w) + 10:.0f}" y="{y + 16}" fill="{TXT}" font-size="14">{label}</text>'
def row(y, text, value, color=GREEN, label=None, scale=17.0, x0=40, xbar=400, opacity=1.0, dim=False):
    return f'<text x="{x0}" y="{y + 16}" fill="{DIM if dim else FG}">{text}</text>' + bar(xbar, y, value * scale, color, label if label is not None else f'{value:.1f}', opacity)
def head(y, text, x0=40):
    return f'<text x="{x0}" y="{y + 16}" fill="{FG}" font-size="15" font-weight="600">{text}</text>'
q4, q3, l4, l3 = n["q4"], n["q3"], n["llama_q4"], n["llama_q3"]
res = q4["resources"]
lines = []
# ---- decode, left column: one block per checkpoint, the engine first, llama.cpp under it
y = 160
lines.append(head(y, "UD-Q4_K_XL (111 GB)")); y += 34
lines.append(row(y, "qwfnfer · short chat", q4["dec_short"])); y += 36
lines.append(row(y, "qwfnfer · 155K-token document, grounded answer", q4["dec_doc"])); y += 36
lines.append(row(y, "qwfnfer · one 14,000-token answer at 160K context", q4["dec_long"], label=f'{q4["dec_long"]:.1f} sustained')); y += 36
lines.append(row(y, f'llama.cpp via {l4["via"]} · short chat', l4["dec_short"], RED, l4.get("dec_short_label"), opacity=.85)); y += 36
lines.append(row(y, f'llama.cpp via {l4["via"]} · {l4["doc_label"]}', l4["dec_doc"], RED, opacity=.85)); y += 50
lines.append(head(y, "UD-Q3_K_XL (90 GB)")); y += 34
lines.append(row(y, "qwfnfer · short chat", q3["dec_short"], label=q3["dec_short_label"], opacity=.6)); y += 36
lines.append(row(y, "qwfnfer · 155K-token document, grounded answer", q3["dec_doc"], opacity=.6)); y += 36
lines.append(row(y, "llama.cpp · long context (mmap paging)", l3["dec"], RED, l3["dec_label"], opacity=.6)); y += 36
y_dec_end = y
# ---- prefill, right column
PF, PX = 0.85, 1080
py = 160
plines = [head(py, "UD-Q4_K_XL", 820)]; py += 34
plines.append(row(py, f'qwfnfer · {q4["prefill_tok"]:,} tokens', q4["prefill"], label=f'{q4["prefill"]:.0f}  ({q4["prefill_s"] / 60:.0f} min)', scale=PF, x0=820, xbar=PX)); py += 36
plines.append(row(py, f'llama.cpp ({l4["via_short"]}) · {l4["prefill_tok"]:,} tokens', l4["prefill"], RED, label=l4["prefill_label"], scale=PF, x0=820, xbar=PX, opacity=.85)); py += 50
plines.append(head(py, "UD-Q3_K_XL", 820)); py += 34
plines.append(row(py, f'qwfnfer · {q4["prefill_tok"]:,} tokens', q3["prefill"], label=f'{q3["prefill"]:.0f}', scale=PF, x0=820, xbar=PX, opacity=.6)); py += 36
plines.append(row(py, "this engine's first version (mmap)", 2.6, label="2.6", scale=PF, x0=820, xbar=PX, opacity=.3)); py += 36
right_end = py + 234
how_y = max(y_dec_end, right_end) + 40
NOTES = n["notes"]
H = how_y + 200 + 50 + 22 * len(NOTES) + 20
notes = "\n".join(f'  <text x="40" y="{H - 20 - 22 * (len(NOTES) - 1 - i)}" fill="{DIM}" font-size="13">{t}</text>' for i, t in enumerate(NOTES))
svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="1400" height="{H}" viewBox="0 0 1400 {H}" font-family="system-ui, -apple-system, Segoe UI, Roboto, sans-serif">
  <rect width="1400" height="{H}" fill="#0f1216"/>
  <text x="40" y="52" fill="{FG}" font-size="28" font-weight="600">Qwen3.8-Flash-Next UD-Q4_K_XL (111 GB, 56B MoE) on one 16 GB GPU + 30 GB RAM + NVMe</text>
  <text x="40" y="82" fill="{DIM}" font-size="16">qwfnfer engine · RTX 4080 SUPER 16 GB · 30 GB RAM · one NVMe · 163,840-token context, KV q4_0 · measured {n["date"]} · UD-Q3_K_XL (90 GB) shown for comparison</text>
  <text x="40" y="104" fill="{DIM}" font-size="13">llama.cpp rows: the same machine and files; Q4 through {l4["via_long"]}</text>

  <text x="40" y="146" fill="{FG}" font-size="18" font-weight="600">Decode, tokens/s</text>
  <g font-size="14">
    {"".join(lines)}
  </g>

  <text x="820" y="146" fill="{FG}" font-size="18" font-weight="600">Prefill of a long document, tokens/s</text>
  <g font-size="14">
    {"".join(plines)}
  </g>

  <text x="820" y="{py + 30}" fill="{FG}" font-size="18" font-weight="600">What the machine looks like while it serves (Q4)</text>
  <g font-size="14" fill="{TXT}">
    <text x="820" y="{py + 62}">RAM in use, whole system: <tspan fill="{FG}" font-weight="600">{res["ram_used_gb"]:.1f} of {res["ram_total_gb"]:.0f} GB</tspan>  (engine {res["rss_gb"]:.1f} GB)</text>
    <text x="820" y="{py + 90}">VRAM in use: <tspan fill="{FG}" font-weight="600">{res["vram_used_gb"]:.1f} of 16 GB</tspan>  (1 GB kept in reserve for the desktop)</text>
    <text x="820" y="{py + 118}">GPU busy: <tspan fill="{FG}" font-weight="600">{res["gpu_util"]:.0f}%</tspan> during decode · CPU: <tspan fill="{FG}" font-weight="600">{res["engine_cores"]:.1f} of 16 threads</tspan> (system {res["sys_busy"]:.0f}% busy)</text>
  </g>
  <rect x="820" y="{py + 138}" width="540" height="96" rx="10" fill="rgba(110,231,183,.07)" stroke="rgba(110,231,183,.35)"/>
  <text x="836" y="{py + 162}" fill="{GREEN}" font-size="14" font-weight="600">The desktop stays usable while it serves</text>
  <text x="836" y="{py + 184}" fill="{TXT}" font-size="13">{res["ram_total_gb"] - res["ram_used_gb"]:.0f} GB of RAM, 13 of 16 CPU threads and the GPU's reserve are left over:</text>
  <text x="836" y="{py + 203}" fill="{TXT}" font-size="13">a browser with a video, a Discord or Teams call and the usual desktop</text>
  <text x="836" y="{py + 222}" fill="{TXT}" font-size="13">run alongside it without touching the token rate.</text>

  <text x="40" y="{how_y}" fill="{FG}" font-size="18" font-weight="600">How</text>
  <g font-size="14" fill="{TXT}">
    <text x="40" y="{how_y + 32}">1. Expert slices read at their natural 0.6–1.2 MB size over io_uring/O_DIRECT — 23× the bandwidth of 4 KiB demand paging on the same NVMe.</text>
    <text x="40" y="{how_y + 60}">2. Three-tier expert cache: VRAM ({q4["tier_gb"]:.1f} GB, ~{q4["tier_blocks"]:,} experts), 12 GB pinned RAM, NVMe — {q4["hit"]*100:.0f}% of lookups hit, {q4["vram_served"]*100:.0f}% served from VRAM.</text>
    <text x="40" y="{how_y + 88}">3. VRAM-resident experts computed inside each layer's replayed CUDA graph; residency looked up on the device, no host round trip.</text>
    <text x="40" y="{how_y + 116}">4. Next-layer routing predicted from the residual and prefetched while the current layer computes; only the reads a token needs are waited for.</text>
    <text x="40" y="{how_y + 144}">5. Decode attention flat at any context: sparse attention over pooled block keys — the same 0.55 ms per layer at 4K and at 160K.</text>
    <text x="40" y="{how_y + 172}">6. Layer-major prefill streams each layer's experts in 16 MB chunks into VRAM and sweeps the whole batch; the tier lends it the memory and takes it back.</text>
    <text x="40" y="{how_y + 200}">7. OpenAI-compatible server (streaming, tool calling, thinking budget, live stats) and a console with hardware-aware settings from a fitted cost model.</text>
  </g>
{notes}
</svg>'''
open("docs/img/results.svg", "w").write(svg)
subprocess.run(["rsvg-convert", "-w", "1400", "docs/img/results.svg", "-o", "docs/img/results.png"], check=True)
print("rendered docs/img/results.png")
