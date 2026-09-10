#!/usr/bin/env bash
# Replay benchmark: one fixed token sequence through a configuration, so every
# configuration sees identical routing (GPU decode is nondeterministic and
# comparing different generated texts produces phantom variance).
#
#   scripts/bench_replay.sh <name> [extra qwfn-gen flags...]
#
#   SCENARIO=short   a framed one-line question, thinking on (default)
#   SCENARIO=doc     the engineering log and the roadmap attached (~40K tokens), then a question
#   MODEL, CTX (131072), GEN (200), RAM (12), VRAM (12) override the defaults; GEN_BIN points
#   the replay run at another qwfn-gen binary for a binary-vs-binary A/B.
#
# The first call for a scenario builds its inputs: the prompt through the model's
# tokenizer (qwfn-tok) and, with one greedy generation at the default settings,
# the replay sequence. Both live in bench/replay/ and are reused after that.
# Every run appends one line to bench/replay/summary.tsv.
set -u
cd "$(dirname "$0")/.."
MODEL=${MODEL:-$HOME/.cache/huggingface/hub/models--unsloth--Qwen3.8-Flash-Next-GGUF/snapshots/5d16c055a7c5cb276e721ee154f9c22420dde2a1/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf}
SCEN=${SCENARIO:-short}; GEN=${GEN:-200}; CTX=${CTX:-131072}; RAM=${RAM:-12}; VRAM=${VRAM:-12}
GEN_BIN=${GEN_BIN:-./build/qwfn-gen}   # an older binary for an A/B (e.g. build/qwfn-gen.pre-pack)
DIR=bench/replay; mkdir -p "$DIR"
PROMPT=$DIR/$SCEN.prompt; REPLAY=$DIR/$SCEN.replay
COMMON=(--ram "$RAM" --vram "$VRAM" --ctx "$CTX" --kv q4_0 --batch 4096 --gen "$GEN")
FILTER="CUDA graph warmup\|^load_backend\|^ggml_cuda_init\|Device 0\|cudaMalloc failed"

for n in qwfn-server qwfn-chat qwfn-gen; do
    if pgrep -x "$n" >/dev/null; then echo "an engine is already running ($n); stop it first" >&2; exit 1; fi
done
[ $# -ge 1 ] || { echo "usage: $0 <name> [qwfn-gen flags...]" >&2; exit 1; }
name=$1; shift

if [ ! -s "$PROMPT" ]; then
    case "$SCEN" in
        short) ./build/qwfn-tok "$MODEL" --chat "Explain in three sentences why mixture-of-experts models are cheaper to run than dense models with the same parameter count." > "$PROMPT" ;;
        doc)   ./build/qwfn-tok "$MODEL" --chat "Using only the attached engineering log and roadmap, name the two largest remaining decode costs on the Q4 file and say what the roadmap proposes for each." --file docs/ENGINEERING.md --file ROADMAP.md > "$PROMPT" ;;
        code)  ./build/qwfn-tok "$MODEL" --chat "Write a Python module with a class LRUCache(capacity) offering get(key) and put(key, value) in O(1) using a doubly linked list and a dict, plus a small unittest suite covering eviction order, update-on-access and capacity 1. Code only, no explanations." --think off > "$PROMPT" ;;
        *) echo "unknown SCENARIO $SCEN" >&2; exit 1 ;;
    esac
    echo "prompt: $(wc -w < "$PROMPT") tokens -> $PROMPT"
fi
if [ ! -s "$REPLAY" ]; then
    echo "generating the replay sequence (greedy, default settings)..."
    ./build/qwfn-gen "$MODEL" --prompt-file "$PROMPT" --save-replay "$REPLAY" "${COMMON[@]}" 2>&1 | grep -v "$FILTER" > "$DIR/$SCEN.gen.log"
    echo "replay: $(wc -l < "$REPLAY") tokens -> $REPLAY"
fi

log=$DIR/$SCEN.$name.log
"$GEN_BIN" "$MODEL" --prompt-file "$PROMPT" --replay-file "$REPLAY" --ppl "${COMMON[@]}" "$@" 2>&1 | grep -v "$FILTER" > "$log"

python3 - "$log" "$SCEN" "$name" "$*" <<'PY'
import re, sys, os
log, scen, name, flags = sys.argv[1:5]
t = open(log).read()
def g(pat, d="-"):
    m = re.search(pat, t); return m.group(1) if m else d
row = {
    "scenario": scen, "name": name,
    "tok/s": g(r"decode: \d+ tokens in [\d.]+ s\s+\(([\d.]+) tok/s\)"),
    "NLL": g(r"replay NLL: ([\d.]+)"),
    "hit%": g(r"expert cache: ([\d.]+)% hit"), "vram%": g(r"hit, ([\d.]+)% from VRAM"),
    "diskGB": g(r"([\d.]+) GB from disk"),
    "graphA_s": g(r"graphA\(GPU\) ([\d.]+) s"), "moecpu_s": g(r"MoE cpu ([\d.]+) s"), "io_s": g(r"\| io ([\d.]+) s"),
    "pred%": g(r"prefetch: ([\d.]+)% of the next"),
    "issued": g(r"\| (\d+) issued"), "used": g(r"issued, (\d+) used"), "wasted": g(r"(\d+) wasted"),
    "gated": g(r"prefetch gate: (\d+) predicted", "0"),
    "rec_us": g(r"recurrent (\d+) us"), "attn_us": g(r"attention (\d+) us \(x"),
    "prefill_tok/s": g(r"prefill: \d+ tokens in [\d.]+ s\s+\(([\d.]+) tok/s\)"),
    "flags": flags,
}
summ = "bench/replay/summary.tsv"
new = not os.path.exists(summ)
with open(summ, "a") as f:
    if new: f.write("\t".join(row.keys()) + "\n")
    f.write("\t".join(str(v) for v in row.values()) + "\n")
print(" ".join(f"{k}={v}" for k, v in row.items() if k != "flags"))
for line in t.splitlines():
    if line.startswith(("prediction by", "prefetch gate")): print("   ", line)
PY
