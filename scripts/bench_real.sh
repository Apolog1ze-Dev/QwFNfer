#!/usr/bin/env bash
# Realistic-workload benchmark for qwfn-chat: real text through the tokenizer and
# chat template, thinking on where a user would have it on, attachments for the
# mid/long prompts, and a multi-turn conversation. Prints the per-turn
# "[prefill N tok X tok/s | generated N tok in T s (X tok/s)]" lines.
#
#   scripts/bench_real.sh [extra qwfn-chat flags...]
#
# MODEL can be overridden in the environment.
set -u
cd "$(dirname "$0")/.."
MODEL=${MODEL:-/home/apolog1ze/.cache/huggingface/hub/models--unsloth--Qwen3.8-Flash-Next-GGUF/snapshots/c8b5954a88c2775c546b92593eda40ea041d3176/UD-Q3_K_XL/Qwen3.8-Flash-Next-UD-Q3_K_XL-00001-of-00003.gguf}
CHAT=./build/qwfn-chat
COMMON=(--ram 12 "$@")

run() {   # name, then qwfn-chat args
    local name=$1; shift
    echo "=================== $name ==================="
    local t0=$(date +%s.%N)
    "$CHAT" "$MODEL" "${COMMON[@]}" "$@" 2>&1 | grep -v "CUDA graph warmup\|^load_backend\|^ggml_cuda_init\|Device 0\|cudaMalloc failed"
    echo "[wall $(python3 -c "print(round($(date +%s.%N) - $t0, 1))") s incl. load]"
    echo
}

# W1: short question, thinking on (the default a user gets), bounded reply.
run "W1 short question, think xhigh, max 400" \
    --max 400 -p "Explain in three sentences why mixture-of-experts models are cheaper to run than dense models with the same parameter count."

# W2: a ~500-token prompt: a small header attached plus a question, no thinking.
run "W2 mid prompt (~500 tok), think off, max 150" \
    --think off --max 150 -f src/qwfn_vocab.h -p "Summarise what this header does in two sentences."

# W3: a ~2500-token prompt: a whole tool source attached, no thinking.
run "W3 long prompt (~2500 tok), think off, max 200" \
    --think off --max 200 -f tools/qwfn_gen.cpp -p "What command-line flags does this tool accept? List them briefly, one per line."

# W4: three-turn chat over stdin, short turns, no thinking.
echo "=================== W4 three-turn chat, think off, max 120 ==================="
t0=$(date +%s.%N)
printf '%s\n' \
    "What is the capital of France?" \
    "Roughly how many people live there, and is that the city proper or the metro area?" \
    "Name two museums there and one thing each is known for." \
    "/quit" | "$CHAT" "$MODEL" "${COMMON[@]}" --think off --max 120 2>&1 \
    | grep -v "CUDA graph warmup\|^load_backend\|^ggml_cuda_init\|Device 0\|cudaMalloc failed"
echo "[wall $(python3 -c "print(round($(date +%s.%N) - $t0, 1))") s incl. load]"
