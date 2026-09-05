#!/usr/bin/env bash
# Start qwfn-server with the flags that measured best for long context, after
# picking the model from what is in the Hugging Face cache.
#
#   scripts/serve.sh                 menu, then start
#   scripts/serve.sh --fast          same, with --skip-miss (+16-29% decode, small NLL cost)
#   scripts/serve.sh --list          just list the models
#   scripts/serve.sh --dry-run       print the command instead of running it
#   scripts/serve.sh --no-vision     leave the vision projector out (images are accepted by
#                                    default when mmproj-*.gguf sits next to the model)
#   scripts/serve.sh 2 --port 8081   pick #2 without asking (the number must come first);
#                                    other flags pass through, e.g. --reserve 1024 on a desktop GPU
#
# Optimal flags (measured on the reference machine, 16 GB GPU + 30 GB RAM):
# --ram 12 --ctx 163840 --batch 4096 --kv q4_0. Change them with
# QWFN_RAM, QWFN_CTX, QWFN_BATCH, QWFN_KV, or by passing the flag explicitly
# (the last occurrence wins in qwfn-server).
set -euo pipefail
cd "$(dirname "$0")/.."

HF=${HF_HOME:-$HOME/.cache/huggingface}/hub
RAM=${QWFN_RAM:-12}; CTX=${QWFN_CTX:-163840}; BATCH=${QWFN_BATCH:-4096}; KV=${QWFN_KV:-q4_0}

# A model number is only recognised as the FIRST argument; anything else passes
# through to qwfn-server (so `--reserve 1024` keeps its value).
pick=""; dry=0; list=0; vision=1; extra=()
if [[ $# -gt 0 && "$1" =~ ^[0-9]+$ ]]; then pick=$1; shift; fi
for a in "$@"; do
    case "$a" in
        --fast)    extra+=(--skip-miss) ;;
        --dry-run) dry=1 ;;
        --list)    list=1 ;;
        --no-vision) vision=0 ;;
        *)         extra+=("$a") ;;
    esac
done

# One line per model: first shard, quant name, total size of its shards.
models=(); names=(); sizes=(); notes=()
while IFS= read -r f; do
    dir=$(dirname "$f"); quant=$(basename "$dir")
    [[ "$quant" == snapshots || "$quant" =~ ^[0-9a-f]{40}$ ]] && quant=$(basename "$f" .gguf)
    total=0
    for s in "$dir"/*.gguf; do total=$(( total + $(stat -Lc %s "$s") )); done
    note=""
    case "$quant" in
        *Q3_K_XL*) note="recommended: 14.7 tok/s at 133K" ;;
        *Q4_K_XL*) note="35-42% slower decode than Q3_K_XL, the quality choice" ;;
        *IQ1_S*)   note="cold checkpoint only, do not serve" ;;
    esac
    models+=("$f"); names+=("$quant"); sizes+=("$total"); notes+=("$note")
done < <(find "$HF" -path "*Qwen3.8-Flash-Next*" -name "*.gguf" \( -name "*-00001-of-*" -o ! -name "*-of-*" \) \
           ! -name "mmproj*" ! -name "mtp-*" 2>/dev/null | sort)

if [ ${#models[@]} -eq 0 ]; then echo "no Qwen3.8-Flash-Next GGUF found under $HF" >&2; exit 1; fi

rec=1
for i in "${!models[@]}"; do [[ "${names[$i]}" == *Q3_K_XL* ]] && rec=$((i + 1)); done
echo "models in $HF:"
for i in "${!models[@]}"; do
    LC_NUMERIC=C printf "  %d) %-14s %5.1f GB   %s\n" $((i + 1)) "${names[$i]}" "$(awk "BEGIN{printf \"%.1f\", ${sizes[$i]} / 1e9}")" "${notes[$i]}"
done
[ $list -eq 1 ] && exit 0

if [ -z "$pick" ]; then
    read -r -p "which one? [$rec] " pick </dev/tty || pick=""
    pick=${pick:-$rec}
fi
if ! [[ "$pick" =~ ^[0-9]+$ ]] || [ "$pick" -lt 1 ] || [ "$pick" -gt ${#models[@]} ]; then echo "bad choice: $pick" >&2; exit 1; fi
model=${models[$((pick - 1))]}

# The machine fits one engine at a time (pinned RAM arena + the VRAM tier).
for n in qwfn-server qwfn-chat qwfn-gen qwfn-gen.late qwfn-gen.prev qwfn-gen.pred; do
    if pgrep -x "$n" >/dev/null; then echo "an engine is already running ($n); stop it first (pkill -x $n)" >&2; exit 1; fi
done
[ -x build/qwfn-server ] || { echo "build/qwfn-server not found; build first (cmake --build build)" >&2; exit 1; }

# The vision projector ships in the snapshot directory above the quant's own (or beside
# the shards): pass it so a harness that sends an image gets an answer, not a 400.
mm=""
if [ $vision -eq 1 ]; then
    for d in "$(dirname "$model")" "$(dirname "$model")/.."; do
        for f in "$d"/mmproj*.gguf; do [ -e "$f" ] && { mm=$f; break 2; }; done
    done
    [ -n "$mm" ] && extra+=(--mmproj "$mm")
fi
cmd=(./build/qwfn-server "$model" --ram "$RAM" --ctx "$CTX" --batch "$BATCH" --kv "$KV" "${extra[@]}")
echo "starting: ${cmd[*]}" | sed "s|$HF/||"
[ $dry -eq 1 ] && exit 0
exec "${cmd[@]}"
