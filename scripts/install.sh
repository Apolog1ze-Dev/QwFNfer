#!/usr/bin/env bash
# qwfnfer installer, one command on Linux:
#
#   curl -fsSL https://raw.githubusercontent.com/Apolog1ze-Dev/QwFN/master/scripts/install.sh | bash
#
# Downloads the latest release bundle (or unpacks QWFN_ZIP=/path/to/qwfnfer-linux-x86_64-cuda.zip),
# installs it under ~/.local/share/qwfnfer, links ~/.local/bin/qwfnfer, checks the GPU driver and
# the C library, runs the engine once, and says how to get the model. Nothing else touches
# your system: uninstall by deleting those two paths.
set -euo pipefail
REPO=${QWFN_REPO:-Apolog1ze-Dev/QwFN}
NAME=qwfnfer-linux-x86_64-cuda
DEST=${QWFN_HOME:-$HOME/.local/share/qwfnfer}
BIN=${QWFN_BIN:-$HOME/.local/bin}
say() { printf '\033[1m%s\033[0m\n' "$*"; }
die() { printf 'qwfnfer install: %s\n' "$*" >&2; exit 1; }

[ "$(uname -s)" = Linux ] || die "Linux only for now: the engine reads the NVMe through io_uring (Windows needs a port of that layer)"
[ "$(uname -m)" = x86_64 ] || die "x86_64 only"
command -v python3 >/dev/null || die "python3 is needed for the console (apt install python3 / dnf install python3)"
command -v unzip >/dev/null || die "unzip is needed (apt install unzip / dnf install unzip)"
glibc=$(python3 -c "import os; print(os.confstr('CS_GNU_LIBC_VERSION').split()[-1])" 2>/dev/null || echo 0)
python3 -c "import sys; sys.exit(0 if tuple(map(int, '$glibc'.split('.'))) >= (2, 34) else 1)" || die "glibc $glibc is older than 2.34; the bundles need Ubuntu 22.04, Fedora 35, Debian 12 or newer"
if command -v nvidia-smi >/dev/null; then
    drv=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 || true)
    gpu=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1 || true)
    say "GPU: ${gpu:-?}, driver ${drv:-?}"
    [ "${drv%%.*}" -ge 580 ] 2>/dev/null || echo "note: the bundled CUDA 13 runtime needs an NVIDIA driver 580 or newer (yours: ${drv:-unknown})"
else
    echo "note: nvidia-smi not found; the engine needs an NVIDIA GPU with driver 580 or newer"
fi

tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
if [ -n "${QWFN_ZIP:-}" ]; then
    cp "$QWFN_ZIP" "$tmp/$NAME.zip"
else
    url="https://github.com/$REPO/releases/latest/download/$NAME.zip"
    say "Downloading $url"
    curl -fL --progress-bar -o "$tmp/$NAME.zip" "$url" || die "download failed. No release yet? Build from source instead (README, 'Building from source'), or pass QWFN_ZIP=/path/to/$NAME.zip"
fi
say "Installing into $DEST"
rm -rf "$DEST"; mkdir -p "$DEST" "$BIN"
unzip -q "$tmp/$NAME.zip" -d "$tmp/x"
mv "$tmp/x/$NAME"/* "$DEST"/
chmod +x "$DEST/qwfnfer" "$DEST/bin/qwfn-server" "$DEST/bin/qwfn-tok"
need=$(cat "$DEST/GLIBC" 2>/dev/null || echo 2.34)
python3 -c "import sys; v=lambda x: tuple(map(int, x.split('.'))); sys.exit(0 if v('$glibc') >= v('$need') else 1)" || die "this bundle needs glibc $need (yours: $glibc): use a newer distribution, or build from source (README)"
ln -sfn "$DEST/qwfnfer" "$BIN/qwfnfer"

# The engine must load: every library it needs is in bin/ except the driver's libcuda.
missing=$(LD_LIBRARY_PATH="$DEST/bin" ldd "$DEST/bin/qwfn-server" | grep "not found" || true)
[ -z "$missing" ] || die "libraries missing on this system: $missing"
{ LD_LIBRARY_PATH="$DEST/bin" "$DEST/bin/qwfn-server" 2>&1 || true; } | grep -q "usage: qwfn-server" || die "the engine did not start (run: LD_LIBRARY_PATH=$DEST/bin $DEST/bin/qwfn-server)"

hf_hub=${HF_HOME:-$HOME/.cache/huggingface}/hub
if ls "$hf_hub"/*Qwen3.8-Flash-Next*/snapshots/*/*/*.gguf >/dev/null 2>&1; then
    say "Model found in the Hugging Face cache."
else
    say "Get the model once (111 GB; UD-Q3_K_XL is the 90 GB faster choice):"
    echo "    pip install -U huggingface_hub"
    echo "    hf download unsloth/Qwen3.8-Flash-Next-GGUF --include \"UD-Q4_K_XL/*\" \"mmproj-F16.gguf\""
fi
say "Installed qwfnfer $(cat "$DEST/VERSION" 2>/dev/null || echo). Start it with:"
echo "    qwfnfer"
case ":$PATH:" in *":$BIN:"*) ;; *) echo "    ($BIN is not in your PATH: run $BIN/qwfnfer, or add it to PATH)" ;; esac
echo "It opens the console at http://127.0.0.1:8090: pick a tier, press Auto-tune & start."
