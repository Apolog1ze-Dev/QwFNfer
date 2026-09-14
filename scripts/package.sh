#!/usr/bin/env bash
# Build the relocatable Linux release bundle: dist/qwfnfer-linux-x86_64-cuda.zip
#
# The bundle carries everything but the NVIDIA driver: the engine and the tokenizer tool
# (built with QWFN_PORTABLE: x86-64-v3 code, libraries next to the binaries; the glibc
# floor is the build machine's, written to the bundle's GLIBC file and refused by install.sh
# on anything older -- so build on the oldest distribution you mean to support), ggml/llama.cpp's shared libraries from a portable build (GGML_NATIVE=OFF, every
# CPU variant, CUDA architectures 75-120), the CUDA runtime libraries NVIDIA redistributes
# (cudart, cublas, cublasLt), liburing and libgomp, the console, the launcher, the README.
#
#   scripts/package.sh                 uses the defaults below
#   GGML_LIBS=... CUDA_LIBS=... VERSION=v0.3 scripts/package.sh
#
# Inputs:
#   LLAMA_CPP_ROOT  llama.cpp source (ggml headers, vendor/)         default ~/.unsloth/llama.cpp
#   GGML_LIBS       portable ggml/llama shared libraries              default ~/.cache/qwfnfer-build/ggml/bin
#                   Built once from Unsloth's llama.cpp (b10798-mix-659e406, the mix the engine
#                   is validated against), library targets only:
#
#                     H=$PWD/cmake/glibc_compat.h
#                     cmake -S ~/.unsloth/llama.cpp -B ~/.cache/qwfnfer-build/ggml -G Ninja \
#                       -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DGGML_BACKEND_DL=ON \
#                       -DGGML_NATIVE=OFF -DGGML_CPU_ALL_VARIANTS=ON \
#                       -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="75;80;86;89;90;120" \
#                       -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
#                       -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_SERVER=OFF \
#                       -DCMAKE_C_FLAGS="-include $H -fno-math-errno" -DCMAKE_CXX_FLAGS="-include $H -fno-math-errno" \
#                       -DCMAKE_CUDA_FLAGS="-Xcompiler=-fno-math-errno -Xcompiler=-include,$H"
#                     cmake --build ~/.cache/qwfnfer-build/ggml
#
#                   On a distribution whose compiler the CUDA toolkit refuses, add
#                   CC=gcc-12 CXX=g++-12 and -DCMAKE_CUDA_HOST_COMPILER=g++-12.
#   CUDA_LIBS       where libcudart/libcublas/libcublasLt live         default /opt/cuda/lib64 or /usr/local/cuda/lib64
set -euo pipefail
cd "$(dirname "$0")/.."
LLAMA_CPP_ROOT=${LLAMA_CPP_ROOT:-$HOME/.unsloth/llama.cpp}
GGML_LIBS=${GGML_LIBS:-$HOME/.cache/qwfnfer-build/ggml/bin}
if [ -z "${CUDA_LIBS:-}" ]; then
    for d in /opt/cuda/lib64 /usr/local/cuda/lib64 /usr/local/cuda/targets/x86_64-linux/lib; do [ -f "$d/libcudart.so.13" ] && CUDA_LIBS=$d && break; done
fi
: "${CUDA_LIBS:?no CUDA runtime libraries found; set CUDA_LIBS}"
VERSION=${VERSION:-$(git describe --tags --always --dirty 2>/dev/null || date +%Y%m%d)}
NAME=qwfnfer-linux-x86_64-cuda
OUT=dist/$NAME

[ -f "$GGML_LIBS/libggml-cuda.so" ] || { echo "no portable ggml build at $GGML_LIBS (libggml-cuda.so missing)" >&2; exit 1; }
echo "== engine (portable build against $GGML_LIBS)"
cmake -S . -B build-portable -G Ninja -DCMAKE_BUILD_TYPE=Release -DQWFN_PORTABLE=ON \
      -DLLAMA_CPP_ROOT="$LLAMA_CPP_ROOT" -DLLAMA_CPP_BUILD="$GGML_LIBS" > build-portable.cmake.log 2>&1 || { tail -20 build-portable.cmake.log; exit 1; }
cmake --build build-portable --target qwfn-server qwfn-tok | tail -2

echo "== bundle $OUT"
rm -rf "$OUT"; mkdir -p "$OUT/bin" "$OUT/tools/console"
cp build-portable/qwfn-server build-portable/qwfn-tok "$OUT/bin/"
# real files under their sonames: a zip carries no symlinks
for so in libggml-base.so.0 libggml.so.0 libllama.so.0 libggml-cuda.so; do cp -L "$GGML_LIBS/$so" "$OUT/bin/"; done
for so in "$GGML_LIBS"/libggml-cpu-*.so; do cp -L "$so" "$OUT/bin/"; done
for so in libcudart.so.13 libcublas.so.13 libcublasLt.so.13; do cp -L "$CUDA_LIBS/$so" "$OUT/bin/"; done
uring=$(ldd build-portable/qwfn-server | awk '/liburing/ {print $3}'); cp -L "$uring" "$OUT/bin/liburing.so.2"
gomp=$(ldd "$GGML_LIBS/libggml-cpu-haswell.so" | awk '/libgomp/ {print $3}'); [ -n "$gomp" ] && cp -L "$gomp" "$OUT/bin/libgomp.so.1"
cp tools/qwfn_console.py tools/qwfn_router.py "$OUT/tools/"; cp tools/console/index.html "$OUT/tools/console/"
cp scripts/qwfnfer "$OUT/qwfnfer"; chmod +x "$OUT/qwfnfer" "$OUT/bin/qwfn-server" "$OUT/bin/qwfn-tok"
mkdir -p "$OUT/scripts"; cp scripts/claude-desktop.sh "$OUT/scripts/"; chmod +x "$OUT/scripts/claude-desktop.sh"
cp README.md LICENSE "$OUT/"; echo "$VERSION" > "$OUT/VERSION"
cat > "$OUT/INSTALL.txt" <<EOF
qwfnfer $VERSION -- Qwen3.8-Flash-Next on one 16 GB GPU (Linux x86_64, NVIDIA)

1. Unzip anywhere and run:   ./qwfnfer
   (opens the console at http://127.0.0.1:8090; needs python3 and an NVIDIA driver 580 or newer)
2. Get the model once:       hf download unsloth/Qwen3.8-Flash-Next-GGUF --include "UD-Q4_K_XL/*" "mmproj-F16.gguf"
   (pip install -U huggingface_hub for the hf command; the console finds the files in the Hugging Face cache)
3. Pick a tier (Chat, Agentic coding, Agentic coding+ or your own Custom one) and press Auto-tune & start:
   the console measures your drive, threads and memory, picks every flag, starts the server and verifies it.
   The endpoint is http://127.0.0.1:8080/v1. Model locations: add any folder that holds the shards.
Everything the engine needs is in bin/ except the NVIDIA driver. See README.md.
EOF

echo "== checks"
missing=$(LD_LIBRARY_PATH="$PWD/$OUT/bin" ldd "$OUT/bin/qwfn-server" "$OUT/bin/libggml-cuda.so" | grep "not found" || true)
[ -z "$missing" ] || { echo "unresolved libraries:"; echo "$missing"; exit 1; }
outside=$(LD_LIBRARY_PATH="$PWD/$OUT/bin" ldd "$OUT/bin/qwfn-server" "$OUT/bin/libggml-cuda.so" | sed -n 's/.*=> \(.*\) (0x.*/\1/p' | grep -v -F "$PWD/$OUT/bin/" | grep -v -E "/(libc|libm|libdl|libpthread|librt|libgcc_s|libstdc\+\+|libcuda|ld-linux)[.-]" | sort -u || true)
[ -z "$outside" ] || echo "note: resolved outside the bundle (expected only base-system libraries): $outside"
floor=$(for f in "$OUT"/bin/*.so* "$OUT"/bin/qwfn-server; do objdump -T "$f" 2>/dev/null | grep -o "GLIBC_[0-9.]*"; done | sort -V | uniq | tail -1)
echo "glibc floor of the bundle: $floor"; echo "${floor#GLIBC_}" > "$OUT/GLIBC"
if { "$OUT/bin/qwfn-server" 2>&1 || true; } | grep -q "usage: qwfn-server"; then echo "qwfn-server runs (libraries from bin/ via RUNPATH)"; else echo "qwfn-server does not start" >&2; exit 1; fi
python3 - "$OUT" "dist/$NAME.zip" <<'EOF'
import os, sys, zipfile, stat
src, dst = sys.argv[1], sys.argv[2]
with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
    for root, dirs, files in os.walk(src):
        for f in sorted(files):
            p = os.path.join(root, f); arc = os.path.relpath(p, os.path.dirname(src))
            zi = zipfile.ZipInfo.from_file(p, arc); zi.compress_type = zipfile.ZIP_DEFLATED
            zi.external_attr = (stat.S_IMODE(os.stat(p).st_mode) | 0o644) << 16   # keep the exec bits
            with open(p, "rb") as fh: z.writestr(zi, fh.read())
print("wrote", dst, "%.0f MB" % (os.path.getsize(dst) / 1e6))
EOF
du -sh "$OUT" | cut -f1 | xargs echo "unpacked:"
