#!/usr/bin/env bash
# Build the relocatable Linux release bundle: dist/qwfnfer-linux-x86_64-cuda.zip
#
# The bundle carries everything but the NVIDIA driver: the engine and the tokenizer tool
# (built with QWFN_PORTABLE: x86-64-v3 code, libraries next to the binaries; the glibc
# floor is the build machine's, written to the bundle's GLIBC file and refused by install.sh
# on anything older -- so build on the oldest distribution you mean to support), ggml/llama.cpp's shared libraries from a portable build (GGML_NATIVE=OFF, every
# CPU variant, CUDA architectures 75-120), the CUDA runtime libraries NVIDIA redistributes
# (cudart, cublas, cublasLt), the console, the launcher, the README.
#
# The C++ and OpenMP runtimes and liburing are fetched from Ubuntu's archive rather than copied
# off this machine, and cached in RUNTIME_LIBS. A copied library is built for the build machine's CPU,
# and a distribution that compiles its packages for AVX-512 puts AVX-512 into all of them: into
# libgomp and liburing as instructions that fault where they are not supported, and into
# libstdc++.a and libgcc.a -- which QWFN_PORTABLE used to link statically -- as an ISA property
# the linker ORs into every binary, marking it "x86-64-v4 needed" whatever -march the engine
# itself was built with and leaving glibc's loader to refuse it on every CPU without AVX-512.
# Ubuntu's amd64 packages are plain x86-64. The ISA check below fails the build if v4 reaches
# the bundle anyway: the startup check cannot see it, running as it does on the one machine
# guaranteed to support whatever it has just compiled.
#
#   scripts/package.sh                 uses the defaults below
#   GGML_LIBS=... CUDA_LIBS=... VERSION=v0.3 scripts/package.sh
#   RUNTIME_LIBS=/path/to/dir scripts/package.sh    four .so files you supply, instead of Ubuntu's
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
#   RUNTIME_LIBS    libstdc++.so.6, libgcc_s.so.1, libgomp.so.1,        default ~/.cache/qwfnfer-build/runtime
#                   liburing.so.2. Downloaded on the first run and kept, each .deb checked
#                   against the SHA256 in the archive's own index; put your own four .so files
#                   there to skip the download entirely.
#   RUNTIME_SUITE   the release they come from                          default noble (24.04 LTS)
#   RUNTIME_MIRROR  the archive to take them from                       default https://archive.ubuntu.com/ubuntu
#                   noble is GCC 14: libstdc++ carries GLIBCXX_3.4.33, which covers libllama's
#                   3.4.31, and the four of them need no more than GLIBC_2.38, which is at or
#                   below the floor the engine sets anyway. Both are checked before the zip is
#                   written, so a different release that does not hold up fails the build.
set -euo pipefail
cd "$(dirname "$0")/.."
LLAMA_CPP_ROOT=${LLAMA_CPP_ROOT:-$HOME/.unsloth/llama.cpp}
GGML_LIBS=${GGML_LIBS:-$HOME/.cache/qwfnfer-build/ggml/bin}
if [ -z "${CUDA_LIBS:-}" ]; then
    for d in /opt/cuda/lib64 /usr/local/cuda/lib64 /usr/local/cuda/targets/x86_64-linux/lib; do [ -f "$d/libcudart.so.13" ] && CUDA_LIBS=$d && break; done
fi
: "${CUDA_LIBS:?no CUDA runtime libraries found; set CUDA_LIBS}"
VERSION=${VERSION:-$(git describe --tags --always --dirty 2>/dev/null || date +%Y%m%d)}
RUNTIME_LIBS=${RUNTIME_LIBS:-$HOME/.cache/qwfnfer-build/runtime}
RUNTIME_SUITE=${RUNTIME_SUITE:-noble}
RUNTIME_SOS="libstdc++.so.6 libgcc_s.so.1 libgomp.so.1 liburing.so.2"
NAME=qwfnfer-linux-x86_64-cuda
OUT=dist/$NAME

[ -f "$GGML_LIBS/libggml-cuda.so" ] || { echo "no portable ggml build at $GGML_LIBS (libggml-cuda.so missing)" >&2; exit 1; }
echo "== engine (portable build against $GGML_LIBS)"
cmake -S . -B build-portable -G Ninja -DCMAKE_BUILD_TYPE=Release -DQWFN_PORTABLE=ON \
      -DLLAMA_CPP_ROOT="$LLAMA_CPP_ROOT" -DLLAMA_CPP_BUILD="$GGML_LIBS" > build-portable.cmake.log 2>&1 || { tail -20 build-portable.cmake.log; exit 1; }
cmake --build build-portable --target qwfn-server qwfn-tok | tail -2

have_runtime=yes
for so in $RUNTIME_SOS; do [ -f "$RUNTIME_LIBS/$so" ] || have_runtime=no; done
if [ "$have_runtime" = no ]; then
    mirror=${RUNTIME_MIRROR:-https://archive.ubuntu.com/ubuntu}
    echo "== runtime libraries ($RUNTIME_SUITE from $mirror -> $RUNTIME_LIBS)"
    work=$RUNTIME_LIBS/.work; rm -rf "$work"; mkdir -p "$work"
    curl -fsSL "$mirror/dists/$RUNTIME_SUITE/main/binary-amd64/Packages.xz" | xz -d > "$work/Packages" \
        || { echo "cannot read the $RUNTIME_SUITE package index from $mirror" >&2; exit 1; }
    for pkg in libstdc++6 libgcc-s1 libgomp1 liburing2; do
        # the index gives the pool path, so nothing here has a package version baked into it
        path=$(awk -v p="$pkg" '$1=="Package:"{c=($2==p)} c&&$1=="Filename:"{print $2; exit}' "$work/Packages")
        sha=$(awk -v p="$pkg" '$1=="Package:"{c=($2==p)} c&&$1=="SHA256:"{print $2; exit}' "$work/Packages")
        [ -n "$path" ] || { echo "$pkg is not in $RUNTIME_SUITE" >&2; exit 1; }
        curl -fsSL -o "$work/$pkg.deb" "$mirror/$path" || { echo "cannot download $mirror/$path" >&2; exit 1; }
        # these end up in a release, so take the index's word for what the file should be
        [ -z "$sha" ] || echo "$sha  $work/$pkg.deb" | sha256sum -c --status \
            || { echo "$pkg.deb does not match the SHA256 the index gives for it" >&2; exit 1; }
        data=$(ar t "$work/$pkg.deb" | grep '^data\.tar' | head -1)
        case $data in
            *.xz)  ar p "$work/$pkg.deb" "$data" | tar -xJf - -C "$work" ;;
            # Debian compresses data.tar with xz, Ubuntu with zstd, and tar shells out for it
            *.zst) command -v zstd > /dev/null || { echo "$RUNTIME_SUITE ships $data; install zstd to unpack it" >&2; exit 1; }
                   ar p "$work/$pkg.deb" "$data" | tar --zstd -xf - -C "$work" ;;
            *)     ar p "$work/$pkg.deb" "$data" | tar -xzf - -C "$work" ;;
        esac
    done
    for so in $RUNTIME_SOS; do
        # The package ships the real file under its full version (libstdc++.so.6.0.33) and a
        # symlink, so -type f picks the file. Searching the whole tree rather than a named
        # libdir: a merged-/usr release has no lib/x86_64-linux-gnu, and find failing on a
        # directory that is not there would take the build down through pipefail.
        src=$(find "$work" -name "$so*" -type f 2>/dev/null | sort | head -1 || true)
        [ -n "$src" ] || { echo "$so is not in the $RUNTIME_SUITE packages" >&2; exit 1; }
        cp "$src" "$RUNTIME_LIBS/$so"
    done
    rm -rf "$work"
    echo "fetched: $(cd "$RUNTIME_LIBS" && ls $RUNTIME_SOS | tr '\n' ' ')"
fi

echo "== bundle $OUT"
rm -rf "$OUT"; mkdir -p "$OUT/bin" "$OUT/tools/console"
cp build-portable/qwfn-server build-portable/qwfn-tok "$OUT/bin/"
# real files under their sonames: a zip carries no symlinks
for so in libggml-base.so.0 libggml.so.0 libllama.so.0 libggml-cuda.so; do cp -L "$GGML_LIBS/$so" "$OUT/bin/"; done
for so in "$GGML_LIBS"/libggml-cpu-*.so; do cp -L "$so" "$OUT/bin/"; done
for so in libcudart.so.13 libcublas.so.13 libcublasLt.so.13; do cp -L "$CUDA_LIBS/$so" "$OUT/bin/"; done
for so in $RUNTIME_SOS; do cp -L "$RUNTIME_LIBS/$so" "$OUT/bin/"; done
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
outside=$(LD_LIBRARY_PATH="$PWD/$OUT/bin" ldd "$OUT/bin/qwfn-server" "$OUT/bin/libggml-cuda.so" | sed -n 's/.*=> \(.*\) (0x.*/\1/p' | grep -v -F "$PWD/$OUT/bin/" | grep -v -E "/(libc|libm|libdl|libpthread|librt|libcuda|ld-linux)[.-]" | sort -u || true)
[ -z "$outside" ] || echo "note: resolved outside the bundle (expected only base-system libraries): $outside"
floor=$(for f in "$OUT"/bin/*.so* "$OUT"/bin/qwfn-server; do objdump -T "$f" 2>/dev/null | grep -o "GLIBC_[0-9.]*"; done | sort -V | uniq | tail -1)
echo "glibc floor of the bundle: $floor"; echo "${floor#GLIBC_}" > "$OUT/GLIBC"
# grep finds nothing in a binary that imports no GLIBCXX at all, and pipefail would take the
# whole build down with it, so every one of these is allowed to come back empty.
cxxneed=$(for f in "$OUT"/bin/qwfn-server "$OUT"/bin/qwfn-tok "$OUT"/bin/libggml*.so* "$OUT"/bin/libllama.so.0; do objdump -T "$f" 2>/dev/null | grep -o "GLIBCXX_[0-9.]*" || true; done | sort -V | uniq | tail -1)
cxxhave=$({ readelf -V "$OUT/bin/libstdc++.so.6" 2>/dev/null | grep -o "GLIBCXX_[0-9.]*" || true; } | sort -V | uniq | tail -1)
if [ -n "$cxxneed" ] && [ "$(printf '%s\n%s\n' "$cxxneed" "$cxxhave" | sort -V | tail -1)" != "$cxxhave" ]; then
    echo "the bundled libstdc++ carries $cxxhave, but the engine and the ggml libraries want $cxxneed" >&2
    echo "take the runtime from a newer release: rm -rf $RUNTIME_LIBS && RUNTIME_SUITE=<newer> scripts/package.sh" >&2
    exit 1
fi
echo "libstdc++: bundled $cxxhave, needed $cxxneed"
# Every CPU the bundle claims to support must be able to load it: x86-64-v3 is the floor the
# engine is compiled for, so anything marked v4 (AVX-512) here came off a machine that builds
# for AVX-512, and would fail on Intel 12th-14th gen consumer parts and Zen 1-3.
# libggml-cpu-*.so are exempt by design: ggml dlopens the variant the CPU supports.
command -v readelf > /dev/null || { echo "readelf (binutils) is needed to check the bundle's ISA level" >&2; exit 1; }
# The CRT objects every executable links -- Scrt1.o, crti.o and crtn.o from glibc, crtbeginS.o
# and crtendS.o from gcc -- come from the build machine's packages. Where those are compiled
# for AVX-512 they carry GNU_PROPERTY_X86_ISA_1_NEEDED=v4 and almost no code at all, and the
# linker ORs that property into everything linked against them: the binary is marked "needs
# AVX-512" without holding a single AVX-512 instruction, and glibc's loader refuses it anyway.
# The ISA *used* property is the arbiter. Where it shows no v4 the marker is false and its v4
# bit is cleared here, leaving the rest of the note -- CET among it -- alone. Where it shows v4
# there is real AVX-512 in the file and the check below fails the build, as it should.
for eng in qwfn-server qwfn-tok; do
    f="$OUT/bin/$eng"
    needed=$(readelf -n "$f" 2>/dev/null | grep -oE "x86 ISA needed:.*" || true)
    used=$(readelf -n "$f" 2>/dev/null | grep -oE "x86 ISA used:.*" || true)
    case $needed in *x86-64-v4*) ;; *) continue ;; esac
    case $used in *x86-64-v4*) continue ;; esac
    objcopy --dump-section .note.gnu.property="$OUT/note.bin" "$f" 2>/dev/null || continue
    if python3 - "$OUT/note.bin" <<'PY'
import struct, sys
# GNU_PROPERTY_X86_ISA_1_NEEDED, cumulative level bits: baseline 1, v2 2, v3 4, v4 8.
# Rewritten in place, so the note keeps its size and every other property it carries.
KEEP, ISA_NEEDED = 0x7, 0xc0008002
b = bytearray(open(sys.argv[1], "rb").read())
o, changed = 0, False
while o + 12 <= len(b):
    namesz, descsz, ntype = struct.unpack_from("<III", b, o)
    name = o + 12; desc = name + ((namesz + 3) & ~3); end = desc + descsz
    if ntype == 5 and bytes(b[name:name + 4]) == b"GNU\0":
        q = desc
        while q + 8 <= end:
            ptype, psz = struct.unpack_from("<II", b, q)
            if ptype == 0 and psz == 0: break
            if ptype == ISA_NEEDED and psz == 4:
                v, = struct.unpack_from("<I", b, q + 8)
                nv = (v & KEEP) or 0x1
                if nv != v: struct.pack_into("<I", b, q + 8, nv); changed = True
            q = q + 8 + ((psz + 7) & ~7)
    o = end + ((-end) & 7)
open(sys.argv[1], "wb").write(b)
sys.exit(0 if changed else 1)
PY
    then
        objcopy --update-section .note.gnu.property="$OUT/note.bin" "$f"
        echo "$eng: cleared a false x86-64-v4 marker left by this machine's CRT objects (the file has no AVX-512 instruction)"
    fi
    rm -f "$OUT/note.bin"
done
v4=""; checked=""
for so in qwfn-server qwfn-tok libggml-base.so.0 libggml.so.0 libllama.so.0 libggml-cuda.so $RUNTIME_SOS; do
    checked="$checked $so"
    if readelf -n "$OUT/bin/$so" 2>/dev/null | grep -qE "x86 ISA (needed|used):.*x86-64-v4"; then v4="$v4 $so"; fi
done
if [ -n "$v4" ]; then
    echo "AVX-512 (x86-64-v4) in:$v4" >&2
    for so in $v4; do readelf -n "$OUT/bin/$so" | grep -oE "x86 ISA (needed|used):.*" | sed "s|^|  $so: |" >&2; done
    echo "this bundle would not start, or would fault, on any CPU without AVX-512. The engine is" >&2
    echo "built -march=x86-64-v3 and the ggml libraries GGML_NATIVE=OFF, so whatever is listed above" >&2
    echo "was linked against, or copied from, something built for AVX-512. For libstdc++, libgcc_s," >&2
    echo "libgomp or liburing: rm -rf $RUNTIME_LIBS and let the archive's be fetched again." >&2
    exit 1
fi
echo "ISA level: x86-64-v3, no AVX-512 in$checked"
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
