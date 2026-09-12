#!/usr/bin/env python3
"""qwfn console: a local web page that finds the downloaded models (the Hugging Face
cache and any folder you add), sizes the server's flags for this machine behind four
tiers (Chat, Agentic coding, Agentic coding+ and your own saved Custom tier),
auto-tunes them on the hardware it finds (the drive's read speed, a thread sweep on
the running server, the KV precision the GPU has room for, the RAM tier the memory
allows, then a measured verification), starts/stops qwfn-server, chats with it and
watches it live (prefill progress, input / cached / output tokens, tokens/s, the
last request, the session's totals).

    python3 tools/qwfn_console.py                 # http://127.0.0.1:8090
    python3 tools/qwfn_console.py --start         # and start the last served model and tier
    python3 tools/qwfn_console.py --port 8091 --server-port 8080

Standard library only. Binds 127.0.0.1. One engine at a time: the console starts
one qwfn-server and refuses to start a second while any qwfn engine is running.
State lives in ~/.cache/qwfn-console: config.json (model locations, custom tiers,
tune results, the last served model), server.log, selftest.json.
"""
import argparse, glob, http.server, json, math, mmap, os, random, signal, socket, struct, subprocess, sys, threading, time, urllib.parse, urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HF = os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub")
# The engine: bin/ in the release bundle, build/ in a source checkout, or QWFN_SERVER.
# The engine: bin/ in the release bundle, build/ in a source checkout (on
# Windows a separate build-win/ is common to keep a Linux build around), or
# QWFN_SERVER.
_EXE = "qwfn-server" + (".exe" if os.name == "nt" else "")
_CANDS = [os.path.join(ROOT, "bin", _EXE), os.path.join(ROOT, "build", _EXE)]
if os.name == "nt":
    _CANDS.append(os.path.join(ROOT, "build-win", _EXE))
SERVER_BIN = os.environ.get("QWFN_SERVER") or next((p for p in _CANDS if os.path.exists(p)), _CANDS[1])
LOG_DIR = os.path.join(os.path.expanduser("~/.cache"), "qwfn-console")
os.makedirs(LOG_DIR, exist_ok=True)
CONFIG_FILE = os.path.join(LOG_DIR, "config.json")

STATE = {"proc": None, "model": None, "settings": None, "started": 0.0, "log": os.path.join(LOG_DIR, "server.log"), "port": 8080, "ext_model": None}
LOCK = threading.Lock()

# ---- persistent config: model locations, custom tiers, tune results, the last served model
CONFIG = {"locations": [], "custom": {}, "tune": {}, "last": {}, "headroom_gb": 3.0}
def load_config():
    try:
        c = json.load(open(CONFIG_FILE))
        for k in CONFIG:
            if k in c and (isinstance(c[k], type(CONFIG[k])) or (isinstance(CONFIG[k], float) and isinstance(c[k], (int, float)))): CONFIG[k] = c[k]
    except Exception:
        pass
def headroom_gb():
    try: return min(16.0, max(0.5, float(CONFIG.get("headroom_gb") or 3.0)))
    except Exception: return 3.0
def save_config():
    try:
        tmp = CONFIG_FILE + ".tmp"
        json.dump(CONFIG, open(tmp, "w"), indent=1); os.replace(tmp, CONFIG_FILE)
    except Exception:
        pass
load_config()

# ---- models ------------------------------------------------------------------
QUANT_BLOCK_MB = {"Q3_K_XL": 2.27, "Q4_K_XL": 3.13, "IQ1_S": 1.62}   # routed-expert bytes per block (GGUF headers, 2026-09-10); the tiers hold blocks of this size
QUANT_CORE_GB  = {"Q3_K_XL": 4.68, "Q4_K_XL": 4.83, "IQ1_S": 4.5}  # dense core on the GPU
COLD_ONLY = ("IQ1_S",)   # 1-bit checkpoints: a cold tier (--cold), never served on their own
ARCH = "qwen4exp"        # the architecture the engine runs

def quant_of(name):
    return next((k for k in QUANT_BLOCK_MB if k in name), None)

def shard_stem(base):
    """Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf -> Qwen3.8-Flash-Next-UD-Q4_K_XL"""
    b = base[:-5] if base.endswith(".gguf") else base
    i = b.find("-of-")
    if i > 6 and b[i - 6] == "-" and b[i - 5:i].isdigit(): b = b[:i - 6]
    return b

_ARCH_CACHE = {}
def gguf_arch(path):
    """general.architecture from the GGUF header (the first keys), None if unreadable."""
    try:
        st = os.stat(path); key = (path, st.st_mtime, st.st_size)
        if key in _ARCH_CACHE: return _ARCH_CACHE[key]
        with open(path, "rb") as f:
            if f.read(4) != b"GGUF": return None
            ver, = struct.unpack("<I", f.read(4))
            if ver < 2: return None
            _, n_kv = struct.unpack("<QQ", f.read(16))
            sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
            def rd_str():
                n, = struct.unpack("<Q", f.read(8)); return f.read(n).decode("utf-8", "replace")
            def skip(t):
                if t == 8: rd_str()
                elif t == 9:
                    et, = struct.unpack("<I", f.read(4)); n, = struct.unpack("<Q", f.read(8))
                    if n > 4096: raise StopIteration   # the tokenizer's arrays: the architecture key comes before them
                    for _ in range(n): skip(et)
                else: f.read(sizes[t])
            arch = None
            try:
                for _ in range(min(n_kv, 96)):
                    k = rd_str(); t, = struct.unpack("<I", f.read(4))
                    if k == "general.architecture" and t == 8: arch = rd_str(); break
                    skip(t)
            except StopIteration:
                pass
            _ARCH_CACHE[key] = arch
            return arch
    except Exception:
        return None

def is_hf_hub(path):
    return os.path.isdir(path) and bool(glob.glob(os.path.join(path, "models--*")))

def model_locations():
    locs = [{"path": HF, "kind": "hf", "builtin": True, "exists": os.path.isdir(HF)}]
    for p in CONFIG["locations"]:
        if p == HF: continue
        locs.append({"path": p, "kind": "file" if os.path.isfile(p) else ("hf" if is_hf_hub(p) else "dir"), "builtin": False, "exists": os.path.exists(p)})
    return locs

def list_gguf(loc):
    p = loc["path"]
    if loc["kind"] == "file": return [p] if p.endswith(".gguf") else []
    if loc["kind"] == "hf":
        return glob.glob(os.path.join(p, "models--*", "snapshots", "*", "*.gguf")) + glob.glob(os.path.join(p, "models--*", "snapshots", "*", "*", "*.gguf"))
    out = []
    base_depth = p.rstrip("/\\").count("/") + p.rstrip("/\\").count("\\")
    for d, dirs, files in os.walk(p, followlinks=True):
        dirs[:] = [x for x in dirs if not x.startswith(".")]
        depth = d.count("/") + d.count("\\")
        if depth - base_depth >= 4: dirs[:] = []
        out += [os.path.join(d, f) for f in files if f.endswith(".gguf")]
    return out

def repo_dirs(model_dir):
    """Every snapshot directory of the model's Hugging Face repo, when it is laid out that way."""
    parts = os.path.normpath(model_dir).split(os.sep)   # normpath first: glob may hand back /
    if "snapshots" in parts:
        i = len(parts) - 1 - parts[::-1].index("snapshots")
        return glob.glob(os.path.join(os.sep.join(parts[:i]), "snapshots", "*"))
    return []

def find_mmproj(model_dir):
    """The vision projector shipped with the model: mmproj-*.gguf next to the shards, in the
    directory above, or in any snapshot directory of the same repo."""
    for d in [model_dir, os.path.dirname(model_dir)] + repo_dirs(model_dir):
        c = sorted(glob.glob(os.path.join(d, "mmproj*.gguf")))
        if c: return c[0]
    return None

def find_mtp(model_dir):
    """The checkpoint's nextn draft head (MTP/mtp-*.gguf), next to the shards, above them, or in
    any snapshot directory of the repo (Hugging Face may have put it in another one)."""
    for d in [model_dir, os.path.dirname(model_dir)] + repo_dirs(model_dir):
        c = sorted(glob.glob(os.path.join(d, "MTP", "mtp-*.gguf")) + glob.glob(os.path.join(d, "mtp-*.gguf")))
        if c: return c[0]
    return None

def repo_label(path, loc):
    parts = os.path.normpath(path).split(os.sep)
    for x in parts:
        if x.startswith("models--"): return x[8:].replace("--", "/")
    return os.path.relpath(os.path.dirname(path), loc["path"]) if loc["kind"] != "file" else os.path.dirname(path)

def scan_models():
    out, seen = [], set()
    for loc in model_locations():
        if not loc["exists"]: continue
        for f in sorted(list_gguf(loc)):
            base = os.path.basename(f)
            if base.startswith(("mmproj", "mtp-")) or ("-of-" in base and "-00001-of-" not in base): continue
            real = os.path.realpath(f)
            if real in seen: continue
            d = os.path.dirname(f); stem = shard_stem(base)
            arch = gguf_arch(f)
            if arch is not None and arch != ARCH: continue
            if arch is None and "Qwen3.8-Flash-Next" not in f: continue
            seen.add(real)
            qdir = os.path.basename(d); key = quant_of(qdir) or quant_of(base)
            if key in COLD_ONLY: continue
            name = qdir if quant_of(qdir) else stem
            shards = [s for s in glob.glob(os.path.join(d, "*.gguf")) if os.path.basename(s).startswith(stem)]
            total = sum(os.stat(s).st_size for s in shards)
            # The first shard of a split GGUF can be tiny (this quant's holds 11 MB of metadata):
            # the drive probe reads the largest one, where the experts are.
            probe_file = max(shards, key=lambda s: os.stat(s).st_size) if shards else f
            mm = find_mmproj(d); mtp = find_mtp(d)
            out.append({"id": len(out), "name": name, "path": f, "dir": d, "location": loc["path"], "repo": repo_label(f, loc),
                        "size_gb": round(total / 1e9, 1), "shards": len(shards), "probe_file": probe_file, "quant": key or name, "known_quant": bool(key),
                        "mmproj": mm, "mmproj_gb": round(os.stat(mm).st_size / 1e9, 2) if mm else 0.0,
                        "mtp": mtp, "mtp_gb": round(os.stat(mtp).st_size / 1e9, 2) if mtp else 0.0})
    return out

def find_model(models, ref):
    """A model by path (the page's key) or by index (older callers)."""
    if ref is None or ref == "": return models[0] if models else None
    s = str(ref)
    m = next((m for m in models if m["path"] == s or m["name"] == s), None)
    if m: return m
    if s.isdigit() and models: return models[min(int(s), len(models) - 1)]
    return None

# ---- hardware ----------------------------------------------------------------
def hardware():
    hw = {"gpu": None, "vram_total_mb": 0, "vram_used_mb": 0, "desktop_gpu": False, "ram_total_gb": 0, "ram_available_gb": 0, "cpu_threads": os.cpu_count() or 1, "cpu": ""}
    try:
        q = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5).stdout.strip().splitlines()
        if q:
            name, tot, used = [x.strip() for x in q[0].split(",")]
            hw.update(gpu=name, vram_total_mb=int(float(tot)), vram_used_mb=int(float(used)))
        apps = subprocess.run(["nvidia-smi", "--query-compute-apps=process_name,used_memory", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5).stdout
        others = [l for l in apps.splitlines() if l.strip() and "qwfn" not in l]
        hw["desktop_gpu"] = bool(others) or bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
        hw["other_gpu_apps"] = len(others)
        hw["qwfn_vram_mb"] = sum(int(float(l.split(",")[1])) for l in apps.splitlines() if "qwfn" in l and "," in l)
    except Exception:
        pass
    try:
        if os.name == "nt":
            import ctypes
            class MS(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            ms = MS(); ms.dwLength = ctypes.sizeof(MS)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms)):
                hw["ram_total_gb"] = round(ms.ullTotalPhys / (1024 ** 3), 1)
                hw["ram_available_gb"] = round(ms.ullAvailPhys / (1024 ** 3), 1)
        else:
            mi = {}
            for line in open("/proc/meminfo"):
                k, v = line.split(":", 1); mi[k] = int(v.strip().split()[0])
            hw["ram_total_gb"] = round(mi.get("MemTotal", 0) / 1048576, 1)
            hw["ram_available_gb"] = round(mi.get("MemAvailable", 0) / 1048576, 1)
    except Exception:
        pass
    try:
        if os.name == "nt":
            out = subprocess.run(["powershell", "-NoProfile", "-Command",
                                 "(Get-CimInstance Win32_Processor | Measure-Object -Property NumberOfCores -Sum).Sum"],
                                 capture_output=True, text=True, timeout=10).stdout.strip()
            if out.isdigit(): hw["cpu_cores"] = int(out)
            hw["cpu"] = subprocess.run(["powershell", "-NoProfile", "-Command",
                                 "(Get-CimInstance Win32_Processor)[0].Name"],
                                 capture_output=True, text=True, timeout=10).stdout.strip()
        else:
            for line in open("/proc/cpuinfo"):
                if line.startswith("model name"): hw["cpu"] = line.split(":", 1)[1].strip(); break
    except Exception:
        pass
    # Physical cores: the thread count the CPU experts want (see the threads field).
    if os.name == "nt":
        # Taken above through Win32_Processor; keep the thread fallback.
        if not hw.get("cpu_cores"): hw["cpu_cores"] = hw["cpu_threads"]
    else:
        try:
            cores = set()
            for p in glob.glob("/sys/devices/system/cpu/cpu[0-9]*/topology/core_id"):
                pkg = p.replace("core_id", "physical_package_id")
                cores.add((open(pkg).read().strip() if os.path.exists(pkg) else "0", open(p).read().strip()))
            hw["cpu_cores"] = len(cores) or hw["cpu_threads"]
        except Exception:
            hw["cpu_cores"] = hw["cpu_threads"]
    hw["smt"] = hw["cpu_threads"] > hw["cpu_cores"]
    return hw

def mem_available_gb():
    if os.name == "nt":
        try:
            # GlobalMemoryStatusEx through ctypes: the same number the engine's
            # clamp_to_available reads (ullAvailPhys).
            import ctypes
            class MS(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            ms = MS(); ms.dwLength = ctypes.sizeof(MS)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms)):
                return ms.ullAvailPhys / (1024 ** 3)
        except Exception:
            pass
        return 0.0
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemAvailable:"): return int(line.split()[1]) / 1048576
    except Exception:
        pass
    return 0.0

def drive_info(path):
    """The block device under a path: its model and whether it spins.
    Linux reads /proc/self/mountinfo and /sys/class/block (btrfs hides the
    device behind an anonymous st_dev); Windows asks each disk drive for its
    model and SSD/rotation through the PowerShell storage cmdlets."""
    info = {"mount": "/", "device": None, "disk": None, "model": None, "rotational": None, "fstype": None}
    if os.name == "nt":
        try:
            drive = os.path.splitdrive(os.path.abspath(path))[0] or "C:"
            info["mount"] = drive + "\\"
            fs = subprocess.run(
                ["powershell", "-NoProfile", "-Command",
                 "(Get-Volume -DriveLetter %s).FileSystemType" % drive.rstrip(":")],
                capture_output=True, text=True, timeout=10).stdout.strip()
            if fs: info["fstype"] = fs
            num = subprocess.run(
                ["powershell", "-NoProfile", "-Command",
                 "Get-Partition -DriveLetter %s | Select-Object -ExpandProperty DiskNumber" % drive.rstrip(":")],
                capture_output=True, text=True, timeout=10).stdout.strip()
            if num.isdigit():
                # DiskNumber maps to Get-Disk's Number (DeviceId in Get-PhysicalDisk
                # is a different counter on some systems).
                out = subprocess.run(
                    ["powershell", "-NoProfile", "-Command",
                     "Get-Disk -Number %s | Select-Object Model,BusType | ConvertTo-Json -Compress" % num],
                    capture_output=True, text=True, timeout=10).stdout
                d = json.loads(out) if out.strip() else {}
                if isinstance(d, list): d = d[0] if d else {}
                if d.get("Model"):
                    info["model"] = d["Model"].strip()
                    info["disk"] = info["model"]
                    info["rotational"] = (d.get("BusType") or "") == "ATA"   # SATA spinning disks report ATA; NVMe/USB report otherwise
        except Exception:
            pass
        return info
    try:
        best = ("", None, None)
        for line in open("/proc/self/mountinfo"):
            f = line.split()
            mnt = f[4].replace("\\040", " ")
            if (path + "/").startswith(mnt.rstrip("/") + "/") and len(mnt) > len(best[0]):
                sep = f.index("-"); best = (mnt, f[sep + 2], f[sep + 1])
        info["mount"], dev, info["fstype"] = best
        if dev and dev.startswith("/dev/"):
            info["device"] = dev
            real = os.path.realpath(dev); name = os.path.basename(real)
            # partition -> its disk (/sys/class/block/<part>/.. is the disk)
            p = os.path.realpath(os.path.join("/sys/class/block", name))
            disk = os.path.basename(os.path.dirname(p)) if os.path.exists(os.path.join(p, "partition")) else name
            info["disk"] = disk
            for fn in ("device/model", "device/name"):
                try: info["model"] = open(os.path.join("/sys/class/block", disk, fn)).read().strip(); break
                except Exception: pass
            try: info["rotational"] = open(os.path.join("/sys/class/block", disk, "queue", "rotational")).read().strip() == "1"
            except Exception: pass
    except Exception:
        pass
    return info

def probe_nvme(path, seconds=2.0, nth=8, bs=2 << 20):
    """Random unbuffered reads of the model file at the size and depth the engine's
    expert reads have (2 MiB blocks, 8 in flight): GB/s. The rate the cost model
    prices a miss at. Linux: O_DIRECT+preadv; Windows: FILE_FLAG_NO_BUFFERING+
    overlapped ReadFile through ctypes -- the same slice-shaped, cache-bypassing
    reads the engine itself issues on each platform."""
    sz = os.path.getsize(path)
    if sz < bs * 64: return None
    tot = [0] * nth; err = [None] * nth
    deadline = time.time() + seconds
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        # use_last_error: ctypes must capture the thread's last error at the
        # ReadFile call itself -- any intervening Python/WinAPI call (what the
        # plain windll GetLastError() would see) has already overwritten it.
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        GENERIC_READ = 0x80000000; FILE_SHARE_READ = 1; OPEN_EXISTING = 3
        FILE_FLAG_NO_BUFFERING = 0x20000000; FILE_FLAG_OVERLAPPED = 0x40000000
        k32.CreateFileW.restype = ctypes.c_void_p
        k32.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                                   ctypes.c_void_p, wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p]
        k32.ReadFile.argtypes = [ctypes.c_void_p, ctypes.c_void_p, wintypes.DWORD,
                                 ctypes.POINTER(wintypes.DWORD), ctypes.c_void_p]
        k32.GetOverlappedResult.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                            ctypes.POINTER(wintypes.DWORD), wintypes.BOOL]
        k32.CloseHandle.argtypes = [ctypes.c_void_p]
        # OVERLAPPED on x64: Internal/InternalHigh are ULONG_PTR (8 bytes);
        # c_ulong would be 4 and hand ReadFile a malformed struct (error 87).
        class OV(ctypes.Structure):
            _fields_ = [("Internal", ctypes.c_ulonglong), ("InternalHigh", ctypes.c_ulonglong),
                        ("Offset", wintypes.DWORD), ("OffsetHigh", wintypes.DWORD),
                        ("hEvent", ctypes.c_void_p)]
        assert ctypes.sizeof(OV) == 32, "OVERLAPPED layout must match the x64 ABI"
        def w(i):
            try:
                h = k32.CreateFileW(os.path.abspath(path), GENERIC_READ, FILE_SHARE_READ,
                                    None, OPEN_EXISTING, FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED, None)
                if not h or h == ctypes.c_void_p(-1).value:
                    err[i] = "CreateFileW failed"; return
                ev = k32.CreateEventW(None, True, False, None)
                # FILE_FLAG_NO_BUFFERING needs a sector-aligned buffer address;
                # malloc (what create_string_buffer uses) only promises 16 B.
                # Allocate with slack and round up to the next 4096 boundary.
                raw = ctypes.create_string_buffer(bs + 4096)
                base = (ctypes.addressof(raw) + 4095) & ~4095
                buf = (ctypes.c_char * bs).from_buffer(raw, base - ctypes.addressof(raw))
                rng = random.Random(1000 + i); n = 0
                while time.time() < deadline:
                    off = rng.randrange(0, (sz - bs) // 4096) * 4096
                    ov = OV(0, 0, off & 0xFFFFFFFF, off >> 32, ev)
                    got = wintypes.DWORD(0)
                    ok = k32.ReadFile(h, buf, bs, ctypes.byref(got), ctypes.byref(ov))
                    if not ok and ctypes.get_last_error() == 997:   # ERROR_IO_PENDING
                        if not k32.GetOverlappedResult(h, ctypes.byref(ov), ctypes.byref(got), True):
                            err[i] = "ReadFile failed"; break
                    elif not ok:
                        err[i] = "ReadFile error %d" % ctypes.get_last_error(); break
                    n += got.value
                tot[i] = n
                k32.CloseHandle(h); k32.CloseHandle(ev)
            except Exception as e:
                err[i] = repr(e)
    else:
        def w(i):
            try:
                fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
                buf = mmap.mmap(-1, bs)   # page-aligned, required by O_DIRECT
                rng = random.Random(1000 + i); n = 0
                while time.time() < deadline:
                    n += os.preadv(fd, [buf], rng.randrange(0, (sz - bs) // 4096) * 4096)
                tot[i] = n; os.close(fd)
            except Exception as e:
                err[i] = repr(e)
    ts = [threading.Thread(target=w, args=(i,)) for i in range(nth)]
    t0 = time.time()
    for t in ts: t.start()
    for t in ts: t.join()
    dt = time.time() - t0
    if all(err): return {"error": err[0]}
    gb = sum(tot) / 1e9
    return {"gbs": round(gb / dt, 2), "gb_read": round(gb, 2), "seconds": round(dt, 2), "block_kib": bs // 1024, "queue": nth}

# ---- settings: a cost model fitted to the reference machine's measurements ----
# Per token: GPU graph time + CPU time for the experts served from RAM + NVMe
# time for the misses. What the hardware and the model decide:
#   blocks in VRAM   = (VRAM - dense core - KV/indexer state at ctx - reserve - desktop use) / block size
#   VRAM-served f_v  = 0.95 (1 - exp(-blocks / 1500)), 6 pts lower after a long document's prefill
#   cache hit  f_h   = 0.97 (1 - exp(-(vram+ram blocks) / 1050))   (the speculative block: 96-97% measured)
#   ms/token         = gpu + 480 [ (f_h - f_v) 0.110 + (1 - f_h) io ] + 8,   io = 1.4 (block/2.4 MB)^1.4 ms per missed expert
# Refit 2026-09-06 to the server at 160K context with the speculative block on (tiers 1647 Q4 / 2364 Q3 blocks,
# 12 GB RAM tier): Q4 short chat 10.4 -> 10.4, Q4 155K document 10.0 -> 10.1, Q3 short 15.8 -> 15.6, Q3 document 14.9 -> 14.9.
CTX_STEPS = [8192, 16384, 32768, 65536, 131072, 163840, 262144]
PRESETS = [("chat", "Chat", 32768, "short conversations; the fastest decode"),
           ("coding", "Agentic coding", 131072, "a coding harness with tool calls; room for a repository's worth of context"),
           ("coding_plus", "Agentic coding+", 262144, "the model's full trained context, for the longest sessions")]
PRESET_IDS = [p[0] for p in PRESETS] + ["custom"]
QUANT_GPU_MS = {"Q3_K_XL": 20.0, "Q4_K_XL": 20.0, "IQ1_S": 16.0}   # graph A and the GPU-served experts, per token (fitted, see predict)
QUANT_CPU_MS = {"Q3_K_XL": 0.105, "Q4_K_XL": 0.076, "IQ1_S": 0.080}   # per CPU-served expert, measured on the replays
NVME_MB_PER_MS = 7.0   # the reference drive's fitted rate (GB/s = MB/ms): a block read costs its bytes at this rate
# What the tune's probe (random 2 MiB O_DIRECT reads, 8 in flight, 2.5 s) reads on the reference
# drive, idle (2026-09-11). Another drive scales the fitted rate by its probe against this one;
# the probe's number is not the fitted constant itself (using it as such read 20% pessimistic).
REF_PROBE_GBS = 6.0
LOOKUPS_PER_TOKEN = 480   # 48 layers x 10 routed experts
# Attention state per 1K tokens of context: the KV cache (by its type), the indexer key
# cache and the pooled block keys; plus the constant DeltaNet state. The KV and indexer
# caches can live in pinned RAM instead of VRAM (--state-host), read over PCIe: measured
# on the doc replay at 131K, +0.35 ms/token for the indexer, +1.7 ms/token for the KV
# cache at q4_0 (it scales with the bytes gathered); the VRAM they free is worth ~4% of
# decode per GB at 131K and is what gives the 256K preset an expert tier at all
# (7.0 -> 9.7 tok/s measured, 2026-09-09).
STATE_HOST_OPTIONS = ["none", "idx", "kv,idx"]
KV_TYPES = ["q4_0", "q8_0", "f16"]
KV_MB_PER_K = {"q4_0": 6.9, "q8_0": 13.1, "f16": 26.2}      # KV MB per 1K tokens
def state_parts(ctx, kv):
    k = ctx / 1024 / 1024
    return {"kv": KV_MB_PER_K[kv] * k, "idx": 3.1 * k, "pooled": 0.8 * k, "delta": 0.113}
def state_gb(ctx, kv):
    return sum(state_parts(ctx, kv).values())
def state_host_gb(ctx, kv, state_host):
    p = state_parts(ctx, kv)
    return (p["kv"] if "kv" in state_host else 0.0) + (p["idx"] if "idx" in state_host else 0.0)
def state_vram_gb(ctx, kv, state_host):
    return state_gb(ctx, kv) - state_host_gb(ctx, kv, state_host)
def state_host_ms(kv, state_host):
    return (0.35 if "idx" in state_host else 0.0) + (1.7 * KV_MB_PER_K[kv] / KV_MB_PER_K["q4_0"] if "kv" in state_host else 0.0)

def predict(quant, tier_gb, ram_gb, long_doc, extra_ms=0.0, nvme_mb_per_ms=None):
    # Per token: graph A, the CPU-served experts, and the blocks read this token. The
    # reads are what the tiers' size buys -- the prefetch turns most of them into "hits"
    # but not into fewer bytes -- so the model is residency, the share of lookups served
    # from VRAM or RAM without a read, fitted to the Q4 replays at 131K with tiers of
    # 4.1K / 5.5K / 7.3K blocks (2026-09-10): reads 23 / 18 / 14% of lookups in chat,
    # 33 / 26 / 20% on a long document. The other constants were fitted to ten measured
    # points (three RAM sizes x two replays on Q4; Q3 and IQ1_S at 10.5 GB): rms error
    # 2.7%. A GB of RAM tier is worth about 3% of decode at 131K.
    block = QUANT_BLOCK_MB.get(quant, 2.4)
    nvme = nvme_mb_per_ms or NVME_MB_PER_MS
    vb = max(0.0, tier_gb) * 1024 / block; rb = max(0.0, ram_gb) * 1024 / block
    miss = (0.63 * math.exp(-(vb + rb) / 6400)) if long_doc else (0.46 * math.exp(-(vb + rb) / 5900))
    f_v = max(0.0, 0.85 * (1 - math.exp(-vb / 1500)) - (0.06 if long_doc else 0.0)) if vb > 0 else 0.0
    cpu = max(0.0, 1.0 - f_v - miss)
    ms = QUANT_GPU_MS.get(quant, 20.0) + extra_ms + LOOKUPS_PER_TOKEN * (cpu * QUANT_CPU_MS.get(quant, 0.09) + miss * block / nvme) + 6.0
    return {"tok_s": round(1000 / ms, 1), "vram_served": round(f_v, 3), "hit": round(1.0 - miss, 3), "blocks": int(vb)}

def tune_calibration(model):
    """What the auto-tune measured for this model on this machine, as the cost model's inputs."""
    t = CONFIG["tune"].get(model["name"]) or {}
    gbs = (t.get("nvme") or {}).get("gbs")
    m = t.get("memory") or {}
    return {"threads": t.get("threads"), "nvme_mb_per_ms": NVME_MB_PER_MS * gbs / REF_PROBE_GBS if gbs else None, "nvme_gbs": gbs, "date": t.get("date"),
            # the server's host memory besides the RAM tier, measured through the tune's run (with the
            # state on the host and the draft head it ran with, so another tier is corrected for both)
            "host_other_gb": m.get("host_other_gb"), "host_state_gb": m.get("state_host_gb", 0.0), "host_mtp": m.get("mtp", False), "mem_min_gb": m.get("min_free_gb")}

def recommend(model, hw, preset="coding", vision=None, state_host=None, kv=None, calib=None, ctx_override=None):
    """Settings for one tier on this machine, with every tier summarised so the page can
    show the choice without a table. `calib` is what the auto-tune measured (threads, the
    drive's rate, a RAM correction); without it the reference machine's constants apply."""
    calib = calib if calib is not None else tune_calibration(model)
    q = model["quant"]
    core = QUANT_CORE_GB.get(q, 4.7)
    nvme = calib.get("nvme_mb_per_ms") or NVME_MB_PER_MS
    # Vision: the projector runs on the CPU (weights in RAM, ~15 s per 1400x1000
    # screenshot on 8 cores), so it costs no VRAM at all -- nothing reserved, nothing
    # lent by the expert tier. Default on whenever the file is there: a harness that
    # sends an image gets an answer instead of a 400.
    if vision is None: vision = bool(model.get("mmproj"))
    vision = bool(vision and model.get("mmproj"))
    vram = hw["vram_total_mb"] / 1024.0
    # Measured 2026-09-10 (VRAM audit): the engine grows ~425 MB after init (48 cached
    # decode graphs 286 MB, the pool, per-call inputs); 512 ran the 131K doc replay, so
    # 768 with a desktop / 640 headless keeps a step of margin and gives the tier a step.
    reserve_base = 768 if hw["desktop_gpu"] else 640
    # What decode allocates after the tier grows with the context (the attention
    # graphs are shaped by the block bucket, flat to ~48K tokens then per 256 blocks,
    # plus the CUDA pool and graph instantiations): measured 2026-09-10 at 101K, 805 MB
    # ran out at the first token after a prefill and 1024 held; 768 held at 43K. The
    # engine applies the same rule when no --reserve is given.
    def reserve_for(c): return int(reserve_base + max(0, c // 1024 - 48) * 8)
    # The desktop's own VRAM use, measured now (minus any qwfn engine).
    used = hw["vram_used_mb"] / 1024.0
    eng_used = hw.get("qwfn_vram_mb", 0) / 1024.0
    desktop_use = max(0.2, used - eng_used) if hw["desktop_gpu"] else 0.1
    avail = hw["ram_available_gb"] if not engines_running() else max(hw["ram_available_gb"], hw["ram_total_gb"] - 8.0)
    # The prefill sweeps every expert once per batch and computes per token, so the
    # batch is the prefill's speed: measured on a 43K document at 131K (2026-09-10),
    # 4096 -> 300 tok/s, 8192 -> 479, 16384 -> 722 (compute-bound from there). What a
    # batch costs is VRAM lent by the expert tier while a prompt streams -- 0.59 GB per
    # 4096 tokens on top of the staging -- and the tier must keep more than that or it is
    # disabled, so the batch is picked per context below: the largest whose lend leaves
    # the tier at least a gigabyte. Decode is unchanged either way (harness, same session).
    BATCH_LEND_GB = {2048: 3.9, 4096: 4.4, 8192: 4.98, 16384: 6.16}
    PREFILL_TPS   = {2048: 240, 4096: 300, 8192: 479, 16384: 722}   # Q4 on the reference NVMe; Q3 reads 36% less per sweep
    q3_adj = 0.27 if q == "Q3_K_XL" else 0.0
    def lend_for(b): return BATCH_LEND_GB[b] - q3_adj
    def batch_for(tier_gb):
        for b in (16384, 8192, 4096, 2048):
            if tier_gb - lend_for(b) >= 1.0: return b
        return 2048
    forced_state = state_host if state_host in STATE_HOST_OPTIONS else None   # None: chosen per context below
    forced_kv = kv if kv in KV_TYPES else None                                   # None: the highest precision the GPU has room for
    # What the engine does with the VRAM left after the dense core and the context's state:
    # it takes OVERHEAD_GB for its CUDA context, decode state and graph arenas (measured
    # against the tier it actually built at 160K), and it only builds an expert tier that
    # can hold the prefill's dynamic buffer (staging, work set, arenas: LEND_GB), because
    # that buffer is lent by the tier while a prompt streams. Below that it runs with no
    # VRAM tier at all: every expert comes from RAM or the NVMe, about 25% slower.
    OVERHEAD_GB = 1.2
    # The draft head, on whenever the file is there: 0.1 GB dense + 0.23 GB of rollback
    # snapshots on the device (a tier step), its 2.7 GB of experts in pinned RAM.
    mtp_on = bool(model.get("mtp"))
    HEAD_VRAM_GB = 0.35 if mtp_on else 0.0
    lend_gb = lend_for(2048)   # the smallest prefill buffer: below this there is no tier at all
    def tier_for(c, kv, sh):
        t = vram - core - state_vram_gb(c, kv, sh) - reserve_for(c) / 1024 - desktop_use - OVERHEAD_GB - HEAD_VRAM_GB
        return t if t >= lend_gb + 0.1 else 0.0
    def state_host_for(c, kv):
        # Measured on the doc replay (Q4, 2026-09-09): the indexer move is a small clean
        # gain from 64K up; moving the KV cache too wins from 128K (126K: 10.5 -> 11.1
        # tok/s, 26 fewer CPU-served experts per token), and at 256K it is what keeps
        # an expert tier alive at all (7.0 -> 9.7 tok/s). Below 128K the KV cache is
        # too small to pay for its ~1.7 ms/token of PCIe gathers.
        if forced_state: return forced_state
        if c < 65536: return "none"
        if c >= 131072 or (tier_for(c, kv, "idx") <= 0 and tier_for(c, kv, "kv,idx") > 0): return "kv,idx"
        return "idx"
    # Host memory the server needs besides the arena, measured 2026-09-10 at 131K with
    # the head and the state on the host: MemAvailable at launch minus its minimum through
    # a 43K prefill and 200 tokens, minus the arena, is 8.0 GB -- the head's pinned experts
    # 2.7 of it, the state on the host ~1.3, the rest 4.0. The tier takes what is left
    # above a floor. Measured: a 16 GB arena left 1.3-1.7 GB free at the worst point and
    # was +16% decode in chat, +19% on a long document against 10.5 GB; 19 GB ran the
    # machine out of memory. The old rule (0.6 x available - 3.2, capped at 12) asked for
    # 12 and the engine's own clamp then built 10; every "bigger RAM tier" comparison
    # before this compared 10.4 with 10.7 GB (docs/ENGINEERING.md, 2026-09-10).
    PROCESS_HOST_GB = 8.0 - 2.7 - state_host_gb(131072, "q4_0", "kv,idx")
    HEAD_HOST_GB = 2.7 if mtp_on else 0.0
    # The floor covers what starts after the server: with 3.5 GB, a coding session
    # through OpenCode (its language servers load after the first tool call) went
    # down to 1.9 GB free and the server stalled on its own /stats. 5 GB: the
    # same session kept 3.4 GB at a 13 GB arena when the tooling was lighter.
    # That was the 5 GB floor. It is now the user's headroom (config, 3 GB by default) plus,
    # until the tune has measured what this server really needs, 2 GB for the estimate's
    # own uncertainty. The tune watches MemAvailable through its run: available at launch
    # minus the minimum minus the arena is the server's host memory besides the tier, and
    # the tier takes everything above the headroom from then on (a GB is ~3% of decode).
    headroom = headroom_gb()
    MODEL_MARGIN_GB = 2.0
    measured = calib.get("host_other_gb")
    def other_for(sh_gb):
        if measured:   # corrected for this tier's state on the host and the draft head
            return float(measured) - float(calib.get("host_state_gb") or 0.0) + sh_gb + HEAD_HOST_GB - (2.7 if calib.get("host_mtp") else 0.0)
        return sh_gb + PROCESS_HOST_GB + HEAD_HOST_GB + MODEL_MARGIN_GB
    def ram_for(sh_gb):
        return max(4, int(avail - other_for(sh_gb) - headroom))
    def ram_reason(sh_gb):
        r = ram_for(sh_gb)
        return "%d GB: %.1f GB available now, minus %.1f GB the server needs besides the tier (%s), minus %.1f GB of headroom%s" % (
            r, avail, other_for(sh_gb), "measured by the tune on %s" % (calib.get("date") or "")[:10] if measured else "estimated; Auto-tune measures it", headroom,
            "; the draft head holds 2.7 GB of that in pinned RAM" if mtp_on else "")
    # Threads for the CPU-served experts: the physical core count, or what the tune
    # measured. Measured 2026-09-10 on 8 cores / 16 threads at 131K: 4 -> 15.2 tok/s,
    # 6 -> 15.5, 8 -> 15.6, 12 -> 15.2, 16 -> 11.5 (SMT threads starve the reader and
    # the graph thread).
    threads = int(calib.get("threads") or hw.get("cpu_cores") or hw.get("cpu_threads") or 8)
    def build(c, kv, sh, with_vision):
        tier = tier_for(c, kv, sh)
        shg = state_host_gb(c, kv, sh); ram = ram_for(shg)
        short = predict(q, tier, ram, False, state_host_ms(kv, sh), nvme); longd = predict(q, tier, ram, True, state_host_ms(kv, sh), nvme)
        b = batch_for(tier) if tier > 0 else 2048
        return {"ctx": c, "kv": kv, "tier_gb": round(tier, 2), "blocks": short["blocks"], "state_gb": round(state_gb(c, kv), 2), "vision": with_vision,
                "state_host": sh, "state_host_gb": round(shg, 2), "state_vram_gb": round(state_vram_gb(c, kv, sh), 2), "ram": ram, "ram_reason": ram_reason(shg), "host_other_gb": round(other_for(shg), 2), "threads": threads,
                "batch": b, "lend_gb": round(lend_for(b), 2), "prefill_tps": int(PREFILL_TPS[b] * min(1.0, nvme / NVME_MB_PER_MS)) if q != "IQ1_S" else 0,
                "tok_s_short": short["tok_s"], "tok_s_long_doc": longd["tok_s"], "vram_served": short["vram_served"], "hit": short["hit"], "reserve": reserve_for(c)}
    def option(c, with_vision):
        # KV precision: q8_0 is the default -- the safer choice for long, exact work (tool
        # arguments, code) -- whenever the plan can afford it: the VRAM expert tier survives
        # and its predicted cost is under 10% of decode (its bytes come out of the expert
        # tier, or, with the cache in RAM, out of the RAM tier and the PCIe gathers per
        # token). f16 when it costs under 2%. q4_0 (a quarter of f16, what the replays
        # measured no different) only where q8_0 would not fit.
        base_sh = state_host_for(c, "q4_0"); base = build(c, "q4_0", base_sh, with_vision)
        if forced_kv:
            o = build(c, forced_kv, state_host_for(c, forced_kv), with_vision); o["kv_reason"] = "your choice"; return o
        tried = []
        for kvt, cap in (("f16", 0.02), ("q8_0", 0.10)):
            o = build(c, kvt, state_host_for(c, kvt), with_vision)
            cost = 1.0 - (o["tok_s_short"] / base["tok_s_short"] if base["tok_s_short"] else 1.0)
            where = ("%.1f ms/token more of PCIe gathers and %d GB less RAM tier" % (state_host_ms(kvt, o["state_host"]) - state_host_ms("q4_0", base_sh), base["ram"] - o["ram"])) if "kv" in o["state_host"] \
                    else ("%.1f GB of expert tier" % max(0.0, base["tier_gb"] - o["tier_gb"]))
            fits = (o["tier_gb"] > 0 or base["tier_gb"] == 0) and o["ram"] >= 4
            tried.append("%s would cost %s (%.0f%% of decode%s)" % (kvt, where, cost * 100, "" if fits else ", and the expert tier would not fit"))
            if fits and cost <= cap:
                o["kv_reason"] = "%s: the plan affords it, %s (%.0f%% of decode, under the %.0f%% cap)%s" % (kvt, where, cost * 100, cap * 100, "; " + "; ".join(tried[:-1]) if tried[:-1] else "")
                return o
        base["kv_reason"] = "q4_0: " + "; ".join(tried)
        return base
    tiers = []
    tune = CONFIG["tune"].get(model["name"]) or {}
    for pid, label, ctx, blurb in PRESETS:
        o = option(ctx, vision); note = ""
        if o["state_host"] != "none": note = "attention state in RAM (%s): %.1f GB of VRAM for the expert tier" % (o["state_host"], o["state_host_gb"])
        if o["tier_gb"] == 0:
            fallback = [option(c, vision) for c in CTX_STEPS if c < ctx]
            fallback = [f for f in fallback if f["tier_gb"] > 0]
            if fallback: o = fallback[-1]; note = "no room for an expert tier at %dK on this GPU: %dK instead" % (ctx // 1024, o["ctx"] // 1024)
            else: note = "no room for a VRAM expert tier on this GPU: experts come from RAM and the NVMe"
        t = {"id": pid, "label": label, "blurb": blurb, "ctx": ctx, "fits": o["ctx"] == ctx, "note": note, "ctx_actual": o["ctx"], "kv": o["kv"], "vision": o["vision"],
             "tier_gb": o["tier_gb"], "blocks": o["blocks"], "tok_s_short": o["tok_s_short"], "tok_s_long_doc": o["tok_s_long_doc"], "vram_served": o["vram_served"],
             "batch": o["batch"], "prefill_tps": o["prefill_tps"], "ram": o["ram"], "threads": o["threads"], "state_host": o["state_host"], "saved": True}
        if tune.get("preset") == pid and tune.get("verify"):
            t["measured"] = {"chat": tune["verify"].get("chat_tps"), "doc": tune["verify"].get("doc_tps"), "prefill": tune["verify"].get("doc_prefill_tps"), "date": tune.get("date")}
        tiers.append(t)
    custom = CONFIG["custom"].get(model["name"])
    if custom:
        try:
            ck = custom.get("kv") if custom.get("kv") in KV_TYPES else "q4_0"; csh = custom.get("state_host") if custom.get("state_host") in STATE_HOST_OPTIONS else "none"
            o = build(int(custom.get("ctx") or 131072), ck, csh, bool(custom.get("vision")) and bool(model.get("mmproj")))
            o["ram"] = int(custom.get("ram") or o["ram"]); o["threads"] = int(custom.get("threads") or o["threads"]); o["batch"] = int(custom.get("batch") or o["batch"])
            short = predict(q, o["tier_gb"], o["ram"], False, state_host_ms(ck, csh), nvme); longd = predict(q, o["tier_gb"], o["ram"], True, state_host_ms(ck, csh), nvme)
            tiers.append({"id": "custom", "label": "Custom", "blurb": "your saved settings" + (" (" + custom["saved"] + ")" if custom.get("saved") else ""), "ctx": o["ctx"], "fits": True, "note": "",
                          "ctx_actual": o["ctx"], "kv": ck, "vision": o["vision"], "tier_gb": o["tier_gb"], "blocks": o["blocks"], "tok_s_short": short["tok_s"], "tok_s_long_doc": longd["tok_s"],
                          "vram_served": short["vram_served"], "batch": o["batch"], "prefill_tps": o["prefill_tps"], "ram": o["ram"], "threads": o["threads"], "state_host": csh, "saved": True})
        except Exception:
            custom = None
    if not custom:
        tiers.append({"id": "custom", "label": "Custom", "blurb": "your own settings: change anything under Advanced and save it here", "saved": False})
    if preset not in PRESET_IDS or (preset == "custom" and not custom): preset = "coding"
    p = next(t for t in tiers if t["id"] == preset)
    vision = p["vision"]; mm_gb = model.get("mmproj_gb", 0.0) if vision else 0.0
    chosen = option(p["ctx_actual"], vision) if preset != "custom" else None
    if preset == "custom":
        ck = custom.get("kv") if custom.get("kv") in KV_TYPES else "q4_0"; csh = custom.get("state_host") if custom.get("state_host") in STATE_HOST_OPTIONS else "none"
        chosen = build(p["ctx"], ck, csh, vision); chosen["kv_reason"] = "your saved tier"
        chosen.update({"ram": p["ram"], "threads": p["threads"], "batch": p["batch"]})
    if ctx_override:   # the advanced form's own context: size everything for it
        try: chosen = option(max(1024, int(ctx_override)), vision)
        except Exception: pass
    out = {
        "ctx": chosen["ctx"], "kv": chosen["kv"], "ram": chosen["ram"], "threads": chosen["threads"], "batch": chosen["batch"], "reserve": chosen["reserve"], "think": "xhigh", "think_budget": 6000,
        "skip_miss": False, "spec_block": True, "port": STATE["port"], "preset": preset,
        "vision": vision, "mmproj": model.get("mmproj"), "mmproj_gb": model.get("mmproj_gb", 0.0),
        "state_host": chosen["state_host"],
        # The draft head, on when the file is there: measured through OpenCode on a coding
        # task (2026-09-10) +12% decode on Q4 and +19% on Q3 at 95-96% acceptance, +7-8% on
        # the replays; it costs a tier step and 2.7 GB of pinned RAM, both priced in.
        "mtp": mtp_on, "mtp_file": model.get("mtp"), "mtp_gb": model.get("mtp_gb", 0.0),
        "kv_reason": chosen.get("kv_reason", ""), "ram_reason": chosen.get("ram_reason", ""), "host_other_gb": chosen.get("host_other_gb"), "headroom_gb": headroom,
        "state_reason": {"none": "attention state in VRAM: the context is short enough", "idx": "indexer cache in RAM: a small clean gain from 64K up",
                         "kv,idx": "indexer + KV cache in RAM: from 128K their VRAM is worth more as expert tier (measured +5% at 126K, and the difference between a tier and none at 256K)"}.get(chosen["state_host"], ""),
        "estimates": {"vram_tier_gb": chosen["tier_gb"], "vram_tier_blocks": chosen["blocks"], "state_gb": chosen["state_gb"], "dense_core_gb": core,
                      "state_host_gb": chosen["state_host_gb"], "state_vram_gb": chosen["state_vram_gb"],
                      "mmproj_gb": round(mm_gb, 2),
                      "desktop_use_gb": round(desktop_use, 2), "decode_tps_short": chosen["tok_s_short"], "decode_tps_long_doc": chosen["tok_s_long_doc"],
                      "vram_served": chosen["vram_served"], "prefill_tps_long": chosen["prefill_tps"], "lend_gb": chosen["lend_gb"]},
        "tiers": tiers, "calibration": {**calib, "cores": hw.get("cpu_cores"), "cpu_threads": hw.get("cpu_threads")},
        "custom_saved": bool(custom), "tune": tune or None,
    }
    if preset == "custom":
        for k in ("think", "think_budget", "skip_miss", "spec_block", "mtp", "port", "reserve", "vision"):
            if k in custom: out[k] = custom[k]
        out["mtp"] = bool(out["mtp"]) and bool(model.get("mtp")); out["vision"] = bool(out["vision"]) and bool(model.get("mmproj"))
    return out

# ---- server control ----------------------------------------------------------
_ENGINES = {"t": 0.0, "v": []}
def engines_running(max_age=0.0):
    if max_age and time.time() - _ENGINES["t"] < max_age: return _ENGINES["v"]
    procs = []
    try:
        if os.name == "nt":
            # Windows: qwfn-server.exe / qwfn-gen.exe via tasklist (no pgrep).
            # Emitted as "<pid> <name>" -- the same shape pgrep -l gives on Linux,
            # which is what stop_server parses (l.split()[0]).
            for name in ("qwfn-server.exe", "qwfn-gen.exe"):
                out = subprocess.run(["tasklist", "/FI", "IMAGENAME eq " + name, "/FO", "CSV", "/NH"],
                                     capture_output=True, text=True, timeout=3).stdout
                for l in out.splitlines():
                    if name in l:
                        f = [x for x in l.replace('"', "").split(",") if x]
                        if len(f) >= 2:
                            procs.append("%s %s" % (f[1], f[0]))   # pid first, like pgrep -l
        else:
            out = subprocess.run(["pgrep", "-a", "-x", "qwfn-server"], capture_output=True, text=True, timeout=3).stdout
            procs = [l for l in out.splitlines() if l.strip()]
            out2 = subprocess.run(["pgrep", "-l", "qwfn-gen"], capture_output=True, text=True, timeout=3).stdout
            procs += [l for l in out2.splitlines() if l.strip()]
    except Exception:
        procs = []
    _ENGINES.update(t=time.time(), v=procs)
    return procs

def port_open(port):
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.5): return True
    except OSError:
        return False

def fetch_json(url, timeout=2.0, data=None):
    try:
        req = urllib.request.Request(url, data=json.dumps(data).encode() if data is not None else None, headers={"Content-Type": "application/json"} if data is not None else {})
        with urllib.request.urlopen(req, timeout=timeout) as r: return json.load(r)
    except Exception:
        return None

def server_argv(model, s):
    argv = [SERVER_BIN, model["path"], "--ram", str(int(s["ram"])), "--ram-frac", "0.85", "--threads", str(int(s.get("threads") or 8)),
            "--ctx", str(int(s["ctx"])), "--batch", str(int(s["batch"])),
            "--kv", s["kv"], "--reserve", str(int(s["reserve"])), "--think", s["think"], "--think-budget", str(int(s["think_budget"])),
            "--port", str(int(s["port"]))]
    if s.get("skip_miss"): argv.append("--skip-miss")
    if s.get("spec_block", True): argv.append("--spec-block")
    if s.get("mtp"): argv += ["--mtp", model["mtp"]]
    if s.get("cold_path"): argv += ["--cold", s["cold_path"]]
    if s.get("vision"): argv += ["--mmproj", model["mmproj"]]
    if s.get("state_host") in ("idx", "kv", "kv,idx"): argv += ["--state-host", s["state_host"]]
    if s.get("prefill_decode_max"): argv += ["--prefill-decode-max", str(int(s["prefill_decode_max"]))]
    if s.get("mtp") and s.get("mtp_drafts"): argv += ["--mtp-drafts", str(int(s["mtp_drafts"]))]
    return argv

def start_server(model, s):
    with LOCK:
        if STATE["proc"] and STATE["proc"].poll() is None: return {"error": "a server started by this console is already running"}
        running = engines_running()
        if running: return {"error": "an engine is already running on this machine (one at a time): " + "; ".join(running)[:300]}
        if not os.path.exists(SERVER_BIN): return {"error": f"{SERVER_BIN} not found: install the release bundle, or build first (cmake --build build)"}
        s = dict(s)
        # --ram-frac 0.85 keeps the engine's own MemAvailable clamp (a backstop for the
        # command line, 0.75 by default) above the size computed here, which is the
        # measured one; with the default clamp the tier came out at 10 GB whatever was asked.
        if s.get("mtp") and s.get("skip_miss"):
            s["mtp"] = False   # the engine would refuse every request: a verified pair and skip-miss do not combine
        if s.get("mtp") and not model.get("mtp"):
            return {"error": "no MTP/mtp-*.gguf in this model's repository: download it into the snapshot directory, or turn the draft head off"}
        if s.get("vision") and not model.get("mmproj"):
            return {"error": "no mmproj-*.gguf next to this model: download mmproj-F16.gguf into its snapshot directory, or turn Vision off"}
        # The memory the plan assumed may be gone (a browser, a build): say so before the engine
        # clamps the tier. Read twice: right after a stop the arena's pages are still coming back.
        other = 10.0 if s.get("host_other_gb") is None else float(s["host_other_gb"])
        free = mem_available_gb()
        if free and s.get("ram") and free - float(s["ram"]) - other < 1.0:
            time.sleep(2.0); free = mem_available_gb()
        if free and s.get("ram") and free - float(s["ram"]) - other < 1.0:
            if s.get("keep_ram"):   # the caller insists (a measurement): warn, do not shrink
                s["ram_note"] = "RAM tier kept at %d GB as asked: %.1f GB is available and the server needs ~%.1f GB besides the tier, so the machine will be short of memory at the worst point" % (int(s["ram"]), free, other)
            else:
                s["ram"] = max(4, int(free - other - 1.0)); s["ram_note"] = "RAM tier reduced to %d GB: %.1f GB is available right now and the server needs %.1f GB besides the tier" % (s["ram"], free, other)
        argv = server_argv(model, s)
        log = open(STATE["log"], "w")
        log.write("$ " + " ".join(argv) + "\n")
        if s.get("ram_note"): log.write("[console] " + s["ram_note"] + "\n")
        if s.get("skip_miss") and not s.get("mtp") and model.get("mtp"):
            log.write("[console] draft head left off: a verified pair and skip-miss do not combine (skip-miss is one token at a time)\n")
        log.flush()
        # The bundle keeps ggml and the CUDA runtime next to the engine; the loader
        # needs the directory on its search path (LD_LIBRARY_PATH on Linux, PATH on
        # Windows, where the CUDA backend LoadLibrary's its DLLs). The bundle layout
        # has the libs next to the binary; a source checkout instead uses the
        # engine's own backend dir (~/.unsloth/llama.cpp/build/bin, the same place
        # qwfn::llama_backend_dir() looks) or an explicit QWFN_BACKENDS.
        env = dict(os.environ); bindir = os.path.dirname(SERVER_BIN)
        lib_name = "libggml-base.so.0" if os.name != "nt" else "ggml-base.dll"
        cands = [bindir]
        if os.name == "nt":
            cands.append(os.path.join(os.environ.get("USERPROFILE") or os.environ.get("HOME") or "", ".unsloth", "llama.cpp", "build", "bin"))
            if os.environ.get("QWFN_BACKENDS"): cands.append(os.environ["QWFN_BACKENDS"])
        libdir = next((d for d in cands if d and os.path.exists(os.path.join(d, lib_name))), None)
        if libdir:
            if os.name == "nt":
                env["PATH"] = libdir + os.pathsep + env.get("PATH", "")
            else:
                env["LD_LIBRARY_PATH"] = libdir + (":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
        try:
            proc = subprocess.Popen(argv, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=env)
        except Exception as e:
            return {"error": f"cannot start: {e}"}
        STATE.update(proc=proc, model=model, settings=s, started=time.time(), port=int(s["port"]))
        CONFIG["last"] = {"model": model["path"], "name": model["name"], "preset": s.get("preset") or "coding"}; save_config()
        return {"ok": True, "pid": proc.pid, "argv": argv, "note": s.get("ram_note")}

def stop_server():
    with LOCK:
        p = STATE["proc"]
        if not p or p.poll() is not None:
            # A server started outside this console (serve.sh, or a console that has since
            # exited) still answers on the port: stop it too, it is the one engine there is.
            pids = [l.split()[0] for l in engines_running() if "qwfn-server" in l]
            for pid in pids:
                try: os.kill(int(pid), signal.SIGTERM)
                except Exception: pass
            return {"ok": True, "note": "stopped the server running outside the console" if pids else "no server running"}
        p.send_signal(signal.SIGTERM)
        for _ in range(50):
            if p.poll() is not None: break
            time.sleep(0.1)
        if p.poll() is None: p.kill()
        STATE["proc"] = None
        return {"ok": True}

def log_tail(n=60):
    try:
        with open(STATE["log"], "rb") as f:
            f.seek(0, 2); size = f.tell(); f.seek(max(0, size - 64000)); data = f.read().decode("utf-8", "replace")
        lines = [l for l in data.splitlines() if "CUDA graph warmup" not in l and "cudaMalloc failed" not in l]
        return lines[-n:]
    except Exception:
        return []

def log_tiers():
    """What the engine actually built, from its log: the VRAM and RAM expert tiers."""
    out = {}
    for l in log_tail(400):
        if "expert VRAM tier:" in l:
            try: out["vram_tier_gb"] = float(l.split("expert VRAM tier:")[1].split("GB")[0]); out["vram_tier_blocks"] = int(l.split("GB,")[1].split("blocks")[0])
            except Exception: pass
        if "expert RAM tier:" in l:
            try: out["ram_tier_gb"] = float(l.split("expert RAM tier:")[1].split("GB")[0])
            except Exception: pass
    return out

def live_state():
    """The cheap poll: process state and the server's own counters, nothing else."""
    p = STATE["proc"]; alive = bool(p) and p.poll() is None
    port = STATE["port"]
    st = {"console_started": alive, "uptime_s": round(time.time() - STATE["started"]) if alive else 0, "port": port, "endpoint": f"http://127.0.0.1:{port}/v1",
          "model": STATE["model"]["name"] if (alive and STATE["model"]) else STATE.get("ext_model"), "state": "stopped", "external": False,
          "tune": {"running": TUNE["running"], "step": TUNE["step"], "progress": TUNE["progress"]}, "selftest_running": SELFTEST["running"]}
    if not alive and p is not None and p.poll() is not None:
        st["exit_code"] = p.returncode; st["state"] = "exited"
    if port_open(port):
        stats = fetch_json(f"http://127.0.0.1:{port}/stats", 1.5)
        if stats:
            st["stats"] = stats; st["state"] = "busy" if stats.get("busy") else "ready"
            if not alive: st["external"] = True
        elif alive: st["state"] = "loading"
    elif alive:
        st["state"] = "loading"
    return st

def status():
    st = live_state()
    p = STATE["proc"]; alive = st["console_started"]; port = st["port"]
    st.update(pid=p.pid if alive else None, settings=STATE["settings"] if alive else None, model_path=STATE["model"]["path"] if (alive and STATE["model"]) else None)
    if st["state"] in ("ready", "busy"):
        props = fetch_json(f"http://127.0.0.1:{port}/props", 2.0)
        st["props"] = props
        if props:
            st["model_id"] = props.get("model")
            if not alive:
                mf = props.get("model_file") or ""
                STATE["ext_model"] = os.path.basename(os.path.dirname(mf)) if mf else (props.get("model") or None)
                st["model"] = STATE["ext_model"]; st["model_path"] = mf or None
    st["log"] = log_tail(40)
    st["engines"] = engines_running(max_age=4.0)
    st["tiers_built"] = log_tiers() if st["state"] in ("ready", "busy") and alive else {}
    return st

def set_threads(n):
    port = STATE["port"]
    r = fetch_json(f"http://127.0.0.1:{port}/props", 5.0, {"threads": int(n)})
    if r and r.get("n_threads") == int(n):
        if STATE["settings"]: STATE["settings"]["threads"] = int(n)
        return {"ok": True, "threads": int(n)}
    return {"error": "the server did not take the thread count (is a request running?)"}

# ---- measurements through the server (the self-test and the tune share them) ----
WORDS = ["amber", "basalt", "cedar", "delta", "ember", "falcon", "garnet", "harbor", "indigo", "juniper", "kestrel", "lagoon",
         "marble", "nectar", "onyx", "pebble", "quartz", "raven", "saffron", "tundra", "umber", "velvet", "willow", "zephyr"]

def selftest_document(n_tokens):
    """A long document made of the engine's own sources (real prose and code, the kind of
    text a coding harness sends), with a passphrase planted at about 40% depth and a
    random header so the server's prefix cache cannot skip the prefill on a repeat run."""
    files = [os.path.join(ROOT, "README.md")] + sorted(glob.glob(os.path.join(ROOT, "src", "*.cpp"))) + sorted(glob.glob(os.path.join(ROOT, "tools", "*.cpp")))
    parts = []
    for f in files:
        try: parts.append("\n\n===== %s =====\n\n" % os.path.relpath(f, ROOT) + open(f, encoding="utf-8", errors="replace").read())
        except Exception: pass
    text = "".join(parts) or ("The quick brown fox jumps over the lazy dog. " * 2000)
    want = int(n_tokens * 3.3)   # this tokenizer takes ~3.3 characters per token on the mix of C++ and prose
    while len(text) < want: text += text
    text = text[:want]
    rng = random.Random()
    phrase = "%s-%s-%d" % (rng.choice(WORDS), rng.choice(WORDS), rng.randint(100, 999))
    at = text.find("\n", int(len(text) * 0.4)) + 1
    needle = "\n\nNOTE FOR THE READER: the passphrase for this self-test is \"%s\". Remember it.\n\n" % phrase
    header = "Self-test document %d\n\n" % rng.randint(10**6, 10**7)
    return header + text[:at] + needle + text[at:], phrase

def chat_request(port, content, max_tokens, timeout=3600):
    return chat_messages(port, [{"role": "user", "content": content}], max_tokens, timeout)

def chat_messages(port, messages, max_tokens, timeout=3600):
    body = json.dumps({"model": "qwfn", "messages": messages, "max_tokens": max_tokens,
                       "reasoning_effort": "off", "temperature": 0.0, "stream": False}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r: return json.load(r)

def wait_ready(log, deadline_s=1200):
    deadline = time.time() + deadline_s
    while time.time() < deadline:
        st = live_state()
        if st["state"] in ("ready", "busy"): return st
        if st["state"] in ("exited", "stopped"):
            raise RuntimeError("the server exited while loading (exit code %s): see the Log tab" % st.get("exit_code"))
        time.sleep(2)
    raise RuntimeError("the server did not become ready within %d minutes" % (deadline_s // 60))

def measure_chat(port):
    # About 250 words: a three-sentence answer (76 tokens) sat inside the ~25-token cache
    # warm-up transient and read 16% under what the same server did on a longer answer.
    r = chat_request(port, "In about 250 words, explain why mixture-of-experts models are cheaper to run than dense models with the same parameter count, and what the catch is.", 400)
    t = r.get("timings") or {}
    return {"tok_s": round(t.get("predicted_per_second", 0), 1), "n": t.get("predicted_n", 0), "prefill_tok_s": round(t.get("prompt_per_second", 0), 0),
            "answer": (r["choices"][0]["message"].get("content") or "").strip()[:400]}

def measure_doc(port, ctx, log):
    n_doc = min(32768, ctx // 2)
    doc, phrase = selftest_document(n_doc)
    log("long document: prefilling about %dK tokens, then a grounded answer" % (n_doc // 1024))
    prompt = doc + "\n\nTwo things: (1) What is the passphrase that the note in this document asks the reader to remember? Quote it exactly. (2) In two sentences, what is this document about?"
    r = chat_request(port, prompt, 160)
    t = r.get("timings") or {}
    answer = (r["choices"][0]["message"].get("content") or "").strip()
    return {"tokens": t.get("prompt_n", 0), "prefill_tok_s": round(t.get("prompt_per_second", 0), 0), "prefill_s": round(t.get("prompt_ms", 0) / 1000, 0),
            "tok_s": round(t.get("predicted_per_second", 0), 1), "n": t.get("predicted_n", 0), "found": phrase.lower() in answer.lower(),
            "phrase": phrase, "answer": answer[:400], "conversation": [{"role": "user", "content": prompt}, {"role": "assistant", "content": answer}]}

# ---- self-test: the chosen model on this hardware, through the server ---------
# Starts the server if none is running (with the settings the page shows), then measures
# what a user would see: a short chat (decode tok/s), and a long document with a passphrase
# planted in it (prefill tok/s, decode tok/s on that context, and whether the answer found
# the passphrase). The server is left running afterwards.
SELFTEST = {"running": False, "step": "", "log": [], "result": None, "error": None, "t0": 0.0}
SELFTEST_FILE = os.path.join(LOG_DIR, "selftest.json")
try: SELFTEST["result"] = json.load(open(SELFTEST_FILE))
except Exception: pass

def run_selftest(model, settings):
    T = SELFTEST
    def log(s): T["step"] = s; T["log"].append("%4.0f s  %s" % (time.time() - T["t0"], s))
    try:
        st = status(); started_here = False
        if st["state"] in ("ready", "busy", "loading"):
            if st.get("model") and st["model"] != model["name"]:
                raise RuntimeError("a different model is being served (%s): stop it first, or test that one" % st["model"])
            log("using the running server" if st["state"] != "loading" else "waiting for the server to finish loading")
        else:
            log("starting the server with the settings shown")
            r = start_server(model, settings)
            if r.get("error"): raise RuntimeError(r["error"])
            started_here = True
        st = wait_ready(log)
        port = st["port"]
        run = STATE["settings"] if STATE["settings"] and st["console_started"] else {}
        props = fetch_json(f"http://127.0.0.1:{port}/props", 2.0) or {}
        ctx = int(run.get("ctx") or props.get("n_ctx") or settings.get("ctx") or 32768)
        log("warm-up")
        chat_request(port, "Say hello in one short sentence.", 24)
        log("short chat: a three-sentence answer")
        chat = measure_chat(port)
        docres = measure_doc(port, ctx, log)
        stats = fetch_json(f"http://127.0.0.1:{port}/stats", 3.0) or {}
        c = stats.get("expert_cache") or {}
        hw = hardware()
        est = recommend(model, hw, run.get("preset") or settings.get("preset") or "coding", run.get("vision"))
        T["result"] = {"model": model["name"], "ctx": ctx, "preset": run.get("preset") or settings.get("preset"), "flags": run,
                       "date": time.strftime("%Y-%m-%d %H:%M"), "chat": chat, "doc": docres,
                       "cache": {"hit": round(c.get("hit_rate", 0), 3), "vram_served": round(c.get("vram_served", 0), 3)},
                       "vram_used_gb": round(hw["vram_used_mb"] / 1024, 1), "vram_total_gb": round(hw["vram_total_mb"] / 1024, 1),
                       "ram_available_gb": hw["ram_available_gb"], "gpu": hw.get("gpu"),
                       "predicted": {"chat": est["estimates"]["decode_tps_short"], "doc": est["estimates"]["decode_tps_long_doc"], "vram_served": est["estimates"]["vram_served"]},
                       "started_server": started_here, "elapsed_s": round(time.time() - T["t0"])}
        try: json.dump(T["result"], open(SELFTEST_FILE, "w"))
        except Exception: pass
        log("done in %d s" % (time.time() - T["t0"]))
    except Exception as e:
        T["error"] = str(e); log("failed: %s" % e)
    finally:
        T["running"] = False

def start_selftest(model, settings):
    with LOCK:
        if SELFTEST["running"] or TUNE["running"]: return {"error": "a self-test or a tune is already running"}
        SELFTEST.update(running=True, step="starting", log=[], error=None, t0=time.time())
    threading.Thread(target=run_selftest, args=(model, settings), daemon=True).start()
    return {"ok": True}

# ---- auto-tune: measure this machine and pick every flag from what it finds --------
# 1. The drive: random 2 MiB O_DIRECT reads of the model file at the engine's depth; the
#    rate re-prices every expert miss in the cost model (and warns about a spinning disk).
# 2. The plan: the tier's settings recomputed with that rate -- the KV precision the GPU
#    has room for, the attention-state placement, the batch the tier can lend, the RAM tier
#    the memory allows -- and the server started with them (restarted if one was running).
# 3. Threads: swept live on the running server (POST /props) with a fixed 64-token decode
#    per point, two passes, the smallest count within 2% of the best wins (it leaves cores
#    for the harness). 4. Memory: MemAvailable watched through it all; a tier that pushed
#    it under the floor is shrunk for next time. 5. Verification: a short chat and a long
#    document with a planted passphrase (prefill and decode tok/s, grounded answer), the
#    numbers the tier cards then show as measured. Everything is saved per model.
TUNE = {"running": False, "step": "", "log": [], "result": None, "error": None, "t0": 0.0, "progress": 0.0, "cancel": False}
# A fixed prompt for the sweep, deterministic at temperature 0 so every point decodes the same
# tokens, but not so predictable that the draft head accepts every draft (a counting prompt
# measured 30 tok/s where chat does 14: two tokens a step, every step).
SWEEP_PROMPT = "In about 150 words, explain how a mixture-of-experts transformer decides which experts each token goes to, and why that makes inference cheaper than a dense model of the same size."

def pick_threads(table, cores):
    """The physical core count, unless another count beat it in EVERY pass by over 3%
    (GPU decode moves ~5% run to run; one lucky pass must not move the choice)."""
    by_n = {t["threads"]: t for t in table}
    if cores not in by_n: return max(by_n, key=lambda n: by_n[n]["tok_s"])
    base = by_n[cores]["tok_s"]
    better = [n for n, t in by_n.items() if n != cores and t["runs"] and min(t["runs"]) > base * 1.03]
    return max(better, key=lambda n: by_n[n]["tok_s"]) if better else cores

def thread_candidates(cores, threads):
    c = set()
    c.add(max(2, cores // 2)); c.add(max(2, cores - 2)); c.add(cores)
    if threads > cores:
        c.add(min(threads, cores + 2)); c.add((cores + threads) // 2); c.add(threads)
    return sorted(x for x in c if 1 <= x <= max(threads, cores))

def run_tune(model, preset, settings_in):
    T = TUNE
    def log(s, prog=None):
        T["step"] = s; T["log"].append("%4.0f s  %s" % (time.time() - T["t0"], s))
        if prog is not None: T["progress"] = prog
    def check_cancel():
        if T["cancel"]: raise RuntimeError("cancelled")
    result = {"model": model["name"], "preset": preset, "date": time.strftime("%Y-%m-%d %H:%M"), "notes": []}
    mem_watch = {"min": 1e9, "stop": False}
    def watch():
        while not mem_watch["stop"]:
            mem_watch["min"] = min(mem_watch["min"], mem_available_gb()); time.sleep(0.5)
    watcher = None
    try:
        # 0. a running server goes first: the drive is probed idle and the memory the plan
        #    takes is measured, not estimated behind a running engine (that estimate,
        #    total minus 8 GB, cost 2 GB of RAM tier on the reference machine)
        st = status()
        if st["state"] in ("ready", "busy", "loading") or engines_running():
            log("stopping the running server%s: the tune restarts it with the planned flags" % ("" if st["console_started"] else " (started outside this console)"), 0.01)
            stop_server()
            for _ in range(40):
                if not engines_running() and not port_open(STATE["port"]): break
                time.sleep(0.5)
            time.sleep(1.5)   # the arena's pages come back to MemAvailable
        check_cancel()
        hw = hardware()
        # 1. the drive
        log("checking the drive under the model", 0.02)
        dinfo = drive_info(model["path"]); result["drive"] = dinfo
        pf = model.get("probe_file") or model["path"]
        if dinfo.get("rotational"):
            result["notes"].append("the model is on a spinning disk (%s): expect a few tokens per second at best; move it to an NVMe" % (dinfo.get("model") or dinfo.get("disk")))
            probe = probe_nvme(pf, seconds=1.0, nth=4)
        else:
            probe = probe_nvme(pf, seconds=2.5)
        check_cancel()
        if probe and not probe.get("error"):
            result["nvme"] = probe
            log("drive: %.1f GB/s random reads (%s)" % (probe["gbs"], dinfo.get("model") or dinfo.get("disk") or "unknown device"), 0.06)
            if probe["gbs"] < 2.5: result["notes"].append("the drive reads %.1f GB/s at the engine's pattern; a Gen4 NVMe does ~7. Expect decode to be limited by expert reads" % probe["gbs"])
        else:
            log("drive probe skipped (%s)" % ((probe or {}).get("error") or "the model file is too small to sample"), 0.06)
        gbs = (result.get("nvme") or {}).get("gbs")
        prev = tune_calibration(model)   # the previous tune's host-memory measurement sizes this run's first tier
        calib = {**prev, "threads": None, "nvme_mb_per_ms": NVME_MB_PER_MS * gbs / REF_PROBE_GBS if gbs else None, "nvme_gbs": gbs}
        # 2. the plan
        result["avail_gb"] = hw["ram_available_gb"]
        log("planning the %s tier for this machine (%.1f GB of RAM available, %.1f GB headroom)" % (preset, hw["ram_available_gb"], headroom_gb()), 0.08)
        plan = recommend(model, hw, preset, None, None, None, calib)
        if preset == "custom" and settings_in: plan = {**plan, **{k: v for k, v in settings_in.items() if k in plan}}
        result["plan"] = {k: plan[k] for k in ("ctx", "kv", "ram", "threads", "batch", "reserve", "state_host", "vision", "mtp", "spec_block", "skip_miss", "think", "think_budget", "port")}
        result["kv_reason"] = plan.get("kv_reason", ""); result["state_reason"] = plan.get("state_reason", "")
        log("KV cache %s (%s)" % (plan["kv"], plan["kv_reason"]))
        log("attention state: %s" % plan["state_reason"])
        log("RAM tier %s" % plan.get("ram_reason", plan["ram"]))
        log("VRAM expert tier ~%.1f GB, batch %d, reserve %d MB" % (plan["estimates"]["vram_tier_gb"], plan["batch"], plan["reserve"]))
        # 3. the server, with the plan
        check_cancel()
        log("starting the server", 0.12)
        r = start_server(model, plan)
        if r.get("error"): raise RuntimeError(r["error"])
        if r.get("note"): result["notes"].append(r["note"])
        st = wait_ready(log); port = st["port"]
        built = log_tiers(); result["built"] = built
        if built.get("vram_tier_gb") is not None:
            log("engine built a %.1f GB VRAM expert tier (planned %.1f) and a %.1f GB RAM tier (planned %d)" % (built["vram_tier_gb"], plan["estimates"]["vram_tier_gb"], built.get("ram_tier_gb", 0), plan["ram"]), 0.3)
            if plan["estimates"]["vram_tier_gb"] > 0 and built["vram_tier_gb"] < plan["estimates"]["vram_tier_gb"] - 1.0:
                result["notes"].append("the VRAM tier came out %.1f GB smaller than planned: something else holds VRAM (check the desktop's GPU use)" % (plan["estimates"]["vram_tier_gb"] - built["vram_tier_gb"]))
        watcher = threading.Thread(target=watch, daemon=True); watcher.start()
        # warm-up: the tiers fill on the first tokens
        log("warm-up", 0.32)
        chat_request(port, SWEEP_PROMPT, 64); chat_request(port, SWEEP_PROMPT, 64)
        check_cancel()
        # 4. verification: what the user will see (a short chat; then a long document, whose
        #    context the thread sweep then runs on)
        log("verifying: a short chat", 0.34)
        chat = measure_chat(port)
        check_cancel()
        doc = measure_doc(port, int(plan["ctx"]), lambda s: log(s, 0.38))
        check_cancel()
        # 5. the thread sweep, live on the server, on the document's context: at short context
        #    the VRAM tier serves nearly every expert and the CPU threads cannot show (16-17
        #    tok/s at every count, measured); on a long context the RAM-served share is real.
        #    Each point extends the conversation, so the prefix is reused and a point costs
        #    its 64 tokens of decode. The physical core count is the default; another count
        #    is taken only when it measures over 3% faster (GPU decode moves ~5% run to run).
        cands = thread_candidates(int(hw.get("cpu_cores") or 1), int(hw.get("cpu_threads") or 1))
        cores = int(hw.get("cpu_cores") or plan["threads"])
        sweep = {}
        if len(cands) >= 2:
            log("thread sweep on the %dK-token context" % (doc["tokens"] // 1024), 0.6)
            conv = list(doc["conversation"]); order = cands + cands[::-1]
            for i, n in enumerate(order):
                check_cancel()
                r = set_threads(n)
                if r.get("error"): raise RuntimeError(r["error"])
                q = "Continue with two more sentences about what this document describes, without repeating yourself."
                res = chat_messages(port, conv + [{"role": "user", "content": q}], 64); t = res.get("timings") or {}
                tps = float(t.get("predicted_per_second") or 0)
                conv += [{"role": "user", "content": q}, {"role": "assistant", "content": (res["choices"][0]["message"].get("content") or "")}]
                if int(t.get("prompt_n") or 0) > 1000: log("  (the prefix was not reused: %d tokens re-read)" % int(t["prompt_n"]))
                sweep.setdefault(n, []).append(tps)
                log("threads %2d: %.1f tok/s%s" % (n, tps, " (pass 2)" if i >= len(cands) else ""), 0.6 + 0.3 * (i + 1) / len(order))
            table = [{"threads": n, "tok_s": round(sum(v) / len(v), 2), "runs": [round(x, 1) for x in v]} for n, v in sorted(sweep.items())]
            by_n = {t["threads"]: t["tok_s"] for t in table}
            best_n = max(by_n, key=by_n.get); best = by_n[best_n]
            base_n = cores if cores in by_n else best_n
            pick = pick_threads(table, cores)
            result["sweep"] = table; result["threads"] = pick
            set_threads(pick)
            if pick == base_n: log("threads: %d, the physical cores (%.1f tok/s; the best of the sweep, %.1f at %d, did not beat it in every pass by over 3%%)" % (pick, by_n[pick], best, best_n), 0.92)
            else: log("threads: %d (%.1f tok/s against %.1f at the %d physical cores, in every pass)" % (pick, by_n[pick], by_n[base_n], base_n), 0.92)
            worst_n = min(by_n, key=by_n.get)
            if by_n[worst_n] < by_n[pick] * 0.9: result["notes"].append("%d threads measured %.0f%% slower than %d: on this CPU the extra threads starve the drive reader and the graph thread" % (worst_n, 100 * (1 - by_n[worst_n] / by_n[pick]), pick))
        else:
            result["threads"] = plan["threads"]; log("thread sweep skipped: %d CPU threads" % hw.get("cpu_threads"), 0.92)
        stats = fetch_json(f"http://127.0.0.1:{port}/stats", 3.0) or {}
        c = stats.get("expert_cache") or {}
        result["verify"] = {"chat_tps": chat["tok_s"], "chat_n": chat["n"], "doc_tps": doc["tok_s"], "doc_prefill_tps": doc["prefill_tok_s"], "doc_tokens": doc["tokens"], "doc_prefill_s": doc["prefill_s"],
                            "found": doc["found"], "hit": round(c.get("hit_rate", 0), 3), "vram_served": round(c.get("vram_served", 0), 3),
                            "predicted_chat": plan["estimates"]["decode_tps_short"], "predicted_doc": plan["estimates"]["decode_tps_long_doc"]}
        log("chat %.1f tok/s (predicted %.1f) · document prefill %.0f tok/s, decode %.1f tok/s (predicted %.1f) · passphrase %s" % (chat["tok_s"], plan["estimates"]["decode_tps_short"], doc["prefill_tok_s"], doc["tok_s"], plan["estimates"]["decode_tps_long_doc"], "found" if doc["found"] else "NOT found"), 0.96)
        if not doc["found"]: result["notes"].append("the answer did not quote the passphrase planted in the document: check the Log tab; a smaller context or the Q4 file may read better")
        # 6. memory: what the server needs besides the tier, measured; then the tier that
        #    leaves the headroom, and a restart with it when that is a different size
        mem_watch["stop"] = True; time.sleep(0.6)
        mn = mem_watch["min"]; result["mem_min_gb"] = round(mn, 1)
        built_ram = float(built.get("ram_tier_gb") or plan["ram"])
        host_other = max(2.0, result["avail_gb"] - mn - built_ram)
        result["memory"] = {"avail_gb": result["avail_gb"], "min_free_gb": round(mn, 1), "ram_tier_gb": built_ram, "host_other_gb": round(host_other, 1),
                            "state_host_gb": plan["estimates"]["state_host_gb"], "mtp": bool(plan["mtp"]), "headroom_gb": headroom_gb()}
        log("memory: %.1f GB free at the worst point; the server needs %.1f GB besides its %.1f GB tier" % (mn, host_other, built_ram))
        new_ram = max(4, int(result["avail_gb"] - host_other - headroom_gb()))
        if abs(new_ram - int(plan["ram"])) >= 1:
            log("re-sizing the RAM tier %d -> %d GB (%.1f GB headroom) and restarting the server with it" % (plan["ram"], new_ram, headroom_gb()), 0.97)
            plan["ram"] = new_ram; plan["host_other_gb"] = round(host_other, 1); result["plan"]["ram"] = new_ram
            stop_server()
            for _ in range(40):
                if not engines_running() and not port_open(STATE["port"]): break
                time.sleep(0.5)
            time.sleep(1.5)
            check_cancel()
            r = start_server(model, {**plan, "threads": result["threads"]})
            if r.get("error"): raise RuntimeError(r["error"])
            if r.get("note"): result["notes"].append(r["note"])
            st = wait_ready(log); port = st["port"]
            built2 = log_tiers(); result["built"] = built2 or built
            mem_watch.update(min=1e9, stop=False); watcher = threading.Thread(target=watch, daemon=True); watcher.start()
            chat_request(port, SWEEP_PROMPT, 64)
            chat2 = measure_chat(port)
            mem_watch["stop"] = True; time.sleep(0.6)
            result["memory"]["after_resize"] = {"ram_tier_gb": (built2 or {}).get("ram_tier_gb", new_ram), "min_free_gb": round(mem_watch["min"], 1), "chat_tps": chat2["tok_s"]}
            result["verify"]["chat_tps_resized"] = chat2["tok_s"]
            log("with the %.1f GB tier: chat %.1f tok/s, %.1f GB free at the worst point" % ((built2 or {}).get("ram_tier_gb", new_ram), chat2["tok_s"], mem_watch["min"]))
        else:
            log("the RAM tier stays at %d GB: that is what the measured memory allows at %.1f GB of headroom" % (plan["ram"], headroom_gb()))
        result["settings"] = {**result["plan"], "threads": result["threads"]}
        result["elapsed_s"] = round(time.time() - T["t0"])
        CONFIG["tune"][model["name"]] = result; save_config()
        T["result"] = result
        log("done in %d s: the server is running with the tuned flags" % result["elapsed_s"], 1.0)
    except Exception as e:
        T["error"] = str(e); log("failed: %s" % e)
        if str(e) == "cancelled" and STATE["settings"]: set_threads(STATE["settings"].get("threads") or 8)
    finally:
        mem_watch["stop"] = True
        T["running"] = False; T["cancel"] = False

def start_tune(model, preset, settings):
    with LOCK:
        if TUNE["running"] or SELFTEST["running"]: return {"error": "a tune or a self-test is already running"}
        TUNE.update(running=True, step="starting", log=[], error=None, t0=time.time(), progress=0.0, cancel=False)
    threading.Thread(target=run_tune, args=(model, preset, settings), daemon=True).start()
    return {"ok": True}

# ---- HTTP -----------------------------------------------------------------------
INDEX = os.path.join(ROOT, "tools", "console", "index.html")
class H(http.server.BaseHTTPRequestHandler):
    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code); self.send_header("Content-Type", "application/json"); self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-store"); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
    def log_message(self, *a): pass
    def do_GET(self):
        path = self.path.split("?")[0]
        q = {k: v[0] for k, v in urllib.parse.parse_qs(self.path.split("?", 1)[1]).items()} if "?" in self.path else {}
        if path in ("/", "/index.html"):
            try: body = open(INDEX, "rb").read()
            except Exception: body = b"<h1>tools/console/index.html missing</h1>"
            self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8"); self.send_header("Cache-Control", "no-store"); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body); return
        if path == "/api/models": return self._json({"models": scan_models(), "hf": HF, "locations": model_locations(), "last": CONFIG["last"]})
        if path == "/api/hardware": return self._json(hardware())
        if path == "/api/config": return self._json({"locations": model_locations(), "custom": CONFIG["custom"], "tune": CONFIG["tune"], "last": CONFIG["last"], "headroom_gb": headroom_gb()})
        if path == "/api/recommend":
            models = scan_models(); m = find_model(models, q.get("model"))
            if not m: return self._json({"error": "no models"}, 404)
            v = q.get("vision", ""); vision = None if v == "" else v in ("1", "true")
            sh = q.get("state_host", "") or None; kv = q.get("kv", "") or None
            return self._json(recommend(m, hardware(), q.get("preset", "coding"), vision, sh, kv, None, q.get("ctx") or None))
        if path == "/api/status": return self._json(status())
        if path == "/api/live": return self._json(live_state())
        if path == "/api/log": return self._json({"log": log_tail(400)})
        if path == "/api/selftest": return self._json({k: SELFTEST[k] for k in ("running", "step", "log", "result", "error")})
        if path == "/api/tune": return self._json({k: TUNE[k] for k in ("running", "step", "log", "result", "error", "progress")})
        self._json({"error": "not found"}, 404)
    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0)); body = json.loads(self.rfile.read(n) or b"{}")
        if self.path in ("/api/start", "/api/selftest", "/api/tune"):
            models = scan_models(); model = find_model(models, body.get("model"))
            if not model: return self._json({"error": "no models"}, 404)
            s = body.get("settings") or recommend(model, hardware(), body.get("preset") or "coding")
            if self.path == "/api/start": return self._json(start_server(model, s))
            if self.path == "/api/selftest": return self._json(start_selftest(model, s))
            return self._json(start_tune(model, body.get("preset") or s.get("preset") or "coding", s))
        if self.path == "/api/tune/cancel":
            TUNE["cancel"] = TUNE["running"]; return self._json({"ok": True})
        if self.path == "/api/stop": return self._json(stop_server())
        if self.path == "/api/threads": return self._json(set_threads(body.get("threads") or 8))
        if self.path == "/api/locations":
            if body.get("add"):
                p = os.path.realpath(os.path.expanduser(str(body["add"]).strip()))
                if not os.path.exists(p): return self._json({"error": "no such path: " + p})
                if not (os.path.isdir(p) or p.endswith(".gguf")): return self._json({"error": "give a folder, or a .gguf file"})
                if p != HF and p not in CONFIG["locations"]: CONFIG["locations"].append(p); save_config()
            if body.get("remove"):
                CONFIG["locations"] = [x for x in CONFIG["locations"] if x != body["remove"]]; save_config()
            return self._json({"ok": True, "locations": model_locations(), "models": scan_models()})
        if self.path == "/api/config":
            if "headroom_gb" in body:
                try: CONFIG["headroom_gb"] = min(16.0, max(0.5, float(body["headroom_gb"]))); save_config()
                except Exception: return self._json({"error": "headroom must be a number of GB"})
            return self._json({"ok": True, "headroom_gb": headroom_gb()})
        if self.path == "/api/custom":
            name = str(body.get("model") or "")
            if not name: return self._json({"error": "which model?"})
            if body.get("delete"): CONFIG["custom"].pop(name, None); save_config(); return self._json({"ok": True})
            s = body.get("settings") or {}
            keep = {k: s[k] for k in ("ctx", "kv", "ram", "threads", "batch", "reserve", "port", "think", "think_budget", "skip_miss", "spec_block", "vision", "mtp", "state_host") if k in s}
            keep["saved"] = time.strftime("%Y-%m-%d %H:%M")
            CONFIG["custom"][name] = keep; save_config()
            return self._json({"ok": True})
        self._json({"error": "not found"}, 404)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--port", type=int, default=8090); ap.add_argument("--server-port", type=int, default=8080)
    ap.add_argument("--start", action="store_true", help="start the last served model and tier right away")
    ap.add_argument("--model", help="with --start: a model path or name instead of the last one"); ap.add_argument("--preset", help="with --start: chat | coding | coding_plus | custom")
    a = ap.parse_args(); STATE["port"] = a.server_port
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", a.port), H)
    print(f"qwfn console on http://127.0.0.1:{a.port}  (server port {a.server_port}, models from {HF}" + (" and %d more location(s)" % len(CONFIG["locations"]) if CONFIG["locations"] else "") + ")", flush=True)
    if a.start:
        models = scan_models(); last = CONFIG["last"]
        m = find_model(models, a.model or last.get("model")) or (models[0] if models else None)
        if not m: print("--start: no model found", flush=True)
        else:
            s = recommend(m, hardware(), a.preset or last.get("preset") or "coding")
            r = start_server(m, s)
            print("--start: " + (r.get("error") or "started %s (%s tier) on port %d" % (m["name"], s["preset"], s["port"])), flush=True)
    try: srv.serve_forever()
    except KeyboardInterrupt: pass
    finally: stop_server()

if __name__ == "__main__":
    main()
