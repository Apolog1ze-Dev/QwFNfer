#!/usr/bin/env python3
"""qwfn console: a local web page to see which models are downloaded, pick one,
get settings for this machine (three presets: chat, agentic coding, agentic coding+),
start/stop qwfn-server, self-test it on this hardware and watch it live (state,
tokens/s, prefill speed, tokens served, endpoint).

    python3 tools/qwfn_console.py            # http://127.0.0.1:8090
    python3 tools/qwfn_console.py --port 8091 --server-port 8080

Standard library only. Binds 127.0.0.1. One engine at a time: the console starts
one qwfn-server and refuses to start a second while any qwfn engine is running.
Vision: when an mmproj-*.gguf sits next to the model (or in its snapshot directory)
the server is started with --mmproj by default, so images sent by a harness work.
"""
import argparse, glob, http.server, json, math, os, random, signal, socket, subprocess, sys, threading, time, urllib.parse, urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HF = os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub")
# The engine: bin/ in the release bundle, build/ in a source checkout, or QWFN_SERVER.
SERVER_BIN = os.environ.get("QWFN_SERVER") or next((p for p in (os.path.join(ROOT, "bin", "qwfn-server"), os.path.join(ROOT, "build", "qwfn-server")) if os.path.exists(p)), os.path.join(ROOT, "build", "qwfn-server"))
LOG_DIR = os.path.join(os.path.expanduser("~/.cache"), "qwfn-console")
os.makedirs(LOG_DIR, exist_ok=True)

STATE = {"proc": None, "model": None, "settings": None, "started": 0.0, "log": os.path.join(LOG_DIR, "server.log"), "port": 8080}
LOCK = threading.Lock()

# ---- models ------------------------------------------------------------------
QUANT_BLOCK_MB = {"Q3_K_XL": 2.4, "Q4_K_XL": 3.4, "IQ1_S": 1.5}   # expert block size, for tier estimates
QUANT_CORE_GB  = {"Q3_K_XL": 4.68, "Q4_K_XL": 4.83, "IQ1_S": 4.5}  # dense core on the GPU
COLD_ONLY = ("IQ1_S",)   # 1-bit checkpoints: a cold tier (--cold), never served on their own

def find_mmproj(model_dir):
    """The vision projector shipped with the model: mmproj-*.gguf in the quant's own
    directory or, as Hugging Face lays this repo out, in the snapshot directory above it."""
    for d in (model_dir, os.path.dirname(model_dir)):
        c = sorted(glob.glob(os.path.join(d, "mmproj*.gguf")))
        if c: return c[0]
    return None

def find_mtp(model_dir):
    """The checkpoint's nextn draft head (MTP/mtp-*.gguf). Hugging Face may have put it in a
    different snapshot directory of the same repo than the quant, so search the whole repo."""
    repo = os.path.dirname(os.path.dirname(os.path.dirname(model_dir)))   # .../models--x--y
    c = sorted(glob.glob(os.path.join(repo, "snapshots", "*", "MTP", "mtp-*.gguf")) + glob.glob(os.path.join(repo, "snapshots", "*", "mtp-*.gguf")))
    return c[0] if c else None

def scan_models():
    out = []
    pattern = os.path.join(HF, "*Qwen3.8-Flash-Next*", "snapshots", "*", "*", "*.gguf")
    for f in sorted(glob.glob(pattern)):
        base = os.path.basename(f)
        if base.startswith("mmproj") or base.startswith("mtp-"): continue
        if "-of-" in base and "-00001-of-" not in base: continue
        d = os.path.dirname(f); quant = os.path.basename(d)
        key = next((k for k in QUANT_BLOCK_MB if k in quant), None)
        if key in COLD_ONLY: continue
        total = sum(os.stat(s).st_size for s in glob.glob(os.path.join(d, "*.gguf")) if not os.path.basename(s).startswith(("mmproj", "mtp-")))
        mm = find_mmproj(d); mtp = find_mtp(d)
        out.append({"id": len(out), "name": quant, "path": f, "size_gb": round(total / 1e9, 1), "quant": key or quant,
                    "mmproj": mm, "mmproj_gb": round(os.stat(mm).st_size / 1e9, 2) if mm else 0.0,
                    "mtp": mtp, "mtp_gb": round(os.stat(mtp).st_size / 1e9, 2) if mtp else 0.0})
    return out

# ---- hardware ----------------------------------------------------------------
def hardware():
    hw = {"gpu": None, "vram_total_mb": 0, "vram_used_mb": 0, "desktop_gpu": False, "ram_total_gb": 0, "ram_available_gb": 0, "cpu_threads": os.cpu_count() or 1}
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
        mi = {}
        for line in open("/proc/meminfo"):
            k, v = line.split(":", 1); mi[k] = int(v.strip().split()[0])
        hw["ram_total_gb"] = round(mi.get("MemTotal", 0) / 1048576, 1)
        hw["ram_available_gb"] = round(mi.get("MemAvailable", 0) / 1048576, 1)
    except Exception:
        pass
    return hw

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
QUANT_GPU_MS = {"Q3_K_XL": 24.0, "Q4_K_XL": 32.0, "IQ1_S": 24.0}   # graph A with the speculative block, per token
LOOKUPS_PER_TOKEN = 480   # 48 layers x 10 routed experts
# Attention state per 1K tokens of context: the KV cache (by its type), the indexer key
# cache and the pooled block keys; plus the constant DeltaNet state. The KV and indexer
# caches can live in pinned RAM instead of VRAM (--state-host), read over PCIe: measured
# on the doc replay at 131K, +0.35 ms/token for the indexer, +1.7 ms/token for the KV
# cache; the VRAM they free is worth ~4% of decode per GB at 131K and is what gives the
# 256K preset an expert tier at all (7.0 -> 9.7 tok/s measured, 2026-09-09).
STATE_HOST_OPTIONS = ["none", "idx", "kv,idx"]
STATE_HOST_MS = {"none": 0.0, "idx": 0.35, "kv": 1.7, "kv,idx": 2.0}
def state_parts(ctx, kv):
    per_k = {"q4_0": 6.9, "q8_0": 13.1, "f16": 26.2}[kv]      # KV MB per 1K tokens
    k = ctx / 1024 / 1024
    return {"kv": per_k * k, "idx": 3.1 * k, "pooled": 0.8 * k, "delta": 0.113}
def state_gb(ctx, kv):
    return sum(state_parts(ctx, kv).values())
def state_host_gb(ctx, kv, state_host):
    p = state_parts(ctx, kv)
    return (p["kv"] if "kv" in state_host else 0.0) + (p["idx"] if "idx" in state_host else 0.0)
def state_vram_gb(ctx, kv, state_host):
    return state_gb(ctx, kv) - state_host_gb(ctx, kv, state_host)

def predict(quant, tier_gb, ram_gb, long_doc, extra_ms=0.0):
    block = QUANT_BLOCK_MB.get(quant, 2.4)
    vb = max(0.0, tier_gb) * 1024 / block; rb = ram_gb * 1024 / block
    f_v = max(0.0, 0.95 * (1 - math.exp(-vb / 1500)) - (0.06 if long_doc else 0.0)) if vb > 0 else 0.0
    f_h = 0.97 * (1 - math.exp(-(vb + rb) / 1050))
    f_h = max(f_h, f_v)
    io = 1.4 * (block / 2.4) ** 1.4
    ms = QUANT_GPU_MS.get(quant, 24.0) + extra_ms + LOOKUPS_PER_TOKEN * ((f_h - f_v) * 0.110 + (1 - f_h) * io) + 8.0
    return {"tok_s": round(1000 / ms, 1), "vram_served": round(f_v, 3), "hit": round(f_h, 3), "blocks": int(vb)}

def recommend(model, hw, preset="coding", vision=None, state_host=None):
    """Settings for one preset on this machine, with every context step's cost and the
    three presets summarised, so the page can show the choice without a table."""
    q = model["quant"]
    core = QUANT_CORE_GB.get(q, 4.7)
    # Vision: the projector's weights stay in host memory and are staged onto the GPU
    # only while an image is encoded (~40 ms per image), so vision costs no VRAM at
    # decode. Default on whenever the file is there: a harness that sends an image gets
    # an answer instead of a 400.
    forced_vision = vision   # None: the model's default (on when the file is there); True/False: the user's choice
    if vision is None: vision = bool(model.get("mmproj"))
    vision = bool(vision and model.get("mmproj"))
    mm_gb = model.get("mmproj_gb", 0.0) if vision else 0.0   # host memory, not VRAM
    vram = hw["vram_total_mb"] / 1024.0
    reserve_mb = 1024 if hw["desktop_gpu"] else 768
    # The desktop's own VRAM use, measured now (minus any qwfn engine).
    used = hw["vram_used_mb"] / 1024.0
    eng_used = hw.get("qwfn_vram_mb", 0) / 1024.0
    desktop_use = max(0.2, used - eng_used) if hw["desktop_gpu"] else 0.1
    avail = hw["ram_available_gb"] if not engines_running() else max(hw["ram_available_gb"], hw["ram_total_gb"] - 8.0)
    batch = 4096 if vram >= 12 else 2048
    forced_state = state_host if state_host in STATE_HOST_OPTIONS else None   # None: chosen per context below
    # What the engine does with the VRAM left after the dense core and the context's state:
    # it takes OVERHEAD_GB for its CUDA context, decode state and graph arenas (measured
    # against the tier it actually built at 160K), and it only builds an expert tier that
    # can hold the prefill's dynamic buffer (staging, work set, arenas: LEND_GB), because
    # that buffer is lent by the tier while a prompt streams. Below that it runs with no
    # VRAM tier at all: every expert comes from RAM or the NVMe, about 25% slower.
    OVERHEAD_GB = 1.2
    lend_gb = (4.4 if batch >= 4096 else 3.9) - (0.27 if q == "Q3_K_XL" else 0.0)
    def tier_for(c, kv, sh):
        t = vram - core - state_vram_gb(c, kv, sh) - reserve_mb / 1024 - desktop_use - OVERHEAD_GB
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
    def ram_for(sh_gb):
        return max(4, min(12, int(0.6 * (avail - sh_gb) - 3.2)))   # the pinned state comes out of the same RAM
    def option(c, with_vision):
        kv = "q4_0" if c >= 32768 else "q8_0"
        sh = state_host_for(c, kv)
        tier = tier_for(c, kv, sh)
        shg = state_host_gb(c, kv, sh); ram = ram_for(shg)
        short = predict(q, tier, ram, False, STATE_HOST_MS[sh]); longd = predict(q, tier, ram, True, STATE_HOST_MS[sh])
        return {"ctx": c, "kv": kv, "tier_gb": round(tier, 2), "blocks": short["blocks"], "state_gb": round(state_gb(c, kv), 2), "vision": with_vision,
                "state_host": sh, "state_host_gb": round(shg, 2), "state_vram_gb": round(state_vram_gb(c, kv, sh), 2), "ram": ram,
                "tok_s_short": short["tok_s"], "tok_s_long_doc": longd["tok_s"], "vram_served": short["vram_served"], "hit": short["hit"]}
    presets = []
    for pid, label, ctx, blurb in PRESETS:
        o = option(ctx, vision); note = ""
        if o["state_host"] != "none": note = "attention state in RAM (%s): %.1f GB of VRAM for the expert tier" % (o["state_host"], o["state_host_gb"])
        if o["tier_gb"] == 0:
            fallback = [option(c, vision) for c in CTX_STEPS if c < ctx]
            fallback = [f for f in fallback if f["tier_gb"] > 0]
            if fallback: o = fallback[-1]; note = "no room for an expert tier at %dK on this GPU: %dK instead" % (ctx // 1024, o["ctx"] // 1024)
            else: note = "no room for a VRAM expert tier on this GPU: experts come from RAM and the NVMe"
        presets.append({"id": pid, "label": label, "blurb": blurb, "ctx": ctx, "fits": o["ctx"] == ctx, "note": note, "ctx_actual": o["ctx"], "kv": o["kv"], "vision": o["vision"],
                        "tier_gb": o["tier_gb"], "blocks": o["blocks"], "tok_s_short": o["tok_s_short"], "tok_s_long_doc": o["tok_s_long_doc"], "vram_served": o["vram_served"]})
    if preset not in [p["id"] for p in presets]: preset = "coding"
    p = next(p for p in presets if p["id"] == preset)
    vision = p["vision"]; mm_gb = model.get("mmproj_gb", 0.0) if vision else 0.0
    options = [option(c, vision) for c in CTX_STEPS]
    chosen = next(o for o in options if o["ctx"] == p["ctx_actual"])
    return {
        "ctx": chosen["ctx"], "kv": chosen["kv"], "ram": chosen["ram"], "batch": batch, "reserve": reserve_mb, "think": "xhigh", "think_budget": 6000,
        "skip_miss": False, "spec_block": True, "port": STATE["port"], "preset": preset,
        "vision": vision, "mmproj": model.get("mmproj"), "mmproj_gb": model.get("mmproj_gb", 0.0),
        "state_host": chosen["state_host"],
        "mtp": False, "mtp_file": model.get("mtp"), "mtp_gb": model.get("mtp_gb", 0.0),
        "estimates": {"vram_tier_gb": chosen["tier_gb"], "vram_tier_blocks": chosen["blocks"], "state_gb": chosen["state_gb"], "dense_core_gb": core,
                      "state_host_gb": chosen["state_host_gb"], "state_vram_gb": chosen["state_vram_gb"],
                      "mmproj_gb": round(mm_gb, 2),
                      "desktop_use_gb": round(desktop_use, 2), "decode_tps_short": chosen["tok_s_short"], "decode_tps_long_doc": chosen["tok_s_long_doc"],
                      "vram_served": chosen["vram_served"], "prefill_tps_long": 275 if q != "IQ1_S" else 0},
        "options": options, "presets": presets,
    }

# ---- server control ----------------------------------------------------------
def engines_running():
    try:
        out = subprocess.run(["pgrep", "-a", "-x", "qwfn-server"], capture_output=True, text=True, timeout=3).stdout
        procs = [l for l in out.splitlines() if l.strip()]
        out2 = subprocess.run(["pgrep", "-l", "qwfn-gen"], capture_output=True, text=True, timeout=3).stdout
        procs += [l for l in out2.splitlines() if l.strip()]
        return procs
    except Exception:
        return []

def port_open(port):
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.5): return True
    except OSError:
        return False

def fetch_json(url, timeout=2.0):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r: return json.load(r)
    except Exception:
        return None

def start_server(model, s):
    with LOCK:
        if STATE["proc"] and STATE["proc"].poll() is None: return {"error": "a server started by this console is already running"}
        running = engines_running()
        if running: return {"error": "an engine is already running on this machine (one at a time): " + "; ".join(running)[:300]}
        if not os.path.exists(SERVER_BIN): return {"error": f"{SERVER_BIN} not found: install the release bundle, or build first (cmake --build build)"}
        argv = [SERVER_BIN, model["path"], "--ram", str(int(s["ram"])), "--ctx", str(int(s["ctx"])), "--batch", str(int(s["batch"])),
                "--kv", s["kv"], "--reserve", str(int(s["reserve"])), "--think", s["think"], "--think-budget", str(int(s["think_budget"])),
                "--port", str(int(s["port"]))]
        if s.get("skip_miss"): argv.append("--skip-miss")
        if s.get("spec_block", True): argv.append("--spec-block")
        if s.get("mtp") and s.get("skip_miss"):
            s["mtp"] = False   # the engine would refuse every request: a verified pair and skip-miss do not combine
        if s.get("mtp"):
            if not model.get("mtp"):
                return {"error": "no MTP/mtp-*.gguf in this model's repository: download it into the snapshot directory, or turn the draft head off"}
            argv += ["--mtp", model["mtp"]]
        if s.get("cold_path"): argv += ["--cold", s["cold_path"]]
        if s.get("vision"):
            if not model.get("mmproj"):
                return {"error": "no mmproj-*.gguf next to this model: download mmproj-F16.gguf into its snapshot directory, or turn Vision off"}
            argv += ["--mmproj", model["mmproj"]]
        if s.get("state_host") in ("idx", "kv", "kv,idx"): argv += ["--state-host", s["state_host"]]
        log = open(STATE["log"], "w")
        log.write("$ " + " ".join(argv) + "\n")
        if s.get("skip_miss") and not s.get("mtp") and model.get("mtp"):
            log.write("[console] draft head left off: a verified pair and skip-miss do not combine (skip-miss is one token at a time)\n")
        log.flush()
        # The bundle keeps ggml, the CUDA runtime and liburing next to the engine; the loader
        # needs the directory for the libraries the CUDA backend dlopens.
        env = dict(os.environ); bindir = os.path.dirname(SERVER_BIN)
        if os.path.exists(os.path.join(bindir, "libggml-base.so.0")):
            env["LD_LIBRARY_PATH"] = bindir + (":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
        try:
            proc = subprocess.Popen(argv, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=env)
        except Exception as e:
            return {"error": f"cannot start: {e}"}
        STATE.update(proc=proc, model=model, settings=s, started=time.time(), port=int(s["port"]))
        return {"ok": True, "pid": proc.pid, "argv": argv}

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

def status():
    p = STATE["proc"]; alive = bool(p) and p.poll() is None
    port = STATE["port"]
    st = {"console_started": alive, "pid": p.pid if alive else None, "uptime_s": round(time.time() - STATE["started"]) if alive else 0,
          "model": STATE["model"]["name"] if (alive and STATE["model"]) else None, "settings": STATE["settings"] if alive else None,
          "port": port, "endpoint": f"http://127.0.0.1:{port}/v1", "state": "stopped", "external": False}
    if not alive and p is not None and p.poll() is not None:
        st["exit_code"] = p.returncode; st["state"] = "exited"
    if port_open(port):
        health = fetch_json(f"http://127.0.0.1:{port}/health", 1.5)
        if health:
            stats = fetch_json(f"http://127.0.0.1:{port}/stats", 2.0); props = fetch_json(f"http://127.0.0.1:{port}/props", 2.0)
            st["stats"] = stats; st["props"] = props
            st["state"] = "busy" if (stats and stats.get("busy")) else "ready"
            if not alive:
                st["external"] = True
                mf = (props or {}).get("model_file") or ""
                st["model"] = os.path.basename(os.path.dirname(mf)) if mf else ((props or {}).get("model") or st["model"])
            if props: st["model_id"] = props.get("model")
    elif alive:
        st["state"] = "loading"
    st["log"] = log_tail(40)
    st["engines"] = engines_running()
    st["selftest_running"] = SELFTEST["running"]
    return st

# ---- self-test: the chosen model on this hardware, through the server ---------
# Starts the server if none is running (with the settings the page shows), then measures
# what a user would see: a short chat (decode tok/s), and a long document with a passphrase
# planted in it (prefill tok/s, decode tok/s on that context, and whether the answer found
# the passphrase). The server is left running afterwards.
SELFTEST = {"running": False, "step": "", "log": [], "result": None, "error": None, "t0": 0.0}
SELFTEST_FILE = os.path.join(LOG_DIR, "selftest.json")
try: SELFTEST["result"] = json.load(open(SELFTEST_FILE))
except Exception: pass
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
    body = json.dumps({"model": "qwfn", "messages": [{"role": "user", "content": content}], "max_tokens": max_tokens,
                       "reasoning_effort": "off", "temperature": 0.0, "stream": False}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r: return json.load(r)

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
        deadline = time.time() + 1200
        while time.time() < deadline:
            st = status()
            if st["state"] in ("ready", "busy"): break
            if st["state"] in ("exited", "stopped"):
                raise RuntimeError("the server exited while loading (exit code %s): see the Log tab" % st.get("exit_code"))
            time.sleep(2)
        else:
            raise RuntimeError("the server did not become ready within 20 minutes")
        port = st["port"]
        run = st.get("settings") or {}
        ctx = int(run.get("ctx") or (st.get("props") or {}).get("n_ctx") or settings.get("ctx") or 32768)
        log("warm-up")
        chat_request(port, "Say hello in one short sentence.", 24)
        log("short chat: a three-sentence answer")
        r = chat_request(port, "Explain in three sentences why mixture-of-experts models are cheaper to run than dense models with the same parameter count.", 200)
        t = r.get("timings") or {}
        chat = {"tok_s": round(t.get("predicted_per_second", 0), 1), "n": t.get("predicted_n", 0), "prefill_tok_s": round(t.get("prompt_per_second", 0), 0),
                "answer": (r["choices"][0]["message"].get("content") or "").strip()[:400]}
        n_doc = min(32768, ctx // 2)
        doc, phrase = selftest_document(n_doc)
        log("long document: prefilling about %dK tokens, then a grounded answer" % (n_doc // 1024))
        r = chat_request(port, doc + "\n\nTwo things: (1) What is the passphrase that the note in this document asks the reader to remember? Quote it exactly. (2) In two sentences, what is this document about?", 160)
        t = r.get("timings") or {}
        answer = (r["choices"][0]["message"].get("content") or "").strip()
        docres = {"tokens": t.get("prompt_n", 0), "prefill_tok_s": round(t.get("prompt_per_second", 0), 0), "prefill_s": round(t.get("prompt_ms", 0) / 1000, 0),
                  "tok_s": round(t.get("predicted_per_second", 0), 1), "n": t.get("predicted_n", 0), "found": phrase.lower() in answer.lower(),
                  "phrase": phrase, "answer": answer[:400]}
        stats = fetch_json(f"http://127.0.0.1:{port}/stats", 3.0) or {}
        c = stats.get("expert_cache") or {}
        hw = hardware()
        est = recommend(model, hw, run.get("preset") or settings.get("preset") or "coding", run.get("vision"))
        o = min(est["options"], key=lambda o: abs(o["ctx"] - ctx))
        T["result"] = {"model": model["name"], "ctx": ctx, "preset": run.get("preset") or settings.get("preset"), "flags": run,
                       "date": time.strftime("%Y-%m-%d %H:%M"), "chat": chat, "doc": docres,
                       "cache": {"hit": round(c.get("hit_rate", 0), 3), "vram_served": round(c.get("vram_served", 0), 3)},
                       "vram_used_gb": round(hw["vram_used_mb"] / 1024, 1), "vram_total_gb": round(hw["vram_total_mb"] / 1024, 1),
                       "ram_available_gb": hw["ram_available_gb"], "gpu": hw.get("gpu"),
                       "predicted": {"chat": o["tok_s_short"], "doc": o["tok_s_long_doc"], "vram_served": o["vram_served"]},
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
        if SELFTEST["running"]: return {"error": "a self-test is already running"}
        SELFTEST.update(running=True, step="starting", log=[], error=None, t0=time.time())
    threading.Thread(target=run_selftest, args=(model, settings), daemon=True).start()
    return {"ok": True}

# ---- HTTP -----------------------------------------------------------------------
INDEX = os.path.join(ROOT, "tools", "console", "index.html")
class H(http.server.BaseHTTPRequestHandler):
    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code); self.send_header("Content-Type", "application/json"); self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
    def log_message(self, *a): pass
    def do_GET(self):
        path = self.path.split("?")[0]
        q = dict(x.split("=", 1) for x in self.path.split("?", 1)[1].split("&") if "=" in x) if "?" in self.path else {}
        if path in ("/", "/index.html"):
            try: body = open(INDEX, "rb").read()
            except Exception: body = b"<h1>tools/console/index.html missing</h1>"
            self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8"); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body); return
        if path == "/api/models": return self._json({"models": scan_models(), "hf": HF})
        if path == "/api/hardware": return self._json(hardware())
        if path == "/api/recommend":
            models = scan_models(); i = int(q.get("model", 0))
            if not models: return self._json({"error": "no models"}, 404)
            v = q.get("vision", ""); vision = None if v == "" else v in ("1", "true")
            sh = urllib.parse.unquote(q.get("state_host", "")) or None
            return self._json(recommend(models[min(i, len(models) - 1)], hardware(), q.get("preset", "coding"), vision, sh))
        if path == "/api/status": return self._json(status())
        if path == "/api/log": return self._json({"log": log_tail(400)})
        if path == "/api/selftest": return self._json({k: SELFTEST[k] for k in ("running", "step", "log", "result", "error")})
        self._json({"error": "not found"}, 404)
    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0)); body = json.loads(self.rfile.read(n) or b"{}")
        if self.path in ("/api/start", "/api/selftest"):
            models = scan_models(); i = int(body.get("model", 0))
            if not models: return self._json({"error": "no models"}, 404)
            model = models[min(i, len(models) - 1)]
            s = body.get("settings") or recommend(model, hardware())
            return self._json(start_server(model, s) if self.path == "/api/start" else start_selftest(model, s))
        if self.path == "/api/stop": return self._json(stop_server())
        self._json({"error": "not found"}, 404)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--port", type=int, default=8090); ap.add_argument("--server-port", type=int, default=8080)
    a = ap.parse_args(); STATE["port"] = a.server_port
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", a.port), H)
    print(f"qwfn console on http://127.0.0.1:{a.port}  (server port {a.server_port}, models from {HF})", flush=True)
    try: srv.serve_forever()
    except KeyboardInterrupt: pass
    finally: stop_server()

if __name__ == "__main__":
    main()
