#!/usr/bin/env python3
"""qwfn console: a local web page to see which models are downloaded, pick one,
get settings recommended for this machine, start/stop qwfn-server and watch it
live (state, tokens/s, prefill speed, tokens served, endpoint).

    python3 tools/qwfn_console.py            # http://127.0.0.1:8090
    python3 tools/qwfn_console.py --port 8091 --server-port 8080

Standard library only. Binds 127.0.0.1. One engine at a time: the console starts
one qwfn-server and refuses to start a second while any qwfn engine is running.
Vision: when an mmproj-*.gguf sits next to the model (or in its snapshot directory)
the server is started with --mmproj by default, so images sent by a harness work.
"""
import argparse, glob, http.server, json, os, re, signal, socket, subprocess, sys, threading, time, urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HF = os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub")
SERVER_BIN = os.path.join(ROOT, "build", "qwfn-server")
LOG_DIR = os.path.join(os.path.expanduser("~/.cache"), "qwfn-console")
os.makedirs(LOG_DIR, exist_ok=True)

STATE = {"proc": None, "model": None, "settings": None, "started": 0.0, "log": os.path.join(LOG_DIR, "server.log"), "port": 8080}
LOCK = threading.Lock()
HOLDS = {}   # screenshot holds, see /api/hold

# ---- models ------------------------------------------------------------------
NOTES = {
    "Q3_K_XL": ("recommended", "14.7 tok/s at 133K on the reference machine; the measured choice"),
    "Q4_K_XL": ("slower", "35-42% slower decode than Q3_K_XL here: expert blocks are 42% larger, so both tiers hold ~30% fewer"),
    "IQ1_S":   ("cold only", "1-bit checkpoint: meant as the cold tier (--cold), not to serve"),
}
QUANT_BLOCK_MB = {"Q3_K_XL": 2.4, "Q4_K_XL": 3.4, "IQ1_S": 1.5}   # expert block size, for tier estimates
QUANT_CORE_GB  = {"Q3_K_XL": 4.68, "Q4_K_XL": 4.83, "IQ1_S": 4.5}  # dense core on the GPU
QUANT_TPS      = {"Q3_K_XL": (17.0, 14.7), "Q4_K_XL": (11.0, 9.5), "IQ1_S": (0, 0)}   # (short ctx, 133K) measured

def find_mmproj(model_dir):
    """The vision projector shipped with the model: mmproj-*.gguf in the quant's own
    directory or, as Hugging Face lays this repo out, in the snapshot directory above it."""
    for d in (model_dir, os.path.dirname(model_dir)):
        c = sorted(glob.glob(os.path.join(d, "mmproj*.gguf")))
        if c: return c[0]
    return None

def scan_models():
    out = []
    pattern = os.path.join(HF, "*Qwen3.8-Flash-Next*", "snapshots", "*", "*", "*.gguf")
    for f in sorted(glob.glob(pattern)):
        base = os.path.basename(f)
        if base.startswith("mmproj") or base.startswith("mtp-"): continue
        if "-of-" in base and "-00001-of-" not in base: continue
        d = os.path.dirname(f); quant = os.path.basename(d)
        total = sum(os.stat(s).st_size for s in glob.glob(os.path.join(d, "*.gguf")) if not os.path.basename(s).startswith(("mmproj", "mtp-")))
        key = next((k for k in NOTES if k in quant), None)
        tag, note = NOTES.get(key, ("", ""))
        mm = find_mmproj(d)
        out.append({"id": len(out), "name": quant, "path": f, "size_gb": round(total / 1e9, 1), "tag": tag, "note": note, "quant": key or quant,
                    "mmproj": mm, "mmproj_gb": round(os.stat(mm).st_size / 1e9, 2) if mm else 0.0})
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

# ---- recommendations: a cost model fitted to this session's measurements -----
# Per token: GPU graph time + CPU time for the experts served from RAM + NVMe
# time for the misses. What the hardware and the model decide:
#   blocks in VRAM   = (VRAM - dense core - KV/indexer state at ctx - reserve - desktop use) / block size
#   VRAM-served f_v  = 0.95 (1 - exp(-blocks / 1500))          fits Q3 86% @3392-3717, Q4 68% @1834, 74% @2212, 78% @2447
#   cache hit  f_h   = 0.97 (1 - exp(-(vram+ram blocks) / 1800)) fits 95-96% (Q3), 94% (Q4); -2 pts on a real long document
#   ms/token         = gpu + 480 [ (f_h - f_v) 0.086 + (1 - f_h) io ] + 8,   io = 1.1 (block/2.4 MB)^1.4 ms per missed expert
# Predicts Q3 4K 18.5 (measured 16.8-18), Q4 4K 10.7 (10.2), Q3 133K document 14.7 (14.7), Q4 133K document 8.7 (8.6).
CTX_STEPS = [8192, 16384, 32768, 65536, 131072, 163840, 262144]
QUANT_GPU_MS = {"Q3_K_XL": 23.0, "Q4_K_XL": 24.0, "IQ1_S": 23.0}
LOOKUPS_PER_TOKEN = 480   # 48 layers x 10 routed experts
def state_gb(ctx, kv):
    per_k = {"q4_0": 6.9, "q8_0": 13.1, "f16": 26.2}[kv]      # KV MB per 1K tokens
    return (ctx / 1024) * (per_k + 3.1 + 0.8) / 1024 + 0.113   # + indexer keys + pooled block keys + DeltaNet state

def predict(quant, tier_gb, ram_gb, long_doc):
    block = QUANT_BLOCK_MB.get(quant, 2.4)
    vb = max(0.0, tier_gb) * 1024 / block; rb = ram_gb * 1024 / block
    import math
    f_v = 0.95 * (1 - math.exp(-vb / 1500))
    f_h = 0.97 * (1 - math.exp(-(vb + rb) / 1800)) - (0.02 if long_doc else 0.0)
    f_h = max(f_h, f_v)
    io = 1.1 * (block / 2.4) ** 1.4
    ms = QUANT_GPU_MS.get(quant, 23.0) + LOOKUPS_PER_TOKEN * ((f_h - f_v) * 0.086 + (1 - f_h) * io) + 8.0
    return {"tok_s": round(1000 / ms, 1), "vram_served": round(f_v, 3), "hit": round(f_h, 3), "blocks": int(vb)}

def recommend(model, hw, priority="balanced", vision=None):
    q = model["quant"]
    core = QUANT_CORE_GB.get(q, 4.7)
    # Vision: the projector is loaded onto the GPU after the expert tier is sized, so
    # qwfn-server adds its file size (+128 MB of graph arena) to --reserve. Default on
    # whenever the file is there: a harness that sends an image gets an answer instead of
    # a 400, and the cost is a few hundred expert blocks.
    if vision is None: vision = bool(model.get("mmproj"))
    vision = bool(vision and model.get("mmproj"))
    mm_gb = (model.get("mmproj_gb", 0.0) + 0.128) if vision else 0.0
    vram = hw["vram_total_mb"] / 1024.0
    reserve_mb = 1024 if hw["desktop_gpu"] else 768
    # The desktop's own VRAM use, measured now (minus any qwfn engine).
    used = hw["vram_used_mb"] / 1024.0
    eng_used = hw.get("qwfn_vram_mb", 0) / 1024.0
    desktop_use = max(0.2, used - eng_used) if hw["desktop_gpu"] else 0.1
    avail = hw["ram_available_gb"] if not engines_running() else max(hw["ram_available_gb"], hw["ram_total_gb"] - 8.0)
    ram = max(4, min(12, int(0.6 * avail - 3.2)))
    batch = 4096 if vram >= 12 else 2048
    options = []
    for c in CTX_STEPS:
        kv = "q4_0" if c >= 32768 else "q8_0"
        tier = vram - core - state_gb(c, kv) - reserve_mb / 1024 - desktop_use - 0.2 - mm_gb   # 0.2: residency tables, work set
        if tier < 2.0: continue
        short = predict(q, tier, ram, False); longd = predict(q, tier, ram, True)
        options.append({"ctx": c, "kv": kv, "tier_gb": round(tier, 2), "blocks": short["blocks"], "state_gb": round(state_gb(c, kv), 2),
                        "tok_s_short": short["tok_s"], "tok_s_long_doc": longd["tok_s"], "vram_served": short["vram_served"], "hit": short["hit"]})
    if not options:
        options.append({"ctx": 8192, "kv": "q8_0", "tier_gb": 0.0, "blocks": 0, "state_gb": round(state_gb(8192, "q8_0"), 2), "tok_s_short": 0, "tok_s_long_doc": 0, "vram_served": 0, "hit": 0})
    best_speed = max(o["tok_s_long_doc"] for o in options)
    if priority == "fastest":
        chosen = next(o for o in options if o["ctx"] >= 32768) if any(o["ctx"] >= 32768 for o in options) else options[-1]
    elif priority == "context":
        chosen = options[-1]
    else:   # balanced: the largest context that keeps long-document decode within 10% of the best
        chosen = [o for o in options if o["tok_s_long_doc"] >= 0.9 * best_speed][-1]
    notes = []
    if q == "IQ1_S": notes.append("This file is the cold checkpoint (1-bit). Serve Q3_K_XL and pass this one with --cold if you want the cold tier; the numbers below assume it could be served.")
    if q == "Q4_K_XL": notes.append("Q4_K_XL's expert blocks are 42% larger than Q3_K_XL's, so every GB of VRAM holds 30% fewer experts and every miss reads more: the table shows what each context costs on this GPU.")
    lo, hi = options[-1], options[0]
    if hi["tok_s_long_doc"] > 0 and lo["tok_s_long_doc"] < 0.85 * hi["tok_s_long_doc"]:
        notes.append("Going from %sK to %sK context costs %d%% of long-document decode on this GPU (%.1f -> %.1f tok/s): the KV/indexer state comes out of the expert tier." % (hi["ctx"] // 1024, lo["ctx"] // 1024, round(100 * (1 - lo["tok_s_long_doc"] / hi["tok_s_long_doc"])), hi["tok_s_long_doc"], lo["tok_s_long_doc"]))
    if hw["desktop_gpu"]: notes.append("A desktop session shares this GPU (%.1f GB in use now): reserve %d MB and that usage are taken off the tier." % (desktop_use, reserve_mb))
    if vision:
        notes.append("Vision on: %s (%.2f GB) is loaded onto the GPU and comes out of the expert tier, about %d blocks. Images arrive as OpenAI image_url content parts (base64 data: URLs); turn it off for a text-only server." % (os.path.basename(model["mmproj"]), model.get("mmproj_gb", 0.0), int(mm_gb * 1024 / QUANT_BLOCK_MB.get(q, 2.4))))
    elif not model.get("mmproj"):
        notes.append("No mmproj-*.gguf next to this model, so image input is unavailable: download mmproj-F16.gguf into the model's snapshot directory to enable it.")
    notes.append("RAM tier %d GB: the engine's own clamp (60%% of available memory minus headroom). More RAM tier moves the cache hit rate by about a point; it is not the lever." % ram)
    notes.append("Thinking: xhigh by default; --think-budget 6000 keeps a hard think under ~8 min at Q4 speed, ~6 at Q3. Harnesses with no thinking toggle can end a message with /no_think.")
    return {
        "ctx": chosen["ctx"], "kv": chosen["kv"], "ram": ram, "batch": batch, "reserve": reserve_mb, "think": "xhigh", "think_budget": 6000,
        "skip_miss": False, "port": STATE["port"], "priority": priority,
        "vision": vision, "mmproj": model.get("mmproj"), "mmproj_gb": model.get("mmproj_gb", 0.0),
        "estimates": {"vram_tier_gb": chosen["tier_gb"], "vram_tier_blocks": chosen["blocks"], "state_gb": chosen["state_gb"], "dense_core_gb": core,
                      "mmproj_gb": round(mm_gb, 2),
                      "desktop_use_gb": round(desktop_use, 2), "decode_tps_short": chosen["tok_s_short"], "decode_tps_133k": chosen["tok_s_long_doc"],
                      "vram_served": chosen["vram_served"], "prefill_tps_long": 275 if q != "IQ1_S" else 0},
        "options": options, "notes": notes,
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
        if not os.path.exists(SERVER_BIN): return {"error": f"{SERVER_BIN} not found; build first (cmake --build build)"}
        argv = [SERVER_BIN, model["path"], "--ram", str(int(s["ram"])), "--ctx", str(int(s["ctx"])), "--batch", str(int(s["batch"])),
                "--kv", s["kv"], "--reserve", str(int(s["reserve"])), "--think", s["think"], "--think-budget", str(int(s["think_budget"])),
                "--port", str(int(s["port"]))]
        if s.get("skip_miss"): argv.append("--skip-miss")
        if s.get("cold_path"): argv += ["--cold", s["cold_path"]]
        if s.get("vision"):
            if not model.get("mmproj"):
                return {"error": "no mmproj-*.gguf next to this model: download mmproj-F16.gguf into its snapshot directory, or turn Vision off"}
            argv += ["--mmproj", model["mmproj"]]
        log = open(STATE["log"], "w")
        log.write("$ " + " ".join(argv) + "\n"); log.flush()
        try:
            proc = subprocess.Popen(argv, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        except Exception as e:
            return {"error": f"cannot start: {e}"}
        STATE.update(proc=proc, model=model, settings=s, started=time.time(), port=int(s["port"]))
        return {"ok": True, "pid": proc.pid, "argv": argv}

def stop_server():
    with LOCK:
        p = STATE["proc"]
        if not p or p.poll() is not None:
            return {"ok": True, "note": "no console-started server running"}
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
    return st

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
        if path in ("/", "/index.html"):
            try: body = open(INDEX, "rb").read()
            except Exception: body = b"<h1>tools/console/index.html missing</h1>"
            self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8"); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body); return
        if path == "/api/models": return self._json({"models": scan_models(), "hf": HF})
        if path == "/api/hardware": return self._json(hardware())
        if path == "/api/recommend":
            q = dict(x.split("=", 1) for x in self.path.split("?", 1)[1].split("&")) if "?" in self.path else {}
            models = scan_models(); i = int(q.get("model", 0))
            if not models: return self._json({"error": "no models"}, 404)
            v = q.get("vision", ""); vision = None if v == "" else v in ("1", "true")
            return self._json(recommend(models[min(i, len(models) - 1)], hardware(), q.get("priority", "balanced"), vision))
        if path == "/api/status": return self._json(status())
        if path == "/api/log": return self._json({"log": log_tail(400)})
        if path == "/api/hold":      # 1x1 GIF, answered when /api/release?token= arrives (or after 10 min)
            q = dict(x.split("=", 1) for x in self.path.split("?", 1)[1].split("&")) if "?" in self.path else {}
            ev = HOLDS.setdefault(q.get("token", ""), threading.Event()); ev.wait(600)
            gif = b"GIF89a\x01\x00\x01\x00\x80\x00\x00\x00\x00\x00\xff\xff\xff!\xf9\x04\x01\x00\x00\x00\x00,\x00\x00\x00\x00\x01\x00\x01\x00\x00\x02\x02D\x01\x00;"
            self.send_response(200); self.send_header("Content-Type", "image/gif"); self.send_header("Content-Length", str(len(gif))); self.end_headers(); self.wfile.write(gif); return
        if path == "/api/release":
            q = dict(x.split("=", 1) for x in self.path.split("?", 1)[1].split("&")) if "?" in self.path else {}
            HOLDS.setdefault(q.get("token", ""), threading.Event()).set(); return self._json({"ok": True})
        self._json({"error": "not found"}, 404)
    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0)); body = json.loads(self.rfile.read(n) or b"{}")
        if self.path == "/api/start":
            models = scan_models(); i = int(body.get("model", 0))
            if not models: return self._json({"error": "no models"}, 404)
            s = body.get("settings") or recommend(models[i], hardware())
            return self._json(start_server(models[i], s))
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
