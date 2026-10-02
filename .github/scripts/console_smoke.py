"""CI smoke test of the console: its page and API, over a model cache holding the synthetic
qwen4exp GGUF that tools/make_synth_gguf.py writes. Usage: console_smoke.py <console port>
(the console already started on it). Exits non-zero on the first broken endpoint's report."""
import json, sys, time, urllib.request

base = "http://127.0.0.1:%d" % int(sys.argv[1])

def get(path, timeout=120):
    with urllib.request.urlopen(base + path, timeout=timeout) as r:
        return r.status, r.read()

for _ in range(90):
    try:
        get("/api/live", 5); break
    except Exception:
        time.sleep(1)
else:
    sys.exit("the console did not come up on " + base)

fails = []
def check(ok, what):
    print(("ok    " if ok else "FAIL  ") + what, flush=True)
    if not ok: fails.append(what)

st, body = get("/")
check(st == 200 and b"<html" in body[:4000].lower(), "the page is served (%d bytes)" % len(body))

m = json.loads(get("/api/models")[1])
names = [x["name"] for x in m["models"]]
check(names == ["UD-Q4_K_XL"], "the model scan finds the synthetic file in the cache: %s (skipped: %s)" % (names, m.get("skipped")))

hw = json.loads(get("/api/hardware")[1])
check(hw.get("ram_total_gb", 0) > 0 and hw.get("ram_available_gb", 0) > 0,
      "memory: %s GB, %s GB available" % (hw.get("ram_total_gb"), hw.get("ram_available_gb")))
check(bool(hw.get("cpu")) and 0 < hw.get("cpu_cores", 0) <= hw.get("cpu_threads", 0),
      "CPU: %r, %s cores, %s threads" % (hw.get("cpu"), hw.get("cpu_cores"), hw.get("cpu_threads")))

rec = json.loads(get("/api/recommend?model=UD-Q4_K_XL&preset=chat")[1])
check("error" not in rec and rec.get("ctx", 0) > 0 and rec.get("threads", 0) > 0,
      "a plan for the chat tier: ctx %s, RAM tier %s GB, %s threads" % (rec.get("ctx"), rec.get("ram"), rec.get("threads")))

status = json.loads(get("/api/status")[1])
check(status.get("engines") == [], "the process scan sees no engine running: %s" % status.get("engines"))

sys.exit(1 if fails else 0)
