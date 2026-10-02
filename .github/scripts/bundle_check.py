"""The release zip as a user gets it, on a machine with no GPU: unpacked into a clean directory,
with a PATH that holds no compiler, CUDA toolkit or build tree (Windows) and no LD_LIBRARY_PATH
(Linux), so every library comes from the bundle's own bin/ or the system.

  1. the bundled qwfn-server loads the tiny random qwen4exp model (tools/make_tiny_model.py) on the
     CPU and answers /health;
  2. with qwfn-gen, qwfn-refdump and qwfn-logits from the same portable build copied next to it,
     .github/scripts/tiny_model_check.py runs out of that bin/: the engine on the bundle's ggml
     libraries gives exactly the tokens llama.cpp gives on them.

usage: bundle_check.py <bundle.zip> <portable build dir> <work dir>"""
import json, os, shutil, socket, stat, subprocess, sys, time, urllib.request, zipfile

zip_path, build, work = (os.path.abspath(a) for a in sys.argv[1:4])
here = os.path.dirname(os.path.abspath(__file__))
ext = ".exe" if os.name == "nt" else ""

shutil.rmtree(work, ignore_errors=True)
os.makedirs(work)
with zipfile.ZipFile(zip_path) as z:
    z.extractall(work)
    for info in z.infolist():   # zipfile drops the execute bits; the archive keeps them
        mode = info.external_attr >> 16
        if mode and not info.is_dir():
            os.chmod(os.path.join(work, info.filename), stat.S_IMODE(mode))
(root,) = [os.path.join(work, d) for d in os.listdir(work)]
bin_dir = os.path.join(root, "bin")
print("bundle: %s (%s)" % (root, ", ".join(sorted(os.listdir(bin_dir)))), flush=True)

env = dict(os.environ)
env.pop("LD_LIBRARY_PATH", None)
if os.name == "nt":
    sysroot = os.environ.get("SystemRoot", r"C:\Windows")
    env["PATH"] = os.pathsep.join([os.path.join(sysroot, "System32"), sysroot, os.path.dirname(sys.executable)])

model = os.path.join(work, "tiny.gguf")
subprocess.run([sys.executable, os.path.join(here, "..", "..", "tools", "make_tiny_model.py"), model], check=True)

# 1. the shipped server, on the CPU
with socket.socket() as s:
    s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]
log_path = os.path.join(work, "server.log")
with open(log_path, "wb") as log:
    srv = subprocess.Popen([os.path.join(bin_dir, "qwfn-server" + ext), model, "--cpu", "--ram", "2", "--ctx", "1024",
                            "--host", "127.0.0.1", "--port", str(port)], stdout=log, stderr=subprocess.STDOUT, env=env)
    health, t0 = None, time.time()
    while time.time() - t0 < 180 and srv.poll() is None:
        try:
            with urllib.request.urlopen("http://127.0.0.1:%d/health" % port, timeout=2) as r:
                health = json.loads(r.read())
            if health.get("status") == "ok": break
        except Exception:
            pass
        time.sleep(1)
    alive = srv.poll() is None
    srv.kill(); srv.wait()
with open(log_path, "rb") as f:
    text = f.read().decode("utf-8", "replace")
ok = alive and bool(health) and health.get("status") == "ok"
print(("ok    " if ok else "FAIL  ") + "the bundled qwfn-server loads the tiny model on the CPU and answers /health: %s" % health, flush=True)
if not ok:
    print(text[-4000:]); sys.exit(1)

# 2. the engine against llama.cpp, on the bundle's libraries
for tool in ("qwfn-gen", "qwfn-refdump", "qwfn-logits"):
    shutil.copy2(os.path.join(build, tool + ext), bin_dir)
r = subprocess.run([sys.executable, os.path.join(here, "tiny_model_check.py"), bin_dir, work], env=env)
sys.exit(r.returncode)
