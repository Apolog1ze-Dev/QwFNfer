"""The engine against llama.cpp on a tiny random qwen4exp model (tools/make_tiny_model.py), with no GPU:

  1. the forward-pass graph (qwfn-logits --qsa) gives llama.cpp's top-10 next-token tokens, logits
     within 0.02, on a 300-token prompt where the sparse attention selects 68 of 300 cells;
  2. the engine itself (qwfn-gen --cpu: the expert cache, direct-I/O reads, the PLE gather, prefill
     and decode) continues that prompt greedily with exactly llama.cpp's 16 tokens, through the
     token-by-token prompt path, the streamed one in several chunks and (Linux) io_uring reads, with the file's tensors
     laid out for the engine's 512-byte bounce path and for its page layout.

usage: tiny_model_check.py <build dir> [work dir]
llama.cpp's reference is qwfn-refdump, built against the same llama.cpp tree."""
import os, random, re, subprocess, sys

build = os.path.abspath(sys.argv[1])
work = os.path.abspath(sys.argv[2] if len(sys.argv) > 2 else ".")
exe = lambda n: os.path.join(build, n + (".exe" if os.name == "nt" else ""))
here = os.path.dirname(os.path.abspath(__file__))
maker = os.path.join(here, "..", "..", "tools", "make_tiny_model.py")

fails = []
def check(ok, what):
    print(("ok    " if ok else "FAIL  ") + what, flush=True)
    if not ok: fails.append(what)

def run(args, timeout=900):
    r = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        print("$ " + " ".join(a if len(a) < 80 else a[:60] + "..." for a in args))
        print(r.stdout[-3000:]); print(r.stderr[-3000:])
    return r

def top10(text):
    return [(int(m.group(1)), float(m.group(2))) for m in re.finditer(r"^\s+\d+\. id\s+(\d+)\s+logit\s+(-?[\d.]+)", text, re.M)][:10]

def generated(text):
    m = re.search(r"^generated:((?: \d+)+)", text, re.M)
    return [int(x) for x in m.group(1).split()] if m else None

# Distinct tokens: a prompt of one repeated token makes every block score of the sparse attention
# tie, and the two implementations break top-k ties in their own order.
rng = random.Random(7)
prompt = ",".join(str(rng.randrange(0, 510)) for _ in range(300))
for align in (32, 4096):
    model = os.path.join(work, "tiny-a%d.gguf" % align)
    run([sys.executable, maker, model, "--align", str(align)]).check_returncode()
    print("\n== %s" % os.path.basename(model), flush=True)

    ref = run([exe("qwfn-refdump"), model, prompt, "--gen", "16", "--filter", "no-intermediates"])
    check(ref.returncode == 0, "llama.cpp loads and runs the tiny model")
    if ref.returncode: continue
    ref_top, ref_gen = top10(ref.stdout), generated(ref.stdout)
    check(len(ref_top) == 10 and ref_gen and len(ref_gen) == 16, "llama.cpp's top-10 logits and 16 greedy tokens: %s" % ref_gen)

    if align == 32:
        ours = run([exe("qwfn-logits"), model, prompt, "--qsa"])
        top = dict(top10(ours.stdout)); want = dict(ref_top)
        # The same ten tokens, each logit within 0.02. Not closer: the engine selects whole blocks
        # of cells (17 blocks, 68 cells here) where llama.cpp selects 64 cells, so the attention
        # differs slightly. The selection bug #12 fixed put a different token in the ten.
        diff = max((abs(top[i] - want[i]) for i in want if i in top), default=float("nan"))
        check(ours.returncode == 0 and set(top) == set(want) and diff < 0.02,
              "the graph's top-10 tokens are llama.cpp's, logits within 0.02 (max diff %.4f)%s" %
              (diff, "" if set(top) == set(want) else "; ours %s, llama.cpp %s" % (sorted(top), sorted(want))))

    paths = [("token-by-token prompt", []),
             ("streamed prompt, 3 chunks of 128", ["--prefill-decode-max", "1", "--batch", "128"])]
    if os.name != "nt": paths.append(("io_uring reads", ["--io-uring"]))   # the thread pool is the default
    for label, extra in paths:
        g = run([exe("qwfn-gen"), model, "--prompt", prompt, "--gen", "16", "--cpu", "--ram", "2", "--ctx", "1024"] + extra)
        if "io_uring_queue_init failed" in g.stdout + g.stderr:   # a container's seccomp, an old kernel
            print("skip  engine, %s: io_uring is not available here" % label, flush=True); continue
        got = generated(g.stdout)
        check(g.returncode == 0 and got == ref_gen, "engine, %s: %s" % (label, got))

print("\n%s (%d failures)" % ("FAILED" if fails else "ALL PASSED", len(fails)))
sys.exit(1 if fails else 0)
