#!/usr/bin/env python3
"""Train the learned routing predictor from an engine dump (QWFN_ROUTE_DUMP).

Per layer L >= 1 the dump holds, for sampled prefill tokens, the router input the
decode-time predictor sees (hc_mix_ffn of the residual entering L) and L's true
routing (top-k ids and gates). The head is a linear map n_embd -> n_expert,
initialised from L's own router matrix (router_L{L}.bin) and fine-tuned with a
soft cross-entropy toward the true gates plus an L2 pull toward the init, so it
can only improve on "the router on the wrong residual". numpy only.

    python3 scripts/train_predictor.py DUMP_DIR OUT.bin [--epochs 8] [--lr 3e-4]
        [--l2 1e-4] [--val 0.1] [--batch 4096] [--layers 1-47] [--n-embd 2560]
        [--n-expert 512] [--n-used 10] [--n-layer 48] [--max-samples N]

Output: OUT.bin = "QWPR", u32 version=1, n_layer, n_embd, n_expert, first_layer,
then per layer first_layer..n_layer-1: F16 W[n_expert][n_embd], F32 b[n_expert]
(layers without data keep their router matrix, bias 0). The metric printed is
recall@k on the held-out split: the share of the true top-k found in the
predicted top-k, which is what the engine's "predicted correctly" reports.
"""
import argparse, os, struct, sys, time
import numpy as np

def parse_layers(s, n_layer):
    if "-" in s:
        a, b = s.split("-"); return list(range(int(a), int(b) + 1))
    return [int(x) for x in s.split(",")]

def load_layer(path, n_embd, n_used, max_samples):
    rec = np.dtype([("x", np.float16, n_embd), ("ids", np.uint16, n_used), ("w", np.float16, n_used)])
    n = os.path.getsize(path) // rec.itemsize
    if n == 0: return None
    arr = np.fromfile(path, dtype=rec, count=n)
    if max_samples and n > max_samples:
        arr = arr[np.random.default_rng(0).choice(n, max_samples, replace=False)]
    x = arr["x"].astype(np.float32)
    ids = arr["ids"].astype(np.int64)
    w = arr["w"].astype(np.float32)
    w /= np.maximum(w.sum(axis=1, keepdims=True), 1e-9)
    return x, ids, w

def recall_at_k(S, ids, k):
    top = np.argpartition(-S, k - 1, axis=1)[:, :k]
    hits = 0
    for t, i in zip(top, ids):
        hits += len(np.intersect1d(t, i, assume_unique=False))
    return hits / (len(ids) * ids.shape[1])

def log_softmax(S):
    m = S.max(axis=1, keepdims=True)
    z = S - m
    return z - np.log(np.exp(z).sum(axis=1, keepdims=True))

def train_layer(x, ids, w, W0, args):
    n, d = x.shape; nx = W0.shape[0]; k = ids.shape[1]
    rng = np.random.default_rng(1)
    perm = rng.permutation(n); nv = max(1, int(n * args.val))
    vi, ti = perm[:nv], perm[nv:]
    xv, idv = x[vi], ids[vi]
    base = recall_at_k(xv @ W0.T, idv, k)
    W = W0.copy(); b = np.zeros(nx, np.float32)
    mW = np.zeros_like(W); vW = np.zeros_like(W); mb = np.zeros_like(b); vb = np.zeros_like(b)
    b1, b2, eps = 0.9, 0.999, 1e-8; step = 0
    best = (base, W0.copy(), b.copy()); bs = args.batch
    for ep in range(args.epochs):
        rng.shuffle(ti)
        for s in range(0, len(ti), bs):
            bi = ti[s:s + bs]; xb = x[bi]; idb = ids[bi]; wb = w[bi]
            S = xb @ W.T + b                                  # [B, nx]
            P = np.exp(log_softmax(S))                        # softmax
            T = np.zeros_like(S); np.put_along_axis(T, idb, wb, axis=1)
            dS = (P - T) / len(bi)                            # d(soft CE)/dS
            gW = dS.T @ xb + 2.0 * args.l2 * (W - W0)
            gb = dS.sum(axis=0)
            step += 1
            mW = b1 * mW + (1 - b1) * gW; vW = b2 * vW + (1 - b2) * gW * gW
            mb = b1 * mb + (1 - b1) * gb; vb = b2 * vb + (1 - b2) * gb * gb
            lr = args.lr * np.sqrt(1 - b2 ** step) / (1 - b1 ** step)
            W -= lr * mW / (np.sqrt(vW) + eps); b -= lr * mb / (np.sqrt(vb) + eps)
        r = recall_at_k(xv @ W.T + b, idv, k)
        if r > best[0]: best = (r, W.copy(), b.copy())
    return base, best

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump"); ap.add_argument("out")
    ap.add_argument("--epochs", type=int, default=8); ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--l2", type=float, default=1e-4); ap.add_argument("--val", type=float, default=0.1)
    ap.add_argument("--batch", type=int, default=4096); ap.add_argument("--layers", default="1-47")
    ap.add_argument("--n-embd", type=int, default=2560); ap.add_argument("--n-expert", type=int, default=512)
    ap.add_argument("--n-used", type=int, default=10); ap.add_argument("--n-layer", type=int, default=48)
    ap.add_argument("--max-samples", type=int, default=0)
    args = ap.parse_args()
    layers = parse_layers(args.layers, args.n_layer)
    first = min(layers)
    heads = {}
    tot_base = tot_best = 0.0; n_done = 0
    for il in range(first, args.n_layer):
        rp = os.path.join(args.dump, f"router_L{il}.bin")
        if not os.path.exists(rp):
            print(f"layer {il:2d}: no router matrix, skipped", flush=True); continue
        W0 = np.fromfile(rp, dtype=np.float32).reshape(args.n_expert, args.n_embd)
        dp = os.path.join(args.dump, f"L{il}.bin")
        data = load_layer(dp, args.n_embd, args.n_used, args.max_samples) if (il in layers and os.path.exists(dp)) else None
        if data is None:
            heads[il] = (W0, np.zeros(args.n_expert, np.float32))
            print(f"layer {il:2d}: no data, router kept", flush=True); continue
        x, ids, w = data
        t0 = time.time()
        base, (r, W, b) = train_layer(x, ids, w, W0, args)
        heads[il] = (W, b)
        tot_base += base; tot_best += r; n_done += 1
        print(f"layer {il:2d}: {len(x):7d} samples  recall@{args.n_used} router {100*base:5.1f}%  ->  trained {100*r:5.1f}%   ({time.time()-t0:.0f} s)", flush=True)
    if n_done:
        print(f"mean over {n_done} layers: router {100*tot_base/n_done:.1f}%  ->  trained {100*tot_best/n_done:.1f}%")
    with open(args.out, "wb") as f:
        f.write(b"QWPR"); f.write(struct.pack("<IIIII", 1, args.n_layer, args.n_embd, args.n_expert, first))
        for il in range(first, args.n_layer):
            W, b = heads.get(il, (np.fromfile(os.path.join(args.dump, f"router_L{il}.bin"), dtype=np.float32).reshape(args.n_expert, args.n_embd), np.zeros(args.n_expert, np.float32)))
            f.write(W.astype(np.float16).tobytes()); f.write(b.astype(np.float32).tobytes())
    print("wrote", args.out, os.path.getsize(args.out) >> 20, "MB")

if __name__ == "__main__":
    main()
