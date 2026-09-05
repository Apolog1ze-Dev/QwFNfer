#!/usr/bin/env python3
"""Derive the previous-token dataset from a DECODE routing dump.

Record i of the derived dump pairs layer L's predictor input at token i-1 with
layer L's true routing at token i: what a whole-token-ahead prefetch could know
at the start of a token. Also prints the trivial baseline -- the share of token
i's experts that were token i-1's -- which the cache already exploits.

    python3 scripts/route_prev_token.py DUMP_DIR OUT_DIR [--n-embd 2560] [--n-used 10]
"""
import os, sys, shutil, struct
import numpy as np

def main():
    src, dst = sys.argv[1], sys.argv[2]
    n_embd = 2560; n_used = 10
    for i, a in enumerate(sys.argv):
        if a == "--n-embd": n_embd = int(sys.argv[i + 1])
        if a == "--n-used": n_used = int(sys.argv[i + 1])
    os.makedirs(dst, exist_ok=True)
    rec = n_embd * 2 + n_used * 4
    xb = n_embd * 2
    tot_same = 0.0; n_layers = 0
    for il in range(1, 200):
        rp = os.path.join(src, f"router_L{il}.bin"); lp = os.path.join(src, f"L{il}.bin")
        if not os.path.exists(lp): break
        shutil.copyfile(rp, os.path.join(dst, f"router_L{il}.bin"))
        data = np.fromfile(lp, dtype=np.uint8)
        n = len(data) // rec
        arr = data[: n * rec].reshape(n, rec)
        out = np.empty((n - 1, rec), dtype=np.uint8)
        out[:, :xb] = arr[:-1, :xb]          # input from token i-1
        out[:, xb:] = arr[1:, xb:]           # routing of token i
        out.tofile(os.path.join(dst, f"L{il}.bin"))
        ids = arr[:, xb:xb + n_used * 2].copy().view(np.uint16).reshape(n, n_used)
        same = np.mean([len(np.intersect1d(ids[i - 1], ids[i])) / n_used for i in range(1, n)])
        tot_same += same; n_layers += 1
        print(f"layer {il:2d}: {n - 1:6d} pairs, same experts as previous token: {100 * same:5.1f}%", flush=True)
    print(f"mean over {n_layers} layers: previous token's routing predicts {100 * tot_same / n_layers:.1f}% of the current one")

if __name__ == "__main__":
    main()
