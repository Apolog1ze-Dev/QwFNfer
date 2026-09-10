#!/usr/bin/env python3
# Measure the real O_DIRECT random-read ceiling of the drive holding the GGUF.
# Usage: python3 scripts/nvme_ceiling.py /path/to/shard.gguf
# A flat number across block sizes AND queue depths means a PCIe link ceiling,
# not NAND. Gen4 x4 saturates at ~6.9-7.0 GB/s; Gen5 x4 at ~13-14 GB/s.
import os, sys, time, threading, mmap, random

p = sys.argv[1]
sz = os.path.getsize(p)

def run(nth, bs, per):
    tot = [0] * nth; err = [None] * nth
    rng = random.Random(12345)
    offs = [[rng.randrange(0, (sz - bs) // 4096) * 4096 for _ in range(per)] for _ in range(nth)]
    def w(i):
        try:
            fd = os.open(p, os.O_RDONLY | os.O_DIRECT)
            mv = memoryview(mmap.mmap(-1, bs))   # page-aligned, required by O_DIRECT
            n = 0
            for o in offs[i]:
                n += os.preadv(fd, [mv], o)
            tot[i] = n; os.close(fd)
        except Exception as e:
            err[i] = repr(e)
    ts = [threading.Thread(target=w, args=(i,)) for i in range(nth)]
    t0 = time.time()
    for t in ts: t.start()
    for t in ts: t.join()
    dt = time.time() - t0
    if err[0]: print("  ERR", err[0]); return
    gb = sum(tot) / 1e9
    print(f"  QD~{nth:2d} bs={bs//1024}KiB  {gb:.2f} GB in {dt:.2f}s = {gb/dt:.2f} GB/s")

for bs in (1 << 20, 2176 * 1024, 8 << 20):
    print(f"random O_DIRECT, block {bs//1024} KiB")
    for nth in (1, 4, 8, 16, 32):
        run(nth, bs, max(8, (3 << 30) // bs // nth))
