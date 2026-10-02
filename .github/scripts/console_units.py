"""Unit checks of the console's log and nvidia-smi parsing, with no server and no GPU:
log_tiers (what the engine built, a disabled VRAM tier read as zero), parse_pcie and link_note
(the PCIe link check of the tune)."""
import importlib.util, os, sys

here = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("qwfn_console", os.path.join(here, "..", "..", "tools", "qwfn_console.py"))
c = importlib.util.module_from_spec(spec); spec.loader.exec_module(c)

fails = []
def check(ok, what):
    print(("ok    " if ok else "FAIL  ") + what)
    if not ok: fails.append(what)

def tiers(lines):
    c.log_tail = lambda n=60: lines
    return c.log_tiers()

t = tiers(["[qwfn] expert VRAM tier: 8.52 GB, 2730 blocks (11.1%)", "[qwfn] expert RAM tier: 15.00 GB arena (pinned), 4800 of 24576 expert blocks (19.5%)"])
check(t == {"vram_tier_gb": 8.52, "vram_tier_blocks": 2730, "ram_tier_gb": 15.0}, "log_tiers: both tiers built %s" % t)
t = tiers(["[qwfn] VRAM tier disabled: no device memory available", "[qwfn] expert RAM tier: 26.96 GB arena (pinned), 12003 of 24576 expert blocks (48.8%)"])
check(t.get("vram_tier_gb") == 0.0 and t.get("vram_tier_disabled") is True, "log_tiers: a disabled VRAM tier reads as zero %s" % t)
t = tiers(["[qwfn] VRAM tier disabled: no device memory available", "[qwfn] expert VRAM tier: 3.00 GB, 900 blocks (3.6%)"])
check(t.get("vram_tier_gb") == 3.0 and "vram_tier_disabled" not in t, "log_tiers: a tier built after a disabled one wins %s" % t)
check(tiers(["nothing"]) == {}, "log_tiers: no tier lines, no tiers")

check(c.parse_pcie("4, 4, 16, 16\n") == {"gen": 4, "gen_max": 4, "width": 16, "width_max": 16}, "parse_pcie: one GPU")
check(c.parse_pcie("1, 5, 16, 16\n4, 4, 8, 8\n")["gen"] == 1, "parse_pcie: the first of two GPUs")
check(c.parse_pcie("") is None and c.parse_pcie("[N/A], 4, 16, 16") is None, "parse_pcie: nothing usable, None")

check(c.link_note({"gen": 4, "gen_max": 4, "width": 16, "width_max": 16}) is None, "link_note: a full link says nothing")
check(c.link_note(None) is None, "link_note: no nvidia-smi says nothing")
n = c.link_note({"gen": 1, "gen_max": 4, "width": 16, "width_max": 16})
check(bool(n) and "Gen1 x16" in n and "Gen4 x16" in n, "link_note: a link stuck at Gen1 is named")
n = c.link_note({"gen": 4, "gen_max": 4, "width": 8, "width_max": 16})
check(bool(n) and "x8" in n, "link_note: a narrow link is named")

sys.exit(1 if fails else 0)
