#!/usr/bin/env python3
"""Write a tiny, complete qwen4exp GGUF with random weights: every tensor the architecture has
(hyper-connections, Gated DeltaNet, sparse attention with its indexer, the PLE n-gram layer,
routed and shared experts), at a few hundred thousand parameters, so the whole forward pass runs
in a fraction of a second on a CPU with no GPU and no 111 GB download.

Both readers load it: the engine (model_index) and llama.cpp's own qwen4exp loader (so
qwfn-refdump gives a reference to compare the engine's logits with). The shapes follow
llama.cpp's load_arch_tensors for qwen4exp, scaled down; the metadata types follow what its
loader accepts. The tokenizer is "none" with a vocab_size, plus a token list the engine counts.

usage: make_tiny_model.py out.gguf [--seed N] [--align N]
       --align 4096 places every tensor on a page boundary, which gives the engine's direct-I/O
       path its page layout; the default (32, GGUF's own) gives it the 512-byte bounce layout.
No dependencies beyond the standard library; the same seed gives the same file on any platform."""

import argparse, math, random, struct, sys
from array import array

# -- shapes ------------------------------------------------------------------------------------
V        = 512                 # vocab
E        = 64                  # n_embd
L        = 8                   # layers: 3 and 7 sparse attention (interval 4), the rest Gated DeltaNet; an attention
                               # layer followed by others makes its output reach the logits
INTERVAL = 4
H, HKV, DH = 4, 2, 32          # attention heads, kv heads, head dim
ROPE_DIM, SECTIONS = 16, [3, 3, 2, 0]          # imrope: sections sum to ROPE_DIM / 2
IDX_H, IDX_D, IDX_TOPK, RATIO = 2, 32, 64, 4   # indexer heads, key length, top-k cells, compress ratio
NX, NX_USED, FF_EXP, FF_SH = 32, 4, 32, 32     # routed experts (the engine ranks 16 candidates: at least 16), used per token, their ffn, shared ffn
D_CONV, D_STATE, N_GROUP, DT_RANK = 4, 16, 2, 4  # DeltaNet: conv, head dim, k heads, v heads
HC, HC_LR = 4, 8               # hyper-connections, their low rank
PLE_LAYER, PLE_NGRAM, PLE_HPN, PLE_CONV = 1, 3, 2, 4
PLE_HEADS = (PLE_NGRAM - 1) * PLE_HPN          # 4 heads x 16 dims = E
D_PLE = E // PLE_HEADS
PLE_VOCAB = [97, 101, 103, 107]                # primes, as in the real table
PLE_MULT = [2654435761, 40503, 2246822519]
PLE_EOS = V - 2

KEY_DIM, VALUE_DIM = D_STATE * N_GROUP, D_STATE * DT_RANK
CONV_DIM = 2 * KEY_DIM + VALUE_DIM
HC_DIM = HC * E

def is_attn(il): return (il + 1) % INTERVAL == 0

# -- GGUF writer (version 3) ------------------------------------------------------------------
T_U32, T_I32, T_F32, T_STR, T_ARR, T_U64 = 4, 5, 6, 8, 9, 10
GGML_F32 = 0

def s(x):
    b = x.encode()
    return struct.pack("<Q", len(b)) + b

def kv(key, typ, val):
    out = s(key) + struct.pack("<I", typ)
    if typ == T_U32: return out + struct.pack("<I", val)
    if typ == T_I32: return out + struct.pack("<i", val)
    if typ == T_F32: return out + struct.pack("<f", val)
    if typ == T_STR: return out + s(val)
    raise ValueError(typ)

def kv_arr(key, elem, vals):
    out = s(key) + struct.pack("<IIQ", T_ARR, elem, len(vals))
    fmt = {T_I32: "<i", T_U32: "<I", T_U64: "<Q"}.get(elem)
    return out + b"".join(s(v) if elem == T_STR else struct.pack(fmt, v) for v in vals)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--align", type=int, default=32)
    a = ap.parse_args()
    rng = random.Random(a.seed)

    A = "qwen4exp."
    meta = [
        kv("general.architecture", T_STR, "qwen4exp"),
        kv("general.name", T_STR, "qwfn tiny test model"),
        kv(A + "context_length", T_U32, 4096),
        kv(A + "embedding_length", T_U32, E),
        kv(A + "block_count", T_U32, L),
        kv(A + "vocab_size", T_U32, V),
        kv(A + "attention.head_count", T_U32, H),
        kv(A + "attention.head_count_kv", T_U32, HKV),
        kv(A + "attention.key_length", T_U32, DH),
        kv(A + "attention.value_length", T_U32, DH),
        kv(A + "attention.layer_norm_rms_epsilon", T_F32, 1e-6),
        kv(A + "rope.freq_base", T_F32, 10000000.0),
        kv(A + "rope.dimension_count", T_U32, ROPE_DIM),
        kv_arr(A + "rope.dimension_sections", T_I32, SECTIONS),
        kv(A + "expert_count", T_U32, NX),
        kv(A + "expert_used_count", T_U32, NX_USED),
        kv(A + "expert_feed_forward_length", T_U32, FF_EXP),
        kv(A + "expert_shared_feed_forward_length", T_U32, FF_SH),
        kv(A + "ssm.conv_kernel", T_U32, D_CONV),
        kv(A + "ssm.inner_size", T_U32, VALUE_DIM),
        kv(A + "ssm.state_size", T_U32, D_STATE),
        kv(A + "ssm.time_step_rank", T_U32, DT_RANK),
        kv(A + "ssm.group_count", T_U32, N_GROUP),
        kv(A + "hyper_connection.count", T_U32, HC),
        kv(A + "hyper_connection.low_rank", T_U32, HC_LR),
        kv(A + "attention.indexer.head_count", T_U32, IDX_H),
        kv(A + "attention.indexer.key_length", T_U32, IDX_D),
        kv(A + "attention.indexer.top_k", T_U32, IDX_TOPK),
        kv_arr(A + "attention.compress_ratios", T_I32, [RATIO if is_attn(il) else 0 for il in range(L)]),
        kv(A + "full_attention_interval", T_U32, INTERVAL),
        kv_arr(A + "ple.layers", T_I32, [PLE_LAYER]),
        kv(A + "ple.ngram_size", T_U32, PLE_NGRAM),
        kv(A + "ple.heads_per_ngram", T_U32, PLE_HPN),
        kv(A + "ple.conv_kernel", T_U32, PLE_CONV),
        kv(A + "embedding_length_per_layer_input", T_U32, D_PLE),
        kv_arr(A + "ple.layer_multipliers", T_U64, PLE_MULT),
        kv_arr(A + "ple.head_offsets", T_U64, [sum(PLE_VOCAB[:h]) for h in range(PLE_HEADS)]),
        kv_arr(A + "ple.head_vocab_sizes", T_U64, PLE_VOCAB),
        kv(A + "ple.eos_token_id", T_U32, PLE_EOS),
        kv("tokenizer.ggml.model", T_STR, "none"),
        kv_arr("tokenizer.ggml.tokens", T_STR, ["<t%d>" % i for i in range(V)]),
    ]
    if a.align != 32: meta.append(kv("general.alignment", T_U32, a.align))

    # -- tensors: (name, ne, init) in llama.cpp's ne order ------------------------------------
    tensors = []
    def t(name, ne, init="w"): tensors.append((name, list(ne), init))
    t("token_embd.weight", [E, V], "embd")
    t("output.weight", [E, V])
    t("output_hc_norm.weight", [E, HC], "norm")
    t("output_hc_down.weight", [HC_DIM, HC_LR])
    t("output_hc_up.weight", [HC_LR, HC_DIM])
    t("per_layer_token_embd.weight", [D_PLE, sum(PLE_VOCAB)], "embd")
    for il in range(L):
        b = "blk.%d." % il
        for m in ("attn", "ffn"):
            t(b + "hc_%s_norm.weight" % m, [E, HC], "norm")
            t(b + "hc_%s_down.weight" % m, [HC_DIM, HC_LR])
            t(b + "hc_%s_up.weight" % m, [HC_LR, HC_DIM])
            t(b + "hc_%s_inject.weight" % m, [HC_DIM, HC])
        if is_attn(il):
            t(b + "attn_q.weight", [E, DH * H * 2])
            t(b + "attn_k.weight", [E, DH * HKV])
            t(b + "attn_v.weight", [E, DH * HKV])
            t(b + "attn_output.weight", [DH * H, E])
            t(b + "attn_q_norm.weight", [DH], "norm")
            t(b + "attn_k_norm.weight", [DH], "norm")
            t(b + "indexer.q_proj.weight", [E, IDX_H * IDX_D])
            t(b + "indexer.k_proj.weight", [E, IDX_D])
            t(b + "indexer.q_norm.weight", [IDX_D], "norm")
            t(b + "indexer.k_norm.weight", [IDX_D], "norm")
        else:
            t(b + "attn_qkv.weight", [E, CONV_DIM])
            t(b + "attn_gate.weight", [E, VALUE_DIM])
            t(b + "ssm_conv1d.weight", [D_CONV, CONV_DIM])
            t(b + "ssm_dt.bias", [DT_RANK], "dt")
            t(b + "ssm_a", [DT_RANK], "a")
            t(b + "ssm_beta.weight", [E, DT_RANK])
            t(b + "ssm_alpha.weight", [E, DT_RANK])
            t(b + "ssm_norm.weight", [D_STATE], "norm")
            t(b + "ssm_out.weight", [VALUE_DIM, E])
        if il == PLE_LAYER:
            t(b + "ple_key.weight", [E, HC_DIM])
            t(b + "ple_value.weight", [E, E])
            for n in ("key", "query", "conv"): t(b + "ple_norm_%s.weight" % n, [E, HC], "norm")
            t(b + "ple_conv1d.weight", [PLE_CONV, HC_DIM])
        t(b + "ffn_gate_inp.weight", [E, NX])
        t(b + "ffn_gate_exps.weight", [E, FF_EXP, NX])
        t(b + "ffn_up_exps.weight", [E, FF_EXP, NX])
        t(b + "ffn_down_exps.weight", [FF_EXP, E, NX])
        t(b + "ffn_gate_inp_shexp.weight", [E])
        t(b + "ffn_gate_shexp.weight", [E, FF_SH])
        t(b + "ffn_up_shexp.weight", [E, FF_SH])
        t(b + "ffn_down_shexp.weight", [FF_SH, E])

    def values(ne, init):
        n = 1
        for d in ne: n *= d
        if init == "norm": return array("f", (1.0 + 0.1 * rng.gauss(0, 1) for _ in range(n)))
        if init == "a":    return array("f", (-rng.uniform(0.2, 1.5) for _ in range(n)))   # decays: A < 0
        if init == "dt":   return array("f", (rng.uniform(0.0, 1.0) for _ in range(n)))
        if init == "embd": return array("f", (rng.gauss(0, 0.5) for _ in range(n)))
        fan_in = ne[0]
        sd = 1.0 / math.sqrt(fan_in)
        return array("f", (rng.gauss(0, sd) for _ in range(n)))

    def pad(n): return (a.align - n % a.align) % a.align
    blobs, infos, off = [], b"", 0
    for name, ne, init in tensors:
        data = values(ne, init)
        if sys.byteorder != "little": data.byteswap()
        raw = data.tobytes()
        infos += s(name) + struct.pack("<I", len(ne)) + b"".join(struct.pack("<Q", d) for d in ne) + struct.pack("<IQ", GGML_F32, off)
        blobs.append(raw + b"\0" * pad(len(raw)))
        off += len(raw) + pad(len(raw))
    head = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(meta)) + b"".join(meta) + infos
    with open(a.out, "wb") as f:
        f.write(head + b"\0" * pad(len(head)))
        for blob in blobs: f.write(blob)
    n_params = sum(math.prod(ne) for _, ne, _ in tensors)
    print("wrote %s: %d tensors, %.2f M parameters, alignment %d" % (a.out, len(tensors), n_params / 1e6, a.align))

if __name__ == "__main__":
    main()
