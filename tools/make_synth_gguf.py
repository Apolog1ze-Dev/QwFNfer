#!/usr/bin/env python3
"""Write a synthetic qwen4exp GGUF for testing the Windows port without the
111 GB checkpoint. Minimal valid metadata the engine reads (see
src/qwfn_model.cpp read_metadata), a few expert tensors with recognizable
payload bytes, and the PLE tensor."""

import os
import struct
import sys

GGUF_MAGIC = b"GGUF"
VERSION = 3
T = {
    "u8": 0,
    "i8": 1,
    "u16": 2,
    "i16": 3,
    "u32": 4,
    "i32": 5,
    "f32": 6,
    "bool": 7,
    "str": 8,
    "arr": 9,
    "u64": 10,
    "i64": 11,
    "f64": 12,
}


def gguf_str(s):
    b = s.encode() if isinstance(s, str) else s
    return struct.pack("<Q", len(b)) + b


def v_u32(x):
    return struct.pack("<II", T["u32"], x)


def v_i32(x):
    return struct.pack("<II", T["i32"], x & 0xFFFFFFFF)


def v_f32(x):
    return struct.pack("<If", T["f32"], x)


def v_str(x):
    return struct.pack("<I", T["str"]) + gguf_str(x)


# GGUF array count is u64 per the spec (llama.cpp's reader matches).
def v_arr_u32(vs):
    return struct.pack("<IIQ", T["arr"], T["u32"], len(vs)) + b"".join(
        struct.pack("<I", x) for x in vs
    )


def v_arr_u64(vs):
    return struct.pack("<IIQ", T["arr"], T["u64"], len(vs)) + b"".join(
        struct.pack("<Q", x) for x in vs
    )


def v_arr_str(vs):
    return struct.pack("<IIQ", T["arr"], T["str"], len(vs)) + b"".join(
        gguf_str(x) for x in vs
    )


A = "qwen4exp."
n_layer, n_expert, n_embd = 2, 4, 64
d_ff, n_ff = 32, 32

# key order irrelevant; values as the parser expects
kvs = [
    ("general.architecture", v_str("qwen4exp")),
    (A + "block_count", v_u32(n_layer)),
    (A + "embedding_length", v_u32(n_embd)),
    (A + "context_length", v_u32(4096)),
    (A + "attention.head_count", v_u32(2)),
    (A + "attention.head_count_kv", v_u32(2)),
    (A + "attention.key_length", v_u32(32)),
    (A + "attention.value_length", v_u32(32)),
    (A + "attention.layer_norm_rms_epsilon", v_f32(1e-6)),
    (A + "full_attention_interval", v_u32(4)),
    (A + "attention.compress_ratios", v_arr_u32([0, 4])),
    (A + "rope.freq_base", v_f32(10000000.0)),
    (A + "rope.dimension_count", v_u32(16)),
    (A + "rope.dimension_sections", v_arr_u32([8, 8, 0, 0])),
    (A + "expert_count", v_u32(n_expert)),
    (A + "expert_used_count", v_u32(2)),
    (A + "expert_feed_forward_length", v_u32(n_ff)),
    (A + "expert_shared_feed_forward_length", v_u32(n_ff)),
    (A + "ssm.conv_kernel", v_u32(4)),
    (A + "ssm.state_size", v_u32(16)),
    (A + "ssm.group_count", v_u32(2)),
    (A + "ssm.time_step_rank", v_u32(8)),
    (A + "ssm.inner_size", v_u32(32)),
    (A + "attention.indexer.head_count", v_u32(2)),
    (A + "attention.indexer.key_length", v_u32(8)),
    (A + "attention.indexer.top_k", v_u32(64)),
    (A + "hyper_connection.count", v_u32(4)),
    (A + "hyper_connection.low_rank", v_u32(8)),
    (A + "ple.layers", v_arr_u32([0, 1])),
    (A + "ple.ngram_size", v_u32(3)),
    (A + "ple.heads_per_ngram", v_u32(8)),
    (A + "ple.conv_kernel", v_u32(4)),
    (A + "embedding_length_per_layer_input", v_u32(16)),
    (A + "ple.head_offsets", v_arr_u64([i * 4 for i in range(16)])),
    (A + "ple.head_vocab_sizes", v_arr_u64([4] * 16)),
    (A + "ple.layer_multipliers", v_arr_u64([2654435761, 40503] * 8)),
    (A + "ple.image_token_id", v_i32(-1)),
    (A + "ple.eos_token_id", v_i32(-1)),
    ("tokenizer.ggml.bos_token_id", v_i32(-1)),
    ("tokenizer.ggml.eos_token_id", v_i32(-1)),
    ("tokenizer.ggml.padding_token_id", v_i32(-1)),
    ("tokenizer.ggml.tokens", v_arr_str(["<unk>"])),
]

# Tensors: experts [ne0, ne1, n_expert] f32; one per layer and part; PLE [16, n_rows]
tensors = []  # (name, ggml_type_f32=0, ne[], data_bytes)
slice_bytes = n_ff * d_ff * 4
for il in range(n_layer):
    for part in ("ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"):
        tensors.append(
            (
                f"blk.{il}.{part}.weight",
                0,
                [n_ff, d_ff, n_expert, 1],
                slice_bytes * n_expert,
            )
        )
ple_rows = 16
tensors.append(
    ("per_layer_token_embd.weight", 0, [16, ple_rows, 1, 1], 16 * ple_rows * 4)
)
# a dense tensor so weights::commit has something to place
tensors.append(("blk.0.attn_norm.weight", 0, [n_embd, 1, 1, 1], n_embd * 4))

n_kv = len(kvs)
n_t = len(tensors)


def build(path):
    try:
        with open(path, "wb") as f:
            data_base = _write(f)
    except OSError as e:
        sys.exit(f"cannot write {path}: {e}")
    print(
        f"wrote {path}: {os.path.getsize(path)} bytes, {n_t} tensors, data at {data_base}"
    )


def _write(f):
    """Emit header + KV + tensor dir + payload to the open file; return the absolute data offset."""
    f.write(GGUF_MAGIC)
    f.write(struct.pack("<I", VERSION))
    f.write(struct.pack("<Q", n_t))
    f.write(struct.pack("<Q", n_kv))
    for k, v in kvs:
        f.write(gguf_str(k))
        f.write(v)
    dir_off = f.tell()
    # one dir entry: u64 name-len + name + u32 ndim + 4*u64 ne + u32 type + u64 offset
    hdr = sum(8 + len(n.encode()) + 4 + 32 + 4 + 8 for n, _t, _ne, _nb in tensors)
    data_base = dir_off + hdr
    # align data to 4096 so expert slices keep natural alignment like the real file
    pad_to_4096 = (4096 - (data_base % 4096)) % 4096
    data_base += pad_to_4096
    off = 0
    infos = []
    for name, ty, ne, nbytes in tensors:
        infos.append((name, ty, ne, off))
        off += nbytes
    for name, ty, ne, toff in infos:
        f.write(gguf_str(name))
        f.write(struct.pack("<I", 4))
        for d in ne:
            f.write(struct.pack("<Q", d))
        f.write(struct.pack("<I", ty))
        # GGUF tensor offsets are relative to the start of the tensor data
        f.write(struct.pack("<Q", toff))
    f.write(b"\x00" * pad_to_4096)
    # payloads: byte pattern (i*31+7)&0xff, same generator as qwfn_io_test
    total = off
    buf = bytearray(total)
    for i in range(total):
        buf[i] = (i * 31 + 7) & 0xFF
    f.write(buf)
    return data_base


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else "synth.gguf")
