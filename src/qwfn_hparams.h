#pragma once
// Hyper-parameters for Qwen3.8-Flash-Next, GGUF architecture string "qwen4exp".
//
// Every field here is read from the GGUF metadata rather than hardcoded; the
// comments record the values observed in unsloth/Qwen3.8-Flash-Next-GGUF
// UD-Q3_K_XL so that a mismatch is obvious during bring-up.

#include <cstdint>
#include <string>
#include <vector>

namespace qwfn {

struct hparams {
    // -- backbone ---------------------------------------------------------
    uint32_t n_layer      = 0;   // 48
    uint32_t n_embd       = 0;   // 2560
    uint32_t n_ctx_train  = 0;   // 262144
    uint32_t n_vocab      = 0;   // 248320

    // -- attention (only on layers where is_attn_layer() is true) ---------
    uint32_t n_head       = 0;   // 24
    uint32_t n_head_kv    = 0;   // 2
    uint32_t n_embd_head_k = 0;  // 256
    uint32_t n_embd_head_v = 0;  // 256
    float    rms_eps      = 1e-6f;

    // A layer is a full (sparse) attention layer when
    // (il + 1) % full_attention_interval == 0; all others are Gated DeltaNet.
    uint32_t full_attention_interval = 4;              // 12 attn / 36 deltanet
    std::vector<int32_t> compress_ratios;              // per-layer, 0 or 4

    // -- rope / mrope -----------------------------------------------------
    float    rope_freq_base = 10000000.0f;
    uint32_t rope_dim       = 64;
    int32_t  mrope_sections[4] = {11, 11, 10, 0};      // t, h, w, (unused)

    // -- MoE --------------------------------------------------------------
    uint32_t n_expert       = 0;  // 512
    uint32_t n_expert_used  = 0;  // 10
    uint32_t n_ff_exp       = 0;  // 640
    uint32_t n_ff_shexp     = 0;  // 640

    // -- Gated DeltaNet ---------------------------------------------------
    uint32_t ssm_d_conv   = 4;
    uint32_t ssm_d_state  = 128;
    uint32_t ssm_n_group  = 16;
    uint32_t ssm_dt_rank  = 48;
    uint32_t ssm_d_inner  = 6144;   // qkv proj is 2048(q) + 2048(k) + 6144(v) = 10240

    // -- Qwen Sparse Attention lightning indexer --------------------------
    uint32_t idx_n_head   = 4;
    uint32_t idx_key_len  = 128;
    uint32_t idx_top_k    = 2048;

    // -- hyper-connections ------------------------------------------------
    // The residual stream carries hc_count parallel copies: width = hc_count*n_embd.
    uint32_t hc_count     = 4;      // -> residual width 10240
    uint32_t hc_low_rank  = 320;

    // -- PLE (per-layer / n-gram embeddings) ------------------------------
    // per_layer_token_embd.weight is [d_ple, sum(ple_head_vocab_sizes)] IQ4_NL.
    // One row is d_ple values; a token draws ple_n_head rows and concatenates
    // them, giving ple_n_head*d_ple == n_embd.
    std::vector<int32_t>  ple_layers;          // {1}
    uint32_t              ple_ngram_size = 3;
    uint32_t              ple_heads_per_ngram = 8;
    uint32_t              ple_conv_kernel = 4;
    uint32_t              d_ple = 160;         // embedding_length_per_layer_input
    std::vector<uint64_t> ple_head_offsets;    // 16 entries, cumulative row offsets
    std::vector<uint64_t> ple_head_vocab_sizes;// 16 primes, ~20,000,0xx each
    std::vector<uint64_t> ple_layer_multipliers; // 3 hash multipliers (ngram_size)
    int32_t               ple_eos_token_id = -1;  // 248044 -- NOT the tokenizer eos (248046)

    // The table is addressed by (ngram_size - 1) orders x heads_per_ngram heads:
    // for this checkpoint, bigram heads 0..7 and trigram heads 8..15.
    uint32_t ple_n_head() const { return (ple_ngram_size - 1) * ple_heads_per_ngram; } // 16

    // -- tokens -----------------------------------------------------------
    int32_t tok_bos = -1, tok_eos = -1, tok_pad = -1, tok_image = -1;

    bool is_attn_layer(uint32_t il) const {
        return full_attention_interval > 0 && ((il + 1) % full_attention_interval) == 0;
    }
    uint32_t n_residual() const { return hc_count * n_embd; }   // 10240

    std::string summary() const;
};

} // namespace qwfn
