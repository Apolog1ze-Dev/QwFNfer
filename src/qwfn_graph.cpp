#include "qwfn_graph.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>

#include "ggml-backend.h"

namespace qwfn {

ggml_tensor * graph_builder::W(const std::string & name) const {
    ggml_tensor * t = w_->get(name);
    if (!t && alt_) t = alt_->get(name);
    if (!t) fprintf(stderr, "[qwfn] missing weight: %s\n", name.c_str());
    return t;
}

ggml_tensor * graph_builder::Wl(int il, const char * suffix) const {
    return W("blk." + std::to_string(il) + "." + suffix);
}

ggml_tensor * graph_builder::hc_mix(ggml_tensor * x, int il, bool ffn, ggml_tensor ** inject) {
    const int64_t hc      = hp_->hc_count;
    const int64_t n_embd  = hp_->n_embd;
    const int64_t hc_dim  = hc * n_embd;
    const int64_t nt      = x->ne[2];

    ggml_tensor * w_norm;
    ggml_tensor * w_down;
    ggml_tensor * w_up;
    ggml_tensor * w_inject = nullptr;
    if (il < 0) {
        w_norm = W("output_hc_norm.weight");
        w_down = W("output_hc_down.weight");
        w_up   = W("output_hc_up.weight");
    } else {
        const char * p = ffn ? "hc_ffn" : "hc_attn";
        w_norm   = Wl(il, (std::string(p) + "_norm.weight").c_str());
        w_down   = Wl(il, (std::string(p) + "_down.weight").c_str());
        w_up     = Wl(il, (std::string(p) + "_up.weight").c_str());
        w_inject = Wl(il, (std::string(p) + "_inject.weight").c_str());
    }

    // RMSNorm reduces over ne0 = n_embd, i.e. within one stream, but the learned
    // gamma spans all hc*n_embd. The converter folded gamma to (1 + w).
    ggml_tensor * xn = ggml_rms_norm(ctx0, x, hp_->rms_eps);
    xn = ggml_reshape_2d(ctx0, xn, hc_dim, nt);
    xn = ggml_mul(ctx0, xn, w_norm);

    ggml_tensor * lo = ggml_mul_mat(ctx0, w_down, xn);                 // [hc_lr, T]
    lo = ggml_silu(ctx0, ggml_scale(ctx0, lo, 1.0f / (float) hc));
    ggml_tensor * gate = ggml_sigmoid(ctx0, ggml_mul_mat(ctx0, w_up, lo));  // [hc_dim, T]

    ggml_tensor * gated = ggml_mul(ctx0, xn, gate);
    gated = ggml_reshape_3d(ctx0, gated, n_embd, hc, nt);

    // Collapse the streams by their mean.
    ggml_tensor * mixed = ggml_cont(ctx0,
            ggml_view_2d(ctx0, gated, n_embd, nt, ggml_row_size(gated->type, n_embd) * hc, 0));
    for (int64_t c = 1; c < hc; ++c) {
        ggml_tensor * s = ggml_view_2d(ctx0, gated, n_embd, nt,
                ggml_row_size(gated->type, n_embd) * hc,
                ggml_row_size(gated->type, n_embd) * c);
        mixed = ggml_add(ctx0, mixed, s);
    }
    mixed = ggml_scale(ctx0, mixed, 1.0f / (float) hc);

    if (inject) {
        *inject = ggml_mul_mat(ctx0, w_inject, xn);                    // [hc, T]
    }
    return mixed;
}

ggml_tensor * graph_builder::hc_combine(ggml_tensor * residual, ggml_tensor * block_out,
                                        ggml_tensor * inject) {
    const int64_t hc     = hp_->hc_count;
    const int64_t n_embd = hp_->n_embd;
    const int64_t nt     = residual->ne[2];

    ggml_tensor * w = ggml_sigmoid(ctx0, ggml_scale(ctx0, inject, 1.0f / (float) hc));
    w = ggml_scale(ctx0, w, 2.0f);
    w = ggml_reshape_3d(ctx0, w, 1, hc, nt);

    ggml_tensor * b = ggml_reshape_3d(ctx0, block_out, n_embd, 1, nt);
    b = ggml_repeat_4d(ctx0, b, n_embd, hc, nt, 1);

    return ggml_add(ctx0, residual, ggml_mul(ctx0, b, w));
}


ggml_tensor * graph_builder::rms(ggml_tensor * x, ggml_tensor * w) const {
    return ggml_mul(ctx0, ggml_rms_norm(ctx0, x, hp_->rms_eps), w);
}

ggml_tensor * graph_builder::conv_with_history(ggml_tensor * state_row, ggml_tensor * x,
                                               int64_t hist, int64_t channels) {
    // state_row is [hist, channels]; x is [channels, T]. ggml_ssm_conv wants the
    // token axis first, so the history is concatenated ahead of x transposed.
    ggml_tensor * st = ggml_reshape_3d(ctx0, state_row, hist, channels, 1);
    ggml_tensor * padded = ggml_concat(ctx0, st, ggml_transpose(ctx0, x), 0);  // [hist+T, channels, 1]

    // Keep the trailing `hist` positions for the next ubatch.
    ggml_tensor * tail = ggml_view_2d(ctx0, padded, hist, channels,
            padded->nb[1], ggml_row_size(padded->type, padded->ne[0] - hist));
    ggml_build_forward_expand(gf_, ggml_cpy(ctx0, ggml_cont(ctx0, tail), state_row));

    return padded;
}

ggml_tensor * graph_builder::deltanet(ggml_tensor * cur, int il) {
    const int64_t head_k    = hp_->ssm_d_state;     // 128
    const int64_t head_v    = hp_->ssm_d_state;     // 128
    const int64_t n_k_heads = hp_->ssm_n_group;     // 16
    const int64_t n_v_heads = hp_->ssm_dt_rank;     // 48
    const int64_t key_dim   = head_k * n_k_heads;   // 2048
    const int64_t value_dim = head_v * n_v_heads;   // 6144
    const int64_t conv_dim  = key_dim * 2 + value_dim;  // 10240
    const int64_t T         = cur->ne[1];

    ggml_tensor * qkv = ggml_mul_mat(ctx0, Wl(il, "attn_qkv.weight"),  cur);   // [10240, T]
    ggml_tensor * z   = ggml_mul_mat(ctx0, Wl(il, "attn_gate.weight"), cur);   // [ 6144, T]

    ggml_tensor * beta = ggml_mul_mat(ctx0, Wl(il, "ssm_beta.weight"), cur);   // [48, T]
    beta = ggml_sigmoid(ctx0, ggml_reshape_4d(ctx0, beta, 1, n_v_heads, T, 1));

    ggml_tensor * alpha = ggml_mul_mat(ctx0, Wl(il, "ssm_alpha.weight"), cur); // [48, T]
    alpha = ggml_reshape_3d(ctx0, alpha, n_v_heads, T, 1);
    // -A_log.exp() * softplus(alpha + dt_bias); ssm_a already holds the negated exp
    ggml_tensor * g = ggml_softplus(ctx0, ggml_add(ctx0, alpha, Wl(il, "ssm_dt.bias")));
    g = ggml_mul(ctx0, g, Wl(il, "ssm_a"));
    g = ggml_reshape_4d(ctx0, g, 1, n_v_heads, T, 1);

    // Short causal conv over all 10240 channels, then SiLU.
    ggml_tensor * padded  = conv_with_history(st_->rs_conv(il), qkv,
                                              hp_->ssm_d_conv - 1, conv_dim);
    ggml_tensor * conv_out = ggml_silu(ctx0, ggml_ssm_conv(ctx0, padded, Wl(il, "ssm_conv1d.weight")));
    // conv_out is [conv_dim, T, 1]

    const size_t es  = ggml_element_size(conv_out);
    const size_t nb1 = ggml_row_size(conv_out->type, conv_dim);

    ggml_tensor * q = ggml_view_4d(ctx0, conv_out, head_k, n_k_heads, T, 1,
            ggml_row_size(conv_out->type, head_k), nb1, nb1 * T, 0);
    ggml_tensor * k = ggml_view_4d(ctx0, conv_out, head_k, n_k_heads, T, 1,
            ggml_row_size(conv_out->type, head_k), nb1, nb1 * T, key_dim * es);
    ggml_tensor * v = ggml_view_4d(ctx0, conv_out, head_v, n_v_heads, T, 1,
            ggml_row_size(conv_out->type, head_v), nb1, nb1 * T, 2 * key_dim * es);

    q = ggml_l2_norm(ctx0, q, hp_->rms_eps);
    k = ggml_l2_norm(ctx0, k, hp_->rms_eps);

    ggml_tensor * s0 = ggml_reshape_4d(ctx0, st_->rs_state(il), head_v, head_v, n_v_heads, 1);

    // The fused op broadcasts the 16 k-heads across the 48 v-heads itself.
    ggml_tensor * result = ggml_gated_delta_net(ctx0, q, k, v, g, beta, s0, /*K=*/1);

    ggml_tensor * out = ggml_view_4d(ctx0, result, head_v, n_v_heads, T, 1,
            ggml_row_size(result->type, head_v),
            ggml_row_size(result->type, head_v * n_v_heads),
            ggml_row_size(result->type, head_v * n_v_heads * T), 0);

    ggml_tensor * s1 = ggml_view_4d(ctx0, result, head_v, head_v, n_v_heads, 1,
            ggml_row_size(result->type, head_v),
            ggml_row_size(result->type, head_v * head_v),
            ggml_row_size(result->type, head_v * head_v * n_v_heads),
            ggml_row_size(result->type, head_v * n_v_heads * T));
    ggml_build_forward_expand(gf_, ggml_cpy(ctx0, s1, st_->rs_state(il)));

    // Gated RMSNorm; sigmoid gate here, unlike Qwen3.5's GDN which uses silu.
    ggml_tensor * zg = ggml_reshape_4d(ctx0, z, head_v, n_v_heads, T, 1);
    ggml_tensor * o  = ggml_mul(ctx0, rms(out, Wl(il, "ssm_norm.weight")), ggml_sigmoid(ctx0, zg));

    o = ggml_reshape_2d(ctx0, o, value_dim, T);
    return ggml_mul_mat(ctx0, Wl(il, "ssm_out.weight"), o);
}


ggml_tensor * graph_builder::qsa_top_k(ggml_tensor * cur, ggml_tensor * inp_pos,
                                       ggml_tensor * kq_mask, const int sections[4], int il,
                                       const qsa_inputs & qsa) {
    const int64_t idx_dim  = hp_->idx_key_len;    // 128
    const int64_t n_idx_h  = hp_->idx_n_head;     // 4
    const int64_t r        = qsa.ratio;
    const int64_t n_blocks = qsa.n_blocks;
    const int64_t T        = cur->ne[1];
    const int64_t n_kv     = n_past_ + T;

    int secs[4] = { sections[0], sections[1], sections[2], sections[3] };

    // Cached indexer keys are stored RAW: pooling happens before norm and rope,
    // so neither may be applied on the way in.
    ggml_tensor * k_raw = ggml_mul_mat(ctx0, Wl(il, "indexer.k_proj.weight"), cur);  // [128, T]
    ggml_tensor * ic = st_->idx_cache(il);
    ggml_build_forward_expand(gf_, ggml_cpy(ctx0,
            ggml_reshape_2d(ctx0, k_raw, idx_dim, T),
            ggml_view_2d(ctx0, ic, idx_dim, T, ggml_row_size(ic->type, idx_dim),
                         ggml_row_size(ic->type, idx_dim) * n_past_)));

    ggml_tensor * k_all = ggml_view_2d(ctx0, ic, idx_dim, n_kv,
                                       ggml_row_size(ic->type, idx_dim), 0);

    // Mean-pool each block's member keys.
    ggml_tensor * members = ggml_get_rows(ctx0, k_all, qsa.blk_cells);          // [128, r*n_blk]
    members = ggml_reshape_3d(ctx0, members, idx_dim, r, n_blocks);
    ggml_tensor * pooled = nullptr;
    for (int64_t i = 0; i < r; i++) {
        ggml_tensor * slice = ggml_cont(ctx0,
                ggml_view_2d(ctx0, members, idx_dim, n_blocks, members->nb[2], i * members->nb[1]));
        pooled = pooled ? ggml_add(ctx0, pooled, slice) : slice;
    }
    pooled = ggml_scale(ctx0, pooled, 1.0f / (float) r);

    pooled = rms(ggml_reshape_3d(ctx0, pooled, idx_dim, n_blocks, 1), Wl(il, "indexer.k_norm.weight"));
    pooled = ggml_reshape_3d(ctx0, pooled, idx_dim, 1, n_blocks);
    pooled = ggml_rope_multi(ctx0, pooled, qsa.blk_pos, nullptr, hp_->rope_dim, secs,
                             GGML_ROPE_TYPE_IMROPE, hp_->n_ctx_train, hp_->rope_freq_base,
                             1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    pooled = ggml_reshape_2d(ctx0, pooled, idx_dim, n_blocks);

    ggml_tensor * q = ggml_mul_mat(ctx0, Wl(il, "indexer.q_proj.weight"), cur);      // [512, T]
    q = ggml_reshape_3d(ctx0, q, idx_dim, n_idx_h, T);
    q = rms(q, Wl(il, "indexer.q_norm.weight"));
    q = ggml_rope_multi(ctx0, q, inp_pos, nullptr, hp_->rope_dim, secs,
                        GGML_ROPE_TYPE_IMROPE, hp_->n_ctx_train, hp_->rope_freq_base,
                        1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

    // Rectify each head's dot product before summing, as in the DeepSeek lightning indexer.
    ggml_tensor * score = ggml_mul_mat(ctx0, pooled,
            ggml_reshape_2d(ctx0, ggml_cont(ctx0, q), idx_dim, n_idx_h * T));        // [n_blk, 4*T]
    score = ggml_reshape_3d(ctx0, score, n_blocks, n_idx_h, T);
    score = ggml_relu(ctx0, score);
    score = ggml_cont(ctx0, ggml_permute(ctx0, score, 1, 0, 2, 3));                  // [4, n_blk, T]
    score = ggml_sum_rows(ctx0, score);                                              // [1, n_blk, T]
    score = ggml_reshape_2d(ctx0, score, n_blocks, T);

    score = ggml_add(ctx0, score, qsa.bias);

    // Select whole BLOCKS -- ceil(width / r) of them, 513 here -- and expand to
    // their cells. Every cell of a block inherits its block's score, so this is
    // the cell-level top-k up to the one partial block it would cut; it is also
    // what the decode path selects, and it removes the [n_kv, T] F32 score
    // expansion that made the prefill graph's memory scale with n_kv * T. The
    // causal mask, added later, still removes future and empty cells.
    const int64_t width = std::min<int64_t>(n_kv, (int64_t) hp_->idx_top_k + r - 1);
    const int64_t kb    = std::min<int64_t>(n_blocks, (width + r - 1) / r);
    ggml_tensor * top   = ggml_top_k(ctx0, score, kb);                                // [kb, T] I32
    ggml_tensor * table = ggml_reshape_3d(ctx0, qsa.blk_cells, r, n_blocks, 1);       // (k, b) -> b*r + k
    ggml_tensor * cells = ggml_get_rows(ctx0, table, ggml_reshape_1d(ctx0, ggml_cont(ctx0, top), kb * T)); // [r, kb*T]
    return ggml_cont(ctx0, ggml_reshape_2d(ctx0, cells, r * kb, T));                  // [r*kb, T]
}

ggml_tensor * graph_builder::sparse_attn(ggml_tensor * cur, ggml_tensor * inp_pos,
                                         ggml_tensor * kq_mask, const int sections[4], int il,
                                         const qsa_inputs * qsa) {
    const int64_t hd    = hp_->n_embd_head_k;   // 256
    const int64_t nh    = hp_->n_head;          // 24
    const int64_t nh_kv = hp_->n_head_kv;       // 2
    const int64_t T     = cur->ne[1];
    const int64_t n_kv  = n_past_ + T;

    // The indexer reads the same block input as q/k/v; no ratio means dense.
    ggml_tensor * top_k = (qsa && qsa->ratio > 0)
        ? qsa_top_k(cur, inp_pos, kq_mask, sections, il, *qsa) : nullptr;

    // One projection emits query and gate INTERLEAVED PER HEAD: for head h, the
    // query occupies [h*2*hd, h*2*hd+hd) and the gate the next hd.
    ggml_tensor * qg = ggml_mul_mat(ctx0, Wl(il, "attn_q.weight"), cur);   // [2*hd*nh, T]
    const size_t es = ggml_element_size(qg);

    ggml_tensor * Q = ggml_view_3d(ctx0, qg, hd, nh, T, es * hd * 2, es * hd * 2 * nh, 0);
    Q = rms(Q, Wl(il, "attn_q_norm.weight"));

    ggml_tensor * gate = ggml_view_3d(ctx0, qg, hd, nh, T, es * hd * 2, es * hd * 2 * nh, es * hd);
    gate = ggml_cont_2d(ctx0, gate, hd * nh, T);

    ggml_tensor * K = ggml_mul_mat(ctx0, Wl(il, "attn_k.weight"), cur);
    K = ggml_reshape_3d(ctx0, K, hd, nh_kv, T);
    K = rms(K, Wl(il, "attn_k_norm.weight"));

    ggml_tensor * V = ggml_mul_mat(ctx0, Wl(il, "attn_v.weight"), cur);
    V = ggml_reshape_3d(ctx0, V, hd, nh_kv, T);

    // Interleaved multimodal RoPE: sections [11,11,10,0] over t/h/w.
    int secs[4] = { sections[0], sections[1], sections[2], sections[3] };
    Q = ggml_rope_multi(ctx0, Q, inp_pos, nullptr, hp_->rope_dim, secs,
                        GGML_ROPE_TYPE_IMROPE, hp_->n_ctx_train, hp_->rope_freq_base,
                        1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    K = ggml_rope_multi(ctx0, K, inp_pos, nullptr, hp_->rope_dim, secs,
                        GGML_ROPE_TYPE_IMROPE, hp_->n_ctx_train, hp_->rope_freq_base,
                        1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

    // Append this ubatch to the caches.
    ggml_tensor * kc = st_->k_cache(il);
    ggml_tensor * vc = st_->v_cache(il);
    const int64_t kv_dim = hd * nh_kv;
    ggml_build_forward_expand(gf_, ggml_cpy(ctx0,
            ggml_reshape_2d(ctx0, K, kv_dim, T),
            ggml_view_2d(ctx0, kc, kv_dim, T, ggml_row_size(kc->type, kv_dim),
                         ggml_row_size(kc->type, kv_dim) * n_past_)));
    ggml_build_forward_expand(gf_, ggml_cpy(ctx0,
            ggml_reshape_2d(ctx0, V, kv_dim, T),
            ggml_view_2d(ctx0, vc, kv_dim, T, ggml_row_size(vc->type, kv_dim),
                         ggml_row_size(vc->type, kv_dim) * n_past_)));

    // ggml_flash_attn_ext wants [head_dim, n_kv, n_head_kv]. The cache is written
    // token-major (kv_dim contiguous per token), so the token stride is the outer
    // one and the head stride the inner one -- nb2 < nb1, which a view handles.
    ggml_tensor * k_all = ggml_view_3d(ctx0, kc, hd, n_kv, nh_kv,
            ggml_row_size(kc->type, kv_dim), ggml_row_size(kc->type, hd), 0);
    ggml_tensor * v_all = ggml_view_3d(ctx0, vc, hd, n_kv, nh_kv,
            ggml_row_size(vc->type, kv_dim), ggml_row_size(vc->type, hd), 0);

    // Restrict the mask to the cells the indexer chose: start from all -inf,
    // write 0 at the selected indices, then add the causal mask back.
    ggml_tensor * mask = kq_mask;
    if (top_k) {
        // Start from all -inf, write 0 at the chosen cells, then add the causal
        // mask back so future and empty cells stay masked. Built at exactly
        // [n_kv, T]: ggml_set_rows requires a->ne[2] == b->ne[2], and the
        // attention mask may carry padding rows that top_k does not.
        // Built in the mask's own type (F16): half the bytes of the F32 build
        // and no cast, which at n_kv * T = 96M is 400 MB less arena.
        ggml_tensor * base = ggml_fill(ctx0,
                ggml_new_tensor_2d(ctx0, kq_mask->type, n_kv, T), -INFINITY);
        ggml_tensor * a4 = ggml_view_4d(ctx0, base, 1, n_kv, T, 1,
                                        base->nb[0], base->nb[1], base->nb[2], 0);

        ggml_tensor * idx = ggml_view_3d(ctx0, top_k, top_k->ne[0], T, 1,
                                         top_k->nb[1], top_k->nb[1] * T, 0);
        ggml_tensor * zeros = ggml_fill(ctx0,
                ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, 1, top_k->ne[0], T, 1), 0.0f);

        ggml_tensor * sel = ggml_set_rows(ctx0, a4, zeros, idx);
        sel = ggml_view_2d(ctx0, sel, n_kv, T, base->nb[1], 0);

        // In place: the causal mask is added into the selection tensor rather
        // than into a second [n_kv, T] buffer. At the chunk cap that is one
        // 96 MB tensor fewer in the arena that made a 131K prefill abort.
        ggml_tensor * causal = ggml_view_2d(ctx0, kq_mask, n_kv, T, kq_mask->nb[1], 0);
        mask = ggml_add_inplace(ctx0, sel, causal);
    }

    ggml_tensor * q = ggml_permute(ctx0, Q, 0, 2, 1, 3);                 // [hd, T, nh]
    ggml_tensor * out = ggml_flash_attn_ext(ctx0, q, k_all, v_all, mask,
                                            1.0f / sqrtf((float) hd), 0.0f, 0.0f);
    ggml_flash_attn_ext_set_prec(out, GGML_PREC_F32);
    out = ggml_reshape_2d(ctx0, out, hd * nh, T);

    out = ggml_mul(ctx0, out, ggml_sigmoid(ctx0, gate));
    return ggml_mul_mat(ctx0, Wl(il, "attn_output.weight"), out);
}

// Pool r raw indexer keys per block (mean), norm, rope at the block positions.
// members is [idx_dim, r*n_blk] in block-major order; returns [idx_dim, n_blk].
static ggml_tensor * qsa_pool_blocks(ggml_context * ctx0, const hparams * hp, ggml_tensor * members,
                                     int64_t idx_dim, int64_t r, int64_t n_blk, ggml_tensor * k_norm_w,
                                     ggml_tensor * blk_pos, const int secs[4]) {
    members = ggml_reshape_3d(ctx0, members, idx_dim, r, n_blk);
    ggml_tensor * pooled = nullptr;
    for (int64_t i = 0; i < r; i++) {
        ggml_tensor * slice = ggml_cont(ctx0,
                ggml_view_2d(ctx0, members, idx_dim, n_blk, members->nb[2], i * members->nb[1]));
        pooled = pooled ? ggml_add(ctx0, pooled, slice) : slice;
    }
    pooled = ggml_scale(ctx0, pooled, 1.0f / (float) r);
    pooled = ggml_mul(ctx0, ggml_rms_norm(ctx0, ggml_reshape_3d(ctx0, pooled, idx_dim, n_blk, 1), hp->rms_eps), k_norm_w);
    pooled = ggml_reshape_3d(ctx0, pooled, idx_dim, 1, n_blk);
    pooled = ggml_rope_multi(ctx0, pooled, blk_pos, nullptr, hp->rope_dim, (int *) secs,
                             GGML_ROPE_TYPE_IMROPE, hp->n_ctx_train, hp->rope_freq_base,
                             1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    return ggml_reshape_2d(ctx0, pooled, idx_dim, n_blk);
}

void graph_builder::qsa_pool_rebuild(int il, const qsa_decode_inputs & qd, int64_t n_whole, ggml_tensor * blk_pos_all) {
    if (n_whole <= 0) return;
    const int64_t idx_dim = hp_->idx_key_len;
    const int64_t r       = qd.ratio;
    int secs[4] = { hp_->mrope_sections[0], hp_->mrope_sections[1], hp_->mrope_sections[2], hp_->mrope_sections[3] };
    // The caches are declared 1D; address them as [row, n_ctx].
    ggml_tensor * ic = st_->idx_cache(il);
    ic = ggml_reshape_2d(ctx0, ic, idx_dim, ic->ne[0] / idx_dim);
    // blk_cells is [r, n_blk_max] with entry (k, b) = b*r + k, so its first
    // r*n_whole entries are exactly cells 0..r*n_whole-1 in block-major order.
    ggml_tensor * cells = ggml_view_1d(ctx0, qd.blk_cells, r * n_whole, 0);
    ggml_tensor * k_all = ggml_view_2d(ctx0, ic, idx_dim, r * n_whole, ggml_row_size(ic->type, idx_dim), 0);
    ggml_tensor * members = ggml_get_rows(ctx0, k_all, cells);                       // [idx_dim, r*n_whole]
    ggml_tensor * pooled = qsa_pool_blocks(ctx0, hp_, members, idx_dim, r, n_whole,
                                           Wl(il, "indexer.k_norm.weight"), blk_pos_all, secs);
    ggml_build_forward_expand(gf_, ggml_cpy(ctx0, pooled,
            ggml_view_2d(ctx0, qd.pool_cache, idx_dim, n_whole, qd.pool_cache->nb[1], 0)));
}

ggml_tensor * graph_builder::sparse_attn_decode(ggml_tensor * cur, ggml_tensor * inp_pos, const int sections[4],
                                                int il, const qsa_decode_inputs & qd) {
    const int64_t hd      = hp_->n_embd_head_k;   // 256
    const int64_t nh      = hp_->n_head;          // 24
    const int64_t nh_kv   = hp_->n_head_kv;       // 2
    const int64_t kv_dim  = hd * nh_kv;
    const int64_t idx_dim = hp_->idx_key_len;     // 128
    const int64_t n_idx_h = hp_->idx_n_head;      // 4
    const int64_t r       = qd.ratio;
    const int64_t NB      = qd.n_bucket;
    const int64_t KB      = qd.k_blocks;
    const int64_t NC      = KB * r;               // cells attended
    int secs[4] = { sections[0], sections[1], sections[2], sections[3] };

    // ---- indexer: raw key in, this token's block pooled -------------------
    ggml_tensor * k_raw = ggml_mul_mat(ctx0, Wl(il, "indexer.k_proj.weight"), cur);     // [128, 1]
    ggml_tensor * ic    = st_->idx_cache(il);                                          // declared 1D
    ic = ggml_reshape_2d(ctx0, ic, idx_dim, ic->ne[0] / idx_dim);
    ggml_tensor * ic_w  = ggml_set_rows(ctx0, ic, k_raw, qd.write_idx);
    ggml_build_forward_expand(gf_, ic_w);
    // Reading through ic_w orders the gather after the write.
    ggml_tensor * members = ggml_get_rows(ctx0, ic_w, qd.member_idx);                 // [128, r]
    ggml_tensor * pooled  = qsa_pool_blocks(ctx0, hp_, members, idx_dim, r, 1,
                                            Wl(il, "indexer.k_norm.weight"), qd.blk_pos, secs);   // [128, 1]
    ggml_tensor * pc_w = ggml_set_rows(ctx0, qd.pool_cache, pooled, qd.blk_idx);
    ggml_build_forward_expand(gf_, pc_w);

    // ---- scores over the bucket, top blocks -> cells ----------------------
    ggml_tensor * q = ggml_mul_mat(ctx0, Wl(il, "indexer.q_proj.weight"), cur);       // [512, 1]
    q = ggml_reshape_3d(ctx0, q, idx_dim, n_idx_h, 1);
    q = rms(q, Wl(il, "indexer.q_norm.weight"));
    q = ggml_rope_multi(ctx0, q, inp_pos, nullptr, hp_->rope_dim, secs,
                        GGML_ROPE_TYPE_IMROPE, hp_->n_ctx_train, hp_->rope_freq_base,
                        1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    ggml_tensor * pool_v = ggml_view_2d(ctx0, pc_w, idx_dim, NB, pc_w->nb[1], 0);      // after the write
    ggml_tensor * score = ggml_mul_mat(ctx0, pool_v,
            ggml_reshape_2d(ctx0, ggml_cont(ctx0, q), idx_dim, n_idx_h));              // [NB, 4]
    score = ggml_relu(ctx0, score);
    score = ggml_cont(ctx0, ggml_transpose(ctx0, score));                             // [4, NB]
    score = ggml_sum_rows(ctx0, score);                                               // [1, NB]
    score = ggml_reshape_2d(ctx0, score, NB, 1);
    score = ggml_add(ctx0, score, ggml_reshape_2d(ctx0, ggml_view_1d(ctx0, qd.bias, NB, 0), NB, 1));
    ggml_tensor * top = ggml_top_k(ctx0, score, KB);                                   // [KB, 1] I32
    ggml_tensor * cells = ggml_get_rows(ctx0, qd.blk_cells, ggml_reshape_1d(ctx0, top, KB));   // [r, KB] I32
    cells = ggml_reshape_1d(ctx0, cells, NC);

    // ---- cell mask: 0 for cells <= n_past, -inf beyond ---------------------
    ggml_tensor * cp   = ggml_get_rows(ctx0, qd.cell_pos, cells);                     // [1, NC]
    cp = ggml_reshape_2d(ctx0, cp, NC, 1);
    ggml_tensor * dpos = ggml_sub(ctx0, cp, qd.npast_f);                              // cell - n_past
    ggml_tensor * ok   = ggml_step(ctx0, ggml_scale_bias(ctx0, dpos, -1.0f, 0.5f));   // 1 if cell <= n_past
    ggml_tensor * mask = ggml_cast(ctx0, ggml_scale_bias(ctx0, ok, 1e30f, -1e30f), GGML_TYPE_F16);  // 0 / -inf

    // ---- q, k, v; cache writes; gather the selected cells -----------------
    ggml_tensor * qg = ggml_mul_mat(ctx0, Wl(il, "attn_q.weight"), cur);              // [2*hd*nh, 1]
    const size_t es = ggml_element_size(qg);
    ggml_tensor * Q = ggml_view_3d(ctx0, qg, hd, nh, 1, es * hd * 2, es * hd * 2 * nh, 0);
    Q = rms(Q, Wl(il, "attn_q_norm.weight"));
    ggml_tensor * gate = ggml_view_3d(ctx0, qg, hd, nh, 1, es * hd * 2, es * hd * 2 * nh, es * hd);
    gate = ggml_cont_2d(ctx0, gate, hd * nh, 1);

    ggml_tensor * K = ggml_reshape_3d(ctx0, ggml_mul_mat(ctx0, Wl(il, "attn_k.weight"), cur), hd, nh_kv, 1);
    K = rms(K, Wl(il, "attn_k_norm.weight"));
    ggml_tensor * V = ggml_reshape_3d(ctx0, ggml_mul_mat(ctx0, Wl(il, "attn_v.weight"), cur), hd, nh_kv, 1);
    Q = ggml_rope_multi(ctx0, Q, inp_pos, nullptr, hp_->rope_dim, secs, GGML_ROPE_TYPE_IMROPE,
                        hp_->n_ctx_train, hp_->rope_freq_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    K = ggml_rope_multi(ctx0, K, inp_pos, nullptr, hp_->rope_dim, secs, GGML_ROPE_TYPE_IMROPE,
                        hp_->n_ctx_train, hp_->rope_freq_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

    ggml_tensor * kc   = st_->k_cache(il);
    ggml_tensor * vc   = st_->v_cache(il);
    kc = ggml_reshape_2d(ctx0, kc, kv_dim, kc->ne[0] / kv_dim);
    vc = ggml_reshape_2d(ctx0, vc, kv_dim, vc->ne[0] / kv_dim);
    ggml_tensor * kc_w = ggml_set_rows(ctx0, kc, ggml_reshape_2d(ctx0, K, kv_dim, 1), qd.write_idx);
    ggml_tensor * vc_w = ggml_set_rows(ctx0, vc, ggml_reshape_2d(ctx0, V, kv_dim, 1), qd.write_idx);
    ggml_build_forward_expand(gf_, kc_w);
    ggml_build_forward_expand(gf_, vc_w);

    auto gather = [&](ggml_tensor * cache_w) {
        ggml_tensor * g = ggml_get_rows(ctx0, cache_w, cells);                        // F32 [kv_dim, NC]
        g = ggml_cast(ctx0, g, GGML_TYPE_F16);
        g = ggml_reshape_3d(ctx0, g, hd, nh_kv, NC);
        return ggml_cont(ctx0, ggml_permute(ctx0, g, 0, 2, 1, 3));                    // [hd, NC, nh_kv]
    };
    ggml_tensor * Kg = gather(kc_w);
    ggml_tensor * Vg = gather(vc_w);

    ggml_tensor * qp  = ggml_permute(ctx0, Q, 0, 2, 1, 3);                            // [hd, 1, nh]
    ggml_tensor * out = ggml_flash_attn_ext(ctx0, qp, Kg, Vg, mask, 1.0f / sqrtf((float) hd), 0.0f, 0.0f);
    ggml_flash_attn_ext_set_prec(out, GGML_PREC_F32);
    out = ggml_reshape_2d(ctx0, out, hd * nh, 1);
    out = ggml_mul(ctx0, out, ggml_sigmoid(ctx0, gate));
    return ggml_mul_mat(ctx0, Wl(il, "attn_output.weight"), out);
}

ggml_tensor * graph_builder::moe(ggml_tensor * cur, int il) {
    const int64_t n_expert      = hp_->n_expert;
    const int64_t n_expert_used = hp_->n_expert_used;
    const int64_t T             = cur->ne[1];

    // Router: softmax over all 512, take the top 10, renormalise.
    ggml_tensor * logits = ggml_mul_mat(ctx0, Wl(il, "ffn_gate_inp.weight"), cur);  // [512, T]
    ggml_tensor * probs  = ggml_soft_max(ctx0, logits);

    ggml_tensor * selected = ggml_top_k(ctx0, probs, n_expert_used);                // [10, T] I32
    ggml_tensor * weights_ = ggml_get_rows(ctx0,
            ggml_reshape_3d(ctx0, probs, 1, n_expert, T), selected);                // [1, 10, T]
    weights_ = ggml_reshape_2d(ctx0, weights_, n_expert_used, T);
    ggml_tensor * denom = ggml_sum_rows(ctx0, weights_);
    weights_ = ggml_div(ctx0, weights_, denom);
    weights_ = ggml_reshape_3d(ctx0, weights_, 1, n_expert_used, T);

    ggml_tensor * x = ggml_reshape_3d(ctx0, cur, cur->ne[0], 1, T);

    ggml_tensor * up   = ggml_mul_mat_id(ctx0, Wl(il, "ffn_up_exps.weight"),   x, selected);
    ggml_tensor * gate = ggml_mul_mat_id(ctx0, Wl(il, "ffn_gate_exps.weight"), x, selected);
    ggml_tensor * act  = ggml_mul(ctx0, ggml_silu(ctx0, gate), up);
    ggml_tensor * down = ggml_mul_mat_id(ctx0, Wl(il, "ffn_down_exps.weight"), act, selected);

    down = ggml_mul(ctx0, down, weights_);
    // Sum the n_expert_used contributions.
    ggml_tensor * moe_out = ggml_view_2d(ctx0, down, down->ne[0], T,
            ggml_row_size(down->type, down->ne[0]) * n_expert_used, 0);
    moe_out = ggml_cont(ctx0, moe_out);
    for (int64_t e = 1; e < n_expert_used; e++) {
        ggml_tensor * s = ggml_view_2d(ctx0, down, down->ne[0], T,
                ggml_row_size(down->type, down->ne[0]) * n_expert_used,
                ggml_row_size(down->type, down->ne[0]) * e);
        moe_out = ggml_add(ctx0, moe_out, s);
    }

    // One always-on shared expert, with its own scalar sigmoid gate per token.
    ggml_tensor * sg = ggml_mul_mat(ctx0, Wl(il, "ffn_gate_shexp.weight"), cur);
    ggml_tensor * su = ggml_mul_mat(ctx0, Wl(il, "ffn_up_shexp.weight"),   cur);
    ggml_tensor * sh = ggml_mul_mat(ctx0, Wl(il, "ffn_down_shexp.weight"),
                                    ggml_mul(ctx0, ggml_silu(ctx0, sg), su));
    ggml_tensor * shared_gate = ggml_sigmoid(ctx0,
            ggml_mul_mat(ctx0, Wl(il, "ffn_gate_inp_shexp.weight"), cur));           // [1, T]
    sh = ggml_mul(ctx0, sh, shared_gate);

    return ggml_add(ctx0, moe_out, sh);
}

ggml_tensor * graph_builder::ple(ggml_tensor * emb, ggml_tensor * hidden, int il) {
    const int64_t hc     = hp_->hc_count;
    const int64_t n_embd = hp_->n_embd;
    const int64_t hc_dim = hc * n_embd;
    const int64_t T      = hidden->ne[2];

    ggml_tensor * key   = ggml_mul_mat(ctx0, Wl(il, "ple_key.weight"),   emb);  // [hc_dim, T]
    ggml_tensor * value = ggml_mul_mat(ctx0, Wl(il, "ple_value.weight"), emb);  // [n_embd, T]

    // Both norms reduce within one hc stream but scale with a full hc_dim gamma.
    auto grouped_norm = [&](ggml_tensor * t, ggml_tensor * w) {
        t = ggml_reshape_3d(ctx0, t, n_embd, hc, T);
        t = ggml_rms_norm(ctx0, t, hp_->rms_eps);
        t = ggml_reshape_2d(ctx0, t, hc_dim, T);
        t = ggml_mul(ctx0, t, w);
        return ggml_reshape_3d(ctx0, t, n_embd, hc, T);
    };

    key = grouped_norm(key, Wl(il, "ple_norm_key.weight"));
    ggml_tensor * query = grouped_norm(hidden, Wl(il, "ple_norm_query.weight"));

    // Per-stream dot product, then a signed square root before the sigmoid.
    ggml_tensor * sdot = ggml_sum_rows(ctx0, ggml_mul(ctx0, key, query));       // [1, hc, T]
    sdot = ggml_scale(ctx0, sdot, 1.0f / sqrtf((float) n_embd));
    ggml_tensor * mag  = ggml_sqrt(ctx0, ggml_clamp(ctx0, ggml_abs(ctx0, sdot), 1e-6f, INFINITY));
    ggml_tensor * gate = ggml_sigmoid(ctx0, ggml_mul(ctx0, ggml_sgn(ctx0, sdot), mag));

    ggml_tensor * v3 = ggml_reshape_3d(ctx0, value, n_embd, 1, T);
    v3 = ggml_repeat_4d(ctx0, v3, n_embd, hc, T, 1);
    ggml_tensor * gated = ggml_mul(ctx0, v3, gate);

    ggml_tensor * normalized = grouped_norm(ggml_reshape_2d(ctx0, gated, hc_dim, T),
                                            Wl(il, "ple_norm_conv.weight"));
    normalized = ggml_reshape_2d(ctx0, normalized, hc_dim, T);

    // Depthwise causal conv, dilated by the n-gram size, as a sum of shifted taps.
    const int64_t kern = hp_->ple_conv_kernel;
    const int64_t dil  = hp_->ple_ngram_size;
    const int64_t hist = (kern - 1) * dil;

    ggml_tensor * padded = conv_with_history(st_->ple_conv(), normalized, hist, hc_dim);
    ggml_tensor * w1d    = Wl(il, "ple_conv1d.weight");   // [kern, hc_dim]

    ggml_tensor * conv_out = nullptr;
    for (int64_t k = 0; k < kern; k++) {
        const int64_t start = hist - (kern - 1 - k) * dil;
        ggml_tensor * shifted = ggml_cont(ctx0, ggml_transpose(ctx0,
                ggml_view_2d(ctx0, padded, T, hc_dim, padded->nb[1],
                             ggml_row_size(padded->type, start))));
        ggml_tensor * wk = ggml_cont(ctx0,
                ggml_view_2d(ctx0, w1d, 1, hc_dim, w1d->nb[1], k * w1d->nb[0]));
        wk = ggml_reshape_1d(ctx0, wk, hc_dim);
        if (wk->type != GGML_TYPE_F32) wk = ggml_cast(ctx0, wk, GGML_TYPE_F32);
        ggml_tensor * term = ggml_mul(ctx0, shifted, wk);
        conv_out = conv_out ? ggml_add(ctx0, conv_out, term) : term;
    }
    conv_out = ggml_silu(ctx0, conv_out);
    conv_out = ggml_reshape_3d(ctx0, ggml_cont(ctx0, conv_out), n_embd, hc, T);

    return ggml_add(ctx0, hidden, ggml_add(ctx0, gated, conv_out));
}


void graph_builder::moe_route(ggml_tensor * cur, int il, ggml_tensor ** sel, ggml_tensor ** w) {
    const int64_t n_expert      = hp_->n_expert;
    const int64_t n_expert_used = hp_->n_expert_used;
    const int64_t T             = cur->ne[1];

    ggml_tensor * logits = ggml_mul_mat(ctx0, Wl(il, "ffn_gate_inp.weight"), cur);
    ggml_tensor * probs  = ggml_soft_max(ctx0, logits);

    ggml_tensor * selected = ggml_top_k(ctx0, probs, n_expert_used);
    ggml_tensor * weights_ = ggml_get_rows(ctx0,
            ggml_reshape_3d(ctx0, probs, 1, n_expert, T), selected);
    weights_ = ggml_reshape_2d(ctx0, weights_, n_expert_used, T);
    weights_ = ggml_div(ctx0, weights_, ggml_sum_rows(ctx0, weights_));

    *sel = selected;
    *w   = weights_;
}

ggml_tensor * graph_builder::moe_route_predict(ggml_tensor * res_hc, int il_next, int k,
                                               ggml_tensor * head_w, ggml_tensor * head_b, ggml_tensor ** x_out) {
    ggml_tensor * cur = hc_mix(res_hc, il_next, /*ffn=*/true, nullptr);
    if (x_out) *x_out = cur;
    // A learned head (trained with qwfn-train-predictor; measured no better) replaces the
    // router matrix when given: same input, weights fine-tuned offline to the
    // routing the next layer actually produces after its own block.
    ggml_tensor * logits = ggml_mul_mat(ctx0, head_w ? head_w : Wl(il_next, "ffn_gate_inp.weight"), cur);
    if (head_b) logits = ggml_add(ctx0, logits, head_b);
    // Only the ranking matters, so the softmax and the renormalisation are
    // skipped. argsort rather than top_k because the result must be ordered:
    // the first n_expert_used entries are scored against the true routing.
    return ggml_argsort_top_k(ctx0, logits, k);
}

ggml_tensor * graph_builder::moe_apply(ggml_tensor * cur, int il,
                                       ggml_tensor * const * gate, ggml_tensor * const * up,
                                       ggml_tensor * const * down, const float * w, int n_used) {
    ggml_tensor * acc = nullptr;
    for (int e = 0; e < n_used; e++) {
        ggml_tensor * g = ggml_mul_mat(ctx0, gate[e], cur);
        ggml_tensor * u = ggml_mul_mat(ctx0, up[e],   cur);
        ggml_tensor * y = ggml_mul_mat(ctx0, down[e], ggml_mul(ctx0, ggml_silu(ctx0, g), u));

        y = ggml_scale(ctx0, y, w[e]);
        acc = acc ? ggml_add(ctx0, acc, y) : y;
    }

    return acc;
}

ggml_tensor * graph_builder::shared_expert(ggml_tensor * cur, int il) {
    ggml_tensor * sg = ggml_mul_mat(ctx0, Wl(il, "ffn_gate_shexp.weight"), cur);
    ggml_tensor * su = ggml_mul_mat(ctx0, Wl(il, "ffn_up_shexp.weight"),   cur);
    ggml_tensor * sh = ggml_mul_mat(ctx0, Wl(il, "ffn_down_shexp.weight"),
                                    ggml_mul(ctx0, ggml_silu(ctx0, sg), su));
    // One scalar sigmoid gate per token, distinct from the routed gates.
    ggml_tensor * g = ggml_sigmoid(ctx0,
            ggml_mul_mat(ctx0, Wl(il, "ffn_gate_inp_shexp.weight"), cur));
    return ggml_mul(ctx0, sh, g);
}


ggml_tensor * graph_builder::moe_apply_batched(ggml_tensor * cur, int il,
                                               const std::vector<expert_group> & groups,
                                               ggml_tensor * perm, ggml_tensor * inv,
                                               ggml_tensor * wperm, int64_t n_tokens, int n_used) {
    // cur is [n_embd, T]; rows are tokens, so get_rows gathers token vectors.
    ggml_tensor * xp = ggml_get_rows(ctx0, cur, perm);          // [n_embd, P] expert-major

    // One destination, written in disjoint row ranges -- NOT a concat left-fold.
    // The fold built a chain as deep as the group count (257 at T=270), whose
    // intermediates are O(G^2) in total size and which pushed the allocator into
    // reusing buffers that the next concat was still reading.
    const int64_t P = perm->ne[0];
    ggml_tensor * yp = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, cur->ne[0], P);
    ggml_set_input(yp);                 // written by cpy, not produced by an op

    bool any = false;
    for (const auto & g : groups) {
        if (g.cnt == 0) continue;
        ggml_tensor * x = ggml_view_2d(ctx0, xp, xp->ne[0], g.cnt,
                                       xp->nb[1], g.off * xp->nb[1]);
        ggml_tensor * a = ggml_mul_mat(ctx0, g.gate, x);
        ggml_tensor * b = ggml_mul_mat(ctx0, g.up,   x);
        ggml_tensor * y = ggml_mul_mat(ctx0, g.down, ggml_mul(ctx0, ggml_silu(ctx0, a), b));
        ggml_tensor * dst = ggml_view_2d(ctx0, yp, yp->ne[0], g.cnt,
                                         yp->nb[1], (size_t) g.off * yp->nb[1]);
        ggml_build_forward_expand(gf_, ggml_cpy(ctx0, y, dst));
        any = true;
    }
    if (!any) return nullptr;

    // Scale each pair by its gate weight, then undo the permutation.
    yp = ggml_mul(ctx0, yp, wperm);                              // wperm is [1, P]
    ggml_tensor * yt = ggml_get_rows(ctx0, yp, inv);             // [n_embd, P] token-major

    // Rows for one token are now adjacent: sum the n_used of them.
    yt = ggml_reshape_3d(ctx0, yt, yt->ne[0], n_used, n_tokens);
    ggml_tensor * acc = ggml_cont(ctx0,
            ggml_view_2d(ctx0, yt, yt->ne[0], n_tokens, yt->nb[2], 0));
    for (int e = 1; e < n_used; e++) {
        acc = ggml_add(ctx0, acc,
                ggml_view_2d(ctx0, yt, yt->ne[0], n_tokens, yt->nb[2], (size_t) e * yt->nb[1]));
    }
    return acc;
}

} // namespace qwfn
