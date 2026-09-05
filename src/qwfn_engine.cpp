#include "qwfn_engine.h"
#include "qwfn_ple.h"

#include <algorithm>
#include <map>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <numeric>
#include <sstream>

namespace qwfn {

static const char * EXP_SUFFIX[EXPERT_NPARTS] = {
    "ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight"
};

static uint16_t f16_of(float f) {
    ggml_fp16_t h = ggml_fp32_to_fp16(f);
    uint16_t o; memcpy(&o, &h, 2); return o;
}

engine::~engine() {
    if (moe_ctx_) {   // an async MoE graph may still be in flight
        ggml_backend_synchronize(w_.backend());
        ggml_free(moe_ctx_);
    }
    for (auto & lg : gA_) {
        if (lg.ga)  ggml_gallocr_free(lg.ga);
        if (lg.ctx) ggml_free(lg.ctx);
    }
    for (auto & mg : gM_) {
        if (mg.ga)  ggml_gallocr_free(mg.ga);
        if (mg.ctx) ggml_free(mg.ctx);
    }
    if (galloc_gpu_) ggml_gallocr_free(galloc_gpu_);
    if (galloc_cpu_) ggml_gallocr_free(galloc_cpu_);
    if (galloc_moe_) ggml_gallocr_free(galloc_moe_);
    if (wbuf_) ggml_backend_buffer_free(wbuf_);
    if (hbuf_) ggml_backend_buffer_free(hbuf_);
    if (galloc_pf_) ggml_gallocr_free(galloc_pf_);
    if (galloc_pf_dyn_) ggml_gallocr_free(galloc_pf_dyn_);
    if (vbuf_) ggml_backend_buffer_free(vbuf_);
    if (vctx_) ggml_free(vctx_);
    if (predbuf_) ggml_backend_buffer_free(predbuf_);
    if (predctx_) ggml_free(predctx_);
    for (FILE * f : dump_f_) if (f) fclose(f);
    for (FILE * f : dump_dec_f_) if (f) fclose(f);
    if (pwbuf_) ggml_backend_buffer_free(pwbuf_);
    if (pwctx_) ggml_free(pwctx_);
    if (qbuf_) ggml_backend_buffer_free(qbuf_);
    if (qctx_) ggml_free(qctx_);
    if (pbuf_) ggml_backend_buffer_free(pbuf_);
    if (pctx_) ggml_free(pctx_);
    if (scr_buf_) ggml_backend_buffer_free(scr_buf_);
    if (scr_ctx_) ggml_free(scr_ctx_);
    if (dbuf_) ggml_backend_buffer_free(dbuf_);
    if (dctx_) ggml_free(dctx_);
    if (wctx_) ggml_free(wctx_);
    if (hctx_) ggml_free(hctx_);
}

bool engine::init(const model_index * hot, const model_index * cold,
                  const engine_config & cfg, const std::string & backend_dir, std::string & err) {
    mi_  = hot;
    cfg_ = cfg;
    hp_  = hot->hp();
    n_vocab_ = hp_.n_vocab;
    if (cfg.indexer_top_k) {
        fprintf(stderr, "[qwfn] indexer top-k %u -> %u\n", hp_.idx_top_k, cfg.indexer_top_k);
        hp_.idx_top_k = cfg.indexer_top_k;
    }

    if (!w_.init(hot, cfg.use_gpu, backend_dir, err)) return false;
    if (!w_.declare_dense_core(err)) return false;
    if (!w_.commit(err)) return false;

    if (!wh_.init(hot, /*prefer_gpu=*/false, backend_dir, err)) return false;

    // ---- MTP draft head (experiment): the nextn block, resident ---------------
    if (!cfg.mtp_path.empty() && cfg.use_gpu) {
        if (!mi_mtp_.load(cfg.mtp_path, err)) return false;
        if (!wm_.init(&mi_mtp_, /*prefer_gpu=*/true, backend_dir, err)) return false;
        for (const auto & kv : mi_mtp_.tensors())
            if (!wm_.declare(kv.first)) { err = "mtp: failed to declare " + kv.first; return false; }
        if (!wm_.commit(err)) return false;
        hpm_ = mi_mtp_.hp();
        hpm_.full_attention_interval = hpm_.n_layer;   // in this index only the last block, the nextn block, is attention
        hpm_.ssm_dt_rank = 1;                           // the index's 48 trunk slots carry no state here; keep theirs tiny
        hpm_.hc_inject_prescaled = false;               // its inject weights are Q8_0 and are not folded
        state_config scm; scm.n_ctx = cfg.n_ctx; scm.type_k = cfg.type_k; scm.type_v = cfg.type_v;
        if (!st_mtp_.init(&hpm_, scm, w_.buft(), err)) return false;
        mtp_on_ = true;
        fprintf(stderr, "[qwfn] mtp: nextn block %u of %s, %.2f GB resident on %s; drafts are scored, not verified (experiment)\n",
                hpm_.n_layer - 1, cfg.mtp_path.c_str(), wm_.bytes() / 1e9, wm_.dev_name());
    }
    if (!wh_.map_shards(err)) return false;
    if (!wh_.declare_mapped("per_layer_token_embd.weight")) {
        err = "per_layer_token_embd.weight missing"; return false;
    }
    // The token embedding table (0.68 GB at Q8_0) is only ever gathered one
    // row per token, so it stays in the host mapping like the PLE table and
    // the rows are uploaded; that is 0.68 GB more of VRAM expert tier.
    if (!wh_.declare_mapped("token_embd.weight")) { err = "token_embd.weight missing"; return false; }
    // Prefill reads experts straight from the mapping; the pages it touches are
    // whatever the routing asks for, and the cache is left alone for decode.
    for (uint32_t il = 0; il < hp_.n_layer; il++)
        for (int q = 0; q < EXPERT_NPARTS; q++)
            if (!wh_.declare_mapped("blk." + std::to_string(il) + "." + EXP_SUFFIX[q])) {
                err = "expert tensor missing on layer " + std::to_string(il); return false;
            }
    have_expert_map_ = true;
    wh_.set_n_threads(cfg.n_threads);
    w_.set_n_threads(cfg.n_threads);

    // State first: the KV cache, indexer keys and recurrent state are not
    // optional, whereas the VRAM expert tier is. Allocating the tier first let
    // it take the memory the state needed and fail the whole engine.
    state_config sc;
    sc.n_ctx  = cfg.n_ctx;
    sc.type_k = cfg.type_k;
    sc.type_v = cfg.type_v;
    if (!st_.init(&hp_, sc, w_.buft(), err)) return false;

    // The persistent work buffers are sized by n_batch and are not optional, so
    // they go in before the VRAM expert tier -- same reasoning as the state
    // above. At ubatch 2048 they are ~420 MB; leaving them until after the tier
    // meant the tier had already taken that memory.
    const int64_t n_embd = hp_.n_embd, hc = hp_.hc_count;
    const int64_t B = cfg.n_batch, U = hp_.n_expert_used, PH = hp_.ple_n_head();
    // Decode and the short-prompt batch never see more than prefill_decode_max
    // tokens at once; the n_batch-sized set exists only during a prefill.
    const int64_t Bd = std::max<int64_t>(1, std::min<int64_t>(B, std::max<uint32_t>(1, cfg.prefill_decode_max)));

    ggml_init_params wp{}; wp.mem_size = ggml_tensor_overhead() * 64; wp.no_alloc = true;
    wctx_ = ggml_init(wp);
    res_[0]   = ggml_new_tensor_3d(wctx_, GGML_TYPE_F32, n_embd, hc, Bd);
    res_[1]   = ggml_new_tensor_3d(wctx_, GGML_TYPE_F32, n_embd, hc, Bd);
    t_cur_    = ggml_new_tensor_2d(wctx_, GGML_TYPE_F32, n_embd, Bd);
    t_emb_    = ggml_new_tensor_2d(wctx_, GGML_TYPE_F32, n_embd, Bd);
    t_sh_     = ggml_new_tensor_2d(wctx_, GGML_TYPE_F32, n_embd, Bd);
    t_pg_     = ggml_new_tensor_2d(wctx_, GGML_TYPE_F32, n_embd, Bd);
    t_pc_     = ggml_new_tensor_2d(wctx_, GGML_TYPE_F32, n_embd, Bd);
    t_ple_    = ggml_new_tensor_2d(wctx_, GGML_TYPE_F32, n_embd, Bd);
    t_inject_ = ggml_new_tensor_2d(wctx_, GGML_TYPE_F32, hc, Bd);
    t_sel_    = ggml_new_tensor_2d(wctx_, GGML_TYPE_I32, U, Bd);
    t_selnext_= ggml_new_tensor_2d(wctx_, GGML_TYPE_I32, QWFN_SPEC_MAX, Bd);
    t_selnext2_= ggml_new_tensor_2d(wctx_, GGML_TYPE_I32, QWFN_SPEC_MAX, Bd);
    t_specscore_= ggml_new_tensor_2d(wctx_, GGML_TYPE_F32, QWFN_SPEC_MAX, Bd);
    t_hcmean_   = ggml_new_tensor_2d(wctx_, GGML_TYPE_F32, hc, 1);   // (1/hc, ...): the stream mean as a matmul
    if (mtp_on_) {
        t_hlast_   = ggml_new_tensor_3d(wctx_, GGML_TYPE_F32, n_embd, hc, Bd);
        t_mtp_pos_ = ggml_new_tensor_1d(wctx_, GGML_TYPE_I32, 4 * Bd);
    }
    // Shape the hyper-connection and PLE norm gammas [hc*n_embd] as [n_embd, hc]
    // -- metadata only, same bytes -- so hc_mix and ple can multiply right after
    // the RMSNorm without a reshape node between them: ggml-cuda fuses
    // (rms_norm, mul) only when the two are adjacent.
    {
        auto reshape_gamma_in = [&](weights & wt, const std::string & name) {
            ggml_tensor * t = wt.get(name);
            if (!t || ggml_nelements(t) != (int64_t) hc * n_embd || t->ne[1] == (int64_t) hc) return;
            if (ggml_blck_size(t->type) != 1) return;
            t->ne[0] = n_embd; t->ne[1] = hc; t->ne[2] = 1; t->ne[3] = 1;
            t->nb[1] = t->nb[0] * n_embd; t->nb[2] = t->nb[1] * hc; t->nb[3] = t->nb[2];
        };
        auto reshape_gamma = [&](const std::string & name) { reshape_gamma_in(w_, name); };
        if (mtp_on_) {
            const std::string b = "blk." + std::to_string(hpm_.n_layer - 1) + ".";
            for (const char * n : { "hc_attn_norm", "hc_ffn_norm", "nextn.hnorm", "nextn.hc_head_norm" })
                reshape_gamma_in(wm_, b + n + ".weight");
        }
        reshape_gamma("output_hc_norm.weight");
        for (uint32_t l = 0; l < hp_.n_layer; l++) {
            const std::string b = "blk." + std::to_string(l) + ".";
            for (const char * n : { "hc_attn_norm", "hc_ffn_norm", "ple_norm_key", "ple_norm_query", "ple_norm_conv" })
                reshape_gamma(b + n + ".weight");
        }
        // Fold the 1/hc of hc_combine's gate into the F32 inject weights: 0.25 is a
        // power of two, so every product and partial sum rounds exactly as before
        // and the scale kernel disappears. Only if every inject weight is F32.
        bool all_f32 = true; std::vector<ggml_tensor *> inj;
        for (uint32_t l = 0; l < hp_.n_layer && all_f32; l++)
            for (const char * n : { "hc_attn_inject", "hc_ffn_inject" }) {
                ggml_tensor * t = w_.get("blk." + std::to_string(l) + "." + n + ".weight");
                if (!t) continue;
                if (t->type != GGML_TYPE_F32) { all_f32 = false; break; }
                inj.push_back(t);
            }
        if (all_f32 && !inj.empty() && !getenv("QWFN_NO_INJECT_FOLD")) {
            std::vector<float> buf;
            for (ggml_tensor * t : inj) {
                buf.resize(ggml_nelements(t));
                ggml_backend_tensor_get(t, buf.data(), 0, buf.size() * sizeof(float));
                for (float & v : buf) v *= 1.0f / (float) hc;
                ggml_backend_tensor_set(t, buf.data(), 0, buf.size() * sizeof(float));
            }
            hp_.hc_inject_prescaled = true;
        }
    }
    pack_n_ = n_embd + 2 * (int64_t) U + 2 * (int64_t) QWFN_SPEC_MAX;
    if ((int64_t) Bd * n_embd >= pack_n_ && cfg.use_gpu && !getenv("QWFN_NO_PACK")) {
        t_pack_ = ggml_view_1d(wctx_, t_cur_, pack_n_, 0);
        pack_host_.resize(pack_n_); pred_next_.resize(QWFN_SPEC_MAX); scores_next_.resize(QWFN_SPEC_MAX);
    }
    gA_pack_.assign(hp_.n_layer, 0);
    // Speculative-block mask by predicted layer, and the per-layer counters.
    spec_block_mask_.assign(hp_.n_layer, cfg.spec_block_layers.empty() ? 1 : 0);
    if (!cfg.spec_block_layers.empty()) {
        const std::string & sl = cfg.spec_block_layers; size_t p = 0;
        while (p < sl.size()) {
            size_t cpos = sl.find(',', p); if (cpos == std::string::npos) cpos = sl.size();
            const std::string tok = sl.substr(p, cpos - p); p = cpos + 1;
            if (tok.empty()) continue;
            const size_t d = tok.find('-');
            const int a = atoi(tok.c_str()), b = d == std::string::npos ? a : atoi(tok.c_str() + d + 1);
            for (int l = std::max(a, 0); l <= b && l < (int) hp_.n_layer; l++) spec_block_mask_[l] = 1;
        }
    }
    pred_hits_layer.assign(hp_.n_layer, 0); pred_total_layer.assign(hp_.n_layer, 0);
    if (cfg.spec_block || cfg.spec_margin > 0.0f) {
        int nb = 0; for (uint32_t l = 1; l < hp_.n_layer; l++) nb += cfg.spec_block ? spec_block_mask_[l] : 0;
        fprintf(stderr, "[qwfn] prefetch: speculative block %s (%d of %u predicted layers), margin gate %s\n",
                cfg.spec_block ? "on" : "off", nb, hp_.n_layer - 1,
                cfg.spec_margin > 0.0f ? (std::to_string(cfg.spec_margin) + (cfg.spec_gate_inflight ? " when >= " + std::to_string(cfg.spec_gate_inflight) + " reads in flight" : "")).c_str() : "off");
    }
    t_w_      = ggml_new_tensor_2d(wctx_, GGML_TYPE_F32, U, Bd);
    t_gids_   = ggml_new_tensor_2d(wctx_, GGML_TYPE_I32, U, 1);
    t_gw_     = ggml_new_tensor_3d(wctx_, GGML_TYPE_F32, 1, U, 1);
    inp_tok_  = ggml_new_tensor_1d(wctx_, GGML_TYPE_I32, Bd);
    inp_pos_  = ggml_new_tensor_1d(wctx_, GGML_TYPE_I32, Bd * 4);
    inp_ple_  = ggml_new_tensor_1d(wctx_, GGML_TYPE_I32, PH * Bd);
    if (getenv("QWFN_ROUTE_DUMP_DECODE")) t_xdec_ = ggml_new_tensor_2d(wctx_, GGML_TYPE_F32, n_embd, hp_.n_layer);
    if (cfg.skip_miss) t_rscale_ = ggml_new_tensor_1d(wctx_, GGML_TYPE_F32, 1);
    wbuf_ = ggml_backend_alloc_ctx_tensors_from_buft(wctx_, w_.buft());
    { std::vector<float> m(hc, 1.0f / (float) hc); ggml_backend_tensor_set(t_hcmean_, m.data(), 0, m.size() * sizeof(float)); }
    if (!wbuf_) { err = "failed to allocate engine work buffer"; return false; }
    if (t_rscale_) { const float one = 1.0f; ggml_backend_tensor_set(t_rscale_, &one, 0, 4); }
    save_work_set(dec_ws_);

    ggml_init_params hpar{}; hpar.mem_size = ggml_tensor_overhead() * 32; hpar.no_alloc = true;
    hctx_ = ggml_init(hpar);
    h_cur_     = ggml_new_tensor_2d(hctx_, GGML_TYPE_F32, n_embd, B);
    h_partial_ = ggml_new_tensor_2d(hctx_, GGML_TYPE_F32, n_embd, B);
    h_ple_     = ggml_new_tensor_2d(hctx_, GGML_TYPE_F32, n_embd, B);
    h_ple_idx_ = ggml_new_tensor_1d(hctx_, GGML_TYPE_I32, PH * B);
    h_tok_     = ggml_new_tensor_1d(hctx_, GGML_TYPE_I32, B);
    h_emb_     = ggml_new_tensor_2d(hctx_, GGML_TYPE_F32, n_embd, B);
    hbuf_ = ggml_backend_alloc_ctx_tensors_from_buft(hctx_, wh_.buft());
    if (!hbuf_) { err = "failed to allocate engine host buffer"; return false; }

    // ---- pinned staging for the async id/weight uploads ----------------------
    if (w_.on_gpu()) {
        ggml_backend_dev_t dev = ggml_backend_buft_get_device(w_.buft());
        ggml_backend_buffer_type_t hb = dev ? ggml_backend_dev_host_buffer_type(dev) : nullptr;
        if (hb) {
            ggml_init_params pp{}; pp.mem_size = ggml_tensor_overhead() * 8; pp.no_alloc = true;
            pctx_    = ggml_init(pp);
            p_gids_  = ggml_new_tensor_1d(pctx_, GGML_TYPE_I32, U);
            p_gw_    = ggml_new_tensor_1d(pctx_, GGML_TYPE_F32, U);
            p_vslot_ = ggml_new_tensor_1d(pctx_, GGML_TYPE_I32, hp_.n_expert);
            p_vmask_ = ggml_new_tensor_1d(pctx_, GGML_TYPE_F32, hp_.n_expert);
            p_pc_    = ggml_new_tensor_1d(pctx_, GGML_TYPE_F32, n_embd);
            pbuf_    = ggml_backend_alloc_ctx_tensors_from_buft(pctx_, hb);
            if (!pbuf_ || ggml_backend_buffer_get_type(pbuf_) != hb) {
                if (pbuf_) ggml_backend_buffer_free(pbuf_);
                ggml_free(pctx_); pbuf_ = nullptr; pctx_ = nullptr; p_gids_ = nullptr; p_gw_ = nullptr;
                p_vslot_ = nullptr; p_vmask_ = nullptr; p_pc_ = nullptr;
            }
        }
    }

    // ---- decode-time sparse attention state --------------------------------
    qsa_ratio_ = 0;
    if (cfg.use_qsa)
        for (uint32_t il = 0; il < hp_.n_layer; il++)
            if (hp_.is_attn_layer(il) && il < hp_.compress_ratios.size() && hp_.compress_ratios[il] > 0)
                { qsa_ratio_ = (uint32_t) hp_.compress_ratios[il]; break; }
    if (qsa_ratio_ && !getenv("QWFN_LEGACY_QSA_DECODE")) {
        const int64_t r = qsa_ratio_, NBmax = (cfg.n_ctx + r - 1) / r, idx_dim = hp_.idx_key_len;
        ggml_init_params qp{}; qp.mem_size = ggml_tensor_overhead() * (hp_.n_layer + 16); qp.no_alloc = true;
        qctx_ = ggml_init(qp);
        pool_cache_.assign(hp_.n_layer, nullptr);
        for (uint32_t il = 0; il < hp_.n_layer; il++)
            if (hp_.is_attn_layer(il)) pool_cache_[il] = ggml_new_tensor_2d(qctx_, GGML_TYPE_F16, idx_dim, NBmax);
        qd_.bias       = ggml_new_tensor_1d(qctx_, GGML_TYPE_F32, NBmax);
        qd_.blk_cells  = ggml_new_tensor_2d(qctx_, GGML_TYPE_I32, r, NBmax);
        qd_.cell_pos   = ggml_new_tensor_2d(qctx_, GGML_TYPE_F32, 1, cfg.n_ctx);
        qd_.write_idx  = ggml_new_tensor_1d(qctx_, GGML_TYPE_I32, 1);
        qd_.member_idx = ggml_new_tensor_1d(qctx_, GGML_TYPE_I32, r);
        qd_.blk_pos    = ggml_new_tensor_1d(qctx_, GGML_TYPE_I32, 4);
        qd_.blk_idx    = ggml_new_tensor_1d(qctx_, GGML_TYPE_I32, 1);
        qd_.npast_f    = ggml_new_tensor_1d(qctx_, GGML_TYPE_F32, 1);
        qbuf_ = ggml_backend_alloc_ctx_tensors_from_buft(qctx_, w_.buft());
        if (!qbuf_) {
            fprintf(stderr, "[qwfn] no device memory for the decode QSA state; decode attention scans the whole context\n");
            ggml_free(qctx_); qctx_ = nullptr; pool_cache_.clear();
        } else {
            ggml_backend_buffer_clear(qbuf_, 0);
            std::vector<int32_t> bc((size_t) r * NBmax);
            for (int64_t b = 0; b < NBmax; b++) for (int64_t k = 0; k < r; k++) bc[b * r + k] = (int32_t) (b * r + k);
            ggml_backend_tensor_set(qd_.blk_cells, bc.data(), 0, bc.size() * 4);
            std::vector<float> cp(cfg.n_ctx);
            for (uint32_t i = 0; i < cfg.n_ctx; i++) cp[i] = (float) i;
            ggml_backend_tensor_set(qd_.cell_pos, cp.data(), 0, cp.size() * 4);
            std::vector<float> ninf(NBmax, -INFINITY);
            ggml_backend_tensor_set(qd_.bias, ninf.data(), 0, ninf.size() * 4);
            qd_.ratio    = qsa_ratio_;
            qd_.k_blocks = (int64_t) ((hp_.idx_top_k + r - 1) + r - 1) / r;   // ceil(width / r)
            fprintf(stderr, "[qwfn] decode QSA state: %.1f MB (pooled block keys for %lld blocks, %lld kept)\n",
                    ggml_backend_buffer_get_size(qbuf_) / 1e6, (long long) NBmax, (long long) qd_.k_blocks);
        }
    }

    // Scratch for the cache-served batched prefill, taken before the VRAM
    // tier sizes itself (it is small: 48 slots of the largest expert, ~170 MB).
    if (cfg.use_gpu && w_.on_gpu() && cfg.cache_batched && cfg.vram_bytes > 0) {
        scr_slots_ = std::max<uint32_t>(1, std::min<uint32_t>(64, cfg.cache_batch_chunk));
        size_t total = 0;
        for (int q = 0; q < EXPERT_NPARTS; q++) {
            size_t mx = 0;
            for (uint32_t il = 0; il < hp_.n_layer; il++)
                mx = std::max<size_t>(mx, mi_->expert_range(il, 0, (expert_part) q).nbytes);
            scr_part_off_[q] = total;
            total += mx * scr_slots_ + (16u << 10);   // room for the zeroed tail
        }
        scr_buf_ = ggml_backend_buft_alloc_buffer(w_.buft(), total);
        if (scr_buf_) {
            scr_base_ = (uint8_t *) ggml_backend_buffer_get_base(scr_buf_);
            ggml_backend_buffer_clear(scr_buf_, 0);
            ggml_init_params sp{}; sp.mem_size = ggml_tensor_overhead() * 4; sp.no_alloc = true;
            scr_ctx_  = ggml_init(sp);
            scr_xfer_ = ggml_new_tensor_1d(scr_ctx_, GGML_TYPE_I8, (int64_t) total);
            scr_xfer_->buffer = scr_buf_;
            scr_xfer_->data   = scr_base_;
        } else {
            fprintf(stderr, "[qwfn] no device memory for the batched-prefill scratch; short prompts go token by token\n");
            scr_slots_ = 0;
        }
    }

    expert_cache::config ec_cfg;
    ec_cfg.ram_bytes     = cfg.ram_bytes;
    // With --cpu the "VRAM" buffer type is host memory: a tier there would just
    // duplicate the arena outside the MemAvailable clamp. GPU only.
    ec_cfg.vram_bytes    = cfg.use_gpu ? cfg.vram_bytes : 0;
    ec_cfg.vram_buft     = w_.buft();
    if (w_.on_gpu()) {
        ggml_backend_dev_t dev = ggml_backend_buft_get_device(w_.buft());
        ec_cfg.host_buft    = dev ? ggml_backend_dev_host_buffer_type(dev) : nullptr;
        ec_cfg.vram_backend = w_.backend();
    }
    ec_cfg.async_promote = getenv("QWFN_SYNC_PROMOTE") == nullptr;
    ec_cfg.use_cold_tier = cfg.use_cold_tier;
    ec_cfg.max_promotions_per_layer = cfg.promote_per_layer;
    ec_cfg.ram_frac      = cfg.ram_frac;
    ec_cfg.ram_headroom  = cfg.ram_headroom;
    ec_cfg.io_backend  = cfg.io_threads ? io_engine::backend::threads : io_engine::backend::uring;
    if (cfg.io_threads) ec_cfg.queue_depth = cfg.io_workers;
    ec_cfg.policy = cfg.evict_policy == 1 ? expert_cache::config::evict_policy::lfu
                  : cfg.evict_policy == 2 ? expert_cache::config::evict_policy::hybrid
                                          : expert_cache::config::evict_policy::lru;
    if (!(cfg.use_gpu && cfg.vram_bytes > 0) || getenv("QWFN_SEPARATE_STAGING")) ec_cfg.lend_bytes = 0;

    // What still has to fit on the device after the tier has taken its share:
    //
    //   - the prefill MoE graph arena. Its xp/yp/yt intermediates are each
    //     [n_embd, n_batch * n_expert_used] F32, ~0.31 MB per token of ubatch,
    //     so a 2048 ubatch wants ~600 MB;
    //   - the per-call input arena, dominated by kq_mask F16 [n_kv, T] and the
    //     QSA bias F32 [n_kv/ratio, T], hence bounded by ubatch_kv_product;
    //   - the 36 replay allocators, created lazily on the first decode token.
    //
    // A fixed 768 MB covered ubatch 256 and failed outright at 2048 --
    // "prefill galloc failed", 15 minutes into a long prompt.
    // What a PREFILL needs on the device, all of it only while one runs: the
    // staging, the n_batch work set, the MoE chunk arena, the attention
    // chunk's inputs and its arena. During decode this memory holds expert
    // slots: it is the expert tier's dynamic buffer (expert_cache lend_bytes),
    // freed when a prefill starts and taken back when it ends. What decode
    // itself allocates after the tier -- the 48 replay allocators, the
    // short-prompt chunk graphs, the head -- is the only reserve left.
    {
        const size_t Tm = std::min<size_t>(B, std::max<uint32_t>(64, cfg.prefill_chunk));
        const size_t pf_graph = (cfg.prefill_on_gpu && cfg.use_gpu)
            ? (size_t) 4 * n_embd * U * Tm * sizeof(float) : 0;
        const size_t kvp = cfg.ubatch_kv_product ? (size_t) cfg.ubatch_kv_product : (size_t) 32 << 20;
        const size_t inputs    = std::min<size_t>(kvp * 3, 512ull << 20);
        const size_t qsa_graph = cfg.use_qsa ? std::min<size_t>(kvp * 14, 1536ull << 20) : 0;
        const size_t work_set  = (size_t) (2 * n_embd * hc + 6 * n_embd + hc + 3 * U + 5 + PH) * B * 4 + (64ull << 20);
        const size_t staging   = (cfg.prefill_on_gpu && cfg.use_gpu) ? prefill_streamer::device_bytes_for(hot) : 0;
        ec_cfg.lend_bytes   = staging + work_set + pf_graph + inputs + qsa_graph;
        ec_cfg.vram_reserve = cfg.vram_reserve ? cfg.vram_reserve : (768ull << 20);   // decode graphs, buckets, scratch
        fprintf(stderr, "[qwfn] prefill VRAM (dynamic, lent by the expert tier): %.2f GB; decode reserve %.0f MB\n",
                ec_cfg.lend_bytes / 1e9, ec_cfg.vram_reserve / 1e6);
    }

    // Decode computes each layer's VRAM-resident routed experts inside that
    // layer's own graph, looking residency up on the device; see eval_batch.
    moe_in_graph_ = w_.on_gpu() && cfg.vram_bytes > 0 && !cfg.use_cold_tier &&
                    !getenv("QWFN_LEGACY_MOE") && !getenv("QWFN_CHECK_MOE") && !getenv("QWFN_MOE_SEPARATE");
    // Experts promoted to VRAM during a layer's fetch were not resident when
    // that layer's graph ran: the next graph computes them from the tier by
    // slot (at most max_promotions_per_layer of them), the "late fold".
    n_late_ = moe_in_graph_ ? (int) std::max<uint32_t>(1, ec_cfg.max_promotions_per_layer) : 0;
    // Before the tier sizes itself from the free VRAM: the heads' 123 MB must come out of the tier, not the decode reserve.
    if (!cfg.predictor_path.empty() && !load_predictor(cfg.predictor_path, err)) return false;
    if (!ec_.init(hot, cold, ec_cfg, err)) return false;
    tier_epoch_seen_ = ec_.tier_epoch();
    if (moe_in_graph_) {
        ggml_init_params vp{}; vp.mem_size = ggml_tensor_overhead() * 2 * hp_.n_layer + 1024; vp.no_alloc = true;
        vctx_ = ggml_init(vp);
        t_vslot_.assign(hp_.n_layer, nullptr); t_vmask_.assign(hp_.n_layer, nullptr);
        vslot_ver_.assign(hp_.n_layer, UINT64_MAX);
        for (uint32_t il = 0; il < hp_.n_layer; il++) {
            t_vslot_[il] = ggml_new_tensor_2d(vctx_, GGML_TYPE_I32, 1, hp_.n_expert);
            t_vmask_[il] = ggml_new_tensor_2d(vctx_, GGML_TYPE_F32, 1, hp_.n_expert);
        }
        vbuf_ = ggml_backend_alloc_ctx_tensors_from_buft(vctx_, w_.buft());
        if (!vbuf_) { err = "failed to allocate the VRAM residency tables"; return false; }
        ggml_backend_buffer_clear(vbuf_, 0);
    }
    if (const char * d = getenv("QWFN_ROUTE_DUMP_DECODE")) {
        dump_dec_dir_ = d;
        tok_sel_.assign((size_t) hp_.n_layer * U, 0); tok_w_.assign((size_t) hp_.n_layer * U, 0.0f);
        fprintf(stderr, "[qwfn] decode routing dump to %s\n", d);
    }
    if (const char * d = getenv("QWFN_ROUTE_DUMP")) {
        dump_dir_ = d;
        if (const char * st = getenv("QWFN_ROUTE_STRIDE")) dump_stride_ = std::max(1, atoi(st));
        fprintf(stderr, "[qwfn] routing dump to %s, every %d-th prefill token\n", d, dump_stride_);
    }
    if (!pf_.init(hot, cfg.io_workers, cfg.io_threads,
                  cfg.prefill_on_gpu ? w_.buft() : nullptr,
                  cfg.prefill_on_gpu ? w_.backend() : nullptr, cfg.prefill_overlap, err)) return false;

    gA_.assign(hp_.n_layer, layer_graph{});
    gA_bucket_.assign(hp_.n_layer, -1);
    gM_.assign(hp_.n_layer, moe_graph{});
    galloc_gpu_ = ggml_gallocr_new(w_.buft());
    galloc_cpu_ = ggml_gallocr_new(wh_.buft());
    galloc_moe_ = ggml_gallocr_new(w_.buft());

    // The legacy per-ubatch prefill's device twins and the prefill MoE
    // allocator are created per prefill (prefill_enter) and freed after.

    logits_.resize((size_t) n_vocab_);
    xfer_.resize((size_t) n_embd * B);
    zeros_.assign((size_t) n_embd * B, 0.0f);
    sel_.resize((size_t) U * B);
    wgt_.resize((size_t) U * B);
    ids_.resize((size_t) U * B);
    return true;
}

// The mul_mat_id MoE over one tier: gate/up/down as [.., .., n_slots] tensors
// aliasing the tier's arrays, experts chosen by `ids` (slot indices, [n, 1]),
// gate weights `w` ([1, n, 1]), the n weighted outputs summed in order. Per
// expert this is the same mmvq / vec_dot kernel the per-expert path ran, the
// same scale-by-weight, and the same left-to-right sum, so it is bit-identical
// to that path -- an expert with weight 0 adds an exact 0.0 and changes
// nothing. Returns the [n_embd, 1] sum.
static ggml_tensor * moe_id_graph(ggml_context * c, const tier_view & tv,
                                  ggml_tensor * ids, ggml_tensor * w, ggml_tensor * x_in,
                                  int64_t n_embd, int64_t n_ff, int n, int64_t T = 1,
                                  bool fused_sum = false) {
    ggml_tensor * as[EXPERT_NPARTS];
    as[EXPERT_GATE] = ggml_new_tensor_3d(c, tv.type[EXPERT_GATE], n_embd, n_ff, tv.n_slots);
    as[EXPERT_UP]   = ggml_new_tensor_3d(c, tv.type[EXPERT_UP],   n_embd, n_ff, tv.n_slots);
    as[EXPERT_DOWN] = ggml_new_tensor_3d(c, tv.type[EXPERT_DOWN], n_ff, n_embd, tv.n_slots);
    for (int q = 0; q < EXPERT_NPARTS; q++) {
        as[q]->buffer = tv.buffer;
        as[q]->data   = tv.part[q];
        as[q]->nb[2]  = tv.stride[q];
        as[q]->nb[3]  = tv.stride[q] * tv.n_slots;
    }
    // x_in is [n_embd, T] token-major; as [n_embd, 1, T] each token's vector is
    // broadcast to its n experts. ids is [n, T], w is [1, n, T].
    ggml_tensor * x    = ggml_view_3d(c, x_in, n_embd, 1, T, x_in->nb[1], x_in->nb[1], 0);
    ggml_tensor * gate = ggml_mul_mat_id(c, as[EXPERT_GATE], x, ids);          // [n_ff, n, T]
    ggml_tensor * up   = ggml_mul_mat_id(c, as[EXPERT_UP],   x, ids);
    ggml_tensor * act  = ggml_swiglu_split(c, gate, up);                        // silu(gate) * up, one op; fused with the matmuls on CUDA
    ggml_tensor * down = ggml_mul_mat_id(c, as[EXPERT_DOWN], act, ids);        // [n_embd, n, T]
    if (fused_sum && T == 1) {
        // The weighted sum of the n expert rows as one matmul over the transposed
        // rows: two kernels instead of a multiply and n-1 adds. A different
        // summation order, so only the GPU graphs ask for it; the CPU path keeps
        // the sequential sum it is validated with.
        ggml_tensor * dt = ggml_reshape_2d(c, ggml_cont(c, ggml_permute(c, down, 1, 0, 2, 3)), n, n_embd);   // [n, n_embd]
        ggml_tensor * wv = ggml_reshape_2d(c, ggml_cont(c, w), n, 1);                                       // [n, 1]
        return ggml_mul_mat(c, dt, wv);                                                                       // [n_embd, 1]
    }
    ggml_tensor * wd   = ggml_mul(c, down, w);
    // Sum the n expert rows of each token, in order.
    ggml_tensor * acc  = ggml_cont(c, ggml_view_2d(c, wd, n_embd, T, wd->nb[2], 0));
    for (int e = 1; e < n; e++)
        acc = ggml_add(c, acc, ggml_view_2d(c, wd, n_embd, T, wd->nb[2], (size_t) e * wd->nb[1]));
    return acc;
}

bool engine::build_moe_gpu_graph(uint32_t il) {
    const tier_view tv = ec_.gpu_tier(il);
    if (tv.n_slots == 0) return false;
    const int64_t n_embd = hp_.n_embd, n_ff = hp_.n_ff_exp;
    const int U = (int) hp_.n_expert_used;
    for (int q = 0; q < EXPERT_NPARTS; q++) {
        const int64_t rows = q == EXPERT_DOWN ? n_embd : n_ff;
        const int64_t cols = q == EXPERT_DOWN ? n_ff : n_embd;
        if (tv.stride[q] != ggml_row_size(tv.type[q], cols) * (size_t) rows) {
            fprintf(stderr, "[qwfn] layer %u: VRAM tier stride is not the natural slice; mul_mat_id MoE disabled\n", il);
            return false;
        }
    }
    moe_graph & mg = gM_[il];
    ggml_init_params p{};
    p.mem_size = ggml_tensor_overhead() * 128 + ggml_graph_overhead_custom(128, false);
    p.no_alloc = true;
    mg.ctx = ggml_init(p);
    mg.gf  = ggml_new_graph_custom(mg.ctx, 128, false);
    ggml_tensor * acc = moe_id_graph(mg.ctx, tv, t_gids_, t_gw_, t_cur_, n_embd, n_ff, U);
    ggml_build_forward_expand(mg.gf, ggml_cpy(mg.ctx, acc,
            ggml_view_2d(mg.ctx, t_pg_, n_embd, 1, t_pg_->nb[1], 0)));
    mg.ga = ggml_gallocr_new(w_.buft());
    if (!mg.ga || !ggml_gallocr_alloc_graph(mg.ga, mg.gf)) {
        if (mg.ga) ggml_gallocr_free(mg.ga);
        ggml_free(mg.ctx);
        mg = moe_graph{};
        static bool warned = false;
        if (!warned) { warned = true; fprintf(stderr, "[qwfn] mul_mat_id MoE graph: no device memory; using the per-expert path\n"); }
        return false;
    }
    return true;
}

void engine::set_embeddings(int32_t pos, const float * emb, int32_t n) {
    const int64_t d = hp_.n_embd;
    const size_t base = ov_.size();
    ov_.resize(base + (size_t) n * d);
    memcpy(ov_.data() + base, emb, (size_t) n * d * sizeof(float));
    for (int32_t i = 0; i < n; i++) ov_pos_.push_back(pos + i);
}

void engine::reset() {
    mtp_have_h_ = false; mtp_kv_valid_ = true; mtp_draft_ = -1;
    st_.reset(); n_past_ = 0;
    pool_dirty_ = true;
    if (qbuf_) {
        const int64_t NBmax = qd_.bias->ne[0];
        std::vector<float> ninf(NBmax, -INFINITY);
        ggml_backend_tensor_set(qd_.bias, ninf.data(), 0, ninf.size() * 4);
    }
}

// Everything the decode attention graph needs for this token: the write row,
// the cells and positions of the block holding it, the bias window around it,
// and the block bucket the graph is shaped for. After a prefill the pooled
// keys of every complete block are rebuilt once from the raw cache.
void engine::qsa_decode_prepare(int32_t n_past) {
    const int64_t r      = qsa_ratio_;
    const int64_t NBmax  = qd_.bias->ne[0];
    const int32_t b_last = n_past / (int32_t) r;
    const int64_t n_bid  = (n_past + 1) / r;                 // whole blocks once this token is in

    const int32_t wi = n_past;
    ggml_backend_tensor_set(qd_.write_idx, &wi, 0, 4);
    std::vector<int32_t> mi(r);
    for (int64_t k = 0; k < r; k++) mi[k] = (int32_t) (b_last * r + k);
    ggml_backend_tensor_set(qd_.member_idx, mi.data(), 0, mi.size() * 4);
    int32_t bp[4] = { (int32_t) (b_last * r), (int32_t) (b_last * r), (int32_t) (b_last * r), (int32_t) (b_last * r) };
    ggml_backend_tensor_set(qd_.blk_pos, bp, 0, sizeof bp);
    ggml_backend_tensor_set(qd_.blk_idx, &b_last, 0, 4);
    const float nf = (float) n_past;
    ggml_backend_tensor_set(qd_.npast_f, &nf, 0, 4);

    // Bias: 0 for whole blocks, 1e9 for the incomplete tail block (always
    // visible), -inf past it. Only the window around b_last can have changed.
    float win[3]; int64_t b0 = std::max<int64_t>(0, b_last - 1), n = 0;
    for (int64_t b = b0; b <= b_last + 1 && b < NBmax; b++, n++)
        win[n] = b < n_bid ? 0.0f : (b == b_last ? 1e9f : -INFINITY);
    ggml_backend_tensor_set(qd_.bias, win, (size_t) b0 * 4, (size_t) n * 4);

    int64_t NB = ((b_last + 1 + 255) / 256) * 256;
    NB = std::max<int64_t>(NB, 768);
    NB = std::min<int64_t>(NB, NBmax);
    qd_.n_bucket = NB;
    qd_.k_blocks = std::min<int64_t>(qd_.k_blocks > 0 ? qd_.k_blocks : NB, NB);

    if (pool_dirty_) {
        // A prefill does not maintain the bias: every block it filled still
        // carries the -inf it was reset to, so the top-k would take the
        // window's blocks and then arbitrary ones -- fine below 513 blocks
        // (everything is selected anyway), a random 1.5% of the context at
        // 133K. Seen as ungrounded answers over a long document while the
        // legacy path over the same caches was grounded. Rewrite it whole.
        {
            std::vector<float> full(NBmax, -INFINITY);
            for (int64_t b = 0; b < n_bid && b < NBmax; b++) full[b] = 0.0f;
            if (b_last < NBmax) full[b_last] = (b_last < n_bid) ? 0.0f : 1e9f;
            ggml_backend_tensor_set(qd_.bias, full.data(), 0, full.size() * 4);
        }
        const int64_t n_whole = n_past / r;
        if (n_whole > 0) {
            std::vector<int32_t> bpa((size_t) 4 * n_whole);
            for (int sec = 0; sec < 4; sec++)
                for (int64_t b = 0; b < n_whole; b++) bpa[sec * n_whole + b] = (int32_t) (b * r);
            for (uint32_t il = 0; il < hp_.n_layer; il++) {
                if (!pool_cache_[il]) continue;
                ggml_init_params p{};
                p.mem_size = ggml_tensor_overhead() * 256 + ggml_graph_overhead_custom(256, false); p.no_alloc = true;
                ggml_context * c = ggml_init(p);
                ggml_cgraph * g = ggml_new_graph_custom(c, 256, false);
                graph_builder gb(c, &hp_, &w_); gb.bind(&st_, g, n_past);
                ggml_tensor * t_bp = ggml_new_tensor_1d(c, GGML_TYPE_I32, 4 * n_whole); ggml_set_input(t_bp);
                qd_.pool_cache = pool_cache_[il];
                gb.qsa_pool_rebuild((int) il, qd_, n_whole, t_bp);
                if (!ggml_gallocr_alloc_graph(galloc_gpu_, g)) { fprintf(stderr, "[qwfn] galloc failed\n"); abort(); }
                ggml_backend_tensor_set(t_bp, bpa.data(), 0, bpa.size() * 4);
                if (ggml_backend_graph_compute(w_.backend(), g) != GGML_STATUS_SUCCESS) {
                    fprintf(stderr, "[qwfn] compute failed\n"); abort();
                }
                ggml_free(c);
            }
        }
        pool_dirty_ = false;
    }
}

std::string engine::memory_summary() const {
    std::ostringstream o;
    o << "dense core " << w_.bytes() / 1e9 << " GB on " << w_.dev_name()
      << " | " << st_.summary()
      << " | expert cache " << ec_.capacity_experts() << " RAM blocks, "
      << ec_.capacity_experts_gpu() << " VRAM blocks";
    return o.str();
}

void engine::run_on(ggml_cgraph * gf, bool gpu) {
    ggml_gallocr_t ga = gpu ? galloc_gpu_ : galloc_cpu_;
    ggml_backend_t be = gpu ? w_.backend() : wh_.backend();
    if (!ggml_gallocr_alloc_graph(ga, gf)) { fprintf(stderr, "[qwfn] galloc failed\n"); abort(); }
    if (ggml_backend_graph_compute(be, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "[qwfn] compute failed\n"); abort();
    }
}

const float * engine::eval(const int32_t * hist, int32_t n_hist, int32_t n_new, std::string & err) {
    if (n_new <= 0 || n_new > (int32_t) cfg_.n_batch) {
        err = "n_new out of range (1.." + std::to_string(cfg_.n_batch) + ")"; return nullptr;
    }
    if (n_past_ + n_new > (int32_t) cfg_.n_ctx) { err = "context exhausted"; return nullptr; }

    // The per-call input arena is dominated by kq_mask F16 [n_kv, T] and the QSA
    // bias F32 [n_blocks, T], both linear in n_kv*T. A ubatch that fits at 16K
    // does not fit at 128K: measured 3452 MiB at n_kv=131072, T=2048, which is
    // more device memory than is left after the dense core and the staging. So
    // cap the product and let long contexts shrink the ubatch themselves rather
    // than failing 15 minutes into a prefill.
    // A short prefill is cheaper token by token: the streaming path's cost is
    // the whole expert set per layer regardless of T, so it only pays off once
    // T is large enough to amortise it.
    const bool as_decode = n_new > 1 && n_new <= (int32_t) cfg_.prefill_decode_max;
    // ... and, with a GPU, as ONE batch whose experts come through the cache.
    static const bool no_cbatch = getenv("QWFN_NO_CACHE_BATCH") != nullptr;
    const bool as_cbatch = as_decode && cfg_.cache_batched && scr_buf_ && !cfg_.use_cold_tier && !no_cbatch;

    const int32_t base = n_hist - n_new;
    // Long prompts: layer-major, one expert sweep per n_batch tokens, the
    // attention chunked to ubatch_kv_product inside. QWFN_LEGACY_PREFILL=1
    // keeps the per-ubatch sweep for comparison.
    static const bool legacy_prefill = getenv("QWFN_LEGACY_PREFILL") != nullptr;
    const bool streamed = n_new > 1 && !as_decode;
    if (streamed && mtp_on_) mtp_kv_valid_ = false;   // the head has no rows for a streamed prefill (not yet built)
    if (streamed && !prefill_enter(err)) return nullptr;
    bool ok = true;
    if (streamed && !legacy_prefill) {
        for (int32_t off = 0; off < n_new && ok; ) {
            const int32_t take = std::min<int32_t>(n_new - off, (int32_t) cfg_.n_batch);
            ok = eval_prefill_big(hist, base + off + take, take, err);
            off += take;
        }
    } else {
        for (int32_t off = 0; off < n_new && ok; ) {
            const int32_t take = as_cbatch ? n_new : as_decode ? 1 : std::min(n_new - off, max_ubatch(n_past_));
            ok = eval_batch(hist, base + off + take, take, err, as_cbatch);
            off += take;
        }
    }
    if (streamed) prefill_leave();
    return ok ? logits_.data() : nullptr;
}


int32_t engine::max_ubatch(int32_t n_past) const {
    if (cfg_.ubatch_kv_product == 0) return (int32_t) cfg_.n_batch;
    const int64_t t = (int64_t) cfg_.ubatch_kv_product / std::max<int64_t>(n_past + 1, 1);
    int32_t take = (int32_t) std::min<int64_t>(t, (int64_t) cfg_.n_batch);
    take &= ~63;                                  // keep ubatches 64-aligned
    return std::max(take, 64);
}


bool engine::build_attn_inputs(int64_t n_past_c, int64_t Tc, attn_inputs & ai, std::string & err) {
    const int64_t n_kv = n_past_c + Tc;
    ggml_init_params ip{}; ip.mem_size = ggml_tensor_overhead() * 16; ip.no_alloc = true;
    ai.ctx = ggml_init(ip);
    ai.kq_mask = ggml_new_tensor_2d(ai.ctx, GGML_TYPE_F16, n_kv, Tc);
    ai.ratio = cfg_.use_qsa ? qsa_ratio_ : 0;
    const uint32_t ratio = ai.ratio;
    const int64_t n_blocks = ratio ? (n_kv + ratio - 1) / ratio : 0;
    if (ratio) {
        ai.qsa.cell_blk  = ggml_new_tensor_1d(ai.ctx, GGML_TYPE_I32, n_kv);
        ai.qsa.blk_cells = ggml_new_tensor_1d(ai.ctx, GGML_TYPE_I32, ratio * n_blocks);
        ai.qsa.blk_pos   = ggml_new_tensor_1d(ai.ctx, GGML_TYPE_I32, 4 * n_blocks);
        ai.qsa.bias      = ggml_new_tensor_2d(ai.ctx, GGML_TYPE_F32, n_blocks, Tc);
        ai.qsa.ratio     = ratio;
        ai.qsa.n_blocks  = n_blocks;
    }
    ai.buf = ggml_backend_alloc_ctx_tensors_from_buft(ai.ctx, w_.buft());
    if (!ai.buf) { ggml_free(ai.ctx); ai.ctx = nullptr; err = "failed to allocate per-chunk attention inputs"; return false; }
    // These are O(n_kv * Tc) per chunk and at 128K ran for a third of the
    // prefill when written element by element. Row-wise fills: a row is a
    // run of zeros then a run of -inf (F16 0x0000 / 0xFC00).
    {
        std::vector<uint16_t> m((size_t) n_kv * Tc);
        const uint16_t ninf = f16_of(-INFINITY);
        for (int64_t i = 0; i < Tc; i++) {
            uint16_t * row = m.data() + i * n_kv;
            const int64_t vis = std::min<int64_t>(n_kv, n_past_c + i + 1);
            memset(row, 0, (size_t) vis * 2);
            std::fill(row + vis, row + n_kv, ninf);
        }
        ggml_backend_tensor_set(ai.kq_mask, m.data(), 0, m.size() * 2);
    }
    if (ratio) {
        const int64_t n_bid = n_kv / ratio;
        const bool have_dead = n_bid < n_blocks;
        const int64_t dead = have_dead ? n_bid : n_blocks - 1;
        std::vector<int32_t> cb(n_kv), bc((size_t) ratio * n_blocks, 0), bp((size_t) 4 * n_blocks, 0);
        std::vector<float> bi((size_t) n_blocks * Tc);
        for (int64_t j = 0; j < n_bid * (int64_t) ratio; j++) cb[j] = (int32_t) (j / ratio);
        for (int64_t j = n_bid * (int64_t) ratio; j < n_kv; j++) cb[j] = (int32_t) dead;
        for (int64_t b = 0; b < n_bid; b++) {
            for (uint32_t k = 0; k < ratio; k++) bc[b * ratio + k] = (int32_t) (b * ratio + k);
            for (int sec = 0; sec < 4; sec++) bp[sec * n_blocks + b] = (int32_t) (b * ratio);
        }
        // Row i: 0 for whole blocks before the tail, 1e9 from the tail block on
        // (still whole), -inf for blocks past n_bid; the dead block gets 1e9.
        for (int64_t i = 0; i < Tc; i++) {
            float * row = bi.data() + i * n_blocks;
            const int64_t q = n_past_c + i;
            const int64_t tail_b = std::min<int64_t>(n_bid, ((q + 1) / ratio));   // first block at/after the tail
            std::fill(row, row + tail_b, 0.0f);
            std::fill(row + tail_b, row + n_bid, 1e9f);
            std::fill(row + n_bid, row + n_blocks, -INFINITY);
            if (have_dead) row[dead] = 1e9f;
        }
        ggml_backend_tensor_set(ai.qsa.cell_blk,  cb.data(), 0, cb.size() * 4);
        ggml_backend_tensor_set(ai.qsa.blk_cells, bc.data(), 0, bc.size() * 4);
        ggml_backend_tensor_set(ai.qsa.blk_pos,   bp.data(), 0, bp.size() * 4);
        ggml_backend_tensor_set(ai.qsa.bias,      bi.data(), 0, bi.size() * 4);
    }
    return true;
}

bool engine::load_predictor(const std::string & path, std::string & err) {
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) { err = "cannot open predictor " + path; return false; }
    char magic[4]; uint32_t ver = 0, nl = 0, ne = 0, nx = 0, first = 0;
    if (fread(magic, 1, 4, f) != 4 || memcmp(magic, "QWPR", 4) != 0 || fread(&ver, 4, 1, f) != 1 ||
        fread(&nl, 4, 1, f) != 1 || fread(&ne, 4, 1, f) != 1 || fread(&nx, 4, 1, f) != 1 || fread(&first, 4, 1, f) != 1) {
        fclose(f); err = "bad predictor header"; return false;
    }
    if (nl != hp_.n_layer || ne != (uint32_t) hp_.n_embd || nx != hp_.n_expert || first >= nl) {
        fclose(f); err = "predictor shape mismatch"; return false;
    }
    ggml_init_params pp{}; pp.mem_size = ggml_tensor_overhead() * 2 * nl + 1024; pp.no_alloc = true;
    predctx_ = ggml_init(pp);
    pred_w_.assign(nl, nullptr); pred_b_.assign(nl, nullptr);
    for (uint32_t il = first; il < nl; il++) {
        pred_w_[il] = ggml_new_tensor_2d(predctx_, GGML_TYPE_F16, ne, nx);
        pred_b_[il] = ggml_new_tensor_1d(predctx_, GGML_TYPE_F32, nx);
    }
    predbuf_ = ggml_backend_alloc_ctx_tensors_from_buft(predctx_, w_.buft());
    if (!predbuf_) { fclose(f); err = "failed to allocate the routing predictor"; return false; }
    std::vector<uint8_t> wbuf((size_t) ne * nx * 2); std::vector<float> bias(nx);
    for (uint32_t il = first; il < nl; il++) {
        if (fread(wbuf.data(), 1, wbuf.size(), f) != wbuf.size() || fread(bias.data(), 4, nx, f) != nx) {
            fclose(f); err = "predictor file truncated"; return false;
        }
        ggml_backend_tensor_set(pred_w_[il], wbuf.data(), 0, wbuf.size());
        ggml_backend_tensor_set(pred_b_[il], bias.data(), 0, (size_t) nx * 4);
    }
    fclose(f);
    fprintf(stderr, "[qwfn] routing predictor: layers %u-%u, %.0f MB on the device\n", first, nl - 1,
            (double) (nl - first) * ((double) ne * nx * 2 + nx * 4) / 1e6);
    return true;
}

bool engine::dump_routers(const std::string & dir, std::string & err) {
    const int64_t n_embd = hp_.n_embd, nx = hp_.n_expert;
    ggml_init_params ip{}; ip.mem_size = ggml_tensor_overhead() * 4; ip.no_alloc = true;
    ggml_context * tc = ggml_init(ip);
    ggml_tensor * dst = ggml_new_tensor_2d(tc, GGML_TYPE_F32, n_embd, nx);
    ggml_backend_buffer_t tb = ggml_backend_alloc_ctx_tensors_from_buft(tc, w_.buft());
    if (!tb) { ggml_free(tc); err = "router dump: no device memory"; return false; }
    std::vector<float> host((size_t) n_embd * nx);
    for (uint32_t il = 1; il < hp_.n_layer; il++) {
        ggml_tensor * W = w_.get(("blk." + std::to_string(il) + ".ffn_gate_inp.weight").c_str());
        if (!W) continue;
        ggml_init_params gp{}; gp.mem_size = ggml_tensor_overhead() * 8 + ggml_graph_overhead(); gp.no_alloc = true;
        ggml_context * c = ggml_init(gp);
        ggml_cgraph * g = ggml_new_graph(c);
        ggml_build_forward_expand(g, ggml_cpy(c, W, dst));   // dequantises into dst
        if (!ggml_gallocr_alloc_graph(galloc_gpu_, g) ||
            ggml_backend_graph_compute(w_.backend(), g) != GGML_STATUS_SUCCESS) {
            ggml_free(c); ggml_backend_buffer_free(tb); ggml_free(tc); err = "router dump: compute failed"; return false;
        }
        ggml_free(c);
        ggml_backend_tensor_get(dst, host.data(), 0, host.size() * sizeof(float));
        const std::string p = dir + "/router_L" + std::to_string(il) + ".bin";
        FILE * f = fopen(p.c_str(), "wb");
        if (!f) { err = "cannot write " + p; ggml_backend_buffer_free(tb); ggml_free(tc); return false; }
        fwrite(host.data(), sizeof(float), host.size(), f); fclose(f);
    }
    ggml_backend_buffer_free(tb); ggml_free(tc);
    return true;
}

void engine::dump_decode_token() {
    if (dump_dec_dir_.empty() || !t_xdec_) return;
    if (!routers_dec_dumped_) { std::string e; if (!dump_routers(dump_dec_dir_, e)) { fprintf(stderr, "[qwfn] %s\n", e.c_str()); dump_dec_dir_.clear(); return; } routers_dec_dumped_ = true; }
    if (dump_dec_f_.empty()) dump_dec_f_.assign(hp_.n_layer, nullptr);
    const int64_t n_embd = hp_.n_embd, U = hp_.n_expert_used;
    std::vector<float> x((size_t) n_embd * hp_.n_layer);
    ggml_backend_tensor_get(t_xdec_, x.data(), 0, x.size() * sizeof(float));
    std::vector<ggml_fp16_t> xh(n_embd), wh(U); std::vector<uint16_t> ids(U);
    for (uint32_t L = 0; L + 1 < hp_.n_layer; L++) {
        const uint32_t il = L + 1;   // file L{il}: input predicting layer il, routing of layer il
        if (!dump_dec_f_[il]) {
            const std::string p = dump_dec_dir_ + "/L" + std::to_string(il) + ".bin";
            dump_dec_f_[il] = fopen(p.c_str(), "ab");
            if (!dump_dec_f_[il]) { fprintf(stderr, "[qwfn] cannot open %s; decode dump off\n", p.c_str()); dump_dec_dir_.clear(); return; }
        }
        ggml_fp32_to_fp16_row(x.data() + (size_t) L * n_embd, xh.data(), n_embd);
        for (int64_t k = 0; k < U; k++) ids[k] = (uint16_t) tok_sel_[(size_t) il * U + k];
        ggml_fp32_to_fp16_row(tok_w_.data() + (size_t) il * U, wh.data(), U);
        fwrite(xh.data(), 2, n_embd, dump_dec_f_[il]); fwrite(ids.data(), 2, U, dump_dec_f_[il]); fwrite(wh.data(), 2, U, dump_dec_f_[il]);
    }
}

// One record per sampled token: x as F16[n_embd], ids as u16[U], gates as F16[U].
void engine::dump_layer(uint32_t il, int64_t T) {
    if (dump_dir_.empty() || il == 0 || !t_xdump_) return;
    if (dump_f_.empty()) dump_f_.assign(hp_.n_layer, nullptr);
    if (!dump_f_[il]) {
        const std::string p = dump_dir_ + "/L" + std::to_string(il) + ".bin";
        dump_f_[il] = fopen(p.c_str(), "ab");
        if (!dump_f_[il]) { fprintf(stderr, "[qwfn] cannot open %s; dump off\n", p.c_str()); dump_dir_.clear(); return; }
    }
    const int64_t n_embd = hp_.n_embd, U = hp_.n_expert_used;
    std::vector<float> x((size_t) n_embd * T);
    ggml_backend_tensor_get(t_xdump_, x.data(), 0, x.size() * sizeof(float));
    std::vector<ggml_fp16_t> xh(n_embd), wh(U); std::vector<uint16_t> ids(U);
    for (int64_t t = 0; t < T; t += dump_stride_) {
        ggml_fp32_to_fp16_row(x.data() + (size_t) t * n_embd, xh.data(), n_embd);
        for (int64_t k = 0; k < U; k++) ids[k] = (uint16_t) sel_[t * U + k];
        ggml_fp32_to_fp16_row(wgt_.data() + (size_t) t * U, wh.data(), U);
        fwrite(xh.data(), 2, n_embd, dump_f_[il]); fwrite(ids.data(), 2, U, dump_f_[il]); fwrite(wh.data(), 2, U, dump_f_[il]);
    }
}

void engine::save_work_set(work_set & ws) const {
    ws.xdump = t_xdump_;
    ws.res[0] = res_[0]; ws.res[1] = res_[1];
    ws.cur = t_cur_; ws.emb = t_emb_; ws.sh = t_sh_; ws.pg = t_pg_; ws.pc = t_pc_; ws.ple = t_ple_;
    ws.inject = t_inject_; ws.sel = t_sel_; ws.w = t_w_; ws.tok = inp_tok_; ws.pos = inp_pos_; ws.plei = inp_ple_;
}
void engine::load_work_set(const work_set & ws) {
    t_xdump_ = ws.xdump;
    res_[0] = ws.res[0]; res_[1] = ws.res[1];
    t_cur_ = ws.cur; t_emb_ = ws.emb; t_sh_ = ws.sh; t_pg_ = ws.pg; t_pc_ = ws.pc; t_ple_ = ws.ple;
    t_inject_ = ws.inject; t_sel_ = ws.sel; t_w_ = ws.w; inp_tok_ = ws.tok; inp_pos_ = ws.pos; inp_ple_ = ws.plei;
}

// A streamed prefill is about to run: hand the expert tier's dynamic buffer
// back to the device, then take the prefill's memory from it -- the
// n_batch work set, the MoE chunk allocator; the staging follows on the first
// load_layer. Undone by prefill_leave().
bool engine::prefill_enter(std::string & err) {
    if (in_prefill_) return true;
    ec_.lend_begin();
    const int64_t n_embd = hp_.n_embd, hc = hp_.hc_count, B = cfg_.n_batch, U = hp_.n_expert_used, PH = hp_.ple_n_head();
    ggml_init_params wp{}; wp.mem_size = ggml_tensor_overhead() * 64; wp.no_alloc = true;
    pwctx_ = ggml_init(wp);
    work_set ws;
    ws.res[0] = ggml_new_tensor_3d(pwctx_, GGML_TYPE_F32, n_embd, hc, B);
    ws.res[1] = ggml_new_tensor_3d(pwctx_, GGML_TYPE_F32, n_embd, hc, B);
    ws.cur    = ggml_new_tensor_2d(pwctx_, GGML_TYPE_F32, n_embd, B);
    ws.emb    = ggml_new_tensor_2d(pwctx_, GGML_TYPE_F32, n_embd, B);
    ws.sh     = ggml_new_tensor_2d(pwctx_, GGML_TYPE_F32, n_embd, B);
    ws.pg     = ggml_new_tensor_2d(pwctx_, GGML_TYPE_F32, n_embd, B);
    ws.pc     = ggml_new_tensor_2d(pwctx_, GGML_TYPE_F32, n_embd, B);
    ws.ple    = ggml_new_tensor_2d(pwctx_, GGML_TYPE_F32, n_embd, B);
    ws.inject = ggml_new_tensor_2d(pwctx_, GGML_TYPE_F32, hc, B);
    ws.sel    = ggml_new_tensor_2d(pwctx_, GGML_TYPE_I32, U, B);
    ws.w      = ggml_new_tensor_2d(pwctx_, GGML_TYPE_F32, U, B);
    ws.tok    = ggml_new_tensor_1d(pwctx_, GGML_TYPE_I32, B);
    ws.pos    = ggml_new_tensor_1d(pwctx_, GGML_TYPE_I32, B * 4);
    ws.plei   = ggml_new_tensor_1d(pwctx_, GGML_TYPE_I32, PH * B);
    if (!dump_dir_.empty()) ws.xdump = ggml_new_tensor_2d(pwctx_, GGML_TYPE_F32, n_embd, B);
    if (pf_.on_device()) {
        d_cur_     = ggml_new_tensor_2d(pwctx_, GGML_TYPE_F32, n_embd, B);
        d_partial_ = ggml_new_tensor_2d(pwctx_, GGML_TYPE_F32, n_embd, B);
    }
    pwbuf_ = ggml_backend_alloc_ctx_tensors_from_buft(pwctx_, w_.buft());
    if (!pwbuf_) { ggml_free(pwctx_); pwctx_ = nullptr; ec_.lend_end(); err = "failed to allocate the prefill work set"; return false; }
    load_work_set(ws);
    if (pf_.on_device()) { galloc_pf_dyn_ = ggml_gallocr_new(w_.buft()); galloc_pf_ = galloc_pf_dyn_; }
    in_prefill_ = true;
    return true;
}

void engine::prefill_leave() {
    if (!in_prefill_) return;
    ec_.settle_promotions();
    pf_.release_device();
    if (galloc_pf_dyn_) { ggml_gallocr_free(galloc_pf_dyn_); galloc_pf_dyn_ = nullptr; galloc_pf_ = nullptr; }
    // The shared allocator grew to the attention chunk's arena; start it over
    // so decode's graphs get a small one.
    if (galloc_gpu_) { ggml_gallocr_free(galloc_gpu_); galloc_gpu_ = ggml_gallocr_new(w_.buft()); }
    load_work_set(dec_ws_);
    d_cur_ = d_partial_ = nullptr;
    if (pwbuf_) { ggml_backend_buffer_free(pwbuf_); pwbuf_ = nullptr; }
    if (pwctx_) { ggml_free(pwctx_); pwctx_ = nullptr; }
    in_prefill_ = false;
    ec_.lend_end();
    if (ec_.tier_epoch() != tier_epoch_seen_) {
        // The dynamic tier moved: every replayed graph that folds an expert
        // matmul over it holds stale pointers. Rebuild them all next token.
        for (auto & lg : gA_) { if (lg.ga) ggml_gallocr_free(lg.ga); if (lg.ctx) ggml_free(lg.ctx); lg = layer_graph{}; }
        for (auto & mg : gM_) { if (mg.ga) ggml_gallocr_free(mg.ga); if (mg.ctx) ggml_free(mg.ctx); mg = moe_graph{}; }
        std::fill(gA_bucket_.begin(), gA_bucket_.end(), -1);
        tier_epoch_seen_ = ec_.tier_epoch();
    }
}

// Layer-major prefill of T (<= n_batch) tokens: for every layer, graph A over
// all T tokens in compute chunks (attention layers chunked further to the
// n_kv * T cap), then ONE sweep of the layer's experts serving all T through
// mul_mat_id in the same chunks. The sweep is what a prefill costs, so its
// count per prompt goes from ceil(n / ubatch), with the ubatch shrinking as
// the context grows, to ceil(n / n_batch).
bool engine::eval_prefill_big(const int32_t * hist, int32_t n_hist, int32_t T, std::string & err) {
    const int64_t n_embd = hp_.n_embd, hc = hp_.hc_count, U = hp_.n_expert_used;
    const int64_t n_past = n_past_, PH = hp_.ple_n_head();
    const auto t0 = std::chrono::steady_clock::now();
    pool_dirty_ = true;
    int sections[4] = { hp_.mrope_sections[0], hp_.mrope_sections[1],
                        hp_.mrope_sections[2], hp_.mrope_sections[3] };
    const int64_t Tm = std::min<int64_t>(T, std::max<uint32_t>(64, cfg_.prefill_chunk));

    // ---- inputs for all T tokens ------------------------------------------
    ggml_backend_tensor_set(inp_tok_, hist + (n_hist - T), 0, (size_t) T * 4);
    {
        std::vector<int32_t> rows((size_t) PH * T);
        for (int64_t i = 0; i < T; i++) {
            const ple_rows r = ple_rows_for(hp_, hist, n_hist, (n_hist - T) + i);
            for (uint32_t h = 0; h < r.n; h++) rows[i * PH + h] = (int32_t) r.row[h];
        }
        ggml_backend_tensor_set(inp_ple_,   rows.data(), 0, rows.size() * 4);
        ggml_backend_tensor_set(h_ple_idx_, rows.data(), 0, rows.size() * 4);
    }
    auto new_ctx = [&](ggml_context ** c, ggml_cgraph ** g) {
        ggml_init_params p{};
        p.mem_size = ggml_tensor_overhead() * 16384 + ggml_graph_overhead_custom(16384, false);
        p.no_alloc = true;
        *c = ggml_init(p);
        *g = ggml_new_graph_custom(*c, 16384, false);
    };
    {   // PLE rows and token embeddings gathered on the host, then uploaded
        ggml_backend_tensor_set(h_tok_, hist + (n_hist - T), 0, (size_t) T * 4);
        ggml_context * c; ggml_cgraph * g; new_ctx(&c, &g);
        ggml_tensor * idx = ggml_view_1d(c, h_ple_idx_, PH * T, 0);
        ggml_tensor * e = ggml_get_rows(c, wh_.get("per_layer_token_embd.weight"), idx);
        e = ggml_reshape_2d(c, e, hp_.d_ple * PH, T);
        ggml_build_forward_expand(g, ggml_cpy(c, e, ggml_view_2d(c, h_ple_, n_embd, T, h_ple_->nb[1], 0)));
        ggml_tensor * te = ggml_get_rows(c, wh_.get("token_embd.weight"), ggml_view_1d(c, h_tok_, T, 0));
        ggml_build_forward_expand(g, ggml_cpy(c, te, ggml_view_2d(c, h_emb_, n_embd, T, h_emb_->nb[1], 0)));
        run_on(g, false); ggml_free(c);
        ggml_backend_tensor_get(h_ple_, xfer_.data(), 0, (size_t) n_embd * T * sizeof(float));
        ggml_backend_tensor_set(t_ple_, xfer_.data(), 0, (size_t) n_embd * T * sizeof(float));
    }
    const bool has_ov = [&] {
        for (int32_t p : ov_pos_) if (p >= n_past && p < n_past + T) return true;
        return false;
    }();
    ggml_backend_tensor_get(h_emb_, xfer_.data(), 0, (size_t) n_embd * T * sizeof(float));
    if (has_ov) {
        for (size_t k = 0; k < ov_pos_.size(); k++) {
            const int32_t p = ov_pos_[k];
            if (p < n_past || p >= n_past + T) continue;
            memcpy(xfer_.data() + (size_t) (p - n_past) * n_embd,
                   ov_.data()  + k * (size_t) n_embd, (size_t) n_embd * sizeof(float));
        }
    }
    ggml_backend_tensor_set(t_emb_, xfer_.data(), 0, (size_t) n_embd * T * sizeof(float));
    {
        ggml_context * c; ggml_cgraph * g; new_ctx(&c, &g);
        ggml_tensor * r = ggml_repeat_4d(c, ggml_reshape_3d(c, ggml_view_2d(c, t_emb_, n_embd, T, t_emb_->nb[1], 0), n_embd, 1, T),
                                         n_embd, hc, T, 1);
        ggml_build_forward_expand(g, ggml_cpy(c, r,
                ggml_view_3d(c, res_[0], n_embd, hc, T, res_[0]->nb[1], res_[0]->nb[2], 0)));
        run_on(g, true); ggml_free(c);
    }

    int cur_res = 0;
    bool pending = false;
    auto vres = [&](ggml_context * c, int which, int64_t off, int64_t n) {
        return ggml_view_3d(c, res_[which], n_embd, hc, n, res_[which]->nb[1], res_[which]->nb[2], (size_t) off * res_[which]->nb[2]);
    };
    auto v2 = [&](ggml_context * c, ggml_tensor * t, int64_t off, int64_t n) {
        return ggml_view_2d(c, t, t->ne[0], n, t->nb[1], (size_t) off * t->nb[1]);
    };

    for (uint32_t il = 0; il < hp_.n_layer; il++) {
        const bool is_ple  = std::find(hp_.ple_layers.begin(), hp_.ple_layers.end(), (int32_t) il) != hp_.ple_layers.end();
        const bool is_attn = hp_.is_attn_layer(il);

        // ---- graph A over all T, chunked ------------------------------------
        for (int64_t off = 0; off < T; ) {
            int64_t Tc = std::min<int64_t>(Tm, T - off);
            if (is_attn) Tc = std::min<int64_t>(Tc, max_ubatch((int32_t) (n_past + off)));
            const int64_t n_past_c = n_past + off;
            attn_inputs ai;
            if (is_attn && !build_attn_inputs(n_past_c, Tc, ai, err)) return false;
            {
                std::vector<int32_t> pos((size_t) 4 * Tc, 0);
                for (int64_t i = 0; i < Tc; i++) pos[i] = pos[Tc + i] = pos[2 * Tc + i] = (int32_t) (n_past_c + i);
                ggml_backend_tensor_set(inp_pos_, pos.data(), 0, pos.size() * 4);
            }
            const auto ta0 = std::chrono::steady_clock::now();
            ggml_context * c; ggml_cgraph * g; new_ctx(&c, &g);
            graph_builder gb(c, &hp_, &w_); gb.bind(&st_, g, n_past_c);
            ggml_tensor * r = vres(c, cur_res, off, Tc);
            if (pending) {
                ggml_tensor * tot = ggml_add(c, ggml_add(c, v2(c, t_sh_, off, Tc), v2(c, t_pg_, off, Tc)), v2(c, t_pc_, off, Tc));
                r = gb.hc_combine(r, tot, v2(c, t_inject_, off, Tc));
            }
            if (t_xdump_ && il > 0) {   // the predictor's input for this layer, see dump_layer
                ggml_tensor * xin = gb.hc_mix(r, il, /*ffn=*/true, nullptr);
                ggml_build_forward_expand(g, ggml_cpy(c, xin, v2(c, t_xdump_, off, Tc)));
            }
            if (is_ple) r = gb.ple(v2(c, t_ple_, off, Tc), r, il);
            ggml_tensor * inject = nullptr;
            ggml_tensor * cur = gb.hc_mix(r, il, /*ffn=*/false, &inject);
            cur = is_attn
                ? gb.sparse_attn(cur, ggml_view_1d(c, inp_pos_, Tc * 4, 0), ai.kq_mask, sections, il, ai.ratio ? &ai.qsa : nullptr)
                : gb.deltanet(cur, il);
            r = gb.hc_combine(r, cur, inject);
            ggml_tensor * cur2 = gb.hc_mix(r, il, /*ffn=*/true, &inject);
            ggml_tensor * sl = nullptr, * wt = nullptr;
            gb.moe_route(cur2, il, &sl, &wt);
            ggml_tensor * sh = gb.shared_expert(cur2, il);
            ggml_build_forward_expand(g, ggml_cpy(c, r,      vres(c, 1 - cur_res, off, Tc)));
            ggml_build_forward_expand(g, ggml_cpy(c, cur2,   v2(c, t_cur_, off, Tc)));
            ggml_build_forward_expand(g, ggml_cpy(c, inject, v2(c, t_inject_, off, Tc)));
            ggml_build_forward_expand(g, ggml_cpy(c, sl,     v2(c, t_sel_, off, Tc)));
            ggml_build_forward_expand(g, ggml_cpy(c, wt,     v2(c, t_w_, off, Tc)));
            ggml_build_forward_expand(g, ggml_cpy(c, sh,     v2(c, t_sh_, off, Tc)));
            run_on(g, true);
            ggml_free(c);
            ai.release();
            t_pf_graphA += std::chrono::duration<double>(std::chrono::steady_clock::now() - ta0).count();
            off += Tc;
        }
        cur_res = 1 - cur_res;
        pending = true;
        ggml_backend_tensor_get(t_sel_, sel_.data(), 0, (size_t) U * T * sizeof(int32_t));
        ggml_backend_tensor_get(t_w_,   wgt_.data(), 0, (size_t) U * T * sizeof(float));
        if (t_xdump_) {
            if (!routers_dumped_) { if (!dump_routers(dump_dir_, err)) return false; routers_dumped_ = true; }
            dump_layer(il, T);
        }

        // ---- MoE: one sweep of this layer's experts, all T tokens -----------
        if (pf_.on_device()) ggml_backend_synchronize(w_.backend());
        const auto tr0 = std::chrono::steady_clock::now();
        if (!pf_.load_layer(il, err)) return false;
        t_pf_read += std::chrono::duration<double>(std::chrono::steady_clock::now() - tr0).count();
        pf_.prefetch_layer((il + 1) % hp_.n_layer);
        const bool on_gpu = pf_.on_device();
        const tier_view staged = pf_.staged_tier();
        for (int64_t off = 0; off < T; off += Tm) {
            const int64_t Tc = std::min<int64_t>(Tm, T - off);
            const auto tm0 = std::chrono::steady_clock::now();
            if (!on_gpu) {
                ggml_backend_tensor_get(t_cur_, xfer_.data(), (size_t) off * n_embd * 4, (size_t) Tc * n_embd * 4);
                ggml_backend_tensor_set(h_cur_, xfer_.data(), (size_t) off * n_embd * 4, (size_t) Tc * n_embd * 4);
            }
            ggml_context * c; ggml_cgraph * g; new_ctx(&c, &g);
            ggml_tensor * t_ids = ggml_new_tensor_2d(c, GGML_TYPE_I32, U, Tc);   ggml_set_input(t_ids);
            ggml_tensor * t_wt  = ggml_new_tensor_3d(c, GGML_TYPE_F32, 1, U, Tc); ggml_set_input(t_wt);
            ggml_tensor * x   = on_gpu ? v2(c, t_cur_, off, Tc) : v2(c, h_cur_, off, Tc);
            ggml_tensor * out = on_gpu ? v2(c, t_pc_,  off, Tc) : v2(c, h_partial_, off, Tc);
            ggml_tensor * o = moe_id_graph(c, staged, t_ids, t_wt, x, n_embd, hp_.n_ff_exp, (int) U, Tc);
            ggml_build_forward_expand(g, ggml_cpy(c, o, out));
            if (!ggml_gallocr_alloc_graph(on_gpu ? galloc_pf_ : galloc_cpu_, g)) {
                err = "prefill galloc failed"; ggml_free(c); return false;
            }
            ggml_backend_tensor_set(t_ids, sel_.data() + (size_t) off * U, 0, (size_t) U * Tc * sizeof(int32_t));
            ggml_backend_tensor_set(t_wt,  wgt_.data() + (size_t) off * U, 0, (size_t) U * Tc * sizeof(float));
            if (ggml_backend_graph_compute(on_gpu ? w_.backend() : wh_.backend(), g) != GGML_STATUS_SUCCESS) {
                err = "prefill compute failed"; ggml_free(c); return false;
            }
            ggml_free(c);
            if (!on_gpu) {
                ggml_backend_tensor_get(h_partial_, xfer_.data(), (size_t) off * n_embd * 4, (size_t) Tc * n_embd * 4);
                ggml_backend_tensor_set(t_pc_, xfer_.data(), (size_t) off * n_embd * 4, (size_t) Tc * n_embd * 4);
            }
            t_pf_moe += std::chrono::duration<double>(std::chrono::steady_clock::now() - tm0).count();
        }
        ggml_backend_tensor_set(t_pg_, zeros_.data(), 0, (size_t) n_embd * T * sizeof(float));

        if (cfg_.prefill_warm > 0) {
            const auto tw0 = std::chrono::steady_clock::now();
            std::vector<expert_cache::warm_item> items;
            std::vector<int32_t> pos(hp_.n_expert, -1);
            for (int64_t t = T - 1; t >= 0; t--) {
                for (int64_t j = 0; j < U; j++) {
                    const int32_t e = sel_[t * U + j];
                    if (e < 0 || e >= (int32_t) hp_.n_expert) continue;
                    if (pos[e] >= 0) { items[pos[e]].count++; continue; }
                    if (items.size() >= cfg_.prefill_warm) continue;
                    expert_cache::warm_item it;
                    it.expert = (uint32_t) e; it.count = 1;
                    for (int q = 0; q < EXPERT_NPARTS; q++) it.part[q] = pf_.host_part_ptr((uint32_t) e, (expert_part) q);
                    pos[e] = (int32_t) items.size();
                    items.push_back(it);
                }
            }
            ec_.warm(il, items.data(), (uint32_t) items.size(), cfg_.prefill_warm_vram);
            t_warm += std::chrono::duration<double>(std::chrono::steady_clock::now() - tw0).count();
        }
    }

    // ---- head: logits for the last position -------------------------------
    {
        ggml_context * c; ggml_cgraph * g; new_ctx(&c, &g);
        graph_builder gb(c, &hp_, &w_); gb.bind(&st_, g, n_past);
        ggml_tensor * r = vres(c, cur_res, T - 1, 1);
        ggml_tensor * tot = ggml_add(c, ggml_add(c, v2(c, t_sh_, T - 1, 1), v2(c, t_pg_, T - 1, 1)), v2(c, t_pc_, T - 1, 1));
        r = gb.hc_combine(r, tot, v2(c, t_inject_, T - 1, 1));
        ggml_tensor * o = gb.hc_mix(r, -1, false, nullptr);
        ggml_tensor * logits = ggml_mul_mat(c, w_.get("output.weight"), o);
        ggml_set_output(logits);
        ggml_build_forward_expand(g, logits);
        run_on(g, true);
        ggml_backend_tensor_get(logits, logits_.data(), 0, logits_.size() * sizeof(float));
        ggml_free(c);
    }
    ec_.settle_promotions();
    n_past_ += T;
    const double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    t_prefill += dt; n_prefill += T;
    return true;
}

bool engine::eval_batch(const int32_t * hist, int32_t n_hist, int32_t T, std::string & err,
                        bool cbatch) {
    const int64_t n_embd = hp_.n_embd, hc = hp_.hc_count, U = hp_.n_expert_used;
    const int64_t n_past = n_past_, n_kv = n_past + T, PH = hp_.ple_n_head();
    // Unit-test hook: the batched path must reduce exactly to the decode path
    // at T == 1, so forcing it there checks the permutation logic in isolation.
    static const bool force_batched = getenv("QWFN_FORCE_BATCHED") != nullptr;
    const bool decode = (T == 1) && !force_batched;
    cbatch = cbatch && !decode && T > 1 && scr_buf_ != nullptr;
    const auto t0 = std::chrono::steady_clock::now();
    // Decode attention with per-token cost flat in the context; anything else
    // (prefill, dense attention) uses the mask-based path and leaves the pooled
    // block keys to be rebuilt on the next decode token.
    const bool use_qd = decode && qbuf_ != nullptr && cfg_.use_qsa;
    if (!decode) pool_dirty_ = true;

    // Decode MoE on the GPU: by default one mul_mat_id per part over the VRAM
    // tier, computed INSIDE the next layer's graph A (which folds it into the
    // residual anyway). That removes a graph submission and a stream sync per
    // layer, and keeps the number of live CUDA graphs at the 36 replayed
    // layers -- ggml caches at most 64, and a separate persistent MoE graph
    // per layer (84 total) made it evict and never capture anything.
    // QWFN_LEGACY_MOE=1 is the per-expert path; QWFN_CHECK_MOE=1 validates the
    // mul_mat_id graph against it (and so runs it separately, not folded).
    static const bool legacy_moe = getenv("QWFN_LEGACY_MOE") != nullptr;
    static const bool check_moe  = getenv("QWFN_CHECK_MOE") != nullptr;
    const bool moe_by_id = !legacy_moe && !cfg_.use_cold_tier;
    // In-graph VRAM MoE: graph A of layer L runs L's VRAM-resident routed
    // experts itself. The router's ids index two device tables (tier slot and
    // 1/0 mask by expert id, uploaded whenever the layer's residency changes)
    // so no host round trip stands between the router and the expert
    // matmuls, and the residual passed to the next layer's router prediction
    // then misses only the CPU-served experts. Same mul_mat_id kernels, same
    // summation order as the host-driven graph: a non-resident expert points
    // at slot 0 with weight 0 either way.
    auto moe_in_graph = [&](uint32_t il) {
        return decode && moe_in_graph_ && ec_.gpu_tier(il).n_slots > 0;
    };
    // In-place graph outputs. Give the node that produces an output the persistent
    // tensor's memory (at dst_off bytes) instead of copying into it afterwards:
    // gallocr leaves a node that already has data alone, views derive theirs from
    // it, and the backend computes straight into it. Every read of the same
    // persistent tensor in this graph -- the fold of the previous layer's partials
    // and activation -- precedes the write in dependency order. One token only,
    // contiguous, same type, and the node must be the view root (offset 0).
    auto place = [&](ggml_tensor * t, ggml_tensor * dst, size_t dst_off) -> bool {
        static const bool no_place = getenv("QWFN_NO_INPLACE") != nullptr;
        if (no_place || T != 1 || !dst || !dst->data || !dst->buffer) return false;
        ggml_tensor * root = t; size_t off = 0;
        while (root->view_src) { off += root->view_offs; root = root->view_src; }
        if (off != 0 || root->data || root->type != dst->type || !ggml_is_contiguous(root)) return false;
        if (ggml_nbytes(root) + dst_off > ggml_nbytes(dst)) return false;
        root->data   = (char *) dst->data + dst_off;
        root->buffer = dst->buffer;
        // An output: the allocator must never hand its memory to a child (a
        // single-child op that can run in place would otherwise write over it),
        // and it must be computed even when nothing else in the graph reads it.
        ggml_set_output(root);
        return true;
    };
    auto upload_vtable = [&](uint32_t il) {
        if (!decode || !moe_in_graph_ || il >= hp_.n_layer) return;
        if (ec_.gpu_tier(il).n_slots == 0 || ec_.vram_version(il) == vslot_ver_[il]) return;
        const size_t ne = hp_.n_expert;
        if (p_vslot_) {
            // Pinned staging, queued on the compute stream: after the promotion
            // copies it describes, before the graph that reads it. One buffer
            // is enough: the previous layer's graph ran synchronously.
            ec_.vram_table(il, (int32_t *) p_vslot_->data, (float *) p_vmask_->data);
            ggml_backend_tensor_set_async(w_.backend(), t_vslot_[il], p_vslot_->data, 0, ne * sizeof(int32_t));
            ggml_backend_tensor_set_async(w_.backend(), t_vmask_[il], p_vmask_->data, 0, ne * sizeof(float));
        } else {
            std::vector<int32_t> s(ne); std::vector<float> m(ne);
            ec_.vram_table(il, s.data(), m.data());
            ggml_backend_tensor_set(t_vslot_[il], s.data(), 0, ne * sizeof(int32_t));
            ggml_backend_tensor_set(t_vmask_[il], m.data(), 0, ne * sizeof(float));
        }
        vslot_ver_[il] = ec_.vram_version(il);
    };
    upload_vtable(0);
    // The late fold of layer `prev`: its promoted experts, by slot, from
    // t_gids_/t_gw_ (zeros when there were none: slot 0 with weight 0).
    auto late_fold = [&](ggml_context * c, uint32_t prev, ggml_tensor * pg) {
        if (!moe_in_graph(prev) || n_late_ <= 0) return pg;
        ggml_tensor * ids = ggml_view_2d(c, t_gids_, n_late_, 1, t_gids_->nb[1], 0);
        ggml_tensor * w   = ggml_view_3d(c, t_gw_, 1, n_late_, 1, t_gw_->nb[1], t_gw_->nb[2], 0);
        return ggml_add(c, pg, moe_id_graph(c, ec_.gpu_tier(prev), ids, w, t_cur_, n_embd, hp_.n_ff_exp, n_late_, 1, /*fused_sum=*/true));
    };

    int sections[4] = { hp_.mrope_sections[0], hp_.mrope_sections[1],
                        hp_.mrope_sections[2], hp_.mrope_sections[3] };

    // ---- per-call inputs --------------------------------------------------
    ggml_backend_tensor_set(inp_tok_, hist + (n_hist - T), 0, (size_t) T * 4);

    std::vector<int32_t> rows((size_t) PH * T);
    for (int64_t i = 0; i < T; i++) {
        const ple_rows r = ple_rows_for(hp_, hist, n_hist, (n_hist - T) + i);
        for (uint32_t h = 0; h < r.n; h++) rows[i * PH + h] = (int32_t) r.row[h];
    }
    ggml_backend_tensor_set(inp_ple_,   rows.data(), 0, rows.size() * 4);
    ggml_backend_tensor_set(h_ple_idx_, rows.data(), 0, rows.size() * 4);

    // M-RoPE positions are section-major: all t, then all h, then all w, then 0.
    std::vector<int32_t> pos((size_t) T * 4, 0);
    for (int64_t i = 0; i < T; i++)
        pos[i] = pos[T + i] = pos[2 * T + i] = (int32_t) (n_past + i);
    ggml_backend_tensor_set(inp_pos_, pos.data(), 0, pos.size() * 4);

    // Shapes vary with n_kv, so the mask and the QSA inputs are built per call --
    // except on the decode QSA path, which needs neither.
    const auto t_in0 = std::chrono::steady_clock::now();
    ggml_init_params ip{}; ip.mem_size = ggml_tensor_overhead() * 16; ip.no_alloc = true;
    ggml_context * ictx = use_qd ? nullptr : ggml_init(ip);
    ggml_tensor * kq_mask = use_qd ? nullptr : ggml_new_tensor_2d(ictx, GGML_TYPE_F16, n_kv, T);

    qsa_inputs qsa;
    uint32_t ratio = 0;
    if (cfg_.use_qsa) {
        for (uint32_t il = 0; il < hp_.n_layer; il++)
            if (hp_.is_attn_layer(il) && il < hp_.compress_ratios.size() && hp_.compress_ratios[il] > 0)
                { ratio = (uint32_t) hp_.compress_ratios[il]; break; }
    }
    const int64_t n_blocks = ratio ? (n_kv + ratio - 1) / ratio : 0;
    if (ratio && !use_qd) {
        qsa.cell_blk  = ggml_new_tensor_1d(ictx, GGML_TYPE_I32, n_kv);
        qsa.blk_cells = ggml_new_tensor_1d(ictx, GGML_TYPE_I32, ratio * n_blocks);
        qsa.blk_pos   = ggml_new_tensor_1d(ictx, GGML_TYPE_I32, 4 * n_blocks);
        qsa.bias      = ggml_new_tensor_2d(ictx, GGML_TYPE_F32, n_blocks, T);
        qsa.ratio     = ratio;
        qsa.n_blocks  = n_blocks;
    }
    ggml_backend_buffer_t ibuf = use_qd ? nullptr : ggml_backend_alloc_ctx_tensors_from_buft(ictx, w_.buft());
    if (!use_qd && !ibuf) { ggml_free(ictx); err = "failed to allocate per-call inputs"; return false; }

    if (use_qd) qsa_decode_prepare((int32_t) n_past);
    if (!use_qd) {
        std::vector<uint16_t> m((size_t) n_kv * T, f16_of(-INFINITY));
        for (int64_t i = 0; i < T; i++)
            for (int64_t j = 0; j <= n_past + i; j++) m[i * n_kv + j] = f16_of(0.0f);
        ggml_backend_tensor_set(kq_mask, m.data(), 0, m.size() * 2);
    }
    if (ratio && !use_qd) {
        const int64_t n_bid = n_kv / ratio;                   // only whole blocks are pooled
        const bool have_dead = n_bid < n_blocks;
        const int64_t dead = have_dead ? n_bid : n_blocks - 1;
        std::vector<int32_t> cb(n_kv), bc((size_t) ratio * n_blocks, 0), bp((size_t) 4 * n_blocks, 0);
        std::vector<float>   bi((size_t) n_blocks * T);
        for (int64_t j = 0; j < n_kv; j++)
            cb[j] = (int32_t) (j < n_bid * (int64_t) ratio ? j / ratio : dead);
        for (int64_t b = 0; b < n_bid; b++) {
            for (uint32_t k = 0; k < ratio; k++) bc[b * ratio + k] = (int32_t) (b * ratio + k);
            for (int sec = 0; sec < 4; sec++) bp[sec * n_blocks + b] = (int32_t) (b * ratio);
        }
        for (int64_t i = 0; i < T; i++) {
            const int64_t q = n_past + i;
            const int64_t tail = ((q + 1) / ratio) * ratio;   // the ragged tail stays visible
            for (int64_t b = 0; b < n_blocks; b++)
                bi[i * n_blocks + b] = (b >= n_bid) ? -INFINITY
                                     : (b * (int64_t) ratio >= tail ? 1e9f : 0.0f);
            if (have_dead) bi[i * n_blocks + dead] = 1e9f;
        }
        ggml_backend_tensor_set(qsa.cell_blk,  cb.data(), 0, cb.size() * 4);
        ggml_backend_tensor_set(qsa.blk_cells, bc.data(), 0, bc.size() * 4);
        ggml_backend_tensor_set(qsa.blk_pos,   bp.data(), 0, bp.size() * 4);
        ggml_backend_tensor_set(qsa.bias,      bi.data(), 0, bi.size() * 4);
    }
    if (decode) t_inputs += std::chrono::duration<double>(std::chrono::steady_clock::now() - t_in0).count();

    auto new_ctx = [&](ggml_context ** c, ggml_cgraph ** g) {
        ggml_init_params p{};
        p.mem_size = ggml_tensor_overhead() * 16384 + ggml_graph_overhead_custom(16384, false);
        p.no_alloc = true;
        *c = ggml_init(p);
        *g = ggml_new_graph_custom(*c, 16384, false);
    };
    auto vtok = [&](ggml_context * c) { return ggml_view_1d(c, inp_tok_, T, 0); };
    auto vpos = [&](ggml_context * c) { return ggml_view_1d(c, inp_pos_, T * 4, 0); };

    // ---- PLE rows and token embeddings gathered on the host, then uploaded --
    {
        ggml_backend_tensor_set(h_tok_, hist + (n_hist - T), 0, (size_t) T * 4);
        ggml_context * c; ggml_cgraph * g; new_ctx(&c, &g);
        ggml_tensor * idx = ggml_view_1d(c, h_ple_idx_, PH * T, 0);
        ggml_tensor * e = ggml_get_rows(c, wh_.get("per_layer_token_embd.weight"), idx);
        e = ggml_reshape_2d(c, e, hp_.d_ple * PH, T);
        ggml_build_forward_expand(g, ggml_cpy(c, e, ggml_view_2d(c, h_ple_, n_embd, T, h_ple_->nb[1], 0)));
        ggml_tensor * te = ggml_get_rows(c, wh_.get("token_embd.weight"), ggml_view_1d(c, h_tok_, T, 0));
        ggml_build_forward_expand(g, ggml_cpy(c, te, ggml_view_2d(c, h_emb_, n_embd, T, h_emb_->nb[1], 0)));
        run_on(g, false); ggml_free(c);
        ggml_backend_tensor_get(h_ple_, xfer_.data(), 0, (size_t) n_embd * T * sizeof(float));
        ggml_backend_tensor_set(t_ple_, xfer_.data(), 0, (size_t) n_embd * T * sizeof(float));
    }

    // ---- embedding + wide residual ----------------------------------------
    // Split in two when image embeddings have to be substituted: gather the
    // token embeddings, patch the image rows on the host, then repeat into the
    // hyper-connection streams.
    const bool has_ov = [&] {
        for (int32_t p : ov_pos_) if (p >= n_past && p < n_past + T) return true;
        return false;
    }();
    ggml_backend_tensor_get(h_emb_, xfer_.data(), 0, (size_t) n_embd * T * sizeof(float));
    if (has_ov) {
        for (size_t k = 0; k < ov_pos_.size(); k++) {
            const int32_t p = ov_pos_[k];
            if (p < n_past || p >= n_past + T) continue;
            memcpy(xfer_.data() + (size_t) (p - n_past) * n_embd,
                   ov_.data()  + k * (size_t) n_embd, (size_t) n_embd * sizeof(float));
        }
    }
    ggml_backend_tensor_set(t_emb_, xfer_.data(), 0, (size_t) n_embd * T * sizeof(float));
    {
        ggml_context * c; ggml_cgraph * g; new_ctx(&c, &g);
        ggml_tensor * r = ggml_repeat_4d(c, ggml_reshape_3d(c, ggml_view_2d(c, t_emb_, n_embd, T, t_emb_->nb[1], 0), n_embd, 1, T),
                                         n_embd, hc, T, 1);
        ggml_build_forward_expand(g, ggml_cpy(c, r,
                ggml_view_3d(c, res_[0], n_embd, hc, T, res_[0]->nb[1], res_[0]->nb[2], 0)));
        run_on(g, true); ggml_free(c);
    }

    int cur_res = 0;
    bool pending = false;
    std::vector<expert_handle> eh(U);
    pred_.clear();
    pred2_a_.clear();
    pred2_b_.clear();

    auto vres = [&](ggml_context * c, int which) {
        return ggml_view_3d(c, res_[which], n_embd, hc, T, res_[which]->nb[1], res_[which]->nb[2], 0);
    };
    auto v2 = [&](ggml_context * c, ggml_tensor * t) {
        return ggml_view_2d(c, t, t->ne[0], T, t->nb[1], 0);
    };

    for (uint32_t il = 0; il < hp_.n_layer; il++) {
        const bool is_ple = std::find(hp_.ple_layers.begin(), hp_.ple_layers.end(), (int32_t) il)
                          != hp_.ple_layers.end();
        // ---- graph A (GPU) -------------------------------------------------
        // Replayable only for the recurrent layers, and only during decode: the
        // sparse-attention layers reshape with n_kv every token.
        const bool replayable = cfg_.reuse_graphs && decode && (!hp_.is_attn_layer(il) || use_qd);
        // An attention graph is shaped by its block bucket; a new bucket means a new graph.
        // Readback pack (t_pack_): decode, one token, the GPU MoE in the graph so
        // nothing on the device needs t_sel_/t_w_ afterwards. Decided here so the
        // build and the readback of a cached graph agree.
        const bool pack_ok = t_pack_ && decode && T == 1 && moe_in_graph(il) && !legacy_moe && !check_moe;
        // A graph that speculatively runs an attention successor's block is shaped by the bucket too.
        const bool spec_attn_next = cfg_.spec_block && decode && use_qd && il + 1 < hp_.n_layer
                                    && hp_.is_attn_layer(il + 1) && spec_block_mask_[il + 1];
        if (replayable && (hp_.is_attn_layer(il) || spec_attn_next) && gA_[il].gf && gA_bucket_[il] != qd_.n_bucket) {
            if (gA_[il].ga)  ggml_gallocr_free(gA_[il].ga);
            if (gA_[il].ctx) ggml_free(gA_[il].ctx);
            gA_[il] = layer_graph{};
        }
        if (replayable && gA_[il].gf) {
            const auto ta0 = std::chrono::steady_clock::now();
            if (!ggml_gallocr_alloc_graph(gA_[il].ga, gA_[il].gf)) { err = "replay alloc failed"; return false; }
            if (ggml_backend_graph_compute(w_.backend(), gA_[il].gf) != GGML_STATUS_SUCCESS) {
                err = "replay compute failed"; return false;
            }
            const double dtA = std::chrono::duration<double>(std::chrono::steady_clock::now() - ta0).count();
            t_layerA += dtA;
            if (hp_.is_attn_layer(il)) { t_layerA_attn += dtA; n_layerA_attn++; }
            else                       { t_layerA_rec  += dtA; n_layerA_rec++;  }
            cur_res = 1 - cur_res;
            pending = true;
        } else {
            const auto ta0 = std::chrono::steady_clock::now();
            ggml_context * c; ggml_cgraph * g; new_ctx(&c, &g);
            graph_builder gb(c, &hp_, &w_); gb.bind(&st_, g, n_past);
            gb.set_gpu_fusion(w_.on_gpu() && !getenv("QWFN_NO_FUSE"), t_hcmean_);

            ggml_tensor * r = vres(c, cur_res);
            if (pending) {
                ggml_tensor * pg = il > 0 ? late_fold(c, il - 1, v2(c, t_pg_)) : v2(c, t_pg_);
                ggml_tensor * tot = t_rscale_
                    ? ggml_add(c, v2(c, t_sh_), ggml_mul(c, ggml_add(c, pg, v2(c, t_pc_)), t_rscale_))
                    : ggml_add(c, ggml_add(c, v2(c, t_sh_), pg), v2(c, t_pc_));
                r = gb.hc_combine(r, tot, v2(c, t_inject_));
            }
            if (is_ple) r = gb.ple(v2(c, t_ple_), r, il);

            ggml_tensor * inject = nullptr;
            ggml_tensor * cur = gb.hc_mix(r, il, /*ffn=*/false, &inject);
            if (!hp_.is_attn_layer(il)) {
                cur = gb.deltanet(cur, il);
            } else if (use_qd) {
                qd_.pool_cache = pool_cache_[il];
                cur = gb.sparse_attn_decode(cur, vpos(c), sections, il, qd_);
            } else {
                cur = gb.sparse_attn(cur, vpos(c), kq_mask, sections, il, ratio ? &qsa : nullptr);
            }
            r = gb.hc_combine(r, cur, inject);

            ggml_tensor * cur2 = gb.hc_mix(r, il, /*ffn=*/true, &inject);
            ggml_tensor * sl = nullptr, * wt = nullptr;
            gb.moe_route(cur2, il, &sl, &wt);
            ggml_tensor * sh = gb.shared_expert(cur2, il);

            ggml_tensor * pg_here = nullptr;
            if (moe_in_graph(il)) {
                ggml_tensor * slc = ggml_is_contiguous(sl) ? sl : ggml_cont(c, sl);
                ggml_tensor * wtc = ggml_is_contiguous(wt) ? wt : ggml_cont(c, wt);
                ggml_tensor * sl1  = ggml_reshape_1d(c, slc, U);
                ggml_tensor * slot = ggml_reshape_2d(c, ggml_get_rows(c, t_vslot_[il], sl1), U, 1);      // [U, 1]
                ggml_tensor * mask = ggml_reshape_3d(c, ggml_get_rows(c, t_vmask_[il], sl1), 1, U, 1);   // [1, U, 1]
                ggml_tensor * w    = ggml_mul(c, ggml_reshape_3d(c, wtc, 1, U, 1), mask);
                pg_here = moe_id_graph(c, ec_.gpu_tier(il), slot, w, cur2, n_embd, hp_.n_ff_exp, (int) U, 1, /*fused_sum=*/true);
                if (place(pg_here, t_pg_, 0)) ggml_build_forward_expand(g, pg_here);
                else ggml_build_forward_expand(g, ggml_cpy(c, pg_here, v2(c, t_pg_)));
            }

            // Speculate the next layer's routing from `r`, which is this layer's
            // residual before its own MoE lands. Costs one extra hc_mix plus a
            // 2560x512 router matmul on the GPU; buys reads that overlap this
            // layer's MoE instead of stalling the next one. The two-ahead
            // prediction applies layer L+2's mixer and router to the same
            // residual -- missing two MoE contributions instead of one, so less
            // accurate, but its reads get two layers of compute to land in.
            ggml_tensor * selnext = nullptr;
            if (cfg_.speculate && decode && il + 1 < hp_.n_layer) {
                // The residual `r` is missing this layer's whole FFN output. The
                // shared expert -- always on, already computed above -- is a
                // large, known part of it, so fold it in before predicting; only
                // the routed sum is then missing. QWFN_PREDICT_PLAIN restores the
                // bare residual for comparison.
                // Measured: folding the shared expert in raises accuracy only
                // 81.2% -> 81.6% and costs an hc_combine per layer in the
                // replayed graph, so it is opt-in (QWFN_PREDICT_SHARED=1).
                // With the VRAM MoE in this graph, the routed sum is mostly known
                // too (90% of routed experts are VRAM-served at 128K): predict from
                // the residual missing only the CPU-served experts.
                // QWFN_PREDICT_PLAIN=1 predicts from the bare residual for comparison.
                static const bool predict_shared = getenv("QWFN_PREDICT_SHARED") != nullptr;
                static const bool predict_plain  = getenv("QWFN_PREDICT_PLAIN") != nullptr;
                ggml_tensor * rp = r;
                if (pg_here && !predict_plain) rp = gb.hc_combine(r, ggml_add(c, sh, pg_here), inject);
                else if (predict_shared)        rp = gb.hc_combine(r, sh, inject);
                // Speculative block (cfg spec_block): the residual predictor's
                // remaining error is layer L+1's own block, which it cannot see.
                // Run that block here, on this residual -- L+1's PLE injection if
                // it has one, its attention mixer, its DeltaNet or decode sparse
                // attention, its combine -- with every state write suppressed, and
                // predict from the residual it produces. The exact graph of L+1
                // recomputes the block on the exact residual and does the writes;
                // an attention block's cache rows written here are rewritten at the
                // same position there before anything else reads them.
                const uint32_t iln = il + 1;
                const bool spec_blk = cfg_.spec_block && spec_block_mask_[iln] && (!hp_.is_attn_layer(iln) || use_qd);
                if (spec_blk) {
                    gb.set_persist(false);
                    const bool ple_next = std::find(hp_.ple_layers.begin(), hp_.ple_layers.end(), (int32_t) iln) != hp_.ple_layers.end();
                    if (ple_next) rp = gb.ple(v2(c, t_ple_), rp, (int) iln);
                    ggml_tensor * inj2 = nullptr;
                    ggml_tensor * c1 = gb.hc_mix(rp, (int) iln, /*ffn=*/false, &inj2);
                    ggml_tensor * blk = nullptr;
                    if (!hp_.is_attn_layer(iln)) {
                        blk = gb.deltanet(c1, (int) iln);
                    } else {
                        qd_.pool_cache = pool_cache_[iln];
                        blk = gb.sparse_attn_decode(c1, vpos(c), sections, (int) iln, qd_);
                    }
                    gb.set_persist(true);
                    rp = gb.hc_combine(rp, blk, inj2);
                }
                // Every candidate up to QWFN_SPEC_MAX comes back with its logit;
                // how many are read is decided on the host (speculate_depth, then
                // the margin gate).
                const int depth = (int) QWFN_SPEC_MAX;
                ggml_tensor * xpred = nullptr, * scores = nullptr;
                selnext = gb.moe_route_predict(rp, (int) iln, depth, pred_w(iln), pred_b(iln), &xpred, &scores);
                if (t_xdec_ && xpred)   // the head's input for layer il+1, kept for the decode dump
                    ggml_build_forward_expand(g, ggml_cpy(c, xpred, ggml_view_1d(c, t_xdec_, n_embd, (size_t) il * t_xdec_->nb[1])));
                if (pack_ok) {   // into the readback pack (I32 -> F32 for the ids), see t_pack_
                    ggml_build_forward_expand(g, ggml_cpy(c, selnext,
                            ggml_view_1d(c, t_pack_, depth, (size_t) (n_embd + 2 * U) * sizeof(float))));
                    if (place(scores, t_cur_, (size_t) (n_embd + 2 * U + depth) * sizeof(float))) ggml_build_forward_expand(g, scores);
                    else ggml_build_forward_expand(g, ggml_cpy(c, scores,
                            ggml_view_1d(c, t_pack_, depth, (size_t) (n_embd + 2 * U + depth) * sizeof(float))));
                } else {
                    ggml_build_forward_expand(g, ggml_cpy(c, selnext,
                            ggml_view_2d(c, t_selnext_, depth, T, t_selnext_->nb[1], 0)));
                    ggml_build_forward_expand(g, ggml_cpy(c, scores,
                            ggml_view_2d(c, t_specscore_, depth, T, t_specscore_->nb[1], 0)));
                }
                if (cfg_.speculate_ahead >= 2 && il + 2 < hp_.n_layer) {
                    ggml_tensor * selnext2 = gb.moe_route_predict(rp, (int) il + 2, depth, pred_w(il + 2), pred_b(il + 2));
                    ggml_build_forward_expand(g, ggml_cpy(c, selnext2,
                            ggml_view_2d(c, t_selnext2_, depth, T, t_selnext2_->nb[1], 0)));
                }
            }

            // Outputs in place where the shapes allow (one token): the node that
            // produces an output gets the persistent tensor's memory, so the copy
            // kernel goes. See `place` above.
            ggml_build_forward_expand(g, place(r, res_[1 - cur_res], 0) ? r : ggml_cpy(c, r, vres(c, 1 - cur_res)));
            ggml_build_forward_expand(g, place(cur2, t_cur_, 0) ? cur2 : ggml_cpy(c, cur2, v2(c, t_cur_)));   // row 0 of t_cur_ is the pack's head
            ggml_build_forward_expand(g, place(inject, t_inject_, 0) ? inject : ggml_cpy(c, inject, v2(c, t_inject_)));
            if (pack_ok) {
                ggml_build_forward_expand(g, place(wt, t_cur_, (size_t) n_embd * sizeof(float)) ? wt
                        : ggml_cpy(c, wt, ggml_view_1d(c, t_pack_, U, (size_t) n_embd * sizeof(float))));
                ggml_build_forward_expand(g, ggml_cpy(c, sl, ggml_view_1d(c, t_pack_, U, (size_t) (n_embd + U) * sizeof(float))));   // I32 -> F32: a real conversion
            } else {
                ggml_build_forward_expand(g, ggml_cpy(c, sl,     v2(c, t_sel_)));
                ggml_build_forward_expand(g, ggml_cpy(c, wt,     v2(c, t_w_)));
            }
            ggml_build_forward_expand(g, place(sh, t_sh_, 0) ? sh : ggml_cpy(c, sh, v2(c, t_sh_)));
            gA_pack_[il] = pack_ok;
            if (getenv("QWFN_GRAPH_STATS") && decode && il < 6) {   // op histogram of one recurrent and one attention graph
                std::map<std::string, int> hist; int n_real = 0;
                for (int ni = 0; ni < ggml_graph_n_nodes(g); ni++) {
                    const ggml_tensor * nd = ggml_graph_node(g, ni);
                    const ggml_op op = nd->op;
                    if (op == GGML_OP_VIEW || op == GGML_OP_RESHAPE || op == GGML_OP_PERMUTE || op == GGML_OP_TRANSPOSE || op == GGML_OP_NONE) continue;
                    std::string nm = op == GGML_OP_UNARY ? std::string("unary:") + ggml_unary_op_name(ggml_get_unary_op(nd))
                                   : op == GGML_OP_GLU ? std::string("glu") : ggml_op_name(op);
                    hist[nm]++; n_real++;
                }
                fprintf(stderr, "[graph-stats] layer %u (%s): %d nodes, %d kernels:", il, hp_.is_attn_layer(il) ? "attention" : "recurrent", ggml_graph_n_nodes(g), n_real);
                for (auto & kv : hist) fprintf(stderr, " %s x%d", kv.first.c_str(), kv.second);
                fprintf(stderr, "\n");
                if (atoi(getenv("QWFN_GRAPH_STATS")) >= 2 && il == 1) {   // the op sequence, views included, to check fusion adjacency
                    fprintf(stderr, "[graph-seq] layer %u:", il);
                    for (int ni = 0; ni < ggml_graph_n_nodes(g); ni++) {
                        const ggml_tensor * nd = ggml_graph_node(g, ni);
                        fprintf(stderr, " %s", nd->op == GGML_OP_UNARY ? ggml_unary_op_name(ggml_get_unary_op(nd)) : ggml_op_name(nd->op));
                    }
                    fprintf(stderr, "\n");
                }
            }
            bool cached = false;
            if (replayable) {
                // Keep the context alive and give the graph its own allocator, so
                // the addresses it was built against do not move next token. If
                // the device cannot spare the buffer, fall back to the shared
                // allocator rather than failing: replay is an optimisation.
                ggml_gallocr_t ga = ggml_gallocr_new(w_.buft());
                if (ga && ggml_gallocr_alloc_graph(ga, g)) {
                    gA_[il].ctx = c; gA_[il].gf = g; gA_[il].ga = ga;
                    gA_bucket_[il] = qd_.n_bucket;
                    cached = true;
                    if (ggml_backend_graph_compute(w_.backend(), g) != GGML_STATUS_SUCCESS) {
                        err = "compute failed"; return false;
                    }
                } else {
                    if (ga) ggml_gallocr_free(ga);
                    static bool warned = false;
                    if (!warned) { warned = true;
                        fprintf(stderr, "[qwfn] graph replay disabled: no device memory for the per-layer allocators\n"); }
                }
            }
            if (!cached) {
                const auto tb = std::chrono::steady_clock::now();
                if (!ggml_gallocr_alloc_graph(galloc_gpu_, g)) { fprintf(stderr, "[qwfn] galloc failed\n"); abort(); }
                const auto tc = std::chrono::steady_clock::now();
                if (ggml_backend_graph_compute(w_.backend(), g) != GGML_STATUS_SUCCESS) {
                    fprintf(stderr, "[qwfn] compute failed\n"); abort();
                }
                const auto td = std::chrono::steady_clock::now();
                if (decode && hp_.is_attn_layer(il)) {
                    t_attn_build   += std::chrono::duration<double>(tb - ta0).count()
                                    + std::chrono::duration<double>(tc - tb).count();
                    t_attn_compute += std::chrono::duration<double>(td - tc).count();
                }
                ggml_free(c);
            }
            const double dtA = std::chrono::duration<double>(std::chrono::steady_clock::now() - ta0).count();
            t_layerA += dtA;
            if (decode) {
                if (hp_.is_attn_layer(il)) { t_layerA_attn += dtA; n_layerA_attn++; }
                else                       { t_layerA_rec  += dtA; n_layerA_rec++;  }
            }
            cur_res = 1 - cur_res;
            pending = true;
        }

        const bool packed = decode && gA_pack_[il];
        if (packed) {
            ggml_backend_tensor_get(t_pack_, pack_host_.data(), 0, (size_t) pack_n_ * sizeof(float));
            const float * pk = pack_host_.data();
            ggml_backend_tensor_set(h_cur_, pk, 0, (size_t) n_embd * sizeof(float));   // host tensor: a memcpy
            for (int64_t e = 0; e < U; e++) { wgt_[e] = pk[n_embd + e]; sel_[e] = (int32_t) lrintf(pk[n_embd + U + e]); }
            for (uint32_t k = 0; k < QWFN_SPEC_MAX; k++) {
                pred_next_[k]   = (int32_t) lrintf(pk[n_embd + 2 * U + k]);
                scores_next_[k] = pk[n_embd + 2 * U + QWFN_SPEC_MAX + k];
            }
        } else {
            ggml_backend_tensor_get(t_sel_, sel_.data(), 0, (size_t) U * T * sizeof(int32_t));
            ggml_backend_tensor_get(t_w_,   wgt_.data(), 0, (size_t) U * T * sizeof(float));
        }
        if (t_xdec_ && decode) {   // this token's true routing of layer il; the record is complete at the last layer
            memcpy(tok_sel_.data() + (size_t) il * U, sel_.data(), (size_t) U * sizeof(int32_t));
            memcpy(tok_w_.data()   + (size_t) il * U, wgt_.data(), (size_t) U * sizeof(float));
            if (il + 1 == hp_.n_layer) dump_decode_token();
        }

        // Score the prediction made one layer ago against what actually happened.
        // Only where one exists: at layer 0, pred_ still holds the prediction made
        // at layer n-2 for layer n-1, already scored; counting it against layer 0's
        // routing added ten near-certain misses per token to the reported rate.
        if (cfg_.speculate && decode && il > 0 && !pred_.empty()) {
            for (int64_t e = 0; e < U; e++) {
                pred_total++; pred_total_layer[il]++;
                for (int64_t k = 0; k < U; k++)
                    if (pred_[k] == sel_[e]) { pred_hits++; pred_hits_layer[il]++; break; }
            }
            // Precision by rank and by confidence margin: the gate's calibration data.
            for (size_t k = 0; k < pred_.size() && k < QWFN_SPEC_MAX; k++) {
                bool ok = false;
                for (int64_t e = 0; e < U; e++) if (pred_[k] == sel_[e]) { ok = true; break; }
                rank_total[k]++; if (ok) rank_hits[k]++;
                if (k < pred_margin_.size()) {
                    const int b = margin_bucket(pred_margin_[k]);
                    margin_total[b]++; if (ok) margin_hits[b]++;
                }
            }
        }
        // And the one made two layers ago, if the two-ahead path is on.
        if (cfg_.speculate && decode && !pred2_a_.empty()) {
            for (int64_t e = 0; e < U; e++) {
                pred2_total++;
                for (int64_t k = 0; k < U; k++)
                    if (pred2_a_[k] == sel_[e]) { pred2_hits++; break; }
            }
        }

        if (decode) {
            // ---- decode: cache-served, one matmul per expert ----------------
            for (int64_t e = 0; e < U; e++) ids_[e] = (uint32_t) sel_[e];
            std::vector<uint8_t> rdy(U, 0);
            const auto ti = std::chrono::steady_clock::now();
            if (deferred_wait_) {   // the previous layer's skipped misses: their reads had a layer to land
                if (!ec_.fetch_end()) { err = "expert read failed"; if (ibuf) ggml_backend_buffer_free(ibuf); if (ictx) ggml_free(ictx); return false; }
                deferred_wait_ = false;
            }
            if (!ec_.fetch_begin(il, (const uint32_t *) ids_.data(), (uint32_t) U,
                                 eh.data(), (bool *) rdy.data())) {
                err = "expert fetch failed"; if (ibuf) ggml_backend_buffer_free(ibuf); if (ictx) ggml_free(ictx); return false;
            }
            t_io += std::chrono::duration<double>(std::chrono::steady_clock::now() - ti).count();

            // Resident now; the misses are streaming in behind these.
            // Overlapping the reads with compute means summing the routed
            // experts in two groups instead of one. That is mathematically the
            // same sum, but this model is violently sensitive to accumulation
            // order -- simply reversing the ten additions changes the second
            // generated token -- so it is opt-in. Off, the engine reproduces
            // llama.cpp exactly; on, it is faster and equally valid but no
            // longer bit-comparable.
            //
            // Device placement is never negotiable: an expert resident in VRAM
            // must run on the GPU whatever the overlap setting, or the CPU
            // backend dereferences a device pointer.
            static const bool overlap = getenv("QWFN_OVERLAP") != nullptr;
            std::vector<int> on_gpu, late_gpu, ready_cpu, late_cpu, all_cpu;
            for (int64_t e = 0; e < U; e++) {
                // `late` only matters to the in-graph path (its graph already ran);
                // the host-driven paths compute promoted experts like any other.
                if (eh[e].on_gpu) { (eh[e].late && moe_in_graph(il) ? late_gpu : on_gpu).push_back((int) e); continue; }
                all_cpu.push_back((int) e);                       // selection order
                (rdy[e] ? ready_cpu : late_cpu).push_back((int) e);
            }

            // Build one MoE graph over `which`, aliasing cached blocks so nothing
            // is copied. Returns the context; the caller decides how to run it.
            auto moe_graph = [&](const std::vector<int> & which,
                                 ggml_tensor * in, ggml_tensor * out,
                                 ggml_cgraph ** g_out) -> ggml_context * {
                ggml_context * c; ggml_cgraph * g; new_ctx(&c, &g);
                graph_builder gb(c, &hp_, &w_); gb.bind(&st_, g, n_past);
                std::vector<ggml_tensor *> tg, tu, td; std::vector<float> ww;
                for (int e : which) {
                    ggml_tensor * a[EXPERT_NPARTS];
                    a[EXPERT_GATE] = ggml_new_tensor_2d(c, eh[e].type[EXPERT_GATE], n_embd, hp_.n_ff_exp);
                    a[EXPERT_UP]   = ggml_new_tensor_2d(c, eh[e].type[EXPERT_UP],   n_embd, hp_.n_ff_exp);
                    a[EXPERT_DOWN] = ggml_new_tensor_2d(c, eh[e].type[EXPERT_DOWN], hp_.n_ff_exp, n_embd);
                    for (int q = 0; q < EXPERT_NPARTS; q++) {
                        a[q]->buffer = eh[e].buffer;
                        a[q]->data   = (void *) eh[e].part[q];
                    }
                    tg.push_back(a[EXPERT_GATE]); tu.push_back(a[EXPERT_UP]); td.push_back(a[EXPERT_DOWN]);
                    ww.push_back(wgt_[e]);
                }
                ggml_tensor * o = gb.moe_apply(v2(c, in), il, tg.data(), tu.data(), td.data(),
                                               ww.data(), (int) which.size());
                ggml_build_forward_expand(g, ggml_cpy(c, o, v2(c, out)));
                *g_out = g;
                return c;
            };

            auto part = [&](const std::vector<int> & which, bool gpu,
                            ggml_tensor * in, ggml_tensor * out) {
                const auto tm0 = std::chrono::steady_clock::now();
                if (which.empty()) {
                    ggml_backend_tensor_set(out, zeros_.data(), 0, (size_t) n_embd * T * sizeof(float));
                    return;
                }
                if (!gpu && moe_by_id) {
                    // One mul_mat_id per part over the RAM arena, experts by
                    // slot. Same kernels and summation order as one mul_mat per
                    // expert, but one activation quantisation instead of two per
                    // expert and a handful of barriers instead of ~15.
                    const tier_view tv = ec_.ram_tier(il);
                    const int n = (int) which.size();
                    ggml_context * c; ggml_cgraph * g; new_ctx(&c, &g);
                    ggml_tensor * ids = ggml_new_tensor_2d(c, GGML_TYPE_I32, n, 1);   ggml_set_input(ids);
                    ggml_tensor * w   = ggml_new_tensor_3d(c, GGML_TYPE_F32, 1, n, 1); ggml_set_input(w);
                    ggml_tensor * acc = moe_id_graph(c, tv, ids, w, in, n_embd, hp_.n_ff_exp, n);
                    ggml_build_forward_expand(g, ggml_cpy(c, acc, v2(c, out)));
                    if (!ggml_gallocr_alloc_graph(galloc_cpu_, g)) { fprintf(stderr, "[qwfn] galloc failed\n"); abort(); }
                    std::vector<int32_t> sid(n); std::vector<float> sw(n);
                    for (int k = 0; k < n; k++) { sid[k] = eh[which[k]].slot; sw[k] = wgt_[which[k]]; }
                    ggml_backend_tensor_set(ids, sid.data(), 0, (size_t) n * sizeof(int32_t));
                    ggml_backend_tensor_set(w,   sw.data(),  0, (size_t) n * sizeof(float));
                    if (ggml_backend_graph_compute(wh_.backend(), g) != GGML_STATUS_SUCCESS) {
                        fprintf(stderr, "[qwfn] compute failed\n"); abort();
                    }
                    ggml_free(c);
                    if (check_moe) {
                        // Diagnostic: the per-expert graph on the same inputs must
                        // give the same bytes.
                        std::vector<float> a(n_embd), b(n_embd);
                        ggml_backend_tensor_get(out, a.data(), 0, n_embd * sizeof(float));
                        ggml_cgraph * g2; ggml_context * c2 = moe_graph(which, in, out, &g2);
                        run_on(g2, false); ggml_free(c2);
                        ggml_backend_tensor_get(out, b.data(), 0, n_embd * sizeof(float));
                        size_t bad = 0; float mx = 0;
                        for (int64_t k = 0; k < n_embd; k++)
                            if (memcmp(&a[k], &b[k], 4) != 0) { bad++; mx = std::max(mx, std::fabs(a[k] - b[k])); }
                        if (bad) fprintf(stderr, "[check-moe] layer %2u CPU: %zu of %lld values differ (max |d| %g)\n",
                                         il, bad, (long long) n_embd, mx);
                        check_moe_cpu_calls++;
                    }
                    const double dt = std::chrono::duration<double>(
                            std::chrono::steady_clock::now() - tm0).count();
                    t_moe_cpu += dt; n_exp_cpu += which.size();
                    return;
                }
                ggml_cgraph * g;
                ggml_context * c = moe_graph(which, in, out, &g);
                run_on(g, gpu); ggml_free(c);
                const double dt = std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - tm0).count();
                if (gpu) { t_moe_gpu += dt; n_exp_gpu += which.size(); }
                else     { t_moe_cpu += dt; n_exp_cpu += which.size(); }
            };

            // Launch the GPU MoE without waiting. It is the same graph part()
            // would run -- identical arithmetic, identical partial sums -- only
            // the wait moves: the kernels execute while this thread waits on the
            // NVMe and runs the CPU MoE. Settled below, before t_pg_ is needed.
            auto moe_gpu_launch = [&](const std::vector<int> & which) {
                const auto tm0 = std::chrono::steady_clock::now();
                if (moe_in_graph(il)) {
                    // Already done: graph A of this layer computed `which` into
                    // t_pg_. The experts promoted by this fetch were not there
                    // yet: their slots go up for the next graph's late fold,
                    // through pinned staging on the compute stream -- after the
                    // promotion copies, before the graph that reads them.
                    if ((int) late_gpu.size() > n_late_) { fprintf(stderr, "[qwfn] %zu late experts, fold holds %d\n", late_gpu.size(), n_late_); abort(); }
                    std::vector<int32_t> gids(U, 0); std::vector<float> gw(U, 0.0f);
                    for (size_t k = 0; k < late_gpu.size(); k++) { gids[k] = eh[late_gpu[k]].slot; gw[k] = wgt_[late_gpu[k]]; }
                    if (p_gids_) {
                        memcpy(p_gids_->data, gids.data(), (size_t) U * sizeof(int32_t));
                        memcpy(p_gw_->data,   gw.data(),   (size_t) U * sizeof(float));
                        ggml_backend_tensor_set_async(w_.backend(), t_gids_, p_gids_->data, 0, (size_t) U * sizeof(int32_t));
                        ggml_backend_tensor_set_async(w_.backend(), t_gw_,   p_gw_->data,   0, (size_t) U * sizeof(float));
                    } else {
                        ggml_backend_tensor_set(t_gids_, gids.data(), 0, (size_t) U * sizeof(int32_t));
                        ggml_backend_tensor_set(t_gw_,   gw.data(),   0, (size_t) U * sizeof(float));
                    }
                    n_exp_gpu += which.size() + late_gpu.size();
                    t_moe_gpu += std::chrono::duration<double>(std::chrono::steady_clock::now() - tm0).count();
                    return;
                }
                if (which.empty()) {
                    ggml_backend_tensor_set(t_pg_, zeros_.data(), 0, (size_t) n_embd * T * sizeof(float));
                    return;
                }
                if (moe_by_id && (gM_[il].gf || build_moe_gpu_graph(il))) {
                    // Slot ids and weights for all n_expert_used positions; the
                    // ones not in VRAM point at slot 0 with weight 0.
                    std::vector<int32_t> gids(U, 0); std::vector<float> gw(U, 0.0f);
                    for (int e : which) { gids[e] = eh[e].slot; gw[e] = wgt_[e]; }
                    ggml_backend_tensor_set(t_gids_, gids.data(), 0, (size_t) U * sizeof(int32_t));
                    ggml_backend_tensor_set(t_gw_,   gw.data(),   0, (size_t) U * sizeof(float));
                    if (!ggml_gallocr_alloc_graph(gM_[il].ga, gM_[il].gf)) { fprintf(stderr, "[qwfn] galloc failed\n"); abort(); }
                    if (ggml_backend_graph_compute_async(w_.backend(), gM_[il].gf) != GGML_STATUS_SUCCESS) {
                        fprintf(stderr, "[qwfn] compute failed\n"); abort();
                    }
                    if (check_moe) {
                        ggml_backend_synchronize(w_.backend());
                        std::vector<float> a(n_embd), b(n_embd);
                        ggml_backend_tensor_get(t_pg_, a.data(), 0, n_embd * sizeof(float));
                        ggml_cgraph * g2; ggml_context * c2 = moe_graph(which, t_cur_, t_pg_, &g2);
                        run_on(g2, true); ggml_free(c2);
                        ggml_backend_tensor_get(t_pg_, b.data(), 0, n_embd * sizeof(float));
                        size_t bad = 0; float mx = 0;
                        for (int64_t k = 0; k < n_embd; k++)
                            if (memcmp(&a[k], &b[k], 4) != 0) { bad++; mx = std::max(mx, std::fabs(a[k] - b[k])); }
                        if (bad) fprintf(stderr, "[check-moe] layer %2u GPU: %zu of %lld values differ (max |d| %g)\n",
                                         il, bad, (long long) n_embd, mx);
                        check_moe_gpu_calls++;
                    }
                } else {
                    ggml_cgraph * g;
                    ggml_context * c = moe_graph(which, t_cur_, t_pg_, &g);
                    if (!ggml_gallocr_alloc_graph(galloc_moe_, g)) { fprintf(stderr, "[qwfn] galloc failed\n"); abort(); }
                    if (ggml_backend_graph_compute_async(w_.backend(), g) != GGML_STATUS_SUCCESS) {
                        fprintf(stderr, "[qwfn] compute failed\n"); abort();
                    }
                    moe_ctx_ = c;
                }
                moe_inflight_ = true;
                t_moe_gpu += std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - tm0).count();
                n_exp_gpu += which.size();
            };
            auto moe_gpu_settle = [&]() {
                if (!moe_inflight_) return;
                const auto tm0 = std::chrono::steady_clock::now();
                ggml_backend_synchronize(w_.backend());
                if (moe_ctx_) ggml_free(moe_ctx_);
                moe_ctx_ = nullptr;
                moe_inflight_ = false;
                const double dt = std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - tm0).count();
                t_moe_gpu += dt; t_moe_gpu_sync += dt;
            };

            // Whether this layer has demand reads in flight. The speculative
            // reads for the next layer share the NVMe with them: issued first,
            // they take ~2/3 of the device's concurrency and a demand burst that
            // should land in ~0.4 ms takes ~1 ms. So when there are demand
            // misses, the speculative submission waits until they have landed.
            // 71% of layers have no miss and keep the full window.
            // Measured on the replayed sequence: deferring costs more in
            // settle-wait at the next layer than it saves in demand-wait here
            // (17.5/19.4 vs 19.2 tok/s), so early submission stays the default
            // and QWFN_PREFETCH_DEFER=1 selects the deferred variant.
            static const bool prefetch_early = getenv("QWFN_PREFETCH_DEFER") == nullptr;
            const bool had_miss = ec_.has_inflight();

            // VRAM-resident experts first: their kernels run behind everything
            // this thread does from here to the settle.
            moe_gpu_launch(on_gpu);

            // Issue the predicted reads, so they overlap this layer's MoE
            // compute rather than stalling the next fetch. One submission for
            // the L+1 set and, when two-ahead is on, the L+2 set.
            bool prefetched = false;
            auto issue_prefetch = [&]() {
                if (prefetched || !cfg_.speculate || il + 1 >= hp_.n_layer) return;
                prefetched = true;
                const uint32_t depth = std::min<uint32_t>(QWFN_SPEC_MAX, std::max<uint32_t>((uint32_t) U, cfg_.speculate_depth));
                const uint32_t K = QWFN_SPEC_MAX;
                pred_.resize(K); spec_scores_.resize(K); pred_margin_.resize(K);
                if (packed) {
                    std::copy(pred_next_.begin(), pred_next_.begin() + K, pred_.begin());
                    std::copy(scores_next_.begin(), scores_next_.begin() + K, spec_scores_.begin());
                } else {
                    ggml_backend_tensor_get(t_selnext_,   pred_.data(),        0, (size_t) K * sizeof(int32_t));
                    ggml_backend_tensor_get(t_specscore_, spec_scores_.data(), 0, (size_t) K * sizeof(float));
                }
                // Margin to the routing cut-off (see engine_config::spec_margin).
                const uint32_t Uu = (uint32_t) U;
                for (uint32_t r = 0; r < K; r++)
                    pred_margin_[r] = Uu < K ? (r < Uu ? spec_scores_[r] - spec_scores_[Uu] : spec_scores_[Uu - 1] - spec_scores_[r]) : 1e9f;
                const bool gate = cfg_.spec_margin > 0.0f && ec_.pf_outstanding() >= cfg_.spec_gate_inflight;
                std::vector<uint32_t> pf1, pf2; pf1.reserve(depth);
                for (uint32_t e = 0; e < depth; e++) {
                    if (gate && pred_margin_[e] < cfg_.spec_margin) { pf_gated++; continue; }
                    pf1.push_back((uint32_t) pred_[e]);
                }

                pred2_a_.swap(pred2_b_);
                pred2_b_.clear();
                expert_cache::pf_set sets[2] = { { il + 1, pf1.data(), (uint32_t) pf1.size() }, {} };
                uint32_t n_sets = 1;
                if (cfg_.speculate_ahead >= 2 && il + 2 < hp_.n_layer) {
                    pred2_b_.resize(depth);
                    ggml_backend_tensor_get(t_selnext2_, pred2_b_.data(), 0, (size_t) depth * sizeof(int32_t));
                    const uint32_t depth2 = std::min(depth, cfg_.speculate_depth2);
                    pf2.resize(depth2);
                    for (uint32_t e = 0; e < depth2; e++) pf2[e] = (uint32_t) pred2_b_[e];
                    if (depth2) sets[n_sets++] = { il + 2, pf2.data(), depth2 };
                }
                ec_.prefetch_begin(sets, n_sets);
            };
            if (!had_miss || prefetch_early) issue_prefetch();

            if (!packed) {
                ggml_backend_tensor_get(t_cur_, xfer_.data(), 0, (size_t) n_embd * T * sizeof(float));
                ggml_backend_tensor_set(h_cur_, xfer_.data(), 0, (size_t) n_embd * T * sizeof(float));
            }

            const size_t nfl = (size_t) n_embd * T;
            std::vector<float> acc(nfl, 0.0f);

            auto add_cpu = [&](const std::vector<int> & which, bool first) {
                if (which.empty()) return;
                part(which, false, h_cur_, h_partial_);
                static const bool cold_debug = getenv("QWFN_COLD_DEBUG") != nullptr;
                if (cold_debug) {   // diagnostic: which expert makes the partial non-finite, and does its payload look sane?
                    std::vector<float> tmp(nfl);
                    ggml_backend_tensor_get(h_partial_, tmp.data(), 0, nfl * sizeof(float));
                    bool bad = false; for (float v : tmp) if (!std::isfinite(v)) { bad = true; break; }
                    if (bad) {
                        for (int e : which) {
                            std::vector<int> one{e};
                            ggml_cgraph * g1; ggml_context * c1 = moe_graph(one, h_cur_, h_partial_, &g1);
                            run_on(g1, false); ggml_free(c1);
                            ggml_backend_tensor_get(h_partial_, tmp.data(), 0, nfl * sizeof(float));
                            bool b1 = false; float mx = 0; for (float v : tmp) { if (!std::isfinite(v)) { b1 = true; break; } mx = std::max(mx, std::fabs(v)); }
                            const ggml_fp16_t d16 = *(const ggml_fp16_t *) eh[e].part[EXPERT_GATE];
                            fprintf(stderr, "[cold-debug] layer %u expert %d %s types %d/%d/%d: %s (max |y| %g), gate block0 d=%g\n",
                                    il, sel_[e], eh[e].from_cold ? "COLD" : "hot", (int) eh[e].type[0], (int) eh[e].type[1], (int) eh[e].type[2],
                                    b1 ? "NON-FINITE" : "finite", mx, ggml_fp16_to_fp32(d16));
                        }
                        part(which, false, h_cur_, h_partial_);   // restore the partial
                    }
                }
                if (first) {
                    ggml_backend_tensor_get(h_partial_, acc.data(), 0, nfl * sizeof(float));
                } else {
                    ggml_backend_tensor_get(h_partial_, xfer_.data(), 0, nfl * sizeof(float));
                    for (size_t k = 0; k < nfl; k++) acc[k] += xfer_[k];
                }
            };

            if (overlap) {
                add_cpu(ready_cpu, true);          // NVMe fills the misses during this
                const auto tw = std::chrono::steady_clock::now();
                if (!ec_.fetch_end()) {
                    err = "expert read failed"; if (ibuf) ggml_backend_buffer_free(ibuf); if (ictx) ggml_free(ictx); return false;
                }
                t_io += std::chrono::duration<double>(std::chrono::steady_clock::now() - tw).count();
                issue_prefetch();                  // no-op if already issued
                add_cpu(late_cpu, ready_cpu.empty());
            } else if (t_rscale_ && !late_cpu.empty()) {
                // skip_miss: compute with what is resident, renormalise the gates
                // of the experts that took part, and let the misses' reads land
                // while the next layer runs (waited for at its fetch).
                issue_prefetch();
                add_cpu(ready_cpu, true);
                float kept = 1.0f; for (int e : late_cpu) kept -= wgt_[e];
                const float scale = kept > 1e-3f ? 1.0f / kept : 1.0f;
                ggml_backend_tensor_set(t_rscale_, &scale, 0, 4);
                n_exp_skipped += late_cpu.size();
                deferred_wait_ = ec_.has_inflight();
            } else {
                const auto tw = std::chrono::steady_clock::now();
                if (!ec_.fetch_end()) {
                    err = "expert read failed"; if (ibuf) ggml_backend_buffer_free(ibuf); if (ictx) ggml_free(ictx); return false;
                }
                t_io += std::chrono::duration<double>(std::chrono::steady_clock::now() - tw).count();
                issue_prefetch();                  // no-op if already issued
                add_cpu(all_cpu, true);            // one group, selection order
                if (t_rscale_) { const float one = 1.0f; ggml_backend_tensor_set(t_rscale_, &one, 0, 4); }
            }
            moe_gpu_settle();                      // t_pg_ is complete past this point
            ec_.settle_promotions();               // the async H2D copies too; frees their RAM slots
            static const bool nan_check = getenv("QWFN_NAN_CHECK") != nullptr;
            if (nan_check) {   // diagnostic: where does a non-finite value first appear?
                auto bad = [&](const float * v, size_t n) { for (size_t k = 0; k < n; k++) if (!std::isfinite(v[k])) return true; return false; };
                std::vector<float> pgh(nfl);
                ggml_backend_tensor_get(t_pg_, pgh.data(), 0, nfl * sizeof(float));
                const bool bc = bad(acc.data(), nfl), bg = bad(pgh.data(), nfl);
                if (bc || bg) {
                    fprintf(stderr, "[nan-check] token %d layer %u: %s%s | experts:", n_past, il, bc ? "CPU partial " : "", bg ? "GPU partial" : "");
                    for (int64_t e = 0; e < U; e++) fprintf(stderr, " %d%s%s%s", sel_[e], eh[e].on_gpu ? "g" : "c", eh[e].from_cold ? "*" : "", eh[e].late ? "L" : "");
                    fprintf(stderr, "\n");
                }
            }
            // The CPU partial goes up asynchronously from pinned memory: the copy
            // is queued on the compute stream ahead of the next layer's graph, and
            // the staging is next written only after that graph has run
            // synchronously, so the DMA has long completed. Pageable memory would
            // make cudaMemcpyAsync block the caller anyway (see the promotions).
            if (p_pc_ && T == 1) {
                memcpy(p_pc_->data, acc.data(), nfl * sizeof(float));
                ggml_backend_tensor_set_async(w_.backend(), t_pc_, p_pc_->data, 0, nfl * sizeof(float));
            } else {
                ggml_backend_tensor_set(t_pc_, acc.data(), 0, nfl * sizeof(float));
            }
            upload_vtable(il + 1);                 // next layer's residency, after this layer's promotions
        } else if (cbatch) {
            // ---- short prompt: the union of its experts through the cache ----
            // Distinct experts of the batch, ascending (ascending file offsets
            // for the reads), fetched in chunks the RAM tier can hold at once.
            // Per chunk: VRAM-resident experts compute from the tier, the rest
            // are uploaded into the scratch and compute from there; both are one
            // mul_mat_id per part with the batch's routing as ids, pairs outside
            // the chunk pointed at a real slot with weight 0. The partials add
            // up in t_pg_. Misses are admitted to the RAM tier as they are read,
            // which is the warm-up generation needs anyway.
            std::vector<uint32_t> cnt(hp_.n_expert, 0);
            for (int64_t k = 0; k < U * T; k++) if (sel_[k] >= 0 && sel_[k] < (int32_t) hp_.n_expert) cnt[sel_[k]]++;
            std::vector<uint32_t> uniq;
            for (uint32_t e = 0; e < hp_.n_expert; e++) if (cnt[e]) uniq.push_back(e);

            const uint32_t chunk = std::max<uint32_t>(1, std::min<uint32_t>(
                    std::min<uint32_t>(scr_slots_, cfg_.cache_batch_chunk),
                    std::max<uint32_t>(1, ec_.ram_tier(il).n_slots / 2)));
            ggml_backend_tensor_set(t_pg_, zeros_.data(), 0, (size_t) n_embd * T * sizeof(float));

            size_t slice[EXPERT_NPARTS];
            for (int q = 0; q < EXPERT_NPARTS; q++) slice[q] = mi_->expert_range(il, 0, (expert_part) q).nbytes;
            const tier_view gt = ec_.gpu_tier(il);
            const tier_view rt = ec_.ram_tier(il);

            std::vector<expert_handle> ch(chunk);
            std::vector<int32_t> pos(hp_.n_expert, -1);

            // Per-term id matrices. ggml's MoE gather assumes an expert appears
            // at most once in a token's list (a single shared "dummy" for every
            // absent slot faulted inside MMQ), so each token's list is its real
            // experts of this term followed by DISTINCT unused ones at weight 0,
            // and the list width is the widest any token needs.
            struct term { std::vector<int32_t> slots;  /* per chunk entry: slot or -1 */
                          std::vector<int32_t> ids; std::vector<float> w; int width = 0; };
            auto build_term = [&](term & tm, std::vector<int32_t> & members /* chunk indices */) {
                tm.width = 0; tm.ids.clear(); tm.w.clear();
                if (members.empty()) return;
                std::vector<std::vector<int32_t>> per_tok(T);   // chunk indices per token
                for (int64_t t = 0; t < T; t++)
                    for (int64_t j = 0; j < U; j++) {
                        const int32_t i = pos[sel_[t * U + j]];
                        if (i >= 0 && tm.slots[i] >= 0) per_tok[t].push_back(i);
                    }
                for (int64_t t = 0; t < T; t++) tm.width = std::max<int>(tm.width, (int) per_tok[t].size());
                if (tm.width == 0) return;
                tm.ids.assign((size_t) tm.width * T, 0); tm.w.assign((size_t) tm.width * T, 0.0f);
                for (int64_t t = 0; t < T; t++) {
                    const auto & lst = per_tok[t];
                    int j = 0;
                    for (int32_t i : lst) {
                        tm.ids[t * tm.width + j] = tm.slots[i];
                        // the gate weight of this (token, expert) pair
                        float wv = 0.0f;
                        for (int64_t jj = 0; jj < U; jj++) if (pos[sel_[t * U + jj]] == i) { wv = wgt_[t * U + jj]; break; }
                        tm.w[t * tm.width + j] = wv;
                        j++;
                    }
                    // pad with members this token does not use (distinct), weight 0
                    for (size_t m = 0; m < members.size() && j < tm.width; m++) {
                        const int32_t i = members[m];
                        if (std::find(lst.begin(), lst.end(), i) != lst.end()) continue;
                        tm.ids[t * tm.width + j] = tm.slots[i];
                        j++;
                    }
                }
            };
            term tt, ts;   // tier term, scratch term

            for (size_t c0 = 0; c0 < uniq.size(); c0 += chunk) {
                const uint32_t n = (uint32_t) std::min<size_t>(chunk, uniq.size() - c0);
                const auto ti = std::chrono::steady_clock::now();
                if (!ec_.fetch(il, uniq.data() + c0, n, ch.data())) {
                    err = "expert fetch failed"; if (ibuf) ggml_backend_buffer_free(ibuf); if (ictx) ggml_free(ictx); return false;
                }
                t_io += std::chrono::duration<double>(std::chrono::steady_clock::now() - ti).count();
                const auto tm0 = std::chrono::steady_clock::now();

                std::fill(pos.begin(), pos.end(), -1);
                tt.slots.assign(n, -1); ts.slots.assign(n, -1);
                std::vector<int32_t> mem_t, mem_s;
                uint32_t m = 0;   // scratch slots used, packed from 0
                for (uint32_t i = 0; i < n; i++) {
                    pos[uniq[c0 + i]] = (int32_t) i;
                    if (ch[i].on_gpu) { tt.slots[i] = ch[i].slot; mem_t.push_back((int32_t) i); continue; }
                    // RAM-resident: into the NEXT scratch slot, on the compute
                    // stream. Packed, not at index i: MMQ over-reads the last
                    // row of every expert into the following slot, and a slot
                    // skipped here would hold an earlier layer's bytes, which
                    // decoded as this layer's type can be NaN scales -- seen as
                    // NaN logits on the second turn of a chat, once some of a
                    // chunk's experts were already in VRAM.
                    ts.slots[i] = (int32_t) m; mem_s.push_back((int32_t) i);
                    for (int q = 0; q < EXPERT_NPARTS; q++)
                        ggml_backend_tensor_set_async(w_.backend(), scr_xfer_, ch[i].part[q],
                                scr_part_off_[q] + (size_t) m * slice[q], slice[q]);
                    m++;
                }
                // The bytes past the last packed slot get read by the same
                // over-read; keep them zero (synchronous memset, disjoint).
                for (int q = 0; q < EXPERT_NPARTS; q++)
                    ggml_backend_tensor_memset(scr_xfer_, 0, scr_part_off_[q] + (size_t) m * slice[q], 8192);

                build_term(tt, mem_t);
                build_term(ts, mem_s);
                if (tt.width == 0 && ts.width == 0) continue;

                ggml_context * c; ggml_cgraph * g; new_ctx(&c, &g);
                ggml_tensor * acc = v2(c, t_pg_);
                ggml_tensor * tid = nullptr, * tw = nullptr, * sid = nullptr, * sw = nullptr;
                if (tt.width > 0) {
                    tid = ggml_new_tensor_2d(c, GGML_TYPE_I32, tt.width, T);   ggml_set_input(tid);
                    tw  = ggml_new_tensor_3d(c, GGML_TYPE_F32, 1, tt.width, T); ggml_set_input(tw);
                    acc = ggml_add(c, acc, moe_id_graph(c, gt, tid, tw, t_cur_, n_embd, hp_.n_ff_exp, tt.width, T));
                }
                if (ts.width > 0) {
                    tier_view sv;
                    for (int q = 0; q < EXPERT_NPARTS; q++) {
                        sv.part[q]   = scr_base_ + scr_part_off_[q];
                        sv.stride[q] = slice[q];
                        sv.type[q]   = rt.type[q];
                    }
                    sv.n_slots = m;
                    sv.buffer  = scr_buf_;
                    sid = ggml_new_tensor_2d(c, GGML_TYPE_I32, ts.width, T);   ggml_set_input(sid);
                    sw  = ggml_new_tensor_3d(c, GGML_TYPE_F32, 1, ts.width, T); ggml_set_input(sw);
                    acc = ggml_add(c, acc, moe_id_graph(c, sv, sid, sw, t_cur_, n_embd, hp_.n_ff_exp, ts.width, T));
                }
                ggml_build_forward_expand(g, ggml_cpy(c, acc, v2(c, t_pg_)));
                if (!ggml_gallocr_alloc_graph(galloc_gpu_, g)) { fprintf(stderr, "[qwfn] galloc failed\n"); abort(); }
                if (tid) { ggml_backend_tensor_set(tid, tt.ids.data(), 0, tt.ids.size() * 4); ggml_backend_tensor_set(tw, tt.w.data(), 0, tt.w.size() * 4); }
                if (sid) { ggml_backend_tensor_set(sid, ts.ids.data(), 0, ts.ids.size() * 4); ggml_backend_tensor_set(sw, ts.w.data(), 0, ts.w.size() * 4); }
                if (ggml_backend_graph_compute(w_.backend(), g) != GGML_STATUS_SUCCESS) {
                    fprintf(stderr, "[qwfn] compute failed\n"); abort();
                }
                ggml_free(c);
                t_moe_gpu += std::chrono::duration<double>(std::chrono::steady_clock::now() - tm0).count();
                n_exp_gpu += n;
            }
            ec_.settle_promotions();
            ggml_backend_tensor_set(t_pc_, zeros_.data(), 0, (size_t) n_embd * T * sizeof(float));
        } else {
            // ---- prefill: expert-major permutation over the whole batch -----
            struct pair { int32_t e, tok, slot; };
            std::vector<pair> pairs((size_t) U * T);
            for (int64_t t = 0; t < T; t++)
                for (int64_t j = 0; j < U; j++)
                    pairs[t * U + j] = { sel_[t * U + j], (int32_t) t, (int32_t) j };
            std::stable_sort(pairs.begin(), pairs.end(),
                             [](const pair & a, const pair & b) { return a.e < b.e; });

            const int64_t Pn = (int64_t) pairs.size();
            std::vector<int32_t> perm(Pn), inv(Pn);
            std::vector<float>   wp(Pn);
            for (int64_t k = 0; k < Pn; k++) {
                perm[k] = pairs[k].tok;
                wp[k]   = wgt_[(size_t) pairs[k].tok * U + pairs[k].slot];
                inv[(size_t) pairs[k].tok * U + pairs[k].slot] = (int32_t) k;
            }

            ggml_context * c; ggml_cgraph * g; new_ctx(&c, &g);
            graph_builder gb(c, &hp_, &w_); gb.bind(&st_, g, n_past);

            ggml_tensor * t_perm = ggml_new_tensor_1d(c, GGML_TYPE_I32, Pn);
            ggml_tensor * t_inv  = ggml_new_tensor_1d(c, GGML_TYPE_I32, Pn);
            ggml_tensor * t_wp   = ggml_new_tensor_2d(c, GGML_TYPE_F32, 1, Pn);

            // Stream this layer's experts in bulk before building the graph.
            // Sequential 16 MB reads instead of 4 KiB mmap faults: 7+ GB/s
            // against 0.13. The decode cache is untouched.
            // The H2D upload below overwrites the single device staging buffer
            // that the PREVIOUS layer's MoE read from. ggml_backend_graph_compute
            // syncs the backend's own stream, but the CUDA set_tensor path copies
            // on cudaStreamPerThread -- a different stream -- so nothing orders
            // the overwrite against work still draining on the compute stream.
            if (pf_.on_device()) ggml_backend_synchronize(w_.backend());
            if (!pf_.load_layer(il, err)) { ggml_free(c); if (ibuf) ggml_backend_buffer_free(ibuf); if (ictx) ggml_free(ictx); return false; }
            // Kick the next layer's bulk read now: it runs on the streamer's
            // reader thread while this layer uploads and computes. Wrapping to
            // layer 0 at the end covers the next ubatch -- and, after the last
            // one, the next turn's prefill.
            pf_.prefetch_layer((il + 1) % hp_.n_layer);

            const bool moe_on_gpu = pf_.on_device();
            ggml_tensor * mh_in  = moe_on_gpu ? d_cur_     : h_cur_;
            ggml_tensor * mh_out = moe_on_gpu ? d_partial_ : h_partial_;

            // One ggml_mul_mat_id per part over the staged layer -- the GGUF
            // tensor layout, [.., .., 512] -- with the routing as its ids. The
            // CUDA MMQ path gathers each expert's tokens on the device, so the
            // host-side permutation, the ~1,500 per-expert matmuls and the
            // [n_embd, 10T] intermediates all go. QWFN_LEGACY_PREFILL_MOE=1
            // keeps the grouped path for comparison.
            static const bool legacy_pf_moe = getenv("QWFN_LEGACY_PREFILL_MOE") != nullptr;
            const bool pf_by_id = moe_on_gpu && !legacy_pf_moe;
            size_t n_groups = 0;
            ggml_tensor * t_ids = nullptr, * t_w = nullptr;
            if (pf_by_id) {
                t_ids = ggml_new_tensor_2d(c, GGML_TYPE_I32, U, T);   ggml_set_input(t_ids);
                t_w   = ggml_new_tensor_3d(c, GGML_TYPE_F32, 1, U, T); ggml_set_input(t_w);
                ggml_tensor * o = moe_id_graph(c, pf_.staged_tier(), t_ids, t_w, v2(c, mh_in),
                                               n_embd, hp_.n_ff_exp, (int) U, T);
                ggml_build_forward_expand(g, ggml_cpy(c, o, v2(c, mh_out)));
            } else {
                std::vector<graph_builder::expert_group> groups;
                for (int64_t k = 0; k < Pn; ) {
                    int64_t k2 = k;
                    while (k2 < Pn && pairs[k2].e == pairs[k].e) k2++;
                    graph_builder::expert_group gr;
                    ggml_tensor * a[EXPERT_NPARTS];
                    a[EXPERT_GATE] = ggml_new_tensor_2d(c, pf_.part_type(EXPERT_GATE), n_embd, hp_.n_ff_exp);
                    a[EXPERT_UP]   = ggml_new_tensor_2d(c, pf_.part_type(EXPERT_UP),   n_embd, hp_.n_ff_exp);
                    a[EXPERT_DOWN] = ggml_new_tensor_2d(c, pf_.part_type(EXPERT_DOWN), hp_.n_ff_exp, n_embd);
                    for (int q = 0; q < EXPERT_NPARTS; q++) {
                        a[q]->buffer = pf_.buffer();
                        a[q]->data   = (void *) pf_.part_ptr((uint32_t) pairs[k].e, (expert_part) q);
                    }
                    gr.gate = a[EXPERT_GATE]; gr.up = a[EXPERT_UP]; gr.down = a[EXPERT_DOWN];
                    gr.off = k; gr.cnt = k2 - k;
                    groups.push_back(gr);
                    k = k2;
                }
                n_groups = groups.size();

                // Diagnostic: with disjoint writes the group order no longer affects
                // the result, so reversing it separates "expert 511 is bad" from
                // "the last group processed is bad".
                if (getenv("QWFN_REVERSE_GROUPS")) std::reverse(groups.begin(), groups.end());

                ggml_tensor * o = gb.moe_apply_batched(v2(c, mh_in), il, groups,
                                                       t_perm, t_inv, t_wp, T, (int) U);
                ggml_build_forward_expand(g, ggml_cpy(c, o, v2(c, mh_out)));
            }

            // The permutation tensors are graph inputs: allocate, then fill.
            ggml_gallocr_t pga = moe_on_gpu ? galloc_pf_ : galloc_cpu_;
            if (!ggml_gallocr_alloc_graph(pga, g)) {
                err = "prefill galloc failed"; ggml_free(c); if (ibuf) ggml_backend_buffer_free(ibuf); if (ictx) ggml_free(ictx); return false;
            }
            ggml_backend_tensor_get(t_cur_, xfer_.data(), 0, (size_t) n_embd * T * sizeof(float));
            if (getenv("QWFN_TRACE_PREFILL")) {
                const size_t nf = (size_t) n_embd * T;
                double sum = 0; int n_nan = 0; float mx = 0;
                for (size_t q = 0; q < nf; q++) {
                    const float v = xfer_[q];
                    if (!std::isfinite(v)) n_nan++;
                    if (std::fabs(v) > mx) mx = std::fabs(v);
                    sum += v;
                }
                fprintf(stderr, "[trace] layer %2u moe_IN   sum=%.6f nonfinite=%d absmax=%.3f\n",
                        il, sum, n_nan, mx);
            }
            ggml_backend_tensor_set(mh_in, xfer_.data(), 0, (size_t) n_embd * T * sizeof(float));
            if (pf_by_id) {
                ggml_backend_tensor_set(t_ids, sel_.data(), 0, (size_t) U * T * sizeof(int32_t));
                ggml_backend_tensor_set(t_w,   wgt_.data(), 0, (size_t) U * T * sizeof(float));
            } else {
                ggml_backend_tensor_set(t_perm, perm.data(), 0, perm.size() * 4);
                ggml_backend_tensor_set(t_inv,  inv.data(),  0, inv.size()  * 4);
                ggml_backend_tensor_set(t_wp,   wp.data(),   0, wp.size()   * 4);
            }
            if (ggml_backend_graph_compute(moe_on_gpu ? w_.backend() : wh_.backend(), g)
                    != GGML_STATUS_SUCCESS) {
                err = "prefill compute failed"; ggml_free(c); if (ibuf) ggml_backend_buffer_free(ibuf); if (ictx) ggml_free(ictx); return false;
            }
            ggml_free(c);

            ggml_backend_tensor_get(mh_out, xfer_.data(), 0, (size_t) n_embd * T * sizeof(float));
            if (getenv("QWFN_TRACE_PREFILL")) {
                const size_t nf = (size_t) n_embd * T;
                uint64_t h = 1469598103934665603ull;
                double sum = 0; int n_nan = 0;
                for (size_t q = 0; q < nf; q++) {
                    const float v = xfer_[q];
                    if (!std::isfinite(v)) n_nan++;
                    sum += v;
                    uint32_t b; memcpy(&b, &v, 4);
                    h = (h ^ b) * 1099511628211ull;
                }
                float mx = 0;
                for (size_t q = 0; q < nf; q++)
                    if (std::isfinite(xfer_[q]) && std::fabs(xfer_[q]) > mx) mx = std::fabs(xfer_[q]);
                fprintf(stderr, "[trace] layer %2u moe_out  sum=%.6f nonfinite=%d absmax=%.3f groups=%zu\n",
                        il, sum, n_nan, mx, n_groups);
                if (n_nan > 0 && n_nan < 100) {
                    // Which (token, channel) went bad, and which expert served
                    // that token -- that identifies the offending group.
                    fprintf(stderr, "          bad at:");
                    int shown = 0;
                    for (size_t q = 0; q < nf && shown < 24; q++) {
                        if (std::isfinite(xfer_[q])) continue;
                        const size_t tokn = q / (size_t) n_embd, ch = q % (size_t) n_embd;
                        fprintf(stderr, " t%zu/c%zu", tokn, ch);
                        shown++;
                    }
                    fprintf(stderr, "\n");
                    // Which expert do ALL the offending tokens have in common?
                    std::vector<int> tally(hp_.n_expert, 0);
                    int n_bad_tok = 0;
                    for (int64_t tk = 0; tk < T; tk++) {
                        bool bad = false;
                        for (int64_t ch = 0; ch < n_embd; ch++)
                            if (!std::isfinite(xfer_[tk * n_embd + ch])) { bad = true; break; }
                        if (!bad) continue;
                        n_bad_tok++;
                        for (uint32_t u = 0; u < hp_.n_expert_used; u++)
                            tally[sel_[tk * hp_.n_expert_used + u]]++;
                    }
                    fprintf(stderr, "          %d bad tokens; experts common to ALL:", n_bad_tok);
                    for (uint32_t e = 0; e < hp_.n_expert; e++)
                        if (tally[e] == n_bad_tok) fprintf(stderr, " %u", e);
                    fprintf(stderr, "  (n_expert=%u)\n", hp_.n_expert);
                }
                (void) h;
            }
            ggml_backend_tensor_set(t_pc_,   xfer_.data(), 0, (size_t) n_embd * T * sizeof(float));
            ggml_backend_tensor_set(t_pg_,      zeros_.data(), 0, (size_t) n_embd * T * sizeof(float));

            // Warm the expert cache from this layer's routing while its slices
            // are still in the host staging: the last tokens' experts first.
            if (cfg_.prefill_warm > 0) {
                const auto tw0 = std::chrono::steady_clock::now();
                std::vector<expert_cache::warm_item> items;
                std::vector<int32_t> pos(hp_.n_expert, -1);
                for (int64_t t = T - 1; t >= 0; t--) {
                    for (int64_t j = 0; j < U; j++) {
                        const int32_t e = sel_[t * U + j];
                        if (e < 0 || e >= (int32_t) hp_.n_expert) continue;
                        if (pos[e] >= 0) { items[pos[e]].count++; continue; }
                        if (items.size() >= cfg_.prefill_warm) continue;
                        expert_cache::warm_item it;
                        it.expert = (uint32_t) e; it.count = 1;
                        for (int q = 0; q < EXPERT_NPARTS; q++) it.part[q] = pf_.host_part_ptr((uint32_t) e, (expert_part) q);
                        pos[e] = (int32_t) items.size();
                        items.push_back(it);
                    }
                }
                ec_.warm(il, items.data(), (uint32_t) items.size(), cfg_.prefill_warm_vram);
                t_warm += std::chrono::duration<double>(std::chrono::steady_clock::now() - tw0).count();
            }
        }
    }
    if (!decode) ec_.settle_promotions();   // warm-up promotions landed

    if (deferred_wait_) { if (!ec_.fetch_end()) { err = "expert read failed"; return false; } deferred_wait_ = false; }

    // ---- MTP experiment, decode: score the previous draft against this token,
    // then draft the token after this one from (h of the previous token, this
    // token's embedding) at the previous token's position. Before the head:
    // it overwrites t_hlast_. ----------------------------------------------
    if (mtp_on_ && T == 1) {
        const int32_t tok = hist[n_hist - 1];
        if (mtp_draft_ >= 0) {
            mtp_n++;
            if (tok == mtp_draft_) mtp_acc++;
            for (int k = 0; k < 3; k++) if (tok == mtp_draft_top_[k]) { mtp_top3++; break; }
        }
        mtp_draft_ = -1;
        if (mtp_have_h_ && mtp_kv_valid_ && n_past >= 1) {
            if (!mtp_draft(n_past - 1, 1, mtp_h_rows_ - 1, 0, nullptr, 0, err)) return false;
        }
    }
    // ---- head --------------------------------------------------------------
    {
        ggml_context * c; ggml_cgraph * g; new_ctx(&c, &g);
        graph_builder gb(c, &hp_, &w_); gb.bind(&st_, g, n_past);
        ggml_tensor * r = vres(c, cur_res);
        if (pending) {
            ggml_tensor * pg = late_fold(c, hp_.n_layer - 1, v2(c, t_pg_));
            ggml_tensor * tot = t_rscale_
                ? ggml_add(c, v2(c, t_sh_), ggml_mul(c, ggml_add(c, pg, v2(c, t_pc_)), t_rscale_))
                : ggml_add(c, ggml_add(c, v2(c, t_sh_), pg), v2(c, t_pc_));
            r = gb.hc_combine(r, tot, v2(c, t_inject_));
        }
        if (mtp_on_)   // the wide residual of every position, for the draft head
            ggml_build_forward_expand(g, ggml_cpy(c, r, ggml_view_3d(c, t_hlast_, n_embd, hc, T, t_hlast_->nb[1], t_hlast_->nb[2], 0)));
        // Only the final position produces logits.
        r = ggml_view_3d(c, r, n_embd, hc, 1, r->nb[1], r->nb[2], (size_t) (T - 1) * r->nb[2]);
        ggml_tensor * o = gb.hc_mix(r, -1, false, nullptr);
        ggml_tensor * logits = ggml_mul_mat(c, w_.get("output.weight"), o);
        ggml_set_output(logits);
        ggml_build_forward_expand(g, logits);
        run_on(g, true);
        ggml_backend_tensor_get(logits, logits_.data(), 0, logits_.size() * sizeof(float));
        ggml_free(c);
    }

    if (mtp_on_) {
        mtp_have_h_ = true; mtp_h_rows_ = T;
        if (T > 1) {
            // A batch is the head's prompt only at the start of the sequence: its
            // rows 0..T-2 pair with embeddings 1..T-1 at positions 0..T-2, and the
            // drafts for positions whose target is inside the prompt are scored.
            if (n_past == 0) {
                if (!mtp_draft(0, T - 1, 0, 1, hist + (n_hist - T) + 2, T - 2, err)) return false;
            } else {
                mtp_kv_valid_ = false;   // a mid-sequence batch: the head's KV would have a gap
            }
        }
    }
    if (ibuf) ggml_backend_buffer_free(ibuf);
    if (ictx) ggml_free(ictx);
    n_past_ += T;

    const double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    if (decode) { t_decode += dt; n_decode += T; } else { t_prefill += dt; n_prefill += T; }
    return true;
}

bool engine::mtp_draft(int64_t pos, int64_t n, int64_t h_row, int64_t e_row,
                       const int32_t * actual, int64_t n_actual, std::string & err) {
    const auto t0 = std::chrono::steady_clock::now();
    const int64_t n_embd = hp_.n_embd, hc = hp_.hc_count;
    attn_inputs ai;
    if (!build_attn_inputs(pos, n, ai, err)) return false;
    {
        std::vector<int32_t> p((size_t) 4 * n, 0);
        for (int64_t i = 0; i < n; i++) p[i] = p[n + i] = p[2 * n + i] = (int32_t) (pos + i);
        ggml_backend_tensor_set(t_mtp_pos_, p.data(), 0, p.size() * sizeof(int32_t));
    }
    ggml_init_params ip{}; ip.mem_size = ggml_tensor_overhead() * 16384 + ggml_graph_overhead_custom(16384, false); ip.no_alloc = true;
    ggml_context * c = ggml_init(ip);
    ggml_cgraph *  g = ggml_new_graph_custom(c, 16384, false);
    graph_builder gb(c, &hpm_, &wm_, &w_); gb.bind(&st_mtp_, g, pos);
    ggml_tensor * h = ggml_view_3d(c, t_hlast_, n_embd, hc, n, t_hlast_->nb[1], t_hlast_->nb[2], (size_t) h_row * t_hlast_->nb[2]);
    ggml_tensor * e = ggml_view_2d(c, t_emb_, n_embd, n, t_emb_->nb[1], (size_t) e_row * t_emb_->nb[1]);
    ggml_tensor * pv = ggml_view_1d(c, t_mtp_pos_, 4 * n, 0);
    int sections[4] = { hp_.mrope_sections[0], hp_.mrope_sections[1], hp_.mrope_sections[2], hp_.mrope_sections[3] };
    ggml_tensor * logits = gb.mtp_head(h, e, pv, ai.kq_mask, sections, (int) hpm_.n_layer - 1);
    ggml_set_output(logits);
    ggml_build_forward_expand(g, logits);
    run_on(g, true);
    const int64_t nv = logits->ne[0];
    std::vector<float> lg((size_t) nv * n);
    ggml_backend_tensor_get(logits, lg.data(), 0, lg.size() * sizeof(float));
    for (int64_t j = 0; j < n; j++) {
        const float * col = lg.data() + j * nv;
        int32_t top[3] = { -1, -1, -1 }; float tv[3] = { -INFINITY, -INFINITY, -INFINITY };
        for (int64_t v = 0; v < nv; v++) {
            const float x = col[v];
            if (x > tv[2]) {
                if (x > tv[0])      { tv[2] = tv[1]; top[2] = top[1]; tv[1] = tv[0]; top[1] = top[0]; tv[0] = x; top[0] = (int32_t) v; }
                else if (x > tv[1]) { tv[2] = tv[1]; top[2] = top[1]; tv[1] = x; top[1] = (int32_t) v; }
                else                { tv[2] = x; top[2] = (int32_t) v; }
            }
        }
        if (actual) {
            if (j < n_actual) { mtp_prompt_n++; if (actual[j] == top[0]) mtp_prompt_acc++; }
        } else {
            mtp_draft_ = top[0];
            for (int k = 0; k < 3; k++) mtp_draft_top_[k] = top[k];
        }
    }
    ai.release();
    ggml_free(c);
    t_mtp += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return true;
}

// Confidence-margin buckets for the prediction statistics: router logit
// differences, finer near zero where the routing cut-off lives.
static const float kMarginEdges[engine::SPEC_MARGIN_BUCKETS - 1] =
    { 0.05f, 0.1f, 0.2f, 0.3f, 0.5f, 0.75f, 1.0f, 1.5f, 2.0f, 3.0f, 5.0f };
int engine::margin_bucket(float m) {
    int b = 0;
    while (b < SPEC_MARGIN_BUCKETS - 1 && m >= kMarginEdges[b]) b++;
    return b;
}
float engine::margin_edge(int b) { return b <= 0 ? 0.0f : kMarginEdges[b - 1]; }

} // namespace qwfn
