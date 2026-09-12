// qwfn-logits -- run the full qwen4exp forward pass over a token sequence and
// print the top logits, so the graph can be checked against llama.cpp.
//
// v1 runs everything on the CPU backend with the shards mmap'd: the 55.82 GB of
// routed experts are never resident, only the pages the router touches. Slow,
// but it isolates correctness from the memory-hierarchy work.

#include "qwfn_graph.h"
#include "qwfn_home.h"
#include "qwfn_model.h"
#include "qwfn_ple.h"
#include "qwfn_state.h"
#include "qwfn_weights.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "ggml-alloc.h"
#include "ggml-backend.h"

using namespace qwfn;

static uint16_t f32_to_f16_bits(float f) {
    ggml_fp16_t h = ggml_fp32_to_fp16(f);
    uint16_t o; memcpy(&o, &h, 2); return o;
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: qwfn-logits <shard.gguf> [tok0,tok1,...] [--gpu]\n");
        return 1;
    }
    std::vector<int32_t> tokens;
    bool use_gpu = false, use_qsa = false;
    int top_k_override = 0;
    for (int i = 2; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--gpu") { use_gpu = true; continue; }
        if (a == "--qsa") { use_qsa = true; continue; }
        if (a == "--top-k" && i + 1 < argc) { top_k_override = atoi(argv[++i]); continue; }
        size_t p = 0;
        while (p < a.size()) {
            size_t c = a.find(',', p);
            if (c == std::string::npos) c = a.size();
            tokens.push_back(atoi(a.substr(p, c - p).c_str()));
            p = c + 1;
        }
    }
    if (tokens.empty()) tokens = { 9707, 11, 1879, 0 };   // arbitrary fixed ids

    model_index mi;
    std::string err;
    if (!mi.load(argv[1], err)) { fprintf(stderr, "error: %s\n", err.c_str()); return 1; }
    if (top_k_override > 0) {
        const_cast<hparams &>(mi.hp()).idx_top_k = (uint32_t) top_k_override;
        printf("indexer top_k overridden to %d\n", top_k_override);
    }
    const hparams & hp = mi.hp();
    printf("%s\n\n", hp.summary().c_str());

    weights w;
    if (!w.init(&mi, use_gpu, qwfn::llama_backend_dir(), err)) {
        fprintf(stderr, "backend: %s\n", err.c_str()); return 1;
    }
    printf("backend: %s\n", w.dev_name());
    if (!w.declare_all_mapped(err)) { fprintf(stderr, "map: %s\n", err.c_str()); return 1; }
    printf("mapped %.2f GB of weights (nothing resident until touched)\n", w.bytes() / 1e9);

    const int64_t T      = (int64_t) tokens.size();
    const int64_t n_past = 0;
    const int64_t n_kv   = n_past + T;

    state_config sc;
    sc.n_ctx  = 2048;                 // small: this is a correctness run
    sc.type_k = GGML_TYPE_F16;
    sc.type_v = GGML_TYPE_F16;
    state st;
    if (!st.init(&hp, sc, w.buft(), err)) { fprintf(stderr, "state: %s\n", err.c_str()); return 1; }
    printf("%s\n\n", st.summary().c_str());

    // ---- inputs -----------------------------------------------------------
    ggml_init_params ip{}; ip.mem_size = ggml_tensor_overhead() * 16; ip.no_alloc = true;
    ggml_context * ictx = ggml_init(ip);
    const int64_t n_head_ple = hp.ple_n_head();
    const int64_t mask_rows  = ((T + 63) / 64) * 64;

    ggml_tensor * inp_tok  = ggml_new_tensor_1d(ictx, GGML_TYPE_I32, T);
    ggml_tensor * inp_ple  = ggml_new_tensor_1d(ictx, GGML_TYPE_I32, n_head_ple * T);
    ggml_tensor * inp_pos  = ggml_new_tensor_1d(ictx, GGML_TYPE_I32, T * 4);
    ggml_tensor * kq_mask  = ggml_new_tensor_2d(ictx, GGML_TYPE_F16, n_kv, mask_rows);
    ggml_backend_buffer_t ibuf = ggml_backend_alloc_ctx_tensors_from_buft(ictx, w.buft());
    if (!ibuf) { fprintf(stderr, "input alloc failed\n"); return 1; }

    ggml_backend_tensor_set(inp_tok, tokens.data(), 0, T * 4);

    std::vector<int32_t> rows(n_head_ple * T);
    for (int64_t i = 0; i < T; i++) {
        const ple_rows r = ple_rows_for(hp, tokens.data(), T, i);
        for (uint32_t h = 0; h < r.n; h++) rows[i * n_head_ple + h] = (int32_t) r.row[h];
    }
    ggml_backend_tensor_set(inp_ple, rows.data(), 0, rows.size() * 4);

    // M-RoPE positions are section-major: t.., h.., w.., then zeros.
    std::vector<int32_t> pos(T * 4, 0);
    for (int64_t i = 0; i < T; i++) {
        pos[i] = pos[T + i] = pos[2 * T + i] = (int32_t) (n_past + i);
    }
    ggml_backend_tensor_set(inp_pos, pos.data(), 0, pos.size() * 4);

    std::vector<uint16_t> mask(n_kv * mask_rows, f32_to_f16_bits(-INFINITY));
    for (int64_t i = 0; i < T; i++)
        for (int64_t j = 0; j <= n_past + i; j++)
            mask[i * n_kv + j] = f32_to_f16_bits(0.0f);
    ggml_backend_tensor_set(kq_mask, mask.data(), 0, mask.size() * 2);

    // ---- QSA host inputs ---------------------------------------------------
    // Contiguous single-sequence cache: cell j holds token j, so blocks are just
    // runs of `ratio` cells and the arithmetic collapses to index maths.
    qsa_inputs qsa;
    ggml_context * qctx = nullptr;
    ggml_backend_buffer_t qbuf = nullptr;
    if (use_qsa) {
        uint32_t r = 0;
        for (uint32_t il = 0; il < hp.n_layer; il++)
            if (hp.is_attn_layer(il) && il < hp.compress_ratios.size() && hp.compress_ratios[il] > 0)
                { r = (uint32_t) hp.compress_ratios[il]; break; }
        if (r == 0) { fprintf(stderr, "no compress ratio in metadata; --qsa unavailable\n"); return 1; }

        const int64_t n_blocks = (n_kv + r - 1) / r;
        const int64_t n_bid    = n_kv / r;                 // whole blocks only
        const bool    have_dead = n_bid < n_blocks;
        const int64_t dead_bid  = have_dead ? n_bid : n_blocks - 1;

        ggml_init_params qp{}; qp.mem_size = ggml_tensor_overhead() * 8; qp.no_alloc = true;
        qctx = ggml_init(qp);
        qsa.cell_blk  = ggml_new_tensor_1d(qctx, GGML_TYPE_I32, n_kv);
        qsa.blk_cells = ggml_new_tensor_1d(qctx, GGML_TYPE_I32, r * n_blocks);
        qsa.blk_pos   = ggml_new_tensor_1d(qctx, GGML_TYPE_I32, 4 * n_blocks);
        qsa.bias      = ggml_new_tensor_2d(qctx, GGML_TYPE_F32, n_blocks, T);
        qsa.ratio     = r;
        qsa.n_blocks  = n_blocks;
        qbuf = ggml_backend_alloc_ctx_tensors_from_buft(qctx, w.buft());
        if (!qbuf) { fprintf(stderr, "qsa alloc failed\n"); return 1; }

        std::vector<int32_t> cb(n_kv), bc(r * n_blocks, 0), bp(4 * n_blocks, 0);
        std::vector<float>   bi(n_blocks * T);
        for (int64_t j = 0; j < n_kv; j++) cb[j] = (int32_t) (j < n_bid * (int64_t) r ? j / r : dead_bid);
        for (int64_t b = 0; b < n_bid; b++) {
            for (uint32_t sIdx = 0; sIdx < r; sIdx++) bc[b * r + sIdx] = (int32_t) (b * r + sIdx);
            for (int sec = 0; sec < 4; sec++) bp[sec * n_blocks + b] = (int32_t) (b * r);
        }
        for (int64_t i = 0; i < T; i++) {
            const int64_t q = n_past + i;
            const int64_t tail_start = ((q + 1) / r) * r;   // the ragged tail is always visible
            for (int64_t b = 0; b < n_blocks; b++)
                bi[i * n_blocks + b] = (b >= n_bid) ? -INFINITY
                                     : (b * (int64_t) r >= tail_start ? 1e9f : 0.0f);
            if (have_dead) bi[i * n_blocks + dead_bid] = 1e9f;
        }
        ggml_backend_tensor_set(qsa.cell_blk,  cb.data(), 0, cb.size() * 4);
        ggml_backend_tensor_set(qsa.blk_cells, bc.data(), 0, bc.size() * 4);
        ggml_backend_tensor_set(qsa.blk_pos,   bp.data(), 0, bp.size() * 4);
        ggml_backend_tensor_set(qsa.bias,      bi.data(), 0, bi.size() * 4);
        printf("QSA: ratio %u, %lld blocks, selection width %lld of %lld cells\n",
               r, (long long) n_blocks,
               (long long) std::min<int64_t>(n_kv, (int64_t) hp.idx_top_k + r - 1), (long long) n_kv);
    }

    // ---- graph ------------------------------------------------------------
    ggml_init_params gp{};
    gp.mem_size = ggml_tensor_overhead() * 100000 + ggml_graph_overhead_custom(65536, false);
    gp.no_alloc = true;
    ggml_context * gctx = ggml_init(gp);
    ggml_cgraph * gf = ggml_new_graph_custom(gctx, 65536, false);

    graph_builder gb(gctx, &hp, &w);
    gb.bind(&st, gf, n_past);

    // Names mirror llama.cpp's cb() tags so qwfn-refdump output lines up 1:1.
    const bool dump = getenv("QWFN_DUMP") != nullptr;
    const int  probe_layer = getenv("QWFN_PROBE_LAYER") ? atoi(getenv("QWFN_PROBE_LAYER")) : 0;
    std::vector<std::pair<std::string, ggml_tensor *>> probes;
    auto probe = [&](const char * name, ggml_tensor * t) {
        if (!dump || !t) return t;
        probes.emplace_back(name, t);
        ggml_set_output(t);
        ggml_build_forward_expand(gf, t);
        return t;
    };

    const int64_t hc = hp.hc_count, n_embd = hp.n_embd;
    int sections[4] = { hp.mrope_sections[0], hp.mrope_sections[1], hp.mrope_sections[2], hp.mrope_sections[3] };

    ggml_tensor * inpL = ggml_get_rows(gctx, w.get("token_embd.weight"), inp_tok);   // [2560, T]
    probe("model.input_embed", inpL);

    // 16 gathered rows of 160 dims, laid out head-slowest, concatenate to n_embd.
    ggml_tensor * ple_emb = ggml_get_rows(gctx, w.get("per_layer_token_embd.weight"), inp_ple);
    ple_emb = ggml_reshape_2d(gctx, ple_emb, hp.d_ple * n_head_ple, T);
    probe("ple_embd", ple_emb);

    // The wide residual starts as hc identical copies of the embedding.
    ggml_tensor * res = ggml_repeat_4d(gctx, ggml_reshape_3d(gctx, inpL, n_embd, 1, T),
                                       n_embd, hc, T, 1);
    probe("hc_init", res);

    for (uint32_t il = 0; il < hp.n_layer; il++) {
        const bool is_ple = std::find(hp.ple_layers.begin(), hp.ple_layers.end(), (int32_t) il) != hp.ple_layers.end();
        if (is_ple) res = gb.ple(ple_emb, res, il);

        ggml_tensor * inject = nullptr;
        ggml_tensor * cur = gb.hc_mix(res, il, /*ffn=*/false, &inject);
        if ((int) il == probe_layer) { probe(("hc_mixed-"  + std::to_string(il)).c_str(), cur);
                                       probe(("hc_inject-" + std::to_string(il)).c_str(), inject); }
        cur = hp.is_attn_layer(il) ? gb.sparse_attn(cur, inp_pos, kq_mask, sections, il,
                                                    use_qsa ? &qsa : nullptr)
                                   : gb.deltanet(cur, il);
        if ((int) il == probe_layer) probe(("attn_output-" + std::to_string(il)).c_str(), cur);
        res = gb.hc_combine(res, cur, inject);
        if ((int) il == probe_layer) probe(("hc_combine-" + std::to_string(il)).c_str(), res);

        cur = gb.hc_mix(res, il, /*ffn=*/true, &inject);
        cur = gb.moe(cur, il);
        if ((int) il == probe_layer) probe(("ffn_out-" + std::to_string(il)).c_str(), cur);
        res = gb.hc_combine(res, cur, inject);
        probe(("l_last-" + std::to_string(il)).c_str(), res);
    }

    // The final mixer is the output norm; qwen4exp has no separate one.
    ggml_tensor * out = gb.hc_mix(res, -1, false, nullptr);
    probe("result_norm", out);
    ggml_tensor * logits = ggml_mul_mat(gctx, w.get("output.weight"), out);
    ggml_set_name(logits, "logits");
    ggml_set_output(logits);
    ggml_build_forward_expand(gf, logits);
    printf("graph: %d nodes\n", ggml_graph_n_nodes(gf));

    ggml_gallocr_t galloc = ggml_gallocr_new(w.buft());
    if (!ggml_gallocr_alloc_graph(galloc, gf)) { fprintf(stderr, "galloc failed\n"); return 1; }

    const auto t0 = std::chrono::steady_clock::now();
    if (ggml_backend_graph_compute(w.backend(), gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "compute failed\n"); return 1;
    }
    const double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    printf("forward: %.2f s for %lld tokens (%.2f tok/s)\n\n", dt, (long long) T, T / dt);

    if (dump) {
        printf("=== qwfnfer intermediates ===\n");
        for (auto & [name, t] : probes) {
            const size_t n = ggml_nelements(t);
            std::vector<float> d(n);
            ggml_backend_tensor_get(t, d.data(), 0, n * sizeof(float));
            double sum = 0, amax = 0;
            for (float v : d) { sum += v; if (fabs(v) > amax) amax = fabs(v); }
            printf("  %-22s [%5lld,%3lld,%3lld] sum %18.9f absmax %14.9f first %.6f %.6f %.6f\n",
                   name.c_str(), (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2],
                   sum, amax, d[0], n > 1 ? d[1] : 0.f, n > 2 ? d[2] : 0.f);
        }
        printf("\n");
    }

    // ---- top logits for the last position ---------------------------------
    const int64_t V = logits->ne[0];
    std::vector<float> lg(V);
    ggml_backend_tensor_get(logits, lg.data(), (size_t) (T - 1) * V * sizeof(float), V * sizeof(float));

    std::vector<int> ord(V);
    for (int64_t i = 0; i < V; i++) ord[i] = (int) i;
    std::partial_sort(ord.begin(), ord.begin() + 10, ord.end(),
                      [&](int a, int b) { return lg[a] > lg[b]; });
    double mx = lg[ord[0]], sum = 0;
    for (int64_t i = 0; i < V; i++) sum += exp(lg[i] - mx);
    printf("top-10 next-token logits (last position):\n");
    for (int i = 0; i < 10; i++)
        printf("  %2d. id %6d  logit %9.4f  p %.5f\n", i + 1, ord[i], lg[ord[i]], exp(lg[ord[i]] - mx) / sum);

    ggml_gallocr_free(galloc);
    ggml_free(gctx);
    ggml_backend_buffer_free(ibuf);
    ggml_free(ictx);
    return 0;
}
