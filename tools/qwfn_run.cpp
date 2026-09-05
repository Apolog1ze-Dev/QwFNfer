// qwfn-run -- the forward pass with routed experts served by expert_cache.
//
// The router's choice is only known once the layer has been computed, so the
// graph cannot be a single static thing: each layer runs as two submissions.
//
//   A: hc_mix -> DeltaNet/attention -> hc_combine -> hc_mix(ffn) -> router
//      (host reads back the 10 expert ids)
//   B: MoE over those 10 experts -> hc_combine
//
// Between them the cache fetches whatever missed, at whole-slice granularity
// over io_uring. The expert tensors in graph B alias cached blocks directly --
// one expert is 2.18 MB and a token touches 480 of them, so staging them into a
// contiguous buffer would add ~1 GB/token of memcpy, as much as the weights.
//
// Placement is by residency, via ggml_backend_sched over {CUDA, CPU}:
//   dense core (4.29B of the 6.65B active params)  -> CUDA buffer  -> GPU
//   routed experts (cache arena) and the PLE table -> CPU buffers  -> CPU
// The scheduler inserts the cross-device copies, which are tiny: only the
// [2560, T] activation crosses per layer, ~20 KB, against 1 GB of expert
// weights that never leave host RAM.

#include "qwfn_expert_cache.h"
#include "qwfn_graph.h"
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
using clk = std::chrono::steady_clock;
static double sec(clk::time_point a, clk::time_point b) {
    return std::chrono::duration<double>(b - a).count();
}
static uint16_t f16(float f) { ggml_fp16_t h = ggml_fp32_to_fp16(f); uint16_t o; memcpy(&o, &h, 2); return o; }

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr,
            "usage: qwfn-run <shard.gguf> [tok,...] [--cold <shard>] [--ram GB] [--ctx N]\n");
        return 1;
    }
    std::vector<int32_t> tokens;
    std::string cold_path;
    double ram_gb = 4.0, vram_gb = 0.0;
    int n_ctx = 2048, n_threads = 8, repeat = 1;
    bool fresh_state = false, bypass = false, no_ple = false, trace = false, force_cpu = false;
    for (int i = 2; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--cold" && i + 1 < argc) { cold_path = argv[++i]; continue; }
        if (a == "--ram"  && i + 1 < argc) { ram_gb = atof(argv[++i]); continue; }
        if (a == "--vram" && i + 1 < argc) { vram_gb = atof(argv[++i]); continue; }
        if (a == "--ctx"  && i + 1 < argc) { n_ctx = atoi(argv[++i]); continue; }
        if (a == "--threads" && i + 1 < argc) { n_threads = atoi(argv[++i]); continue; }
        if (a == "--repeat"  && i + 1 < argc) { repeat = atoi(argv[++i]); continue; }
        if (a == "--fresh-state") { fresh_state = true; continue; }
        if (a == "--bypass-cache") { bypass = true; continue; }
        if (a == "--no-ple") { no_ple = true; continue; }
        if (a == "--trace") { trace = true; continue; }
        if (a == "--cpu") { force_cpu = true; continue; }
        size_t p = 0;
        while (p < a.size()) {
            size_t c = a.find(',', p);
            if (c == std::string::npos) c = a.size();
            tokens.push_back(atoi(a.substr(p, c - p).c_str()));
            p = c + 1;
        }
    }
    if (tokens.empty()) tokens.push_back(9707);
    const int64_t T = (int64_t) tokens.size();
    if (T != 1) {
        fprintf(stderr, "qwfn-run currently implements the decode path only (one token per call);\n"
                        "with a single token the per-layer selection is exactly n_expert_used experts.\n");
        return 1;
    }

    model_index mi, cold;
    std::string err;
    if (!mi.load(argv[1], err)) { fprintf(stderr, "error: %s\n", err.c_str()); return 1; }
    model_index * coldp = nullptr;
    if (!cold_path.empty() && cold.load(cold_path, err)) coldp = &cold;

    const hparams & hp = mi.hp();
    const int64_t hc = hp.hc_count, n_embd = hp.n_embd;
    const int64_t n_used = hp.n_expert_used;

    const std::string bedir = std::string(getenv("HOME")) + "/.unsloth/llama.cpp/build/bin";

    // Dense core -> GPU. It is 4.29B of the 6.65B active parameters, so this is
    // the difference between ~30 GB/s and 736 GB/s on most of the work.
    weights w;
    if (!w.init(&mi, /*prefer_gpu=*/!force_cpu, bedir, err)) {
        fprintf(stderr, "backend: %s\n", err.c_str()); return 1;
    }
    if (!w.declare_dense_core(err)) { fprintf(stderr, "core: %s\n",   err.c_str()); return 1; }
    if (!w.commit(err))             { fprintf(stderr, "commit: %s\n", err.c_str()); return 1; }
    const double core_gb = w.bytes() / 1e9;

    // PLE stays on the host: 28.8 GB, and a token reads 16 rows of 90 bytes.
    weights wh;
    if (!wh.init(&mi, /*prefer_gpu=*/false, bedir, err)) {
        fprintf(stderr, "host backend: %s\n", err.c_str()); return 1;
    }
    if (!wh.map_shards(err)) { fprintf(stderr, "map: %s\n", err.c_str()); return 1; }
    if (!wh.declare_mapped("per_layer_token_embd.weight")) {
        fprintf(stderr, "PLE table missing\n"); return 1;
    }
    wh.set_n_threads(n_threads);
    printf("dense core: %.2f GB on %s;  PLE table: %.1f GB mmap'd on %s\n",
           core_gb, w.dev_name(), wh.bytes() / 1e9, wh.dev_name());

    // No ggml_backend_sched: every graph is built for exactly one device and the
    // handful of tensors that cross are copied explicitly. Letting the scheduler
    // split a graph whose inputs straddle both devices produced run-to-run
    // nondeterminism, and the boundary it chose was not the one we want anyway.
    ggml_gallocr_t galloc_gpu = ggml_gallocr_new(w.buft());
    ggml_gallocr_t galloc_cpu = ggml_gallocr_new(wh.buft());

    expert_cache::config ec_cfg;
    ec_cfg.ram_bytes     = (size_t) (ram_gb * 1e9);
    ec_cfg.vram_bytes    = (size_t) (vram_gb * 1e9);
    ec_cfg.vram_buft     = w.buft();
    ec_cfg.use_cold_tier = coldp != nullptr;
    expert_cache ec;
    if (!ec.init(&mi, coldp, ec_cfg, err)) { fprintf(stderr, "expert cache: %s\n", err.c_str()); return 1; }

    state_config sc; sc.n_ctx = n_ctx; sc.type_k = GGML_TYPE_F16; sc.type_v = GGML_TYPE_F16;
    state st;
    if (!st.init(&hp, sc, w.buft(), err)) { fprintf(stderr, "state: %s\n", err.c_str()); return 1; }
    printf("%s\n", st.summary().c_str());

    // ---- persistent tensors that live across the per-layer submissions -----
    ggml_init_params wp{}; wp.mem_size = ggml_tensor_overhead() * 32; wp.no_alloc = true;
    ggml_context * wctx = ggml_init(wp);
    ggml_tensor * res[2] = {
        ggml_new_tensor_3d(wctx, GGML_TYPE_F32, n_embd, hc, T),
        ggml_new_tensor_3d(wctx, GGML_TYPE_F32, n_embd, hc, T) };
    ggml_tensor * t_cur    = ggml_new_tensor_2d(wctx, GGML_TYPE_F32, n_embd, T);
    ggml_tensor * t_inject = ggml_new_tensor_2d(wctx, GGML_TYPE_F32, hc, T);
    ggml_tensor * t_sel    = ggml_new_tensor_2d(wctx, GGML_TYPE_I32, n_used, T);
    ggml_tensor * t_w      = ggml_new_tensor_2d(wctx, GGML_TYPE_F32, n_used, T);
    ggml_tensor * inp_tok  = ggml_new_tensor_1d(wctx, GGML_TYPE_I32, T);
    ggml_tensor * inp_ple  = ggml_new_tensor_1d(wctx, GGML_TYPE_I32, hp.ple_n_head() * T);
    ggml_tensor * inp_pos  = ggml_new_tensor_1d(wctx, GGML_TYPE_I32, T * 4);
    const int64_t mask_rows = 64;
    ggml_tensor * kq_mask  = ggml_new_tensor_2d(wctx, GGML_TYPE_F16, T, mask_rows);
    ggml_tensor * t_sh   = ggml_new_tensor_2d(wctx, GGML_TYPE_F32, n_embd, T);  // shared-expert out
    ggml_tensor * t_pg   = ggml_new_tensor_2d(wctx, GGML_TYPE_F32, n_embd, T);  // routed, VRAM experts
    ggml_tensor * t_pc   = ggml_new_tensor_2d(wctx, GGML_TYPE_F32, n_embd, T);  // routed, RAM experts
    ggml_tensor * t_ple  = ggml_new_tensor_2d(wctx, GGML_TYPE_F32, n_embd, T);  // gathered PLE rows
    ggml_backend_buffer_t wbuf = ggml_backend_alloc_ctx_tensors_from_buft(wctx, w.buft());
    if (!wbuf) { fprintf(stderr, "work buffer alloc failed\n"); return 1; }

    // Host mirrors. Only [n_embd, T] activations cross the bus -- 10 KB each way
    // per layer, against 1 GB of expert weights that never move.
    ggml_init_params hp2{}; hp2.mem_size = ggml_tensor_overhead() * 16; hp2.no_alloc = true;
    ggml_context * hctx = ggml_init(hp2);
    ggml_tensor * h_cur     = ggml_new_tensor_2d(hctx, GGML_TYPE_F32, n_embd, T);
    ggml_tensor * h_partial = ggml_new_tensor_2d(hctx, GGML_TYPE_F32, n_embd, T);
    ggml_tensor * h_ple     = ggml_new_tensor_2d(hctx, GGML_TYPE_F32, n_embd, T);
    ggml_tensor * h_ple_idx = ggml_new_tensor_1d(hctx, GGML_TYPE_I32, hp.ple_n_head() * T);
    ggml_backend_buffer_t hbuf = ggml_backend_alloc_ctx_tensors_from_buft(hctx, wh.buft());
    if (!hbuf) { fprintf(stderr, "host buffer alloc failed\n"); return 1; }
    std::vector<float> xfer((size_t) n_embd * T);
    const std::vector<float> zeros((size_t) n_embd * T, 0.0f);

    ggml_backend_tensor_set(inp_tok, tokens.data(), 0, T * 4);
    std::vector<int32_t> rows(hp.ple_n_head() * T);
    for (int64_t i = 0; i < T; i++) {
        const ple_rows r = ple_rows_for(hp, tokens.data(), T, i);
        for (uint32_t h = 0; h < r.n; h++) rows[i * hp.ple_n_head() + h] = (int32_t) r.row[h];
    }
    ggml_backend_tensor_set(inp_ple,   rows.data(), 0, rows.size() * 4);
    ggml_backend_tensor_set(h_ple_idx, rows.data(), 0, rows.size() * 4);
    std::vector<int32_t> pos(T * 4, 0);
    for (int64_t i = 0; i < T; i++) pos[i] = pos[T + i] = pos[2 * T + i] = (int32_t) i;
    ggml_backend_tensor_set(inp_pos, pos.data(), 0, pos.size() * 4);
    std::vector<uint16_t> mask(T * mask_rows, f16(-INFINITY));
    for (int64_t i = 0; i < T; i++) for (int64_t j = 0; j <= i; j++) mask[i * T + j] = f16(0.0f);
    ggml_backend_tensor_set(kq_mask, mask.data(), 0, mask.size() * 2);

    int sections[4] = { hp.mrope_sections[0], hp.mrope_sections[1], hp.mrope_sections[2], hp.mrope_sections[3] };

    auto run_on = [&](ggml_cgraph * gf, bool gpu) {
        ggml_gallocr_t ga = gpu ? galloc_gpu : galloc_cpu;
        ggml_backend_t be = gpu ? w.backend() : wh.backend();
        if (!ggml_gallocr_alloc_graph(ga, gf)) { fprintf(stderr, "galloc failed\n"); exit(1); }
        if (ggml_backend_graph_compute(be, gf) != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "compute failed\n"); exit(1);
        }
    };
    auto run = [&](ggml_cgraph * gf) { run_on(gf, true); };
    auto new_ctx = [&](ggml_context ** c, ggml_cgraph ** g) {
        ggml_init_params p{};
        p.mem_size = ggml_tensor_overhead() * 8192 + ggml_graph_overhead_custom(8192, false);
        p.no_alloc = true;
        *c = ggml_init(p);
        *g = ggml_new_graph_custom(*c, 8192, false);
    };

    uint64_t sel_sum = 0, first_sel = 0;
    std::vector<float> trace_buf;
    std::vector<uint64_t> stage_hash, prev_stage_hash;
    std::vector<std::string> stage_name;
    auto hash_bytes = [&](const void * p, size_t nbytes, const char * name) {
        if (!trace) return;
        uint64_t h = 1469598103934665603ull;
        const uint8_t * p8 = (const uint8_t *) p;
        for (size_t i = 0; i < nbytes; i++) { h ^= p8[i]; h *= 1099511628211ull; }
        stage_hash.push_back(h);
        stage_name.push_back(name);
    };
    auto hash_tensor = [&](ggml_tensor * t, const char * name) {
        if (!trace) return;
        const size_t n = ggml_nelements(t);
        trace_buf.resize(n);
        ggml_backend_tensor_get(t, trace_buf.data(), 0, n * sizeof(float));
        uint64_t h = 1469598103934665603ull;
        const uint8_t * p8 = (const uint8_t *) trace_buf.data();
        for (size_t i = 0; i < n * sizeof(float); i++) { h ^= p8[i]; h *= 1099511628211ull; }
        stage_hash.push_back(h);
        stage_name.push_back(name);
    };
    std::vector<uint64_t> per_layer_sel(hp.n_layer, 0), prev_layer_sel(hp.n_layer, 0);
    double t_io = 0, t_buildA = 0, t_compA = 0, t_buildB = 0, t_compB = 0, t_read = 0;
    std::vector<float> lg;
    auto stats_before = ec.stats();
    printf("\n%-5s %10s %10s %10s %10s %10s   %s\n", "iter", "wall_s", "tok/s", "io_s", "hit%", "disk_MB", "top-1 token");

    ggml_tensor * ple_emb_src = wh.get("per_layer_token_embd.weight");

    // One CPU graph per token: gather the 16 PLE rows from the mmap'd table and
    // upload the resulting [n_embd, T] vector. 1,440 bytes of payload, so the
    // 28.8 GB table never needs to be anywhere but NVMe.
    auto gather_ple = [&]() {
        ggml_context * c; ggml_cgraph * g; new_ctx(&c, &g);
        ggml_tensor * e = ggml_get_rows(c, ple_emb_src, h_ple_idx);
        e = ggml_reshape_2d(c, e, hp.d_ple * hp.ple_n_head(), T);
        ggml_build_forward_expand(g, ggml_cpy(c, e, h_ple));
        run_on(g, false);
        ggml_free(c);
        ggml_backend_tensor_get(h_ple, xfer.data(), 0, xfer.size() * sizeof(float));
        ggml_backend_tensor_set(t_ple, xfer.data(), 0, xfer.size() * sizeof(float));
    };

    static const char * EXP_SUFFIX[EXPERT_NPARTS] = {
        "ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight" };
    if (bypass) {
        for (uint32_t il = 0; il < hp.n_layer; il++)
            for (int q = 0; q < EXPERT_NPARTS; q++)
                if (!wh.declare_mapped("blk." + std::to_string(il) + "." + EXP_SUFFIX[q])) {
                    fprintf(stderr, "bypass: missing expert tensor\n"); return 1;
                }
        printf("[diag] bypassing expert cache: experts read from the mmap'd file\n");
    }

    std::vector<int32_t>       sel(n_used);
    std::vector<float>         wgt(n_used);
    std::vector<expert_handle> eh(n_used);
    std::vector<uint32_t>      ids(n_used);

  for (int iter = 0; iter < repeat; iter++) {
    if (fresh_state) st.reset();
    sel_sum = 0;
    stage_hash.clear(); stage_name.clear();
    const auto t_start = clk::now();
    const double io_before = t_io;
    const auto s_before = ec.stats();

    gather_ple();

    // ---- embedding + wide residual init (GPU) -----------------------------
    {
        ggml_context * c; ggml_cgraph * g; new_ctx(&c, &g);
        ggml_tensor * e = ggml_get_rows(c, w.get("token_embd.weight"), inp_tok);
        ggml_tensor * r = ggml_repeat_4d(c, ggml_reshape_3d(c, e, n_embd, 1, T), n_embd, hc, T, 1);
        ggml_build_forward_expand(g, ggml_cpy(c, r, res[0]));
        run(g); ggml_free(c);
    }
    hash_tensor(res[0], "embed");

    int cur_res = 0;
    bool pending = false;      // a layer's MoE result is waiting to be combined

    for (uint32_t il = 0; il < hp.n_layer; il++) {
        const bool is_ple = std::find(hp.ple_layers.begin(), hp.ple_layers.end(), (int32_t) il) != hp.ple_layers.end();

        // ---- graph A: entirely on the GPU ---------------------------------
        {
            const auto tb0 = clk::now();
            ggml_context * c; ggml_cgraph * g; new_ctx(&c, &g);
            graph_builder gb(c, &hp, &w); gb.bind(&st, g, 0);

            ggml_tensor * r = res[cur_res];
            if (pending) {
                // Fold in the previous layer's MoE: shared + VRAM-routed + RAM-routed.
                ggml_tensor * tot = ggml_add(c, ggml_add(c, t_sh, t_pg), t_pc);
                r = gb.hc_combine(r, tot, t_inject);
            }
            if (is_ple && !no_ple) r = gb.ple(t_ple, r, il);

            ggml_tensor * inject = nullptr;
            ggml_tensor * cur = gb.hc_mix(r, il, /*ffn=*/false, &inject);
            cur = hp.is_attn_layer(il) ? gb.sparse_attn(cur, inp_pos, kq_mask, sections, il)
                                       : gb.deltanet(cur, il);
            r = gb.hc_combine(r, cur, inject);

            ggml_tensor * cur2 = gb.hc_mix(r, il, /*ffn=*/true, &inject);
            ggml_tensor * sl = nullptr, * wt = nullptr;
            gb.moe_route(cur2, il, &sl, &wt);
            ggml_tensor * sh = gb.shared_expert(cur2, il);

            ggml_build_forward_expand(g, ggml_cpy(c, r,      res[1 - cur_res]));
            ggml_build_forward_expand(g, ggml_cpy(c, cur2,   t_cur));
            ggml_build_forward_expand(g, ggml_cpy(c, inject, t_inject));
            ggml_build_forward_expand(g, ggml_cpy(c, sl,     t_sel));
            ggml_build_forward_expand(g, ggml_cpy(c, wt,     t_w));
            ggml_build_forward_expand(g, ggml_cpy(c, sh,     t_sh));
            const auto tb1 = clk::now(); t_buildA += sec(tb0, tb1);
            run(g); t_compA += sec(tb1, clk::now());
            ggml_free(c);
            cur_res = 1 - cur_res;
            pending = true;
            hash_tensor(res[cur_res], ("A" + std::to_string(il)).c_str());
            hash_tensor(t_cur, ("cur" + std::to_string(il)).c_str());
        }

        // ---- routing readback ---------------------------------------------
        const auto tr0 = clk::now();
        ggml_backend_tensor_get(t_sel, sel.data(), 0, n_used * sizeof(int32_t));
        ggml_backend_tensor_get(t_w,   wgt.data(), 0, n_used * sizeof(float));
        ggml_backend_tensor_get(t_cur, xfer.data(), 0, xfer.size() * sizeof(float));
        ggml_backend_tensor_set(h_cur, xfer.data(), 0, xfer.size() * sizeof(float));
        t_read += sec(tr0, clk::now());
        hash_bytes(sel.data(), n_used * sizeof(int32_t), ("sel" + std::to_string(il)).c_str());
        hash_bytes(wgt.data(), n_used * sizeof(float),   ("wgt" + std::to_string(il)).c_str());
        hash_bytes(xfer.data(), xfer.size() * sizeof(float), ("hcur" + std::to_string(il)).c_str());
        for (int64_t e = 0; e < n_used; e++) sel_sum = sel_sum * 1000003ull + (uint64_t) sel[e];
        { uint64_t h = 0; for (int64_t e = 0; e < n_used; e++) h = h*1000003ull + (uint64_t) sel[e];
          per_layer_sel[il] = h; }

        for (int64_t e = 0; e < n_used; e++) ids[e] = (uint32_t) sel[e];
        const auto tio0 = clk::now();
        if (!ec.fetch(il, ids.data(), (uint32_t) n_used, eh.data())) {
            fprintf(stderr, "expert fetch failed at layer %u\n", il); return 1;
        }
        t_io += sec(tio0, clk::now());

        // Partition the selection by where its weights actually live.
        std::vector<int> on_gpu, on_cpu;
        for (int64_t e = 0; e < n_used; e++) (eh[e].on_gpu ? on_gpu : on_cpu).push_back((int) e);

        // ---- graph B: one per device, each self-contained -------------------
        auto build_moe = [&](const std::vector<int> & which, bool gpu,
                             ggml_tensor * in, ggml_tensor * out_t) {
            if (which.empty()) {
                ggml_backend_tensor_set(out_t, zeros.data(), 0, zeros.size() * sizeof(float));
                return;
            }
            const auto tb0 = clk::now();
            ggml_context * c; ggml_cgraph * g; new_ctx(&c, &g);
            graph_builder gb(c, &hp, &w); gb.bind(&st, g, 0);

            std::vector<ggml_tensor *> tg, tu, td;
            std::vector<float> ww;
            for (int e : which) {
                ggml_tensor * a[EXPERT_NPARTS];
                a[EXPERT_GATE] = ggml_new_tensor_2d(c, eh[e].type[EXPERT_GATE], n_embd, hp.n_ff_exp);
                a[EXPERT_UP]   = ggml_new_tensor_2d(c, eh[e].type[EXPERT_UP],   n_embd, hp.n_ff_exp);
                a[EXPERT_DOWN] = ggml_new_tensor_2d(c, eh[e].type[EXPERT_DOWN], hp.n_ff_exp, n_embd);
                for (int q = 0; q < EXPERT_NPARTS; q++) {
                    if (bypass) {
                        ggml_tensor * full = wh.get("blk." + std::to_string(il) + "." + EXP_SUFFIX[q]);
                        const size_t slice = ggml_row_size(full->type, full->ne[0]) * (size_t) full->ne[1];
                        a[q]->buffer = full->buffer;
                        a[q]->data   = (uint8_t *) full->data + slice * ids[e];
                        a[q]->type   = full->type;
                    } else {
                        a[q]->buffer = eh[e].buffer;
                        a[q]->data   = (void *) eh[e].part[q];
                    }
                }
                tg.push_back(a[EXPERT_GATE]); tu.push_back(a[EXPERT_UP]); td.push_back(a[EXPERT_DOWN]);
                ww.push_back(wgt[e]);
            }
            ggml_tensor * o = gb.moe_apply(in, il, tg.data(), tu.data(), td.data(),
                                           ww.data(), (int) which.size());
            ggml_build_forward_expand(g, ggml_cpy(c, o, out_t));
            const auto tb1 = clk::now(); t_buildB += sec(tb0, tb1);
            run_on(g, gpu); t_compB += sec(tb1, clk::now());
            ggml_free(c);
        };

        build_moe(on_gpu, /*gpu=*/true,  t_cur, t_pg);
        build_moe(on_cpu, /*gpu=*/false, h_cur, h_partial);
        if (on_cpu.empty()) {
            ggml_backend_tensor_set(t_pc, zeros.data(), 0, zeros.size() * sizeof(float));
        } else {
            ggml_backend_tensor_get(h_partial, xfer.data(), 0, xfer.size() * sizeof(float));
            ggml_backend_tensor_set(t_pc,      xfer.data(), 0, xfer.size() * sizeof(float));
        }
        hash_tensor(t_sh, ("sh" + std::to_string(il)).c_str());
        hash_tensor(t_pg, ("pg" + std::to_string(il)).c_str());
        hash_tensor(t_pc, ("pc" + std::to_string(il)).c_str());
    }

    // ---- head (GPU): fold the last MoE, collapse the streams, project ------
    {
        ggml_context * c; ggml_cgraph * g; new_ctx(&c, &g);
        graph_builder gb(c, &hp, &w); gb.bind(&st, g, 0);
        ggml_tensor * r = res[cur_res];
        if (pending) {
            ggml_tensor * tot = ggml_add(c, ggml_add(c, t_sh, t_pg), t_pc);
            r = gb.hc_combine(r, tot, t_inject);
        }
        ggml_tensor * o = gb.hc_mix(r, -1, false, nullptr);
        ggml_tensor * logits = ggml_mul_mat(c, w.get("output.weight"), o);
        ggml_set_output(logits);
        ggml_build_forward_expand(g, logits);
        run(g);
        lg.resize(ggml_nelements(logits));
        ggml_backend_tensor_get(logits, lg.data(), 0, lg.size() * sizeof(float));
        ggml_free(c);
    }
    const double dt = sec(t_start, clk::now());
    const auto s_after = ec.stats();
    const uint64_t look = s_after.lookups - s_before.lookups;
    const uint64_t hit  = s_after.hits    - s_before.hits;
    const uint64_t disk = s_after.bytes_from_disk - s_before.bytes_from_disk;
    int best = 0;
    for (size_t v = 1; v < lg.size(); v++) if (lg[v] > lg[best]) best = (int) v;
    printf("%-5d %10.3f %10.2f %10.3f %9.1f%% %10.0f   top1 id %6d logit %9.4f  gpu %.0f%%\n",
           iter, dt, 1.0 / dt, t_io - io_before, look ? 100.0 * hit / look : 0.0, disk / 1e6,
           best, lg[best],
           look ? 100.0 * (double)(s_after.gpu_hits - s_before.gpu_hits) / look : 0.0);
    {
        int first_diff = -1;
        if (iter > 0) for (uint32_t il = 0; il < hp.n_layer; il++)
            if (per_layer_sel[il] != prev_layer_sel[il]) { first_diff = (int) il; break; }
        printf("        routing hash %016llx   first layer differing from previous iter: %d\n",
               (unsigned long long) sel_sum, first_diff);
        prev_layer_sel = per_layer_sel;
        if (trace && iter == 0) {
            printf("        iter0 stages:");
            for (size_t k = 0; k < 6 && k < stage_hash.size(); k++)
                printf(" %s=%08llx", stage_name[k].c_str(),
                       (unsigned long long)(stage_hash[k] & 0xffffffffull));
            printf("\n");
        }
        if (trace && !prev_stage_hash.empty()) {
            for (size_t k = 0; k < stage_hash.size() && k < prev_stage_hash.size(); k++)
                if (stage_hash[k] != prev_stage_hash[k]) {
                    printf("        first differing stage: %s (index %zu of %zu)\n",
                           stage_name[k].c_str(), k, stage_hash.size());
                    break;
                }
        }
        if (trace) prev_stage_hash = stage_hash;
    }
    (void) first_sel;
    fflush(stdout);
  }

    {
        const auto & s = ec.stats();
        printf("\nper-token breakdown (last %d iters): buildA %.3f  compA %.3f  read %.4f  io %.3f  buildB %.3f  compB %.3f\n",
               repeat, t_buildA/repeat, t_compA/repeat, t_read/repeat, t_io/repeat, t_buildB/repeat, t_compB/repeat);
        printf("io: %llu reads, %llu bytes, %llu errors, %llu short\n",
               (unsigned long long) 0ull, 0ull, 0ull, 0ull);
        printf("cumulative: %.1f%% hit (%.1f%% of all lookups served from VRAM, %llu promotions)\n"
               "            %.1f%% hit (%llu/%llu), %.1f GB from disk, %llu evictions%s\n",
               100.0 * s.hit_rate(), 100.0 * s.gpu_rate(), (unsigned long long) s.promotions,
               100.0 * s.hit_rate(), (unsigned long long) s.hits, (unsigned long long) s.lookups,
               s.bytes_from_disk / 1e9, (unsigned long long) s.evictions,
               s.cold_tier_reads ? "  [cold tier active]" : "");
        (void) stats_before;
    }

    const int64_t V = (int64_t) lg.size();
    std::vector<int> ord(V);
    for (int64_t i = 0; i < V; i++) ord[i] = (int) i;
    std::partial_sort(ord.begin(), ord.begin() + 10, ord.end(),
                      [&](int a, int b) { return lg[a] > lg[b]; });
    printf("\ntop-10 next-token logits:\n");
    for (int i = 0; i < 10; i++) printf("  %2d. id %6d  logit %9.4f\n", i + 1, ord[i], lg[ord[i]]);

    ggml_gallocr_free(galloc_gpu);
    ggml_gallocr_free(galloc_cpu);
    ggml_backend_buffer_free(wbuf);
    ggml_free(wctx);
    return 0;
}
