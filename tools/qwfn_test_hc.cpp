// qwfn-test-hc -- run the hyper-connection mixer on real layer-0 weights and dump
// the result, so it can be checked against an independent implementation.

#include "qwfn_graph.h"
#include "qwfn_model.h"
#include "qwfn_home.h"
#include "qwfn_weights.h"

#include <cstdio>
#include <cstring>
#include <cmath>
#include <cstdlib>
#include <string>
#include <vector>

#include "ggml-alloc.h"
#include "ggml-backend.h"

using namespace qwfn;

// Deterministic and trivially reproducible in any language.
static float gen(uint32_t & s) {
    s = s * 1664525u + 1013904223u;
    return (float) ((s >> 8) & 0xFFFF) / 32768.0f - 1.0f;
}

int main(int argc, char ** argv) {
    if (argc < 2) { fprintf(stderr, "usage: qwfn-test-hc <shard.gguf> [n_tokens] [out.bin]\n"); return 1; }
    const int64_t T = argc > 2 ? atoll(argv[2]) : 3;
    const char * outpath = argc > 3 ? argv[3] : "hc_out.bin";

    model_index mi;
    std::string err;
    if (!mi.load(argv[1], err)) { fprintf(stderr, "error: %s\n", err.c_str()); return 1; }
    const hparams & hp = mi.hp();

    weights w;
    if (!w.init(&mi, /*prefer_gpu=*/true, qwfn::llama_backend_dir(), err)) {
        fprintf(stderr, "backend init: %s\n", err.c_str()); return 1;
    }
    printf("backend: %s (gpu=%d)\n", w.dev_name(), (int) w.on_gpu());

    for (const char * n : { "blk.0.hc_attn_norm.weight", "blk.0.hc_attn_down.weight",
                            "blk.0.hc_attn_up.weight",   "blk.0.hc_attn_inject.weight" }) {
        if (!w.declare(n)) { fprintf(stderr, "missing %s\n", n); return 1; }
    }
    if (!w.commit(err)) { fprintf(stderr, "commit: %s\n", err.c_str()); return 1; }
    printf("loaded %.2f MB of weights\n", w.bytes() / 1e6);

    const int64_t hc = hp.hc_count, n_embd = hp.n_embd;

    // Input residual [n_embd, hc, T], in its own buffer.
    ggml_init_params ip{}; ip.mem_size = ggml_tensor_overhead() * 8; ip.no_alloc = true;
    ggml_context * ictx = ggml_init(ip);
    ggml_tensor * x = ggml_new_tensor_3d(ictx, GGML_TYPE_F32, n_embd, hc, T);
    ggml_set_name(x, "x");
    ggml_backend_buffer_t ibuf = ggml_backend_alloc_ctx_tensors_from_buft(ictx, w.buft());
    if (!ibuf) { fprintf(stderr, "input alloc failed\n"); return 1; }

    std::vector<float> xd((size_t) n_embd * hc * T);
    uint32_t seed = 12345u;
    for (auto & v : xd) v = gen(seed);
    ggml_backend_tensor_set(x, xd.data(), 0, xd.size() * sizeof(float));

    // Build the graph.
    ggml_init_params gp{}; gp.mem_size = ggml_tensor_overhead() * 2048 + ggml_graph_overhead(); gp.no_alloc = true;
    ggml_context * gctx = ggml_init(gp);
    graph_builder gb(gctx, &hp, &w);

    ggml_cgraph * gf = ggml_new_graph(gctx);
    ggml_tensor * inject = nullptr;
    ggml_tensor * mixed  = gb.hc_mix(x, 0, /*ffn=*/false, &inject);
    ggml_tensor * comb   = gb.hc_combine(x, mixed, inject);
    ggml_set_name(mixed,  "mixed");
    ggml_set_name(inject, "inject");
    ggml_set_name(comb,   "combined");
    // Without this the allocator reuses inject's buffer for hc_combine's first
    // intermediate, and the readback sees 2*sigmoid(inject/hc) instead.
    ggml_set_output(mixed);
    ggml_set_output(inject);
    ggml_set_output(comb);
    ggml_build_forward_expand(gf, mixed);
    ggml_build_forward_expand(gf, inject);
    ggml_build_forward_expand(gf, comb);

    ggml_gallocr_t galloc = ggml_gallocr_new(w.buft());
    if (!ggml_gallocr_alloc_graph(galloc, gf)) { fprintf(stderr, "galloc failed\n"); return 1; }
    if (ggml_backend_graph_compute(w.backend(), gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "compute failed\n"); return 1;
    }

    auto dump = [&](ggml_tensor * t, const char * label, FILE * f) {
        const size_t n = ggml_nelements(t);
        std::vector<float> d(n);
        ggml_backend_tensor_get(t, d.data(), 0, n * sizeof(float));
        double sum = 0, absmax = 0;
        for (float v : d) { sum += v; if (fabs(v) > absmax) absmax = fabs(v); }
        printf("  %-9s [%5lld,%3lld,%3lld]  sum %14.6f  absmax %11.6f  first %.6f %.6f %.6f\n",
               label, (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2],
               sum, absmax, d[0], d.size() > 1 ? d[1] : 0.f, d.size() > 2 ? d[2] : 0.f);
        if (f) { int64_t ne = (int64_t) n; fwrite(&ne, 8, 1, f); fwrite(d.data(), 4, n, f); }
    };

    FILE * f = fopen(outpath, "wb");
    printf("hc_mix / hc_combine on layer 0, T=%lld:\n", (long long) T);
    dump(mixed,  "mixed",  f);
    dump(inject, "inject", f);
    dump(comb,   "combined", f);
    if (f) fclose(f);
    printf("wrote %s\n", outpath);

    ggml_gallocr_free(galloc);
    ggml_free(gctx);
    ggml_backend_buffer_free(ibuf);
    ggml_free(ictx);
    return 0;
}
