// qwfn-refdump -- run llama.cpp's own qwen4exp graph and print a checksum for
#include <algorithm>
// every named intermediate, so our graph can be bisected against it.
//
// llama.cpp tags its intermediates with cb(): "hc_mixed-3", "ffn_moe_out-0",
// "result_norm", and so on. Matching those names to ours pinpoints the first
// layer where the two diverge.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

static std::vector<std::string> g_filters;
static bool wanted(const char * name) {
    if (g_filters.empty()) return true;
    for (const auto & f : g_filters)
        if (strstr(name, f.c_str())) return true;
    return false;
}

static bool eval_cb(ggml_tensor * t, bool ask, void * /*ud*/) {
    if (ask) return wanted(t->name);
    if (!wanted(t->name)) return true;

    const size_t n = ggml_nelements(t);
    std::vector<float> buf;
    double sum = 0, absmax = 0;
    float f0 = 0, f1 = 0, f2 = 0;

    if (t->type == GGML_TYPE_F32) {
        buf.resize(n);
        ggml_backend_tensor_get(t, buf.data(), 0, n * sizeof(float));
        for (size_t i = 0; i < n; i++) { sum += buf[i]; if (fabs(buf[i]) > absmax) absmax = fabs(buf[i]); }
        f0 = n > 0 ? buf[0] : 0; f1 = n > 1 ? buf[1] : 0; f2 = n > 2 ? buf[2] : 0;
    } else if (t->type == GGML_TYPE_I32) {
        std::vector<int32_t> ib(n);
        ggml_backend_tensor_get(t, ib.data(), 0, n * sizeof(int32_t));
        for (size_t i = 0; i < n; i++) sum += ib[i];
        f0 = n > 0 ? (float) ib[0] : 0; f1 = n > 1 ? (float) ib[1] : 0; f2 = n > 2 ? (float) ib[2] : 0;
    } else {
        printf("  %-28s [%5lld,%5lld,%4lld] %-8s (skipped)\n", t->name,
               (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2], ggml_type_name(t->type));
        return true;
    }

    printf("  %-28s [%5lld,%5lld,%4lld] sum %18.9f absmax %14.9f first %.6f %.6f %.6f\n",
           t->name, (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2],
           sum, absmax, f0, f1, f2);
    return true;
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: qwfn-refdump <model.gguf> [tok,tok,..] [--filter substr,substr]\n");
        return 1;
    }
    std::vector<llama_token> toks;
    int64_t top_k_override = 0;
    int32_t ngl = 0;
    int n_gen = 0;
    for (int i = 2; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--gpu") { ngl = 999; continue; }
        if (a == "--gen" && i + 1 < argc) { n_gen = atoi(argv[++i]); continue; }
        if (a == "--top-k" && i + 1 < argc) { top_k_override = atoll(argv[++i]); continue; }
        if (a == "--filter" && i + 1 < argc) {
            std::string f = argv[++i];
            size_t p = 0;
            while (p < f.size()) {
                size_t c = f.find(',', p);
                if (c == std::string::npos) c = f.size();
                g_filters.push_back(f.substr(p, c - p));
                p = c + 1;
            }
            continue;
        }
        size_t p = 0;
        while (p < a.size()) {
            size_t c = a.find(',', p);
            if (c == std::string::npos) c = a.size();
            toks.push_back(atoi(a.substr(p, c - p).c_str()));
            p = c + 1;
        }
    }
    if (toks.empty()) toks.push_back(9707);

    // The backend .so files live next to libllama; load them before any model.
    {
        const char * home = getenv("HOME");
        std::string dir = std::string(home ? home : ".") + "/.unsloth/llama.cpp/build/bin";
        ggml_backend_load_all_from_path(dir.c_str());
    }
    llama_backend_init();

    llama_model_kv_override ovr[2] = {};
    llama_model_params mp = llama_model_default_params();
    if (top_k_override > 0) {
        ovr[0].tag = LLAMA_KV_OVERRIDE_TYPE_INT;
        snprintf(ovr[0].key, sizeof(ovr[0].key), "qwen4exp.attention.indexer.top_k");
        ovr[0].val_i64 = top_k_override;
        ovr[1].key[0] = 0;              // terminator
        mp.kv_overrides = ovr;
        printf("indexer top_k overridden to %lld\n", (long long) top_k_override);
    }
    mp.n_gpu_layers = ngl;            // 0 = CPU, matching our validation path
    mp.load_mode    = LLAMA_LOAD_MODE_MMAP;
    mp.lazy_mode    = LLAMA_LAZY_MODE_AUTO;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) { fprintf(stderr, "failed to load model\n"); return 1; }

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx            = 2048;
    cp.n_batch          = 512;
    cp.n_ubatch         = 512;
    cp.cb_eval          = eval_cb;
    cp.cb_eval_user_data = nullptr;
    cp.type_k           = GGML_TYPE_F16;
    cp.type_v           = GGML_TYPE_F16;
    cp.flash_attn_type  = LLAMA_FLASH_ATTN_TYPE_ENABLED;   // match our graph's kernel
    cp.n_threads        = 8;
    cp.n_threads_batch  = 8;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "failed to create context\n"); return 1; }

    printf("=== llama.cpp intermediates for %zu token(s) ===\n", toks.size());
    llama_batch batch = llama_batch_get_one(toks.data(), (int32_t) toks.size());
    if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "decode failed\n"); return 1; }

    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const float * lg = llama_get_logits_ith(ctx, (int32_t) toks.size() - 1);
    std::vector<int> ord(n_vocab);
    for (int i = 0; i < n_vocab; i++) ord[i] = i;
    std::partial_sort(ord.begin(), ord.begin() + 10, ord.end(),
                      [&](int a, int b) { return lg[a] > lg[b]; });
    printf("\ntop-10 logits:\n");
    for (int i = 0; i < 10; i++) printf("  %2d. id %6d  logit %9.4f\n", i + 1, ord[i], lg[ord[i]]);

    if (n_gen > 0) {
        // Greedy continuation, so the whole sequence can be compared token for token.
        printf("\ngenerated:");
        std::vector<llama_token> cur = toks;
        for (int i = 0; i < n_gen; i++) {
            const float * l = llama_get_logits_ith(ctx, -1);
            int best = 0;
            for (int v = 1; v < n_vocab; v++) if (l[v] > l[best]) best = v;
            printf(" %d", best);
            fflush(stdout);
            llama_token t = best;
            llama_batch b = llama_batch_get_one(&t, 1);
            if (llama_decode(ctx, b) != 0) { fprintf(stderr, "\ndecode failed\n"); break; }
        }
        printf("\n");
    }

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
