// qwfn-gen -- prefill a prompt, then decode greedily, tracking position.
//
// Works in token ids so it can be compared against llama.cpp directly without
// needing a tokenizer.

#include "qwfn_engine.h"
#include "qwfn_model.h"

#include <algorithm>
#include <cmath>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace qwfn;

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr,
            "usage: qwfn-gen <shard.gguf> [--prompt id,id,...] [--gen N] [--ctx N]\n"
            "                [--ram GB] [--vram GB] [--batch N] [--threads N] [--cpu] [--no-qsa]\n");
        return 1;
    }
    std::vector<int32_t> prompt, replay;
    std::string cold_path;
    bool want_ppl = false;   // with --replay-file: mean NLL of the replayed tokens (a quality number)
    int n_gen = 16;
    engine_config cfg;
    cfg.n_ctx = 4096; cfg.n_batch = 128; cfg.ram_bytes = 8e9; cfg.vram_bytes = 0;

    for (int i = 2; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() { return argv[++i]; };
        if (a == "--gen"     && i + 1 < argc) { n_gen = atoi(next()); continue; }
        if (a == "--ctx"     && i + 1 < argc) { cfg.n_ctx = (uint32_t) atoi(next()); continue; }
        if (a == "--batch"   && i + 1 < argc) { cfg.n_batch = (uint32_t) atoi(next()); continue; }
        if (a == "--ram"     && i + 1 < argc) { cfg.ram_bytes = (size_t)(atof(next()) * 1e9); continue; }
        if (a == "--vram"    && i + 1 < argc) { cfg.vram_bytes = (size_t)(atof(next()) * 1e9); continue; }
        if (a == "--threads" && i + 1 < argc) { cfg.n_threads = atoi(next()); continue; }
        if (a == "--cpu")    { cfg.use_gpu = false; continue; }
        if (a == "--io-threads" && i + 1 < argc) { cfg.io_threads = true; cfg.io_workers = (unsigned) atoi(next()); continue; }
        if (a == "--io-uring") { cfg.io_threads = false; continue; }
        if (a == "--kv" && i + 1 < argc) { std::string v = next();
            cfg.type_k = cfg.type_v = (v == "q8_0") ? GGML_TYPE_Q8_0 :
                                      (v == "q4_0") ? GGML_TYPE_Q4_0 : GGML_TYPE_F16; continue; }
        if (a == "--no-reuse") { cfg.reuse_graphs = false; continue; }
        if (a == "--no-speculate") { cfg.speculate = false; continue; }
        if (a == "--skip-miss") { cfg.skip_miss = true; continue; }
        if (a == "--ppl") { want_ppl = true; continue; }
        if (a == "--reserve" && i + 1 < argc) { cfg.vram_reserve = (size_t) atof(next()) * (1ull << 20); continue; }
        if (a == "--predictor" && i + 1 < argc) { cfg.predictor_path = next(); continue; }
        if (a == "--spec-depth" && i + 1 < argc) { cfg.speculate_depth = (uint32_t) atoi(next());
            if (cfg.speculate_depth == 0) cfg.speculate = false; continue; }
        if (a == "--spec-ahead" && i + 1 < argc) { cfg.speculate_ahead = (uint32_t) atoi(next()); continue; }
        if (a == "--spec-depth2" && i + 1 < argc) { cfg.speculate_depth2 = (uint32_t) atoi(next()); continue; }
        if (a == "--ram-frac" && i + 1 < argc) { cfg.ram_frac = atof(next()); continue; }
        // 0 forces the batched prefill path at every size. Reference runs want
        // this: token-by-token prefill is a different (equally valid) summation
        // order, and this model turns that into different tokens.
        if (a == "--prefill-decode-max" && i + 1 < argc) { cfg.prefill_decode_max = (uint32_t) atoi(next()); continue; }
        if (a == "--vram-reserve" && i + 1 < argc) { cfg.vram_reserve = (size_t)(atof(next()) * 1e6); continue; }
        if (a == "--no-prefill-overlap") { cfg.prefill_overlap = false; continue; }
        if (a == "--replay-file" && i + 1 < argc) {
            // Feed these ids as the "generated" tokens instead of sampling, so
            // every configuration sees identical routing. GPU decode is
            // nondeterministic, and comparing different generated texts was
            // measured to produce 1.5 tok/s of phantom variance.
            FILE * f = fopen(next(), "rb");
            if (!f) { fprintf(stderr, "error: cannot open replay file\n"); return 1; }
            int v; while (fscanf(f, "%d%*[ ,\n\t\r]", &v) == 1) replay.push_back(v);
            fclose(f);
            continue;
        }
        if (a == "--promote" && i + 1 < argc) { cfg.promote_per_layer = (uint32_t) atoi(next()); continue; }
        if (a == "--evict" && i + 1 < argc) { std::string v = next();
            cfg.evict_policy = v == "lfu" ? 1 : v == "hybrid" ? 2 : 0; continue; }
        if (a == "--no-qsa") { cfg.use_qsa = false; continue; }
        if (a == "--prefill-cpu") { cfg.prefill_on_gpu = false; continue; }
        if (a == "--prefill-gpu") { cfg.prefill_on_gpu = true; continue; }  // FAST BUT BROKEN >T~200
        if (a == "--ubatch-kv" && i + 1 < argc) { cfg.ubatch_kv_product = (uint64_t)(atof(next()) * 1e6); continue; }
        if (a == "--indexer-top-k" && i + 1 < argc) { cfg.indexer_top_k = (uint32_t) atoi(next()); continue; }
        if (a == "--cold" && i + 1 < argc) { cold_path = next(); cfg.use_cold_tier = true; continue; }
        if (a == "--prompt-file" && i + 1 < argc) {
            // Long contexts blow past ARG_MAX on the command line.
            FILE * f = fopen(next(), "rb");
            if (!f) { fprintf(stderr, "error: cannot open prompt file\n"); return 1; }
            int v; while (fscanf(f, "%d%*[ ,\n\t\r]", &v) == 1) prompt.push_back(v);
            fclose(f);
            continue;
        }
        if (a == "--prompt"  && i + 1 < argc) {
            std::string t = next(); size_t p = 0;
            while (p < t.size()) {
                size_t c = t.find(',', p); if (c == std::string::npos) c = t.size();
                prompt.push_back(atoi(t.substr(p, c - p).c_str())); p = c + 1;
            }
            continue;
        }
    }
    if (prompt.empty()) prompt = { 9707, 11, 1879, 0 };

    model_index mi;
    std::string err;
    if (!mi.load(argv[1], err)) { fprintf(stderr, "error: %s\n", err.c_str()); return 1; }

    model_index cold;
    model_index * coldp = nullptr;
    if (!cold_path.empty()) {
        if (cold.load(cold_path, err)) { coldp = &cold; printf("cold tier: %s\n", cold_path.c_str()); }
        else fprintf(stderr, "cold tier unavailable: %s\n", err.c_str());
    }

    engine eng;
    if (!eng.init(&mi, coldp, cfg, std::string(getenv("HOME")) + "/.unsloth/llama.cpp/build/bin", err)) {
        fprintf(stderr, "engine init: %s\n", err.c_str()); return 1;
    }
    printf("%s\n\n", eng.memory_summary().c_str());

    std::vector<int32_t> hist = prompt;
    const float * lg = nullptr;

    // ---- prefill in ubatches ----------------------------------------------
    {
        const auto t0 = std::chrono::steady_clock::now();
        int32_t done = 0;
        while (done < (int32_t) prompt.size()) {
            const int32_t take = std::min<int32_t>(cfg.n_batch, (int32_t) prompt.size() - done);
            lg = eng.eval(hist.data(), done + take, take, err);
            if (!lg) { fprintf(stderr, "prefill failed: %s\n", err.c_str()); return 1; }
            done += take;
        }
        const double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        printf("prefill: %zu tokens in %.2f s  (%.1f tok/s)\n", prompt.size(), dt, prompt.size() / dt);
    }

    // ---- greedy decode ------------------------------------------------------
    printf(replay.empty() ? "generated:" : "replaying:");
    if (!replay.empty()) n_gen = std::min<int>(n_gen, (int) replay.size());
    const auto t0 = std::chrono::steady_clock::now();
    double nll_sum = 0.0; int nll_n = 0;
    for (int i = 0; i < n_gen; i++) {
        int best = 0;
        if (replay.empty()) {
            for (int64_t v = 1; v < eng.n_vocab(); v++) if (lg[v] > lg[best]) best = (int) v;
        } else {
            best = replay[i];
            if (want_ppl) {   // -log softmax(lg)[best]
                float mx = lg[0]; for (int64_t v = 1; v < eng.n_vocab(); v++) mx = std::max(mx, lg[v]);
                double z = 0.0; for (int64_t v = 0; v < eng.n_vocab(); v++) z += std::exp((double) lg[v] - mx);
                nll_sum += -((double) lg[best] - mx - std::log(z)); nll_n++;
            }
        }
        printf(" %d", best);
        fflush(stdout);
        hist.push_back(best);
        lg = eng.eval(hist.data(), (int32_t) hist.size(), 1, err);
        if (!lg) { fprintf(stderr, "\ndecode failed: %s\n", err.c_str()); return 1; }
    }
    const double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    printf("\n\ndecode: %d tokens in %.2f s  (%.2f tok/s), n_past=%d\n", n_gen, dt, n_gen / dt, eng.n_past());
    if (nll_n) printf("replay NLL: %.4f per token (ppl %.2f) over %d tokens\n", nll_sum / nll_n, std::exp(nll_sum / nll_n), nll_n);
    if (eng.n_exp_skipped) printf("skipped experts: %llu (misses computed without)\n", (unsigned long long) eng.n_exp_skipped);

    const auto & s = eng.cache_stats();
    printf("expert cache: %.1f%% hit, %.1f%% from VRAM, %.2f GB from disk\n",
           100.0 * s.hit_rate(), 100.0 * s.gpu_rate(), s.bytes_from_disk / 1e9);
    printf("decode split: graphA(GPU) %.2f s | MoE gpu %.2f s (%llu experts, sync-wait %.2f s) | MoE cpu %.2f s (%llu experts) | io %.2f s\n",
           eng.t_layerA, eng.t_moe_gpu, (unsigned long long) eng.n_exp_gpu, eng.t_moe_gpu_sync,
           eng.t_moe_cpu, (unsigned long long) eng.n_exp_cpu, eng.t_io);
    if (eng.n_exp_cpu && eng.n_exp_gpu)
        printf("              per expert: gpu %.0f us, cpu %.0f us  (%.1fx)\n",
               eng.t_moe_gpu / eng.n_exp_gpu * 1e6, eng.t_moe_cpu / eng.n_exp_cpu * 1e6,
               (eng.t_moe_cpu / eng.n_exp_cpu) / (eng.t_moe_gpu / eng.n_exp_gpu));
    if (eng.n_layerA_rec && eng.n_layerA_attn)
        printf("              graphA per decode layer: recurrent %.0f us (x%llu), attention %.0f us (x%llu)"
               " [attention: build+alloc %.0f us, gpu %.0f us; per-token inputs %.0f us]\n",
               eng.t_layerA_rec / eng.n_layerA_rec * 1e6, (unsigned long long) eng.n_layerA_rec,
               eng.t_layerA_attn / eng.n_layerA_attn * 1e6, (unsigned long long) eng.n_layerA_attn,
               eng.t_attn_build / eng.n_layerA_attn * 1e6, eng.t_attn_compute / eng.n_layerA_attn * 1e6,
               eng.n_decode ? eng.t_inputs / eng.n_decode * 1e6 : 0.0);
    printf("prefetch: %.1f%% of the next layer's experts predicted correctly",
           eng.pred_total ? 100.0 * eng.pred_hits / eng.pred_total : 0.0);
    if (eng.pred2_total)
        printf(" (two ahead: %.1f%%)", 100.0 * eng.pred2_hits / eng.pred2_total);
    printf(" | %llu issued, %llu used (%.1f%%), %llu wasted\n",
           (unsigned long long) s.pf_issued, (unsigned long long) s.pf_used,
           s.pf_issued ? 100.0 * s.pf_used / s.pf_issued : 0.0,
           (unsigned long long) s.pf_wasted);
    printf("io breakdown: submit %.2f s, promote %.2f s, wait %.2f s | %llu bursts, %llu reads, "
           "%.1f reads/burst, %.0f KiB/read\n",
           s.t_submit, s.t_promote, s.t_wait,
           (unsigned long long) s.n_bursts, (unsigned long long) s.n_reads,
           s.n_bursts ? (double) s.n_reads / s.n_bursts : 0.0,
           s.n_reads ? s.bytes_from_disk / (double) s.n_reads / 1024.0 : 0.0);
    printf("             wait-only bandwidth: %.2f GB/s   (device peak measured 7.0 GB/s)\n",
           s.t_wait > 0 ? s.bytes_from_disk / s.t_wait / 1e9 : 0.0);
    {
        const auto & io = eng.cache_io();
        printf("             io_engine: backend=%s, O_DIRECT=%d, %llu reads, %.2f GB actually read, "
               "prep %.2f s, io_uring_submit %.2f s\n",
               io.which() == io_engine::backend::threads ? "threads" : "uring",
               (int) io.direct_io(), (unsigned long long) io.stat_reads,
               io.stat_bytes / 1e9, io.stat_t_prep, io.stat_t_submit_syscall);
    }
    printf("engine time: prefill %.2f s / %lld tok, decode %.2f s / %lld tok, expert io %.2f s\n",
           eng.t_prefill, (long long) eng.n_prefill, eng.t_decode, (long long) eng.n_decode, eng.t_io);
    if (eng.n_prefill > 0 && (eng.t_pf_graphA > 0 || eng.t_pf_moe > 0))
        printf("prefill split: dense graphs %.2f s | expert reads (blocking) %.2f s | MoE %.2f s | warm-up %.2f s  (of %.2f s)\n",
               eng.t_pf_graphA, eng.t_pf_read, eng.t_pf_moe, eng.t_warm, eng.t_prefill);
    if (s.warm_admitted || s.warm_promoted)
        printf("cache warm-up from prefill: %llu blocks into RAM, %llu on to VRAM, %.2f s\n",
               (unsigned long long) s.warm_admitted, (unsigned long long) s.warm_promoted, eng.t_warm);
    return 0;
}
