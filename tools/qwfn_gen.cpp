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
    std::string cold_path, save_replay;   // --save-replay FILE: the generated ids, one per line, for a later --replay-file
    bool want_ppl = false;   // with --replay-file: mean NLL of the replayed tokens (a quality number)
    bool pair_test = false, rollback_test = false;   // exercise the two-token decode step without the head
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
        if (a == "--state-host" && i + 1 < argc) {   // none | idx | kv | kv,idx
            std::string v = next();
            cfg.idx_host = v.find("idx") != std::string::npos;
            cfg.kv_host  = v.find("kv")  != std::string::npos;
            continue;
        }
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
        if (a == "--spec-margin" && i + 1 < argc) { cfg.spec_margin = (float) atof(next()); continue; }
        if (a == "--spec-gate-inflight" && i + 1 < argc) { cfg.spec_gate_inflight = (uint32_t) atoi(next()); continue; }
        if (a == "--spec-block") { cfg.spec_block = true; continue; }
        if (a == "--spec-block-layers" && i + 1 < argc) { cfg.spec_block = true; cfg.spec_block_layers = next(); continue; }
        if (a == "--mtp" && i + 1 < argc) { cfg.mtp_path = next(); cfg.rollback_snapshots = true; continue; }
        if (a == "--pair-test") { pair_test = true; continue; }          // decode the replay two tokens per step
        if (a == "--rollback-test") { rollback_test = true; cfg.rollback_snapshots = true; continue; }  // every token as the first of a pair with a wrong second, then roll back
        if (a == "--ram-frac" && i + 1 < argc) { cfg.ram_frac = atof(next()); continue; }
        // 0 forces the batched prefill path at every size. Reference runs want
        // this: token-by-token prefill is a different (equally valid) summation
        // order, and this model turns that into different tokens.
        if (a == "--prefill-decode-max" && i + 1 < argc) { cfg.prefill_decode_max = (uint32_t) atoi(next()); continue; }
        if (a == "--gate-drop" && i + 1 < argc) { cfg.gate_drop = (float) atof(next()); continue; }
        if (a == "--vram-reserve" && i + 1 < argc) { cfg.vram_reserve = (size_t)(atof(next()) * 1e6); continue; }
        if (a == "--no-prefill-overlap") { cfg.prefill_overlap = false; continue; }
        if (a == "--save-replay" && i + 1 < argc) { save_replay = next(); continue; }
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
    auto argmax = [&](const float * l) { int b = 0; for (int64_t v = 1; v < eng.n_vocab(); v++) if (l[v] > l[b]) b = (int) v; return b; };
    auto score  = [&](const float * l, int32_t tok) {   // -log softmax(l)[tok]
        float mx = l[0]; for (int64_t v = 1; v < eng.n_vocab(); v++) mx = std::max(mx, l[v]);
        double z = 0.0; for (int64_t v = 0; v < eng.n_vocab(); v++) z += std::exp((double) l[v] - mx);
        const double v = -((double) l[tok] - mx - std::log(z));
        nll_sum += v; nll_n++;
        static const bool verbose = getenv("QWFN_PPL_VERBOSE") != nullptr;
        if (verbose) fprintf(stderr, "[nll] #%d tok %d: %.6f\n", nll_n, tok, v);
    };
    const bool use_mtp = !cfg.mtp_path.empty() && !getenv("QWFN_MTP_NOVERIFY");   // NOVERIFY: the head loaded (its VRAM taken) but the plain loop
    if ((pair_test || rollback_test) && replay.empty()) { fprintf(stderr, "--pair-test / --rollback-test need --replay-file\n"); return 1; }
    uint64_t sp_steps = 0, sp_acc = 0, sp_single = 0;
    if (!use_mtp && !pair_test && !rollback_test) {
        for (int i = 0; i < n_gen; i++) {
            int best = 0;
            if (replay.empty()) best = argmax(lg);
            else { best = replay[i]; if (want_ppl) score(lg, best); }
            printf(" %d", best);
            fflush(stdout);
            hist.push_back(best);
            lg = eng.eval(hist.data(), (int32_t) hist.size(), 1, err);
            if (!lg) { fprintf(stderr, "\ndecode failed: %s\n", err.c_str()); return 1; }
        }
    } else {
        // Speculative loop. `next` is the token to feed; the head's draft for the
        // token after it (or, in the tests, the replay's own next token / a wrong
        // one) rides along as the second of a pair. Position 0's logits verify the
        // draft; position 1's are the next token's if it is accepted. The replay
        // is the ground truth for acceptance when replaying; argmax when not.
        // NLL covers replay[0..n_gen) as in the plain loop.
        int i = 0;
        int32_t next = replay.empty() ? argmax(lg) : replay[0];
        if (want_ppl && !replay.empty()) score(lg, replay[0]);
        if (use_mtp && !eng.mtp_step(&next, 1, err)) { fprintf(stderr, "\nmtp: %s\n", err.c_str()); return 1; }
        int produced = 0;
        auto next_after = [&](const float * l, int k) -> int32_t {   // the token at replay index k, or argmax
            if (replay.empty()) return argmax(l);
            return k < (int) replay.size() ? replay[k] : -1;
        };
        while (produced < n_gen) {
            const int32_t draft = use_mtp ? eng.mtp_draft_id() : -1;
            const bool room = produced + 1 < n_gen && (replay.empty() || i + 1 < (int) replay.size());
            const bool pair = room && (pair_test || rollback_test || draft >= 0);
            if (pair) {
                const int32_t second = pair_test ? replay[i + 1]
                                     : rollback_test ? (int32_t) ((replay[i + 1] + (getenv("QWFN_RB_WRONG") ? atoi(getenv("QWFN_RB_WRONG")) : 1)) % eng.n_vocab())
                                     : draft;
                hist.push_back(next); hist.push_back(second);
                if (!eng.eval_decode(hist.data(), (int32_t) hist.size(), 2, err)) { fprintf(stderr, "\ndecode failed: %s\n", err.c_str()); return 1; }
                const float * l0 = eng.logits_pos(0), * l1 = eng.logits_pos(1);
                printf(" %d", next); fflush(stdout); produced++;
                const int32_t y = next_after(l0, i + 1);
                if (want_ppl && !replay.empty() && i + 1 < n_gen) score(l0, replay[i + 1]);
                sp_steps++;
                const bool accept = pair_test ? true : rollback_test ? false : (y == second);
                if (accept) {
                    printf(" %d", second); fflush(stdout); produced++;
                    const int32_t next2 = next_after(l1, i + 2);
                    if (want_ppl && !replay.empty() && i + 2 < n_gen) score(l1, replay[i + 2]);
                    i += 2; sp_acc++;
                    if (next2 < 0) break;
                    if (use_mtp) { const int32_t toks[2] = { second, next2 }; if (!eng.mtp_step(toks, 2, err)) { fprintf(stderr, "\nmtp: %s\n", err.c_str()); return 1; } }
                    next = next2;
                } else {
                    if (!eng.rollback(err)) { fprintf(stderr, "\nrollback: %s\n", err.c_str()); return 1; }
                    hist.pop_back();
                    i += 1;
                    if (y < 0) break;
                    if (use_mtp && !eng.mtp_step(&y, 1, err)) { fprintf(stderr, "\nmtp: %s\n", err.c_str()); return 1; }
                    next = y;
                }
            } else {
                hist.push_back(next);
                lg = eng.eval_decode(hist.data(), (int32_t) hist.size(), 1, err);
                if (!lg) { fprintf(stderr, "\ndecode failed: %s\n", err.c_str()); return 1; }
                printf(" %d", next); fflush(stdout); produced++; sp_single++;
                const int32_t y = next_after(lg, i + 1);
                if (want_ppl && !replay.empty() && i + 1 < n_gen) score(lg, replay[i + 1]);
                i += 1;
                if (y < 0) break;
                if (use_mtp && !eng.mtp_step(&y, 1, err)) { fprintf(stderr, "\nmtp: %s\n", err.c_str()); return 1; }
                next = y;
            }
        }
        n_gen = produced;
    }
    const double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    printf("\n\ndecode: %d tokens in %.2f s  (%.2f tok/s), n_past=%d\n", n_gen, dt, n_gen / dt, eng.n_past());
    if (!save_replay.empty()) {
        FILE * f = fopen(save_replay.c_str(), "wb");
        if (f) { for (size_t i = hist.size() - n_gen; i < hist.size(); i++) fprintf(f, "%d\n", hist[i]); fclose(f);
                 printf("saved %d generated ids to %s\n", n_gen, save_replay.c_str()); }
        else fprintf(stderr, "cannot write %s\n", save_replay.c_str());
    }
    if (nll_n) printf("replay NLL: %.4f per token (ppl %.2f) over %d tokens\n", nll_sum / nll_n, std::exp(nll_sum / nll_n), nll_n);
    if (eng.n_exp_skipped) printf("skipped experts: %llu (misses computed without)\n", (unsigned long long) eng.n_exp_skipped);
    if (getenv("QWFN_VRAM_AUDIT")) {
        size_t ab = 0, mb = 0; int ag = 0, mg = 0; eng.graph_buffer_bytes(ab, ag, mb, mg);
        printf("VRAM audit: cached decode graphs hold %.0f MB of activations in %d layer graphs (%.1f MB each), MoE graphs %.0f MB in %d\n",
               ab / 1e6, ag, ag ? ab / 1e6 / ag : 0.0, mb / 1e6, mg);
    }
    if (eng.n_exp_dropped) printf("dropped experts: %llu (gate below --gate-drop; %.1f%% of routed)\n", (unsigned long long) eng.n_exp_dropped,
                                  100.0 * eng.n_exp_dropped / std::max<uint64_t>(1, eng.n_exp_dropped + eng.n_exp_gpu + eng.n_exp_cpu + eng.n_exp_skipped));

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
    if (sp_steps || sp_single)
        printf("speculative: %llu pair steps, %llu accepted (%.1f%%), %llu single steps, %llu rollbacks (%.3f s), head %.2f s\n",
               (unsigned long long) sp_steps, (unsigned long long) sp_acc, sp_steps ? 100.0 * sp_acc / sp_steps : 0.0,
               (unsigned long long) sp_single, (unsigned long long) eng.n_rollback, eng.t_rollback, eng.t_mtp);
    if (eng.mtp_n || eng.mtp_prompt_n)
        printf("mtp draft: %llu decode drafts scored, %llu accepted (%.1f%%), top-3 %.1f%% | prompt: %llu scored, %.1f%% accepted | %.2f s in the head\n",
               (unsigned long long) eng.mtp_n, (unsigned long long) eng.mtp_acc,
               eng.mtp_n ? 100.0 * eng.mtp_acc / eng.mtp_n : 0.0, eng.mtp_n ? 100.0 * eng.mtp_top3 / eng.mtp_n : 0.0,
               (unsigned long long) eng.mtp_prompt_n, eng.mtp_prompt_n ? 100.0 * eng.mtp_prompt_acc / eng.mtp_prompt_n : 0.0, eng.t_mtp);
    if (eng.pf_gated)
        printf("prefetch gate: %llu predicted candidates not read (margin < %.2f%s)\n",
               (unsigned long long) eng.pf_gated, cfg.spec_margin,
               cfg.spec_gate_inflight ? ", only with reads in flight" : "");
    {   // precision by predicted rank, then by confidence margin, then recall by layer
        printf("prediction by rank, precision %%:");
        for (int k = 0; k < (int) QWFN_SPEC_MAX; k++)
            if (eng.rank_total[k]) printf(" r%d %.0f", k + 1, 100.0 * eng.rank_hits[k] / eng.rank_total[k]);
        printf("\n");
        unsigned long long mt = 0; for (int b = 0; b < engine::SPEC_MARGIN_BUCKETS; b++) mt += eng.margin_total[b];
        printf("prediction by margin [edge+) P(correct)%% / share%%:");
        for (int b = 0; b < engine::SPEC_MARGIN_BUCKETS; b++)
            if (eng.margin_total[b])
                printf(" [%.2g) %.0f/%.1f", engine::margin_edge(b), 100.0 * eng.margin_hits[b] / eng.margin_total[b],
                       mt ? 100.0 * eng.margin_total[b] / mt : 0.0);
        printf("\n");
        std::vector<std::pair<double, int>> acc;
        for (size_t l = 0; l < eng.pred_total_layer.size(); l++)
            if (eng.pred_total_layer[l]) acc.push_back({ 100.0 * eng.pred_hits_layer[l] / eng.pred_total_layer[l], (int) l });
        std::sort(acc.begin(), acc.end());
        printf("prediction by layer, worst 10:");
        for (size_t i = 0; i < acc.size() && i < 10; i++) printf(" L%d %.0f", acc[i].second, acc[i].first);
        if (acc.size() > 10) printf("  | best: L%d %.0f", acc.back().second, acc.back().first);
        printf("\n");
    }
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
