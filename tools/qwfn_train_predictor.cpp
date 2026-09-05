// Train the learned routing predictor from an engine dump (QWFN_ROUTE_DUMP).
//
// Same objective as scripts/train_predictor.py -- a linear head n_embd ->
// n_expert per layer, initialised from that layer's router matrix, fine-tuned
// with a soft cross-entropy toward the true gates plus an L2 pull to the init --
// but the matmuls run on the GPU through ggml (numpy here is linked against the
// reference BLAS and takes eleven minutes per layer; this takes seconds).
// Adam runs on the host: the gradient is 5 MB per step, the update trivial.
//
//   qwfn-train-predictor DUMP_DIR OUT.bin [--epochs 8] [--lr 3e-4] [--l2 1e-4]
//       [--val 0.1] [--batch 4096] [--first 1] [--last 47] [--max-samples N]
//       [--n-embd 2560] [--n-expert 512] [--n-used 10] [--n-layer 48]
//
// OUT.bin: "QWPR", u32 version=1, n_layer, n_embd, n_expert, first_layer, then
// per layer first..n_layer-1: F16 W[n_expert][n_embd], F32 b[n_expert].
// Printed per layer: recall@n_used on the held-out split for the router init
// and for the trained head -- the share of the true top-k found in the
// predicted top-k, which is what the engine's "predicted correctly" reports.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <string>
#include <vector>

struct args_t {
    std::string dump, out;
    int epochs = 8; float lr = 3e-4f, l2 = 1e-4f, val = 0.1f;
    int batch = 4096, first = 1, last = 47, max_samples = 0;
    int n_embd = 2560, n_expert = 512, n_used = 10, n_layer = 48;
    int pred_k = 0;   // report recall of the true top-n_used within the predicted top-pred_k (0 = n_used)
    bool val_tail = false;   // validate on the LAST val fraction of records (whole generations), not a random split
};

static bool read_file(const std::string & p, std::vector<uint8_t> & out) {
    FILE * f = fopen(p.c_str(), "rb"); if (!f) return false;
    fseek(f, 0, SEEK_END); const long n = ftell(f); fseek(f, 0, SEEK_SET);
    out.resize((size_t) n);
    const bool ok = n == 0 || fread(out.data(), 1, (size_t) n, f) == (size_t) n;
    fclose(f); return ok;
}

// recall of the k true ids in the top-pk of each column of S [n_expert, n]
static double recall_at_k(const float * S, int n_expert, int n, const uint16_t * ids, int k, int pk) {
    std::vector<int> idx(n_expert);
    size_t hits = 0;
    for (int t = 0; t < n; t++) {
        const float * s = S + (size_t) t * n_expert;
        std::iota(idx.begin(), idx.end(), 0);
        std::partial_sort(idx.begin(), idx.begin() + pk, idx.end(), [&](int a, int b) { return s[a] > s[b]; });
        for (int j = 0; j < k; j++)
            for (int i = 0; i < pk; i++) if (idx[i] == ids[(size_t) t * k + j]) { hits++; break; }
    }
    return (double) hits / ((double) n * k);
}

int main(int argc, char ** argv) {
    args_t a;
    for (int i = 1; i < argc; i++) {
        std::string s = argv[i];
        auto next = [&]() { return argv[++i]; };
        if (s == "--epochs" && i + 1 < argc) { a.epochs = atoi(next()); continue; }
        if (s == "--lr" && i + 1 < argc) { a.lr = (float) atof(next()); continue; }
        if (s == "--l2" && i + 1 < argc) { a.l2 = (float) atof(next()); continue; }
        if (s == "--val" && i + 1 < argc) { a.val = (float) atof(next()); continue; }
        if (s == "--batch" && i + 1 < argc) { a.batch = atoi(next()); continue; }
        if (s == "--first" && i + 1 < argc) { a.first = atoi(next()); continue; }
        if (s == "--last" && i + 1 < argc) { a.last = atoi(next()); continue; }
        if (s == "--max-samples" && i + 1 < argc) { a.max_samples = atoi(next()); continue; }
        if (s == "--n-embd" && i + 1 < argc) { a.n_embd = atoi(next()); continue; }
        if (s == "--n-expert" && i + 1 < argc) { a.n_expert = atoi(next()); continue; }
        if (s == "--n-used" && i + 1 < argc) { a.n_used = atoi(next()); continue; }
        if (s == "--n-layer" && i + 1 < argc) { a.n_layer = atoi(next()); continue; }
        if (s == "--pred-k" && i + 1 < argc) { a.pred_k = atoi(next()); continue; }
        if (s == "--val-tail") { a.val_tail = true; continue; }
        if (a.dump.empty()) a.dump = s; else if (a.out.empty()) a.out = s;
        else { fprintf(stderr, "unknown argument %s\n", argv[i]); return 1; }
    }
    if (a.dump.empty() || a.out.empty()) { fprintf(stderr, "usage: qwfn-train-predictor DUMP_DIR OUT.bin [options]\n"); return 1; }
    const int D = a.n_embd, E = a.n_expert, K = a.n_used, B = a.batch;
    const int PK = a.pred_k > 0 ? a.pred_k : K;

    // This llama.cpp build ships its backends as shared objects next to
    // libggml.so; load them from there (QWFN_GGML_BACKENDS overrides).
    ggml_backend_load_all();
    if (!ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU)) {
        const char * dir = getenv("QWFN_GGML_BACKENDS");
#ifdef LLAMA_CPP_BUILD_DIR
        if (!dir) dir = LLAMA_CPP_BUILD_DIR;
#endif
        if (dir) ggml_backend_load_all_from_path(dir);
    }
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (!dev) dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (!dev) { fprintf(stderr, "[train] no ggml backend found; set QWFN_GGML_BACKENDS to the directory holding libggml-cuda.so\n"); return 1; }
    ggml_backend_t be = ggml_backend_dev_init(dev, nullptr);
    ggml_backend_buffer_type_t buft = ggml_backend_dev_buffer_type(dev);
    fprintf(stderr, "[train] backend %s\n", ggml_backend_name(be));

    // Every layer's head, to be written at the end (router kept where no data).
    std::vector<std::vector<float>> outW(a.n_layer), outb(a.n_layer);
    double sum_base = 0, sum_best = 0; int n_done = 0;

    for (int il = a.first; il < a.n_layer; il++) {
        std::vector<uint8_t> rb;
        if (!read_file(a.dump + "/router_L" + std::to_string(il) + ".bin", rb) || rb.size() != (size_t) E * D * 4) {
            fprintf(stderr, "layer %2d: no router matrix, skipped\n", il); continue;
        }
        std::vector<float> W0((size_t) E * D); memcpy(W0.data(), rb.data(), rb.size());
        std::vector<float> W = W0, b(E, 0.0f);
        outW[il] = W0; outb[il] = b;
        if (il > a.last) { printf("layer %2d: router kept\n", il); continue; }

        std::vector<uint8_t> db;
        const size_t rec = (size_t) D * 2 + (size_t) K * 2 + (size_t) K * 2;
        if (!read_file(a.dump + "/L" + std::to_string(il) + ".bin", db) || db.size() < rec) {
            printf("layer %2d: no data, router kept\n", il); fflush(stdout); continue;
        }
        const auto t0 = std::chrono::steady_clock::now();
        size_t N = db.size() / rec;
        std::vector<size_t> order(N); std::iota(order.begin(), order.end(), 0);
        std::mt19937_64 rng(1);
        if (a.max_samples > 0 && N > (size_t) a.max_samples) N = a.max_samples;
        const size_t Nv = std::max<size_t>(1, (size_t) (N * a.val)), Nt = N - Nv;
        if (a.val_tail) {
            // Records are in generation order: the tail is whole generations the
            // head never saw. Validation goes first in `order`, then the shuffled rest.
            std::vector<size_t> tail(order.begin() + (N - Nv), order.begin() + N);
            std::vector<size_t> head(order.begin(), order.begin() + (N - Nv));
            std::shuffle(head.begin(), head.end(), rng);
            order.assign(tail.begin(), tail.end()); order.insert(order.end(), head.begin(), head.end());
        } else {
            std::shuffle(order.begin(), order.end(), rng);
        }
        order.resize(N);
        // Decode records into X (F32, column per sample), ids, gates (renormalised).
        std::vector<float> X((size_t) N * D); std::vector<uint16_t> ids((size_t) N * K); std::vector<float> g((size_t) N * K);
        for (size_t n = 0; n < N; n++) {
            const uint8_t * r = db.data() + order[n] * rec;
            ggml_fp16_to_fp32_row((const ggml_fp16_t *) r, X.data() + n * D, D);
            memcpy(ids.data() + n * K, r + (size_t) D * 2, (size_t) K * 2);
            ggml_fp16_to_fp32_row((const ggml_fp16_t *) (r + (size_t) D * 2 + (size_t) K * 2), g.data() + n * K, K);
            float s = 0; for (int j = 0; j < K; j++) s += g[n * K + j];
            for (int j = 0; j < K; j++) g[n * K + j] /= std::max(s, 1e-9f);
        }
        db.clear(); db.shrink_to_fit();
        // Validation first Nv samples, training the rest (already shuffled).
        const float * Xv = X.data(); const uint16_t * idv = ids.data();
        const size_t tr0 = Nv;

        // Device: X (all samples), W, b, T (targets of one batch).
        ggml_init_params ip{}; ip.mem_size = ggml_tensor_overhead() * 8; ip.no_alloc = true;
        ggml_context * pc = ggml_init(ip);
        ggml_tensor * Xd = ggml_new_tensor_2d(pc, GGML_TYPE_F32, D, (int64_t) N);
        ggml_tensor * Wd = ggml_new_tensor_2d(pc, GGML_TYPE_F32, D, E);
        ggml_tensor * bd = ggml_new_tensor_1d(pc, GGML_TYPE_F32, E);
        ggml_tensor * Td = ggml_new_tensor_2d(pc, GGML_TYPE_F32, E, B);
        ggml_backend_buffer_t pb = ggml_backend_alloc_ctx_tensors_from_buft(pc, buft);
        if (!pb) { fprintf(stderr, "layer %2d: cannot allocate %.1f GB on the device\n", il, (double) N * D * 4 / 1e9); return 1; }
        ggml_backend_tensor_set(Xd, X.data(), 0, X.size() * 4);
        ggml_backend_tensor_set(Wd, W.data(), 0, W.size() * 4);
        ggml_backend_tensor_set(bd, b.data(), 0, b.size() * 4);
        ggml_gallocr_t ga = ggml_gallocr_new(buft);

        // Scores of `n` samples starting at `off`: S = W x + b, downloaded.
        std::vector<float> S((size_t) E * B);
        auto scores = [&](size_t off, int n, float * out) {
            ggml_init_params gp{}; gp.mem_size = ggml_tensor_overhead() * 16 + ggml_graph_overhead(); gp.no_alloc = true;
            ggml_context * c = ggml_init(gp); ggml_cgraph * gr = ggml_new_graph(c);
            ggml_tensor * Xb = ggml_view_2d(c, Xd, D, n, Xd->nb[1], off * Xd->nb[1]);
            ggml_tensor * Sg = ggml_add(c, ggml_mul_mat(c, Wd, Xb), bd);
            ggml_build_forward_expand(gr, Sg);
            ggml_gallocr_alloc_graph(ga, gr); ggml_backend_graph_compute(be, gr);
            ggml_backend_tensor_get(Sg, out, 0, (size_t) E * n * 4);
            ggml_free(c);
        };
        auto val_recall = [&]() {
            double hits = 0; size_t done = 0;
            while (done < Nv) {
                const int n = (int) std::min<size_t>(B, Nv - done);
                scores(done, n, S.data());
                hits += recall_at_k(S.data(), E, n, idv + done * K, K, PK) * n;
                done += n;
            }
            return hits / Nv;
        };
        const double base = val_recall();
        double best = base; std::vector<float> bestW = W0, bestb = b;

        // Adam state on the host.
        std::vector<float> mW(W.size(), 0.0f), vW(W.size(), 0.0f), mb(E, 0.0f), vb(E, 0.0f), gW(W.size()), gb(E), T((size_t) E * B);
        const float b1 = 0.9f, b2 = 0.999f, eps = 1e-8f; int step = 0;
        std::vector<size_t> tidx(Nt); std::iota(tidx.begin(), tidx.end(), tr0);
        for (int ep = 0; ep < a.epochs; ep++) {
            // Whole batches only; the training set is shuffled once, and the
            // batch order reshuffled per epoch by rotating the start.
            const size_t nb = Nt / B;
            std::vector<size_t> starts(nb); for (size_t i = 0; i < nb; i++) starts[i] = tr0 + i * B;
            std::shuffle(starts.begin(), starts.end(), rng);
            for (size_t bi = 0; bi < nb; bi++) {
                const size_t off = starts[bi];
                std::fill(T.begin(), T.end(), 0.0f);
                for (int t = 0; t < B; t++)
                    for (int j = 0; j < K; j++) T[(size_t) t * E + ids[(off + t) * K + j]] = g[(off + t) * K + j];
                ggml_backend_tensor_set(Td, T.data(), 0, T.size() * 4);

                ggml_init_params gp{}; gp.mem_size = ggml_tensor_overhead() * 32 + ggml_graph_overhead(); gp.no_alloc = true;
                ggml_context * c = ggml_init(gp); ggml_cgraph * gr = ggml_new_graph(c);
                ggml_tensor * Xb = ggml_view_2d(c, Xd, D, B, Xd->nb[1], off * Xd->nb[1]);
                ggml_tensor * Sg = ggml_add(c, ggml_mul_mat(c, Wd, Xb), bd);          // [E, B]
                ggml_tensor * P  = ggml_soft_max(c, Sg);
                ggml_tensor * dS = ggml_scale(c, ggml_sub(c, P, Td), 1.0f / B);        // [E, B]
                ggml_tensor * XbT = ggml_cont(c, ggml_transpose(c, Xb));               // [B, D]
                ggml_tensor * dST = ggml_cont(c, ggml_transpose(c, dS));               // [B, E]
                ggml_tensor * gWd = ggml_mul_mat(c, XbT, dST);                         // [D, E] = X dS^T
                ggml_tensor * gbd = ggml_sum_rows(c, dST);                             // [1, E]
                ggml_build_forward_expand(gr, gWd); ggml_build_forward_expand(gr, gbd);
                if (!ggml_gallocr_alloc_graph(ga, gr) || ggml_backend_graph_compute(be, gr) != GGML_STATUS_SUCCESS) {
                    fprintf(stderr, "layer %2d: compute failed\n", il); return 1;
                }
                ggml_backend_tensor_get(gWd, gW.data(), 0, gW.size() * 4);
                ggml_backend_tensor_get(gbd, gb.data(), 0, gb.size() * 4);
                ggml_free(c);

                step++;
                const float lr_t = a.lr * std::sqrt(1.0f - std::pow(b2, (float) step)) / (1.0f - std::pow(b1, (float) step));
                for (size_t i = 0; i < W.size(); i++) {
                    const float gr_ = gW[i] + 2.0f * a.l2 * (W[i] - W0[i]);
                    mW[i] = b1 * mW[i] + (1 - b1) * gr_; vW[i] = b2 * vW[i] + (1 - b2) * gr_ * gr_;
                    W[i] -= lr_t * mW[i] / (std::sqrt(vW[i]) + eps);
                }
                for (int i = 0; i < E; i++) {
                    mb[i] = b1 * mb[i] + (1 - b1) * gb[i]; vb[i] = b2 * vb[i] + (1 - b2) * gb[i] * gb[i];
                    b[i] -= lr_t * mb[i] / (std::sqrt(vb[i]) + eps);
                }
                ggml_backend_tensor_set(Wd, W.data(), 0, W.size() * 4);
                ggml_backend_tensor_set(bd, b.data(), 0, b.size() * 4);
            }
            const double r = val_recall();
            if (r > best) { best = r; bestW = W; bestb = b; }
        }
        outW[il] = bestW; outb[il] = bestb;
        sum_base += base; sum_best += best; n_done++;
        const double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        printf("layer %2d: %7zu samples  recall@%d/%d router %5.1f%%  ->  trained %5.1f%%   (%.0f s)\n",
               il, N, K, PK, 100 * base, 100 * best, dt);
        fflush(stdout);
        ggml_gallocr_free(ga); ggml_backend_buffer_free(pb); ggml_free(pc);
    }
    if (n_done) printf("mean over %d layers: router %.1f%%  ->  trained %.1f%%\n", n_done, 100 * sum_base / n_done, 100 * sum_best / n_done);

    FILE * f = fopen(a.out.c_str(), "wb");
    if (!f) { fprintf(stderr, "cannot write %s\n", a.out.c_str()); return 1; }
    const uint32_t hdr[5] = { 1u, (uint32_t) a.n_layer, (uint32_t) D, (uint32_t) E, (uint32_t) a.first };
    fwrite("QWPR", 1, 4, f); fwrite(hdr, 4, 5, f);
    std::vector<ggml_fp16_t> wh((size_t) E * D);
    for (int il = a.first; il < a.n_layer; il++) {
        if (outW[il].empty()) { fprintf(stderr, "layer %d has no head (router matrix missing); output incomplete\n", il); fclose(f); return 1; }
        ggml_fp32_to_fp16_row(outW[il].data(), wh.data(), (int64_t) E * D);
        fwrite(wh.data(), 2, wh.size(), f); fwrite(outb[il].data(), 4, E, f);
    }
    fclose(f);
    printf("wrote %s\n", a.out.c_str());
    ggml_backend_free(be);
    return 0;
}
