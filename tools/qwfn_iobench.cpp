// qwfn-iobench -- measure the thing the whole engine is built around: how fast
// expert slices and PLE rows can actually be pulled off NVMe, and what that
// implies for tokens per second.
//
// Includes an mmap demand-paging mode that reproduces what llama.cpp does today,
// so the two can be compared on an identical access pattern.

#include "qwfn_expert_cache.h"
#include "qwfn_io.h"
#include "qwfn_model.h"
#include "qwfn_ple.h"

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <random>
#include <string>
#include <sys/mman.h>
#include <unistd.h>
#include <vector>

using namespace qwfn;
using clk = std::chrono::steady_clock;
static double secs(clk::time_point a, clk::time_point b) {
    return std::chrono::duration<double>(b - a).count();
}

// Zipf sampler over n_expert experts. alpha=0 is uniform; the measured 94% hit
// rate from a 23 GB page cache over 55.8 GB of experts implies substantial skew.
struct zipf {
    std::vector<double> cdf;
    void init(uint32_t n, double alpha) {
        cdf.resize(n);
        double s = 0;
        for (uint32_t i = 0; i < n; i++) { s += 1.0 / std::pow((double) (i + 1), alpha); cdf[i] = s; }
        for (uint32_t i = 0; i < n; i++) cdf[i] /= s;
    }
    uint32_t operator()(std::mt19937_64 & g) const {
        const double u = std::uniform_real_distribution<double>(0, 1)(g);
        return (uint32_t) (std::lower_bound(cdf.begin(), cdf.end(), u) - cdf.begin());
    }
};

static void pick_experts(const zipf & z, std::mt19937_64 & g, uint32_t n_expert,
                         uint32_t k, std::vector<uint32_t> & out) {
    out.clear();
    while (out.size() < k) {
        const uint32_t e = std::min(z(g), n_expert - 1);
        if (std::find(out.begin(), out.end(), e) == out.end()) out.push_back(e);
    }
}

// ---------------------------------------------------------------- raw ceiling
static void bench_ceiling(model_index & mi, unsigned qd, int n_tokens_equiv) {
    printf("\n== raw expert-slice fetch over io_uring (no cache, every access a miss) ==\n");
    std::string err;
    io_engine io;
    if (!io.init(mi.shard_paths(), qd, true, err)) { printf("  init failed: %s\n", err.c_str()); return; }

    const hparams & hp = mi.hp();
    const uint32_t k = hp.n_expert_used;
    std::mt19937_64 g(1234);
    zipf z; z.init(hp.n_expert, 0.0);   // uniform: worst case for locality

    // One landing buffer per outstanding request.
    const size_t MAXREQ = 64 * EXPERT_NPARTS;
    std::vector<uint8_t *> bufs(MAXREQ);
    for (auto & b : bufs) b = (uint8_t *) dio_alloc(4u << 20);

    for (unsigned depth : {4u, 16u, 32u, 64u}) {
        if (depth > qd) continue;
        io.stat_reads = io.stat_bytes = 0;
        std::vector<uint32_t> ids;
        std::vector<io_request> reqs;
        uint64_t tags[256];
        uint64_t bytes = 0;
        const auto t0 = clk::now();

        for (int t = 0; t < n_tokens_equiv; t++) {
            for (uint32_t il = 0; il < hp.n_layer; il++) {
                pick_experts(z, g, hp.n_expert, k, ids);
                reqs.clear();
                for (uint32_t i = 0; i < k; i++)
                    for (int q = 0; q < EXPERT_NPARTS; q++) {
                        const byte_range r = mi.expert_range(il, ids[i], (expert_part) q);
                        reqs.push_back(io_request{ r.shard, r.offset, r.nbytes,
                                                   bufs[reqs.size() % MAXREQ], 0 });
                        bytes += r.nbytes;
                    }
                size_t sub = 0;
                while (sub < reqs.size()) {
                    const size_t want = std::min<size_t>(depth - io.in_flight(), reqs.size() - sub);
                    const size_t got = want ? io.submit(reqs.data() + sub, want) : 0;
                    sub += got;
                    if (io.in_flight() >= depth || (got == 0 && io.in_flight()))
                        io.reap(tags, 256, 1);
                }
                while (io.in_flight()) io.reap(tags, 256, 1);
            }
        }
        const double dt = secs(t0, clk::now());
        printf("  QD %-3u : %6.2f GB/s   %8.1f MB/token   -> %6.2f tok/s if every access misses\n",
               depth, bytes / dt / 1e9, bytes / 1e6 / n_tokens_equiv, n_tokens_equiv / dt);
    }
    for (auto b : bufs) dio_free(b);
}

// --------------------------------------------------------- mmap demand paging
static void bench_mmap(model_index & mi, int n_tokens_equiv) {
    printf("\n== same access pattern via mmap demand paging (what llama.cpp does today) ==\n");
    const hparams & hp = mi.hp();
    std::vector<void *> maps(mi.shard_paths().size(), nullptr);
    std::vector<size_t> sizes(mi.shard_paths().size(), 0);
    for (size_t i = 0; i < mi.shard_paths().size(); i++) {
        int fd = open(mi.shard_paths()[i].c_str(), O_RDONLY);
        if (fd < 0) { printf("  open failed\n"); return; }
        const off_t sz = lseek(fd, 0, SEEK_END);
        void * p = mmap(nullptr, (size_t) sz, PROT_READ, MAP_PRIVATE, fd, 0);
        close(fd);
        if (p == MAP_FAILED) { printf("  mmap failed\n"); return; }
        madvise(p, (size_t) sz, MADV_RANDOM);
        maps[i] = p; sizes[i] = (size_t) sz;
    }

    std::mt19937_64 g(1234);
    zipf z; z.init(hp.n_expert, 0.0);
    std::vector<uint32_t> ids;
    uint64_t bytes = 0, acc = 0;
    const auto t0 = clk::now();
    for (int t = 0; t < n_tokens_equiv; t++) {
        for (uint32_t il = 0; il < hp.n_layer; il++) {
            pick_experts(z, g, hp.n_expert, hp.n_expert_used, ids);
            for (uint32_t i = 0; i < ids.size(); i++)
                for (int q = 0; q < EXPERT_NPARTS; q++) {
                    const byte_range r = mi.expert_range(il, ids[i], (expert_part) q);
                    const volatile uint8_t * p = (const uint8_t *) maps[r.shard] + r.offset;
                    for (uint32_t o = 0; o < r.nbytes; o += 4096) acc += p[o];
                    bytes += r.nbytes;
                }
        }
    }
    const double dt = secs(t0, clk::now());
    printf("  demand paging: %6.2f GB/s -> %6.2f tok/s if every access misses  (checksum %" PRIu64 ")\n",
           bytes / dt / 1e9, n_tokens_equiv / dt, acc & 0xFF);
    for (size_t i = 0; i < maps.size(); i++) if (maps[i]) munmap(maps[i], sizes[i]);
}

// ------------------------------------------------------- cached decode loop
static void bench_tokens(model_index & mi, model_index * cold, size_t ram_bytes,
                         double alpha, int n_tokens, bool use_cold) {
    printf("\n== simulated decode: %d tokens, %.1f GB RAM cache, zipf alpha=%.2f, cold tier %s ==\n",
           n_tokens, ram_bytes / 1e9, alpha, use_cold && cold ? "on" : "off");
    expert_cache::config cfg;
    cfg.ram_bytes     = ram_bytes;
    cfg.use_cold_tier = use_cold;
    cfg.queue_depth   = 256;

    expert_cache ec;
    std::string err;
    if (!ec.init(&mi, cold, cfg, err)) { printf("  init failed: %s\n", err.c_str()); return; }
    printf("  cache holds %zu expert blocks (%.1f%% of %u)\n", ec.capacity_experts(),
           100.0 * ec.capacity_experts() / (mi.hp().n_layer * mi.hp().n_expert),
           mi.hp().n_layer * mi.hp().n_expert);

    const hparams & hp = mi.hp();
    std::mt19937_64 g(99);
    zipf z; z.init(hp.n_expert, alpha);
    std::vector<uint32_t> ids;
    std::vector<expert_handle> hs(hp.n_expert_used);

    // Warm the cache so the steady state is what gets measured.
    const int warm = std::min(n_tokens, 24);
    for (int t = 0; t < warm; t++)
        for (uint32_t il = 0; il < hp.n_layer; il++) {
            pick_experts(z, g, hp.n_expert, hp.n_expert_used, ids);
            ec.fetch(il, ids.data(), (uint32_t) ids.size(), hs.data());
        }

    const auto base = ec.stats();
    const auto t0 = clk::now();
    for (int t = 0; t < n_tokens; t++)
        for (uint32_t il = 0; il < hp.n_layer; il++) {
            pick_experts(z, g, hp.n_expert, hp.n_expert_used, ids);
            if (!ec.fetch(il, ids.data(), (uint32_t) ids.size(), hs.data())) { printf("  fetch failed\n"); return; }
        }
    const double dt = secs(t0, clk::now());
    const auto s = ec.stats();
    const uint64_t look = s.lookups - base.lookups, hit = s.hits - base.hits;
    const uint64_t disk = s.bytes_from_disk - base.bytes_from_disk;

    printf("  hit rate      : %.1f%%  (%" PRIu64 " of %" PRIu64 ")\n", 100.0 * hit / look, hit, look);
    printf("  disk traffic  : %.0f MB/token   %.2f GB/s\n", disk / 1e6 / n_tokens, disk / dt / 1e9);
    printf("  I/O-only rate : %.2f tok/s   (%.1f ms/token spent fetching experts)\n",
           n_tokens / dt, dt * 1000 / n_tokens);
}

// ------------------------------------------------------------------- PLE
static void bench_ple(model_index & mi, size_t row_cache_bytes, int n_tokens) {
    printf("\n== PLE n-gram table gather (%.0f MB row cache) ==\n", row_cache_bytes / 1e6);
    std::string err;
    io_engine io;
    if (!io.init(mi.shard_paths(), 256, true, err)) { printf("  init failed: %s\n", err.c_str()); return; }
    ple_table ple;
    if (!ple.init(&mi, &io, row_cache_bytes, err)) { printf("  init failed: %s\n", err.c_str()); return; }

    std::mt19937_64 g(7);
    std::vector<int32_t> toks(n_tokens);
    for (auto & t : toks) t = (int32_t) (g() % mi.hp().n_vocab);

    // Decode: one position at a time, blocking.
    const uint8_t * rows[32];
    auto t0 = clk::now();
    for (int i = 0; i < n_tokens; i++) {
        const ple_rows r = ple.rows_for(toks.data(), n_tokens, i);
        ple.gather(r, rows);
    }
    double dt = secs(t0, clk::now());
    printf("  decode  : %.2f us/token  -> %.0f tok/s ceiling from PLE alone\n",
           dt * 1e6 / n_tokens, n_tokens / dt);

    // Prefill: whole prompt at once, sorted + deduped.
    std::vector<ple_rows> batch(n_tokens);
    for (int i = 0; i < n_tokens; i++) batch[i] = ple.rows_for(toks.data(), n_tokens, i);
    t0 = clk::now();
    ple.gather_batch(batch);
    dt = secs(t0, clk::now());
    printf("  prefill : %d tokens (%d rows) in %.2f s -> %.0f tok/s, %" PRIu64 " dedup hits\n",
           n_tokens, n_tokens * (int) mi.hp().ple_n_head(), dt, n_tokens / dt, ple.stat_dedup_saved);
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr,
            "usage: qwfn-iobench <shard.gguf> [options]\n"
            "  --cold <shard.gguf>   second checkpoint for the cold tier (e.g. UD-IQ1_S)\n"
            "  --ram <GB>            RAM cache size for the decode simulation (default 4,\n"
            "                        clamped to 60%% of MemAvailable minus 3 GB headroom)\n"
            "  --alpha <f>           zipf skew for simulated routing (default 0.9)\n"
            "  --tokens <n>          tokens to simulate (default 40)\n"
            "  --skip-mmap           do not run the demand-paging comparison\n");
        return 1;
    }
    std::string cold_path;
    double ram_gb = 4.0, alpha = 0.9;   // conservative default: the arena is anonymous RAM
    int n_tokens = 40;
    bool skip_mmap = false;
    for (int i = 2; i < argc; i++) {
        const std::string a = argv[i];
        if (a == "--cold" && i + 1 < argc) cold_path = argv[++i];
        else if (a == "--ram" && i + 1 < argc) ram_gb = atof(argv[++i]);
        else if (a == "--alpha" && i + 1 < argc) alpha = atof(argv[++i]);
        else if (a == "--tokens" && i + 1 < argc) n_tokens = atoi(argv[++i]);
        else if (a == "--skip-mmap") skip_mmap = true;
    }

    model_index mi;
    std::string err;
    if (!mi.load(argv[1], err)) { fprintf(stderr, "error: %s\n", err.c_str()); return 1; }
    printf("%s\n", mi.hp().summary().c_str());

    model_index cold;
    model_index * coldp = nullptr;
    if (!cold_path.empty()) {
        if (cold.load(cold_path, err)) coldp = &cold;
        else fprintf(stderr, "cold tier unavailable: %s\n", err.c_str());
    }

    bench_ceiling(mi, 64, 3);
    if (!skip_mmap) bench_mmap(mi, 1);
    bench_ple(mi, 256ull << 20, 4000);
    bench_tokens(mi, coldp, (size_t) (ram_gb * 1e9), alpha, n_tokens, false);
    if (coldp) bench_tokens(mi, coldp, (size_t) (ram_gb * 1e9), alpha, n_tokens, true);
    return 0;
}
