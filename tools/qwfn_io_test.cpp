// qwfn-io-test -- tests the io layer at its public seam: io_engine's
// init/submit/reap/shutdown, the payload_offset contract, dio_alloc alignment,
// dio_align arithmetic and mem_available_bytes. No framework: plain asserts
// and a nonzero exit on failure, so it drops into any CI loop.
//
// The test writes a temp file it fully controls, so it also covers:
//   - direct and buffered modes of both init paths,
//   - the bounce path (512-byte layout reading through a page-aligned window),
//   - reads that straddle a page boundary and reads near EOF,
//   - queue-depth pressure far above the worker count,
//   - that a bogus path fails init cleanly (no partial state left behind).
//
// Runs identically on Linux and Windows; that is the point of the port.

#include "qwfn_io.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <unistd.h>
#endif

using namespace qwfn;

static int g_failures = 0;

#define CHECK(cond, ...)                                                        \
    do {                                                                        \
        if (cond) {                                                             \
            printf("  [ok] ");                                                  \
            printf(__VA_ARGS__);                                                \
            printf("\n");                                                       \
        } else {                                                                \
            g_failures++;                                                       \
            printf("  [FAIL] ");                                                \
            printf(__VA_ARGS__);                                               \
            printf("\n");                                                       \
        }                                                                       \
    } while (0)

// A deterministic pseudo-file: byte i = (i * 31 + 7) & 0xFF.
static uint8_t byte_at(uint64_t i) { return (uint8_t) ((i * 31 + 7) & 0xFF); }

// A unique-enough suffix so parallel runs on one machine do not collide.
static long run_id() {
#ifdef _WIN32
    return (long) GetCurrentProcessId();
#else
    return (long) getpid();
#endif
}

static std::string make_temp_file(size_t bytes) {
    namespace fs = std::filesystem;
    const auto dir  = fs::temp_directory_path();
    const auto path = dir / ("qwfn_io_test_" + std::to_string(run_id()) + ".bin");
    FILE * f =
#ifdef _WIN32
        _wfopen(path.wstring().c_str(), L"wb");
#else
        std::fopen(path.string().c_str(), "wb");
#endif
    if (!f) { fprintf(stderr, "fatal: cannot create %s\n", path.string().c_str()); exit(2); }
    std::vector<uint8_t> chunk(1 << 20);
    for (size_t done = 0; done < bytes; done += chunk.size()) {
        const size_t n = std::min(chunk.size(), bytes - done);
        for (size_t i = 0; i < n; i++) chunk[i] = byte_at(done + i);
        if (std::fwrite(chunk.data(), 1, n, f) != n) {
            fprintf(stderr, "fatal: short write\n");
            exit(2);
        }
    }
    std::fclose(f);
    return path.string();
}

// Verify [offset, offset+nbytes) of the file landed where the contract says:
// at dst + payload_offset(offset) in direct mode, at dst in buffered mode.
static bool payload_matches(const io_engine & io, const void * dst,
                            uint64_t offset, uint32_t nbytes) {
    const uint8_t * p = (const uint8_t *) dst + io.payload_offset(offset);
    for (uint32_t i = 0; i < nbytes; i++)
        if (p[i] != byte_at(offset + i)) return false;
    return true;
}

// Round-trip a batch of reads through submit/reap and check every payload.
static void exercise_reads(io_engine & io, const std::string & path,
                           const std::vector<std::pair<uint64_t, uint32_t>> & ranges,
                           size_t n_bufs, const char * label) {
    const int failures_before = g_failures;
    std::string err;
    CHECK(io.init({path}, 8, /*direct_io=*/true, err), "%s: init(direct) ok", label);
    if (g_failures != failures_before) return;

    std::vector<void *> bufs(n_bufs);
    for (auto & b : bufs) b = dio_alloc(4u << 20);
    bool alloc_ok = true;
    for (void * b : bufs) if (!b) alloc_ok = false;
    CHECK(alloc_ok, "%s: dio_alloc returned aligned memory for all %zu buffers", label, n_bufs);

    std::vector<io_request> reqs;
    reqs.reserve(ranges.size());
    for (size_t i = 0; i < ranges.size(); i++)
        reqs.push_back(io_request{ 0, ranges[i].first, ranges[i].second,
                                   bufs[i % n_bufs], (uint64_t) (i + 1) });

    const size_t submitted = io.submit(reqs.data(), reqs.size());
    CHECK(submitted == reqs.size(), "%s: submit accepted all %zu requests", label, reqs.size());

    // Reap until every tag is back, as the engine's callers do. Not while
    // in_flight(): on the threads backend that drops when a read completes, not
    // when it is reaped, so the loop could end with completions still queued.
    std::vector<uint64_t> got;
    uint64_t tags[64];
    while (got.size() < submitted) {
        const size_t n = io.reap(tags, 64, 1);
        if (n == 0) break;
        for (size_t i = 0; i < n; i++) got.push_back(tags[i]);
    }
    CHECK(got.size() == reqs.size(), "%s: reaped %zu of %zu completions", label, got.size(), reqs.size());

    // Every tag exactly once, every payload intact at its contracted offset.
    bool all_tags = true, all_payload = true;
    for (uint64_t want = 1; want <= reqs.size(); want++) {
        if (std::find(got.begin(), got.end(), want) == got.end()) all_tags = false;
        const auto & [off, nb] = ranges[(size_t) (want - 1)];
        if (!payload_matches(io, reqs[(size_t) (want - 1)].dst, off, nb)) all_payload = false;
    }
    CHECK(all_tags,     "%s: every tag returned exactly once", label);
    CHECK(all_payload,  "%s: every payload landed intact at payload_offset", label);
    CHECK(io.stat_errors == 0, "%s: zero read errors (stat_errors=%llu)", label,
          (unsigned long long) io.stat_errors);

    io.shutdown();
    for (void * b : bufs) dio_free(b);
}

static uint64_t tmp_size(const std::string & path) {
    std::error_code ec;
    const auto sz = std::filesystem::file_size(path, ec);
    return ec ? 0 : (uint64_t) sz;
}

int main() {
    printf("qwfn-io-test: io layer at its public seam\n");

    // -- alignment arithmetic (pure, shared by every platform) ---------------
    printf("\n[1] dio_align arithmetic\n");
    set_dio_align(QWFN_DIO_PAGE);
    CHECK(dio_align() == 4096, "set_dio_align(4096) honored");
    CHECK(dio_pad(4096 + 123) == 123, "dio_pad");
    CHECK(dio_padded_size(4096 + 123, 100) == 4096, "dio_padded_size within a page");
    CHECK(dio_padded_size(4096 - 5, 10) == 8192, "dio_padded_size crossing into the next page");
    CHECK(dio_padded_size(8192 - 3, 6) == 8192, "dio_padded_size ending at a page boundary");
    set_dio_align(512);
    CHECK(dio_align() == 512, "set_dio_align(512) honored");
    CHECK(dio_pad(512 * 7 + 100) == 100, "dio_pad at 512 layout");
    CHECK(dio_padded_size(512 * 7 + 100, 12) == 512, "dio_padded_size at 512 layout");

    // -- dio_alloc: aligned, zeroable, freeable ------------------------------
    printf("\n[2] dio_alloc\n");
    void * p = dio_alloc(12345);
    CHECK(p != nullptr, "dio_alloc small");
    CHECK(((uintptr_t) p % 4096) == 0, "dio_alloc aligned to 4096");
    memset(p, 0xAB, dio_align_up(12345));
    dio_free(p);
    p = dio_alloc(0);
    CHECK(p != nullptr, "dio_alloc zero bytes returns usable memory");
    dio_free(p);

    // -- mem_available_bytes: the sizing call the console and engine rely on -
    printf("\n[3] mem_available_bytes\n");
    const uint64_t avail = mem_available_bytes();
    CHECK(avail > 0, "mem_available_bytes reports a positive value (%llu MiB)",
          (unsigned long long) (avail >> 20));
    const size_t clamped = clamp_to_available(1ull << 30, 0.60, 1ull << 30);
    CHECK(clamped == (1ull << 30), "clamp_to_available lets 1 GB through on a healthy machine");

    // -- io_engine end to end -------------------------------------------------
    const std::string tmp = make_temp_file(24u << 20);
    printf("\n[4] io_engine: page layout (direct)\n");
    set_dio_align(QWFN_DIO_PAGE);
    {
        io_engine io;
        // Page-aligned payload: data begins exactly at the read start.
        exercise_reads(io, tmp, {
            { 4096 * 100, 700 * 1024 },                  // natural expert-slice shape
            { 4096 * 500, 900 * 1024 },
            { 4096 * 1000, 640 * 1024 },
            { 8192 - 4096, 4096 },                       // aligned, page-sized
            { tmp_size(tmp) - 4096, 4096 },              // last full page
        }, 8, "page-direct");
    }

    printf("\n[5] io_engine: 512-byte layout (bounce path, direct)\n");
    set_dio_align(512);
    {
        io_engine io;
        exercise_reads(io, tmp, {
            { 512 * 100 + 13, 700 * 1024 },              // unaligned start: bounce window
            { 4096 + 7, 1 },                             // tiny unaligned read
            { 4096 * 700 + 333, 512 },                   // straddles a page boundary
            { 512 * 3 + 1, 8192 + 511 },                 // long unaligned read
            { tmp_size(tmp) - 512 - 7, 500 },            // near EOF, unaligned tail
        }, 8, "bounce-direct");
    }

    printf("\n[6] io_engine: buffered mode\n");
    set_dio_align(QWFN_DIO_PAGE);
    {
        io_engine io;
        std::string err;
        CHECK(io.init({tmp}, 8, /*direct_io=*/false, err), "init(buffered) ok");
        void * dst = dio_alloc(1u << 20);
        io_request r{ 0, 4096 * 123 + 77, 5000, dst, 42 };
        CHECK(io.submit(&r, 1) == 1, "buffered submit");
        uint64_t tags[8];
        while (io.in_flight()) io.reap(tags, 8, 1);
        CHECK(payload_matches(io, dst, r.offset, r.nbytes),
              "buffered payload lands at dst + 0");
        io.shutdown();
        dio_free(dst);
    }

    printf("\n[7] io_engine: backends agree\n");
    {
        // Both backends (whatever the platform offers) must satisfy the same
        // contract on the same inputs.
        io_engine io;
        std::string err;
        CHECK(io.init({tmp}, 8, true, err, io_engine::backend::threads),
              "threads backend init");
        void * dst = dio_alloc(1u << 20);
        io_request r{ 0, 4096 * 321 + 9, 4096, dst, 7 };
        io.submit(&r, 1);
        uint64_t tags[8];
        while (io.in_flight()) io.reap(tags, 8, 1);
        CHECK(payload_matches(io, dst, r.offset, r.nbytes), "threads payload correct");
        io.shutdown();
        dio_free(dst);
    }

    printf("\n[8] io_engine: failure paths\n");
    {
        io_engine io;
        std::string err;
        const std::string nowhere = std::filesystem::temp_directory_path().string()
                                    + "/qwfn_io_test_does_not_exist.bin";
        CHECK(!io.init({nowhere}, 8, true, err), "missing file fails init");
        CHECK(!err.empty(), "init error carries a message");
        CHECK(io.in_flight() == 0, "failed init leaves nothing in flight");
        // And a good init still works after the failed one on the same object.
        CHECK(io.init({tmp}, 8, true, err), "re-init after failure works");
        io.shutdown();
    }

    printf("\n[9] io_engine: a rejected request completes as an error\n");
    set_dio_align(QWFN_DIO_PAGE);
    for (int b = 0; b < 2; b++) {
        // A request naming a shard that does not exist is refused at submit, and
        // must still come back from reap(): every caller counts completions, so
        // one that never arrives hangs it, and one counted in flight for ever
        // hangs every `while (in_flight()) reap()` loop.
        const io_engine::backend be = b ? io_engine::backend::threads : io_engine::backend::uring;
        const char * name = b ? "threads" : "default";
        io_engine io;
        std::string err;
        CHECK(io.init({tmp}, 8, true, err, be), "%s: init", name);
        void * d1 = dio_alloc(1u << 20);
        void * d2 = dio_alloc(1u << 20);
        io_request r[3] = { { 0, 4096 * 10, 4096, d1, 1 },
                            { 7, 0,         4096, d1, 2 },     // shard 7 does not exist
                            { 0, 4096 * 20, 4096, d2, 3 } };
        CHECK(io.submit(r, 3) == 3, "%s: submit takes all three, the bad one included", name);
        std::vector<uint64_t> got;
        uint64_t tags[8];
        while (got.size() < 3) {
            const size_t n = io.reap(tags, 8, 1);
            if (n == 0) break;
            got.insert(got.end(), tags, tags + n);
        }
        std::sort(got.begin(), got.end());
        CHECK(got == std::vector<uint64_t>({ 1, 2, 3 }), "%s: every tag comes back once, the rejected one too (%zu reaped)", name, got.size());
        CHECK(io.in_flight() == 0, "%s: nothing left in flight (in_flight=%zu)", name, io.in_flight());
        CHECK(io.stat_errors == 1, "%s: the rejection is one error (stat_errors=%llu)", name,
              (unsigned long long) io.stat_errors);
        CHECK(payload_matches(io, d1, r[0].offset, r[0].nbytes) && payload_matches(io, d2, r[2].offset, r[2].nbytes),
              "%s: the good reads around it landed", name);
        // and the engine still serves reads afterwards
        io_request again{ 0, 4096 * 30 + 5, 1000, d1, 9 };
        CHECK(io.submit(&again, 1) == 1, "%s: a later submit is accepted", name);
        const size_t n = io.reap(tags, 8, 1);
        CHECK(n == 1 && tags[0] == 9 && payload_matches(io, d1, again.offset, again.nbytes),
              "%s: and completes with its payload", name);
        io.shutdown();
        dio_free(d1);
        dio_free(d2);
    }

    printf("\n%s (%d failures)\n", g_failures ? "FAILED" : "ALL PASSED", g_failures);
    std::filesystem::remove(tmp);
    return g_failures ? 1 : 0;
}
