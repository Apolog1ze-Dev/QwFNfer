#include "qwfn_io.h"

#ifdef _WIN32
// Windows runs the thread-pool backend over unbuffered reads (see qwfn_io.h).
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <filesystem>
#else
#include <liburing.h>
#include <fcntl.h>
#include <unistd.h>
#endif

#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>

namespace qwfn {

static uint64_t g_dio_align = 512;
uint64_t dio_align() { return g_dio_align; }
void     set_dio_align(uint64_t a) { g_dio_align = a == QWFN_DIO_PAGE ? QWFN_DIO_PAGE : 512; }

#ifdef _WIN32

// --- aligned allocation ----------------------------------------------------
// Same contract as the POSIX side: 4096-aligned, dio_align_up-sized, freeable
// with dio_free. _aligned_malloc already rounds internally; the explicit
// rounding keeps the *size* contract identical (>= dio_align_up(bytes)).
void * dio_alloc(size_t bytes) {
    const size_t sz = dio_align_up(bytes ? bytes : 1);
    return _aligned_malloc(sz, 4096);
}

void dio_free(void * p) { _aligned_free(p); }

// --- available memory ------------------------------------------------------
// GlobalMemoryStatusEx's ullAvailPhys is the Windows analogue of
// MemAvailable: what can be handed out without swapping to the pagefile.
uint64_t mem_available_bytes() {
    MEMORYSTATUSEX s;
    s.dwLength = sizeof(s);
    if (!GlobalMemoryStatusEx(&s)) return 0;
    return s.ullAvailPhys;
}

#else // !_WIN32

void * dio_alloc(size_t bytes) {
    void * p = nullptr;
    const size_t sz = dio_align_up(bytes);
    if (posix_memalign(&p, 4096, sz) != 0) return nullptr;
    return p;
}

void dio_free(void * p) { free(p); }

uint64_t mem_available_bytes() {
    FILE * f = fopen("/proc/meminfo", "r");
    if (!f) return 0;
    char line[256];
    uint64_t kb = 0;
    while (fgets(line, sizeof(line), f)) {
        if (sscanf(line, "MemAvailable: %lu kB", &kb) == 1) break;
    }
    fclose(f);
    return kb * 1024ull;
}

#endif // _WIN32

size_t clamp_to_available(size_t want, double frac, size_t headroom) {
    const uint64_t avail = mem_available_bytes();
    if (avail == 0) return want;                       // unknown: trust the caller
    const uint64_t budget = (uint64_t) ((double) avail * frac);
    const uint64_t safe   = budget > headroom ? budget - headroom : 0;
    if (safe == 0) return 0;
    if ((uint64_t) want <= safe) return want;
    fprintf(stderr,
            "[qwfn] requested %.1f GB RAM tier but only %.1f GB is available; "
            "clamping to %.1f GB (%.0f%% of available memory minus %.1f GB headroom)\n",
            want / 1e9, avail / 1e9, safe / 1e9, frac * 100, headroom / 1e9);
    return (size_t) safe;
}

io_engine::~io_engine() { shutdown(); }

// --- shard files -------------------------------------------------------------
// The one place the platforms differ below the backends: how a shard is opened
// for direct reads, read at an offset, and closed. Everything else -- the thread
// pool, the bounce path, the completion accounting -- is shared, so a fix to one
// platform's backend is a fix to both.

#ifdef _WIN32

// FILE_FLAG_NO_BUFFERING is the documented Windows equivalent of O_DIRECT:
// sector-aligned reads that bypass the cache manager, with the same alignment
// contract (offset, length and destination follow dio_align()). The handle is
// opened overlapped so positional reads from several workers run in parallel; a
// synchronous handle serializes them.
static bool open_shard(const std::string & p, bool & direct, io_engine::file_t & out, std::string & err) {
    const std::wstring wp = std::filesystem::path(p).wstring();
    HANDLE h = CreateFileW(wp.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                           FILE_FLAG_OVERLAPPED | (direct ? FILE_FLAG_NO_BUFFERING : 0), nullptr);
    if (h == INVALID_HANDLE_VALUE && direct) {
        // Some filesystems refuse unbuffered access; fall back rather than fail, as on Linux.
        h = CreateFileW(wp.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                        FILE_FLAG_OVERLAPPED, nullptr);
        if (h != INVALID_HANDLE_VALUE) direct = false;
    }
    if (h == INVALID_HANDLE_VALUE) {
        char buf[256] = {};
        FormatMessageA(FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
                       nullptr, GetLastError(), 0, buf, sizeof(buf), nullptr);
        err = "open failed for " + p + ": " + buf;
        return false;
    }
    out = h;
    return true;
}

static void close_shard(io_engine::file_t f) { CloseHandle((HANDLE) f); }

// n bytes at off, looping over short pieces like the POSIX pread loop; returns
// the bytes read (short at end of file) or -1. Each worker thread has its own
// event: an OVERLAPPED with a NULL event shared by several threads makes
// GetOverlappedResult unreliable, and the io test caught exactly that.
static int64_t read_full(io_engine::file_t f, void * dst, size_t n, uint64_t off) {
    struct thread_event { HANDLE h = nullptr; ~thread_event() { if (h) CloseHandle(h); } };
    static thread_local thread_event tev;   // closed when the worker exits
    if (!tev.h && !(tev.h = CreateEventW(nullptr, TRUE, FALSE, nullptr))) return -1;
    const HANDLE ev = tev.h;
    size_t done = 0;
    while (done < n) {
        OVERLAPPED ov{};
        const uint64_t at = off + done;
        ov.Offset     = (DWORD) (at & 0xFFFFFFFFull);
        ov.OffsetHigh = (DWORD) (at >> 32);
        ov.hEvent     = ev;
        DWORD got = 0;
        if (!ReadFile((HANDLE) f, (char *) dst + done, (DWORD) (n - done), nullptr, &ov)) {
            const DWORD e = GetLastError();
            if (e == ERROR_HANDLE_EOF) break;
            if (e != ERROR_IO_PENDING) return done ? (int64_t) done : -1;
        }
        if (!GetOverlappedResult((HANDLE) f, &ov, &got, TRUE)) {
            if (GetLastError() == ERROR_HANDLE_EOF) break;
            return done ? (int64_t) done : -1;
        }
        if (got == 0) break;
        done += got;
    }
    return (int64_t) done;
}

#else // !_WIN32

static bool open_shard(const std::string & p, bool & direct, io_engine::file_t & out, std::string & err) {
    int fd = ::open(p.c_str(), O_RDONLY | (direct ? O_DIRECT : 0));
    if (fd < 0 && direct) {
        // Some filesystems refuse O_DIRECT; fall back rather than fail.
        fd = ::open(p.c_str(), O_RDONLY);
        if (fd >= 0) direct = false;
    }
    if (fd < 0) {
        err = "open failed for " + p + ": " + strerror(errno);
        return false;
    }
    out = fd;
    return true;
}

static void close_shard(io_engine::file_t f) { ::close(f); }

// pread is positional and thread-safe, so the shard fds are shared by the workers.
static int64_t read_full(io_engine::file_t fd, void * dst, size_t n, uint64_t off) {
    size_t done = 0;
    while (done < n) {
        const ssize_t r = ::pread(fd, (char *) dst + done, n - done, (off_t) (off + done));
        if (r <= 0) break;
        done += (size_t) r;
    }
    return (int64_t) done;
}

#endif // _WIN32

bool io_engine::init(const std::vector<std::string> & paths, unsigned queue_depth,
                     bool direct_io, std::string & err, backend be) {
    shutdown();
    direct_ = direct_io;
#ifdef _WIN32
    be_     = backend::threads;   // no io_uring here; see the backend notes in qwfn_io.h
    (void) be;
#else
    be_     = be;
#endif
    qd_     = queue_depth ? queue_depth : 256;

    for (const auto & p : paths) {
        file_t f{};
        if (!open_shard(p, direct_, f, err)) { shutdown(); return false; }
        fds_.push_back(f);
    }

    // With a 512-byte layout a direct read of the exact window would be served
    // buffered on a 4096-sector filesystem: the workers read a page-aligned
    // window into their own buffer and copy the payload into the slot instead.
    bounce_ = direct_ && dio_align() < QWFN_DIO_PAGE;
    if (be_ == backend::threads) {
        const unsigned n = qd_ ? std::min(qd_, 32u) : 8u;
        stop_ = false;
        for (unsigned i = 0; i < n; i++) workers_.emplace_back([this] { worker_loop(); });
        return true;
    }

#ifndef _WIN32
    ring_ = (io_uring *) calloc(1, sizeof(io_uring));
    if (!ring_) { err = "out of memory allocating io_uring"; shutdown(); return false; }

    int rc = io_uring_queue_init(qd_, ring_, 0);
    if (rc < 0) {
        free(ring_);
        ring_ = nullptr;
        err = std::string("io_uring_queue_init failed: ") + strerror(-rc);
        shutdown();
        return false;
    }

    // Registering the fds removes a per-op file table lookup. If it fails we must
    // fall back to real fds: submitting with IOSQE_FIXED_FILE against an
    // unregistered table makes every read fail with -EBADF, and the destination
    // buffer then keeps whatever malloc left there.
    const int rr = io_uring_register_files(ring_, fds_.data(), (unsigned) fds_.size());
    registered_files = rr == 0;
    if (!registered_files) {
        fprintf(stderr, "[qwfn] io_uring_register_files failed (%s); using plain fds\n", strerror(-rr));
    }
#endif
    return true;
}

void io_engine::shutdown() {
    if (!workers_.empty()) {
        { std::lock_guard<std::mutex> lk(mtx_); stop_ = true; }
        cv_work_.notify_all();
        for (auto & t : workers_) if (t.joinable()) t.join();
        workers_.clear();
        q_.clear(); done_.clear();
        stop_ = false;
    }
#ifndef _WIN32
    if (ring_) {
        io_uring_queue_exit(ring_);
        free(ring_);
        ring_ = nullptr;
    }
#endif
    for (file_t f : fds_) close_shard(f);
    fds_.clear();
    rejected_.clear();
    in_flight_ = 0;
    min_expect_ = 0;
}

size_t io_engine::submit(const io_request * reqs, size_t n) {
    if (be_ == backend::threads) {
        const auto t0 = std::chrono::steady_clock::now();
        size_t queued_jobs = 0;
        {
            std::lock_guard<std::mutex> lk(mtx_);
            for (size_t i = 0; i < n; i++) {
                const io_request & r = reqs[i];
                if (r.shard < 0 || (size_t) r.shard >= fds_.size() || !r.dst || r.nbytes == 0) {
                    // Completed here as an error rather than dropped: see rejected_.
                    // Returning n while quietly queueing fewer jobs is what used to
                    // leave fetch_end() waiting on a completion nothing would produce.
                    // It goes straight to done_ and is NOT counted in in_flight_: on
                    // this backend in_flight_ is reads not yet completed, which the
                    // workers decrement, and none will ever decrement it for this one
                    // -- counting it left in_flight() stuck above zero for good, and
                    // every `while (in_flight()) reap(...)` loop waiting forever.
                    stat_errors++;
                    done_.push_back(r.tag);
                    continue;
                }
                uint64_t off = r.offset;
                uint32_t len = r.nbytes;
                if (direct_) { off = dio_align_down(r.offset); len = dio_padded_size(r.offset, r.nbytes); }
                q_.push_back(job{ r.shard, off, len, r.dst, r.tag, r.offset, r.nbytes });
                in_flight_++;
                queued_jobs++;
            }
        }
        // notify_all(), not one notify_one() per job. Waking exactly as many
        // workers as there are jobs looks cheaper, but it was measured slower:
        // k notify_one() calls are k futex syscalls made serially by the
        // submitting thread, and the workers they wake start one after another,
        // while a single notify_all() starts them together. On an i9-12900H /
        // RTX 3080 Ti Laptop, UD-Q4_K_XL, 128-token decode, the per-job version
        // raised the per-token read wait from 63 to 68 ms (7.92 -> 7.75 tok/s).
        if (queued_jobs) cv_work_.notify_all();
        if (queued_jobs < n) cv_done_.notify_all();   // the rejected ones are already in done_
        stat_t_prep += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        return n;
    }

#ifndef _WIN32
    if (!ring_) return 0;
    // `prepped` counts SQEs; `accepted` counts requests this call takes
    // responsibility for completing, which includes the rejected ones.
    size_t prepped = 0, accepted = 0;
    const auto t_prep0 = std::chrono::steady_clock::now();

    for (size_t i = 0; i < n; i++) {
        const io_request & r = reqs[i];
        if (r.shard < 0 || (size_t) r.shard >= fds_.size() || !r.dst || r.nbytes == 0) {
            // Completed here as an error rather than dropped: see rejected_.
            // Skipping it used to shift every later request's position in the
            // caller's retry loop, which then resubmitted the same bad request
            // for ever.
            stat_errors++;
            rejected_.push_back(r.tag);
            in_flight_++;
            accepted++;
            continue;
        }

        io_uring_sqe * sqe = io_uring_get_sqe(ring_);
        if (!sqe) break;   // ring full; caller should reap and retry

        uint64_t off = r.offset;
        uint32_t len = r.nbytes;
        if (direct_) {
            off = dio_align_down(r.offset);
            len = dio_padded_size(r.offset, r.nbytes);
        }

        io_uring_prep_read(sqe, registered_files ? r.shard : fds_[r.shard], r.dst, len, off);
        if (registered_files) sqe->flags |= IOSQE_FIXED_FILE;
        expect_[prepped & 1023] = len;
        if (min_expect_ == 0 || len < min_expect_) min_expect_ = len;
        io_uring_sqe_set_data64(sqe, r.tag);
        prepped++;
        accepted++;
    }

    const auto t_prep1 = std::chrono::steady_clock::now();
    stat_t_prep += std::chrono::duration<double>(t_prep1 - t_prep0).count();

    if (prepped) {
        const int rc = io_uring_submit(ring_);
        stat_t_submit_syscall += std::chrono::duration<double>(
                std::chrono::steady_clock::now() - t_prep1).count();
        // A short or failed submit does NOT lose the entries: liburing has
        // already advanced the SQ tail, so whatever the kernel did not take
        // stays in the ring and goes out on the next io_uring_submit(). They
        // will therefore all complete, and all of them count as in flight --
        // what used to hang the decode loop was that nothing called submit()
        // again before the wait, so reap() flushes the ring before waiting.
        if (rc < 0 || (size_t) rc < prepped) stat_errors++;
        in_flight_ += prepped;
    }
    return accepted;
#else
    return 0;
#endif
}

size_t io_engine::reap(uint64_t * tags_out, size_t max_tags, size_t min_complete) {
    if (be_ == backend::threads) {
        size_t got = 0;
        std::unique_lock<std::mutex> lk(mtx_);
        while (got < max_tags) {
            if (done_.empty()) {
                if (got >= min_complete) break;
                // Nothing in flight and nothing done: a caller whose count has
                // drifted would wait here forever. Return short instead; the
                // caller reports a failed read, which beats a silent hang.
                if (in_flight_ == 0) break;
                cv_done_.wait(lk, [this] { return !done_.empty() || in_flight_ == 0; });
                if (done_.empty()) break;
            }
            tags_out[got++] = done_.front();
            done_.pop_front();
        }
        return got;
    }

#ifndef _WIN32
    if (!ring_ || in_flight_ == 0) return 0;

    size_t got = 0;
    if (min_complete > in_flight_) min_complete = in_flight_;

    // Requests rejected at submit: complete in the only sense the caller tracks.
    while (got < max_tags && !rejected_.empty()) {
        tags_out[got++] = rejected_.back();
        rejected_.pop_back();
        in_flight_--;
    }
    if (in_flight_ == 0) return got;

    // io_uring_submit() takes as many SQEs as the kernel will accept and leaves
    // the rest in the ring for the next submit. Nothing else calls submit()
    // between the caller's last one and the wait below, so a short submit would
    // park entries the kernel never saw and io_uring_wait_cqe() would block on
    // completions that cannot arrive. Flush first.
    if (io_uring_sq_ready(ring_) > 0) {
        const int rc = io_uring_submit(ring_);
        if (rc <= 0 && io_uring_sq_ready(ring_) > 0) {
            // The ring will not drain. Hand back what is already there rather
            // than waiting for ever; the caller reports a short read, which is
            // recoverable, where a hang is not.
            stat_errors++;
            min_complete = 0;
        }
    }

    while (got < max_tags) {
        io_uring_cqe * cqe = nullptr;
        int rc;
        if (got < min_complete) {
            rc = io_uring_wait_cqe(ring_, &cqe);
        } else {
            rc = io_uring_peek_cqe(ring_, &cqe);
            if (rc == -EAGAIN || !cqe) break;
        }
        if (rc < 0) { stat_errors++; break; }

        if (cqe->res < 0) {
            stat_errors++;
        } else {
            stat_reads++;
            stat_bytes += (uint64_t) cqe->res;
            if ((uint32_t) cqe->res < min_expect_) stat_short++;
        }
        tags_out[got++] = io_uring_cqe_get_data64(cqe);
        io_uring_cqe_seen(ring_, cqe);
        in_flight_--;
        if (in_flight_ == 0) break;
    }
    return got;
#else
    return 0;
#endif
}


void io_engine::worker_loop() {
    for (;;) {
        job j;
        {
            std::unique_lock<std::mutex> lk(mtx_);
            cv_work_.wait(lk, [this] { return stop_ || !q_.empty(); });
            if (stop_ && q_.empty()) return;
            j = q_.front();
            q_.pop_front();
        }
        int64_t got = 0;
        if (bounce_) {
            // The page-aligned window around the requested range, into this
            // worker's buffer; the payload then goes where the 512-byte layout
            // expects it. A window past the end of a shard reads short, which is
            // fine as long as the payload arrived.
            static thread_local uint8_t * scratch = nullptr;
            static thread_local size_t    scratch_bytes = 0;
            const uint64_t w0 = j.ooff & ~(QWFN_DIO_PAGE - 1);
            const uint64_t w1 = (j.ooff + j.onb + QWFN_DIO_PAGE - 1) & ~(QWFN_DIO_PAGE - 1);
            const size_t   wl = (size_t) (w1 - w0);
            if (scratch_bytes < wl) {
                if (scratch) dio_free(scratch);
                scratch_bytes = wl + (1u << 20);
                scratch = (uint8_t *) dio_alloc(scratch_bytes);
            }
            const int64_t need = (int64_t) (j.ooff - w0 + j.onb);
            if (scratch) got = read_full(fds_[j.shard], scratch, wl, w0);
            if (got >= need) {
                memcpy((char *) j.dst + dio_pad(j.ooff), scratch + (j.ooff - w0), j.onb);
                got = (int64_t) j.len;   // the caller's notion of a complete read
            } else {
                got = 0;
            }
        } else {
            got = read_full(fds_[j.shard], j.dst, j.len, j.off);
        }
        {
            std::lock_guard<std::mutex> lk(mtx_);
            if (got < (int64_t) j.len) stat_errors++;
            else { stat_reads++; stat_bytes += (uint64_t) got; }
            done_.push_back(j.tag);
            in_flight_--;
        }
        cv_done_.notify_all();
    }
}

} // namespace qwfn
