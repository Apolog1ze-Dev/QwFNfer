#include "qwfn_io.h"

#include <liburing.h>

#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <unistd.h>

namespace qwfn {

static uint64_t g_dio_align = 512;
uint64_t dio_align() { return g_dio_align; }
void     set_dio_align(uint64_t a) { g_dio_align = a == QWFN_DIO_PAGE ? QWFN_DIO_PAGE : 512; }

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

size_t clamp_to_available(size_t want, double frac, size_t headroom) {
    const uint64_t avail = mem_available_bytes();
    if (avail == 0) return want;                       // unknown: trust the caller
    const uint64_t budget = (uint64_t) ((double) avail * frac);
    const uint64_t safe   = budget > headroom ? budget - headroom : 0;
    if (safe == 0) return 0;
    if ((uint64_t) want <= safe) return want;
    fprintf(stderr,
            "[qwfn] requested %.1f GB RAM tier but only %.1f GB is available; "
            "clamping to %.1f GB (%.0f%% of MemAvailable minus %.1f GB headroom)\n",
            want / 1e9, avail / 1e9, safe / 1e9, frac * 100, headroom / 1e9);
    return (size_t) safe;
}

io_engine::~io_engine() { shutdown(); }

bool io_engine::init(const std::vector<std::string> & paths, unsigned queue_depth,
                     bool direct_io, std::string & err, backend be) {
    shutdown();
    direct_ = direct_io;
    be_     = be;
    qd_     = queue_depth ? queue_depth : 256;

    for (const auto & p : paths) {
        int flags = O_RDONLY;
        if (direct_) flags |= O_DIRECT;
        int fd = ::open(p.c_str(), flags);
        if (fd < 0 && direct_) {
            // Some filesystems refuse O_DIRECT; fall back rather than fail.
            fd = ::open(p.c_str(), O_RDONLY);
            if (fd >= 0) direct_ = false;
        }
        if (fd < 0) {
            err = "open failed for " + p + ": " + strerror(errno);
            shutdown();
            return false;
        }
        fds_.push_back(fd);
    }

    // With a 512-byte layout a direct read of the exact window would be served
    // buffered on a 4096-sector filesystem: the workers read a page-aligned
    // window into their own buffer and copy the payload into the slot instead.
    bounce_ = direct_ && dio_align() < QWFN_DIO_PAGE;
    if (be_ == backend::threads) {
        // pread is positional and thread-safe, so the shard fds are shared.
        const unsigned n = qd_ ? std::min(qd_, 32u) : 8u;
        stop_ = false;
        for (unsigned i = 0; i < n; i++) workers_.emplace_back([this] { worker_loop(); });
        return true;
    }

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
    if (ring_) {
        io_uring_queue_exit(ring_);
        free(ring_);
        ring_ = nullptr;
    }
    for (int fd : fds_) if (fd >= 0) ::close(fd);
    fds_.clear();
    in_flight_ = 0;
}

size_t io_engine::submit(const io_request * reqs, size_t n) {
    if (be_ == backend::threads) {
        const auto t0 = std::chrono::steady_clock::now();
        {
            std::lock_guard<std::mutex> lk(mtx_);
            for (size_t i = 0; i < n; i++) {
                const io_request & r = reqs[i];
                if (r.shard < 0 || (size_t) r.shard >= fds_.size() || !r.dst || r.nbytes == 0) continue;
                uint64_t off = r.offset;
                uint32_t len = r.nbytes;
                if (direct_) { off = dio_align_down(r.offset); len = dio_padded_size(r.offset, r.nbytes); }
                q_.push_back(job{ r.shard, off, len, r.dst, r.tag, r.offset, r.nbytes });
                in_flight_++;
            }
        }
        cv_work_.notify_all();
        stat_t_prep += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        return n;
    }

    if (!ring_) return 0;
    size_t queued = 0;
    const auto t_prep0 = std::chrono::steady_clock::now();

    for (size_t i = 0; i < n; i++) {
        const io_request & r = reqs[i];
        if (r.shard < 0 || (size_t) r.shard >= fds_.size() || !r.dst || r.nbytes == 0) continue;

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
        expect_[queued & 1023] = len;
        if (min_expect_ == 0 || len < min_expect_) min_expect_ = len;
        io_uring_sqe_set_data64(sqe, r.tag);
        queued++;
    }

    const auto t_prep1 = std::chrono::steady_clock::now();
    stat_t_prep += std::chrono::duration<double>(t_prep1 - t_prep0).count();

    if (queued) {
        int rc = io_uring_submit(ring_);
        stat_t_submit_syscall += std::chrono::duration<double>(
                std::chrono::steady_clock::now() - t_prep1).count();
        if (rc < 0) { stat_errors++; return 0; }
        in_flight_ += queued;
    }
    return queued;
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

    if (!ring_ || in_flight_ == 0) return 0;

    size_t got = 0;
    if (min_complete > in_flight_) min_complete = in_flight_;

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
        ssize_t got = 0;
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
            const ssize_t need = (ssize_t) (j.ooff - w0 + j.onb);
            if (scratch) {
                while (got < (ssize_t) wl) {
                    const ssize_t r = ::pread(fds_[j.shard], scratch + got, wl - got, (off_t) (w0 + got));
                    if (r <= 0) break;
                    got += r;
                }
            }
            if (got >= need) {
                memcpy((char *) j.dst + dio_pad(j.ooff), scratch + (j.ooff - w0), j.onb);
                got = (ssize_t) j.len;   // the caller's notion of a complete read
            } else {
                got = 0;
            }
        } else {
            while (got < (ssize_t) j.len) {
                const ssize_t r = ::pread(fds_[j.shard], (char *) j.dst + got,
                                          j.len - got, (off_t) (j.off + got));
                if (r <= 0) break;
                got += r;
            }
        }
        {
            std::lock_guard<std::mutex> lk(mtx_);
            if (got < (ssize_t) j.len) stat_errors++;
            else { stat_reads++; stat_bytes += (uint64_t) got; }
            done_.push_back(j.tag);
            in_flight_--;
        }
        cv_done_.notify_all();
    }
}

} // namespace qwfn
