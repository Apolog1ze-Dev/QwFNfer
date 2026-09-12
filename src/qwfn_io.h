#pragma once
// io_uring read engine.
//
// This exists because the measured bottleneck on the target machine is not
// bandwidth but *granularity*. Demand-paging the expert weights through mmap
// yields 4.4 KiB average reads at queue depth ~4 and tops out near 0.30 GB/s.
// The same NVMe sustains 7.0 GB/s when asked for 640 KiB at queue depth 4 --
// which is exactly the size of one expert slice. So: explicit, batched,
// slice-sized, O_DIRECT reads.
//
// O_DIRECT also keeps the kernel page cache out of the way. We manage the RAM
// tier ourselves; letting the page cache mirror it would halve our effective
// capacity on a 30 GB machine.

#include <condition_variable>
#include <cstdint>
#include <deque>
#include <mutex>
#include <thread>
#include <string>
#include <vector>

#include "qwfn_platform.h"

#ifdef _WIN32
// On Windows the thread-pool backend (also the engine's Linux default) is
// the only backend; the ring pointer below is unused and stays null.
struct io_uring;   // never defined on this platform; the member exists only to
                   // keep the class layout identical across platforms
#else
struct io_uring;
#endif

namespace qwfn {

// The alignment the RAM tier's slots and the prefill staging are laid out to.
// A direct read needs offset, length and destination aligned to what the
// FILESYSTEM requires, not the device: NVMe reports 512, but btrfs (sectorsize
// 4096) serves anything not 4096-aligned through the page cache, silently, and
// measured on 2026-09-09 every expert read of this engine was buffered. The
// layout is therefore 4096 whenever every expert slice stride in the file is a
// page multiple, so the payload sits at a fixed offset per part and the read
// lands in place with no copy (the Q4 file); it is 512 otherwise (the Q3 file:
// IQ3_XXS slices step by 512 mod 4096), and the thread backend then reads a
// page-aligned window into a per-worker bounce buffer and copies the payload
// into the slot -- still a direct read, no page cache. Set by the engine before
// any layout is computed.
uint64_t dio_align();
void     set_dio_align(uint64_t a);
static constexpr uint64_t QWFN_DIO_PAGE = 4096;

inline uint64_t dio_align_down(uint64_t x) { return x & ~(dio_align() - 1); }
inline uint64_t dio_align_up  (uint64_t x) { return (x + dio_align() - 1) & ~(dio_align() - 1); }

// Bytes of slack between the start of an aligned read and the requested data.
inline uint32_t dio_pad(uint64_t offset) { return (uint32_t) (offset - dio_align_down(offset)); }

// Size of the aligned read needed to cover [offset, offset+nbytes).
inline uint32_t dio_padded_size(uint64_t offset, uint32_t nbytes) {
    return (uint32_t) (dio_align_up(dio_pad(offset) + nbytes));
}

struct io_request {
    int      shard   = 0;        // index into the paths passed to init()
    uint64_t offset  = 0;        // logical byte offset of the wanted data
    uint32_t nbytes  = 0;        // logical size of the wanted data
    void *   dst     = nullptr;  // dio_align()-aligned, >= dio_padded_size() bytes
    uint64_t tag     = 0;        // returned verbatim on completion
};

class io_engine {
public:
    io_engine() = default;
    ~io_engine();
    io_engine(const io_engine &) = delete;
    io_engine & operator=(const io_engine &) = delete;

    // Two backends behind one interface.
    //
    //   uring   : io_uring. On this filesystem io_uring_submit() turns out to
    //             execute the reads inline rather than queueing them, so a
    //             burst gets far less than the concurrency it asked for --
    //             measured 4.07 GB/s on the engine's 8-read burst.
    //   threads : a pool of workers doing blocking positional preadv. Real
    //             kernel-level parallelism; measured 5.30 GB/s on the same
    //             burst shape.
    //
    // Windows runs the threads backend only, over unbuffered overlapped reads
    // (FILE_FLAG_NO_BUFFERING): sector-aligned slice reads that bypass the
    // cache manager, the documented Windows equivalent of O_DIRECT. Windows'
    // own IoRing (io_uring's SQ/CQ design, Win11 21H2+) was considered and
    // deliberately not used: it is Win11-only, its versioned op set is still
    // maturing, and its win is syscall batching at thousands of in-flight
    // ops -- not the engine's ~8-deep expert bursts, where per-read syscall
    // count is 1 in both designs. Measured on this project's target NVMe
    // (random 640 KiB-1 MiB slices, QD 4-64): thread pool + unbuffered reads
    // 3.4-4.8 GB/s vs IoRing 1.3-1.9 GB/s -- the port is 2.5-3.6x faster.
    // The submit/reap shape below is IoRing-like, so a future backend can
    // slot in without touching this interface.
    enum class backend { uring, threads };

    // queue_depth is the io_uring ring size / the worker count.
    bool init(const std::vector<std::string> & paths, unsigned queue_depth,
              bool direct_io, std::string & err, backend be = backend::uring);

    backend which() const { return be_; }
    void shutdown();

    // Queue reads. Returns the number accepted (short only if the ring is full).
    // Data for request i lands at dst + dio_pad(offset) when direct I/O is on,
    // and at dst when it is off.
    size_t submit(const io_request * reqs, size_t n);

    // Collect completions. Blocks until at least min_complete have arrived
    // (0 = purely opportunistic). Returns how many tags were written out.
    size_t reap(uint64_t * tags_out, size_t max_tags, size_t min_complete);

    size_t in_flight() const { return in_flight_; }
    bool   direct_io() const { return direct_; }

    // Offset within dst where the requested bytes actually begin.
    uint32_t payload_offset(uint64_t offset) const { return direct_ ? dio_pad(offset) : 0; }

    // Cumulative counters, for the benchmark and the runtime stats line.
    uint64_t stat_reads = 0, stat_bytes = 0, stat_errors = 0, stat_short = 0;
    double   stat_t_submit_syscall = 0;   // time inside io_uring_submit()
    double   stat_t_prep = 0;             // time building SQEs
    bool     registered_files = false;

private:
    backend          be_ = backend::threads;
    io_uring *       ring_ = nullptr;   // POSIX uring backend only; null on Windows

    // --- thread-pool backend ---
    struct job { int shard; uint64_t off; uint32_t len; void * dst; uint64_t tag; uint64_t ooff; uint32_t onb; };   // ooff/onb: the requested range, for the bounce path
    std::vector<std::thread>  workers_;
    std::deque<job>           q_;
    std::deque<uint64_t>      done_;
    std::mutex                mtx_;
    std::condition_variable   cv_work_, cv_done_;
    bool                      stop_ = false;
    void worker_loop();
    std::vector<int> fds_;
    bool             direct_ = true;
    bool             bounce_ = false;   // direct reads through a page-aligned per-worker buffer (512-byte slot layout)
    size_t           in_flight_ = 0;
    unsigned         qd_ = 0;
    uint32_t         expect_[1024] = {};
    uint32_t         min_expect_ = 0;
};

// Page-aligned allocation helper for O_DIRECT destination buffers.
void * dio_alloc(size_t bytes);
void   dio_free(void * p);

// --- memory safety -------------------------------------------------------
// This engine's RAM tier is one large anonymous arena, and the target machine
// has 30 GB of RAM behind 30 GB of zram swap at vm.swappiness=150. Asking for
// more than the kernel can actually spare does not fail the allocation -- it
// gets compressed into zram, which itself consumes RAM, and the box OOMs.
// Every arena sizing therefore goes through clamp_to_available().

// MemAvailable from /proc/meminfo: the kernel's own estimate of what can be
// handed out without swapping. Returns 0 if it cannot be read.
uint64_t mem_available_bytes();

// Largest arena we are willing to take: `frac` of MemAvailable, minus a fixed
// headroom for activations, CUDA host buffers and the rest of the desktop.
// Never returns more than `want`.
size_t clamp_to_available(size_t want, double frac = 0.60,
                          size_t headroom = 3ull << 30);

} // namespace qwfn
