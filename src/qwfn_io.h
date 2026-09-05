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

struct io_uring;

namespace qwfn {

// Logical block size assumed for O_DIRECT alignment. NVMe reports 512; reads
// must have offset, length and buffer address all aligned to this.
static constexpr uint64_t QWFN_DIO_ALIGN = 512;

inline uint64_t dio_align_down(uint64_t x) { return x & ~(QWFN_DIO_ALIGN - 1); }
inline uint64_t dio_align_up  (uint64_t x) { return (x + QWFN_DIO_ALIGN - 1) & ~(QWFN_DIO_ALIGN - 1); }

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
    void *   dst     = nullptr;  // QWFN_DIO_ALIGN-aligned, >= dio_padded_size() bytes
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
    backend          be_ = backend::uring;
    io_uring *       ring_ = nullptr;

    // --- thread-pool backend ---
    struct job { int shard; uint64_t off; uint32_t len; void * dst; uint64_t tag; };
    std::vector<std::thread>  workers_;
    std::deque<job>           q_;
    std::deque<uint64_t>      done_;
    std::mutex                mtx_;
    std::condition_variable   cv_work_, cv_done_;
    bool                      stop_ = false;
    void worker_loop();
    std::vector<int> fds_;
    bool             direct_ = true;
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
