#pragma once
// Three-tier routed-expert cache.
//
// The model activates 10 of 512 experts on each of 48 layers, so one generated
// token touches 480 expert blocks -- about 1.05 GB of quantized weights out of
// a 55.8 GB expert set. At the 15 tok/s target that is ~15.8 GB/s of demand.
//
// Nothing holds 55.8 GB here, so the cache is what makes the target reachable:
//   T0  VRAM   ~4.7 GB free after the dense core, KV and vision  (~2,100 experts)
//   T1  RAM    ~19 GB                                            (~8,700 experts)
//   T2  NVMe   everything else, read at slice granularity over io_uring
//
// Routing is strongly skewed: a 23 GB page cache over 55.8 GB of experts was
// measured serving ~94% of activations, so a managed cache with frequency-based
// admission should do at least as well.
//
// Misses may optionally be served from a second, more aggressively quantized
// checkpoint (UD-IQ1_S): a cold expert then costs ~0.96 MB instead of ~2.18 MB.
// Quality loss is confined to rarely-routed experts. The handle reports the
// ggml_type actually obtained so the caller dequantizes correctly.

#include "qwfn_io.h"
#include "qwfn_model.h"

#include "ggml-backend.h"

#include <cstdint>
#include <string>
#include <vector>

namespace qwfn {

struct expert_handle {
    const uint8_t *       part[EXPERT_NPARTS] = {nullptr, nullptr, nullptr};
    ggml_type             type[EXPERT_NPARTS] = {GGML_TYPE_F32, GGML_TYPE_F32, GGML_TYPE_F32};
    // Which buffer `part` points into. The caller attaches this to the graph
    // tensor, and ggml_backend_sched then places the matmul on that device:
    // VRAM-resident experts compute on the GPU, RAM-resident ones on the CPU.
    ggml_backend_buffer_t buffer = nullptr;
    bool                  on_gpu = false;
    bool                  from_cold = false;
    // Slot index inside the tier `part` points into (VRAM tier when on_gpu,
    // RAM tier otherwise): the expert id a mul_mat_id over that tier wants.
    int32_t               slot = -1;
    // Promoted to VRAM by this very fetch: a graph that ran before the fetch
    // did not see it there, so the caller computes it separately (engine: the
    // late fold).
    bool                  late = false;
};

// One tier of one layer as three expert-major arrays, one per part, so that a
// single ggml_mul_mat_id over [n_embd, n_ff, n_slots] selects experts by slot.
// The VRAM tier is laid out this way natively (natural row stride, contiguous);
// the RAM arena keeps its interleaved gate|up|down blocks and is viewed with a
// per-slot byte stride, which the CPU kernels address in bytes.
struct tier_view {
    uint8_t *             part[EXPERT_NPARTS]   = {nullptr, nullptr, nullptr};   // slot 0 of each part
    size_t                stride[EXPERT_NPARTS] = {0, 0, 0};                     // slot-to-slot, bytes
    ggml_type             type[EXPERT_NPARTS]   = {GGML_TYPE_F32, GGML_TYPE_F32, GGML_TYPE_F32};
    uint32_t              n_slots = 0;
    ggml_backend_buffer_t buffer  = nullptr;
};

struct expert_cache_stats {
    uint64_t lookups = 0, hits = 0, misses = 0;
    uint64_t bytes_from_disk = 0;
    uint64_t cold_tier_reads = 0;
    uint64_t evictions = 0;
    uint64_t gpu_hits = 0;      // served from the VRAM tier
    uint64_t promotions = 0;    // blocks copied RAM -> VRAM
    uint64_t pf_issued  = 0;    // blocks speculatively fetched
    uint64_t pf_used    = 0;    // of those, later actually requested
    uint64_t pf_wasted  = 0;    // evicted or replaced before use
    uint64_t ram_released = 0;  // RAM slots freed once the block reached VRAM
    uint64_t upgrades   = 0;    // cold blocks refetched at full precision on reuse
    uint64_t warm_admitted = 0; // blocks copied into the RAM tier from a prefill's staging
    uint64_t warm_promoted = 0; // of those, pushed on to VRAM
    double   t_warm     = 0;
    // Where the time inside a fetch actually goes.
    double   t_submit   = 0;    // building + submitting the reads
    double   t_promote  = 0;    // host->device copies warming the VRAM tier
    double   t_wait     = 0;    // blocked on completions
    uint64_t n_bursts   = 0;    // fetch_begin calls that issued at least one read
    uint64_t n_reads    = 0;    // individual io_uring reads issued
    uint64_t bytes_read = 0;
    double   hit_rate() const { return lookups ? (double) hits / (double) lookups : 0.0; }
    double   gpu_rate() const { return lookups ? (double) gpu_hits / (double) lookups : 0.0; }
};

class expert_cache {
public:
    struct config {
        size_t ram_bytes      = 16ull << 30;  // T1 capacity
        // The MemAvailable clamp on ram_bytes: frac of MemAvailable minus
        // headroom (see clamp_to_available). Defaults match the historical
        // guard; raising frac trades the rest of the machine for hit rate.
        double ram_frac       = 0.60;
        size_t ram_headroom   = 3ull << 30;
        size_t vram_bytes     = 0;            // T0 capacity; 0 disables the GPU tier
        ggml_backend_buffer_type_t vram_buft = nullptr;
        // Pinned (cudaMallocHost) memory for the RAM arena, when the device
        // offers it. A host->device copy from pageable memory is staged through
        // the driver and blocks the caller even when issued "async"; from pinned
        // memory it is a DMA the copy engine runs on its own. Measured: a 2.18 MB
        // block takes 135 us pageable, 94 us pinned, and ~0 us of caller time
        // when pinned and asynchronous.
        ggml_backend_buffer_type_t host_buft = nullptr;
        // Backend whose stream the promotion copies are queued on. The engine
        // synchronises it once per layer anyway (the async GPU MoE settle), and
        // settle_promotions() is where the RAM slots are finally released.
        ggml_backend_t vram_backend = nullptr;
        bool async_promote = true;
        // Extra device bytes appended to the tier that the prefill streamer
        // borrows as its staging. During decode they hold expert slots like
        // the rest of the tier (the layers whose slots fall in that tail are
        // invalidated when a prefill starts and refilled afterwards), so the
        // 1.79 GB staging is no longer idle for the whole of generation.
        size_t lend_bytes = 0;
        // Device memory to leave alone when sizing the VRAM tier. The replayed
        // layer graphs each keep their own allocator and are created lazily on
        // the first decode token -- long after the tier has sized itself -- so
        // without a reservation the tier takes their memory and the first token
        // dies with a cudaMalloc failure. Measured need is well under this; the
        // slack also absorbs the per-call input arena, which grows with n_kv.
        size_t vram_reserve = 768ull << 20;
        bool   use_cold_tier  = true;         // serve misses from the IQ1_S checkpoint
        unsigned queue_depth  = 256;
        io_engine::backend io_backend = io_engine::backend::uring;
        // Bound the H2D traffic spent warming T0: one block is ~2.18 MB, so
        // unbounded promotion would cost more than it saves on the first tokens.
        uint32_t max_promotions_per_layer = 2;
        // Eviction samples this many slots and drops the coldest. Sampling keeps
        // it O(1); 8 was measurably worse than LFU-optimal, so widen it.
        uint32_t evict_samples = 16;
        // Which signal decides the victim. Routing drifts as the text evolves,
        // so long-run frequency alone ages badly; recency tracks the drift.
        enum class evict_policy { lru, lfu, hybrid } policy = evict_policy::lru;
        // Halve every expert counter after this many lookups, so an expert that
        // was hot early cannot pin a slot forever.
        uint64_t age_every = 1u << 16;
    };

    bool init(const model_index * hot, const model_index * cold,
              const config & cfg, std::string & err);
    void shutdown();

    // Blocking fetch of `n` experts on one layer. `out` must hold n handles.
    bool fetch(uint32_t layer, const uint32_t * expert_ids, uint32_t n, expert_handle * out);

    // Split form. fetch_begin() fills every handle and submits reads for the
    // misses without waiting for anything: `ready[i]` says whether expert i's
    // weights are valid now. An expert whose speculative read is still in
    // flight comes back with ready = false and no new read; fetch_end() waits
    // for those and for the demand reads. The caller computes the ready ones
    // while every read lands -- the engine keeps that exact by summing every
    // CPU expert once, in selection order, after both passes.
    bool fetch_begin(uint32_t layer, const uint32_t * expert_ids, uint32_t n,
                     expert_handle * out, bool * ready);
    bool fetch_end();
    // Wait only for the in-flight speculative reads the last fetch_begin found,
    // flipping their `ready` flags; the demand reads stay in flight.
    bool settle_pending(const uint32_t * expert_ids, uint32_t n, bool * ready);

    // --- speculative prefetch -------------------------------------------
    // Issue reads for one or more layers' *predicted* selections without
    // waiting. The reads then overlap the previous layer's MoE compute, so a
    // correct prediction is indistinguishable from a cache hit and the queue
    // stops draining to zero between layers. Passing two sets (L+1 and L+2)
    // widens the window: the L+2 reads get two layers of compute to land in.
    //
    // Completion is tracked per block, and fetch_begin(L) waits only for the
    // entries targeting layer L. Waiting for everything outstanding would put
    // the L+2 reads -- issued one MoE ago, almost never landed yet -- on L+1's
    // critical path, which measured as whole seconds of stall.
    // The VRAM residency of one layer as two tables indexed by expert id: the
    // tier slot (0 when absent) and a 1/0 mask -- what a graph needs to run the
    // resident experts by mul_mat_id without asking the host. vram_version()
    // changes whenever the tables would.
    uint64_t vram_version(uint32_t layer) const;
    void     vram_table(uint32_t layer, int32_t * slot, float * mask) const;

    struct pf_set {
        uint32_t         layer = 0;
        const uint32_t * ids   = nullptr;
        uint32_t         n     = 0;
    };
    void prefetch_begin(const pf_set * sets, uint32_t n_sets);
    void prefetch_begin(uint32_t layer, const uint32_t * expert_ids, uint32_t n) {
        pf_set s{layer, expert_ids, n};
        prefetch_begin(&s, 1);
    }
    void prefetch_settle();               // drain everything outstanding

    // Fire-and-forget prefetch used by the router-lookahead path. Reads land in
    // the cache; a later fetch() for the same experts then hits.
    void prefetch(uint32_t layer, const uint32_t * expert_ids, uint32_t n);

    // Pin a precomputed hot set (from qwfn-profile) so it is never evicted.
    void pin(const std::vector<uint32_t> & packed_keys);

    // Reads submitted by the last fetch_begin() and not yet waited for, or
    // speculative reads it found still in flight.
    bool has_inflight() const { return inflight_reqs_ > 0 || inflight_cold_reqs_ > 0 || !pending_.empty(); }
    // What the cache is blocked on right now, for a stall watchdog: 0 nothing,
    // 1 demand reads (fetch_end), 2 speculative reads (settle). Racy by design.
    int    wait_state() const { return wait_state_; }
    size_t pf_outstanding() const { return pf_reads_outstanding_; }   // speculative reads in flight
    // Promotions a fetch may make; a two-token step looks up ~1.7x the experts and gets a budget to match.
    void   set_max_promotions(uint32_t n) { cfg_.max_promotions_per_layer = n; }
    size_t wait_count() const { return wait_state_ == 1 ? inflight_reqs_ + inflight_cold_reqs_ : wait_state_ == 2 ? pf_reads_outstanding_ : 0; }

    // Wait for the asynchronous promotion copies queued since the last call and
    // release the RAM slots they read from. Until then those slots stay valid
    // and cannot be evicted, which is what makes the copy safe: nothing can
    // refill a block the copy engine may still be reading.
    void settle_promotions();
    bool arena_pinned() const { return arena_pinned_; }

    // The VRAM tier is two buffers: a permanent one and a dynamic one of
    // lend_bytes that holds expert slots during decode and is FREED for the
    // duration of a prefill, so the prefill's staging, batch buffers and graph
    // arenas can take that memory. lend_begin() invalidates those layers'
    // slots and frees the buffer; lend_end() allocates it again, empty, and
    // refills it from the RAM tier. Pointers into it change across that, so
    // tier_epoch() advances and graphs built against it must be rebuilt.
    size_t    lent_bytes() const { return extra_bytes_; }
    uint64_t  tier_epoch() const { return tier_epoch_; }
    void      lend_begin();
    void      lend_end();

    // Tier layouts for the mul_mat_id MoE paths. n_slots == 0 when absent.
    tier_view gpu_tier(uint32_t layer) const;
    tier_view ram_tier(uint32_t layer) const;

    // Warm the tiers from a prefill. The streaming prefill bypasses the cache,
    // so generation after a prompt used to start cold (11.4 tok/s against 15+
    // warm). The prompt's own routing is the best predictor of what the next
    // tokens will route to: `items` lists experts most-useful-first (the last
    // tokens' picks, then earlier ones) with their use counts and pointers to
    // their three slices in host memory. The first n_vram are also pushed to
    // the VRAM tier. Copies run on several threads; promotions are the usual
    // asynchronous ones, settled by settle_promotions().
    struct warm_item {
        uint32_t        expert = 0;
        uint32_t        count  = 0;
        const uint8_t * part[EXPERT_NPARTS] = {nullptr, nullptr, nullptr};
    };
    void warm(uint32_t layer, const warm_item * items, uint32_t n, uint32_t n_vram);

    // A CPU backend buffer spanning the whole arena. Expert tensors in the graph
    // alias cached blocks through this instead of being copied: one expert is
    // 2.18 MB and a token touches 480 of them, so copying would add ~1 GB of
    // memory traffic per token -- as much as the weights themselves.
    ggml_backend_buffer_t arena_buffer() const { return arena_buf_; }
    ggml_backend_buffer_t vram_buffer()  const { return vram_buf_; }
    ggml_backend_buffer_t vram_extra_buffer() const { return vram_extra_; }
    const io_engine &     io() const { return cold_ ? io_cold_ : io_hot_; }
    uint8_t *             arena()        const { return arena_; }

    const expert_cache_stats & stats() const { return st_; }
    size_t   capacity_experts() const { return total_slots_; }
    size_t   capacity_experts_gpu() const { return total_gslots_; }
    uint32_t block_bytes(uint32_t layer) const { return layer < blk_.size() ? blk_[layer].block_bytes : 0; }

    static uint32_t pack(uint32_t layer, uint32_t id) { return (layer << 16) | id; }

private:
    // Per-layer slot pool. Every layer sees identical traffic (10 experts per
    // token), so each gets an equal *byte* share of the arena; slot counts then
    // differ because a few layers quantize ffn_down at Q8_0 and are bigger.
    struct layer_pool {
        uint32_t block_bytes = 0;     // gate+up+down, each padded for O_DIRECT
        uint32_t part_bytes[EXPERT_NPARTS] = {0, 0, 0};   // natural slice size of each part
        uint32_t part_off[EXPERT_NPARTS] = {0, 0, 0};   // offset of each part in a slot
        uint32_t part_pay[EXPERT_NPARTS] = {0, 0, 0};   // payload offset inside each part
        ggml_type part_type[EXPERT_NPARTS] = {GGML_TYPE_F32, GGML_TYPE_F32, GGML_TYPE_F32};
        // The cold checkpoint quantizes differently and its tensors sit at
        // different file offsets, so a slot filled from it needs its own types
        // and payload offsets. Without this a cold block is later served as a
        // hit and decoded with the hot types, which is silently catastrophic.
        ggml_type cold_type[EXPERT_NPARTS] = {GGML_TYPE_F32, GGML_TYPE_F32, GGML_TYPE_F32};
        uint32_t  cold_pay[EXPERT_NPARTS]  = {0, 0, 0};
        uint32_t n_slots = 0;
        uint8_t * base = nullptr;                  // arena_ + offset

        std::vector<uint16_t> slot_expert;         // expert id in slot, or 0xFFFF
        std::vector<uint8_t>  slot_valid;
        std::vector<uint8_t>  slot_cold;      // provenance: filled from the cold checkpoint
        std::vector<uint8_t>  slot_speculative; // admitted by a prediction, not yet used
        std::vector<uint8_t>  slot_pinned;
        std::vector<uint32_t> slot_freq;
        // Frequency belongs to the EXPERT, not the slot. Keeping it per-slot
        // destroyed the count on eviction and started every admission at 1, so
        // a freshly admitted block was almost always the next victim -- the
        // cache could not tell a one-off expert from a hot one it had just
        // re-admitted. 512 counters per layer is 98 KB for the whole model.
        std::vector<uint32_t> ef;
        std::vector<uint64_t> slot_used;   // tick of last use, for recency
        std::vector<int32_t>  expert_slot;         // expert id -> slot, or -1
        // Whether this expert has ever been fetched. First touch may come from
        // the cold checkpoint; anything fetched again is worth full precision,
        // so the cold tier stays confined to the one-off tail it was meant for.
        std::vector<uint8_t>  seen;

        // T0: device memory, but NOT the same layout. Three expert-major
        // arrays (gate, up, down) with the natural slice size as the slot
        // stride, so one mul_mat_id covers every VRAM-resident expert of the
        // layer. CUDA's mmvq derives the expert stride as nb[2]/type_size with
        // integer division, so the interleaved 2,176,000-byte block stride
        // (not a multiple of iq3_xxs's 98-byte block) would silently read the
        // wrong addresses; natural slices are whole rows of whole blocks.
        uint32_t              g_slots = 0;
        uint8_t *             g_part[EXPERT_NPARTS]       = {nullptr, nullptr, nullptr};
        uint32_t              g_part_bytes[EXPERT_NPARTS] = {0, 0, 0};
        bool                  g_in_lent = false;   // slots live in the dynamic (prefill-time freed) buffer
        bool                  g_lent    = false;   // ... and that buffer is gone right now
        ggml_backend_buffer_t g_buf     = nullptr; // the buffer the slot arrays live in
        size_t                g_off     = 0;       // offset of the first array within it
        std::vector<uint16_t> g_slot_expert;
        std::vector<uint8_t>  g_valid;
        std::vector<uint8_t>  g_cold;
        std::vector<uint32_t> g_freq;
        std::vector<uint64_t> g_used;      // tick of last use, for recency
        std::vector<uint64_t> g_used_fetch; // fetch_count_ of last use, for staleness
        std::vector<int32_t>  g_expert_slot;
        uint64_t              g_ver = 1;          // bumped on every residency change
    };

    // Copy one host block into a device slot, evicting the coldest if needed.
    bool promote(layer_pool & lp, uint32_t expert_id, const uint8_t * host_block);
    void fill_gpu_handle(const layer_pool & lp, uint32_t gslot, expert_handle & h) const;

    int32_t  find_slot(layer_pool & lp, uint32_t expert_id) const;
    // Slots referenced by the fetch() call in progress. Every handle it has
    // already returned points into one of these, so none may be evicted until
    // the caller is done with them.
    std::vector<int32_t> live_;
    // Outstanding reads from fetch_begin(), waiting for fetch_end().
    size_t               inflight_reqs_ = 0;
    uint32_t             inflight_layer_ = 0;
    std::vector<int32_t> inflight_slots_;
    std::vector<uint32_t> inflight_experts_;
    size_t               inflight_cold_reqs_ = 0;   // of those, reads from the cold checkpoint (its own engine)
    // Experts of the fetch in progress whose speculative read had not landed at
    // fetch_begin: valid after fetch_end (or settle_pending).
    struct pending_spec { uint32_t expert; int32_t slot; };
    std::vector<pending_spec> pending_;
    bool read_block_now(layer_pool & lp, uint32_t layer, int32_t slot, uint32_t expert);   // a blocking full-precision read into a claimed slot
    volatile int         wait_state_ = 0;

    // Outstanding speculative reads, one entry per claimed block. The io tag is
    // the entry's index, so completions can be matched out of order and a block
    // goes valid the moment its last read lands -- independent of any other
    // entry. The table is append-only while reads are in flight (tags index it)
    // and compacted whenever it drains.
    struct pf_entry {
        uint32_t layer = 0, expert = 0;
        int32_t  slot  = -1;
        uint8_t  remaining = 0;    // reads still in flight for this block
    };
    std::vector<pf_entry> pf_pending_;
    size_t                pf_reads_outstanding_ = 0;
    void pf_reap(size_t min_complete);        // drain completions, mark blocks valid
    void prefetch_settle_layer(uint32_t layer);
    void prefetch_settle_for(uint32_t layer, const uint32_t * ids, uint32_t n);
    bool pf_layer_pending(uint32_t layer) const;
    int32_t  choose_victim(layer_pool & lp);
    uint8_t * slot_ptr(layer_pool & lp, uint32_t slot) const { return lp.base + (size_t) slot * lp.block_bytes; }
    void     fill_handle(const layer_pool & lp, uint32_t slot, expert_handle & h) const;

    const model_index * hot_ = nullptr;
    const model_index * cold_ = nullptr;
    io_engine io_hot_, io_cold_;
    // Speculative reads get their own engine. Sharing one would let fetch_end()
    // drain the prefetch's completions -- reap() returns everything available,
    // not just what was asked for -- which both deadlocks the settle and marks
    // slots valid before their data has landed.
    io_engine io_pf_;
    config    cfg_;

    uint8_t *              arena_ = nullptr;
    size_t                 arena_bytes_ = 0;
    ggml_backend_buffer_t  arena_buf_ = nullptr;
    ggml_backend_buffer_t  arena_hostbuf_ = nullptr;   // owns arena_ when pinned
    bool                   arena_pinned_ = false;
    bool                   last_promote_async_ = false;
    struct pending_rel { uint32_t layer, expert; int32_t slot; };
    std::vector<pending_rel> pending_release_;
    std::vector<layer_pool> blk_;
    size_t                 total_slots_ = 0;
    size_t                 total_gslots_ = 0;
    ggml_backend_buffer_t  vram_buf_ = nullptr;      // permanent part of the tier
    ggml_backend_buffer_t  vram_extra_ = nullptr;    // dynamic part, absent during a prefill
    size_t                 extra_bytes_ = 0;
    uint64_t               tier_epoch_ = 0;
    ggml_context *         xfer_ctx_ = nullptr;
    ggml_tensor *          xfer_     = nullptr;   // scratch handle for H2D copies
    uint64_t               tick_ = 0;
    uint64_t               fetch_epoch_ = 0;   // tick_ at the start of the current fetch
    uint64_t               fetch_count_ = 0;   // fetch_begin calls so far (48 per decoded token)
    uint64_t               ef_ticks_ = 0;   // lookups since the last halving
    expert_cache_stats     st_;
};

} // namespace qwfn
