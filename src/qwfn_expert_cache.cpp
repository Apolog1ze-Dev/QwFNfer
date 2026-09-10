#include "qwfn_expert_cache.h"

#include <algorithm>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <cstring>
#include <thread>

namespace qwfn {

namespace {
inline uint64_t xorshift(uint64_t & s) {
    s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s;
}
constexpr uint16_t SLOT_EMPTY = 0xFFFF;
} // namespace

bool expert_cache::init(const model_index * hot, const model_index * cold,
                        const config & cfg, std::string & err) {
    shutdown();
    hot_  = hot;
    cold_ = cfg.use_cold_tier ? cold : nullptr;
    cfg_  = cfg;

    if (!io_hot_.init(hot->shard_paths(), cfg.queue_depth, /*direct_io=*/true, err, cfg.io_backend)) return false;
    {
        // The prefetch reads either file: hot shards first, the cold ones after.
        std::vector<std::string> pfp = hot->shard_paths();
        n_hot_shards_ = (uint32_t) pfp.size();
        if (cold_) for (const auto & p : cold_->shard_paths()) pfp.push_back(p);
        if (!io_pf_.init(pfp, cfg.queue_depth, /*direct_io=*/true, err, cfg.io_backend)) return false;
    }
    if (cold_ && !io_cold_.init(cold_->shard_paths(), cfg.queue_depth, true, err, cfg.io_backend)) {
        fprintf(stderr, "[qwfn] cold tier disabled: %s\n", err.c_str());
        cold_ = nullptr;
        err.clear();
    }

    const uint32_t n_layer = hot->hp().n_layer;
    blk_.resize(n_layer);

    // Work out each layer's slot layout, and check the assumption the O_DIRECT
    // path relies on: an expert slice is a whole number of 512-byte blocks, so
    // the read-around padding is constant per (layer, part) rather than
    // per-expert.

    for (uint32_t il = 0; il < n_layer; il++) {
        layer_pool & lp = blk_[il];
        uint32_t off = 0;
        for (int q = 0; q < EXPERT_NPARTS; q++) {
            const byte_range r0 = hot->expert_range(il, 0, (expert_part) q);
            const byte_range r1 = hot->expert_range(il, 1, (expert_part) q);
            if (!r0.valid()) { err = "missing expert tensor on layer " + std::to_string(il); return false; }
            if (r1.valid() && ((r1.offset - r0.offset) % dio_align()) != 0) {
                err = "expert slice stride is not " + std::to_string(dio_align()) + "-byte aligned on layer " + std::to_string(il) +
                      "; the direct I/O layout cannot be used";
                return false;
            }
            const tensor_ref * t = hot->find("blk." + std::to_string(il) + "." +
                (q == EXPERT_GATE ? "ffn_gate_exps.weight" : q == EXPERT_UP ? "ffn_up_exps.weight" : "ffn_down_exps.weight"));
            lp.part_type[q]  = t ? t->type : GGML_TYPE_F32;
            lp.part_off[q]   = off;
            lp.part_pay[q]   = io_hot_.payload_offset(r0.offset);
            lp.part_bytes[q] = r0.nbytes;
            if (cold_) {
                const byte_range cr = cold_->expert_range(il, 0, (expert_part) q);
                const tensor_ref * ct = cold_->find("blk." + std::to_string(il) + "." +
                    (q == EXPERT_GATE ? "ffn_gate_exps.weight" : q == EXPERT_UP ? "ffn_up_exps.weight" : "ffn_down_exps.weight"));
                lp.cold_type[q] = ct ? ct->type : lp.part_type[q];
                lp.cold_pay[q]  = io_cold_.payload_offset(cr.offset);
                if (il == 0 && getenv("QWFN_COLD_DEBUG"))
                    fprintf(stderr, "[cold-debug] layer 0 part %d: hot off %llu pay %u type %d | cold off %llu pad %u pay %u type %d (ct %s) nbytes %u/%u\n",
                            q, (unsigned long long) r0.offset, lp.part_pay[q], (int) lp.part_type[q],
                            (unsigned long long) cr.offset, (unsigned) (cr.offset % dio_align()), lp.cold_pay[q], (int) lp.cold_type[q], ct ? "found" : "MISSING",
                            cr.nbytes, r0.nbytes);
            }
            // A slot holds a block from either file: size it for the larger.
            size_t part_slot = dio_padded_size(r0.offset, r0.nbytes);
            if (cold_) part_slot = std::max(part_slot, (size_t) dio_padded_size(cold_->expert_range(il, 0, (expert_part) q).offset,
                                                                                 cold_->expert_range(il, 0, (expert_part) q).nbytes));
            off += part_slot;
        }
        lp.block_bytes = off;
    }

    // Never take more RAM than the kernel says it can spare. On a 30 GB box
    // sitting behind 30 GB of zram, over-committing here does not fail the
    // allocation -- it OOMs the machine.
    const size_t ram_budget = clamp_to_available(cfg.ram_bytes, cfg.ram_frac, cfg.ram_headroom);
    if (ram_budget < (1ull << 30)) {
        err = "less than 1 GB can be spared for the expert RAM tier (MemAvailable = " +
              std::to_string(mem_available_bytes() >> 20) + " MiB); free memory or lower --ram";
        return false;
    }

    // Equal byte share per layer: traffic per layer is identical (n_expert_used
    // activations each), so bytes -- not slot count -- is the fair unit.
    const size_t per_layer_bytes = ram_budget / n_layer;
    arena_bytes_ = 0;
    for (uint32_t il = 0; il < n_layer; il++) {
        layer_pool & lp = blk_[il];
        lp.n_slots = (uint32_t) std::max<size_t>(1, per_layer_bytes / lp.block_bytes);
        lp.n_slots = std::min<uint32_t>(lp.n_slots, hot->hp().n_expert);
        arena_bytes_ += (size_t) lp.n_slots * lp.block_bytes;
    }

    // Rounding slots up per layer can push the total past the budget; re-check.
    if (arena_bytes_ > ram_budget) {
        const double shrink = (double) ram_budget / (double) arena_bytes_;
        arena_bytes_ = 0;
        for (uint32_t il = 0; il < n_layer; il++) {
            layer_pool & lp = blk_[il];
            lp.n_slots = (uint32_t) std::max<size_t>(1, (size_t) (lp.n_slots * shrink));
            arena_bytes_ += (size_t) lp.n_slots * lp.block_bytes;
        }
    }

    // Pinned when the device offers a host buffer type. The CUDA host type
    // silently falls back to an ordinary CPU buffer if pinning fails, and that
    // one is neither pinned nor page-aligned, so check both before trusting it.
    arena_pinned_ = false;
    if (cfg.host_buft && !getenv("QWFN_PAGEABLE_ARENA")) {
        arena_hostbuf_ = ggml_backend_buft_alloc_buffer(cfg.host_buft, arena_bytes_);
        uint8_t * p = arena_hostbuf_ ? (uint8_t *) ggml_backend_buffer_get_base(arena_hostbuf_) : nullptr;
        if (arena_hostbuf_ && p && ggml_backend_buffer_get_type(arena_hostbuf_) == cfg.host_buft &&
            ((uintptr_t) p % 4096) == 0) {
            arena_ = p;
            arena_pinned_ = true;
        } else {
            if (arena_hostbuf_) ggml_backend_buffer_free(arena_hostbuf_);
            arena_hostbuf_ = nullptr;
            fprintf(stderr, "[qwfn] pinned expert arena unavailable; using pageable memory\n");
        }
    }
    if (!arena_) arena_ = (uint8_t *) dio_alloc(arena_bytes_);
    if (!arena_) { err = "failed to allocate " + std::to_string(arena_bytes_ >> 20) + " MiB expert arena"; return false; }
    size_t cursor = 0;
    total_slots_ = 0;
    for (uint32_t il = 0; il < n_layer; il++) {
        layer_pool & lp = blk_[il];
        lp.base = arena_ + cursor;
        cursor += (size_t) lp.n_slots * lp.block_bytes;
        lp.slot_expert.assign(lp.n_slots, SLOT_EMPTY);
        lp.slot_valid.assign(lp.n_slots, 0);
        lp.slot_cold.assign(lp.n_slots, 0);
        lp.slot_speculative.assign(lp.n_slots, 0);
        lp.slot_pinned.assign(lp.n_slots, 0);
        lp.slot_freq.assign(lp.n_slots, 0);
        lp.ef.assign(hot->hp().n_expert, 0);
        lp.slot_used.assign(lp.n_slots, 0);
        lp.expert_slot.assign(hot->hp().n_expert, -1);
        lp.seen.assign(hot->hp().n_expert, 0);
        lp.hotw.assign(hot->hp().n_expert, 0);
        total_slots_ += lp.n_slots;
    }
    // ---- T0: mirror the same block layout in device memory -----------------
    if (cfg.vram_bytes > 0 && cfg.vram_buft) {
        size_t want = 0;
        std::vector<uint32_t> gslots(n_layer);
        std::vector<size_t>   nat(n_layer, 0);    // natural (unpadded) bytes per expert
        for (uint32_t il = 0; il < n_layer; il++) {
            for (int q = 0; q < EXPERT_NPARTS; q++) {
                blk_[il].g_part_bytes[q] = hot->expert_range(il, 0, (expert_part) q).nbytes;
                nat[il] += blk_[il].g_part_bytes[q];
            }
        }
        // Every layer's slot arrays end with TIER_PAD zeroed bytes. CUDA's MMQ
        // (the batched mul_mat_id kernel) over-reads the last row of a
        // quantised matrix by up to MATRIX_ROW_PADDING elements; ffn_down is
        // 640 wide (640 % 512 != 0), so the last slot of a layer's down array
        // reads into whatever follows -- the next layer's gate array, another
        // quant type, NaN scales. The pad is never written and the lent tail is
        // re-zeroed when it comes back. (T=1 uses mmvq, which reads whole rows
        // only; the bug showed on the batched path as NaN logits on the second
        // turn of a chat, once some experts were VRAM-resident.)
        constexpr size_t TIER_PAD = 4096;
        const size_t per_layer_g = cfg.vram_bytes / n_layer;
        for (uint32_t il = 0; il < n_layer; il++) {
            gslots[il] = (uint32_t) std::min<size_t>(hot->hp().n_expert,
                            std::max<size_t>(1, per_layer_g / nat[il]));
            want += (size_t) gslots[il] * nat[il] + TIER_PAD;
        }
        // Tail slack so a kernel that over-reads past the last row of the last
        // slot (CUDA's MMQ does, by MATRIX_ROW_PADDING) stays inside the buffer.
        want += 1ull << 20;
        // The dynamic part: the last layers' arrays, freed while a prefill
        // runs so its staging, batch buffers and arenas can have that memory.
        // The back-off keeps at least that much: a tier is an optimisation,
        // the prefill is not.
        const size_t lend = cfg.lend_bytes;
        if (lend && want < lend + (1ull << 20)) want = lend + (1ull << 20);
        // Back off rather than fail: the tier is an optimisation, and asking
        // for more than the device has left should cost throughput, not the run.
        // Step down in 4% increments, not quarters: coarse steps threw away up
        // to a quarter of the device memory that was actually free, and 8% steps
        // still turned a 0.4 GB draft head into a 0.73 GB loss of tier; every
        // 2.18 MB block that fits is an expert that computes 3.2x faster.
        // Probe once with the reservation included, then release it, so the
        // back-off converges on a size that still leaves room for the graphs.
        if (ggml_backend_buffer_t probe = ggml_backend_buft_alloc_buffer(cfg.vram_buft, cfg.vram_reserve)) {
            ggml_backend_buffer_free(probe);
        }
        ggml_backend_buffer_t fit = nullptr;
        while (want > std::max<size_t>(256ull << 20, lend + (1ull << 20)) &&
               !(fit = ggml_backend_buft_alloc_buffer(cfg.vram_buft, want + cfg.vram_reserve))) {
            want = 1ull << 20;
            for (uint32_t il = 0; il < n_layer; il++) {
                gslots[il] = (uint32_t) std::max<size_t>(1, (size_t) (gslots[il] * 0.96));
                want += (size_t) gslots[il] * nat[il] + TIER_PAD;
            }
            if (lend && want < lend + (1ull << 20)) want = lend + (1ull << 20);
        }
        if (fit) ggml_backend_buffer_free(fit);
        if (!fit) {
            fprintf(stderr, "[qwfn] VRAM tier disabled: no device memory available\n");
        } else {
            // Layout: layers in order; the first layer whose arrays would end
            // past (want - lend) starts the dynamic buffer.
            std::vector<size_t> lbytes(n_layer);
            for (uint32_t il = 0; il < n_layer; il++) lbytes[il] = (size_t) gslots[il] * nat[il] + TIER_PAD;
            uint32_t K = n_layer;
            if (lend) {
                size_t acc = 0;
                for (uint32_t il = 0; il < n_layer; il++) {
                    if (acc + lbytes[il] > want - lend) { K = il; break; }
                    acc += lbytes[il];
                }
            }
            size_t perm_bytes = 1ull << 20, ext_bytes = 0;
            for (uint32_t il = 0; il < K; il++)       perm_bytes += lbytes[il];
            for (uint32_t il = K; il < n_layer; il++) ext_bytes  += lbytes[il];
            if (ext_bytes) ext_bytes += 1ull << 20;
            vram_buf_ = ggml_backend_buft_alloc_buffer(cfg.vram_buft, perm_bytes);
            if (vram_buf_ && ext_bytes) {
                vram_extra_ = ggml_backend_buft_alloc_buffer(cfg.vram_buft, ext_bytes);
                if (!vram_extra_) {   // no split possible: everything permanent, nothing dynamic
                    ggml_backend_buffer_free(vram_buf_);
                    vram_buf_ = ggml_backend_buft_alloc_buffer(cfg.vram_buft, perm_bytes + ext_bytes);
                    K = n_layer; ext_bytes = 0;
                }
            }
            if (!vram_buf_) {
                fprintf(stderr, "[qwfn] VRAM tier disabled: no device memory available\n");
            } else {
                extra_bytes_ = ext_bytes;
                // Every slot must decode to something finite from the start: the
                // mul_mat_id MoE points experts that live elsewhere at slot 0 with
                // weight 0, and 0 * NaN would poison the sum. Zero bytes decode to
                // zero for every quant type in this model.
                ggml_backend_buffer_clear(vram_buf_, 0);
                if (vram_extra_) ggml_backend_buffer_clear(vram_extra_, 0);
                size_t goff_p = 0, goff_e = 0;
                for (uint32_t il = 0; il < n_layer; il++) {
                    layer_pool & lp = blk_[il];
                    lp.g_slots   = gslots[il];
                    lp.g_in_lent = il >= K;
                    lp.g_buf     = lp.g_in_lent ? vram_extra_ : vram_buf_;
                    size_t & goff = lp.g_in_lent ? goff_e : goff_p;
                    lp.g_off = goff;
                    uint8_t * gbase = (uint8_t *) ggml_backend_buffer_get_base(lp.g_buf);
                    for (int q = 0; q < EXPERT_NPARTS; q++) {
                        lp.g_part[q] = gbase + goff;
                        goff += (size_t) lp.g_slots * lp.g_part_bytes[q];
                    }
                    goff += TIER_PAD;
                    lp.g_slot_expert.assign(lp.g_slots, SLOT_EMPTY);
                    lp.g_valid.assign(lp.g_slots, 0);
                    lp.g_cold.assign(lp.g_slots, 0);
                    lp.g_freq.assign(lp.g_slots, 0);
                    lp.g_used.assign(lp.g_slots, 0);
                    lp.g_used_fetch.assign(lp.g_slots, 0);
                    lp.g_expert_slot.assign(hot->hp().n_expert, -1);
                    total_gslots_ += lp.g_slots;
                }
                // One scratch tensor, repointed per copy: ggml_backend_tensor_set
                // dispatches on tensor->buffer, so this is how host bytes reach VRAM.
                ggml_init_params xp{}; xp.mem_size = ggml_tensor_overhead() * 4; xp.no_alloc = true;
                xfer_ctx_ = ggml_init(xp);
                xfer_ = ggml_new_tensor_1d(xfer_ctx_, GGML_TYPE_I8, 1);
                xfer_->buffer = vram_buf_;
                fprintf(stderr, "[qwfn] expert VRAM tier: %.2f GB, %zu blocks (%.1f%%)%s\n",
                        (perm_bytes + ext_bytes) / 1e9, total_gslots_,
                        100.0 * (double) total_gslots_ / (double) (n_layer * hot->hp().n_expert),
                        ext_bytes ? (", of which " + std::to_string(n_layer - K) + " layers (" +
                                     std::to_string(ext_bytes >> 20) + " MiB) are handed to the prefill").c_str() : "");
            }
        }
    }

    arena_buf_ = ggml_backend_cpu_buffer_from_ptr(arena_, arena_bytes_);
    if (!arena_buf_) { err = "failed to wrap the expert arena in a ggml buffer"; return false; }

    fprintf(stderr, "[qwfn] expert RAM tier: %.2f GB arena%s, %zu of %u expert blocks (%.1f%%)\n",
            arena_bytes_ / 1e9, arena_pinned_ ? " (pinned)" : "", total_slots_, n_layer * hot->hp().n_expert,
            100.0 * (double) total_slots_ / (double) (n_layer * hot->hp().n_expert));
    return true;
}

void expert_cache::shutdown() {
    settle_promotions();   // copies still reading the arena must land before it goes
    if (xfer_ctx_) { ggml_free(xfer_ctx_); xfer_ctx_ = nullptr; xfer_ = nullptr; }
    if (vram_buf_) { ggml_backend_buffer_free(vram_buf_); vram_buf_ = nullptr; }
    if (vram_extra_) { ggml_backend_buffer_free(vram_extra_); vram_extra_ = nullptr; }
    extra_bytes_ = 0;
    total_gslots_ = 0;
    if (arena_buf_) { ggml_backend_buffer_free(arena_buf_); arena_buf_ = nullptr; }
    if (arena_hostbuf_) { ggml_backend_buffer_free(arena_hostbuf_); arena_hostbuf_ = nullptr; arena_ = nullptr; }
    else if (arena_) { dio_free(arena_); arena_ = nullptr; }
    arena_pinned_ = false;
    blk_.clear();
    io_hot_.shutdown();
    io_pf_.shutdown();
    io_cold_.shutdown();
    pf_pending_.clear();
    pf_reads_outstanding_ = 0;
    total_slots_ = 0;
    arena_bytes_ = 0;
}

int32_t expert_cache::find_slot(layer_pool & lp, uint32_t expert_id) const {
    const int32_t s = lp.expert_slot[expert_id];
    if (s < 0) return -1;
    if (lp.slot_expert[s] != (uint16_t) expert_id || !lp.slot_valid[s]) return -1;
    return s;
}

int32_t expert_cache::choose_victim(layer_pool & lp) {
    // Prefer a never-used slot, else sampled-LFU over 8 candidates. Sampling
    // keeps eviction O(1) while still tracking the heavy skew in routing.
    //
    // A slot claimed earlier in this same fetch() is still in flight -- its read
    // has been submitted but not reaped, so it holds an expert id with
    // slot_valid == 0. Handing it out again would point two experts at one block
    // and let one of them read the other's weights. That produced fluent but
    // wrong output that varied run to run.
    static thread_local uint64_t rng = 0x243F6A8885A308D3ull;
    auto in_flight = [&](uint32_t s) {
        if (lp.slot_expert[s] != SLOT_EMPTY && !lp.slot_valid[s]) return true;   // read in flight
        // Already handed to the caller during this same fetch(): evicting it now
        // would silently repoint an earlier expert at another expert's weights.
        for (int32_t l : live_) if (l == (int32_t) s) return true;
        // An asynchronous promotion may still be reading it.
        for (const pending_rel & r : pending_release_)
            if (r.slot == (int32_t) s && &blk_[r.layer] == &lp) return true;
        return false;
    };

    int32_t best = -1;
    uint64_t best_score = UINT64_MAX;
    for (uint32_t k = 0; k < cfg_.evict_samples; k++) {
        const uint32_t s = (uint32_t) (xorshift(rng) % lp.n_slots);
        if (lp.slot_pinned[s] || in_flight(s)) continue;
        if (lp.slot_expert[s] == SLOT_EMPTY) return (int32_t) s;
        // lru    : evict the slot untouched longest.
        // lfu    : evict the expert with the fewest lifetime uses (survives eviction).
        // hybrid : frequency, tie-broken by recency, with recency dominating
        //          once counts are close -- a cheap stand-in for W-TinyLFU.
        uint64_t score;
        switch (cfg_.policy) {
            case config::evict_policy::lfu: score = lp.ef[lp.slot_expert[s]]; break;
            case config::evict_policy::hybrid:
                score = (uint64_t) lp.ef[lp.slot_expert[s]] * 4 + (lp.slot_used[s] >> 4); break;
            default: score = lp.slot_used[s]; break;
        }
        if (score < best_score) { best_score = score; best = (int32_t) s; }
    }
    if (best < 0) {
        for (uint32_t s = 0; s < lp.n_slots; s++)
            if (!lp.slot_pinned[s] && !in_flight(s)) { best = (int32_t) s; break; }
    }
    return best;
}

void expert_cache::fill_handle(const layer_pool & lp, uint32_t slot, expert_handle & h) const {
    uint8_t * base = lp.base + (size_t) slot * lp.block_bytes;
    const bool cold = lp.slot_cold[slot] != 0;
    for (int q = 0; q < EXPERT_NPARTS; q++) {
        h.part[q] = base + lp.part_off[q] + (cold ? lp.cold_pay[q]  : lp.part_pay[q]);
        h.type[q] = cold ? lp.cold_type[q] : lp.part_type[q];
    }
    h.buffer    = arena_buf_;
    h.on_gpu    = false;
    h.from_cold = cold;
    h.slot      = (int32_t) slot;
}

void expert_cache::fill_gpu_handle(const layer_pool & lp, uint32_t gslot, expert_handle & h) const {
    for (int q = 0; q < EXPERT_NPARTS; q++) {
        h.part[q] = lp.g_part[q] + (size_t) gslot * lp.g_part_bytes[q];
        h.type[q] = lp.part_type[q];
    }
    h.buffer    = lp.g_buf;
    h.on_gpu    = true;
    h.from_cold = false;
    h.slot      = (int32_t) gslot;
    h.late      = false;
}

tier_view expert_cache::gpu_tier(uint32_t layer) const {
    tier_view v;
    if (layer >= blk_.size() || !vram_buf_) return v;
    const layer_pool & lp = blk_[layer];
    if (lp.g_slots == 0 || lp.g_lent) return v;
    for (int q = 0; q < EXPERT_NPARTS; q++) {
        v.part[q]   = lp.g_part[q];
        v.stride[q] = lp.g_part_bytes[q];
        v.type[q]   = lp.part_type[q];
    }
    v.n_slots = lp.g_slots;
    v.buffer  = lp.g_buf;
    return v;
}

tier_view expert_cache::ram_tier(uint32_t layer) const {
    tier_view v;
    if (layer >= blk_.size() || !arena_) return v;
    const layer_pool & lp = blk_[layer];
    for (int q = 0; q < EXPERT_NPARTS; q++) {
        v.part[q]   = lp.base + lp.part_off[q] + lp.part_pay[q];
        v.stride[q] = lp.block_bytes;
        v.type[q]   = lp.part_type[q];
    }
    v.n_slots = lp.n_slots;
    v.buffer  = arena_buf_;
    return v;
}

tier_view expert_cache::ram_tier_cold(uint32_t layer) const {
    tier_view v;
    if (layer >= blk_.size() || !arena_ || !cold_) return v;
    const layer_pool & lp = blk_[layer];
    for (int q = 0; q < EXPERT_NPARTS; q++) {
        v.part[q]   = lp.base + lp.part_off[q] + lp.cold_pay[q];
        v.stride[q] = lp.block_bytes;
        v.type[q]   = lp.cold_type[q];
    }
    v.n_slots = lp.n_slots;
    v.buffer  = arena_buf_;
    return v;
}

expert_cache::census expert_cache::ram_census() const {
    census c;
    for (const layer_pool & lp : blk_) {
        c.slots += lp.n_slots;
        for (uint32_t s = 0; s < lp.n_slots; s++) {
            const auto e = lp.slot_expert[s];
            if (e == SLOT_EMPTY) { c.empty++; continue; }
            if (!lp.slot_valid[s]) c.inflight++;
            if (lp.slot_cold[s]) { c.cold++; if (e < lp.hotw.size() && lp.hotw[e]) c.cold_hotw++; } else c.hot++;
            if (lp.slot_speculative[s]) c.speculative++;
        }
        for (auto h : lp.hotw) c.hotw_marked += h ? 1 : 0;
    }
    return c;
}

bool expert_cache::would_promote(layer_pool & lp, uint32_t expert_id) {
    if (!vram_buf_ || lp.g_slots == 0 || lp.g_lent) return false;
    static const bool vram_lru = getenv("QWFN_VRAM_LRU") != nullptr;
    constexpr uint64_t STALE_FETCHES = 48 * 24;
    static thread_local uint64_t rng = 0x2545F4914F6CDD1Dull;
    uint32_t worst = UINT32_MAX; bool any = false;
    for (uint32_t k = 0; k < 16; k++) {
        const uint32_t s = (uint32_t) (xorshift(rng) % lp.g_slots);
        if (lp.g_slot_expert[s] == SLOT_EMPTY) return true;
        if (lp.g_used[s] >= fetch_epoch_) continue;
        if (vram_lru) return true;
        if (fetch_count_ - lp.g_used_fetch[s] > STALE_FETCHES) return true;
        worst = std::min(worst, lp.ef[lp.g_slot_expert[s]]); any = true;
    }
    return any && lp.ef[expert_id] > worst;
}

bool expert_cache::promote(layer_pool & lp, uint32_t expert_id, const uint8_t * host_block) {
    if (!vram_buf_ || lp.g_slots == 0 || lp.g_lent) return false;
    // A block from the cold checkpoint has other quant types and sizes; the
    // per-part device arrays hold one type per part, so it stays on the CPU.
    {
        const int32_t rs = lp.expert_slot[expert_id];
        if (rs >= 0 && lp.slot_cold[rs]) return false;
    }

    // Victim: of 16 sampled slots, the lowest lifetime frequency, and only if
    // the incoming expert is more frequent -- otherwise the tier thrashes.
    // Measured against recency (QWFN_VRAM_LRU=1): recency evicts 96 blocks a
    // token and the hot set cycles out to disk (hit 96.1% -> 94.4%, 40% more
    // reads, 20 -> 16 tok/s). What was wrong with frequency was only that a
    // long prefill's counts froze the tier on the prompt's experts; that is
    // handled by capping the warm-up's bump and halving every counter when a
    // prefill ends (lend_end), not by changing the rule.
    // Hybrid, measured both ways on their own: pure recency churns the hot
    // set out to disk at short context (96.1% -> 94.4% hit), pure frequency
    // keeps a long prompt's experts in the tier after it (VRAM-served 59-65%
    // against 80%). So: a resident unused for STALE_FETCHES fetches (24
    // decoded tokens) is replaceable regardless of its count, the oldest of
    // them first; otherwise the least frequent, and only by a more frequent
    // newcomer. QWFN_VRAM_LRU=1 is pure recency.
    static const bool vram_lru = getenv("QWFN_VRAM_LRU") != nullptr;
    constexpr uint64_t STALE_FETCHES = 48 * 24;
    static thread_local uint64_t rng = 0x853C49E6748FEA9Bull;
    int32_t  victim = -1, stale = -1;
    uint64_t oldest = UINT64_MAX, stale_oldest = UINT64_MAX;
    uint32_t worst  = UINT32_MAX;
    for (uint32_t k = 0; k < 16; k++) {
        const uint32_t s = (uint32_t) (xorshift(rng) % lp.g_slots);
        if (lp.g_slot_expert[s] == SLOT_EMPTY) { victim = (int32_t) s; stale = -1; break; }
        if (lp.g_used[s] >= fetch_epoch_) continue;              // this token's
        if (vram_lru) { if (lp.g_used[s] < oldest) { oldest = lp.g_used[s]; victim = (int32_t) s; } continue; }
        if (fetch_count_ - lp.g_used_fetch[s] > STALE_FETCHES && lp.g_used_fetch[s] < stale_oldest) {
            stale_oldest = lp.g_used_fetch[s]; stale = (int32_t) s;
        }
        const uint32_t f = lp.ef[lp.g_slot_expert[s]];
        if (f < worst) { worst = f; victim = (int32_t) s; }
    }
    if (stale >= 0) victim = stale;
    if (victim < 0) return false;
    const uint32_t incoming = lp.ef[expert_id];
    if (!vram_lru && stale < 0 && lp.g_slot_expert[victim] != SLOT_EMPTY && worst >= incoming) return false;

    if (lp.g_slot_expert[victim] != SLOT_EMPTY) lp.g_expert_slot[lp.g_slot_expert[victim]] = -1;

    // Asynchronous when the arena is pinned. The earlier attempt at this
    // measured no gain because the arena was pageable: cudaMemcpyAsync from
    // pageable memory is staged through the driver and blocks the caller
    // regardless. From pinned memory it is a DMA the caller does not wait for.
    // The lifetime hazard that attempt had -- the RAM slot being released and
    // refilled by a disk read while the copy still read it -- is closed by
    // deferring the release to settle_promotions(), after a stream sync.
    last_promote_async_ = cfg_.async_promote && cfg_.vram_backend && arena_pinned_;
    xfer_->buffer = lp.g_buf;
    for (int q = 0; q < EXPERT_NPARTS; q++) {
        xfer_->data  = lp.g_part[q] + (size_t) victim * lp.g_part_bytes[q];
        xfer_->ne[0] = lp.g_part_bytes[q];
        xfer_->nb[1] = xfer_->nb[2] = xfer_->nb[3] = lp.g_part_bytes[q];
        const uint8_t * src = host_block + lp.part_off[q] + lp.part_pay[q];
        if (last_promote_async_) ggml_backend_tensor_set_async(cfg_.vram_backend, xfer_, src, 0, lp.g_part_bytes[q]);
        else                     ggml_backend_tensor_set(xfer_, src, 0, lp.g_part_bytes[q]);
    }

    lp.g_slot_expert[victim]  = (uint16_t) expert_id;
    lp.g_valid[victim]        = 1;
    lp.g_cold[victim]         = 0;
    lp.g_freq[victim]         = incoming;   // kept for stats only
    lp.g_used[victim]         = tick_;
    lp.g_used_fetch[victim]   = fetch_count_;
    lp.g_expert_slot[expert_id] = victim;
    lp.g_ver++;
    st_.promotions++;
    return true;
}

bool expert_cache::fetch(uint32_t layer, const uint32_t * expert_ids, uint32_t n, expert_handle * out) {
    std::vector<char> ready(n);
    if (!fetch_begin(layer, expert_ids, n, out, (bool *) ready.data())) return false;
    return fetch_end();
}

bool expert_cache::fetch_begin(uint32_t layer, const uint32_t * expert_ids, uint32_t n,
                               expert_handle * out, bool * ready) {
    if (layer >= blk_.size()) return false;
    const auto t_enter = std::chrono::steady_clock::now();
    // Nothing is waited for here. A speculative read for one of these experts
    // that has not landed yet is reported as not ready and settled in
    // fetch_end(), after the demand reads have been submitted, so the caller
    // can compute the experts it already has while both kinds of read land.
    // In-flight slots are never chosen as victims.
    layer_pool & lp = blk_[layer];
    tick_++;
    fetch_epoch_ = tick_;
    fetch_count_++;
    live_.clear();
    pending_.clear();
    // Mark every VRAM-resident expert of this token as in use BEFORE any
    // promotion runs: a promotion picks a recency victim, and an expert later
    // in this same list still carried last token's tick, so it could be
    // evicted moments before it was looked up -- a disk read if it had no RAM
    // copy. Measured: 94.4% -> hit rate against 96.1% with the old rule.
    if (vram_buf_ && lp.g_slots > 0 && !lp.g_lent)
        for (uint32_t i = 0; i < n; i++) {
            const uint32_t e = expert_ids[i];
            if (e >= lp.g_expert_slot.size()) continue;
            const int32_t gs = lp.g_expert_slot[e];
            if (gs >= 0 && lp.g_valid[gs] && lp.g_slot_expert[gs] == (uint16_t) e) { lp.g_used[gs] = tick_; lp.g_used_fetch[gs] = fetch_count_; }
        }

    // The same for the RAM tier. A resident block of an expert later in this
    // list still carries last token's recency and, for a tail expert, a low
    // count, so a miss earlier in the list could pick it as the victim: a
    // prefetched block evicted moments before it was looked up, and the miss
    // that causes evicts another. Listing them as live protects them.
    for (uint32_t i = 0; i < n; i++) {
        const uint32_t e = expert_ids[i];
        if (e >= lp.expert_slot.size()) continue;
        const int32_t s = lp.expert_slot[e];
        if (s >= 0 && lp.slot_expert[s] == (uint16_t) e) live_.push_back(s);
    }

    // Up to n_expert_used experts x 3 parts in flight for this layer.
    io_request reqs[64 * EXPERT_NPARTS];
    bool       req_cold[64 * EXPERT_NPARTS];
    uint32_t   miss_slot[64];
    uint32_t   miss_expert[64];
    size_t     n_req = 0, n_miss = 0;

    uint32_t promoted = 0;
    const double promote_base = st_.t_promote;

    for (uint32_t i = 0; i < n; i++) {
        const uint32_t e = expert_ids[i];
        st_.lookups++;
        lp.ef[e]++;
        if (++ef_ticks_ >= cfg_.age_every) {
            // Halve every counter so old popularity decays; ordering is
            // preserved, and it keeps an early-hot expert from pinning a slot.
            ef_ticks_ = 0;
            for (auto & lay : blk_) for (auto & f : lay.ef) f >>= 1;
        }

        // T0: already in VRAM, so this expert's matmul runs on the GPU.
        if (vram_buf_ && lp.g_slots > 0 && !lp.g_lent) {
            const int32_t gs = lp.g_expert_slot[e];
            if (gs >= 0 && lp.g_valid[gs] && lp.g_slot_expert[gs] == (uint16_t) e) {
                st_.hits++; st_.gpu_hits++;
                lp.g_freq[gs]++;
                lp.g_used[gs] = tick_;
                lp.g_used_fetch[gs] = fetch_count_;
                fill_gpu_handle(lp, (uint32_t) gs, out[i]);
                ready[i] = true;
                continue;
            }
        }

        // Claimed by a speculative read that has not landed: a hit, waited for in
        // fetch_end. A cold block in flight counts too (it used to fall through to
        // the miss path, which read the expert again into a second slot and left the
        // landed block an orphan: with most prefetches cold, a third of them were
        // wasted that way and the RAM tier lost the slots they held -- hit 97.7 ->
        // 88%). If it is a VRAM candidate it is served cold this once and re-read
        // hot at its next reuse.
        {
            const int32_t cs = lp.expert_slot[e];
            if (cs >= 0 && lp.slot_expert[cs] == (uint16_t) e && !lp.slot_valid[cs]) {
                if (lp.slot_cold[cs] && !lp.hotw[e] && would_promote(lp, e)) lp.hotw[e] = 1;
                st_.hits++;
                lp.slot_freq[cs]++;
                lp.slot_used[cs] = ++tick_;
                if (lp.slot_speculative[cs]) { st_.pf_used++; lp.slot_speculative[cs] = 0; }
                live_.push_back(cs);
                fill_handle(lp, (uint32_t) cs, out[i]);
                ready[i] = false;
                pending_.push_back(pending_spec{ e, cs });
                continue;
            }
        }
        const int32_t s = find_slot(lp, e);

        // A block from the cold file is served as it is, unless this expert would
        // now earn a VRAM slot: then it is re-read at full precision on this fetch
        // so the promotion can follow. A cold block cannot be promoted, and the RAM
        // tier's frequency rule would keep it stuck there at CPU speed, holding a
        // slot that VRAM should hold (measured without the re-read: hit 97.7 ->
        // 89.8%, VRAM-served 64 -> 52%). The prefetch and the miss path read the
        // candidates hot by the same rule, so this re-read is left to the blocks
        // whose standing changed between the read and the reuse. (Re-reading every
        // reused cold block, when nothing was read hot yet, doubled the fill:
        // 12 -> 6 tok/s.)
        const bool upgrade = s >= 0 && lp.slot_cold[s] && would_promote(lp, e);
        if (upgrade) lp.hotw[e] = 1;

        if (s >= 0 && !upgrade) {
            st_.hits++;
            lp.slot_freq[s]++;
            lp.slot_used[s] = ++tick_;
            if (lp.slot_speculative[s]) { st_.pf_used++; lp.slot_speculative[s] = 0; }
            live_.push_back(s);
            fill_handle(lp, (uint32_t) s, out[i]);
            ready[i] = true;
            if (promoted < cfg_.max_promotions_per_layer) {
                const auto tp = std::chrono::steady_clock::now();
                if (promote(lp, e, slot_ptr(lp, (uint32_t) s))) {
                    promoted++;
                    // Hand back the VRAM copy and release the RAM slot. Keeping
                    // both meant the tiers overlapped instead of adding: a
                    // lookup checks VRAM first, so the RAM copy was dead weight
                    // occupying capacity. Freeing it turns RAM+VRAM into one
                    // larger cache. The block itself stays intact for this
                    // token -- nothing can be admitted into the slot until this
                    // layer is fetched again.
                    const int32_t gs = lp.g_expert_slot[e];
                    if (gs >= 0) {
                        fill_gpu_handle(lp, (uint32_t) gs, out[i]);
                        out[i].late = true;
                        st_.gpu_hits++;
                        // The copy may still be reading the slot: keep it
                        // protected until the next stream sync. It stays
                        // resident after that as a fallback (see
                        // settle_promotions).
                        if (last_promote_async_) pending_release_.push_back(pending_rel{ layer, e, s });
                    }
                }
                st_.t_promote += std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - tp).count();
            }
            continue;
        }
        st_.misses++;
        if (upgrade) st_.upgrades++;

        // A miss reads the hot file when the expert would earn a VRAM slot now, by
        // the rule the prefetch applies. Without this a miss during the fill came
        // in cold, was reused while the tier still had empty slots (no re-read
        // then) and sat in RAM unpromotable for the rest of the run.
        if (cold_ && !upgrade && !lp.hotw[e] && would_promote(lp, e)) lp.hotw[e] = 1;
        const bool take_cold = cold_ != nullptr && !upgrade && !lp.hotw[e];
        lp.seen[e] = 1;
        const model_index * src = take_cold ? cold_ : hot_;
        if (take_cold) st_.cold_tier_reads++;

        const int32_t v = upgrade ? s : choose_victim(lp);
        if (v < 0) return false;
        if (!upgrade && lp.slot_expert[v] != SLOT_EMPTY) {
            if (lp.slot_speculative[v]) st_.pf_wasted++;   // prefetched, never used, evicted by a miss
            lp.expert_slot[lp.slot_expert[v]] = -1;
            st_.evictions++;
        }
        lp.slot_expert[v]  = (uint16_t) e;
        lp.slot_valid[v]   = 0;
        lp.slot_freq[v]    = 1;
        lp.slot_used[v]    = ++tick_;
        lp.expert_slot[e]  = v;
        lp.slot_cold[v]    = take_cold ? 1 : 0;
        lp.slot_speculative[v] = 0;
        if (upgrade && e < lp.g_expert_slot.size()) {
            const int32_t gs = lp.g_expert_slot[e];   // discard the stale cold VRAM copy
            if (gs >= 0) { lp.g_slot_expert[gs] = SLOT_EMPTY; lp.g_valid[gs] = 0; lp.g_expert_slot[e] = -1; lp.g_ver++; }
        }
        live_.push_back(v);

        uint8_t * base = slot_ptr(lp, (uint32_t) v);
        for (int q = 0; q < EXPERT_NPARTS; q++) {
            const byte_range br = src->expert_range(layer, e, (expert_part) q);
            if (!br.valid()) return false;
            req_cold[n_req] = take_cold;
            reqs[n_req++] = io_request{ br.shard, br.offset, br.nbytes,
                                        base + lp.part_off[q], (uint64_t) n_miss };
            st_.bytes_from_disk += br.nbytes;
        }
        fill_handle(lp, (uint32_t) v, out[i]);
        miss_slot[n_miss]   = (uint32_t) v;
        miss_expert[n_miss] = e;
        ready[i] = false;
        n_miss++;
    }

    inflight_reqs_  = 0;
    inflight_layer_ = layer;
    inflight_slots_.clear();
    inflight_experts_.clear();
    if (n_req == 0) {
        st_.t_submit += std::chrono::duration<double>(
                std::chrono::steady_clock::now() - t_enter).count() - (st_.t_promote - promote_base);
        return true;
    }

    // Submit and return. The device fills these while the caller computes the
    // experts that were already resident. Hot and cold blocks live in different
    // files: each request goes to its file's engine, and fetch_end drains both.
    // (One engine for all of them read hot misses out of the cold shards.)
    io_request hot_reqs[64 * EXPERT_NPARTS], cold_reqs[64 * EXPERT_NPARTS];
    size_t n_hot = 0, n_cold = 0;
    for (size_t k = 0; k < n_req; k++) (req_cold[k] ? cold_reqs[n_cold++] : hot_reqs[n_hot++]) = reqs[k];
    auto submit_all = [&](io_engine & eng, const io_request * rq, size_t n, size_t & inflight) {
        size_t submitted = 0, reaped = 0; uint64_t tags[256];
        while (submitted < n) {
            const size_t k = eng.submit(rq + submitted, n - submitted);
            if (k == 0) {   // ring full: drain one, then keep going
                const size_t got = eng.reap(tags, 256, 1);
                if (got == 0) return false;
                reaped += got; continue;
            }
            submitted += k;
        }
        inflight = n - std::min(n, reaped);
        return true;
    };
    if (!submit_all(io_hot_, hot_reqs, n_hot, inflight_reqs_)) return false;
    if (n_cold && !submit_all(io_cold_, cold_reqs, n_cold, inflight_cold_reqs_)) return false;
    st_.n_bursts++;
    st_.n_reads += n_req;
    st_.t_submit += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t_enter).count() - st_.t_promote + promote_base;
    for (size_t k = 0; k < n_miss; k++) {
        inflight_slots_.push_back((int32_t) miss_slot[k]);
        inflight_experts_.push_back(miss_expert[k]);
    }
    return true;
}

bool expert_cache::read_block_now(layer_pool & lp, uint32_t layer, int32_t slot, uint32_t expert) {
    io_request reqs[EXPERT_NPARTS];
    uint8_t * base = slot_ptr(lp, (uint32_t) slot);
    for (int q = 0; q < EXPERT_NPARTS; q++) {
        const byte_range br = hot_->expert_range(layer, expert, (expert_part) q);
        if (!br.valid()) return false;
        reqs[q] = io_request{ br.shard, br.offset, br.nbytes, base + lp.part_off[q], 0 };
        st_.bytes_from_disk += br.nbytes;
    }
    size_t submitted = 0; uint64_t tags[16]; size_t reaped = 0;
    while (submitted < EXPERT_NPARTS) {
        const size_t k = io_hot_.submit(reqs + submitted, EXPERT_NPARTS - submitted);
        if (k == 0) { const size_t got = io_hot_.reap(tags, 16, 1); if (got == 0) return false; reaped += got; continue; }
        submitted += k;
    }
    while (reaped < EXPERT_NPARTS) { const size_t got = io_hot_.reap(tags, 16, EXPERT_NPARTS - reaped); if (got == 0) return false; reaped += got; }
    st_.n_reads += EXPERT_NPARTS;
    lp.slot_valid[slot] = 1;
    lp.slot_cold[slot]  = 0;
    return true;
}

bool expert_cache::fetch_batch(uint32_t layer, const uint32_t * expert_ids, uint32_t n, expert_handle * out,
                               uint8_t * bounce, size_t bounce_bytes) {
    if (layer >= blk_.size()) return false;
    layer_pool & lp = blk_[layer];
    if (!bounce || bounce_bytes < (size_t) n * lp.block_bytes) return false;
    std::vector<io_request> reqs; reqs.reserve((size_t) n * EXPERT_NPARTS);
    uint32_t n_miss = 0;
    for (uint32_t i = 0; i < n; i++) {
        const uint32_t e = expert_ids[i];
        out[i] = expert_handle{};
        if (e >= hot_->hp().n_expert) return false;
        if (vram_buf_ && lp.g_slots > 0 && !lp.g_lent) {
            const int32_t gs = lp.g_expert_slot[e];
            if (gs >= 0 && lp.g_valid[gs] && lp.g_slot_expert[gs] == (uint16_t) e) { fill_gpu_handle(lp, (uint32_t) gs, out[i]); continue; }
        }
        {
            const int32_t s = lp.expert_slot[e];
            if (s >= 0 && lp.slot_expert[s] == (uint16_t) e && lp.slot_valid[s] && !lp.slot_cold[s]) { fill_handle(lp, (uint32_t) s, out[i]); continue; }
        }
        // An empty RAM slot takes the block (a fresh session's tier fills with the
        // prompt's experts, as it should); a full tier is left alone and the block
        // goes to the bounce. Never an eviction, never a frequency count.
        int32_t adopt = -1;
        {
            static thread_local uint64_t rng = 0x9E3779B97F4A7C15ull;
            for (int k = 0; k < 16 && lp.n_slots > 0; k++) {
                const uint32_t s = (uint32_t) (xorshift(rng) % lp.n_slots);
                if (lp.slot_expert[s] == SLOT_EMPTY && !lp.slot_pinned[s]) { adopt = (int32_t) s; break; }
            }
        }
        uint8_t * base = adopt >= 0 ? slot_ptr(lp, (uint32_t) adopt) : bounce + (size_t) n_miss * lp.block_bytes;
        if (adopt < 0) n_miss++;
        for (int q = 0; q < EXPERT_NPARTS; q++) {
            const byte_range br = hot_->expert_range(layer, e, (expert_part) q);
            if (!br.valid()) return false;
            reqs.push_back(io_request{ br.shard, br.offset, br.nbytes, base + lp.part_off[q], 0 });
            out[i].part[q] = base + lp.part_off[q] + lp.part_pay[q];
            out[i].type[q] = lp.part_type[q];
            st_.bytes_from_disk += br.nbytes;
        }
        out[i].buffer = adopt >= 0 ? arena_buf_ : nullptr; out[i].on_gpu = false; out[i].from_cold = false; out[i].slot = adopt;
        if (adopt >= 0) {
            lp.slot_expert[adopt] = (uint16_t) e; lp.slot_valid[adopt] = 1; lp.slot_freq[adopt] = 1;
            lp.slot_used[adopt] = ++tick_; lp.slot_cold[adopt] = 0; lp.slot_speculative[adopt] = 0;
            lp.expert_slot[e] = adopt;
        }
    }
    st_.batch_lookups += n; st_.batch_misses += n_miss;
    // Every miss's reads, a window at a time.
    size_t submitted = 0, reaped = 0; uint64_t tags[256];
    while (submitted < reqs.size() || reaped < reqs.size()) {
        if (submitted < reqs.size()) {
            const size_t k = io_hot_.submit(reqs.data() + submitted, std::min<size_t>(64, reqs.size() - submitted));
            submitted += k;
            if (k > 0 && submitted < reqs.size()) continue;
        }
        const size_t got = io_hot_.reap(tags, 256, reaped < submitted ? 1 : 0);
        if (got == 0 && reaped < submitted) return false;
        reaped += got;
        if (submitted == reqs.size() && reaped == reqs.size()) break;
    }
    st_.n_reads += reqs.size();
    return true;
}

bool expert_cache::settle_pending(const uint32_t * expert_ids, uint32_t n, bool * ready) {
    if (pending_.empty()) return true;
    const auto tw = std::chrono::steady_clock::now();
    layer_pool & lp = blk_[inflight_layer_];
    std::vector<uint32_t> ids; ids.reserve(pending_.size());
    for (const pending_spec & p : pending_) ids.push_back(p.expert);
    wait_state_ = 2;
    prefetch_settle_for(inflight_layer_, ids.data(), (uint32_t) ids.size());
    wait_state_ = 0;
    for (const pending_spec & p : pending_) {
        // The speculative engine lost track of this read (its count drifted and
        // was reset): fetch the block now rather than compute on stale bytes.
        if (lp.slot_expert[p.slot] != (uint16_t) p.expert) return false;
        if (!lp.slot_valid[p.slot] && !read_block_now(lp, inflight_layer_, p.slot, p.expert)) return false;
        if (ready) for (uint32_t i = 0; i < n; i++) if (expert_ids[i] == p.expert) ready[i] = true;
    }
    pending_.clear();
    st_.t_wait += std::chrono::duration<double>(std::chrono::steady_clock::now() - tw).count();
    return true;
}

bool expert_cache::fetch_end() {
    if (!settle_pending(nullptr, 0, nullptr)) return false;
    if (inflight_reqs_ == 0 && inflight_cold_reqs_ == 0) return true;
    const auto tw = std::chrono::steady_clock::now();

    auto drain = [&](io_engine & eng, size_t n) {
        uint64_t tags[256]; size_t done = 0;
        while (done < n) { const size_t got = eng.reap(tags, 256, n - done); if (got == 0) return false; done += got; }
        return true;
    };
    wait_state_ = 1;
    const bool ok = drain(io_hot_, inflight_reqs_) && drain(io_cold_, inflight_cold_reqs_);
    wait_state_ = 0;
    if (!ok) {
        fprintf(stderr, "[qwfn] expert read wait returned short (%zu hot + %zu cold expected): the I/O engine reports nothing in flight\n",
                inflight_reqs_, inflight_cold_reqs_);
    }
    inflight_reqs_ = inflight_cold_reqs_ = 0;
    if (!ok) return false;
    layer_pool & lp = blk_[inflight_layer_];
    for (size_t k = 0; k < inflight_slots_.size(); k++) {
        const int32_t v = inflight_slots_[k];
        if (lp.slot_expert[v] == (uint16_t) inflight_experts_[k]) lp.slot_valid[v] = 1;
    }
    st_.t_wait += std::chrono::duration<double>(std::chrono::steady_clock::now() - tw).count();
    st_.bytes_read = io_hot_.stat_bytes + io_cold_.stat_bytes;
    return true;
}

void expert_cache::prefetch(uint32_t layer, const uint32_t * expert_ids, uint32_t n) {
    if (layer >= blk_.size()) return;
    layer_pool & lp = blk_[layer];
    io_request reqs[64 * EXPERT_NPARTS];
    std::vector<int32_t> claimed;
    size_t n_req = 0;
    for (uint32_t i = 0; i < n && n_req + EXPERT_NPARTS <= sizeof(reqs) / sizeof(reqs[0]); i++) {
        const uint32_t e = expert_ids[i];
        claimed.push_back(-1);
        if (find_slot(lp, e) >= 0) continue;
        const int32_t v = choose_victim(lp);
        if (v < 0) continue;
        if (lp.slot_expert[v] != SLOT_EMPTY) lp.expert_slot[lp.slot_expert[v]] = -1;
        claimed.back() = v;
        lp.slot_expert[v] = (uint16_t) e;
        lp.slot_valid[v]  = 0;   // becomes valid only once the read is reaped
        lp.slot_freq[v]   = 1;
        lp.expert_slot[e] = v;
        uint8_t * base = slot_ptr(lp, (uint32_t) v);
        for (int q = 0; q < EXPERT_NPARTS; q++) {   // prefetches always take full precision
            const byte_range br = hot_->expert_range(layer, e, (expert_part) q);
            if (!br.valid()) continue;
            reqs[n_req++] = io_request{ br.shard, br.offset, br.nbytes, base + lp.part_off[q], 0 };
        }
    }
    if (!n_req) return;

    // Submit and reap here. True fire-and-forget prefetch needs a persistent
    // in-flight table so a later fetch() can wait on a specific block; until
    // that exists, marking a slot resident before its read lands would be the
    // same aliasing hazard choose_victim guards against.
    io_engine * eng = &io_hot_;
    size_t sub = 0;
    uint64_t tags[256];
    while (sub < n_req) {
        const size_t k = eng->submit(reqs + sub, n_req - sub);
        if (k == 0) { if (eng->reap(tags, 256, 1) == 0) return; continue; }
        sub += k;
    }
    size_t done = 0;
    while (done < n_req) {
        const size_t got = eng->reap(tags, 256, n_req - done);
        if (got == 0) return;
        done += got;
    }
    for (uint32_t i = 0; i < n && i < claimed.size(); i++) {
        const int32_t v = claimed[i];
        if (v >= 0) lp.slot_valid[v] = 1;
    }
}

void expert_cache::settle_promotions() {
    if (pending_release_.empty()) return;
    if (cfg_.vram_backend) ggml_backend_synchronize(cfg_.vram_backend);
    // The RAM copy stays. With recency-based VRAM eviction a promoted expert
    // can leave VRAM again soon; if its RAM copy is gone that is a disk read
    // (measured: hit rate 96.1% -> 94.4%, 40% more reads). Lookups check VRAM
    // first, so the RAM copy is never touched again and recency retires it
    // on its own; while it lives it is a fallback. Only the eviction
    // protection is lifted here.
    pending_release_.clear();
}

void expert_cache::lend_begin() {
    if (!vram_extra_) return;
    settle_promotions();   // copies still landing in those slots must finish first
    for (layer_pool & lp : blk_) {
        if (!lp.g_in_lent) continue;
        lp.g_lent = true;
        std::fill(lp.g_slot_expert.begin(), lp.g_slot_expert.end(), SLOT_EMPTY);
        std::fill(lp.g_valid.begin(), lp.g_valid.end(), 0);
        std::fill(lp.g_expert_slot.begin(), lp.g_expert_slot.end(), -1);
        lp.g_ver++;
    }
    ggml_backend_buffer_free(vram_extra_);
    vram_extra_ = nullptr;
    tier_epoch_++;
}

void expert_cache::lend_end() {
    if (!extra_bytes_ || vram_extra_) return;
    // A prefill just bumped the prompt's experts' counters (capped, but over
    // many ubatches); halve everything so the generation that follows can
    // displace them as its own routing settles.
    for (layer_pool & lp : blk_) for (auto & f : lp.ef) f >>= 1;
    vram_extra_ = ggml_backend_buft_alloc_buffer(cfg_.vram_buft, extra_bytes_);
    if (!vram_extra_) {
        static bool warned = false;
        if (!warned) { warned = true; fprintf(stderr, "[qwfn] could not take the dynamic VRAM tier back after the prefill; those layers stay CPU-served until the next one\n"); }
        return;
    }
    // Zeroed: slot 0 of every layer is the zero-weight dummy of the mul_mat_id
    // MoE, and stale bytes decoded as this layer's type can be NaN scales.
    ggml_backend_buffer_clear(vram_extra_, 0);
    uint8_t * gbase = (uint8_t *) ggml_backend_buffer_get_base(vram_extra_);
    for (layer_pool & lp : blk_) {
        if (!lp.g_in_lent) continue;
        lp.g_buf = vram_extra_;
        size_t goff = lp.g_off;
        for (int q = 0; q < EXPERT_NPARTS; q++) {
            lp.g_part[q] = gbase + goff;
            goff += (size_t) lp.g_slots * lp.g_part_bytes[q];
        }
        lp.g_lent = false;
    }
    tier_epoch_++;
    // Those layers come back with empty VRAM tiers. Refill them from what the
    // RAM tier holds -- the prefill's warm-up just put the prompt's experts
    // there -- hottest first, up to the tier's size. Asynchronous copies from
    // the pinned arena; the RAM slots are protected until the next settle.
    for (uint32_t il = 0; il < blk_.size(); il++) {
        layer_pool & lp = blk_[il];
        if (!lp.g_in_lent || lp.g_slots == 0) continue;
        std::vector<std::pair<uint32_t, uint32_t>> cand;   // (ef, expert)
        for (uint32_t s = 0; s < lp.n_slots; s++)
            if (lp.slot_valid[s] && lp.slot_expert[s] != SLOT_EMPTY && !lp.slot_cold[s])
                cand.emplace_back(lp.ef[lp.slot_expert[s]], lp.slot_expert[s]);
        std::sort(cand.begin(), cand.end(), [](const auto & a, const auto & b) { return a.first > b.first; });
        uint32_t n = 0;
        for (const auto & c : cand) {
            if (n >= lp.g_slots) break;
            const uint32_t e = c.second;
            const int32_t  s = lp.expert_slot[e];
            if (s < 0 || !lp.slot_valid[s] || lp.slot_expert[s] != (uint16_t) e) continue;
            if (!promote(lp, e, slot_ptr(lp, (uint32_t) s))) continue;
            n++;
            st_.warm_promoted++;
            if (last_promote_async_) pending_release_.push_back(pending_rel{ il, e, s });
        }
    }
}

void expert_cache::warm(uint32_t layer, const warm_item * items, uint32_t n, uint32_t n_vram) {
    if (layer >= blk_.size() || n == 0) return;
    const auto t0 = std::chrono::steady_clock::now();
    layer_pool & lp = blk_[layer];
    live_.clear();

    struct copy { uint8_t * dst; const uint8_t * src; size_t n; };
    std::vector<copy>     copies;
    std::vector<int32_t>  admitted;      // slots filled by this call
    std::vector<uint32_t> to_promote;

    for (uint32_t i = 0; i < n; i++) {
        const uint32_t e = items[i].expert;
        if (e >= lp.ef.size()) continue;
        // Capped: a prompt's raw use counts (hundreds over a long prefill)
        // would outrank anything decode admits afterwards -- promote() only
        // displaces a colder resident -- and the VRAM tier then freezes on the
        // prompt's experts. Measured after a 32K prefill: 34% VRAM-served.
        lp.ef[e]  += std::min<uint32_t>(items[i].count, 4);
        lp.seen[e] = 1;

        if (lp.g_slots > 0) {
            const int32_t gs = lp.g_expert_slot[e];
            if (gs >= 0 && lp.g_valid[gs] && lp.g_slot_expert[gs] == (uint16_t) e) {
                lp.g_freq[gs] += items[i].count;
                continue;                                    // already where it is most useful
            }
        }
        int32_t s = find_slot(lp, e);
        if (s >= 0 && lp.slot_cold[s]) {                     // replace a cold copy outright
            lp.expert_slot[e] = -1; lp.slot_expert[s] = SLOT_EMPTY; lp.slot_valid[s] = 0;
            s = -1;
        }
        if (s < 0) {
            s = choose_victim(lp);
            if (s < 0) continue;
            if (lp.slot_expert[s] != SLOT_EMPTY) {
                if (lp.slot_speculative[s]) st_.pf_wasted++;
                lp.expert_slot[lp.slot_expert[s]] = -1;
                st_.evictions++;
            }
            lp.slot_expert[s]      = (uint16_t) e;
            lp.slot_valid[s]       = 0;                      // valid once the copy is done
            lp.slot_cold[s]        = 0;
            lp.slot_speculative[s] = 0;
            lp.slot_freq[s]        = 1;
            lp.expert_slot[e]      = s;
            uint8_t * base = slot_ptr(lp, (uint32_t) s);
            for (int q = 0; q < EXPERT_NPARTS; q++)
                if (items[i].part[q])
                    copies.push_back(copy{ base + lp.part_off[q] + lp.part_pay[q], items[i].part[q], lp.part_bytes[q] });
            admitted.push_back(s);
        } else {
            lp.slot_freq[s]++;
        }
        lp.slot_used[s] = ++tick_;
        live_.push_back(s);
        if (i < n_vram) to_promote.push_back(e);
    }

    // 48 x ~100 MB of memcpy per ubatch is a second on one core; spread it.
    if (!copies.empty()) {
        const unsigned nt = std::min<unsigned>(8, std::max<unsigned>(1, std::thread::hardware_concurrency() / 2));
        if (copies.size() < 6 || nt <= 1) {
            for (const copy & c : copies) memcpy(c.dst, c.src, c.n);
        } else {
            std::vector<std::thread> th;
            for (unsigned k = 0; k < nt; k++)
                th.emplace_back([&, k] { for (size_t i = k; i < copies.size(); i += nt) memcpy(copies[i].dst, copies[i].src, copies[i].n); });
            for (auto & t : th) t.join();
        }
    }
    for (int32_t s : admitted) lp.slot_valid[s] = 1;
    st_.warm_admitted += admitted.size();

    for (uint32_t e : to_promote) {
        const int32_t s = lp.expert_slot[e];
        if (s < 0 || !lp.slot_valid[s] || lp.slot_expert[s] != (uint16_t) e) continue;
        if (!promote(lp, e, slot_ptr(lp, (uint32_t) s))) continue;
        st_.warm_promoted++;
        if (last_promote_async_) pending_release_.push_back(pending_rel{ layer, e, s });
    }
    live_.clear();
    st_.t_warm += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

void expert_cache::pin(const std::vector<uint32_t> & packed_keys) {
    for (uint32_t k : packed_keys) {
        const uint32_t l = k >> 16, e = k & 0xFFFF;
        if (l >= blk_.size() || e >= hot_->hp().n_expert) continue;
        const int32_t s = blk_[l].expert_slot[e];
        if (s >= 0) blk_[l].slot_pinned[s] = 1;
    }
}


// Drain completions off the speculative engine, marking each block valid the
// moment its last read has landed. Tags index pf_pending_.
void expert_cache::pf_reap(size_t min_complete) {
    if (pf_reads_outstanding_ == 0) return;
    uint64_t tags[256];
    size_t need = std::min(min_complete, pf_reads_outstanding_);
    do {
        wait_state_ = need > 0 ? 2 : wait_state_;
        const size_t got = io_pf_.reap(tags, 256, need);
        wait_state_ = 0;
        if (got == 0) {
            if (need > 0) {
                // Asked for a completion and got none: the outstanding count has
                // drifted past what the engine holds. Resynchronise rather than
                // wait for a read that is not there.
                fprintf(stderr, "[qwfn] speculative read count drifted (%zu outstanding, engine idle); resetting\n", pf_reads_outstanding_);
                pf_reads_outstanding_ = 0;
            }
            break;
        }
        for (size_t k = 0; k < got; k++) {
            pf_entry & e = pf_pending_[tags[k]];
            if (e.remaining > 0 && --e.remaining == 0) {
                layer_pool & lp = blk_[e.layer];
                if (lp.slot_expert[e.slot] == (uint16_t) e.expert) lp.slot_valid[e.slot] = 1;
            }
        }
        pf_reads_outstanding_ -= std::min(got, pf_reads_outstanding_);
        need = need > got ? need - got : 0;
    } while (need > 0 && pf_reads_outstanding_ > 0);
}

bool expert_cache::pf_layer_pending(uint32_t layer) const {
    for (const pf_entry & e : pf_pending_)
        if (e.layer == layer && e.remaining > 0) return true;
    return false;
}

// Wait only for the entries targeting `layer` -- the layer whose fetch is about
// to run. Later layers' entries keep flying and go valid whenever they land.
void expert_cache::prefetch_settle_layer(uint32_t layer) {
    while (pf_reads_outstanding_ > 0 && pf_layer_pending(layer)) pf_reap(1);
    if (pf_reads_outstanding_ == 0) pf_pending_.clear();
}

void expert_cache::prefetch_settle_for(uint32_t layer, const uint32_t * ids, uint32_t n) {
    auto needed_pending = [&]() {
        for (const pf_entry & e : pf_pending_) {
            if (e.layer != layer || e.remaining == 0) continue;
            for (uint32_t i = 0; i < n; i++) if (ids[i] == e.expert) return true;
        }
        return false;
    };
    while (pf_reads_outstanding_ > 0 && needed_pending()) pf_reap(1);
    if (pf_reads_outstanding_ == 0) pf_pending_.clear();
}

uint64_t expert_cache::vram_version(uint32_t layer) const {
    return layer < blk_.size() ? blk_[layer].g_ver : 0;
}

void expert_cache::vram_table(uint32_t layer, int32_t * slot, float * mask) const {
    const uint32_t n_expert = hot_->hp().n_expert;
    std::fill(slot, slot + n_expert, 0);
    std::fill(mask, mask + n_expert, 0.0f);
    if (layer >= blk_.size()) return;
    const layer_pool & lp = blk_[layer];
    if (!vram_buf_ || lp.g_slots == 0 || lp.g_lent) return;
    for (uint32_t e = 0; e < n_expert && e < lp.g_expert_slot.size(); e++) {
        const int32_t gs = lp.g_expert_slot[e];
        if (gs >= 0 && lp.g_valid[gs] && lp.g_slot_expert[gs] == (uint16_t) e) { slot[e] = gs; mask[e] = 1.0f; }
    }
}

void expert_cache::prefetch_settle() {
    while (pf_reads_outstanding_ > 0) pf_reap(1);
    pf_pending_.clear();
}

void expert_cache::prefetch_begin(const pf_set * sets, uint32_t n_sets) {
    pf_reap(0);                                   // opportunistic drain
    if (pf_reads_outstanding_ == 0) pf_pending_.clear();
    // The table is tag-addressed and append-only while reads are in flight;
    // this bound should be unreachable (two sets of <= n_expert_used each).
    if (pf_pending_.size() > 4096) prefetch_settle();

    io_request reqs[128 * EXPERT_NPARTS];
    size_t n_req = 0;

    for (uint32_t s = 0; s < n_sets; s++) {
        const uint32_t layer = sets[s].layer;
        if (layer >= blk_.size()) continue;
        layer_pool & lp = blk_[layer];

        for (uint32_t i = 0; i < sets[s].n && n_req + EXPERT_NPARTS <= sizeof(reqs)/sizeof(reqs[0]); i++) {
            const uint32_t e = sets[s].ids[i];
            if (e >= hot_->hp().n_expert) continue;
            if (lp.g_slots > 0 && lp.g_expert_slot[e] >= 0) continue;   // already in VRAM
            // Resident, being fetched, or already inbound from an earlier
            // prediction (the L+1 and L+2 sets for one layer overlap heavily,
            // and a slot claimed by an in-flight read is not yet `valid`, so
            // find_slot alone would re-claim and re-read it).
            const int32_t cs = lp.expert_slot[e];
            if (cs >= 0 && lp.slot_expert[cs] == (uint16_t) e) continue;

            const int32_t v = choose_victim(lp);
            if (v < 0) continue;
            if (lp.slot_expert[v] != SLOT_EMPTY) {
                // A speculative block that never got used is pure waste; count it.
                if (lp.slot_speculative[v]) st_.pf_wasted++;
                lp.expert_slot[lp.slot_expert[v]] = -1;
                st_.evictions++;
            }
            lp.slot_expert[v]      = (uint16_t) e;
            lp.slot_valid[v]       = 0;      // valid once its reads land
            lp.slot_freq[v]        = 1;
            lp.slot_used[v]        = ++tick_;
            lp.slot_speculative[v] = 1;
            lp.expert_slot[e]      = v;
            lp.seen[e]             = 1;

            const uint64_t tag = (uint64_t) pf_pending_.size();
            pf_pending_.push_back(pf_entry{ layer, e, v, 0 });
            pf_entry & ent = pf_pending_.back();

            uint8_t * base = slot_ptr(lp, (uint32_t) v);
            if (cold_ && !lp.hotw[e] && would_promote(lp, e)) lp.hotw[e] = 1;   // a VRAM candidate is read at full precision
            const bool pf_cold = cold_ != nullptr && !lp.hotw[e];   // the tail comes from the cold file
            lp.slot_cold[v] = pf_cold ? 1 : 0;
            if (pf_cold) st_.cold_tier_reads++;
            for (int q = 0; q < EXPERT_NPARTS; q++) {
                const byte_range br = (pf_cold ? cold_ : hot_)->expert_range(layer, e, (expert_part) q);
                if (!br.valid()) continue;
                reqs[n_req++] = io_request{ br.shard + (pf_cold ? (int) n_hot_shards_ : 0), br.offset, br.nbytes, base + lp.part_off[q], tag };
                ent.remaining++;
                st_.bytes_from_disk += br.nbytes;
            }
            if (ent.remaining == 0) {   // nothing to read: mark it now
                if (lp.slot_expert[v] == (uint16_t) e) lp.slot_valid[v] = 1;
                pf_pending_.pop_back();
                continue;
            }
            st_.pf_issued++;
        }
    }

    if (n_req == 0) return;

    // Account each read as it is accepted, so pf_reap can make progress if the
    // ring ever fills mid-submit.
    size_t sub = 0;
    while (sub < n_req) {
        const size_t k = io_pf_.submit(reqs + sub, n_req - sub);
        sub += k;
        pf_reads_outstanding_ += k;
        if (k == 0) pf_reap(1);
    }
}

} // namespace qwfn
