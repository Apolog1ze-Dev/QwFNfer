#include "qwfn_ple.h"

#include <algorithm>
#include <cstring>

namespace qwfn {

bool ple_table::init(const model_index * mi, io_engine * io, size_t row_cache_bytes, std::string & err) {
    mi_ = mi;
    io_ = io;
    const tensor_ref * t = mi->ple_tensor();
    if (!t) { err = "per_layer_token_embd.weight not present in checkpoint"; return false; }

    row_bytes_  = (uint32_t) ggml_row_size(t->type, t->ne[0]);
    n_rows_     = (uint64_t) t->ne[1];
    // Each cached row must be able to absorb O_DIRECT read-around slack.
    slot_bytes_ = (uint32_t) dio_align_up(row_bytes_ + dio_align());

    n_slots_ = row_cache_bytes / slot_bytes_;
    if (n_slots_ == 0) n_slots_ = 1;
    if (pool_) dio_free(pool_);
    pool_bytes_ = n_slots_ * (size_t) slot_bytes_;
    pool_ = (uint8_t *) dio_alloc(pool_bytes_);
    if (!pool_) { err = "ple row cache allocation failed"; return false; }
    memset(pool_, 0, pool_bytes_);
    slot_row_.assign(n_slots_, UINT64_MAX);
    slot_valid_.assign(n_slots_, 0);
    index_.reserve(n_slots_ * 2);
    return true;
}

ple_rows ple_rows_for(const hparams & hp, const int32_t * tokens, int64_t n_tokens, int64_t i) {
    const uint32_t n_gram   = hp.ple_ngram_size;          // 3
    const uint32_t per_gram = hp.ple_heads_per_ngram;     // 8
    const uint32_t nh       = hp.ple_n_head();            // (3-1)*8 = 16
    const int32_t  eos      = hp.ple_eos_token_id;        // 248044

    ple_rows out;
    out.n = nh;

    // Build the n-gram window ending at i. An EOS anywhere in the window resets
    // everything at or before it; a predecessor before the sequence start reads
    // as EOS. The token's own EOS does not cut its own context.
    int64_t ctx[QWFN_MAX_PLE_NGRAM];
    ctx[0] = (int64_t) tokens[i];
    bool cut = false;
    for (uint32_t s = 1; s < n_gram; s++) {
        const int64_t pos = i - (int64_t) s;
        const int32_t t = (cut || pos < 0) ? -1 : tokens[pos];
        cut = cut || t < 0 || t == eos;
        ctx[s] = cut ? (int64_t) eos : (int64_t) t;
    }

    // One hash per n-gram order, shared by that order's heads; the heads are
    // decorrelated only by their distinct prime moduli.
    for (uint32_t n = 2; n <= n_gram; n++) {
        uint64_t mixed = (uint64_t) ctx[0] * hp.ple_layer_multipliers[0];
        for (uint32_t j = 1; j < n; j++) {
            mixed ^= (uint64_t) ctx[j] * hp.ple_layer_multipliers[j];
        }
        const uint32_t base = (n - 2) * per_gram;
        for (uint32_t g = 0; g < per_gram; g++) {
            const uint32_t h = base + g;
            out.row[h] = mixed % hp.ple_head_vocab_sizes[h] + hp.ple_head_offsets[h];
        }
    }
    return out;
}

ple_rows ple_table::rows_for(const int32_t * tokens, int64_t n_tokens, int64_t i) const {
    return ple_rows_for(mi_->hp(), tokens, n_tokens, i);
}

const uint8_t * ple_table::cached(uint64_t row) const {
    auto it = index_.find(row);
    if (it == index_.end()) return nullptr;
    const uint32_t s = it->second;
    if (!slot_valid_[s] || slot_row_[s] != row) return nullptr;
    const uint64_t off = mi_->ple_row_range(row).offset;
    return pool_ + (size_t) s * slot_bytes_ + io_->payload_offset(off);
}

uint8_t * ple_table::admit(uint64_t row) {
    // Direct-mapped by row hash; on collision the incumbent is simply replaced.
    const uint32_t s = (uint32_t) ((row * 0x9E3779B97F4A7C15ull) >> 40) % (uint32_t) n_slots_;
    if (slot_valid_[s] && slot_row_[s] != row) index_.erase(slot_row_[s]);
    slot_row_[s]   = row;
    slot_valid_[s] = 0;          // becomes valid once the read completes
    index_[row]    = s;
    return pool_ + (size_t) s * slot_bytes_;
}

bool ple_table::gather(const ple_rows & rows, const uint8_t ** out_rows) {
    io_request reqs[32];
    uint32_t   need[32];
    size_t     n_need = 0;

    for (uint32_t h = 0; h < rows.n; h++) {
        if (const uint8_t * p = cached(rows.row[h])) { out_rows[h] = p; stat_hits++; continue; }
        stat_misses++;
        const byte_range br = mi_->ple_row_range(rows.row[h]);
        uint8_t * slot = admit(rows.row[h]);
        reqs[n_need] = io_request{ br.shard, br.offset, br.nbytes, slot, (uint64_t) h };
        need[n_need] = h;
        out_rows[h]  = slot + io_->payload_offset(br.offset);
        n_need++;
    }
    if (n_need == 0) return true;

    if (io_->submit(reqs, n_need) != n_need) return false;
    uint64_t tags[32];
    size_t done = 0;
    while (done < n_need) {
        const size_t got = io_->reap(tags, 32, n_need - done);
        if (got == 0) return false;
        done += got;
    }
    for (size_t k = 0; k < n_need; k++) {
        auto it = index_.find(rows.row[need[k]]);
        if (it != index_.end()) slot_valid_[it->second] = 1;
    }
    return true;
}

bool ple_table::gather_batch(const std::vector<ple_rows> & batch) {
    // Collect every row the batch will touch, then sort + dedup by offset so the
    // device sees an ascending sweep instead of scattered seeks.
    std::vector<uint64_t> want;
    want.reserve(batch.size() * 16);
    for (const auto & r : batch)
        for (uint32_t h = 0; h < r.n; h++)
            if (!cached(r.row[h])) want.push_back(r.row[h]);

    const size_t before = want.size();
    std::sort(want.begin(), want.end());
    want.erase(std::unique(want.begin(), want.end()), want.end());
    stat_dedup_saved += before - want.size();
    stat_misses      += want.size();

    // Stream them through the ring, keeping it topped up.
    const size_t WINDOW = 192;
    std::vector<io_request> reqs;
    std::vector<uint64_t>   inflight_rows;
    uint64_t tags[256];
    size_t i = 0;

    while (i < want.size() || io_->in_flight() > 0) {
        reqs.clear();
        inflight_rows.clear();
        while (i < want.size() && reqs.size() < WINDOW && io_->in_flight() + reqs.size() < WINDOW) {
            const uint64_t row = want[i++];
            const byte_range br = mi_->ple_row_range(row);
            uint8_t * slot = admit(row);
            reqs.push_back(io_request{ br.shard, br.offset, br.nbytes, slot, row });
            inflight_rows.push_back(row);
        }
        if (!reqs.empty() && io_->submit(reqs.data(), reqs.size()) == 0) return false;

        const size_t want_done = io_->in_flight() > WINDOW / 2 ? io_->in_flight() - WINDOW / 2 : io_->in_flight();
        const size_t got = io_->reap(tags, 256, want_done ? 1 : 0);
        for (size_t k = 0; k < got; k++) {
            auto it = index_.find(tags[k]);
            if (it != index_.end() && slot_row_[it->second] == tags[k]) slot_valid_[it->second] = 1;
        }
        if (got == 0 && reqs.empty() && i >= want.size()) break;
    }
    return true;
}

} // namespace qwfn
