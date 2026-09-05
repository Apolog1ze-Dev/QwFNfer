#pragma once
// PLE: the 51.2B-parameter n-gram embedding table.
//
//   per_layer_token_embd.weight = [160, 320001536] IQ4_NL = 28.80 GB
//   16 heads x ~20,000,0xx rows (each head's row count is a distinct prime)
//   one row = 160 IQ4_NL values = 90 bytes
//
// Per generated token the model reads exactly ple_n_head (16) rows -- 1,440
// bytes of payload scattered over 28.8 GB. That is ~16 random reads/token,
// which is why this table belongs on NVMe and not in RAM: at 15 tok/s it is
// ~240 IOPS, against a device that does 150,000. Keeping it on disk frees
// 28.8 GB of the 30 GB machine for what actually needs bandwidth: experts.
//
// Prefill is the case that needs care. A 128K-token prompt performs 2.1M row
// lookups; issued naively as demand faults that is ~30 s. Because every input
// token id is known up front, the whole set of rows can be computed, sorted by
// file offset and deduplicated, turning random I/O into a near-sequential
// sweep.

#include "qwfn_model.h"
#include "qwfn_io.h"

#include <cstdint>
#include <unordered_map>
#include <vector>

namespace qwfn {

static constexpr uint32_t QWFN_MAX_PLE_NGRAM = 8;

// Row indices touched by one token position.
struct ple_rows {
    uint64_t row[32];   // ple_n_head() entries are filled
    uint32_t n = 0;
};

// Row indices for the n-gram window ending at position i.
//
//   for n in 2..ngram_size:
//     mixed = t[0]*m[0] ^ t[1]*m[1] ^ ... ^ t[n-1]*m[n-1]
//     for the heads_per_ngram heads of order n:  row_h = mixed % vocab[h] + offset[h]
//
// XOR, not sum; one hash per order, shared across that order's heads, which are
// decorrelated only by their distinct prime moduli. EOS resets the window and a
// predecessor before the sequence start reads as EOS.
ple_rows ple_rows_for(const hparams & hp, const int32_t * tokens, int64_t n_tokens, int64_t i);

class ple_table {
public:
    bool init(const model_index * mi, io_engine * io, size_t row_cache_bytes, std::string & err);

    // Rows for the n-gram window ending at position i of `tokens`.
    //
    //   for n in 2..ngram_size:
    //     mixed = t[0]*m[0] ^ t[1]*m[1] ^ ... ^ t[n-1]*m[n-1]
    //     for the heads_per_ngram heads of order n:
    //       row_h = mixed % vocab[h] + offset[h]
    //
    // XOR, not sum; one hash per order shared across that order's 8 heads.
    // EOS resets the window. `tokens` must contain position i and its
    // predecessors, so the caller passes the sequence so far.
    ple_rows rows_for(const int32_t * tokens, int64_t n_tokens, int64_t i) const;

    // Decode path: fetch the rows for one position. Returns payload pointers.
    // Issues reads through io_engine and blocks until they land.
    bool gather(const ple_rows & rows, const uint8_t ** out_rows);

    // Prefill path: fetch rows for many positions at once, sorted and
    // deduplicated by file offset. Results are placed in the row cache.
    bool gather_batch(const std::vector<ple_rows> & batch);

    uint32_t row_bytes() const { return row_bytes_; }
    uint64_t n_rows()    const { return n_rows_; }

    uint64_t stat_hits = 0, stat_misses = 0, stat_dedup_saved = 0;

private:
    const uint8_t * cached(uint64_t row) const;
    uint8_t *       admit(uint64_t row);

    const model_index * mi_ = nullptr;
    io_engine *         io_ = nullptr;

    uint32_t row_bytes_  = 0;   // 90 for IQ4_NL x 160
    uint32_t slot_bytes_ = 0;   // row_bytes_ padded for O_DIRECT
    uint64_t n_rows_     = 0;

    // Small direct-mapped cache of hot rows. Common trigrams are heavily
    // Zipfian, so even a few hundred MB removes most repeat lookups.
    std::vector<uint8_t>                     pool_;
    std::vector<uint64_t>                    slot_row_;   // which row occupies each slot
    std::vector<uint8_t>                     slot_valid_;
    size_t                                   n_slots_ = 0;
    std::unordered_map<uint64_t, uint32_t>   index_;      // row -> slot
};

} // namespace qwfn
