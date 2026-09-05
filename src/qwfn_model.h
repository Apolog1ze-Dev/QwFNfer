#pragma once
// Model index: parses the GGUF shards of a Qwen3.8-Flash-Next checkpoint and
// records, for every tensor, which file it lives in and at what byte offset.
//
// Nothing here reads weight data. The point is to know exactly where each of
// the 24,576 (layer, expert) blocks lives on disk so the expert cache can
// fetch whole slices with io_uring instead of faulting them in 4KiB at a time.

#include "qwfn_hparams.h"

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "ggml.h"

struct gguf_context;

namespace qwfn {

struct tensor_ref {
    std::string name;
    int         shard       = -1;   // index into model_index::shard_paths()
    uint64_t    file_offset = 0;    // absolute offset within that shard file
    ggml_type   type        = GGML_TYPE_F32;
    int64_t     ne[4]       = {1, 1, 1, 1};
    uint64_t    nbytes      = 0;

    int64_t n_elements() const { return ne[0] * ne[1] * ne[2] * ne[3]; }
};

// A contiguous byte range on disk holding one expert's gate/up/down matrix.
struct byte_range {
    int      shard  = -1;
    uint64_t offset = 0;
    uint32_t nbytes = 0;
    bool valid() const { return shard >= 0 && nbytes > 0; }
};

enum expert_part : int { EXPERT_GATE = 0, EXPERT_UP = 1, EXPERT_DOWN = 2, EXPERT_NPARTS = 3 };

class model_index {
public:
    // path may be any shard; siblings are discovered via the -0000N-of-0000M
    // naming convention and split.count metadata.
    bool load(const std::string & path, std::string & err);

    const hparams & hp() const { return hp_; }
    const std::vector<std::string> & shard_paths() const { return shard_paths_; }

    const tensor_ref * find(const std::string & name) const;
    const std::unordered_map<std::string, tensor_ref> & tensors() const { return tensors_; }

    // Byte range of one expert's weight slice. `part` selects gate/up/down.
    byte_range expert_range(uint32_t layer, uint32_t expert_id, expert_part part) const;

    // Total bytes for one whole expert (gate+up+down) on the given layer.
    uint32_t expert_block_bytes(uint32_t layer) const;

    // Byte range of a single PLE row (d_ple values, IQ4_NL) at absolute row index.
    byte_range ple_row_range(uint64_t row) const;
    const tensor_ref * ple_tensor() const { return ple_; }

    // Aggregate accounting, used by the planner and reported by qwfn-inspect.
    uint64_t bytes_experts() const { return bytes_experts_; }
    uint64_t bytes_ple()     const { return bytes_ple_; }
    uint64_t bytes_dense()   const { return bytes_dense_; }

    std::string arch;   // expected "qwen4exp"

private:
    bool read_metadata(gguf_context * ctx, std::string & err);

    hparams                                     hp_;
    std::vector<std::string>                    shard_paths_;
    std::unordered_map<std::string, tensor_ref> tensors_;

    // Cached lookups for the hot paths, indexed [layer][part].
    std::vector<std::array<const tensor_ref *, EXPERT_NPARTS>> expert_tensors_;
    const tensor_ref * ple_ = nullptr;

    uint64_t bytes_experts_ = 0, bytes_ple_ = 0, bytes_dense_ = 0;
};

} // namespace qwfn
