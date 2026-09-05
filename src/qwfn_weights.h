#pragma once
// Weight residency: declares ggml tensors mirroring the GGUF, allocates them on a
// backend, and uploads their bytes straight from the shard files.
//
// Only the dense core goes through here -- 5.35 GB, which fits in VRAM with room to
// spare. The 55.82 GB of routed experts are owned by expert_cache and never become
// ggml tensors with backend storage; the 28.80 GB PLE table stays on NVMe entirely.

#include "qwfn_model.h"

#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

#include "ggml.h"
#include "ggml-backend.h"

namespace qwfn {

class weights {
public:
    ~weights();
    weights() = default;
    weights(const weights &) = delete;
    weights & operator=(const weights &) = delete;

    // backend_dir is where libggml-cuda.so lives; empty uses the default search.
    bool init(const model_index * mi, bool prefer_gpu,
              const std::string & backend_dir, std::string & err);

    // Phase 1: declare a tensor. Returns nullptr if the name is not in the GGUF.
    ggml_tensor * declare(const std::string & name);

    // Declare everything except the routed experts and the PLE table -- i.e. the
    // dense core that stays resident.
    bool declare_dense_core(std::string & err);

    // Same, but only names for which `accept` returns true.
    bool declare_dense_core(std::string & err, const std::function<bool(const std::string &)> & accept);

    // Phase 2: allocate the buffer and upload every declared tensor.
    bool commit(std::string & err);

    // --- mmap path -------------------------------------------------------
    // Instead of copying weights into a backend buffer, map the shards and point
    // tensors straight into the mapping. Required for the 55.82 GB of routed
    // experts, which cannot be resident: the pages the router actually touches
    // are faulted in, the rest never are. CPU backend only.
    bool map_shards(std::string & err);
    ggml_tensor * declare_mapped(const std::string & name);
    bool declare_all_mapped(std::string & err);

    ggml_tensor * get(const std::string & name) const;

    ggml_context *             ctx()     const { return ctx_; }
    ggml_backend_t             backend() const { return backend_; }
    ggml_backend_buffer_type_t buft()    const { return buft_; }
    bool                       on_gpu()  const { return on_gpu_; }
    size_t                     bytes()   const { return declared_bytes_; }
    const char *               dev_name() const;

    // The CPU backend is a dynamically loaded module, so its thread setter is
    // reached through the registry rather than linked directly.
    void set_n_threads(int n);

private:
    const model_index * mi_ = nullptr;

    ggml_context *             ctx_     = nullptr;
    ggml_backend_t             backend_ = nullptr;
    ggml_backend_dev_t         dev_     = nullptr;
    ggml_backend_buffer_type_t buft_    = nullptr;
    ggml_backend_buffer_t      buf_     = nullptr;
    bool                       on_gpu_  = false;

    std::unordered_map<std::string, ggml_tensor *> by_name_;
    std::vector<std::pair<ggml_tensor *, const tensor_ref *>> pending_;
    size_t declared_bytes_ = 0;

    std::vector<void *>                map_base_;
    std::vector<size_t>                map_size_;
    std::vector<ggml_backend_buffer_t> map_buf_;
    size_t mapped_bytes_ = 0;
};

} // namespace qwfn
