#include "qwfn_weights.h"

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <vector>

namespace qwfn {

weights::~weights() {
    for (auto b : map_buf_) if (b) ggml_backend_buffer_free(b);
    for (size_t i = 0; i < map_base_.size(); i++)
        if (map_base_[i]) munmap(map_base_[i], map_size_[i]);
    if (buf_)     ggml_backend_buffer_free(buf_);
    if (ctx_)     ggml_free(ctx_);
    if (backend_) ggml_backend_free(backend_);
}

void weights::set_n_threads(int n) {
    if (!backend_ || !dev_) return;
    auto fn = (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(
            ggml_backend_dev_backend_reg(dev_), "ggml_backend_set_n_threads");
    if (fn) fn(backend_, n);
}

const char * weights::dev_name() const {
    return dev_ ? ggml_backend_dev_name(dev_) : "none";
}

bool weights::init(const model_index * mi, bool prefer_gpu,
                   const std::string & backend_dir, std::string & err) {
    mi_ = mi;

    if (backend_dir.empty()) ggml_backend_load_all();
    else                     ggml_backend_load_all_from_path(backend_dir.c_str());

    if (prefer_gpu) {
        dev_ = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
        on_gpu_ = dev_ != nullptr;
    }
    if (!dev_) {
        dev_ = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        on_gpu_ = false;
    }
    if (!dev_) { err = "no ggml backend device available"; return false; }

    backend_ = ggml_backend_dev_init(dev_, nullptr);
    if (!backend_) { err = "failed to init backend device"; return false; }
    buft_ = ggml_backend_dev_buffer_type(dev_);

    // no_alloc: tensors are declared first, then backed by one buffer in commit().
    // Headroom for ~1500 tensor descriptors.
    ggml_init_params ip{};
    ip.mem_size   = ggml_tensor_overhead() * 4096;
    ip.mem_buffer = nullptr;
    ip.no_alloc   = true;
    ctx_ = ggml_init(ip);
    if (!ctx_) { err = "ggml_init failed"; return false; }
    return true;
}

ggml_tensor * weights::declare(const std::string & name) {
    auto it = by_name_.find(name);
    if (it != by_name_.end()) return it->second;

    const tensor_ref * ref = mi_->find(name);
    if (!ref) return nullptr;

    // GGUF stores ne[] already in ggml order, so this mirrors the file exactly.
    ggml_tensor * t = ggml_new_tensor_4d(ctx_, ref->type, ref->ne[0], ref->ne[1], ref->ne[2], ref->ne[3]);
    if (!t) return nullptr;
    ggml_set_name(t, name.c_str());

    by_name_[name] = t;
    pending_.emplace_back(t, ref);
    declared_bytes_ += ref->nbytes;
    return t;
}

bool weights::declare_dense_core(std::string & err) {
    return declare_dense_core(err, nullptr);
}

bool weights::declare_dense_core(std::string & err,
                                 const std::function<bool(const std::string &)> & accept) {
    for (const auto & kv : mi_->tensors()) {
        const std::string & n = kv.first;
        // Routed experts are streamed by expert_cache; the PLE table lives on NVMe.
        if (n.find("_exps.weight") != std::string::npos) continue;
        if (n == "per_layer_token_embd.weight")          continue;
        if (n == "token_embd.weight")                    continue;   // host mapping; rows are gathered there
        if (accept && !accept(n))                        continue;
        if (!declare(n)) { err = "failed to declare " + n; return false; }
    }
    return true;
}

bool weights::commit(std::string & err) {
    buf_ = ggml_backend_alloc_ctx_tensors_from_buft(ctx_, buft_);
    if (!buf_) {
        err = "failed to allocate " + std::to_string(declared_bytes_ >> 20) + " MiB on " + dev_name();
        return false;
    }

    // One fd per shard, plain buffered reads: this is a single 5.35 GB pass at load
    // time, not a hot path, and the page cache warming here is harmless.
    std::vector<int> fds;
    for (const auto & p : mi_->shard_paths()) {
        int fd = ::open(p.c_str(), O_RDONLY);
        if (fd < 0) {
            for (int f : fds) ::close(f);
            err = "open failed: " + p + ": " + strerror(errno);
            return false;
        }
        fds.push_back(fd);
    }

    std::vector<uint8_t> staging;
    bool ok = true;
    for (auto & [t, ref] : pending_) {
        staging.resize(ref->nbytes);
        size_t done = 0;
        while (done < ref->nbytes) {
            const ssize_t n = ::pread(fds[ref->shard], staging.data() + done,
                                      ref->nbytes - done, (off_t) (ref->file_offset + done));
            if (n <= 0) { err = "short read on " + ref->name; ok = false; break; }
            done += (size_t) n;
        }
        if (!ok) break;
        ggml_backend_tensor_set(t, staging.data(), 0, ref->nbytes);
    }

    for (int f : fds) ::close(f);
    return ok;
}

ggml_tensor * weights::get(const std::string & name) const {
    auto it = by_name_.find(name);
    return it == by_name_.end() ? nullptr : it->second;
}


bool weights::map_shards(std::string & err) {
    for (const auto & p : mi_->shard_paths()) {
        int fd = ::open(p.c_str(), O_RDONLY);
        if (fd < 0) { err = "open failed: " + p; return false; }
        const off_t sz = ::lseek(fd, 0, SEEK_END);
        void * base = ::mmap(nullptr, (size_t) sz, PROT_READ, MAP_PRIVATE, fd, 0);
        ::close(fd);
        if (base == MAP_FAILED) { err = "mmap failed: " + p; return false; }

        // Expert access is scattered by the router; sequential readahead would
        // only evict pages we still want.
        ::madvise(base, (size_t) sz, MADV_RANDOM);

        map_base_.push_back(base);
        map_size_.push_back((size_t) sz);
        map_buf_.push_back(ggml_backend_cpu_buffer_from_ptr(base, (size_t) sz));
        mapped_bytes_ += (size_t) sz;
    }
    return true;
}

ggml_tensor * weights::declare_mapped(const std::string & name) {
    auto it = by_name_.find(name);
    if (it != by_name_.end()) return it->second;

    const tensor_ref * ref = mi_->find(name);
    if (!ref) return nullptr;
    if (ref->shard < 0 || (size_t) ref->shard >= map_base_.size()) return nullptr;

    ggml_tensor * t = ggml_new_tensor_4d(ctx_, ref->type, ref->ne[0], ref->ne[1], ref->ne[2], ref->ne[3]);
    if (!t) return nullptr;
    ggml_set_name(t, name.c_str());

    // Point the tensor at its bytes inside the mapping. No copy, no allocation.
    t->buffer = map_buf_[ref->shard];
    t->data   = (char *) map_base_[ref->shard] + ref->file_offset;

    by_name_[name] = t;
    declared_bytes_ += ref->nbytes;
    return t;
}

bool weights::declare_all_mapped(std::string & err) {
    if (map_base_.empty() && !map_shards(err)) return false;
    for (const auto & kv : mi_->tensors()) {
        if (!declare_mapped(kv.first)) { err = "failed to map " + kv.first; return false; }
    }
    return true;
}

} // namespace qwfn
