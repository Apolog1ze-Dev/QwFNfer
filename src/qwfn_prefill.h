#pragma once
// Bulk expert streamer for prefill.
//
// Prefill is not a caching workload. With T tokens in a micro-batch the distinct
// experts per layer is ~512*(1-(1-10/512)^T) -- at T=256 that is 509 of 512 -- so
// essentially the whole expert tensor is needed, once, and then never again until
// the next micro-batch. Faulting it in 4 KiB at a time through mmap measures
// 0.13 GB/s. Reading the same bytes contiguously in 16 MB chunks measures 7.0-9.7.
//
// So: host staging filled by a handful of large sequential reads, and the MoE
// graph's expert tensors point into it (or into its device mirror). The decode
// cache is left completely alone.
//
// With overlap on there are two host staging buffers and a reader thread: while
// layer L uploads and computes out of one buffer, layer L+1's read streams into
// the other. The read is most of a prefill layer's wall time, so hiding it is
// worth the second buffer -- which is allocated up front, before the expert RAM
// tier sizes itself, so the memory guard accounts for it.

#include "qwfn_expert_cache.h"
#include "qwfn_io.h"
#include "qwfn_model.h"

#include "ggml-backend.h"

#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace qwfn {

class prefill_streamer {
public:
    ~prefill_streamer();
    prefill_streamer() = default;
    prefill_streamer(const prefill_streamer &) = delete;
    prefill_streamer & operator=(const prefill_streamer &) = delete;

    // `dev_buft` is optional. When present the staged experts are uploaded to
    // the device and the MoE runs there: prefill gives every expert ~T*10/512
    // tokens of work, so the matmuls are wide and the GPU wins by orders of
    // magnitude over the CPU dequant path. The upload is ~1.5 GB per layer at
    // ~25 GB/s, far cheaper than the CPU compute it replaces.
    // `overlap` enables the second staging buffer and the reader thread.
    // `dev_backend` (optional, with dev_buft) is the backend whose stream the
    // upload is queued on. Queued there it is ordered by construction: after
    // the previous layer's MoE that read the staging, before the next one that
    // needs it, with no host-side wait at all.
    bool init(const model_index * mi, unsigned io_workers, bool io_threads,
              ggml_backend_buffer_type_t dev_buft, ggml_backend_t dev_backend,
              bool overlap, std::string & err);

    // Device bytes the staging needs for this model (the largest layer + slack).
    static size_t device_bytes_for(const model_index * mi);
    // The device staging is allocated on the first load_layer() after init or
    // release_device(), and release_device() frees it: it only needs to exist
    // while a prefill runs, and the expert VRAM tier has the memory otherwise.
    void release_device();

    // Make `layer` current: wait for its read (started here if it was not
    // prefetched), then mirror it into the device staging if one exists.
    bool load_layer(uint32_t layer, std::string & err);

    // Start reading `layer` into the spare buffer on the reader thread. No-op
    // if it is already staged, already being read, or overlap is off.
    void prefetch_layer(uint32_t layer);

    // Where expert `e`'s part landed. Valid until the next load_layer().
    const uint8_t * part_ptr(uint32_t e, expert_part part) const;
    // The same slice in the HOST staging (for the cache warm-up after a layer).
    const uint8_t * host_part_ptr(uint32_t e, expert_part part) const;
    // Slot-to-slot stride of part `part` where part_ptr points (device or host).
    size_t          part_stride(expert_part part) const { return dev_base_ ? dev_slice_[part] : slice_[part]; }
    // The staged layer as one tier view: [.., .., n_expert] per part.
    tier_view       staged_tier() const;
    ggml_type       part_type(expert_part part) const { return part_type_[part]; }
    // Staging wrapped as a CPU buffer, so graph tensors can alias it.
    ggml_backend_buffer_t buffer() const { return dev_buf_ ? dev_buf_ : hb_[front_].buf; }
    bool                  on_device() const { return dev_buft_ != nullptr; }
    uint32_t        layer_loaded() const { return loaded_; }

    double   t_read = 0;      // caller-visible wait for reads (hidden reads cost ~0)
    double   t_upload = 0;
    uint64_t bytes_read = 0;

private:
    // One host staging buffer plus the layer it holds. The per-part layout
    // depends on the layer (spans and quant types differ), so it lives with
    // the buffer the read filled.
    struct hbuf {
        uint8_t *             p     = nullptr;
        ggml_backend_buffer_t buf   = nullptr;   // ggml wrapper over p
        // When the device has a host (pinned) buffer type, p comes from it so
        // the H2D upload is a real DMA at PCIe speed instead of a staged copy
        // through the driver's bounce buffer. Null when p is a plain dio_alloc.
        ggml_backend_buffer_t pinned = nullptr;
        int32_t               layer = -1;        // what the buffer holds / is receiving
        bool                  ready = false;     // read complete
        bool                  failed = false;
        size_t    part_off[EXPERT_NPARTS] = {0, 0, 0};
        size_t    part_pad[EXPERT_NPARTS] = {0, 0, 0};
        size_t    slice[EXPERT_NPARTS]    = {0, 0, 0};
        ggml_type ptype[EXPERT_NPARTS]    = {GGML_TYPE_F32, GGML_TYPE_F32, GGML_TYPE_F32};
    };

    // The actual read: fills b's data and layout for `layer`. Reader thread
    // only (the io engine is not shared between threads).
    bool read_into(hbuf & b, uint32_t layer, std::string & err);
    void reader_loop();
    // Queue `layer` for the reader and return the buffer it will land in, or
    // nullptr if it is already staged or in flight. Caller holds m_.
    hbuf * enqueue_locked(uint32_t layer);

    const model_index * mi_ = nullptr;
    io_engine           io_;
    size_t    stage_bytes_ = 0;
    hbuf      hb_[2];
    int       front_ = 0;
    bool      overlap_ = false;

    std::thread             reader_;
    std::mutex              m_;
    std::condition_variable cv_;      // reader wakes on work, caller on ready
    int32_t   want_layer_ = -1;       // pending job for the reader
    int       want_buf_   = 0;
    bool      reading_    = false;
    bool      stop_       = false;
    std::string reader_err_;

    // Layout of the *current* front buffer, copied out on load_layer so the
    // hot accessors stay branch-free.
    size_t   part_off_[EXPERT_NPARTS] = {0, 0, 0};
    size_t   part_pad_[EXPERT_NPARTS] = {0, 0, 0};
    size_t   slice_[EXPERT_NPARTS]    = {0, 0, 0};
    // Device layout is NOT the host layout. ggml's CUDA buffer type pads every
    // quantised tensor by MATRIX_ROW_PADDING and aligns to 128; aliasing experts
    // back to back violates both, and the MMQ kernels read past the end of the
    // last expert's last row. Give each expert its own padded, aligned slot.
    size_t   dev_slice_[EXPERT_NPARTS] = {0, 0, 0};
    size_t   dev_off_[EXPERT_NPARTS]   = {0, 0, 0};
    size_t   dev_bytes_ = 0;
    ggml_type part_type_[EXPERT_NPARTS] = {GGML_TYPE_F32, GGML_TYPE_F32, GGML_TYPE_F32};
    ggml_backend_buffer_t dev_buf_ = nullptr;
    ggml_backend_buffer_type_t dev_buft_ = nullptr;
    ggml_backend_t        dev_backend_ = nullptr;
    bool ensure_device(std::string & err);
    uint8_t *             dev_base_ = nullptr;
    ggml_context *        xfer_ctx_ = nullptr;
    ggml_tensor *         xfer_ = nullptr;
    uint32_t loaded_ = UINT32_MAX;
    bool     host_pinned_ = false;
};

} // namespace qwfn
