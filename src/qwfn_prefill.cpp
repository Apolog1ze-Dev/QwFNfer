#include "qwfn_prefill.h"

#include <cstdlib>
#include <cstring>
#include <vector>

#include <algorithm>
#include <chrono>
#include <cstdio>

namespace qwfn {

static const char * PART_SUFFIX[EXPERT_NPARTS] = {
    "ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight"
};

// Large enough that the device is streaming rather than paying per-request
// latency; 16 MB measured 7.8-9.7 GB/s against 2.8 at 1 MB.
static constexpr size_t CHUNK = 16ull << 20;

// Enough for MATRIX_ROW_PADDING (512) elements of the widest quantised row,
// per part, with room to spare.
static constexpr size_t PAD_SLACK = 1ull << 20;

// Device layout: each part is its 512 expert slices back to back at the natural
// stride, i.e. exactly the GGUF tensor [n, m, 512], so one ggml_mul_mat_id per
// part covers the whole layer. CUDA's MMQ over-reads past the last row of a
// quantised matrix by up to MATRIX_ROW_PADDING elements; between experts that
// lands in the next expert's bytes (finite, and multiplied by zero-padded
// activations), and past expert 511 it lands in PAD_SLACK, which the per-load
// clear keeps zeroed. Natural slices are 512-byte multiples, so every expert
// stays 512-aligned. (The earlier per-expert padded slots were only ever
// needed for that last-expert tail.)
static size_t dev_slot_bytes(size_t nbytes) { return nbytes; }
// Zeroed after each part on every load. MMQ's over-read is at most
// MATRIX_ROW_PADDING (512) elements of one row: under 1 KiB for every type here.
static constexpr size_t TAIL_CLEAR = 8192;

size_t prefill_streamer::device_bytes_for(const model_index * mi) {
    // Worst-case device layout across all layers, plus slack for the tail clear.
    size_t dev_worst = 0;
    for (uint32_t il = 0; il < mi->hp().n_layer; il++) {
        size_t tot = 0;
        for (int q = 0; q < EXPERT_NPARTS; q++)
            tot += dev_slot_bytes(mi->expert_range(il, 0, (expert_part) q).nbytes)
                 * mi->hp().n_expert;
        dev_worst = std::max(dev_worst, tot);
    }
    return dev_worst + PAD_SLACK;   // PAD_SLACK (1 MiB) >= 3 * TAIL_CLEAR
}

prefill_streamer::~prefill_streamer() {
    if (reader_.joinable()) {
        { std::lock_guard<std::mutex> lk(m_); stop_ = true; }
        cv_.notify_all();
        reader_.join();
    }
    release_device();
    for (auto & b : hb_) {
        if (b.buf)    ggml_backend_buffer_free(b.buf);
        if (b.pinned) ggml_backend_buffer_free(b.pinned);
        else if (b.p) dio_free(b.p);
    }
    io_.shutdown();
}

bool prefill_streamer::init(const model_index * mi, unsigned io_workers, bool io_threads,
                            ggml_backend_buffer_type_t dev_buft, ggml_backend_t dev_backend,
                            bool overlap, std::string & err) {
    mi_ = mi;
    overlap_ = overlap;
    dev_backend_ = dev_buft ? dev_backend : nullptr;
    if (!io_.init(mi->shard_paths(), io_workers ? io_workers : 8, /*direct_io=*/true, err,
                  io_threads ? io_engine::backend::threads : io_engine::backend::uring)) {
        return false;
    }

    // Size the staging for the largest layer: a few layers quantise ffn_down at
    // Q8_0 and are meaningfully bigger than the IQ4_NL majority.
    size_t worst = 0;
    for (uint32_t il = 0; il < mi->hp().n_layer; il++) {
        size_t tot = 0;
        for (int q = 0; q < EXPERT_NPARTS; q++) {
            const byte_range r = mi->expert_range(il, 0, (expert_part) q);
            tot += dio_align_up((size_t) r.nbytes * mi->hp().n_expert + dio_align());
        }
        worst = std::max(worst, tot);
    }
    // CUDA's MMQ kernels read quantized rows in MATRIX_ROW_PADDING (512) element
    // chunks, which is why ggml's CUDA buffer type pads every quantized tensor's
    // allocation. Expert ffn_down is [640, 2560] and 640 % 512 = 128, so the
    // kernel over-reads past the end of each expert slice. Packed back to back
    // that is harmless for experts 0..510 -- it lands in the next expert and is
    // multiplied by zero-padded activations -- but past the LAST expert it ran
    // off the end of the staging into uninitialised device memory, and
    // NaN * 0 = NaN. Hence a handful of NaNs, only once a prompt is long enough
    // to route something to expert 511. Slack + a zeroed buffer fixes it.
    stage_bytes_ = worst + PAD_SLACK;

    dev_bytes_ = device_bytes_for(mi);

    // Pinned host staging when there is a device to upload to. Measured on the
    // reference machine: the upload of one layer runs at 11 GB/s from pageable
    // memory as 512 synchronous copies, 21.9 GB/s as one strided copy, and
    // 26.8 GB/s as one strided copy from pinned memory. cudaMallocHost returns
    // page-aligned memory, so O_DIRECT reads land in it directly. Falls back to
    // a plain aligned allocation if the pinned one is refused or misaligned.
    ggml_backend_buffer_type_t host_buft = nullptr;
    if (dev_buft && !getenv("QWFN_PREFILL_PAGEABLE")) {
        ggml_backend_dev_t dev = ggml_backend_buft_get_device(dev_buft);
        if (dev) host_buft = ggml_backend_dev_host_buffer_type(dev);
    }
    const int n_bufs = overlap_ ? 2 : 1;
    host_pinned_ = host_buft != nullptr;
    for (int i = 0; i < n_bufs; i++) {
        if (host_buft) {
            ggml_backend_buffer_t pb = ggml_backend_buft_alloc_buffer(host_buft, stage_bytes_);
            uint8_t * p = pb ? (uint8_t *) ggml_backend_buffer_get_base(pb) : nullptr;
            // The CUDA host type silently hands back an ordinary CPU buffer when
            // pinning fails; that one is neither pinned nor page-aligned.
            if (pb && p && ggml_backend_buffer_get_type(pb) == host_buft && ((uintptr_t) p % 4096) == 0) {
                hb_[i].pinned = pb;
                hb_[i].p      = p;
            } else {
                if (pb) ggml_backend_buffer_free(pb);
                host_pinned_ = false;
            }
        }
        if (!hb_[i].p) hb_[i].p = (uint8_t *) dio_alloc(stage_bytes_);
        if (!hb_[i].p) {
            err = "failed to allocate " + std::to_string(stage_bytes_ >> 20) + " MiB prefill staging";
            return false;
        }
        memset(hb_[i].p, 0, stage_bytes_);
        hb_[i].buf = ggml_backend_cpu_buffer_from_ptr(hb_[i].p, stage_bytes_);
        if (!hb_[i].buf) { err = "failed to wrap the prefill staging buffer"; return false; }
    }
    dev_buft_ = dev_buft;   // allocated lazily, see ensure_device()
    if (overlap_) reader_ = std::thread(&prefill_streamer::reader_loop, this);

    fprintf(stderr, "[qwfn] prefill streamer: host %d x %.2f GB%s, device %.2f GB (%s), %zu MB chunks%s\n",
            n_bufs, stage_bytes_ / 1e9, host_pinned_ ? " (pinned)" : "", dev_bytes_ / 1e9,
            dev_buft_ ? "device, allocated per prefill" : "host",
            CHUNK >> 20, overlap_ ? ", read-ahead on" : "");
    return true;
}

bool prefill_streamer::ensure_device(std::string & err) {
    if (!dev_buft_ || dev_buf_) return true;
    dev_buf_ = ggml_backend_buft_alloc_buffer(dev_buft_, dev_bytes_);
    if (!dev_buf_) { err = "failed to allocate the prefill device staging (" + std::to_string(dev_bytes_ >> 20) + " MiB)"; return false; }
    dev_base_ = (uint8_t *) ggml_backend_buffer_get_base(dev_buf_);
    // Any over-read now lands on zeros rather than whatever the allocator
    // handed us.
    ggml_backend_buffer_clear(dev_buf_, 0);
    ggml_init_params xp{}; xp.mem_size = ggml_tensor_overhead() * 4; xp.no_alloc = true;
    xfer_ctx_ = ggml_init(xp);
    xfer_ = ggml_new_tensor_1d(xfer_ctx_, GGML_TYPE_I8, (int64_t) dev_bytes_);
    xfer_->buffer = dev_buf_;
    xfer_->data   = dev_base_;
    loaded_ = UINT32_MAX;   // whatever the host holds is not on the device
    return true;
}

void prefill_streamer::release_device() {
    if (xfer_ctx_) { ggml_free(xfer_ctx_); xfer_ctx_ = nullptr; xfer_ = nullptr; }
    if (dev_buf_)  { ggml_backend_buffer_free(dev_buf_); dev_buf_ = nullptr; }
    dev_base_ = nullptr;
    loaded_   = UINT32_MAX;
}

// The actual read, into buffer `b`. Runs on the reader thread when overlap is
// on and on the caller otherwise -- never both at once, so the io engine has a
// single user.
bool prefill_streamer::read_into(hbuf & b, uint32_t layer, std::string & err) {
    const uint32_t n_expert = mi_->hp().n_expert;

    size_t off = 0;
    std::vector<io_request> reqs;
    struct copy_job { uint8_t * dst; const uint8_t * src; size_t n; };
    std::vector<copy_job> copies;
    for (int q = 0; q < EXPERT_NPARTS; q++) {
        const byte_range r0 = mi_->expert_range(layer, 0, (expert_part) q);
        if (!r0.valid()) { err = "missing expert tensor"; return false; }
        const tensor_ref * t = mi_->find("blk." + std::to_string(layer) + "." + PART_SUFFIX[q]);
        b.ptype[q] = t ? t->type : GGML_TYPE_F32;
        b.slice[q] = r0.nbytes;

        // All n_expert slices are contiguous from the first one, so this is one
        // sequential range; O_DIRECT wants the start rounded down.
        const uint64_t begin = dio_align_down(r0.offset);
        const size_t   span  = dio_align_up((r0.offset - begin) + (size_t) r0.nbytes * n_expert);
        b.part_off[q] = off;
        b.part_pad[q] = r0.offset - begin;
        if (getenv("QWFN_CHECK_STRIDE")) {
            // Everything here assumes expert e lives at r0.offset + e*r0.nbytes
            // in the same shard. Verify that against the index for the last
            // expert, which is the one producing NaN.
            const byte_range rl = mi_->expert_range(layer, n_expert - 1, (expert_part) q);
            const uint64_t assumed = r0.offset + (uint64_t) (n_expert - 1) * r0.nbytes;
            fprintf(stderr, "[stride] layer %2u part %d: e0 shard=%d off=%llu nbytes=%u | "
                            "e511 shard=%d off=%llu nbytes=%u | assumed=%llu %s\n",
                    layer, q, r0.shard, (unsigned long long) r0.offset, r0.nbytes,
                    rl.shard, (unsigned long long) rl.offset, rl.nbytes,
                    (unsigned long long) assumed,
                    (rl.offset == assumed && rl.shard == r0.shard && rl.nbytes == r0.nbytes)
                        ? "OK" : "*** MISMATCH ***");
        }

        // Slices the RAM tier holds are copied from it (below, while the reads
        // are in flight) and split the sequential range into runs of missing
        // experts; each run is read in CHUNK pieces from its aligned edges. An
        // aligned edge may re-read the fringe of a held slice: the same bytes.
        std::vector<const uint8_t *> held(n_expert, nullptr);
        for (const ram_slice & r : b.residents) if (r.expert < n_expert) held[r.expert] = r.part[q];
        for (uint32_t e = 0; e < n_expert; ) {
            if (held[e]) {
                copies.push_back(copy_job{ b.p + off + b.part_pad[q] + (size_t) e * r0.nbytes, held[e], r0.nbytes });
                bytes_from_ram += r0.nbytes;
                e++; continue;
            }
            uint32_t e2 = e + 1;
            while (e2 < n_expert && !held[e2]) e2++;
            const uint64_t rs = r0.offset + (uint64_t) e * r0.nbytes, re = r0.offset + (uint64_t) e2 * r0.nbytes;
            const uint64_t a0 = dio_align_down(rs), a1 = std::min<uint64_t>(dio_align_up(re), begin + span);
            for (uint64_t p = a0; p < a1; p += CHUNK) {
                const size_t len = (size_t) std::min<uint64_t>(CHUNK, a1 - p);
                reqs.push_back(io_request{ r0.shard, p, (uint32_t) len, b.p + off + (size_t) (p - begin), 0 });
            }
            bytes_read += a1 - a0;
            e = e2;
        }
        off += span;
    }

    // Keep the device busy: submit a window, drain, refill.
    const size_t WINDOW = 32;
    uint64_t tags[256];
    size_t i = 0, done = 0;
    // `done` counts completions, not bytes: both backends report a short read
    // as a completion and only bump stat_errors. Without this snapshot a
    // truncated chunk leaves the previous layer's experts in that slice of the
    // staging and the MoE computes on them -- fluent, wrong, and unrepeatable.
    const uint64_t err0 = io_.stat_errors;
    // The tier's slices are copied while the first window of reads is in flight.
    size_t ci = 0;
    while (done < reqs.size()) {
        while (i < reqs.size() && io_.in_flight() < WINDOW) {
            const size_t k = io_.submit(&reqs[i], std::min(WINDOW - io_.in_flight(), reqs.size() - i));
            if (k == 0) break;
            i += k;
        }
        if (ci < copies.size()) { const copy_job & c = copies[ci++]; memcpy(c.dst, c.src, c.n); continue; }
        const size_t got = io_.reap(tags, 256, 1);
        if (got == 0) { err = "prefill expert read failed"; return false; }
        done += got;
    }
    for (; ci < copies.size(); ci++) memcpy(copies[ci].dst, copies[ci].src, copies[ci].n);
    if (io_.stat_errors != err0) { err = "prefill expert read short or failed"; return false; }
    return true;
}

void prefill_streamer::reader_loop() {
    for (;;) {
        int32_t layer; int t;
        {
            std::unique_lock<std::mutex> lk(m_);
            cv_.wait(lk, [&] { return stop_ || want_layer_ >= 0; });
            if (stop_) return;
            layer = want_layer_; t = want_buf_;
            want_layer_ = -1;
            reading_ = true;
        }
        std::string e;
        const bool ok = read_into(hb_[t], (uint32_t) layer, e);
        {
            std::lock_guard<std::mutex> lk(m_);
            hb_[t].ready  = ok;
            hb_[t].failed = !ok;
            if (!ok) reader_err_ = e;
            reading_ = false;
        }
        cv_.notify_all();
    }
}

void prefill_streamer::fill_residents(hbuf & b, uint32_t layer, bool from_ram) {
    b.residents.clear();
    if (from_ram && res_src_) res_src_(layer, b.residents);
}

prefill_streamer::hbuf * prefill_streamer::enqueue_locked(uint32_t layer, bool from_ram) {
    // Already staged or already the target of a queued/running read?
    for (auto & b : hb_)
        if (b.layer == (int32_t) layer) return &b;
    // A queued-but-unstarted job is superseded; untag its buffer.
    if (want_layer_ >= 0) hb_[want_buf_].layer = -1;
    // Never target the buffer a read is landing in; otherwise the caller is
    // done with the old front, so either remaining buffer is fair game.
    const int t = reading_ ? 1 - want_buf_ : 1 - front_;
    hb_[t].layer  = (int32_t) layer;
    hb_[t].ready  = false;
    hb_[t].failed = false;
    fill_residents(hb_[t], layer, from_ram);
    want_layer_ = (int32_t) layer;
    want_buf_   = t;
    cv_.notify_all();
    return &hb_[t];
}

void prefill_streamer::prefetch_layer(uint32_t layer, bool from_ram) {
    if (!overlap_) return;
    std::lock_guard<std::mutex> lk(m_);
    if (stop_) return;
    if (hb_[0].layer == (int32_t) layer || hb_[1].layer == (int32_t) layer) return;
    // One speculative job at a time; a layer that is actually needed jumps the
    // queue via load_layer instead.
    if (want_layer_ >= 0 || reading_) return;
    const int t = 1 - front_;
    hb_[t].layer  = (int32_t) layer;
    hb_[t].ready  = false;
    hb_[t].failed = false;
    fill_residents(hb_[t], layer, from_ram);
    want_layer_ = (int32_t) layer;
    want_buf_   = t;
    cv_.notify_all();
}

bool prefill_streamer::load_layer(uint32_t layer, std::string & err) {
    if (!ensure_device(err)) return false;
    if (layer == loaded_) return true;
    const auto t0 = std::chrono::steady_clock::now();

    int idx = 0;
    if (!overlap_) {
        hbuf & b = hb_[0];
        if (b.layer != (int32_t) layer || !b.ready) {
            b.layer = (int32_t) layer;
            b.ready = false;
            fill_residents(b, layer, true);
            if (!read_into(b, layer, err)) { b.layer = -1; return false; }
            b.ready = true;
        }
    } else {
        std::unique_lock<std::mutex> lk(m_);
        hbuf * b = enqueue_locked(layer, true);   // a load runs inside a prefill: the tier is stable
        idx = (int) (b - hb_);
        cv_.wait(lk, [&] { return (b->ready || b->failed) && b->layer == (int32_t) layer; });
        if (b->failed) { err = reader_err_; b->layer = -1; return false; }
        front_ = idx;
    }
    hbuf & b = hb_[idx];
    for (int q = 0; q < EXPERT_NPARTS; q++) {
        part_off_[q]  = b.part_off[q];
        part_pad_[q]  = b.part_pad[q];
        slice_[q]     = b.slice[q];
        part_type_[q] = b.ptype[q];
    }
    t_read += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

    if (dev_buf_) {
        const uint32_t n_expert = mi_->hp().n_expert;
        // Each expert gets its own padded, 128-aligned slot, and the whole buffer
        // is cleared first.
        //
        // CUDA's MMQ kernels over-read past the final partial row of a quantised
        // tensor -- ffn_down is [640, 2560] and 640 % 512 = 128 -- which ggml
        // normally absorbs by padding every tensor's allocation. Aliasing experts
        // back to back gives them no padding, so the read runs into whatever
        // follows. Clearing ONCE at init is not enough, twice over: layers differ
        // in size (the one iq4_xs/q8_0 layer needs 1.78 GB, a plain layer 1.11),
        // and the padded stride differs per layer too, so both the tail and the
        // inter-expert gaps otherwise hold a previous layer's stale q8_0 bytes --
        // which decode as iq4_nl to NaN scales. Clear per load; a 1.8 GB device
        // memset is ~3 ms against a 12 s ubatch.
        const auto tu = std::chrono::steady_clock::now();
        size_t d = 0;
        for (int q = 0; q < EXPERT_NPARTS; q++) {
            dev_slice_[q] = dev_slot_bytes(slice_[q]);
            dev_off_[q]   = d;
            const size_t end = d + (size_t) (n_expert - 1) * dev_slice_[q] + slice_[q];
            if (end > dev_bytes_ || end + TAIL_CLEAR > dev_bytes_) {
                err = "prefill device staging too small: need " + std::to_string(end + TAIL_CLEAR) +
                      ", have " + std::to_string(dev_bytes_);
                return false;
            }
            // Zero the bytes past this part's last expert: MMQ over-reads the
            // final partial row (ffn_down is 640 wide, 640 % 512 = 128) and a
            // smaller layer's tail otherwise holds a bigger layer's stale bytes,
            // which decoded as another type give NaN scales. ggml's tensor
            // memset runs on the driver's per-thread stream and waits for it,
            // so it is complete before the asynchronous upload below is even
            // queued -- unlike a whole-buffer clear, which the compute stream
            // does not wait for. (A whole-buffer clear was the old code; that
            // race showed as a different first token in one run out of ~four.)
            ggml_backend_tensor_memset(xfer_, 0, end, TAIL_CLEAR);
            // One strided copy per part (cudaMemcpy2D underneath) instead of
            // 512 synchronous copies: 11 -> 22 GB/s from pageable memory,
            // 27 GB/s from the pinned staging. Same bytes, same layout.
            //
            // Queued on the DEVICE backend's stream, not the driver's
            // per-thread default stream: ggml's synchronous 2D copy returns
            // before the DMA has landed, and the compute stream does not wait
            // for the default stream, so the MoE could read a half-uploaded
            // layer (seen once in five runs as a different first token). On the
            // backend stream the copy is ordered after the previous MoE and
            // before the next one, and the host does not wait for it either.
            static const bool legacy_upload = getenv("QWFN_LEGACY_UPLOAD") != nullptr;
            const uint8_t * src = b.p + part_off_[q] + part_pad_[q];
            if (legacy_upload || !dev_backend_) {
                for (uint32_t e = 0; e < n_expert; e++)
                    ggml_backend_tensor_set(xfer_, src + (size_t) e * slice_[q],
                            dev_off_[q] + (size_t) e * dev_slice_[q], slice_[q]);
            } else {
                ggml_backend_tensor_set_2d_async(dev_backend_, xfer_, src,
                        dev_off_[q], slice_[q], n_expert, dev_slice_[q], slice_[q]);
            }
            d += dev_slice_[q] * n_expert;
        }
        t_upload += std::chrono::duration<double>(std::chrono::steady_clock::now() - tu).count();
        if (getenv("QWFN_VERIFY_UPLOAD")) {
            // Byte-compare every expert slice in its padded device slot against
            // the host staging. Slow (1536 synchronous reads per layer); a
            // diagnostic, not a mode.
            if (dev_backend_) ggml_backend_synchronize(dev_backend_);
            std::vector<uint8_t> back;
            size_t bad = 0;
            for (int q = 0; q < EXPERT_NPARTS; q++) {
                back.resize(slice_[q]);
                for (uint32_t e = 0; e < n_expert; e++) {
                    ggml_backend_tensor_get(xfer_, back.data(), dev_off_[q] + (size_t) e * dev_slice_[q], slice_[q]);
                    const uint8_t * src = b.p + part_off_[q] + part_pad_[q] + (size_t) e * slice_[q];
                    if (memcmp(back.data(), src, slice_[q]) != 0) bad++;
                }
            }
            fprintf(stderr, "[upload] layer %2u mismatching slices=%zu of %u | slice=%zu/%zu/%zu pad=%zu/%zu/%zu\n",
                    layer, bad, 3 * n_expert, slice_[0], slice_[1], slice_[2],
                    part_pad_[0], part_pad_[1], part_pad_[2]);
            for (int q = 0; q < EXPERT_NPARTS; q++) {
                size_t worst_al = 4096;
                for (uint32_t e = 0; e < 512; e++) {
                    const uintptr_t a = (uintptr_t) part_ptr(e, (expert_part) q);
                    size_t al = 1; while (al < 4096 && (a % (al * 2)) == 0) al *= 2;
                    worst_al = std::min(worst_al, al);
                }
                fprintf(stderr, "          part %d: worst pointer alignment = %zu bytes\n", q, worst_al);
            }
        }
    }

    loaded_ = layer;
    return true;
}

const uint8_t * prefill_streamer::part_ptr(uint32_t e, expert_part part) const {
    if (dev_base_) return dev_base_ + dev_off_[part] + (size_t) e * dev_slice_[part];
    return hb_[front_].p + part_off_[part] + part_pad_[part] + (size_t) e * slice_[part];
}

const uint8_t * prefill_streamer::host_part_ptr(uint32_t e, expert_part part) const {
    return hb_[front_].p + part_off_[part] + part_pad_[part] + (size_t) e * slice_[part];
}

tier_view prefill_streamer::staged_tier() const {
    tier_view v;
    for (int q = 0; q < EXPERT_NPARTS; q++) {
        v.part[q]   = (uint8_t *) part_ptr(0, (expert_part) q);
        v.stride[q] = part_stride((expert_part) q);
        v.type[q]   = part_type_[q];
    }
    v.n_slots = mi_->hp().n_expert;
    v.buffer  = buffer();
    return v;
}

} // namespace qwfn
