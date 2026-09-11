#include "qwfn_vision.h"

#include "gguf.h"

#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_JPEG
#define STBI_ONLY_PNG
#define STBI_ONLY_BMP
#define STBI_ONLY_GIF
#include "stb_image.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdlib>

namespace qwfn {

// ---- image loading ----------------------------------------------------------

bool image_u8::load(const std::string & path, std::string & err) {
    int w = 0, h = 0, c = 0;
    unsigned char * data = stbi_load(path.c_str(), &w, &h, &c, 3);
    if (!data) { err = "cannot decode image " + path + ": " + stbi_failure_reason(); return false; }
    nx = w; ny = h;
    rgb.assign(data, data + (size_t) w * h * 3);
    stbi_image_free(data);
    return true;
}

bool image_u8::load_memory(const uint8_t * data, size_t n, std::string & err) {
    int w = 0, h = 0, c = 0;
    unsigned char * px = stbi_load_from_memory(data, (int) n, &w, &h, &c, 3);
    if (!px) { err = std::string("cannot decode image: ") + stbi_failure_reason(); return false; }
    nx = w; ny = h;
    rgb.assign(px, px + (size_t) w * h * 3);
    stbi_image_free(px);
    return true;
}

// ---- preprocessing ----------------------------------------------------------

// "smart resize" from the transformers reference: preserve aspect, align to a
// multiple of patch*merge, then pull inside the token budget.
static void smart_resize(int w, int h, int align, int min_px, int max_px,
                         int & w_out, int & h_out) {
    auto round_by = [&](float x) { return (int) std::lround(x / align) * align; };
    auto ceil_by  = [&](float x) { return (int) std::ceil (x / align) * align; };
    auto floor_by = [&](float x) { return (int) std::floor(x / align) * align; };

    w_out = std::max(align, round_by((float) w));
    h_out = std::max(align, round_by((float) h));

    if ((int64_t) w_out * h_out > max_px) {
        const float beta = std::sqrt((float) h * w / max_px);
        w_out = std::max(align, floor_by(w / beta));
        h_out = std::max(align, floor_by(h / beta));
    } else if ((int64_t) w_out * h_out < min_px) {
        const float beta = std::sqrt((float) min_px / ((float) h * w));
        w_out = ceil_by(w * beta);
        h_out = ceil_by(h * beta);
    }
}

static float cubic(float x) {                       // Catmull-Rom / a = -0.5
    x = std::fabs(x);
    if (x <= 1.0f) return ((1.5f * x - 2.5f) * x) * x + 1.0f;
    if (x <  2.0f) return (((-0.5f * x + 2.5f) * x - 4.0f) * x) + 2.0f;
    return 0.0f;
}

// Bicubic to match the reference preprocessor (RESIZE_ALGO_BICUBIC).
static void resize_bicubic(const image_u8 & src, int dw, int dh, std::vector<uint8_t> & dst) {
    dst.assign((size_t) dw * dh * 3, 0);
    const float sx = (float) src.nx / dw, sy = (float) src.ny / dh;
    for (int y = 0; y < dh; y++) {
        const float fy = (y + 0.5f) * sy - 0.5f;
        const int   iy = (int) std::floor(fy);
        for (int x = 0; x < dw; x++) {
            const float fx = (x + 0.5f) * sx - 0.5f;
            const int   ix = (int) std::floor(fx);
            for (int c = 0; c < 3; c++) {
                float acc = 0, wsum = 0;
                for (int m = -1; m <= 2; m++) {
                    const int py = std::clamp(iy + m, 0, src.ny - 1);
                    const float wy = cubic(fy - (iy + m));
                    for (int n = -1; n <= 2; n++) {
                        const int px = std::clamp(ix + n, 0, src.nx - 1);
                        const float w = wy * cubic(fx - (ix + n));
                        acc  += w * src.rgb[((size_t) py * src.nx + px) * 3 + c];
                        wsum += w;
                    }
                }
                const float v = wsum > 0 ? acc / wsum : 0.0f;
                dst[((size_t) y * dw + x) * 3 + c] = (uint8_t) std::clamp(v, 0.0f, 255.0f);
            }
        }
    }
}

void vision_encoder::plan(int nx, int ny, int & grid_w, int & grid_h) const {
    // Budget from the reference: 8..4096 tokens, one per (patch*merge)^2 pixels.
    const int patch_area = (int) (hp_.patch * hp_.patch * hp_.merge * hp_.merge);
    int w = 0, h = 0;
    smart_resize(nx, ny, (int) hp_.align(), 8 * patch_area, 4096 * patch_area, w, h);
    grid_w = w / (int) hp_.align();
    grid_h = h / (int) hp_.align();
}

// ---- loading ----------------------------------------------------------------

vision_encoder::~vision_encoder() {
    if (galloc_) ggml_gallocr_free(galloc_);
    if (buf_)    ggml_backend_buffer_free(buf_);
    if (ctx_)    ggml_free(ctx_);
}

ggml_tensor * vision_encoder::get(const std::string & name) const {
    return ggml_get_tensor(ctx_, name.c_str());
}

void vision_encoder::set_n_threads(int n) {
    if (!backend_ || n <= 0) return;
    // The CPU module is loaded dynamically, so its setter comes from the registry.
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend_));
    auto fn = reg ? (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads") : nullptr;
    if (fn) fn(backend_, n);
}

bool vision_encoder::load(const std::string & path, ggml_backend_t backend,
                          ggml_backend_buffer_type_t buft, std::string & err) {
    backend_ = backend;
    buft_    = buft;
    stage_   = false;
    cpu_     = ggml_backend_dev_type(ggml_backend_get_device(backend)) == GGML_BACKEND_DEVICE_TYPE_CPU;

    // no_alloc: read metadata first, then place the tensors on the backend and
    // stream the data in. Loading into host memory first would cost 0.9 GB.
    ggml_context * meta = nullptr;
    gguf_init_params gp{};
    gp.no_alloc = true;
    gp.ctx      = &meta;
    gguf_context * gc = gguf_init_from_file(path.c_str(), gp);
    if (!gc) { err = "failed to open mmproj " + path; return false; }

    auto u32 = [&](const char * k, uint32_t & dst) {
        const int64_t i = gguf_find_key(gc, k);
        if (i >= 0) dst = (uint32_t) gguf_get_val_u32(gc, i);
    };
    auto f32 = [&](const char * k, float & dst) {
        const int64_t i = gguf_find_key(gc, k);
        if (i >= 0) dst = gguf_get_val_f32(gc, i);
    };
    u32("clip.vision.embedding_length",       hp_.n_embd);
    u32("clip.vision.feed_forward_length",    hp_.n_ff);
    u32("clip.vision.attention.head_count",   hp_.n_head);
    u32("clip.vision.block_count",            hp_.n_layer);
    u32("clip.vision.patch_size",             hp_.patch);
    u32("clip.vision.image_size",             hp_.image_size);
    u32("clip.vision.projection_dim",         hp_.proj_dim);
    hp_.merge = 2;
    u32("clip.vision.spatial_merge_size",     hp_.merge);
    f32("clip.vision.attention.layer_norm_epsilon", hp_.eps);
    {
        const int64_t i = gguf_find_key(gc, "clip.vision.image_mean");
        if (i >= 0 && gguf_get_arr_n(gc, i) == 3) {
            const float * a = (const float *) gguf_get_arr_data(gc, i);
            for (int k = 0; k < 3; k++) hp_.mean[k] = a[k];
        }
        const int64_t j = gguf_find_key(gc, "clip.vision.image_std");
        if (j >= 0 && gguf_get_arr_n(gc, j) == 3) {
            const float * a = (const float *) gguf_get_arr_data(gc, j);
            for (int k = 0; k < 3; k++) hp_.std_[k] = a[k];
        }
    }
    if (!hp_.n_embd || !hp_.n_layer || !hp_.n_head) {
        err = "mmproj is missing vision hyper-parameters"; gguf_free(gc); ggml_free(meta); return false;
    }

    // Rebuild the tensors in our own context so we own their lifetime.
    const int n_tensors = (int) gguf_get_n_tensors(gc);
    ggml_init_params ip{};
    ip.mem_size = ggml_tensor_overhead() * (n_tensors + 8);
    ip.no_alloc = true;
    ctx_ = ggml_init(ip);

    // On the CPU the linear layers are converted to BF16 at load: the tiled GEMM
    // has an AVX-512 BF16 path, measured 17.2 -> 15.4 s on a 1400x1000 screenshot
    // with embeddings at rounding level (first values move in the 4th decimal).
    // Q8_0 was no faster (16.0 s) and cannot take ffn_down (4304 is not a
    // multiple of 32). QWFN_VISION_WTYPE=f16|bf16|q8_0 overrides, an experiment knob.
    ggml_type wtype = cpu_ ? GGML_TYPE_BF16 : GGML_TYPE_COUNT;
    if (const char * e = getenv("QWFN_VISION_WTYPE")) {
        if (!strcmp(e, "f16"))  wtype = GGML_TYPE_COUNT;
        if (!strcmp(e, "bf16")) wtype = GGML_TYPE_BF16;
        if (!strcmp(e, "q8_0")) wtype = GGML_TYPE_Q8_0;
    }
    auto is_linear = [](const std::string & n) {
        return n.size() > 7 && n.compare(n.size() - 7, 7, ".weight") == 0 &&
               n != "v.position_embd.weight" && n.rfind("v.patch_embd", 0) != 0;
    };
    for (int i = 0; i < n_tensors; i++) {
        const char * name = gguf_get_tensor_name(gc, i);
        ggml_tensor * src = ggml_get_tensor(meta, name);
        ggml_type ty = src->type;
        if (wtype != GGML_TYPE_COUNT && src->type == GGML_TYPE_F16 && ggml_n_dims(src) == 2 && is_linear(name) &&
            src->ne[0] % ggml_blck_size(wtype) == 0) ty = wtype;
        ggml_tensor * dst = ggml_new_tensor(ctx_, ty, ggml_n_dims(src), src->ne);
        ggml_set_name(dst, name);
    }

    // The weights' home is host memory, pinned when the backend offers it, and
    // they are staged per image; on a CPU backend the host buffer is the
    // compute buffer and nothing is staged.
    ggml_backend_buffer_type_t wbuft = buft_;
    {
        ggml_backend_dev_t dev = ggml_backend_buft_get_device(buft_);
        ggml_backend_buffer_type_t h = dev ? ggml_backend_dev_host_buffer_type(dev) : nullptr;
        if (h && h != buft_) { wbuft = h; stage_ = true; }
    }
    buf_ = ggml_backend_alloc_ctx_tensors_from_buft(ctx_, wbuft);
    if (!buf_) { err = "failed to allocate mmproj weights"; gguf_free(gc); ggml_free(meta); return false; }

    FILE * f = fopen(path.c_str(), "rb");
    if (!f) { err = "cannot reopen mmproj"; gguf_free(gc); ggml_free(meta); return false; }
    const size_t data_off = gguf_get_data_offset(gc);
    std::vector<uint8_t> tmp;
    for (int i = 0; i < n_tensors; i++) {
        const char * name = gguf_get_tensor_name(gc, i);
        ggml_tensor * dst = get(name);
        const size_t nb   = gguf_get_tensor_size(gc, i);
        tmp.resize(nb);
        if (fseek(f, (long) (data_off + gguf_get_tensor_offset(gc, i)), SEEK_SET) != 0 ||
            fread(tmp.data(), 1, nb, f) != nb) {
            err = std::string("short read for mmproj tensor ") + name;
            fclose(f); gguf_free(gc); ggml_free(meta); return false;
        }
        if (dst->type != gguf_get_tensor_type(gc, i)) {
            const int64_t n = ggml_nelements(dst);
            std::vector<float> f32(n);
            ggml_fp16_to_fp32_row((const ggml_fp16_t *) tmp.data(), f32.data(), n);
            std::vector<uint8_t> conv(ggml_nbytes(dst));
            if (dst->type == GGML_TYPE_BF16) ggml_fp32_to_bf16_row(f32.data(), (ggml_bf16_t *) conv.data(), n);
            else ggml_quantize_chunk(dst->type, f32.data(), conv.data(), 0, dst->ne[1], dst->ne[0], nullptr);
            ggml_backend_tensor_set(dst, conv.data(), 0, conv.size());
        } else ggml_backend_tensor_set(dst, tmp.data(), 0, nb);
    }
    fclose(f);
    gguf_free(gc);
    ggml_free(meta);

    // Bind the named weights.
    pe0_        = get("v.patch_embd.weight");
    pe1_        = get("v.patch_embd.weight.1");
    patch_bias_ = get("v.patch_embd.bias");
    pos_embd_   = get("v.position_embd.weight");
    post_ln_w_  = get("v.post_ln.weight");
    post_ln_b_  = get("v.post_ln.bias");
    mm0_w_      = get("mm.0.weight");   mm0_b_ = get("mm.0.bias");
    mm2_w_      = get("mm.2.weight");   mm2_b_ = get("mm.2.bias");
    if (!pe0_ || !pos_embd_ || !mm0_w_ || !mm2_w_) {
        err = "mmproj is missing expected tensors"; return false;
    }

    layers_.resize(hp_.n_layer);
    for (uint32_t il = 0; il < hp_.n_layer; il++) {
        const std::string p = "v.blk." + std::to_string(il) + ".";
        layer & L = layers_[il];
        L.ln1_w = get(p + "ln1.weight");     L.ln1_b = get(p + "ln1.bias");
        L.qkv_w = get(p + "attn_qkv.weight"); L.qkv_b = get(p + "attn_qkv.bias");
        L.out_w = get(p + "attn_out.weight"); L.out_b = get(p + "attn_out.bias");
        L.ln2_w = get(p + "ln2.weight");     L.ln2_b = get(p + "ln2.bias");
        L.up_w  = get(p + "ffn_up.weight");  L.up_b  = get(p + "ffn_up.bias");
        L.down_w= get(p + "ffn_down.weight");L.down_b= get(p + "ffn_down.bias");
        if (!L.qkv_w || !L.up_w) { err = "mmproj block " + std::to_string(il) + " incomplete"; return false; }
    }

    galloc_ = ggml_gallocr_new(buft_);
    // Flash attention at this head size (72), on this backend? Ask it about the
    // exact op the graph will carry; otherwise attention materialises its scores.
    // The CPU backend says yes but its kernel is a per-key loop: 52 s against
    // 17 s materialised on a 1400x1000 screenshot, so there attention goes
    // through the GEMMs, chunked over the queries to bound the score matrix.
    if (!cpu_) {
        ggml_init_params pp{}; pp.mem_size = ggml_tensor_overhead() * 8; pp.no_alloc = true;
        ggml_context * pc = ggml_init(pp);
        const int64_t d = hp_.d_head(), n = 64, h = hp_.n_head;
        ggml_tensor * q = ggml_new_tensor_3d(pc, GGML_TYPE_F32, d, n, h);
        ggml_tensor * k = ggml_new_tensor_3d(pc, GGML_TYPE_F16, d, n, h);
        ggml_tensor * v = ggml_new_tensor_3d(pc, GGML_TYPE_F16, d, n, h);
        ggml_tensor * o = ggml_flash_attn_ext(pc, q, k, v, nullptr, 1.0f / std::sqrt((float) d), 0.0f, 0.0f);
        ggml_flash_attn_ext_set_prec(o, GGML_PREC_F32);
        use_fa_ = ggml_backend_supports_op(backend_, o);
        ggml_free(pc);
    }
    if (const char * e = getenv("QWFN_VISION_FA")) use_fa_ = atoi(e) != 0;   // experiment knob
    fprintf(stderr, "[qwfn] vision: %u blocks, n_embd %u, patch %u, merge %u -> %u, attention: %s, weights %s\n",
            hp_.n_layer, hp_.n_embd, hp_.patch, hp_.merge, hp_.proj_dim,
            use_fa_ ? "flash" : cpu_ ? "materialised scores, chunked" : "materialised scores",
            cpu_ ? (wtype == GGML_TYPE_BF16 ? "in RAM (linears BF16), computed on the CPU" : "in RAM, computed on the CPU")
                 : stage_ ? "in host memory, staged per image" : "on the device");
    return true;
}

// ---- staging: weights host -> device for one encode -------------------------

bool vision_encoder::stage_in(std::string & err) {
    if (!stage_ || dbuf_) return true;
    const size_t align = ggml_backend_buft_get_alignment(buft_);
    auto padded = [&](size_t n) { return (n + align - 1) / align * align; };
    size_t total = 0;
    for (ggml_tensor * t = ggml_get_first_tensor(ctx_); t; t = ggml_get_next_tensor(ctx_, t)) total += padded(ggml_nbytes(t));
    dbuf_ = ggml_backend_buft_alloc_buffer(buft_, total);
    if (!dbuf_) {
        err = "not enough free VRAM to stage the vision projector (" + std::to_string(total >> 20) + " MB)";
        return false;
    }
    ggml_backend_buffer_set_usage(dbuf_, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    char * base = (char *) ggml_backend_buffer_get_base(dbuf_);
    size_t off = 0;
    saved_.clear();
    for (ggml_tensor * t = ggml_get_first_tensor(ctx_); t; t = ggml_get_next_tensor(ctx_, t)) {
        const size_t nb = ggml_nbytes(t);
        void * host = t->data;
        saved_.emplace_back(t->data, t->buffer);
        t->buffer = dbuf_;
        t->data   = base + off;
        ggml_backend_tensor_set(t, host, 0, nb);   // a DMA from pinned memory
        off += padded(nb);
    }
    return true;
}

void vision_encoder::stage_out() {
    if (!dbuf_) return;
    size_t i = 0;
    for (ggml_tensor * t = ggml_get_first_tensor(ctx_); t; t = ggml_get_next_tensor(ctx_, t), i++) {
        t->data = saved_[i].first; t->buffer = saved_[i].second;
    }
    ggml_backend_buffer_free(dbuf_); dbuf_ = nullptr;
    // The activation arena was carved from the same borrowed VRAM: return it too,
    // or the tier cannot take its dynamic buffer back after the prefill.
    if (galloc_) { ggml_gallocr_free(galloc_); galloc_ = ggml_gallocr_new(buft_); }
}

// ---- encode -----------------------------------------------------------------

bool vision_encoder::encode(const image_u8 & img, std::vector<float> & out,
                            int & n_out, int & grid_w, int & grid_h, std::string & err) {
    if (!ctx_) { err = "vision encoder not loaded"; return false; }
    // Weights onto the device for the duration of this encode (no-op when they live there).
    if (!stage_in(err)) return false;
    struct unstage { vision_encoder * v; ~unstage() { v->stage_out(); } } unstage_guard{this};

    const int patch_area = (int) (hp_.patch * hp_.patch * hp_.merge * hp_.merge);
    int W = 0, H = 0;
    smart_resize(img.nx, img.ny, (int) hp_.align(), 8 * patch_area, 4096 * patch_area, W, H);

    std::vector<uint8_t> px;
    resize_bicubic(img, W, H, px);

    const int pw = W / (int) hp_.patch,  ph = H / (int) hp_.patch;   // patch grid
    const int n_pos = pw * ph;
    grid_w = pw / (int) hp_.merge; grid_h = ph / (int) hp_.merge;
    n_out  = grid_w * grid_h;

    const int64_t n_embd = hp_.n_embd, n_head = hp_.n_head, d_head = hp_.d_head();

    // ---- inputs ------------------------------------------------------------
    ggml_init_params ip{};
    ip.mem_size = ggml_tensor_overhead() * 8;
    ip.no_alloc = true;
    ggml_context * ictx = ggml_init(ip);
    ggml_tensor * inp_raw   = ggml_new_tensor_4d(ictx, GGML_TYPE_F32, W, H, 3, 1);
    ggml_tensor * positions = ggml_new_tensor_1d(ictx, GGML_TYPE_I32, n_pos * 4);
    ggml_tensor * pos_in    = ggml_new_tensor_4d(ictx, GGML_TYPE_F32, n_embd, pw, ph, 1);
    ggml_backend_buffer_t ibuf = ggml_backend_alloc_ctx_tensors_from_buft(ictx, buft_);
    if (!ibuf) { ggml_free(ictx); err = "failed to allocate vision inputs"; return false; }

    // planar [W,H,3], normalised
    {
        std::vector<float> f((size_t) W * H * 3);
        for (int c = 0; c < 3; c++)
            for (int y = 0; y < H; y++)
                for (int x = 0; x < W; x++)
                    f[((size_t) c * H + y) * W + x] =
                        (px[((size_t) y * W + x) * 3 + c] / 255.0f - hp_.mean[c]) / hp_.std_[c];
        ggml_backend_tensor_set(inp_raw, f.data(), 0, f.size() * sizeof(float));
    }
    // M-RoPE positions: 2x2 block order, [h,w,h,w] planes of n_pos each.
    {
        std::vector<int32_t> p((size_t) n_pos * 4);
        int ptr = 0;
        for (int y = 0; y < ph; y += (int) hp_.merge)
            for (int x = 0; x < pw; x += (int) hp_.merge)
                for (int dy = 0; dy < (int) hp_.merge; dy++)
                    for (int dx = 0; dx < (int) hp_.merge; dx++) {
                        p[            ptr] = y + dy;
                        p[  n_pos +   ptr] = x + dx;
                        p[2*n_pos +   ptr] = y + dy;
                        p[3*n_pos +   ptr] = x + dx;
                        ptr++;
                    }
        ggml_backend_tensor_set(positions, p.data(), 0, p.size() * sizeof(int32_t));
    }
    // Learned position embeddings, bilinearly resized on the host from the
    // 48x48 grid the checkpoint stores, in raster order. The graph then applies
    // the same 2x2 interleave it applies to the patches.
    {
        const int src_side = (int) std::lround(std::sqrt((double) pos_embd_->ne[1]));
        std::vector<float> src((size_t) pos_embd_->ne[0] * pos_embd_->ne[1]);
        ggml_backend_tensor_get(pos_embd_, src.data(), 0, src.size() * sizeof(float));
        std::vector<float> dst((size_t) n_embd * n_pos);
        // align_corners bilinear, matching GGML_SCALE_FLAG_ALIGN_CORNERS
        auto coord = [](int i, int n_dst, int n_src) {
            return n_dst > 1 ? (float) i * (n_src - 1) / (n_dst - 1) : 0.0f;
        };
        for (int y = 0; y < ph; y++) {
            const float fy = coord(y, ph, src_side);
            const int y0 = (int) fy, y1 = std::min(y0 + 1, src_side - 1);
            const float wy = fy - y0;
            for (int x = 0; x < pw; x++) {
                const float fx = coord(x, pw, src_side);
                const int x0 = (int) fx, x1 = std::min(x0 + 1, src_side - 1);
                const float wx = fx - x0;
                const size_t o = (size_t) (y * pw + x) * n_embd;
                const size_t a = (size_t) (y0 * src_side + x0) * n_embd;
                const size_t b = (size_t) (y0 * src_side + x1) * n_embd;
                const size_t c = (size_t) (y1 * src_side + x0) * n_embd;
                const size_t d = (size_t) (y1 * src_side + x1) * n_embd;
                for (int64_t k = 0; k < n_embd; k++)
                    dst[o + k] = (1 - wy) * ((1 - wx) * src[a + k] + wx * src[b + k])
                               +      wy  * ((1 - wx) * src[c + k] + wx * src[d + k]);
            }
        }
        ggml_backend_tensor_set(pos_in, dst.data(), 0, dst.size() * sizeof(float));
    }

    // ---- graph -------------------------------------------------------------
    // Materialised attention is chunked over the queries so one chunk's scores
    // [n_pos, chunk, n_head] stay under ~256 MB: 768 queries for a 1400x1000
    // screenshot (5,456 patches), 256 for the largest image the token budget
    // allows (16,384 patches, where the full matrix would be 17 GB).
    int chunk = n_pos;
    if (!use_fa_) {
        const int64_t budget = 256ll << 20;
        chunk = (int) std::max<int64_t>(64, std::min<int64_t>(n_pos, budget / ((int64_t) n_pos * n_head * sizeof(float))));
        chunk = (chunk + 15) / 16 * 16;
    }
    const int n_chunks = (n_pos + chunk - 1) / chunk;
    const size_t graph_size = 512 + (size_t) hp_.n_layer * (48 + (size_t) n_chunks * 14);
    ggml_init_params gp{};
    gp.mem_size = ggml_tensor_overhead() * graph_size + ggml_graph_overhead_custom(graph_size, false);
    gp.no_alloc = true;
    ggml_context * c = ggml_init(gp);
    ggml_cgraph * gf = ggml_new_graph_custom(c, graph_size, false);

    auto norm = [&](ggml_tensor * x, ggml_tensor * w, ggml_tensor * b) {
        x = ggml_norm(c, x, hp_.eps);
        x = ggml_mul(c, x, w);
        return b ? ggml_add(c, x, b) : x;
    };
    // The 2x2 interleave the tower expects: patches within a merge block become
    // adjacent, so the merger's reshape groups the right four.
    auto interleave = [&](ggml_tensor * t) {
        t = ggml_cont_4d(c, t, n_embd * 2, pw / 2, ph, 1);
        t = ggml_reshape_4d(c, t, n_embd * 2, pw / 2, 2, ph / 2);
        t = ggml_permute(c, t, 0, 2, 1, 3);
        return ggml_cont_3d(c, t, n_embd, (int64_t) pw * ph, 1);
    };

    // patch embedding: two kernels summed (still image -> same frame twice)
    ggml_tensor * cur = ggml_add(c,
        ggml_conv_2d(c, pe0_, inp_raw, hp_.patch, hp_.patch, 0, 0, 1, 1),
        ggml_conv_2d(c, pe1_, inp_raw, hp_.patch, hp_.patch, 0, 0, 1, 1));
    cur = ggml_permute(c, cur, 1, 2, 0, 3);          // [w,h,c,b] -> [c,w,h,b]
    cur = interleave(cur);
    if (patch_bias_) cur = ggml_add(c, cur, patch_bias_);
    cur = ggml_add(c, cur, interleave(pos_in));

    const float kq_scale = 1.0f / std::sqrt((float) d_head);
    int sections[4] = { (int) d_head / 4, (int) d_head / 4, (int) d_head / 4, (int) d_head / 4 };

    for (uint32_t il = 0; il < hp_.n_layer; il++) {
        const layer & L = layers_[il];
        ggml_tensor * res = cur;

        ggml_tensor * x = norm(cur, L.ln1_w, L.ln1_b);
        x = ggml_add(c, ggml_mul_mat(c, L.qkv_w, x), L.qkv_b);

        ggml_tensor * Q = ggml_view_3d(c, x, d_head, n_head, n_pos,
                                       ggml_row_size(x->type, d_head), x->nb[1], 0);
        ggml_tensor * K = ggml_view_3d(c, x, d_head, n_head, n_pos,
                                       ggml_row_size(x->type, d_head), x->nb[1],
                                       ggml_row_size(x->type, n_embd));
        ggml_tensor * V = ggml_view_3d(c, x, d_head, n_head, n_pos,
                                       ggml_row_size(x->type, d_head), x->nb[1],
                                       ggml_row_size(x->type, 2 * n_embd));

        Q = ggml_rope_multi(c, ggml_cont(c, Q), positions, nullptr, (int) d_head / 2,
                            sections, GGML_ROPE_TYPE_VISION, 32768, 10000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
        K = ggml_rope_multi(c, ggml_cont(c, K), positions, nullptr, (int) d_head / 2,
                            sections, GGML_ROPE_TYPE_VISION, 32768, 10000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);

        // Full bidirectional attention, no mask. Flash attention, as llama.cpp's
        // clip does it: the [n_pos, n_pos, n_head] score matrix is never
        // materialised. It was -- 1.9 GB for a 1,364-token screenshot, 17 GB
        // for the largest image the token budget allows -- and when that
        // allocation failed the next image crashed inside the allocator.
        ggml_tensor * q = ggml_permute(c, Q, 0, 2, 1, 3);                 // [d,pos,head]
        ggml_tensor * kqv;
        if (use_fa_) {
            ggml_tensor * k = ggml_cast(c, ggml_permute(c, K, 0, 2, 1, 3), GGML_TYPE_F16);
            ggml_tensor * v = ggml_cast(c, ggml_permute(c, V, 0, 2, 1, 3), GGML_TYPE_F16);
            kqv = ggml_flash_attn_ext(c, q, k, v, nullptr, kq_scale, 0.0f, 0.0f);   // [d,head,pos]
            ggml_flash_attn_ext_set_prec(kqv, GGML_PREC_F32);
            kqv = ggml_reshape_2d(c, kqv, n_embd, n_pos);
        } else {
            // The CPU's tiled GEMM wants the reduction length a multiple of 16
            // and both operands contiguous: the head size 72 is padded to 80
            // with zeros (the dot products are unchanged), Q and K are laid out
            // [d,pos,head], and each query chunk is copied out contiguous.
            // Without the padding the scores fell to the per-row dot path.
            // (BF16 scores, padded to 96 for the BF16 GEMM, measured 15.6 -> 15.0 s: not taken.)
            const int64_t d_pad = (d_head + 15) / 16 * 16;
            ggml_tensor * Qp = d_pad != d_head ? ggml_pad(c, Q, (int) (d_pad - d_head), 0, 0, 0) : Q;
            ggml_tensor * Kp = d_pad != d_head ? ggml_pad(c, K, (int) (d_pad - d_head), 0, 0, 0) : K;
            ggml_tensor * qa = ggml_cont(c, ggml_permute(c, Qp, 0, 2, 1, 3));   // [d_pad,pos,head]
            ggml_tensor * k  = ggml_cont(c, ggml_permute(c, Kp, 0, 2, 1, 3));   // [d_pad,pos,head]
            ggml_tensor * v  = ggml_cont(c, ggml_permute(c, V, 1, 2, 0, 3));    // [pos,d,head]
            ggml_tensor * acc = nullptr;
            for (int s0 = 0; s0 < n_pos; s0 += chunk) {
                const int n = std::min(chunk, n_pos - s0);
                ggml_tensor * qi = ggml_cont(c, ggml_view_3d(c, qa, d_pad, n, n_head, qa->nb[1], qa->nb[2], (size_t) s0 * qa->nb[1]));
                ggml_tensor * kq = ggml_mul_mat(c, k, qi);                        // [pos,n,head]
                kq = ggml_soft_max_ext(c, kq, nullptr, kq_scale, 0.0f);
                ggml_tensor * oi = ggml_mul_mat(c, v, kq);                        // [d,n,head]
                oi = ggml_cont_2d(c, ggml_permute(c, oi, 0, 2, 1, 3), n_embd, n);
                acc = acc ? ggml_concat(c, acc, oi, 1) : oi;
            }
            kqv = acc;                                                            // [n_embd,pos]
        }

        x = ggml_add(c, ggml_mul_mat(c, L.out_w, kqv), L.out_b);
        cur = ggml_add(c, x, res);

        res = cur;
        x = norm(cur, L.ln2_w, L.ln2_b);
        x = ggml_add(c, ggml_mul_mat(c, L.up_w, x), L.up_b);
        x = ggml_gelu(c, x);
        x = ggml_add(c, ggml_mul_mat(c, L.down_w, x), L.down_b);
        cur = ggml_add(c, x, res);
    }

    cur = norm(cur, post_ln_w_, post_ln_b_);

    // merger: fold each 2x2 block into one vector, then the two-layer MLP
    cur = ggml_reshape_2d(c, cur, n_embd * 4, (int64_t) n_pos / 4);
    cur = ggml_add(c, ggml_mul_mat(c, mm0_w_, cur), mm0_b_);
    cur = ggml_gelu(c, cur);
    cur = ggml_add(c, ggml_mul_mat(c, mm2_w_, cur), mm2_b_);
    ggml_set_output(cur);
    ggml_build_forward_expand(gf, cur);

    if (!ggml_gallocr_alloc_graph(galloc_, gf)) {
        err = "vision graph allocation failed (" + std::to_string(n_pos) + " patches; not enough free VRAM for this image)";
        // A failed reserve leaves the allocator referencing buffers it has
        // freed, and the next alloc_graph dereferences them: start it over.
        ggml_gallocr_free(galloc_); galloc_ = ggml_gallocr_new(buft_);
        ggml_free(c); ggml_backend_buffer_free(ibuf); ggml_free(ictx); return false;
    }
    fprintf(stderr, "[qwfn] vision graph: %d patches -> %d tokens, arena %.0f MB%s\n", n_pos, n_out,
            ggml_gallocr_get_buffer_size(galloc_, 0) / 1e6,
            use_fa_ ? " (flash attention)" : cpu_ ? (", attention in " + std::to_string(n_chunks) + " chunks of " + std::to_string(chunk)).c_str() : "");
    if (ggml_backend_graph_compute(backend_, gf) != GGML_STATUS_SUCCESS) {
        err = "vision graph compute failed";
        ggml_free(c); ggml_backend_buffer_free(ibuf); ggml_free(ictx); return false;
    }

    out.resize((size_t) hp_.proj_dim * n_out);
    ggml_backend_tensor_get(cur, out.data(), 0, out.size() * sizeof(float));

    ggml_free(c);
    ggml_backend_buffer_free(ibuf);
    ggml_free(ictx);
    return true;
}

}  // namespace qwfn
