// qwfn_vision -- the SigLIP2 tower + qwen3vl merger from the mmproj file.
//
// Separate GGUF, separate architecture ("clip"), separate graph. It runs once
// per image and produces [proj_dim, n_out] embeddings that are spliced into the
// text stream in place of the <|image_pad|> tokens, so nothing about the
// language model's own graph changes.
//
// Shape of this model, read from mmproj-F16.gguf:
//   27 blocks, n_embd 1152, ffn 4304 (GELU, no gate), 16 heads (d_head 72)
//   patch 16, spatial merge 2, LayerNorm eps 1e-6 (with biases)
//   patch embedding is TWO conv2d kernels summed (temporal merge; for a still
//   image both see the same frame)
//   learned position embeddings on a 48x48 grid, bilinearly resized per image
//   M-RoPE inside the tower, 4 position components per patch
//   merger: [n_embd*4, n_pos/4] -> mm.0 -> GELU -> mm.2 -> [2560, n_pos/4]
//   is_deepstack_layers is all false for this checkpoint, so no deepstack path

#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#include <cstdint>
#include <string>
#include <vector>

namespace qwfn {

struct vision_hparams {
    uint32_t n_embd = 0, n_ff = 0, n_head = 0, n_layer = 0;
    uint32_t patch = 0, merge = 0, image_size = 0, proj_dim = 0;
    float    eps = 1e-6f;
    float    mean[3] = {0.5f, 0.5f, 0.5f};
    float    std_[3] = {0.5f, 0.5f, 0.5f};

    uint32_t d_head()   const { return n_head ? n_embd / n_head : 0; }
    // One output token per merge x merge block of patches.
    uint32_t align()    const { return patch * merge; }
};

// A decoded image, RGB8, tightly packed.
struct image_u8 {
    int nx = 0, ny = 0;
    std::vector<uint8_t> rgb;          // nx * ny * 3
    bool load(const std::string & path, std::string & err);
    bool load_memory(const uint8_t * data, size_t n, std::string & err);
};

class vision_encoder {
public:
    ~vision_encoder();
    vision_encoder() = default;
    vision_encoder(const vision_encoder &) = delete;
    vision_encoder & operator=(const vision_encoder &) = delete;

    bool load(const std::string & mmproj_path, ggml_backend_t backend,
              ggml_backend_buffer_type_t buft, std::string & err);

    bool loaded() const { return ctx_ != nullptr; }
    const vision_hparams & hp() const { return hp_; }

    // Resize (bicubic, aspect preserved, aligned to patch*merge and clamped to
    // the token budget), normalise, run the tower, and return the projected
    // embeddings as [proj_dim, n_out] in row-major order (n_out vectors of
    // proj_dim floats). grid_w/grid_h are the MERGED grid, so n_out = gw * gh.
    bool encode(const image_u8 & img, std::vector<float> & out,
                int & n_out, int & grid_w, int & grid_h, std::string & err);

    // How many image tokens an image of this size will occupy, without running
    // the tower -- needed to lay out the prompt before encoding.
    void plan(int nx, int ny, int & grid_w, int & grid_h) const;

private:
    struct layer {
        ggml_tensor * ln1_w = nullptr, * ln1_b = nullptr;
        ggml_tensor * qkv_w = nullptr, * qkv_b = nullptr;
        ggml_tensor * out_w = nullptr, * out_b = nullptr;
        ggml_tensor * ln2_w = nullptr, * ln2_b = nullptr;
        ggml_tensor * up_w  = nullptr, * up_b  = nullptr;
        ggml_tensor * down_w = nullptr, * down_b = nullptr;
    };

    ggml_tensor * get(const std::string & name) const;

    vision_hparams hp_;
    ggml_context * ctx_ = nullptr;              // holds the tensor metadata
    ggml_backend_buffer_t buf_ = nullptr;       // holds the weights
    ggml_backend_t        backend_ = nullptr;
    ggml_backend_buffer_type_t buft_ = nullptr;
    ggml_gallocr_t        galloc_ = nullptr;
    bool                  use_fa_ = false;      // flash attention supported by the backend at this head size

    std::vector<layer> layers_;
    ggml_tensor * pe0_ = nullptr, * pe1_ = nullptr, * patch_bias_ = nullptr;
    ggml_tensor * pos_embd_ = nullptr;
    ggml_tensor * post_ln_w_ = nullptr, * post_ln_b_ = nullptr;
    ggml_tensor * mm0_w_ = nullptr, * mm0_b_ = nullptr;
    ggml_tensor * mm2_w_ = nullptr, * mm2_b_ = nullptr;
};

}  // namespace qwfn
