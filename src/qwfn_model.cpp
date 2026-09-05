#include "qwfn_model.h"

#include "gguf.h"

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <regex>
#include <sstream>

namespace qwfn {

namespace {

int64_t key(const gguf_context * c, const std::string & k) { return gguf_find_key(c, k.c_str()); }

uint32_t get_u32(const gguf_context * c, const std::string & k, uint32_t def) {
    const int64_t id = key(c, k);
    if (id < 0) return def;
    switch (gguf_get_kv_type(c, id)) {
        case GGUF_TYPE_UINT8:  return gguf_get_val_u8 (c, id);
        case GGUF_TYPE_INT8:   return (uint32_t) gguf_get_val_i8 (c, id);
        case GGUF_TYPE_UINT16: return gguf_get_val_u16(c, id);   // split.count lands here
        case GGUF_TYPE_INT16:  return (uint32_t) gguf_get_val_i16(c, id);
        case GGUF_TYPE_UINT32: return gguf_get_val_u32(c, id);
        case GGUF_TYPE_INT32:  return (uint32_t) gguf_get_val_i32(c, id);
        case GGUF_TYPE_UINT64: return (uint32_t) gguf_get_val_u64(c, id);
        case GGUF_TYPE_INT64:  return (uint32_t) gguf_get_val_i64(c, id);
        default: return def;
    }
}

float get_f32(const gguf_context * c, const std::string & k, float def) {
    const int64_t id = key(c, k);
    if (id < 0 || gguf_get_kv_type(c, id) != GGUF_TYPE_FLOAT32) return def;
    return gguf_get_val_f32(c, id);
}

int32_t get_i32(const gguf_context * c, const std::string & k, int32_t def) {
    const int64_t id = key(c, k);
    if (id < 0) return def;
    switch (gguf_get_kv_type(c, id)) {
        case GGUF_TYPE_UINT8:  return gguf_get_val_u8 (c, id);
        case GGUF_TYPE_UINT16: return gguf_get_val_u16(c, id);
        case GGUF_TYPE_INT16:  return gguf_get_val_i16(c, id);
        case GGUF_TYPE_INT32:  return gguf_get_val_i32(c, id);
        case GGUF_TYPE_UINT32: return (int32_t) gguf_get_val_u32(c, id);
        case GGUF_TYPE_INT64:  return (int32_t) gguf_get_val_i64(c, id);
        case GGUF_TYPE_UINT64: return (int32_t) gguf_get_val_u64(c, id);
        default: return def;
    }
}

// Arrays in this checkpoint appear as INT32 (compress_ratios, ple.layers,
// rope.dimension_sections) or UINT64 (ple head offsets / vocab sizes /
// multipliers). Read either into the width the caller wants.
std::vector<int32_t> get_arr_i32(const gguf_context * c, const std::string & k) {
    std::vector<int32_t> out;
    const int64_t id = key(c, k);
    if (id < 0 || gguf_get_kv_type(c, id) != GGUF_TYPE_ARRAY) return out;
    const size_t n = gguf_get_arr_n(c, id);
    const void * d = gguf_get_arr_data(c, id);
    out.resize(n);
    switch (gguf_get_arr_type(c, id)) {
        case GGUF_TYPE_INT32:  for (size_t i = 0; i < n; i++) out[i] = ((const int32_t  *) d)[i]; break;
        case GGUF_TYPE_UINT32: for (size_t i = 0; i < n; i++) out[i] = (int32_t) ((const uint32_t *) d)[i]; break;
        case GGUF_TYPE_INT64:  for (size_t i = 0; i < n; i++) out[i] = (int32_t) ((const int64_t  *) d)[i]; break;
        case GGUF_TYPE_UINT64: for (size_t i = 0; i < n; i++) out[i] = (int32_t) ((const uint64_t *) d)[i]; break;
        default: out.clear(); break;
    }
    return out;
}

std::vector<uint64_t> get_arr_u64(const gguf_context * c, const std::string & k) {
    std::vector<uint64_t> out;
    const int64_t id = key(c, k);
    if (id < 0 || gguf_get_kv_type(c, id) != GGUF_TYPE_ARRAY) return out;
    const size_t n = gguf_get_arr_n(c, id);
    const void * d = gguf_get_arr_data(c, id);
    out.resize(n);
    switch (gguf_get_arr_type(c, id)) {
        case GGUF_TYPE_UINT64: for (size_t i = 0; i < n; i++) out[i] = ((const uint64_t *) d)[i]; break;
        case GGUF_TYPE_INT64:  for (size_t i = 0; i < n; i++) out[i] = (uint64_t) ((const int64_t *) d)[i]; break;
        case GGUF_TYPE_UINT32: for (size_t i = 0; i < n; i++) out[i] = ((const uint32_t *) d)[i]; break;
        case GGUF_TYPE_INT32:  for (size_t i = 0; i < n; i++) out[i] = (uint64_t) ((const int32_t *) d)[i]; break;
        default: out.clear(); break;
    }
    return out;
}

std::string get_str(const gguf_context * c, const std::string & k, const std::string & def) {
    const int64_t id = key(c, k);
    if (id < 0 || gguf_get_kv_type(c, id) != GGUF_TYPE_STRING) return def;
    return gguf_get_val_str(c, id);
}

// "...-00001-of-00003.gguf" -> the path for shard `n` (1-based).
std::string shard_path_for(const std::string & any_shard, int n, int total) {
    static const std::regex re(R"((.*-)(\d{5})(-of-)(\d{5})(\.gguf)$)");
    std::smatch m;
    if (!std::regex_match(any_shard, m, re)) return {};
    char buf[16];
    snprintf(buf, sizeof(buf), "%05d", n);
    char tot[16];
    snprintf(tot, sizeof(tot), "%05d", total);
    return m[1].str() + buf + m[3].str() + tot + m[5].str();
}

} // namespace

bool model_index::read_metadata(gguf_context * c, std::string & err) {
    arch = get_str(c, "general.architecture", "");
    if (arch != "qwen4exp") {
        err = "unexpected architecture '" + arch + "' (this engine only implements qwen4exp / Qwen3.8-Flash-Next)";
        return false;
    }
    const std::string A = arch + ".";

    hp_.n_layer     = get_u32(c, A + "block_count", 0);
    hp_.n_embd      = get_u32(c, A + "embedding_length", 0);
    hp_.n_ctx_train = get_u32(c, A + "context_length", 0);

    hp_.n_head        = get_u32(c, A + "attention.head_count", 0);
    hp_.n_head_kv     = get_u32(c, A + "attention.head_count_kv", 0);
    hp_.n_embd_head_k = get_u32(c, A + "attention.key_length", 0);
    hp_.n_embd_head_v = get_u32(c, A + "attention.value_length", 0);
    hp_.rms_eps       = get_f32(c, A + "attention.layer_norm_rms_epsilon", 1e-6f);

    hp_.full_attention_interval = get_u32(c, A + "full_attention_interval", 4);
    hp_.compress_ratios         = get_arr_i32(c, A + "attention.compress_ratios");

    hp_.rope_freq_base = get_f32(c, A + "rope.freq_base", 10000000.0f);
    hp_.rope_dim       = get_u32(c, A + "rope.dimension_count", 64);
    {
        const auto secs = get_arr_i32(c, A + "rope.dimension_sections");
        for (size_t i = 0; i < secs.size() && i < 4; i++) hp_.mrope_sections[i] = secs[i];
    }

    hp_.n_expert      = get_u32(c, A + "expert_count", 0);
    hp_.n_expert_used = get_u32(c, A + "expert_used_count", 0);
    hp_.n_ff_exp      = get_u32(c, A + "expert_feed_forward_length", 0);
    hp_.n_ff_shexp    = get_u32(c, A + "expert_shared_feed_forward_length", 0);

    hp_.ssm_d_conv  = get_u32(c, A + "ssm.conv_kernel", 4);
    hp_.ssm_d_state = get_u32(c, A + "ssm.state_size", 128);
    hp_.ssm_n_group = get_u32(c, A + "ssm.group_count", 16);
    hp_.ssm_dt_rank = get_u32(c, A + "ssm.time_step_rank", 48);
    hp_.ssm_d_inner = get_u32(c, A + "ssm.inner_size", 6144);

    hp_.idx_n_head  = get_u32(c, A + "attention.indexer.head_count", 4);
    hp_.idx_key_len = get_u32(c, A + "attention.indexer.key_length", 128);
    hp_.idx_top_k   = get_u32(c, A + "attention.indexer.top_k", 2048);

    hp_.hc_count    = get_u32(c, A + "hyper_connection.count", 4);
    hp_.hc_low_rank = get_u32(c, A + "hyper_connection.low_rank", 320);

    hp_.ple_layers            = get_arr_i32(c, A + "ple.layers");
    hp_.ple_ngram_size        = get_u32(c, A + "ple.ngram_size", 3);
    hp_.ple_heads_per_ngram   = get_u32(c, A + "ple.heads_per_ngram", 8);
    hp_.ple_conv_kernel       = get_u32(c, A + "ple.conv_kernel", 4);
    hp_.d_ple                 = get_u32(c, A + "embedding_length_per_layer_input", 160);
    hp_.ple_head_offsets      = get_arr_u64(c, A + "ple.head_offsets");
    hp_.ple_head_vocab_sizes  = get_arr_u64(c, A + "ple.head_vocab_sizes");
    hp_.ple_layer_multipliers = get_arr_u64(c, A + "ple.layer_multipliers");
    hp_.tok_image             = get_i32(c, A + "ple.image_token_id", -1);
    hp_.ple_eos_token_id      = get_i32(c, A + "ple.eos_token_id", -1);

    hp_.tok_bos = get_i32(c, "tokenizer.ggml.bos_token_id", -1);
    hp_.tok_eos = get_i32(c, "tokenizer.ggml.eos_token_id", -1);
    hp_.tok_pad = get_i32(c, "tokenizer.ggml.padding_token_id", -1);
    {
        const int64_t id = key(c, "tokenizer.ggml.tokens");
        if (id >= 0) hp_.n_vocab = (uint32_t) gguf_get_arr_n(c, id);
    }

    if (hp_.n_layer == 0 || hp_.n_embd == 0 || hp_.n_expert == 0) {
        err = "missing core hyper-parameters in GGUF metadata";
        return false;
    }
    // Sanity: the PLE heads must reconstitute a full hidden vector.
    if (!hp_.ple_head_offsets.empty() && hp_.ple_head_offsets.size() != hp_.ple_n_head()) {
        fprintf(stderr, "[qwfn] warning: ple.head_offsets has %zu entries but "
                "(ngram_size-1)*heads_per_ngram = %u\n",
                hp_.ple_head_offsets.size(), hp_.ple_n_head());
    }
    if (!hp_.ple_head_offsets.empty() && hp_.ple_n_head() * hp_.d_ple != hp_.n_embd) {
        fprintf(stderr, "[qwfn] warning: ple_n_head(%u) * d_ple(%u) = %u != n_embd(%u)\n",
                hp_.ple_n_head(), hp_.d_ple, hp_.ple_n_head() * hp_.d_ple, hp_.n_embd);
    }
    return true;
}

bool model_index::load(const std::string & path, std::string & err) {
    gguf_init_params p{};
    p.no_alloc = true;
    p.ctx      = nullptr;

    gguf_context * head = gguf_init_from_file(path.c_str(), p);
    if (!head) { err = "failed to open GGUF: " + path; return false; }
    if (!read_metadata(head, err)) { gguf_free(head); return false; }

    const int total = (int) get_u32(head, "split.count", 1);
    gguf_free(head);

    shard_paths_.clear();
    if (total <= 1) {
        shard_paths_.push_back(path);
    } else {
        for (int i = 1; i <= total; i++) {
            std::string sp = shard_path_for(path, i, total);
            if (sp.empty() || !std::filesystem::exists(sp)) {
                err = "missing shard " + std::to_string(i) + " of " + std::to_string(total) +
                      " (expected " + (sp.empty() ? "<unparsable name>" : sp) + ")";
                return false;
            }
            shard_paths_.push_back(sp);
        }
    }

    for (size_t s = 0; s < shard_paths_.size(); s++) {
        gguf_context * c = gguf_init_from_file(shard_paths_[s].c_str(), p);
        if (!c) { err = "failed to open shard " + shard_paths_[s]; return false; }
        const uint64_t base = gguf_get_data_offset(c);
        const int64_t  n    = gguf_get_n_tensors(c);
        for (int64_t i = 0; i < n; i++) {
            tensor_ref t;
            t.name        = gguf_get_tensor_name(c, i);
            t.shard       = (int) s;
            t.file_offset = base + gguf_get_tensor_offset(c, i);
            t.type        = gguf_get_tensor_type(c, i);
            t.nbytes      = gguf_get_tensor_size(c, i);
            const int64_t * ne = gguf_get_tensor_ne(c, i);
            for (int d = 0; d < 4; d++) t.ne[d] = ne[d];
            tensors_.emplace(t.name, std::move(t));
        }
        gguf_free(c);
    }

    // Resolve the hot-path tensors once.
    expert_tensors_.assign(hp_.n_layer, {nullptr, nullptr, nullptr});
    static const char * part_suffix[EXPERT_NPARTS] = {
        "ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight"
    };
    for (uint32_t il = 0; il < hp_.n_layer; il++) {
        for (int q = 0; q < EXPERT_NPARTS; q++) {
            const std::string nm = "blk." + std::to_string(il) + "." + part_suffix[q];
            const tensor_ref * t = find(nm);
            expert_tensors_[il][q] = t;
            if (t) bytes_experts_ += t->nbytes;
        }
    }
    ple_ = find("per_layer_token_embd.weight");
    if (ple_) bytes_ple_ = ple_->nbytes;

    uint64_t total_bytes = 0;
    for (const auto & kv : tensors_) total_bytes += kv.second.nbytes;
    bytes_dense_ = total_bytes - bytes_experts_ - bytes_ple_;

    return true;
}

const tensor_ref * model_index::find(const std::string & name) const {
    auto it = tensors_.find(name);
    return it == tensors_.end() ? nullptr : &it->second;
}

byte_range model_index::expert_range(uint32_t layer, uint32_t expert_id, expert_part part) const {
    byte_range r;
    if (layer >= expert_tensors_.size() || expert_id >= hp_.n_expert) return r;
    const tensor_ref * t = expert_tensors_[layer][part];
    if (!t) return r;

    // Expert tensors are [ne0, ne1, n_expert]; expert e occupies one ne2 slice.
    // A slice is ne1 rows of ne0 elements, so its size follows ggml row sizing
    // (which accounts for the quant block layout of IQ3_XXS / IQ4_NL / Q8_0).
    const size_t slice = ggml_row_size(t->type, t->ne[0]) * (size_t) t->ne[1];
    r.shard  = t->shard;
    r.offset = t->file_offset + (uint64_t) expert_id * slice;
    r.nbytes = (uint32_t) slice;
    return r;
}

uint32_t model_index::expert_block_bytes(uint32_t layer) const {
    uint32_t n = 0;
    for (int q = 0; q < EXPERT_NPARTS; q++) n += expert_range(layer, 0, (expert_part) q).nbytes;
    return n;
}

byte_range model_index::ple_row_range(uint64_t row) const {
    byte_range r;
    if (!ple_) return r;
    // [d_ple, n_rows]: one row is d_ple contiguous values in the tensor's quant
    // format (IQ4_NL -> 160/32*18 = 90 bytes).
    const size_t row_bytes = ggml_row_size(ple_->type, ple_->ne[0]);
    if (row >= (uint64_t) ple_->ne[1]) return r;
    r.shard  = ple_->shard;
    r.offset = ple_->file_offset + row * row_bytes;
    r.nbytes = (uint32_t) row_bytes;
    return r;
}

std::string hparams::summary() const {
    std::ostringstream o;
    o << "qwen4exp: " << n_layer << " layers, d_model " << n_embd
      << ", residual " << n_residual() << " (" << hc_count << " hyper-connections)\n"
      << "  experts     : " << n_expert << " total, top-" << n_expert_used
      << " + 1 shared, ffn " << n_ff_exp << "\n"
      << "  layer mix   : every " << full_attention_interval << "th is sparse attention -> "
      << (n_layer / full_attention_interval) << " attn / "
      << (n_layer - n_layer / full_attention_interval) << " gated-deltanet\n"
      << "  attention   : " << n_head << " heads / " << n_head_kv << " kv, head_dim "
      << n_embd_head_k << ", indexer " << idx_n_head << "x" << idx_key_len
      << " top-" << idx_top_k << "\n"
      << "  deltanet    : d_inner " << ssm_d_inner << ", state " << ssm_d_state
      << ", groups " << ssm_n_group << ", conv " << ssm_d_conv << "\n"
      << "  ple         : " << ple_n_head() << " heads x " << d_ple
      << " dims, " << ple_ngram_size << "-gram, layers[0]="
      << (ple_layers.empty() ? -1 : ple_layers[0]) << "\n"
      << "  context     : " << n_ctx_train << ", rope base " << rope_freq_base
      << ", mrope [" << mrope_sections[0] << "," << mrope_sections[1] << ","
      << mrope_sections[2] << "," << mrope_sections[3] << "]\n"
      << "  vocab       : " << n_vocab << " (bos " << tok_bos << ", eos " << tok_eos
      << ", image " << tok_image << ")";
    return o.str();
}

} // namespace qwfn
