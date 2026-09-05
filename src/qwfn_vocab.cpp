#include "qwfn_vocab.h"

#include "llama.h"

#include <cstdio>

namespace qwfn {

vocab::~vocab() {
    if (model_) llama_model_free(model_);
}

bool vocab::load(const std::string & path, std::string & err) {
    static bool backend_ready = false;
    if (!backend_ready) { llama_backend_init(); backend_ready = true; }

    llama_model_params mp = llama_model_default_params();
    mp.vocab_only = true;              // parse the tokenizer, skip all 91 GB
    mp.n_gpu_layers = 0;

    model_ = llama_model_load_from_file(path.c_str(), mp);
    if (!model_) { err = "failed to load vocab from " + path; return false; }

    v_ = llama_model_get_vocab(model_);
    if (!v_) { err = "model has no vocab"; return false; }
    return true;
}

std::vector<int32_t> vocab::encode(const std::string & text,
                                   bool add_special, bool parse_special) const {
    if (!v_) return {};
    // Negative return = -(required size); one retry at that size always suffices.
    std::vector<int32_t> out(text.size() + 16);
    int32_t n = llama_tokenize(v_, text.data(), (int32_t) text.size(),
                               out.data(), (int32_t) out.size(),
                               add_special, parse_special);
    if (n < 0) {
        out.resize(-n);
        n = llama_tokenize(v_, text.data(), (int32_t) text.size(),
                           out.data(), (int32_t) out.size(),
                           add_special, parse_special);
        if (n < 0) return {};
    }
    out.resize(n);
    return out;
}

std::string vocab::piece(int32_t tok, bool special) const {
    if (!v_) return {};
    char buf[256];
    int32_t n = llama_token_to_piece(v_, tok, buf, (int32_t) sizeof(buf), 0, special);
    if (n < 0) {
        std::string big((size_t) -n, '\0');
        n = llama_token_to_piece(v_, tok, big.data(), -n, 0, special);
        if (n < 0) return {};
        big.resize(n);
        return big;
    }
    return std::string(buf, (size_t) n);
}

std::string vocab::decode(const std::vector<int32_t> & toks, bool unparse_special) const {
    if (!v_ || toks.empty()) return {};
    std::string out(toks.size() * 8 + 64, '\0');
    int32_t n = llama_detokenize(v_, toks.data(), (int32_t) toks.size(),
                                 out.data(), (int32_t) out.size(),
                                 false, unparse_special);
    if (n < 0) {
        out.assign((size_t) -n, '\0');
        n = llama_detokenize(v_, toks.data(), (int32_t) toks.size(),
                             out.data(), (int32_t) out.size(),
                             false, unparse_special);
        if (n < 0) return {};
    }
    out.resize(n);
    return out;
}

bool    vocab::is_eog(int32_t tok) const { return v_ && llama_vocab_is_eog(v_, tok); }
int32_t vocab::bos() const { return v_ ? llama_vocab_bos(v_) : -1; }
int32_t vocab::eos() const { return v_ ? llama_vocab_eos(v_) : -1; }
int32_t vocab::n_tokens() const { return v_ ? llama_vocab_n_tokens(v_) : 0; }

}  // namespace qwfn
