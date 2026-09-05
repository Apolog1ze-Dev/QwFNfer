// qwfn_vocab -- text <-> token ids, borrowed from llama.cpp.
//
// The engine's boundary is int32_t ids (engine::eval), which is the right place
// for it: tokenization is a pure string problem with no bearing on the parts of
// this project that are interesting. Rather than reimplement Qwen's BPE and its
// 248,320-entry vocab, this opens the same GGUF a second time with
// `vocab_only = true` -- llama.cpp then parses the tokenizer metadata and skips
// every weight, so it costs a few hundred milliseconds and no meaningful memory,
// not a second 91 GB load.
//
// This is the ONLY part of the runtime that links libllama.

#pragma once

#include <string>
#include <vector>
#include <cstdint>

struct llama_model;
struct llama_vocab;

namespace qwfn {

class vocab {
public:
    ~vocab();
    vocab() = default;
    vocab(const vocab &) = delete;
    vocab & operator=(const vocab &) = delete;

    // `path` is the first shard; llama.cpp follows the -%05d-of-%05d pattern.
    bool load(const std::string & path, std::string & err);

    // `add_special` prepends BOS when the model asks for it; `parse_special`
    // makes <|im_start|>-style markers tokenize as single tokens rather than
    // as their literal characters. Chat framing needs both.
    std::vector<int32_t> encode(const std::string & text,
                                bool add_special = false,
                                bool parse_special = true) const;

    // One id -> its piece. `special` renders control tokens visibly, which is
    // what a --raw debugging view wants and a chat view does not.
    std::string piece(int32_t tok, bool special = false) const;

    std::string decode(const std::vector<int32_t> & toks,
                       bool unparse_special = false) const;

    // End-of-generation covers EOS *and* the template's turn terminators
    // (<|im_end|>), which is what actually ends an assistant turn.
    bool    is_eog(int32_t tok) const;
    int32_t bos() const;
    int32_t eos() const;
    int32_t n_tokens() const;

private:
    llama_model        * model_ = nullptr;
    const llama_vocab  * v_     = nullptr;
};

}  // namespace qwfn
