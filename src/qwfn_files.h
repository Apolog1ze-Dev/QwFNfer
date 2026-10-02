#pragma once
// Files a client attaches to a message, as text the model can read.
//
// The chat endpoints take images (the vision projector reads them) and text; a client that
// sends anything else -- an OpenAI `file` part, an Anthropic `document` that is not plain
// text -- used to have it dropped without a word, so the model answered as if nothing had
// been attached. Here a text file reads as itself, a .zip as the text files inside it (source
// trees, logs and notes are what people zip), and anything else as one line naming it and why
// it was not read, so the model can say so instead of guessing.

#include <cstddef>
#include <cstdint>
#include <string>

namespace qwfn {

struct file_text_limits {
    size_t max_file  = 4u << 20;    // bytes of one file's text
    size_t max_total = 16u << 20;   // bytes of text from one attachment, archive included
};

// The attachment `name` (media type `media_type`, may be empty), `n` bytes at `data`, as text.
// Never fails: what cannot be read is named in the result.
std::string file_as_text(const std::string & name, const std::string & media_type,
                         const uint8_t * data, size_t n, const file_text_limits & lim = {});

// Text is valid UTF-8 without NUL bytes; a leading byte-order mark is allowed.
bool looks_like_text(const uint8_t * data, size_t n);

} // namespace qwfn
