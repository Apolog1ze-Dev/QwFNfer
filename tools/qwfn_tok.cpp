// qwfn-tok -- text in, token ids out, for building qwfn-gen prompt and replay
// files with the model's own tokenizer.
//
//   qwfn-tok <shard.gguf> --raw FILE            ids of FILE, no framing
//   qwfn-tok <shard.gguf> --chat QUESTION [--file PATH]... [--think xhigh|medium|low|off]
//                                              one user turn framed exactly as qwfn-chat
//                                              frames it, attachments fenced as <file>
//
// Ids go to stdout separated by spaces, which is what qwfn-gen's --prompt-file
// and --replay-file read. Attachments are encoded with parse_special=false, like
// qwfn-chat does, so their text cannot forge a turn boundary.

#include "qwfn_template.h"
#include "qwfn_vocab.h"

#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using namespace qwfn;

static bool slurp(const std::string & path, std::string & out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    std::stringstream ss; ss << f.rdbuf(); out = ss.str();
    return true;
}

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: qwfn-tok <shard.gguf> --raw FILE | --chat QUESTION [--file PATH]... [--think MODE]\n");
        return 1;
    }
    std::string raw, question, effort = "xhigh";
    std::vector<std::string> files;
    for (int i = 2; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--raw"   && i + 1 < argc) { raw = argv[++i]; continue; }
        if (a == "--chat"  && i + 1 < argc) { question = argv[++i]; continue; }
        if (a == "--file"  && i + 1 < argc) { files.push_back(argv[++i]); continue; }
        if (a == "--think" && i + 1 < argc) { effort = argv[++i]; continue; }
        fprintf(stderr, "unknown option: %s\n", a.c_str()); return 1;
    }
    if (!effort_valid(effort)) { fprintf(stderr, "bad --think: %s\n", effort.c_str()); return 1; }

    vocab vb; std::string err;
    if (!vb.load(argv[1], err)) { fprintf(stderr, "vocab: %s\n", err.c_str()); return 1; }

    std::vector<int32_t> ids;
    if (!raw.empty()) {
        std::string text;
        if (!slurp(raw, text)) { fprintf(stderr, "cannot read %s\n", raw.c_str()); return 1; }
        ids = vb.encode(text, false, false);
    } else {
        // Framing (parse_special) and content (plain text) are encoded separately.
        auto add = [&](const std::string & s, bool special) {
            std::vector<int32_t> t = vb.encode(s, false, special);
            ids.insert(ids.end(), t.begin(), t.end());
        };
        add(build_system_block(effort, ""), true);
        add("<|im_start|>user\n", true);
        for (const std::string & p : files) {
            std::string text;
            if (!slurp(p, text)) { fprintf(stderr, "cannot read %s\n", p.c_str()); return 1; }
            add("<file path=\"" + p + "\">\n", false);
            add(text, false);
            add("\n</file>\n\n", false);
        }
        add(question, false);
        add(turn_tail(effort != "off"), true);
    }
    for (size_t i = 0; i < ids.size(); i++) printf(i ? " %d" : "%d", ids[i]);
    printf("\n");
    fprintf(stderr, "%zu tokens\n", ids.size());
    return 0;
}
