// qwfn-chat -- type text, get text. Interactive or one-shot.
//
// The engine's boundary is token ids; this is the only thing above it. The chat
// framing follows the template embedded in the GGUF (tokenizer.chat_template,
// 9,993 chars) rather than a guess at it. Two details from that template matter
// and are easy to get wrong:
//
//   * The generation prompt pre-fills the model INTO the think block:
//     "<|im_start|>assistant\n<think>\n". Thinking is the default; turning it
//     off means emitting a pre-closed "<think>\n\n</think>\n\n" instead, not
//     omitting the block.
//   * Reasoning effort is not a sampler knob -- it is a system message the
//     template injects. xhigh (the template's own default) and low have text;
//     medium deliberately has none.
//
// History is kept as tokens and only ever appended to, never re-rendered. What
// the model generates after our "<think>\n" prefix is exactly what the template
// would have rendered for a completed assistant turn, so appending its output
// verbatim reproduces the template for the next turn. That is also why context
// survives across turns for free: the engine tracks n_past, so turn N prefills
// only its own new tokens, not the conversation so far.

#include "qwfn_engine.h"
#include "qwfn_model.h"
#include "qwfn_vocab.h"
#include "qwfn_vision.h"
#include "qwfn_template.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include <iostream>
#include <random>
#include <string>
#include <vector>

using namespace qwfn;

using clk = std::chrono::steady_clock;
static double since(clk::time_point t) {
    return std::chrono::duration<double>(clk::now() - t).count();
}


// ---- incremental UTF-8 ------------------------------------------------------
// A multi-byte character can straddle two tokens, so bytes are buffered and only
// complete sequences are printed. Without this, streaming output mangles any
// non-ASCII text at token boundaries -- which any real HTML or source file hits.
struct utf8_stream {
    std::string pending;

    std::string feed(const std::string & bytes) {
        pending += bytes;
        size_t cut = pending.size();
        for (size_t back = 0; back < 4 && back < pending.size(); back++) {
            const size_t i = pending.size() - 1 - back;
            const unsigned char c = (unsigned char) pending[i];
            if ((c & 0xC0) == 0x80) continue;                 // continuation
            const size_t need = (c & 0x80) == 0x00 ? 1
                              : (c & 0xE0) == 0xC0 ? 2
                              : (c & 0xF0) == 0xE0 ? 3
                              : (c & 0xF8) == 0xF0 ? 4 : 1;
            if (i + need > pending.size()) cut = i;            // incomplete tail
            break;
        }
        std::string out = pending.substr(0, cut);
        pending.erase(0, cut);
        return out;
    }
};

struct sampler {
    float temp = 0.0f, top_p = 0.95f;
    int   top_k = 40;
    std::mt19937 rng{0xC0FFEEu};

    int pick(const float * lg, int64_t n) {
        if (temp <= 0.0f) {                                    // greedy
            int best = 0;
            for (int64_t v = 1; v < n; v++) if (lg[v] > lg[best]) best = (int) v;
            return best;
        }
        const int k = (int) std::min<int64_t>(top_k > 0 ? top_k : n, n);
        std::vector<int> idx(n);
        for (int64_t v = 0; v < n; v++) idx[v] = (int) v;
        std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
                          [&](int a, int b) { return lg[a] > lg[b]; });
        idx.resize(k);

        const float mx = lg[idx[0]];
        std::vector<float> p(k);
        double sum = 0;
        for (int i = 0; i < k; i++) { p[i] = std::exp((lg[idx[i]] - mx) / temp); sum += p[i]; }
        for (int i = 0; i < k; i++) p[i] = (float) (p[i] / sum);

        double cum = 0;                                        // nucleus
        int keep = k;
        for (int i = 0; i < k; i++) { cum += p[i]; if (cum >= top_p) { keep = i + 1; break; } }

        std::uniform_real_distribution<double> U(0.0, cum);
        double r = U(rng), acc = 0;
        for (int i = 0; i < keep; i++) { acc += p[i]; if (r <= acc) return idx[i]; }
        return idx[0];
    }
};

// ---- attachments ------------------------------------------------------------
struct attachment { std::string path, body; size_t n_tok = 0; };

// An image already run through the vision tower: the embeddings are held until
// the next message, then spliced over its <|image_pad|> placeholders.
struct pending_image {
    std::string        path;
    std::vector<float> emb;
    int n_tok = 0, gw = 0, gh = 0;
};

// Refuse binaries rather than feed the model megabytes of mojibake.
static bool looks_binary(const std::string & s) {
    const size_t n = std::min<size_t>(s.size(), 8192);
    for (size_t i = 0; i < n; i++) if (s[i] == '\0') return true;
    return false;
}

static std::string trim(const std::string & s);

// Paths as a person actually produces them at a console: "~/x.html", a path
// dragged into the terminal (which arrives quoted, or with spaces backslash-
// escaped), or a bare relative path. fopen() understands none of the first three.
static std::string expand_path(std::string p) {
    p = trim(p);
    if (p.size() >= 2 && ((p.front() == '"'  && p.back() == '"') ||
                          (p.front() == '\'' && p.back() == '\''))) {
        p = p.substr(1, p.size() - 2);          // dropped-in quoted path
    } else {
        std::string un;                          // "foo\ bar.html" -> "foo bar.html"
        for (size_t i = 0; i < p.size(); i++) {
            if (p[i] == '\\' && i + 1 < p.size()) { un += p[++i]; continue; }
            un += p[i];
        }
        p = un;
    }
    if (p == "~" || p.rfind("~/", 0) == 0) {
        const char * home = getenv("HOME");
        if (home) p = std::string(home) + p.substr(1);
    }
    return p;
}

static bool read_file(const std::string & path, std::string & out, std::string & err) {
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) { err = "cannot open " + path; return false; }
    fseek(f, 0, SEEK_END);
    const long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (sz < 0) { fclose(f); err = "cannot size " + path; return false; }
    out.resize((size_t) sz);
    const size_t got = sz ? fread(out.data(), 1, (size_t) sz, f) : 0;
    fclose(f);
    out.resize(got);
    if (looks_binary(out)) { err = path + " looks binary; only text files can be attached"; return false; }
    return true;
}

static std::string trim(const std::string & s) {
    const size_t a = s.find_first_not_of(" \t\r\n");
    if (a == std::string::npos) return {};
    const size_t b = s.find_last_not_of(" \t\r\n");
    return s.substr(a, b - a + 1);
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr,
          "usage: qwfn-chat <shard.gguf> [options]\n"
          "\n"
          "  -p, --prompt TEXT   one-shot: answer TEXT and exit (default: interactive)\n"
          "  -f, --file PATH     attach a text file to the first message (repeatable)\n"
          "  -i, --image PATH    attach an image (needs --mmproj) (repeatable)\n"
          "      --mmproj PATH   vision projector gguf; enables images\n"
          "      --system TEXT   system message\n"
          "      --think LEVEL   xhigh (default) | medium | low | off\n"
          "      --hide-think    generate reasoning but do not print it\n"
          "      --temp F        0 = greedy (default)   --top-p F   --top-k N   --seed N\n"
          "      --max N         max tokens per reply (default 512; 0 = fill the context)\n"
          "      --ignore-eos    keep generating past end-of-turn (long-context testing)\n"
          "      --ctx N         context (default 8192)      --batch N   (default 1024)\n"
          "      --ram GB        --vram GB   --threads N   --cpu   --kv f16|q8_0|q4_0 (default q8_0)\n"
          "      --ubatch-kv M   cap on n_kv*ubatch in millions for the prefill graph (default 96)\n"
          "      --ram-frac F    MemAvailable share the RAM tier may take (default 0.60)\n"
          "      --spec-ahead N  predict 1 or 2 layers ahead (default 2)\n"
          "      --no-prefill-overlap   single prefill staging buffer, saves ~1.8 GB RAM\n"
          "\n"
          "Interactive commands:\n"
          "  /file PATH   attach a text file to the next message (rest of line = path)\n"
          "  /image PATH  attach an image to the next message (needs --mmproj)\n"
          "  /files       list what is attached      /drop   discard attachments\n"
          "  /stats       session totals so far      /reset  clear the conversation\n"
          "  /quit        exit\n");
        return 1;
    }

    std::string one_shot, system_msg, effort = "xhigh", mmproj_path;
    std::vector<std::string> startup_files, startup_images;
    bool hide_think = false, interactive = true, ignore_eos = false;
    int  max_gen = 512;                    // 0 = until the context is full
    sampler smp;
    engine_config cfg;
    // vram defaults high on purpose: the tier self-tunes down to whatever the
    // device can spare (state and dense core are allocated first), and without
    // it every routed expert computes on the CPU at 3.2x the cost. --vram 0
    // still disables it. --batch 1024: prefill throughput scales linearly with
    // the ubatch and the input arena is capped by ubatch_kv_product anyway.
    // --batch 4096: the prefill streams each layer's experts once per batch,
    // so a bigger batch is proportionally fewer 53 GB sweeps per prompt; the
    // compute runs in 2048-token chunks regardless. Costs ~0.7 GB of VRAM in
    // work buffers against 1024. Measured: 32K tokens in 101 s at 4096.
    cfg.n_ctx = 8192; cfg.n_batch = 4096; cfg.ram_bytes = 8e9; cfg.vram_bytes = 12e9;
    // q8_0 KV by default: the intended use is 128K+ context, where an f16 cache
    // costs 1.5 GB of expert slots and measured slower (KV type comparison on the reference machine).
    cfg.type_k = cfg.type_v = GGML_TYPE_Q8_0;

    for (int i = 2; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() { return argv[++i]; };
        if ((a == "-p" || a == "--prompt") && i + 1 < argc) { one_shot = next(); interactive = false; continue; }
        if ((a == "-f" || a == "--file")   && i + 1 < argc) { startup_files.push_back(next()); continue; }
        if (a == "--system" && i + 1 < argc) { system_msg = next(); continue; }
        if (a == "--mmproj" && i + 1 < argc) { mmproj_path = next(); continue; }
        if ((a == "-i" || a == "--image") && i + 1 < argc) { startup_images.push_back(next()); continue; }
        if (a == "--think"  && i + 1 < argc) { effort = next(); continue; }
        if (a == "--hide-think") { hide_think = true; continue; }
        if (a == "--ignore-eos") { ignore_eos = true; continue; }
        if (a == "--temp"   && i + 1 < argc) { smp.temp  = (float) atof(next()); continue; }
        if (a == "--top-p"  && i + 1 < argc) { smp.top_p = (float) atof(next()); continue; }
        if (a == "--top-k"  && i + 1 < argc) { smp.top_k = atoi(next()); continue; }
        if (a == "--seed"   && i + 1 < argc) { smp.rng.seed((unsigned) atoi(next())); continue; }
        if (a == "--max"    && i + 1 < argc) { max_gen = atoi(next()); continue; }
        if (a == "--ctx"    && i + 1 < argc) { cfg.n_ctx = (uint32_t) atoi(next()); continue; }
        if (a == "--batch"  && i + 1 < argc) { cfg.n_batch = (uint32_t) atoi(next()); continue; }
        if (a == "--ubatch-kv" && i + 1 < argc) { cfg.ubatch_kv_product = (uint64_t)(atof(next()) * 1e6); continue; }
        if (a == "--indexer-top-k" && i + 1 < argc) { cfg.indexer_top_k = (uint32_t) atoi(next()); continue; }
        if (a == "--ram"    && i + 1 < argc) { cfg.ram_bytes = (size_t)(atof(next()) * 1e9); continue; }
        if (a == "--vram"   && i + 1 < argc) { cfg.vram_bytes = (size_t)(atof(next()) * 1e9); continue; }
        if (a == "--threads"&& i + 1 < argc) { cfg.n_threads = atoi(next()); continue; }
        if (a == "--ram-frac" && i + 1 < argc) { cfg.ram_frac = atof(next()); continue; }
        if (a == "--spec-ahead" && i + 1 < argc) { cfg.speculate_ahead = (uint32_t) atoi(next()); continue; }
        if (a == "--no-prefill-overlap") { cfg.prefill_overlap = false; continue; }
        if (a == "--cpu")   { cfg.use_gpu = false; continue; }
        if (a == "--no-qsa") { cfg.use_qsa = false; continue; }
        if (a == "--prefill-cpu") { cfg.prefill_on_gpu = false; continue; }
        if (a == "--prefill-gpu") { cfg.prefill_on_gpu = true; continue; }  // FAST BUT BROKEN >T~200
        if (a == "--skip-miss") { cfg.skip_miss = true; continue; }
        if (a == "--reserve" && i + 1 < argc) { cfg.vram_reserve = (size_t) atof(next()) * (1ull << 20); continue; }
        if (a == "--predictor" && i + 1 < argc) { cfg.predictor_path = next(); continue; }
        if (a == "--spec-depth" && i + 1 < argc) { cfg.speculate_depth = (uint32_t) atoi(argv[++i]);
            if (cfg.speculate_depth == 0) cfg.speculate = false; continue; }
        if (a == "--spec-ahead" && i + 1 < argc) { cfg.speculate_ahead = (uint32_t) atoi(argv[++i]); continue; }
        if (a == "--spec-depth2" && i + 1 < argc) { cfg.speculate_depth2 = (uint32_t) atoi(argv[++i]); continue; }
        if (a == "--spec-margin" && i + 1 < argc) { cfg.spec_margin = (float) atof(next()); continue; }
        if (a == "--spec-gate-inflight" && i + 1 < argc) { cfg.spec_gate_inflight = (uint32_t) atoi(next()); continue; }
        if (a == "--spec-block") { cfg.spec_block = true; continue; }
        if (a == "--spec-block-layers" && i + 1 < argc) { cfg.spec_block = true; cfg.spec_block_layers = next(); continue; }
        if (a == "--state-host" && i + 1 < argc) {   // none | idx | kv | kv,idx
            std::string v = next();
            cfg.idx_host = v.find("idx") != std::string::npos;
            cfg.kv_host  = v.find("kv")  != std::string::npos;
            continue;
        }
        if (a == "--kv" && i + 1 < argc) {
            std::string v = next();
            cfg.type_k = cfg.type_v = (v == "q8_0") ? GGML_TYPE_Q8_0 :
                                      (v == "q4_0") ? GGML_TYPE_Q4_0 : GGML_TYPE_F16;
            continue;
        }
        fprintf(stderr, "unknown option: %s\n", a.c_str());
        return 1;
    }
    if (!effort_valid(effort)) {
        fprintf(stderr, "--think must be xhigh, medium, low or off\n"); return 1;
    }
    const bool thinking = effort != "off";
    const bool color = isatty(fileno(stdout));

    auto dim   = [&]() { if (color) printf("\033[2m"); };
    auto undim = [&]() { if (color) printf("\033[0m"); };
    // Erase the live status line. Without the erase, a shorter update leaves the
    // tail of the previous one on screen.
    auto clear_line = [&]() { if (color) { printf("\r\033[K"); fflush(stdout); } };

    std::string err;
    qwfn::vocab vb;
    fprintf(stderr, "loading tokenizer...\n");
    if (!vb.load(argv[1], err)) { fprintf(stderr, "error: %s\n", err.c_str()); return 1; }

    model_index mi;
    if (!mi.load(argv[1], err)) { fprintf(stderr, "error: %s\n", err.c_str()); return 1; }

    engine eng;
    if (!eng.init(&mi, nullptr, cfg, std::string(getenv("HOME")) + "/.unsloth/llama.cpp/build/bin", err)) {
        fprintf(stderr, "engine init: %s\n", err.c_str()); return 1;
    }
    fprintf(stderr, "%s\n", eng.memory_summary().c_str());

    qwfn::vision_encoder vis;
    int32_t tok_image_pad = -1;
    if (!mmproj_path.empty()) {
        if (!vis.load(mmproj_path, eng.backend(), eng.buft(), err)) {
            fprintf(stderr, "vision: %s\n", err.c_str()); return 1;
        }
        const auto ip = vb.encode("<|image_pad|>", false, true);
        if (ip.size() != 1) { fprintf(stderr, "vision: <|image_pad|> is not a single token\n"); return 1; }
        tok_image_pad = ip[0];
    }

    std::vector<int32_t> hist;
    int32_t fed = 0;                       // how much of hist the engine has seen
    std::vector<attachment> pending;       // attached, not yet sent
    std::vector<pending_image> images;     // encoded, not yet sent

    // ---- session totals -----------------------------------------------------
    int64_t s_pre_tok = 0, s_dec_tok = 0;
    double  s_pre_t = 0,   s_dec_t = 0;
    int     s_turns = 0;

    auto attach_image = [&](const std::string & raw) {
        if (!vis.loaded()) { printf("  images need --mmproj\n"); return; }
        const std::string path = expand_path(raw);
        qwfn::image_u8 img;
        std::string e;
        if (!img.load(path, e)) { printf("  %s\n", e.c_str()); return; }
        pending_image pi; pi.path = path;
        const auto t0 = clk::now();
        // The projector's weights are staged onto the device for the encode; the
        // tier's dynamic buffer is where they go, as for the server.
        if (vis.weights_on_host()) eng.vram_lend_begin();
        if (!vis.encode(img, pi.emb, pi.n_tok, pi.gw, pi.gh, e)) { eng.vram_lend_end(); printf("  %s\n", e.c_str()); return; }
        printf("  attached %s  (%dx%d -> %dx%d grid, %d image tokens, %.2f s)\n",
               path.c_str(), img.nx, img.ny, pi.gw, pi.gh, pi.n_tok, since(t0));
        images.push_back(std::move(pi));
    };

    auto attach = [&](const std::string & raw) {
        const std::string path = expand_path(raw);
        std::string body, e;
        if (!read_file(path, body, e)) { printf("  %s\n", e.c_str()); return; }
        attachment at{path, body, 0};
        at.n_tok = vb.encode(body, false, false).size();
        const int32_t room = (int32_t) cfg.n_ctx - (int32_t) hist.size();
        printf("  attached %s  (%zu bytes, ~%zu tokens", path.c_str(), body.size(), at.n_tok);
        if ((int32_t) at.n_tok >= room)
            printf(") -- WARNING: bigger than the %d tokens of context left\n", room);
        else
            printf(", %.0f%% of remaining context)\n", 100.0 * at.n_tok / (room > 0 ? room : 1));
        pending.push_back(std::move(at));
    };

    // The system block, built exactly as the template does.
    auto build_preamble = [&]() { return build_system_block(effort, system_msg); };

    auto reset_conversation = [&]() {
        eng.reset();
        hist.clear();
        fed = 0;
        const std::string pre = build_preamble();
        if (!pre.empty()) {
            auto t = vb.encode(pre, false, true);
            hist.insert(hist.end(), t.begin(), t.end());
        }
    };
    reset_conversation();

    // One user turn -> streamed reply. Returns false on a fatal engine error.
    auto turn = [&](const std::string & user) -> bool {
        // Attachments ride in front of the message, fenced so the model can tell
        // file content from instructions.
        std::string body;
        for (const auto & a : pending) {
            body += "<file path=\"" + a.path + "\">\n" + a.body;
            if (!a.body.empty() && a.body.back() != '\n') body += "\n";
            body += "</file>\n\n";
        }
        pending.clear();
        body += user;

        // Framing and content are tokenized SEPARATELY. parse_special scans the
        // whole string, so encoding them together lets a file that merely
        // contains the literal text "<|im_end|>" turn into a real control token
        // and forge a turn boundary mid-attachment. (This file contains those
        // literals, so `-f tools/qwfn_chat.cpp` used to break itself.) Content
        // is encoded with parse_special=false and can only ever be plain text.
        const std::string tt = qwfn::turn_tail(thinking);

        std::vector<int32_t> t;
        auto app = [&](const std::vector<int32_t> & v) { t.insert(t.end(), v.begin(), v.end()); };
        app(vb.encode("<|im_start|>user\n", false, true));

        // Images go in ahead of the text as <|vision_start|> <|image_pad|>xN
        // <|vision_end|>. The pads are real tokens so the sequence and the PLE
        // window stay well formed; only their embeddings get replaced.
        std::vector<std::pair<int32_t, const pending_image *>> splices;  // offset in t
        for (const auto & pi : images) {
            app(vb.encode("<|vision_start|>", false, true));
            splices.emplace_back((int32_t) t.size(), &pi);
            t.insert(t.end(), (size_t) pi.n_tok, tok_image_pad);
            app(vb.encode("<|vision_end|>", false, true));
        }
        if (!body.empty()) app(vb.encode(body, false, /*parse_special=*/false));
        app(vb.encode(tt, false, true));
        const int32_t n_pre = (int32_t) t.size();

        if ((int32_t) hist.size() + n_pre >= (int32_t) cfg.n_ctx) {
            printf("  that message needs %d tokens but only %d are left of %d. /reset first.\n",
                   n_pre, (int32_t) cfg.n_ctx - (int32_t) hist.size(), (int) cfg.n_ctx);
            return true;
        }
        const int32_t t_base = (int32_t) hist.size();
        hist.insert(hist.end(), t.begin(), t.end());
        for (const auto & sp : splices)
            eng.set_embeddings(t_base + sp.first, sp.second->emb.data(), sp.second->n_tok);
        images.clear();

        // ---- prefill, with a live rate --------------------------------------
        const auto tp = clk::now();
        const float * lg = nullptr;
        const int32_t pre_start = fed;
        const int32_t pre_total = (int32_t) hist.size() - pre_start;
        while (fed < (int32_t) hist.size()) {
            const int32_t take = std::min<int32_t>(cfg.n_batch, (int32_t) hist.size() - fed);
            lg = eng.eval(hist.data(), fed + take, take, err);
            if (!lg) { fprintf(stderr, "\nprefill failed: %s\n", err.c_str()); return false; }
            fed += take;
            // Only worth showing when there is more than one ubatch to do --
            // otherwise it flashes once and is gone.
            if (color && pre_total > (int32_t) cfg.n_batch) {
                const double el = since(tp);
                printf("\r\033[2m[prefill %d/%d tok  %.1f tok/s]\033[0m\033[K",
                       fed - pre_start, pre_total, (fed - pre_start) / (el > 0 ? el : 1e-9));
                fflush(stdout);
            }
        }
        const double t_pre = since(tp);
        clear_line();
        s_pre_tok += pre_total; s_pre_t += t_pre;

        // ---- decode ---------------------------------------------------------
        // Budget the reply against the context wall. Generating into it makes
        // eval() fail, and before this it took the whole process down with it.
        const int32_t room   = (int32_t) cfg.n_ctx - (int32_t) hist.size() - 2;
        const int     budget = max_gen > 0 ? std::min<int>(max_gen, std::max<int32_t>(room, 0))
                                           : std::max<int32_t>(room, 0);
        bool hit_wall = false, printed_any = false;
        bool in_think = thinking;
        utf8_stream us;
        int n = 0;
        const auto td = clk::now();
        double last_status = 0;

        if (in_think && !hide_think) { dim(); printf("[thinking] "); }
        fflush(stdout);

        for (; n < budget; n++) {
            const int tok = smp.pick(lg, eng.n_vocab());
            if (vb.is_eog(tok)) {
                hist.push_back(tok); n++;   // not evaluated yet; fed re-syncs below
                // Ignoring end-of-turn is how a long-context run actually reaches
                // the context wall; otherwise the model stops after a few hundred
                // tokens and nothing is exercised.
                if (!ignore_eos) break;
                if ((int32_t) hist.size() + 1 > (int32_t) cfg.n_ctx) { hit_wall = true; break; }
                lg = eng.eval(hist.data(), (int32_t) hist.size(), 1, err);
                if (!lg) { hit_wall = true; break; }
                fed = (int32_t) hist.size();
                continue;
            }

            const std::string piece = vb.piece(tok, false);
            hist.push_back(tok);

            const bool suppressed = in_think && hide_think;
            if (in_think && piece.find("</think>") != std::string::npos) {
                in_think = false;
                if (hide_think) clear_line();
                else { undim(); printf("\n\n"); }
                us.pending.clear();
            } else if (!suppressed) {
                const std::string out = us.feed(piece);
                if (!out.empty()) { printf("%s", out.c_str()); fflush(stdout); printed_any = true; }
            } else if (color) {
                // Nothing is being printed, so the line is free for a live rate.
                // Hidden reasoning runs for minutes; without this it looks hung.
                const double el = since(td);
                if (el - last_status > 0.2) {
                    last_status = el;
                    printf("\r\033[2m[reasoning %d tok  %.1f tok/s]\033[0m\033[K",
                           n, n / (el > 0 ? el : 1e-9));
                    fflush(stdout);
                }
            }

            if ((int32_t) hist.size() + 1 > (int32_t) cfg.n_ctx) { hit_wall = true; n++; break; }
            lg = eng.eval(hist.data(), (int32_t) hist.size(), 1, err);
            if (!lg) { hit_wall = true; n++; break; }
            fed = (int32_t) hist.size();
        }
        const double t_dec = since(td);
        if (in_think && hide_think) clear_line();
        const std::string tail = us.feed("");
        if (!tail.empty()) printf("%s", tail.c_str());
        s_dec_tok += n; s_dec_t += t_dec; s_turns++;

        // Close the turn so the next one continues the template correctly.
        //
        // The subtle case is running out of budget while still inside <think>.
        // Closing that with a bare <|im_end|> leaves an unterminated think block
        // in the history; the next turn then sees a malformed assistant turn and
        // reasons forever without ever emitting an answer.
        std::string closing;
        if (in_think) closing += "\n</think>\n\n";
        if (hist.empty() || !vb.is_eog(hist.back())) closing += "<|im_end|>\n";
        else                                         closing += "\n";
        auto e = vb.encode(closing, false, true);
        hist.insert(hist.end(), e.begin(), e.end());

        // Re-sync from the engine rather than computing an index into hist.
        // The sampled end-of-turn token is appended but never evaluated, so any
        // hand-computed offset here is one too high; the next prefill then skips
        // <|im_end|> entirely and the model sees an unterminated assistant turn,
        // drifting one token further every turn. n_past is the only authority on
        // what the KV actually holds.
        fed = eng.n_past();

        const bool wall = hit_wall || (n >= budget && (max_gen <= 0 || budget < max_gen));
        const bool truncated = !wall && max_gen > 0 && n >= max_gen;

        undim(); printf("\n"); dim();
        if (wall)
            printf("[context full at %d tokens -- stopped cleanly; /reset to continue]\n", eng.n_past());
        else if (truncated)
            printf(in_think
                   ? "[cut off at --max %d while still reasoning -- raise --max, or use --think low/off]\n"
                   : "[cut off at --max %d]\n", max_gen);
        else if (hide_think && !printed_any)
            printf("[the whole reply was reasoning; nothing to show with --hide-think]\n");

        printf("[prefill %d tok %.1f tok/s | generated %d tok in %.1f s (%.1f tok/s) | ctx %d/%d]",
               pre_total, pre_total / (t_pre > 0 ? t_pre : 1e-9),
               n, t_dec, n / (t_dec > 0 ? t_dec : 1e-9),
               eng.n_past(), (int) cfg.n_ctx);
        undim(); printf("\n");
        return true;
    };

    auto session_summary = [&]() {
        if (s_turns == 0) return;
        dim();
        printf("\nsession: %d repl%s\n", s_turns, s_turns == 1 ? "y" : "ies");
        if (eng.pred_total) printf("  prefetch   %.1f%% of the next layer's experts predicted (%llu scored)\n",
               100.0 * eng.pred_hits / eng.pred_total, (unsigned long long) eng.pred_total);
        printf("  prefill    %lld tok in %.1f s   (%.1f tok/s avg)\n",
               (long long) s_pre_tok, s_pre_t, s_pre_tok / (s_pre_t > 0 ? s_pre_t : 1e-9));
        printf("  generated  %lld tok in %.1f s   (%.1f tok/s avg)\n",
               (long long) s_dec_tok, s_dec_t, s_dec_t > 0 ? s_dec_tok / s_dec_t : 0.0);
        printf("  total      %.1f s of model time, context %d/%d\n",
               s_pre_t + s_dec_t, eng.n_past(), (int) cfg.n_ctx);
        undim();
    };

    for (const auto & f : startup_files)  attach(f);
    for (const auto & f : startup_images) attach_image(f);

    if (!interactive) {
        if (!turn(one_shot)) return 1;
        session_summary();
        return 0;
    }

    const bool stdin_tty = isatty(fileno(stdin));
    printf("\nqwfn-chat ready. /file PATH attaches a file, /reset clears, /quit exits.\n");
    if (!stdin_tty) printf("(reading piped input)\n");
    std::string line;
    bool any = false;
    for (;;) {
        printf("\n");
        if (color) printf("\033[1m");
        printf("> ");
        if (color) printf("\033[0m");
        fflush(stdout);
        if (!std::getline(std::cin, line)) {
            if (!any && !stdin_tty) {
                printf("\n\nNo input on stdin. The interactive prompt needs a terminal --\n"
                       "run this from a shell, or pipe messages in:\n"
                       "  printf 'hello\\n/quit\\n' | qwfn-chat MODEL --ram 12\n"
                       "or ask one question and exit:\n"
                       "  qwfn-chat MODEL --ram 12 -p \"your question\"\n");
            }
            break;
        }
        const std::string cmd = trim(line);
        if (cmd.empty()) continue;
        if (cmd == "/quit" || cmd == "/exit") break;
        if (cmd == "/reset") {
            reset_conversation(); pending.clear(); images.clear(); eng.clear_embeddings();
            printf("(conversation cleared)\n");
            continue;
        }
        if (cmd == "/drop")  { pending.clear(); images.clear(); printf("(attachments discarded)\n"); continue; }
        if (cmd == "/stats") { session_summary(); continue; }
        if (cmd == "/files") {
            if (pending.empty()) printf("(nothing attached)\n");
            for (const auto & a : pending)
                printf("  %s  (%zu bytes, ~%zu tokens)\n", a.path.c_str(), a.body.size(), a.n_tok);
            continue;
        }
        if (cmd.rfind("/image", 0) == 0 && (cmd.size() == 6 || cmd[6] == ' ')) {
            const std::string path = trim(cmd.substr(6));
            if (path.empty()) printf("  usage: /image PATH\n");
            else attach_image(path);
            continue;
        }
        if (cmd.rfind("/file", 0) == 0 && (cmd.size() == 5 || cmd[5] == ' ')) {
            // Rest of the line is the path, so paths containing spaces work.
            const std::string path = trim(cmd.substr(5));
            if (path.empty()) printf("  usage: /file PATH\n");
            else attach(path);
            continue;
        }
        if (cmd[0] == '/') { printf("  unknown command: %s\n", cmd.c_str()); continue; }

        printf("\n");
        any = true;
        if (!turn(line)) return 1;
    }
    session_summary();
    printf("\n");
    return 0;
}
