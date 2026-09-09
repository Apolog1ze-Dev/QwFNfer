// qwfn-server -- an OpenAI-compatible HTTP front end.
//
// One engine, one request at a time. That is not a simplification to be fixed
// later: the engine holds a single KV cache plus DeltaNet recurrent state and
// short-conv history, all of which are sequential accumulations over one
// sequence. Two interleaved conversations would corrupt each other, so requests
// take a mutex and run to completion.
//
// The one optimisation that IS sound here is prefix continuation. A harness
// replays the whole conversation every turn, and if the new token sequence is an
// exact EXTENSION of what the engine has already consumed, we can feed only the
// tail -- position only ever moves forward, so the recurrent state stays valid.
// Anything else (edited history, a new conversation, a regenerate) resets and
// re-prefills, because the recurrent layers cannot be rewound: unlike a KV cache
// you cannot simply forget the tail of a scan.

#include "qwfn_engine.h"
#include "qwfn_model.h"
#include "qwfn_vocab.h"
#include "qwfn_vision.h"
#include "qwfn_template.h"

#include "httplib.h"
#include "nlohmann/json.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <unordered_map>
#include <sys/stat.h>
#include <execinfo.h>
#include <csignal>
#include <atomic>
#include <pthread.h>
#include <vector>

using namespace qwfn;
using json = nlohmann::ordered_json;
using clk  = std::chrono::steady_clock;

static double since(clk::time_point t) {
    return std::chrono::duration<double>(clk::now() - t).count();
}
static int64_t now_unix() {
    return std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}

// ---- base64, for data: image URLs -------------------------------------------
static bool b64_decode(const std::string & in, std::vector<uint8_t> & out) {
    auto val = [](char c) -> int {
        if (c >= 'A' && c <= 'Z') return c - 'A';
        if (c >= 'a' && c <= 'z') return c - 'a' + 26;
        if (c >= '0' && c <= '9') return c - '0' + 52;
        if (c == '+') return 62;
        if (c == '/') return 63;
        return -1;
    };
    int acc = 0, bits = 0;
    out.clear();
    for (char c : in) {
        if (c == '=' || c == '\n' || c == '\r' || c == ' ') continue;
        const int v = val(c);
        if (v < 0) return false;
        acc = (acc << 6) | v;
        bits += 6;
        if (bits >= 8) { bits -= 8; out.push_back((uint8_t) ((acc >> bits) & 0xFF)); }
    }
    return true;
}

// Sampling settings. The two presets are the model's own: the GGUF embeds the
// thinking one (temp 1.0, top_p 0.95, top_k 20); the model card wants temp 0.7,
// top_p 0.8, top_k 20, presence_penalty 1.5 when thinking is off. A request
// overrides any field; POST /props changes the presets for everyone.
struct sampling {
    float temp = 1.0f, top_p = 0.95f, min_p = 0.0f;
    int   top_k = 20;
    float presence_penalty = 0.0f, frequency_penalty = 0.0f, repeat_penalty = 1.0f;
    int   repeat_last_n = 64;
    json to_json() const {
        return {{"temperature", temp}, {"top_p", top_p}, {"top_k", top_k}, {"min_p", min_p},
                {"presence_penalty", presence_penalty}, {"frequency_penalty", frequency_penalty},
                {"repeat_penalty", repeat_penalty}, {"repeat_last_n", repeat_last_n}};
    }
    // Take whatever fields `j` carries (a request body or a /props update).
    void from_json(const json & j) {
        if (j.contains("temperature"))       temp = j["temperature"].get<float>();
        if (j.contains("top_p"))             top_p = j["top_p"].get<float>();
        if (j.contains("top_k"))             top_k = j["top_k"].get<int>();
        if (j.contains("min_p"))             min_p = j["min_p"].get<float>();
        if (j.contains("presence_penalty"))  presence_penalty = j["presence_penalty"].get<float>();
        if (j.contains("frequency_penalty")) frequency_penalty = j["frequency_penalty"].get<float>();
        if (j.contains("repeat_penalty"))    repeat_penalty = j["repeat_penalty"].get<float>();
        if (j.contains("repeat_last_n"))     repeat_last_n = j["repeat_last_n"].get<int>();
    }
};
static sampling preset_thinking()     { sampling s; return s; }
static sampling preset_non_thinking() { sampling s; s.temp = 0.7f; s.top_p = 0.8f; s.top_k = 20; s.presence_penalty = 1.5f; return s; }

struct sampler {
    sampling     cfg;
    std::mt19937 rng{0xC0FFEEu};
    std::vector<int32_t> gen;   // tokens generated so far, for the penalties

    int pick(const float * lg_in, int64_t n) {
        std::vector<float> pen;
        const float * lg = lg_in;
        const bool penalise = cfg.repeat_last_n != 0 && !gen.empty() &&
            (cfg.presence_penalty != 0.0f || cfg.frequency_penalty != 0.0f || cfg.repeat_penalty != 1.0f);
        if (penalise) {   // over the last repeat_last_n GENERATED tokens, llama.cpp semantics
            pen.assign(lg_in, lg_in + n);
            const size_t from = cfg.repeat_last_n > 0 && gen.size() > (size_t) cfg.repeat_last_n ? gen.size() - cfg.repeat_last_n : 0;
            std::unordered_map<int32_t, int> cnt;
            for (size_t i = from; i < gen.size(); i++) cnt[gen[i]]++;
            for (const auto & [t, c] : cnt) {
                if (t < 0 || t >= n) continue;
                float & v = pen[t];
                if (cfg.repeat_penalty != 1.0f) v = v > 0 ? v / cfg.repeat_penalty : v * cfg.repeat_penalty;
                v -= cfg.presence_penalty + cfg.frequency_penalty * c;
            }
            lg = pen.data();
        }
        if (cfg.temp <= 0.0f) {
            int best = 0;
            for (int64_t v = 1; v < n; v++) if (lg[v] > lg[best]) best = (int) v;
            return best;
        }
        const int k = (int) std::min<int64_t>(cfg.top_k > 0 ? cfg.top_k : n, n);
        std::vector<int> idx(n);
        for (int64_t v = 0; v < n; v++) idx[v] = (int) v;
        std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
                          [&](int a, int b) { return lg[a] > lg[b]; });
        idx.resize(k);
        const float mx = lg[idx[0]];
        std::vector<float> p(k);
        double sum = 0;
        for (int i = 0; i < k; i++) { p[i] = std::exp((lg[idx[i]] - mx) / cfg.temp); sum += p[i]; }
        for (int i = 0; i < k; i++) p[i] = (float) (p[i] / sum);
        int keep = k;
        if (cfg.min_p > 0.0f) { const float floor_ = cfg.min_p * p[0]; keep = 1; while (keep < k && p[keep] >= floor_) keep++; }
        double cum = 0; int keep_p = keep;
        for (int i = 0; i < keep; i++) { cum += p[i]; if (cum >= cfg.top_p) { keep_p = i + 1; break; } }
        keep = keep_p;
        std::uniform_real_distribution<double> U(0.0, cum);
        double r = U(rng), acc = 0;
        for (int i = 0; i < keep; i++) { acc += p[i]; if (r <= acc) return idx[i]; }
        return idx[0];
    }
};

// What a harness polls for a live counter: written by the generating thread
// per token, read by /stats, /metrics and /slots without the engine mutex.
// ---- tool calling ------------------------------------------------------------
// The model's own template (Qwen3.8 Flash Next): functions listed as JSON in
// the system turn, calls emitted as
//   <tool_call>\n<function=NAME>\n<parameter=KEY>\nVALUE\n</parameter>\n</function>\n</tool_call>
// and results fed back as a user turn of <tool_response> blocks. The server
// translates OpenAI `tools` / `tool_calls` / role "tool" both ways.
static const char * TOOLS_INSTRUCTIONS =
    "\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n"
    "<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n"
    "<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n"
    "</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n"
    "- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n"
    "- Required parameters MUST be specified\n"
    "- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n"
    "- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n"
    "</IMPORTANT>";

static std::string render_tools_block(const json & tools) {
    if (!tools.is_array() || tools.empty()) return {};
    std::string s = "# Tools\n\nYou have access to the following functions:\n\n<tools>";
    for (const auto & t : tools) s += "\n" + t.dump();
    s += "\n</tools>";
    s += TOOLS_INSTRUCTIONS;
    return s;
}

// OpenAI arguments come as a JSON string (or, from some clients, an object).
static json tool_args_object(const json & fn) {
    if (!fn.contains("arguments")) return json::object();
    const json & a = fn["arguments"];
    if (a.is_object()) return a;
    if (a.is_string()) { try { json o = json::parse(a.get<std::string>()); if (o.is_object()) return o; } catch (...) {} }
    return json::object();
}

// A stable key for "is this the reply we just produced": names and arguments.
// Arguments compare as sorted-key JSON: a harness replays them re-encoded
// (Unsloth Studio sorts the keys), and a key order the model never chose must
// not cost the conversation its prefix continuation.
static std::string tool_calls_key(const json & tool_calls) {
    if (!tool_calls.is_array()) return {};
    std::string k;
    for (const auto & tc : tool_calls) {
        const json & fn = tc.contains("function") ? tc["function"] : tc;
        k += fn.value("name", "") + "(" + nlohmann::json(tool_args_object(fn)).dump() + ");";
    }
    return k;
}

// Parse every complete <tool_call> block in `text`. Parameter values are typed
// by the tool's schema (a "string" stays a string; anything else is parsed as
// JSON when it parses). Returns the text before the first block.
// A call starts only at the template's full marker: a bare "<tool_call>" in
// prose ("I already sent a <tool_call>...") is text, not a call.
static const char * TC_OPEN = "<tool_call>";
static const char * TC_MARK = "<tool_call>\n<function=";
static const char * TC_CLOSE = "</tool_call>";
static size_t find_tool_block(const std::string & text, size_t from) {
    const std::string mark = TC_MARK;
    size_t p = text.find(TC_OPEN, from);
    while (p != std::string::npos && text.compare(p, mark.size(), mark) != 0) p = text.find(TC_OPEN, p + 1);
    return p;
}
static std::string parse_tool_calls(const std::string & text, const json & tools, json & calls, size_t * consumed = nullptr) {
    calls = json::array();
    const std::string OPEN = TC_OPEN, CLOSE = TC_CLOSE;
    size_t first = find_tool_block(text, 0);
    std::string before = text.substr(0, first == std::string::npos ? text.size() : first);
    size_t pos = first, end_consumed = first == std::string::npos ? text.size() : first;
    while (pos != std::string::npos) {
        // A block ends at </tool_call>. The last block of a reply that the model
        // closed with </function> and then ended (no </tool_call>) counts too:
        // the call is complete, only the wrapper is missing.
        size_t close = text.find(CLOSE, pos), after = close == std::string::npos ? 0 : close + CLOSE.size();
        if (close == std::string::npos) {
            const size_t fc = text.find("</function>", pos);
            if (fc == std::string::npos || text.find(OPEN, fc) != std::string::npos) break;
            close = fc + 11; after = text.size();
        }
        const std::string block = text.substr(pos + OPEN.size(), close - pos - OPEN.size());
        end_consumed = after;
        pos = find_tool_block(text, end_consumed);
        const size_t f = block.find("<function=");
        if (f == std::string::npos) continue;
        const size_t fe = block.find('>', f);
        if (fe == std::string::npos) continue;
        const std::string name = block.substr(f + 10, fe - f - 10);
        json schema;   // parameters.properties of this tool, for typing
        if (tools.is_array())
            for (const auto & t : tools) {
                const json & fn = t.contains("function") ? t["function"] : t;
                if (fn.value("name", "") == name && fn.contains("parameters") && fn["parameters"].contains("properties")) schema = fn["parameters"]["properties"];
            }
        json args = json::object();
        size_t q = fe + 1;
        while (true) {
            const size_t ps = block.find("<parameter=", q);
            if (ps == std::string::npos) break;
            const size_t pe = block.find('>', ps);
            if (pe == std::string::npos) break;
            const std::string key = block.substr(ps + 11, pe - ps - 11);
            size_t vs = pe + 1;
            if (vs < block.size() && block[vs] == '\n') vs++;
            // A value runs to </parameter>; one the model never closed runs to the
            // end of its function -- the same cut the streamed parser makes.
            size_t ve = block.find("</parameter>", vs);
            const bool closed = ve != std::string::npos;
            if (!closed) { ve = block.find("</function>", vs); if (ve == std::string::npos) ve = block.size(); }
            std::string val = block.substr(vs, ve - vs);
            if (!val.empty() && val.back() == '\n') val.pop_back();
            const std::string ty = schema.is_object() && schema.contains(key) ? schema[key].value("type", "") : "";
            json v = val;
            if (ty != "string") { try { json parsed = json::parse(val); if (ty.empty() ? !parsed.is_string() : true) v = parsed; } catch (...) {} }
            args[key] = v;
            if (!closed) break;
            q = ve + 12;
        }
        calls.push_back(json{{"id", "call_" + std::to_string(calls.size()) + "_" + std::to_string(now_unix() % 100000)},
                             {"type", "function"},
                             {"function", {{"name", name}, {"arguments", args.dump()}}}});
    }
    if (consumed) *consumed = end_consumed;
    while (!before.empty() && (before.back() == '\n' || before.back() == ' ')) before.pop_back();
    return before;
}

// Token pieces can end inside a multi-byte UTF-8 character (an em dash, CJK,
// an emoji split over two tokens). nlohmann::json::dump() throws on invalid
// UTF-8, and an exception out of a streaming provider makes httplib close the
// connection with no trailer and no log line -- "peer closed connection
// without sending complete message body" on the client. So deltas hold back an
// incomplete tail until the next piece completes it.
static size_t utf8_incomplete_tail(const std::string & s) {
    const size_t n = s.size();
    for (size_t back = 1; back <= 3 && back <= n; back++) {
        const unsigned char c = (unsigned char) s[n - back];
        if ((c & 0xC0) == 0x80) continue;          // continuation byte: keep looking for the lead
        const size_t need = (c & 0x80) == 0 ? 1 : (c & 0xE0) == 0xC0 ? 2 : (c & 0xF0) == 0xE0 ? 3 : (c & 0xF8) == 0xF0 ? 4 : 1;
        return back < need ? back : 0;             // lead byte found: is the sequence complete?
    }
    return 0;
}
static std::string json_dump(const json & j) {   // never throws on bad UTF-8
    return j.dump(-1, ' ', false, json::error_handler_t::replace);
}
// The inside of a JSON string literal for `s`, escaped exactly as json::dump does it.
static std::string json_escape(const std::string & s) {
    const std::string d = json_dump(json(s));
    return d.substr(1, d.size() - 2);
}
static uint64_t hash_bytes(const void * p, size_t n) {   // FNV-1a
    uint64_t h = 1469598103934665603ull;
    const uint8_t * b = (const uint8_t *) p;
    for (size_t i = 0; i < n; i++) { h ^= b[i]; h *= 1099511628211ull; }
    return h;
}

// Streams one <tool_call> block as OpenAI tool_calls deltas WHILE the model is
// writing it. The opening delta (index, id, name) goes out as soon as
// "<function=NAME>" is complete; the arguments follow as a JSON object built up
// in fragments, so the concatenation the client accumulates is what the batch
// parser would have produced. A parameter the tool's schema types as "string"
// (which is what carries code and file contents) streams as it is written,
// JSON-escaped; one typed "array" or "object" (an edit list) streams as the raw
// JSON the model writes; any other parameter is held to its </parameter> and
// typed like the batch parser types it.
//
// Before this the whole block was held until </tool_call>: a 4,000-token file
// written into one parameter meant five minutes of silence on the wire, longer
// than a harness's read timeout (Unsloth Studio's proxy: 300 s), and the
// generation was cut off from the client's side with the work half done.
struct tool_streamer {
    const json & tools;
    size_t block_start;                 // index of "<tool_call>" in the text
    int    index;                       // tool_calls[].index for this call
    std::string call_id;
    std::function<bool(const json &)> emit;   // one tool_calls delta; false = write failed

    enum { HEADER, PARAMS, VALUE_STR, VALUE_HELD, CLOSED } st = HEADER;
    bool   raw = false;                 // VALUE_STR streams the value unquoted and unescaped (array/object)
    bool   opened = false;              // the opening delta has been sent
    size_t pos = 0;                     // parse cursor into the text
    json   schema;                      // parameters.properties of the tool, for typing
    std::string key, ty;                // the parameter being written and its schema type
    int    n_params = 0;
    size_t val_start = 0, val_emitted = 0;

    tool_streamer(const json & t, size_t start, int idx, std::string id, std::function<bool(const json &)> e)
        : tools(t), block_start(start), index(idx), call_id(std::move(id)), emit(std::move(e)) {}

    bool args(const std::string & frag) {
        return emit(json{{"index", index}, {"function", {{"arguments", frag}}}});
    }
    bool close_object() {   // the end of the arguments object
        st = CLOSED;
        return !opened || args(n_params == 0 ? "{}" : "}");
    }
    // The earliest closer at or after `from`: </parameter>, </function> or </tool_call>.
    static size_t next_closer(const std::string & t, size_t from, size_t & len) {
        static const char * C[3] = {"</parameter>", "</function>", "</tool_call>"};
        size_t best = std::string::npos; len = 0;
        for (const char * c : C) {
            const size_t p = t.find(c, from);
            if (p != std::string::npos && (best == std::string::npos || p < best)) { best = p; len = strlen(c); }
        }
        return best;
    }
    json typed(std::string val) const {
        if (!val.empty() && val.back() == '\n') val.pop_back();
        json v = val;
        if (ty != "string") { try { json parsed = json::parse(val); if (ty.empty() ? !parsed.is_string() : true) v = parsed; } catch (...) {} }
        return v;
    }
    // Feed the text so far (the whole accumulated content, block_start-relative
    // positions are absolute into it). Returns false only on a write failure.
    bool feed(const std::string & t) {
        while (true) {
            if (st == CLOSED) return true;
            if (st == HEADER) {
                const size_t f = block_start + strlen(TC_MARK);
                const size_t fe = t.find('>', f), tc = t.find(TC_CLOSE, f);
                if (tc != std::string::npos && (fe == std::string::npos || tc < fe)) { st = CLOSED; return true; }   // no name: not a call
                if (fe == std::string::npos || fe + 1 >= t.size()) return true;   // need the name, and the byte after it
                const std::string name = t.substr(f, fe - f);
                for (const auto & tool : tools) {
                    const json & fn = tool.contains("function") ? tool["function"] : tool;
                    if (fn.value("name", "") == name && fn.contains("parameters") && fn["parameters"].contains("properties")) schema = fn["parameters"]["properties"];
                }
                pos = fe + 1; if (t[pos] == '\n') pos++;
                opened = true; st = PARAMS;
                if (!emit(json{{"index", index}, {"id", call_id}, {"type", "function"},
                               {"function", {{"name", name}, {"arguments", ""}}}})) return false;
                continue;
            }
            if (st == PARAMS) {
                size_t q = pos;
                while (q < t.size() && isspace((unsigned char) t[q])) q++;
                if (q >= t.size()) return true;
                if (t.compare(q, 11, "<parameter=") == 0) {
                    const size_t pe = t.find('>', q);
                    if (pe == std::string::npos || pe + 1 >= t.size()) return true;   // need the key, and the byte after it
                    key = t.substr(q + 11, pe - q - 11);
                    pos = pe + 1; if (t[pos] == '\n') pos++;
                    val_start = val_emitted = pos;
                    ty = schema.is_object() && schema.contains(key) ? schema[key].value("type", "") : "";
                    std::string frag = (n_params == 0 ? "{" : ",") + json_dump(json(key)) + ":";
                    n_params++;
                    raw = ty == "array" || ty == "object";
                    if (ty == "string") { frag += "\""; st = VALUE_STR; }
                    else if (raw)       { st = VALUE_STR; }
                    else                { st = VALUE_HELD; }
                    if (!args(frag)) return false;
                    continue;
                }
                if (t.compare(q, 11, "</function>") == 0) { pos = q + 11; return close_object(); }
                if (t.compare(q, 12, TC_CLOSE) == 0)      { pos = q;      return close_object(); }
                // Something else. Still a possible prefix of one of the tags: wait.
                // Otherwise skip to the next tag, as the batch parser's find() does.
                const size_t avail = t.size() - q;
                for (const char * tag : {"<parameter=", "</function>", "</tool_call>"})
                    if (avail < strlen(tag) && strncmp(tag, t.c_str() + q, avail) == 0) return true;
                size_t best = t.find("<parameter=", q); size_t len = 0;
                const size_t c = next_closer(t, q, len);
                if (c != std::string::npos && (best == std::string::npos || c < best)) best = c;
                if (best == std::string::npos) return true;
                pos = best;
                continue;
            }
            if (st == VALUE_STR) {
                size_t len = 0;
                const size_t c = next_closer(t, val_emitted, len);
                if (c != std::string::npos) {
                    size_t vend = c;
                    if (vend > val_start && t[vend - 1] == '\n') vend--;
                    std::string frag = vend > val_emitted ? (raw ? t.substr(val_emitted, vend - val_emitted) : json_escape(t.substr(val_emitted, vend - val_emitted))) : std::string();
                    if (!raw) frag += "\"";
                    if (!frag.empty() && !args(frag)) return false;
                    if (t.compare(c, len, "</parameter>") == 0) { pos = c + len; st = PARAMS; continue; }
                    pos = t.compare(c, len, "</function>") == 0 ? c + len : c;
                    return close_object();
                }
                // No closer yet: emit what cannot still become one.
                const size_t avail = t.size() - val_emitted;
                size_t hold = 0;
                for (size_t k = std::min<size_t>(13, avail); k > 0 && !hold; k--)
                    for (const char * tag : {"\n</parameter>", "</parameter>", "\n</function>", "</function>", "\n</tool_call>", "</tool_call>"})
                        if (strncmp(tag, t.c_str() + t.size() - k, k) == 0) { hold = k; break; }
                const size_t upto = t.size() - hold;
                if (upto > val_emitted) {
                    const std::string piece = t.substr(val_emitted, upto - val_emitted);
                    if (!args(raw ? piece : json_escape(piece))) return false;
                    val_emitted = upto;
                }
                return true;
            }
            if (st == VALUE_HELD) {
                size_t len = 0;
                const size_t c = next_closer(t, val_start, len);
                if (c == std::string::npos) return true;
                if (!args(json_dump(typed(t.substr(val_start, c - val_start))))) return false;
                if (t.compare(c, len, "</parameter>") == 0) { pos = c + len; st = PARAMS; continue; }
                pos = t.compare(c, len, "</function>") == 0 ? c + len : c;
                return close_object();
            }
        }
    }
};

// Backtraces without a debugger (ptrace is restricted on this machine): a
// fatal signal prints the dying thread's stack; SIGUSR2, sent by the stall
// watchdog to the generating thread, prints where it is stuck.
static void print_backtrace(const char * why) {
    void * frames[64];
    const int n = backtrace(frames, 64);
    char head[160];
    const int hl = snprintf(head, sizeof head, "\n[qwfn-server] === %s: backtrace of thread %lu (%d frames) ===\n", why, (unsigned long) pthread_self(), n);
    (void) !write(2, head, hl);
    backtrace_symbols_fd(frames, n, 2);
}
static void on_fatal(int sig) {
    print_backtrace(sig == SIGSEGV ? "SIGSEGV" : sig == SIGABRT ? "SIGABRT" : sig == SIGBUS ? "SIGBUS" : sig == SIGFPE ? "SIGFPE" : "fatal signal");
    signal(sig, SIG_DFL); raise(sig);
}
static void on_stall_probe(int) { print_backtrace("STALL probe (SIGUSR2)"); }
static pthread_t g_gen_thread;
static std::atomic<bool> g_gen_thread_set{false};

struct live_stats {
    std::mutex mu;
    bool   busy = false;
    int    n_prompt = 0, n_gen = 0;
    double t_prompt = 0, t_gen = 0;          // seconds, the current or last request
    double t_prompt_total = 0, t_gen_total = 0;
    long long n_prompt_total = 0, n_gen_total = 0, n_requests = 0;
    long long n_pairs_total = 0, n_accepted_total = 0;   // the draft head's pairs
    int    n_past = 0;
    json timings() {   // llama.cpp's field names
        std::lock_guard<std::mutex> lk(mu);
        return {{"prompt_n", n_prompt}, {"prompt_ms", t_prompt * 1e3},
                {"prompt_per_second", t_prompt > 0 ? n_prompt / t_prompt : 0.0},
                {"predicted_n", n_gen}, {"predicted_ms", t_gen * 1e3},
                // The first token comes straight from the prompt's logits at ~0 ms:
                // no rate until the clock has something to divide by.
                {"predicted_per_second", t_gen >= 0.1 ? n_gen / t_gen : 0.0}};
    }
};

// ---- server state -----------------------------------------------------------
// content parts -> text, with images encoded to embeddings on the way past
struct pending_img { std::vector<float> emb; int n_tok = 0; };

struct server {
    model_index    mi;
    engine         eng;
    qwfn::vocab    vb;
    vision_encoder vis;
    int32_t        tok_image_pad = -1;

    std::mutex           mu;
    std::vector<int32_t> consumed;      // exactly what the engine has evaluated
    // The image embeddings inside `consumed`, by position (a hash of each). The
    // pads of any two images are the same tokens, so a prompt whose TOKENS extend
    // the consumed prefix can still carry a different image at the same place --
    // an edited turn -- and that has to re-prefill, not continue.
    std::unordered_map<int32_t, uint64_t> consumed_img;

    // The last reply, kept as TOKENS. Re-tokenizing an assistant turn from its
    // text does not reproduce the tokens that were generated: the framing and
    // the content are encoded separately, and BPE merges across the seam
    // differently ("<think>\n" + "\n</think>" vs "<think>\n\n</think>"). Without
    // this, a replayed conversation never matches and every turn re-prefills the
    // whole history.
    std::vector<int32_t> last_gen;      // tokens generated last time
    std::vector<int32_t> last_prompt;   // the prompt those tokens continued
    json                 last_msgs;     // the message list that produced it
    std::string          last_content, last_reasoning, last_tool_key;
    bool                 last_thinking = true;
    std::string          model_id = "qwen3.8-flash-next";
    std::string          model_file;   // the shard the server was started with, for /props
    uint32_t             n_ctx = 0, n_batch = 0;

    // Live-adjustable defaults (GET/POST /props) and the live counter.
    std::mutex   props_mu;
    std::string  def_effort = "xhigh";
    sampling     preset_think = preset_thinking(), preset_nothink = preset_non_thinking();
    int          def_max_tokens = 0;
    int          def_reasoning_budget = 0;   // max reasoning tokens, 0 = unlimited (see generate)
    live_stats   live;

    // A request's prompt, as tokens plus any image embeddings to splice in.
    struct prompt {
        std::vector<int32_t> tok;
        std::vector<std::pair<int32_t, std::vector<float>>> splices;  // offset, emb
    };

};

static bool render_content(server & S, const json & content, std::string & text,
                           std::vector<std::pair<size_t, pending_img>> & imgs,
                           std::string & err) {
    if (content.is_string()) { text += content.get<std::string>(); return true; }
    if (!content.is_array()) { text += content.dump(); return true; }
    for (const auto & part : content) {
        const std::string ty = part.value("type", "text");
        if (ty == "text") { text += part.value("text", ""); continue; }
        if (ty == "image_url" || ty == "input_image") {
            if (!S.vis.loaded()) { err = "server was started without --mmproj; images are not supported"; return false; }
            std::string url;
            if (part.contains("image_url")) {
                url = part["image_url"].is_string() ? part["image_url"].get<std::string>()
                                                    : part["image_url"].value("url", "");
            } else url = part.value("image_url", "");
            const size_t comma = url.find(",");
            if (url.rfind("data:", 0) != 0 || comma == std::string::npos) {
                err = "only data: image URLs are supported (base64)"; return false;
            }
            std::vector<uint8_t> raw;
            if (!b64_decode(url.substr(comma + 1), raw)) { err = "bad base64 in image_url"; return false; }
            image_u8 img;
            if (!img.load_memory(raw.data(), raw.size(), err)) return false;
            pending_img pi;
            int gw = 0, gh = 0;
            const auto t0 = clk::now();
            // Room for the staged weights and the arena: the tier's dynamic buffer,
            // which the prefill that follows would take anyway.
            if (S.vis.weights_on_host()) S.eng.vram_lend_begin();
            if (!S.vis.encode(img, pi.emb, pi.n_tok, gw, gh, err)) { S.eng.vram_lend_end(); return false; }
            fprintf(stderr, "[qwfn-server] image: %zu bytes, %dx%d px -> %dx%d grid, %d tokens, encoded in %.2f s\n",
                    raw.size(), img.nx, img.ny, gw, gh, pi.n_tok, since(t0));
            imgs.emplace_back(text.size(), std::move(pi));   // marker position in text
            continue;
        }
    }
    return true;
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr,
          "usage: qwfn-server <shard.gguf> [options]\n"
          "\n"
          "      --host HOST     bind address (default 127.0.0.1)\n"
          "      --port N        port (default 8080)\n"
          "      --mmproj PATH   vision projector gguf; enables image input. Its weights stay in host memory and are\n"
          "                      staged onto the GPU per image (~60 ms), so vision costs no VRAM at decode\n"
          "      --state-host V  attention state in pinned host memory: none (default) | idx | kv,idx. The VRAM it held goes to\n"
          "                      the expert tier; costs ~0.35 ms/token (idx) or ~2 ms/token (kv,idx) of PCIe reads\n"
          "      --alias NAME    model id reported by /v1/models\n"
          "      --think LEVEL   default reasoning effort: xhigh|medium|low|off\n"
          "      --think-budget N  max reasoning tokens per answer (0 = unlimited); also POST /props {\"reasoning_budget\":N} or per request\n"
          "      --ctx N         context (default 32768)   --batch N (default 2048)\n"
          "      --ram GB        --vram GB   --threads N   --cpu   --kv f16|q8_0\n"
          "      --reserve MB    VRAM kept free after the expert tier is sized (default 768; raise it on a desktop GPU)\n"
          "      --ram-frac F    MemAvailable share the RAM tier may take (default 0.60)\n"
          "      --spec-ahead N  predict 1 or 2 layers ahead (default 2)\n"
          "      --no-prefill-overlap   single prefill staging buffer, saves ~1.8 GB RAM\n"
          "\n"
          "Endpoints: GET /health, GET /v1/models,\n"
          "           POST /v1/chat/completions (stream supported; timings_per_token:true adds live tok/s to every chunk),\n"
          "           GET /props, POST /props (reasoning_effort, max_tokens, thinking/non_thinking sampling presets),\n"
          "           GET /stats (live tok/s, context, expert cache), GET /slots, GET /metrics (Prometheus),\n"
          "           POST /v1/completions\n");
        return 1;
    }

    std::string host = "127.0.0.1", mmproj_path, alias, def_effort = "xhigh";
    int def_reasoning_budget = 0;
    int port = 8080;
    engine_config cfg;
    // vram defaults high on purpose: the tier self-tunes down to whatever the
    // device can spare, and without it every routed expert computes on the CPU
    // at 3.2x the cost. --vram 0 still disables it.
    cfg.n_ctx = 32768; cfg.n_batch = 4096; cfg.ram_bytes = 8e9; cfg.vram_bytes = 12e9;   // see qwfn-chat
    cfg.type_k = cfg.type_v = GGML_TYPE_Q8_0;   // see qwfn-chat

    for (int i = 2; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() { return argv[++i]; };
        if (a == "--host"   && i + 1 < argc) { host = next(); continue; }
        if (a == "--port"   && i + 1 < argc) { port = atoi(next()); continue; }
        if (a == "--mmproj" && i + 1 < argc) { mmproj_path = next(); continue; }
        if (a == "--alias"  && i + 1 < argc) { alias = next(); continue; }
        if (a == "--think"  && i + 1 < argc) { def_effort = next(); continue; }
        if (a == "--think-budget" && i + 1 < argc) { def_reasoning_budget = atoi(next()); continue; }
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
        if (a == "--no-qsa"){ cfg.use_qsa = false; continue; }
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
        if (a == "--mtp" && i + 1 < argc) { cfg.mtp_path = next(); cfg.rollback_snapshots = true; continue; }   // the nextn draft head: pairs verified by the trunk, exact
        if (a == "--kv" && i + 1 < argc) {
            std::string v = next();
            cfg.type_k = cfg.type_v = (v == "q8_0") ? GGML_TYPE_Q8_0 :
                                      (v == "q4_0") ? GGML_TYPE_Q4_0 : GGML_TYPE_F16;
            continue;
        }
        fprintf(stderr, "unknown option: %s\n", a.c_str());
        return 1;
    }
    if (!effort_valid(def_effort)) { fprintf(stderr, "--think must be xhigh|medium|low|off\n"); return 1; }

    server S;
    S.n_ctx = cfg.n_ctx; S.n_batch = cfg.n_batch; S.def_effort = def_effort; S.def_reasoning_budget = def_reasoning_budget;
    S.model_file = argv[1];
    std::string err;

    fprintf(stderr, "loading tokenizer...\n");
    if (!S.vb.load(argv[1], err)) { fprintf(stderr, "error: %s\n", err.c_str()); return 1; }
    if (!S.mi.load(argv[1], err)) { fprintf(stderr, "error: %s\n", err.c_str()); return 1; }
    // The vision projector is loaded onto the device AFTER the engine has sized
    // its expert tier from the free VRAM: reserve its size up front, or it comes
    // out of the decode reserve and the first CUDA graph instantiation fails.
    // The projector's weights live in host memory and are staged into the tier's
    // lent buffer while an image is encoded, so nothing is reserved for them.
    if (!S.eng.init(&S.mi, nullptr, cfg,
                    std::string(getenv("HOME")) + "/.unsloth/llama.cpp/build/bin", err)) {
        fprintf(stderr, "engine init: %s\n", err.c_str()); return 1;
    }
    fprintf(stderr, "%s\n", S.eng.memory_summary().c_str());
    if (!mmproj_path.empty()) {
        if (!S.vis.load(mmproj_path, S.eng.backend(), S.eng.buft(), err)) {
            fprintf(stderr, "vision: %s\n", err.c_str()); return 1;
        }
        const auto ip = S.vb.encode("<|image_pad|>", false, true);
        if (ip.size() != 1) { fprintf(stderr, "vision: <|image_pad|> is not one token\n"); return 1; }
        S.tok_image_pad = ip[0];
    }
    if (!alias.empty()) S.model_id = alias;

    // ---- prompt construction -----------------------------------------------
    // Framing is tokenized with parse_special=true, message content with false,
    // so a message merely containing the text "<|im_end|>" cannot forge a turn.
    // A message's content as JSON for render_content: null (OpenAI tool-call
    // replies carry content: null) becomes an empty string.
    auto content_of = [](const json & m) -> json {
        if (!m.contains("content") || m["content"].is_null()) return json("");
        return m["content"];
    };

    auto build_prompt = [&](const json & messages, const std::string & effort,
                            bool thinking, const std::string & tools_block, const std::string & forced,
                            server::prompt & P, std::string & e) -> bool {
        auto enc_sp = [&](const std::string & s) { return S.vb.encode(s, false, true); };
        auto enc_pl = [&](const std::string & s) { return S.vb.encode(s, false, false); };
        auto app    = [&](const std::vector<int32_t> & v) { P.tok.insert(P.tok.end(), v.begin(), v.end()); };
        auto open_assistant = [&]() {
            app(enc_sp("<|im_start|>assistant\n"));
            app(enc_sp(thinking && forced.empty() ? "<think>\n" : "<think>\n\n</think>\n\n"));
            // A forced call opening: <tool_call> is one of the model's own tokens
            // (tokenized as framing), the function name is text.
            if (!forced.empty()) { app(enc_sp("<tool_call>\n")); app(enc_pl(forced.substr(strlen("<tool_call>\n")))); }
        };
        // The template's rendering of an assistant message's tool calls, after its
        // content. <tool_call> and </tool_call> are single tokens in this vocabulary
        // (user-defined, like <think>): encoded as plain text they become BPE
        // fragments the model never produced, so the history reads as foreign.
        auto app_tool_calls = [&](const json & tcs, bool has_content) {
            bool first = true;
            for (const auto & tc : tcs) {
                const json & fn = tc.contains("function") ? tc["function"] : tc;
                const std::string name = fn.value("name", "");
                if (name.empty()) continue;
                if (first && has_content) app(enc_pl("\n\n")); else if (!first) app(enc_pl("\n"));
                app(enc_sp("<tool_call>\n"));
                std::string body = "<function=" + name + ">\n";
                const json args = tool_args_object(fn);
                for (auto it = args.begin(); it != args.end(); ++it)
                    body += "<parameter=" + it.key() + ">\n" + (it.value().is_string() ? it.value().get<std::string>() : it.value().dump()) + "\n</parameter>\n";
                body += "</function>\n";
                app(enc_pl(body));
                app(enc_sp("</tool_call>"));
                first = false;
            }
        };
        // A user or tool message. Tool results are grouped into one user turn
        // of <tool_response> blocks, as the template does.
        auto render_user_or_tool = [&](const json & m, const std::string & prev_role, const std::string & next_role, std::string & err) -> bool {
            const std::string role = m.value("role", "user");
            std::string text;
            std::vector<std::pair<size_t, pending_img>> imgs;
            if (!render_content(S, content_of(m), text, imgs, err)) return false;
            if (role == "tool") {
                if (prev_role != "tool") app(enc_sp("<|im_start|>user"));
                app(enc_sp("\n<tool_response>\n"));
                if (!text.empty()) app(enc_pl(text));
                app(enc_sp("\n</tool_response>"));
                if (next_role != "tool") app(enc_sp("<|im_end|>\n"));
                return true;
            }
            app(enc_sp("<|im_start|>" + role + "\n"));
            // Parts in the order they were sent, as the template renders them:
            // the text up to each image, then <|vision_start|> <|image_pad|> x N
            // <|vision_end|> in its place. The pads are real tokens, so the
            // sequence and the PLE n-gram window stay well formed; only their
            // embeddings are replaced (see engine::set_embeddings).
            size_t at = 0;
            for (auto & im : imgs) {
                if (im.first > at) app(enc_pl(text.substr(at, im.first - at)));
                at = im.first;
                app(enc_sp("<|vision_start|>"));
                P.splices.emplace_back((int32_t) P.tok.size(), std::move(im.second.emb));
                P.tok.insert(P.tok.end(), (size_t) im.second.n_tok, S.tok_image_pad);
                app(enc_sp("<|vision_end|>"));
            }
            if (at < text.size()) app(enc_pl(text.substr(at)));
            app(enc_sp("<|im_end|>\n"));
            return true;
        };
        auto role_at = [&](size_t i) { return i < messages.size() ? messages[i].value("role", "user") : std::string(); };

        // A harness replays the whole conversation each turn. If this request is
        // the previous one plus (our reply, a new user turn), the prompt is the
        // previous prompt plus the exact tokens we generated plus the new turn --
        // no re-tokenising, so the engine can continue from where it stopped.
        const size_t nprev = S.last_msgs.is_array() ? S.last_msgs.size() : 0;
        if (nprev > 0 && !S.last_gen.empty() && messages.size() >= nprev + 2 &&
            std::equal(S.last_msgs.begin(), S.last_msgs.end(), messages.begin()) &&
            messages[nprev].value("role", "") == "assistant" &&
            content_of(messages[nprev]).is_string() && content_of(messages[nprev]).get<std::string>() == S.last_content &&
            tool_calls_key(messages[nprev].value("tool_calls", json::array())) == S.last_tool_key) {
            P.tok = S.last_prompt;
            P.tok.insert(P.tok.end(), S.last_gen.begin(), S.last_gen.end());
            for (size_t m = nprev + 1; m < messages.size(); m++) {
                const std::string role = messages[m].value("role", "user");
                if (role == "assistant") { P.tok.clear(); break; }   // not a clean extension
                if (!render_user_or_tool(messages[m], role_at(m - 1), role_at(m + 1), e)) return false;
            }
            if (!P.tok.empty()) { open_assistant(); return true; }
            P.splices.clear();   // fall through to a full rebuild
        }

        std::string system_msg;
        size_t first = 0;
        if (!messages.empty() && messages[0].value("role", "") == "system") {
            std::vector<std::pair<size_t, pending_img>> ignore;
            if (!render_content(S, messages[0].value("content", json("")), system_msg, ignore, e)) return false;
            first = 1;
        }
        app(enc_sp(build_system_block(effort, system_msg, tools_block)));

        for (size_t m = first; m < messages.size(); m++) {
            const std::string role = messages[m].value("role", "user");
            if (role == "assistant") {
                std::string text;
                std::vector<std::pair<size_t, pending_img>> imgs;
                if (!render_content(S, content_of(messages[m]), text, imgs, e)) return false;
                const std::string rc = messages[m].value("reasoning_content", "");
                const json tcs = messages[m].value("tool_calls", json::array());
                // If this is verbatim the reply we just produced, replay the
                // exact tokens so the engine can continue instead of re-prefilling.
                if (!S.last_gen.empty() && text == S.last_content && tool_calls_key(tcs) == S.last_tool_key &&
                    (rc.empty() || rc == S.last_reasoning)) {
                    app(enc_sp("<|im_start|>assistant\n"));
                    app(enc_sp(S.last_thinking ? "<think>\n" : "<think>\n\n</think>\n\n"));
                    P.tok.insert(P.tok.end(), S.last_gen.begin(), S.last_gen.end());
                    continue;
                }
                app(enc_sp("<|im_start|>assistant\n<think>\n"));
                if (!rc.empty()) app(enc_pl(rc));
                app(enc_sp("\n</think>\n\n"));
                if (!text.empty()) app(enc_pl(text));
                if (tcs.is_array() && !tcs.empty()) app_tool_calls(tcs, !text.empty());
                app(enc_sp("<|im_end|>\n"));
                continue;
            }
            if (!render_user_or_tool(messages[m], m > first ? role_at(m - 1) : std::string(), role_at(m + 1), e)) return false;
        }
        open_assistant();
        return true;
    };

    // ---- generation ---------------------------------------------------------
    struct gen_result {
        std::string reasoning, content, finish = "stop";
        int n_prompt = 0, n_gen = 0;
        int n_pairs = 0, n_accepted = 0;     // speculative pairs verified, and how many held
        double t_prompt = 0, t_gen = 0;
        bool reasoning_budget_hit = false;
    };

    // on_delta(text, is_reasoning) is called as tokens land; return false to stop.
    // on_delta(text, is_reasoning) is called as tokens land; return false to stop.
    // on_tick() is called after every prefill batch and every generated token,
    // whether or not anything was emitted: a streaming client uses it to keep
    // bytes moving through a silent stretch.
    auto generate = [&](const server::prompt & P, sampler & smp, int max_tok,
                        bool thinking, const std::vector<std::string> & stops,
                        const std::function<bool(const std::string &, bool)> & on_delta,
                        gen_result & R, std::string & e, int reasoning_budget = 0,
                        const std::function<void()> & on_tick = nullptr) -> bool {
        // Prefix continuation: only valid when the new prompt strictly extends
        // what the engine already holds.
        bool extend = P.tok.size() >= S.consumed.size() && !S.consumed.empty() &&
                      std::equal(S.consumed.begin(), S.consumed.end(), P.tok.begin());
        // ... including its images: the pads match any image of the same size.
        if (extend)
            for (const auto & sp : P.splices)
                if (sp.first < (int32_t) S.consumed.size()) {
                    const auto it = S.consumed_img.find(sp.first);
                    if (it == S.consumed_img.end() ||
                        it->second != hash_bytes(sp.second.data(), sp.second.size() * sizeof(float))) { extend = false; break; }
                }
        if (!extend) {
            S.eng.reset();
            S.eng.clear_embeddings();
            S.consumed.clear();
            S.consumed_img.clear();
        }
        if ((int32_t) P.tok.size() >= (int32_t) S.n_ctx) {
            e = "context_length_exceeded: prompt of " + std::to_string(P.tok.size())
              + " tokens exceeds the " + std::to_string(S.n_ctx) + " token context";
            return false;
        }
        for (const auto & sp : P.splices) {
            if (sp.first < (int32_t) S.consumed.size()) continue;   // already evaluated (and verified above)
            S.eng.set_embeddings(sp.first, sp.second.data(),
                                 (int32_t) (sp.second.size() / 2560));
            S.consumed_img[sp.first] = hash_bytes(sp.second.data(), sp.second.size() * sizeof(float));
        }

        std::vector<int32_t> hist = P.tok;
        int32_t fed = (int32_t) S.consumed.size();
        R.n_prompt = (int32_t) hist.size() - fed;
        { std::lock_guard<std::mutex> lk(S.live.mu); S.live.busy = true; S.live.n_prompt = R.n_prompt; S.live.n_gen = 0; S.live.t_prompt = 0; S.live.t_gen = 0; S.live.n_requests++; }
        g_gen_thread = pthread_self(); g_gen_thread_set = true;
        smp.gen.clear();

        const auto tp = clk::now();
        const float * lg = nullptr;
        while (fed < (int32_t) hist.size()) {
            const int32_t take = std::min<int32_t>(S.n_batch, (int32_t) hist.size() - fed);
            lg = S.eng.eval(hist.data(), fed + take, take, e);
            if (!lg) return false;
            fed += take;
            if (on_tick) on_tick();
        }
        R.t_prompt = since(tp);
        { std::lock_guard<std::mutex> lk(S.live.mu); S.live.t_prompt = R.t_prompt; S.live.n_past = S.eng.n_past(); }

        const int32_t room = (int32_t) S.n_ctx - (int32_t) hist.size() - 2;
        const int budget = std::max(0, max_tok > 0 ? std::min(max_tok, room) : room);

        bool in_think = thinking;
        // The template puts "\n\n" between </think> and the answer. Harnesses
        // compare strings, so the answer must not start with it.
        bool content_started = false;
        std::string acc;                 // everything emitted, for stop matching
        const auto td = clk::now();
        int n = 0;
        // With the draft head loaded (--mtp), a sampled token goes in as a pair
        // with the head's draft for the one after it; the trunk's logits at the
        // first position sample the real next token, and when it is the draft,
        // the second position's logits are already the one after. `tok_in` marks
        // a token that came in that way: in the history, evaluated, sampled
        // from, so nothing to do at the evaluation point but move on.
        int32_t tok = smp.pick(lg, S.eng.n_vocab());
        if (!S.eng.mtp_step(&tok, 1, e)) return false;
        bool tok_in = false; int32_t tok_next = -1;
        for (; n < budget; n++) {
            { std::lock_guard<std::mutex> lk(S.live.mu); S.live.n_gen = n + 1; S.live.t_gen = since(td); S.live.n_past = S.eng.n_past(); }
            if (!tok_in) hist.push_back(tok);
            if (S.vb.is_eog(tok)) { n++; break; }
            const std::string piece = S.vb.piece(tok, false);
            if (!tok_in) smp.gen.push_back(tok);

            if (in_think && piece.find("</think>") != std::string::npos) {
                in_think = false;
            } else if (in_think && reasoning_budget > 0 && n + 1 >= reasoning_budget) {
                // Thinking budget (Qwen's mechanism): tell the model time is up,
                // close the think block, and let it answer from what it has.
                // Harness timeouts (15 min in one) are shorter than an xhigh
                // think on a hard prompt at ~12 tok/s.
                (in_think ? R.reasoning : R.content) += piece;
                if (!on_delta(piece, true)) { R.finish = "stop"; n++; break; }
                const std::string cut = "\n\nConsidering the limited time by the user, I have to give the solution based on the thinking directly now.\n</think>\n\n";
                const auto ct = S.vb.encode(cut, false, true);
                for (int32_t t : ct) hist.push_back(t);
                R.reasoning += "\n\n[thinking budget reached]";
                if ((int32_t) hist.size() + 1 > (int32_t) S.n_ctx) { R.finish = "length"; n++; break; }
                // The token just sampled has not been evaluated yet (unless it
                // came in as an accepted draft): it goes in with the injected
                // phrase, or the engine ends one position behind the history.
                lg = S.eng.eval(hist.data(), (int32_t) hist.size(), (int32_t) ct.size() + (tok_in ? 0 : 1), e);
                if (!lg) return false;
                in_think = false;
                R.reasoning_budget_hit = true;
                n++;
                tok = smp.pick(lg, S.eng.n_vocab()); tok_in = false;
                if (!S.eng.mtp_step(&tok, 1, e)) return false;
                continue;
            } else {
                std::string emit = piece;
                if (!in_think && !content_started) {
                    const size_t nb = emit.find_first_not_of(" \t\r\n");
                    if (nb == std::string::npos) emit.clear();
                    else { emit = emit.substr(nb); content_started = true; }
                }
                (in_think ? R.reasoning : R.content) += emit;
                acc += emit;
                if (!emit.empty() && !on_delta(emit, in_think)) { R.finish = "stop"; n++; break; }
                bool hit = false;
                for (const auto & s : stops)
                    if (!s.empty() && R.content.size() >= s.size() &&
                        R.content.compare(R.content.size() - s.size(), s.size(), s) == 0) {
                        R.content.erase(R.content.size() - s.size());
                        hit = true; break;
                    }
                if (hit) { R.finish = "stop"; n++; break; }
            }

            if ((int32_t) hist.size() + 1 > (int32_t) S.n_ctx) { R.finish = "length"; n++; break; }
            if (on_tick) on_tick();
            if (tok_in) { tok = tok_next; tok_in = false; continue; }   // already evaluated with its predecessor
            const int32_t draft = S.eng.mtp_draft_id();
            if (draft >= 0 && !S.vb.is_eog(draft) && n + 1 < budget && (int32_t) hist.size() + 2 <= (int32_t) S.n_ctx) {
                hist.push_back(draft);
                if (!S.eng.eval_decode(hist.data(), (int32_t) hist.size(), 2, e)) return false;
                const float * l0 = S.eng.logits_pos(0), * l1 = S.eng.logits_pos(1);
                const int32_t y = smp.pick(l0, S.eng.n_vocab());
                R.n_pairs++;
                if (y == draft) {
                    R.n_accepted++;
                    smp.gen.push_back(draft);
                    tok_next = smp.pick(l1, S.eng.n_vocab());
                    const int32_t two[2] = { draft, tok_next };
                    if (!S.eng.mtp_step(two, 2, e)) return false;
                    tok = draft; tok_in = true;
                } else {
                    if (!S.eng.rollback(e)) return false;
                    hist.pop_back();
                    tok = y;
                    if (!S.eng.mtp_step(&tok, 1, e)) return false;
                }
            } else {
                lg = S.eng.eval(hist.data(), (int32_t) hist.size(), 1, e);
                if (!lg) return false;
                tok = smp.pick(lg, S.eng.n_vocab());
                if (!S.eng.mtp_step(&tok, 1, e)) return false;
            }
        }
        if (n >= budget && budget > 0) R.finish = "length";
        R.t_gen = since(td);
        R.n_gen = n;
        { std::lock_guard<std::mutex> lk(S.live.mu); S.live.busy = false; S.live.n_gen = n; S.live.t_gen = R.t_gen;
          S.live.n_prompt_total += R.n_prompt; S.live.t_prompt_total += R.t_prompt; S.live.n_gen_total += n; S.live.t_gen_total += R.t_gen; S.live.n_past = S.eng.n_past();
          S.live.n_pairs_total += R.n_pairs; S.live.n_accepted_total += R.n_accepted; }

        // Close the turn so the next request can continue from here. The sampled
        // end-of-turn token was appended but never evaluated, so the engine's
        // n_past -- not hist.size() -- is what it has actually consumed.
        S.consumed.assign(hist.begin(), hist.begin() + S.eng.n_past());
        S.last_gen.assign(hist.begin() + P.tok.size(), hist.end());
        // Generation stops before <|im_end|>\n closes the turn; add it so a
        // replayed history lines up with what the engine will next be fed.
        for (int32_t t : S.vb.encode("\n", false, true)) S.last_gen.push_back(t);
        S.last_prompt    = P.tok;
        S.last_content   = R.content;
        S.last_reasoning = R.reasoning;
        S.last_thinking  = thinking;
        return true;
    };

    // ---- HTTP ---------------------------------------------------------------
    httplib::Server svr;
    svr.set_payload_max_length(256ull << 20);   // base64 images are bulky
    // httplib's socket timeouts default to 5 s per write and per read. A client
    // UI that stops draining the stream for 5 s (rendering a long reasoning
    // trace) would get the connection cut without a trailer and without a log
    // line here -- "peer closed connection without sending complete message
    // body" on its side. A local server can afford to wait.
    signal(SIGSEGV, on_fatal); signal(SIGABRT, on_fatal); signal(SIGBUS, on_fatal); signal(SIGFPE, on_fatal);
    signal(SIGUSR2, on_stall_probe);
    // A local web page (the console, a harness) may read /stats and /props
    // from another origin: allow it.
    svr.set_default_headers({{"Access-Control-Allow-Origin", "*"}, {"Access-Control-Allow-Headers", "Content-Type, Authorization"},
                             {"Access-Control-Allow-Methods", "GET, POST, OPTIONS"}});
    svr.Options(R"(.*)", [](const httplib::Request &, httplib::Response & res) { res.status = 204; });
    svr.set_write_timeout(3600, 0);
    // Stall watchdog: a generation that produces no token for 30 s is logged
    // with what the expert cache is waiting on, every 30 s until it moves.
    std::thread([&]() {
        int last_n = -1; double stalled = 0;
        for (;;) {
            std::this_thread::sleep_for(std::chrono::seconds(5));
            bool busy; int n;
            { std::lock_guard<std::mutex> lk(S.live.mu); busy = S.live.busy; n = S.live.n_prompt * 0 + S.live.n_gen; }
            if (!busy) { last_n = -1; stalled = 0; continue; }
            if (n != last_n) { last_n = n; stalled = 0; continue; }
            stalled += 5;
            if (stalled >= 30 && ((int) stalled % 30) == 0) {
                const int ws = S.eng.cache_wait_state();
                fprintf(stderr, "[qwfn-server] STALL: no new token for %.0f s at generated token %d; expert cache waiting on %s (%zu reads)\n",
                        stalled, n, ws == 1 ? "demand reads" : ws == 2 ? "speculative reads" : "nothing (compute or lock)", S.eng.cache_wait_count());
                if (g_gen_thread_set) pthread_kill(g_gen_thread, SIGUSR2);   // the stuck thread prints its own stack
            }
        }
    }).detach();
    svr.set_read_timeout(600, 0);

    auto fail = [](httplib::Response & res, int code, const std::string & msg,
                   const std::string & type = "invalid_request_error") {
        res.status = code;
        res.set_content(json{{"error", {{"message", msg}, {"type", type}}}}.dump(2, ' ', false, json::error_handler_t::replace),
                        "application/json");
    };

    svr.Get("/health", [&](const httplib::Request &, httplib::Response & res) {
        res.set_content(json{{"status", "ok"}}.dump(), "application/json");
    });

    svr.Get("/v1/models", [&](const httplib::Request &, httplib::Response & res) {
        res.set_content(json{
            {"object", "list"},
            {"data", json::array({ json{
                {"id", S.model_id}, {"object", "model"},
                {"created", now_unix()}, {"owned_by", "qwfnfer"}} })}
        }.dump(2, ' ', false, json::error_handler_t::replace), "application/json");
    });

    // ---- settings a harness can read and change live -------------------------
    auto props_json = [&]() {
        std::lock_guard<std::mutex> lk(S.props_mu);
        return json{
            {"model", S.model_id}, {"n_ctx", S.n_ctx}, {"n_batch", S.n_batch},
            {"default_generation_settings", {
                {"reasoning_effort", S.def_effort}, {"max_tokens", S.def_max_tokens},
                {"reasoning_budget", S.def_reasoning_budget},
                {"thinking", S.preset_think.to_json()}, {"non_thinking", S.preset_nothink.to_json()}}},
            {"skip_miss", cfg.skip_miss}, {"spec_block", cfg.spec_block}, {"mtp", S.eng.mtp_loaded()}, {"model_file", S.model_file},
            {"vision", S.vis.loaded()}, {"vision_weights", S.vis.loaded() ? "host" : "off"},
            {"state_host", cfg.kv_host && cfg.idx_host ? "kv,idx" : cfg.kv_host ? "kv" : cfg.idx_host ? "idx" : "none"},
            {"total_slots", 1}};
    };
    svr.Get("/props", [&](const httplib::Request &, httplib::Response & res) {
        res.set_content(props_json().dump(2, ' ', false, json::error_handler_t::replace), "application/json");
    });
    // POST /props: {"reasoning_effort":"off", "max_tokens":1024,
    //               "thinking":{"temperature":..}, "non_thinking":{...}}
    svr.Post("/props", [&](const httplib::Request & req, httplib::Response & res) {
        json body;
        try { body = json::parse(req.body); } catch (const std::exception & ex) { fail(res, 400, std::string("bad JSON: ") + ex.what()); return; }
        {
            std::lock_guard<std::mutex> lk(S.props_mu);
            if (body.contains("reasoning_effort")) {
                const std::string e = body["reasoning_effort"].get<std::string>();
                if (!effort_valid(e)) { fail(res, 400, "reasoning_effort must be xhigh|medium|low|off"); return; }
                S.def_effort = e;
            }
            if (body.contains("max_tokens"))   S.def_max_tokens = body["max_tokens"].get<int>();
            if (body.contains("reasoning_budget")) S.def_reasoning_budget = body["reasoning_budget"].get<int>();
            if (body.contains("thinking"))     S.preset_think.from_json(body["thinking"]);
            if (body.contains("non_thinking")) S.preset_nothink.from_json(body["non_thinking"]);
            // Flat sampling fields apply to both presets.
            S.preset_think.from_json(body); S.preset_nothink.from_json(body);
        }
        res.set_content(props_json().dump(2, ' ', false, json::error_handler_t::replace), "application/json");
    });
    // ---- the live counter ------------------------------------------------------
    auto stats_json = [&]() {
        json t = S.live.timings();
        const auto & c = S.eng.cache_stats();   // racy reads of plain counters: a monitor, not a ledger
        long long np, ng, nr, npair, nacc; double tp, tg; bool busy; int n_past;
        { std::lock_guard<std::mutex> lk(S.live.mu); np = S.live.n_prompt_total; ng = S.live.n_gen_total; nr = S.live.n_requests;
          tp = S.live.t_prompt_total; tg = S.live.t_gen_total; busy = S.live.busy; n_past = S.live.n_past;
          npair = S.live.n_pairs_total; nacc = S.live.n_accepted_total; }
        return json{
            {"busy", busy},
            {"prompt", {{"n", t["prompt_n"]}, {"ms", t["prompt_ms"]}, {"tokens_per_second", t["prompt_per_second"]}}},
            {"generation", {{"n", t["predicted_n"]}, {"ms", t["predicted_ms"]}, {"tokens_per_second", t["predicted_per_second"]}}},
            {"context", {{"n_past", n_past}, {"n_ctx", S.n_ctx}}},
            {"totals", {{"requests", nr}, {"prompt_tokens", np}, {"prompt_tokens_per_second", tp > 0 ? np / tp : 0.0},
                        {"generated_tokens", ng}, {"generated_tokens_per_second", tg > 0 ? ng / tg : 0.0}}},
            {"expert_cache", {{"hit_rate", c.hit_rate()}, {"vram_served", c.gpu_rate()},
                              {"bytes_from_disk", c.bytes_from_disk}}},
            {"speculative", {{"pairs", npair}, {"accepted", nacc}, {"acceptance", npair ? (double) nacc / npair : 0.0}}},
            {"timings", t}};
    };
    svr.Get("/stats", [&](const httplib::Request &, httplib::Response & res) {
        res.set_content(stats_json().dump(2, ' ', false, json::error_handler_t::replace), "application/json");
    });
    svr.Get("/slots", [&](const httplib::Request &, httplib::Response & res) {
        json st = stats_json();
        res.set_content(json::array({ json{
            {"id", 0}, {"id_task", -1}, {"is_processing", st["busy"]},
            {"n_ctx", S.n_ctx}, {"n_past", st["context"]["n_past"]},
            {"model", S.model_id}, {"params", props_json()["default_generation_settings"]},
            {"next_token", {{"n_decoded", st["generation"]["n"]}}}} }).dump(2, ' ', false, json::error_handler_t::replace), "application/json");
    });
    svr.Get("/metrics", [&](const httplib::Request &, httplib::Response & res) {
        json st = stats_json();
        char buf[2048];
        snprintf(buf, sizeof buf,
            "# HELP llamacpp:prompt_tokens_total Number of prompt tokens processed.\n# TYPE llamacpp:prompt_tokens_total counter\nllamacpp:prompt_tokens_total %lld\n"
            "# HELP llamacpp:tokens_predicted_total Number of generation tokens processed.\n# TYPE llamacpp:tokens_predicted_total counter\nllamacpp:tokens_predicted_total %lld\n"
            "# HELP llamacpp:prompt_tokens_seconds Average prompt throughput in tokens/s.\n# TYPE llamacpp:prompt_tokens_seconds gauge\nllamacpp:prompt_tokens_seconds %.2f\n"
            "# HELP llamacpp:predicted_tokens_seconds Average generation throughput in tokens/s (current or last request).\n# TYPE llamacpp:predicted_tokens_seconds gauge\nllamacpp:predicted_tokens_seconds %.2f\n"
            "# HELP llamacpp:n_busy_slots_per_decode Busy slots.\n# TYPE llamacpp:n_busy_slots_per_decode gauge\nllamacpp:n_busy_slots_per_decode %d\n"
            "# HELP qwfn:expert_cache_hit_rate Share of expert lookups served from VRAM or RAM.\n# TYPE qwfn:expert_cache_hit_rate gauge\nqwfn:expert_cache_hit_rate %.4f\n"
            "# HELP qwfn:expert_vram_rate Share of expert lookups served from VRAM.\n# TYPE qwfn:expert_vram_rate gauge\nqwfn:expert_vram_rate %.4f\n"
            "# HELP qwfn:expert_bytes_from_disk_total Bytes of experts read from the NVMe.\n# TYPE qwfn:expert_bytes_from_disk_total counter\nqwfn:expert_bytes_from_disk_total %llu\n",
            st["totals"]["prompt_tokens"].get<long long>(), st["totals"]["generated_tokens"].get<long long>(),
            st["prompt"]["tokens_per_second"].get<double>(), st["generation"]["tokens_per_second"].get<double>(),
            st["busy"].get<bool>() ? 1 : 0,
            st["expert_cache"]["hit_rate"].get<double>(), st["expert_cache"]["vram_served"].get<double>(),
            (unsigned long long) st["expert_cache"]["bytes_from_disk"].get<double>());
        res.set_content(buf, "text/plain; version=0.0.4");
    });

    svr.Post("/v1/chat/completions", [&](const httplib::Request & req, httplib::Response & res) {
        json body;
        try { body = json::parse(req.body); }
        catch (const std::exception & ex) { fail(res, 400, std::string("bad JSON: ") + ex.what()); return; }
        if (!body.contains("messages") || !body["messages"].is_array()) {
            fail(res, 400, "messages is required"); return;
        }

        // Thinking: either OpenAI-ish reasoning_effort, or the Qwen template's
        // own chat_template_kwargs.enable_thinking.
        std::string effort;
        { std::lock_guard<std::mutex> lk(S.props_mu); effort = body.value("reasoning_effort", S.def_effort); }
        if (body.contains("chat_template_kwargs")) {
            const auto & k = body["chat_template_kwargs"];
            if (k.contains("enable_thinking") && !k["enable_thinking"].get<bool>()) effort = "off";
        }
        if (!effort_valid(effort)) { fail(res, 400, "reasoning_effort must be xhigh|medium|low|off"); return; }

        // Qwen's soft switches, for harnesses that show no thinking toggle: a
        // trailing "/think" or "/no_think" in the last user message sets the
        // mode for this request and is stripped before the model sees it.
        json msgs = body["messages"];
        {
            auto strip_tag = [](std::string & txt) -> int {   // 0 = /no_think, 1 = /think, -1 = none
                const size_t e = txt.find_last_not_of(" \t\r\n");
                if (e == std::string::npos) return -1;
                std::string t = txt.substr(0, e + 1);
                for (int which = 0; which < 2; which++) {
                    const std::string tag = which ? "/think" : "/no_think";
                    if (t.size() >= tag.size() && t.compare(t.size() - tag.size(), tag.size(), tag) == 0 &&
                        (t.size() == tag.size() || isspace((unsigned char) t[t.size() - tag.size() - 1]))) {
                        txt = t.substr(0, t.size() - tag.size());
                        while (!txt.empty() && isspace((unsigned char) txt.back())) txt.pop_back();
                        return which;
                    }
                }
                return -1;
            };
            std::string def_now; { std::lock_guard<std::mutex> lk(S.props_mu); def_now = S.def_effort; }
            for (int m = (int) msgs.size() - 1; m >= 0; m--) {
                if (msgs[m].value("role", "") != "user") continue;
                int r = -1;
                if (msgs[m]["content"].is_string()) {
                    std::string c = msgs[m]["content"].get<std::string>();
                    if ((r = strip_tag(c)) >= 0) msgs[m]["content"] = c;
                } else if (msgs[m]["content"].is_array()) {
                    for (auto & part : msgs[m]["content"])
                        if (part.value("type", "") == "text") {
                            std::string c = part.value("text", "");
                            if ((r = strip_tag(c)) >= 0) { part["text"] = c; break; }
                        }
                }
                if (r >= 0) effort = r == 0 ? "off" : (def_now != "off" ? def_now : "xhigh");
                break;
            }
        }
        const bool thinking = effort != "off";

        sampler smp;
        int max_tok = 0;
        { std::lock_guard<std::mutex> lk(S.props_mu); smp.cfg = thinking ? S.preset_think : S.preset_nothink; max_tok = S.def_max_tokens; }
        smp.cfg.from_json(body);               // any sampling field in the request wins
        if (body.contains("seed") && body["seed"].is_number_integer())
            smp.rng.seed((unsigned) body["seed"].get<long long>());
        else smp.rng.seed((unsigned) std::chrono::steady_clock::now().time_since_epoch().count());
        if (body.contains("max_completion_tokens")) max_tok = body["max_completion_tokens"].get<int>();
        else if (body.contains("max_tokens"))       max_tok = body["max_tokens"].get<int>();
        const bool timings_per_token = body.value("timings_per_token", false);
        // Tools: rendered into the system turn unless tool_choice is "none";
        // "required" or a named function forces the call's opening.
        json tools = body.contains("tools") && body["tools"].is_array() ? body["tools"] : json::array();
        std::string forced;
        if (body.contains("tool_choice")) {
            const json & tc = body["tool_choice"];
            if (tc.is_string()) {
                if (tc == "none") tools = json::array();
                // "required" opens the block and stops there: a prefix ending in
                // "<function=" ends on a token boundary the model never produces
                // and it continued with a call id ("call_5067") as the name.
                else if (tc == "required" && !tools.empty()) forced = "<tool_call>\n";
            } else if (tc.is_object() && tc.contains("function") && !tools.empty()) {
                forced = "<tool_call>\n<function=" + tc["function"].value("name", "") + ">\n";
            }
        }
        const std::string tools_block = render_tools_block(tools);
        int reasoning_budget; { std::lock_guard<std::mutex> lk(S.props_mu); reasoning_budget = S.def_reasoning_budget; }
        if (body.contains("reasoning_budget")) reasoning_budget = body["reasoning_budget"].get<int>();

        std::vector<std::string> stops;
        if (body.contains("stop")) {
            if (body["stop"].is_string()) stops.push_back(body["stop"].get<std::string>());
            else for (const auto & s : body["stop"]) stops.push_back(s.get<std::string>());
        }
        const bool stream = body.value("stream", false);
        // Unique per request: the tool-call ids a harness keys its cards and
        // replays on are derived from it, and two rounds of one conversation used
        // to hand out the same "call_0_<seconds>" id.
        static std::atomic<long long> req_seq{0};
        const std::string id = "chatcmpl-" + std::to_string(now_unix()) + "-" + std::to_string(++req_seq);

        // One generation at a time. For a streamed response the lock must
        // outlive this handler: the chunked provider runs after it returns, and
        // a second request arriving mid-stream (a harness generating a title,
        // the next turn) must wait, not enter the engine. So the lock is shared
        // with the provider and released when the stream is done.
        // One line per request, so a harness's exact ask is visible in the log.
        size_t n_images = 0;
        for (const auto & m : msgs)
            if (m.contains("content") && m["content"].is_array())
                for (const auto & part : m["content"]) { const std::string ty = part.value("type", ""); if (ty == "image_url" || ty == "input_image") n_images++; }
        fprintf(stderr, "[qwfn-server] request from %s: %zu messages, %zu images, stream=%d, max_tokens=%d, effort=%s, budget=%d, temp=%.2f, timings_per_token=%d, tools=%zu%s, keys:",
                req.remote_addr.c_str(), msgs.size(), n_images, (int) stream, max_tok, effort.c_str(), reasoning_budget, smp.cfg.temp, (int) timings_per_token,
                tools.size(), forced.empty() ? "" : " (forced)");
        for (auto it = body.begin(); it != body.end(); ++it) fprintf(stderr, " %s", it.key().c_str());
        fprintf(stderr, "\n");

        auto lk = std::make_shared<std::unique_lock<std::mutex>>(S.mu);

        server::prompt P;
        std::string e;
        if (!build_prompt(msgs, effort, thinking, tools_block, forced, P, e)) { fail(res, 400, e); return; }

        if (!stream) {
            gen_result R;
            if (!generate(P, smp, max_tok, thinking, stops,
                          [](const std::string &, bool) { return true; }, R, e, reasoning_budget)) {
                S.last_msgs = json();   // cache is no longer trustworthy
                const bool client = e.rfind("context_length_exceeded", 0) == 0;
                fail(res, client ? 400 : 500, e,
                     client ? "invalid_request_error" : "server_error");
                return;
            }
            S.last_msgs = msgs;
            json calls; const std::string text = parse_tool_calls(forced + R.content, tools, calls);
            S.last_tool_key = tool_calls_key(calls); S.last_content = calls.empty() ? R.content : text;
            json msg{{"role", "assistant"}, {"content", calls.empty() ? json(R.content) : (text.empty() ? json(nullptr) : json(text))}};
            if (!calls.empty()) { msg["tool_calls"] = calls; R.finish = "tool_calls"; }
            if (!R.reasoning.empty()) msg["reasoning_content"] = R.reasoning;
            res.set_content(json{
                {"id", id}, {"object", "chat.completion"}, {"created", now_unix()},
                {"model", S.model_id},
                {"choices", json::array({ json{
                    {"index", 0}, {"message", msg}, {"finish_reason", R.finish}} })},
                {"usage", {{"prompt_tokens", R.n_prompt},
                           {"completion_tokens", R.n_gen},
                           {"total_tokens", R.n_prompt + R.n_gen}}},
                {"timings", S.live.timings()}
            }.dump(2, ' ', false, json::error_handler_t::replace), "application/json");
            return;
        }

        // Streaming: the provider owns the engine lock until it is done.
        res.set_header("Cache-Control", "no-cache");
        res.set_chunked_content_provider("text/event-stream",
            [&, id, P, smp, max_tok, thinking, stops, timings_per_token, lk, reasoning_budget, tools, forced,
             msgs_copy = msgs](size_t, httplib::DataSink & sink) mutable {
                bool write_failed = false;
                auto last_write = clk::now();
                auto send = [&](const json & j) {
                    const std::string s = "data: " + json_dump(j) + "\n\n";
                    const bool ok = sink.write(s.data(), s.size());
                    last_write = clk::now();
                    return ok;
                };
                // A silent stretch -- a long prefill, a tool call being written,
                // a held-back parameter -- still has to put bytes on the wire:
                // Unsloth Studio's proxy reads the stream with a 300 s per-read
                // timeout and reports "Timeout waiting for custom response" when
                // nothing arrives, with the generation still running here. An
                // SSE comment line is ignored by every client (Studio relays
                // non-data lines untouched).
                auto tick = [&]() {
                    if (write_failed || since(last_write) < 15.0) return;
                    static const std::string ka = ": keepalive\n\n";
                    if (!sink.write(ka.data(), ka.size())) write_failed = true;
                    last_write = clk::now();
                };
                std::string held;   // incomplete UTF-8 tail of the previous delta, per stream
                auto complete = [&](std::string piece) {   // returns what may be emitted now
                    piece = held + piece; held.clear();
                    const size_t t = utf8_incomplete_tail(piece);
                    if (t) { held = piece.substr(piece.size() - t); piece.erase(piece.size() - t); }
                    return piece;
                };
                send(json{{"id", id}, {"object", "chat.completion.chunk"},
                          {"created", now_unix()}, {"model", S.model_id},
                          {"choices", json::array({ json{
                              {"index", 0},
                              {"delta", {{"role", "assistant"}}},
                              {"finish_reason", nullptr}} })}});

                gen_result R;
                std::string e2;
                // Reasoning deltas are coalesced (every 100 ms or 16 tokens): a
                // 15-minute think is ~13,000 tokens, and a UI re-rendering its
                // reasoning block per chunk is what stalls the socket.
                std::string rbuf; int rcount = 0; auto rlast = clk::now();
                auto emit = [&](const json & d) {
                    json chunk{{"id", id}, {"object", "chat.completion.chunk"},
                               {"created", now_unix()}, {"model", S.model_id},
                               {"choices", json::array({ json{
                                   {"index", 0}, {"delta", d},
                                   {"finish_reason", nullptr}} })}};
                    if (timings_per_token) chunk["timings"] = S.live.timings();   // live tok/s per chunk
                    if (!send(chunk)) { write_failed = true; return false; }
                    return true;
                };
                auto flush_reasoning = [&]() {
                    if (rbuf.empty()) return true;
                    const bool ok = emit(json{{"reasoning_content", rbuf}});
                    rbuf.clear(); rcount = 0; rlast = clk::now();
                    return ok;
                };
                // Tool calls are never streamed as text: content before the first
                // <tool_call> streams normally (holding back a possible partial
                // tag); a block streams as tool_calls deltas while it is written
                // (see tool_streamer) and the text after it streams as content.
                std::string tacc = forced; size_t temitted = 0; bool tool_mode = !forced.empty();
                int n_calls = 0;
                std::unique_ptr<tool_streamer> ts;
                const bool with_tools = tools.is_array() && !tools.empty();
                const std::string mark = TC_MARK;
                auto open_streamer = [&](size_t block_start) {
                    ts.reset(new tool_streamer(tools, block_start, n_calls, "call_" + std::to_string(n_calls) + "_" + id.substr(9),
                                               [&](const json & d) { return emit(json{{"tool_calls", json::array({d})}}); }));
                    n_calls++;
                };
                if (tool_mode) open_streamer(0);
                auto drain_content = [&]() {
                    if (!with_tools) return emit(json{{"content", tacc.substr(temitted)}}) && (temitted = tacc.size(), true);
                    while (true) {
                        if (!tool_mode) {
                            // A real block start, or a bare tag that is still undecided
                            // (not enough characters yet to tell it from the marker)?
                            size_t p = tacc.find(TC_OPEN, temitted); bool undecided = false;
                            while (p != std::string::npos) {
                                if (tacc.size() - p < mark.size()) { undecided = true; break; }
                                if (tacc.compare(p, mark.size(), mark) == 0) break;
                                p = tacc.find(TC_OPEN, p + 1);          // prose: keep looking
                            }
                            if (p != std::string::npos && !undecided) {
                                std::string t = tacc.substr(temitted, p - temitted);
                                while (!t.empty() && (t.back() == '\n' || t.back() == ' ')) t.pop_back();
                                if (!t.empty() && !emit(json{{"content", t}})) return false;
                                temitted = p; tool_mode = true; open_streamer(p);
                            } else {
                                // Emit everything except a tail that could still become the marker.
                                size_t hold = undecided ? tacc.size() - p : 0;
                                if (!undecided)
                                    for (size_t k = std::min(mark.size() - 1, tacc.size() - temitted); k > 0; k--)
                                        if (mark.compare(0, k, tacc, tacc.size() - k, k) == 0) { hold = k; break; }
                                const std::string t = tacc.substr(temitted, tacc.size() - temitted - hold);
                                if (!t.empty() && !emit(json{{"content", t}})) return false;
                                temitted = tacc.size() - hold;
                                return true;
                            }
                        }
                        if (!ts->feed(tacc)) return false;
                        const size_t close = tacc.find(TC_CLOSE, temitted);
                        if (close == std::string::npos) return true;
                        if (ts->opened && ts->st != tool_streamer::CLOSED && !ts->close_object()) return false;
                        ts.reset();
                        temitted = close + 12;
                        tool_mode = false;   // text after a block (the template forbids it, models do it) streams as content
                    }
                };
                bool ok = false; std::string aborted;
                try {
                    ok = generate(P, smp, max_tok, thinking, stops,
                        [&](const std::string & piece_in, bool is_reasoning) {
                            const std::string piece = complete(piece_in);
                            if (piece.empty()) return true;
                            if (is_reasoning) {
                                rbuf += piece; rcount++;
                                if (rcount < 16 && since(rlast) < 0.1) return true;
                                return flush_reasoning();
                            }
                            if (!flush_reasoning()) return false;
                            tacc += piece;
                            return drain_content();
                        }, R, e2, reasoning_budget, tick);
                    if (ok) {
                        flush_reasoning();
                        if (tool_mode && ts) {
                            // The reply ended inside a block. Closed with </function>:
                            // a complete call, the batch parser below agrees. Cut off
                            // mid-way: leave the fragment as it is -- a client that
                            // cannot parse the arguments describes the call instead
                            // of running it, which is the right outcome for a call
                            // the model never finished. Never a name: plain text.
                            ts->feed(tacc);
                            if (!ts->opened && temitted < tacc.size()) emit(json{{"content", tacc.substr(temitted)}});
                            temitted = tacc.size();
                        }
                        if (temitted < tacc.size()) { emit(json{{"content", tacc.substr(temitted)}}); temitted = tacc.size(); }
                    }
                } catch (const std::exception & ex) {
                    aborted = ex.what();
                } catch (...) {
                    aborted = "unknown exception";
                }
                if (!aborted.empty()) {
                    // The engine's history is unknown from here: drop the prefix
                    // cache so the next request starts clean instead of running
                    // on a desynchronised state.
                    fprintf(stderr, "[qwfn-server] %s: stream aborted by an exception after %d generated tokens: %s\n", id.c_str(), R.n_gen, aborted.c_str());
                    S.last_msgs = json(); S.consumed.clear(); S.eng.reset(); S.eng.clear_embeddings();
                    { std::lock_guard<std::mutex> lg(S.live.mu); S.live.busy = false; }
                    send(json{{"error", {{"message", "stream aborted: " + aborted}, {"type", "server_error"}}}});
                    const std::string done = "data: [DONE]\n\n";
                    sink.write(done.data(), done.size());
                    sink.done();
                    lk->unlock();
                    return true;
                }

                if (ok) {
                    json calls; const std::string text = parse_tool_calls(forced + R.content, tools, calls);
                    S.last_tool_key = tool_calls_key(calls); S.last_content = calls.empty() ? R.content : text;
                    if (!calls.empty()) R.finish = "tool_calls";
                }
                if (write_failed) {
                    fprintf(stderr, "[qwfn-server] %s: client stopped reading after %d prompt + %d generated tokens (%.0f s); stream dropped\n",
                            id.c_str(), R.n_prompt, R.n_gen, R.t_prompt + R.t_gen);
                } else if (!ok) {
                    fprintf(stderr, "[qwfn-server] %s: generation failed after %d tokens: %s\n", id.c_str(), R.n_gen, e2.c_str());
                } else {
                    fprintf(stderr, "[qwfn-server] %s: prompt %d tok %.1f tok/s | generated %d tok (%zu reasoning chars%s) in %.1f s, %.1f tok/s, finish %s%s\n",
                            id.c_str(), R.n_prompt, R.t_prompt > 0 ? R.n_prompt / R.t_prompt : 0.0, R.n_gen, R.reasoning.size(),
                            R.reasoning_budget_hit ? ", budget hit" : "", R.t_gen, R.t_gen > 0 ? R.n_gen / R.t_gen : 0.0, R.finish.c_str(),
                            R.n_pairs ? (" | drafts: " + std::to_string(R.n_accepted) + " of " + std::to_string(R.n_pairs) + " pairs accepted").c_str() : "");
                }
                if (!ok) {
                    S.last_msgs = json();
                    send(json{{"error", {{"message", e2}, {"type", "server_error"}}}});
                } else {
                    S.last_msgs = msgs_copy;
                    send(json{{"id", id}, {"object", "chat.completion.chunk"},
                              {"created", now_unix()}, {"model", S.model_id},
                              {"choices", json::array({ json{
                                  {"index", 0}, {"delta", json::object()},
                                  {"finish_reason", R.finish}} })},
                              {"usage", {{"prompt_tokens", R.n_prompt},
                                         {"completion_tokens", R.n_gen},
                                         {"total_tokens", R.n_prompt + R.n_gen}}},
                              {"timings", S.live.timings()}});
                }
                const std::string done = "data: [DONE]\n\n";
                sink.write(done.data(), done.size());
                sink.done();
                lk->unlock();
                // true: the body is complete (done() wrote the trailer). false
                // would be "cancelled" to httplib, which then closes a keep-alive
                // connection the client is about to reuse.
                return true;
            });
    });

    // Raw completion: no chat framing at all, for perplexity-style harnesses.
    svr.Post("/v1/completions", [&](const httplib::Request & req, httplib::Response & res) {
        json body;
        try { body = json::parse(req.body); }
        catch (const std::exception & ex) { fail(res, 400, std::string("bad JSON: ") + ex.what()); return; }
        const std::string p = body.value("prompt", "");
        sampler smp;
        { std::lock_guard<std::mutex> lk(S.props_mu); smp.cfg = S.preset_nothink; }
        smp.cfg.from_json(body);
        int max_tok = body.value("max_tokens", 128);
        std::vector<std::string> stops;
        if (body.contains("stop")) {
            if (body["stop"].is_string()) stops.push_back(body["stop"].get<std::string>());
            else for (const auto & s : body["stop"]) stops.push_back(s.get<std::string>());
        }

        std::lock_guard<std::mutex> lk(S.mu);
        S.last_msgs = json();            // raw completions break the chat chain
        server::prompt P;
        P.tok = S.vb.encode(p, false, true);
        gen_result R;
        std::string e;
        if (!generate(P, smp, max_tok, /*thinking=*/false, stops,
                      [](const std::string &, bool) { return true; }, R, e)) {
            fail(res, 500, e, "server_error"); return;
        }
        res.set_content(json{
            {"id", "cmpl-" + std::to_string(now_unix())}, {"object", "text_completion"},
            {"created", now_unix()}, {"model", S.model_id},
            {"choices", json::array({ json{
                {"index", 0}, {"text", R.reasoning + R.content},
                {"finish_reason", R.finish}} })},
            {"usage", {{"prompt_tokens", R.n_prompt}, {"completion_tokens", R.n_gen},
                       {"total_tokens", R.n_prompt + R.n_gen}}}
        }.dump(2, ' ', false, json::error_handler_t::replace), "application/json");
    });

    fprintf(stderr, "qwfn-server listening on http://%s:%d  (model id: %s%s)\n",
            host.c_str(), port, S.model_id.c_str(),
            S.vis.loaded() ? ", vision enabled" : "");
    if (!svr.listen(host.c_str(), port)) {
        fprintf(stderr, "failed to bind %s:%d\n", host.c_str(), port);
        return 1;
    }
    return 0;
}
