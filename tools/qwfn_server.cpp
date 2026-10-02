// qwfn-server -- an HTTP front end speaking the OpenAI chat API and the
// Anthropic Messages API (Claude Code).
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

    // The k largest logits, descending. A partial_sort over the whole vocabulary
    // (248K entries) cost 1.5-3 ms per call and a sampled pair step makes three
    // or four of them; instead two linear passes -- the max, then every logit
    // within 40 temperatures of it (a relative probability of e^-40, nothing
    // at float precision) -- and a partial_sort over the few that survive.
    // Same k best, same order, whenever at least k survive; when fewer do,
    // the ones left out had zero probability anyway.
    void top_indices(const float * lg, int64_t n, int k, std::vector<int> & idx) const {
        float mx = lg[0];
        for (int64_t v = 1; v < n; v++) mx = std::max(mx, lg[v]);
        const float margin = 40.0f * std::max(cfg.temp, 1e-3f);
        const float floor_ = mx - margin;
        idx.clear();
        for (int64_t v = 0; v < n; v++) if (lg[v] >= floor_) idx.push_back((int) v);
        const int kk = (int) std::min<size_t>((size_t) k, idx.size());
        std::partial_sort(idx.begin(), idx.begin() + kk, idx.end(), [&](int a, int b) { return lg[a] > lg[b]; });
        idx.resize(kk);
    }

    // The distribution pick() samples from -- penalties, top-k, temperature,
    // min-p, top-p -- as (token, probability) over the kept candidates. Empty
    // at temperature 0 (greedy: pick() takes the argmax).
    std::vector<std::pair<int32_t, float>> dist(const float * lg_in, int64_t n) {
        std::vector<std::pair<int32_t, float>> out;
        if (cfg.temp <= 0.0f) return out;
        std::vector<float> pen;
        const float * lg = lg_in;
        const bool penalise = cfg.repeat_last_n != 0 && !gen.empty() &&
            (cfg.presence_penalty != 0.0f || cfg.frequency_penalty != 0.0f || cfg.repeat_penalty != 1.0f);
        if (penalise) {
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
        std::vector<int> idx;
        top_indices(lg, n, (int) std::min<int64_t>(cfg.top_k > 0 ? cfg.top_k : n, n), idx);
        const int k = (int) idx.size();
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
        out.reserve(keep);
        for (int i = 0; i < keep; i++) out.emplace_back(idx[i], (float) (p[i] / cum));
        return out;
    }
    int32_t sample(const std::vector<std::pair<int32_t, float>> & d) {
        std::uniform_real_distribution<double> U(0.0, 1.0);
        double r = U(rng), acc = 0;
        for (const auto & [t, p] : d) { acc += p; if (r <= acc) return t; }
        return d.empty() ? 0 : d.back().first;
    }
    static float prob_of(const std::vector<std::pair<int32_t, float>> & d, int32_t t) {
        for (const auto & [x, p] : d) if (x == t) return p;
        return 0.0f;
    }

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
        std::vector<int> idx;
        top_indices(lg, n, (int) std::min<int64_t>(cfg.top_k > 0 ? cfg.top_k : n, n), idx);
        const int k = (int) idx.size();
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

// The reply text a client echoes back may differ from what the server kept in
// trailing whitespace: a streamed reply emitted "OK\n\n" before its tool block
// was recognised while parse_tool_calls() keeps the text trimmed, and clients
// trim on their own. Matching modulo trailing whitespace is what keeps the
// prefix continuation alive across a tool-calling turn.
static bool same_reply_text(const std::string & a, const std::string & b) {
    size_t na = a.size(), nb = b.size();
    while (na > 0 && (a[na - 1] == '\n' || a[na - 1] == ' ' || a[na - 1] == '\r' || a[na - 1] == '\t')) na--;
    while (nb > 0 && (b[nb - 1] == '\n' || b[nb - 1] == ' ' || b[nb - 1] == '\r' || b[nb - 1] == '\t')) nb--;
    return na == nb && a.compare(0, na, b, 0, nb) == 0;
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
    // The current (or, when idle, the last) request: its whole prompt, the part reused
    // from the engine's prefix, the part being prefilled and the tokens generated.
    int    n_input = 0, n_cached = 0, n_prompt = 0, n_gen = 0;
    double t_prompt = 0, t_gen = 0;          // seconds, the current or last request
    // Prefill progress: new prompt tokens done so far, fractional inside a batch (the
    // engine reports each layer of a streamed batch), and the clock it runs on, so a
    // reader sees the rate move while the prefill is still going.
    bool   prefilling = false;
    double prompt_done = 0, prompt_base = 0;
    std::chrono::steady_clock::time_point t_prompt0;
    double t_prompt_total = 0, t_gen_total = 0;
    long long n_prompt_total = 0, n_gen_total = 0, n_requests = 0, n_input_total = 0, n_cached_total = 0;
    long long n_pairs_total = 0, n_accepted_total = 0, n_drafted_total = 0;   // the draft head's verify steps, drafts accepted, drafts proposed
    int    n_past = 0;
    // The last request that completed, kept apart from the live counters so a monitor
    // can show it while the next one runs.
    struct snapshot {
        int n_input = 0, n_cached = 0, n_prompt = 0, n_gen = 0, n_pairs = 0, n_accepted = 0, n_drafted = 0;
        double t_prompt = 0, t_gen = 0; std::string finish; long long when = 0; bool ok = true;
    } last;
    bool have_last = false;
    double prompt_seconds_locked() const {   // mu held
        if (prefilling) return std::chrono::duration<double>(std::chrono::steady_clock::now() - t_prompt0).count();
        return t_prompt;
    }
    json timings() {   // llama.cpp's field names, plus the prompt's cached share and live progress
        std::lock_guard<std::mutex> lk(mu);
        const double tp = prompt_seconds_locked();
        const double done = prefilling ? prompt_done : (double) n_prompt;
        return {{"prompt_n", n_prompt}, {"prompt_ms", tp * 1e3},
                {"prompt_per_second", tp > 0.05 ? done / tp : 0.0},
                {"prompt_cached_n", n_cached}, {"prompt_done_n", done}, {"prompt_input_n", n_input}, {"prefilling", prefilling},
                {"predicted_n", n_gen}, {"predicted_ms", t_gen * 1e3},
                // The first token comes straight from the prompt's logits at ~0 ms:
                // no rate until the clock has something to divide by.
                {"predicted_per_second", t_gen >= 0.1 ? n_gen / t_gen : 0.0}};
    }
    json last_json() {
        std::lock_guard<std::mutex> lk(mu);
        if (!have_last) return nullptr;
        return {{"input_tokens", last.n_input}, {"cached_tokens", last.n_cached}, {"prompt_tokens", last.n_prompt}, {"generated_tokens", last.n_gen},
                {"prompt_ms", last.t_prompt * 1e3}, {"prompt_tokens_per_second", last.t_prompt > 0 ? last.n_prompt / last.t_prompt : 0.0},
                {"generation_ms", last.t_gen * 1e3}, {"tokens_per_second", last.t_gen > 0 ? last.n_gen / last.t_gen : 0.0},
                {"finish_reason", last.finish}, {"ok", last.ok}, {"completed_at", last.when},
                {"speculative", {{"pairs", last.n_pairs}, {"accepted", last.n_accepted}, {"drafted", last.n_drafted}}}};
    }
};

// ---- server state -----------------------------------------------------------
// content parts -> text, with images encoded to embeddings on the way past
struct pending_img { std::vector<float> emb; int n_tok = 0; };

struct server {
    model_index    mi;
    uint32_t       mtp_drafts = 1;   // --mtp-drafts: the most drafts a verify step carries
    engine         eng;
    qwfn::vocab    vb;
    vision_encoder vis;
    ggml_backend_t vis_backend = nullptr;   // the CPU backend the projector runs on
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
    std::string  dump_dir;                   // POST /props {"dump_requests": DIR}: every request body is written there, for harness debugging
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
            // On the CPU: no VRAM is borrowed from the expert tier for this.
            if (!S.vis.encode(img, pi.emb, pi.n_tok, gw, gh, err)) return false;
            fprintf(stderr, "[qwfn-server] image: %zu bytes, %dx%d px -> %dx%d grid, %d tokens, encoded in %.2f s\n",
                    raw.size(), img.nx, img.ny, gw, gh, pi.n_tok, since(t0));
            imgs.emplace_back(text.size(), std::move(pi));   // marker position in text
            continue;
        }
    }
    return true;
}

// ---- the Anthropic Messages API ---------------------------------------------
// POST /v1/messages is what Claude Code (and the Anthropic SDKs) speak when
// ANTHROPIC_BASE_URL points here. A request is translated into the OpenAI
// shape the chat path handles -- system + messages + tools + tool_choice --
// generated by the same code, and the reply goes back as Anthropic content
// blocks and stream events. Fields with no counterpart here are accepted and
// ignored: cache_control, metadata, context_management, output_config.format,
// the beta tool fields (strict, defer_loading). Server tools (a `type` other
// than "custom", no input_schema) are dropped: nothing here would run them.
static std::string anthropic_text(const json & c) {   // a string, or the text blocks of a list
    if (c.is_string()) return c.get<std::string>();
    std::string s;
    if (c.is_array())
        for (const auto & b : c)
            if (b.value("type", "") == "text") { if (!s.empty()) s += "\n\n"; s += b.value("text", ""); }
    return s;
}
// A list of parts that are all text collapses to one string: Claude Code sends
// its newest message as a one-block list carrying cache_control and replays it
// as a plain string, and the prefix continuation compares the messages as sent.
static json compact_parts(const json & parts) {
    std::string s; bool all_text = parts.is_array();
    if (all_text) for (const auto & p : parts) { if (p.value("type", "") != "text") { all_text = false; break; } if (!s.empty()) s += "\n\n"; s += p.value("text", ""); }
    return all_text ? json(s) : parts;
}
// One Anthropic content block as an OpenAI content part; null for a block
// with nothing to render here (a tool reference, a server tool's result).
static bool anthropic_part(const json & b, json & part, std::string & err) {
    part = nullptr;
    const std::string ty = b.value("type", "");
    if (ty == "text") { part = json{{"type", "text"}, {"text", b.value("text", "")}}; return true; }
    const json src = b.contains("source") && b["source"].is_object() ? b["source"] : json::object();
    if (ty == "image") {
        if (src.value("type", "") != "base64") { err = "image sources must be base64 (this server does not fetch URLs)"; return false; }
        part = json{{"type", "image_url"}, {"image_url", {{"url", "data:" + src.value("media_type", "image/png") + ";base64," + src.value("data", "")}}}};
        return true;
    }
    if (ty == "document") {
        // A text document reads as text; anything else (a PDF) is named, so the
        // model knows what it was not given, instead of failing the turn.
        const std::string text = src.value("type", "") == "text" ? src.value("data", "")
                               : "[document not available: " + src.value("media_type", "unknown type") + " is not supported by this server]";
        part = json{{"type", "text"}, {"text", text}};
        return true;
    }
    return true;
}
// The request as the chat path takes it. Thinking: "disabled" turns it off;
// "enabled" turns it on at the server's effort (xhigh when the server's default
// is off) with budget_tokens as a cap under the server's own; "adaptive" (what
// Claude Code sends a model it does not know) leaves the server's default.
// output_config.effort low / medium / max maps onto low / medium / xhigh and
// "high" (what Claude Code sends by default) keeps the server's level; it only
// modulates a thinking that is on.
static bool anthropic_to_openai(const json & a, const std::string & def_effort, int def_budget, json & o, std::string & err) {
    if (!a.contains("messages") || !a["messages"].is_array()) { err = "messages: field required"; return false; }
    o = json::object();
    json msgs = json::array();
    if (a.contains("system")) {
        const std::string sys = anthropic_text(a["system"]);
        if (!sys.empty()) msgs.push_back(json{{"role", "system"}, {"content", sys}});
    }
    for (const auto & m : a["messages"]) {
        const std::string role = m.value("role", "user");
        const json c = m.contains("content") ? m["content"] : json("");
        if (role == "assistant") {
            // text, thinking and tool_use blocks -> content, reasoning_content, tool_calls
            std::string text, reasoning; json calls = json::array();
            if (c.is_string()) text = c.get<std::string>();
            else if (c.is_array())
                for (const auto & b : c) {
                    const std::string ty = b.value("type", "");
                    if (ty == "text")          text += b.value("text", "");
                    else if (ty == "thinking") reasoning += b.value("thinking", "");
                    else if (ty == "tool_use")
                        calls.push_back(json{{"id", b.value("id", "")}, {"type", "function"},
                                             {"function", {{"name", b.value("name", "")},
                                                           {"arguments", (b.contains("input") && b["input"].is_object() ? b["input"] : json::object()).dump()}}}});
                }
            json msg{{"role", "assistant"}, {"content", text}};
            if (!reasoning.empty()) msg["reasoning_content"] = reasoning;
            if (!calls.empty()) msg["tool_calls"] = calls;
            msgs.push_back(msg);
            continue;
        }
        // A user turn (or a system entry mid-conversation): each tool_result is
        // a tool message, the other blocks one message of parts, in the order sent.
        if (c.is_string()) { msgs.push_back(json{{"role", role}, {"content", c}}); continue; }
        if (!c.is_array()) { err = "messages[].content must be a string or a list of blocks"; return false; }
        json parts = json::array();
        auto flush = [&]() { if (!parts.empty()) { msgs.push_back(json{{"role", role}, {"content", compact_parts(parts)}}); parts = json::array(); } };
        for (const auto & b : c) {
            if (b.value("type", "") == "tool_result") {
                flush();
                const json rc = b.contains("content") ? b["content"] : json("");
                json content = json::array();
                if (rc.is_string()) content = rc;
                else if (rc.is_array())
                    for (const auto & rb : rc) { json p; if (!anthropic_part(rb, p, err)) return false; if (!p.is_null()) content.push_back(p); }
                msgs.push_back(json{{"role", "tool"}, {"tool_call_id", b.value("tool_use_id", "")}, {"content", compact_parts(content)}});
                continue;
            }
            json p;
            if (!anthropic_part(b, p, err)) return false;
            if (!p.is_null()) parts.push_back(p);
        }
        flush();
    }
    o["messages"] = msgs;

    if (a.contains("tools") && a["tools"].is_array()) {
        json tools = json::array();
        for (const auto & t : a["tools"]) {
            if (!t.is_object() || !t.contains("input_schema") || (t.contains("type") && t["type"] != "custom")) continue;
            json fn{{"name", t.value("name", "")}};
            if (t.contains("description")) fn["description"] = t["description"];
            fn["parameters"] = t["input_schema"];
            tools.push_back(json{{"type", "function"}, {"function", fn}});
        }
        o["tools"] = tools;
    }
    if (a.contains("tool_choice") && a["tool_choice"].is_object()) {
        const json & tc = a["tool_choice"];
        const std::string ty = tc.value("type", "auto");
        if (ty == "any")       o["tool_choice"] = "required";
        else if (ty == "none") o["tool_choice"] = "none";
        else if (ty == "tool") o["tool_choice"] = json{{"type", "function"}, {"function", {{"name", tc.value("name", "")}}}};
    }
    for (const char * k : {"max_tokens", "temperature", "top_p", "top_k", "stream"}) if (a.contains(k)) o[k] = a[k];
    if (a.contains("stop_sequences")) o["stop"] = a["stop_sequences"];

    std::string effort;
    if (a.contains("thinking") && a["thinking"].is_object()) {
        const json & t = a["thinking"];
        const std::string ty = t.value("type", "adaptive");
        if (ty == "disabled") effort = "off";
        else if (ty == "enabled") {
            effort = def_effort != "off" ? def_effort : "xhigh";
            if (t.contains("budget_tokens") && t["budget_tokens"].is_number_integer()) {
                const int b = t["budget_tokens"].get<int>();
                if (b > 0) o["reasoning_budget"] = def_budget > 0 ? std::min(b, def_budget) : b;
            }
        }
    }
    const bool on = effort.empty() ? def_effort != "off" : effort != "off";
    if (on && a.contains("output_config") && a["output_config"].is_object() &&
        a["output_config"].contains("effort") && a["output_config"]["effort"].is_string()) {
        const std::string lv = a["output_config"]["effort"].get<std::string>();
        effort = lv == "low" ? "low" : lv == "medium" ? "medium" : lv == "max" ? "xhigh" : effort.empty() ? def_effort : effort;
    }
    if (!effort.empty()) o["reasoning_effort"] = effort;
    return true;
}
static const char * anthropic_stop_reason(const std::string & finish, const std::string & stop_seq) {
    if (finish == "tool_calls") return "tool_use";
    if (finish == "length")     return "max_tokens";
    return stop_seq.empty() ? "end_turn" : "stop_sequence";
}
static std::string tool_use_id(const std::string & id) {   // "call_N_stamp" -> "toolu_N_stamp"
    return id.rfind("call_", 0) == 0 ? "toolu_" + id.substr(5) : id;
}
// A thinking block carries an opaque signature the client hands back
// unchanged; nothing here verifies it, but the block is well formed with one.
static std::string thinking_signature(const std::string & t) {
    char b[32];
    snprintf(b, sizeof b, "qwfn%016llx", (unsigned long long) hash_bytes(t.data(), t.size()));
    return b;
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr,
          "usage: qwfn-server <shard.gguf> [options]\n"
          "\n"
          "      --host HOST     bind address (default 127.0.0.1)\n"
          "      --port N        port (default 8080)\n"
          "      --mmproj PATH   vision projector gguf; enables image input. It runs on the CPU (weights in RAM, no VRAM\n"
          "                      at all): a 1400x1000 screenshot encodes in ~15 s, a 320x240 image in 0.3 s\n"
          "      --vision-threads N  threads for the image encode (default: --threads, the physical cores)\n"
          "      --state-host V  attention state in pinned host memory: none (default) | idx | kv,idx. The VRAM it held goes to\n"
          "                      the expert tier; costs ~0.35 ms/token (idx) or ~2 ms/token (kv,idx) of PCIe reads\n"
          "      --alias NAME    model id reported by /v1/models\n"
          "      --think LEVEL   default reasoning effort: xhigh|medium|low|off\n"
          "      --think-budget N  max reasoning tokens per answer (0 = unlimited); also POST /props {\"reasoning_budget\":N} or per request\n"
          "      --ctx N         context (default 32768)   --batch N (default 2048)\n"
          "      --ram GB        --vram GB   --threads N   --cpu   --kv f16|q8_0\n"
          "      --reserve MB    VRAM kept free after the expert tier is sized (default 768; raise it on a desktop GPU)\n"
          "      --ram-frac F    MemAvailable share the RAM tier may take (default 0.75)\n"
          "      --spec-ahead N  predict 1 or 2 layers ahead (default 2)\n"
          "      --no-prefill-overlap   single prefill staging buffer, saves ~1.8 GB RAM\n"
          "\n"
          "Endpoints: GET /health, GET /v1/models,\n"
          "           POST /v1/chat/completions (stream supported; timings_per_token:true adds live tok/s to every chunk),\n"
          "           POST /v1/messages, POST /v1/messages/count_tokens (the Anthropic Messages API: Claude Code with\n"
          "               ANTHROPIC_BASE_URL=http://127.0.0.1:PORT; thinking and tool_use blocks, streamed),\n"
          "           GET /props, POST /props (reasoning_effort, max_tokens, thinking/non_thinking sampling presets),\n"
          "           GET /stats (live tok/s, context, expert cache), GET /slots, GET /metrics (Prometheus),\n"
          "           POST /v1/completions\n");
        return 1;
    }

    std::string host = "127.0.0.1", mmproj_path, alias, def_effort = "xhigh";
    int vision_threads = 0;
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
        if (a == "--vision-threads" && i + 1 < argc) { vision_threads = atoi(next()); continue; }
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
        // Prompts up to this many new tokens go through the cache-batched decode
        // path (one batch, experts through the tiers) instead of the streamed
        // prefill, whose cost is a full expert sweep (~12 s on Q4) whatever T is.
        if (a == "--prefill-decode-max" && i + 1 < argc) { cfg.prefill_decode_max = (uint32_t) atoi(argv[++i]); continue; }
        if (a == "--gate-drop" && i + 1 < argc) { cfg.gate_drop = (float) atof(argv[++i]); continue; }
        if (a == "--mtp-drafts" && i + 1 < argc) { cfg.mtp_drafts = (uint32_t) std::max(1, std::min(3, atoi(argv[++i]))); continue; }
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
    S.n_ctx = cfg.n_ctx; S.n_batch = cfg.n_batch; S.mtp_drafts = cfg.mtp_drafts; S.def_effort = def_effort; S.def_reasoning_budget = def_reasoning_budget;
    S.model_file = argv[1];
    std::string err;

    fprintf(stderr, "loading tokenizer...\n");
    if (!S.vb.load(argv[1], err)) { fprintf(stderr, "error: %s\n", err.c_str()); return 1; }
    if (!S.mi.load(argv[1], err)) { fprintf(stderr, "error: %s\n", err.c_str()); return 1; }
    // The vision projector runs on the CPU backend (its weights in RAM, the
    // graph on the cores), so it takes no VRAM: nothing is reserved for it and
    // the expert tier is not lent while an image is encoded. Loaded after the
    // engine, which loads the ggml backends.
    if (!S.eng.init(&S.mi, nullptr, cfg,
                    std::string(getenv("HOME")) + "/.unsloth/llama.cpp/build/bin", err)) {
        fprintf(stderr, "engine init: %s\n", err.c_str()); return 1;
    }
    fprintf(stderr, "%s\n", S.eng.memory_summary().c_str());
    S.eng.prefill_progress = [&S](uint32_t il, uint32_t nl, int32_t T) {
        std::lock_guard<std::mutex> lk(S.live.mu);
        S.live.prompt_done = S.live.prompt_base + (double) T * (il + 1) / nl;
    };
    S.eng.set_mtp_logits(true);   // the draft is sampled from the head's distribution at temperature
    if (!mmproj_path.empty()) {
        S.vis_backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
        if (!S.vis_backend) { fprintf(stderr, "vision: no CPU backend\n"); return 1; }
        if (!S.vis.load(mmproj_path, S.vis_backend, ggml_backend_get_default_buffer_type(S.vis_backend), err)) {
            fprintf(stderr, "vision: %s\n", err.c_str()); return 1;
        }
        // Physical cores, like the engine's own workers: 8 threads encoded a
        // 1400x1000 screenshot in 14.2 s where 16 took 15.5 (SMT starves the GEMMs).
        S.vis.set_n_threads(vision_threads > 0 ? vision_threads : cfg.n_threads);
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
            // Parts in the order they were sent, as the template renders them:
            // the text up to each image, then <|vision_start|> <|image_pad|> x N
            // <|vision_end|> in its place. The pads are real tokens, so the
            // sequence and the PLE n-gram window stay well formed; only their
            // embeddings are replaced (see engine::set_embeddings). A tool result
            // carries them the same way (the screenshot a harness's tool took).
            auto app_parts = [&]() {
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
            };
            if (role == "tool") {
                if (prev_role != "tool") app(enc_sp("<|im_start|>user"));
                app(enc_sp("\n<tool_response>\n"));
                app_parts();
                app(enc_sp("\n</tool_response>"));
                if (next_role != "tool") app(enc_sp("<|im_end|>\n"));
                return true;
            }
            app(enc_sp("<|im_start|>" + role + "\n"));
            app_parts();
            app(enc_sp("<|im_end|>\n"));
            return true;
        };
        auto role_at = [&](size_t i) { return i < messages.size() ? messages[i].value("role", "user") : std::string(); };

        // A harness replays the whole conversation each turn. If this request is
        // the previous one plus (our reply, a new user turn), the prompt is the
        // previous prompt plus the exact tokens we generated plus the new turn --
        // no re-tokenising, so the engine can continue from where it stopped.
        const size_t nprev = S.last_msgs.is_array() ? S.last_msgs.size() : 0;
        if (nprev > 0 && !S.last_gen.empty() && messages.size() >= nprev + 2) {
            size_t diff = 0;
            while (diff < nprev && S.last_msgs[diff] == messages[diff]) diff++;
            const json & reply = messages[nprev];
            const bool text_same  = content_of(reply).is_string() && same_reply_text(content_of(reply).get<std::string>(), S.last_content);
            const bool calls_same = tool_calls_key(reply.value("tool_calls", json::array())) == S.last_tool_key;
            if (diff == nprev && reply.value("role", "") == "assistant" && text_same && calls_same) {
                P.tok = S.last_prompt;
                P.tok.insert(P.tok.end(), S.last_gen.begin(), S.last_gen.end());
                for (size_t m = nprev + 1; m < messages.size(); m++) {
                    const std::string role = messages[m].value("role", "user");
                    if (role == "assistant") { P.tok.clear(); break; }   // not a clean extension
                    if (!render_user_or_tool(messages[m], role_at(m - 1), role_at(m + 1), e)) return false;
                }
                if (!P.tok.empty()) { open_assistant(); return true; }
                P.splices.clear();   // fall through to a full rebuild
            } else if (diff < nprev) {
                // The harness changed something it had already sent, and the
                // recurrent state cannot be rewound to the change: the whole
                // history is re-prefilled. Named, so the cause can be found.
                fprintf(stderr, "[qwfn-server] prefix lost: message %zu of %zu (%s) is not what the previous request sent (%zu -> %zu bytes)\n",
                        diff, messages.size(), messages[diff].value("role", "?").c_str(), S.last_msgs[diff].dump().size(), messages[diff].dump().size());
            } else {
                fprintf(stderr, "[qwfn-server] prefix lost: the reply at message %zu came back different from what was generated (role %s, text %s, tool calls %s)\n",
                        nprev, reply.value("role", "?").c_str(), text_same ? "same" : "differs", calls_same ? "same" : "differ");
            }
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
                if (!S.last_gen.empty() && same_reply_text(text, S.last_content) && tool_calls_key(tcs) == S.last_tool_key &&
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
        std::string stop_seq;                                       // the stop sequence that ended the reply, when one did
        int n_input = 0, n_cached = 0, n_prompt = 0, n_gen = 0;   // whole prompt, reused prefix, prefilled, generated
        int n_pairs = 0, n_accepted = 0, n_drafted = 0;     // verify steps, drafts accepted, drafts proposed
        double t_prompt = 0, t_gen = 0;
        bool reasoning_budget_hit = false;
    };

    // The prefix the engine can continue from: everything it holds, when the
    // prompt strictly extends it (images included: the pads match any image of
    // the same size, so each one's embedding is compared by hash); else 0. A prompt
    // equal to it has no tail to produce logits from, so it re-prefills.
    auto prefix_reuse = [&](const server::prompt & P) -> int32_t {
        if (S.consumed.empty() || P.tok.size() <= S.consumed.size() ||
            !std::equal(S.consumed.begin(), S.consumed.end(), P.tok.begin())) return 0;
        for (const auto & sp : P.splices)
            if (sp.first < (int32_t) S.consumed.size()) {
                const auto it = S.consumed_img.find(sp.first);
                if (it == S.consumed_img.end() ||
                    it->second != hash_bytes(sp.second.data(), sp.second.size() * sizeof(float))) return 0;
            }
        return (int32_t) S.consumed.size();
    };

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
        if (prefix_reuse(P) == 0) {
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
        // From here on this call mutates the engine's sequence state. A failure
        // after this point can leave it inconsistent: the layer graphs advance the
        // DeltaNet scan as they run, while n_past_ is only bumped once the whole
        // eval succeeds, so a mid-eval error (a short expert read, say) leaves some
        // layers ahead of the token count. KV and the indexer survive that -- they
        // are positional and get overwritten -- but a scan cannot be rewound, so the
        // next request must not continue from it.
        bool state_dirty = false;
        struct dirty_guard {
            server & st; const bool & dirty; bool ok = false;
            ~dirty_guard() {
                if (ok || !dirty) return;
                st.last_msgs = json(); st.consumed.clear(); st.consumed_img.clear();
                st.eng.reset(); st.eng.clear_embeddings();
            }
        } dg{S, state_dirty};
        for (const auto & sp : P.splices) {
            if (sp.first < (int32_t) S.consumed.size()) continue;   // already evaluated (and verified above)
            state_dirty = true;
            S.eng.set_embeddings(sp.first, sp.second.data(),
                                 (int32_t) (sp.second.size() / 2560));
            S.consumed_img[sp.first] = hash_bytes(sp.second.data(), sp.second.size() * sizeof(float));
        }

        std::vector<int32_t> hist = P.tok;
        if ((int32_t) P.tok.size() > (int32_t) S.consumed.size()) state_dirty = true;   // a tail will be fed
        int32_t fed = (int32_t) S.consumed.size();
        const int32_t fed0 = fed;
        R.n_input  = (int32_t) hist.size();
        R.n_cached = fed;
        R.n_prompt = (int32_t) hist.size() - fed;
        { std::lock_guard<std::mutex> lk(S.live.mu); S.live.busy = true; S.live.n_input = R.n_input; S.live.n_cached = R.n_cached; S.live.n_prompt = R.n_prompt; S.live.n_gen = 0;
          S.live.t_prompt = 0; S.live.t_gen = 0; S.live.prompt_done = 0; S.live.prompt_base = 0; S.live.prefilling = R.n_prompt > 0; S.live.t_prompt0 = clk::now(); S.live.n_requests++; }
        // Whatever way this returns (an eval error, a client that went away), the
        // counters must not say "busy" forever.
        struct busy_guard { live_stats & L; ~busy_guard() { std::lock_guard<std::mutex> lk(L.mu); L.busy = false; L.prefilling = false; } } guard{S.live};
        g_gen_thread = pthread_self(); g_gen_thread_set = true;
        smp.gen.clear();

        const auto tp = clk::now();
        const float * lg = nullptr;
        while (fed < (int32_t) hist.size()) {
            const int32_t take = std::min<int32_t>(S.n_batch, (int32_t) hist.size() - fed);
            { std::lock_guard<std::mutex> lk(S.live.mu); S.live.prompt_base = fed - fed0; S.live.prompt_done = fed - fed0; }
            lg = S.eng.eval(hist.data(), fed + take, take, e);
            if (!lg) return false;
            fed += take;
            { std::lock_guard<std::mutex> lk(S.live.mu); S.live.prompt_base = fed - fed0; S.live.prompt_done = fed - fed0; S.live.n_past = S.eng.n_past(); }
            if (on_tick) on_tick();
        }
        R.t_prompt = since(tp);
        { std::lock_guard<std::mutex> lk(S.live.mu); S.live.t_prompt = R.t_prompt; S.live.prefilling = false; S.live.prompt_done = R.n_prompt; S.live.n_past = S.eng.n_past(); }

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
        std::vector<int32_t> evald;   // accepted drafts still to emit (already evaluated), after `tok`
        // Draft length: the head's acceptance per draft position, tracked as running
        // means; a position is drafted while the chance that everything before it
        // and it are accepted beats what the extra position costs the step (~0.4 of
        // a single-token step, measured). Capped by --mtp-drafts.
        static float acc_at[engine::MTP_MAX_DRAFTS] = { 0.85f, 0.80f, 0.75f };
        static const float draft_cost = getenv("QWFN_DRAFT_COST") ? (float) atof(getenv("QWFN_DRAFT_COST")) : 0.7f;   // measured: a position costs ~0.6 of a single-token step, more as the queue deepens
        auto drafts_wanted = [&]() {
            // The k that maximises expected tokens per unit of step cost: tokens(k) =
            // 1 + a1 + a1 a2 + ... , cost(k) = 1 + k * draft_cost.
            const int cap = (int) std::min<uint32_t>(S.mtp_drafts, (uint32_t) engine::MTP_MAX_DRAFTS);
            int best = 1; float best_rate = 0.0f, tokens = 1.0f, p = 1.0f;
            for (int k = 1; k <= cap; k++) {
                p *= acc_at[k - 1]; tokens += p;
                const float rate = tokens / (1.0f + (float) k * draft_cost);
                if (rate > best_rate) { best_rate = rate; best = k; }
            }
            return best;
        };
        // Layer 0's reads for the tokens the next eval will take, issued as soon as they
        // are known (engine::spec_layer0): the bonus token before the head runs, the pair
        // once drafted, a sampled token before its eval.
        auto spec_l0 = [&](const int32_t * toks, int n, int T) {
            for (int k = 0; k < n; k++) hist.push_back(toks[k]);
            std::string se; S.eng.spec_layer0(hist.data(), (int32_t) hist.size(), T, se);
            for (int k = 0; k < n; k++) hist.pop_back();
        };
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
                        R.stop_seq = s; hit = true; break;
                    }
                if (hit) { R.finish = "stop"; n++; break; }
            }

            if ((int32_t) hist.size() + 1 > (int32_t) S.n_ctx) { R.finish = "length"; n++; break; }
            if (on_tick) on_tick();
            if (tok_in) {   // already evaluated with its predecessor: the next accepted draft, then the token after them
                if (!evald.empty()) { tok = evald.front(); evald.erase(evald.begin()); continue; }
                tok = tok_next; tok_in = false; continue;
            }
            // The draft: the head's argmax when sampling is greedy; at temperature,
            // a sample from the head's own distribution under the request's
            // sampler (speculative sampling: accept with probability
            // min(1, p(d)/q(d)), on rejection draw from the residual p - q, so
            // every emitted token is distributed exactly as the trunk's p).
            // Sampling the draft rather than taking its argmax is what keeps the
            // acceptance near the greedy rate when the trunk itself is sampled.
            static const bool argmax_draft = getenv("QWFN_MTP_ARGMAX_DRAFT") != nullptr;   // the old rule, for A/B
            // The drafts of this step. Greedy: the head's argmax chain. At temperature:
            // each draft sampled from the head's distribution under the request's
            // sampler, the next one chained from that sample, so the acceptance test
            // below sees the distribution the draft was drawn from.
            const bool sampled = smp.cfg.temp > 0.0f && !argmax_draft && S.eng.mtp_logits() != nullptr;
            std::vector<int32_t> drafts; std::vector<std::vector<std::pair<int32_t, float>>> qs;
            if (S.eng.mtp_draft_id() >= 0) {
                const int want = drafts_wanted();
                if (!sampled) {
                    if (!S.eng.mtp_draft_more(want, e)) return false;
                    for (int k = 0; k < S.eng.mtp_draft_count(); k++) drafts.push_back(S.eng.mtp_draft_k(k));
                } else {
                    for (int k = 0; k < want; k++) {
                        const float * hl = S.eng.mtp_logits_k(k);
                        if (!hl) break;
                        auto qd = smp.dist(hl, S.eng.n_vocab());
                        if (qd.empty()) break;
                        const int32_t d = smp.sample(qd);
                        drafts.push_back(d); qs.push_back(std::move(qd));
                        if (k + 1 < want && !S.eng.mtp_draft_next(e, d)) return false;
                    }
                }
                // No draft past an end-of-generation token, the budget or the context.
                for (size_t k = 0; k < drafts.size(); k++) if (S.vb.is_eog(drafts[k])) { drafts.resize(k); qs.resize(std::min(qs.size(), k)); break; }
                while (!drafts.empty() && (n + (int) drafts.size() >= budget || (int32_t) hist.size() + 1 + (int32_t) drafts.size() > (int32_t) S.n_ctx)) { drafts.pop_back(); if (qs.size() > drafts.size()) qs.pop_back(); }
            }
            const int K = (int) drafts.size();
            if (K > 0) {
                {
                    std::vector<int32_t> step(drafts);
                    spec_l0(step.data(), K, K + 1);
                }
                for (int32_t d : drafts) hist.push_back(d);
                if (!S.eng.eval_decode(hist.data(), (int32_t) hist.size(), K + 1, e)) return false;
                R.n_pairs++; R.n_drafted += K;
                // Verify position by position: accept draft j against the trunk's logits at
                // position j; the first rejection ends the step with a token drawn there.
                int j = 0; int32_t y = -1;
                std::uniform_real_distribution<double> U(0.0, 1.0);
                for (j = 0; j < K; j++) {
                    const float * lj = S.eng.logits_pos(j);
                    bool accept;
                    if (!sampled || j >= (int) qs.size()) {
                        y = smp.pick(lj, S.eng.n_vocab());
                        accept = y == drafts[j];
                    } else {
                        const auto pd = smp.dist(lj, S.eng.n_vocab());
                        const float p_d = sampler::prob_of(pd, drafts[j]), q_d = sampler::prob_of(qs[j], drafts[j]);
                        accept = q_d <= 0.0f || p_d >= q_d || U(smp.rng) < (double) p_d / (double) q_d;
                        if (!accept) {
                            // The residual max(0, p - q), normalised, over p's candidates.
                            std::vector<std::pair<int32_t, float>> res; double sum = 0;
                            for (const auto & [t, p] : pd) { const float r = p - sampler::prob_of(qs[j], t); if (r > 0) { res.emplace_back(t, r); sum += r; } }
                            if (res.empty() || sum <= 0) y = smp.sample(pd);
                            else { for (auto & [t, r] : res) r = (float) (r / sum); y = smp.sample(res); }
                        }
                    }
                    acc_at[j] += 0.05f * ((accept ? 1.0f : 0.0f) - acc_at[j]);
                    if (!accept) break;
                    R.n_accepted++;
                    smp.gen.push_back(drafts[j]);
                }
                std::vector<int32_t> fed(drafts.begin(), drafts.begin() + j);
                if (j == K) {
                    y = smp.pick(S.eng.logits_pos(K), S.eng.n_vocab());
                } else {
                    if (!S.eng.rollback_n(K - j, e)) return false;
                    hist.resize(hist.size() - (size_t) (K - j));
                }
                fed.push_back(y);
                spec_l0(&y, 1, 1);
                if (!S.eng.mtp_step(fed.data(), (int) fed.size(), e)) return false;
                if (j > 0) { tok = drafts[0]; evald.assign(drafts.begin() + 1, drafts.begin() + j); tok_next = y; tok_in = true; }
                else       { tok = y; }
            } else {
                lg = S.eng.eval(hist.data(), (int32_t) hist.size(), 1, e);
                if (!lg) return false;
                tok = smp.pick(lg, S.eng.n_vocab());
                if (S.eng.mtp_on()) spec_l0(&tok, 1, 1);   // a lead only when the head runs next
                if (!S.eng.mtp_step(&tok, 1, e)) return false;
            }
        }
        if (n >= budget && budget > 0) R.finish = "length";
        R.t_gen = since(td);
        R.n_gen = n;
        { std::lock_guard<std::mutex> lk(S.live.mu); S.live.busy = false; S.live.n_gen = n; S.live.t_gen = R.t_gen;
          S.live.n_prompt_total += R.n_prompt; S.live.t_prompt_total += R.t_prompt; S.live.n_gen_total += n; S.live.t_gen_total += R.t_gen; S.live.n_past = S.eng.n_past();
          S.live.n_input_total += R.n_input; S.live.n_cached_total += R.n_cached;
          S.live.n_pairs_total += R.n_pairs; S.live.n_accepted_total += R.n_accepted; S.live.n_drafted_total += R.n_drafted;
          S.live.last = live_stats::snapshot{R.n_input, R.n_cached, R.n_prompt, R.n_gen, R.n_pairs, R.n_accepted, R.n_drafted, R.t_prompt, R.t_gen, R.finish, now_unix(), true};
          S.live.have_last = true; }

        // Close the turn so the next request can continue from here. The sampled
        // end-of-turn token was appended but never evaluated, so the engine's
        // n_past -- not hist.size() -- is what it has actually consumed.
        dg.ok = true;   // finished cleanly: the state describes exactly S.consumed
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
    // SO_REUSEADDR, as llama-server sets, instead of httplib's default SO_REUSEPORT. A
    // supervisor may reuse one child port across servers: connections the previous
    // server closed stay in TIME_WAIT on it for up to 60 s, and Linux lets a new bind through
    // only when both sockets carry the same option. With REUSEPORT only, switching between
    // this server and llama-server or vLLM failed with "address in use" in both directions
    // until TIME_WAIT expired. REUSEPORT would also let a stale instance share the port.
    svr.set_socket_options([](socket_t sock) { httplib::set_socket_opt(sock, SOL_SOCKET, SO_REUSEADDR, 1); });
    // httplib's socket timeouts default to 5 s per write and per read. A client
    // UI that stops draining the stream for 5 s (rendering a long reasoning
    // trace) would get the connection cut without a trailer and without a log
    // line here -- "peer closed connection without sending complete message
    // body" on its side. A local server can afford to wait.
    // prime the unwinder: glibc's first backtrace() dlopens libgcc_s, taking the loader and malloc locks.
    // Done first inside the stall probe's handler, it deadlocked a generation thread interrupted inside
    // malloc (a long first-request JIT compile) for good. Loaded here, while nothing is interrupted.
    { void * f[2]; (void) backtrace(f, 2); }
    signal(SIGSEGV, on_fatal); signal(SIGABRT, on_fatal); signal(SIGBUS, on_fatal); signal(SIGFPE, on_fatal);
    signal(SIGUSR2, on_stall_probe);
    // A local web page (the console, a harness) may read /stats and /props
    // from another origin: allow it.
    svr.set_default_headers({{"Access-Control-Allow-Origin", "*"}, {"Access-Control-Allow-Headers", "Content-Type, Authorization"},
                             {"Access-Control-Allow-Methods", "GET, POST, OPTIONS"}});
    svr.Options(R"(.*)", [](const httplib::Request &, httplib::Response & res) { res.status = 204; });
    svr.set_write_timeout(3600, 0);
    // Stall watchdog: a request that makes no progress is logged with what the expert cache is
    // waiting on, and the stuck thread prints its own stack. Progress is prefill layers AND generated
    // tokens: the engine advances prompt_done after every layer of every chunk, so a long prefill that
    // is moving is not a stall (it used to be reported every 30 s as "no new token ... at generated
    // token 0"). Thresholds: 30 s while generating, 120 s while prefilling, because one layer of the
    // first prefill after a load can spend ~60 s in kernel JIT.
    std::thread([&]() {
        double last_p = -1, stalled = 0;
        for (;;) {
            std::this_thread::sleep_for(std::chrono::seconds(5));
            bool busy, prefilling; int n; double done, input;
            { std::lock_guard<std::mutex> lk(S.live.mu); busy = S.live.busy; prefilling = S.live.prefilling;
              n = S.live.n_gen; done = S.live.prompt_done; input = (double) S.live.n_prompt; }
            if (!busy) { last_p = -1; stalled = 0; continue; }
            const double p = done + (double) n;
            if (p != last_p) { last_p = p; stalled = 0; continue; }
            stalled += 5;
            const double limit = prefilling ? 120 : 30;
            if (stalled >= limit && ((int) stalled % 30) == 0) {
                const int ws = S.eng.cache_wait_state();
                const char * what = ws == 1 ? "demand reads" : ws == 2 ? "speculative reads" : "nothing (compute or lock)";
                if (prefilling)
                    fprintf(stderr, "[qwfn-server] STALL: prefill has not advanced for %.0f s (at %.0f of %.0f prompt tokens); expert cache waiting on %s (%zu reads)\n",
                            stalled, done, input, what, S.eng.cache_wait_count());
                else
                    fprintf(stderr, "[qwfn-server] STALL: no new token for %.0f s at generated token %d; expert cache waiting on %s (%zu reads)\n",
                            stalled, n, what, S.eng.cache_wait_count());
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
                {"created", now_unix()}, {"owned_by", "qwfnfer"},
                // The Claude tier this model stands in for, for clients that
                // discover models by it (Claude Desktop); everyone else ignores it.
                {"display_name", S.model_id}, {"anthropic_family_tier", "sonnet"}} })}
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
            {"dump_requests", S.dump_dir},
            {"skip_miss", cfg.skip_miss}, {"spec_block", cfg.spec_block}, {"mtp", S.eng.mtp_loaded()}, {"model_file", S.model_file},
            {"vision", S.vis.loaded()}, {"vision_weights", S.vis.loaded() ? "cpu" : "off"},
            {"state_host", cfg.kv_host && cfg.idx_host ? "kv,idx" : cfg.kv_host ? "kv" : cfg.idx_host ? "idx" : "none"},
            {"n_threads", S.eng.n_threads()}, {"kv_type", cfg.type_k == GGML_TYPE_Q4_0 ? "q4_0" : cfg.type_k == GGML_TYPE_Q8_0 ? "q8_0" : "f16"},
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
            if (body.contains("dump_requests")) S.dump_dir = body["dump_requests"].is_string() ? body["dump_requests"].get<std::string>() : "";
            if (body.contains("thinking"))     S.preset_think.from_json(body["thinking"]);
            if (body.contains("non_thinking")) S.preset_nothink.from_json(body["non_thinking"]);
            // Flat sampling fields apply to both presets.
            S.preset_think.from_json(body); S.preset_nothink.from_json(body);
        }
        // CPU threads for the RAM-served experts, applied between requests (the console's
        // auto-tune sweeps it on the running server). Refused while a generation holds the engine.
        if (body.contains("threads")) {
            const int n = body["threads"].get<int>();
            if (n < 1 || n > 512) { fail(res, 400, "threads must be 1..512"); return; }
            std::unique_lock<std::mutex> lk(S.mu, std::try_to_lock);
            if (!lk.owns_lock()) { fail(res, 409, "a request is running; set threads between requests"); return; }
            S.eng.set_n_threads(n);
            fprintf(stderr, "[qwfn-server] threads set to %d\n", n);
        }
        res.set_content(props_json().dump(2, ' ', false, json::error_handler_t::replace), "application/json");
    });
    // ---- the live counter ------------------------------------------------------
    auto stats_json = [&]() {
        json t = S.live.timings();
        const auto & c = S.eng.cache_stats();   // racy reads of plain counters: a monitor, not a ledger
        long long np, ng, nr, npair, nacc, ndraft; double tp, tg; bool busy; int n_past;
        { std::lock_guard<std::mutex> lk(S.live.mu); np = S.live.n_prompt_total; ng = S.live.n_gen_total; nr = S.live.n_requests;
          tp = S.live.t_prompt_total; tg = S.live.t_gen_total; busy = S.live.busy; n_past = S.live.n_past;
          npair = S.live.n_pairs_total; nacc = S.live.n_accepted_total; ndraft = S.live.n_drafted_total; }
        long long ninp, ncach; { std::lock_guard<std::mutex> lk(S.live.mu); ninp = S.live.n_input_total; ncach = S.live.n_cached_total; }
        return json{
            {"busy", busy},
            // prompt: the current request's prompt while busy, the last one's when idle.
            // input = cached (reused from the engine's prefix) + n (prefilled); done counts
            // the prefilled tokens so far, fractional inside a batch.
            {"prompt", {{"n", t["prompt_n"]}, {"ms", t["prompt_ms"]}, {"tokens_per_second", t["prompt_per_second"]},
                        {"input", t["prompt_input_n"]}, {"cached", t["prompt_cached_n"]}, {"done", t["prompt_done_n"]}, {"prefilling", t["prefilling"]}}},
            {"generation", {{"n", t["predicted_n"]}, {"ms", t["predicted_ms"]}, {"tokens_per_second", t["predicted_per_second"]}}},
            {"last", S.live.last_json()},
            {"context", {{"n_past", n_past}, {"n_ctx", S.n_ctx}}},
            {"threads", S.eng.n_threads()},
            {"totals", {{"requests", nr}, {"prompt_tokens", np}, {"prompt_tokens_per_second", tp > 0 ? np / tp : 0.0},
                        {"input_tokens", ninp}, {"cached_tokens", ncach},
                        {"generated_tokens", ng}, {"generated_tokens_per_second", tg > 0 ? ng / tg : 0.0},
                        {"prompt_seconds", tp}, {"generation_seconds", tg}}},
            {"expert_cache", {{"hit_rate", c.hit_rate()}, {"vram_served", c.gpu_rate()},
                              {"bytes_from_disk", c.bytes_from_disk},
                              // the raw counters, so a harness can difference two samples
                              {"lookups", c.lookups}, {"hits", c.hits}, {"gpu_hits", c.gpu_hits},
                              {"promotions", c.promotions}, {"pf_issued", c.pf_issued}, {"pf_used", c.pf_used}}},
            {"speculative", {{"pairs", npair}, {"accepted", nacc}, {"drafted", ndraft}, {"acceptance", ndraft ? (double) nacc / ndraft : 0.0},
                             {"tokens_per_step", npair ? (double) (npair + nacc) / npair : 1.0}}},
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

    // ---- one chat request, whatever format it came in ---------------------------
    // The OpenAI shape is the internal one: /v1/chat/completions takes it as it
    // is, /v1/messages (the Anthropic Messages API) is translated into it first.
    // From there everything is shared -- the request's fields, the log line, the
    // prompt, the generation, the tool-call parsing and the bookkeeping that
    // lets the next turn continue the engine's prefix. Only the wire format
    // differs, and each route supplies its own.
    struct chat_request {
        json msgs, tools;
        std::string effort, tools_block, forced, id, stamp;   // stamp: unique per request; the tool-call ids derive from it
        bool thinking = true, stream = false, timings_per_token = false;
        sampler smp;
        int max_tok = 0, reasoning_budget = 0;
        std::vector<std::string> stops;
        size_t n_images = 0;
    };
    // Unique per request: the tool-call ids a harness keys its cards and
    // replays on are derived from it, and two rounds of one conversation used
    // to hand out the same "call_0_<seconds>" id.
    std::atomic<long long> req_seq{0};
    auto parse_chat_request = [&](const json & body, const char * id_prefix, chat_request & Q, std::string & err) -> bool {
        if (!body.contains("messages") || !body["messages"].is_array()) { err = "messages is required"; return false; }

        // Thinking: either OpenAI-ish reasoning_effort, or the Qwen template's
        // own chat_template_kwargs.enable_thinking.
        { std::lock_guard<std::mutex> lk(S.props_mu); Q.effort = body.value("reasoning_effort", S.def_effort); }
        if (body.contains("chat_template_kwargs")) {
            const auto & k = body["chat_template_kwargs"];
            if (k.contains("enable_thinking") && !k["enable_thinking"].get<bool>()) Q.effort = "off";
        }
        if (!effort_valid(Q.effort)) { err = "reasoning_effort must be xhigh|medium|low|off"; return false; }

        // Qwen's soft switches, for harnesses that show no thinking toggle: a
        // trailing "/think" or "/no_think" in the last user message sets the
        // mode for this request and is stripped before the model sees it.
        Q.msgs = body["messages"];
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
            for (int m = (int) Q.msgs.size() - 1; m >= 0; m--) {
                if (Q.msgs[m].value("role", "") != "user") continue;
                int r = -1;
                if (Q.msgs[m]["content"].is_string()) {
                    std::string c = Q.msgs[m]["content"].get<std::string>();
                    if ((r = strip_tag(c)) >= 0) Q.msgs[m]["content"] = c;
                } else if (Q.msgs[m]["content"].is_array()) {
                    for (auto & part : Q.msgs[m]["content"])
                        if (part.value("type", "") == "text") {
                            std::string c = part.value("text", "");
                            if ((r = strip_tag(c)) >= 0) { part["text"] = c; break; }
                        }
                }
                if (r >= 0) Q.effort = r == 0 ? "off" : (def_now != "off" ? def_now : "xhigh");
                break;
            }
        }
        Q.thinking = Q.effort != "off";

        { std::lock_guard<std::mutex> lk(S.props_mu); Q.smp.cfg = Q.thinking ? S.preset_think : S.preset_nothink; Q.max_tok = S.def_max_tokens; Q.reasoning_budget = S.def_reasoning_budget; }
        Q.smp.cfg.from_json(body);               // any sampling field in the request wins
        if (body.contains("seed") && body["seed"].is_number_integer())
            Q.smp.rng.seed((unsigned) body["seed"].get<long long>());
        else Q.smp.rng.seed((unsigned) std::chrono::steady_clock::now().time_since_epoch().count());
        if (body.contains("max_completion_tokens")) Q.max_tok = body["max_completion_tokens"].get<int>();
        else if (body.contains("max_tokens"))       Q.max_tok = body["max_tokens"].get<int>();
        Q.timings_per_token = body.value("timings_per_token", false);
        // Tools: rendered into the system turn unless tool_choice is "none";
        // "required" or a named function forces the call's opening.
        Q.tools = body.contains("tools") && body["tools"].is_array() ? body["tools"] : json::array();
        if (body.contains("tool_choice")) {
            const json & tc = body["tool_choice"];
            if (tc.is_string()) {
                if (tc == "none") Q.tools = json::array();
                // "required" opens the block and stops there: a prefix ending in
                // "<function=" ends on a token boundary the model never produces
                // and it continued with a call id ("call_5067") as the name.
                else if (tc == "required" && !Q.tools.empty()) Q.forced = "<tool_call>\n";
            } else if (tc.is_object() && tc.contains("function") && !Q.tools.empty()) {
                Q.forced = "<tool_call>\n<function=" + tc["function"].value("name", "") + ">\n";
            }
        }
        Q.tools_block = render_tools_block(Q.tools);
        if (body.contains("reasoning_budget")) Q.reasoning_budget = body["reasoning_budget"].get<int>();
        if (body.contains("stop")) {
            if (body["stop"].is_string()) Q.stops.push_back(body["stop"].get<std::string>());
            else for (const auto & s : body["stop"]) Q.stops.push_back(s.get<std::string>());
        }
        Q.stream = body.value("stream", false);
        Q.stamp = std::to_string(now_unix()) + "-" + std::to_string(++req_seq);
        Q.id = std::string(id_prefix) + Q.stamp;
        for (const auto & m : Q.msgs)
            if (m.contains("content") && m["content"].is_array())
                for (const auto & part : m["content"]) { const std::string ty = part.value("type", ""); if (ty == "image_url" || ty == "input_image") Q.n_images++; }
        return true;
    };
    // One line per request, so a harness's exact ask is visible in the log.
    auto log_request = [&](const httplib::Request & req, const chat_request & Q, const json & body, const char * kind) {
        fprintf(stderr, "[qwfn-server] request from %s%s: %zu messages, %zu images, stream=%d, max_tokens=%d, effort=%s, budget=%d, temp=%.2f, timings_per_token=%d, tools=%zu%s, keys:",
                req.remote_addr.c_str(), kind, Q.msgs.size(), Q.n_images, (int) Q.stream, Q.max_tok, Q.effort.c_str(), Q.reasoning_budget, Q.smp.cfg.temp, (int) Q.timings_per_token,
                Q.tools.size(), Q.forced.empty() ? "" : " (forced)");
        for (auto it = body.begin(); it != body.end(); ++it) fprintf(stderr, " %s", it.key().c_str());
        fprintf(stderr, "\n");
        std::string dump; { std::lock_guard<std::mutex> lk(S.props_mu); dump = S.dump_dir; }
        if (!dump.empty()) {
            const std::string path = dump + "/" + Q.id + ".json";
            if (FILE * f = fopen(path.c_str(), "w")) { fwrite(req.body.data(), 1, req.body.size(), f); fclose(f); }
            else fprintf(stderr, "[qwfn-server] cannot write %s\n", path.c_str());
        }
    };
    auto too_long = [](const std::string & e) { return e.rfind("context_length_exceeded", 0) == 0; };
    auto openai_usage = [](const gen_result & R) {
        return json{{"prompt_tokens", R.n_input}, {"prompt_tokens_details", {{"cached_tokens", R.n_cached}}},
                    {"completion_tokens", R.n_gen}, {"total_tokens", R.n_input + R.n_gen}};
    };

    // What a generation produced: the reply, its tool calls parsed out, and the
    // prose before the first call (the whole reply when there is none).
    struct chat_result { gen_result R; json calls; std::string text; };
    auto finish_request = [&](const chat_request & Q, chat_result & C) {
        C.text = parse_tool_calls(Q.forced + C.R.content, Q.tools, C.calls);
        S.last_tool_key = tool_calls_key(C.calls); S.last_content = C.calls.empty() ? C.R.content : C.text;
        if (!C.calls.empty()) C.R.finish = "tool_calls";
        S.last_msgs = Q.msgs;
    };
    auto log_done = [&](const chat_request & Q, const gen_result & R) {
        fprintf(stderr, "[qwfn-server] %s: prompt %d tok (%d cached) %.1f tok/s | generated %d tok (%zu reasoning chars%s) in %.1f s, %.1f tok/s, finish %s%s\n",
                Q.id.c_str(), R.n_prompt, R.n_cached, R.t_prompt > 0 ? R.n_prompt / R.t_prompt : 0.0, R.n_gen, R.reasoning.size(),
                R.reasoning_budget_hit ? ", budget hit" : "", R.t_gen, R.t_gen > 0 ? R.n_gen / R.t_gen : 0.0, R.finish.c_str(),
                R.n_pairs ? (" | drafts: " + std::to_string(R.n_accepted) + " of " + std::to_string(R.n_drafted) + " accepted over " + std::to_string(R.n_pairs) + " steps").c_str() : "");
    };
    auto run_batch = [&](chat_request & Q, const server::prompt & P, chat_result & C, std::string & err) -> bool {
        if (!generate(P, Q.smp, Q.max_tok, Q.thinking, Q.stops,
                      [](const std::string &, bool) { return true; }, C.R, err, Q.reasoning_budget)) {
            S.last_msgs = json();   // cache is no longer trustworthy
            fprintf(stderr, "[qwfn-server] %s: generation failed after %d tokens: %s\n", Q.id.c_str(), C.R.n_gen, err.c_str());
            return false;
        }
        finish_request(Q, C);
        log_done(Q, C.R);
        return true;
    };

    // A streamed generation, format-agnostic. The route supplies the writers;
    // the driver coalesces reasoning, splits tool blocks out of the content
    // while they are written (see tool_streamer), keeps bytes moving through a
    // silent stretch, and does the bookkeeping when the reply is done.
    struct stream_events {
        std::function<bool(const std::string &)> reasoning, content;
        std::function<bool(const json &)> tool_delta;   // an OpenAI tool_calls delta: with an id it opens a call, else it extends the arguments
        std::function<bool()> tool_end;                 // the open call's block is complete
        std::function<bool()> keepalive;                // nothing has gone out for 15 s
        bool write_failed = false;                      // the route sets it when a write fails...
        clk::time_point last_write = clk::now();        // ...and stamps this on every write
    };
    struct stream_outcome { chat_result C; bool ok = false; std::string aborted, err; };
    auto run_stream = [&](chat_request & Q, const server::prompt & P, stream_events & ev) -> stream_outcome {
        stream_outcome out;
        gen_result & R = out.C.R;
        // A silent stretch -- a long prefill, a tool call being written, a
        // held-back parameter -- still has to put bytes on the wire: Unsloth
        // Studio's proxy reads the stream with a 300 s per-read timeout and
        // reports "Timeout waiting for custom response" when nothing arrives,
        // with the generation still running here; Claude Code drops a stream
        // silent for 300 s the same way.
        auto tick = [&]() {
            if (ev.write_failed || since(ev.last_write) < 15.0) return;
            if (!ev.keepalive()) ev.write_failed = true;
            ev.last_write = clk::now();
        };
        std::string held;   // incomplete UTF-8 tail of the previous delta
        auto complete = [&](std::string piece) {   // returns what may be emitted now
            piece = held + piece; held.clear();
            const size_t t = utf8_incomplete_tail(piece);
            if (t) { held = piece.substr(piece.size() - t); piece.erase(piece.size() - t); }
            return piece;
        };
        // Reasoning deltas are coalesced (every 100 ms or 16 tokens): a
        // 15-minute think is ~13,000 tokens, and a UI re-rendering its
        // reasoning block per chunk is what stalls the socket.
        std::string rbuf; int rcount = 0; auto rlast = clk::now();
        auto flush_reasoning = [&]() {
            if (rbuf.empty()) return true;
            const bool ok = ev.reasoning(rbuf);
            rbuf.clear(); rcount = 0; rlast = clk::now();
            return ok;
        };
        // Tool calls are never streamed as text: content before the first
        // <tool_call> streams normally (holding back a possible partial tag);
        // a block streams as tool-call deltas while it is written and the text
        // after it streams as content.
        std::string tacc = Q.forced; size_t temitted = 0; bool tool_mode = !Q.forced.empty();
        int n_calls = 0;
        std::unique_ptr<tool_streamer> ts;
        const bool with_tools = Q.tools.is_array() && !Q.tools.empty();
        const std::string mark = TC_MARK;
        auto open_streamer = [&](size_t block_start) {
            ts.reset(new tool_streamer(Q.tools, block_start, n_calls, "call_" + std::to_string(n_calls) + "_" + Q.stamp, ev.tool_delta));
            n_calls++;
        };
        if (tool_mode) open_streamer(0);
        auto drain_content = [&]() {
            if (!with_tools) return ev.content(tacc.substr(temitted)) && (temitted = tacc.size(), true);
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
                        if (!t.empty() && !ev.content(t)) return false;
                        temitted = p; tool_mode = true; open_streamer(p);
                    } else {
                        // Emit everything except a tail that could still become the marker.
                        size_t hold = undecided ? tacc.size() - p : 0;
                        if (!undecided)
                            for (size_t k = std::min(mark.size() - 1, tacc.size() - temitted); k > 0; k--)
                                if (mark.compare(0, k, tacc, tacc.size() - k, k) == 0) { hold = k; break; }
                        const std::string t = tacc.substr(temitted, tacc.size() - temitted - hold);
                        if (!t.empty() && !ev.content(t)) return false;
                        temitted = tacc.size() - hold;
                        return true;
                    }
                }
                if (!ts->feed(tacc)) return false;
                const size_t close = tacc.find(TC_CLOSE, temitted);
                if (close == std::string::npos) return true;
                if (ts->opened && ts->st != tool_streamer::CLOSED && !ts->close_object()) return false;
                if (ts->opened && !ev.tool_end()) return false;
                ts.reset();
                temitted = close + 12;
                tool_mode = false;   // text after a block (the template forbids it, models do it) streams as content
            }
        };
        try {
            out.ok = generate(P, Q.smp, Q.max_tok, Q.thinking, Q.stops,
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
                }, R, out.err, Q.reasoning_budget, tick);
            if (out.ok) {
                flush_reasoning();
                if (tool_mode && ts) {
                    // The reply ended inside a block. Closed with </function>:
                    // a complete call, the batch parser agrees. Cut off mid-way:
                    // leave the fragment as it is -- a client that cannot parse
                    // the arguments describes the call instead of running it,
                    // which is the right outcome for a call the model never
                    // finished. Never a name: plain text.
                    ts->feed(tacc);
                    if (!ts->opened && temitted < tacc.size()) ev.content(tacc.substr(temitted));
                    temitted = tacc.size();
                }
                if (temitted < tacc.size()) { ev.content(tacc.substr(temitted)); temitted = tacc.size(); }
            }
        } catch (const std::exception & ex) {
            out.aborted = ex.what();
        } catch (...) {
            out.aborted = "unknown exception";
        }
        if (!out.aborted.empty()) {
            // The engine's history is unknown from here: drop the prefix cache
            // so the next request starts clean instead of running on a
            // desynchronised state.
            fprintf(stderr, "[qwfn-server] %s: stream aborted by an exception after %d generated tokens: %s\n", Q.id.c_str(), R.n_gen, out.aborted.c_str());
            S.last_msgs = json(); S.consumed.clear(); S.eng.reset(); S.eng.clear_embeddings();
            { std::lock_guard<std::mutex> lg(S.live.mu); S.live.busy = false; }
            return out;
        }
        if (out.ok) finish_request(Q, out.C); else S.last_msgs = json();
        if (ev.write_failed)
            fprintf(stderr, "[qwfn-server] %s: client stopped reading after %d prompt + %d generated tokens (%.0f s); stream dropped\n",
                    Q.id.c_str(), R.n_prompt, R.n_gen, R.t_prompt + R.t_gen);
        else if (!out.ok)
            fprintf(stderr, "[qwfn-server] %s: generation failed after %d tokens: %s\n", Q.id.c_str(), R.n_gen, out.err.c_str());
        else
            log_done(Q, R);
        return out;
    };

    svr.Post("/v1/chat/completions", [&](const httplib::Request & req, httplib::Response & res) {
        json body;
        try { body = json::parse(req.body); }
        catch (const std::exception & ex) { fail(res, 400, std::string("bad JSON: ") + ex.what()); return; }
        auto Q = std::make_shared<chat_request>();
        std::string e;
        if (!parse_chat_request(body, "chatcmpl-", *Q, e)) { fail(res, 400, e); return; }
        log_request(req, *Q, body, "");

        // One generation at a time. For a streamed response the lock must
        // outlive this handler: the chunked provider runs after it returns, and
        // a second request arriving mid-stream (a harness generating a title,
        // the next turn) must wait, not enter the engine. So the lock is shared
        // with the provider and released when the stream is done.
        auto lk = std::make_shared<std::unique_lock<std::mutex>>(S.mu);
        auto P = std::make_shared<server::prompt>();
        if (!build_prompt(Q->msgs, Q->effort, Q->thinking, Q->tools_block, Q->forced, *P, e)) { fail(res, 400, e); return; }

        if (!Q->stream) {
            chat_result C;
            if (!run_batch(*Q, *P, C, e)) {
                fail(res, too_long(e) ? 400 : 500, e, too_long(e) ? "invalid_request_error" : "server_error");
                return;
            }
            json msg{{"role", "assistant"}, {"content", C.calls.empty() ? json(C.R.content) : (C.text.empty() ? json(nullptr) : json(C.text))}};
            if (!C.calls.empty()) msg["tool_calls"] = C.calls;
            if (!C.R.reasoning.empty()) msg["reasoning_content"] = C.R.reasoning;
            res.set_content(json{
                {"id", Q->id}, {"object", "chat.completion"}, {"created", now_unix()},
                {"model", S.model_id},
                {"choices", json::array({ json{
                    {"index", 0}, {"message", msg}, {"finish_reason", C.R.finish}} })},
                {"usage", openai_usage(C.R)},
                {"timings", S.live.timings()}
            }.dump(2, ' ', false, json::error_handler_t::replace), "application/json");
            return;
        }

        // Streaming: the provider owns the engine lock until it is done.
        res.set_header("Cache-Control", "no-cache");
        res.set_chunked_content_provider("text/event-stream",
            [&, Q, P, lk](size_t, httplib::DataSink & sink) mutable {
                stream_events ev;
                auto send = [&](const json & j) {
                    const std::string s = "data: " + json_dump(j) + "\n\n";
                    const bool ok = sink.write(s.data(), s.size());
                    ev.last_write = clk::now();
                    if (!ok) ev.write_failed = true;
                    return ok;
                };
                auto chunk = [&](const json & d, const json & finish) {
                    return json{{"id", Q->id}, {"object", "chat.completion.chunk"}, {"created", now_unix()}, {"model", S.model_id},
                                {"choices", json::array({ json{{"index", 0}, {"delta", d}, {"finish_reason", finish}} })}};
                };
                auto emit = [&](const json & d) {
                    json c = chunk(d, nullptr);
                    if (Q->timings_per_token) c["timings"] = S.live.timings();   // live tok/s per chunk
                    return send(c);
                };
                send(chunk(json{{"role", "assistant"}}, nullptr));
                ev.reasoning  = [&](const std::string & t) { return emit(json{{"reasoning_content", t}}); };
                ev.content    = [&](const std::string & t) { return emit(json{{"content", t}}); };
                ev.tool_delta = [&](const json & d) { return emit(json{{"tool_calls", json::array({d})}}); };
                ev.tool_end   = [&]() { return true; };
                // An SSE comment line is ignored by every client (Studio relays
                // non-data lines untouched).
                ev.keepalive  = [&]() { static const std::string ka = ": keepalive\n\n"; return sink.write(ka.data(), ka.size()); };

                stream_outcome out = run_stream(*Q, *P, ev);
                if (!out.aborted.empty()) {
                    send(json{{"error", {{"message", "stream aborted: " + out.aborted}, {"type", "server_error"}}}});
                } else if (!out.ok) {
                    send(json{{"error", {{"message", out.err}, {"type", "server_error"}}}});
                } else {
                    json last = chunk(json::object(), out.C.R.finish);
                    last["usage"] = openai_usage(out.C.R);
                    last["timings"] = S.live.timings();
                    send(last);
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

    // ---- the Anthropic Messages API: Claude Code --------------------------------
    // ANTHROPIC_BASE_URL=http://127.0.0.1:8080 (with any ANTHROPIC_AUTH_TOKEN and
    // an ANTHROPIC_MODEL; the model name is echoed, not checked). Thinking comes
    // back as thinking blocks, tool calls as tool_use blocks streamed as
    // input_json_delta while the model writes them, usage as input / cached /
    // output tokens. See anthropic_to_openai for what is mapped and dropped.
    auto fail_anthropic = [](httplib::Response & res, int code, const std::string & msg,
                             const std::string & type = "invalid_request_error") {
        res.status = code;
        res.set_content(json{{"type", "error"}, {"error", {{"type", type}, {"message", msg}}}}.dump(2, ' ', false, json::error_handler_t::replace),
                        "application/json");
    };
    // Claude Code compacts its conversation when a rejection carries the API's
    // own wording for a prompt over the context.
    auto too_long_msg = [&](size_t n_tok) {
        return "prompt is too long: " + std::to_string(n_tok) + " tokens > " + std::to_string(S.n_ctx) + " maximum";
    };
    auto anthropic_usage = [](const gen_result & R) {
        return json{{"input_tokens", R.n_prompt}, {"cache_creation_input_tokens", 0}, {"cache_read_input_tokens", R.n_cached}, {"output_tokens", R.n_gen}};
    };
    auto anthropic_message = [&](const chat_request & Q, const std::string & model, const chat_result & C) {
        json content = json::array();
        if (!C.R.reasoning.empty())
            content.push_back(json{{"type", "thinking"}, {"thinking", C.R.reasoning}, {"signature", thinking_signature(C.R.reasoning)}});
        const std::string & text = C.calls.empty() ? C.R.content : C.text;
        if (!text.empty()) content.push_back(json{{"type", "text"}, {"text", text}});
        for (const auto & tc : C.calls)
            content.push_back(json{{"type", "tool_use"}, {"id", tool_use_id(tc["id"].get<std::string>())},
                                   {"name", tc["function"]["name"]}, {"input", tool_args_object(tc["function"])}});
        return json{{"id", Q.id}, {"type", "message"}, {"role", "assistant"}, {"model", model}, {"content", content},
                    {"stop_reason", anthropic_stop_reason(C.R.finish, C.R.stop_seq)},
                    {"stop_sequence", C.R.stop_seq.empty() ? json(nullptr) : json(C.R.stop_seq)},
                    {"usage", anthropic_usage(C.R)}};
    };
    // The translated request, parsed as the chat path parses its own.
    auto parse_anthropic = [&](const httplib::Request & req, httplib::Response & res, const char * id_prefix, json & body, chat_request & Q) -> bool {
        try { body = json::parse(req.body); }
        catch (const std::exception & ex) { fail_anthropic(res, 400, std::string("bad JSON: ") + ex.what()); return false; }
        std::string def_effort; int def_budget;
        { std::lock_guard<std::mutex> lk(S.props_mu); def_effort = S.def_effort; def_budget = S.def_reasoning_budget; }
        json oai; std::string e;
        if (!anthropic_to_openai(body, def_effort, def_budget, oai, e)) { fail_anthropic(res, 400, e); return false; }
        if (!parse_chat_request(oai, id_prefix, Q, e)) { fail_anthropic(res, 400, e); return false; }
        return true;
    };

    svr.Post("/v1/messages", [&](const httplib::Request & req, httplib::Response & res) {
        json body;
        auto Q = std::make_shared<chat_request>();
        if (!parse_anthropic(req, res, "msg_", body, *Q)) return;
        const std::string model = body.value("model", S.model_id);
        log_request(req, *Q, body, " (anthropic)");

        auto lk = std::make_shared<std::unique_lock<std::mutex>>(S.mu);
        auto P = std::make_shared<server::prompt>();
        std::string e;
        if (!build_prompt(Q->msgs, Q->effort, Q->thinking, Q->tools_block, Q->forced, *P, e)) { fail_anthropic(res, 400, e); return; }

        if (!Q->stream) {
            chat_result C;
            if (!run_batch(*Q, *P, C, e)) {
                if (too_long(e)) fail_anthropic(res, 400, too_long_msg(P->tok.size()));
                else             fail_anthropic(res, 500, e, "api_error");
                return;
            }
            res.set_content(anthropic_message(*Q, model, C).dump(2, ' ', false, json::error_handler_t::replace), "application/json");
            return;
        }

        res.set_header("Cache-Control", "no-cache");
        res.set_chunked_content_provider("text/event-stream",
            [&, Q, P, lk, model](size_t, httplib::DataSink & sink) mutable {
                stream_events ev;
                auto send = [&](const char * event, const json & j) {
                    const std::string s = std::string("event: ") + event + "\ndata: " + json_dump(j) + "\n\n";
                    const bool ok = sink.write(s.data(), s.size());
                    ev.last_write = clk::now();
                    if (!ok) ev.write_failed = true;
                    return ok;
                };
                // message_start goes out before the prefill, so its usage is the
                // prompt as it will be counted: what the engine already holds and
                // the rest; message_delta repeats it measured.
                const int32_t reuse = prefix_reuse(*P);
                send("message_start", json{{"type", "message_start"}, {"message", {
                    {"id", Q->id}, {"type", "message"}, {"role", "assistant"}, {"model", model}, {"content", json::array()},
                    {"stop_reason", nullptr}, {"stop_sequence", nullptr},
                    {"usage", {{"input_tokens", (int) P->tok.size() - reuse}, {"cache_creation_input_tokens", 0},
                               {"cache_read_input_tokens", reuse}, {"output_tokens", 0}}}}}});
                // Content blocks: one open at a time, indexed in the order opened.
                int next_index = 0, open = -1;
                std::string open_type, thinking_acc;
                auto delta = [&](const json & d) {
                    return send("content_block_delta", json{{"type", "content_block_delta"}, {"index", open}, {"delta", d}});
                };
                auto close_block = [&]() {
                    if (open < 0) return true;
                    if (open_type == "thinking" && !delta(json{{"type", "signature_delta"}, {"signature", thinking_signature(thinking_acc)}})) return false;
                    const bool ok = send("content_block_stop", json{{"type", "content_block_stop"}, {"index", open}});
                    open = -1; open_type.clear();
                    return ok;
                };
                auto open_block = [&](const std::string & ty, const json & block) {
                    if (!close_block()) return false;
                    open = next_index++; open_type = ty;
                    return send("content_block_start", json{{"type", "content_block_start"}, {"index", open}, {"content_block", block}});
                };
                ev.reasoning = [&](const std::string & t) {
                    if (open_type != "thinking" && !open_block("thinking", json{{"type", "thinking"}, {"thinking", ""}})) return false;
                    thinking_acc += t;
                    return delta(json{{"type", "thinking_delta"}, {"thinking", t}});
                };
                ev.content = [&](const std::string & t) {
                    if (t.empty()) return true;
                    if (open_type != "text" && !open_block("text", json{{"type", "text"}, {"text", ""}})) return false;
                    return delta(json{{"type", "text_delta"}, {"text", t}});
                };
                ev.tool_delta = [&](const json & d) {
                    if (d.contains("id"))   // the opening delta: the name is known, the arguments follow
                        return open_block("tool_use", json{{"type", "tool_use"}, {"id", tool_use_id(d["id"].get<std::string>())},
                                                           {"name", d["function"]["name"]}, {"input", json::object()}});
                    const std::string frag = d["function"].value("arguments", "");
                    return frag.empty() || delta(json{{"type", "input_json_delta"}, {"partial_json", frag}});
                };
                ev.tool_end  = [&]() { return close_block(); };
                ev.keepalive = [&]() { return send("ping", json{{"type", "ping"}}); };

                stream_outcome out = run_stream(*Q, *P, ev);
                if (!out.aborted.empty() || !out.ok) {
                    const bool client = out.aborted.empty() && too_long(out.err);
                    send("error", json{{"type", "error"}, {"error", {
                        {"type", client ? "invalid_request_error" : "api_error"},
                        {"message", !out.aborted.empty() ? "stream aborted: " + out.aborted : client ? too_long_msg(P->tok.size()) : out.err}}}});
                    sink.done();
                    lk->unlock();
                    return true;
                }
                close_block();
                send("message_delta", json{{"type", "message_delta"},
                    {"delta", {{"stop_reason", anthropic_stop_reason(out.C.R.finish, out.C.R.stop_seq)},
                               {"stop_sequence", out.C.R.stop_seq.empty() ? json(nullptr) : json(out.C.R.stop_seq)}}},
                    {"usage", anthropic_usage(out.C.R)}});
                send("message_stop", json{{"type", "message_stop"}});
                sink.done();
                lk->unlock();
                return true;
            });
    });

    // The exact count needs the tokenizer's view of the whole prompt (framing,
    // the tools block, the images encoded) and that is built under the engine
    // lock; while a generation holds it, an estimate goes back instead of a wait.
    svr.Post("/v1/messages/count_tokens", [&](const httplib::Request & req, httplib::Response & res) {
        json body; chat_request Q;
        if (!parse_anthropic(req, res, "count_", body, Q)) return;
        std::unique_lock<std::mutex> lk(S.mu, std::try_to_lock);
        if (!lk.owns_lock()) {
            res.set_content(json{{"input_tokens", (long long) (req.body.size() / 4)}}.dump(), "application/json");
            return;
        }
        server::prompt P; std::string e;
        if (!build_prompt(Q.msgs, Q.effort, Q.thinking, Q.tools_block, Q.forced, P, e)) { fail_anthropic(res, 400, e); return; }
        res.set_content(json{{"input_tokens", (long long) P.tok.size()}}.dump(), "application/json");
    });
    // Claude Code warms its connection with HEAD /api/hello (httplib answers a
    // HEAD from the GET handler).
    svr.Get("/api/hello", [&](const httplib::Request &, httplib::Response & res) {
        res.set_content(json{{"ok", true}}.dump(), "application/json");
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
            {"usage", {{"prompt_tokens", R.n_input}, {"prompt_tokens_details", {{"cached_tokens", R.n_cached}}},
                       {"completion_tokens", R.n_gen}, {"total_tokens", R.n_input + R.n_gen}}}
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
