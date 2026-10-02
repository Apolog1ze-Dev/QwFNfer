#include "qwfn_files.h"

// stbi_zlib_decode_noheader_buffer: raw deflate, what a zip entry holds. The implementation is
// compiled once, in qwfn_vision.cpp, for the image decoder.
#include "stb_image.h"

#include <algorithm>
#include <climits>
#include <cstring>
#include <vector>

namespace qwfn {

static uint16_t rd16(const uint8_t * p) { return (uint16_t) (p[0] | (p[1] << 8)); }
static uint32_t rd32(const uint8_t * p) {
    return (uint32_t) p[0] | ((uint32_t) p[1] << 8) | ((uint32_t) p[2] << 16) | ((uint32_t) p[3] << 24);
}

bool looks_like_text(const uint8_t * d, size_t n) {
    size_t i = (n >= 3 && d[0] == 0xEF && d[1] == 0xBB && d[2] == 0xBF) ? 3 : 0;
    while (i < n) {
        const uint8_t c = d[i];
        if (c == 0) return false;
        if (c < 0x80) { i++; continue; }
        const size_t len = (c >= 0xC2 && c <= 0xDF) ? 2 : (c >= 0xE0 && c <= 0xEF) ? 3 : (c >= 0xF0 && c <= 0xF4) ? 4 : 0;
        if (!len || i + len > n) return false;
        for (size_t k = 1; k < len; k++) if ((d[i + k] & 0xC0) != 0x80) return false;
        // overlong forms, UTF-16 surrogates, past U+10FFFF
        if ((c == 0xE0 && d[i + 1] < 0xA0) || (c == 0xED && d[i + 1] >= 0xA0) ||
            (c == 0xF0 && d[i + 1] < 0x90) || (c == 0xF4 && d[i + 1] >= 0x90)) return false;
        i += len;
    }
    return true;
}

// The text of one file, at most `cap` bytes of it, cut at a character boundary.
static void append_file(std::string & out, const std::string & name, const uint8_t * d, size_t n, size_t cap) {
    if (n >= 3 && d[0] == 0xEF && d[1] == 0xBB && d[2] == 0xBF) { d += 3; n -= 3; }
    size_t take = std::min(n, cap);
    while (take < n && take > 0 && (d[take] & 0xC0) == 0x80) take--;   // not inside a character
    out += "--- file: " + name + " ---\n";
    out.append((const char *) d, take);
    if (take && d[take - 1] != '\n') out += '\n';
    if (take < n) out += "[... cut: the first " + std::to_string(take) + " of " + std::to_string(n) + " bytes]\n";
    out += "--- end of " + name + " ---\n";
}

static std::string size_str(size_t b) {
    return b >= (1u << 20) ? std::to_string(b >> 20) + " MB" : std::to_string(b) + " bytes";
}

static bool is_zip(const uint8_t * d, size_t n) {
    return n >= 4 && d[0] == 'P' && d[1] == 'K' && ((d[2] == 3 && d[3] == 4) || (d[2] == 5 && d[3] == 6));
}

struct zip_entry {
    std::string name;
    uint16_t flags = 0, method = 0;
    uint32_t csize = 0, usize = 0, local = 0;
};

// The central directory: the end record (22 bytes and a comment of up to 64 KiB) at the end of
// the file points at it.
static bool zip_directory(const uint8_t * d, size_t n, std::vector<zip_entry> & out, std::string & why) {
    if (n < 22) { why = "too short to be a zip"; return false; }
    size_t eocd = SIZE_MAX;
    const size_t lo = n > 22 + 65535 ? n - 22 - 65535 : 0;
    for (size_t p = n - 22 + 1; p-- > lo;) if (rd32(d + p) == 0x06054b50) { eocd = p; break; }
    if (eocd == SIZE_MAX) { why = "no zip directory in it"; return false; }
    const uint32_t count = rd16(d + eocd + 10), cd_size = rd32(d + eocd + 12), cd_off = rd32(d + eocd + 16);
    if (count == 0xFFFF || cd_size == 0xFFFFFFFFu || cd_off == 0xFFFFFFFFu) { why = "a ZIP64 archive, which this server does not read"; return false; }
    if ((uint64_t) cd_off + cd_size > eocd) { why = "a damaged zip directory"; return false; }
    size_t p = cd_off;
    for (uint32_t i = 0; i < count; i++) {
        if (p + 46 > eocd || rd32(d + p) != 0x02014b50) { why = "a damaged zip directory"; return false; }
        zip_entry e;
        e.flags  = rd16(d + p + 8);
        e.method = rd16(d + p + 10);
        e.csize  = rd32(d + p + 20);
        e.usize  = rd32(d + p + 24);
        e.local  = rd32(d + p + 42);
        const size_t nl = rd16(d + p + 28), xl = rd16(d + p + 30), cl = rd16(d + p + 32);
        if (p + 46 + nl > eocd) { why = "a damaged zip directory"; return false; }
        e.name.assign((const char *) d + p + 46, nl);
        out.push_back(std::move(e));
        p += 46 + nl + xl + cl;
    }
    return true;
}

static std::string zip_as_text(const std::string & name, const uint8_t * d, size_t n, const file_text_limits & lim) {
    std::vector<zip_entry> entries;
    std::string why;
    if (!zip_directory(d, n, entries, why)) return "[attached file not read: " + name + " (" + why + ")]\n";

    std::string body, skipped;
    size_t total = 0, n_files = 0, n_read = 0;
    std::vector<uint8_t> buf;
    for (const zip_entry & e : entries) {
        if (e.name.empty() || e.name.back() == '/') continue;                      // a directory
        const size_t slash = e.name.rfind('/');
        const std::string base = slash == std::string::npos ? e.name : e.name.substr(slash + 1);
        if (e.name.rfind("__MACOSX/", 0) == 0 || base.rfind("._", 0) == 0) continue;   // macOS resource forks
        n_files++;
        const std::string path = name + "/" + e.name;
        auto skip = [&](const std::string & reason) { skipped += "[not read: " + path + " (" + reason + ")]\n"; };
        if (e.flags & 1)                                        { skip("encrypted"); continue; }
        if (e.method != 0 && e.method != 8)                     { skip("compressed with zip method " + std::to_string(e.method)); continue; }
        if (e.csize == 0xFFFFFFFFu || e.usize == 0xFFFFFFFFu)   { skip("a ZIP64 entry"); continue; }
        if (base.size() > 4 && (base.compare(base.size() - 4, 4, ".zip") == 0 || base.compare(base.size() - 4, 4, ".ZIP") == 0)) { skip("a zip inside the zip"); continue; }
        // The declared size is the whole buffer the inflate may fill: a stream that would write
        // more fails, whatever the directory claimed, so an archive cannot expand past it.
        const size_t room = std::min(lim.max_total - std::min(total, lim.max_total), (size_t) INT_MAX - 1);
        if (e.usize > room)                                     { skip(size_str(e.usize) + ", past the " + size_str(lim.max_total) + " of text read from one attachment"); continue; }
        if ((uint64_t) e.local + 30 > n || rd32(d + e.local) != 0x04034b50) { skip("damaged"); continue; }
        const uint64_t data = (uint64_t) e.local + 30 + rd16(d + e.local + 26) + rd16(d + e.local + 28);
        if (data + e.csize > n)                                 { skip("damaged"); continue; }
        const uint8_t * src = d + data;
        size_t got = 0;
        if (e.method == 0) {
            if (e.csize != e.usize)                             { skip("damaged"); continue; }
            buf.assign(src, src + e.usize);
            got = buf.size();
        } else {
            buf.resize(std::max<size_t>(e.usize, 1));
            const int r = e.csize ? stbi_zlib_decode_noheader_buffer((char *) buf.data(), (int) buf.size(), (const char *) src, (int) std::min<size_t>(e.csize, INT_MAX)) : 0;
            if (r < 0 || (uint32_t) r != e.usize)               { skip("damaged"); continue; }
            got = (size_t) r;
        }
        if (!looks_like_text(buf.data(), got))                  { skip("binary"); continue; }
        const size_t before = body.size();
        append_file(body, path, buf.data(), got, std::min(lim.max_file, room));
        total += body.size() - before;
        n_read++;
    }
    std::string out = "--- archive: " + name + " (" + std::to_string(n_files) + (n_files == 1 ? " file, " : " files, ") +
                      std::to_string(n_read) + " read as text) ---\n";
    out += body;
    out += skipped;
    out += "--- end of archive " + name + " ---\n";
    return out;
}

std::string file_as_text(const std::string & name_in, const std::string & media_type,
                         const uint8_t * data, size_t n, const file_text_limits & lim) {
    const std::string name = name_in.empty() ? "attachment" : name_in;
    if (is_zip(data, n)) return zip_as_text(name, data, n, lim);
    if (looks_like_text(data, n)) { std::string out; append_file(out, name, data, n, std::min(lim.max_file, lim.max_total)); return out; }
    return "[attached file not read: " + name + (media_type.empty() ? "" : " (" + media_type + ")") +
           ": this server reads text files and zip archives of them]\n";
}

} // namespace qwfn
