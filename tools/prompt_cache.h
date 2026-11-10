// tierlm - a conversation's opening (the system message: instructions and tool descriptions), saved on the drive.
//
// A coding agent starts every session with the same 10 to 20 thousand tokens. Reading them takes most of a minute on
// a big model; what the model holds after reading them (its context memory, the hybrid models' running memories,
// the draft block's) is a few GB that an NVMe drive reads back in seconds. So the first time an opening is read,
// that state is written to tierlm-cache/ next to the settings file, under a name made from the model and the
// opening's tokens; a later conversation that starts with the same tokens, on the same model with the same context
// memory precision, loads it instead of reading the text again, also after tierlm or the model was started again.
//
// The file holds the tokens themselves and a line identifying the model, and both must match exactly, so a
// name clash or a stale file can't put the wrong state under a conversation. Oldest-used files go first when the
// folder passes its size limit.
#pragma once

#include "model.h"
#include "speculate.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <system_error>
#include <vector>

namespace pcache {

namespace fs = std::filesystem;
using tierlm::Model;
using tierlm::Speculator;

struct Entry {
    std::vector<int32_t> tokens;
    Model::Snapshot snap;
    bool has_spec = false;
    Speculator::State spec;
};

inline uint64_t fnv(const void* p, size_t n, uint64_t h) {
    const unsigned char* b = (const unsigned char*)p;
    for (size_t i = 0; i < n; ++i) { h ^= b[i]; h *= 1099511628211ULL; }
    return h;
}

// The file's name: two 64-bit fingerprints of the model's line and the tokens (what's inside is still compared).
inline std::string file_name(const std::string& identity, const int32_t* toks, size_t n) {
    uint64_t a = fnv(identity.data(), identity.size(), 1469598103934665603ULL);
    a = fnv(toks, n * sizeof(int32_t), a);
    uint64_t b = fnv(toks, n * sizeof(int32_t), 0x9E3779B97F4A7C15ULL);
    b = fnv(identity.data(), identity.size(), b);
    char buf[48];
    std::snprintf(buf, sizeof buf, "%016llx%016llx.tlmpc", (unsigned long long)a, (unsigned long long)b);
    return buf;
}

static const char kMagic[8] = {'T', 'L', 'M', 'P', 'C', '0', '0', '1'};
static const uint64_t kEnd = 0x444E455F434D4C54ULL;                 // written last: a file cut short doesn't have it

template <class T> inline void put(std::ofstream& f, const T& v) { f.write((const char*)&v, sizeof v); }
template <class T> inline bool get(std::ifstream& f, T& v) { f.read((char*)&v, sizeof v); return (bool)f; }
template <class T> inline void put_vec(std::ofstream& f, const std::vector<T>& v) {
    put(f, (uint64_t)v.size());
    if (!v.empty()) f.write((const char*)v.data(), (std::streamsize)(v.size() * sizeof(T)));
}
// (limit: the most elements a sane file can hold, so a damaged length can't ask for all the memory there is)
template <class T> inline bool get_vec(std::ifstream& f, std::vector<T>& v, uint64_t limit) {
    uint64_t n = 0;
    if (!get(f, n) || n > limit) return false;
    v.resize((size_t)n);
    if (n) f.read((char*)v.data(), (std::streamsize)(n * sizeof(T)));
    return (bool)f;
}

inline size_t bytes_of(const Entry& e) {
    return e.snap.bytes + e.tokens.size() * 8 + e.snap.rope.size() * 12 + e.spec.h_last.size() * 4 + 4096;
}

// Writes to <file>.part, then renames: a file with the final name is whole.
inline bool write(const fs::path& file, const std::string& identity, const Entry& e) {
    std::error_code ec;
    fs::create_directories(file.parent_path(), ec);
    const fs::path part = fs::path(file).concat(".part");
    {
        std::ofstream f(part, std::ios::binary | std::ios::trunc);
        if (!f) return false;
        f.write(kMagic, sizeof kMagic);
        put(f, (uint64_t)identity.size());
        f.write(identity.data(), (std::streamsize)identity.size());
        put_vec(f, e.tokens);
        put(f, (int32_t)e.snap.n_past);
        put(f, (uint64_t)e.snap.rope.size());
        for (const auto& r : e.snap.rope) { const int32_t v[3] = {r[0], r[1], r[2]}; f.write((const char*)v, sizeof v); }
        put_vec(f, e.snap.tok);
        put(f, (uint64_t)e.snap.qsa.size());
        for (const auto& q : e.snap.qsa) put_vec(f, q);
        put(f, (uint64_t)e.snap.parts.size());
        for (const auto& p : e.snap.parts) put_vec(f, p);
        put(f, (uint8_t)(e.has_spec ? 1 : 0));
        put_vec(f, e.spec.h_last);
        put(f, (int32_t)e.spec.h_pos);
        put(f, (int32_t)e.spec.mtp_valid);
        put(f, kEnd);
        f.flush();
        if (!f) { f.close(); fs::remove(part, ec); return false; }
    }
    fs::remove(file, ec);
    fs::rename(part, file, ec);
    if (ec) { fs::remove(part, ec); return false; }
    return true;
}

// False when the file is missing, damaged, for another model, or for other tokens.
inline bool read(const fs::path& file, const std::string& identity, const int32_t* toks, size_t n, Entry& e) {
    std::error_code ec;
    const uint64_t size = fs::file_size(file, ec);
    if (ec) return false;
    std::ifstream f(file, std::ios::binary);
    if (!f) return false;
    char magic[8];
    f.read(magic, sizeof magic);
    if (!f || std::memcmp(magic, kMagic, sizeof kMagic) != 0) return false;
    uint64_t len = 0;
    if (!get(f, len) || len > 4096) return false;
    std::string id((size_t)len, '\0');
    f.read(&id[0], (std::streamsize)len);
    if (!f || id != identity) return false;
    if (!get_vec(f, e.tokens, size / 4) || e.tokens.size() != n || !std::equal(e.tokens.begin(), e.tokens.end(), toks)) return false;
    int32_t n_past = 0;
    uint64_t cnt = 0;
    if (!get(f, n_past) || n_past != (int32_t)n) return false;
    e.snap.n_past = n_past;
    if (!get(f, cnt) || cnt > size / 12) return false;
    e.snap.rope.resize((size_t)cnt);
    for (auto& r : e.snap.rope) { int32_t v[3]; f.read((char*)v, sizeof v); r = {v[0], v[1], v[2]}; }
    if (!f || !get_vec(f, e.snap.tok, size / 4)) return false;
    if (!get(f, cnt) || cnt > 4096) return false;
    e.snap.qsa.resize((size_t)cnt);
    for (auto& q : e.snap.qsa) if (!get_vec(f, q, size / 4)) return false;
    if (!get(f, cnt) || cnt > 65536) return false;
    e.snap.parts.resize((size_t)cnt);
    e.snap.bytes = 0;
    for (auto& p : e.snap.parts) { if (!get_vec(f, p, size)) return false; e.snap.bytes += p.size(); }
    uint8_t hs = 0;
    int32_t h_pos = 0, mtp_valid = 0;
    if (!get(f, hs) || !get_vec(f, e.spec.h_last, size / 4) || !get(f, h_pos) || !get(f, mtp_valid)) return false;
    e.has_spec = hs != 0; e.spec.h_pos = h_pos; e.spec.mtp_valid = mtp_valid;
    uint64_t end = 0;
    return get(f, end) && end == kEnd;
}

// Marks a file as just used (the oldest-used go first).
inline void touch(const fs::path& file) {
    std::error_code ec;
    fs::last_write_time(file, fs::file_time_type::clock::now(), ec);
}

struct Usage { int files = 0; uint64_t bytes = 0; };

// The folder's files and size; leftovers of interrupted writes are removed. With a limit, the oldest-used files are
// removed until the folder, plus `incoming` bytes about to be written, fits.
inline Usage tidy(const fs::path& dir, uint64_t limit = 0, uint64_t incoming = 0) {
    Usage u;
    std::error_code ec;
    if (!fs::is_directory(dir, ec)) return u;
    struct F { fs::path p; uint64_t size; fs::file_time_type t; };
    std::vector<F> files;
    for (const auto& e : fs::directory_iterator(dir, ec)) {
        if (!e.is_regular_file(ec)) continue;
        const std::string ext = e.path().extension().u8string();
        if (ext == ".part") {                                     // (not one being written right now: older than a minute)
            const auto age = fs::file_time_type::clock::now() - e.last_write_time(ec);
            if (!ec && age > std::chrono::minutes(1)) fs::remove(e.path(), ec);
            continue;
        }
        if (ext != ".tlmpc") continue;
        files.push_back({e.path(), (uint64_t)e.file_size(ec), e.last_write_time(ec)});
    }
    std::sort(files.begin(), files.end(), [](const F& a, const F& b) { return a.t < b.t; });
    for (const F& f : files) { u.files++; u.bytes += f.size; }
    if (limit)
        for (const F& f : files) {
            if (u.bytes + incoming <= limit) break;
            fs::remove(f.p, ec);
            if (!ec) { u.files--; u.bytes -= f.size; }
        }
    return u;
}

}
