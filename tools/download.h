#pragma once
// Downloading a file the user asked for (the embedding model, from Hugging Face) and checking it.
//
// The transfer itself is done by curl, which Windows 10 and 11 include (C:\Windows\System32\curl.exe) and Linux has,
// so tierlm needs no TLS library of its own. Hugging Face says each file's SHA-256 in the first response
// (X-Linked-Etag) and its size (X-Linked-Size); the download goes to a .part file, is checked against both, and only
// then gets its real name.
#include <algorithm>
#include <atomic>
#include <cctype>
#include <cstdlib>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace download {

namespace fs = std::filesystem;

// ---- SHA-256 (FIPS 180-4) ----
class Sha256 {
public:
    Sha256() { reset(); }
    void reset() {
        static const uint32_t init[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
        std::memcpy(h_, init, sizeof h_);
        len_ = 0; n_ = 0;
    }
    void update(const void* data, size_t size) {
        const uint8_t* p = (const uint8_t*)data;
        len_ += size;
        while (size) {
            const size_t k = std::min(size, (size_t)64 - n_);
            std::memcpy(buf_ + n_, p, k);
            n_ += k; p += k; size -= k;
            if (n_ == 64) { block(buf_); n_ = 0; }
        }
    }
    std::string hex() {
        const uint64_t bits = len_ * 8;
        const uint8_t one = 0x80, zero = 0;
        update(&one, 1);
        while (n_ != 56) update(&zero, 1);
        uint8_t l[8];
        for (int i = 0; i < 8; ++i) l[i] = (uint8_t)(bits >> (56 - 8 * i));
        update(l, 8);
        char out[65];
        for (int i = 0; i < 8; ++i) std::snprintf(out + 8 * i, 9, "%08x", h_[i]);
        return std::string(out, 64);
    }
private:
    static uint32_t rotr(uint32_t x, int n) { return x >> n | x << (32 - n); }
    void block(const uint8_t* b) {
        static const uint32_t k[64] = {
            0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01,
            0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
            0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
            0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
            0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116, 0x1e376c08,
            0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
            0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};
        uint32_t w[64];
        for (int i = 0; i < 16; ++i) w[i] = (uint32_t)b[4 * i] << 24 | (uint32_t)b[4 * i + 1] << 16 | (uint32_t)b[4 * i + 2] << 8 | b[4 * i + 3];
        for (int i = 16; i < 64; ++i) {
            const uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
            const uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }
        uint32_t a = h_[0], bb = h_[1], c = h_[2], d = h_[3], e = h_[4], f = h_[5], g = h_[6], h = h_[7];
        for (int i = 0; i < 64; ++i) {
            const uint32_t t1 = h + (rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25)) + ((e & f) ^ (~e & g)) + k[i] + w[i];
            const uint32_t t2 = (rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22)) + ((a & bb) ^ (a & c) ^ (bb & c));
            h = g; g = f; f = e; e = d + t1; d = c; c = bb; bb = a; a = t1 + t2;
        }
        h_[0] += a; h_[1] += bb; h_[2] += c; h_[3] += d; h_[4] += e; h_[5] += f; h_[6] += g; h_[7] += h;
    }
    uint32_t h_[8];
    uint8_t buf_[64];
    uint64_t len_;
    size_t n_;
};

inline std::string sha256_file(const fs::path& p) {
    std::ifstream f(p, std::ios::binary);
    Sha256 s;
    std::vector<char> buf(1 << 20);
    while (f) {
        f.read(buf.data(), (std::streamsize)buf.size());
        if (f.gcount() > 0) s.update(buf.data(), (size_t)f.gcount());
    }
    return s.hex();
}

// ---- running curl ----
// Runs a program with these arguments (no shell, so paths need no quoting); returns its exit code, -1 if it
// couldn't start.
inline int run(const std::vector<std::string>& args) {
#ifdef _WIN32
    auto wide = [](const std::string& s) {
        std::wstring w(MultiByteToWideChar(CP_UTF8, 0, s.c_str(), (int)s.size(), nullptr, 0), L'\0');
        MultiByteToWideChar(CP_UTF8, 0, s.c_str(), (int)s.size(), w.data(), (int)w.size());
        return w;
    };
    std::wstring cmd;
    for (const std::string& a : args) {                       // Windows' quoting rules for one argument
        std::wstring w = wide(a), q = L"\"";
        size_t slashes = 0;
        for (wchar_t ch : w) {
            if (ch == L'\\') { ++slashes; continue; }
            if (ch == L'"') q.append(slashes * 2 + 1, L'\\'); else q.append(slashes, L'\\');
            slashes = 0;
            q += ch;
        }
        q.append(slashes * 2, L'\\');
        q += L"\"";
        cmd += (cmd.empty() ? L"" : L" ") + q;
    }
    STARTUPINFOW si{}; si.cb = sizeof si;
    PROCESS_INFORMATION pi{};
    if (!CreateProcessW(nullptr, cmd.data(), nullptr, nullptr, FALSE, CREATE_NO_WINDOW, nullptr, nullptr, &si, &pi)) return -1;
    WaitForSingleObject(pi.hProcess, INFINITE);
    DWORD code = 1;
    GetExitCodeProcess(pi.hProcess, &code);
    CloseHandle(pi.hProcess); CloseHandle(pi.hThread);
    return (int)code;
#else
    std::vector<char*> argv;
    for (const std::string& a : args) argv.push_back(const_cast<char*>(a.c_str()));
    argv.push_back(nullptr);
    const pid_t pid = fork();
    if (pid < 0) return -1;
    if (pid == 0) { execvp(argv[0], argv.data()); _exit(127); }
    int st = 0;
    if (waitpid(pid, &st, 0) < 0) return -1;
    return WIFEXITED(st) ? WEXITSTATUS(st) : -1;
#endif
}

inline std::string curl_exe() {
#ifdef _WIN32
    char sys[MAX_PATH] = {0};
    if (GetSystemDirectoryA(sys, MAX_PATH)) {
        const fs::path p = fs::path(sys) / "curl.exe";
        std::error_code ec;
        if (fs::exists(p, ec)) return p.string();
    }
#endif
    return "curl";
}

// A download in progress or finished, for the page to show.
struct Status {
    std::atomic<bool> running{false};
    std::atomic<long long> bytes{0}, total{0};
    std::string state = "idle", error, file;        // idle / downloading / checking / done / error (under the server's lock)
};

// The first response's headers (Hugging Face: before its redirect to the file's storage), names in lower case.
inline std::string header(const std::string& headers, const std::string& name) {
    size_t at = 0;
    while (at < headers.size()) {
        size_t end = headers.find('\n', at);
        if (end == std::string::npos) end = headers.size();
        std::string line = headers.substr(at, end - at);
        at = end + 1;
        const size_t colon = line.find(':');
        if (colon == std::string::npos) continue;
        std::string k = line.substr(0, colon);
        for (char& ch : k) ch = (char)std::tolower((unsigned char)ch);
        if (k != name) continue;
        std::string v = line.substr(colon + 1);
        while (!v.empty() && (v.back() == '\r' || v.back() == ' ' || v.back() == '"')) v.pop_back();
        size_t s = 0;
        while (s < v.size() && (v[s] == ' ' || v[s] == '"')) ++s;
        return v.substr(s);
    }
    return {};
}

}  // namespace download
