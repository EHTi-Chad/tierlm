#include "tokenizer.h"
#include "unicode_tables.h"
#include "gguf.h"
#include <algorithm>
#include <climits>
#include <stdexcept>
#include <utility>

namespace tierlm {
namespace {

template <size_t N>
bool in(const uni::Range (&t)[N], uint32_t cp) {
    size_t lo = 0, hi = N;
    while (lo < hi) {
        size_t mid = (lo + hi) / 2;
        if (cp < t[mid].lo) hi = mid;
        else if (cp > t[mid].hi) lo = mid + 1;
        else return true;
    }
    return false;
}
bool is_letter(uint32_t c) { return in(uni::kLetter, c); }
bool is_number(uint32_t c) { return in(uni::kNumber, c); }
bool is_space(uint32_t c)  { return in(uni::kSpace, c); }
bool is_other(uint32_t c)  { return !is_letter(c) && !is_number(c) && !is_space(c); }

struct Cp { uint32_t cp; size_t off, len; };   // code point + its byte span in the input

std::vector<Cp> utf8_decode(const std::string& s) {
    std::vector<Cp> out;
    size_t i = 0;
    while (i < s.size()) {
        unsigned char c = s[i];
        uint32_t cp; size_t n;
        if (c < 0x80)                                   { cp = c;        n = 1; }
        else if ((c >> 5) == 0x6  && i + 1 < s.size())  { cp = c & 0x1F; n = 2; }
        else if ((c >> 4) == 0xE  && i + 2 < s.size())  { cp = c & 0x0F; n = 3; }
        else if ((c >> 3) == 0x1E && i + 3 < s.size())  { cp = c & 0x07; n = 4; }
        else                                            { cp = 0xFFFD;   n = 1; }  // invalid byte: keep it as-is
        for (size_t k = 1; k < n; ++k) cp = (cp << 6) | (s[i + k] & 0x3F);
        out.push_back({cp, i, n});
        i += n;
    }
    return out;
}

std::string utf8_encode(uint32_t cp) {
    std::string s;
    if (cp < 0x80)         { s += char(cp); }
    else if (cp < 0x800)   { s += char(0xC0 | (cp >> 6));  s += char(0x80 | (cp & 0x3F)); }
    else if (cp < 0x10000) { s += char(0xE0 | (cp >> 12)); s += char(0x80 | ((cp >> 6) & 0x3F)); s += char(0x80 | (cp & 0x3F)); }
    else                   { s += char(0xF0 | (cp >> 18)); s += char(0x80 | ((cp >> 12) & 0x3F));
                             s += char(0x80 | ((cp >> 6) & 0x3F)); s += char(0x80 | (cp & 0x3F)); }
    return s;
}

// GPT-2 split pattern, hand-compiled:
//   's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
std::vector<std::pair<size_t, size_t>> pretokenize(const std::string& s) {
    const auto cps = utf8_decode(s);
    const size_t n = cps.size();
    std::vector<std::pair<size_t, size_t>> out;
    auto C = [&](size_t k) { return cps[k].cp; };
    auto emit = [&](size_t a, size_t b) {           // code points [a, b) -> byte span
        size_t off = cps[a].off, end = cps[b - 1].off + cps[b - 1].len;
        out.push_back({off, end - off});
    };

    size_t i = 0;
    while (i < n) {
        if (C(i) == '\'' && i + 1 < n) {             // contractions
            uint32_t a = C(i + 1), b = i + 2 < n ? C(i + 2) : 0;
            if (a == 's' || a == 't' || a == 'm' || a == 'd')                                { emit(i, i + 2); i += 2; continue; }
            if ((a == 'r' && b == 'e') || (a == 'v' && b == 'e') || (a == 'l' && b == 'l'))  { emit(i, i + 3); i += 3; continue; }
        }
        size_t j = (C(i) == ' ' && i + 1 < n) ? i + 1 : i;   // optional single leading space
        uint32_t h = C(j);
        bool (*cls)(uint32_t) = is_letter(h) ? is_letter : is_number(h) ? is_number : is_other(h) ? is_other : nullptr;
        if (cls) {
            size_t k = j + 1;
            while (k < n && cls(C(k))) ++k;
            emit(i, k); i = k; continue;
        }
        size_t k = i;                                // whitespace run
        while (k < n && is_space(C(k))) ++k;
        if (k == n || k - i == 1) { emit(i, k); i = k; }           // \s+ to end, or a lone space char
        else                      { emit(i, k - 1); i = k - 1; }   // leave the last one to prefix the next word
    }
    return out;
}

}  // namespace

Tokenizer Tokenizer::from_gguf(const std::string& path) {
    gguf_init_params p{ /*no_alloc=*/true, /*ctx=*/nullptr };
    gguf_context* g = gguf_init_from_file(path.c_str(), p);
    if (!g) throw std::runtime_error("failed to read GGUF: " + path);

    int64_t mk = gguf_find_key(g, "tokenizer.ggml.model");
    std::string model = mk < 0 ? "" : gguf_get_val_str(g, mk);
    if (model != "gpt2") { gguf_free(g); throw std::runtime_error("unsupported tokenizer type: " + model); }

    Tokenizer t;
    int64_t tk = gguf_find_key(g, "tokenizer.ggml.tokens");
    if (tk < 0) { gguf_free(g); throw std::runtime_error("GGUF has no tokenizer vocabulary"); }
    const size_t nt = gguf_get_arr_n(g, tk);
    t.id_to_token_.resize(nt);
    for (size_t i = 0; i < nt; ++i) {
        t.id_to_token_[i] = gguf_get_arr_str(g, tk, i);
        t.token_to_id_.emplace(t.id_to_token_[i], (int32_t)i);
    }
    int64_t yk = gguf_find_key(g, "tokenizer.ggml.token_type");
    t.token_type_.assign(nt, 1);
    if (yk >= 0 && gguf_get_arr_type(g, yk) == GGUF_TYPE_INT32 && gguf_get_arr_n(g, yk) == nt) {
        const int32_t* ty = (const int32_t*)gguf_get_arr_data(g, yk);
        t.token_type_.assign(ty, ty + nt);
    }
    for (size_t i = 0; i < nt; ++i)
        if ((t.token_type_[i] == 3 || t.token_type_[i] == 4) && !t.id_to_token_[i].empty())
            t.added_.push_back({t.id_to_token_[i], (int32_t)i});
    std::stable_sort(t.added_.begin(), t.added_.end(),
                     [](const auto& a, const auto& b) { return a.first.size() > b.first.size(); });

    int64_t rk = gguf_find_key(g, "tokenizer.ggml.merges");          // optional: a byte-only vocab has none
    const size_t nm = rk < 0 ? 0 : gguf_get_arr_n(g, rk);
    for (size_t i = 0; i < nm; ++i) t.merge_rank_.emplace(gguf_get_arr_str(g, rk, i), (int32_t)i);

    auto u32 = [&](const char* k) -> int32_t {
        int64_t id = gguf_find_key(g, k);
        return id < 0 ? -1 : (int32_t)gguf_get_val_u32(g, id);
    };
    t.bos_ = u32("tokenizer.ggml.bos_token_id");
    t.eos_ = u32("tokenizer.ggml.eos_token_id");
    gguf_free(g);

    int extra = 0;                                   // GPT-2 bytes_to_unicode()
    for (int b = 0; b < 256; ++b) {
        bool direct = (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255);
        t.byte_to_uni_[b] = utf8_encode(direct ? (uint32_t)b : 256u + extra++);
        t.uni_to_byte_[t.byte_to_uni_[b]] = (uint8_t)b;
    }
    return t;
}

void Tokenizer::bpe(const std::string& chunk, std::vector<int32_t>& out) const {
    std::vector<std::string> sym;
    for (unsigned char c : chunk) sym.push_back(byte_to_uni_[c]);

    while (sym.size() > 1) {
        int best = INT_MAX; size_t at = 0;
        for (size_t i = 0; i + 1 < sym.size(); ++i) {
            auto it = merge_rank_.find(sym[i] + " " + sym[i + 1]);
            if (it != merge_rank_.end() && it->second < best) { best = it->second; at = i; }
        }
        if (best == INT_MAX) break;
        const std::string a = sym[at], b = sym[at + 1];   // merge every occurrence of the winning pair
        std::vector<std::string> next;
        next.reserve(sym.size());
        for (size_t i = 0; i < sym.size();) {
            if (i + 1 < sym.size() && sym[i] == a && sym[i + 1] == b) { next.push_back(a + b); i += 2; }
            else { next.push_back(sym[i]); i += 1; }
        }
        sym.swap(next);
    }
    for (const auto& s : sym) {
        auto it = token_to_id_.find(s);
        if (it == token_to_id_.end()) throw std::runtime_error("BPE produced a token not in the vocab");
        out.push_back(it->second);
    }
}

void Tokenizer::encode_plain(const std::string& text, std::vector<int32_t>& out) const {
    for (const auto& [off, len] : pretokenize(text)) bpe(text.substr(off, len), out);
}

// Added tokens are split out first (leftmost-longest), like Hugging Face does;
// only the text between them goes through the pre-tokenizer and BPE.
std::vector<int32_t> Tokenizer::encode(const std::string& text, bool parse_special) const {
    std::vector<int32_t> ids;
    size_t start = 0, i = 0;
    while (i < text.size()) {
        int32_t hit = -1; size_t hit_len = 0;
        for (const auto& [s, id] : added_) {
            if (!parse_special && token_type_[id] == 3) continue;
            if (text.compare(i, s.size(), s) == 0) { hit = id; hit_len = s.size(); break; }
        }
        if (hit < 0) { ++i; continue; }
        if (i > start) encode_plain(text.substr(start, i - start), ids);
        ids.push_back(hit);
        i += hit_len;
        start = i;
    }
    if (start < text.size()) encode_plain(text.substr(start), ids);
    return ids;
}

std::string Tokenizer::decode(const std::vector<int32_t>& ids) const {
    std::string out;
    for (int32_t id : ids) {
        if (id < 0 || (size_t)id >= id_to_token_.size()) continue;
        const std::string& tok = id_to_token_[id];
        if (token_type_[id] == 3 || token_type_[id] == 4) { out += tok; continue; }   // added tokens are raw text
        for (const auto& c : utf8_decode(tok)) {
            std::string ch = tok.substr(c.off, c.len);
            auto it = uni_to_byte_.find(ch);
            out += it != uni_to_byte_.end() ? std::string(1, (char)it->second) : ch;
        }
    }
    return out;
}

}  // namespace tierlm
